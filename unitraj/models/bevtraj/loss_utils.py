import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.bevtraj.utility import batch_nms


class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config

        self.goal_FDE_loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)

    def forward(self, out, gt, center_gt_final_valid_idx, traj_data):
        modes_preds = out['predicted_probability'] # [B, K]
        preds = out['predicted_trajectory'] # [K, T, B, 5]
        pred_vels = out['predicted_velocity'] # [K, T, B, 2]

        anchor_pos = out['anchor_pos']

        dense_future_pred = out['dense_future_pred']

        state_pred = out['state_pred']

        gt_decoder = gt[0]
        gt_dense_future_trajs = gt[1]
        
        anchor_pos_set = (anchor_pos, out.get('anchor_pos_topk'))
        # anchor_pos_set = out['goal_reg_list'][-1].permute(1, 0, 2)
        decoder_loss = self.get_decoder_loss_hard_assign(
            modes_preds=modes_preds,
            preds=preds,
            pred_vels=pred_vels,
            anchor_pos_set=anchor_pos_set,
            gt_decoder=gt_decoder,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
        )

        state_query_loss = self.get_state_query_loss(
            state_pred=state_pred,
            gt=gt_decoder,
        )

        dense_future_loss = self.get_dense_future_prediction_loss(dense_future_pred, gt_dense_future_trajs)

        total_loss = decoder_loss + state_query_loss + dense_future_loss
        return total_loss

    def get_decoder_loss_hard_assign( # EDA
        self,
        modes_preds,                 # list of [B, K]
        preds,                       # list of [K, T, B, 5]
        pred_vels,                   # list of [K, T, B, 2]
        anchor_pos_set,              # [anchor_pos_full[B,64,2], anchor_pos_topk[B,K,2]]
        gt_decoder,                  # [B, T, 5] -> (x, y, vx, vy, valid)
        center_gt_final_valid_idx,   # [B]
    ):
        if isinstance(anchor_pos_set, (tuple, list)):
            anchor_pos_full = anchor_pos_set[0]
            anchor_pos_topk = anchor_pos_set[1]
        else:
            anchor_pos_full = anchor_pos_set
            anchor_pos_topk = None

        device = gt_decoder.device
        B = gt_decoder.size(0)
        b_idx = torch.arange(B, device=device)

        gt_xy = gt_decoder[..., :2]                     # [B, T, 2]
        gt_vel = gt_decoder[..., 2:4]                   # [B, T, 2]
        gt_mask = gt_decoder[..., 4].float()            # [B, T]
        final_idx = center_gt_final_valid_idx.long()    # [B]
        gt_goal = gt_xy[b_idx, final_idx]               # [B, 2]
        valid_final = gt_mask[b_idx, final_idx]         # [B]

        # initial positive idx from SGCP anchors (full set)
        with torch.no_grad():
            dist = (anchor_pos_full.detach() - gt_goal[:, None, :]).norm(dim=-1)  # [B, 64]
            hard_idx = dist.argmin(dim=-1)                                             # [B]

        w_cls = self.config.get('cls_weight', 2.0)
        w_reg = self.config.get('reg_weight', 1.0)
        w_vel = self.config.get('vel_weight', 0.2)
        w_cls_final = self.config.get('cls_weight_final', 2.0)
        w_reg_final = self.config.get('reg_weight_final', 1.0)
        w_vel_final = self.config.get('vel_weight_final', w_vel)

        # EDA-related config
        num_inter_layers = int(self.config.get('num_inter_layers', 2))
        use_distinct_anchors = self.config.get('distinct_anchors', False)
        distinct_nms_thresh = float(self.config.get('distinct_nms_thresh', -1.0))

        total = 0.0
        num_layers = len(preds)

        for layer_idx, (pred_scores, pred, pred_vel) in enumerate(zip(modes_preds, preds, pred_vels)):
            is_last_layer = (layer_idx == num_layers - 1)
            cur_w_cls = w_cls_final if is_last_layer else w_cls
            cur_w_reg = w_reg_final if is_last_layer else w_reg
            cur_w_vel = w_vel_final if is_last_layer else w_vel

            # pred: [K, T, B, 5] -> [B, K, T, 5]
            pred_trajs = pred.permute(2, 0, 1, 3).contiguous()
            pred_vel = pred_vel.permute(2, 0, 1, 3).contiguous()   # [B, K, T, 2]

            # ---------- Evolving Anchors ----------
            positive_layer_idx = (layer_idx // num_inter_layers) * num_inter_layers - 1
            if positive_layer_idx < 0:
                # first-stage anchor from SGCP anchors
                if pred_scores.size(1) == anchor_pos_full.size(1):
                    cur_anchor_pos = anchor_pos_full
                elif anchor_pos_topk is not None and pred_scores.size(1) == anchor_pos_topk.size(1):
                    cur_anchor_pos = anchor_pos_topk
                else:
                    cur_anchor_pos = anchor_pos_full[:, :pred_scores.size(1)]

                anchor_trajs = cur_anchor_pos.detach().unsqueeze(2)
                dist = (cur_anchor_pos.detach() - gt_goal[:, None, :]).norm(dim=-1)
            else:
                # use previous anchor trajectories
                anchor_trajs = preds[positive_layer_idx].permute(2, 0, 1, 3).detach()  # [B, K, T, 5]
                dist = ((gt_xy[:, None, :, :] - anchor_trajs[..., 0:2]).norm(dim=-1) * gt_mask[:, None, :]).sum(dim=-1)

            # ---------- Distinct Anchors ----------
            select_mask = torch.ones_like(pred_scores, dtype=torch.bool)
            if use_distinct_anchors:
                if distinct_nms_thresh < 0:
                    top_idx = pred_scores.argmax(dim=-1)  # [B]
                    top_traj = pred_trajs[b_idx, top_idx, :, :2]  # [B, T, 2]
                    top_traj_length = torch.norm(torch.diff(top_traj, dim=1), dim=-1).sum(dim=-1)

                    upper_dist, lower_dist = 3.5, 2.5
                    upper_length, lower_length = 50.0, 10.0
                    scalar = 1.5

                    dist_thresh = lower_dist + scalar * (top_traj_length - lower_length) / (upper_length - lower_length)
                    dist_thresh = torch.maximum(dist_thresh, torch.full_like(dist_thresh, lower_dist))
                    dist_thresh = torch.minimum(dist_thresh, torch.full_like(dist_thresh, upper_dist))
                else:
                    dist_thresh = distinct_nms_thresh

                select_mask = batch_nms(
                    pred_trajs=anchor_trajs,
                    pred_scores=pred_scores.sigmoid(),
                    dist_thresh=dist_thresh,
                    num_ret_modes=anchor_trajs.shape[1],
                    return_mask=True,
                )

            # Evolving + Distinct
            dist = dist.masked_fill(~select_mask, 1e10)
            hard_idx = dist.argmin(dim=-1)  # [B]

            # MTR nll_loss_gmm_direct expects log_std, but MotionRegHead outputs sigma
            mu = pred_trajs[..., :2]
            log_std = torch.log(pred_trajs[..., 2:4].clamp_min(1e-6))
            rho = pred_trajs[..., 4:5]
            pred_trajs_gmm = torch.cat([mu, log_std, rho], dim=-1)

            loss_reg_gmm, hard_idx = self.nll_loss_gmm_direct(
                pred_scores=pred_scores,                 # [B, K]
                pred_trajs=pred_trajs_gmm,               # [B, K, T, 5]
                gt_trajs=gt_xy,                          # [B, T, 2]
                gt_valid_mask=gt_mask,                   # [B, T]
                pre_nearest_mode_idxs=hard_idx,
                timestamp_loss_weight=None,
                use_square_gmm=False,
            )

            pred_vel_pos = pred_vel[b_idx, hard_idx]  # [B, T, 2]
            loss_reg_vel = F.l1_loss(pred_vel_pos, gt_vel, reduction='none')
            loss_reg_vel = (loss_reg_vel * gt_mask[:, :, None]).sum(dim=-1).sum(dim=-1)

            # ---------- BCE classification ----------
            bce_target = torch.zeros_like(pred_scores)   # [B, K]
            bce_target[b_idx, hard_idx] = 1.0
            loss_cls = F.binary_cross_entropy_with_logits(pred_scores, bce_target, reduction='none')  # [B, K]
            loss_cls = (loss_cls * select_mask.float()).sum(dim=-1)                                    # [B]

            layer_loss = cur_w_reg * loss_reg_gmm + cur_w_vel * loss_reg_vel + cur_w_cls * loss_cls
            layer_loss = (layer_loss * valid_final).sum() / valid_final.sum().clamp_min(1.0)
            total = total + layer_loss

        return total / num_layers
    
    def get_goal_prediction_loss(self, goal_reg_list, gt, center_gt_final_valid_idx):
        """
        goal_reg: each [K, B, 2]
        goal_FDE: each [B, K]
        gt: [B, T, 3]  # (x, y, valid)
        center_gt_final_valid_idx: [B]
        """
        num_layers = len(goal_reg_list)

        device = gt.device
        B = gt.size(0)
        b_idx = torch.arange(B, device=device)
        final_idx = center_gt_final_valid_idx.long()

        gt_goal = gt[b_idx, final_idx, :2]            # [B, 2]
        valid_final = gt[b_idx, final_idx, -1].float() # [B]

        total_loss = 0.0
        for goal_reg in goal_reg_list:
            # ---------------- reg loss ----------------
            # goal_reg: [K, B, 2] -> [B, K, 2]
            goal_reg_bk2 = goal_reg.permute(1, 0, 2)
            dist = torch.norm(goal_reg_bk2 - gt_goal.unsqueeze(1), p=2, dim=-1)  # [B, K]
            reg_loss_per_sample = dist.min(dim=1)[0]                               # [B]
            reg_loss = (reg_loss_per_sample * valid_final).sum() / valid_final.sum().clamp_min(1.0)

            total_loss = total_loss + self.config['goal_reg_weight'] * reg_loss

        return total_loss / num_layers
    
    def get_state_query_loss(self, state_pred, gt):
        """
        state_pred: [B, T, 2]
        gt:         [B, T, 5]  (x, y, vx, vy, valid)
        """
        gt_xy = gt[..., :2]
        gt_mask = gt[..., 4].float()  # [B,T]

        loss_xy = F.smooth_l1_loss(state_pred, gt_xy, reduction='none').sum(dim=-1)  # [B,T]
        loss_xy = (loss_xy * gt_mask).sum() / gt_mask.sum().clamp_min(1.0)

        w = float(self.config.get('state_query_weight', 1.0))
        return w * loss_xy

    def get_dense_future_prediction_loss(self, prediction, gt):
        obj_trajs_future_state = gt['obj_trajs_future_state']
        obj_trajs_future_mask = gt['obj_trajs_future_mask']
        pred_dense_trajs = prediction # (num_center_objects, num_objects, num_future_frames, 7)
        assert pred_dense_trajs.shape[-1] == 7
        assert obj_trajs_future_state.shape[-1] == 4

        pred_dense_trajs_gmm, pred_dense_trajs_vel = pred_dense_trajs[:, :, :, 0:5], pred_dense_trajs[:, :, :, 5:7]

        loss_reg_vel = F.l1_loss(pred_dense_trajs_vel, obj_trajs_future_state[:, :, :, 2:4], reduction='none')
        loss_reg_vel = (loss_reg_vel * obj_trajs_future_mask[:, :, :, None]).sum(dim=-1).sum(dim=-1)

        num_center_objects, num_objects, num_timestamps, _ = pred_dense_trajs.shape
        fake_scores = pred_dense_trajs.new_zeros((num_center_objects, num_objects)).view(-1,
                                                                                         1)  # (num_center_objects * num_objects, 1)

        temp_pred_trajs = pred_dense_trajs_gmm.contiguous().view(num_center_objects * num_objects, 1, num_timestamps, 5)
        temp_gt_idx = torch.zeros(num_center_objects * num_objects).long()  # (num_center_objects * num_objects)
        temp_gt_trajs = obj_trajs_future_state[:, :, :, 0:2].contiguous().view(num_center_objects * num_objects,
                                                                               num_timestamps, 2)
        temp_gt_trajs_mask = obj_trajs_future_mask.view(num_center_objects * num_objects, num_timestamps)
        loss_reg_gmm, _ = self.nll_loss_gmm_direct(
            pred_scores=fake_scores, pred_trajs=temp_pred_trajs, gt_trajs=temp_gt_trajs,
            gt_valid_mask=temp_gt_trajs_mask,
            pre_nearest_mode_idxs=temp_gt_idx,
            timestamp_loss_weight=None, use_square_gmm=False,
        )
        loss_reg_gmm = loss_reg_gmm.view(num_center_objects, num_objects)

        loss_reg = loss_reg_vel + loss_reg_gmm

        obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0

        loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / torch.clamp_min(obj_valid_mask.sum(dim=-1),
                                                                                     min=1.0)
        loss_reg = loss_reg.mean()

        return loss_reg
    
    def nll_loss_gmm_direct(self, pred_scores, pred_trajs, gt_trajs, gt_valid_mask, pre_nearest_mode_idxs=None,
                            timestamp_loss_weight=None, use_square_gmm=False, log_std_range=(-1.609, 5.0), rho_limit=0.5):
        """
        GMM Loss for Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
        Written by Shaoshuai Shi 

        Args:
            pred_scores (batch_size, num_modes):
            pred_trajs (batch_size, num_modes, num_timestamps, 5 or 3)
            gt_trajs (batch_size, num_timestamps, 2):
            gt_valid_mask (batch_size, num_timestamps):
            timestamp_loss_weight (num_timestamps):
        """
        if use_square_gmm:
            assert pred_trajs.shape[-1] == 3
        else:
            assert pred_trajs.shape[-1] == 5

        batch_size = pred_scores.shape[0]

        if pre_nearest_mode_idxs is not None:
            nearest_mode_idxs = pre_nearest_mode_idxs
        else:
            distance = (pred_trajs[:, :, :, 0:2] - gt_trajs[:, None, :, :]).norm(dim=-1)
            distance = (distance * gt_valid_mask[:, None, :]).sum(dim=-1)

            nearest_mode_idxs = distance.argmin(dim=-1)
        nearest_mode_bs_idxs = torch.arange(batch_size).type_as(nearest_mode_idxs)  # (batch_size, 2)

        nearest_trajs = pred_trajs[nearest_mode_bs_idxs, nearest_mode_idxs]  # (batch_size, num_timestamps, 5)
        res_trajs = gt_trajs - nearest_trajs[:, :, 0:2]  # (batch_size, num_timestamps, 2)
        dx = res_trajs[:, :, 0]
        dy = res_trajs[:, :, 1]

        if use_square_gmm:
            log_std1 = log_std2 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            std1 = std2 = torch.exp(log_std1)  # (0.2m to 150m)
            rho = torch.zeros_like(log_std1)
        else:
            log_std1 = torch.clip(nearest_trajs[:, :, 2], min=log_std_range[0], max=log_std_range[1])
            log_std2 = torch.clip(nearest_trajs[:, :, 3], min=log_std_range[0], max=log_std_range[1])
            std1 = torch.exp(log_std1)  # (0.2m to 150m)
            std2 = torch.exp(log_std2)  # (0.2m to 150m)
            rho = torch.clip(nearest_trajs[:, :, 4], min=-rho_limit, max=rho_limit)

        gt_valid_mask = gt_valid_mask.type_as(pred_scores)
        if timestamp_loss_weight is not None:
            gt_valid_mask = gt_valid_mask * timestamp_loss_weight[None, :]

        # -log(a^-1 * e^b) = log(a) - b
        reg_gmm_log_coefficient = log_std1 + log_std2 + 0.5 * torch.log(1 - rho ** 2)  # (batch_size, num_timestamps)
        reg_gmm_exp = (0.5 * 1 / (1 - rho ** 2)) * (
                (dx ** 2) / (std1 ** 2) + (dy ** 2) / (std2 ** 2) - 2 * rho * dx * dy / (
                std1 * std2))  # (batch_size, num_timestamps)

        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

        return reg_loss, nearest_mode_idxs

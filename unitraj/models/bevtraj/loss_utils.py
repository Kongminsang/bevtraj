import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.bevtraj.utility import batch_nms


class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config

        self.goal_FDE_loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        self.smoothness_dt = float(self.config.get('smoothness_dt', 0.1))
        self.curvature_min_speed = float(self.config.get('curvature_min_speed', 0.5))
        self.initial_branch_warmup_epochs = int(
            self.config.get('initial_branch_warmup_epochs', 7)
        )
        self.initial_reg_aux_weight = float(
            self.config.get('initial_reg_aux_weight', 0.2)
        )
        self.inv_smoothness_dt2 = 1.0 / (self.smoothness_dt ** 2)
        self.inv_smoothness_dt3 = 1.0 / (self.smoothness_dt ** 3)

    def get_trajectory_smoothness_loss(self, pred_xy, valid_mask):
        valid_mask = valid_mask.to(pred_xy.dtype)

        disp = pred_xy[:, 1:, :] - pred_xy[:, :-1, :]
        accel = (disp[:, 1:, :] - disp[:, :-1, :]) * self.inv_smoothness_dt2

        accel_valid = valid_mask[:, :-2] * valid_mask[:, 1:-1] * valid_mask[:, 2:]
        loss_acc = (accel.square().sum(dim=-1) * accel_valid).sum()
        loss_acc = loss_acc / accel_valid.sum().clamp_min(1.0)

        jerk = (accel[:, 1:, :] - accel[:, :-1, :]) / self.smoothness_dt
        jerk_valid = accel_valid[:, :-1] * accel_valid[:, 1:]
        loss_jerk = (jerk.square().sum(dim=-1) * jerk_valid).sum()
        loss_jerk = loss_jerk / jerk_valid.sum().clamp_min(1.0)

        # Direction is undefined at low speeds, so only compare consecutive
        # displacements that are both above the configured speed threshold.
        disp_norm = torch.norm(disp, dim=-1)
        min_disp = self.curvature_min_speed * self.smoothness_dt
        direction_valid = (
            (disp_norm[:, :-1] > min_disp)
            & (disp_norm[:, 1:] > min_disp)
        ).to(pred_xy.dtype)
        curvature_valid = accel_valid * direction_valid

        vel_hat = disp / disp_norm.unsqueeze(-1).clamp_min(1e-6)
        cos_theta = (vel_hat[:, :-1, :] * vel_hat[:, 1:, :]).sum(dim=-1)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        loss_curvature = ((1.0 - cos_theta) * curvature_valid).sum()
        loss_curvature = loss_curvature / curvature_valid.sum().clamp_min(1.0)

        return loss_acc, loss_jerk, loss_curvature

    def forward(self, out, gt, center_gt_final_valid_idx, traj_data, current_epoch=0):
        modes_preds = out['predicted_probability'] # [B, K]
        preds = out['predicted_trajectory'] # [K, T, B, 5]
        pred_vels = out['predicted_velocity'] # [K, T, B, 2]

        anchor_pos = out['anchor_pos']
        goal_position = out['predicted_goal_position']
        goal_probability = out['predicted_goal_probability']
        goal_anchor_position = out['goal_anchor_position']
        goal_FDE = out['predicted_goal_FDE']

        dense_future_pred = out['dense_future_pred']

        state_pred = out['state_pred']

        gt_decoder = gt[0]
        gt_dense_future_trajs = gt[1]

        weighted_goal_position, positive_goal_component = \
            self.get_positive_goal_component(
                goal_probability=goal_probability,
                goal_anchor_position=goal_anchor_position,
                gt=gt_decoder,
                center_gt_final_valid_idx=center_gt_final_valid_idx,
            )
        
        decoder_loss = self.get_decoder_loss_hard_assign(
            modes_preds=modes_preds,
            preds=preds,
            pred_vels=pred_vels,
            anchor_pos=anchor_pos,
            goal_anchor_pos=weighted_goal_position,
            gt_decoder=gt_decoder,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
            current_epoch=current_epoch,
        )

        goal_prob_loss = self.get_goal_prob_loss(
            goal_probability=goal_probability,
            goal_anchor_position=goal_anchor_position,
            positive_goal_component=positive_goal_component,
            gt=gt_decoder,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
        )

        goal_fde_loss = self.get_goal_fde_loss(
            goal_position=goal_position,
            goal_FDE=goal_FDE,
            gt=gt_decoder,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
        )

        state_query_loss = self.get_state_query_loss(
            state_pred=state_pred,
            gt=gt_decoder,
        )

        dense_future_loss = self.get_dense_future_prediction_loss(dense_future_pred, gt_dense_future_trajs)

        total_loss = (
            decoder_loss
            + goal_prob_loss
            + goal_fde_loss
            + state_query_loss
            + dense_future_loss
        )
        return total_loss

    @staticmethod
    def get_initial_pair_assignment(
        pred_trajs,
        goal_anchor_pos,
        gt_xy,
        gt_mask,
        gt_goal,
        select_mask,
    ):
        """Match a goal mode first, then its regressed/predefined pair."""
        B, num_candidates, _, _ = pred_trajs.shape
        num_goal_modes = goal_anchor_pos.size(1)
        assert num_candidates == num_goal_modes * 2
        assert select_mask.shape == (B, num_candidates)

        pair_select_mask = select_mask.reshape(B, num_goal_modes, 2)
        mode_select_mask = pair_select_mask.any(dim=-1)
        assert mode_select_mask.any(dim=-1).all()

        goal_dist = (goal_anchor_pos - gt_goal[:, None, :]).norm(dim=-1)
        goal_dist = goal_dist.masked_fill(~mode_select_mask, 1e10)
        goal_mode_idx = goal_dist.argmin(dim=-1)

        branch_offsets = torch.arange(2, device=pred_trajs.device)
        pair_indices = goal_mode_idx[:, None] * 2 + branch_offsets[None, :]
        b_idx = torch.arange(B, device=pred_trajs.device)
        pair_trajs = pred_trajs[b_idx[:, None], pair_indices, :, :2]

        branch_dist = (pair_trajs - gt_xy[:, None, :, :]).norm(dim=-1)
        branch_dist = (branch_dist * gt_mask[:, None, :]).sum(dim=-1)
        branch_dist = branch_dist / gt_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

        pair_is_selected = pair_select_mask[b_idx, goal_mode_idx]
        branch_dist = branch_dist.masked_fill(~pair_is_selected, 1e10)
        branch_idx = branch_dist.argmin(dim=-1)
        hard_idx = pair_indices[b_idx, branch_idx]
        return pair_indices, hard_idx

    def get_decoder_loss_hard_assign( # EDA
        self,
        modes_preds,                 # list of [B, K]
        preds,                       # list of [K, T, B, 5]
        pred_vels,                   # list of [K, T, B, 2]
        anchor_pos,                  # [B, K, 2]
        goal_anchor_pos,             # [B, M, 2], before pair expansion
        gt_decoder,                  # [B, T, 5] -> (x, y, vx, vy, valid)
        center_gt_final_valid_idx,   # [B]
        current_epoch=0,
    ):
        device = gt_decoder.device
        B = gt_decoder.size(0)
        b_idx = torch.arange(B, device=device)

        gt_xy = gt_decoder[..., :2]                     # [B, T, 2]
        gt_vel = gt_decoder[..., 2:4]                   # [B, T, 2]
        gt_mask = gt_decoder[..., 4].float()            # [B, T]
        final_idx = center_gt_final_valid_idx.long()    # [B]
        gt_goal = gt_xy[b_idx, final_idx]               # [B, 2]
        valid_final = gt_mask[b_idx, final_idx]         # [B]

        w_cls = self.config.get('cls_weight', 2.0)
        w_reg = self.config.get('reg_weight', 1.0)
        w_vel = self.config.get('vel_weight', 0.2)
        w_cls_final = self.config.get('cls_weight_final', 2.0)
        w_reg_final = self.config.get('reg_weight_final', 1.0)
        w_vel_final = self.config.get('vel_weight_final', w_vel)
        w_acc = float(self.config.get('acc_weight', 0.0))
        w_jerk = float(self.config.get('jerk_weight', 0.0))
        w_curvature = float(self.config.get('curvature_weight', 0.0))
        w_acc_final = float(self.config.get('acc_weight_final', w_acc))
        w_jerk_final = float(self.config.get('jerk_weight_final', w_jerk))
        w_curvature_final = float(self.config.get('curvature_weight_final', w_curvature))

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
            cur_w_acc = w_acc_final if is_last_layer else w_acc
            cur_w_jerk = w_jerk_final if is_last_layer else w_jerk
            cur_w_curvature = w_curvature_final if is_last_layer else w_curvature

            # pred: [K, T, B, 5] -> [B, K, T, 5]
            pred_trajs = pred.permute(2, 0, 1, 3).contiguous()
            pred_vel = pred_vel.permute(2, 0, 1, 3).contiguous()   # [B, K, T, 2]

            # ---------- Evolving Anchors ----------
            positive_layer_idx = (layer_idx // num_inter_layers) * num_inter_layers - 1
            uses_goal_anchors = positive_layer_idx < 0
            if uses_goal_anchors:
                # first-stage anchor from SGCP anchors
                cur_anchor_pos = anchor_pos[:, :pred_scores.size(1)]

                anchor_trajs = cur_anchor_pos.detach().unsqueeze(2)
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

            # Match initial predictions hierarchically: select the SGCP goal
            # mode first, then compare its regressed/predefined trajectories.
            with torch.no_grad():
                if uses_goal_anchors:
                    pair_indices, matched_idx = self.get_initial_pair_assignment(
                        pred_trajs=pred_trajs.detach(),
                        goal_anchor_pos=goal_anchor_pos.detach(),
                        gt_xy=gt_xy,
                        gt_mask=gt_mask,
                        gt_goal=gt_goal,
                        select_mask=select_mask,
                    )
                    in_branch_warmup = current_epoch < self.initial_branch_warmup_epochs
                    # The predefined mean is fixed in the initial prediction,
                    # so use the regressed component for trajectory supervision
                    # while both branches are classification positives.
                    hard_idx = pair_indices[:, 0] if in_branch_warmup else matched_idx
                else:
                    dist = dist.masked_fill(~select_mask, 1e10)
                    hard_idx = dist.argmin(dim=-1)

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

            # Once hard branch matching starts, keep the regressed branch from
            # starving when the predefined trajectory is currently closer.
            loss_reg_aux = torch.zeros_like(loss_reg_gmm)
            if (
                uses_goal_anchors
                and not in_branch_warmup
                and self.initial_reg_aux_weight > 0
            ):
                reg_idx = pair_indices[:, 0]
                predefined_is_positive = hard_idx != reg_idx
                if predefined_is_positive.any():
                    loss_reg_aux, _ = self.nll_loss_gmm_direct(
                        pred_scores=pred_scores,
                        pred_trajs=pred_trajs_gmm,
                        gt_trajs=gt_xy,
                        gt_valid_mask=gt_mask,
                        pre_nearest_mode_idxs=reg_idx,
                        timestamp_loss_weight=None,
                        use_square_gmm=False,
                    )
                    loss_reg_aux = loss_reg_aux * predefined_is_positive.float()

            pred_xy_pos = mu[b_idx, hard_idx]  # [B, T, 2]
            pred_vel_pos = pred_vel[b_idx, hard_idx]  # [B, T, 2]
            loss_reg_vel = F.l1_loss(pred_vel_pos, gt_vel, reduction='none')
            loss_reg_vel = (loss_reg_vel * gt_mask[:, :, None]).sum(dim=-1).sum(dim=-1)
            loss_acc, loss_jerk, loss_curvature = self.get_trajectory_smoothness_loss(pred_xy_pos, gt_mask)

            # ---------- BCE classification ----------
            bce_target = torch.zeros_like(pred_scores)   # [B, K]
            if uses_goal_anchors and in_branch_warmup:
                bce_target.scatter_(1, pair_indices, 1.0)
            else:
                bce_target[b_idx, hard_idx] = 1.0
            loss_cls = F.binary_cross_entropy_with_logits(pred_scores, bce_target, reduction='none')  # [B, K]
            loss_cls = (loss_cls * select_mask.float()).sum(dim=-1)                                    # [B]

            layer_loss = cur_w_reg * (
                loss_reg_gmm + self.initial_reg_aux_weight * loss_reg_aux
            ) + cur_w_vel * loss_reg_vel + cur_w_cls * loss_cls
            layer_loss = (layer_loss * valid_final).sum() / valid_final.sum().clamp_min(1.0)
            layer_loss = layer_loss + cur_w_acc * loss_acc + cur_w_jerk * loss_jerk + cur_w_curvature * loss_curvature
            total = total + layer_loss

        return total / num_layers
    
    @staticmethod
    def get_positive_goal_component(
        goal_probability,
        goal_anchor_position,
        gt,
        center_gt_final_valid_idx,
    ):
        """Hard-assign each sample to the closest weighted goal component."""
        B, K, A = goal_probability.shape
        assert goal_anchor_position.shape == (B, K, A, 2)

        b_idx = torch.arange(B, device=goal_probability.device)
        final_idx = center_gt_final_valid_idx.long()
        gt_goal = gt[b_idx, final_idx, :2]

        with torch.no_grad():
            weighted_goal_position = (
                goal_probability.detach().unsqueeze(-1)
                * goal_anchor_position.detach()
            ).sum(dim=2)
            goal_distance = (
                weighted_goal_position - gt_goal[:, None]
            ).square().sum(dim=-1)
            positive_goal_component = goal_distance.argmin(dim=-1)

        return weighted_goal_position, positive_goal_component

    def get_goal_prob_loss(
        self,
        goal_probability,
        goal_anchor_position,
        positive_goal_component,
        gt,
        center_gt_final_valid_idx,
    ):
        """Train only the hard-assigned mode's distribution over its anchors."""
        eps = 1e-9
        entropy_weight = float(self.config.get('entropy_weight', 0.3))
        kl_weight = float(self.config.get('kl_weight', 1.0))
        sigma = float(self.config.get('goal_prob_sigma', 2.0))
        if sigma <= 0:
            raise ValueError('goal_prob_sigma must be positive')

        B, K, A = goal_probability.shape
        assert goal_anchor_position.shape == (B, K, A, 2)
        assert positive_goal_component.shape == (B,)

        device = goal_probability.device
        b_idx = torch.arange(B, device=device)
        final_idx = center_gt_final_valid_idx.long()
        gt_goal = gt[b_idx, final_idx, :2]
        valid_final = gt[b_idx, final_idx, -1].float()
        valid_count = valid_final.sum().clamp_min(1.0)

        # Only the selected goal component receives distribution supervision.
        goal_probability = goal_probability[b_idx, positive_goal_component]
        goal_anchor_position = goal_anchor_position[
            b_idx, positive_goal_component
        ]

        # Isotropic Gaussian likelihood p(goal_gt | anchor), with its constant
        # omitted because it cancels during posterior normalization.
        sq_dist = (
            goal_anchor_position - gt_goal[:, None]
        ).square().sum(dim=-1)
        log_likelihood = -0.5 * sq_dist / (sigma ** 2)

        # q(anchor) is the GT-conditioned posterior formed from the predicted
        # prior and the spatial likelihood.
        prior = goal_probability.clamp_min(eps)
        prior = prior / prior.sum(dim=-1, keepdim=True)
        log_prior = prior.log()
        log_posterior = log_likelihood + log_prior
        log_posterior = log_posterior - torch.logsumexp(
            log_posterior, dim=-1, keepdim=True
        )
        posterior = log_posterior.exp()

        nll_per_sample = ((-log_likelihood) * posterior).sum(dim=-1)
        nll = (nll_per_sample * valid_final).sum() / valid_count

        entropy_per_sample = -(
            posterior * log_posterior
        ).sum(dim=-1)
        posterior_entropy = (
            entropy_per_sample * valid_final
        ).sum() / valid_count

        kl_per_sample = (
            posterior * (log_posterior - log_prior)
        ).sum(dim=-1)
        kl_loss = (kl_per_sample * valid_final).sum() / valid_count

        return (
            nll
            + entropy_weight * posterior_entropy
            + kl_weight * kl_loss
        )

    def get_goal_fde_loss(
        self,
        goal_position,
        goal_FDE,
        gt,
        center_gt_final_valid_idx,
    ):
        """
        goal_position: [K, B, 2], discrete argmax anchor coordinates
        goal_FDE: [B, K]
        gt: [B, T, 5]  # (x, y, vx, vy, valid)
        center_gt_final_valid_idx: [B]
        """
        device = gt.device
        B = gt.size(0)
        b_idx = torch.arange(B, device=device)
        final_idx = center_gt_final_valid_idx.long()

        gt_goal = gt[b_idx, final_idx, :2]            # [B, 2]
        valid_final = gt[b_idx, final_idx, -1].float() # [B]

        # FDE is a detached quality target: it trains the ranking head without
        # leaking gradients into the proposed goal coordinates.
        final_goal_position = goal_position.permute(1, 0, 2).detach()
        FDE_gt = torch.norm(
            final_goal_position - gt_goal.unsqueeze(1), p=2, dim=-1
        )
        valid_rows = valid_final.bool()
        if valid_rows.any():
            disp_loss = self.goal_FDE_loss(
                goal_FDE[valid_rows], FDE_gt[valid_rows]
            )
        else:
            disp_loss = goal_FDE.sum() * 0.0

        return self.config.get('disp_weight', 1.0) * disp_loss
    
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

        w = float(self.config.get('dense_future_weight', 0.5))
        return w * loss_reg
    
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

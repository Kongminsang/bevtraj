import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.bevtraj.utility import batch_nms


class Criterion(nn.Module):
    def __init__(self, config, goal_cluster_anchors, goal_cluster_centroids):
        super(Criterion, self).__init__()
        self.config = config

        self.goal_FDE_loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        self.register_buffer('goal_cluster_anchors', goal_cluster_anchors.float(), persistent=False)
        self.register_buffer('goal_cluster_centroids', goal_cluster_centroids.float(), persistent=False)
        self.goal_positive_distance_threshold = float(self.config.get('goal_positive_distance_threshold', 4.0))
        self.goal_positive_temperature = float(
            self.config.get('goal_positive_temperature', self.goal_positive_distance_threshold)
        )
        self.smoothness_dt = float(self.config.get('smoothness_dt', 0.1))
        self.curvature_min_speed = float(self.config.get('curvature_min_speed', 0.5))
        self.inv_smoothness_dt2 = 1.0 / (self.smoothness_dt ** 2)
        self.inv_smoothness_dt3 = 1.0 / (self.smoothness_dt ** 3)
        default_goal_sigma = self.config.get('goal_prob_sigma', 2.5)
        self.goal_prob_lateral_sigma = float(self.config.get('goal_prob_lateral_sigma', default_goal_sigma))
        self.goal_prob_progress_sigma = float(self.config.get('goal_prob_progress_sigma', 8.0))

    def get_trajectory_smoothness_loss(self, pred_xy, valid_mask, mode_weights=None):
        if mode_weights is not None:
            valid_mask = valid_mask[:, None]
            mode_weights = mode_weights[..., None]
        else:
            mode_weights = 1.0
        valid_mask = valid_mask.to(pred_xy.dtype)

        disp = pred_xy[..., 1:, :] - pred_xy[..., :-1, :]
        accel = (disp[..., 1:, :] - disp[..., :-1, :]) * self.inv_smoothness_dt2

        accel_valid = valid_mask[..., :-2] * valid_mask[..., 1:-1] * valid_mask[..., 2:]
        weighted_accel_valid = accel_valid * mode_weights
        loss_acc = (accel.square().sum(dim=-1) * weighted_accel_valid).sum()
        loss_acc = loss_acc / weighted_accel_valid.sum().clamp_min(1.0)

        jerk = (accel[..., 1:, :] - accel[..., :-1, :]) / self.smoothness_dt
        jerk_valid = accel_valid[..., :-1] * accel_valid[..., 1:]
        weighted_jerk_valid = jerk_valid * mode_weights
        loss_jerk = (jerk.square().sum(dim=-1) * weighted_jerk_valid).sum()
        loss_jerk = loss_jerk / weighted_jerk_valid.sum().clamp_min(1.0)

        # Direction is undefined at low speeds, so only compare consecutive
        # displacements that are both above the configured speed threshold.
        disp_norm = torch.norm(disp, dim=-1)
        min_disp = self.curvature_min_speed * self.smoothness_dt
        direction_valid = ((disp_norm[..., :-1] > min_disp) & (disp_norm[..., 1:] > min_disp)).to(pred_xy.dtype)
        curvature_valid = accel_valid * direction_valid

        vel_hat = disp / disp_norm.unsqueeze(-1).clamp_min(1e-6)
        cos_theta = (vel_hat[..., :-1, :] * vel_hat[..., 1:, :]).sum(dim=-1)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        weighted_curvature_valid = curvature_valid * mode_weights
        loss_curvature = ((1.0 - cos_theta) * weighted_curvature_valid).sum()
        loss_curvature = loss_curvature / weighted_curvature_valid.sum().clamp_min(1.0)

        return loss_acc, loss_jerk, loss_curvature

    def get_positive_goal_weights(self, gt_goal):
        """Return centroid-weighted positive modes whose anchor regions contain each GT goal."""
        with torch.no_grad():
            anchor_distance_sq = (
                gt_goal[:, None, None] - self.goal_cluster_anchors[None]
            ).square().sum(dim=-1)
            min_anchor_distance_sq = anchor_distance_sq.amin(dim=-1)
            positive_mask = min_anchor_distance_sq <= self.goal_positive_distance_threshold ** 2

            missing_positive = ~positive_mask.any(dim=-1)
            if missing_positive.any():
                nearest_mode = min_anchor_distance_sq[missing_positive].argmin(dim=-1)
                positive_mask[missing_positive, nearest_mode] = True

            centroid_distance = torch.norm(
                gt_goal[:, None] - self.goal_cluster_centroids[None], dim=-1
            )
            logits = -centroid_distance / self.goal_positive_temperature
            return logits.masked_fill(~positive_mask, float('-inf')).softmax(dim=-1)

    def forward(self, out, gt, center_gt_final_valid_idx, traj_data):
        modes_preds = out['predicted_probability'] # [B, K]
        preds = out['predicted_trajectory'] # [K, T, B, 5]
        pred_vels = out['predicted_velocity'] # [K, T, B, 2]

        predicted_goal_position = out['predicted_goal_position']
        # goal_probability = out['predicted_goal_probability']
        # goal_anchor_position = out['goal_anchor_position']
        goal_FDE = out['predicted_goal_FDE']

        dense_future_pred = out['dense_future_pred']

        state_pred = out['state_pred']

        gt_decoder = gt[0]
        gt_dense_future_trajs = gt[1]

        b_idx = torch.arange(gt_decoder.size(0), device=gt_decoder.device)
        gt_goal = gt_decoder[b_idx, center_gt_final_valid_idx.long(), :2]
        positive_goal_weights = self.get_positive_goal_weights(gt_goal)
        
        decoder_loss = self.get_decoder_loss(
            modes_preds=modes_preds,
            preds=preds,
            pred_vels=pred_vels,
            predicted_goal_position=predicted_goal_position,
            positive_goal_weights=positive_goal_weights,
            gt_decoder=gt_decoder,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
        )

        # goal_prob_loss = self.get_goal_prob_loss(
        #     goal_probability=goal_probability,
        #     goal_anchor_position=goal_anchor_position,
        #     positive_goal_component=positive_goal_component,
        #     gt=gt_decoder,
        #     center_gt_final_valid_idx=center_gt_final_valid_idx,
        # )

        goal_prediction_loss = self.get_goal_prediction_loss(
            predicted_goal_position=predicted_goal_position,
            goal_FDE=goal_FDE,
            positive_goal_weights=positive_goal_weights,
            gt=gt_decoder,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
        )

        state_query_loss = self.get_state_query_loss(state_pred=state_pred, gt=gt_decoder)

        dense_future_loss = self.get_dense_future_prediction_loss(dense_future_pred, gt_dense_future_trajs)

        total_loss = decoder_loss + goal_prediction_loss + state_query_loss + dense_future_loss
        return total_loss

    def get_decoder_loss( # EDA
        self,
        modes_preds,                 # list of [B, K]
        preds,                       # list of [K, T, B, 5]
        pred_vels,                   # list of [K, T, B, 2]
        predicted_goal_position,     # [B, K, 2]
        positive_goal_weights,       # [B, K]
        gt_decoder,                  # [B, T, 5] -> (x, y, vx, vy, valid)
        center_gt_final_valid_idx,   # [B]
    ):
        device = gt_decoder.device
        B = gt_decoder.size(0)
        b_idx = torch.arange(B, device=device)

        gt_xy = gt_decoder[..., :2]                     # [B, T, 2]
        gt_vel = gt_decoder[..., 2:4]                   # [B, T, 2]
        gt_mask = gt_decoder[..., 4].float()            # [B, T]
        final_idx = center_gt_final_valid_idx.long()    # [B]
        valid_final = gt_mask[b_idx, final_idx]         # [B]

        w_cls = self.config.get('cls_weight', 2.0)
        w_reg = self.config.get('reg_weight', 1.0)
        w_vel = self.config.get('vel_weight', 0.2)
        w_acc = float(self.config.get('acc_weight', 0.0))
        w_jerk = float(self.config.get('jerk_weight', 0.0))
        w_curvature = float(self.config.get('curvature_weight', 0.0))

        # EDA-related config
        num_inter_layers = int(self.config.get('num_inter_layers', 2))
        use_distinct_anchors = self.config.get('distinct_anchors', False)
        distinct_nms_thresh = float(self.config.get('distinct_nms_thresh', -1.0))

        total = 0.0
        num_layers = len(preds)

        for layer_idx, (pred_scores, pred, pred_vel) in enumerate(zip(modes_preds, preds, pred_vels)):
            # pred: [K, T, B, 5] -> [B, K, T, 5]
            pred_trajs = pred.permute(2, 0, 1, 3).contiguous()
            pred_vel = pred_vel.permute(2, 0, 1, 3).contiguous()   # [B, K, T, 2]

            # ---------- Evolving Anchors ----------
            positive_layer_idx = (layer_idx // num_inter_layers) * num_inter_layers - 1
            if positive_layer_idx < 0:
                cur_goal_position = predicted_goal_position[:, :pred_scores.size(1)]
                anchor_trajs = cur_goal_position.detach().unsqueeze(2)
            else:
                # use previous anchor trajectories
                anchor_trajs = preds[positive_layer_idx].permute(2, 0, 1, 3).detach()  # [B, K, T, 5]

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

            positive_mask = positive_goal_weights[:, :pred_scores.size(1)] > 0
            select_mask = select_mask | positive_mask

            if positive_layer_idx < 0:
                layer_positive_weights = positive_goal_weights[:, :pred_scores.size(1)]

            # MTR nll_loss_gmm_direct expects log_std, but MotionRegHead outputs sigma
            mu = pred_trajs[..., :2]
            log_std = torch.log(pred_trajs[..., 2:4].clamp_min(1e-6))
            rho = pred_trajs[..., 4:5]
            pred_trajs_gmm = torch.cat([mu, log_std, rho], dim=-1)

            if positive_layer_idx >= 0:
                dist = ((gt_xy[:, None] - mu.detach()).norm(dim=-1) * gt_mask[:, None]).sum(dim=-1)
                logits = -dist / self.goal_positive_temperature
                layer_positive_weights = logits.masked_fill(~positive_mask, float('-inf')).softmax(dim=-1)

            loss_reg_gmm = self.nll_loss_gmm_soft_assign(
                pred_trajs=pred_trajs_gmm,
                gt_trajs=gt_xy,
                gt_valid_mask=gt_mask,
                mode_weights=layer_positive_weights,
            )
            gt_vel_expanded = gt_vel[:, None].expand_as(pred_vel)
            loss_reg_vel = F.l1_loss(pred_vel, gt_vel_expanded, reduction='none')
            loss_reg_vel = (loss_reg_vel * gt_mask[:, None, :, None]).sum(dim=-1).sum(dim=-1)
            loss_reg_vel = (loss_reg_vel * layer_positive_weights).sum(dim=-1)
            loss_acc, loss_jerk, loss_curvature = self.get_trajectory_smoothness_loss(
                mu, gt_mask, layer_positive_weights
            )
            bce_target = positive_mask.float()
            loss_cls = F.binary_cross_entropy_with_logits(pred_scores, bce_target, reduction='none')
            bce_weight = torch.where(positive_mask, layer_positive_weights, torch.ones_like(pred_scores))
            loss_cls = (loss_cls * bce_weight * select_mask.float()).sum(dim=-1)

            layer_loss = w_reg * loss_reg_gmm + w_vel * loss_reg_vel + w_cls * loss_cls
            layer_loss = (layer_loss * valid_final).sum() / valid_final.sum().clamp_min(1.0)
            layer_loss = layer_loss + w_acc * loss_acc + w_jerk * loss_jerk + w_curvature * loss_curvature
            total = total + layer_loss

        return total / num_layers

    # def get_goal_prob_loss(
    #     self,
    #     goal_probability,
    #     goal_anchor_position,
    #     positive_goal_component,
    #     gt,
    #     center_gt_final_valid_idx,
    # ):
    #     """Train only the hard-assigned mode's distribution over its anchors."""
    #     eps = 1e-9
    #     entropy_weight = float(self.config.get('entropy_weight', 0.3))
    #     kl_weight = float(self.config.get('kl_weight', 1.0))
    #
    #     B, K, A = goal_probability.shape
    #     assert goal_anchor_position.shape == (B, K, A, 2)
    #     assert positive_goal_component.shape == (B,)
    #
    #     device = goal_probability.device
    #     b_idx = torch.arange(B, device=device)
    #     final_idx = center_gt_final_valid_idx.long()
    #     valid_final = gt[b_idx, final_idx, -1].float()
    #     valid_count = valid_final.sum().clamp_min(1.0)
    #
    #     goal_probability = goal_probability[b_idx, positive_goal_component]
    #     goal_anchor_position = goal_anchor_position[b_idx, positive_goal_component]
    #     path_cost = self._get_goal_path_cost(
    #         goal_position=goal_anchor_position,
    #         gt=gt,
    #         center_gt_final_valid_idx=center_gt_final_valid_idx,
    #         lateral_sigma=self.goal_prob_lateral_sigma,
    #         progress_sigma=self.goal_prob_progress_sigma,
    #     )
    #     log_likelihood = -path_cost
    #     prior = goal_probability.clamp_min(eps)
    #     prior = prior / prior.sum(dim=-1, keepdim=True)
    #     log_prior = prior.log()
    #     log_posterior = log_likelihood + log_prior
    #     log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=-1, keepdim=True)
    #     posterior = log_posterior.exp()
    #     nll_per_sample = ((-log_likelihood) * posterior).sum(dim=-1)
    #     nll = (nll_per_sample * valid_final).sum() / valid_count
    #     entropy_per_sample = -(posterior * log_posterior).sum(dim=-1)
    #     posterior_entropy = (entropy_per_sample * valid_final).sum() / valid_count
    #     kl_per_sample = (posterior * (log_posterior - log_prior)).sum(dim=-1)
    #     kl_loss = (kl_per_sample * valid_final).sum() / valid_count
    #     return nll + entropy_weight * posterior_entropy + kl_weight * kl_loss

    def get_goal_prediction_loss(
        self,
        predicted_goal_position,
        goal_FDE,
        positive_goal_weights,
        gt,
        center_gt_final_valid_idx,
    ):
        """
        predicted_goal_position: [B, K, 2], attention-weighted goal coordinates
        goal_FDE: [B, K]
        positive_goal_weights: [B, K], centroid-distance weights over cluster-matched modes
        gt: [B, T, 5]  # (x, y, vx, vy, valid)
        center_gt_final_valid_idx: [B]
        """
        device = gt.device
        B = gt.size(0)
        b_idx = torch.arange(B, device=device)
        final_idx = center_gt_final_valid_idx.long()

        gt_goal = gt[b_idx, final_idx, :2]            # [B, 2]
        valid_final = gt[b_idx, final_idx, -1].float() # [B]

        position_error = torch.norm(predicted_goal_position - gt_goal[:, None], p=2, dim=-1)
        position_loss_per_sample = (position_error * positive_goal_weights).sum(dim=-1)
        position_loss = (position_loss_per_sample * valid_final).sum() / valid_final.sum().clamp_min(1.0)

        # FDE is a detached quality target: it trains the ranking head without
        # leaking gradients into the proposed goal coordinates.
        final_goal_position = predicted_goal_position.detach()
        FDE_gt = torch.norm(final_goal_position - gt_goal.unsqueeze(1), p=2, dim=-1)
        valid_rows = valid_final.bool()
        if valid_rows.any():
            disp_loss = self.goal_FDE_loss(goal_FDE[valid_rows], FDE_gt[valid_rows])
        else:
            disp_loss = goal_FDE.sum() * 0.0

        return (
            self.config.get('goal_position_weight', 1.0) * position_loss
            + self.config.get('disp_weight', 1.0) * disp_loss
        )
    
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
        pred_dense_trajs = prediction  # (num_center_objects, num_objects, num_future_frames, 7)
        assert pred_dense_trajs.shape[-1] == 7
        assert obj_trajs_future_state.shape[-1] == 4

        pred_dense_trajs_gmm = pred_dense_trajs[:, :, :, 0:5]
        pred_dense_trajs_vel = pred_dense_trajs[:, :, :, 5:7]

        loss_reg_vel = F.l1_loss(pred_dense_trajs_vel, obj_trajs_future_state[:, :, :, 2:4], reduction='none')
        loss_reg_vel = (loss_reg_vel * obj_trajs_future_mask[:, :, :, None]).sum(dim=-1).sum(dim=-1)

        num_center_objects, num_objects, num_timestamps, _ = pred_dense_trajs.shape
        total_objects = num_center_objects * num_objects
        fake_scores = pred_dense_trajs.new_zeros(total_objects, 1)
        temp_pred_trajs = pred_dense_trajs_gmm.contiguous().view(total_objects, 1, num_timestamps, 5)
        temp_gt_idx = torch.zeros(total_objects).long()
        temp_gt_trajs = obj_trajs_future_state[:, :, :, 0:2].contiguous().view(total_objects, num_timestamps, 2)
        temp_gt_trajs_mask = obj_trajs_future_mask.view(total_objects, num_timestamps)
        loss_reg_gmm, _ = self.nll_loss_gmm_direct(
            pred_scores=fake_scores,
            pred_trajs=temp_pred_trajs,
            gt_trajs=temp_gt_trajs,
            gt_valid_mask=temp_gt_trajs_mask,
            pre_nearest_mode_idxs=temp_gt_idx,
            timestamp_loss_weight=None,
            use_square_gmm=False,
        )
        loss_reg_gmm = loss_reg_gmm.view(num_center_objects, num_objects)

        loss_reg = loss_reg_vel + loss_reg_gmm

        obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0

        valid_objects = torch.clamp_min(obj_valid_mask.sum(dim=-1), min=1.0)
        loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / valid_objects
        loss_reg = loss_reg.mean()

        w = float(self.config.get('dense_future_weight', 0.5))
        return w * loss_reg

    def nll_loss_gmm_soft_assign(self, pred_trajs, gt_trajs, gt_valid_mask, mode_weights):
        B, K, T, C = pred_trajs.shape
        flat_pred_trajs = pred_trajs.reshape(B * K, 1, T, C)
        flat_gt_trajs = gt_trajs[:, None].expand(-1, K, -1, -1).reshape(B * K, T, 2)
        flat_gt_valid_mask = gt_valid_mask[:, None].expand(-1, K, -1).reshape(B * K, T)
        flat_scores = pred_trajs.new_zeros(B * K, 1)
        flat_mode_idx = torch.zeros(B * K, device=pred_trajs.device, dtype=torch.long)
        loss_reg, _ = self.nll_loss_gmm_direct(
            pred_scores=flat_scores,
            pred_trajs=flat_pred_trajs,
            gt_trajs=flat_gt_trajs,
            gt_valid_mask=flat_gt_valid_mask,
            pre_nearest_mode_idxs=flat_mode_idx,
            timestamp_loss_weight=None,
            use_square_gmm=False,
        )
        return (loss_reg.reshape(B, K) * mode_weights).sum(dim=-1)
    
    def nll_loss_gmm_direct(
        self,
        pred_scores,
        pred_trajs,
        gt_trajs,
        gt_valid_mask,
        pre_nearest_mode_idxs=None,
        timestamp_loss_weight=None,
        use_square_gmm=False,
        log_std_range=(-1.609, 5.0),
        rho_limit=0.5,
    ):
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
            (dx ** 2) / (std1 ** 2)
            + (dy ** 2) / (std2 ** 2)
            - 2 * rho * dx * dy / (std1 * std2)
        )  # (batch_size, num_timestamps)

        reg_loss = ((reg_gmm_log_coefficient + reg_gmm_exp) * gt_valid_mask).sum(dim=-1)

        return reg_loss, nearest_mode_idxs

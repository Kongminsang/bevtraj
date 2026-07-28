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
        self.inv_smoothness_dt2 = 1.0 / (self.smoothness_dt ** 2)
        self.inv_smoothness_dt3 = 1.0 / (self.smoothness_dt ** 3)
        default_goal_sigma = self.config.get('goal_prob_sigma', 2.5)
        self.goal_prob_lateral_sigma = float(self.config.get('goal_prob_lateral_sigma', default_goal_sigma))
        self.goal_prob_progress_sigma = float(self.config.get('goal_prob_progress_sigma', 8.0))

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
        direction_valid = ((disp_norm[:, :-1] > min_disp) & (disp_norm[:, 1:] > min_disp)).to(pred_xy.dtype)
        curvature_valid = accel_valid * direction_valid

        vel_hat = disp / disp_norm.unsqueeze(-1).clamp_min(1e-6)
        cos_theta = (vel_hat[:, :-1, :] * vel_hat[:, 1:, :]).sum(dim=-1)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        loss_curvature = ((1.0 - cos_theta) * curvature_valid).sum()
        loss_curvature = loss_curvature / curvature_valid.sum().clamp_min(1.0)

        return loss_acc, loss_jerk, loss_curvature

    def forward(self, out, gt, center_gt_final_valid_idx, traj_data):
        modes_preds = out['predicted_probability'] # [B, K]
        preds = out['predicted_trajectory'] # [K, T, B, 5]
        pred_vels = out['predicted_velocity'] # [K, T, B, 2]

        predicted_goal_position = out['predicted_goal_position']
        goal_probability = out['predicted_goal_probability']
        goal_anchor_position = out['goal_anchor_position']
        goal_FDE = out['predicted_goal_FDE']

        dense_future_pred = out['dense_future_pred']

        state_pred = out['state_pred']

        gt_decoder = gt[0]
        gt_dense_future_trajs = gt[1]

        positive_goal_component = self.get_positive_goal_component(
            goal_probability=goal_probability,
            goal_anchor_position=goal_anchor_position,
            gt=gt_decoder,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
        )
        
        decoder_loss = self.get_decoder_loss_hard_assign(
            modes_preds=modes_preds,
            preds=preds,
            pred_vels=pred_vels,
            predicted_goal_position=predicted_goal_position,
            gt_decoder=gt_decoder,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
        )

        goal_prob_loss = self.get_goal_prob_loss(
            goal_probability=goal_probability,
            goal_anchor_position=goal_anchor_position,
            positive_goal_component=positive_goal_component,
            gt=gt_decoder,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
        )

        goal_fde_loss = self.get_goal_fde_loss(
            predicted_goal_position=predicted_goal_position,
            goal_FDE=goal_FDE,
            gt=gt_decoder,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
        )

        state_query_loss = self.get_state_query_loss(state_pred=state_pred, gt=gt_decoder)

        dense_future_loss = self.get_dense_future_prediction_loss(dense_future_pred, gt_dense_future_trajs)

        total_loss = decoder_loss + goal_prob_loss + goal_fde_loss + state_query_loss + dense_future_loss
        return total_loss

    def get_decoder_loss_hard_assign( # EDA
        self,
        modes_preds,                 # list of [B, K]
        preds,                       # list of [K, T, B, 5]
        pred_vels,                   # list of [K, T, B, 2]
        predicted_goal_position,     # [B, K, 2]
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
        gt_goal = gt_xy[b_idx, final_idx]               # [B, 2]
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
                # first-stage anchor from SGCP anchors
                cur_goal_position = predicted_goal_position[:, :pred_scores.size(1)]

                anchor_trajs = cur_goal_position.detach().unsqueeze(2)
                dist = (cur_goal_position.detach() - gt_goal[:, None, :]).norm(dim=-1)
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

            pred_xy_pos = mu[b_idx, hard_idx]  # [B, T, 2]
            pred_vel_pos = pred_vel[b_idx, hard_idx]  # [B, T, 2]
            loss_reg_vel = F.l1_loss(pred_vel_pos, gt_vel, reduction='none')
            loss_reg_vel = (loss_reg_vel * gt_mask[:, :, None]).sum(dim=-1).sum(dim=-1)
            loss_acc, loss_jerk, loss_curvature = self.get_trajectory_smoothness_loss(pred_xy_pos, gt_mask)

            # ---------- BCE classification ----------
            bce_target = torch.zeros_like(pred_scores)   # [B, K]
            bce_target[b_idx, hard_idx] = 1.0
            loss_cls = F.binary_cross_entropy_with_logits(pred_scores, bce_target, reduction='none')  # [B, K]
            loss_cls = (loss_cls * select_mask.float()).sum(dim=-1)                                    # [B]

            layer_loss = w_reg * loss_reg_gmm + w_vel * loss_reg_vel + w_cls * loss_cls
            layer_loss = (layer_loss * valid_final).sum() / valid_final.sum().clamp_min(1.0)
            layer_loss = layer_loss + w_acc * loss_acc + w_jerk * loss_jerk + w_curvature * loss_curvature
            total = total + layer_loss

        return total / num_layers
    
    @staticmethod
    def _get_goal_path_cost(
        goal_position,
        gt,
        center_gt_final_valid_idx,
        lateral_sigma,
        progress_sigma,
    ):
        """
        Compute a Frenet-like cost between goals and the valid GT path.

        The current target position (the local-coordinate origin) is prepended
        to the future GT trajectory. Goals are projected onto the resulting
        polyline and penalized separately by lateral distance and remaining
        path progress. The first and last non-degenerate segments are extended
        so that goals behind the start or beyond the endpoint still receive an
        along-path progress error instead of an isotropic endpoint error.

        Args:
            goal_position: [B, ..., 2]
            gt: [B, T, 5], with validity in the last channel
            center_gt_final_valid_idx: [B]
        Returns:
            cost: [B, ...]
        """
        B = gt.size(0)
        assert goal_position.shape[0] == B
        assert goal_position.shape[-1] == 2

        original_shape = goal_position.shape[1:-1]
        goals = goal_position.reshape(B, -1, 2)
        gt_xy = gt[..., :2].to(goal_position.dtype)
        gt_valid = gt[..., -1].bool()

        origin = torch.zeros(B, 1, 2, device=goal_position.device, dtype=goal_position.dtype)
        path_points = torch.cat([origin, gt_xy], dim=1)
        point_valid = torch.cat(
            [torch.ones(B, 1, device=gt.device, dtype=torch.bool), gt_valid],
            dim=1,
        )

        segment_start = path_points[:, :-1]
        segment_delta = path_points[:, 1:] - segment_start
        segment_length = segment_delta.norm(dim=-1)
        segment_valid = point_valid[:, :-1] & point_valid[:, 1:] & (segment_length > 1e-6)

        segment_length_sq = segment_delta.square().sum(dim=-1)
        relative = goals[:, :, None, :] - segment_start[:, None, :, :]
        raw_t = (relative * segment_delta[:, None]).sum(dim=-1)
        raw_t = raw_t / segment_length_sq[:, None].clamp_min(1e-12)
        clamped_t = raw_t.clamp(0.0, 1.0)

        bounded_projection = segment_start[:, None] + clamped_t[..., None] * segment_delta[:, None]
        bounded_sq_dist = (goals[:, :, None] - bounded_projection).square().sum(dim=-1)
        bounded_sq_dist = bounded_sq_dist.masked_fill(~segment_valid[:, None], float('inf'))
        selected_segment = bounded_sq_dist.argmin(dim=-1)

        gather_idx = selected_segment[..., None]
        selected_raw_t = raw_t.gather(2, gather_idx).squeeze(-1)
        selected_clamped_t = clamped_t.gather(2, gather_idx).squeeze(-1)

        segment_order = torch.arange(segment_valid.size(1), device=gt.device)[None]
        first_segment = segment_valid.long().argmax(dim=-1)
        last_segment = (segment_order * segment_valid.long()).max(dim=-1).values
        extrapolate = (
            (selected_segment == first_segment[:, None])
            & (selected_raw_t < 0.0)
        ) | (
            (selected_segment == last_segment[:, None])
            & (selected_raw_t > 1.0)
        )
        selected_t = torch.where(extrapolate, selected_raw_t, selected_clamped_t)

        gather_index = selected_segment[..., None].expand(-1, -1, 2)
        selected_start = segment_start.gather(1, gather_index)
        selected_delta = segment_delta.gather(1, gather_index)
        projection = selected_start + selected_t[..., None] * selected_delta
        lateral_sq_dist = (goals - projection).square().sum(dim=-1)

        valid_segment_length = segment_length * segment_valid
        segment_end_progress = valid_segment_length.cumsum(dim=-1)
        segment_start_progress = segment_end_progress - valid_segment_length
        selected_start_progress = segment_start_progress.gather(1, selected_segment)
        selected_length = segment_length.gather(1, selected_segment)
        projected_progress = selected_start_progress + selected_t * selected_length
        total_progress = segment_end_progress[:, -1]
        progress_error = projected_progress - total_progress[:, None]

        cost = (
            0.5 * lateral_sq_dist / (lateral_sigma ** 2)
            + 0.5 * progress_error.square() / (progress_sigma ** 2)
        )

        # Rows without a non-degenerate valid segment correspond to stationary
        # (or invalid) GT. Use the endpoint distance for stationary rows; the
        # caller masks invalid rows from the final loss.
        has_path = segment_valid.any(dim=-1)
        b_idx = torch.arange(B, device=gt.device)
        final_idx = center_gt_final_valid_idx.long()
        gt_goal = gt_xy[b_idx, final_idx]
        endpoint_cost = (goals - gt_goal[:, None]).square().sum(dim=-1) * (0.5 / (lateral_sigma ** 2))
        cost = torch.where(has_path[:, None], cost, endpoint_cost)

        return cost.reshape(B, *original_shape)

    def get_positive_goal_component(
        self,
        goal_probability,
        goal_anchor_position,
        gt,
        center_gt_final_valid_idx,
    ):
        """Hard-assign each sample using the weighted goal's GT-path cost."""
        B, K, A = goal_probability.shape
        assert goal_anchor_position.shape == (B, K, A, 2)

        with torch.no_grad():
            weighted_goal_position = (
                goal_probability.detach().unsqueeze(-1)
                * goal_anchor_position.detach()
            ).sum(dim=2)
            goal_path_cost = self._get_goal_path_cost(
                goal_position=weighted_goal_position,
                gt=gt,
                center_gt_final_valid_idx=center_gt_final_valid_idx,
                lateral_sigma=self.goal_prob_lateral_sigma,
                progress_sigma=self.goal_prob_progress_sigma,
            )
            positive_goal_component = goal_path_cost.argmin(dim=-1)

        return positive_goal_component

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

        B, K, A = goal_probability.shape
        assert goal_anchor_position.shape == (B, K, A, 2)
        assert positive_goal_component.shape == (B,)

        device = goal_probability.device
        b_idx = torch.arange(B, device=device)
        final_idx = center_gt_final_valid_idx.long()
        valid_final = gt[b_idx, final_idx, -1].float()
        valid_count = valid_final.sum().clamp_min(1.0)

        # Only the selected goal component receives distribution supervision.
        goal_probability = goal_probability[b_idx, positive_goal_component]
        goal_anchor_position = goal_anchor_position[b_idx, positive_goal_component]

        # Anisotropic path-tube likelihood: deviations across the GT path are
        # penalized more strongly than along-path progress (speed) errors.
        path_cost = self._get_goal_path_cost(
            goal_position=goal_anchor_position,
            gt=gt,
            center_gt_final_valid_idx=center_gt_final_valid_idx,
            lateral_sigma=self.goal_prob_lateral_sigma,
            progress_sigma=self.goal_prob_progress_sigma,
        )
        log_likelihood = -path_cost

        # q(anchor) is the GT-conditioned posterior formed from the predicted
        # prior and the spatial likelihood.
        prior = goal_probability.clamp_min(eps)
        prior = prior / prior.sum(dim=-1, keepdim=True)
        log_prior = prior.log()
        log_posterior = log_likelihood + log_prior
        log_posterior = log_posterior - torch.logsumexp(log_posterior, dim=-1, keepdim=True)
        posterior = log_posterior.exp()

        nll_per_sample = ((-log_likelihood) * posterior).sum(dim=-1)
        nll = (nll_per_sample * valid_final).sum() / valid_count

        entropy_per_sample = -(posterior * log_posterior).sum(dim=-1)
        posterior_entropy = (entropy_per_sample * valid_final).sum() / valid_count

        kl_per_sample = (posterior * (log_posterior - log_prior)).sum(dim=-1)
        kl_loss = (kl_per_sample * valid_final).sum() / valid_count

        return nll + entropy_weight * posterior_entropy + kl_weight * kl_loss

    def get_goal_fde_loss(self, predicted_goal_position, goal_FDE, gt, center_gt_final_valid_idx):
        """
        predicted_goal_position: [B, K, 2], discrete argmax anchor coordinates
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
        final_goal_position = predicted_goal_position.detach()
        FDE_gt = torch.norm(final_goal_position - gt_goal.unsqueeze(1), p=2, dim=-1)
        valid_rows = valid_final.bool()
        if valid_rows.any():
            disp_loss = self.goal_FDE_loss(goal_FDE[valid_rows], FDE_gt[valid_rows])
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

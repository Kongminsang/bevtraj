from scipy import special
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Laplace
import torch.nn.functional as F

class Criterion(nn.Module):
    def __init__(self, config):
        super(Criterion, self).__init__()
        self.config = config
        self.traj_type_weights = config.get('traj_type_weights', False)
        self.goal_FDE_loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        if self.traj_type_weights:
            if config['dataset_name'] == 'nusc':
                self.register_buffer('TRAJ_TYPE_WEIGHTS', torch.tensor([0.8543, 0.5764, 0.8082, 0.7961, 1.9628, 0.7278, 1.5563, 0.7181], dtype=torch.float32))
            elif config['dataset_name'] == 'argo2':
                self.register_buffer('TRAJ_TYPE_WEIGHTS', torch.tensor([0.7329, 0.5631, 1.0101, 1.0163, 1.9166, 0.8291, 2.0218, 0.8902], dtype=torch.float32))
            else: raise ValueError('Unknown dataset name')

    def forward(self, out, gt, center_gt_final_valid_idx, traj_data):
        modes_preds = out['predicted_probability'] # [B, K]
        preds = out['predicted_trajectory'] # [K, T, B, 5]
        goal_FDE = out['predicted_goal_FDE'] # [B, K]
        goal_reg = out['predicted_goal_reg'] # [K, B, 2]
        dense_future_pred = out['dense_future_pred']
        gt_decoder = gt[0]
        gt_dense_future_trajs = gt[1]
        
        total_loss = 0
        for mode_pred, pred in zip(modes_preds, preds):
            layer_loss = self.nll_loss_multimodes(mode_pred, pred, gt_decoder, center_gt_final_valid_idx, traj_data)
            total_loss += layer_loss

        goal_prediction_loss = self.get_goal_prediction_loss(goal_reg, goal_FDE, gt_decoder, traj_data)
        dense_future_loss = self.get_dense_future_prediction_loss(dense_future_pred, gt_dense_future_trajs)
        total_loss = total_loss / len(preds) + goal_prediction_loss + dense_future_loss
        return total_loss

    def get_BVG_distributions(self, pred):
        B = pred.size(0)
        T = pred.size(1)
        mu_x = pred[:, :, 0].unsqueeze(2)
        mu_y = pred[:, :, 1].unsqueeze(2)
        sigma_x = pred[:, :, 2]
        sigma_y = pred[:, :, 3]
        rho = pred[:, :, 4]

        cov = torch.zeros((B, T, 2, 2)).to(pred.device)
        cov[:, :, 0, 0] = sigma_x ** 2
        cov[:, :, 1, 1] = sigma_y ** 2
        cov[:, :, 0, 1] = rho * sigma_x * sigma_y
        cov[:, :, 1, 0] = rho * sigma_x * sigma_y

        biv_gauss_dist = MultivariateNormal(loc=torch.cat((mu_x, mu_y), dim=-1), covariance_matrix=cov)
        return biv_gauss_dist

    def get_Laplace_dist(self, pred):
        return Laplace(pred[:, :, :2], pred[:, :, 2:4])

    def nll_pytorch_dist(self, pred, data, mask, rtn_loss=True):
        # biv_gauss_dist = get_BVG_distributions(pred)
        biv_gauss_dist = self.get_Laplace_dist(pred)
        num_active_per_timestep = mask.sum()
        data_reshaped = data[:, :, :2]
        if rtn_loss:
            # return (-biv_gauss_dist.log_prob(data)).sum(1)  # Gauss
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(-1) * mask).sum(1)  # Laplace
        else:
            # return (-biv_gauss_dist.log_prob(data)).sum(-1)  # Gauss
            # need to multiply by masks
            # return (-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=(1, 2))  # Laplace
            return ((-biv_gauss_dist.log_prob(data_reshaped)).sum(dim=2) * mask).sum(1)  # Laplace

    def nll_loss_multimodes(self, modes_pred, pred, data, center_gt_final_valid_idx, traj_data):
        """NLL loss multimodes for training. MFP Loss function
        Args:
          modes_pred: [B, K], prior prob over modes
          pred: [K, T, B, 5]
          data: [B, T, 5]
          noise is optional
        """
        mask = data[..., -1]

        entropy_weight = self.config['entropy_weight']
        kl_weight = self.config['kl_weight']
        use_FDEADE_aux_loss = self.config['use_FDEADE_aux_loss']

        modes = len(pred)
        nSteps, batch_sz, dim = pred[0].shape

        # compute posterior probability based on predicted prior and likelihood of predicted trajectory.
        log_lik = np.zeros((batch_sz, modes))
        with torch.no_grad():
            for kk in range(modes):
                nll = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=False)
                log_lik[:, kk] = -nll.cpu().numpy()

        priors = modes_pred.detach().cpu().numpy()
        log_posterior_unnorm = log_lik + np.log(priors)
        log_posterior = log_posterior_unnorm - special.logsumexp(log_posterior_unnorm, axis=-1).reshape((batch_sz, -1))
        post_pr = np.exp(log_posterior)
        post_pr = torch.tensor(post_pr).float().to(data.device)

        # Compute loss.
        loss = 0.0
        for kk in range(modes):
            nll_k = self.nll_pytorch_dist(pred[kk].transpose(0, 1), data, mask, rtn_loss=True) * post_pr[:, kk]
            loss += nll_k.mean()

        # Adding entropy loss term to ensure that individual predictions do not try to cover multiple modes.
        entropy_vals = []
        for kk in range(modes):
            entropy_vals.append(self.get_BVG_distributions(pred[kk]).entropy())
        entropy_vals = torch.stack(entropy_vals).permute(2, 0, 1)
        entropy_loss = torch.mean((entropy_vals).sum(2).max(1)[0])
        loss += entropy_weight * entropy_loss

        # KL divergence between the prior and the posterior distributions.
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # type: ignore
        kl_loss = kl_weight * kl_loss_fn(torch.log(modes_pred), post_pr)

        # compute ADE/FDE loss - L2 norms with between best predictions and GT.
        if use_FDEADE_aux_loss:
            adefde_loss = self.l2_loss_fde(pred, data, mask, traj_data)
        else:
            adefde_loss = torch.tensor(0.0).to(data.device)

        # post_entropy
        final_loss = loss + kl_loss + adefde_loss

        return final_loss

    def l2_loss_fde(self, pred, data, mask, traj_data):
        fde_loss = (torch.norm((pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1) * mask[:,
                                                                                                                 -1:])
        ade_loss = (torch.norm((pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2,
                               dim=-1) * mask.unsqueeze(0)).mean(dim=2).transpose(0, 1)
        if self.traj_type_weights:
            type_weight = self.TRAJ_TYPE_WEIGHTS[traj_data['trajectory_type']].unsqueeze(1) # size(B, 1)
            loss, min_inds = ((fde_loss + ade_loss) * type_weight).min(dim=1)
        else:
            loss, min_inds = (fde_loss + ade_loss).min(dim=1)
        return 100.0 * loss.mean()
    
    def get_goal_prediction_loss(self, goal_reg, goal_FDE, gt, traj_data):
        mask = gt[..., -1]
        criterion = nn.SmoothL1Loss(beta=1.0)
        
        reg_loss = (torch.norm((goal_reg[:, :, :2].permute(1, 0, 2) - gt[:, -1, :2].unsqueeze(1)), 2, dim=-1) * mask[:, -1:])
        
        goal_reg_detached = goal_reg.detach()
        FDE_gt = (torch.norm(goal_reg_detached[:, :, :2].permute(1, 0, 2) - gt[:, -1, :2].unsqueeze(1), 2, dim=-1) * mask[:, -1:])
        disp_loss = criterion(goal_FDE, FDE_gt)
        
        if self.traj_type_weights:
            type_weight = self.TRAJ_TYPE_WEIGHTS[traj_data['trajectory_type']].unsqueeze(1) # size(B, 1)
            reg_loss = (reg_loss * type_weight).min(dim=1)[0].mean()
            disp_loss = (disp_loss * type_weight).mean()
        else:
            reg_loss = reg_loss.min(dim=1)[0].mean()
            disp_loss = disp_loss.mean()
        
        return self.config['goal_reg_loss_weight'] * reg_loss + disp_loss

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

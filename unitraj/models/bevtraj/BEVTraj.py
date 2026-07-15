import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf

from unitraj.models.bevtraj.loss_utils import Criterion
from unitraj.models.bevtraj.pre_encoder import BEVTrajPreEncoder
from unitraj.models.bevtraj.scene_context_encoder import BEVTrajSceneContextEncoder
from unitraj.models.bevtraj.decoder import BEVTrajDecoder
from unitraj.models.bevtraj.custom_lr_sched import WarmupCosLR
try:
    from unitraj.models.base_model import BaseModel
except:
    from unitraj.models.bevtraj.base_model_local import BaseModel


def batch_nms(pred_trajs, pred_scores, dist_thresh, num_ret_modes=6):
    """

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 7)
        pred_scores (batch_size, num_modes):
        dist_thresh (float):
        num_ret_modes (int, optional): Defaults to 6.

    Returns:
        ret_trajs (batch_size, num_ret_modes, num_timestamps, 5)
        ret_scores (batch_size, num_ret_modes)
        ret_idxs (batch_size, num_ret_modes)
    """
    batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape

    sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
    bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes)
    sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
    sorted_pred_trajs = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)
    sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]  # (batch_size, num_modes, 7)

    dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
    point_cover_mask = (dist < dist_thresh)

    point_val = sorted_pred_scores.clone()  # (batch_size, N)
    point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)

    ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
    ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim)
    ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)
    bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

    for k in range(num_ret_modes):
        cur_idx = point_val.argmax(dim=-1)  # (batch_size)
        ret_idxs[:, k] = cur_idx

        new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
        point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
        point_val_selected[bs_idxs, cur_idx] = -1
        point_val += point_val_selected

        ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
        ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

    bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_ret_modes)

    ret_idxs = sorted_idxs[bs_idxs, ret_idxs]
    return ret_trajs, ret_scores, ret_idxs


class BEVTraj(BaseModel):
    def __init__(self, config):
        super(BEVTraj, self).__init__(config)
        
        self.config = OmegaConf.to_container(config, resolve=True)
        self.t = config['past_len']
        self.optimizer_cfg = self.config['optimizer']
        self.scheduler_cfg = self.config['scheduler']
        
        bev_feat_dim = sum(config['SENSOR_ENCODER']['decoder']['neck']['out_channels'])
        sc_feat_dim = config['SCENE_CONTEXT_ENCODER']['d_model']
        dec_dim = config['DECODER']['d_model']
        
        self.pre_encoder = BEVTrajPreEncoder(self.config['PRE_ENCODER'])
        self.scene_context_encoder = BEVTrajSceneContextEncoder(
                        self.config['SCENE_CONTEXT_ENCODER'], config['PRE_ENCODER']['d_model'], bev_feat_dim)
        self.decoder = BEVTrajDecoder(self.config['DECODER'])
        self.criterion = Criterion(self.config['loss'])
        
        self.bev_feat_down = nn.Sequential(
                nn.Conv2d(bev_feat_dim, dec_dim, kernel_size=1),
                nn.GroupNorm(num_groups=8, num_channels=dec_dim),
                nn.ReLU()
            ) if dec_dim != bev_feat_dim else nn.Identity()
        self.sc_feat_down = nn.Sequential(
                nn.Linear(sc_feat_dim, dec_dim),
                nn.LayerNorm(dec_dim),
                nn.ReLU()
            ) if dec_dim != sc_feat_dim else nn.Identity()
        
        print("BEVTraj model initialized.")
        
    def forward(self, batch, is_validation):
        traj_data = batch['traj_data']['input_dict']
        # bev_feat_raw = batch['bev_feat']
        ec_dynamics, tc_dynamics, ego_dynamics = self.prepare_decoder_input(traj_data)
        
        # encoding
        pre_encoder_emb = self.pre_encoder(traj_data)
        
        # bev_feature, seg_loss = self.sensor_encoder.get_bev_feature(sensor_data['batch_input_dict'], sensor_data['data_samples'])
        # bev_feature = self.sensor_encoder.get_bev_feature(bev_feat_raw)[0]
        B = pre_encoder_emb.shape[0]
        bev_feature = torch.randn(B, 512, 128, 128, device=pre_encoder_emb.device)

        scene_context_feature, dense_future_pred = self.scene_context_encoder(traj_data, pre_encoder_emb, bev_feature, ego_dynamics)
        
        # decoding
        bev_feature = self.bev_feat_down(bev_feature)
        scene_context_feature = self.sc_feat_down(scene_context_feature)
        obj_valid_mask = traj_data['obj_trajs_mask'].sum(dim=-1) > 0
        output = self.decoder(
            scene_context_feature,
            bev_feature,
            ec_dynamics,
            tc_dynamics,
            ego_dynamics,
            dense_future_pred=dense_future_pred,
            obj_valid_mask=obj_valid_mask,
            target_idx=traj_data['track_index_to_predict'],
        )
        
        # get loss
        output['dense_future_pred'] = dense_future_pred
        loss = self.get_loss(traj_data, output)
        
        last_logit = output['predicted_probability'][-1]
        last_prob = F.softmax(last_logit, dim=-1)
        last_traj = output['predicted_trajectory'][-1].permute(2, 0, 1, 3)

        anchor_pos = output['anchor_pos']
        goal_position = anchor_pos.permute(1, 0, 2).contiguous()
        if is_validation:
            last_traj, last_prob, ret_idxs = batch_nms(last_traj, last_prob, dist_thresh=2.5, num_ret_modes=10)
            batch_idx = torch.arange(B, device=ret_idxs.device)[:, None]
            goal_position = anchor_pos[batch_idx, ret_idxs].permute(1, 0, 2).contiguous()
        
        prediction = {'predicted_probability': last_prob,
                      'predicted_trajectory': last_traj,
                      'dense_future_pred': dense_future_pred,
                      'goal_position': goal_position}
        
        return prediction, loss

    
    def get_loss(self, traj_data, prediction):
        ground_truth = []
        decoder_gt = torch.cat(
            [traj_data['center_gt_trajs'], traj_data['center_gt_trajs_mask'].unsqueeze(-1)],
            dim=-1
        )
        ground_truth.append(decoder_gt)
        dense_future_gt = {'obj_trajs_future_state': traj_data['obj_trajs_future_state'], 'obj_trajs_future_mask': traj_data['obj_trajs_future_mask']}
        ground_truth.append(dense_future_gt)
        loss = self.criterion(prediction, ground_truth, traj_data['center_gt_final_valid_idx'], traj_data)
        
        return loss
    
    def prepare_decoder_input(self, traj_data):
        agents_in = traj_data['obj_trajs'] # (B, N, t, _)
        B_idx = torch.arange(agents_in.size(0), device=agents_in.device)
        target_idx = traj_data['track_index_to_predict']
        ego_idx = traj_data['ego_index']
        
        # ego-vehicle centric target agent dynamics
        ego_loc = agents_in[B_idx, ego_idx, -1:, :2].repeat(1, self.t, 1) # (B, t, 2)
        ego_sin, ego_cos = agents_in[B_idx, ego_idx, -1:, -6:-5].repeat(1, self.t, 1), agents_in[B_idx, ego_idx, -1:, -5:-4].repeat(1, self.t, 1) # (B, t, 1)
        
        rotation_matrix = torch.stack([
            torch.cat([ego_cos, -ego_sin], dim=-1),
            torch.cat([ego_sin, ego_cos], dim=-1)
        ], dim=-2)
        
        target_loc = agents_in[B_idx, target_idx, :, :2] # (B, t, 2)
        target_vel = agents_in[B_idx, target_idx, :, -4:-2] # (B, t, 2)
        target_acc = agents_in[B_idx, target_idx, :, -2:] # (B, t, 2)
        target_size = agents_in[B_idx, target_idx, :, 3:6] # (B, t, 3)
        
        target_loc = target_loc - ego_loc
        target_loc, target_vel, target_acc = map(lambda x: torch.matmul(x.unsqueeze(-2), rotation_matrix).squeeze(-2), \
            (target_loc, target_vel, target_acc))
        
        ego_centric_dynamics = torch.cat([target_loc, target_vel, target_acc, target_size], dim=-1) # (B, t, 9)
        
        # (target_agnet-centric) target agent dynamics
        tc_indices = [0, 1, -4, -3, -2, -1, 3, 4, 5]
        target_agent_dynamics = agents_in[B_idx, target_idx, ...][..., tc_indices] # (B, t, 9)
        
        # ego-vehicle dynamics
        ego_dynamics = {
            'ego_x': agents_in[B_idx, ego_idx, -1, 0:1], # (B, 1)
            'ego_y': agents_in[B_idx, ego_idx, -1, 1:2], # (B, 1)
            'ego_sin': agents_in[B_idx, ego_idx, -1, -6:-5], # (B, 1)
            'ego_cos': agents_in[B_idx, ego_idx, -1, -5:-4], # (B, 1)
        }
        
        return ego_centric_dynamics, target_agent_dynamics, ego_dynamics
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_cfg)
        scheduler = WarmupCosLR(optimizer, **self.scheduler_cfg)
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch, is_validation=False)
        self.log_info(batch['traj_data'], batch_idx, prediction, status='train')
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch, is_validation=True)
        self.log_info(batch['traj_data'], batch_idx, prediction, status='val')
        return loss

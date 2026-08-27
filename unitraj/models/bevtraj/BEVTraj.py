import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf

from unitraj.models.bevtraj.bevfusion import BEVFusion
from unitraj.models.bevtraj.loss_utils import Criterion
from unitraj.models.bevtraj.pre_encoder import BEVTrajPreEncoder
from unitraj.models.bevtraj.scene_context_encoder import BEVTrajSceneContextEncoder
from unitraj.models.bevtraj.decoder import BEVTrajDecoder
from unitraj.models.bevtraj.custom_lr_sched import WarmupCosLR
from unitraj.models.base_model import BaseModel
from unitraj.models.bevtraj.utility import batch_nms


class BEVTraj(BaseModel):
    def __init__(self, config):
        super(BEVTraj, self).__init__(config)
        
        self.config = OmegaConf.to_container(config, resolve=True)
        self.optimizer_cfg = self.config['optimizer']
        self.scheduler_cfg = self.config['scheduler']
        
        bev_feat_dim = sum(config['SENSOR_ENCODER']['decoder']['neck']['out_channels'])
        sc_feat_dim = config['SCENE_CONTEXT_ENCODER']['d_model']
        dec_dim = config['DECODER']['d_model']
        
        self.pre_encoder = BEVTrajPreEncoder(self.config['PRE_ENCODER'])
        self.sensor_encoder = BEVFusion(**self.config['SENSOR_ENCODER'])
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
        
    def forward(self, batch):
        traj_data = batch['traj_data']['input_dict']
        sensor_data = batch['sensor_data']
        tc_dynamics, ego_dynamics, agent_history = self.prepare_decoder_input(traj_data)
        
        # encoding
        pre_encoder_emb = self.pre_encoder(traj_data)
        bev_feature = self.sensor_encoder.get_bev_feature(sensor_data['batch_input_dict'], sensor_data['data_samples'])
        scene_context_feature, dense_future_pred = self.scene_context_encoder(traj_data, pre_encoder_emb, bev_feature, ego_dynamics)
        
        # decoding
        bev_feature = self.bev_feat_down(bev_feature)
        scene_context_feature = self.sc_feat_down(scene_context_feature)
        obj_valid_mask = traj_data['obj_trajs_mask'].sum(dim=-1) > 0
        output = self.decoder(
            scene_context_feature,
            bev_feature,
            tc_dynamics,
            ego_dynamics,
            dense_future_pred=dense_future_pred,
            obj_valid_mask=obj_valid_mask,
            target_idx=traj_data['track_index_to_predict'],
            agent_history=agent_history,
        )
        
        # get loss
        output['dense_future_pred'] = dense_future_pred
        loss = self.get_loss(traj_data, output)
        
        last_logit = output['predicted_probability'][-1]
        last_prob = F.softmax(last_logit, dim=-1)
        initial_traj = output['predicted_trajectory'][0].permute(2, 0, 1, 3)
        last_traj = output['predicted_trajectory'][-1].permute(2, 0, 1, 3)

        predicted_goal_position = output['predicted_goal_position']
        last_traj, last_prob, ret_idxs = batch_nms(last_traj, last_prob, dist_thresh=2.5, num_ret_modes=10)
        batch_idx = torch.arange(last_traj.size(0), device=ret_idxs.device)[:, None]
        initial_traj = initial_traj[batch_idx, ret_idxs]
        goal_position = predicted_goal_position[
            batch_idx, ret_idxs
        ].permute(1, 0, 2).contiguous()
        
        prediction = {'predicted_probability': last_prob,
                      'initial_predicted_trajectory': initial_traj,
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
        loss = self.criterion(prediction, ground_truth, traj_data['center_gt_final_valid_idx'])
        
        return loss
    
    def prepare_decoder_input(self, traj_data):
        agents_in = traj_data['obj_trajs'] # (B, N, t, _)
        B_idx = torch.arange(agents_in.size(0), device=agents_in.device)
        target_idx = traj_data['track_index_to_predict']
        ego_idx = traj_data['ego_index']
        
        # (target_agent-centric) target agent dynamics
        tc_indices = [0, 1, -4, -3, -2, -1, 3, 4, 5]
        target_agent_dynamics = agents_in[B_idx, target_idx, ...][..., tc_indices] # (B, t, 9)
        
        # ego-vehicle dynamics
        ego_dynamics = {
            'ego_x': agents_in[B_idx, ego_idx, -1, 0:1], # (B, 1)
            'ego_y': agents_in[B_idx, ego_idx, -1, 1:2], # (B, 1)
            'ego_sin': agents_in[B_idx, ego_idx, -1, -6:-5], # (B, 1)
            'ego_cos': agents_in[B_idx, ego_idx, -1, -5:-4], # (B, 1)
        }

        # Agent positions are already expressed in the target-agent frame.
        agent_history = {
            'positions': traj_data['obj_trajs_pos'][..., :2],  # (B, N, t, 2)
            'valid_mask': traj_data['obj_trajs_mask'].bool(),  # (B, N, t)
        }
        
        return (
            target_agent_dynamics,
            ego_dynamics,
            agent_history,
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_cfg)
        scheduler = WarmupCosLR(optimizer, **self.scheduler_cfg)
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.log_info(batch['traj_data'], batch_idx, prediction, status='train')
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.log_info(batch['traj_data'], batch_idx, prediction, status='val')
        return loss

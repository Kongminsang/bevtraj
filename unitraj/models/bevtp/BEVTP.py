import torch
import torch.nn as nn

from omegaconf import OmegaConf

from unitraj.models.bevtp.bevfusion import BEVFusion
from unitraj.models.bevtp.loss_utils import Criterion
from unitraj.models.bevtp.pre_encoder import BEVTPPreEncoder
from unitraj.models.bevtp.scene_context_encoder import BEVTPSceneContextEncoder
from unitraj.models.bevtp.decoder import BEVTPDecoder
from unitraj.models.bevtp.custom_lr_sched import WarmupCosLR
from unitraj.models.base_model import BaseModel


class BEVTP(BaseModel):
    def __init__(self, config):
        super(BEVTP, self).__init__(config)
        
        self.config = OmegaConf.to_container(config, resolve=True)
        self.t = config['past_len']
        self.optimizer_cfg = self.config['optimizer']
        self.scheduler_cfg = self.config['scheduler']
        
        bev_feat_dim = sum(config['SENSOR_ENCODER']['decoder']['neck']['out_channels'])
        sc_feat_dim = config['SCENE_CONTEXT_ENCODER']['d_model']
        dec_dim = config['DECODER']['d_model']
        
        self.pre_encoder = BEVTPPreEncoder(self.config['PRE_ENCODER']) 
        self.sensor_encoder = BEVFusion(**self.config['SENSOR_ENCODER'])
        self.scene_context_encoder = BEVTPSceneContextEncoder(
                        self.config['SCENE_CONTEXT_ENCODER'], config['PRE_ENCODER']['d_model'], bev_feat_dim)
        self.decoder = BEVTPDecoder(self.config['DECODER'])
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
        
        print("BEVTP model initialized.")
        
    def forward(self, batch):
        traj_data = batch['traj_data']['input_dict']
        sensor_data = batch['sensor_data']
        ec_dynamics, tc_dynamics, ego_dynamics = self.prepare_decoder_input(traj_data)
        
        # encoding
        pre_encoder_emb = self.pre_encoder(traj_data)
        bev_feature, seg_loss = self.sensor_encoder.get_bev_feature(sensor_data['batch_input_dict'], sensor_data['data_samples'])
        scene_context_feature, dense_future_pred = self.scene_context_encoder(traj_data, pre_encoder_emb, bev_feature)
        
        # decoding
        bev_feature = self.bev_feat_down(bev_feature)
        scene_context_feature = self.sc_feat_down(scene_context_feature)
        output = self.decoder(scene_context_feature, bev_feature, ec_dynamics, tc_dynamics, ego_dynamics)
        
        # get loss
        output['dense_future_pred'] = dense_future_pred
        loss = self.get_loss(traj_data, output)
        
        last_prop = output['predicted_probability'][-1]
        last_traj = output['predicted_trajectory'][-1].permute(2, 0, 1, 3)
        prediction = {'predicted_probability': last_prop, 'predicted_trajectory': last_traj, 'dense_future_pred': dense_future_pred}
        
        return prediction, loss + seg_loss

    
    def get_loss(self, traj_data, prediction):
        ground_truth = []
        decoder_gt = torch.cat([traj_data['center_gt_trajs'][..., :2], traj_data['center_gt_trajs_mask'].unsqueeze(-1)],
                                 dim=-1)
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
            'ego_loc': agents_in[B_idx, ego_idx, -1, :2], # (B, 2)
            'ego_sin': agents_in[B_idx, ego_idx, -1, -6:-5], # (B, 1)
            'ego_cos': agents_in[B_idx, ego_idx, -1, -5:-4], # (B, 1)
        }
        
        return ego_centric_dynamics, target_agent_dynamics, ego_dynamics
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_cfg)
        scheduler = WarmupCosLR(optimizer, **self.scheduler_cfg)
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.compute_official_evaluation(batch, prediction)
        self.log_info(batch['traj_data'], batch_idx, prediction, status='train')
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, loss = self.forward(batch)
        self.compute_official_evaluation(batch, prediction)
        self.log_info(batch['traj_data'], batch_idx, prediction, status='val')
        return loss
import math
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.bevtraj.bev_deformable_aggregation import BDA_ENC
from unitraj.models.bevtraj.decoder_deform_attn import BEVDeformCrossAttn
from unitraj.models.bevtraj.linear import FFN, GatedFusion, MLP, MotionRegHead, MotionVelHead
from unitraj.models.bevtraj.utility import ego_to_target, target_to_ego


ASSET_DIR = Path(__file__).resolve().parent


class BEVTrajSceneContextEncoder(nn.Module):
    def __init__(self, config, pre_enc_dim, bev_feat_dim):
        super().__init__()
        self.config = config
        self.D = config['d_model']
        self.T = config['future_len']
        self.num_dense_objects = config.get('num_dense_objects', 8)
        self.dense_goal_temperature = config.get('dense_goal_temperature', 0.1)
        self.dense_refinement_smoothing_sigma = config.get('dense_refinement_smoothing_sigma', 1.5)

        self.agent_proj = MLP(pre_enc_dim, self.D, self.D, 2)
        self.bev_feat_down = nn.Sequential(
            nn.Conv2d(bev_feat_dim, self.D, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=self.D),
            nn.ReLU()
        ) if self.D != bev_feat_dim else nn.Identity()
        self.bda = BDA_ENC(config['bda_enc'], d_model=self.D)

        self.dense_anchor_encoder = MLP(2, self.D, self.D, 2)
        self.dense_anchor_attn = nn.MultiheadAttention(
            self.D, config['num_attn_head'], dropout=config.get('dropout_of_attn', 0.1), batch_first=True
        )
        self.dense_anchor_norm = nn.LayerNorm(self.D)
        self.dense_agent_q = nn.Linear(self.D, self.D, bias=False)
        self.dense_anchor_k = nn.Linear(self.D, self.D, bias=False)
        self.dense_bda_score = MLP(self.D, self.D, 1, 2)

        trajectory_path = ASSET_DIR / config['dense_trajectory_file_name']
        with open(trajectory_path, 'rb') as f:
            trajectory_set = pickle.load(f)['VEHICLE']
        self.register_buffer('dense_trajectory_set', torch.from_numpy(trajectory_set).float(), persistent=False)

        dense_dca_config = dict(config['dense_deform_cross_attn'])
        dense_dca_config['dim'] = self.D
        self.dense_traj_encoder = MLP(4, self.D, self.D, 2)
        self.dense_query_scale = MLP(self.D, self.D, self.D, 2)
        self.dense_bev_cross_attn = BEVDeformCrossAttn(**dense_dca_config)
        self.dense_fusion = GatedFusion(self.D)
        self.dense_motion_reg = MotionRegHead(self.D)
        self.dense_motion_vel = MotionVelHead(self.D)

        self.dense_future_encoder = MLP(4, self.D, self.D, 2)
        self.dense_temporal_attn = nn.MultiheadAttention(
            self.D, config['num_attn_head'], dropout=config.get('dropout_of_attn', 0.1), batch_first=True
        )
        self.dense_social_attn = nn.MultiheadAttention(
            self.D, config['num_attn_head'], dropout=config.get('dropout_of_attn', 0.1), batch_first=True
        )
        self.dense_future_norm = nn.ModuleList([nn.LayerNorm(self.D) for _ in range(2)])
        self.dense_future_ffn = FFN(self.D, self.D * 2, dropout=config.get('dropout_of_attn', 0.1))

        radius = config.get('dense_refinement_smoothing_kernel_size', 7) // 2
        self.dense_refinement_smoothing_kernel_size = radius * 2 + 1
        self.register_buffer('dense_smoothing_steps', torch.arange(-radius, radius + 1).float(), persistent=False)

    @staticmethod
    def place_trajectory(trajectory, obj_pos, obj_heading):
        sin, cos = obj_heading.unbind(dim=-1)
        rotation = torch.stack([
            torch.stack([cos, sin], dim=-1),
            torch.stack([-sin, cos], dim=-1)
        ], dim=-2)
        return torch.matmul(trajectory, rotation) + obj_pos.unsqueeze(-2)

    def smooth_dense_offset(self, offset):
        B, num_objects, T, _ = offset.shape
        channels = offset.permute(0, 1, 3, 2).reshape(B * num_objects, 2, T)
        radius = self.dense_refinement_smoothing_kernel_size // 2
        windows = F.pad(channels, (radius, radius), mode='replicate').unfold(
            -1, self.dense_refinement_smoothing_kernel_size, 1
        )
        steps = self.dense_smoothing_steps.to(offset.dtype)
        weights = torch.exp(-0.5 * (steps / self.dense_refinement_smoothing_sigma).square())
        weights = weights / weights.sum()
        smoothed = (windows * weights).sum(dim=-1)
        return smoothed.reshape(B, num_objects, 2, T).permute(0, 1, 3, 2).contiguous()

    def encode_dense_future(self, prediction, agent_feature, obj_valid_mask):
        B, num_objects, T, _ = prediction.shape
        feature = self.dense_future_encoder(torch.cat([prediction[..., :2], prediction[..., -2:]], dim=-1))
        feature = feature + agent_feature.unsqueeze(2)
        temporal = feature.reshape(B * num_objects, T, self.D)
        temporal = self.dense_future_norm[0](
            self.dense_temporal_attn(temporal, temporal, temporal, need_weights=False)[0] + temporal
        ).reshape(B, num_objects, T, self.D)
        social = temporal.permute(0, 2, 1, 3).reshape(B * T, num_objects, self.D)
        social_mask = ~obj_valid_mask[:, None].expand(-1, T, -1).reshape(B * T, num_objects)
        social = self.dense_future_norm[1](
            self.dense_social_attn(social, social, social, key_padding_mask=social_mask, need_weights=False)[0] + social
        ).reshape(B, T, num_objects, self.D).permute(0, 2, 1, 3).contiguous()
        return self.dense_future_ffn(social) * obj_valid_mask[:, :, None, None]

    def predict_dense_future(self, traj_data, agent_feature, bda_feature, ref_pos_target, bev_feature, ego_dyn):
        B, num_objects, R, _ = bda_feature.shape
        anchor_feature = self.dense_anchor_encoder(ref_pos_target).reshape(B * num_objects, R, self.D)
        anchor_feature = self.dense_anchor_norm(
            self.dense_anchor_attn(anchor_feature, anchor_feature, anchor_feature, need_weights=False)[0]
            + anchor_feature
        ).reshape(B, num_objects, R, self.D)
        geometric_logits = torch.einsum(
            'bod,bord->bor', self.dense_agent_q(agent_feature), self.dense_anchor_k(anchor_feature)
        ) / math.sqrt(self.D)
        raster_logits = self.dense_bda_score(bda_feature).squeeze(-1)
        anchor_weight = ((geometric_logits.float() + raster_logits.float()) / self.dense_goal_temperature).softmax(dim=-1)
        predicted_goal = (anchor_weight.unsqueeze(-1) * ref_pos_target).sum(dim=2)

        obj_pos = traj_data['obj_trajs'][:, :num_objects, -1, :2]
        obj_heading = traj_data['obj_trajs'][:, :num_objects, -1, -6:-4]
        candidate_endpoints = self.place_trajectory(
            self.dense_trajectory_set[:, -1][None, None].expand(B, num_objects, -1, -1), obj_pos, obj_heading
        )
        trajectory_idx = (candidate_endpoints - predicted_goal.unsqueeze(2)).square().sum(dim=-1).argmin(dim=-1)
        trajectory = self.place_trajectory(self.dense_trajectory_set[trajectory_idx], obj_pos, obj_heading)
        velocity = torch.diff(trajectory, dim=2, prepend=obj_pos.unsqueeze(2)) / self.config.get('dt', 0.1)
        vector_feature = self.dense_traj_encoder(torch.cat([trajectory, velocity], dim=-1)) + agent_feature.unsqueeze(2)

        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'], ego_dyn['ego_y'], ego_dyn['ego_sin'], ego_dyn['ego_cos']
        )
        ref_points = target_to_ego(
            trajectory.reshape(B, num_objects * self.T, 2), trans_x, trans_y, rot_sin, rot_cos
        ).reshape(B, num_objects, self.T, 2).permute(1, 0, 2, 3)
        vector_feature = vector_feature.permute(1, 0, 2, 3)
        raster_feature = self.dense_bev_cross_attn(
            vector_feature, bev_feature, self.dense_query_scale(vector_feature), ref_points,
            identity=torch.zeros_like(vector_feature)
        )
        fused_feature = self.dense_fusion(raster_feature, vector_feature)
        distribution = self.dense_motion_reg(fused_feature).permute(1, 0, 2, 3)
        pred_xy = trajectory + self.smooth_dense_offset(distribution[..., :2])
        pred_vel = self.dense_motion_vel(fused_feature).permute(1, 0, 2, 3)
        return torch.cat([pred_xy, distribution[..., 2:], pred_vel], dim=-1), predicted_goal

    def forward(self, traj_data, pre_encoder_feature, bev_feature, ego_dyn):
        agent_feature = self.agent_proj(pre_encoder_feature)
        bev_feature = self.bev_feat_down(bev_feature)
        bda_feature, ref_pos_ego = self.bda(traj_data, bev_feature, ego_dyn)
        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'], ego_dyn['ego_y'], ego_dyn['ego_sin'], ego_dyn['ego_cos']
        )
        ref_pos_target = ego_to_target(ref_pos_ego, trans_x, trans_y, rot_sin, rot_cos)
        num_objects = self.num_dense_objects
        num_anchors = bda_feature.size(1) // num_objects
        bda_feature = bda_feature.reshape(bda_feature.size(0), num_objects, num_anchors, self.D)
        ref_pos_target = ref_pos_target.reshape(ref_pos_target.size(0), num_objects, num_anchors, 2)
        dense_agent_feature = agent_feature[:, :num_objects]
        obj_valid_mask = traj_data['obj_trajs_mask'][:, :num_objects].any(dim=-1)
        dense_future_pred, dense_future_goal = self.predict_dense_future(
            traj_data, dense_agent_feature, bda_feature, ref_pos_target, bev_feature, ego_dyn
        )
        dense_future_pred = dense_future_pred * obj_valid_mask[:, :, None, None]
        dense_future_goal = dense_future_goal * obj_valid_mask[:, :, None]
        dense_future_feature = self.encode_dense_future(dense_future_pred, dense_agent_feature, obj_valid_mask)
        return agent_feature, dense_future_feature, dense_future_pred, dense_future_goal

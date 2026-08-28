import copy
import math
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.bevtraj.bev_deformable_aggregation import BDA_DEC
from unitraj.models.bevtraj.decoder_deform_attn import BEVDeformCrossAttn
from unitraj.models.bevtraj.linear import FFN, GatedFusion, MLP, MotionRegHead, MotionVelHead
from unitraj.models.bevtraj.temporal_sequential_module import TemporalMHA, TemporalMHA_NoTimePE
from unitraj.models.bevtraj.utility import gen_sineembed_for_position, target_to_ego


ASSET_DIR = Path(__file__).resolve().parent / 'assets'


class QueryConditionedDynamics(nn.Module):
    def __init__(self, query_dim, hidden_dim):
        super().__init__()
        self.modulator = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

    def forward(self, dynamics, query_emb):
        gamma, beta = self.modulator(query_emb).chunk(2, dim=-1)
        return (1 + gamma) * dynamics + beta


class BEVTrajDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.T = config['future_len']
        self.D = config['d_model']
        self.Q_D = config['query_dims']
        self.num_heads = config['num_heads']
        self.spa_pos_T = config['spa_pos_T']
        self.to_pos_Q = MLP(self.Q_D, self.Q_D, self.D, 2)
        self.norm = nn.ModuleList([nn.LayerNorm(self.D) for _ in range(3)])
        self.temp_self_attn = TemporalMHA(self.D, self.num_heads, config['dropout'])
        self.bev_cross_attn = BEVDeformCrossAttn(**config['deform_cross_attn'])
        self.hybrid_self_attn = nn.MultiheadAttention(self.D, self.num_heads, dropout=config['dropout'])
        self.ffn = FFN(self.D, config['ffn_dims'], 2, config['dropout'])

    def forward(self, mode_embed, bev_feat, query_scale, ref_points, ego_dyn, time_pe):
        M, B, T, D = mode_embed.shape
        temporal = mode_embed.permute(2, 1, 0, 3).reshape(T, B * M, D)
        temporal = self.norm[0](self.temp_self_attn(temporal, time_pe) + temporal)
        mode_embed = temporal.reshape(T, B, M, D).permute(2, 1, 0, 3).contiguous()
        mode_embed = mode_embed + self.to_pos_Q(
            gen_sineembed_for_position(ref_points, hidden_dim=self.Q_D, temperature=self.spa_pos_T)
        )
        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'], ego_dyn['ego_y'], ego_dyn['ego_sin'], ego_dyn['ego_cos']
        )
        ref_points_ego = target_to_ego(
            ref_points.permute(1, 0, 2, 3).reshape(B, M * T, 2),
            trans_x, trans_y, rot_sin, rot_cos
        ).reshape(B, M, T, 2).permute(1, 0, 2, 3)
        mode_embed = self.norm[1](
            self.bev_cross_attn(mode_embed, bev_feat, query_scale, ref_points_ego)
        )
        hybrid = mode_embed.permute(0, 2, 1, 3).reshape(M * T, B, D)
        hybrid = self.norm[2](
            self.hybrid_self_attn(hybrid, hybrid, hybrid, need_weights=False)[0] + hybrid
        )
        mode_embed = hybrid.reshape(M, T, B, D).permute(0, 2, 1, 3).contiguous()
        return self.ffn(mode_embed)


class TemporalModeCompressor(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.query = MLP(dim * 2, dim, dim, 2)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        M, B, T, D = tokens.shape
        sequence = tokens.permute(2, 1, 0, 3).reshape(T, B * M, D)
        query = self.query(torch.cat([tokens.mean(dim=2), tokens[:, :, -1]], dim=-1))
        pooled = self.attn(
            query.permute(1, 0, 2).reshape(1, B * M, D), sequence, sequence, need_weights=False
        )[0]
        return self.norm(pooled.reshape(B, M, D).permute(1, 0, 2) + query)


class BEVTrajDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.T = config['future_len']
        self.D = config['d_model']
        self.K = config['num_modes']
        self.num_heads = config['num_heads']
        self.query_scale_dims = config['query_scale_dims']
        self.spa_pos_T = config['spa_pos_T']
        self.dropout = config['dropout']
        self.L_dec = config['num_decoder_layers']
        self.grid_size = config['grid_size']
        self.goal_attn_temperature = config.get('goal_attn_temperature', 0.1)
        self.refinement_smoothing_kernel_size = config.get('refinement_smoothing_kernel_size', 7)
        self.refinement_smoothing_sigma = config.get('refinement_smoothing_sigma', 1.5)
        self.dt = config.get('dt', 0.1)

        self.dca_cfg = dict(config['deform_cross_attn'])
        self.dca_cfg['dim'] = self.D
        self.dca_itr_cfg = dict(config['deform_cross_attn_itr'])
        self.dca_itr_cfg['dim'] = self.D
        decoder_layer_config = {
            'future_len': self.T,
            'd_model': self.D,
            'query_dims': self.query_scale_dims,
            'ffn_dims': config['ffn_dims'],
            'spa_pos_T': self.spa_pos_T,
            'dropout': self.dropout,
            'num_heads': self.num_heads,
            'deform_cross_attn': self.dca_itr_cfg
        }

        self.bda_sgcp = BDA_DEC(config['bda_dec'], self.D)
        with open(ASSET_DIR / config['goal_anchor_file_name'], 'rb') as f:
            goal_anchor_data = pickle.load(f)['VEHICLE']
        goal_anchor_indices = torch.as_tensor(goal_anchor_data['anchor_indices'], dtype=torch.long)
        goal_attn_mask = torch.ones(self.K, self.bda_sgcp.anchors.size(0), dtype=torch.bool)
        goal_attn_mask.scatter_(1, goal_anchor_indices, False)
        self.register_buffer('goal_attn_mask', goal_attn_mask, persistent=False)
        self.goal_query = nn.Parameter(torch.empty(self.K, 1, self.D))
        nn.init.xavier_uniform_(self.goal_query)
        self.goal_mode_attn = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)
        self.goal_mode_norm = nn.LayerNorm(self.D)
        self.goal_target_conditioner = QueryConditionedDynamics(self.D, self.D)
        self.goal_raster_q = nn.Linear(self.D, self.D, bias=False)
        self.goal_raster_k = nn.Linear(self.D, self.D, bias=False)
        self.goal_anchor_encoder = MLP(2, self.D, self.D, 2)
        self.goal_anchor_attn = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout, batch_first=True)
        self.goal_anchor_norm = nn.LayerNorm(self.D)
        self.goal_geometric_q = nn.Linear(self.D, self.D, bias=False)
        self.goal_geometric_k = nn.Linear(self.D, self.D, bias=False)
        self.goal_logit_gate = MLP(self.D * 2, self.D, 2, 2)
        self.goal_feature_fusion = GatedFusion(self.D)
        self.goal_FDE = MLP(self.D, self.D, 1, 2)

        with open(ASSET_DIR / config['trajectory_file_name'], 'rb') as f:
            trajectory_set = pickle.load(f)['VEHICLE']
        self.register_buffer('trajectory_set', torch.from_numpy(trajectory_set).float(), persistent=False)

        self.time_embedding_mlp = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, self.D))
        self.register_buffer('future_time', torch.arange(self.T).float().unsqueeze(-1), persistent=False)
        self.time_emb_alpha = nn.Parameter(torch.tensor(1.0))
        self.get_query_scale_itp = MLP(self.D, self.query_scale_dims, self.query_scale_dims, 2)
        self.initial_bev_cross_attn = BEVDeformCrossAttn(**self.dca_cfg)
        self.initial_bev_norm = nn.LayerNorm(self.D)
        self.initial_bev_ffn = FFN(self.D, config['ffn_dims'], 2, self.dropout)
        self.initial_fusion = GatedFusion(self.D)
        self.motion_reg_l1 = MotionRegHead(self.D)
        self.motion_vel_l1 = MotionVelHead(self.D)

        self.state_context_cross_attn = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)
        self.state_temporal_attn = TemporalMHA_NoTimePE(self.D, self.num_heads, self.dropout)
        self.state_norm = nn.ModuleList([nn.LayerNorm(self.D) for _ in range(2)])
        self.state_reg = MLP(self.D, self.D, 2, 2)

        self.trajectory_state_proj = MLP(6, self.D, self.D, 2)
        self.interaction_geometry_proj = MLP(6, self.D, self.D, 2)
        self.interaction_fusion = MLP(self.D * 3, self.D, self.D, 2)
        self.interaction_temporal_attn = TemporalMHA_NoTimePE(self.D, self.num_heads, self.dropout)
        self.interaction_norm = nn.LayerNorm(self.D)
        self.interaction_ffn = FFN(self.D, config['ffn_dims'], 2, self.dropout)

        self.score_bev_query_scale = MLP(self.D, self.query_scale_dims, self.query_scale_dims, 2)
        self.score_bev_cross_attn = BEVDeformCrossAttn(**self.dca_itr_cfg)
        self.score_bev_pool = TemporalModeCompressor(self.D, self.num_heads, self.dropout)
        self.score_interaction_pool = TemporalModeCompressor(self.D, self.num_heads, self.dropout)
        self.score_fusion = GatedFusion(self.D)
        self.mode_prob_head_l1 = MLP(self.D, self.D, 1, 2)
        self.mode_prob_head = MLP(self.D, self.D, 1, 2)

        decoder_layer = BEVTrajDecoderLayer(decoder_layer_config)
        self.dec_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.L_dec - 1)])
        self.refinement_fusions = nn.ModuleList([GatedFusion(self.D) for _ in range(self.L_dec - 1)])
        self.get_query_scale_itr = MLP(self.D, self.query_scale_dims, self.query_scale_dims, 2)
        self.motion_reg = MotionRegHead(self.D)
        self.motion_vel = MotionVelHead(self.D)
        self.refinement_smoothing_sigma_head = nn.Linear(self.D, 1)
        nn.init.zeros_(self.refinement_smoothing_sigma_head.weight)
        nn.init.constant_(
            self.refinement_smoothing_sigma_head.bias, math.log(math.expm1(self.refinement_smoothing_sigma))
        )
        radius = self.refinement_smoothing_kernel_size // 2
        self.register_buffer('refinement_smoothing_steps', torch.arange(-radius, radius + 1).float(), persistent=False)

    def build_time_pe(self, B, M, dtype):
        time = (self.future_time * self.dt + 0.1).to(dtype=dtype)
        embedding = self.time_emb_alpha * self.time_embedding_mlp(time)
        return embedding[:, None, None].expand(-1, B, M, -1).reshape(self.T, B * M, self.D)

    def smooth_refinement_offset(self, raw_offset, sigma):
        M, B, T, _ = raw_offset.shape
        channels = raw_offset.permute(0, 1, 3, 2).reshape(M * B, 2, T)
        radius = self.refinement_smoothing_kernel_size // 2
        windows = F.pad(channels, (radius, radius), mode='replicate').unfold(
            -1, self.refinement_smoothing_kernel_size, 1
        )
        steps = self.refinement_smoothing_steps.to(raw_offset.dtype)
        weights = torch.exp(-0.5 * (steps / sigma.unsqueeze(-1)).square())
        weights = weights / weights.sum(dim=-1, keepdim=True)
        smoothed = (windows * weights.reshape(M * B, 1, T, -1)).sum(dim=-1)
        return smoothed.reshape(M, B, 2, T).permute(0, 1, 3, 2).contiguous()

    def goal_candidate_proposal(self, bev_feat, agent_context, ego_dyn, target_idx):
        bda_token, bda_pos = self.bda_sgcp(bev_feat, ego_dyn)
        B, R, _ = bda_token.shape
        mode_query = self.goal_query.expand(-1, B, -1)
        mode_query = self.goal_mode_norm(
            self.goal_mode_attn(mode_query, mode_query, mode_query, need_weights=False)[0] + mode_query
        )
        batch_idx = torch.arange(B, device=bev_feat.device)
        target_feature = agent_context[batch_idx, target_idx.long()][None].expand(self.K, -1, -1)
        conditioned_target = self.goal_target_conditioner(target_feature, mode_query)

        raster_q = self.goal_raster_q(mode_query).permute(1, 0, 2)
        raster_k = self.goal_raster_k(bda_token)
        raster_pair = raster_q.unsqueeze(2) * raster_k.unsqueeze(1)
        raster_logits = raster_pair.sum(dim=-1) / math.sqrt(self.D)

        anchor_feature = self.goal_anchor_encoder(self.bda_sgcp.anchors)[None]
        anchor_feature = self.goal_anchor_norm(
            self.goal_anchor_attn(anchor_feature, anchor_feature, anchor_feature, need_weights=False)[0]
            + anchor_feature
        )[0]
        geometric_q = self.goal_geometric_q(conditioned_target).permute(1, 0, 2)
        geometric_k = self.goal_geometric_k(anchor_feature)
        geometric_pair = geometric_q.unsqueeze(2) * geometric_k[None, None]
        geometric_logits = geometric_pair.sum(dim=-1) / math.sqrt(self.D)

        gate = self.goal_logit_gate(torch.cat([raster_pair, geometric_pair], dim=-1)).softmax(dim=-1)
        goal_logits = gate[..., 0] * raster_logits + gate[..., 1] * geometric_logits
        goal_logits = (goal_logits.float() / self.goal_attn_temperature).masked_fill(
            self.goal_attn_mask[None], float('-inf')
        )
        goal_weight = goal_logits.softmax(dim=-1)
        goal_position = (bda_pos[:, None] * goal_weight.unsqueeze(-1)).sum(dim=2).permute(1, 0, 2)

        raster_mode_feature = torch.einsum('bkr,brd->kbd', goal_weight.to(bda_token.dtype), bda_token)
        goal_feature = self.goal_feature_fusion(raster_mode_feature, conditioned_target)
        goal_fde = self.goal_FDE(goal_feature).squeeze(-1).T
        return mode_query, goal_position, goal_fde

    def build_state_query(self, agent_context, agent_valid_mask):
        B = agent_context.size(0)
        time = (self.future_time * self.dt + 0.1).to(agent_context.dtype)
        state_query = self.time_emb_alpha * self.time_embedding_mlp(time)
        state_query = state_query[:, None].expand(-1, B, -1)
        memory = agent_context.permute(1, 0, 2)
        state_query = self.state_norm[0](
            self.state_context_cross_attn(
                state_query, memory, memory, key_padding_mask=~agent_valid_mask, need_weights=False
            )[0] + state_query
        )
        return self.state_norm[1](self.state_temporal_attn(state_query) + state_query)

    def build_interaction_tokens(
        self, pred_xy, pred_vel, state_query, dense_future_feature,
        dense_future_pred, dense_obj_valid_mask, target_idx
    ):
        M, B, T, _ = pred_xy.shape
        displacement = torch.diff(pred_xy, dim=2, prepend=torch.zeros_like(pred_xy[:, :, :1]))
        trajectory_feature = self.trajectory_state_proj(
            torch.cat([pred_xy, pred_vel, displacement], dim=-1)
        ) + state_query.permute(1, 0, 2).unsqueeze(0)

        obj_xy = dense_future_pred[..., :2].unsqueeze(1)
        obj_vel = dense_future_pred[..., -2:].unsqueeze(1)
        traj_xy = pred_xy.permute(1, 0, 2, 3).unsqueeze(2)
        traj_vel = pred_vel.permute(1, 0, 2, 3).unsqueeze(2)
        rel_xy = traj_xy - obj_xy
        distance = rel_xy.norm(dim=-1)
        rel_direction = rel_xy / distance.unsqueeze(-1).clamp_min(1e-3)
        closing_speed = -((traj_vel - obj_vel) * rel_direction).sum(dim=-1)
        obj_speed = obj_vel.norm(dim=-1)
        object_idx = torch.arange(dense_future_pred.size(1), device=pred_xy.device)
        valid_mask = dense_obj_valid_mask & (object_idx[None] != target_idx[:, None])
        valid_mask = valid_mask[:, None, :, None].expand(-1, M, -1, T)
        masked_distance = distance.masked_fill(~valid_mask, 1e4)
        weight = torch.softmax(-masked_distance, dim=2) * valid_mask
        weight = weight / weight.sum(dim=2, keepdim=True).clamp_min(1e-6)

        interaction_geometry = torch.cat([
            (weight.unsqueeze(-1) * rel_xy).sum(dim=2),
            (weight * distance).sum(dim=2).unsqueeze(-1),
            masked_distance.amin(dim=2).masked_fill(~valid_mask.any(dim=2), 0).unsqueeze(-1),
            (weight * closing_speed).sum(dim=2).unsqueeze(-1),
            (weight * obj_speed).sum(dim=2).unsqueeze(-1)
        ], dim=-1)
        interaction_geometry = self.interaction_geometry_proj(interaction_geometry).permute(1, 0, 2, 3)
        dense_feature = torch.einsum('bmot,botd->bmtd', weight, dense_future_feature).permute(1, 0, 2, 3)
        interaction = self.interaction_fusion(
            torch.cat([trajectory_feature, dense_feature, interaction_geometry], dim=-1)
        )
        sequence = interaction.permute(2, 1, 0, 3).reshape(T, B * M, self.D)
        sequence = self.interaction_norm(self.interaction_temporal_attn(sequence) + sequence)
        interaction = sequence.reshape(T, B, M, self.D).permute(2, 1, 0, 3).contiguous()
        return self.interaction_ffn(interaction)

    def sample_trajectory_bev(self, query, pred_xy, bev_feat, ego_dyn):
        M, B, T, _ = pred_xy.shape
        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'], ego_dyn['ego_y'], ego_dyn['ego_sin'], ego_dyn['ego_cos']
        )
        ref_points = target_to_ego(
            pred_xy.permute(1, 0, 2, 3).reshape(B, M * T, 2),
            trans_x, trans_y, rot_sin, rot_cos
        ).reshape(B, M, T, 2).permute(1, 0, 2, 3)
        return self.score_bev_cross_attn(
            query, bev_feat, self.score_bev_query_scale(query), ref_points, identity=torch.zeros_like(query)
        )

    def score_predicted_trajectory(
        self, mode_embed, state_query, pred_traj, pred_vel, bev_feat, ego_dyn,
        dense_future_feature, dense_future_pred, dense_obj_valid_mask, target_idx, score_head
    ):
        bev_score_feature = self.sample_trajectory_bev(mode_embed, pred_traj[..., :2], bev_feat, ego_dyn)
        interaction_feature = self.build_interaction_tokens(
            pred_traj[..., :2], pred_vel, state_query, dense_future_feature,
            dense_future_pred, dense_obj_valid_mask, target_idx
        )
        bev_mode = self.score_bev_pool(bev_score_feature)
        interaction_mode = self.score_interaction_pool(interaction_feature)
        return score_head(self.score_fusion(bev_mode, interaction_mode)).squeeze(-1).T

    def initial_prediction(
        self, mode_query, agent_context, agent_valid_mask, dense_future_feature,
        dense_future_pred, dense_obj_valid_mask, bev_feat, predicted_goal_position, ego_dyn, target_idx
    ):
        M, B, _ = mode_query.shape
        endpoint_distance = (
            predicted_goal_position[:, :, None] - self.trajectory_set[None, None, :, -1]
        ).square().sum(dim=-1)
        trajectory_idx = endpoint_distance.argmin(dim=-1)
        ref_trajectory = self.trajectory_set[trajectory_idx].permute(1, 0, 2, 3).contiguous()
        ref_velocity = torch.diff(
            ref_trajectory, dim=2, prepend=torch.zeros_like(ref_trajectory[:, :, :1])
        ) / self.dt

        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'], ego_dyn['ego_y'], ego_dyn['ego_sin'], ego_dyn['ego_cos']
        )
        ref_points_ego = target_to_ego(
            ref_trajectory.permute(1, 0, 2, 3).reshape(B, M * self.T, 2),
            trans_x, trans_y, rot_sin, rot_cos
        ).reshape(B, M, self.T, 2).permute(1, 0, 2, 3)
        time_pe = self.build_time_pe(B, M, mode_query.dtype)
        mode_embed = mode_query.unsqueeze(2) + time_pe.reshape(self.T, B, M, self.D).permute(2, 1, 0, 3)
        mode_embed = self.initial_bev_norm(
            self.initial_bev_cross_attn(
                mode_embed, bev_feat, self.get_query_scale_itp(mode_embed), ref_points_ego,
                identity=torch.zeros_like(mode_embed)
            )
        )
        mode_embed = self.initial_bev_ffn(mode_embed)

        state_query = self.build_state_query(agent_context, agent_valid_mask)
        interaction_embed = self.build_interaction_tokens(
            ref_trajectory, ref_velocity, state_query, dense_future_feature,
            dense_future_pred, dense_obj_valid_mask, target_idx
        )
        fused_feature = self.initial_fusion(mode_embed, interaction_embed)
        distribution = self.motion_reg_l1(fused_feature)
        pred_xy = ref_trajectory + distribution[..., :2]
        pred_traj = torch.cat([pred_xy, distribution[..., 2:]], dim=-1)
        pred_vel = self.motion_vel_l1(fused_feature)
        mode_prob = self.score_predicted_trajectory(
            mode_embed, state_query, pred_traj, pred_vel, bev_feat, ego_dyn,
            dense_future_feature, dense_future_pred, dense_obj_valid_mask, target_idx,
            self.mode_prob_head_l1
        )
        state_pred = self.state_reg(state_query).permute(1, 0, 2).contiguous()
        return mode_embed, interaction_embed, state_query, mode_prob, pred_traj, pred_vel, state_pred

    def forward(
        self, agent_context, dense_future_feature, bev_feat, ego_dyn,
        dense_future_pred, agent_valid_mask, dense_obj_valid_mask, target_idx
    ):
        B = bev_feat.size(0)
        mode_query, goal_position, goal_fde = self.goal_candidate_proposal(
            bev_feat, agent_context, ego_dyn, target_idx
        )
        predicted_goal_position = goal_position.permute(1, 0, 2).contiguous()
        mode_embed, interaction_embed, state_query, mode_prob, pred_traj, pred_vel, state_pred = self.initial_prediction(
            mode_query, agent_context, agent_valid_mask, dense_future_feature,
            dense_future_pred, dense_obj_valid_mask, bev_feat,
            predicted_goal_position.detach(), ego_dyn, target_idx
        )

        mode_probs = [mode_prob]
        pred_trajs = [pred_traj.permute(0, 2, 1, 3)]
        pred_vels = [pred_vel.permute(0, 2, 1, 3)]
        refinement_smoothing_sigmas = []
        ref_points = pred_traj[..., :2].detach()
        time_pe = self.build_time_pe(B, mode_embed.size(0), mode_embed.dtype)

        for layer, fusion in zip(self.dec_layers, self.refinement_fusions):
            mode_embed = layer(
                mode_embed, bev_feat, self.get_query_scale_itr(mode_embed),
                ref_points, ego_dyn, time_pe
            )
            interaction_embed = self.build_interaction_tokens(
                ref_points, pred_vel, state_query, dense_future_feature,
                dense_future_pred, dense_obj_valid_mask, target_idx
            )
            fused_feature = fusion(mode_embed, interaction_embed)
            raw_distribution = self.motion_reg(fused_feature)
            smoothing_sigma = F.softplus(
                self.refinement_smoothing_sigma_head(fused_feature).squeeze(-1)
            ).clamp_min(1e-3)
            pred_xy = ref_points + self.smooth_refinement_offset(raw_distribution[..., :2], smoothing_sigma)
            pred_traj = torch.cat([pred_xy, raw_distribution[..., 2:]], dim=-1)
            pred_vel = self.motion_vel(fused_feature)
            mode_prob = self.score_predicted_trajectory(
                mode_embed, state_query, pred_traj, pred_vel, bev_feat, ego_dyn,
                dense_future_feature, dense_future_pred, dense_obj_valid_mask, target_idx,
                self.mode_prob_head
            )
            ref_points = pred_xy.detach()
            mode_probs.append(mode_prob)
            pred_trajs.append(pred_traj.permute(0, 2, 1, 3))
            pred_vels.append(pred_vel.permute(0, 2, 1, 3))
            refinement_smoothing_sigmas.append(smoothing_sigma)

        return {
            'predicted_probability': mode_probs,
            'predicted_trajectory': pred_trajs,
            'predicted_velocity': pred_vels,
            'predicted_goal_position': predicted_goal_position,
            'predicted_goal_FDE': goal_fde,
            'refinement_smoothing_sigma': torch.stack(refinement_smoothing_sigmas),
            'state_pred': state_pred
        }

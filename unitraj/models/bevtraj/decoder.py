import math
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.bevtraj.bev_deformable_aggregation import BDA_DEC
from unitraj.models.bevtraj.decoder_deform_attn import BEVDeformCrossAttn
from unitraj.models.bevtraj.linear import MLP, FFN, MotionRegHead, MotionVelHead
from unitraj.models.bevtraj.utility import gen_sineembed_for_position, target_to_ego

from unitraj.models.bevtraj.temporal_sequential_module import TemporalMHA, TemporalMHA_NoTimePE


class BEVTrajDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.T = config['future_len']
        self.D = config['d_model']
        self.Q_D = config['query_dims']
        self.ffn_D = config['ffn_dims']
        
        self.K = config['num_modes']
        self.dropout = config['dropout']
        self.num_heads = config['num_heads']
        self.spa_pos_T = config['spa_pos_T']
        
        self.to_pos_Q = MLP(self.Q_D, self.Q_D, self.Q_D, 2)
        self.norm = nn.ModuleList([nn.LayerNorm(self.D) for _ in range(3)])
        # self.temp_self_attn = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)

        # exp: temporal PE (time_embedding_mlp)
        self.temp_self_attn = TemporalMHA(self.D, self.num_heads, self.dropout)

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(self.D, self.num_heads,
                                                                    dim_feedforward=self.ffn_D, dropout=self.dropout)
        self.bev_cross_attn = BEVDeformCrossAttn(**config['deform_cross_attn'])

        # hybrid self-attn: token 길이를 K*T로 보고 attention
        self.hybrid_self_attn = nn.MultiheadAttention(
            self.D, self.num_heads, dropout=self.dropout
        )

        self.ffn = FFN(self.D, self.ffn_D, 2)
    
    def forward(self, dec_embed, scene_context, bev_feat, query_scale, ref_points, ego_dyn, time_pe):
        '''
        Args:
            dec_embed: [T, B*K, D]
            scene_context: [t, B, D]
            bev_feat: [B, D, H, W]
            query_scale: [T, B*K, d]
            ref_points: [K, B, T, 2]
        '''
        B = bev_feat.size(0)
        num_modes = ref_points.size(0)
        scene_context = scene_context 
        
        # ============================== target-centric(tc) modeling ==============================
        
        # dec_embed = self.norm[0](self.temp_self_attn(query=dec_embed, key=dec_embed, value=dec_embed)[0] + dec_embed)

        # exp: temporal PE (time_embedding_mlp)
        temp_out = self.temp_self_attn(dec_embed, time_pe)
        dec_embed = self.norm[0](temp_out + dec_embed)
        
        # get positional query
        query_sine_embed = gen_sineembed_for_position(ref_points, hidden_dim=self.Q_D, temperature=self.spa_pos_T)
        tc_pos_Q = self.to_pos_Q(query_sine_embed)
        
        dec_embed, query_scale = map(
            lambda t: t.reshape(self.T, B, num_modes, -1).permute(2, 1, 0, 3),
            (dec_embed, query_scale),
        )
        dec_embed = dec_embed + tc_pos_Q
        dec_embed = self.transformer_decoder_layer(
            tgt=dec_embed.reshape(num_modes, B * self.T, -1),
            memory=scene_context,
        ).reshape(num_modes, B, self.T, -1)
        
        # ============================== ego-centric(ec) modeling ==============================
        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'],
            ego_dyn['ego_y'],
            ego_dyn['ego_sin'],
            ego_dyn['ego_cos'],
        )
        ref_points_flat = ref_points.permute(1, 0, 2, 3).reshape(B, num_modes * self.T, 2)
        ref_points_flat = target_to_ego(ref_points_flat, trans_x, trans_y, rot_sin, rot_cos)
        ref_points = ref_points_flat.reshape(B, num_modes, self.T, 2).permute(1, 0, 2, 3)
        
        # cross attn with bev feature
        dec_embed = self.norm[1](self.bev_cross_attn(dec_embed, bev_feat, query_scale, ref_points))

        # 5) hybrid self-attn on K*T tokens
        hybrid_tokens = dec_embed.permute(0, 2, 1, 3).reshape(num_modes * self.T, B, self.D)  # [K*T,B,D]
        hybrid_out = self.hybrid_self_attn(
            query=hybrid_tokens, key=hybrid_tokens, value=hybrid_tokens
        )[0]
        hybrid_tokens = self.norm[2](hybrid_out + hybrid_tokens)
        # restore [K,B,T,D]
        dec_embed = hybrid_tokens.reshape(num_modes, self.T, B, self.D).permute(0, 2, 1, 3).contiguous()
        # =================

        dec_embed = self.norm[2](self.ffn(dec_embed))
        
        return dec_embed


class TemporalModeCompressor(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.seed_proj = MLP(dim * 3, dim, dim, 2)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.out_proj = MLP(dim * 4, dim, dim, 2)
        self.norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim, ffn_dim, 2, dropout=dropout)

    def forward(self, tokens):
        """
        tokens: [M, B, T, D]
        returns: [M, B, D]
        """
        M, B, T, D = tokens.shape
        seq = tokens.permute(2, 1, 0, 3).reshape(T, B * M, D)

        mean_pool = tokens.mean(dim=2)
        last_token = tokens[:, :, -1, :]
        max_pool = tokens.amax(dim=2)

        seed = self.seed_proj(torch.cat([mean_pool, last_token, max_pool], dim=-1))
        query = seed.permute(1, 0, 2).reshape(1, B * M, D)
        attn_pool = self.temporal_attn(query=query, key=seq, value=seq)[0]
        attn_pool = attn_pool.reshape(B, M, D).permute(1, 0, 2).contiguous()

        fused = self.out_proj(torch.cat([attn_pool, mean_pool, last_token, max_pool], dim=-1))
        fused = self.norm(fused)
        return self.ffn(fused)


class CheapTrajectoryScorer(nn.Module):
    def __init__(self, dim, ffn_dim, dropout, grid_size, future_len, num_points=8, bev_dim=64):
        super().__init__()
        self.dim = dim
        self.future_len = future_len
        self.num_points = min(int(num_points), future_len)
        self.bev_dim = int(bev_dim)

        num_groups = 8 if self.bev_dim % 8 == 0 else 1
        self.bev_reduce = nn.Sequential(
            nn.Conv2d(dim, self.bev_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=self.bev_dim),
            nn.ReLU(),
        )

        self.query_proj = MLP(dim * 3, dim, dim, 2)
        self.query_norm = nn.LayerNorm(dim)

        self.query_step_proj = MLP(dim, self.bev_dim, self.bev_dim, 2)
        self.bev_token_proj = MLP(self.bev_dim, self.bev_dim, self.bev_dim, 2)
        self.bev_step_fuse = MLP(self.bev_dim * 2, self.bev_dim, self.bev_dim, 2)
        self.bev_temporal_proj = MLP(self.bev_dim * 3, self.bev_dim, self.bev_dim, 2)
        self.bev_norm = nn.LayerNorm(self.bev_dim)

        self.fuse = MLP(dim + self.bev_dim, dim, dim, 2)
        self.fuse_norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim, ffn_dim, 2, dropout=dropout)
        self.out_head = MLP(dim, dim, 1, 2)

        time_idx = torch.linspace(0, future_len - 1, steps=self.num_points).round().long()
        self.register_buffer('sparse_time_idx', time_idx)
        self.register_buffer('denorm_scale', torch.tensor(grid_size, dtype=torch.float32))

    def project_bev(self, bev_feat):
        return self.bev_reduce(bev_feat)

    def sample_bev_tokens(self, pred_xy, bev_feat, ego_dyn):
        M, B, _, _ = pred_xy.shape
        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'],
            ego_dyn['ego_y'],
            ego_dyn['ego_sin'],
            ego_dyn['ego_cos'],
        )

        pred_xy_sparse = pred_xy.index_select(2, self.sparse_time_idx).detach()
        pred_xy_flat = pred_xy_sparse.permute(1, 0, 2, 3).reshape(B, M * self.num_points, 2)
        pred_xy_ego = target_to_ego(pred_xy_flat, trans_x, trans_y, rot_sin, rot_cos)
        grid = pred_xy_ego / self.denorm_scale[None, None, :]
        grid = grid.reshape(B, M * self.num_points, 1, 2)
        grid[..., 1] = -grid[..., 1]

        sampled = F.grid_sample(
            bev_feat,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False,
        )
        sampled = sampled.reshape(B, self.bev_dim, M, self.num_points).permute(2, 0, 3, 1).contiguous()
        return sampled

    def aggregate_bev_tokens(self, bev_tokens, traj_query):
        query_sparse = traj_query.index_select(2, self.sparse_time_idx)
        query_sparse = self.query_step_proj(query_sparse)

        bev_tokens = self.bev_token_proj(bev_tokens)
        bev_tokens = self.bev_step_fuse(torch.cat([bev_tokens, query_sparse], dim=-1))

        mean_bev = bev_tokens.mean(dim=2)
        last_bev = bev_tokens[:, :, -1, :]
        max_bev = bev_tokens.amax(dim=2)
        bev_mode = self.bev_temporal_proj(torch.cat([mean_bev, last_bev, max_bev], dim=-1))
        return self.bev_norm(bev_mode)

    def forward(self, traj_query, pred_xy, bev_feat, ego_dyn):
        mean_q = traj_query.mean(dim=2)
        last_q = traj_query[:, :, -1, :]
        max_q = traj_query.amax(dim=2)
        query_mode = self.query_norm(self.query_proj(torch.cat([mean_q, last_q, max_q], dim=-1)))

        bev_tokens = self.sample_bev_tokens(pred_xy, bev_feat, ego_dyn)
        bev_mode = self.aggregate_bev_tokens(bev_tokens, traj_query)

        fused = self.fuse(torch.cat([query_mode, bev_mode], dim=-1))
        fused = self.fuse_norm(fused)
        fused = self.ffn(fused)
        return self.out_head(fused).squeeze(dim=-1).T


class BEVTrajDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # short refs
        self.t = config['past_len']
        self.T = config['future_len']
        self.D = config['d_model']
        self.ffn_D = config['ffn_dims']
        self.t_D = config['t_dims']
        self.T_D = config['T_dims']
        self.K = config['num_modes']
        self.target_attr = config['target_attr']
        self.query_scale_dims = config['query_scale_dims']
        self.mode_pos_T = config['mode_pos_T']
        self.spa_pos_T = config['spa_pos_T']
        self.dropout = config['dropout']
        self.L_goal_proposal = config['num_goal_proposal_layers']
        self.L_dec = config['num_decoder_layers']
        self.num_heads = config['num_heads']
        self.grid_size = config['grid_size']
        self.num_heavy_refinement_score_layers = int(
            config.get('num_heavy_refinement_score_layers', 1)
        )
        
        self.dca_cfg = config['deform_cross_attn']
        self.dca_cfg['dim'] = self.D

        self.dca_itr_cfg = config['deform_cross_attn_itr']
        self.dca_itr_cfg['dim'] = self.D

        self.dec_layer_config = {
            'future_len': self.T,
            'd_model': self.D,
            'query_dims': self.query_scale_dims,
            'ffn_dims': self.ffn_D,
            'spa_pos_T': self.spa_pos_T,
            'num_modes': self.K,
            'dropout': self.dropout,
            'num_heads': self.num_heads,
            'deform_cross_attn': self.dca_itr_cfg,
        }
        
        # ============================ Goal Candidate Proposal==========================

        self.bda_sgcp = BDA_DEC(self.config['bda_dec'], self.D)

        # regression
        self.goal_proposal = []
        for _ in range(self.L_goal_proposal):
            goal_proposal_layer = nn.ModuleDict({
                'deform_cross_attn': BEVDeformCrossAttn(**self.dca_cfg),
                'norm1': nn.LayerNorm(self.D),
                'mode_self_attn': nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout),
                'norm2': nn.LayerNorm(self.D),
                'ffn': FFN(self.D, self.ffn_D, 2),
                'norm3': nn.LayerNorm(self.D),
            })
            self.goal_proposal.append(goal_proposal_layer)
        self.goal_proposal = nn.ModuleList(self.goal_proposal)

        self.get_query_scale_sgcp = MLP(self.D, self.query_scale_dims, self.query_scale_dims, 2)
        self.goal_reg = MLP(self.D, self.D, 2, 2)

        self.register_buffer('denorm_scale', torch.tensor(self.grid_size, dtype=torch.float32))

        # ============================ Initial Prediction ==============================
        self.get_query_scale_itp = MLP(self.D, self.query_scale_dims, self.query_scale_dims, 2)
        self.norm_l1 = nn.ModuleList([nn.LayerNorm(self.D) for _ in range(3)])
        
        self.context_cross_attn_l1 = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)
        self.bev_cross_attn_l1 = BEVDeformCrossAttn(**self.dca_cfg)
        self.ffn_l1 = FFN(self.D, self.ffn_D, 2)
        
        # self.tmp_MLP = nn.ModuleList([
        #     nn.Sequential(nn.Linear(self.D, self.T_D * self.T), nn.GELU()),
        #     nn.Sequential(nn.Linear(self.T_D, self.D), nn.GELU())
        # ])
        self.mode_prob_head_l1 = MLP(self.D, self.D, 1, 2)
        self.motion_reg_l1 = MotionRegHead(self.D)
        self.motion_vel_l1 = MotionVelHead(self.D)

        # trajectory reasonableness scorer
        self.traj_motion_proj = MLP(11, self.D, self.D, 2)
        self.traj_history_proj = MLP(self.target_attr, self.D, self.D, 2)
        self.traj_history_attn = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)
        self.traj_bev_query_scale = MLP(self.D, self.query_scale_dims, self.query_scale_dims, 2)
        self.traj_bev_cross_attn = BEVDeformCrossAttn(**self.dca_itr_cfg)
        self.traj_bev_proj = MLP(self.D, self.D, self.D, 2)
        self.traj_interaction_proj = MLP(6, self.D, self.D, 2)
        self.traj_scene_context_proj = MLP(self.D, self.D, self.D, 2)
        self.traj_interaction_fuse_proj = MLP(self.D * 2, self.D, self.D, 2)
        self.traj_query_pool = TemporalModeCompressor(self.D, self.num_heads, self.ffn_D, self.dropout)
        self.traj_dyn_pool = TemporalModeCompressor(self.D, self.num_heads, self.ffn_D, self.dropout)
        self.traj_bev_pool = TemporalModeCompressor(self.D, self.num_heads, self.ffn_D, self.dropout)
        self.traj_interaction_pool = TemporalModeCompressor(self.D, self.num_heads, self.ffn_D, self.dropout)
        self.traj_score_fuse = MLP(self.D * 4, self.D, self.D, 2)
        self.traj_score_norm = nn.LayerNorm(self.D)
        self.traj_score_ffn = FFN(self.D, self.ffn_D, 2)
        self.traj_score_cheap = CheapTrajectoryScorer(
            self.D,
            self.ffn_D,
            self.dropout,
            self.grid_size,
            self.T,
            num_points=config.get('cheap_scorer_num_points', 8),
            bev_dim=config.get('cheap_scorer_dim', 64),
        )

        # exp: DeMo-like ITP (state consistency branch)
        self.state_norm_l1 = nn.ModuleList([nn.LayerNorm(self.D) for _ in range(2)])
        self.state_context_cross_attn_l1 = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)
        self.state_temp_self_attn_l1 = TemporalMHA_NoTimePE(self.D, self.num_heads, self.dropout)
        # state query auxiliary prediction head (B,T,2 supervision)
        self.state_reg_l1 = MLP(self.D, self.D, 2, 2)


        # ============================ Iterative Refinement ============================

        # exp: temporal PE (time_embedding_mlp)
        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, self.D)
        )
        self.register_buffer("future_time", torch.arange(self.T).float().unsqueeze(-1))
        self.dt = config.get("dt", 0.1)
        self.time_emb_alpha = nn.Parameter(torch.tensor(1.0))
        
        # self.mode_sep_enc = ModeSeperationEncoding(self.D, self.dropout, mode_num=self.K, temperature=self.mode_pos_T)
        self.get_query_scale_itr = MLP(self.query_scale_dims, self.query_scale_dims, self.query_scale_dims, 2)
        
        dec_layer = BEVTrajDecoderLayer(self.dec_layer_config)
        self.dec_layers = nn.ModuleList([copy.deepcopy(dec_layer) for _ in range(self.L_dec - 1)])
        
        self.mode_prob_head = MLP(self.D, self.D, 1, 2)
        self.motion_reg = MotionRegHead(self.D)
        self.motion_vel = MotionVelHead(self.D)

        # exp: sample-conditioned deterministic code
        # self.temp_pos_enc = TemporalPositionalEncoding(self.D, self.dropout, future_len=self.T, temperature=10000)

    def build_time_pe(self, B, K, dtype):
        t = self.future_time * self.dt + 0.1
        t = t.to(dtype) # [T,1]
        pe = self.time_embedding_mlp(t)  # [T,D]
        pe = pe[:, None, None, :].repeat(1, B, K, 1)  # [T,B,K,D]
        pe = pe.reshape(self.T, B * K, self.D)  # [T,BK,D]

        return self.time_emb_alpha * pe

    def goal_candidate_proposal(self, bev_feat, ec_dyn, tc_dyn, ego_dyn):
        bda_token, anchor_pos = self.bda_sgcp(bev_feat, ec_dyn, tc_dyn, ego_dyn)

        mode_query = bda_token.permute(1, 0, 2).contiguous()

        return mode_query, anchor_pos

    def build_traj_motion_tokens(self, pred_traj, pred_vel):
        pred_xy = pred_traj[..., :2]

        disp = torch.zeros_like(pred_xy)
        disp[..., 1:, :] = pred_xy[..., 1:, :] - pred_xy[..., :-1, :]

        accel = torch.zeros_like(pred_vel)
        accel[..., 1:, :] = pred_vel[..., 1:, :] - pred_vel[..., :-1, :]

        speed = pred_vel.norm(dim=-1, keepdim=True)
        accel_norm = accel.norm(dim=-1, keepdim=True)

        traj_feat = torch.cat(
            [
                pred_xy,
                disp,
                pred_vel,
                pred_traj[..., 2:4],
                pred_traj[..., 4:5],
                speed,
                accel_norm,
            ],
            dim=-1,
        )
        return self.traj_motion_proj(traj_feat)

    def sample_traj_bev_tokens(self, traj_query, pred_xy, bev_feat, ego_dyn):
        M, B, T, _ = pred_xy.shape
        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'],
            ego_dyn['ego_y'],
            ego_dyn['ego_sin'],
            ego_dyn['ego_cos'],
        )

        pred_xy_flat = pred_xy.permute(1, 0, 2, 3).reshape(B, M * T, 2)
        pred_xy_ego = target_to_ego(pred_xy_flat, trans_x, trans_y, rot_sin, rot_cos)
        pred_xy_ego = pred_xy_ego.reshape(B, M, T, 2).permute(1, 0, 2, 3).contiguous()

        bev_tokens = self.traj_bev_cross_attn(
            dec_embed=traj_query,
            bev_feat=bev_feat,
            query_scale=self.traj_bev_query_scale(traj_query),
            ref_points=pred_xy_ego,
            identity=torch.zeros_like(traj_query),
        )
        return self.traj_bev_proj(bev_tokens)

    def build_dynamic_context(self, traj_query, tc_dyn):
        M, B, T, D = traj_query.shape
        history_tokens = self.traj_history_proj(tc_dyn).permute(1, 0, 2).contiguous()
        history_tokens = history_tokens.unsqueeze(2).repeat(1, 1, M, 1).reshape(self.t, B * M, D)

        query = traj_query.permute(2, 1, 0, 3).reshape(T, B * M, D)
        dyn_ctx = self.traj_history_attn(query=query, key=history_tokens, value=history_tokens)[0]
        return dyn_ctx.reshape(T, B, M, D).permute(2, 1, 0, 3).contiguous()

    def build_interaction_tokens(
        self,
        pred_xy,
        pred_vel,
        scene_context=None,
        dense_future_pred=None,
        obj_valid_mask=None,
        target_idx=None,
    ):
        M, B, T, _ = pred_xy.shape
        if dense_future_pred is None or scene_context is None:
            return pred_xy.new_zeros(M, B, T, self.D)

        obj_xy = dense_future_pred[..., :2]
        obj_vel = dense_future_pred[..., -2:]
        num_objects = obj_xy.size(1)
        scene_obj_tokens = self.traj_scene_context_proj(scene_context)

        traj_xy = pred_xy.permute(1, 0, 2, 3).unsqueeze(2)
        traj_vel = pred_vel.permute(1, 0, 2, 3).unsqueeze(2)
        obj_xy = obj_xy.unsqueeze(1)
        obj_vel = obj_vel.unsqueeze(1)

        rel_xy = traj_xy - obj_xy
        dist = rel_xy.norm(dim=-1)
        rel_dir = rel_xy / dist.unsqueeze(-1).clamp_min(1e-3)
        closing = -((traj_vel - obj_vel) * rel_dir).sum(dim=-1)
        obj_speed = obj_vel.norm(dim=-1)

        if obj_valid_mask is None:
            valid_mask = torch.ones(B, num_objects, device=pred_xy.device, dtype=torch.bool)
        else:
            valid_mask = obj_valid_mask.bool()

        if target_idx is not None:
            valid_mask = valid_mask.clone()
            valid_mask[torch.arange(B, device=pred_xy.device), target_idx.long()] = False

        valid_mask = valid_mask[:, None, :, None].expand(-1, M, -1, T)
        masked_dist = dist.masked_fill(~valid_mask, 1e4)
        weights = torch.softmax(-masked_dist, dim=2) * valid_mask.float()
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)

        agg_rel_xy = (weights.unsqueeze(-1) * rel_xy).sum(dim=2)
        agg_dist = (weights * dist).sum(dim=2).unsqueeze(-1)
        min_dist = masked_dist.amin(dim=2).unsqueeze(-1)
        agg_closing = (weights * closing).sum(dim=2).unsqueeze(-1)
        agg_obj_speed = (weights * obj_speed).sum(dim=2).unsqueeze(-1)

        interaction_feat = torch.cat(
            [agg_rel_xy, agg_dist, min_dist, agg_closing, agg_obj_speed],
            dim=-1,
        )
        interaction_tokens = self.traj_interaction_proj(interaction_feat)
        scene_tokens = torch.einsum('bmot,bod->bmtd', weights, scene_obj_tokens)
        interaction_tokens = self.traj_interaction_fuse_proj(
            torch.cat([interaction_tokens, scene_tokens], dim=-1)
        )
        return interaction_tokens.permute(1, 0, 2, 3).contiguous()

    def score_predicted_trajectory(
        self,
        dec_embed,
        pred_traj,
        pred_vel,
        bev_feat,
        ego_dyn,
        tc_dyn,
        scene_context=None,
        dense_future_pred=None,
        obj_valid_mask=None,
        target_idx=None,
        mode_prob_head=None,
    ):
        pred_xy = pred_traj[..., :2]
        motion_tokens = self.build_traj_motion_tokens(pred_traj, pred_vel)
        traj_query = dec_embed + motion_tokens

        dyn_tokens = self.build_dynamic_context(traj_query, tc_dyn)
        bev_tokens = self.sample_traj_bev_tokens(traj_query, pred_xy, bev_feat, ego_dyn)
        interaction_tokens = self.build_interaction_tokens(
            pred_xy,
            pred_vel,
            scene_context=scene_context,
            dense_future_pred=dense_future_pred,
            obj_valid_mask=obj_valid_mask,
            target_idx=target_idx,
        )

        traj_query_mode = self.traj_query_pool(traj_query)
        dyn_tokens_mode = self.traj_dyn_pool(dyn_tokens)
        bev_tokens_mode = self.traj_bev_pool(bev_tokens)
        interaction_tokens_mode = self.traj_interaction_pool(interaction_tokens)

        fused = torch.cat(
            [traj_query_mode, dyn_tokens_mode, bev_tokens_mode, interaction_tokens_mode],
            dim=-1,
        )
        fused = self.traj_score_fuse(fused)
        fused = self.traj_score_norm(fused)
        fused = self.traj_score_ffn(fused)
        return mode_prob_head(fused).squeeze(dim=-1).T

    def score_predicted_trajectory_cheap(self, dec_embed, pred_traj, pred_vel, bev_feat, ego_dyn):
        pred_xy = pred_traj[..., :2]
        motion_tokens = self.build_traj_motion_tokens(pred_traj, pred_vel)
        traj_query = dec_embed + motion_tokens
        return self.traj_score_cheap(traj_query, pred_xy, bev_feat, ego_dyn)

    def initial_prediction(
        self,
        mode_query,
        scene_context,
        bev_feat,
        anchor_pos,
        ego_dyn,
        tc_dyn,
        scene_context_tokens=None,
        dense_future_pred=None,
        obj_valid_mask=None,
        target_idx=None,
    ):
        M, B, _ = mode_query.shape

        # ===================== mode localization branch =====================
        query_scale = self.get_query_scale_itp(mode_query)

        mode_embed = self.norm_l1[0](
            self.context_cross_attn_l1(
                query=mode_query, key=scene_context, value=scene_context
            )[0] + mode_query
        )

        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'],
            ego_dyn['ego_y'],
            ego_dyn['ego_sin'],
            ego_dyn['ego_cos'],
        )

        anchor_pos = target_to_ego(
            anchor_pos, trans_x, trans_y, rot_sin, rot_cos
        ).permute(1, 0, 2)  # [M, B, 2] (ego coord for BEV cross-attn)

        mode_embed = self.norm_l1[1](
            self.bev_cross_attn_l1(
                dec_embed=mode_embed,
                bev_feat=bev_feat,
                query_scale=query_scale,
                ref_points=anchor_pos,
            )
        )
        mode_embed = self.norm_l1[2](self.ffn_l1(mode_embed))  # [M,B,D]

        # ===================== state consistency branch =====================
        t = (self.future_time * self.dt + 0.1).to(
            device=mode_query.device, dtype=mode_query.dtype
        )

        state_query = self.time_emb_alpha * self.time_embedding_mlp(t)  # [T,D]
        state_query = state_query.unsqueeze(1).repeat(1, B, 1)  # [T,B,D]

        state_query = self.state_norm_l1[0](
            self.state_context_cross_attn_l1(
                query=state_query, key=scene_context, value=scene_context
            )[0] + state_query
        )

        state_query = self.state_norm_l1[1](
            self.state_temp_self_attn_l1(state_query, None) + state_query
        )

        state_pred = self.state_reg_l1(state_query).permute(1, 0, 2).contiguous()

        # ===================== hybrid coupling =====================
        mode_bt = mode_embed.permute(1, 0, 2).unsqueeze(2)  # [B,M,1,D]
        state_bt = state_query.permute(1, 0, 2).unsqueeze(1)  # [B,1,T,D]

        dec_embed_T = mode_bt + state_bt  # [B,M,T,D]
        dec_embed_T = dec_embed_T.permute(1, 0, 2, 3).contiguous()  # [M,B,T,D]

        # ===================== trajectory prediction =====================
        out_dist = self.motion_reg_l1(dec_embed_T)  # [M,B,T,5]
        out_vel = self.motion_vel_l1(dec_embed_T)  # [M,B,T,2]
        mode_prob = self.score_predicted_trajectory(
            dec_embed=dec_embed_T,
            pred_traj=out_dist,
            pred_vel=out_vel,
            bev_feat=bev_feat,
            ego_dyn=ego_dyn,
            tc_dyn=tc_dyn,
            scene_context=scene_context_tokens,
            dense_future_pred=dense_future_pred,
            obj_valid_mask=obj_valid_mask,
            target_idx=target_idx,
            mode_prob_head=self.mode_prob_head_l1,
        )

        return dec_embed_T, mode_prob, out_dist, out_vel, state_pred

    def forward(self, scene_context, bev_feat, ec_dyn, tc_dyn, ego_dyn, **kwargs):

        B, _, _ = ec_dyn.shape
        scene_context_tokens = scene_context
        n = scene_context.shape[1]

        scene_context_repeat = scene_context.unsqueeze(2).repeat(1, 1, self.T, 1)
        scene_context_repeat = scene_context_repeat.permute(1, 0, 2, 3).reshape(n, B * self.T, -1)
        scene_context = scene_context.permute(1, 0, 2)

        # -------------------Goal Candidate Proposal -----------------
        mode_query, anchor_pos = \
            self.goal_candidate_proposal(
                bev_feat, ec_dyn, tc_dyn, ego_dyn
            )
        anchor_pos_detached = anchor_pos.detach()

        dense_future_pred = kwargs.get('dense_future_pred')
        obj_valid_mask = kwargs.get('obj_valid_mask')
        target_idx = kwargs.get('target_idx')

        # -------------------- Initial Prediction --------------------
        dec_embed, init_mode_prob, init_pred_traj, init_pred_vel, state_pred = \
            self.initial_prediction(
                mode_query,
                scene_context,
                bev_feat,
                anchor_pos_detached,
                ego_dyn,
                tc_dyn,
                scene_context_tokens=scene_context_tokens,
                dense_future_pred=dense_future_pred,
                obj_valid_mask=obj_valid_mask,
                target_idx=target_idx,
            )
            # self.initial_prediction(mode_query, scene_context, bev_feat, anchor_pos, ego_dyn)
        
        mode_probs = [init_mode_prob]
        pred_trajs = [init_pred_traj.permute(0, 2, 1, 3)]
        pred_vels = [init_pred_vel.permute(0, 2, 1, 3)]

        ref_points = init_pred_traj[..., :2].detach().clone()
        num_modes = ref_points.size(0)

        dec_embed = dec_embed.permute(2, 1, 0, 3).reshape(self.T, B * num_modes, -1)

        # exp: sample-conditioned deterministic code
        # dec_embed = self.temp_pos_enc(dec_embed)

        # exp: temporal PE (time_embedding_mlp)
        time_pe = self.build_time_pe(B, num_modes, dec_embed.dtype)
        cheap_bev_feat = None

        for lid, layer in enumerate(self.dec_layers):
            query_scale = self.get_query_scale_itr(dec_embed)
            dec_embed = layer(
                dec_embed=dec_embed,
                scene_context=scene_context_repeat,
                bev_feat=bev_feat,
                query_scale=query_scale,
                ref_points=ref_points,
                ego_dyn=ego_dyn,
                time_pe=time_pe,
                )
            
            pred_traj_raw = self.motion_reg(dec_embed)          # [K, B, T, 5]
            pred_xy = pred_traj_raw[..., :2] + ref_points       # out-of-place
            pred_traj = torch.cat([pred_xy, pred_traj_raw[..., 2:]], dim=-1)
            ref_points = pred_xy.detach().clone()

            pred_vel = self.motion_vel(dec_embed) # [K, B, T, 2]
            if lid < self.num_heavy_refinement_score_layers:
                mode_prob = self.score_predicted_trajectory(
                    dec_embed=dec_embed,
                    pred_traj=pred_traj,
                    pred_vel=pred_vel,
                    bev_feat=bev_feat,
                    ego_dyn=ego_dyn,
                    tc_dyn=tc_dyn,
                    scene_context=scene_context_tokens,
                    dense_future_pred=dense_future_pred,
                    obj_valid_mask=obj_valid_mask,
                    target_idx=target_idx,
                    mode_prob_head=self.mode_prob_head,
                )
            else:
                if cheap_bev_feat is None:
                    cheap_bev_feat = self.traj_score_cheap.project_bev(bev_feat)
                mode_prob = self.score_predicted_trajectory_cheap(
                    dec_embed=dec_embed,
                    pred_traj=pred_traj,
                    pred_vel=pred_vel,
                    bev_feat=cheap_bev_feat,
                    ego_dyn=ego_dyn,
                )

            pred_traj = pred_traj.permute(0, 2, 1, 3).contiguous()
            pred_vel = pred_vel.permute(0, 2, 1, 3).contiguous()
            mode_probs.append(mode_prob)
            pred_trajs.append(pred_traj)
            pred_vels.append(pred_vel)
            
            dec_embed = dec_embed.permute(2, 1, 0, 3).reshape(self.T, B * num_modes, -1)
            
        output = {'predicted_probability': mode_probs,
                  'predicted_trajectory': pred_trajs,
                  'predicted_velocity': pred_vels,
                  'anchor_pos' : anchor_pos,
                #   'init_top_idx': init_top_idx,                # [B, K]
                  'state_pred': state_pred, # [B, T, 2]
                #   'goal_FDE_list': goal_FDE_list,
                }
        return output
    

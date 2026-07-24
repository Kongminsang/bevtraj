import math
import copy
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.bevtraj.bev_deformable_aggregation import BDA_DEC
from unitraj.models.bevtraj.decoder_deform_attn import BEVDeformCrossAttn
from unitraj.models.bevtraj.linear import MLP, FFN, MotionRegHead, MotionVelHead
from unitraj.models.bevtraj.utility import gen_sineembed_for_position, target_to_ego

from unitraj.models.bevtraj.temporal_sequential_module import TemporalMHA, TemporalMHA_NoTimePE


MODEL_DIR = Path(__file__).resolve().parent


class QueryConditionedDynamics(nn.Module):
    """FiLM-condition feature tokens with a broadcastable query embedding."""

    def __init__(self, query_dim, hidden_dim):
        super().__init__()

        self.modulator = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

        # Start from an identity transform and learn query-specific modulation.
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

    def forward(self, dynamics, query_emb):
        gamma, beta = self.modulator(query_emb).chunk(2, dim=-1)
        return (1 + gamma) * dynamics + beta


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
        self.goal_score_hidden_dim = int(config.get('goal_score_hidden_dim', max(self.D // 4, 32)))
        self.goal_flow_distance_threshold = float(config.get('goal_flow_distance_threshold', 10.0))
        self.goal_flow_num_neighbors = int(config.get('goal_flow_num_neighbors', 8))
        self.goal_flow_hidden_dim = int(config.get('goal_flow_hidden_dim', max(self.D // 4, 32)))
        self.goal_flow_anchor_chunk_size = int(config.get('goal_flow_anchor_chunk_size', 64))
        self.goal_heading_chord_steps = int(config.get('goal_heading_chord_steps', 5))
        self.L_dec = config['num_decoder_layers']
        self.num_heads = config['num_heads']
        self.grid_size = config['grid_size']
        self.num_heavy_refinement_score_layers = int(config.get('num_heavy_refinement_score_layers', 1))
        self.score_time_stride_late_layers = int(config.get('score_time_stride_late_layers', 1))
        self.score_time_full_refine_layers = int(config.get('score_time_full_refine_layers', 1))
        self.refinement_smoothing_kernel_size = int(config.get('refinement_smoothing_kernel_size', 7))
        self.refinement_smoothing_sigma = float(config.get('refinement_smoothing_sigma', 1.5))
        self.refinement_smoothing_alpha = float(config.get('refinement_smoothing_alpha', 0.0))

        smoothing_radius = self.refinement_smoothing_kernel_size // 2
        smoothing_steps = torch.arange(-smoothing_radius, smoothing_radius + 1, dtype=torch.float32)
        smoothing_kernel = torch.exp(-0.5 * (smoothing_steps / self.refinement_smoothing_sigma).square())
        smoothing_kernel = smoothing_kernel / smoothing_kernel.sum()
        self.register_buffer('refinement_smoothing_kernel', smoothing_kernel.view(1, 1, -1), persistent=False)
        
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

        self.goal_anchor_file_name = config['goal_anchor_file_name']
        file_path = MODEL_DIR / self.goal_anchor_file_name
        with open(file_path, 'rb') as f:
            goal_anchor_data = pickle.load(f)['VEHICLE']
        goal_anchor_indices = torch.as_tensor(goal_anchor_data['anchor_indices'], dtype=torch.long)
        goal_attn_mask = torch.ones(self.K, self.bda_sgcp.anchors.size(0), dtype=torch.bool)
        goal_attn_mask.scatter_(1, goal_anchor_indices, False)
        self.register_buffer('goal_attn_mask', goal_attn_mask, persistent=False)
        self.register_buffer('goal_anchor_indices', goal_anchor_indices, persistent=False)

        # Estimate the target's arrival heading at every dense anchor once
        # at construction time. A compact, stable trajectory set can be
        # selected independently from initial_prediction's templates.
        self.goal_heading_trajectory_file_name = config.get(
            'goal_heading_trajectory_file_name',
            config.get('trajectory_file_name', 'trajectory_set_64_60.pkl'),
        )
        file_path = MODEL_DIR / self.goal_heading_trajectory_file_name
        with open(file_path, 'rb') as f:
            heading_trajectory_set = torch.from_numpy(pickle.load(f)['VEHICLE']).float()
        (
            goal_anchor_template_idx,
            goal_anchor_heading,
            goal_anchor_heading_sincos,
        ) = self.precompute_goal_anchor_headings(
            self.bda_sgcp.anchors.detach().cpu(),
            heading_trajectory_set,
            chord_steps=self.goal_heading_chord_steps,
        )
        self.register_buffer('goal_anchor_template_idx', goal_anchor_template_idx, persistent=False)
        self.register_buffer('goal_anchor_heading', goal_anchor_heading, persistent=False)
        self.register_buffer('goal_anchor_heading_sincos', goal_anchor_heading_sincos, persistent=False)

        # Keep per-agent tokens in a small hidden space through top-k
        # pooling; only the per-anchor aggregate is projected to D.
        goal_flow_state_dim = 15
        self.goal_flow_state_encoder = MLP(goal_flow_state_dim, self.goal_flow_hidden_dim, self.goal_flow_hidden_dim, 3)
        self.goal_flow_agent_score = MLP(self.goal_flow_hidden_dim, self.goal_flow_hidden_dim, 1, 2)
        self.goal_flow_pool_proj = MLP(self.goal_flow_hidden_dim * 3 + 1, self.D, self.D, 2)
        self.goal_flow_norm = nn.LayerNorm(self.D)

        # BDA and traffic-flow tokens describe different kinds of evidence.
        # Keep a dedicated mode query for each source so neither representation
        # has to decode a token in which the two modalities were mixed early.
        self.goal_bda_query = nn.Parameter(torch.empty(self.K, 1, self.D))
        nn.init.xavier_uniform_(self.goal_bda_query)
        self.goal_flow_query = nn.Parameter(torch.empty(self.K, 1, self.D))
        nn.init.xavier_uniform_(self.goal_flow_query)

        # Both branches use the same cluster mask, but have completely separate
        # cross-attention, self-attention, and FFN parameters.
        self.goal_proposal = nn.ModuleList()
        for _ in range(self.L_goal_proposal):
            goal_layer = {
                'bda_cross_attn': nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout),
                'bda_cross_norm': nn.LayerNorm(self.D),
                'bda_self_attn': nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout),
                'bda_self_norm': nn.LayerNorm(self.D),
                'bda_ffn': FFN(self.D, self.ffn_D, 2),
                'bda_ffn_norm': nn.LayerNorm(self.D),
                'flow_cross_attn': nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout),
                'flow_cross_norm': nn.LayerNorm(self.D),
                'flow_self_attn': nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout),
                'flow_self_norm': nn.LayerNorm(self.D),
                'flow_ffn': FFN(self.D, self.ffn_D, 2),
                'flow_ffn_norm': nn.LayerNorm(self.D),
            }
            self.goal_proposal.append(nn.ModuleDict(goal_layer))

        # Cross-attention builds mode context; a separate mode-conditioned head
        # predicts the endpoint-anchor distribution.
        self.goal_bda_conditioner = QueryConditionedDynamics(self.D, self.D)
        self.goal_bda_score_proj = MLP(self.D, self.D, self.goal_score_hidden_dim, 2)
        self.goal_flow_conditioner = QueryConditionedDynamics(self.D, self.D)
        self.goal_flow_score_proj = MLP(self.D, self.D, self.goal_score_hidden_dim, 2)
        self.goal_score_fusion = MLP(2 * self.goal_score_hidden_dim, self.goal_score_hidden_dim, 1, 2)

        # The selected anchor evidence is fused back into the downstream mode
        # representation so the hard goal and mode semantics stay aligned.
        self.goal_selected_mode_fusion = MLP(4 * self.D, self.D, self.D, 2)
        self.goal_selected_mode_norm = nn.LayerNorm(self.D)
        self.goal_FDE = MLP(self.D, self.D, 1, 2)

        self.register_buffer('denorm_scale', torch.tensor(self.grid_size, dtype=torch.float32))

        # ============================ Initial Prediction ==============================
        self.trajectory_file_name = config['trajectory_file_name']
        self.num_waypoints = config['num_waypoints']
        file_path = MODEL_DIR / self.trajectory_file_name
        with open(file_path, 'rb') as f:
            trajectory_set = pickle.load(f)['VEHICLE']
        self.register_buffer('trajectory_set', torch.from_numpy(trajectory_set).float(), persistent=False)

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

    def smooth_refinement_offset(self, raw_offset):
        """Apply fixed residual Gaussian smoothing along the time dimension."""
        assert raw_offset.ndim == 4
        assert raw_offset.size(2) == self.T
        assert raw_offset.size(3) == 2

        if (
            self.refinement_smoothing_alpha == 0.0
            or self.refinement_smoothing_kernel_size == 1
        ):
            return raw_offset

        num_modes, batch_size, num_steps, _ = raw_offset.shape
        offset_channels = raw_offset.permute(0, 1, 3, 2).reshape(
            num_modes * batch_size, 2, num_steps
        )
        smoothing_radius = self.refinement_smoothing_kernel_size // 2
        offset_channels = F.pad(
            offset_channels,
            (smoothing_radius, smoothing_radius),
            mode='replicate',
        )
        kernel = self.refinement_smoothing_kernel.to(dtype=raw_offset.dtype).expand(2, -1, -1).contiguous()
        smooth_offset = F.conv1d(
            offset_channels,
            kernel,
            groups=2,
        )
        smooth_offset = smooth_offset.reshape(
            num_modes, batch_size, 2, num_steps
        ).permute(0, 1, 3, 2).contiguous()

        return raw_offset + self.refinement_smoothing_alpha * (
            smooth_offset - raw_offset
        )

    @staticmethod
    def precompute_goal_anchor_headings(anchors, trajectories, chord_steps=5):
        """Map each anchor to a nearby trajectory's robust terminal heading.

        Args:
            anchors: Dense goal anchors in the current target frame, [A, 2].
            trajectories: Target-frame trajectory prototypes, [L, T, 2].
            chord_steps: Number of terminal intervals used for the heading chord.

        Returns:
            Nearest template indices [A], angles [A], and (sin, cos) [A, 2].
        """
        anchors = torch.as_tensor(anchors, dtype=torch.float32)
        trajectories = torch.as_tensor(trajectories, dtype=torch.float32)

        num_steps = trajectories.size(1)
        chord_steps = max(1, min(int(chord_steps), num_steps - 1)) if num_steps > 1 else 0
        if chord_steps > 0:
            terminal_vec = trajectories[:, -1] - trajectories[:, -1 - chord_steps]

            # If a prototype is stationary over the terminal chord, use its
            # most recent non-zero displacement instead of an arbitrary angle.
            step_delta = trajectories[:, 1:] - trajectories[:, :-1]
            moving = step_delta.square().sum(dim=-1) > 1e-8
            step_indices = torch.arange(step_delta.size(1), dtype=torch.long).view(1, -1)
            last_moving_idx = torch.where(
                moving, step_indices, step_indices.new_full(step_indices.shape, -1)
            ).amax(dim=-1)
            fallback_delta = step_delta[
                torch.arange(step_delta.size(0)),
                last_moving_idx.clamp_min(0),
            ]
            use_fallback = terminal_vec.square().sum(dim=-1) <= 1e-8
            terminal_vec = torch.where(use_fallback.unsqueeze(-1), fallback_delta, terminal_vec)
        else:
            terminal_vec = trajectories[:, -1].clone()

        # A fully stationary trajectory falls back to its endpoint direction;
        # the degenerate origin case uses the target's forward (+x) direction.
        stationary = terminal_vec.square().sum(dim=-1) <= 1e-8
        terminal_vec = torch.where(stationary.unsqueeze(-1), trajectories[:, -1], terminal_vec)
        stationary = terminal_vec.square().sum(dim=-1) <= 1e-8
        forward = terminal_vec.new_tensor([1.0, 0.0]).expand_as(terminal_vec)
        terminal_vec = torch.where(stationary.unsqueeze(-1), forward, terminal_vec)
        terminal_direction = F.normalize(terminal_vec, dim=-1)

        endpoint_distance_sq = (anchors[:, None] - trajectories[None, :, -1]).square().sum(dim=-1)
        template_idx = endpoint_distance_sq.argmin(dim=-1)
        anchor_direction = terminal_direction[template_idx]
        anchor_heading = torch.atan2(anchor_direction[:, 1], anchor_direction[:, 0])
        anchor_heading_sincos = torch.stack([anchor_direction[:, 1], anchor_direction[:, 0]], dim=-1)
        return template_idx, anchor_heading, anchor_heading_sincos

    @staticmethod
    def rotate_to_goal_frame(vectors, goal_heading_sincos):
        """Rotate row-vector XY data by the negative goal heading."""
        sin_heading = goal_heading_sincos[..., 0]
        cos_heading = goal_heading_sincos[..., 1]
        x_local = vectors[..., 0] * cos_heading + vectors[..., 1] * sin_heading
        y_local = -vectors[..., 0] * sin_heading + vectors[..., 1] * cos_heading
        return torch.stack([x_local, y_local], dim=-1)

    def build_goal_anchor_flow_tokens(self, agent_history, target_idx, output_dtype):
        """Encode nearby agents in every anchor's arrival-pose frame.

        Current position, heading, velocity, and acceleration directly describe
        local traffic motion. Size/type retain agent semantics. The target is
        excluded because its history is already encoded by BDA_DEC.
        """
        history_pos = agent_history['positions']
        history_mask = agent_history['valid_mask'].bool()
        current_pos = history_pos[:, :, -1].float()
        current_valid = history_mask[:, :, -1]
        B, num_agents, _ = current_pos.shape
        num_anchors = self.bda_sgcp.anchors.size(0)

        def current_feature(name, width, default=None):
            feature = agent_history.get(name)
            if feature is not None:
                return feature[:, :, -1].float()
            if default is not None:
                return default
            return current_pos.new_zeros(B, num_agents, width)

        current_velocity = current_feature('velocity', 2)
        current_acceleration = current_feature('acceleration', 2)
        if history_mask.size(-1) > 1:
            acceleration_valid = history_mask[:, :, -2:].all(dim=-1)
            acceleration_valid = acceleration_valid.unsqueeze(-1).to(current_acceleration.dtype)
            current_acceleration = current_acceleration * acceleration_valid
        current_size = current_feature('size', 3)
        current_type = current_feature('type', 3)

        current_heading = agent_history.get('heading')
        if current_heading is None:
            speed = current_velocity.norm(dim=-1, keepdim=True)
            direction = current_velocity / speed.clamp_min(1e-4)
            default_direction = torch.cat([torch.zeros_like(speed), torch.ones_like(speed)], dim=-1)
            # Convert (cos, sin) velocity direction to dataset order (sin, cos).
            current_heading = torch.stack([direction[..., 1], direction[..., 0]], dim=-1)
            current_heading = torch.where(
                (speed > 1e-4).expand_as(current_heading),
                current_heading,
                default_direction,
            )
        else:
            current_heading = current_heading[:, :, -1].float()

        if num_agents == 0:
            empty = current_pos.new_zeros(B, num_anchors, self.D)
            return empty.to(dtype=output_dtype), empty[..., :1].bool()

        target_idx = target_idx.to(device=current_pos.device, dtype=torch.long)
        current_valid = current_valid.clone()
        current_valid[torch.arange(B, device=current_pos.device), target_idx] = False

        num_neighbors = min(self.goal_flow_num_neighbors, num_agents)
        radius_sq = self.goal_flow_distance_threshold ** 2
        flow_chunks = []
        presence_chunks = []

        for anchor_start in range(0, num_anchors, self.goal_flow_anchor_chunk_size):
            anchor_end = min(anchor_start + self.goal_flow_anchor_chunk_size, num_anchors)
            anchor_pos = self.bda_sgcp.anchors[anchor_start:anchor_end].float()
            anchor_heading = self.goal_anchor_heading_sincos[anchor_start:anchor_end].float()
            chunk_size = anchor_pos.size(0)

            delta_all = current_pos[:, None] - anchor_pos[None, :, None]
            distance_sq = delta_all.square().sum(dim=-1)
            pair_valid = current_valid[:, None] & (distance_sq <= radius_sq)
            masked_distance_sq = distance_sq.masked_fill(~pair_valid, float('inf'))
            nearest_distance_sq, nearest_idx = masked_distance_sq.topk(
                num_neighbors, dim=-1, largest=False, sorted=True
            )
            nearest_valid = torch.isfinite(nearest_distance_sq)

            def gather_agents(feature):
                feature = feature[:, None].expand(-1, chunk_size, -1, -1)
                return torch.gather(
                    feature,
                    dim=2,
                    index=nearest_idx.unsqueeze(-1).expand(
                        -1, -1, -1, feature.size(-1)
                    ),
                )

            neighbor_pos = gather_agents(current_pos)
            neighbor_velocity = gather_agents(current_velocity)
            neighbor_acceleration = gather_agents(current_acceleration)
            neighbor_heading = gather_agents(current_heading)
            neighbor_size = gather_agents(current_size)
            neighbor_type = gather_agents(current_type)

            pose_heading = anchor_heading[None, :, None]
            local_pos = self.rotate_to_goal_frame(neighbor_pos - anchor_pos[None, :, None], pose_heading)
            local_velocity = self.rotate_to_goal_frame(neighbor_velocity, pose_heading)
            local_acceleration = self.rotate_to_goal_frame(neighbor_acceleration, pose_heading)

            agent_sin = neighbor_heading[..., 0]
            agent_cos = neighbor_heading[..., 1]
            goal_sin = pose_heading[..., 0]
            goal_cos = pose_heading[..., 1]
            relative_heading = torch.stack(
                [
                    agent_sin * goal_cos - agent_cos * goal_sin,
                    agent_cos * goal_cos + agent_sin * goal_sin,
                ],
                dim=-1,
            )
            neighbor_distance = nearest_distance_sq.clamp_min(0).sqrt()
            normalized_distance = (neighbor_distance / self.goal_flow_distance_threshold).unsqueeze(-1)

            local_state = torch.cat(
                [
                    local_pos,
                    local_velocity,
                    local_acceleration,
                    relative_heading,
                    neighbor_size,
                    neighbor_type,
                    normalized_distance,
                ],
                dim=-1,
            )
            local_state = local_state.masked_fill(~nearest_valid.unsqueeze(-1), 0.0).to(dtype=output_dtype)
            agent_token = self.goal_flow_state_encoder(local_state)

            score = self.goal_flow_agent_score(agent_token).squeeze(-1)
            score = score.float() - normalized_distance.squeeze(-1)
            score = score.masked_fill(~nearest_valid, float('-inf'))
            has_neighbor = nearest_valid.any(dim=-1, keepdim=True)
            safe_score = torch.where(has_neighbor, score, torch.zeros_like(score))
            agent_weight = safe_score.softmax(dim=-1).to(agent_token.dtype)
            agent_weight = agent_weight * nearest_valid.to(agent_token.dtype)
            agent_weight = agent_weight / agent_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            attention_pool = (agent_weight.unsqueeze(-1) * agent_token).sum(dim=-2)
            valid_weight = nearest_valid.to(agent_token.dtype).unsqueeze(-1)
            mean_pool = (agent_token * valid_weight).sum(dim=-2) / (
                valid_weight.sum(dim=-2).clamp_min(1.0)
            )
            max_pool = agent_token.masked_fill(
                ~nearest_valid.unsqueeze(-1),
                torch.finfo(agent_token.dtype).min,
            ).amax(dim=-2)
            max_pool = torch.where(has_neighbor, max_pool, torch.zeros_like(max_pool))
            occupancy = nearest_valid.sum(dim=-1, keepdim=True).to(agent_token.dtype) / float(num_neighbors)
            pooled_flow = torch.cat([attention_pool, mean_pool, max_pool, occupancy], dim=-1)
            flow_token = self.goal_flow_pool_proj(pooled_flow)
            flow_token = self.goal_flow_norm(flow_token)
            flow_token = flow_token * has_neighbor.to(flow_token.dtype)
            flow_chunks.append(flow_token)
            presence_chunks.append(has_neighbor)

        return torch.cat(flow_chunks, dim=1), torch.cat(presence_chunks, dim=1)

    def get_late_score_time_indices(self, device):
        stride = self.score_time_stride_late_layers
        if stride <= 1:
            return None
        return torch.arange(0, self.T, stride, device=device, dtype=torch.long)

    def goal_candidate_proposal(
        self,
        bev_feat,
        ec_dyn,
        tc_dyn,
        ego_dyn,
        agent_history,
        target_idx,
    ):
        # ====================== BDA anchor-token encoding ======================
        bda_token, bda_pos = self.bda_sgcp(bev_feat, ec_dyn, tc_dyn, ego_dyn)
        B = bda_token.size(0)

        # Anchor-local traffic flow remains a separate token stream. Each dense
        # anchor uses the precomputed arrival heading as its local +x direction.
        goal_flow_token, goal_flow_presence = self.build_goal_anchor_flow_tokens(
            agent_history, target_idx, output_dtype=bda_token.dtype
        )

        # Add the anchor position to each modality independently. This preserves
        # exact anchor identity without concatenating BDA and flow information.
        bda_pos_embed = gen_sineembed_for_position(
            bda_pos, hidden_dim=self.D, temperature=self.spa_pos_T
        ).to(dtype=bda_token.dtype)
        bda_key = (bda_token + bda_pos_embed).permute(1, 0, 2)
        bda_value = bda_token.permute(1, 0, 2)
        flow_key = (goal_flow_token + bda_pos_embed).permute(1, 0, 2)
        flow_value = goal_flow_token.permute(1, 0, 2)

        # ============== Independent cluster-restricted refinement =============
        bda_mode_query = self.goal_bda_query.expand(-1, B, -1)
        flow_mode_query = self.goal_flow_query.expand(-1, B, -1)

        # Exclude empty flow anchors. Fully empty mode rows use a safe mask and
        # are zeroed after attention to avoid all-masked softmax NaNs.
        flow_anchor_valid = goal_flow_presence.squeeze(-1)
        flow_attn_mask = self.goal_attn_mask[None] | ~flow_anchor_valid[:, None]
        flow_has_anchor = ~flow_attn_mask.all(dim=-1)
        safe_flow_attn_mask = torch.where(
            flow_has_anchor[..., None], flow_attn_mask, self.goal_attn_mask[None]
        )
        safe_flow_attn_mask = safe_flow_attn_mask[:, None].expand(-1, self.num_heads, -1, -1)
        safe_flow_attn_mask = safe_flow_attn_mask.reshape(
            B * self.num_heads, self.K, bda_token.size(1)
        )

        for layer in self.goal_proposal:
            bda_cross_out = layer['bda_cross_attn'](
                query=bda_mode_query,
                key=bda_key,
                value=bda_value,
                attn_mask=self.goal_attn_mask,
                need_weights=False,
            )[0]
            bda_mode_query = layer['bda_cross_norm'](bda_mode_query + bda_cross_out)
            bda_self_out = layer['bda_self_attn'](
                bda_mode_query, bda_mode_query, bda_mode_query,
                need_weights=False,
            )[0]
            bda_mode_query = layer['bda_self_norm'](bda_mode_query + bda_self_out)
            bda_mode_query = layer['bda_ffn_norm'](layer['bda_ffn'](bda_mode_query))

            flow_cross_out = layer['flow_cross_attn'](
                query=flow_mode_query,
                key=flow_key,
                value=flow_value,
                attn_mask=safe_flow_attn_mask,
                need_weights=False,
            )[0]
            flow_cross_out = flow_cross_out * flow_has_anchor.T[..., None].to(flow_cross_out.dtype)
            flow_mode_query = layer['flow_cross_norm'](flow_mode_query + flow_cross_out)
            flow_self_out = layer['flow_self_attn'](
                flow_mode_query, flow_mode_query, flow_mode_query,
                need_weights=False,
            )[0]
            flow_mode_query = layer['flow_self_norm'](flow_mode_query + flow_self_out)
            flow_mode_query = layer['flow_ffn_norm'](layer['flow_ffn'](flow_mode_query))

        # =================== Goal aggregation and output heads =================
        # Gather each mode's cluster before conditioning so scoring does not
        # materialize every mode against the full dense anchor bank.
        bda_mode_query_bk = bda_mode_query.permute(1, 0, 2)
        cluster_bda_token = bda_token[:, self.goal_anchor_indices]
        conditioned_bda_token = self.goal_bda_conditioner(
            cluster_bda_token, bda_mode_query_bk[:, :, None]
        )
        score_features = [self.goal_bda_score_proj(conditioned_bda_token)]

        flow_mode_query_bk = flow_mode_query.permute(1, 0, 2)
        cluster_flow_presence = goal_flow_presence[:, self.goal_anchor_indices]
        cluster_flow_token = goal_flow_token[:, self.goal_anchor_indices]
        conditioned_flow_token = self.goal_flow_conditioner(
            cluster_flow_token, flow_mode_query_bk[:, :, None]
        )
        flow_presence = cluster_flow_presence.to(conditioned_flow_token.dtype)
        conditioned_flow_token = conditioned_flow_token * flow_presence
        flow_score_feature = self.goal_flow_score_proj(conditioned_flow_token)
        score_features.append(flow_score_feature * flow_presence)

        goal_logits = self.goal_score_fusion(torch.cat(score_features, dim=-1)).squeeze(-1)
        goal_probability = goal_logits.float().softmax(dim=-1)
        goal_anchor_position = bda_pos[:, self.goal_anchor_indices]

        # Forward selection is exactly the argmax anchor. The soft component of
        # this straight-through weight lets downstream trajectory losses update
        # the new anchor scorer through the selected-token representation.
        local_goal_idx = goal_probability.argmax(dim=-1)
        hard_selection = F.one_hot(
            local_goal_idx, num_classes=goal_probability.size(-1)
        ).to(goal_probability.dtype)
        selection_weight = hard_selection + goal_probability - goal_probability.detach()
        goal_position = goal_anchor_position.gather(
            dim=2,
            index=local_goal_idx[:, :, None, None].expand(-1, -1, 1, 2),
        ).squeeze(2).permute(1, 0, 2).contiguous()

        selected_bda_token = (
            selection_weight.to(conditioned_bda_token.dtype).unsqueeze(-1)
            * conditioned_bda_token
        ).sum(dim=2)
        selected_mode_features = [bda_mode_query_bk, selected_bda_token]

        # Use flow in the downstream query only when the selected anchor has flow.
        flow_selection_weight = selection_weight.to(conditioned_flow_token.dtype)
        selected_flow_gate = (flow_selection_weight.unsqueeze(-1) * flow_presence).sum(dim=2)
        selected_flow_token = (
            flow_selection_weight.unsqueeze(-1) * conditioned_flow_token
        ).sum(dim=2)
        gated_flow_mode_query = flow_mode_query_bk * selected_flow_gate
        gated_selected_flow_token = selected_flow_token * selected_flow_gate
        selected_mode_features.extend([gated_flow_mode_query, gated_selected_flow_token])

        selected_mode_delta = self.goal_selected_mode_fusion(
            torch.cat(selected_mode_features, dim=-1)
        )
        mode_query = self.goal_selected_mode_norm(bda_mode_query_bk + selected_mode_delta)
        mode_query = mode_query.permute(1, 0, 2).contiguous()

        goal_FDE = self.goal_FDE(mode_query).squeeze(-1).T
        return mode_query, goal_position, goal_probability, goal_anchor_position, goal_FDE

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
        time_indices=None,
    ):
        if time_indices is not None:
            dec_embed = dec_embed.index_select(2, time_indices)
            pred_traj = pred_traj.index_select(2, time_indices)
            pred_vel = pred_vel.index_select(2, time_indices)
            if dense_future_pred is not None:
                dense_future_pred = dense_future_pred.index_select(2, time_indices)

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
        predicted_goal_position,
        ego_dyn,
        tc_dyn,
        scene_context_tokens=None,
        dense_future_pred=None,
        obj_valid_mask=None,
        target_idx=None,
    ):
        M, B, _ = mode_query.shape

        # ===================== mode localization branch =====================
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

        # Select the trajectory template whose endpoint is closest to each goal.
        endpoint_distance = (
            predicted_goal_position[:, :, None]
            - self.trajectory_set[None, None, :, -1]
        ).square().sum(dim=-1)
        trajectory_idx = endpoint_distance.argmin(dim=-1)
        ref_trajectory = self.trajectory_set[trajectory_idx]  # [B, M, L, 2]

        # BEV references use the ego frame.
        steps_per_waypoint = self.T // self.num_waypoints
        ref_waypoints = ref_trajectory[:, :, steps_per_waypoint - 1::steps_per_waypoint]
        ref_waypoints = target_to_ego(
            ref_waypoints.reshape(B, M * self.num_waypoints, 2),
            trans_x,
            trans_y,
            rot_sin,
            rot_cos,
        ).reshape(B, M, self.num_waypoints, 2).permute(1, 0, 2, 3)

        # Identify waypoint queries with the time PE at the end of each temporal
        # block (with the default 12 waypoints: 0.5s, ..., 6.0s).
        waypoint_time_pe = self.build_time_pe(B, M, mode_embed.dtype)[
            steps_per_waypoint - 1::steps_per_waypoint
        ]
        waypoint_time_pe = waypoint_time_pe.reshape(
            self.num_waypoints, B, M, self.D
        ).permute(2, 1, 0, 3)
        mode_embed = mode_embed.unsqueeze(2) + waypoint_time_pe
        query_scale = self.get_query_scale_itp(mode_embed)

        mode_embed = self.norm_l1[1](
            self.bev_cross_attn_l1(
                dec_embed=mode_embed,
                bev_feat=bev_feat,
                query_scale=query_scale,
                ref_points=ref_waypoints,
            )
        )
        mode_embed = self.norm_l1[2](self.ffn_l1(mode_embed))  # [M,B,12,D]

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
        mode_bt = mode_embed.permute(1, 0, 2, 3).unsqueeze(3)
        state_bt = state_query.permute(1, 0, 2).reshape(
            B, 1, self.num_waypoints, steps_per_waypoint, self.D
        )

        dec_embed_T = mode_bt + state_bt  # [B,M,12,5,D]
        dec_embed_T = dec_embed_T.reshape(B, M, self.T, self.D)
        dec_embed_T = dec_embed_T.permute(1, 0, 2, 3).contiguous()

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

        dense_future_pred = kwargs.get('dense_future_pred')
        obj_valid_mask = kwargs.get('obj_valid_mask')
        target_idx = kwargs['target_idx']
        agent_history = kwargs['agent_history']

        scene_context_repeat = scene_context.unsqueeze(2).repeat(1, 1, self.T, 1)
        scene_context_repeat = scene_context_repeat.permute(1, 0, 2, 3).reshape(n, B * self.T, -1)
        scene_context = scene_context.permute(1, 0, 2)

        # -------------------Goal Candidate Proposal -----------------
        (
            mode_query,
            goal_position,
            goal_probability,
            goal_anchor_position,
            goal_FDE,
        ) = \
            self.goal_candidate_proposal(
                bev_feat,
                ec_dyn,
                tc_dyn,
                ego_dyn,
                agent_history,
                target_idx,
            )
        predicted_goal_position = goal_position.permute(1, 0, 2).contiguous()
        predicted_goal_position_detached = predicted_goal_position.detach()

        # -------------------- Initial Prediction --------------------
        dec_embed, init_mode_prob, init_pred_traj, init_pred_vel, state_pred = \
            self.initial_prediction(
                mode_query,
                scene_context,
                bev_feat,
                predicted_goal_position_detached,
                ego_dyn,
                tc_dyn,
                scene_context_tokens=scene_context_tokens,
                dense_future_pred=dense_future_pred,
                obj_valid_mask=obj_valid_mask,
                target_idx=target_idx,
            )
        
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
        late_score_time_indices = self.get_late_score_time_indices(dec_embed.device)
        for layer_idx, layer in enumerate(self.dec_layers):
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
            refinement_offset = self.smooth_refinement_offset(
                pred_traj_raw[..., :2]
            )
            pred_xy = refinement_offset + ref_points            # out-of-place
            pred_traj = torch.cat([pred_xy, pred_traj_raw[..., 2:]], dim=-1)
            ref_points = pred_xy.detach().clone()

            pred_vel = self.motion_vel(dec_embed) # [K, B, T, 2]
            score_time_indices = None
            if layer_idx >= self.score_time_full_refine_layers:
                score_time_indices = late_score_time_indices
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
                time_indices=score_time_indices,
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
                  'predicted_goal_position': predicted_goal_position,
                  'predicted_goal_probability': goal_probability,
                  'goal_anchor_position': goal_anchor_position,
                  'predicted_goal_FDE': goal_FDE,
                #   'init_top_idx': init_top_idx,                # [B, K]
                  'state_pred': state_pred, # [B, T, 2]
                }
        return output
    

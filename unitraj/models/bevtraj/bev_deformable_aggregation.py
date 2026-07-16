import math
import copy
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.bevtraj.linear import build_mlp, MLP, FFN
from unitraj.models.bevtraj.utility import gen_sineembed_for_position, target_to_ego
from unitraj.models.bevtraj.temporal_sequential_module import TemporalMHA


MODEL_DIR = Path(__file__).resolve().parent


class QueryConditionedDynamics(nn.Module):
    """
    dynamics: [batch_size, query_num, dyn_dim]
    query_emb: [batch_size, query_num, query_dim]
    FiLM: Feature-wise Linear Modulation
    
    """
    def __init__(self, query_dim, hidden_dim):
        super().__init__()
        
        self.modulator = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        
        # Initialize the last layer to produce zero modulation at the beginning
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)
        
        
    def forward(self, dynamics, query_emb):
        B, N, _ = query_emb.shape
        gamma_beta = self.modulator(query_emb) #(B, N, 2 * hidden_dim)
        gamma, beta = gamma_beta.chunk (2, dim=-1) #(B, N, hidden_dim) each
        conditioned = (1 + gamma) * dynamics + beta #(B, N, hidden_dim)
        return conditioned


class DeformAttn(nn.Module):
    def __init__(self, config, d_model, grid_size):
        super(DeformAttn, self).__init__()
        
        self.config = config
        self.D = d_model
        self.n_heads = config['num_heads']
        self.n_points = config['num_key_points']
        self.head_dims = self.D // self.n_heads
        
        self.sampling_offsets = build_mlp(self.D, self.D, self.n_heads * self.n_points * 2, dropout=0.0)
        self.attn_weights = build_mlp(self.D * 2, self.D, self.n_heads * self.n_points, dropout=0.0)
        self.value_proj = nn.Conv2d(self.head_dims, self.head_dims, kernel_size=1)
        self.output_proj = build_mlp(self.D * 2, self.D, self.D, dropout=0.0)
        
        self.register_buffer('offset_normalizer', torch.tensor(grid_size, dtype=torch.float32))
        
    def forward(self, query_content, query_pos, ref_pos, bev_feat):
        B, _, H, W = bev_feat.shape
        N = query_content.shape[1]

        value = bev_feat.reshape(B, self.n_heads, -1, H, W).reshape(B*self.n_heads, -1, H, W)
        value = self.value_proj(value)

        sampling_offsets = self.sampling_offsets(query_pos).reshape(B, N, self.n_heads, self.n_points, 2).permute(0, 2, 1, 3, 4)
        sampling_locations = ref_pos.unsqueeze(1).unsqueeze(3) + sampling_offsets / self.offset_normalizer[None, None, None, None, :]
        sampling_locations = sampling_locations.reshape(B*self.n_heads, N, self.n_points, 2)

        sampling_locations[..., 1] = sampling_locations[..., 1] * -1 # Align with F.grid_sample coordinate system

        sampled_feature = F.grid_sample(value, sampling_locations, align_corners=False, mode='bilinear')
        sampled_feature = sampled_feature.reshape(B, self.n_heads, -1, N, self.n_points).permute(0, 1, 3, 4, 2)

        attn_weights = self.attn_weights(torch.cat([query_content, query_pos], dim=-1)).reshape(B, N, self.n_heads, self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.permute(0, 2, 1, 3).unsqueeze(-1)

        attn_outputs = torch.sum(sampled_feature * attn_weights, dim=3)
        attn_outputs = attn_outputs.permute(0, 2, 1, 3).reshape(B, N, -1)

        output = self.output_proj(torch.cat([attn_outputs, query_content], dim=-1))
        return output
        

class BDALayer(nn.Module):
    def __init__(self, config, d_model, grid_size):
        super(BDALayer, self).__init__()
        
        self.config = config
        self.D = d_model
        self.dropout = config['dropout']
        
        self.self_attn = nn.MultiheadAttention(self.D, config['num_heads'], dropout=self.dropout, batch_first=True)
        self.cross_attn = DeformAttn(config['deform_attn'], self.D, grid_size)
        self.ffn = FFN(self.D, config['ffn_dims'], num_fcs=2, dropout=self.dropout)
        
        self.norm_layers= nn.ModuleList([nn.LayerNorm(self.D) for _ in range(3)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(self.dropout) for _ in range(2)])
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    
class BDALayer_ENC(BDALayer):
    def __init__(self, config, d_model, grid_size):
        super(BDALayer_ENC, self).__init__(config, d_model, grid_size)

    def forward(self, ba_query, query_pos, ref_pos, bev_feat):
        tgt = self.with_pos_embed(ba_query, query_pos)
        tgt, _ = self.self_attn(tgt, tgt, ba_query)
        tgt = self.norm_layers[0](ba_query + self.dropout_layers[0](tgt))
        
        tgt2 = self.cross_attn(tgt, query_pos, ref_pos, bev_feat)
        tgt2 = self.norm_layers[1](tgt + self.dropout_layers[1](tgt2))
        
        output = self.norm_layers[2](self.ffn(tgt2))
        return output
    

class BDALayer_DEC(BDALayer):
    def __init__(self, config, d_model, grid_size):
        super(BDALayer_DEC, self).__init__(config, d_model, grid_size)

        self.pos_scale = build_mlp(self.D, self.D, self.D, dropout=self.dropout)
        self.query_pos = build_mlp(self.D, self.D, self.D, dropout=self.dropout)
    
    def forward(self, ba_query, query_sine_embed, ref_pos, bev_feat, lid):
        # Self-Attention
        pos_scale = self.pos_scale(ba_query) if lid != 0 else 1
        self_query_pos = pos_scale * self.query_pos(query_sine_embed)
        
        tgt = self.with_pos_embed(ba_query, self_query_pos)
        tgt, _ = self.self_attn(tgt, tgt, ba_query)
        tgt = self.norm_layers[0](ba_query + self.dropout_layers[0](tgt))
        
        # Cross-Attention
        cross_pos_scale = self.pos_scale(tgt)
        cross_query_pos = cross_pos_scale * self.query_pos(query_sine_embed)

        tgt2 = self.cross_attn(tgt, cross_query_pos, ref_pos, bev_feat)
        tgt2 = self.norm_layers[1](tgt + self.dropout_layers[1](tgt2))
        
        output = self.norm_layers[2](self.ffn(tgt2))
        return output
    
    
class BEVDeformableAggregation(nn.Module, ABC):
    def __init__(self, config, d_model):
        super(BEVDeformableAggregation, self).__init__()
        
        self.config = config
        self.D = d_model
        self.dropout = config['dropout']
        self.num_ba_query = config['num_ba_query']
        self.grid_size = config['grid_size']
        self.refine_share_param = config['refine_share_param']
        
        self.ba_query = nn.Parameter(torch.zeros(self.num_ba_query, self.D), requires_grad=True)
            
        self.register_buffer('denorm_scale', torch.tensor(self.grid_size, dtype=torch.float32))
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class BDA_ENC(BEVDeformableAggregation):
    def __init__(self, config, d_model):
        super(BDA_ENC, self).__init__(config, d_model)

        self.use_anchor_points = config['use_anchor_points']
        self.anchor_file_name = config['anchor_file_name']

        if self.use_anchor_points:
            file_path = MODEL_DIR / self.anchor_file_name
            with open(file_path, 'rb') as f:
                anchors = pickle.load(f)
            self.register_buffer('anchors', torch.from_numpy(anchors['VEHICLE']).float())
        else:
            self.ref_pos = nn.Parameter(self.create_uniform_2d_grid_tensor(self.num_ba_query), requires_grad=True)

        self.pos_scale = build_mlp(self.D, self.D, self.D, dropout=self.dropout)
        self.query_pos = build_mlp(self.D, self.D, self.D, dropout=self.dropout)

        self.bda_layers = nn.ModuleList([
            BDALayer_ENC(self.config['bda_layer'], self.D, self.grid_size)
            for _ in range(self.config['num_bda_layers'])
        ])

    def create_uniform_2d_grid_tensor(self, n_points):
        side = int(n_points ** 0.5)
        if side ** 2 != n_points:
            raise ValueError("n_points == n * n")

        x = torch.linspace(-1, 1, side)
        y = torch.linspace(-1, 1, side)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([xx, yy], dim=-1)
        return grid.reshape(-1, 2)

    def place_template_points(self, tc_pos, obj_pos, obj_heading):
        """
        Place template points around each object by applying its pose (position + heading).

        Args:
            tc_pos (torch.Tensor): (B, A, K, 2) (target-centric coordinates)
            obj_pos (torch.Tensor): (B, A, 2) (x, y)
            obj_heading (torch.Tensor): (B, A, 2) (sin, cos)

        Returns:
            new_pos (torch.Tensor): (B, A, K, 2) template positions for each object
        """
        sin, cos = obj_heading[..., 0], obj_heading[..., 1]  # (B,A)

        rot_mat = torch.stack([
            torch.stack([cos, sin], dim=-1),
            torch.stack([-sin, cos], dim=-1)
        ], dim=-2)   # (B,A,2,2)

        rotated = torch.matmul(tc_pos, rot_mat)  # (B,A,K,2)
        new_pos = rotated + obj_pos[:, :, None, :]   # (B,A,K,2)

        return new_pos
    
    def forward(self, traj_data, bev_feat, ego_dyn):
        # create reference points based on object positions and headings
        if self.use_anchor_points:
            obj_pos = traj_data['obj_trajs'][:, :8, -1, 0:2] # x, y
            obj_heading = traj_data['obj_trajs'][:, :8, -1, -6:-4] # sin, cos
            
            B, A, _ = obj_pos.shape
            K = self.anchors.shape[0]
            
            anchor_pos = self.anchors[None, None, :, :]
            anchor_pos = anchor_pos.repeat(B, A, 1, 1)
            ref_pos_target = self.place_template_points(anchor_pos, obj_pos, obj_heading).reshape(B, A*K, 2) # (B, 256, 2)

            trans_x, trans_y, rot_sin, rot_cos = (
                ego_dyn['ego_x'],
                ego_dyn['ego_y'],
                ego_dyn['ego_sin'],
                ego_dyn['ego_cos'],
            )
            # ref_pos_ego = target_to_ego(ref_pos_target, trans_x, trans_y, rot_sin, rot_cos)
            # ref_pos = ref_pos_ego / self.denorm_scale[None, None, :] # normalize
            ref_pos = target_to_ego(ref_pos_target, trans_x, trans_y, rot_sin, rot_cos)
            ref_pos_norm = ref_pos / self.denorm_scale[None, None, :]
        else:
            B = bev_feat.shape[0]
            ref_pos_norm = self.ref_pos[None].repeat(B, 1, 1)
            ref_pos = ref_pos_norm * self.denorm_scale[None, None, :]
        
        # BEV Deformable Aggregation
        output = self.ba_query[None].repeat(B, 1, 1)
        query_sine_embed = gen_sineembed_for_position(ref_pos, hidden_dim=self.D, temperature=10000)
        pos_q = self.query_pos(query_sine_embed)

        for lid, layer in enumerate(self.bda_layers):
            query_pos = self.pos_scale(output) * pos_q
            output = layer(output, query_pos, ref_pos_norm, bev_feat)
            
        return output, ref_pos


class BDA_DEC(BEVDeformableAggregation):
    def __init__(self, config, d_model):
        super(BDA_DEC, self).__init__(config, d_model)

        self.t = config['past_len']
        self.t_D = config['t_dims']

        self.dynamics_enc = nn.ModuleDict({
            'tc': MLP(self.config['target_attr'], self.D, self.D, 2),
            'tc_q': QueryConditionedDynamics(self.D, self.D),
            'motion_fuse': MLP(self.D + self.D, self.D, self.D, 3),
        })

        self.anchor_enc = nn.ModuleDict({
            'tc': MLP(self.D, self.D, self.D, 2),
            'rel': MLP(8, self.D, self.D, 2),
        })
        self.tc_temporal_mha = TemporalMHA(self.D, self.config['bda_layer']['num_heads'], self.dropout)
        self.tc_temporal_norm = nn.LayerNorm(self.D)
        self.tc_anchor_attn = nn.MultiheadAttention(self.D, self.config['bda_layer']['num_heads'], dropout=self.dropout, batch_first=True)
        self.tc_anchor_norm = nn.LayerNorm(self.D)
        self.history_time_embedding_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, self.D),
        )
        self.register_buffer('history_time', torch.arange(self.t).float().unsqueeze(-1))
        self.history_dt = config.get('dt', 0.1)
        self.history_time_alpha = nn.Parameter(torch.tensor(1.0))
        self.motion_to_query_pos = MLP(self.D, self.D, self.D, 2)
        self.motion_gate = MLP(self.D + self.D, self.D, self.D, 2)

        self.bda_layers = nn.ModuleList([
            BDALayer_DEC(self.config['bda_layer'], self.D, self.grid_size)
            for _ in range(self.config['num_bda_layers'])
        ])

        self.anchor_file_name = config['anchor_file_name']
        file_path = MODEL_DIR / self.anchor_file_name
        with open(file_path, 'rb') as f:
            anchors = pickle.load(f)
        self.register_buffer('anchors', torch.from_numpy(anchors['VEHICLE']).float())

        # Keep one BDA token per dense anchor. SGCP later compresses this bank into
        # the configured number of trajectory modes.
        self.num_ba_query = self.anchors.shape[0]
        self.ba_query_dec = nn.Parameter(
            torch.zeros(self.num_ba_query, self.D), requires_grad=True
        )

    def build_anchor_relation_feat(self, ref_pos_target, tc_dyn):
        eps = 1e-3

        dx = ref_pos_target[..., 0]
        dy = ref_pos_target[..., 1]
        r = torch.sqrt(dx.square() + dy.square() + eps)
        sin_theta = dy / r
        cos_theta = dx / r

        target_vel = tc_dyn[:, -1, 2:4]
        speed = torch.norm(target_vel, dim=-1, keepdim=True)
        heading = target_vel / speed.clamp_min(eps)
        heading = heading.unsqueeze(1).expand(-1, self.num_ba_query, -1)
        heading_perp = torch.stack([-heading[..., 1], heading[..., 0]], dim=-1)

        longi = (ref_pos_target * heading).sum(dim=-1)
        lateral = (ref_pos_target * heading_perp).sum(dim=-1)
        time_to_reach = r / speed.clamp_min(eps)

        rel_feat = torch.stack([
            dx, dy, r,
            sin_theta, cos_theta,
            longi, lateral,
            time_to_reach.squeeze(-1),
        ], dim=-1)
        return rel_feat

    def build_history_time_pe(self, batch_size, dtype, device):
        t = (self.history_time * self.history_dt).to(device=device, dtype=dtype)
        pe = self.history_time_embedding_mlp(t)  # [t, D]
        pe = pe.unsqueeze(1).repeat(1, batch_size, 1)  # [t, B, D]
        return self.history_time_alpha * pe

    def forward(self, bev_feat, ec_dyn, tc_dyn, ego_dyn):
        B = bev_feat.shape[0]
        # output = self.ba_query[None].repeat(B, 1, 1)
        output = self.ba_query_dec[None].repeat(B, 1, 1) # kong_fixme

        ref_pos_target = self.anchors[None].repeat(B, 1, 1)

        trans_x, trans_y, rot_sin, rot_cos = (
            ego_dyn['ego_x'],
            ego_dyn['ego_y'],
            ego_dyn['ego_sin'],
            ego_dyn['ego_cos'],
        )
        ref_pos = target_to_ego(ref_pos_target, trans_x, trans_y, rot_sin, rot_cos)

        # =============================== prototype ================================
        # tc features build query semantics, while ec coordinates remain responsible
        # for BEV lookup through ref_pos. Before encoding anchor positions, explicitly
        # describe how each anchor relates to current target motion.

        tc_step_feat = self.dynamics_enc['tc'](tc_dyn)  # (B, t, D)
        tc_time_pe = self.build_history_time_pe(B, tc_step_feat.dtype, tc_step_feat.device)
        tc_step_feat_temporal = self.tc_temporal_mha(
            tc_step_feat.permute(1, 0, 2),
            tc_time_pe,
        ).permute(1, 0, 2)
        tc_step_feat = self.tc_temporal_norm(tc_step_feat + tc_step_feat_temporal)

        anchor_tc = gen_sineembed_for_position(ref_pos_target, hidden_dim=self.D, temperature=10000)
        anchor_feat = self.anchor_enc['tc'](anchor_tc)

        anchor_rel_feat = self.build_anchor_relation_feat(ref_pos_target, tc_dyn)
        anchor_rel_feat = self.anchor_enc['rel'](anchor_rel_feat)
        anchor_feat = anchor_feat + anchor_rel_feat

        tc_d = self.tc_anchor_attn(
            query=anchor_feat,
            key=tc_step_feat,
            value=tc_step_feat,
        )[0]
        tc_d = self.tc_anchor_norm(tc_d + anchor_feat)
        tc_d = self.dynamics_enc['tc_q'](tc_d, anchor_feat) # (B, num_ba_query, D)
        motion_feat = self.dynamics_enc['motion_fuse'](torch.cat([tc_d, anchor_feat], dim=-1))
        output = output + motion_feat

        ref_pos_norm = ref_pos / self.denorm_scale[None, None, :]
        motion_bias = self.motion_to_query_pos(motion_feat)
        motion_gate = torch.sigmoid(self.motion_gate(torch.cat([anchor_feat, motion_feat], dim=-1)))
        query_sine_embed = anchor_feat + motion_gate * motion_bias
        # ==========================================================================

        for lid, layer in enumerate(self.bda_layers):
            output = layer(output, query_sine_embed, ref_pos_norm, bev_feat, lid)
            
        return output, ref_pos_target

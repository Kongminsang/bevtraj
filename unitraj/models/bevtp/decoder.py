import math
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from unitraj.models.bevtp.decoder_deform_attn2d import DeformableCrossAttention2D_Q, DeformableCrossAttention2D_K, DeformableCrossAttention2D_layer1
from unitraj.models.bevtp.linear import MLP, FFN, MotionRegHead, MotionClsHead
from unitraj.models.bevtp.positional_encoding_utils import gen_sineembed_for_position


class TemporalPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, future_len=12, temperature=500.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.T = future_len
        pe = torch.zeros(future_len, d_model)
        position = torch.arange(0, future_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(temperature) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class BEVTPDecoderLayer(nn.Module):
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
        self.temp_self_attn = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(self.D, self.num_heads,
                                                                    dim_feedforward=self.ffn_D, dropout=self.dropout)
        self.bev_cross_attn = DeformableCrossAttention2D_Q(**config['deform_cross_attn_query'])
        self.ffn = FFN(self.D, self.ffn_D, 2)
    
    def forward(self, dec_embed, scene_context, bev_feat, query_scale, ref_points, ego_dynamics):
        '''
        Args:
            dec_embed: [T, B*K, D]
            scene_context: [t, B, D]
            bev_feat: [B, D, H, W]
            query_scale: [T, B*K, d]
            ref_points: [K, B, T, 2]
        '''
        B = bev_feat.size(0)
        scene_context = scene_context 
        
        # ============================== target-centric(tc) modeling ==============================
        
        dec_embed = self.norm[0](self.temp_self_attn(query=dec_embed, key=dec_embed, value=dec_embed)[0] + dec_embed)
        
        # get positional query
        query_sine_embed = gen_sineembed_for_position(ref_points, hidden_dim=self.Q_D, temperature=self.spa_pos_T)
        tc_pos_Q = self.to_pos_Q(query_sine_embed)
        
        dec_embed, query_scale = map(lambda t: t.reshape(self.T, B, self.K, -1).permute(2, 1, 0, 3), (dec_embed, query_scale))
        dec_embed = dec_embed + tc_pos_Q
        dec_embed = self.transformer_decoder_layer(tgt=dec_embed.reshape(self.K, B*self.T, -1),
                                                   memory=scene_context).reshape(self.K, B, self.T, -1)
        
        # ============================== ego-centric(ec) modeling ==============================
        
        # coord transform
        ego_loc, ego_sin, ego_cos = ego_dynamics['ego_loc'], ego_dynamics['ego_sin'], ego_dynamics['ego_cos']
        ego_loc, ego_sin, ego_cos = map(lambda t: t.unsqueeze(1).unsqueeze(0).repeat(self.K, 1, self.T, 1), (ego_loc, ego_sin, ego_cos)) # (K, B, T, _)
        
        rotation_matrix = torch.stack([
            torch.cat([ego_cos, -ego_sin], dim=-1),
            torch.cat([ego_sin, ego_cos], dim=-1)
        ], dim=-2)
        
        ref_points = ref_points - ego_loc
        ref_points = torch.matmul(ref_points.unsqueeze(-2), rotation_matrix).squeeze(-2) # (K, B, T, 2)
        
        # adequate coord system for F.grid_sample in bev_cross_attn
        ref_points[..., 1] *= -1 # (BEV feature coordinates have inverted y-axis compared to unitraj)
        
        # cross attn with bev feature
        dec_embed = self.norm[1](self.bev_cross_attn(dec_embed, bev_feat, query_scale, ref_points))
        dec_embed = self.norm[2](self.ffn(dec_embed))
        
        return dec_embed
        
        
class BEVTPDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.t = config['past_len']
        self.T = config['future_len']
        
        self.D = config['d_model']
        self.ffn_D = config['ffn_dims']
        self.t_D = config['t_dims']
        self.T_D = config['T_dims']
        
        self.K = config['num_modes']
        self.target_attr = config['target_attr']
        self.query_scale_dims = config['query_scale_dims']
        self.tem_pos_T = config['tem_pos_T']
        self.spa_pos_T = config['spa_pos_T']
        
        self.dropout = config['dropout']
        self.L_goal_proposal = config['num_goal_proposal_layers']
        self.L_dec = config['num_decoder_layers']
        self.num_heads = config['num_heads']
        
        self.dca_k_cfg = config['deform_cross_attn_key']
        self.dca_q_cfg = config['deform_cross_attn_query']
        self.dca_k_cfg['dim'] = self.dca_q_cfg['dim'] = self.D
        
        self.dec_layer_config = {
            'future_len': self.T,
            'd_model': self.D,
            'query_dims': self.query_scale_dims,
            'ffn_dims': self.ffn_D,
            'spa_pos_T': self.spa_pos_T,
            'num_modes': self.K,
            'dropout': self.dropout,
            'num_heads': self.num_heads,
            'deform_cross_attn_query': self.dca_q_cfg,
        }
        
        self.Q = nn.Parameter(torch.empty(self.K, 1, self.D), requires_grad=True)
        nn.init.xavier_uniform_(self.Q)
        
        # ============================== Goal Candidate Proposal ==============================
        
        self.ec_dynamic_encoder = nn.ModuleDict({
            'enc': MLP(self.target_attr, self.D, self.t_D, 2),
            'enc_t': MLP(self.t_D * self.t, self.D//2, self.D//2, 2),
            'enc_Q': MLP(self.D//2 + self.D, self.D, self.D, 1),
        })
        
        self.tc_dynamic_encoder = nn.ModuleDict({
            'enc': MLP(self.target_attr, self.D, self.t_D, 2),
            'enc_t': MLP(self.t_D * self.t, self.D//2, self.D//2, 2),
            'enc_Q': MLP(self.D//2 + self.D, self.D, self.D, 1),
        })
        
        self.goal_proposal = []
        for _ in range(self.L_goal_proposal):
            goal_proposal_layer = nn.ModuleDict({
                'deform_cross_attn_key': DeformableCrossAttention2D_K(**self.dca_k_cfg),
                'norm1': nn.LayerNorm(self.D),
                'mode_self_attn': nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout),
                'norm2': nn.LayerNorm(self.D),
                'ffn': FFN(self.D, self.ffn_D, 2),
                'norm3': nn.LayerNorm(self.D),
            })
            self.goal_proposal.append(goal_proposal_layer)
        self.goal_proposal = nn.ModuleList(self.goal_proposal)
            
        self.goal_proposal_reg = MLP(self.D, self.D, 2, 2)
        self.goal_proposal_FDE = MLP(self.D, self.D, 1, 2)
        
        # ============================== Trajectory Initial Prediction ==============================
        
        self.get_query_scale_l1 = MLP(self.D, self.query_scale_dims, self.query_scale_dims, 2)
        self.norm_l1 = nn.ModuleList([nn.LayerNorm(self.D) for _ in range(3)])
        
        self.context_cross_attn_l1 = nn.MultiheadAttention(self.D, self.num_heads, dropout=self.dropout)
        self.bev_cross_attn_l1 = DeformableCrossAttention2D_layer1(**self.dca_q_cfg)
        self.ffn_l1 = FFN(self.D, self.ffn_D, 2)
        
        self.tmp_MLP = nn.ModuleList([
            nn.Sequential(nn.Linear(self.D, self.T_D * self.T), nn.GELU()),
            nn.Sequential(nn.Linear(self.T_D, self.D), nn.GELU())
        ])
        self.motion_cls_l1 = MotionClsHead(self.D, self.T_D, self.T)
        self.motion_reg_l1 = MotionRegHead(self.D)
        
        # ============================== Trajectory Iterative Refinement ==============================
        
        self.temp_pos_enc = TemporalPositionalEncoding(self.D, self.dropout, future_len=self.T, temperature=self.tem_pos_T)
        self.get_query_scale_T = MLP(self.query_scale_dims, self.query_scale_dims, self.query_scale_dims, 2)
        
        dec_layer = BEVTPDecoderLayer(self.dec_layer_config)
        self.dec_layers = nn.ModuleList([copy.deepcopy(dec_layer) for _ in range(self.L_dec - 1)])
        
        self.motion_cls = MotionClsHead(self.D, self.T_D, self.T)
        self.motion_reg = MotionRegHead(self.D)
        
    def goal_candidate_proposal(self, ec_dynamics, tc_dynamics, bev_feat):
        B = ec_dynamics.shape[0]
        ec_dynamics = self.ec_dynamic_encoder['enc'](ec_dynamics).reshape(B, -1)
        ec_dynamics = self.ec_dynamic_encoder['enc_t'](ec_dynamics).unsqueeze(0).repeat(self.K, 1, 1)
        mode_query = self.ec_dynamic_encoder['enc_Q'](torch.cat([ec_dynamics, self.Q.repeat(1, B, 1)], dim=-1))
        
        tc_dynamics = self.tc_dynamic_encoder['enc'](tc_dynamics).reshape(B, -1)
        tc_dynamics = self.tc_dynamic_encoder['enc_t'](tc_dynamics).unsqueeze(0).repeat(self.K, 1, 1)
        
        for i, layer in enumerate(self.goal_proposal):
            mode_query = layer['norm1'](layer['deform_cross_attn_key']( \
                    mode_query = mode_query, bev_feat = bev_feat))
            if i == 0:
                mode_query = self.tc_dynamic_encoder['enc_Q'](torch.cat([tc_dynamics, mode_query], dim=-1))
                
            mode_query = layer['norm2'](layer['mode_self_attn']( \
                    query = mode_query, key = mode_query, value = mode_query)[0] + mode_query)
            mode_query = layer['norm3'](layer['ffn'](mode_query))
            
        goal_reg = self.goal_proposal_reg(mode_query)
        goal_FDE = self.goal_proposal_FDE(mode_query).squeeze(dim=-1).T
        
        return mode_query, goal_reg, goal_FDE

    def initial_prediction(self, mode_query, scene_context, bev_feat, goal_candidate, ego_dynamics):
        K, B, _ = mode_query.shape
        query_scale = self.get_query_scale_l1(mode_query)
        dec_embed = self.norm_l1[0](self.context_cross_attn_l1(query=mode_query, key=scene_context, 
                                                                     value=scene_context)[0] + mode_query)
        
        # coord transform of goal candidates (target-centric -> ego-centric)
        ego_loc, ego_sin, ego_cos = ego_dynamics['ego_loc'], ego_dynamics['ego_sin'], ego_dynamics['ego_cos']
        ego_loc, ego_sin, ego_cos = map(lambda t: t.unsqueeze(0).repeat(K, 1, 1), (ego_loc, ego_sin, ego_cos))
        
        rotation_matrix = torch.stack([
            torch.cat([ego_cos, -ego_sin], dim=-1),
            torch.cat([ego_sin, ego_cos], dim=-1)
        ], dim=-2)
        
        goal_candidate = goal_candidate - ego_loc
        goal_candidate = torch.matmul(goal_candidate.unsqueeze(-2), rotation_matrix).squeeze(-2)
        
        # adequate coord system for F.grid_sample in bev_cross_attn
        goal_candidate[..., 1] *= -1 # (BEV feature coordinates have inverted y-axis compared to unitraj)
        
        # cross attn with scene feature
        dec_embed = self.norm_l1[1](self.bev_cross_attn_l1(dec_embed=dec_embed, bev_feat=bev_feat,
                                                                 query_scale = query_scale, ref_points = goal_candidate))
        dec_embed = self.norm_l1[2](self.ffn_l1(dec_embed))
        
        dec_embed_T = self.tmp_MLP[0](dec_embed).reshape(K, B, self.T, -1)
        dec_embed_T = self.tmp_MLP[1](dec_embed_T)
        
        mode_prob = F.softmax(self.motion_cls_l1(dec_embed_T), dim=0).squeeze(dim=-1).T
        out_dist = self.motion_reg_l1(dec_embed_T)
        
        return dec_embed_T, mode_prob, out_dist
    
    def forward(self, scene_context, bev_feat, ec_dynamics, tc_dynamics, ego_dynamics, **kwargs):
        """
        Args:
            scene_context: [B, n, D]
            bev_feature: [B, D, H, W]
            ec(ego_centric)_dynamics: [B, t, self.target_attr]
            tc(target_agent_cetric)_dynamics: [B, t, self.target_attr]
        Returns:
            output: dictionary of predicted_probability and predicted_trajectory
        """
        B, t, _ = ec_dynamics.shape
        n = scene_context.shape[1]
        scene_context_repeat = scene_context.unsqueeze(2).repeat(1, 1, self.T, 1)
        scene_context_repeat = scene_context_repeat.permute(1, 0, 2, 3).reshape(n, B*self.T, -1)
        scene_context = scene_context.permute(1, 0, 2)
        
        mode_query, goal_reg, goal_FDE = self.goal_candidate_proposal(ec_dynamics, tc_dynamics, bev_feat)
        goal_candidate = goal_reg.detach()
        
        dec_embed, init_mode_prob, init_pred_traj = self.initial_prediction(mode_query, scene_context, bev_feat, goal_candidate, ego_dynamics)
        
        ref_points = init_pred_traj[..., :2].detach()
        mode_probs = [init_mode_prob]
        pred_trajs = [init_pred_traj.permute(0, 2, 1, 3)]
        
        dec_embed = dec_embed.permute(2, 1, 0, 3).reshape(self.T, B*self.K, -1)
        dec_embed = self.temp_pos_enc(dec_embed)
        for layer in self.dec_layers:
            query_scale = self.get_query_scale_T(dec_embed)
            dec_embed = layer(
                dec_embed=dec_embed,
                scene_context=scene_context_repeat,
                bev_feat=bev_feat,
                query_scale=query_scale,
                ref_points=ref_points,
                ego_dynamics=ego_dynamics)
            mode_prob = F.softmax(self.motion_cls(dec_embed), dim=0).squeeze(dim=-1).T
            pred_traj = self.motion_reg(dec_embed)
            
            pred_traj[..., :2] += ref_points
            new_ref_points = pred_traj[..., :2]
            ref_points = new_ref_points.detach()
            pred_traj = pred_traj.permute(0, 2, 1, 3)
                
            mode_probs.append(mode_prob)
            pred_trajs.append(pred_traj)
            
            dec_embed = dec_embed.permute(2, 1, 0, 3).reshape(self.T, B*self.K, -1)
            
        output = {'predicted_probability': mode_probs,
                  'predicted_trajectory': pred_trajs,
                  'predicted_goal_FDE': goal_FDE,
                  'predicted_goal_reg': goal_reg}
        return output
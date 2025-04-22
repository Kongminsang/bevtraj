import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from unitraj.models.bevtp.linear import build_mlp, MLP, FFN
from unitraj.models.bevtp.positional_encoding_utils import gen_sineembed_for_position


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_tanh(x, eps=1e-5):
    x = x.clamp(min=-1 + eps, max=1 - eps)
    return torch.atanh(x)


class DeformAttn(nn.Module):
    def __init__(self, config, d_model, grid_size):
        super(DeformAttn, self).__init__()
        
        self.config = config
        self.D = d_model
        self.n_heads = config['num_heads']
        self.n_points = config['num_key_points']
        self.head_dims = self.D // self.n_heads
        
        self.sampling_offsets = build_mlp(self.D, self.D, self.n_heads * self.n_points * 2, dropout=0.0)
        self.attn_weights = build_mlp(self.D, self.D, self.n_heads * self.n_points, dropout=0.0)
        self.value_proj = nn.Conv2d(self.head_dims, self.head_dims, kernel_size=1)
        self.output_proj = build_mlp(self.D, self.D, self.D, dropout=0.0)
        
        self.register_buffer('offset_normalizer', torch.tensor(grid_size, dtype=torch.float32))
        
    def forward(self, ba_query, ref_pos, bev_feat):
        B, _, H, W = bev_feat.shape
        N = ba_query.shape[1]
        
        value = bev_feat.reshape(B, self.n_heads, -1, H, W).reshape(B*self.n_heads, -1, H, W)
        value = self.value_proj(value)
        
        sampling_offsets = self.sampling_offsets(ba_query).reshape(B, N, self.n_heads, self.n_points, 2).permute(0, 2, 1, 3, 4)
        sampling_locations = ref_pos.unsqueeze(1).unsqueeze(3) + sampling_offsets / self.offset_normalizer[None, None, None, None, :]
        sampling_locations = sampling_locations.reshape(B*self.n_heads, N, self.n_points, 2)
        
        sampled_feature = F.grid_sample(value, sampling_locations, align_corners=False, mode='bilinear')
        sampled_feature = sampled_feature.reshape(B, self.n_heads, -1, N, self.n_points).permute(0, 1, 3, 4, 2)
        
        attn_weights = self.attn_weights(ba_query).reshape(B, N, self.n_heads, self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.permute(0, 2, 1, 3).unsqueeze(-1)
        
        attn_outputs = torch.sum(sampled_feature * attn_weights, dim=3)
        attn_outputs = attn_outputs.permute(0, 2, 1, 3).reshape(B, N, -1)
        
        output = self.output_proj(attn_outputs)
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
        
    def forward(self, ba_query, query_pos, ref_pos, bev_feat):
        tgt = self.with_pos_embed(ba_query, query_pos)
        tgt, _ = self.self_attn(tgt, tgt, ba_query)
        tgt = self.norm_layers[0](ba_query + self.dropout_layers[0](tgt))
        
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), ref_pos, bev_feat)
        tgt2 = self.norm_layers[1](tgt + self.dropout_layers[1](tgt2))
        
        output = self.norm_layers[2](self.ffn(tgt2))
        return output
        

class BEVDeformableAggregation(nn.Module):
    def __init__(self, config, d_model):
        super(BEVDeformableAggregation, self).__init__()
        
        self.config = config
        self.D = d_model
        self.dropout = config['dropout']
        self.num_ba_query = config['num_ba_query']
        self.grid_size = config['grid_size']
        
        self.ba_query = nn.Parameter(torch.zeros(self.num_ba_query, self.D), requires_grad=True)
        # self.ref_pos = nn.Parameter(torch.empty(self.num_ba_query, 2), requires_grad=True)
        # nn.init.xavier_uniform_(self.ref_pos)
        self.ref_pos = nn.Parameter(self.create_uniform_2d_grid_tensor(self.num_ba_query), requires_grad=True)
        
        self.pos_scale = build_mlp(self.D, self.D, self.D, dropout=self.dropout)
        self.query_pos = build_mlp(self.D, self.D, self.D, dropout=self.dropout)
        
        self.bda_layers = nn.ModuleList([
            BDALayer(self.config['bda_layer'], self.D, self.grid_size)
            for _ in range(self.config['num_bda_layers'])
        ])
        
        ref_pos_refine = MLP(self.D, self.D, 2, 3)
        self.ref_pos_refine = _get_clones(ref_pos_refine, self.config['num_bda_layers'])
        
    def create_uniform_2d_grid_tensor(self, n_points):
        side = int(n_points ** 0.5)
        if side ** 2 != n_points:
            raise ValueError("n_points == n * n")

        x = torch.linspace(-1, 1, side)
        y = torch.linspace(-1, 1, side)
        yy, xx = torch.meshgrid(y, x, indexing='ij')  # 'ij' to preserve (y,x) order
        grid = torch.stack([xx, yy], dim=-1)  # shape: (side, side, 2)
        return grid.reshape(-1, 2)            # shape: (n_points, 2)
        
    def forward(self, bev_feat):
        
        B = bev_feat.shape[0]
        output, raw_ref_pos = map(lambda x: x[None].repeat(B, 1, 1), [self.ba_query, self.ref_pos])
        ref_pos = torch.tanh(raw_ref_pos)
        
        for lid, layer in enumerate(self.bda_layers):
            query_sine_embed = gen_sineembed_for_position(ref_pos, hidden_dim=self.D, temperature=20)
            query_pos = self.pos_scale(output) * self.query_pos(query_sine_embed)
            
            output = layer(output, query_pos, ref_pos, bev_feat)
            
            tmp = self.ref_pos_refine[lid](output)
            ref_pos = torch.tanh(tmp + inverse_tanh(ref_pos))
            
        ref_pos[..., 0] = ref_pos[..., 0] * self.grid_size[0]
        ref_pos[..., 1] = ref_pos[..., 1] * self.grid_size[1]
        
        return output, ref_pos
        
        
        
        
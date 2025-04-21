import torch
import torch.nn.functional as F
from torch import nn, einsum
from time import time
from einops import rearrange
from unitraj.models.bevtp.positional_encoding_utils import gen_sineembed_for_position
from unitraj.models.bevtp.linear import MLP


# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

# tensor helpers

def create_grid_like(t, dim = 0):
    h, w, device = *t.shape[-2:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(w, device = device),
        torch.arange(h, device = device),
    indexing = 'xy'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

def normalize_grid(grid, w, h, dim = 1, out_dim = -1, ref_from_Q=False):
    # normalizes a grid to range from -1 to 1
    grid_w, grid_h = grid.unbind(dim = dim) # grid_w: x, grid_h: y
    
    if ref_from_Q:
        grid_w = 2.0 * grid_w / max(w, 1)
        grid_h = 2.0 * grid_h / max(h, 1)
    else:
        grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0
        grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0

    return torch.stack((grid_w, grid_h), dim = out_dim) # grid_w: x, grid_h: y


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

    
class PositionalEncoding2D_K(nn.Module):
    def __init__(self, dim, out_dim, *, heads, offset_groups, depth):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups
        
        self.mlp = nn.ModuleList([])
        self.mlp.append(nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        ))
        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))
        self.mlp.append(nn.Linear(dim, out_dim))
        
    def forward(self, grid_kv):
        bias = grid_kv
        for layer in self.mlp:
            bias = layer(bias)
        
        bias = rearrange(bias, 'B H W D -> B D H W')
        
        return bias
    
# main class

class DeformableCrossAttention2D_K(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        num_heads = 8,
        dropout = 0.,
        downsample_factor = 4,
        offset_scale = None,
        offset_groups = None,
        offset_kernel_size = 6,
        group_key_values = True,
    ):
        super().__init__()
        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, num_heads)
        assert divisible_by(num_heads, offset_groups)

        inner_dim = dim_head * num_heads
        self.dim = dim
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.num_heads = num_heads
        self.offset_groups = offset_groups
        offset_dims = inner_dim // offset_groups

        self.downsample_factor = downsample_factor

        self.to_offsets = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2),
            nn.GELU(),
            nn.Conv2d(offset_dims, 2, 1, bias = False),
            nn.Tanh(),
            Scale(offset_scale)
        )
        self.pos_bias = PositionalEncoding2D_K(dim // 4, dim // offset_groups, offset_groups = offset_groups, heads = num_heads, depth = 3)

        self.dropout = nn.Dropout(dropout)
        self.to_temp_k = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self,
                mode_query,
                bev_feat,
                identity=None,
                return_vgrid = False):
        """
        B - batch
        K - num_modes
        H - height
        W - width
        D - dimension
        G - offset_groups
        
        mode_query: (K, B, D)
        bev_feat: (B, D, H, W)
        """
        if identity is None:
            identity = mode_query
            
        mode_query = mode_query.permute(1, 0, 2)
        num_heads, B, K = self.num_heads, mode_query.shape[0], mode_query.shape[1]
        
        # get temp_key for calculating offsets
        
        temp_k = self.to_temp_k(bev_feat)

        # calculate offsets
        
        group = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)
        grouped_keys = group(temp_k)
        offsets = self.to_offsets(grouped_keys)
        
        # calculate grid + offsets
        
        grid = create_grid_like(offsets)
        vgrid = grid + offsets
        vgrid_scaled = normalize_grid(vgrid, w=vgrid.shape[-1], h=vgrid.shape[-2])
        
        # sample features by bilinear interpolation
        
        kv_feats = F.grid_sample(
            group(bev_feat),
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
        
        # Positional Encoding
        
        pos_bias = self.pos_bias(vgrid_scaled)
        kv_feats = kv_feats + pos_bias
        kv_feats = rearrange(kv_feats, '(B G) D ... -> B (G D) ...', B = B)
        
        # derive (q, k, v)
        
        q, k, v = self.to_q(mode_query), self.to_k(kv_feats), self.to_v(kv_feats)
        
        # split out heads & inner_product
        
        q = rearrange(q, 'B K (h d) -> B h K d', B=B, K=K, h=num_heads, d=self.dim_head)
        k, v = map(lambda t: rearrange(t, 'B (h d) ... -> B h (...) d', h = num_heads), (k, v))
            
        # multi-head attention
        
        q = q * self.scale
        
        sim = einsum('B h K d, B h N d -> B h K N', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach() # numerical stability
        
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)
        
        out = einsum('B h K N, B h N d -> B h K d', attn, v)
        out = rearrange(out, 'B h K d -> B K (h d)')
        out = self.to_out(out).permute(1, 0, 2).contiguous()
        
        if return_vgrid:
            return identity + out, vgrid

        return identity + out
    
    
class DeformableCrossAttention2D_Q(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        num_heads = 8,
        offset_groups = None,
        dropout = 0.,
        offset_scale = 4,
        x_bounds = [-51.2, 51.2],
        y_bounds = [-51.2, 51.2]
    ):
        super().__init__()

        inner_dim = dim_head * num_heads
        self.scale = dim_head ** -0.5
        self.num_heads = num_heads
        self.offset_groups = default(offset_groups, num_heads)

        offset_dims = inner_dim // self.offset_groups

        self.offset_scale = offset_scale
        
        assert torch.isclose(torch.abs(torch.tensor(x_bounds[0])), 
                     torch.abs(torch.tensor(x_bounds[1]))), "x range must be symmetric"
        assert torch.isclose(torch.abs(torch.tensor(y_bounds[0])), 
                     torch.abs(torch.tensor(y_bounds[1]))), "y range must be symmetric"
        self.p_w = x_bounds[1] - x_bounds[0]
        self.p_h = y_bounds[1] - y_bounds[0]
        
        self.to_con_q = nn.Linear(dim, inner_dim//2)
        self.to_con_k = nn.Conv2d(dim, inner_dim//2, 1, bias = False)
        self.to_v = nn.Conv2d(dim, inner_dim//2, 1, bias = False)
        
        self.to_pos_q = MLP(inner_dim//2, inner_dim//2, inner_dim//2, 2)
        self.to_pos_k = MLP(inner_dim//2, inner_dim//2, inner_dim//2, 2)

        self.dropout = nn.Dropout(dropout)
        
        self.to_offsets = nn.Sequential(
            nn.Linear(offset_dims//2, offset_dims),
            nn.GELU(),
            nn.Linear(offset_dims, 2),
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.to_out = nn.Linear(inner_dim//2, dim)
    
    def forward(self, dec_embed, bev_feat, query_scale, ref_points, identity=None, return_vgrid=False):
        """
        B - batch
        K - num_modes
        T - future_timestamps
        H - height
        W - width
        D - dimension
        
        dec_embed: (K, B, T, D)
        bev_feat: (B, D, H, W)
        query_scale: (K, B, T, D)
        ref_points: (K, B, T, 2)
        """
        
        if identity is None:
            identity = dec_embed
            
        dec_embed, query_scale, ref_points = map(lambda t: t.permute(1, 0, 2, 3), (dec_embed, query_scale, ref_points))
        num_heads, offset_groups, B, K, T = self.num_heads, self.offset_groups, *dec_embed.shape[:3]

        # get con_q & calculate offsets 

        con_q = self.to_con_q(dec_embed)
        offsets = self.to_offsets(con_q.reshape(B, K, T, offset_groups, -1))

        # calculate grid + offsets

        vgrid = ref_points.unsqueeze(-2) + offsets
        vgrid = vgrid.reshape(B, K*T, offset_groups, -1)
        vgrid_scaled = normalize_grid(vgrid, w=self.p_w, h=self.p_h, dim=-1, out_dim=-1, ref_from_Q=True)
        
        # get con_k / values

        con_k = self.to_con_k(bev_feat)
        con_k = F.grid_sample(
            con_k,
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        v = self.to_v(bev_feat)
        v = F.grid_sample(
            v,
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
        
        # get pos_q, pos_k
        
        ref_scaled = normalize_grid(ref_points, w=self.p_w, h=self.p_h, dim=-1, out_dim=-1, ref_from_Q=True)
        query_sine_embed = gen_sineembed_for_position(ref_scaled, hidden_dim=256, temperature=40)
        pos_q = self.to_pos_q(query_sine_embed)
        pos_q = pos_q * query_scale
        
        key_sine_embed = gen_sineembed_for_position(vgrid_scaled, hidden_dim=256, temperature=40)
        pos_k = self.to_pos_k(key_sine_embed)
        
        # split out heads
        
        BS = B * K * T
        con_q, pos_q = map(lambda t: t.unsqueeze(-2), (con_q, pos_q))
        con_k, v = map(lambda t: t.permute(0, 2, 3, 1), (con_k, v))
        
        con_q, con_k, pos_q, pos_k, v = [t[0].reshape(BS, t[1], num_heads, -1) \
            for t in ((con_q, 1), (con_k, offset_groups), (pos_q, 1), (pos_k, offset_groups), (v, offset_groups))]
        
        q, k = map(lambda t: torch.cat(t, dim=-1), ([con_q, pos_q], [con_k, pos_k]))
        q, k, v = map(lambda t: t.permute(0, 2, 1, 3), (q, k, v))
        
        # multi-head attention

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach() # numerical stability

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, K, T, -1).permute(1, 0, 2, 3)
        out = self.to_out(out)

        if return_vgrid:
            return identity + out, vgrid

        return identity + out


class DeformableCrossAttention2D_layer1(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        num_heads = 8,
        offset_groups = None,
        dropout = 0.,
        offset_scale = 4,
        x_bounds = [-51.2, 51.2],
        y_bounds = [-51.2, 51.2],
    ):
        super().__init__()

        inner_dim = dim_head * num_heads
        self.scale = dim_head ** -0.5
        self.num_heads = num_heads
        self.offset_groups = default(offset_groups, num_heads)

        offset_dims = inner_dim // self.offset_groups

        self.offset_scale = offset_scale
        
        assert torch.isclose(torch.abs(torch.tensor(x_bounds[0])), 
                     torch.abs(torch.tensor(x_bounds[1]))), "x range must be symmetric"
        assert torch.isclose(torch.abs(torch.tensor(y_bounds[0])), 
                     torch.abs(torch.tensor(y_bounds[1]))), "y range must be symmetric"
        self.p_w = x_bounds[1] - x_bounds[0]
        self.p_h = y_bounds[1] - y_bounds[0]
        
        self.to_con_q = nn.Linear(dim, inner_dim//2)
        self.to_con_k = nn.Conv2d(dim, inner_dim//2, 1, bias = False)
        self.to_v = nn.Conv2d(dim, inner_dim//2, 1, bias = False)
        
        self.to_pos_q = MLP(inner_dim//2, inner_dim//2, inner_dim//2, 2)
        self.to_pos_k = MLP(inner_dim//2, inner_dim//2, inner_dim//2, 2)

        self.dropout = nn.Dropout(dropout)
        
        self.to_offsets = nn.Sequential(
            nn.Linear(offset_dims//2, offset_dims),
            nn.GELU(),
            nn.Linear(offset_dims, 2),
            nn.Tanh(),
            Scale(offset_scale)
        )

        self.to_out = nn.Linear(inner_dim//2, dim)
    
    def forward(self, dec_embed, bev_feat, query_scale, ref_points, identity=None, return_vgrid=False):
        """
        B - batch
        K - num_modes
        T - future_timestamps
        H - height
        W - width
        D - dimension
        
        dec_embed: (K, B, D)
        bev_feat: (B, D, H, W)
        query_scale: (K, B, d)
        ref_points: (K, B, 2)
        """
        if identity is None:
            identity = dec_embed
        
        dec_embed, query_scale, ref_points = map(lambda t: t.permute(1, 0, 2), (dec_embed, query_scale, ref_points))
        num_heads, offset_groups, B, K = self.num_heads, self.offset_groups, *dec_embed.shape[:2]

        # get con_q & calculate offsets 

        con_q = self.to_con_q(dec_embed)
        offsets = self.to_offsets(con_q.reshape(B, K, offset_groups, -1))

        # calculate grid + offsets

        vgrid = ref_points.unsqueeze(-2) + offsets
        vgrid_scaled = normalize_grid(vgrid, w=self.p_w, h=self.p_h, dim=-1, out_dim=-1, ref_from_Q=True)
        
        # get con_k / values

        con_k = self.to_con_k(bev_feat)
        con_k = F.grid_sample(
            con_k,
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        v = self.to_v(bev_feat)
        v = F.grid_sample(
            v,
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)
        
        # get pos_q, pos_k
        
        ref_scaled = normalize_grid(ref_points, w=self.p_w, h=self.p_h, dim=-1, out_dim=-1, ref_from_Q=True)
        query_sine_embed = gen_sineembed_for_position(ref_scaled, hidden_dim=256, temperature=40)
        pos_q = self.to_pos_q(query_sine_embed)
        pos_q = pos_q * query_scale
        
        key_sine_embed = gen_sineembed_for_position(vgrid_scaled, hidden_dim=256, temperature=40)
        pos_k = self.to_pos_k(key_sine_embed)
        
        # split out heads
        
        BS = B * K
        con_q, pos_q = map(lambda t: t.unsqueeze(-2), (con_q, pos_q))
        con_k, v = map(lambda t: t.permute(0, 2, 3, 1), (con_k, v))
        
        con_q, con_k, pos_q, pos_k, v = [t[0].reshape(BS, t[1], num_heads, -1) \
            for t in ((con_q, 1), (con_k, offset_groups), (pos_q, 1), (pos_k, offset_groups), (v, offset_groups))]
        
        q, k = map(lambda t: torch.cat(t, dim=-1), ([con_q, pos_q], [con_k, pos_k]))
        q, k, v = map(lambda t: t.permute(0, 2, 1, 3), (q, k, v))
        
        # multi-head attention

        q = q * self.scale

        sim = einsum('B h i d, B h j d -> B h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach() # numerical stability

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('B h i j, B h j d -> B h i d', attn ,v)
        out = out.permute(0, 2, 1, 3).reshape(B, K, -1).permute(1, 0, 2)
        out = self.to_out(out)

        if return_vgrid:
            return identity + out, vgrid

        return identity + out

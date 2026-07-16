import torch
import torch.nn.functional as F
from torch import nn, einsum

from unitraj.models.bevtraj.utility import gen_sineembed_for_position
from unitraj.models.bevtraj.linear import MLP


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

    
# main class

class BEVDeformCrossAttn(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        num_heads = 8,
        num_sampling_points = 6,
        dropout = 0.,
        grid_size = [51.2, 51.2],
        multi_range = False,
        time_scale_range = (0.5, 2.0),
        offset_group_scales = (0.5, 1.0, 2.0),
    ):
        super().__init__()

        inner_dim = dim_head * num_heads
        self.scale = dim_head ** -0.5
        self.num_heads = num_heads
        self.num_sampling_points = num_sampling_points
        self.multi_range = multi_range
        self.time_scale_range = time_scale_range
        self.kv_dim = inner_dim // 2
        assert divisible_by(self.kv_dim, self.num_heads), "kv dim must be divisible by num_heads"
        self.head_dim = self.kv_dim // self.num_heads
        
        self.to_con_q = nn.Linear(dim, self.kv_dim)
        self.to_con_k = nn.Conv2d(dim, self.kv_dim, 1, bias = False)
        self.to_v = nn.Conv2d(dim, self.kv_dim, 1, bias = False)
        
        self.to_pos_q = MLP(self.kv_dim, self.kv_dim, self.kv_dim, 2)
        self.to_pos_k = MLP(self.kv_dim, self.kv_dim, self.head_dim, 2)

        self.dropout = nn.Dropout(dropout)
        
        self.to_offsets = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.head_dim, 2 * self.num_sampling_points),
        )
        self.register_buffer('denorm_scale', torch.tensor(grid_size, dtype=torch.float32))
        self.register_buffer('offset_normalizer', torch.tensor(grid_size, dtype=torch.float32))
        if self.multi_range:
            group_scales = torch.tensor(offset_group_scales, dtype=torch.float32)
            points_per_group = self.num_sampling_points // group_scales.numel()
            self.register_buffer(
                'offset_point_scales',
                group_scales.repeat_interleave(points_per_group),
                persistent=False,
            )

        self.to_out = nn.Linear(self.kv_dim, dim)
    
    def forward(self, dec_embed, bev_feat, query_scale, ref_points, identity=None, return_vgrid=False):
        """
        B - batch
        K - num_modes
        T - sequence length (waypoints for ITP, timestamps for ITR)
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
            
        num_heads, num_points, K, B = self.num_heads, self.num_sampling_points, *dec_embed.shape[:2]
        T = dec_embed.shape[2]
        permute_pattern = (1, 0, 2, 3)
        dec_embed, query_scale, ref_points = map(lambda t: t.permute(*permute_pattern).contiguous(), (dec_embed, query_scale, ref_points))
        
        # get con_q & calculate offsets 

        con_q = self.to_con_q(dec_embed)
        B, K, T, _ = con_q.shape
        Q = K * T
        con_q = con_q.reshape(B, K, T, num_heads, self.head_dim)
        offsets = self.to_offsets(con_q).reshape(
            B, K, T, num_heads, num_points, 2
        )
        if self.multi_range:
            time_scales = torch.linspace(
                self.time_scale_range[0],
                self.time_scale_range[1],
                T,
                device=offsets.device,
                dtype=offsets.dtype,
            )
            offsets = offsets * time_scales[None, None, :, None, None, None]
            offsets = offsets * self.offset_point_scales[
                None, None, None, None, :, None
            ].type_as(offsets)

        ref_pos_norm = (
            ref_points[:, :, :, None, None, :]
            / self.denorm_scale[None, None, None, None, None, :]
        )
        vgrid_scaled = (
            ref_pos_norm
            + offsets
            / self.offset_normalizer[None, None, None, None, None, :]
        )
        vgrid = (
            vgrid_scaled
            * self.denorm_scale[None, None, None, None, None, :]
        )
        vgrid_q = vgrid.reshape(B, Q, num_heads, num_points, 2)
        vgrid_scaled_q = vgrid_scaled.reshape(
            B, Q, num_heads, num_points, 2
        )

        # calculate grid + offsets

        vgrid_scaled_fliped = vgrid_scaled_q.clone()
        vgrid_scaled_fliped[..., -1] = vgrid_scaled_fliped[..., -1] * -1 # Align with F.grid_sample coordinate system
        vgrid_scaled_fliped = vgrid_scaled_fliped.permute(0, 2, 1, 3, 4).reshape(B * num_heads, Q, num_points, 2)
        
        # get con_k / values

        def gs(x):
            _, _, H, W = x.shape
            x = x.reshape(B, num_heads, self.head_dim, H, W).reshape(B * num_heads, self.head_dim, H, W)
            x = F.grid_sample(
                x,
                vgrid_scaled_fliped,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )
            x = x.reshape(B, num_heads, self.head_dim, Q, num_points).permute(0, 3, 1, 4, 2)
            return x

        con_k = gs(self.to_con_k(bev_feat))
        v = gs(self.to_v(bev_feat))
        
        # get pos_q, pos_k
        
        query_sine_embed = gen_sineembed_for_position(ref_points, hidden_dim=self.kv_dim, temperature=10000)
        pos_q = self.to_pos_q(query_sine_embed)
        pos_q = pos_q * query_scale
        pos_q = pos_q.reshape(B, Q, num_heads, self.head_dim)
        
        key_sine_embed = gen_sineembed_for_position(
            vgrid_q.reshape(B, Q, num_heads * num_points, 2),
            hidden_dim=self.kv_dim,
            temperature=10000,
        ).reshape(B, Q, num_heads, num_points, self.kv_dim)
        pos_k = self.to_pos_k(key_sine_embed)
        
        # split out heads
        BS = B * Q
        con_q = con_q.reshape(BS, num_heads, 1, self.head_dim)
        pos_q = pos_q.reshape(BS, num_heads, 1, self.head_dim)
        con_k = con_k.reshape(BS, num_heads, num_points, self.head_dim)
        pos_k = pos_k.reshape(BS, num_heads, num_points, self.head_dim)
        v = v.reshape(BS, num_heads, num_points, self.head_dim)

        q = torch.cat([con_q, pos_q], dim=-1)
        k = torch.cat([con_k, pos_k], dim=-1)
        
        # multi-head attention

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach() # numerical stability

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v).squeeze(2)
        out = out.reshape(B, Q, self.kv_dim)
        out = out.reshape(B, K, T, self.kv_dim).permute(*permute_pattern)
        out = self.to_out(out)

        if return_vgrid:
            return identity + out, vgrid

        return identity + out

import math
import torch

def gen_sineembed_for_position(pos_tensor, hidden_dim=256, temperature=10000):
    pos_tensor_dim_4 = False
    if pos_tensor.dim() == 4:
        B, K, N = pos_tensor.shape[:3]
        pos_tensor = pos_tensor.reshape(B, K*N, -1)
        pos_tensor_dim_4 = True
        
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / half_hidden_dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    if pos_tensor_dim_4:
        pos = pos.view(B, K, N, -1)
    return pos


# coordinate system transformation

def ego_to_target(center_pos, t_x, t_y, r_s, r_c):
    """
    center_pos: (B, N, 2)
    returns: (B, N, 2)
    """
    R = torch.stack([
        torch.cat([r_c,  r_s], dim=-1),
        torch.cat([-r_s, r_c], dim=-1),
    ], dim=1)  # (B, 2, 2)
    center = center_pos @ R

    trans = torch.stack([t_x, t_y], dim=-1)  # (B, 1, 2)
    center = center + trans
    
    return center

def target_to_ego(center_pos, t_x, t_y, r_s, r_c):
    """
    center_pos: (B, N, 2)
    returns: (B, N, 2)
    """
    trans = torch.stack([t_x, t_y], dim=-1)  # (B, 1, 2)
    center = center_pos - trans

    R = torch.stack([
        torch.cat([r_c, -r_s], dim=-1),
        torch.cat([r_s,  r_c], dim=-1),
    ], dim=1)  # (B, 2, 2)

    return center @ R


def batch_nms(pred_trajs, pred_scores, dist_thresh, num_ret_modes=6, return_mask=False):
    """

    Args:
        pred_trajs (batch_size, num_modes, num_timestamps, 7)
        pred_scores (batch_size, num_modes)
        dist_thresh (batch_size)
        num_ret_modes (int, optional): Defaults to 6.

    Returns:
        ret_trajs (batch_size, num_ret_modes, num_timestamps, 7)
        ret_scores (batch_size, num_ret_modes)
        ret_idxs (batch_size, num_ret_modes)
    """
    batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape

    sorted_idxs = pred_scores.argsort(dim=-1, descending=True)
    bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes)
    sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs]
    sorted_pred_trajs = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, 7)
    sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]  # (batch_size, num_modes, 7)

    dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1)
    if isinstance(dist_thresh, float):
        point_cover_mask = (dist < dist_thresh)
    else:
        point_cover_mask = (dist < dist_thresh[:, None, None])

    point_val = sorted_pred_scores.clone()  # (batch_size, N)
    point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)
    ret_mask_sorted = torch.ones_like(point_val).bool() # (batch_size, N)

    ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long()
    ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim)
    ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes)
    bs_idxs = torch.arange(batch_size).type_as(ret_idxs)

    for k in range(num_ret_modes):
        cur_idx = point_val.argmax(dim=-1) # (batch_size)
        ret_idxs[:, k] = cur_idx

        new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
        filter_mask = new_cover_mask.clone()
        filter_mask[bs_idxs, cur_idx] = False
        filter_mask *= (point_val.max(dim=-1, keepdim=True).values > 0)
        ret_mask_sorted[filter_mask] = False

        point_val = point_val * (~new_cover_mask).float()  # (batch_size, N)
        point_val_selected[bs_idxs, cur_idx] = -1
        point_val += point_val_selected

        ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
        ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

    bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None]

    if return_mask:
        ret_mask = torch.zeros_like(ret_mask_sorted)
        ret_mask_sorted[torch.cumsum(ret_mask_sorted, dim=-1) > num_ret_modes] = False
        ret_mask[bs_idxs, sorted_idxs] = ret_mask_sorted
        return ret_mask
    ret_idxs = sorted_idxs[bs_idxs, ret_idxs]
    return ret_trajs, ret_scores, ret_idxs
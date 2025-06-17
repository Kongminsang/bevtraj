import torch
import torch.nn as nn

from unitraj.models.bevtraj.mtr.MTR_utils import PointNetPolylineEncoder, get_batch_offsets, build_mlps
from unitraj.models.bevtraj.mtr.ops.knn import knn_utils
from unitraj.models.bevtraj.mtr.transformer import transformer_encoder_layer

from unitraj.models.bevtraj.linear import build_mlp
from unitraj.models.bevtraj.bev_deformable_aggregation import BEVDeformableAggregation
from unitraj.models.bevtraj.positional_encoding_utils import gen_sineembed_for_position


class BEVTrajSceneContextEncoder(nn.Module):
    def __init__(self, config, pre_enc_dim, bev_feat_dim):
        super(BEVTrajSceneContextEncoder, self).__init__()
        self.config = config
        self.D = config['d_model']
        self.future_len = config['future_len']
        self.use_local_attn = config.get('use_local_attn', False)
        self.num_of_attn_neighbors = config['num_of_attn_neighbors']
        
        # Pointnet-like encoder
        self.agent_pointnet_encoder = self.build_pointnet_encoder(
            in_channels=pre_enc_dim + 1,
            hidden_dim=self.config['pointnet_hidden_dim'],
            num_layers=self.config['pointnet_num_layer'],
            out_channels=self.D)
        
        # BEV Deformable Aggregation
        self.bev_feat_down = nn.Sequential(
                nn.Conv2d(bev_feat_dim, self.D, kernel_size=1),
                nn.GroupNorm(num_groups=8, num_channels=self.D),
                nn.ReLU()
            ) if self.D != bev_feat_dim else nn.Identity()
        self.bda = BEVDeformableAggregation(self.config['bev_deformable_aggregation'], d_model=self.D)
        
        # local(global) self-attn
        self_attn_layers = []
        for _ in range(self.config['num_attn_layers']):
            self_attn_layers.append(self.build_transformer_encoder_layer(
                d_model=self.D,
                nhead=self.config['num_attn_head'],
                dropout=self.config.get('dropout_of_attn', 0.1),
                normalize_before=False,
                use_local_attn=self.use_local_attn
            ))
        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        
        # dense future prediction
        self.in_proj_obj = build_mlp(self.D, self.D, self.D, dropout=0.0)
        self.build_dense_future_prediction_layers(
            hidden_dim=self.D, num_future_frames=self.future_len
        )
        
    def build_pointnet_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_pointnet_encoder = PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_pointnet_encoder
    
    def build_transformer_encoder_layer(self, d_model, nhead, dropout=0.1, normalize_before=False,
                                        use_local_attn=False):
        single_encoder_layer = transformer_encoder_layer.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            normalize_before=normalize_before, use_local_attn=use_local_attn
        )
        return single_encoder_layer
    
    def build_dense_future_prediction_layers(self, hidden_dim, num_future_frames):
        self.obj_pos_encoding_layer = build_mlps(
            c_in=2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True
        )
        self.dense_future_head = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 7], ret_before_act=True
        )

        self.future_traj_mlps = build_mlps(
            c_in=4 * num_future_frames, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True,
            without_norm=True
        )
        self.traj_fusion_mlps = build_mlps(
            c_in=hidden_dim * 2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True,
            without_norm=True
        )
    
    def apply_global_attn(self, x, x_mask, x_pos):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)

        batch_size, N, d_model = x.shape
        x_t = x.permute(1, 0, 2)
        x_mask_t = x_mask.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)

        pos_embedding = gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)
            
        for layer in self.self_attn_layers:
            x_t = layer(
                src=x_t,
                src_key_padding_mask=~x_mask_t,
                pos=pos_embedding
            )
            
        x_out = x_t.permute(1, 0, 2)  # (batch_size, N, d_model)
        return x_out

    def apply_local_attn(self, x, x_mask, x_pos, num_of_neighbors):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        batch_size, N, d_model = x.shape

        x_stack_full = x.view(-1, d_model)  # (batch_size * N, d_model)
        x_mask_stack = x_mask.view(-1)
        x_pos_stack_full = x_pos.view(-1, 3)
        batch_idxs_full = torch.arange(batch_size).type_as(x)[:, None].repeat(1, N).view(-1).int()  # (batch_size * N)

        # filter invalid elements
        x_stack = x_stack_full[x_mask_stack]
        x_pos_stack = x_pos_stack_full[x_mask_stack]
        batch_idxs = batch_idxs_full[x_mask_stack]

        # knn
        batch_offsets = get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).int()  # (batch_size + 1)
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]

        index_pair = knn_utils.knn_batch_mlogk(
            x_pos_stack, x_pos_stack, batch_idxs, batch_offsets, num_of_neighbors
        )  # (num_valid_elems, K)

        # positional encoding
        pos_embedding = \
            gen_sineembed_for_position(x_pos_stack[None, :, 0:2], hidden_dim=d_model)[0]

        # local attn
        output = x_stack
        for layer in self.self_attn_layers:
            output = layer(
                src=output,
                pos=pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs
            )

        ret_full_feature = torch.zeros_like(x_stack_full)  # (batch_size * N, d_model)
        ret_full_feature[x_mask_stack] = output

        ret_full_feature = ret_full_feature.view(batch_size, N, d_model)
        return ret_full_feature
    
    def apply_dense_future_prediction(self, obj_feature, obj_mask, obj_pos):
        num_center_objects, num_objects, _ = obj_feature.shape

        # dense future prediction
        obj_pos_valid = obj_pos[obj_mask][..., 0:2]
        obj_feature_valid = obj_feature[obj_mask]
        obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)
        obj_fused_feature_valid = torch.cat((obj_pos_feature_valid, obj_feature_valid), dim=-1)

        pred_dense_trajs_valid = self.dense_future_head(obj_fused_feature_valid)
        pred_dense_trajs_valid = pred_dense_trajs_valid.view(pred_dense_trajs_valid.shape[0], self.future_len, 7)

        temp_center = pred_dense_trajs_valid[:, :, 0:2] + obj_pos_valid[:, None, 0:2]
        pred_dense_trajs_valid = torch.cat((temp_center, pred_dense_trajs_valid[:, :, 2:]), dim=-1)

        # future feature encoding and fuse to past obj_feature
        obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, -2, -1]].flatten(start_dim=1,
                                                                                      end_dim=2)  # (num_valid_objects, C)
        obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)

        obj_full_trajs_feature = torch.cat((obj_feature_valid, obj_future_feature_valid), dim=-1)
        obj_feature_valid = self.traj_fusion_mlps(obj_full_trajs_feature)

        ret_obj_feature = torch.zeros_like(obj_feature)
        ret_obj_feature[obj_mask] = obj_feature_valid

        ret_pred_dense_future_trajs = obj_feature.new_zeros(num_center_objects, num_objects, self.future_len, 7)
        ret_pred_dense_future_trajs[obj_mask] = pred_dense_trajs_valid

        return ret_obj_feature, ret_pred_dense_future_trajs
    
    def ego_to_target(self, center_pos, t_x, t_y, r_s, r_c):
        """
        Convert batched ego-centric points to target-agent-centric coordinates using matrix multiplication.
        
        Args:
            center_pos (torch.Tensor): Shape (B, n, 2), ego-centric coordinates.
            t_x (torch.Tensor): Shape (B, 1), ego's x position in target-agent-centric coordinates.
            t_y (torch.Tensor): Shape (B, 1), ego's y position in target-agent-centric coordinates.
            r_s (torch.Tensor): Shape (B, 1), sin of ego heading.
            r_c (torch.Tensor): Shape (B, 1), cos of ego heading.
        
        Returns:
            torch.Tensor: Transformed points in target-agent-centric coordinates, shape (B, n, 2).
        """
        n = center_pos.shape[1]
        center_pos = center_pos - torch.cat([t_x, t_y], dim=-1).unsqueeze(1).repeat(1, n, 1)  # Shape: (B, n, 2)
        
        rotation_matrix = torch.stack([
            torch.stack([r_c, r_s], dim=-1),
            torch.stack([-r_s, r_c], dim=-1)
        ], dim=-2)
    
        rotation_matrix = rotation_matrix.squeeze(1)
        rotated_points = torch.matmul(center_pos, rotation_matrix)
        
        return rotated_points
    
    def forward(self, traj_data, pre_encoder_emb, bev_feature):
        obj_trajs, obj_trajs_mask, obj_trajs_last_pos, ego_idx = (traj_data[k] \
                        for k in ['obj_trajs', 'obj_trajs_mask', 'obj_trajs_last_pos', 'ego_index'])

        B, num_objects, _, _ = pre_encoder_emb.shape
        
        # Pointnet-like encoder
        obj_trajs_in = torch.cat((pre_encoder_emb, obj_trajs_mask[:, :, :, None].type_as(pre_encoder_emb)), dim=-1)
        obj_feat = self.agent_pointnet_encoder(obj_trajs_in, obj_trajs_mask)
        
        # BEV Deformable Aggregation
        bev_feat = self.bev_feat_down(bev_feature)
        ba_feat, ref_pos_ego = self.bda(bev_feat) # BEV Aggregated feature
        
        # ego-centric -> target-centric
        B_idx = torch.arange(B, device=obj_trajs.device)
        ego_traj = obj_trajs[B_idx, ego_idx, :, :].unsqueeze(1)
        trans_x, trans_y, rot_sin, rot_cos = ego_traj[:, :, -1, 0], ego_traj[:, :, -1,  1], ego_traj[:, :, -1, -6], ego_traj[:, :, -1, -5]
        ref_pos_target = self.ego_to_target(ref_pos_ego, trans_x, trans_y, rot_sin, rot_cos)
        
        # local(global) self-attention
        zero_tensor = torch.zeros((B, ba_feat.shape[1], 1), device=ref_pos_target.device)
        ref_pos_target = torch.cat((ref_pos_target, zero_tensor), dim=-1)
        
        obj_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)
        ba_valid_mask = torch.ones((B, ba_feat.shape[1]), device=obj_valid_mask.device, dtype=torch.bool)
        
        global_token_feat = torch.cat((obj_feat, ba_feat), dim=1)
        global_token_mask = torch.cat((obj_valid_mask, ba_valid_mask), dim=1)
        global_token_pos = torch.cat((obj_trajs_last_pos, ref_pos_target), dim=1)
        
        if self.use_local_attn:
            global_token_feat = self.apply_local_attn(
                x=global_token_feat, x_mask=global_token_mask, x_pos=global_token_pos,
                num_of_neighbors=self.num_of_attn_neighbors
            )
        else:
            global_token_feat = self.apply_global_attn(
                x=global_token_feat, x_mask=global_token_mask, x_pos=global_token_pos
            )
        
        # dense future prediction
        obj_feat = global_token_feat[:, :num_objects]
        obj_feat_valid = self.in_proj_obj(obj_feat[obj_valid_mask])
        obj_feat = obj_feat.new_zeros(B, num_objects, obj_feat_valid.shape[-1])
        obj_feat[obj_valid_mask] = obj_feat_valid
        
        obj_feat, dense_future_pred = self.apply_dense_future_prediction(
            obj_feature=obj_feat, obj_mask=obj_valid_mask, obj_pos=obj_trajs_last_pos
        )
        
        return obj_feat, dense_future_pred
        
         
        
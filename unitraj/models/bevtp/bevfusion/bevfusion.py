from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from .mmdet3d_modules import Base3DDetector
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization

from .encoders import BEVFusionSparseEncoder
from .backbones import SECOND
from .necks import SECONDFPN
from .fuser import ConvFuser
from .backbones import SwinTransformer
from .necks import GeneralizedLSSFPN
from .vtransform import LSSTransform
from .heads import BEVSegmentationHead

class BEVFusion(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        encoders: Dict[str, Any] = None,    
        fuser: Dict[str, Any] = None,
        decoder: Dict[str, Any] = None,
        heads: Dict[str, Any] = None,
        init_cfg: OptMultiConfig = None,
        weight_path: str = None,
        dataset_name: str = 'nusc',
        bev_map_segmentation: bool = False,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)
        self.encoders = nn.ModuleDict()
        
        if encoders.get("camera") is not None and encoders["camera"]:
            self.encoders["camera"] = nn.ModuleDict(
            {   
                "backbone": SwinTransformer(**encoders["camera"]["backbone"]),
                "neck": GeneralizedLSSFPN(**encoders["camera"]["neck"]),
                "vtransform": LSSTransform(**encoders["camera"]["vtransform"]),
            })
            
        
        
        if encoders.get("lidar") is not None:
            self.encoders["lidar"] = nn.ModuleDict(
            {   
                "backbone": BEVFusionSparseEncoder(**encoders["lidar"]["backbone"]),
            })
        
        if fuser is not None:
            self.fuser = ConvFuser(**fuser)
        else:
            self.fuser = None
        
        self.decoder = nn.ModuleDict(
            {
                "backbone": SECOND(**decoder["backbone"]),
                "neck": SECONDFPN(**decoder["neck"]),
            }
        )
        
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = BEVSegmentationHead(**heads[name], 
                        override_class=bev_map_segmentation, dataset_name=dataset_name)

        self.init_weights()
        
        if weight_path is not None:
            self.load_weights(weight_path)
        else:
            self.init_weights()
            
        self.dataset_name = dataset_name
        self.bev_map_segmentation = bev_map_segmentation
            
    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        if "objects" in self.heads:
            log_vars = []
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    log_vars.append([loss_name, loss_value.mean()])
                elif is_list_of(loss_value, torch.Tensor):
                    log_vars.append(
                        [loss_name,
                         sum(_loss.mean() for _loss in loss_value)])
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')

            loss = sum(value for key, value in log_vars if 'loss' in key)
            log_vars.insert(0, ['loss', loss])
            log_vars = OrderedDict(log_vars)  # type: ignore

            for loss_name, loss_value in log_vars.items():
                # reduce loss when distributed training
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                log_vars[loss_name] = loss_value.item()

            return loss, log_vars  # type: ignore
        
        elif "map" in self.heads:
            log_vars = []
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    log_vars.append([loss_name, loss_value.mean()])
                elif is_list_of(loss_value, torch.Tensor):
                    log_vars.append(
                        [loss_name,
                         sum(_loss.mean() for _loss in loss_value)])
                else:
                    raise TypeError(
                        f'{loss_name} is not a tensor or list of tensors')

            loss = sum(value for key, value in log_vars if 'focal' in key)
            log_vars.insert(0, ['map_loss', loss])
            log_vars = OrderedDict(log_vars)  # type: ignore

            for loss_name, loss_value in log_vars.items():
                # reduce loss when distributed training
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                log_vars[loss_name] = loss_value.item()

            return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()
    
    def load_state_dict(self, state_dict, strict = True):
        
        super().load_state_dict(state_dict, strict)

    def load_weights(self, weight_path: str) -> None:
        state_dict = torch.load(weight_path, map_location="cpu")
        print(f"Loading weights from {weight_path}")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        exclude_prefix = "encoders.camera.vtransform.frustum"
        filtered_state_dict = OrderedDict(
            (k, v) for k, v in state_dict.items() if not k.startswith(exclude_prefix)
        )
        self.load_state_dict(filtered_state_dict, strict=False)

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self.heads, 'objects') and self.heads["objects"] is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self.heads, 'map') and self.heads["map"] is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.encoders["camera"]["vtransform"](
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        # gt_seg_list = []
        # for data_samples in batch_data_samples:
        #     batch_gt_seg = data_samples.gt_pts_seg.masks_bev
        #     gt_seg_list.append({"gt_masks_bev": batch_gt_seg})
            
        if self.with_bbox_head:
            outputs = self.heads["objects"].predict(feats, batch_input_metas)
            res = self.add_pred_to_datasample(batch_data_samples, outputs)
            
            return res
        
        if self.with_seg_head:
            outputs = self.heads["map"].predict(feats, batch_input_metas)
            res = self.add_pred_to_datasample(batch_data_samples, outputs)
            
            return res
        

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            features.append(img_feature)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        return x

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        bevfusion_loss = dict()
        
        if self.with_bbox_head:
            bbox_loss = self.heads["objects"].loss(feats, batch_data_samples)
            bevfusion_loss = bbox_loss
        if self.with_seg_head:
            seg_loss = self.heads["map"].loss(feats, batch_data_samples)
            if self.with_bbox_head:
                bevfusion_loss = bevfusion_loss + seg_loss
            else:
                bevfusion_loss = seg_loss
        losses.update(bevfusion_loss)

        return losses

    def get_bev_feature(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)[0]
        
        # hack implementation for handling different lidar coords
        if self.dataset_name == 'argo2':
            x = torch.rot90(x, k=1, dims=(2,3))
        
        if self.bev_map_segmentation:
            seg_loss = self.heads['map'].loss(feats, batch_data_samples)
        else:
            seg_loss = torch.tensor(0.0).to(feats.device)
        return feats, seg_loss

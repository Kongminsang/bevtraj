from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from mmengine.structures import InstanceData
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sigmoid_xent_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float().to(device)
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

class BEVGridTransform(nn.Module):
    def __init__(
        self,
        *,
        input_scope: List[Tuple[float, float, float]],
        output_scope: List[Tuple[float, float, float]],
        prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
            self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x
    

class BEVSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_transform: Dict[str, Any],
        classes: List[str],
        loss_type: str,
        use_grid_transform: bool = True,
        override_class: bool = False,
        dataset_name: str = 'nusc'
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        if override_class:
            if dataset_name == 'nusc':
                self.classes = ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'divider']
            elif dataset_name == 'argo2':
                self.classes = ['drivable_area', 'ped_crossing', 'solid_lines', 'dashed_lines', 'solid_dash_lines', 'dash_solid_lines']
            else: 
                raise ValueError(f"unsupported dataset: {dataset_name}")
        self.loss_type = loss_type
        self.use_grid_transform = use_grid_transform

        self.transform = BEVGridTransform(**grid_transform)
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, len(classes), 1),
        ).to(device)
        
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(x, (list, tuple)):
            x = x[0]
        if self.use_grid_transform:
            x = self.transform(x)
        x = self.classifier(x)

        for index, name in enumerate(self.classes):
            if self.loss_type == "xent":
                loss = sigmoid_xent_loss(x[:, index], target[:, index])
            elif self.loss_type == "focal":
                loss = sigmoid_focal_loss(x[:, index, :, :], target[:, index, :, :])
            else:
                raise ValueError(f"unsupported loss: {self.loss}")
        return loss * 0.5

    def predict_forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        if isinstance(x, (list, tuple)):
            x = x[0]

        x = self.transform(x)
        x = self.classifier(x)
        
        return torch.sigmoid(x)

            
        
    def loss(self, batch_feats, batch_data_samples):
        """Loss function for BEVSegmentationHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                'gt_masks_bev.
        Returns:
            dict[str:torch.Tensor]: Loss of segmentation.
        """
        batch_input_metas, batch_gt_pts_seg = [], []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_pts_seg.append(data_sample.gt_pts_seg)
        target_list = [torch.tensor(target.masks_bev) for idx, target in enumerate(batch_gt_pts_seg)]
        batch_gt_masks_bev = torch.stack(target_list)
        losses = self(batch_feats, batch_gt_masks_bev)
        
        return losses
        
    def predict(self, batch_feats, batch_data_samples):
        """predict function for BEVSegmentationHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                'gt_masks_bev.
        Returns:
            dict[str:torch.Tensor]: prediction of segmentation.
        """
        #tensor_list = [tensor[i] for i in range(tensor.shape[0])]
        rets = []
        ret_layer = []
        prediction = self.predict_forward(batch_feats)
        pred_seg_list = [prediction[i] for i in range(prediction.shape[0])]
        temp_instances = InstanceData()
        for i in range(len(pred_seg_list)):
            temp_instances.pred_masks_bev = pred_seg_list[i]
            ret_layer.append(temp_instances)
        
        rets.append(ret_layer)
        assert len(
            rets
        ) == 1, f'only support one layer now, but get {len(rets)} layers'
        
        return rets[0]

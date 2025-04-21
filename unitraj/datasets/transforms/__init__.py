# Copyright (c) OpenMMLab. All rights reserved.
from .dbsampler import DataBaseSampler
from .formating import Pack3DDetInputs
from .loading import (LidarDet3DInferencerLoader, LoadAnnotations3D,
                      LoadImageFromFileMono3D,
                      LoadPointsFromDict,  MonoDet3DInferencerLoader,
                      MultiModalityDet3DInferencerLoader, NormalizePointsColor,
                      PointSegClassMapping)
from .test_time_aug import MultiScaleFlipAug3D
# yapf: disable
from .transforms_3d import (AffineResize, BackgroundPointsFilter,
                            GlobalAlignment,
                            IndoorPatchPointSample, IndoorPointSample,
                            LaserMix, MultiViewWrapper, ObjectNameFilter,
                            ObjectNoise, ObjectRangeFilter, ObjectSample,
                            PhotoMetricDistortion3D, PointSample, PointShuffle,
                            PointsRangeFilter, PolarMix, RandomDropPointsColor,
                            RandomFlip3D, RandomJitterPoints, RandomResize3D,
                            RandomShiftScale, Resize3D, VoxelBasedPointSampler)
from .bevfusion import (BEVLoadMultiViewImageFromFiles,
                        ImageAug3D, BEVFusionRandomFlip3D, MIT_GlobalRotScaleTrans,
                        BEVFusionGlobalRotScaleTrans, GridMask, LoadPointsFromMultiSweeps, LoadPointsFromFile, LoadBEVSegmentation,
                        LoadMultiViewImageFromFiles, ImageNormalize, )

__all__ = {
    'DataBaseSampler': DataBaseSampler, 
    'Pack3DDetInputs': Pack3DDetInputs, 
    'LidarDet3DInferencerLoader': LidarDet3DInferencerLoader, 
    'LoadAnnotations3D': LoadAnnotations3D,
    'LoadImageFromFileMono3D': LoadImageFromFileMono3D, 
    'LoadMultiViewImageFromFiles': LoadMultiViewImageFromFiles, 
    'LoadPointsFromDict': LoadPointsFromDict,
    'LoadPointsFromFile': LoadPointsFromFile, 
    'LoadPointsFromMultiSweeps': LoadPointsFromMultiSweeps, 
    'MonoDet3DInferencerLoader': MonoDet3DInferencerLoader,
    'MultiModalityDet3DInferencerLoader': MultiModalityDet3DInferencerLoader, 
    'NormalizePointsColor': NormalizePointsColor, 
    'PointSegClassMapping': PointSegClassMapping,
    'MultiScaleFlipAug3D': MultiScaleFlipAug3D, 
    'AffineResize': AffineResize, 
    'BackgroundPointsFilter': BackgroundPointsFilter,
    'GlobalAlignment': GlobalAlignment, 
    'MIT_GlobalRotScaleTrans': MIT_GlobalRotScaleTrans, 
    'IndoorPatchPointSample': IndoorPatchPointSample,
    'IndoorPointSample': IndoorPointSample, 
    'LaserMix': LaserMix, 
    'MultiViewWrapper': MultiViewWrapper, 
    'ObjectNameFilter': ObjectNameFilter,
    'ObjectNoise': ObjectNoise, 
    'ObjectRangeFilter': ObjectRangeFilter, 
    'ObjectSample': ObjectSample, 
    'PhotoMetricDistortion3D': PhotoMetricDistortion3D,
    'PointSample': PointSample, 
    'PointShuffle': PointShuffle, 
    'PointsRangeFilter': PointsRangeFilter, 
    'PolarMix': PolarMix,
    'RandomDropPointsColor': RandomDropPointsColor, 
    'RandomFlip3D': RandomFlip3D, 
    'RandomJitterPoints': RandomJitterPoints, 
    'RandomResize3D': RandomResize3D,
    'RandomShiftScale': RandomShiftScale, 
    'Resize3D': Resize3D, 
    'VoxelBasedPointSampler': VoxelBasedPointSampler,
    'BEVLoadMultiViewImageFromFiles': BEVLoadMultiViewImageFromFiles, 
    'ImageAug3D': ImageAug3D, 
    'BEVFusionRandomFlip3D': BEVFusionRandomFlip3D,
    'BEVFusionGlobalRotScaleTrans': BEVFusionGlobalRotScaleTrans, 
    'GridMask': GridMask,
    'ImageNormalize': ImageNormalize,
    'LoadBEVSegmentation': LoadBEVSegmentation,
}

def build_transform(cfg):
    transform_type = cfg.pop('type', None)
    if transform_type in __all__:
        return __all__[transform_type](**cfg)
    else:
        raise KeyError(f'Transform type \'{transform_type}\' is not available.')
        
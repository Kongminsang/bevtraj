from .base_dataset import BaseDataset
from .nuscenes.combined_nuscenes import CombinedNuScenes
from .argo2.combined_argo2 import CombinedArgo2


__all__ = {
    "BaseDataset": BaseDataset,
    "CombinedNuScenes": CombinedNuScenes,
    "CombinedArgo2": CombinedArgo2,
}


def build_dataset(config, val=False):
    dataset = __all__[config.dataset_type](
        config=config, is_validation=val
    )
    return dataset
import torch
import random
import pickle
import numpy as np

from copy import deepcopy
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from ..base_dataset import BaseDataset
from .nuscenes_dataset import NuScenesDataset


class CombinedNuScenes(Dataset):
    def __init__(self, config=None, is_validation=False):
        self.is_validation = is_validation
        self.config = deepcopy(config)
        self.sensor_config = OmegaConf.to_container(self.config.SENSOR_DATASET, resolve=True)
        self.traj_dataset = BaseDataset(config.TRAJ_DATASET, is_validation)
        self.sensor_dataset = NuScenesDataset(**self.sensor_config)
        self.data_chunk_size = 1
        self.token2idx_dict_path = config.token2idx_dict_path
        self.token2idx = self.load_pkl_file(self.token2idx_dict_path)
        
    def __len__(self):
        return self.traj_dataset.__len__()

    def __getitem__(self, idx):
        traj_data = self.traj_dataset.__getitem__(idx)[0]
        sample_idx = self.traj2sensor(traj_data)
        
        sensor_data = self.sensor_dataset.__getitem__(sample_idx)
        post_sensor_idx = sensor_data['data_samples'].sample_idx
        
        if sample_idx != post_sensor_idx:
            post_traj_idx = self._rand_another()
            traj_data, sensor_data = self.__getitem__(post_traj_idx)

        return traj_data, sensor_data
    
    def traj2sensor(self, traj_data):
        sample_token = traj_data["scenario_id"].split('_')[2]
        sample_idx = self.token2idx[sample_token]
        return sample_idx
    
    def _rand_another(self) -> int:
        num = list(range(len(self)))
        choosen_list = random.sample(num, 1)
        return choosen_list[0]
    
    def load_pkl_file(self, file_path):
        data = pickle.load(open(file_path, "rb"))
        return data
            
    def collate_fn(self, data_batch):
        # Collate for trajectory data
        traj_batch_list = [data_list[0] for data_list in data_batch]

        batch_size = len(traj_batch_list)
        key_to_list = {key: [traj_batch_list[bs_idx][key] for bs_idx in range(batch_size)] 
                       for key in traj_batch_list[0].keys()}

        traj_input_dict = {}
        for key, val_list in key_to_list.items():
            try:
                traj_input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
            except Exception:
                traj_input_dict[key] = val_list

        traj_input_dict['center_objects_type'] = traj_input_dict['center_objects_type'].numpy()
        traj_batch_dict = {'batch_size': batch_size, 'input_dict': traj_input_dict, 'batch_sample_count': batch_size}

        # Collate for sensor_data
        sensor_batch_list = [data_list[1] for data_list in data_batch]

        data_samples_list = [sensor_data["data_samples"] for sensor_data in sensor_batch_list]
        points_list = [sensor_data['inputs']["points"] for sensor_data in sensor_batch_list]
        img_list = [sensor_data['inputs']["img"] for sensor_data in sensor_batch_list]

        batched_img = torch.stack(img_list, dim=0)

        batch_input_dict = {"points": points_list, "imgs": batched_img}
        sensor_batch_dict = {'data_samples': data_samples_list, 'batch_input_dict': batch_input_dict}

        # return batch
        batch = {'traj_data': traj_batch_dict, 'sensor_data': sensor_batch_dict}
        return batch

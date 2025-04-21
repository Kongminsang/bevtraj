# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from os import path as osp
import pickle

from mmdet3d_tools.dataset_converters import nuscenes_converter as nuscenes_converter
from mmdet3d_tools.dataset_converters import argo2_converter as argo2_converter
from mmdet3d_tools.dataset_converters.update_infos_to_v2 import update_pkl_infos


def make_token2idx(data_list, out_dir, is_val=False):
    token2idx = {}
    for i, data in enumerate(data_list):
        token = data['token']
        idx = data['sample_idx']
        if i != idx:
            raise ValueError("Index not match")
        token2idx[token] = idx
    
    file_name = "sample_token2idx_val.pkl" if is_val else "sample_token2idx_train.pkl"
    file_path = os.path.join(out_dir, file_name)
    
    with open(file_path, "wb") as f:
        pickle.dump(token2idx, f)


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_test_path)
        return

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    
    train_data_list = update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_train_path)
    val_data_list = update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_val_path)
    
    make_token2idx(train_data_list, out_dir=out_dir, is_val=False)
    make_token2idx(val_data_list, out_dir=out_dir, is_val=True)
    
    
def argo2_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    max_sweeps=10,
                    num_workers=8):
    """Prepare data related to argoverse 2 sensor dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
    """
    if version != 'trainval':
        raise NotImplementedError(f'Unsupported Argo2 version {version}!')
    
    argo2_converter.create_argo2_infos(root_path, info_prefix, max_sweeps=max_sweeps, num_workers=num_workers)
    
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    
    train_data_list = update_pkl_infos('argo2', out_dir=out_dir, pkl_path=info_train_path)
    val_data_list = update_pkl_infos('argo2', out_dir=out_dir, pkl_path=info_val_path)
    
    make_token2idx(train_data_list, out_dir=out_dir, is_val=False)
    make_token2idx(val_data_list, out_dir=out_dir, is_val=True)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='nuscenes', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/nuscenes',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')
    
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'argo2': 
        argo2_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            num_workers=args.workers)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')

import argparse
import os

import yaml
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(__file__))


def load_config(path):
    """ load config file"""
    path = os.path.join(ROOT, "configs", f'{path}.yaml')
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='cluster')
    parser.add_argument('--exp_name', '-e', default="test", type=str)
    parser.add_argument('--devices', '-d', nargs='+', default=[0, 1, 2, 3], type=int)
    parser.add_argument('--debug', '-g', action='store_true')
    parser.add_argument('--ckpt_path', '-p', type=str, default=None)
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--use_ewc', '-ew', action='store_true')
    parser.add_argument('--use_smart_sampler', '-ss', action='store_true')
    args = parser.parse_args()
    return args


def save_config_as_txt(cfg):
    """
    Save the current YAML configuration as a TXT file in the appropriate directory.
    """
    
    exp_name = cfg.exp_name
    model_name = cfg.MODEL.NAME
    experiment_dir = os.path.join(f'experiment/{model_name}', exp_name)

    os.makedirs(experiment_dir, exist_ok=True)

    txt_path = os.path.join(experiment_dir, f"{exp_name}.txt")
    with open(txt_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
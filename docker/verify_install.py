#!/usr/bin/env python3
"""Verify the CUDA/OpenMMLab stack and BEVTraj native extensions."""

import argparse
import importlib
import importlib.metadata
from pathlib import Path
import sys


# BEVTraj is normally launched as ``python unitraj/<script>.py`` and several
# modules consequently use script-style imports such as ``datasets`` and
# ``utils``. Reproduce that import path when verification is launched from the
# docker directory.
UNITRAJ_ROOT = Path(__file__).resolve().parents[1] / 'unitraj'
sys.path.insert(0, str(UNITRAJ_ROOT))


EXPECTED_VERSIONS = {
    'pip': '24.2',
    'setuptools': '60.2.0',
    'wheel': '0.44.0',
    'torch': '1.12.1+cu116',
    'torchvision': '0.13.1+cu116',
    'torchaudio': '0.12.1+cu116',
    'mmcv': '2.1.0',
    'mmengine': '0.10.5',
    'mmdet': '3.3.0',
    'mmdet3d': '1.4.0',
    'cumm-cu118': '0.4.11',
    'spconv-cu118': '2.3.6',
    'pytorch-lightning': '2.1.0',
    'numpy': '1.24.4',
    'triton': '3.0.0',
    'gymnasium': '1.0.0',
    'metadrive-simulator': '0.4.2.3',
    'scenarionet': '0.0.1',
}

NATIVE_EXTENSIONS = (
    'unitraj.models.bevtraj.mtr.ops.knn.knn_cuda',
    'unitraj.models.bevtraj.mtr.ops.attention.attention_cuda',
    'unitraj.models.bevtraj.bevfusion.ops.bev_pool.bev_pool_ext',
    'unitraj.models.bevtraj.bevfusion.ops.voxel.voxel_layer',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--skip-gpu',
        action='store_true',
        help='Only check imports; intended for image build time.',
    )
    parser.add_argument(
        '--expect-gpus',
        type=int,
        default=4,
        help='Minimum GPU count expected at runtime (default: 4).',
    )
    parser.add_argument(
        '--expect-capability',
        default='8.9',
        help='Expected CUDA compute capability (default: 8.9 for RTX 4070/4090).',
    )
    return parser.parse_args()


def check_versions():
    errors = []
    for package, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package)
        status = 'OK' if actual == expected else 'MISMATCH'
        print(f'[{status}] {package}: {actual} (expected {expected})')
        if actual != expected:
            errors.append(f'{package}=={actual}, expected {expected}')

    import torch

    status = 'OK' if torch.version.cuda == '11.6' else 'MISMATCH'
    print(f'[{status}] PyTorch CUDA: {torch.version.cuda} (expected 11.6)')
    if torch.version.cuda != '11.6':
        errors.append(f'PyTorch CUDA is {torch.version.cuda}, expected 11.6')
    return errors


def check_imports():
    errors = []
    modules = (
        'torch',
        'torchvision',
        'mmcv',
        'mmcv.ops',
        'mmengine',
        'mmdet',
        'mmdet3d',
        'spconv.pytorch',
        *NATIVE_EXTENSIONS,
    )
    for module in modules:
        try:
            importlib.import_module(module)
            print(f'[OK] import {module}')
        except Exception as exc:  # noqa: BLE001 - report every binary import failure
            print(f'[FAIL] import {module}: {exc}')
            errors.append(f'import {module}: {exc}')
    return errors


def check_gpus(expected_count, expected_capability):
    import torch

    errors = []
    if not torch.cuda.is_available():
        return errors + ['torch.cuda.is_available() is False']

    count = torch.cuda.device_count()
    print(f'[INFO] visible GPUs: {count}')
    if count < expected_count:
        errors.append(f'only {count} GPUs visible, expected at least {expected_count}')

    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        capability = f'{props.major}.{props.minor}'
        print(
            f'[OK] cuda:{index}: {props.name}, '
            f'compute capability {capability}, '
            f'{props.total_memory / 1024**3:.1f} GiB'
        )
        if capability != expected_capability:
            errors.append(
                f'cuda:{index} capability is {capability}, '
                f'expected {expected_capability}'
            )

    value = torch.tensor([1.0], device='cuda')
    if value.add(1).item() != 2:
        errors.append('CUDA smoke calculation returned an unexpected value')
    else:
        print('[OK] CUDA smoke calculation')
    return errors


def main():
    args = parse_args()
    errors = check_versions()
    errors.extend(check_imports())
    if not args.skip_gpu:
        errors.extend(check_gpus(args.expect_gpus, args.expect_capability))

    if errors:
        print('\nVerification failed:', file=sys.stderr)
        for error in errors:
            print(f'  - {error}', file=sys.stderr)
        return 1

    mode = 'image build' if args.skip_gpu else 'GPU runtime'
    print(f'\nBEVTraj {mode} verification passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

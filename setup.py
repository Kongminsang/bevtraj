import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources, define_macros=None):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources],
        define_macros=define_macros or [],
        extra_compile_args={
            'cxx': ['-O2'],
            'nvcc': ['-O2'],
        },
    )
    return cuda_ext


if __name__ == '__main__':
    setup(
        name='bevtraj',
        version='1.0',
        description='bevtraj',
        author='Minsang Kong',
        author_email='gms0725@kookmin.ac.kr',
        license='MIT',
        packages=find_packages(
            exclude=[
                'tools',
                'data',
                'build',
                'ckpt',
                'docs',
                'experiment',
                'lightning_logs',
                'outputs',
                'pretraining_ckpt',
                'wandb',
                'tmp',
                '.vscode',
                '*.egg-info'
            ]
        ),
        cmdclass={
            'build_ext': BuildExtension,
        },

        ext_modules=[
            make_cuda_ext(
                name='knn_cuda',
                module='unitraj.models.bevtraj.mtr.ops.knn',
                sources=[
                    'src/knn.cpp',
                    'src/knn_gpu.cu',
                    'src/knn_api.cpp',
                ],
            ),
            make_cuda_ext(
                name='attention_cuda',
                module='unitraj.models.bevtraj.mtr.ops.attention',
                sources=[
                    'src/attention_api.cpp',
                    'src/attention_func_v2.cpp',
                    'src/attention_func.cpp',
                    'src/attention_value_computation_kernel_v2.cu',
                    'src/attention_value_computation_kernel.cu',
                    'src/attention_weight_computation_kernel_v2.cu',
                    'src/attention_weight_computation_kernel.cu',
                ],
            ),
            make_cuda_ext(
                name='bev_pool_ext',
                module='unitraj.models.bevtraj.bevfusion.ops.bev_pool',
                sources=[
                    'src/bev_pool.cpp',
                    'src/bev_pool_cuda.cu',
                ],
            ),
            make_cuda_ext(
                name='voxel_layer',
                module='unitraj.models.bevtraj.bevfusion.ops.voxel',
                sources=[
                    'src/voxelization.cpp',
                    'src/scatter_points_cpu.cpp',
                    'src/scatter_points_cuda.cu',
                    'src/voxelization_cpu.cpp',
                    'src/voxelization_cuda.cu',
                ],
                define_macros=[('WITH_CUDA', None)],
            ),
        ],
        package_data={'unitraj.models.bevtraj': ['*.pkl']},
        zip_safe=False,
    )

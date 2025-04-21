import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


if __name__ == '__main__':
    setup(
        name='bevtp',
        version=1.0,
        description='bevtp',
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
                module='unitraj.models.bevtp.mtr.ops.knn',
                sources=[
                    'src/knn.cpp',
                    'src/knn_gpu.cu',
                    'src/knn_api.cpp',
                ],
            ),
            make_cuda_ext(
                name='attention_cuda',
                module='unitraj.models.bevtp.mtr.ops.attention',
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
        ],
    )

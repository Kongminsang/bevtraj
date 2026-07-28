# syntax=docker/dockerfile:1

ARG CUDA_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8

ENV CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    MAX_JOBS=${MAX_JOBS} \
    TORCH_CUDA_ARCH_LIST=8.6+PTX \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libx11-6 \
        libxcursor1 \
        libxext6 \
        libxi6 \
        libxrandr2 \
        libxrender1 \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && python -m pip install --upgrade \
        pip==24.2 \
        setuptools==60.2.0 \
        wheel==0.44.0

WORKDIR /opt/bevtraj

# Reproduce the supplied server environment. PyTorch 1.12 only recognizes
# architectures through SM 8.6, so include PTX for forward compatibility with
# the Ada (SM 8.9) RTX 4070/4090 while compiling with the CUDA 11.8 toolkit.
RUN python -m pip install \
        --no-deps \
        torch==1.12.1+cu116 \
        torchvision==0.13.1+cu116 \
        torchaudio==0.12.1+cu116 \
        --index-url https://download.pytorch.org/whl/cu116

# Install the remaining dependencies from PyPI in a separate layer. Keeping
# the large PyTorch wheel above isolated means dependency changes do not force
# that wheel to be downloaded again.
COPY docker/requirements.txt docker/constraints.txt docker/

RUN python -m pip install \
        --constraint docker/constraints.txt \
        --requirement docker/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu116 \
    && python -m pip install \
        --no-deps \
        metadrive-simulator==0.4.2.3 \
    && python -m pip install \
        --no-deps \
        "scenarionet @ git+https://github.com/metadriverse/scenarionet.git@e956a03d80c30b65ad9ca0e625876a01484e5279" \
    && python -m pip install \
        mmcv==2.1.0 \
        --only-binary=mmcv \
        --find-links https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html

COPY . .

# Builds all four project-native CUDA extensions for Ada GPUs:
# attention_cuda, knn_cuda, bev_pool_ext, and voxel_layer.
RUN python -m pip install --verbose --no-build-isolation --editable .

RUN python docker/verify_install.py --skip-gpu

CMD ["/bin/bash"]

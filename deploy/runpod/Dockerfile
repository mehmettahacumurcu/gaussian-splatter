# 4DGS Studio backend — RunPod / cloud GPU image.
#
# Base: NVIDIA's runtime CUDA image (matches RunPod 4090/3090 hosts).
# Python 3.11, PyTorch 2.5.1 cu124 wheels, gsplat 1.5.3 (JIT-compiles on
# first import — ~2-3 min, then cached).
#
# Build (locally or on the pod):
#   docker build -t 4dgs-studio:cu121 -f deploy/runpod/Dockerfile .
#
# Run (port 8000 maps through RunPod's HTTP proxy):
#   docker run --gpus all -p 8000:8000 \
#       -e RUNPOD_AUTH_TOKEN=$(openssl rand -hex 32) \
#       -v /workspace/data:/app/data \
#       4dgs-studio:cu121

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    # gsplat needs a writable build dir; default it under /app for clarity.
    TORCH_EXTENSIONS_DIR=/app/.torch_extensions \
    # Avoid VRAM fragmentation (matches scripts/start-backend.bat).
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps. NB: we deliberately do NOT apt-install colmap. The Ubuntu
# package forces software OpenGL on headless pods which silently falls back
# to CPU SIFT (~6-10x slower). Instead we install conda-forge's CUDA-enabled
# colmap below.
#   - python3.11 + venv + headers (gsplat compiles C++ extensions)
#   - build-essential + ninja (for the JIT compile)
#   - ffmpeg (frame extraction)
#   - git, curl, wget, ca-certificates (cloning, model downloads, miniconda)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        build-essential ninja-build \
        ffmpeg \
        git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python/python3.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Miniconda + CUDA-enabled COLMAP from conda-forge.
ENV MINICONDA_DIR=/opt/miniconda \
    PATH=/opt/miniconda/bin:${PATH}
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda_installer.sh && \
    bash /tmp/miniconda_installer.sh -b -p /opt/miniconda && \
    rm /tmp/miniconda_installer.sh && \
    /opt/miniconda/bin/conda install -y -c conda-forge colmap && \
    /opt/miniconda/bin/conda clean -afy && \
    ln -sf /opt/miniconda/bin/colmap /usr/bin/colmap

WORKDIR /app

# PyTorch first (its own index URL); pinning matches requirements.txt header.
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124

# Project deps. Copy requirements only first so the layer is cached when only
# project source changes.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# Project source.
COPY backend       /app/backend
COPY scripts       /app/scripts
COPY environment.yml /app/environment.yml

# Pre-create the data dir; the user mounts a volume here at runtime.
RUN mkdir -p /app/data /app/.torch_extensions

# Boot script — binds 0.0.0.0:8000 (RunPod proxies it to the pod's public URL).
COPY deploy/runpod/start.sh /app/deploy/runpod/start.sh
RUN chmod +x /app/deploy/runpod/start.sh

EXPOSE 8000

ENTRYPOINT ["/app/deploy/runpod/start.sh"]

#!/usr/bin/env bash
# Alternative to building the Docker image: run inside a stock RunPod
# "PyTorch 2.x + CUDA 12.1" container, clone the repo, install deps, start
# the backend. Fastest path if you don't want to build/push an image.
#
# Usage on the pod (after SSH-ing in):
#   curl -sSL https://raw.githubusercontent.com/<user>/<repo>/<branch>/deploy/runpod/pod-bootstrap.sh | bash -s -- <git-url> <branch>
# Or copy this file over and run:
#   bash pod-bootstrap.sh https://github.com/mehmettahacumurcu/gaussian-splatter.git feat/sota-verification

set -euo pipefail

REPO_URL="${1:-https://github.com/mehmettahacumurcu/gaussian-splatter.git}"
BRANCH="${2:-feat/sota-verification}"
WORKDIR="${WORKDIR:-/workspace/4dgs-studio}"

echo "==> Bootstrapping 4DGS Studio backend"
echo "    repo:   ${REPO_URL}"
echo "    branch: ${BRANCH}"
echo "    dir:    ${WORKDIR}"

# Apt deps (RunPod stock images don't always have colmap / ffmpeg).
echo "==> apt-get install (colmap, ffmpeg, build-essential, ninja-build)"
apt-get update -y >/dev/null
apt-get install -y --no-install-recommends \
    colmap ffmpeg build-essential ninja-build git curl ca-certificates \
    >/dev/null
rm -rf /var/lib/apt/lists/*

# Clone or refresh.
mkdir -p "$(dirname "${WORKDIR}")"
if [[ -d "${WORKDIR}/.git" ]]; then
    echo "==> Repo already present — fetching ${BRANCH}"
    git -C "${WORKDIR}" fetch origin "${BRANCH}"
    git -C "${WORKDIR}" checkout "${BRANCH}"
    git -C "${WORKDIR}" reset --hard "origin/${BRANCH}"
else
    echo "==> Cloning repo"
    git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${WORKDIR}"
fi

cd "${WORKDIR}"

# Pip deps. The stock RunPod PyTorch image already has torch/torchvision —
# only install if not already present. (Stock image: torch 2.4 / cu121.)
echo "==> pip install requirements"
python -m pip install --upgrade pip setuptools wheel >/dev/null
if ! python -c "import torch" >/dev/null 2>&1; then
    python -m pip install \
        torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124
fi
python -m pip install -r requirements.txt

# Pre-build extension cache locations.
mkdir -p /workspace/.torch_extensions /workspace/.cache/huggingface
export TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions
export HF_HOME=/workspace/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

echo
echo "==> Bootstrap complete."
echo "    Repo:     ${WORKDIR}"
echo "    To start backend (foreground):"
echo "      cd ${WORKDIR} && bash deploy/runpod/start.sh"
echo "    To start in the background and tail logs:"
echo "      nohup bash deploy/runpod/start.sh > /workspace/backend.log 2>&1 &"
echo "      tail -f /workspace/backend.log"
echo
echo "==> Don't forget to export RUNPOD_AUTH_TOKEN before starting."
echo "    export RUNPOD_AUTH_TOKEN=\$(openssl rand -hex 32)"
echo "    echo \"\$RUNPOD_AUTH_TOKEN\"   # save this — you'll send it from your laptop"

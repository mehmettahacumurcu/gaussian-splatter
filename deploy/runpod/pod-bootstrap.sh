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
MINICONDA_DIR="${MINICONDA_DIR:-/opt/miniconda}"

echo "==> Bootstrapping 4DGS Studio backend"
echo "    repo:   ${REPO_URL}"
echo "    branch: ${BRANCH}"
echo "    dir:    ${WORKDIR}"

# ---------------------------------------------------------------------------
# Apt deps (no colmap here — apt's colmap forces software OpenGL on headless
# pods which silently falls back to CPU SIFT, ~6-10x slower on big videos.
# We install conda-forge's CUDA-enabled colmap below instead).
# ---------------------------------------------------------------------------
echo "==> apt-get install (ffmpeg, build-essential, ninja-build, git, curl)"
apt-get update -y >/dev/null
apt-get install -y --no-install-recommends \
    ffmpeg build-essential ninja-build \
    git curl ca-certificates wget \
    >/dev/null
rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Miniconda + CUDA-enabled COLMAP from conda-forge.
# conda-forge's colmap build has CUDA SIFT extraction + matching enabled out
# of the box, which is what we need for fast preprocessing on big videos.
# Skip if already installed (e.g. on a re-run with a persistent volume).
# ---------------------------------------------------------------------------
if [[ ! -x "${MINICONDA_DIR}/bin/conda" ]]; then
    echo "==> Installing miniconda to ${MINICONDA_DIR}"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda_installer.sh
    bash /tmp/miniconda_installer.sh -b -p "${MINICONDA_DIR}"
    rm -f /tmp/miniconda_installer.sh
else
    echo "==> miniconda already at ${MINICONDA_DIR}"
fi
export PATH="${MINICONDA_DIR}/bin:${PATH}"

# Persist conda PATH for future shells.
if ! grep -q "${MINICONDA_DIR}/bin" /root/.bashrc 2>/dev/null; then
    echo "export PATH=\"${MINICONDA_DIR}/bin:\$PATH\"" >> /root/.bashrc
fi

if ! command -v colmap >/dev/null 2>&1 \
   || [[ "$(readlink -f "$(command -v colmap)")" != *miniconda* ]]; then
    echo "==> conda install colmap (CUDA-enabled, ~3-5 min first time)"
    "${MINICONDA_DIR}/bin/conda" install -y -c conda-forge colmap \
        >/tmp/conda-colmap.log 2>&1 || {
        echo "[!] conda install colmap failed; see /tmp/conda-colmap.log"
        tail -n 30 /tmp/conda-colmap.log
        exit 1
    }
else
    echo "==> conda-forge colmap already installed"
fi

# Replace any apt-installed /usr/bin/colmap with a wrapper that points to the
# conda binary. This way every consumer of /usr/bin/colmap (subprocess calls
# in pipeline.py, etc.) gets the CUDA build.
if [[ -e /usr/bin/colmap && ! -L /usr/bin/colmap ]]; then
    mv /usr/bin/colmap /usr/bin/colmap.real.bak 2>/dev/null || true
fi
ln -sf "${MINICONDA_DIR}/bin/colmap" /usr/bin/colmap
echo "==> /usr/bin/colmap -> $(readlink -f /usr/bin/colmap)"

# Quick smoke test — colmap --help should not crash.
if ! /usr/bin/colmap --help >/dev/null 2>&1; then
    echo "[!] colmap --help failed; conda install may be broken."
    /usr/bin/colmap --help 2>&1 | tail -10 || true
    exit 1
fi

# ---------------------------------------------------------------------------
# Repo clone / refresh.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Pip deps. The stock RunPod PyTorch image already has torch/torchvision —
# only install if not already present.
# ---------------------------------------------------------------------------
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
echo "    Conda:    ${MINICONDA_DIR}/bin/conda"
echo "    Colmap:   $(command -v colmap)  ($(/usr/bin/colmap --version 2>/dev/null | head -1 || echo unknown))"
echo
echo "    To start backend (foreground):"
echo "      cd ${WORKDIR} && bash deploy/runpod/start.sh"
echo "    To start in the background and tail logs:"
echo "      nohup bash deploy/runpod/start.sh > /workspace/backend.log 2>&1 &"
echo "      tail -f /workspace/backend.log"
echo
echo "==> Optional: export RUNPOD_AUTH_TOKEN before starting (cloud security)."
echo "    export RUNPOD_AUTH_TOKEN=\$(openssl rand -hex 32)"
echo "    echo \"\$RUNPOD_AUTH_TOKEN\"   # save this — paste into desktop app's connection settings"

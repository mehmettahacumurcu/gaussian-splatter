#!/usr/bin/env bash
# Stage a scene dataset to /dev/shm (RAM disk) for fast training.
#
# RunPod's MFS network filesystem (/workspace) has high per-file latency
# that bottlenecks training at ~3 it/s with 13% GPU util on A100. Copying
# the dataset to /dev/shm bypasses MFS — expected 10x speedup.
#
# Usage:
#   bash deploy/runpod/stage_dataset.sh flame_steak
#
# Then before training, point the API at /dev/shm/data/<scene>/ via:
#   export DATA_ROOT=/dev/shm/4dgs-studio/data
# (the backend reads DATA_ROOT — see backend/api.py paths setup).

set -euo pipefail

SCENE="${1:?Usage: stage_dataset.sh <scene_name>}"
SOURCE="${SOURCE:-/workspace/4dgs-studio/data/$SCENE}"
TARGET="${TARGET:-/dev/shm/4dgs-studio/data/$SCENE}"

if [[ ! -d "$SOURCE" ]]; then
    echo "[stage] ERROR: source not found: $SOURCE"
    exit 1
fi

# Check available RAM
SOURCE_GB=$(du -sb "$SOURCE" | awk '{ printf "%.1f", $1/1024/1024/1024 }')
SHM_AVAIL_GB=$(df -BG --output=avail /dev/shm | tail -1 | tr -d 'G ')
echo "[stage] source $SOURCE_GB GB → /dev/shm (avail ${SHM_AVAIL_GB}G)"
if (( $(echo "$SOURCE_GB > $SHM_AVAIL_GB - 5" | bc -l) )); then
    echo "[stage] ERROR: not enough RAM. Need $SOURCE_GB GB, have ${SHM_AVAIL_GB}G."
    echo "[stage] Tip: If only frames+depth+masks needed, exclude videos/ via --exclude."
    exit 1
fi

mkdir -p "$(dirname "$TARGET")"
echo "[stage] copying $SOURCE → $TARGET"
START=$(date +%s)
# Use rsync if available (resumable), else cp
if command -v rsync >/dev/null 2>&1; then
    rsync -a --info=progress2 "$SOURCE/" "$TARGET/"
else
    cp -r "$SOURCE" "$TARGET"
fi
END=$(date +%s)
echo "[stage] done in $((END-START)) sec"
echo "[stage] export DATA_ROOT=/dev/shm/4dgs-studio/data"
echo "[stage] (or pass scene_dir override to your job submission)"

#!/usr/bin/env bash
# Stage a scene dataset to /dev/shm (RAM disk) for fast training.
#
# RunPod's MFS network filesystem (/workspace) has high per-file latency
# that bottlenecks training at ~3 it/s with 13% GPU util on A100. Copying
# the dataset to /dev/shm bypasses MFS — expected 10x speedup (~30+ it/s).
#
# Usage (full flow on the pod):
#   # 1. Stage the dataset to RAM:
#   bash deploy/runpod/stage_dataset.sh flame_steak
#
#   # 2. Point the backend at /dev/shm BEFORE launching uvicorn:
#   export FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data
#   nohup bash deploy/runpod/start.sh > /workspace/backend.log 2>&1 &
#
#   # 3. Submit your job — the backend will read frames/depth/masks from RAM.
#   curl -X POST -F "scene=flame_steak" ... http://localhost:8000/process
#
# How it works: backend/config.py reads FOURDGS_DATA_ROOT at module-load.
# If set, scene_paths() routes ALL data lookups through the override path
# instead of the default <project>/data/.

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

# ----------------------------------------------------------------------
# Persist outputs back to /workspace via symlinks.
#
# Why: training writes PLY checkpoints, eval renders, and cache markers
# under <DATA_ROOT>/<scene>/output and <DATA_ROOT>/<scene>/.cache_markers/.
# If those land on /dev/shm they're lost on pod restart.
# Solution: replace the /dev/shm/<scene>/output dir with a symlink to
# /workspace/<scene>/output. Writes go through the symlink → persistent.
# Same for .cache_markers (so re-runs hit cache instead of redoing 30 min
# of preprocessing).
# Reads of these are rare during training (checkpoints every Nk iters,
# cache markers once at start), so MFS latency penalty is negligible.
# ----------------------------------------------------------------------
WORKSPACE_SCENE="/workspace/4dgs-studio/data/$SCENE"
for subdir in "output" ".cache_markers"; do
    persistent="$WORKSPACE_SCENE/$subdir"
    staged="$TARGET/$subdir"
    mkdir -p "$persistent"
    rm -rf "$staged"  # delete copy if rsync brought it; replace with symlink
    ln -sfn "$persistent" "$staged"
    echo "[stage] symlink: $staged -> $persistent (writes persist on /workspace)"
done

echo ""
echo "============================================================"
echo "  NEXT STEP: export the env var BEFORE starting the backend:"
echo ""
echo "    export FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data"
echo ""
echo "  Then launch start.sh (or restart it if already running)."
echo "  Verify in backend.log:"
echo "    [config] DATA_ROOT overridden via FOURDGS_DATA_ROOT=/dev/shm/..."
echo ""
echo "  Outputs (PLY, eval, cache markers) land on /workspace via"
echo "  symlinks — they persist across pod restart."
echo "============================================================"

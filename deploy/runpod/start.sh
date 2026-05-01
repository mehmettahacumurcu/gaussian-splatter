#!/usr/bin/env bash
# 4DGS Studio backend boot — binds 0.0.0.0:8000 for RunPod's HTTP proxy.
#
# Env vars consumed:
#   RUNPOD_AUTH_TOKEN  Required for cloud. Bearer token clients must send.
#                      If unset, backend runs unauthenticated (dev only).
#   CORS_ALLOW_ORIGINS Optional. Comma-separated extra CORS origins.
#                      Default (localhost / Tauri) is always allowed.
#   PORT               Port to bind. Default 8000.
#   WORKERS            uvicorn workers. Keep 1 (GPU-bound, single JobManager).

set -euo pipefail

cd /app

PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

if [[ -z "${RUNPOD_AUTH_TOKEN:-}" ]]; then
  echo "[start.sh] WARNING: RUNPOD_AUTH_TOKEN is unset — backend will accept"
  echo "[start.sh]          unauthenticated requests. OK for local dev only."
else
  echo "[start.sh] Bearer-token auth enabled (token len=${#RUNPOD_AUTH_TOKEN})"
fi

echo "[start.sh] Binding 0.0.0.0:${PORT}"
exec python -m uvicorn backend.api:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --log-level info

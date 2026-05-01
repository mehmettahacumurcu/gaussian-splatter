#!/usr/bin/env bash
# 4DGS Studio backend boot — binds 0.0.0.0:8000 for RunPod's HTTP proxy.
#
# Auto-detects:
#   - Project root from this script's location (../../ from deploy/runpod/start.sh)
#   - A Python interpreter that has uvicorn installed (so it doesn't matter
#     whether deps live in conda env, system python, or /usr/local).
#
# Env var overrides:
#   BACKEND_PYTHON     Force a specific python binary path (skip auto-detect).
#   BACKEND_WORKDIR    Force project root.
#   RUNPOD_AUTH_TOKEN  Bearer token; backend rejects unauthenticated requests when set.
#   CORS_ALLOW_ORIGINS Comma-separated extra CORS origins.
#   PORT               Port to bind. Default 8000.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${BACKEND_WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ ! -d "${PROJECT_DIR}/backend" ]]; then
  echo "[start.sh] ERROR: ${PROJECT_DIR}/backend not found." >&2
  echo "[start.sh] Set BACKEND_WORKDIR to the directory containing backend/." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
echo "[start.sh] Project dir: ${PROJECT_DIR}"

# ---------------------------------------------------------------------------
# Find a Python that has uvicorn installed.
#
# Multiple Pythons coexist on RunPod's PyTorch image:
#   /opt/miniconda/bin/python  (added by our pod-bootstrap when conda colmap installs)
#   /usr/local/bin/python      (RunPod base image's primary python with torch et al.)
#   /usr/bin/python3           (Ubuntu system python)
#
# pip install in pod-bootstrap targets whichever 'python' is first on PATH at
# that moment, which depends on PATH ordering. start.sh shouldn't assume.
# Probe each candidate; pick the first one where `import uvicorn` works.
# ---------------------------------------------------------------------------
PYTHON_BIN="${BACKEND_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in \
      /usr/local/bin/python \
      /usr/local/bin/python3 \
      /opt/miniconda/bin/python \
      /usr/bin/python3 \
      python3 \
      python; do
    if command -v "${candidate}" >/dev/null 2>&1 \
       && "${candidate}" -c "import uvicorn, fastapi, backend.api" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[start.sh] ERROR: no Python with uvicorn + fastapi + backend.api importable found." >&2
  echo "[start.sh] Tried: /usr/local/bin/python, /opt/miniconda/bin/python, /usr/bin/python3, python3, python" >&2
  echo "[start.sh] Run on the pod:" >&2
  echo "[start.sh]   /usr/local/bin/python -m pip install -r ${PROJECT_DIR}/requirements.txt" >&2
  echo "[start.sh] or set BACKEND_PYTHON=/path/to/python" >&2
  exit 1
fi
echo "[start.sh] Using python: ${PYTHON_BIN}"

PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

if [[ -z "${RUNPOD_AUTH_TOKEN:-}" ]]; then
  echo "[start.sh] WARNING: RUNPOD_AUTH_TOKEN is unset -- backend will accept"
  echo "[start.sh]          unauthenticated requests. OK for local dev only."
else
  echo "[start.sh] Bearer-token auth enabled (token len=${#RUNPOD_AUTH_TOKEN})"
fi

echo "[start.sh] Binding 0.0.0.0:${PORT}"
exec "${PYTHON_BIN}" -m uvicorn backend.api:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --log-level info

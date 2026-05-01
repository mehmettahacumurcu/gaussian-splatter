#!/usr/bin/env bash
# Trigger the SOTA verification job (flame_steak / N3V Plenoptic Video) on a
# remote 4DGS Studio backend. Run this from your laptop after the pod is up
# and `data/flame_steak/` has been uploaded.
#
# Usage:
#   BACKEND_URL=https://abc123-8000.proxy.runpod.net \
#   AUTH_TOKEN=<the token you exported on the pod> \
#   bash deploy/runpod/bench_flame_steak.sh
#
# Optional:
#   PRESET=cloud    (default; alternatives: ultra, ultra_clean)
#   SCENE=flame_steak

set -euo pipefail

BACKEND_URL="${BACKEND_URL:?Set BACKEND_URL to the pod public URL (incl. port path).}"
SCENE="${SCENE:-flame_steak}"
PRESET="${PRESET:-cloud}"

if [[ "${PRESET}" != "ultra" && "${PRESET}" != "ultra_clean" && "${PRESET}" != "cloud" ]]; then
    echo "Refusing PRESET=${PRESET} -- for SOTA verification use ultra / ultra_clean / cloud." >&2
    exit 1
fi

AUTH_HEADER=()
if [[ -n "${AUTH_TOKEN:-}" ]]; then
    AUTH_HEADER=(-H "Authorization: Bearer ${AUTH_TOKEN}")
fi

echo "==> Health check: ${BACKEND_URL}/"
curl --fail --silent --show-error "${AUTH_HEADER[@]}" "${BACKEND_URL}/" | head -c 400
echo

echo "==> Submitting SOTA verification job"
echo "    scene=${SCENE}  mode=dynamic  preset=${PRESET}  nvs_eval=true  skip_foundation=false"
RESPONSE="$(curl --fail --silent --show-error -X POST \
    "${AUTH_HEADER[@]}" \
    -F "scene=${SCENE}" \
    -F "mode=dynamic" \
    -F "preset=${PRESET}" \
    -F "nvs_eval=true" \
    -F "skip_foundation=false" \
    "${BACKEND_URL}/process")"
echo "$RESPONSE"

JOB_ID="$(printf '%s' "$RESPONSE" | python -c "import json,sys; print(json.load(sys.stdin).get('job_id',''))")"
if [[ -z "$JOB_ID" ]]; then
    echo "Could not parse job_id from response. Aborting." >&2
    exit 1
fi

echo
echo "==> Job submitted: ${JOB_ID}"
echo "    Status:    ${BACKEND_URL}/status/${JOB_ID}"
echo "    Eval JSON: data/${SCENE}/output/eval/nvs_eval.json (when complete)"
echo "    Orbit:     ${BACKEND_URL}/jobs/${SCENE}/orbit.mp4"
echo
echo "    Tail status with:"
echo "      watch -n 30 \"curl -s -H \\\"Authorization: Bearer \$AUTH_TOKEN\\\" ${BACKEND_URL}/status/${JOB_ID}\""
echo
echo "    On completion, pull the eval JSON locally and run:"
echo "      scp <pod>:/path/to/4dgs-studio/data/${SCENE}/output/eval/nvs_eval.json ./"
echo "      python scripts/sota_compare.py ${SCENE} --eval-path nvs_eval.json"

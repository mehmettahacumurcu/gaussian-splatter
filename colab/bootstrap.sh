#!/usr/bin/env bash
# Colab environment bootstrap for the 4dgs-studio verification notebooks.
# Idempotent: safe to re-run. Run from the repo root (where requirements.txt is).
#
#   !bash colab/bootstrap.sh            # core deps + gsplat (Phase 2 notebook)
#   !bash colab/bootstrap.sh --colmap   # also install COLMAP (SOTA notebook)
#
# NOTE: requirements.txt pins numpy<2.0. On Colab (numpy 2.x) pip will downgrade
# it; if a later import complains about numpy, do Runtime -> Restart session once,
# then re-run the import cell (you do NOT need to re-run this script).
set -e

WITH_COLMAP=0
for a in "$@"; do [ "$a" = "--colmap" ] && WITH_COLMAP=1; done

echo "== torch (Colab preinstalled) =="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
assert torch.cuda.is_available(), "No CUDA GPU! Runtime -> Change runtime type -> GPU."
PY

echo "== python deps (requirements.txt; gsplat==1.5.3 is in there and JIT-builds) =="
pip install -q -r requirements.txt

if [ "$WITH_COLMAP" = "1" ]; then
  echo "== COLMAP (SOTA pipeline pose estimation) =="
  apt-get -qq update >/dev/null && apt-get -qq install -y colmap >/dev/null
  if colmap -h >/dev/null 2>&1; then echo "colmap: OK"; else echo "WARN: colmap not on PATH"; fi
fi

echo "== warm gsplat JIT (compile now, not mid-training; ~2-3 min first time) =="
python - <<'PY' || echo "NOTE: if this failed on numpy, do Runtime -> Restart, then re-run the import cell."
import torch, gsplat
print("gsplat", gsplat.__version__, "ok")
PY

echo "== bootstrap done =="

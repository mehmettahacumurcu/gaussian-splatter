#!/usr/bin/env bash
# Colab environment bootstrap for the 4dgs-studio verification notebooks.
# Idempotent: safe to re-run. Run from the repo root (where requirements.txt is).
#
#   !bash colab/bootstrap.sh                # core deps + gsplat (Phase 2 notebook)
#   !bash colab/bootstrap.sh --colmap       # + apt COLMAP (CPU SIFT only; CUDA-less build)
#   !bash colab/bootstrap.sh --colmap-cuda  # + conda-forge CUDA COLMAP for headless GPU SIFT
#                                           #   (installs a /usr/local/bin/colmap wrapper;
#                                           #    auto-falls-back to the apt CPU build on failure)
#
# NOTE: requirements.txt pins numpy<2.0. On Colab (numpy 2.x) pip will downgrade
# it; if a later import complains about numpy, do Runtime -> Restart session once,
# then re-run the import cell (you do NOT need to re-run this script).
set -e

WITH_COLMAP=0
WITH_COLMAP_CUDA=0
for a in "$@"; do
  [ "$a" = "--colmap" ]      && WITH_COLMAP=1
  [ "$a" = "--colmap-cuda" ] && WITH_COLMAP_CUDA=1
done

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

# ---------------------------------------------------------------------------
# --colmap-cuda : headless GPU-SIFT COLMAP via conda-forge (proven Colab T4,
# 2026-07-05). The apt COLMAP above has no CUDA and its GL SiftGPU crashes
# headless, forcing the pipeline onto --colmap-cpu (1-3 h exhaustive matching).
# This mode installs a conda-forge CUDA COLMAP with micromamba and shadows apt's
# binary with a wrapper on /usr/local/bin, so the pipeline gets GPU SIFT with
# ZERO code change (it resolves plain `colmap` off PATH via shutil.which).
# Measured 33.3x extract+match speedup vs CPU on a T4.
# ---------------------------------------------------------------------------

# Print the first CUDA colmap binary under known /content locations: the
# fresh-solve dir first, then any colmap-env* a Drive tarball may have unpacked
# (member dir is named colmap-env3 in the 2026-07-05 artifact). Empty if none.
_find_cuda_colmap() {
  ls -1 /content/colmap-cuda-env/bin/colmap /content/colmap-env*/bin/colmap \
    2>/dev/null | head -1
}

# Gate an env ($1 = path to a colmap binary). Returns 0 ONLY if the dynamic
# linker closure is complete (zero "not found") AND the binary is the CUDA build
# that runs headless. Echoes the reason on failure so a bad Colab run is legible.
_gate_colmap() {
  local bin="$1" lib missing banner rc
  [ -x "$bin" ] || { echo "  gate: $bin not executable"; return 1; }
  lib="$(dirname "$(dirname "$bin")")/lib"

  # (i) ldd must resolve everything -- a clean 3.11.1 env shows zero "not found";
  #     a broken closure is exactly how the 4.1.0 trap manifests (exit 127 later).
  missing=$(LD_LIBRARY_PATH="$lib" ldd "$bin" 2>/dev/null | grep 'not found' || true)
  if [ -n "$missing" ]; then
    echo "  gate: unresolved shared libraries:"; echo "$missing" | sed 's/^/    /'
    return 1
  fi
  # (ii) headless banner: conda colmap links Qt (needs offscreen platform) and its
  #      env lib dir on the loader path; must exit 0 and advertise CUDA
  #      (3.11.1 prints '... with CUDA').
  banner=$(QT_QPA_PLATFORM=offscreen LD_LIBRARY_PATH="$lib" "$bin" -h 2>&1); rc=$?
  if [ "$rc" -ne 0 ]; then echo "  gate: 'colmap -h' exited $rc"; return 1; fi
  if ! printf '%s\n' "$banner" | grep -qi 'with CUDA'; then
    echo "  gate: banner does not advertise CUDA:"
    printf '%s\n' "$banner" | head -2 | sed 's/^/    /'
    return 1
  fi
  return 0
}

# Full CUDA-COLMAP setup. On success installs /usr/local/bin/colmap and returns
# 0; on any recoverable failure returns non-zero and leaves NO wrapper (its
# absence is how the notebook detects CPU mode). Invoked as an `if` condition on
# purpose: that disables `set -e` inside the function, so a failed gate returns
# cleanly instead of aborting the whole bootstrap.
setup_colmap_cuda() {
  local mamba=/content/bin/micromamba
  local env_dir=/content/colmap-cuda-env
  local drive_tar=/content/drive/MyDrive/4dgs/colmap-cuda/colmap-env.tar.gz
  local colmap_bin="" lib ver

  # 1. micromamba static binary (once). '-C /content' MUST precede the member
  #    name: GNU tar applies -C only to members listed AFTER it.
  if [ ! -x "$mamba" ]; then
    echo "  micromamba: installing static binary..."
    mkdir -p /content/bin
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C /content bin/micromamba \
      || { echo "  micromamba download/extract failed"; return 1; }
    chmod +x "$mamba"
  fi
  [ -x "$mamba" ] || { echo "  micromamba missing after install"; return 1; }

  # 2. Obtain a GATED env: reuse -> Drive tarball -> fresh solve.
  # 2a. reuse an already-good env (idempotent re-run)
  colmap_bin=$(_find_cuda_colmap)
  if [ -n "$colmap_bin" ] && _gate_colmap "$colmap_bin"; then
    echo "  reusing existing CUDA env: $colmap_bin"
  else
    colmap_bin=""
    # 2b. Drive tarball restore (optimization; Drive is usually NOT mounted this
    #     early -- the notebooks mount it last -- so a missing tarball is normal).
    if [ -f "$drive_tar" ]; then
      echo "  restoring env from Drive tarball ($drive_tar)..."
      tar -xzf "$drive_tar" -C /content || true
      colmap_bin=$(_find_cuda_colmap)
      if [ -n "$colmap_bin" ] && _gate_colmap "$colmap_bin"; then
        echo "  Drive tarball env passed the gate"
      else
        echo "  Drive tarball env unusable -- falling through to a fresh solve"
        colmap_bin=""
      fi
    fi
    # 2c. FRESH SOLVE -- single micromamba create with ONLY colmap, version-pinned.
    #     WHY pin colmap=3.11.* (both traps verified live on Colab T4, 2026-07-05):
    #       * THE 4.1.0 TRAP: bare 'colmap=*=*cuda*' resolves to 4.1.0 cuda_129,
    #         whose conda-forge run-deps are BROKEN (binary links libfaiss /
    #         libOpenImageIO but the package declares neither => exit 127 at load)
    #         AND which RENAMED the --SiftExtraction.use_gpu option our 3.x pipeline
    #         calls pass ("unrecognised option").
    #       * THE PYTHON-PREPIN TRAP: python=3.10 env first, colmap after, also
    #         pulled a broken lib set. A single fresh solve of ONLY colmap avoids it.
    #     3.11.1 predates both: clean ldd closure + working headless GPU SIFT.
    if [ -z "$colmap_bin" ]; then
      echo "  fresh solve: micromamba create -p $env_dir 'colmap=3.11.*=*cuda*' ..."
      "$mamba" create -y -q -p "$env_dir" -c conda-forge 'colmap=3.11.*=*cuda*' \
        || { echo "  micromamba solve failed"; return 1; }
      colmap_bin="$env_dir/bin/colmap"
      _gate_colmap "$colmap_bin" || { echo "  fresh-solve env failed the gate"; return 1; }
    fi
  fi

  # 3. Install the PATH wrapper. /usr/local/bin precedes /usr/bin on Colab, so
  #    this shadows apt's colmap. The wrapper exports the two runtime vars the
  #    conda binary needs (offscreen Qt + env lib dir) then execs the real bin.
  #    Heredoc is unquoted so $lib / $colmap_bin expand now; \$@ / \${...} stay
  #    literal so they resolve when the wrapper itself runs.
  lib="$(dirname "$(dirname "$colmap_bin")")/lib"
  cat > /usr/local/bin/colmap <<EOF
#!/usr/bin/env bash
# Auto-generated by colab/bootstrap.sh --colmap-cuda -- wraps the conda-forge CUDA
# COLMAP so it runs headless (offscreen Qt + env lib dir on the loader path).
export QT_QPA_PLATFORM=offscreen
export LD_LIBRARY_PATH="$lib:\${LD_LIBRARY_PATH}"
exec "$colmap_bin" "\$@"
EOF
  chmod +x /usr/local/bin/colmap || { echo "  failed to install wrapper"; return 1; }

  ver=$(QT_QPA_PLATFORM=offscreen LD_LIBRARY_PATH="$lib" "$colmap_bin" -h 2>&1 \
        | grep -i colmap | head -1)
  echo "colmap-cuda: OK (${ver:-CUDA build}) -> /usr/local/bin/colmap"
  return 0
}

if [ "$WITH_COLMAP_CUDA" = "1" ]; then
  echo "== COLMAP CUDA (headless GPU SIFT via micromamba) =="
  # `if ! func`: the function runs in a condition context, which disables `set -e`
  # inside it, so a failed gate returns cleanly and we fall back below instead of
  # aborting the whole bootstrap.
  if ! setup_colmap_cuda; then
    echo "!! WARN: CUDA COLMAP setup failed -- falling back to the apt CPU build."
    echo "!!       The pipeline will then need --colmap-cpu (slow exhaustive matching)."
    rm -f /usr/local/bin/colmap        # no wrapper => notebook selects --colmap-cpu
    hash -r 2>/dev/null || true         # forget any just-removed wrapper location
    apt-get -qq update >/dev/null && apt-get -qq install -y colmap >/dev/null
    if colmap -h >/dev/null 2>&1; then echo "colmap: OK (apt CPU fallback)"; else echo "WARN: colmap not on PATH"; fi
  fi
fi

echo "== warm gsplat JIT (compile now, not mid-training; ~2-3 min first time) =="
python - <<'PY' || echo "NOTE: if this failed on numpy, do Runtime -> Restart, then re-run the import cell."
import torch, gsplat
print("gsplat", gsplat.__version__, "ok")
PY

echo "== bootstrap done =="

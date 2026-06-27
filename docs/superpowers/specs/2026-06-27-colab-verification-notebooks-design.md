# Colab verification notebooks — design

**Date:** 2026-06-27
**Status:** approved
**Goal:** Run the pending GPU verifications on Colab Pro+ (A100 / high-RAM), since the
local RTX 3060 Ti is the bottleneck for the strongest static-3DGS runs.

## Why

Two GPU-dependent tasks were blocked on local hardware:
1. **Phase 2 verification (P2-V)** — confirm the strip-to-core Phase 2 changes
   (`fourier_K=0` in static, single-frame static export) did not regress quality.
2. **SOTA trust-builder** — train the strongest static preset on a published benchmark
   scene and compare to baselines (`docs/SOTA_VERIFICATION.md`).

## Scope

Two notebooks, in order: a cheap sanity pass on `myroom`, then the high-value SOTA run on
a Mip-NeRF 360 scene. Out of scope: 4D / dynamic verification (project is static-first),
cloud deployment scaffolding.

## Approach (chosen: A)

Two thin notebooks + tracked shared helpers in `colab/`. The fragile parts (env build,
result parsing, copy-back) live once; notebooks are orchestration only. (`notebooks/` is
gitignored, so the dir is `colab/`.)

### Enabler — two flags on `scripts/static_3dgs.py`
The script hardcoded `skip_foundation=True` and did not enable NVS eval, so it could not
produce a held-out PSNR or use depth supervision. Added:
- `--nvs-eval` → `cfg.train.nvs_eval_enabled = True` (every-8 interleaved held-out eval →
  `output/eval/nvs_eval.json`, the input `scripts/sota_compare.py` reads).
- `--foundation` → `skip_foundation = not args.foundation` (default preserves old behavior;
  set it to run Metric3D depth supervision for max-quality runs).

## Components

```
colab/
  README.md            # how to open + run both notebooks
  bootstrap.sh         # idempotent env: pip deps, gsplat 1.5.3 JIT build, optional COLMAP
  verify_helpers.py    # read_metrics(), check_phase2(), gpu_mem_summary(), copy_results_to_drive()
  phase2_verify.ipynb  # NB1
  sota_verify.ipynb    # NB2
```

Each notebook: GPU check → clone repo at branch `chore/strip-to-core` → `bootstrap.sh` →
data → run `scripts/static_3dgs.py` → check/verdict → copy results to Drive.

## Data flow

- **`myroom` (NB1):** Google Drive mount + symlink into `data/myroom` (upload once;
  `frames/` + `colmap/`, `depth/` only if `--foundation`). Persists across sessions.
- **`garden` (NB2):** downloaded in-notebook from Mip-NeRF 360 (`360_v2.zip`), `images_4`
  (1/4-res, the 3DGS outdoor eval standard) copied into `data/garden/images/`. Pipeline
  then runs its own COLMAP + Metric3D depth — the honest end-to-end test.

## Verification / success criteria

- **NB1 (`check_phase2`):** held-out PSNR ≥ 27 dB (vs ~29 local baseline → no regression
  from `fourier_K=0`) **and** exactly 1 `frame_*.ply` (the P2-2 static-export fix). Closes P2-V.
- **NB2 (`sota_compare.py garden`):** verdict band vs 27.41 dB (3DGS) baseline —
  SOTA-tier / Below SOTA / Algorithmic gap.

## Decisions

- SOTA scene default `garden` (vanilla-3DGS baseline, apples-to-apples); `bonsai` available
  via one variable (indoor, tougher Scaffold-GS 32.70 dB target).
- Target A100 high-RAM runtime.

## Known caveats

- `requirements.txt` pins `numpy<2.0`; on Colab (numpy 2.x) pip downgrades it → a one-time
  `Runtime → Restart` may be needed before the gsplat import. Documented in `bootstrap.sh`.
- apt COLMAP on Colab may be CPU-only → garden pose estimation is the slow step. If too
  slow, switch to an indoor scene or (future) adapt Mip-NeRF 360's bundled COLMAP poses.
- Eval resolution (preset resize) vs the paper's exact eval res is approximate; the verdict
  bands (±1/±3 dB) tolerate it.

## Static verification done

`py_compile` (static_3dgs.py, verify_helpers.py), `bash -n bootstrap.sh`, `nbformat.validate`
on both notebooks. Runtime verification is the user's Colab run (the point of the notebooks).

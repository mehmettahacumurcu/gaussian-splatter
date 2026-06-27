# Colab verification notebooks

Two notebooks for running the static-3DGS verification on Colab Pro+ (A100 / high-RAM),
because the local RTX 3060 Ti is the bottleneck.

| Notebook | Purpose | GPU time |
|----------|---------|----------|
| `phase2_verify.ipynb` | Confirm the Phase 2 changes (`fourier_K=0`, single-frame static export) did not regress quality on `myroom`. Closes task **P2-V**. | ~10–25 min |
| `sota_verify.ipynb` | The trust-builder: train static `premium` on a Mip-NeRF 360 scene (`garden`) with foundation depth + NVS eval, then `sota_compare.py` for a baseline-anchored verdict. | ~1–4 h |

## How to open

1. Upload the `.ipynb` to Colab (or open it from GitHub: *File → Open notebook → GitHub →
   `mehmettahacumurcu/gaussian-splatter` → branch `chore/strip-to-core` → `colab/...`).
2. **Runtime → Change runtime type → A100 GPU** (High-RAM).
3. Run cells top to bottom. Each notebook clones the repo at branch `chore/strip-to-core`
   (where the Phase 2 work lives) and runs `colab/bootstrap.sh`.

## Data

- **Phase 2 (`myroom`)** comes from **Google Drive** (you upload it once). The notebook
  mounts Drive and symlinks it into `data/myroom`. You need `frames/` + `colmap/`
  (+ `depth/` only if you pass `--foundation`).
- **SOTA (`garden`)** is **downloaded in-notebook** from the Mip-NeRF 360 dataset — no
  Drive needed.

## Shared pieces

- `bootstrap.sh` — idempotent env setup (deps, gsplat 1.5.3 JIT build, optional COLMAP).
- `verify_helpers.py` — `check_phase2()`, `read_metrics()`, `gpu_mem_summary()`,
  `copy_results_to_drive()`.

The notebooks call the repo's own scripts (`scripts/static_3dgs.py`,
`scripts/sota_compare.py`) — the same code paths you run locally, just on a bigger GPU.

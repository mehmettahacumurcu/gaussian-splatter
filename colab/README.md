# Colab verification notebooks

Two notebooks for running the static-3DGS verification on Colab Pro+ (A100 / high-RAM),
because the local RTX 3060 Ti is the bottleneck.

| Notebook | Purpose | GPU time |
|----------|---------|----------|
| `phase2_verify.ipynb` | Confirm the Phase 2 changes (`fourier_K=0`, single-frame static export) did not regress quality on `myroom`. Closes task **P2-V**. | ~10–25 min |
| `sota_verify.ipynb` | The trust-builder: train static `premium` on a Mip-NeRF 360 scene (`garden`) with foundation depth + NVS eval, then `sota_compare.py` for a baseline-anchored verdict. | ~2–6 h |

## Verified results

| Date | Run | Verdict |
|------|-----|---------|
| 2026-07-03 | **P2-V** (`myroom`, balanced+foundation+nvs-eval, A100) | **PASS** — held-out PSNR **29.16 dB** (local baseline ~29.0, floor 27.0), SSIM 0.9071, LPIPS 0.1508, n=38; exactly 1 ply frame; final N=267,220. Phase 2 (`fourier_K=0` + static export) confirmed regression-free. |
| 2026-07-04 | **SOTA** (`garden`, premium+foundation+nvs-eval, CPU COLMAP, native `images_4` res, A100) | **Below SOTA** — held-out PSNR **24.94 dB** vs 27.41 dB (3DGS), ΔPSNR **−2.47 dB** (tunable band); SSIM 0.7801 (3DGS 0.868); **LPIPS-VGG 0.0776 beats the published 0.103**; n=23 (every-8). Read: `premium` is a perceptual preset (λ_lpips 0.15, λ_depth 0.15, aniso reg, 1M cap) so it trades PSNR/SSIM for LPIPS vs vanilla 3DGS's pure L1+SSIM at ~5–6M gaussians — part of the gap is by construction. Next lever: PSNR-parity `sota` preset (λ_lpips=0, no depth prior, 6M cap, vanilla densify schedule). |

## How to open

1. Upload the `.ipynb` to Colab (or open it from GitHub: *File → Open notebook → GitHub →
   `mehmettahacumurcu/gaussian-splatter` → branch `chore/strip-to-core` → `colab/...`).
2. **Runtime → Change runtime type → A100 GPU** (High-RAM).
3. Run cells top to bottom. Each notebook clones the repo at branch `chore/strip-to-core`
   (where the Phase 2 work lives) and runs `colab/bootstrap.sh`.

## Data

- **Phase 2 (`myroom`)** comes from **Google Drive** (you upload it once). The notebook
  mounts Drive, **copies the folder to the local VM disk**, and links `data/myroom` to the
  local copy — never train through the Drive mount itself: the trainer reads a frame +
  depth map every iteration and Drive FUSE reads are network round-trips (measured
  ~0.2 it/s vs GPU-bound from local disk). Upload the **entire** `data/myroom/`
  folder: `video.mp4` + `frames/` + `colmap/` + `depth/` + the hidden `.cache_markers/`.
  `video.mp4` is required (`_resolve_input` only accepts `images/` or `video.mp4`; a bare
  `frames/` is rejected) and `.cache_markers/` is what lets COLMAP + depth cache-hit —
  without it everything recomputes from scratch. Do **not** upload `output/` (a stale eval
  could shadow the fresh run's verdict).
- **SOTA (`garden`)** is **downloaded in-notebook** from the Mip-NeRF 360 dataset — no
  Drive needed.

## Outputs — viewing the splats

Each notebook's last step saves to Drive under `MyDrive/4dgs/`:
- `splats/<scene>.ply` — **the raw splat, saved first and unconditionally.** This is the
  "just show me the splat" path: open it in any 3DGS viewer (e.g. https://superspl.at/editor),
  no project needed. It survives even if the world-wrap step errors.
- `results/<scene>_results/` — eval JSON, ply, logs, orbit.mp4.
- `worlds/<scene>/` — a **walkable world bundle** (`output/world/0-world.ply` + collider),
  best-effort.

To walk it in the project: download `MyDrive/4dgs/worlds/<scene>/` into your local
`worlds/<scene>/`, run `cd frontend && npm run dev`, open the **Interactive** page, and pick
the scene in the world dropdown (entries for `garden` and `myroom` are pre-added to
`WorldSelector.tsx`).

## Shared pieces

- `bootstrap.sh` — idempotent env setup (deps, gsplat 1.5.3 JIT build, optional COLMAP).
- `verify_helpers.py` — `check_phase2()`, `read_metrics()`, `gpu_mem_summary()`,
  `save_splat_to_drive()`, `copy_results_to_drive()`, `wrap_and_save_world()`.

The notebooks call the repo's own scripts (`scripts/static_3dgs.py`,
`scripts/sota_compare.py`) — the same code paths you run locally, just on a bigger GPU.

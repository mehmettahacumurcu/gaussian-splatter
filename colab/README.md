# Colab verification notebooks

Two notebooks for running the static-3DGS verification on Colab Pro+ (A100 / high-RAM),
because the local RTX 3060 Ti is the bottleneck.

| Notebook | Purpose | GPU time |
|----------|---------|----------|
| `phase2_verify.ipynb` | Confirm the Phase 2 changes (`fourier_K=0`, single-frame static export) did not regress quality on `myroom`. Closes task **P2-V**. | ~10–25 min |
| `sota_verify.ipynb` | The trust-builder: train static `premium` on a Mip-NeRF 360 scene (`garden`) with foundation depth + NVS eval, then `sota_compare.py` for a baseline-anchored verdict. | ~2–6 h |
| `colmap_cuda_build.ipynb` | CUDA COLMAP feasibility — install or build a headless GPU-SIFT COLMAP, run a GPU vs CPU timing experiment on a real scene, persist the artifact to Drive. Feeds a future `bootstrap.sh --colmap-cuda`. | ~10–45 min |

## Verified results

| Date | Run | Verdict |
|------|-----|---------|
| 2026-07-03 | **P2-V** (`myroom`, balanced+foundation+nvs-eval, A100) | **PASS** — held-out PSNR **29.16 dB** (local baseline ~29.0, floor 27.0), SSIM 0.9071, LPIPS 0.1508, n=38; exactly 1 ply frame; final N=267,220. Phase 2 (`fourier_K=0` + static export) confirmed regression-free. |
| 2026-07-04 | **SOTA** (`garden`, premium+foundation+nvs-eval, CPU COLMAP, native `images_4` res, A100) | **Below SOTA** — held-out PSNR **24.94 dB** vs 27.41 dB (3DGS), ΔPSNR **−2.47 dB** (tunable band); SSIM 0.7801 (3DGS 0.868); **LPIPS-VGG 0.0776 beats the published 0.103**; n=23 (every-8). Read: `premium` is a perceptual preset (λ_lpips 0.15, λ_depth 0.15, aniso reg, 1M cap) so it trades PSNR/SSIM for LPIPS vs vanilla 3DGS's pure L1+SSIM at ~5–6M gaussians — part of the gap is by construction. Next lever: PSNR-parity `sota` preset (λ_lpips=0, no depth prior, 6M cap, vanilla densify schedule). |

## How to open

1. Upload the `.ipynb` to Colab (or open it from GitHub: *File → Open notebook → GitHub →
   `mehmettahacumurcu/gaussian-splatter` → branch `chore/strip-to-core` → `colab/...`).
2. **Runtime → Change runtime type → A100 GPU** (High-RAM).
3. Run cells top to bottom. The two verification notebooks clone the repo at branch
   `chore/strip-to-core` (where the Phase 2 work lives) and run `colab/bootstrap.sh`;
   `colmap_cuda_build.ipynb` is standalone (no clone — pure build experiment).

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

## Why a separate CUDA COLMAP notebook?

`bootstrap.sh --colmap` installs COLMAP via `apt-get`. That binary is compiled without CUDA:
its OpenGL SiftGPU backend requires a display and crashes immediately on Colab's headless VM
even when a GPU is present. So every pipeline run today passes `--colmap-cpu`, which serialises
exhaustive matching onto the CPU and wastes 1–3 h of A100 credits on a task that GPU SIFT can
finish in minutes.

`colmap_cuda_build.ipynb` answers the feasibility question first: it tries to install a
CUDA-enabled COLMAP via conda-forge (Rung 1, ~5 min), falls back to a source build with
`-DGUI_ENABLED=OFF` and a fat CUDA arch list if that fails (Rung 2, ~20–35 min), then runs
the real headless crash test — `feature_extractor` + `exhaustive_matcher` with
`--SiftExtraction.use_gpu 1` on a real scene and records a GPU vs CPU speedup. If the experiment
passes, a follow-up commit wires the tarball from Drive into `bootstrap.sh --colmap-cuda`.

## Shared pieces

- `bootstrap.sh` — idempotent env setup (deps, gsplat 1.5.3 JIT build, optional COLMAP).
- `verify_helpers.py` — `check_phase2()`, `read_metrics()`, `gpu_mem_summary()`,
  `save_splat_to_drive()`, `copy_results_to_drive()`, `wrap_and_save_world()`.

The notebooks call the repo's own scripts (`scripts/static_3dgs.py`,
`scripts/sota_compare.py`) — the same code paths you run locally, just on a bigger GPU.

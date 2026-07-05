# Colab notebooks

Notebooks for running the static-3DGS pipeline on Colab Pro+ (A100 / high-RAM), because
the local RTX 3060 Ti is the bottleneck: two verification notebooks, a CUDA-COLMAP build
experiment, and a production video→world runner.

| Notebook | Purpose | GPU time |
|----------|---------|----------|
| `phase2_verify.ipynb` | Confirm the Phase 2 changes (`fourier_K=0`, single-frame static export) did not regress quality on `myroom`. Closes task **P2-V**. | ~10–25 min |
| `sota_verify.ipynb` | The trust-builder: train static `premium` **or** `sota` (selectable `PRESET`) on a Mip-NeRF 360 scene (`garden`) with NVS eval, then `sota_compare.py` for a baseline-anchored verdict. Uses CUDA COLMAP (GPU SIFT) via `--colmap-cuda`. | ~1.5–4 h |
| `video_to_world.ipynb` | Video on Drive → max-quality splat + gravity-aligned walkable world (`ultra` preset). Extracts frames itself at native res, then runs photo-set mode. | ~5–8 h |
| `colmap_cuda_build.ipynb` | CUDA COLMAP feasibility — install or build a headless GPU-SIFT COLMAP, run a GPU vs CPU timing experiment on a real scene, persist the artifact to Drive. Feeds `bootstrap.sh --colmap-cuda` (now wired). | ~10–45 min |

## Verified results

| Date | Run | Verdict |
|------|-----|---------|
| 2026-07-03 | **P2-V** (`myroom`, balanced+foundation+nvs-eval, A100) | **PASS** — held-out PSNR **29.16 dB** (local baseline ~29.0, floor 27.0), SSIM 0.9071, LPIPS 0.1508, n=38; exactly 1 ply frame; final N=267,220. Phase 2 (`fourier_K=0` + static export) confirmed regression-free. |
| 2026-07-04 | **SOTA** (`garden`, premium+foundation+nvs-eval, CPU COLMAP, native `images_4` res, A100) | **Below SOTA** — held-out PSNR **24.94 dB** vs 27.41 dB (3DGS), ΔPSNR **−2.47 dB** (tunable band); SSIM 0.7801 (3DGS 0.868); **LPIPS-VGG 0.0776 beats the published 0.103**; n=23 (every-8). Read: `premium` is a perceptual preset (λ_lpips 0.15, λ_depth 0.15, aniso reg, 1M cap) so it trades PSNR/SSIM for LPIPS vs vanilla 3DGS's pure L1+SSIM at ~5–6M gaussians — part of the gap is by construction. Next lever: PSNR-parity `sota` preset (λ_lpips=0, no depth prior, 6M cap, vanilla densify schedule). |
| 2026-07-05 | **CUDA COLMAP feasibility** (`colmap_cuda_build.ipynb`, T4) | **PASS** — rung 1 (conda-forge `colmap 3.11.1 cuda` + `ceres-solver 2.2.0`, clean `ldd` closure, headless GPU SIFT exits 0); **33.3x** GPU-vs-CPU extract+match speedup (GPU 1.5 s vs CPU 49.9 s) on the **15-image Unsplash fallback** — matching quality is meaningless there, only the GPU code path is proven. Artifact + report at `MyDrive/4dgs/colmap-cuda/` (`colmap-env.tar.gz` + `colmap_cuda_report.json`). Now wired into `bootstrap.sh --colmap-cuda`. |
| 2026-07-05 | **myroom-max** (`video_to_world.ipynb`, ultra+foundation+native-res, CUDA COLMAP, A100 80GB) | **DONE** — first production run of the notebook: 2:05 1080p phone video → walkable world in **420.3 min** (~7 h, in the 5–8 h estimate). COLMAP total **10.3 min** (GPU SIFT — was 1–3 h on CPU); **285/502 frames registered** in one connected model (12 chunks; gaps = motion-blurred pans at ~0:26–0:36, 0:52–0:59, 1:06–1:13, 1:28–1:35, 1:38–1:46); 35,880 sparse pts. Final **N=3.15M** (hit 3M cap @ iter ~6k), train PSNR ~29–32 dB @ native 1920, 5.0 it/s final stage (65.6 GB VRAM). No held-out metrics (`RUN_EVAL=False`, keeper run). **Verdict: pipeline delivered, capture is the limiter** — desk/monitor region sharp (screen content legible); weak spots map to the data: white duvet swirls (textureless fabric), edge-of-coverage mush, ceiling holes (barely filmed). Next capture: slower pans, more light, locked exposure. |

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
- **`video_to_world` (any video)** takes a raw video from **Drive** and extracts frames
  **itself** at native resolution — avoiding the two traps of the preset video path: the
  fps-explosion (premium's `fps=30` makes ~3750 frames from a 2-min clip and kills the CPU
  COLMAP mapper) and the unconditional upscale (the scale filter forces the long edge up
  even when the source is smaller). The frames land in `data/<scene>/images/`, so the
  pipeline runs in **photo-set mode** (used as-is: no fps logic, no resize).

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
CUDA-enabled COLMAP via conda-forge (Rung 1, ~1–2 min), falls back to a source build with
`-DGUI_ENABLED=OFF` and a fat CUDA arch list if that fails (Rung 2, ~20–35 min), then runs
the real headless crash test — `feature_extractor` + `exhaustive_matcher` with
`--SiftExtraction.use_gpu 1` on a real scene and records a GPU vs CPU speedup.

**That experiment passed (2026-07-05, T4), so this is no longer hypothetical:
`bootstrap.sh --colmap-cuda` now exists.** It single-solves a pinned `colmap=3.11.*=*cuda*`
env with micromamba (or restores the Drive tarball when mounted), gates it (`ldd` closure +
a headless CUDA-banner check), and installs a `/usr/local/bin/colmap` wrapper that shadows
apt's binary so the pipeline gets GPU SIFT with no code change. `sota_verify.ipynb` calls
`--colmap-cuda` and passes `--colmap-cpu` only when the wrapper is absent (automatic
fallback if the CUDA setup fails).

## Shared pieces

- `bootstrap.sh` — idempotent env setup (deps, gsplat 1.5.3 JIT build, optional COLMAP).
- `verify_helpers.py` — `check_phase2()`, `read_metrics()`, `gpu_mem_summary()`,
  `save_splat_to_drive()`, `copy_results_to_drive()`, `wrap_and_save_world()`.

The notebooks call the repo's own scripts (`scripts/static_3dgs.py`,
`scripts/sota_compare.py`) — the same code paths you run locally, just on a bigger GPU.

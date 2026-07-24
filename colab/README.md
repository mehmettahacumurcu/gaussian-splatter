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
| `learned_quality_cache_audit.ipynb` | CPU-only safety gate: restores frame selection, verifies the owned COLMAP cache, qualifies tracks, and publishes the receipt required by the learned A100 run. | CPU only |
| `learned_quality_a100_experiment.ipynb` | Maximum-quality learned preprocessing and 120k-iteration Gaussian training, resumable from verified Drive milestones. Run the CPU audit first. | Several hours |
| `learned_quality_floor_recovery.ipynb` | Final bounded floor-hole diagnostic: restores the verified lineage once and compares one automatic 5k floor-seed arm with the preserved A100 legacy control. Never starts 120k or overwrites a production result. | ~1-3 h including Drive staging |
| `learned_quality_legacy_control_5k.ipynb` | Reproduce the passing legacy-control 5K arm once, then publish the byte-identical raw trainer PLY beside the production-polish candidate and its acceptance report. Never starts 120K. | ~1-2 h including one Drive staging pass |

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

## Isolated learned-quality CPU audit and A100 test

Use this two-notebook flow for the experimental maximum-quality room runner:

1. Open `learned_quality_cache_audit.ipynb` in a **CPU High-RAM** runtime, enter the
   input folder relative to `MyDrive` (for the current test, exactly `myroom_test`),
   and choose **Runtime -> Run all**.
2. Continue only after the notebook prints `TRACK AUDIT PASSED` and a verified COLMAP
   fingerprint. It flushes Drive and releases the CPU runtime automatically.
3. Open `learned_quality_a100_experiment.ipynb` in a fresh **A100 High-RAM** runtime,
   enter the same folder, and choose **Runtime -> Run all**.
4. Confirm visible audit, selection, and COLMAP cache hits before learned inference.
   During optical flow, the notebook prints inference, validation, residual, motion,
   and artifact-publication counters. Gaussian training prints live iteration output.
5. The final cell publishes the result to Drive, flushes pending writes, and releases
   the A100 runtime on either success or failure.

The notebook deliberately does not replace or modify the production result. It writes
the learned experiment beside the capture as:

```text
MyDrive/<input_folder>_learned_test_result/
```

Before Gaussian training starts, the notebooks publish an owned, hash-verified recovery
cache beside the capture:

```text
MyDrive/<input_folder>_learned_test_cache/
```

The A100 notebook prints every preprocessing stage as it runs. Recovery boundaries now
cover selection, verified COLMAP/base evidence, semantic evidence, optical flow, fused
masks, geometry, and final pre-training state. If a later stage fails, rerunning the same
pinned notebooks with unchanged input restores the newest valid milestone into fresh
local Colab storage instead of repeating completed work. The previous room run did not
save its completed optical-flow arrays, so optical flow must run one more time; after the
corrected run publishes the motion milestone, later retries can restore it. Changed
inputs, model pins, preprocessing settings, tools, or producer code cause a visible cache
miss and safe recomputation; incomplete or modified cache files are never trusted. Do
not select the `_learned_test_cache` folder as notebook input.

Compare that folder with the existing `MyDrive/<input_folder>_result/`. Inspect
`experiment_report.json`, `quality_report.json`, and the four images under
`diagnostics/`: masks, depth, geometry, and final render contact sheets. The output
contains `splat.ply` for SuperSplat or the project's existing viewer; the notebook does
not bundle a web viewer.

Local tests prove the contracts, orchestration, publishing isolation, and notebook
structure. Whether the learned path improves the room is intentionally undecided until
the real A100 capture finishes and those reports are reviewed.

## A100 structural diagnostic matrix

Use `learned_quality_training_ablation.ipynb` after the CPU track audit when a learned
run trains successfully but its scene is visually unstable or no longer recognizable.
The notebook requires an **A100 80 GB High-RAM** runtime. It stages the verified cache
and the historical learned result once, then runs seven deterministic 5K experiments:
legacy control, fixed topology, dense seeds, masks, depth, adaptive density, and the
full learned combination. It records checkpoints immediately before and after the
first density transition, plus fixed-camera and perturbed-camera renders.

The notebook stops after publishing `<input_folder>_training_ablation`. It never starts
a full 120K training run and never publishes a replacement PLY. Review
`diagnostic_summary.md`, `diagnostic_matrix.json`, and the plots under `plots/` before
changing the production pipeline.

## Final bounded floor-recovery test

After the CPU track audit and A100 structural diagnostic matrix have both completed for
the same input, run `learned_quality_floor_recovery.ipynb` in a fresh **A100 80 GB
High-RAM** runtime. Set `INPUT_FOLDER`, choose **Runtime -> Run all**, and let the final
cell publish `<input_folder>_floor_recovery_diagnostic`.

The notebook stages the verified lineage once, automatically fits a reliable floor
plane, finds empty floor cells, and trains one 5k candidate with at most 150,000
low-opacity floor seeds. It ignores semantic masks, excludes confirmed motion and sky,
uses the proven legacy density schedule, and stops after the comparison. A completed
diagnostic may be either accepted or safely rejected; read `decision.json`. Production
results are never replaced. See `docs/FLOOR_RECOVERY_DIAGNOSTIC.md` for exact gates,
monitoring commands, and output interpretation.

## Legacy-control 5K raw versus polished PLY

Use `learned_quality_legacy_control_5k.ipynb` when you want a directly viewable
before/after comparison of the configuration that passed the structural diagnostic.
It requires the same passing CPU track audit, verified final pre-training cache, and
completed A100 diagnostic matrix. Run it in a fresh **A100 80 GB High-RAM** runtime,
enter the original input folder (for example `myroom_test`), and choose
**Runtime -> Run all**.

The notebook stages the verified lineage once and trains exactly one deterministic
5K legacy-control arm. It publishes:

```text
MyDrive/<input>_legacy_control_5k_result/
  raw_legacy_control_5k.ply
  polished_legacy_control_5k.ply
  polish_report.json
  metrics.jsonl
  contact_005000.png
```

The raw file is copied byte-for-byte from the trainer. The polished file is the
candidate produced by the unchanged production polisher, which removes unsafe
low-opacity, oversized, highly anisotropic, and sparse-bound outliers and applies
the guarded outer-crop fade. Render gates still decide whether that candidate is
accepted. The candidate is retained for this diagnostic even when rejected, and
`polish_report.json` explains why. Neither file replaces a production result.

## Shared pieces

- `bootstrap.sh` — idempotent env setup (deps, gsplat 1.5.3 JIT build, optional COLMAP).
- `verify_helpers.py` — `check_phase2()`, `read_metrics()`, `gpu_mem_summary()`,
  `save_splat_to_drive()`, `copy_results_to_drive()`, `wrap_and_save_world()`.

The notebooks call the repo's own scripts (`scripts/static_3dgs.py`,
`scripts/sota_compare.py`) — the same code paths you run locally, just on a bigger GPU.

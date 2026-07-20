# Training Ablation Notebook Design

**Date:** 2026-07-21

**Status:** Approved for implementation

**Scope:** Standalone Colab diagnostic; production and legacy notebook behavior stays unchanged.

## Objective

Build one A100 High-RAM notebook that reuses the verified `myroom_test` selection,
geometry, and learned-evidence cache to identify which learned training change caused
the catastrophic quality regression. The notebook must restore Drive data exactly once,
run every experiment from the same local inputs, compare like-for-like quality, publish
only diagnostic evidence, and release the runtime automatically.

The notebook is not a replacement training pipeline and must never publish to
`<input>_result` or `<input>_learned_test_result`.

## Approaches Considered

### Cumulative ladder

Start with legacy training and add dense seeds, masks, depth, and adaptive density in
sequence. This is cheap, but the first failure can hide whether a later feature also
fails independently. Rejected as the primary design.

### Full factorial

Run every combination of four learned features. This is definitive but requires sixteen
or more GPU runs and wastes A100 credits. Rejected as the default.

### Isolated main effects with targeted interactions

Run a legacy control, four one-feature experiments, and the full learned stack. Only if
the isolated runs pass while the full stack fails, run targeted pairwise interactions.
This isolates the first-order cause while preserving a bounded GPU budget. Selected.

## Experiment Matrix

All experiments use the exact same registered cameras, RGB inventory, camera order,
random seed, resolution schedule, iteration count, and fixed evaluation views.

| ID | Dense seeds | Learned masks | Learned depth | Adaptive density |
|---|---:|---:|---:|---:|
| `legacy_control` | off | off | off | off |
| `dense_seeds_only` | on | off | off | off |
| `masks_only` | off | on | off | off |
| `depth_only` | off | off | on | off |
| `adaptive_density_only` | off | off | off | on |
| `full_learned` | on | on | on | on |

Each primary experiment runs for 5,000 iterations. The notebook records checkpoints at
500, 1,000, 2,500, and 5,000 iterations. A feature experiment is compared against the
legacy control at the same checkpoint, never against unrelated historical metrics.

If every isolated experiment passes but `full_learned` fails, the notebook runs only
the six pairwise combinations for 2,500 iterations. If a single isolated experiment
fails, no interaction runs are needed. The final report distinguishes `isolated_cause`,
`interaction_cause`, `multiple_independent_causes`, and `inconclusive`.

The rejected photometric correction is not an ablation variable because the failed
production run trained on original RGB. The 6M cap is reported but not ablated because
the failed run never reached the legacy 3M cap.

## One-Time Drive Staging

The notebook mounts Drive once and derives:

- input: `MyDrive/<input>`
- source cache: `MyDrive/<input>_learned_test_cache`
- output: `MyDrive/<input>_training_ablation`
- local staging: `/content/4dgs-ablation/<run_id>/inputs`

It validates ownership, CPU audit receipt, milestone manifests, hashes, available local
disk, and the pinned source commit before starting a GPU experiment. It then restores the
selection, geometry, final pretraining evidence, masks, depth, and dense-seed archive to
local storage once. All Drive reads and full-payload hashes end at this boundary.

The staged input tree is treated as immutable. Frames, masks, and depth are linked into
per-experiment scene directories from the local tree. Mutable COLMAP text files,
augmented `points3D.txt`, trainer outputs, and logs are private to each experiment.
No experiment may read a training payload through `/content/drive`.

The notebook writes a small JSON progress receipt after each completed experiment so a
runtime failure preserves the diagnostic state. Combined reports and images are copied
to Drive once after all experiments. Large model checkpoints and PLY files remain local
and are deleted after their statistics and preview renders are collected.

## Determinism and Isolation

The harness uses a fixed experiment seed for Python, NumPy, PyTorch, and CUDA. Camera
sampling uses a dedicated generator so model initialization cannot change the camera
sequence. Every experiment starts in a fresh subprocess, uses a fresh output directory,
and releases GPU tensors before the next experiment.

The legacy control uses the existing production density controller and trusted sparse
COLMAP points. Feature experiments change exactly the flag named by the row. Shared
training settings are held constant at the learned run's 720-pixel first stage so the
comparison measures training behavior rather than resolution changes.

## Measurements and Safety Gates

At each checkpoint, evaluate the same registered-view sample and record:

- masked and unmasked PSNR, SSIM, and L1;
- training loss components;
- Gaussian count and clone/split/prune totals;
- opacity quantiles;
- SH DC and higher-order coefficient quantiles;
- fraction of base colors clipping to black or white;
- scale and anisotropy quantiles;
- non-finite tensors;
- a fixed-view render contact sheet.

The versioned `ablation-v1` gate applies these exact rules:

- abort immediately on any non-finite model tensor;
- abort when the fraction of `alpha >= 0.1` Gaussians whose base RGB clips to pure
  white exceeds `max(10%, legacy_control + 5 percentage points)`;
- abort when more than 0.5% of `alpha >= 0.1` Gaussians have anisotropy above 30;
- abort when more than 0.5% of `alpha >= 0.1` Gaussians exceed
  `0.03 * robust_scene_extent`, where robust extent is the 99.5th percentile radius;
- after the 1,000-iteration checkpoint, abort when fixed-view PSNR is at least 3 dB
  below the legacy control at two consecutive matching checkpoints.

The legacy control is usable only if its 5,000-iteration fixed-view PSNR is at least
15 dB and it passes every structural rule. An aborted feature experiment is a diagnostic
failure result, not a notebook failure.

The notebook itself stops without further GPU work if local staging, source pinning,
cache integrity, camera joins, or the legacy control fails. A failed legacy control makes
all feature comparisons invalid and must produce `inconclusive` rather than blaming a
learned feature.

## Outputs

`MyDrive/<input>_training_ablation/` contains:

```text
_SUCCESS.json
ablation_report.json
ablation_summary.md
metrics.csv
plots/
  psnr_curves.png
  gaussian_count.png
  color_saturation.png
contact_sheets/
  <experiment>_<checkpoint>.png
experiments/
  <experiment>/receipt.json
diagnostics/
  staging_manifest.json
  environment.json
```

`_SUCCESS.json` means the diagnostic matrix completed and was published atomically. It
does not claim that any training configuration passed. The winning or failing features
are stated only in `ablation_report.json`.

## Notebook User Flow

1. Choose an A100 High-RAM runtime.
2. Enter a MyDrive-relative input folder such as `myroom_test`.
3. Run all cells.
4. Approve Drive access once.
5. Watch local staging, then experiment/checkpoint progress.
6. Read the printed cause classification and Drive report path.
7. The notebook flushes Drive and unassigns the runtime on success or failure.

## Implementation Boundaries

- Add a standalone ablation module and CLI under `experiments/learned_quality` and
  `scripts/`; do not overload the production runner.
- Add an optional deterministic camera-sampling generator at the shared trainer boundary;
  its default remains unchanged.
- Reuse existing cache, evidence, trainer, rendering, and notebook-generation code.
- Generate and commit `colab/learned_quality_training_ablation.ipynb`; do not hand-edit
  notebook JSON.
- Keep the legacy A notebook and the existing learned experiment notebook byte-for-byte
  behaviorally unchanged.

## Verification

Automated tests cover matrix generation, one-time staging, prohibition of Drive reads
during experiments, local scene isolation, deterministic camera order, early-abort
classification, interaction scheduling, atomic report publication, notebook source pin,
Drive folder derivation, and runtime release. A CPU smoke test uses tiny synthetic scenes;
no local test requires an NVIDIA GPU or Google Drive.

# A100 Structural Diagnostic Matrix Design

**Date:** 2026-07-21

**Status:** Approved for implementation

**Scope:** Standalone A100 diagnostic. It stops after publishing the matrix and never starts a 120K production run.

## Objective

The existing ablation detects catastrophic Gaussian parameters, but the successful
120K learned run proved that the absence of giant white splats is not sufficient. Its
published PLY was still spatially unrecognizable: the room layout, walls, floor, and
furniture could not be understood in free view.

Build an A100-only forensic notebook that separates parameter stability from structural
fidelity. It must reuse the verified Drive cache once, compare controlled training
variants on identical local inputs, identify the first transition that destroys room
structure, compare against the preserved 120K failed-quality run, publish a diagnostic
matrix, and release the runtime.

## Approaches Considered

### Repair only the iteration-500 gate

Let the existing six variants continue past the first density event and keep the current
checkpoints. This is inexpensive, but it cannot distinguish a bad initialization from a
density-induced regression and still treats recognizability as an indirect PSNR signal.

### Geometry-only audit before training

Analyze cameras, sparse geometry, dense seeds, and masks without optimizing Gaussians.
This is cheap and can reveal invalid inputs, but it cannot show whether an initially
usable scene is later damaged by density control, opacity resets, or learned losses.

### Two-axis forensic timeline matrix

Measure both numerical stability and fixed-view structural fidelity before and after
the first density event, then continue every primary experiment to a bounded 5K
endpoint. Add a fixed-topology control and compare the results to the preserved 120K
run. This gives the strongest diagnosis without paying for another full run. Selected.

## Experiment Matrix

Every experiment uses the same verified camera model, source frames, camera sampling
sequence, random seed, 720-pixel training resolution, loss defaults, and evaluation
views.

| ID | Dense seeds | Masks | Depth | Adaptive density | Density events |
|---|---:|---:|---:|---:|---:|
| `legacy_control` | off | off | off | off | on |
| `fixed_topology_control` | off | off | off | off | off |
| `dense_seeds_only` | on | off | off | off | on |
| `masks_only` | off | on | off | off | on |
| `depth_only` | off | off | on | off | on |
| `adaptive_density_only` | off | off | off | on | on |
| `full_learned` | on | on | on | on | on |

All seven primary experiments run for 5,000 iterations. No conditional pairwise or 120K
training is launched by this notebook. The matrix is the terminal product of the run.

## Forensic Timeline

The trainer records the same fixed views at:

- iteration 0: initialized Gaussian scene;
- iteration 100: early appearance fitting;
- iteration 499: immediately before the first configured density event;
- iteration 500: immediately after the first density event;
- iteration 600: immediate recovery window;
- iteration 1,000: early convergence;
- iteration 2,500: middle convergence;
- iteration 5,000: bounded diagnostic endpoint.

Iteration 0 is an explicit trainer diagnostic callback before optimization. Iteration 499
and 500 are separate observations; the latter is intentionally collected after density
and opacity operations, matching the state that proceeds into the next optimization
step.

## Measurements

### Stability axis

At every checkpoint record:

- Gaussian count plus clone, split, prune, and opacity-reset deltas;
- visible-white, high-anisotropy, oversized, out-of-bounds, and non-finite fractions;
- opacity, scale, anisotropy, SH-DC, and SH-rest quantiles;
- robust scene extent and Gaussian centroid covariance eigenvalues;
- fractions outside the camera-centre scene envelope and outside registered camera
  frusta.

### Structural-fidelity axis

Use a deterministic coverage sample of registered training cameras. For every sampled
camera, save ground truth, rendered RGB, absolute error, and rendered depth. Aggregate:

- unmasked and validity-masked PSNR, SSIM, and L1;
- LPIPS when the installed model is available, with an explicit unavailable marker
  rather than silently substituting zero;
- edge L1 and edge correlation to emphasize walls, furniture boundaries, and layout;
- valid-pixel coverage and rendered alpha coverage;
- rendered-depth finite coverage, median, spread, and adjacent-view consistency.

Also render deterministic small translations around the fixed cameras. Training-camera
replay that looks acceptable while perturbed views collapse is classified as weak 3D
consistency rather than healthy reconstruction.

The report includes side-by-side fixed-view and perturbed-view contact sheets so the
room layout can be inspected directly instead of relying on a single scalar.

## Historical 120K Reference

The notebook reads the preserved learned-result folder once during local staging. It
imports the old run manifest, experiment report, quality report, density history, final
render contact sheet, and final PLY when available. The historical analyzer records:

- its 120,000-iteration configuration and training duration;
- final Gaussian count and PLY parameter distributions;
- density/reset trajectory summaries;
- geometry acceptance failures and registration statistics;
- preserved render diagnostics.

Historical data is a reference row, not a control checkpoint. Missing or unsupported
properties are reported as unavailable and never make the new matrix fail.

## Decisions and Safety Gates

The old iteration-500 hard structural abort is removed. Early white, anisotropy, and
oversized measurements are telemetry because the L4 diagnostic showed the legacy
control also exceeded the old oversized threshold.

An experiment stops immediately only for:

- non-finite model tensors;
- CUDA out-of-memory or an infrastructure failure;
- a render failure that prevents diagnostic evidence from being produced.

All other quality problems are classified after the 5K timeline. A persistent failure
requires at least two post-density checkpoints, except for a severe structural event
that exceeds a versioned emergency limit. The fixed-topology control is used to decide
whether the first density transition, rather than the initial geometry or appearance
loss, caused the regression.

One experiment failure is recorded and the remaining experiments continue. A failed
experiment never deletes completed results from another row.

## Diagnosis Categories

The final report may assign multiple evidence-backed causes:

- `input_geometry_failure`: initialization is already unrecognizable;
- `density_transition_failure`: legacy degrades at 500 while fixed topology remains
  structurally stable;
- `dense_seed_failure`, `mask_failure`, `depth_failure`, or
  `adaptive_density_failure`: an isolated feature underperforms the matched control;
- `interaction_failure`: isolated rows are usable but the full learned row is not;
- `weak_3d_consistency`: fixed camera replay is materially better than perturbed views;
- `export_or_viewer_suspect`: local diagnostic renders remain recognizable while the
  historical exported PLY does not;
- `inconclusive`: required evidence is missing or controls are unusable.

## One-Time Drive Boundary

The notebook mounts Drive once and restores the verified selection, geometry,
pretraining evidence, and historical result into local storage. Every experiment reads
only the immutable local staging tree. Drive is used again only for small durable
progress receipts and final atomic publication.

The result is written to `MyDrive/<input>_training_ablation`. It contains no production
PLY and cannot overwrite `<input>_result` or `<input>_learned_test_result`.

## Output Contract

```text
<input>_training_ablation/
  _SUCCESS.json
  diagnostic_matrix.json
  diagnostic_summary.md
  metrics.csv
  historical_120k.json
  environment.json
  staging_manifest.json
  plots/
    fixed_view_quality.png
    structural_fidelity.png
    gaussian_count.png
    density_events.png
  contact_sheets/
    <experiment>_<checkpoint>_fixed.png
    <experiment>_<checkpoint>_perturbed.png
  experiments/
    <experiment>/receipt.json
```

`_SUCCESS.json` means the complete requested diagnostic matrix was published. It does
not authorize or start a full-quality run.

## Notebook Flow

1. Select an A100 High-RAM runtime.
2. Enter a MyDrive-relative input folder.
3. Run all cells and approve Drive once.
4. Watch local staging and the seven timeline rows.
5. Read the printed cause classification and result path.
6. The notebook flushes Drive and releases the runtime.

There is deliberately no cell that launches a 120K run.

## Verification

CPU tests cover the matrix, iteration-zero callback, pre/post-density checkpoint order,
fixed-topology behavior, persistence-based decisions, structural and edge metrics,
historical PLY parsing, experiment-continuation behavior, output schema, notebook A100
preflight, source pin, Drive boundary, automatic runtime release, and absence of a 120K
execution path. Existing production runner defaults remain unchanged.

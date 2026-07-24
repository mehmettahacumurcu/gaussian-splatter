# Legacy-Control Native-1080p 30K Local Checkpoint Run

**Status:** Draft for written review
**Date:** 2026-07-24
**Branch:** `feature/learned-quality-a100`

## Objective

Build one A100 High-RAM Colab notebook that extends the successful raw
`legacy_control` experiment into a bounded quality-limit test:

- use the same verified 800-frame selection, COLMAP solution, and pre-training
  lineage that produced the recognizable raw 5K splat;
- restore that lineage from Drive exactly once per Colab session;
- train one continuous optimizer from iteration 0 through iteration 30,000;
- train at the selected frames' native 1920x1080 resolution, without
  upscaling;
- export raw, unpolished PLY snapshots at 5K, 10K, 15K, 20K, 25K, and 30K;
- retain local training checkpoints at the same boundaries;
- write all run outputs only under `/content`;
- leave the Colab runtime, Drive mount, and completed notebook session alive.

The run is intentionally not a new learned-quality experiment. It isolates
whether longer fixed-topology refinement improves the promising raw 5K result.

## Evidence Behind the Design

The successful raw 5K legacy-control result was recognizable and preserved
substantially more room structure than the learned 120K output. Its relevant
training properties were:

- legacy COLMAP initialization;
- pure photometric reconstruction objective;
- no learned depth supervision;
- no learned masks in the training loss;
- no learned dense seeds;
- no deformation;
- no adaptive-density experiment;
- density events from iteration 500 through iteration 4,499;
- one opacity reset at iteration 3,000;
- a six-million-Gaussian safety cap;
- deterministic frame/camera sampling.

The raw 5K model ended immediately after the final density period. Gaussians
created near iteration 4,500 therefore received only about 500 fixed-topology
updates. The new run preserves the successful density behavior, then gives the
same topology another 25,500 refinement iterations. This directly tests the
hypothesis that the wardrobe-colored floaters are late density products that
can settle or lose opacity with more optimization.

## Non-Goals

This notebook will not:

- run the full learned-quality reconstruction pipeline;
- recompute frame selection, COLMAP, depth, flow, semantics, or masks;
- introduce learned losses, learned seeds, or learned density behavior;
- continue from the 5K PLY with a fresh optimizer;
- start a second trainer for any checkpoint;
- run the PLY polish stage;
- generate reports, receipts, contact sheets, or diagnostic matrices;
- publish PLYs, checkpoints, metrics, or reports to Drive;
- unmount Drive or automatically release the Colab runtime;
- start a 120K production run.

## Notebook Contract

The notebook exposes only the input folder parameter and a fixed runtime
profile:

```python
INPUT_FOLDER = "myroom_test"
RUNTIME_PROFILE = "a100_legacy_control_native_1080p_30k_local"
```

The profile is intentionally not configurable from notebook form fields. This
prevents an accidental change from turning the comparison into a different
experiment.

The notebook must verify:

- an NVIDIA A100 with at least 75 GiB VRAM;
- enough free `/content` disk for staging, checkpoints, and six PLYs;
- the pinned repository revision;
- the matching verified CPU track-audit receipt;
- the matching verified final pre-training cache;
- an exact frame/camera join before training;
- native selected-frame dimensions of 1920x1080 for this experiment.

The joined camera set must resolve to exactly 1920x1080. A lower-resolution
source is not upscaled, and a higher or mixed-resolution source is not silently
resized under this fixed experiment profile. Any mismatch fails before
training, keeping the advertised comparison unambiguous.

## Cache and Staging Flow

Drive is an input source only:

1. mount Drive;
2. locate the matching verified track-audit and final pre-training lineage;
3. verify cache manifests and artifact hashes;
4. copy the complete required payload once into the run's local staging root;
5. construct every training input from the local copy;
6. perform no further Drive cache reads during training;
7. leave Drive mounted at notebook completion.

The implementation must not restore the same milestone separately for each
checkpoint. All six PLY snapshots come from one trainer process and one local
staging tree.

## Training Configuration

The run uses the legacy-control settings that succeeded at 5K, with only
resolution and total duration changed:

| Setting | Value |
|---|---:|
| Total iterations | 30,000 |
| Training resolution | Native 1920x1080 |
| Native-resolution mode | Enabled |
| Camera sampling seed | 1701 |
| Loss | L1 + existing legacy SSIM term |
| Learned depth loss | Disabled |
| LPIPS loss | Disabled |
| Learned masks | Disabled |
| Learned dense seeds | Disabled |
| Deformation | Disabled |
| Density start | 500 |
| Density end | 4,500 exclusive |
| Density interval | 100 |
| Opacity reset interval | 3,000 |
| Maximum Gaussians | 6,000,000 |
| PLY/checkpoint interval | 5,000 |

Because opacity reset is evaluated only inside the density window, this
configuration performs exactly one reset at iteration 3,000. No cloning,
splitting, pruning, or reset occurs after iteration 4,499. Iterations
4,500-30,000 are fixed-topology refinement.

The trainer must be invoked once. The optimizer, learning-rate schedule,
Gaussian tensors, and sampling generator remain continuous across every 5K
boundary.

## Local Output Contract

Each run creates a fresh root:

```text
/content/legacy_control_1080p_30k/<run_id>/
  inputs/
  training/
    logs/
      metrics.jsonl
      events.log
  ply/
    legacy_control_005000.ply
    legacy_control_010000.ply
    legacy_control_015000.ply
    legacy_control_020000.ply
    legacy_control_025000.ply
    legacy_control_030000.ply
  checkpoints/
    legacy_control_005000.pt
    legacy_control_010000.pt
    legacy_control_015000.pt
    legacy_control_020000.pt
    legacy_control_025000.pt
    legacy_control_030000.pt
```

`metrics.jsonl` and `events.log` are the trainer's ordinary live logs, not
generated experiment reports. No `report/` directory or report JSON is
created.

Every PLY is the raw trainer state at that exact iteration. No normalization,
polish, filtering, or quality gate modifies it.

Each checkpoint contains enough state to resume the same run locally:

- iteration number;
- Gaussian model state;
- deformation state, even though deformation is disabled;
- optimizer state;
- learning-rate scheduler state when applicable;
- random generator states needed for deterministic camera sampling;
- scene extent and spherical-harmonic degree.

Checkpoint and PLY writes use a temporary sibling file followed by an atomic
rename. Earlier successful snapshots are never deleted when a later boundary
fails.

## Live Operation

The notebook prints:

- the pinned source revision;
- the local run root;
- staging progress;
- the current training iteration from `metrics.jsonl`;
- the PLY/checkpoint path whenever a 5K boundary is completed.

It also includes a short, non-destructive monitoring cell that tails the latest
metric rows, lists completed PLYs and checkpoints, and displays GPU usage.

The main cell returns normally after iteration 30,000. It does not call
`drive.flush_and_unmount()` or `runtime.unassign()`. The user can inspect and
download files from the Colab file browser while the session remains
connected.

## Failure Behavior

Preflight and staging failures occur before GPU training. During training:

- a failed later snapshot does not remove earlier completed snapshots;
- a Python exception prints its traceback and preserves the local run tree;
- the notebook does not publish a failure report to Drive;
- the notebook does not release the runtime in a `finally` block;
- disk-space guards run before training and before each large snapshot;
- a non-finite model state fails the current export before rename.

The user accepts that `/content` is ephemeral. If Colab disconnects or the
runtime is recycled before files are downloaded, local-only outputs can be
lost.

## Test Strategy

Implementation starts with failing tests that prove:

1. the profile is native 1920x1080 and exactly 30,000 iterations;
2. density stops at 4,500 and only the 3,000 opacity reset is possible;
3. learned features and polish are disabled;
4. a single trainer instance serves all six export boundaries;
5. exports use exact zero-padded filenames;
6. PLY and resumable checkpoint writes are atomic;
7. completed earlier artifacts survive a later export failure;
8. cache staging happens once and training uses only local staged paths;
9. no Drive result publication, report generation, unmount, or runtime release
   is reachable;
10. the generated notebook contains the fixed profile and a live-monitoring
    cell.

Focused unit tests cover the new run module, script entry point, and notebook
generator. Existing learned-quality and legacy-control export tests remain
unchanged and must continue to pass.

## Acceptance Criteria

The implementation is complete when:

- the generated notebook is pinned to the verified implementation revision;
- all focused and regression tests pass;
- one invocation produces the six named raw PLY snapshots from one continuous
  30K trainer run;
- no polish or report artifacts are created;
- no result is written to Drive;
- the notebook remains connected and usable after success or failure.

# Legacy-Control 5K PLY Export Design

**Date:** 2026-07-24

**Status:** Proposed

## Objective

Provide one bounded A100 notebook that exactly reproduces the passing
`legacy_control` row from the completed structural diagnostic and preserves its
raw 5,000-iteration PLY for free-view inspection.

The notebook is an export experiment, not a production run. It must never run
the full learned feature stack, the complete diagnostic matrix, PLY polish, or
120,000 iterations.

## Reproduced Candidate

The exported candidate keeps the inputs and controls used by the successful
diagnostic row:

- the verified smart selection and output-first Round-0 reconstruction;
- the accepted sparse camera model and original RGB frames;
- deterministic seed `1701` and deterministic camera sampling;
- the Ultra static training base with the diagnostic overrides;
- 5,000 iterations at a fixed 720-pixel long edge;
- standard density events from iteration 500 through 4,500;
- no learned dense-seed augmentation;
- no learned validity mask in the training loss;
- no learned depth supervision;
- no adaptive-density controller.

The implementation must select the existing `legacy_control` variant and use
the existing ablation scene builder and training adapter. It must not duplicate
the configuration in a second hand-maintained implementation.

## Notebook

Add:

```text
colab/learned_quality_legacy_control_5k.ipynb
```

The notebook:

1. accepts a MyDrive-relative `INPUT_FOLDER`;
2. requires an A100-class GPU with at least 75 GiB VRAM and High-RAM local
   memory;
3. mounts Drive once;
4. clones and pins the reviewed source revision;
5. installs the same isolated learned environment and verifies the model
   manifest;
6. restores the verified selection and final pre-training lineage once into
   session-local storage;
7. runs only `legacy_control`;
8. prints checkpoint metrics at 0, 100, 499, 500, 600, 1,000, 2,500, and
   5,000;
9. publishes the raw PLY and compact evidence atomically;
10. flushes Drive and releases the runtime on success or failure.

## Drive Inputs and Provenance

The run requires:

- a matching passing CPU track-audit receipt;
- the verified output-first pre-training cache;
- the completed training-ablation report for the same source and selection;
- a passing `legacy_control` receipt at iteration 5,000.

The small historical receipt is used only as a provenance target. Training
still executes again because the diagnostic intentionally deleted its raw PLY.
The new receipt records both the historical checkpoint metrics and the
reproduced checkpoint metrics so divergence is visible.

All large cache artifacts are restored exactly once. Training reads only the
immutable session-local snapshot.

## Output Contract

The result is isolated from both production result folders:

```text
MyDrive/<input>_legacy_control_5k_result/
  _SUCCESS.json
  splat.ply
  receipt.json
  metrics.jsonl
  run_manifest.json
  contact_005000.png
  contact_005000_fixed.png
  contact_005000_perturbed.png
  provenance.json
```

`splat.ply` is the trainer's raw iteration-5,000 PLY. No polish or geometry
acceptance wrapper may alter it.

Publication uses a temporary sibling directory followed by an atomic owned
replacement. It may replace only a folder carrying this exporter’s ownership
marker and must never modify:

- `<input>_result`;
- `<input>_learned_test_result`;
- `<input>_training_ablation`;
- `<input>_floor_recovery_diagnostic`.

## Quality and Failure Handling

Success requires:

- all eight diagnostic checkpoints;
- completion at exactly 5,000 iterations;
- the existing `legacy_control` gate to pass;
- a finite, structurally valid static PLY;
- successful publication of every required file.

If cache, provenance, training, gate, PLY validation, or publication fails, the
runner writes a small durable failure receipt to
`<input>_legacy_control_5k_failures/<run-id>.json`. It does not publish a
partial result and does not fall back to learned features or a longer run.

## Implementation Boundaries

Add a focused contract, runner, CLI, and notebook generator under the existing
`experiments.learned_quality` package. Reuse:

- output-first cache staging from the ablation runner;
- `primary_variants()` to select `legacy_control`;
- `run_ablation_experiment()` for the exact training path;
- existing diagnostic contact-sheet generation;
- existing static PLY validation and atomic copy helpers;
- existing notebook bootstrap, manifest verification, Drive flushing, and
  runtime-release patterns.

The general diagnostic matrix remains unchanged and continues deleting its
temporary PLYs.

## Verification

CPU tests must prove:

- only `legacy_control` is selected;
- its feature set is empty and density events remain enabled;
- the resolved run is exactly 5,000 iterations and fixed 720p;
- staging is local and performed once;
- the historical receipt must match the active source and selection;
- the raw PLY is preserved byte-for-byte;
- all required evidence is published atomically;
- unrelated result folders cannot be overwritten;
- no full matrix, learned-feature arm, polish step, or 120K path is reachable;
- notebook source pin, A100 preflight, unbuffered output, Drive flush, and
  runtime release are present.


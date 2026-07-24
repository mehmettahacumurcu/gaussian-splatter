# Legacy-Control 5K PLY Export Design

**Date:** 2026-07-24

**Status:** Approved

## Objective

Provide one bounded A100 notebook that exactly reproduces the passing
`legacy_control` row from the completed structural diagnostic and preserves
both its raw 5,000-iteration PLY and a production-algorithm polish candidate
for free-view inspection.

The notebook is an export experiment, not a production run. It must never run
the full learned feature stack, the complete diagnostic matrix, or 120,000
iterations.

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
9. runs the production polish filter against a diagnostic acceptance proxy
   without changing the restored reconstruction or production gates;
10. publishes the raw PLY, the polish candidate, and compact comparison
    evidence atomically;
11. flushes Drive and releases the runtime on success or failure.

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
  raw_legacy_control_5k.ply
  polished_legacy_control_5k.ply
  polish_report.json
  receipt.json
  metrics.jsonl
  run_manifest.json
  contact_005000.png
  contact_005000_fixed.png
  contact_005000_perturbed.png
  provenance.json
```

`raw_legacy_control_5k.ply` is the trainer's raw iteration-5,000 PLY and must
be preserved byte-for-byte.

`polished_legacy_control_5k.ply` is the candidate emitted by the existing
production `polish_static_ply` algorithm. The restored output-first Round-0
geometry did not pass the production reconstruction gate, so the isolated
exporter must construct an in-memory diagnostic acceptance proxy that changes
only `GateDecision.passed`, `GateDecision.failures`, and
`GateDecision.retry_recommended`. The original reconstruction contract remains
unchanged and its strict failures are recorded in `polish_report.json`.

The exporter publishes the candidate even when the polish regression gate
rejects it and would normally select the raw PLY. This is deliberate: the
purpose is manual raw-versus-polished comparison, not production selection.
If polish cannot produce a structurally valid non-empty candidate, the export
fails rather than publishing a fake or duplicate polished file.

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
- finite, structurally valid raw and polish-candidate PLYs;
- a byte-identical published raw PLY;
- a polish report containing the original geometry failures, removal counts,
  opacity-mass loss, render deltas, acceptance decision, and rejection reasons;
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
- existing production PLY polish, static PLY validation, and atomic copy
  helpers;
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
- the polish candidate is distinct, structurally valid, and is preserved even
  when its regression gate rejects it;
- only the isolated diagnostic proxy can bypass reconstruction acceptance;
- the original reconstruction decision and strict failures are not mutated;
- the comparison report exposes Gaussian counts, opacity loss, render deltas,
  polish acceptance, and rejection reasons;
- all required evidence is published atomically;
- unrelated result folders cannot be overwritten;
- no full matrix, learned-feature arm, or 120K path is reachable;
- notebook source pin, A100 preflight, unbuffered output, Drive flush, and
  runtime release are present.

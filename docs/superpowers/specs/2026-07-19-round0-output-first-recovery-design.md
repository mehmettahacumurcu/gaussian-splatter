# Round 0 Output-First Recovery Design

**Date:** 2026-07-19

**Status:** Review requested

**Target:** Produce one transparent best-effort Gaussian splat from the durable
Round 0 room checkpoints before spending more A100 time on Round 1.

## Context

The room experiment has complete, immutable Round 0 checkpoints for selection,
base evidence, semantic evidence, motion evidence, and fused masks. Its two
evaluated geometry candidates were structurally valid but failed three strict
quality gates:

| Candidate | Registered | Maximum interior gap | Median reprojection | Points |
| --- | ---: | ---: | ---: | ---: |
| Classical | 661/800 (82.63%) | 9.590 s | 1.217 px | 65,518 |
| Learned hybrid | 651/800 (81.37%) | 2.585 s | 1.194 px | 61,137 |

The candidate directories were local to a terminated Colab runtime and were not
published as a geometry milestone. The Round 0 base-evidence checkpoint still
references its durable classical COLMAP prepass, whose dominant model registered
658/800 frames and contains 65,367 sparse points. Round 1 has durable selection,
COLMAP, base-evidence, and semantic checkpoints, but it has no motion, masks,
evaluated geometry, or final pretraining checkpoint.

Repeating strict Round 0 geometry and then finishing Round 1 would risk another
multi-hour A100 run without a splat. The immediate product decision is therefore
to recover the complete Round 0 evidence branch, accept its cached dominant
COLMAP model only under an explicit guarded policy, and train it before resuming
Round 1 work.

## Decision

The isolated learned-quality notebook gains an opt-in recovery mode named
`round0_output_first_v1`. It restores the complete Round 0 masks branch, uses
the exact COLMAP checkpoint referenced by that branch, measures the restored
model without rerunning COLMAP, and proceeds when either the existing strict
gate passes or the guarded best-effort gate below passes.

The strict gate result is never rewritten. A separate experiment-only
acceptance record states whether training used `strict` or `best_effort`, binds
that decision to the selection and model digests, and preserves every strict
failure and metric in the reports. The result folder remains
`<input_folder>_learned_test_result` and is marked as best effort when applicable.

The mode is fixed inside the standalone A100 experiment notebook. It is not
added to the frontend, API, notebook generator, legacy notebook, or production
static pipeline presets. All existing callers continue to use strict geometry
validation by default.

## Options considered

### Output-first Round 0 recovery

Restore the deepest complete Round 0 evidence branch, reuse its referenced
COLMAP model, finish final pretraining, and train. This has the highest chance
of producing a splat today and is selected.

### Timed Round 1 attempt followed by Round 0 fallback

Round 1 would still need optical flow, fused masks, and geometry comparison.
Those stages can consume several hours before fallback begins, so the result is
not time-bounded enough for the immediate goal.

### Strict quality-only execution

This preserves the original research gate but has already produced no splat
after substantial paid compute. It remains the policy for normal learned-quality
runs and for the later Round 1 comparison, but not for this recovery run.

## Recovery branch selection

Recovery is lineage-based, not newest-folder-based:

1. Reproduce or restore the canonical initial Smart selection for the input.
2. Resolve the newest valid `masks` milestone whose dependency graph terminates
   at that initial selection fingerprint.
3. Restore its `base_evidence`, `semantic`, `motion`, and `masks` dependencies.
4. Ignore the newer incomplete Round 1 branch because it has no compatible
   masks milestone.
5. Read `BaseEvidenceState.colmap_ref` and restore that exact owned COLMAP
   checkpoint into a fresh local directory.

No input file or Drive checkpoint is mutated. A missing, corrupt, incompatible,
or lineage-mismatched Round 0 milestone stops before paid inference rather than
silently recomputing it in output-first mode.

## One-pass Drive restore

Current recovery can read large Drive artifacts repeatedly while validating,
copying, and re-inventorying them. Output-first recovery changes the restore
path to one verified remote read per payload file:

1. Validate cache ownership, checkpoint kind, fingerprint, schema, upstream
   references, manifest digest, and `_SUCCESS.json` using small metadata files.
2. Copy every manifest entry to a fresh local destination while computing its
   SHA-256 digest and byte count in the same stream.
3. Compare the observed digest and size with the immutable manifest before the
   file is accepted.
4. Validate the completed local inventory without rereading the Drive payload.
5. Delete only the incomplete local destination on mismatch and leave Drive
   untouched.

The existing strict restore contract remains: a corrupt payload is never
rehydrated. The optimization removes redundant remote hashing; it does not
trust unchecked model, mask, depth, flow, or image bytes.

## Guarded geometry acceptance

The restored checkpoint is measured with the existing `measure_models` and
`evaluate_reconstruction` functions against the exact restored selection. The
dominant model may be accepted as `best_effort` only when all of these
`output-first-v1` checks pass:

- `registered_ratio >= 0.80`;
- `registered_share >= 0.95`;
- `max_interior_gap_s <= 10.0`;
- `start_gap_s <= 1.0` and `end_gap_s <= 1.0`;
- `median_reprojection_error_px <= 1.5`;
- `p95_reprojection_error_px <= 2.5`;
- `median_track_length >= 3.0`;
- `sparse_point_count >= 10_000`; and
- names, intrinsics, and poses are valid and finite.

Only the strict failures `registered_ratio`, `interior_gap`, and
`median_reprojection` may be waived. A failure of dominant-component integrity,
endpoint coverage, p95 reprojection, track length, model validity, finite-value
validation, manifest joins, file hashes, or the minimum point count is a hard
stop.

Among multiple restored sparse models, models that pass the guarded checks are
ranked by endpoint coverage, smaller interior gap, larger registered count,
lower median and p95 reprojection, longer tracks, and point count. The selected
model is copied into a fresh validated geometry directory. No COLMAP mapper,
feature extractor, matcher, DA3 anchor/metric-sky pass, segmentation model, or
optical-flow model is run before geometry acceptance. The existing final-pose
depth stage remains required after acceptance.

## Acceptance contract and training validation

An experiment-only `GeometryAcceptance` record contains:

- policy version and mode (`strict` or `best_effort`);
- selection image-set digest;
- hashes of `cameras.txt`, `images.txt`, and `points3D.txt`;
- the original strict decision and failures;
- the complete measured metrics;
- guarded check results; and
- the chosen checkpoint fingerprint.

The shared validated-training entrypoint receives an optional reconstruction
validator callback. Its default callback preserves the current behavior: both
the stored decision and a fresh measurement must pass the strict gate. The
learned experiment supplies a guarded callback only when the run spec selects
`round0_output_first_v1`. That callback rehashes and remeasures the copied model,
requires an exact acceptance-record join, and reruns every guarded check before
training. It does not synthesize a passing `GateDecision`.

This default-off dependency-injection seam is the only shared production-code
behavior change. Regression tests must prove that omitted callbacks retain the
current strict rejection behavior.

## Remaining execution flow

After geometry acceptance, the existing learned pipeline runs:

1. final static-track qualification against the accepted poses;
2. photometric validation;
3. final-pose metric-depth alignment;
4. dense seed fusion;
5. publication of geometry and final-pretraining milestones;
6. maximum-quality Gaussian training with live iteration output;
7. portable `splat.ply`, diagnostics, quality report, experiment report, and
   `_SUCCESS` publication; and
8. Drive flush/unmount followed by Colab runtime release.

If a failure occurs after a newly completed milestone, the milestone remains
reusable. The notebook releases the runtime after success or failure. The input,
legacy `<input_folder>_result`, and Round 1 checkpoints are never modified.

## Reporting

The notebook prints a prominent banner before Gaussian training:

```text
[GEOMETRY] BEST-EFFORT ROUND 0 ACCEPTED
Strict failures: registered_ratio, interior_gap, median_reprojection
Registered: <count>/<selected> | max gap: <seconds> | median reproj: <pixels>
```

The final `quality_report.json`, `experiment_report.json`, training status, and
success marker include `geometry_acceptance_mode`, policy version, strict
failures, guarded metrics, and the COLMAP checkpoint fingerprint. Contact sheets
and `splat.ply` remain required publication artifacts. The report must never
describe a best-effort model as strict-pass geometry.

## Failure behavior

Output-first mode stops without recomputation when:

- the complete Round 0 masks lineage cannot be restored;
- the referenced COLMAP checkpoint is missing or corrupt;
- no sparse model passes the guarded checks;
- acceptance metadata does not join the restored selection and model exactly;
- final evidence fails its existing typed contracts;
- Gaussian training fails; or
- final Drive publication cannot be verified.

Every failure receipt names the failed stage, newest durable milestone, strict
and guarded geometry failures when available, and the next reusable stage.
Automatic Drive flush and runtime release remain unconditional.

## Testing and verification

Automated tests must demonstrate:

- output-first mode selects the complete Round 0 masks lineage instead of the
  newer incomplete Round 1 semantic lineage;
- restored COLMAP is measured without invoking classical or hybrid runners;
- the known Round 0 metric shape passes the guarded policy and remains a strict
  failure;
- each guarded threshold boundary is inclusive and a value immediately outside
  it is rejected;
- an unapproved strict failure cannot be waived;
- acceptance is bound to exact selection and model hashes;
- the guarded training validator remeasures the model and detects any mutation;
- the default training validator remains strict and unchanged;
- one-pass restore detects size, digest, missing-file, unexpected-file, marker,
  manifest, and upstream corruption while reading every remote payload once;
- geometry and final-pretraining milestones resume after interruption;
- reports and `_SUCCESS` label best-effort geometry explicitly;
- the notebook is pinned to the verified commit, displays live subprocess
  output, flushes Drive, and releases the runtime; and
- legacy notebook generation and production static-pipeline tests remain
  unchanged.

Verification includes the focused learned-quality tests, cache corruption and
restore tests, backend strict-training tests, notebook smoke tests, formatting,
and a clean generated-notebook diff inspection. The A100 notebook is not
published until all locally runnable verification passes.

## Success criteria

The recovery is ready to run when one pinned A100 notebook can restore the
existing complete Round 0 evidence graph, accept only the guarded cached model,
begin Gaussian training without rerunning earlier learned/COLMAP stages, publish
all required artifacts to `myroom_test_learned_test_result`, and release the
runtime. Round 1 remains fully available for a later strict comparison run.

## Out of scope

- Completing or scoring Round 1.
- Recreating the lost Round 0 learned-hybrid candidate.
- Changing strict production reconstruction thresholds.
- Adding a frontend/API/notebook-generator option.
- Modifying the legacy notebook or result folder.
- Adding a web viewer.
- Claiming Luma-equivalent or strict-pass quality from a best-effort result.

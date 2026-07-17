# Learned Evidence Resume and Track Qualification Design

**Status:** Proposed

**Date:** 2026-07-18

## Context

The learned-quality A100 experiment currently has two durable recovery
boundaries: verified COLMAP and complete pre-training. A real 801-frame run
restored COLMAP successfully, completed semantic inference and the expensive
bidirectional optical-flow calculation, then failed when mask fusion rejected a
COLMAP track observation whose coordinates were non-finite or outside the
canonical image dimensions. Because complete pre-training had not yet been
published, runtime release discarded the successfully computed learned
evidence.

This exposes two independent defects:

1. The COLMAP static-track producer does not enforce the contract required by
   its consumers. It accepts raw observations without checking finite image
   bounds and retains tracks with only two observations, while downstream
   photometric processing requires at least three producer-qualified
   observations.
2. The recovery architecture loses completed learned stages when a later
   pre-training stage fails.

This design supersedes the decision in
`2026-07-17-learned-pretraining-cache-stage-output-design.md` that learned
substages remain uncached. The existing result path, legacy notebook, A100
quality policies, pinned models, and automatic runtime release remain
unchanged.

## Decision

The learned-quality experiment will qualify static tracks at their producer
boundary, run a CPU-safe audit before learned inference, and publish immutable,
typed milestone checkpoints for expensive learned evidence. A rerun restores
the newest compatible checkpoint dependency graph and resumes at the first
missing stage.

No coordinate is clamped and no lossy artifact representation is introduced.
Invalid evidence is discarded with an audit trail; incompatible or incomplete
checkpoints are ignored safely.

## Options considered

### Track patch with the existing two checkpoints

This is the smallest change, but a different failure between COLMAP and final
pre-training would still discard semantic, flow, geometry, or depth work. It
does not meet the paid-runtime recovery requirement.

### Immutable milestone checkpoints

Each expensive stage owns a portable checkpoint containing only its outputs and
fingerprinted references to its upstream checkpoints. This avoids duplicating
the whole run at every boundary while permitting deterministic graph restore.
It adds several typed serialization contracts but does not change quality
algorithms. This is the selected approach.

### Per-frame and per-pair journal

This could resume in the middle of semantic inference or optical flow and could
also reduce peak memory through streaming. It requires a substantial rewrite of
model batching, pair evaluation, aggregation, and publication. It is deferred;
milestone recovery protects completed expensive stages with substantially less
risk.

## Static-track qualification

### Producer contract

The runtime will build a canonical dimension map from the selected frame files
before reading COLMAP tracks. A track observation is accepted only when:

- its image joins exactly one selected canonical frame;
- its frame identifier is unique within the track;
- `x` and `y` are finite real values;
- `0 <= x < width` and `0 <= y < height`; and
- the referenced COLMAP point identifier matches the track identifier.

An invalid observation is discarded, never clamped. Clamping would invent
geometric support at an image boundary and could incorrectly protect sky or
dynamic pixels.

A static track is published only when:

- its identifier is a unique nonnegative integer;
- its XYZ coordinates and mean reprojection error are finite;
- its reprojection error is nonnegative; and
- at least three valid, unique observations remain.

This single producer contract satisfies optical-flow calibration, mask fusion,
sky support, and photometric validation. Consumers retain defensive validation
but must not discover ordinary raw-COLMAP qualification failures after learned
inference.

### Audit and systematic-mismatch guard

Qualification produces a deterministic `track_audit.json` containing:

- raw, accepted, and rejected track counts;
- raw, accepted, and rejected observation counts;
- rejection counts by reason;
- affected image names and canonical dimensions;
- bounded examples containing track id, image name, coordinate, and reason; and
- a digest of the accepted track payload.

The audit fails before semantic or optical-flow inference when no qualified
tracks remain or when more than one percent of otherwise joined observations
are non-finite or outside canonical bounds. A small number of isolated COLMAP
outliers is filtered safely; a widespread mismatch is treated as a likely
orientation, scaling, or stale-cache defect and is not hidden.

The threshold and audit schema are versioned preprocessing policy. Changing
either invalidates dependent checkpoints.

## Checkpoint architecture

### Milestones

The Drive cache gains these typed checkpoint kinds:

1. `selection`: selected image files, inventory, selection manifest, and source
   manifest.
2. `base_evidence`: DA3 anchors, verified COLMAP reference, metric depth and
   sky, canonical rigid scene, and track audit.
3. `semantic`: semantic masks and their manifest.
4. `motion`: bidirectional optical-flow evidence and its manifest.
5. `masks`: fused quality masks and training-validity masks.
6. `geometry`: accepted classical or hybrid model, candidate reports, and any
   backfilled selection/evidence graph.
7. `pretraining`: photometric decision, final pose-conditioned depth, validated
   dense seeds, and the complete prepared-training contract.

The existing `colmap` checkpoint remains independently reusable. The existing
complete-pretraining restore behavior becomes the `pretraining` milestone and
continues to skip directly to Gaussian training.

Each milestone stores only artifacts first produced at that boundary. Its
manifest names the exact upstream checkpoint fingerprints it consumes. These
references form a dependency graph rather than forcing an artificial linear
chain: `semantic` and `motion` both depend on `base_evidence`, while `masks`
depends on `base_evidence`, `semantic`, and `motion`. Restore walks the graph
from the newest desired milestone to `selection`, validates every edge, copies
artifacts into run-local `/content`, and rehydrates typed service outputs with
only local absolute paths. A semantic-only code change therefore does not
invalidate reusable motion evidence.

### Fingerprints

Every checkpoint fingerprint binds:

- input inventory digest;
- selected-frame image-set digest;
- checkpoint schema and relevant policy values;
- the sorted map of exact upstream checkpoint fingerprints;
- pinned learned-model manifest and resolved revisions when applicable;
- relevant tool versions;
- deterministic producer-code digest for that milestone; and
- manifests and SHA-256 digests of all direct input artifacts.

A notebook-output, training-only, or later-stage fix does not invalidate an
unaffected earlier milestone. A change to track qualification invalidates
`base_evidence` and every downstream milestone while preserving the verified
COLMAP checkpoint.

### Drive publication

Publication follows the generation protocol already proven compatible with the
Colab Drive mount:

- write to a unique owned generation directory;
- write and fsync every artifact and manifest;
- reread and validate the complete inventory;
- write `_SUCCESS.json` last; and
- consider only generations with a valid success marker reusable.

The design does not depend on atomic directory rename, which Google DriveFS may
reject. A failed generation cannot replace or invalidate an older valid
generation. Cleanup removes only pipeline-owned stale or superseded
generations after a newer generation has been validated.

The cache layout is:

```text
<data_folder_name>_learned_test_cache/
  _OWNERSHIP.json
  colmap/<fingerprint>/
  selection/<fingerprint>/<generation>/
  base_evidence/<fingerprint>/<generation>/
  semantic/<fingerprint>/<generation>/
  motion/<fingerprint>/<generation>/
  masks/<fingerprint>/<generation>/
  geometry/<fingerprint>/<generation>/
  pretraining/<fingerprint>/<generation>/
```

Stage artifacts remain lossless. For the current full-HD dataset, the expected
additional Drive footprint is approximately 15--30 GB. The notebook prints the
actual published byte count for every milestone.

## Resume flow

1. Inventory the Drive input and validate cache ownership.
2. Probe `pretraining`; if valid, restore it and proceed to training.
3. Otherwise probe milestones newest-first and validate the entire referenced
   dependency graph.
4. Restore the newest valid graph to a fresh run-local root.
5. Run the first missing stage only.
6. Validate and publish its milestone before starting the next expensive stage.
7. Repeat through `pretraining`, then begin Gaussian training.

A corrupt newest milestone is reported and ignored without deleting older valid
milestones. A valid upstream milestone remains reusable when a downstream
milestone is corrupt or incompatible.

If geometry comparison requests selection backfill, the new selection digest
creates a new checkpoint branch. The original branch remains valid and is not
mutated. Backfill evidence is checkpointed under the new branch before a second
geometry round.

## Progress and failure behavior

Optical flow emits flushed progress events after each completed pair batch:

```text
[OPTICAL FLOW] 590/800 pairs - 73.8% - 31:42 elapsed
```

The event includes completed and total pairs, elapsed time, current pair-batch
size, process RSS, and CUDA memory. CPU residual evaluation emits equivalent
pair progress, and artifact publication emits completed and total frame counts.
No estimated completion time is claimed until at least two measured batches are
available.

Every failure receipt and diagnostic log records:

- the failed stage and substage;
- the newest durable milestone;
- the exact milestone expected on the next run;
- track-audit summary when available; and
- local artifacts that were not yet durably published.

Runtime flush and release behavior remains unchanged. The pipeline never starts
the next expensive stage until the preceding milestone has a verified Drive
success marker.

## CPU-only audit gate

A CPU-safe command and notebook cell validate the input identity, COLMAP cache,
camera dimensions, track qualification, checkpoint ownership, checkpoint
manifests, and model manifest without loading learned models or requiring CUDA.
It produces a small audit receipt on Drive.

The A100 notebook requires a passing receipt whose input digest, COLMAP
fingerprint, qualification-policy version, and audit producer-code digest match
the pending run. A missing or stale receipt causes an immediate, explicit stop
before learned inference. Training-only and notebook-output changes do not
invalidate the receipt. The user can therefore validate the known cached
dataset in a free CPU session before spending A100 credits.

## Testing

Automated tests must cover:

- COLMAP observations containing NaN, infinity, negative coordinates, exact
  right/bottom boundaries, and values just inside every boundary;
- invalid-observation filtering without coordinate clamping;
- track removal below three surviving unique observations;
- finite XYZ and reprojection-error qualification;
- deterministic audit counts, examples, and digest;
- early systematic-mismatch failure before semantic/model factories are called;
- save, validation, restore, and typed rehydration for every milestone;
- upstream reuse after downstream corruption or incompatibility;
- producer-code invalidation scoped to the affected milestone and descendants;
- DriveFS-compatible generation publication without directory rename;
- restart after failures following semantic, motion, masks, geometry, and final
  depth;
- optical-flow inference, residual-evaluation, and publication progress events;
- CPU-only audit receipt creation, staleness, and A100 enforcement;
- unchanged legacy pipeline and learned result-folder contracts; and
- generated-notebook smoke coverage for resume messages, live output, Drive
  flush, and runtime release.

Before publishing a new notebook, the cached room dataset must pass the CPU-only
audit. No A100 rerun is recommended until unit, integration, cache-corruption,
and notebook smoke suites pass and the notebook is pinned to the verified
runtime commit.

## Consequences

- Late preprocessing failures no longer discard all prior learned evidence.
- Raw COLMAP edge cases are handled once at the producer boundary and reported
  before expensive inference.
- Drive storage and publication time increase, but successful milestones become
  reusable and exact.
- Cache serialization code and migration tests become more complex.
- Mid-batch semantic or optical-flow recovery remains out of scope; the current
  stage restarts from its beginning if it fails before publishing its milestone.
- The legacy notebook, Gaussian training algorithm, maximum-quality settings,
  and final result path do not change.

## Out of scope

- Resuming from a partial Gaussian-training iteration.
- Lossy flow, depth, mask, or image compression.
- Clamping invalid COLMAP observations.
- Changing learned-model checkpoints or quality thresholds unrelated to track
  qualification.
- Changing the legacy notebook pipeline or adding a web viewer.

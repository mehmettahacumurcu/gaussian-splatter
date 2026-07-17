# Learned Pre-Training Cache and Stage Output Design

## Purpose

The learned-quality A100 notebook must make long-running work visible and avoid
repeating verified preprocessing after a later failure. A user who reruns the
same input with compatible preprocessing must see which work is restored from
Google Drive, which work is running, and when Gaussian training begins.

This change applies only to the learned-quality experiment notebook. It does not
change the legacy pipeline, quality algorithms, result folder contract, or
automatic Drive flush and runtime release behavior.

## User-visible contracts

For an input folder named `<data_folder_name>`, the notebook continues to publish
the successful result to:

```text
<data_folder_name>_learned_test_result/
```

It stores pipeline-owned reusable preprocessing beside the input at:

```text
<data_folder_name>_learned_test_cache/
```

The existing diagnostics location remains unchanged. A cache folder is not a
result and must never satisfy result publication checks.

The final execution cell prints live, flushed stage events in this form:

```text
[STAGE 05/15] START COLMAP mapping
[HEARTBEAT] COLMAP mapping running - 08:00 elapsed
[STAGE 05/15] DONE  COLMAP mapping - 398/412 registered - 12:41
[CACHE SAVE] Verified COLMAP checkpoint saved to Drive
[CACHE HIT] Complete pre-training checkpoint restored from Drive
[STAGE 14/15] START Gaussian training
```

Stage numbers may change when stages are added, but each event includes a stable
machine-readable stage identifier in the structured run log. Human-readable
notebook output includes the label, status, elapsed time, and useful summary.
Existing trainer iteration output remains visible and unbuffered.

COLMAP does not expose a reliable total percentage for every command. The
notebook therefore streams available COLMAP output and prints a periodic elapsed
time heartbeat while a quiet COLMAP subprocess remains alive. It must not invent
a percentage.

## Architecture

### Stage reporter

A small reporter owns all stage lifecycle output. It supports:

- run start and run identifier;
- stage start, completion, failure, and elapsed duration;
- cache hit, miss, invalidation, restore, and save events;
- periodic heartbeats for quiet long-running subprocesses; and
- a structured JSON-lines run log used in diagnostics and reports.

Messages are flushed immediately. Exceptions pass through unchanged after a
failure event is recorded, preserving the current diagnostic behavior. The
static runner reports top-level boundaries while learned reconstruction reports
its internal boundaries through the same interface.

The visible stage set includes input discovery, preflight, source copy or cache
restore, frame selection, DA3 anchors, classical COLMAP prepass, metric depth and
sky, semantic masks, optical flow, mask fusion, geometry comparison,
photometric validation, final-pose depth, dense-seed fusion, cache publication,
Gaussian training, polish, report assembly, validation, and Drive publication.

### Cache manager

A learned pre-training cache manager is responsible for cache fingerprints,
validation, atomic publication, restoration, and ownership checks. The runner
does not manipulate Drive cache directories directly.

There are two reusable checkpoint boundaries:

1. **Verified COLMAP checkpoint.** Published immediately after the classical
   prepass model and its registration gate pass. A failure in a later learned
   stage can then reuse COLMAP.
2. **Complete pre-training checkpoint.** Published after final geometry,
   photometric transforms, validated depth, dense seeds, masks, selected frames,
   and all training-input invariants pass. A failure in training or a later stage
   can then skip all preprocessing.

Learned substages between those boundaries are visible but are not independently
cached. This keeps recovery useful without creating many fragile serialization
contracts.

### Portable checkpoint contract

Each checkpoint contains only paths relative to its checkpoint root, a manifest,
required artifacts, per-file sizes and SHA-256 hashes, and a success marker
written last. It must contain enough metadata to reconstruct the existing
selection and learned-reconstruction service outputs without relying on Python
pickle or absolute `/content` paths.

Restoration first validates the manifest and all required files, then copies the
checkpoint to local `/content` storage. Training and reconstruction never operate
directly on the Drive mount. Restored objects reference only the new local paths.

## Fingerprints and invalidation

Cache compatibility is determined at the checkpoint boundary. Fingerprints
include:

- input inventory digest, including source relative paths, sizes, and hashes;
- frame-selection settings and selected-frame manifest digest;
- quality settings that affect preprocessing;
- preprocessing policy and cache schema versions;
- pinned learned-model manifest and resolved model revisions;
- relevant tool versions, including COLMAP; and
- a deterministic digest of preprocessing producer code used by that checkpoint.

The full repository commit is recorded for audit but is not itself an
invalidation key. This permits a training-only or notebook-output fix to reuse
verified preprocessing. A change to preprocessing code changes the producer-code
digest and invalidates the affected checkpoint.

The complete pre-training checkpoint depends on the verified COLMAP fingerprint
and all later learned-stage inputs. An invalid COLMAP checkpoint necessarily
invalidates the complete checkpoint.

On a cache miss or invalidation, the notebook prints the reason and recomputes the
stage. It never silently accepts an older or merely similar cache.

## Drive publication and ownership

Checkpoint data is assembled in a sibling staging directory. Files and hashes
are validated before a success marker is written last, then the complete staging
directory is promoted into the pipeline-owned cache location. A partial staging
directory or missing success marker is never reusable.

The cache root has an ownership marker containing the generator identifier,
schema version, and input-folder identity. Existing data may be replaced only
when this marker proves that the pipeline owns the target. If an unowned folder
already occupies the cache path, the run fails safely instead of deleting or
overwriting it.

The owned root separates the checkpoint types and their staging data:

```text
<data_folder_name>_learned_test_cache/
  _OWNERSHIP.json
  colmap/<fingerprint>/
  pretraining/<fingerprint>/
  staging/<checkpoint-type>-<run-id>/
```

Each checkpoint is promoted independently. A failed complete-pretraining cache
write therefore cannot damage a valid COLMAP checkpoint. Once a new checkpoint
has been promoted successfully, older owned generations of that checkpoint type
are removed to avoid unbounded Drive growth. Failed runs may leave an owned
staging directory; future runs may remove only stale staging directories that
carry the same ownership identity.

Checkpoint publication completes before Gaussian training starts. The success
marker is written last and filesystem writes are synchronized so that an
ordinary training failure followed by runtime release leaves a reusable cache.
The pipeline cannot guarantee persistence if Colab or Drive suffers an abrupt
external disconnect during the cache write itself.

## Data flow

1. Resolve and inventory the Drive input.
2. Inspect hardware and run preflight checks.
3. Compute compatible checkpoint fingerprints.
4. Prefer a valid complete pre-training checkpoint:
   - restore it to the run-local work root;
   - reconstruct the service output contracts; and
   - proceed directly to training.
5. Otherwise restore a valid COLMAP checkpoint when available.
6. Run and report any missing preprocessing stages.
7. Validate and publish the COLMAP checkpoint at its boundary if newly produced.
8. Validate and publish the complete pre-training checkpoint.
9. Run Gaussian training with live iteration output.
10. Polish, validate, and publish the normal result.
11. Flush Drive and release the runtime on either success or failure, as today.

## Failure behavior

- A failed stage prints a `FAIL` event with elapsed time before diagnostics are
  published.
- Corrupt, incomplete, unowned, or incompatible cache data is never restored.
- A cache validation failure is normally a visible cache miss followed by safe
  recomputation. Unsafe ownership conflicts and local restore corruption are hard
  failures with diagnostics.
- Cache publication failure stops the run before training. This guarantees the
  notebook does not spend A100 training time after promising a reusable
  checkpoint that was not actually saved.
- Trainer failures retain the complete pre-training checkpoint and continue to
  use the existing diagnostic, Drive flush, and runtime release path.
- Cache reuse does not weaken any existing reconstruction or learned-evidence
  quality gate.

## Testing

Automated tests cover:

- ordered, flushed start/done/fail stage events and elapsed durations;
- heartbeat output for a quiet simulated subprocess;
- continued visibility of trainer iteration output;
- COLMAP checkpoint save, validation, restore, and object rehydration;
- complete pre-training checkpoint save, validation, restore, and object
  rehydration;
- cache hits that skip the corresponding mocked services;
- input, setting, model, tool, producer-code, and schema invalidation;
- training-only code changes that do not invalidate preprocessing;
- corrupted files, missing markers, path traversal, symlinks, and ownership
  conflicts;
- atomic publication and rejection of partial staging data;
- failure before COLMAP, between checkpoints, during cache publication, and
  during training;
- unchanged result and diagnostics folder contracts; and
- generated-notebook smoke checks for unbuffered execution, stage visibility,
  cache location, Drive flush, and runtime release.

The existing learned-quality reconstruction, duplicate-track, COLMAP probe,
notebook generation, and static-pipeline suites remain regression gates.

## Out of scope

- Changing the legacy notebook pipeline.
- Altering selection, reconstruction, masking, depth, geometry, training, or
  polish quality algorithms.
- Resuming from a partially completed Gaussian-training iteration.
- Caching each learned evidence substage separately.
- Providing a web viewer or changing the final result folder.

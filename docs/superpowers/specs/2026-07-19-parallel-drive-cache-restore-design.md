# Parallel Drive Cache Restore Design

## Objective

Reduce the A100 time spent restoring existing learned-quality milestones from
Google Drive while preserving the current checkpoint format, source immutability,
and per-file SHA-256 verification.

Success means that cache generations containing many independent files can be
read concurrently, a one-worker run remains behaviorally compatible with the
existing sequential implementation, and no caller observes or retains a partial
restore after an error or interruption.

## Chosen approach

Use a bounded `ThreadPoolExecutor` inside `_copy_verified_payload`. The production
default is four workers. The internal keyword-only `max_workers` parameter accepts
plain integers from 1 through 16 and is clamped to the manifest file count. This
keeps the production request fan-out conservative for Google Drive while allowing
deterministic tests and later evidence-based tuning.

Two alternatives are intentionally deferred:

- Keeping the sequential implementation is safest but repeats the measured
  bottleneck and does not meet the latency goal.
- Retrofitting tar or zip archives could improve small-file throughput further,
  but would reread and mutate existing Drive caches, require a new secure
  extraction format, consume substantial temporary space, and delay the current
  recovery run.

## Architecture

`experiments/learned_quality/cache.py` remains the sole owner of restore
semantics. No manifest schema, checkpoint fingerprint, Drive directory, notebook
input, or public API changes.

The implementation is split into small private units:

1. `_copy_verified_payload(..., *, max_workers=4)` validates the worker bound,
   parses the signed manifest, prepares destination directories, schedules files,
   aggregates progress, and performs the final metadata inventory check.
2. `_copy_verified_file(...)` owns one file from source open through destination
   close. It retains the existing 8 MiB block size, exclusive `"xb"` output,
   source-size checks, copied-byte checks, and SHA-256 comparison.
3. A lock-protected progress object records verified bytes and emits complete,
   monotonic messages at every 512 MiB threshold.
4. A cooperative cancellation event lets running workers stop between blocks
   after the first failure. Pending work is cancelled, all workers are joined,
   and the original exception is re-raised.

Only the bounded number of active files is scheduled at once. This prevents an
800-file manifest from creating an unbounded queue and ensures cancellation can
stop before opening most Drive files.

## Data and integrity flow

The existing verified manifest remains authoritative:

1. Validate every relative path, size, and SHA-256 row before creating the
   destination.
2. Resolve and validate source files beneath the immutable Drive payload root.
3. Pre-create destination directories serially and reject file/directory prefix
   conflicts before Drive content reads begin.
4. Copy up to four files concurrently. Each worker opens its source once, hashes
   the bytes while writing locally, and verifies both byte count and SHA-256.
5. After all workers succeed, compare the restored tree's paths and sizes with
   the manifest.
6. Print completion only after all hashes and the final inventory pass.

The source cache is read-only throughout. Parallelism does not change the bytes,
the manifest, or cache lineage.

## Failure and interruption semantics

The first worker failure is authoritative. The coordinator signals cancellation,
cancels pending futures, waits for every running worker to stop, and then
re-raises that original exception. It must not replace it with `CancelledError`.

Existing caller behavior remains intact:

- Milestone corruption or ordinary I/O becomes a cleaned cache miss.
- COLMAP corruption remains a cleaned hard failure.
- Pretraining corruption remains a cleaned hard failure after candidate
  selection.

`restore_milestone` will also clean its destination for `BaseException` values
such as `KeyboardInterrupt`, then re-raise them. This closes the only discovered
cleanup gap and prevents an interrupted parallel restore from leaving a local
tree that blocks the next attempt.

No automatic checksum retry is allowed. A size or hash mismatch is corruption,
not throttling, and must remain visible. A one-worker mode remains available
internally if future Colab evidence shows Drive request throttling.

## Progress reporting

The start message includes file count, total GiB, and active worker count.
Workers update one shared byte counter under a lock. Threshold messages remain at
512 MiB, are monotonic, and cannot interleave. The final message is emitted once
and only after verification completes.

## Test strategy

All changes follow a red-green TDD cycle in
`tests/experiments/learned_quality/test_cache.py`:

- A barrier-backed test proves four file workers overlap.
- Same-size source mutations prove every manifest SHA-256 is checked.
- A sentinel worker failure proves the original exception survives, workers are
  joined, and the caller removes the destination.
- Progress tests prove serialized, monotonic thresholds and one final message.
- Executor spies prove the four-worker default, file-count clamping, and the
  16-worker hard bound.
- Invalid values, including booleans and floats, fail before destination creation.
- One-worker mode proves one source open per file, including a zero-byte file,
  and preserves final inventory semantics.
- An interruption test proves `restore_milestone` cleans the local destination
  before re-raising `KeyboardInterrupt`.

After targeted tests pass, run the entire learned-quality suite and both notebook
smoke tests. Repin the A100 notebook only after the implementation commit is
pushed and the full verification is green.

## Out of scope

- Changing cache schemas or fingerprints.
- Repacking existing Drive data.
- Uploading archives or modifying source cache generations.
- Parallelizing cache publication.
- Claiming a fixed speedup factor; Drive throughput is externally variable.

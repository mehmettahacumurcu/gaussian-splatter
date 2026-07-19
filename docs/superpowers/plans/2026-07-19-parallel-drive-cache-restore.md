# Parallel Drive Cache Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore existing learned-quality Drive milestones with four concurrent checksum-verifying file workers while preserving cache compatibility, cleanup, and source immutability.

**Architecture:** Refactor the existing single-pass copy loop into one per-file verifier plus a bounded coordinator. The coordinator validates all inputs before destination creation, keeps no more than the active worker count in flight, serializes verified-byte progress, joins workers before propagating the first failure, and leaves caller-specific cleanup semantics intact.

**Tech Stack:** Python 3.12 standard library (`concurrent.futures`, `hashlib`, `pathlib`, `threading`), pytest 9, nbformat, Google Drive FUSE through the existing filesystem interface.

## Global Constraints

- Production restore concurrency defaults to exactly 4 files.
- `max_workers` must be a plain `int` from 1 through 16; booleans, floats, strings, `None`, zero, negatives, and values above 16 are rejected before destination creation.
- Active workers equal `min(max_workers, manifest_file_count)`; an empty manifest creates no executor.
- Every source file is opened exactly once, streamed in 8 MiB blocks, byte-count checked, and SHA-256 checked against the existing signed manifest.
- Source cache generations are read-only; no archive, repack, manifest, schema, fingerprint, or Drive-layout change is allowed.
- No more than the active worker count of futures may be submitted at once.
- The first worker failure is re-raised unchanged only after pending work is cancelled and running workers have quiesced.
- Progress is lock-serialized, monotonic, reported per 512 MiB of fully verified files, and completed exactly once after final inventory validation.
- Existing milestone soft-miss, COLMAP hard-failure, and pretraining hard-failure semantics remain unchanged.
- `restore_milestone` removes a partial local destination and re-raises non-recoverable `BaseException` values, including `KeyboardInterrupt`.
- Use `C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe` for local tests because the default Anaconda interpreter lacks PyTorch.
- Work only on `feature/learned-quality-a100`; do not push `main`, force-push, create tags, or modify secrets.

---

### Task 1: Bounded parallel verified restore

**Files:**
- Modify: `experiments/learned_quality/cache.py:1-25`
- Modify: `experiments/learned_quality/cache.py:655-723`
- Modify: `experiments/learned_quality/cache.py:1782-1956`
- Modify: `tests/experiments/learned_quality/test_cache.py`

**Interfaces:**
- Consumes: the existing `_ManifestEntry`, `_manifest_file_rows`, `_regular_directory`, `_regular_file`, `_payload_metadata_inventory`, and caller cleanup contracts.
- Produces: `_copy_verified_payload(generation, destination, manifest, *, max_workers=4) -> None`; `_copy_verified_file(source_root, target, row, cancellation, progress) -> int`; private worker validation and progress helpers.

- [ ] **Step 1: Add deterministic parallel-copy fixtures and failing tests**

Add imports for `threading`, `time` only if required by an existing fixture, and the module alias already used by cache tests. Add this helper shape, using the existing `LearnedCheckpointStore.publish_generation` signature from the file rather than bypassing production publication:

```python
def _published_parallel_payload(
    tmp_path: Path,
    *,
    file_count: int,
    file_size: int = 8,
) -> tuple[Path, dict[str, object], dict[str, bytes]]:
    source = tmp_path / "parallel-source"
    source.mkdir()
    expected: dict[str, bytes] = {}
    for index in range(file_count):
        relative = Path(f"group-{index % 2}") / f"file-{index:02d}.bin"
        payload = bytes([index + 1]) * file_size
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        expected[relative.as_posix()] = payload
    store = LearnedCheckpointStore(tmp_path / "parallel-cache", input_identity="a" * 64)
    generation = store.publish_generation(
        CheckpointKind.COLMAP,
        fingerprint="b" * 64,
        run_id="parallel-test",
        source_root=source,
    )
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    return generation, manifest, expected
```

Write these tests with the exact contracts below:

```python
def test_copy_verified_payload_runs_file_workers_concurrently(tmp_path, monkeypatch):
    # Four wrapped workers must meet at Barrier(4, timeout=3).
    # Record four distinct thread identifiers, then call the real worker.

@pytest.mark.parametrize("corrupt_index", range(4))
def test_copy_verified_payload_verifies_sha256_for_every_manifest_row(...):
    # Replace one Drive payload with different bytes of identical length.
    # Expect ValueError containing both "hash mismatch" and the relative path.

@pytest.mark.parametrize(
    ("file_count", "configured", "expected"),
    ((12, 4, 4), (3, 16, 3), (1, 8, 1)),
)
def test_copy_verified_payload_clamps_workers_to_file_count(...):
    # Spy on the module ThreadPoolExecutor factory and assert the exact count.

@pytest.mark.parametrize("value", (True, False, None, 0, -1, 17, 1.0, "4"))
def test_copy_verified_payload_rejects_invalid_worker_bounds_before_destination(...):
    # Expect "max_workers must be a plain integer from 1 through 16".
    # Assert the destination was never created and no executor was constructed.

def test_copy_verified_payload_default_worker_count_is_four(...):
    # Omit max_workers, publish at least eight files, and assert executor size 4.

def test_copy_verified_payload_one_worker_preserves_single_pass_semantics(...):
    # Include nested and zero-byte files. Assert exact bytes and one source open.
    # Make _sha256 fail if invoked for a Drive payload to prove no prehash was added.
```

- [ ] **Step 2: Run the new tests and capture the red state**

Run:

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest `
  tests/experiments/learned_quality/test_cache.py `
  -k 'copy_verified_payload' -q
```

Expected: failures because `_copy_verified_payload` has no `max_workers` keyword and `_copy_verified_file` does not exist.

- [ ] **Step 3: Implement worker validation, path preparation, per-file verification, and progress**

Add these constants near the cache module constants:

```python
_RESTORE_DEFAULT_WORKERS = 4
_RESTORE_MAX_WORKERS = 16
_RESTORE_BLOCK_BYTES = 8 * 1024 * 1024
_RESTORE_PROGRESS_STEP_BYTES = 512 * 1024 * 1024
```

Add standard-library imports:

```python
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from threading import Event, Lock
```

Implement a validator that uses `type(max_workers) is int`, rejects values outside 1 through 16 with the exact test message, and returns zero for zero manifest rows or `min(max_workers, file_count)` otherwise.

Implement a lock-protected progress class with this public behavior:

```python
class _RestoreProgress:
    def __init__(self, total_bytes: int) -> None: ...
    def add_verified(self, byte_count: int) -> None: ...
```

`add_verified` increments only after a complete file passes its digest. Under one lock, emit every crossed 512 MiB threshold using integer byte state so tests can spy without relying on rounded GiB text.

Implement the per-file helper with this control flow:

```python
def _copy_verified_file(source_root, target, row, cancellation, progress):
    payload_relative = PurePosixPath(*row.relative_path.parts[1:])
    source_file = _regular_file(source_root.joinpath(*payload_relative.parts), "restore source")
    source_file.relative_to(source_root)
    if source_file.stat().st_size != row.size_bytes:
        raise ValueError(f"restore source size mismatch: {payload_relative.as_posix()}")
    copied = target.joinpath(*payload_relative.parts)
    digest = hashlib.sha256()
    copied_size = 0
    with source_file.open("rb") as source_stream, copied.open("xb") as target_stream:
        while not cancellation.is_set():
            block = source_stream.read(_RESTORE_BLOCK_BYTES)
            if not block:
                break
            copied_size += len(block)
            digest.update(block)
            target_stream.write(block)
    if cancellation.is_set():
        return copied_size
    if copied_size != row.size_bytes:
        raise ValueError(f"restore source size changed: {payload_relative.as_posix()}")
    if digest.hexdigest() != row.sha256:
        raise ValueError(f"restore source hash mismatch: {payload_relative.as_posix()}")
    progress.add_verified(copied_size)
    return copied_size
```

Before any source file is opened, validate all destination paths and serially create their parents beneath the new target. Allow `mkdir(..., exist_ok=True)` to reject file/directory prefix conflicts before Drive reads.

- [ ] **Step 4: Implement the bounded coordinator and first-error semantics**

Replace the sequential loop with a coordinator that:

```python
active: dict[Future[int], _ManifestEntry] = {}
rows_iter = iter(rows)
first_error: BaseException | None = None
```

Submit at most `active_workers` initial futures. Repeatedly call
`wait(tuple(active), return_when=FIRST_COMPLETED)`. For each completed future,
call `future.result()` inside `except BaseException as error`; preserve only the
first error and set the cancellation event. Submit one replacement row only when
no error has occurred. On error, cancel remaining futures, exit the scheduling
loop, allow the executor to join running work, and then `raise first_error`.
Never raise a cancellation artifact in place of the original worker failure.

After successful joining, compare `_payload_metadata_inventory(target)` to the
manifest metadata and print the single completion message. Include active worker
count in the start message.

- [ ] **Step 5: Add failure, progress, and interruption regression tests**

Add:

```python
def test_colmap_restore_propagates_first_worker_failure_after_quiescence_and_cleans(...):
    # Two wrapped workers meet at a barrier. One raises a pre-created RuntimeError.
    # The other signals completion after cancellation. Assert exception identity,
    # worker completion, and that restore_colmap removed the destination.

def test_copy_verified_payload_progress_is_serialized_monotonic_and_completes_once(...):
    # Monkeypatch _RESTORE_PROGRESS_STEP_BYTES to 8 for four eight-byte files.
    # Spy on integer progress values; assert sorted unique thresholds, one start,
    # and exactly one final line occurring last.

def test_restore_milestone_interrupt_cleans_destination_and_reraises(...):
    # Monkeypatch _copy_verified_payload to create destination then raise one
    # pre-created KeyboardInterrupt. Assert identity and destination absence.
```

Update `restore_milestone` to preserve its existing recoverable exception tuple as
a cleaned `None` return, followed by a second `except BaseException:` branch that
cleans the target and re-raises.

- [ ] **Step 6: Run targeted cache tests and the complete learned-quality suite**

Run:

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest `
  tests/experiments/learned_quality/test_cache.py -q
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest `
  tests/experiments/learned_quality -q
```

Expected: exit 0. Only the three existing opt-in real-GPU tests may be skipped.

- [ ] **Step 7: Review and commit Task 1**

Run `git diff --check`, inspect `git status --short`, stage only the cache implementation and tests, inspect `git diff --staged`, then commit:

```powershell
git commit -m "perf: parallelize verified Drive restores" -m "Large learned-quality milestones contain independent files, so bounded concurrent reads overlap Drive latency while retaining one-pass SHA-256 verification and deterministic cleanup."
git push
```

---

### Task 2: Pin and verify the accelerated A100 notebook

**Files:**
- Modify mechanically: `colab/learned_quality_a100_experiment.ipynb`

**Interfaces:**
- Consumes: the pushed Task 1 implementation commit SHA.
- Produces: a deterministic checked-in notebook whose `COMMIT_SHA` and generator metadata both reference Task 1.

- [ ] **Step 1: Regenerate the notebook from the pushed implementation commit**

Run:

```powershell
$commit = (git rev-parse 'HEAD^{commit}').Trim()
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' `
  -m experiments.learned_quality.notebook `
  --commit $commit `
  --output colab/learned_quality_a100_experiment.ipynb
```

Do not hand-edit notebook JSON. Confirm both occurrences of the full SHA:

```powershell
rg -n "$commit|COMMIT_SHA|commit_sha" colab/learned_quality_a100_experiment.ipynb
```

- [ ] **Step 2: Run notebook and cache integration verification**

Run:

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest `
  tests/integration/test_learned_quality_notebook_smoke.py `
  tests/integration/test_learned_quality_audit_notebook_smoke.py `
  tests/experiments/learned_quality/test_cache.py -q
```

Expected: exit 0 with no failures.

- [ ] **Step 3: Review, commit, and push the notebook pin**

Run `git diff --check`, stage only the notebook, inspect `git diff --staged`, and commit:

```powershell
git commit -m "chore: repin parallel restore notebook" -m "The A100 run must check out the verified concurrent cache reader rather than the earlier sequential implementation."
git push
```

- [ ] **Step 4: Perform final branch verification**

Run:

```powershell
git status --short
git rev-parse HEAD
git rev-parse origin/feature/learned-quality-a100
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest `
  tests/experiments/learned_quality `
  tests/integration/test_learned_quality_notebook_smoke.py `
  tests/integration/test_learned_quality_audit_notebook_smoke.py -q
```

Expected: clean status, identical local/remote SHAs, exit 0, and only the three existing opt-in hardware tests skipped.

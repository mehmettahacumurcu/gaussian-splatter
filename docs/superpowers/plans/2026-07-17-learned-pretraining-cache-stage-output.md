# Learned Pre-Training Cache and Stage Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live learned-notebook stage output plus verified Drive checkpoints that reuse COLMAP after a later preprocessing failure and reuse the complete training-ready state after a training failure.

**Architecture:** A reusable progress reporter is injected into the existing static runner and exposed to learned substages through a scoped context. A learned-only cache manager publishes two portable, hash-verified checkpoint types to a pipeline-owned sibling Drive folder and rehydrates them into run-local `/content` paths. The static runner gains optional pre-training restore/save hooks, leaving the legacy service composition unchanged.

**Tech Stack:** Python 3.12, dataclasses, JSON/JSONL, SHA-256, pathlib/shutil, subprocess, pytest, nbformat, Google Colab Drive mount.

## Global Constraints

- Apply only to the learned-quality A100 experiment; do not change legacy notebook behavior.
- Keep results at `<data_folder_name>_learned_test_result/` and caches at `<data_folder_name>_learned_test_cache/`.
- Keep automatic Drive flush/unmount and runtime release on terminal success or failure.
- Do not change selection, reconstruction, masking, depth, geometry, training, or polish quality algorithms.
- Never use pickle or absolute `/content` paths in a Drive checkpoint.
- Never reuse partial, corrupt, incompatible, symlinked, path-traversing, or unowned data.
- Restore into local `/content`; never train directly from the Drive mount.
- Key caches by preprocessing producer code, not the whole repository commit, so training-only fixes retain compatibility.
- Do not invent a COLMAP percentage; stream available output and show elapsed-time heartbeats.
- Use TDD and commit/push each verified milestone on `feature/learned-quality-a100`.

## File Structure

- Create `backend/static_pipeline/progress.py` for stage events, scoped reporting, and heartbeats.
- Modify `backend/static_pipeline/runner.py` for optional progress and pre-training cache hooks.
- Modify `backend/static_pipeline/colmap.py` to use the heartbeat executor for long commands.
- Create `experiments/learned_quality/cache.py` for ownership, fingerprints, portable state, validation, publication, and restore.
- Modify `experiments/learned_quality/contracts.py`, `runtime.py`, `runner.py`, and `notebook.py` for learned cache integration.
- Regenerate only `colab/learned_quality_a100_experiment.ipynb` and update `colab/README.md`.
- Add focused tests under `tests/static_pipeline`, `tests/experiments/learned_quality`, and the learned notebook smoke test.

---

### Task 1: Flushed Stage Reporter and Runner Hooks

**Files:**
- Create: `backend/static_pipeline/progress.py`
- Create: `tests/static_pipeline/test_progress.py`
- Modify: `backend/static_pipeline/runner.py`
- Modify: `tests/static_pipeline/test_runner.py`

**Interfaces:**
- Produces: `StageDefinition`, `StageReporter`, `use_stage_reporter`, `current_stage_reporter`, and `run_with_heartbeat`.
- Produces: `PretrainingRestore` plus optional `RunnerServices.restore_pretraining` and `save_pretraining` hooks.
- Produces: optional `reporter` argument on `run_static_notebook`.

- [ ] **Step 1: Write failing reporter lifecycle tests**

Create an in-memory flushed stream and deterministic clock fixture. Test exact start/done lines, JSONL order, elapsed formatting, and unchanged exception propagation:

```python
def test_stage_reporter_flushes_console_and_jsonl(tmp_path):
    output = RecordingStream()
    reporter = StageReporter(
        (StageDefinition("selection", "Frame selection"),),
        stream=output,
        clock=iter((10.0, 12.5)).__next__,
    )
    reporter.bind_log(tmp_path / "progress.jsonl", run_id="run-1")
    with reporter.stage("selection", summary=lambda: "412 frames"):
        pass
    assert output.lines == [
        "[STAGE 01/01] START Frame selection",
        "[STAGE 01/01] DONE  Frame selection - 412 frames - 00:02",
    ]
    assert output.flush_count == 2
    assert [row["status"] for row in read_jsonl(tmp_path / "progress.jsonl")] == ["start", "done"]


def test_stage_reporter_records_failure_and_reraises(tmp_path):
    error = RuntimeError("boom")
    with pytest.raises(RuntimeError) as caught:
        with test_reporter(tmp_path).stage("selection"):
            raise error
    assert caught.value is error
```

- [ ] **Step 2: Run red test**

Run: `python -m pytest tests/static_pipeline/test_progress.py -q`

Expected: collection fails because `backend.static_pipeline.progress` does not exist.

- [ ] **Step 3: Implement reporter and scoped context**

Implement `StageDefinition(stage_id: str, label: str)` and these exact public
signatures:

- `StageReporter.bind_log(path: Path, *, run_id: str) -> None`
- `StageReporter.stage(stage_id: str, *, summary: Callable[[], str | None] | None = None) -> ContextManager[None]`
- `StageReporter.cache_event(action: str, label: str, detail: str) -> None`
- `StageReporter.heartbeat(stage_id: str, elapsed_seconds: float) -> None`
- `StageReporter.skip(stage_id: str, reason: str) -> None`
- `use_stage_reporter(reporter: StageReporter | None) -> ContextManager[None]`
- `current_stage_reporter() -> StageReporter | None`

Validate unique safe IDs, use strict JSON with `allow_nan=False`, flush every console event, store active state in `ContextVar`, and re-raise errors unchanged.

- [ ] **Step 4: Write and implement heartbeat tests**

Use a fake `Popen` that remains alive through two intervals. Assert inherited stdout/stderr, exact command preservation, two heartbeat events, clean completion, and `CalledProcessError` for nonzero status with `check=True`.

Implement `run_with_heartbeat(command: Sequence[str], *, stage_id: str,
check: bool = False, process_factory: ProcessFactory = subprocess.Popen,
heartbeat_seconds: float = 60.0) -> subprocess.CompletedProcess[str]`, with a
small `ProcessFactory` protocol that accepts the command and returns an object
providing `poll()`, `wait()`, `args`, and `returncode`.

Production polling must not pipe COLMAP output or busy-wait. Tests inject clock/wait helpers.

- [ ] **Step 5: Write failing static-runner cache-hook tests**

Assert a restore hit skips copy/select/reconstruct, a miss preserves the legacy order, save occurs after a passing reconstruction and before training, and save failure prevents training and publishes diagnostics:

```python
def test_restore_hit_skips_pretraining_services(tmp_path):
    services = services_with_restore(PretrainingRestore(selection, reconstruction))
    run_static_notebook(spec, runtime_paths=paths, services=services, reporter=reporter)
    assert "restore_pretraining" in calls
    assert not {"copy_input", "select", "colmap_gate"} & set(calls)
    assert calls.index("restore_pretraining") < calls.index("train")


def test_save_finishes_before_training(tmp_path):
    run_static_notebook(spec, runtime_paths=paths, services=services_with_save(), reporter=reporter)
    assert calls.index("save_pretraining") < calls.index("train")
```

- [ ] **Step 6: Add optional hooks and top-level stage scopes**

Add:

```python
@dataclass(frozen=True)
class PretrainingRestore:
    selection: object
    reconstruction: object
```

Place `restore_pretraining` and `save_pretraining` as defaulted final fields on `RunnerServices`. Bind `logs/progress.jsonl` after creating `run_root`. Attempt restore after inventory/preflight; otherwise preserve copy/select/reconstruct and save only after the reconstruction decision passes. Wrap training through Drive publication in reporter scopes. Add the progress log to diagnostics when present.

- [ ] **Step 7: Verify and commit Task 1**

Run:

```text
python -m pytest tests/static_pipeline/test_progress.py tests/static_pipeline/test_runner.py -q
python -m ruff check backend/static_pipeline/progress.py backend/static_pipeline/runner.py tests/static_pipeline/test_progress.py tests/static_pipeline/test_runner.py
python -m black --check backend/static_pipeline/progress.py backend/static_pipeline/runner.py tests/static_pipeline/test_progress.py tests/static_pipeline/test_runner.py
```

Expected: all pass. Commit: `feat: add live notebook stage reporting`.

---

### Task 2: Owned, Portable, Hash-Verified Checkpoint Store

**Files:**
- Create: `experiments/learned_quality/cache.py`
- Create: `tests/experiments/learned_quality/test_cache.py`
- Modify: `experiments/learned_quality/contracts.py`
- Modify: `tests/experiments/learned_quality/test_contracts.py`

**Interfaces:**
- Produces: `CACHE_SUFFIX`, `derive_learned_cache_root`, `CheckpointKind`, `CheckpointInputs`, and `LearnedCheckpointStore`.
- Consumes: existing `stage_fingerprint`, learned/static dataclasses, source digest, model manifest, tool versions, and producer-file paths.

- [ ] **Step 1: Test and add cache path contract**

Assert `/drive/MyDrive/myroom_test` derives `/drive/MyDrive/myroom_test_learned_test_cache`, and reject cache folders as input. Add:

```python
CACHE_SUFFIX = "_learned_test_cache"


def derive_learned_cache_root(input_folder: Path) -> Path:
    return input_folder.with_name(f"{input_folder.name}{CACHE_SUFFIX}")
```

- [ ] **Step 2: Write ownership and atomic-publication tests**

Cover owned-root creation, unowned-root rejection, input-identity mismatch, success marker written last, independent checkpoint generations, partial staging rejection, and cleanup limited to the same generator/input ownership. Require:

```json
{"generator_id":"4dgs-studio.learned-quality-a100","input_identity":"<sha256>","schema_version":1}
```

- [ ] **Step 3: Implement ownership and generation publication**

Define:

```python
class CheckpointKind(StrEnum):
    COLMAP = "colmap"
    PRETRAINING = "pretraining"


@dataclass(frozen=True)
class CheckpointInputs:
    source_digest: str
    settings: Mapping[str, object]
    model_manifest_sha256: str
    tool_versions: Mapping[str, str]
    producer_code_sha256: str
    upstream_fingerprint: str | None = None
```

Stage under `staging/<kind>-<run-id>`, validate copied regular files, write `manifest.json`, write `_SUCCESS.json` last, synchronize, promote without replacement, then remove older owned generations of only that kind.

- [ ] **Step 4: Write portable codec integrity tests**

Build a nested real fixture using `SelectionManifest`, `ReconstructionBundle`, `LearnedReconstructionOutput`, and learned artifact dataclasses. Assert type/tuple/frozenset preservation, relative path rebasing, current inventory replacement, and every file hash. Reject unknown type tags, absolute paths, `..`, backslashes, symlinks, missing/changed files, duplicate paths, non-finite numbers, and unregistered dataclasses.

- [ ] **Step 5: Implement the allow-listed codec**

Use a static registry of final-output dataclasses. Encode tagged dataclasses, tuples, frozensets, mappings, primitives, and relative paths. Decode only registered types, enforce exact field sets, rebase under the local restore root, and replace `SelectionOutput.inventory` with the current source inventory. Never dynamically import cache-supplied names.

- [ ] **Step 6: Write and implement fingerprint invalidation tests**

Assert changes in source, preprocessing settings, model manifest, COLMAP version, producer code, schema, or upstream COLMAP fingerprint invalidate. Assert a training-only source-file change does not.

Hash relevant producer files as sorted `(repo-relative path, sha256)` rows. Record repository commit for audit but exclude it from `stage_fingerprint`.

- [ ] **Step 7: Verify and commit Task 2**

Run:

```text
python -m pytest tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_contracts.py tests/static_pipeline/test_stage_cache.py -q
python -m ruff check experiments/learned_quality/cache.py experiments/learned_quality/contracts.py tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_contracts.py
python -m black --check experiments/learned_quality/cache.py experiments/learned_quality/contracts.py tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_contracts.py
```

Expected: all pass. Commit: `feat: add verified learned checkpoint storage`.

---

### Task 3: COLMAP Checkpoint Reuse and Learned Substage Output

**Files:**
- Modify: `backend/static_pipeline/colmap.py`
- Modify: `experiments/learned_quality/runtime.py`
- Modify: `tests/static_pipeline/test_colmap.py`
- Modify: `tests/experiments/learned_quality/test_runtime.py`

**Interfaces:**
- Consumes: `current_stage_reporter`, `run_with_heartbeat`, and `LearnedCheckpointStore`.
- Produces: optional `checkpoint_store` parameter on `run_learned_reconstruction` and a restored normal `ColmapAttempt`.

- [ ] **Step 1: Write COLMAP executor regression tests**

Inject a command executor into `run_colmap_attempt`. Assert feature extraction, matching, mapping, and model conversion preserve exact arguments and `check=True`, route through `stage_id="classical_colmap"`, and propagate nonzero status unchanged.

- [ ] **Step 2: Route long COLMAP commands through heartbeats**

Keep short help/version probes on `subprocess.run`. Route expensive commands through `run_with_heartbeat` without capturing stdout/stderr, so native COLMAP output stays visible.

- [ ] **Step 3: Write learned substage reporter tests**

Mock learned stage functions and assert ordered start/done events for these exact IDs:

```text
da3_anchor
classical_colmap
da3_metric_sky
semantic_masks
optical_flow
mask_fusion
geometry_comparison
photometric_validation
final_pose_depth
dense_seed_fusion
```

Raise from optical flow and assert a `FAIL` event followed by the identical exception object.

- [ ] **Step 4: Add scoped learned reporting without moving algorithms**

Wrap each existing runtime call using `current_stage_reporter().stage(stage_id)` when a reporter is active and `nullcontext()` otherwise. Preserve policies, thresholds, batching, CUDA release order, call order, and return values exactly.

- [ ] **Step 5: Write a COLMAP cache-hit test**

Use mocked learned models and a fake classical runner that creates a minimal real text COLMAP model. First execution calls the runner and publishes. Delete local work, rerun with the same inputs, and assert the runner is skipped, restored paths are under the new local root, model measurement and rigid-scene validation still run, and `[CACHE HIT]` appears. Change a selected frame hash and assert a miss plus another classical call.

- [ ] **Step 6: Integrate verified classical-prepass checkpointing**

Thread optional `checkpoint_store` through `run_learned_reconstruction` and `_run_evidence_cycle`. Compute the COLMAP fingerprint after selection. Restore into `classical-prepass` on a valid hit; otherwise run the current classical candidate. Require the existing readable-model and rigid-scene invariants before publishing and again after restoring. Do not introduce a new quality threshold.

- [ ] **Step 7: Verify and commit Task 3**

Run:

```text
python -m pytest tests/static_pipeline/test_colmap.py tests/experiments/learned_quality/test_runtime.py -q
python -m ruff check backend/static_pipeline/colmap.py experiments/learned_quality/runtime.py tests/static_pipeline/test_colmap.py tests/experiments/learned_quality/test_runtime.py
python -m black --check backend/static_pipeline/colmap.py experiments/learned_quality/runtime.py tests/static_pipeline/test_colmap.py tests/experiments/learned_quality/test_runtime.py
```

Expected: all pass, including duplicate-static-track coverage. Commit: `feat: cache verified learned COLMAP preprocessing`.

---

### Task 4: Complete Pre-Training Restore Before Gaussian Training

**Files:**
- Modify: `experiments/learned_quality/cache.py`
- Modify: `experiments/learned_quality/runner.py`
- Modify: `backend/static_pipeline/runner.py`
- Modify: `tests/experiments/learned_quality/test_cache.py`
- Modify: `tests/experiments/learned_quality/test_runner.py`
- Modify: `tests/static_pipeline/test_runner.py`

**Interfaces:**
- Consumes: current source inventory, run-local root, model manifest, `SelectionOutput`, `LearnedReconstructionOutput`, and Task 1 hooks.
- Produces: learned-only closures that return `PretrainingRestore` or publish the complete snapshot before training.

- [ ] **Step 1: Write end-to-end mocked resume test**

Use real portable fixtures and mocked expensive services. First run must call copy, selection, reconstruction, save, then fail in training after the checkpoint succeeds. Use a fresh work root and rerun; assert copy, selection, reconstruction, and COLMAP are skipped while training receives paths under the fresh work root:

The test records first-run and second-run calls in separate lists. The first run
uses `pytest.raises(RuntimeError, match="training failed")` and asserts
`first_calls.index("save_pretraining") < first_calls.index("train")`. The second
run uses a new `NotebookRuntimePaths` work root and asserts:

```python
assert second_calls == [
    "restore_pretraining",
    "train",
    "polish",
    "metadata_preview",
    "assemble_reports",
    "validate_bundle",
    "publish",
]
assert all(path.is_relative_to(second_work_root) for path in training_paths)
```

- [ ] **Step 2: Write complete-cache failure tests**

Assert changed artifact bytes cause visible invalidation and safe recomputation; an unowned root causes a hard ownership error before training; restore-copy corruption is a hard failure; and cache publication failure prevents training but reaches learned diagnostics.

- [ ] **Step 3: Implement snapshot selection and rehydration**

Snapshot only preprocessing-owned roots referenced by the two service outputs: selection/backfill frames, learned evidence, reconstruction candidates/final model, photometric output, final-pose depth, validated depth, and a verified model-manifest copy. Exclude raw input, training, polish, reports, bundle, and diagnostics. Store service state in `state.json`, inventory all files, and rebase every decoded path to the fresh local restore root.

- [ ] **Step 4: Compose learned cache hooks**

In `make_learned_quality_services`, provide:

```python
def restore_pretraining(*, source_inventory, spec, hardware, run_root):
    return checkpoint_store.restore_pretraining(
        source_inventory=source_inventory,
        spec=spec,
        hardware=hardware,
        destination=run_root / "pretraining-restored",
    )


def save_pretraining(*, source_inventory, selection, reconstruction, spec, hardware, run_root):
    checkpoint_store.publish_pretraining(
        source_inventory=source_inventory,
        selection=selection,
        reconstruction=reconstruction,
        spec=spec,
        hardware=hardware,
        run_root=run_root,
    )
```

Wrap the default reconstruction callable so it receives the same store for COLMAP. Custom contexts without a store remain unchanged.

- [ ] **Step 5: Define learned stage order and summaries**

Create one stage-definition tuple covering discovery, preflight, cache restore, source copy, selection, all learned substages, cache save, training, polish, metadata, reports, validation, and result publication. Print cache miss reasons, Drive paths, registered/selected counts, and elapsed durations. Do not alter structured quality-report contracts.

- [ ] **Step 6: Preserve learned failure diagnostics**

Ensure `logs/progress.jsonl` is published when bound, alongside existing `pipeline.log` and `experiment_report.json`. Trainer failures retain the complete cache and still flush Drive and release the runtime.

- [ ] **Step 7: Verify and commit Task 4**

Run:

```text
python -m pytest tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_runner.py tests/static_pipeline/test_runner.py -q
python -m pytest tests/experiments/learned_quality -q
python -m ruff check backend/static_pipeline/runner.py experiments/learned_quality/cache.py experiments/learned_quality/runner.py tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_runner.py tests/static_pipeline/test_runner.py
python -m black --check backend/static_pipeline/runner.py experiments/learned_quality/cache.py experiments/learned_quality/runner.py tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_runner.py tests/static_pipeline/test_runner.py
```

Expected: dependency-light tests pass; only declared GPU tests may skip. Commit: `feat: resume learned runs from Drive preprocessing`.

---

### Task 5: Generated Notebook, Documentation, and Full Verification

**Files:**
- Modify: `experiments/learned_quality/notebook.py`
- Modify: `tests/integration/test_learned_quality_notebook_smoke.py`
- Modify: `colab/README.md`
- Regenerate: `colab/learned_quality_a100_experiment.ipynb`

**Interfaces:**
- Consumes: the verified runtime commit produced by Tasks 1-4.
- Produces: deterministic checked-in notebook pinned to that immutable commit.

- [ ] **Step 1: Write notebook smoke-test failure**

Assert the path-spec cell includes `_learned_test_cache` and `Pre-training cache:`. Retain assertions for `PYTHONUNBUFFERED`, `-u`, module execution, result validation, Drive flush before runtime release, and no viewer assets.

- [ ] **Step 2: Run red smoke test**

Run: `python -m pytest tests/integration/test_learned_quality_notebook_smoke.py -q`

Expected: the new cache-path assertion fails.

- [ ] **Step 3: Update generator and documentation**

Print the expected cache sibling in the path-spec cell. Document cache hits/misses, both checkpoint boundaries, ownership/invalidation, elapsed COLMAP heartbeats, training iteration output, and that training cannot resume from a partial iteration.

- [ ] **Step 4: Commit/push runtime, then regenerate notebook**

Confirm Tasks 1-4 are pushed. Resolve `git rev-parse HEAD^{commit}`. Generate the notebook through the existing generator using that full 40-character runtime commit; never hand-edit notebook JSON. The future notebook-only commit must not become the inner runtime pin.

- [ ] **Step 5: Run focused and broad verification**

Run:

```text
python -m pytest tests/static_pipeline/test_progress.py tests/static_pipeline/test_runner.py tests/static_pipeline/test_stage_cache.py tests/static_pipeline/test_colmap.py -q
python -m pytest tests/experiments/learned_quality -q
python -m pytest tests/integration/test_learned_quality_bootstrap.py tests/integration/test_learned_quality_notebook_smoke.py -q
python -m ruff check backend/static_pipeline experiments/learned_quality tests/static_pipeline tests/experiments/learned_quality tests/integration/test_learned_quality_notebook_smoke.py
python -m black --check backend/static_pipeline experiments/learned_quality tests/static_pipeline tests/experiments/learned_quality tests/integration/test_learned_quality_notebook_smoke.py
git diff --check
```

Expected: dependency-light tests pass; only explicit GPU gates may skip.

- [ ] **Step 6: Audit, commit, and push notebook milestone**

Confirm the generated notebook contains the runtime pin, unbuffered module execution, cache-path output, Drive flush/unmount, and `runtime.unassign()`. Confirm no legacy notebook changed. Review `git status` and `git diff --staged`, commit `feat: add resumable learned notebook preprocessing`, push, and verify local HEAD equals `origin/feature/learned-quality-a100`.

---

## Final Acceptance

- First run prints every long learned boundary and quiet COLMAP heartbeats.
- Failure after COLMAP leaves a verified COLMAP Drive generation.
- Failure during Gaussian training leaves a verified complete pre-training generation.
- A compatible fresh runtime restores the complete generation and reaches training without COLMAP.
- Changed, partial, corrupt, unsafe, or incompatible data never yields a cache hit.
- Training-only source changes retain cache compatibility.
- Trainer iterations remain live and unbuffered.
- Result, diagnostics, Drive flush, and runtime-release contracts remain unchanged.
- Checked-in notebook is deterministic and pinned to the verified runtime commit.

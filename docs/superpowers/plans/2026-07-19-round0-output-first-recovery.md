# Round 0 Output-First Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated, opt-in A100 notebook mode that restores the complete Round 0 learned-quality evidence lineage, transparently accepts only a guarded cached COLMAP model, trains one maximum-quality Gaussian splat, publishes it to Drive, and releases the Colab runtime.

**Architecture:** Keep every production and legacy caller strict by default. Thread `round0_output_first_v1` only from the standalone experiment notebook through the learned-quality service closure, restore the exact COLMAP checkpoint referenced by the complete masks lineage, and represent the exception with a separate immutable acceptance record rather than changing `GateDecision`. Add a default-off validation callback at the shared training boundary and replace redundant Drive rereads with manifest-guided verified streaming copies.

**Tech Stack:** Python 3.12, Pydantic v2, frozen dataclasses, pathlib, hashlib, pytest, Google Colab notebook JSON, existing learned-quality milestone/cache contracts, existing static Gaussian training pipeline.

## Global Constraints

- The opt-in mode name is exactly `round0_output_first_v1`; the default is exactly `strict`.
- The legacy notebook, frontend, API, generated public notebook flow, production quality presets, and `<input_folder>_result` remain unchanged.
- The recovery result path remains `<input_folder>_learned_test_result`; no web viewer is added.
- A best-effort acceptance never changes `GateDecision.passed`, removes strict failures, or claims strict-pass geometry.
- Guard thresholds are inclusive: registered ratio `0.80`, registered share `0.95`, maximum interior gap `10.0 s`, start/end gap `1.0 s`, median reprojection `1.5 px`, p95 reprojection `2.5 px`, median track length `3.0`, sparse points `10_000`, and valid finite names/intrinsics/poses.
- Only strict failures `registered_ratio`, `interior_gap`, and `median_reprojection` may be waived.
- Recovery restores the complete masks lineage and its exact `BaseEvidenceState.colmap_ref`; it never reruns COLMAP, DA3, semantic masks, or optical flow before geometry acceptance.
- Cache payload bytes are copied from Drive once while their SHA-256 and size are verified against the immutable manifest.
- Every success or failure path retains unconditional Drive flush/unmount and Colab runtime release.
- Tests use `C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe` because the default Anaconda interpreter lacks Torch.

---

## File responsibility map

- `experiments/learned_quality/contracts.py`: opt-in run mode and immutable geometry-acceptance contract.
- `experiments/learned_quality/geometry.py`: guarded policy evaluation and cached-COLMAP recovery; no training or Drive concerns.
- `experiments/learned_quality/milestones.py`: portable geometry acceptance in milestone state.
- `experiments/learned_quality/runtime.py`: lineage restore, strict/recovery branch selection, and milestone fingerprints.
- `experiments/learned_quality/runner.py`: service closure, reports, and experiment orchestration.
- `backend/static_pipeline/training.py`: default-strict validation seam shared by all training callers.
- `experiments/learned_quality/training.py`: guarded validator supplied only by recovery mode.
- `experiments/learned_quality/cache.py`: metadata validation plus one-pass payload copy and verification.
- `experiments/learned_quality/notebook.py`: fixed recovery-mode A100 notebook and explicit best-effort UI text.
- `tests/experiments/learned_quality/*`: learned-contract, geometry, runtime, cache, reports, runner, and notebook coverage.
- `tests/static_pipeline/test_training.py`: proof that the shared default remains strict.
- `colab/learned_quality_a100_experiment.ipynb`: generated, commit-pinned runnable artifact.

---

### Task 1: Add the opt-in run mode and acceptance contract

**Files:**
- Modify: `experiments/learned_quality/contracts.py`
- Test: `tests/experiments/learned_quality/test_contracts.py`

**Interfaces:**
- Produces: `RecoveryMode = Literal["strict", "round0_output_first_v1"]`.
- Produces: `GeometryAcceptance(policy_version, mode, selection_digest, model_hashes, strict_failures, metrics, checks, colmap_fingerprint)`.
- Produces: `LearnedReconstructionOutput.acceptance: GeometryAcceptance | None` for use by runtime, training, reports, and milestone serialization.

- [ ] **Step 1: Write failing contract tests**

```python
def test_learned_spec_defaults_to_strict_recovery_mode() -> None:
    spec = LearnedQualityRunSpec(input_folder="room")
    assert spec.recovery_mode == "strict"


def test_learned_spec_accepts_only_named_round0_recovery_mode() -> None:
    spec = LearnedQualityRunSpec(
        input_folder="room",
        recovery_mode="round0_output_first_v1",
    )
    assert spec.recovery_mode == "round0_output_first_v1"
    with pytest.raises(ValidationError):
        LearnedQualityRunSpec(input_folder="room", recovery_mode="best_effort")


def test_geometry_acceptance_freezes_hashes_and_checks(metrics: ModelMetrics) -> None:
    acceptance = GeometryAcceptance(
        policy_version="output-first-v1",
        mode="best_effort",
        selection_digest="a" * 64,
        model_hashes={name: "b" * 64 for name in MODEL_FILES},
        strict_failures=("registered_ratio",),
        metrics=metrics,
        checks={"registered_ratio": True},
        colmap_fingerprint="c" * 64,
    )
    with pytest.raises(TypeError):
        acceptance.model_hashes["images.txt"] = "d" * 64
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/experiments/learned_quality/test_contracts.py -q
```

Expected: collection or assertion failure because `recovery_mode` and `GeometryAcceptance` do not exist.

- [ ] **Step 3: Implement the exact contract**

```python
RecoveryMode = Literal["strict", "round0_output_first_v1"]
GeometryAcceptanceMode = Literal["strict", "best_effort"]
GeometryPolicyVersion = Literal["strict-v1", "output-first-v1"]
MODEL_TEXT_FILES = ("cameras.txt", "images.txt", "points3D.txt")


class LearnedQualityRunSpec(StrictModel):
    schema_version: Literal[1] = 1
    input_folder: str
    recovery_mode: RecoveryMode = "strict"
    publish: LearnedPublishSpec = Field(default_factory=LearnedPublishSpec)


@dataclass(frozen=True)
class GeometryAcceptance:
    policy_version: GeometryPolicyVersion
    mode: GeometryAcceptanceMode
    selection_digest: str
    model_hashes: Mapping[str, str]
    strict_failures: tuple[str, ...]
    metrics: ModelMetrics
    checks: Mapping[str, bool]
    colmap_fingerprint: str

    def __post_init__(self) -> None:
        digests = {
            "selection_digest": self.selection_digest,
            "colmap_fingerprint": self.colmap_fingerprint,
            **dict(self.model_hashes),
        }
        if set(self.model_hashes) != set(MODEL_TEXT_FILES):
            raise ValueError("geometry acceptance must hash all COLMAP text files")
        for label, digest in digests.items():
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(f"{label} must be a lowercase SHA-256")
        object.__setattr__(
            self,
            "model_hashes",
            MappingProxyType(dict(sorted(self.model_hashes.items()))),
        )
        object.__setattr__(
            self,
            "checks",
            MappingProxyType(dict(sorted(self.checks.items()))),
        )
```

Add `acceptance: GeometryAcceptance | None = None` to `LearnedReconstructionOutput`. Import `MappingProxyType` and `ModelMetrics`. Preserve `to_static_run_spec` unchanged so the mode never leaks into the production static specification.

- [ ] **Step 4: Run contract and serialization tests and confirm GREEN**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/experiments/learned_quality/test_contracts.py tests/experiments/learned_quality/test_cache.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the contract milestone**

```powershell
git add experiments/learned_quality/contracts.py tests/experiments/learned_quality/test_contracts.py
git diff --staged
git commit -m "feat: add output-first recovery contract"
git push
```

---

### Task 2: Implement guarded cached-COLMAP geometry acceptance

**Files:**
- Modify: `experiments/learned_quality/geometry.py`
- Test: `tests/experiments/learned_quality/test_geometry.py`

**Interfaces:**
- Consumes: `GeometryAcceptance`, `MODEL_TEXT_FILES`, `SelectionOutput`, `LearnedCheckpointStore.restore_colmap`.
- Produces: `evaluate_output_first_checks(decision: GateDecision) -> Mapping[str, bool]`.
- Produces: `accept_output_first_geometry(selection, restored_attempt, output_root, checkpoint_fingerprint) -> tuple[ReconstructionBundle, tuple[GeometryCandidateReport, ...], GeometryAcceptance]`.
- Produces: `OutputFirstGeometryError` carrying `strict_failures` and `guarded_failures` for the durable failure receipt.

- [ ] **Step 1: Add failing boundary, waiver, and no-rerun tests**

```python
@pytest.mark.parametrize(
    ("field", "accepted", "rejected"),
    [
        ("registered_ratio", 0.80, 0.799999),
        ("registered_share", 0.95, 0.949999),
        ("max_interior_gap_s", 10.0, 10.000001),
        ("start_gap_s", 1.0, 1.000001),
        ("end_gap_s", 1.0, 1.000001),
        ("median_reprojection_error_px", 1.5, 1.500001),
        ("p95_reprojection_error_px", 2.5, 2.500001),
        ("median_track_length", 3.0, 2.999999),
        ("sparse_point_count", 10_000, 9_999),
    ],
)
def test_output_first_thresholds_are_inclusive(field, accepted, rejected):
    manifest = _manifest(100)
    model = Path("model")
    passing_metrics = _metrics(
        model,
        manifest,
        registered_ratio=0.80,
        registered_share=0.95,
        max_interior_gap_s=10.0,
        start_gap_s=1.0,
        end_gap_s=1.0,
        median_error=1.5,
        p95_error=2.5,
        median_track=3.0,
        sparse_points=10_000,
    )
    attribute = {
        "median_error": "median_reprojection_error_px",
        "p95_error": "p95_reprojection_error_px",
        "median_track": "median_track_length",
        "sparse_points": "sparse_point_count",
    }.get(field, field)
    passing = _decision(
        dataclasses.replace(passing_metrics, **{attribute: accepted}),
        passed=False,
        failures=("registered_ratio", "interior_gap", "median_reprojection"),
    )
    failing = _decision(
        dataclasses.replace(passing_metrics, **{attribute: rejected}),
        passed=False,
        failures=("registered_ratio", "interior_gap", "median_reprojection"),
    )
    assert all(evaluate_output_first_checks(passing).values())
    assert not all(evaluate_output_first_checks(failing).values())


def test_output_first_never_waives_unapproved_strict_failure() -> None:
    manifest = _manifest(100)
    decision = _decision(
        _metrics(Path("model"), manifest, sparse_points=10_000),
        passed=False,
        failures=("dominant_component",),
    )
    checks = evaluate_output_first_checks(decision)
    assert checks["strict_failures_are_waivable"] is False


def test_cached_geometry_restores_and_measures_without_running_colmap(
    tmp_path: Path,
    selection: SelectionOutput,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(800)
    frames = _artifacts(tmp_path / "frames", manifest)
    selection = SelectionOutput(
        inventory=cast(SourceInventory, SimpleNamespace(digest=manifest.source_digest)),
        manifest=manifest,
        frames_dir=tmp_path / "frames",
        source_manifest_path=tmp_path / "selection_manifest.json",
    )
    model = tmp_path / "attempt" / "sparse" / "0"
    model.mkdir(parents=True)
    for name in MODEL_TEXT_FILES:
        (model / name).write_text(f"{name}\n", encoding="utf-8")
    attempt = ColmapAttempt(
        root=tmp_path / "attempt",
        database_path=tmp_path / "attempt" / "colmap.db",
        model_dirs=(model,),
        colmap_version="COLMAP 3.12.6",
        fingerprint="c" * 64,
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("recovery reran a geometry candidate")

    monkeypatch.setattr(geometry, "run_classical_colmap", fail_if_called, raising=False)
    monkeypatch.setattr(geometry, "run_learned_hybrid", fail_if_called, raising=False)
    room_metrics = _metrics(
        model,
        manifest,
        registered_ratio=658 / 800,
        registered_count=658,
        registered_share=1.0,
        max_interior_gap_s=9.59,
        median_error=1.217,
        p95_error=2.194,
        median_track=5.0,
        sparse_points=65_367,
    )
    strict = _decision(
        room_metrics,
        passed=False,
        failures=("registered_ratio", "interior_gap", "median_reprojection"),
        retry=True,
    )
    monkeypatch.setattr(geometry, "measure_models", lambda *_args: (room_metrics,))
    monkeypatch.setattr(
        geometry,
        "evaluate_reconstruction",
        lambda *_args, **_kwargs: strict,
    )
    bundle, reports, acceptance = accept_output_first_geometry(
        selection,
        attempt,
        tmp_path / "accepted",
        checkpoint_fingerprint="c" * 64,
    )
    assert bundle.decision.passed is False
    assert acceptance.mode == "best_effort"
    assert acceptance.strict_failures == bundle.decision.failures
    assert reports[0].candidate_id == "classical"
```

Also add a fixture matching the known room shape (`658/800`, 65,367 points) and assert it is accepted while its strict `GateDecision` remains failed.

- [ ] **Step 2: Run geometry tests and confirm RED**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/experiments/learned_quality/test_geometry.py -q
```

Expected: import failures for the two new functions.

- [ ] **Step 3: Implement guarded evaluation, ranking, hashing, and publication**

```python
OUTPUT_FIRST_ALLOWED_FAILURES = frozenset(
    {"registered_ratio", "interior_gap", "median_reprojection"}
)


def evaluate_output_first_checks(decision: GateDecision) -> Mapping[str, bool]:
    m = decision.dominant
    return MappingProxyType(
        {
            "registered_ratio": m.registered_ratio >= 0.80,
            "registered_share": m.registered_share >= 0.95,
            "interior_gap": m.max_interior_gap_s <= 10.0,
            "start_gap": m.start_gap_s <= 1.0,
            "end_gap": m.end_gap_s <= 1.0,
            "median_reprojection": m.median_reprojection_error_px <= 1.5,
            "p95_reprojection": m.p95_reprojection_error_px <= 2.5,
            "median_track_length": m.median_track_length >= 3.0,
            "sparse_point_count": m.sparse_point_count >= 10_000,
            "model_validity": m.valid_names_intrinsics_and_poses,
            "strict_failures_are_waivable": set(decision.failures)
            <= OUTPUT_FIRST_ALLOWED_FAILURES,
        }
    )


def _output_first_rank(decision: GateDecision) -> tuple[object, ...]:
    m = decision.dominant
    endpoint_count = int(m.start_gap_s <= 1.0) + int(m.end_gap_s <= 1.0)
    return (
        endpoint_count,
        -m.max_interior_gap_s,
        m.registered_count,
        -m.median_reprojection_error_px,
        -m.p95_reprojection_error_px,
        m.median_track_length,
        m.sparse_point_count,
    )


def _model_text_hashes(model_dir: Path) -> Mapping[str, str]:
    return MappingProxyType(
        {name: _sha256_file(model_dir / name) for name in MODEL_TEXT_FILES}
    )
```

`accept_output_first_geometry` must call existing `measure_models` and `evaluate_reconstruction` for every restored sparse model, keep candidates whose guarded checks all pass, select the maximum candidate using `_output_first_rank`, copy the model through `_publish_validated_model`, create a `GeometryCandidateReport` whose `candidate_id` is `classical`, and construct `GeometryAcceptance` with the copied model hashes. If the original decision is strict-pass, record `policy_version="strict-v1"` and `mode="strict"`; otherwise record `policy_version="output-first-v1"` and `mode="best_effort"`. Raise `OutputFirstGeometryError("restored Round 0 COLMAP model failed output-first-v1", strict_failures=decision.failures, guarded_failures=tuple(name for name, passed in checks.items() if not passed))` if no candidate passes.

- [ ] **Step 4: Run geometry tests and confirm GREEN**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/experiments/learned_quality/test_geometry.py -q
```

Expected: all geometry tests pass, including every threshold and the known Round 0 metric case.

- [ ] **Step 5: Commit the guarded geometry milestone**

```powershell
git add experiments/learned_quality/geometry.py tests/experiments/learned_quality/test_geometry.py
git diff --staged
git commit -m "feat: accept guarded round0 geometry"
git push
```

---

### Task 3: Route the complete masks lineage through recovery without recomputation

**Files:**
- Modify: `experiments/learned_quality/cache.py`
- Modify: `experiments/learned_quality/milestones.py`
- Modify: `experiments/learned_quality/runtime.py`
- Modify: `experiments/learned_quality/runner.py`
- Test: `tests/experiments/learned_quality/test_milestones.py`
- Test: `tests/experiments/learned_quality/test_cache.py`
- Test: `tests/experiments/learned_quality/test_runtime.py`
- Test: `tests/experiments/learned_quality/test_runner.py`

**Interfaces:**
- Consumes: `RecoveryMode`, `GeometryAcceptance`, `accept_output_first_geometry`.
- Produces: `GeometryMilestoneState.acceptance`.
- Produces: `LearnedCheckpointStore.find_latest_complete_lineage(root_kind, selection_fingerprint) -> Mapping[CheckpointKind, MilestoneRef] | None`.
- Produces: `_restore_output_first_evidence(...) -> LearnedArtifacts` which restores and never computes.
- Produces: `run_learned_reconstruction(selection, spec, hardware, output_root, *, checkpoint_store=None, milestone_session=None, recovery_mode: RecoveryMode = "strict")`.
- Produces: `make_learned_quality_services(context, *, cache_root=None, recovery_mode: RecoveryMode = "strict")`.

- [ ] **Step 1: Add failing lineage and milestone tests**

```python
def test_latest_complete_lineage_chooses_round0_masks_not_newer_semantic(
    tmp_path: Path,
) -> None:
    store, round0, round1 = _published_round0_and_round1_graphs(tmp_path)
    refs = store.find_latest_complete_lineage(
        CheckpointKind.MASKS,
        selection_fingerprint=round0[CheckpointKind.SELECTION].fingerprint,
    )
    assert refs == {
        kind: round0[kind]
        for kind in (
            CheckpointKind.SELECTION,
            CheckpointKind.COLMAP,
            CheckpointKind.BASE_EVIDENCE,
            CheckpointKind.SEMANTIC,
            CheckpointKind.MOTION,
            CheckpointKind.MASKS,
        )
    }
    assert round1[CheckpointKind.SEMANTIC] not in refs.values()


def test_recovery_evidence_requires_complete_masks_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = cast(SelectionOutput, SimpleNamespace())
    session = cast(
        MilestoneSession,
        SimpleNamespace(
            selection_ref=MilestoneRef(CheckpointKind.SELECTION, "a" * 64),
            store=SimpleNamespace(find_latest_complete_lineage=lambda *_a, **_k: None),
        ),
    )
    monkeypatch.setattr(runtime_module, "run_semantic_evidence", _fail_if_called)
    monkeypatch.setattr(runtime_module, "run_motion_evidence", _fail_if_called)
    with pytest.raises(ValueError, match="complete Round 0 masks lineage"):
        _restore_output_first_evidence(
            selection,
            (),
            tmp_path / "recovery",
            milestone_session=session,
        )


def test_geometry_milestone_preserves_acceptance(
    tmp_path: Path,
    acceptance: GeometryAcceptance,
    bundle: ReconstructionBundle,
    candidate: GeometryCandidateReport,
) -> None:
    state = GeometryMilestoneState(
        bundle=bundle,
        frames_dir=tmp_path / "frames",
        geometry_candidates=(candidate,),
        acceptance=acceptance,
    )
    assert state.acceptance is acceptance
    assert GeometryMilestoneState.__dataclass_params__.frozen is True
```

Define `_published_round0_and_round1_graphs` beside the test using the existing
`MilestoneState`, `MilestoneRef`, and `store.publish_milestone` helpers already
used in `test_store_publishes_and_restores_portable_milestone_graph`. It must
publish Round 0 through `masks`, then publish a second selection/base/semantic
branch with later file mtimes but no motion or masks. Define `_fail_if_called`
as `raise AssertionError("recovery attempted paid recomputation")`.

Add a runner test asserting `run_learned_quality_notebook` closes over `spec.recovery_mode`, while `make_learned_quality_services(context)` defaults to `strict`.

- [ ] **Step 2: Run runtime/milestone/runner tests and confirm RED**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/experiments/learned_quality/test_milestones.py tests/experiments/learned_quality/test_runtime.py tests/experiments/learned_quality/test_runner.py -q
```

Expected: signature, missing-method, and missing-field failures.

- [ ] **Step 3: Thread mode and acceptance through the isolated learned path**

```python
@dataclass(frozen=True)
class GeometryMilestoneState:
    bundle: ReconstructionBundle
    frames_dir: Path
    geometry_candidates: tuple[GeometryCandidateReport, ...]
    acceptance: GeometryAcceptance | None = None


def _geometry_milestone_ref(
    session: MilestoneSession,
    masks_ref: MilestoneRef,
    hardware: HardwareInfo,
    recovery_mode: RecoveryMode,
) -> MilestoneRef:
    settings = {
        "recovery_mode": recovery_mode,
        "guard_policy": "output-first-v1"
        if recovery_mode == "round0_output_first_v1"
        else "strict-v1",
    }
    if recovery_mode == "strict":
        settings.update(
            candidate_rounds=2,
            backfill_max_per_interval=2,
            use_gpu=hardware.colmap_gpu_sift is True,
        )
    return session.make_ref(
        CheckpointKind.GEOMETRY,
        upstream={
            CheckpointKind.SELECTION: session.selection_ref.fingerprint,
            CheckpointKind.MASKS: masks_ref.fingerprint,
        },
        settings=settings,
        producer_paths=_GEOMETRY_PRODUCER_PATHS,
    )
```

In `run_learned_reconstruction`, preserve the strict body. For recovery mode, require the restored masks graph, read `base.colmap_ref`, call `checkpoint_store.restore_colmap` into a fresh local directory, call `accept_output_first_geometry`, print the explicit `[GEOMETRY] BEST-EFFORT ROUND 0 ACCEPTED` banner when applicable, store the acceptance in `GeometryMilestoneState`, and continue through the existing final track audit, photometric, final-pose depth, dense-seed, and final-pretraining code.

Add `find_latest_complete_lineage` by enumerating owned generations of
`root_kind`, sorting valid candidates by `_SUCCESS.json` mtime newest-first,
calling existing `validate_milestone_graph` for each root, and returning the
first graph containing the requested selection fingerprint and every upstream
kind required by the root. `_restore_output_first_evidence` calls this method
for `CheckpointKind.MASKS`, restores the base, semantic, motion, and masks refs
with `_restore_stage_state`, and raises before any model runner when one state is
missing or has the wrong type.

```python
def make_learned_quality_services(
    context: LearnedQualityContext,
    *,
    cache_root: Path | None = None,
    recovery_mode: RecoveryMode = "strict",
) -> StaticPipelineServices:
    def cached_reconstruct(selection, static_spec, hardware, output_root):
        return context.reconstruct(
            selection,
            static_spec,
            hardware,
            output_root,
            checkpoint_store=checkpoint_store,
            milestone_session=milestone_session,
            recovery_mode=recovery_mode,
        )
```

Make `run_learned_quality_notebook` pass `spec.recovery_mode` into `make_learned_quality_services`. Do not add the mode to `StaticNotebookRunSpec`.

- [ ] **Step 4: Run runtime/milestone/runner tests and confirm GREEN**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/experiments/learned_quality/test_milestones.py tests/experiments/learned_quality/test_runtime.py tests/experiments/learned_quality/test_runner.py -q
```

Expected: all selected tests pass, including strict-path regression tests.

- [ ] **Step 5: Commit the recovery-lineage milestone**

```powershell
git add experiments/learned_quality/cache.py experiments/learned_quality/milestones.py experiments/learned_quality/runtime.py experiments/learned_quality/runner.py tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_milestones.py tests/experiments/learned_quality/test_runtime.py tests/experiments/learned_quality/test_runner.py
git diff --staged
git commit -m "feat: restore round0 evidence lineage"
git push
```

---

### Task 4: Add a default-strict training validation seam

**Files:**
- Modify: `backend/static_pipeline/training.py`
- Modify: `experiments/learned_quality/training.py`
- Test: `tests/static_pipeline/test_training.py`
- Test: `tests/experiments/learned_quality/test_training.py`

**Interfaces:**
- Produces: `ReconstructionValidator = Callable[[PreparedTrainingInput, GateDecision, Mapping[str, str]], None]`.
- Produces: `run_validated_training(prepared, spec, *, source_long_edge=None, pipeline_runner=None, reconstruction_validator: ReconstructionValidator | None = None)`.
- Consumes: `GeometryAcceptance` from `LearnedReconstructionOutput` in guarded learned-quality validation.

- [ ] **Step 1: Add failing default-strict and guarded-integrity tests**

```python
def test_default_training_validator_still_rejects_stored_failed_decision(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    failed = replace(
        prepared.reconstruction.decision,
        passed=False,
        failures=("registered_ratio",),
    )
    prepared = replace(
        prepared,
        reconstruction=replace(
            prepared.reconstruction,
            decision=failed,
            decisions=(failed,),
        ),
    )
    runner = RecordingRunner()
    with pytest.raises(ValueError, match="passing reconstruction decision"):
        run_validated_training(prepared, _spec(), pipeline_runner=runner)
    assert runner.calls == []


def test_custom_validator_receives_fresh_measurement_and_model_hashes(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    observed: dict[str, object] = {}
    def validator(prepared, current_decision, model_hashes):
        observed.update(decision=current_decision, hashes=dict(model_hashes))
    run_validated_training(
        prepared,
        _spec(),
        pipeline_runner=RecordingRunner(),
        reconstruction_validator=validator,
    )
    decision = cast(GateDecision, observed["decision"])
    assert decision.dominant.model_dir == prepared.reconstruction.accepted_model_dir
    assert set(observed["hashes"]) == set(MODEL_TEXT_FILES)


def test_output_first_validator_rejects_model_mutation(
    tmp_path: Path,
    learned_reconstruction: LearnedReconstructionOutput,
) -> None:
    prepared = _prepared(tmp_path)
    expected = dict(learned_reconstruction.acceptance.model_hashes)
    observed = dict(expected)
    observed["images.txt"] = "d" * 64
    validator = _output_first_training_validator(learned_reconstruction)
    with pytest.raises(ValueError, match="acceptance model hash"):
        validator(
            prepared,
            learned_reconstruction.decision,
            observed,
        )


def test_run_learned_training_supplies_guarded_validator_only_for_best_effort(
    tmp_path: Path,
    learned_reconstruction: LearnedReconstructionOutput,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = cast(
        PreparedTrainingInput,
        SimpleNamespace(reconstruction=learned_reconstruction.bundle),
    )
    captured: dict[str, object] = {}
    fake_experiment = SimpleNamespace(
        density_history_path=tmp_path / "density.json",
        final_gaussian_count=1,
        training_rgb_digest="a" * 64,
        fallbacks=(),
    )
    fake_experiment.density_history_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        learned_training,
        "make_experiment_pipeline_runner",
        lambda *_args, **_kwargs: fake_experiment,
    )
    def fake_validated(*_args: object, **kwargs: object) -> TrainingResult:
        captured.update(kwargs)
        return cast(
            TrainingResult,
            SimpleNamespace(
                raw_ply_path=tmp_path / "splat.ply",
                status={},
                resolved_config=object(),
                run_manifest_path=tmp_path / "run_manifest.json",
            ),
        )
    run_learned_training(
        prepared,
        object(),
        learned_reconstruction,
        validated_training_runner=fake_validated,
    )
    assert callable(captured["reconstruction_validator"])
```

Define the `learned_reconstruction` fixture in this test module from a minimal
`ReconstructionBundle`, the existing `_fixture` artifacts, and a
`GeometryAcceptance` whose metrics and three model hashes match that bundle.

- [ ] **Step 2: Run training tests and confirm RED**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/static_pipeline/test_training.py tests/experiments/learned_quality/test_training.py -q
```

Expected: unexpected keyword/signature failures for `reconstruction_validator`.

- [ ] **Step 3: Extract the strict default and add the guarded callback**

Move the decision checks out of `_validate_manifest` so it remains structural. Re-measure exactly once in `_validate_inputs`, hash all three model text files, and call either the provided validator or `_require_strict_reconstruction`.

```python
ReconstructionValidator = Callable[
    [PreparedTrainingInput, GateDecision, Mapping[str, str]],
    None,
]


def _require_strict_reconstruction(
    prepared: PreparedTrainingInput,
    current_decision: GateDecision,
    model_hashes: Mapping[str, str],
) -> None:
    del model_hashes
    stored = prepared.reconstruction.decision
    if not stored.passed or stored.failures:
        raise ValueError("training requires a passing reconstruction decision")
    if not current_decision.passed:
        failures = ", ".join(current_decision.failures) or "unknown"
        raise ValueError(
            "accepted COLMAP model no longer passes the reconstruction gate: "
            f"{failures}"
        )
```

The learned guarded validator must require: recovery mode selected, non-null acceptance, exact selection digest, exact three file hashes, exact strict failures, exact measured metrics, exact checkpoint fingerprint, and all freshly evaluated output-first checks true. It must raise on any mismatch and must not mutate or replace the measured `GateDecision`.

```python
def _output_first_training_validator(reconstruction: LearnedReconstructionOutput):
    def validate(prepared, current_decision, model_hashes):
        acceptance = reconstruction.acceptance
        if acceptance is None or acceptance.policy_version != "output-first-v1":
            raise ValueError("output-first training requires a guarded acceptance")
        if acceptance.selection_digest != prepared.selection_digest:
            raise ValueError("geometry acceptance selection digest changed")
        if dict(acceptance.model_hashes) != dict(model_hashes):
            raise ValueError("geometry acceptance model hash changed")
        if acceptance.strict_failures != current_decision.failures:
            raise ValueError("geometry acceptance strict failures changed")
        if acceptance.metrics != current_decision.dominant:
            raise ValueError("geometry acceptance metrics changed")
        if not all(evaluate_output_first_checks(current_decision).values()):
            raise ValueError("geometry no longer passes output-first-v1")
    return validate
```

`run_learned_training` passes this callback only when `reconstruction.acceptance.mode == "best_effort"`; otherwise it omits the keyword and therefore uses the strict default.

- [ ] **Step 4: Run training tests and confirm GREEN**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/static_pipeline/test_training.py tests/experiments/learned_quality/test_training.py -q
```

Expected: all selected tests pass and the strict default regression test rejects the known failed decision.

- [ ] **Step 5: Commit the validator milestone**

```powershell
git add backend/static_pipeline/training.py experiments/learned_quality/training.py tests/static_pipeline/test_training.py tests/experiments/learned_quality/test_training.py
git diff --staged
git commit -m "feat: validate guarded geometry before training"
git push
```

---

### Task 5: Restore Drive payloads with one verified remote read

**Files:**
- Modify: `experiments/learned_quality/cache.py`
- Test: `tests/experiments/learned_quality/test_cache.py`
- Test: `tests/experiments/learned_quality/test_milestones.py`

**Interfaces:**
- Produces: `_validated_generation_metadata(generation, kind, fingerprint) -> tuple[dict[str, object], tuple[ManifestEntry, ...]] | None`.
- Produces: `_copy_manifest_payload(generation, destination, entries) -> None`.
- Preserves: `find_generation`, `restore_milestone`, and `restore_colmap` public return/error behavior.

- [ ] **Step 1: Add failing one-read and corruption tests**

```python
def _published_colmap(tmp_path: Path) -> tuple[LearnedCheckpointStore, Path]:
    store = LearnedCheckpointStore(
        tmp_path / "myroom_test_learned_test_cache",
        input_identity="a" * 64,
    )
    attempt_root = tmp_path / "attempt"
    model = write_colmap_text_model(
        attempt_root / "sparse" / "0",
        ("frame_000000.png",),
        (0.2,),
        (4,),
    )
    database = attempt_root / "colmap.db"
    database.write_bytes(b"database")
    attempt = ColmapAttempt(
        root=attempt_root,
        database_path=database,
        model_dirs=(model,),
        colmap_version="COLMAP 3.12.6",
        fingerprint="b" * 64,
    )
    generation = store.publish_colmap(
        attempt,
        fingerprint="b" * 64,
        run_id="run-1",
    )
    return store, generation


def test_restore_reads_every_remote_payload_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, generation = _published_colmap(tmp_path)
    payload = generation / "payload"
    reads: Counter[Path] = Counter()
    real_open = Path.open

    def counted_open(path: Path, *args: object, **kwargs: object):
        candidate = Path(path)
        mode = str(args[0]) if args else str(kwargs.get("mode", "r"))
        if candidate.is_relative_to(payload) and "r" in mode:
            reads[candidate.resolve()] += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    restored = store.restore_colmap(
        "b" * 64,
        destination=tmp_path / "restored",
    )
    assert restored is not None
    assert reads
    assert set(reads.values()) == {1}


@pytest.mark.parametrize("mutation", ["size", "digest", "missing", "unexpected"])
def test_one_pass_restore_rejects_payload_corruption(
    tmp_path: Path,
    mutation: str,
) -> None:
    store, generation = _published_colmap(tmp_path)
    source = generation / "payload" / "attempt" / "colmap.db"
    if mutation == "size":
        source.write_bytes(source.read_bytes() + b"x")
    elif mutation == "digest":
        source.write_bytes(b"changed!")
    elif mutation == "missing":
        source.unlink()
    else:
        (generation / "payload" / "unexpected.bin").write_bytes(b"unexpected")
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="manifest|payload|unexpected"):
        store.restore_colmap("b" * 64, destination=destination)
    assert not destination.exists()
```

Keep and extend existing marker, manifest, upstream, and copy-corruption tests. Add the same single-read assertion for `restore_colmap`.

- [ ] **Step 2: Run cache tests and confirm RED**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_milestones.py -q
```

Expected: the read counter reports multiple reads per payload file.

- [ ] **Step 3: Implement metadata-only validation and streaming copy**

Define a private immutable manifest entry and validate safe paths, unique sorted entries, nonnegative sizes, lowercase SHA-256 digests, manifest digest, success marker, schema, kind, fingerprint, and upstream without opening payload files.

```python
@dataclass(frozen=True)
class _ManifestEntry:
    relative_path: PurePosixPath
    size_bytes: int
    sha256: str


def _copy_manifest_payload(
    generation: Path,
    destination: Path,
    entries: tuple[_ManifestEntry, ...],
) -> None:
    source_root = _regular_directory(generation / "payload", "payload")
    destination.mkdir(parents=False)
    expected_paths = set()
    expected_sources = {
        source_root.joinpath(*entry.relative_path.parts[1:]).resolve(strict=False)
        for entry in entries
    }
    actual_sources = {
        path.resolve(strict=True)
        for path in source_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_sources != expected_sources:
        raise ValueError("checkpoint payload contains missing or unexpected files")
    for entry in entries:
        if entry.relative_path.parts[0] != "payload":
            raise ValueError("manifest entry must be rooted at payload")
        relative = PurePosixPath(*entry.relative_path.parts[1:])
        source = source_root.joinpath(*relative.parts)
        target = destination.joinpath(*relative.parts)
        _regular_file(source, "checkpoint payload")
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        count = 0
        with source.open("rb") as reader, target.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                writer.write(chunk)
                digest.update(chunk)
                count += len(chunk)
        if count != entry.size_bytes or digest.hexdigest() != entry.sha256:
            raise ValueError("checkpoint payload differs from its manifest")
        expected_paths.add(target.resolve(strict=True))
    actual_paths = {
        path.resolve(strict=True)
        for path in destination.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != expected_paths:
        raise ValueError("restored checkpoint contains unexpected files")
```

Change `find_generation` and graph discovery to validate only metadata. `restore_milestone`, `restore_colmap`, and `restore_pretraining` call `_copy_manifest_payload` once and then decode/validate exclusively from the local destination. Remove source `_inventory` calls from restore paths. On mismatch, remove only the fresh local destination; never alter Drive.

- [ ] **Step 4: Run cache tests and confirm GREEN**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_milestones.py -q
```

Expected: corruption cases fail safely and every successfully restored remote payload file is opened exactly once.

- [ ] **Step 5: Commit the one-pass restore milestone**

```powershell
git add experiments/learned_quality/cache.py tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_milestones.py
git diff --staged
git commit -m "perf: restore drive checkpoints in one pass"
git push
```

---

### Task 6: Report best-effort status and generate the fixed A100 notebook

**Files:**
- Modify: `experiments/learned_quality/runner.py`
- Modify: `experiments/learned_quality/reports.py`
- Modify: `experiments/learned_quality/publish.py`
- Modify: `experiments/learned_quality/notebook.py`
- Modify: `scripts/learned_quality_run.py`
- Test: `tests/experiments/learned_quality/test_reports.py`
- Test: `tests/experiments/learned_quality/test_runner.py`
- Test: `tests/experiments/learned_quality/test_publish.py`
- Test: `tests/experiments/learned_quality/test_cli.py`
- Test: `tests/integration/test_learned_quality_notebook_smoke.py`
- Modify (generated): `colab/learned_quality_a100_experiment.ipynb`

**Interfaces:**
- Consumes: `LearnedReconstructionOutput.acceptance`.
- Produces report fields: `geometry_acceptance_mode`, `geometry_policy_version`, `geometry_strict_failures`, `geometry_guarded_metrics`, `geometry_colmap_fingerprint`.
- Produces notebook spec field: `recovery_mode: "round0_output_first_v1"`.
- Produces failure-receipt fields: `geometry_strict_failures` and `geometry_guarded_failures` when guarded geometry stops.

- [ ] **Step 1: Add failing report and notebook tests**

```python
def test_best_effort_acceptance_payload_never_claims_strict_pass() -> None:
    metrics = ModelMetrics(
        model_dir=Path("accepted"),
        registered_names=frozenset({"frame_000000.png"}),
        registered_count=658,
        registered_ratio=658 / 800,
        registered_share=1.0,
        temporal_coverage_s=95.0,
        max_interior_gap_s=9.59,
        start_gap_s=0.0,
        end_gap_s=0.0,
        median_reprojection_error_px=1.217,
        p95_reprojection_error_px=2.194,
        median_track_length=5.0,
        sparse_point_count=65_367,
        valid_names_intrinsics_and_poses=True,
    )
    failures = ("registered_ratio", "interior_gap", "median_reprojection")
    acceptance = GeometryAcceptance(
        policy_version="output-first-v1",
        mode="best_effort",
        selection_digest="a" * 64,
        model_hashes={name: "b" * 64 for name in MODEL_TEXT_FILES},
        strict_failures=failures,
        metrics=metrics,
        checks={"registered_ratio": True},
        colmap_fingerprint="c" * 64,
    )
    reconstruction = SimpleNamespace(
        acceptance=acceptance,
        decision=SimpleNamespace(failures=failures),
    )
    payload = _geometry_acceptance_payload(reconstruction)
    assert payload["geometry_acceptance_mode"] == "best_effort"
    assert payload["geometry_policy_version"] == "output-first-v1"
    assert payload["geometry_strict_failures"] == list(failures)
    assert payload["geometry_colmap_fingerprint"] == "c" * 64
    assert payload["geometry_guarded_metrics"]["registered_count"] == 658


def test_publish_success_marker_labels_best_effort_geometry(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    bundle = make_ready_bundle(
        tmp_path / "bundle",
        run_id="run-1",
        geometry_acceptance={
            "mode": "best_effort",
            "policy_version": "output-first-v1",
            "strict_failures": ["registered_ratio"],
            "colmap_fingerprint": "c" * 64,
        },
    )
    receipt = publish_learned_result(bundle, input_folder, run_id="run-1")
    success = json.loads((receipt.final_path / "_SUCCESS").read_text())
    assert success["geometry_acceptance_mode"] == "best_effort"
    assert success["geometry_policy_version"] == "output-first-v1"


def test_a100_notebook_is_fixed_to_round0_output_first_and_releases_runtime():
    notebook = build_learned_quality_a100_notebook(commit_sha="a" * 40)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "'recovery_mode': 'round0_output_first_v1'" in source
    assert "BEST-EFFORT ROUND 0" in source
    assert "PYTHONUNBUFFERED" in source
    assert "drive.flush_and_unmount()" in source
    assert "runtime.unassign()" in source
```

Add a strict report regression asserting strict results still report `strict` and do not acquire recovery-only claims.

- [ ] **Step 2: Run report/notebook tests and confirm RED**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/experiments/learned_quality/test_reports.py tests/experiments/learned_quality/test_runner.py tests/experiments/learned_quality/test_publish.py tests/experiments/learned_quality/test_cli.py tests/integration/test_learned_quality_notebook_smoke.py -q
```

Expected: missing report keys and missing notebook recovery mode.

- [ ] **Step 3: Add explicit acceptance payloads and notebook text**

```python
def _geometry_acceptance_payload(reconstruction: LearnedReconstructionOutput):
    acceptance = reconstruction.acceptance
    if acceptance is None:
        return {
            "geometry_acceptance_mode": "strict",
            "geometry_policy_version": "strict-v1",
            "geometry_strict_failures": list(reconstruction.decision.failures),
            "geometry_guarded_metrics": None,
            "geometry_colmap_fingerprint": None,
        }
    return {
        "geometry_acceptance_mode": acceptance.mode,
        "geometry_policy_version": acceptance.policy_version,
        "geometry_strict_failures": list(acceptance.strict_failures),
        "geometry_guarded_metrics": {
            "registered_count": acceptance.metrics.registered_count,
            "registered_ratio": acceptance.metrics.registered_ratio,
            "registered_share": acceptance.metrics.registered_share,
            "max_interior_gap_s": acceptance.metrics.max_interior_gap_s,
            "start_gap_s": acceptance.metrics.start_gap_s,
            "end_gap_s": acceptance.metrics.end_gap_s,
            "median_reprojection_error_px": (
                acceptance.metrics.median_reprojection_error_px
            ),
            "p95_reprojection_error_px": (
                acceptance.metrics.p95_reprojection_error_px
            ),
            "median_track_length": acceptance.metrics.median_track_length,
            "sparse_point_count": acceptance.metrics.sparse_point_count,
            "valid_names_intrinsics_and_poses": (
                acceptance.metrics.valid_names_intrinsics_and_poses
            ),
        },
        "geometry_colmap_fingerprint": acceptance.colmap_fingerprint,
    }
```

Merge this payload into `quality_report.json`, `experiment_report.json`, training status, and the owned success metadata used by learned-quality publication. Update `make_ready_bundle` in the tests to accept an optional `geometry_acceptance` mapping and place it in both the experiment report and run manifest. Use `status="best_effort"` only when the acceptance mode is best effort; do not change the process receipt status from `success`.

Extend the failure payload in `scripts/learned_quality_run.py` with:

```python
"geometry_strict_failures": getattr(exc, "strict_failures", None),
"geometry_guarded_failures": getattr(exc, "guarded_failures", None),
```

Add a script test that raises `OutputFirstGeometryError` through a stub runner and
asserts both tuples are serialized as JSON arrays alongside the existing durable
milestone and next-stage fields.

In `build_a100_notebook`, write exactly:

```python
RUN_SPEC = {
    "schema_version": 1,
    "input_folder": folder.as_posix(),
    "recovery_mode": "round0_output_first_v1",
    "publish": {"replace_owned_result": True},
}
```

Update the title and instructions to say this notebook restores the complete Round 0 cache and publishes a transparent best-effort result. Preserve the unbuffered subprocess, required artifact checks, Drive flush/unmount, and runtime release cells.

- [ ] **Step 4: Run report/notebook tests and confirm GREEN**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/experiments/learned_quality/test_reports.py tests/experiments/learned_quality/test_runner.py tests/experiments/learned_quality/test_publish.py tests/experiments/learned_quality/test_cli.py tests/integration/test_learned_quality_notebook_smoke.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit and push the verified code before pinning the notebook**

```powershell
git add experiments/learned_quality/runner.py experiments/learned_quality/reports.py experiments/learned_quality/publish.py experiments/learned_quality/notebook.py scripts/learned_quality_run.py tests/experiments/learned_quality/test_reports.py tests/experiments/learned_quality/test_runner.py tests/experiments/learned_quality/test_publish.py tests/experiments/learned_quality/test_cli.py tests/integration/test_learned_quality_notebook_smoke.py
git diff --staged
git commit -m "feat: publish round0 best-effort recovery"
git push
```

- [ ] **Step 6: Generate the notebook pinned to the verified code commit**

```powershell
$commit = git rev-parse HEAD
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m experiments.learned_quality.notebook --commit $commit --output colab/learned_quality_a100_experiment.ipynb
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/integration/test_learned_quality_notebook_smoke.py -q
git add colab/learned_quality_a100_experiment.ipynb
git diff --staged --stat
git diff --staged -- colab/learned_quality_a100_experiment.ipynb
git commit -m "chore: pin round0 recovery notebook"
git push
```

Expected: generated notebook test passes; the notebook diff contains the pinned source commit, the fixed recovery mode, live output, and unconditional runtime release.

---

### Task 7: Run the complete regression and hand off the exact Colab path

**Files:**
- Verify: all modified files
- Verify unchanged behavior: legacy notebook and static pipeline tests

**Interfaces:**
- Produces: one pushed, commit-pinned A100 recovery notebook ready for `myroom_test`.

- [ ] **Step 1: Run focused learned-quality and static-training regression**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/experiments/learned_quality tests/static_pipeline/test_training.py -q
```

Expected: all tests pass with no skips introduced for the new recovery behavior.

- [ ] **Step 2: Run legacy notebook/static pipeline regression**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m pytest tests/notebooks tests/static_pipeline -q
```

Expected: all tests pass; no legacy notebook snapshots change.

- [ ] **Step 3: Run formatting and repository-defined checks**

```powershell
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m ruff check backend experiments tests
& 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -m ruff format --check backend experiments tests
```

Expected: both commands exit zero. If the repository does not include Ruff in the pinned environment, record that exact unavailable-command output and rely on pytest plus `python -m compileall backend experiments`.

- [ ] **Step 4: Verify branch, commit, generated notebook pin, and clean worktree**

```powershell
git status --short --branch
git log -5 --oneline
git grep -n "round0_output_first_v1" -- experiments colab tests
$pinned = & 'C:\Users\TAHA\AppData\Local\Programs\Python\Python312\python.exe' -c "import json; n=json.load(open('colab/learned_quality_a100_experiment.ipynb', encoding='utf-8')); print(next(line.split('=')[1].strip().strip(chr(39)) for c in n['cells'] for line in c.get('source', []) if line.startswith('SOURCE_COMMIT =')))"
git cat-file -e "$pinned^{commit}"
```

Expected: branch is `feature/learned-quality-a100`, worktree is clean, all new behavior is confined to the isolated learned-quality path plus the default-off training seam, and the pinned commit exists.

- [ ] **Step 5: Commit any final test-only correction and push**

Only if Step 1–4 required a correction:

```powershell
git status --short
git add -u
git diff --staged
git commit -m "test: verify round0 recovery workflow"
git push
```

If no correction was required, do not create an empty commit.

- [ ] **Step 6: Hand off the exact run path and expected behavior**

Provide the GitHub/Colab notebook path `colab/learned_quality_a100_experiment.ipynb`, input text `myroom_test`, and state the expected visible sequence: selection/masks lineage cache hits, cached COLMAP restore, best-effort acceptance banner, final pretraining, live Gaussian iteration output, publication to `MyDrive/myroom_test_learned_test_result`, Drive flush, and runtime release. Explicitly warn that a missing/corrupt checkpoint or a failed guarded measurement stops safely before training.

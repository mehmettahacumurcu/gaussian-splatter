# Learned Evidence Resume and Track Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify raw COLMAP tracks before learned inference and resume the learned-quality A100 pipeline from verified Drive milestones instead of repeating completed semantic, motion, mask, geometry, or final-pretraining work.

**Architecture:** A focused track-qualification module turns raw COLMAP text into one canonical, audited `StaticTrack` contract. The existing learned checkpoint store is extended with lossless typed milestone snapshots whose manifests reference an upstream fingerprint graph. Runtime orchestration restores the newest valid graph, publishes each completed expensive boundary, reports optical-flow subprogress, and requires a CPU-generated audit receipt before an A100 run.

**Tech Stack:** Python 3.12, dataclasses, pathlib, JSON/JSONL, SHA-256, Pillow, NumPy, pytest, Google Colab DriveFS, nbformat.

## Global Constraints

- Apply only to the learned-quality A100 experiment; do not change the legacy notebook pipeline.
- Keep successful results at `<data_folder_name>_learned_test_result/` and owned cache data at `<data_folder_name>_learned_test_cache/`.
- Preserve automatic Drive flush/unmount and runtime release on success or failure.
- Do not clamp COLMAP coordinates and do not introduce lossy image, depth, flow, or mask encoding.
- Retain a static track only when its geometry is finite and at least three finite, unique, in-bounds observations survive.
- Fail before learned inference when more than one percent of joined COLMAP observations are invalid or no qualified tracks remain.
- Never train or run learned inference directly from the Drive mount; restore into a fresh local `/content` root.
- Never reuse a partial, corrupt, incompatible, symlinked, path-traversing, or unowned checkpoint.
- Preserve reuse of the existing verified room COLMAP generation through an exact, allow-listed legacy-fingerprint migration; never rerun COLMAP merely because downstream runtime/track code changed.
- A stage begins only after the preceding milestone has a verified Drive success marker.
- Use TDD and commit/push every independently verified task on `feature/learned-quality-a100`.

## File Structure

- Create `experiments/learned_quality/tracks.py` for COLMAP parsing, qualification, audit reports, and track policy.
- Create `experiments/learned_quality/milestones.py` for milestone state types, fingerprints, dependency graphs, restore assembly, and scoped producer paths.
- Create `experiments/learned_quality/audit.py` and `scripts/learned_quality_cache_audit.py` for the CPU-only receipt workflow.
- Modify `experiments/learned_quality/cache.py` to publish and restore generic typed milestones while leaving the COLMAP checkpoint contract byte-compatible.
- Modify `experiments/learned_quality/runtime.py` to consume qualified tracks and restore/publish learned evidence boundaries.
- Modify `experiments/learned_quality/flow.py` and `model_adapters.py` to emit pair inference, residual evaluation, and publication progress without changing numerical outputs.
- Modify `experiments/learned_quality/runner.py`, `backend/static_pipeline/runner.py`, and `progress.py` for selection restore/save, resume orchestration, failure receipts, and progress events.
- Modify `experiments/learned_quality/notebook.py`; create the generated CPU audit notebook; regenerate the A100 notebook only after runtime verification.
- Add focused tests under `tests/experiments/learned_quality`, `tests/static_pipeline`, and `tests/integration`.

---

### Task 1: Producer-Qualified COLMAP Tracks and Deterministic Audit

**Files:**
- Create: `experiments/learned_quality/tracks.py`
- Create: `tests/experiments/learned_quality/test_tracks.py`
- Modify: `experiments/learned_quality/runtime.py`
- Modify: `tests/experiments/learned_quality/test_runtime.py`

**Interfaces:**
- Produces: `TrackQualificationPolicy`, `TrackAuditReport`, `QualifiedStaticTracks`, `TrackQualificationError`, and `qualify_colmap_static_tracks`.
- Changes: `_rigid_scene(..., static_tracks: tuple[StaticTrack, ...])` consumes already-qualified tracks instead of parsing raw COLMAP text.

- [ ] **Step 1: Write failing boundary and non-finite observation tests**

Create real two-line COLMAP `images.txt` and `points3D.txt` fixtures plus 8x6 PNG frames. Cover `NaN`, infinity, negative values, `x == width`, `y == height`, and values immediately inside each boundary:

```python
@pytest.mark.parametrize(
    ("x", "y", "reason"),
    (
        (float("nan"), 2.0, "non_finite"),
        (float("inf"), 2.0, "non_finite"),
        (-0.01, 2.0, "out_of_bounds"),
        (8.0, 2.0, "out_of_bounds"),
        (2.0, 6.0, "out_of_bounds"),
    ),
)
def test_qualification_discards_invalid_observation_without_clamping(
    tmp_path, x, y, reason
):
    model, frames = colmap_track_fixture(tmp_path, observations=((x, y), (1, 1), (2, 2), (3, 3)))
    result = qualify_colmap_static_tracks(
        model,
        frames,
        tmp_path / "track_audit.json",
        policy=TrackQualificationPolicy(),
    )
    assert len(result.tracks) == 1
    assert tuple((item.x, item.y) for item in result.tracks[0].observations) == (
        (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)
    )
    assert result.report.rejected_observations_by_reason[reason] == 1


def test_qualification_accepts_values_immediately_inside_bounds(tmp_path):
    result = qualify_fixture(tmp_path, ((0.0, 0.0), (7.999999, 5.999999), (4.0, 3.0)))
    assert len(result.tracks) == 1
```

- [ ] **Step 2: Run the red tests**

Run: `python -m pytest tests/experiments/learned_quality/test_tracks.py -q`

Expected: collection fails because `experiments.learned_quality.tracks` does not exist.

- [ ] **Step 3: Implement immutable qualification types and parser**

Implement these exact public interfaces:

```python
TRACK_AUDIT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrackQualificationPolicy:
    minimum_observations: int = 3
    maximum_invalid_fraction: float = 0.01
    maximum_examples: int = 32


@dataclass(frozen=True)
class TrackAuditReport:
    schema_version: int
    raw_track_count: int
    accepted_track_count: int
    rejected_track_count: int
    raw_observation_count: int
    accepted_observation_count: int
    rejected_observation_count: int
    rejected_tracks_by_reason: Mapping[str, int]
    rejected_observations_by_reason: Mapping[str, int]
    examples: tuple[Mapping[str, object], ...]
    accepted_tracks_sha256: str


@dataclass(frozen=True)
class QualifiedStaticTracks:
    tracks: tuple[StaticTrack, ...]
    report: TrackAuditReport
    audit_path: Path
    audit_sha256: str


class TrackQualificationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        report: TrackAuditReport,
        audit_path: Path,
    ) -> None:
        self.report = report
        self.audit_path = audit_path
        super().__init__(message)


def qualify_colmap_static_tracks(
    model_dir: Path,
    frames: tuple[FrameArtifact, ...],
    audit_path: Path,
    *,
    policy: TrackQualificationPolicy,
) -> QualifiedStaticTracks:
    """Return producer-qualified static tracks and a strict persisted audit."""
```

Parse COLMAP text without trusting image names, numeric values, point-index joins, or duplicate frame observations. Read canonical dimensions from the selected image files through Pillow. Discard invalid observations, discard invalid-geometry tracks, require three surviving observations, sort tracks by id and observations by canonical frame order, serialize strict JSON with `allow_nan=False`, and hash the accepted payload.

- [ ] **Step 4: Add systematic-mismatch and track-geometry tests**

Test duplicate frame observations, fewer than three survivors, non-finite XYZ, negative/non-finite reprojection error, missing images, point-id mismatch, zero qualified tracks, and invalid observation fraction greater than `0.01`. Assert `TrackQualificationError` still writes a complete audit before raising.

```python
def test_systematic_coordinate_mismatch_writes_audit_then_fails(tmp_path):
    model, frames = mismatch_fixture(tmp_path, invalid=2, total=100)
    with pytest.raises(TrackQualificationError, match="systematic") as caught:
        qualify_colmap_static_tracks(
            model,
            frames,
            tmp_path / "audit.json",
            policy=TrackQualificationPolicy(maximum_invalid_fraction=0.01),
        )
    assert caught.value.audit_path.is_file()
    assert caught.value.report.rejected_observation_count == 2
```

- [ ] **Step 5: Move producer responsibility out of runtime**

Delete `_image_point_rows` and `_static_tracks` from `runtime.py`. Immediately after the classical COLMAP model is restored or produced, call `qualify_colmap_static_tracks` and only then continue to metric depth, semantic masks, or optical flow. Change `_rigid_scene` to accept `static_tracks` and never parse raw tracks itself:

```python
def _rigid_scene(
    frames: tuple[FrameArtifact, ...],
    model_dir: Path,
    depths: tuple[tuple[Path, str], ...],
    *,
    static_tracks: tuple[StaticTrack, ...],
) -> RigidSceneEvidence:
    """Bind already-qualified static tracks to camera and depth evidence."""
```

Add a runtime-order test whose detector/model factories raise if invoked; a bad track must fail first with a published audit.

- [ ] **Step 6: Verify and commit Task 1**

Run:

```text
python -m pytest tests/experiments/learned_quality/test_tracks.py tests/experiments/learned_quality/test_runtime.py tests/experiments/learned_quality/test_flow.py tests/experiments/learned_quality/test_masks.py tests/experiments/learned_quality/test_photometric.py -q
python -m ruff check experiments/learned_quality/tracks.py experiments/learned_quality/runtime.py tests/experiments/learned_quality/test_tracks.py tests/experiments/learned_quality/test_runtime.py
python -m black --check experiments/learned_quality/tracks.py experiments/learned_quality/runtime.py tests/experiments/learned_quality/test_tracks.py tests/experiments/learned_quality/test_runtime.py
```

Expected: all pass. Commit and push: `fix: qualify COLMAP tracks before learned inference`.

---

### Task 2: Generic Milestone Store and Dependency-Graph Validation

**Files:**
- Create: `experiments/learned_quality/milestones.py`
- Create: `tests/experiments/learned_quality/test_milestones.py`
- Modify: `experiments/learned_quality/cache.py`
- Modify: `tests/experiments/learned_quality/test_cache.py`

**Interfaces:**
- Preserves: existing `checkpoint_fingerprint(CheckpointKind.COLMAP, ...)` bytes and current COLMAP generation lookup.
- Produces: new milestone kinds, `MilestoneInputs`, `MilestoneRef`, `MilestoneState`, `milestone_fingerprint`, `publish_milestone`, `restore_milestone`, and `validate_milestone_graph`.
- Produces: `compatible_colmap_fingerprints` to probe the current raw-COLMAP producer fingerprint followed by allow-listed legacy fingerprints.

- [ ] **Step 1: Freeze the existing COLMAP fingerprint with a regression test**

Use the room-cache-compatible fixture and assert the current digest remains exact across this task:

```python
def test_colmap_fingerprint_remains_backward_compatible():
    assert checkpoint_fingerprint(CheckpointKind.COLMAP, pinned_colmap_inputs()) == (
        "684800f437e38a019e9d96b6928b6a5d3995b93cc6c925eccd620a926756ec82"
    )
```

Run the test before editing and retain it permanently.

The current fingerprint includes `runtime.py`, so Task 1 necessarily changes
its producer digest even though raw COLMAP output is unchanged. Add a second
fixture that reads the three producer files at commit `1684964` through
`git show`, computes the frozen legacy producer digest, and proves the complete
room key above. Do not hardcode a cache hit based only on folder name.

- [ ] **Step 2: Write failing milestone fingerprint and graph tests**

Test deterministic sorted upstream mappings, transitive invalidation, unrelated sibling reuse, cycle rejection, missing upstream rejection, wrong-kind references, and corrupt-success-marker fallback:

```python
def test_semantic_change_does_not_invalidate_motion_sibling():
    base = milestone_ref(CheckpointKind.BASE_EVIDENCE, "a" * 64)
    semantic_v1 = fingerprint_semantic(base, producer="b" * 64)
    semantic_v2 = fingerprint_semantic(base, producer="c" * 64)
    motion_v1 = fingerprint_motion(base, producer="d" * 64)
    assert semantic_v1 != semantic_v2
    assert motion_v1 == fingerprint_motion(base, producer="d" * 64)
```

- [ ] **Step 3: Add milestone value types and scoped fingerprints**

Extend `CheckpointKind` with `SELECTION`, `BASE_EVIDENCE`, `SEMANTIC`, `MOTION`, `MASKS`, and `GEOMETRY`; leave `COLMAP` behavior unchanged. Add in `milestones.py`:

```python
@dataclass(frozen=True)
class MilestoneRef:
    kind: CheckpointKind
    fingerprint: str


@dataclass(frozen=True)
class MilestoneInputs:
    source_digest: str
    selection_digest: str
    settings: Mapping[str, object]
    model_manifest_sha256: str
    tool_versions: Mapping[str, str]
    producer_code_sha256: str
    upstream: Mapping[CheckpointKind, str]


@dataclass(frozen=True)
class MilestoneState:
    ref: MilestoneRef
    upstream: Mapping[CheckpointKind, str]
    value: object
    artifact_roots: Mapping[str, Path]


def milestone_fingerprint(kind: CheckpointKind, inputs: MilestoneInputs) -> str:
    """Hash canonical milestone inputs and their sorted upstream map."""
```

Reject `COLMAP` in `milestone_fingerprint`; it remains on the legacy compatible function.

Add:

```python
def compatible_colmap_fingerprints(
    current: CheckpointInputs,
    *,
    legacy_producer_digests: tuple[str, ...],
) -> tuple[str, ...]:
    """Return current then allow-listed legacy fingerprints without duplicates."""
```

The allow-list contains only the verified producer digest derived from commit
`1684964`. On a legacy hit, restore and run the full readable-model,
registration, selected-frame, and rigid-scene validation, then publish the same
validated attempt under the current fingerprint. Write the new success marker
before considering legacy migration complete. Never accept a legacy producer
digest that is not compiled into the runtime and covered by the frozen-key test.

- [ ] **Step 4: Write failing portable milestone publication tests**

Publish a semantic milestone and restore it into a fresh local root. Assert all paths are local, hashes match, `_SUCCESS.json` is written last, no rename is required, and only direct semantic artifacts are stored. Add rejection tests for absolute state paths, `..`, backslashes, symlinks, changed bytes, duplicate inventory entries, unknown type tags, and invalid upstream maps.

- [ ] **Step 5: Implement generic state publication and restoration**

Add exact store methods with these signatures:

```text
LearnedCheckpointStore.publish_milestone(state: MilestoneState, *, run_id: str) -> Path
LearnedCheckpointStore.restore_milestone(ref: MilestoneRef, *, destination: Path, source_inventory: object) -> MilestoneState | None
LearnedCheckpointStore.validate_milestone_graph(root: MilestoneRef) -> tuple[MilestoneRef, ...]
```

Reuse the allow-listed checkpoint codec and generation success-marker protocol. Add only the dataclasses required by stage state to `_checkpoint_types`. Store the upstream map in both `manifest.json` and `_SUCCESS.json`; reread both before reuse. Keep referenced generations during cleanup and remove only unreferenced, pipeline-owned generations.

- [ ] **Step 6: Verify corruption and reference-aware cleanup**

Create two semantic generations and a masks generation referencing the first. Publish the second, run cleanup, and assert the first remains because masks references it. Remove masks, run cleanup, and assert only the newest semantic generation remains. Verify partial Drive writes never become a hit.

- [ ] **Step 7: Verify and commit Task 2**

Run:

```text
python -m pytest tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_milestones.py tests/static_pipeline/test_stage_cache.py -q
python -m ruff check experiments/learned_quality/cache.py experiments/learned_quality/milestones.py tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_milestones.py
python -m black --check experiments/learned_quality/cache.py experiments/learned_quality/milestones.py tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_milestones.py
```

Expected: all pass and the backward-compatible COLMAP fingerprint remains exact. Commit and push: `feat: add learned milestone checkpoint graph`.

---

### Task 3: Selection, Base, Semantic, Motion, and Mask Resume

**Files:**
- Modify: `backend/static_pipeline/runner.py`
- Modify: `experiments/learned_quality/milestones.py`
- Modify: `experiments/learned_quality/runtime.py`
- Modify: `experiments/learned_quality/runner.py`
- Modify: `tests/static_pipeline/test_runner.py`
- Modify: `tests/experiments/learned_quality/test_milestones.py`
- Modify: `tests/experiments/learned_quality/test_runtime.py`
- Modify: `tests/experiments/learned_quality/test_runner.py`

**Interfaces:**
- Produces: optional selection restore/save hooks in `RunnerServices`.
- Produces: `BaseEvidenceState`, `SemanticMilestoneState`, `MotionMilestoneState`, `MasksMilestoneState`, and `RestoredEvidenceGraph`.
- Changes: `_run_evidence_cycle` accepts a `MilestoneSession` and resumes stage-by-stage.

- [ ] **Step 1: Write failing optional selection-hook tests**

Assert a selection hit skips source copy and selection but still reconstructs, a miss preserves current legacy order, save occurs before reconstruction, and hook failures publish diagnostics. Default `None` hooks must leave every legacy test unchanged.

```python
assert calls_on_hit == [
    "discover", "preflight", "restore_pretraining", "restore_selection",
    "reconstruct", "save_pretraining", "train", "publish",
]
```

- [ ] **Step 2: Add selection hooks without changing legacy services**

Append defaulted fields to `RunnerServices`:

```python
restore_selection: Callable[..., object | None] | None = None
save_selection: Callable[..., object] | None = None
```

Probe complete pretraining first, then selection. On a selection miss run the existing copy/select path and publish selection before reconstruction. Never call these hooks for legacy services.

- [ ] **Step 3: Define exact stage-state contracts**

Add:

```python
@dataclass(frozen=True)
class BaseEvidenceState:
    anchors: AnchorInferenceResult
    depths: tuple[tuple[Path, str], ...]
    sky: tuple[FrameArtifact, ...]
    scene: RigidSceneEvidence
    track_audit: QualifiedStaticTracks
    colmap_ref: str


@dataclass(frozen=True)
class SemanticMilestoneState:
    semantic: SemanticEvidence


@dataclass(frozen=True)
class MotionMilestoneState:
    motion: MotionEvidence


@dataclass(frozen=True)
class MasksMilestoneState:
    masks: MaskFusionEvidence


@dataclass(frozen=True)
class RestoredEvidenceGraph:
    base: BaseEvidenceState
    semantic: SemanticEvidence
    motion: MotionEvidence
    masks: MaskFusionEvidence | None
    refs: Mapping[CheckpointKind, MilestoneRef]
```

Register these types with the static checkpoint allow-list. Each state owns only its direct artifact roots.

- [ ] **Step 4: Write stage-resume integration tests**

Use tiny real artifact files and mocked model functions. Parameterize failures immediately after base, semantic, motion, and masks publication. For every rerun, assert only the first missing stage and later stages execute:

```python
@pytest.mark.parametrize(
    ("durable_kind", "expected_calls"),
    (
        (CheckpointKind.BASE_EVIDENCE, ("semantic", "motion", "masks")),
        (CheckpointKind.SEMANTIC, ("motion", "masks")),
        (CheckpointKind.MOTION, ("masks",)),
        (CheckpointKind.MASKS, ()),
    ),
)
def test_evidence_cycle_resumes_from_newest_valid_graph(
    durable_kind,
    expected_calls,
    milestone_fixture,
):
    session, selection, expected = milestone_fixture.prime_through(durable_kind)
    milestone_fixture.calls.clear()
    actual = milestone_fixture.run(selection, session=session)
    assert tuple(milestone_fixture.calls) == expected_calls
    assert milestone_fixture.artifact_bytes(actual) == expected
```

Implement the parameterized body by priming the fake store through
`durable_kind`, clearing the call recorder, running `_run_evidence_cycle`, and
asserting `tuple(calls) == expected_calls` plus byte-identical restored outputs.

Change one semantic producer file and assert semantic plus masks rerun while reusable motion remains a hit.

- [ ] **Step 5: Implement `MilestoneSession` and evidence-cycle restore**

Implement a learned-only coordinator with these exact methods:

```text
MilestoneSession.restore(kind: CheckpointKind, fingerprint: str, destination: Path) -> object | None
MilestoneSession.publish(kind: CheckpointKind, value: object, artifact_roots: Mapping[str, Path], upstream: Mapping[CheckpointKind, str]) -> MilestoneRef
MilestoneSession.cache_event(action: str, kind: CheckpointKind, detail: str) -> None
```

Refactor `_run_evidence_cycle` into the existing algorithmic sequence with restore/publish boundaries. Keep every existing policy, batch size, model release, numerical function, and file format unchanged. Publish and validate each milestone before entering the next expensive stage.

- [ ] **Step 6: Preserve diagnostics and newest durable milestone**

When a stage fails, attach `durable_milestone_kind`, `durable_milestone_fingerprint`, and `next_stage_id` to the exception. Add these to the receipt and `quality_report.json`. Assert a motion hit followed by mask failure reports motion as the restart point.

- [ ] **Step 7: Verify and commit Task 3**

Run:

```text
python -m pytest tests/static_pipeline/test_runner.py tests/experiments/learned_quality/test_milestones.py tests/experiments/learned_quality/test_runtime.py tests/experiments/learned_quality/test_runner.py -q
python -m pytest tests/experiments/learned_quality/test_flow.py tests/experiments/learned_quality/test_masks.py tests/experiments/learned_quality/test_segmentation.py -q
python -m ruff check backend/static_pipeline/runner.py experiments/learned_quality/milestones.py experiments/learned_quality/runtime.py experiments/learned_quality/runner.py tests/static_pipeline/test_runner.py tests/experiments/learned_quality/test_milestones.py tests/experiments/learned_quality/test_runtime.py tests/experiments/learned_quality/test_runner.py
python -m black --check backend/static_pipeline/runner.py experiments/learned_quality/milestones.py experiments/learned_quality/runtime.py experiments/learned_quality/runner.py tests/static_pipeline/test_runner.py tests/experiments/learned_quality/test_milestones.py tests/experiments/learned_quality/test_runtime.py tests/experiments/learned_quality/test_runner.py
```

Expected: all pass. Commit and push: `feat: resume learned evidence milestones`.

---

### Task 4: Geometry and Final-Pretraining Milestone Assembly

**Files:**
- Modify: `experiments/learned_quality/milestones.py`
- Modify: `experiments/learned_quality/runtime.py`
- Modify: `experiments/learned_quality/cache.py`
- Modify: `tests/experiments/learned_quality/test_milestones.py`
- Modify: `tests/experiments/learned_quality/test_runtime.py`
- Modify: `tests/experiments/learned_quality/test_cache.py`

**Interfaces:**
- Produces: `GeometryMilestoneState`, `FinalPretrainingState`, and `assemble_learned_reconstruction`.
- Migrates: complete-pretraining storage from a cumulative duplicate snapshot to a terminal state referencing the validated milestone graph.

- [ ] **Step 1: Write geometry hit and backfill-branch tests**

Assert a geometry hit skips both classical and hybrid candidate runners. If round zero requests backfill, assert the new selection digest creates a separate selection/base/semantic/motion/masks graph, the original graph remains valid, and round one geometry references only the backfilled graph.

- [ ] **Step 2: Define terminal state contracts**

```python
@dataclass(frozen=True)
class GeometryMilestoneState:
    bundle: ReconstructionBundle
    frames_dir: Path
    geometry_candidates: tuple[GeometryCandidateReport, ...]


@dataclass(frozen=True)
class FinalPretrainingState:
    photometric: object
    depth: object
    dense_seeds: object
    final_stage_records: tuple[StageRecord, ...]
    model_manifest_path: Path


def assemble_learned_reconstruction(
    *,
    geometry: GeometryMilestoneState,
    base: BaseEvidenceState,
    semantic: SemanticMilestoneState,
    motion: MotionMilestoneState,
    masks: MasksMilestoneState,
    final: FinalPretrainingState | None,
) -> LearnedReconstructionOutput:
    """Reassemble the existing service contract from restored milestones."""
```

- [ ] **Step 3: Write graph-assembly equality tests**

Build a normal in-memory `LearnedReconstructionOutput`, split it into milestone states, serialize/restore each into different local roots, assemble it, and assert dataclass equality plus all path containment and file hashes. Ensure no pretraining generation duplicates semantic, motion, or masks files.

- [ ] **Step 4: Add geometry restore/publish and final assembly**

Probe geometry after masks. If missing, run the current comparison unchanged and publish it before photometric validation. Publish `FinalPretrainingState` only after photometric validation, final-pose depth, dense-seed fusion, and prepared-training validation pass. Complete-pretraining restore must restore the dependency graph, assemble the exact existing output contract, and skip directly to training.

- [ ] **Step 5: Implement reference-aware migration behavior**

Continue accepting existing version-1 complete-pretraining generations when valid. Publish new final states with milestone schema version 2. Never rewrite or delete the verified COLMAP generation. Ignore an incompatible old pretraining generation as a visible cache miss.

- [ ] **Step 6: Verify and commit Task 4**

Run:

```text
python -m pytest tests/experiments/learned_quality/test_milestones.py tests/experiments/learned_quality/test_runtime.py tests/experiments/learned_quality/test_cache.py tests/experiments/learned_quality/test_training.py -q
python -m ruff check experiments/learned_quality/milestones.py experiments/learned_quality/runtime.py experiments/learned_quality/cache.py tests/experiments/learned_quality/test_milestones.py tests/experiments/learned_quality/test_runtime.py tests/experiments/learned_quality/test_cache.py
python -m black --check experiments/learned_quality/milestones.py experiments/learned_quality/runtime.py experiments/learned_quality/cache.py tests/experiments/learned_quality/test_milestones.py tests/experiments/learned_quality/test_runtime.py tests/experiments/learned_quality/test_cache.py
```

Expected: all pass. Commit and push: `feat: checkpoint learned geometry and final preprocessing`.

---

### Task 5: Optical-Flow Substage Progress and Bounded Observability

**Files:**
- Modify: `backend/static_pipeline/progress.py`
- Modify: `experiments/learned_quality/flow.py`
- Modify: `experiments/learned_quality/model_adapters.py`
- Modify: `tests/static_pipeline/test_progress.py`
- Modify: `tests/experiments/learned_quality/test_flow.py`
- Modify: `tests/experiments/learned_quality/test_model_adapters.py`

**Interfaces:**
- Produces: `StageReporter.progress` and `FlowProgressCallback`.
- Changes: optional callback parameters only; existing callers and numerical return values remain compatible.

- [ ] **Step 1: Write flushed structured-progress tests**

```python
reporter.progress(
    "optical_flow",
    substage="inference",
    completed=590,
    total=800,
    details={"batch_size": 2, "rss_bytes": 89_000_000_000},
)
```

Assert immediate console flush, strict JSONL, stable fields, percentage derived only from integer counts, and rejection of invalid/non-finite details.

- [ ] **Step 2: Implement reporter progress events**

Add:

```python
def progress(
    self,
    stage_id: str,
    *,
    substage: str,
    completed: int,
    total: int,
    details: Mapping[str, object] | None = None,
) -> None:
    """Emit one flushed, strict, monotonic stage-progress event."""
```

Format human output as `[OPTICAL FLOW] inference 590/800 - 73.8%`. Include elapsed time from the active stage and never invent ETA without measured rates.

- [ ] **Step 3: Write callback-order tests for all optical substages**

With four tiny pairs, assert callbacks for `inference`, `prediction_validation`, `residual_evaluation`, `motion_evaluation`, and `artifact_publication` are monotonic and end exactly at total. Snapshot all numerical arrays before and after adding callbacks and assert byte equality.

- [ ] **Step 4: Thread optional callbacks through flow and adapter**

Define:

```python
FlowProgressCallback = Callable[[str, int, int, Mapping[str, object]], None]
```

Add optional `progress: FlowProgressCallback | None = None` to `run_motion_evidence` and `SeaRaftTorchAdapter.infer_bidirectional`. Emit after completed batches/pairs/frames, never from inside a partially committed operation. Report process RSS via `resource` on Linux and CUDA allocated/reserved bytes when Torch exposes them.

- [ ] **Step 5: Verify and commit Task 5**

Run:

```text
python -m pytest tests/static_pipeline/test_progress.py tests/experiments/learned_quality/test_flow.py tests/experiments/learned_quality/test_model_adapters.py -q
python -m ruff check backend/static_pipeline/progress.py experiments/learned_quality/flow.py experiments/learned_quality/model_adapters.py tests/static_pipeline/test_progress.py tests/experiments/learned_quality/test_flow.py tests/experiments/learned_quality/test_model_adapters.py
python -m black --check backend/static_pipeline/progress.py experiments/learned_quality/flow.py experiments/learned_quality/model_adapters.py tests/static_pipeline/test_progress.py tests/experiments/learned_quality/test_flow.py tests/experiments/learned_quality/test_model_adapters.py
```

Expected: all pass. Commit and push: `feat: report optical flow pair progress`.

---

### Task 6: CPU-Only Audit Receipt and A100 Enforcement

**Files:**
- Create: `experiments/learned_quality/audit.py`
- Create: `scripts/learned_quality_cache_audit.py`
- Create: `tests/experiments/learned_quality/test_audit.py`
- Modify: `experiments/learned_quality/runner.py`
- Modify: `tests/experiments/learned_quality/test_runner.py`

**Interfaces:**
- Produces: `AuditInputs`, `AuditReceipt`, `audit_producer_digest`, `run_cache_audit`, `publish_audit_receipt`, and `require_audit_receipt`.
- The audit path is `<cache>/audits/<fingerprint>/_SUCCESS.json`.

- [ ] **Step 1: Write CPU-only audit receipt tests**

Assert the audit imports no Torch/Transformers/learned adapters, restores the existing COLMAP checkpoint, qualifies tracks, validates checkpoint ownership and model manifest, publishes a strict receipt, and never invokes CUDA or learned model factories.

```python
@dataclass(frozen=True)
class AuditReceipt:
    schema_version: int
    input_digest: str
    colmap_fingerprint: str
    qualification_policy_sha256: str
    audit_producer_code_sha256: str
    track_audit_sha256: str
    passed: bool
```

- [ ] **Step 2: Write stale and corrupt receipt tests**

Change input bytes, COLMAP fingerprint, policy, audit producer code, track audit, success marker, or ownership identity and assert immediate rejection. A training-only source change must retain compatibility.

- [ ] **Step 3: Implement the audit module and CLI**

Add:

```python
def run_cache_audit(
    *,
    input_path: Path,
    cache_root: Path,
    local_root: Path,
    policy: TrackQualificationPolicy,
) -> AuditReceipt:
    """Recreate/restore selection, restore COLMAP, qualify tracks, and publish a receipt."""


def require_audit_receipt(
    cache_root: Path,
    *,
    expected: AuditInputs,
) -> AuditReceipt:
    """Return the exact compatible passing receipt or raise before A100 work."""
```

The CLI accepts `--spec`, mounts no Drive itself, prints each audit boundary,
writes `/content/learned_audit_result.json`, and exits nonzero on any mismatch.
It first restores a compatible selection milestone. When none exists—as with the
current room cache—it copies the video locally, runs the deterministic CPU frame
selection, publishes the new selection milestone, and then proves that the
resulting digest resolves the existing COLMAP checkpoint. This makes the first
CPU audit also remove the seven-minute frame-selection cost from the next A100
run.

- [ ] **Step 4: Enforce the receipt before learned model preflight**

In `run_learned_quality_notebook`, validate the receipt after source inventory/runtime checks but before `learned_model_preflight`, source copy, or learned model loading. Failure receipts must say to run the CPU audit notebook and must still flush Drive/release runtime.

- [ ] **Step 5: Verify and commit Task 6**

Run:

```text
python -m pytest tests/experiments/learned_quality/test_audit.py tests/experiments/learned_quality/test_runner.py -q
python -m ruff check experiments/learned_quality/audit.py scripts/learned_quality_cache_audit.py tests/experiments/learned_quality/test_audit.py experiments/learned_quality/runner.py tests/experiments/learned_quality/test_runner.py
python -m black --check experiments/learned_quality/audit.py scripts/learned_quality_cache_audit.py tests/experiments/learned_quality/test_audit.py experiments/learned_quality/runner.py tests/experiments/learned_quality/test_runner.py
```

Expected: all pass. Commit and push: `feat: gate A100 runs with CPU cache audit`.

---

### Task 7: Generated Audit/A100 Notebooks and Full Verification

**Files:**
- Modify: `experiments/learned_quality/notebook.py`
- Modify: `tests/integration/test_learned_quality_notebook_smoke.py`
- Create: `tests/integration/test_learned_quality_audit_notebook_smoke.py`
- Modify: `colab/README.md`
- Generate: `colab/learned_quality_cache_audit.ipynb`
- Regenerate: `colab/learned_quality_a100_experiment.ipynb`

**Interfaces:**
- Produces: a CPU audit notebook and a separately pinned A100 notebook that accepts only the matching audit receipt.

- [ ] **Step 1: Write failing notebook smoke tests**

The CPU notebook must mount Drive, accept the same MyDrive-relative folder, install only CPU audit dependencies, run the module unbuffered, verify the receipt, flush/unmount, and release the CPU runtime. The A100 notebook must require the receipt, print restored milestone hits, preserve live training output, and retain terminal runtime release.

- [ ] **Step 2: Add deterministic audit notebook generation**

Expose separate generator entry points with these exact signatures:

```text
build_learned_quality_audit_notebook(*, commit_sha: str) -> nbformat.NotebookNode
build_learned_quality_a100_notebook(*, commit_sha: str) -> nbformat.NotebookNode
```

Do not hand-edit notebook JSON. Keep exact 40-character runtime pins in notebook metadata and clone cells.

- [ ] **Step 3: Document the two-step user flow**

Document:

1. Run the CPU audit notebook with `myroom_test`.
2. Confirm `TRACK AUDIT PASSED` and the verified COLMAP fingerprint.
3. Open a fresh A100 notebook and run all.
4. Confirm audit receipt and newest milestone hits before learned inference.
5. Observe pair progress, Gaussian iterations, final Drive publication, and runtime release.

- [ ] **Step 4: Verify runtime before generating notebooks**

Run the complete dependency-light suite plus targeted corruption/resume suites:

```text
python -m pytest tests/experiments/learned_quality tests/static_pipeline tests/integration/test_learned_quality_bootstrap.py -q
python -m ruff check backend/static_pipeline experiments/learned_quality scripts/learned_quality_cache_audit.py tests/static_pipeline tests/experiments/learned_quality
python -m black --check backend/static_pipeline experiments/learned_quality scripts/learned_quality_cache_audit.py tests/static_pipeline tests/experiments/learned_quality
git diff --check
```

Expected: all dependency-light tests pass; only explicit CUDA tests may skip.

- [ ] **Step 5: Commit/push runtime and generate pinned notebooks**

Commit/push all runtime changes. Capture `git rev-parse HEAD` as the immutable runtime pin. Generate both notebooks using that exact SHA. Run both notebook smoke suites and assert the A100 notebook does not clone the later notebook-only commit.

- [ ] **Step 6: Prove COLMAP cache compatibility and legacy isolation**

Run the explicit current/legacy fingerprint suite. Assert the frozen legacy
fixture resolves
`684800f437e38a019e9d96b6928b6a5d3995b93cc6c925eccd620a926756ec82`, a
legacy hit is migrated to the new fingerprint without invoking a COLMAP
subprocess, and changed selection/input/tool settings reject both keys. Assert
no legacy notebook file changed.

- [ ] **Step 7: Commit/push notebooks and verify remote state**

Review `git status` and `git diff --staged`, commit `docs: publish audited resumable A100 notebooks`, push, and verify local HEAD equals `origin/feature/learned-quality-a100`. Record the runtime pin and notebook commit in the final handoff.

---

## Final Acceptance

- The cached room COLMAP generation retains the exact known fingerprint and restores without rerunning COLMAP.
- A CPU-only run produces a strict passing track audit and receipt before A100 credits are used.
- Non-finite/out-of-bounds observations are discarded without clamping; systematic mismatches fail before learned models.
- Static tracks exposed to all consumers contain finite geometry and at least three valid unique observations.
- Failures after semantic, motion, masks, geometry, or final preprocessing retain the preceding verified milestone.
- A rerun restores the newest compatible dependency graph and executes only missing/invalidated stages.
- Semantic-only producer changes do not invalidate a reusable motion sibling; downstream masks invalidate correctly.
- Optical flow prints inference, validation, residual, evaluation, and publication progress with exact counts.
- Stage artifacts remain lossless and every Drive generation is owned, fingerprinted, hash-verified, and success-marked last.
- Complete preprocessing still starts 120,000-iteration maximum-quality Gaussian training with live output.
- Result, diagnostics, legacy pipeline, Drive flush, and runtime-release contracts remain unchanged.
- Checked-in CPU and A100 notebooks are deterministic and pinned to the verified runtime commit.

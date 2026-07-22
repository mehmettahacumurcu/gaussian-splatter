# Automatic Floor-Hole Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bounded A100 diagnostic that adds only verified, multi-view floor-hole seeds to the recognizable legacy training profile and decides whether floor coverage improves without damaging room structure.

**Architecture:** Extend the standalone learned-quality diagnostic infrastructure, not the production runner. Expose a typed motion/sky-filtered depth candidate cloud, implement deterministic floor-plane/hole/seed geometry in a focused module, inject the verified seed slice through an asserting trainer customizer, and run one 5K comparison arm against the staged legacy reference.

**Tech Stack:** Python 3.12, NumPy, SciPy, PyTorch, Pillow, Pydantic, nbformat, pytest, Google Colab A100 High-RAM, Google Drive staging.

## Global Constraints

- Recover only automatically detected missing floor regions.
- Use at most 150,000 floor seeds and require at least 1,000 seeds.
- Require one source camera plus at least two additional agreeing cameras.
- Exclude only confirmed motion and confirmed sky during floor candidate selection.
- Disable semantic training-validity masks and adaptive density.
- Use validated depth loss and the standard legacy density controller.
- Initialize only the appended floor seed slice with opacity logit `-4.0` and scale no larger than one-quarter of the floor grid-cell width.
- Stop after 5,000 iterations; no code path may launch 120,000 iterations.
- Stage Drive inputs once, use local files during training, and never overwrite production results.
- Fail closed when plane, hole, seed, reference, point-order, or diagnostic contracts do not match.

---

## File Structure

- `experiments/learned_quality/depth.py`: produce a validated, source-indexed multi-view depth candidate cloud while preserving the existing global dense-seed API.
- `experiments/learned_quality/floor_recovery.py`: pure geometry, artifact contracts, deterministic floor plane/hole/seed generation, and artifact persistence.
- `experiments/learned_quality/floor_recovery_training.py`: local scene preparation, seed-slice initialization, 5K training, floor metrics, and candidate PLY collection.
- `experiments/learned_quality/floor_recovery_runner.py`: run specification, one-time staging, reference validation, orchestration, decision gates, and atomic diagnostic publication.
- `experiments/learned_quality/floor_recovery_notebook.py`: immutable-SHA A100 High-RAM notebook generator.
- `scripts/learned_quality_floor_recovery_run.py`: subprocess-safe CLI and receipt writer.
- `colab/learned_quality_floor_recovery.ipynb`: generated user entrypoint.
- `tests/experiments/learned_quality/test_floor_recovery.py`: floor geometry and artifact tests.
- `tests/experiments/learned_quality/test_floor_recovery_training.py`: initialization, training-profile, metrics, and fail-closed tests.
- `tests/experiments/learned_quality/test_floor_recovery_runner.py`: staging, decisions, publication, and failure-receipt tests.
- `tests/experiments/learned_quality/test_floor_recovery_cli.py`: CLI and notebook contract tests.
- `tests/experiments/learned_quality/test_depth.py`: supported-depth-cloud regression tests.
- `tests/integration/test_learned_quality_floor_recovery_notebook_smoke.py`: generated notebook smoke test.
- `experiments/learned_quality/README.md` and `colab/README.md`: operator documentation.

---

### Task 1: Expose motion/sky-filtered supported depth candidates

**Files:**
- Modify: `experiments/learned_quality/depth.py`
- Modify: `tests/experiments/learned_quality/test_depth.py`

**Interfaces:**
- Produces: `SupportedDepthCloud` and `build_supported_depth_cloud(...) -> SupportedDepthCloud`.
- Preserves: `validate_depth_and_fuse_seeds(...)` byte-for-byte behavior under its default hard/uncertain mask mode.

- [ ] **Step 1: Write failing tests for candidate provenance and mask selection**

```python
def test_supported_cloud_uses_motion_and_sky_but_not_semantic(tmp_path: Path) -> None:
    inputs = depth_fixture(tmp_path, frame_count=3)
    inputs.paint_mask("semantic_confirmed", frame=0, xy=(2, 2), value=True)
    cloud = build_supported_depth_cloud(
        inputs.frames,
        inputs.final_depth,
        inputs.masks,
        inputs.model_dir,
        policy=DenseSeedPolicy(0.20, 0.05, 0.0, 0.001, 2),
        mask_mode="motion_sky",
    )
    assert (cloud.source_frame_index == 0).any()
    assert cloud.source_xy.shape == (cloud.xyz.shape[0], 2)


def test_supported_cloud_requires_three_total_views(tmp_path: Path) -> None:
    inputs = depth_fixture(tmp_path, frame_count=3, agreeing_targets=1)
    cloud = build_supported_depth_cloud(
        inputs.frames,
        inputs.final_depth,
        inputs.masks,
        inputs.model_dir,
        policy=DenseSeedPolicy(0.20, 0.05, 0.0, 0.001, 2),
        mask_mode="motion_sky",
    )
    assert cloud.xyz.shape == (0, 3)
```

- [ ] **Step 2: Run the focused tests and verify the missing API failure**

Run: `python -m pytest tests/experiments/learned_quality/test_depth.py -k supported_cloud -q`

Expected: collection fails because `SupportedDepthCloud` and `build_supported_depth_cloud` do not exist.

- [ ] **Step 3: Add the typed cloud and explicit mask mode**

```python
DepthMaskMode = Literal["hard_uncertain", "motion_sky"]


@dataclass(frozen=True)
class SupportedDepthCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    confidence: np.ndarray
    view_support: np.ndarray
    source_frame_index: np.ndarray
    source_xy: np.ndarray
    camera_centers: np.ndarray
    source_model_digest: str
    source_mask_digest: str
    source_depth_digest: str
    source_frame_digest: str


def build_supported_depth_cloud(
    frames: tuple[FrameArtifact, ...],
    final_depth: PoseConditionedDepthResult,
    masks: MaskFusionEvidence,
    winner_model_dir: Path,
    *,
    policy: DenseSeedPolicy,
    mask_mode: DepthMaskMode = "hard_uncertain",
) -> SupportedDepthCloud:
    inputs = _validate_inputs(
        frames, final_depth, masks, winner_model_dir, policy, mask_mode=mask_mode
    )
    return _supported_depth_cloud(inputs, policy)
```

In `_validate_inputs`, keep the existing `hard_exclude | uncertain` branch as the
default. The `motion_sky` branch loads and digest-validates
`motion_confirmed_path` and `sky_confirmed_path`, combines only those masks with
invalid depth/confidence, and never reads semantic or training-validity masks.
`_supported_depth_cloud` retains the source frame index and `(x, y)` pixel for every
accepted row while using the existing `_world_points` and `_project_support` math.

- [ ] **Step 4: Run depth tests**

Run: `python -m pytest tests/experiments/learned_quality/test_depth.py -q`

Expected: all depth tests pass, including existing deterministic dense-seed tests.

- [ ] **Step 5: Commit the public candidate-cloud boundary**

```powershell
git add experiments/learned_quality/depth.py tests/experiments/learned_quality/test_depth.py
git commit -m "feat: expose supported depth candidates"
```

---

### Task 2: Implement deterministic floor-plane, hole, and seed geometry

**Files:**
- Create: `experiments/learned_quality/floor_recovery.py`
- Create: `tests/experiments/learned_quality/test_floor_recovery.py`

**Interfaces:**
- Consumes: `SupportedDepthCloud` from Task 1 and sparse points loaded from the accepted COLMAP text model.
- Produces: `FloorRecoveryPolicy`, `FloorPlane`, `FloorHoleMap`, `FloorSeedArtifact`, `estimate_floor_plane`, `build_floor_hole_map`, and `generate_floor_seed_artifact`.

- [ ] **Step 1: Write failing synthetic-room tests**

```python
POLICY = FloorRecoveryPolicy()


def test_floor_hole_seeds_stay_inside_known_patch(tmp_path: Path) -> None:
    sparse, cameras, cloud = synthetic_room_with_floor_hole()
    plane = estimate_floor_plane(sparse, cameras, policy=POLICY, seed=7)
    holes = build_floor_hole_map(sparse, cloud, plane, policy=POLICY)
    artifact = generate_floor_seed_artifact(
        sparse, cloud, plane, holes, tmp_path / "floor-artifact", policy=POLICY
    )
    with np.load(artifact.npz_path, allow_pickle=False) as values:
        xyz = values["xyz"]
    assert len(xyz) >= POLICY.minimum_seed_count
    assert np.all((xyz[:, 0] >= -0.5) & (xyz[:, 0] <= 0.5))
    assert np.all((xyz[:, 2] >= 1.0) & (xyz[:, 2] <= 2.0))


def test_weak_floor_plane_fails_closed() -> None:
    with pytest.raises(ValueError, match="reliable floor plane"):
        estimate_floor_plane(collinear_points(), camera_centers(), policy=POLICY, seed=0)


def test_floor_seed_cap_is_deterministic(tmp_path: Path) -> None:
    first = build_artifact(tmp_path / "a", max_seed_count=150_000)
    second = build_artifact(tmp_path / "b", max_seed_count=150_000)
    assert first.point_count == 150_000
    assert first.content_fingerprint == second.content_fingerprint
```

- [ ] **Step 2: Run the new test module and verify import failure**

Run: `python -m pytest tests/experiments/learned_quality/test_floor_recovery.py -q`

Expected: collection fails because `floor_recovery.py` does not exist.

- [ ] **Step 3: Implement immutable policies and artifacts**

```python
@dataclass(frozen=True)
class FloorRecoveryPolicy:
    plane_inlier_fraction: float = 0.02
    minimum_plane_inliers: int = 100
    minimum_plane_support_fraction: float = 0.005
    maximum_up_angle_degrees: float = 15.0
    minimum_above_below_ratio: float = 4.0
    grid_fraction: float = 0.01
    footprint_erosion_cells: int = 1
    sparse_support_radius_cells: float = 1.5
    minimum_component_cells: int = 4
    minimum_seed_count: int = 1_000
    maximum_seed_count: int = 150_000


@dataclass(frozen=True)
class FloorPlane:
    normal: np.ndarray
    offset: float
    basis_u: np.ndarray
    basis_v: np.ndarray
    robust_radius: float
    inlier_tolerance: float
    inlier_count: int
    inlier_fraction: float
    above_below_ratio: float
    median_camera_height: float


@dataclass(frozen=True)
class FloorHoleMap:
    origin_uv: np.ndarray
    cell_width: float
    shape: tuple[int, int]
    observed: np.ndarray
    holes: np.ndarray
    component_ids: np.ndarray
    initial_hole_cells: int


@dataclass(frozen=True)
class FloorSeedArtifact:
    npz_path: Path
    metadata_path: Path
    point_count: int
    content_fingerprint: str
```

Validate every scalar, shape, dtype, and array for finiteness in `__post_init__`.
Implement deterministic RANSAC, camera-side normal orientation, least-squares plane
refinement, plane-basis projection, one-cell binary erosion, four-cell connected
component filtering, half-cell voxel fusion, plane projection, median RGB, stable
lexicographic ranking, and a 150K hard cap exactly as specified.

- [ ] **Step 4: Add deterministic artifact persistence**

Write `floor_seeds.npz`, `floor_plane.json`, `floor_holes.json`, and
`floor_seeds.json` through an adjacent staging directory. Hash every file, fsync the
tree, atomically promote it, and reject non-canonical/existing output roots. Include
source digests, policy values, rejection counts, hole counts, and schema
`learned_quality.floor_recovery.v1` in the content fingerprint.

- [ ] **Step 5: Run floor geometry tests**

Run: `python -m pytest tests/experiments/learned_quality/test_floor_recovery.py -q`

Expected: all plane, hole, exclusion, view-support, fusion, cap, fingerprint, and fail-closed tests pass.

- [ ] **Step 6: Commit floor geometry**

```powershell
git add experiments/learned_quality/floor_recovery.py tests/experiments/learned_quality/test_floor_recovery.py
git commit -m "feat: add automatic floor-hole seeding"
```

---

### Task 3: Add asserting floor-seed initialization and bounded training

**Files:**
- Create: `experiments/learned_quality/floor_recovery_training.py`
- Create: `tests/experiments/learned_quality/test_floor_recovery_training.py`
- Modify: `experiments/learned_quality/ablation_training.py`
- Modify: `tests/experiments/learned_quality/test_ablation_training.py`

**Interfaces:**
- Consumes: `FloorSeedArtifact`, staged `LearnedArtifacts`, accepted model, and the existing `AblationDiagnosticCollector`.
- Produces: `FloorTrainingResult` and `run_floor_recovery_training(...) -> FloorTrainingResult`.

- [ ] **Step 1: Write failing seed-slice and training-profile tests**

```python
def test_initialize_floor_seed_slice_changes_only_trailing_rows() -> None:
    trainer = fake_trainer(original_count=4, floor_xyz=FLOOR_XYZ, floor_rgb=FLOOR_RGB)
    before = snapshot_gaussians(trainer.gs)
    initialize_floor_seed_slice(
        trainer,
        original_count=4,
        floor_xyz=FLOOR_XYZ,
        floor_rgb=FLOOR_RGB,
        maximum_scale=0.025,
    )
    assert_gaussian_rows_equal(trainer.gs, before, slice(0, 4))
    assert torch.allclose(trainer.gs.opacities[4:], torch.full((3, 1), -4.0))
    assert torch.all(trainer.gs.get_scales[4:] <= 0.025 + 1e-7)


def test_point_order_mismatch_fails_before_iteration_zero() -> None:
    trainer = fake_trainer(original_count=4, floor_xyz=FLOOR_XYZ.flip(0), floor_rgb=FLOOR_RGB)
    with pytest.raises(RuntimeError, match="floor seed slice does not match"):
        initialize_floor_seed_slice(
            trainer, original_count=4, floor_xyz=FLOOR_XYZ,
            floor_rgb=FLOOR_RGB, maximum_scale=0.025
        )


def test_floor_profile_is_depth_plus_legacy_density_without_masks() -> None:
    profile = make_floor_recovery_static_spec(base_spec())
    assert profile.quality.n_iters == 5_000
    assert profile.quality.advanced.lambda_depth is None
    prepared = prepare_floor_recovery_scene(staged_inputs(), floor_artifact())
    assert prepared.validity_mask is None
    assert prepared.adaptive_density is False
```

- [ ] **Step 2: Run tests and verify missing training API**

Run: `python -m pytest tests/experiments/learned_quality/test_floor_recovery_training.py -q`

Expected: collection fails because the training module does not exist.

- [ ] **Step 3: Implement scene preparation and trainer customization**

Copy the accepted COLMAP model to a fresh local scene, append only the verified floor
seed rows, copy RGB and validated depth, and pass `validity_mask=None`. Set the same
720-pixel schedule, random seed, 5K budget, and legacy density settings as
`legacy_control`/`depth_only`.

```python
def initialize_floor_seed_slice(
    trainer: Any,
    *,
    original_count: int,
    floor_xyz: np.ndarray,
    floor_rgb: np.ndarray,
    maximum_scale: float,
) -> None:
    expected = original_count + len(floor_xyz)
    if trainer.gs.num_points != expected:
        raise RuntimeError("floor seed initialization count mismatch")
    seed_slice = slice(original_count, expected)
    actual_xyz = trainer.gs.means.detach()[seed_slice].cpu().numpy()
    if not np.allclose(actual_xyz, floor_xyz, rtol=0.0, atol=1e-5):
        raise RuntimeError("floor seed slice does not match the verified artifact")
    actual_rgb = (
        trainer.gs.sh_dc.detach()[seed_slice, 0, :].cpu().numpy()
        * 0.28209479177387814
        + 0.5
    )
    if not np.allclose(actual_rgb, floor_rgb / 255.0, rtol=0.0, atol=1e-5):
        raise RuntimeError("floor seed colors do not match the verified artifact")
    with torch.no_grad():
        trainer.gs.opacities[seed_slice].fill_(-4.0)
        trainer.gs.scales[seed_slice].clamp_(max=math.log(maximum_scale))
        trainer.gs.quats[seed_slice].zero_()
        trainer.gs.quats[seed_slice, 0] = 1.0
        trainer.gs.sh_rest[seed_slice].zero_()
```

The customizer installs the standard legacy density controller, applies the seed-slice
initializer before the iteration-zero callback, and records a verified initialization
receipt. Initial point subsampling must be disabled; count/order assertions are
mandatory.

- [ ] **Step 4: Extend the diagnostic collector with an optional floor probe**

Add an optional `floor_probe` argument with a default of `None` so every existing
ablation call remains unchanged. At each checkpoint, render the floor footprint and
record floor alpha coverage, residual hole cells, plane depth error, perturbed depth
disagreement, and seeded-cell survival. Save the floor overlay and contact-sheet tiles.

- [ ] **Step 5: Run training and ablation regression tests**

Run: `python -m pytest tests/experiments/learned_quality/test_floor_recovery_training.py tests/experiments/learned_quality/test_ablation_training.py -q`

Expected: floor tests pass and existing seven-row ablation behavior is unchanged.

- [ ] **Step 6: Commit bounded training integration**

```powershell
git add experiments/learned_quality/floor_recovery_training.py experiments/learned_quality/ablation_training.py tests/experiments/learned_quality/test_floor_recovery_training.py tests/experiments/learned_quality/test_ablation_training.py
git commit -m "feat: train verified floor recovery seeds"
```

---

### Task 4: Orchestrate one-time staging, comparison, and atomic publication

**Files:**
- Create: `experiments/learned_quality/floor_recovery_runner.py`
- Create: `tests/experiments/learned_quality/test_floor_recovery_runner.py`

**Interfaces:**
- Consumes: existing `stage_ablation_inputs`, verified structural diagnostic reference, Tasks 1-3, and A100 hardware information.
- Produces: `FloorRecoveryRunSpec`, `FloorRecoveryDecision`, `FloorRecoveryRunResult`, and `run_floor_recovery_diagnostic(...)`.

- [ ] **Step 1: Write failing orchestration and gate tests**

```python
def test_floor_decision_requires_coverage_and_structure() -> None:
    decision = decide_floor_recovery(
        legacy=legacy_metrics(psnr=28.11, white=0.008),
        candidate=floor_metrics(
            psnr=27.4, white=0.010, oversized=0.004,
            out_of_bounds=0.0, hole_reduction=0.31,
            alpha_gain=0.14, perturbed_depth_ratio=1.05,
        ),
    )
    assert decision.passed is True


def test_runner_does_not_train_when_plane_is_rejected(tmp_path: Path) -> None:
    result = run_floor_recovery_diagnostic(
        spec=run_spec(tmp_path), services=services(reject_plane=True)
    )
    assert result.status == "rejected"
    assert result.decision.reason == "unreliable_floor_plane"
    assert services.training_calls == 0


def test_publication_cannot_target_production_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="floor recovery diagnostic"):
        validate_result_target(tmp_path / "myroom_learned_test_result")
```

- [ ] **Step 2: Run runner tests and verify missing module failure**

Run: `python -m pytest tests/experiments/learned_quality/test_floor_recovery_runner.py -q`

Expected: collection fails because `floor_recovery_runner.py` does not exist.

- [ ] **Step 3: Implement strict run and decision contracts**

```python
class FloorRecoveryPublishSpec(StrictModel):
    replace_owned_result: bool = True


class FloorRecoveryRunSpec(StrictModel):
    input_folder: str
    runtime_profile: Literal["a100_floor_recovery"] = "a100_floor_recovery"
    publish: FloorRecoveryPublishSpec = FloorRecoveryPublishSpec()


@dataclass(frozen=True)
class FloorRecoveryDecision:
    passed: bool
    reason: str
    failures: tuple[str, ...]


@dataclass(frozen=True)
class FloorRecoveryRunResult:
    run_id: str
    status: Literal["success", "rejected", "failed"]
    final_path: Path | None
    decision: FloorRecoveryDecision | None
```

Implement the exact gates from the approved spec: hole reduction >=25%, alpha gain
>=10 percentage points, PSNR deficit <=1 dB, white <=1.5%, oversized <=1%,
out-of-bounds <=0.1%, non-finite zero, and perturbed floor-depth ratio <=1.10.

- [ ] **Step 4: Stage and validate the legacy reference once**

Reuse `stage_ablation_inputs` for verified pretraining data. Copy only the legacy
receipt, metrics, reference checkpoint/contact sheets, and reference PLY from
`<input>_training_ablation` into local staging. Validate selection, camera model,
resolution, random seed, checkpoint schedule, and training configuration before any
floor geometry work. After staging, reject every training-time path under
`/content/drive`.

- [ ] **Step 5: Implement staged execution and atomic publication**

Generate the supported cloud, floor plane, hole map, and seeds; run the single 5K arm;
write decision/report/CSV/contact sheets/checkpoints/candidate PLY; publish through a
fresh sibling staging directory to `<input>_floor_recovery_diagnostic`.
`_SUCCESS.json` means diagnostic completion, while `decision.json.pass` is the recovery
verdict. Rejections publish evidence and skip training. Failures publish a compact
receipt without `_SUCCESS.json`.

- [ ] **Step 6: Run runner tests**

Run: `python -m pytest tests/experiments/learned_quality/test_floor_recovery_runner.py -q`

Expected: staging, reference mismatch, rejection, pass/fail gates, owned replacement,
no-overwrite, and failure-receipt tests pass.

- [ ] **Step 7: Commit the diagnostic runner**

```powershell
git add experiments/learned_quality/floor_recovery_runner.py tests/experiments/learned_quality/test_floor_recovery_runner.py
git commit -m "feat: orchestrate floor recovery diagnostic"
```

---

### Task 5: Add the pinned A100 notebook and CLI

**Files:**
- Create: `experiments/learned_quality/floor_recovery_notebook.py`
- Create: `scripts/learned_quality_floor_recovery_run.py`
- Create: `tests/experiments/learned_quality/test_floor_recovery_cli.py`
- Create: `tests/integration/test_learned_quality_floor_recovery_notebook_smoke.py`
- Generate: `colab/learned_quality_floor_recovery.ipynb`

**Interfaces:**
- Consumes: `run_floor_recovery_diagnostic` from Task 4.
- Produces: `/content/learned_floor_recovery_result.json` and the generated Colab notebook.

- [ ] **Step 1: Write failing CLI/notebook contract tests**

```python
def test_notebook_is_a100_high_ram_bounded_and_releases_runtime() -> None:
    notebook = build_floor_recovery_notebook(commit_sha="a" * 40)
    source = "\n".join(cell["source"] for cell in notebook["cells"])
    assert "A100" in source and "75" in source
    assert "5000" in source
    assert "120000" not in source
    assert "runtime.unassign()" in source
    assert "_floor_recovery_diagnostic" in source
    assert "_learned_test_result" not in result_assignment(source)


def test_cli_always_writes_a_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "RESULT_PATH", tmp_path / "result.json")
    monkeypatch.setattr(cli, "_load_runner", lambda: raising_runner)
    assert cli.main(["--spec", str(valid_spec(tmp_path))]) == 1
    assert json.loads(cli.RESULT_PATH.read_text())["status"] == "failed"
```

- [ ] **Step 2: Run notebook/CLI tests and verify missing APIs**

Run: `python -m pytest tests/experiments/learned_quality/test_floor_recovery_cli.py tests/integration/test_learned_quality_floor_recovery_notebook_smoke.py -q`

Expected: collection fails because the notebook generator and CLI do not exist.

- [ ] **Step 3: Implement the receipt-safe CLI**

Parse `--spec`, load the pinned runner lazily, serialize success/rejection/failure with
strict JSON, and atomically replace `/content/learned_floor_recovery_result.json`.
Return zero for completed success or completed rejection and one for infrastructure or
contract failure.

- [ ] **Step 4: Implement and generate the notebook**

The notebook must validate A100 >=75 GiB, mount Drive once, normalize the MyDrive-relative
input, reject output suffixes, clone the exact SHA, install locked dependencies, print
stage and checkpoint events unbuffered, invoke the CLI once, verify the result contract,
flush Drive, and release the runtime in `finally`.

Run: `python -m experiments.learned_quality.floor_recovery_notebook --output colab/learned_quality_floor_recovery.ipynb --commit <implementation-sha>`

Expected: one generated notebook whose metadata generator ID is
`4dgs-studio.learned-floor-recovery-notebook`.

- [ ] **Step 5: Run CLI and notebook tests**

Run: `python -m pytest tests/experiments/learned_quality/test_floor_recovery_cli.py tests/integration/test_learned_quality_floor_recovery_notebook_smoke.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit CLI and generator before final pin**

```powershell
git add experiments/learned_quality/floor_recovery_notebook.py scripts/learned_quality_floor_recovery_run.py tests/experiments/learned_quality/test_floor_recovery_cli.py tests/integration/test_learned_quality_floor_recovery_notebook_smoke.py
git commit -m "feat: add floor recovery A100 notebook"
```

---

### Task 6: Document, verify, pin, and publish the implementation

**Files:**
- Modify: `experiments/learned_quality/README.md`
- Modify: `colab/README.md`
- Regenerate: `colab/learned_quality_floor_recovery.ipynb`

**Interfaces:**
- Produces: pushed feature branch and an immutable notebook link ready for the user.

- [ ] **Step 1: Document the bounded workflow**

Add the notebook purpose, A100 High-RAM requirement, expected staging/training time,
terminal output folder, pass-versus-completion meaning, inspection-only PLY, live
progress commands, and explicit statement that the notebook cannot start a 120K run.

- [ ] **Step 2: Run focused tests**

Run:

```powershell
python -m pytest `
  tests/experiments/learned_quality/test_depth.py `
  tests/experiments/learned_quality/test_floor_recovery.py `
  tests/experiments/learned_quality/test_floor_recovery_training.py `
  tests/experiments/learned_quality/test_floor_recovery_runner.py `
  tests/experiments/learned_quality/test_floor_recovery_cli.py `
  tests/integration/test_learned_quality_floor_recovery_notebook_smoke.py -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run learned-quality regression tests**

Run: `python -m pytest tests/experiments/learned_quality tests/integration/test_learned_quality_ablation_notebook_smoke.py tests/integration/test_learned_quality_notebook_smoke.py -q`

Expected: all learned-quality and existing notebook tests pass.

- [ ] **Step 4: Verify source safety and plan coverage**

Run:

```powershell
rg -n "120000|learned_test_result|_result" experiments/learned_quality/floor_recovery* scripts/learned_quality_floor_recovery_run.py colab/learned_quality_floor_recovery.ipynb
git diff --check
git status --short
```

Expected: `120000` is absent; production suffixes appear only in rejection checks/docs;
the diff has no whitespace errors; only intended files are modified.

- [ ] **Step 5: Commit implementation and obtain its immutable SHA**

```powershell
git add experiments/learned_quality colab/README.md docs/superpowers/plans/2026-07-22-automatic-floor-hole-recovery.md
git commit -m "feat: add bounded floor recovery diagnostic"
git rev-parse HEAD
```

- [ ] **Step 6: Regenerate and test the notebook at the implementation SHA**

```powershell
$sha = git rev-parse HEAD
python -m experiments.learned_quality.floor_recovery_notebook --output colab/learned_quality_floor_recovery.ipynb --commit $sha
python -m pytest tests/experiments/learned_quality/test_floor_recovery_cli.py tests/integration/test_learned_quality_floor_recovery_notebook_smoke.py -q
```

Expected: the generated notebook contains exactly `$sha` and notebook tests pass.

- [ ] **Step 7: Commit and push the notebook pin**

```powershell
git add colab/learned_quality_floor_recovery.ipynb
git commit -m "chore: pin floor recovery notebook"
git status --short
git diff --cached
git push -u origin feature/learned-quality-a100
```

Expected: clean worktree synchronized with `origin/feature/learned-quality-a100`.

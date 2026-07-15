# A100 Learned-Quality Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one isolated, Run-All-safe A100 Colab notebook that compares
Classical and learned-hybrid geometry, trains the strongest passing candidate
with validated learned evidence and adaptive density, and publishes exactly
`<input_folder>_learned_test_result` without changing the production notebook
workflow.

**Architecture:** Reuse `run_static_notebook()` through an injected
`RunnerServices` composition. All learned models, geometry comparison, artifact
contracts, training adapters, reports, and publishing live under
`experiments/learned_quality`; shared runtime changes are limited to default-off
trainer extension seams whose absent path executes the existing expressions.
Each GPU model is a lazy, injectable backend so CPU tests verify orchestration
without downloading checkpoints.

**Tech Stack:** Python 3.11, Pydantic 2, PyTorch, NumPy, Pillow/OpenCV, PyCOLMAP
3.12.6, COLMAP 3.11 CUDA wrapper, DA3, Grounding DINO, SAM 2.1, SEA-RAFT,
pytest, nbformat, Google Colab, Google Drive.

## Global Constraints

- The existing frontend, FastAPI routes, notebook generator, product presets,
  `_production_services()`, and legacy notebooks remain behaviorally unchanged.
- The only success destination is the exact sibling
  `<input_folder_name>_learned_test_result`; `<input_folder_name>_result` remains
  byte-identical and is never an ownership or replacement target.
- Failure publishes only under
  `<input_folder_name>_learned_test_diagnostics/<run_id>`.
- Smart selection, Ultra frame budget 800, 120,000 iterations, native final
  resolution, and the existing Ultra losses/polish/export remain locked.
- Six million Gaussians is an emergency ceiling, not a target; reserve
  `max(10 GiB, 15% total VRAM)` and never overshoot in one density event.
- Dense learned initialization has a one-million-point corruption ceiling, not
  a target.
- DA3 anchors are 96 below 70 GiB and 120 at or above 70 GiB, with
  `process_res=504`; anchor OOM aborts rather than lowering quality.
- Protected packages are `torch`, `torchvision`, `numpy`, and `gsplat`; learned
  installation may not change their versions or module paths.
- Checkpoints and source repositories use the exact full revisions in the
  approved design specification.
- User strings are JSON values and argv elements, never generated source or
  shell interpolation.
- No bundled HTML, JavaScript, WASM, or web viewer.
- GPU integration tests are marked `gpu` and `integration`; local success claims
  cover only tests actually run on available hardware.

## File Structure

```text
experiments/__init__.py
experiments/learned_quality/{contracts,dependencies,lifecycle,da3}.py
experiments/learned_quality/{segmentation,flow,masks,photometric}.py
experiments/learned_quality/{geometry,depth,density,training}.py
experiments/learned_quality/{reports,publish,runner,notebook}.py
experiments/learned_quality/requirements-lock.txt
scripts/learned_quality_run.py
colab/learned_quality_a100_experiment.ipynb
tests/model/test_validity_mask.py
tests/model/test_density_extension_seam.py
tests/experiments/learned_quality/
tests/integration/test_learned_quality_notebook_smoke.py
```

Shared default-off changes are restricted to
`backend/model/losses_perceptual.py`, `backend/model/trainer.py`, and
`backend/pipeline.py`.

---

### Task 1: Locked experiment contracts and paths

**Files:**

- Create: `experiments/__init__.py`
- Create: `experiments/learned_quality/__init__.py`
- Create: `experiments/learned_quality/contracts.py`
- Create: `tests/experiments/__init__.py`
- Create: `tests/experiments/learned_quality/__init__.py`
- Create: `tests/experiments/learned_quality/test_contracts.py`

**Interfaces:** Produces `LearnedQualityRunSpec`, `parse_learned_spec_json()`,
`to_static_run_spec()`, exact result/diagnostic paths, model refs, stage records,
and the artifact/result dataclasses consumed by later tasks.

- [ ] **Step 1: Write failing contract tests**

```python
def test_static_conversion_is_locked() -> None:
    learned = LearnedQualityRunSpec(input_folder="captures/room")
    static = to_static_run_spec(learned)
    assert static.input_folder == "captures/room"
    assert static.frame_selection.mode.value == "smart"
    assert static.quality.profile.value == "ultra"
    assert static.quality.n_iters == 120_000
    assert static.quality.max_gaussians == 6_000_000
    assert static.quality.advanced.density_start_iter == 500
    assert static.quality.advanced.density_end_iter == 80_000
    assert static.quality.advanced.density_interval == 100


def test_paths_are_exact_siblings(tmp_path: Path) -> None:
    source = tmp_path / "2026-11-room"
    assert derive_learned_result_path(source) == tmp_path / (
        "2026-11-room_learned_test_result"
    )
    assert derive_learned_diagnostics_root(source) == tmp_path / (
        "2026-11-room_learned_test_diagnostics"
    )
```

Add parametrized rejections for absolute paths, traversal, backslashes, control
characters, Drive root, and all result/diagnostic suffix inputs.

- [ ] **Step 2: Confirm tests fail for the missing package**

Run `python -m pytest tests/experiments/learned_quality/test_contracts.py -q`.
Expected: collection fails with `ModuleNotFoundError` for `experiments`.

- [ ] **Step 3: Implement strict contracts**

```python
RESULT_SUFFIX = "_learned_test_result"
DIAGNOSTICS_SUFFIX = "_learned_test_diagnostics"
GENERATOR_ID = "4dgs-studio.learned-quality-a100"


class LearnedPublishSpec(StrictModel):
    replace_owned_result: bool = True


class LearnedQualityRunSpec(StrictModel):
    schema_version: Literal[1] = 1
    input_folder: str
    publish: LearnedPublishSpec = Field(default_factory=LearnedPublishSpec)

    @field_validator("input_folder")
    @classmethod
    def normalize_folder(cls, value: str) -> str:
        canonical = normalize_input_folder(value)
        if canonical.endswith((RESULT_SUFFIX, DIAGNOSTICS_SUFFIX)):
            raise ValueError("Choose the input folder, not an experiment output")
        return canonical


def to_static_run_spec(spec: LearnedQualityRunSpec) -> StaticNotebookRunSpec:
    return StaticNotebookRunSpec(
        input_folder=spec.input_folder,
        frame_selection=FrameSelectionSpec(mode=FrameSelectionMode.SMART),
        quality=NotebookQualitySpec(
            profile=NotebookQualityProfile.ULTRA,
            n_iters=120_000,
            max_gaussians=6_000_000,
            advanced=NotebookAdvancedConfig(
                density_start_iter=500,
                density_end_iter=80_000,
                density_interval=100,
                densify_grad_threshold=1e-4,
            ),
        ),
        publish=PublishSpec(
            replace_owned_result=spec.publish.replace_owned_result,
        ),
    )
```

Define frozen `ModelRef`, `FrameArtifact`, `StageRecord`, `LearnedArtifacts`,
`GeometryCandidateReport`, `LearnedReconstructionOutput`, and
`LearnedTrainingOutput`. The reconstruction output forwards `decision`,
`selected_manifest`, `accepted_model_dir`, `attempts`, and `decisions`.

- [ ] **Step 4: Run the contract tests**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 5: Commit the contracts**

```powershell
git add experiments tests/experiments/learned_quality/test_contracts.py
git diff --staged --check
git commit -m "feat: lock learned experiment contracts"
```

---

### Task 2: Protected dependency installation and model lifecycle

**Files:**

- Create: `experiments/learned_quality/requirements-lock.txt`
- Create: `experiments/learned_quality/dependencies.py`
- Create: `experiments/learned_quality/lifecycle.py`
- Create: `tests/experiments/learned_quality/test_dependencies.py`
- Create: `tests/experiments/learned_quality/test_lifecycle.py`

**Interfaces:** Consumes `ModelRef`; produces `install_learned_environment()`,
`verify_learned_environment()`, `snapshot_pinned_model()`, protected runtime
snapshots, `release_cuda_model()`, and `run_with_smaller_batch_retry()`.

- [ ] **Step 1: Write failing guard/lifecycle tests**

```python
def test_protected_package_in_pip_plan_is_rejected() -> None:
    report = {"install": [{"metadata": {"name": "torch", "version": "2.7.0"}}]}
    with pytest.raises(ProtectedRuntimeError, match="torch"):
        assert_no_protected_changes(report)
```

Also test removal/path changes, 40-character model revisions, and lifecycle
cleanup (`to("cpu")`, reference release, `gc.collect()`, conditional CUDA cache).
Test that a batch/chunk-independent stage retries exactly once at the declared
smaller size after CUDA OOM, records both sizes/outcomes, releases the model
before retry, and re-raises the second failure. The DA3 anchor stage must reject
this helper and abort without changing its locked anchor count or resolution.

- [ ] **Step 2: Confirm dependency tests fail**

Run `python -m pytest tests/experiments/learned_quality/test_dependencies.py tests/experiments/learned_quality/test_lifecycle.py -q`.

- [ ] **Step 3: Add the exact lock and guarded installer**

```text
transformers==4.57.6
huggingface-hub==0.36.2
tokenizers==0.22.1
safetensors==0.6.2
pycolmap==3.12.6
omegaconf==2.3.0
hydra-core==1.3.2
iopath==0.1.10
portalocker==3.2.0
addict==2.4.0
moviepy==1.0.3
trimesh==4.7.4
evo==1.36.5
```

Create `/content/learned-env` with `--system-site-packages`, dry-run pip with a
JSON report, reject changes to Torch/torchvision/NumPy/gsplat, and install the
lock with `--no-deps`. The lock excludes those four packages and xformers.

Clone source without running repository package metadata:

```text
DA3      3fe327a6abe2e5db95b54444ea95463dbfef5610
SAM2     2b90b9f5ceec907a1c18123530e92e794ad901a4
SEA-RAFT 9137517ba24e628442aec097d3afe71d03503b75
```

Download checkpoints with exact revisions:

```text
depth-anything/DA3-BASE f4a6c9b3c95e41c82048423d3493a81ec3fa810e
depth-anything/DA3METRIC-LARGE 4010e39f3634a45bc60553321fb49fb760bd594e
IDEA-Research/grounding-dino-tiny a2bb814dd30d776dcf7e30523b00659f4f141c71
facebook/sam2.1-hiera-large 665f8e2ad61cf5f53d65644ff27c8ee525124610
MemorySlices/Tartan-C-T-TSKH-spring540x960-M eb97ef34ba5d856c3fa2cdcd073150c057ac8b69
```

Record requested/resolved revisions, repository HEADs, `pip freeze`, and
before/after protected module snapshots in `model_manifest.json`.

Implement `run_with_smaller_batch_retry(stage, initial_size, retry_size,
callable)` only for adapters that explicitly declare chunk independence. It
catches CUDA OOM only, releases the failed model/runtime, retries once, and
returns a report containing initial/retry sizes and outcomes. Numerical, shape,
download, and import failures never enter this fallback.

- [ ] **Step 4: Run dependency tests**

Run the Step 2 command. Expected: all pass with injected subprocess/HF fakes.

- [ ] **Step 5: Commit dependency safety**

```powershell
git add experiments/learned_quality/requirements-lock.txt experiments/learned_quality/dependencies.py experiments/learned_quality/lifecycle.py tests/experiments/learned_quality
git diff --staged --check
git commit -m "feat: guard learned model dependencies"
```

---

### Task 3: DA3 cameras, sky, and final-pose depth adapters

**Files:**

- Create: `experiments/learned_quality/da3.py`
- Create: `tests/experiments/learned_quality/test_da3.py`
- Create: `tests/experiments/learned_quality/test_da3_gpu.py`

**Interfaces:** Produces `select_anchor_indices()`, `normalize_w2c()`,
`robust_shared_pinhole()`, `run_anchor_inference()`, `run_metric_sky()`, and
`run_pose_conditioned_depth()`.

- [ ] **Step 1: Write pure failing DA3 tests**

```python
@pytest.mark.parametrize("vram_gb, expected", [(40.0, 96), (69.9, 96), (70.0, 120), (80.0, 120)])
def test_anchor_budget(vram_gb: float, expected: int) -> None:
    indices = select_anchor_indices(frame_count=800, vram_gb=vram_gb)
    assert len(indices) == expected
    assert indices[0] == 0
    assert indices[-1] == 799
    assert indices == tuple(sorted(set(indices)))
```

Cover 3x4/4x4 W2C normalization, invalid rotation/homogeneous data, robust K,
exact names, overlapping 48-frame chunks, and explicit anchor OOM.

- [ ] **Step 2: Confirm DA3 tests fail**

Run `python -m pytest tests/experiments/learned_quality/test_da3.py -q`.

- [ ] **Step 3: Implement injectable DA3 adapters**

```python
prediction = model.inference(
    [str(path) for path in anchor_paths],
    process_res=504,
    process_res_method="upper_bound_resize",
    use_ray_pose=True,
    ref_view_strategy="middle",
)
```

Normalize to finite 4x4 OpenCV/COLMAP W2C, use a robust shared `PINHOLE` K,
save depth/confidence plus JSON atomically, use DA3Metric only for sky/metric
diagnostics, and run final-pose depth in 48-frame chunks with 24-frame stride
and `align_to_input_ext_scale=True`. DA3Metric per-frame work and final-pose
depth chunks use the one-time smaller-batch/chunk helper; DA3 anchor inference
explicitly does not and aborts on OOM at the locked 96/120 anchors and 504 px.

- [ ] **Step 4: Add marked GPU shape tests**

Mark with `gpu` and `integration`; use 3-6 tiny frames and assert finite outputs.

- [ ] **Step 5: Run CPU tests and commit**

```powershell
python -m pytest tests/experiments/learned_quality/test_da3.py -q
git add experiments/learned_quality/da3.py tests/experiments/learned_quality/test_da3.py tests/experiments/learned_quality/test_da3_gpu.py
git diff --staged --check
git commit -m "feat: add pinned DA3 experiment adapters"
```

---

### Task 4: Semantic, flow, mask-fusion, and photometric evidence

**Files:**

- Create: `experiments/learned_quality/segmentation.py`
- Create: `experiments/learned_quality/flow.py`
- Create: `experiments/learned_quality/masks.py`
- Create: `experiments/learned_quality/photometric.py`
- Create: `tests/experiments/learned_quality/test_segmentation.py`
- Create: `tests/experiments/learned_quality/test_flow.py`
- Create: `tests/experiments/learned_quality/test_masks.py`
- Create: `tests/experiments/learned_quality/test_photometric.py`
- Create: `tests/experiments/learned_quality/test_learned_models_gpu.py`

**Interfaces:** Consumes DA3 and Classical geometry; produces semantic/flow
artifacts, four-map evidence, both explicit mask polarities, and an
accepted-or-rejected photometric artifact.

- [ ] **Step 1: Write failing evidence tests**

Enforce the exact prompt, direct detection per frame, propagation unable to
self-confirm, unknown flow never dynamic, cycle/z-buffer rejection, ceiling-sky
rejection, 45/55% runaway handling, and:

```python
assert colmap_mask_path.name == "frame_000000.png.png"
assert colmap_keep[static_y, static_x] == 255
assert colmap_keep[dynamic_y, dynamic_x] == 0
assert training_validity[static_y, static_x] == 1.0
assert training_validity[dynamic_y, dynamic_x] == 0.0
```

Photometric tests recover known gain/bias, enforce gauge/clamps, held-out
improvement, clipping rejection above 0.5%, and unchanged geometry hashes.

- [ ] **Step 2: Confirm evidence tests fail**

Run `python -m pytest tests/experiments/learned_quality/test_segmentation.py tests/experiments/learned_quality/test_flow.py tests/experiments/learned_quality/test_masks.py tests/experiments/learned_quality/test_photometric.py -q`.

- [ ] **Step 3: Implement semantic adapters**

Use this immutable prompt:

```text
person. child. hand. dog. cat. animal. bicycle. motorcycle. scooter. car. truck. bus.
```

Run Grounding DINO and SAM direct prediction on every frame. Propagation may
fill/stabilize only when neighboring direct or confirmed-motion evidence exists.
Their independent per-frame/clip batches use the one-time smaller-batch retry
helper and record the fallback. No adapter silently changes model, resolution,
threshold, prompt, or output semantics.

- [ ] **Step 4: Implement flow and fusion**

Run SEA-RAFT forward/backward with uncertainty. Implement rigid projection,
z-buffer/depth-edge/cycle gates, static-track threshold calibration, and
temporal agreement. Preserve `semantic_confirmed`, `sky_confirmed`,
`motion_confirmed`, and `uncertain`; only the first three form `hard_exclude`.
SEA-RAFT pair batches use the same one-time smaller-batch retry contract.

- [ ] **Step 5: Implement deterministic photometric validation**

Fit per-frame RGB affine transforms on confirmed-static tracks with one
reference gauge, gain `[0.75, 1.33]`, bias `[-0.10, 0.10]`, temporal smoothing,
held-out improvement, and clipping guard. Geometry always reads original RGB.

- [ ] **Step 6: Add marked learned-model GPU tests**

With 3-6 tiny frames, run real Grounding DINO, SAM 2.1, and SEA-RAFT adapters;
assert expected shapes, finite values, exact frame/pair joins, and model release
between stages. Mark every test `gpu` and `integration`.

- [ ] **Step 7: Run evidence tests and commit**

```powershell
python -m pytest tests/experiments/learned_quality/test_segmentation.py tests/experiments/learned_quality/test_flow.py tests/experiments/learned_quality/test_masks.py tests/experiments/learned_quality/test_photometric.py -q
git add experiments/learned_quality/segmentation.py experiments/learned_quality/flow.py experiments/learned_quality/masks.py experiments/learned_quality/photometric.py tests/experiments/learned_quality
git diff --staged --check
git commit -m "feat: build learned scene evidence"
```

---

### Task 5: Deterministic Classical versus learned-hybrid geometry

**Files:**

- Create: `experiments/learned_quality/geometry.py`
- Create: `tests/experiments/learned_quality/test_geometry.py`
- Create: `tests/experiments/learned_quality/test_geometry_pycolmap.py`

**Interfaces:** Consumes the selected manifest, original frames, DA3 anchors,
and COLMAP keep masks. Produces `GeometryComparison` and a forwarding
`LearnedReconstructionOutput` compatible with production polish/metadata.

- [ ] **Step 1: Write failing geometry tests**

Use fake branch runners to assert identical manifest/frame digests, fresh
directories/databases, failed candidates never training, normalized closest-
failure ranking, the approved winner tuple, and one canonical backfill with
`max_per_interval=2` rerunning both branches once.

```python
def test_winner_key_is_exact() -> None:
    assert winner_key(candidate) == (
        round(-candidate.dominant.registered_ratio, 9),
        -candidate.dominant.registered_count,
        round(-candidate.dominant.registered_share, 9),
        -candidate.covered_endpoint_count,
        round(candidate.dominant.max_interior_gap_s, 9),
        round(candidate.dominant.median_reprojection_error_px, 9),
        round(candidate.dominant.p95_reprojection_error_px, 9),
        round(-candidate.dominant.median_track_length, 9),
        -candidate.dominant.sparse_point_count,
        candidate.candidate_id,
    )
```

Implement the closest-failure key exactly as
`(failed_gate_count, round(sum(normalized_deficits), 9), candidate_id)`. A
minimum gate contributes `max(0, (threshold - value) /
max(abs(threshold), 1e-9))`; a maximum gate uses the mirrored expression; each
failed Boolean gate contributes `1`. Reject missing/non-finite required metrics
instead of assigning a synthetic score.

- [ ] **Step 2: Confirm geometry tests fail**

Run `python -m pytest tests/experiments/learned_quality/test_geometry.py -q`.

- [ ] **Step 3: Implement the Classical candidate**

Wrap `run_colmap_attempt()`, `measure_models()`, and
`evaluate_reconstruction()` without changing production implementations. The
comparison owns the shared backfill; do not invoke the production one-branch
retry helper.

- [ ] **Step 4: Implement the learned-hybrid candidate**

Use a fresh database and exact filename/image-ID joins. Extract and match real
features with the keep-mask path; write a shared `PINHOLE` cameras-and-images
known-pose model from DA3 anchors; triangulate with `clear_points=True` and fixed
intrinsics; run fixed-intrinsics BA; then try focal-only BA and accept only when
metrics improve, no gate worsens, and focal drift is at most 5%. Register the
remaining frames and convert final models to text. Never call the DA3 exporter.

- [ ] **Step 5: Add a marked PyCOLMAP fixture test**

Mark `gpu` and `integration`; build 3-6 known-camera frames, verify database IDs,
triangulate a shared track, and check camera convention. CPU tests inject a fake
PyCOLMAP backend.

- [ ] **Step 6: Run geometry CPU tests and commit**

```powershell
python -m pytest tests/experiments/learned_quality/test_geometry.py -q
git add experiments/learned_quality/geometry.py tests/experiments/learned_quality/test_geometry.py tests/experiments/learned_quality/test_geometry_pycolmap.py
git diff --staged --check
git commit -m "feat: compare learned and classical geometry"
```

---

### Task 6: Final-pose depth validation and dense seed fusion

**Files:**

- Create: `experiments/learned_quality/depth.py`
- Create: `tests/experiments/learned_quality/test_depth.py`

**Interfaces:** Consumes pose-conditioned DA3 depth, winner geometry, masks, and
original RGB. Produces trainer-compatible depth maps and a validated dense-seed
NPZ/JSON pair.

- [ ] **Step 1: Write failing depth tests**

Test exact joins, confidence rejection, zero invalid/uncertain boundary depth,
support from two neighboring views, deterministic voxel/color fusion, sparse
preservation, and a hard `1_000_000` ceiling.

- [ ] **Step 2: Confirm depth tests fail**

Run `python -m pytest tests/experiments/learned_quality/test_depth.py -q`.

- [ ] **Step 3: Implement validation and fusion**

Write exactly:

```text
xyz          float32 [N, 3]
rgb          uint8   [N, 3]
confidence   float32 [N]
view_support uint16  [N]
```

Reject non-finite/out-of-bounds projections, require neighboring-depth
agreement, dilate invalid boundaries, fuse colors by median, voxel-filter and
sort deterministically by support/confidence, then apply the corruption ceiling.
Record source model/mask/depth/frame digests in the sidecar. Write depth as
`<frame_stem>_depth.npy`.

- [ ] **Step 4: Run tests and commit**

```powershell
python -m pytest tests/experiments/learned_quality/test_depth.py -q
git add experiments/learned_quality/depth.py tests/experiments/learned_quality/test_depth.py
git diff --staged --check
git commit -m "feat: validate learned depth and dense seeds"
```

---

### Task 7: Default-off validity masking across losses and metrics

**Files:**

- Modify: `backend/model/losses_perceptual.py:27-82`
- Modify: `backend/model/trainer.py:34-60,949-1002,1192-1243,1460-1600,1752-1780,2183-2245`
- Create: `tests/model/test_validity_mask.py`
- Create: `tests/model/test_validity_mask_gpu.py`

**Interfaces:** Adds `validity_mask: Sequence[torch.Tensor] | None = None` to
`Trainer4DGS.train()` (`1=valid`, `0=excluded`) and `spatial: bool = False` to
`LPIPSLoss`; both defaults preserve the exact legacy branch.

- [ ] **Step 1: Write failing mask and equivalence tests**

Assert `None` is bit-identical with unchanged history/logger keys. Test
normalized L1, spatial SSIM/LPIPS, depth support, soft weights, empty support,
zero invalid gradient, conservative resize, invalid ranges/shapes/lengths,
multiview rejection, and rejection with active `lambda_mask_motion`.

- [ ] **Step 2: Confirm focused tests fail**

Run `python -m pytest tests/model/test_validity_mask.py -q`.

- [ ] **Step 3: Implement a separately gated masked branch**

Keep current expressions untouched when `validity_mask is None`. Otherwise,
normalize pixel losses by valid weight; composite invalid prediction pixels with
detached targets; erode/pool support to SSIM/LPIPS receptive fields; compute
valid-weight PSNR; combine depth with learned/edit validity; and record skipped
terms when support is empty. Only masked training constructs spatial LPIPS.

- [ ] **Step 4: Run focused and legacy tests**

Run `python -m pytest tests/model/test_validity_mask.py tests/test_static_trainer_init.py -q`.

- [ ] **Step 5: Add the marked masked-training GPU smoke**

Use a 3-6-frame fixture for 100 iterations. Verify excluded RGB pixels have
zero gradient, all losses/metrics remain finite, and the exported static
`frame_0000.ply` has a valid portable PLY header/body. Mark the test `gpu` and
`integration`.

- [ ] **Step 6: Commit the seam**

```powershell
git add backend/model/losses_perceptual.py backend/model/trainer.py tests/model/test_validity_mask.py tests/model/test_validity_mask_gpu.py
git diff --staged --check
git commit -m "feat: add default-off validity masking"
```

---

### Task 8: Adaptive density and default-off trainer extension

**Files:**

- Modify: `backend/model/trainer.py:28,949-1002,2035-2148`
- Modify: `backend/pipeline.py:69-99,956-1015,1089-1128`
- Create: `experiments/learned_quality/density.py`
- Create: `tests/experiments/learned_quality/test_density.py`
- Create: `tests/model/test_density_extension_seam.py`

**Interfaces:** Production controllers keep exact `accumulate()`/`step()` calls;
experiment controllers may expose `accumulate_view()`/`step_at()`.
`run_pipeline()` gains default-`None` `trainer_customizer` and
`trainer_train_kwargs`. `Trainer4DGS.train()` gains a default-`None`
`density_quality_probe`. Produces an injectable `AdaptiveDensityController`.

- [ ] **Step 1: Write failing seam/controller tests**

Verify default call/stat equivalence, exact experiment view/iteration context,
prune-before-grow, cap no-overshoot, top-gradient selection, exact fake-VRAM
slots, 10k calibration, invalid-calibration stop, two-view denominator, three
low-candidate windows, quality baseline plus three low deltas, resolution reset,
continued pruning, optimizer state shapes, and JSON-safe events. Assert the
default quality probe is never called; an injected probe runs only at iterations
divisible by 5,000, receives fixed camera indices/resolution/SH degree, forwards
to `record_quality()`, and resets plateau patience after a resolution change.

- [ ] **Step 2: Confirm density tests fail**

Run `python -m pytest tests/model/test_density_extension_seam.py tests/experiments/learned_quality/test_density.py -q`.

- [ ] **Step 3: Add feature-detected calls**

```python
accumulate_view = getattr(self.density, "accumulate_view", None)
if accumulate_view is None:
    self.density.accumulate(self.gs)
else:
    view_id = (cam_id, idx) if is_multiview else idx
    accumulate_view(self.gs, view_id=view_id)

step_at = getattr(self.density, "step_at", None)
if step_at is None:
    stats = self.density.step(
        self.gs,
        optimizer=self.optimizer,
        dynamic_densify_scale=self.dynamic_densify_scale,
    )
else:
    stats = step_at(
        self.gs,
        optimizer=self.optimizer,
        dynamic_densify_scale=self.dynamic_densify_scale,
        iteration=it,
    )
```

Call `trainer_customizer(trainer)` only when supplied. Merge only non-colliding
`trainer_train_kwargs`; reject collisions with explicit production arguments.

Add this default-off trainer callback:

```python
density_quality_probe: Callable[
    ["Trainer4DGS", int, tuple[int, int], int], float
] | None = None
```

At each 5,000-iteration boundary, and only when the callback is supplied:

```python
aggregate = density_quality_probe(self, it, (Ws, Hs), active_sh_degree)
record_quality = getattr(self.density, "record_quality", None)
if record_quality is not None:
    record_quality(
        iteration=it,
        aggregate_valid_psnr_db=float(aggregate),
        resolution=(Ws, Hs),
    )
```

The experiment closure captures eight deterministic coverage-spaced frame
indices plus their cameras and validity masks, renders only those views, and
returns aggregate valid-pixel PSNR. The `None` branch performs no callback,
render, RNG draw, or history/log mutation.

- [ ] **Step 4: Implement adaptive invariants**

Prune first and filter controller state with the same keep mask. Track first and
second distinct views. Clone/split each consume one net slot. At synchronized
events select at most:

```python
min(
    eligible_count,
    hard_cap - post_prune_count,
    vram_growth_slots,
    10_000 if calibration_pending else eligible_count,
)
```

Calibrate `ceil(max(analytical_raw, observed_delta/net_growth) * 1.5)`; invalid
calibration stops growth. `record_quality()` needs a baseline plus three
consecutive `<0.02 dB` deltas and resets on resolution change. Growth stop never
stops pruning or training.

- [ ] **Step 5: Run density and production tests**

Run `python -m pytest tests/model/test_density_extension_seam.py tests/experiments/learned_quality/test_density.py tests/test_static_trainer_init.py -q`.

- [ ] **Step 6: Commit adaptive density**

```powershell
git add backend/model/trainer.py backend/pipeline.py experiments/learned_quality/density.py tests/model/test_density_extension_seam.py tests/experiments/learned_quality/test_density.py
git diff --staged --check
git commit -m "feat: add isolated adaptive density control"
```

---

### Task 9: Validated learned training adapter

**Files:**

- Create: `experiments/learned_quality/training.py`
- Create: `tests/experiments/learned_quality/test_training.py`

**Interfaces:** Consumes the passing reconstruction and accepted learned
artifacts. Produces `LearnedTrainingOutput` through production validation and an
injected pipeline runner.

- [ ] **Step 1: Write failing boundary tests**

Assert original hashes/gate are checked before mutation; corrected RGB is
installed only inside the custom runner; depth/validity names join exactly;
sparse points survive dense augmentation; density uses the customizer; and
production calls never see experiment inputs.

- [ ] **Step 2: Confirm training tests fail**

Run `python -m pytest tests/experiments/learned_quality/test_training.py -q`.

- [ ] **Step 3: Implement the runner**

```python
run_validated_training(
    prepared,
    static_spec,
    pipeline_runner=make_experiment_pipeline_runner(artifacts, density_factory),
)
```

After validation copies original inputs, verify/install accepted training RGB,
depth, validity tensors, and an augmented copy of `points3D.txt`; never mutate
the gated model. Write cache markers. Call real `backend.pipeline.run_pipeline()`
with existing Ultra config, the controller customizer, and only experiment
validity/quality kwargs. Pass the fixed eight-view `density_quality_probe`
through `trainer_train_kwargs` alongside `validity_mask`. Persist density
history, final count, RGB digest, and fallbacks.

- [ ] **Step 4: Run training regressions and commit**

```powershell
python -m pytest tests/experiments/learned_quality/test_training.py tests/static_pipeline/test_training.py -q
git add experiments/learned_quality/training.py tests/experiments/learned_quality/test_training.py
git diff --staged --check
git commit -m "feat: train with validated learned evidence"
```

---

### Task 10: Reports and transactional learned-result publishing

**Files:**

- Create: `experiments/learned_quality/reports.py`
- Create: `experiments/learned_quality/publish.py`
- Create: `tests/experiments/learned_quality/test_reports.py`
- Create: `tests/experiments/learned_quality/test_publish.py`

**Interfaces:** Produces four contact sheets, experiment/model/candidate reports,
a custom validator, exact-suffix publisher, and exact-suffix diagnostics.

- [ ] **Step 1: Write failing report/publisher tests**

Require the complete inventory, strict JSON, PLY/PNG signatures, checksum
coverage, no symlinks/web assets, experiment ownership, atomic rollback, refusal
to replace unowned folders, diagnostic allowlisting, and byte-identical legacy
`<input>_result` across success and injected failures.

- [ ] **Step 2: Confirm tests fail**

Run `python -m pytest tests/experiments/learned_quality/test_reports.py tests/experiments/learned_quality/test_publish.py -q`.

- [ ] **Step 3: Assemble the complete bundle**

Call production base report assembly, then add experiment artifacts, augment
quality JSON, set experiment generator identity locally, and recompute inventory
after all files exist. Require:

```text
splat.ply
preview.png
scene_metadata.json
quality_report.json
run_manifest.json
selection_manifest.json
experiment_report.json
geometry_candidates.json
model_manifest.json
diagnostics/masks_contact_sheet.png
diagnostics/depth_contact_sheet.png
diagnostics/geometry_contact_sheet.png
diagnostics/final_render_contact_sheet.png
diagnostics/density_history.json
diagnostics/photometric_report.json
logs/pipeline.log
```

- [ ] **Step 4: Implement experiment ownership and rollback**

Mirror the proven stage/copy/verify/backup/final/rollback algorithm, but derive
only `_learned_test_result` and authenticate only
`4dgs-studio.learned-quality-a100`. Diagnostics accept no PLY and write only to
`_learned_test_diagnostics/<run_id>`.

- [ ] **Step 5: Run tests and commit**

```powershell
python -m pytest tests/experiments/learned_quality/test_reports.py tests/experiments/learned_quality/test_publish.py tests/static_pipeline/test_publish.py -q
git add experiments/learned_quality/reports.py experiments/learned_quality/publish.py tests/experiments/learned_quality
git diff --staged --check
git commit -m "feat: publish learned test results safely"
```

---

### Task 11: Isolated services, orchestration, and CLI

**Files:**

- Create: `experiments/learned_quality/runner.py`
- Create: `scripts/learned_quality_run.py`
- Create: `tests/experiments/learned_quality/test_runner.py`
- Create: `tests/experiments/learned_quality/test_cli.py`

**Interfaces:** Produces `preflight_learned_runtime()`,
`make_learned_quality_services()`, `run_learned_quality_notebook()`, and CLI
receipts/exit codes.

- [ ] **Step 1: Write failing orchestration tests**

Assert A100 plus at least 39 GiB; Smart selection precedes learned work; exact
dependency-cycle order; both branches share one backfill; training follows a
passing winner; models release sequentially; late failures publish diagnostics;
production service identities remain unchanged; and CLI help/invalid specs load
no Torch or models.

- [ ] **Step 2: Confirm runner/CLI tests fail**

Run `python -m pytest tests/experiments/learned_quality/test_runner.py tests/experiments/learned_quality/test_cli.py -q`.

- [ ] **Step 3: Compose injected services**

Reuse production source discovery/copy, Smart selection, polish, metadata, and
base report assembly. Replace reconstruct, train, preflight, experiment report
assembly, validation, result publish, and diagnostics. Invoke:

```python
run_static_notebook(
    to_static_run_spec(spec),
    runtime_paths=runtime_paths,
    services=make_learned_quality_services(context),
)
```

Order: selection; DA3 anchors; Classical geometry; DINO/SAM/sky; bidirectional
flow; geometry masks; Hybrid geometry; optional shared backfill; winner; final
masks; photometric validation; final-pose depth; dense seeds; validated Ultra
training; polish; reports; publish. Catch late-stage failures and attach learned
diagnostics when production runner coverage ends.

- [ ] **Step 4: Implement deferred CLI receipts**

Mirror `scripts/notebook_static_run.py`; write `learned_run_result.json`, return
2 for invalid spec, 1 for failed run, 0 for success, and preserve diagnostics.

- [ ] **Step 5: Run orchestration and production regressions**

```powershell
python -m pytest tests/experiments/learned_quality/test_runner.py tests/experiments/learned_quality/test_cli.py tests/static_pipeline/test_runner.py -q
python scripts/learned_quality_run.py --help
```

- [ ] **Step 6: Commit orchestration**

```powershell
git add experiments/learned_quality/runner.py scripts/learned_quality_run.py tests/experiments/learned_quality
git diff --staged --check
git commit -m "feat: orchestrate learned quality experiment"
```

---

### Task 12: Run-All-safe static Colab notebook

**Files:**

- Create: `experiments/learned_quality/notebook.py`
- Create: `colab/learned_quality_a100_experiment.ipynb`
- Create: `tests/integration/test_learned_quality_notebook_smoke.py`

**Interfaces:** Produces deterministic cells tagged `title`, `config`,
`preflight`, `drive`, `path-spec`, `checkout`, `bootstrap`,
`learned-dependencies`, `verify`, `execute`, `validate`, and `summary`.

- [ ] **Step 1: Write failing notebook tests**

Require exactly one form field:

```python
INPUT_FOLDER = ""  # @param {type:"string"}
```

Assert preflight before mount; prompt/path validation after mount; full 40-char
SHA; bootstrap before guarded learned install; checked argv execution; exact
learned sibling and experiment `_SUCCESS`; all four contact sheets; and no code
or shell interpolation from malicious folder text. Assert execution uses
`/content/learned-env/bin/python`, whose venv has `--system-site-packages`, so
the guarded learned lock is visible without replacing bootstrap Torch/gsplat.

- [ ] **Step 2: Confirm notebook test fails**

Run `python -m pytest tests/integration/test_learned_quality_notebook_smoke.py -q`.

- [ ] **Step 3: Build deterministic cells**

Mount Drive, normalize the MyDrive-relative field using Task 1 behavior, write
`/content/learned_spec.json` with `json.dump`, clone one pinned commit, run
`colab/static_notebook_bootstrap.sh`, install/verify learned dependencies in
fresh subprocesses, and execute exactly:

```python
subprocess.run(
    [
        "/content/learned-env/bin/python",
        "scripts/learned_quality_run.py",
        "--spec",
        "/content/learned_spec.json",
    ],
    cwd=SOURCE_ROOT,
    check=True,
)
```

The final cell displays the four contact sheets and prints paths; no viewer.

- [ ] **Step 4: Generate with a reachable source pin**

Commit/push Tasks 1-11 first, then:

```powershell
$sourceCommit = git rev-parse HEAD
git merge-base --is-ancestor $sourceCommit origin/feature/static-5-6-flags
```

Generate the notebook with that SHA. The notebook commit may follow because all
runtime code exists at the pinned parent.

- [ ] **Step 5: Run smoke and commit**

```powershell
python -m pytest tests/integration/test_learned_quality_notebook_smoke.py -q
git add experiments/learned_quality/notebook.py colab/learned_quality_a100_experiment.ipynb tests/integration/test_learned_quality_notebook_smoke.py
git diff --staged --check
git commit -m "feat: add learned quality A100 notebook"
```

---

### Task 13: Regression verification, evidence, and final push

**Files:**

- Modify: `colab/README.md`
- Create: `docs/learned-quality-a100-test.md`

**Interfaces:** Produces user run instructions, local verification evidence, a
pushed notebook, and the explicit boundary before the real A100 room acceptance.

- [ ] **Step 1: Document the run**

Document: A100 High-RAM; Run All; MyDrive-relative input; compare
`<input>_learned_test_result` with untouched `<input>_result`; inspect four
contact sheets and `experiment_report.json`; quality acceptance waits for the
real room run.

- [ ] **Step 2: Run focused experiment tests**

Run `python -m pytest tests/experiments/learned_quality tests/model/test_validity_mask.py tests/model/test_density_extension_seam.py -q`.

- [ ] **Step 3: Run notebook and production regressions**

Run `python -m pytest tests/integration/test_learned_quality_notebook_smoke.py tests/integration/test_generated_notebook_smoke.py tests/static_pipeline -q`.

- [ ] **Step 4: Run non-GPU and frontend isolation checks**

```powershell
python -m pytest -m "not gpu and not integration" -q
npm --prefix frontend test
npm --prefix frontend run build
git diff a001515 -- colab/video_to_world.ipynb colab/sota_verify.ipynb colab/phase2_verify.ipynb frontend backend/notebooks backend/static_pipeline
git diff a001515 -- backend/model/trainer.py backend/model/losses_perceptual.py backend/pipeline.py
```

Only approved default-off trainer/pipeline seams may appear under `backend`; no
frontend, generator, `_production_services()`, preset, or legacy-notebook change.

- [ ] **Step 5: Inspect and commit documentation**

```powershell
git status --short
git add colab/README.md docs/learned-quality-a100-test.md
git diff --staged --check
git diff --staged
git commit -m "docs: explain learned A100 quality test"
```

- [ ] **Step 6: Push and verify tracking**

```powershell
git push
git rev-list --left-right --count 'HEAD...@{upstream}'
```

Expected: `0  0`.

- [ ] **Step 7: Hand off real A100 acceptance**

Report notebook path/SHA, exact Drive output, locally run commands/results, GPU
tests not run locally, and the reports/contact sheets to bring back. Do not claim
Luma parity or improvement before the room run.

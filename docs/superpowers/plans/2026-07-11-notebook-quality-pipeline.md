# Automated Static Splat Notebook and Quality Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a notebook-first static Gaussian-splat workflow that turns an ordinary Google Drive video or image folder into a quality-gated, portable `<input_folder_name>_result` bundle through one Colab Run-All.

**Architecture:** Keep generated notebooks thin and data-driven: the frontend submits a versioned `StaticNotebookRunSpec`, the backend returns a pinned notebook, and `scripts/notebook_static_run.py` invokes an isolated `backend/static_pipeline/` runner. The runner owns source discovery, frame selection, clean COLMAP attempts, reconstruction gates, legacy static-training adaptation, guarded polish, metadata, and transactional Drive publishing while the existing notebooks, `/process`, `JobManager`, and `Static3DSubmit` retain their entrypoints and behavior.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic 2, nbformat 5, OpenCV/FFmpeg/FFprobe, COLMAP, NumPy/SciPy/Pillow/plyfile, PyTorch/gsplat, React 19, TypeScript, Vitest, Testing Library, Google Colab and Google Drive.

## Global Constraints

- Work only on `feature/static-5-6-flags`; do not push `main`, `master`, or `develop` without explicit approval.
- Preserve all pre-existing dirty work, especially `backend/api.py`, `backend/static_presets.py`, `tests/test_api_static_presets.py`, and `frontend/src/components/Static3DSubmit.test.tsx`; Task 0 is the only task authorized to review, finish, and commit those exact preset changes, and every other task stages only its intentional files.
- Do not edit `colab/video_to_world.ipynb`, `colab/sota_verify.ipynb`, or `colab/phase2_verify.ipynb`.
- Do not implement the deferred phone-pose B/C capture tracks, ARKit/VIO adapters, or an App Store recording integration in this plan.
- Keep `/process`, `JobManager`, and `Static3DSubmit` available as the legacy local workflow; notebook generation must not create a job or navigate to Jobs.
- Accept ordinary sources only: exactly one root `.mp4`, `.mov`, or `.m4v`; supported root images; or supported images under one `images/` directory.
- Treat the Drive input as read-only and derive the only successful destination as the exact sibling `<input_folder_name>_result`.
- Product profiles are exactly: Balanced/L4 `30,000 / 250,000 / 300 / 1280`, High `50,000 / 500,000 / 450 / 1920`, Premium `100,000 / 1,000,000 / 600 / 2560`, and Ultra `120,000 / 3,000,000 / 800 / native`.
- Smart selection is the default; Fixed FPS is the comparison baseline with default FPS `4`; downstream reconstruction gates, training, polish, and publishing must be identical.
- Candidate analysis uses a 320 px long edge at no more than 12 FPS and never upscales source imagery.
- Permit one Smart-only backfill retry and add no more than two bridge candidates per uncovered interval while remaining within the profile budget.
- Matcher policy is exact: CUDA with at most 800 frames uses exhaustive; CPU first attempt uses sequential overlap 20; CPU retry uses exhaustive at at most 300 frames and sequential overlap 40 above 300.
- The final reconstruction gate is exact: registration at least 90% (95% target), dominant component at least 95% of the union registered across components, interior gaps at most 2 seconds, endpoints within 1 second, median reprojection error at most 1 px, p95 at most 2.5 px, median track length at least 3, and valid unique names/intrinsics/poses.
- Polish acceptance is exact: remove at most 15% of Gaussians, lose at most 5% opacity mass, mean PSNR drop at most 0.25 dB, mean SSIM drop at most 0.005, and no sampled view loses more than 1 dB PSNR.
- Emit orientation metadata only when the dominant-plane inlier ratio is at least 0.5 and camera-up/plane-up agree within 20 degrees.
- The portable artifact is a standard INRIA 3DGS `splat.ply`; do not claim `.splat` or SPZ output and do not bundle a web viewer.
- Generated notebooks must use argument arrays plus `subprocess.run(argv, check=True)`, never interpolate user strings into shell/Python source, never use mutable `git pull`, and never require a runtime restart.
- Every completed task ends with focused tests, a review of `git status` and `git diff --staged`, a Conventional Commit, and an automatic push to the feature branch per `AGENTS.md`.

---

## Locked File Map

| Responsibility | Files |
|---|---|
| RunSpec, Drive path policy, notebook product profiles | `backend/notebooks/drive_paths.py`, `backend/notebooks/models.py`, `backend/notebooks/presets.py` |
| Source discovery and immutable inventory | `backend/static_pipeline/contracts.py`, `backend/static_pipeline/sources.py` |
| Unified fixed/smart selection and backfill | `backend/static_pipeline/selection.py`, `backend/static_pipeline/selection_metrics.py` |
| Fingerprints and atomic stage promotion | `backend/static_pipeline/stage_cache.py` |
| Fresh COLMAP commands/attempts | `backend/static_pipeline/colmap.py`, `backend/preprocess/run_colmap.py` |
| Model metrics, ranking, gates, retry | `backend/static_pipeline/reconstruction.py`, `backend/preprocess/parse_colmap.py` |
| Exact frame/pose/depth joins | `backend/preprocess/frame_alignment.py`, `backend/pipeline.py` |
| Legacy static-training adapter | `backend/static_pipeline/training.py`, `backend/notebooks/training_config.py` |
| Guarded PLY polish | `backend/static_pipeline/polish.py` |
| Metadata, preview and optional world assets | `backend/static_pipeline/scene_metadata.py` |
| Bundle validation and transactional publishing | `backend/static_pipeline/publish.py` |
| End-to-end orchestration and CLI | `backend/static_pipeline/runner.py`, `scripts/notebook_static_run.py` |
| Notebook generation, release source and HTTP boundary | `colab/static_notebook_bootstrap.sh`, `backend/notebooks/static_template.py`, `backend/notebooks/builder.py`, `backend/notebooks/source.py`, `backend/notebooks/routes.py`, `backend/api.py` |
| Frontend domain/API | `frontend/src/notebook/types.ts`, `drivePath.ts`, `reducer.ts`, `frontend/src/api.ts` |
| Frontend UI | `frontend/src/notebook/*.tsx`, `notebook.css`, `frontend/src/App.tsx` |

---

### Task 0: Finish and Land the Existing Static-Preset Compatibility WIP

**Files:**
- Modify: `backend/api.py` (preserve the current narrow preset diff)
- Create from existing untracked work: `backend/static_presets.py`
- Test from existing untracked work: `tests/test_api_static_presets.py`
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/components/Static3DSubmit.tsx`
- Test from existing untracked work: `frontend/src/components/Static3DSubmit.test.tsx`

**Interfaces:**
- Consumes: the already-present dirty preset changes and `scripts.static_3dgs.PRESETS`.
- Produces: one clean, committed compatibility baseline where the legacy local form accepts every real static runner preset, including SOTA and Ultra; it does not alter any existing preset values or notebook.

- [ ] **Step 1: Review the existing dirty diff before touching it**

Run:

```powershell
git status --short --branch
git diff -- backend/api.py
git diff --no-index -- NUL backend/static_presets.py
git diff --no-index -- NUL tests/test_api_static_presets.py
Get-Content -Raw frontend/src/components/Static3DSubmit.test.tsx
```

Expected: `backend/api.py` only delegates static preset names/application to the new helper; the helper derives names from `PRESETS`, makes `sota`/`ultra` native-resolution, and the tests cover that behavior. Stop and resolve with the user if unrelated edits appear in those exact files.

- [ ] **Step 2: Confirm the current frontend test fails for the known missing cards**

Run: `npm --prefix frontend test -- src/components/Static3DSubmit.test.tsx`

Expected: FAIL because SOTA and Ultra are not yet rendered.

- [ ] **Step 3: Complete the legacy frontend preset union and cards**

Change the existing union in `frontend/src/api.ts` to:

```typescript
export type StaticPreset = "fast" | "balanced" | "high" | "premium" | "sota" | "ultra";
```

Append these exact `PresetSpec` entries after Premium in `Static3DSubmit.tsx` without changing the four existing entries:

```typescript
{
  id: "sota",
  name: "SOTA",
  badge: "PSNR",
  duration: "benchmark",
  desc: "30k iter · native res · 6M cap · vanilla 3DGS parity benchmark",
},
{
  id: "ultra",
  name: "Ultra",
  badge: "MAX",
  duration: "5-8 saat",
  desc: "120k iter · native res · 3M cap · A100 80 GB sınıfı",
},
```

This only exposes runner presets that already exist; it does not change the default `balanced` local selection.

- [ ] **Step 4: Run backend and frontend compatibility tests**

Run:

```powershell
python -m pytest tests/test_api_static_presets.py -q
npm --prefix frontend test -- src/components/Static3DSubmit.test.tsx
```

Expected: backend `2 passed`; frontend test PASS.

- [ ] **Step 5: Stage only the reviewed compatibility files, inspect, commit, and push**

```powershell
git add backend/api.py backend/static_presets.py tests/test_api_static_presets.py frontend/src/api.ts frontend/src/components/Static3DSubmit.tsx frontend/src/components/Static3DSubmit.test.tsx
git status --short
git diff --staged
git commit -m "fix: expose all static runner presets"
git push
```

Expected: unrelated scripts/world folders remain untracked and unstaged; `backend/api.py` is clean for Task 13's later two-line router integration.

---

### Task 1: Versioned RunSpec, Drive Path Policy, and Product Profiles

**Files:**
- Create: `backend/notebooks/__init__.py`
- Create: `backend/notebooks/drive_paths.py`
- Create: `backend/notebooks/models.py`
- Create: `backend/notebooks/presets.py`
- Modify: `requirements.txt`
- Test: `tests/notebooks/test_drive_paths.py`
- Test: `tests/notebooks/test_models.py`
- Test: `tests/notebooks/test_presets.py`

**Interfaces:**
- Consumes: `scripts.static_3dgs.PRESETS`; Task 0's public compatibility helper remains available to the legacy API but notebook profile metadata is derived directly from the runner preset table.
- Produces: `normalize_input_folder(raw: str) -> str`, `derive_result_folder(input_folder: str) -> str`, `StaticNotebookRunSpec`, `parse_run_spec_json(data: str | bytes) -> StaticNotebookRunSpec`, `get_notebook_profile(profile) -> StaticNotebookProfile`, and `get_static_preset_manifest() -> StaticNotebookPresetManifest`.

- [ ] **Step 1: Add dependency floors used by the contract and later notebook route tests**

Add these exact lines under the Backend API and testing sections of `requirements.txt`:

```text
pydantic>=2.6,<3
nbformat>=5.10,<6
httpx>=0.27,<1
```

Install only these lightweight contract/test dependencies in the active development environment before running the new tests:

```powershell
python -m pip install "fastapi>=0.110" "pydantic>=2.6,<3" "nbformat>=5.10,<6" "httpx>=0.27,<1"
```

- [ ] **Step 2: Write failing path-policy tests**

Create `tests/notebooks/test_drive_paths.py` with parameterized acceptance and every rejection class:

```python
import pytest

from backend.notebooks.drive_paths import derive_result_folder, normalize_input_folder


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [("captures/myroom", "captures/myroom"), ("MyDrive/captures/myroom", "captures/myroom")],
)
def test_normalizes_my_drive_relative_paths(raw: str, canonical: str) -> None:
    assert normalize_input_folder(raw) == canonical
    assert derive_result_folder(canonical) == "captures/myroom_result"


@pytest.mark.parametrize(
    "raw",
    ["", "MyDrive", "MyDrive/", "/content/drive/MyDrive/x", "C:/x", "a\\b", "a/../b", "a/./b", "a//b", "/a/b", "a/b/", "a/result_result", "a/\x00b"],
)
def test_rejects_unsafe_or_ambiguous_paths(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_input_folder(raw)
```

- [ ] **Step 3: Run the path tests and confirm the missing module failure**

Run: `python -m pytest tests/notebooks/test_drive_paths.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'backend.notebooks'`.

- [ ] **Step 4: Implement one canonical Drive-path boundary**

Create `backend/notebooks/drive_paths.py` with this complete policy:

```python
from __future__ import annotations

import re
from pathlib import PurePosixPath


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")


def normalize_input_folder(raw: str) -> str:
    value = raw.strip()
    if value == "MyDrive":
        raise ValueError("Choose a folder below MyDrive, not the Drive root")
    if value.startswith("MyDrive/"):
        value = value[len("MyDrive/") :]
    if not value or "\\" in value or _CONTROL.search(value):
        raise ValueError("Input must be a non-empty MyDrive-relative folder")
    if value.startswith("/") or value.endswith("/") or _DRIVE_LETTER.match(value):
        raise ValueError("Absolute runtime paths are not accepted")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Path contains an empty or traversal segment")
    if parts[-1].endswith("_result"):
        raise ValueError("Choose the input folder, not a result folder")
    canonical = PurePosixPath(*parts).as_posix()
    if canonical in {"", "."}:
        raise ValueError("Choose a folder below MyDrive")
    return canonical


def derive_result_folder(input_folder: str) -> str:
    canonical = normalize_input_folder(input_folder)
    path = PurePosixPath(canonical)
    return path.with_name(f"{path.name}_result").as_posix()
```

- [ ] **Step 5: Write failing RunSpec and preset-manifest tests**

Create `tests/notebooks/test_models.py` and `tests/notebooks/test_presets.py` with these assertions:

```python
import json
import pytest
from pydantic import ValidationError

from backend.notebooks.models import StaticNotebookRunSpec, parse_run_spec_json


def test_runspec_defaults_are_smart_balanced_l4_and_fixed_four() -> None:
    spec = StaticNotebookRunSpec(input_folder="MyDrive/captures/room")
    assert spec.input_folder == "captures/room"
    assert spec.frame_selection.mode.value == "smart"
    assert spec.frame_selection.fixed_fps == 4
    assert spec.quality.profile.value == "balanced_l4"
    assert spec.quality.n_iters is None
    assert spec.publish.replace_owned_result is True


def test_runspec_rejects_unknown_advanced_keys_and_out_of_bounds_overrides() -> None:
    with pytest.raises(ValidationError):
        StaticNotebookRunSpec.model_validate({
            "schema_version": 1,
            "input_folder": "captures/room",
            "quality": {"n_iters": 999, "advanced": {"colmap_axis": "guess"}},
        })


def test_parser_round_trips_only_json_data() -> None:
    payload = json.dumps({"schema_version": 1, "input_folder": "captures/room;print('x')"})
    spec = parse_run_spec_json(payload)
    assert spec.input_folder == "captures/room;print('x')"
```

```python
from backend.notebooks.models import NotebookQualityProfile
from backend.notebooks.presets import get_notebook_profile, get_static_preset_manifest
from scripts.static_3dgs import PRESETS


def test_product_profiles_match_approved_budgets_and_legacy_training_values() -> None:
    expected = {
        "balanced_l4": ("balanced", 30_000, 250_000, 300, 1280),
        "high": ("high", 50_000, 500_000, 450, 1920),
        "premium": ("premium", 100_000, 1_000_000, 600, 2560),
        "ultra": ("ultra", 120_000, 3_000_000, 800, None),
    }
    for profile_id, values in expected.items():
        profile = get_notebook_profile(NotebookQualityProfile(profile_id))
        assert (profile.legacy_preset, profile.n_iters, profile.max_gaussians, profile.selected_frame_budget, profile.resolution_long_edge_cap) == values
        assert PRESETS[profile.legacy_preset]["n_iters"] == profile.n_iters
        assert PRESETS[profile.legacy_preset]["max_gaussians"] == profile.max_gaussians


def test_manifest_has_one_default_and_backend_owned_bounds() -> None:
    manifest = get_static_preset_manifest()
    assert manifest.default_profile == NotebookQualityProfile.BALANCED_L4
    assert len(manifest.profiles) == 4
    assert manifest.override_limits.n_iters.min == 1_000
    assert manifest.override_limits.n_iters.max == 120_000
```

- [ ] **Step 6: Run the contract tests and confirm they fail before implementation**

Run: `python -m pytest tests/notebooks/test_models.py tests/notebooks/test_presets.py -q`

Expected: FAIL because `backend.notebooks.models` and `backend.notebooks.presets` do not exist.

- [ ] **Step 7: Implement the strict Pydantic RunSpec**

Create `backend/notebooks/models.py` with these exact public types and validators:

```python
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .drive_paths import normalize_input_folder


class FrameSelectionMode(str, Enum):
    SMART = "smart"
    FIXED_FPS = "fixed_fps"


class NotebookQualityProfile(str, Enum):
    BALANCED_L4 = "balanced_l4"
    HIGH = "high"
    PREMIUM = "premium"
    ULTRA = "ultra"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrameSelectionSpec(StrictModel):
    mode: FrameSelectionMode = FrameSelectionMode.SMART
    fixed_fps: Annotated[int, Field(ge=1, le=30)] = 4


class NotebookAdvancedConfig(StrictModel):
    run_eval: bool | None = None
    foundation: bool | None = None
    resolution_long_edge_cap: Annotated[int | None, Field(ge=320, le=3840)] = None
    lambda_ssim: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    lambda_lpips: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    lambda_depth: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    density_start_iter: Annotated[int | None, Field(ge=0, le=120_000)] = None
    density_end_iter: Annotated[int | None, Field(ge=0, le=120_000)] = None
    density_interval: Annotated[int | None, Field(ge=10, le=5_000)] = None
    densify_grad_threshold: Annotated[float | None, Field(gt=0.0, le=0.1)] = None
    prune_min_opacity: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    prune_max_scale: Annotated[float | None, Field(gt=0.0, le=1.0)] = None
    opacity_reset_interval: Annotated[int | None, Field(ge=0, le=120_000)] = None
    sh_degree: Annotated[int | None, Field(ge=0, le=3)] = None
    multires_schedule: list[tuple[int, int]] | None = None

    @field_validator("multires_schedule")
    @classmethod
    def validate_schedule(cls, value: list[tuple[int, int]] | None) -> list[tuple[int, int]] | None:
        if value is None:
            return value
        if not value or value[0][0] != 0:
            raise ValueError("multires_schedule must start at iteration 0")
        starts = [stage[0] for stage in value]
        if starts != sorted(set(starts)) or any(start < 0 or edge < 320 or edge > 3840 for start, edge in value):
            raise ValueError("multires_schedule stages must be unique, increasing, and 320..3840 px")
        return value


class NotebookQualitySpec(StrictModel):
    profile: NotebookQualityProfile = NotebookQualityProfile.BALANCED_L4
    n_iters: Annotated[int | None, Field(ge=1_000, le=120_000)] = None
    max_gaussians: Annotated[int | None, Field(ge=50_000, le=6_000_000)] = None
    advanced: NotebookAdvancedConfig = Field(default_factory=NotebookAdvancedConfig)

    @model_validator(mode="after")
    def validate_iteration_relationships(self) -> "NotebookQualitySpec":
        advanced = self.advanced
        if advanced.density_start_iter is not None and advanced.density_end_iter is not None and advanced.density_start_iter >= advanced.density_end_iter:
            raise ValueError("density_start_iter must be below density_end_iter")
        return self


class PublishSpec(StrictModel):
    replace_owned_result: bool = True


class StaticNotebookRunSpec(StrictModel):
    schema_version: Literal[1] = 1
    input_folder: str
    frame_selection: FrameSelectionSpec = Field(default_factory=FrameSelectionSpec)
    quality: NotebookQualitySpec = Field(default_factory=NotebookQualitySpec)
    publish: PublishSpec = Field(default_factory=PublishSpec)

    @field_validator("input_folder")
    @classmethod
    def normalize_folder(cls, value: str) -> str:
        return normalize_input_folder(value)

    @model_validator(mode="after")
    def validate_effective_iteration_bounds(self) -> "StaticNotebookRunSpec":
        from .presets import get_notebook_profile

        effective_iters = self.quality.n_iters or get_notebook_profile(self.quality.profile).n_iters
        advanced = self.quality.advanced
        if advanced.density_start_iter is not None and advanced.density_start_iter >= effective_iters:
            raise ValueError("density_start_iter must be below effective n_iters")
        if advanced.density_end_iter is not None and advanced.density_end_iter > effective_iters:
            raise ValueError("density_end_iter cannot exceed effective n_iters")
        if advanced.multires_schedule and advanced.multires_schedule[-1][0] >= effective_iters:
            raise ValueError("multires_schedule stages must start before effective n_iters")
        return self


def parse_run_spec_json(data: str | bytes) -> StaticNotebookRunSpec:
    return StaticNotebookRunSpec.model_validate_json(data)
```

- [ ] **Step 8: Implement the backend-owned product profile manifest**

Create `backend/notebooks/presets.py` with a frozen `StaticNotebookProfile` dataclass and strict Pydantic `NumericLimit`, `OverrideLimits`, `NotebookProfileMetadata`, and `StaticNotebookPresetManifest` response models. `StaticNotebookProfile` contains `id`, `label`, `description`, `legacy_preset`, `n_iters`, `max_gaussians`, `selected_frame_budget`, `resolution_long_edge_cap`, `intended_gpu`, `minimum_vram_gb`, `foundation_default`, `run_eval_default`, and `warnings`. Assert training-value parity with `scripts.static_3dgs.PRESETS` in tests. Use exactly these rows:

```python
class NumericLimit(StrictModel):
    min: int
    max: int


class OverrideLimits(StrictModel):
    fixed_fps: NumericLimit
    n_iters: NumericLimit
    max_gaussians: NumericLimit


class NotebookProfileMetadata(StrictModel):
    id: NotebookQualityProfile
    label: str
    description: str
    n_iters: int
    max_gaussians: int
    selected_frame_budget: int
    resolution_long_edge_cap: int | None
    intended_gpu: str
    minimum_vram_gb: int
    foundation_default: bool
    run_eval_default: bool
    warnings: list[str]


class StaticNotebookPresetManifest(StrictModel):
    schema_version: Literal[1] = 1
    default_profile: NotebookQualityProfile
    profiles: list[NotebookProfileMetadata]
    override_limits: OverrideLimits
```

```python
PROFILE_ROWS = (
    ("balanced_l4", "Balanced / L4", "Low-mid product run with L4 headroom", "balanced", 30_000, 250_000, 300, 1280, "NVIDIA L4 24 GB", 22, True, False, ()),
    ("high", "High", "Higher-detail run with more memory and time", "high", 50_000, 500_000, 450, 1920, "32 GB class", 30, True, False, ("Use a runtime with at least 30 GB VRAM",)),
    ("premium", "Premium", "Large-memory high-quality run", "premium", 100_000, 1_000_000, 600, 2560, "48 GB class", 46, True, False, ("Long-running 46+ GB profile",)),
    ("ultra", "Ultra", "Maximum-quality native-resolution run", "ultra", 120_000, 3_000_000, 800, None, "A100 80 GB class", 75, True, False, ("A100 80 GB class runtime required",)),
)
```

The serialized manifest must contain `schema_version=1`, `default_profile="balanced_l4"`, all profile metadata, and these central limits:

```python
override_limits = {
    "fixed_fps": {"min": 1, "max": 30},
    "n_iters": {"min": 1_000, "max": 120_000},
    "max_gaussians": {"min": 50_000, "max": 6_000_000},
}
```

- [ ] **Step 9: Run focused and dirty-work compatibility tests**

Run: `python -m pytest tests/notebooks/test_drive_paths.py tests/notebooks/test_models.py tests/notebooks/test_presets.py tests/test_api_static_presets.py -q`

Expected: all tests PASS, including the two pre-existing static-preset tests.

- [ ] **Step 10: Commit and push Task 1**

```powershell
git add requirements.txt backend/notebooks/__init__.py backend/notebooks/drive_paths.py backend/notebooks/models.py backend/notebooks/presets.py tests/notebooks/test_drive_paths.py tests/notebooks/test_models.py tests/notebooks/test_presets.py
git status --short
git diff --staged
git commit -m "feat: add static notebook run contract"
git push
```

---

### Task 2: Immutable Source Discovery, Inventory, and Local Copy

**Files:**
- Create: `backend/static_pipeline/__init__.py`
- Create: `backend/static_pipeline/contracts.py`
- Create: `backend/static_pipeline/sources.py`
- Test: `tests/static_pipeline/test_sources.py`

**Interfaces:**
- Consumes: an already resolved local copy of the user's Drive input root.
- Produces: `SourceFile`, `SourceInventory`, `sha256_file(path, chunk_size)`, `discover_source(input_root)`, and `copy_input_read_only(source, destination) -> SourceInventory`.

- [ ] **Step 1: Write failing source-layout and digest tests**

Create `tests/static_pipeline/test_sources.py` with fixtures that cover the three accepted layouts, ambiguity, and byte identity:

```python
from pathlib import Path

import pytest

from backend.static_pipeline.sources import copy_input_read_only, discover_source


def test_discovers_exactly_one_root_video_and_hashes_bytes(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    video = root / "room.mov"
    video.write_bytes(b"first")
    first = discover_source(root)
    video.write_bytes(b"second")
    second = discover_source(root)
    assert first.kind == "video"
    assert first.digest != second.digest


@pytest.mark.parametrize("layout", ["root_images", "images_dir"])
def test_discovers_photo_sets_in_supported_locations(tmp_path: Path, layout: str) -> None:
    root = tmp_path / layout
    image_root = root if layout == "root_images" else root / "images"
    image_root.mkdir(parents=True)
    (image_root / "IMG_0001.JPG").write_bytes(b"one")
    inventory = discover_source(root)
    assert inventory.kind == "photo_set"
    assert [item.relative_path for item in inventory.media_files] == (["IMG_0001.JPG"] if layout == "root_images" else ["images/IMG_0001.JPG"])


@pytest.mark.parametrize("mutator", ["multiple_videos", "mixed", "both_image_locations", "empty"])
def test_rejects_ambiguous_or_empty_layouts(tmp_path: Path, mutator: str) -> None:
    root = tmp_path / mutator
    root.mkdir()
    if mutator in {"multiple_videos", "mixed"}:
        (root / "a.mp4").write_bytes(b"a")
    if mutator == "multiple_videos":
        (root / "b.mov").write_bytes(b"b")
    if mutator in {"mixed", "both_image_locations"}:
        (root / "a.jpg").write_bytes(b"a")
    if mutator == "both_image_locations":
        (root / "images").mkdir()
        (root / "images" / "b.png").write_bytes(b"b")
    with pytest.raises(ValueError):
        discover_source(root)


def test_copy_does_not_mutate_source_and_verifies_inventory(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "frame.jpg").write_bytes(b"pixels")
    before = discover_source(source)
    copied = copy_input_read_only(source, tmp_path / "local")
    after = discover_source(source)
    assert copied.digest == before.digest == after.digest
```

- [ ] **Step 2: Run the source tests and confirm collection failure**

Run: `python -m pytest tests/static_pipeline/test_sources.py -q`

Expected: FAIL with missing `backend.static_pipeline.sources`.

- [ ] **Step 3: Add immutable source contracts**

Create `backend/static_pipeline/contracts.py` with these exact dataclasses and JSON-safe serialization:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SourceInventory:
    schema_version: int
    root: Path
    kind: Literal["video", "photo_set"]
    media_files: tuple[SourceFile, ...]
    all_files: tuple[SourceFile, ...]
    digest: str

    def to_manifest_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["root"] = str(self.root)
        return value
```

- [ ] **Step 4: Implement strict discovery and a verified copy**

Create `backend/static_pipeline/sources.py`. The implementation must:

```python
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
```

Use sorted POSIX-relative names, reject symlinks that resolve outside `input_root`, and reject supported media in any nested location other than the single accepted `images/` directory. Hash every file with 8 MiB chunks, compute the inventory digest from canonical JSON with `sort_keys=True` and compact separators, and copy through a fresh sibling directory named `<destination>.tmp-<uuid>`. After copying, call `discover_source()` on the staged copy, compare its media relative names/sizes/hashes to the source inventory, and use `os.replace()` only when the target does not exist. Never call `dirs_exist_ok=True` and never write inside the source.

The digest input must have this exact shape:

```python
digest_payload = {
    "schema_version": 1,
    "kind": kind,
    "files": [
        {"relative_path": item.relative_path, "size_bytes": item.size_bytes, "sha256": item.sha256}
        for item in all_files
    ],
}
```

- [ ] **Step 5: Run focused source tests**

Run: `python -m pytest tests/static_pipeline/test_sources.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit and push Task 2**

```powershell
git add backend/static_pipeline/__init__.py backend/static_pipeline/contracts.py backend/static_pipeline/sources.py tests/static_pipeline/test_sources.py
git status --short
git diff --staged
git commit -m "feat: inventory static notebook sources"
git push
```

---

### Task 3: Unified Selection Manifest and Fixed-FPS/Photo Baselines

**Files:**
- Modify: `backend/static_pipeline/contracts.py`
- Create: `backend/static_pipeline/selection.py`
- Test: `tests/static_pipeline/test_selection.py`
- Test: `tests/static_pipeline/fixtures.py`

**Interfaces:**
- Consumes: `SourceInventory`, a `SelectionPolicy`, and FFprobe/FFmpeg binaries available in the runtime.
- Produces: `FrameMetrics`, `FrameRecord`, `SelectionPolicy`, `SelectionManifest`, `select_frames(inventory, output_dir, policy, media=None) -> SelectionManifest`, `bound_selection_for_reconstruction(inventory, manifest, output_dir, media=None, max_frames=800) -> SelectionManifest`, and `write_selection_manifest(path, manifest) -> Path`.

- [ ] **Step 1: Write failing unified-manifest baseline tests**

Create tests using an injected `MediaBackend` fake so unit tests do not need real codecs:

```python
def test_fixed_video_selection_is_deterministic_and_atomic(tmp_path, video_inventory, fake_media_backend):
    policy = SelectionPolicy(mode="fixed_fps", frame_budget=300, resolution_long_edge_cap=1280, fixed_fps=4)
    first = select_frames(video_inventory, tmp_path / "frames", policy, media=fake_media_backend)
    second = select_frames(video_inventory, tmp_path / "frames_again", policy, media=fake_media_backend)
    assert first.effective_mode == "fixed_fps"
    assert first.image_set_digest == second.image_set_digest
    assert len(first.selected_frames) <= 300
    assert all(frame.output_name == f"frame_{index:06d}.png" for index, frame in enumerate(first.selected_frames))


def test_photo_fixed_mode_keeps_all_images_and_records_effective_mode(tmp_path, photo_inventory, fake_media_backend):
    policy = SelectionPolicy(mode="fixed_fps", frame_budget=300, resolution_long_edge_cap=1280, fixed_fps=4)
    manifest = select_frames(photo_inventory, tmp_path / "frames", policy, media=fake_media_backend)
    assert manifest.effective_mode == "photo_set_all"
    assert len(manifest.selected_frames) == len(photo_inventory.media_files)


def test_large_photo_all_manifest_gets_a_recorded_colmap_guardrail(tmp_path, large_photo_inventory, fake_media_backend):
    policy = SelectionPolicy(mode="fixed_fps", frame_budget=300, resolution_long_edge_cap=1280, fixed_fps=4)
    original = select_frames(large_photo_inventory, tmp_path / "all-frames", policy, media=fake_media_backend)
    bounded = bound_selection_for_reconstruction(large_photo_inventory, original, tmp_path / "colmap-frames", media=fake_media_backend)
    assert len(original.selected_frames) == len(large_photo_inventory.media_files)
    assert len(bounded.selected_frames) == 300
    assert bounded.reconstruction_guardrail == "uniform_profile_cap"


def test_selection_replaces_stale_tail_files(tmp_path, video_inventory, fake_media_backend):
    target = tmp_path / "frames"
    target.mkdir()
    (target / "frame_999999.png").write_bytes(b"stale")
    select_frames(video_inventory, target, SelectionPolicy(mode="fixed_fps", frame_budget=3, resolution_long_edge_cap=1280), media=fake_media_backend)
    assert not (target / "frame_999999.png").exists()
    assert sorted(path.name for path in target.glob("*.png")) == ["frame_000000.png", "frame_000001.png", "frame_000002.png"]
```

- [ ] **Step 2: Run and confirm the missing selection contracts**

Run: `python -m pytest tests/static_pipeline/test_selection.py -q`

Expected: FAIL because `SelectionPolicy` and `select_frames` are undefined.

- [ ] **Step 3: Add exact selection contracts**

Append these frozen dataclasses to `backend/static_pipeline/contracts.py`:

```python
@dataclass(frozen=True)
class FrameMetrics:
    sharpness: float
    exposure_score: float
    duplicate_similarity: float
    overlap_score: float


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    source_relative_path: str
    source_index: int | None
    source_pts: int | None
    timestamp_s: float | None
    output_name: str
    sha256: str
    selected: bool
    metrics: FrameMetrics | None
    selection_score: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SelectionPolicy:
    mode: Literal["smart", "fixed_fps"]
    frame_budget: int
    resolution_long_edge_cap: int | None
    fixed_fps: int = 4
    candidate_fps: int = 12
    candidate_long_edge: int = 320
    version: str = "selection-v1"


@dataclass(frozen=True)
class SelectionManifest:
    schema_version: int
    source_digest: str
    effective_mode: Literal["smart", "fixed_fps", "photo_set_all"]
    policy: SelectionPolicy
    frames: tuple[FrameRecord, ...]
    image_set_digest: str
    reconstruction_guardrail: str | None = None

    @property
    def selected_frames(self) -> tuple[FrameRecord, ...]:
        return tuple(frame for frame in self.frames if frame.selected)
```

- [ ] **Step 4: Implement a protocol-backed, atomic baseline selector**

In `backend/static_pipeline/selection.py`, define a `MediaBackend` protocol with `probe_video(path)`, `video_timeline(path, fps_limit)`, `extract_video_indices(path, indices, destination, long_edge_cap)`, and `copy_photo(path, destination, long_edge_cap)`. Implement `FfmpegMediaBackend` using `ffprobe -show_frames -select_streams v:0 -show_entries frame=best_effort_timestamp,best_effort_timestamp_time:stream=avg_frame_rate,width,height:stream_tags=rotate` and FFmpeg select-script files passed as argv; do not build a shell string. Record the display rotation and apply it exactly once during native extraction so portrait iPhone `.mov` files remain upright.

Fixed video indices are chosen by target timestamps `k / fixed_fps` for integer `k >= 0`, each mapped to the nearest not-yet-used source PTS, then uniformly reduced to `frame_budget` if required. Candidate analysis uses every source frame when source FPS is below 12 and otherwise caps analysis at 12 FPS. Photo fixed mode preserves the natural filename order and ignores `fixed_fps`. Resize filters must use `min(source_long_edge, cap)` so no path upscales.

The initial Fixed photo manifest records every input image unchanged as `photo_set_all`. If that set exceeds the profile budget or absolute 800-frame matcher guardrail, `bound_selection_for_reconstruction` creates a separate uniformly order-sampled native-frame manifest capped at `min(profile budget, 800)`, marks non-chosen records `colmap_budget_guardrail`, and sets `reconstruction_guardrail="uniform_profile_cap"`. Preserve both manifests in the report so the safety cap is explicit rather than silently redefining the source-selection baseline.

Materialize all images into a new sibling staging directory, hash the final PNG bytes, verify the exact expected names, write `selection_manifest.json`, then replace the complete target directory. The manifest digest is calculated from ordered `(frame_id, output_name, sha256)` tuples.

Use this exact immutable frame ID rule:

```python
def make_frame_id(source_digest: str, source_index: int | None, timestamp_s: float | None, relative_path: str) -> str:
    identity = f"{source_digest}|{source_index}|{timestamp_s}|{relative_path}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:24]
```

- [ ] **Step 5: Run baseline selection tests**

Run: `python -m pytest tests/static_pipeline/test_selection.py -q`

Expected: all fixed-video, photo-set-all, deterministic digest, no-upscale, and stale-tail tests PASS.

- [ ] **Step 6: Commit and push Task 3**

```powershell
git add backend/static_pipeline/contracts.py backend/static_pipeline/selection.py tests/static_pipeline/test_selection.py tests/static_pipeline/fixtures.py
git status --short
git diff --staged
git commit -m "feat: add deterministic frame selection baseline"
git push
```

---

### Task 4: Smart Candidate Scoring, Window Selection, and Gap Backfill

**Files:**
- Create: `backend/static_pipeline/selection_metrics.py`
- Modify: `backend/static_pipeline/contracts.py`
- Modify: `backend/static_pipeline/selection.py`
- Modify: `tests/static_pipeline/test_selection.py`

**Interfaces:**
- Consumes: 320 px candidate RGB arrays at no more than 12 FPS and the Task 3 manifest contract.
- Produces: `UncoveredInterval`, `score_candidates(candidates) -> tuple[ScoredCandidate, ...]`, `choose_smart_candidates(candidates, budget)`, and `plan_backfill(inventory, manifest, uncovered, output_dir, media=None, max_per_interval=2) -> SelectionManifest`.

- [ ] **Step 1: Add failing deterministic smart-selection tests**

Add synthetic candidates with sharp, blurred, clipped, duplicate, and continuity-break frames. Assert:

```python
def test_smart_selection_prefers_sharp_exposed_nonduplicates_with_temporal_coverage():
    selected = choose_smart_candidates(synthetic_candidates(), budget=6)
    assert [item.source_index for item in selected] == [0, 3, 6, 9, 12, 15]
    assert all(left.timestamp_s < right.timestamp_s for left, right in zip(selected, selected[1:]))


def test_backfill_adds_at_most_two_bridges_per_gap_and_keeps_budget(tmp_path, smart_inventory, smart_manifest, fake_media_backend):
    gaps = (UncoveredInterval(start_s=4.0, end_s=7.0, missing_frame_ids=("a", "b", "c")),)
    backfilled = plan_backfill(smart_inventory, smart_manifest, gaps, tmp_path / "backfill", media=fake_media_backend, max_per_interval=2)
    added = set(frame.frame_id for frame in backfilled.selected_frames) - set(frame.frame_id for frame in smart_manifest.selected_frames)
    assert len(added) <= 2
    assert len(backfilled.selected_frames) <= smart_manifest.policy.frame_budget


def test_fixed_mode_rejects_backfill(tmp_path, fixed_inventory, fixed_manifest, fake_media_backend):
    with pytest.raises(ValueError, match="Smart"):
        plan_backfill(fixed_inventory, fixed_manifest, (), tmp_path / "backfill", media=fake_media_backend, max_per_interval=2)
```

- [ ] **Step 2: Run the new tests and confirm missing smart helpers**

Run: `python -m pytest tests/static_pipeline/test_selection.py -q`

Expected: FAIL with missing `choose_smart_candidates`, `UncoveredInterval`, or `plan_backfill`.

- [ ] **Step 3: Implement explicit, reproducible candidate metrics**

Append this shared interval contract to `backend/static_pipeline/contracts.py` before implementing the selector/reconstruction users:

```python
@dataclass(frozen=True)
class UncoveredInterval:
    start_s: float
    end_s: float
    missing_frame_ids: tuple[str, ...]
```

In `selection_metrics.py`, compute:

```python
sharpness = float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()))
dark_fraction = float(np.mean(gray <= 8))
bright_fraction = float(np.mean(gray >= 247))
midtone = 1.0 - abs(float(gray.mean()) - 127.5) / 127.5
exposure_score = float(np.clip(0.6 * (1.0 - dark_fraction - bright_fraction) + 0.4 * midtone, 0.0, 1.0))
duplicate_similarity = float(cv2.matchTemplate(previous_gray_64, gray_64, cv2.TM_CCOEFF_NORMED)[0, 0])
overlap_score = good_orb_matches / max(min(previous_keypoints, current_keypoints), 1)
```

For the first candidate, set duplicate similarity to `0.0`; for later candidates compare against the immediately preceding analyzed candidate. Normalize sharpness and overlap with deterministic 5th/95th percentile clipping. Use the exact total score:

```python
total_score = 0.45 * sharpness_norm + 0.25 * exposure_score + 0.20 * overlap_norm + 0.10 * (1.0 - duplicate_similarity)
```

If ORB yields fewer than 12 descriptors, set overlap to `0.0` and add reason `low_visual_overlap`. Reject candidates with `dark_fraction + bright_fraction > 0.35` as `bad_exposure`, the bottom 10% sharpness as `blurred`, and similarity above `0.985` as `near_duplicate`, except when retaining one is needed to cover an otherwise empty temporal window.

- [ ] **Step 4: Implement deterministic window coverage and bridge retention**

Partition duration into `min(frame_budget, candidate_count)` equal windows. Choose the highest score in every non-empty window, then reserve bridge candidates where adjacent winners have overlap below `0.12`. If reservations exceed budget, remove the lowest-scoring redundant non-endpoint candidate whose neighbors both overlap at least `0.20`. Always keep the earliest and latest usable candidate. Return records in timestamp order.

For image sets, use EXIF `DateTimeOriginal` when all usable values exist; otherwise natural filename order. Apply the same metrics and window logic without pretending filenames are FPS timestamps.

- [ ] **Step 5: Implement one bounded backfill planner**

For each uncovered interval, select up to two previously rejected candidates inside the interval by `(overlap_score, total_score, -abs(timestamp-midpoint))`, mark their reason `colmap_gap_backfill`, then remove equally many lowest-scoring redundant selected records outside all uncovered intervals. Materialize the revised native-resolution image set into another fresh directory and return a new `SelectionManifest` whose selected count is never above the original budget and whose image-set digest is recomputed from the new bytes.

- [ ] **Step 6: Run all selection tests, including a tiny real-media integration test**

Run: `python -m pytest tests/static_pipeline/test_selection.py -q`

Expected: all tests PASS. Mark the real FFmpeg fixture `@pytest.mark.integration` if FFmpeg is absent; pure selection tests must always run.

- [ ] **Step 7: Commit and push Task 4**

```powershell
git add backend/static_pipeline/contracts.py backend/static_pipeline/selection_metrics.py backend/static_pipeline/selection.py tests/static_pipeline/test_selection.py
git status --short
git diff --staged
git commit -m "feat: add smart frame selection and backfill"
git push
```

---

### Task 5: Stage Fingerprints and Confirmed Extraction Cache Fixes

**Files:**
- Create: `backend/static_pipeline/stage_cache.py`
- Modify: `backend/preprocess/extract_frames.py`
- Modify: `backend/preprocess/run_colmap.py`
- Modify: `backend/pipeline.py`
- Test: `tests/static_pipeline/test_stage_cache.py`
- Test: `tests/preprocess/test_extract_frames.py`
- Test: `tests/preprocess/test_run_colmap.py`

**Interfaces:**
- Consumes: source/selection digests and JSON-safe settings.
- Produces: `stage_fingerprint(stage, inputs, settings, policy_version, tools=None) -> str`, `cache_matches(marker_path, expected_fingerprint, required_paths) -> bool`, `promote_directory(staging, target) -> None`; adds `sequential_overlap` to legacy `run_colmap()` and makes invalidated legacy extraction remove the whole old frame directory.

- [ ] **Step 1: Write failing fingerprint, stale-tail, no-upscale, and overlap-forwarding tests**

```python
def test_source_or_selection_change_invalidates_stage():
    a = stage_fingerprint("frames", inputs={"source": "a", "selection": "one"}, settings={"cap": 1280}, policy_version="v1")
    b = stage_fingerprint("frames", inputs={"source": "b", "selection": "one"}, settings={"cap": 1280}, policy_version="v1")
    c = stage_fingerprint("frames", inputs={"source": "a", "selection": "two"}, settings={"cap": 1280}, policy_version="v1")
    assert len({a, b, c}) == 3


def test_extract_frames_scale_filter_never_upscales(mocker, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    mocker.patch("backend.preprocess.extract_frames._check_ffmpeg")
    output = tmp_path / "frames"
    output.mkdir()
    (output / "frame_0001.png").write_bytes(b"frame")
    seen = {}
    def fake_run(command, check):
        seen["command"] = command
        (output / "frame_0001.png").write_bytes(b"new-frame")
    mocker.patch("backend.preprocess.extract_frames.subprocess.run", side_effect=fake_run)
    extract_frames(video, output, fps=4, resize_long_edge=1280, overwrite=True)
    vf = seen["command"][seen["command"].index("-vf") + 1]
    assert "min(iw,1280)" in vf
    assert "min(ih,1280)" in vf


def test_sequential_overlap_reaches_colmap_command(mocker, tmp_path):
    commands = capture_colmap_commands(mocker, tmp_path, sequential=True, sequential_overlap=20)
    matcher = next(command for command in commands if "sequential_matcher" in command)
    assert matcher[-2:] == ["--SequentialMatching.overlap", "20"]
```

- [ ] **Step 2: Run focused tests and confirm current failures**

Run: `python -m pytest tests/static_pipeline/test_stage_cache.py tests/preprocess/test_extract_frames.py tests/preprocess/test_run_colmap.py -q`

Expected: FAIL because stage cache is missing, the current scale expression can upscale, and sequential overlap is not forwarded.

- [ ] **Step 3: Implement canonical stage fingerprints and safe promotion**

`stage_fingerprint` must hash this canonical payload and reject non-JSON values rather than `default=str` coercion:

```python
payload = {
    "schema_version": 1,
    "stage": stage,
    "inputs": dict(sorted(inputs.items())),
    "settings": settings,
    "policy_version": policy_version,
    "tools": dict(sorted(tools.items())),
}
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
return hashlib.sha256(encoded).hexdigest()
```

`cache_matches` must require marker schema `1`, exact fingerprint equality, and every required path. `promote_directory` must require an existing staging directory and non-existing target, then use same-parent `os.replace`; callers own backup/restore when replacement is allowed.

Every downstream fingerprint includes the exact upstream fingerprint/digest in `inputs` (source -> selection -> COLMAP -> training/polish). Therefore a changed source byte or selection manifest cannot reuse a downstream marker even when human-visible settings are unchanged.

- [ ] **Step 4: Fix no-upscale extraction and complete invalidation**

Change the resize filter in `backend/preprocess/extract_frames.py` to:

```python
vf_parts.append(
    f"scale='if(gt(iw,ih),min(iw,{resize_long_edge}),-2)'"
    f":'if(gt(iw,ih),-2,min(ih,{resize_long_edge}))'"
)
```

At `backend/pipeline.py`'s cache-miss extraction call, pass `overwrite=True`. This branch has already concluded that cache reuse is invalid; removing the complete old directory prevents stale tail PNGs from surviving under a newly written marker.

- [ ] **Step 5: Forward sequential overlap without changing legacy defaults**

Add `sequential_overlap: int = 10` to `run_colmap()`. Append:

```python
if sequential:
    extra_match += ["--SequentialMatching.overlap", str(sequential_overlap)]
```

Forward `cfg.preprocess.sequential_overlap` from `backend/pipeline.py`. Legacy callers retain the current default `10`; the new notebook policy will explicitly pass `20` or `40`.

- [ ] **Step 6: Run focused and legacy preprocessing tests**

Run: `python -m pytest tests/static_pipeline/test_stage_cache.py tests/preprocess/test_extract_frames.py tests/preprocess/test_run_colmap.py -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit and push Task 5**

```powershell
git add backend/static_pipeline/stage_cache.py backend/preprocess/extract_frames.py backend/preprocess/run_colmap.py backend/pipeline.py tests/static_pipeline/test_stage_cache.py tests/preprocess/test_extract_frames.py tests/preprocess/test_run_colmap.py
git status --short
git diff --staged
git commit -m "fix: invalidate stale frames and forward colmap overlap"
git push
```

---

### Task 6: Fresh COLMAP Attempts and Exact Matcher Policy

**Files:**
- Create: `backend/static_pipeline/colmap.py`
- Modify: `backend/static_pipeline/contracts.py`
- Test: `tests/static_pipeline/test_colmap.py`

**Interfaces:**
- Consumes: one complete selected-frame directory, `SelectionManifest`, resolved COLMAP executable, and attempt index `0` or `1`.
- Produces: `ColmapPolicy`, `ColmapAttempt`, `probe_colmap_gpu_support(colmap_exe, probe_root) -> bool`, `choose_colmap_policy(use_gpu, selected_count, attempt_index)`, `build_colmap_commands(frames_dir, attempt_dir, colmap_exe, policy)`, and `run_colmap_attempt(frames_dir, attempt_dir, policy, colmap_exe=None, on_progress=None) -> ColmapAttempt`.

- [ ] **Step 1: Write failing policy and clean-staging tests**

Create `tests/static_pipeline/test_colmap.py`:

```python
from pathlib import Path

import pytest

from backend.static_pipeline.colmap import build_colmap_commands, choose_colmap_policy, run_colmap_attempt


@pytest.mark.parametrize(
    ("use_gpu", "count", "attempt", "matcher", "overlap"),
    [
        (True, 300, 0, "exhaustive", 0),
        (True, 800, 1, "exhaustive", 0),
        (False, 300, 0, "sequential", 20),
        (False, 300, 1, "exhaustive", 0),
        (False, 301, 1, "sequential", 40),
    ],
)
def test_matcher_policy(use_gpu, count, attempt, matcher, overlap):
    policy = choose_colmap_policy(use_gpu=use_gpu, selected_count=count, attempt_index=attempt)
    assert policy.matcher == matcher
    assert policy.sequential_overlap == overlap


def test_policy_rejects_unbounded_sets_and_extra_attempts():
    with pytest.raises(ValueError):
        choose_colmap_policy(use_gpu=True, selected_count=801, attempt_index=0)
    with pytest.raises(ValueError):
        choose_colmap_policy(use_gpu=True, selected_count=300, attempt_index=2)


def test_commands_use_argv_and_forward_cpu_overlap(tmp_path: Path):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_000000.png").write_bytes(b"png")
    policy = choose_colmap_policy(use_gpu=False, selected_count=1, attempt_index=0)
    commands = build_colmap_commands(frames, tmp_path / "attempt", "colmap", policy)
    matcher = next(command for command in commands if "sequential_matcher" in command)
    assert matcher[-2:] == ("--SequentialMatching.overlap", "20")
    assert all(isinstance(command, tuple) for command in commands)


def test_attempt_refuses_existing_directory(tmp_path: Path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    with pytest.raises(FileExistsError):
        run_colmap_attempt(tmp_path / "frames", attempt, choose_colmap_policy(True, 1, 0))


def test_gpu_probe_falls_back_to_cpu_when_headless_sift_fails(tmp_path, mocker):
    mocker.patch("backend.static_pipeline.colmap.subprocess.run", side_effect=subprocess.CalledProcessError(1, ["colmap"]))
    assert probe_colmap_gpu_support("colmap", tmp_path / "probe") is False
```

- [ ] **Step 2: Run and confirm the missing COLMAP policy module**

Run: `python -m pytest tests/static_pipeline/test_colmap.py -q`

Expected: FAIL during collection with missing `backend.static_pipeline.colmap`.

- [ ] **Step 3: Add COLMAP attempt contracts**

Append to `contracts.py`:

```python
@dataclass(frozen=True)
class ColmapPolicy:
    camera_model: str
    matcher: Literal["sequential", "exhaustive"]
    sequential_overlap: int
    use_gpu: bool
    version: str = "colmap-policy-v1"


@dataclass(frozen=True)
class ColmapAttempt:
    root: Path
    database_path: Path
    model_dirs: tuple[Path, ...]
    colmap_version: str
    fingerprint: str
```

- [ ] **Step 4: Implement the exact policy function**

```python
def choose_colmap_policy(*, use_gpu: bool, selected_count: int, attempt_index: int) -> ColmapPolicy:
    if selected_count < 1 or selected_count > 800:
        raise ValueError("COLMAP policy accepts 1..800 selected frames")
    if attempt_index not in {0, 1}:
        raise ValueError("Only the initial attempt and one backfill retry are allowed")
    if use_gpu:
        return ColmapPolicy("PINHOLE", "exhaustive", 0, True)
    if attempt_index == 0:
        return ColmapPolicy("PINHOLE", "sequential", 20, False)
    if selected_count <= 300:
        return ColmapPolicy("PINHOLE", "exhaustive", 0, False)
    return ColmapPolicy("PINHOLE", "sequential", 40, False)
```

`probe_colmap_gpu_support` creates two tiny deterministic checkerboard PNGs in a fresh local probe directory and runs COLMAP feature extraction with GPU SIFT enabled and a checked argv. Return true only when the command succeeds and the database exists; on failure, remove the probe directory, record the reason, and return false. This distinguishes a CUDA Torch runtime from Colab's possible CPU-only/headless COLMAP fallback.

- [ ] **Step 5: Build and run every COLMAP command from a fresh directory**

`build_colmap_commands` must return feature extractor, chosen matcher, and mapper argv tuples. CPU policies append both `--SiftExtraction.use_gpu 0` and `--SiftMatching.use_gpu 0`. Sequential policies append exactly `--SequentialMatching.overlap <policy value>`. All paths are individual argv values.

`run_colmap_attempt` must execute this state machine:

```python
if attempt_dir.exists():
    raise FileExistsError(attempt_dir)
attempt_dir.mkdir(parents=True)
for command in build_colmap_commands(frames_dir, attempt_dir, colmap_exe, policy):
    subprocess.run(command, check=True)
model_dirs = tuple(sorted(path for path in (attempt_dir / "sparse").iterdir() if path.is_dir()))
if not model_dirs:
    raise RuntimeError("COLMAP produced no sparse model")
for model_dir in model_dirs:
    subprocess.run(
        (colmap_exe, "model_converter", "--input_path", str(model_dir), "--output_path", str(model_dir), "--output_type", "TXT"),
        check=True,
    )
```

Capture `colmap --version`, include it with selection digest and policy in `stage_fingerprint`, and return every converted model. Do not select a model by byte size and do not reuse `colmap.db` or `sparse/` from another attempt.

- [ ] **Step 6: Run focused tests with subprocesses mocked**

Run: `python -m pytest tests/static_pipeline/test_colmap.py tests/preprocess/test_run_colmap.py -q`

Expected: all tests PASS and every mock invocation receives a tuple/list argv plus `check=True`.

- [ ] **Step 7: Commit and push Task 6**

```powershell
git add backend/static_pipeline/contracts.py backend/static_pipeline/colmap.py tests/static_pipeline/test_colmap.py
git status --short
git diff --staged
git commit -m "feat: run colmap in fresh quality-gated attempts"
git push
```

---

### Task 7: COLMAP Model Metrics, Ranking, Gates, and One Retry

**Files:**
- Modify: `backend/preprocess/parse_colmap.py`
- Create: `backend/static_pipeline/reconstruction.py`
- Modify: `backend/static_pipeline/contracts.py`
- Test: `tests/static_pipeline/test_reconstruction.py`
- Test: `tests/static_pipeline/fixtures.py`

**Interfaces:**
- Consumes: converted text model directories and the current `SelectionManifest`.
- Produces: exact-model readers, `ModelMetrics`, `UncoveredInterval`, `GateDecision`, `ReconstructionBundle`, `measure_models`, `rank_models`, `evaluate_reconstruction(models, manifest, attempt_index=0)`, and `reconstruct_with_gate`.

- [ ] **Step 1: Add generated text-model fixtures and failing production-gate tests**

In `tests/static_pipeline/fixtures.py`, add `write_colmap_text_model(root, registered_names, point_errors, track_lengths, invalid_pose=False)`. It writes valid `cameras.txt`, two-line records in `images.txt`, and `points3D.txt`; fixture image names come from the manifest and no PNG payload is needed.

Create these tests:

```python
def test_fragmented_285_of_502_case_fails_and_requests_smart_backfill(fragmented_manifest, model_factory):
    model_dirs = model_factory(component_sizes=(285, 40, 30, 25, 20, 18, 16, 14, 12, 10, 7, 5))
    decision = evaluate_reconstruction(measure_models(model_dirs, fragmented_manifest), fragmented_manifest)
    assert decision.passed is False
    assert decision.dominant.registered_count == 285
    assert decision.retry_recommended is True
    assert "registered_ratio" in decision.failures


def test_healthy_95_of_100_model_passes(healthy_manifest, model_factory):
    model = model_factory(component_sizes=(95,), median_error=0.6, p95_error=1.8, median_track=4)[0]
    decision = evaluate_reconstruction(measure_models((model,), healthy_manifest), healthy_manifest)
    assert decision.passed is True


def test_registration_beats_binary_size_when_ranking(healthy_manifest, model_factory):
    strong, weak = model_factory(component_sizes=(95, 40), byte_padding=(0, 100_000))
    ranked = rank_models(measure_models((weak, strong), healthy_manifest))
    assert ranked[0].registered_count == 95


def test_invalid_names_or_nonfinite_pose_fail(healthy_manifest, model_factory):
    bad = model_factory(component_sizes=(95,), unknown_name=True, invalid_pose=True)[0]
    decision = evaluate_reconstruction(measure_models((bad,), healthy_manifest), healthy_manifest)
    assert decision.passed is False
    assert "valid_names_intrinsics_poses" in decision.failures
```

- [ ] **Step 2: Run reconstruction tests and confirm current selection-by-size behavior is insufficient**

Run: `python -m pytest tests/static_pipeline/test_reconstruction.py -q`

Expected: FAIL because exact-model metrics and gates do not exist.

- [ ] **Step 3: Expose exact-directory COLMAP readers without breaking legacy wrappers**

Refactor `backend/preprocess/parse_colmap.py` so:

```python
def parse_cameras_from_model(model_dir: str | Path) -> dict[str, dict]:
    model = Path(model_dir)
    if not (model / "cameras.txt").is_file() or not (model / "images.txt").is_file():
        raise FileNotFoundError(f"Incomplete COLMAP text model: {model}")
    return _parse_camera_and_image_text(model)


def load_points3d_from_model(model_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _parse_points3d_text(Path(model_dir) / "points3D.txt")


def parse_cameras(colmap_dir: str | Path) -> dict[str, dict]:
    return parse_cameras_from_model(_find_sparse_dir(Path(colmap_dir)))
```

Keep existing return keys and legacy `load_points3d`/`load_points3d_with_confidence` behavior. Parse and retain image IDs, exact names, `K`, `R`, `t`, `w2c`, width, and height. Detect duplicate image names while reading the two-line records and raise `ValueError` before constructing the name-keyed mapping so duplicates cannot be silently overwritten.

- [ ] **Step 4: Add reconstruction contracts**

```python
@dataclass(frozen=True)
class ModelMetrics:
    model_dir: Path
    registered_names: frozenset[str]
    registered_count: int
    registered_ratio: float
    registered_share: float
    temporal_coverage_s: float
    max_interior_gap_s: float
    start_gap_s: float
    end_gap_s: float
    median_reprojection_error_px: float
    p95_reprojection_error_px: float
    median_track_length: float
    sparse_point_count: int
    valid_names_intrinsics_and_poses: bool


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    dominant: ModelMetrics
    failures: tuple[str, ...]
    uncovered_intervals: tuple[UncoveredInterval, ...]
    retry_recommended: bool


@dataclass(frozen=True)
class ReconstructionBundle:
    selected_manifest: SelectionManifest
    accepted_model_dir: Path
    decision: GateDecision
    attempts: tuple[ColmapAttempt, ...]
```

- [ ] **Step 5: Implement metrics, temporal gaps, ranking, and the exact gate**

Match registered image names to manifest `output_name` exactly. Mark validity false for duplicates, unknown/missing names, non-finite `K`/`w2c`, non-positive focal lengths, invalid homogeneous rows, or singular rotations.

Compute dominant share against the union of registered names across every component. Build uncovered intervals from consecutive missing selected records bracketed by registered records; endpoints are distances from the first/last selected timestamp to the first/last registered timestamp. For photo sets without timestamps, use selection order with one unit per adjacent image and report the same normalized coverage checks in `quality_report.json` as order-based rather than seconds.

Sort by this exact tuple, descending:

```python
(
    model.registered_count,
    model.temporal_coverage_s,
    -model.max_interior_gap_s,
    model.registered_share,
    -model.median_reprojection_error_px,
    -model.p95_reprojection_error_px,
    model.median_track_length,
    model.sparse_point_count,
)
```

Evaluate the dominant model with this exact failure map:

```python
checks = {
    "registered_ratio": dominant.registered_ratio >= 0.90,
    "dominant_component": dominant.registered_share >= 0.95,
    "interior_gap": dominant.max_interior_gap_s <= 2.0,
    "start_endpoint": dominant.start_gap_s <= 1.0,
    "end_endpoint": dominant.end_gap_s <= 1.0,
    "median_reprojection": dominant.median_reprojection_error_px <= 1.0,
    "p95_reprojection": dominant.p95_reprojection_error_px <= 2.5,
    "median_track_length": dominant.median_track_length >= 3.0,
    "valid_names_intrinsics_poses": dominant.valid_names_intrinsics_and_poses,
}
failures = tuple(name for name, passed in checks.items() if not passed)
```

Gate failures use stable keys: `registered_ratio`, `dominant_component`, `interior_gap`, `start_endpoint`, `end_endpoint`, `median_reprojection`, `p95_reprojection`, `median_track_length`, and `valid_names_intrinsics_poses`. Registration from 90% through below 95% passes but adds the non-fatal report warning `below_95_percent_registration_target`. `retry_recommended` is true only for Smart mode on attempt zero.

- [ ] **Step 6: Orchestrate no more than one Smart backfill retry**

Implement:

```python
def reconstruct_with_gate(
    manifest: SelectionManifest,
    *,
    use_gpu: bool,
    attempt_root: Path,
    run_attempt: Callable[[SelectionManifest, Path, ColmapPolicy], ColmapAttempt],
    materialize_backfill: Callable[[SelectionManifest, tuple[UncoveredInterval, ...]], SelectionManifest],
) -> ReconstructionBundle:
    attempts: list[ColmapAttempt] = []
    current = manifest
    for attempt_index in range(2):
        policy = choose_colmap_policy(use_gpu=use_gpu, selected_count=len(current.selected_frames), attempt_index=attempt_index)
        attempt = run_attempt(current, attempt_root / f"attempt-{attempt_index}", policy)
        attempts.append(attempt)
        decision = evaluate_reconstruction(measure_models(attempt.model_dirs, current), current, attempt_index=attempt_index)
        if decision.passed:
            validated_staging = attempt_root / f"validated.tmp-{attempt_index}"
            validated_final = attempt_root / "validated"
            shutil.copytree(decision.dominant.model_dir, validated_staging)
            promote_directory(validated_staging, validated_final)
            return ReconstructionBundle(current, validated_final, decision, tuple(attempts))
        if not decision.retry_recommended or attempt_index == 1:
            raise ReconstructionGateError(decision, tuple(attempts))
        current = materialize_backfill(current, decision.uncovered_intervals)
    raise AssertionError("bounded reconstruction loop exhausted")
```

`ReconstructionGateError` carries the final decision/attempts for diagnostics; it must be raised before any training call.

Add a test with a pre-existing validated directory plus a failing new attempt and assert the previous validated directory remains byte-identical. A failed attempt is never promoted, and a successful accepted model is copied/promoted only after its gate passes.

- [ ] **Step 7: Run gate, parser, and bounded-retry tests**

Run: `python -m pytest tests/static_pipeline/test_reconstruction.py tests/preprocess/test_run_colmap.py -q`

Expected: all tests PASS, including 285/502 failure, 95/100 success, exact names, and one-retry enforcement.

- [ ] **Step 8: Commit and push Task 7**

```powershell
git add backend/preprocess/parse_colmap.py backend/static_pipeline/contracts.py backend/static_pipeline/reconstruction.py tests/static_pipeline/test_reconstruction.py tests/static_pipeline/fixtures.py
git status --short
git diff --staged
git commit -m "feat: gate fragmented colmap reconstructions"
git push
```

---

### Task 8: Filename-Safe Alignment and Validated Legacy Training Adapter

**Files:**
- Create: `backend/preprocess/frame_alignment.py`
- Modify: `backend/pipeline.py`
- Create: `backend/notebooks/training_config.py`
- Create: `backend/static_pipeline/training.py`
- Test: `tests/preprocess/test_frame_alignment.py`
- Test: `tests/notebooks/test_training_config.py`
- Test: `tests/static_pipeline/test_training.py`

**Interfaces:**
- Consumes: validated `ReconstructionBundle`, selected images, and `StaticNotebookRunSpec`.
- Produces: `RegisteredFrame`, `join_registered_frames`, `ResolvedStaticConfig`, `resolve_static_training_config(spec, source_long_edge=None)`, `PreparedTrainingInput`, and `run_validated_training` returning the raw standard PLY plus training summary.

- [ ] **Step 1: Write a failing regression test for the confirmed positional depth bug**

```python
def test_join_skips_unregistered_middle_frame_without_pose_shift(tmp_path):
    frames = tmp_path / "frames"
    depth = tmp_path / "depth"
    frames.mkdir()
    depth.mkdir()
    for index in range(3):
        (frames / f"frame_{index:06d}.png").write_bytes(b"png")
        (depth / f"frame_{index:06d}_depth.npy").write_bytes(b"depth")
    cameras = {
        "frame_000000.png": camera(tx=0.0),
        "frame_000002.png": camera(tx=2.0),
    }
    joined = join_registered_frames(frames, cameras, depth_dir=depth)
    assert [item.image_name for item in joined] == ["frame_000000.png", "frame_000002.png"]
    assert joined[1].depth_path.name == "frame_000002_depth.npy"
    assert joined[1].w2c[0, 3] == 2.0
```

- [ ] **Step 2: Run and confirm the helper is missing**

Run: `python -m pytest tests/preprocess/test_frame_alignment.py -q`

Expected: FAIL with missing `backend.preprocess.frame_alignment`.

- [ ] **Step 3: Implement exact-name joins and patch both pipeline consumers**

Create:

```python
@dataclass(frozen=True)
class RegisteredFrame:
    frame_id: str | None
    image_name: str
    image_path: Path
    depth_path: Path | None
    K: np.ndarray
    w2c: np.ndarray
    timestamp_s: float | None
```

`join_registered_frames` must index physical frames by `path.name`, reject duplicate basenames, require every camera name to exist physically, and join optional manifest records by `output_name`. Build depth names with `f"{image_path.stem}_depth.npy"`. Sort by manifest timestamp/order when supplied and by natural image name otherwise.

Replace `backend/pipeline.py`'s current `frame_paths_all[:n]` plus sorted-pose block with values derived from the same `RegisteredFrame` objects. Reuse the helper at the later training frame/camera join so depth alignment and training cannot diverge.

- [ ] **Step 4: Write failing profile-resolution and prepared-training tests**

```python
def test_balanced_l4_resolves_legacy_balanced_with_safe_overrides():
    spec = StaticNotebookRunSpec(input_folder="captures/room")
    cfg, resolved = resolve_static_training_config(spec)
    assert resolved.legacy_preset == "balanced"
    assert cfg.train.n_iters == 30_000
    assert cfg.train.max_gaussians == 250_000
    assert cfg.preprocess.resize_long_edge == 1280


def test_run_validated_training_never_calls_pipeline_before_gate(tmp_path, mocker):
    pipeline = mocker.Mock()
    with pytest.raises(ValueError, match="validated"):
        run_validated_training(unvalidated_prepared_input(tmp_path), StaticNotebookRunSpec(input_folder="captures/room"), pipeline_runner=pipeline)
    pipeline.assert_not_called()
```

- [ ] **Step 5: Implement the public training-configuration adapter**

`resolve_static_training_config` starts with `default_config()`, applies the profile's `legacy_preset` through Task 0's public `backend.static_presets.apply_static_preset_for_api`, applies nullable primary overrides, then applies only non-null fields from the approved `NotebookAdvancedConfig`. It sets `cfg.train.static_mode=True`, `cfg.export.num_timestamps=1`, `cfg.train.nvs_eval_enabled` to the explicit `run_eval` value or profile default, and native resolution only for Ultra. Foundation defaults to `True` for all four product profiles and can be explicitly disabled in Advanced. It clamps the final long edge to `min(source_long_edge, advanced cap or profile cap)` when a cap exists and never increases it above source dimensions.

Return a frozen `ResolvedStaticConfig` containing profile, legacy preset, `foundation`, `run_eval`, every effective override, and a SHA-256 digest of canonical `dataclasses.asdict(cfg)`. The runner combines it with selection mode and source/selection digests in `run_manifest.json` rather than pretending those values are training configuration.

- [ ] **Step 6: Implement a validated-scene adapter around the proven trainer**

Define:

```python
@dataclass(frozen=True)
class PreparedTrainingInput:
    run_id: str
    data_root: Path
    scene_name: str
    frames_dir: Path
    reconstruction: ReconstructionBundle
    source_digest: str
    selection_digest: str
```

`run_validated_training` must require `reconstruction.decision.passed`, create a fresh `data_root/scene_name`, copy the exact selected frames to `frames/`, copy only the accepted text model to `colmap/sparse/0/`, and write legacy frame/COLMAP markers with the resolved config so `run_pipeline()` skips preprocessing. Before importing `backend.config` or `backend.pipeline`, the notebook runner sets `FOURDGS_DATA_ROOT` to the unique local `data_root`; keep those imports inside the function. Call `run_pipeline` with `force_preprocess=False`, `skip_export=False`, and `skip_foundation=not resolved.foundation`, and require `output/ply/frame_0000.ply` to be newly created during this run. Return its path and status; do not publish here.

- [ ] **Step 7: Run focused config, adapter, and legacy pipeline tests**

Run: `python -m pytest tests/preprocess/test_frame_alignment.py tests/notebooks/test_training_config.py tests/static_pipeline/test_training.py tests/test_static_trainer_init.py -q`

Expected: all tests PASS and the mocked pipeline is called only with a passing gate.

- [ ] **Step 8: Commit and push Task 8**

```powershell
git add backend/preprocess/frame_alignment.py backend/pipeline.py backend/notebooks/training_config.py backend/static_pipeline/training.py tests/preprocess/test_frame_alignment.py tests/notebooks/test_training_config.py tests/static_pipeline/test_training.py
git status --short
git diff --staged
git commit -m "fix: align registered frames by immutable name"
git push
```

---

### Task 9: Regression-Guarded Static PLY Polish

**Files:**
- Create: `backend/static_pipeline/polish.py`
- Modify: `backend/static_pipeline/contracts.py`
- Test: `tests/static_pipeline/test_polish.py`

**Interfaces:**
- Consumes: raw standard PLY, accepted reconstruction, exact registered frames, and an injectable render evaluator.
- Produces: `PolishPolicy`, `PolishReport`, `validate_static_ply`, and `polish_static_ply`; raw PLY is immutable and remains the fallback.

- [ ] **Step 1: Write failing validation and regression-guard tests**

```python
def test_nonfinite_fields_and_bad_quaternions_fail_validation(tmp_path, ply_factory):
    bad = ply_factory(tmp_path / "bad.ply", nan_position=True, zero_quaternion=True)
    with pytest.raises(ValueError):
        validate_static_ply(bad)


@pytest.mark.parametrize(
    ("removed", "opacity_loss", "mean_psnr_drop", "mean_ssim_drop", "single_drop"),
    [(0.16, 0.01, 0.0, 0.0, 0.0), (0.01, 0.06, 0.0, 0.0, 0.0), (0.01, 0.01, 0.26, 0.0, 0.0), (0.01, 0.01, 0.0, 0.006, 0.0), (0.01, 0.01, 0.0, 0.0, 1.01)],
)
def test_any_acceptance_limit_keeps_raw(tmp_path, polish_fixture, removed, opacity_loss, mean_psnr_drop, mean_ssim_drop, single_drop):
    report = polish_fixture(removed, opacity_loss, mean_psnr_drop, mean_ssim_drop, single_drop)
    assert report.accepted is False
    assert report.selected_path == report.raw_path


def test_conservative_candidate_is_selected(tmp_path, polish_fixture):
    report = polish_fixture(0.08, 0.02, 0.10, 0.002, 0.5)
    assert report.accepted is True
    assert report.selected_path == report.candidate_path
```

- [ ] **Step 2: Run and confirm the polish stage is missing**

Run: `python -m pytest tests/static_pipeline/test_polish.py -q`

Expected: FAIL with missing `backend.static_pipeline.polish`.

- [ ] **Step 3: Add exact polish contracts and acceptance limits**

```python
@dataclass(frozen=True)
class PolishPolicy:
    max_removed_fraction: float = 0.15
    max_opacity_mass_loss: float = 0.05
    max_mean_psnr_drop_db: float = 0.25
    max_mean_ssim_drop: float = 0.005
    max_single_view_psnr_drop_db: float = 1.0
    min_opacity: float = 0.005
    max_relative_scale: float = 0.03
    max_anisotropy: float = 30.0
    crop_margin_fraction: float = 0.05


@dataclass(frozen=True)
class PolishReport:
    accepted: bool
    raw_path: Path
    candidate_path: Path | None
    selected_path: Path
    original_count: int
    kept_count: int
    opacity_mass_loss: float
    render_metrics: dict[str, object]
    reasons: tuple[str, ...]
```

- [ ] **Step 4: Implement conservative geometry/visibility pruning and edge fade**

Read the structured vertex array with `plyfile`. Require all standard INRIA fields and finite positions, SH features, opacity logits, log scales, and rotations; normalize nonzero quaternions in the candidate but fail zero/non-finite quaternions.

Compute alpha with sigmoid, scale with exponentiation, scene center with coordinate medians, and robust extent from the 99.5th percentile radius rather than the maximum. Candidate keep mask requires alpha at least `0.005`, max scale at most `0.03 * robust_extent`, anisotropy at most `30`, presence in at least one sampled registered camera frustum, and position inside the 0.5/99.5 percentile sparse-point bounds expanded by 5%. Evaluate frustums in chunks of at most 100,000 Gaussians.

Within the outer 5% crop margin, keep points but multiply alpha by a smoothstep distance-to-core factor and convert back to logits. Never add sky/background Gaussians.

- [ ] **Step 5: Render sampled cameras before and after, then choose candidate or raw**

Load each INRIA PLY into renderer tensors by stacking `x/y/z`, exponentiating `scale_0..2`, normalizing `rot_0..3`, applying sigmoid to `opacity`, and reconstructing colors as DC plus `f_rest` reshaped from exporter order `(N, 3*K)` back to `(N, K, 3)`. Use `backend.model.renderer.render_view` and reuse `_scale_K`, `_psnr`, and `_ssim` from `backend.eval.nvs_eval`; load exact ground-truth images from the matching `RegisteredFrame` records.

Sample at most 12 registered cameras uniformly over selection order. The evaluator returns per-view PSNR and SSIM for raw and candidate. Accept the candidate only when every exact policy limit passes; otherwise copy/select the raw PLY and record stable reason keys. Always validate the selected PLY again and include counts, opacity loss, per-view deltas, means, and decision in the report.

- [ ] **Step 6: Run polish tests**

Run: `python -m pytest tests/static_pipeline/test_polish.py -q`

Expected: all tests PASS, including invalid field rejection and raw fallback for each regression limit.

- [ ] **Step 7: Commit and push Task 9**

```powershell
git add backend/static_pipeline/contracts.py backend/static_pipeline/polish.py tests/static_pipeline/test_polish.py
git status --short
git diff --staged
git commit -m "feat: guard static splat polish against regressions"
git push
```

---

### Task 10: Confidence-Gated Scene Metadata and Preview

**Files:**
- Create: `backend/static_pipeline/scene_metadata.py`
- Test: `tests/static_pipeline/test_scene_metadata.py`

**Interfaces:**
- Consumes: selected final PLY and accepted reconstruction cameras/sparse points.
- Produces: `estimate_camera_up(cameras) -> np.ndarray`, `build_scene_metadata(ply_path, reconstruction) -> dict`, `write_scene_metadata`, `render_preview`, and optional viewer `world/` metadata; no web app files.

- [ ] **Step 1: Write failing confidence and default-camera tests**

```python
def test_low_plane_confidence_omits_orientation_transform(scene_fixture):
    scene = scene_fixture(plane_inlier=0.49, agreement_deg=5.0)
    metadata = build_scene_metadata(scene.ply_path, scene.reconstruction)
    assert metadata["orientation"]["applied"] is False
    assert "transform" not in metadata["orientation"]


def test_camera_plane_disagreement_omits_transform(scene_fixture):
    scene = scene_fixture(plane_inlier=0.8, agreement_deg=21.0)
    metadata = build_scene_metadata(scene.ply_path, scene.reconstruction)
    assert metadata["orientation"]["applied"] is False


def test_default_view_is_a_real_registered_camera_near_median(scene_fixture):
    scene = scene_fixture(plane_inlier=0.8, agreement_deg=10.0)
    metadata = build_scene_metadata(scene.ply_path, scene.reconstruction)
    assert metadata["default_camera"]["image_name"] in scene.cameras
    assert metadata["default_camera"]["source"] == "registered_colmap_camera"
    assert metadata["environment"]["sky"] == "viewer_recommendation_only"
```

- [ ] **Step 2: Run and confirm the metadata module is missing**

Run: `python -m pytest tests/static_pipeline/test_scene_metadata.py -q`

Expected: FAIL with missing `backend.static_pipeline.scene_metadata`.

- [ ] **Step 3: Combine camera-up and plane evidence conservatively**

Reuse `backend.image_to_scene.orientation.estimate_world_orientation` and `rotate_points`, but do not call `wrap_scene_as_world` because its confidence threshold and envelope output differ. Compute each camera's raw up as `R.T @ [0, -1, 0]`, flip signs toward the median direction, reject angular outliers above 30 degrees, and normalize the robust mean.

Compute agreement with:

```python
agreement_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(camera_up, plane.up_raw)), -1.0, 1.0))))
apply_orientation = plane.plane_inlier_frac >= 0.5 and agreement_deg <= 20.0
```

Only when `apply_orientation` is true may metadata include the quaternion transform.

- [ ] **Step 4: Derive bounds, ground, camera, environment, and viewer assets**

Use robust 0.5/99.5 percentile point bounds and `derive_minimal_collider` in the chosen metadata frame. Select the registered camera center nearest the coordinate-wise median camera center, and use its actual pose/intrinsics as `default_camera`. Emit schema version, coordinate convention, bounds, navigation/ground estimate, orientation evidence, default camera, source registered camera list, and environment recommendations. Sky/background are metadata strings/colors only.

Write optional `world/collider.json`, `world/trajectory.json`, and `world/request.json`; do not copy HTML, JavaScript, WASM, or a viewer bundle.

- [ ] **Step 5: Render an always-present preview without affecting acceptance**

Render the selected PLY from `default_camera` to `preview.png`; if the GPU renderer fails after training, use a deterministic sparse-point projection fallback so a successful bundle still has a preview. Record fallback usage as a warning in `quality_report.json`. Optional orbit rendering may produce `orbit.mp4`, but its failure must not fail the required bundle.

- [ ] **Step 6: Run metadata tests**

Run: `python -m pytest tests/static_pipeline/test_scene_metadata.py tests/image_to_scene/test_orientation.py tests/image_to_scene/test_collider.py -q`

Expected: all tests PASS and existing image-to-scene orientation/collider behavior remains green.

- [ ] **Step 7: Commit and push Task 10**

```powershell
git add backend/static_pipeline/scene_metadata.py tests/static_pipeline/test_scene_metadata.py
git status --short
git diff --staged
git commit -m "feat: derive confidence-gated scene metadata"
git push
```

---

### Task 11: Transactional Result Bundle and Drive Publishing

**Files:**
- Create: `backend/static_pipeline/publish.py`
- Modify: `backend/static_pipeline/contracts.py`
- Test: `tests/static_pipeline/test_publish.py`

**Interfaces:**
- Consumes: a verified local bundle, canonical Drive input folder, run ID, and `replace_owned_result`.
- Produces: `ArtifactRecord`, `PublishReceipt`, `derive_result_path`, `inventory_bundle`, `validate_bundle`, `is_owned_result`, `publish_result`, and `publish_diagnostics`.

- [ ] **Step 1: Write failing result-contract and rollback tests**

```python
REQUIRED = {"splat.ply", "scene_metadata.json", "preview.png", "quality_report.json", "run_manifest.json", "logs"}


def test_result_path_is_exact_sibling(tmp_path):
    assert derive_result_path(tmp_path / "captures" / "room") == tmp_path / "captures" / "room_result"


def test_owned_result_is_replaced_without_merging_stale_files(tmp_path, valid_bundle):
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    final = tmp_path / "room_result"
    make_owned_result(final)
    (final / "stale.bin").write_bytes(b"old")
    receipt = publish_result(valid_bundle, input_folder, run_id="run-2", replace_owned_result=True)
    assert receipt.final_path == final
    assert not (final / "stale.bin").exists()
    assert (final / "_SUCCESS").is_file()


def test_unowned_result_is_never_deleted(tmp_path, valid_bundle):
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    final = tmp_path / "room_result"
    final.mkdir()
    (final / "mine.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(PublishOwnershipError):
        publish_result(valid_bundle, input_folder, run_id="run-2", replace_owned_result=True)
    assert (final / "mine.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("failure_point", ["stage_verify", "backup_move", "final_move", "final_verify"])
def test_failure_restores_previous_good_result(tmp_path, valid_bundle, failing_file_ops, failure_point):
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    final = make_owned_result(tmp_path / "room_result", marker="previous")
    with pytest.raises(PublishError):
        publish_result(valid_bundle, input_folder, run_id="run-2", replace_owned_result=True, file_ops=failing_file_ops(failure_point))
    assert json.loads((final / "_SUCCESS").read_text(encoding="utf-8"))["marker"] == "previous"
```

- [ ] **Step 2: Run and confirm transactional publishing is missing**

Run: `python -m pytest tests/static_pipeline/test_publish.py -q`

Expected: FAIL with missing `backend.static_pipeline.publish`.

- [ ] **Step 3: Add non-circular artifact and receipt contracts**

```python
GENERATOR_ID = "4dgs-studio.static-notebook"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ArtifactRecord:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class PublishReceipt:
    run_id: str
    final_path: Path
    artifacts: tuple[ArtifactRecord, ...]
    manifest_sha256: str
    replaced_previous: bool
```

`run_manifest.json` inventories every artifact except itself and `_SUCCESS`; `_SUCCESS` records the manifest hash. This deliberately avoids a self-referential checksum.

`is_owned_result(path)` returns true only when both JSON files parse, `run_manifest.json` has the exact generator ID, supported schema `1`, and `status="success"`, and `_SUCCESS` repeats the same generator ID/run ID plus the actual SHA-256 of the manifest. A folder with only a similarly named file is unowned.

- [ ] **Step 4: Validate local bundles before touching Drive**

Require `splat.ply`, `scene_metadata.json`, `preview.png`, `quality_report.json`, `run_manifest.json`, and a non-empty `logs/`. Validate the PLY header, parse all JSON, require manifest `generator_id`, schema `1`, matching run ID, `status="success"`, source/artifact hashes, repository commit, tool versions, timing, and hardware. Reject symlinks and paths outside the bundle. Optional `world/` and `orbit.mp4` are included only when present.

- [ ] **Step 5: Implement stage, verify, success-last, backup, finalize, and restore**

Use exact sibling names:

```python
final = derive_result_path(input_folder)
staged = final.with_name(f"{final.name}.__tmp__{run_id}")
backup = final.with_name(f"{final.name}.__backup__{run_id}")
```

Copy the local bundle into a new staged path without merging, verify every inventory hash from the Drive copy, then write `_SUCCESS` last with `generator_id`, run ID, manifest SHA-256, and timestamp. If final exists, require `replace_owned_result=True` and `is_owned_result(final)`, then move it to backup. Move staged to final and verify again. Delete backup only after final verification. On any exception after backup creation, remove an invalid new final if owned by this run and restore backup. Prefer same-parent rename; a `FileOps` fallback performs copy-to-new-path plus full verification and source removal, never directory merge.

- [ ] **Step 6: Implement failure diagnostics without success semantics**

`publish_diagnostics(input_folder, run_id, files)` may create only `<input>_result_diagnostics/<run_id>/` with logs, contact sheet, quality report, and uncovered-interval report. It must reject `splat.ply`, never write `_SUCCESS`, and never touch `<input>_result`.

- [ ] **Step 7: Run publishing tests**

Run: `python -m pytest tests/static_pipeline/test_publish.py -q`

Expected: all tests PASS for owned replacement, unowned preservation, no merge, checksum verification, every injected failure, and backup restoration.

- [ ] **Step 8: Commit and push Task 11**

```powershell
git add backend/static_pipeline/contracts.py backend/static_pipeline/publish.py tests/static_pipeline/test_publish.py
git status --short
git diff --staged
git commit -m "feat: publish splat results transactionally"
git push
```

---

### Task 12: End-to-End Runner, Hardware Preflight, Reports, and CLI

**Files:**
- Create: `backend/static_pipeline/runner.py`
- Create: `scripts/notebook_static_run.py`
- Test: `tests/static_pipeline/test_runner.py`
- Test: `tests/notebooks/test_runner_cli.py`

**Interfaces:**
- Consumes: validated `StaticNotebookRunSpec`, mounted Drive root, local work root, and all Tasks 1-11 boundaries.
- Produces: `NotebookRuntimePaths`, `HardwareInfo`, `NotebookRunResult`, `runtime_paths_from_env`, `preflight_runtime`, `run_static_notebook`, and CLI `main(argv) -> int` with public argument `--spec`.

- [ ] **Step 1: Write failing preflight and orchestration-order tests**

```python
def test_balanced_l4_accepts_24gb_and_rejects_clearly_unsafe_vram():
    spec = StaticNotebookRunSpec(input_folder="captures/room")
    preflight_runtime(spec, HardwareInfo(gpu_name="NVIDIA L4", vram_gb=24.0, cuda_available=True, disk_free_gb=120.0), input_size_gb=2.0)
    with pytest.raises(RuntimePreflightError, match="Balanced / L4"):
        preflight_runtime(spec, HardwareInfo(gpu_name="T4", vram_gb=16.0, cuda_available=True, disk_free_gb=120.0), input_size_gb=2.0)


def test_runner_never_trains_before_gate_and_publishes_only_after_bundle_validation(tmp_path, boundaries):
    result = run_static_notebook(boundaries.spec, runtime_paths=boundaries.paths, services=boundaries.services)
    assert boundaries.calls == [
        "resolve_input", "discover", "preflight", "copy_input", "select", "colmap_gate",
        "train", "polish", "metadata_preview", "assemble_reports", "validate_bundle", "publish",
    ]
    assert result.final_path.name == "room_result"


def test_gate_failure_publishes_diagnostics_but_never_trains(tmp_path, failing_gate_boundaries):
    with pytest.raises(ReconstructionGateError):
        run_static_notebook(failing_gate_boundaries.spec, runtime_paths=failing_gate_boundaries.paths, services=failing_gate_boundaries.services)
    assert "train" not in failing_gate_boundaries.calls
    assert "publish" not in failing_gate_boundaries.calls
    assert failing_gate_boundaries.calls[-1] == "publish_diagnostics"


def test_smart_and_fixed_share_identical_downstream_configuration(comparison_boundaries):
    smart = comparison_boundaries.spec_for("smart")
    fixed = comparison_boundaries.spec_for("fixed_fps")
    run_static_notebook(smart, runtime_paths=comparison_boundaries.smart_paths, services=comparison_boundaries.smart_services)
    run_static_notebook(fixed, runtime_paths=comparison_boundaries.fixed_paths, services=comparison_boundaries.fixed_services)
    assert comparison_boundaries.smart_training_config == comparison_boundaries.fixed_training_config
    assert comparison_boundaries.smart_polish_policy == comparison_boundaries.fixed_polish_policy
    assert comparison_boundaries.smart_publish_policy == comparison_boundaries.fixed_publish_policy
```

- [ ] **Step 2: Run and confirm the orchestration boundary is missing**

Run: `python -m pytest tests/static_pipeline/test_runner.py -q`

Expected: FAIL with missing runner types/functions.

- [ ] **Step 3: Implement runtime paths and a fail-fast hardware preflight**

```python
@dataclass(frozen=True)
class NotebookRuntimePaths:
    drive_root: Path
    work_root: Path


@dataclass(frozen=True)
class HardwareInfo:
    gpu_name: str
    vram_gb: float
    cuda_available: bool
    disk_free_gb: float


@dataclass(frozen=True)
class NotebookRunResult:
    run_id: str
    final_path: Path
    local_bundle: Path
    quality_report_path: Path
    manifest_path: Path


def runtime_paths_from_env() -> NotebookRuntimePaths:
    return NotebookRuntimePaths(
        drive_root=Path(os.environ.get("STATIC_NOTEBOOK_DRIVE_ROOT", "/content/drive/MyDrive")).resolve(),
        work_root=Path(os.environ.get("STATIC_NOTEBOOK_WORK_ROOT", "/content/4dgs-runs")).resolve(),
    )
```

Detect CUDA/GPU/VRAM with `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`, disk with `shutil.disk_usage`, Python/tool versions with checked subprocesses, and require thresholds from the backend profile manifest: 22, 30, 46, and 75 GB respectively. `preflight_runtime(spec, hardware, input_size_gb)` requires at least `max(25, input_size_gb * 4 + 15)` GB free. Probe COLMAP GPU SIFT separately through Task 6; do not infer it from Torch CUDA. Print the detected model, VRAM, disk, COLMAP matcher mode, profile, iterations, Gaussian cap, frame budget, and resolution before input copying.

- [ ] **Step 4: Implement the exact state machine with dependency injection**

Create a `RunnerServices` dataclass with callables named `discover_source`, `copy_input`, `select_frames`, `reconstruct`, `train`, `polish`, `build_metadata_preview`, `validate_bundle`, `publish_result`, and `publish_diagnostics`; a production factory wires the real functions and tests inject fakes. `run_static_notebook` must:

1. Generate a collision-resistant run ID and fresh `work_root/run_id`.
2. Resolve `drive_root/spec.input_folder` including symlinks, verify the resolved path remains under `drive_root`, derive output independently, and discover source inventory read-only to obtain input size.
3. Preflight before source copy or GPU work.
4. Copy input locally, rediscover it, and compare inventory.
5. Resolve profile to `SelectionPolicy`; select frames, preserve the source-selection manifest, apply the explicit Task 3 reconstruction guardrail when needed, and write both manifests.
6. Run gated COLMAP with at most one Smart backfill.
7. Set `FOURDGS_DATA_ROOT` before deferred training imports and train only a passing reconstruction.
8. Polish with raw fallback, derive metadata/preview/world assets, and optionally orbit.
9. Build `quality_report.json`, `run_manifest.json`, and `logs/` in a new local bundle.
10. Validate locally and publish transactionally to the exact sibling result.

On source/selection/reconstruction failures, render a selected/rejected contact sheet, component table, and uncovered-interval JSON, write actionable local reports, and call `publish_diagnostics`; on training/polish/metadata/publish failure, preserve any prior result and never create a new success marker.

- [ ] **Step 5: Record complete quality and provenance reports**

`quality_report.json` must include selection mode/metrics/rejections, all COLMAP component metrics and gaps, the 95% registration target warning when applicable, gate decision, metrics explicitly labelled `training_view_checks` unless true holdout evaluation was requested, training summary, polish acceptance/reasons/deltas, orientation evidence, preview/orbit warnings, and capture limitations.

`run_manifest.json` must include `generator_id`, schema `1`, status, full RunSpec, resolved config, source inventory/hashes, selection digest, accepted model, repository commit, Python/CUDA/COLMAP/FFmpeg/PyTorch/gsplat versions, timing per stage, hardware, non-self artifact inventory, and selected profile. Use canonical JSON and `allow_nan=False`.

- [ ] **Step 6: Write failing CLI tests, including malformed-data-before-heavy-import behavior**

```python
def test_cli_parses_spec_and_writes_machine_readable_receipt(tmp_path, monkeypatch):
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text('{"schema_version":1,"input_folder":"captures/room"}', encoding="utf-8")
    monkeypatch.setattr("scripts.notebook_static_run._load_runner", lambda: fake_success)
    assert main(["--spec", str(spec_path)]) == 0
    receipt = json.loads((tmp_path / "run_result.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "success"


def test_cli_rejects_malformed_spec_before_importing_training(tmp_path, monkeypatch):
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text("not-json", encoding="utf-8")
    imported = []
    monkeypatch.setattr("builtins.__import__", tracking_import(imported))
    assert main(["--spec", str(spec_path)]) == 2
    assert "backend.pipeline" not in imported
```

- [ ] **Step 7: Implement a thin CLI with heavy imports after validation**

`build_parser()` exposes only required `--spec Path`. `_load_runner()` imports and returns `backend.static_pipeline.runner.run_static_notebook`. `main()` reads bytes, calls `parse_run_spec_json`, invokes `_load_runner()` only after validation, writes `run_result.json` beside the spec, prints concise JSON, and returns `0`. Validation errors return `2`; runtime errors write `status="failed"` plus error type/message/diagnostics path and return `1`. Never catch an error and return success.

- [ ] **Step 8: Run runner and CLI tests**

Run: `python -m pytest tests/static_pipeline/test_runner.py tests/notebooks/test_runner_cli.py -q`

Expected: all tests PASS and `python scripts/notebook_static_run.py --help` exits `0` without importing Torch.

- [ ] **Step 9: Commit and push Task 12**

```powershell
git add backend/static_pipeline/runner.py scripts/notebook_static_run.py tests/static_pipeline/test_runner.py tests/notebooks/test_runner_cli.py
git status --short
git diff --staged
git commit -m "feat: orchestrate static notebook runs"
git push
```

---

### Task 13: Safe Notebook Builder, Dedicated Bootstrap, and API Endpoints

**Files:**
- Create: `colab/static_notebook_bootstrap.sh`
- Create: `backend/notebooks/static_template.py`
- Create: `backend/notebooks/builder.py`
- Create: `backend/notebooks/source.py`
- Create: `backend/notebooks/routes.py`
- Modify: `backend/api.py`
- Test: `tests/notebooks/test_builder.py`
- Test: `tests/notebooks/test_source.py`
- Test: `tests/notebooks/test_routes.py`

**Interfaces:**
- Consumes: `StaticNotebookRunSpec` plus `NotebookSource(repo_url, commit_sha, generator_id, generator_version)`.
- Produces: `build_static_notebook`, `serialize_notebook`, `notebook_download_filename`, `resolve_notebook_source`, `static_notebook_router`, `GET /notebooks/static/presets`, and `POST /notebooks/static`.

- [ ] **Step 1: Write failing structural, injection, and pin tests**

```python
SOURCE = NotebookSource(
    repo_url="https://github.com/mehmettahacumurcu/gaussian-splatter.git",
    commit_sha="a" * 40,
    generator_id="4dgs-studio.static-notebook",
    generator_version=1,
)


def test_notebook_has_stable_run_all_cells_and_parses():
    spec = StaticNotebookRunSpec(input_folder="captures/room")
    raw = serialize_notebook(build_static_notebook(spec, source=SOURCE))
    notebook = nbformat.reads(raw.decode("utf-8"), as_version=4)
    assert [cell.metadata["tags"][0] for cell in notebook.cells] == [
        "title", "run-spec", "preflight", "drive", "checkout", "bootstrap", "execute", "validate", "summary",
    ]


def test_malicious_folder_is_only_json_data():
    spec = StaticNotebookRunSpec(input_folder="captures/room;__import__('os').system('id')")
    notebook = build_static_notebook(spec, source=SOURCE)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert code.count("room;__import__") == 1
    assert "shell=True" not in code
    assert "git pull" not in code
    assert all(not line.lstrip().startswith(("!", "%")) for line in code.splitlines())
    ast.parse(code)


def test_builder_rejects_non_full_commit_sha():
    with pytest.raises(ValueError):
        build_static_notebook(StaticNotebookRunSpec(input_folder="captures/room"), source=replace(SOURCE, commit_sha="main"))
```

- [ ] **Step 2: Run builder tests and confirm the generator is missing**

Run: `python -m pytest tests/notebooks/test_builder.py -q`

Expected: FAIL with missing builder/template modules.

- [ ] **Step 3: Add a dedicated Run-All-safe bootstrap wrapper**

Keep the three legacy notebooks and `colab/bootstrap.sh` unchanged. `colab/static_notebook_bootstrap.sh` runs `bash colab/bootstrap.sh --colmap-cuda`, then validates NumPy, Torch, CUDA, gsplat, COLMAP, FFmpeg, Pydantic, and nbformat in fresh Python/process invocations under `set -euo pipefail`. It must fail rather than suggest a runtime restart.

- [ ] **Step 4: Build fixed structural cells and embed exactly one canonical JSON value**

Use `nbformat.v4.new_notebook/new_markdown_cell/new_code_cell`. Build the RunSpec cell with canonical JSON represented as one Python string literal and write `/content/run_spec.json`; no user value appears in any other cell:

```python
canonical_json = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)
run_spec_source = (
    "import json\n"
    f"RUN_SPEC_JSON = {canonical_json!r}\n"
    "RUN_SPEC = json.loads(RUN_SPEC_JSON)\n"
    "with open('/content/run_spec.json', 'w', encoding='utf-8') as handle:\n"
    "    json.dump(RUN_SPEC, handle, sort_keys=True, separators=(',', ':'))\n"
)
```

```python
@dataclass(frozen=True)
class NotebookSource:
    repo_url: str
    commit_sha: str
    generator_id: str
    generator_version: int


def build_static_notebook(spec: StaticNotebookRunSpec, *, source: NotebookSource) -> nbformat.NotebookNode:
    if not re.fullmatch(r"[0-9a-f]{40}", source.commit_sha):
        raise ValueError("Notebook source requires a full 40-character commit SHA")
    return nbformat.v4.new_notebook(cells=build_static_cells(spec, source))


def serialize_notebook(notebook: nbformat.NotebookNode) -> bytes:
    return nbformat.writes(notebook, version=4).encode("utf-8")
```

The cells perform, in order:

1. title/instructions;
2. safe RunSpec data write;
3. subprocess-based GPU/VRAM/disk/Python preflight without importing Torch/NumPy into the notebook kernel;
4. `google.colab.drive.mount('/content/drive')` and canonical path display;
5. delete only a fixed `/content/gaussian-splatter-src`, clone the fixed repo URL, and `git checkout --detach <40-hex SHA>` with checked argv;
6. run `bash colab/static_notebook_bootstrap.sh` in the checkout with `check=True`;
7. run `[sys.executable, 'scripts/notebook_static_run.py', '--spec', '/content/run_spec.json']` with `check=True`;
8. read `/content/run_result.json`, require status success, verify the exact result path and `_SUCCESS`;
9. print result path, PLY, preview, quality report, and SuperSplat/our-viewer guidance without embedding a viewer.

- [ ] **Step 5: Write failing remote-source tests**

```python
def test_source_uses_full_head_only_when_upstream_contains_it(fake_git):
    fake_git.head = "b" * 40
    fake_git.upstream_contains_head = True
    source = resolve_notebook_source(git=fake_git)
    assert source.commit_sha == "b" * 40
    assert source.repo_url == "https://github.com/mehmettahacumurcu/gaussian-splatter.git"


def test_source_rejects_unpushed_head(fake_git):
    fake_git.head = "b" * 40
    fake_git.upstream_contains_head = False
    with pytest.raises(NotebookSourceError, match="push"):
        resolve_notebook_source(git=fake_git)
```

- [ ] **Step 6: Resolve a remotely reachable full SHA without a self-referential constant**

`resolve_notebook_source` uses the fixed HTTPS repo URL, accepts optional deployment `STATIC_NOTEBOOK_COMMIT_SHA`, otherwise reads `git rev-parse HEAD`, requires exactly 40 lowercase hex characters, resolves `@{u}`, and runs `git merge-base --is-ancestor <sha> <upstream>`. Failure tells the operator to push or configure a reachable release SHA. This resolves the release-pin problem: the generated notebook contains one immutable full SHA, while the generator refuses a local unpushed commit.

- [ ] **Step 7: Write failing route tests against an isolated FastAPI app**

```python
def test_presets_route_is_backend_owned(client):
    response = client.get("/notebooks/static/presets")
    assert response.status_code == 200
    assert response.json()["default_profile"] == "balanced_l4"
    assert [profile["id"] for profile in response.json()["profiles"]] == ["balanced_l4", "high", "premium", "ultra"]


def test_generate_route_returns_attachment_without_job_manager(client, mocker):
    manager = mocker.patch("backend.job_manager.get_manager")
    response = client.post("/notebooks/static", json={"schema_version": 1, "input_folder": "captures/room"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ipynb+json")
    assert response.headers["content-disposition"] == 'attachment; filename="room_static_splat.ipynb"'
    manager.assert_not_called()
```

- [ ] **Step 8: Implement the isolated router and tightly include it in the application**

`routes.py` uses `APIRouter(prefix="/notebooks/static", tags=["notebooks"])`. GET returns `get_static_preset_manifest()`. POST accepts `StaticNotebookRunSpec`, calls `resolve_notebook_source()`, builds the notebook, and returns raw bytes via `Response` with media type `application/x-ipynb+json` and a sanitized ASCII attachment filename from the canonical leaf.

Test the router by including it in a tiny test-only FastAPI instance so tests do not import Torch or start `JobManager`. Task 0 leaves `backend/api.py` clean; make only two tight additions: import `static_notebook_router`, then call `app.include_router(static_notebook_router)` immediately after app construction. Review the full diff before staging.

- [ ] **Step 9: Run builder, source, route, and preset compatibility tests**

Run: `python -m pytest tests/notebooks/test_builder.py tests/notebooks/test_source.py tests/notebooks/test_routes.py tests/test_api_static_presets.py -q`

Expected: all tests PASS; invalid bodies return `422`; POST never calls `/process` or `JobManager`.

- [ ] **Step 10: Commit and push Task 13**

```powershell
git add colab/static_notebook_bootstrap.sh backend/notebooks/static_template.py backend/notebooks/builder.py backend/notebooks/source.py backend/notebooks/routes.py backend/api.py tests/notebooks/test_builder.py tests/notebooks/test_source.py tests/notebooks/test_routes.py
git status --short
git diff --staged
git commit -m "feat: generate pinned static splat notebooks"
git push
```

---

### Task 14: Frontend Notebook Domain, Validation, Reducer, and API Client

**Files:**
- Create: `frontend/src/notebook/types.ts`
- Create: `frontend/src/notebook/drivePath.ts`
- Create: `frontend/src/notebook/reducer.ts`
- Modify: `frontend/src/api.ts`
- Test: `frontend/src/notebook/__tests__/drivePath.test.ts`
- Test: `frontend/src/notebook/__tests__/reducer.test.ts`
- Test: `frontend/src/api.notebook.test.ts`

**Interfaces:**
- Consumes: backend preset manifest and the Task 1 JSON contract.
- Produces: frontend RunSpec/profile types, `resolveDriveFolder`, `notebookReducer`, `validateNotebookDraft`, `buildStaticNotebookRunSpec`, `getStaticNotebookPresets`, and `generateStaticNotebook`.

- [ ] **Step 1: Write failing Drive-path parity tests**

```typescript
import { describe, expect, it } from "vitest";
import { resolveDriveFolder } from "../drivePath";

describe("resolveDriveFolder", () => {
  it.each(["captures/myroom", "MyDrive/captures/myroom"])("normalizes %s", (raw) => {
    expect(resolveDriveFolder(raw)).toEqual({
      ok: true,
      canonicalInput: "captures/myroom",
      displayInput: "MyDrive/captures/myroom",
      displayOutput: "MyDrive/captures/myroom_result",
    });
  });

  it.each(["", "MyDrive", "/content/drive/MyDrive/x", "C:/x", "a\\b", "a/../b", "a/./b", "a//b", "a/x_result"])("rejects %s", (raw) => {
    expect(resolveDriveFolder(raw).ok).toBe(false);
  });
});
```

- [ ] **Step 2: Run and confirm the notebook frontend domain is missing**

Run: `npm --prefix frontend test -- src/notebook/__tests__/drivePath.test.ts`

Expected: FAIL with unresolved `../drivePath`.

- [ ] **Step 3: Add exact shared frontend types**

Create `types.ts` with:

```typescript
export type FrameSelectionMode = "smart" | "fixed_fps";
export type NotebookQualityProfileId = "balanced_l4" | "high" | "premium" | "ultra";
export type MultiresStage = [startIter: number, longEdge: number];

export interface NotebookAdvancedValues {
  run_eval: boolean;
  foundation: boolean;
  resolution_long_edge_cap: number;
  lambda_ssim: number;
  lambda_lpips: number;
  lambda_depth: number;
  density_start_iter: number;
  density_end_iter: number;
  density_interval: number;
  densify_grad_threshold: number;
  prune_min_opacity: number;
  prune_max_scale: number;
  opacity_reset_interval: number;
  sh_degree: number;
  multires_schedule: MultiresStage[];
}

export type NotebookAdvancedConfig = Partial<NotebookAdvancedValues>;

export interface StaticNotebookRunSpec {
  schema_version: 1;
  input_folder: string;
  frame_selection: { mode: FrameSelectionMode; fixed_fps: number };
  quality: {
    profile: NotebookQualityProfileId;
    n_iters: number | null;
    max_gaussians: number | null;
    advanced: NotebookAdvancedConfig;
  };
  publish: { replace_owned_result: true };
}

export interface NotebookProfileMetadata {
  id: NotebookQualityProfileId;
  label: string;
  description: string;
  n_iters: number;
  max_gaussians: number;
  selected_frame_budget: number;
  resolution_long_edge_cap: number | null;
  intended_gpu: string;
  minimum_vram_gb: number;
  foundation_default: boolean;
  run_eval_default: boolean;
  warnings: string[];
}

export interface StaticNotebookPresetsResponse {
  schema_version: 1;
  default_profile: NotebookQualityProfileId;
  profiles: NotebookProfileMetadata[];
  override_limits: {
    fixed_fps: { min: number; max: number };
    n_iters: { min: number; max: number };
    max_gaussians: { min: number; max: number };
  };
}

export interface GeneratedNotebook { blob: Blob; filename: string }
```

- [ ] **Step 4: Implement frontend Drive-path parity**

`resolveDriveFolder(raw)` mirrors the backend rejection/normalization rules and returns either the exact success object in Step 1 or `{ok:false,error:string}`. Do not accept a browser-local file picker path; this field names an existing Drive folder.

```typescript
export type DriveFolderResolution =
  | { ok: true; canonicalInput: string; displayInput: string; displayOutput: string }
  | { ok: false; error: string };

export function resolveDriveFolder(raw: string): DriveFolderResolution;
```

- [ ] **Step 5: Write failing reducer/default/serialization tests**

```typescript
it("defaults to Smart, FPS 4, and backend Balanced/L4", () => {
  const state = initialNotebookGeneratorState();
  expect(state.draft.frameSelection).toEqual({ mode: "smart", fixedFps: "4" });
  expect(state.draft.quality.profile).toBe("balanced_l4");
  expect(state.draft.quality.nIters).toBe("");
});

it("keeps explicit overrides across profile changes and reset clears them", () => {
  let state = readyState();
  state = notebookReducer(state, { type: "iterations_changed", value: "42000" });
  state = notebookReducer(state, { type: "profile_changed", value: "high" });
  expect(state.draft.quality.nIters).toBe("42000");
  state = notebookReducer(state, { type: "primary_overrides_reset" });
  expect(state.draft.quality.nIters).toBe("");
  expect(buildStaticNotebookRunSpec(state.draft, PRESETS).quality.n_iters).toBeNull();
});

it("compacts empty advanced fields and parses multires schedule", () => {
  const draft = draftWithAdvanced({ lambda_ssim: "0.25", multires_schedule: "0:720,25000:1080" });
  expect(buildStaticNotebookRunSpec(draft, PRESETS).quality.advanced).toEqual({
    lambda_ssim: 0.25,
    multires_schedule: [[0, 720], [25000, 1080]],
  });
});

it("any edit invalidates a generated artifact", () => {
  const state = notebookReducer(generatedState(), { type: "input_changed", value: "captures/new" });
  expect(state.generation).toEqual({ status: "idle" });
});
```

- [ ] **Step 6: Implement draft/reducer/validation contracts with string numeric inputs**

Use `NotebookDraft`, `NotebookAdvancedDraft`, `PresetsState`, `GenerationState`, and `NotebookGeneratorState` exactly as described in the design spec. Editable numerics stay strings so blank and partial input remain stable. Initial mode is Smart, fixed FPS is string `"4"`, profile is `balanced_l4`, primary overrides are empty, all advanced strings are empty, boolean advanced fields are `null`, and generation is idle.

```typescript
export interface NotebookAdvancedDraft {
  run_eval: boolean | null;
  foundation: boolean | null;
  resolution_long_edge_cap: string;
  lambda_ssim: string;
  lambda_lpips: string;
  lambda_depth: string;
  density_start_iter: string;
  density_end_iter: string;
  density_interval: string;
  densify_grad_threshold: string;
  prune_min_opacity: string;
  prune_max_scale: string;
  opacity_reset_interval: string;
  sh_degree: string;
  multires_schedule: string;
}

export interface NotebookDraft {
  inputFolderRaw: string;
  frameSelection: { mode: FrameSelectionMode; fixedFps: string };
  quality: {
    profile: NotebookQualityProfileId;
    nIters: string;
    maxGaussians: string;
    advanced: NotebookAdvancedDraft;
  };
}

export type PresetsState =
  | { status: "loading" }
  | { status: "ready"; value: StaticNotebookPresetsResponse }
  | { status: "error"; message: string };

export type GenerationState =
  | { status: "idle" }
  | { status: "generating" }
  | { status: "ready"; artifact: GeneratedNotebook }
  | { status: "error"; message: string };

export interface NotebookGeneratorState {
  draft: NotebookDraft;
  presets: PresetsState;
  generation: GenerationState;
}

export interface NotebookDraftValidation {
  valid: boolean;
  path: DriveFolderResolution;
  errors: Record<string, string>;
  warnings: string[];
}
```

Reducer actions cover preset loading/success/failure, every draft mutation, primary reset, generation start/success/failure, edit, and regenerate. Every draft mutation resets a ready/error artifact to idle. Preset load failure blocks generation and supports retry. Validation uses backend-supplied override bounds and parses multires syntax `start:edge,start:edge`; it does not duplicate profile values.

- [ ] **Step 7: Write failing raw-response API tests**

```typescript
it("posts only JSON to the notebook endpoint and parses filename star", async () => {
  server.respondWith("/notebooks/static", new Response(new Blob(["{}"]), {
    status: 200,
    headers: { "Content-Disposition": "attachment; filename*=UTF-8''room%20scan.ipynb" },
  }));
  const artifact = await generateStaticNotebook(SPEC);
  expect(server.lastRequest.url).toEndWith("/notebooks/static");
  expect(server.lastRequest.init.method).toBe("POST");
  expect(server.lastRequest.init.body).toBe(JSON.stringify(SPEC));
  expect(artifact.filename).toBe("room scan.ipynb");
});
```

- [ ] **Step 8: Implement preset GET and notebook attachment POST**

Append to `frontend/src/api.ts` without changing `submitJob()`:

```typescript
export async function getStaticNotebookPresets(): Promise<StaticNotebookPresetsResponse> {
  return fetchJson<StaticNotebookPresetsResponse>("/notebooks/static/presets");
}

export async function generateStaticNotebook(spec: StaticNotebookRunSpec): Promise<GeneratedNotebook> {
  const response = await fetch(`${getApiBase()}/notebooks/static`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json", Accept: "application/x-ipynb+json" }),
    body: JSON.stringify(spec),
  });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`HTTP ${response.status} ${response.statusText}: ${body || response.url}`);
  }
  return { blob: await response.blob(), filename: parseAttachmentFilename(response.headers.get("Content-Disposition")) ?? "static_splat.ipynb" };
}
```

`parseAttachmentFilename` handles quoted `filename`, RFC 5987 `filename*`, strips path separators/control characters, and returns null for empty values.

- [ ] **Step 9: Run frontend domain/API tests**

Run: `npm --prefix frontend test -- src/notebook/__tests__/drivePath.test.ts src/notebook/__tests__/reducer.test.ts src/api.notebook.test.ts`

Expected: all tests PASS.

- [ ] **Step 10: Commit and push Task 14**

```powershell
git add frontend/src/notebook/types.ts frontend/src/notebook/drivePath.ts frontend/src/notebook/reducer.ts frontend/src/api.ts frontend/src/notebook/__tests__/drivePath.test.ts frontend/src/notebook/__tests__/reducer.test.ts frontend/src/api.notebook.test.ts
git status --short
git diff --staged
git commit -m "feat: add notebook generator frontend state"
git push
```

---

### Task 15: Top-to-Bottom Notebook Wizard and Legacy Local-Job Preservation

**Required implementation skill:** Announce and use `frontend-design:frontend-design` before changing UI/CSS.

**Files:**
- Create: `frontend/src/notebook/NotebookGeneratorPanel.tsx`
- Create: `frontend/src/notebook/DriveFolderSection.tsx`
- Create: `frontend/src/notebook/FrameSelectionSection.tsx`
- Create: `frontend/src/notebook/QualitySection.tsx`
- Create: `frontend/src/notebook/NotebookAdvancedConfig.tsx`
- Create: `frontend/src/notebook/NotebookReview.tsx`
- Create: `frontend/src/notebook/NotebookDownload.tsx`
- Create: `frontend/src/notebook/notebook.css`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/src/notebook/__tests__/NotebookGeneratorPanel.test.tsx`
- Test: `frontend/src/notebook/__tests__/NotebookDownload.test.tsx`
- Test: `frontend/src/App.test.tsx`

**Interfaces:**
- Consumes: Task 14 reducer/helpers/API.
- Produces: a default Notebook tab that generates/downloads an artifact without job navigation, plus an unchanged Local Job tab rendering the existing `JobSubmitPanel`.

- [ ] **Step 1: Lock the specific design direction before JSX**

Subject: a technical creator turning a raw capture into a portable spatial asset. Single job: move confidently from Drive folder to downloadable Run-All notebook.

Use this compact token system in `notebook.css`, scoped under `.notebook-generator`:

```css
--nb-canvas: #0d1117;
--nb-panel: #151b23;
--nb-line: #2b3747;
--nb-text: #edf3fa;
--nb-cobalt: #5b91ff;
--nb-signal: #62d5c4;
```

Use `"IBM Plex Sans", "Segoe UI", sans-serif` for headings/body and `"JetBrains Mono", "Cascadia Code", monospace` for paths/data. Derive muted and warning colors from the six scoped tokens plus the existing application warning token. The signature element is one vertical **capture-to-splat signal rail** connecting the genuinely ordered sections; selected/completed stages illuminate cyan. Profile choices form a compact **VRAM ladder**, with bar lengths reflecting backend frame/Gaussian budgets instead of generic equal cards. Keep all other decoration restrained.

Desktop/mobile structure:

```text
+ Notebook: capture to splat --------------------------------------+
|  o  Google Drive input       [captures/myroom                 ] |
|  |  Output                   MyDrive/captures/myroom_result      |
|  o  Frame selection          [Smart recommended] [Fixed FPS]    |
|  o  Quality / VRAM ladder    Balanced | High | Premium | Ultra  |
|  o  Overrides                Iterations [default]  Gauss [default]|
|  o  Advanced                 [collapsed approved controls]       |
|  o  Review                   paths + effective config + warnings |
|  *  Generate notebook        [Generate and download notebook]    |
+-----------------------------------------------------------------+
```

At narrow widths the rail remains at the left and every control becomes one column. Respect visible keyboard focus and `prefers-reduced-motion`; use one short rail-state transition only.

- [ ] **Step 2: Write failing top-to-bottom behavior tests**

```typescript
it("renders the approved order with Smart and Balanced/L4 defaults", async () => {
  render(<NotebookGeneratorPanel />);
  await screen.findByText("Balanced / L4");
  const headings = screen.getAllByRole("heading", { level: 2 }).map((node) => node.textContent);
  expect(headings).toEqual(["Google Drive input folder", "Frame selection", "Quality profile", "Iterations and maximum Gaussians", "Advanced settings", "Review"]);
  expect(screen.getByRole("radio", { name: /Smart/ })).toBeChecked();
  expect(screen.getByRole("radio", { name: /Balanced \/ L4/ })).toBeChecked();
  expect(screen.queryByLabelText("Frames per second")).not.toBeInTheDocument();
});

it("shows FPS only for baseline and submits the exact RunSpec", async () => {
  const user = userEvent.setup();
  render(<NotebookGeneratorPanel />);
  await user.type(await screen.findByLabelText("Drive folder"), "captures/myroom");
  await user.click(screen.getByRole("radio", { name: /Fixed FPS/ }));
  expect(screen.getByLabelText("Frames per second")).toHaveValue(4);
  await user.click(screen.getByRole("button", { name: "Generate and download notebook" }));
  expect(generateStaticNotebook).toHaveBeenCalledWith(EXPECTED_FIXED_SPEC);
});

it("invalid input blocks generation and explains the correction", async () => {
  render(<NotebookGeneratorPanel />);
  await userEvent.type(await screen.findByLabelText("Drive folder"), "../room");
  expect(screen.getByRole("button", { name: "Generate and download notebook" })).toBeDisabled();
  expect(screen.getByRole("alert")).toHaveTextContent("relative folder");
});
```

- [ ] **Step 3: Implement focused section components with the locked signatures**

Use these prop boundaries:

```typescript
export function NotebookGeneratorPanel(): JSX.Element;
export function DriveFolderSection(props: { value: string; resolution: DriveFolderResolution; onChange(value: string): void }): JSX.Element;
export function FrameSelectionSection(props: { mode: FrameSelectionMode; fixedFps: string; fixedFpsError?: string; onModeChange(mode: FrameSelectionMode): void; onFixedFpsChange(value: string): void }): JSX.Element;
export function QualitySection(props: { profiles: readonly NotebookProfileMetadata[]; value: NotebookDraft["quality"]; errors: Partial<Record<"profile" | "nIters" | "maxGaussians", string>>; onProfileChange(profile: NotebookQualityProfileId): void; onIterationsChange(value: string): void; onMaxGaussiansChange(value: string): void; onResetOverrides(): void }): JSX.Element;
export function NotebookAdvancedConfig(props: { value: NotebookAdvancedDraft; errors: Partial<Record<keyof NotebookAdvancedDraft, string>>; onChange<K extends keyof NotebookAdvancedDraft>(key: K, value: NotebookAdvancedDraft[K]): void }): JSX.Element;
export function NotebookReview(props: { spec: StaticNotebookRunSpec | null; profile: NotebookProfileMetadata | null; path: DriveFolderResolution; warnings: readonly string[]; generating: boolean; onGenerate(): void }): JSX.Element;
```

Advanced settings use a native `<details>` collapsed by default and expose only the approved allowlist. Review shows exact MyDrive input/output, Smart/fixed mode, effective defaults/overrides, GPU warning, no-web-viewer note, and one consistent `Generate and download notebook` action.

- [ ] **Step 4: Implement reducer/API lifecycle and actionable errors**

`NotebookGeneratorPanel` loads presets on mount, offers `Retry profile loading`, validates on every edit, and builds the RunSpec only when valid. Generation errors remain on the same Notebook page with a specific retry action. A successful artifact switches only the panel's state; it does not accept `onJobSubmitted`, call `/process`, or change tabs.

Within `QualitySection`, render `Quality profile` and `Iterations and maximum Gaussians` as two consecutive level-2 stage headings so the visual/semantic order matches the approved flow even though one focused component owns both profile metadata and its two primary overrides.

- [ ] **Step 5: Write failing Blob URL lifecycle tests**

```typescript
it("downloads once automatically, supports download again, and revokes URLs", () => {
  const create = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:notebook");
  const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
  const { rerender, unmount } = render(<NotebookDownload artifact={FIRST} onRegenerate={vi.fn()} onEdit={vi.fn()} />);
  expect(create).toHaveBeenCalledWith(FIRST.blob);
  expect(downloadClick).toHaveBeenCalledWith("blob:notebook", FIRST.filename);
  rerender(<NotebookDownload artifact={SECOND} onRegenerate={vi.fn()} onEdit={vi.fn()} />);
  expect(revoke).toHaveBeenCalledWith("blob:notebook");
  unmount();
  expect(revoke).toHaveBeenCalledTimes(2);
});
```

- [ ] **Step 6: Implement download ownership and regenerate/edit actions**

`NotebookDownload` alone owns `URL.createObjectURL`. It triggers one automatic download for each new artifact, renders filename plus `Download again`, `Regenerate`, and `Edit settings`, and revokes the old URL on artifact replacement and unmount. `Edit settings` returns to review without discarding form values; any subsequent form mutation invalidates the old artifact.

- [ ] **Step 7: Add the Notebook tab while preserving Local Job**

Change `Tab` to include `"notebook"`, default state to `"notebook"`, and render tab order `Notebook`, `Local Job`, `Jobs`, `Viewer`, `Analiz`, `Eval`, `Interactive`. `Local Job` retains the existing `"submit"` value and exact `<JobSubmitPanel onJobSubmitted={handleJobSubmitted} />`; do not modify `JobSubmitPanel.tsx` or `Static3DSubmit.tsx` in this task.

Add `App.test.tsx` assertions that Notebook is default, Local Job still renders the mocked legacy panel, a generated notebook never switches to Jobs, and legacy job submission still does.

- [ ] **Step 8: Run component, App, full frontend, and build checks**

Run:

```powershell
npm --prefix frontend test -- src/notebook/__tests__/NotebookGeneratorPanel.test.tsx src/notebook/__tests__/NotebookDownload.test.tsx src/App.test.tsx
npm --prefix frontend test
npm --prefix frontend run build
```

Expected: all tests PASS and TypeScript/Vite build succeeds.

- [ ] **Step 9: Visually inspect desktop and narrow layouts**

Start the frontend, use the in-app browser at desktop and 390 px width, capture screenshots, and verify section order, rail alignment, focus states, collapsed advanced panel, error copy, loading, successful download state, and Local Job accessibility. Revise only scoped `notebook.css`; do not destabilize viewer/job styles.

- [ ] **Step 10: Commit and push Task 15**

```powershell
git add frontend/src/notebook/NotebookGeneratorPanel.tsx frontend/src/notebook/DriveFolderSection.tsx frontend/src/notebook/FrameSelectionSection.tsx frontend/src/notebook/QualitySection.tsx frontend/src/notebook/NotebookAdvancedConfig.tsx frontend/src/notebook/NotebookReview.tsx frontend/src/notebook/NotebookDownload.tsx frontend/src/notebook/notebook.css frontend/src/notebook/__tests__/NotebookGeneratorPanel.test.tsx frontend/src/notebook/__tests__/NotebookDownload.test.tsx frontend/src/App.tsx frontend/src/App.test.tsx
git status --short
git diff --staged
git commit -m "feat: add top-to-bottom notebook generator"
git push
```

---

### Task 16: Release Reachability, Full Regression, Generated-Notebook Smoke, and Operator Docs

**Files:**
- Create: `tests/integration/test_generated_notebook_smoke.py`
- Create: `docs/STATIC_NOTEBOOK.md`
- Test: all focused and regression suites from prior tasks.

**Interfaces:**
- Consumes: the fully implemented and pushed feature-branch HEAD.
- Produces: reproducible full-SHA smoke evidence, user/operator instructions, and explicit hardware-validation commands.

- [ ] **Step 1: Push the current implementation before the notebook smoke**

Run:

```powershell
git status --short --branch
git push
git merge-base --is-ancestor HEAD '@{u}'
if ($LASTEXITCODE -ne 0) { throw 'HEAD is not reachable from upstream' }
```

Expected: clean intentional work, successful push, and reachability exit code `0`.

- [ ] **Step 2: Add a generated-notebook smoke with mocked Drive/process boundaries**

The integration test generates a notebook from a malicious-but-valid folder string, parses it with nbformat, executes safe structural cells in a temporary namespace with mocked `google.colab.drive.mount` and `subprocess.run`, verifies the exact CLI argv, writes a fake successful receipt/folder, and confirms the validation/summary cells complete. It also asserts the three legacy notebooks have no git diff.

- [ ] **Step 3: Document the exact user and operator flow**

`docs/STATIC_NOTEBOOK.md` covers:

1. accepted Drive layouts and path examples;
2. Smart versus Fixed FPS comparison meaning;
3. all four backend-sourced profiles and L4 default;
4. Generate, connect GPU runtime, Run all, grant Drive access;
5. exact `<input>_result` inventory and standard PLY/SuperSplat usage;
6. diagnostics location and each COLMAP gate failure;
7. replacement ownership/backup behavior;
8. no bundled viewer, no fake sky geometry, and our viewer metadata;
9. operator release-SHA configuration and Colab dependency policy;
10. L4 and A100 validation commands without claiming unrun hardware evidence.

- [ ] **Step 4: Run the complete non-GPU verification matrix**

Run:

```powershell
python -m pytest tests/notebooks tests/static_pipeline tests/preprocess tests/image_to_scene tests/test_api_static_presets.py tests/test_static_trainer_init.py -m "not gpu and not integration" -q
python -m pytest tests/integration/test_generated_notebook_smoke.py -q
python -m pytest -m "not gpu and not integration" -q
python scripts/notebook_static_run.py --help
npm --prefix frontend test
npm --prefix frontend run build
git diff --exit-code -- colab/video_to_world.ipynb colab/sota_verify.ipynb colab/phase2_verify.ipynb
```

Expected: every command exits `0`; all legacy notebooks are unchanged.

- [ ] **Step 5: Add but do not falsely claim real hardware acceptance**

On an NVIDIA L4 24 GB runtime, run Balanced/L4 twice against the same input, once Smart and once Fixed FPS 4, and record peak VRAM, selected frames, registration/gaps, timing, polish decision, and training-view metrics. Acceptance target is peak VRAM below roughly 20 GB. On A100 80 GB, run at least one Premium or Ultra validation. These commands are documented and marked manual/hardware; if this implementation session lacks those GPUs, report them as pending user validation rather than passing.

- [ ] **Step 6: Verify portable artifacts**

Open the resulting standard INRIA `splat.ply` in SuperSplat, load `scene_metadata.json`/optional `world/` in this project's viewer, verify `preview.png`, and confirm the result contains no HTML/JS/WASM viewer. Record evidence in the quality report or release notes, not in code comments.

- [ ] **Step 7: Commit and push Task 16**

```powershell
git add tests/integration/test_generated_notebook_smoke.py docs/STATIC_NOTEBOOK.md
git status --short
git diff --staged
git commit -m "test: verify static notebook release workflow"
git push
```

---

## Final Acceptance Checklist

- [ ] Frontend produces a valid notebook from a MyDrive-relative folder; Smart and Balanced/L4 are selected by default.
- [ ] Fixed FPS 4 changes only selection/backfill behavior and both modes record their effective selection.
- [ ] The 285/502 fragmented fixture cannot call training; the 95/100 healthy fixture can.
- [ ] No stale frame tail, database, sparse component, or positional pose/depth pairing can survive into a new accepted run.
- [ ] Successful Run-All publishes exactly one verified `<input_folder_name>_result` with standard `splat.ply`, metadata, preview, reports, logs, and `_SUCCESS`.
- [ ] Failed runs preserve any prior owned result and may write diagnostics only under `<input>_result_diagnostics/<run_id>`.
- [ ] Polish never replaces raw output when any exact geometry, opacity, PSNR, or SSIM limit fails.
- [ ] Orientation is omitted below 0.5 plane inliers or above 20 degrees camera/plane disagreement; sky remains metadata only.
- [ ] Generated notebooks pin a remotely reachable 40-character commit and use checked argv subprocesses with no restart or mutable pull.
- [ ] Existing notebooks, `/process`, JobManager, Static3DSubmit, viewer, and local job navigation remain available.
- [ ] Non-GPU tests and frontend build pass locally; L4/A100 items are reported honestly as run or pending.

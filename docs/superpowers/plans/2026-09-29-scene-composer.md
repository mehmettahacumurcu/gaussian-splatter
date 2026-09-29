# Scene Composer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users place extra splats (`.ply`) and meshes (`.glb`) into a trained splat scene with a gizmo, crop / colour-match / snap them, save the scene, and export one aligned `merged.ply` + meshes.

**Architecture:** Browser preview with Spark + R3F (`frontend/src/compose/`), numpy bake on the backend (`backend/compose/`), a JSON `SceneDoc` shared by both, and a golden fixture (`tests/fixtures/compose_golden.json`) that both test suites check so preview and bake cannot drift.

**Tech Stack:** FastAPI, pydantic v2, numpy, plyfile, pytest (conda env `gs4d`); React 19, three 0.184, @react-three/fiber 9, @react-three/drei 10, @sparkjsdev/spark 0.1.10, vitest.

Spec: `docs/superpowers/specs/2026-09-29-scene-composer-design.md`

**Commands** (run from `4dgs-studio/`):

- Backend tests: `E:/anaconda3/envs/gs4d/python.exe -m pytest tests/compose -q`
- Frontend tests: `cd frontend && npx vitest run src/compose`
- Frontend typecheck: `cd frontend && npx tsc --noEmit -p tsconfig.json`

**Spike results (Task 0, done while planning, from Spark 0.1.10 source):**

- A `SplatEdit` whose ancestor is a `SplatMesh` is a mesh-local edit (`compileScene` only collects edits without a `SplatMesh` ancestor as global). ✔
- `recolor` and `objectModifier` act on the colour *after* SH has been added, i.e. on the total view-dependent colour. `objectModifier` runs before `recolor`. ✔ Preview applies the whole colour matrix `M` in one `objectModifier` (a `DynoMat3` uniform) so order matches the bake.
- BOX SDF uses `abs(sdfPos) - sizes.xyz` with `sizes = sdf.scale` → `SplatEditSdf.scale` is the **half size**. SDF coords come from `sdf.matrixWorld`, so an SDF child of the `SplatMesh` lives in the splat's raw local frame. ✔
- `SplatMesh.raycast` works but ignores edits → crop filtering of hits is done by us.
- After changing a uniform used by `objectModifier`, call `mesh.updateVersion()`.

---

## File structure

Backend (new package `backend/compose/`):

| File | Responsibility |
|---|---|
| `__init__.py` | package docstring |
| `plyio.py` | `GaussianCloud`, header parse/validate, read/write standard 3DGS PLY |
| `sh_rotation.py` | SH basis (3DGS constants), per-band rotation matrices, `rotate_sh_rest`, `eval_sh_color` |
| `bake.py` | `Placement/Crop/ColorAdjust`, quaternion helpers, crop, colour matrix, `transform_cloud`, `merge_clouds` |
| `glb.py` | GLB validate, `placement_matrix`, `wrap_with_transform` |
| `models.py` | pydantic `SceneDoc` & friends |
| `store.py` | `ComposeStore` disk layout, safe ids, uploads, scenes |
| `exporter.py` | `placement_for`, `missing_assets`, `run_export` job body |
| `routes.py` | `build_compose_router(data_root, get_manager)` |

Modified: `backend/api.py` (router + CORS PUT), `backend/job_manager.py` (compose phases, unknown phases, download_url).

Frontend (new folder `frontend/src/compose/`):

| File | Responsibility |
|---|---|
| `types.ts` | TS mirror of the SceneDoc / Asset models |
| `sceneDoc.ts` | reducer + `newObjectId` |
| `transformMath.ts` | transform/crop helpers shared by preview, snap and tests |
| `colorMath.ts` | colour matrix (same formula as backend) + hex helpers |
| `composeApi.ts` | `/compose/*` client |
| `registry.ts` | `ObjectRegistry` – live three.js objects per scene object |
| `SplatObject.tsx`, `MeshObject.tsx` | scene object renderers |
| `ComposeViewport.tsx` | Canvas, Spark, OrbitControls, gizmo |
| `snap.ts` | snap-to-ground |
| `ObjectListPanel.tsx`, `InspectorPanel.tsx`, `AssetPicker.tsx` | panels |
| `ComposePage.tsx`, `compose.css` | tab root + styles |

Modified: `frontend/src/api.ts` (export `fetchJson`, `authHeaders`), `frontend/src/App.tsx` (tab), `frontend/src/App.test.tsx`, `frontend/src/components/JobsList.tsx`.

Tests: `tests/compose/{__init__.py,helpers.py,golden.py,test_plyio.py,test_sh_rotation.py,test_bake.py,test_glb.py,test_golden.py,test_routes.py,test_job_manager_compose.py,test_gpu_parity.py}`, `tests/fixtures/compose_golden.json`, `frontend/src/compose/__tests__/*.test.ts`.

---

### Task 1: PLY I/O

**Files:**
- Create: `backend/compose/__init__.py`, `backend/compose/plyio.py`
- Create: `tests/compose/__init__.py`, `tests/compose/helpers.py`, `tests/compose/test_plyio.py`

- [ ] **Step 1: Write helpers and failing tests**

`tests/compose/__init__.py`: empty file.

`tests/compose/helpers.py`:

```python
"""Shared builders for scene-composer tests."""
from __future__ import annotations

import json
import struct

import numpy as np

from backend.compose.plyio import GaussianCloud


def random_cloud(n: int, degree: int = 3, seed: int = 0) -> GaussianCloud:
    rng = np.random.default_rng(seed)
    k = (degree + 1) ** 2 - 1
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return GaussianCloud(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        log_scales=rng.normal(-3.0, 0.5, size=(n, 3)).astype(np.float32),
        quats=quats.astype(np.float32),
        opacities=rng.normal(size=n).astype(np.float32),
        sh_dc=rng.normal(0.0, 0.5, size=(n, 3)).astype(np.float32),
        sh_rest=rng.normal(0.0, 0.2, size=(n, k, 3)).astype(np.float32),
    )


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    from backend.compose.bake import quat_to_matrix

    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return quat_to_matrix(tuple(q))


def minimal_glb(gltf: dict | None = None, bin_payload: bytes = b"\x01\x02\x03\x04") -> bytes:
    gltf = gltf or {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "n0", "children": [1]}, {"name": "n1"}],
    }
    js = json.dumps(gltf).encode("utf-8")
    js += b" " * (-len(js) % 4)
    bin_payload += b"\x00" * (-len(bin_payload) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_payload)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(js), 0x4E4F534A) + js
        + struct.pack("<II", len(bin_payload), 0x004E4942) + bin_payload
    )
```

`tests/compose/test_plyio.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from backend.compose.plyio import PlyFormatError, read_ply, read_ply_header, validate_ply, write_ply
from tests.compose.helpers import random_cloud


def test_round_trip_preserves_every_field(tmp_path):
    cloud = random_cloud(50, degree=3, seed=1)
    path = write_ply(cloud, tmp_path / "a.ply")
    back = read_ply(path)
    for field in ("means", "log_scales", "quats", "opacities", "sh_dc", "sh_rest"):
        np.testing.assert_array_equal(getattr(back, field), getattr(cloud, field))
    assert back.sh_degree == 3


def test_header_reports_count_and_fields(tmp_path):
    path = write_ply(random_cloud(7, degree=3), tmp_path / "a.ply")
    count, props = read_ply_header(path)
    assert count == 7
    assert "f_rest_44" in props and "rot_3" in props


def test_reads_spirula_style_file_with_extra_fields_and_no_rest(tmp_path):
    names = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "extra_thing")
    arr = np.zeros(4, dtype=[(n, "f4") for n in names])
    arr["x"] = np.arange(4)
    arr["rot_0"] = 1.0
    path = tmp_path / "splat.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))

    cloud = read_ply(path)

    assert cloud.sh_degree == 0
    assert cloud.sh_rest.shape == (4, 0, 3)
    np.testing.assert_array_equal(cloud.means[:, 0], np.arange(4))


def test_rejects_plain_mesh_ply(tmp_path):
    arr = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    path = tmp_path / "mesh.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    with pytest.raises(PlyFormatError, match="missing fields"):
        validate_ply(path)


def test_rejects_non_ply_bytes(tmp_path):
    path = tmp_path / "x.ply"
    path.write_bytes(b"hello world")
    with pytest.raises(PlyFormatError):
        validate_ply(path)


def test_rejects_unsupported_rest_count(tmp_path):
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
             "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] + [f"f_rest_{i}" for i in range(5)]
    arr = np.zeros(2, dtype=[(n, "f4") for n in names])
    path = tmp_path / "odd.ply"
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    with pytest.raises(PlyFormatError, match="f_rest"):
        validate_ply(path)
```

- [ ] **Step 2: Run to verify failure**

Run: `E:/anaconda3/envs/gs4d/python.exe -m pytest tests/compose/test_plyio.py -q`
Expected: FAIL / collection error `ModuleNotFoundError: No module named 'backend.compose'`.

- [ ] **Step 3: Implement**

`backend/compose/__init__.py`:

```python
"""Scene composer: place splats and meshes into a splat scene and bake the result."""
```

`backend/compose/plyio.py`:

```python
"""Standard INRIA 3DGS .ply read/write for the scene composer.

Same on-disk layout as ``backend/export/to_splat.write_ply`` but without the
torch import, so composer code and its tests stay light.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_FIELDS = (
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)
# Number of f_rest_* fields for SH degree 0..3.
_VALID_REST_FIELD_COUNTS = (0, 9, 24, 45)


class PlyFormatError(ValueError):
    """The file is not a usable Gaussian-splat .ply."""


@dataclass
class GaussianCloud:
    means: np.ndarray       # (N, 3) float32
    log_scales: np.ndarray  # (N, 3) float32, log-space
    quats: np.ndarray       # (N, 4) float32, (w, x, y, z), not necessarily unit
    opacities: np.ndarray   # (N,)   float32, logit-space
    sh_dc: np.ndarray       # (N, 3) float32
    sh_rest: np.ndarray     # (N, K, 3) float32, K = (degree + 1)^2 - 1

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @property
    def sh_degree(self) -> int:
        return sh_degree_for_coeffs(self.sh_rest.shape[1])


def sh_degree_for_coeffs(k: int) -> int:
    for degree in range(4):
        if (degree + 1) ** 2 - 1 == k:
            return degree
    raise PlyFormatError(f"Unsupported SH coefficient count: {k}")


def read_ply_header(path: str | Path) -> tuple[int, list[str]]:
    """Parse only the header: (vertex count, vertex property names)."""
    count: int | None = None
    props: list[str] = []
    in_vertex = False
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise PlyFormatError("Not a PLY file (missing 'ply' magic)")
        for _ in range(10_000):
            raw = fh.readline()
            if not raw:
                raise PlyFormatError("PLY header has no end_header")
            line = raw.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "element":
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    try:
                        count = int(parts[2])
                    except ValueError as exc:
                        raise PlyFormatError(f"Bad vertex count: {parts[2]}") from exc
            elif parts[0] == "property" and in_vertex:
                props.append(parts[-1])
        else:
            raise PlyFormatError("PLY header too long")
    if count is None:
        raise PlyFormatError("PLY has no vertex element")
    return count, props


def validate_ply(path: str | Path) -> int:
    """Check that ``path`` is a Gaussian-splat PLY. Returns the Gaussian count."""
    count, props = read_ply_header(path)
    missing = [f for f in REQUIRED_FIELDS if f not in props]
    if missing:
        raise PlyFormatError(f"Not a Gaussian-splat PLY, missing fields: {', '.join(missing)}")
    n_rest = sum(1 for p in props if p.startswith("f_rest_"))
    if n_rest not in _VALID_REST_FIELD_COUNTS:
        raise PlyFormatError(f"Unsupported number of f_rest fields: {n_rest}")
    if count <= 0:
        raise PlyFormatError("PLY contains no Gaussians")
    return count


def read_ply(path: str | Path) -> GaussianCloud:
    from plyfile import PlyData

    validate_ply(path)
    vertex = PlyData.read(str(path))["vertex"].data
    n = len(vertex)

    def col(name: str) -> np.ndarray:
        return np.asarray(vertex[name], dtype=np.float32)

    rest_names = sorted(
        (f for f in vertex.dtype.names if f.startswith("f_rest_")),
        key=lambda f: int(f[len("f_rest_"):]),
    )
    k = len(rest_names) // 3
    if rest_names:
        # Channel-major on disk: f_rest_{c*K + k}.
        flat = np.stack([col(f) for f in rest_names], axis=1)
        sh_rest = flat.reshape(n, 3, k).transpose(0, 2, 1).copy()
    else:
        sh_rest = np.zeros((n, 0, 3), dtype=np.float32)

    return GaussianCloud(
        means=np.stack([col("x"), col("y"), col("z")], axis=1),
        log_scales=np.stack([col(f"scale_{i}") for i in range(3)], axis=1),
        quats=np.stack([col(f"rot_{i}") for i in range(4)], axis=1),
        opacities=col("opacity"),
        sh_dc=np.stack([col(f"f_dc_{i}") for i in range(3)], axis=1),
        sh_rest=sh_rest,
    )


def write_ply(cloud: GaussianCloud, path: str | Path) -> Path:
    from plyfile import PlyData, PlyElement

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = cloud.count
    k = cloud.sh_rest.shape[1]
    dtype = (
        [(p, "f4") for p in ("x", "y", "z", "nx", "ny", "nz")]
        + [(f"f_dc_{i}", "f4") for i in range(3)]
        + [(f"f_rest_{i}", "f4") for i in range(3 * k)]
        + [("opacity", "f4")]
        + [(f"scale_{i}", "f4") for i in range(3)]
        + [(f"rot_{i}", "f4") for i in range(4)]
    )
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = cloud.means[:, 0], cloud.means[:, 1], cloud.means[:, 2]
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    for i in range(3):
        arr[f"f_dc_{i}"] = cloud.sh_dc[:, i]
    rest_flat = cloud.sh_rest.transpose(0, 2, 1).reshape(n, -1)
    for i in range(3 * k):
        arr[f"f_rest_{i}"] = rest_flat[:, i]
    arr["opacity"] = cloud.opacities
    for i in range(3):
        arr[f"scale_{i}"] = cloud.log_scales[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = cloud.quats[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))
    return path
```

- [ ] **Step 4: Run tests** — `E:/anaconda3/envs/gs4d/python.exe -m pytest tests/compose/test_plyio.py -q` → all PASS. (`helpers.random_rotation` imports `bake` lazily, so it is unused until Task 3.)

- [ ] **Step 5: Commit** — `git add backend/compose tests/compose && git commit -m "feat(compose): add torch-free 3DGS PLY reader and writer"`

---

### Task 2: SH rotation

**Files:**
- Create: `backend/compose/sh_rotation.py`, `tests/compose/test_sh_rotation.py`

- [ ] **Step 1: Failing tests** — `tests/compose/test_sh_rotation.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from backend.compose.sh_rotation import band_rotation_matrices, eval_sh_color, rotate_sh_rest
from tests.compose.helpers import random_rotation


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_rotated_sh_in_rotated_direction_equals_original(degree):
    rng = np.random.default_rng(degree)
    k = (degree + 1) ** 2 - 1
    for _ in range(5):
        r = random_rotation(rng)
        dc = rng.normal(size=(20, 3))
        rest = rng.normal(size=(20, k, 3))
        dirs = rng.normal(size=(20, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

        rotated = rotate_sh_rest(rest, r)

        np.testing.assert_allclose(
            eval_sh_color(dc, rotated, dirs @ r.T), eval_sh_color(dc, rest, dirs), atol=1e-6,
        )


def test_identity_rotation_gives_identity_matrices():
    for l, d in band_rotation_matrices(np.eye(3), 3).items():
        np.testing.assert_allclose(d, np.eye(2 * l + 1), atol=1e-9)


def test_band_matrices_compose_and_are_orthogonal():
    rng = np.random.default_rng(7)
    r1, r2 = random_rotation(rng), random_rotation(rng)
    d1 = band_rotation_matrices(r1, 3)
    d2 = band_rotation_matrices(r2, 3)
    d12 = band_rotation_matrices(r1 @ r2, 3)
    for l in (1, 2, 3):
        np.testing.assert_allclose(d12[l], d1[l] @ d2[l], atol=1e-9)
        np.testing.assert_allclose(d1[l] @ d1[l].T, np.eye(2 * l + 1), atol=1e-9)


def test_degree_zero_is_a_noop():
    rest = np.zeros((3, 0, 3), dtype=np.float32)
    assert rotate_sh_rest(rest, np.eye(3)).shape == (3, 0, 3)
```

- [ ] **Step 2: Run** — `... -m pytest tests/compose/test_sh_rotation.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement** — `backend/compose/sh_rotation.py`:

```python
"""Rotate real spherical-harmonic colour coefficients (INRIA 3DGS convention).

Instead of closed-form Wigner-D matrices, each band's rotation matrix is solved
numerically from the *same* basis functions the renderer uses, so sign and
ordering conventions cannot disagree with gsplat / 3DGS.
"""
from __future__ import annotations

import numpy as np

from .plyio import sh_degree_for_coeffs

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435)

# Column ranges of each band inside f_rest (band 0 is the DC term).
BAND_SLICES = {1: slice(0, 3), 2: slice(3, 8), 3: slice(8, 15)}


def sh_basis(dirs: np.ndarray, degree: int) -> np.ndarray:
    """Basis values for bands 1..degree at unit ``dirs`` (M, 3) → (M, (degree+1)^2 - 1)."""
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    cols: list[np.ndarray] = []
    if degree >= 1:
        cols += [-C1 * y, C1 * z, -C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        cols += [
            C2[0] * x * y,
            C2[1] * y * z,
            C2[2] * (2.0 * zz - xx - yy),
            C2[3] * x * z,
            C2[4] * (xx - yy),
        ]
    if degree >= 3:
        cols += [
            C3[0] * y * (3.0 * xx - yy),
            C3[1] * x * y * z,
            C3[2] * y * (4.0 * zz - xx - yy),
            C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            C3[4] * x * (4.0 * zz - xx - yy),
            C3[5] * z * (xx - yy),
            C3[6] * x * (xx - 3.0 * yy),
        ]
    if not cols:
        return np.zeros((dirs.shape[0], 0))
    return np.stack(cols, axis=1)


def eval_sh_color(sh_dc: np.ndarray, sh_rest: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Rendered colour (N, 3) for per-Gaussian unit view ``dirs`` (N, 3), unclamped."""
    degree = sh_degree_for_coeffs(sh_rest.shape[1])
    basis = sh_basis(dirs, degree)
    return 0.5 + C0 * sh_dc + np.einsum("nk,nkc->nc", basis, sh_rest)


def _fibonacci_sphere(n: int = 256) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


_SAMPLE_DIRS = _fibonacci_sphere()


def band_rotation_matrices(rotation: np.ndarray, degree: int) -> dict[int, np.ndarray]:
    """Per band ``l``: ``D_l`` with ``c'_l = D_l @ c_l`` so that ``SH'(R d) = SH(d)``."""
    if degree == 0:
        return {}
    r = np.asarray(rotation, dtype=np.float64)
    a = sh_basis(_SAMPLE_DIRS, degree)
    b = sh_basis(_SAMPLE_DIRS @ r, degree)  # rows are R^T d
    out: dict[int, np.ndarray] = {}
    for band in range(1, degree + 1):
        s = BAND_SLICES[band]
        # Solve Y(d) D = Y(R^T d) in the least-squares sense (exact for SH).
        out[band] = np.linalg.lstsq(a[:, s], b[:, s], rcond=None)[0]
    return out


def rotate_sh_rest(sh_rest: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate (N, K, 3) higher-order SH coefficients by ``rotation`` (3, 3)."""
    degree = sh_degree_for_coeffs(sh_rest.shape[1])
    if degree == 0:
        return sh_rest.copy()
    out = np.empty_like(sh_rest)
    for band, d in band_rotation_matrices(rotation, degree).items():
        s = BAND_SLICES[band]
        out[:, s, :] = np.einsum("ij,njc->nic", d, sh_rest[:, s, :])
    return out
```

- [ ] **Step 4: Run** — `tests/compose/helpers.random_rotation` needs `backend.compose.bake.quat_to_matrix`, so these tests pass once Task 3 Step 3 exists. Implement Tasks 2 and 3 back to back, then run both test files: expected PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(compose): rotate SH colour coefficients numerically"`

---

### Task 3: Bake (transform, crop, colour, merge)

**Files:**
- Create: `backend/compose/bake.py`, `tests/compose/test_bake.py`

- [ ] **Step 1: Failing tests** — `tests/compose/test_bake.py`:

```python
from __future__ import annotations

import math

import numpy as np
import pytest

from backend.compose.bake import (
    ColorAdjust, Crop, Placement, color_matrix, crop_mask, merge_clouds, transform_cloud,
)
from backend.compose.plyio import GaussianCloud
from backend.compose.sh_rotation import eval_sh_color
from tests.compose.helpers import random_cloud

S45 = math.sin(math.pi / 4)


def _dirs(n, seed=0):
    d = np.random.default_rng(seed).normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def test_identity_placement_is_a_noop():
    cloud = random_cloud(30)
    out = transform_cloud(cloud, Placement())
    for field in ("means", "log_scales", "quats", "opacities", "sh_dc"):
        np.testing.assert_array_equal(getattr(out, field), getattr(cloud, field))
    np.testing.assert_allclose(out.sh_rest, cloud.sh_rest, atol=1e-6)


def test_translation_moves_only_means():
    cloud = random_cloud(10)
    out = transform_cloud(cloud, Placement(position=(1.0, -2.0, 3.0)))
    np.testing.assert_allclose(out.means, cloud.means + np.array([1, -2, 3]), atol=1e-6)
    np.testing.assert_array_equal(out.quats, cloud.quats)


def test_rotation_rotates_means_and_gaussian_orientation():
    cloud = GaussianCloud(
        means=np.array([[1.0, 0.0, 0.0]], np.float32),
        log_scales=np.zeros((1, 3), np.float32),
        quats=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        opacities=np.zeros(1, np.float32),
        sh_dc=np.zeros((1, 3), np.float32),
        sh_rest=np.zeros((1, 0, 3), np.float32),
    )
    rz90 = Placement(quaternion_xyzw=(0.0, 0.0, S45, S45))
    out = transform_cloud(cloud, rz90)
    np.testing.assert_allclose(out.means, [[0.0, 1.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(out.quats, [[S45, 0.0, 0.0, S45]], atol=1e-6)


def test_uniform_scale_scales_means_and_log_scales():
    cloud = random_cloud(10)
    out = transform_cloud(cloud, Placement(scale=2.0))
    np.testing.assert_allclose(out.means, cloud.means * 2, atol=1e-6)
    np.testing.assert_allclose(out.log_scales, cloud.log_scales + math.log(2.0), atol=1e-6)


def test_crop_mask_axis_aligned_and_rotated():
    means = np.array([[0, 0, 0], [0.9, 0, 0], [1.1, 0, 0], [0, 0, 0.99]], np.float32)
    box = Crop(center=(0, 0, 0), half_size=(1, 1, 1), quaternion_xyzw=(0, 0, 0, 1))
    assert crop_mask(means, box).tolist() == [True, True, False, True]

    rotated = Crop(center=(0, 0, 0), half_size=(1, 0.1, 1),
                   quaternion_xyzw=(0, 0, math.sin(math.pi / 8), math.cos(math.pi / 8)))
    pts = np.array([[0.5, 0.5, 0.0], [0.5, -0.5, 0.0]], np.float32)
    assert crop_mask(pts, rotated).tolist() == [True, False]


def test_crop_happens_in_local_frame_before_transform():
    cloud = random_cloud(3)
    cloud.means[:] = [[0, 0, 0], [5, 0, 0], [0.2, 0, 0]]
    placement = Placement(
        position=(10.0, 0.0, 0.0),
        crop=Crop(center=(0, 0, 0), half_size=(1, 1, 1), quaternion_xyzw=(0, 0, 0, 1)),
    )
    out = transform_cloud(cloud, placement)
    np.testing.assert_allclose(out.means, [[10, 0, 0], [10.2, 0, 0]], atol=1e-6)


def test_exposure_plus_one_doubles_rendered_colour():
    cloud = random_cloud(40, seed=2)
    dirs = _dirs(40)
    out = transform_cloud(cloud, Placement(color=ColorAdjust(exposure=1.0)))
    np.testing.assert_allclose(
        eval_sh_color(out.sh_dc, out.sh_rest, dirs),
        2.0 * eval_sh_color(cloud.sh_dc, cloud.sh_rest, dirs),
        atol=1e-5,
    )


def test_saturation_zero_gives_grey():
    cloud = random_cloud(40, seed=3)
    out = transform_cloud(cloud, Placement(color=ColorAdjust(saturation=0.0)))
    rgb = eval_sh_color(out.sh_dc, out.sh_rest, _dirs(40))
    np.testing.assert_allclose(rgb[:, 0], rgb[:, 1], atol=1e-5)
    np.testing.assert_allclose(rgb[:, 1], rgb[:, 2], atol=1e-5)


def test_colour_matrix_applies_to_every_view_direction():
    adj = ColorAdjust(exposure=0.3, tint=(1.2, 0.9, 0.7), saturation=1.4)
    cloud = random_cloud(40, seed=4)
    dirs = _dirs(40, seed=1)
    out = transform_cloud(cloud, Placement(color=adj))
    np.testing.assert_allclose(
        eval_sh_color(out.sh_dc, out.sh_rest, dirs),
        eval_sh_color(cloud.sh_dc, cloud.sh_rest, dirs) @ color_matrix(adj).T,
        atol=1e-5,
    )


def test_rotation_keeps_view_dependent_colour_consistent():
    cloud = random_cloud(40, seed=5)
    q = np.array([0.3, -0.2, 0.5, 0.78])
    q /= np.linalg.norm(q)
    placement = Placement(quaternion_xyzw=tuple(q))
    from backend.compose.bake import quat_to_matrix

    r = quat_to_matrix(tuple(q))
    dirs = _dirs(40, seed=2)
    out = transform_cloud(cloud, placement)
    np.testing.assert_allclose(
        eval_sh_color(out.sh_dc, out.sh_rest, dirs @ r.T),
        eval_sh_color(cloud.sh_dc, cloud.sh_rest, dirs),
        atol=1e-5,
    )


def test_merge_pads_lower_sh_degree():
    a = random_cloud(5, degree=1, seed=1)
    b = random_cloud(3, degree=3, seed=2)
    merged = merge_clouds([a, b])
    assert merged.count == 8
    assert merged.sh_rest.shape == (8, 15, 3)
    np.testing.assert_array_equal(merged.sh_rest[:5, :3], a.sh_rest)
    assert not merged.sh_rest[:5, 3:].any()
    np.testing.assert_array_equal(merged.means[5:], b.means)


def test_merge_requires_at_least_one_cloud():
    with pytest.raises(ValueError, match="Nothing to merge"):
        merge_clouds([])
```

- [ ] **Step 2: Run** → FAIL (module missing).

- [ ] **Step 3: Implement** — `backend/compose/bake.py`:

```python
"""Bake scene-composer objects into one Gaussian cloud (pure numpy).

Order per object: crop (object-local) → uniform scale → rotation → translation,
then SH rotation and colour adjustment. Must match the Spark preview in
``frontend/src/compose`` (checked by the shared golden fixture).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .plyio import GaussianCloud
from .sh_rotation import C0, rotate_sh_rest

Vec3 = tuple[float, float, float]
QuatXYZW = tuple[float, float, float, float]

LUMA = np.array([0.2126, 0.7152, 0.0722])  # Rec.709


@dataclass(frozen=True)
class Crop:
    center: Vec3
    half_size: Vec3
    quaternion_xyzw: QuatXYZW = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class ColorAdjust:
    exposure: float = 0.0
    tint: Vec3 = (1.0, 1.0, 1.0)
    saturation: float = 1.0


@dataclass(frozen=True)
class Placement:
    position: Vec3 = (0.0, 0.0, 0.0)
    quaternion_xyzw: QuatXYZW = (0.0, 0.0, 0.0, 1.0)
    scale: float = 1.0
    crop: Crop | None = None
    color: ColorAdjust | None = None


def quat_to_matrix(q_xyzw: QuatXYZW) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64) / np.linalg.norm(q_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ⊗ b`` for one quaternion ``a`` (4,) and many ``b`` (N, 4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=1)


def crop_mask(means: np.ndarray, crop: Crop) -> np.ndarray:
    """True for Gaussians whose centre lies inside the (oriented) crop box."""
    r = quat_to_matrix(crop.quaternion_xyzw)
    local = (means.astype(np.float64) - np.asarray(crop.center)) @ r  # rows: R^T (p - c)
    return np.all(np.abs(local) <= np.asarray(crop.half_size), axis=1)


def color_matrix(adj: ColorAdjust) -> np.ndarray:
    """``M = S · diag(gain)``: exposure/tint first, then saturation around luma."""
    gain = (2.0 ** adj.exposure) * np.asarray(adj.tint, dtype=np.float64)
    luma = np.outer(np.ones(3), LUMA)
    sat = luma + adj.saturation * (np.eye(3) - luma)
    return sat @ np.diag(gain)


def _apply_color(sh_dc: np.ndarray, sh_rest: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Rendered colour is 0.5 + C0*dc + Σ rest·Y, so applying M to (0.5 + C0*dc)
    # and to every rest coefficient equals M · colour for every view direction.
    rgb = 0.5 + C0 * sh_dc.astype(np.float64)
    dc = (rgb @ m.T - 0.5) / C0
    rest = np.einsum("ij,nkj->nki", m, sh_rest)
    return dc.astype(np.float32), rest.astype(np.float32)


def _select(cloud: GaussianCloud, keep: np.ndarray) -> GaussianCloud:
    return GaussianCloud(
        means=cloud.means[keep], log_scales=cloud.log_scales[keep], quats=cloud.quats[keep],
        opacities=cloud.opacities[keep], sh_dc=cloud.sh_dc[keep], sh_rest=cloud.sh_rest[keep],
    )


def transform_cloud(cloud: GaussianCloud, placement: Placement) -> GaussianCloud:
    src = cloud if placement.crop is None else _select(cloud, crop_mask(cloud.means, placement.crop))
    q = np.asarray(placement.quaternion_xyzw, dtype=np.float64)
    q = q / np.linalg.norm(q)
    r = quat_to_matrix(tuple(q))
    s = float(placement.scale)

    means = (s * (src.means.astype(np.float64) @ r.T) + np.asarray(placement.position)).astype(np.float32)
    # Unit q_o keeps the norm of the stored (possibly unnormalised) quaternion.
    q_wxyz = np.array([q[3], q[0], q[1], q[2]])
    quats = quat_multiply_wxyz(q_wxyz, src.quats.astype(np.float64)).astype(np.float32)
    log_scales = (src.log_scales.astype(np.float64) + math.log(s)).astype(np.float32)
    sh_rest = rotate_sh_rest(src.sh_rest, r).astype(np.float32)
    sh_dc = src.sh_dc.copy()
    if placement.color is not None:
        sh_dc, sh_rest = _apply_color(sh_dc, sh_rest, color_matrix(placement.color))
    return GaussianCloud(means, log_scales, quats, src.opacities.copy(), sh_dc, sh_rest)


def merge_clouds(clouds: list[GaussianCloud]) -> GaussianCloud:
    if not clouds:
        raise ValueError("Nothing to merge: no visible splat objects")
    k = max(c.sh_rest.shape[1] for c in clouds)
    rest = [np.pad(c.sh_rest, ((0, 0), (0, k - c.sh_rest.shape[1]), (0, 0))) for c in clouds]
    return GaussianCloud(
        means=np.concatenate([c.means for c in clouds]),
        log_scales=np.concatenate([c.log_scales for c in clouds]),
        quats=np.concatenate([c.quats for c in clouds]),
        opacities=np.concatenate([c.opacities for c in clouds]),
        sh_dc=np.concatenate([c.sh_dc for c in clouds]),
        sh_rest=np.concatenate(rest),
    )
```

Note: for the identity test, `log(1) = 0` and `s*(p@I)+0` are exact in float64, so every field except `sh_rest` round-trips bit-identically; `sh_rest` goes through a least-squares identity and is compared with `atol=1e-6`.

- [ ] **Step 4: Run** — `... -m pytest tests/compose/test_bake.py tests/compose/test_sh_rotation.py -q` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(compose): bake transforms, crop and colour into Gaussian clouds"`

---

### Task 4: Golden fixture shared with the frontend

**Files:**
- Create: `tests/compose/golden.py`, `tests/compose/test_golden.py`, `tests/fixtures/compose_golden.json` (generated)

- [ ] **Step 1: Write generator + test**

`tests/compose/golden.py`:

```python
"""Golden cases shared by backend bake tests and frontend preview tests.

Regenerate after an intentional convention change:
    python -m tests.compose.golden
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from backend.compose.bake import ColorAdjust, Crop, color_matrix, crop_mask, quat_to_matrix

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "compose_golden.json"


def _unit(q):
    n = math.sqrt(sum(c * c for c in q))
    return [c / n for c in q]


TRANSFORMS = [
    {"position": [0.5, -1.0, 2.0], "quaternion": _unit([0.2, 0.3, -0.1, 0.9]), "scale": 1.7},
    {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)], "scale": 0.5},
]
POINTS = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-0.3, 0.25, 1.5]]
CROPS = [{"center": [0.1, 0.2, 0.0], "halfSize": [0.5, 0.3, 0.4],
          "quaternion": [0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)]}]
CROP_POINTS = [[0.1, 0.2, 0.0], [0.5, 0.2, 0.0], [0.1, 0.45, 0.0], [-0.2, 0.5, 0.0],
               [0.8, 0.2, 0.0], [0.1, 0.2, 0.39], [0.1, 0.2, 0.41]]
COLORS = [{"exposure": 0.5, "tint": [1.1, 0.9, 0.8], "saturation": 0.6},
          {"exposure": -1.0, "tint": [1.0, 1.0, 1.0], "saturation": 1.5}]
RGBS = [[0.2, 0.4, 0.6], [0.9, 0.1, 0.3], [0.5, 0.5, 0.5]]


def build_golden() -> dict:
    transform_cases = []
    for t in TRANSFORMS:
        r = quat_to_matrix(tuple(t["quaternion"]))
        pts = np.asarray(POINTS, dtype=np.float64)
        expected = t["scale"] * (pts @ r.T) + np.asarray(t["position"])
        transform_cases.append({"transform": t, "points": POINTS, "expected": expected.tolist()})
    crop_cases = []
    for c in CROPS:
        crop = Crop(center=tuple(c["center"]), half_size=tuple(c["halfSize"]),
                    quaternion_xyzw=tuple(c["quaternion"]))
        inside = crop_mask(np.asarray(CROP_POINTS, dtype=np.float64), crop)
        crop_cases.append({"crop": c, "points": CROP_POINTS, "inside": inside.tolist()})
    color_cases = []
    for c in COLORS:
        m = color_matrix(ColorAdjust(exposure=c["exposure"], tint=tuple(c["tint"]),
                                     saturation=c["saturation"]))
        color_cases.append({"color": c, "rgb": RGBS, "expected": (np.asarray(RGBS) @ m.T).tolist()})
    return {"transform_cases": transform_cases, "crop_cases": crop_cases, "color_cases": color_cases}


def main() -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(build_golden(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
```

`tests/compose/test_golden.py`:

```python
from __future__ import annotations

import json

import numpy as np

from backend.compose.bake import Crop, Placement, transform_cloud
from backend.compose.plyio import GaussianCloud
from tests.compose.golden import GOLDEN_PATH, build_golden


def _cloud_at(points):
    n = len(points)
    return GaussianCloud(
        means=np.asarray(points, np.float32), log_scales=np.zeros((n, 3), np.float32),
        quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)), opacities=np.zeros(n, np.float32),
        sh_dc=np.zeros((n, 3), np.float32), sh_rest=np.zeros((n, 0, 3), np.float32),
    )


def test_committed_fixture_matches_generator():
    committed = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(build_golden()))
```

Also add:


```python
def test_bake_reproduces_golden_transforms():
    golden = build_golden()
    for case in golden["transform_cases"]:
        t = case["transform"]
        out = transform_cloud(_cloud_at(case["points"]), Placement(
            position=tuple(t["position"]), quaternion_xyzw=tuple(t["quaternion"]), scale=t["scale"]))
        np.testing.assert_allclose(out.means, case["expected"], atol=1e-5)


def test_bake_reproduces_golden_crops():
    golden = build_golden()
    for case in golden["crop_cases"]:
        c = case["crop"]
        crop = Crop(center=tuple(c["center"]), half_size=tuple(c["halfSize"]),
                    quaternion_xyzw=tuple(c["quaternion"]))
        out = transform_cloud(_cloud_at(case["points"]), Placement(crop=crop))
        kept = [p for p, inside in zip(case["points"], case["inside"]) if inside]
        np.testing.assert_allclose(out.means, kept, atol=1e-6)
```

- [ ] **Step 2: Generate fixture** — `E:/anaconda3/envs/gs4d/python.exe -m tests.compose.golden` → `wrote .../tests/fixtures/compose_golden.json`. Inspect: crop `inside` must contain both `true` and `false`.

- [ ] **Step 3: Run** — `... -m pytest tests/compose/test_golden.py -q` → PASS.

- [ ] **Step 4: Commit** — `git commit -m "test(compose): add golden fixture shared by bake and preview"`

---

### Task 5: GLB helpers

**Files:** Create `backend/compose/glb.py`, `tests/compose/test_glb.py`

- [ ] **Step 1: Failing tests** — `tests/compose/test_glb.py`:

```python
from __future__ import annotations

import json
import math
import struct

import numpy as np
import pytest

from backend.compose.glb import GlbFormatError, placement_matrix, validate_glb, wrap_with_transform
from tests.compose.helpers import minimal_glb


def _chunks(data: bytes):
    _, _, length = struct.unpack_from("<III", data, 0)
    assert length == len(data)
    off, out = 12, []
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        out.append((ctype, data[off + 8: off + 8 + clen]))
        off += 8 + clen
    return out


def test_wrap_adds_root_node_with_matrix(tmp_path):
    src = tmp_path / "in.glb"
    src.write_bytes(minimal_glb())
    m = placement_matrix((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), 2.0)

    wrap_with_transform(src, tmp_path / "out" / "o.glb", m)

    data = (tmp_path / "out" / "o.glb").read_bytes()
    chunks = _chunks(data)
    gltf = json.loads(chunks[0][1])
    root = gltf["nodes"][-1]
    assert root["name"] == "compose_root"
    assert root["children"] == [0]
    assert gltf["scenes"][0]["nodes"] == [len(gltf["nodes"]) - 1]
    assert root["matrix"] == [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 2, 3, 1]
    assert chunks[1][1] == _chunks(minimal_glb())[1][1]
    assert len(chunks[0][1]) % 4 == 0
    validate_glb(tmp_path / "out" / "o.glb")


def test_wrap_without_scenes_uses_parentless_nodes(tmp_path):
    src = tmp_path / "in.glb"
    src.write_bytes(minimal_glb({"asset": {"version": "2.0"},
                                 "nodes": [{"children": [1]}, {}, {}]}))
    wrap_with_transform(src, tmp_path / "o.glb", np.eye(4))
    gltf = json.loads(_chunks((tmp_path / "o.glb").read_bytes())[0][1])
    assert gltf["nodes"][-1]["children"] == [0, 2]
    assert gltf["scene"] == 0


def test_placement_matrix_rotation():
    s = math.sqrt(0.5)
    m = placement_matrix((0.0, 0.0, 0.0), (0.0, 0.0, s, s), 1.0)
    np.testing.assert_allclose(m[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-9)


@pytest.mark.parametrize("payload", [b"nope" * 10, struct.pack("<III", 0x46546C67, 1, 12)])
def test_validate_rejects_bad_files(tmp_path, payload):
    p = tmp_path / "bad.glb"
    p.write_bytes(payload)
    with pytest.raises(GlbFormatError):
        validate_glb(p)
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** — `backend/compose/glb.py`:

```python
"""Minimal GLB (binary glTF 2.0) helpers: validate, and wrap a scene in a transform node."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942


class GlbFormatError(ValueError):
    """The file is not a usable binary glTF 2.0 file."""


def _read_chunks(data: bytes) -> list[tuple[int, bytes]]:
    if len(data) < 20:
        raise GlbFormatError("File too small to be a GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise GlbFormatError("Not a GLB file (bad magic)")
    if version != 2:
        raise GlbFormatError(f"Unsupported glTF version {version}")
    if length > len(data):
        raise GlbFormatError("GLB file is truncated")
    chunks: list[tuple[int, bytes]] = []
    off = 12
    while off + 8 <= length:
        clen, ctype = struct.unpack_from("<II", data, off)
        body = data[off + 8: off + 8 + clen]
        if len(body) != clen:
            raise GlbFormatError("GLB chunk is truncated")
        chunks.append((ctype, body))
        off += 8 + clen
    if not chunks or chunks[0][0] != CHUNK_JSON:
        raise GlbFormatError("GLB must start with a JSON chunk")
    return chunks


def _parse_json(chunk: bytes) -> dict:
    try:
        return json.loads(chunk.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlbFormatError(f"Invalid GLB JSON chunk: {exc}") from exc


def validate_glb(path: str | Path) -> dict:
    return _parse_json(_read_chunks(Path(path).read_bytes())[0][1])


def placement_matrix(position, quaternion_xyzw, scale: float) -> np.ndarray:
    from .bake import quat_to_matrix

    m = np.eye(4)
    m[:3, :3] = float(scale) * quat_to_matrix(tuple(quaternion_xyzw))
    m[:3, 3] = position
    return m


def wrap_with_transform(src: str | Path, dst: str | Path, matrix: np.ndarray) -> Path:
    """Copy ``src`` to ``dst`` with every scene root re-parented under one node carrying ``matrix``."""
    chunks = _read_chunks(Path(src).read_bytes())
    gltf = _parse_json(chunks[0][1])
    nodes = gltf.setdefault("nodes", [])
    scenes = gltf.get("scenes")
    scene_idx = gltf.get("scene", 0)
    if scenes and 0 <= scene_idx < len(scenes):
        roots = list(scenes[scene_idx].get("nodes", []))
    else:
        children = {c for node in nodes for c in node.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
        scenes = gltf["scenes"] = [{"nodes": []}]
        scene_idx = 0
    wrapper: dict = {
        "name": "compose_root",
        "matrix": [float(v) for v in np.asarray(matrix, dtype=np.float64).T.reshape(-1)],  # column-major
    }
    if roots:
        wrapper["children"] = roots
    nodes.append(wrapper)
    scenes[scene_idx]["nodes"] = [len(nodes) - 1]
    gltf["scene"] = scene_idx

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    body = [(CHUNK_JSON, json_bytes)] + chunks[1:]
    total = 12 + sum(8 + len(b) for _, b in body)
    out = bytearray(struct.pack("<III", GLB_MAGIC, 2, total))
    for ctype, b in body:
        out += struct.pack("<II", len(b), ctype) + b
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(bytes(out))
    return dst
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** — `git commit -m "feat(compose): wrap GLB scenes in a placement transform"`

---

### Task 6: Models, store, exporter, routes

**Files:** Create `backend/compose/models.py`, `store.py`, `exporter.py`, `routes.py`, `tests/compose/test_routes.py`

- [ ] **Step 1: Failing tests** — `tests/compose/test_routes.py`:

```python
from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.compose.plyio import read_ply, write_ply
from backend.compose.routes import build_compose_router
from tests.compose.helpers import minimal_glb, random_cloud


class FakeManager:
    """Runs submitted jobs synchronously and records their results."""

    def __init__(self):
        self.jobs: dict[str, SimpleNamespace] = {}
        self.results: dict[str, dict] = {}

    def create(self, scene: str, smoke_test: bool = False):
        job = SimpleNamespace(id=f"job{len(self.jobs)}", scene=scene)
        self.jobs[job.id] = job
        return job

    def submit(self, job_id, runner):
        self.results[job_id] = runner(lambda *args, **kwargs: None)


def _setup(tmp_path):
    manager = FakeManager()
    app = FastAPI()
    app.include_router(build_compose_router(tmp_path, lambda: manager))
    return TestClient(app), manager


def _ply_bytes(tmp_path, n=20, seed=0, name="x.ply"):
    return write_ply(random_cloud(n, seed=seed), tmp_path / "src" / name).read_bytes()


def _upload(client, filename, data):
    return client.post("/compose/assets", files={"file": (filename, data, "application/octet-stream")})


def test_upload_ply_and_glb_are_listed(tmp_path):
    client, _ = _setup(tmp_path)
    r1 = _upload(client, "statue.ply", _ply_bytes(tmp_path))
    r2 = _upload(client, "chair.glb", minimal_glb())
    assert r1.status_code == 201 and r1.json()["kind"] == "splat" and r1.json()["name"] == "statue"
    assert r2.status_code == 201 and r2.json()["kind"] == "mesh"
    ids = {a["id"] for a in client.get("/compose/assets").json()}
    assert {r1.json()["id"], r2.json()["id"]} <= ids


def test_upload_rejects_wrong_extension_and_corrupt_files(tmp_path):
    client, _ = _setup(tmp_path)
    assert _upload(client, "notes.txt", b"hi").status_code == 400
    bad = _upload(client, "broken.ply", b"ply\nformat ascii 1.0\nend_header\n")
    assert bad.status_code == 400
    leftovers = list((tmp_path / "compose" / "assets").glob("*"))
    assert leftovers == []


def test_upload_filename_cannot_escape_assets_dir(tmp_path):
    client, _ = _setup(tmp_path)
    r = _upload(client, "../../evil.ply", _ply_bytes(tmp_path))
    assert r.status_code == 201
    assert r.json()["name"] == "evil"
    assert not (tmp_path.parent / "evil.ply").exists()
    assert (tmp_path / "compose" / "assets" / f"{r.json()['id']}.ply").is_file()


def test_pipeline_results_are_assets_and_downloadable(tmp_path):
    client, _ = _setup(tmp_path)
    write_ply(random_cloud(5), tmp_path / "garden" / "output" / "ply" / "frame_0000.ply")
    assets = {a["id"]: a for a in client.get("/compose/assets").json()}
    assert assets["scene__garden"]["source"] == "pipeline"
    r = client.get("/compose/assets/scene__garden/file")
    assert r.status_code == 200 and r.content.startswith(b"ply")
    assert client.get("/compose/assets/scene__..%2F/file").status_code == 404
    assert client.get("/compose/assets/nope/file").status_code == 404


def test_scene_create_save_load_roundtrip(tmp_path):
    client, _ = _setup(tmp_path)
    base = _upload(client, "room.ply", _ply_bytes(tmp_path)).json()
    doc = client.post("/compose/scenes", json={"name": "Oda", "base_asset": base["id"]}).json()
    assert [o["role"] for o in doc["objects"]] == ["base"]

    statue = _upload(client, "statue.ply", _ply_bytes(tmp_path, seed=1)).json()
    doc["objects"].append({
        "id": "o_statue", "kind": "splat", "asset": statue["id"], "name": "statue",
        "transform": {"position": [1, 0, 0], "quaternion": [0, 0, 0, 1], "scale": 0.5},
        "crop": {"center": [0, 0, 0], "halfSize": [1, 1, 1], "quaternion": [0, 0, 0, 1]},
        "color": {"exposure": 0.5, "tint": [1, 1, 1], "saturation": 1},
    })
    saved = client.put(f"/compose/scenes/{doc['id']}", json=doc)
    assert saved.status_code == 200, saved.text
    loaded = client.get(f"/compose/scenes/{doc['id']}").json()
    assert loaded["objects"][1]["transform"]["scale"] == 0.5
    listed = client.get("/compose/scenes").json()
    assert listed[0]["id"] == doc["id"] and listed[0]["object_count"] == 2


def test_scene_validation_errors(tmp_path):
    client, _ = _setup(tmp_path)
    base = _upload(client, "room.ply", _ply_bytes(tmp_path)).json()
    mesh = _upload(client, "chair.glb", minimal_glb()).json()
    assert client.post("/compose/scenes", json={"name": "x", "base_asset": mesh["id"]}).status_code == 400
    assert client.post("/compose/scenes", json={"name": "x", "base_asset": "a_missing"}).status_code == 400
    doc = client.post("/compose/scenes", json={"name": "x", "base_asset": base["id"]}).json()
    url = f"/compose/scenes/{doc['id']}"

    moved = {**doc, "objects": [{**doc["objects"][0], "transform": {"position": [1, 0, 0]}}]}
    assert client.put(url, json=moved).status_code == 422
    two_bases = {**doc, "objects": doc["objects"] + [{**doc["objects"][0], "id": "o_b2"}]}
    assert client.put(url, json=two_bases).status_code == 422
    mesh_crop = {**doc, "objects": doc["objects"] + [{
        "id": "o_m", "kind": "mesh", "asset": mesh["id"], "name": "m",
        "crop": {"center": [0, 0, 0], "halfSize": [1, 1, 1]}}]}
    assert client.put(url, json=mesh_crop).status_code == 422
    kind_mismatch = {**doc, "objects": doc["objects"] + [{
        "id": "o_k", "kind": "splat", "asset": mesh["id"], "name": "k"}]}
    assert client.put(url, json=kind_mismatch).status_code == 400
    assert client.put(f"/compose/scenes/s_other", json=doc).status_code == 400
    assert client.get("/compose/scenes/s_nope").status_code == 404
    assert client.get("/compose/scenes/..").status_code == 404


def test_export_bakes_merged_ply_meshes_and_zip(tmp_path):
    client, manager = _setup(tmp_path)
    base = _upload(client, "room.ply", _ply_bytes(tmp_path, n=30, seed=1)).json()
    statue = _upload(client, "statue.ply", _ply_bytes(tmp_path, n=40, seed=2)).json()
    chair = _upload(client, "chair.glb", minimal_glb()).json()
    doc = client.post("/compose/scenes", json={"name": "Bahçe", "base_asset": base["id"]}).json()
    doc["objects"] += [
        {"id": "o_statue", "kind": "splat", "asset": statue["id"], "name": "statue",
         "transform": {"position": [5, 0, 0], "quaternion": [0, 0, 0, 1], "scale": 1},
         "crop": {"center": [0, 0, 0], "halfSize": [0.5, 0.5, 0.5], "quaternion": [0, 0, 0, 1]}},
        {"id": "o_hidden", "kind": "splat", "asset": statue["id"], "name": "hidden", "visible": False},
        {"id": "o_chair", "kind": "mesh", "asset": chair["id"], "name": "chair",
         "transform": {"position": [0, 1, 0], "quaternion": [0, 0, 0, 1], "scale": 1}},
    ]
    assert client.put(f"/compose/scenes/{doc['id']}", json=doc).status_code == 200

    r = client.post(f"/compose/scenes/{doc['id']}/export")

    assert r.status_code == 202
    result = manager.results[r.json()["job_id"]]
    assert result["download_url"] == f"/compose/exports/{doc['id']}/download"
    assert manager.jobs[r.json()["job_id"]].scene == f"compose-{doc['id']}"
    statue_src = read_ply(tmp_path / "compose" / "assets" / f"{statue['id']}.ply")
    inside = int(np.all(np.abs(statue_src.means) <= 0.5, axis=1).sum())
    merged = read_ply(tmp_path / "compose" / "exports" / doc["id"] / "merged.ply")
    assert merged.count == 30 + inside == result["gaussians"]

    z = client.get(result["download_url"])
    assert z.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(z.content)).namelist())
    assert names == {"merged.ply", "meshes/o_chair.glb", "scene.json"}


def test_export_rejects_missing_assets(tmp_path):
    client, manager = _setup(tmp_path)
    base = _upload(client, "room.ply", _ply_bytes(tmp_path)).json()
    doc = client.post("/compose/scenes", json={"name": "x", "base_asset": base["id"]}).json()
    (tmp_path / "compose" / "assets" / f"{base['id']}.ply").unlink()
    r = client.post(f"/compose/scenes/{doc['id']}/export")
    assert r.status_code == 400 and base["id"] in r.json()["detail"]
    assert manager.jobs == {}
    assert client.get(f"/compose/exports/{doc['id']}/download").status_code == 404
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement models** — `backend/compose/models.py`:

```python
"""Pydantic models for the scene composer (SceneDoc v1)."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Safe for use in file names: no separators, no leading dot.
ID_PATTERN = r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}$"

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # three.js order: x, y, z, w


def _unit_quat(q: Quat) -> Quat:
    norm = math.sqrt(sum(c * c for c in q))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"quaternion must be unit length (norm={norm:.4f})")
    return tuple(c / norm for c in q)  # type: ignore[return-value]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Transform(_Strict):
    position: Vec3 = (0.0, 0.0, 0.0)
    quaternion: Quat = (0.0, 0.0, 0.0, 1.0)
    scale: float = Field(1.0, gt=0)

    @field_validator("quaternion")
    @classmethod
    def _unit(cls, v: Quat) -> Quat:
        return _unit_quat(v)

    def is_identity(self, tol: float = 1e-6) -> bool:
        return (
            all(abs(c) <= tol for c in self.position)
            and all(abs(a - b) <= tol for a, b in zip(self.quaternion, (0.0, 0.0, 0.0, 1.0)))
            and abs(self.scale - 1.0) <= tol
        )


class CropBox(_Strict):
    center: Vec3 = (0.0, 0.0, 0.0)
    halfSize: Vec3
    quaternion: Quat = (0.0, 0.0, 0.0, 1.0)

    @field_validator("quaternion")
    @classmethod
    def _unit(cls, v: Quat) -> Quat:
        return _unit_quat(v)

    @field_validator("halfSize")
    @classmethod
    def _positive(cls, v: Vec3) -> Vec3:
        if any(c <= 0 for c in v):
            raise ValueError("crop halfSize must be > 0 on every axis")
        return v


class ColorAdjust(_Strict):
    exposure: float = Field(0.0, ge=-3.0, le=3.0)
    tint: Vec3 = (1.0, 1.0, 1.0)
    saturation: float = Field(1.0, ge=0.0, le=2.0)

    @field_validator("tint")
    @classmethod
    def _tint_range(cls, v: Vec3) -> Vec3:
        if any(c < 0 or c > 2 for c in v):
            raise ValueError("tint channels must be within 0..2")
        return v


class SceneObject(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    kind: Literal["splat", "mesh"]
    asset: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1, max_length=128)
    role: Literal["base", "object"] = "object"
    visible: bool = True
    transform: Transform = Field(default_factory=Transform)
    crop: CropBox | None = None
    color: ColorAdjust | None = None

    @model_validator(mode="after")
    def _mesh_has_no_splat_edits(self) -> "SceneObject":
        if self.kind == "mesh" and (self.crop is not None or self.color is not None):
            raise ValueError(f"object {self.id}: crop and colour are only supported on splats")
        return self


class SceneDoc(_Strict):
    version: Literal[1] = 1
    id: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1, max_length=128)
    viewUp: Literal["y", "-y"] = "y"
    objects: list[SceneObject] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_objects(self) -> "SceneDoc":
        ids = [o.id for o in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object ids must be unique")
        bases = [o for o in self.objects if o.role == "base"]
        if len(bases) != 1:
            raise ValueError("scene must have exactly one base object")
        base = bases[0]
        if base.kind != "splat":
            raise ValueError("base object must be a splat")
        if not base.transform.is_identity():
            raise ValueError("base object transform must be identity")
        return self


class Asset(_Strict):
    id: str
    kind: Literal["splat", "mesh"]
    name: str
    size_bytes: int
    source: Literal["upload", "pipeline"]
    created_ts: float


class SceneSummary(BaseModel):
    id: str
    name: str
    updated_ts: float
    object_count: int


class CreateSceneRequest(_Strict):
    name: str = Field(min_length=1, max_length=128)
    base_asset: str = Field(pattern=ID_PATTERN)


class ExportResponse(BaseModel):
    job_id: str
```


- [ ] **Step 4: Implement store** — `backend/compose/store.py`:

```python
"""On-disk layout for composer assets, scene documents and exports.

    <data_root>/compose/assets/<asset_id>.{ply,glb}  + <asset_id>.json (Asset)
    <data_root>/compose/scenes/<scene_id>.json       (SceneDoc)
    <data_root>/compose/exports/<scene_id>/          (merged.ply, meshes/, scene.json)
    <data_root>/compose/exports/<scene_id>.zip

Pipeline results (<data_root>/<scene>/output/ply/*.ply) appear as read-only
assets with id ``scene__<scene>``. Ids are generated here and checked against
``ID_PATTERN`` before they touch a path, so user input never becomes a path.
"""
from __future__ import annotations

import os
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import BinaryIO

from .glb import GlbFormatError, validate_glb
from .models import ID_PATTERN, Asset, SceneDoc, SceneObject, SceneSummary
from .plyio import PlyFormatError, validate_ply

_ID_RE = re.compile(ID_PATTERN)
PIPELINE_PREFIX = "scene__"
_KIND_BY_EXT = {".ply": "splat", ".glb": "mesh"}
_EXT_BY_KIND = {"splat": ".ply", "mesh": ".glb"}


class AssetError(ValueError):
    """Upload rejected (wrong type or unreadable content)."""


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class ComposeStore:
    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.root = self.data_root / "compose"
        self.assets_dir = self.root / "assets"
        self.scenes_dir = self.root / "scenes"
        self.exports_dir = self.root / "exports"

    def _ensure_dirs(self) -> None:
        for d in (self.assets_dir, self.scenes_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- assets ---------------------------------------------------------
    def save_upload(self, filename: str, src: BinaryIO) -> Asset:
        ext = Path(filename).suffix.lower()
        kind = _KIND_BY_EXT.get(ext)
        if kind is None:
            raise AssetError("Only .ply (splat) and .glb (mesh) files are supported")
        self._ensure_dirs()
        asset_id = new_id("a")
        final = self.assets_dir / f"{asset_id}{ext}"
        part = self.assets_dir / f"{asset_id}{ext}.part"
        try:
            with open(part, "wb") as fh:
                shutil.copyfileobj(src, fh, 1024 * 1024)
            if kind == "splat":
                validate_ply(part)
            else:
                validate_glb(part)
        except (PlyFormatError, GlbFormatError) as exc:
            part.unlink(missing_ok=True)
            raise AssetError(str(exc)) from exc
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, final)
        asset = Asset(
            id=asset_id,
            kind=kind,
            name=Path(filename.replace("\\", "/")).stem[:128] or asset_id,
            size_bytes=final.stat().st_size,
            source="upload",
            created_ts=time.time(),
        )
        _atomic_write_text(self.assets_dir / f"{asset_id}.json", asset.model_dump_json())
        return asset

    def _upload_path(self, asset: Asset) -> Path:
        return self.assets_dir / f"{asset.id}{_EXT_BY_KIND[asset.kind]}"

    def _pipeline_ply(self, scene_name: str) -> Path | None:
        ply_dir = self.data_root / scene_name / "output" / "ply"
        plys = sorted(ply_dir.glob("*.ply")) if ply_dir.is_dir() else []
        return plys[-1] if plys else None

    def _pipeline_asset(self, scene_name: str) -> Asset | None:
        ply = self._pipeline_ply(scene_name)
        if ply is None:
            return None
        st = ply.stat()
        return Asset(id=PIPELINE_PREFIX + scene_name, kind="splat", name=scene_name,
                     size_bytes=st.st_size, source="pipeline", created_ts=st.st_mtime)

    def list_assets(self) -> list[Asset]:
        out: list[Asset] = []
        if self.assets_dir.is_dir():
            for meta in sorted(self.assets_dir.glob("*.json")):
                asset = self.get_asset(meta.stem)
                if asset is not None:
                    out.append(asset)
        if self.data_root.is_dir():
            for d in sorted(self.data_root.iterdir()):
                if d.is_dir() and d.name != "compose" and _ID_RE.match(PIPELINE_PREFIX + d.name) \
                        and _ID_RE.match(d.name):
                    asset = self._pipeline_asset(d.name)
                    if asset is not None:
                        out.append(asset)
        return out

    def get_asset(self, asset_id: str) -> Asset | None:
        if not _ID_RE.match(asset_id):
            return None
        if asset_id.startswith(PIPELINE_PREFIX):
            name = asset_id[len(PIPELINE_PREFIX):]
            return self._pipeline_asset(name) if _ID_RE.match(name) else None
        meta = self.assets_dir / f"{asset_id}.json"
        if not meta.is_file():
            return None
        try:
            asset = Asset.model_validate_json(meta.read_text(encoding="utf-8"))
        except ValueError:
            return None
        return asset if self._upload_path(asset).is_file() else None

    def asset_path(self, asset_id: str) -> Path | None:
        asset = self.get_asset(asset_id)
        if asset is None:
            return None
        if asset.source == "pipeline":
            return self._pipeline_ply(asset.name)
        return self._upload_path(asset)

    # -- scenes ---------------------------------------------------------
    def create_scene(self, name: str, base: Asset) -> SceneDoc:
        doc = SceneDoc(
            id=new_id("s"),
            name=name,
            objects=[SceneObject(id=new_id("o"), kind="splat", asset=base.id, name=base.name, role="base")],
        )
        self.save_scene(doc)
        return doc

    def save_scene(self, doc: SceneDoc) -> None:
        self._ensure_dirs()
        _atomic_write_text(self.scenes_dir / f"{doc.id}.json", doc.model_dump_json(indent=2))

    def load_scene(self, scene_id: str) -> SceneDoc | None:
        if not _ID_RE.match(scene_id):
            return None
        path = self.scenes_dir / f"{scene_id}.json"
        if not path.is_file():
            return None
        return SceneDoc.model_validate_json(path.read_text(encoding="utf-8"))

    def list_scenes(self) -> list[SceneSummary]:
        out: list[SceneSummary] = []
        if not self.scenes_dir.is_dir():
            return out
        for path in self.scenes_dir.glob("*.json"):
            try:
                doc = SceneDoc.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out.append(SceneSummary(id=doc.id, name=doc.name, updated_ts=path.stat().st_mtime,
                                    object_count=len(doc.objects)))
        out.sort(key=lambda s: s.updated_ts, reverse=True)
        return out

    # -- exports --------------------------------------------------------
    def export_dir(self, scene_id: str) -> Path:
        return self.exports_dir / scene_id

    def export_zip(self, scene_id: str) -> Path:
        return self.exports_dir / f"{scene_id}.zip"
```

- [ ] **Step 5: Implement exporter** — `backend/compose/exporter.py`:

```python
"""Export job body: bake a saved SceneDoc into merged.ply + meshes + zip."""
from __future__ import annotations

import os
import shutil
import zipfile
from typing import Any, Callable

from .bake import ColorAdjust, Crop, Placement, merge_clouds, transform_cloud
from .glb import placement_matrix, wrap_with_transform
from .models import SceneDoc, SceneObject
from .plyio import read_ply, write_ply
from .store import ComposeStore

ProgressFn = Callable[..., None]


def placement_for(obj: SceneObject) -> Placement:
    t = obj.transform
    crop = None
    if obj.crop is not None:
        crop = Crop(center=obj.crop.center, half_size=obj.crop.halfSize,
                    quaternion_xyzw=obj.crop.quaternion)
    color = None
    if obj.color is not None:
        color = ColorAdjust(exposure=obj.color.exposure, tint=obj.color.tint,
                            saturation=obj.color.saturation)
    return Placement(position=t.position, quaternion_xyzw=t.quaternion, scale=t.scale,
                     crop=crop, color=color)


def missing_assets(doc: SceneDoc, store: ComposeStore) -> list[str]:
    return [o.asset for o in doc.objects if o.visible and store.asset_path(o.asset) is None]


def run_export(doc: SceneDoc, store: ComposeStore, on_progress: ProgressFn) -> dict[str, Any]:
    missing = missing_assets(doc, store)
    if missing:
        raise FileNotFoundError(f"Missing assets: {', '.join(missing)}")
    visible = [o for o in doc.objects if o.visible]
    splats = [o for o in visible if o.kind == "splat"]
    meshes = [o for o in visible if o.kind == "mesh"]
    on_progress("compose_load", 1.0, f"{len(visible)} obje hazır",
                {"splats": len(splats), "meshes": len(meshes)})

    store.exports_dir.mkdir(parents=True, exist_ok=True)
    tmp = store.exports_dir / f"{doc.id}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    final = store.export_dir(doc.id)
    try:
        clouds = []
        for i, obj in enumerate(splats):
            on_progress("compose_bake", i / max(len(splats), 1), f"{obj.name} işleniyor", {})
            clouds.append(transform_cloud(read_ply(store.asset_path(obj.asset)), placement_for(obj)))
        merged = merge_clouds(clouds)
        del clouds
        on_progress("compose_write", 0.0, "merged.ply yazılıyor", {"gaussians": merged.count})
        write_ply(merged, tmp / "merged.ply")
        for obj in meshes:
            t = obj.transform
            wrap_with_transform(store.asset_path(obj.asset), tmp / "meshes" / f"{obj.id}.glb",
                                placement_matrix(t.position, t.quaternion, t.scale))
        (tmp / "scene.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    on_progress("compose_write", 0.6, "zip hazırlanıyor", {})
    zip_path = store.export_zip(doc.id)
    zip_tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(zip_tmp, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(final.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(final).as_posix())
    os.replace(zip_tmp, zip_path)
    on_progress("compose_write", 1.0, "Export hazır", {})
    return {
        "scene_id": doc.id,
        "gaussians": merged.count,
        "meshes": len(meshes),
        "download_url": f"/compose/exports/{doc.id}/download",
    }
```

- [ ] **Step 6: Implement routes** — `backend/compose/routes.py`:

```python
"""/compose API: assets, scene documents and export jobs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .exporter import missing_assets, run_export
from .models import Asset, CreateSceneRequest, ExportResponse, SceneDoc, SceneSummary
from .store import AssetError, ComposeStore

_MEDIA_TYPES = {"splat": "application/octet-stream", "mesh": "model/gltf-binary"}


def _kind_errors(doc: SceneDoc, store: ComposeStore) -> list[str]:
    errors = []
    for obj in doc.objects:
        asset = store.get_asset(obj.asset)
        if asset is not None and asset.kind != obj.kind:
            errors.append(f"object {obj.id} is a {obj.kind} but asset {obj.asset} is a {asset.kind}")
    return errors


def build_compose_router(data_root: str | Path, get_manager: Callable[[], Any]) -> APIRouter:
    store = ComposeStore(data_root)
    router = APIRouter(prefix="/compose", tags=["compose"])

    @router.get("/assets", response_model=list[Asset])
    def list_assets() -> list[Asset]:
        return store.list_assets()

    @router.post("/assets", response_model=Asset, status_code=201)
    def upload_asset(file: UploadFile = File(...)) -> Asset:
        try:
            return store.save_upload(file.filename or "", file.file)
        except AssetError as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.get("/assets/{asset_id}/file")
    def asset_file(asset_id: str) -> FileResponse:
        asset = store.get_asset(asset_id)
        path = store.asset_path(asset_id) if asset else None
        if asset is None or path is None:
            raise HTTPException(404, f"Asset not found: {asset_id}")
        return FileResponse(path, media_type=_MEDIA_TYPES[asset.kind],
                            filename=f"{asset.name}{path.suffix}")

    @router.get("/scenes", response_model=list[SceneSummary])
    def list_scenes() -> list[SceneSummary]:
        return store.list_scenes()

    @router.post("/scenes", response_model=SceneDoc, status_code=201)
    def create_scene(req: CreateSceneRequest) -> SceneDoc:
        base = store.get_asset(req.base_asset)
        if base is None:
            raise HTTPException(400, f"Base asset not found: {req.base_asset}")
        if base.kind != "splat":
            raise HTTPException(400, "Base asset must be a splat (.ply)")
        return store.create_scene(req.name, base)

    @router.get("/scenes/{scene_id}", response_model=SceneDoc)
    def get_scene(scene_id: str) -> SceneDoc:
        doc = store.load_scene(scene_id)
        if doc is None:
            raise HTTPException(404, f"Scene not found: {scene_id}")
        return doc

    @router.put("/scenes/{scene_id}", response_model=SceneDoc)
    def save_scene(scene_id: str, doc: SceneDoc) -> SceneDoc:
        if doc.id != scene_id:
            raise HTTPException(400, "Scene id in body does not match the URL")
        errors = _kind_errors(doc, store)
        if errors:
            raise HTTPException(400, "; ".join(errors))
        store.save_scene(doc)
        return doc

    @router.post("/scenes/{scene_id}/export", response_model=ExportResponse, status_code=202)
    def export_scene(scene_id: str) -> ExportResponse:
        doc = store.load_scene(scene_id)
        if doc is None:
            raise HTTPException(404, f"Scene not found: {scene_id}")
        missing = missing_assets(doc, store)
        if missing:
            raise HTTPException(400, f"Missing assets: {', '.join(missing)}")
        if not any(o.visible and o.kind == "splat" for o in doc.objects):
            raise HTTPException(400, "No visible splat objects to export")
        manager = get_manager()
        job = manager.create(scene=f"compose-{scene_id}")
        manager.submit(job.id, lambda cb: run_export(doc, store, cb))
        return ExportResponse(job_id=job.id)

    @router.get("/exports/{scene_id}/download")
    def download_export(scene_id: str) -> FileResponse:
        doc = store.load_scene(scene_id)
        zip_path = store.export_zip(scene_id) if doc else None
        if doc is None or zip_path is None or not zip_path.is_file():
            raise HTTPException(404, f"No export for scene: {scene_id}")
        return FileResponse(zip_path, media_type="application/zip", filename=f"{doc.name}.zip")

    return router
```

- [ ] **Step 7: Run** — `... -m pytest tests/compose -q` → PASS.
- [ ] **Step 8: Commit** — `git commit -m "feat(compose): add scene store, export job and /compose API"`

---

### Task 7: JobManager + api.py wiring

**Files:** Modify `backend/job_manager.py`, `backend/api.py`; Create `tests/compose/test_job_manager_compose.py`

- [ ] **Step 1: Failing tests** — `tests/compose/test_job_manager_compose.py`:

```python
from __future__ import annotations

import pytest

from backend.job_manager import JobManager, _compute_overall


def test_compose_phases_have_their_own_progress_scale():
    assert _compute_overall("compose_load", 1.0) == pytest.approx(0.05)
    assert _compute_overall("compose_bake", 0.5) == pytest.approx(0.40)
    assert _compute_overall("compose_write", 1.0) == pytest.approx(1.0)


def test_known_pipeline_phases_are_unchanged():
    assert _compute_overall("training", 0.0) == pytest.approx((0.05 + 0.15 + 0.02) / 0.90)


def test_unknown_phase_returns_none():
    assert _compute_overall("stage_input", 0.3) is None


def test_unknown_phase_never_jumps_to_done_or_goes_backwards(tmp_path):
    manager = JobManager(data_dir=tmp_path)
    fresh = manager.create(scene="img")
    manager._update_phase(fresh.id, "stage_input", 0.3, "", {})
    assert manager.get(fresh.id).overall_progress == pytest.approx(0.3)

    trained = manager.create(scene="vid")
    manager._update_phase(trained.id, "export", 1.0, "", {})
    manager._update_phase(trained.id, "eval", 0.1, "", {})
    assert manager.get(trained.id).overall_progress == pytest.approx(1.0)
    manager.shutdown(wait=False)


def test_completed_job_uses_result_download_url(tmp_path):
    manager = JobManager(data_dir=tmp_path)
    job = manager.create(scene="compose-s_1")
    manager._mark_completed(job.id, result={"download_url": "/compose/exports/s_1/download"})
    assert manager.get(job.id).download_url == "/compose/exports/s_1/download"
    manager.shutdown(wait=False)
```

(Check `JobManager.shutdown` signature with `grep -n "def shutdown" backend/job_manager.py`; if absent, use `manager._executor.shutdown(wait=False)`.)

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** in `backend/job_manager.py`:

Replace `_compute_overall` with:

```python
# Scene-composer export jobs have their own short phase sequence.
COMPOSE_PHASE_WEIGHTS = {
    "compose_load":  0.05,
    "compose_bake":  0.70,
    "compose_write": 0.25,
}


def _progress_in(weights: dict[str, float], phase_name: str, phase_progress: float) -> float:
    total = sum(weights.values()) or 1.0
    done = 0.0
    for name, weight in weights.items():
        if name == phase_name:
            done += weight * max(0.0, min(1.0, phase_progress))
            break
        done += weight
    return min(done / total, 1.0)


def _compute_overall(phase_name: str, phase_progress: float,
                     skip_foundation: bool = True) -> float | None:
    """Hangi fazdayız + o faz ne kadar tamamlandı → toplam ilerleme (0-1).

    Bilinmeyen fazlar için None döner; çağıran taraf ilerlemeyi geri almadan
    kendi başına günceller.
    """
    if phase_name in COMPOSE_PHASE_WEIGHTS:
        return _progress_in(COMPOSE_PHASE_WEIGHTS, phase_name, phase_progress)
    if phase_name in PHASE_WEIGHTS:
        weights = dict(PHASE_WEIGHTS)
        if skip_foundation:
            weights["foundation"] = 0.0
        return _progress_in(weights, phase_name, phase_progress)
    return None
```

In `_update_phase`, replace `job.overall_progress = _compute_overall(phase, progress)` with:

```python
            overall = _compute_overall(phase, progress)
            if overall is None:
                # Bilinmeyen faz (eval, image-to-splat aşamaları, ...): hemen
                # %100'e atlama ve geri gitme — mevcut değerle maksimumunu al.
                overall = max(job.overall_progress, max(0.0, min(1.0, progress)))
            job.overall_progress = overall
```

In `_mark_completed`, replace the "İndirilebilir URL'i set et" block with:

```python
            if result and result.get("download_url"):
                job.download_url = result["download_url"]
            else:
                # İndirilebilir URL'i set et
                ply_dir = self.data_dir / job.scene / "output" / "ply"
                if ply_dir.exists():
                    job.ply_dir = str(ply_dir)
                    job.download_url = f"/download/{job_id}"
```

In `backend/api.py`:

```python
from .compose.routes import build_compose_router
from .config import DATA_ROOT, default_config, cloud_config, local_max_config, safe_4d_8gb_config, scene_paths
...
app.include_router(static_notebook_router)
app.include_router(build_compose_router(DATA_ROOT, get_manager))
...
    allow_methods=["GET", "POST", "PUT"],
```

- [ ] **Step 4: Run** — `... -m pytest tests/compose tests/test_api_static_presets.py -q` → PASS. Also `python -c "import backend.api"` must succeed.
- [ ] **Step 5: Commit** — `git commit -m "feat(compose): wire /compose router and export job progress"`

---

### Task 8: GPU parity test (optional marker)

**Files:** Create `tests/compose/test_gpu_parity.py`

- [ ] **Step 1: Write test**

```python
from __future__ import annotations

import numpy as np
import pytest

from backend.compose.bake import Placement, quat_to_matrix, transform_cloud
from tests.compose.helpers import random_cloud

pytestmark = pytest.mark.gpu


def _render(torch, rasterization, cloud, viewmat):
    dev = "cuda"
    means = torch.tensor(cloud.means, device=dev)
    quats = torch.tensor(cloud.quats, device=dev)
    scales = torch.exp(torch.tensor(cloud.log_scales, device=dev))
    opac = torch.sigmoid(torch.tensor(cloud.opacities, device=dev))
    sh = torch.cat([torch.tensor(cloud.sh_dc, device=dev)[:, None], torch.tensor(cloud.sh_rest, device=dev)], 1)
    k = torch.tensor([[200.0, 0, 64], [0, 200.0, 64], [0, 0, 1]], device=dev)[None]
    img, _, _ = rasterization(means, quats, scales, opac, sh,
                              torch.tensor(viewmat, dtype=torch.float32, device=dev)[None], k, 128, 128,
                              sh_degree=3)
    return img[0].cpu().numpy()


def test_rotated_scene_matches_rotated_camera():
    torch = pytest.importorskip("torch")
    gsplat = pytest.importorskip("gsplat")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cloud = random_cloud(3000, degree=3, seed=11)
    cloud.means *= 0.5
    cloud.opacities[:] = 2.0
    q = np.array([0.2, -0.4, 0.3, 0.84])
    q /= np.linalg.norm(q)
    r = quat_to_matrix(tuple(q))
    view = np.eye(4)
    view[2, 3] = 4.0  # camera 4 units in front of the cloud
    view_rot = view @ np.block([[r.T, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])

    original = _render(torch, gsplat.rasterization, cloud, view)
    rotated = _render(torch, gsplat.rasterization, transform_cloud(cloud, Placement(quaternion_xyzw=tuple(q))), view_rot)

    assert np.abs(original - rotated).max() < 2e-3
```

- [ ] **Step 2: Run** — `... -m pytest tests/compose/test_gpu_parity.py -q` → PASS on the local GPU (skips elsewhere). **Commit** — `git commit -m "test(compose): gsplat parity for rotated SH"`

---

### Task 9: Frontend types, reducer, math, API client

**Files:**
- Modify: `frontend/src/api.ts` (export `authHeaders` and `fetchJson`)
- Create: `frontend/src/compose/{types.ts,sceneDoc.ts,transformMath.ts,colorMath.ts,composeApi.ts}`
- Create tests: `frontend/src/compose/__tests__/{sceneDoc.test.ts,golden.test.ts,composeApi.test.ts}`

- [ ] **Step 1: Failing tests**

`frontend/src/compose/__tests__/golden.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import golden from "../../../../tests/fixtures/compose_golden.json";
import { applyColor } from "../colorMath";
import { applyTransform, isInsideCrop, uniformScaleFrom } from "../transformMath";
import type { ColorAdjust, CropBox, Transform, Vec3 } from "../types";

function close(a: number[], b: number[], eps = 1e-5) {
  a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(eps));
}

describe("preview math matches the backend golden fixture", () => {
  it("transforms points like the bake", () => {
    for (const c of golden.transform_cases) {
      c.points.forEach((p, i) => close(applyTransform(c.transform as Transform, p as Vec3), c.expected[i]));
    }
  });

  it("crops like the bake", () => {
    for (const c of golden.crop_cases) {
      const inside = c.points.map((p) => isInsideCrop(c.crop as CropBox, p as Vec3));
      expect(inside).toEqual(c.inside);
      expect(inside).toContain(true);
      expect(inside).toContain(false);
    }
  });

  it("colour-matches like the bake", () => {
    for (const c of golden.color_cases) {
      c.rgb.forEach((rgb, i) => close(applyColor(c.color as ColorAdjust, rgb as Vec3), c.expected[i]));
    }
  });
});

describe("uniformScaleFrom", () => {
  it("follows the axis the gizmo changed", () => {
    expect(uniformScaleFrom(1, { x: 1, y: 2.5, z: 1 })).toBe(2.5);
    expect(uniformScaleFrom(2, { x: 0.5, y: 2, z: 2 })).toBe(0.5);
    expect(uniformScaleFrom(1, { x: -3, y: 1, z: 1 })).toBeGreaterThan(0);
  });
});
```

`frontend/src/compose/__tests__/sceneDoc.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { composeReducer, initialComposeState, newObjectId } from "../sceneDoc";
import type { SceneDoc, SceneObject } from "../types";

const base: SceneObject = {
  id: "o_base", kind: "splat", asset: "scene__garden", name: "garden", role: "base", visible: true,
  transform: { position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
};
const statue: SceneObject = {
  id: "o_statue", kind: "splat", asset: "a_1", name: "statue", role: "object", visible: true,
  transform: { position: [1, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
};
const doc: SceneDoc = { version: 1, id: "s_1", name: "x", viewUp: "y", objects: [base] };

function loaded() {
  return composeReducer(initialComposeState, { type: "load", doc });
}

describe("composeReducer", () => {
  it("loads clean and adding selects + dirties", () => {
    const s0 = loaded();
    expect(s0.dirty).toBe(false);
    const s1 = composeReducer(s0, { type: "add", object: statue });
    expect(s1.doc?.objects.map((o) => o.id)).toEqual(["o_base", "o_statue"]);
    expect(s1.selectedId).toBe("o_statue");
    expect(s1.dirty).toBe(true);
  });

  it("never moves, removes or duplicates the base", () => {
    const s0 = loaded();
    const moved = composeReducer(s0, {
      type: "setTransform", id: "o_base",
      transform: { position: [5, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
    });
    expect(moved).toBe(s0);
    expect(composeReducer(s0, { type: "remove", id: "o_base" })).toBe(s0);
    expect(composeReducer(s0, { type: "duplicate", id: "o_base", newId: "o_x" })).toBe(s0);
  });

  it("allows cropping and colouring the base", () => {
    const s = composeReducer(loaded(), {
      type: "setCrop", id: "o_base",
      crop: { center: [0, 0, 0], halfSize: [1, 1, 1], quaternion: [0, 0, 0, 1] },
    });
    expect(s.doc?.objects[0].crop?.halfSize).toEqual([1, 1, 1]);
  });

  it("duplicates, removes and clears selection", () => {
    let s = composeReducer(loaded(), { type: "add", object: statue });
    s = composeReducer(s, { type: "duplicate", id: "o_statue", newId: "o_copy" });
    expect(s.doc?.objects[2]).toMatchObject({ id: "o_copy", name: "statue kopya", asset: "a_1" });
    expect(s.selectedId).toBe("o_copy");
    s = composeReducer(s, { type: "remove", id: "o_copy" });
    expect(s.selectedId).toBeNull();
    expect(s.doc?.objects).toHaveLength(2);
  });

  it("ignores crop/colour on meshes", () => {
    const mesh: SceneObject = { ...statue, id: "o_mesh", kind: "mesh" };
    const s0 = composeReducer(loaded(), { type: "add", object: mesh });
    const s1 = composeReducer(s0, {
      type: "setColor", id: "o_mesh", color: { exposure: 1, tint: [1, 1, 1], saturation: 1 },
    });
    expect(s1).toBe(s0);
  });

  it("markSaved clears dirty", () => {
    const s = composeReducer(composeReducer(loaded(), { type: "add", object: statue }), { type: "markSaved" });
    expect(s.dirty).toBe(false);
  });

  it("generates safe unique ids", () => {
    const a = newObjectId();
    expect(a).toMatch(/^o_[0-9a-f]{12}$/);
    expect(newObjectId()).not.toBe(a);
  });
});
```

`frontend/src/compose/__tests__/composeApi.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { assetFileUrl, saveScene, uploadAsset } from "../composeApi";
import type { SceneDoc } from "../types";

afterEach(() => vi.unstubAllGlobals());

function okJson(body: unknown) {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

describe("compose API", () => {
  it("uploads as multipart form data", async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({ id: "a_1" }));
    vi.stubGlobal("fetch", fetchMock);
    await uploadAsset(new File(["ply"], "statue.ply"));
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/compose\/assets$/);
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("file")).toBeInstanceOf(File);
  });

  it("saves scenes with PUT and a JSON body", async () => {
    const doc: SceneDoc = { version: 1, id: "s_1", name: "x", viewUp: "y", objects: [] };
    const fetchMock = vi.fn().mockResolvedValue(okJson(doc));
    vi.stubGlobal("fetch", fetchMock);
    await saveScene(doc);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/compose\/scenes\/s_1$/);
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body)).toEqual(doc);
  });

  it("builds asset file urls", () => {
    expect(assetFileUrl("scene__garden")).toMatch(/\/compose\/assets\/scene__garden\/file$/);
  });
});
```

- [ ] **Step 2: Run** — `cd frontend && npx vitest run src/compose` → FAIL (modules missing).

- [ ] **Step 3: Implement**

`frontend/src/api.ts`: change `function authHeaders(` → `export function authHeaders(` and `async function fetchJson<T>(` → `export async function fetchJson<T>(`.

`frontend/src/compose/types.ts`:

```ts
/** TS mirror of backend/compose/models.py (SceneDoc v1). */
export type Vec3 = [number, number, number];
/** three.js order: x, y, z, w */
export type Quat = [number, number, number, number];

export interface Transform {
  position: Vec3;
  quaternion: Quat;
  scale: number;
}

export interface CropBox {
  center: Vec3;
  halfSize: Vec3;
  quaternion: Quat;
}

export interface ColorAdjust {
  exposure: number;
  tint: Vec3;
  saturation: number;
}

export type ObjectKind = "splat" | "mesh";
export type ViewUp = "y" | "-y";

export interface SceneObject {
  id: string;
  kind: ObjectKind;
  asset: string;
  name: string;
  role: "base" | "object";
  visible: boolean;
  transform: Transform;
  crop?: CropBox | null;
  color?: ColorAdjust | null;
}

export interface SceneDoc {
  version: 1;
  id: string;
  name: string;
  viewUp: ViewUp;
  objects: SceneObject[];
}

export interface Asset {
  id: string;
  kind: ObjectKind;
  name: string;
  size_bytes: number;
  source: "upload" | "pipeline";
  created_ts: number;
}

export interface SceneSummary {
  id: string;
  name: string;
  updated_ts: number;
  object_count: number;
}

export type GizmoMode = "translate" | "rotate" | "scale";

export const IDENTITY_TRANSFORM: Transform = { position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 };
export const DEFAULT_COLOR: ColorAdjust = { exposure: 0, tint: [1, 1, 1], saturation: 1 };
```

`frontend/src/compose/sceneDoc.ts`:

```ts
import type { ColorAdjust, CropBox, SceneDoc, SceneObject, Transform, ViewUp } from "./types";

export interface ComposeState {
  doc: SceneDoc | null;
  selectedId: string | null;
  dirty: boolean;
}

export const initialComposeState: ComposeState = { doc: null, selectedId: null, dirty: false };

export type ComposeAction =
  | { type: "load"; doc: SceneDoc }
  | { type: "close" }
  | { type: "markSaved" }
  | { type: "select"; id: string | null }
  | { type: "add"; object: SceneObject }
  | { type: "remove"; id: string }
  | { type: "duplicate"; id: string; newId: string }
  | { type: "setTransform"; id: string; transform: Transform }
  | { type: "setCrop"; id: string; crop: CropBox | null }
  | { type: "setColor"; id: string; color: ColorAdjust | null }
  | { type: "setVisible"; id: string; visible: boolean }
  | { type: "rename"; id: string; name: string }
  | { type: "setViewUp"; viewUp: ViewUp };

export function newObjectId(): string {
  const bytes = new Uint8Array(6);
  crypto.getRandomValues(bytes);
  return `o_${Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")}`;
}

function find(state: ComposeState, id: string): SceneObject | undefined {
  return state.doc?.objects.find((o) => o.id === id);
}

function update(state: ComposeState, id: string, fn: (o: SceneObject) => SceneObject): ComposeState {
  if (!state.doc || !find(state, id)) return state;
  return {
    ...state,
    dirty: true,
    doc: { ...state.doc, objects: state.doc.objects.map((o) => (o.id === id ? fn(o) : o)) },
  };
}

export function composeReducer(state: ComposeState, action: ComposeAction): ComposeState {
  switch (action.type) {
    case "load":
      return { doc: action.doc, selectedId: null, dirty: false };
    case "close":
      return initialComposeState;
    case "markSaved":
      return { ...state, dirty: false };
    case "select":
      return { ...state, selectedId: action.id };
    case "add":
      if (!state.doc) return state;
      return {
        doc: { ...state.doc, objects: [...state.doc.objects, action.object] },
        selectedId: action.object.id,
        dirty: true,
      };
    case "remove": {
      const target = find(state, action.id);
      if (!state.doc || !target || target.role === "base") return state;
      return {
        doc: { ...state.doc, objects: state.doc.objects.filter((o) => o.id !== action.id) },
        selectedId: state.selectedId === action.id ? null : state.selectedId,
        dirty: true,
      };
    }
    case "duplicate": {
      const target = find(state, action.id);
      if (!state.doc || !target || target.role === "base") return state;
      const copy: SceneObject = { ...structuredClone(target), id: action.newId, name: `${target.name} kopya` };
      return { doc: { ...state.doc, objects: [...state.doc.objects, copy] }, selectedId: copy.id, dirty: true };
    }
    case "setTransform": {
      const target = find(state, action.id);
      if (!target || target.role === "base") return state;
      return update(state, action.id, (o) => ({ ...o, transform: action.transform }));
    }
    case "setCrop": {
      const target = find(state, action.id);
      if (!target || target.kind !== "splat") return state;
      return update(state, action.id, (o) => ({ ...o, crop: action.crop }));
    }
    case "setColor": {
      const target = find(state, action.id);
      if (!target || target.kind !== "splat") return state;
      return update(state, action.id, (o) => ({ ...o, color: action.color }));
    }
    case "setVisible":
      return update(state, action.id, (o) => ({ ...o, visible: action.visible }));
    case "rename":
      return update(state, action.id, (o) => ({ ...o, name: action.name }));
    case "setViewUp":
      if (!state.doc || state.doc.viewUp === action.viewUp) return state;
      return { ...state, doc: { ...state.doc, viewUp: action.viewUp }, dirty: true };
  }
}
```

`frontend/src/compose/transformMath.ts`:

```ts
import { Matrix4, Object3D, Quaternion, Vector3 } from "three";
import type { CropBox, Transform, Vec3 } from "./types";

/** Same order as backend bake: scale → rotate → translate. */
export function transformMatrix(t: Transform): Matrix4 {
  return new Matrix4().compose(
    new Vector3(...t.position),
    new Quaternion(...t.quaternion).normalize(),
    new Vector3(t.scale, t.scale, t.scale),
  );
}

export function applyTransform(t: Transform, p: Vec3): Vec3 {
  const v = new Vector3(...p).applyMatrix4(transformMatrix(t));
  return [v.x, v.y, v.z];
}

/** Crop test in the object's local frame: |R_cᵀ (p − c)| ≤ halfSize (matches bake.crop_mask). */
export function isInsideCrop(crop: CropBox, localPoint: Vec3): boolean {
  const inv = new Quaternion(...crop.quaternion).normalize().invert();
  const v = new Vector3(...localPoint).sub(new Vector3(...crop.center)).applyQuaternion(inv);
  return (
    Math.abs(v.x) <= crop.halfSize[0] && Math.abs(v.y) <= crop.halfSize[1] && Math.abs(v.z) <= crop.halfSize[2]
  );
}

/** TransformControls scales one axis at a time; follow the axis that changed most. */
export function uniformScaleFrom(prev: number, s: { x: number; y: number; z: number }): number {
  let best = s.x;
  for (const c of [s.y, s.z]) if (Math.abs(c - prev) > Math.abs(best - prev)) best = c;
  return Math.max(Math.abs(best), 1e-4);
}

export function readTransform(obj: Object3D): Transform {
  const q = obj.quaternion;
  return {
    position: [obj.position.x, obj.position.y, obj.position.z],
    quaternion: [q.x, q.y, q.z, q.w],
    scale: obj.scale.x,
  };
}

export function readCrop(obj: Object3D): CropBox {
  const q = obj.quaternion;
  const h = (v: number) => Math.max(Math.abs(v), 1e-3);
  return {
    center: [obj.position.x, obj.position.y, obj.position.z],
    halfSize: [h(obj.scale.x), h(obj.scale.y), h(obj.scale.z)],
    quaternion: [q.x, q.y, q.z, q.w],
  };
}
```

`frontend/src/compose/colorMath.ts`:

```ts
import type { ColorAdjust, Vec3 } from "./types";

const LUMA: Vec3 = [0.2126, 0.7152, 0.0722]; // Rec.709, same as backend bake

/** Row-major 3×3 M = S · diag(gain) (matches backend bake.color_matrix). */
export function colorMatrix(c: ColorAdjust): number[] {
  const g = 2 ** c.exposure;
  const gain = c.tint.map((t) => t * g);
  const m: number[] = [];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const s = LUMA[j] + c.saturation * ((i === j ? 1 : 0) - LUMA[j]);
      m.push(s * gain[j]);
    }
  }
  return m;
}

export function applyColor(c: ColorAdjust, rgb: Vec3): Vec3 {
  const m = colorMatrix(c);
  return [0, 1, 2].map((i) => m[3 * i] * rgb[0] + m[3 * i + 1] * rgb[1] + m[3 * i + 2] * rgb[2]) as Vec3;
}

export function tintToHex(t: Vec3): string {
  return `#${t.map((v) => Math.round(Math.min(Math.max(v, 0), 1) * 255).toString(16).padStart(2, "0")).join("")}`;
}

export function hexToTint(hex: string): Vec3 {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
```

`frontend/src/compose/composeApi.ts`:

```ts
import { fetchJson } from "../api";
import { getApiBase, withTokenParam } from "../connection";
import type { Asset, SceneDoc, SceneSummary } from "./types";

const JSON_HEADERS = { "Content-Type": "application/json" };
const enc = encodeURIComponent;

export function listAssets(): Promise<Asset[]> {
  return fetchJson<Asset[]>("/compose/assets");
}

export function uploadAsset(file: File): Promise<Asset> {
  const form = new FormData();
  form.append("file", file);
  return fetchJson<Asset>("/compose/assets", { method: "POST", body: form });
}

export function assetFileUrl(assetId: string): string {
  return withTokenParam(`${getApiBase()}/compose/assets/${enc(assetId)}/file`);
}

export function listScenes(): Promise<SceneSummary[]> {
  return fetchJson<SceneSummary[]>("/compose/scenes");
}

export function createScene(name: string, baseAsset: string): Promise<SceneDoc> {
  return fetchJson<SceneDoc>("/compose/scenes", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ name, base_asset: baseAsset }),
  });
}

export function getScene(id: string): Promise<SceneDoc> {
  return fetchJson<SceneDoc>(`/compose/scenes/${enc(id)}`);
}

export function saveScene(doc: SceneDoc): Promise<SceneDoc> {
  return fetchJson<SceneDoc>(`/compose/scenes/${enc(doc.id)}`, {
    method: "PUT",
    headers: JSON_HEADERS,
    body: JSON.stringify(doc),
  });
}

export function exportScene(id: string): Promise<{ job_id: string }> {
  return fetchJson<{ job_id: string }>(`/compose/scenes/${enc(id)}/export`, { method: "POST" });
}

export function backendUrl(path: string): string {
  return withTokenParam(`${getApiBase()}${path}`);
}
```

- [ ] **Step 4: Run** — `cd frontend && npx vitest run src/compose` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(compose): add scene document reducer, preview math and API client"`

---

### Task 10: 3D components (registry, splat, mesh, viewport, snap)

**Files:** Create `frontend/src/compose/{registry.ts,SplatObject.tsx,MeshObject.tsx,ComposeViewport.tsx,snap.ts}`

No unit tests (WebGL); verified by typecheck here and manually in Task 12.

- [ ] **Step 1: registry** — `frontend/src/compose/registry.ts`:

```ts
import type { Box3, Group, Object3D } from "three";
import type { SplatMesh } from "@sparkjsdev/spark";

/** Live three.js objects behind each scene object (filled by the renderers). */
export interface RegistryEntry {
  group: Group;
  splat?: SplatMesh;
  cropTarget?: Object3D;
  /** Splat bounds in the object's local (raw file) frame, centres only. */
  localBounds?: Box3;
}

export class ObjectRegistry {
  private entries = new Map<string, RegistryEntry>();

  set(id: string, entry: RegistryEntry): void {
    this.entries.set(id, entry);
  }

  patch(id: string, patch: Partial<RegistryEntry>): void {
    const entry = this.entries.get(id);
    if (entry) this.entries.set(id, { ...entry, ...patch });
  }

  get(id: string): RegistryEntry | undefined {
    return this.entries.get(id);
  }

  delete(id: string, group?: Group): void {
    // Only drop the entry the caller owns (a remount may already have replaced it).
    if (!group || this.entries.get(id)?.group === group) this.entries.delete(id);
  }

  all(): [string, RegistryEntry][] {
    return Array.from(this.entries.entries());
  }
}
```

- [ ] **Step 2: SplatObject** — `frontend/src/compose/SplatObject.tsx`:

```tsx
import { useEffect, useMemo, useRef } from "react";
import type { ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import {
  SplatEdit, SplatEditRgbaBlendMode, SplatEditSdf, SplatEditSdfType, SplatMesh, dyno,
} from "@sparkjsdev/spark";
import { colorMatrix } from "./colorMath";
import type { ObjectRegistry } from "./registry";
import { isInsideCrop } from "./transformMath";
import { DEFAULT_COLOR, type SceneObject } from "./types";

interface Props {
  object: SceneObject;
  url: string;
  selected: boolean;
  registry: ObjectRegistry;
  onSelect: (id: string) => void;
  onError: (id: string, message: string) => void;
}

const noRaycast: THREE.Object3D["raycast"] = () => {};

/** Splat with crop (mesh-local SplatEdit) and colour matrix (objectModifier). */
export function SplatObject({ object, url, selected, registry, onSelect, onError }: Props) {
  const groupRef = useRef<THREE.Group>(null);
  const colorMat = useMemo(() => new THREE.Matrix3(), []);

  const mesh = useMemo(() => {
    const uniform = new dyno.DynoMat3({ value: colorMat });
    const modifier = dyno.dynoBlock(
      { gsplat: dyno.Gsplat },
      { gsplat: dyno.Gsplat },
      ({ gsplat }) => {
        if (!gsplat) throw new Error("gsplat input missing");
        const { rgb } = dyno.splitGsplat(gsplat).outputs;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const mixed = dyno.mul(uniform, rgb) as any;
        return { gsplat: dyno.combineGsplat({ gsplat, rgb: mixed }) };
      },
    );
    return new SplatMesh({ url, objectModifier: modifier, editable: true });
  }, [url, colorMat]);

  const crop = useMemo(() => {
    const edit = new SplatEdit({ rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY, softEdge: 0, sdfSmooth: 0 });
    // Inverted box: everything OUTSIDE gets opacity × 0. SDF scale = half size.
    const sdf = new SplatEditSdf({ type: SplatEditSdfType.BOX, invert: true, opacity: 0, color: new THREE.Color(1, 1, 1) });
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)),
      new THREE.LineBasicMaterial({ color: 0xffb347 }),
    );
    outline.raycast = noRaycast;
    sdf.add(outline);
    edit.add(sdf);
    return { edit, sdf, outline };
  }, []);

  useEffect(() => {
    let cancelled = false;
    mesh.initialized
      .then(() => {
        if (!cancelled) registry.patch(object.id, { localBounds: mesh.getBoundingBox(true) });
      })
      .catch((e: unknown) => {
        if (!cancelled) onError(object.id, e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
      mesh.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mesh]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    registry.set(object.id, { group, splat: mesh });
    return () => registry.delete(object.id, group);
  }, [object.id, mesh, registry]);

  useEffect(() => {
    const m = colorMatrix(object.color ?? DEFAULT_COLOR);
    colorMat.set(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
    mesh.updateVersion();
  }, [object.color, colorMat, mesh]);

  useEffect(() => {
    const c = object.crop;
    if (!c) {
      mesh.remove(crop.edit);
      registry.patch(object.id, { cropTarget: undefined });
    } else {
      crop.sdf.position.set(...c.center);
      crop.sdf.quaternion.set(...c.quaternion);
      crop.sdf.scale.set(...c.halfSize);
      if (crop.edit.parent !== mesh) mesh.add(crop.edit);
      registry.patch(object.id, { cropTarget: crop.sdf });
    }
    mesh.updateVersion();
  }, [object.crop, object.id, crop, mesh, registry]);

  useEffect(() => {
    crop.outline.visible = selected;
  }, [selected, crop]);

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > 4) return; // orbit drag, not a click
    if (object.crop) {
      const local = mesh.worldToLocal(e.point.clone());
      if (!isInsideCrop(object.crop, [local.x, local.y, local.z])) return; // hidden splat: let it pass
    }
    e.stopPropagation();
    onSelect(object.id);
  };

  const t = object.transform;
  return (
    <group
      ref={groupRef}
      position={t.position}
      quaternion={t.quaternion}
      scale={t.scale}
      visible={object.visible}
      onClick={handleClick}
    >
      <primitive object={mesh} />
    </group>
  );
}
```

- [ ] **Step 3: MeshObject** — `frontend/src/compose/MeshObject.tsx`:

```tsx
import { useEffect, useRef, useState } from "react";
import type { ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { ObjectRegistry } from "./registry";
import type { SceneObject } from "./types";

interface Props {
  object: SceneObject;
  url: string;
  registry: ObjectRegistry;
  onSelect: (id: string) => void;
  onError: (id: string, message: string) => void;
}

function disposeTree(root: THREE.Object3D) {
  root.traverse((node) => {
    const mesh = node as THREE.Mesh;
    mesh.geometry?.dispose();
    const mats = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
    mats.forEach((m) => m.dispose());
  });
}

export function MeshObject({ object, url, registry, onSelect, onError }: Props) {
  const groupRef = useRef<THREE.Group>(null);
  const [scene, setScene] = useState<THREE.Group | null>(null);

  useEffect(() => {
    let cancelled = false;
    let loaded: THREE.Group | null = null;
    new GLTFLoader()
      .loadAsync(url)
      .then((gltf) => {
        loaded = gltf.scene;
        if (cancelled) disposeTree(loaded);
        else setScene(loaded);
      })
      .catch((e: unknown) => {
        if (!cancelled) onError(object.id, e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
      if (loaded) disposeTree(loaded);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;
    registry.set(object.id, { group });
    return () => registry.delete(object.id, group);
  }, [object.id, registry]);

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > 4) return;
    e.stopPropagation();
    onSelect(object.id);
  };

  const t = object.transform;
  return (
    <group
      ref={groupRef}
      position={t.position}
      quaternion={t.quaternion}
      scale={t.scale}
      visible={object.visible}
      onClick={handleClick}
    >
      {scene && <primitive object={scene} />}
    </group>
  );
}
```

- [ ] **Step 4: snap** — `frontend/src/compose/snap.ts`:

```ts
import * as THREE from "three";
import { SplatMesh } from "@sparkjsdev/spark";
import type { ObjectRegistry, RegistryEntry } from "./registry";
import { isInsideCrop } from "./transformMath";
import type { SceneDoc, SceneObject, Vec3 } from "./types";

function localBoundsWithCrop(obj: SceneObject, entry: RegistryEntry): THREE.Box3 | null {
  if (!entry.localBounds) return null;
  const box = entry.localBounds.clone();
  if (obj.crop) {
    const c = obj.crop;
    const m = new THREE.Matrix4().compose(
      new THREE.Vector3(...c.center),
      new THREE.Quaternion(...c.quaternion),
      new THREE.Vector3(...c.halfSize),
    );
    box.intersect(new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1)).applyMatrix4(m));
  }
  return box.isEmpty() ? null : box;
}

/** World-space bounds of an object (cropped for splats). */
export function worldBounds(obj: SceneObject, entry: RegistryEntry): THREE.Box3 | null {
  entry.group.updateWorldMatrix(true, true);
  if (obj.kind === "mesh") {
    const box = new THREE.Box3().setFromObject(entry.group);
    return box.isEmpty() ? null : box;
  }
  const local = localBoundsWithCrop(obj, entry);
  return local ? local.applyMatrix4(entry.group.matrixWorld) : null;
}

/**
 * New position that drops ``id`` onto the first visible surface below it
 * (along −up), or null when nothing is hit. Hits on cropped-away splats are skipped.
 */
export function snapToGround(doc: SceneDoc, registry: ObjectRegistry, id: string): Vec3 | null {
  const obj = doc.objects.find((o) => o.id === id);
  const entry = registry.get(id);
  if (!obj || !entry || obj.role === "base") return null;
  const box = worldBounds(obj, entry);
  if (!box) return null;

  const up = new THREE.Vector3(0, doc.viewUp === "-y" ? -1 : 1, 0);
  const center = box.getCenter(new THREE.Vector3());
  const bottomY = up.y > 0 ? box.min.y : box.max.y;
  const bottom = new THREE.Vector3(center.x, bottomY, center.z);
  const height = box.max.y - box.min.y;
  const origin = bottom.clone().addScaledVector(up, height / 2);

  const byMesh = new Map<THREE.Object3D, SceneObject>();
  const targets: THREE.Object3D[] = [];
  for (const [otherId, other] of registry.all()) {
    const otherObj = doc.objects.find((o) => o.id === otherId);
    if (!otherObj || otherId === id || !otherObj.visible) continue;
    targets.push(other.group);
    if (other.splat) byMesh.set(other.splat, otherObj);
  }

  const raycaster = new THREE.Raycaster(origin, up.clone().negate());
  for (const hit of raycaster.intersectObjects(targets, true)) {
    if (hit.object instanceof SplatMesh) {
      const owner = byMesh.get(hit.object);
      if (owner?.crop) {
        const local = hit.object.worldToLocal(hit.point.clone());
        if (!isInsideCrop(owner.crop, [local.x, local.y, local.z])) continue;
      }
    }
    const delta = hit.point.clone().sub(bottom).dot(up);
    const p = obj.transform.position;
    return [p[0] + up.x * delta, p[1] + up.y * delta, p[2] + up.z * delta];
  }
  return null;
}
```

- [ ] **Step 5: Viewport** — `frontend/src/compose/ComposeViewport.tsx`:

```tsx
import { useEffect, useMemo, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, TransformControls } from "@react-three/drei";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import * as THREE from "three";
import { SparkRenderer } from "@sparkjsdev/spark";
import { MeshObject } from "./MeshObject";
import type { ObjectRegistry } from "./registry";
import { SplatObject } from "./SplatObject";
import { readCrop, readTransform, uniformScaleFrom } from "./transformMath";
import type { CropBox, GizmoMode, SceneDoc, Transform } from "./types";

interface ViewportProps {
  doc: SceneDoc;
  active: boolean;
  selectedId: string | null;
  gizmoMode: GizmoMode;
  cropEditing: boolean;
  registry: ObjectRegistry;
  assetUrl: (assetId: string) => string;
  onSelect: (id: string | null) => void;
  onTransformCommit: (id: string, t: Transform) => void;
  onCropCommit: (id: string, crop: CropBox) => void;
  onError: (id: string, message: string) => void;
  onControls: (controls: OrbitControlsImpl | null) => void;
}

function SparkLayer() {
  const gl = useThree((s) => s.gl);
  const spark = useMemo(() => {
    const s = new SparkRenderer({ renderer: gl });
    s.raycast = () => {};
    return s;
  }, [gl]);
  useEffect(() => () => (spark as unknown as { dispose?: () => void }).dispose?.(), [spark]);
  return <primitive object={spark} />;
}

function ControlsBridge({ onControls }: { onControls: ViewportProps["onControls"] }) {
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null;
  useEffect(() => {
    onControls(controls);
    return () => onControls(null);
  }, [controls, onControls]);
  return null;
}

function Gizmo(props: Pick<ViewportProps, "doc" | "selectedId" | "gizmoMode" | "cropEditing" | "registry" | "onTransformCommit" | "onCropCommit">) {
  const { doc, selectedId, gizmoMode, cropEditing, registry, onTransformCommit, onCropCommit } = props;
  const obj = selectedId ? doc.objects.find((o) => o.id === selectedId) : undefined;
  const isCrop = !!obj && cropEditing && !!obj.crop;
  const [target, setTarget] = useState<THREE.Object3D | null>(null);

  // Registry entries appear asynchronously (after mount / load); re-check each frame.
  useFrame(() => {
    let next: THREE.Object3D | null = null;
    if (obj) {
      const entry = registry.get(obj.id);
      if (isCrop) next = entry?.cropTarget ?? null;
      else if (obj.role !== "base") next = entry?.group ?? null;
    }
    if (next !== target) setTarget(next);
  });

  if (!obj || !target) return null;
  return (
    <TransformControls
      object={target}
      mode={gizmoMode}
      space={isCrop ? "local" : "world"}
      onMouseDown={() => {
        target.userData.scaleAtStart = target.scale.x;
      }}
      onObjectChange={() => {
        if (!isCrop && gizmoMode === "scale") {
          target.scale.setScalar(uniformScaleFrom(target.userData.scaleAtStart ?? target.scale.x, target.scale));
        }
      }}
      onMouseUp={() => {
        if (isCrop) onCropCommit(obj.id, readCrop(target));
        else onTransformCommit(obj.id, readTransform(target));
      }}
    />
  );
}

export function ComposeViewport(props: ViewportProps) {
  const { doc, active, selectedId, registry, assetUrl, onSelect, onError, onControls } = props;
  const upSign = doc.viewUp === "-y" ? -1 : 1;
  return (
    <Canvas
      // Remount on viewUp change: OrbitControls reads camera.up only at construction.
      key={`${doc.id}:${doc.viewUp}`}
      frameloop={active ? "always" : "never"}
      camera={{ position: [0, 1.5 * upSign, 4], up: [0, upSign, 0], fov: 60, near: 0.01, far: 2000 }}
      onPointerMissed={() => onSelect(null)}
      style={{ width: "100%", height: "100%", background: "#101015" }}
    >
      <ambientLight intensity={0.7} />
      <directionalLight position={[5, 10 * upSign, 5]} intensity={1.2} />
      <SparkLayer />
      {doc.objects.map((o) =>
        o.kind === "splat" ? (
          <SplatObject
            key={o.id}
            object={o}
            url={assetUrl(o.asset)}
            selected={o.id === selectedId}
            registry={registry}
            onSelect={onSelect}
            onError={onError}
          />
        ) : (
          <MeshObject key={o.id} object={o} url={assetUrl(o.asset)} registry={registry} onSelect={onSelect} onError={onError} />
        ),
      )}
      <OrbitControls makeDefault enableDamping={false} />
      <ControlsBridge onControls={onControls} />
      <Gizmo {...props} />
    </Canvas>
  );
}
```

(`three-stdlib` ships with drei; if the type import path fails, use `import type { OrbitControls as OrbitControlsImpl } from "three/examples/jsm/controls/OrbitControls.js"`.)

- [ ] **Step 6: Typecheck** — `cd frontend && npx tsc --noEmit -p tsconfig.json` → no errors in `src/compose`. Fix typing of `dyno.mul` / `DynoMat3` generics with casts if needed.
- [ ] **Step 7: Commit** — `git commit -m "feat(compose): render splats and meshes with gizmo, crop and colour preview"`

---

### Task 11: Panels, page, App integration

**Files:** Create `frontend/src/compose/{ObjectListPanel.tsx,InspectorPanel.tsx,AssetPicker.tsx,ComposePage.tsx,compose.css}`; modify `frontend/src/App.tsx`, `frontend/src/App.test.tsx`, `frontend/src/components/JobsList.tsx`.

- [ ] **Step 1: Failing App test** — add to `frontend/src/App.test.tsx` a mock and test:

```tsx
vi.mock("./compose/ComposePage", () => ({ ComposePage: () => <div data-testid="compose-page" /> }));
...
  it("opens the scene composer tab", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Sahne Editörü" }));
    expect(screen.getByTestId("compose-page")).toBeInTheDocument();
  });
```

Run `npx vitest run src/App.test.tsx` → FAIL (no such button).

- [ ] **Step 2: Panels**

`frontend/src/compose/AssetPicker.tsx`:

```tsx
import { useRef, useState } from "react";
import type { Asset, ObjectKind } from "./types";

interface Props {
  assets: Asset[];
  kinds?: ObjectKind[];
  busy?: boolean;
  actionLabel: string;
  onPick: (asset: Asset) => void;
  onUpload: (file: File) => void;
}

function formatMb(bytes: number) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function AssetPicker({ assets, kinds = ["splat", "mesh"], busy, actionLabel, onPick, onUpload }: Props) {
  const [assetId, setAssetId] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const options = assets.filter((a) => kinds.includes(a.kind));
  const accept = kinds.map((k) => (k === "splat" ? ".ply" : ".glb")).join(",");
  const chosen = options.find((a) => a.id === assetId);

  return (
    <div className="compose-asset-picker">
      <select value={assetId} onChange={(e) => setAssetId(e.target.value)} disabled={busy}>
        <option value="">Asset seç…</option>
        {options.map((a) => (
          <option key={a.id} value={a.id}>
            {a.kind === "splat" ? "◉" : "▲"} {a.name} {a.source === "pipeline" ? "(pipeline)" : ""} · {formatMb(a.size_bytes)}
          </option>
        ))}
      </select>
      <div className="compose-row">
        <button type="button" className="btn-primary" disabled={!chosen || busy} onClick={() => chosen && onPick(chosen)}>
          {actionLabel}
        </button>
        <button type="button" className="btn-secondary" disabled={busy} onClick={() => fileRef.current?.click()}>
          Dosya yükle
        </button>
        <input
          ref={fileRef}
          type="file"
          accept={accept}
          hidden
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) onUpload(file);
          }}
        />
      </div>
    </div>
  );
}
```

`frontend/src/compose/ObjectListPanel.tsx`:

```tsx
import type { SceneObject } from "./types";

interface Props {
  objects: SceneObject[];
  selectedId: string | null;
  errors: Record<string, string>;
  onSelect: (id: string) => void;
  onToggleVisible: (id: string, visible: boolean) => void;
  onDuplicate: (id: string) => void;
  onRemove: (id: string) => void;
}

export function ObjectListPanel({ objects, selectedId, errors, onSelect, onToggleVisible, onDuplicate, onRemove }: Props) {
  return (
    <ul className="compose-object-list">
      {objects.map((o) => (
        <li key={o.id} className={o.id === selectedId ? "selected" : ""} onClick={() => onSelect(o.id)}>
          <button
            type="button"
            className="compose-icon-btn"
            title={o.visible ? "Gizle" : "Göster"}
            onClick={(e) => {
              e.stopPropagation();
              onToggleVisible(o.id, !o.visible);
            }}
          >
            {o.visible ? "👁" : "—"}
          </button>
          <span className="compose-object-name">{o.name}</span>
          <span className="compose-badge">{o.kind === "splat" ? "splat" : "mesh"}</span>
          {o.role === "base" && <span className="compose-badge base">base</span>}
          {errors[o.id] && (
            <span className="compose-badge error" title={errors[o.id]}>
              hata
            </span>
          )}
          {o.role !== "base" && o.id === selectedId && (
            <span className="compose-object-actions">
              <button type="button" className="compose-icon-btn" title="Kopyala (Ctrl+D)" onClick={(e) => { e.stopPropagation(); onDuplicate(o.id); }}>
                ⧉
              </button>
              <button type="button" className="compose-icon-btn" title="Sil (Del)" onClick={(e) => { e.stopPropagation(); onRemove(o.id); }}>
                ✕
              </button>
            </span>
          )}
        </li>
      ))}
    </ul>
  );
}
```

`frontend/src/compose/InspectorPanel.tsx`:

```tsx
import { useEffect, useState } from "react";
import { Euler, Quaternion } from "three";
import { hexToTint, tintToHex } from "./colorMath";
import { DEFAULT_COLOR, type ColorAdjust, type SceneObject, type Transform } from "./types";

interface Props {
  object: SceneObject;
  cropEditing: boolean;
  onRename: (name: string) => void;
  onTransform: (t: Transform) => void;
  onToggleCrop: (enabled: boolean) => void;
  onCropEditing: (editing: boolean) => void;
  onColor: (c: ColorAdjust | null) => void;
  onSnap: () => void;
}

const RAD = Math.PI / 180;

function NumberField({ value, step = 0.01, disabled, onCommit }: { value: number; step?: number; disabled?: boolean; onCommit: (v: number) => void }) {
  const [text, setText] = useState(value.toFixed(3));
  useEffect(() => setText(value.toFixed(3)), [value]);
  const commit = () => {
    const v = parseFloat(text);
    if (Number.isFinite(v) && v !== value) onCommit(v);
    else setText(value.toFixed(3));
  };
  return (
    <input
      type="number"
      step={step}
      value={text}
      disabled={disabled}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === "Enter" && commit()}
    />
  );
}

export function InspectorPanel({ object, cropEditing, onRename, onTransform, onToggleCrop, onCropEditing, onColor, onSnap }: Props) {
  const [name, setName] = useState(object.name);
  useEffect(() => setName(object.name), [object.name]);
  const t = object.transform;
  const locked = object.role === "base";
  const euler = new Euler().setFromQuaternion(new Quaternion(...t.quaternion));
  const deg = [euler.x / RAD, euler.y / RAD, euler.z / RAD];
  const color = object.color ?? DEFAULT_COLOR;

  const setPos = (i: number, v: number) => {
    const position = [...t.position] as Transform["position"];
    position[i] = v;
    onTransform({ ...t, position });
  };
  const setRot = (i: number, v: number) => {
    const d = [...deg];
    d[i] = v;
    const q = new Quaternion().setFromEuler(new Euler(d[0] * RAD, d[1] * RAD, d[2] * RAD));
    onTransform({ ...t, quaternion: [q.x, q.y, q.z, q.w] });
  };
  const setColor = (patch: Partial<ColorAdjust>) => onColor({ ...color, ...patch });

  return (
    <div className="compose-inspector">
      <label className="compose-field">
        <span>Ad</span>
        <input value={name} onChange={(e) => setName(e.target.value)} onBlur={() => name.trim() && name !== object.name && onRename(name.trim())} />
      </label>

      <h4>Transform {locked && <small>(base kilitli)</small>}</h4>
      <div className="compose-vec"><span>Konum</span>{t.position.map((v, i) => <NumberField key={i} value={v} disabled={locked} onCommit={(n) => setPos(i, n)} />)}</div>
      <div className="compose-vec"><span>Dönüş °</span>{deg.map((v, i) => <NumberField key={i} value={v} step={1} disabled={locked} onCommit={(n) => setRot(i, n)} />)}</div>
      <div className="compose-vec"><span>Ölçek</span><NumberField value={t.scale} disabled={locked} onCommit={(n) => n > 0 && onTransform({ ...t, scale: n })} /></div>
      {!locked && (
        <button type="button" className="btn-secondary" onClick={onSnap}>
          Zemine oturt
        </button>
      )}

      {object.kind === "splat" && (
        <>
          <h4>Crop</h4>
          <label className="compose-check">
            <input type="checkbox" checked={!!object.crop} onChange={(e) => onToggleCrop(e.target.checked)} /> Crop kutusu
          </label>
          {object.crop && (
            <label className="compose-check">
              <input type="checkbox" checked={cropEditing} onChange={(e) => onCropEditing(e.target.checked)} /> Kutuyu gizmo ile düzenle
            </label>
          )}

          <h4>Renk</h4>
          <label className="compose-field">
            <span>Pozlama {color.exposure.toFixed(2)}</span>
            <input type="range" min={-3} max={3} step={0.05} value={color.exposure} onChange={(e) => setColor({ exposure: parseFloat(e.target.value) })} />
          </label>
          <label className="compose-field">
            <span>Doygunluk {color.saturation.toFixed(2)}</span>
            <input type="range" min={0} max={2} step={0.05} value={color.saturation} onChange={(e) => setColor({ saturation: parseFloat(e.target.value) })} />
          </label>
          <label className="compose-field">
            <span>Ton</span>
            <input type="color" value={tintToHex(color.tint)} onChange={(e) => setColor({ tint: hexToTint(e.target.value) })} />
          </label>
          <button type="button" className="btn-secondary" disabled={!object.color} onClick={() => onColor(null)}>
            Rengi sıfırla
          </button>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Page** — `frontend/src/compose/ComposePage.tsx`:

```tsx
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { getJobStatus, type Job } from "../api";
import { AssetPicker } from "./AssetPicker";
import { ComposeViewport } from "./ComposeViewport";
import {
  assetFileUrl, backendUrl, createScene, exportScene, getScene, listAssets, listScenes, saveScene, uploadAsset,
} from "./composeApi";
import { InspectorPanel } from "./InspectorPanel";
import { ObjectListPanel } from "./ObjectListPanel";
import { ObjectRegistry } from "./registry";
import { composeReducer, initialComposeState, newObjectId } from "./sceneDoc";
import { snapToGround } from "./snap";
import type { Asset, GizmoMode, SceneObject, SceneSummary } from "./types";
import "./compose.css";

type Status = { kind: "info" | "error"; text: string } | null;

function message(e: unknown): string {
  const text = e instanceof Error ? e.message : String(e);
  const m = text.match(/"detail":\s*("(?:[^"\\]|\\.)*"|\[.*\])/);
  if (!m) return text;
  try {
    const detail = JSON.parse(m[1]);
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  } catch {
    return text;
  }
}

export function ComposePage({ active }: { active: boolean }) {
  const [state, dispatch] = useReducer(composeReducer, initialComposeState);
  const { doc, selectedId, dirty } = state;
  const [scenes, setScenes] = useState<SceneSummary[]>([]);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [status, setStatus] = useState<Status>(null);
  const [busy, setBusy] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [gizmoMode, setGizmoMode] = useState<GizmoMode>("translate");
  const [cropEditing, setCropEditing] = useState(false);
  const [newName, setNewName] = useState("");
  const [exportJob, setExportJob] = useState<Job | null>(null);
  const registry = useMemo(() => new ObjectRegistry(), []);
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const selected = doc?.objects.find((o) => o.id === selectedId) ?? null;

  const refresh = useCallback(async () => {
    try {
      const [s, a] = await Promise.all([listScenes(), listAssets()]);
      setScenes(s);
      setAssets(a);
    } catch (e) {
      setStatus({ kind: "error", text: `Backend'e ulaşılamadı: ${message(e)}` });
    }
  }, []);

  useEffect(() => {
    if (active) void refresh();
  }, [active, refresh]);

  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  useEffect(() => {
    if (!selected || !selected.crop) setCropEditing(false);
  }, [selected]);

  const run = async (label: string, fn: () => Promise<void>) => {
    setBusy(true);
    setStatus({ kind: "info", text: label });
    try {
      await fn();
    } catch (e) {
      setStatus({ kind: "error", text: message(e) });
    } finally {
      setBusy(false);
    }
  };

  const openScene = (id: string) =>
    run("Sahne açılıyor…", async () => {
      const loaded = await getScene(id);
      setErrors({});
      setExportJob(null);
      dispatch({ type: "load", doc: loaded });
      setStatus(null);
    });

  const createNew = (base: Asset) =>
    run("Sahne oluşturuluyor…", async () => {
      const created = await createScene(newName.trim() || base.name, base.id);
      setErrors({});
      setExportJob(null);
      dispatch({ type: "load", doc: created });
      setNewName("");
      setStatus(null);
      void refresh();
    });

  const closeScene = () => {
    if (dirty && !window.confirm("Kaydedilmemiş değişiklikler kaybolacak. Devam edilsin mi?")) return;
    dispatch({ type: "close" });
    void refresh();
  };

  const addAsset = (asset: Asset) => {
    if (!doc) return;
    const target = controlsRef.current?.target;
    const object: SceneObject = {
      id: newObjectId(),
      kind: asset.kind,
      asset: asset.id,
      name: asset.name,
      role: "object",
      visible: true,
      transform: {
        position: target ? [target.x, target.y, target.z] : [0, 0, 0],
        // glTF is +Y up; flip meshes into COLMAP-style (−Y up) scenes.
        quaternion: asset.kind === "mesh" && doc.viewUp === "-y" ? [1, 0, 0, 0] : [0, 0, 0, 1],
        scale: 1,
      },
    };
    dispatch({ type: "add", object });
  };

  const upload = (file: File, thenAdd: boolean) =>
    run(`${file.name} yükleniyor…`, async () => {
      const asset = await uploadAsset(file);
      setAssets(await listAssets());
      if (thenAdd) addAsset(asset);
      setStatus({ kind: "info", text: `${asset.name} yüklendi` });
    });

  const save = () =>
    run("Kaydediliyor…", async () => {
      if (!doc) return;
      await saveScene(doc);
      dispatch({ type: "markSaved" });
      setStatus({ kind: "info", text: "Kaydedildi" });
    });

  const startExport = () =>
    run("Export başlatılıyor…", async () => {
      if (!doc) return;
      if (dirty) {
        await saveScene(doc);
        dispatch({ type: "markSaved" });
      }
      const { job_id } = await exportScene(doc.id);
      setExportJob(await getJobStatus(job_id));
      setStatus(null);
    });

  useEffect(() => {
    if (!exportJob || exportJob.status === "completed" || exportJob.status === "failed") return;
    const timer = window.setTimeout(async () => {
      try {
        setExportJob(await getJobStatus(exportJob.id));
      } catch (e) {
        setStatus({ kind: "error", text: message(e) });
      }
    }, 1000);
    return () => window.clearTimeout(timer);
  }, [exportJob]);

  const toggleCrop = (enabled: boolean) => {
    if (!selected) return;
    if (!enabled) {
      dispatch({ type: "setCrop", id: selected.id, crop: null });
      return;
    }
    const bounds = registry.get(selected.id)?.localBounds;
    const center = bounds ? bounds.getCenter(bounds.min.clone()) : null;
    const size = bounds ? bounds.getSize(bounds.min.clone()) : null;
    dispatch({
      type: "setCrop",
      id: selected.id,
      crop: {
        center: center ? [center.x, center.y, center.z] : [0, 0, 0],
        halfSize: size ? [Math.max(size.x / 2, 1e-3), Math.max(size.y / 2, 1e-3), Math.max(size.z / 2, 1e-3)] : [1, 1, 1],
        quaternion: [0, 0, 0, 1],
      },
    });
    setCropEditing(true);
  };

  const snap = () => {
    if (!doc || !selected) return;
    const position = snapToGround(doc, registry, selected.id);
    if (!position) {
      setStatus({ kind: "error", text: "Altında yüzey bulunamadı (obje yüklenmemiş olabilir)" });
      return;
    }
    dispatch({ type: "setTransform", id: selected.id, transform: { ...selected.transform, position } });
    setStatus(null);
  };

  useEffect(() => {
    if (!active || !doc) return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.ctrlKey && e.key.toLowerCase() === "d") {
        e.preventDefault();
        if (selectedId) dispatch({ type: "duplicate", id: selectedId, newId: newObjectId() });
        return;
      }
      if (e.ctrlKey && e.key.toLowerCase() === "s") {
        e.preventDefault();
        void save();
        return;
      }
      switch (e.key.toLowerCase()) {
        case "w": setGizmoMode("translate"); break;
        case "e": setGizmoMode("rotate"); break;
        case "r": setGizmoMode("scale"); break;
        case "escape": dispatch({ type: "select", id: null }); setCropEditing(false); break;
        case "delete": if (selectedId) dispatch({ type: "remove", id: selectedId }); break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, doc, selectedId]);

  const onError = useCallback((id: string, msg: string) => setErrors((prev) => ({ ...prev, [id]: msg })), []);
  const onSelect = useCallback((id: string | null) => dispatch({ type: "select", id }), []);
  const onControls = useCallback((c: OrbitControlsImpl | null) => {
    controlsRef.current = c;
  }, []);

  const statusBar = status && <div className={`compose-status ${status.kind}`}>{status.text}</div>;

  if (!doc) {
    return (
      <div className="compose-home">
        <section>
          <h2>Sahneler</h2>
          {scenes.length === 0 && <p className="muted">Henüz kayıtlı sahne yok.</p>}
          <ul className="compose-scene-list">
            {scenes.map((s) => (
              <li key={s.id}>
                <span>{s.name}</span>
                <small>{s.object_count} obje · {new Date(s.updated_ts * 1000).toLocaleString()}</small>
                <button type="button" className="btn-primary" disabled={busy} onClick={() => openScene(s.id)}>
                  Aç
                </button>
              </li>
            ))}
          </ul>
        </section>
        <section>
          <h2>Yeni sahne</h2>
          <p className="muted">Base sahne olarak bir splat seç (bizim pipeline sonucu ya da Spirula'dan indirilen .ply).</p>
          <label className="compose-field">
            <span>Ad</span>
            <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="ör. bahce_heykel" />
          </label>
          <AssetPicker assets={assets} kinds={["splat"]} busy={busy} actionLabel="Oluştur" onPick={createNew} onUpload={(f) => upload(f, false)} />
        </section>
        {statusBar}
      </div>
    );
  }

  return (
    <div className="compose-editor">
      <div className="compose-toolbar">
        <button type="button" className="btn-secondary" onClick={closeScene}>← Sahneler</button>
        <strong className="compose-title">{doc.name}{dirty ? " •" : ""}</strong>
        <div className="compose-segment">
          {(["translate", "rotate", "scale"] as GizmoMode[]).map((m, i) => (
            <button key={m} type="button" className={gizmoMode === m ? "active" : ""} onClick={() => setGizmoMode(m)}>
              {["Taşı (W)", "Döndür (E)", "Ölçekle (R)"][i]}
            </button>
          ))}
        </div>
        <label className="compose-inline">
          Yukarı
          <select value={doc.viewUp} onChange={(e) => dispatch({ type: "setViewUp", viewUp: e.target.value as "y" | "-y" })}>
            <option value="y">+Y</option>
            <option value="-y">−Y (COLMAP)</option>
          </select>
        </label>
        <span className="compose-spacer" />
        <button type="button" className="btn-secondary" disabled={busy || !dirty} onClick={save}>Kaydet</button>
        <button type="button" className="btn-primary" disabled={busy || exportJob?.status === "running" || exportJob?.status === "queued"} onClick={startExport}>
          Export
        </button>
        {exportJob && (
          <span className="compose-export">
            {exportJob.status === "completed" && exportJob.download_url ? (
              <a className="btn-secondary" href={backendUrl(exportJob.download_url)} target="_blank" rel="noreferrer">.zip indir</a>
            ) : exportJob.status === "failed" ? (
              <span className="err" title={exportJob.error ?? ""}>Export başarısız</span>
            ) : (
              <span>{Math.round(exportJob.overall_progress * 100)}% · {exportJob.phase?.message ?? ""}</span>
            )}
          </span>
        )}
      </div>
      <div className="compose-body">
        <aside className="compose-left">
          <h3>Objeler</h3>
          <ObjectListPanel
            objects={doc.objects}
            selectedId={selectedId}
            errors={errors}
            onSelect={(id) => dispatch({ type: "select", id })}
            onToggleVisible={(id, visible) => dispatch({ type: "setVisible", id, visible })}
            onDuplicate={(id) => dispatch({ type: "duplicate", id, newId: newObjectId() })}
            onRemove={(id) => dispatch({ type: "remove", id })}
          />
          <h3>Obje ekle</h3>
          <AssetPicker assets={assets} busy={busy} actionLabel="Sahneye ekle" onPick={addAsset} onUpload={(f) => upload(f, true)} />
        </aside>
        <main className="compose-viewport">
          <ComposeViewport
            doc={doc}
            active={active}
            selectedId={selectedId}
            gizmoMode={gizmoMode}
            cropEditing={cropEditing}
            registry={registry}
            assetUrl={assetFileUrl}
            onSelect={onSelect}
            onTransformCommit={(id, transform) => dispatch({ type: "setTransform", id, transform })}
            onCropCommit={(id, crop) => dispatch({ type: "setCrop", id, crop })}
            onError={onError}
            onControls={onControls}
          />
        </main>
        <aside className="compose-right">
          {selected ? (
            <InspectorPanel
              key={selected.id}
              object={selected}
              cropEditing={cropEditing}
              onRename={(name) => dispatch({ type: "rename", id: selected.id, name })}
              onTransform={(transform) => dispatch({ type: "setTransform", id: selected.id, transform })}
              onToggleCrop={toggleCrop}
              onCropEditing={setCropEditing}
              onColor={(color) => dispatch({ type: "setColor", id: selected.id, color })}
              onSnap={snap}
            />
          ) : (
            <p className="muted">Düzenlemek için bir obje seç.</p>
          )}
        </aside>
      </div>
      {statusBar}
    </div>
  );
}
```

- [ ] **Step 4: CSS** — `frontend/src/compose/compose.css`:

```css
.compose-host { width: 100%; height: 100%; }
.compose-home { padding: 20px 24px; max-width: 960px; margin: 0 auto; height: 100%; overflow: auto; display: grid; gap: 24px; align-content: start; }
.compose-home h2 { margin: 0 0 8px; font-size: 16px; }
.compose-scene-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
.compose-scene-list li { display: grid; grid-template-columns: 1fr auto auto; gap: 12px; align-items: center; padding: 8px 10px; background: var(--bg-2); border: 1px solid var(--border); border-radius: 6px; }
.compose-scene-list small, .compose-home .muted, .compose-right .muted { color: var(--text-muted); }
.compose-editor { display: grid; grid-template-rows: auto 1fr auto; height: 100%; }
.compose-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 8px 12px; background: var(--bg-1); border-bottom: 1px solid var(--border); }
.compose-title { font-size: 14px; }
.compose-spacer { flex: 1; }
.compose-segment { display: inline-flex; border: 1px solid var(--border-1); border-radius: 6px; overflow: hidden; }
.compose-segment button { background: var(--bg-2); color: var(--text); border: 0; padding: 5px 10px; cursor: pointer; font-size: 12px; }
.compose-segment button.active { background: var(--accent-bg); }
.compose-inline { display: inline-flex; gap: 6px; align-items: center; font-size: 12px; color: var(--text-muted); }
.compose-export { font-size: 12px; }
.compose-export .err { color: var(--err); }
.compose-body { display: grid; grid-template-columns: 260px 1fr 280px; min-height: 0; }
.compose-left, .compose-right { background: var(--bg-1); padding: 10px 12px; overflow: auto; min-height: 0; }
.compose-left { border-right: 1px solid var(--border); }
.compose-right { border-left: 1px solid var(--border); }
.compose-left h3 { margin: 12px 0 6px; font-size: 12px; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.04em; }
.compose-viewport { position: relative; min-height: 0; min-width: 0; }
.compose-object-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
.compose-object-list li { display: flex; align-items: center; gap: 6px; padding: 4px 6px; border-radius: 4px; cursor: pointer; font-size: 13px; }
.compose-object-list li:hover { background: var(--bg-2); }
.compose-object-list li.selected { background: var(--bg-3); outline: 1px solid var(--accent); }
.compose-object-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.compose-object-actions { display: inline-flex; gap: 2px; }
.compose-icon-btn { background: transparent; border: 0; color: var(--text); cursor: pointer; padding: 0 4px; font-size: 13px; }
.compose-badge { font-size: 10px; padding: 1px 5px; border-radius: 3px; background: var(--bg-3); color: var(--text-muted); }
.compose-badge.base { color: var(--accent); }
.compose-badge.error { color: var(--err); }
.compose-asset-picker { display: grid; gap: 6px; }
.compose-asset-picker select { width: 100%; }
.compose-row { display: flex; gap: 6px; flex-wrap: wrap; }
.compose-inspector { display: grid; gap: 8px; font-size: 13px; }
.compose-inspector h4 { margin: 8px 0 0; font-size: 12px; color: var(--text-muted); text-transform: uppercase; }
.compose-field { display: grid; gap: 4px; }
.compose-field span { font-size: 12px; color: var(--text-muted); }
.compose-vec { display: grid; grid-template-columns: 56px repeat(3, 1fr); gap: 4px; align-items: center; }
.compose-vec > span { font-size: 12px; color: var(--text-muted); }
.compose-vec input { width: 100%; min-width: 0; }
.compose-check { display: flex; align-items: center; gap: 6px; }
.compose-status { padding: 6px 12px; font-size: 12px; background: var(--bg-1); border-top: 1px solid var(--border); }
.compose-status.error { color: var(--err); }
@media (max-width: 900px) {
  .compose-body { grid-template-columns: 1fr; grid-template-rows: auto 50vh auto; }
  .compose-left, .compose-right { border: 0; }
}
```

- [ ] **Step 5: App integration** — in `frontend/src/App.tsx`:

```tsx
import { ComposePage } from "./compose/ComposePage";
type Tab = "notebook" | "submit" | "jobs" | "viewer" | "analytics" | "eval" | "interactive" | "compose";
...
  // Sahne Editörü: ilk açılışta mount edilir, sonra sekme değişse de state kaybolmasın diye açık kalır.
  const [composeMounted, setComposeMounted] = useState(false);
  useEffect(() => {
    if (tab === "compose") setComposeMounted(true);
  }, [tab]);
...
          <button
            className={`tab-btn ${tab === "compose" ? "active" : ""}`}
            onClick={() => setTab("compose")}
          >
            Sahne Editörü
          </button>
...
        {composeMounted && (
          <div className="compose-host" style={{ display: tab === "compose" ? "block" : "none" }}>
            <ComposePage active={tab === "compose"} />
          </div>
        )}
```

(Place the nav button after "Interactive"; place the host block after the interactive block inside `<main className="app-content">`. `compose-host` has `height: 100%` from compose.css; the page's CSS is imported by ComposePage.)

- [ ] **Step 6: JobsList** — in `frontend/src/components/JobsList.tsx`:

```tsx
import { getApiBase, withTokenParam } from "../connection";
...
          {job.status === "completed" && (
            <div className="job-actions">
              {!job.scene.startsWith("compose-") && (
                <button className="btn-primary" onClick={onView}>
                  Viewer'da aç
                </button>
              )}
              <a
                className="btn-secondary"
                href={job.download_url ? withTokenParam(`${getApiBase()}${job.download_url}`) : downloadUrl(job.id)}
                target="_blank"
                rel="noreferrer"
              >
                .zip indir
              </a>
            </div>
          )}
```

- [ ] **Step 7: Run** — `cd frontend && npx vitest run && npx tsc --noEmit -p tsconfig.json` → PASS / no errors.
- [ ] **Step 8: Commit** — `git commit -m "feat(compose): add Sahne Editörü tab with panels and export flow"`

---

### Task 12: Manual verification + docs

- [ ] **Step 1:** Start backend (`scripts\start-backend.bat`) and frontend (`npm run tauri dev` or `npm run dev`), open "Sahne Editörü".
- [ ] **Step 2:** Create scene from a pipeline result (`scene__<name>`); upload a second `.ply`, add it; move/rotate/scale with W/E/R; verify uniform scale.
- [ ] **Step 3:** Enable crop, edit box with gizmo, confirm hidden splats outside the box; snap to ground; change exposure/saturation/tint and see live change.
- [ ] **Step 4:** Add a `.glb`, place it; Save; reload page; reopen scene → same layout.
- [ ] **Step 5:** Export; watch progress; download zip; open `merged.ply` in the Viewer tab or another viewer and compare with the editor.
- [ ] **Step 6:** README: add a short "Sahne Editörü" section (what it does, where files live, `.glb` only).
- [ ] **Step 7:** Run full suites: `E:/anaconda3/envs/gs4d/python.exe -m pytest tests/compose tests/notebooks -q` and `cd frontend && npx vitest run`. Commit + push.

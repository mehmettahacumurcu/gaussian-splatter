# Scene Composer Design

**Date:** 2026-09-29
**Status:** Approved for implementation
**Branch:** `feature/scene-composer`

## 1. Summary

A Postshot-style "import object into scene" feature. The user opens a base
Gaussian-splat scene, places additional objects into it with a transform gizmo,
and exports the result:

- **Splat objects** (`.ply`, standard INRIA 3DGS layout) from our pipeline
  (`data/<scene>/output/ply`) or from the Spirula pipeline (`splat.ply`
  downloaded from Drive) or any other trainer.
- **Mesh objects** (`.glb` only in v1).

Editing happens live in the browser (Spark + three.js). Export bakes every
visible splat object into one merged `merged.ply` on the backend (numpy), and
writes each mesh as a `.glb` whose root carries its placement transform, so
`merged.ply` and the meshes line up when imported together elsewhere
(e.g. Unreal).

The feature works on final `.ply` files, so it is independent of which
pipeline produced them.

## 2. Scope

In v1:

- New "Sahne Editörü" tab in the frontend.
- Assets: upload `.ply` / `.glb` from disk, plus pipeline results listed from
  `data/*/output/ply`.
- Scene documents saved on the backend, reopenable and editable.
- Per object: move / rotate / uniform scale gizmo, list, delete, duplicate,
  show/hide.
- Crop box per splat object.
- Snap-to-ground (drop object onto the surface below it).
- Colour matching per splat object: exposure, RGB tint, saturation.
- Export job: `merged.ply` + `meshes/*.glb` + `scene.json`, downloadable zip.

Out of scope (v1): undo/redo, OBJ/FBX import, non-uniform scale, mesh→splat
conversion, mesh lighting/shadow matching, 4D sequences, running inside the
Colab notebooks.

## 3. Architecture

Approach A: **preview in the browser, bake on the backend.** Every edit is
applied instantly in Spark; the same parameters are applied deterministically
by numpy at export. A shared golden fixture keeps the two conventions from
drifting.

### 3.1 Frontend (`frontend/src/compose/`)

| Unit | Purpose |
|---|---|
| `sceneDoc.ts` | `SceneDoc` types, reducer (add / remove / duplicate / update transform, crop, colour, visibility; base lock), pure helpers |
| `colorMath.ts` | Colour matrix `M = S·diag(g)` (same formula as backend) |
| `composeApi.ts` | Client for `/compose/*` |
| `ComposePage.tsx` | Tab root: scene picker/creation, toolbar, layout, save/export flow |
| `ComposeViewport.tsx` | R3F `Canvas`, `OrbitControls`, `SparkRenderer`, selection, `TransformControls` |
| `SplatObject.tsx` | `SplatMesh` + crop `SplatEdit` + recolor/saturation modifier |
| `MeshObject.tsx` | GLTF load |
| `ObjectListPanel.tsx`, `InspectorPanel.tsx`, `AssetPicker.tsx` | Side panels |
| `snap.ts` | Snap-to-ground raycast helper |

### 3.2 Backend (`backend/compose/`)

| Unit | Purpose |
|---|---|
| `models.py` | Pydantic `SceneDoc` v1 + asset models, validation |
| `plyio.py` | Read / write standard 3DGS PLY (plyfile, no torch). Unknown extra fields (e.g. `nx,ny,nz`) ignored on read |
| `sh_rotation.py` | Numerical per-band SH rotation matrices |
| `bake.py` | Pure numpy: crop, transform, SH rotate, colour, merge |
| `glb.py` | Wrap a `.glb` scene in a new root node with a matrix (pure Python) |
| `store.py` | Disk layout for assets / scenes / exports, safe ids |
| `routes.py` | `build_compose_router(root, manager_getter)` → `APIRouter(prefix="/compose")` |

### 3.3 Disk layout

```
data/compose/assets/<asset_id>.{ply,glb}
data/compose/assets/<asset_id>.json          # {id, kind, name, size_bytes, created_ts}
data/compose/scenes/<scene_id>.json          # SceneDoc
data/compose/exports/<scene_id>/{merged.ply, meshes/<object_id>.glb, scene.json}
data/compose/exports/<scene_id>.zip
```

Pipeline results are exposed as virtual assets with id `scene__<name>`,
resolving to the last `.ply` in `data/<name>/output/ply/`. Asset ids are
generated server-side (`a_<hex12>`) and validated against `^[A-Za-z0-9_.-]+$`;
user file names never become paths.

## 4. Scene document (v1)

```json
{
  "version": 1,
  "id": "s_1a2b3c4d5e6f",
  "name": "bahce_heykel",
  "viewUp": "y",
  "objects": [
    { "id": "o1", "kind": "splat", "asset": "scene__garden", "name": "garden",
      "role": "base", "visible": true,
      "transform": { "position": [0,0,0], "quaternion": [0,0,0,1], "scale": 1 } },
    { "id": "o2", "kind": "splat", "asset": "a_9f...", "name": "statue",
      "role": "object", "visible": true,
      "transform": { "position": [1.2,0,-0.5], "quaternion": [0,0.38,0,0.92], "scale": 0.8 },
      "crop": { "center": [0,0.4,0], "halfSize": [0.5,0.6,0.5], "quaternion": [0,0,0,1] },
      "color": { "exposure": 0.2, "tint": [1,0.97,0.93], "saturation": 1.1 } },
    { "id": "o3", "kind": "mesh", "asset": "a_4c...", "name": "chair",
      "role": "object", "visible": true,
      "transform": { "position": [0,0,1], "quaternion": [0,0,0,1], "scale": 1 } }
  ]
}
```

Rules:

- **Frame:** editor world = raw coordinate frame of the base splat. Exactly one
  object has `role: "base"`; it must be a splat and its transform must be
  identity. `merged.ply` is therefore aligned with the original scene.
- **`viewUp`:** `"y"` or `"-y"` (COLMAP-style). View-only: sets camera up and
  the snap direction; never baked.
- **Quaternions** in JSON are three.js order `[x, y, z, w]`. PLY stores
  `rot_0..rot_3 = (w, x, y, z)`. Conversion lives in one tested helper per side.
- **Scale** is a single positive number (uniform). Non-uniform scale of a
  rotated Gaussian is not a Gaussian with axis-aligned log scales, so it is
  not offered.
- **Crop** (splat only, optional) is defined in the object's *local* frame
  (moves with the object). A Gaussian is kept iff its centre satisfies
  `|R_cᵀ(p − c)| ≤ halfSize` per axis. Hard test, no soft edge.
- **Colour** (splat only, optional): `exposure` in stops (−3…3), `tint` RGB
  (0…2 each), `saturation` (0…2).

Validation (pydantic, at request time → HTTP 400/422, never inside a job):
scale > 0, halfSize > 0, quaternion norm within 1e-3 of 1 (then normalized),
unique object ids, exactly one base, base identity, base is splat, crop/colour
only on splats, mesh assets are `.glb`, splat assets are `.ply`.

## 5. Bake math (`bake.py`)

For each visible splat object with transform `(t, q_o → R, s)`:

1. **Crop** in local frame (before transform).
2. **Position:** `p' = s·R·p + t`.
3. **Rotation:** `q' = q_o ⊗ q` (Hamilton product, `wxyz`). `q_o` is unit, so the
   stored norm of `q` is preserved (renderers normalise).
4. **Scale:** `log_scale' = log_scale + ln(s)`. Opacity unchanged.
5. **SH rotation:** DC unchanged. For each band `l = 1..deg`, a matrix `D_l`
   (size `(2l+1)²`) is solved numerically: sample `M ≥ 200` unit directions
   `d`, evaluate the real SH basis `Y_l` with the exact 3DGS/gsplat constants
   and sign convention, and solve `Y_l(d) · D_lᵀ = Y_l(R⁻¹ d)` by least
   squares. Coefficients: `rest'_l = D_l · rest_l` per colour channel. The
   property guaranteed by tests: `SH'(R d) = SH(d)` for all `d`.
   SH degree is inferred from the number of `f_rest_*` fields
   (0, 9, 24, 45 → degree 0, 1, 2, 3); other counts are rejected.
   The `f_rest` layout is channel-major (`f_rest_{c·K + k}`), as written by
   `export/to_splat.write_ply`.
6. **Colour:** `g = 2^exposure · tint`, `L = 1·wᵀ` with Rec.709 luma
   `w = (0.2126, 0.7152, 0.0722)`, `S = L + sat·(I − L)`, `M = S·diag(g)`.
   `dc' = (M·(0.5 + C0·dc) − 0.5)/C0` and `rest'_k = M·rest_k`. Because the
   rendered colour is `0.5 + C0·dc + Σ rest·Y`, this equals `M · colour` for
   every view direction.
7. **Merge:** concatenate objects in document order; pad lower SH degrees with
   zeros to the maximum degree; write with `plyio.write_ply` (same layout as
   `export/to_splat.write_ply`, `nx,ny,nz = 0`).

Meshes: `glb.wrap_with_transform(src, dst, matrix4)` parses the GLB JSON chunk,
appends a node with `matrix` (column-major 4×4 of `T·R·s`) whose children are
the default scene's root nodes, and makes it the scene's only root. Binary
chunk untouched.

## 6. API (`/compose`)

| Method | Path | Behaviour |
|---|---|---|
| GET | `/compose/assets` | Uploaded + pipeline assets |
| POST | `/compose/assets` | Multipart upload; checks extension + content (PLY header has `x,y,z,f_dc_*,opacity,scale_*,rot_*`; GLB magic `glTF` v2). Returns asset |
| GET | `/compose/assets/{id}/file` | `FileResponse` (Spark / GLTFLoader read it via `withTokenParam`) |
| GET | `/compose/scenes` | List `{id, name, updated_ts, object_count}` |
| POST | `/compose/scenes` | Create from `{name, base_asset}` → SceneDoc |
| GET | `/compose/scenes/{id}` | SceneDoc |
| PUT | `/compose/scenes/{id}` | Save (validated) |
| POST | `/compose/scenes/{id}/export` | Validates, checks every asset exists (400 listing missing), submits a JobManager job, returns `{job_id}` |
| GET | `/compose/exports/{id}/download` | Streams the on-disk zip |

Changes to existing code:

- `api.py`: `include_router(build_compose_router(...))`; CORS `allow_methods`
  gains `PUT`.
- `job_manager.py`: add phases `compose_load`, `compose_bake`, `compose_write`
  to `PHASE_WEIGHTS` as a separate ordering group, and make unknown phases
  report their own progress instead of jumping to 100%.
  `_mark_completed` uses `result["download_url"]` when present.
- `JobsList.tsx`: use `job.download_url` when set; hide "Viewer'da aç" for
  `compose:` jobs.

Export job writes into `exports/<id>.tmp/`, then atomically replaces
`exports/<id>/`, then writes the zip (to a temp name, then rename). Export runs
on the shared single-worker JobManager queue (waits behind training; accepted
for v1).

## 7. Editor interactions

- **Layout:** object list (left), viewport (centre), inspector (right),
  toolbar: add asset, upload, gizmo mode (W/E/R), snap to ground, save, export.
- **Selection:** click; splats via `SplatMesh.raycast`, meshes via standard
  raycast. `Esc` deselect, `Delete` remove, `Ctrl+D` duplicate. Base cannot be
  deleted, moved or duplicated.
- **Gizmo:** drei `TransformControls`. Scale mode forces uniform scale (average
  of axes). Object moves directly during drag; reducer commit on drag end.
- **Crop:** "Crop" toggle in inspector creates a default box around the
  object's bounds and shows a wireframe; "edit box" mode points the gizmo at the
  box. Preview: `SplatEdit` child of the `SplatMesh` with a `BOX`
  `SplatEditSdf` and `invert`, `opacity 0`, `softEdge 0` (fallback if scoping
  fails: `objectModifier` dyno zeroing opacity outside the box).
- **Snap to ground:** cast from the bottom-centre of the object's (cropped)
  world bounding box along −up against other visible objects; place the bottom
  on the hit point.
- **Colour:** sliders exposure, saturation; tint colour picker. Preview applies
  the full matrix `M` in one `objectModifier` (a `DynoMat3` uniform); Spark runs
  it after SH evaluation, so it equals the bake for every view direction.
- **Save:** explicit save button (and Ctrl+S), dirty indicator. The editor stays
  mounted when switching tabs (render loop paused), so nothing is lost; closing
  the scene or the window with unsaved changes asks for confirmation.

## 8. Error handling

- Invalid upload → 400 with reason.
- Missing asset at scene load → object shows an error badge; editor keeps
  working. Export → 400 listing missing assets.
- Spark / GLTF load failure → per-object error badge.
- Export failure → job FAILED with message; no partial export left behind.

## 9. Testing

Backend (`tests/compose/`, gs4d env):

- `plyio`: round-trip; Spirula-like file with extra fields; bad files rejected.
- `sh_rotation`: `SH'(R d) = SH(d)` for random rotations, degrees 1–3;
  identity → identity; composition `D(R1 R2) = D(R1) D(R2)`.
- `bake`: identity no-op; translate / rotate / scale on synthetic Gaussians;
  quaternion order; crop keeps exactly the inside centres; exposure +1 doubles
  colour; saturation 0 → grey; mixed SH degrees padded; golden fixture.
- `glb`: wrapped file parses, root matrix correct, binary chunk unchanged.
- `routes` (TestClient on a bare FastAPI app + fake manager): upload
  validation, path safety, scene CRUD, validation 400s, export end-to-end on a
  tiny fixture, zip download.
- `job_manager`: unknown phase progress, compose phases, download_url.
- Optional `gpu` marker: render rotated splat from camera C vs original from
  `R⁻¹C` with gsplat.

Frontend (vitest): reducer, quaternion/transform helpers against the same
golden fixture JSON (single file `tests/fixtures/compose_golden.json`, imported
by both suites via relative path), colour matrix, compose API client.

Task 0 spike: confirm in Spark 0.1.10 that `SplatEdit` under a `SplatMesh`
only affects that mesh, how `recolor` combines, and that `SplatMesh.raycast`
returns hits.

## 10. Amendment (2026-09-30): arbitrary up vector, found in browser testing

Manual testing on `myroom_v2` showed the `viewUp ∈ {y, -y}` assumption is wrong
for real captures: COLMAP frames are arbitrary (this room's floor normal is
≈ (−0.94, −0.11, −0.32), i.e. the scene lies on its side), so snap-to-ground
found nothing below objects. The repo already has a tested estimator,
`backend/image_to_scene/orientation.py::estimate_world_orientation` (RANSAC
floor plane, content-asymmetry sign; ~2 s on 100k Gaussians).

Changes:

- **SceneDoc:** new optional field `up: [x,y,z] | null` (unit; norm within
  1e-3, then normalised; finite). When set it overrides `viewUp`; effective up =
  `up ?? (viewUp == "-y" ? [0,-1,0] : [0,1,0])`. Still view-only (camera up,
  snap direction, mesh insert orientation); never baked. Backward compatible.
- **Orientation endpoint:** `GET /compose/assets/{id}/orientation` →
  `{up, tilt_deg, plane_inlier_frac, above_below_ratio, measured}` (`measured` =
  a floor plane was found; false means `up` is a +Y placeholder) for splat assets
  (400 for meshes, 404 unknown), computed from means + sigmoid(opacity)
  weights, cached on disk keyed by asset id + file size + mtime.
- **Scene creation** sets `up` from the base asset's orientation (falls back to
  null on failure).
- **Editor:** "Yukarı" selector offers `Otomatik (zemin)`, `+Y`, `−Y`; snap and
  the orbit camera use the effective up vector (any direction); inserted meshes
  are rotated so their +Y matches it. Splat objects get a **"Dikleştir"**
  (straighten) action: rotate the object so its own estimated up (from its
  asset's orientation) maps onto the scene up, keeping position and scale.
- **Bug fix:** inspector number fields could re-commit a stale value when a
  field was left right after pressing Enter (the field text was reset to the
  old value until the prop update arrived), silently undoing the edit.

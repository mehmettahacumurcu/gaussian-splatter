# Automated Static Splat Notebook and Quality Pipeline Design

**Date:** 2026-07-11
**Status:** Approved in conversation
**Implementation approach:** Thin generated notebook plus a dedicated, testable runner

## Summary

The product will generate a Google Colab notebook for ordinary static-scene video or image inputs stored in Google Drive. A user will select a Drive folder, optionally compare smart frame selection with the legacy fixed-FPS baseline, select a quality profile, download the notebook, connect a Colab runtime, and use **Run all**. A successful run will write one portable Gaussian-splat result bundle to the exact sibling folder `<input_folder_name>_result`.

The notebook remains orchestration-only. Frame selection, COLMAP reconstruction, quality gates, training, post-training polish, and artifact publishing live in regular Python modules behind a versioned `RunSpec`. The existing notebooks and local `/process` workflow remain unchanged and available as legacy references.

Phone-pose capture modes, ARKit adapters, and the former B/C capture design are deferred. This design uses ordinary videos or image sets only.

## Goals

1. Generate a safe, reproducible `.ipynb` from the frontend without starting a backend training job.
2. Make the notebook truly Run-All: mount Drive, validate, clone a pinned commit, bootstrap, preprocess, train, polish, and publish without manual cell edits or a runtime restart.
3. Prevent poor or stale COLMAP reconstructions from reaching multi-hour training.
4. Make smart frame selection optional and enabled by default, with a controlled fixed-FPS baseline for quality comparisons.
5. Expose four product-quality profiles, defaulting to a profile designed for an NVIDIA L4-class 24 GB GPU.
6. Preserve the Drive input as read-only and publish successful output exactly to `<input_folder_name>_result`.
7. Produce a standard INRIA 3DGS PLY that opens in SuperSplat and a metadata bundle consumable by this project's viewer.
8. Add conservative, regression-guarded splat cleanup and Luma-like presentation metadata without baking fake sky geometry into the model.

## Non-goals

- No phone-tracking, ARKit, VIO, NeRFCapture, CamTrackAR, or other posed-capture integration.
- No web viewer inside the result folder.
- No change to the existing `video_to_world.ipynb`, `sota_verify.ipynb`, or `phase2_verify.ipynb` behavior.
- No replacement of the existing local `/process` and `Static3DSubmit` job workflow.
- No guarantee that software can reconstruct areas that are unfilmed or blurred in every source frame.
- No SPZ or `.splat` export claim in v1; the real portable artifact is standard PLY.
- No mandatory NVS holdout for product runs because withholding input views lowers keeper-run coverage.

## Why the Current Path Loses Quality

The `myroom-max` production run is the primary evidence:

- A 2:05 phone video produced 502 fixed-FPS frames.
- Exhaustive COLMAP matching completed, but motion-blurred spans split the correspondence graph into 12 sparse models.
- The chosen model registered only 285/502 frames (56.8%).
- The pipeline treated any nonempty model as success and spent roughly seven hours training it.

The audit also confirmed independent correctness defects:

1. Frame cache invalidation calls extraction without `overwrite=True`, so old PNGs can survive while a new settings marker is written.
2. Frame cache identity covers settings but not source-video identity or the selected-frame manifest.
3. COLMAP reruns reuse `colmap.db` and `sparse/`, allowing stale features, images, matches, and models to survive.
4. Sparse models are ranked by binary byte size instead of registered-image count and reconstruction quality.
5. Downstream depth alignment can pair the first physical frames with sorted registered poses rather than matching by frame filename when registration is incomplete.
6. `sequential_overlap` is included in the cache hash but is not forwarded to the COLMAP command.

Simply selecting exhaustive matching is not the solution: the failed production run already used it. The quality path must improve the input correspondence graph, retry from clean state, and reject weak reconstructions.

## Chosen Architecture

```text
Frontend notebook wizard
        |
        | POST /notebooks/static (RunSpec JSON)
        v
Notebook generator service
        |
        | returns versioned .ipynb attachment
        v
Generated Colab notebook
        |
        | scripts/notebook_static_run.py --spec run_spec.json
        v
Input discovery -> selection -> COLMAP -> gate/retry -> train -> polish
        |
        v
Transactional publish to <input_folder_name>_result
```

### Why this architecture

- The notebook is short, inspectable, and stable.
- Pipeline logic is unit-testable without parsing notebook cells.
- The same runner can be exercised locally, in CI with mocked boundaries, and in Colab.
- Values are passed as data through JSON and argument arrays, not interpolated into executable Python or shell strings.
- Legacy notebooks remain golden behavior references rather than becoming mutable templates.

## Frontend Experience

The existing submit area becomes notebook-first while keeping the local job submitter available separately.

### Top-to-bottom flow

1. **Google Drive input folder**
2. **Smart Frame Selection**
3. **Quality profile**
4. **Iterations and maximum Gaussians**
5. **Advanced settings** (collapsed)
6. **Review**
7. **Generate and download notebook**

Notebook generation stays on the Notebook page. It does not create a `JobManager` entry and does not navigate to Jobs.

### Drive folder field

The canonical value is relative to `MyDrive`, for example `captures/myroom`. The UI may accept an optional leading `MyDrive/` and remove it during normalization.

The UI displays but does not allow editing the derived output:

```text
Input:  MyDrive/captures/myroom
Output: MyDrive/captures/myroom_result
```

Validation rejects:

- empty values or the Drive root;
- `..`, backslashes, control characters, and absolute runtime paths;
- a leaf already ending in `_result`;
- ambiguous or empty path segments.

The backend repeats all path validation and independently derives the output.

### Smart Frame Selection control

This is a two-choice segmented control rather than an unlabelled boolean:

- **Smart — Recommended** (default)
- **Fixed FPS — Baseline**

When Fixed FPS is selected, an FPS field appears with a default of `4`, matching the established `video_to_world.ipynb` baseline. Smart mode does not expose blur thresholds, overlap thresholds, or candidate-analysis FPS as casual user controls.

For a fair comparison, only selection and smart gap backfill differ. COLMAP validation, training configuration, post-training polish, and publishing are identical. Both `run_manifest.json` and `quality_report.json` record the selected mode.

The fixed-FPS path still uses corrected cache behavior, clean COLMAP staging, filename-safe alignment, and reconstruction gates. It is a baseline, not an opt-out from correctness checks.

### Quality profiles

| Profile | Iterations | Max Gaussians | Selected-frame budget | Resolution policy | Intended use |
|---|---:|---:|---:|---|---|
| **Balanced / L4** (default) | 30,000 | 250,000 | 300 | 1280 px long-edge cap | Low-mid product run with roughly 18-20 GB VRAM target on an L4-class 24 GB GPU |
| **High** | 50,000 | 500,000 | 450 | Up to 1080p | Higher-detail run with more memory/time |
| **Premium** | 100,000 | 1,000,000 | 600 | Up to 1440p | Large-memory high-quality run |
| **Ultra** | 120,000 | 3,000,000 | 800 | Native resolution, bounded by source | Maximum-quality A100 80 GB-class run |

The existing `fast`, `balanced`, `high`, `premium`, `sota`, and `ultra` backend identifiers remain accepted for compatibility. The product-facing Balanced/L4 profile is a deliberately bounded notebook profile; `sota` remains an expert benchmark/PSNR-parity profile rather than a general visual-quality choice.

The notebook preflight reports detected GPU model, VRAM, disk space, and selected profile. It fails before preprocessing when a choice is clearly unsafe. Balanced/L4 is designed with headroom rather than assuming all nominal 24 GB is available.

Iterations and max Gaussians are nullable overrides. `null` means the profile default. The UI shows effective values and provides **Reset to profile default**.

### Advanced settings

The v1 allowlist contains only these stable static-run fields:

- `run_eval`;
- `foundation`;
- `resolution_long_edge_cap`;
- `lambda_ssim`, `lambda_lpips`, and `lambda_depth`;
- `density_start_iter`, `density_end_iter`, and `density_interval`;
- `densify_grad_threshold`;
- `prune_min_opacity` and `prune_max_scale`;
- `opacity_reset_interval`;
- `sh_degree`;
- `multires_schedule`.

COLMAP axis conventions, cache keys, acceptance thresholds, and smart-selection internals are versioned pipeline policy, not ordinary frontend knobs.

### Frontend component boundaries

- `NotebookGeneratorPanel`: reducer/state machine and artifact lifecycle.
- `DriveFolderSection`: path input, validation, and derived output preview.
- `FrameSelectionSection`: Smart versus Fixed-FPS control.
- `QualitySection`: four product profile cards and two primary overrides.
- `NotebookAdvancedConfig`: static-only advanced allowlist.
- `NotebookReview`: effective RunSpec, warnings, paths, and generation action.
- `NotebookDownload`: attachment download, regenerate/edit actions, and Blob URL cleanup.

Preset metadata comes from the backend rather than another frontend hard-coded table.

## API and RunSpec Contract

### Endpoints

```http
GET /notebooks/static/presets
POST /notebooks/static
```

The POST is synchronous and returns:

```http
200 OK
Content-Type: application/x-ipynb+json
Content-Disposition: attachment; filename="<scene>_static_splat.ipynb"
```

It does not call `/process` or `JobManager`.

### Versioned RunSpec

```json
{
  "schema_version": 1,
  "input_folder": "captures/myroom",
  "frame_selection": {
    "mode": "smart",
    "fixed_fps": 4
  },
  "quality": {
    "profile": "balanced_l4",
    "n_iters": null,
    "max_gaussians": null,
    "advanced": {}
  },
  "publish": {
    "replace_owned_result": true
  }
}
```

The generator embeds exactly one safely JSON-encoded RunSpec cell. User strings are never interpolated as source fragments.

### Stable runner

The notebook invokes one command using an argument array:

```text
python scripts/notebook_static_run.py --spec /content/run_spec.json
```

The runner validates the schema again and applies all overrides through a public configuration contract. It must not mutate private module state from notebook cells. The existing `static_3dgs.py` runner may be extended to accept a config file, or `notebook_static_run.py` may adapt a validated RunSpec to it; the public contract remains JSON data.

## Generated Notebook Cell Flow

1. **Title and embedded RunSpec**
2. **GPU, VRAM, CUDA, disk, and Python preflight**
3. **Mount Google Drive and resolve canonical input/output paths**
4. **Clone a fresh repository and checkout a pinned, remotely reachable commit SHA**
5. **Bootstrap dependencies and validate them in fresh subprocesses**
6. **Copy the complete input folder into a unique local workspace**
7. **Run `notebook_static_run.py --spec ...` with `subprocess.run(argv, check=True)`**
8. **Validate the locally staged result inventory and checksums**
9. **Publish through a temporary Drive folder and finalize the exact result path**
10. **Print a concise result and diagnostics summary**

Run-All constraints:

- no mutable `git pull`;
- no shell command assembled from user strings;
- no `!{command}` whose failure can be ignored;
- no Torch/NumPy imports in the notebook kernel before bootstrap;
- no runtime restart requirement;
- no training or symlinking directly through Google Drive.

## Input Discovery

The input folder supports one unambiguous source layout:

1. exactly one root video with extension `.mp4`, `.mov`, or `.m4v`; or
2. supported root-level images; or
3. a supported `images/` directory.

The runner rejects multiple videos, mixed video-and-image layouts, empty image sets, and nested ambiguity. It records the full source inventory and hashes in `run_manifest.json`.

The Drive folder is never modified. The runner copies it locally and performs all frame extraction, caching, COLMAP, training, and export inside a fresh workspace.

## Smart Frame Selection

Smart selection is a deterministic two-pass process.

### Pass 1: candidate analysis

- Decode timestamped 320 px long-edge candidates at up to 12 FPS from the original video; videos below 12 FPS use every source frame.
- Preserve source PTS/timestamps and orientation.
- Calculate sharpness, exposure, duplicate similarity, and visual-overlap/continuity metrics.
- Partition time into windows derived from video duration and the profile's selected-frame budget, then preserve temporal order.

### Pass 2: native selected-frame extraction

- Choose the strongest candidate per window while preserving bridge frames and coverage.
- Apply a profile-derived selected-frame budget.
- Extract only selected source timestamps at the profile's permitted resolution without upscaling.
- Assign immutable frame IDs and write `selection_manifest.json`.

The one allowed backfill retry adds at most two additional bridge candidates per uncovered interval, then reapplies the profile budget by removing the weakest redundant frames outside those intervals.

The manifest includes source hash, selector version/policy, selected and rejected candidates, metrics, rejection reasons, timestamps, and the digest of the final image set.

Smart mode does not promise recovery when every frame in an interval is blurred. That condition is reported as a capture limitation.

### Fixed-FPS baseline

Fixed-FPS mode uses deterministic time sampling at the selected FPS and writes the same manifest schema. It does not run smart gap backfill. This lets all downstream code consume one selected-frame contract.

For image-set input, Smart mode scores blur, exposure, duplicates, and visual continuity using EXIF capture time when available and natural filename order otherwise. Fixed-FPS mode means **all input images unchanged**; the FPS value is ignored and the effective mode is recorded as `photo_set_all`.

## COLMAP Reconstruction and Gates

### Cache and staging corrections

- Frame cache identity includes source inventory/hash, extraction policy, selector version, and settings.
- COLMAP cache identity includes the selected-frame digest, camera/matcher policy, and COLMAP version.
- Invalidated extraction replaces the complete frame directory and removes stale tail frames.
- A COLMAP attempt runs in a new staging directory containing no previous DB or sparse model.
- A failed attempt cannot damage a previously validated reconstruction.
- Successful staged output is promoted only after validation.
- Cache invalidation propagates to downstream stages.

### Matcher policy

Matching is runtime policy, not a primary frontend quality control. The initial v1 policy is explicit:

- CUDA COLMAP with at most 800 selected frames uses exhaustive matching.
- CPU COLMAP uses sequential matching with overlap 20 on the first attempt.
- After smart backfill, CUDA COLMAP again uses exhaustive matching when the augmented set remains at most 800 frames.
- After smart backfill on CPU, sets of at most 300 frames escalate to exhaustive matching; larger sets use sequential overlap 40.
- When exhaustive matching already ran, the retry changes the selected bridge frames rather than rerunning the same inputs and expecting different information.
- Inputs above 800 selected frames are reduced deterministically to the profile budget before reconstruction rather than sent to an unbounded matcher.

These constants are part of a versioned reconstruction policy and are covered by command-construction tests. They are not frontend knobs.

### Model ranking

Sparse submodels are ranked by:

1. registered-image count;
2. temporal coverage and maximum gap;
3. registered share of the total selected set;
4. track/reprojection quality;
5. usable sparse-point support.

Binary file size is not a quality score.

### Initial production gate

After at most one smart backfill retry, the dominant model must satisfy:

- at least 90% of selected frames registered (95% target);
- the dominant model contains at least 95% of all images registered across every produced submodel;
- no interior uncovered interval longer than 2 seconds;
- capture endpoints covered within 1 second;
- median reprojection error at most 1 px;
- p95 reprojection error at most 2.5 px;
- median point track length at least 3;
- all registered image names unique and present;
- finite, valid intrinsics and poses;
- one dominant reconstruction component suitable for the training set.

Failure stops before training. The report includes a contact sheet, component summary, registered ratio, and uncovered source-video intervals.

Thresholds are policy-versioned and may be calibrated from real runs, but lowering them is not a silent fallback.

### Downstream alignment

Every frame, pose, depth map, and optional derived artifact is joined by immutable frame ID or exact filename. Positional list pairing is forbidden. This fixes the current incomplete-registration depth-alignment defect.

## Training and Quality Profiles

The selected profile resolves into a complete validated training configuration. Smart versus fixed-FPS selection does not change training hyperparameters.

Product runs use confidence-based sparse initialization, bounded resolution with no upscaling, static foundation depth where enabled by the profile, and the existing density/anisotropy controls. `n_iters` and `max_gaussians` overrides must remain within backend-defined safe bounds.

True NVS evaluation stays optional. Product quality checks may render sampled registered training cameras, and the report must label them as training-view checks rather than held-out metrics.

## Guarded Post-training Polish

The final PLY passes through a conservative `polish_static_ply` stage:

1. Validate finite positions, features, opacity, scales, and normalized quaternions.
2. Prune using opacity, robust relative scale/anisotropy, chunked multi-camera frustum support, and conservative observed-core bounds.
3. Fade opacity only in a narrow crop margin rather than making a hard visual cut.
4. Render sampled registered cameras before and after polish.
5. Reject the polished output and keep the raw export if removal limits, weighted-opacity loss, or render-regression limits are exceeded.

The initial conservative acceptance limits are:

- at most 15% of Gaussians removed;
- at most 5% of total opacity mass removed;
- mean sampled-view PSNR drop at most 0.25 dB;
- mean sampled-view SSIM drop at most 0.005;
- no individual sampled view loses more than 1.0 dB PSNR.

Any exceeded limit selects the raw PLY and records the rejected polish metrics.

The cleanup stage complements, rather than replaces, training-time opacity and scale pruning.

### Orientation and environment metadata

- Estimate viewer-up from COLMAP camera-up agreement plus a fitted dominant plane.
- Apply an orientation metadata transform only when camera-up and plane-normal estimates agree within 20 degrees and the plane fit has at least a 0.5 inlier ratio.
- Derive navigation bounds, ground estimate, and default camera from observed reconstruction/camera data.
- Use a real central registered camera for the initial view rather than a hard-coded origin.
- Emit sky/background/environment recommendations as metadata only.

No synthetic sky Gaussians are added to `splat.ply`. SuperSplat can choose its own background; this project's viewer can consume the metadata.

## Result Contract

A successful run publishes exactly:

```text
MyDrive/<parent>/<input_folder_name>_result/
  splat.ply
  scene_metadata.json
  preview.png
  quality_report.json
  run_manifest.json
  logs/
  world/                     # optional viewer enhancement
  orbit.mp4                 # optional when successfully produced
  _SUCCESS
```

`splat.ply` is a standard INRIA 3DGS PLY. The `world/` directory contains viewer metadata/assets, not a bundled web application.

`quality_report.json` contains selection mode and metrics, COLMAP components/registration/gaps/reprojection/tracks, training summary, cleanup decision/reasons, orientation confidence, and warnings.

`run_manifest.json` contains the RunSpec, effective config, source and artifact hashes, repository commit, tool versions, timings, hardware, artifact inventory, and success status.

## Transactional Drive Publishing

1. Assemble and verify the complete bundle locally.
2. Copy it to `<input_folder_name>_result.__tmp__<run_id>` in the same Drive parent.
3. Verify required files and checksums from the staged Drive copy.
4. Write `_SUCCESS` last.
5. If an owned final result exists, move it to `<input_folder_name>_result.__backup__<run_id>`.
6. Move the staged directory to the exact final result path and verify it again.
7. Delete the backup only after final verification; restore it if finalization fails.

An existing final folder is replaceable only when `run_manifest.json` contains the expected `generator_id`, supported manifest schema, and successful ownership marker, and replacement was enabled in the RunSpec. An unowned folder causes a safe failure rather than deletion. A failed new run preserves the previous good result. Same-parent directory moves are preferred; when the Drive filesystem cannot provide them, the publisher uses copy-and-verify with the same backup/restore states rather than merging into the final directory.

Replacements do not merge directories, preventing stale artifacts from surviving.

When reconstruction fails, a small sibling diagnostics path is allowed:

```text
<input_folder_name>_result_diagnostics/<run_id>/
```

It may contain the quality report, contact sheet, logs, and uncovered-interval report, but never masquerades as a successful splat result.

## Error Handling

| Failure | Behavior |
|---|---|
| Missing/ambiguous Drive input | Fail before copying or GPU work with the accepted layouts |
| No CUDA or unsafe VRAM for profile | Fail in preflight and recommend a lower profile |
| Dependency/bootstrap failure | Raise from a checked subprocess; do not continue |
| Selection finds inadequate source coverage | Publish diagnostics; do not run COLMAP/training |
| COLMAP gate fails after allowed retry | Publish diagnostics; do not train |
| Training fails or emits stale/missing output | Do not publish or replace the prior result |
| Polish regresses sampled renders | Fall back automatically to the validated raw PLY and record the reason |
| Drive staging verification fails | Preserve the prior result and retain clear logs |
| Final path is unowned | Fail safely; never delete it |

## Testing Strategy

### Frontend

- Drive path normalization and exact sibling result preview.
- Smart selected by default; Fixed FPS reveals and serializes FPS.
- Balanced/L4 selected by default.
- Profile defaults, overrides, and reset behavior.
- Review payload exactly matches the effective UI state.
- Attachment filename parsing, Blob lifecycle, regenerate/edit, and error retry.
- Notebook generation stays on the Notebook page.
- Existing local job submitter remains reachable and unchanged.

### Backend and generator

- RunSpec schema, bounds, traversal, control-character, and code-injection tests.
- Preset endpoint is sourced from backend metadata.
- Generated notebook parses through `nbformat` and has stable cell ordering.
- Values appear only as JSON data, not executable interpolation.
- Pinned commit and checked-subprocess behavior.
- Input discovery for one video, root images, and `images/`; rejection of ambiguous layouts.
- Input remains unchanged through a mocked run.

### Selection, cache, and COLMAP

- Source-byte changes invalidate frames even with unchanged settings.
- Selection-manifest changes invalidate COLMAP.
- Settings mismatch and force mode replace frames and remove stale tail files.
- A staged rerun starts without old DB/sparse models; failed staging preserves a prior valid reconstruction.
- A smaller-byte model with more registered images ranks above a larger-byte weak model.
- A synthetic 502-frame/12-component/285-main-model fixture blocks training and requests backfill.
- A healthy 95/100 fixture passes.
- Sequential-overlap configuration reaches the command when sequential matching is selected.
- Registered names align exact corresponding depth/derived files.

### Polish and publishing

- Invalid PLY fields fail validation.
- Conservative pruning and crop limits.
- Render-regression rejection keeps the raw PLY.
- Metadata transform is omitted when orientation confidence is low.
- Failed runs never create `_SUCCESS` or replace a good result.
- Owned result replacement works; unowned result replacement fails.
- Artifact inventory and checksums match the finalized folder.

### Hardware and end-to-end acceptance

- Generated-notebook smoke run with mocked Drive/subprocess boundaries.
- Balanced/L4 real smoke run with peak VRAM target below roughly 20 GB.
- Smart-on versus Smart-off comparison using identical downstream configuration and recorded registration, coverage, and visual metrics.
- At least one Premium/Ultra A100 validation.
- `splat.ply` opens in SuperSplat and this project's viewer consumes `scene_metadata.json`/`world/`.

## Compatibility and Existing Work

- Do not modify the three existing verification/production notebooks as part of generation.
- Keep `/process`, `JobManager`, and `Static3DSubmit` behavior intact.
- Preserve the current dirty worktree, including the modified `backend/api.py` and untracked static-preset tests/helpers.
- The new preset manifest must eliminate future frontend/backend preset duplication without overwriting unrelated local changes.
- All implementation work stays on a feature branch and commits only intentional files.

## Acceptance Criteria

The feature is complete when:

1. The frontend generates a valid notebook from a MyDrive-relative folder and the approved controls.
2. Smart and fixed-FPS notebooks differ only in selection/backfill behavior and record that difference.
3. Balanced/L4 is the default and passes a real L4 smoke run within the VRAM target.
4. A weak fragmented reconstruction such as the documented 285/502 case cannot start training.
5. Cache invalidation and clean staging prevent stale frames or COLMAP models from being accepted.
6. A successful Run-All creates exactly one validated `<input_folder_name>_result` bundle with standard PLY and no web viewer.
7. A failed run preserves any prior good result and produces actionable diagnostics.
8. Post-training polish cannot silently replace the raw model when sampled renders regress.
9. Existing notebooks and local submission behavior continue to work.

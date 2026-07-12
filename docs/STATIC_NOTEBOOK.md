# Static Gaussian Splat Notebook

The Notebook tab turns an existing Google Drive capture into a pinned Colab
notebook. In Colab, choose a suitable NVIDIA GPU runtime, use **Run all**, grant
Drive access, and wait for a verified result folder beside the capture.

## 1. Put the capture in MyDrive

Enter a path relative to `MyDrive`, for example `captures/myroom` or
`MyDrive/captures/myroom`. Do not paste a Windows path, `/content/drive/...`, a
Drive URL, or a folder ending in `_result`.

The input folder must use exactly one media layout:

- one supported video at the folder root; or
- photos directly at the folder root; or
- photos directly inside an `images/` subfolder.

Do not mix these layouts. Other non-media files are allowed and are included in
the source fingerprint.

## 2. Choose frame selection

- **Smart (default):** analyzes sharpness, exposure, duplicates, and overlap,
  then selects a bounded set. If the first COLMAP attempt exposes a recoverable
  coverage gap, it may replace a small number of frames once and retry.
- **Fixed FPS:** samples the video timeline uniformly at the selected rate. FPS
  4 is the default baseline. It uses the same reconstruction gate, training,
  polish, metadata, and publication stages as Smart.

The result records the effective selection, every rejection reason, and any
Smart backfill. Comparing Smart and Fixed therefore compares frame selection,
not two different trainers.

## 3. Choose a backend-owned quality profile

The app loads these values from the backend rather than hard-coding UI defaults:

| Profile | Minimum VRAM | Iterations | Gaussian cap | Frame budget | Long edge |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced / L4 (default) | 22 GB | 30,000 | 250,000 | 300 | 1280 |
| High | 30 GB | 50,000 | 500,000 | 450 | 1920 |
| Premium | 46 GB | 100,000 | 1,000,000 | 600 | 2560 |
| Ultra | 75 GB | 120,000 | 3,000,000 | 800 | native |

Primary and advanced overrides are optional. Blank controls keep the selected
profile value. A generated notebook performs an early VRAM check, and the runner
checks VRAM again together with input-dependent free disk requirements.

## 4. Generate and run

1. Review the exact `MyDrive` input and `<input>_result` paths.
2. Select **Generate and download notebook**.
3. Open the `.ipynb` in Colab.
4. Connect a GPU runtime appropriate for the profile.
5. Select **Runtime → Run all**.
6. Grant the notebook permission to mount Google Drive.

The notebook clones the fixed HTTPS repository, checks out one remotely reachable
40-character commit in detached mode, runs the dedicated bootstrap with checked
argument lists, and executes the validated RunSpec. It never performs `git pull`
or interpolates the Drive folder into a shell command.

## 5. Result folder

For `MyDrive/captures/myroom`, success is published to
`MyDrive/captures/myroom_result`. The required contents are:

- `splat.ply` — standard portable INRIA/3DGS PLY;
- `scene_metadata.json` — bounds, real COLMAP camera data, orientation evidence,
  navigation hints, and environment recommendations;
- `preview.png` — GPU render or deterministic sparse fallback;
- `quality_report.json` — selection, every COLMAP attempt/component, training,
  polish, orientation, and warning evidence;
- `run_manifest.json` — RunSpec, resolved config, source and artifact hashes,
  versions, hardware, timing, and commit provenance;
- `selection_manifest.json` and, when selection changed,
  `source_selection_manifest.json`;
- non-empty `logs/`; and
- `_SUCCESS`, written only after the copied manifest hash verifies.

Optional `world/` contains JSON-only viewer hints, and optional `orbit.mp4` may be
present. The result deliberately contains no HTML, JavaScript, WASM, or bundled
web viewer. Open `splat.ply` in [SuperSplat](https://superspl.at/editor) or use the
project viewer with `scene_metadata.json` and optional `world/` files.

The environment/sky recommendation is metadata only. The pipeline does not add
fake sky geometry to the splat.

## 6. Reconstruction failures and diagnostics

A failed source, selection, or reconstruction may write only
`<input>_result_diagnostics/<run_id>/`. It never writes `_SUCCESS` there and
never replaces a prior result. The quality report explains these gate names:

| Gate | Meaning / required value |
| --- | --- |
| `registered_ratio` | At least 90% of selected frames registered. A 90–95% pass is recorded as a warning. |
| `dominant_component` | The accepted component owns at least 95% of all registered images. |
| `interior_gap` | No interior capture gap exceeds 2 seconds (or the corresponding photo-order coverage rule). |
| `start_endpoint` | Start coverage gap is at most 1 second. |
| `end_endpoint` | End coverage gap is at most 1 second. |
| `median_reprojection` | Median reprojection error is at most 1 pixel. |
| `p95_reprojection` | 95th-percentile reprojection error is at most 2.5 pixels. |
| `median_track_length` | Median sparse-point track length is at least 3 observations. |

Improve failures by capturing a slow, steady loop with consistent exposure,
visible texture, overlap, and complete start/end coverage. Smart performs at most
one bounded gap-repair retry; it does not weaken these gates.

## 7. Safe replacement

Publication first validates a fresh sibling staging copy and writes `_SUCCESS`
last. An existing `<input>_result` is replaceable only when its manifest and
success marker authenticate it as this generator's output and replacement is
enabled. The old result moves to a backup, the new result is moved and verified,
and only then is the backup removed. Any failure through final verification
restores the previous good result. Unowned folders and symlink-modified results
are never deleted or merged.

## 8. Operator release pinning

By default the API pins `git rev-parse HEAD` only when that full SHA is an
ancestor of the configured upstream. Push the release commit before generating a
notebook. Detached/package deployments can explicitly set a trusted immutable
release:

```bash
export STATIC_NOTEBOOK_COMMIT_SHA=<40-lowercase-hex-release-sha>
```

The notebook runs `colab/static_notebook_bootstrap.sh`, which keeps the three
legacy notebooks and `colab/bootstrap.sh` unchanged. The wrapper installs the
CUDA COLMAP policy and then validates NumPy, Torch/CUDA, gsplat, COLMAP, FFmpeg,
Pydantic, and nbformat in fresh processes. A failed validation stops Run all; it
does not report success or ask the user to continue after a hidden restart.

## 9. Manual hardware acceptance (pending until run)

Local non-GPU tests do not prove L4/A100 memory or quality. These checks remain
manual and must not be marked passed without their resulting reports.

### NVIDIA L4 24 GB

Generate two Balanced/L4 notebooks for the same capture: one Smart and one Fixed
FPS 4. Before and during each run, record GPU memory:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
watch -n 2 nvidia-smi --query-compute-apps=used_memory --format=csv,noheader
```

For each result, retain `run_manifest.json` and `quality_report.json`, and compare
selected frames, registered ratio/components/gaps, total and per-stage timing,
polish decision, and metrics labelled `training_view_checks`. The planned L4
acceptance target is peak VRAM below roughly 20 GB; this repository does not claim
that target has been observed in this implementation session.

### NVIDIA A100 80 GB

Run at least one Premium or Ultra notebook and retain the same evidence:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
python - <<'PY'
import torch
print(torch.cuda.get_device_name())
print(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
PY
```

No L4 or A100 hardware acceptance run was performed as part of the local
implementation and non-GPU verification documented here.

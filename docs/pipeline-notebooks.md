# Reusable Colab notebook generator

Open the existing **Notebook Generator** screen and select a pipeline:

| Pipeline | Input | Processing |
|---|---|---|
| Our preprocessing + our trainer | MyDrive capture folder | Existing static pipeline and quality profiles |
| Spirula dataset + our trainer | Dataset ZIP or folder | Import existing images and COLMAP cameras/points, then train our model; no SfM rerun |
| Spirula preprocessing + Spirula trainer | Video, dataset ZIP or folder | Pinned Spirula installation, SfM if needed, geometry and training |

Paths are relative to MyDrive, for example `GaussianTests/inputs/room.MOV` or
`GaussianTests/inputs/room.zip`. Do not enter a Windows path or a Drive sharing URL.
Downloading a notebook does not start a GPU session. Open it in Colab, select a
GPU, and run its cells. Each external-pipeline run gets a unique Drive folder.

## Presets

| Preset | Hybrid | Spirula |
|---|---|---|
| Baseline | 1M / 30k, 1920px | 1M / 30k, medium training quality |
| Quality experiment | 3M / 50k, 1920px | 6M / 60k, high training quality |
| Ultra experiment | 5M / 60k, 2560px | 10M / 80k, ultra training quality |

Iterations and Gaussian budgets can be overridden. Presets are experiments,
not a ranking or a guarantee of better output. They are not exact replicas of
an arbitrary local Spirula `config.json`. Minimum VRAM checks are conservative
preflight policies, not measured peak-memory guarantees for every dataset.
Custom larger budgets may still exhaust memory; they are never silently reduced.

The hybrid importer supports one calibration and SIMPLE_PINHOLE, PINHOLE,
SIMPLE_RADIAL or OPENCV cameras. It undistorts images and observations together.
It uses `images/` and `sparse/<model>/`; depth/normal maps are not used. Its
Gaussian cap is nominal and checkpoints do not contain full optimizer state.

Spirula dataset mode reuses sparse reconstruction but regenerates geometry.
Video mode exposes extraction FPS and SfM quality. Normal maps are always
generated; depth guidance is optional. The quality gate stops incomplete
reconstructions unless the user explicitly chooses the partial option.
Exit code 3 from SfM is followed by model inspection, not treated as total failure.
Geometry exit -11 is recoverable only when the completion marker and every
selected image's decoded normal/depth output validate. Other failures propagate.
Drive outages preserve local Spirula logs and offer an export retry cell.

## Command line

From the repository root, with Python dependencies installed:

```powershell
python -m scripts.generate_pipeline_notebook --pipeline spirula --input-mode video --input-path GaussianTests/inputs/room.MOV --preset quality --fps 4 --output room_spirula.ipynb
python -m scripts.generate_pipeline_notebook --pipeline hybrid --input-mode dataset_zip --input-path GaussianTests/inputs/room.zip --preset baseline --output room_hybrid.ipynb
python -m scripts.generate_pipeline_notebook --pipeline native --input-mode folder --input-path captures/room --preset baseline --output room_native.ipynb
```

The CLI refuses to overwrite an existing notebook. The existing source resolver
requires a pushed commit/upstream or an explicitly configured release SHA.
The native path keeps its existing immutable repository checkout. Hybrid
notebooks embed current trainer sources and the importer with a SHA-256 check;
no E: drive, developer desktop path or external builder script is needed.

## Maintenance

- `backend/notebooks/pipeline_library.py`: validated request, presets, safe literal
  parameter replacement, deterministic source bundle and dispatch.
- `backend/notebooks/templates/`: clean versioned notebook recipes and importer.
  These replace the old per-experiment builders under E:; edit them here once.
- `GET /notebooks/static/pipelines`: external-pipeline preset catalog.
- `POST /notebooks/static/pipeline`: notebook attachment for any of the three paths.
- Existing `POST /notebooks/static` remains compatible with the original UI/API.

The Spirula recipe downloads the official **v2026.9.24** Linux Vulkan binary from
[Spirula Studio](https://github.com/harry7557558/spirula-studio) and verifies the
pinned checksum. It does not vendor Spirula source or binaries. The inherited
Vulkan recovery only extracts matching userspace libraries for the supported
580.82.07 driver; it does not install a kernel driver or promise compatibility
with future Colab images. Retest a smoke run when changing the release/runtime.

Run `python -m pytest tests/notebooks` and the frontend notebook tests after
editing recipes. These validate generation, request handling and configuration;
they do not replace a Colab GPU smoke test. The new integration has not itself
been trained end to end in Colab.

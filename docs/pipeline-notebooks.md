# Reusable Colab notebook generator

Open the existing **Notebook Generator** screen and select a pipeline:

| Pipeline | Input | Processing |
|---|---|---|
| Our preprocessing + our trainer | MyDrive capture folder | Existing static pipeline and quality profiles |
| Spirula dataset + our trainer | Dataset ZIP or folder | Import existing images and COLMAP cameras/points, then train our model; no SfM rerun |
| Spirula preprocessing + Spirula trainer | Video, dataset ZIP or folder | Pinned Spirula installation, SfM if needed, geometry and training |
| Spirula dataset hazırlama | Videos and photo folders | Preprocessing only; detailed camera/mask/geometry controls, dataset ZIP and quality report |
| Metinden hedef splat (Colab) | Text prompt, seed and quality preset | SDXL reference image → TRELLIS native Gaussians → normalized 3DGS PLY and turntable GIF |

See [Spirula preprocessing](spirula-preprocessing.md) for the preparation-only screen, reusable settings files and output layout.

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
The text-to-splat recipe embeds its own runtime and records local HEAD for
provenance; it does **not** require pushing a feature branch.

## Text → target splat B (v2, Colab first)

1. In **Notebook Generator**, choose **Metinden hedef splat (Colab)**.
2. Enter **Nesne tarifi (PROMPT)**, for example `a red sports car`, set
   **Tohum (SEED)** to `42`, and start with **Dengeli · L4 24 GB**.
   Under optional settings, use **Hedef splat sayısı = 200000** for a first morph
   experiment. Negative prompt, style and a custom Drive folder are optional.
3. Click **Metinden splat notebook’unu indir**, open the downloaded `.ipynb` in
   Google Colab, select an **L4** or **A100** GPU, and choose **Run all / Tümünü
   çalıştır**. Complete Colab's Drive authorization prompt. No local model
   inference or multi-GB download happens in the app.
4. Download `target.ply` from
   `MyDrive/GaussianTests/text_to_splat/<prompt-slug>-<prompt-hash>/`.
   A custom output path is relative to MyDrive in the app/CLI; the generated
   notebook exposes its full `/content/drive/MyDrive/...` path as `OUTPUT_DIR`.
5. Open **Sahne Editörü** and your existing myroom scene. Under **Obje ekle**,
   click **Dosya yükle**, upload `target.ply`, select it and click **Sahneye ekle**.
   Center, rotate and scale B to match the intended transition. Select **A** on
   myroom's row and **B** on the new object's row. In **Morph**, select
   **Şekil koruyan** or **Parçacık bulutu**, then **Morph hazırla → Oynat**.
   See [morph controls](splat-morph.md) for recording and budgets; morph currently
   accepts at most 1,000,000 raw splats per asset.

If an already-running backend was started without auto-reload, it continues to
serve the old catalog until the user restarts it. This change does not restart
or stop running servers. CLI generation can be used immediately.

```powershell
python -m scripts.generate_pipeline_notebook --pipeline text_to_splat --prompt "a red sports car" --seed 42 --preset baseline --target-splat-count 200000 --output red_car.ipynb
```

All parameters use validated JSON/Python literals through `replace_assignments`,
including quotes, Unicode and multiline prompts; no prompt is interpreted as
code or shell arguments. Prompt length is 1–2000 characters, negative prompt
0–2000, style 0–500, seed 0–2147483647. SDXL's text encoders can truncate long
prompts despite the UI's character allowance, so short object descriptions are
recommended. `target_splat_count` is absent or 1000–2000000. It is an **upper
cap**, using deterministic importance sampling without replacement; a smaller
native result is preserved. Thinning is lossy and does not lower inference VRAM.

| Preset | Image / structure / latent steps | Required VRAM policy |
| --- | --- | --- |
| baseline / Dengeli | 25 / 12 / 12 | 20 GiB (L4 24 GB) |
| quality / Kaliteli | 40 / 20 / 20 | 20 GiB (L4 24 GB) |
| ultra | 50 / 25 / 25 | 38 GiB (A100 40/80 GB) |

All use 1024px SDXL images and the same TRELLIS Gaussian decoder. Presets change
sampling steps, not model resolution or a guaranteed quality ranking. These are
conservative preflight policies, not measured VRAM peaks. Low host RAM, unusual
object density or busy GPUs can still exhaust memory. T4 and the local 8 GB GPU
are intentionally rejected before model downloads.

The recipe uses SDXL base 1.0 followed by `microsoft/TRELLIS-image-large`.
Microsoft recommends the image-conditioned route over its direct text models
for richer detail. TRELLIS.2's inspected release exposes mesh/GLB inference,
without a shipped Gaussian decoder. See the [dated research, alternatives,
licenses, wheel evidence and exact pins](text-to-splat-research.md).

Colab's default Python changed to 3.13 in August 2026. The notebook leaves its
kernel intact for Drive/display and uses uv `0.8.22` to create a private
Python **3.11.11**, torch **2.4.0+cu121**, torchvision **0.19.0+cu121**, xformers
**0.0.27.post2**, spconv-cu120 **2.3.6** / cumm-cu120 **0.4.11** environment.
Transitive dependencies are locked; only prebuilt wheels are installed. No
Kaolin, nvdiffrast, flash-attn or native preview rasterizer is needed. Tiny CUDA
kernels and all four selected model imports are checked before weights download.
Only Gaussian models are loaded; explicit, version-checked import patches remove
unused upstream mesh/rendering dependencies without changing model arithmetic.
At least 35 GiB free Colab disk space is required. Fresh sessions redownload the
environment/models; completed Drive stages survive.

Outputs in `OUTPUT_DIR`:

- `target.ply`: binary little-endian standard INRIA 3DGS. Fields include `x/y/z`,
  zero normals, `f_dc_0..2`, `f_rest_0..44`, opacity **logit**, scale **natural
  logarithm**, and normalized rotation **wxyz**. TRELLIS predicts SH0; higher SH
  fields are explicitly zero padded, not invented view-dependent detail.
- `turntable.gif`: 36-frame, 256px approximate CPU Gaussian preview, using at
  most 12,000 important splats. It uses projected anisotropic covariance and
  depth-sorted SH0 blending; it is not a Spark rendering or quality benchmark.
- `input.png`, `cutout.png`: generated reference and transparent background mask.
- `gaussians_raw.npz`: activated Gaussian arrays for resumable export, no pickle.
- `manifest.json`: prompt/style/negative, effective image prompt, seed, quality,
  exact model revisions, recipe hash, package/GPU versions, stage times and
  checksums, input/output counts, coordinate transform and preview limitations.
- `logs/`: setup and stage logs (also retained on Colab's local disk).

Coordinates use a right-handed **+Y-up** frame. TRELLIS source `(x,y,z)` maps to
`(x,z,-y)` with the same −90° X rotation applied to each quaternion. The original
Gaussian-centre AABB is centered at the origin and its longest side scaled to
one unit; covariance scales receive the same uniform factor. These units are
**not metres**. COLMAP world scale/orientation is arbitrary, so the user must
align B in the editor; normalization does not register it to myroom automatically.

Run all again with unchanged settings to resume. Stage reuse requires matching
configuration/recipe hashes and output SHA256 checksums. Failed/corrupt stages
are regenerated and downstream stages invalidated. Use a **new OUTPUT_DIR**
when changing prompt, seed, preset or splat cap; the notebook refuses to overwrite
a different experiment or an unowned nonempty folder. A GPU stage interrupted
mid-inference restarts that stage; this is not sampler checkpoint recovery.
Drive disconnects can interrupt writes; local logs survive but unfinished model
inference is not automatically exported from a disconnected runtime.

Local validation covers notebook AST/nbformat, literal safety, HTTP/CLI behavior,
normalization/quaternion math, SH ordering, log/logit round trips, cache
invalidation and GIF decoding. The same synthetic export is loaded through the
frontend's actual `@mkkellogg/gaussian-splats-3d` PLY parser. Spark rendering and
Colab L4/A100 inference remain empirical checks. No multi-GB model weights were
downloaded or run locally; this delivery does not claim measured visual quality,
installation duration, peak VRAM or an end-to-end successful Colab run.

Validation on 2026-10-07: `E:\anaconda3\envs\gs4d\python.exe -m pytest tests/notebooks`
passed **155 tests** (temporary files redirected inside the worktree);
`npx vitest run` passed **313 tests / 37 files** with two local workers;
`npm run build` passed with the existing bundle-size warning. The Python run
also exposed an older Spirula `hashlib.file_digest` dependency on Python 3.11;
its streaming checksum helper now supports the requested Python 3.10 test env.

## Maintenance

- `backend/notebooks/pipeline_library.py`: validated request, presets, safe literal
  parameter replacement, deterministic source bundle and dispatch.
- `backend/notebooks/templates/`: clean versioned notebook recipes and importer.
  These replace the old per-experiment builders under E:; edit them here once.
- `GET /notebooks/static/pipelines`: external-pipeline preset catalog.
- `POST /notebooks/static/pipeline`: notebook attachment for native, hybrid, Spirula or text-to-splat.
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

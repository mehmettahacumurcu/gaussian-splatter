# Reusable Colab notebook generator

Open the existing **Notebook Generator** screen and select a pipeline:

| Pipeline | Input | Processing |
|---|---|---|
| Our preprocessing + our trainer | MyDrive capture folder | Existing static pipeline and quality profiles |
| Spirula dataset + our trainer | Dataset ZIP or folder | Import existing images and COLMAP cameras/points, then train our model; no SfM rerun |
| Spirula preprocessing + Spirula trainer | Video, dataset ZIP or folder | Pinned Spirula installation, SfM if needed, geometry and training |
| Spirula dataset hazırlama | Videos and photo folders | Preprocessing only; detailed camera/mask/geometry controls, dataset ZIP and quality report |
| Metinden hedef splat (Colab) | Text prompt, model/GPU settings and stage quality controls | Selectable image → native TRELLIS or experimental TRELLIS.2 mesh-to-splat → standard PLY and GIF |

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

## Text → target splat B (template v2, selectable models)

1. In **Notebook Generator**, choose **Metinden hedef splat (Colab)**.
2. Enter **Nesne tarifi (PROMPT)**, for example `a red sports car`, and a seed.
   Open **MODEL SETTINGS / MODEL AYARLARI** to choose a GPU profile, image model,
   background remover and 3D route. Every GPU profile defaults to stable native
   TRELLIS; **L4 24 GB** pairs it with SDXL + U2Net. TRELLIS.2 is an experimental
   opt-in. Each choice includes a Turkish explanation.
3. Adjust image steps/guidance/resolution and 3D sampler controls if needed.
   Negative prompt is available for SDXL, FLUX.1-dev and Qwen-Image; it is disabled
   for schnell and klein. Mesh view/fit controls appear for TRELLIS.2.
4. Download the notebook, open it in Colab, select the matching GPU and run cells.
   GPU, system RAM, disk and model access checks run before weight downloads.
   Complete Drive authorization. Gated models also require prior HF access and a
   `HF_TOKEN` Colab secret with notebook access enabled.
5. Download `target.ply` from
   `MyDrive/GaussianTests/text_to_splat/<prompt-slug>-<prompt-hash>/`, or your
   custom MyDrive-relative output folder.
6. In **Sahne Editörü → Obje ekle → Dosya yükle**, upload `target.ply` and add it
   to the scene. Align its position/rotation/scale, select your room as A and the
   generated object as B, then use **Morph hazırla → Oynat**.
   See [morph controls](splat-morph.md); the morph path accepts at most 1,000,000
   raw splats per asset.

Downloading a notebook does not run models or download weights locally. Profiles
do not guarantee that Colab will offer the named GPU. A running backend without
auto-reload continues serving its old catalog until restarted by the user; these
changes do not stop or restart the running servers.

### Model comparison and access

Checked against official model cards and HF metadata on **7 October 2026**.
VRAM figures below are conservative image/geometry stage admission policies,
not measured peak usage. Stages run sequentially; the total requirement is their
maximum. See [exact revisions, source evidence and limits](text-to-splat-research.md).

| Option | What changes / Turkish explanation | VRAM and license/access | Status |
| --- | --- | --- | --- |
| [SDXL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | Dengeli başlangıç; negatif istem destekli, küçük görüntü aşaması. | Image 12 GiB; CreativeML Open RAIL++-M, ungated. | Implemented, default. |
| [FLUX.1-dev 12B](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Ayrıntı için daha yavaş büyük model; negatif istem true CFG ile ek hesaplama yapar. | Image 36 GiB; non-commercial model license; HF acceptance + `HF_TOKEN`. | Implemented. |
| [FLUX.1-schnell 12B](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | 1–4 adımda hızlı deneme; guidance 0, negatif istem yok. | Image 36 GiB; Apache 2.0, currently gated: HF acceptance + `HF_TOKEN`. | Implemented. |
| [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Hızlı 4 adımlı damıtılmış model; guidance 1, negatif istem yok. | Image 16 GiB; Apache 2.0, ungated. | Implemented. |
| [Qwen-Image 20B](https://huggingface.co/Qwen/Qwen-Image) | Karmaşık istem/yazı için büyük model; daha yavaş, yüksek RAM gerekir, negatif istem destekli. | Image 60 GiB; Apache 2.0, ungated. | Implemented. |
| [U2Net/rembg](https://github.com/danielgatis/rembg/blob/v2.0.59/rembg/sessions/u2net.py) | CPU üzerinde hafif ve hızlı maske; ince kenarlar kaybolabilir. | CPU; rembg MIT, U-2-Net source Apache 2.0; independent ONNX weights license unverified. | Implemented, default. |
| [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) | İnce sınırlar için yüksek çözünürlüklü maske; daha ağır, görüntü modelinden sonra GPU kullanır. | Official weights/code MIT, ungated. | Implemented; no BRIA model. |
| [TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) | Doğrudan eğitilmiş Gaussian çözücü; daha kısa ve az bağımlılıklı yol. | Geometry 20 GiB; MIT, DINOv2 Apache 2.0. | Implemented, default. |
| [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) → mesh → gsplat | Daha yeni geometri; N görünüm ve 3DGS fit ek süre ister; PBR özellikleri splat'a tam taşınmaz. | Geometry 28 GiB, cap >150k: 32 GiB; MIT + required DINOv3 custom license, manual Meta approval + `HF_TOKEN`. | Implemented, experimental. |
| [Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) | Deneysel/sonra: şekil/doku CUDA bağımlılıkları ve texture bake henüz entegre değil. | Upstream reports 29 GB combined, not validated here; Tencent Community License with territorial limits. | Disabled stub; requests rejected. |

[FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) and
[FLUX.2-dev 32B](https://huggingface.co/black-forest-labs/FLUX.2-dev) also exist and
have gated non-commercial licenses. They are verified alternatives, not menu
options or implemented routes. No unverified model ID is accepted.

### GPU presets and quality knobs

| GPU profile | Default models | Image / sparse / SLAT steps | Mesh views / fit iterations / splat cap | VRAM policy |
| --- | --- | --- | --- | --- |
| L4 24 GB | SDXL / U2Net / TRELLIS | 25 / 12 / 12 | 24 / 1,500 / 50k; inactive | 20 GiB |
| A100 80 GB | FLUX.1-dev / BiRefNet / TRELLIS | 28 / 20 / 20 | 48 / 3,000 / 100k; inactive | 36 GiB |
| H100 80 GB | Qwen / BiRefNet / TRELLIS | 40 / 24 / 24 | 64 / 4,000 / 150k; inactive | 60 GiB |
| RTX PRO 6000 96 GB | Qwen / BiRefNet / TRELLIS | 40 / 24 / 24 | 72 / 5,000 / 200k; inactive | 60 GiB |

All presets use stable TRELLIS by default and retain their image models,
background removers and quality budgets. Mesh budgets apply only after explicitly
selecting experimental TRELLIS.2. That route requires CUDA source builds with
`nvcc`/`g++` and manual DINOv3 license approval. It has not been validated end to
end in a clean Colab GPU session.
The RTX PRO 6000 preset does not add native Blackwell support: its default can
be generated with a compatibility warning, but actual Blackwell hardware is
rejected by the native route's runtime preflight before downloads.

- **Image steps / guidance / resolution:** more steps usually take longer;
  guidance changes prompt adherence and can overconstrain the image. Choose
  512/768/1024 square pixels; resolution can increase VRAM. Qwen uses true CFG,
  while FLUX.1-dev uses guidance distillation and additionally true CFG when a
  negative prompt is supplied. Schnell requires 1–4 steps/guidance 0; distilled
  klein uses guidance 1.
- **Seed / TRELLIS seed:** repeatable image/3D starting noise, independently
  configurable. Cross-GPU bitwise reproducibility is not guaranteed.
- **Sparse-structure steps / CFG:** control coarse 3D occupancy and adherence to
  the reference. **SLAT steps / CFG:** control subsequent geometry/appearance.
  More steps are slower; stronger CFG is not a guaranteed quality improvement.
- **Mesh views / fit iterations / splat cap:** more directions improve coverage,
  more iterations spend longer matching those views, more splats increase
  fitting capacity and memory. TRELLIS.2 uses a fixed 512 reconstruction route
  and 512px training views; it does not expose an untested 1536 mesh preset.
- **Target splat count:** optional final cap, 1,000–2,000,000, separate from mesh
  fitting capacity (1,000–300,000). Deterministic thinning is lossy; it does not
  lower model inference memory. A smaller native result is preserved.

Legacy `baseline`/`quality`/`ultra` still work. Quality uses 20 sparse/SLAT
steps; ultra uses 25 and a minimum 38 GiB geometry policy. Explicit field values
override defaults. The notebook form and UI expose the same allow-listed models
and numeric limits.

In the downloaded Colab form, changing `GPU_PRESET` applies its combinations
only when `APPLY_GPU_PRESET` is enabled. Disable that checkbox to use the explicit
model and numeric fields again. Unsupported negative-prompt fields are hidden
when generating a schnell/klein notebook and rejected by runtime validation.

### Runtime, credentials and outputs

Image/background work uses private Python 3.11.11, torch 2.7.1 CUDA 12.8,
Diffusers 0.37.0 and Transformers 4.57.6. It passes PNG files to a second process:
native TRELLIS keeps torch 2.4 CUDA 12.1 and its existing wheels-only lock, while
TRELLIS.2 has a separate modern environment and compiles CUDA dependencies.
Colab's own kernel remains intact for Drive/display.

Driver 570+, adequate free VRAM, disk and available system RAM are required.
The largest image policy (Qwen) needs 56 GiB available system RAM and 90 GiB free
disk, plus 40 GiB disk for mesh setup. Smaller models use lower policies; exact
values are in the [research notes](text-to-splat-research.md). An incompatible
combination fails in Turkish instead of silently choosing a smaller model.
T4 and local 8 GB GPUs do not meet the complete pipeline's minimum.

RTX PRO 6000 Blackwell is explicitly rejected with the old native TRELLIS route
because its pinned torch 2.4/CUDA 12.1 wheels do not support that architecture.
Its preset still defaults to stable TRELLIS; it never silently selects an
experimental route. Explicitly opting into TRELLIS.2 requires a CUDA 12.8/12.9
toolkit/compiler in Colab; CUDA 13 is rejected for this cu128 recipe. This
adaptation has not been smoke-tested on those GPUs. GPU memory alone cannot
guarantee native extension compatibility.

For FLUX.1-dev/schnell, accept the model's HF conditions. For TRELLIS.2,
request and receive access to
[DINOv3 ViT-L/16](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
Then add `HF_TOKEN` using Colab's key/Secrets panel and enable notebook access.
An environment variable is also supported. The token is never a form field,
embedded literal, command argument or manifest value.

Outputs in `OUTPUT_DIR`:

- `target.ply`: binary little-endian standard INRIA 3DGS, with `x/y/z`, zero
  normals, SH DC, zero-padded `f_rest_0..44`, opacity logits, natural-log scales
  and normalized wxyz quaternions.
- `turntable.gif`: approximate 36-frame, 256px CPU preview, using at most 12,000
  important splats; not a Spark rendering or benchmark.
- `input.png`, `cutout.png`: reference image and selected background mask.
- `gaussians_raw.npz`: activated Gaussian arrays for resumable export; no pickle.
- Mesh route also preserves `mesh.npz` and `mesh_views.npz`. It initializes
  splats over the mesh surface, then optimizes positions, rotations, scales,
  opacities and colors with gsplat 1.5.3 and RGB/alpha loss. Population is fixed
  at the cap; there is no densification. It fits unlit color, not full PBR.
- `manifest.json`: settings, each chosen model ID/revision/license, required
  encoder/shared-decoder provenance, package/GPU versions, stage times/checksums,
  recipe hash, normalization and preview limitations.
- `logs/`: setup/stage logs, also retained on Colab disk.

Coordinates are right-handed **+Y-up**. Source `(x,y,z)` maps to `(x,z,-y)`
with the same −90° X rotation applied to quaternions. Gaussian-centre bounds are
centered at the origin and the longest side scaled to one; covariance scales
receive the same factor. Units are **not metres**. Align the generated B asset
to your scene in the editor.

All prompt/style strings are encoded with validated Python/JSON literals;
enums use allow-lists, never arbitrary model IDs or install URLs. Quoted,
Unicode and multiline prompts cannot become code or shell arguments. Length
limits remain prompt 1–2,000, negative prompt 0–2,000, style 0–500; seed is
0–2,147,483,647. Model text encoders can still truncate long descriptions.

Run all again with unchanged settings to resume. Reuse requires matching
configuration/recipe hashes and output checksums. Corrupt stages and downstream
results are regenerated. Use a **new OUTPUT_DIR** for another configuration:
different experiments and unowned nonempty folders are not overwritten.
Interrupted inference restarts that stage; this is not sampler/optimizer
checkpoint recovery. Drive failures can interrupt writes; local logs survive.

The existing CLI remains available for the default route:

```powershell
python -m scripts.generate_pipeline_notebook --pipeline text_to_splat --prompt "a red sports car" --seed 42 --preset baseline --target-splat-count 200000 --output red_car.ipynb
```

Local validation covers settings and request validation, safe literal replacement,
AST compilation of every notebook code cell, nbformat, runtime stage routing,
Gaussian export/math/cache helpers, the real frontend PLY parser and panel tests.
Required commands are `E:\anaconda3\envs\gs4d\python.exe -m pytest tests/notebooks`,
`npx vitest run` and `npm run build` (the latter two in `frontend/`).
No local test establishes model quality, peak memory, setup duration or a
successful cloud run. Clean Colab checks remain necessary, especially for the
experimental TRELLIS.2 route.

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

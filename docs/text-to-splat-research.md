# Text → target splat: model and Colab research

Checked **7 October 2026** against official model cards, Hugging Face model
metadata, source files and package metadata. No model weights were downloaded or
run locally. These checks establish existence, revisions and declared licenses;
they do not establish visual quality, peak VRAM or a successful Colab run.

## Implemented choices and deferred routes

The generator runs the image model and background remover in a private modern
environment, writes `input.png` and `cutout.png`, then starts a separate
reconstruction process. The original TRELLIS torch 2.4 pin no longer constrains
the image model. Both implemented routes export normalized standard 3DGS
`target.ply`.

| Model / route | Turkish explanation: quality, speed and memory | Verified license and access | Implementation |
| --- | --- | --- | --- |
| [SDXL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | Dengeli başlangıç; küçük görüntü aşaması, negatif istem destekli. Görüntü eşiği 12 GiB; varsayılan tüm hat 20 GiB. | CreativeML Open RAIL++-M; ungated. | Implemented; default. |
| [FLUX.1-dev, 12B](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Ayrıntılı görüntüler için daha yavaş büyük model; eşik 36 GiB. Negatif istem true CFG ile ek hesaplama yapar. | FLUX.1-dev Non-Commercial License; HF acceptance + `HF_TOKEN`. The card separately describes permitted uses of outputs. | Implemented. |
| [FLUX.1-schnell, 12B](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | 1–4 adımda hızlı deneme; büyük ağırlıklar nedeniyle eşik yine 36 GiB. Guidance 0; bu tarifte negatif istem yok. | Apache 2.0; **currently gated**, despite its permissive license. HF acceptance + `HF_TOKEN`. | Implemented. |
| [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 4 adımlı hızlı yeni seçenek; guidance 1, negatif istem yok. Tarif eşiği 16 GiB; upstream CPU-offload örneği yaklaşık 13 GB bildirir. | Apache 2.0; ungated. | Implemented. |
| [Qwen-Image, 20B](https://huggingface.co/Qwen/Qwen-Image) | Karmaşık istem ve yazı üretimi için büyük model; daha yavaş, 60 GiB görüntü eşiği ve yüksek sistem RAM gerektirir. Negatif istem destekli. | Apache 2.0; ungated. | Implemented; 80/96 GB GPU class. |
| [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), [FLUX.2-dev, 32B](https://huggingface.co/black-forest-labs/FLUX.2-dev) | Daha büyük doğrulanmış alternatifler; Colab bellek bütçesi bu tarifte sınanmadı. | FLUX Non-Commercial License; gated. | Existence verified; **not integrated or selectable**. |
| [U2Net via rembg](https://github.com/danielgatis/rembg/blob/v2.0.59/rembg/sessions/u2net.py) | CPU üzerinde hafif maskeleme; basit siluetlerde hızlı başlangıç, ince kenarlar kaybolabilir. Ayrı GPU bütçesi istemez. | rembg MIT; [U-2-Net source](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE) Apache 2.0. Separate ONNX weight license **unverified**. | Implemented; default. |
| [BiRefNet official weights](https://huggingface.co/ZhengPeng7/BiRefNet) | Yüksek çözünürlüklü maske, ince sınırlar için daha güçlü seçenek; U2Net'ten ağır. Görüntü modeli boşaltıldıktan sonra GPU kullanır. | Model card explicitly MIT; ungated. Remote implementation and weights share a pinned revision. | Implemented; `ZhengPeng7/BiRefNet`, **not BRIA RMBG-2.0**. |
| [TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) → native Gaussian | Doğrudan eğitilmiş Gaussian çözücü; kısa ve daha az bağımlılıklı yol. Eşik 20 GiB; upstream minimumu 16 GB. | TRELLIS code/checkpoints MIT; DINOv2 Apache 2.0. | Implemented; default route. |
| [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) → mesh → views → gsplat | Daha yeni geometri; görünüm üretimi ve optimizasyon kurulumu/çalışmayı uzatır. Mesh eşiği 28 GiB, 150k üzeri fit bütçesinde 32 GiB; upstream minimumu 24 GB. | TRELLIS.2 MIT; required [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) has its own license and **manual Meta approval** + `HF_TOKEN`. | Implemented **experimental** route; not an upstream Gaussian decoder. |
| [Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) → mesh → splat | Deneysel/sonra: şekil + doku + özel CUDA rasterizer kurulumu bu tarife eklenmedi. Upstream 10 GB şekil / 21 GB doku / 29 GB birlikte bildirir. | Tencent Hunyuan 3D 2.1 Community License; territory excludes EU, UK and South Korea; HF metadata ungated. | **Disabled stub only**; API/notebook rejects it before downloads. |

“Eşik” values are project admission policies, not measured peaks. Sequential GPU
stages use the largest stage budget, not the sum. Steps mainly increase time;
resolution and fit capacity can increase memory. The policy does not promise
that fewer steps make a large model fit. The comparison describes intended
tradeoffs, not a local quality ranking. BiRefNet's high-resolution segmentation
design makes it worth testing for fine edges; neither remover guarantees correct
hair, holes or transparent objects.

## GPU profiles and settings

| GPU preset | Default combination | Image / sparse / SLAT steps | Mesh views / iterations / cap | Minimum policy |
| --- | --- | --- | --- | --- |
| L4 24 GB | SDXL + U2Net + native TRELLIS | 25 / 12 / 12 | 24 / 1,500 / 50,000; inactive on native route | 20 GiB |
| A100 80 GB | FLUX.1-dev + BiRefNet + native TRELLIS | 28 / 20 / 20 | 48 / 3,000 / 100,000; inactive on native route | 36 GiB |
| H100 80 GB | Qwen-Image + BiRefNet + TRELLIS.2 | 40 / 24 / 24 | 64 / 4,000 / 150,000 | 60 GiB |
| RTX PRO 6000 96 GB | Qwen-Image + BiRefNet + TRELLIS.2 | 40 / 24 / 24 | 72 / 5,000 / 200,000 | 60 GiB |

Profiles set combinations; they do not provision or guarantee GPU availability
in Colab. Explicit overrides win over defaults. Legacy `baseline`/`quality`/
`ultra` presets remain compatible: `quality` uses 20 sparse/SLAT steps;
`ultra` uses 25 and at least 38 GiB geometry admission. Image resolutions are
512, 768 and 1024 square pixels; a higher image size does not change native
TRELLIS's Gaussian decoder resolution.

`seed` controls the reference image; `trellis_seed` independently controls 3D.
Sparse-structure steps/CFG affect coarse shape; SLAT steps/CFG affect finer
geometry/appearance. Stronger CFG can follow conditioning more closely but also
distort results. TRELLIS.2 translates the controls to its `guidance_strength`
argument, with separate shape/texture stages. Mesh views improve directional
coverage, iterations control fitting effort and `mesh_splat_cap` bounds
trainable splats. `target_splat_count` is a separate final export cap, not a
model-inference-memory control.

## Environment isolation and CUDA requirements

The Colab kernel retains Drive/display duties. uv `0.8.22` creates private
Python `3.11.11` environments in the run's Colab work directory. Notebook
generation performs no local model download or package installation.

| Environment | Selected dependencies | Installation policy |
| --- | --- | --- |
| Image + background | torch `2.7.1+cu128`, torchvision `0.22.1+cu128`, diffusers `0.37.0`, transformers `4.57.6`, accelerate `1.12.0`, huggingface-hub `0.36.2` | Private environment; direct dependency pins, binary wheels. Model CPU offload releases GPU memory between components. |
| Native TRELLIS | torch `2.4.0+cu121`, torchvision `0.19.0+cu121`, xformers `0.0.27.post2`, spconv-cu120 `2.3.6`, cumm-cu120 `0.4.11` | Existing complete lock; prebuilt wheels only. Explicit import patches omit unused mesh/render modules. |
| TRELLIS.2 + fitting | Modern CUDA 12.8 torch, xformers, pinned TRELLIS.2/CuMesh/FlexGEMM/nvdiffrast source, gsplat `1.5.3` | Separate environment; **CUDA source compilation required**. This route is not covered by the native route's wheels-only guarantee. |

The image stack requires NVIDIA driver 570+ by policy. RTX PRO 6000 Blackwell
cannot use the old native TRELLIS torch 2.4/CUDA 12.1 extensions: that combination
is rejected before downloads and its preset selects the modern mesh route.
Large VRAM alone does not establish GPU architecture compatibility. The mesh
route also needs a compatible CUDA toolkit/compiler; native imports and small
CUDA kernels are checked before expensive model inference.

Current image policies reserve 35/60/60/40/90 GiB free disk and 12/40/40/20/56
GiB available system RAM for SDXL/dev/schnell/klein/Qwen respectively. Mesh setup
adds 40 GiB disk. These estimates can reject a runtime that might work with
different offloading. Models are never silently substituted or quantized.

Upstream TRELLIS.2 documents torch 2.6/CUDA 12.4 and A100/H100 testing. The
recipe's CUDA 12.8 adaptation needs a clean Colab test; it is not claimed as
upstream-tested Blackwell support. Modern environment direct pins are not full
transitive locks; resolved packages are recorded in the manifest.
[Official setup](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/setup.sh),
[PyTorch versions](https://pytorch.org/get-started/previous-versions/),
[gsplat 1.5.3](https://github.com/nerfstudio-project/gsplat/tree/v1.5.3).

## Immutable model revisions

HF SHAs were read from `https://huggingface.co/api/models/<model-id>`. Selected
model IDs, revisions and licenses, including the image encoder and shared sparse
decoder, are recorded in `manifest.json`.

| Artifact | Verified revision / evidence |
| --- | --- |
| SDXL base 1.0 | [`462165984030d82259a11f4367a4eed129e94a7b`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b) |
| FLUX.1-dev | [`3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21) |
| FLUX.1-schnell | [`741f7c3ce8b383c54771c7003378a50191e9efe9`](https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/741f7c3ce8b383c54771c7003378a50191e9efe9) |
| FLUX.2-klein-4B | [`e7b7dc27f91deacad38e78976d1f2b499d76a294`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294) |
| Qwen-Image | [`75e0b4be04f60ec59a75f475837eced720f823b6`](https://huggingface.co/Qwen/Qwen-Image/tree/75e0b4be04f60ec59a75f475837eced720f823b6) |
| BiRefNet weights + remote implementation | [`e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`](https://huggingface.co/ZhengPeng7/BiRefNet/tree/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4) |
| TRELLIS-image-large, including TRELLIS.2's sparse decoder | [`25e0d31ffbebe4b5a97464dd851910efc3002d96`](https://huggingface.co/microsoft/TRELLIS-image-large/tree/25e0d31ffbebe4b5a97464dd851910efc3002d96) |
| TRELLIS.2-4B | [`af44b45f2e35a493886929c6d786e563ec68364d`](https://huggingface.co/microsoft/TRELLIS.2-4B/tree/af44b45f2e35a493886929c6d786e563ec68364d) |
| DINOv3 ViT-L/16 | [`ea8dc2863c51be0a264bab82070e3e8836b02d51`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/tree/ea8dc2863c51be0a264bab82070e3e8836b02d51); [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license) |
| DINOv2 with four register tokens | Code [`7764ea0f912e53c92e82eb78a2a1631e92725fc8`](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8), `dinov2_vitl14_reg`; [fixed checkpoint URL](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth), downloaded SHA256 recorded. |
| U2Net ONNX | `md5:60024c5c889badc19c04ad937298a77b` from [pinned rembg loader](https://github.com/danielgatis/rembg/blob/v2.0.59/rembg/sessions/u2net.py); downloaded SHA256 recorded. Independent weight license unverified. |
| Hunyuan3D-2.1, deferred | [`0b94677654c57bb9a6b6845cd7b704ccf551d327`](https://huggingface.co/tencent/Hunyuan3D-2.1/tree/0b94677654c57bb9a6b6845cd7b704ccf551d327); [community license](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/0b94677654c57bb9a6b6845cd7b704ccf551d327/LICENSE) |
| Unintegrated FLUX.2-klein-9B / FLUX.2-dev | `92196c8e11f7b6cf2b7493e037d8c5345c559216` / `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c` |

Code pins verified through official GitHub APIs:

| Source | Commit |
| --- | --- |
| [TRELLIS](https://github.com/microsoft/TRELLIS/tree/442aa1e1afb9014e80681d3bf604e8d728a86ee7) | `442aa1e1afb9014e80681d3bf604e8d728a86ee7` |
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2/tree/75fbf0183001ed9876c8dbb35de6b68552ee08bd) | `75fbf0183001ed9876c8dbb35de6b68552ee08bd` |
| [CuMesh](https://github.com/JeffreyXiang/CuMesh/tree/12289e1062f0603f2f0d0771b02e1395d247f26f) | `12289e1062f0603f2f0d0771b02e1395d247f26f` |
| [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM/tree/6dd94a859c26ee8246888502eada3dd8ad85532e) | `6dd94a859c26ee8246888502eada3dd8ad85532e` |
| [nvdiffrast v0.4.0](https://github.com/NVlabs/nvdiffrast/tree/253ac4fcea7de5f396371124af597e6cc957bfae) | `253ac4fcea7de5f396371124af597e6cc957bfae` |
| [utils3d](https://github.com/EasternJournalist/utils3d/tree/9a4eb15e4021b67b12c460c7057d642626897ec8) | `9a4eb15e4021b67b12c460c7057d642626897ec8` |
| [gsplat v1.5.3](https://github.com/nerfstudio-project/gsplat/tree/937e29912570c372bed6747a5c9bf85fed877bae) | `937e29912570c372bed6747a5c9bf85fed877bae` |

## Reconstruction and export limits

Native TRELLIS fetches only `pipeline.json` and four model pairs under `ckpts/`:
`ss_dec_conv3d_16l8_fp16`, `ss_flow_img_dit_L_16l8_fp16`,
`slat_dec_gs_swin8_B_64l8gs32_fp16` and `slat_flow_img_dit_L_64l8p2_fp16`.
A local config requests `formats=["gaussian"]`, without modifying shared Hub
snapshots or downloading unused mesh/radiance-field models.

TRELLIS.2 uses its fixed `512` pipeline with only the 512 model checkpoints and
low-VRAM loading. The generated mesh is simplified to at most 500,000 faces;
vertex base color/alpha are saved in `mesh.npz`. Golden-angle sphere cameras
render 512px views onto a white background and save `mesh_views.npz`. Surface-area
sampling initializes a fixed population of splats; Adam optimizes means,
quaternions, scales, opacity and RGB against RGB/alpha L1 loss with gsplat
`1.5.3`. There is no densification. This small embedded trainer does not reproduce every feature
or the claimed quality of gsplat's `simple_trainer`. Unlit base-color views lose
metallic, roughness, environment lighting and full PBR behavior in SH0 fitting.
The conversion is lossy; extra splats cannot recover absent mesh details.
[TRELLIS.2 API](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/pipelines/trellis2_image_to_3d.py),
[material sampling](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/representations/mesh/base.py),
[gsplat API](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py).

The fitting library uses [Apache 2.0](https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/LICENSE).
CuMesh and FlexGEMM declare MIT; utils3d declares MIT. nvdiffrast `0.4.0` uses the
[NVIDIA Source Code License (1-Way Commercial)](https://github.com/NVlabs/nvdiffrast/blob/v0.4.0/LICENSE.txt),
so the entire mesh stack must not be described as MIT-only. The route does not
install flash-attn or nvdiffrec.

The mesh recipe suppresses upstream BRIA RMBG-2.0 loading and consumes the
selected cutout. DINOv3 and the original TRELLIS sparse decoder remain required
and appear separately in the manifest. `HF_TOKEN` comes only from Colab Secrets
or the process environment; it is never serialized in notebook settings or the
manifest.

Both exporters store opacity logits, natural-log scales and normalized wxyz
quaternions, SH0 color and explicitly zero-padded higher SH coefficients.
Coordinates use a centered, longest-side-one, right-handed +Y-up frame. Means
and covariance rotations share the transform; units are not metres and do not
automatically align the asset to a captured scene. `turntable.gif` is an
approximate CPU preview, not a Spark rendering benchmark.

## Remaining empirical checks

A clean Colab run is needed for each image/route/GPU combination, particularly
TRELLIS.2 native compilation and Blackwell. Local tests cover safe literals,
enums, notebook AST/nbformat, stage routing, export and UI behavior. They do not
measure install time, quality, peak memory or real-inference Drive recovery.
Access approval, networking, disk, system RAM, native compiler compatibility and
the modern environments' unpinned transitive dependencies remain runtime risks.

Hunyuan3D is deferred because shape/texture setup, custom rasterization and
texture baking need a separate implementation and Colab validation. The
disabled entry is not a runnable alternative. Its verified metadata distinguishes
a real model from an unverified future option.
[Official Hunyuan3D requirements](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1#usage).

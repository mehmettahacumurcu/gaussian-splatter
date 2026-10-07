# Text → target splat: model and Colab research

Checked **7 October 2026** against upstream repositories, model cards, release
announcements and package metadata. This records the engineering choice for the
`text_to_splat` notebook. It is not a measured comparison of generated assets:
models were not downloaded or executed on the local 8 GB GPU.

## Choice and alternatives

Use **SDXL base 1.0 → background removal → TRELLIS-image-large → its native
Gaussian decoder**. Generate one isolated object, retain its reference image,
release the image pipeline before loading TRELLIS, and export activated Gaussian
attributes into standard 3DGS PLY. This is the chosen balance of quality,
installation complexity and L4/A100 feasibility, not a claim that SDXL has the
best image quality of every 2026 model.

| Route | Evidence and decision |
| --- | --- |
| TRELLIS-text-large / text-xlarge | Direct text-conditioned 1.1B / 2.0B models exist and can decode Gaussians. Microsoft explicitly recommends its image-conditioned models for better performance; direct text would save the image stage but forgo that recommendation. [Official TRELLIS repository](https://github.com/microsoft/TRELLIS), [text model card](https://huggingface.co/microsoft/TRELLIS-text-large). |
| SDXL → TRELLIS-image-large | SDXL base can operate without the refiner, supports negative prompts and mature FP16 Diffusers inference. Keeping the generated reference image also makes an unsuitable silhouette visible. TRELLIS has a trained Gaussian decoder and declares a 16 GB minimum, with A100/A6000 tested upstream. We target L4 24 GB or A100, using sequential stages. [SDXL model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [TRELLIS requirements](https://github.com/microsoft/TRELLIS#-installation). |
| FLUX → TRELLIS-image-large | FLUX.1-schnell is 12B and Apache 2.0; FLUX.1-dev uses its own non-commercial model license. Both currently require accepting Hugging Face access conditions. The larger image stage adds download, memory/offload and access friction, so it is not the default first Colab notebook. No local head-to-head quality result is asserted. [Schnell card](https://huggingface.co/black-forest-labs/FLUX.1-schnell), [dev card](https://huggingface.co/black-forest-labs/FLUX.1-dev). |
| FLUX.2 klein → TRELLIS-image-large | A credible newer alternative: klein 4B is Apache 2.0 and its model card reports about 13 GB VRAM, so it can fit an L4. However, the reference stack pins torch 2.8.0 / transformers 4.56.1, whereas this TRELLIS wheel set targets torch 2.4.0. It would need another isolated stack or separately validated newer native wheels. SDXL keeps this first notebook on one resolved environment with established negative-prompt support. This is an installation-complexity choice, not evidence that SDXL wins image quality. [Klein 4B card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/README.md), [reference dependencies](https://github.com/black-forest-labs/flux2/blob/main/pyproject.toml). |
| TRELLIS.2 | Its released inference API returns an O-Voxel/PBR mesh and exports GLB. The inspected release does not expose the original TRELLIS Gaussian decoder. It requires at least 24 GB and CUDA extension installation; mesh-to-splat fitting would introduce a separate pipeline. Despite broad family-level marketing language about Gaussian representations, the released code is the deciding source. [Official TRELLIS.2 installation and example](https://github.com/microsoft/TRELLIS.2). |
| GaussianAnything | A real image/text-to-surfel-Gaussian alternative, but its documented setup compiles PyTorch3D, diff-surfel-rasterization and simple-knn; the demo was tested on V100 32 GiB. Its surfel output and two-stage setup add compatibility work. [Official release](https://github.com/NIRVANALAN/GaussianAnything). |
| GaussianGrow, CVPR 2026 | The released route takes input geometry and text through four stages, with several local CUDA extensions, Hunyuan3D texture generation and Gaussian refinement. Its default profile recommends at least 48 GB; it is not the lightweight prompt-only Colab route requested here. [Official release and hardware notes](https://github.com/weiqi-zhang/GaussianGrow). |

## Current Colab and binary dependency strategy

Google announced the Python 3.13 transition on 18 August 2026. Its past-runtime
table lists 2026.07 as Python 3.12.13 / PyTorch 2.11.0. Depending on Colab's
current base Python or torch therefore cannot reproduce TRELLIS's older native
extensions. Keep Drive mounting/display in the notebook kernel and execute model
stages in a private, pinned Python 3.11 environment. No kernel restart or system
Python replacement is required. [Google's upgrade announcement](https://github.com/googlecolab/colabtools/issues/6081),
[official runtime table](https://research.google.com/colaboratory/runtime-version-faq.html).

The selected native stack is:

| Component | Pin / source |
| --- | --- |
| Python | 3.11.11, private uv-managed environment |
| PyTorch / torchvision | `2.4.0+cu121` / `0.19.0+cu121`, [official previous-version instructions](https://pytorch.org/get-started/previous-versions/#v240) |
| xformers | `0.0.27.post2`, cp311 manylinux2014 wheel in the [official CUDA 12.1 index](https://download.pytorch.org/whl/cu121/xformers/) |
| spconv / cumm | `spconv-cu120==2.3.6`, `cumm-cu120==0.4.11`; [spconv wheel files](https://pypi.org/project/spconv-cu120/2.3.6/#files), [cumm files](https://pypi.org/project/cumm-cu120/0.4.11/#files) |
| Diffusers stack | `diffusers==0.30.3`, `transformers==4.44.2`, `accelerate==0.34.2`, `huggingface-hub==0.25.2`, `safetensors==0.4.5` |
| Numerical / image stack | `numpy==1.26.4`, `pillow==10.4.0`, `rembg==2.0.59`, `onnxruntime==1.19.2`, `opencv-python-headless==4.10.0.84`, `scipy==1.14.1`, `scikit-image==0.24.0`, `numba==0.60.0`, `pymatting==1.1.12` |
| Small inference dependencies | `plyfile==1.1`, `easydict==1.13`, `tqdm==4.66.5`; spconv's Python dependencies include `pccm==0.4.16`, `ccimport==0.4.4`, `pybind11==2.13.6`, `fire==0.7.1` |

Package metadata was resolved successfully for Linux x86-64, glibc 2.28,
Python 3.11 with `uv pip compile --no-build`. This checks dependency resolution
and binary availability, not CUDA execution. Package versions and upstream
revisions must still be recorded in each run's manifest. Native extensions use
wheels only; a missing wheel should fail instead of starting a long build.

Set `ATTN_BACKEND=xformers` and `SPCONV_ALGO=native`. Sparse TRELLIS attention does
not provide the dense module's SDPA fallback, so dropping both xformers and
flash-attn would break inference. The spconv project documents compatibility
between different CUDA minor versions and includes architecture 80 (A100) and
89 (L4) in its CUDA 11.8+ prebuilt matrix. A current Colab driver can run the
private CUDA 12.1 torch runtime without installing a system CUDA toolkit.
[spconv installation notes](https://github.com/traveller59/spconv#prebuilt),
[pinned sparse attention implementation](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/sparse/attention/full_attn.py).

Kaolin, nvdiffrast, diff-octree-rasterization, mip-splatting and flash-attn are
unnecessary for this Gaussian-only generation/export path. Small explicit patches
remove eager imports of renderers, mesh/radiance-field decoders, the direct text
pipeline (Open3D), and `utils3d` from Gaussian module import time. They do not
change model arithmetic. The Gaussian decoder uses windowed `swin` attention;
`vox2seq`, imported lazily for serialized attention, is not called. Render the
turntable with the notebook's NumPy/Pillow approximation so preview generation
does not add a CUDA renderer. For reference, the official gsplat 2.4/12.4 binary
index currently lists cp310, not cp311, making it unsuitable for this exact env.
[pinned Gaussian decoder configuration](https://huggingface.co/microsoft/TRELLIS-image-large/blob/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.json),
[gsplat binary index](https://docs.gsplat.studio/whl/pt24cu124/gsplat/).

## Model revisions and minimal downloads

| Artifact | Verified revision |
| --- | --- |
| Microsoft TRELLIS code | `442aa1e1afb9014e80681d3bf604e8d728a86ee7` |
| `microsoft/TRELLIS-image-large` | `25e0d31ffbebe4b5a97464dd851910efc3002d96` |
| `stabilityai/stable-diffusion-xl-base-1.0` | `462165984030d82259a11f4367a4eed129e94a7b` |
| Meta DINOv2 code | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| DINO model | `dinov2_vitl14_reg`; [official fixed weight URL](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth) |
| Background remover | rembg `2.0.59`, U2Net; its loader verifies `md5:60024c5c889badc19c04ad937298a77b` for [the release artifact](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx) |

DINOv2 must be the version **with four register tokens**, not the ordinary
`facebook/dinov2-large` model. Pinning the source commit prevents a moving
`torch.hub` branch; the official checkpoint URL has no upstream SHA256 pin, so
record the actual downloaded checkpoint hash for auditability. U2Net's release
checksum comes from the [pinned rembg loader](https://github.com/danielgatis/rembg/blob/v2.0.59/rembg/sessions/u2net.py).

Download TRELLIS `pipeline.json` and the `.json` / `.safetensors` pair for only
these four prefixes, preserving their paths:

- `ckpts/ss_dec_conv3d_16l8_fp16`
- `ckpts/ss_flow_img_dit_L_16l8_fp16`
- `ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16`
- `ckpts/slat_flow_img_dit_L_64l8p2_fp16`

Use a local pipeline config containing only those model keys and request
`formats=["gaussian"]`. Do not edit the shared Hub snapshot in place or load mesh
and radiance-field weights and discard them afterward.
[Pinned pipeline config](https://huggingface.co/microsoft/TRELLIS-image-large/blob/25e0d31ffbebe4b5a97464dd851910efc3002d96/pipeline.json).

## Export and license implications

TRELLIS Gaussian coordinates are Z-up in a normalized object space. The native
PLY method applies a default rotation that maps positive Z to negative Y. Our
exporter must state its own convention explicitly and rotate both means and
quaternions together. COLMAP scenes have arbitrary world orientation and scale;
normalizing an object does not align it to a room automatically. The origin,
up-axis and scale factor belong in the manifest.
[Pinned camera construction](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/utils/render_utils.py),
[Gaussian activation and PLY code](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/representations/gaussian/gaussian_model.py).

Export `get_xyz`, `get_scaling`, `get_rotation`, `get_opacity` and DC spherical
harmonics. The raw model fields contain biases and softplus scales and cannot be
copied directly into conventional PLY. Standard PLY stores opacity logits,
logarithmic scales and wxyz quaternions. This decoder predicts SH degree zero;
zero-valued higher SH coefficients can be added for readers requiring SH3.
The turntable is an approximate SH0 preview, not a calibrated rendering-quality
benchmark.

TRELLIS code and checkpoints use MIT. SDXL uses CreativeML Open RAIL++-M, which
permits the intended non-commercial experiments subject to its use restrictions;
the combined route should not be labeled wholly MIT. DINOv2 is Apache 2.0,
rembg is MIT, and U2Net is Apache 2.0. No Kaolin or INRIA rasterizer dependency is
included in this route. [TRELLIS license](https://github.com/microsoft/TRELLIS/blob/main/LICENSE),
[SDXL license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md),
[DINOv2 license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE),
[rembg license](https://github.com/danielgatis/rembg/blob/main/LICENSE.txt),
[U2Net license](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE).

## Remaining empirical checks

A clean L4/A100 Colab run is still required to establish actual install time,
peak CPU/GPU memory, inference duration, visual quality and full Drive resume
behavior. Wheel metadata resolution and local helper tests do not establish
those results. Dense prompts can exceed VRAM; small details, glass, wheels and
unseen sides may be imperfect. Low-RAM runtimes can fail while loading model
weights even when GPU memory is adequate. Downloads remain several GB, and
access/network/rate-limit changes may affect future runs. Reducing a generated
splat count is lossy and does not reduce model inference memory.

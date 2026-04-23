# 4DGS Studio

Tek bir videodan 4D Gaussian Splatting sahnesi üreten masaüstü uygulamasının backend prototipi (Faz 1-6).

> Hedef: Çektiğin bir videoyu serbest kameralı 4D deneyime dönüştür. (Kendi 4DV.ai versiyonun.)

## Şu an ne çalışır

| Faz | Modül | Durum |
|----|------|-------|
| 1 | Ortam (`requirements.txt`, `environment.yml`) | hazır |
| 2a | Frame çıkarma — `extract_frames.py` | hazır (ffmpeg) |
| 2b | COLMAP SfM — `run_colmap.py` | hazır |
| 2c | COLMAP parser — `parse_colmap.py` | hazır |
| 3a | Metric3D-v2 derinlik | hazır (torch.hub) |
| 3b | CoTracker tracking | hazır (torch.hub) |
| 3c | Dinamik maske | hazır (Farneback fallback; SAM2 stub) |
| 4a | `GaussianModel` | hazır |
| 4b | `DeformationField` (HexPlane + MLP) | hazır |
| 4c | Kovaryans + EWA projeksiyon | hazır |
| 5  | gsplat renderer + Trainer + ADC | hazır |
| 6  | `.ply` export (3DGS uyumlu) | hazır |
| 7  | FastAPI backend | sonraki sprint |
| 8  | Tauri/React frontend | sonraki sprint |

## Kurulum

> **Windows kullanıcıları:** Kurulum öncesi [`docs/WINDOWS_SETUP.md`](docs/WINDOWS_SETUP.md)'i mutlaka oku. Visual Studio 2022 Build Tools, CUDA 12.6 toolkit ve gsplat için bir tek satırlık patch gerekli.

```bash
# 1) Conda ortamı (COLMAP + ffmpeg dahil)
conda env create -f environment.yml
conda activate gs4d

# 2) PyTorch — conda dışında, doğrulanmış versiyonla
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3) Geri kalan Python paketleri
pip install -r requirements.txt
```

> Neden `cu124` ama `nvcc 12.6`? PyTorch wheel'leri forward-compatible — cu124 binary'si cu126 runtime'ında çalışır. torch 2.11+ pre-release'leri gsplat 1.5.3 ile UYUMSUZ.

### Doğrulama

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import gsplat; print('gsplat OK')"   # Windows'ta ilk import ~2-3 dk (JIT compile)
colmap -h
```

## Hızlı başlangıç

### Smoke test (önce bunu çalıştır — 2-3 dk)

Küçük parametrelerle pipeline'ın baştan sona çalıştığını doğrula:

```bash
# Test videonu kopyala
mkdir -p data/test_scene
cp /yol/to/video.mp4 data/test_scene/video.mp4

# Smoke test: 500 iter, 480x270, 10 timestamp
python -m backend.pipeline data/test_scene/video.mp4 --scene test_scene --smoke-test
```

Windows'ta stdout redirect yaparsan (örn. `> build_log.txt`):

```cmd
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
python -m backend.pipeline "data\test_scene\video.mp4" --scene test_scene --smoke-test
```

### Full run (30-60 dk, GPU tam yüklü)

Gerçek kalite için:

```bash
python -m backend.pipeline data/test_scene/video.mp4 --scene test_scene
```

- 30k iterasyon, 640×360, 60 timestamp
- Loss ~0.02-0.04, PSNR ~28-32 beklenir
- N ~100k-500k gaussian (sahne karmaşıklığına bağlı)

Çıktılar:
- `data/test_scene/frames/`     — PNG kareler
- `data/test_scene/colmap/`     — COLMAP sparse rekonstrüksiyon
- `data/test_scene/output/ckpt/`— eğitim checkpoint'leri
- `data/test_scene/output/ply/` — `frame_0000.ply` … `frame_0059.ply`

`.ply` dosyaları [antimatter15/splat](https://antimatter15.com/splat/) gibi açık kaynak viewer'larda görüntülenebilir.

## Faz faz çalıştır

```bash
# Sadece frame çıkar
python -m backend.preprocess.extract_frames data/test_scene/video.mp4 data/test_scene/frames --fps 10

# Sadece COLMAP
python -m backend.preprocess.run_colmap data/test_scene/frames data/test_scene/colmap

# COLMAP çıktısını incele
python -m backend.preprocess.parse_colmap data/test_scene/colmap

# Sadece export (önceden eğitilmiş checkpoint'ten)
python -m backend.export.to_splat data/test_scene/output/ckpt/ckpt_final.pt data/test_scene/output/ply
```

## Konfigürasyon

`backend/config.py` içinde tüm parametreler bulunur. İki preset:

- `default_config()` — RTX 3060 Ti 8GB (640×360, 30k iter, ~30-60 dk)
- `cloud_config()`   — RTX 4090 24GB (1920×1080, 60k iter)

Pipeline'ı cloud config ile çalıştırmak için: `--cloud` bayrağı.

## Sonraki adımlar

- [ ] Faz 7: FastAPI backend (`api.py`, `/process`, `/status`, `/download`)
- [ ] Faz 8: Tauri + React frontend (Viewer4D, TimelineSlider)
- [ ] SAM2 entegrasyonu (gerçek dinamik maske, Farneback yerine)
- [ ] Depth/track consistency loss → deformation training'i regularize et
- [ ] Multi-resolution training (coarse-to-fine)

## Klasör yapısı

```
4dgs-studio/
├── backend/
│   ├── preprocess/   # Faz 2-3
│   │   ├── extract_frames.py
│   │   ├── run_colmap.py
│   │   ├── parse_colmap.py
│   │   ├── depth_estimate.py
│   │   ├── point_tracking.py
│   │   └── dynamic_mask.py
│   ├── model/        # Faz 4-5
│   │   ├── gaussian_model.py
│   │   ├── deformation.py
│   │   ├── covariance.py
│   │   ├── renderer.py
│   │   ├── density_control.py
│   │   └── trainer.py
│   ├── export/       # Faz 6
│   │   └── to_splat.py
│   ├── config.py
│   └── pipeline.py
├── data/             # video, frames, colmap, output (gitignore)
├── requirements.txt
├── environment.yml
└── README.md
```

## Olası sorunlar

**Windows'a özel sorunlar** için [`docs/WINDOWS_SETUP.md`](docs/WINDOWS_SETUP.md) — özellikle gsplat JIT compile, MSVC flag'leri, Unicode/encoding.

| Belirti | Olası neden | Çözüm |
|---------|-------------|-------|
| `ImportError: cannot import name 'csrc' from 'gsplat'` | gsplat CUDA ext compile fail | [WINDOWS_SETUP §3](docs/WINDOWS_SETUP.md#3-gsplat-cuda-extension--jit-compile) |
| `UnicodeEncodeError: 'charmap'` | Windows cp1254 + redirect | `set PYTHONIOENCODING=utf-8` |
| COLMAP `mapper` başarısız / 2 parça | Az feature / az hareket | `--exhaustive` matcher dene; FPS düşür |
| 8GB VRAM yetmez | Resolution çok yüksek | `config.train.image_resolution = (480, 270)` |
| Loss düşmüyor | LR çok yüksek | `lr_means` ve `lr_deform` yarıya indir |
| `metric3d` indirme hatası | torch.hub cache | `~/.cache/torch/hub/` sil, tekrar dene |

## Test edilmiş konfigürasyon

- **OS:** Windows 11
- **GPU:** RTX 3060 Ti 8GB
- **Python:** 3.10 (conda env `gs4d`)
- **torch:** 2.5.1+cu124
- **gsplat:** 1.5.3 (Windows'ta JIT compile, tek satırlık patch uygulanmış — bkz. WINDOWS_SETUP §3.2)
- **CUDA toolkit:** 12.6, **MSVC:** 19.38 (VS 2022 Build Tools)
- **Smoke test süresi:** ~30-60 sn (500 iter, 480×270)
- **Full run süresi:** ~30-60 dk (30k iter, 640×360)

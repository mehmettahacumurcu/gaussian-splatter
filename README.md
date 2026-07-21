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
| 7  | FastAPI backend (`api.py`, `/process` `/status` `/download`) | hazır |
| 8a | Tauri + React 4D viewer (`frontend/`) | hazır |
| 8b | Tam UI (job list, upload form, live progress) | sonraki sprint |

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

## A100 learned-training diagnosis

The isolated [training ablation notebook](docs/TRAINING_ABLATION_NOTEBOOK.md)
compares the legacy control against dense seeds, masks, depth supervision,
adaptive density, and their full learned combination. It stages verified Drive
evidence to local Colab disk once, runs deterministic short experiments, and
publishes reports to `<input>_training_ablation`. It does not alter the legacy
notebook or publish a production PLY.

## HTTP API (Faz 7)

Pipeline'ı CLI'dan çalıştırmak yerine HTTP üzerinden kullanabilirsin. Uygulama içi (Tauri frontend, web browser) ya da dışarıdan (curl, httpie) erişim için.

### Server'ı başlat

```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Geliştirme sırasında auto-reload:

```bash
uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

Swagger UI interaktif dokümantasyon: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Endpoint'ler

| Method | Path | Amaç |
|--------|------|------|
| `GET` | `/` | Servis sağlığı + GPU durumu |
| `POST` | `/process` | Video yükle + pipeline job'ı başlat |
| `GET` | `/status/{job_id}` | Job ilerlemesi (faz, progress 0-1, mesaj) |
| `GET` | `/jobs` | Tüm job'ların listesi |
| `GET` | `/download/{job_id}` | Bitmiş job'un .ply'larını zip olarak indir |

### Hızlı test — curl

```bash
# Health
curl http://127.0.0.1:8000/

# Job başlat (smoke test)
curl -F "video=@data/test_scene/video.mp4" \
     -F "scene=test_scene" \
     -F "smoke_test=true" \
     http://127.0.0.1:8000/process
# → {"job_id": "abc-123...", "status": "queued", "status_url": "/status/abc-123..."}

# İlerlemeyi poll et
curl http://127.0.0.1:8000/status/abc-123...
# → {"status": "running", "phase": {"name": "training", "progress": 0.42, ...}, ...}

# Sonucu indir
curl -o result.zip http://127.0.0.1:8000/download/abc-123...
```

### Job yaşam döngüsü

```
queued → running → completed   (başarılı)
                 ↘ failed       (exception — error alanında traceback)
```

Tek GPU var, bu yüzden `ThreadPoolExecutor(max_workers=1)` — eşzamanlı gelen iki request sırayla işlenir. İkincisi `queued` statüsünde bekler.

**Not:** Job state in-memory. Server restart'ında biten/bekleyen jobs bilgisi kaybolur. İleride SQLite persist eklenir.

## Masaüstü 4D Viewer (Faz 8a)

Tauri + React tabanlı native pencere. Eğittiğin .ply'ları 4D zaman çizgisiyle inceler.

### Kurulum (tek seferlik)

Gerekli: Node.js 18+, Rust (rustup), Microsoft Edge WebView2 (Windows 10/11'de genelde kurulu).

```bash
cd frontend
npm install
```

### Hızlı başlangıç — tek tık

Proje kökündeki **`run.bat`**'e çift tıkla. İki pencere otomatik açılır:

- **4DGS Backend** penceresi: vcvars64 + conda env + UTF-8 + uvicorn
- **4DGS Frontend** penceresi: node_modules kontrolü + `npm run tauri dev`

`scripts/start-backend.bat` ve `scripts/start-frontend.bat` ayrı ayrı da çalışır (debug için).

### Manuel — iki terminal

**Terminal 1 (backend):**

```cmd
conda activate gs4d
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

**Terminal 2 (frontend, conda olmadan):**

```cmd
cd frontend
npm run tauri dev
```

İlk sefer Rust compile eder (~1-2 dk). Sonra native Windows penceresi açılır.

### Kullanım

1. Backend'de önce bir job tamamla (curl ile `POST /process`, ya da mevcut bir job_id kullan)
2. Viewer'da job_id'yi yapıştır → **Yükle**
3. Ya da **"Son tamamlanmış"** butonu → otomatik bulur
4. Splat yüklenince: mouse ile orbit, wheel ile zoom, right-click ile pan
5. Altta timeline: scrubbable slider + ▶ play/pause + hız (0.25x-4x) + loop toggle

### Production build

```cmd
cd frontend
npm run tauri build
```

`src-tauri/target/release/` altında `.exe` + MSI installer çıkar.

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

- [x] ~~Faz 7: FastAPI backend (`api.py`, `/process`, `/status`, `/download`)~~
- [x] ~~Faz 8a: Tauri + React 4D viewer~~
- [ ] Faz 8b: Tam UI — job list, video upload form, live progress
- [ ] Job state persistence (SQLite) — şu an in-memory
- [ ] SAM2 entegrasyonu (gerçek dinamik maske, Farneback yerine)
- [ ] Depth/track consistency loss → deformation training'i regularize et
- [ ] Multi-resolution training (coarse-to-fine)
- [ ] Eval metrikleri: PSNR/SSIM/LPIPS scriptleri

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
│   │   ├── renderer.py
│   │   ├── density_control.py
│   │   └── trainer.py
│   ├── export/       # Faz 6
│   │   └── to_splat.py
│   ├── config.py
│   ├── pipeline.py
│   ├── api.py          # Faz 7: FastAPI app (9 endpoint)
│   ├── api_models.py   # Faz 7: Pydantic şemaları
│   └── job_manager.py  # Faz 7: thread-safe job registry + executor
├── frontend/         # Faz 8a: Tauri + React 4D viewer
│   ├── src/
│   │   ├── App.tsx               # Ana uygulama
│   │   ├── App.css
│   │   ├── api.ts                # FastAPI HTTP client
│   │   └── components/
│   │       ├── SplatViewer.tsx   # three.js + GaussianSplats3D
│   │       └── TimelineSlider.tsx # scrubbable + play/pause
│   ├── src-tauri/    # Rust shell
│   ├── package.json
│   └── tsconfig.json
├── docs/
│   └── WINDOWS_SETUP.md
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

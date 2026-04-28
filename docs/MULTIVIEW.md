# Multi-view Pipeline (v5.0) — User Guide

> Branch: `feat/multiview`
> Status: **Sprint 1-2 done, Sprint 3-5 (full pipeline integration) pending**
>
> Bu dokümanda **MVP single-view-proxy** modunu açıklıyoruz — flame_steak gibi
> Neural 3D Video dataset'lerini bugün test edebilmek için. Tam multi-view
> N-camera supervision sonraki sprint'lerde gelecek.

---

## Multi-view Sahne Tanımı

Bizim format:

```
data/<scene>/
├── videos/                    ← multi-view giriş (Sprint 1.1)
│   ├── cam00.mp4
│   ├── cam01.mp4
│   └── ...
├── poses_bounds.npy           ← N3V calibration (LLFF format)
├── calibration.json           ← human-readable parsed (load_n3v üretir)
├── frames_multiview/          ← extract sonrası
│   ├── cam00/frame_0000.png
│   └── ...
└── output/                    ← training output (single 4DGS scene)
```

**Auto-detection:** `data/<scene>/videos/` klasörü varsa ve >=2 cam dosyası
varsa multi-view. Aksi halde mevcut single-view path.

---

## Sprint 1.2 + 2 Tamamlandı — Test Edebileceğin Adımlar

### Adım 1 — Neural 3D Video Dataset İndir

[github.com/facebookresearch/Neural_3D_Video Releases](https://github.com/facebookresearch/Neural_3D_Video/releases)

İndir:
- `flame_steak.zip` — 18 kamera × ~10 sn × 30 fps (ateş, dramatic)
- veya `cook_spinach.zip`, `coffee_martini.zip`, `sear_steak.zip`

Extract → `~/Downloads/flame_steak/{cam00.mp4, cam01.mp4, ..., cam17.mp4, poses_bounds.npy}`.

### Adım 2 — N3V Format → Bizim Layout

```bash
conda activate gs4d
cd C:\Users\TAHA\Desktop\gaussian-splatter\Gaussian Splatter\4dgs-studio

python scripts/load_n3v.py \
  --src C:\Users\TAHA\Downloads\flame_steak \
  --dst data\flame_steak
```

Beklenen çıktı:
```
✓ 18 cam mp4 kopyalandı
✓ poses_bounds.npy parse edildi
✓ calibration.json yazıldı (18 kamera, fx=..., near=..., far=...)
```

### Adım 3 — Multi-view Sahne Bilgilerini Doğrula

```bash
python scripts/run_multiview.py data/flame_steak --info
```

Beklenen:
```
Multi-view detected: True
Cameras: 18 → ['cam00', 'cam01', ..., 'cam17']
Has poses_bounds.npy: True
Has calibration.json: True
Scene centroid: [...]
Scene extent: ...
```

### Adım 4 — Frame Extract (Multi-camera Paralel)

```bash
python scripts/run_multiview.py data/flame_steak --extract --fps 20 --edge 1280
```

Her cam için `frames_multiview/cam00/frame_0000.png ...` üretilir. Cache check
var (mevcut frames'i atlar).

### Adım 5 — Single-view-Proxy ile Train (MVP)

Pipeline tam multi-view'a entegre olmadığı için, **bir kamerayı seç**, single-view
gibi train et. cam05 (orta, yan görünüm) genelde iyi tercih:

```bash
python scripts/run_multiview.py data/flame_steak --select-cam cam05
```

Bu:
- `data/flame_steak/videos/cam05.mp4` → `data/flame_steak/video.mp4` (symlink)
- `data/flame_steak/frames_multiview/cam05/` → `data/flame_steak/frames/` (symlink)

Sonra mevcut single-view pipeline çalışır:

```bash
curl -X POST http://127.0.0.1:8000/process ^
  -F "video=@data\flame_steak\video.mp4" ^
  -F "scene=flame_steak" ^
  -F "static_max=true" ^
  -F "iters=15000"
```

Veya frontend'den `flame_steak` scene yükle, **Static Max** preset + **iters=15000** override.

### Beklenen Süre

| Faz | Süre (15k iter) |
|---|---|
| Frame extract (1 cam, ~300 frame) | 2 dk |
| COLMAP exhaustive | 30-60 dk |
| MiDaS vit_large | 10-15 dk |
| CoTracker online | 5-10 dk |
| Mask | 3 dk |
| Init confidence subsample | 30 sn |
| Training 15k iter | 60-80 dk |
| Export 90 ts | 3 dk |
| **Total** | **~2.5-3.5 saat** |

---

## Tam Multi-view (Sprint 3+) Roadmap

Full N-camera supervision için yapılacaklar:

### Sprint 3 — Pipeline Multi-view Branching
`backend/pipeline.py`:
- `is_multiview_scene(scene)` ile detect
- Multi-view ise:
  - `extract_frames_multiview()` çağır (paralel)
  - `parse_n3v_calibration()` (poses_bounds varsa COLMAP atla)
  - veya custom multi-cam COLMAP exhaustive
  - `init_random_points_in_bbox()` (sparse cloud yok ise)
- Single-view ise eski path (mevcut)

### Sprint 4 — Trainer Multi-view Dataloader
`backend/model/trainer.py`:
- Per-iter random `(cam_idx, t_idx)` sampling
- Multi-cam image cache (RAM, all cams × all frames preloaded)
- Loss aggregation across views same time
- Optional: multi-view consistency loss

### Sprint 5 — API + Frontend
- `api.py`: multi-view auto-detect + multiview override fields
- `JobSubmitPanel.tsx`: pipeline mode toggle (single / multi)
- Multi-file upload veya Tauri folder picker (task #36)

---

## Single-view Korunuyor

Mevcut tüm single-view sahneleri (`banana_demo`, `banana_static_max`, vb.)
**aynı şekilde çalışıyor**. v5.0 değişiklikleri **opt-in**:
- `data/<scene>/videos/` klasörü yoksa → single-view (eski davranış)
- `data/<scene>/videos/cam*.mp4` var ama henüz pipeline integration yoksa →
  warning log, single-view'e fall back

---

## Sources

- [Neural 3D Video (Meta, CVPR 2022)](https://github.com/facebookresearch/Neural_3D_Video)
- [4DV.ai](https://www.4dv.ai/) — production multi-view 4DGS reference
- [Spacetime Gaussians (OPPO, CVPR 2024)](https://oppo-us-research.github.io/SpacetimeGaussians-website/)
- [HustVL 4DGaussians](https://github.com/hustvl/4DGaussians) — bizim training mimarimizin temeli

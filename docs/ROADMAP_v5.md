# 4DGS Studio v5.0 — Multi-View Production Roadmap

**Created:** 2026-04-27
**Goal:** 4dv.ai-grade quality multi-view 4D Gaussian Splatting
**Approach:** Phased upgrades, low-risk first, high-impact prioritization

---

## Phase 0 — Tamamlanan Mimari Foundation (Mevcut Durum)

✅ Multi-time bootstrap COLMAP (configurable timestamps, default 5, premium 25)
✅ Hybrid init (COLMAP cloud + scene-region random fill)
✅ Premium COLMAP settings (OPENCV camera model + per-cam K + 8192 SIFT + BA refinement)
✅ MVS dense reconstruction option (sparse → dense 500k-2M points)
✅ Per-cam K rescale (raw video → loaded resolution)
✅ Static phase warmup (8k iter cap, deformation frozen)
✅ L1 sparsity on Fourier coefficients (implicit static-dynamic separation)
✅ Density controller multi-view fix (visibility-aware n_obs counting)
✅ Spatial smoothness disabled in multi-view (statik bölge motion bulaşması yok)
✅ num_timestamps subsampling (RAM 21GB → 4.2GB)
✅ Scene extent estimation from look-points (foreground-focused)
✅ Convention: LLFF → OpenCV (4DGaussians-standard row-permute + col-flip)

**Phase 0 sonu beklenen PSNR (premium overnight):** 28-32 (4dv.ai'nin %75'i)

---

## 🟢 Phase 1 — Low Risk + High Impact (HAFTA 1)

**Hedef:** Mimari risk almadan kalitede büyük sıçrama. Hepsi mevcut altyapı ile uyumlu.

### 1.1 Heavy Preprocessing Cache Infrastructure
- **Etki:** Workflow ⭐⭐⭐
- **Süre:** 1 gün
- **Risk:** LOW
- **Bağımlılık:** Yok
- **İş:**
  - `pipeline.py` her preprocessing adımına strict cache check
  - `data/<scene>/.cache_manifest.yaml` — preprocess settings hash
  - Force-rerun flag (`--force-preprocess`)
  - Skip-if-cached behavior her aşama için (frame extract, COLMAP, depth, mask, tracks)

### 1.2 Heavy Preprocess Script
- **Etki:** Workflow ⭐⭐⭐
- **Süre:** 1 gün
- **Risk:** LOW
- **Bağımlılık:** 1.1
- **İş:** `scripts/heavy_preprocess.py` — premium profile, scene için tek seferlik tam preprocess
  ```python
  python scripts/heavy_preprocess.py --scene flame_steak --profile premium
  ```
- Profile presets: `standard` / `high` / `premium`
- 12-saat overnight runs için ideal (3 sahne paralel veya seri)

### 1.3 Cache Hash + Manifest Validation
- **Etki:** Workflow ⭐⭐
- **Süre:** Yarım gün
- **Risk:** LOW
- **Bağımlılık:** 1.1
- **İş:** Settings'ten SHA256 hash, manifest dosyası, training runtime'da validate

### 1.4 Per-Cam Multi-View Depth Estimation
- **Etki:** Kalite ⭐⭐⭐⭐
- **Süre:** 1 gün
- **Risk:** LOW
- **Bağımlılık:** Yok (foundation models mevcut)
- **İş:**
  - `pipeline.py` multi-view branch'te foundation guard `not is_mv` → kaldır
  - Multi-view'da her cam için `estimate_depth(frames_multiview/cam00/, depth_multiview/cam00/)`
  - `trainer.py` multi-view branch'te per-cam depth load
  - Depth loss multi-view'da aktive

### 1.5 Per-Cam Multi-View Dynamic Mask
- **Etki:** Kalite ⭐⭐⭐
- **Süre:** 1 gün
- **Risk:** LOW
- **Bağımlılık:** Yok
- **İş:**
  - SAM2 + frame difference per cam
  - `masks_multiview/cam00/...` directory layout
  - Trainer'da per-cam mask load + `lambda_mask_motion` weighted recon

### 1.6 LPIPS Perceptual Loss
- **Etki:** Kalite ⭐⭐⭐
- **Süre:** Yarım gün
- **Risk:** LOW
- **Bağımlılık:** Yok
- **İş:** `pip install lpips`, trainer'a LPIPS loss ekle (lambda_lpips), config + API expose

### 1.7 Multi-View Consistency Loss
- **Etki:** Kalite ⭐⭐⭐
- **Süre:** 1 gün
- **Risk:** LOW (config'te zaten var, sadece implementasyon)
- **Bağımlılık:** Yok
- **İş:** Trainer iter loop'unda iki cam render, cross-view SSIM consistency loss

### 1.8 Optical Flow Loss (RAFT)
- **Etki:** Kalite ⭐⭐⭐⭐⭐
- **Süre:** 2 gün
- **Risk:** LOW-MEDIUM
- **Bağımlılık:** Yok
- **İş:**
  - RAFT pretrained model entegrasyonu (`backend/preprocess/optical_flow.py`)
  - Per-frame-pair flow computation, cache to `flow_multiview/cam00/...`
  - Trainer'a flow loss: `||rendered_flow - gt_flow||`
  - lambda_flow config + warmup

### 1.9 Densify Dynamics Multi-View Tuning
- **Etki:** Kalite ⭐⭐
- **Süre:** Yarım gün
- **Risk:** LOW
- **Bağımlılık:** 1.4-1.8 sonrası
- **İş:** Phase 1 değişiklikleri sonrası densify_grad_threshold + clone/split ratio fine-tune

**Phase 1 toplam:** ~7-8 gün
**Phase 1 sonu beklenen PSNR:** 32-34 (4dv.ai'nin %80-85'i)

---

## 🟡 Phase 2 — Medium Risk + High Impact (HAFTA 2-3)

**Hedef:** Mimari değişikliklerle architectural completeness.

### 2.1 Static/Dynamic Explicit Split ⭐ EN KRİTİK ⭐
- **Etki:** Kalite ⭐⭐⭐⭐⭐
- **Süre:** 4-5 gün
- **Risk:** MEDIUM (architectural change)
- **Bağımlılık:** 1.5 (mask tabanlı seleksiyon için)
- **İş:**
  - `GaussianModel`'a `is_static: torch.Tensor[bool]` flag
  - Trainer'da branch render: static gaussian'lar deformation BYPASS
  - Densify static/dynamic ayrı tutar (farklı thresholds olabilir)
  - Static/dynamic seleksiyon: motion mask + frame variance + opacity heuristics
  - Init: hepsi static, dynamic mask'ten seçim → bazıları dynamic flag'lenir

### 2.2 Multi-Resolution Training
- **Etki:** Kalite ⭐⭐⭐
- **Süre:** 2 gün
- **Risk:** LOW-MED
- **Bağımlılık:** Phase 1
- **İş:**
  - Schedule: 240p (10k iter) → 480p (20k iter) → 720p (20k iter)
  - Her resolution upgrade'de gaussian density artar
  - K rescale her transition'da

### 2.3 Camera Pose Refinement
- **Etki:** Kalite ⭐⭐
- **Süre:** 2 gün
- **Risk:** MEDIUM (gradient stability)
- **Bağımlılık:** Yok
- **İş:**
  - cam_K, cam_w2c trainable params (lr çok düşük: 1e-7)
  - Joint BA: gaussian + camera optimize
  - Stability tricks: gradient clipping, separate optimizer

### 2.4 Background Skybox / Environment Map
- **Etki:** Kalite ⭐⭐⭐
- **Süre:** 2-3 gün
- **Risk:** LOW-MED
- **Bağımlılık:** 2.1 sonrası
- **İş:**
  - Distance-based gaussian split: yakın → foreground, uzak → background
  - VEYA environment map (sphere mesh + texture)
  - Foreground gaussian'lar background'a dağılmaz

### 2.5 Better Deformation Field (HexPlane Multi-Resolution)
- **Etki:** Kalite ⭐⭐
- **Süre:** 3 gün
- **Risk:** MEDIUM
- **Bağımlılık:** Yok
- **İş:**
  - HexPlane multi-scale (4 resolution levels)
  - Hash grid encoding alternatif
  - Deformation MLP capacity artırma

**Phase 2 toplam:** ~14 gün
**Phase 2 sonu beklenen PSNR:** 34-36 (4dv.ai'nin %90-95'i)

---

## 🟠 Phase 3 — Polish + Production (HAFTA 4)

**Hedef:** Production readiness, file size, deployment.

### 3.1 Gaussian Compression
- **Etki:** File size ⭐⭐⭐ (200MB → 20-50MB)
- **Süre:** 3 gün
- **Risk:** LOW
- **İş:** Quantization (16-bit → 8-bit positions, 4-bit colors), Draco/MeshOpt codec

### 3.2 Adaptive Spherical Harmonics Degree
- **Etki:** Memory ⭐⭐
- **Süre:** 3 gün
- **Risk:** MEDIUM (gsplat değişiklik gerek)
- **İş:** Per-Gaussian SH degree (0-3), düşük detay = düşük degree

### 3.3 Test-Time Optimization
- **Etki:** Kalite ⭐⭐
- **Süre:** 2 gün
- **Risk:** LOW
- **İş:** Per-frame fine-tune option (render time +30 sec, quality +0.5 dB)

### 3.4 API + UX Improvements
- **Etki:** Workflow ⭐⭐
- **Süre:** 1-2 gün
- **Risk:** LOW
- **İş:**
  - Optional `video` parameter for multi-view scenes
  - `/process-scene` endpoint (no upload, scene_name only)
  - Multi-file upload for new multi-view scenes
  - Per-cam mask/depth API exposure

### 3.5 Camera Path Generation
- **Etki:** Demo ⭐⭐
- **Süre:** 2 gün
- **Risk:** LOW
- **İş:** Otomatik orbital camera path, demo video export

**Phase 3 toplam:** ~12 gün

---

## 🔴 Phase 4 — High Risk / Experimental (DEFER)

**Hedef:** SOTA research-level. Phase 1-3 sonrası gap kalırsa.

### 4.1 Anti-Aliasing (Mip-Splatting)
- **Süre:** 1-2 hafta
- **Risk:** **HIGH** — Custom CUDA kernel veya gsplat fork

### 4.2 Background NeRF (foreground 4DGS + background NeRF)
- **Süre:** 2 hafta
- **Risk:** **HIGH** — Major refactor

### 4.3 GAN Adversarial Refinement
- **Süre:** 1-2 hafta
- **Risk:** **HIGH** — Training instability

### 4.4 Custom Rasterization (Mip + LOD)
- **Süre:** 3-4 hafta
- **Risk:** **HIGH** — Mostly research

**Phase 4 toplam:** 3-7 hafta (opsiyonel)

---

## Risk-Impact Matrisi

```
                LOW IMPACT          MEDIUM IMPACT       HIGH IMPACT
LOW RISK     │  3.4 API/UX      │  1.6 LPIPS        │  1.1 Cache infra    │
             │  3.5 Camera path │  1.7 MV consist.  │  1.2 Heavy preproc  │
             │                  │  2.2 Multi-res    │  1.4 Per-cam depth  │
             │                  │                   │  1.5 Per-cam mask   │
             │                  │                   │  1.8 Flow loss      │
             ├──────────────────┼───────────────────┼─────────────────────┤
MEDIUM RISK  │  3.2 Adaptive SH │  2.3 Cam refine   │  2.1 Static/Dyn split│
             │                  │  2.4 Skybox bg    │                     │
             │                  │  2.5 Better deform│                     │
             │                  │  3.1 Compression  │                     │
             │                  │  3.3 Test-time opt│                     │
             ├──────────────────┼───────────────────┼─────────────────────┤
HIGH RISK    │  4.3 GAN refine  │  4.2 Background NF│  4.1 Mip-Splatting  │
             │  4.4 Custom rast.│                   │                     │
```

---

## Beklenen Kalite Trajektörisi

| Phase Bittiğinde | PSNR | Loss | 4dv.ai Eşitlik |
|---|---|---|---|
| Phase 0 (mevcut) | 28-32 | 0.05-0.07 | %75 |
| Phase 1 (low risk paketi) | 32-34 | 0.04-0.05 | %85 |
| Phase 2 (mimari) | 34-36 | 0.03-0.04 | %95 |
| Phase 3 (polish) | 35-37 | 0.025-0.03 | %100 |
| Phase 4 (advanced) | 36-38+ | 0.02 | %100+ |

---

## Risk Yönetimi (Her Upgrade İçin)

1. **Branch:** `feat/phaseX-XX_topic` (main'e direkt push yok)
2. **Smoke test:** Küçük scene'de hızlı validation (banana_demo, 2k iter)
3. **Regression check:** Eski sahneler hâlâ ≥ baseline kalitede mi?
4. **Merge sonrası full validation** ana sahnede
5. **Rollback hazır:** Kalite düşerse hemen revert

**Phase 4 için extra:** Sandboxed gsplat fork ayrı repo, GAN training ayrı GPU isolation, production main'i risketme.

---

## Engineering Lessons Learned (Phase 0'dan)

1. **Multi-view supervision yetersiz:** Per-cam supervision frekansı = iters/n_cam → uzun training (50k+ iter) zorunlu
2. **Random init kötü:** Densify+prune dynamics COLMAP-init varsayar, random init için modify gerekli (yapıldı)
3. **Convention bug'ları sessiz:** Visual debug (render frame) sayısal metric'ten daha güvenilir
4. **Linux mount sync delay:** Büyük dosya yazma sonrası bekleme/retry gerekli
5. **Scene extent overshoot:** N3V LLFF "far" değeri background dahil → foreground-focused estimation gerekli
6. **L1 sparsity ≠ static-dynamic separation:** Sparsity baskısı dynamic motion'u da bastırır, explicit split daha iyi

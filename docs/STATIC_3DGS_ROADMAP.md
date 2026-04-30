# Static 3D Gaussian Splatting — Yol Haritası

**Hedef:** Mevcut 4D dynamic 4DGS Studio'ya **static 3D mode** ekleyerek tek-frame multi-view veya foto seti input'ları işleyebilen, hız-kalite dengesi ayarlanabilir bir pipeline kurmak.

**Bağlam:** 4D pipeline çalışıyor (deformation field + Fourier trajectories). 3D static = "deformation = identity" özel durumu. Çoğu kod reuse edilebilir, ek ~10-15% kod yeter.

---

## Mevcut codebase avantajları (static 3D için zaten orada)

✅ **gsplat CUDA rasterization** — 3DGS standart kütüphane
✅ **COLMAP pipeline** — multi-view foto setlerinden cam pose + sparse cloud
✅ **GaussianModel** — means/scales/quats/opacities/SH params + density controller
✅ **Adaptive densification + opacity reset** — INRIA standart 3DGS davranışı
✅ **Anti-streak regularizer (`lambda_aniso`)** — Mip-Splatting "lite" — yüksek aspect ratio Gaussian'ları cezalandırır
✅ **VGG-LPIPS** — perceptual loss, edge sharpness
✅ **Multi-resolution training schedule** — coarse-to-fine, premium quality
✅ **Background distance flag** — distant gauss bypass
✅ **PLY export** + Spark.js viewer
✅ **NaN guard, gradient clip, ckpt rollback** — stability

## Eksik (static için ekleyeceğimiz)

❌ **Static mode flag** — `cfg.disable_deformation` veya benzeri
❌ **Photo set input** (video yerine) — `data/<scene>/images/IMG_*.jpg`
❌ **3D-specific COLMAP preset** — exhaustive matching, OPENCV camera, BA refine
❌ **Speed/quality preset'leri** — `fast` / `balanced` / `high` (4D'deki standard/high/premium'a benzer)
❌ **Mesh export** — marching cubes (opt-in, Phase 4)
❌ **Compact-3DGS quantization** — file size compression (post-process, Phase 5)

---

## Karşılaştırılan 3DGS variant'ları

Web araştırmasından (2024-2025 SOTA):

| Variant | Yıl | Strength | Hız | Bizim için |
|---------|-----|----------|-----|------------|
| **Vanilla 3DGS** (Kerbl SIGGRAPH 2023) | 2023 | Stabil, INRIA standart | 30-45 dk @ 30k iter | ✅ **Faz 1 — gsplat zaten kullanıyoruz** |
| **Mip-Splatting** | CVPR 2024 | Anti-aliasing filtreleri | Vanilla'dan ~%10 yavaş | ⚠ Custom CUDA kernel gerek (gsplat fork). Phase 4 deferred |
| **2DGS** | 2024 | 2D disk splats, surface reconstruction, en hızlı surface | ~30 dk | ✅ **Faz 4 — mesh export için** |
| **GOF (Gaussian Opacity Fields)** | 2024 | Ray-tracing volume rendering, direct geometry extraction | Vanilla'dan biraz yavaş | ⚠ Major refactor, gsplat tabanlı değil |
| **SuGaR** | CVPR 2024 | Surface-aligned + Poisson mesh reconstruction | +30-60 dk training sonrası | ⚠ Mesh-specific, opsiyonel Phase 5 |
| **Compact-3DGS** | 2024 | Quantization + entropy coding, **39-55× compression** | Post-process | ✅ **Faz 5 — file size optimization** |
| **YOGO** | 2025 | Production-ready, deterministic budget control | Vanilla benzeri | ⚠ Yeni, henüz prod-test değil |

**Karar:** Vanilla 3DGS (gsplat) tabanlı, Phase 1+2'de eklediğimiz iyileştirmelerle (LPIPS, multi-res, BG flag, lambda_aniso) production-tier. Mesh + compression ileri faz.

---

## Yol haritası — 5 fazlı plan

### 🟢 Faz 1 — Static core mode (2-3 gün, low risk)

**Hedef:** Tek bir flag ile mevcut pipeline'ı static moda alma. 4D kodlarını bypass et.

**Yapılacaklar:**

1. **`cfg.train.static_mode: bool = False`** — yeni config alanı
2. **Trainer flag handling:**
   - `is_multiview` ve `is_multiview and not static_mode` koşulları
   - Static phase warmup yok (deformation zaten yok)
   - Fourier trajectory bypass (`gs.fourier_pos_coeffs = None` zorla)
   - MLP deformation forward atla
   - `_apply_deformation` → identity döndür
3. **DeformationField construct skip:**
   - `if cfg.train.static_mode: deform = None`
   - Trainer'da `if self.deform is None:` kontrol
4. **Photo-set input desteği:**
   - `data/<scene>/images/*.jpg` yapısı
   - Veya video → frames (mevcut)
   - `pipeline.py`'de `is_static_scene(scene)` helper
5. **3D-specific COLMAP preset:**
   - `colmap_camera_model = "OPENCV"` (distortion model)
   - `colmap_matching = "exhaustive"` (foto set için)
   - `single_camera = "0"` (her foto farklı pozisyon)
6. **Test scene:** Mevcut HyperNeRF cut-lemon t=0 frame'leri veya Tanks & Temples Truck

**Etki:** Mevcut pipeline static modda çalışır; 4D kodları conditional skip.

**Süre tahmini:** 3060 Ti, 30k iter, 720p — **30-45 dk** (4D'den daha hızlı, deformation yok).

**Beklenen kalite:** PSNR 26-30 dB (cut-lemon static), Tanks&Temples Truck ~22-26 dB (geniş scene).

---

### 🟢 Faz 2 — Hız-kalite preset'leri (1-2 gün)

**Hedef:** Kullanıcının "5 dk smoke" → "production" → "premium" arası seçim yapabilmesi.

**Preset table:**

| Preset | Iter | Resolution | Init points | Densify cap | Süre @3060 Ti | Kalite (PSNR Tanks&Temples Truck tahmini) |
|--------|------|------------|-------------|-------------|----------------|---------|
| **fast** | 7,000 | 720p | 50k random + 30k COLMAP | 100k cap | **5-8 dk** | 23-26 dB |
| **balanced** | 30,000 | 1080p | 100k random + 80k COLMAP | 250k cap | **30-45 dk** | 26-30 dB |
| **high** | 50,000 | 1080p | 100k random + 150k COLMAP dense MVS | 500k cap | **2-3 saat** | 28-32 dB |
| **premium** | 100,000 | original (4K+) | 100k random + 300k dense MVS | 1M cap | **6-10 saat** | 30-34 dB (4dv.ai-tier) |

**Önemli ayarlar preset'te:**
- `densify_grad_threshold` — fast'te 4e-4, premium'da 2e-4 (daha hassas)
- `lambda_lpips` — fast'te 0, balanced+'da 0.1
- `multires_schedule` — high+'da `[(0, 480), (15000, 720), (35000, 1080)]`
- `opacity_reset_interval` — 3000 (standart 3DGS)

**Yeni script:** `scripts/static_3dgs.py` — `heavy_preprocess.py` benzeri, static-specific runner:

```bash
python scripts/static_3dgs.py --scene truck --preset balanced
```

---

### 🟡 Faz 3 — Photo set + sparse view desteği (1 hafta)

**Hedef:** Video yerine **multi-view foto seti** input olarak destekle. Ayrıca sparse-view (10-20 foto) optimizasyonları.

**Yapılacaklar:**

1. **Photo set input:**
   - `data/<scene>/images/IMG_*.jpg` veya `.png`
   - Frame extract'i atla (zaten image var)
   - COLMAP exhaustive matching photo set için zorunlu
2. **Sparse view (10-20 foto) iyileştirmeleri:**
   - SfM init point sayısı az → init random fill ratio yüksek
   - Densify daha agresif (start_iter=200, threshold 1.5e-4)
   - LPIPS lambda yüksek (0.15) — recon supervision yetersiz, perceptual prior yardımcı
3. **Auto-detection:** `is_static_scene(scene)` — `images/` klasörü varsa static, `video.mp4` varsa video
4. **Background distance ratio düşürme:** Outdoor scene'lerde scene_extent büyük olur, ratio=1.5 daha agresif BG flag

---

### 🟠 Faz 4 — Mesh export (2DGS, 1-2 hafta, opsiyonel)

**Hedef:** Static reconstruction sonrası mesh çıkar (Blender/Unity import için).

**Yöntem:** **2DGS** (2D disk splats) — gsplat fork'ında experimental support var, ya da SuGaR-style Poisson reconstruction.

**Yapılacaklar:**

1. **2DGS opt-in flag:** `cfg.model.use_2dgs = True` — Gaussian'lar 2D disk olarak parameterize edilir (z-scale = 0)
2. **Marching cubes mesh extraction:** density field → 3D mesh
3. **Texture baking:** Gaussian SH coefficients → mesh vertex colors
4. **Export format:** `.obj` + `.mtl` veya `.glb`

**Süre:** Mesh extraction ~10-20 dk training sonrası.

**Use case:** 3D printing, Blender import, AR/VR asset.

---

### 🟠 Faz 5 — Compact-3DGS compression (1 hafta)

**Hedef:** PLY file size 10-50× azalt (web/mobile için).

**Yöntem:** **Compact-3DGS** (Lee et al. 2024)

1. **K-means clustering** Gaussian SH coefficients için
2. **Quantization:** position int16, scale int8, color uint8
3. **Entropy coding:** zstd veya custom

**Boyut karşılaştırma:**
- Standart PLY (250k gauss): ~100-150 MB
- Compact-3DGS: ~3-5 MB (40-50× compression)
- Hala WebGL/Spark.js ile renderlanabilir (decompression overhead minimal)

**Frontend etkisi:** Tauri app PLY yerine `.compact3dgs` formatı yükler, indirme süresi sıçrar. Web demo (online viewer) için kritik.

---

## Implementation sırası önerisi

### Bu hafta (low risk, hızlı kazanım)

**Faz 1 + Faz 2** — static core + preset'ler.

- **Gün 1-2:** Faz 1 — `static_mode` flag, 4D bypass kodları, photo-set input
- **Gün 3:** Faz 2 — `scripts/static_3dgs.py` runner, 4 preset tanımı
- **Gün 4:** Test — `cookie` veya `hypernerf_cutlemon` t=0 ile mini smoke + balanced run
- **Gün 5:** Commit + push

**Çıktı:** Mevcut pipeline static modda çalışır, 5 dk → 3 saat arası kalite tier'ları seçilebilir.

### Önümüzdeki hafta (orta risk)

**Faz 3** — sparse view + photo set polish.

### İleride (deferred)

**Faz 4 (mesh)** ve **Faz 5 (compression)** — kullanıcı talep ederse.

---

## Hız-kalite trade-off matrisi

```
                      DÜŞÜK KALİTE       ORTA KALİTE        YÜKSEK KALİTE
                      (PSNR 22-26)       (PSNR 26-30)       (PSNR 30-34)
HIZLI (5-30 dk)       fast preset        ⚠ erişilemez       ⚠ erişilemez
ORTA (30 dk - 2 sa)   ⚠ over-spec        balanced preset    ⚠ erişilemez
YAVAŞ (2-10 sa)       ⚠ over-spec        ⚠ over-spec        high / premium preset
```

**Pratik kullanım:**
- **5 dk smoke (fast):** sahne çalışıyor mu kontrol — kabataslak preview
- **30 dk (balanced):** sosyal medya, portföy demosu
- **2-3 saat (high):** profesyonel render, müşteri deliverable
- **6-10 saat (premium):** 4dv.ai-tier "wow factor" demos

---

## Ekstra düşünceler

### 4D pipeline ile birlikte yaşama

Static mode 4D kodlarını **disable** eder, 4D modunu etkilemez. Aynı codebase, aynı trainer, sadece flag farkı. Codebase ek karmaşıklık ~%10.

### Frontend etkisi

Static scene timeline'sız (tek frame), frontend dinamik vs static otomatik tespit eder:
- **Dynamic:** timeline + play/pause + speed (mevcut)
- **Static:** timeline gizli, sadece 3D viewport + WASD camera + screenshot

JobSubmitPanel'de yeni "Static / 4D" toggle, preset selector.

### Test scene'ler

- **Standard benchmarks:** Tanks & Temples (Truck, Train, Family), Mip-NeRF 360 (Bicycle, Garden, Bonsai), DeepBlending (Drjohnson, Playroom)
- **Bizim mevcut:** `cookie`, `split-cookie`, `hypernerf_cutlemon` t=0 frame
- **Yeni öneriler:** kullanıcı kendi photo set'ini upload ederek test

### Phase 1+2 kazanımları aktarımı

- **VGG-LPIPS:** static'te de aktif (config-driven, kod değişikliği yok)
- **Multi-res training:** static'te de çalışır
- **BG distance flag:** static'te ÖZELLİKLE faydalı (outdoor scenes'te distant sky/ufuk gauss'ları)
- **Density adaptive:** zaten static-tested INRIA original

---

## Sources

- [Awesome 3D Gaussian Splatting Paper List (MrNeRF)](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)
- [SuGaR (CVPR 2024)](https://github.com/Anttwo/SuGaR)
- [2DGS (2024)](https://surfsplatting.github.io/)
- [Gaussian Opacity Fields (GOF)](https://niujinshuchong.github.io/gaussian-opacity-fields/)
- [Compression in 3D Gaussian Splatting Survey](https://arxiv.org/html/2502.19457v1)
- [Surface Reconstruction Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453780/)
- [3D GS Review 2025](https://link.springer.com/article/10.1007/s10489-026-07227-9)
- [FlashGS CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Feng_FlashGS_Efficient_3D_Gaussian_Splatting_for_Large-scale_and_High-resolution_Rendering_CVPR_2025_paper.pdf)

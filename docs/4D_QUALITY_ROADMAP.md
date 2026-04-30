# 4D Gaussian Splatting — Eksik Parçalar ve Kalite Yol Haritası

**Hedef:** 4D pipeline'ı production-tier yapmak. 4dv.ai parite + novel view synthesis + smooth playback + sparse-view robustluğu.

**Şu an nerede duruyoruz (tamamlanan büyük taşlar):**

- Phase 1.1–1.9: Heavy preprocess cache, multi-view depth (Metric3D-Large), per-cam dynamic mask (SAM2), VGG-LPIPS, multi-view consistency, RAFT optical flow, densify MV tuning
- Phase 2.1–2.5: Static/Dynamic auto-promote, multi-resolution training schedule, camera pose refinement (joint BA), background distance flag, multi-resolution HexPlane
- Single-view + multi-view N3V uçtan uca çalışıyor (banana_demo, flame_steak)

**Eksik kalan / kalite kapayan parçalar (12 madde):**

---

## 1. Novel View Synthesis (NVS) Evaluation Framework ⭐ PRIORITY

**Problem:** Şu an "kalite" sadece training PSNR ile ölçülüyor. Held-out view'da kalite bilinmiyor — overfit kontrol yok. 4dv.ai-tier render demosu yapamıyoruz.

**Çözüm:** End-to-end NVS değerlendirme hat'tı:
- **Held-out test camera** — multi-view'da 1 cam train'den çıkarılır (zaten N3V'de cam00 test ayrılmış, ama metric çıkarılmıyor)
- **Test-time render** — held-out cam'in tüm timestep'lerinde render → ground truth ile PSNR/SSIM/LPIPS
- **Smooth orbit camera path** — eğitilmiş cam pozlarından spline interpolasyon, novel viewpoint'lerde render
- **MP4 export** — orbit/test render'dan video çıktısı, `output/<scene>/eval/`'a yazılsın
- **Frontend "Eval" tab** — NVS metrikleri grafik + indirilebilir mp4

**Effort:** 2-3 gün  
**Impact:** Çok yüksek — kalite regression test, demo materyali  
**Risk:** Düşük

**Implementation:**
- `backend/eval/nvs_eval.py` (yeni) — held-out cam metrics hesaplama
- `backend/eval/orbit_render.py` (yeni) — Catmull-Rom spline, smooth cam path generation
- `backend/eval/video_export.py` (yeni) — torch frames → mp4 (imageio/ffmpeg)
- `pipeline.py` — Faz 7 olarak training sonrası eval
- `backend/api.py` — `GET /jobs/{id}/eval`, `GET /jobs/{id}/orbit.mp4`
- Frontend yeni tab "Eval"

---

## 2. Mip-Splatting Anti-Aliasing 🎯 PRIORITY

**Problem:** Mevcut gsplat'ta multi-resolution rendering aliasing yapıyor. Yakın kamera shot'larında detay, uzaklarda blur olmuyor — Mip-NeRF360-style anti-aliasing yok. Bu 4dv.ai-tier görüntü kalitesinde belirgin fark yaratan bir parça.

**Çözüm:** Mip-Splatting paper (CVPR 2024) — 3D smoothing filter + 2D anti-aliasing filter:
- Her Gaussian'a max projeksiyon size'a göre alt sınır eklenir (3D filter)
- Ekran-uzayında Gaussian'a max-pixel-size garantisi (2D filter)
- Her iki filtre PSNR +1-2 dB, multi-scale render dramatik temizlenir

**Effort:** 1-2 hafta (gsplat fork gerekli VEYA python-side post-process)  
**Impact:** Yüksek (PSNR + visual quality)  
**Risk:** Orta (CUDA kernel modifikasyonu)

**İki yaklaşım:**
- A) **gsplat fork** — `_make_mip_safe()` 3D filter scale'lere ekle, rasterize'a geçirken kernel'a 2D filter parametresi
- B) **Python-side approximation** — render öncesi her Gaussian scale'e min epsilon, render sonrası bilateral filter (kalite kazanım %50, effort %20)

**Tercih:** Önce B (hızlı kazanım), sonra A (premium quality için).

---

## 3. Better Initialization (DUSt3R / VGGSfM) 🟢 PRIORITY (sparse-view için kritik)

**Problem:** COLMAP SfM seyrek-view'da (10-20 foto) yetersiz; ya başarısız ya çok az nokta üretir. Bu durumda 3DGS init noktası yok, training divergence riski yüksek.

**Çözüm:** Modern transformer-based dense init:
- **DUSt3R** (2024) — 2 image'tan dense pointmap + relative pose. Sparse-view'da COLMAP'tan 10× daha güvenilir.
- **VGGSfM** (Meta CVPR 2024) — feed-forward SfM, gradient-based BA, end-to-end training.
- **MASt3R** (DUSt3R++) — daha iyi feature matching.

**Use case'ler:**
- Static 3DGS sparse photo set (10-20 foto, klasik 3DGS başarısız)
- 4D dynamic'te kısa video (< 100 frame) — COLMAP yine yetersiz olabilir

**Effort:** 1 hafta (DUSt3R weights download + integration + COLMAP fallback chain)  
**Impact:** Yüksek (sparse-view güvenirlik)  
**Risk:** Düşük (mevcut COLMAP'tan ayrı path, fallback hep var)

**Implementation:**
- `backend/preprocess/dust3r_init.py` (yeni) — DUSt3R inference + pointmap → COLMAP-format conversion
- `cfg.preprocess.init_method = "colmap" | "dust3r" | "vggsfm" | "auto"` (auto = sparse-view detect)
- Pipeline'da COLMAP başarısız olursa otomatik DUSt3R fallback

---

## 4. Background NeRF / Sky Model 🟡

**Problem:** Outdoor scene'lerde uzak gökyüzü/ufuk Gaussian'larla temsil edilince floater + aliasing oluşur. Phase 2.4'te "BG distance flag" ekledik ama bu sadece distant gauss'lara deformation bypass — gerçek BG model değil.

**Çözüm:** Mip-NeRF360-style "ring NeRF" arka plan:
- Belirli mesafe sonrası küçük bir NeRF (ring-based parametrization)
- Foreground Gaussian'lar + BG NeRF birlikte render (composite)
- Outdoor scene'lerde sky/horizon dramatik temiz

**Effort:** 2-3 hafta  
**Impact:** Orta-Yüksek (sadece outdoor scene'lerde belirgin)  
**Risk:** Yüksek (rendering pipeline'da composite gerekli, mevcut gsplat path'e ekleme)

**Karar:** Phase 4'e ertelenebilir — outdoor scene test setimiz yok.

---

## 5. Sparse-View Diffusion Prior 🟠

**Problem:** Çok seyrek view (3-5 foto) durumunda standart reconstruction loss yetersiz. Sahnenin görmeyen bölgeleri (occluded, geometry holes) garbage çıktı verir.

**Çözüm:** Stable Diffusion / latent diffusion based view synthesis prior:
- **DreamGaussian / Reconstruct Anything Model (RAM)** — sparse-view + diffusion guidance
- **4D-fy / DreamScene4D** — tek görüntü/video → 4D Gaussian
- SDS (Score Distillation Sampling) loss — diffusion model rendered view'ı "olası" yöne pushlar

**Effort:** 3-4 hafta  
**Impact:** Yüksek (sadece sparse-view, dense-video'da etkisiz)  
**Risk:** Yüksek (diffusion model integration, GPU memory)

**Karar:** Tier 4'e ertelenebilir.

---

## 6. Trajectory Smoothing Temporal Regularizer 🟡

**Problem:** Mevcut `lambda_smoothness` sadece D(t) vs D(t+1) farkı. Higher-order smoothness yok (acceleration, jerk). Bu özellikle slow-motion render'da titreme görünür.

**Çözüm:** 2nd-order ve 3rd-order temporal smoothness:
- `lambda_accel` — D(t-1) - 2D(t) + D(t+1) cezası (acceleration)
- `lambda_jerk` — 3rd derivative ceza
- Adaptive — sadece dynamic gaussian'lara uygula (Phase 2.1 split bunu mümkün kılar)

**Effort:** 2-3 gün  
**Impact:** Orta (slow-motion render'da görünür, normal hızda subtle)  
**Risk:** Düşük

**Implementation:**
- `backend/model/trainer.py` — `_compute_motion_regs()` içine 2nd order ekle
- `cfg.train.lambda_accel = 0.0` default

---

## 7. Adaptive Densify for Dynamic Regions 🟢

**Problem:** Dynamic regions (yüksek motion mask vote) detay daha çok ister; static regions zaten oturmuş. Mevcut densify yapımız dynamic/static fark gözetmez.

**Çözüm:** Phase 2.1 split kullanarak:
- Dynamic gaussian'lar için `densify_grad_threshold * 0.5` (2× hassas)
- Static'lerde mevcut threshold
- Densify pruning de aynı şekilde adaptive

**Effort:** 1 gün  
**Impact:** Orta (motion-rich scene'lerde görünür)  
**Risk:** Düşük

---

## 8. 4D Test Camera (spatial + temporal hold-out) 🟢

**Problem:** N3V'de spatial held-out (cam00) var. Ama temporal held-out yok — train'de gördüğü timestep'leri novel-time'da test etmiyoruz.

**Çözüm:** Train timestep'leri her N'de bir atla, test'te göster:
- 100 frame video → 90 train + 10 held-out timestep
- Held-out timestep'lerde tüm cam'lardan render → temporal interpolation kalitesi

**Effort:** 1-2 gün (Madde 1 NVS framework'üne ek)  
**Impact:** Orta-Yüksek (motion overfit detect)  
**Risk:** Düşük

---

## 9. Mesh Extraction (2DGS / SuGaR) 🟠

**Problem:** Splat output sadece point cloud — 3D printing / Blender / Unity import için mesh gerekir.

**Çözüm:** İki opsiyon:
- **2DGS** — Gaussian'ları 2D disk olarak parametre et (z-scale = 0), surface reconstruction direkt
- **SuGaR** — splat sonrası Poisson reconstruction

**Effort:** 2-3 hafta  
**Impact:** Orta (sadece mesh output use-case)  
**Risk:** Yüksek

**Karar:** Static 3DGS Roadmap Faz 4'te zaten ertelenmişti — orada bekle.

---

## 10. Compact-3DGS Compression 🟠

**Problem:** PLY dosyası 250k gauss için 100-150 MB. Web/mobile için fazla.

**Çözüm:** Compact-3DGS paper:
- K-means SH coefficient quantization
- Position int16, scale int8, color uint8
- 40-50× compression, ~3-5 MB
- WebGL/Spark.js decompress overhead minimal

**Effort:** 1 hafta  
**Impact:** Orta (web demo için kritik, lokal viewer için nötr)  
**Risk:** Düşük

**Karar:** Static 3DGS Faz 5'te ertelenmiş — orada bekle.

---

## 11. Adaptive SH Degree Schedule 🟢

**Problem:** SH degree=3 ile training boyunca tüm dataset üzerinde train ediliyor. Erken iter'lerde DC + 1st order yeterken, late iter'lerde 3rd order detail kazandırır.

**Çözüm:** Iter-based schedule:
- 0-5k iter: SH degree 0 (DC only)
- 5k-15k: SH degree 1
- 15k-25k: SH degree 2
- 25k+: SH degree 3

**Effort:** 1 gün  
**Impact:** Düşük-Orta (PSNR +0.3-0.5 dB)  
**Risk:** Düşük

---

## 12. Camera Refinement Convergence Improvements 🟢

**Problem:** Phase 2.3'te eklenen joint BA basit Adam optimizer ile. Convergence yavaş, large-scale scene'lerde drift görülebilir.

**Çözüm:**
- Camera refinement için ayrı warmup schedule (mevcut'ta var ama default 5000 iter geç)
- Levenberg-Marquardt-style adaptive step size
- Cam parametreleri için separate gradient norm clipping

**Effort:** 2 gün  
**Impact:** Orta (large outdoor scene'lerde görünür)  
**Risk:** Orta

---

# Öncelik Matrisi

| # | Madde | Impact | Effort | Risk | Tier |
|---|-------|--------|--------|------|------|
| 1 | NVS Evaluation framework | Çok Yüksek | 2-3 gün | Düşük | **🟢 Tier 1 — bu hafta** |
| 2 | Mip-Splatting (B yaklaşımı) | Yüksek | 1 hafta | Orta | **🟢 Tier 1 — bu hafta** |
| 3 | DUSt3R sparse init | Yüksek | 1 hafta | Düşük | **🟢 Tier 1 — bu hafta** |
| 7 | Adaptive densify dynamic | Orta | 1 gün | Düşük | 🟢 Tier 1 |
| 8 | 4D temporal test cam | Orta-Yüksek | 1-2 gün | Düşük | 🟢 Tier 1 |
| 6 | Higher-order trajectory smoothing | Orta | 2-3 gün | Düşük | 🟡 Tier 2 |
| 11 | Adaptive SH schedule | Düşük-Orta | 1 gün | Düşük | 🟡 Tier 2 |
| 12 | Cam refinement improvements | Orta | 2 gün | Orta | 🟡 Tier 2 |
| 4 | Background NeRF | Orta-Yüksek | 2-3 hafta | Yüksek | 🟠 Tier 3 |
| 2A | Mip-Splatting CUDA fork | Yüksek | 1-2 hafta | Yüksek | 🟠 Tier 3 |
| 5 | Diffusion prior (SDS) | Yüksek | 3-4 hafta | Yüksek | 🔴 Tier 4 |
| 9 | Mesh extraction | Orta | 2-3 hafta | Yüksek | 🔴 Tier 4 |
| 10 | Compact-3DGS compression | Orta | 1 hafta | Düşük | 🔴 Tier 4 |

---

# Önerilen Sıra (3-4 hafta plan)

**Hafta 1:** Madde 1 + 8 (NVS evaluation framework + temporal hold-out)
- Tier 0 deliverable: ilk gerçek "demo material" — orbit cam mp4 + held-out PSNR/SSIM/LPIPS

**Hafta 2:** Madde 2-B + 7 (Mip-Splatting Python-side + adaptive densify)
- Tier 0 deliverable: PSNR +1-2 dB ölçülebilir kazanım

**Hafta 3:** Madde 3 (DUSt3R sparse init)
- Tier 0 deliverable: 10-20 foto sparse-view scene end-to-end çalışıyor

**Hafta 4:** Madde 6 + 11 + 12 (Tier 2 küçük iyileştirmeler)
- Tier 0 deliverable: cumulative PSNR +0.5-1 dB

---

# Quick wins (1 günden az)

Bu maddelerden bağımsız, hemen yapılabilir küçük temizlikler:
- ⚡ **Run logger fsync bug** (Task #73) — log'lar bazen aborted run'da kaybolur
- ⚡ **Densify history persistent** — densify adaptive davranışı için per-gauss accum gradient'ları ckpt'ye yaz/oku
- ⚡ **Test scene config presets** — Mip-NeRF360 / Tanks&Temples / N3V için per-scene tuned hyperparam preset'leri

---

# Notlar

- Phase 1+2 implementations zaten 4D'nin %70'ini kapatıyor. Bu dokümandaki maddeler "production-tier'a son %30"u kapatır.
- Madde 1 (NVS eval) ve Madde 2 (Mip-Splatting) en hızlı görsel kazanım sağlar.
- Madde 3 (DUSt3R) sparse-view scenarios'u açar — kullanıcı seyrek foto/kısa video atarsa şu an pipeline başarısız.
- Diffusion prior (Madde 5) ve mesh export (Madde 9) deferred kalsın — kullanıcı talep eder ya da spesifik use-case çıkarsa.

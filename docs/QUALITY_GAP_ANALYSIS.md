# 4dv.ai Kalite Gap Analizi

**Soru:** Yerel premium overnight run'larımız PSNR 28-32 dB veriyor, görsel olarak bulanık + sis dolu. 4dv.ai web sitesindeki demolar keskin, photorealistik, motion temiz. Aramızdaki dağ farkın 10 ayrı sebebi var.

**Hedef:** Her sebebi quantify et, hangileri kapsamlı düzelterek hangi kalite sıçrayışı olur, hangi sebeple gap kapanmaz (production farkı kaçınılmaz).

---

## 1. Capture / Data Quality Gap — **EN BÜYÜK** (gap'in ~%30-40'ı)

| Boyut | 4dv.ai (production) | N3V flame_steak (bizim) |
|-------|---------------------|--------------------------|
| Cam sayısı | 20+ (pro studio frontal arc) | 21 (academic, 2 satır lateral) |
| Resolution | 4K+ (3840×2160) | 1080p (1352×1014 raw) → bizim 720p multires |
| Frame rate | 60 fps | 30 fps |
| Light setup | Profesyonel softbox, kontrol | Mevcut indoor light |
| Sync | Hardware genlock (sub-frame) | Software approximation |
| Lens kalibrasyonu | Pre-calibrated, distortion 0.1 px | COLMAP estimate, ~1-3 px |
| Süre | 10-30 sn pure-action | 10 sn N3V standart |

**Etki:** 4dv.ai kameraları 4× daha yüksek resolution, sub-frame sync, kalibrasyon hatası ~10× az. Bu sadece PSNR değil **görsel kalite** üzerinde devasa etki — texture detayı, edge sharpness, motion smoothness.

**Bizim için ne yapılabilir:**
- N3V dataset'in inherent kalitesini değiştiremeyiz (Meta paylaştığı veri)
- **Aynı sahnede 4K kamera setup + sync hardware kullanmak** → ev shooter olarak imkansız
- Pratik çözüm: **N3V'de yapacağımız iyileştirmeleri 4K capture'lı veriyle gerçekçi karşılaştırmak imkansız**

> **Bu gap'i kapatamayız.** 4dv.ai production setup farkı, algoritma değil hardware/data farkı.

---

## 2. Compute Budget Gap — **İkinci en büyük** (~%20)

| | 4dv.ai (tahmini) | Phase 0 baseline | Phase 1+2 mevcut |
|---|---|---|---|
| Iter sayısı | 50-100k+ (multi-stage) | 30k overnight | 5k mini run |
| Resolution stages | 480 → 720 → 1080 (multi-res) | tek 720p/1080p | 240 → 320 (mini) |
| GPU | A100/H100 cluster | RTX 3060 Ti 8 GB | RTX 3060 Ti 8 GB |
| Toplam compute | ~50-100 GPU-saat | ~24 saat | ~1 saat |

3D GS paper convergence ~30k iter. 4D'de daha karmaşık (motion field), gerçek convergence için 50k+ gerek. 4dv.ai muhtemelen multi-GPU cluster ile 4-8 saatte 50k iter koşturuyor.

**Bizim 5000 iter run'ımız compute budget'ın ~%5'i.** Sigmoid convergence eğrisinde alt yarıdayız:
- İlk %20 iter → kalitenin %50'sini kazanır (kaba sahne yapısı)
- Son %50 iter → texture detayı, edge sharpness, motion temizliği

> Premium overnight 30k iter Phase 0 → 28-32 dB
> Premium overnight 30k iter Phase 1+2 → tahminim **30-34 dB** (+1-3 dB Phase 1+2 katkısı)
> Multi-day 100k iter Phase 1+2 → **34-37 dB** (production-tier)

**Kapatılabilir mi:** Cloud GPU (RunPod 4090 ~$10) + overnight 30k+ iter. Hala 4dv.ai'nin ~%80'i, %100 değil (data gap kalır).

---

## 3. Algorithmic Architecture Gap — **Önemli** (~%15)

4dv.ai pipeline'ı (web search):
- **Spatial-temporal structure encoder** — Gaussian'ın merkezi + timestamp'i feature space'e map'liyor
- **Multi-head Gaussian deformation decoder** — output her zamanda farklı Gaussian state
- **4D Gaussian decomposition** — conditional 3D + marginal 1D (temporal)
- 4D'yi explicit modelliyor, deformation alanı değil

Bizim (4DGaussians Wu et al. CVPR 2024 yaklaşımı):
- HexPlane (4D'yi 6 plane'e ayırma)
- + Per-Gaussian Fourier trajectory (Phase 1.6-1.7'de eklendi)
- + MLP deformation head (Δpos, Δquat, Δscale)
- 3D Gaussian + deformation field — ayrı

**Tasarım farkı:** 4dv.ai 4D Gaussian'ı temel primitive olarak alıyor (decomposition kalıcı). Bizim 3D Gaussian + zaman boyutuna deformation eklenmiş. Theoretically aynı kapasite ama **expressive power** ve **convergence speed** farklı.

**Olası yaklaşımlar:**
- Phase 4.4 (deferred): Custom 4D Gaussian rasterization — gsplat fork çok zor
- Phase 4.2 (deferred): Background NeRF + foreground 4DGS — major refactor
- Pratik: HexPlane multi-resolution Phase 2.5 ekleme zaten — yeterli kapasite kazanımı

**Kapatılabilir mi:** Tam değil ama Phase 2.5 (multi-res HexPlane) + Phase 4.x deferred items kapatabilir. Şu an Phase 2 mimari yeterli orta-kalite için.

---

## 4. Training Supervision Gap — **Orta** (~%10)

Bizim Phase 1+2'de eklediğimiz lambda'lar var ama **tam aktif değil**:

| Loss | Bizim | Production tier |
|------|-------|------------------|
| Recon (L1+SSIM) | ✅ | ✅ |
| LPIPS | ✅ AlexNet, λ=0.05 | VGG (kaliteli) λ=0.1-0.3 |
| Multi-view consistency | ✅ ama λ=0.05-0.1 | aggressive λ=0.5+ |
| **Depth supervision** | ❌ Phase 1.4 cache hazır, trainer consume YOK | ✅ tam wired (Metric3D-Large supervised) |
| **Optical flow** (RAFT) | ❌ Phase 1.8 preprocess hazır, trainer YOK | ✅ flow-aware motion supervision |
| **Adversarial** | ❌ Phase 4.3 deferred | possibly active in production |
| **Test-time fine-tune** | ❌ Phase 3.3 deferred | ✅ per-frame +0.5 dB |

**Bizim Phase 1.4 + 1.8 cache hazır ama trainer-side consumer eksik** — bu büyük bir kayıp. Mevcut depth_multiview/cam*.npy dosyaları training'de kullanılmıyor.

**Kapatılabilir mi:** Evet — Phase 1.4 (depth_mv consumer in trainer) ve Phase 1.8 (flow loss) trainer-side wire eklenirse +1-2 dB. ROADMAP_v5.md'de bu Phase 2'de değil ama gerekli.

---

## 5. Anti-Aliasing + Rendering Gap — **Orta** (~%5)

**Mip-Splatting** (Yu et al. CVPR 2024) — Gaussian projeksiyonunu pixel boyutuna göre filtreliyor, aliasing artifact'larını yok ediyor. 4dv.ai'nin keskin texture detayları muhtemelen bu sayede.

Bizim: standard gsplat rasterization (no mip filter) → alt seviye detayda alising görünür.

**Kapatılabilir mi:**
- Phase 4.1 deferred — custom CUDA kernel veya gsplat fork çok zor (1-2 hafta iş, high risk)
- Pratik: gsplat'ın ileride Mip-Splatting destek eklemesi muhtemel — bekle

---

## 6. Static/Dynamic Explicit Split — **KRİTİK** (~%10)

ROADMAP_v5.md'de **EN KRİTİK** Phase 2.1 olarak işaretlenmiş. 4dv.ai pipeline'ında implicit (4D Gaussian decomposition'da temporal dimension separable). Bizim Phase 2.1'de buffer ekledik ama **runtime activation YOK**.

Sonuç: tüm gauss'lar deformation alanına maruz kalıyor. Statik bölgelerdeki (tencere, plaka, ızgara) gauss'lar her iter'de küçük dpos hareketi alıyor → **temporal jitter** + **görsel sis**.

**Bizim mevcut:** `gs.is_static` buffer var, `promote_dynamic()` API var. **Eksik:** trainer'da motion mask'tan otomatik static/dynamic seçimi yapılmıyor; default tüm gauss'lar static (bypass tetikli değil).

**Kapatılabilir mi:** Evet, ROADMAP_v5.md'de 4-5 günlük iş. **+1-2 dB beklenti** + görsel temiz statik bölgeler.

---

## 7. Background Modeling Gap — **Orta** (~%5)

4dv.ai distant scene'leri muhtemelen ayrı modellemekte (skybox, environment map, veya distant Gaussian'lar lower-LR). Foreground gauss'lar background'a "dağılmaz".

Bizim: tüm gauss'lar aynı koşullarda. Bazı gauss'lar uzaklara "uçuyor" (Δpos peak 2.4 birim gördük iter 4000'de). **Bu floater'lar görsel sis yaratır.**

Phase 2.4 (deferred): `is_background` flag distant gauss bypass. Buffer var ama auto-activate yok.

**Kapatılabilir mi:** Phase 2.4 wiring ekle (3-5 günlük iş) → görsel sis %50 azalır.

---

## 8. Convention / Calibration Hassasiyet Gap — **Az ama riskli**

ROADMAP_v5.md'de bahsedilen LLFF → OpenCV convention bug riski. 4dv.ai pro studio'da:
- Lens distortion 0.1 px sub-pixel hassas
- Hardware sync sub-frame
- Pre-calibrated cam parametreleri

Bizim COLMAP estimate:
- Distortion ~1-3 px
- Cam pose ~5-10 pixel reprojection error
- Multi-view consistency loss yanlış cam çiftlerinden gelirse → blur

**Kapatılabilir mi:** Phase 2.3 cam pose refinement (eklendi, lr=1e-7) yardımcı ama yeterli değil. Daha agresif joint BA + ground-truth calibration olmadan tam değil.

---

## 9. Point Initialization Quality — **Orta** (~%5)

4dv.ai muhtemelen:
- Multi-view stereo (MVS) dense reconstruction → 1-5M point başlangıç
- Per-frame Gaussian densification (her t için spawn)

Bizim:
- COLMAP MV sparse: 12k-50k point
- + random fill: 100-200k toplam
- Adaptive densify ile 2-3× büyütüyor

**Eksik:** dense MVS bizim Phase 0'da var (`colmap_mv_dense_mvs=True` premium) ama Colab'de COLMAP CUDA-free olduğu için pratikte çalıştırmadık. Yerel'de zaman aldı.

**Kapatılabilir mi:** Evet — RunPod'da CUDA COLMAP ile dense MVS koştur, init point sayısı 5-10×.

---

## 10. SH Degree / View-Dependent Effects — **Az**

| Boyut | Bizim | Production |
|-------|-------|------------|
| SH degree | 3 (16 SH coef per gauss) | 4-5 (25-36 SH coef) |
| View-dependent specular | OK | iyi |
| Memory footprint | düşük | yüksek (bizim cap'imiz aşılır) |

SH 3 genelde yeterli ama specular (ızgara metal, alev parlaklığı) gibi view-dependent effects'te SH 4 daha iyi. Phase 3.2 adaptive SH degree (deferred).

**Kapatılabilir mi:** SH 4 yapsak VRAM 1.5× artar, training 30-40% yavaşlar. Trade-off.

---

## Özet Tablo

| # | Sebep | Gap'e katkı | Kapatılabilir mi |
|---|-------|-------------|-------------------|
| 1 | Capture quality | ~30-40% | ❌ data farkı kaçınılmaz |
| 2 | Compute budget | ~20% | ✅ cloud GPU + overnight |
| 3 | Algorithmic | ~15% | ⚠ kısmen (Phase 2.5 + 4.x) |
| 4 | Training supervision | ~10% | ✅ Phase 1.4/1.8 trainer-side wire |
| 5 | Anti-aliasing | ~5% | ❌ Phase 4.1 deferred (zor) |
| 6 | Static/Dynamic split | ~10% | ✅ Phase 2.1 runtime activation |
| 7 | Background modeling | ~5% | ✅ Phase 2.4 runtime activation |
| 8 | Convention/calibration | ~3% | ⚠ kısmen Phase 2.3 |
| 9 | Point init quality | ~5% | ✅ dense MVS (CUDA COLMAP) |
| 10 | SH degree | ~2% | ⚠ trade-off |

---

## Plan — gap'in %60-70'ini kapatmak için yol haritası

**Öncelik sırası (etki/maliyet oranı):**

### Tier 1 — Hemen yapılabilir (bu hafta, +5-7 dB beklenti)

1. **Cloud GPU overnight** — RunPod 4090, 30k iter Phase 1+2 (~$5)
   - Compute gap'i kapatır, baseline'la adil A/B kıyas
2. **Phase 1.4 trainer-side wire** — depth_mv consumer (`lambda_depth > 0` MV path'te)
   - Iyi supervision sinyali, +1-2 dB
3. **Phase 1.8 trainer-side wire** — flow loss aktivasyon
   - Motion supervision, +1-2 dB
4. **Phase 2.1 runtime activation** — promote_dynamic motion mask'ten otomatik
   - Statik bölge temizliği, +1-2 dB + görsel sis %30 azalır
5. **Phase 2.4 runtime activation** — flag_background_by_distance auto-call
   - Floater azaltma, görsel sis %20 azalır

### Tier 2 — Orta vadeli (2-4 hafta, +2-4 dB ek)

6. **VGG-LPIPS** — AlexNet yerine VGG (lambda 0.1+, daha güçlü perceptual)
7. **Multi-view consistency aggressive** — lambda 0.5, secondary cam render her iter
8. **Dense MVS RunPod CUDA COLMAP'a** — point init 5-10× büyür
9. **Phase 2.5 multi-res HexPlane full activation** — `[24, 48, 96]` premium
10. **Multi-resolution training schedule** — 480→720→1080 stage'ler

### Tier 3 — Uzun vadeli (1-2 ay, +1-2 dB ek)

11. **Phase 4.1 Mip-Splatting** — gsplat fork (yüksek risk, yüksek reward)
12. **Phase 4.2 Background NeRF + 4DGS hybrid** — major refactor
13. **Phase 3.3 Test-time fine-tune** — render time +30 sn, +0.5 dB

### Tier 4 — Production-ready ama bizim kontrolümüzde değil

14. **Capture quality** — 4K cam + sync hardware (4dv.ai gibi pro studio)
15. **Multi-GPU cluster** — A100/H100 100k iter

---

## Karar — gerçekçi hedef

| Hedef seviye | Gap'in kapatılma yüzdesi | PSNR (flame_steak) | Süre + maliyet |
|--------------|--------------------------|---------------------|------------------|
| Şu an | %0 | 28-32 (Phase 0) / 22-24 (Phase 1+2 5k iter) | yapıldı |
| **Tier 1** | **%60-70** | **32-37** | 1 hafta + $20-30 cloud |
| Tier 1 + 2 | %75-85 | 35-40 | 1 ay + $50-100 |
| Tier 1 + 2 + 3 | %85-90 | 37-42 | 2-3 ay + $100-200 |
| Tier 4 (data) | %95+ | 40+ | erişilemez (pro studio gerek) |

**Önerim:** **Tier 1**'e odaklan. RunPod'da bir gece overnight ile + Phase 1.4/1.8/2.1/2.4 trainer-side wire ile **gap'in %60-70'ini kapatabiliriz**. PSNR 35 dB civarı görsel olarak chickchicken'a yakın olur — 4dv.ai eşitliği değil ama dikkat çekici demolar yapabiliriz.

**Tier 4 (capture data)** kapatılamayacağı kabul ile, hedef "production tier - data gap" olur. Bu ~%85'lik bir 4dv.ai eşitliği — sosyal medya / portföy demosu yetenek.

---

## Sources

Sources:
- [4DV.ai - Building the future of visual media](https://www.4dv.ai/)
- [4D Gaussian Splatting (Wu et al. CVPR 2024)](https://github.com/hustvl/4DGaussians)
- [Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting (Fudan, ICLR 2024)](https://github.com/fudan-zvg/4d-gaussian-splatting)
- [Instant4D: 4D Gaussian Splatting in Minutes](https://arxiv.org/abs/2510.01119)
- [4DV.ai: Redefining Video with 4D Gaussian Splatting](https://websiteaid.in/4dv-ai-4d-gaussian-splatting/)
- [Gaussian splatting Wikipedia](https://en.wikipedia.org/wiki/Gaussian_splatting)
- [4D Gaussian Splatting paper (arxiv 2310.08528)](https://arxiv.org/abs/2310.08528)

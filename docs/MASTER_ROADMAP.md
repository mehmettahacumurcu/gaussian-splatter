# Master Roadmap — 4DGS Studio (Eylül 2026 sonrası)

> **Bağlam**: 70+ task tamamlandı, banana_high_v3 50k iter çalışıyor.
> Architectural foundation sağlam (Per-Gaussian Fourier, depth align, NaN
> guards, hard caps). Motion sayısal olarak gerçekleşiyor (Δpos artıyor)
> ama görsel olarak subtle. Bu doc kalan tüm yapılacakları priority sırasında
> listeler.

---

## 🔴 KRITIK — şimdi acil

### #1: Logger bug — dosyaya yazmıyor (sometimes)

**Belirti**: Frontend Analytics tab data gösterirken, Linux'tan dosya 0 byte gözüküyor.
Sonradan dolu görünüyor. Fronted'i okurken backend'in yazdığı API endpoint'inden
data alıyor — yani işlemde memory'de var. Ama disk yazma asenkron / gecikmeli.

**Hipotezler**:

1. **Windows file buffering geç flush**: Python `open(path, "w", buffering=1)`
   line-buffered ama Windows OS-level write-back cache var. SSD'ye geç yazıyor.
   Linux mount Windows'tan okuyor ama henüz flush edilmemiş satırları görmüyor.

2. **Re-submission truncate**: Aynı scene adıyla resubmit edildiğinde `"w"` mode
   dosyayı sıfırlar. Eski training thread'i kendi handle'ını kullanmaya devam
   eder. Sonuç: yeni run'ın init'i eski run'ın output'unu siliyor.

3. **Linux mount cache lag**: Linux `/sessions/.../mnt/...` Windows path'ine
   bind mount. Write-back cache Linux'a delay'le sync oluyor. Test: `fsync`
   atılırsa hemen mi gözükür?

**Fix planı**:

- `"w"` → `"a"` (append mode) — yeni run varsa session'lar separator ile ayır
- Periodic `os.fsync(fd)` her 100 metric'te bir
- atexit handler `proper_close()` — graceful shutdown'da flush garanti
- Re-submission detection: aynı scene + ongoing run = block veya queue
- Fallback: in-memory ring buffer + final dump

**Tahmini iş**: 2 saat
**Priority**: Yüksek (analytics güvenilir olmalı)
**Task ID**: yeni — eklenecek

---

### #2: Push beklemede

`feat/per-gaussian-fourier` branch'inde 70+ commit'lik değişiklik. v3.7.x'lar
push edilmedi henüz. **Banana_high_v3 bittiğinde push at**:

```cmd
git add -A
git commit -F COMMIT_MSG.txt   # COMMIT_MSG güncellenmeli
git push
```

COMMIT_MSG.txt güncelleme: v3.7.5, v3.7.6, ultra v2, high preset, depth align
bug fix, init subsample, vs.

**Tahmini iş**: 30 dk (commit message + push)
**Priority**: Yüksek (kod kaybı riski)

---

## 🟠 TIER 1 — RAFT Optical Flow Loss

> **En yüksek motion impact**. CoTracker 8GB VRAM'de fail oluyor, dense flow
> alternatifi. Detaylı plan: `docs/TIER1_TIER2_ROADMAP.md`.

### Quick recap

- **Sorun**: Track loss ~0 → motion direction sinyali yok → recon-driven
  gaussian motion chaotic
- **Çözüm**: RAFT model frame pair'ler arası dense optical flow → her pixel
  motion vector → trainer'da gaussian projeksiyon ile compare

### Phase breakdown

| Phase | İş | Süre |
|---|---|---|
| A | `optical_flow.py` — RAFT preprocessing | 3 saat |
| B | Pipeline Faz 3d | 1 saat |
| C | Trainer integration (loss, sampling, projection) | 5 saat |
| D | Config + API + Frontend override | 2 saat |
| E | Validation (smoke + full) | 6 saat |
| **Toplam** | | **~17 saat** |

### Beklenen etki

- Δpos mean **+50-100%**
- Motion direction accuracy: %30 → **%80+**
- Visible motion: subtle → **smooth + directional**

**Task ID**: #70
**Priority**: KRİTİK — bir sonraki büyük iş

---

## 🟡 TIER 2 — Static/Dynamic Gaussian Separation

> Detaylı plan: `docs/TIER1_TIER2_ROADMAP.md`.

### Recap

- Banana sahnesinin %80'i statik (masa, plaka, sabit objeler)
- Şu an her gaussian her iter'de motion compute → static jitter, wasted compute
- Mask projection ile her gaussian'ı static/dynamic etiketle, static'leri **dondur**

### Beklenen etki

- Compute %30 azalır (1.4× hızlı training)
- Static frames PSNR 22→25+ (jitter elimine)
- Dynamic motion 5× sharper (concentrate on object)

### Phase breakdown

| Phase | İş | Süre |
|---|---|---|
| A | Mask projection at init | 3 saat |
| B | Trainer respect flag | 2 saat |
| C | Density control sync | 3 saat |
| D | Optimizer mask | 2 saat |
| E | Periodic re-evaluation | 2 saat |
| F | Visualization | 2 saat |
| **Toplam** | | **~14 saat** |

**Task ID**: #71
**Priority**: Yüksek — Tier 1 sonrası ikinci büyük iş

---

## 🟢 TIER 0 — Motion Visualization Tools (hızlı win)

> Yeni tier — Tier 1/2'den önce yapılabilir, **mevcut motion'u görmek**
> için araçlar. Düşük effort, immediate value.

### #T0.1: `scripts/motion_diff.py`

Tek script, frame 0 ve frame N'i alır, motion'u görselleştirir:

```python
# Usage
python scripts/motion_diff.py banana_high_v3 --start 0 --end 45

# Output
# motion_diff_0_to_45.png — gaussian'ları renklendirir:
#   gri = static (Δpos < threshold)
#   yeşil = orta motion
#   kırmızı = yüksek motion
# Render edip PNG yazar.
```

**Tahmini iş**: 2 saat
**Etki**: "Motion var mı yok mu?" sorusunu kesin cevaplar
**Priority**: Orta — Tier 1 başlamadan önce yararlı

---

### #T0.2: Motion magnification toggle (viewer)

Trainer çıktısını değiştirmeden, **viewer'da Δpos × N** magnify:

```typescript
// SplatViewer.tsx
<select value={motionMagnify}>
  <option value={1}>1× (real)</option>
  <option value={2}>2×</option>
  <option value={5}>5×</option>
  <option value={10}>10×</option>
</select>
```

**Implementation**: 
- PLY'leri yüklerken Fourier coeffs'ı multiply et (eğer PLY içinde varsa)
- Veya: server-side regenerate PLY with magnified motion

**Backend implementation** (server-side approach):
- Yeni endpoint: `/jobs/<id>/ply/<idx>?motion_scale=5`
- Recovery script benzeri logic — checkpoint'ten model rebuild → multiply
  fourier_coeffs by N → re-export PLY

**Tahmini iş**: 4 saat
**Etki**: "Motion 1× görünmüyor, 5× ile bak" diagnostics
**Priority**: Orta — demo için faydalı

---

### #T0.3: Motion trail overlay (viewer)

Frame T'de, frame 0..T arası tüm pozisyonları **stroke** olarak çiz:

```
Static gaussian:  o (tek nokta)
Dynamic gaussian: o━━━>o (motion trail)
```

**Implementation**: Three.js `Line2` ile her dynamic gaussian için trace.

**Tahmini iş**: 4 saat
**Etki**: Animation history at-a-glance
**Priority**: Düşük — nice-to-have

---

## 🔵 TIER 3 — Multi-View Supervision (vrig datasets)

### Concept

vrig-peel-banana, vrig-3dprinter etc. **çift kameralı** validation rig
datasetler. Aynı timestep, 2 farklı view → güçlü 3D motion sinyali.
Bizim pipeline tek video stream gibi treat ediyor → multi-view kaybı.

### Implementation phases

| Phase | İş | Süre |
|---|---|---|
| A | HyperNeRF metadata loader (multi-cam) | 4 saat |
| B | Pipeline multi-stream extract | 3 saat |
| C | Trainer simultaneous multi-view loss | 6 saat |
| D | Validation | 4 saat |
| **Toplam** | | **~17 saat** |

### Beklenen etki

- Triangulation-grade motion direction (geometry-grounded)
- Per-frame, per-view consistency loss
- 3D motion direction much better than RAFT 2D-only flow

**Priority**: Tier 1 + 2 sonrası
**Task ID**: yeni — eklenecek
**Risk**: Medium-High (data loader complex)

---

## 🟣 TIER 4 — Architectural alternatives (uzun vade)

### #T4.1: SC-GS Control Points (instead of per-gaussian Fourier)

Şu anki: 16k × 60 = 960k motion params (per-gaussian).
SC-GS: 100 control point × 60 = 6k motion params (160× az).
Her gaussian KNN'le en yakın 4 control'ün motion'unu blend.

**Avantaj**:
- Param 160× az, training 2× hızlı
- Spatial coherence native (yakın gaussian benzer motion)
- Motion smooth ve coherent

**Dezavantaj**:
- 1-2 günlük refactor
- Density control daha karmaşık (control points dinamik mi static mi?)

**Priority**: Tier 1+2 motion problemini çözse SC-GS gerekmez.
Eğer Tier 1+2 yetersiz kalırsa düşün.

---

### #T4.2: D-NeRF dataset desteği

D-NeRF synthetic, perfect motion ground truth.
Format farklı: multi-view per timestep (vs HyperNeRF camera trajectory).

**Conversion needed**:
- Reorganize multi-view → single-view pseudo-trajectory
- Or: native multi-view loader (Tier 3 ile sinerjik)

**Tahmini iş**: 1 gün

**Priority**: Düşük — synthetic data demo amaçlı, real-world test'in yerini almaz.

---

## 🌍 OPERASYONEL (sistem geliştirme)

### #O1: Cloud GPU integration

Daha önce planı bırakıldı (`docs/CLOUD_SETUP.md` planlama doc'u). Tier 1+2
büyük training run'ları için tekrar gündeme alınmalı.

**Önerilen yaklaşım**: RunPod (ucuz, basit) veya Modal.com (Python SDK).

**Quick option**: `scripts/modal_run.py` — local'de geliştir, cloud'da train.

**Priority**: Tier 1+2 işbaşına çıktığında değerlendir
**İş**: 1-2 gün setup

---

### #O2: Logger redesign

Yukarıdaki logger bug fix'ten sonra, daha sağlam bir logger arch:

- **SQLite-based metrics store**: dosya yerine .db, append-safe + queryable
- **Streaming protocol**: WebSocket ile frontend'e direkt push (refresh gerek yok)
- **Multi-run comparison**: birden fazla run'ı yan yana chart'la
- **Hyperparameter sweep**: aynı sahne farklı param'larla — yan yana karşılaştır

**Priority**: Düşük — mevcut logger fix'le yeter
**İş**: 2-3 gün

---

### #O3: Config presets registry

Şu an presets backend kod'unda hardcoded (smoke, micro, full, high, ultra, cloud).
JSON registry ile dinamik preset yönetimi:

```yaml
# presets/banana_optimized.yaml
n_iters: 60000
image_resolution: [800, 450]
density_end_iter: 40000
...
```

Frontend dropdown'dan preset seç → backend yükle.

**Priority**: Düşük — şu anki preset count manageable
**İş**: 4 saat

---

### #O4: Multi-scene job batching

Şu an tek seferde 1 job. Birden fazla scene'i sırayla işle:

```
[1] cookie + High preset
[2] chickchicken + High preset (queued)
[3] banana + Ultra preset (queued)
```

Gece batch çalıştır, sabaha 3 sonuç bul.

**Priority**: Orta — productivity için faydalı
**İş**: 1 gün

---

## 🎨 KULLANICI DENEYİMİ

### #UX1: Viewer karşılaştırma modu

İki sahneyi yan yana göster:
- Sol: `banana_high` v1
- Sağ: `banana_high_v3`
- Senkron timeline scrub

Bug ayıklama ve karşılaştırma için kritik.

**Priority**: Orta
**İş**: 4 saat

---

### #UX2: Tauri folder picker

Daha önce task #36 olarak planlandı, deferred. Tauri dialog plugin install,
"Diskten yükle" butonunu native dialog ile yap.

**Priority**: Düşük — manuel path girme yeterli
**İş**: 2 saat

---

### #UX3: Real-time preview during training

Frontend training boyunca bir sample frame render gösterir:
- Backend her 5000 iter'de geçici PLY çıkar
- Frontend yükle, mini-viewer
- Eğitim devam ederken görsel ilerleme

**Priority**: Düşük — analytics tab zaten yeterli  
**İş**: 6 saat

---

## 📊 PERFORMANCE OPTIMIZATIONS

### #P1: Mixed precision training

```python
with torch.cuda.amp.autocast():
    loss = compute_loss(...)
scaler.scale(loss).backward()
```

VRAM ↓ 30-40%, hız ↑ 1.5-2× (Ampere kart için).

**Risk**: Numeric stability — gsplat fp16 destekliyor mu test edilmeli.
**Priority**: Düşük (3060 Ti FP16 has limited benefit)
**İş**: 1 gün

---

### #P2: gsplat compile time fix

Her backend restart sonrası gsplat 1-2 dk compile ediyor. Cache fix:
- Pre-compile cache
- TORCH_CUDA_ARCH_LIST env var (3060 Ti = sm_86)

**Priority**: Orta — productivity
**İş**: 2 saat

---

### #P3: Async PLY export

Şu an export training sonrası senkron, 60-120 PLY sırayla 2-5 dk. 
Async + progress reporting:
- Background thread her PLY yazımını paralel
- Frontend'e progress event

**Priority**: Düşük — sadece final 2 dk
**İş**: 3 saat

---

## 🧪 RESEARCH / EXPERIMENT

### #R1: Adaptive frame sampling

Motion peak frame'lerinde dense sampling, statik bölgelerde sparse:
- Optical flow magnitude → sampling weight
- Motion'lu frame'ler training'de daha sık çekilir

**Effect**: motion learning'in iter budget'ı motion'a yoğunlaşır
**İş**: 1 gün

---

### #R2: Curriculum on time horizon

Önce sadece frame 0..N/4, sonra 0..N/2, sonra 0..N:
- Network önce kısa motion, sonra uzun
- Gradient hierarchy daha düzgün

**İş**: 4 saat
**Risk**: Convergence yavaşlayabilir

---

### #R3: Velocity-based motion prior

Δpos yerine velocity v(t):
- pos(t) = pos(0) + ∫v(τ)dτ
- v Fourier serisi
- Doğal smoothness (1st derivative bounded)

**İş**: 1-2 gün — büyük refactor
**Effect**: Smoother trajectories, fewer high-freq artifacts

---

## 📋 ÖNCELİKLENDİRİLMİŞ TIMELINE

### Hafta 1 (bu hafta)
1. **Banana_high_v3 bitsin** (ETA bu akşam 18:30)
2. **Logger bug fix** (#1, 2 saat)
3. **Push** (#2, 30 dk)
4. **Tier 0 motion_diff script** (#T0.1, 2 saat) — visualize current motion

### Hafta 2
5. **Tier 1 RAFT** (#70, 17 saat = 2-3 iş günü)
   - Phase A-B: preprocessing + pipeline
   - Phase C: trainer integration
   - Phase D-E: API + UI + validation

### Hafta 3
6. **Tier 1 + Tier 2 ortak validation** (full run karşılaştırma)
7. **Tier 2 Static/Dynamic** (#71, 14 saat)
8. **Tier 0 motion magnification** (#T0.2, 4 saat) — demo için

### Hafta 4
9. **Tier 1+2 final benchmark** — banana, chickchicken, cookie hep
10. **Tier 3 multi-view** kararı (gerekiyor mu?)
11. **Operational items** (logger redesign, multi-job batch)

### Sonra
- **Tier 4 SC-GS** (gerekirse)
- **Cloud integration** (Tier 1+2 sonrası productivity için)
- **D-NeRF support**
- **UX improvements**

---

## 🎯 BAŞARI KRİTERLERİ

### Tier 1 sonrası (2-3 hafta sonra)
- [ ] Banana_high run'da Δpos mean ≥ 0.7
- [ ] Motion direction qualitatively correct (manual eye check)
- [ ] PSNR ≥ 24

### Tier 2 sonrası (3-4 hafta)
- [ ] Static frames jitter-free
- [ ] Dynamic motion sharply localized
- [ ] PSNR static frames ≥ 28

### Combined (ay sonu)
- [ ] **Visible motion** in viewer when scrubbing 60-frame animation
- [ ] At least one demo-worthy run (banana veya chickchicken)
- [ ] Static rendering quality competitive with INRIA static 3DGS

---

## 💭 OPEN QUESTIONS

1. **Motion gerçekten subtle mi yoksa visualization sorunu mu?**
   - Tier 0 motion_diff script bunu kesin cevaplayacak
   - Eğer motion büyük ama görünmüyorsa → viewer issue
   - Eğer motion gerçekten küçük → Tier 1 RAFT şart

2. **Custom dramatic dataset gerekli mi?**
   - Telefonla 5-10 sn motion video çekmek
   - HyperNeRF subtle vs custom dramatic — hangi data ile demo
   - Bu sıralama'da TIER 1+2 sonrası karar

3. **Cloud GPU şart mı?**
   - 3060 Ti'da High preset 3 saat — manageable
   - Ultra preset 6-9 saat — gece çalışsın
   - Cloud sadece **iterate hızı** için fayda
   - Karar: Tier 1+2 başarılı olursa cloud gerek, yoksa lokal yeter

4. **vrig multi-view fundamental mı yoksa nice-to-have mı?**
   - Tier 3 gerek mi? Önce Tier 1+2 sonucunu gör
   - Eğer Tier 1+2 yeterli motion verirse Tier 3 unnecessary

---

## 🗂 İlgili dokümanlar

- `docs/TIER1_TIER2_ROADMAP.md` — Tier 1 ve Tier 2 detaylı planı
- `docs/MOTION_PLANNING_DEEP.md` — Motion subtle olmasının matematiksel sebebi + dataset stratejisi
- `docs/DEFORMATION_ARCHITECTURE_IDEAS.md` — 7 mimari yaklaşımın karşılaştırması (SC-GS, control points, etc.)
- `docs/WINDOWS_SETUP.md` — kurulum + troubleshooting

---

_Son güncelleme: 2026-04-25 (banana_high_v3 ortası)_
_Tüm task ID'leri ve süre tahminleri yaklaşıktır._

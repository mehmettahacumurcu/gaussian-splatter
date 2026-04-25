# Motion Geliştirme — Derin Planlama

> **Bağlam**: 50+ run yapıldı, motion mevcut runlarda matematiksel olarak ölçülebilir
> ama gözle görülemiyor. Bu doküman PROBLEMI deeply analiz eder ve mimari değişiklik
> öncesi alternatif yolları, frame rate tradeoff'larını, data seçim stratejisini
> ve görselleştirme araçlarını planlar.

---

## 🎯 GERÇEK problem — matematik ile

### Mevcut motion'un boyutu

Banana_high v1 ölçümü:
- Scene extent: **108 birim** (COLMAP world coords)
- Δpos mean: **0.83 unit**
- Oran: 0.83 / 108 = **%0.77** — sahnenin %1'inden az

### Bu visually neye karşılık geliyor?

Tauri viewer:
- Pencere genişliği: ~1280 px
- Sahne render genişliği (default zoom): pencere'nin ~%60'ı = **768 px**
- Motion'un pixel karşılığı: %0.77 × 768 = **~6 pixel**

**6 pixel**, 60-90 frame animasyonda yayılınca:
- Frame başına: ~0.1 pixel
- İnsan gözü: hareket olarak algılamaz (jitter ile karışır)

### Karşılaştırma — D-NeRF JumpingJacks

D-NeRF (synthetic, perfect motion):
- Scene extent: ~3 unit (compact)
- Insan figürü tüm scene'i kaplar
- Δpos amplitude (kol kaldırma): ~1 unit = **%33 of scene**
- Pixel: %33 × 768 = **256 pixel motion**!

D-NeRF'ün motion görünür olmasının matematiksel sebebi: **physical_motion / scene_extent oranı**.

### HyperNeRF datasetlerinin zorluğu

HyperNeRF tasarımı:
- Kamera **orbit** yapar (geniş scene_extent yaratır)
- Object motion **tabletop scale** (5-10 cm physical)
- Oran: 0.05m / 1m scene = **%5** physical
- Ama camera arc nedeniyle scene_extent'i sahte büyütüyor → effective motion oranı **%0.5**

**Sonuç**: Mimari ne kadar iyi olsa, HyperNeRF orbital scenes'lerde motion daima subtle olacak.

---

## 🔧 Çözüm yolları (mimari değişiklik dışında)

### 1. Frame rate / temporal density

#### Mevcut durum
- HyperNeRF source: 30 fps (banana 343 frame)
- Pipeline ffmpeg extract: **fps=10** → ~115 frame
- Pipeline kullanıyor: 115 frame, 1/115 = %0.87 of timeline per frame
- Δpos per frame: 0.83 / 115 = **0.007 unit** ortalama

#### Daha yüksek fps deneyelim
- fps=15 → 172 frame, daha smooth ama frame başına motion daha küçük
- fps=20 → 230 frame, çok smooth ama her frame çok benzer
- fps=30 (full) → 343 frame, COLMAP yorulur ama maximum temporal info

#### Daha düşük fps?
- fps=5 → 57 frame, frame başına Δpos 2× → daha "choppy" ama her frame daha farklı
- fps=3 → 34 frame, çok az frame, motion stretching loss

#### Recommendation
**fps=10 sweet spot** mevcut config için. Düşürmek smooth animation kaybeder, yükseltmek COLMAP/foundation süresini şişirir.

**Ama**: Eğer düz fps yerine **adaptive sampling** olsa — motion'lu region'larda yüksek fps, statikte düşük fps. Bu mimari değişiklik (out of scope).

#### Frame count'u DAHA AZ alarak motion görünür yapmak?
İlginç fikir: Sadece "kritik 20-30 frame" al (motion peak'leri). Δpos cumulative 0.83 birden 1.5'a çıkar görünüm değişimi büyür.
- Pros: Visible motion artırır
- Cons: Smooth animation kaybeder, scrubbing kötü
- **Not for production**, sadece demo için

### 2. Dataset seçimi — en büyük etken

Motion görünürlüğü için sıralı liste:

#### Tier S (en büyük motion):
- **D-NeRF synthetic** — JumpingJacks, T-Rex, HellWarrior, Hook, Lego, Bouncing
  - Synthetic, perfect ground truth motion
  - Scene compact, motion amplitude %30-50 of scene
  - **DOWNSIDE**: multi-view per timestep format, conversion gerekir

#### Tier A (büyük real motion):
- **HyperNeRF vrig-3dprinter** — mekanik motion, dramatic
- **HyperNeRF vrig-peel-banana** — şu an kullanıyoruz (orta seviye)
- **Custom phone video** — kontrolünde, dramatic seçebilirsin

#### Tier B (orta motion):
- **HyperNeRF interp-* datasets** — interp-aleks-teapot, interp-chickchicken
  - Designed for interpolation
  - Daha keskin motion than vrig

#### Tier C (subtle motion — şu anki test):
- HyperNeRF cookie, chickchicken, cutlemon
- Object motion small relative to scene

#### Hızlı kazanım: Custom phone video
**Ne çekersin?**
- **Top yuvarlama** (masada): tüm scene yatay 1m, top motion 0.5m = **%50 of scene** ✓✓✓
- **Pendulum** (ipte sallanan top): periyodik, Fourier'ın ideal kullanımı
- **Kart atma** (deste'den masaya tek kart): dramatic, izole motion
- **Su dökme** (bardağa): sürekli, akış benzeri (zor ama dramatic)
- **El sallaması** (sandalyede oturan kişi): büyük açısal motion

**Recording rules** (önemli):
- Sabit tripod, kamera **statik** (çok az zoom drift)
- 5-10 saniye süre
- 30+ fps
- İyi ışık, yumuşak gölgeler
- Background dokulu (COLMAP feature için)
- Subject scene'in %30-60'ını kaplasın

### 3. Görselleştirme araçları (mevcut motion'u görmek)

Şu an motion ölçülüyor (Δpos 0.83) ama görünmüyor. **Bakış açısını değiştirelim**:

#### A) Motion diff visualizer (script — mevcut)
`scripts/motion_diff.py` (yeni, planlamada):
- Frame 0 ve frame 45 PLY'lerini al
- Hareket eden gaussian'ları **kırmızı**, statikleri **gri** renklendir
- Tek bir PNG render — motion regions immediately visible

#### B) Motion magnification (in viewer)
Trainer çıktısını değiştirmeden, **viewer'da Δpos × N** uygula:
- Toggle: "Motion 1× / 2× / 5× / 10×"
- Fourier coeffs × N (multiplied at render time)
- Realistic değil ama **diagnostic** — network ne öğrendi, görünür yap

#### C) Motion trail overlay
Frame T'de, frame 0..T arası tüm pozisyonları stroke olarak çiz:
- Dynamic gaussian'ların izi (trace) görülür
- Static'ler tek nokta
- Animation history at-a-glance

#### D) Side-by-side comparison
- Sol: input video frame N
- Sağ: rendered frame N (deformation applied)
- Time scrub her ikisini senkron

### 4. Multi-view supervision (vrig datasets)

**Bilmediğimiz şey**: vrig-peel-banana **çift kameralı**!
- "vrig" = validation rig
- 2 kamera senkron yakalama
- Bizim pipeline tek video stream gibi muamele ediyor → multi-view sinyali kaybediyoruz

**Düzgün kullanırsak**:
- Aynı timestep, 2 farklı view → güçlü 3D motion sinyali
- "Triangulation" gibi: 3D motion 2 view'den çıkarılır
- Her gaussian için motion direction çok daha doğru

**İmplementation effort**: Yüksek (data loading + camera handling + loss formulation). Tier 3 mertebesinde.

---

## 🌙 Bu akşamki full run önerisi

### Mevcut durum (banana_high_v3 ~1.5h sonra biter)
- Track loss: 0 (CoTracker fail)
- Motion: subtle (banana sahnesinin doğal limiti)
- Static: clean (yeni scale clamp)

### Tonight's full run options

#### Option A: **Aynı preset, farklı dataset** (önerilen)
- HyperNeRF chickchicken'ı yeniden dene (chickchicken_high_v2 olarak)
- Avantaj: Cookie/banana'dan farklı motion karakteri
- Tahmin: Δpos benzer büyüklük (subtle)
- **Düşük yeni-bilgi getirisi**

#### Option B: **Custom phone video — DRAMATIC motion** (en yüksek bilgi)
- 5-10 saniye, sabit kamera, dramatic motion
- Top yuvarlama, kart atma, salınan top
- Pipeline'a doğrudan ver
- **Tahmin: Δpos %30-50 of scene (D-NeRF mertebesi)**
- **VİZÜEL motion KESİN görünür**
- Risk: COLMAP custom video'da feature bulamazsa fail

#### Option C: **HyperNeRF vrig-3dprinter** (mekanik motion)
- 3D printer çalışıyor — robotic motion, predictable
- Indir et + MP4'e dönüştür + High preset
- Avantaj: scene_extent küçük olabilir (table-top printer)
- Tahmin: Δpos %5-15 of scene

#### Option D: **D-NeRF JumpingJacks** (motion gold standard)
- Synthetic perfect data
- BÜYÜK iş: D-NeRF format ≠ pipeline'ımıza uyumlu
- Multi-view per timestep, single-camera kabul etmiyor
- Conversion script yazmak gerekir (4-6 saat iş)
- Bu akşam **uygun değil**

### Önerim: **Option B — kendin çek**

Karar gerekçesi:
1. **5 dakikalık çekim** + 30 sn upload + High preset (3-4 saat)
2. Sonuç bu gece bitebilir
3. **Visible motion garantili** (matematik açık)
4. Kendi data'n → dataset bias yok
5. Beğenmezsen başka çekim → iterate

### Custom video çekim rehberi (telefonla)

#### Setup
- Telefon **tripod / kitap stack** üzerinde sabit
- 30 fps minimum (60 fps ideal)
- Resolution **1080p** (4K overkill, COLMAP yavaşlar)
- Yatay (landscape) yönelim
- Yapay ışık veya pencere ışığı, **sabit** (titremeyen)
- Background DOKULU olsun (boş duvar değil — kitaplık, halı, mat panel)

#### Süre + frame
- **5-7 saniye** ideal (150-210 frame at 30fps)
- Pipeline fps=10 ile extract → 50-70 frame
- 8 saniye max (yoksa COLMAP zorlanır)

#### Motion seçimi
**Top 3 öneri**:

1. **Top yuvarlama (DRAMATİK)**
   - Düz masa, kamera masaya bakıyor
   - Tenis topu masanın bir ucundan ittir → diğer uca gider
   - 4-5 saniyelik motion
   - %50+ scene motion ✓

2. **Kart atma (DRAMATİK + İZOLE)**
   - Iskambil destesi elinde
   - Tek tek 5-10 kart at masaya
   - Her kart yeni motion event
   - Multi-stage motion ✓

3. **Pendulum (PERIYODİK — Fourier ideal)**
   - İpten asılan top / oyuncak
   - Sallan, kameraya yakın+uzak
   - **Periyodik motion** = Fourier mimari **ideal eşleşme**

#### Çekim sonrası
1. Dosyayı `data\<scene>\video.mp4` olarak yerleştir
2. **High preset** ile submit
3. Foundation atla = ☐ unchecked
4. 3-4 saat sonra dön

### Eğer custom çekim olmazsa fallback

#### Plan B: cookie ile **chickchicken_high_v3** (cookie değil chickchicken)
- chickchicken'ın motion'u banana'dan farklı (cutting motion, lokal)
- Bu kez track loss çalışsın diye:
  - Frontend'i kapalı tut (Tauri 1-2 GB kazan)
  - Backend env var ile başla (yeni `PYTORCH_CUDA_ALLOC_CONF` aktif olsun)
- 3 saat sonra sonuç

---

## 📊 Frame rate analiz (derinlemesine)

### Tradeoff matrisi

| fps | Frame count (343 source) | COLMAP süresi | Motion per frame | Smooth animation | Visible jump |
|---|---|---|---|---|---|
| 5 | ~57 | 15 dk | 1.7% | Choppy | Yes (her frame'de görünür) |
| **10 (current)** | **~115** | **30 dk** | **0.87%** | OK | Subtle |
| 15 | ~172 | 60 dk | 0.58% | Smooth | Çok subtle |
| 20 | ~230 | 90 dk | 0.43% | Çok smooth | İmkansız |
| 30 | 343 | 2-3 saat | 0.29% | Excellent | Görünmez |

### "Adaptive sampling" düşüncesi (gelecek)

Mevcut: uniform fps, her frame eşit aralık.

Daha iyi olabilir mi:
- Motion peak frame'lerinde fps=20 (smooth)
- Statik frame'lerde fps=5 (overhead az)
- Otomatik: optical flow magnitude ile karar ver

**İlk yapma sırasında değil**, optimize aşaması sonrası eklenebilir.

### Pratik öneri

**fps=10 koru** (default). fps değişimi bu projedeki motion problemine çözüm değil — **dataset karakteri** ana belirleyici.

---

## 🧪 Tonight's full run — çekirdek karar

Eğer custom video çekersen → **Option B**, en yüksek visible motion garantisi.

Eğer çekmezsen → **Option C** (vrig-3dprinter download + MP4 + run) — 30 dk indirme + 4 saat training.

Her iki durumda kod değişikliği gerek yok. Bu akşam **bilgi getirisi maksimum** sonuç.

---

## 🔮 Yarın ve sonrası

### Tier 0 (yarın sabah, 30 dk)
**Mevcut motion'u GÖRMEK için tool yaz**:
- `scripts/motion_diff.py` — frame 0 vs frame N karşılaştırma
- "Motion magnification" viewer toggle (Fourier × 5×)
- Sonra: "ah motion VAR, sadece scrubbing'de görmedik"

### Tier 1 (RAFT, hafta sonu sonu, 2-3 gün)
- [Tier 1 doc'da detaylı](TIER1_TIER2_ROADMAP.md)
- Önemli: Motion **direction** doğrulanır

### Tier 2 (Static/Dynamic, sonraki hafta, 1.5 gün)
- Static jitter eliminate
- Dynamic motion concentrate

### Tier 3 (Multi-view supervision, daha sonra)
- vrig dataset'ler için dual-camera
- Mimari değişiklik orta-yüksek

---

## 💭 Felsefi not

**Bilmemiz gereken**: 4D Gaussian Splatting'in **doğal sınırları** var.
- Static GS (3D) her zaman dramatic
- 4D dynamic = static + delta. Delta küçükse, görsel olarak subtle.
- HyperNeRF datasetleri **subtle motion** içerir (bilim için, demo için değil)
- Demo için **dramatic source** seçmek SOTA paper'ların ortak stratejisi

4D-GaussianSplatting paper'ı bu yüzden **JumpingJacks** ile gösterir, **cutlemon** ile değil.

Bizim sonuçlarımız aslında **literatür ile tutarlı** — sadece dataset seçimi yanıltıcıydı.

---

## ✅ Karar matrisi

Bu gece ne yapacağız (1 saat içinde başlatılabilir):

| Seçenek | Süre | Visible motion | Yeni bilgi | Difficulty |
|---|---|---|---|---|
| A. Aynı banana, farklı preset | 3 saat | Subtle (banana limiti) | Az | 0 |
| **B. Custom phone video** | **3 saat + 5 dk çekim** | **Yüksek (dramatic)** | **Çok** | 0 |
| C. HyperNeRF vrig-3dprinter | 3.5 saat (download dahil) | Orta-yüksek (mekanik) | Orta | 0 |
| D. Plan ve bekle | 0 saat | — | Plan refining | 0 |

**Recommendation: B**.

Çekim hazırla, akşam yemeğine kadar pipeline'ı çalıştır, sabaha kadar bitir, **visible motion** ile uyan.

---

_Son güncelleme: 2026-04-25 (training ortası planlama)_
_Bağlı dokümanlar: TIER1_TIER2_ROADMAP.md, DEFORMATION_ARCHITECTURE_IDEAS.md_

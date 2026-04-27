# 4D Viewer Playback v2 — Production Solutions Research

> **Update:** Önceki araştırma (`VIEWER_PLAYBACK_RESEARCH.md`) 2026-04-25'teydi.
> 8 ay içinde 4DGS web ekosistemi dramatik genişledi — production-grade
> "YouTube gibi" 4D streaming çözümleri çıktı, WebGPU engines geldi,
> SuperSplat timeline ekledi.
>
> **Tarih:** 2026-04-26
> **Trigger:** Bizim "cached blob swap" pattern'i frame başına 50-150 ms,
> 10 fps playback için yeterli akıcılık vermiyor. Production siteler bu
> sorunu nasıl çözüyor?

---

## 1. Mevcut Limitasyonumuz

```
Mount  → 90 PLY blob fetch → RAM cache (~1.3 GB)
play   → her frame: addSplatScene(N+1) + removeSplatScene(N)  ~50-150 ms
sonuç  → "fotoğraf akışı", smooth 30fps DEĞİL, scrub OK ama auto-play kasıyor
```

**Temel mimari kısıtı:** her PLY frame **bağımsız bir 3DGS scene**. Her frame
swap'inde gauss buffer'ı yeniden parse edilip GPU'ya yükleniyor. Bu "static
3DGS × 90" yaklaşımı — gerçek 4DGS değil.

---

## 2. Production Landscape (2026 Q2)

### 🏆 Gracia (gracia.ai) — En Olgun Production Çözüm
**Ne:** "4D Gaussian Splatting infrastructure" startup, $1.7M seed.
"YouTube gibi 4DGS streaming" claim ediyor — sound + post-effects + relighting.

**Nasıl çalışıyor:**
- Custom 4DGS encoding format (tescilli)
- Server-side compression (bizim 970 MB → muhtemelen 30-100 MB)
- Browser-side **streaming decoder** (WebGL2 + custom shader)
- PlayCanvas engine ile entegre

**Demo:** [demo.gracia.ai/playcanvas.html](https://demo.gracia.ai/playcanvas.html) —
gerçek volumetric video sound'la birlikte browser'da. **Test ettim, akıcı.**

**Sınırlama:**
- Tescilli encoding/format → bizim PLY çıktımızı doğrudan beslemiyor
- API/license bilgisi yok (henüz public SDK)
- Format converter yazmamız gerek (PLY 90×→ Gracia format)

---

### 🚀 Visionary (Visionary-Laboratory) — En Hızlı Open Source
**Ne:** WebGPU + ONNX Runtime tabanlı, "World Model Carrier" platform.
Aralık 2025 release, açık kaynak (MIT-ish).

**Performans:**
- **2-16 ms/frame** on RTX 4090
- WebGL viewers'tan **135× hızlı**
- WebGPU compute shaders ile full GPU sort

**Mimari:**
- **Standardized "Gaussian Generator contract"** — plug-and-play algoritma
- 3DGS, MLP-based 3DGS, **4DGS**, Neural Avatars hepsi destekleniyor
- ONNX inference per-frame → neural deformation field browser'da çalışır
- Tek sayfa "click-to-run"

**Bizim durumumuza uyum:**
- 4DGS hot path: ✅ destekleniyor
- Per-frame ONNX inference: ✅ — bizim DeformationField'ı ONNX'e export edip
  browser'da çalıştırabiliriz! (Tier 4 senaryosu)

**Sınırlama:**
- Tauri WebView2 WebGPU desteği henüz default-disabled (Windows 11'de flag gerekli)
- ONNX export pipeline yazılması gerek

**Repo:** https://github.com/Visionary-Laboratory/visionary
**Paper:** arXiv:2512.08478

---

### 🎯 PlayCanvas + SuperSplat — En Olgun OSS Editor
**Ne:** PlayCanvas engine üzerine kurulu açık kaynak browser editor.
SuperSplat 1.13'te 4DGS timeline, 2.0'da Walk Mode + flythrough eklendi.

**Çalışma prensibi:**
- Numbered PLY sequence import (`frame_0001.ply` ... `frame_NNNN.ply`)
- Built-in timeline UI (play/pause/scrub)
- WebGL2 underlying renderer
- Plus: voxel-based collision system → "karakter gibi yürüme"

**Bizim case:**
- ✅ 90 PLY format direkt destekleniyor
- ✅ Bütün stack'i değiştirmek zorunda değiliz, sadece **frontend'i SuperSplat
  embed'iyle değiştirmek**
- ⚠️ İçeride SuperSplat hâlâ scene-swap pattern'i kullanıyor olabilir (kaynak
  kontrol edilmeli) — ama UI/UX ve dahili optimizasyonları olgun

**Spark.js'le birlikte SuperSplat'in modernize edilmiş v2'sini düşünüyorlar
(2026 roadmap).**

---

### ⚡ Spark.js (sparkjsdev/spark) — World Labs'in Çözümü
**Ne:** World Labs'in özel olarak 4DGS ve dynamic content için yazdığı
Three.js renderer. Spark 2.0 (Nisan 2026) en güncel.

**Kritik özellik:**
> "Spark started as an internal 3DGS renderer because existing web
> renderers had shortcomings, such as **inability to dynamically animate
> the splats (4DGS)**."

**4DGS desteği:**
- "**Interpolating between scanned 4DGS frames**" — explicit feature
- "Animated transitions, GLSL or computation graphs"
- Recoloring + opacity + SDF clipping per-splat

**Dyno sistemi (Spark 2.0):**
```js
// Bizim Fourier coeff'leri shader'a gönderebiliriz
splatMesh.objectModifier = dyno.add(
  splatPosition,
  fourierAnimation(time, fourierCoeffs, K)
);
```
Bu **statik tek bir gauss set** + **GLSL shader'da time-based deformation**.
PLY swap yok, GPU'da continuous animation.

**Bizim durumumuza tam uyum:**
- Backend zaten `fourier_pos_coeffs` üretiyor → Spark Dyno ile decode
- 90 PLY swap'i tamamen ortadan kalkar
- Sliding-time slider → continuous, smooth playback (60 fps native)

---

## 3. Akademik Background — Streaming/Compression Methods

Production siteler bunlardan beslenmiş. Anlamak faydalı.

### 4DGaussians (HustVL, CVPR 2024)
[guanjunwu.github.io/4dgs](https://guanjunwu.github.io/4dgs/) — bizim training
mimarimiz tam bu (HexPlane + MLP). Render real-time 82 fps @ 800×800 RTX 3090.

### Spacetime Gaussians (OPPO, CVPR 2024)
[oppo-us-research.github.io/SpacetimeGaussians-website](https://oppo-us-research.github.io/SpacetimeGaussians-website/) —
temporal opacity + parametric motion. Tek static gauss set, time-dependent
visibility. Compact representation.

### GIFStream (XDimLab, 2025)
[xdimlab.github.io/GIFStream](https://xdimlab.github.io/GIFStream/) — feature
stream + canonical space + deformation field. **30 Mbps real-time on 4090.**
"Immersive video" olarak tanımlanıyor. Compression'a en yakın production.

### StreamSTGS (2025)
[arXiv:2511.06046](https://arxiv.org/html/2511.06046) — spatial+temporal
gaussian grids → image/video compression. Network adaptive bitrate.

### Play4D (NVIDIA Research, 2025)
[research.nvidia.com/publication/2025-12_play4d](https://research.nvidia.com/publication/2025-12_play4d-accelerated-and-interactive-free-viewpoint-video-streaming-virtual) —
free-viewpoint video for VR + light field displays. Henüz code yok.

### RetimeGS (2026)
[arXiv:2603.13783](https://arxiv.org/html/2603.13783) — continuous-time
reconstruction, slow-motion. Mevcut 4DGS'lerin ghosting problemini çözüyor.

---

## 4. Çözüm Yolları (4 Tier)

### Tier 1 — Spark.js + Dyno Pattern (~1-2 hafta)
**Effort:** Orta. **Risk:** Düşük. **Etki:** Büyük (60 fps native playback).

**Plan:**
1. `frontend/src/components/SplatViewer.tsx` rewrite — `@mkkellogg`
   yerine `@sparkjs/spark` import
2. Backend export değişikliği:
   - 90 PLY yerine **tek statik gauss set + Fourier coeff binary**
   - `gaussians.ply` (~9 MB, time=0 frame'in tüm props)
   - `fourier.bin` (~30 MB, fp16, N×K×6)
3. Dyno shader yaz:
   ```glsl
   vec3 dpos = vec3(0);
   for (int k = 0; k < K; k++) {
     float phase = 2.0 * PI * float(k+1) * uTime;
     dpos += fourierA[k] * sin(phase) + fourierB[k] * cos(phase);
   }
   splat.center += dpos;
   ```
4. UI: continuous slider 0..1 (frame index değil, normalized time)

**Avantaj:**
- PLY swap tamamen ortadan kalkar
- Tek upload (~40 MB), browser'da unlimited time slider
- 60 fps native, smooth video
- Backend export 90× → 1× (5-10 dk export → 30 sn)

**Dezavantaj:**
- HexPlane MLP residual kaybedilir (sadece per-gaussian Fourier korunuyor).
  Çözüm: training sonu MLP(t)'yi K=12-16 Fourier'a "bake" eden script
- Spark API öğrenmek 2-3 gün

---

### Tier 2 — Spark.js + 90 PLY Loaded Once (~3-5 gün)
**Effort:** Düşük. **Risk:** Düşük. **Etki:** Orta (10-20 fps smooth).

Daha az radikal: 90 PLY'yi tek seferde yükle, opacity Dyno ile aktif olanı
göster. Bizim `setSplatSceneOpacity` ile aynı fikir ama Spark'ın GPU sort'u
sayesinde 90 simultane scene daha hızlı sort edilir.

**Plan:**
1. Spark.js'e geç, 90 PLY paralel yükle (tek API call)
2. Per-mesh opacity Dyno (current frame index = 1.0, diğerleri 0.0)
3. Frame slider → opacity uniform update (instant, GPU)

**Avantaj:**
- Backend değişikliği yok (mevcut PLY export çalışır)
- Implementation hızlı

**Dezavantaj:**
- 810 MB GPU RAM (90 × 9 MB) — 8 GB karta sığar ama tight
- Frame-frame interpolation yok (sadece discrete switch)
- Yine "snap to frame" hissi

---

### Tier 3 — Gracia Format (~1-2 ay)
**Effort:** Yüksek. **Risk:** Orta (kapalı format reverse-engineer). **Etki:** Production-grade.

Eğer Gracia bir SDK / open format publish ederse:
1. Backend'den 4DGS → Gracia format converter yaz
2. Frontend'e Gracia/PlayCanvas player embed
3. CDN streaming (sunucu lazım) ya da local file

**Avantaj:** Production-tested, sound+effects+relighting

**Dezavantaj:** Format kapalı, SDK çıkana kadar bekleme zorunlu. Kendi
infrastructure (compression backend) gerek.

**Şu anki status:** Gracia Mart 2025'te seed aldı, henüz public SDK yok.
2026 Q3-Q4 muhtemel. Şimdilik "watch only".

---

### Tier 4 — Visionary Engine Fork (~1 ay)
**Effort:** Yüksek. **Risk:** Orta. **Etki:** Maksimum.

Visionary'nin "Gaussian Generator contract"ına bizim trainer'ı entegre et.
Per-frame ONNX inference için **DeformationField'ı ONNX'e export**:

1. `backend/export/to_onnx.py` — DeformationField + Fourier decoder ONNX
2. Visionary fork → custom Generator class
3. Browser'da: tek statik gauss + ONNX inference per frame (~5 ms RTX 4090)

**Avantaj:**
- 135× WebGL'den hızlı
- Tam neural pipeline (HexPlane + MLP residual korunur)
- 4DGS kalitesini browser'da kayıpsız sergiler

**Dezavantaj:**
- Tauri WebView2 WebGPU flag gerek (Win11 chromium 121+)
- ONNX export validation karmaşık
- Visionary stable'a oturmadı henüz (Aralık 2025 release)

---

## 5. Decision Matrix

| Çözüm | Effort | Risk | FPS hedefi | PLY swap | Backend dokunma |
|---|---|---|---|---|---|
| **Mevcut (mkkellogg cached swap)** | 0 | 0 | 5-7 fps | Var | Yok |
| **Tier 1 — Spark Fourier Dyno** | 🟡 | 🟢 | **60 fps** | **Yok** | Export değişir |
| **Tier 2 — Spark opacity** | 🟢 | 🟢 | 15-25 fps | Var ama GPU | Yok |
| **Tier 3 — Gracia** | 🔴 | 🟡 | 60 fps | Yok | Format converter |
| **Tier 4 — Visionary** | 🔴 | 🟡 | **240+ fps** | Yok | ONNX export |

---

## 6. Önerilen Yol Haritası

### Sprint 1 (~1 hafta) — Hızlı POC
**Tier 2** ile başla:
- Spark.js'e geç (3-4 gün)
- 90 PLY → 90 SplatMesh + opacity toggle (1-2 gün)
- Frame slider'la test
- 15-25 fps gelir mi gör

Bu sırada:
- Spark Dyno API'sini deeply incele
- Tier 1 için PoC: 5-frame example ile Fourier shader test et

### Sprint 2 (~2 hafta) — Tier 1'e geçiş
- Backend export refactor:
  - `bake_fourier_from_mlp.py` script (MLP residual'ı Fourier K=16'ya fit et)
  - Yeni format export: `gaussians.ply` + `fourier.bin`
- Frontend Spark Dyno shader:
  - Per-gaussian Fourier decode
  - Continuous time slider
- 60 fps test

### Sprint 3 (gelecek) — Tier 4 değerlendirmesi
- Visionary stable v1.0 çıktığında
- ONNX export pipeline kurulup yoksa Tier 1 ile yetinilir

---

## 7. Hemen Yapılabilecek Düşük Maliyetli Test

**Önce production sitelerle CTRL+karşılaştırma:**

1. **Gracia demo'sunu aç:** [demo.gracia.ai/playcanvas.html](https://demo.gracia.ai/playcanvas.html)
   - Browser'da gerçek 4DGS streaming gör. **Bu bizim hedefimiz.**

2. **SuperSplat 4DGS timeline test et:** [superspl.at/editor](https://superspl.at/editor)
   - Numbered PLY sequence yükle (banana_static_max'in 90 PLY'ini)
   - Built-in timeline ile playback test et
   - SuperSplat akıcılık veriyorsa **mevcut bizim viewer'ın yetersizliği**
     altyapısal değil, implementation kalitesinde

3. **Visionary GitHub'ı incele:** [Visionary-Laboratory/visionary](https://github.com/Visionary-Laboratory/visionary)
   - Demo varsa aç
   - Repository read

**Bu 3 testin 30 dk içinde yapılması bize hangi tier'a gidileceğini netleştirir.**

---

## 8. Sources

### Production
- [Gracia AI](https://gracia.ai/) · [Demo](https://demo.gracia.ai/playcanvas.html) · [80.lv haber](https://80.lv/articles/playcanvas-can-now-load-gracia-s-4d-gaussian-splats) · [80.lv stream haber](https://80.lv/articles/gaussian-splatting-videos-can-now-be-streamed-like-regular-videos)
- [Visionary GitHub](https://github.com/Visionary-Laboratory/visionary) · [Paper arXiv:2512.08478](https://arxiv.org/abs/2512.08478) · [Hugging Face page](https://huggingface.co/papers/2512.08478)
- [Spark.js (World Labs)](https://sparkjs.dev/) · [GitHub](https://github.com/sparkjsdev/spark) · [Spark 2.0 blog](https://www.worldlabs.ai/blog/spark-2.0)
- [SuperSplat](https://superspl.at) · [PlayCanvas Walk Mode](https://blog.playcanvas.com/new-in-supersplat-walk-mode-streamed-lod-and-easy-upload/)

### Akademik (4DGS streaming/compression)
- [4DGaussians (HustVL, CVPR 2024)](https://guanjunwu.github.io/4dgs/) · [GitHub](https://github.com/hustvl/4DGaussians)
- [Spacetime Gaussians (OPPO, CVPR 2024)](https://oppo-us-research.github.io/SpacetimeGaussians-website/) · [GitHub](https://github.com/oppo-us-research/SpacetimeGaussians)
- [GIFStream (XDimLab, 2025)](https://xdimlab.github.io/GIFStream/)
- [StreamSTGS (2025)](https://arxiv.org/html/2511.06046)
- [4D-MoDe (2025)](https://arxiv.org/html/2509.17506)
- [HPC streaming compression (2026)](https://arxiv.org/html/2602.00671)
- [RetimeGS continuous-time (2026)](https://arxiv.org/html/2603.13783)
- [Play4D NVIDIA Research (2025)](https://research.nvidia.com/publication/2025-12_play4d-accelerated-and-interactive-free-viewpoint-video-streaming-virtual)
- [Splat4D Visual-AI](https://visual-ai.github.io/splat4d/)
- [Instant4D (arXiv 2510.01119)](https://arxiv.org/html/2510.01119v1)
- [Temporal Smoothness 4DGS (2025)](https://arxiv.org/html/2507.17336)

### WebGPU alternatifler (Tier 4 background)
- [Visionary efficient-coder review](https://www.xugj520.cn/en/archives/webgpu-3d-gaussian-splatting-browser.html)
- [MarcusAndreasSvensson/gaussian-splatting-webgpu](https://github.com/MarcusAndreasSvensson/gaussian-splatting-webgpu)
- [Scthe/gaussian-splatting-webgpu](https://github.com/Scthe/gaussian-splatting-webgpu)

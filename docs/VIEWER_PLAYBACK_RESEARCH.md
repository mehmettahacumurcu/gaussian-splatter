# Viewer Playback Smoothness — Araştırma Notları

> **Mod:** Sadece research, hiçbir implementation yok.
> **Tarih:** 2026-04-25
> **Bağlam:** v3.7.9 viewer rewrite'tan sonra motion artık doğrulandı (banana,
> chickchicken, cookie hepsinde net hareket görünüyor) ama frame swap pattern
> ~50-150 ms/frame nedeniyle 10 fps playback "fotoğraf akışı" gibi hissediyor,
> gerçek video akıcılığı yok. Bu doküman alternatif yaklaşımları kıyaslıyor.

---

## 1. Mevcut Durum (v3.7.9, baseline)

**Pattern:** Cached blob swap.

```
Mount  → frame 0 fetch + addSplatScene (blocking)
       → background prefetch (3 paralel, currentFrame'den genişleyerek)
currentFrame değişir
       → addSplatScene(yeni)        ~30-100 ms
       → removeSplatScene(0) (eski) ~10-50 ms
       → loadedFrame = target
```

**Ölçülen latency:** Tek swap ~50-150 ms (CPU-side scene tear-down + worker upload).
**Sonuç:** Scrub akıcı ama auto-play 10 fps'te belirgin gecikme/seğirme.

**Neden yavaş:** `@mkkellogg/gaussian-splats-3d` her `addSplatScene` çağrısında
PLY'yi parse edip GPU buffer'a re-upload ediyor. Bizim case'imizde 90 frame
× ~36k gaussian × ~250 byte = ~810 MB arası worker traffic.

---

## 2. Seçenek Matrisi

| # | Yaklaşım | Effort | Risk | Beklenen Akıcılık | Uyum |
|---|----------|--------|------|-------------------|------|
| A | @mkkellogg + opacity toggle (remove yerine) | Düşük | Orta | Belki 2× iyileşme | Mevcut stack |
| B | @mkkellogg `addSplatScenes` (hepsini önden) | Düşük | Yüksek (RAM) | Akıcı ama RAM patlaması | Mevcut stack |
| C | Spark.js — N PLY + Dyno time switch | Orta | Orta | Akıcı (GPU-side) | THREE.js port |
| D | Spark.js — single packed + per-Gaussian Fourier Dyno | Yüksek | Düşük (eğer çalışırsa) | Native 60 fps | THREE.js port + backend export |
| E | SuperSplat / PlayCanvas embed | Düşük | Yüksek (kontrol kaybı) | Akıcı (third-party) | iframe / external app |
| F | WebGPU custom renderer (Scthe/web-splat fork) | Çok yüksek | Çok yüksek | Akıcı | Yeniden yazım |
| G | GPU buffer in-place update | Çok yüksek | Çok yüksek | En akıcı | Library iç müdahale |

Aşağıda her seçeneği detaylı inceliyoruz.

---

## 3. Seçenek A — @mkkellogg + Opacity Toggle

**Fikir:** `removeSplatScene` + `addSplatScene` yerine tüm frame'leri scene
listesine ekle, sadece `setSplatSceneOpacity(idx, 0/1)` ile aktif olanı
göster. Initial cost yüksek (90 PLY upload), runtime cost ~0.

**Library API:**
- `addSplatScene(url, options)` — tek scene
- `addSplatScenes(arr, options)` — toplu yükle
- `setSplatSceneOpacity(idx, value)` — public; opacity 0..1
- `setSplatSceneVisibility(idx, bool)` — public; ama bizim deneyimimizde
  scene.visible = false **respect edilmiyor** (mesh draw call yine yapılıyor,
  90 scene aynı anda render ediliyor)

**Bizim bug history'miz:** v3.6/v3.7'de `setSceneVisibility` ile 90 scene'i
toggle etmeyi denedik → tüm 90 frame superimposed render edildi.
**Hipotez (test edilmeli):** `setSplatSceneVisibility` no-op ama
`setSplatSceneOpacity(idx, 0)` GERÇEKTEN draw atlatabilir mi? Library
source'una bakmadan emin değiliz. README'de opacity API public diye geçiyor
([sparkjsdev/spark dokümantasyonu vs. mkkellogg](#sources)).

**Riskler:**
- Library 90 scene aynı anda render ederken FPS düşebilir
  (depth sort × 90 scene × 36k gaussian = ~3M splat sort her frame)
- Opacity = 0 olsa bile sort pipeline çalışır (z-order için splat'lar listede)
- ~810 MB GPU memory (90 × 9 MB per scene)
- 8 GB RTX 3060 Ti: training kapalıyken yeterli, simultane training varsa OOM

**Avantaj:**
- Mevcut kod minimum değişiklik (~50 satır)
- Hızlı denemesi yapılabilir
- Çıktı: işe yararsa Spark/WebGPU'ya geçişi tamamen by-pass eder

**Tahmin:** %30 ihtimal işe yarar. Library 90 simültane scene için
optimize edilmemiş.

**Validasyon yöntemi:**
1. Test job ile 30 frame yükle
2. `setSplatSceneOpacity` ile 1'i aktif, 29'u 0
3. FPS göstergesi ekle (Three.js stats)
4. <30 fps düşerse Plan A çöp.

---

## 4. Seçenek B — `addSplatScenes` (toplu)

**Fikir:** `viewer.addSplatScenes([...90])` ile tek seferde yükle, sonra
visibility/opacity API ile toggle.

**Aslında bu A'nın yükleme kısmı.** Tek API call avantajı: progressive load
sırası optimize, paralel parse. Ama runtime davranış aynı (A ile birlikte
ele alınmalı).

**Disqualified:** Tek başına çözüm değil; A'nın bir parçası.

---

## 5. Seçenek C — Spark.js (N PLY + Dyno Time Switch)

**Spark.js nedir:**
World Labs (`sparkjsdev/spark`) tarafından geliştirilmiş, THREE.js için
"advanced" 3DGS renderer. WebGL2 tabanlı, ply / sogs / spz / splat / ksplat
destekliyor. **Kritik özellik:** "Dyno" sistemi — splat üretimini /
modifikasyonunu GLSL'ye compile eden node graph.

**Yaklaşım C:**
- 90 PLY'yi `PackedSplats` olarak yükle (her biri ayrı SplatMesh)
- Bir `objectModifier` Dyno yaz: `currentTime` uniform'ına göre splat opacity
- Aktif olmayan SplatMesh'lerin alpha → 0 (GLSL'de)
- Opacity 0 olan splat'ların depth sort'a katılmaması için gpuCull (Spark
  buna izin veriyormuş, sparkjs.dev/docs/system-design'a göre)

**Avantaj:**
- Spark sort'u GPU'da yapıyor (mkkellogg CPU/worker'da)
- Per-frame uniform update → 60 fps (GPU bottleneck'e ulaşana kadar)
- Library aktif geliştirme altında (Spark 2.0 — Nisan 2026 release)

**Dezavantaj:**
- Frontend rewrite gerekli — `SplatViewer.tsx` baştan yazılır
- @mkkellogg ile API uyumlu değil (THREE.Object3D olarak ekleniyor ama
  init/dispose pattern farklı)
- 90 SplatMesh × ~810 MB GPU yine sorun

**Effort:** ~6-10 saat frontend, 0 backend.

---

## 6. Seçenek D — Spark.js + Per-Gaussian Fourier Dyno (En Radikal)

**Fikir:** PLY swap'i tamamen ortadan kaldır.

Bizim backend'imiz zaten **per-Gaussian Fourier coefficients** üretiyor
(`fourier_pos_coeffs: (N, K, 2, 3)`). Mevcut export pipeline bu coeff'lerden
90 timestamp'te 90 PLY üretiyor (ön-pişmiş). Bunun yerine:

1. Backend'den **tek dosya** export et:
   - `gaussians.ply` → static base (means, scales, quats, opacities, SH)
   - `fourier_coeffs.bin` → (N, K, 2, 3) fp16 binary
2. Frontend Dyno yaz:
   ```glsl
   // pseudo-Dyno
   vec3 deltaPos = vec3(0.0);
   for (int k = 0; k < K; k++) {
       float phase = 2.0 * PI * float(k+1) * uTime;
       deltaPos += fourierCoeffs[k][0] * sin(phase) +
                   fourierCoeffs[k][1] * cos(phase);
   }
   splat.center += deltaPos;
   ```
3. Browser'da `uTime` slider, 60 fps continuous animation.

**Avantaj:**
- Tek upload (~9 MB statik + ~70 MB Fourier coefficients K=8)
- Animation native GPU, frame swap yok
- Smooth video playback DOĞAL — between-frame interpolation zaten denklemde
- Backend export zamanı 90×'tan 1×'e düşer (önemli; ultra preset 90 PLY
  export ~5-10 dakika)

**Dezavantaj:**
- Frontend ciddi rewrite + Dyno custom shader
- Backend'de yeni "compact 4D" export format
- MLP deformation (HexPlane) bu yöntemle çalışmaz — sadece per-gaussian
  Fourier kısmı taşınabilir. MLP residual'ı korumak için ya fallback PLY
  ya da MLP'nin çıktısını FOURIER'a "bake" etme adımı.

**Bake fikri:** Training sonunda her gaussian için MLP(t)'yi 90 timestamp'te
örnekle, en küçük kareler ile Fourier K=12 katsayılarına fit et. MLP'yi yok et,
saf Fourier model üret. Bu zaten 4DGS paper'ın ana mimarisi (HexPlane'siz).

**Effort:** ~15-25 saat (frontend + backend export + bake script).

**Bu seçenek = uzun vadeli "doğru" çözüm.** Spark zaten tam bu use-case
için yapıldı ("dynamic 3D Gaussian Splatting renderer for the web").

---

## 7. Seçenek E — SuperSplat / PlayCanvas Embed

**SuperSplat:** PlayCanvas Engine üzerine kurulu, browser-based 3DGS editor.
**Sürüm 1.13'ten itibaren 4D playback var:** numbered PLY sequence
(frame_001.ply..frame_090.ply) → built-in timeline.

**Embed senaryosu:**
- Tauri pencerede SuperSplat'i iframe olarak göster
- API üzerinden frame folder gönder
- Timeline UI hazır (set keyframes, play/pause)

**Avantaj:**
- Zero rewrite — third party halletti
- 4D timeline + camera path + flythrough (SuperSplat 2.0 ekledi)
- Editor mode bonus: kullanıcı manuel splat editing yapabilir

**Dezavantaj:**
- iframe = limited control. Kendi UI'mızı (HyperparameterPanel,
  JobsList, Analytics) entegre edemiyoruz.
- Tauri WebView2 ↔ iframe permission/postMessage karmaşası
- "Open source ama GPL" — lisans dikkat gerek (kontrol edilmeli)
- Bizim PLY format ile %100 uyumluluk garantisi yok (SuperSplat custom
  splat header'ları kullanabilir)

**Bu seçenek "demo gece" çözüm değil, "MVP shortcut" çözüm.**
Eğer ürün kararı "kendi viewer'ımızdan vazgeç" ise hızlı yol. Aksi halde
araştırılmaya değmez.

---

## 8. Seçenek F — WebGPU Custom Renderer

**Adaylar:**
- `Scthe/gaussian-splatting-webgpu` — TypeScript, basit WebGPU port
- `KeKsBoTer/web-splat` — Rust + WGPU + WASM
- `cvlab-epfl/gaussian-splatting-web` — rasterization-based (compute değil)
- `MarcusAndreasSvensson/gaussian-splatting-webgpu` — TypeScript

**WebGPU avantajı:** WebGL2'den ~2-5× hızlı (compute shader, daha iyi
pipeline). Sort'u native GPU'da yapabiliriz (radix sort).

**Dezavantaj — kritik:**
- Tauri WebView2 (Edge Chromium) WebGPU desteği henüz default-disabled
  (Windows 11'de flag gerekli)
- 4D playback hiçbir WebGPU port'unda hazır değil — yine custom Dyno-eqv
  yazılması lazım
- Effort: ~30-50 saat (renderer + 4D extension)

**Verdict:** Ölü uç. Spark zaten WebGL2 ile yeterince hızlı,
WebGPU'ya geçişin marjinal kazancı bizim case'imizde 0.

---

## 9. Seçenek G — GPU Buffer In-Place Update

**Fikir:** Library'nin internal gaussian buffer'ına direkt erişim al,
her frame'de sadece pozisyon delta'sını yaz.

**Sorun:** `@mkkellogg/gaussian-splats-3d` private API. Library iç state'ine
müdahale = monkey-patch. Library upgrade'lerde kırılır.

**Spark eşdeğeri:** Spark'ın `SplatMesh.constructGenerator()` API'si
zaten bunu public hale getiriyor — ki bu Seçenek D ile aynı.

**Verdict:** Standalone bir seçenek değil; D'nin alt-mekaniği.

---

## 10. Karar Kriterleri

| Kriter | Ağırlık | A | C | D | E |
|--------|---------|---|---|---|---|
| Effort | 3 | ✅ Düşük | 🟡 Orta | ❌ Yüksek | ✅ Düşük |
| Akıcılık tahmini | 5 | 🟡 Orta | ✅ İyi | ✅ Mükemmel | ✅ İyi |
| Kontrol/ownership | 4 | ✅ | ✅ | ✅ | ❌ Kayıp |
| Risk | 3 | 🟡 | 🟡 | ✅ | ❌ Yüksek |
| Uzun vadeli evrim | 4 | ❌ Cul-de-sac | 🟡 | ✅ | ❌ |

**Skor (kaba):**
- A: 3 + 2.5 + 4 + 1.5 + 0 = **11** (10 üzerinden, hızlı validasyon)
- C: 1.5 + 5 + 4 + 1.5 + 2 = **14**
- D: 0 + 5 + 4 + 3 + 4 = **16** ⭐
- E: 3 + 5 + 0 + 0 + 0 = **8**

**Pratik öneri:** Önce A'yı 1 saatte test et (deny edersen erken kaybetmedin).
Sonra D'ye git (gerçek doğru çözüm).

---

## 11. Önerilen Yol Haritası

### Aşama 1 — Hızlı Validasyon (1 saat)
Seçenek A: `setSplatSceneOpacity` ile 30 frame test. FPS ölç.
- ✅ ≥ 30 fps → bitti, A'yı production'a al.
- ❌ < 30 fps → Aşama 2.

### Aşama 2 — Orta Vade (1 hafta)
Seçenek C: Spark.js'e geçiş, 90 SplatMesh + opacity Dyno.
- Mkkellogg → Spark migration (~6 saat)
- 90 frame opacity-toggle test
- ✅ ≥ 30 fps → ürünleştir.
- ❌ → Aşama 3.

### Aşama 3 — Uzun Vade (2-3 hafta)
Seçenek D: Per-Gaussian Fourier Dyno + backend bake.
1. Backend: `bake_fourier.py` — MLP(t)'yi K=12 Fourier'a fit et
2. Backend: yeni export format (`gaussians.ply` + `fourier.bin`)
3. Frontend: Spark Dyno custom shader ile Fourier decode
4. UX: continuous time slider (frame index değil, 0-1 normalize)

---

## 12. Açık Sorular

1. **Library iç davranışı:** `@mkkellogg`'da `setSplatSceneOpacity(i, 0)`
   gerçekten draw'ı atlatıyor mu yoksa sadece alpha'yı 0 yapıp yine sort'a
   sokuyor mu? **Yöntem:** library source'una baktıktan sonra cevap.
   ([source]: GitHub mkkellogg/GaussianSplats3D)

2. **MLP residual'ı kurtarmak:** Bake-to-Fourier sırasında MLP'nin korelasyonlu
   düzeltmelerini Fourier K=12 ne kadar iyi yakalar? Test sahnesi olarak
   `cookie_v3.7_smoke` ile pilot bake yapılabilir. PSNR loss ~0.5 dB altında
   ise kabul edilebilir.

3. **Spark Tauri uyumu:** Spark THREE.js r150+ gerektiriyor. Bizim Tauri
   webview Chromium 120+ — uyumlu olmalı ama spec edilmeli.

4. **HexPlane'in geleceği:** D seçeneği saf Fourier'a indiriyor — HexPlane
   training-zamanı residual olarak kalır (network'ün "global" düzeltme
   alanı). Sadece **export** bake edilir, training pipeline değişmez.

5. **Memory budget:** 90 PLY × 9 MB = 810 MB GPU RAM. 3060 Ti 8 GB → training
   kapalıyken OK ama browser other tabs için tight. Compact format
   (`packed-splats` Spark) ile 9 MB → ~3 MB inebilir.

---

## 13. Sources

### Spark.js (sparkjsdev/spark)
- [GitHub - sparkjsdev/spark](https://github.com/sparkjsdev/spark)
- [Spark Documentation Home](https://sparkjs.dev/)
- [Overview](https://sparkjs.dev/docs/overview/)
- [SplatMesh API](https://sparkjs.dev/docs/splat-mesh/)
- [Dyno Overview](https://sparkjs.dev/docs/dyno-overview/)
- [Procedural Splats](https://sparkjs.dev/docs/procedural-splats/)
- [System Design](https://sparkjs.dev/docs/system-design/)
- [New Features in 2.0](https://sparkjs.dev/docs/new-features-2.0/)
- [Streaming 3DGS worlds on the web (World Labs blog)](https://www.worldlabs.ai/blog/spark-2.0)
- [Show HN: Spark](https://news.ycombinator.com/item?id=44249565)

### SuperSplat / PlayCanvas
- [SuperSplat 1.13 plays back animated '4D Gaussian Splats'](https://www.cgchannel.com/2025/01/superspat-1-13-plays-back-animated-4d-gaussian-splats/)
- [SuperSplat 2.0 lets you create flythroughs](https://www.cgchannel.com/2025/02/supersplat-2-0-lets-you-create-flythroughs-of-3dgs-scans/)
- [Timeline | PlayCanvas Developer Site](https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/supersplat/timeline/)
- [SuperSplat — The Home for 3D Gaussian Splatting](https://superspl.at)
- [SuperSplat Editor (live)](https://superspl.at/editor)
- [SuperSplat: Free Open-Source Gaussian Splatting Editor — Review 2026](https://www.thefuture3d.com/software/supersplat/)

### @mkkellogg/gaussian-splats-3d
- [GitHub - mkkellogg/GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)
- [README.md](https://github.com/mkkellogg/GaussianSplats3D/blob/main/README.md)
- [Releases](https://github.com/mkkellogg/GaussianSplats3D/releases)
- [npm package](https://www.npmjs.com/package/@mkkellogg/gaussian-splats-3d)
- [Typed fork (gle-gaussian-splat-3d)](https://github.com/guyettinger/gle-gaussian-splat-3d)

### WebGPU alternatifler
- [Scthe/gaussian-splatting-webgpu](https://github.com/Scthe/gaussian-splatting-webgpu)
- [MarcusAndreasSvensson/gaussian-splatting-webgpu](https://github.com/MarcusAndreasSvensson/gaussian-splatting-webgpu)
- [KeKsBoTer/web-splat (Rust + WGPU)](https://github.com/KeKsBoTer/web-splat)
- [cvlab-epfl/gaussian-splatting-web](https://github.com/cvlab-epfl/gaussian-splatting-web)
- [antimatter15/splat (WebGL)](https://github.com/antimatter15/splat)

### Genel ekosistem (2026 durumu)
- [The State of Gaussian Splatting in 2026](https://www.thefuture3d.com/blog/state-of-gaussian-splatting-2026/)
- [4D Gaussian Splatting: What It Is and Why It Matters](https://www.thefuture3d.com/blog-0/2026/4/4/4d-gaussian-splatting-what-it-is/)
- [awesome-gaussian-splatting list](https://github.com/tomiwaAdey/awesome-gaussian-splatting)
- [Gauzilla Pro — 4D digital twins](https://www.webgpu.com/showcase/gauzilla-rust-gaussian-splatting-digital-twins/)

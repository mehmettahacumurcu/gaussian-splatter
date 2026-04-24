# Deformation Field Mimari — Fikir Havuzu

> **Bağlam**: cutlemon_full_recovered'da motion neredeyse görünmez (median Δpos=%0.035 of scene).
> v3 training param'ları bunu büyük ihtimalle fix'leyecek ama **mimari limiti** de var.
> Bu dosya gelecek iterasyon için beyin fırtınası notları.

---

## Mevcut mimari (v3)

```
(x, y, z, t)
  ↓ HexPlane: 6 planes × 96² × 48 feat  (4D feature grid)
  ↓ concat → 288-dim vektor
  ↓ Fourier time encoding (+13 dim)
  ↓ MLP 4 layer × 512 width             (~4M params)
  ↓
(Δpos[3], Δquat[4], Δscale[3])
```

**Güçlü yan**: HexPlane 4D dense kodlama, expressive.
**Zayıf yan**: MLP **GLOBAL** — tüm gaussian'lar aynı fonksiyondan beslenir.
Spatial locality yok. "Kesilen limon'un yanı" ile "arka plan duvarı" aynı MLP'yi çağırır.

Kapasite problemi OLMAYABILIR — signal/inductive bias problem daha olası.
v3 training değişiklikleri (gevşek reg + higher lr_deform + track loss güçlendirme)
önce test edilmeli. Onlar yetersiz kalırsa aşağıdaki mimari değişiklikleri yapılır.

---

## 7 alternatif yaklaşım — karşılaştırma

### 1. Daha büyük MLP — naif upscale

- Width 512 → 2048, depth 4 → 8, residual + LayerNorm
- Param: 4M → 60M
- **Fayda**: Marjinal (diminishing returns)
- **Risk**: Param artar ama öğrenme sinyali aynı
- **Derecelendirme**: ⭐⭐

### 2. Per-Gaussian Fourier trajectories (4DGS original paper) ⭐⭐⭐⭐⭐

```
pos(t) = mean + Σ [A_k · cos(2πkt) + B_k · sin(2πkt)]   k=0..K
```

- K=8 → 16 coeffs × 3 axis = 48 float per gaussian
- 50k gaussian × 48 × 4 byte = **9.6 MB** extra
- MLP'yi KALDIR (rotation/scale için hâlâ tut)
- **Fayda**: Motion çok daha iyi, spatial locality native, paradigma değişikliği
- **Referans**: Yang et al 2024 "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering"
- **Implementation**: GaussianModel'e `fourier_pos_coeffs` Parameter ekle,
  trainer'da deform yerine bunu kullan

### 3. Split MLP heads ⭐⭐⭐⭐

```
Shared trunk (HexPlane + 2 layer)
  ↓
 ┌────┬────┬────┐
 │Pos │Rot │Scale│     (her biri 2 layer head)
 └────┴────┴────┘
```

- Param ≈ aynı
- **Fayda**: Konvergens hızlanır, head'ler specialize olur
- **Implementation**: 30 dakika iş. Quick win.

### 4. Control points + RBF blend ⭐⭐⭐

- K=100-500 control point MLP tarafından deforme edilir
- Her gaussian en yakın M control point'ten RBF-weighted blend alır:
  ```
  dpos(gaussian) = Σ w_i(distance) · dpos(control_i)
  ```
- K-Means ile initial positions'tan control points seçilir
- **Fayda**: Articulated motion (el, cisim) için ideal
- **Zayıf**: Fluid/chaotic motion için kötü
- **Referans**: SC-GS (Huang 2024)

### 5. Neural Flow / velocity field ⭐⭐⭐

- MLP `v(x,y,z,t)` velocity döndürür
- `pos(t) = ODE integrate v from t=0 to t`
- diffrax veya scipy
- **Fayda**: Doğal smooth, t-continuous, fiziksel
- **Zayıf**: Her iter ODE solve → **5× yavaş training**
- Akademik olarak çekici, 3060 Ti için pratik değil

### 6. SE(3) rigid body parts ⭐⭐⭐⭐

- Gaussian'ları K=50-200 küme halinde grupla (K-Means initial positions'ta)
- Her küme kendi SE(3) trajectory (rot + trans over time)
- Her gaussian parent cluster'ın hareketini takip eder
- **Param**: K × T × 6 DOF ≈ 50 × 60 × 6 = 18k (çok az)
- **Fayda**: Rigid object motion için mükemmel
- **Zayıf**: Fluid/deformable nesne için kötü
- **Referans**: Dynamic 3D Gaussians (Luiten 2023)
- **Cutlemon için UYGUN** (kesim kolu ve limon aslında rigid)

### 7. Hibrit: Per-gaussian Fourier + mini MLP correction ⭐⭐⭐⭐⭐

```
pos(t) = mean
       + per_gaussian_fourier(t)       ← local identity preserving
       + global_mlp(hexplane(x,y,z,t)) ← global corrective term
```

- #2 primary, MLP düzeltici
- **Fayda**: Robust + expressive, best of both worlds
- **Param**: +9.6 MB (fourier) + mevcut MLP

---

## Karşılaştırma tablosu

| # | Yaklaşım | Param Δ | Hız Δ | Motion kalitesi | Impl maliyeti |
|---|---|---|---|---|---|
| 1 | Big MLP | +50M | -15% | +10% | Trivial |
| 2 | **Per-gaussian Fourier** | +10MB | -5% | **+80%** | Orta |
| 3 | Split MLP heads | 0 | 0 | +15% | Trivial |
| 4 | Control points RBF | -3M | +10% | +30% (rigid), -20% (fluid) | Yüksek |
| 5 | Neural Flow ODE | 0 | -60% | +40% | Çok yüksek |
| 6 | SE(3) rigid parts | -3M | +20% | +60% (rigid), -40% (fluid) | Orta |
| 7 | **Hibrit #2+MLP** | +10MB | -8% | **+100%** | Orta-yüksek |

---

## Referans gerçek dünya (SOTA)

| Paper | Yaklaşım | Not |
|---|---|---|
| 4DGS (Wu et al 2023) | HexPlane + MLP | Şu an bizim |
| **4D-GaussianSplatting (Yang et al 2024)** | **Per-gaussian polynomial trajectories** | **SOTA** |
| Deformable 3DGS (Yang 2024) | Time-conditioned MLP | Bize benzer |
| Dynamic 3DGS (Luiten 2023) | SE(3) rigid bodies | #6 |
| SC-GS (Huang 2024) | Sparse control points + KNN blend | #4 |

**Trend**: Per-gaussian trajectory ailesi tutarlı biçimde kazanıyor.

---

## Öneri sıralaması

### Öncelik 1: Split MLP heads (#3)
- 30 dakika iş, risk sıfır
- Quick win ~%15 kalite artışı
- Önce bunu dene, v3 result'ları görünce karar ver

### Öncelik 2: Per-gaussian Fourier trajectories (#2)
- 1-2 günlük iş, paradigma değişikliği
- Motion'u unlock eder
- 4DGS paper'ı tam bunu yapıyor

### Öncelik 3: Hibrit (#7)
- #2 çalıştıktan sonra
- En güçlü yaklaşım, biraz daha karmaşık
- Production-quality için final adım

### Sakla (şimdi değil)
- **#1 Big MLP** — param artırmak signal eksikliğini çözmez
- **#4 Control points** — scene-specific, general-purpose değil
- **#5 Neural Flow** — çok yavaş, pratik değil
- **#6 SE(3) rigid** — deformable (yüz, sıvı) scene'lerde başarısız

---

## Karar kriterleri (bir sonraki tur için)

1. **Ne tür sahneler hedef?**
   - Rigid object (mekanik, oyuncak) → #6
   - Organic/deformable (yüz, sıvı, yaprak) → #2 veya #7
   - Karma/general → #7

2. **Training süresi kısıtı?**
   - Kısıt var → #3 (minimal overhead)
   - Kısıt yok → #7 (best quality)

3. **CoTracker sinyali sağlam mı?**
   - Per-gaussian trajectory için track loss kritik
   - OOM fallback ile artık güvenilir

---

## Notlar

- v3 training run sonucu motion hâlâ zayıfsa → **#3 önce, sonra #2**
- v3 motion iyi ise → sadece #3 uygulansa yeter, #2/#7 overkill olabilir
- Her mimari değişikliği için **micro preflight** önce — yeni kod çalışıyor mu
  sanity check
- Her denemede **eski deform kodu korunsun** (config flag ile switch)

---

_Son güncelleme: 2026-04-24 (v3 full run öncesi)_

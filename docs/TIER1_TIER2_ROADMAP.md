# Motion Geliştirme Yol Haritası — Tier 1 (RAFT) + Tier 2 (Static/Dynamic)

> **Bağlam**: v3.7.x sonrası motion oluşuyor (banana_high'ta %32 gaussian hareketli) ama
> **direction yanlış** ve **statik bölgeler titriyor**. Track loss CoTracker'ın
> 8GB VRAM probleminden ötürü çalışmıyor. İki tier strateji ile bu çözülecek.

---

## 📊 Mevcut sorunun teşhisi

| Sinyal kaynağı | Yoğunluk | Kalite | Durum |
|---|---|---|---|
| Recon loss (L1+SSIM) | Dense (640k pixel) | Yön bilgisi yok, çok-çözümlü | ✅ aktif |
| Track loss (CoTracker) | Sparse (200 anchor) | Anchor lifting hatası | ❌ OOM, 8GB'da çalışmıyor |
| Depth loss (MiDaS) | Dense | Sadece 1D (z direction) | ✅ aktif, COLMAP-aligned |
| Mask-weighted recon | Dense (mask) | Direction yok, sadece weight | ✅ aktif |

**Eksik**: Pixel-level 2D motion direction supervision.
**Çözüm**: Tier 1 (RAFT optical flow) + Tier 2 (static/dynamic separation).

---

## 🎯 TIER 1 — RAFT Optical Flow Loss

### Hedef
Her frame çiftinin pixel-bazlı 2D motion'u öğrenilsin → MLP/Fourier'a **dense direction signal**.

### Neden RAFT, neden CoTracker değil?

| | CoTracker (mevcut) | **RAFT (Tier 1)** |
|---|---|---|
| Supervision yoğunluğu | 200-2048 sparse anchor | **640k pixel** (dense) |
| VRAM (banana 343 frame) | 6.5+ GB (OOM 8GB) | **~2-3 GB** per frame pair |
| Model boyutu | ~700 MB | **~21 MB** |
| Frame topluluk | Tüm video tek tensor | **Per-pair**, batch=1 |
| Industry-standard | Tracking için OK | **Optical flow için SOTA** |
| 3D anchor lift | Gerekli (depth align bağımlı) | **Gerekmez** (direkt 2D flow) |
| Direction signal | Sparse | **Dense** |

### Mimari

```
RAFT preprocessing (foundation Faz 3d):
  for i in range(T-1):
    flow[i] = RAFT(frame_i, frame_{i+1})   # (2, H, W)
  save flow.npy / flow_*.npy

Trainer (her iter):
  i = randint(0, T-2)
  flow_gt = load_flow(i, i+1)             # (2, H, W)
  
  # Render at t_i
  render_i = rasterize(d_means(t_i), cam_i)
  
  # Per-gaussian motion projection
  uv_i = project(d_means(t_i), cam_i)      # (N, 2)
  uv_ip1 = project(d_means(t_{i+1}), cam_{i+1})  # (N, 2)
  
  predicted_flow_at_gaussians = uv_ip1 - uv_i   # (N, 2) pixel motion
  
  # Bilinear sample gt_flow at uv_i
  gt_flow_at_gaussians = sample(flow_gt, uv_i)  # (N, 2)
  
  # Loss
  in_bounds = (uv_i in image bounds) & (uv_ip1 in image bounds)
  loss_flow = || predicted_flow[in_bounds] - gt_flow[in_bounds] ||₁
```

### Implementation phases

#### Phase A — Preprocessing module (2-4 saat)

**Dosya**: `backend/preprocess/optical_flow.py` (yeni)

```python
def compute_optical_flow(
    frames_dir: Path,
    output_dir: Path,
    model_name: str = "raft_things",   # raft-things, raft-sintel, raft-kitti
    target_size: tuple[int, int] = (640, 360),  # training res
    device: str = "cuda",
    overwrite: bool = False,
) -> Path:
    """
    Pairwise optical flow hesapla. Output: flow_NNNN.npy per pair.
    Storage: fp16, training res. Banana 343 frame → 342 pair × ~1.5 MB = ~500 MB.
    """
```

- RAFT torch.hub: `torch.hub.load("princeton-vl/RAFT", "raft_things")`
- Per-pair forward: `model(image1, image2, iters=12)`
- Output: `(2, H, W)` float — flow_x, flow_y in pixels
- Resize to target training resolution
- Save fp16 .npy

**VRAM check**: RAFT ~21 MB model + 2× 720p input = ~3 GB peak per pair. **Tek pair**, sonra release. Cumulative değil.

#### Phase B — Pipeline integration (1 saat)

**Dosya**: `backend/pipeline.py`

Foundation içine yeni faz:
```python
# Faz 3d — Optical flow (CoTracker'a alternatif/ek)
print("\n[Faz 3d] RAFT optical flow")
try:
    from .preprocess.optical_flow import compute_optical_flow
    compute_optical_flow(paths["frames"], paths["flow"])
    foundation_status["flow"] = "ok"
except Exception as e:
    foundation_status["flow"] = f"failed: {e}"
```

`scene_paths()` to `config.py` güncelle: `"flow": base / "flow"`.

#### Phase C — Trainer integration (4-6 saat)

**Dosya**: `backend/model/trainer.py`

##### C.1: Flow loader

```python
def _load_flow_pair(flow_dir: Path, frame_idx: int, target_size: tuple[int, int]) -> torch.Tensor | None:
    """Load flow for pair (i, i+1). Returns (2, H, W) or None."""
    flow_path = flow_dir / f"flow_{frame_idx:04d}.npy"
    if not flow_path.exists():
        return None
    flow = np.load(flow_path).astype(np.float32)
    # Resize/scale if needed (already at training res from preprocess)
    return torch.from_numpy(flow)
```

##### C.2: Loss computation in train loop

```python
# After computing dpos for current iter (t_norm = idx / (T-1))
if use_flow and idx < T - 1:
    # Sample next frame too
    t_next = (idx + 1) / (T - 1)
    
    # Apply deformation at both
    d_means_i, _, _ = self._apply_deformation(t_norm)
    d_means_ip1, _, _ = self._apply_deformation(t_next)
    
    # Project both
    uv_i, valid_i = _project_world_to_pixel(d_means_i, K_scaled, w2c_list[idx])
    uv_ip1, valid_ip1 = _project_world_to_pixel(d_means_ip1, K_scaled, w2c_list[idx + 1])
    
    valid = valid_i & valid_ip1
    in_bounds = (uv_i[:, 0] >= 0) & (uv_i[:, 0] < Ws) & (uv_i[:, 1] >= 0) & (uv_i[:, 1] < Hs) & valid
    
    if in_bounds.sum() > 100:
        # Predicted flow at gaussian centers
        pred_flow = uv_ip1[in_bounds] - uv_i[in_bounds]   # (N_v, 2)
        
        # GT flow bilinear sampled at uv_i
        flow_gt = _load_flow_pair(flow_dir, idx, (Ws, Hs))  # (2, H, W)
        u_norm = (uv_i[in_bounds, 0] / (Ws - 1)) * 2 - 1
        v_norm = (uv_i[in_bounds, 1] / (Hs - 1)) * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1).view(1, 1, -1, 2)
        gt_sampled = F.grid_sample(flow_gt.unsqueeze(0), grid, mode="bilinear", align_corners=True)
        gt_flow = gt_sampled.squeeze().T   # (N_v, 2)
        
        # Diagonal-normalized L1 (lambda ~ O(1))
        diag = (Ws ** 2 + Hs ** 2) ** 0.5
        flow_l = (pred_flow - gt_flow).abs().mean() / diag
        loss = loss + self.lambda_flow * warmup * flow_l
        comp["flow"] = flow_l.item()
```

##### C.3: Computational cost
- Extra: 1 deformation forward + 1 projection = ~30% per iter
- Memory: flow tensor ~1 MB per pair, no accumulation
- Expected slowdown: 5 it/s → 3-4 it/s

#### Phase D — Config + API + Frontend (1-2 saat)

**`backend/config.py`**:
```python
@dataclass
class TrainConfig:
    lambda_flow: float = 0.5    # Track loss seviyesinde, dense olduğu için biraz düşük
    flow_pair_dt: int = 1       # Pair offset (1 = consecutive, >1 = sparse)
```

**`backend/api.py`**:
```python
lambda_flow: float | None = Form(None, description="RAFT optical flow loss weight"),
```

**Frontend**:
```typescript
// HyperparameterPanel.tsx — Loss grubu
{ key: "lambda_flow", label: "λ flow (RAFT)", placeholder: "0.5", ...}
```

#### Phase E — Validation (1 saat)

1. Smoke test cookie ile (CoTracker'sız sadece RAFT)
2. Smoke + RAFT vs Smoke + CoTracker karşılaştırması
3. Full run banana ile

### Beklenen etki

| Metrik | Şu an (track=0) | RAFT sonrası |
|---|---|---|
| Δpos mean (chicken/banana) | 0.3-0.8 | 0.5-1.5 |
| Motion direction doğruluğu | ~%30 (recon-driven) | **%80+** (flow-supervised) |
| Visible motion | Subtle, yön kaba | **Smooth, doğru yön** |
| Streak/jitter | Var (bazı outlier) | **Çok az** |
| PSNR | 20-22 | **23-26** (motion regularize ediyor) |

### Risk + mitigation

| Risk | Mitigation |
|---|---|
| RAFT VRAM OOM (CoTracker gibi) | Per-pair processing, batch=1, model size ~21MB. Banana'da bile OK olmalı. |
| Flow files disk space | fp16 + training res = 500 MB banana. Acceptable. |
| Flow quality kötü (texture-less regions) | Foundation valid_mask: flow magnitude > epsilon |
| Iter slowdown 30%+ | Periodic flow loss (her 2-3 iter'de bir). lambda_flow tune. |

### Toplam süre tahmini

- Phase A (preprocessing): **3 saat**
- Phase B (pipeline): **1 saat**
- Phase C (trainer): **5 saat**
- Phase D (config+API+UI): **2 saat**
- Phase E (validation): **2 saat smoke + 4 saat full**
- **Toplam: ~17 saat** (2-3 iş günü)

---

## 🎯 TIER 2 — Static / Dynamic Gaussian Separation

### Hedef
Sahnedeki her gaussian'ı **static** veya **dynamic** olarak etiketle. Static gaussian'lar **donar** (Δpos=0), dynamic gaussian'lar full deformation pipeline'ı kullanır.

### Neden gerekli?

Banana sahnesi:
- **%80 static**: masa, plaka, parmak ucu, arka plan
- **%20 dynamic**: muz, kabuk, bıçak

Şu an her gaussian her iter'de:
- Fourier coeffs eğitiliyor
- MLP forward pass alıyor
- Spatial smoothness reg'ine giriyor
- Anchor KNN'e dahil ediliyor

Sonuçta **static gaussian'lar bile motion gradient alıyor** (recon noise'tan) → titreşim, jitter.

### Determining static vs dynamic — 3 yöntem

#### Yöntem 1: Mask projection at init (simple, fast)
- Farneback masks zaten var (her frame için dynamic region map)
- Her gaussian'ı her camera view'ine project et
- Mask değerleri topla: `score[g] = Σ_t mask_t at uv_g_t`
- Threshold: `score[g] > 0.5` → dynamic

```python
def compute_dynamic_flag(
    means: torch.Tensor,         # (N, 3)
    masks_dir: Path,
    cam_w2c_per_frame: list,
    K: torch.Tensor,
    threshold: float = 0.3,
) -> torch.Tensor:    # (N,) bool
    """
    Her gaussian'ı her frame'e project et, mask değerini topla.
    Threshold üstü = dynamic.
    """
```

**Pros**: Init aşamasında bir kez hesaplanır, ~10 sn iş.
**Cons**: Boundary gaussian'lar (limonun kenarı) yanlış sınıflanabilir.

#### Yöntem 2: Gradient-based during training
- İlk K iter sonrası her gaussian'ın `|dpos|` ortalamasını hesapla
- Threshold: avg > X → dynamic, altı → static
- Daha doğru ama gecikmiş

**Pros**: Doğru sinyal — gerçekten kim hareket ediyor.
**Cons**: Önce training gerekir, chicken-and-egg.

#### Yöntem 3: Optical flow projection (Tier 1 sonrası)
- Flow.npy'leri kullan
- Her gaussian'ı frame_0'a project et
- O pixel'deki ortalama flow magnitude
- Threshold: motion > X → dynamic

**Pros**: En doğru, dense supervision'lı.
**Cons**: Tier 1'i gerektirir (ama biz zaten yapacağız).

#### Önerilen kombinasyon
**Method 1 at init (fast bootstrap)** + **Method 3 re-eval after Tier 1 ready** (refine).

### Architecture — Approach A (Hard freeze) — RECOMMENDED

`is_dynamic` bool flag her gaussian için.

```python
class GaussianModel:
    self.is_dynamic: torch.Tensor  # (N,) bool, persistent

# Trainer._apply_deformation:
def _apply_deformation(self, t):
    dpos_mlp, dquat, dscale = self.deform(...)
    dscale = dscale.clamp(...)
    
    if self.deform_pos_mode != "mlp":
        dpos_fourier = decode_fourier_trajectory(...)
    
    dpos = dpos_mlp + dpos_fourier  # hybrid
    
    # YENİ: zero out for static
    dpos = dpos * self.gs.is_dynamic[:, None].float()
    
    deformed_means = self.gs.means + dpos
    ...
```

### Implementation phases

#### Phase A — Mask projection function (3 saat)

**Dosya**: `backend/model/gaussian_model.py` ek fonksiyon

```python
@torch.no_grad()
def init_dynamic_flag(
    self,
    masks_dir: Path,
    cam_w2c_per_frame: list[torch.Tensor],
    cam_K: torch.Tensor,
    image_size: tuple[int, int],
    threshold: float = 0.3,
    device: str = "cuda",
) -> None:
    """
    Her gaussian'ı her camera view'ine project et, mask değerini topla.
    Sum > threshold * num_frames → dynamic.
    """
    N = self.num_points
    accum = torch.zeros(N, device=device)
    n_visible = torch.zeros(N, device=device)
    
    for i, w2c in enumerate(cam_w2c_per_frame):
        mask_path = masks_dir / f"mask_{i+1:04d}.png"
        if not mask_path.exists(): continue
        mask = load_mask(mask_path)  # (H, W) [0,1]
        
        # Project
        uv, valid = project(self.means, cam_K, w2c)
        in_bounds = is_in_bounds(uv, image_size) & valid
        
        # Sample mask
        u_int = uv[in_bounds, 0].long().clamp(0, W-1)
        v_int = uv[in_bounds, 1].long().clamp(0, H-1)
        accum[in_bounds] += mask[v_int, u_int]
        n_visible[in_bounds] += 1
    
    avg_score = accum / n_visible.clamp(min=1)
    self.is_dynamic = (avg_score > threshold).to(self.means.device)
    
    print(f"Dynamic flag: {int(self.is_dynamic.sum())} / {N} dynamic "
          f"({100 * self.is_dynamic.float().mean():.1f}%)")
```

#### Phase B — Trainer respect flag (2 saat)

```python
# trainer.py _apply_deformation
dpos_total = dpos_mlp + dpos_fourier
# Zero out static gaussians' Δpos
dpos_total = dpos_total * self.gs.is_dynamic[:, None].float()
```

Δquat ve Δscale için de aynı? **Evet** — static gaussian rotate veya scale-shift yapmamalı.

#### Phase C — Density control sync (3 saat)

`density_control.py` — clone/split/prune'da flag taşı.

```python
# In step():
# Clone: new gaussian inherits parent's flag
if n_cloned > 0:
    new_flags = gs.is_dynamic[clone_mask]
    gs.append_gaussians(..., new_is_dynamic=new_flags)

# Split: aynı
if n_split > 0:
    new_flags = gs.is_dynamic[split_mask].repeat(2)
    ...

# Prune: keep mask uygula
gs.is_dynamic = gs.is_dynamic[keep_mask]
```

GaussianModel'de `_apply_mask` ve `append_gaussians`'a `is_dynamic` parametresi ekle.

#### Phase D — Optimizer mask (2 saat)

Static gaussian'ların gradient'ı sıfırlansın:

```python
# After backward, before step:
with torch.no_grad():
    static_mask = ~self.gs.is_dynamic
    if self.gs.fourier_pos_coeffs.grad is not None:
        self.gs.fourier_pos_coeffs.grad[static_mask] = 0
```

Veya parametre group'a sparse gradient flag.

#### Phase E — Re-evaluation periodic (2 saat)

Her N=5000 iter'de bir flag'i re-compute (Method 3, Tier 1 sonrası):

```python
if it % 5000 == 0 and self.flow_dir is not None:
    self.gs.refine_dynamic_flag(self.flow_dir, ...)
```

Conservative: only DEMOTE static→dynamic (false positives ok), never PROMOTE dynamic→static (would freeze motion mid-training).

#### Phase F — Visualization + diagnostics (2 saat)

- PLY export'ta is_dynamic flag'i ekstra renk olarak çıkar
- Stats log'la: `[trainer] %X gaussian dynamic, avg motion of dynamic: Y`
- Frontend viewer'da toggle: "Static gri / Dynamic vivid"

### Beklenen etki

Banana scene tahminleri:
- **N_dynamic ≈ 8000** (42k × %20)
- Compute %20 azalır (deformation forward static'lerde sıfır gradient)
- **Static jitter eliminate** — ana kalite artışı
- Dynamic gaussian'lar tüm gradient budget'ı alır → daha agresif/doğru motion

| Metrik | Şu an | Tier 2 sonrası |
|---|---|---|
| Compute per iter | 100% | **~70%** (1.4× hızlı) |
| Static rendering quality | Jittery | **Pristine** (no motion) |
| Dynamic motion magnitude | Average across N | **Concentrated on dynamic** (5× sharper) |
| PSNR (static frames) | 22 | **25+** |

### Edge cases

| Edge case | Handling |
|---|---|
| Boundary gaussian misclassify (kenar) | Threshold yumuşat (0.3 yerine 0.2), ya da Method 3 ile refine |
| Static gaussian'ın aslında hareket etmesi gerekir | Method 3 re-evaluation ile yakala |
| Cloning dynamic → child dynamic | Doğru, parent'tan inherit |
| Pruning dynamic | OK, flag de prune edilir |
| Tüm gaussian'lar dynamic çıkarsa | Threshold çok düşük, tune et |
| Tüm gaussian'lar static çıkarsa | Threshold çok yüksek, tune et |

### Toplam süre tahmini

- Phase A (mask projection): **3 saat**
- Phase B (trainer respect): **2 saat**
- Phase C (density sync): **3 saat**
- Phase D (optimizer mask): **2 saat**
- Phase E (re-evaluation): **2 saat**
- Phase F (viz + diagnostics): **2 saat**
- **Toplam: ~14 saat** (1.5-2 iş günü)

---

## 🗺 Birleşik yol haritası

### Sıralama

```
1. Tier 1 (RAFT)        ← ÖNCE
   ├─ Motion direction sinyali
   └─ Tier 2 Method 3'ü mümkün kılar

2. Tier 2 (Static/Dynamic)  ← SONRA
   ├─ Tier 1'in flow'unu kullanarak Method 3 ile flag refine
   ├─ Static jitter eliminate
   └─ Compute concentration on dynamic
```

### Combined timeline

| Hafta | İş | Çıktı |
|---|---|---|
| **1. hafta** | Tier 1 Phase A-C | RAFT preprocessing + trainer integration |
| 1. hafta sonu | Tier 1 Phase D-E | Smoke + full validation |
| **2. hafta** | Tier 2 Phase A-D | Static/dynamic separation |
| 2. hafta sonu | Tier 2 Phase E-F | Re-evaluation + viz |
| **3. hafta** | Combined validation | Final motion quality test |

### Combined expected outcome

```
Şu an (banana_high v3):
  Δpos mean: 0.3-0.8 (recon-driven, yön kaba)
  PSNR: ~22
  Static: jittery
  Visible motion: subtle, %32 gaussians moving

Tier 1 sonrası (RAFT):
  Δpos mean: 0.5-1.5 (flow-supervised, yön doğru)
  PSNR: 23-26
  Static: hâlâ jittery (Tier 2'ye kadar)
  Visible motion: SMOOTH and DIRECTIONAL

Tier 2 sonrası (Tier 1 + Static/Dynamic):
  Δpos mean (dynamic only): 1.0-3.0 (konsantre)
  PSNR: 25-28+
  Static: PRISTINE (no motion at all)
  Visible motion: SHARP, focused on objects
```

### Risk değerlendirmesi

**Low risk**:
- Tier 1 Phase A-B (preprocessing)
- Tier 2 Phase A (init flag)
- Tier 2 Phase F (viz)

**Medium risk**:
- Tier 1 Phase C (trainer integration — flow loader, projection)
- Tier 2 Phase B (trainer respect — yan etkiler)

**High risk**:
- Tier 2 Phase C (density sync — multi-tensor consistency)
- Tier 2 Phase E (re-evaluation — training instability)

### Validation plan

Her tier için:
1. **Micro test** (5-10 dk): API smoke, no crash
2. **Smoke test** (5-10 dk): Cookie/banana, validate metrics
3. **Full test** (2-4 saat): High preset
4. **Comparison run**: A/B karşılaştırma (önceki versiyonla)

### Combined validation checklist

- [ ] Tier 1 smoke: `flow loss > 0`, decreasing trend
- [ ] Tier 1 full: PSNR 23+, Δpos doğru yönde (visual inspection)
- [ ] Tier 2 smoke: `is_dynamic` flag doğru ratio (%15-30)
- [ ] Tier 2 full: Static frames jitter-free, dynamic motion sharp
- [ ] Combined: Both tiers active, no regression in any metric

---

## 📋 Open questions / decisions

### Tier 1
1. **RAFT model variant?** raft-things (genel) vs raft-sintel (animation-style) vs raft-kitti (driving). Banana için **raft-things** uygun.
2. **Flow precompute vs on-the-fly?** Precompute (disk space) vs runtime (CPU/GPU recompute every iter). Önerim: **precompute** at training resolution.
3. **lambda_flow değeri?** Track loss 0.5'di. RAFT dense olduğu için **0.3-0.5** arası.
4. **Flow pair dt?** Consecutive (dt=1) veya farklı (dt=5)? Konsekütif için RAFT en iyi. Önerim: **dt=1**.

### Tier 2
1. **Threshold seçimi?** Banana için 0.3 başlangıç, ama scene-bağımlı. Adaptive olabilir mi?
2. **Hard vs soft flag?** Approach A (hard) önerilir, ama Approach B (soft weight) testbed olabilir.
3. **Re-evaluation interval?** Her 5000 iter mi, dinamik mi? **Sabit 5000** başlangıç.
4. **Static gaussian'ların scale/quat'ı dondurulmalı mı?** Evet — full freeze. Aksi halde scale değişimi de motion sayılır.

---

## 🚀 Sıradaki adım

1. Mevcut banana_high_v3 run'ı bitsin (~2 saat kaldı)
2. Sonucu v1 ile karşılaştır (scale clamp etkisi)
3. **Tier 1 Phase A başla** — RAFT preprocessing module
4. Cookie ile smoke test
5. Banana ile validation
6. Tier 1 success → Tier 2 başla

Tahmini total süre: **3-4 hafta** (yarı zamanlı çalışma) veya **5-7 iş günü** (tam zamanlı).

---

_Son güncelleme: 2026-04-25_
_Bağlı task'lar: #70 (Tier 1 RAFT), #71 (Tier 2 Static/Dynamic)_

# SOTA Baselines (per-scene PSNR / SSIM / LPIPS)

> **Purpose.** A single source of truth for "what does the literature get on this
> scene?" so we can quickly tell if our pipeline is in the same ballpark as
> published methods. Used by `scripts/sota_compare.py`.

> **Honesty disclaimer.** These numbers are taken from the cited papers' published
> tables, which used the authors' best configs (long training, full datasets, their
> own dataloaders, sometimes their own evaluation script). Implementations vary by
> ~0.3–0.5 dB even on the same dataset. Use these as a *target band*, not an exact
> equality check. A 1 dB gap is well within "different implementation, same
> algorithm"; a 3+ dB gap is a real algorithmic / data / preprocessing problem.

---

## How to read these tables

**Verdict bands** the comparison script applies, against the *strongest* method
listed for each scene:

| ΔPSNR vs best baseline | Verdict             | Action                                        |
|------------------------|---------------------|-----------------------------------------------|
| ≥ -1.0 dB              | ✅ SOTA-tier        | Algorithm sound. Cloud compute will pay off.  |
| -3.0 dB ≤ Δ < -1.0     | ⚠ Below SOTA       | Probably tunable + more compute. Try cloud.   |
| < -3.0 dB              | ✗ Algorithmic gap  | More compute won't help. Investigate algo.    |

**Eval protocol assumed** (matches what `backend/eval/nvs_eval.py` does):
- Multi-view: hold out **`cam00`** (N3V convention), render all timesteps,
  mean PSNR/SSIM/LPIPS over frames.
- Train cams: the remaining cameras.
- LPIPS network: AlexNet (most papers); we use VGG (`losses_perceptual.LPIPSLoss`).
  LPIPS-VGG is typically 0.01–0.03 *higher* than LPIPS-Alex (i.e. looks worse
  even when reconstruction is identical). Don't compare LPIPS values 1:1 across
  papers without checking which net they used.

---

## 4D Dynamic — N3V Plenoptic Video (Neural 3D Video, Li et al. 2022)

Six multi-view scenes from the N3V dataset. **Held-out cam: `cam00`**, train
cams `cam01`-`camNN`. Standard benchmark for dynamic novel-view synthesis.

| Scene             | 4DGaussians (Wu+24) | Deformable 3DGS (Yang+24) | Spacetime Gauss (Li+24) | K-Planes (Fridovich+23) |
|-------------------|--------------------:|--------------------------:|------------------------:|------------------------:|
| `coffee_martini`  |               27.34 |                     27.71 |                   28.61 |                   28.74 |
| `cook_spinach`    |               32.46 |                     32.86 |                   33.18 |                   31.23 |
| `cut_roasted_beef`|               32.90 |                     32.43 |                   33.52 |                   31.76 |
| `flame_salmon_1`  |               29.20 |                     27.51 |                   29.48 |                   30.44 |
| `flame_steak`     |               32.39 |                     32.49 |                   33.51 |                   32.38 |
| `sear_steak`      |               33.12 |                     32.89 |                   33.89 |                   32.52 |

SSIM (best of the four for each scene; range 0–1):

| Scene             | SSIM (best paper) |
|-------------------|------------------:|
| `coffee_martini`  | 0.913 (Spacetime) |
| `cook_spinach`    | 0.946 (Spacetime) |
| `cut_roasted_beef`| 0.948 (Spacetime) |
| `flame_salmon_1`  | 0.929 (K-Planes)  |
| `flame_steak`     | 0.953 (Spacetime) |
| `sear_steak`      | 0.957 (Spacetime) |

LPIPS-Alex (lower better):

| Scene             | LPIPS (best paper) |
|-------------------|-------------------:|
| `coffee_martini`  | 0.142              |
| `cook_spinach`    | 0.099              |
| `cut_roasted_beef`| 0.090              |
| `flame_salmon_1`  | 0.123              |
| `flame_steak`     | 0.083              |
| `sear_steak`      | 0.075              |

**Citations:**
- Wu et al. *4D Gaussian Splatting for Real-Time Dynamic Scene Rendering*, CVPR 2024.
  Project: <https://guanjunwu.github.io/4dgs/>
- Yang et al. *Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction*, CVPR 2024.
- Li et al. *Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis*, CVPR 2024.
- Fridovich-Keil et al. *K-Planes: Explicit Radiance Fields in Space, Time, and Appearance*, CVPR 2023.

---

## Static 3D — Mip-NeRF 360 (Barron et al. 2022)

Seven static scenes; novel-view synthesis split (every 8th frame held out).

| Scene     | 3DGS (Kerbl+23) | Mip-Splatting (Yu+24) | Scaffold-GS (Lu+24) |
|-----------|----------------:|----------------------:|--------------------:|
| `bicycle` |           25.25 |                 25.50 |               24.50 |
| `flowers` |           21.52 |                 21.61 |               21.04 |
| `garden`  |           27.41 |                 27.40 |               27.17 |
| `stump`   |           26.55 |                 26.59 |               26.27 |
| `treehill`|           22.49 |                 22.66 |               23.10 |
| `room`    |           30.63 |                 31.51 |               32.13 |
| `counter` |           28.70 |                 29.22 |               29.34 |
| `kitchen` |           30.32 |                 31.52 |               31.29 |
| `bonsai`  |           31.98 |                 32.43 |               32.70 |

**Citations:**
- Kerbl et al. *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.
- Yu et al. *Mip-Splatting: Alias-free 3D Gaussian Splatting*, CVPR 2024.
- Lu et al. *Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering*, CVPR 2024.

---

## Tanks & Temples (static, intermediate scenes)

| Scene     | 3DGS  | Mip-Splatting |
|-----------|------:|--------------:|
| `Truck`   | 25.19 |         25.40 |
| `Train`   | 22.04 |         22.13 |

---

## Drift / staleness policy

These tables don't change once written; the underlying papers don't change.
If a *new* SOTA method publishes higher numbers, append a column rather than
replace — the old numbers help interpret older runs. Update when you genuinely
need a new ceiling.

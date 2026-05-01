# SOTA Verification Runbook

> **Goal.** Decide whether our 4D / static GS pipeline is currently SOTA-tier or
> not. The answer determines whether moving to cloud compute will improve
> quality or just cost money on a flat plateau.

> **TL;DR.** Run the strongest local preset on one published benchmark scene
> with NVS evaluation enabled. After training, run `scripts/sota_compare.py
> <scene>`. Read the verdict. Act on it.

---

## Why this matters

Cloud compute is only worth the spend if the *algorithm* is already SOTA-tier
and the bottleneck is genuinely hardware (longer training, higher resolution,
larger Gaussian budget). If we are 3+ dB below the published numbers on a
benchmark scene at our strongest preset, no amount of GPU-hours fixes that —
something is wrong with preprocessing, foundation models, or the algorithm
itself.

The verification protocol below is deliberately cheap to run *once*:
- 1 scene
- 1 strongest preset
- automatic NVS eval at the end
- ~6-10 hours on a single 8GB GPU
- one number to read at the end

---

## Recommended scenes (already in `data/`)

| Scene         | Mode    | Status in repo            | Notes                                                   |
|---------------|---------|---------------------------|---------------------------------------------------------|
| `flame_steak` | dynamic | **available** (21 cams)   | N3V benchmark. Strong 4DGaussians/Spacetime baselines.  |

**For static 3D verification:** download a Mip-NeRF 360 scene (e.g. `garden` or
`bonsai`) and place it under `data/<scene>/images/`. Then submit with
`mode=static, preset=premium`. Static benchmarks aren't pre-staged in the repo.

---

## How to run the verification (4D / `flame_steak`)

### Step 1 — Submit the job

Open the studio frontend (or call the API directly). Submit a 4D Dynamic job
with these settings:

- **Scene:** `flame_steak`
- **Preset:** `ultra` (or `ultra_clean` if you've seen streak artifacts before;
  `cloud` if the GPU is a 4090+ and you want max-quality)
- **NVS Evaluation:** **on** ← critical. The verification depends on this.
- **Foundation models:** on (default for these presets)

Equivalent CLI (if you'd rather not use the UI):

```bash
# Adjust the API_BASE if the backend isn't on localhost.
curl -X POST http://127.0.0.1:8000/process \
  -F "scene=flame_steak" \
  -F "mode=dynamic" \
  -F "preset=ultra" \
  -F "nvs_eval=true"
```

The job already has the data; **no video upload is needed** for a multi-view
scene that's pre-staged.

### Step 2 — Wait

Expected wall time on RTX 3060 Ti / 8 GB:

| Preset        | Training | Total (with foundation + COLMAP) |
|---------------|---------:|---------------------------------:|
| `ultra`       | ~6-8 h   | ~7-9 h                           |
| `ultra_clean` | ~6-8 h   | ~7-9 h                           |
| `cloud`       | ~3-4 h   | ~4-6 h (RTX 4090)                |

Faster GPUs scale roughly linearly. Monitor progress in the **Analiz** tab —
the new health banner will tell you mid-training whether the run is converging
or stalled.

### Step 3 — Read the verdict

Once the job finishes, NVS eval runs automatically and writes
`data/flame_steak/output/eval/nvs_eval.json`. Then:

```bash
python scripts/sota_compare.py flame_steak
```

You'll see one of three verdicts. **Act on it as follows.**

---

## Interpreting the verdict

### ✅ SOTA-tier  (ΔPSNR ≥ -1.0 dB vs best paper)

Algorithm is sound. The remaining quality lever is compute: longer training,
higher resolution, larger N cap. **This is exactly the case where cloud GPUs
pay off.**

Next step: scaffold cloud deployment (Path 2 from the strategy discussion —
RunPod-hosted backend, frontend points at remote `API_BASE`).

### ⚠ Below SOTA  (-3.0 ≤ ΔPSNR < -1.0 dB)

Tunable. Likely a combination of compute and configuration. Before paying for
cloud:

1. Re-run with a stronger preset on the same hardware (`ultra` → `ultra_clean`
   → `cloud`). Sometimes a 1-2 dB gap closes just by enabling more knobs.
2. Verify foundation models loaded the heavy variants:
   - `metric3d_model = metric3d_vit_large` (instead of `_small`)
   - CoTracker grid 30+
3. Check density schedule: `density_end_iter` should be ~80% of total iters,
   `max_gaussians` should be high enough not to clip (200k-500k for N3V scenes).

If a second pass with these tweaks stays below -1 dB, then it's worth trying
cloud — at the next preset tier with the same algorithm.

### ✗ Algorithmic gap  (ΔPSNR < -3.0 dB)

Cloud will not save you. Ranked culprits to investigate (cheapest first):

1. **Eval protocol mismatch.** Make sure we held out the right camera
   (`cam00` for N3V) and the right number of frames. Check
   `nvs_eval.json` → `held_out_cam` and `n_frames`.
2. **COLMAP / calibration.** Open `data/flame_steak/calibration.json` — are the
   camera poses sane? Check the orbit video — does it look "from the wrong
   angle"? If yes, COLMAP failed silently.
3. **Foundation model regression.** Run with `skip_foundation=true` and see if
   PSNR moves *up* — if so, foundation supervision is hurting (depth scale
   misaligned, masks oversegmenting).
4. **Density-control bug.** Check the analytics — if `n_points` plateaus far
   below `max_gaussians`, the density loop isn't growing fast enough.
5. **A missing algorithmic component.** Compare our pipeline summary
   (`output/logs/summary.json`) to the reference paper config. Anything
   notably absent (HexPlane, Fourier-K, motion regularizers)?

After fixing the root cause, re-run this verification.

---

## Sanity-checking the existing `flame_steak` output

`data/flame_steak/output/` already has a checkpoint from April — but it's a
**5000-iter smoke run** (per `output/logs/summary.json`), no NVS eval. That run
is far below SOTA *by design*; it's not informative about the pipeline's
ceiling.

Before kicking off the long verification run, you can confirm the harness
works end-to-end by submitting a fresh `smoke` preset job with NVS eval on:
takes ~10 minutes, will produce a (low) PSNR number, and proves
`sota_compare.py` reads it correctly. Then commit to the long `ultra` run.

---

## Multi-scene verification (optional, more work)

If a single scene's verdict feels noisy, repeat the protocol on 2-3 N3V scenes
(`flame_steak` + `cook_spinach` + `sear_steak` are well-behaved) and average
the deltas. A consistent ~-2 dB across three scenes is far more informative
than -2 dB on one.

This is *recommended before publishing or shipping*, not before deciding to
try cloud — for the cloud question, one decisive scene is enough.

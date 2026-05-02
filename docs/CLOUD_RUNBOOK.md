# Cloud SOTA Run — Step-by-Step Runbook (SV banana_demo + MV flame_steak)

> **Purpose:** Reproducible recipe for the `feat/sota-quality-fixes` cloud run on a fresh RunPod A100 80 GB. Tests **TWO scenes** alongside each other:
> - 🍌 **`banana_demo`** (single-view HyperNeRF — production-realistic case)
> - 🔥 **`flame_steak`** (multi-view N3V — paper-comparable benchmark)
>
> **Why two scenes:** SV is your real production target (users upload one phone video), MV gives you a paper-comparable PSNR number. Same pod, same backend, sequential job submissions (max_workers=1) so no resource contention.
>
> **Expected results:**
> - SV banana_demo: PSNR ~22-26 dB (HyperNeRF paper baseline range), ~1-1.5h
> - MV flame_steak: PSNR ~30-32 dB (N3V paper baseline 33.51 dB), ~2-3h
> - Total: ~3-5h pod time, ~$5-8

---

## Where commands run — colour codes

When you see a fenced code block, the **header above it** tells you where to run it:

- 🖥️ **LAPTOP — PowerShell** — your Windows machine, in PowerShell
- 🌐 **BROWSER** — runpod.io console (clicking, not typing)
- 🛰️ **POD — SSH terminal** — inside the RunPod pod, after you've SSH'd in
- 📱 **LAPTOP — Desktop App** — the 4DGS Viewer GUI (if you use it; CLI is in this runbook)

Mixing these up is the #1 cause of confusion — pay attention to the headers.

---

## Phase 0 — Pre-flight (do this BEFORE booting the pod)

### 🖥️ LAPTOP — PowerShell

```powershell
cd "C:\Users\TAHA\Desktop\gaussian-splatter\Gaussian Splatter\4dgs-studio"

# Make sure you have all the latest fixes:
git checkout feat/sota-quality-fixes
git pull
git log --oneline -5
```

**Expected output (latest commits should include):**
```
ea50d4a docs: cloud SOTA run runbook (every step, every shell, every gotcha)
2e5d5db feat(perf): wire FOURDGS_DATA_ROOT env var for /dev/shm staging
9e5bdbf fix(api): cap max_gaussians on smoke + micro presets
9794d50 fix(renderer): use rasterize_mode='antialiased' instead of antialiased=True
179c29f fix(config): dedupe lambda_accel + mip_scale_floor_frac defaults
```

If you don't see `ea50d4a` at the top → something's wrong with the pull. Stop and fix that first.

### 🖥️ LAPTOP — PowerShell

Verify SSH key + both scenes locally:

```powershell
Test-Path "$env:USERPROFILE\.ssh\id_ed25519"        # expect True
Test-Path "$env:USERPROFILE\.ssh\id_ed25519.pub"    # expect True

# Single-view banana_demo (we only need video.mp4 — cloud regenerates the rest):
"banana video.mp4: " + ((Test-Path "data\banana_demo\video.mp4") -and ((Get-Item "data\banana_demo\video.mp4").Length -gt 1MB))

# Multi-view flame_steak (we need videos/, calibration.json, poses_bounds.npy):
"flame videos:        " + (Get-ChildItem "data\flame_steak\videos" 2>$null).Count
"flame calibration:   " + (Test-Path "data\flame_steak\calibration.json")
"flame poses_bounds:  " + (Test-Path "data\flame_steak\poses_bounds.npy")
```

**Expected:**
- `banana video.mp4: True` (~49 MB)
- `flame videos: 21`
- `flame calibration: True`
- `flame poses_bounds: True`

If anything's missing, fix it before proceeding.

---

## Phase 1 — Boot the pod (~2 min)

### 🌐 BROWSER — runpod.io console

1. Open https://www.runpod.io/console/pods → click **+ Deploy**.
2. Filters / config:
   - **GPU:** `A100 80GB PCIe` (Community Cloud, ~$1.50/h) or `A100 80GB SXM4` if PCIe is unavailable.
   - **Container image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
     (The `devel` variant is non-negotiable — gsplat needs the CUDA toolchain to JIT-compile.)
   - **Container Disk:** `30 GB`
   - **Volume:** create persistent `/workspace` of **150 GB**
     (Two scenes' foundation outputs + PLY checkpoints can hit ~80-100 GB — 150 GB gives margin.)
   - **Expose HTTP Ports:** add `8000`
   - **SSH:** make sure your SSH public key is in **Settings → SSH Public Keys** (one-time setup).
3. Click **Deploy On-Demand**. Wait for status `Running`.
4. Click the pod's **Connect** button. Copy two things and save in a notepad:
   - **Public URL** — `https://<pod-id>-8000.proxy.runpod.net`
   - **SSH command** — `ssh root@<HOST> -p <PORT> -i ~/.ssh/id_ed25519`. Extract `<HOST>` and `<PORT>`.

---

## Phase 2 — Bootstrap the pod (~5-10 min)

### 🛰️ POD — SSH terminal

Open a fresh PowerShell on your laptop and run the SSH command from Phase 1. You'll land at `root@<container-id>:/#`.

```bash
# Fetch the bootstrap script from the new branch:
curl -fsSL \
  https://raw.githubusercontent.com/mehmettahacumurcu/gaussian-splatter/feat/sota-quality-fixes/deploy/runpod/pod-bootstrap.sh \
  -o /tmp/pb.sh

# Run it. Installs miniconda (to /workspace/miniconda — survives pod restart),
# CUDA-enabled COLMAP from conda-forge, all pip deps, clones repo.
bash /tmp/pb.sh \
  https://github.com/mehmettahacumurcu/gaussian-splatter.git \
  feat/sota-quality-fixes
```

Wait for `==> Bootstrap complete.` (~5-10 min).

### 🛰️ POD — SSH terminal (verify)

```bash
which colmap                                      # /usr/bin/colmap (symlink → conda)
ls /workspace/miniconda/bin/python                # exists
ls /workspace/4dgs-studio/                        # has backend/, deploy/, frontend/, scripts/
df -h /workspace                                  # should read total ~150G
```

---

## Phase 3 — Upload both scenes from laptop (~10-30 min)

We upload only the **raw inputs**. Cloud regenerates preprocessing from scratch — that's the production realistic case AND lets us verify the SV pipeline (COLMAP runs from scratch on banana since there's no calibration to bypass).

### 🖥️ LAPTOP — PowerShell (open a NEW window — keep the SSH window open)

```powershell
# Set variables (replace with YOUR values from Phase 1):
$POD_HOST = "157.157.x.x"           # ← from your SSH command
$POD_PORT = "12345"                  # ← from your SSH command
$POD_KEY  = "$env:USERPROFILE\.ssh\id_ed25519"

# Make destination dirs on the pod:
ssh -p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no root@$POD_HOST `
  'mkdir -p /workspace/4dgs-studio/data/banana_demo /workspace/4dgs-studio/data/flame_steak'

# Upload SV banana_demo (just the video — ~49 MB, ~1-2 min):
scp -P $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no `
  "data\banana_demo\video.mp4" `
  root@${POD_HOST}:/workspace/4dgs-studio/data/banana_demo/

# Upload MV flame_steak (videos + calibration — ~1.1 GB, ~10-30 min):
scp -r -P $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no `
  "data\flame_steak\videos" `
  "data\flame_steak\calibration.json" `
  "data\flame_steak\poses_bounds.npy" `
  root@${POD_HOST}:/workspace/4dgs-studio/data/flame_steak/
```

### 🖥️ LAPTOP — PowerShell

If your SCP gets dropped mid-upload:
```powershell
# Wipe partial state on the pod, then re-run:
ssh -p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no root@$POD_HOST `
  'rm -rf /workspace/4dgs-studio/data/flame_steak/videos'
# (Then repeat the relevant scp command.)
```

### 🛰️ POD — SSH terminal (verify upload)

```bash
ls /workspace/4dgs-studio/data/banana_demo/
# Expect: video.mp4

ls /workspace/4dgs-studio/data/flame_steak/
# Expect: calibration.json, poses_bounds.npy, videos/

ls /workspace/4dgs-studio/data/flame_steak/videos/ | wc -l
# Expect: 21

du -sh /workspace/4dgs-studio/data/banana_demo/ /workspace/4dgs-studio/data/flame_steak/
# Expect: banana ~49M, flame ~1.1G
```

---

## Phase 4 — Stage BOTH scenes to RAM disk (~30 sec total)

### 🛰️ POD — SSH terminal

```bash
# Stage banana (small, ~5 sec):
bash /workspace/4dgs-studio/deploy/runpod/stage_dataset.sh banana_demo

# Stage flame_steak (small upload — just videos, ~30 sec):
bash /workspace/4dgs-studio/deploy/runpod/stage_dataset.sh flame_steak
```

For each, you should see:
```
[stage] copying ... → /dev/shm/4dgs-studio/data/<scene>
[stage] symlink: ... output -> /workspace/.../output
[stage] symlink: ... .cache_markers -> /workspace/.../.cache_markers
============================================================
  NEXT STEP: export FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data
============================================================
```

### 🛰️ POD — SSH terminal (verify staging)

```bash
ls /dev/shm/4dgs-studio/data/
# Expect: banana_demo/  flame_steak/

# Confirm the cache_markers + output dirs are SYMLINKS (not real dirs):
ls -la /dev/shm/4dgs-studio/data/banana_demo/.cache_markers
# Expect: ... .cache_markers -> /workspace/4dgs-studio/data/banana_demo/.cache_markers
ls -la /dev/shm/4dgs-studio/data/flame_steak/.cache_markers
# Same pattern for flame_steak

df -h /dev/shm
# Expect: usage well under capacity (we only copied raw inputs)
```

---

## Phase 5 — Start the backend (~10 sec)

### 🛰️ POD — SSH terminal

```bash
cd /workspace/4dgs-studio

# Generate auth token. SAVE this to your notepad:
export RUNPOD_AUTH_TOKEN="$(openssl rand -hex 32)"
echo "============================================"
echo "  TOKEN: $RUNPOD_AUTH_TOKEN"
echo "============================================"

# CRITICAL: point the backend at /dev/shm BEFORE launching uvicorn:
export FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data

# Pin the model + extension caches to /workspace (survives pod restart):
export HF_HOME=/workspace/.cache/huggingface
export TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512   # NO expandable_segments

# Launch uvicorn in background:
nohup bash deploy/runpod/start.sh > /workspace/backend.log 2>&1 &
disown

# Wait + verify:
sleep 5
tail -n 20 /workspace/backend.log
```

**Look for these specific lines in the tail:**
```
[start.sh] Project dir: /workspace/4dgs-studio
[start.sh] Bearer-token auth enabled (token len=64)
[config] DATA_ROOT overridden via FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data    ← CRITICAL
INFO:     Uvicorn running on http://0.0.0.0:8000
[api] JobManager hazır (max_workers=1, GPU sıralaması aktif)
```

**If `[config] DATA_ROOT overridden` is missing** — env var wasn't set before launch. Fix:
```bash
pkill -f 'uvicorn backend.api' || true
# Re-export FOURDGS_DATA_ROOT (and the other env vars), re-launch.
```

---

## Phase 6 — SV banana_demo smoke (~30-45 min, ~$0.75)

This is the safety check. Banana is our smaller scene, runs fastest, and validates the **single-view path** (COLMAP from scratch + SV foundation + SV training). If THIS works, both real runs are highly likely to work.

### 🛰️ POD — SSH terminal

```bash
TOKEN="$RUNPOD_AUTH_TOKEN"

# Submit SV banana smoke. Foundation runs from scratch (~25-35 min on A100),
# then 500 iters of training (~3-5 min).
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "scene=banana_demo" \
  -F "mode=dynamic" \
  -F "preset=smoke" \
  -F "nvs_eval=1" \
  -F "lambda_lpips=0.1" \
  -F "lpips_net=alex" \
  -F "lambda_flow=0.05" \
  -F "lr_cam_K=0.0000001" \
  -F "lr_cam_w2c=0.0000001" \
  -F "cam_refine_start_iter=200" \
  http://localhost:8000/process
```

**Expected response:**
```json
{"job_id":"<uuid>","status":"running","status_url":"/status/<uuid>","message":"Job kuyruğa alındı"}
```

> Note: `lambda_multiview_consistency` is NOT overridden — it auto-skips on SV scenes (no second cam to render).

### 🛰️ POD — SSH terminal (in a SECOND shell, monitor)

```bash
# Filtered tail (no polling spam):
tail -f /workspace/backend.log | grep --line-buffered -vE "GET /|HTTP/1.1"
```

You should see (in order):
```
[v5.0] Single-view scene: banana_demo
[ffmpeg] frame extraction (343 frames)
[colmap] feature_extractor → feature_matcher → mapper           ← runs from scratch (~5-10 min)
[Faz 3] Single-view foundation (depth + tracks + masks)
  [metric3d] vit_large depth                                    ← ~10-15 min
  [cotracker] tracking grid                                    ← ~5-10 min
  [sam2] dynamic mask                                           ← ~5 min
[Faz 4] Gaussian model + DeformationField
[train] iter 50 ... loss=... psnr=...
[eval] PSNR=...
status: completed
```

### 🛰️ POD — SSH terminal (third shell, status snapshot)

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/jobs | python3 -m json.tool | head -50
```

### 🟢 GO / 🔴 NO-GO decision

✅ **GO** (proceed to Phase 7): banana smoke `status=completed`, no error, eval shows PSNR > 0.

❌ **NO-GO**: any traceback in error field. Most fixes are 1-line. Ping me with the error.

> **Why we run banana smoke first (not flame_steak smoke):** banana is smaller → cheaper validation. AND it tests the SV path which is where most of our SOTA-tier defaults are unverified. If SV smoke passes, MV will almost certainly pass.

---

## Phase 7 — The two real runs (sequential, max_workers=1)

The job manager runs ONE job at a time. We submit both back-to-back; banana finishes first, then flame_steak starts automatically.

### Phase 7a — SV banana_demo real run (~1-1.5h)

#### 🛰️ POD — SSH terminal

```bash
# Cloud preset has all SOTA-tier knobs baked in. Override iters=100k for max quality.
# lambda_multiview_consistency in cloud preset auto-skips on SV (no error).
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "scene=banana_demo" \
  -F "mode=dynamic" \
  -F "preset=cloud" \
  -F "nvs_eval=1" \
  -F "iters=100000" \
  http://localhost:8000/process
```

**Foundation will be CACHE HIT** (smoke generated everything). So this run is just init + 100k iters of training + eval.
- Expected wall time: **45 min - 1.5h** on A100 (small scene, ~25-35 it/s with /dev/shm)
- Expected PSNR: **22-26 dB** (HyperNeRF range)

### Phase 7b — MV flame_steak real run (queues automatically after 7a)

#### 🛰️ POD — SSH terminal (don't wait for 7a — queue this immediately)

```bash
# Same recipe, different scene. JobManager will queue it behind banana.
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "scene=flame_steak" \
  -F "mode=dynamic" \
  -F "preset=cloud" \
  -F "nvs_eval=1" \
  -F "iters=100000" \
  http://localhost:8000/process
```

**This will start AFTER banana's real run finishes.**
- Foundation runs from scratch on flame_steak (~1.5-2h — vit_large depth + RAFT flow + Farneback masks for 21 cams × 100 frames)
- Then 100k iters training (~1-2h)
- Total wall time: **2.5-4h** after banana finishes
- Expected PSNR: **30-32 dB** (N3V flame_steak baseline 33.51 dB)

> **Why no flame_steak smoke?** The banana smoke already validated that the pipeline boots, the new losses fire, gsplat works, env vars work. The only NEW thing flame_steak adds is the MV-specific code paths (`lambda_multiview_consistency`, calibration bypass). Those are well-tested locally already. Saving the ~$3 of a redundant smoke.
>
> **If you want max safety:** add a flame_steak smoke before 7b. It'll hit cache for masks/depth that we ran in earlier sessions IF we'd uploaded them — but since we only uploaded videos+calibration, foundation will run from scratch (~1.5-2h, ~$2.50 wasted on a redundant validation). Not worth it.

---

## Phase 8 — Monitor (~3-5 hours total)

### 🛰️ POD — SSH terminal (filtered tail in shell #2)

```bash
tail -f /workspace/backend.log | grep --line-buffered -vE "GET /|HTTP/1.1"
```

### 🛰️ POD — SSH terminal (status snapshot in shell #3)

```bash
# Run periodically:
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/jobs | python3 -c "
import json, sys
data = json.load(sys.stdin)
for j in data.get('jobs', []):
    p = j['phase']
    print(f\"  {j['scene']:15} {j['status']:10} phase={p['name']:12} prog={p['progress']*100:5.1f}%  msg={p['message'][:60]}\")
"

# GPU usage:
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader
```

### Healthy progression milestones

#### Banana_demo (SV) — Phase 7a
| Wall time | Phase | What you should see |
|---|---|---|
| 0-1 min | `frames` | ffmpeg extracts ~343 frames |
| 1-10 min | `colmap` | mapper registers cameras (NO calibration bypass on SV) |
| 10-35 min | `foundation` | Metric3D + CoTracker + SAM2 (or cache hit if smoke ran) |
| 35-45 min | `init` | model build, gsplat JIT (~80 sec) |
| 45 min - end | `training` | `it/s` plateaus at ~25-40 it/s |
| ~1.5h total | `completed` | PSNR ~22-26 dB |

#### Flame_steak (MV) — Phase 7b
| Wall time (from 7b start) | Phase | What you should see |
|---|---|---|
| 0-2 min | `frames_mv` + `colmap_mv` | "skipping COLMAP, using N3V poses" — calibration bypass |
| 2 min - 1.5h | `foundation` | depth_mv + masks_mv + flow_mv from scratch |
| 1.5h - 1.6h | `init` | model build for 21-cam scene |
| 1.6h - end | `training` | bigger N → ~15-25 it/s plateau |
| ~3-4h total | `completed` | PSNR ~30-32 dB |

### Health checkpoints — interrupt the run if these fail

#### For each scene:
| At iter | Expected | If you see... |
|---|---|---|
| 5,000 | PSNR ≥ 18 dB (banana) / 22 dB (flame_steak) | Below → paste the log + Analiz tab snapshot |
| 30,000 | PSNR ≥ 22 dB / 28 dB | Below → joint BA might be diverging |
| 60,000 | PSNR ≥ 24 dB / 30 dB | Below → check what's missing |
| 100,000 (final) | PSNR target met | See Phase 9 |

---

## Phase 9 — Read the verdicts

### 🛰️ POD — SSH terminal

```bash
# Show both completed jobs with full result + eval PSNR:
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/jobs | python3 -m json.tool
```

Look at each job's `result.eval` — `PSNR=NN.NN, orbit=ok`.

### Compare to baselines

| Scene | Result | Paper baseline | Verdict thresholds |
|---|---|---|---|
| 🍌 **banana_demo** (SV HyperNeRF) | (your PSNR) | ~22-25 dB (HyperNeRF) | ≥24 dB ✅ paper-tier; 20-24 dB ⚠ tunable; <20 dB ✗ gap |
| 🔥 **flame_steak** (MV N3V) | (your PSNR) | **33.51 dB** (Spacetime Gaussians) | ≥32.5 dB ✅ SOTA-tier; 30.5-32.5 ⚠ tunable; <30.5 ✗ gap |

### 🖥️ LAPTOP — PowerShell (download both sets of artifacts)

```powershell
$POD_HOST = "157.157.x.x"     # same as Phase 3
$POD_PORT = "12345"
$POD_KEY  = "$env:USERPROFILE\.ssh\id_ed25519"

mkdir cloud_results -ErrorAction SilentlyContinue
mkdir cloud_results\banana_demo -ErrorAction SilentlyContinue
mkdir cloud_results\flame_steak -ErrorAction SilentlyContinue

# Banana artifacts:
scp -P $POD_PORT -i $POD_KEY root@${POD_HOST}:/workspace/4dgs-studio/data/banana_demo/output/eval/nvs_eval.json `
  cloud_results\banana_demo\nvs_eval.json
scp -P $POD_PORT -i $POD_KEY root@${POD_HOST}:/workspace/4dgs-studio/data/banana_demo/output/eval/orbit.mp4 `
  cloud_results\banana_demo\orbit.mp4

# Flame_steak artifacts:
scp -P $POD_PORT -i $POD_KEY root@${POD_HOST}:/workspace/4dgs-studio/data/flame_steak/output/eval/nvs_eval.json `
  cloud_results\flame_steak\nvs_eval.json
scp -P $POD_PORT -i $POD_KEY root@${POD_HOST}:/workspace/4dgs-studio/data/flame_steak/output/eval/orbit.mp4 `
  cloud_results\flame_steak\orbit.mp4
```

### 🖥️ LAPTOP — PowerShell (run the comparator)

```powershell
& "E:\anaconda3\envs\gs4d\python.exe" scripts/sota_compare.py flame_steak `
  --eval-path cloud_results\flame_steak\nvs_eval.json
```

(`sota_compare.py` may not have a banana baseline — check the script. If not, just eyeball the PSNR.)

---

## Phase 10 — Tear down (don't forget!)

### 🌐 BROWSER — runpod.io console

Click your pod → **Stop** (paused billing, ~$0.07/h while stopped, volume kept) or **Terminate** (deletes everything, $0).

For tomorrow's verification: **Stop**. If you don't plan another run within a week: **Terminate** (after pulling artifacts).

---

## Common failure modes + fixes

### "TypeError: rasterization() got an unexpected keyword argument 'rasterize_mode'"

The pod's pip pulled a different gsplat than 1.5.x.

**Fix on the pod:**
```bash
/usr/local/bin/python -c "import gsplat; print(gsplat.__version__)"
/usr/local/bin/python -m pip install --force-reinstall gsplat==1.5.3
pkill -f 'uvicorn backend.api'
# (Re-export env vars + nohup ...)
```

### Banana COLMAP fails on the SV path

Banana_demo is a normal phone-style video — COLMAP usually works. But if it fails:
```bash
# Check the COLMAP log:
grep -A 30 "colmap" /workspace/backend.log | tail -50

# Common causes: too few features (try lower fps), too fast camera motion (sequential matching fails),
# blurry frames. Try with overrides:
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -F "scene=banana_demo" -F "mode=dynamic" -F "preset=smoke" -F "nvs_eval=1" \
  -F "fps=15"  -F "colmap_matching=exhaustive" \
  http://localhost:8000/process
```

### Foundation phase runs even though we expected cache hits

Cache markers were lost (probably symlink issue or stale copy on /workspace).

**Fix:**
```bash
ls -la /dev/shm/4dgs-studio/data/banana_demo/.cache_markers
# Expected: ... .cache_markers -> /workspace/4dgs-studio/data/banana_demo/.cache_markers
# If it's a real dir, redo Phase 4 (re-stage).
```

### Pod runs out of disk during flow generation

Volume is full.

**Fix:**
```bash
df -h /workspace
# If >95%: most likely PLY checkpoints. Keep only latest few:
cd /workspace/4dgs-studio/data/<scene>/output/ply/
ls -t | tail -n +5 | xargs rm -f
```

OR resize volume in RunPod console (requires pod restart).

### `[config] DATA_ROOT overridden ...` line is missing

Forgot to `export FOURDGS_DATA_ROOT=...` BEFORE launching uvicorn.

**Fix:**
```bash
pkill -f 'uvicorn backend.api'
cd /workspace/4dgs-studio
export FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data
export RUNPOD_AUTH_TOKEN="<your-saved-token>"
export HF_HOME=/workspace/.cache/huggingface
export TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
nohup bash deploy/runpod/start.sh > /workspace/backend.log 2>&1 &
disown
sleep 5
grep "DATA_ROOT overridden" /workspace/backend.log
```

### Backend dies silently mid-run

Same recovery — pkill + restart. Cache markers persist (via symlink to /workspace) so the resumed run skips redone preprocessing.

---

## Cost summary (worst-case, A100 80GB at $1.50/h)

| Phase | Time | Cost |
|---|---|---|
| Bootstrap | 10 min | $0.25 |
| Data upload (your end) | 10-30 min | $0.25-0.75 |
| 🍌 Banana smoke + foundation | 30-45 min | $0.75-1.13 |
| 🍌 Banana real run (100k iters, foundation cached) | 45 min - 1.5h | $1.13-2.25 |
| 🔥 Flame_steak real run (foundation runs from scratch + 100k iters) | 2.5-4h | $3.75-6.00 |
| Eval + tear-down | 10-30 min | $0.25-0.75 |
| **TOTAL** | **~4-7h** | **$6-11** |

If smoke fails and you have to debug: add another ~$1-2.

---

## TL;DR — paste these in order

1. 🌐 **BROWSER**: deploy A100 80GB, 150 GB volume, expose 8000.
2. 🛰️ **POD**: bootstrap (`bash /tmp/pb.sh ...`)
3. 🖥️ **LAPTOP**: scp banana_demo/video.mp4 + flame_steak/{videos, calibration.json, poses_bounds.npy}
4. 🛰️ **POD**: `bash deploy/runpod/stage_dataset.sh banana_demo` then `... flame_steak`
5. 🛰️ **POD**: export env vars (FOURDGS_DATA_ROOT, RUNPOD_AUTH_TOKEN, ...) + start uvicorn
6. 🛰️ **POD**: 🍌 banana smoke curl → wait ~30-45 min → verify completed
7. 🛰️ **POD**: 🍌 banana real run curl with `iters=100000`
8. 🛰️ **POD**: 🔥 flame_steak real run curl (queues behind banana)
9. 🛰️ **POD**: monitor for ~3-5 hours total
10. 🖥️ **LAPTOP**: scp both nvs_eval.json + orbit.mp4 back
11. 🌐 **BROWSER**: Stop the pod

If step 6 (banana smoke) passes, both real runs are essentially guaranteed.

Good luck. Two scenes = two PSNR numbers tomorrow. Let me know how each lands.

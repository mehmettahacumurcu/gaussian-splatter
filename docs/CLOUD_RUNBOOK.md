# Cloud SOTA Run — Step-by-Step Runbook

> **Purpose:** Reproducible recipe for the `feat/sota-quality-fixes` cloud run on a fresh RunPod A100 80 GB. Aimed at 100k-iter SOTA verification on `flame_steak` (N3V).
>
> **Expected result:** PSNR 30-32 dB, total time ~2-3 hours, cost ~$3-6.

---

## Where commands run — colour codes

When you see a fenced code block, the **header above it** tells you where to run it:

- 🖥️ **LAPTOP — PowerShell** — your Windows machine, in PowerShell
- 🌐 **BROWSER** — runpod.io console (clicking, not typing)
- 🛰️ **POD — SSH terminal** — inside the RunPod pod, after you've SSH'd in
- 📱 **LAPTOP — Desktop App** — the 4DGS Viewer GUI (if you decide to use it; CLI works too)

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
2e5d5db feat(perf): wire FOURDGS_DATA_ROOT env var for /dev/shm staging
9e5bdbf fix(api): cap max_gaussians on smoke + micro presets
9794d50 fix(renderer): use rasterize_mode='antialiased' instead of antialiased=True
179c29f fix(config): dedupe lambda_accel + mip_scale_floor_frac defaults
adc10de fix(trainer): corrupt-cache resilience in depth + frame loaders
```

If you don't see `2e5d5db` at the top → something's wrong with the pull. Stop and fix that first.

### 🖥️ LAPTOP — PowerShell

Verify your SSH key exists and your dataset is intact:

```powershell
Test-Path "$env:USERPROFILE\.ssh\id_ed25519"        # expect True
Test-Path "$env:USERPROFILE\.ssh\id_ed25519.pub"    # expect True

# Verify dataset
$base = "data\flame_steak"
"videos:           " + (Get-ChildItem "$base\videos" 2>$null).Count
"frames_multiview: " + (Get-ChildItem "$base\frames_multiview" -Directory 2>$null).Count + " cams"
"depth_multiview:  " + ((Get-ChildItem "$base\depth_multiview" -Recurse -File 2>$null) | Measure-Object).Count + " files"
"masks_multiview:  " + ((Get-ChildItem "$base\masks_multiview" -Recurse -File 2>$null) | Measure-Object).Count + " files"
"flow_multiview:   " + ((Get-ChildItem "$base\flow_multiview" -Recurse -File 2>$null) | Measure-Object).Count + " files"
```

**Expected (already verified):**
- videos: 21
- frames_multiview: 21 cams
- depth_multiview: 2100 files
- masks_multiview: 2100 files
- flow_multiview: 2079 files

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
     (was 80 GB before; we bumped to 150 because flow + depth + outputs at 1080p eat ~50-80 GB.)
   - **Expose HTTP Ports:** add `8000`
   - **SSH:** make sure your SSH public key is in **Settings → SSH Public Keys** (one-time setup).
3. Click **Deploy On-Demand**. Wait for status `Running`.
4. Click the pod's **Connect** button. From the modal, copy two things and save them in a notepad:
   - **Public URL** — looks like `https://<pod-id>-8000.proxy.runpod.net`. (Won't use today; nice to have.)
   - **SSH command** — looks like `ssh root@157.157.x.x -p 12345 -i ~/.ssh/id_ed25519`.
     Extract `<HOST>` (`157.157.x.x`) and `<PORT>` (`12345`).

---

## Phase 2 — Bootstrap the pod (~5 min)

### 🛰️ POD — SSH terminal

Open a fresh PowerShell on your laptop and run the SSH command from Phase 1. You should land at a `root@<container-id>:/#` prompt.

Then on the pod:

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

**This takes ~5-10 min the first time.** Watch for `==> Bootstrap complete.` at the end.

If it errors out at `conda install colmap+faiss` → check `/tmp/conda-colmap.log` and paste me the last 30 lines.

### 🛰️ POD — SSH terminal (verify)

```bash
which colmap                                      # /usr/bin/colmap (symlink to /workspace/miniconda/bin/colmap)
ls /workspace/miniconda/bin/python                # exists
ls /workspace/4dgs-studio/                        # repo cloned, contains backend/, deploy/, frontend/, scripts/
df -h /workspace                                  # your 150 GB volume — should read total ~150G
```

---

## Phase 3 — Upload data from laptop (~30-60 min, depending on upload speed)

The pod's `/workspace` is empty for `data/`. We need to ship the scene from your laptop.

### 🖥️ LAPTOP — PowerShell (open a NEW window — keep the SSH window open)

```powershell
# Set variables (replace with YOUR values from Phase 1):
$POD_HOST = "157.157.x.x"           # ← from your SSH command
$POD_PORT = "12345"                  # ← from your SSH command
$POD_KEY  = "$env:USERPROFILE\.ssh\id_ed25519"
$LOCAL    = "C:\Users\TAHA\Desktop\gaussian-splatter\Gaussian Splatter\4dgs-studio\data\flame_steak"

# Make destination dir on the pod:
ssh -p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no root@$POD_HOST `
  'mkdir -p /workspace/4dgs-studio/data/flame_steak'

# Upload only the inputs (~16 GB at full res — let cloud regenerate the 17 GB depth + 5 GB flow with bigger models locally we'd lose vit_giant2):
# OR upload everything (~25 GB) to skip foundation entirely. Both are valid; choose:

# OPTION A (recommended for SOTA quality — ~16 GB, ~30-60 min upload):
# Pod will regenerate depth with vit_giant2 (better than your local vit_small) and flow with RAFT.
scp -r -P $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no `
  "$LOCAL\videos" `
  "$LOCAL\calibration.json" `
  "$LOCAL\poses_bounds.npy" `
  root@${POD_HOST}:/workspace/4dgs-studio/data/flame_steak/

# OPTION B (if upload is too slow — ~6 GB, ~10-20 min upload, but uses YOUR locally-generated foundation):
# Skips foundation re-generation. Loses ~0.5 dB potentially because depth was vit_small not vit_giant2.
# scp -r -P $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no `
#   "$LOCAL\videos" "$LOCAL\calibration.json" "$LOCAL\poses_bounds.npy" `
#   "$LOCAL\depth_multiview" "$LOCAL\masks_multiview" "$LOCAL\flow_multiview" `
#   "$LOCAL\.cache_markers" `
#   root@${POD_HOST}:/workspace/4dgs-studio/data/flame_steak/
```

**Pick Option A** for tomorrow's run — the +0.5 dB from `vit_giant2` matters and the ~30 extra min on the pod is cheap ($1-1.50 extra).

### 🖥️ LAPTOP — PowerShell

If your SCP gets dropped mid-upload:
```powershell
# Wipe partial state on the pod, then re-run scp:
ssh -p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=no root@$POD_HOST `
  'rm -rf /workspace/4dgs-studio/data/flame_steak/videos'
# (Then repeat the scp command above.)
```

### 🛰️ POD — SSH terminal (verify upload)

```bash
ls /workspace/4dgs-studio/data/flame_steak/
# Expect: calibration.json, poses_bounds.npy, videos/

ls /workspace/4dgs-studio/data/flame_steak/videos/ | wc -l
# Expect: 21

du -sh /workspace/4dgs-studio/data/flame_steak/
# Expect: ~1.1 GB (Option A) or ~16-20 GB (Option B)
```

---

## Phase 4 — Stage data to RAM disk (Option A only, ~30 sec)

### 🛰️ POD — SSH terminal

```bash
# Run the staging script. Copies videos+calibration to /dev/shm,
# symlinks output/ + .cache_markers/ back to /workspace for persistence.
bash /workspace/4dgs-studio/deploy/runpod/stage_dataset.sh flame_steak
```

**Expected output:**
```
[stage] source 1.1 GB → /dev/shm (avail 100G)
[stage] copying ... → /dev/shm/4dgs-studio/data/flame_steak
[stage] done in N sec
[stage] symlink: ... output -> /workspace/.../output (writes persist on /workspace)
[stage] symlink: ... .cache_markers -> /workspace/.../.cache_markers
============================================================
  NEXT STEP: export the env var BEFORE starting the backend:
    export FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data
============================================================
```

If you went with **Option B** (full upload), staging won't help much — the data is already on /workspace. Skip this phase.

---

## Phase 5 — Start the backend (~10 sec)

### 🛰️ POD — SSH terminal

```bash
cd /workspace/4dgs-studio

# Generate auth token. SAVE this to your notepad (you'll paste it into curl + desktop app):
export RUNPOD_AUTH_TOKEN="$(openssl rand -hex 32)"
echo "============================================"
echo "  TOKEN: $RUNPOD_AUTH_TOKEN"
echo "  COPY THIS NOW — it's needed for every API call."
echo "============================================"

# CRITICAL: point the backend at /dev/shm BEFORE launching uvicorn (Phase 4 only):
export FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data

# Pin the model + extension caches to /workspace (survives pod restart):
export HF_HOME=/workspace/.cache/huggingface
export TORCH_EXTENSIONS_DIR=/workspace/.torch_extensions
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512   # NO expandable_segments — known gsplat crash

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
[start.sh] Using python: /usr/local/bin/python   (or /workspace/miniconda/bin/python)
[start.sh] Bearer-token auth enabled (token len=64)
[config] DATA_ROOT overridden via FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data    ← CRITICAL
INFO:     Uvicorn running on http://0.0.0.0:8000
[api] JobManager hazır (max_workers=1, GPU sıralaması aktif)
```

**If you don't see `[config] DATA_ROOT overridden ...`** — the env var wasn't set before launching uvicorn. Fix:
```bash
pkill -f 'uvicorn backend.api' || true
# Then re-export FOURDGS_DATA_ROOT and re-launch.
```

---

## Phase 6 — Pre-flight smoke (~5 min, ~$0.10)

This is the safety net. We submit a `smoke` preset with all the new SOTA losses overridden ON. If it completes, the real run will too.

### 🛰️ POD — SSH terminal

```bash
# Re-export TOKEN if you opened a new shell:
TOKEN="$RUNPOD_AUTH_TOKEN"

curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "scene=flame_steak" \
  -F "mode=dynamic" \
  -F "preset=smoke" \
  -F "nvs_eval=1" \
  -F "lambda_lpips=0.1" \
  -F "lpips_net=alex" \
  -F "lambda_flow=0.05" \
  -F "lambda_multiview_consistency=0.1" \
  -F "lr_cam_K=0.0000001" \
  -F "lr_cam_w2c=0.0000001" \
  -F "cam_refine_start_iter=200" \
  http://localhost:8000/process
```

**Expected response:**
```json
{"job_id":"<uuid>","status":"running","status_url":"/status/<uuid>","message":"Job kuyruğa alındı"}
```

### 🛰️ POD — SSH terminal (in a SECOND shell, monitor)

```bash
# Watch pipeline events filtered (no polling spam):
tail -f /workspace/backend.log | grep --line-buffered -vE "GET /|HTTP/1.1"
```

You should see (in order):
- `[bootstrap] use_provided_poses=True ... → skipping COLMAP, using N3V poses directly`
- `[Faz 3-MV] Multi-view per-cam foundation` followed by depth/masks/flow generation
  - First-time foundation will take **~1.5-2 hours** (vit_giant2 depth + RAFT flow)
  - **This is unavoidable on a fresh pod** — flow and depth aren't cached yet
- `[Faz 4] Gaussian model + DeformationField`
- `[train] iter 50 ... loss=...  psnr=...`
- `[eval] PSNR=...`
- `status: completed`

### 🛰️ POD — SSH terminal (check final state)

After ~5-10 min from the smoke submit (foundation runs THEN smoke trains):
```bash
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/jobs | python3 -m json.tool
```

**Two outcomes:**

✅ **GO**: Last job has `status=completed`, no `error`, `eval` shows a PSNR number > 0.
→ Proceed to Phase 7 (the real run).

❌ **NO-GO**: `status=failed` with a traceback in the `error` field.
→ Paste the full error to me. Most fixes are 1-line edits I can make remotely while your pod stays alive.

> **Note on smoke timing:** the FIRST smoke on a fresh pod takes ~1.5-2 hours because foundation runs from scratch. The smoke training itself is only ~5 min, but foundation generation is the long pole. After the first run, subsequent runs use cached depth/flow/masks (cache markers persist via the symlink).
>
> **Optimization:** if foundation has already run for the smoke, the real cloud run skips it (cache hit) and finishes much faster. So the 1.5-2 h foundation cost is paid ONCE.

---

## Phase 7 — The real cloud run (~1-2 hours after foundation, 100k iters)

### 🛰️ POD — SSH terminal

```bash
# Submit cloud preset with iters=100k override.
# Cloud preset already includes ALL the SOTA-tier knobs, so we just override iters
# and pass nvs_eval. No other overrides needed.
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "scene=flame_steak" \
  -F "mode=dynamic" \
  -F "preset=cloud" \
  -F "nvs_eval=1" \
  -F "iters=100000" \
  http://localhost:8000/process
```

**Expected response:**
```json
{"job_id":"<uuid>","status":"running",...}
```

### What the cloud preset has baked in (no overrides needed)

```
n_iters:                       60_000  →  100_000 (your override)
image_resolution:              1920×1080
batch_size:                    4
max_gaussians:                 0 (unlimited)
hexplane_resolution:           96
hexplane_feat_dim:             48
mlp_width:                     512
mlp_depth:                     4
metric3d_model:                vit_large    ← upgraded from vit_small
cotracker_grid_size:           60
lambda_lpips:                  0.1          ← perceptual loss ON
lpips_net:                     alex
lambda_multiview_consistency:  0.1          ← MV cross-cam supervision ON
lambda_flow:                   0.05         ← RAFT optical flow loss ON
lambda_aniso:                  0.02         ← anti-streak ON
lambda_accel:                  2e-4         ← temporal smoothness ON
lr_cam_K:                      1e-7         ← joint BA on intrinsics ON
lr_cam_w2c:                    1e-7         ← joint BA on extrinsics ON
cam_refine_start_iter:         8000
```

> **Why 100k iters?** Cloud preset's 60k default is conservative. Going to 100k:
> - +66% training time (~30-50 min extra at ~30 it/s)
> - Typical PSNR gain: +0.5 to +1.0 dB
> - Diminishing returns past 100k; 100k is the sweet spot for SOTA on N3V.

---

## Phase 8 — Monitor (~1-2 hours)

### 🛰️ POD — SSH terminal (in a second shell, leave open)

```bash
# Live filtered tail:
tail -f /workspace/backend.log | grep --line-buffered -vE "GET /|HTTP/1.1"
```

### 🛰️ POD — SSH terminal (third shell, periodic snapshot)

```bash
# Status check every ~5 min:
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/jobs | \
  python3 -c "import json, sys; j = json.load(sys.stdin)['jobs'][0]; \
              p = j['phase']; \
              print(f\"phase={p['name']}  msg={p['message']}  prog={p['progress']*100:.1f}%  status={j['status']}\")"

# GPU usage:
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader
```

### Healthy progression milestones

| Wall time from job submit | Phase | What you should see |
|---|---|---|
| 0-2 min | `frames_mv` + `colmap_mv` | both should be **cache hits** (we did smoke first) |
| 2-5 min | `foundation` | depth_mv + masks_mv + flow_mv all **cache hits** |
| 5-15 min | `init` | model build, gsplat JIT compile (~80 sec) |
| 15 min | `[train] iter 50` first appears | starts training |
| 15 min - end | training | `it/s` should plateau at **~25-35 it/s** if /dev/shm staging worked |
| ~1.5-2 hours | `[train] iter 100000` | training done |
| +20-30 min | `eval` | NVS metrics + orbit.mp4 rendering |
| ~2-2.5 h total | `status: completed` | **VERDICT READY** |

### Health checkpoints — interrupt the run if these fail

| At iter | Expected | If you see... |
|---|---|---|
| 5,000 | PSNR ≥ 22 dB | < 18 dB → something's wrong, paste the Analiz tab + log lines |
| 30,000 | PSNR ≥ 28 dB | < 25 dB → joint BA might be diverging; consider `-F "lr_cam_K=0"` re-submit |
| 60,000 | PSNR ≥ 30 dB | < 27 dB → quality features misconfigured |
| 100,000 (final) | PSNR ≥ 30-31 dB | < 28 dB → tunable; > 32 dB → SOTA-tier ✅ |

---

## Phase 9 — Read the verdict

### 🛰️ POD — SSH terminal

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/jobs | python3 -m json.tool
```

Look at the latest job's `result.eval` field — it'll have `PSNR=NN.NN, orbit=ok`.

### Compare to paper baseline

| Verdict | ΔPSNR vs paper baseline 33.51 dB | Result |
|---|---|---|
| ✅ **SOTA-tier** | ≥ −1 dB → ≥ **32.5 dB** | Algorithm is paper-class. Compute = the lever, you're done. |
| ⚠ **Below SOTA, tunable** | −3 to −1 dB → **30.5 - 32.5 dB** | Try larger overrides next time. Probably joint BA noise. |
| ✗ **Algorithmic gap** | < −3 dB → < **30.5 dB** | Pipeline has a bug — don't keep paying for cloud, debug locally. |

### 🖥️ LAPTOP — PowerShell (download the artifacts)

```powershell
# Pull the eval JSON + orbit.mp4 to your laptop:
$POD_HOST = "157.157.x.x"     # same as Phase 3
$POD_PORT = "12345"
$POD_KEY  = "$env:USERPROFILE\.ssh\id_ed25519"

scp -P $POD_PORT -i $POD_KEY root@${POD_HOST}:/workspace/4dgs-studio/data/flame_steak/output/eval/nvs_eval.json `
  flame_steak_nvs_eval.json
scp -P $POD_PORT -i $POD_KEY root@${POD_HOST}:/workspace/4dgs-studio/data/flame_steak/output/eval/orbit.mp4 `
  flame_steak_orbit.mp4

# Optional: pull all PLY checkpoints (4-8 GB, for local viewer):
mkdir flame_steak_ply
scp -r -P $POD_PORT -i $POD_KEY root@${POD_HOST}:/workspace/4dgs-studio/data/flame_steak/output/ply/ `
  flame_steak_ply/
```

### 🖥️ LAPTOP — PowerShell (run the comparator)

```powershell
& "E:\anaconda3\envs\gs4d\python.exe" scripts/sota_compare.py flame_steak --eval-path flame_steak_nvs_eval.json
```

This prints the verdict in human-readable form.

---

## Phase 10 — Tear down (don't forget!)

### 🌐 BROWSER — runpod.io console

Click your pod → **Stop** (or **Terminate**).

- **Stop** = pause billing for compute (~$0). Volume keeps your data + bootstrap state. ~$0.07/h while stopped.
- **Terminate** = also delete the volume. Recover NOTHING.

For tomorrow's verification: **Stop**. If you don't plan another run within a week, **Terminate** (after pulling artifacts).

---

## Common failure modes + fixes

### "TypeError: rasterization() got an unexpected keyword argument 'rasterize_mode'"

The pod's pip pulled a different gsplat than 1.5.x.

**Fix on the pod:**
```bash
# Check version:
/usr/local/bin/python -c "import gsplat; print(gsplat.__version__)"

# Force 1.5.3:
/usr/local/bin/python -m pip install --force-reinstall gsplat==1.5.3

# Restart backend (Phase 5 again):
pkill -f 'uvicorn backend.api'
# (re-export env vars + nohup ...)
```

### Foundation phase runs even though we expected cache hits

Cache markers were lost (probably symlink issue or stale copy on /workspace).

**Fix:**
```bash
# Verify symlink is in place:
ls -la /dev/shm/4dgs-studio/data/flame_steak/.cache_markers
# Expected: ... .cache_markers -> /workspace/4dgs-studio/data/flame_steak/.cache_markers

# If it's a real dir instead of a symlink, redo Phase 4.
```

### Pod runs out of disk during flow generation

Volume is full.

**Fix:**
```bash
df -h /workspace
# If >95%: delete unnecessary stuff. Most likely culprit: old PLY checkpoints.
ls -la /workspace/4dgs-studio/data/flame_steak/output/ply/
# Keep only the latest few:
cd /workspace/4dgs-studio/data/flame_steak/output/ply/
ls -t | tail -n +5 | xargs rm -f
```

OR resize volume in RunPod console (requires pod restart → re-bootstrap, ~5 min).

### `[config] DATA_ROOT overridden ...` line is missing from backend.log

You forgot to `export FOURDGS_DATA_ROOT=...` BEFORE launching uvicorn.

**Fix:**
```bash
pkill -f 'uvicorn backend.api'
cd /workspace/4dgs-studio
export FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data
# (also re-export RUNPOD_AUTH_TOKEN, HF_HOME, TORCH_EXTENSIONS_DIR, PYTORCH_CUDA_ALLOC_CONF)
nohup bash deploy/runpod/start.sh > /workspace/backend.log 2>&1 &
disown
sleep 5
grep "DATA_ROOT overridden" /workspace/backend.log
# Should now print the override line.
```

### Backend dies silently mid-run

Same recovery as above — pkill + restart + re-submit. Cache markers will let it resume.

---

## Cost summary

Worst-case (all phases, fresh pod, A100 80GB at $1.50/h):

| Phase | Time | Cost |
|---|---|---|
| Bootstrap | 10 min | $0.25 |
| Data upload (your end) | 30-60 min | $0.75-1.50 |
| Smoke pre-flight + foundation regen | 1.5-2 h | $2.25-3.00 |
| Real cloud run (100k iters, with cache hits from smoke) | 1-2 h | $1.50-3.00 |
| **Total** | **~3-5 h** | **$5-8** |

If smoke fails and you have to debug: add another ~$1-2 in pod time.

---

## TL;DR — paste these in order

1. 🌐 **BROWSER**: deploy A100 80GB, 150 GB volume, expose 8000.
2. 🛰️ **POD**: bootstrap (`bash /tmp/pb.sh ...`)
3. 🖥️ **LAPTOP**: scp scene to pod
4. 🛰️ **POD**: `bash deploy/runpod/stage_dataset.sh flame_steak`
5. 🛰️ **POD**: export env vars + start uvicorn
6. 🛰️ **POD**: smoke pre-flight curl → wait → verify completed
7. 🛰️ **POD**: real run curl with `iters=100000`
8. 🛰️ **POD**: monitor for ~2 hours
9. 🖥️ **LAPTOP**: scp artifacts back, run `sota_compare.py`
10. 🌐 **BROWSER**: Stop the pod

If step 6 (smoke) passes, step 7 is essentially guaranteed. The smoke is your safety net.

Good luck. Let me know how it goes.

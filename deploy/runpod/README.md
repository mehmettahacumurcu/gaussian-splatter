# RunPod Deployment — Cloud SOTA Verification

Run the SOTA verification (`docs/SOTA_VERIFICATION.md`) on a rented cloud GPU.
Your local 3060 Ti is too thin to give a clean verdict; on a 24-80 GB cloud
card the verdict actually means something.

This runbook is **phase by phase**. Each phase has a fixed end state — finish
the phase, then move on. If something goes wrong, the troubleshooting section
at the bottom maps phase → fix.

**Total cost for one verification run:** roughly **$2-15** depending on GPU
choice (see Phase 1) and **3-6 hours** of training wall time.

---

## Pick your GPU

You said 4090 is not enough. Here is the menu, ranked by quality ceiling.
For 4D SOTA on N3V, **A100 80GB** is the sweet spot — big enough VRAM to fit
1M+ Gaussians, half the price of H100, and perfectly competitive with paper
hardware.

| GPU                | VRAM   | RunPod cost (≈)  | Speed (rel.) | Recommended use                                |
|--------------------|-------:|-----------------:|-------------:|------------------------------------------------|
| RTX 4090           |  24 GB | $0.40 - 0.60 /h  | 1.0×         | Smoke / cheap iteration. Was your starting pt. |
| RTX 6000 Ada       |  48 GB | $0.80 - 1.00 /h  | 1.1×         | Bigger N cap than 4090, similar speed.         |
| A40                |  48 GB | $0.40 - 0.60 /h  | 0.7×         | Cheapest 48 GB option; older arch.             |
| **A100 80GB PCIe** |  80 GB | **$1.50 - 2.00 /h** | **1.5×**  | **Recommended for SOTA verification.**         |
| A100 80GB SXM      |  80 GB | $1.80 - 2.50 /h  | 1.6×         | Same VRAM as PCIe; SXM only matters multi-GPU. |
| H100 PCIe          |  80 GB | $2.50 - 3.00 /h  | 2.0×         | Faster A100; ~$15 for a 5h run.                |
| H100 SXM           |  80 GB | $3.00 - 4.00 /h  | 2.2×         | Fastest commonly available card.               |
| H200               | 141 GB | $4.00 - 5.00 /h  | 2.3×         | Overkill unless you push to 2M+ Gaussians.     |

Prices fluctuate — check the live RunPod console. **Community Cloud** is
usually 30-50% cheaper than **Secure Cloud**; Community is fine for a single
training run as long as you finish before someone with higher priority bumps
you. If you need uninterruptible, pick Secure.

### Hyperparam overrides for big-VRAM cards

The default `cloud` preset is sized for a 24 GB 4090. To actually use the
extra VRAM on an 80 GB card, override these via the Hyperparameter panel
(or pass them as form fields — see Phase 5).

Settings you set in the UI as **overrides** (everything else inherits from
the `cloud` preset):

| Field                   | 24 GB (default) | 48 GB             | 80 GB (A100/H100)  |
|-------------------------|-----------------|-------------------|--------------------|
| `max_gaussians`         | 1,000,000       | 1,500,000         | **2,500,000**      |
| `iters`                 | 60,000          | 80,000            | **100,000**        |
| `resolution`            | `1920x1080`     | `1920x1080`       | `1920x1080` (or `2560x1440` if you want max paper-tier; doubles training time) |
| `mlp_width`             | 512             | 640               | **768**            |
| `hexplane_resolution`   | 96              | 112               | **128**            |
| `metric3d_model`        | `metric3d_vit_small` | `metric3d_vit_large` | `metric3d_vit_giant2` |

The "**bold**" 80 GB column is what I'd suggest for the verification run.
This is closer to paper config and will exercise the algorithm at full
capacity. Expect ~4-6 hours wall time on an A100.

---

## Phase 0 — One-time RunPod account setup

Skip this section if you already use RunPod.

1. Sign up at <https://www.runpod.io/> and add a payment method. The
   minimum credit deposit is $10.
2. Generate an SSH key on your laptop if you don't already have one:
   - **Windows (Git Bash / PowerShell):** `ssh-keygen -t ed25519 -C "your@email"` (accept defaults).
   - The public key is at `~/.ssh/id_ed25519.pub` — copy its contents.
3. Add it in **RunPod Console → Settings → SSH Public Keys → New SSH Key**.
4. (Optional) install the [RunPod CLI](https://docs.runpod.io/cli/) — not
   required, the web UI works for everything below.

**Phase 0 done when:** you can log into RunPod and your SSH key is registered.

---

## Phase 1 — Start the pod (every run, 1-2 minutes)

1. Go to **RunPod Console → Pods → Deploy**.
2. Pick the GPU from the table above. **A100 80GB PCIe** is the recommendation.
3. **Container image:**
   ```
   runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
   ```
   The `devel` variant is essential — it has the CUDA toolchain `gsplat`
   needs to JIT-compile.
4. **Container Disk:** `30 GB` (enough for repo + model weights + outputs).
5. **Volume:** create a persistent **`/workspace` volume of 80 GB**. This
   caches foundation-model weights and lets you re-use across pod restarts
   without re-downloading.
6. **Expose HTTP Ports:** add port `8000`. RunPod gives you a public URL
   like `https://<pod-id>-8000.proxy.runpod.net` — write it down, you'll
   paste it into the desktop app.
7. Click **Deploy On-Demand**. First start takes ~1-2 minutes.
8. From the pod's **Connect** tab, copy the SSH command. It looks like
   `ssh root@<host>.proxy.runpod.net -p <port> -i ~/.ssh/id_ed25519`.

**Phase 1 done when:** the pod is "Running" and you've copied:
- the **public URL** (`https://<pod-id>-8000.proxy.runpod.net`)
- the **SSH command** (for Phase 2 + 3)

---

## Phase 2 — Set up the backend on the pod (every run, ~5 minutes)

SSH into the pod with the command from Phase 1, then run:

```bash
# 1. Fetch the bootstrap script directly + run it. Takes ~3-5 min first time.
#    The bootstrap clones the repo into /workspace/4dgs-studio itself —
#    no intermediate /workspace/repo clone needed.
#    If a stale /workspace/repo or /workspace/4dgs-studio exists from a
#    previous attempt on this volume, wipe them first:
#      rm -rf /workspace/repo /workspace/4dgs-studio
curl -fsSL \
  https://raw.githubusercontent.com/mehmettahacumurcu/gaussian-splatter/feat/sota-verification/deploy/runpod/pod-bootstrap.sh \
  -o /tmp/pod-bootstrap.sh
bash /tmp/pod-bootstrap.sh \
  https://github.com/mehmettahacumurcu/gaussian-splatter.git \
  feat/sota-verification

# 2. Move into the project root.
cd /workspace/4dgs-studio

# 3. Generate an auth token. Save it somewhere — you need it on your laptop.
export RUNPOD_AUTH_TOKEN="$(openssl rand -hex 32)"
echo
echo "==================================================================="
echo "  TOKEN: $RUNPOD_AUTH_TOKEN"
echo "  COPY THIS NOW — you will paste it into the desktop app."
echo "==================================================================="
echo

# 4. Start the backend in the background, log to a file.
nohup bash deploy/runpod/start.sh > /workspace/backend.log 2>&1 &
disown

# 5. Confirm it started.
sleep 3
tail -n 10 /workspace/backend.log
# You should see: "Uvicorn running on http://0.0.0.0:8000"
```

**Phase 2 done when:** `tail` shows `Uvicorn running on http://0.0.0.0:8000`.

> **Note:** the bootstrap installs **conda-forge's CUDA-enabled COLMAP** under
> `/workspace/miniconda` (not Ubuntu's apt build). The apt build forces
> software OpenGL on headless cloud GPUs and silently falls back to CPU SIFT
> — for a long video that's a 6-10x slowdown on preprocessing. With the
> conda build, COLMAP feature extraction + matching runs on the GPU at full
> speed.
>
> **Why `/workspace/miniconda` and not `/opt/miniconda`?** `/workspace` is
> the persistent volume — surviving pod stop/resume/resize — while `/opt`
> is the container disk that gets wiped each restart. Persisting miniconda
> means the ~10 min `conda install colmap+faiss` step only runs once per
> volume, not once per pod boot. Re-bootstraps on the same volume are
> near-instant.

### Performance optimization — stage the dataset to RAM

`/workspace` is a network filesystem (MFS). Per-file latency is high enough
that training stalls at ~3 it/s with 13% GPU utilization on an A100 — the
GPU spends most of its time waiting for the next frame. Copying the scene
to `/dev/shm` (RAM disk) bypasses MFS entirely and pushes training to
~30+ it/s.

After Phase 4 (data uploaded) and before Phase 5 (job submit), run on the
pod:

```bash
cd /workspace/4dgs-studio
bash deploy/runpod/stage_dataset.sh flame_steak
export FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data
# Restart the backend so it picks up the new DATA_ROOT:
pkill -f 'uvicorn backend.api' || true
nohup bash deploy/runpod/start.sh > /workspace/backend.log 2>&1 &
disown

# Verify the override took effect:
grep "DATA_ROOT overridden" /workspace/backend.log
# Expected: [config] DATA_ROOT overridden via FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data
```

`backend/config.py` reads `FOURDGS_DATA_ROOT` at module-load. If set, ALL
scene path lookups (frames, depth, masks, flow, output) route through
`/dev/shm` instead of `/workspace/4dgs-studio/data/`.

`stage_dataset.sh` automatically symlinks `output/` and `.cache_markers/`
inside the staged scene back to `/workspace/<scene>/`, so:
- **PLY checkpoints + eval renders persist** across pod restart (writes
  go through the symlink to MFS).
- **Cache markers persist** — re-runs of the same scene skip the 30-60
  min preprocessing if the markers are still valid.
- The bulk read traffic (frames + depth + masks + flow) stays on
  `/dev/shm` for the speedup.

The helper sanity-checks `/dev/shm` size before copying — flame_steak is
~6 GB, your pod's RAM should comfortably fit it. For huge datasets, set
`SOURCE`/`TARGET` env vars or use `rsync --exclude` patterns directly.

---

## Phase 3 — Connect your desktop app to the pod (~30 seconds)

This is the new part — no more curl wrangling.

1. Open the 4DGS Studio desktop app on your laptop. (If the backend was
   already running locally, leave it; the desktop app can switch between
   local and cloud at any time.)
2. In the topbar, click the **⚙ icon** next to the backend status badge.
3. The **Backend Connection** modal opens. Fill in:
   - **API Base URL:** the public URL from Phase 1
     (e.g. `https://abc123-8000.proxy.runpod.net`).
   - **Auth Token:** the token printed in Phase 2.
4. Click **Test connection**. Within ~1 second you should see:
   > ✓ Reachable. GPU available: `NVIDIA A100 80GB PCIe`.
   If you see ✗, check the troubleshooting section.
5. Click **Save**.

The badge in the topbar now shows `CLOUD Backend OK · GPU: ... · Aktif: 0`.
Every API call from now on goes to the pod.

**Phase 3 done when:** the topbar badge says **CLOUD** and the GPU name
matches your pod's GPU.

---

## Phase 4 — Upload the dataset (every run, 10-30 min on a home line)

`flame_steak/` is multi-GB, mostly preprocessed frames + flow + masks. From
**your laptop** (Git Bash on Windows, or any Unix shell):

```bash
# Substitute your local path + the SSH host/port/key from Phase 1.
LOCAL_DATA="C:/Users/TAHA/Desktop/gaussian-splatter/Gaussian Splatter/4dgs-studio/data/flame_steak"
POD_HOST="<the host from your SSH command>"
POD_PORT="<the port from your SSH command>"
POD_KEY="$HOME/.ssh/id_ed25519"

rsync -avzP --partial \
  -e "ssh -p $POD_PORT -i $POD_KEY" \
  "$LOCAL_DATA" \
  root@$POD_HOST:/workspace/4dgs-studio/data/
```

`--partial` resumes if your connection drops. If `rsync` isn't on Windows
Git Bash, install it via `pacman -S rsync` (MSYS2) or use:

```bash
scp -r -P $POD_PORT -i $POD_KEY \
  "$LOCAL_DATA" \
  root@$POD_HOST:/workspace/4dgs-studio/data/
```

(scp doesn't resume — be sure your connection is stable.)

**Phase 4 done when:** the upload completes and `ssh` to the pod shows the
data:

```bash
ssh root@$POD_HOST -p $POD_PORT 'ls /workspace/4dgs-studio/data/flame_steak/'
# expect: calibration.json colmap_multiview depth_multiview flame_steak ...
```

---

## Phase 5 — Submit the verification job (~10 seconds, then wait)

Now that the desktop app talks to the pod, just use the UI:

1. Open the **Yeni Job** tab.
2. Pick mode **🎬 4D Dynamic** (or just leave default).
3. **Sahne adı:** `flame_steak`.
4. **Preset:** `Cloud ☁` (the entry that says `RunPod / RTX 4090`). For
   bigger cards, also expand the **Hiperparametreler** panel and enter the
   80 GB overrides from the GPU table above.
5. Make sure the **NVS Evaluation** checkbox is **on**. This is how you
   get a verdict at the end.
6. Click **4D Dynamic Job başlat**.

Switch to the **Jobs** tab — your job should appear within 1-2 seconds with
status `queued` → `running`. Switch to the **Analiz** tab to watch live
metrics stream in.

If you want to do this from a script instead, the curl wrapper still works:

```bash
BACKEND_URL=https://<pod-id>-8000.proxy.runpod.net \
AUTH_TOKEN=<your token> \
PRESET=cloud \
bash deploy/runpod/bench_flame_steak.sh
```

**Phase 5 done when:** the Jobs tab shows the `flame_steak` job in
`running` state and the Analiz tab is drawing charts.

---

## Phase 6 — Wait + watch

Expected wall time:

| GPU         | Foundation | Training      | Eval      | Total          |
|-------------|-----------:|--------------:|----------:|---------------:|
| RTX 4090    | ~30-60 min | 3-4 h (60k)   | 10-20 min | ~4-5 h         |
| A100 80GB   | ~25-45 min | 3-4 h (100k)  | 15-30 min | **~4-6 h**     |
| H100 80GB   | ~20-30 min | 1.5-2 h (100k)| 15-30 min | ~2-3 h         |

Things to watch in the desktop app:

- **Analiz tab:** PSNR climbing into the high 20s / low 30s. If after 10k
  iters PSNR is below 22 dB, something's wrong — see troubleshooting.
- **Jobs tab:** the row's progress bar moves from 0% to 100%. The phase
  pill cycles `extract → colmap → foundation → train → eval`.
- **Eval tab:** stays empty until the very end (`eval` phase). When it
  populates, training is done.

You can close the desktop app and reopen it any time — connection settings
persist; the backend keeps running on the pod.

**Phase 6 done when:** the Jobs tab row goes `completed`. The Eval tab
shows PSNR/SSIM/LPIPS and an orbit video.

---

## Phase 7 — Read the verdict

You can do this from the desktop app or from a terminal.

### From the desktop app

The Eval tab now shows the held-out cam metrics. Compare PSNR to:
- **`flame_steak`** paper baseline: **33.51 dB** (Spacetime Gaussians).
- ΔPSNR ≥ -1 dB → **SOTA-tier** ✅. Compute pays off; pipeline is sound.
- -3 dB ≤ ΔPSNR < -1 dB → **Below SOTA** ⚠. Tunable; try larger overrides.
- ΔPSNR < -3 dB → **Algorithmic gap** ✗. Don't keep paying for cloud; fix
  the pipeline first. See `docs/SOTA_VERIFICATION.md` for triage.

### From a terminal (more detail)

```bash
# 1. Pull the eval JSON from the pod.
scp -P $POD_PORT -i $POD_KEY \
  root@$POD_HOST:/workspace/4dgs-studio/data/flame_steak/output/eval/nvs_eval.json \
  ./flame_steak_nvs_eval.json

# 2. Run the comparator (it prints the verdict).
python scripts/sota_compare.py flame_steak --eval-path flame_steak_nvs_eval.json
```

### Pulling the orbit video

The desktop app's Eval tab can play it directly (the `?token=` query-param
auth fallback handles `<video>` tags). Or download:

```bash
scp -P $POD_PORT -i $POD_KEY \
  root@$POD_HOST:/workspace/4dgs-studio/data/flame_steak/output/eval/orbit.mp4 \
  ./flame_steak_orbit.mp4
```

**Phase 7 done when:** you have a verdict (✅ / ⚠ / ✗).

---

## Phase 8 — Tear down

In the RunPod console, click **Stop** on the pod. Important:

- **Stop** stops billing for compute but keeps the volume (cheap).
- **Terminate** also deletes the volume (loses your foundation-model cache,
  bootstrap state, etc).

For a one-off verification: **Terminate** is fine if you don't plan another
run soon. For repeat runs: **Stop** the pod and just **Resume** it later —
your volume will still have everything pre-installed.

---

## Live PLY playback — known limitation

The Viewer tab fetches PLYs from the backend. Through RunPod's HTTP proxy
those fetches work, but high-frame-rate playback is bandwidth-bound (10-30
MB/s typical home connection vs. 200-500 MB/s local). Your verdict comes
from the Eval tab — the orbit video — not live PLY playback. To view the
trained scene smoothly:

```bash
# Pull all PLYs locally.
mkdir -p ~/4dgs-flame-steak-output
rsync -avzP -e "ssh -p $POD_PORT -i $POD_KEY" \
  root@$POD_HOST:/workspace/4dgs-studio/data/flame_steak/output/ply/ \
  ~/4dgs-flame-steak-output/

# Then point a *local* backend (set in the desktop app's connection
# settings: http://127.0.0.1:8000, no token) at the local data folder.
```

---

## Troubleshooting

### "Test connection" shows `✗ Failed to fetch`
- Did Phase 1 expose port `8000`? Re-check the pod's Connect tab.
- Is the URL exactly `https://<pod-id>-8000.proxy.runpod.net` (with
  `-8000-`, not `-22-`)?
- Did Phase 2 finish the bootstrap? `ssh` in and check
  `tail /workspace/backend.log`.

### "Test connection" shows `✗ HTTP 401: Missing or invalid bearer token`
- The token in the desktop app doesn't match `RUNPOD_AUTH_TOKEN` on the pod.
- Re-run on the pod: `echo $RUNPOD_AUTH_TOKEN` — check it matches.
- If you restarted the pod, the env var is gone — re-export it and restart
  the backend.

### Job submits but immediately fails with "scene not found"
- Phase 4 didn't put data in the right place. Verify:
  ```bash
  ssh root@$POD_HOST -p $POD_PORT 'ls /workspace/4dgs-studio/data/flame_steak/'
  ```
  Should list `calibration.json`, `frames_multiview/`, etc.

### Backend logs show `gpu_available: false`
- Either the pod was created without a GPU (re-create with one), or the
  RunPod template didn't pass through `--gpus all` (rare; raise a support
  ticket).

### gsplat first import takes 2-3 minutes
- Normal — JIT compile. Cached after that under `/workspace/.torch_extensions`.
- If you re-create the pod with a fresh volume, expect this once again.

### Out of memory on an A100 80GB
- You probably set `max_gaussians` too high *and* increased resolution.
  Drop one of them, or pull `mlp_width` back to 640.

### Live training metrics stop updating in Analiz
- Auto-refresh polls every 3s; if the pod's HTTP proxy is rate-limiting,
  refresh manually with the **Refresh** button. Otherwise check
  `tail -f /workspace/backend.log` on the pod for errors.

### Want to run the verification on a different scene
- Override via `SCENE=cook_spinach` in `bench_flame_steak.sh`, or just pick
  the scene name in the UI. Make sure the data is uploaded first.

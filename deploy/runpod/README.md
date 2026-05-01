# RunPod Deployment — SOTA Verification on a Cloud GPU

This folder packages the 4DGS Studio backend so you can run the SOTA
verification (`docs/SOTA_VERIFICATION.md`) on a rented RunPod 4090 instead of
your local 3060 Ti — the only way to get a verdict that isn't compute-capped
by 8 GB VRAM.

The end-to-end flow takes about **5 minutes of setup + ~3-5 hours of training**
and costs roughly **$2-4** at current RunPod rates ($0.40-0.60/hour for a 4090).

---

## What's in this folder

| File                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `Dockerfile`          | CUDA 12.1 + Python 3.11 + project deps + COLMAP. Bake-once image.       |
| `start.sh`            | Container entrypoint. Binds `0.0.0.0:8000`. Reads `RUNPOD_AUTH_TOKEN`.  |
| `pod-bootstrap.sh`    | Alternative to Docker: run on a stock RunPod PyTorch container.         |
| `bench_flame_steak.sh`| Submits the verification job from your laptop.                          |
| `.dockerignore`       | Keeps the build context lean.                                           |

---

## Prerequisites

- A RunPod account with a payment method.
- The branch with the verification harness pushed to a remote you can `git clone`
  from the pod. The default in `pod-bootstrap.sh` points at this repo's
  `feat/sota-verification` branch.
- About 3-5 GB free upload bandwidth (one-time `data/flame_steak/` push). On a
  reasonable home connection this is 5-30 minutes.
- An SSH client (Windows: WSL, Git Bash, or PowerShell with OpenSSH).

---

## Path A — Bootstrap on a stock RunPod image (recommended for first run)

This skips `docker build` entirely and just runs Python on the pod. Fastest path
to "is it working?"

### 1. Create the pod

1. Log in to <https://www.runpod.io/console/pods>.
2. Click **Deploy** → **GPU Pods**.
3. **GPU type:** `RTX 4090` (24 GB VRAM is essential). `RTX 3090` works too.
4. **Container image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
   (or whatever the latest `runpod/pytorch:*-cuda12.x-devel` is — the `devel`
   variant has the CUDA toolchain gsplat needs to JIT-compile).
5. **Container disk:** 30 GB (enough for repo + model weights + outputs).
6. **Volume:** create a persistent volume on `/workspace` (50 GB). This caches
   foundation-model weights between pod restarts.
7. **Expose HTTP ports:** add port `8000`. RunPod will give you a public URL
   like `https://<pod-id>-8000.proxy.runpod.net`.
8. **Start the pod.** First start takes ~1-2 minutes.

### 2. SSH into the pod

From the RunPod web UI, copy the SSH command (looks like
`ssh root@<host> -p <port> -i ~/.ssh/id_ed25519`). Run it on your laptop.

### 3. Bootstrap the repo

On the pod (you'll be `root@`):

```bash
# Pull the bootstrap script and run it. Takes ~3-5 minutes (apt + pip).
curl -sSL https://raw.githubusercontent.com/mehmettahacumurcu/gaussian-splatter/feat/sota-verification/4dgs-studio/deploy/runpod/pod-bootstrap.sh \
  | bash -s -- https://github.com/mehmettahacumurcu/gaussian-splatter.git feat/sota-verification

# OR if the repo is private / curl can't fetch raw:
git clone --depth 1 --branch feat/sota-verification https://github.com/mehmettahacumurcu/gaussian-splatter.git /workspace/repo
bash /workspace/repo/4dgs-studio/deploy/runpod/pod-bootstrap.sh \
     https://github.com/mehmettahacumurcu/gaussian-splatter.git feat/sota-verification
```

### 4. Generate an auth token + start the backend

```bash
cd /workspace/4dgs-studio
export RUNPOD_AUTH_TOKEN="$(openssl rand -hex 32)"
echo "TOKEN: $RUNPOD_AUTH_TOKEN"   # copy this — you'll need it from your laptop

# Optional: also allow your local Tauri origin via CORS.
# export CORS_ALLOW_ORIGINS="tauri://localhost,https://tauri.localhost"

# Start backend in the background, log to a file.
nohup bash deploy/runpod/start.sh > /workspace/backend.log 2>&1 &
disown
sleep 3
tail -n 20 /workspace/backend.log
```

You should see `Uvicorn running on http://0.0.0.0:8000`. Confirm from your
laptop:

```bash
curl -H "Authorization: Bearer $RUNPOD_AUTH_TOKEN" \
  https://<pod-id>-8000.proxy.runpod.net/
```

Expected: a JSON health blob with `gpu_available: true`.

### 5. Upload the dataset

From your laptop (use the same SSH connection details):

```bash
# Adjust path to your local 4dgs-studio repo.
LOCAL_DATA="C:/Users/TAHA/Desktop/gaussian-splatter/Gaussian Splatter/4dgs-studio/data/flame_steak"
POD_HOST="<host>"
POD_PORT="<port>"
POD_KEY="$HOME/.ssh/id_ed25519"

# rsync resumes on disconnect — better than scp for multi-GB uploads on a
# home connection.
rsync -avzP --partial -e "ssh -p $POD_PORT -i $POD_KEY" \
  "$LOCAL_DATA" \
  root@$POD_HOST:/workspace/4dgs-studio/data/
```

This uploads `frames_multiview/`, `colmap_multiview/`, `depth_multiview/`,
`flow_multiview/`, `masks_multiview/`, `videos/`, `calibration.json`,
`poses_bounds.npy` — about 3-5 GB. Expect 10-30 min on a typical home line.

### 6. Submit the verification job

From your laptop:

```bash
cd /path/to/local/4dgs-studio
BACKEND_URL=https://<pod-id>-8000.proxy.runpod.net \
AUTH_TOKEN=<the token you saved> \
PRESET=cloud \
bash deploy/runpod/bench_flame_steak.sh
```

This POSTs `/process` with `scene=flame_steak, mode=dynamic, preset=cloud,
nvs_eval=true, skip_foundation=false` and prints the `job_id`.

### 7. Wait

`cloud` preset on a 4090 takes ~3-5 hours total (foundation models ~30-60 min,
training 60k iters ~3-4 hours, NVS eval ~10-20 min).

Monitor on the pod:

```bash
ssh root@$POD_HOST -p $POD_PORT
tail -f /workspace/backend.log
```

Or poll status from your laptop:

```bash
watch -n 60 \
  "curl -s -H 'Authorization: Bearer $AUTH_TOKEN' \
     https://<pod-id>-8000.proxy.runpod.net/status/<job_id> | python -m json.tool"
```

The Analiz tab in the local Tauri app **will not** show this remote job's
metrics yet — that needs the frontend `API_BASE` toggle (separate work). For
now use the curl polling.

### 8. Pull results back

When the job's `status` is `completed`:

```bash
# Eval JSON (small — KB).
scp -P $POD_PORT -i $POD_KEY \
  root@$POD_HOST:/workspace/4dgs-studio/data/flame_steak/output/eval/nvs_eval.json \
  ./flame_steak_nvs_eval.json

# Orbit video (small — MB).
scp -P $POD_PORT -i $POD_KEY \
  root@$POD_HOST:/workspace/4dgs-studio/data/flame_steak/output/eval/orbit.mp4 \
  ./flame_steak_orbit.mp4

# (Optional) trained PLYs — multi-GB. Skip unless you want to view locally.
# rsync -avzP -e "ssh -p $POD_PORT -i $POD_KEY" \
#   root@$POD_HOST:/workspace/4dgs-studio/data/flame_steak/output/ply/ \
#   ./flame_steak_ply/
```

### 9. Read the verdict

```bash
python scripts/sota_compare.py flame_steak --eval-path flame_steak_nvs_eval.json
```

You'll see one of `SOTA-tier ✅` / `Below SOTA ⚠` / `Algorithmic gap ✗`. See
`docs/SOTA_VERIFICATION.md` for what to do with each.

### 10. Tear down the pod

Stop the pod from the RunPod UI. Storage volume bills separately — keep the
`/workspace` volume if you'll re-run, otherwise delete it.

---

## Path B — Build the Docker image (faster restarts; recommended after Path A works)

If you'll iterate, building the image once and running it cuts cold-start time
to ~30 s instead of 3-5 min.

```bash
# Build locally (or on the pod).
docker build -t 4dgs-studio:cu121 -f deploy/runpod/Dockerfile .

# Push to a registry RunPod can pull from (Docker Hub, GHCR, etc.):
docker tag 4dgs-studio:cu121 <your-registry>/4dgs-studio:cu121
docker push <your-registry>/4dgs-studio:cu121
```

In the RunPod UI, use `<your-registry>/4dgs-studio:cu121` as the container
image. RunPod will pull it on pod creation. Pod boot then runs `start.sh`
automatically; SSH in to set `RUNPOD_AUTH_TOKEN` and restart, or pass it via
the pod's environment-variables UI.

---

## Troubleshooting

**`401 Unauthorized` on every request from my laptop.**
You set `RUNPOD_AUTH_TOKEN` on the pod but didn't pass it from the client.
Always include `-H "Authorization: Bearer $AUTH_TOKEN"` in `curl`, or set
`AUTH_TOKEN=...` for `bench_flame_steak.sh`.

**`gsplat` import takes 2-3 minutes the first time.**
Normal — gsplat 1.5.3 JIT-compiles a CUDA extension. The compiled artifact is
cached under `/workspace/.torch_extensions` (persistent volume) so subsequent
imports are <1 s.

**`scp` / `rsync` is very slow.**
Home upload speeds. The flame_steak data is 3-5 GB; use `rsync --partial
--progress` so it resumes on disconnect, and start it during a meal.

**`curl` says `Could not resolve host: <pod-id>-8000.proxy.runpod.net`.**
Did you expose port 8000 in the RunPod pod settings? Check the pod's "Connect"
tab — you should see an HTTP service link there.

**Backend logs show `gpu_available: false`.**
Either: pod was created without a GPU (re-create with one), or the container
isn't using `--gpus all` (the RunPod template should do this automatically;
double-check in pod settings).

**Job fails with "scene not found".**
The data didn't upload to the right path. The pipeline expects
`/workspace/4dgs-studio/data/flame_steak/`. Verify with
`ssh root@... 'ls /workspace/4dgs-studio/data/flame_steak/'`.

**Out of memory on a 4090.**
The `cloud` preset is sized for 24 GB but with foundation models loaded the
peak can spike. Try `ultra` instead (8 GB cap) — slightly less aggressive.

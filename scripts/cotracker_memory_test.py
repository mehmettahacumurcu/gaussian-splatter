"""CoTracker memory diagnostic — neden 8 GB karta 4 GiB allocate edemiyor?

Bu script step-by-step şu durumu test eder:
1. Baseline GPU memory (Python henüz CUDA init etmeden)
2. CUDA context overhead (torch.cuda.init)
3. Saf 4 GiB single allocation (CoTracker'dan bağımsız WDDM cap testi)
4. CoTracker model load (henüz forward yok)
5. Video tensor upload (varying sizes)
6. CoTracker forward pass (gerçek tracking)

Her adımda:
- nvidia-smi total VRAM kullanımı (subprocess)
- torch.cuda.mem_get_info() (driver-level free)
- torch.cuda.memory_summary() (PyTorch reserved/allocated)

Kullanım:
    python scripts/cotracker_memory_test.py
    python scripts/cotracker_memory_test.py --frames-dir data/banana_demo/frames
    python scripts/cotracker_memory_test.py --skip-real-test  # sadece allocation testleri

Hedef: hangi adımda fail ettiğini göstererek WDDM cap mı, fragmentation mı,
context overhead mı, yoksa Tauri/browser leak'i mi bulmak.
"""
from __future__ import annotations
import argparse
import gc
import os
import subprocess
import sys
import time
from pathlib import Path

# expandable_segments default açık, ama explicit log için
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def nvidia_smi_snapshot(label: str) -> dict:
    """nvidia-smi'den total VRAM ve top processes oku."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5
        )
        used, free, total = [int(x.strip()) for x in r.stdout.strip().split(",")]
        print(f"  [nvidia-smi @ {label}] used={used} MB / free={free} MB / total={total} MB")

        # Top processes
        r2 = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5
        )
        for line in r2.stdout.strip().split("\n"):
            if line.strip():
                print(f"    proc: {line.strip()}")

        return {"used_mb": used, "free_mb": free, "total_mb": total}
    except Exception as e:
        print(f"  [nvidia-smi error] {e}")
        return {}


def torch_mem_snapshot(label: str):
    """PyTorch tarafından görünen memory."""
    import torch
    if not torch.cuda.is_available():
        return
    free_b, total_b = torch.cuda.mem_get_info()
    alloc_mb = torch.cuda.memory_allocated() / 1024 / 1024
    reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
    print(f"  [torch @ {label}]")
    print(f"    driver free   = {free_b / 1e9:.2f} GB")
    print(f"    driver total  = {total_b / 1e9:.2f} GB")
    print(f"    pytorch alloc = {alloc_mb:.1f} MB")
    print(f"    pytorch reser = {reserved_mb:.1f} MB")


def divider(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def step_baseline():
    divider("STEP 0 — BASELINE (Python yüklü, CUDA henüz init değil)")
    nvidia_smi_snapshot("baseline")
    print("\n  → Bu satır, Python process'i import torch dışında bir şey yapmadan")
    print("    sistemde hangi process'lerin GPU kullandığını gösterir.")
    print("    DWM ne kadar? Tauri ne kadar? Browser? — buradan görünür.")


def step_cuda_init():
    divider("STEP 1 — CUDA CONTEXT INIT (PyTorch tek tensor allocate)")
    import torch
    print("  → Tiny tensor (~16 byte) allocate ediyoruz, CUDA context açılsın")
    t0 = time.perf_counter()
    x = torch.zeros(4, device="cuda")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  CUDA context init: {elapsed * 1000:.0f} ms")
    nvidia_smi_snapshot("after_context_init")
    torch_mem_snapshot("after_context_init")
    print("\n  → CUDA context tipik 400-1000 MB consume eder (driver caches, kernels, vs.).")
    print("    Eğer baseline'a göre artış 1+ GB ise context overhead büyük.")
    return x


def step_naive_4gib_alloc():
    divider("STEP 2 — SAF 4.03 GiB SINGLE ALLOCATION (CoTracker'dan bağımsız)")
    import torch
    target_bytes = int(4.03 * 1024 ** 3)
    n_floats = target_bytes // 4
    print(f"  → torch.empty({n_floats}, dtype=float32) — tam 4.03 GiB tek allocation")
    print(f"    (CoTracker'ın yapmaya çalıştığı allocation ile aynı boyut)")
    nvidia_smi_snapshot("before_4gib")
    torch_mem_snapshot("before_4gib")
    try:
        t0 = time.perf_counter()
        x = torch.empty(n_floats, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"\n  ✓ 4.03 GiB BAŞARILI ({elapsed * 1000:.0f} ms)")
        print(f"  → Bu CoTracker fail'inin DOĞRUDAN allocation cap problemi olmadığını gösterir.")
        print(f"    Problem CoTracker'ın forward pass'inde transient allocations'da olmalı.")
        nvidia_smi_snapshot("after_4gib")
        torch_mem_snapshot("after_4gib")
        del x
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n  ✗ 4.03 GiB FAIL: {str(e)[:200]}")
        print(f"  → Bu WDDM single-allocation cap (Hipotez 1) DOĞRULANIR.")
        print(f"    8 GB consumer NVIDIA Windows'ta 4 GiB tek allocation reddedildi.")
        print(f"    Çözümler: smaller allocations (chunking) veya driver/Windows tuning.")
        nvidia_smi_snapshot("after_4gib_fail")
        torch_mem_snapshot("after_4gib_fail")
        return False


def step_ramp_alloc():
    divider("STEP 3 — RAMP — 1, 2, 3, 4, 5 GiB tek allocation testi")
    import torch
    sizes_gib = [1.0, 2.0, 3.0, 3.5, 3.9, 4.03, 4.5, 5.0]
    last_ok = 0.0
    for gib in sizes_gib:
        n = int(gib * 1024 ** 3 // 4)
        try:
            x = torch.empty(n, dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            print(f"  ✓ {gib:.2f} GiB OK")
            last_ok = gib
            del x
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  ✗ {gib:.2f} GiB FAIL")
            print(f"\n  → Single-allocation cap ~{last_ok:.2f} GiB civarında.")
            print(f"    8 GB karta normal beklenti 6-7 GB tek allocation OK olmasıydı.")
            return last_ok
    print(f"\n  → Cap >= {sizes_gib[-1]:.2f} GiB. CoTracker problemi başka yerde.")
    return sizes_gib[-1]


def step_cotracker_load():
    divider("STEP 4 — COTRACKER MODEL LOAD (henüz forward yok)")
    import torch
    nvidia_smi_snapshot("before_cotracker_load")
    torch_mem_snapshot("before_cotracker_load")
    print("  → torch.hub.load('facebookresearch/co-tracker', 'cotracker2')")
    t0 = time.perf_counter()
    try:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
        model = model.to("cuda").eval()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"  ✓ Model load OK ({elapsed:.1f}s)")
        nvidia_smi_snapshot("after_cotracker_load")
        torch_mem_snapshot("after_cotracker_load")

        # Internal resolution gizli mi?
        for attr in ("interp_shape", "model_resolution", "input_resolution"):
            val = getattr(model, attr, None)
            if val is not None:
                print(f"    model.{attr} = {val}")

        return model
    except Exception as e:
        print(f"  ✗ Load fail: {e}")
        return None


def step_cotracker_forward(model, frames_dir: Path, max_frames: int = 50, edge: int = 480):
    divider(f"STEP 5 — COTRACKER FORWARD ({max_frames} frame, edge={edge})")
    import torch
    import cv2
    import numpy as np

    files = sorted(frames_dir.glob("frame_*.png"))[:max_frames]
    if not files:
        print(f"  ✗ {frames_dir} altında frame_*.png yok")
        return

    print(f"  → {len(files)} frame yüklüyor (CPU)...")
    arrs = []
    for fp in files:
        img = cv2.imread(str(fp))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if max(h, w) > edge:
            s = edge / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
        arrs.append(img)
    video = np.stack(arrs)
    tensor_cpu = torch.from_numpy(video).permute(0, 3, 1, 2).float().unsqueeze(0)
    print(f"    CPU tensor shape: {tuple(tensor_cpu.shape)}, "
          f"size: {tensor_cpu.numel() * 4 / 1e6:.1f} MB")

    nvidia_smi_snapshot("before_video_to_gpu")
    torch_mem_snapshot("before_video_to_gpu")
    print(f"  → Video → GPU")
    try:
        video_gpu = tensor_cpu.to("cuda")
        torch.cuda.synchronize()
        print(f"  ✓ Video upload OK")
        nvidia_smi_snapshot("after_video_to_gpu")
        torch_mem_snapshot("after_video_to_gpu")
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ✗ Video upload OOM: {str(e)[:120]}")
        return

    print(f"  → CoTracker forward (grid_size=15)")
    try:
        with torch.no_grad():
            tracks, vis = model(video_gpu, grid_size=15)
        torch.cuda.synchronize()
        print(f"  ✓ Forward OK — tracks shape: {tuple(tracks.shape)}")
        nvidia_smi_snapshot("after_forward")
        torch_mem_snapshot("after_forward")
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n  ✗ FORWARD OOM (BURADA HEP TAKILIYORDUK):")
        print(f"    {str(e)[:300]}")
        print(f"\n  → memory_summary at fail point:")
        print(torch.cuda.memory_summary(abbreviated=False))
    except Exception as e:
        print(f"  ✗ Forward fail (non-OOM): {e}")


def step_online_variant_load():
    divider("STEP 6 — COTRACKER2_ONLINE (sliding window, fallback adayı)")
    import torch
    nvidia_smi_snapshot("before_online_load")
    try:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
        model = model.to("cuda").eval()
        torch.cuda.synchronize()
        print(f"  ✓ cotracker2_online load OK")
        nvidia_smi_snapshot("after_online_load")
        torch_mem_snapshot("after_online_load")
        print(f"  → Online variant memory profilini sliding window üzerinden yapar.")
        print(f"    Eğer offline (cotracker2) OOM olup online OK ise bu doğrulanır.")
        return model
    except Exception as e:
        print(f"  ✗ Online load fail: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="CoTracker memory diagnostic")
    parser.add_argument("--frames-dir", type=str,
                        default=r"C:\Users\TAHA\Desktop\gaussian-splatter\Gaussian Splatter\4dgs-studio\data\banana_demo\frames",
                        help="Frame klasörü (banana_demo varsayılan)")
    parser.add_argument("--max-frames", type=int, default=50,
                        help="Forward testinde kaç frame")
    parser.add_argument("--edge", type=int, default=480,
                        help="Forward testinde max edge")
    parser.add_argument("--skip-real-test", action="store_true",
                        help="Sadece allocation testleri, CoTracker forward'ı atla")
    parser.add_argument("--skip-online", action="store_true",
                        help="Online variant testini atla")
    args = parser.parse_args()

    print(f"\n  CoTracker Memory Diagnostic")
    print(f"  PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}\n")

    # Step 0
    step_baseline()

    # Step 1
    import torch
    if not torch.cuda.is_available():
        print("CUDA yok — test anlamsız")
        sys.exit(1)
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Driver capability: cap = {torch.cuda.get_device_capability(0)}")
    _ctx_tensor = step_cuda_init()

    # Step 2 — saf 4.03 GiB
    cap_4gib_ok = step_naive_4gib_alloc()

    # Step 3 — ramp
    last_ok = step_ramp_alloc()

    # Step 4 — model load
    model = step_cotracker_load()

    # Step 5 — gerçek forward
    if model is not None and not args.skip_real_test:
        step_cotracker_forward(model, Path(args.frames_dir),
                               max_frames=args.max_frames, edge=args.edge)
        # Free model before next variant
        try:
            model.cpu()
            del model
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    # Step 6 — online variant
    if not args.skip_online:
        step_online_variant_load()

    divider("ÖZET")
    print(f"  Single-alloc max: {last_ok:.2f} GiB")
    print(f"  4.03 GiB tek seferlik alloc: {'OK' if cap_4gib_ok else 'FAIL'}")
    if not cap_4gib_ok:
        print(f"  → WDDM cap ya da fragmentation. Tauri / browser kapatıp tekrar dene.")
    elif last_ok >= 5.0:
        print(f"  → Single-alloc cap problemi yok. CoTracker fail'i forward'da transient")
        print(f"    allocations'dan. cotracker2_online veya temporal chunking şart.")
    print()


if __name__ == "__main__":
    main()

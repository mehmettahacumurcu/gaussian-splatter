"""Helpers for the Colab verification notebooks (colab/*.ipynb).

Standalone — stdlib only, plus an optional torch import for GPU memory stats.
Run from the repo root, after the pipeline has produced data/<scene>/output/.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def _eval_path(scene: str) -> Path:
    for c in (
        Path("data") / scene / "output" / "eval" / "nvs_eval.json",
        Path("data") / scene / "eval" / "nvs_eval.json",
    ):
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No nvs_eval.json for '{scene}'. Did you run the pipeline with --nvs-eval?"
    )


def read_metrics(scene: str) -> dict:
    """Return {'kind','psnr','ssim','lpips','n_frames'} from the NVS eval report."""
    report = json.loads(_eval_path(scene).read_text(encoding="utf-8"))
    if report.get("held_out_metrics"):
        m, kind = report["held_out_metrics"], "multi-view held-out"
    elif report.get("temporal_holdout"):
        m, kind = report["temporal_holdout"], "single-view holdout"
    else:
        raise ValueError(f"eval report has no held-out / temporal metrics: {report}")
    return {
        "kind": kind,
        "psnr": float(m["psnr"]),
        "ssim": float(m["ssim"]),
        "lpips": float(m["lpips"]),
        "n_frames": m.get("n_frames"),
    }


def count_ply_frames(scene: str) -> int:
    ply_dir = Path("data") / scene / "output" / "ply"
    return len(sorted(ply_dir.glob("frame_*.ply"))) if ply_dir.exists() else 0


def gpu_mem_summary() -> None:
    try:
        import torch  # optional
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(
                f"  GPU: {torch.cuda.get_device_name(0)} | "
                f"peak alloc {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | "
                f"free {free / 1e9:.2f}/{total / 1e9:.2f} GB"
            )
        else:
            print("  (no CUDA device visible)")
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"  (gpu mem unavailable: {e})")


def check_phase2(scene: str = "myroom", baseline_psnr: float = 29.0,
                 min_psnr: float = 27.0) -> bool:
    """Phase 2 no-regression gate. Returns True only if every check passes.

    Confirms the fourier_K=0 change did not drop quality (held-out PSNR stays in
    band) and that the static-export fix writes exactly one clean ply frame.
    """
    print(f"=== Phase 2 verification - {scene} ===")
    ok = True
    try:
        m = read_metrics(scene)
        print(f"  held-out PSNR : {m['psnr']:.2f} dB  "
              f"(local baseline ~{baseline_psnr:.1f} dB, floor {min_psnr:.1f})")
        print(f"  SSIM / LPIPS  : {m['ssim']:.4f} / {m['lpips']:.4f}  "
              f"[{m['kind']}, n={m['n_frames']}]")
        if m["psnr"] < min_psnr:
            print("  [FAIL] PSNR below floor - possible regression from fourier_K=0")
            ok = False
        else:
            print("  [OK] PSNR within no-regression band")
    except Exception as e:
        print(f"  [FAIL] could not read eval metrics: {e}")
        ok = False

    n = count_ply_frames(scene)
    if n == 1:
        print("  [OK] static export wrote exactly 1 ply frame (P2-2 fix)")
    else:
        print(f"  [FAIL] expected 1 ply frame, found {n} (P2-2 static export)")
        ok = False

    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def copy_results_to_drive(scene: str, drive_dir: str) -> Path:
    """Copy output/{eval,ply,logs} + any orbit.mp4 to a Drive folder. Returns dst."""
    src = Path("data") / scene / "output"
    dst = Path(drive_dir) / f"{scene}_results"
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ("eval", "ply", "logs"):
        s = src / sub
        if s.exists():
            shutil.copytree(s, dst / sub, dirs_exist_ok=True)
    for mp4 in src.rglob("orbit.mp4"):
        shutil.copy2(mp4, dst / mp4.name)
    print(f"  results -> {dst}")
    return dst

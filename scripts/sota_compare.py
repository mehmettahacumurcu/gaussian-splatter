"""SOTA comparison — read NVS eval report and compare against published baselines.

Usage:
    python scripts/sota_compare.py <scene>
    python scripts/sota_compare.py flame_steak
    python scripts/sota_compare.py flame_steak --eval-path data/flame_steak/output/eval/nvs_eval.json

Reads the JSON report written by `backend/eval/nvs_eval.save_eval_report` after
training (when `nvs_eval=true` was set on the job), looks up the matching
baseline from the BASELINES table below, and prints a verdict:

    ✅ SOTA-tier        ΔPSNR ≥ -1.0 dB   → algorithm sound, cloud compute will help
    ⚠ Below SOTA       -3.0 ≤ ΔPSNR < -1.0 → tunable + more compute likely
    ✗ Algorithmic gap  ΔPSNR < -3.0       → more compute won't help, fix algo first

The baselines mirror docs/SOTA_BASELINES.md. They are the *strongest* PSNR
across the methods listed there — i.e. the hardest target band.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Optional

# Windows consoles default to cp1254 / cp1252 and choke on the verdict glyphs.
# Force UTF-8 for stdout/stderr so ✓ ⚠ ✗ render reliably.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ---- Baselines: best published PSNR per scene -----------------------------
# Source: docs/SOTA_BASELINES.md (which method gives the best is noted).
# Keep in sync if you edit the doc.

BASELINES_4D: dict[str, dict[str, float]] = {
    # N3V Plenoptic Video — held-out cam00, mean over all timesteps.
    "coffee_martini":   {"psnr": 28.74, "ssim": 0.913, "lpips": 0.142, "best_method": "K-Planes"},
    "cook_spinach":     {"psnr": 33.18, "ssim": 0.946, "lpips": 0.099, "best_method": "Spacetime"},
    "cut_roasted_beef": {"psnr": 33.52, "ssim": 0.948, "lpips": 0.090, "best_method": "Spacetime"},
    "flame_salmon_1":   {"psnr": 30.44, "ssim": 0.929, "lpips": 0.123, "best_method": "K-Planes"},
    "flame_steak":      {"psnr": 33.51, "ssim": 0.953, "lpips": 0.083, "best_method": "Spacetime"},
    "sear_steak":       {"psnr": 33.89, "ssim": 0.957, "lpips": 0.075, "best_method": "Spacetime"},
}

BASELINES_STATIC: dict[str, dict[str, float]] = {
    # Mip-NeRF 360 — every-8th-frame held-out.
    "bicycle":  {"psnr": 25.50, "best_method": "Mip-Splatting"},
    "flowers":  {"psnr": 21.61, "best_method": "Mip-Splatting"},
    "garden":   {"psnr": 27.41, "best_method": "3DGS"},
    "stump":    {"psnr": 26.59, "best_method": "Mip-Splatting"},
    "treehill": {"psnr": 23.10, "best_method": "Scaffold-GS"},
    "room":     {"psnr": 32.13, "best_method": "Scaffold-GS"},
    "counter":  {"psnr": 29.34, "best_method": "Scaffold-GS"},
    "kitchen":  {"psnr": 31.52, "best_method": "Mip-Splatting"},
    "bonsai":   {"psnr": 32.70, "best_method": "Scaffold-GS"},
    # Tanks & Temples
    "Truck":    {"psnr": 25.40, "best_method": "Mip-Splatting"},
    "Train":    {"psnr": 22.13, "best_method": "Mip-Splatting"},
}

ALL_BASELINES: dict[str, dict[str, float]] = {**BASELINES_4D, **BASELINES_STATIC}


# ---- Verdict thresholds ---------------------------------------------------

VERDICT_SOTA_DELTA   = -1.0  # ≥ this → SOTA-tier
VERDICT_GAP_DELTA    = -3.0  # ≥ this (and < SOTA) → below SOTA, tunable
                             # < this → algorithmic gap


def find_eval_report(scene: str, explicit_path: Optional[Path]) -> Optional[Path]:
    """Locate the NVS eval JSON.

    Default lookup order (relative to project root):
        data/<scene>/output/eval/nvs_eval.json
        data/<scene>/eval/nvs_eval.json
    """
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None

    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "data" / scene / "output" / "eval" / "nvs_eval.json",
        project_root / "data" / scene / "eval" / "nvs_eval.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def extract_metrics(report: dict) -> Optional[dict]:
    """Pull the held-out (multi-view) or temporal-holdout (single-view) metrics."""
    if "held_out_metrics" in report and report["held_out_metrics"]:
        return {
            "kind": "multi-view held-out",
            "cam":  report.get("held_out_cam", "?"),
            "psnr":  report["held_out_metrics"]["psnr"],
            "ssim":  report["held_out_metrics"]["ssim"],
            "lpips": report["held_out_metrics"]["lpips"],
            "n_frames": report["held_out_metrics"].get("n_frames"),
        }
    if "temporal_holdout" in report and report["temporal_holdout"]:
        return {
            "kind": "temporal holdout (single-view)",
            "cam":  None,
            "psnr":  report["temporal_holdout"]["psnr"],
            "ssim":  report["temporal_holdout"]["ssim"],
            "lpips": report["temporal_holdout"]["lpips"],
            "n_frames": report["temporal_holdout"].get("n_frames"),
        }
    return None


def verdict_for_delta(delta: float) -> tuple[str, str]:
    """Return (label, color hint) for a PSNR delta vs the strongest baseline."""
    if delta >= VERDICT_SOTA_DELTA:
        return "SOTA-tier", "ok"
    if delta >= VERDICT_GAP_DELTA:
        return "Below SOTA", "warn"
    return "Algorithmic gap", "err"


def color(text: str, kind: str) -> str:
    """ANSI-color a string. kind in {ok, warn, err, dim, bold}."""
    if not sys.stdout.isatty():
        return text
    codes = {"ok": "\033[92m", "warn": "\033[93m", "err": "\033[91m",
             "dim": "\033[2m", "bold": "\033[1m"}
    return f"{codes.get(kind, '')}{text}\033[0m"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare an NVS eval report to published SOTA baselines.")
    parser.add_argument("scene", nargs="?", help="Scene name (e.g. flame_steak, bicycle). Required unless --list.")
    parser.add_argument("--eval-path", type=Path, default=None,
                        help="Override path to nvs_eval.json (default: data/<scene>/output/eval/nvs_eval.json).")
    parser.add_argument("--list", action="store_true",
                        help="List known baselines and exit.")
    args = parser.parse_args()

    if args.list:
        print("4D Dynamic (N3V Plenoptic Video):")
        for name, b in BASELINES_4D.items():
            print(f"  {name:20s} PSNR={b['psnr']:.2f} dB  ({b['best_method']})")
        print("\nStatic 3D (Mip-NeRF 360 + Tanks&Temples):")
        for name, b in BASELINES_STATIC.items():
            print(f"  {name:20s} PSNR={b['psnr']:.2f} dB  ({b['best_method']})")
        return 0

    if args.scene is None:
        parser.error("scene is required (or use --list to see known scenes)")

    scene = args.scene
    baseline = ALL_BASELINES.get(scene)

    if baseline is None:
        print(color(f"⚠ No published baseline for scene '{scene}'.", "warn"))
        print("  Known scenes (use --list for full table):")
        print(f"    {', '.join(sorted(ALL_BASELINES.keys()))}")
        return 2

    eval_path = find_eval_report(scene, args.eval_path)
    if eval_path is None:
        print(color(f"✗ No NVS eval report found for '{scene}'.", "err"))
        print("  Looked in:")
        print(f"    data/{scene}/output/eval/nvs_eval.json")
        print(f"    data/{scene}/eval/nvs_eval.json")
        print("\n  To produce one: submit a job for this scene with NVS Evaluation enabled,")
        print("  or override with --eval-path PATH.")
        return 1

    try:
        report = json.loads(eval_path.read_text())
    except Exception as e:
        print(color(f"✗ Could not parse {eval_path}: {e}", "err"))
        return 1

    metrics = extract_metrics(report)
    if metrics is None:
        print(color(f"✗ Eval report has no held-out / temporal-holdout metrics: {eval_path}", "err"))
        return 1

    delta_psnr = metrics["psnr"] - baseline["psnr"]
    label, kind = verdict_for_delta(delta_psnr)

    # ---- Output ----
    print(color(f"SOTA verification — {scene}", "bold"))
    print(color(f"  eval source: {eval_path}", "dim"))
    print()
    print(f"  Eval kind     : {metrics['kind']}"
          + (f" (cam={metrics['cam']})" if metrics["cam"] else "")
          + (f" · {metrics['n_frames']} frames" if metrics["n_frames"] else ""))
    print(f"  Our PSNR      : {metrics['psnr']:.2f} dB")
    print(f"  Our SSIM      : {metrics['ssim']:.4f}")
    print(f"  Our LPIPS-VGG : {metrics['lpips']:.4f}  "
          + color("(papers typically report LPIPS-Alex; ours runs ~0.01-0.03 higher)", "dim"))
    print()
    print(f"  Best baseline : {baseline['psnr']:.2f} dB  ({baseline['best_method']})")
    sign = "+" if delta_psnr >= 0 else ""
    print(f"  ΔPSNR         : {color(sign + f'{delta_psnr:.2f} dB', kind)}")
    print()
    print(f"  Verdict       : {color(label, kind)}")

    # Action hint
    print()
    if kind == "ok":
        print("  → Algorithm is in the SOTA band. Compute (longer training, higher")
        print("    resolution, larger N cap) is a reasonable next lever. Cloud will help.")
    elif kind == "warn":
        print("  → Within the 'tunable' band. If this was a product preset (premium etc.),")
        print("    part of the gap is the objective, not the pipeline: LPIPS/depth losses and")
        print("    the N cap trade PSNR away. Re-run the PSNR-parity preset first:")
        print("      python scripts/static_3dgs.py --scene <scene> --preset sota --nvs-eval --native-res")
        print("    (no --foundation). If STILL below -1 dB, investigate density control + COLMAP.")
    else:
        print("  → Gap is too large to be compute-bound. Likely culprits:")
        print("      • mismatched eval protocol (held-out cam, frame count)")
        print("      • preprocessing (COLMAP failure, wrong calibration)")
        print("      • foundation-model regression (Metric3D depth scale, CoTracker quality)")
        print("      • a missing algorithmic component vs the reference paper")
        print("    Investigate before paying for cloud GPUs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

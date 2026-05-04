"""Re-run NVS eval against an existing checkpoint, with the K-scaling fix.

Diagnoses the eval-K-scaling bug that was producing PSNR ~8 dB regardless of
model quality. Loads ckpt_final.pt, parses COLMAP, and runs eval_temporal_holdout
under four (optionally fewer) configurations so you can see the fix in isolation:

    1) old K (broken)  + tail-10pct holdout   — should reproduce the 8.x dB
    2) old K (broken)  + interleaved every-8  — still broken because K is wrong
    3) fixed K         + tail-10pct holdout   — should jump to ~training PSNR
    4) fixed K         + interleaved every-8  — should also be ~training PSNR

Usage:
    python scripts/reeval_nvs.py --scene train

Notes:
  * Trainer currently samples ALL frame indices uniformly (trainer.py:1318), so
    none of the "holdouts" were excluded during training. Configs (3) and (4)
    therefore measure training-view fit, not true novel-view PSNR — a sanity
    test, not a quality claim. True NVS requires teaching the trainer to skip
    holdout indices (separate fix).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import scene_paths  # noqa: E402
from backend.preprocess.parse_colmap import parse_cameras  # noqa: E402
from backend.model.gaussian_model import GaussianModel  # noqa: E402
from backend.model.deformation import DeformationField  # noqa: E402
from backend.eval.nvs_eval import eval_temporal_holdout  # noqa: E402


def _scale_K(K_native: torch.Tensor, w_native: int, h_native: int,
             w_render: int, h_render: int) -> torch.Tensor:
    sx, sy = w_render / w_native, h_render / h_native
    K = K_native.clone()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def _infer_deform_config(state_dict: dict) -> dict:
    """Mirror of scripts/export_from_ckpt.py — infer deform shape from ckpt keys."""
    keys = list(state_dict.keys())
    multires_keys = [k for k in keys if k.startswith("hexplane.planes_list.")]
    if multires_keys:
        scale_indices = sorted({int(k.split(".")[2]) for k in multires_keys})
        resolutions, feat_dims = [], []
        for s in scale_indices:
            sample_key = next(
                k for k in multires_keys
                if k.startswith(f"hexplane.planes_list.{s}.planes.0")
            )
            shape = state_dict[sample_key].shape
            feat_dims.append(int(shape[1]))
            resolutions.append(int(shape[2]))
        cfg = dict(
            multires_resolutions=resolutions,
            multires_feat_dim=feat_dims[0],
        )
    else:
        sr_keys = [k for k in keys if k.startswith("hexplane.planes.")]
        if not sr_keys:
            cfg = dict(resolution=96, feat_dim=48)
        else:
            shape = state_dict[sr_keys[0]].shape
            cfg = dict(resolution=int(shape[2]), feat_dim=int(shape[1]))

    if "mlp.0.weight" in state_dict:
        cfg["mlp_width"] = int(state_dict["mlp.0.weight"].shape[0])
    else:
        cfg["mlp_width"] = 256
    linear_indices = sorted({
        int(k.split(".")[1]) for k in keys
        if k.startswith("mlp.") and k.endswith(".weight")
    })
    cfg["mlp_depth"] = max(1, len(linear_indices) - 1)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-run NVS eval on an existing checkpoint")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="Checkpoint path (default: ckpt_final.pt or latest)")
    ap.add_argument("--render-w", type=int, default=1280)
    ap.add_argument("--render-h", type=int, default=720)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-old", action="store_true",
                    help="Skip the broken-K configs (saves ~half the eval time)")
    ap.add_argument("--skip-lpips", action="store_true",
                    help="Skip LPIPS (much faster)")
    args = ap.parse_args()

    paths = scene_paths(args.scene)
    if not paths["base"].exists():
        print(f"Scene not found: {paths['base']}")
        return 1

    # ---- Resolve checkpoint ----
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_dir = paths["output"] / "ckpt"
        if (ckpt_dir / "ckpt_final.pt").exists():
            ckpt_path = ckpt_dir / "ckpt_final.pt"
        else:
            ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
            if not ckpts:
                print(f"No checkpoints in {ckpt_dir}")
                return 1
            ckpt_path = ckpts[-1]

    print(f"\n{'='*72}")
    print("  NVS re-eval (K-scaling fix verification)")
    print(f"{'='*72}")
    print(f"  scene:   {args.scene}")
    print(f"  ckpt:    {ckpt_path}")
    print(f"  render:  {args.render_w} x {args.render_h}")
    print(f"  device:  {args.device}")
    print()

    # ---- Parse COLMAP ----
    cams = parse_cameras(paths["colmap"])
    cam_names_sorted = sorted(cams.keys())
    first_cam = cams[cam_names_sorted[0]]
    K_native = torch.from_numpy(first_cam["K"]).float()
    native_w, native_h = int(first_cam["width"]), int(first_cam["height"])

    frame_paths = []
    w2c_list = []
    for name in cam_names_sorted:
        fp = Path(paths["frames"]) / name
        if not fp.exists():
            continue
        frame_paths.append(fp)
        w2c_list.append(torch.from_numpy(cams[name]["w2c"]).float())

    T = len(frame_paths)
    print(f"  COLMAP: {len(cams)} cams, {T} frames matched")
    print(f"  native: {native_w} x {native_h}")
    print(f"  K_native:")
    print(f"    fx={K_native[0,0]:.2f} fy={K_native[1,1]:.2f} "
          f"cx={K_native[0,2]:.2f} cy={K_native[1,2]:.2f}")

    K_fixed = _scale_K(K_native, native_w, native_h, args.render_w, args.render_h)
    print(f"  K_fixed (scaled to {args.render_w}x{args.render_h}):")
    print(f"    fx={K_fixed[0,0]:.2f} fy={K_fixed[1,1]:.2f} "
          f"cx={K_fixed[0,2]:.2f} cy={K_fixed[1,2]:.2f}")
    print()

    # ---- Load checkpoint ----
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    gs = GaussianModel.from_checkpoint(ckpt["gs"]).to(args.device)
    print(f"  GaussianModel: N={gs.num_points:,}, sh_degree={gs.sh_degree}")

    deform = None
    if "deform" in ckpt:
        try:
            dcfg = _infer_deform_config(ckpt["deform"])
            deform = DeformationField(**dcfg).to(args.device)
            deform.load_state_dict(ckpt["deform"])
            print(f"  DeformationField: loaded ({dcfg})")
        except Exception as e:
            print(f"  DeformationField: load failed — {e}; using static (deform=None)")
            deform = None

    scene_extent = float(ckpt.get("scene_extent", 1.0))
    print(f"  scene_extent: {scene_extent:.3f}")
    print()

    # ---- Run eval configs ----
    holdout_tail = list(range(max(0, T - max(1, T // 10)), T))
    holdout_inter = list(range(7, T, 8))
    print(f"  holdout_tail   = last {len(holdout_tail)} frames "
          f"(idx {holdout_tail[0]}..{holdout_tail[-1]})")
    print(f"  holdout_inter  = every 8th, {len(holdout_inter)} frames "
          f"(idx {holdout_inter[0]}..{holdout_inter[-1]})")
    print()

    configs = []
    if not args.skip_old:
        configs.append(("OLD K (broken) + tail-10pct       ", K_native, holdout_tail))
        configs.append(("OLD K (broken) + interleaved-1/8  ", K_native, holdout_inter))
    configs.append(("FIXED K        + tail-10pct       ", K_fixed, holdout_tail))
    configs.append(("FIXED K        + interleaved-1/8  ", K_fixed, holdout_inter))

    print(f"  {'config':<38}  {'PSNR':>7}  {'SSIM':>6}  {'LPIPS':>6}  {'N':>4}")
    print("  " + "-" * 70)
    for label, K, idx_list in configs:
        m = eval_temporal_holdout(
            gs=gs, deform=deform,
            cam_K=K, cam_w2c_per_frame=w2c_list,
            frame_paths=frame_paths, holdout_indices=idx_list,
            width=args.render_w, height=args.render_h,
            static_mode=True,
            scene_extent=scene_extent, device=args.device,
            skip_lpips=args.skip_lpips,
        )
        psnr = m["psnr_mean"]
        ssim = m["ssim_mean"]
        lpips = m["lpips_mean"]
        n = m["n_frames"]
        lpips_s = f"{lpips:.4f}" if not (lpips != lpips) else "  n/a"  # NaN check
        print(f"  {label}  {psnr:>7.2f}  {ssim:>6.4f}  {lpips_s:>6}  {n:>4}")

    print()
    print("  Reading the table:")
    print("    OLD K rows  : reproduce the original ~8 dB (K mismatched to render res)")
    print("    FIXED K rows: should land near training PSNR (model is healthy)")
    print("    Big gap between OLD and FIXED rows = K-scaling bug confirmed and fixed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

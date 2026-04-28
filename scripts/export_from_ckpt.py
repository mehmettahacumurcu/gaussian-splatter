"""Standalone PLY export from a saved checkpoint.

Training run kesildiyse veya bitmeden frontend'de gormek istersen, en son
yazilmis ckpt'tan PLY frame'leri uretir. Ckpt state_dict'inden config'i
otomatik infer eder (Phase 2.5 multi-res HexPlane dahil).

Usage:
    # Otomatik en son ckpt'i bul + 30 frame export
    python scripts/export_from_ckpt.py --scene flame_steak

    # Spesifik ckpt + 60 frame
    python scripts/export_from_ckpt.py --scene flame_steak \\
        --ckpt data/flame_steak/output/ckpt/ckpt_003000.pt --num-timestamps 60

    # Frontend hizli on-izleme icin az frame
    python scripts/export_from_ckpt.py --scene flame_steak --num-timestamps 15
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import scene_paths  # noqa: E402
from backend.model.gaussian_model import GaussianModel  # noqa: E402
from backend.model.deformation import DeformationField  # noqa: E402
from backend.export.to_splat import export_to_ply  # noqa: E402


def _infer_deform_config(state_dict: dict) -> dict:
    """state_dict key'lerinden DeformationField parametrelerini cikar."""
    keys = list(state_dict.keys())

    # Phase 2.5 — multi-res: keys gibi 'hexplane.planes_list.0.planes.0'
    multires_keys = [k for k in keys if k.startswith("hexplane.planes_list.")]
    is_multires = len(multires_keys) > 0

    if is_multires:
        # Kac scale var
        scale_indices = sorted({
            int(k.split(".")[2]) for k in multires_keys
        })
        # Her scale'in resolution + feat_dim
        resolutions = []
        feat_dims = []
        for s in scale_indices:
            sample_key = next(
                k for k in multires_keys
                if k.startswith(f"hexplane.planes_list.{s}.planes.0")
            )
            shape = state_dict[sample_key].shape  # (1, feat_dim, R, R)
            feat_dims.append(int(shape[1]))
            resolutions.append(int(shape[2]))
        # feat_dim ortak olmali; degilse warning
        if len(set(feat_dims)) > 1:
            print(f"⚠ Multi-res feat_dim'ler farkli: {feat_dims}, ilki kullanilir")
        feat_dim = feat_dims[0]
        cfg = dict(
            multires_resolutions=resolutions,
            multires_feat_dim=feat_dim,
        )
    else:
        # Single-res HexPlane — keys 'hexplane.planes.0' formatinda
        sr_keys = [k for k in keys if k.startswith("hexplane.planes.")]
        if not sr_keys:
            print("⚠ HexPlane key bulunamadi, default 96/48 kullanilir")
            cfg = dict(resolution=96, feat_dim=48)
        else:
            sample_key = sr_keys[0]
            shape = state_dict[sample_key].shape
            cfg = dict(resolution=int(shape[2]), feat_dim=int(shape[1]))

    # MLP genisligi: ilk Linear weight
    mlp0_key = "mlp.0.weight"
    if mlp0_key in state_dict:
        cfg["mlp_width"] = int(state_dict[mlp0_key].shape[0])
    else:
        cfg["mlp_width"] = 256

    # MLP derinligi: kac Linear var (her Linear weight + bias = 2 key,
    # SiLU non-param. Son Linear output 10. So count Linear / 2 - 1 hidden).
    linear_indices = sorted({
        int(k.split(".")[1]) for k in keys
        if k.startswith("mlp.") and k.endswith(".weight")
    })
    # Her hidden Linear+SiLU 2 layer alir, son Linear (output) 1 layer.
    # mlp_depth = hidden Linear sayisi
    cfg["mlp_depth"] = max(1, len(linear_indices) - 1)

    return cfg


def main():
    ap = argparse.ArgumentParser(description="Ckpt -> PLY export")
    ap.add_argument("--scene", required=True, help="Scene adi")
    ap.add_argument("--ckpt", default=None,
                    help="Ckpt path (default: en son ckpt_*.pt)")
    ap.add_argument("--num-timestamps", type=int, default=30,
                    help="Kac PLY frame uretsin (default 30)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    paths = scene_paths(args.scene)
    if not paths["base"].exists():
        print(f"✗ Scene yok: {paths['base']}")
        sys.exit(1)

    # Ckpt bul
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_dir = paths["base"] / "output" / "ckpt"
        if not ckpt_dir.exists():
            print(f"✗ Ckpt klasoru yok: {ckpt_dir}")
            sys.exit(1)

        # En son ckpt: 'ckpt_final.pt' varsa onu, yoksa en yuksek iter
        if (ckpt_dir / "ckpt_final.pt").exists():
            ckpt_path = ckpt_dir / "ckpt_final.pt"
        else:
            ckpts = list(ckpt_dir.glob("ckpt_*.pt"))
            if not ckpts:
                print(f"✗ Hic ckpt yok: {ckpt_dir}")
                sys.exit(1)
            # iter numarasi sirala
            def _iter_num(p: Path) -> int:
                stem = p.stem  # ckpt_003000
                num = stem.replace("ckpt_", "").replace("final", "9999999")
                try:
                    return int(num)
                except ValueError:
                    return 0
            ckpt_path = sorted(ckpts, key=_iter_num)[-1]

    print(f"\n{'='*70}")
    print(f"  Ckpt -> PLY Export")
    print(f"{'='*70}")
    print(f"  Scene:     {args.scene}")
    print(f"  Ckpt:      {ckpt_path}")
    print(f"  Frames:    {args.num_timestamps}")
    print(f"  Device:    {args.device}")
    print()

    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)

    # Gauss model yukle
    gs = GaussianModel.from_checkpoint(ckpt["gs"])
    print(f"  GaussianModel: N={gs.num_points:,}, sh_degree={gs.sh_degree}, "
          f"fourier_K={gs.fourier_K}")

    # Deform model yukle (ckpt state'inden config infer)
    deform = None
    if "deform" in ckpt:
        deform_cfg = _infer_deform_config(ckpt["deform"])
        print(f"  Deform config (inferred): {deform_cfg}")
        try:
            deform = DeformationField(**deform_cfg).to(args.device)
            deform.load_state_dict(ckpt["deform"])
            print(f"  ✓ Deformation field yuklendi")
        except Exception as e:
            print(f"  ⚠ Deform yukleme hatasi: {e}")
            print(f"    Static export (deformation=None) ile devam edilir")
            deform = None
    else:
        print(f"  ⚠ Ckpt'te deform key yok — static-only export")

    scene_extent = float(ckpt.get("scene_extent", 1.0))
    print(f"  Scene extent: {scene_extent:.3f}")

    output_dir = paths["base"] / "output" / "ply"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Exporting to: {output_dir}")
    plys = export_to_ply(
        gs=gs.to(args.device), deform=deform, output_dir=output_dir,
        num_timestamps=args.num_timestamps,
        scene_extent=scene_extent, device=args.device,
    )
    print(f"\n  ✓ {len(plys)} PLY uretildi -> {output_dir}")
    print(f"  Frontend'de '{args.scene}' yazarak yuklenebilir.")


if __name__ == "__main__":
    main()

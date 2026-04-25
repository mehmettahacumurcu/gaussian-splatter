"""Faz 6 — Eğitilmiş 4DGS modelini her timestamp için .ply olarak export et.

Çıktı formatı, INRIA 3DGS .ply formatına uyumlu:
  vertex props:
    x, y, z, nx, ny, nz,
    f_dc_0..f_dc_2, f_rest_0..f_rest_(K*3-1),
    opacity, scale_0..scale_2, rot_0..rot_3
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List

from ..model.gaussian_model import GaussianModel
from ..model.deformation import DeformationField, decode_fourier_trajectory


def write_ply(
    means: np.ndarray,        # (N, 3) float
    log_scales: np.ndarray,   # (N, 3) — log-space (3DGS .ply formatı log saklar)
    quats: np.ndarray,        # (N, 4) (w, x, y, z), normalize edilmiş
    opacities_logit: np.ndarray,  # (N,) — logit-space (3DGS formatı logit saklar)
    sh_dc: np.ndarray,        # (N, 3) — DC bileşeni
    sh_rest: np.ndarray,      # (N, K, 3) — yüksek frekans
    output_path: str | Path,
) -> Path:
    """3DGS-uyumlu .ply yaz."""
    from plyfile import PlyData, PlyElement

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N = means.shape[0]
    K = sh_rest.shape[1]
    sh_rest_flat = sh_rest.transpose(0, 2, 1).reshape(N, -1)  # (N, 3*K)

    dtype_full = (
        [(p, "f4") for p in ("x", "y", "z", "nx", "ny", "nz")]
        + [(f"f_dc_{i}", "f4") for i in range(3)]
        + [(f"f_rest_{i}", "f4") for i in range(3 * K)]
        + [("opacity", "f4")]
        + [(f"scale_{i}", "f4") for i in range(3)]
        + [(f"rot_{i}", "f4") for i in range(4)]
    )
    arr = np.empty(N, dtype=dtype_full)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    for i in range(3):
        arr[f"f_dc_{i}"] = sh_dc[:, i]
    for i in range(3 * K):
        arr[f"f_rest_{i}"] = sh_rest_flat[:, i]
    arr["opacity"] = opacities_logit
    for i in range(3):
        arr[f"scale_{i}"] = log_scales[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = quats[:, i]

    PlyData([PlyElement.describe(arr, "vertex")]).write(str(output_path))
    return output_path


@torch.no_grad()
def export_to_ply(
    gs: GaussianModel,
    deform: DeformationField | None,
    output_dir: str | Path,
    num_timestamps: int = 60,
    scene_extent: float = 1.0,
    device: str = "cuda",
) -> List[Path]:
    """
    Her zaman adımı için deformed Gaussian'ları .ply olarak yaz.

    Args:
        gs: Eğitilmiş GaussianModel
        deform: DeformationField — None ise sadece statik tek frame export edilir
        output_dir: Çıktı klasörü
        num_timestamps: Kaç zaman noktası (örn. 60 → 0.0, 1/59, 2/59, ..., 1.0)
        scene_extent: Deformation field'ın means'i normalize ettiği değer

    Returns:
        Yazılan .ply dosyalarının listesi.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gs = gs.to(device).eval()
    if deform is not None:
        deform = deform.to(device).eval()

    timestamps = [0.0] if deform is None else [
        t / max(num_timestamps - 1, 1) for t in range(num_timestamps)
    ]

    written: List[Path] = []
    for t_idx, t in enumerate(timestamps):
        if deform is not None:
            dpos_mlp, dquat, dscale = deform(gs.means, t, scene_extent)
            # v3.5 safety: dscale clamp (trainer'da zaten clamp'lanıyor ama
            # eski checkpoint'lerde uygulanmamış olabilir, export burada da yapsın)
            dscale = dscale.clamp(min=-2.0, max=2.0)
            # v3.6.2: dpos_mlp clamp
            mlp_cap = scene_extent * 0.1
            dpos_mlp = dpos_mlp.clamp(min=-mlp_cap, max=mlp_cap)

            # v3.6 / Yol C: Fourier trajectory ekle (varsa)
            if getattr(gs, "fourier_pos_coeffs", None) is not None:
                dpos_fourier = decode_fourier_trajectory(gs.fourier_pos_coeffs, t)
                dpos = dpos_mlp + dpos_fourier
            else:
                dpos = dpos_mlp
            # v3.6.2: total dpos clamp
            total_cap = scene_extent * 0.2
            dpos = dpos.clamp(min=-total_cap, max=total_cap)

            d_means       = (gs.means + dpos)
            d_log_scales  = (gs.scales + dscale)         # log-space toplam
            d_quats       = F.normalize(gs.quats + dquat, dim=-1)
            # Final clamp on absolute log_scale — scene_extent cap
            max_log = torch.log(torch.tensor(max(scene_extent, 1.0), device=d_log_scales.device))
            d_log_scales = d_log_scales.clamp(max=float(max_log))
        else:
            d_means       = gs.means
            d_log_scales  = gs.scales
            d_quats       = F.normalize(gs.quats, dim=-1)

        out_path = output_dir / f"frame_{t_idx:04d}.ply"
        write_ply(
            means          = d_means.detach().cpu().numpy(),
            log_scales     = d_log_scales.detach().cpu().numpy(),
            quats          = d_quats.detach().cpu().numpy(),
            opacities_logit= gs.opacities.detach().cpu().numpy().squeeze(-1),
            sh_dc          = gs.sh_dc.detach().cpu().numpy().squeeze(1),
            sh_rest        = gs.sh_rest.detach().cpu().numpy(),
            output_path    = out_path,
        )
        written.append(out_path)
    print(f"✓ {len(written)} frame export edildi → {output_dir}")
    return written


def load_checkpoint_and_export(
    ckpt_path: str | Path,
    output_dir: str | Path,
    num_timestamps: int = 60,
    device: str = "cuda",
) -> List[Path]:
    """Checkpoint dosyasından modeli yükle ve export et."""
    ckpt = torch.load(ckpt_path, map_location=device)
    gs = GaussianModel.from_checkpoint(ckpt["gs"])
    deform = None
    if "deform" in ckpt:
        # Deformation field varsayılan boyutlarla rebuild edilir;
        # checkpoint state_dict ile gerçek boyutlar yüklenir
        from ..model.deformation import DeformationField
        # Resolution & feat_dim'i state_dict'ten çıkar
        plane_keys = [k for k in ckpt["deform"] if k.startswith("hexplane.planes.")]
        if plane_keys:
            sample_key = sorted(plane_keys)[0]
            shape = ckpt["deform"][sample_key].shape  # (1, feat_dim, R, R)
            feat_dim, R = shape[1], shape[2]
        else:
            feat_dim, R = 32, 64
        # MLP genişliği
        mlp_w = ckpt["deform"].get("mlp.0.weight", torch.zeros(256, 1)).shape[0]
        deform = DeformationField(resolution=R, feat_dim=feat_dim, mlp_width=mlp_w)
        deform.load_state_dict(ckpt["deform"])

    return export_to_ply(
        gs, deform, output_dir,
        num_timestamps=num_timestamps,
        scene_extent=float(ckpt.get("scene_extent", 1.0)),
        device=device,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Checkpoint → .ply export")
    p.add_argument("ckpt", type=str)
    p.add_argument("output_dir", type=str)
    p.add_argument("--num-timestamps", type=int, default=60)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    load_checkpoint_and_export(args.ckpt, args.output_dir,
                               args.num_timestamps, args.device)

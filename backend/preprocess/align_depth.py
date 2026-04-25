"""Faz 3a.5 — MiDaS depth'i COLMAP scale'e align et.

NEDEN GEREKLİ:
    MiDaS scale-invariant relative depth üretir. Scale ARBITRARY.
    COLMAP sparse reconstruction kendi scale'inde (scene_extent'e göre).
    Anchor unprojection (CoTracker track lift) MiDaS depth + COLMAP K/pose
    ile çalıştığından, scale uyumsuzluğu 3D anchor'ları YANLIŞ konuma koyar.
    Track loss asla sıfıra yaklaşamaz (chickchicken'da 0.15 sabit kaldı).

ÇÖZÜM:
    Her frame için:
      1. COLMAP 3D sparse points'i frame kamera'ya project et
      2. Görünür points için (u, v, z_colmap) çifti topla
      3. Aynı (u, v) pixel'lerde MiDaS depth değeri oku (z_midas)
      4. Median ratio: scale = median(z_colmap / z_midas)
      5. Depth map'i scale'le: aligned = midas * scale  (+ save)

    Sonuç: align edilmiş depth COLMAP world scale'inde.
    Anchor unprojection doğru 3D koordinat üretir.
    Track loss azalmaya başlar.
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from typing import Sequence


def align_depth_to_colmap(
    depth_dir: str | Path,
    frame_paths: Sequence[Path],
    cam_K: torch.Tensor,                              # (3, 3) intrinsic at frame res
    cam_w2c_per_frame: Sequence[torch.Tensor],        # List of (4, 4) world-to-cam
    colmap_xyz: torch.Tensor,                         # (M, 3) COLMAP sparse 3D points
    frame_size: tuple[int, int],                      # (W, H) of frame images
    overwrite: bool = True,
    min_valid_points: int = 10,
    device: str = "cpu",
) -> dict:
    """
    Her frame'in depth .npy'sını COLMAP scale'e align et.

    Returns:
        {
            "n_frames_aligned": int,
            "n_frames_skipped": int,
            "global_scale_median": float,    # Frames arası ortanca scale
            "scales_per_frame": list[float], # İzleme için
        }
    """
    depth_dir = Path(depth_dir)
    W, H = frame_size
    if len(frame_paths) != len(cam_w2c_per_frame):
        raise ValueError(
            f"frame_paths ({len(frame_paths)}) ve cam_w2c ({len(cam_w2c_per_frame)}) "
            f"uzunlukları eşleşmiyor"
        )

    cam_K = cam_K.to(device)
    colmap_xyz = colmap_xyz.to(device)
    ones = torch.ones(colmap_xyz.shape[0], 1, device=device)
    pts_h = torch.cat([colmap_xyz, ones], dim=-1)  # (M, 4)

    scales: list[float] = []
    n_aligned, n_skipped = 0, 0

    for i, frame_path in enumerate(frame_paths):
        depth_file = depth_dir / f"{Path(frame_path).stem}_depth.npy"
        if not depth_file.exists():
            n_skipped += 1
            continue

        try:
            depth = np.load(depth_file).astype(np.float32)
        except Exception as e:
            print(f"⚠ Depth .npy yüklenemedi: {depth_file} ({e})")
            n_skipped += 1
            continue

        dh, dw = depth.shape
        if (dw, dh) != (W, H):
            # Depth .npy farklı resolution'da — eşleştir (rare)
            import cv2
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        w2c = cam_w2c_per_frame[i].to(device)
        # Project COLMAP points → this camera
        cam_pts = (w2c @ pts_h.T).T          # (M, 4)
        z_colmap = cam_pts[:, 2]              # (M,)
        valid_z = z_colmap > 0.01

        # Perspective divide
        z_safe = z_colmap.clamp(min=0.01).unsqueeze(-1)
        cam_xy = cam_pts[:, :3] / z_safe      # (M, 3) with z=1 normalized
        img_h = (cam_K @ cam_xy.T).T          # (M, 3)
        u = img_h[:, 0]
        v = img_h[:, 1]

        in_bounds = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        valid = valid_z & in_bounds
        n_valid = int(valid.sum().item())

        if n_valid < min_valid_points:
            print(f"⚠ Frame {i} ({Path(frame_path).name}): "
                  f"sadece {n_valid} geçerli COLMAP point — align edilmedi")
            n_skipped += 1
            scales.append(1.0)
            continue

        u_int = u[valid].cpu().numpy().astype(np.int64)
        v_int = v[valid].cpu().numpy().astype(np.int64)
        u_int = np.clip(u_int, 0, W - 1)
        v_int = np.clip(v_int, 0, H - 1)
        z_colmap_v = z_colmap[valid].cpu().numpy()

        z_midas_v = depth[v_int, u_int]
        z_midas_safe = np.maximum(z_midas_v, 1e-3)

        ratio = z_colmap_v / z_midas_safe
        # Outlier'ları filtrele (üst/alt %5'i kırp, sonra median)
        ratio_finite = ratio[np.isfinite(ratio) & (ratio > 1e-6) & (ratio < 1e6)]
        if len(ratio_finite) < min_valid_points:
            n_skipped += 1
            scales.append(1.0)
            continue
        q5, q95 = np.percentile(ratio_finite, [5, 95])
        trimmed = ratio_finite[(ratio_finite >= q5) & (ratio_finite <= q95)]
        scale = float(np.median(trimmed))
        scales.append(scale)

        if overwrite:
            aligned = depth * scale
            np.save(depth_file, aligned.astype(np.float32))
        n_aligned += 1

    global_scale = float(np.median(scales)) if scales else 1.0
    print(f"✓ Depth align: {n_aligned} frame aligned, {n_skipped} skipped. "
          f"Global median scale: {global_scale:.4f}")
    if scales:
        print(f"  Per-frame scale range: [{min(scales):.3f}, {max(scales):.3f}]")

    return {
        "n_frames_aligned": n_aligned,
        "n_frames_skipped": n_skipped,
        "global_scale_median": global_scale,
        "scales_per_frame": scales,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MiDaS depth → COLMAP scale alignment")
    p.add_argument("depth_dir", type=str, help="Data directory with *_depth.npy")
    p.add_argument("colmap_dir", type=str, help="Data directory with COLMAP output")
    p.add_argument("frames_dir", type=str)
    p.add_argument("--no-overwrite", action="store_true", help="Sadece raporla, yazma")
    args = p.parse_args()

    # Standalone kullanım için basit CLI — pipeline tarafından direkt çağrılır aslında
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.preprocess.parse_colmap import parse_cameras, load_points3d

    cams = parse_cameras(args.colmap_dir)
    xyz, _rgb = load_points3d(args.colmap_dir)

    # Collect frame paths matching COLMAP image order
    frames_dir = Path(args.frames_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    assert len(frame_paths) == len(cams), (
        f"Frame/camera mismatch: {len(frame_paths)} vs {len(cams)}"
    )

    # K ve w2c listeleri
    K_first = torch.from_numpy(cams[0]["K"]).float()
    W = cams[0]["width"]
    H = cams[0]["height"]
    w2c_list = [torch.from_numpy(c["w2c"]).float() for c in cams]

    align_depth_to_colmap(
        depth_dir=args.depth_dir,
        frame_paths=frame_paths,
        cam_K=K_first,
        cam_w2c_per_frame=w2c_list,
        colmap_xyz=torch.from_numpy(xyz).float(),
        frame_size=(W, H),
        overwrite=not args.no_overwrite,
    )

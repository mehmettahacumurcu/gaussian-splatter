"""4D Quality v6.1 — DUSt3R sparse-view initialization (Madde 3).

Klasik COLMAP SfM'in seyrek-view (10-20 foto) durumunda yetersiz olduğu
senaryolarda DUSt3R kullanır. DUSt3R transformer-based dense pointmap +
relative pose tahmini yapar; sparse-view için 10× daha güvenilir.

DUSt3R repo: https://github.com/naver/dust3r
Pretrained weights: ~500 MB ilk run'da download.

Bu module opt-in: cfg.preprocess.init_method = "dust3r" veya "auto".
Auto modda frame_count < sparse_view_threshold_frames (default 20) ise tetiklenir.

Çıktı format: COLMAP-uyumlu sparse cloud (xyz + rgb numpy arrays) + cam pose dict.
Pipeline'ın geri kalanı bu çıktıyı COLMAP çıktısıyla aynı şekilde işler.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch


def is_dust3r_available() -> bool:
    """DUSt3R kurulu mu kontrol et (lazy import)."""
    try:
        import dust3r  # noqa: F401
        return True
    except ImportError:
        return False


def install_hint() -> str:
    return (
        "DUSt3R kurulu degil. Kurulum:\n"
        "  pip install git+https://github.com/naver/dust3r\n"
        "  ya da repo clone et + 'pip install -e .' calistir.\n"
        "Pretrained weights ilk run'da otomatik indirilir (~500 MB)."
    )


def estimate_sparse_init(
    frame_paths: list,
    output_dir: Path,
    image_size: int = 512,
    device: str = "cuda",
) -> dict:
    """DUSt3R ile sparse cloud + cam pose tahmini.

    Args:
      frame_paths: input image paths (RGB)
      output_dir: cache klasoru — pointmap.npy, cameras.json yazilir
      image_size: DUSt3R inference resolution (default 512)
      device: cuda|cpu
    Returns:
      dict: {
        "xyz": (N, 3) np.ndarray,
        "rgb": (N, 3) np.ndarray uint8,
        "cameras": {frame_name: {"K": (3,3), "w2c": (4,4)}},
        "n_points": int,
      }
    Raises:
      ImportError: DUSt3R yoksa
      RuntimeError: inference fail
    """
    if not is_dust3r_available():
        raise ImportError(install_hint())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache check
    cache_xyz = output_dir / "pointmap.npy"
    cache_rgb = output_dir / "pointmap_rgb.npy"
    cache_cams = output_dir / "cameras.json"
    if cache_xyz.exists() and cache_rgb.exists() and cache_cams.exists():
        import json
        xyz = np.load(cache_xyz)
        rgb = np.load(cache_rgb)
        with open(cache_cams) as f:
            cameras = json.load(f)
        cameras_t = {
            k: {
                "K": np.array(v["K"], dtype=np.float32),
                "w2c": np.array(v["w2c"], dtype=np.float32),
            }
            for k, v in cameras.items()
        }
        print(f"[DUSt3R] Cache hit: {len(xyz)} point, {len(cameras_t)} cam")
        return {"xyz": xyz, "rgb": rgb, "cameras": cameras_t, "n_points": len(xyz)}

    # Run DUSt3R
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    print(f"[DUSt3R] Loading model + running inference on {len(frame_paths)} images...")

    # Model — varsayilan checkpoint
    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    images = load_images([str(p) for p in frame_paths], size=image_size)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1)

    # Global alignment
    scene = global_aligner(
        output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer,
    )
    _ = scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

    # Extract dense pointcloud + colors
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    img_list = scene.imgs
    rgbs = [(img * 255).astype(np.uint8) if img.dtype != np.uint8 else img for img in img_list]

    # Concat all confident points
    xyz_all, rgb_all = [], []
    for pts, mask, rgb in zip(pts3d, confidence_masks, rgbs):
        pts_np = pts.detach().cpu().numpy() if isinstance(pts, torch.Tensor) else pts
        mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        valid = pts_np[mask_np]
        valid_rgb = rgb[mask_np] if rgb.shape[:2] == mask_np.shape else None
        xyz_all.append(valid)
        if valid_rgb is not None:
            rgb_all.append(valid_rgb)

    xyz = np.concatenate(xyz_all, axis=0)
    rgb = (
        np.concatenate(rgb_all, axis=0) if rgb_all
        else np.full((len(xyz), 3), 128, dtype=np.uint8)
    )

    # Subsample to manageable size (max 200k)
    if len(xyz) > 200_000:
        idx = np.random.choice(len(xyz), 200_000, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    # Camera poses
    cams = scene.get_im_poses()  # (N, 4, 4) cam-to-world
    Ks = scene.get_intrinsics()  # (N, 3, 3)
    cameras = {}
    for i, fp in enumerate(frame_paths):
        c2w = cams[i].detach().cpu().numpy() if isinstance(cams[i], torch.Tensor) else cams[i]
        w2c = np.linalg.inv(c2w)
        K_i = Ks[i].detach().cpu().numpy() if isinstance(Ks[i], torch.Tensor) else Ks[i]
        cameras[Path(fp).name] = {
            "K": K_i.astype(np.float32),
            "w2c": w2c.astype(np.float32),
        }

    # Cache
    np.save(cache_xyz, xyz.astype(np.float32))
    np.save(cache_rgb, rgb.astype(np.uint8))
    import json
    with open(cache_cams, "w") as f:
        json.dump(
            {k: {"K": v["K"].tolist(), "w2c": v["w2c"].tolist()}
             for k, v in cameras.items()},
            f, indent=2,
        )

    print(f"[DUSt3R] Done: {len(xyz)} point, {len(cameras)} cam")
    return {"xyz": xyz, "rgb": rgb, "cameras": cameras, "n_points": len(xyz)}


def should_use_dust3r(
    n_frames: int,
    init_method: str,
    threshold: int = 20,
) -> bool:
    """init_method'a gore DUSt3R kullanilsin mi karar ver."""
    if init_method == "dust3r":
        return True
    if init_method == "auto":
        return n_frames < threshold
    return False

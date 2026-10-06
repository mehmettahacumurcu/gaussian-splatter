"""Load a COLMAP scene (DL3DV gaussian_splat layout) into GPU-ready arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from .colmap_io import read_sparse_model


@dataclass
class Scene:
    name: str
    image_paths: list[Path]  # sorted by file name (= capture order for DL3DV)
    width: int
    height: int
    Ks: np.ndarray  # (F, 3, 3) at the loaded image resolution
    w2c: np.ndarray  # (F, 4, 4) world -> camera, normalised world
    points_xyz: np.ndarray  # (N, 3) normalised world
    points_rgb: np.ndarray  # (N, 3) float in [0, 1]
    scene_scale: float  # radius of the camera cloud after normalisation

    @property
    def num_frames(self) -> int:
        return len(self.image_paths)

    def camera_centers(self) -> np.ndarray:
        r = self.w2c[:, :3, :3]
        t = self.w2c[:, :3, 3]
        return -np.einsum("fij,fi->fj", r, t)

    def view_dirs(self) -> np.ndarray:
        # Camera +z (forward) in world coordinates = third row of R.
        return self.w2c[:, 2, :3]

    def load_image(self, i: int) -> np.ndarray:
        """Return frame i as float32 HxWx3 in [0, 1] at the scene resolution."""
        img = PILImage.open(self.image_paths[i]).convert("RGB")
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height), PILImage.BICUBIC)
        return np.asarray(img, dtype=np.float32) / 255.0


def _normalise(c2w_centers: np.ndarray) -> tuple[np.ndarray, float]:
    center = c2w_centers.mean(axis=0)
    radius = float(np.linalg.norm(c2w_centers - center, axis=1).max())
    scale = 1.0 / max(radius, 1e-8)
    return center, scale


def load_colmap_scene(
    scene_dir: Path,
    images_subdir: str = "images_4",
    sparse_subdir: str = "sparse/0",
    max_points: int = 300_000,
    name: str | None = None,
) -> Scene:
    """Load a scene whose undistorted images live in ``scene_dir/images_subdir``.

    COLMAP intrinsics are rescaled to the actual image size, so downsampled
    image folders (images_2/4/8) work with the full-resolution model.
    """
    scene_dir = Path(scene_dir)
    model = read_sparse_model(scene_dir / sparse_subdir)
    img_dir = scene_dir / images_subdir
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Missing image folder {img_dir}")

    by_name = {Path(im.name).name: im for im in model.images.values()}
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    files = [p for p in files if p.name in by_name]
    if len(files) < 10:
        raise ValueError(f"{scene_dir}: only {len(files)} registered images found in {images_subdir}")

    with PILImage.open(files[0]) as probe:
        width, height = probe.size

    Ks, w2cs = [], []
    for p in files:
        im = by_name[p.name]
        cam = model.cameras[im.camera_id]
        if cam.has_distortion():
            raise ValueError(f"{scene_dir}: camera {cam.id} is distorted ({cam.model}); use undistorted images")
        fx, fy, cx, cy = cam.intrinsics()
        sx, sy = width / cam.width, height / cam.height
        Ks.append(np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], np.float64))
        w2cs.append(im.world_to_camera())
    Ks = np.stack(Ks)
    w2c = np.stack(w2cs)

    # Normalise the world so cameras sit inside the unit sphere.
    centers = -np.einsum("fij,fi->fj", w2c[:, :3, :3], w2c[:, :3, 3])
    center, scale = _normalise(centers)
    c2w = np.linalg.inv(w2c)
    c2w[:, :3, 3] = (c2w[:, :3, 3] - center) * scale
    w2c = np.linalg.inv(c2w)
    xyz = (model.points_xyz - center) * scale
    rgb = model.points_rgb.astype(np.float32) / 255.0

    # Drop far outliers and subsample very dense clouds.
    keep = np.linalg.norm(xyz, axis=1) < 20.0
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    if len(xyz) < 100:
        raise ValueError(f"{scene_dir}: sparse cloud too small ({len(xyz)} points)")

    return Scene(
        name=name or scene_dir.name,
        image_paths=files,
        width=width,
        height=height,
        Ks=Ks.astype(np.float32),
        w2c=w2c.astype(np.float32),
        points_xyz=xyz.astype(np.float32),
        points_rgb=rgb,
        scene_scale=1.0,
    )

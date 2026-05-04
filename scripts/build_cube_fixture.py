"""Build a tiny synthetic test scene for the edit integration test.

Vectorized rasterization — all 10 frames build in a few seconds.

Output: tests/fixtures/cube_scene/{frames/, depth/, colmap/sparse/0/, output/ckpt/}

Run: python scripts/build_cube_fixture.py
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

H, W = 120, 160
FX = FY = 140.0
CX = W / 2
CY = H / 2

CUBE_HALF = 0.25
PLANE_Y = -0.5
PLANE_HALF = 2.0
CUBE_RGB = np.array([220, 60, 60], dtype=np.float32)    # red cube
PLANE_RGB = np.array([140, 140, 140], dtype=np.float32) # gray plane
BG_RGB = np.array([20, 22, 30], dtype=np.float32)

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "cube_scene"


def hemisphere_poses(n: int = 10) -> list[np.ndarray]:
    """Return n w2c (4x4) matrices on a hemisphere of radius 2 looking at origin."""
    poses = []
    radius = 2.0
    for i in range(n):
        theta = 2 * np.pi * i / n
        phi = np.pi / 4 + 0.1 * np.sin(2 * theta)
        cam_pos = np.array([
            radius * np.cos(phi) * np.cos(theta),
            radius * np.sin(phi),
            radius * np.cos(phi) * np.sin(theta),
        ])
        # Look-at origin, world-up = (0, 1, 0)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up_world = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up_world)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        # Convention: COLMAP/OpenCV camera +Z faces forward.
        # c2w columns: cam_x=right, cam_y=-up (OpenCV Y-down), cam_z=forward
        R_c2w = np.stack([right, -up, forward], axis=1)
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ cam_pos
        w2c = np.eye(4)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        poses.append(w2c)
    return poses


def render_view_vectorized(w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ray-cast: cube + plane scene. Returns (rgb H×W×3 uint8, depth H×W float32)."""
    K_inv = np.linalg.inv(
        np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float32)
    )
    R_c2w = w2c[:3, :3].T
    t_w2c = w2c[:3, 3]
    cam_pos = -R_c2w @ t_w2c  # (3,)

    # Pixel grid → ray directions (world frame)
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1)          # (H, W, 3)
    rays_cam = pix @ K_inv.T                                       # (H, W, 3)
    rays_world = rays_cam @ R_c2w.T                                # (H, W, 3)
    rays_world /= np.linalg.norm(rays_world, axis=-1, keepdims=True)

    o = np.broadcast_to(cam_pos[None, None, :], (H, W, 3))

    # === Plane intersection (y = PLANE_Y) ===
    ry = rays_world[..., 1]
    safe_ry = np.where(np.abs(ry) > 1e-6, ry, 1.0)
    t_plane = np.where(np.abs(ry) > 1e-6, (PLANE_Y - o[..., 1]) / safe_ry, np.inf)
    t_plane = np.where(t_plane > 1e-3, t_plane, np.inf)
    p_plane = o + t_plane[..., None] * rays_world
    plane_hit = (
        (np.abs(p_plane[..., 0]) <= PLANE_HALF)
        & (np.abs(p_plane[..., 2]) <= PLANE_HALF)
        & np.isfinite(t_plane)
    )
    t_plane = np.where(plane_hit, t_plane, np.inf)

    # === Cube intersection (AABB centered at origin, half = CUBE_HALF) ===
    safe_d = np.where(np.abs(rays_world) > 1e-6, rays_world, 1e-6)
    inv_d = 1.0 / safe_d
    t1 = (-CUBE_HALF - o) * inv_d
    t2 = ( CUBE_HALF - o) * inv_d
    tmin_each = np.minimum(t1, t2)
    tmax_each = np.maximum(t1, t2)
    t_enter = np.max(tmin_each, axis=-1)   # (H, W)
    t_exit  = np.min(tmax_each, axis=-1)
    cube_hit = t_exit > np.maximum(t_enter, 1e-3)
    t_cube = np.where(cube_hit, t_enter, np.inf)

    # Compose: nearest intersection wins
    rgb   = np.broadcast_to(BG_RGB[None, None, :], (H, W, 3)).copy()
    depth = np.full((H, W), 100.0, dtype=np.float32)

    plane_winner = (t_plane < t_cube) & np.isfinite(t_plane)
    cube_winner  = (t_cube <= t_plane) & np.isfinite(t_cube)

    rgb[plane_winner] = PLANE_RGB
    rgb[cube_winner]  = CUBE_RGB
    depth[plane_winner] = t_plane[plane_winner].astype(np.float32)
    depth[cube_winner]  = t_cube[cube_winner].astype(np.float32)

    return rgb.astype(np.uint8), depth


def quat_from_R(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix → quaternion (w, x, y, z), w > 0."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        diag = [R[0, 0], R[1, 1], R[2, 2]]
        i = int(np.argmax(diag))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s;                 qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    return float(qw), float(qx), float(qy), float(qz)


def write_colmap(poses: list[np.ndarray], frame_names: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cameras.txt").write_text(
        f"# Camera list\n1 PINHOLE {W} {H} {FX} {FY} {CX} {CY}\n"
    )
    lines = ["# Image list\n"]
    for i, (w2c, name) in enumerate(zip(poses, frame_names)):
        qw, qx, qy, qz = quat_from_R(w2c[:3, :3])
        tx, ty, tz = w2c[:3, 3]
        lines.append(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {name}\n\n")
    (out_dir / "images.txt").write_text("".join(lines))
    (out_dir / "points3D.txt").write_text("# 3D points\n")


def build_fake_checkpoint(out_path: Path) -> None:
    """Create a checkpoint compatible with GaussianModel.from_checkpoint().

    GaussianModel.__init__(init_points, init_colors, sh_degree, fourier_K)
    Checkpoint format: torch.save({iter, gs: state_for_save(), scene_extent, sh_degree}, path)
    state_for_save() = state_dict() (CPU tensors) | {sh_degree, num_points, fourier_K}
    """
    from backend.model.gaussian_model import GaussianModel

    rng = np.random.default_rng(123)

    # --- Cube surface points (1000) ---
    cube_pts = []
    for _ in range(1000):
        face = rng.integers(0, 6)
        u = rng.uniform(-CUBE_HALF, CUBE_HALF)
        v = rng.uniform(-CUBE_HALF, CUBE_HALF)
        coords = [u, v, 0.0]
        coords[face // 2] = CUBE_HALF if (face % 2 == 0) else -CUBE_HALF
        cube_pts.append(coords)

    # --- Plane points (4000) ---
    plane_pts = [
        [rng.uniform(-PLANE_HALF, PLANE_HALF), PLANE_Y, rng.uniform(-PLANE_HALF, PLANE_HALF)]
        for _ in range(4000)
    ]

    pts = np.array(cube_pts + plane_pts, dtype=np.float32)   # (5000, 3)
    cols_cube  = np.tile(CUBE_RGB  / 255.0, (1000, 1)).astype(np.float32)
    cols_plane = np.tile(PLANE_RGB / 255.0, (4000, 1)).astype(np.float32)
    cols = np.concatenate([cols_cube, cols_plane], axis=0)    # (5000, 3)

    gs = GaussianModel(
        init_points=torch.from_numpy(pts),
        init_colors=torch.from_numpy(cols),
        sh_degree=3,
        fourier_K=0,   # static scene — no Fourier trajectory
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gs_state = gs.state_for_save()   # state_dict() (CPU) + sh_degree/num_points/fourier_K
    torch.save({
        "iter": 0,
        "gs": gs_state,
        "scene_extent": 4.0,
        "sh_degree": 3,
    }, out_path)
    print(f"  Checkpoint saved: {out_path.stat().st_size / 1024:.1f} KB, {gs.num_points} gaussians")


def main() -> None:
    import time
    t0 = time.time()

    print(f"Building cube fixture at {FIXTURE_DIR}")
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    frames_dir = FIXTURE_DIR / "frames"
    depth_dir  = FIXTURE_DIR / "depth"
    colmap_dir = FIXTURE_DIR / "colmap" / "sparse" / "0"
    ckpt_path  = FIXTURE_DIR / "output" / "ckpt" / "ckpt_final.pt"

    frames_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    poses = hemisphere_poses(10)
    frame_names = [f"frame_{i:06d}.png" for i in range(10)]

    for i, w2c in enumerate(poses):
        rgb, depth = render_view_vectorized(w2c)
        Image.fromarray(rgb).save(frames_dir / frame_names[i])
        np.save(depth_dir / f"frame_{i:06d}_depth.npy", depth)

    print(f"  Rendered 10 frames at {W}x{H}  ({time.time() - t0:.1f}s)")

    write_colmap(poses, frame_names, colmap_dir)
    print(f"  COLMAP files at {colmap_dir}")

    build_fake_checkpoint(ckpt_path)

    elapsed = time.time() - t0
    print(f"Cube fixture build complete in {elapsed:.1f}s.")
    if elapsed > 30:
        print(f"  WARNING: exceeded 30s budget ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()

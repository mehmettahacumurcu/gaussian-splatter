"""Generate (degraded render, real photo, reference photo) pairs for one scene.

For each recipe a deliberately weak splat is trained on a sparse subset of the
frames; every non-training frame is rendered and saved next to its real photo.
The reference is the real photo of the closest training camera.

Output is one tar per scene so Drive sees a single large atomic file:

    index.json
    <recipe>/deg/<frame>.jpg   render of the weak splat
    <recipe>/gt/<frame>.jpg    real photo at the same pose
    <recipe>/ref/<frame>.jpg   real photo of a training camera (shared)
    eval/<frame>.jpg           (test scenes) never-trained frames for distill-back
"""
from __future__ import annotations

import json
import shutil
import tarfile
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .gs_trainer import TrainConfig, View, params_to_cpu_state, psnr, render, train_gaussians
from .scene import Scene, load_colmap_scene
from .splits import RECIPES, make_split, nearest_reference, subsample
from .workspace import Workspace, atomic_copy

JPEG_QUALITY = 95


def _save_jpg(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, quality=JPEG_QUALITY)


def _fname(i: int) -> str:
    return f"{i:05d}.jpg"


def generate_scene_pairs(
    ws: Workspace,
    scene_name: str,
    scene_dir: Path,
    recipes: list[str],
    is_test: bool,
    max_pairs_per_recipe: int = 100,
    images_subdir: str = "images_4",
    device: str = "cuda",
    log=print,
) -> Path:
    t0 = time.time()
    scene = load_colmap_scene(scene_dir, images_subdir=images_subdir, name=scene_name)
    log(f"  {scene_name[:12]}: {scene.num_frames} frames {scene.width}x{scene.height}, {len(scene.points_xyz)} points")
    # uint8 keeps a 300-frame 960p scene at ~0.5 GB of RAM.
    photos_u8 = [(scene.load_image(i) * 255 + 0.5).astype(np.uint8) for i in range(scene.num_frames)]
    photo = lambda i: photos_u8[i].astype(np.float32) / 255.0
    centers, forwards = scene.camera_centers(), scene.view_dirs()
    Ks = torch.as_tensor(scene.Ks, device=device)
    w2c = torch.as_tensor(scene.w2c, device=device)

    work = ws.local / "pairs_tmp" / scene_name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    index = {
        "scene": scene_name,
        "width": scene.width,
        "height": scene.height,
        "is_test": is_test,
        "cameras": {"K": scene.Ks.tolist(), "w2c": scene.w2c.tolist()},
        "recipes": {},
    }

    for rname in recipes:
        recipe = RECIPES[rname]
        split = make_split(scene.num_frames, recipe)
        views = [View(torch.as_tensor(photo(i), device=device), w2c[i], Ks[i]) for i in split.train]
        cfg = TrainConfig(steps=recipe.steps, seed=zlib.crc32(scene_name.encode()) % 10_000)
        t1 = time.time()
        params = train_gaussians(
            views, scene.width, scene.height, cfg, init_xyz=scene.points_xyz, init_rgb=scene.points_rgb
        )
        train_s = time.time() - t1

        pair_frames = subsample(split.pair, max_pairs_per_recipe)
        items = []
        with torch.no_grad():
            for i in pair_frames:
                img, _ = render(params, w2c[i], Ks[i], scene.width, scene.height, cfg.sh_degree)
                deg = img.float().cpu().numpy()
                ref = nearest_reference(i, split.train, centers, forwards)
                _save_jpg(deg, work / rname / "deg" / _fname(i))
                _save_jpg(photo(i), work / rname / "gt" / _fname(i))
                ref_path = work / rname / "ref" / _fname(ref)
                if not ref_path.exists():
                    _save_jpg(photo(ref), ref_path)
                items.append(
                    {
                        "frame": i,
                        "ref": ref,
                        "deg": f"{rname}/deg/{_fname(i)}",
                        "gt": f"{rname}/gt/{_fname(i)}",
                        "ref_path": f"{rname}/ref/{_fname(ref)}",
                        "psnr": round(psnr(img, torch.as_tensor(photo(i), device=device)), 3),
                    }
                )
        if is_test:
            # Distill-back needs every training photo and the splat itself.
            for i in split.train:
                p = work / rname / "ref" / _fname(i)
                if not p.exists():
                    _save_jpg(photo(i), p)
            out = ws.splat_path(scene_name, rname)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = ws.local / "splat_tmp.pt"
            torch.save(params_to_cpu_state(params), tmp)
            atomic_copy(tmp, out)
            tmp.unlink(missing_ok=True)

        mean_psnr = float(np.mean([it["psnr"] for it in items])) if items else float("nan")
        index["recipes"][rname] = {
            "train": split.train,
            "pair": split.pair,
            "eval": split.eval,
            "steps": recipe.steps,
            "num_gaussians": int(len(params["means"])),
            "train_seconds": round(train_s, 1),
            "mean_psnr": round(mean_psnr, 3),
            "items": items,
        }
        log(
            f"    {rname}: train {len(split.train)} views, {len(params['means'])} gaussians, "
            f"{train_s:.0f}s, {len(items)} pairs, render PSNR {mean_psnr:.2f}"
        )
        del params, views
        torch.cuda.empty_cache()

    if is_test:
        ev = make_split(scene.num_frames, RECIPES[recipes[0]]).eval
        for i in ev:
            _save_jpg(photo(i), work / "eval" / _fname(i))

    (work / "index.json").write_text(json.dumps(index), encoding="utf-8")
    local_tar = ws.local / f"{scene_name}.tar"
    with tarfile.open(local_tar, "w") as tar:
        tar.add(work, arcname=scene_name)
    atomic_copy(local_tar, ws.pair_tar(scene_name))
    local_tar.unlink(missing_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    log(f"  {scene_name[:12]}: done in {time.time() - t0:.0f}s")
    return ws.pair_tar(scene_name)


def extract_pairs(ws: Workspace, scenes: list[str]) -> Path:
    """Extract the tars of ``scenes`` to local disk (skips already extracted ones)."""
    out = ws.local_pairs
    out.mkdir(parents=True, exist_ok=True)
    for s in scenes:
        if (out / s / "index.json").exists():
            continue
        tar_path = ws.pair_tar(s)
        if not tar_path.exists():
            raise FileNotFoundError(f"Pairs for scene {s} are missing: {tar_path}")
        with tarfile.open(tar_path) as tar:
            tar.extractall(out)
    return out


def load_index(pairs_root: Path, scene: str) -> dict:
    return json.loads((pairs_root / scene / "index.json").read_text(encoding="utf-8"))

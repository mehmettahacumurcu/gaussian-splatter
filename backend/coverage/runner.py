"""The coverage job: read splat + cameras, scan, score the views, write the heatmap."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..compose.plyio import read_ply, write_ply
from . import compute
from .colmap import CameraSet, read_cameras
from .detect import find_images
from .store import CoverageStore

ProgressFn = Callable[[str, float, str, dict], None]
TEST_PREFIXES = ("eval_", "test_")


def is_test_view(name: str) -> bool:
    """Held-out views (Spirula ``--eval-mode filename``) are scored but not counted as coverage."""
    return Path(name).name.lower().startswith(TEST_PREFIXES)


def camera_to_world(w2c: np.ndarray) -> np.ndarray:
    R, t = w2c[:3, :3], w2c[:3, 3]
    m = np.eye(4)
    m[:3, :3] = R.T
    m[:3, 3] = -R.T @ t
    return m


def view_records(cams: CameraSet, stats: list[dict], test: list[bool]) -> list[dict]:
    out = []
    for i, name in enumerate(cams.names):
        w, h = (int(v) for v in cams.sizes[i])
        fy = float(cams.K[i][1, 1])
        out.append({
            "index": i, "name": name, "test": test[i],
            "c2w": [round(float(v), 6) for v in camera_to_world(cams.w2c[i]).reshape(-1)],
            "fov_y": float(np.degrees(2 * np.arctan(h / 2 / fy))), "aspect": w / h,
            "score": round(stats[i]["score"], 4), "weak": round(stats[i]["weak"], 4),
        })
    return out


def world_up(cams: CameraSet) -> list[float]:
    """Average 'up' of the cameras (COLMAP image y points down)."""
    ups = [-camera_to_world(m)[:3, 1] for m in cams.w2c]
    up = np.mean(ups, axis=0)
    norm = np.linalg.norm(up)
    return (up / norm).tolist() if norm > 1e-6 else [0.0, -1.0, 0.0]


def run_coverage(
    store: CoverageStore, result_id: str, name: str, ply: Path, sparse: Path,
    on_progress: ProgressFn, cancelled: Callable[[], bool] = lambda: False, device: str = "cuda",
) -> dict[str, Any]:
    import torch

    started = time.time()
    on_progress("coverage_load", 0.0, "Kameralar okunuyor", {})
    cams = read_cameras(sparse)
    test = [is_test_view(n) for n in cams.names]
    train_idx = [i for i, t in enumerate(test) if not t]
    if not train_idx:
        raise ValueError("Modelde eğitim kamerası yok (hepsi eval_/test_)")
    train = cams.subset(train_idx)
    on_progress("coverage_load", 0.3, f"Splat okunuyor: {ply.name}", {"cameras": len(cams)})
    cloud = read_ply(ply)
    t = compute.splat_tensors(cloud, device)
    on_progress("coverage_load", 1.0, f"{cloud.count:,} splat, {len(train)} eğitim kamerası", {})

    tmp = store.tmp_dir(result_id)
    try:
        cov = compute.accumulate(t, train, on_progress=lambda p, m: on_progress("coverage_scan", p, m, {}),
                                 cancelled=cancelled)
        s = compute.score(cov["count"])
        summary = compute.summary(cov, s, cloud.opacities)
        if summary["opaque_splats"] and summary["unseen_share"] > 0.8:
            raise ValueError(
                f"Splatın %{summary['unseen_share'] * 100:.0f}'ini hiçbir kamera görmüyor: "
                "PLY ile COLMAP modeli aynı koordinatlarda değil gibi (farklı koşunun modeli olabilir).")
        stats = compute.view_stats(t, s, cams, on_progress=lambda p, m: on_progress("coverage_views", p, m, {}))
        del t
        torch.cuda.empty_cache()

        on_progress("coverage_write", 0.1, "Isı haritası yazılıyor", {})
        write_ply(compute.heat_cloud(cloud, s), tmp / "heat.ply")
        np.savez_compressed(tmp / "coverage.npz", count=cov["count"], spread=cov["spread"],
                            px_per_unit=cov["px_per_unit"], score=s)
        views = view_records(cams, stats, test)
        (tmp / "views.json").write_text(json.dumps(views), encoding="utf-8")
        images = find_images(sparse, cams.names)
        meta = {
            "id": result_id, "name": name, "created_at": time.time(),
            "ply_path": str(ply), "sparse_dir": str(sparse), "images_dir": str(images) if images else None,
            "cameras": len(cams), "test_views": sum(test), "up": world_up(train),
            "seconds": round(time.time() - started, 1), "summary": summary,
        }
        (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        store.publish(result_id, tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    on_progress("coverage_write", 1.0, "Hazır", {})
    return {"coverage_id": result_id, **summary}

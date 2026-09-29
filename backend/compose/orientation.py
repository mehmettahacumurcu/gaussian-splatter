"""Per-asset "up" estimation for the scene composer.

Real reconstructions are not aligned to +Y (a COLMAP capture can have its floor
normal anywhere on the sphere), so snap-to-ground and the orbit camera need the
measured up direction. This wraps the RANSAC floor-plane estimator from
``image_to_scene.orientation`` and caches the answer next to the composer data:

    <data_root>/compose/cache/<sha1 of asset_id>.orientation.json

The cache entry records the source file's size and mtime_ns and is recomputed
whenever either differs.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING

import numpy as np

from ..image_to_scene.orientation import estimate_world_orientation
from .plyio import read_ply
from .store import _atomic_write_text

if TYPE_CHECKING:
    from .store import ComposeStore


class MeshAssetError(ValueError):
    """Up estimation is only defined for splat assets."""


def _cache_path(store: "ComposeStore", asset_id: str):
    digest = hashlib.sha1(asset_id.encode("utf-8")).hexdigest()
    return store.root / "cache" / f"{digest}.orientation.json"


def _read_cache(path, size: int, mtime_ns: int) -> dict | None:
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry["size"] != size or entry["mtime_ns"] != mtime_ns:
            return None
        res = entry["result"]
        up = [float(c) for c in res["up"]]
        if len(up) != 3:
            return None
        return {
            "up": up,
            "tilt_deg": float(res["tilt_deg"]),
            "plane_inlier_frac": float(res["plane_inlier_frac"]),
            "above_below_ratio": float(res["above_below_ratio"]),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _compute(ply_path) -> dict:
    cloud = read_ply(ply_path)
    opac = np.clip(cloud.opacities.astype(np.float64), -30.0, 30.0)
    weights = 1.0 / (1.0 + np.exp(-opac))
    wo = estimate_world_orientation(cloud.means, weights)
    up = np.asarray(wo.up_raw, dtype=np.float64)
    norm = float(np.linalg.norm(up))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("orientation estimate is degenerate")
    up = up / norm
    result = {
        "up": [float(c) for c in up],
        "tilt_deg": float(wo.tilt_deg),
        "plane_inlier_frac": float(wo.plane_inlier_frac),
        "above_below_ratio": float(wo.above_below_ratio),
    }
    if not all(math.isfinite(v) for v in (*result["up"], result["tilt_deg"],
                                          result["plane_inlier_frac"],
                                          result["above_below_ratio"])):
        raise ValueError("orientation estimate is not finite")
    return result


def estimate_asset_up(store: "ComposeStore", asset_id: str) -> dict | None:
    """Return the up estimate for a splat asset, or None if the asset is unknown.

    Raises ``MeshAssetError`` for mesh assets.
    """
    asset = store.get_asset(asset_id)
    if asset is None:
        return None
    if asset.kind != "splat":
        raise MeshAssetError(f"Up estimation needs a splat asset; {asset_id} is a {asset.kind}")
    path = store.asset_path(asset_id)
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return None

    cache = _cache_path(store, asset_id)
    cached = _read_cache(cache, st.st_size, st.st_mtime_ns)
    if cached is not None:
        return cached

    result = _compute(path)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(cache, json.dumps(
            {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "result": result}))
    except OSError:
        pass  # the cache is an optimisation; never fail the request over it
    return result

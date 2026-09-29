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
from .fsutil import atomic_write_text
from .plyio import read_means_opacity

if TYPE_CHECKING:
    from .store import ComposeStore


class MeshAssetError(ValueError):
    """Up estimation is only defined for splat assets."""


def _cache_path(store: "ComposeStore", asset_id: str):
    digest = hashlib.sha1(asset_id.encode("utf-8")).hexdigest()
    return store.root / "cache" / f"{digest}.orientation.json"


CACHE_VERSION = 1


def _validated(result: dict) -> dict:
    """Normalise and check an orientation result; raises ValueError if invalid."""
    try:
        up = [float(c) for c in result["up"]]
        tilt = float(result["tilt_deg"])
        frac = float(result["plane_inlier_frac"])
        ratio = float(result["above_below_ratio"])
        measured = result["measured"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed orientation result: {exc}") from exc
    if len(up) != 3 or not all(math.isfinite(c) for c in up):
        raise ValueError("up must be 3 finite numbers")
    norm = math.sqrt(sum(c * c for c in up))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"up must be unit length (norm={norm:.4f})")
    if not all(math.isfinite(v) for v in (tilt, frac, ratio)):
        raise ValueError("orientation metrics must be finite")
    if not isinstance(measured, bool):
        raise ValueError("measured must be a bool")
    return {
        "up": [c / norm for c in up],
        "tilt_deg": tilt,
        "plane_inlier_frac": frac,
        "above_below_ratio": ratio,
        "measured": measured,
    }


def _read_cache(path, size: int, mtime_ns: int) -> dict | None:
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry["v"] != CACHE_VERSION or entry["size"] != size or entry["mtime_ns"] != mtime_ns:
            return None
        return _validated(entry["result"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _compute(ply_path) -> dict:
    means, opacities = read_means_opacity(ply_path)
    weights = 1.0 / (1.0 + np.exp(-np.clip(opacities.astype(np.float64), -30.0, 30.0)))
    wo = estimate_world_orientation(means, weights)
    frac = float(wo.plane_inlier_frac)
    return _validated({
        "up": [float(c) for c in wo.up_raw],
        "tilt_deg": float(wo.tilt_deg),
        "plane_inlier_frac": frac,
        "above_below_ratio": float(wo.above_below_ratio),
        # The estimator returns a placeholder +Y with zero inliers when it has
        # too few points or finds no plane; that is "unmeasured", not a floor.
        "measured": frac > 0.0,
    })


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
        atomic_write_text(cache, json.dumps(
            {"v": CACHE_VERSION, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
             "result": result}))
    except OSError:
        pass  # the cache is an optimisation; never fail the request over it
    return result

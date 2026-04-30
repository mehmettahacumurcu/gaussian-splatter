"""Phase 1.1-1.3 — Heavy preprocessing cache utilities.

Tum preprocessing adimlari icin "skip-if-cached" davranisi tek bir yerde
toplaniyor. Pipeline.py'deki cagrilar bu helper'lari kullanir.

Tasarim:
  - Her preprocessing step'in bir "is_cached" check'i var (output dosyalari
    valid mi)
  - force_preprocess=True ile cache bypass edilir (hepsi yeniden kosacak)
  - Phase 1.3: Settings hash validation. Her step'in baglandigi config
    alanlari _STEP_SETTINGS'te tanimli. Cache marker'a settings_hash yazilir;
    bir sonraki check'te current cfg ile karsilastirilir, mismatch -> miss.
    Bu, "fps 10 -> 20 yaptim, frames'i tekrar koylasin" gibi durumlari
    otomatik halleder.

Kullanim ornegi:

    from .cache_utils import is_step_cached, log_cache, write_cache_marker

    if not force_preprocess and is_step_cached(paths, "depth", cfg=cfg, scene_dir=paths["base"]):
        log_cache("depth", paths["depth"], hit=True)
        # skip step
    else:
        log_cache("depth", paths["depth"], hit=False)
        run_depth_estimation(...)
        write_cache_marker(paths["base"], "depth", {"count": 300}, cfg=cfg)
"""
from __future__ import annotations
import hashlib
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional


# Step definitions — her adimin output formatini tanimla.
# (path_key in scene_paths, validation_strategy, expected_pattern)
_STEP_VALIDATORS = {
    # path_key, glob_pattern_or_file, min_count
    "frames":       ("frames",       "frame_*.png", 1),
    "frames_mv":    ("frames_mv",    "cam*/frame_*.png", 1),
    "colmap":       ("colmap",       "sparse/*/cameras.txt", 1),
    "colmap_mv":    ("colmap_mv",    "sparse/*/cameras.txt", 1),
    "depth":        ("depth",        "*_depth.npy", 1),  # MiDaS/Metric3D depth maps (.npy)
    "depth_mv":     ("depth_mv",     "cam*/*_depth.npy", 1),  # per-cam (.npy)
    "tracks":       ("tracks",       "tracks.pt", 1),  # single file
    "masks":        ("masks",        "mask_*.png", 1),
    "masks_mv":     ("masks_mv",     "cam*/mask_*.png", 1),
    "mvs_dense":    ("colmap_mv",    "dense/fused.ply", 1),
    # Phase 1.8 — RAFT optical flow
    "flow":         ("flow",         "forward_*.pt", 1),
    "flow_mv":      ("flow_mv",      "cam*/forward_*.pt", 1),
}


# Phase 1.3 — Step settings dependencies.
# Her step'in cache'inin baglandigi config alanlari (dotted path, cfg.<a>.<b>).
# Bu alanlardan biri degisirse settings_hash degisir, cache invalidate olur.
_STEP_SETTINGS: dict[str, list[str]] = {
    "frames": [
        "preprocess.fps",
        "preprocess.resize_long_edge",
    ],
    "frames_mv": [
        "preprocess.fps",
        "preprocess.resize_long_edge",
        "preprocess.multiview_parallel_extract",
    ],
    "colmap": [
        "preprocess.colmap_camera_model",
        "preprocess.colmap_matching",
        "preprocess.sequential_overlap",
        "preprocess.init_subsample_mode",
    ],
    "colmap_mv": [
        "preprocess.colmap_camera_model",
        "preprocess.colmap_mv_timestamps",
        "preprocess.colmap_mv_dense_mvs",
        "preprocess.colmap_matching",
    ],
    "depth": [
        "foundation.metric3d_model",
        "preprocess.resize_long_edge",  # depth resolution depth frame'e bagli
    ],
    "depth_mv": [
        "foundation.metric3d_model",
        "preprocess.resize_long_edge",
    ],
    "tracks": [
        "foundation.cotracker_grid_size",
        "foundation.cotracker_num_points",
    ],
    "masks": [
        "foundation.sam2_threshold",
    ],
    "masks_mv": [
        "foundation.sam2_threshold",
    ],
    "mvs_dense": [
        "preprocess.colmap_mv_dense_mvs",
        "preprocess.colmap_mv_timestamps",
    ],
    "flow": [
        "preprocess.fps",
        "preprocess.resize_long_edge",
    ],
    "flow_mv": [
        "preprocess.fps",
        "preprocess.resize_long_edge",
    ],
}


def _resolve_path(obj: Any, dotted: str) -> Any:
    """cfg.preprocess.fps gibi 'preprocess.fps' yolunu cozer."""
    cur = obj
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


def _to_jsonable(v: Any) -> Any:
    """Path/dataclass -> JSON-friendly."""
    if isinstance(v, Path):
        return str(v)
    if is_dataclass(v):
        return asdict(v)
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_jsonable(x) for k, x in v.items()}
    return v


def get_step_settings(cfg: Any, step: str) -> dict[str, Any]:
    """Step'in cache'ini etkileyen ayarlari extract et."""
    if cfg is None or step not in _STEP_SETTINGS:
        return {}
    out = {}
    for path in _STEP_SETTINGS[step]:
        try:
            out[path] = _to_jsonable(_resolve_path(cfg, path))
        except AttributeError:
            # Older config (eski snapshot) — alan yok, atla
            out[path] = None
    return out


def compute_settings_hash(cfg: Any, step: str) -> str:
    """Step'in current settings'ini deterministik hash'le.

    SHA256 hex (12 char prefix dondururuz, marker'da okunur tutmak icin).
    """
    settings = get_step_settings(cfg, step)
    blob = json.dumps(settings, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def is_step_cached(
    paths: dict,
    step: str,
    min_count: Optional[int] = None,
    cfg: Any = None,
    scene_dir: Optional[Path] = None,
) -> bool:
    """Bir preprocessing step'in output'u zaten valid mi?

    Validation order:
      1) Files exist (glob pattern + min_count).
      2) [Phase 1.3] cfg verilirse: marker.settings_hash == compute_settings_hash(cfg, step).
         scene_dir verilmezse paths["base"] kullanilir.

    Args:
        paths:      scene_paths(scene) cikti'si (dict path_key -> Path)
        step:       _STEP_VALIDATORS'taki bir key
        min_count:  glob match >= bu sayi olmali (default _STEP_VALIDATORS'tan)
        cfg:        Config dataclass; verilirse settings hash karsilastirmasi yapilir
        scene_dir:  Marker'larin oldugu base dir (default paths["base"])

    Returns:
        True = cache valid (skip step), False = miss (run step)
    """
    if step not in _STEP_VALIDATORS:
        return False
    path_key, pattern, default_min = _STEP_VALIDATORS[step]
    if min_count is None:
        min_count = default_min

    # 1) File existence
    base = paths.get(path_key)
    if base is None or not Path(base).exists():
        return False
    base = Path(base)
    matches = list(base.glob(pattern))
    if len(matches) < min_count:
        return False

    # 2) Phase 1.3 settings hash check (yalnizca cfg verilmisse)
    if cfg is not None:
        sd = scene_dir if scene_dir is not None else paths.get("base")
        if sd is None:
            return True  # marker dir bulunamiyor -> existence check yeterli
        marker = read_cache_marker(Path(sd), step)
        if marker is None:
            # File var ama marker yok -> eski cache, hash bilinmiyor.
            # Conservative davran: miss say (yeniden kos + marker yaz).
            return False
        cached_hash = marker.get("settings_hash")
        current_hash = compute_settings_hash(cfg, step)
        if cached_hash != current_hash:
            return False

    return True


def log_cache(step: str, path: Path, hit: bool, count: Optional[int] = None,
              reason: Optional[str] = None) -> None:
    """Konsola tutarli format'ta cache hit/miss log."""
    icon = "✓" if hit else "→"
    status = "CACHE HIT (skip)" if hit else "computing"
    suffix = f" — {count} item" if count is not None else ""
    if reason and not hit:
        suffix += f" [{reason}]"
    print(f"  {icon} [{step}] {status}{suffix} @ {path}")


def cache_summary(paths: dict, cfg: Any = None) -> dict:
    """Tum step'lerin cache durumunu raporla (debug/diagnostic)."""
    summary = {}
    for step in _STEP_VALIDATORS:
        summary[step] = is_step_cached(paths, step, cfg=cfg)
    return summary


def write_cache_marker(
    scene_dir: Path,
    step: str,
    metadata: dict,
    cfg: Any = None,
) -> None:
    """Bir adim tamamlanldi diye .cache_marker yaz.

    Phase 1.3: cfg verilirse settings_hash + settings (debug icin) eklenir.
    timestamp her zaman yazilir.
    """
    marker_dir = Path(scene_dir) / ".cache_markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"{step}.json"

    payload = {
        "step": step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metadata": metadata,
    }
    if cfg is not None:
        payload["settings_hash"] = compute_settings_hash(cfg, step)
        payload["settings"] = get_step_settings(cfg, step)

    marker.write_text(json.dumps(payload, indent=2, default=str))


def read_cache_marker(scene_dir: Path, step: str) -> Optional[dict]:
    """Marker oku, yoksa None."""
    marker = Path(scene_dir) / ".cache_markers" / f"{step}.json"
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text())
    except Exception:
        return None


def invalidate_cache(scene_dir: Path, step: Optional[str] = None) -> None:
    """Cache marker'lari sil. step=None tum marker'lari siler."""
    marker_dir = Path(scene_dir) / ".cache_markers"
    if not marker_dir.exists():
        return
    if step is None:
        for f in marker_dir.glob("*.json"):
            f.unlink()
    else:
        marker = marker_dir / f"{step}.json"
        if marker.exists():
            marker.unlink()


# ---------------------------------------------------------------------------
# Phase 1.3 — Cache manifest (toplu rapor)
# ---------------------------------------------------------------------------

def build_manifest(scene_dir: Path, cfg: Any = None) -> dict:
    """Sahnenin tum step marker'larini tek manifest icine topla.

    Output:
        {
          "scene_dir": "...",
          "generated_at": "2026-04-28T12:34:56",
          "steps": {
              "frames":     {"cached": True/False, "marker": {...}, "current_hash": "...",
                             "match": True/False},
              ...
          }
        }

    cfg verilirse current_hash ve match alanlari doldurulur.
    """
    sd = Path(scene_dir)
    out: dict[str, Any] = {
        "scene_dir": str(sd),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "steps": {},
    }
    for step in _STEP_VALIDATORS:
        marker = read_cache_marker(sd, step)
        entry: dict[str, Any] = {
            "cached": marker is not None,
            "marker": marker,
        }
        if cfg is not None:
            cur = compute_settings_hash(cfg, step)
            entry["current_hash"] = cur
            entry["match"] = bool(marker and marker.get("settings_hash") == cur)
        out["steps"][step] = entry
    return out


def write_manifest(scene_dir: Path, cfg: Any = None) -> Path:
    """Manifest'i .cache_markers/_manifest.json olarak yaz, path dondur."""
    manifest = build_manifest(scene_dir, cfg=cfg)
    sd = Path(scene_dir)
    marker_dir = sd / ".cache_markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    out_path = marker_dir / "_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    return out_path

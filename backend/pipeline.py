"""End-to-end pipeline orchestrator.

Bir video → 4DGS sahnesi:
  1) Frame çıkarma (ffmpeg)
  2) COLMAP SfM
  3) Foundation modeller (depth + tracking + masking)  [opsiyonel: --skip-foundation]
  4) Gaussian model + DeformationField init
  5) Training loop
  6) Export (.ply)

Kullanım:
  python -m backend.pipeline data/test_scene/video.mp4 --scene test_scene
"""
from __future__ import annotations
import argparse
import sys
import time
import numpy as np
import torch
from pathlib import Path
from typing import Any, Callable

from .config import default_config, cloud_config, scene_paths, Config, is_multiview_scene
# v5.0 — multi-view orchestration
from .preprocess.multiview_pipeline import prepare_multiview_scene
from .preprocess.extract_frames import extract_frames
from .preprocess.run_colmap     import run_colmap
from .preprocess.parse_colmap   import parse_cameras, load_points3d, scene_extent as compute_scene_extent
from .preprocess.frame_alignment import join_registered_frames
# Phase 1.1 + 1.3: heavy preprocessing cache utilities
from .preprocess.cache_utils    import (
    is_step_cached, log_cache, write_cache_marker,
    compute_settings_hash, read_cache_marker,
)
from .model.gaussian_model      import GaussianModel
from .model.deformation         import DeformationField
from .model.trainer             import Trainer4DGS
from .export.to_splat           import export_to_ply
from .run_logger                import RunLogger


# progress_callback imzası:
#   (phase_name: str, progress_0_1: float, message: str, details: dict) -> None
ProgressCallback = Callable[[str, float, str, dict[str, Any]], None]


def _noop_cb(phase: str, progress: float, message: str = "",
             details: dict[str, Any] | None = None) -> None:
    """Callback verilmediğinde kullanılan no-op."""
    pass


def _static_trainer_overrides(cfg: Config) -> tuple[int, str]:
    """(fourier_K, deform_pos_mode) — static modda ikisi birlikte degismeli.

    Static 3DGS'te temporal Fourier trajectory yok: fourier_K=0 ile per-gaussian
    (N, K, 2, 3) coeffs + Adam state hic allocate edilmez (8 GB kartlarda gercek
    VRAM kazanci; parametrelerin LR'i zaten 0'di). Ama fourier_K=0 iken trainer
    init'i 'hybrid'/'fourier' deform_pos_mode'u (dogru olarak) ValueError'la
    reddeder — mod 'mlp' olmali. Static modda deformation train()'de zaten
    bypass edildigi icin mod inert'tir, sadece validation'i gecmesi gerekir
    (image_to_scene/runner.py'daki static B trainer ile ayni cozum).
    """
    if getattr(cfg.train, "static_mode", False):
        return 0, "mlp"
    return cfg.model.fourier_K, cfg.model.deform_pos_mode


_EXPERIMENT_TRAIN_KWARGS = frozenset(
    {"validity_mask", "density_quality_probe"}
)
_EXPLICIT_PIPELINE_TRAIN_KWARGS = frozenset(
    {
        "n_iters", "image_size", "ckpt_dir", "ckpt_interval", "log_interval",
        "progress_callback", "depth_dir", "mask_dir", "tracks_path", "flow_dir",
        "run_logger", "mv_frame_paths", "mv_cam_K", "mv_w2c", "mv_test_camera",
        "depth_mv_dir", "masks_mv_dir", "flow_mv_dir", "auto_static_dynamic",
        "static_dynamic_threshold", "static_mode", "sh_progressive_schedule",
        "lambda_accel", "cam_grad_clip_norm", "mip_scale_floor_frac",
        "dynamic_densify_scale", "preload_to_ram", "holdout_indices",
        "cancel_check", "edit_mask_stack",
    }
)


def _apply_trainer_customizer(
    trainer: Trainer4DGS,
    customizer: Callable[[Trainer4DGS], None] | None,
) -> None:
    if customizer is not None:
        customizer(trainer)


def _validated_trainer_train_kwargs(
    values: dict[str, Any] | None,
) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, dict) or not all(isinstance(key, str) for key in values):
        raise ValueError("trainer_train_kwargs must be a string-keyed dict")
    collisions = sorted(set(values) & _EXPLICIT_PIPELINE_TRAIN_KWARGS)
    if collisions:
        raise ValueError(
            "trainer_train_kwargs collides with explicit pipeline arguments: "
            + ", ".join(collisions)
        )
    unknown = sorted(set(values) - _EXPERIMENT_TRAIN_KWARGS)
    if unknown:
        raise ValueError(
            "trainer_train_kwargs contains unknown experiment arguments: "
            + ", ".join(unknown)
        )
    return dict(values)


def _preserve_experiment_depth(values: dict[str, Any] | None) -> bool:
    return values is not None


def run_pipeline(
    video_path: str | Path,
    scene_name: str = "test_scene",
    cfg: Config | None = None,
    skip_foundation: bool = True,
    skip_training: bool = False,
    skip_export: bool = False,
    progress_callback: ProgressCallback | None = None,
    force_preprocess: bool = False,
    cancel_check: Callable[[], bool] | None = None,
    trainer_customizer: Callable[[Trainer4DGS], None] | None = None,
    trainer_train_kwargs: dict[str, Any] | None = None,
) -> dict:
    """
    Returns: { phase: durum }

    Args:
        progress_callback: (phase, progress_0_1, message, details) → None
            Her faz başlangıcında ve bitişinde, ayrıca training sırasında
            periyodik olarak çağrılır. None ise hiçbir şey yapmaz.
        force_preprocess: True ise tum cache'leri yoksay, butun preprocessing
            adimlari (frame extract, COLMAP, depth, tracks, masks, MVS) sifirdan
            kosulur. Default False — cache hit'lerde skip eder, training'e
            hizli baslar (Phase 1.1: heavy preprocess once, train many times).
    """
    if cfg is None:
        cfg = default_config()
    paths = scene_paths(scene_name)
    paths["base"].mkdir(parents=True, exist_ok=True)
    (paths["output"] / "logs").mkdir(parents=True, exist_ok=True)

    cb: ProgressCallback = progress_callback or _noop_cb
    status = {}

    # Static 3DGS Faz 1 — static_mode aktifse 4D dynamic loss'lari + motion regs
    # otomatik 0'lanir. Foundation modellerden DEPTH ise hala calisir (geometric
    # prior + sparse-view yardimcisi); tracks/masks/flow ise statik sahnede anlamsiz.
    # Bu nedenle skip_foundation FALSE birakilir (depth icin), trainer track/mask/flow
    # lambda'lari 0 oldugu icin onlari kullanmaz.
    experiment_training = _preserve_experiment_depth(trainer_train_kwargs)
    static_mode_active = bool(getattr(cfg.train, "static_mode", False))
    if static_mode_active:
        # 4D-only loss'larin lambda'larini sifirla. NOT: lambda_depth listede DEGIL —
        # static modda da geometric supervision olarak kullanilir.
        for _attr in ("lambda_mask_motion", "lambda_track",
                      "lambda_flow", "lambda_smoothness", "lambda_rigidity",
                      "lambda_deform_reg", "lambda_fourier_reg",
                      "lambda_multiview_consistency"):
            try:
                if getattr(cfg.train, _attr, 0.0) > 0:
                    setattr(cfg.train, _attr, 0.0)
            except Exception:
                pass
        # Phase 2.1 auto-promote gereksiz — statik sahnede dynamic ayrimi yok.
        try:
            cfg.train.auto_static_dynamic = False
        except Exception:
            pass
        # lambda_depth > 0 ise foundation depth gerekli — skip_foundation'i KAPATMA.
        # Eger user explicit skip_foundation=True dediyse, lambda_depth da 0'la
        # (depth dosyasi yok, trainer crash etmesin).
        wants_depth = float(getattr(cfg.train, "lambda_depth", 0.0)) > 0
        if skip_foundation and wants_depth and not experiment_training:
            print("[Static 3DGS] skip_foundation=True ama lambda_depth>0 → "
                  "lambda_depth=0 zorlandi (depth dosyasi yok)")
            cfg.train.lambda_depth = 0.0
        elif wants_depth and experiment_training:
            print(f"[Static 3DGS] validated experiment depth aktif "
                  f"(lambda_depth={cfg.train.lambda_depth})")
        elif wants_depth:
            print(f"[Static 3DGS] depth supervision aktif (lambda_depth="
                  f"{cfg.train.lambda_depth}) — foundation depth phase calisacak, "
                  f"tracks/masks/flow ise pipeline tarafinda atlanacak")

    # Run logger — tüm pipeline'ı takip eder, metrics/events/summary yazar
    run_logger = RunLogger(paths["output"] / "logs", scene=scene_name)
    run_logger.log_event(
        "config",
        skip_foundation=skip_foundation,
        skip_training=skip_training,
        skip_export=skip_export,
        video_path=str(video_path),
        static_mode=static_mode_active,
    )
    try:
        # Config'i de event olarak yaz (dataclass → dict)
        from dataclasses import asdict as _asdict
        run_logger.log_event("config_snapshot", **{
            "cfg": _asdict(cfg),
        })
    except Exception:
        pass

    # -------- v5.0: Multi-view scene detection --------
    is_mv = is_multiview_scene(scene_name)
    mv_ctx = None
    if is_mv:
        print(f"\n[v5.0] Multi-view scene: {scene_name}")
        run_logger.log_event("scene:multiview_detected", scene=scene_name)
        try:
            mv_ctx = prepare_multiview_scene(paths, cfg, on_progress=cb,
                                              force_preprocess=force_preprocess)
            print(f"  ✓ {mv_ctx['n_cameras']} cam, {len(mv_ctx['train_cams'])} train, "
                  f"primary={mv_ctx['primary_cam']}, test={mv_ctx['test_cam']}")
            status["frames"] = f"multi-view ok ({mv_ctx['n_cameras']} cams)"
            status["colmap"] = f"N3V calibration ({mv_ctx['n_cameras']} cams)"
        except Exception as e:
            print(f"  ✗ Multi-view prep failed: {e}, falling back to single-view")
            run_logger.log_event("scene:multiview_failed", error=str(e)[:200])
            is_mv = False
            mv_ctx = None

    # -------- Faz 2a: Frame çıkarma (single-view only) --------
    if not is_mv:
        images_dir = paths["base"] / "images"
        frames_dir = paths["frames"]
        has_photo_set = images_dir.exists() and any(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG"))
            + list(images_dir.glob("*.png")) + list(images_dir.glob("*.PNG"))
            + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.JPEG"))
        )

        print("\n[Faz 2a] Frame çıkarma")
        cb("frames", 0.0, "Video karelere ayrılıyor", {})
        # Phase 1.3: settings hash karsilastirmasi — fps/resize_long_edge degisirse re-extract
        frames_marker = read_cache_marker(paths["base"], "frames")
        cur_frames_hash = compute_settings_hash(cfg, "frames")
        existing_frames = (
            frames_dir.exists() and any(frames_dir.glob("frame_*.png"))
        )
        frames_settings_match = (
            frames_marker is not None
            and frames_marker.get("settings_hash") == cur_frames_hash
        )

        def _photo_set_to_frames():
            """images/ -> frames/ kopyala, resize_long_edge uygula, cache marker yaz."""
            print(f"  [Photo set adapter] images/ -> frames/ kopyalaniyor")
            frames_dir.mkdir(parents=True, exist_ok=True)
            # Eski frame'leri temizle (settings degistiyse stale olur)
            for old in frames_dir.glob("frame_*.png"):
                try: old.unlink()
                except Exception: pass
            from PIL import Image
            all_imgs = []
            for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG", "*.jpeg", "*.JPEG"):
                all_imgs.extend(images_dir.glob(ext))
            all_imgs = sorted(set(all_imgs))
            resize_le = int(cfg.preprocess.resize_long_edge or 0)
            for i, img_path in enumerate(all_imgs):
                try:
                    img = Image.open(img_path).convert("RGB")
                    if resize_le > 0:
                        w, h = img.size
                        if max(w, h) > resize_le:
                            if w >= h:
                                new_w, new_h = resize_le, int(h * resize_le / w)
                            else:
                                new_h, new_w = resize_le, int(w * resize_le / h)
                            img = img.resize((new_w, new_h), Image.LANCZOS)
                    img.save(frames_dir / f"frame_{i:06d}.png")
                except Exception as _e:
                    print(f"    ! {img_path.name} skip: {_e}")
            n_done = len(list(frames_dir.glob("frame_*.png")))
            print(f"  ok: {n_done} frame yazildi (resize_long_edge={resize_le})")
            write_cache_marker(paths["base"], "frames", {
                "n_frames": n_done,
                "fps": cfg.preprocess.fps,
                "resize_long_edge": resize_le,
                "source": "photo_set",
            }, cfg=cfg)
            return n_done

        if has_photo_set:
            # Photo set source-of-truth: video gerekli degil, settings degisiminde
            # adapter re-copy yapar.
            if (not force_preprocess) and existing_frames and frames_settings_match:
                n_frames_existing = len(list(frames_dir.glob("frame_*.png")))
                log_cache("frames", frames_dir, hit=True, count=n_frames_existing)
                status["frames"] = f"ok (cache hit, photo-set, {n_frames_existing} frames)"
            else:
                if existing_frames and not frames_settings_match and frames_marker is not None:
                    log_cache("frames", frames_dir, hit=False,
                              reason=f"settings_hash mismatch ({frames_marker.get('settings_hash')} → {cur_frames_hash})")
                try:
                    n_done = _photo_set_to_frames()
                    status["frames"] = f"photo_set ok ({n_done} frame)"
                except ImportError:
                    raise RuntimeError(
                        "Photo set adapter PIL gerektirir. pip install pillow"
                    )
        elif (not force_preprocess) and existing_frames and frames_settings_match:
            n_frames_existing = len(list(frames_dir.glob("frame_*.png")))
            log_cache("frames", frames_dir, hit=True, count=n_frames_existing)
            status["frames"] = f"ok (cache hit, {n_frames_existing} frames)"
        else:
            # Standart video extract — DEFENSIVE: video yoksa ve images/ varsa
            # photo-set adapter'a duser (sadece has_photo_set True iken yukarida
            # yakalanmali ama uvicorn reload'unda kod sirasi kacirilirsa burda
            # son savunma.)
            if existing_frames and not frames_settings_match and frames_marker is not None:
                log_cache("frames", frames_dir, hit=False,
                          reason=f"settings_hash mismatch ({frames_marker.get('settings_hash')} → {cur_frames_hash})")
            video_path_p = Path(video_path)
            if not video_path_p.exists():
                if has_photo_set:
                    print(f"  [Defensive] video yok ama images/ var → photo-set adapter")
                    try:
                        n_done = _photo_set_to_frames()
                        status["frames"] = f"photo_set defensive ok ({n_done} frame)"
                    except ImportError:
                        raise RuntimeError("Photo set adapter PIL gerektirir. pip install pillow")
                else:
                    raise FileNotFoundError(
                        f"Ne video.mp4 ne images/ klasoru bulundu:\n"
                        f"  video: {video_path}\n"
                        f"  images: {images_dir}\n"
                        f"Coz: data/{scene_name}/images/IMG_*.jpg KOY veya video.mp4 ekle."
                    )
            else:
                extract_frames(
                    video_path, frames_dir,
                    fps=cfg.preprocess.fps,
                    resize_long_edge=cfg.preprocess.resize_long_edge,
                    overwrite=True,
                )
                n_frames_done = len(list(frames_dir.glob("frame_*.png")))
                write_cache_marker(paths["base"], "frames", {
                    "n_frames": n_frames_done,
                    "fps": cfg.preprocess.fps,
                    "resize_long_edge": cfg.preprocess.resize_long_edge,
                }, cfg=cfg)
                status["frames"] = "ok"
        cb("frames", 1.0, "Kareler hazır", {})

    # -------- Faz 2b-c: COLMAP (cache'li + progress hook) --------
    if is_mv:
        # v5.0: Multi-view'da COLMAP atlanır, calibration N3V'den gelir
        print("\n[Faz 2b] COLMAP atlandı — multi-view N3V calibration kullanılıyor")
        cams = {}  # multi-view'de mv_ctx kullanılacak, single-view dict format gerekmez
        xyz = mv_ctx["init_xyz"]
        rgb = mv_ctx["init_rgb"]
        cb("colmap", 1.0, f"N3V calibration: {mv_ctx['n_cameras']} cams", {"multi_view": True})
    else:
      print("\n[Faz 2b] COLMAP SfM")
      cb("colmap", 0.0, "COLMAP başlıyor", {})
      # Cache check — eğer sparse zaten hazırsa tekrar koşma.
      # Phase 1.3: settings hash check — colmap_matching/init_subsample_mode degisirse rerun.
      # (cache_utils helpers module-level import edildi, lokal binding yok)
      colmap_marker = read_cache_marker(paths["base"], "colmap")
      cur_colmap_hash = compute_settings_hash(cfg, "colmap")
      colmap_settings_match = (
          colmap_marker is not None
          and colmap_marker.get("settings_hash") == cur_colmap_hash
      )
      # 4D Quality v6.1 — Madde 3: DUSt3R sparse-view init opt-in
      init_method = getattr(cfg.preprocess, "init_method", "colmap")
      sparse_threshold = int(getattr(cfg.preprocess, "sparse_view_threshold_frames", 20))
      n_frames_for_init = len(list(paths["frames"].glob("frame_*.png"))) if paths["frames"].exists() else 0
      want_dust3r = (
          init_method == "dust3r"
          or (init_method == "auto" and 0 < n_frames_for_init < sparse_threshold)
      )
      dust3r_done = False
      if want_dust3r:
          try:
              from .preprocess.dust3r_init import (
                  estimate_sparse_init, is_dust3r_available, install_hint,
              )
              if not is_dust3r_available():
                  print(f"[pipeline] DUSt3R yok, COLMAP'a duser:\n  {install_hint()}")
              else:
                  frame_paths_init = sorted(paths["frames"].glob("frame_*.png"))
                  print(f"[pipeline] DUSt3R sparse init ({n_frames_for_init} frame, "
                        f"method={init_method})")
                  d3r = estimate_sparse_init(
                      frame_paths_init, paths["base"] / "dust3r_init",
                      device="cuda" if torch.cuda.is_available() else "cpu",
                  )
                  cams = d3r["cameras"]
                  for name in cams:
                      if "width" not in cams[name]:
                          cams[name]["width"] = int(cfg.preprocess.resize_long_edge or 960)
                          cams[name]["height"] = int(cams[name]["width"] * 9 / 16)
                  xyz = d3r["xyz"]
                  rgb = d3r["rgb"]
                  print(f"  ✓ DUSt3R: {len(cams)} cam, {len(xyz)} point")
                  cb("colmap", 1.0, f"DUSt3R init: {len(cams)} cam", {"dust3r": True})
                  status["colmap"] = f"DUSt3R: {len(cams)} cam, {len(xyz)} point"
                  dust3r_done = True
          except Exception as e:
              import traceback
              traceback.print_exc()
              print(f"[pipeline] DUSt3R basarisiz, COLMAP'a duser: {e}")

      if dust3r_done:
        # COLMAP fazini atla — cams/xyz/rgb DUSt3R'dan dolduruldu
        pass
      else:
       try:
        if force_preprocess or not colmap_settings_match:
            # Cache invalidate — eski parse fail-fast yerine direk RuntimeError trigger
            if colmap_marker is not None and not colmap_settings_match:
                print(f"[pipeline] COLMAP settings degisti "
                      f"(cache={colmap_marker.get('settings_hash')} vs current={cur_colmap_hash}) "
                      f"→ rerun")
            raise RuntimeError("force/settings invalidation")
        cams = parse_cameras(paths["colmap"])
        xyz, rgb = load_points3d(paths["colmap"])
        print(f"✓ COLMAP cache hit: {len(cams)} kamera, {len(xyz)} nokta (rerun atlandı)")
        cb("colmap", 1.0, f"cache hit: {len(cams)} kamera", {"cache": True})
       except (FileNotFoundError, RuntimeError):
        # COLMAP stream progress → pipeline callback'e relay
        def _colmap_on_progress(frac: float, msg: str) -> None:
            cb("colmap", frac, msg, {"colmap_fraction": frac})

        # v3.9: COLMAP matching strategy config'ten okunur
        is_sequential = getattr(cfg.preprocess, "colmap_matching", "sequential") != "exhaustive"
        print(f"[pipeline] COLMAP matching: {getattr(cfg.preprocess, 'colmap_matching', 'sequential')} (sequential={is_sequential})")
        run_colmap(
            paths["frames"], paths["colmap"],
            camera_model=cfg.preprocess.colmap_camera_model,
            use_gpu=cfg.preprocess.colmap_use_gpu,
            sequential=is_sequential,
            sequential_overlap=cfg.preprocess.sequential_overlap,
            colmap_exe=cfg.preprocess.colmap_exe,
            on_progress=_colmap_on_progress,
        )
        cams = parse_cameras(paths["colmap"])
        xyz, rgb = load_points3d(paths["colmap"])
        # Phase 1.3: settings_hash marker yaz, bir sonraki run cache hit alabilsin
        try:
            write_cache_marker(paths["base"], "colmap", {
                "n_cameras": len(cams),
                "n_points": int(len(xyz)),
                "matching": "sequential" if is_sequential else "exhaustive",
                "camera_model": cfg.preprocess.colmap_camera_model,
            }, cfg=cfg)
        except Exception as _e:
            print(f"[pipeline] ⚠ COLMAP cache marker yazilirken hata: {_e}")
    if not is_mv:
        status["colmap"] = f"{len(cams)} kamera, {len(xyz)} nokta"
        cb("colmap", 1.0, f"{len(cams)} kamera, {len(xyz)} 3B nokta",
           {"cameras": len(cams), "points": len(xyz)})

        # --native-res: train/eval cozunurlugunu COLMAP'in gordugu kaynak frame
        # boyutuna cek (photo-set protokol paritesi, orn. Mip-NeRF360 images_4).
        # Preset'in sabit cozunurlugu kaynaktan buyukse GT upsample + aspect
        # stretch olur; baseline'larla karsilastirilan metrikler bozulur.
        if getattr(cfg.train, "native_resolution", False) and cams:
            _cam0 = cams[sorted(cams.keys())[0]]
            _nw, _nh = int(_cam0.get("width", 0)), int(_cam0.get("height", 0))
            if _nw > 0 and _nh > 0:
                if (_nw, _nh) != tuple(cfg.train.image_resolution):
                    print(f"[pipeline] native-res: image_resolution "
                          f"{tuple(cfg.train.image_resolution)} -> ({_nw}, {_nh})")
                    cfg.train.image_resolution = (_nw, _nh)
                _long_native = max(_nw, _nh)
                _sched = [tuple(s) for s in (getattr(cfg.train, "multires_schedule", []) or [])]
                if _sched:
                    _clamped: list[tuple[int, int]] = []
                    for _it, _edge in _sched:
                        _e = min(int(_edge), _long_native)
                        if _clamped and _e <= _clamped[-1][1]:
                            continue  # clamp sonrasi non-increasing step'leri at
                        _clamped.append((int(_it), _e))
                    if _clamped != _sched:
                        print(f"[pipeline] native-res: multires_schedule "
                              f"{_sched} -> {_clamped}")
                        cfg.train.multires_schedule = _clamped

    # -------- Faz 3: Foundation modeller (opsiyonel, resilient) --------
    # Phase 1.4: Multi-view'da per-cam depth (Metric3D/MiDaS) calisiyor.
    # Tracks/masks per-cam Phase 1.5'te aktif olacak (su an MV'de sadece depth).
    if not skip_foundation and is_mv:
        foundation_status_mv = {"depth_mv": "pending", "masks_mv": "pending", "flow_mv": "skip"}
        print("\n[Faz 3-MV] Multi-view per-cam foundation (depth + masks + flow)")
        cb("foundation", 0.0, "Per-cam depth (multi-view)", {})

        # 3a-MV — Depth per cam
        # Beklenen min count: cam basina (frames - 5) tolerans
        try:
            sample_cam = next(iter(mv_ctx["frame_paths_per_cam"].values()))
            n_frames_per_cam = len(sample_cam)
        except Exception:
            n_frames_per_cam = 0
        n_cams_mv = len(mv_ctx.get("frame_paths_per_cam", {}))
        # Multi-view glob: "cam*/*_depth.npy" — toplam asgari count
        min_total = max(1, n_cams_mv * (n_frames_per_cam - 5))
        depth_mv_cached = (
            not force_preprocess
            and is_step_cached(paths, "depth_mv", min_count=min_total,
                               cfg=cfg, scene_dir=paths["base"])
        )
        if depth_mv_cached:
            n_depth_total = len(list(paths["depth_mv"].glob("cam*/*_depth.npy")))
            log_cache("depth_mv", paths["depth_mv"], hit=True, count=n_depth_total)
            foundation_status_mv["depth_mv"] = f"ok (cache hit, {n_depth_total} files)"
            run_logger.log_event("depth_mv:cache_hit", n_files=n_depth_total)
        else:
            log_cache("depth_mv", paths["depth_mv"], hit=False)
            try:
                from .preprocess.depth_multiview import estimate_depth_multiview
                def _depth_mv_cb(frac, msg):
                    cb("foundation", 0.0 + 0.5 * frac, msg, {"depth_mv_frac": frac})
                estimate_depth_multiview(
                    paths["frames_mv"],
                    paths["depth_mv"],
                    model_name=cfg.foundation.metric3d_model,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    overwrite=False,
                    on_progress=_depth_mv_cb,
                )
                n_depth_total = len(list(paths["depth_mv"].glob("cam*/*_depth.npy")))
                foundation_status_mv["depth_mv"] = f"ok ({n_depth_total} files)"
                write_cache_marker(paths["base"], "depth_mv", {
                    "model": cfg.foundation.metric3d_model,
                    "n_cams": n_cams_mv,
                    "n_frames_per_cam": n_frames_per_cam,
                    "n_depth_total": n_depth_total,
                }, cfg=cfg)
                # VRAM cleanup — sonraki phase'lerde tracks/masks gelecek
                try:
                    from .preprocess.depth_estimate import release_models as _rd
                    _rd()
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"⚠ Multi-view depth basarisiz, atlanir: {e}")
                foundation_status_mv["depth_mv"] = f"failed: {e}"

        # 3b-MV — Masks per cam (Phase 1.5)
        masks_mv_cached = (
            not force_preprocess
            and is_step_cached(paths, "masks_mv",
                               min_count=max(1, n_cams_mv * (n_frames_per_cam - 5)),
                               cfg=cfg, scene_dir=paths["base"])
        )
        if masks_mv_cached:
            n_masks_total = len(list(paths["masks_mv"].glob("cam*/mask_*.png")))
            log_cache("masks_mv", paths["masks_mv"], hit=True, count=n_masks_total)
            foundation_status_mv["masks_mv"] = f"ok (cache hit, {n_masks_total} files)"
            run_logger.log_event("masks_mv:cache_hit", n_files=n_masks_total)
        else:
            log_cache("masks_mv", paths["masks_mv"], hit=False)
            try:
                from .preprocess.dynamic_mask_multiview import compute_dynamic_masks_multiview
                def _masks_mv_cb(frac, msg):
                    cb("foundation", 0.5 + 0.5 * frac, msg, {"masks_mv_frac": frac})
                compute_dynamic_masks_multiview(
                    paths["frames_mv"], paths["masks_mv"],
                    overwrite=False, on_progress=_masks_mv_cb,
                )
                n_masks_total = len(list(paths["masks_mv"].glob("cam*/mask_*.png")))
                foundation_status_mv["masks_mv"] = f"ok ({n_masks_total} files)"
                write_cache_marker(paths["base"], "masks_mv", {
                    "n_cams": n_cams_mv,
                    "n_frames_per_cam": n_frames_per_cam,
                    "n_masks_total": n_masks_total,
                }, cfg=cfg)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"⚠ Multi-view mask basarisiz, atlanir: {e}")
                foundation_status_mv["masks_mv"] = f"failed: {e}"

        # 3c-MV — Optical flow (Phase 1.8). Sadece training trainer.lambda_flow > 0
        # ise gerekli. Cache'lenir, training tarafi opsiyonel consume eder.
        if getattr(cfg.train, "lambda_flow", 0.0) > 0:
            flow_mv_cached = (
                not force_preprocess
                and is_step_cached(paths, "flow_mv",
                                   min_count=max(1, n_cams_mv * (n_frames_per_cam - 6)),
                                   cfg=cfg, scene_dir=paths["base"])
            )
            if flow_mv_cached:
                n_flow_total = len(list(paths["flow_mv"].glob("cam*/forward_*.pt")))
                log_cache("flow_mv", paths["flow_mv"], hit=True, count=n_flow_total)
                foundation_status_mv["flow_mv"] = f"ok (cache hit, {n_flow_total})"
            else:
                log_cache("flow_mv", paths["flow_mv"], hit=False)
                try:
                    from .preprocess.optical_flow import (
                        estimate_flow_multiview, release_models as _rf,
                    )
                    estimate_flow_multiview(
                        paths["frames_mv"], paths["flow_mv"],
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        overwrite=False,
                    )
                    n_flow_total = len(list(paths["flow_mv"].glob("cam*/forward_*.pt")))
                    foundation_status_mv["flow_mv"] = f"ok ({n_flow_total} files)"
                    write_cache_marker(paths["base"], "flow_mv", {
                        "n_cams": n_cams_mv,
                        "n_frames_per_cam": n_frames_per_cam,
                        "n_flow_total": n_flow_total,
                    }, cfg=cfg)
                    _rf()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"⚠ Multi-view flow basarisiz, atlanir: {e}")
                    foundation_status_mv["flow_mv"] = f"failed: {e}"
        else:
            foundation_status_mv["flow_mv"] = "skip (lambda_flow=0)"

        status["foundation_mv"] = ", ".join(f"{k}={v}" for k, v in foundation_status_mv.items())
        cb("foundation", 1.0, f"MV foundation: {status['foundation_mv']}", foundation_status_mv)

    if not skip_foundation and not is_mv:
        foundation_status = {"depth": "pending", "tracks": "pending", "masks": "pending"}

        # 3a — Depth (Phase 1.1: cache check — sahnede yeterli depth varsa skip)
        print("\n[Faz 3a] MiDaS derinlik tahmini")
        cb("foundation", 0.0, "Derinlik tahmini", {})
        # Beklenen depth count = frame count (her PNG icin bir depth)
        n_frames = len(list(paths["frames"].glob("frame_*.png"))) if paths["frames"].exists() else 0
        depth_cached = (
            not force_preprocess
            and is_step_cached(paths, "depth", min_count=max(1, n_frames - 5),
                               cfg=cfg, scene_dir=paths["base"])
        )
        if depth_cached:
            n_depth = len(list(paths["depth"].glob("*_depth.npy")))
            log_cache("depth", paths["depth"], hit=True, count=n_depth)
            foundation_status["depth"] = f"ok (cache hit, {n_depth} files)"
            run_logger.log_event("depth:cache_hit", n_files=n_depth)
        else:
            log_cache("depth", paths["depth"], hit=False)
            try:
                from .preprocess.depth_estimate import estimate_depth
                estimate_depth(paths["frames"], paths["depth"],
                               model_name=cfg.foundation.metric3d_model)
                foundation_status["depth"] = "ok"
                write_cache_marker(paths["base"], "depth", {
                    "model": cfg.foundation.metric3d_model,
                    "n_frames": n_frames,
                }, cfg=cfg)
            except Exception as e:
                print(f"⚠ Derinlik tahmini başarısız, atlanıyor: {e}")
                foundation_status["depth"] = f"failed: {e}"

        # 3a.5 — MiDaS → COLMAP scale alignment (v3.7 / Option B)
        # MiDaS relative depth üretiyor, COLMAP world scale ile uyumsuz.
        # Anchor unprojection doğru 3D koordinat üretebilsin diye align ediyoruz.
        # Track loss'un 0.15'te takılmasının ana sebebi bu uyumsuzluktu.
        # Phase 1.1: depth_align cache check — marker varsa skip.
        run_logger.log_event("phase:start", phase="depth_align")
        align_cached = (
            not force_preprocess
            and (paths["base"] / ".cache_markers" / "depth_align.json").exists()
        )
        if align_cached:
            log_cache("depth_align", paths["depth"], hit=True)
            run_logger.log_event("depth_align:cache_hit")
            foundation_status["depth"] = (foundation_status.get("depth") or "ok") + " (aligned, cached)"
        elif foundation_status["depth"].startswith("ok"):
            try:
                print("\n[Faz 3a.5] Depth → COLMAP scale alignment")
                from .preprocess.align_depth import align_depth_to_colmap
                # Kamera, frame ve depth ilişkisini exact image name ile kur.
                registered_frames = join_registered_frames(
                    paths["frames"], cams, depth_dir=paths["depth"]
                )
                first_registered = registered_frames[0]
                first_cam = cams[first_registered.image_name]
                frame_paths_all = [frame.image_path for frame in registered_frames]
                w2c_list = [
                    torch.from_numpy(frame.w2c).float()
                    for frame in registered_frames
                ]
                K_first = torch.from_numpy(first_registered.K).float()
                W_frame = int(first_cam["width"])
                H_frame = int(first_cam["height"])
                xyz_t = torch.from_numpy(xyz).float()

                align_stats = align_depth_to_colmap(
                    depth_dir=paths["depth"],
                    frame_paths=frame_paths_all,
                    cam_K=K_first,
                    cam_w2c_per_frame=w2c_list,
                    colmap_xyz=xyz_t,
                    frame_size=(W_frame, H_frame),
                    overwrite=True,
                )
                foundation_status["depth"] = f"ok (COLMAP-aligned, scale={align_stats['global_scale_median']:.3f})"
                run_logger.log_event(
                    "depth_align:ok",
                    n_aligned=align_stats["n_frames_aligned"],
                    n_skipped=align_stats["n_frames_skipped"],
                    global_scale=align_stats["global_scale_median"],
                )
                write_cache_marker(paths["base"], "depth_align", {
                    "n_aligned": align_stats["n_frames_aligned"],
                    "global_scale": align_stats["global_scale_median"],
                })
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"⚠ Depth align başarısız (kritik değil, training devam): {e}")
                print(tb)
                run_logger.log_event("depth_align:failed", error=str(e)[:200])
                # Alignment opsiyonel, pipeline devam eder
        else:
            run_logger.log_event("depth_align:skipped",
                                 reason=f"depth status={foundation_status['depth']}")

        cb("foundation", 0.33, "Depth done, CoTracker başlıyor", {})

        # Static 3DGS Faz 1 — tracks/masks/flow statik sahnede anlamsiz.
        # Sadece depth (yukarida) hesaplandi, gerisi atlanir.
        if static_mode_active:
            foundation_status["tracks"] = "skip (static_mode)"
            foundation_status["masks"]  = "skip (static_mode)"
            foundation_status["flow"]   = "skip (static_mode)"
            n_frames_for_mask = len(list(paths["frames"].glob("frame_*.png"))) if paths["frames"].exists() else 0
            tracks_path = paths["tracks"] / "tracks.pt"
            try:
                from .preprocess.depth_estimate import release_models as _rd
                _rd()
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            cb("foundation", 0.66, "Tracks/Masks atlandi (static_mode)", {"static": True})
            print("[Faz 3] Static modda tracks/masks/flow atlandi, depth aktif")
        else:
            print("\n[Faz 3b] CoTracker piksel takibi")
            tracks_path = paths["tracks"] / "tracks.pt"
            # Phase 1.3: settings hash da kontrol et (cotracker_grid_size degisirse miss)
            tracks_cached = (
                not force_preprocess
                and tracks_path.exists()
                and tracks_path.stat().st_size > 1000  # not corrupt
                and is_step_cached(paths, "tracks", cfg=cfg, scene_dir=paths["base"])
            )
            if tracks_cached:
                log_cache("tracks", tracks_path, hit=True)
                foundation_status["tracks"] = f"ok (cache hit, {tracks_path.stat().st_size // 1024} KB)"
                run_logger.log_event("tracks:cache_hit",
                                     size_bytes=tracks_path.stat().st_size)
            else:
                log_cache("tracks", tracks_path, hit=False)
                try:
                    from .preprocess.depth_estimate import release_models as release_depth_models
                    release_depth_models()
                except Exception as _e:
                    print(f"  ⚠ depth model release: {_e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                try:
                    from .preprocess.point_tracking import track_points
                    track_points(paths["frames"], tracks_path,
                                 grid_size=cfg.foundation.cotracker_grid_size)
                    foundation_status["tracks"] = "ok"
                    write_cache_marker(paths["base"], "tracks", {
                        "grid_size": cfg.foundation.cotracker_grid_size,
                        "size_bytes": tracks_path.stat().st_size if tracks_path.exists() else 0,
                    }, cfg=cfg)
                except Exception as e:
                    print(f"⚠ Tracking başarısız, atlanıyor: {e}")
                    foundation_status["tracks"] = f"failed: {e}"
            # Cleanup after CoTracker too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            cb("foundation", 0.66, "Tracks done, dinamik maske başlıyor", {})

            # 3c — Dynamic mask (Phase 1.1: cache check)
            print("\n[Faz 3c] Dinamik maske")
            n_frames_for_mask = len(list(paths["frames"].glob("frame_*.png"))) if paths["frames"].exists() else 0
            masks_cached = (
                not force_preprocess
                and is_step_cached(paths, "masks", min_count=max(1, n_frames_for_mask - 5),
                                   cfg=cfg, scene_dir=paths["base"])
            )
            if masks_cached:
                n_masks = len(list(paths["masks"].glob("*.png")))
                log_cache("masks", paths["masks"], hit=True, count=n_masks)
                foundation_status["masks"] = f"ok (cache hit, {n_masks} files)"
                run_logger.log_event("masks:cache_hit", n_files=n_masks)
            else:
                log_cache("masks", paths["masks"], hit=False)
                try:
                    from .preprocess.dynamic_mask import compute_dynamic_masks
                    compute_dynamic_masks(paths["frames"], paths["masks"])
                    foundation_status["masks"] = "ok"
                    write_cache_marker(paths["base"], "masks", {
                        "n_frames": n_frames_for_mask,
                    }, cfg=cfg)
                except Exception as e:
                    print(f"⚠ Maske başarısız, atlanıyor: {e}")
                    foundation_status["masks"] = f"failed: {e}"

        # 3d — Phase 1.8 SV: RAFT optical flow (lambda_flow>0 ise preprocess)
        # Static modda zaten lambda_flow=0 zorlandi, yine de explicit guard ekle.
        if not static_mode_active and getattr(cfg.train, "lambda_flow", 0.0) > 0:
            flow_cached_sv = (
                not force_preprocess
                and is_step_cached(paths, "flow",
                                   min_count=max(1, n_frames_for_mask - 6),
                                   cfg=cfg, scene_dir=paths["base"])
            )
            if flow_cached_sv:
                n_flow = len(list(paths["flow"].glob("forward_*.pt")))
                log_cache("flow", paths["flow"], hit=True, count=n_flow)
                foundation_status["flow"] = f"ok (cache hit, {n_flow})"
            else:
                log_cache("flow", paths["flow"], hit=False)
                try:
                    from .preprocess.optical_flow import (
                        estimate_flow, release_models as _rf,
                    )
                    estimate_flow(
                        paths["frames"], paths["flow"],
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        overwrite=False,
                    )
                    n_flow = len(list(paths["flow"].glob("forward_*.pt")))
                    foundation_status["flow"] = f"ok ({n_flow})"
                    write_cache_marker(paths["base"], "flow", {
                        "n_frames": n_frames_for_mask,
                        "n_flow_total": n_flow,
                    }, cfg=cfg)
                    _rf()
                except Exception as e:
                    print(f"⚠ SV flow basarisiz, atlanir: {e}")
                    foundation_status["flow"] = f"failed: {e}"
        else:
            foundation_status["flow"] = "skip (lambda_flow=0)"

        status["foundation"] = ", ".join(f"{k}={v}" for k, v in foundation_status.items())
        cb("foundation", 1.0, f"Foundation bitti: {status['foundation']}", foundation_status)
    else:
        # The else here covers two cases: (1) skip_foundation=True (user opted out)
        # OR (2) is_mv=True (multi-view scene; SV foundation not applicable, but MV
        # foundation already ran above). Distinguish them in the log so users
        # don't think MV foundation was skipped.
        if skip_foundation:
            print("\n[Faz 3] Foundation modeller atlandı (skip_foundation=True)")
        else:
            print("\n[Faz 3] Single-view foundation atlandı (multi-view scene; MV foundation completed above)")
        status["foundation"] = "skipped"
        cb("foundation", 1.0, "Atlandı", {"skipped": True})

    # -------- Faz 4-5: Model + Training --------
    if skip_training:
        print("\n[Faz 4-5] Training atlandı (--skip-training)")
        status["training"] = "skipped"
        return status

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠ CUDA bulunamadı — training çok yavaş olacak (test amaçlı)")

    print("\n[Faz 4] Gaussian model + DeformationField")
    cb("init", 0.0, "Model başlatılıyor", {})
    if is_mv:
        # v5.0: Multi-view'da init points zaten mv_ctx'ten gelir (random in bbox).
        # .float() = float32, gsplat zorunlu (float64 → "expected Float but found Double").
        init_pts = torch.from_numpy(xyz).float()
        init_rgb = torch.from_numpy(rgb).float() / 255.0  # zaten 0-255 değil 0-1 ama uyumluluk için
        if init_rgb.max() > 1.5:
            init_rgb = init_rgb / 255.0
    else:
        init_pts = torch.from_numpy(xyz).float()
        init_rgb = torch.from_numpy(rgb).float() / 255.0

    # v3.7.3 -> v3.9: Initial point subsample.
    # Mode:
    #   - "random" (default): cap'in %70'ine random downsample (eski davranis)
    #   - "confidence": COLMAP track length / (1 + reproj_error) skoruyla
    #     top-K se?, outlier'lari at. Static quality icin onerilen.
    if cfg.train.max_gaussians > 0 and not is_mv:
        target_init = int(cfg.train.max_gaussians * 0.7)
        if init_pts.shape[0] > target_init:
            mode = getattr(cfg.preprocess, "init_subsample_mode", "random")
            print(f"⚠ Initial COLMAP points ({init_pts.shape[0]:,}) > target "
                  f"({target_init:,}) — mode={mode}")
            if mode == "confidence":
                try:
                    from .preprocess.parse_colmap import load_points3d_with_confidence
                    _xyz, _rgb, track_len, reproj_err = load_points3d_with_confidence(paths["colmap"])
                    if track_len.shape[0] != init_pts.shape[0]:
                        print(f"  ⚠ confidence shape mismatch ({track_len.shape[0]} vs {init_pts.shape[0]}), random fallback")
                        perm = torch.randperm(init_pts.shape[0])[:target_init]
                    else:
                        confidence = track_len.astype(np.float32) / (1.0 + reproj_err)
                        top_idx = np.argsort(-confidence)[:target_init]
                        perm = torch.from_numpy(top_idx.astype("int64"))
                        median_kept = float(np.median(track_len[top_idx]))
                        median_err  = float(np.median(reproj_err[top_idx]))
                        print(f"  ✓ confidence subsample: track_len median={median_kept:.0f}, err median={median_err:.2f}")
                except Exception as e:
                    print(f"  ⚠ confidence subsample fail ({e}), random fallback")
                    perm = torch.randperm(init_pts.shape[0])[:target_init]
            else:
                perm = torch.randperm(init_pts.shape[0])[:target_init]
            init_pts = init_pts[perm]
            init_rgb = init_rgb[perm]
            run_logger.log_event("init_subsample",
                                 before=int(xyz.shape[0]),
                                 after=int(init_pts.shape[0]),
                                 max_gaussians=cfg.train.max_gaussians,
                                 mode=mode)

    # Static 3DGS — no temporal Fourier trajectory; force fourier_K=0 so the
    # per-gaussian (N, K, 2, 3) coeffs and their Adam state are never allocated
    # (frees real GPU memory on 8 GB cards; these params had LR=0 anyway).
    # deform_pos_mode da birlikte 'mlp' olur — trainer init'i fourier_K=0 ile
    # 'hybrid'i reddediyor (Colab P2-V 2026-07-03 crash'i).
    _fourier_K, _deform_pos_mode = _static_trainer_overrides(cfg)
    gs = GaussianModel(init_pts, init_colors=init_rgb,
                       sh_degree=cfg.model.sh_degree,
                       fourier_K=_fourier_K)
    # Static 3DGS Faz 1 — DeformationField construct skip if static_mode
    if getattr(cfg.train, "static_mode", False):
        print(f"  [Static 3DGS] DeformationField construct atlandi (static_mode=True)")
        # Hala None geçirmemek için minimal placeholder — _apply_deformation
        # static_mode kontrolüyle bypass eder, ama ckpt save'de gerekli
        deform = DeformationField(
            resolution=8, feat_dim=4, mlp_width=32, mlp_depth=1,
            num_time_freqs=2,
        )
        # Tüm deformation parametrelerini gradient'siz yap (LR=0 zaten ama kesin)
        for p in deform.parameters():
            p.requires_grad_(False)
    else:
        deform = DeformationField(
            resolution=cfg.model.hexplane_resolution,
            feat_dim=cfg.model.hexplane_feat_dim,
            mlp_width=cfg.model.mlp_width,
            mlp_depth=cfg.model.mlp_depth,
            num_time_freqs=cfg.model.num_time_freqs,
            # Phase 2.5 — opt-in multi-res HexPlane
            multires_resolutions=getattr(cfg.model, "multires_resolutions", None) or None,
            multires_feat_dim=getattr(cfg.model, "multires_feat_dim", None),
        )
    if is_mv:
        extent = float(mv_ctx["scene_extent"])
    else:
        extent = compute_scene_extent(xyz)
    print(f"  Sahne kapsamı: {extent:.3f}, başlangıç Gaussian: {gs.num_points:,}")
    cb("init", 1.0, f"Başlangıç Gaussian: {gs.num_points:,}",
       {"num_points": gs.num_points, "scene_extent": float(extent)})

    # Phase 2.4 — Background auto-flag (distant gauss bypass deformation).
    # Scene center'dan 'ratio*extent' uzakta olan gauss'lar BG flag'lenir,
    # trainer render'da bunlar deformation almaz (floater azaltma).
    bg_ratio = float(getattr(cfg.train, "bg_distance_ratio", 2.0))
    if bg_ratio > 0 and hasattr(gs, "flag_background_by_distance"):
        if is_mv:
            scene_center_arr = mv_ctx.get("scene_center", None)
            if scene_center_arr is None:
                scene_center_arr = xyz.mean(axis=0)
        else:
            scene_center_arr = xyz.mean(axis=0)
        scene_center_t = torch.from_numpy(np.asarray(scene_center_arr)).float()
        n_bg = gs.flag_background_by_distance(
            scene_center=scene_center_t,
            scene_extent=extent,
            ratio=bg_ratio,
        )
        print(f"  [Phase 2.4] Background auto-flag: {n_bg:,} gauss "
              f"({100 * n_bg / max(gs.num_points, 1):.1f}%) "
              f"distant > {bg_ratio * extent:.2f} units")

    # Frame yolları + kamera pozları
    if is_mv:
        # v5.0: Multi-view — primary cam frame_paths fallback (single-view path için)
        # Ama trainer'a aslında mv_* args geçeceğiz
        frame_paths = mv_ctx["primary_frame_paths"]
        K_first = mv_ctx["primary_K"]
        w2c_list = mv_ctx["primary_w2c_per_frame"]
        print(f"  Multi-view: {len(mv_ctx['train_cams'])} train cam x {len(frame_paths)} frame")
    else:
        # Fiziksel frame ve COLMAP kamerasını exact image name ile eşleştir.
        registered_frames = join_registered_frames(paths["frames"], cams)
        frame_paths = [frame.image_path for frame in registered_frames]
        w2c_list = [
            torch.from_numpy(frame.w2c).float() for frame in registered_frames
        ]
        K_first = torch.from_numpy(registered_frames[0].K).float()
        print(f"  Eğitim için {len(frame_paths)} frame eşleşti")

    print("\n[Faz 5] Training loop")
    trainer = Trainer4DGS(
        gs, deform,
        device=device,
        scene_extent=extent,
        lr_means=cfg.train.lr_means,
        lr_scales=cfg.train.lr_scales,
        lr_quats=cfg.train.lr_quats,
        lr_opacities=cfg.train.lr_opacities,
        lr_sh_dc=cfg.train.lr_sh_dc,
        lr_sh_rest=cfg.train.lr_sh_rest,
        lr_deform=cfg.train.lr_deform,
        density_start_iter=cfg.train.density_start_iter,
        density_end_iter=cfg.train.density_end_iter,
        density_interval=cfg.train.density_interval,
        densify_grad_threshold=cfg.train.densify_grad_threshold,
        prune_min_opacity=cfg.train.prune_min_opacity,
        prune_max_scale=cfg.train.prune_max_scale,
        lambda_ssim=cfg.train.lambda_ssim,
        lambda_deform_reg=cfg.train.lambda_deform_reg,
        lambda_smoothness=cfg.train.lambda_smoothness,
        lambda_rigidity=cfg.train.lambda_rigidity,
        lambda_depth=cfg.train.lambda_depth,
        lambda_mask_motion=cfg.train.lambda_mask_motion,
        lambda_track=cfg.train.lambda_track,
        track_sample_k=cfg.train.track_sample_k,
        lambda_scale=cfg.train.lambda_scale,
        # v3.8 — anisotropy + tightened dpos clamp
        lambda_aniso=cfg.train.lambda_aniso,
        aniso_threshold=cfg.train.aniso_threshold,
        dpos_total_cap_frac=cfg.train.dpos_total_cap_frac,
        opacity_reset_interval=cfg.train.opacity_reset_interval,
        warmup_iters=cfg.train.warmup_iters,
        # v3.6 / Yol C — Per-gaussian Fourier trajectory
        deform_pos_mode=_deform_pos_mode,
        lr_fourier=cfg.train.lr_fourier,
        lambda_fourier_reg=cfg.train.lambda_fourier_reg,
        # v3.7.2 — N hard cap
        max_gaussians=cfg.train.max_gaussians,
        # Phase 1.6/1.7/1.8/1.9 — perceptual + MV consistency + flow + densify scale
        lambda_lpips=getattr(cfg.train, "lambda_lpips", 0.0),
        lpips_net=getattr(cfg.train, "lpips_net", "alex"),
        lpips_warmup_iters=getattr(cfg.train, "lpips_warmup_iters", 1000),
        lambda_multiview_consistency=getattr(cfg.train, "lambda_multiview_consistency", 0.0),
        lambda_flow=getattr(cfg.train, "lambda_flow", 0.0),
        flow_warmup_iters=getattr(cfg.train, "flow_warmup_iters", 1000),
        densify_mv_threshold_scale=getattr(cfg.train, "densify_mv_threshold_scale", 0.7),
        # Phase 2.2 — Multi-resolution schedule
        multires_schedule=getattr(cfg.train, "multires_schedule", None),
        # Phase 2.3 — Camera pose refinement
        lr_cam_K=getattr(cfg.train, "lr_cam_K", 0.0),
        lr_cam_w2c=getattr(cfg.train, "lr_cam_w2c", 0.0),
        cam_refine_start_iter=getattr(cfg.train, "cam_refine_start_iter", 5000),
    )
    _apply_trainer_customizer(trainer, trainer_customizer)
    # Foundation çıktıları varsa trainer'a ver (stage 2 loss'lar için)
    depth_dir_arg = (
        paths["depth"]
        if ((not skip_foundation or experiment_training) and paths["depth"].exists())
        else None
    )
    mask_dir_arg  = paths["masks"] if (not skip_foundation and paths["masks"].exists()) else None
    tracks_file = paths["tracks"] / "tracks.pt"
    tracks_path_arg = tracks_file if (not skip_foundation and tracks_file.exists()) else None
    # Trainer ilerlemesini pipeline callback'ine relay eden köprü:
    # trainer iç ilerlemesini (0-1 arası) "training" fazına map ederiz.
    def _train_progress(iter_idx: int, total: int, loss: float,
                        psnr_val: float, n_pts: int) -> None:
        cb(
            "training",
            iter_idx / max(total, 1),
            f"iter {iter_idx}/{total} — loss={loss:.4f} psnr={psnr_val:.2f} N={n_pts:,}",
            {
                "iter": iter_idx, "total": total,
                "loss": loss, "psnr": psnr_val, "num_points": n_pts,
            },
        )

    cb("training", 0.0, "Training başlıyor", {"total_iters": cfg.train.n_iters})
    run_logger.phase_start("training")
    # v5.0: Multi-view args hazirla
    mv_frame_paths_arg = None
    mv_cam_K_arg = None
    mv_w2c_arg = None
    mv_test_camera_arg = None
    if is_mv and mv_ctx is not None:
        mv_frame_paths_arg = mv_ctx["frame_paths_per_cam"]
        mv_cam_K_arg = {c: torch.from_numpy(np.array(mv_ctx["calibration"][c]["K"])).float()
                        for c in mv_ctx["calibration"]}
        mv_w2c_arg = {c: torch.from_numpy(np.array(mv_ctx["calibration"][c]["w2c"])).float()
                      for c in mv_ctx["calibration"]}
        mv_test_camera_arg = mv_ctx["test_cam"]
        print(f"[pipeline.mv] Trainer multi-view args: "
              f"{len(mv_frame_paths_arg)} cams, test={mv_test_camera_arg}")

    # Phase 1.4 + 1.5 + 1.8 — Multi-view foundation directories (cache-only).
    # depth_mv_dir/masks_mv_dir/flow_mv_dir trainer-side wired ile aktif olur.
    depth_mv_dir_arg = (
        paths["depth_mv"]
        if (is_mv and paths["depth_mv"].exists()
            and any(paths["depth_mv"].rglob("*_depth.npy")))
        else None
    )
    masks_mv_dir_arg = (
        paths["masks_mv"]
        if (is_mv and paths["masks_mv"].exists()
            and any(paths["masks_mv"].rglob("mask_*.png")))
        else None
    )
    flow_mv_dir_arg = (
        paths["flow_mv"]
        if (is_mv and paths["flow_mv"].exists()
            and any(paths["flow_mv"].rglob("forward_*.pt")))
        else None
    )
    # Phase 1.8 single-view — flow_dir trainer wire
    flow_dir_arg = (
        paths["flow"]
        if (not is_mv and paths["flow"].exists()
            and any(paths["flow"].glob("forward_*.pt")))
        else None
    )

    # ---- NVS hold-out indices (single source of truth for trainer + Faz 7 eval) ----
    # SV path only. Multi-view's held-out test cam is handled separately via
    # mv_test_camera. If nvs_eval_enabled is off we still compute (cheap) so the
    # trainer can optionally exclude views, but pass [] to keep behavior unchanged.
    sv_holdout_indices: list[int] = []
    if (not is_mv) and len(frame_paths) > 5 and getattr(cfg.train, "nvs_eval_enabled", False):
        T_sv = len(frame_paths)
        if getattr(cfg.train, "static_mode", False):
            # Static: every 8th interleaved (Mip-NeRF360 / vanilla 3DGS convention)
            sv_holdout_indices = list(range(7, T_sv, 8))
        else:
            # 4D: contiguous tail = novel-time eval
            sv_holdout_indices = list(range(max(0, T_sv - max(1, T_sv // 10)), T_sv))
        print(f"[pipeline.holdout] SV NVS hold-out: {len(sv_holdout_indices)} of {T_sv} frames "
              f"({'static_interleaved' if cfg.train.static_mode else 'temporal_tail'})")

    experiment_train_kwargs = _validated_trainer_train_kwargs(
        trainer_train_kwargs
    )
    history = trainer.train(
        frame_paths, K_first, w2c_list,
        n_iters=cfg.train.n_iters,
        image_size=cfg.train.image_resolution,
        ckpt_dir=paths["output"] / "ckpt",
        ckpt_interval=cfg.train.ckpt_interval,
        log_interval=cfg.train.log_interval,
        progress_callback=_train_progress,
        depth_dir=depth_dir_arg,
        mask_dir=mask_dir_arg,
        tracks_path=tracks_path_arg,
        flow_dir=flow_dir_arg,  # Phase 1.8 single-view RAFT flow
        run_logger=run_logger,
        mv_frame_paths=mv_frame_paths_arg,
        mv_cam_K=mv_cam_K_arg,
        mv_w2c=mv_w2c_arg,
        mv_test_camera=mv_test_camera_arg,
        # Phase 1.4 / 1.8 — multi-view depth + flow trainer wire
        depth_mv_dir=depth_mv_dir_arg,
        masks_mv_dir=masks_mv_dir_arg,
        flow_mv_dir=flow_mv_dir_arg,
        # Phase 2.1 — Static/Dynamic auto-promote
        auto_static_dynamic=getattr(cfg.train, "auto_static_dynamic", True),
        static_dynamic_threshold=getattr(cfg.train, "static_dynamic_threshold", 0.10),
        # Static 3DGS Faz 1 — 4D bypass
        static_mode=getattr(cfg.train, "static_mode", False),
        # 4D Quality v6.1 — runtime knobs
        sh_progressive_schedule=getattr(cfg.train, "sh_progressive_schedule", False),
        lambda_accel=getattr(cfg.train, "lambda_accel", 0.0),
        cam_grad_clip_norm=getattr(cfg.train, "cam_grad_clip_norm", 0.0),
        mip_scale_floor_frac=getattr(cfg.train, "mip_scale_floor_frac", 0.0),
        dynamic_densify_scale=getattr(cfg.train, "dynamic_densify_scale", 1.0),
        # Perf — RAM preload (eliminates per-iter MFS latency on cloud filesystems)
        preload_to_ram=getattr(cfg.train, "preload_to_ram", False),
        # NVS hold-out — exclude these frame indices from SV training so eval
        # measures real novel-view PSNR. Recomputed identically in Faz 7.
        holdout_indices=(sv_holdout_indices if sv_holdout_indices else None),
        # Cooperative cancel — passed through from run_pipeline caller.
        cancel_check=cancel_check,
        **experiment_train_kwargs,
    )
    run_logger.phase_end(
        "training",
        final_loss=history["loss"][-1] if history["loss"] else None,
        final_psnr=history["psnr"][-1] if history["psnr"] else None,
        final_n=history["n_pts"][-1] if history["n_pts"] else None,
    )
    # Gerçek log entry sayısından effective iter hesapla (cfg.n_iters değil — aborted olabilir)
    actual_logs = len(history.get("loss", []))
    effective_iters = actual_logs * cfg.train.log_interval
    if effective_iters < cfg.train.n_iters:
        status["training"] = f"⚠ {effective_iters} iter (konfigdeki {cfg.train.n_iters} tamamlanamadı), son loss={history['loss'][-1]:.4f}"
        run_logger.warn("training_incomplete",
                        effective_iters=effective_iters,
                        config_iters=cfg.train.n_iters)
    else:
        status["training"] = f"{effective_iters} iter, son loss={history['loss'][-1]:.4f}"
    cb("training", 1.0, f"Training bitti, son loss={history['loss'][-1]:.4f}",
       {"final_loss": history["loss"][-1] if history["loss"] else None,
        "final_psnr": history["psnr"][-1] if history["psnr"] else None,
        "final_num_points": history["n_pts"][-1] if history["n_pts"] else None})

    # -------- Faz 6: Export --------
    if skip_export:
        print("\n[Faz 6] Export atlandı (--skip-export)")
        status["export"] = "skipped"
        return status

    print("\n[Faz 6] PLY export")
    # Static 3DGS — no deformation: export a single clean frame. Passing the
    # (inert, frozen-random) placeholder deform here would otherwise apply
    # per-timestamp garbage offsets across num_timestamps frames.
    _static = getattr(cfg.train, "static_mode", False)
    _n_ts = 1 if _static else cfg.export.num_timestamps
    cb("export", 0.0, f"{_n_ts} timestamp export ediliyor", {})
    export_to_ply(
        gs=trainer.gs,
        deform=None if _static else trainer.deform,
        output_dir=paths["output"] / "ply",
        num_timestamps=_n_ts,
        scene_extent=extent,
        device=device,
    )
    status["export"] = f"{_n_ts} timestamp"
    cb("export", 1.0, f"{_n_ts} .ply yazıldı",
       {"num_timestamps": _n_ts,
        "ply_dir": str(paths["output"] / "ply")})

    # -------- Faz 7: NVS Evaluation (4D Quality v6.1 — Madde 1+8) --------
    if getattr(cfg.train, "nvs_eval_enabled", False):
        print("\n[Faz 7] NVS Evaluation — held-out cam metrics + orbit render")
        cb("eval", 0.0, "NVS evaluation basliyor", {})
        try:
            from .eval.nvs_eval import (
                _scale_K, eval_held_out_camera, eval_temporal_holdout,
                save_eval_report,
            )
            from .eval.orbit_render import render_orbit_video
            eval_dir = paths["output"] / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            report = {"scene": scene_name, "type": "mv" if is_mv else "sv"}

            # _scale_K (hoisted into eval.nvs_eval) rescales COLMAP-native K to
            # the render resolution; without it reprojection collapses and PSNR
            # floor-pegs at ~8 dB regardless of model quality.
            _render_w, _render_h = cfg.train.image_resolution

            if is_mv and mv_ctx is not None and mv_ctx.get("test_cam"):
                # Held-out test cam (N3V cam00 default)
                test_cam = mv_ctx["test_cam"]
                _calib = mv_ctx["calibration"][test_cam]
                test_K_native = torch.from_numpy(np.array(_calib["K"])).float()
                test_w2c = torch.from_numpy(np.array(_calib["w2c"])).float()
                test_native_w = int(_calib.get("width", _render_w))
                test_native_h = int(_calib.get("height", _render_h))
                test_K = _scale_K(test_K_native, test_native_w, test_native_h,
                                  _render_w, _render_h)
                test_frame_paths = mv_ctx["frame_paths_per_cam"].get(test_cam, [])
                if test_frame_paths:
                    cb("eval", 0.2, f"Held-out cam {test_cam} render", {})
                    held_out = eval_held_out_camera(
                        gs=trainer.gs, deform=trainer.deform,
                        cam_K=test_K, cam_w2c=test_w2c,
                        frame_paths=test_frame_paths,
                        width=_render_w,
                        height=_render_h,
                        static_mode=getattr(cfg.train, "static_mode", False),
                        scene_extent=extent, device=device,
                    )
                    report["held_out_cam"] = test_cam
                    report["held_out_metrics"] = {
                        "psnr": held_out["psnr_mean"],
                        "ssim": held_out["ssim_mean"],
                        "lpips": held_out["lpips_mean"],
                        "n_frames": held_out["n_frames"],
                    }
                    print(f"  Held-out [{test_cam}]: PSNR={held_out['psnr_mean']:.2f} "
                          f"SSIM={held_out['ssim_mean']:.4f} LPIPS={held_out['lpips_mean']:.4f}")
            elif not is_mv and len(frame_paths) > 5:
                # Reuse the same hold-out indices the trainer was told to skip
                # (single source of truth — see sv_holdout_indices construction
                # before the trainer.train() call). With the trainer fix these
                # indices are TRUE novel views; comparable to paper baselines.
                holdout = list(sv_holdout_indices)
                holdout_kind = (
                    "static_interleaved_every_8"
                    if getattr(cfg.train, "static_mode", False)
                    else "temporal_tail_10pct"
                )
                # Scale K to render resolution
                _first_cam = cams[sorted(cams.keys())[0]]
                _native_w = int(_first_cam["width"])
                _native_h = int(_first_cam["height"])
                K_eval = _scale_K(K_first, _native_w, _native_h, _render_w, _render_h)
                cb("eval", 0.2, f"Temporal hold-out ({len(holdout)} frame, {holdout_kind})", {})
                t_metrics = eval_temporal_holdout(
                    gs=trainer.gs, deform=trainer.deform,
                    cam_K=K_eval, cam_w2c_per_frame=w2c_list,
                    frame_paths=frame_paths, holdout_indices=holdout,
                    width=_render_w,
                    height=_render_h,
                    static_mode=getattr(cfg.train, "static_mode", False),
                    scene_extent=extent, device=device,
                )
                report["temporal_holdout"] = {
                    "psnr": t_metrics["psnr_mean"],
                    "ssim": t_metrics["ssim_mean"],
                    "lpips": t_metrics["lpips_mean"],
                    "n_frames": t_metrics["n_frames"],
                    "holdout_kind": holdout_kind,
                }
                print(f"  Temporal hold-out [{holdout_kind}]: "
                      f"PSNR={t_metrics['psnr_mean']:.2f} "
                      f"SSIM={t_metrics['ssim_mean']:.4f} LPIPS={t_metrics['lpips_mean']:.4f}")

            # Smooth orbit video
            cb("eval", 0.6, "Orbit render", {})
            n_orbit = int(getattr(cfg.train, "nvs_eval_orbit_frames", 60))
            fps_orbit = int(getattr(cfg.train, "nvs_eval_orbit_fps", 30))
            orbit_path = eval_dir / "orbit.mp4"
            train_w2c_for_orbit = list(w2c_list) if not is_mv else [
                torch.from_numpy(np.array(mv_ctx["calibration"][c]["w2c"])).float()
                for c in mv_ctx["train_cams"]
            ]
            if not is_mv:
                _first_cam = cams[sorted(cams.keys())[0]]
                K_for_orbit = _scale_K(
                    K_first, int(_first_cam["width"]), int(_first_cam["height"]),
                    _render_w, _render_h,
                )
            else:
                _ref = mv_ctx["calibration"][mv_ctx["train_cams"][0]]
                K_native = torch.from_numpy(np.array(_ref["K"])).float()
                K_for_orbit = _scale_K(
                    K_native, int(_ref.get("width", _render_w)),
                    int(_ref.get("height", _render_h)),
                    _render_w, _render_h,
                )
            orbit_result = render_orbit_video(
                gs=trainer.gs, deform=trainer.deform,
                K=K_for_orbit, train_w2c=train_w2c_for_orbit,
                out_path=orbit_path,
                width=_render_w,
                height=_render_h,
                num_frames=n_orbit, fps=fps_orbit,
                static_mode=getattr(cfg.train, "static_mode", False),
                scene_extent=extent, device=device,
            )
            report["orbit"] = orbit_result

            # JSON report yaz
            save_eval_report(report, eval_dir)
            status["eval"] = (
                f"PSNR={report.get('held_out_metrics', report.get('temporal_holdout', {})).get('psnr', 'n/a')}, "
                f"orbit={'ok' if orbit_result.get('success') else 'fail'}"
            )
            cb("eval", 1.0, f"NVS eval bitti: {status['eval']}", report)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠ NVS eval failed (non-critical): {e}")
            status["eval"] = f"failed: {e}"

    # Run summary — final metrics, config, PLY stats, phase durations
    try:
        ply_dir = paths["output"] / "ply"
        ply_files = sorted(ply_dir.glob("*.ply")) if ply_dir.exists() else []
        total_ply_bytes = sum(p.stat().st_size for p in ply_files)
        run_logger.save_summary({
            "status": status,
            "config": cfg,
            "final_loss": history["loss"][-1] if history.get("loss") else None,
            "final_psnr": history["psnr"][-1] if history.get("psnr") else None,
            "final_n_points": history["n_pts"][-1] if history.get("n_pts") else None,
            "ply_count": len(ply_files),
            "ply_total_bytes": total_ply_bytes,
            "scene_extent": float(extent),
        })
    except Exception as _e:
        print(f"  ⚠ summary save failed: {_e}")
    run_logger.close()

    return status


def main():
    p = argparse.ArgumentParser(description="4DGS Studio - end-to-end pipeline")
    p.add_argument("video", type=str, help="Giriş video dosyası")
    p.add_argument("--scene", default="test_scene", help="Sahne adı (data/<scene>/ altında)")
    p.add_argument("--cloud", action="store_true", help="Cloud config kullan (RTX 4090)")
    p.add_argument("--skip-foundation", action="store_true",
                   help="Foundation model çıkarımını atla (varsayılan)")
    p.add_argument("--with-foundation", action="store_true",
                   help="Foundation modelleri çalıştır")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--colmap-exe", default=None,
                   help="COLMAP exe tam yolu (PATH'te yoksa). "
                        "Alternatif: COLMAP_EXE ortam değişkeni.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Hızlı uçtan uca test: 500 iter, 480x270, 10 timestamp (~1-2 dk)")
    p.add_argument("--force-preprocess", action="store_true",
                   help="Phase 1.1: tum cache'leri yoksay, preprocessing baştan kossun "
                        "(frames/COLMAP/depth/tracks/masks/MVS). "
                        "Default: cache hit'lerde skip (heavy preprocess once, train many times).")
    args = p.parse_args()

    cfg = cloud_config() if args.cloud else default_config()
    if args.colmap_exe:
        cfg.preprocess.colmap_exe = args.colmap_exe

    if args.smoke_test:
        cfg.train.n_iters = 500
        cfg.train.image_resolution = (480, 270)
        cfg.train.ckpt_interval = 500
        cfg.train.log_interval = 25
        cfg.train.density_start_iter = 100
        cfg.train.density_end_iter = 400
        cfg.train.density_interval = 50
        cfg.export.num_timestamps = 10
        print("→ SMOKE TEST modu: 500 iter, 480×270, 10 timestamp")

    skip_found = not args.with_foundation if args.with_foundation else True
    if args.skip_foundation:
        skip_found = True

    t0 = time.time()
    status = run_pipeline(
        args.video, args.scene, cfg,
        skip_foundation=skip_found,
        skip_training=args.skip_training,
        skip_export=args.skip_export,
        force_preprocess=args.force_preprocess,
    )
    print(f"\n=== Pipeline tamamlandi ({time.time()-t0:.1f}s) ===")
    for k, v in status.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

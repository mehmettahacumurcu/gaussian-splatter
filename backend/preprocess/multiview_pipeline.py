"""Multi-view pipeline orchestration (v5.0).

Bu modul mevcut backend/pipeline.py'i degistirmeden multi-view sahnelerin
ozel hazirlama adimlarini yapar. pipeline.py icinden cagrilir:

    if is_multiview_scene(scene):
        ctx = prepare_multiview_scene(paths, cfg)
        # ctx -> {primary_cam, cams_dict, frame_paths_primary, ...}
        # Sonra mevcut trainer'a primary_cam frame_paths ile cagr
    else:
        # Eski single-view path

Yapilan isler:
  1. videos/cam*.mp4 detection
  2. Multi-camera frame extraction (parallel, cached)
  3. poses_bounds.npy parse → calibration.json
  4. Primary camera secimi (test_cam haric en buyuk index)
  5. Trainer icin "single-view-equivalent" args hazirla (primary cam frame'leri)

Multi-view N-cam supervision ileride trainer.py'a (cam, t) sampling
eklendiginde aktif olacak. Su an sadece "smart single-view" mode.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .multiview import (
    extract_frames_multiview,
    parse_n3v_calibration,
    estimate_scene_extent_from_n3v,
    init_random_points_in_bbox,
    multiview_scene_info,
)


def _bootstrap_colmap_init(
    paths: dict,
    frames_dict: Dict[str, List[Path]],
    cfg,
    on_progress=None,
    force_preprocess: bool = False,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """v5.0 ARCHITECTURAL FIX — Multi-cam SfM bootstrap.

    Trainer'in densify+prune dinamigi COLMAP-init varsayar (10-30k anlamli
    sparse cloud). Random init ile multi-view'da N 100k → 27k duser ve
    PSNR 18'de plateau. Cozum: t=0'da tum kameralarin frame'lerini al
    (21 image) → COLMAP exhaustive matching → sparse cloud.

    Single-cam COLMAP CALISMAZ (kamera N3V'de static, parallax yok).
    Multi-cam tek-timestep COLMAP problem'i cozer: 21 farkli viewpoint
    ayni sahnenin (t=0) → ideal SfM input.

    Args:
        paths: scene_paths dict
        frames_dict: cam_name → list of frame paths (extract_frames_multiview output)
        cfg: Config
        on_progress: optional progress callback

    Returns:
        cams: COLMAP-derived camera dict (cam_name → {K, w2c, R, t, width, height})
        xyz: (N, 3) sparse cloud points (in COLMAP world)
        rgb: (N, 3) point colors (uint8)

    Raises:
        Exception: COLMAP basarisizsa (caller fallback'e dusebilir)
    """
    # Lazy import (run_colmap eski single-view yolunda)
    from .run_colmap import run_colmap, run_mvs_dense_reconstruction
    from .parse_colmap import parse_cameras, load_points3d, _find_sparse_dir
    from .cache_utils import (
        is_step_cached, log_cache, write_cache_marker,
        compute_settings_hash, read_cache_marker,
    )

    colmap_dir = paths.get("colmap_mv", paths["base"] / "colmap_multiview")
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # 1) MULTI-TIME COLMAP — her cam'den N timestep al, scene'in farkli
    # zaman dilimlerinden feature toplayarak SfM kalitesini artir.
    #
    # v5.0.1 PREMIUM: Per-cam subfolder yapisi → her cam'in kendi K estimation'i
    # (single_camera_per_folder=1). 21 farkli fiziksel kamera → 21 distinct
    # intrinsic. Tek-K shared modeli yerine per-cam = daha hassas SfM.
    #
    # Layout:
    #   tmp_images/
    #     cam00/
    #       t0000.png, t0060.png, t0120.png, ...
    #     cam01/
    #       t0000.png, ...
    n_timestamps = getattr(cfg.preprocess, "colmap_mv_timestamps", 5)
    tmp_images = colmap_dir / "input_images"
    tmp_images.mkdir(exist_ok=True)
    # Eski dosya/folder'lari temizle (run yeniden, layout degismis olabilir)
    for old in list(tmp_images.iterdir()):
        try:
            if old.is_dir():
                shutil.rmtree(old)
            else:
                old.unlink()
        except OSError:
            pass
    n_imgs = 0
    for cam_name, cam_frames in sorted(frames_dict.items()):
        if not cam_frames:
            continue
        T = len(cam_frames)
        if T <= n_timestamps:
            indices = list(range(T))
        elif n_timestamps == 1:
            indices = [0]
        else:
            indices = [int(round(i * (T - 1) / (n_timestamps - 1)))
                       for i in range(n_timestamps)]
        cam_dir = tmp_images / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        for t_idx in indices:
            src = Path(cam_frames[t_idx])
            # cam00/t0001.png — subfolder = cam_name
            dst = cam_dir / f"t{t_idx:04d}.png"
            shutil.copy2(src, dst)
            n_imgs += 1
    print(f"[bootstrap-colmap] {n_imgs} images organized "
          f"({len(frames_dict)} cams × {n_timestamps} timestamps, per-cam subfolders)")

    # 2) COLMAP CACHE CHECK — eger sparse reconstruction zaten varsa skip.
    # Phase 1.1: force_preprocess=True ile cache bypass.
    # Phase 1.3: settings hash check — colmap_mv_timestamps / colmap_matching /
    # colmap_mv_dense_mvs degisirse cache invalid.
    cache_valid = False
    if not force_preprocess:
        try:
            if (colmap_dir / "sparse").exists():
                cached_sparse = _find_sparse_dir(colmap_dir)
                files_ok = (cached_sparse / "cameras.txt").exists() and (cached_sparse / "points3D.txt").exists()
                if files_ok:
                    # Phase 1.3 hash karsilastirmasi
                    marker = read_cache_marker(paths["base"], "colmap_mv")
                    if marker is None:
                        # Eski cache (marker yok) — conservative: invalidate, hash yaz
                        print(f"[bootstrap-colmap] ⚠ legacy cache (marker yok), settings hash bilinmiyor → cache miss")
                        cache_valid = False
                    else:
                        cur_hash = compute_settings_hash(cfg, "colmap_mv")
                        if marker.get("settings_hash") == cur_hash:
                            cache_valid = True
                            print(f"[bootstrap-colmap] ✓ COLMAP cache HIT @ {cached_sparse} "
                                  f"(settings_hash={cur_hash})")
                        else:
                            print(f"[bootstrap-colmap] ⚠ settings degisti "
                                  f"(cache={marker.get('settings_hash')} vs current={cur_hash}) → cache miss")
                            cache_valid = False
        except Exception as _e:
            print(f"[bootstrap-colmap] cache check failed: {_e}")
            cache_valid = False
    else:
        print(f"[bootstrap-colmap] force_preprocess=True → cache bypass, COLMAP yeniden kosacak")

    if not cache_valid:
        # FIX (Phase 1.3 patch): Cache miss durumunda eski COLMAP artifact'larini sil.
        # database.db, sparse/, dense/ folder'lari onceki run'dan kalmis olabilir;
        # COLMAP eski database'i kullanip "image not found" warning'leri uretiyor
        # ve BA conditioning bozuluyor (CHOLMOD failures). Temiz state olusturalim.
        for stale in ["database.db", "database.db-journal", "database.db-wal", "database.db-shm"]:
            stale_path = colmap_dir / stale
            if stale_path.exists():
                try:
                    stale_path.unlink()
                    print(f"[bootstrap-colmap] cache reset: deleted {stale}")
                except OSError as _e:
                    print(f"[bootstrap-colmap] ⚠ {stale} silinemedi: {_e}")
        for stale_dir in ["sparse", "dense"]:
            d = colmap_dir / stale_dir
            if d.exists():
                try:
                    shutil.rmtree(d)
                    print(f"[bootstrap-colmap] cache reset: deleted {stale_dir}/")
                except OSError as _e:
                    print(f"[bootstrap-colmap] ⚠ {stale_dir}/ silinemedi: {_e}")

        # PREMIUM COLMAP — OPENCV camera model + per-cam K + max features
        print(f"[bootstrap-colmap] Running PREMIUM COLMAP on {n_imgs} images...")
        print(f"[bootstrap-colmap] Settings: OPENCV camera model + per-cam K + 8192 SIFT + BA refinement")
        if on_progress:
            on_progress("colmap_bootstrap", 0.0, "Multi-cam SfM bootstrap", {})

        def _cb(frac, msg):
            if on_progress:
                on_progress("colmap_bootstrap", frac, msg, {})

        # PREMIUM CONFIGURATION:
        #  - camera_model="OPENCV" -> distortion (k1, k2, p1, p2) modeling
        #  - single_camera="per_folder" -> her cam folder'i icin distinct K
        #  - max_num_features=8192 -> SIFT density 4x artir (default ~2000)
        #  - estimate_affine_shape=1 + domain_size_pooling=1 -> SIFT robustness
        # NOT: COLMAP versiyonlari arg desteginde tutarsiz. Matcher/Mapper extras
        # kaldi - default ayarlar zaten BA otomatik focal/principal refine yapar.
        extra_feat = [
            "--SiftExtraction.max_num_features", "8192",       # 4x default density
            "--SiftExtraction.estimate_affine_shape", "1",     # robust descriptors
            "--SiftExtraction.domain_size_pooling", "1",       # scale invariance
        ]
        extra_match: list[str] = []   # default matching — guided_matching uyumsuz
        extra_mapper: list[str] = []  # default mapper — BA refinement otomatik
        run_colmap(
            tmp_images,
            colmap_dir,
            camera_model="OPENCV",  # PINHOLE'den upgrade — distortion modeling
            use_gpu=cfg.preprocess.colmap_use_gpu,
            sequential=False,  # exhaustive — multi-cam icin zorunlu
            colmap_exe=getattr(cfg.preprocess, "colmap_exe", None),
            on_progress=_cb,
            single_camera="per_folder",  # her cam'in kendi K'si
            extra_feat_args=extra_feat,
            extra_match_args=extra_match,
            extra_mapper_args=extra_mapper,
            glob_pattern="**/*.png",  # subfolder'lardan oku
        )
        # Phase 1.3: settings hash + metadata marker yaz, bir sonraki run cache hit alabilsin.
        try:
            best = _find_sparse_dir(colmap_dir)
            n_pts_line = (best / "points3D.txt")
            n_lines = sum(1 for _ in n_pts_line.open()) if n_pts_line.exists() else 0
            write_cache_marker(paths["base"], "colmap_mv", {
                "n_images": n_imgs,
                "n_timestamps": n_timestamps,
                "matching": "exhaustive",
                "camera_model": "OPENCV",
                "sparse_dir": best.name,
                "approx_points": max(0, n_lines - 3),
            }, cfg=cfg)
            print(f"[bootstrap-colmap] cache marker yazildi (settings_hash={compute_settings_hash(cfg, 'colmap_mv')})")
        except Exception as _e:
            print(f"[bootstrap-colmap] ⚠ cache marker yazilirken hata (kritik degil): {_e}")

    # 3) Parse output — multi-time'da her cam birden cok kez registered olur
    # (her timestep ayri image). Static cam'lar icin tum poses ayni olmalidir
    # (kucuk numerik fark var). Per-cam tek pose dondur (median of translations,
    # ilk rotation — static cam'lar icin).
    raw_cams = parse_cameras(colmap_dir)  # {image_name: {K, R, t, w2c, w, h}}
    sparse_xyz, sparse_rgb = load_points3d(colmap_dir)

    # 3b) MVS DENSE RECONSTRUCTION (opsiyonel, premium overnight icin)
    # Sparse SfM ~50-150k point verir. MVS pipeline (image_undistorter +
    # patch_match_stereo + stereo_fusion) ~500k-2M dense point uretir.
    # Time cost: ~30-90 min (525 image, RTX 3060 Ti).
    # Cikarilan dense cloud sparse'tan COK daha iyi 4DGS init olur.
    use_dense_mvs = bool(getattr(cfg.preprocess, "colmap_mv_dense_mvs", False))
    if use_dense_mvs:
        print(f"\n[bootstrap-colmap] Dense MVS reconstruction baslatiliyor (premium)...")
        try:
            # FIX: hardcoded sparse/0 yerine en buyuk sparse subdir'i kullan
            # COLMAP fragmente reconstruction yapabilir (sparse/0, sparse/1, sparse/2)
            # parse_cameras zaten en buyugunu seciyordu, MVS de ayni kullanmali
            sparse_best_dir = _find_sparse_dir(colmap_dir)
            print(f"[bootstrap-colmap] MVS sparse dir: {sparse_best_dir.name}")
            dense_dir = colmap_dir / "dense"
            # Eski dense output varsa sil (bug'lı 116-point reconstruction'ı)
            if dense_dir.exists():
                shutil.rmtree(dense_dir)
            fused_ply = run_mvs_dense_reconstruction(
                image_dir=tmp_images,
                sparse_dir=sparse_best_dir,
                dense_dir=dense_dir,
                colmap_exe=getattr(cfg.preprocess, "colmap_exe", None),
                on_progress=lambda f, m: _cb(0.95 + f * 0.05, f"dense MVS: {m}"),
                geom_consistency=True,
                max_image_size=2000,
            )
            # Dense PLY'i parse et
            from plyfile import PlyData
            ply = PlyData.read(str(fused_ply))
            dense_xyz = np.stack([
                np.array(ply["vertex"]["x"]),
                np.array(ply["vertex"]["y"]),
                np.array(ply["vertex"]["z"]),
            ], axis=1).astype(np.float32)
            if "red" in ply["vertex"].properties[0].name or any("red" in p.name for p in ply["vertex"].properties):
                dense_rgb = np.stack([
                    np.array(ply["vertex"]["red"]),
                    np.array(ply["vertex"]["green"]),
                    np.array(ply["vertex"]["blue"]),
                ], axis=1).astype(np.uint8)
            else:
                dense_rgb = np.full((len(dense_xyz), 3), 128, dtype=np.uint8)
            print(f"[bootstrap-colmap] ✓ MVS dense cloud: {len(dense_xyz):,} points "
                  f"(sparse {len(sparse_xyz):,} → dense {len(dense_xyz):,})")
            # Phase 1.3: MVS dense marker
            try:
                write_cache_marker(paths["base"], "mvs_dense", {
                    "n_dense_points": int(len(dense_xyz)),
                    "n_sparse_points": int(len(sparse_xyz)),
                    "geom_consistency": True,
                }, cfg=cfg)
            except Exception as _e:
                print(f"[bootstrap-colmap] ⚠ mvs_dense marker hata: {_e}")
            # Kullan: dense cloud daha kaliteli init
            xyz, rgb = dense_xyz, dense_rgb
        except Exception as e:
            print(f"[bootstrap-colmap] ⚠ Dense MVS failed: {e}")
            print(f"[bootstrap-colmap]   → Sparse cloud'a fallback")
            xyz, rgb = sparse_xyz, sparse_rgb
    else:
        xyz, rgb = sparse_xyz, sparse_rgb

    # Group by cam_name. Premium layout: "cam00/t0001.png" → "cam00"
    # Eski (flat) layout fallback: "cam00_t0001.png" → "cam00"
    cam_groups: Dict[str, List[Dict]] = {}
    for img_name, info in raw_cams.items():
        # Path separator (forward slash veya backslash)
        if "/" in img_name or "\\" in img_name:
            cam_name = img_name.replace("\\", "/").split("/")[0]
        else:
            stem = Path(img_name).stem
            if "_t" in stem:
                cam_name = stem.rsplit("_t", 1)[0]
            else:
                cam_name = stem
        cam_groups.setdefault(cam_name, []).append(info)

    # Per-cam pose: translation median + first rotation (static cams)
    cams: Dict[str, Dict] = {}
    for cam_name, infos in cam_groups.items():
        if not infos:
            continue
        # Translations: stack and median (outlier-resistant)
        ts = np.stack([np.array(inf["t"]) for inf in infos])  # (M, 3)
        t_med = np.median(ts, axis=0)
        # K: first (should be near-identical for same physical cam)
        K = np.array(infos[0]["K"])
        # Rotation: first (slerp-averaging skip — static cams should agree)
        R = np.array(infos[0]["R"])
        # Build w2c from R + t_med
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = t_med
        cams[cam_name] = {
            "K": K,
            "R": R,
            "t": t_med,
            "w2c": w2c,
            "width": int(infos[0]["width"]),
            "height": int(infos[0]["height"]),
        }

    print(f"[bootstrap-colmap] ✓ {len(cams)} unique cams "
          f"({len(raw_cams)} image registrations from {n_imgs} input images), "
          f"{len(xyz):,} sparse points")
    if on_progress:
        on_progress("colmap_bootstrap", 1.0,
                    f"{len(cams)} cams, {len(xyz)} points", {})
    return cams, xyz, rgb


def prepare_multiview_scene(
    paths: dict,
    cfg,
    on_progress=None,
    force_preprocess: bool = False,
) -> Dict:
    """Multi-view sahne hazirlama. Returns context dict.

    context = {
        "n_cameras": int,
        "train_cams": list[str],
        "test_cam": str,
        "primary_cam": str,                # frontend'de viewer icin secilen
        "calibration": dict,                # cam_name -> {K, w2c, R, t, w, h}
        "frame_paths_per_cam": dict,        # cam_name -> list[Path]
        "primary_frame_paths": list[Path],  # mevcut trainer'a verilecek
        "primary_K": torch.Tensor (3,3),
        "primary_w2c_per_frame": list[torch.Tensor],  # tum t'lerde ayni K/w2c
        "scene_centroid": np.ndarray (3,),
        "scene_extent": float,
        "init_xyz": np.ndarray (N, 3),      # random init (no COLMAP)
        "init_rgb": np.ndarray (N, 3),
    }
    """
    print(f"\n[multiview] Preparing scene...")
    info = multiview_scene_info(paths)
    print(f"  Cameras: {info['n_cameras']}")
    print(f"  Has poses_bounds: {info['has_poses']}")
    print(f"  Has calibration: {info['has_calibration']}")

    # 1) Frame extraction (paralel)
    # Phase 1.3: settings hash check — fps/resize_long_edge degisirse overwrite=True
    from .cache_utils import (
        compute_settings_hash, read_cache_marker, write_cache_marker,
    )
    if on_progress:
        on_progress("frames_mv", 0.0, "Multi-cam frame extraction", {})
    overwrite_frames = bool(force_preprocess)
    if not overwrite_frames:
        try:
            marker = read_cache_marker(paths["base"], "frames_mv")
            cur_hash = compute_settings_hash(cfg, "frames_mv")
            if marker is None:
                # Marker yok ama frames_mv klasorunde dosya varsa: ilk kez,
                # overwrite gerekli degil (extract_frames_multiview kendi cache'liyor),
                # marker'i sonra yazacagiz.
                pass
            elif marker.get("settings_hash") != cur_hash:
                print(f"[multiview] frames_mv settings degisti "
                      f"(cache={marker.get('settings_hash')} vs current={cur_hash}) "
                      f"→ frames yeniden extract edilecek")
                overwrite_frames = True
        except Exception as _e:
            print(f"[multiview] frames_mv cache check hata: {_e}")
    frames_dict = extract_frames_multiview(
        paths["videos_mv"],
        paths["frames_mv"],
        fps=cfg.preprocess.fps,
        resize_long_edge=cfg.preprocess.resize_long_edge,
        overwrite=overwrite_frames,
    )
    # Phase 1.3: marker yaz (settings_hash + cam/frame counts)
    try:
        write_cache_marker(paths["base"], "frames_mv", {
            "n_cams": len(frames_dict),
            "n_frames_per_cam": (
                len(next(iter(frames_dict.values()))) if frames_dict else 0
            ),
            "fps": cfg.preprocess.fps,
            "resize_long_edge": cfg.preprocess.resize_long_edge,
        }, cfg=cfg)
    except Exception as _e:
        print(f"[multiview] ⚠ frames_mv marker yazilirken hata: {_e}")
    if on_progress:
        on_progress("frames_mv", 1.0,
                    f"{len(frames_dict)} cam frames ready", {})

    # 1b) v5.0.1: num_timestamps subsampling (RAM control).
    # Multi-view'da trainer her (cam, t) cift'ini cache'liyor. fps=30 × 10sn
    # video × 20 cam × 720x405 RGB float = ~21 GB RAM. Bu kabul edilemez.
    # cfg.train.num_timestamps ile T'yi kucult: orn 60 frame → 4.2 GB (safe).
    target_T = getattr(cfg.train, "num_timestamps", 0) or 0
    sample_T = next(iter(frames_dict.values()))
    full_T = len(sample_T)
    if 0 < target_T < full_T:
        # Esit aralikli indeksler (temporal coverage'i koru)
        indices = [int(round(i * (full_T - 1) / (target_T - 1)))
                   for i in range(target_T)]
        for cam_name in list(frames_dict.keys()):
            frames_dict[cam_name] = [frames_dict[cam_name][i] for i in indices]
        print(f"  [subsample] T: {full_T} → {target_T} frame (her ~{full_T/target_T:.1f} frame'de 1)")
    else:
        print(f"  [subsample] T: {full_T} frame (no subsample)")

    # 2) Calibration parse (poses_bounds.npy varsa)
    calib = None
    if paths["calibration"].exists():
        calib = parse_n3v_calibration(paths["calibration"])
        print(f"  Calibration parsed: {len(calib)} cameras")
    else:
        # poses_bounds.npy yoksa — load_n3v.py kosulmamis olabilir
        # TODO: COLMAP multi-cam fallback (Sprint 6)
        raise NotImplementedError(
            "Multi-view scene'de calibration.json yok. "
            "scripts/load_n3v.py ile N3V format'i convert et veya "
            "poses_bounds.npy ekle. COLMAP multi-cam fallback henuz yok."
        )

    # 2b) **K SCALE FIX** — N3V poses_bounds.npy K matrisi raw video cozunurlugunde
    # (orn 2704x2028, Meta orijinal recording). Ama ffmpeg ile cikartilan frame'ler
    # cfg.preprocess.resize_long_edge'a gore kuculmus (orn 960). Trainer
    # frame'lerin loaded boyutunu kullaniyor → K loaded scale'de olmali.
    # Bu olcekleme yapilmazsa fx, cx ~3x buyuk → gaussian'lar goruntu disinda
    # projekte → gri kutu, ogrenme yok.
    import imageio.v2 as imageio
    sample_path = next(iter(frames_dict.values()))[0]
    sample_img = imageio.imread(str(sample_path))
    loaded_h, loaded_w = sample_img.shape[:2]
    print(f"  Loaded frame size: {loaded_w}x{loaded_h}")
    for cam_name, cam in calib.items():
        raw_w = int(cam["width"])
        raw_h = int(cam["height"])
        sx_pre = loaded_w / raw_w
        sy_pre = loaded_h / raw_h
        K_raw = np.array(cam["K"], dtype=np.float64)
        K_loaded = K_raw.copy()
        K_loaded[0, 0] *= sx_pre  # fx
        K_loaded[0, 2] *= sx_pre  # cx
        K_loaded[1, 1] *= sy_pre  # fy
        K_loaded[1, 2] *= sy_pre  # cy
        cam["K"] = K_loaded.tolist()
        cam["width"] = loaded_w
        cam["height"] = loaded_h
        if cam_name == sorted(calib.keys())[0]:
            print(f"  K rescale: raw {raw_w}x{raw_h} -> loaded {loaded_w}x{loaded_h} "
                  f"(sx={sx_pre:.3f}, sy={sy_pre:.3f})")
            print(f"  K_loaded[0,0]={K_loaded[0,0]:.1f}, K_loaded[0,2]={K_loaded[0,2]:.1f}")

    # 2c) **BOOTSTRAP COLMAP** — t=0 multi-cam SfM ile sparse cloud + COLMAP poses.
    # N3V poses dogru olabilir ama random init densify+prune dinamigi icin
    # yetersiz (gaussian'lar dogru yerde olmadigindan static phase oturmuyor).
    # COLMAP cikti'yi kullanarak hem init hem (opsiyonel) poses replace edilir.
    #
    # GUARD: Eger cfg.preprocess.use_provided_poses=True (default) ve
    # calibration.json mevcut ise, COLMAP tamamen atlanir. N3V (ve LLFF)
    # icin saglanan calibration submillimeter-accurate; near-coplanar 21-cam
    # rig'lerde COLMAP BA Cholesky failure → degraded poses (~1-2 dB PSNR
    # cost). Downstream init_xyz_colmap=None pathi (estimate_scene_extent_from_n3v
    # + init_random_points_in_bbox) zaten safe.
    skip_colmap = (
        bool(getattr(cfg.preprocess, "use_provided_poses", True))
        and paths["calibration"].exists()
    )
    init_xyz_colmap = None
    init_rgb_colmap = None
    colmap_cams = None
    if skip_colmap:
        print(
            f"  [bootstrap] use_provided_poses=True + calibration.json present "
            f"→ skipping COLMAP, using N3V poses directly"
        )
    else:
        try:
            colmap_cams, init_xyz_colmap, init_rgb_colmap = _bootstrap_colmap_init(
                paths, frames_dict, cfg, on_progress=on_progress,
                force_preprocess=force_preprocess,
            )
            # STRICT VALIDATION: tum cam'lar COLMAP'ta register olmali, ve
            # yeterli sparse cloud uretilmeli. Aksi halde world-mixing bug
            # (kismi COLMAP + kismi N3V → farkli world frame, training collapse).
            # NOT: colmap_cams artik per-cam grouped (multi-time bootstrap).
            new_calib: Dict[str, Dict] = {}
            for cam_name, cam_info in colmap_cams.items():
                new_calib[cam_name] = {
                    "K": np.array(cam_info["K"], dtype=np.float64),
                    "w2c": np.array(cam_info["w2c"], dtype=np.float64),
                    "R": np.array(cam_info["R"], dtype=np.float64),
                    "t": np.array(cam_info["t"], dtype=np.float64),
                    "width": int(cam_info["width"]),
                    "height": int(cam_info["height"]),
                }
            n_n3v = len(calib)
            n_colmap = len(new_calib)
            n_pts = len(init_xyz_colmap) if init_xyz_colmap is not None else 0
            MIN_POINTS = 1500   # smoke test threshold — densify buyutebilir
            MIN_CAMS_RATIO = 0.95  # ≥ 95% cam register olmali (world-mixing onleme)
            if n_colmap < n_n3v * MIN_CAMS_RATIO:
                raise RuntimeError(
                    f"COLMAP only registered {n_colmap}/{n_n3v} cams "
                    f"(threshold {MIN_CAMS_RATIO*100}%). World-mixing risk. "
                    f"Fallback to N3V + random init."
                )
            if n_pts < MIN_POINTS:
                raise RuntimeError(
                    f"COLMAP produced only {n_pts} points (threshold {MIN_POINTS}). "
                    f"Insufficient for densify+prune dynamics. "
                    f"Fallback to N3V + random init."
                )
            # Sample K + w2c print for sanity
            sample_cam = sorted(new_calib.keys())[0]
            K0 = new_calib[sample_cam]["K"]
            w2c0 = new_calib[sample_cam]["w2c"]
            c2w0 = np.linalg.inv(w2c0)
            pos0 = c2w0[:3, 3]
            fwd0 = c2w0[:3, 2]
            print(f"  [bootstrap] {sample_cam} K[0,0]={K0[0,0]:.1f} K[0,2]={K0[0,2]:.1f}")
            print(f"  [bootstrap] {sample_cam} pos={pos0.round(2)} fwd={fwd0.round(2)}")
            calib = new_calib
            print(f"  [bootstrap] ✓ Using COLMAP-derived calibration ({len(calib)} cams, {n_pts:,} points)")
        except Exception as e:
            print(f"  [bootstrap] ✗ COLMAP failed: {e}")
            print(f"  [bootstrap] Falling back to N3V calibration + random init")
            init_xyz_colmap = None
            init_rgb_colmap = None

    # 3) Test/train camera ayrimi
    all_cams = sorted(calib.keys())
    test_cam = cfg.preprocess.multiview_test_camera
    if test_cam and test_cam in all_cams:
        train_cams = [c for c in all_cams if c != test_cam]
    else:
        train_cams = all_cams
        test_cam = None
    print(f"  Train cams: {len(train_cams)}, Test cam: {test_cam}")

    # 4) Primary camera (orta-civar bir cam, su an tek-cam supervision icin)
    primary_cam = train_cams[len(train_cams) // 2] if train_cams else all_cams[0]
    print(f"  Primary camera (single-view supervision): {primary_cam}")

    # 5) Primary cam'in frame_paths'i + K + w2c
    primary_frame_paths = frames_dict.get(primary_cam, [])
    if not primary_frame_paths:
        raise RuntimeError(f"Primary cam {primary_cam} icin frame yok")

    primary_K = torch.from_numpy(np.array(calib[primary_cam]["K"], dtype=np.float64)).float()
    primary_w2c = torch.from_numpy(np.array(calib[primary_cam]["w2c"], dtype=np.float64)).float()
    # Multi-view'da camera fixed, tum frame'lerde ayni → list[w2c] = [w2c] * T
    T = len(primary_frame_paths)
    primary_w2c_per_frame = [primary_w2c.clone() for _ in range(T)]

    # 6) Scene extent — COLMAP varsa points'ten (outlier-rezistant), yoksa N3V kamera projection.
    if init_xyz_colmap is not None and len(init_xyz_colmap) > 100:
        # 50th percentile (median) cluster center — outlier-resistant
        centroid = np.median(init_xyz_colmap, axis=0).astype(np.float32)
        dists = np.linalg.norm(init_xyz_colmap - centroid, axis=1)
        # 75th percentile dist = scene region (foreground), outlier'lar disinda
        # Tight extent foreground'a focus, prune_max_scale daha cok korur.
        extent = float(np.percentile(dists, 75) * 1.5)
        print(f"  [scene] COLMAP-derived center (median): {centroid}")
        print(f"  [scene] COLMAP-derived extent (75% radius x1.5): {extent:.2f}")
    else:
        centroid, extent = estimate_scene_extent_from_n3v(calib)
    print(f"  Scene centroid: {centroid}")
    print(f"  Scene extent: {extent:.2f}")

    # 7) Init points — HYBRID: COLMAP sparse + random fill scene region.
    # COLMAP cloud'u foreground anchor saglar (gercek scene yapisi), random
    # fill densify icin "hammadde" — buyumesi gereken bolgelerde gaussian
    # yetersiz olmasin. Random fill scene_extent radius'da centroid etrafinda.
    if init_xyz_colmap is not None and len(init_xyz_colmap) > 100:
        # COLMAP cloud filter: only keep points within 1.5x extent of centroid
        # (outlier background point'leri at, foreground'a odaklan)
        keep_mask = dists < extent
        colmap_xyz = init_xyz_colmap[keep_mask].astype(np.float32)
        colmap_rgb = init_rgb_colmap[keep_mask]
        if colmap_rgb.dtype != np.uint8:
            colmap_rgb = colmap_rgb.astype(np.uint8)
        n_filtered = len(colmap_xyz)
        n_dropped = len(init_xyz_colmap) - n_filtered

        # Random fill points in scene region (centroid +/- extent box)
        n_random = max(50_000, cfg.train.max_gaussians or 100_000) - n_filtered
        if n_random > 0:
            random_xyz, random_rgb = init_random_points_in_bbox(
                centroid, extent, n_random,
            )
            init_xyz = np.concatenate([colmap_xyz, random_xyz], axis=0)
            init_rgb = np.concatenate([colmap_rgb, random_rgb], axis=0)
        else:
            init_xyz = colmap_xyz
            init_rgb = colmap_rgb

        print(f"  Initial points: {n_filtered:,} COLMAP (filtered, {n_dropped} outlier dropped) "
              f"+ {n_random:,} random fill = {len(init_xyz):,} total")
    else:
        n_init = max(50_000, cfg.train.max_gaussians or 100_000)
        print(f"  Initial random points: {n_init} (COLMAP fallback)")
        init_xyz, init_rgb = init_random_points_in_bbox(centroid, extent, n_init)

    return {
        "n_cameras": len(all_cams),
        "train_cams": train_cams,
        "test_cam": test_cam,
        "primary_cam": primary_cam,
        "calibration": calib,
        "frame_paths_per_cam": frames_dict,
        "primary_frame_paths": primary_frame_paths,
        "primary_K": primary_K,
        "primary_w2c_per_frame": primary_w2c_per_frame,
        "scene_centroid": centroid,
        "scene_extent": extent,
        "init_xyz": init_xyz,
        "init_rgb": init_rgb,
    }

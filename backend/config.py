"""4DGS Studio — Merkezi konfigürasyon.

Pipeline boyunca tüm sabit parametreler burada toplanır.
RTX 3060 Ti (8GB VRAM) için optimize edilmiş varsayılanlar.
Ağır iş için CloudConfig kullan (RunPod RTX 4090 vb.)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DATA_ROOT — where scene folders (frames_multiview/, depth_multiview/, ...) live.
# Default: <project>/data/. Override via FOURDGS_DATA_ROOT env var.
#
# Use case: on RunPod, /workspace/<repo>/data is on MFS network filesystem
# which bottlenecks training at ~3 it/s. Stage the dataset to /dev/shm via
# deploy/runpod/stage_dataset.sh, then `export FOURDGS_DATA_ROOT=/dev/shm/4dgs-studio/data`
# before launching the backend → ~30+ it/s (compute-bound, not IO-bound).
import os as _os
_data_root_env = _os.environ.get("FOURDGS_DATA_ROOT")
if _data_root_env:
    DATA_ROOT = Path(_data_root_env).resolve()
    print(f"[config] DATA_ROOT overridden via FOURDGS_DATA_ROOT={DATA_ROOT}")
else:
    DATA_ROOT = PROJECT_ROOT / "data"


def scene_paths(scene_name: str) -> dict[str, Path]:
    """Bir sahne için standart klasör yapısı döndür.

    v5.0: Multi-view path'leri eklendi (yeni, opsiyonel).
    Auto-detection: pipeline 'videos/' klasörü varsa multi-view mode kullanır.
    'video.mp4' tek dosya varsa single-view (eski davranış).
    """
    base = DATA_ROOT / scene_name
    return {
        "base":    base,
        # Single-view (mevcut, backward-compat)
        "video":   base / "video.mp4",
        "frames":  base / "frames",
        "colmap":  base / "colmap",
        "depth":   base / "depth",
        "tracks":  base / "tracks",
        "masks":   base / "masks",
        "output":  base / "output",
        # v5.0 — Multi-view (yeni, opsiyonel)
        "videos_mv":           base / "videos",            # cam00.mp4, cam01.mp4, ...
        "frames_mv":           base / "frames_multiview",  # frames_multiview/cam00/frame_0000.png
        "colmap_mv":           base / "colmap_multiview",  # multi-cam reconstruction
        "depth_mv":            base / "depth_multiview",   # depth_multiview/cam00/...
        "masks_mv":            base / "masks_multiview",
        "flow":                base / "flow",              # Phase 1.8 — single-view forward flow
        "flow_mv":             base / "flow_multiview",    # Phase 1.8 — per-cam forward flow
        "poses_bounds":        base / "poses_bounds.npy",  # N3V format (opsiyonel)
        "calibration":         base / "calibration.json",  # custom multi-cam intrinsics (opsiyonel)
    }


def is_multiview_scene(scene_name: str) -> bool:
    """v5.0: Sahnenin multi-view mode'da olup olmadığını tespit et.

    Heuristic: 'videos/' klasörü VAR ve içinde >=2 video dosyası varsa multi-view.
    Aksi halde single-view (mevcut 'video.mp4' veya 'frames/' kullanır).
    """
    paths = scene_paths(scene_name)
    videos_mv = paths["videos_mv"]
    if not videos_mv.exists() or not videos_mv.is_dir():
        return False
    # Windows NTFS case-insensitive: glob("*.mp4") ve glob("*.MP4") ayni dosyalari
    # iki kez yakalayabilir → set ile dedupe.
    video_files = set(videos_mv.glob("cam*.mp4")) | set(videos_mv.glob("cam*.MP4"))
    return len(video_files) >= 2


def is_static_scene(scene_name: str) -> bool:
    """Static 3DGS — Faz 1: photo-set / sparse-view sahnesini tespit et.

    Heuristic (oncelik sirasi):
      1. data/<scene>/static.flag → varsa True (manuel override).
      2. data/<scene>/images/ klasoru var ve >=3 jpg/png/exr varsa True
         (klasik 3DGS / NeRF Synthetic / Tanks & Temples / Mip-NeRF360 yapisi).
      3. data/<scene>/video.mp4 yok, frames/ yok ama colmap/ var → True.
         (kullanici sadece COLMAP klasoru sagladiysa fotograf seti demektir).
      4. Yukaridakilerin hicbiri yoksa False (4D dynamic veya video pipeline'i).

    Bu helper sadece tespit eder; cfg.train.static_mode'u set ETMEZ. Ust katman
    (CLI / pipeline / preset script) bu sonuca gore kararini verir.
    """
    paths = scene_paths(scene_name)
    base = paths["base"]
    if not base.exists():
        return False
    # 1) Manuel flag
    if (base / "static.flag").exists():
        return True
    # 2) images/ klasoru (klasik 3DGS yapisi)
    images_dir = base / "images"
    if images_dir.exists() and images_dir.is_dir():
        n_imgs = 0
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            n_imgs += len(list(images_dir.glob(f"*{ext}")))
            if n_imgs >= 3:
                return True
    # 3) Sadece colmap/ var, video/frames yok
    has_video = paths["video"].exists()
    has_frames = paths["frames"].exists() and any(paths["frames"].glob("frame_*.png"))
    has_colmap = paths["colmap"].exists() and (paths["colmap"] / "sparse").exists()
    if has_colmap and not has_video and not has_frames:
        return True
    return False


def list_multiview_cameras(scene_name: str) -> list[str]:
    """Multi-view sahnesinin kameralarını listele (cam00, cam01, ...).

    Returns:
        Sıralı kamera ID listesi (örn ['cam00', 'cam01', 'cam02']).
        Boş liste = single-view veya kamera yok.
    """
    paths = scene_paths(scene_name)
    videos_mv = paths["videos_mv"]
    if not videos_mv.exists():
        return []
    # Windows NTFS case-insensitive dedupe (path -> stem set).
    files = set(videos_mv.glob("cam*.mp4")) | set(videos_mv.glob("cam*.MP4"))
    return sorted({p.stem for p in files})


# ---------------------------------------------------------------------------
# Faz 2 — Video ön işleme
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    fps: int = 10
    resize_long_edge: int | None = 960
    colmap_camera_model: str = "PINHOLE"
    colmap_use_gpu: bool = True
    sequential_overlap: int = 10
    # v3.9: COLMAP matching strategy.
    # - "sequential": yakin frame'leri match eder (hizli, default)
    # - "exhaustive": tum frame'ler birbirleriyle match (yavas N^2,
    #   orbital camera icin loop closure saglar)
    colmap_matching: str = "sequential"
    # v3.9: Initial point subsample mode.
    # - "random": cap'in %70'ine random downsample (eski, hizli)
    # - "confidence": COLMAP track length / (1 + reproj_error) skoruyla
    #   high-confidence noktalari sec, outlier'lari at
    init_subsample_mode: str = "random"
    colmap_exe: str | None = None
    # 4D Quality v6.1 — Madde 3: Sparse-view init method.
    # "colmap" (default), "dust3r" (transformer-based dense pointmap, opt-in),
    # "auto" (sparse-view detect: <20 frame ise dust3r, aksi colmap).
    # DUSt3R model weights ilk run'da indirilir (~500 MB).
    init_method: str = "colmap"
    # DUSt3R sparse-view threshold — auto modda bu sayidan az frame varsa DUSt3R.
    sparse_view_threshold_frames: int = 20
    # v5.0 — Multi-view configuration
    # multiview_mode: 'auto' (detect from data/), 'single', 'multi'
    multiview_mode: str = "auto"
    # Eger poses_bounds.npy varsa (N3V format), COLMAP atlanir mi?
    # True (default): multiview_pipeline calibration.json varsa
    # _bootstrap_colmap_init'i tamamen atlar ve N3V poses'lari direkt kullanir.
    # Init points init_random_points_in_bbox + estimate_scene_extent_from_n3v
    # ile uretilir. ~27dk preprocess tasarrufu, +1.0-2.5 dB PSNR (N3V near-
    # coplanar 21-cam rig'lerde COLMAP dense BA Cholesky failure cikariyor).
    # False: her zaman COLMAP bootstrap (eski davranis).
    use_provided_poses: bool = True
    # Multi-view'de hangi kamera'yi "test" (held-out) olarak ayir?
    # N3V convention: cam00 test, kalanlar train. None ise hepsi train.
    multiview_test_camera: str | None = "cam00"
    # Per-camera frame extraction: hepsini paralel mi yap, sirayla mi?
    multiview_parallel_extract: bool = True
    # Multi-view bootstrap COLMAP: kac timestep kullan (her cam'den).
    # 1 = tek timestep (en hizli, ~2.7k point) — chickchicken kalitesi disinda
    # 5 = standart (5x feature, 25x matching, ~20-40k point) ← ONERILEN
    # 10 = derin (10x feature, ~50k point, ~10-15dk preprocess)
    # 25 = premium overnight (25x feature, 137k matching pair, ~80-150k sparse, ~45-90 dk)
    colmap_mv_timestamps: int = 5
    # Multi-view bootstrap: MVS dense reconstruction (premium, +30-90 dk).
    # Sparse SfM 50-150k point → dense 500k-2M point. 4DGS init kalitesi
    # dramatik artar (chickchicken seviyesi guarantee).
    colmap_mv_dense_mvs: bool = False


# ---------------------------------------------------------------------------
# Faz 3 — Foundation modeller
# ---------------------------------------------------------------------------

@dataclass
class FoundationConfig:
    metric3d_model: str = "metric3d_vit_small"
    cotracker_num_points: int = 2048
    cotracker_grid_size: int = 30
    sam2_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Faz 4-5 — Gaussian model + training
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    init_random_points: int = 0
    sh_degree: int = 3
    # Deformation field — T7 perf fix (3060 Ti 8GB target):
    # Eski default'lar (96/48/512/4) per iter MLP forward'ı 100k+ gauss icin
    # ciddi maliyetliydi. 4-5 katini deformation MLP'sine harciyorduk + uzun
    # training (10+ saat). Cloud config (cloud_config()) bu degerleri tekrar
    # 96/48/512/4'e itiyor — 24GB GPU + 60k iter butcesi ile capacity
    # gerekli oldugunda kullanilsin.
    hexplane_resolution: int = 64
    hexplane_feat_dim: int = 32
    mlp_width: int = 384
    mlp_depth: int = 3
    num_time_freqs: int = 6
    # v3.6 / Yol C — Per-gaussian Fourier trajectory (4DGS paper SOTA)
    # "mlp"     -> eski global MLP (tum pozisyon MLP'den)
    # "fourier" -> saf per-gaussian trajectory (MLP dpos kullanilmaz, dquat/dscale icin MLP)
    # "hybrid"  -> her ikisi (MLP global bias + per-gaussian ozgun trajectory)
    deform_pos_mode: str = "hybrid"
    fourier_K: int = 8               # Frekans sayisi: 48 param/gaussian (K x 2 x 3)
    # Phase 2.5 — Multi-resolution HexPlane. Bos = single-res (mevcut).
    # Onerilen: [24, 48, 96] (3 scale, total feat 6*24*3 = 432).
    multires_resolutions: list = field(default_factory=list)
    multires_feat_dim: int = 24


@dataclass
class TrainConfig:
    n_iters: int = 30_000
    image_resolution: tuple[int, int] = (640, 360)
    batch_size: int = 1
    lambda_ssim: float = 0.2
    # Adam learning rates
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh_dc: float = 2.5e-3
    lr_sh_rest: float = 2.5e-3 / 20
    lr_deform: float = 3e-3             # v3: 1e-3 -> 3e-3 (motion daha hizli ogren)
    lr_fourier: float = 3e-3            # v3.6.1: 5e-3 -> 3e-3 (kontrolsuz buyume azalt)
    # Fourier regularizer (low-pass prior)
    lambda_fourier_reg: float = 1e-3    # v3.6.1: 1e-4 -> 1e-3 (10x guclu)
    # Density control
    density_start_iter: int = 500
    density_end_iter: int = 22_000
    density_interval: int = 100
    densify_grad_threshold: float = 2e-4
    prune_min_opacity: float = 0.005
    # T9 perf fix: 3060 Ti 8GB icin guvenli hard cap. Eskiden 0 (sinirsiz),
    # buyuk sahnelerde split sirasinda N=200k+ olup OOM ediyordu. 400k cap
    # ile training stuck/abort durumu ortadan kalkmali. Cloud config (24GB)
    # bu degeri 0'a (sinirsiz) ceker.
    max_gaussians: int = 400_000
    prune_max_scale: float = 0.02       # v3.2: FRACTION of scene_extent
    opacity_reset_interval: int = 3000  # INRIA-3DGS standard (was 0)
    # Motion regularizers (Stage 1) — v3.4 denge
    lambda_deform_reg: float = 3e-5
    lambda_smoothness: float = 2e-4
    lambda_rigidity: float = 2e-4
    # Scale + anti-streak (v3 / v3.8)
    lambda_scale: float = 5e-3
    lambda_aniso: float = 0.01  # anti-streak base (0.01-0.05 onerilen, was 0.0)
    aniso_threshold: float = 5.0
    dpos_total_cap_frac: float = 0.2
    # Foundation model losses (Stage 2)
    lambda_depth: float = 0.1
    lambda_mask_motion: float = 2.0
    lambda_track: float = 0.5
    track_sample_k: int = 256
    warmup_iters: int = 500
    ckpt_interval: int = 1000
    log_interval: int = 50
    # v5.0 — Multi-view training
    # Multi-view: her iter'de kac kamera sample edilir (>=1).
    # Higher -> daha tutarli multi-view supervision, daha yavas iter.
    multiview_cams_per_iter: int = 1
    # Multi-view consistency loss: ayni 3D point farkli cam'lardan benzer renk vermeli.
    # 0 = kapali, >0 = aktif (Phase 1.7 implementation).
    lambda_multiview_consistency: float = 0.05  # SOTA-tier MV cross-cam supervision (was 0.0)
    # Phase 1.6 — LPIPS perceptual loss. 0 = off.
    # VGG (kaliteli, edge sharpness) vs alex (hizli ama low-detail).
    # Production tier: vgg + lambda 0.1+. Mini smoke: alex + 0.05.
    lambda_lpips: float = 0.05  # perceptual loss base (was 0.0)
    lpips_net: str = "alex"  # alex (hizli, pipeline default) | vgg (kaliteli) | squeeze
    lpips_warmup_iters: int = 1000
    # Phase 1.8 — RAFT optical flow loss. 0 = off.
    # NOTE: 0.05 onerilen for SOTA-tier (cloud_config); pahali (RAFT inference)
    # ve foundation phase'de gating'lenir, bu yuzden default 0.0 kaliyor.
    lambda_flow: float = 0.0
    flow_warmup_iters: int = 1000
    # Phase 1.9 — Densify dynamics tuning (multi-view spesifik defaults)
    # Multi-view'da daha aggresif densify gerekir (her cam ayri view).
    densify_mv_threshold_scale: float = 0.7  # 1.0 = single-view ile ayni, 0.7 = %30 daha hassas
    # Phase 2.2 — Multi-resolution training schedule (multi-stage)
    # Liste: [(iter, long_edge), ...]. Bos = single-resolution.
    # Orn: [(0, 480), (10000, 720), (25000, 1080)]
    multires_schedule: list = field(default_factory=list)
    # Phase 2.3 — Camera pose refinement (Joint BA)
    # 0.0 = kapali, lr_cam_K=1e-7, lr_cam_w2c=1e-7 onerilen.
    lr_cam_K: float = 0.0
    lr_cam_w2c: float = 0.0
    cam_refine_start_iter: int = 5000  # warmup sonrasi cam refine basla
    # Phase 2.4 — Background distance ratio. 0 = kapali, 2.0 = 2x scene_extent ote
    bg_distance_ratio: float = 2.0
    # Phase 2.1 — Static/Dynamic auto-promote (multi-view only).
    # masks_mv'dan motion vote toplayip dinamik gauss seçer.
    # 0 = kapali (manuel API), >0 = auto-promote threshold (motion-vote frac).
    auto_static_dynamic: bool = True
    static_dynamic_threshold: float = 0.10  # gauss %10+ frame motion -> dynamic
    # Static 3DGS — Faz 1: 4D dynamic features tamamen bypass.
    # True: deformation MLP + Fourier trajectory yok, sadece statik 3D.
    # Mevcut 4D kodu olduğu gibi reuse eder, training/inference sade 3DGS.
    static_mode: bool = False
    # 4D Quality v6.1 — Madde 11: Adaptive SH degree schedule
    # True: SH degree iter'a göre 0→1→2→3 artarak progresif öğrenme.
    # Erken iter'lerde DC + 1st order yeterken late iter'lerde 3rd order detail kazandırır.
    # Schedule: %25 / %50 / %75 / %100 of n_iters bandlarinda degree 0/1/2/3.
    sh_progressive_schedule: bool = True
    # 4D Quality v6.1 — Madde 7: Adaptive densify dynamic regions
    # Dynamic gauss'lar icin densify_grad_threshold * dynamic_densify_scale.
    # 0.5 = %50 hassas (dynamic'te 2x daha agresif densify). 1.0 = no-op.
    dynamic_densify_scale: float = 0.5
    # 4D Quality v6.1 — Madde 6: 2nd-order temporal smoothness (acceleration ceza)
    # D(t-1) - 2D(t) + D(t+1) magnitude on dpos. 0 = off.
    # 2026-05-02 audit: was 0.0 default; bumped to 1e-4 (recommended for SOTA).
    lambda_accel: float = 1e-4
    # 4D Quality v6.1 — Madde 12: Cam refinement gradient norm clipping
    # cam_K + cam_w2c parametreleri icin ayri grad_norm clip degeri. 0 = off.
    cam_grad_clip_norm: float = 1.0
    # 4D Quality v6.1 — Madde 2-B: Mip-Splatting Python-side anti-aliasing
    # 3D scale floor — her gauss'a min scale = mip_scale_floor_frac × distance_to_nearest_cam.
    # 0.0 = off. 0.0005-0.002 onerilen. Anti-aliasing approximation; tam Mip-Splatting CUDA
    # kernel degil, ama ekran-uzayinda yakin gauss'larin "noktalasmasini" engeller.
    # 2026-05-02 audit: was 0.0 default; bumped to 0.001 (recommended).
    mip_scale_floor_frac: float = 0.001
    # 4D Quality v6.1 — NVS evaluation
    # Training sonrasi held-out cam metrics + orbit cam mp4 render.
    # False default: pipeline'a ek faz eklemeden run. UI/CLI flag ile aktive edilir.
    nvs_eval_enabled: bool = False
    nvs_eval_orbit_frames: int = 60  # orbit mp4 frame count
    nvs_eval_orbit_fps: int = 30
    # Perf — RAM preload all frames/depth/masks at training start.
    # False (default) keeps the existing lazy disk-read path. True triggers
    # ThreadPoolExecutor(16) parallel preload at master cache resolution
    # (uint8 RGB / fp16 depth / uint8 mask). Eliminates per-iter MFS network
    # filesystem latency on cloud setups (RunPod, etc). flame_steak at 1080p
    # ~53 GB total — fits comfortably in 232 GB cloud RAM.
    preload_to_ram: bool = False


@dataclass
class ExportConfig:
    num_timestamps: int = 60
    format: str = "ply"


# ---------------------------------------------------------------------------
# Birlesik
# ---------------------------------------------------------------------------

@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    foundation: FoundationConfig = field(default_factory=FoundationConfig)
    model:      ModelConfig      = field(default_factory=ModelConfig)
    train:      TrainConfig      = field(default_factory=TrainConfig)
    export:     ExportConfig     = field(default_factory=ExportConfig)


def default_config() -> Config:
    """Varsayilan (RTX 3060 Ti 8GB) konfigurasyon."""
    return Config()


def cloud_config() -> Config:
    """Daha agir is icin cloud GPU (RTX 4090 24GB / A100 80GB) konfigurasyonu.

    SOTA-tier overrides: stronger perceptual + MV consistency, RAFT flow loss
    enabled (foundation phase generates), joint bundle adjustment (cam refine).
    """
    cfg = default_config()
    cfg.preprocess.fps = 24
    cfg.preprocess.resize_long_edge = 1920
    cfg.foundation.metric3d_model = "metric3d_vit_large"
    cfg.foundation.cotracker_num_points = 4096
    cfg.foundation.cotracker_grid_size = 60
    # Cloud GPU'da deformation field'i full kapasiteye geri al — 24GB VRAM
    # buyuk MLP'i tasiyor + 60k iter butcesi ek capacity'i kullanabiliyor.
    # Default config bu degerleri 64/32/384/3'e indiriyor (3060 Ti 8GB icin).
    cfg.model.hexplane_resolution = 96
    cfg.model.hexplane_feat_dim = 48
    cfg.model.mlp_width = 512
    cfg.model.mlp_depth = 4
    cfg.train.n_iters = 60_000
    cfg.train.image_resolution = (1920, 1080)
    cfg.train.batch_size = 4
    # 24GB VRAM has headroom — lift the 8GB cap.
    cfg.train.max_gaussians = 0
    # SOTA-tier quality overrides (override defaults UPWARD for 24-80 GB cards)
    cfg.train.lambda_lpips = 0.1                  # stronger perceptual
    cfg.train.lambda_multiview_consistency = 0.1  # stronger MV
    cfg.train.lambda_aniso = 0.02
    cfg.train.lambda_accel = 2e-4
    cfg.train.lambda_flow = 0.05                  # enable RAFT flow loss
    cfg.train.lr_cam_K = 1e-7                     # joint BA
    cfg.train.lr_cam_w2c = 1e-7
    cfg.train.cam_refine_start_iter = 8000
    cfg.export.num_timestamps = 120
    return cfg


def local_max_config() -> Config:
    """Max-quality config for 8GB GPU (RTX 3060 Ti). 14-22h overnight runs.
    Pushes 8GB to its limits: 350k Gaussians, HexPlane 96/40, MLP 512/4,
    Fourier K=12, all v6.1 quality features enabled, perceptual loss on (alex)."""
    cfg = default_config()
    # Preprocess
    cfg.preprocess.fps = 15
    cfg.preprocess.resize_long_edge = 1280
    cfg.preprocess.colmap_matching = "exhaustive"
    cfg.preprocess.init_subsample_mode = "confidence"
    # Foundation (small models, must fit alongside training in 8GB)
    cfg.foundation.metric3d_model = "metric3d_vit_small"
    cfg.foundation.cotracker_grid_size = 22
    cfg.foundation.cotracker_num_points = 1400
    # Model
    cfg.model.hexplane_resolution = 96
    cfg.model.hexplane_feat_dim = 40
    cfg.model.mlp_width = 512
    cfg.model.mlp_depth = 4
    cfg.model.num_time_freqs = 8
    cfg.model.fourier_K = 12
    cfg.model.deform_pos_mode = "hybrid"
    cfg.model.sh_degree = 3
    # Train
    cfg.train.n_iters = 80_000
    cfg.train.image_resolution = (800, 450)
    cfg.train.batch_size = 1
    cfg.train.max_gaussians = 350_000
    cfg.train.density_start_iter = 800
    cfg.train.density_end_iter = 45_000
    cfg.train.density_interval = 200
    cfg.train.densify_grad_threshold = 3e-4
    cfg.train.prune_max_scale = 0.012
    cfg.train.warmup_iters = 1500
    cfg.train.lambda_aniso = 0.02
    cfg.train.aniso_threshold = 5.0
    cfg.train.dpos_total_cap_frac = 0.08
    cfg.train.lambda_rigidity = 8e-4
    cfg.train.lambda_fourier_reg = 5e-3
    cfg.train.lambda_accel = 2e-4
    cfg.train.lambda_flow = 0.05
    cfg.train.flow_warmup_iters = 1500
    cfg.train.lambda_lpips = 0.05
    cfg.train.lpips_net = "alex"
    cfg.train.lpips_warmup_iters = 1500
    cfg.train.dynamic_densify_scale = 0.4
    cfg.train.mip_scale_floor_frac = 0.0015
    cfg.train.cam_grad_clip_norm = 1.0
    cfg.train.lr_cam_K = 1e-7
    cfg.train.lr_cam_w2c = 1e-7
    cfg.train.cam_refine_start_iter = 8000
    cfg.train.nvs_eval_enabled = True
    cfg.train.ckpt_interval = 4000
    # Export
    cfg.export.num_timestamps = 120
    return cfg

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
    video_files = list(videos_mv.glob("cam*.mp4")) + list(videos_mv.glob("cam*.MP4"))
    return len(video_files) >= 2


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
    cams = sorted([
        p.stem for p in videos_mv.glob("cam*.mp4")
    ] + [
        p.stem for p in videos_mv.glob("cam*.MP4")
    ])
    return cams


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
    # v5.0 — Multi-view configuration
    # multiview_mode: 'auto' (detect from data/), 'single', 'multi'
    multiview_mode: str = "auto"
    # Eger poses_bounds.npy varsa (N3V format), COLMAP atlanir mi?
    use_provided_poses: bool = True
    # Multi-view'de hangi kamera'yi "test" (held-out) olarak ayir?
    # N3V convention: cam00 test, kalanlar train. None ise hepsi train.
    multiview_test_camera: str | None = "cam00"
    # Per-camera frame extraction: hepsini paralel mi yap, sirayla mi?
    multiview_parallel_extract: bool = True


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
    # Deformation field — v2 genisletildi
    hexplane_resolution: int = 96
    hexplane_feat_dim: int = 48
    mlp_width: int = 512
    mlp_depth: int = 4
    num_time_freqs: int = 6
    # v3.6 / Yol C — Per-gaussian Fourier trajectory (4DGS paper SOTA)
    # "mlp"     -> eski global MLP (tum pozisyon MLP'den)
    # "fourier" -> saf per-gaussian trajectory (MLP dpos kullanilmaz, dquat/dscale icin MLP)
    # "hybrid"  -> her ikisi (MLP global bias + per-gaussian ozgun trajectory)
    deform_pos_mode: str = "hybrid"
    fourier_K: int = 8               # Frekans sayisi: 48 param/gaussian (K x 2 x 3)


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
    max_gaussians: int = 0
    prune_max_scale: float = 0.02       # v3.2: FRACTION of scene_extent
    opacity_reset_interval: int = 0
    # Motion regularizers (Stage 1) — v3.4 denge
    lambda_deform_reg: float = 3e-5
    lambda_smoothness: float = 2e-4
    lambda_rigidity: float = 2e-4
    # Scale + anti-streak (v3 / v3.8)
    lambda_scale: float = 5e-3
    lambda_aniso: float = 0.0
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
    # 0 = kapali (default), >0 = aktif (Sprint 4 implementation).
    lambda_multiview_consistency: float = 0.0


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
    """Daha agir is icin cloud GPU (RTX 4090 24GB) konfigurasyonu."""
    cfg = default_config()
    cfg.preprocess.fps = 24
    cfg.preprocess.resize_long_edge = 1920
    cfg.foundation.metric3d_model = "metric3d_vit_large"
    cfg.foundation.cotracker_num_points = 4096
    cfg.foundation.cotracker_grid_size = 60
    cfg.train.n_iters = 60_000
    cfg.train.image_resolution = (1920, 1080)
    cfg.train.batch_size = 4
    cfg.export.num_timestamps = 120
    return cfg

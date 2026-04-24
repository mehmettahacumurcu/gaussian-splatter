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
    """Bir sahne için standart klasör yapısı döndür."""
    base = DATA_ROOT / scene_name
    return {
        "base":    base,
        "video":   base / "video.mp4",
        "frames":  base / "frames",
        "colmap":  base / "colmap",
        "depth":   base / "depth",
        "tracks":  base / "tracks",
        "masks":   base / "masks",
        "output":  base / "output",
    }


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
    colmap_exe: str | None = None


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
    # Deformation field — v2 genişletildi
    hexplane_resolution: int = 96
    hexplane_feat_dim: int = 48
    mlp_width: int = 512
    mlp_depth: int = 4
    num_time_freqs: int = 6


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
    lr_deform: float = 3e-3             # v3: 1e-3 → 3e-3 — motion'un daha hızlı öğrenilmesi için (ÇALIŞIYOR)
    # Density control — v3.1: aggressive prune revert, extended end korundu
    density_start_iter: int = 500
    density_end_iter: int = 22_000      # v3: 15k → 22k (final prune'lar için)
    density_interval: int = 100
    densify_grad_threshold: float = 2e-4
    prune_min_opacity: float = 0.005
    prune_max_scale: float = 0.02       # v3.2: FRACTION of scene_extent (INRIA original intent)
                                        # trainer.py içinde: effective = prune_max_scale * scene_extent
                                        # 0.02 × 70 = 1.4 units → reasonable bloat cap
    # Opacity reset — v3.1: kapatıldı (density control zaten prune yapıyor,
    # bu interval aggressive prune ile birleşince %95 gaussian öldü)
    opacity_reset_interval: int = 0     # v3.1: 3000 → 0 (KAPA)
    # Motion regularizers (Stage 1) — v3.4: DENGE.
    # v3.2: reg güçlü → motion yok
    # v3.3: reg=0 → scales inf, training patladı
    # v3.4: reg 1/10 of v3.2 — stability için minimum, motion'u killing değil
    lambda_deform_reg: float = 3e-5     # v3.4: was 3e-4 (v3.2), 0 (v3.3) → 3e-5 (1/10)
    lambda_smoothness: float = 2e-4     # v3.4: was 2e-3 (v3.2), 0 (v3.3) → 2e-4 (1/10)
    lambda_rigidity: float = 2e-4       # v3.4: aynı, 1/10 of v3.2
    # Scale regularizer — v3.1: ASIMETRIK hinge (sadece scene_extent %5 üzerini cezalandır)
    # v3'te symmetric formul tüm scale'leri 1'e itip homogenization yaratmıştı.
    # v3.1'de trainer.py içinde threshold-based hinge kullanılıyor (bkz. orada).
    lambda_scale: float = 5e-3          # v3: 1e-3 → 5e-3 (5× güçlü ama asimetrik, sadece outlier hit)
    # Foundation model losses (Stage 2) — v3.4 denge
    lambda_depth: float = 0.1
    lambda_mask_motion: float = 2.0     # v3.4: 1.0 → 2.0 (denge, v3.3'teki 3.0 overshoot)
    lambda_track: float = 0.3           # v3.4: 0.1 (v3.2) × 3 = 0.3 (v3.3'teki 1.0 patlattı)
    track_sample_k: int = 256           # Her iter kaç track sample'lansın
    # Warmup (regularizer'lar linear 0 → full over first N iter)
    warmup_iters: int = 500             # v3: 2000 → 500 (reg'ler erken devreye girsin ama yumuşak)
    # Checkpoint
    ckpt_interval: int = 1000
    log_interval: int = 50


@dataclass
class ExportConfig:
    num_timestamps: int = 60
    format: str = "ply"


# ---------------------------------------------------------------------------
# Birleşik
# ---------------------------------------------------------------------------

@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    foundation: FoundationConfig = field(default_factory=FoundationConfig)
    model:      ModelConfig      = field(default_factory=ModelConfig)
    train:      TrainConfig      = field(default_factory=TrainConfig)
    export:     ExportConfig     = field(default_factory=ExportConfig)


def default_config() -> Config:
    """Varsayılan (RTX 3060 Ti 8GB) konfigürasyon."""
    return Config()


def cloud_config() -> Config:
    """Daha ağır iş için cloud GPU (RTX 4090 24GB) konfigürasyonu."""
    cfg = default_config()
    cfg.preprocess.fps = 24
    cfg.preprocess.resize_long_edge = 1920
    cfg.foundation.metric3d_model = "metric3d_vit_large"
    cfg.foundation.cotracker_num_points = 4096
    cfg.train.image_resolution = (1920, 1080)
    cfg.train.n_iters = 60_000
    cfg.export.num_timestamps = 120
    return cfg

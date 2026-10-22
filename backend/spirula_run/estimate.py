"""Rough VRAM and training-time estimate for a Spirula run, shown live next to the settings.

Calibrated on the 3060 Ti (myroom, 270 frames at 1920x1080, ``--cache-images disk``):
memory is a fixed part that grows with the training resolution plus a per-splat part
(parameters + gradients + two Adam moments for 59 SH3 floats is 944 B, plus densify
stats and allocator slack). Training time per step grows with splats and pixels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from . import settings as S

# Measured peaks over idle: 1.00M splats -> 1458 MiB, 1.71M -> 1830 MiB (1080p, divisor 1);
# 1M at divisor 2 used ~170 MiB less than at divisor 1. The fit errs on the high side.
BASE_MIB = 370            # Spirula/Vulkan context + kernels, independent of the scene
MIB_PER_MEGAPIXEL = 115   # render/loss buffers per training-image megapixel
BYTES_PER_SPLAT = 900
# 8k steps averaging ~0.3M splats: 28.7 ms/step; 30k steps at 1M: 37.6 ms/step (both 1080p).
STEP_MS_BASE = 11.5        # per training megapixel
STEP_MS_PER_MSPLAT = 16.0  # per million live splats (average over the run)
AVG_SPLAT_SHARE = 0.85     # densification ramps up to the cap; the run averages below it
SAFETY_MIB = 600          # keep this free for the driver and spikes

# Colab speeds are relative to the 3060 Ti that the constants above come from.
GPU_PROFILES: dict[str, dict] = {
    "local": dict(label="Bu PC", vram_gib=None, speed=1.0),
    "T4": dict(label="Colab T4", vram_gib=15.0, speed=0.45),
    "L4": dict(label="Colab L4", vram_gib=22.5, speed=0.9),
    "A100": dict(label="Colab A100", vram_gib=40.0, speed=2.6),
    "H100": dict(label="Colab H100", vram_gib=80.0, speed=4.0),
    "G4": dict(label="Colab G4 (RTX PRO 6000)", vram_gib=96.0, speed=3.5),
}


@dataclass
class Estimate:
    vram_mib: int
    vram_splats_mib: int
    vram_base_mib: int
    available_mib: int | None
    fits: bool | None
    max_splats: int | None       # largest cap that still fits on this GPU
    train_seconds: int
    gpu: str
    gpu_label: str
    train_megapixels: float

    def as_dict(self) -> dict:
        return asdict(self)


def train_megapixels(width: int, height: int, divisor: int) -> float:
    return (width / divisor) * (height / divisor) / 1e6


def estimate(settings: S.SpirulaSettings, image_width: int = 1920, image_height: int = 1080,
             gpu: str = "local", local_total_mib: int | None = None, local_used_mib: int | None = None) -> Estimate:
    run = S.resolve(settings)
    profile = GPU_PROFILES.get(gpu, GPU_PROFILES["local"])
    mp = train_megapixels(image_width, image_height, run.resolution_divisor)
    base = BASE_MIB + MIB_PER_MEGAPIXEL * mp
    splats = BYTES_PER_SPLAT * run.cap_max / 2**20
    vram = base + splats

    if gpu == "local":
        # Other apps already hold VRAM on a desktop GPU; count what's actually free.
        available = None if local_total_mib is None else local_total_mib - (local_used_mib or 0)
    else:
        available = int(profile["vram_gib"] * 1024)
    fits = max_splats = None
    if available is not None:
        budget = available - SAFETY_MIB
        fits = vram <= budget
        max_splats = max(0, int((budget - base) * 2**20 / BYTES_PER_SPLAT))

    step_ms = STEP_MS_BASE * mp + STEP_MS_PER_MSPLAT * AVG_SPLAT_SHARE * run.cap_max / 1e6
    seconds = run.iterations * step_ms / 1000 / profile["speed"]
    return Estimate(vram_mib=int(round(vram)), vram_splats_mib=int(round(splats)), vram_base_mib=int(round(base)),
                    available_mib=available, fits=fits, max_splats=max_splats, train_seconds=int(round(seconds)),
                    gpu=gpu, gpu_label=profile["label"], train_megapixels=round(mp, 2))

from __future__ import annotations

from typing import Any

from scripts.static_3dgs import PRESETS, _apply_preset


NATIVE_RES_STATIC_PRESETS = {"sota", "ultra"}


def static_preset_names() -> set[str]:
    return set(PRESETS)


def apply_static_preset_for_api(cfg: Any, preset_name: str) -> str:
    name = (preset_name or "balanced").strip().lower()
    if name not in PRESETS:
        raise ValueError(f"unknown static preset: {preset_name}")

    _apply_preset(cfg, name)
    if name in NATIVE_RES_STATIC_PRESETS:
        cfg.train.native_resolution = True
    return name

"""Public entry for sub-project B.

run_image_to_scene(image_path, scene_name, cfg, progress_callback) -> dict

Implemented in Task 9.1.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable


def run_image_to_scene(
    image_path: str | Path,
    scene_name: str,
    cfg: Any = None,
    progress_callback: Callable | None = None,
) -> dict:
    raise NotImplementedError("Task 9.1 implements this.")

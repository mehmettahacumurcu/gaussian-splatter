"""Sub-project B — single-image full-scene splat reconstruction.

Spec: docs/superpowers/specs/2026-05-19-single-image-fullscene-splat-design.md

Public entry: backend.image_to_scene.runner.run_image_to_scene(...)
"""
from .runner import run_image_to_scene  # re-export

__all__ = ["run_image_to_scene"]

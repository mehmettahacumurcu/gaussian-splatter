"""Regression: static_mode must be able to construct Trainer4DGS.

Colab P2-V run (2026-07-03) crashed at trainer init: static_mode forces
fourier_K=0 in GaussianModel (Phase 2 memory saver), but run_pipeline still
passed cfg.model.deform_pos_mode — default 'hybrid' — which the trainer's
init validation (correctly) rejects with fourier_K=0. deform_pos_mode is
inert in static mode (deformation is bypassed in train()), so the pipeline
must force 'mlp' alongside fourier_K=0, mirroring image_to_scene/runner.py.
"""
from __future__ import annotations

import pytest
import torch

from backend.config import default_config
from backend.model.deformation import DeformationField
from backend.model.gaussian_model import GaussianModel
from backend.model.trainer import Trainer4DGS
from backend.pipeline import _static_trainer_overrides


def _tiny_gs(fourier_K: int) -> GaussianModel:
    pts = torch.rand(8, 3)
    return GaussianModel(pts, sh_degree=0, fourier_K=fourier_K)


def _placeholder_deform() -> DeformationField:
    # Same minimal placeholder run_pipeline builds in static mode
    return DeformationField(resolution=8, feat_dim=4, mlp_width=32,
                            mlp_depth=1, num_time_freqs=2)


def test_static_mode_overrides_fourier_and_deform_mode():
    cfg = default_config()
    cfg.train.static_mode = True
    assert _static_trainer_overrides(cfg) == (0, "mlp")


def test_dynamic_mode_keeps_config_values():
    cfg = default_config()
    cfg.train.static_mode = False
    assert _static_trainer_overrides(cfg) == (
        cfg.model.fourier_K, cfg.model.deform_pos_mode,
    )


def test_trainer_constructs_with_static_overrides():
    cfg = default_config()
    cfg.train.static_mode = True
    fourier_K, deform_pos_mode = _static_trainer_overrides(cfg)
    gs = _tiny_gs(fourier_K)
    Trainer4DGS(gs, _placeholder_deform(), device="cpu", scene_extent=1.0,
                deform_pos_mode=deform_pos_mode)


def test_trainer_rejects_hybrid_with_fourier_k0():
    # The exact Colab crash: hybrid needs per-gaussian fourier coeffs
    gs = _tiny_gs(0)
    with pytest.raises(ValueError, match="fourier_K"):
        Trainer4DGS(gs, _placeholder_deform(), device="cpu", scene_extent=1.0,
                    deform_pos_mode="hybrid")

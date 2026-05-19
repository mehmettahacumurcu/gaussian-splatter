from backend.image_to_scene.config import ImageToSceneConfig, profile
import pytest


def test_default_profile_has_30_views():
    cfg = profile("default")
    assert cfg.n_views == 30
    assert cfg.train_iterations == 3000


def test_fast_profile_has_15_views():
    cfg = profile("fast")
    assert cfg.n_views == 15
    assert cfg.train_iterations == 1500


def test_quality_profile_has_50_views():
    cfg = profile("quality")
    assert cfg.n_views == 50
    assert cfg.train_iterations == 5000


def test_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown profile"):
        profile("nonsense")


def test_bubble_radius_default_5m():
    cfg = profile("default")
    assert cfg.bubble_radius_m == 5.0

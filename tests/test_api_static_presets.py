from backend.config import default_config
from backend.static_presets import apply_static_preset_for_api, static_preset_names
from scripts.static_3dgs import PRESETS


def test_api_static_preset_names_track_static_runner_presets():
    assert static_preset_names() == set(PRESETS)


def test_api_applies_sota_and_ultra_static_presets_without_fallback():
    cases = {
        "sota": 6_000_000,
        "ultra": 3_000_000,
    }

    for preset_name, max_gaussians in cases.items():
        cfg = default_config()

        applied = apply_static_preset_for_api(cfg, preset_name)

        assert applied == preset_name
        assert cfg.train.static_mode is True
        assert cfg.train.max_gaussians == max_gaussians
        assert cfg.train.native_resolution is True

from backend.notebooks.models import NotebookQualityProfile
from backend.notebooks.presets import get_notebook_profile, get_static_preset_manifest
from scripts.static_3dgs import PRESETS


def test_product_profiles_match_approved_budgets_and_legacy_training_values() -> None:
    expected = {
        "balanced_l4": ("balanced", 30_000, 250_000, 300, 1280),
        "high": ("high", 50_000, 500_000, 450, 1920),
        "premium": ("premium", 100_000, 1_000_000, 600, 2560),
        "ultra": ("ultra", 120_000, 3_000_000, 800, None),
    }
    for profile_id, values in expected.items():
        profile = get_notebook_profile(NotebookQualityProfile(profile_id))
        assert (profile.legacy_preset, profile.n_iters, profile.max_gaussians, profile.selected_frame_budget, profile.resolution_long_edge_cap) == values
        assert PRESETS[profile.legacy_preset]["n_iters"] == profile.n_iters
        assert PRESETS[profile.legacy_preset]["max_gaussians"] == profile.max_gaussians


def test_manifest_has_one_default_and_backend_owned_bounds() -> None:
    manifest = get_static_preset_manifest()
    assert manifest.default_profile == NotebookQualityProfile.BALANCED_L4
    assert len(manifest.profiles) == 4
    assert manifest.override_limits.n_iters.min == 1_000
    assert manifest.override_limits.n_iters.max == 120_000

from __future__ import annotations

import json

import pytest
import torch

from backend.model.gaussian_model import GaussianModel
from experiments.learned_quality.density import (
    AdaptiveDensityController,
    AdaptiveDensityPolicy,
)


GIB = 1024**3


def _model(count: int = 4) -> GaussianModel:
    points = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10.0
    model = GaussianModel(points, sh_degree=0, fourier_K=0)
    model.scales.data.fill_(torch.log(torch.tensor(0.005)))
    model.opacities.data.fill_(2.0)
    return model


def _controller(
    *,
    hard_cap: int = 20,
    grad_threshold: float = 0.1,
    memory_probe=None,
) -> AdaptiveDensityController:
    return AdaptiveDensityController(
        AdaptiveDensityPolicy(
            hard_cap=hard_cap,
            grad_threshold=grad_threshold,
            split_scale_threshold=0.02,
            min_opacity=0.005,
            max_scale=0.1,
            analytical_bytes_per_gaussian=1024,
        ),
        memory_probe=memory_probe or (lambda: (70 * GIB, 80 * GIB)),
    )


def _observe(
    controller: AdaptiveDensityController,
    model: GaussianModel,
    gradients: list[float],
    *views: object,
) -> None:
    for view in views:
        model.means.grad = torch.tensor(gradients).unsqueeze(-1).repeat(1, 3)
        controller.accumulate_view(model, view_id=view)


def test_requires_two_distinct_views_and_selects_top_gradient() -> None:
    model = _model()
    controller = _controller(hard_cap=5)
    original = model.means.detach().clone()

    _observe(controller, model, [1.0, 4.0, 3.0, 2.0], "same", "same")
    no_growth = controller.step_at(model, iteration=100)
    assert no_growth["after"] == 4

    _observe(controller, model, [1.0, 4.0, 3.0, 2.0], "left", "right")
    growth = controller.step_at(model, iteration=200)

    assert growth["after"] == 5
    assert torch.equal(model.means[-1], original[1])


def test_prunes_before_growth_and_never_overshoots_cap() -> None:
    model = _model(5)
    model.opacities.data[0] = -20.0
    controller = _controller(hard_cap=5)
    _observe(controller, model, [5.0] * 5, 0, 1)

    event = controller.step_at(model, iteration=100)

    assert event["pruned"] == 1
    assert event["post_prune"] == 4
    assert event["selected"] == 1
    assert model.num_points == 5


def test_vram_slots_and_calibration_limit_growth() -> None:
    snapshots = iter(
        [
            (10 * GIB + 2500, 40 * GIB),
            (10 * GIB + 500, 40 * GIB),
        ]
    )
    model = _model(4)
    controller = _controller(
        hard_cap=100,
        memory_probe=lambda: next(snapshots),
    )
    _observe(controller, model, [5.0] * 4, "a", "b")

    event = controller.step_at(model, iteration=100)

    assert event["vram_growth_slots"] == 2
    assert event["selected"] == 2
    assert event["calibrated_bytes_per_gaussian"] == 1536


def test_invalid_memory_calibration_stops_growth_but_not_pruning() -> None:
    model = _model(3)
    controller = _controller(memory_probe=lambda: (float("nan"), 80 * GIB))
    _observe(controller, model, [5.0] * 3, "a", "b")
    first = controller.step_at(model, iteration=100)
    assert first["selected"] == 0
    assert first["growth_stopped_reason"] == "invalid_memory_calibration"

    model.opacities.data[0] = -20.0
    second = controller.step_at(model, iteration=200)
    assert second["pruned"] == 1


def test_three_empty_candidate_windows_stop_future_growth() -> None:
    model = _model(3)
    controller = _controller(grad_threshold=10.0)
    for iteration in (100, 200, 300):
        _observe(controller, model, [1.0] * 3, iteration, -iteration)
        event = controller.step_at(model, iteration=iteration)

    assert event["growth_stopped_reason"] == "low_candidate_patience"
    controller.policy = AdaptiveDensityPolicy(
        **{**controller.policy.to_dict(), "grad_threshold": 0.1}
    )
    _observe(controller, model, [20.0] * 3, "new-a", "new-b")
    assert controller.step_at(model, iteration=400)["selected"] == 0


def test_quality_plateau_needs_baseline_plus_three_low_deltas_and_resets() -> None:
    controller = _controller()

    controller.record_quality(5000, 20.0, (640, 360))
    controller.record_quality(10000, 20.01, (640, 360))
    controller.record_quality(15000, 20.02, (640, 360))
    assert not controller.growth_stopped
    controller.record_quality(20000, 20.03, (640, 360))
    assert controller.growth_stopped_reason == "quality_plateau"

    controller.record_quality(25000, 20.04, (1280, 720))
    assert not controller.growth_stopped
    assert controller.quality_low_delta_count == 0


def test_optimizer_state_shapes_follow_density_mutations() -> None:
    model = _model(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sum(parameter.sum() for parameter in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    controller = _controller(hard_cap=5)
    _observe(controller, model, [5.0] * 4, "a", "b")

    controller.step_at(model, optimizer=optimizer, iteration=100)

    for parameter in model.parameters():
        state = optimizer.state.get(parameter)
        if state and "exp_avg" in state and parameter.ndim > 0:
            assert state["exp_avg"].shape == parameter.shape
            assert state["exp_avg_sq"].shape == parameter.shape


def test_density_history_is_json_safe() -> None:
    model = _model(3)
    controller = _controller(hard_cap=4)
    _observe(controller, model, [5.0] * 3, "a", "b")
    controller.step_at(model, iteration=100)
    controller.record_quality(5000, 20.0, (640, 360))

    assert json.loads(json.dumps(controller.history)) == controller.history


def test_policy_rejects_non_emergency_cap() -> None:
    with pytest.raises(ValueError, match="6,000,000"):
        AdaptiveDensityPolicy(hard_cap=6_000_001)

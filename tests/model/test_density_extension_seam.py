from __future__ import annotations

import inspect

import pytest

from backend.model.trainer import (
    Trainer4DGS,
    _density_accumulate,
    _density_step,
    _run_density_quality_probe,
)
from backend.pipeline import (
    _apply_trainer_customizer,
    _preserve_experiment_depth,
    _validated_trainer_train_kwargs,
    run_pipeline,
)


class LegacyDensity:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def accumulate(self, model: object) -> None:
        self.calls.append(("accumulate", model))

    def step(self, model: object, **kwargs: object) -> dict[str, int]:
        self.calls.append(("step", model, kwargs))
        return {"after": 1}


class ExperimentDensity:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def accumulate_view(self, model: object, *, view_id: object) -> None:
        self.calls.append(("accumulate_view", model, view_id))

    def step_at(self, model: object, **kwargs: object) -> dict[str, int]:
        self.calls.append(("step_at", model, kwargs))
        return {"after": 2}


def test_default_density_calls_are_exactly_legacy() -> None:
    controller = LegacyDensity()
    model = object()
    optimizer = object()

    _density_accumulate(controller, model, view_id=17)
    result = _density_step(
        controller,
        model,
        optimizer=optimizer,
        dynamic_densify_scale=0.7,
        iteration=500,
    )

    assert controller.calls == [
        ("accumulate", model),
        (
            "step",
            model,
            {"optimizer": optimizer, "dynamic_densify_scale": 0.7},
        ),
    ]
    assert result == {"after": 1}


def test_experiment_density_receives_view_and_iteration_context() -> None:
    controller = ExperimentDensity()
    model = object()

    _density_accumulate(controller, model, view_id=("cam-a", 12))
    result = _density_step(
        controller,
        model,
        optimizer=None,
        dynamic_densify_scale=1.0,
        iteration=700,
    )

    assert controller.calls == [
        ("accumulate_view", model, ("cam-a", 12)),
        (
            "step_at",
            model,
            {
                "optimizer": None,
                "dynamic_densify_scale": 1.0,
                "iteration": 700,
            },
        ),
    ]
    assert result == {"after": 2}


def test_quality_probe_is_default_off_and_runs_only_each_5000() -> None:
    calls: list[tuple[object, ...]] = []

    class Density:
        def record_quality(self, **kwargs: object) -> None:
            calls.append(("record", kwargs))

    class Trainer:
        density = Density()

    trainer = Trainer()

    assert _run_density_quality_probe(None, trainer, 5000, (640, 360), 3) is None
    assert (
        _run_density_quality_probe(
            lambda *args: calls.append(("probe", args)) or 22.5,
            trainer,
            4999,
            (640, 360),
            2,
        )
        is None
    )
    assert calls == []

    value = _run_density_quality_probe(
        lambda *args: calls.append(("probe", args)) or 22.5,
        trainer,
        5000,
        (1280, 720),
        3,
    )
    assert value == 22.5
    assert calls == [
        ("probe", (trainer, 5000, (1280, 720), 3)),
        (
            "record",
            {
                "iteration": 5000,
                "aggregate_valid_psnr_db": 22.5,
                "resolution": (1280, 720),
            },
        ),
    ]


def test_train_and_pipeline_extension_defaults_are_none() -> None:
    train = inspect.signature(Trainer4DGS.train).parameters
    pipeline = inspect.signature(run_pipeline).parameters

    assert train["density_quality_probe"].default is None
    assert pipeline["trainer_customizer"].default is None
    assert pipeline["trainer_train_kwargs"].default is None


def test_customizer_is_called_only_when_supplied() -> None:
    trainer = object()
    calls: list[object] = []

    _apply_trainer_customizer(trainer, None)
    _apply_trainer_customizer(trainer, calls.append)

    assert calls == [trainer]


def test_extra_train_kwargs_reject_explicit_collisions() -> None:
    assert _validated_trainer_train_kwargs(None) == {}
    assert _validated_trainer_train_kwargs(
        {"validity_mask": ("mask",), "density_quality_probe": "probe"}
    ) == {"validity_mask": ("mask",), "density_quality_probe": "probe"}

    with pytest.raises(ValueError, match="n_iters"):
        _validated_trainer_train_kwargs({"n_iters": 1})
    with pytest.raises(ValueError, match="unknown"):
        _validated_trainer_train_kwargs({"unknown": 1})


def test_only_explicit_experiment_evidence_preserves_precomputed_depth() -> None:
    assert not _preserve_experiment_depth(None)
    assert _preserve_experiment_depth({})
    assert _preserve_experiment_depth({"validity_mask": ("mask",)})

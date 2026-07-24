from __future__ import annotations

import inspect

import pytest
import torch

from backend.model.trainer import (
    Trainer4DGS,
    _normalize_diagnostic_iterations,
    _run_training_diagnostic,
    _sample_training_index,
)
from backend.pipeline import _validated_trainer_train_kwargs


def test_seeded_camera_generator_replays_the_same_sampling_sequence() -> None:
    first = torch.Generator(device="cpu").manual_seed(1701)
    second = torch.Generator(device="cpu").manual_seed(1701)

    first_sequence = tuple(_sample_training_index(800, first) for _ in range(20))
    second_sequence = tuple(_sample_training_index(800, second) for _ in range(20))

    assert first_sequence == second_sequence
    assert len(set(first_sequence)) > 1
    assert all(0 <= index < 800 for index in first_sequence)


def test_camera_sampling_preserves_legacy_global_rng_when_generator_is_none() -> None:
    torch.manual_seed(91)
    expected = tuple(int(torch.randint(0, 7, (1,)).item()) for _ in range(5))
    torch.manual_seed(91)

    actual = tuple(_sample_training_index(7, None) for _ in range(5))

    assert actual == expected


def test_diagnostic_callback_runs_only_at_normalized_checkpoints() -> None:
    checkpoints = _normalize_diagnostic_iterations(
        (2_500, 500, 500, 1_000, 0),
        5_000,
    )
    calls: list[tuple[object, int, tuple[int, int], int]] = []
    trainer = object()

    for iteration in (0, 499, 500, 999, 1_000, 2_499, 2_500, 5_000):
        _run_training_diagnostic(
            lambda *values: calls.append(values),
            trainer,
            iteration,
            checkpoints,
            (1280, 720),
            3,
        )

    assert checkpoints == frozenset({0, 500, 1_000, 2_500})
    assert calls == [
        (trainer, 0, (1280, 720), 3),
        (trainer, 500, (1280, 720), 3),
        (trainer, 1_000, (1280, 720), 3),
        (trainer, 2_500, (1280, 720), 3),
    ]


@pytest.mark.parametrize("values", [(-1,), (5_001,)])
def test_diagnostic_checkpoints_must_fit_the_training_run(values: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="diagnostic_iterations"):
        _normalize_diagnostic_iterations(values, 5_000)


def test_train_invokes_initial_diagnostic_before_the_optimizer_loop() -> None:
    source = inspect.getsource(Trainer4DGS.train)

    initial = source.index("initial_diagnostic_sh_degree")
    loop = source.index("for it in range(1, n_iters + 1)")

    assert initial < loop


def test_train_diagnostic_extensions_are_default_off() -> None:
    parameters = inspect.signature(Trainer4DGS.train).parameters

    assert parameters["camera_generator"].default is None
    assert parameters["diagnostic_iterations"].default is None
    assert parameters["diagnostic_callback"].default is None
    assert parameters["sh_progressive_horizon_iters"].default is None


def test_pipeline_accepts_only_the_explicit_diagnostic_extensions() -> None:
    callback = object()
    generator = object()
    values = {
        "camera_generator": generator,
        "diagnostic_iterations": (500, 1_000),
        "diagnostic_callback": callback,
        "sh_progressive_horizon_iters": 5_000,
    }

    assert _validated_trainer_train_kwargs(values) == values
    with pytest.raises(ValueError, match="unknown"):
        _validated_trainer_train_kwargs({**values, "diagnostic_output": "unsafe"})


def test_train_uses_explicit_progressive_sh_horizon_when_provided() -> None:
    source = inspect.getsource(Trainer4DGS.train)

    assert source.count("sh_schedule_horizon") >= 5
    assert "progressive_sh_degree(it, n_iters" not in source

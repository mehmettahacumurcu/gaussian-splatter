from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from experiments.learned_quality.lifecycle import (
    BatchAttemptRecord,
    BatchRetryResult,
    VramReleaseRecord,
    release_cuda_model,
    run_with_smaller_batch_retry,
)


class FakeCudaOutOfMemoryError(RuntimeError):
    pass


class FakeCuda:
    def __init__(
        self,
        events: list[str],
        *,
        available: bool,
        allocated: tuple[int, ...] = (),
        reserved: tuple[int, ...] = (),
    ) -> None:
        self.events = events
        self.available = available
        self.allocated = list(allocated)
        self.reserved = list(reserved)

    def is_available(self) -> bool:
        self.events.append("cuda.is_available")
        return self.available

    def memory_allocated(self) -> int:
        self.events.append("cuda.memory_allocated")
        return self.allocated.pop(0)

    def memory_reserved(self) -> int:
        self.events.append("cuda.memory_reserved")
        return self.reserved.pop(0)

    def empty_cache(self) -> None:
        self.events.append("cuda.empty_cache")


class FakeModel:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def to(self, device: str) -> None:
        self.events.append(f"model.to:{device}")


class FailingModel(FakeModel):
    def __init__(self, events: list[str], error: BaseException) -> None:
        super().__init__(events)
        self.error = error

    def to(self, device: str) -> None:
        super().to(device)
        raise self.error


def _fake_torch(cuda: FakeCuda) -> object:
    return SimpleNamespace(
        cuda=cuda,
        OutOfMemoryError=FakeCudaOutOfMemoryError,
    )


def _release_record() -> VramReleaseRecord:
    return VramReleaseRecord(
        cuda_available=False,
        allocated_before_bytes=None,
        reserved_before_bytes=None,
        allocated_after_bytes=None,
        reserved_after_bytes=None,
        moved_to_cpu=False,
        gc_ran=True,
        cache_cleared=False,
    )


@pytest.mark.parametrize(
    ("stage", "initial_size", "retry_size", "chunk_independent", "message"),
    [
        ("", 8, 4, True, "stage"),
        ("   ", 8, 4, True, "stage"),
        ("bad\nstage", 8, 4, True, "stage"),
        ("semantic", True, 1, True, "initial_size"),
        ("semantic", 0, 1, True, "initial_size"),
        ("semantic", 8, True, True, "retry_size"),
        ("semantic", 8, 0, True, "retry_size"),
        ("semantic", 8, 8, True, "smaller"),
        ("semantic", 8, 9, True, "smaller"),
        ("semantic", 8, 4, False, "chunk-independent"),
    ],
)
def test_retry_validation_rejects_before_callbacks(
    stage: str,
    initial_size: int,
    retry_size: int,
    chunk_independent: bool,
    message: str,
) -> None:
    calls: list[str] = []

    with pytest.raises(ValueError, match=message):
        run_with_smaller_batch_retry(
            stage,
            initial_size,
            retry_size,
            lambda _size: calls.append("operation"),
            chunk_independent=chunk_independent,
            release=lambda: calls.append("release"),  # type: ignore[arg-type]
            torch_module=_fake_torch(FakeCuda([], available=False)),
        )

    assert calls == []


def test_da3_anchor_retry_is_forbidden_without_mutating_quality_locks() -> None:
    anchor_count = 120
    process_resolution = 504
    calls: list[str] = []

    with pytest.raises(ValueError, match="da3_anchor"):
        run_with_smaller_batch_retry(
            "da3_anchor",
            anchor_count,
            96,
            lambda _size: calls.append("operation"),
            chunk_independent=True,
            release=lambda: calls.append("release"),  # type: ignore[arg-type]
            torch_module=_fake_torch(FakeCuda([], available=True)),
        )

    assert calls == []
    assert anchor_count == 120
    assert process_resolution == 504


def test_release_cuda_model_on_cpu_moves_model_and_keeps_caller_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    cuda = FakeCuda(events, available=False)
    model = FakeModel(events)
    original_model = model

    def collect() -> int:
        events.append("gc.collect")
        return 0

    monkeypatch.setattr("experiments.learned_quality.lifecycle.gc.collect", collect)
    record = release_cuda_model(model, torch_module=_fake_torch(cuda))

    assert events == ["cuda.is_available", "model.to:cpu", "gc.collect"]
    assert model is original_model
    assert record == VramReleaseRecord(
        cuda_available=False,
        allocated_before_bytes=None,
        reserved_before_bytes=None,
        allocated_after_bytes=None,
        reserved_after_bytes=None,
        moved_to_cpu=True,
        gc_ran=True,
        cache_cleared=False,
    )
    with pytest.raises(FrozenInstanceError):
        record.gc_ran = False  # type: ignore[misc]


def test_initial_success_returns_one_immutable_attempt_without_release() -> None:
    operation_sizes: list[int] = []
    release_calls: list[str] = []

    result = run_with_smaller_batch_retry(
        "semantic",
        8,
        4,
        lambda size: operation_sizes.append(size) or "artifact",
        chunk_independent=True,
        release=lambda: release_calls.append("release"),  # type: ignore[arg-type]
        torch_module=_fake_torch(FakeCuda([], available=False)),
    )

    assert operation_sizes == [8]
    assert release_calls == []
    assert result == BatchRetryResult(
        stage="semantic",
        value="artifact",
        initial_size=8,
        retry_size=None,
        attempts=(BatchAttemptRecord(size=8, outcome="succeeded"),),
        release_record=None,
    )
    with pytest.raises(FrozenInstanceError):
        result.retry_size = 4  # type: ignore[misc]


def test_cuda_release_records_vram_and_drops_local_reference_before_gc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    cuda = FakeCuda(
        events,
        available=True,
        allocated=(900, 100),
        reserved=(1_200, 300),
    )
    model = FakeModel(events)

    def collect() -> int:
        release_frame = inspect.currentframe().f_back  # type: ignore[union-attr]
        local_dropped = release_frame.f_locals["model"] is None
        events.append(f"gc.collect:local_dropped={local_dropped}")
        return 7

    monkeypatch.setattr("experiments.learned_quality.lifecycle.gc.collect", collect)
    record = release_cuda_model(model, torch_module=_fake_torch(cuda))

    assert events == [
        "cuda.is_available",
        "cuda.memory_allocated",
        "cuda.memory_reserved",
        "model.to:cpu",
        "gc.collect:local_dropped=True",
        "cuda.empty_cache",
        "cuda.memory_allocated",
        "cuda.memory_reserved",
    ]
    assert record == VramReleaseRecord(
        cuda_available=True,
        allocated_before_bytes=900,
        reserved_before_bytes=1_200,
        allocated_after_bytes=100,
        reserved_after_bytes=300,
        moved_to_cpu=True,
        gc_ran=True,
        cache_cleared=True,
    )


def test_model_to_failure_still_cleans_cuda_and_reraises_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    cuda = FakeCuda(
        events,
        available=True,
        allocated=(500, 50),
        reserved=(800, 80),
    )
    move_error = RuntimeError("cpu move failed")
    model = FailingModel(events, move_error)

    def collect() -> int:
        events.append("gc.collect")
        return 0

    monkeypatch.setattr("experiments.learned_quality.lifecycle.gc.collect", collect)
    with pytest.raises(RuntimeError, match="cpu move failed") as caught:
        release_cuda_model(model, torch_module=_fake_torch(cuda))

    assert caught.value is move_error
    assert events == [
        "cuda.is_available",
        "cuda.memory_allocated",
        "cuda.memory_reserved",
        "model.to:cpu",
        "gc.collect",
        "cuda.empty_cache",
        "cuda.memory_allocated",
        "cuda.memory_reserved",
    ]


def test_cuda_oom_releases_once_before_one_smaller_retry() -> None:
    events: list[str] = []
    first_error = FakeCudaOutOfMemoryError("A100 exhausted")
    release_record = _release_record()

    def operation(size: int) -> str:
        events.append(f"operation:{size}")
        if size == 8:
            raise first_error
        return "artifact"

    def release() -> VramReleaseRecord:
        events.append("release")
        return release_record

    result = run_with_smaller_batch_retry(
        "semantic",
        8,
        4,
        operation,
        chunk_independent=True,
        release=release,
        torch_module=_fake_torch(FakeCuda([], available=True)),
    )

    assert events == ["operation:8", "release", "operation:4"]
    assert result == BatchRetryResult(
        stage="semantic",
        value="artifact",
        initial_size=8,
        retry_size=4,
        attempts=(
            BatchAttemptRecord(
                size=8,
                outcome="cuda_oom",
                error_type="FakeCudaOutOfMemoryError",
                error_message="A100 exhausted",
            ),
            BatchAttemptRecord(size=4, outcome="succeeded"),
        ),
        release_record=release_record,
    )


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ValueError("non-finite output"), id="numerical"),
        pytest.param(RuntimeError("shape mismatch"), id="shape"),
        pytest.param(ImportError("adapter import failed"), id="import"),
        pytest.param(OSError("checkpoint download failed"), id="download"),
        pytest.param(MemoryError("CPU allocation failed"), id="cpu-memory"),
        pytest.param(
            RuntimeError("CUDA out of memory. but not the injected type"),
            id="message-does-not-override-typed-api",
        ),
    ],
)
def test_non_cuda_oom_failure_escapes_without_release_or_retry(
    error: Exception,
) -> None:
    operation_sizes: list[int] = []
    release_calls: list[str] = []

    def operation(size: int) -> None:
        operation_sizes.append(size)
        raise error

    with pytest.raises(type(error)) as caught:
        run_with_smaller_batch_retry(
            "semantic",
            8,
            4,
            operation,
            chunk_independent=True,
            release=lambda: release_calls.append("release"),  # type: ignore[arg-type]
            torch_module=_fake_torch(FakeCuda([], available=True)),
        )

    assert caught.value is error
    assert operation_sizes == [8]
    assert release_calls == []


def test_legacy_cuda_message_fallback_is_narrow_and_retries_once() -> None:
    events: list[str] = []
    torch_without_oom_type = SimpleNamespace(
        cuda=FakeCuda([], available=True),
    )

    def operation(size: int) -> str:
        events.append(f"operation:{size}")
        if size == 8:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return "artifact"

    result = run_with_smaller_batch_retry(
        "flow",
        8,
        2,
        operation,
        chunk_independent=True,
        release=lambda: events.append("release") or _release_record(),
        torch_module=torch_without_oom_type,
    )

    assert events == ["operation:8", "release", "operation:2"]
    assert result.retry_size == 2
    assert tuple(attempt.outcome for attempt in result.attempts) == (
        "cuda_oom",
        "succeeded",
    )


def test_cuda_words_inside_an_ordinary_runtime_error_do_not_trigger_fallback() -> None:
    error = RuntimeError("wrapper saw CUDA out of memory. downstream failed")
    release_calls: list[str] = []

    with pytest.raises(RuntimeError) as caught:
        run_with_smaller_batch_retry(
            "flow",
            8,
            2,
            lambda _size: (_ for _ in ()).throw(error),
            chunk_independent=True,
            release=lambda: release_calls.append("release"),  # type: ignore[arg-type]
            torch_module=SimpleNamespace(cuda=FakeCuda([], available=True)),
        )

    assert caught.value is error
    assert release_calls == []


def test_release_failure_is_chained_from_initial_cuda_oom_without_retry() -> None:
    initial_error = FakeCudaOutOfMemoryError("initial oom")
    release_error = RuntimeError("release failed")
    operation_sizes: list[int] = []
    release_calls: list[str] = []

    def operation(size: int) -> None:
        operation_sizes.append(size)
        raise initial_error

    def release() -> VramReleaseRecord:
        release_calls.append("release")
        raise release_error

    with pytest.raises(RuntimeError, match="release failed") as caught:
        run_with_smaller_batch_retry(
            "semantic",
            8,
            4,
            operation,
            chunk_independent=True,
            release=release,
            torch_module=_fake_torch(FakeCuda([], available=True)),
        )

    assert caught.value is release_error
    assert caught.value.__cause__ is initial_error
    assert operation_sizes == [8]
    assert release_calls == ["release"]


@pytest.mark.parametrize(
    ("retry_error", "retry_outcome"),
    [
        (FakeCudaOutOfMemoryError("retry oom"), "cuda_oom"),
        (ValueError("retry output invalid"), "failed"),
    ],
)
def test_second_failure_is_reraised_with_two_attempts_and_no_third_call(
    retry_error: Exception,
    retry_outcome: str,
) -> None:
    initial_error = FakeCudaOutOfMemoryError("initial oom")
    operation_sizes: list[int] = []
    release_record = _release_record()
    release_calls: list[str] = []

    def operation(size: int) -> None:
        operation_sizes.append(size)
        if size == 8:
            raise initial_error
        raise retry_error

    def release() -> VramReleaseRecord:
        release_calls.append("release")
        return release_record

    with pytest.raises(type(retry_error)) as caught:
        run_with_smaller_batch_retry(
            "semantic",
            8,
            4,
            operation,
            chunk_independent=True,
            release=release,
            torch_module=_fake_torch(FakeCuda([], available=True)),
        )

    assert caught.value is retry_error
    assert operation_sizes == [8, 4]
    assert release_calls == ["release"]
    attempts = caught.value.batch_retry_attempts  # type: ignore[attr-defined]
    assert tuple(attempt.size for attempt in attempts) == (8, 4)
    assert tuple(attempt.outcome for attempt in attempts) == (
        "cuda_oom",
        retry_outcome,
    )
    assert caught.value.batch_retry_release_record is release_record  # type: ignore[attr-defined]

from __future__ import annotations

import gc
import importlib
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar


T = TypeVar("T")
AttemptOutcome = Literal["succeeded", "cuda_oom", "failed"]


@dataclass(frozen=True)
class VramReleaseRecord:
    cuda_available: bool
    allocated_before_bytes: int | None
    reserved_before_bytes: int | None
    allocated_after_bytes: int | None
    reserved_after_bytes: int | None
    moved_to_cpu: bool
    gc_ran: bool
    cache_cleared: bool


@dataclass(frozen=True)
class BatchAttemptRecord:
    size: int
    outcome: AttemptOutcome
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class BatchRetryResult(Generic[T]):
    stage: str
    value: T
    initial_size: int
    retry_size: int | None
    attempts: tuple[BatchAttemptRecord, ...]
    release_record: VramReleaseRecord | None


def _resolve_torch(torch_module: object | None) -> object:
    if torch_module is not None:
        return torch_module
    return importlib.import_module("torch")


def release_cuda_model(
    model: object | None,
    *,
    torch_module: object | None = None,
) -> VramReleaseRecord:
    """Release this function's model reference; callers must clear their own."""

    torch_api = _resolve_torch(torch_module)
    cuda = getattr(torch_api, "cuda")
    cuda_available = bool(cuda.is_available())
    allocated_before = int(cuda.memory_allocated()) if cuda_available else None
    reserved_before = int(cuda.memory_reserved()) if cuda_available else None
    moved_to_cpu = False
    move_error: BaseException | None = None
    move_traceback = None
    move = getattr(model, "to", None) if model is not None else None
    if callable(move):
        try:
            move("cpu")
            moved_to_cpu = True
        except BaseException as error:
            move_error = error
            move_traceback = error.__traceback__
    del move
    model = None
    cleanup_error: BaseException | None = None
    try:
        gc.collect()
    except BaseException as error:
        cleanup_error = error
    cache_cleared = False
    if cuda_available:
        try:
            cuda.empty_cache()
            cache_cleared = True
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
    allocated_after: int | None = None
    reserved_after: int | None = None
    if cuda_available:
        try:
            allocated_after = int(cuda.memory_allocated())
            reserved_after = int(cuda.memory_reserved())
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
    if move_error is not None:
        if cleanup_error is not None and hasattr(move_error, "add_note"):
            move_error.add_note(f"cleanup also failed: {cleanup_error!r}")
        raise move_error.with_traceback(move_traceback)
    if cleanup_error is not None:
        raise cleanup_error
    return VramReleaseRecord(
        cuda_available=cuda_available,
        allocated_before_bytes=allocated_before,
        reserved_before_bytes=reserved_before,
        allocated_after_bytes=allocated_after,
        reserved_after_bytes=reserved_after,
        moved_to_cpu=moved_to_cpu,
        gc_ran=True,
        cache_cleared=cache_cleared,
    )


def _validate_retry_request(
    stage: str,
    initial_size: int,
    retry_size: int,
    chunk_independent: bool,
) -> None:
    if (
        not isinstance(stage, str)
        or not stage.strip()
        or any(unicodedata.category(character) == "Cc" for character in stage)
    ):
        raise ValueError("stage must be non-empty and control-free")
    if type(initial_size) is not int or initial_size <= 0:
        raise ValueError("initial_size must be a positive plain integer")
    if type(retry_size) is not int or retry_size <= 0:
        raise ValueError("retry_size must be a positive plain integer")
    if retry_size >= initial_size:
        raise ValueError("retry_size must be smaller than initial_size")
    locked_da3_stage = stage.strip()
    if locked_da3_stage in {"da3_anchor", "da3_final_pose"}:
        raise ValueError(
            f"{locked_da3_stage} cannot use a quality-reducing retry"
        )
    if chunk_independent is not True:
        raise ValueError("stage must explicitly declare chunk-independent output")


def _cuda_oom_types(torch_api: object) -> tuple[type[BaseException], ...]:
    candidates = (
        getattr(torch_api, "OutOfMemoryError", None),
        getattr(getattr(torch_api, "cuda", None), "OutOfMemoryError", None),
    )
    result: list[type[BaseException]] = []
    for candidate in candidates:
        if (
            isinstance(candidate, type)
            and issubclass(candidate, BaseException)
            and candidate not in result
        ):
            result.append(candidate)
    return tuple(result)


def _is_cuda_oom(error: BaseException, torch_api: object) -> bool:
    oom_types = _cuda_oom_types(torch_api)
    if oom_types:
        return isinstance(error, oom_types)
    if type(error) is not RuntimeError:
        return False
    message = str(error).strip().casefold()
    return message == "cuda out of memory" or message.startswith(
        "cuda out of memory."
    )


def _failed_attempt(
    size: int,
    outcome: AttemptOutcome,
    error: BaseException,
) -> BatchAttemptRecord:
    try:
        error_message = str(error)
    except BaseException:
        error_message = None
    return BatchAttemptRecord(
        size=size,
        outcome=outcome,
        error_type=type(error).__name__,
        error_message=error_message,
    )


def _attach_retry_failure(
    error: BaseException,
    attempts: tuple[BatchAttemptRecord, ...],
    release_record: VramReleaseRecord,
) -> None:
    try:
        setattr(error, "batch_retry_attempts", attempts)
        setattr(error, "batch_retry_release_record", release_record)
    except BaseException:
        try:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(
                    "batch retry attempts: "
                    + ", ".join(
                        f"{attempt.size}:{attempt.outcome}" for attempt in attempts
                    )
                )
        except BaseException:
            pass


def run_with_smaller_batch_retry(
    stage: str,
    initial_size: int,
    retry_size: int,
    operation: Callable[[int], T],
    *,
    chunk_independent: bool,
    release: Callable[[], VramReleaseRecord],
    torch_module: object | None = None,
) -> BatchRetryResult[T]:
    _validate_retry_request(stage, initial_size, retry_size, chunk_independent)
    try:
        value = operation(initial_size)
    except Exception as initial_error:
        if not isinstance(initial_error, RuntimeError):
            raise
        try:
            torch_api = _resolve_torch(torch_module)
        except Exception as resolution_error:
            raise initial_error from resolution_error
        if not _is_cuda_oom(initial_error, torch_api):
            raise
        initial_attempt = _failed_attempt(
            initial_size,
            "cuda_oom",
            initial_error,
        )
        try:
            release_record = release()
            if not isinstance(release_record, VramReleaseRecord):
                raise TypeError("release must return VramReleaseRecord")
        except Exception as release_error:
            raise release_error from initial_error
        try:
            retry_value = operation(retry_size)
        except Exception as retry_error:
            retry_outcome: AttemptOutcome = (
                "cuda_oom" if _is_cuda_oom(retry_error, torch_api) else "failed"
            )
            attempts = (
                initial_attempt,
                _failed_attempt(retry_size, retry_outcome, retry_error),
            )
            _attach_retry_failure(retry_error, attempts, release_record)
            raise
        return BatchRetryResult(
            stage=stage,
            value=retry_value,
            initial_size=initial_size,
            retry_size=retry_size,
            attempts=(
                initial_attempt,
                BatchAttemptRecord(size=retry_size, outcome="succeeded"),
            ),
            release_record=release_record,
        )
    return BatchRetryResult(
        stage=stage,
        value=value,
        initial_size=initial_size,
        retry_size=None,
        attempts=(BatchAttemptRecord(size=initial_size, outcome="succeeded"),),
        release_record=None,
    )

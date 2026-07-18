from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


_STAGE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_ACTIVE_REPORTER: ContextVar[StageReporter | None]


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    label: str


class StageReporter:
    def __init__(
        self,
        definitions: Sequence[StageDefinition],
        *,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        rows = tuple(definitions)
        stage_ids = tuple(row.stage_id for row in rows)
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("stage IDs must be unique")
        if any(not _STAGE_ID.fullmatch(stage_id) for stage_id in stage_ids):
            raise ValueError("stage IDs must be safe lowercase identifiers")
        if any(not row.label.strip() for row in rows):
            raise ValueError("stage labels must be non-empty")
        self._definitions = rows
        self._by_id = {row.stage_id: row for row in rows}
        self._positions = {
            row.stage_id: index for index, row in enumerate(rows, start=1)
        }
        self._stream = stream or sys.stdout
        self._clock = clock
        self._log_path: Path | None = None
        self._run_id: str | None = None
        self._active_started: dict[str, float] = {}
        self._progress_counts: dict[tuple[str, str], tuple[int, int]] = {}

    def bind_log(self, path: Path, *, run_id: str) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a Path")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8", newline="\n")
        self._log_path = path
        self._run_id = run_id

    @contextmanager
    def stage(
        self,
        stage_id: str,
        *,
        summary: Callable[[], str | None] | None = None,
    ) -> Iterator[None]:
        definition = self._definition(stage_id)
        if stage_id in self._active_started:
            raise ValueError(f"stage is already active: {stage_id}")
        started = self._clock()
        self._active_started[stage_id] = started
        self._emit(
            f"{self._prefix(definition)} START {definition.label}",
            {"stage_id": stage_id, "status": "start"},
        )
        try:
            yield
        except BaseException as error:
            elapsed = self._clock() - started
            self._emit(
                f"{self._prefix(definition)} FAIL  {definition.label} - "
                f"{_format_elapsed(elapsed)} - {error}",
                {
                    "stage_id": stage_id,
                    "status": "fail",
                    "elapsed_seconds": elapsed,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
            self._clear_active_stage(stage_id)
            raise
        elapsed = self._clock() - started
        detail = summary() if summary is not None else None
        detail_text = f" - {detail}" if detail else ""
        self._emit(
            f"{self._prefix(definition)} DONE  {definition.label}"
            f"{detail_text} - {_format_elapsed(elapsed)}",
            {
                "stage_id": stage_id,
                "status": "done",
                "elapsed_seconds": elapsed,
                "summary": detail,
            },
        )
        self._clear_active_stage(stage_id)

    def progress(
        self,
        stage_id: str,
        *,
        substage: str,
        completed: int,
        total: int,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Emit one flushed, strict, monotonic stage-progress event."""
        self._definition(stage_id)
        if stage_id not in self._active_started:
            raise ValueError("progress requires an active stage")
        if not isinstance(substage, str) or not _STAGE_ID.fullmatch(substage):
            raise ValueError("progress substage must be a safe lowercase identifier")
        if type(completed) is not int or type(total) is not int:
            raise TypeError("progress counts must be plain integers")
        if total <= 0 or completed < 0 or completed > total:
            raise ValueError("progress counts must satisfy 0 <= completed <= total")
        key = (stage_id, substage)
        previous = self._progress_counts.get(key)
        if previous is not None and (total != previous[1] or completed < previous[0]):
            raise ValueError("progress counts must be monotonic with a stable total")
        normalized_details = _progress_details(details)
        self._progress_counts[key] = (completed, total)
        elapsed = self._clock() - self._active_started[stage_id]
        percent = completed * 100.0 / total
        visible_stage = stage_id.replace("_", " ").upper()
        visible_substage = substage.replace("_", " ")
        self._emit(
            f"[{visible_stage}] {visible_substage} {completed}/{total} - "
            f"{percent:.1f}% - {_format_elapsed(elapsed)} elapsed",
            {
                "stage_id": stage_id,
                "status": "progress",
                "substage": substage,
                "completed": completed,
                "total": total,
                "percent": percent,
                "elapsed_seconds": elapsed,
                "details": normalized_details,
            },
        )

    def _clear_active_stage(self, stage_id: str) -> None:
        self._active_started.pop(stage_id, None)
        for key in tuple(self._progress_counts):
            if key[0] == stage_id:
                del self._progress_counts[key]

    def cache_event(self, action: str, label: str, detail: str) -> None:
        if not _STAGE_ID.fullmatch(action):
            raise ValueError("cache action must be a safe lowercase identifier")
        if not label.strip() or not detail.strip():
            raise ValueError("cache event label and detail must be non-empty")
        visible_action = action.replace("_", " ").upper()
        self._emit(
            f"[CACHE {visible_action}] {label} - {detail}",
            {
                "status": "cache",
                "cache_action": action,
                "label": label,
                "detail": detail,
            },
        )

    def heartbeat(self, stage_id: str, elapsed_seconds: float) -> None:
        definition = self._definition(stage_id)
        self._emit(
            f"[HEARTBEAT] {definition.label} running - "
            f"{_format_elapsed(elapsed_seconds)} elapsed",
            {
                "stage_id": stage_id,
                "status": "heartbeat",
                "elapsed_seconds": elapsed_seconds,
            },
        )

    def skip(self, stage_id: str, reason: str) -> None:
        definition = self._definition(stage_id)
        if not reason.strip():
            raise ValueError("skip reason must be non-empty")
        self._emit(
            f"{self._prefix(definition)} SKIP  {definition.label} - {reason}",
            {
                "stage_id": stage_id,
                "status": "skipped",
                "reason": reason,
            },
        )

    def _definition(self, stage_id: str) -> StageDefinition:
        try:
            return self._by_id[stage_id]
        except KeyError as error:
            raise ValueError(f"unknown stage ID: {stage_id}") from error

    def _prefix(self, definition: StageDefinition) -> str:
        return (
            f"[STAGE {self._positions[definition.stage_id]:02d}/"
            f"{len(self._definitions):02d}]"
        )

    def _emit(self, message: str, payload: dict[str, object]) -> None:
        self._stream.write(message + "\n")
        self._stream.flush()
        if self._log_path is None:
            return
        row = {
            "schema_version": 1,
            "run_id": self._run_id,
            **payload,
        }
        encoded = json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        with self._log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


def _progress_details(details: Mapping[str, object] | None) -> dict[str, object]:
    if details is None:
        return {}
    if not isinstance(details, Mapping) or any(
        not isinstance(key, str) or not key for key in details
    ):
        raise ValueError("progress details must be finite JSON")

    def validate(value: object) -> object:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("progress details must be finite JSON")
            return value
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) or not key for key in value):
                raise ValueError("progress details must be finite JSON")
            return {key: validate(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [validate(item) for item in value]
        raise ValueError("progress details must be finite JSON")

    return {key: validate(value) for key, value in details.items()}


_ACTIVE_REPORTER = ContextVar("static_pipeline_progress_reporter", default=None)


@contextmanager
def use_stage_reporter(reporter: StageReporter | None) -> Iterator[None]:
    token = _ACTIVE_REPORTER.set(reporter)
    try:
        yield
    finally:
        _ACTIVE_REPORTER.reset(token)


def current_stage_reporter() -> StageReporter | None:
    return _ACTIVE_REPORTER.get()


def run_with_heartbeat(
    command: Sequence[str],
    *,
    stage_id: str,
    check: bool = False,
    process_factory: Callable[..., object] = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
    heartbeat_seconds: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    args = [str(part) for part in command]
    process = process_factory(args)
    started = clock()
    while True:
        try:
            returncode = process.wait(timeout=heartbeat_seconds)
            break
        except subprocess.TimeoutExpired:
            reporter = current_stage_reporter()
            if reporter is not None:
                reporter.heartbeat(stage_id, clock() - started)
    completed = subprocess.CompletedProcess(args, int(returncode))
    if check:
        completed.check_returncode()
    return completed

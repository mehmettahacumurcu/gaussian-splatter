from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from backend.static_pipeline.progress import (
    StageDefinition,
    StageReporter,
    current_stage_reporter,
    run_with_heartbeat,
    use_stage_reporter,
)


class RecordingStream:
    def __init__(self) -> None:
        self.buffer = ""
        self.flush_count = 0

    def write(self, value: str) -> int:
        self.buffer += value
        return len(value)

    def flush(self) -> None:
        self.flush_count += 1

    @property
    def lines(self) -> list[str]:
        return self.buffer.splitlines()


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stage_reporter_flushes_console_and_jsonl(tmp_path: Path) -> None:
    output = RecordingStream()
    times = iter((10.0, 12.5))
    reporter = StageReporter(
        (StageDefinition("selection", "Frame selection"),),
        stream=output,
        clock=lambda: next(times),
    )
    log_path = tmp_path / "progress.jsonl"
    reporter.bind_log(log_path, run_id="run-1")

    with reporter.stage("selection", summary=lambda: "412 frames"):
        pass

    assert output.lines == [
        "[STAGE 01/01] START Frame selection",
        "[STAGE 01/01] DONE  Frame selection - 412 frames - 00:02",
    ]
    assert output.flush_count == 2
    rows = _rows(log_path)
    assert [row["status"] for row in rows] == ["start", "done"]
    assert all(row["run_id"] == "run-1" for row in rows)
    assert rows[-1]["elapsed_seconds"] == 2.5


def test_stage_reporter_records_failure_and_reraises(tmp_path: Path) -> None:
    output = RecordingStream()
    times = iter((20.0, 23.0))
    reporter = StageReporter(
        (StageDefinition("selection", "Frame selection"),),
        stream=output,
        clock=lambda: next(times),
    )
    log_path = tmp_path / "progress.jsonl"
    reporter.bind_log(log_path, run_id="run-2")
    error = RuntimeError("boom")

    with pytest.raises(RuntimeError) as caught:
        with reporter.stage("selection"):
            raise error

    assert caught.value is error
    assert output.lines[-1] == "[STAGE 01/01] FAIL  Frame selection - 00:03 - boom"
    assert _rows(log_path)[-1]["error_type"] == "RuntimeError"


@pytest.mark.parametrize(
    "definitions",
    [
        (StageDefinition("duplicate", "One"), StageDefinition("duplicate", "Two")),
        (StageDefinition("not safe", "Unsafe"),),
    ],
)
def test_stage_reporter_rejects_ambiguous_stage_ids(
    definitions: tuple[StageDefinition, ...],
) -> None:
    with pytest.raises(ValueError):
        StageReporter(definitions)


def test_stage_reporter_emits_cache_heartbeat_and_skip_events(tmp_path: Path) -> None:
    output = RecordingStream()
    reporter = StageReporter(
        (StageDefinition("classical_colmap", "COLMAP mapping"),),
        stream=output,
    )
    log_path = tmp_path / "progress.jsonl"
    reporter.bind_log(log_path, run_id="run-3")

    reporter.cache_event("hit", "COLMAP checkpoint", "/drive/cache/colmap")
    reporter.heartbeat("classical_colmap", 125.0)
    reporter.skip("classical_colmap", "restored from Drive")

    assert output.lines == [
        "[CACHE HIT] COLMAP checkpoint - /drive/cache/colmap",
        "[HEARTBEAT] COLMAP mapping running - 02:05 elapsed",
        "[STAGE 01/01] SKIP  COLMAP mapping - restored from Drive",
    ]
    assert [row["status"] for row in _rows(log_path)] == [
        "cache",
        "heartbeat",
        "skipped",
    ]


def test_active_reporter_is_scoped_and_restored() -> None:
    outer = StageReporter((StageDefinition("outer", "Outer"),))
    inner = StageReporter((StageDefinition("inner", "Inner"),))

    assert current_stage_reporter() is None
    with use_stage_reporter(outer):
        assert current_stage_reporter() is outer
        with use_stage_reporter(inner):
            assert current_stage_reporter() is inner
        assert current_stage_reporter() is outer
    assert current_stage_reporter() is None


class FakeProcess:
    def __init__(self, command: list[str], returncode: int) -> None:
        self.args = command
        self.returncode = returncode
        self.wait_calls = 0

    def wait(self, *, timeout: float) -> int:
        self.wait_calls += 1
        if self.wait_calls < 3:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self.returncode


def test_run_with_heartbeat_preserves_output_and_reports_quiet_process() -> None:
    output = RecordingStream()
    times = iter((0.0, 60.0, 120.0))
    reporter = StageReporter(
        (StageDefinition("classical_colmap", "COLMAP mapping"),),
        stream=output,
    )
    created: list[tuple[list[str], dict[str, object], FakeProcess]] = []

    def process_factory(command: list[str], **kwargs: object) -> FakeProcess:
        process = FakeProcess(command, 0)
        created.append((command, kwargs, process))
        return process

    with use_stage_reporter(reporter):
        completed = run_with_heartbeat(
            ["colmap", "mapper"],
            stage_id="classical_colmap",
            check=True,
            process_factory=process_factory,
            clock=lambda: next(times),
            heartbeat_seconds=60.0,
        )

    assert completed.args == ["colmap", "mapper"]
    assert completed.returncode == 0
    assert created[0][1] == {}
    assert created[0][2].wait_calls == 3
    assert output.lines == [
        "[HEARTBEAT] COLMAP mapping running - 01:00 elapsed",
        "[HEARTBEAT] COLMAP mapping running - 02:00 elapsed",
    ]


def test_run_with_heartbeat_raises_for_nonzero_checked_process() -> None:
    def process_factory(command: list[str], **_kwargs: object) -> FakeProcess:
        process = FakeProcess(command, 7)
        process.wait_calls = 2
        return process

    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_with_heartbeat(
            ["colmap", "mapper"],
            stage_id="classical_colmap",
            check=True,
            process_factory=process_factory,
        )

    assert caught.value.returncode == 7
    assert caught.value.cmd == ["colmap", "mapper"]

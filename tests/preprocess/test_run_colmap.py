from __future__ import annotations

import ast
import importlib
from pathlib import Path

colmap_module = importlib.import_module("backend.preprocess.run_colmap")


def _capture_colmap_commands(
    monkeypatch,
    tmp_path: Path,
    *,
    sequential: bool,
    sequential_overlap: int | None = None,
) -> list[list[str]]:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_000001.png").write_bytes(b"frame")
    output = tmp_path / "colmap"
    commands: list[list[str]] = []

    monkeypatch.setattr(colmap_module, "_resolve_colmap_exe", lambda _value: "colmap")

    def fake_stream(
        command: list[str],
        _parser,
        _on_progress,
        _phase_start: float,
        _phase_end: float,
        _phase_name: str,
    ) -> None:
        commands.append(command)
        if command[1] == "mapper":
            model = output / "sparse" / "0"
            model.mkdir(parents=True)
            (model / "cameras.bin").write_bytes(b"model")

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        commands.append(command)

    monkeypatch.setattr(colmap_module, "_stream_subprocess", fake_stream)
    monkeypatch.setattr(colmap_module.subprocess, "run", fake_run)

    kwargs = (
        {"sequential_overlap": sequential_overlap}
        if sequential_overlap is not None
        else {}
    )
    colmap_module.run_colmap(frames, output, sequential=sequential, **kwargs)
    return commands


def test_sequential_overlap_reaches_colmap_command(
    monkeypatch,
    tmp_path: Path,
) -> None:
    commands = _capture_colmap_commands(
        monkeypatch,
        tmp_path,
        sequential=True,
        sequential_overlap=20,
    )
    matcher = next(command for command in commands if "sequential_matcher" in command)

    assert matcher[-2:] == ["--SequentialMatching.overlap", "20"]


def test_sequential_matcher_uses_legacy_default_overlap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    commands = _capture_colmap_commands(
        monkeypatch,
        tmp_path,
        sequential=True,
    )
    matcher = next(command for command in commands if "sequential_matcher" in command)

    assert matcher[-2:] == ["--SequentialMatching.overlap", "10"]


def test_exhaustive_matcher_does_not_receive_sequential_overlap(
    monkeypatch,
    tmp_path: Path,
) -> None:
    commands = _capture_colmap_commands(
        monkeypatch,
        tmp_path,
        sequential=False,
        sequential_overlap=40,
    )
    matcher = next(command for command in commands if "exhaustive_matcher" in command)

    assert "--SequentialMatching.overlap" not in matcher


def test_pipeline_forwards_configured_sequential_overlap() -> None:
    pipeline_path = Path(__file__).parents[2] / "backend" / "pipeline.py"
    tree = ast.parse(pipeline_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_colmap"
    ]

    assert len(calls) == 1
    overlap = next(
        (
            keyword.value
            for keyword in calls[0].keywords
            if keyword.arg == "sequential_overlap"
        ),
        None,
    )
    assert isinstance(overlap, ast.Attribute)
    assert overlap.attr == "sequential_overlap"
    assert isinstance(overlap.value, ast.Attribute)
    assert overlap.value.attr == "preprocess"
    assert isinstance(overlap.value.value, ast.Name)
    assert overlap.value.value.id == "cfg"

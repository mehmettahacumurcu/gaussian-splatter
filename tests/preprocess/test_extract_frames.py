from __future__ import annotations

import ast
import importlib
from pathlib import Path

extract_module = importlib.import_module("backend.preprocess.extract_frames")


def test_extract_frames_scale_filter_never_upscales(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    output = tmp_path / "frames"
    output.mkdir()
    (output / "frame_0001.png").write_bytes(b"old")
    (output / "frame_9999.png").write_bytes(b"stale-tail")
    seen: dict[str, list[str]] = {}

    monkeypatch.setattr(extract_module, "_check_ffmpeg", lambda: None)

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        seen["command"] = command
        output.mkdir(parents=True, exist_ok=True)
        (output / "frame_0001.png").write_bytes(b"new")

    monkeypatch.setattr(extract_module.subprocess, "run", fake_run)

    frames = extract_module.extract_frames(
        video,
        output,
        fps=4,
        resize_long_edge=1280,
        overwrite=True,
    )

    command = seen["command"]
    vf = command[command.index("-vf") + 1]
    assert "min(iw,1280)" in vf
    assert "min(ih,1280)" in vf
    assert frames == [output / "frame_0001.png"]
    assert not (output / "frame_9999.png").exists()


def test_pipeline_cache_miss_forces_complete_frame_replacement() -> None:
    pipeline_path = Path(__file__).parents[2] / "backend" / "pipeline.py"
    tree = ast.parse(pipeline_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "extract_frames"
    ]

    assert len(calls) == 1
    overwrite = next(
        (keyword.value for keyword in calls[0].keywords if keyword.arg == "overwrite"),
        None,
    )
    assert isinstance(overwrite, ast.Constant)
    assert overwrite.value is True

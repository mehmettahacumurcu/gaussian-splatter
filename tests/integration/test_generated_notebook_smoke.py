from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

from backend.notebooks.builder import build_static_notebook
from backend.notebooks.models import StaticNotebookRunSpec
from backend.notebooks.source import resolve_notebook_source


LEGACY_NOTEBOOKS = (
    "colab/video_to_world.ipynb",
    "colab/sota_verify.ipynb",
    "colab/phase2_verify.ipynb",
)


def test_generated_notebook_safe_cells_execute_with_mocked_boundaries(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    subprocess.run(
        ["git", "diff", "--exit-code", "--", *LEGACY_NOTEBOOKS],
        check=True,
        capture_output=True,
        text=True,
    )
    spec = StaticNotebookRunSpec(
        input_folder="captures/room;__import__('os').system('id')"
    )
    source = resolve_notebook_source()
    notebook = build_static_notebook(spec, source=source)
    assert [cell.metadata["tags"][0] for cell in notebook.cells] == [
        "title",
        "run-spec",
        "preflight",
        "drive",
        "checkout",
        "bootstrap",
        "execute",
        "validate",
        "summary",
    ]

    content_root = tmp_path.as_posix()
    drive_module = types.ModuleType("google.colab.drive")
    mounted: list[str] = []
    drive_module.mount = lambda path: mounted.append(path)
    colab_module = types.ModuleType("google.colab")
    colab_module.drive = drive_module
    google_module = types.ModuleType("google")
    google_module.colab = colab_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)
    monkeypatch.setitem(sys.modules, "google.colab.drive", drive_module)

    cli_calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        args = [str(item) for item in argv]
        if args[0] == "nvidia-smi":
            return subprocess.CompletedProcess(args, 0, "NVIDIA L4, 24576\n", "")
        if args[:2] == [sys.executable, "--version"]:
            return subprocess.CompletedProcess(args, 0, "Python 3.12\n", "")
        if args[:2] == ["git", "clone"]:
            Path(args[-1]).mkdir(parents=True)
            return subprocess.CompletedProcess(args, 0, "", "")
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, source.commit_sha + "\n", "")
        if args[0] == "bash":
            return subprocess.CompletedProcess(args, 0, "", "")
        if "scripts/notebook_static_run.py" in args:
            cli_calls.append(args)
            input_folder = (
                tmp_path
                / "drive"
                / "MyDrive"
                / "captures"
                / "room;__import__('os').system('id')"
            )
            result = input_folder.with_name(input_folder.name + "_result")
            result.mkdir(parents=True)
            for name in (
                "splat.ply",
                "preview.png",
                "quality_report.json",
                "scene_metadata.json",
            ):
                (result / name).write_bytes(b"artifact")
            (result / "_SUCCESS").write_text(
                json.dumps({"run_id": "run-smoke"}), encoding="utf-8"
            )
            (tmp_path / "run_result.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "run_id": "run-smoke",
                        "final_path": str(result),
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: types.SimpleNamespace(free=100 * 1024**3),
    )
    namespace: dict[str, object] = {}
    for cell_index in range(1, 9):
        source_code = notebook.cells[cell_index].source.replace(
            "/content", content_root
        )
        exec(compile(source_code, f"cell-{cell_index}", "exec"), namespace)

    assert [Path(path) for path in mounted] == [tmp_path / "drive"]
    assert len(cli_calls) == 1
    assert cli_calls[0][:3] == [
        sys.executable,
        "scripts/notebook_static_run.py",
        "--spec",
    ]
    assert Path(cli_calls[0][3]) == tmp_path / "run_spec.json"
    assert "room;__import__" not in " ".join(cli_calls[0])
    output = capsys.readouterr().out
    assert "Result folder:" in output
    assert "SuperSplat" in output

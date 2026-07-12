from __future__ import annotations

import ast
from dataclasses import replace

import nbformat
import pytest

from backend.notebooks.builder import (
    build_static_notebook,
    notebook_download_filename,
    serialize_notebook,
)
from backend.notebooks.models import StaticNotebookRunSpec
from backend.notebooks.source import NotebookSource


SOURCE = NotebookSource(
    repo_url="https://github.com/mehmettahacumurcu/gaussian-splatter.git",
    commit_sha="a" * 40,
    generator_id="4dgs-studio.static-notebook",
    generator_version=1,
)


def test_notebook_has_stable_run_all_cells_and_parses() -> None:
    spec = StaticNotebookRunSpec(input_folder="captures/room")
    raw = serialize_notebook(build_static_notebook(spec, source=SOURCE))
    notebook = nbformat.reads(raw.decode("utf-8"), as_version=4)
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


def test_malicious_folder_is_only_json_data() -> None:
    spec = StaticNotebookRunSpec(
        input_folder="captures/room;__import__('os').system('id')"
    )
    notebook = build_static_notebook(spec, source=SOURCE)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert code.count("room;__import__") == 1
    assert "shell=True" not in code
    assert "git pull" not in code
    assert all(not line.lstrip().startswith(("!", "%")) for line in code.splitlines())
    ast.parse(code)


def test_builder_rejects_non_full_commit_sha() -> None:
    with pytest.raises(ValueError, match="40-character"):
        build_static_notebook(
            StaticNotebookRunSpec(input_folder="captures/room"),
            source=replace(SOURCE, commit_sha="main"),
        )


def test_download_filename_is_ascii_and_uses_canonical_leaf() -> None:
    assert notebook_download_filename("captures/room") == "room_static_splat.ipynb"
    assert notebook_download_filename("captures/çalışma") == "calsma_static_splat.ipynb"


def test_checkout_and_execution_use_checked_argv_only() -> None:
    notebook = build_static_notebook(
        StaticNotebookRunSpec(input_folder="captures/room"), source=SOURCE
    )
    checkout = notebook.cells[4].source
    preflight = notebook.cells[2].source
    execute = notebook.cells[6].source
    assert "minimum_vram" in preflight
    assert "import torch" not in preflight
    assert "import numpy" not in preflight
    assert "subprocess.run" in checkout
    assert "check=True" in checkout
    assert "checkout', '--detach'" in checkout
    assert "subprocess.run" in execute
    assert "check=True" in execute

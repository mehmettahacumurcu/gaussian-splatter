from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat

from experiments.learned_quality.floor_recovery_notebook import (
    build_floor_recovery_notebook,
)


ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "colab" / "learned_quality_floor_recovery.ipynb"
PINNED_CODE_COMMIT = "e6ef7aa0268b45bbe1b71f37a21e236accaa67da"


def test_floor_recovery_notebook_is_pinned_bounded_and_run_all_safe() -> None:
    commit = "a" * 40
    notebook = build_floor_recovery_notebook(commit_sha=commit)

    assert [cell.metadata["tags"][0] for cell in notebook.cells] == [
        "title",
        "config",
        "preflight",
        "drive",
        "path-spec",
        "checkout",
        "bootstrap",
        "learned-dependencies",
        "verify",
        "execute",
    ]
    sources = "\n".join(cell.source for cell in notebook.cells)
    assert sources.count('INPUT_FOLDER = ""  # @param {type:"string"}') == 1
    assert sources.count('RUNTIME_PROFILE = "a100_floor_recovery"') == 1
    assert "A100 with at least 75 GiB VRAM required" in notebook.cells[2].source
    assert "At least 100 GiB host RAM required" in notebook.cells[2].source
    assert "L4" not in notebook.cells[2].source
    assert sources.count("drive.mount") == 1
    assert "_training_ablation" in notebook.cells[4].source
    assert "_floor_recovery_diagnostic" in notebook.cells[4].source
    assert "_SUCCESS.json" in notebook.cells[4].source
    assert commit in notebook.cells[5].source
    assert "static_notebook_bootstrap.sh" in notebook.cells[6].source
    assert "download_with_backoff" in notebook.cells[7].source
    assert "max_workers'] = 1" in notebook.cells[7].source
    assert "verify_learned_environment" in notebook.cells[8].source

    execute = notebook.cells[9].source
    assert '"scripts.learned_quality_floor_recovery_run"' in execute
    assert '"--source-revision"' in execute
    assert '"--model-manifest"' in execute
    assert "environment['PYTHONUNBUFFERED'] = '1'" in execute
    assert "check=False" in execute
    assert "decision.json" in execute
    assert "comparison.json" in execute
    assert "candidate.ply" in execute
    assert "No full 120K training was started" in execute
    assert "drive.flush_and_unmount()" in execute
    assert "runtime.unassign()" in execute
    assert execute.find("drive.flush_and_unmount()") < execute.find(
        "runtime.unassign()"
    )
    assert "except BaseException" in execute
    assert "finally:" in execute
    assert "shell=True" not in sources
    assert "viewer.html" not in sources
    assert ".wasm" not in sources
    assert "120000" not in sources


def test_checked_in_floor_recovery_notebook_matches_generator() -> None:
    checked_in = nbformat.read(GENERATED, as_version=4)
    checkout = next(
        cell for cell in checked_in.cells if "checkout" in cell.metadata.get("tags", [])
    )
    match = re.search(r"COMMIT_SHA = ['\"]([0-9a-f]{40})['\"]", checkout.source)

    assert match is not None
    assert match.group(1) == PINNED_CODE_COMMIT

    expected = build_floor_recovery_notebook(commit_sha=PINNED_CODE_COMMIT)
    assert json.loads(nbformat.writes(checked_in)) == json.loads(nbformat.writes(expected))

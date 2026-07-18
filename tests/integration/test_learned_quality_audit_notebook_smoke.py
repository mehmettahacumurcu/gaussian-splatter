from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat

from experiments.learned_quality.notebook import (
    build_learned_quality_audit_notebook,
)


ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "colab" / "learned_quality_cache_audit.ipynb"


def test_cpu_audit_notebook_is_run_all_safe_and_has_no_learned_downloads() -> None:
    commit = "a" * 40
    notebook = build_learned_quality_audit_notebook(commit_sha=commit)

    assert [cell.metadata["tags"][0] for cell in notebook.cells] == [
        "title",
        "config",
        "drive",
        "path-spec",
        "checkout",
        "audit-dependencies",
        "execute",
    ]
    sources = "\n".join(cell.source for cell in notebook.cells)
    assert 'INPUT_FOLDER = ""  # @param {type:"string"}' in sources
    assert "drive.mount" in sources
    assert commit in notebook.cells[4].source
    assert "ffmpeg" in notebook.cells[5].source
    assert "opencv-python-headless" in notebook.cells[5].source
    assert "torch" not in notebook.cells[5].source
    assert "materialize_pinned_assets" not in sources
    execute = notebook.cells[6].source
    assert '"scripts.learned_quality_cache_audit"' in execute
    assert '"-u"' in execute
    assert "TRACK AUDIT PASSED" in execute
    assert "drive.flush_and_unmount()" in execute
    assert "runtime.unassign()" in execute
    assert execute.find("drive.flush_and_unmount()") < execute.find(
        "runtime.unassign()"
    )
    assert "shell=True" not in sources


def test_checked_in_cpu_audit_notebook_matches_generator() -> None:
    checked_in = nbformat.read(GENERATED, as_version=4)
    checkout = next(
        cell for cell in checked_in.cells if cell.metadata["tags"] == ["checkout"]
    )
    match = re.search(r"COMMIT_SHA = '([0-9a-f]{40})'", checkout.source)
    assert match is not None
    expected = build_learned_quality_audit_notebook(commit_sha=match.group(1))
    assert json.loads(nbformat.writes(checked_in)) == json.loads(
        nbformat.writes(expected)
    )

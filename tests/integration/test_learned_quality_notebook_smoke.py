from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import nbformat

from experiments.learned_quality.notebook import build_learned_quality_a100_notebook


ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "colab" / "learned_quality_a100_experiment.ipynb"


def test_notebook_is_deterministic_run_all_safe_and_pinned() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    notebook = build_learned_quality_a100_notebook(commit_sha=commit)
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
    assert "learned_quality_cache_audit.ipynb" in notebook.cells[0].source
    assert sources.count('# @param {type:"string"}') == 1
    assert 'INPUT_FOLDER = ""  # @param {type:"string"}' in sources
    assert notebook.cells[2].source.find("nvidia-smi") >= 0
    assert notebook.cells[2].source.find("A100") >= 0
    assert "drive.mount" in notebook.cells[3].source
    assert "input(" in notebook.cells[4].source
    assert "_learned_test_cache" in notebook.cells[4].source
    assert "Pre-training cache:" in notebook.cells[4].source
    assert "CACHE_PATH" in notebook.cells[4].source
    assert commit in notebook.cells[5].source
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert "static_notebook_bootstrap.sh" in notebook.cells[6].source
    assert "install_learned_environment" in notebook.cells[7].source
    assert "materialize_pinned_assets" in notebook.cells[7].source
    assert "verify_learned_environment" in notebook.cells[8].source
    execute = notebook.cells[9].source
    assert '"/content/learned-env/bin/python"' in execute
    assert '"-u"' in execute
    assert '"-m"' in execute
    assert '"scripts.learned_quality_run"' in execute
    assert '"scripts/learned_quality_run.py"' not in execute
    assert '"--spec"' in execute
    assert "environment['PYTHONUNBUFFERED'] = '1'" in execute
    assert "check=False" in execute
    assert "except BaseException" in execute
    assert "finally:" in execute
    assert "drive.flush_and_unmount()" in execute
    assert "runtime.unassign()" in execute
    assert execute.find("drive.flush_and_unmount()") < execute.find(
        "runtime.unassign()"
    )
    assert "receipt.get('status') == 'success'" in execute
    assert "_SUCCESS" in execute
    assert "_learned_test_result" in execute
    assert "shell=True" not in sources
    for name in (
        "masks_contact_sheet.png",
        "depth_contact_sheet.png",
        "geometry_contact_sheet.png",
        "final_render_contact_sheet.png",
    ):
        assert name in execute
    assert not any(suffix in sources for suffix in ("viewer.html", ".wasm", ".js'"))


def test_checked_in_notebook_matches_generator() -> None:
    checked_in = nbformat.read(GENERATED, as_version=4)
    checkout = next(
        cell for cell in checked_in.cells if cell.metadata["tags"] == ["checkout"]
    )
    match = re.search(r"COMMIT_SHA = '([0-9a-f]{40})'", checkout.source)
    assert match is not None
    expected = build_learned_quality_a100_notebook(commit_sha=match.group(1))
    assert json.loads(nbformat.writes(checked_in)) == json.loads(
        nbformat.writes(expected)
    )

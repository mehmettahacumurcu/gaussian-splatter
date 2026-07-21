from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat

from experiments.learned_quality.ablation_notebook import (
    build_learned_quality_ablation_notebook,
)


ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "colab" / "learned_quality_training_ablation.ipynb"


def test_ablation_notebook_is_pinned_run_all_safe_and_local_first() -> None:
    commit = "964caa3672fcfe575582e9f7abfe8cdedffa3c2d"
    notebook = build_learned_quality_ablation_notebook(commit_sha=commit)

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
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert sources.count('INPUT_FOLDER = ""  # @param {type:"string"}') == 1
    assert sources.count(
        'RUNTIME_PROFILE = "l4_diagnostic"  # @param '
        '["l4_diagnostic", "a100_reference"]'
    ) == 1

    preflight = notebook.cells[2].source
    assert "nvidia-smi" in preflight
    assert "RUNTIME_PROFILE == 'l4_diagnostic'" in preflight
    assert "L4 or A100 with at least 22 GiB VRAM required" in preflight
    assert "RUNTIME_PROFILE == 'a100_reference'" in preflight
    assert "A100 with at least 75 GiB VRAM required" in preflight
    assert "unsupported runtime profile" in preflight
    assert sources.count("drive.mount") == 1
    assert "_training_ablation" in notebook.cells[4].source
    assert "_learned_test_cache" in notebook.cells[4].source
    assert "'runtime_profile': RUNTIME_PROFILE" in notebook.cells[4].source
    assert commit in notebook.cells[5].source
    assert "static_notebook_bootstrap.sh" in notebook.cells[6].source
    assert "install_learned_environment" in notebook.cells[7].source
    assert "verify_learned_environment" in notebook.cells[8].source

    execute = notebook.cells[9].source
    assert '"scripts.learned_quality_ablation_run"' in execute
    assert '"--source-revision"' in execute
    assert '"--model-manifest"' in execute
    assert "COMMIT_SHA" in execute
    assert "environment['PYTHONUNBUFFERED'] = '1'" in execute
    assert "check=False" in execute
    assert "ablation_report.json" in execute
    assert "ablation_summary.md" in execute
    assert "metrics.csv" in execute
    assert "psnr_plot.png" in execute
    assert "_SUCCESS.json" in execute
    assert "drive.flush_and_unmount()" in execute
    assert "runtime.unassign()" in execute
    assert execute.find("drive.flush_and_unmount()") < execute.find(
        "runtime.unassign()"
    )
    assert "except BaseException" in execute
    assert "finally:" in execute
    assert "shell=True" not in sources
    assert "splat.ply" not in sources
    assert "viewer.html" not in sources
    assert ".wasm" not in sources


def test_checked_in_ablation_notebook_matches_generator() -> None:
    checked_in = nbformat.read(GENERATED, as_version=4)
    checkout = next(
        cell for cell in checked_in.cells if cell.metadata["tags"] == ["checkout"]
    )
    match = re.search(r"COMMIT_SHA = '([0-9a-f]{40})'", checkout.source)
    assert match is not None
    expected = build_learned_quality_ablation_notebook(
        commit_sha=match.group(1)
    )
    assert json.loads(nbformat.writes(checked_in)) == json.loads(
        nbformat.writes(expected)
    )

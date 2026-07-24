from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat

from experiments.learned_quality.legacy_control_long_run_notebook import (
    build_legacy_control_long_run_notebook,
)


ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "colab" / "learned_quality_legacy_control_1080p_30k.ipynb"
PINNED_CODE_COMMIT = "0b46f884c048cda6ecf5995426c983eb34c0524a"


def test_local_30k_notebook_is_pinned_native_and_keeps_session_alive() -> None:
    commit = "a" * 40
    notebook = build_legacy_control_long_run_notebook(commit_sha=commit)

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
        "monitor",
    ]
    sources = "\n".join(cell.source for cell in notebook.cells)
    assert sources.count('INPUT_FOLDER = ""  # @param {type:"string"}') == 1
    assert (
        sources.count(
            'RUNTIME_PROFILE = "a100_legacy_control_native_1080p_30k_local"'
        )
        == 1
    )
    assert "1920x1080" in sources
    assert "30000" in sources
    assert "5000" in sources
    assert "10000" in sources
    assert "15000" in sources
    assert "20000" in sources
    assert "25000" in sources
    assert "A100 with at least 75 GiB VRAM required" in notebook.cells[2].source
    assert "At least 100 GiB host RAM required" in notebook.cells[2].source
    assert "At least 120 GiB local disk required" in notebook.cells[2].source
    assert sources.count("drive.mount") == 1
    assert "_training_ablation" in notebook.cells[4].source
    assert "_learned_test_cache" in notebook.cells[4].source
    assert commit in notebook.cells[5].source
    assert "static_notebook_bootstrap.sh" in notebook.cells[6].source
    assert "download_with_backoff" in notebook.cells[7].source
    assert "verify_learned_environment" in notebook.cells[8].source

    execute = notebook.cells[9].source
    assert '"scripts.learned_quality_legacy_control_long_run"' in execute
    assert '"--source-revision"' in execute
    assert '"--model-manifest"' in execute
    assert "check=True" in execute
    assert "drive.flush_and_unmount" not in execute
    assert "runtime.unassign" not in execute
    assert "finally:" not in execute

    monitor = notebook.cells[10].source
    assert "metrics.jsonl" in monitor
    assert "nvidia-smi" in monitor
    assert "legacy_control_005000.ply" in monitor
    assert "legacy_control_030000.pt" in monitor

    for forbidden in ("polish", "report", "receipt"):
        assert forbidden not in sources.casefold()
    assert "shell=True" not in sources
    assert "120000" not in sources
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, f"<notebook:{cell.id}>", "exec")


def test_checked_in_local_30k_notebook_matches_generator() -> None:
    checked_in = nbformat.read(GENERATED, as_version=4)
    checkout = next(
        cell for cell in checked_in.cells if "checkout" in cell.metadata.get("tags", [])
    )
    match = re.search(r"COMMIT_SHA = ['\"]([0-9a-f]{40})['\"]", checkout.source)

    assert match is not None
    assert match.group(1) == PINNED_CODE_COMMIT
    expected = build_legacy_control_long_run_notebook(
        commit_sha=PINNED_CODE_COMMIT
    )
    assert json.loads(nbformat.writes(checked_in)) == json.loads(
        nbformat.writes(expected)
    )

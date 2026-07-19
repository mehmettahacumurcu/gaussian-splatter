from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.learned_quality.contracts import derive_learned_result_path
from experiments.learned_quality.publish import (
    LearnedFileOps,
    LearnedPublishError,
    LearnedPublishOwnershipError,
    is_owned_learned_result,
    publish_learned_diagnostics,
    publish_learned_result,
)
from tests.experiments.learned_quality.test_reports import make_ready_bundle


def test_publish_uses_exact_learned_sibling_and_never_touches_legacy(
    tmp_path: Path,
) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    legacy = tmp_path / "room_result"
    legacy.mkdir()
    (legacy / "keep.txt").write_text("legacy", encoding="utf-8")
    bundle = make_ready_bundle(tmp_path / "new", run_id="run-1")

    receipt = publish_learned_result(bundle, input_folder, run_id="run-1")

    assert receipt.final_path == derive_learned_result_path(input_folder)
    assert is_owned_learned_result(receipt.final_path)
    assert (legacy / "keep.txt").read_text(encoding="utf-8") == "legacy"


def test_publish_success_marker_labels_best_effort_geometry(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    bundle = make_ready_bundle(
        tmp_path / "bundle",
        run_id="run-1",
        geometry_acceptance={
            "mode": "best_effort",
            "policy_version": "output-first-v1",
            "strict_failures": ["registered_ratio"],
            "guarded_metrics": {"registered_count": 658},
            "colmap_fingerprint": "c" * 64,
        },
    )

    receipt = publish_learned_result(bundle, input_folder, run_id="run-1")
    success = json.loads((receipt.final_path / "_SUCCESS").read_text())

    assert success["geometry_acceptance_mode"] == "best_effort"
    assert success["geometry_policy_version"] == "output-first-v1"


def test_publish_refuses_to_replace_unowned_result(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    final = derive_learned_result_path(input_folder)
    final.mkdir()
    (final / "mine.txt").write_text("user", encoding="utf-8")
    bundle = make_ready_bundle(tmp_path / "bundle", run_id="run-1")

    with pytest.raises(LearnedPublishOwnershipError, match="unowned"):
        publish_learned_result(
            bundle,
            input_folder,
            run_id="run-1",
            replace_owned_result=True,
        )

    assert (final / "mine.txt").read_text(encoding="utf-8") == "user"


def test_failed_replacement_rolls_back_owned_result(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    old_bundle = make_ready_bundle(tmp_path / "old", run_id="old-run")
    old = publish_learned_result(old_bundle, input_folder, run_id="old-run")
    old_manifest = (old.final_path / "run_manifest.json").read_bytes()
    new_bundle = make_ready_bundle(tmp_path / "new", run_id="new-run")

    class FailingOps(LearnedFileOps):
        def checkpoint(self, name: str) -> None:
            if name == "final_move":
                raise OSError("injected final failure")

    with pytest.raises(LearnedPublishError, match="final_move"):
        publish_learned_result(
            new_bundle,
            input_folder,
            run_id="new-run",
            replace_owned_result=True,
            file_ops=FailingOps(),
        )

    assert is_owned_learned_result(old.final_path)
    assert (old.final_path / "run_manifest.json").read_bytes() == old_manifest


def test_diagnostics_allowlist_rejects_ply_and_web_assets(tmp_path: Path) -> None:
    input_folder = tmp_path / "room"
    input_folder.mkdir()
    log = tmp_path / "pipeline.log"
    report = tmp_path / "experiment_report.json"
    ply = tmp_path / "splat.ply"
    web = tmp_path / "viewer.html"
    log.write_text("failed", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    ply.write_text("ply", encoding="utf-8")
    web.write_text("viewer", encoding="utf-8")

    destination = publish_learned_diagnostics(
        input_folder,
        "failed-run",
        {"logs/pipeline.log": log, "experiment_report.json": report},
    )
    assert destination == tmp_path / "room_learned_test_diagnostics" / "failed-run"
    assert (destination / "logs" / "pipeline.log").is_file()

    with pytest.raises(ValueError, match="splat.ply"):
        publish_learned_diagnostics(input_folder, "ply-run", {"splat.ply": ply})
    with pytest.raises(ValueError, match="web"):
        publish_learned_diagnostics(input_folder, "web-run", {"viewer.html": web})

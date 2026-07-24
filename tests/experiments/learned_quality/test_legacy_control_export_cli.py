from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import learned_quality_legacy_control_export_run


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "learned_quality_legacy_control_export_run.py"
PIN = "a" * 40


def test_help_is_lazy_and_does_not_import_torch() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--spec" in completed.stdout
    assert "--source-revision" in completed.stdout
    assert "torch" not in completed.stderr.casefold()


def test_invalid_spec_returns_two_and_writes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text('{"input_folder":"../escape"}', encoding="utf-8")
    receipt = tmp_path / "result.json"
    monkeypatch.setattr(
        learned_quality_legacy_control_export_run,
        "RESULT_PATH",
        receipt,
    )

    exit_code = learned_quality_legacy_control_export_run.main(
        ["--spec", str(spec), "--source-revision", PIN]
    )

    assert exit_code == 2
    assert json.loads(receipt.read_text())["status"] == "invalid_spec"


def test_success_writes_both_ply_paths_and_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text('{"input_folder":"myroom_test"}', encoding="utf-8")
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "result.json"
    destination = tmp_path / "myroom_test_legacy_control_5k_result"
    destination.mkdir()
    raw = destination / "raw_legacy_control_5k.ply"
    polished = destination / "polished_legacy_control_5k.ply"
    raw.write_bytes(b"raw")
    polished.write_bytes(b"polished")
    monkeypatch.setattr(
        learned_quality_legacy_control_export_run,
        "RESULT_PATH",
        receipt,
    )
    captured: dict[str, object] = {}

    def runner(active_spec: object, **kwargs: object) -> SimpleNamespace:
        captured.update({"spec": active_spec, **kwargs})
        return SimpleNamespace(
            run_id="run-1",
            final_path=destination,
            raw_ply_path=raw,
            polished_ply_path=polished,
            polish_accepted=False,
            polish_reasons=("mean_psnr_drop",),
        )

    monkeypatch.setattr(
        learned_quality_legacy_control_export_run,
        "_load_runner",
        lambda: runner,
    )

    exit_code = learned_quality_legacy_control_export_run.main(
        [
            "--spec",
            str(spec),
            "--source-revision",
            PIN,
            "--model-manifest",
            str(manifest),
        ]
    )
    payload = json.loads(receipt.read_text())

    assert exit_code == 0
    assert captured["expected_source_revision"] == PIN
    assert captured["model_manifest_path"] == manifest.resolve()
    assert payload["status"] == "success"
    assert payload["raw_ply_path"] == str(raw)
    assert payload["polished_ply_path"] == str(polished)
    assert payload["polish_accepted"] is False
    assert payload["polish_reasons"] == ["mean_psnr_drop"]
    assert payload["full_120k_training_started"] is False


def test_missing_published_ply_is_reported_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text('{"input_folder":"myroom_test"}', encoding="utf-8")
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "result.json"
    failure_root = tmp_path / "failures"
    monkeypatch.setattr(
        learned_quality_legacy_control_export_run,
        "RESULT_PATH",
        receipt,
    )
    monkeypatch.setattr(
        learned_quality_legacy_control_export_run,
        "FAILURE_DRIVE_ROOT",
        failure_root,
    )

    monkeypatch.setattr(
        learned_quality_legacy_control_export_run,
        "_load_runner",
        lambda: lambda *_args, **_kwargs: SimpleNamespace(
            run_id="run-1",
            final_path=tmp_path / "missing-result",
            raw_ply_path=tmp_path / "missing-raw.ply",
            polished_ply_path=tmp_path / "missing-polished.ply",
            polish_accepted=False,
            polish_reasons=("mean_psnr_drop",),
        ),
    )

    exit_code = learned_quality_legacy_control_export_run.main(
        [
            "--spec",
            str(spec),
            "--source-revision",
            PIN,
            "--model-manifest",
            str(manifest),
        ]
    )

    assert exit_code == 1
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "failed"
    assert "published PLY is missing" in payload["error_message"]
    assert Path(payload["durable_failure_path"]).is_file()


def test_runner_exception_writes_durable_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text('{"input_folder":"myroom_test"}', encoding="utf-8")
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "result.json"
    failure_root = tmp_path / "failures"
    monkeypatch.setattr(
        learned_quality_legacy_control_export_run,
        "RESULT_PATH",
        receipt,
    )
    monkeypatch.setattr(
        learned_quality_legacy_control_export_run,
        "FAILURE_DRIVE_ROOT",
        failure_root,
    )

    def fail(*_args: object, **_kwargs: object) -> object:
        error = RuntimeError("polish exploded")
        error.legacy_control_ply_rescue_path = str(
            tmp_path / "myroom_test_legacy_control_5k_result"
        )
        raise error

    monkeypatch.setattr(
        learned_quality_legacy_control_export_run,
        "_load_runner",
        lambda: fail,
    )

    exit_code = learned_quality_legacy_control_export_run.main(
        [
            "--spec",
            str(spec),
            "--source-revision",
            PIN,
            "--model-manifest",
            str(manifest),
        ]
    )

    assert exit_code == 1
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "failed"
    assert payload["error_message"] == "polish exploded"
    assert payload["ply_rescue_path"] == str(
        tmp_path / "myroom_test_legacy_control_5k_result"
    )
    failure_path = Path(payload["durable_failure_path"])
    assert failure_path.is_file()
    durable = json.loads(failure_path.read_text())
    assert durable["full_120k_training_started"] is False
    assert durable["source_revision"] == PIN

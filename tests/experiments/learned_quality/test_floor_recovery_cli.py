from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.learned_quality.floor_recovery_runner import FloorRecoveryDecision
from scripts import learned_quality_floor_recovery_run


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "learned_quality_floor_recovery_run.py"
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
    monkeypatch.setattr(learned_quality_floor_recovery_run, "RESULT_PATH", receipt)

    exit_code = learned_quality_floor_recovery_run.main(
        ["--spec", str(spec), "--source-revision", PIN]
    )

    assert exit_code == 2
    assert json.loads(receipt.read_text())["status"] == "invalid_spec"


@pytest.mark.parametrize("status", ("success", "rejected"))
def test_completed_diagnostic_writes_receipt_and_returns_zero(
    status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text('{"input_folder":"myroom_test"}', encoding="utf-8")
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "result.json"
    monkeypatch.setattr(learned_quality_floor_recovery_run, "RESULT_PATH", receipt)
    captured: dict[str, object] = {}

    def runner(active_spec, **kwargs):
        captured.update({"spec": active_spec, **kwargs})
        return SimpleNamespace(
            status=status,
            run_id="run-1",
            final_path=Path(
                "/content/drive/MyDrive/myroom_test_floor_recovery_diagnostic"
            ),
            decision=FloorRecoveryDecision(
                status == "success",
                "passed" if status == "success" else "quality_gates_failed",
                () if status == "success" else ("floor_alpha_gain",),
            ),
        )

    monkeypatch.setattr(
        learned_quality_floor_recovery_run,
        "_load_runner",
        lambda: runner,
    )

    exit_code = learned_quality_floor_recovery_run.main(
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
    assert payload["status"] == status
    assert payload["full_120k_training_started"] is False
    assert payload["final_path"].endswith("_floor_recovery_diagnostic")


def test_runner_exception_writes_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text('{"input_folder":"myroom_test"}', encoding="utf-8")
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "result.json"
    failure_root = tmp_path / "drive"
    monkeypatch.setattr(learned_quality_floor_recovery_run, "RESULT_PATH", receipt)
    monkeypatch.setattr(
        learned_quality_floor_recovery_run,
        "FAILURE_DRIVE_ROOT",
        failure_root,
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("staging exploded")

    monkeypatch.setattr(
        learned_quality_floor_recovery_run,
        "_load_runner",
        lambda: fail,
    )

    exit_code = learned_quality_floor_recovery_run.main(
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
    assert payload["error_message"] == "staging exploded"
    failure_path = Path(payload["durable_failure_path"])
    assert failure_path.is_file()
    assert failure_path.is_relative_to(failure_root)
    assert json.loads(failure_path.read_text(encoding="utf-8"))["status"] == "failed"

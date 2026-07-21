from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import learned_quality_ablation_run


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "learned_quality_ablation_run.py"
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
    monkeypatch.setattr(learned_quality_ablation_run, "RESULT_PATH", receipt)

    exit_code = learned_quality_ablation_run.main(
        ["--spec", str(spec), "--source-revision", PIN]
    )

    assert exit_code == 2
    assert json.loads(receipt.read_text())["status"] == "invalid_spec"


def test_runtime_profile_is_strictly_parsed_before_invoking_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(
        '{"input_folder":"myroom_test","runtime_profile":"a100_reference"}',
        encoding="utf-8",
    )
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "result.json"
    monkeypatch.setattr(learned_quality_ablation_run, "RESULT_PATH", receipt)
    captured: dict[str, object] = {}

    def runner(active_spec, **_kwargs):
        captured["spec"] = active_spec
        return SimpleNamespace(
            complete=True,
            run_id="run-1",
            final_path=Path("/drive/result"),
            matrix=SimpleNamespace(
                diagnosis=SimpleNamespace(kind="inconclusive", causes=()),
                results={},
                errors={},
            ),
        )

    monkeypatch.setattr(learned_quality_ablation_run, "_load_runner", lambda: runner)

    exit_code = learned_quality_ablation_run.main(
        [
            "--spec",
            str(spec),
            "--source-revision",
            PIN,
            "--model-manifest",
            str(manifest),
        ]
    )

    assert exit_code == 0
    assert captured["spec"].runtime_profile == "a100_reference"


def test_success_receipt_contains_diagnosis_and_result_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text('{"input_folder":"myroom_test"}', encoding="utf-8")
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "result.json"
    monkeypatch.setattr(learned_quality_ablation_run, "RESULT_PATH", receipt)
    result = SimpleNamespace(
        complete=True,
        run_id="run-1",
        final_path=Path("/content/drive/MyDrive/myroom_test_training_ablation"),
        matrix=SimpleNamespace(
            diagnosis=SimpleNamespace(kind="isolated_cause", causes=(("depth",),)),
            results={"legacy_control": object()},
            errors={},
        ),
    )
    captured: dict[str, object] = {}

    def runner(active_spec, **kwargs):
        captured.update({"spec": active_spec, **kwargs})
        return result

    monkeypatch.setattr(learned_quality_ablation_run, "_load_runner", lambda: runner)

    exit_code = learned_quality_ablation_run.main(
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
    assert payload["diagnosis"] == "isolated_cause"
    assert payload["final_path"].endswith("myroom_test_training_ablation")


def test_partial_matrix_returns_one_but_preserves_result_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text('{"input_folder":"myroom_test"}', encoding="utf-8")
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "result.json"
    monkeypatch.setattr(learned_quality_ablation_run, "RESULT_PATH", receipt)
    monkeypatch.setattr(
        learned_quality_ablation_run,
        "_load_runner",
        lambda: lambda *_args, **_kwargs: SimpleNamespace(
            complete=False,
            run_id="run-2",
            final_path=Path("/drive/result"),
            matrix=SimpleNamespace(
                diagnosis=SimpleNamespace(kind="inconclusive", causes=()),
                results={"legacy_control": object()},
                errors={"masks_only": "child failed"},
            ),
        ),
    )

    exit_code = learned_quality_ablation_run.main(
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
    assert payload["status"] == "partial"
    assert payload["errors"] == {"masks_only": "child failed"}


def test_runner_exception_writes_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text('{"input_folder":"myroom_test"}', encoding="utf-8")
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "result.json"
    monkeypatch.setattr(learned_quality_ablation_run, "RESULT_PATH", receipt)

    def fail(*_args, **_kwargs):
        raise RuntimeError("staging exploded")

    monkeypatch.setattr(learned_quality_ablation_run, "_load_runner", lambda: fail)

    exit_code = learned_quality_ablation_run.main(
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

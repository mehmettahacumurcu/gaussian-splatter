from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import notebook_static_run


def test_cli_parses_spec_and_writes_machine_readable_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text(
        '{"schema_version":1,"input_folder":"captures/room"}',
        encoding="utf-8",
    )

    def fake_success(spec):
        assert spec.input_folder == "captures/room"
        return SimpleNamespace(
            run_id="run-1",
            final_path=Path("/content/drive/MyDrive/captures/room_result"),
            local_bundle=Path("/content/4dgs-runs/run-1/bundle"),
            quality_report_path=Path("quality_report.json"),
            manifest_path=Path("run_manifest.json"),
        )

    monkeypatch.setattr(notebook_static_run, "_load_runner", lambda: fake_success)

    assert notebook_static_run.main(["--spec", str(spec_path)]) == 0
    receipt = json.loads((tmp_path / "run_result.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "success"
    assert receipt["final_path"].endswith("room_result")


def test_cli_rejects_malformed_spec_before_loading_runner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text("not-json", encoding="utf-8")
    loaded = []
    monkeypatch.setattr(
        notebook_static_run,
        "_load_runner",
        lambda: loaded.append(True),
    )
    monkeypatch.delitem(sys.modules, "backend.pipeline", raising=False)

    assert notebook_static_run.main(["--spec", str(spec_path)]) == 2
    assert loaded == []
    assert "backend.pipeline" not in sys.modules
    receipt = json.loads((tmp_path / "run_result.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "invalid_spec"


def test_cli_runtime_failure_is_never_reported_as_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec_path = tmp_path / "run_spec.json"
    spec_path.write_text(
        '{"schema_version":1,"input_folder":"captures/room"}',
        encoding="utf-8",
    )

    def fail(_spec):
        raise RuntimeError("COLMAP failed")

    monkeypatch.setattr(notebook_static_run, "_load_runner", lambda: fail)

    assert notebook_static_run.main(["--spec", str(spec_path)]) == 1
    receipt = json.loads((tmp_path / "run_result.json").read_text(encoding="utf-8"))
    assert receipt == {
        "diagnostics_path": None,
        "error_message": "COLMAP failed",
        "error_type": "RuntimeError",
        "status": "failed",
    }

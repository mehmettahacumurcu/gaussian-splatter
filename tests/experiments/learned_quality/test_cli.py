from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import learned_quality_run
from experiments.learned_quality.geometry import OutputFirstGeometryError


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "learned_quality_run.py"


def test_help_is_lazy_and_does_not_import_torch_or_models() -> None:
    code = (
        "import runpy,sys; "
        f"sys.argv={[str(SCRIPT), '--help']!r}; "
        f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--spec" in completed.stdout
    assert "torch" not in completed.stderr.casefold()


def test_invalid_spec_returns_two_and_writes_learned_receipt(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text('{"input_folder":"../escape"}', encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--spec", str(spec)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    receipt = json.loads((tmp_path / "learned_run_result.json").read_text())
    assert completed.returncode == 2
    assert receipt["status"] == "invalid_spec"


def test_guarded_geometry_failure_receipt_exposes_both_failure_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tmp_path / "learned_spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_folder": "myroom_test",
                "recovery_mode": "round0_output_first_v1",
            }
        ),
        encoding="utf-8",
    )
    failure = OutputFirstGeometryError(
        "guard rejected",
        candidates=(),
        strict_failures=("registered_ratio",),
        guarded_failures=("sparse_point_count",),
    )

    def fail_runner(_spec: object) -> None:
        raise failure

    monkeypatch.setattr(learned_quality_run, "_load_runner", lambda: fail_runner)

    exit_code = learned_quality_run.main(["--spec", str(spec)])
    receipt = json.loads((tmp_path / "learned_run_result.json").read_text())

    assert exit_code == 1
    assert receipt["geometry_strict_failures"] == ["registered_ratio"]
    assert receipt["geometry_guarded_failures"] == ["sparse_point_count"]

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import learned_quality_legacy_control_long_run


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "learned_quality_legacy_control_long_run.py"
REVISION = "a" * 40


def _write_spec(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_folder": "myroom_test",
                "runtime_profile": (
                    "a100_legacy_control_native_1080p_30k_local"
                ),
            }
        ),
        encoding="utf-8",
    )


def test_help_is_lazy_and_does_not_import_torch() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "local" in completed.stdout.lower()
    assert "torch" not in completed.stderr.lower()


def test_cli_prints_only_local_snapshot_paths(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    spec_path = tmp_path / "spec.json"
    manifest = tmp_path / "manifest.json"
    _write_spec(spec_path)
    manifest.write_text("{}", encoding="utf-8")
    local_root = tmp_path / "run"
    plys = tuple(
        local_root / "ply" / f"legacy_control_{iteration:06d}.ply"
        for iteration in range(5_000, 30_001, 5_000)
    )
    checkpoints = tuple(
        local_root / "checkpoints" / f"legacy_control_{iteration:06d}.pt"
        for iteration in range(5_000, 30_001, 5_000)
    )
    for path in (*plys, *checkpoints):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    calls: list[object] = []

    def runner(spec: object, **kwargs: object) -> object:
        calls.append((spec, kwargs))
        return SimpleNamespace(
            run_id="run-id",
            local_root=local_root,
            ply_paths=plys,
            checkpoint_paths=checkpoints,
        )

    monkeypatch.setattr(
        learned_quality_legacy_control_long_run,
        "_load_runner",
        lambda: runner,
    )

    code = learned_quality_legacy_control_long_run.main(
        [
            "--spec",
            str(spec_path),
            "--source-revision",
            REVISION,
            "--model-manifest",
            str(manifest),
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert len(calls) == 1
    assert f"LOCAL_RUN_ROOT={local_root.resolve()}" in output
    assert output.count("PLY=") == 6
    assert output.count("CHECKPOINT=") == 6
    assert "drive/MyDrive" not in output
    assert "receipt" not in output.lower()
    assert "report" not in output.lower()


def test_cli_failure_is_nonzero_without_writing_durable_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    spec_path = tmp_path / "spec.json"
    manifest = tmp_path / "manifest.json"
    _write_spec(spec_path)
    manifest.write_text("{}", encoding="utf-8")

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("training stopped")

    monkeypatch.setattr(
        learned_quality_legacy_control_long_run,
        "_load_runner",
        lambda: fail,
    )

    code = learned_quality_legacy_control_long_run.main(
        [
            "--spec",
            str(spec_path),
            "--source-revision",
            REVISION,
            "--model-manifest",
            str(manifest),
        ]
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "training stopped" in captured.err
    assert list(tmp_path.rglob("*.json")) == [manifest, spec_path]

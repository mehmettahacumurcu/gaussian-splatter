from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from PIL import Image

from experiments.learned_quality.contracts import GENERATOR_ID
from experiments.learned_quality.reports import (
    finalize_learned_bundle,
    validate_learned_bundle,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _png(path: Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), (value, value, value)).save(path)


def _base_bundle(tmp_path: Path, run_id: str = "run-1") -> tuple[Path, dict[str, Path]]:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "splat.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 0\n",
        encoding="ascii",
    )
    _png(bundle / "preview.png")
    _json(bundle / "scene_metadata.json", {"schema_version": 1})
    _json(bundle / "quality_report.json", {"schema_version": 1, "status": "passed"})
    _json(bundle / "selection_manifest.json", {"schema_version": 1})
    _json(
        bundle / "run_manifest.json",
        {
            "generator_id": "4dgs-studio.static-notebook",
            "schema_version": 1,
            "run_id": run_id,
            "status": "success",
            "source": {"digest": "a" * 64},
            "repository_commit": "deadbeef",
            "tool_versions": {"python": "3.11"},
            "timing": {"total": 1.0},
            "hardware": {"gpu": "A100"},
            "artifacts": [],
        },
    )
    (bundle / "logs").mkdir()
    (bundle / "logs" / "pipeline.log").write_text("ok\n", encoding="utf-8")

    inputs = tmp_path / "report-inputs"
    model_manifest = inputs / "model_manifest.json"
    photometric = inputs / "photometric_report.json"
    density = inputs / "density_history.json"
    _json(model_manifest, {"models": []})
    _json(photometric, {"decision": "accepted"})
    _json(density, [])
    sheets = {}
    for index, name in enumerate(
        (
            "masks_contact_sheet.png",
            "depth_contact_sheet.png",
            "geometry_contact_sheet.png",
            "final_render_contact_sheet.png",
        )
    ):
        path = inputs / name
        _png(path, 90 + index)
        sheets[name] = path
    return bundle, {
        "model_manifest": model_manifest,
        "photometric": photometric,
        "density": density,
        **sheets,
    }


def make_ready_bundle(tmp_path: Path, run_id: str = "run-1") -> Path:
    bundle, inputs = _base_bundle(tmp_path, run_id)
    finalize_learned_bundle(
        bundle,
        run_id=run_id,
        experiment_report={"schema_version": 1, "status": "passed"},
        geometry_candidates=[{"candidate_id": "classical", "passed": True}],
        model_manifest_path=inputs["model_manifest"],
        contact_sheets={
            name: inputs[name]
            for name in (
                "masks_contact_sheet.png",
                "depth_contact_sheet.png",
                "geometry_contact_sheet.png",
                "final_render_contact_sheet.png",
            )
        },
        density_history_path=inputs["density"],
        photometric_report_path=inputs["photometric"],
    )
    return bundle


def test_finalize_builds_complete_experiment_inventory(tmp_path: Path) -> None:
    bundle = make_ready_bundle(tmp_path)

    inventory = validate_learned_bundle(bundle, run_id="run-1")
    manifest = json.loads((bundle / "run_manifest.json").read_text())
    quality = json.loads((bundle / "quality_report.json").read_text())

    assert manifest["generator_id"] == GENERATOR_ID
    assert manifest["artifacts"] == [asdict(record) for record in inventory]
    assert quality["learned_quality"]["experiment_report"] == "experiment_report.json"
    assert (bundle / "diagnostics" / "density_history.json").is_file()
    assert len(list((bundle / "diagnostics").glob("*_contact_sheet.png"))) == 4


def test_validator_rejects_tampering_symlinks_and_web_assets(tmp_path: Path) -> None:
    bundle = make_ready_bundle(tmp_path)
    (bundle / "experiment_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        validate_learned_bundle(bundle)

    bundle = make_ready_bundle(tmp_path / "second")
    (bundle / "viewer.html").write_text("viewer", encoding="utf-8")
    with pytest.raises(ValueError, match="web asset"):
        validate_learned_bundle(bundle)


def test_finalizer_rejects_missing_contact_sheet(tmp_path: Path) -> None:
    bundle, inputs = _base_bundle(tmp_path)
    with pytest.raises(ValueError, match="contact sheets"):
        finalize_learned_bundle(
            bundle,
            run_id="run-1",
            experiment_report={},
            geometry_candidates=[],
            model_manifest_path=inputs["model_manifest"],
            contact_sheets={},
            density_history_path=inputs["density"],
            photometric_report_path=inputs["photometric"],
        )

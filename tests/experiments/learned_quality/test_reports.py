from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from experiments.learned_quality.contracts import GENERATOR_ID
from backend.static_pipeline.contracts import ModelMetrics
from experiments.learned_quality.contracts import (
    MODEL_TEXT_FILES,
    GeometryAcceptance,
)
from experiments.learned_quality.runner import _geometry_acceptance_payload
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


def make_ready_bundle(
    tmp_path: Path,
    run_id: str = "run-1",
    geometry_acceptance: dict[str, object] | None = None,
) -> Path:
    bundle, inputs = _base_bundle(tmp_path, run_id)
    experiment_report: dict[str, object] = {"schema_version": 1, "status": "passed"}
    if geometry_acceptance is not None:
        experiment_report.update(
            geometry_acceptance_mode=geometry_acceptance["mode"],
            geometry_policy_version=geometry_acceptance["policy_version"],
            geometry_strict_failures=geometry_acceptance["strict_failures"],
            geometry_guarded_metrics=geometry_acceptance.get("guarded_metrics"),
            geometry_colmap_fingerprint=geometry_acceptance["colmap_fingerprint"],
        )
    finalize_learned_bundle(
        bundle,
        run_id=run_id,
        experiment_report=experiment_report,
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


def test_best_effort_acceptance_payload_never_claims_strict_pass() -> None:
    metrics = ModelMetrics(
        model_dir=Path("accepted"),
        registered_names=frozenset({"frame_000000.png"}),
        registered_count=658,
        registered_ratio=658 / 800,
        registered_share=1.0,
        temporal_coverage_s=95.0,
        max_interior_gap_s=9.59,
        start_gap_s=0.0,
        end_gap_s=0.0,
        median_reprojection_error_px=1.217,
        p95_reprojection_error_px=2.194,
        median_track_length=5.0,
        sparse_point_count=65_367,
        valid_names_intrinsics_and_poses=True,
    )
    failures = ("registered_ratio", "interior_gap", "median_reprojection")
    acceptance = GeometryAcceptance(
        policy_version="output-first-v1",
        mode="best_effort",
        selection_digest="a" * 64,
        model_hashes={name: "b" * 64 for name in MODEL_TEXT_FILES},
        strict_failures=failures,
        metrics=metrics,
        checks={"registered_ratio": True},
        colmap_fingerprint="c" * 64,
    )
    reconstruction = SimpleNamespace(
        acceptance=acceptance,
        decision=SimpleNamespace(failures=failures),
    )

    payload = _geometry_acceptance_payload(reconstruction)

    assert payload["geometry_acceptance_mode"] == "best_effort"
    assert payload["geometry_policy_version"] == "output-first-v1"
    assert payload["geometry_strict_failures"] == list(failures)
    assert payload["geometry_colmap_fingerprint"] == "c" * 64
    assert payload["geometry_guarded_metrics"]["registered_count"] == 658


def test_strict_acceptance_payload_has_no_recovery_claims() -> None:
    reconstruction = SimpleNamespace(
        acceptance=None,
        decision=SimpleNamespace(failures=()),
    )

    payload = _geometry_acceptance_payload(reconstruction)

    assert payload == {
        "geometry_acceptance_mode": "strict",
        "geometry_policy_version": "strict-v1",
        "geometry_strict_failures": [],
        "geometry_guarded_metrics": None,
        "geometry_colmap_fingerprint": None,
    }


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

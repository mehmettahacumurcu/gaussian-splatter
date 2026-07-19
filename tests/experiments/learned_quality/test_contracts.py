from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from backend.static_pipeline.contracts import (
    ColmapAttempt,
    GateDecision,
    ModelMetrics,
    ReconstructionBundle,
    SelectionManifest,
)
from experiments.learned_quality.contracts import (
    CACHE_SUFFIX,
    DIAGNOSTICS_SUFFIX,
    GENERATOR_ID,
    RESULT_SUFFIX,
    FrameArtifact,
    GeometryAcceptance,
    GeometryCandidateReport,
    LearnedArtifacts,
    LearnedQualityRunSpec,
    LearnedReconstructionOutput,
    LearnedTrainingOutput,
    MODEL_TEXT_FILES,
    ModelRef,
    StageRecord,
    derive_learned_cache_root,
    derive_learned_diagnostics_root,
    derive_learned_result_path,
    parse_learned_spec_json,
    to_static_run_spec,
)


def _acceptance_metrics(tmp_path: Path) -> ModelMetrics:
    return ModelMetrics(
        model_dir=tmp_path / "model",
        registered_names=frozenset({"frame_000000.png"}),
        registered_count=1,
        registered_ratio=0.8,
        registered_share=1.0,
        temporal_coverage_s=1.0,
        max_interior_gap_s=1.0,
        start_gap_s=0.0,
        end_gap_s=0.0,
        median_reprojection_error_px=1.2,
        p95_reprojection_error_px=2.0,
        median_track_length=5.0,
        sparse_point_count=10_000,
        valid_names_intrinsics_and_poses=True,
    )


def test_static_conversion_is_locked() -> None:
    learned = LearnedQualityRunSpec(input_folder="captures/room")
    static = to_static_run_spec(learned)

    assert static.input_folder == "captures/room"
    assert static.frame_selection.mode.value == "smart"
    assert static.quality.profile.value == "ultra"
    assert static.quality.n_iters == 120_000
    assert static.quality.max_gaussians == 6_000_000
    assert static.quality.advanced.density_start_iter == 500
    assert static.quality.advanced.density_end_iter == 80_000
    assert static.quality.advanced.density_interval == 100
    assert static.quality.advanced.densify_grad_threshold == pytest.approx(1e-4)
    assert static.publish.replace_owned_result is True


def test_learned_spec_defaults_to_strict_recovery_mode() -> None:
    spec = LearnedQualityRunSpec(input_folder="room")

    assert spec.recovery_mode == "strict"


def test_learned_spec_accepts_only_named_round0_recovery_mode() -> None:
    spec = LearnedQualityRunSpec(
        input_folder="room",
        recovery_mode="round0_output_first_v1",
    )

    assert spec.recovery_mode == "round0_output_first_v1"
    with pytest.raises(ValidationError):
        LearnedQualityRunSpec(
            input_folder="room",
            recovery_mode="best_effort",  # type: ignore[arg-type]
        )


def test_geometry_acceptance_freezes_hashes_and_checks(tmp_path: Path) -> None:
    acceptance = GeometryAcceptance(
        policy_version="output-first-v1",
        mode="best_effort",
        selection_digest="a" * 64,
        model_hashes={name: "b" * 64 for name in MODEL_TEXT_FILES},
        strict_failures=("registered_ratio",),
        metrics=_acceptance_metrics(tmp_path),
        checks={"registered_ratio": True},
        colmap_fingerprint="c" * 64,
    )

    assert tuple(acceptance.model_hashes) == tuple(sorted(MODEL_TEXT_FILES))
    with pytest.raises(TypeError):
        acceptance.model_hashes["images.txt"] = "d" * 64  # type: ignore[index]
    with pytest.raises(TypeError):
        acceptance.checks["registered_ratio"] = False  # type: ignore[index]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selection_digest", "not-a-digest"),
        ("colmap_fingerprint", "D" * 64),
    ],
)
def test_geometry_acceptance_rejects_invalid_identity_digests(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    kwargs = {
        "policy_version": "output-first-v1",
        "mode": "best_effort",
        "selection_digest": "a" * 64,
        "model_hashes": {name: "b" * 64 for name in MODEL_TEXT_FILES},
        "strict_failures": ("registered_ratio",),
        "metrics": _acceptance_metrics(tmp_path),
        "checks": {"registered_ratio": True},
        "colmap_fingerprint": "c" * 64,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        GeometryAcceptance(**kwargs)  # type: ignore[arg-type]


def test_static_conversion_forwards_publish_replacement_choice() -> None:
    learned = LearnedQualityRunSpec(
        input_folder="captures/room",
        publish={"replace_owned_result": False},
    )

    assert to_static_run_spec(learned).publish.replace_owned_result is False


def test_parse_learned_spec_json_normalizes_drive_relative_folder() -> None:
    spec = parse_learned_spec_json(
        b'{"schema_version":1,"input_folder":"MyDrive/captures/room",'
        b'"publish":{"replace_owned_result":false}}'
    )

    assert spec.input_folder == "captures/room"
    assert spec.publish.replace_owned_result is False


def test_learned_spec_is_strict() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        parse_learned_spec_json(
            '{"schema_version":1,"input_folder":"captures/room",'
            '"quality":{"profile":"ultra"}}'
        )


@pytest.mark.parametrize(
    "input_folder",
    [
        "/captures/room",
        "C:/captures/room",
        "captures/../room",
        r"captures\room",
        "captures/room\x00",
        "MyDrive",
        f"room{RESULT_SUFFIX}",
        f"captures/room{RESULT_SUFFIX}",
        f"MyDrive/captures/room{RESULT_SUFFIX}",
        f"room{DIAGNOSTICS_SUFFIX}",
        f"captures/room{DIAGNOSTICS_SUFFIX}",
        f"MyDrive/captures/room{DIAGNOSTICS_SUFFIX}",
        f"room{CACHE_SUFFIX}",
        f"captures/room{CACHE_SUFFIX}",
        f"MyDrive/captures/room{CACHE_SUFFIX}",
    ],
)
def test_learned_spec_rejects_unsafe_or_output_folders(input_folder: str) -> None:
    with pytest.raises(ValidationError):
        LearnedQualityRunSpec(input_folder=input_folder)


@pytest.mark.parametrize(
    "control",
    [
        pytest.param("\t", id="tab"),
        pytest.param("\n", id="lf"),
        pytest.param("\r", id="cr"),
    ],
)
@pytest.mark.parametrize(
    "template",
    [
        pytest.param("{}captures/room", id="leading"),
        pytest.param("captures/room{}", id="trailing"),
    ],
)
def test_learned_spec_rejects_boundary_control_characters(
    control: str,
    template: str,
) -> None:
    with pytest.raises(ValidationError):
        LearnedQualityRunSpec(input_folder=template.format(control))


def test_paths_are_exact_siblings(tmp_path: Path) -> None:
    source = tmp_path / "2026-11-room"

    assert derive_learned_result_path(source) == tmp_path / (
        "2026-11-room_learned_test_result"
    )
    assert derive_learned_diagnostics_root(source) == tmp_path / (
        "2026-11-room_learned_test_diagnostics"
    )
    assert derive_learned_cache_root(source) == tmp_path / (
        "2026-11-room_learned_test_cache"
    )


def test_constants_are_locked() -> None:
    assert RESULT_SUFFIX == "_learned_test_result"
    assert DIAGNOSTICS_SUFFIX == "_learned_test_diagnostics"
    assert CACHE_SUFFIX == "_learned_test_cache"
    assert GENERATOR_ID == "4dgs-studio.learned-quality-a100"


def test_artifact_and_stage_contract_defaults_are_frozen(tmp_path: Path) -> None:
    model = ModelRef(
        repo_id="depth-anything/DA3-BASE",
        revision="f" * 40,
        code_commit="3" * 40,
        license_id="Apache-2.0",
    )
    frame = FrameArtifact(
        image_name="frame_000000.png",
        frame_id="frame-000000",
        path=tmp_path / "frame_000000.png",
        sha256="a" * 64,
    )
    stage = StageRecord(stage_id="da3-anchors", status="accepted")
    artifacts = LearnedArtifacts(stage_records=(stage,))

    assert model.code_commit == "3" * 40
    assert frame.image_name == "frame_000000.png"
    assert stage.details == {}
    assert artifacts.stage_records == (stage,)
    with pytest.raises(FrozenInstanceError):
        stage.status = "failed"  # type: ignore[misc]


def test_geometry_candidate_contract_has_locked_fields(tmp_path: Path) -> None:
    attempt = cast(ColmapAttempt, object())
    decision = cast(GateDecision, object())
    manifest = cast(SelectionManifest, object())

    candidate = GeometryCandidateReport(
        candidate_id="classical",
        attempt=attempt,
        decision=decision,
        model_dir=tmp_path / "model",
        selected_manifest=manifest,
        frame_set_digest="b" * 64,
        covered_endpoint_count=2,
    )

    assert candidate.attempt is attempt
    assert candidate.decision is decision
    assert candidate.selected_manifest is manifest


def test_reconstruction_output_forwards_bundle_contract(tmp_path: Path) -> None:
    decision = object()
    manifest = object()
    model_dir = tmp_path / "validated"
    attempts = (object(),)
    decisions = (decision,)
    bundle = cast(
        ReconstructionBundle,
        SimpleNamespace(
            decision=decision,
            selected_manifest=manifest,
            accepted_model_dir=model_dir,
            attempts=attempts,
            decisions=decisions,
        ),
    )
    output = LearnedReconstructionOutput(
        bundle=bundle,
        frames_dir=tmp_path / "frames",
        artifacts=LearnedArtifacts(),
        geometry_candidates=(),
    )

    assert output.decision is decision
    assert output.selected_manifest is manifest
    assert output.accepted_model_dir == model_dir
    assert output.attempts is attempts
    assert output.decisions is decisions
    with pytest.raises(FrozenInstanceError):
        output.frames_dir = tmp_path / "other"  # type: ignore[misc]


def test_training_output_contract_is_frozen(tmp_path: Path) -> None:
    resolved = object()
    output = LearnedTrainingOutput(
        raw_ply_path=tmp_path / "splat.ply",
        status={"state": "complete"},
        resolved_config=resolved,  # type: ignore[arg-type]
        run_manifest_path=tmp_path / "run_manifest.json",
        density_history_path=tmp_path / "density_history.json",
        final_gaussian_count=6_000_000,
        training_rgb_digest="c" * 64,
    )

    assert output.resolved_config is resolved
    assert output.fallbacks == ()
    with pytest.raises(FrozenInstanceError):
        output.final_gaussian_count = 1  # type: ignore[misc]

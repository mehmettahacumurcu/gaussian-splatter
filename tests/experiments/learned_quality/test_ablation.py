from __future__ import annotations

from experiments.learned_quality.ablation import (
    AblationCheckpoint,
    ExperimentOutcome,
    StructuralMetrics,
    classify_experiments,
    evaluate_checkpoint,
    pairwise_variants,
    primary_variants,
)


FEATURES = frozenset({"dense_seeds", "masks", "depth", "adaptive_density"})


def _checkpoint(
    experiment_id: str,
    iteration: int,
    *,
    psnr: float = 20.0,
    white: float = 0.01,
    anisotropic: float = 0.0,
    oversized: float = 0.0,
    nonfinite: int = 0,
) -> AblationCheckpoint:
    return AblationCheckpoint(
        experiment_id=experiment_id,
        iteration=iteration,
        psnr_unmasked=psnr,
        psnr_masked=psnr,
        ssim_unmasked=0.8,
        ssim_masked=0.8,
        l1_unmasked=0.1,
        l1_masked=0.1,
        gaussian_count=10_000,
        structural=StructuralMetrics(
            visible_white_fraction=white,
            high_anisotropy_fraction=anisotropic,
            oversized_fraction=oversized,
            nonfinite_count=nonfinite,
        ),
    )


def test_primary_matrix_changes_one_feature_at_a_time() -> None:
    variants = primary_variants()

    assert tuple(variant.experiment_id for variant in variants) == (
        "legacy_control",
        "fixed_topology_control",
        "dense_seeds_only",
        "masks_only",
        "depth_only",
        "adaptive_density_only",
        "full_learned",
    )
    assert variants[0].features == frozenset()
    assert variants[0].density_events is True
    assert variants[1].features == frozenset()
    assert variants[1].density_events is False
    assert tuple(variant.features for variant in variants[2:6]) == (
        frozenset({"dense_seeds"}),
        frozenset({"masks"}),
        frozenset({"depth"}),
        frozenset({"adaptive_density"}),
    )
    assert variants[-1].features == FEATURES
    assert all(variant.n_iterations == 5_000 for variant in variants)
    assert all(
        variant.checkpoints == (0, 100, 499, 500, 600, 1_000, 2_500, 5_000)
        for variant in variants
    )


def test_pairwise_matrix_is_complete_and_deterministic() -> None:
    variants = pairwise_variants()

    assert len(variants) == 6
    assert tuple(variant.experiment_id for variant in variants) == (
        "dense_seeds__masks",
        "dense_seeds__depth",
        "dense_seeds__adaptive_density",
        "masks__depth",
        "masks__adaptive_density",
        "depth__adaptive_density",
    )
    assert {variant.features for variant in variants} == {
        frozenset(pair)
        for pair in (
            ("dense_seeds", "masks"),
            ("dense_seeds", "depth"),
            ("dense_seeds", "adaptive_density"),
            ("masks", "depth"),
            ("masks", "adaptive_density"),
            ("depth", "adaptive_density"),
        )
    }
    assert all(variant.n_iterations == 2_500 for variant in variants)
    assert all(variant.checkpoints == (500, 1_000, 2_500) for variant in variants)


def test_nonfinite_structural_metrics_abort_immediately() -> None:
    decision = evaluate_checkpoint(
        _checkpoint("dense_seeds_only", 500, nonfinite=1),
        control=_checkpoint("legacy_control", 500),
    )

    assert decision.stop
    assert decision.reasons == ("nonfinite_values",)


def test_structural_issues_are_telemetry_before_final_checkpoint() -> None:
    decision = evaluate_checkpoint(
        _checkpoint(
            "full_learned",
            500,
            white=0.16,
            anisotropic=0.006,
            oversized=0.006,
        ),
        control=_checkpoint("legacy_control", 500, white=0.10),
    )

    assert not decision.stop
    assert decision.reasons == (
        "visible_white_fraction",
        "anisotropy_fraction",
        "oversized_fraction",
    )


def test_persistent_structural_issues_fail_only_at_the_5k_endpoint() -> None:
    first = evaluate_checkpoint(
        _checkpoint("full_learned", 500, oversized=0.02),
        control=_checkpoint("legacy_control", 500),
    )
    second = evaluate_checkpoint(
        _checkpoint("full_learned", 600, oversized=0.02),
        control=_checkpoint("legacy_control", 600),
        previous_structural_streaks=first.structural_streaks,
    )
    final = evaluate_checkpoint(
        _checkpoint("full_learned", 5_000, oversized=0.02),
        control=_checkpoint("legacy_control", 5_000),
        previous_structural_streaks=second.structural_streaks,
    )

    assert not first.stop
    assert not second.stop
    assert final.stop
    assert final.reasons == ("oversized_fraction",)
    assert final.structural_streaks["oversized_fraction"] == 3


def test_severe_structural_event_still_aborts_immediately() -> None:
    decision = evaluate_checkpoint(
        _checkpoint("full_learned", 500, oversized=0.10),
        control=_checkpoint("legacy_control", 500),
    )

    assert decision.stop
    assert decision.reasons == ("severe_oversized_fraction",)


def test_psnr_gate_requires_two_consecutive_deficits_after_iteration_1000() -> None:
    first = evaluate_checkpoint(
        _checkpoint("depth_only", 1_000, psnr=16.9),
        control=_checkpoint("legacy_control", 1_000, psnr=20.0),
    )
    second = evaluate_checkpoint(
        _checkpoint("depth_only", 2_500, psnr=17.0),
        control=_checkpoint("legacy_control", 2_500, psnr=20.0),
        previous_psnr_deficits=first.consecutive_psnr_deficits,
    )
    final = evaluate_checkpoint(
        _checkpoint("depth_only", 5_000, psnr=17.0),
        control=_checkpoint("legacy_control", 5_000, psnr=20.0),
        previous_psnr_deficits=second.consecutive_psnr_deficits,
    )

    assert not first.stop
    assert first.consecutive_psnr_deficits == 1
    assert not second.stop
    assert second.consecutive_psnr_deficits == 2
    assert final.stop
    assert final.consecutive_psnr_deficits == 3
    assert final.reasons == ("psnr_deficit",)


def test_legacy_control_must_be_structurally_sound_and_reach_15db() -> None:
    decision = evaluate_checkpoint(
        _checkpoint("legacy_control", 5_000, psnr=14.99),
        control=None,
    )

    assert decision.stop
    assert decision.reasons == ("unusable_control_psnr",)


def test_classifier_finds_one_isolated_cause_without_pairwise_runs() -> None:
    outcomes = {
        variant.experiment_id: ExperimentOutcome(
            variant.experiment_id,
            passed=variant.experiment_id != "depth_only",
        )
        for variant in primary_variants()
    }
    outcomes["full_learned"] = ExperimentOutcome("full_learned", passed=False)

    diagnosis = classify_experiments(outcomes)

    assert diagnosis.kind == "isolated_cause"
    assert diagnosis.causes == (("depth",),)
    assert not diagnosis.requires_pairwise


def test_classifier_finds_multiple_independent_causes() -> None:
    outcomes = {
        variant.experiment_id: ExperimentOutcome(variant.experiment_id, passed=True)
        for variant in primary_variants()
    }
    outcomes["masks_only"] = ExperimentOutcome("masks_only", passed=False)
    outcomes["depth_only"] = ExperimentOutcome("depth_only", passed=False)
    outcomes["full_learned"] = ExperimentOutcome("full_learned", passed=False)

    diagnosis = classify_experiments(outcomes)

    assert diagnosis.kind == "multiple_independent_causes"
    assert diagnosis.causes == (("masks",), ("depth",))
    assert not diagnosis.requires_pairwise


def test_classifier_requests_pairwise_then_finds_interaction() -> None:
    outcomes = {
        variant.experiment_id: ExperimentOutcome(
            variant.experiment_id,
            passed=variant.experiment_id != "full_learned",
        )
        for variant in primary_variants()
    }

    before_pairs = classify_experiments(outcomes)
    assert before_pairs.kind == "inconclusive"
    assert before_pairs.requires_pairwise

    outcomes.update(
        {
            variant.experiment_id: ExperimentOutcome(
                variant.experiment_id,
                passed=variant.experiment_id != "dense_seeds__depth",
            )
            for variant in pairwise_variants()
        }
    )
    after_pairs = classify_experiments(outcomes)

    assert after_pairs.kind == "interaction_cause"
    assert after_pairs.causes == (("dense_seeds", "depth"),)
    assert not after_pairs.requires_pairwise


def test_classifier_is_inconclusive_when_control_or_full_run_is_unusable() -> None:
    assert classify_experiments({}).kind == "inconclusive"
    diagnosis = classify_experiments(
        {
            "legacy_control": ExperimentOutcome("legacy_control", passed=False),
            "full_learned": ExperimentOutcome("full_learned", passed=False),
        }
    )
    assert diagnosis.kind == "inconclusive"
    assert not diagnosis.requires_pairwise


def test_classifier_identifies_density_transition_when_fixed_topology_survives() -> None:
    outcomes = {
        variant.experiment_id: ExperimentOutcome(variant.experiment_id, passed=True)
        for variant in primary_variants()
    }
    outcomes["legacy_control"] = ExperimentOutcome("legacy_control", passed=False)
    outcomes["fixed_topology_control"] = ExperimentOutcome(
        "fixed_topology_control",
        passed=True,
    )

    diagnosis = classify_experiments(outcomes)

    assert diagnosis.kind == "density_transition_failure"
    assert diagnosis.causes == (("density_events",),)
    assert not diagnosis.requires_pairwise

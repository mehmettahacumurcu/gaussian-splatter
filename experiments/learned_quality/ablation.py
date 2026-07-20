from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Literal, Mapping


AblationFeature = Literal[
    "dense_seeds",
    "masks",
    "depth",
    "adaptive_density",
]
DiagnosisKind = Literal[
    "isolated_cause",
    "interaction_cause",
    "multiple_independent_causes",
    "inconclusive",
]

FEATURE_ORDER: tuple[AblationFeature, ...] = (
    "dense_seeds",
    "masks",
    "depth",
    "adaptive_density",
)
PRIMARY_CHECKPOINTS = (500, 1_000, 2_500, 5_000)
PAIRWISE_CHECKPOINTS = (500, 1_000, 2_500)


@dataclass(frozen=True)
class AblationVariant:
    experiment_id: str
    features: frozenset[AblationFeature]
    n_iterations: int
    checkpoints: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id cannot be empty")
        if not self.features.issubset(FEATURE_ORDER):
            raise ValueError("unknown ablation feature")
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if (
            not self.checkpoints
            or tuple(sorted(set(self.checkpoints))) != self.checkpoints
            or self.checkpoints[-1] != self.n_iterations
        ):
            raise ValueError("checkpoints must be unique, ordered, and end at n_iterations")


@dataclass(frozen=True)
class StructuralMetrics:
    visible_white_fraction: float
    high_anisotropy_fraction: float
    oversized_fraction: float
    nonfinite_count: int

    def __post_init__(self) -> None:
        for name in (
            "visible_white_fraction",
            "high_anisotropy_fraction",
            "oversized_fraction",
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.nonfinite_count < 0:
            raise ValueError("nonfinite_count cannot be negative")


@dataclass(frozen=True)
class AblationCheckpoint:
    experiment_id: str
    iteration: int
    psnr_unmasked: float
    psnr_masked: float
    ssim_unmasked: float
    ssim_masked: float
    l1_unmasked: float
    l1_masked: float
    gaussian_count: int
    structural: StructuralMetrics


@dataclass(frozen=True)
class AblationGatePolicy:
    absolute_white_fraction: float = 0.10
    control_white_margin: float = 0.05
    high_anisotropy_fraction: float = 0.005
    oversized_fraction: float = 0.005
    psnr_deficit_db: float = 3.0
    psnr_deficit_start_iteration: int = 1_000
    psnr_deficit_patience: int = 2
    control_min_psnr_db: float = 15.0
    control_final_iteration: int = 5_000


@dataclass(frozen=True)
class GateDecision:
    stop: bool
    reasons: tuple[str, ...]
    consecutive_psnr_deficits: int = 0

    @property
    def passed(self) -> bool:
        return not self.stop


@dataclass(frozen=True)
class ExperimentOutcome:
    experiment_id: str
    passed: bool


@dataclass(frozen=True)
class AblationDiagnosis:
    kind: DiagnosisKind
    causes: tuple[tuple[AblationFeature, ...], ...] = ()
    requires_pairwise: bool = False


def primary_variants() -> tuple[AblationVariant, ...]:
    isolated = tuple(
        AblationVariant(
            experiment_id=f"{feature}_only",
            features=frozenset({feature}),
            n_iterations=5_000,
            checkpoints=PRIMARY_CHECKPOINTS,
        )
        for feature in FEATURE_ORDER
    )
    return (
        AblationVariant(
            experiment_id="legacy_control",
            features=frozenset(),
            n_iterations=5_000,
            checkpoints=PRIMARY_CHECKPOINTS,
        ),
        *isolated,
        AblationVariant(
            experiment_id="full_learned",
            features=frozenset(FEATURE_ORDER),
            n_iterations=5_000,
            checkpoints=PRIMARY_CHECKPOINTS,
        ),
    )


def pairwise_variants() -> tuple[AblationVariant, ...]:
    return tuple(
        AblationVariant(
            experiment_id=f"{left}__{right}",
            features=frozenset({left, right}),
            n_iterations=2_500,
            checkpoints=PAIRWISE_CHECKPOINTS,
        )
        for left, right in combinations(FEATURE_ORDER, 2)
    )


def evaluate_checkpoint(
    current: AblationCheckpoint,
    *,
    control: AblationCheckpoint | None,
    previous_psnr_deficits: int = 0,
    policy: AblationGatePolicy = AblationGatePolicy(),
) -> GateDecision:
    reasons: list[str] = []
    structural = current.structural

    if structural.nonfinite_count:
        reasons.append("nonfinite_values")

    white_limit = policy.absolute_white_fraction
    if control is not None:
        white_limit = max(
            white_limit,
            control.structural.visible_white_fraction + policy.control_white_margin,
        )
    if structural.visible_white_fraction > white_limit:
        reasons.append("visible_white_fraction")
    if structural.high_anisotropy_fraction > policy.high_anisotropy_fraction:
        reasons.append("anisotropy_fraction")
    if structural.oversized_fraction > policy.oversized_fraction:
        reasons.append("oversized_fraction")

    consecutive_deficits = 0
    if control is not None and current.iteration >= policy.psnr_deficit_start_iteration:
        if control.psnr_unmasked - current.psnr_unmasked >= policy.psnr_deficit_db:
            consecutive_deficits = previous_psnr_deficits + 1
        if consecutive_deficits >= policy.psnr_deficit_patience:
            reasons.append("psnr_deficit")

    if (
        current.experiment_id == "legacy_control"
        and current.iteration == policy.control_final_iteration
        and current.psnr_unmasked < policy.control_min_psnr_db
    ):
        reasons.append("unusable_control_psnr")

    return GateDecision(
        stop=bool(reasons),
        reasons=tuple(reasons),
        consecutive_psnr_deficits=consecutive_deficits,
    )


def classify_experiments(
    outcomes: Mapping[str, ExperimentOutcome],
) -> AblationDiagnosis:
    control = outcomes.get("legacy_control")
    full = outcomes.get("full_learned")
    if control is None or full is None or not control.passed:
        return AblationDiagnosis("inconclusive")
    if full.passed:
        return AblationDiagnosis("inconclusive")

    feature_by_isolated_id = {
        f"{feature}_only": feature for feature in FEATURE_ORDER
    }
    if any(experiment_id not in outcomes for experiment_id in feature_by_isolated_id):
        return AblationDiagnosis("inconclusive")
    isolated_causes = tuple(
        (feature,)
        for experiment_id, feature in feature_by_isolated_id.items()
        if not outcomes[experiment_id].passed
    )
    if len(isolated_causes) == 1:
        return AblationDiagnosis("isolated_cause", causes=isolated_causes)
    if len(isolated_causes) > 1:
        return AblationDiagnosis(
            "multiple_independent_causes",
            causes=isolated_causes,
        )

    pairs = pairwise_variants()
    if any(variant.experiment_id not in outcomes for variant in pairs):
        return AblationDiagnosis("inconclusive", requires_pairwise=True)
    interaction_causes = tuple(
        tuple(feature for feature in FEATURE_ORDER if feature in variant.features)
        for variant in pairs
        if not outcomes[variant.experiment_id].passed
    )
    if interaction_causes:
        return AblationDiagnosis("interaction_cause", causes=interaction_causes)
    return AblationDiagnosis("inconclusive")

from __future__ import annotations

from dataclasses import dataclass, field
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
    "density_transition_failure",
    "inconclusive",
]
DiagnosisCause = AblationFeature | Literal["density_events"]

FEATURE_ORDER: tuple[AblationFeature, ...] = (
    "dense_seeds",
    "masks",
    "depth",
    "adaptive_density",
)
PRIMARY_CHECKPOINTS = (0, 100, 499, 500, 600, 1_000, 2_500, 5_000)
PAIRWISE_CHECKPOINTS = (500, 1_000, 2_500)


@dataclass(frozen=True)
class AblationVariant:
    experiment_id: str
    features: frozenset[AblationFeature]
    n_iterations: int
    checkpoints: tuple[int, ...]
    density_events: bool = True

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
            or self.checkpoints[0] < 0
        ):
            raise ValueError("checkpoints must be unique, ordered, and end at n_iterations")


@dataclass(frozen=True)
class StructuralMetrics:
    visible_white_fraction: float
    high_anisotropy_fraction: float
    oversized_fraction: float
    nonfinite_count: int
    out_of_bounds_fraction: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "visible_white_fraction",
            "high_anisotropy_fraction",
            "oversized_fraction",
            "out_of_bounds_fraction",
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
    edge_l1: float = 0.0
    edge_correlation: float = 0.0
    lpips_unmasked: float | None = None
    alpha_coverage: float = 0.0
    depth_finite_fraction: float = 0.0
    depth_median: float = 0.0
    perturbed_alpha_coverage: float = 0.0
    perturbed_depth_finite_fraction: float = 0.0


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
    structural_patience: int = 2
    structural_start_iteration: int = 500
    severe_white_fraction: float = 0.50
    severe_high_anisotropy_fraction: float = 0.05
    severe_oversized_fraction: float = 0.05


@dataclass(frozen=True)
class GateDecision:
    stop: bool
    reasons: tuple[str, ...]
    consecutive_psnr_deficits: int = 0
    structural_streaks: Mapping[str, int] = field(default_factory=dict)

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
    causes: tuple[tuple[DiagnosisCause, ...], ...] = ()
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
        AblationVariant(
            experiment_id="fixed_topology_control",
            features=frozenset(),
            n_iterations=5_000,
            checkpoints=PRIMARY_CHECKPOINTS,
            density_events=False,
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
    previous_structural_streaks: Mapping[str, int] | None = None,
    policy: AblationGatePolicy = AblationGatePolicy(),
) -> GateDecision:
    reasons: list[str] = []
    stop_reasons: list[str] = []
    structural = current.structural

    if structural.nonfinite_count:
        return GateDecision(stop=True, reasons=("nonfinite_values",))

    severe_checks = (
        (
            "severe_visible_white_fraction",
            structural.visible_white_fraction,
            policy.severe_white_fraction,
        ),
        (
            "severe_anisotropy_fraction",
            structural.high_anisotropy_fraction,
            policy.severe_high_anisotropy_fraction,
        ),
        (
            "severe_oversized_fraction",
            structural.oversized_fraction,
            policy.severe_oversized_fraction,
        ),
    )
    severe_reasons = tuple(
        name for name, value, limit in severe_checks if value > limit
    )
    if severe_reasons:
        return GateDecision(stop=True, reasons=severe_reasons)

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

    prior_streaks = dict(previous_structural_streaks or {})
    structural_streaks: dict[str, int] = {}
    structural_reason_names = (
        "visible_white_fraction",
        "anisotropy_fraction",
        "oversized_fraction",
    )
    for name in structural_reason_names:
        if current.iteration >= policy.structural_start_iteration and name in reasons:
            structural_streaks[name] = prior_streaks.get(name, 0) + 1
        else:
            structural_streaks[name] = 0

    consecutive_deficits = 0
    if control is not None and current.iteration >= policy.psnr_deficit_start_iteration:
        if control.psnr_unmasked - current.psnr_unmasked >= policy.psnr_deficit_db:
            consecutive_deficits = previous_psnr_deficits + 1
        if (
            current.iteration == policy.control_final_iteration
            and consecutive_deficits >= policy.psnr_deficit_patience
        ):
            stop_reasons.append("psnr_deficit")

    if current.iteration == policy.control_final_iteration:
        stop_reasons.extend(
            name
            for name in structural_reason_names
            if structural_streaks[name] >= policy.structural_patience
        )

    if (
        current.experiment_id == "legacy_control"
        and current.iteration == policy.control_final_iteration
        and current.psnr_unmasked < policy.control_min_psnr_db
    ):
        stop_reasons.append("unusable_control_psnr")

    return GateDecision(
        stop=bool(stop_reasons),
        reasons=tuple(stop_reasons or reasons),
        consecutive_psnr_deficits=consecutive_deficits,
        structural_streaks=structural_streaks,
    )


def classify_experiments(
    outcomes: Mapping[str, ExperimentOutcome],
) -> AblationDiagnosis:
    control = outcomes.get("legacy_control")
    fixed_topology = outcomes.get("fixed_topology_control")
    full = outcomes.get("full_learned")
    if control is not None and not control.passed:
        if fixed_topology is not None and fixed_topology.passed:
            return AblationDiagnosis(
                "density_transition_failure",
                causes=(("density_events",),),
            )
        return AblationDiagnosis("inconclusive")
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

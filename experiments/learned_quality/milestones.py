from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from backend.static_pipeline.stage_cache import stage_fingerprint
from backend.static_pipeline.contracts import ReconstructionBundle

from .cache import (
    CheckpointInputs,
    CheckpointKind,
    LearnedCheckpointStore,
    checkpoint_fingerprint,
    producer_code_digest,
)
from .contracts import (
    FrameArtifact,
    GeometryAcceptance,
    GeometryCandidateReport,
    LearnedArtifacts,
    LearnedReconstructionOutput,
    StageRecord,
)
from .da3 import AnchorInferenceResult
from .flow import MotionEvidence, RigidSceneEvidence
from .masks import MaskFusionEvidence
from .segmentation import SemanticEvidence
from .tracks import QualifiedStaticTracks


MILESTONE_SCHEMA_VERSION = 1
LEGACY_COLMAP_PRODUCER_DIGESTS = (
    "498b687580e7cd45adc4ff973704e195b7714daf7a6c95d445660f57bfbb98da",
    "d45ac8a109637dec223e14a0b3b1e25790cf655c29093c6a2ec784ffe9b517a8",
)
_PREFERRED_COLMAP_PRODUCER_MIGRATIONS = {
    "7c2fa3dd0461838d10474006a078d95e0797780fb1a4f7b7a4f54c42587254f9": (
        "498b687580e7cd45adc4ff973704e195b7714daf7a6c95d445660f57bfbb98da"
    ),
}


_EXPECTED_UPSTREAM_KINDS = {
    CheckpointKind.SELECTION: frozenset(),
    CheckpointKind.BASE_EVIDENCE: frozenset(
        {CheckpointKind.SELECTION, CheckpointKind.COLMAP}
    ),
    CheckpointKind.SEMANTIC: frozenset({CheckpointKind.BASE_EVIDENCE}),
    CheckpointKind.MOTION: frozenset({CheckpointKind.BASE_EVIDENCE}),
    CheckpointKind.MASKS: frozenset(
        {
            CheckpointKind.BASE_EVIDENCE,
            CheckpointKind.SEMANTIC,
            CheckpointKind.MOTION,
        }
    ),
    CheckpointKind.GEOMETRY: frozenset(
        {CheckpointKind.SELECTION, CheckpointKind.MASKS}
    ),
    CheckpointKind.PRETRAINING: frozenset(
        {CheckpointKind.GEOMETRY, CheckpointKind.MASKS}
    ),
}


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _kind(value: object, label: str) -> CheckpointKind:
    if not isinstance(value, CheckpointKind):
        raise TypeError(f"{label} must be a CheckpointKind")
    return value


def _upstream_map(
    value: Mapping[CheckpointKind, str],
) -> Mapping[CheckpointKind, str]:
    if not isinstance(value, Mapping):
        raise TypeError("upstream must be a mapping")
    normalized: dict[CheckpointKind, str] = {}
    for raw_kind, raw_fingerprint in value.items():
        active_kind = _kind(raw_kind, "upstream kind")
        if active_kind in normalized:
            raise ValueError("upstream kinds must be unique")
        normalized[active_kind] = _digest(
            raw_fingerprint,
            f"{active_kind.value} upstream fingerprint",
        )
    return MappingProxyType(
        dict(sorted(normalized.items(), key=lambda item: item[0].value))
    )


@dataclass(frozen=True)
class MilestoneRef:
    kind: CheckpointKind
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _kind(self.kind, "kind"))
        object.__setattr__(
            self,
            "fingerprint",
            _digest(self.fingerprint, "fingerprint"),
        )


@dataclass(frozen=True)
class MilestoneInputs:
    source_digest: str
    selection_digest: str
    settings: Mapping[str, object]
    model_manifest_sha256: str
    tool_versions: Mapping[str, str]
    producer_code_sha256: str
    upstream: Mapping[CheckpointKind, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_digest", _digest(self.source_digest, "source_digest")
        )
        object.__setattr__(
            self,
            "selection_digest",
            _digest(self.selection_digest, "selection_digest"),
        )
        object.__setattr__(
            self,
            "model_manifest_sha256",
            _digest(self.model_manifest_sha256, "model_manifest_sha256"),
        )
        object.__setattr__(
            self,
            "producer_code_sha256",
            _digest(self.producer_code_sha256, "producer_code_sha256"),
        )
        if not isinstance(self.settings, Mapping) or any(
            not isinstance(key, str) or not key for key in self.settings
        ):
            raise ValueError("settings must map non-empty string keys")
        if not isinstance(self.tool_versions, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in self.tool_versions.items()
        ):
            raise ValueError("tool_versions must map non-empty strings")
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))
        object.__setattr__(
            self,
            "tool_versions",
            MappingProxyType(dict(sorted(self.tool_versions.items()))),
        )
        object.__setattr__(self, "upstream", _upstream_map(self.upstream))


@dataclass(frozen=True)
class MilestoneState:
    ref: MilestoneRef
    upstream: Mapping[CheckpointKind, str]
    value: object
    artifact_roots: Mapping[str, Path]

    def __post_init__(self) -> None:
        if not isinstance(self.ref, MilestoneRef):
            raise TypeError("ref must be a MilestoneRef")
        if self.ref.kind is CheckpointKind.COLMAP:
            raise ValueError("COLMAP uses the legacy checkpoint state")
        object.__setattr__(self, "upstream", _upstream_map(self.upstream))
        if not isinstance(self.artifact_roots, Mapping):
            raise TypeError("artifact_roots must be a mapping")
        normalized: dict[str, Path] = {}
        for label, root in self.artifact_roots.items():
            if (
                not isinstance(label, str)
                or not label
                or label in {".", ".."}
                or "/" in label
                or "\\" in label
                or ":" in label
            ):
                raise ValueError("artifact root labels must be safe path components")
            if not isinstance(root, Path):
                raise TypeError("artifact roots must be Paths")
            normalized[label] = root
        object.__setattr__(
            self,
            "artifact_roots",
            MappingProxyType(dict(sorted(normalized.items()))),
        )


@dataclass(frozen=True)
class BaseEvidenceState:
    anchors: AnchorInferenceResult
    depths: tuple[tuple[Path, str], ...]
    sky: tuple[FrameArtifact, ...]
    scene: RigidSceneEvidence
    track_audit: QualifiedStaticTracks
    colmap_ref: str


@dataclass(frozen=True)
class SemanticMilestoneState:
    semantic: SemanticEvidence


@dataclass(frozen=True)
class MotionMilestoneState:
    motion: MotionEvidence


@dataclass(frozen=True)
class MasksMilestoneState:
    masks: MaskFusionEvidence


@dataclass(frozen=True)
class GeometryMilestoneState:
    bundle: ReconstructionBundle
    frames_dir: Path
    geometry_candidates: tuple[GeometryCandidateReport, ...]
    acceptance: GeometryAcceptance | None = None


@dataclass(frozen=True)
class FinalPretrainingState:
    photometric: object
    depth: object
    dense_seeds: object
    final_stage_records: tuple[StageRecord, ...]
    model_manifest_path: Path


@dataclass(frozen=True)
class RestoredEvidenceGraph:
    base: BaseEvidenceState
    semantic: SemanticEvidence
    motion: MotionEvidence
    masks: MaskFusionEvidence | None
    refs: Mapping[CheckpointKind, MilestoneRef]


@dataclass(frozen=True)
class MilestoneSession:
    store: LearnedCheckpointStore
    source_inventory: object
    source_digest: str
    selection_digest: str
    model_manifest_sha256: str
    tool_versions: Mapping[str, str]
    selection_ref: MilestoneRef
    repository_root: Path
    run_id: str
    external_roots: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.store, LearnedCheckpointStore):
            raise TypeError("store must be a LearnedCheckpointStore")
        object.__setattr__(
            self, "source_digest", _digest(self.source_digest, "source_digest")
        )
        object.__setattr__(
            self,
            "selection_digest",
            _digest(self.selection_digest, "selection_digest"),
        )
        object.__setattr__(
            self,
            "model_manifest_sha256",
            _digest(self.model_manifest_sha256, "model_manifest_sha256"),
        )
        if (
            not isinstance(self.selection_ref, MilestoneRef)
            or self.selection_ref.kind is not CheckpointKind.SELECTION
        ):
            raise ValueError("selection_ref must reference a selection milestone")
        if not isinstance(self.tool_versions, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in self.tool_versions.items()
        ):
            raise ValueError("tool_versions must map non-empty strings")
        object.__setattr__(
            self,
            "tool_versions",
            MappingProxyType(dict(sorted(self.tool_versions.items()))),
        )
        repository_root = Path(self.repository_root)
        if repository_root.is_symlink() or not repository_root.is_dir():
            raise ValueError("repository_root must be a regular directory")
        object.__setattr__(
            self,
            "repository_root",
            repository_root.resolve(strict=True),
        )
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        if not isinstance(self.external_roots, Mapping):
            raise TypeError("external_roots must be a mapping")
        normalized_external_roots = {
            label: Path(root).resolve(strict=True)
            for label, root in self.external_roots.items()
        }
        object.__setattr__(
            self,
            "external_roots",
            MappingProxyType(dict(sorted(normalized_external_roots.items()))),
        )

    def make_ref(
        self,
        kind: CheckpointKind,
        *,
        upstream: Mapping[CheckpointKind, str],
        settings: Mapping[str, object],
        producer_paths: Sequence[str] = (),
        producer_code_sha256: str | None = None,
        selection_digest: str | None = None,
    ) -> MilestoneRef:
        active_kind = _kind(kind, "kind")
        normalized_upstream = validate_upstream_kinds(active_kind, upstream)
        if producer_code_sha256 is None:
            if not producer_paths:
                raise ValueError("producer_paths cannot be empty")
            active_producer_digest = producer_code_digest(
                self.repository_root,
                producer_paths,
            )
        else:
            if producer_paths:
                raise ValueError(
                    "provide producer_paths or producer_code_sha256, not both"
                )
            active_producer_digest = _digest(
                producer_code_sha256,
                "producer_code_sha256",
            )
        return MilestoneRef(
            active_kind,
            milestone_fingerprint(
                active_kind,
                MilestoneInputs(
                    source_digest=self.source_digest,
                    selection_digest=(
                        self.selection_digest
                        if selection_digest is None
                        else _digest(selection_digest, "selection_digest")
                    ),
                    settings=settings,
                    model_manifest_sha256=self.model_manifest_sha256,
                    tool_versions=self.tool_versions,
                    producer_code_sha256=active_producer_digest,
                    upstream=normalized_upstream,
                ),
            ),
        )

    def branch(
        self,
        *,
        selection_digest: str,
        selection_ref: MilestoneRef,
        selection_root: Path,
    ) -> MilestoneSession:
        if selection_ref.kind is not CheckpointKind.SELECTION:
            raise ValueError("selection_ref must reference a selection milestone")
        return replace(
            self,
            selection_digest=_digest(selection_digest, "selection_digest"),
            selection_ref=selection_ref,
            external_roots={"selection": Path(selection_root)},
        )

    def bind_external_root(self, label: str, root: Path) -> MilestoneSession:
        if (
            not isinstance(label, str)
            or not label
            or label in {".", ".."}
            or "/" in label
            or "\\" in label
        ):
            raise ValueError("external root labels must be safe path components")
        resolved = Path(root).resolve(strict=True)
        existing = self.external_roots.get(label)
        if existing is not None and existing != resolved:
            raise ValueError("external root labels cannot be rebound")
        return replace(
            self,
            external_roots={**self.external_roots, label: resolved},
        )

    def restore(self, ref: MilestoneRef, destination: Path) -> MilestoneState | None:
        return self.store.restore_milestone(
            ref,
            destination=destination,
            source_inventory=self.source_inventory,
            external_roots=self.external_roots,
        )

    def publish(
        self,
        ref: MilestoneRef,
        *,
        upstream: Mapping[CheckpointKind, str],
        value: object,
        artifact_roots: Mapping[str, Path],
    ) -> Path:
        state = MilestoneState(
            ref=ref,
            upstream=upstream,
            value=value,
            artifact_roots=artifact_roots,
        )
        return self.store.publish_milestone(
            state,
            run_id=f"{self.run_id}-{ref.kind.value}",
            external_roots=self.external_roots,
        )


def assemble_learned_reconstruction(
    *,
    geometry: GeometryMilestoneState,
    base: BaseEvidenceState,
    semantic: SemanticMilestoneState,
    motion: MotionMilestoneState,
    masks: MasksMilestoneState,
    evidence_stage_records: tuple[StageRecord, ...],
    final: FinalPretrainingState | None,
) -> LearnedReconstructionOutput:
    """Reassemble the public reconstruction contract from milestone states."""
    if not isinstance(geometry, GeometryMilestoneState):
        raise TypeError("geometry must be a GeometryMilestoneState")
    if not isinstance(base, BaseEvidenceState):
        raise TypeError("base must be a BaseEvidenceState")
    if not isinstance(semantic, SemanticMilestoneState):
        raise TypeError("semantic must be a SemanticMilestoneState")
    if not isinstance(motion, MotionMilestoneState):
        raise TypeError("motion must be a MotionMilestoneState")
    if not isinstance(masks, MasksMilestoneState):
        raise TypeError("masks must be a MasksMilestoneState")
    if type(evidence_stage_records) is not tuple:
        raise TypeError("evidence_stage_records must be a tuple")
    final_records: tuple[StageRecord, ...] = ()
    photometric = None
    depth = None
    dense_seeds = None
    model_manifest_path = None
    if final is not None:
        if not isinstance(final, FinalPretrainingState):
            raise TypeError("final must be a FinalPretrainingState or None")
        final_records = final.final_stage_records
        photometric = final.photometric
        depth = final.depth
        dense_seeds = final.dense_seeds
        model_manifest_path = final.model_manifest_path
    artifacts = LearnedArtifacts(
        da3=base.anchors,
        base_evidence=base,
        semantic=semantic.semantic,
        flow=motion.motion,
        masks=masks.masks,
        photometric=photometric,
        depth=depth,
        dense_seeds=dense_seeds,
        model_manifest_path=model_manifest_path,
        stage_records=(*evidence_stage_records, *final_records),
    )
    return LearnedReconstructionOutput(
        bundle=geometry.bundle,
        frames_dir=geometry.frames_dir,
        artifacts=artifacts,
        geometry_candidates=geometry.geometry_candidates,
        acceptance=geometry.acceptance,
    )


def expected_upstream_kinds(kind: CheckpointKind) -> frozenset[CheckpointKind]:
    active_kind = _kind(kind, "kind")
    if active_kind is CheckpointKind.COLMAP:
        return frozenset()
    try:
        return _EXPECTED_UPSTREAM_KINDS[active_kind]
    except KeyError as error:
        raise ValueError(f"unsupported milestone kind: {active_kind.value}") from error


def validate_upstream_kinds(
    kind: CheckpointKind,
    upstream: Mapping[CheckpointKind, str],
) -> Mapping[CheckpointKind, str]:
    normalized = _upstream_map(upstream)
    expected = expected_upstream_kinds(kind)
    if frozenset(normalized) != expected:
        raise ValueError(
            f"{kind.value} upstream kinds must be "
            f"{sorted(item.value for item in expected)}"
        )
    return normalized


def milestone_fingerprint(kind: CheckpointKind, inputs: MilestoneInputs) -> str:
    """Hash canonical milestone inputs and their sorted upstream map."""
    active_kind = _kind(kind, "kind")
    if active_kind is CheckpointKind.COLMAP:
        raise ValueError("COLMAP uses the legacy checkpoint fingerprint function")
    if not isinstance(inputs, MilestoneInputs):
        raise TypeError("inputs must be MilestoneInputs")
    fingerprint_inputs = {
        "source": inputs.source_digest,
        "selection": inputs.selection_digest,
        "model_manifest": inputs.model_manifest_sha256,
        "producer_code": inputs.producer_code_sha256,
    }
    fingerprint_inputs.update(
        {
            f"upstream:{upstream_kind.value}": fingerprint
            for upstream_kind, fingerprint in inputs.upstream.items()
        }
    )
    return stage_fingerprint(
        f"learned_{active_kind.value}",
        inputs=fingerprint_inputs,
        settings=dict(inputs.settings),
        policy_version=f"learned-milestone-v{MILESTONE_SCHEMA_VERSION}",
        tools=dict(inputs.tool_versions),
    )


def compatible_colmap_fingerprints(
    current: CheckpointInputs,
    *,
    legacy_producer_digests: tuple[str, ...],
) -> tuple[str, ...]:
    """Return verified-compatible COLMAP fingerprints without duplicates."""
    if not isinstance(current, CheckpointInputs):
        raise TypeError("current must be CheckpointInputs")
    if not isinstance(legacy_producer_digests, tuple):
        raise TypeError("legacy_producer_digests must be a tuple")
    producer_digests = [current.producer_code_sha256, *legacy_producer_digests]
    preferred = _PREFERRED_COLMAP_PRODUCER_MIGRATIONS.get(current.producer_code_sha256)
    if preferred in legacy_producer_digests:
        producer_digests.remove(preferred)
        producer_digests.insert(0, preferred)
    result: list[str] = []
    for producer_digest in producer_digests:
        active_digest = _digest(producer_digest, "legacy producer digest")
        fingerprint = checkpoint_fingerprint(
            CheckpointKind.COLMAP,
            replace(current, producer_code_sha256=active_digest),
        )
        if fingerprint not in result:
            result.append(fingerprint)
    return tuple(result)

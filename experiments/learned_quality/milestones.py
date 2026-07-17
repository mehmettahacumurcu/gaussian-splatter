from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from backend.static_pipeline.stage_cache import stage_fingerprint

from .cache import (
    CheckpointInputs,
    CheckpointKind,
    checkpoint_fingerprint,
)


MILESTONE_SCHEMA_VERSION = 1
LEGACY_COLMAP_PRODUCER_DIGESTS = (
    "d45ac8a109637dec223e14a0b3b1e25790cf655c29093c6a2ec784ffe9b517a8",
)


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
    """Return current then allow-listed legacy fingerprints without duplicates."""
    if not isinstance(current, CheckpointInputs):
        raise TypeError("current must be CheckpointInputs")
    if not isinstance(legacy_producer_digests, tuple):
        raise TypeError("legacy_producer_digests must be a tuple")
    result: list[str] = []
    for producer_digest in (current.producer_code_sha256, *legacy_producer_digests):
        active_digest = _digest(producer_digest, "legacy producer digest")
        fingerprint = checkpoint_fingerprint(
            CheckpointKind.COLMAP,
            replace(current, producer_code_sha256=active_digest),
        )
        if fingerprint not in result:
            result.append(fingerprint)
    return tuple(result)

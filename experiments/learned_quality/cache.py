from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from threading import Event, Lock
from types import MappingProxyType
from typing import TYPE_CHECKING

from backend.static_pipeline.sources import _atomic_promote_no_replace
from backend.static_pipeline.stage_cache import stage_fingerprint

from .contracts import GENERATOR_ID

if TYPE_CHECKING:
    from .milestones import MilestoneRef, MilestoneState


_SCHEMA_VERSION = 1
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_UNSUPPORTED_NOREPLACE_ERRNOS = frozenset(
    {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}
)
_RESTORE_DEFAULT_WORKERS = 4
_RESTORE_MAX_WORKERS = 16
_RESTORE_BLOCK_BYTES = 8 * 1024 * 1024
_RESTORE_PROGRESS_STEP_BYTES = 512 * 1024 * 1024
_PRETRAINING_ROOTS = frozenset(
    {
        "selection",
        "selection-backfill",
        "learned-evidence-0",
        "learned-evidence-1",
        "reconstruction",
        "photometric",
        "final-pose-depth",
        "validated-depth",
    }
)


class LearnedCacheOwnershipError(RuntimeError):
    pass


class CheckpointKind(StrEnum):
    COLMAP = "colmap"
    SELECTION = "selection"
    BASE_EVIDENCE = "base_evidence"
    SEMANTIC = "semantic"
    MOTION = "motion"
    MASKS = "masks"
    GEOMETRY = "geometry"
    PRETRAINING = "pretraining"


@dataclass(frozen=True)
class CheckpointInputs:
    source_digest: str
    settings: Mapping[str, object]
    model_manifest_sha256: str
    tool_versions: Mapping[str, str]
    producer_code_sha256: str
    upstream_fingerprint: str | None = None

    def __post_init__(self) -> None:
        _require_digest(self.source_digest, "source_digest")
        _require_digest(self.model_manifest_sha256, "model_manifest_sha256")
        _require_digest(self.producer_code_sha256, "producer_code_sha256")
        if self.upstream_fingerprint is not None:
            _require_digest(self.upstream_fingerprint, "upstream_fingerprint")
        if not isinstance(self.settings, Mapping):
            raise TypeError("settings must be a mapping")
        if not isinstance(self.tool_versions, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in self.tool_versions.items()
        ):
            raise ValueError("tool_versions must map non-empty strings")


def checkpoint_fingerprint(
    kind: CheckpointKind,
    inputs: CheckpointInputs,
) -> str:
    active_kind = _require_kind(kind)
    if not isinstance(inputs, CheckpointInputs):
        raise TypeError("inputs must be CheckpointInputs")
    fingerprint_inputs = {
        "source": inputs.source_digest,
        "model_manifest": inputs.model_manifest_sha256,
        "producer_code": inputs.producer_code_sha256,
    }
    if inputs.upstream_fingerprint is not None:
        fingerprint_inputs["upstream"] = inputs.upstream_fingerprint
    return stage_fingerprint(
        f"learned_{active_kind.value}",
        inputs=fingerprint_inputs,
        settings=dict(inputs.settings),
        policy_version=f"learned-checkpoint-v{_SCHEMA_VERSION}",
        tools=dict(inputs.tool_versions),
    )


def producer_code_digest(
    repository_root: Path,
    relative_paths: Sequence[str],
) -> str:
    root = _regular_directory(Path(repository_root), "repository_root")
    rows = []
    for raw_relative in sorted(relative_paths):
        relative = _safe_relative_path(raw_relative)
        path = _regular_file(root.joinpath(*relative.parts), "producer source")
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("producer source escapes repository_root") from error
        rows.append((relative.as_posix(), _sha256(path)))
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_checkpoint_state(value: object, *, snapshot_root: Path) -> object:
    root = _regular_directory(Path(snapshot_root), "snapshot_root")
    return _CheckpointEncoder(root).encode(value)


def decode_checkpoint_state(
    payload: object,
    *,
    restore_root: Path,
    source_inventory: object,
    external_roots: Mapping[str, Path] | None = None,
) -> object:
    root = _regular_directory(Path(restore_root), "restore_root")
    return _CheckpointDecoder(
        root,
        source_inventory,
        external_roots=external_roots or {},
    ).decode(payload)


class _CheckpointEncoder:
    def __init__(
        self,
        root: Path,
        path_rewrites: tuple[tuple[Path, Path], ...] = (),
        external_path_rewrites: tuple[tuple[str, Path], ...] = (),
    ) -> None:
        self.root = root
        self.path_rewrites = path_rewrites
        self.external_path_rewrites = external_path_rewrites
        self._object_ids: dict[int, str] = {}

    def encode(self, value: object) -> object:
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("checkpoint floats must be finite")
            return value
        if isinstance(value, Path):
            if not self.path_rewrites and not self.external_path_rewrites:
                return {
                    "__kind__": "path",
                    "relative_path": _path_relative_to_root(value, self.root),
                }
            resolved = value.resolve(strict=True)
            direct_matches = []
            for source_root, destination_root in self.path_rewrites:
                try:
                    relative = resolved.relative_to(source_root)
                except ValueError:
                    continue
                direct_matches.append(destination_root.joinpath(*relative.parts))
            external_matches = []
            for label, source_root in self.external_path_rewrites:
                try:
                    relative = resolved.relative_to(source_root)
                except ValueError:
                    continue
                external_matches.append((label, relative))
            if len(direct_matches) + len(external_matches) != 1:
                raise ValueError(
                    "milestone artifact paths must belong to exactly one root"
                )
            if direct_matches:
                return {
                    "__kind__": "path",
                    "relative_path": _path_relative_to_root(
                        direct_matches[0], self.root
                    ),
                }
            label, relative = external_matches[0]
            return {
                "__kind__": "external_path",
                "root": label,
                "relative_path": _safe_relative_path(relative.as_posix()).as_posix(),
            }
        if is_dataclass(value) and not isinstance(value, type):
            return self._encode_dataclass(value)
        if isinstance(value, tuple):
            return {"__kind__": "tuple", "items": [self.encode(item) for item in value]}
        if isinstance(value, frozenset):
            items = [self.encode(item) for item in value]
            items.sort(key=_canonical_sort_key)
            return {"__kind__": "frozenset", "items": items}
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("checkpoint mappings require string keys")
            return {
                "__kind__": "mapping",
                "items": [[key, self.encode(value[key])] for key in sorted(value)],
            }
        raise ValueError(f"checkpoint value type is not registered: {type(value)!r}")

    def _encode_dataclass(self, value: object) -> object:
        registry = _checkpoint_types()
        type_name = f"{type(value).__module__}.{type(value).__qualname__}"
        if registry.get(type_name) is not type(value):
            raise ValueError(f"checkpoint dataclass is not registered: {type_name}")
        identity = id(value)
        existing = self._object_ids.get(identity)
        if existing is not None:
            return {"__kind__": "ref", "object_id": existing}
        object_id = f"o{len(self._object_ids) + 1}"
        self._object_ids[identity] = object_id
        omitted = {"inventory"} if type_name.endswith(".SelectionOutput") else set()
        encoded_fields = {
            field.name: self.encode(getattr(value, field.name))
            for field in fields(value)
            if field.name not in omitted
        }
        return {
            "__kind__": "dataclass",
            "type": type_name,
            "object_id": object_id,
            "fields": encoded_fields,
        }


class _CheckpointDecoder:
    def __init__(
        self,
        root: Path,
        source_inventory: object,
        *,
        external_roots: Mapping[str, Path],
    ) -> None:
        self.root = root
        self.source_inventory = source_inventory
        self.external_roots = {
            label: _regular_directory(Path(path), f"{label} external root")
            for label, path in external_roots.items()
        }
        self._objects: dict[str, object] = {}

    def decode(self, payload: object) -> object:
        if payload is None or isinstance(payload, (bool, str, int)):
            return payload
        if isinstance(payload, float):
            if not math.isfinite(payload):
                raise ValueError("checkpoint floats must be finite")
            return payload
        if not isinstance(payload, dict):
            raise ValueError("checkpoint value must be a tagged JSON object")
        kind = payload.get("__kind__")
        if kind == "path":
            _require_keys(payload, {"__kind__", "relative_path"})
            relative = _safe_relative_path(payload["relative_path"])
            return _resolved_restore_path(self.root, relative)
        if kind == "external_path":
            _require_keys(payload, {"__kind__", "root", "relative_path"})
            label = payload["root"]
            if not isinstance(label, str) or label not in self.external_roots:
                raise ValueError("checkpoint external root is unavailable")
            relative = _safe_relative_path(payload["relative_path"])
            return _resolved_restore_path(self.external_roots[label], relative)
        if kind == "tuple":
            _require_keys(payload, {"__kind__", "items"})
            return tuple(self.decode(item) for item in _require_list(payload["items"]))
        if kind == "frozenset":
            _require_keys(payload, {"__kind__", "items"})
            return frozenset(
                self.decode(item) for item in _require_list(payload["items"])
            )
        if kind == "mapping":
            _require_keys(payload, {"__kind__", "items"})
            result: dict[str, object] = {}
            for row in _require_list(payload["items"]):
                if (
                    not isinstance(row, list)
                    or len(row) != 2
                    or not isinstance(row[0], str)
                    or row[0] in result
                ):
                    raise ValueError("checkpoint mapping rows are malformed")
                result[row[0]] = self.decode(row[1])
            return result
        if kind == "ref":
            _require_keys(payload, {"__kind__", "object_id"})
            object_id = _require_object_id(payload["object_id"])
            try:
                return self._objects[object_id]
            except KeyError as error:
                raise ValueError(
                    "checkpoint reference targets an unknown object"
                ) from error
        if kind == "dataclass":
            return self._decode_dataclass(payload)
        raise ValueError(f"unknown checkpoint value kind: {kind!r}")

    def _decode_dataclass(self, payload: dict[str, object]) -> object:
        _require_keys(payload, {"__kind__", "type", "object_id", "fields"})
        type_name = payload["type"]
        if not isinstance(type_name, str):
            raise ValueError("unknown checkpoint type")
        try:
            active_type = _checkpoint_types()[type_name]
        except KeyError as error:
            raise ValueError(f"unknown checkpoint type: {type_name}") from error
        object_id = _require_object_id(payload["object_id"])
        if object_id in self._objects:
            raise ValueError("checkpoint object IDs must be unique")
        raw_fields = payload["fields"]
        if not isinstance(raw_fields, dict) or any(
            not isinstance(key, str) for key in raw_fields
        ):
            raise ValueError("checkpoint dataclass fields are malformed")
        expected = {field.name for field in fields(active_type)}
        is_selection = type_name.endswith(".SelectionOutput")
        if is_selection:
            expected.remove("inventory")
        if set(raw_fields) != expected:
            raise ValueError("checkpoint dataclass fields do not match the schema")
        values = {key: self.decode(raw_fields[key]) for key in sorted(raw_fields)}
        if is_selection:
            values["inventory"] = self.source_inventory
        instance = active_type(**values)
        self._objects[object_id] = instance
        return instance


def _checkpoint_types() -> dict[str, type[object]]:
    from backend.static_pipeline.contracts import (
        ColmapAttempt,
        FrameMetrics,
        FrameRecord,
        GateDecision,
        ModelMetrics,
        ReconstructionBundle,
        SelectionManifest,
        SelectionPolicy,
        UncoveredInterval,
    )
    from backend.static_pipeline.runner import SelectionOutput

    from .contracts import (
        FrameArtifact,
        GeometryCandidateReport,
        LearnedArtifacts,
        LearnedReconstructionOutput,
        ModelRef,
        StageRecord,
    )
    from .da3 import (
        AnchorInferenceResult,
        CameraRecord,
        FramePredictionArtifact,
        PinholeCamera,
    )
    from .depth import (
        DenseSeedArtifact,
        DenseSeedPolicy,
        DepthValidationResult,
        ValidatedDepthFrame,
    )
    from .flow import (
        FlowGatePolicy,
        MotionEvidence,
        MotionFrameEvidence,
        RigidFrameEvidence,
        RigidSceneEvidence,
        StaticTrack,
        TrackObservation,
    )
    from .lifecycle import BatchAttemptRecord
    from .masks import FusedMaskFrame, MaskFusionEvidence, MaskFusionPolicy
    from .milestones import (
        BaseEvidenceState,
        FinalPretrainingState,
        GeometryMilestoneState,
        MasksMilestoneState,
        MotionMilestoneState,
        SemanticMilestoneState,
    )
    from .photometric import PhotometricEvidence, RgbAffineTransform
    from .segmentation import SemanticEvidence, SemanticFrameEvidence, SemanticPolicy
    from .tracks import QualifiedStaticTracks, TrackAuditReport

    types = (
        SelectionOutput,
        SelectionPolicy,
        FrameMetrics,
        FrameRecord,
        SelectionManifest,
        ColmapAttempt,
        ModelMetrics,
        UncoveredInterval,
        GateDecision,
        ReconstructionBundle,
        ModelRef,
        FrameArtifact,
        StageRecord,
        LearnedArtifacts,
        GeometryCandidateReport,
        LearnedReconstructionOutput,
        BatchAttemptRecord,
        PinholeCamera,
        FramePredictionArtifact,
        CameraRecord,
        AnchorInferenceResult,
        SemanticPolicy,
        SemanticFrameEvidence,
        SemanticEvidence,
        FlowGatePolicy,
        RigidFrameEvidence,
        TrackObservation,
        StaticTrack,
        RigidSceneEvidence,
        MotionFrameEvidence,
        MotionEvidence,
        MaskFusionPolicy,
        FusedMaskFrame,
        MaskFusionEvidence,
        RgbAffineTransform,
        PhotometricEvidence,
        DenseSeedPolicy,
        ValidatedDepthFrame,
        DenseSeedArtifact,
        DepthValidationResult,
        TrackAuditReport,
        QualifiedStaticTracks,
        BaseEvidenceState,
        GeometryMilestoneState,
        FinalPretrainingState,
        SemanticMilestoneState,
        MotionMilestoneState,
        MasksMilestoneState,
    )
    return {f"{item.__module__}.{item.__qualname__}": item for item in types}


class LearnedCheckpointStore:
    def __init__(self, cache_root: Path, *, input_identity: str) -> None:
        self.cache_root = Path(cache_root).resolve(strict=False)
        self.input_identity = _require_digest(input_identity, "input_identity")

    def publish_generation(
        self,
        kind: CheckpointKind,
        *,
        fingerprint: str,
        run_id: str,
        source_root: Path,
        upstream: Mapping[CheckpointKind, str] | None = None,
        artifact_roots: Mapping[str, str] | None = None,
        external_root_labels: Sequence[str] = (),
    ) -> Path:
        active_kind = _require_kind(kind)
        active_fingerprint = _require_digest(fingerprint, "fingerprint")
        if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must be a safe non-empty identifier")
        source = _regular_directory(Path(source_root), "source_root")
        _inventory(source)
        metadata = _generation_metadata(
            upstream,
            artifact_roots,
            external_root_labels,
        )
        self._ensure_owned_root(create=True)

        recovered = self.find_generation(active_kind, active_fingerprint)
        if recovered is not None:
            return recovered

        target_parent = self.cache_root / active_kind.value
        target_parent.mkdir(parents=True, exist_ok=True)
        target = target_parent / active_fingerprint
        if os.path.lexists(target):
            if self._valid_generation(target, active_kind, active_fingerprint):
                return target
            self._remove_owned_tree(target)

        staging_parent = self.cache_root / "staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = staging_parent / f"{active_kind.value}-{run_id}"
        if os.path.lexists(staging):
            self._remove_owned_tree(staging)
        staging.mkdir()
        staging_complete = False
        try:
            payload_root = staging / "payload"
            shutil.copytree(source, payload_root, copy_function=shutil.copy2)
            files = _inventory(payload_root, relative_prefix="payload")
            manifest = {
                "schema_version": _SCHEMA_VERSION,
                "kind": active_kind.value,
                "fingerprint": active_fingerprint,
                "files": files,
            }
            manifest.update(metadata)
            manifest_path = staging / "manifest.json"
            _write_json(manifest_path, manifest)
            success = {
                "schema_version": _SCHEMA_VERSION,
                "kind": active_kind.value,
                "fingerprint": active_fingerprint,
                "manifest_sha256": _sha256(manifest_path),
            }
            if "upstream" in metadata:
                success["upstream"] = metadata["upstream"]
            _write_json(
                staging / "_SUCCESS.json",
                success,
            )
            if not self._valid_generation(
                staging,
                active_kind,
                active_fingerprint,
            ):
                raise ValueError("staged cache generation failed verification")
            staging_complete = True
            sync = getattr(os, "sync", None)
            if callable(sync):
                sync()
            self._promote_generation(
                staging,
                target,
                kind=active_kind,
                fingerprint=active_fingerprint,
            )
        except BaseException:
            if os.path.lexists(staging) and not (
                staging_complete
                and self._valid_generation(
                    staging,
                    active_kind,
                    active_fingerprint,
                )
            ):
                self._remove_owned_tree(staging)
            raise

        self._remove_older_generations(active_kind, keep=target)
        return target

    def publish_milestone(
        self,
        state: MilestoneState,
        *,
        run_id: str,
        external_roots: Mapping[str, Path] | None = None,
    ) -> Path:
        from .milestones import (
            MilestoneRef,
            MilestoneState,
            validate_upstream_kinds,
        )

        if not isinstance(state, MilestoneState):
            raise TypeError("state must be a MilestoneState")
        upstream = validate_upstream_kinds(state.ref.kind, state.upstream)
        for upstream_kind, fingerprint in upstream.items():
            upstream_ref = MilestoneRef(upstream_kind, fingerprint)
            try:
                self.validate_milestone_graph(upstream_ref)
            except ValueError as error:
                raise ValueError(
                    f"missing upstream milestone: {upstream_kind.value}/{fingerprint}"
                ) from error
        if not state.artifact_roots:
            raise ValueError("milestone must own at least one direct artifact root")
        resolved_roots: dict[str, Path] = {}
        for label, raw_root in state.artifact_roots.items():
            resolved_roots[label] = _regular_directory(
                Path(raw_root), f"{label} artifact root"
            )
        resolved_external_roots: dict[str, Path] = {}
        for label, raw_root in (external_roots or {}).items():
            if label in resolved_roots:
                raise ValueError("direct and external artifact root labels must differ")
            resolved_external_roots[label] = _regular_directory(
                Path(raw_root), f"{label} external root"
            )
        root_rows = tuple([*resolved_roots.items(), *resolved_external_roots.items()])
        for index, (left_label, left_root) in enumerate(root_rows):
            for right_label, right_root in root_rows[index + 1 :]:
                try:
                    right_root.relative_to(left_root)
                except ValueError:
                    pass
                else:
                    raise ValueError(
                        f"artifact roots overlap: {left_label} and {right_label}"
                    )
                try:
                    left_root.relative_to(right_root)
                except ValueError:
                    pass
                else:
                    raise ValueError(
                        f"artifact roots overlap: {left_label} and {right_label}"
                    )

        with tempfile.TemporaryDirectory(
            prefix=f"learned-{state.ref.kind.value}-milestone-"
        ) as temporary:
            snapshot = Path(temporary).resolve(strict=True)
            path_rewrites: list[tuple[Path, Path]] = []
            artifact_manifest: dict[str, str] = {}
            for label, source in sorted(resolved_roots.items()):
                destination = snapshot / "artifacts" / label
                shutil.copytree(source, destination, copy_function=shutil.copy2)
                path_rewrites.append((source, destination.resolve(strict=True)))
                artifact_manifest[label] = f"artifacts/{label}"
            encoded = _CheckpointEncoder(
                snapshot,
                tuple(path_rewrites),
                tuple(sorted(resolved_external_roots.items())),
            ).encode(state.value)
            _write_json(snapshot / "state.json", encoded)
            return self.publish_generation(
                state.ref.kind,
                fingerprint=state.ref.fingerprint,
                run_id=run_id,
                source_root=snapshot,
                upstream=upstream,
                artifact_roots=artifact_manifest,
                external_root_labels=tuple(resolved_external_roots),
            )

    def restore_milestone(
        self,
        ref: MilestoneRef,
        *,
        destination: Path,
        source_inventory: object,
        external_roots: Mapping[str, Path] | None = None,
    ) -> MilestoneState | None:
        from .milestones import MilestoneRef, MilestoneState

        if not isinstance(ref, MilestoneRef):
            raise TypeError("ref must be a MilestoneRef")
        located = self._find_generation_metadata(ref.kind, ref.fingerprint)
        if located is None:
            return None
        generation, manifest = located
        try:
            self.validate_milestone_graph(ref)
        except ValueError:
            return None
        target = Path(destination)
        if os.path.lexists(target):
            raise FileExistsError(f"milestone restore destination exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            _copy_verified_payload(generation, target, manifest)
            upstream, raw_artifact_roots, external_labels = _manifest_metadata(manifest)
            resolved_external_roots = {
                label: _regular_directory(Path(path), f"{label} external root")
                for label, path in (external_roots or {}).items()
            }
            if set(resolved_external_roots) != set(external_labels):
                raise ValueError("milestone external roots do not match its manifest")
            payload = json.loads((target / "state.json").read_text(encoding="utf-8"))
            value = decode_checkpoint_state(
                payload,
                restore_root=target,
                source_inventory=source_inventory,
                external_roots=resolved_external_roots,
            )
            artifact_roots = {
                label: _resolved_restore_path(
                    target,
                    _safe_relative_path(relative),
                )
                for label, relative in raw_artifact_roots.items()
            }
            for path in _checkpoint_state_paths(value):
                resolved = Path(path).resolve(strict=True)
                if not any(
                    resolved == artifact_root or resolved.is_relative_to(artifact_root)
                    for artifact_root in (
                        *artifact_roots.values(),
                        *resolved_external_roots.values(),
                    )
                ):
                    raise ValueError(
                        "restored milestone state escapes its artifact roots"
                    )
            return MilestoneState(
                ref=ref,
                upstream=upstream,
                value=value,
                artifact_roots=artifact_roots,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            if os.path.lexists(target):
                self._remove_restore_tree(target)
            return None
        except BaseException:
            if os.path.lexists(target):
                self._remove_restore_tree(target)
            raise

    def validate_milestone_graph(self, root: MilestoneRef) -> tuple[MilestoneRef, ...]:
        from .milestones import (
            MilestoneRef,
            expected_upstream_kinds,
        )

        if not isinstance(root, MilestoneRef):
            raise TypeError("root must be a MilestoneRef")
        visiting: set[MilestoneRef] = set()
        visited: set[MilestoneRef] = set()
        result: list[MilestoneRef] = []
        observed_edges: dict[MilestoneRef, Mapping[CheckpointKind, str]] = {}

        def visit(ref: MilestoneRef) -> None:
            if ref in visiting:
                raise ValueError("milestone dependency graph contains a cycle")
            if ref in visited:
                return
            located = self._find_generation_metadata(ref.kind, ref.fingerprint)
            if located is None:
                raise ValueError(
                    f"missing upstream milestone: {ref.kind.value}/{ref.fingerprint}"
                )
            _generation, manifest = located
            visiting.add(ref)
            if ref.kind is CheckpointKind.COLMAP:
                upstream: Mapping[CheckpointKind, str] = {}
            else:
                upstream, _, _ = _manifest_metadata(manifest)
            observed_edges[ref] = upstream
            for upstream_kind, fingerprint in sorted(
                upstream.items(), key=lambda item: item[0].value
            ):
                visit(MilestoneRef(upstream_kind, fingerprint))
            visiting.remove(ref)
            visited.add(ref)
            result.append(ref)

        visit(root)
        for ref, upstream in observed_edges.items():
            if ref.kind is CheckpointKind.COLMAP:
                continue
            expected = expected_upstream_kinds(ref.kind)
            if frozenset(upstream) != expected:
                raise ValueError(
                    f"{ref.kind.value} upstream kinds must be "
                    f"{sorted(item.value for item in expected)}"
                )
        return tuple(result)

    def find_latest_complete_lineage(
        self,
        root_kind: CheckpointKind,
        *,
        selection_fingerprint: str,
    ) -> Mapping[CheckpointKind, MilestoneRef] | None:
        from .milestones import MilestoneRef

        active_kind = _require_kind(root_kind)
        if active_kind is CheckpointKind.COLMAP:
            raise ValueError("a milestone lineage cannot be rooted at COLMAP")
        selection_digest = _require_digest(
            selection_fingerprint,
            "selection_fingerprint",
        )
        if not self._ensure_owned_root(create=False):
            return None
        self._recover_staged_generations(active_kind)
        parent = self.cache_root / active_kind.value
        if not parent.is_dir() or parent.is_symlink():
            return None
        candidates: list[tuple[int, MilestoneRef]] = []
        for generation in sorted(parent.iterdir(), key=lambda item: item.name):
            try:
                fingerprint = _require_digest(generation.name, "fingerprint")
            except ValueError:
                continue
            if self._generation_metadata(generation, active_kind, fingerprint) is None:
                continue
            try:
                modified = (generation / "_SUCCESS.json").stat().st_mtime_ns
            except OSError:
                continue
            candidates.append((modified, MilestoneRef(active_kind, fingerprint)))
        for _modified, root in sorted(
            candidates,
            key=lambda item: (item[0], item[1].fingerprint),
            reverse=True,
        ):
            try:
                graph = self.validate_milestone_graph(root)
            except ValueError:
                continue
            by_kind = {ref.kind: ref for ref in graph}
            if len(by_kind) != len(graph):
                continue
            selection = by_kind.get(CheckpointKind.SELECTION)
            if selection is None or selection.fingerprint != selection_digest:
                continue
            return MappingProxyType(
                dict(sorted(by_kind.items(), key=lambda item: item[0].value))
            )
        return None

    def cleanup_unreferenced_milestones(self, keep: MilestoneRef) -> None:
        from .milestones import MilestoneRef

        if not isinstance(keep, MilestoneRef):
            raise TypeError("keep must be a MilestoneRef")
        generation = self.find_generation(keep.kind, keep.fingerprint)
        if generation is None:
            raise ValueError("keep milestone is not a valid generation")
        self._remove_older_generations(keep.kind, keep=generation)

    def probe_drive_publication(self, *, run_id: str) -> None:
        if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must be a safe non-empty identifier")
        self._ensure_owned_root(create=True)
        fingerprint = hashlib.sha256(
            f"drive-cache-probe:{self.input_identity}".encode("utf-8")
        ).hexdigest()
        probe_parent = self.cache_root / "probe"
        probe_root = probe_parent / run_id
        if os.path.lexists(probe_root):
            self._remove_owned_tree(probe_root)
        probe_root.mkdir(parents=True)
        staging = probe_root / "staging"
        target = probe_root / "published"
        try:
            payload_root = staging / "payload"
            payload_root.mkdir(parents=True)
            (payload_root / "probe.txt").write_text(
                "learned-quality-drive-cache-probe\n",
                encoding="utf-8",
            )
            manifest = {
                "schema_version": _SCHEMA_VERSION,
                "kind": CheckpointKind.COLMAP.value,
                "fingerprint": fingerprint,
                "files": _inventory(payload_root, relative_prefix="payload"),
            }
            manifest_path = staging / "manifest.json"
            _write_json(manifest_path, manifest)
            _write_json(
                staging / "_SUCCESS.json",
                {
                    "schema_version": _SCHEMA_VERSION,
                    "kind": CheckpointKind.COLMAP.value,
                    "fingerprint": fingerprint,
                    "manifest_sha256": _sha256(manifest_path),
                },
            )
            if not self._valid_generation(
                staging,
                CheckpointKind.COLMAP,
                fingerprint,
            ):
                raise ValueError("Drive cache probe staging failed verification")
            sync = getattr(os, "sync", None)
            if callable(sync):
                sync()
            self._promote_generation(
                staging,
                target,
                kind=CheckpointKind.COLMAP,
                fingerprint=fingerprint,
            )
            if not self._valid_generation(
                target,
                CheckpointKind.COLMAP,
                fingerprint,
            ):
                raise ValueError("Drive cache publication probe failed verification")
        finally:
            if os.path.lexists(probe_root):
                self._remove_owned_tree(probe_root)
            try:
                probe_parent.rmdir()
            except OSError:
                pass

    def _promote_generation(
        self,
        staging: Path,
        target: Path,
        *,
        kind: CheckpointKind,
        fingerprint: str,
    ) -> None:
        try:
            _atomic_promote_no_replace(staging, target)
            return
        except OSError as error:
            if error.errno not in _UNSUPPORTED_NOREPLACE_ERRNOS:
                raise
            if os.path.lexists(target):
                raise FileExistsError(
                    errno.EEXIST,
                    os.strerror(errno.EEXIST),
                    str(target),
                ) from error
        self._promote_generation_with_success_marker(
            staging,
            target,
            kind=kind,
            fingerprint=fingerprint,
        )

    def _promote_generation_with_success_marker(
        self,
        staging: Path,
        target: Path,
        *,
        kind: CheckpointKind,
        fingerprint: str,
    ) -> None:
        if not self._valid_generation(staging, kind, fingerprint):
            raise ValueError("staged cache generation is invalid")
        target.mkdir(exist_ok=False)
        try:
            shutil.copytree(
                staging / "payload",
                target / "payload",
                copy_function=shutil.copy2,
            )
            shutil.copy2(staging / "manifest.json", target / "manifest.json")
            if _inventory(staging / "payload") != _inventory(target / "payload"):
                raise ValueError("Drive cache payload changed during publication")
            if _sha256(staging / "manifest.json") != _sha256(target / "manifest.json"):
                raise ValueError("Drive cache manifest changed during publication")
            success_bytes = (staging / "_SUCCESS.json").read_bytes()
            with (target / "_SUCCESS.json").open("xb") as stream:
                stream.write(success_bytes)
                stream.flush()
            sync = getattr(os, "sync", None)
            if callable(sync):
                sync()
            if not self._valid_generation(target, kind, fingerprint):
                raise ValueError("Drive cache generation failed final verification")
        except BaseException:
            if os.path.lexists(target):
                self._remove_owned_tree(target)
            raise
        try:
            self._remove_owned_tree(staging)
        except OSError:
            pass

    def publish_colmap(
        self,
        attempt: object,
        *,
        fingerprint: str,
        run_id: str,
    ) -> Path:
        from backend.static_pipeline.contracts import ColmapAttempt

        if not isinstance(attempt, ColmapAttempt):
            raise TypeError("attempt must be a ColmapAttempt")
        attempt_root = _regular_directory(attempt.root, "COLMAP attempt root")
        with tempfile.TemporaryDirectory(
            prefix="learned-colmap-checkpoint-",
            dir=attempt_root.parent,
        ) as temporary:
            snapshot = Path(temporary).resolve(strict=True)
            copied_root = snapshot / "attempt"
            shutil.copytree(attempt_root, copied_root, copy_function=shutil.copy2)
            relative_models = tuple(
                model.resolve(strict=True).relative_to(attempt_root)
                for model in attempt.model_dirs
            )
            state = ColmapAttempt(
                root=copied_root,
                database_path=copied_root
                / attempt.database_path.resolve(strict=True).relative_to(attempt_root),
                model_dirs=tuple(
                    copied_root / relative for relative in relative_models
                ),
                colmap_version=attempt.colmap_version,
                fingerprint=attempt.fingerprint,
            )
            _write_json(
                snapshot / "state.json",
                encode_checkpoint_state(state, snapshot_root=snapshot),
            )
            return self.publish_generation(
                CheckpointKind.COLMAP,
                fingerprint=fingerprint,
                run_id=run_id,
                source_root=snapshot,
            )

    def restore_colmap(
        self,
        fingerprint: str,
        *,
        destination: Path,
    ) -> object | None:
        from backend.static_pipeline.contracts import ColmapAttempt

        located = self._find_generation_metadata(
            CheckpointKind.COLMAP,
            fingerprint,
        )
        if located is None:
            target_generation = (
                self.cache_root / CheckpointKind.COLMAP.value / fingerprint
            )
            if os.path.lexists(target_generation):
                raise ValueError("COLMAP checkpoint metadata differs from its manifest")
            return None
        generation, manifest = located
        target = Path(destination)
        if os.path.lexists(target):
            raise FileExistsError(f"COLMAP restore destination exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            _copy_verified_payload(generation, target, manifest)
            payload = json.loads((target / "state.json").read_text(encoding="utf-8"))
            restored = decode_checkpoint_state(
                payload,
                restore_root=target,
                source_inventory=object(),
            )
            if not isinstance(restored, ColmapAttempt):
                raise ValueError("COLMAP checkpoint state has the wrong type")
            expected_root = (target / "attempt").resolve(strict=True)
            if restored.root != expected_root:
                raise ValueError("COLMAP checkpoint root is malformed")
            _regular_file(restored.database_path, "restored COLMAP database")
            if not restored.model_dirs:
                raise ValueError("COLMAP checkpoint contains no sparse model")
            for model in restored.model_dirs:
                model_root = _regular_directory(model, "restored COLMAP model")
                for name in ("cameras.txt", "images.txt", "points3D.txt"):
                    _regular_file(model_root / name, f"restored COLMAP {name}")
            return restored
        except BaseException:
            if os.path.lexists(target):
                self._remove_restore_tree(target)
            raise

    def publish_pretraining(
        self,
        *,
        source_inventory: object,
        selection: object,
        reconstruction: object,
        fingerprint: str,
        run_id: str,
        run_root: Path,
    ) -> Path:
        from backend.static_pipeline.runner import SelectionOutput

        from .contracts import LearnedReconstructionOutput

        if not isinstance(selection, SelectionOutput):
            raise TypeError("selection must be a SelectionOutput")
        if not isinstance(reconstruction, LearnedReconstructionOutput):
            raise TypeError("reconstruction must be a LearnedReconstructionOutput")
        source_digest = getattr(source_inventory, "digest", None)
        if source_digest != self.input_identity:
            raise ValueError("source inventory does not match cache input identity")
        if selection.inventory.digest != source_digest:
            raise ValueError("selection inventory does not match current source")
        root = _regular_directory(Path(run_root), "pretraining run root")
        model_manifest = reconstruction.artifacts.model_manifest_path
        if model_manifest is None:
            raise ValueError("learned reconstruction is missing its model manifest")
        manifest_file = _regular_file(
            Path(model_manifest),
            "learned model manifest",
        )
        state = {"selection": selection, "reconstruction": reconstruction}
        referenced_roots = _pretraining_referenced_roots(
            state,
            run_root=root,
            model_manifest=manifest_file,
        )
        with tempfile.TemporaryDirectory(
            prefix="learned-pretraining-checkpoint-",
            dir=root.parent,
        ) as temporary:
            snapshot = Path(temporary).resolve(strict=True)
            for relative_name in sorted(referenced_roots):
                source = root / relative_name
                destination = snapshot / relative_name
                if source.is_dir():
                    shutil.copytree(source, destination, copy_function=shutil.copy2)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            manifest_copy = snapshot / "model-manifest" / "model_manifest.json"
            manifest_copy.parent.mkdir(parents=True, exist_ok=False)
            shutil.copy2(manifest_file, manifest_copy)
            relocated = _relocate_checkpoint_state(
                state,
                source_root=root,
                destination_root=snapshot,
                model_manifest=manifest_file,
                model_manifest_copy=manifest_copy,
            )
            _write_json(
                snapshot / "state.json",
                encode_checkpoint_state(relocated, snapshot_root=snapshot),
            )
            return self.publish_generation(
                CheckpointKind.PRETRAINING,
                fingerprint=fingerprint,
                run_id=run_id,
                source_root=snapshot,
            )

    def restore_pretraining(
        self,
        *,
        source_inventory: object,
        destination: Path,
        expected_fingerprint: object,
    ) -> object | None:
        from collections.abc import Callable

        from backend.static_pipeline.runner import PretrainingRestore, SelectionOutput

        from .contracts import LearnedReconstructionOutput

        if not isinstance(expected_fingerprint, Callable):
            raise TypeError("expected_fingerprint must be callable")
        if getattr(source_inventory, "digest", None) != self.input_identity:
            raise ValueError("source inventory does not match cache input identity")
        candidates = self._valid_pretraining_generations()
        if not candidates:
            return None
        target = Path(destination)
        if os.path.lexists(target):
            raise FileExistsError(f"pretraining restore destination exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        for generation in candidates:
            try:
                manifest = self._generation_metadata(
                    generation,
                    CheckpointKind.PRETRAINING,
                    generation.name,
                )
                if manifest is None:
                    continue
                _copy_verified_payload(generation, target, manifest)
                payload = json.loads(
                    (target / "state.json").read_text(encoding="utf-8")
                )
                decoded = decode_checkpoint_state(
                    payload,
                    restore_root=target,
                    source_inventory=source_inventory,
                )
                if not isinstance(decoded, dict) or set(decoded) != {
                    "selection",
                    "reconstruction",
                }:
                    raise ValueError("pretraining checkpoint state is malformed")
                selection = decoded["selection"]
                reconstruction = decoded["reconstruction"]
                if not isinstance(selection, SelectionOutput) or not isinstance(
                    reconstruction,
                    LearnedReconstructionOutput,
                ):
                    raise ValueError("pretraining checkpoint state has the wrong type")
                expected = _require_digest(
                    expected_fingerprint(selection),
                    "expected pretraining fingerprint",
                )
                if generation.name != expected:
                    self._remove_restore_tree(target)
                    continue
                _validate_restored_pretraining(selection, reconstruction, target)
                return PretrainingRestore(selection, reconstruction)
            except BaseException:
                if os.path.lexists(target):
                    self._remove_restore_tree(target)
                raise
        return None

    def _valid_pretraining_generations(self) -> tuple[Path, ...]:
        if not self._ensure_owned_root(create=False):
            return ()
        self._recover_staged_generations(CheckpointKind.PRETRAINING)
        parent = self.cache_root / CheckpointKind.PRETRAINING.value
        if not parent.is_dir():
            return ()
        result = []
        for candidate in sorted(parent.iterdir(), key=lambda item: item.name):
            try:
                fingerprint = _require_digest(candidate.name, "fingerprint")
            except ValueError:
                continue
            manifest = self._generation_metadata(
                candidate,
                CheckpointKind.PRETRAINING,
                fingerprint,
            )
            if manifest is not None:
                if "upstream" in manifest:
                    # Terminal milestone generations are restored after selection
                    # by the dependency graph, not by the legacy cumulative codec.
                    continue
                result.append(candidate)
        return tuple(result)

    def find_generation(
        self,
        kind: CheckpointKind,
        fingerprint: str,
    ) -> Path | None:
        active_kind = _require_kind(kind)
        active_fingerprint = _require_digest(fingerprint, "fingerprint")
        if not self._ensure_owned_root(create=False):
            return None
        target = self.cache_root / active_kind.value / active_fingerprint
        if os.path.lexists(target):
            if self._valid_generation(target, active_kind, active_fingerprint):
                return target
        return self._recover_staged_generation(active_kind, active_fingerprint)

    def _generation_metadata(
        self,
        generation: Path,
        kind: CheckpointKind,
        fingerprint: str,
    ) -> dict[str, object] | None:
        try:
            return _read_verified_generation_metadata(
                generation,
                kind,
                fingerprint,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None

    def _find_generation_metadata(
        self,
        kind: CheckpointKind,
        fingerprint: str,
    ) -> tuple[Path, dict[str, object]] | None:
        active_kind = _require_kind(kind)
        active_fingerprint = _require_digest(fingerprint, "fingerprint")
        if not self._ensure_owned_root(create=False):
            return None
        target = self.cache_root / active_kind.value / active_fingerprint
        if os.path.lexists(target):
            manifest = self._generation_metadata(
                target,
                active_kind,
                active_fingerprint,
            )
            if manifest is not None:
                return target, manifest
        recovered = self._recover_staged_generation(active_kind, active_fingerprint)
        if recovered is None:
            return None
        manifest = self._generation_metadata(
            recovered,
            active_kind,
            active_fingerprint,
        )
        if manifest is None:
            return None
        return recovered, manifest

    def _recover_staged_generations(self, kind: CheckpointKind) -> None:
        staging_parent = self.cache_root / "staging"
        if not staging_parent.is_dir() or staging_parent.is_symlink():
            return
        fingerprints: set[str] = set()
        for candidate in sorted(staging_parent.iterdir(), key=lambda item: item.name):
            if not candidate.name.startswith(f"{kind.value}-"):
                continue
            try:
                manifest = json.loads(
                    (candidate / "manifest.json").read_text(encoding="utf-8")
                )
                fingerprint = _require_digest(
                    manifest.get("fingerprint"),
                    "fingerprint",
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue
            if self._valid_generation(candidate, kind, fingerprint):
                fingerprints.add(fingerprint)
        for fingerprint in sorted(fingerprints):
            self._recover_staged_generation(kind, fingerprint)

    def _recover_staged_generation(
        self,
        kind: CheckpointKind,
        fingerprint: str,
    ) -> Path | None:
        staging_parent = self.cache_root / "staging"
        if not staging_parent.is_dir() or staging_parent.is_symlink():
            return None
        target = self.cache_root / kind.value / fingerprint
        for candidate in sorted(staging_parent.iterdir(), key=lambda item: item.name):
            if not candidate.name.startswith(
                f"{kind.value}-"
            ) or not self._valid_generation(
                candidate,
                kind,
                fingerprint,
            ):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(target) and not self._valid_generation(
                target,
                kind,
                fingerprint,
            ):
                self._remove_owned_tree(target)
            try:
                self._promote_generation(
                    candidate,
                    target,
                    kind=kind,
                    fingerprint=fingerprint,
                )
            except FileExistsError:
                if not self._valid_generation(target, kind, fingerprint):
                    raise
            if not self._valid_generation(target, kind, fingerprint):
                raise ValueError("recovered cache generation failed verification")
            self._remove_older_generations(kind, keep=target)
            return target
        return None

    def _ensure_owned_root(self, *, create: bool) -> bool:
        if not os.path.lexists(self.cache_root):
            if not create:
                return False
            self.cache_root.parent.mkdir(parents=True, exist_ok=True)
            self.cache_root.mkdir()
            _write_json(self.cache_root / "_OWNERSHIP.json", self._ownership())
            return True
        if self.cache_root.is_symlink() or not self.cache_root.is_dir():
            raise LearnedCacheOwnershipError("cache root is not an owned directory")
        marker = self.cache_root / "_OWNERSHIP.json"
        if marker.is_symlink() or not marker.is_file():
            raise LearnedCacheOwnershipError("existing cache root is not owned")
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LearnedCacheOwnershipError(
                "cache ownership marker is invalid"
            ) from error
        if not isinstance(payload, dict) or payload.get("generator_id") != GENERATOR_ID:
            raise LearnedCacheOwnershipError("existing cache root is not owned")
        if payload.get("input_identity") != self.input_identity:
            raise LearnedCacheOwnershipError("cache root belongs to a different input")
        if payload != self._ownership():
            raise LearnedCacheOwnershipError("cache ownership marker is invalid")
        return True

    def _ownership(self) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "generator_id": GENERATOR_ID,
            "input_identity": self.input_identity,
        }

    def _valid_generation(
        self,
        generation: Path,
        kind: CheckpointKind,
        fingerprint: str,
    ) -> bool:
        try:
            manifest = _read_verified_generation_metadata(
                generation,
                kind,
                fingerprint,
            )
            actual = _inventory(
                generation / "payload",
                relative_prefix="payload",
            )
            return manifest["files"] == actual
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return False

    def _referenced_generations(self) -> frozenset[tuple[CheckpointKind, str]]:
        referenced: set[tuple[CheckpointKind, str]] = set()
        for owner_kind in CheckpointKind:
            parent = self.cache_root / owner_kind.value
            if not parent.is_dir() or parent.is_symlink():
                continue
            for candidate in sorted(parent.iterdir(), key=lambda item: item.name):
                try:
                    fingerprint = _require_digest(candidate.name, "fingerprint")
                except ValueError:
                    continue
                if not self._valid_generation(candidate, owner_kind, fingerprint):
                    continue
                try:
                    manifest = _read_generation_manifest(
                        candidate,
                        owner_kind,
                        fingerprint,
                    )
                    upstream, _, _ = _manifest_metadata(manifest)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    continue
                referenced.update(upstream.items())
        return frozenset(referenced)

    def _remove_older_generations(self, kind: CheckpointKind, *, keep: Path) -> None:
        parent = self.cache_root / kind.value
        referenced = self._referenced_generations()
        for candidate in parent.iterdir():
            if candidate == keep:
                continue
            try:
                fingerprint = _require_digest(candidate.name, "fingerprint")
            except ValueError:
                self._remove_owned_tree(candidate)
                continue
            if (kind, fingerprint) not in referenced:
                self._remove_owned_tree(candidate)

    def _remove_owned_tree(self, path: Path) -> None:
        candidate = Path(path)
        root = self.cache_root.resolve(strict=True)
        resolved_parent = candidate.parent.resolve(strict=True)
        try:
            resolved_parent.relative_to(root)
        except ValueError as error:
            raise LearnedCacheOwnershipError(
                "refusing to remove a path outside the owned cache root"
            ) from error
        if candidate.is_symlink():
            candidate.unlink()
        elif candidate.is_dir():
            shutil.rmtree(candidate)
        elif os.path.lexists(candidate):
            candidate.unlink()

    @staticmethod
    def _remove_restore_tree(path: Path) -> None:
        candidate = Path(path)
        resolved_parent = candidate.parent.resolve(strict=True)
        resolved_candidate = resolved_parent / candidate.name
        if resolved_candidate.parent != resolved_parent:
            raise ValueError("invalid restore cleanup path")
        if candidate.is_symlink():
            candidate.unlink()
        elif candidate.is_dir():
            shutil.rmtree(candidate)
        elif os.path.lexists(candidate):
            candidate.unlink()


def _checkpoint_state_paths(value: object) -> tuple[Path, ...]:
    from backend.static_pipeline.runner import SelectionOutput

    paths: list[Path] = []
    visited: set[int] = set()

    def visit(item: object) -> None:
        if isinstance(item, Path):
            paths.append(item)
            return
        if item is None or isinstance(item, (bool, str, int, float)):
            return
        if is_dataclass(item) and not isinstance(item, type):
            identity = id(item)
            if identity in visited:
                return
            visited.add(identity)
            for field in fields(item):
                if isinstance(item, SelectionOutput) and field.name == "inventory":
                    continue
                visit(getattr(item, field.name))
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise ValueError("checkpoint mappings require string keys")
                visit(nested)
            return
        if isinstance(item, (tuple, frozenset)):
            for nested in item:
                visit(nested)
            return
        raise ValueError(f"checkpoint value type is not registered: {type(item)!r}")

    visit(value)
    return tuple(paths)


def _pretraining_referenced_roots(
    state: object,
    *,
    run_root: Path,
    model_manifest: Path,
) -> frozenset[str]:
    roots: set[str] = set()
    for raw_path in _checkpoint_state_paths(state):
        path = Path(raw_path)
        if path.is_symlink():
            raise ValueError("pretraining checkpoint paths cannot be symlinks")
        resolved = path.resolve(strict=True)
        if resolved == model_manifest:
            continue
        try:
            relative = resolved.relative_to(run_root)
        except ValueError as error:
            raise ValueError("pretraining artifact escapes the run root") from error
        if not relative.parts or relative.parts[0] not in _PRETRAINING_ROOTS:
            raise ValueError(
                f"pretraining artifact is outside an owned stage: {relative}"
            )
        roots.add(relative.parts[0])
    if not roots:
        raise ValueError("pretraining checkpoint has no run-local artifacts")
    return frozenset(roots)


def _relocate_checkpoint_state(
    value: object,
    *,
    source_root: Path,
    destination_root: Path,
    model_manifest: Path,
    model_manifest_copy: Path,
) -> object:
    from backend.static_pipeline.runner import SelectionOutput

    memo: dict[int, object] = {}

    def relocate(item: object) -> object:
        if item is None or isinstance(item, (bool, str, int, float)):
            return item
        if isinstance(item, Path):
            resolved = item.resolve(strict=True)
            if resolved == model_manifest:
                return model_manifest_copy.resolve(strict=True)
            try:
                relative = resolved.relative_to(source_root)
            except ValueError as error:
                raise ValueError("pretraining artifact escapes the run root") from error
            return destination_root.joinpath(*relative.parts).resolve(strict=True)
        if is_dataclass(item) and not isinstance(item, type):
            identity = id(item)
            existing = memo.get(identity)
            if existing is not None:
                return existing
            values = {}
            for field in fields(item):
                current = getattr(item, field.name)
                if isinstance(item, SelectionOutput) and field.name == "inventory":
                    values[field.name] = current
                else:
                    values[field.name] = relocate(current)
            relocated = type(item)(**values)
            memo[identity] = relocated
            return relocated
        if isinstance(item, tuple):
            return tuple(relocate(nested) for nested in item)
        if isinstance(item, frozenset):
            return frozenset(relocate(nested) for nested in item)
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("checkpoint mappings require string keys")
            return {key: relocate(item[key]) for key in sorted(item)}
        raise ValueError(f"checkpoint value type is not registered: {type(item)!r}")

    return relocate(value)


def _validate_restored_pretraining(
    selection: object,
    reconstruction: object,
    restore_root: Path,
) -> None:
    root = _regular_directory(restore_root, "restored pretraining root")
    for path in _checkpoint_state_paths(
        {"selection": selection, "reconstruction": reconstruction}
    ):
        resolved = Path(path).resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "restored pretraining artifact escapes the local root"
            ) from error
    if reconstruction.selected_manifest is not selection.manifest:
        raise ValueError("restored selection and reconstruction manifests differ")
    decision = reconstruction.decision
    if not decision.passed or decision.failures:
        raise ValueError("restored reconstruction does not pass its quality gate")
    _regular_directory(selection.frames_dir, "restored selected frames")
    _regular_file(selection.source_manifest_path, "restored selection manifest")
    model_root = _regular_directory(
        reconstruction.accepted_model_dir,
        "restored accepted model",
    )
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        _regular_file(model_root / name, f"restored accepted model {name}")
    _regular_file(
        reconstruction.artifacts.model_manifest_path,
        "restored learned model manifest",
    )


def _require_kind(value: CheckpointKind) -> CheckpointKind:
    if not isinstance(value, CheckpointKind):
        raise TypeError("kind must be a CheckpointKind")
    return value


def _generation_metadata(
    upstream: Mapping[CheckpointKind, str] | None,
    artifact_roots: Mapping[str, str] | None,
    external_root_labels: Sequence[str],
) -> dict[str, object]:
    if upstream is None and artifact_roots is None:
        if external_root_labels:
            raise ValueError("external roots require milestone metadata")
        return {}
    if upstream is None or artifact_roots is None:
        raise ValueError("milestone metadata requires upstream and artifact_roots")
    if not isinstance(upstream, Mapping):
        raise TypeError("upstream must be a mapping")
    upstream_payload: dict[str, str] = {}
    for raw_kind, raw_fingerprint in upstream.items():
        active_kind = _require_kind(raw_kind)
        upstream_payload[active_kind.value] = _require_digest(
            raw_fingerprint,
            f"{active_kind.value} upstream fingerprint",
        )
    if not isinstance(artifact_roots, Mapping):
        raise TypeError("artifact_roots must be a mapping")
    roots_payload: dict[str, str] = {}
    for label, raw_relative in artifact_roots.items():
        if (
            not isinstance(label, str)
            or not label
            or label in {".", ".."}
            or any(separator in label for separator in ("/", "\\", ":"))
        ):
            raise ValueError("artifact root labels must be safe path components")
        relative = _safe_relative_path(raw_relative)
        if relative.parts != ("artifacts", label):
            raise ValueError("artifact roots must use artifacts/<label>")
        roots_payload[label] = relative.as_posix()
    labels: list[str] = []
    for label in external_root_labels:
        if (
            not isinstance(label, str)
            or not label
            or label in {".", ".."}
            or any(separator in label for separator in ("/", "\\", ":"))
            or label in labels
        ):
            raise ValueError("external root labels must be unique safe components")
        if label in roots_payload:
            raise ValueError("direct and external artifact root labels must differ")
        labels.append(label)
    return {
        "upstream": dict(sorted(upstream_payload.items())),
        "artifact_roots": dict(sorted(roots_payload.items())),
        "external_root_labels": sorted(labels),
    }


def _manifest_metadata(
    manifest: Mapping[str, object],
) -> tuple[
    Mapping[CheckpointKind, str],
    Mapping[str, str],
    tuple[str, ...],
]:
    has_upstream = "upstream" in manifest
    has_roots = "artifact_roots" in manifest
    if not has_upstream and not has_roots:
        return MappingProxyType({}), MappingProxyType({}), ()
    if not has_upstream or not has_roots:
        raise ValueError("milestone manifest metadata is incomplete")
    raw_upstream = manifest["upstream"]
    raw_roots = manifest["artifact_roots"]
    if not isinstance(raw_upstream, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw_upstream.items()
    ):
        raise ValueError("milestone upstream map is malformed")
    upstream: dict[CheckpointKind, str] = {}
    for raw_kind, raw_fingerprint in raw_upstream.items():
        try:
            active_kind = CheckpointKind(raw_kind)
        except ValueError as error:
            raise ValueError(f"unknown upstream kind: {raw_kind}") from error
        upstream[active_kind] = _require_digest(
            raw_fingerprint,
            f"{active_kind.value} upstream fingerprint",
        )
    if not isinstance(raw_roots, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw_roots.items()
    ):
        raise ValueError("milestone artifact roots are malformed")
    roots: dict[str, str] = {}
    for label, raw_relative in raw_roots.items():
        if (
            not label
            or label in {".", ".."}
            or any(separator in label for separator in ("/", "\\", ":"))
        ):
            raise ValueError("milestone artifact root label is malformed")
        relative = _safe_relative_path(raw_relative)
        if relative.parts != ("artifacts", label):
            raise ValueError("milestone artifact root path is malformed")
        roots[label] = relative.as_posix()
    raw_external_labels = manifest.get("external_root_labels", [])
    if not isinstance(raw_external_labels, list):
        raise ValueError("milestone external root labels are malformed")
    external_labels: list[str] = []
    for label in raw_external_labels:
        if (
            not isinstance(label, str)
            or not label
            or label in {".", ".."}
            or any(separator in label for separator in ("/", "\\", ":"))
            or label in external_labels
            or label in roots
        ):
            raise ValueError("milestone external root label is malformed")
        external_labels.append(label)
    return (
        MappingProxyType(
            dict(sorted(upstream.items(), key=lambda item: item[0].value))
        ),
        MappingProxyType(dict(sorted(roots.items()))),
        tuple(sorted(external_labels)),
    )


def _read_generation_manifest(
    generation: Path,
    kind: CheckpointKind,
    fingerprint: str,
) -> dict[str, object]:
    root = _regular_directory(generation, "generation")
    manifest_path = _regular_file(root / "manifest.json", "manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("generation manifest must be an object")
    if manifest.get("kind") != kind.value or manifest.get("fingerprint") != fingerprint:
        raise ValueError("generation manifest identity is malformed")
    return manifest


@dataclass(frozen=True)
class _ManifestEntry:
    relative_path: PurePosixPath
    size_bytes: int
    sha256: str


def _manifest_file_rows(
    manifest: Mapping[str, object],
) -> tuple[_ManifestEntry, ...]:
    raw_rows = manifest.get("files")
    if not isinstance(raw_rows, list):
        raise ValueError("generation file manifest must be a list")
    rows: list[_ManifestEntry] = []
    observed: set[str] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict) or set(raw_row) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("generation file row is malformed")
        relative = _safe_relative_path(raw_row["relative_path"])
        if len(relative.parts) < 2 or relative.parts[0] != "payload":
            raise ValueError("generation files must be rooted below payload")
        relative_text = relative.as_posix()
        if relative_text in observed:
            raise ValueError("generation file paths must be unique")
        observed.add(relative_text)
        size = raw_row["size_bytes"]
        if type(size) is not int or size < 0:
            raise ValueError("generation file size is malformed")
        digest = _require_digest(raw_row["sha256"], "generation file digest")
        rows.append(_ManifestEntry(relative, size, digest))
    if [row.relative_path.as_posix() for row in rows] != sorted(observed):
        raise ValueError("generation file rows must be canonically sorted")
    return tuple(rows)


def _payload_metadata_inventory(
    root: Path,
) -> tuple[tuple[str, int], ...]:
    base = _regular_directory(root, "payload root")
    rows: list[tuple[str, int]] = []
    for path in sorted(base.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("checkpoint trees cannot contain symlinks")
        if path.is_dir():
            continue
        file_path = _regular_file(path, "checkpoint artifact")
        relative = _safe_relative_path(
            f"payload/{file_path.relative_to(base).as_posix()}"
        )
        rows.append((relative.as_posix(), file_path.stat().st_size))
    return tuple(rows)


def _read_verified_generation_metadata(
    generation: Path,
    kind: CheckpointKind,
    fingerprint: str,
) -> dict[str, object]:
    root = _regular_directory(generation, "generation")
    manifest_path = _regular_file(root / "manifest.json", "manifest")
    success_path = _regular_file(root / "_SUCCESS.json", "success marker")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(success, dict):
        raise ValueError("generation metadata must contain JSON objects")
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("generation schema version is unsupported")
    if manifest.get("kind") != kind.value or manifest.get("fingerprint") != fingerprint:
        raise ValueError("generation identity is malformed")
    base_keys = {"schema_version", "kind", "fingerprint", "files"}
    metadata_keys = {"upstream", "artifact_roots"}
    external_keys = {"external_root_labels"}
    manifest_keys = frozenset(manifest)
    if manifest_keys == frozenset(base_keys):
        upstream_payload = None
    elif manifest_keys in {
        frozenset(base_keys | metadata_keys),
        frozenset(base_keys | metadata_keys | external_keys),
    }:
        upstream, _, _ = _manifest_metadata(manifest)
        upstream_payload = {
            upstream_kind.value: upstream_fingerprint
            for upstream_kind, upstream_fingerprint in upstream.items()
        }
    else:
        raise ValueError("generation manifest fields are malformed")
    rows = _manifest_file_rows(manifest)
    expected_success = {
        "schema_version": _SCHEMA_VERSION,
        "kind": kind.value,
        "fingerprint": fingerprint,
        "manifest_sha256": _sha256(manifest_path),
    }
    if upstream_payload is not None:
        expected_success["upstream"] = upstream_payload
    if success != expected_success:
        raise ValueError("generation success marker is malformed")
    expected_metadata = tuple(
        (row.relative_path.as_posix(), row.size_bytes) for row in rows
    )
    if _payload_metadata_inventory(root / "payload") != expected_metadata:
        raise ValueError("generation payload metadata differs from its manifest")
    return manifest


class _RestoreProgress:
    def __init__(self, total_bytes: int) -> None:
        self._total_bytes = total_bytes
        self._verified_bytes = 0
        self._next_report = _RESTORE_PROGRESS_STEP_BYTES
        self._lock = Lock()

    def add_verified(self, byte_count: int) -> None:
        with self._lock:
            self._verified_bytes += byte_count
            while self._verified_bytes >= self._next_report:
                print(
                    f"[CACHE RESTORE] {self._next_report}/{self._total_bytes} "
                    f"bytes verified ({self._next_report / (1024**3):.2f}/"
                    f"{self._total_bytes / (1024**3):.2f} GiB)",
                    flush=True,
                )
                self._next_report += _RESTORE_PROGRESS_STEP_BYTES


def _restore_worker_count(max_workers: object, file_count: int) -> int:
    if (
        type(max_workers) is not int
        or max_workers < 1
        or max_workers > _RESTORE_MAX_WORKERS
    ):
        raise ValueError("max_workers must be a plain integer from 1 through 16")
    if file_count == 0:
        return 0
    return min(max_workers, file_count)


def _prepare_restore_target(target: Path, rows: Sequence[_ManifestEntry]) -> None:
    paths = [PurePosixPath(*row.relative_path.parts[1:]) for row in rows]
    rendered = {path.as_posix() for path in paths}
    for path in paths:
        for depth in range(1, len(path.parts)):
            if PurePosixPath(*path.parts[:depth]).as_posix() in rendered:
                raise ValueError("restore destination has file/directory conflict")
    target.mkdir()
    for path in paths:
        copied = target.joinpath(*path.parts)
        try:
            copied.relative_to(target)
        except ValueError as error:
            raise ValueError("restore destination escapes the restore root") from error
        copied.parent.mkdir(parents=True, exist_ok=True)


def _copy_verified_file(
    source_root: Path,
    target: Path,
    row: _ManifestEntry,
    cancellation: Event,
    progress: _RestoreProgress,
) -> int:
    payload_relative = PurePosixPath(*row.relative_path.parts[1:])
    source_file = _regular_file(
        source_root.joinpath(*payload_relative.parts), "restore source"
    )
    source_file.relative_to(source_root)
    if source_file.stat().st_size != row.size_bytes:
        raise ValueError(f"restore source size mismatch: {payload_relative.as_posix()}")
    copied = target.joinpath(*payload_relative.parts)
    digest = hashlib.sha256()
    copied_size = 0
    with source_file.open("rb") as source_stream, copied.open("xb") as target_stream:
        while not cancellation.is_set():
            block = source_stream.read(_RESTORE_BLOCK_BYTES)
            if not block:
                break
            copied_size += len(block)
            digest.update(block)
            target_stream.write(block)
    if cancellation.is_set():
        return copied_size
    if copied_size != row.size_bytes:
        raise ValueError(f"restore source size changed: {payload_relative.as_posix()}")
    if digest.hexdigest() != row.sha256:
        raise ValueError(f"restore source hash mismatch: {payload_relative.as_posix()}")
    progress.add_verified(copied_size)
    return copied_size


def _copy_verified_payload(
    generation: Path,
    destination: Path,
    manifest: Mapping[str, object],
    *,
    max_workers: int = _RESTORE_DEFAULT_WORKERS,
) -> None:
    rows = _manifest_file_rows(manifest)
    active_workers = _restore_worker_count(max_workers, len(rows))
    source_root = _regular_directory(generation / "payload", "payload root")
    target = Path(destination)
    if os.path.lexists(target):
        raise FileExistsError(f"restore destination exists: {target}")
    _prepare_restore_target(target, rows)
    total_bytes = sum(row.size_bytes for row in rows)
    progress = _RestoreProgress(total_bytes)
    print(
        f"[CACHE RESTORE] Streaming {len(rows)} files with {active_workers} workers "
        f"({total_bytes / (1024**3):.2f} GiB) from Drive...",
        flush=True,
    )
    cancellation = Event()
    active: dict[Future[int], _ManifestEntry] = {}
    rows_iter = iter(rows)
    first_error: BaseException | None = None
    if active_workers:
        with ThreadPoolExecutor(max_workers=active_workers) as executor:
            for _ in range(active_workers):
                try:
                    row = next(rows_iter)
                except StopIteration:
                    break
                future = executor.submit(
                    _copy_verified_file, source_root, target, row, cancellation, progress
                )
                active[future] = row
            while active:
                completed, _pending = wait(tuple(active), return_when=FIRST_COMPLETED)
                for future in completed:
                    del active[future]
                    try:
                        future.result()
                    except BaseException as error:
                        if first_error is None:
                            first_error = error
                            cancellation.set()
                if first_error is not None:
                    for future in active:
                        future.cancel()
                    break
                for _ in completed:
                    try:
                        row = next(rows_iter)
                    except StopIteration:
                        break
                    replacement = executor.submit(
                        _copy_verified_file,
                        source_root,
                        target,
                        row,
                        cancellation,
                        progress,
                    )
                    active[replacement] = row
    if first_error is not None:
        raise first_error
    expected_metadata = tuple(
        (row.relative_path.as_posix(), row.size_bytes) for row in rows
    )
    actual_metadata = _payload_metadata_inventory(target)
    if actual_metadata != expected_metadata:
        raise ValueError("restored checkpoint metadata differs from Drive")
    print("[CACHE RESTORE] Verified streaming copy complete.", flush=True)


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a regular non-symlink directory")
    return path.resolve(strict=True)


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _inventory(
    root: Path, *, relative_prefix: str | None = None
) -> list[dict[str, object]]:
    base = _regular_directory(root, "inventory root")
    rows: list[dict[str, object]] = []
    for path in sorted(base.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError("checkpoint trees cannot contain symlinks")
        if path.is_dir():
            continue
        file_path = _regular_file(path, "checkpoint artifact")
        relative = file_path.relative_to(base).as_posix()
        if relative_prefix is not None:
            relative = f"{relative_prefix}/{relative}"
        _safe_relative_path(relative)
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": file_path.stat().st_size,
                "sha256": _sha256(file_path),
            }
        )
    return rows


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("checkpoint path must be a safe POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("checkpoint path must be a safe POSIX relative path")
    return path


def _path_relative_to_root(path: Path, root: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("checkpoint artifact paths must be absolute Paths")
    if candidate.is_symlink():
        raise ValueError("checkpoint artifact paths cannot be symlinks")
    resolved = candidate.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("checkpoint artifact path escapes snapshot_root") from error
    for depth in range(1, len(relative.parts) + 1):
        if root.joinpath(*relative.parts[:depth]).is_symlink():
            raise ValueError("checkpoint artifact paths cannot contain symlinks")
    return _safe_relative_path(relative.as_posix()).as_posix()


def _resolved_restore_path(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise ValueError("restored checkpoint paths cannot be symlinks")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("restored checkpoint path escapes restore_root") from error
    for depth in range(1, len(relative.parts) + 1):
        if root.joinpath(*relative.parts[:depth]).is_symlink():
            raise ValueError("restored checkpoint paths cannot contain symlinks")
    return resolved


def _canonical_sort_key(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _require_keys(payload: Mapping[str, object], expected: set[str]) -> None:
    if set(payload) != expected:
        raise ValueError("checkpoint tagged value has unexpected fields")


def _require_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("checkpoint items must be a list")
    return value


def _require_object_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("o")
        or not value[1:].isdigit()
    ):
        raise ValueError("checkpoint object_id is malformed")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

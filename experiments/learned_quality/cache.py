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
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from backend.static_pipeline.sources import _atomic_promote_no_replace
from backend.static_pipeline.stage_cache import stage_fingerprint

from .contracts import GENERATOR_ID


_SCHEMA_VERSION = 1
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_UNSUPPORTED_NOREPLACE_ERRNOS = frozenset(
    {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}
)
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
) -> object:
    root = _regular_directory(Path(restore_root), "restore_root")
    return _CheckpointDecoder(root, source_inventory).decode(payload)


class _CheckpointEncoder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._object_ids: dict[int, str] = {}

    def encode(self, value: object) -> object:
        if value is None or isinstance(value, (bool, str, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("checkpoint floats must be finite")
            return value
        if isinstance(value, Path):
            return {
                "__kind__": "path",
                "relative_path": _path_relative_to_root(value, self.root),
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
    def __init__(self, root: Path, source_inventory: object) -> None:
        self.root = root
        self.source_inventory = source_inventory
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
    from .flow import FlowGatePolicy, MotionEvidence, MotionFrameEvidence
    from .lifecycle import BatchAttemptRecord
    from .masks import FusedMaskFrame, MaskFusionEvidence, MaskFusionPolicy
    from .photometric import PhotometricEvidence, RgbAffineTransform
    from .segmentation import SemanticEvidence, SemanticFrameEvidence, SemanticPolicy

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
    ) -> Path:
        active_kind = _require_kind(kind)
        active_fingerprint = _require_digest(fingerprint, "fingerprint")
        if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("run_id must be a safe non-empty identifier")
        source = _regular_directory(Path(source_root), "source_root")
        _inventory(source)
        self._ensure_owned_root(create=True)

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
            manifest_path = staging / "manifest.json"
            _write_json(manifest_path, manifest)
            _write_json(
                staging / "_SUCCESS.json",
                {
                    "schema_version": _SCHEMA_VERSION,
                    "kind": active_kind.value,
                    "fingerprint": active_fingerprint,
                    "manifest_sha256": _sha256(manifest_path),
                },
            )
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
            if os.path.lexists(staging):
                self._remove_owned_tree(staging)
            raise

        self._remove_older_generations(active_kind, keep=target)
        return target

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

        generation = self.find_generation(CheckpointKind.COLMAP, fingerprint)
        if generation is None:
            return None
        target = Path(destination)
        if os.path.lexists(target):
            raise FileExistsError(f"COLMAP restore destination exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = generation / "payload"
        try:
            shutil.copytree(source, target, copy_function=shutil.copy2)
            if _inventory(source) != _inventory(target):
                raise ValueError("restored COLMAP checkpoint differs from Drive")
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
            source = generation / "payload"
            try:
                shutil.copytree(source, target, copy_function=shutil.copy2)
                if _inventory(source) != _inventory(target):
                    raise ValueError(
                        "restored pretraining checkpoint differs from Drive"
                    )
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
        parent = self.cache_root / CheckpointKind.PRETRAINING.value
        if not parent.is_dir():
            return ()
        result = []
        for candidate in sorted(parent.iterdir(), key=lambda item: item.name):
            try:
                fingerprint = _require_digest(candidate.name, "fingerprint")
            except ValueError:
                continue
            if self._valid_generation(
                candidate,
                CheckpointKind.PRETRAINING,
                fingerprint,
            ):
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
        if not os.path.lexists(target):
            return None
        if not self._valid_generation(target, active_kind, active_fingerprint):
            return None
        return target

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
            root = _regular_directory(generation, "generation")
            manifest_path = _regular_file(root / "manifest.json", "manifest")
            success_path = _regular_file(root / "_SUCCESS.json", "success marker")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            success = json.loads(success_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or not isinstance(success, dict):
                return False
            if manifest.get("schema_version") != _SCHEMA_VERSION:
                return False
            if manifest.get("kind") != kind.value:
                return False
            if manifest.get("fingerprint") != fingerprint:
                return False
            if success != {
                "schema_version": _SCHEMA_VERSION,
                "kind": kind.value,
                "fingerprint": fingerprint,
                "manifest_sha256": _sha256(manifest_path),
            }:
                return False
            expected = manifest.get("files")
            if not isinstance(expected, list):
                return False
            actual = _inventory(root / "payload", relative_prefix="payload")
            return expected == actual
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return False

    def _remove_older_generations(self, kind: CheckpointKind, *, keep: Path) -> None:
        parent = self.cache_root / kind.value
        for candidate in parent.iterdir():
            if candidate != keep:
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

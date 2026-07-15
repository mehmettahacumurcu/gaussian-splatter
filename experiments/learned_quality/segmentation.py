from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
import weakref
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Final, Protocol, TypeVar, cast

import numpy as np
from PIL import Image

from backend.static_pipeline.stage_cache import promote_directory

from .contracts import FrameArtifact, ModelRef, StageRecord
from .dependencies import CHECKPOINT_MODEL_REFS
from .lifecycle import (
    BatchAttemptRecord,
    BatchRetryResult,
    VramReleaseRecord,
    run_with_smaller_batch_retry,
)


TRANSIENT_PROMPT: Final[str] = (
    "person. child. hand. dog. cat. animal. bicycle. motorcycle. "
    "scooter. car. truck. bus."
)


def _model_ref(repo_id: str) -> ModelRef:
    matches = tuple(model for model in CHECKPOINT_MODEL_REFS if model.repo_id == repo_id)
    if len(matches) != 1:
        raise RuntimeError(f"expected one pinned model ref for {repo_id}")
    return matches[0]


GROUNDING_DINO_MODEL_REF: Final[ModelRef] = _model_ref(
    "IDEA-Research/grounding-dino-tiny"
)
SAM_MODEL_REF: Final[ModelRef] = _model_ref("facebook/sam2.1-hiera-large")


@dataclass(frozen=True)
class SemanticPolicy:
    dino_box_threshold: float
    dino_text_threshold: float
    propagation_clip_frames: int
    propagation_neighbor_frames: int


@dataclass(frozen=True)
class DetectionBox:
    box_id: str
    xyxy: tuple[float, float, float, float]
    score: float
    phrase: str


@dataclass(frozen=True)
class SamBoxPrompt:
    frame: FrameArtifact
    box: DetectionBox


@dataclass(frozen=True)
class SemanticFrameEvidence:
    frame: FrameArtifact
    width: int
    height: int
    direct_confirmed_path: Path
    direct_confirmed_sha256: str
    propagated_candidate_path: Path
    propagated_candidate_sha256: str
    direct_support_path: Path
    direct_support_sha256: str


@dataclass(frozen=True)
class SemanticEvidence:
    policy: SemanticPolicy
    prompt: str
    frames: tuple[SemanticFrameEvidence, ...]
    stage_records: tuple[StageRecord, ...]
    manifest_path: Path
    manifest_sha256: str


class GroundingDinoAdapter(Protocol):
    model_ref: ModelRef

    def detect_batch(
        self,
        frames: tuple[FrameArtifact, ...],
        *,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Mapping[str, tuple[DetectionBox, ...]]: ...


class SamAdapter(Protocol):
    model_ref: ModelRef

    def refine_batch(
        self,
        prompts: tuple[SamBoxPrompt, ...],
    ) -> Mapping[tuple[str, str], object]: ...

    def propagate(
        self,
        frames: tuple[FrameArtifact, ...],
        direct_masks: Mapping[str, object],
        *,
        clip_frames: int,
    ) -> Mapping[str, tuple[object, tuple[str, ...]]]: ...


_SAFE_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_FORBIDDEN = frozenset('<>:"\\|?*')
_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


@dataclass(frozen=True)
class _FrameSnapshot:
    size_bytes: int
    sha256: str
    shape: tuple[int, int]


def _validate_policy(policy: SemanticPolicy) -> None:
    if not isinstance(policy, SemanticPolicy):
        raise ValueError("semantic policy is malformed")
    for name, value in (
        ("dino_box_threshold", policy.dino_box_threshold),
        ("dino_text_threshold", policy.dino_text_threshold),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{name} must be finite and within [0, 1]")
    if (
        type(policy.propagation_clip_frames) is not int
        or policy.propagation_clip_frames <= 0
    ):
        raise ValueError("propagation_clip_frames must be a positive plain integer")
    if (
        type(policy.propagation_neighbor_frames) is not int
        or policy.propagation_neighbor_frames < 0
    ):
        raise ValueError(
            "propagation_neighbor_frames must be a nonnegative plain integer"
        )


def _validate_relative_image_name(value: str) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ValueError("frame image_name must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("frame image_name must be a safe relative POSIX path")
    for part in path.parts:
        reserved_stem = part.split(".", 1)[0].upper()
        if (
            part != part.strip()
            or part.endswith(".")
            or any(character in _WINDOWS_FORBIDDEN for character in part)
            or reserved_stem in _WINDOWS_RESERVED
        ):
            raise ValueError("frame image_name must be a portable relative POSIX path")
    return tuple(part.casefold() for part in path.parts)


def _validate_frames(
    frames: tuple[FrameArtifact, ...],
    *,
    expected: tuple[_FrameSnapshot, ...] | None = None,
) -> tuple[_FrameSnapshot, ...]:
    if not isinstance(frames, tuple) or not frames:
        raise ValueError("frames must be a non-empty immutable tuple")
    if expected is not None and len(expected) != len(frames):
        raise ValueError("frame snapshot count changed")
    frame_ids: set[str] = set()
    image_name_keys: set[tuple[str, ...]] = set()
    resolved_paths: set[Path] = set()
    snapshots: list[_FrameSnapshot] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, FrameArtifact):
            raise ValueError("frame record is malformed")
        if not isinstance(frame.frame_id, str) or _SAFE_ID.fullmatch(frame.frame_id) is None:
            raise ValueError("frame_id must be a safe plain identifier")
        image_name_key = _validate_relative_image_name(frame.image_name)
        if frame.frame_id in frame_ids or image_name_key in image_name_keys:
            raise ValueError("frame identities and image names must be unique")
        frame_ids.add(frame.frame_id)
        image_name_keys.add(image_name_key)
        if not isinstance(frame.path, Path) or not frame.path.is_absolute():
            raise ValueError("frame path must be absolute")
        try:
            resolved_path = frame.path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("frame path must resolve to a regular file") from exc
        if resolved_path != frame.path:
            raise ValueError("frame path must not contain symlink or lexical aliases")
        if resolved_path in resolved_paths:
            raise ValueError("frame paths must resolve uniquely")
        resolved_paths.add(resolved_path)
        if frame.path.is_symlink() or not frame.path.is_file():
            raise ValueError("frame path must be a regular non-symlink file")
        if not isinstance(frame.sha256, str) or _SHA256.fullmatch(frame.sha256) is None:
            raise ValueError("frame sha256 is malformed")
        try:
            data = frame.path.read_bytes()
        except OSError as exc:
            raise ValueError(f"frame image is unreadable for {frame.frame_id}") from exc
        size_bytes = len(data)
        digest = sha256(data).hexdigest()
        prior = None if expected is None else expected[index]
        if prior is not None and size_bytes != prior.size_bytes:
            raise ValueError(f"frame size changed for {frame.frame_id}")
        if prior is not None and digest != prior.sha256:
            raise ValueError(f"frame digest changed for {frame.frame_id}")
        if digest != frame.sha256:
            raise ValueError(f"frame digest changed for {frame.frame_id}")
        try:
            with Image.open(BytesIO(data)) as image:
                image.verify()
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
        except (OSError, ValueError) as exc:
            raise ValueError(f"frame image is unreadable for {frame.frame_id}") from exc
        if width <= 0 or height <= 0:
            raise ValueError(f"frame dimensions are invalid for {frame.frame_id}")
        shape = (height, width)
        if prior is not None and shape != prior.shape:
            raise ValueError(f"frame dimensions changed for {frame.frame_id}")
        try:
            if frame.path.stat().st_size != size_bytes:
                raise ValueError(f"frame size changed for {frame.frame_id}")
        except OSError as exc:
            raise ValueError(f"frame image is unreadable for {frame.frame_id}") from exc
        snapshots.append(
            _FrameSnapshot(
                size_bytes=size_bytes,
                sha256=digest,
                shape=shape,
            )
        )
    return tuple(snapshots)


def _validate_output_dir(output_dir: Path) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        raise ValueError("semantic output directory must be absolute")
    if path.is_symlink() or path.exists():
        raise ValueError("semantic output directory must be a new plain path")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve() != parent:
        raise ValueError("semantic output parent must be a plain directory")
    return path


_ChunkValueT = TypeVar("_ChunkValueT")


def _chunks(
    values: tuple[_ChunkValueT, ...],
    size: int,
) -> tuple[tuple[_ChunkValueT, ...], ...]:
    if type(size) is not int or size <= 0:
        raise ValueError("batch size must be a positive plain integer")
    return tuple(values[index : index + size] for index in range(0, len(values), size))


def _validate_model_ref(actual: object, expected: ModelRef, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} adapter model ref does not match the pinned model")


def _validate_box(
    box: DetectionBox,
    *,
    width: int,
    height: int,
    threshold: float,
) -> None:
    if not isinstance(box, DetectionBox):
        raise ValueError("detector box is malformed")
    if not isinstance(box.box_id, str) or _SAFE_ID.fullmatch(box.box_id) is None:
        raise ValueError("detector box_id is malformed")
    if not isinstance(box.xyxy, tuple) or len(box.xyxy) != 4:
        raise ValueError("detector box coordinates are malformed")
    try:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy)
    except (TypeError, ValueError) as exc:
        raise ValueError("detector box coordinates are malformed") from exc
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("detector box coordinates must be finite")
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError("detector box is outside the frame")
    if (
        isinstance(box.score, bool)
        or not isinstance(box.score, (int, float))
        or not math.isfinite(float(box.score))
        or not threshold <= float(box.score) <= 1.0
    ):
        raise ValueError("detector box score is invalid or below policy threshold")
    if (
        not isinstance(box.phrase, str)
        or not box.phrase.strip()
        or any(unicodedata.category(character) == "Cc" for character in box.phrase)
    ):
        raise ValueError("detector phrase is malformed")


def _as_binary_mask(
    value: object,
    *,
    shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{label} shape does not match the frame")
    if array.dtype == np.bool_:
        return np.array(array, dtype=np.bool_, copy=True)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label} must be a binary numeric map")
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise ValueError(f"{label} must be a binary numeric map") from exc
    if not bool(np.all(finite)) or not bool(np.all((array == 0) | (array == 1))):
        raise ValueError(f"{label} must contain only finite zero/one values")
    return np.array(array, dtype=np.bool_, copy=True)


def _run_detector(
    frames: tuple[FrameArtifact, ...],
    shapes: tuple[tuple[int, int], ...],
    policy: SemanticPolicy,
    detector: GroundingDinoAdapter,
    batch_size: int,
) -> tuple[tuple[DetectionBox, ...], ...]:
    _validate_model_ref(detector.model_ref, GROUNDING_DINO_MODEL_REF, "detector")
    frame_index = {frame.frame_id: index for index, frame in enumerate(frames)}
    detected: list[tuple[DetectionBox, ...] | None] = [None] * len(frames)
    for batch in _chunks(frames, batch_size):
        result = detector.detect_batch(
            batch,
            prompt=TRANSIENT_PROMPT,
            box_threshold=float(policy.dino_box_threshold),
            text_threshold=float(policy.dino_text_threshold),
        )
        expected_ids = tuple(frame.frame_id for frame in batch)
        if not isinstance(result, Mapping) or tuple(result) != expected_ids:
            raise ValueError("detector frame join does not match the requested batch")
        for frame in batch:
            boxes = result[frame.frame_id]
            if not isinstance(boxes, tuple):
                raise ValueError("detector boxes must use immutable tuples")
            index = frame_index[frame.frame_id]
            height, width = shapes[index]
            seen_box_ids: set[str] = set()
            for box in boxes:
                _validate_box(
                    box,
                    width=width,
                    height=height,
                    threshold=float(policy.dino_box_threshold),
                )
                if box.box_id in seen_box_ids:
                    raise ValueError("detector box ids must be unique within a frame")
                seen_box_ids.add(box.box_id)
            detected[index] = boxes
    if any(boxes is None for boxes in detected):
        raise ValueError("detector omitted a requested frame")
    return tuple(boxes for boxes in detected if boxes is not None)


def _run_sam(
    frames: tuple[FrameArtifact, ...],
    shapes: tuple[tuple[int, int], ...],
    detections: tuple[tuple[DetectionBox, ...], ...],
    policy: SemanticPolicy,
    sam: SamAdapter,
    batch_size: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    _validate_model_ref(sam.model_ref, SAM_MODEL_REF, "SAM")
    frame_index = {frame.frame_id: index for index, frame in enumerate(frames)}
    direct = [np.zeros(shape, dtype=np.bool_) for shape in shapes]
    prompts = tuple(
        SamBoxPrompt(frame=frame, box=box)
        for frame, boxes in zip(frames, detections, strict=True)
        for box in boxes
    )
    for batch in _chunks(prompts, batch_size) if prompts else ():
        refined = sam.refine_batch(batch)
        expected_keys = tuple(
            (prompt.frame.frame_id, prompt.box.box_id) for prompt in batch
        )
        if not isinstance(refined, Mapping) or tuple(refined) != expected_keys:
            raise ValueError("SAM box join does not match the requested prompts")
        for prompt in batch:
            key = (prompt.frame.frame_id, prompt.box.box_id)
            index = frame_index[prompt.frame.frame_id]
            direct[index] |= _as_binary_mask(
                refined[key],
                shape=shapes[index],
                label=f"SAM direct mask {key}",
            )

    clips = _chunks(frames, policy.propagation_clip_frames)
    clip_index_by_frame_id = {
        frame.frame_id: clip_index
        for clip_index, clip in enumerate(clips)
        for frame in clip
    }
    propagated: list[np.ndarray | None] = [None] * len(frames)
    direct_support: list[np.ndarray | None] = [None] * len(frames)
    for clip_batch in _chunks(clips, batch_size):
        batch_frames = tuple(frame for clip in clip_batch for frame in clip)
        batch_direct = MappingProxyType(
            {
                frame.frame_id: np.array(direct[frame_index[frame.frame_id]], copy=True)
                for frame in batch_frames
            }
        )
        propagated_result = sam.propagate(
            batch_frames,
            batch_direct,
            clip_frames=policy.propagation_clip_frames,
        )
        expected_frame_ids = tuple(frame.frame_id for frame in batch_frames)
        if (
            not isinstance(propagated_result, Mapping)
            or tuple(propagated_result) != expected_frame_ids
        ):
            raise ValueError("SAM propagation frame join does not match exact clips")

        for frame in batch_frames:
            index = frame_index[frame.frame_id]
            item = propagated_result[frame.frame_id]
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[1], tuple)
            ):
                raise ValueError("SAM propagation result is malformed")
            candidate = _as_binary_mask(
                item[0],
                shape=shapes[index],
                label=f"SAM propagated candidate {frame.frame_id}",
            )
            supporting_ids = item[1]
            if len(set(supporting_ids)) != len(supporting_ids):
                raise ValueError("SAM propagation support ids must be unique")
            has_nearby_direct = False
            for supporting_id in supporting_ids:
                source_index = frame_index.get(supporting_id)
                if source_index is None:
                    raise ValueError("SAM propagation references an unknown source frame")
                if (
                    clip_index_by_frame_id[supporting_id]
                    != clip_index_by_frame_id[frame.frame_id]
                ):
                    raise ValueError("SAM propagation support crosses an exact clip boundary")
                if not bool(np.any(direct[source_index])):
                    raise ValueError("SAM propagation source has no direct evidence")
                if abs(source_index - index) > policy.propagation_neighbor_frames:
                    raise ValueError("SAM propagation source is outside the neighbor policy")
                has_nearby_direct = True
            propagated[index] = candidate
            direct_support[index] = np.array(
                candidate if has_nearby_direct else np.zeros_like(candidate),
                copy=True,
            )
    if any(mask is None for mask in propagated) or any(
        mask is None for mask in direct_support
    ):
        raise ValueError("SAM propagation omitted an exact clip frame")
    return tuple(
        (
            np.array(direct[index], copy=True),
            cast(np.ndarray, propagated[index]),
            cast(np.ndarray, direct_support[index]),
        )
        for index in range(len(frames))
    )


def _sha256_path(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_png(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
        path,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    with path.open("rb+") as stream:
        os.fsync(stream.fileno())
    return _sha256_path(path)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if os.name == "nt":
            return
        raise
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            if os.name != "nt":
                raise
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())
    for directory in sorted(
        directories,
        key=lambda path: len(path.relative_to(root).parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def _model_payload(model: ModelRef) -> dict[str, object]:
    return {
        "code_commit": model.code_commit,
        "license_id": model.license_id,
        "repo_id": model.repo_id,
        "revision": model.revision,
    }


def _attempt_payload(attempt: BatchAttemptRecord) -> dict[str, object]:
    return asdict(attempt)


def _release_payload(release: VramReleaseRecord) -> dict[str, object]:
    return asdict(release)


def _freeze_record_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_record_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_record_value(item) for item in value)
    return value


def _json_record_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_record_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_record_value(item) for item in value]
    return value


_AdapterT = TypeVar("_AdapterT")
_StageValueT = TypeVar("_StageValueT")
_UNSET_MODEL = object()


def _attach_model_release_error(
    primary: BaseException,
    cleanup: BaseException,
) -> None:
    try:
        setattr(primary, "model_release_error", cleanup)
    except BaseException:
        pass
    try:
        add_note = getattr(primary, "add_note", None)
        if callable(add_note):
            add_note(f"model release also failed: {cleanup!r}")
    except BaseException:
        pass


def _run_model_stage(
    stage_id: str,
    *,
    factory: Callable[[], _AdapterT],
    operation: Callable[[_AdapterT, int], _StageValueT],
    initial_batch_size: int,
    retry_batch_size: int,
    release_model: Callable[[object], VramReleaseRecord],
) -> tuple[BatchRetryResult[_StageValueT], VramReleaseRecord]:
    current: _AdapterT | object = _UNSET_MODEL
    released_refs: list[weakref.ReferenceType[object]] = []
    non_weakref_released_ids: set[int] = set()

    def was_released(candidate: object) -> bool:
        if id(candidate) in non_weakref_released_ids:
            return True
        return any(reference() is candidate for reference in released_refs)

    def remember_released(model: object) -> None:
        try:
            released_refs.append(weakref.ref(model))
        except TypeError:
            non_weakref_released_ids.add(id(model))

    def run(batch_size: int) -> _StageValueT:
        nonlocal current
        if current is _UNSET_MODEL:
            candidate = factory()
            if was_released(candidate):
                raise RuntimeError(
                    f"{stage_id} factory returned a previously released adapter"
                )
            current = candidate
        return operation(cast(_AdapterT, current), batch_size)

    def release_current() -> VramReleaseRecord:
        nonlocal current
        if current is _UNSET_MODEL:
            raise RuntimeError(f"{stage_id} has no model instance to release")
        model = current
        current = _UNSET_MODEL
        remember_released(model)
        record = release_model(model)
        if not isinstance(record, VramReleaseRecord):
            raise TypeError("release_model must return VramReleaseRecord")
        return record

    retry_result: BatchRetryResult[_StageValueT] | None = None
    final_release: VramReleaseRecord | None = None
    try:
        retry_result = run_with_smaller_batch_retry(
            stage_id,
            initial_batch_size,
            retry_batch_size,
            run,
            chunk_independent=True,
            release=release_current,
        )
    except BaseException as primary:
        if current is not _UNSET_MODEL:
            try:
                release_current()
            except BaseException as cleanup:
                _attach_model_release_error(primary, cleanup)
        raise
    else:
        if current is not _UNSET_MODEL:
            final_release = release_current()
    if retry_result is None or final_release is None:
        raise RuntimeError(f"{stage_id} completed without lifecycle evidence")
    return retry_result, final_release


def _stage_record(
    stage_id: str,
    model_ref: ModelRef,
    attempts: tuple[BatchAttemptRecord, ...],
    retry_release: VramReleaseRecord | None,
    release: VramReleaseRecord,
) -> StageRecord:
    details: dict[str, object] = {
        "attempts": tuple(_attempt_payload(attempt) for attempt in attempts),
        "model": _model_payload(model_ref),
        "release": _release_payload(release),
    }
    if retry_release is not None:
        details["retry_release"] = _release_payload(retry_release)
    return StageRecord(
        stage_id=stage_id,
        status="fallback" if retry_release is not None else "accepted",
        details=cast(Mapping[str, object], _freeze_record_value(details)),
    )


def _stage_payload(record: StageRecord) -> dict[str, object]:
    return {
        "details": _json_record_value(record.details),
        "peak_vram_bytes": record.peak_vram_bytes,
        "stage_id": record.stage_id,
        "status": record.status,
        "vram_after_bytes": record.vram_after_bytes,
        "vram_before_bytes": record.vram_before_bytes,
    }


def _frame_set_digest(frames: tuple[FrameArtifact, ...]) -> str:
    data = _json_bytes(
        [
            {
                "frame_id": frame.frame_id,
                "image_name": frame.image_name,
                "path": str(frame.path),
                "sha256": frame.sha256,
            }
            for frame in frames
        ]
    )
    return sha256(data).hexdigest()


def _publish(
    output_dir: Path,
    frames: tuple[FrameArtifact, ...],
    shapes: tuple[tuple[int, int], ...],
    maps: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
    policy: SemanticPolicy,
    stage_records: tuple[StageRecord, ...],
) -> SemanticEvidence:
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", suffix=".staging", dir=output_dir.parent)
    )
    try:
        final_records: list[SemanticFrameEvidence] = []
        manifest_frames: list[dict[str, object]] = []
        for index, (frame, shape, frame_maps) in enumerate(
            zip(frames, shapes, maps, strict=True)
        ):
            height, width = shape
            relative_root = Path("maps") / f"{index:06d}_{frame.frame_id}"
            staged_paths = {
                "direct_confirmed": staging / relative_root / "direct_confirmed.png",
                "propagated_candidate": staging
                / relative_root
                / "propagated_candidate.png",
                "direct_support": staging / relative_root / "direct_support.png",
            }
            digests = {
                name: _write_png(path, mask)
                for (name, path), mask in zip(
                    staged_paths.items(), frame_maps, strict=True
                )
            }
            final_paths = {
                name: output_dir / path.relative_to(staging)
                for name, path in staged_paths.items()
            }
            record = SemanticFrameEvidence(
                frame=frame,
                width=width,
                height=height,
                direct_confirmed_path=final_paths["direct_confirmed"],
                direct_confirmed_sha256=digests["direct_confirmed"],
                propagated_candidate_path=final_paths["propagated_candidate"],
                propagated_candidate_sha256=digests["propagated_candidate"],
                direct_support_path=final_paths["direct_support"],
                direct_support_sha256=digests["direct_support"],
            )
            final_records.append(record)
            manifest_frames.append(
                {
                    "frame": {
                        "frame_id": frame.frame_id,
                        "image_name": frame.image_name,
                        "path": str(frame.path),
                        "sha256": frame.sha256,
                    },
                    "height": height,
                    "maps": {
                        name: {"path": str(final_paths[name]), "sha256": digests[name]}
                        for name in (
                            "direct_confirmed",
                            "propagated_candidate",
                            "direct_support",
                        )
                    },
                    "width": width,
                }
            )

        manifest_payload = {
            "frame_set_digest": _frame_set_digest(frames),
            "frames": manifest_frames,
            "models": {
                "grounding_dino": _model_payload(GROUNDING_DINO_MODEL_REF),
                "sam": _model_payload(SAM_MODEL_REF),
            },
            "policy": asdict(policy),
            "prompt": TRANSIENT_PROMPT,
            "schema_version": 1,
            "stage_records": [_stage_payload(record) for record in stage_records],
        }
        manifest_data = _json_bytes(manifest_payload)
        staged_manifest = staging / "semantic_manifest.json"
        _write_bytes(staged_manifest, manifest_data)
        _fsync_directory_tree(staging)
        _validate_output_dir(output_dir)
        promote_directory(staging, output_dir)
        _fsync_directory(output_dir.parent)
        manifest_path = output_dir / "semantic_manifest.json"
        return SemanticEvidence(
            policy=policy,
            prompt=TRANSIENT_PROMPT,
            frames=tuple(final_records),
            stage_records=stage_records,
            manifest_path=manifest_path,
            manifest_sha256=sha256(manifest_data).hexdigest(),
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_semantic_evidence(
    frames: tuple[FrameArtifact, ...],
    output_dir: Path,
    *,
    policy: SemanticPolicy,
    detector_factory: Callable[[], GroundingDinoAdapter],
    sam_factory: Callable[[], SamAdapter],
    initial_batch_size: int,
    retry_batch_size: int,
    release_model: Callable[[object], VramReleaseRecord],
) -> SemanticEvidence:
    _validate_policy(policy)
    frame_snapshots = _validate_frames(frames)
    shapes = tuple(snapshot.shape for snapshot in frame_snapshots)
    final_output = _validate_output_dir(output_dir)
    if type(initial_batch_size) is not int or initial_batch_size <= 0:
        raise ValueError("initial_batch_size must be a positive plain integer")
    if type(retry_batch_size) is not int or retry_batch_size <= 0:
        raise ValueError("retry_batch_size must be a positive plain integer")
    if retry_batch_size >= initial_batch_size:
        raise ValueError("retry_batch_size must be smaller than initial_batch_size")
    if not callable(detector_factory) or not callable(sam_factory):
        raise ValueError("model factories must be callable")
    if not callable(release_model):
        raise ValueError("release_model must be callable")

    detector_retry, detector_release = _run_model_stage(
        "semantic_dino",
        factory=detector_factory,
        operation=lambda detector, batch_size: _run_detector(
            frames,
            shapes,
            policy,
            detector,
            batch_size,
        ),
        initial_batch_size=initial_batch_size,
        retry_batch_size=retry_batch_size,
        release_model=release_model,
    )
    detections = detector_retry.value

    sam_retry, sam_release = _run_model_stage(
        "semantic_sam",
        factory=sam_factory,
        operation=lambda sam, batch_size: _run_sam(
            frames,
            shapes,
            detections,
            policy,
            sam,
            batch_size,
        ),
        initial_batch_size=initial_batch_size,
        retry_batch_size=retry_batch_size,
        release_model=release_model,
    )
    maps = sam_retry.value
    _validate_frames(frames, expected=frame_snapshots)

    return _publish(
        final_output,
        frames,
        shapes,
        maps,
        policy,
        (
            _stage_record(
                "semantic_dino",
                GROUNDING_DINO_MODEL_REF,
                detector_retry.attempts,
                detector_retry.release_record,
                detector_release,
            ),
            _stage_record(
                "semantic_sam",
                SAM_MODEL_REF,
                sam_retry.attempts,
                sam_retry.release_record,
                sam_release,
            ),
        ),
    )

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from PIL import Image

from .contracts import FrameArtifact
from .flow import StaticTrack, TrackObservation


TRACK_AUDIT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrackQualificationPolicy:
    minimum_observations: int = 3
    maximum_invalid_fraction: float = 0.01
    maximum_examples: int = 32

    def __post_init__(self) -> None:
        if type(self.minimum_observations) is not int or self.minimum_observations < 3:
            raise ValueError("minimum_observations must be a plain integer >= 3")
        if (
            isinstance(self.maximum_invalid_fraction, bool)
            or not isinstance(self.maximum_invalid_fraction, Real)
            or not math.isfinite(float(self.maximum_invalid_fraction))
            or not 0.0 <= float(self.maximum_invalid_fraction) <= 1.0
        ):
            raise ValueError("maximum_invalid_fraction must be finite and in [0, 1]")
        if type(self.maximum_examples) is not int or self.maximum_examples < 0:
            raise ValueError("maximum_examples must be a plain nonnegative integer")


@dataclass(frozen=True)
class TrackAuditReport:
    schema_version: int
    raw_track_count: int
    accepted_track_count: int
    rejected_track_count: int
    raw_observation_count: int
    accepted_observation_count: int
    rejected_observation_count: int
    rejected_tracks_by_reason: Mapping[str, int]
    rejected_observations_by_reason: Mapping[str, int]
    examples: tuple[Mapping[str, object], ...]
    accepted_tracks_sha256: str


@dataclass(frozen=True)
class QualifiedStaticTracks:
    tracks: tuple[StaticTrack, ...]
    report: TrackAuditReport
    audit_path: Path
    audit_sha256: str


class TrackQualificationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        report: TrackAuditReport,
        audit_path: Path,
    ) -> None:
        self.report = report
        self.audit_path = audit_path
        super().__init__(message)


@dataclass(frozen=True)
class _CanonicalFrame:
    artifact: FrameArtifact
    order: int
    width: int
    height: int


@dataclass(frozen=True)
class _ImagePoint:
    x: float | None
    y: float | None
    point_id: int | None


@dataclass(frozen=True)
class _ColmapImage:
    image_id: int
    image_name: str
    points: tuple[_ImagePoint, ...]


@dataclass(frozen=True)
class _RawTrack:
    row_index: int
    track_id: int | None
    xyz: tuple[float | None, float | None, float | None]
    error: float | None
    observations: tuple[tuple[int | None, int | None], ...]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_coordinate(value: float | None) -> float | str | None:
    if value is None:
        return None
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "inf"
    if value == -math.inf:
        return "-inf"
    return value


def _parse_images(model_dir: Path) -> tuple[dict[int, _ColmapImage], set[int]]:
    path = model_dir / "images.txt"
    lines = [
        raw.strip()
        for raw in path.read_text(encoding="utf-8").splitlines()
        if not raw.lstrip().startswith("#")
    ]
    images: dict[int, _ColmapImage] = {}
    duplicate_ids: set[int] = set()
    index = 0
    while index < len(lines):
        header_line = lines[index]
        index += 1
        if not header_line:
            continue
        header = header_line.split(maxsplit=9)
        if len(header) < 10:
            continue
        image_id = _parse_int(header[0])
        point_line = lines[index] if index < len(lines) else ""
        index += 1
        if image_id is None:
            continue
        tokens = point_line.split()
        points: list[_ImagePoint] = []
        for offset in range(0, len(tokens), 3):
            row = tokens[offset : offset + 3]
            if len(row) != 3:
                points.append(_ImagePoint(None, None, None))
                continue
            points.append(
                _ImagePoint(
                    x=_parse_float(row[0]),
                    y=_parse_float(row[1]),
                    point_id=_parse_int(row[2]),
                )
            )
        if image_id in images:
            duplicate_ids.add(image_id)
            continue
        images[image_id] = _ColmapImage(
            image_id=image_id,
            image_name=header[9],
            points=tuple(points),
        )
    return images, duplicate_ids


def _parse_tracks(model_dir: Path) -> tuple[_RawTrack, ...]:
    tracks: list[_RawTrack] = []
    for row_index, raw in enumerate(
        (model_dir / "points3D.txt").read_text(encoding="utf-8").splitlines()
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        padded = parts + [""] * max(0, 8 - len(parts))
        observations: list[tuple[int | None, int | None]] = []
        for offset in range(8, len(parts), 2):
            pair = parts[offset : offset + 2]
            observations.append(
                (
                    _parse_int(pair[0]) if pair else None,
                    _parse_int(pair[1]) if len(pair) == 2 else None,
                )
            )
        tracks.append(
            _RawTrack(
                row_index=row_index,
                track_id=_parse_int(padded[0]),
                xyz=(
                    _parse_float(padded[1]),
                    _parse_float(padded[2]),
                    _parse_float(padded[3]),
                ),
                error=_parse_float(padded[7]),
                observations=tuple(observations),
            )
        )
    return tuple(tracks)


def _canonical_frames(
    frames: tuple[FrameArtifact, ...],
) -> tuple[dict[str, _CanonicalFrame], set[str]]:
    by_name: dict[str, _CanonicalFrame] = {}
    duplicate_names: set[str] = set()
    seen_frame_ids: set[str] = set()
    for order, frame in enumerate(frames):
        if frame.frame_id in seen_frame_ids:
            raise ValueError("selected frames contain a duplicate canonical frame id")
        seen_frame_ids.add(frame.frame_id)
        with Image.open(frame.path) as opened:
            width, height = opened.size
        if width <= 0 or height <= 0:
            raise ValueError(
                f"selected frame has invalid dimensions: {frame.image_name}"
            )
        canonical = _CanonicalFrame(frame, order, width, height)
        if frame.image_name in by_name:
            duplicate_names.add(frame.image_name)
        else:
            by_name[frame.image_name] = canonical
    return by_name, duplicate_names


def _accepted_track_payload(tracks: tuple[StaticTrack, ...]) -> dict[str, object]:
    return {
        "tracks": [
            {
                "track_id": track.track_id,
                "xyz": list(track.xyz),
                "mean_reprojection_error": track.mean_reprojection_error,
                "observations": [
                    {
                        "frame_id": observation.frame_id,
                        "x": observation.x,
                        "y": observation.y,
                    }
                    for observation in track.observations
                ],
            }
            for track in tracks
        ]
    }


def _freeze_counts(counts: Counter[str]) -> Mapping[str, int]:
    return MappingProxyType(dict(sorted(counts.items())))


def _freeze_examples(
    examples: list[dict[str, object]], maximum: int
) -> tuple[Mapping[str, object], ...]:
    ordered = sorted(
        examples,
        key=lambda item: json.dumps(
            item, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    )[:maximum]
    return tuple(MappingProxyType(item) for item in ordered)


def _report_payload(
    report: TrackAuditReport,
    *,
    policy: TrackQualificationPolicy,
    canonical_frames: dict[str, _CanonicalFrame],
    joined_coordinate_observation_count: int,
    invalid_coordinate_observation_count: int,
) -> dict[str, object]:
    invalid_fraction = (
        invalid_coordinate_observation_count / joined_coordinate_observation_count
        if joined_coordinate_observation_count
        else 0.0
    )
    return {
        "schema_version": report.schema_version,
        "policy": {
            "minimum_observations": policy.minimum_observations,
            "maximum_invalid_fraction": policy.maximum_invalid_fraction,
            "maximum_examples": policy.maximum_examples,
        },
        "raw_track_count": report.raw_track_count,
        "accepted_track_count": report.accepted_track_count,
        "rejected_track_count": report.rejected_track_count,
        "raw_observation_count": report.raw_observation_count,
        "accepted_observation_count": report.accepted_observation_count,
        "rejected_observation_count": report.rejected_observation_count,
        "rejected_tracks_by_reason": dict(report.rejected_tracks_by_reason),
        "rejected_observations_by_reason": dict(report.rejected_observations_by_reason),
        "joined_coordinate_observation_count": joined_coordinate_observation_count,
        "invalid_coordinate_observation_count": invalid_coordinate_observation_count,
        "invalid_coordinate_fraction": invalid_fraction,
        "canonical_frames": [
            {
                "image_name": item.artifact.image_name,
                "frame_id": item.artifact.frame_id,
                "width": item.width,
                "height": item.height,
            }
            for item in sorted(canonical_frames.values(), key=lambda value: value.order)
        ],
        "examples": [dict(item) for item in report.examples],
        "accepted_tracks_sha256": report.accepted_tracks_sha256,
    }


def qualify_colmap_static_tracks(
    model_dir: Path,
    frames: tuple[FrameArtifact, ...],
    audit_path: Path,
    *,
    policy: TrackQualificationPolicy,
) -> QualifiedStaticTracks:
    """Return producer-qualified static tracks and a strict persisted audit."""
    model_dir = model_dir.resolve()
    audit_path = audit_path.resolve()
    canonical_frames, ambiguous_names = _canonical_frames(frames)
    images, duplicate_image_ids = _parse_images(model_dir)
    raw_tracks = _parse_tracks(model_dir)
    track_id_counts = Counter(
        track.track_id
        for track in raw_tracks
        if track.track_id is not None and track.track_id >= 0
    )

    rejected_tracks: Counter[str] = Counter()
    rejected_observations: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    accepted_tracks: list[StaticTrack] = []
    raw_observation_count = 0
    accepted_observation_count = 0
    joined_coordinate_observation_count = 0
    invalid_coordinate_observation_count = 0

    def reject_observation(
        reason: str,
        *,
        track: _RawTrack,
        image: _ColmapImage | None,
        coordinate: _ImagePoint | None,
    ) -> None:
        rejected_observations[reason] += 1
        examples.append(
            {
                "entity": "observation",
                "reason": reason,
                "track_id": track.track_id,
                "image_name": image.image_name if image is not None else None,
                "coordinate": (
                    (
                        _safe_coordinate(coordinate.x),
                        _safe_coordinate(coordinate.y),
                    )
                    if coordinate is not None
                    else None
                ),
            }
        )

    for raw_track in raw_tracks:
        track_reason = None
        if raw_track.track_id is None or raw_track.track_id < 0:
            track_reason = "invalid_track_id"
        elif track_id_counts[raw_track.track_id] != 1:
            track_reason = "duplicate_track_id"
        elif any(value is None or not math.isfinite(value) for value in raw_track.xyz):
            track_reason = "invalid_geometry"
        elif (
            raw_track.error is None
            or not math.isfinite(raw_track.error)
            or raw_track.error < 0.0
        ):
            track_reason = "invalid_reprojection_error"

        observations: list[tuple[int, TrackObservation]] = []
        observed_frame_ids: set[str] = set()
        for image_id, point_index in raw_track.observations:
            raw_observation_count += 1
            image = images.get(image_id) if image_id is not None else None
            coordinate = None
            if (
                image is not None
                and point_index is not None
                and 0 <= point_index < len(image.points)
            ):
                coordinate = image.points[point_index]
            if image_id is None or point_index is None:
                reject_observation(
                    "malformed_observation",
                    track=raw_track,
                    image=image,
                    coordinate=coordinate,
                )
                continue
            if image_id in duplicate_image_ids:
                reject_observation(
                    "ambiguous_image_id",
                    track=raw_track,
                    image=image,
                    coordinate=coordinate,
                )
                continue
            if image is None:
                reject_observation(
                    "missing_image",
                    track=raw_track,
                    image=None,
                    coordinate=None,
                )
                continue
            canonical = canonical_frames.get(image.image_name)
            if image.image_name in ambiguous_names:
                reject_observation(
                    "ambiguous_selected_image",
                    track=raw_track,
                    image=image,
                    coordinate=coordinate,
                )
                continue
            if canonical is None:
                reject_observation(
                    "unselected_image",
                    track=raw_track,
                    image=image,
                    coordinate=coordinate,
                )
                continue
            if coordinate is None:
                reject_observation(
                    "missing_point_reference",
                    track=raw_track,
                    image=image,
                    coordinate=None,
                )
                continue
            if coordinate.point_id != raw_track.track_id:
                reject_observation(
                    "point_id_mismatch",
                    track=raw_track,
                    image=image,
                    coordinate=coordinate,
                )
                continue
            joined_coordinate_observation_count += 1
            if (
                coordinate.x is None
                or coordinate.y is None
                or not math.isfinite(coordinate.x)
                or not math.isfinite(coordinate.y)
            ):
                invalid_coordinate_observation_count += 1
                reject_observation(
                    "non_finite",
                    track=raw_track,
                    image=image,
                    coordinate=coordinate,
                )
                continue
            if not (
                0.0 <= coordinate.x < canonical.width
                and 0.0 <= coordinate.y < canonical.height
            ):
                invalid_coordinate_observation_count += 1
                reject_observation(
                    "out_of_bounds",
                    track=raw_track,
                    image=image,
                    coordinate=coordinate,
                )
                continue
            if canonical.artifact.frame_id in observed_frame_ids:
                reject_observation(
                    "duplicate_frame_observation",
                    track=raw_track,
                    image=image,
                    coordinate=coordinate,
                )
                continue
            observed_frame_ids.add(canonical.artifact.frame_id)
            observations.append(
                (
                    canonical.order,
                    TrackObservation(
                        frame_id=canonical.artifact.frame_id,
                        x=coordinate.x,
                        y=coordinate.y,
                    ),
                )
            )

        if track_reason is None and len(observations) < policy.minimum_observations:
            track_reason = "insufficient_observations"
        if track_reason is not None:
            rejected_tracks[track_reason] += 1
            rejected_observations["parent_track_rejected"] += len(observations)
            examples.append(
                {
                    "entity": "track",
                    "reason": track_reason,
                    "track_id": raw_track.track_id,
                    "row_index": raw_track.row_index,
                }
            )
            continue

        ordered_observations = tuple(
            observation
            for _, observation in sorted(observations, key=lambda row: row[0])
        )
        accepted_observation_count += len(ordered_observations)
        accepted_tracks.append(
            StaticTrack(
                track_id=raw_track.track_id,
                xyz=tuple(float(value) for value in raw_track.xyz),
                mean_reprojection_error=float(raw_track.error),
                observations=ordered_observations,
            )
        )

    tracks = tuple(sorted(accepted_tracks, key=lambda track: track.track_id))
    accepted_payload = _strict_json_bytes(_accepted_track_payload(tracks))
    frozen_examples = _freeze_examples(examples, policy.maximum_examples)
    report = TrackAuditReport(
        schema_version=TRACK_AUDIT_SCHEMA_VERSION,
        raw_track_count=len(raw_tracks),
        accepted_track_count=len(tracks),
        rejected_track_count=len(raw_tracks) - len(tracks),
        raw_observation_count=raw_observation_count,
        accepted_observation_count=accepted_observation_count,
        rejected_observation_count=raw_observation_count - accepted_observation_count,
        rejected_tracks_by_reason=_freeze_counts(rejected_tracks),
        rejected_observations_by_reason=_freeze_counts(rejected_observations),
        examples=frozen_examples,
        accepted_tracks_sha256=_sha256_bytes(accepted_payload),
    )
    audit_payload = _strict_json_bytes(
        _report_payload(
            report,
            policy=policy,
            canonical_frames=canonical_frames,
            joined_coordinate_observation_count=joined_coordinate_observation_count,
            invalid_coordinate_observation_count=invalid_coordinate_observation_count,
        )
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_bytes(audit_payload)
    audit_sha256 = _sha256_bytes(audit_payload)

    invalid_fraction = (
        invalid_coordinate_observation_count / joined_coordinate_observation_count
        if joined_coordinate_observation_count
        else 0.0
    )
    if invalid_fraction > float(policy.maximum_invalid_fraction):
        raise TrackQualificationError(
            "systematic COLMAP coordinate mismatch exceeds the permitted fraction",
            report=report,
            audit_path=audit_path,
        )
    if not tracks:
        raise TrackQualificationError(
            "no qualified COLMAP static tracks remain",
            report=report,
            audit_path=audit_path,
        )
    return QualifiedStaticTracks(
        tracks=tracks,
        report=report,
        audit_path=audit_path,
        audit_sha256=audit_sha256,
    )

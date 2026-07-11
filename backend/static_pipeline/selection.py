from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from bisect import bisect_left
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Literal, Protocol, Sequence

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

from .contracts import (
    FrameRecord,
    SelectionManifest,
    SelectionPolicy,
    SourceInventory,
    UncoveredInterval,
)
from .selection_metrics import (
    CandidateFrame,
    ScoredCandidate,
    choose_smart_candidates,
    pairwise_overlap,
    score_candidates,
)
from .sources import sha256_file


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    avg_fps: float
    rotation_degrees: int
    rotation_source: Literal["display_matrix", "tag", "none"] = "none"


@dataclass(frozen=True)
class TimelineFrame:
    source_index: int
    source_pts: int | None
    timestamp_s: float


class MediaBackend(Protocol):
    def probe_video(self, path: Path) -> VideoProbe: ...

    def video_timeline(
        self, path: Path, fps_limit: int | None
    ) -> tuple[TimelineFrame, ...]: ...

    def extract_video_indices(
        self,
        path: Path,
        indices: tuple[int, ...],
        destination: Path,
        long_edge_cap: int | None,
    ) -> None: ...

    def copy_photo(
        self,
        path: Path,
        destination: Path,
        long_edge_cap: int | None,
    ) -> None: ...


def _parse_fraction(value: object) -> float:
    try:
        fraction = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(fraction) if fraction.denominator else 0.0


def _normalize_rotation(value: object) -> int:
    try:
        degrees = float(value)
    except (TypeError, ValueError):
        return 0
    return int(round(degrees / 90.0) * 90) % 360


def _probe_from_payload(payload: dict[str, object]) -> VideoProbe:
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise RuntimeError("ffprobe did not return a video stream")
    stream = streams[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("ffprobe returned invalid video dimensions") from exc

    rotation_value: object = None
    rotation_source: Literal["display_matrix", "tag", "none"] = "none"
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for entry in side_data:
            if isinstance(entry, dict) and entry.get("rotation") is not None:
                rotation_value = entry["rotation"]
                rotation_source = "display_matrix"
                break
    if rotation_value is None:
        tags = stream.get("tags")
        if isinstance(tags, dict):
            rotation_value = tags.get("rotate")
            if rotation_value is not None:
                rotation_source = "tag"

    return VideoProbe(
        width=width,
        height=height,
        avg_fps=_parse_fraction(stream.get("avg_frame_rate", "0/1")),
        rotation_degrees=_normalize_rotation(rotation_value),
        rotation_source=rotation_source,
    )


class FfmpegMediaBackend:
    def __init__(
        self,
        *,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
    ) -> None:
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary

    def _ffprobe(self, path: Path, *, frames: bool) -> dict[str, object]:
        entries = (
            "frame=best_effort_timestamp,best_effort_timestamp_time:"
            "stream=avg_frame_rate,width,height:stream_tags=rotate:"
            "stream_side_data=rotation"
        )
        command = [
            self.ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "v:0",
        ]
        if frames:
            command.append("-show_frames")
        command.extend(
            [
                "-show_streams",
                "-show_entries",
                entries,
                "-of",
                "json",
                str(path),
            ]
        )
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ffprobe returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("ffprobe returned an invalid JSON root")
        return payload

    def probe_video(self, path: Path) -> VideoProbe:
        return _probe_from_payload(self._ffprobe(Path(path), frames=False))

    def video_timeline(
        self, path: Path, fps_limit: int | None
    ) -> tuple[TimelineFrame, ...]:
        if fps_limit is not None and fps_limit <= 0:
            raise ValueError("fps_limit must be positive when set")
        payload = self._ffprobe(Path(path), frames=True)
        probe = _probe_from_payload(payload)
        raw_frames = payload.get("frames")
        if not isinstance(raw_frames, list):
            raise RuntimeError("ffprobe did not return video frames")

        timeline: list[TimelineFrame] = []
        for source_index, raw in enumerate(raw_frames):
            if not isinstance(raw, dict):
                continue
            try:
                timestamp = float(raw["best_effort_timestamp_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(timestamp):
                continue
            try:
                source_pts = int(raw["best_effort_timestamp"])
            except (KeyError, TypeError, ValueError):
                source_pts = None
            timeline.append(
                TimelineFrame(
                    source_index=source_index,
                    source_pts=source_pts,
                    timestamp_s=timestamp,
                )
            )
        timeline.sort(key=lambda frame: (frame.timestamp_s, frame.source_index))
        if not timeline:
            raise RuntimeError("ffprobe returned no timestamped video frames")
        if fps_limit is None or (0 < probe.avg_fps < fps_limit):
            return tuple(timeline)

        targets = _target_timestamps(
            timeline[0].timestamp_s,
            timeline[-1].timestamp_s,
            fps_limit,
        )
        return _nearest_monotonic_frames(timeline, targets)

    def extract_video_indices(
        self,
        path: Path,
        indices: tuple[int, ...],
        destination: Path,
        long_edge_cap: int | None,
    ) -> None:
        ordered_indices = tuple(sorted(set(indices)))
        if not ordered_indices or ordered_indices[0] < 0:
            raise ValueError("indices must contain non-negative frame indices")
        if long_edge_cap is not None and long_edge_cap <= 0:
            raise ValueError("long_edge_cap must be positive when set")

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        probe = self.probe_video(Path(path))
        select_expression = "+".join(f"eq(n,{index})" for index in ordered_indices)
        filters = [f"select='{select_expression}'"]
        if probe.rotation_degrees == 90:
            filters.append(
                "transpose=cclock"
                if probe.rotation_source == "display_matrix"
                else "transpose=clock"
            )
        elif probe.rotation_degrees == 180:
            filters.append("rotate=PI:ow=iw:oh=ih")
        elif probe.rotation_degrees == 270:
            filters.append(
                "transpose=clock"
                if probe.rotation_source == "display_matrix"
                else "transpose=cclock"
            )

        display_width, display_height = probe.width, probe.height
        if probe.rotation_degrees in {90, 270}:
            display_width, display_height = display_height, display_width
        if (
            long_edge_cap is not None
            and max(display_width, display_height) > long_edge_cap
        ):
            if display_width >= display_height:
                filters.append(f"scale={long_edge_cap}:-2")
            else:
                filters.append(f"scale=-2:{long_edge_cap}")

        script_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".ffilter",
                prefix="selection-",
                dir=destination.parent,
                delete=False,
            ) as script:
                script.write(",".join(filters))
                script_path = Path(script.name)
            output_pattern = destination / "frame_%06d.png"
            subprocess.run(
                [
                    self.ffmpeg_binary,
                    "-y",
                    "-loglevel",
                    "error",
                    "-noautorotate",
                    "-i",
                    str(path),
                    "-filter_script:v",
                    str(script_path),
                    "-vsync",
                    "0",
                    "-start_number",
                    "0",
                    str(output_pattern),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            if script_path is not None:
                script_path.unlink(missing_ok=True)

    def copy_photo(
        self,
        path: Path,
        destination: Path,
        long_edge_cap: int | None,
    ) -> None:
        if long_edge_cap is not None and long_edge_cap <= 0:
            raise ValueError("long_edge_cap must be positive when set")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            if long_edge_cap is not None and max(image.size) > long_edge_cap:
                image.thumbnail(
                    (long_edge_cap, long_edge_cap),
                    Image.Resampling.LANCZOS,
                )
            image.save(destination, format="PNG")


def make_frame_id(
    source_digest: str,
    source_index: int | None,
    timestamp_s: float | None,
    relative_path: str,
) -> str:
    identity = f"{source_digest}|{source_index}|{timestamp_s}|{relative_path}".encode(
        "utf-8"
    )
    return hashlib.sha256(identity).hexdigest()[:24]


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    pieces: list[tuple[int, object]] = []
    for piece in re.split(r"(\d+)", value):
        if piece.isdigit():
            pieces.append((0, int(piece)))
        else:
            pieces.append((1, piece.casefold()))
    pieces.append((2, value))
    return tuple(pieces)


def _uniform_indices(count: int, limit: int) -> tuple[int, ...]:
    if count < 0 or limit <= 0:
        raise ValueError("count must be non-negative and limit must be positive")
    if count <= limit:
        return tuple(range(count))
    if limit == 1:
        return (0,)
    return tuple(position * (count - 1) // (limit - 1) for position in range(limit))


def _target_timestamps(start: float, end: float, fps: int) -> tuple[float, ...]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    target_count = max(1, math.floor((end - start) * fps + 1e-9) + 1)
    return tuple(start + target_index / fps for target_index in range(target_count))


def _nearest_monotonic_frames(
    timeline: Sequence[TimelineFrame],
    targets: Sequence[float],
) -> tuple[TimelineFrame, ...]:
    ordered = sorted(
        timeline,
        key=lambda frame: (frame.timestamp_s, frame.source_index),
    )
    timestamps = [frame.timestamp_s for frame in ordered]
    cursor = 0
    selected: list[TimelineFrame] = []
    for target in targets:
        if cursor >= len(ordered):
            break
        right = bisect_left(timestamps, target, lo=cursor)
        candidate_indices: list[int] = []
        if right < len(ordered):
            candidate_indices.append(right)
        if right - 1 >= cursor:
            candidate_indices.append(right - 1)
        winner_index = min(
            candidate_indices,
            key=lambda index: (
                round(abs(ordered[index].timestamp_s - target), 12),
                ordered[index].timestamp_s,
                ordered[index].source_index,
            ),
        )
        selected.append(ordered[winner_index])
        cursor = winner_index + 1
    return tuple(selected)


def _validate_policy(policy: SelectionPolicy) -> None:
    if policy.mode not in ("smart", "fixed_fps"):
        raise ValueError(f"unsupported selection mode: {policy.mode}")
    if policy.frame_budget <= 0:
        raise ValueError("frame_budget must be positive")
    if policy.mode == "fixed_fps":
        if policy.fixed_fps <= 0:
            raise ValueError("fixed_fps must be positive")
    else:
        if policy.candidate_fps <= 0:
            raise ValueError("candidate_fps must be positive")
        if policy.candidate_long_edge <= 0:
            raise ValueError("candidate_long_edge must be positive")
        if policy.candidate_fps > 12:
            raise ValueError("candidate_fps cannot exceed 12")
        if policy.candidate_long_edge > 320:
            raise ValueError("candidate_long_edge cannot exceed 320")
    if (
        policy.resolution_long_edge_cap is not None
        and policy.resolution_long_edge_cap <= 0
    ):
        raise ValueError("resolution_long_edge_cap must be positive when set")


def _choose_fixed_frames(
    timeline: Sequence[TimelineFrame],
    *,
    fixed_fps: int,
    frame_budget: int,
) -> tuple[TimelineFrame, ...]:
    if not timeline:
        raise ValueError("video timeline is empty")
    ordered = sorted(
        timeline,
        key=lambda frame: (frame.timestamp_s, frame.source_index),
    )
    start = ordered[0].timestamp_s
    end = ordered[-1].timestamp_s
    selected = _nearest_monotonic_frames(
        ordered,
        _target_timestamps(start, end, fixed_fps),
    )
    bounded = _uniform_indices(len(selected), frame_budget)
    return tuple(selected[index] for index in bounded)


def _image_set_digest(records: Sequence[FrameRecord]) -> str:
    payload = [
        [record.frame_id, record.output_name, record.sha256]
        for record in records
        if record.selected
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_payload(manifest: SelectionManifest) -> dict[str, object]:
    return asdict(manifest)


def write_selection_manifest(path: Path, manifest: SelectionManifest) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _manifest_payload(manifest),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def _publish_directory(staged: Path, target: Path) -> None:
    backup = target.with_name(f"{target.name}.backup-{uuid.uuid4().hex}")
    had_target = os.path.lexists(target)
    if had_target:
        os.rename(target, backup)
    try:
        os.rename(staged, target)
    except BaseException:
        if os.path.lexists(target):
            _remove_path(target)
        if had_target and os.path.lexists(backup):
            os.rename(backup, target)
        raise
    else:
        if had_target and os.path.lexists(backup):
            try:
                _remove_path(backup)
            except BaseException:
                try:
                    _remove_path(target)
                    os.rename(backup, target)
                except BaseException as rollback_error:
                    raise RuntimeError(
                        "selection was published but backup cleanup and rollback failed"
                    ) from rollback_error
                raise


def _fresh_stage(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    stage.mkdir()
    return stage


def _ensure_output_outside_source(target: Path, source_root: Path) -> None:
    resolved_target = target.resolve(strict=False)
    resolved_source = source_root.resolve(strict=True)
    try:
        resolved_target.relative_to(resolved_source)
    except ValueError:
        pass
    else:
        raise ValueError("selection output must not overlap the immutable source input")
    try:
        resolved_source.relative_to(resolved_target)
    except ValueError:
        return
    raise ValueError("selection output must not overlap the immutable source input")


def _verify_png_set(directory: Path, count: int) -> None:
    expected = [f"frame_{index:06d}.png" for index in range(count)]
    entries = sorted(directory.iterdir(), key=lambda path: path.name)
    actual = [path.name for path in entries]
    if actual != expected or any(not path.is_file() for path in entries):
        raise RuntimeError(
            f"media backend produced unexpected staged entries: {actual}"
        )


def _extract_video_frames_ordered(
    backend: MediaBackend,
    source_path: Path,
    source_indices: Sequence[int],
    destination: Path,
    long_edge_cap: int | None,
) -> None:
    requested = tuple(source_indices)
    if len(set(requested)) != len(requested):
        raise ValueError("video source indices must be unique")
    sorted_indices = tuple(sorted(requested))
    raw = destination / f".video-raw-{uuid.uuid4().hex}"
    try:
        backend.extract_video_indices(
            source_path,
            sorted_indices,
            raw,
            long_edge_cap,
        )
        _verify_png_set(raw, len(sorted_indices))
        source_position = {
            source_index: position
            for position, source_index in enumerate(sorted_indices)
        }
        for output_index, source_index in enumerate(requested):
            shutil.move(
                raw / f"frame_{source_position[source_index]:06d}.png",
                destination / f"frame_{output_index:06d}.png",
            )
    finally:
        if os.path.lexists(raw):
            _remove_path(raw)


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to decode candidate image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _smart_photo_order(inventory: SourceInventory) -> tuple[str, ...]:
    natural = sorted(
        (item.relative_path for item in inventory.media_files),
        key=_natural_key,
    )
    dated: list[tuple[datetime, str]] = []
    for relative_path in natural:
        try:
            with Image.open(inventory.root / relative_path) as image:
                exif = image.getexif()
                raw_value = exif.get(36867)
                if raw_value is None:
                    raw_value = exif.get_ifd(ExifTags.IFD.Exif).get(36867)
            if raw_value is None:
                return tuple(natural)
            text_value = (
                raw_value.decode("utf-8", errors="strict")
                if isinstance(raw_value, bytes)
                else str(raw_value)
            )
            parsed = datetime.strptime(text_value, "%Y:%m:%d %H:%M:%S")
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            return tuple(natural)
        dated.append((parsed, relative_path))
    dated.sort(key=lambda item: (item[0], _natural_key(item[1])))
    return tuple(relative_path for _, relative_path in dated)


def _smart_key(candidate: ScoredCandidate) -> tuple[float, int, str]:
    coordinate = (
        candidate.timestamp_s
        if candidate.timestamp_s is not None
        else float(candidate.ordinal)
    )
    return (coordinate, candidate.ordinal, candidate.frame_id)


def _analyze_source_candidates(
    inventory: SourceInventory,
    policy: SelectionPolicy,
    backend: MediaBackend,
) -> tuple[ScoredCandidate, ...]:
    with tempfile.TemporaryDirectory(
        prefix="static-selection-candidates-"
    ) as temporary:
        analysis_dir = Path(temporary)
        candidates: list[CandidateFrame] = []
        if inventory.kind == "video":
            source_file = inventory.media_files[0]
            source_path = inventory.root / source_file.relative_path
            timeline = backend.video_timeline(source_path, policy.candidate_fps)
            _extract_video_frames_ordered(
                backend,
                source_path,
                [frame.source_index for frame in timeline],
                analysis_dir,
                policy.candidate_long_edge,
            )
            _verify_png_set(analysis_dir, len(timeline))
            for ordinal, frame in enumerate(timeline):
                candidates.append(
                    CandidateFrame(
                        frame_id=make_frame_id(
                            inventory.digest,
                            frame.source_index,
                            frame.timestamp_s,
                            source_file.relative_path,
                        ),
                        source_relative_path=source_file.relative_path,
                        source_index=frame.source_index,
                        source_pts=frame.source_pts,
                        timestamp_s=frame.timestamp_s,
                        ordinal=ordinal,
                        rgb=_read_rgb(analysis_dir / f"frame_{ordinal:06d}.png"),
                    )
                )
        else:
            relative_paths = _smart_photo_order(inventory)
            for ordinal, relative_path in enumerate(relative_paths):
                output = analysis_dir / f"frame_{ordinal:06d}.png"
                backend.copy_photo(
                    inventory.root / relative_path,
                    output,
                    policy.candidate_long_edge,
                )
                candidates.append(
                    CandidateFrame(
                        frame_id=make_frame_id(
                            inventory.digest,
                            None,
                            None,
                            relative_path,
                        ),
                        source_relative_path=relative_path,
                        source_index=None,
                        source_pts=None,
                        timestamp_s=None,
                        ordinal=ordinal,
                        rgb=_read_rgb(output),
                    )
                )
            _verify_png_set(analysis_dir, len(relative_paths))

        scored = score_candidates(candidates)
        empty_rgb = np.empty((0, 0, 3), dtype=np.uint8)
        return tuple(
            replace(item, frame=replace(item.frame, rgb=empty_rgb)) for item in scored
        )


def _materialize_scored_candidates(
    inventory: SourceInventory,
    selected: Sequence[ScoredCandidate],
    stage: Path,
    backend: MediaBackend,
    long_edge_cap: int | None,
) -> tuple[ScoredCandidate, ...]:
    ordered = tuple(sorted(selected, key=_smart_key))
    if inventory.kind == "video":
        source_path = inventory.root / inventory.media_files[0].relative_path
        indices = [candidate.source_index for candidate in ordered]
        if any(index is None for index in indices):
            raise ValueError("video candidates require source indices")
        _extract_video_frames_ordered(
            backend,
            source_path,
            [int(index) for index in indices],
            stage,
            long_edge_cap,
        )
    else:
        for output_index, candidate in enumerate(ordered):
            backend.copy_photo(
                inventory.root / candidate.source_relative_path,
                stage / f"frame_{output_index:06d}.png",
                long_edge_cap,
            )
    _verify_png_set(stage, len(ordered))
    return ordered


def _smart_records(
    all_candidates: Sequence[ScoredCandidate],
    selected: Sequence[ScoredCandidate],
    stage: Path,
) -> tuple[FrameRecord, ...]:
    ordered_all = sorted(all_candidates, key=_smart_key)
    selected_by_id = {
        candidate.frame_id: (output_index, candidate)
        for output_index, candidate in enumerate(selected)
    }
    records: list[FrameRecord] = []
    for candidate in ordered_all:
        selected_value = selected_by_id.get(candidate.frame_id)
        is_selected = selected_value is not None
        output_name = (
            f"frame_{selected_value[0]:06d}.png" if selected_value is not None else ""
        )
        selected_candidate = (
            selected_value[1] if selected_value is not None else candidate
        )
        reasons = selected_candidate.reasons
        if not is_selected:
            reasons = _append_reason(reasons, "not_selected_by_smart_policy")
        records.append(
            FrameRecord(
                frame_id=candidate.frame_id,
                source_relative_path=candidate.source_relative_path,
                source_index=candidate.source_index,
                source_pts=candidate.source_pts,
                timestamp_s=candidate.timestamp_s,
                output_name=output_name,
                sha256=sha256_file(stage / output_name) if is_selected else "",
                selected=is_selected,
                metrics=candidate.metrics,
                selection_score=candidate.total_score,
                reasons=reasons,
            )
        )
    return tuple(records)


def _video_records(
    inventory: SourceInventory,
    relative_path: str,
    selected: Sequence[TimelineFrame],
    stage: Path,
) -> tuple[FrameRecord, ...]:
    records: list[FrameRecord] = []
    for output_index, timeline_frame in enumerate(selected):
        output_name = f"frame_{output_index:06d}.png"
        records.append(
            FrameRecord(
                frame_id=make_frame_id(
                    inventory.digest,
                    timeline_frame.source_index,
                    timeline_frame.timestamp_s,
                    relative_path,
                ),
                source_relative_path=relative_path,
                source_index=timeline_frame.source_index,
                source_pts=timeline_frame.source_pts,
                timestamp_s=timeline_frame.timestamp_s,
                output_name=output_name,
                sha256=sha256_file(stage / output_name),
                selected=True,
                metrics=None,
                selection_score=None,
                reasons=("fixed_fps",),
            )
        )
    return tuple(records)


def _photo_records(
    inventory: SourceInventory,
    relative_paths: Sequence[str],
    stage: Path,
) -> tuple[FrameRecord, ...]:
    records: list[FrameRecord] = []
    for output_index, relative_path in enumerate(relative_paths):
        output_name = f"frame_{output_index:06d}.png"
        records.append(
            FrameRecord(
                frame_id=make_frame_id(
                    inventory.digest,
                    None,
                    None,
                    relative_path,
                ),
                source_relative_path=relative_path,
                source_index=None,
                source_pts=None,
                timestamp_s=None,
                output_name=output_name,
                sha256=sha256_file(stage / output_name),
                selected=True,
                metrics=None,
                selection_score=None,
                reasons=("photo_set_all",),
            )
        )
    return tuple(records)


def select_frames(
    inventory: SourceInventory,
    output_dir: Path,
    policy: SelectionPolicy,
    media: MediaBackend | None = None,
) -> SelectionManifest:
    _validate_policy(policy)
    if inventory.schema_version != 1:
        raise ValueError("unsupported source inventory schema")

    backend: MediaBackend = media or FfmpegMediaBackend()
    target = Path(output_dir)
    _ensure_output_outside_source(target, inventory.root)
    stage = _fresh_stage(target)
    try:
        if policy.mode == "smart":
            analyzed = _analyze_source_candidates(inventory, policy, backend)
            chosen = choose_smart_candidates(analyzed, policy.frame_budget)
            selected_scored = _materialize_scored_candidates(
                inventory,
                chosen,
                stage,
                backend,
                policy.resolution_long_edge_cap,
            )
            records = _smart_records(analyzed, selected_scored, stage)
            effective_mode = "smart"
        elif inventory.kind == "video":
            source_file = inventory.media_files[0]
            source_path = inventory.root / source_file.relative_path
            timeline = backend.video_timeline(source_path, None)
            selected = _choose_fixed_frames(
                timeline,
                fixed_fps=policy.fixed_fps,
                frame_budget=policy.frame_budget,
            )
            _extract_video_frames_ordered(
                backend,
                source_path,
                [frame.source_index for frame in selected],
                stage,
                policy.resolution_long_edge_cap,
            )
            _verify_png_set(stage, len(selected))
            records = _video_records(
                inventory,
                source_file.relative_path,
                selected,
                stage,
            )
            effective_mode = "fixed_fps"
        else:
            media_files = sorted(
                inventory.media_files,
                key=lambda item: _natural_key(item.relative_path),
            )
            relative_paths = tuple(item.relative_path for item in media_files)
            for output_index, relative_path in enumerate(relative_paths):
                backend.copy_photo(
                    inventory.root / relative_path,
                    stage / f"frame_{output_index:06d}.png",
                    policy.resolution_long_edge_cap,
                )
            _verify_png_set(stage, len(relative_paths))
            records = _photo_records(inventory, relative_paths, stage)
            effective_mode = "photo_set_all"

        manifest = SelectionManifest(
            schema_version=1,
            source_digest=inventory.digest,
            effective_mode=effective_mode,
            policy=policy,
            frames=records,
            image_set_digest=_image_set_digest(records),
        )
        write_selection_manifest(stage / "selection_manifest.json", manifest)
        _publish_directory(stage, target)
        return manifest
    finally:
        if os.path.lexists(stage):
            _remove_path(stage)


def _append_reason(reasons: tuple[str, ...], reason: str) -> tuple[str, ...]:
    return reasons if reason in reasons else reasons + (reason,)


def _candidate_in_interval(
    candidate: ScoredCandidate,
    interval: UncoveredInterval,
) -> bool:
    if candidate.frame_id in interval.missing_frame_ids:
        return True
    return (
        candidate.timestamp_s is not None
        and interval.start_s <= candidate.timestamp_s <= interval.end_s
    )


def _candidate_in_any_interval(
    candidate: ScoredCandidate,
    intervals: Sequence[UncoveredInterval],
) -> bool:
    return any(_candidate_in_interval(candidate, interval) for interval in intervals)


def _canonicalize_intervals(
    uncovered: Sequence[UncoveredInterval],
) -> tuple[UncoveredInterval, ...]:
    ordered = sorted(
        uncovered,
        key=lambda interval: (
            interval.start_s,
            interval.end_s,
            interval.missing_frame_ids,
        ),
    )
    merged: list[UncoveredInterval] = []
    for interval in ordered:
        if (
            not math.isfinite(interval.start_s)
            or not math.isfinite(interval.end_s)
            or interval.end_s < interval.start_s
        ):
            raise ValueError("uncovered intervals must be finite and ordered")
        if merged and interval.start_s <= merged[-1].end_s:
            previous = merged[-1]
            merged[-1] = UncoveredInterval(
                start_s=previous.start_s,
                end_s=max(previous.end_s, interval.end_s),
                missing_frame_ids=tuple(
                    sorted(
                        set(previous.missing_frame_ids)
                        | set(interval.missing_frame_ids)
                    )
                ),
            )
        else:
            merged.append(
                UncoveredInterval(
                    start_s=interval.start_s,
                    end_s=interval.end_s,
                    missing_frame_ids=tuple(sorted(set(interval.missing_frame_ids))),
                )
            )
    return tuple(merged)


def plan_backfill(
    inventory: SourceInventory,
    manifest: SelectionManifest,
    uncovered: Sequence[UncoveredInterval],
    output_dir: Path,
    media: MediaBackend | None = None,
    max_per_interval: int = 2,
) -> SelectionManifest:
    if manifest.policy.mode != "smart" or manifest.effective_mode != "smart":
        raise ValueError("Smart selection is required for gap backfill")
    if manifest.source_digest != inventory.digest:
        raise ValueError("selection manifest does not belong to this source inventory")
    _validate_policy(manifest.policy)
    if any(
        "colmap_gap_backfill_attempted" in frame.reasons for frame in manifest.frames
    ):
        raise ValueError("Smart gap backfill has already been attempted")
    if max_per_interval <= 0:
        raise ValueError("max_per_interval must be positive")
    intervals = _canonicalize_intervals(uncovered)
    if not intervals:
        return manifest

    target = Path(output_dir)
    _ensure_output_outside_source(target, inventory.root)
    backend: MediaBackend = media or FfmpegMediaBackend()
    analyzed = _analyze_source_candidates(inventory, manifest.policy, backend)
    scored_by_id = {candidate.frame_id: candidate for candidate in analyzed}
    records_by_id = {frame.frame_id: frame for frame in manifest.frames}
    if set(records_by_id) != set(scored_by_id):
        raise ValueError("Smart candidate inventory changed before backfill")

    selected_ids = {frame.frame_id for frame in manifest.selected_frames}
    per_interval_limit = min(max_per_interval, 2)
    planned: list[ScoredCandidate] = []
    planned_ids: set[str] = set()
    for interval in intervals:
        midpoint = (interval.start_s + interval.end_s) / 2.0
        pool = [
            candidate
            for candidate in analyzed
            if candidate.frame_id not in selected_ids
            and candidate.frame_id not in planned_ids
            and _candidate_in_interval(candidate, interval)
        ]
        pool.sort(
            key=lambda candidate: (
                candidate.metrics.overlap_score,
                candidate.total_score,
                -abs(
                    (
                        candidate.timestamp_s
                        if candidate.timestamp_s is not None
                        else midpoint
                    )
                    - midpoint
                ),
                -candidate.ordinal,
                candidate.frame_id,
            ),
            reverse=True,
        )
        for candidate in pool[:per_interval_limit]:
            planned.append(candidate)
            planned_ids.add(candidate.frame_id)

    ordered_original = sorted(
        (scored_by_id[frame_id] for frame_id in selected_ids),
        key=_smart_key,
    )
    endpoint_ids = (
        {ordered_original[0].frame_id, ordered_original[-1].frame_id}
        if ordered_original
        else set()
    )
    accepted_additions: set[str] = set()
    removed_ids: set[str] = set()
    for addition in planned:
        tentative_ids = selected_ids | {addition.frame_id}
        tentative = sorted(
            (scored_by_id[frame_id] for frame_id in tentative_ids),
            key=_smart_key,
        )
        removable: list[ScoredCandidate] = []
        for index in range(1, len(tentative) - 1):
            candidate = tentative[index]
            if (
                candidate.frame_id in endpoint_ids
                or candidate.frame_id in accepted_additions
                or candidate.frame_id == addition.frame_id
                or _candidate_in_any_interval(candidate, intervals)
            ):
                continue
            if pairwise_overlap(tentative[index - 1], tentative[index + 1]) >= 0.20:
                removable.append(candidate)
        if not removable:
            continue
        victim = min(
            removable,
            key=lambda candidate: (
                candidate.total_score,
                candidate.ordinal,
                candidate.frame_id,
            ),
        )
        selected_ids.remove(victim.frame_id)
        removed_ids.add(victim.frame_id)
        selected_ids.add(addition.frame_id)
        accepted_additions.add(addition.frame_id)

    stage = _fresh_stage(target)
    try:
        selected_scored = _materialize_scored_candidates(
            inventory,
            [scored_by_id[frame_id] for frame_id in selected_ids],
            stage,
            backend,
            manifest.policy.resolution_long_edge_cap,
        )
        output_by_id = {
            candidate.frame_id: output_index
            for output_index, candidate in enumerate(selected_scored)
        }
        updated_records: list[FrameRecord] = []
        for record in manifest.frames:
            is_selected = record.frame_id in selected_ids
            output_index = output_by_id.get(record.frame_id)
            reasons = record.reasons
            if record.frame_id in accepted_additions:
                reasons = _append_reason(reasons, "colmap_gap_backfill")
            if record.frame_id in removed_ids:
                reasons = _append_reason(reasons, "colmap_gap_replaced")
            reasons = _append_reason(reasons, "colmap_gap_backfill_attempted")
            output_name = (
                f"frame_{output_index:06d}.png" if output_index is not None else ""
            )
            updated_records.append(
                replace(
                    record,
                    output_name=output_name,
                    sha256=sha256_file(stage / output_name) if is_selected else "",
                    selected=is_selected,
                    reasons=reasons,
                )
            )
        records = tuple(updated_records)
        backfilled = SelectionManifest(
            schema_version=1,
            source_digest=inventory.digest,
            effective_mode="smart",
            policy=manifest.policy,
            frames=records,
            image_set_digest=_image_set_digest(records),
            reconstruction_guardrail=manifest.reconstruction_guardrail,
        )
        write_selection_manifest(stage / "selection_manifest.json", backfilled)
        _publish_directory(stage, target)
        return backfilled
    finally:
        if os.path.lexists(stage):
            _remove_path(stage)


def bound_selection_for_reconstruction(
    inventory: SourceInventory,
    manifest: SelectionManifest,
    output_dir: Path,
    media: MediaBackend | None = None,
    max_frames: int = 800,
) -> SelectionManifest:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    _validate_policy(manifest.policy)
    if manifest.source_digest != inventory.digest:
        raise ValueError("selection manifest does not belong to this source inventory")

    cap = min(manifest.policy.frame_budget, max_frames, 800)
    selected = manifest.selected_frames
    if len(selected) <= cap:
        return manifest
    if inventory.kind != "photo_set" or manifest.effective_mode != "photo_set_all":
        raise ValueError("reconstruction guardrail currently applies to photo sets")

    backend: MediaBackend = media or FfmpegMediaBackend()
    selected_positions = _uniform_indices(len(selected), cap)
    selected_ids = {selected[position].frame_id for position in selected_positions}
    selected_order = [selected[position] for position in selected_positions]
    target = Path(output_dir)
    _ensure_output_outside_source(target, inventory.root)
    stage = _fresh_stage(target)
    try:
        updated_selected: dict[str, FrameRecord] = {}
        for output_index, record in enumerate(selected_order):
            output_name = f"frame_{output_index:06d}.png"
            backend.copy_photo(
                inventory.root / record.source_relative_path,
                stage / output_name,
                manifest.policy.resolution_long_edge_cap,
            )
            updated_selected[record.frame_id] = replace(
                record,
                output_name=output_name,
                sha256=sha256_file(stage / output_name),
                selected=True,
                reasons=_append_reason(record.reasons, "uniform_profile_cap"),
            )
        _verify_png_set(stage, cap)

        updated_frames = tuple(
            updated_selected[record.frame_id]
            if record.frame_id in selected_ids
            else replace(
                record,
                output_name="",
                sha256="",
                selected=False,
                reasons=_append_reason(record.reasons, "colmap_budget_guardrail"),
            )
            for record in manifest.frames
        )
        bounded = SelectionManifest(
            schema_version=1,
            source_digest=inventory.digest,
            effective_mode=manifest.effective_mode,
            policy=manifest.policy,
            frames=updated_frames,
            image_set_digest=_image_set_digest(updated_frames),
            reconstruction_guardrail="uniform_profile_cap",
        )
        write_selection_manifest(stage / "selection_manifest.json", bounded)
        _publish_directory(stage, target)
        return bounded
    finally:
        if os.path.lexists(stage):
            _remove_path(stage)

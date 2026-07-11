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
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Protocol, Sequence

from PIL import Image, ImageOps

from .contracts import (
    FrameRecord,
    SelectionManifest,
    SelectionPolicy,
    SourceInventory,
)
from .sources import sha256_file


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    avg_fps: float
    rotation_degrees: int


@dataclass(frozen=True)
class TimelineFrame:
    source_index: int
    source_pts: int | None
    timestamp_s: float


class MediaBackend(Protocol):
    def probe_video(self, path: Path) -> VideoProbe: ...

    def video_timeline(
        self, path: Path, fps_limit: int
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
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for entry in side_data:
            if isinstance(entry, dict) and entry.get("rotation") is not None:
                rotation_value = entry["rotation"]
                break
    if rotation_value is None:
        tags = stream.get("tags")
        if isinstance(tags, dict):
            rotation_value = tags.get("rotate")

    return VideoProbe(
        width=width,
        height=height,
        avg_fps=_parse_fraction(stream.get("avg_frame_rate", "0/1")),
        rotation_degrees=_normalize_rotation(rotation_value),
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

    def video_timeline(self, path: Path, fps_limit: int) -> tuple[TimelineFrame, ...]:
        if fps_limit <= 0:
            raise ValueError("fps_limit must be positive")
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
        if 0 < probe.avg_fps < fps_limit:
            return tuple(timeline)

        selected: list[TimelineFrame] = []
        next_timestamp = timeline[0].timestamp_s
        step = 1.0 / fps_limit
        for frame in timeline:
            if not selected or frame.timestamp_s + 1e-12 >= next_timestamp:
                selected.append(frame)
                next_timestamp = frame.timestamp_s + step
        return tuple(selected)

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
            filters.append("transpose=clock")
        elif probe.rotation_degrees == 180:
            filters.append("rotate=PI:ow=iw:oh=ih")
        elif probe.rotation_degrees == 270:
            filters.append("transpose=cclock")

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


def _validate_policy(policy: SelectionPolicy) -> None:
    if policy.frame_budget <= 0:
        raise ValueError("frame_budget must be positive")
    if policy.fixed_fps <= 0 or policy.candidate_fps <= 0:
        raise ValueError("FPS values must be positive")
    if policy.candidate_long_edge <= 0:
        raise ValueError("candidate_long_edge must be positive")
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
    ordered = sorted(
        timeline, key=lambda frame: (frame.timestamp_s, frame.source_index)
    )
    if not ordered:
        raise ValueError("video timeline is empty")
    start = ordered[0].timestamp_s
    end = ordered[-1].timestamp_s
    target_count = max(1, math.floor((end - start) * fixed_fps + 1e-9) + 1)
    available = list(ordered)
    selected: list[TimelineFrame] = []
    for target_index in range(target_count):
        if not available:
            break
        target = start + target_index / fixed_fps
        winner = min(
            available,
            key=lambda frame: (
                round(abs(frame.timestamp_s - target), 12),
                frame.timestamp_s,
                frame.source_index,
            ),
        )
        selected.append(winner)
        available.remove(winner)
    selected.sort(key=lambda frame: (frame.timestamp_s, frame.source_index))
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
            _remove_path(backup)


def _fresh_stage(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    stage.mkdir()
    return stage


def _ensure_output_outside_source(target: Path, source_root: Path) -> None:
    resolved_target = target.parent.resolve(strict=False) / target.name
    try:
        resolved_target.relative_to(source_root.resolve(strict=True))
    except ValueError:
        return
    raise ValueError("selection output must not be inside the immutable source input")


def _verify_png_set(directory: Path, count: int) -> None:
    expected = [f"frame_{index:06d}.png" for index in range(count)]
    actual = sorted(path.name for path in directory.glob("*.png") if path.is_file())
    if actual != expected:
        raise RuntimeError(f"media backend produced unexpected PNG set: {actual}")


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
    if policy.mode != "fixed_fps":
        raise ValueError("Smart selection is implemented by the next pipeline stage")
    if inventory.schema_version != 1:
        raise ValueError("unsupported source inventory schema")

    backend: MediaBackend = media or FfmpegMediaBackend()
    target = Path(output_dir)
    _ensure_output_outside_source(target, inventory.root)
    stage = _fresh_stage(target)
    try:
        if inventory.kind == "video":
            source_file = inventory.media_files[0]
            source_path = inventory.root / source_file.relative_path
            timeline = backend.video_timeline(source_path, policy.candidate_fps)
            selected = _choose_fixed_frames(
                timeline,
                fixed_fps=policy.fixed_fps,
                frame_budget=policy.frame_budget,
            )
            backend.extract_video_indices(
                source_path,
                tuple(frame.source_index for frame in selected),
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

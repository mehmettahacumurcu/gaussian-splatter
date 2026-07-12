from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from PIL import Image, ImageOps

from backend.static_pipeline.contracts import SourceInventory
from backend.static_pipeline.selection import TimelineFrame, VideoProbe
from backend.static_pipeline.sources import discover_source


def _write_image(
    path: Path, *, size: tuple[int, int] = (8, 6), value: int = 64
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value, value)).save(path)


def write_colmap_text_model(
    root: Path,
    registered_names: tuple[str, ...] | list[str],
    point_errors: tuple[float, ...] | list[float],
    track_lengths: tuple[int, ...] | list[int],
    invalid_pose: bool = False,
) -> Path:
    """Write a deterministic exact-directory COLMAP text model for gate tests."""

    if len(point_errors) != len(track_lengths):
        raise ValueError("point_errors and track_lengths must have equal length")
    model = Path(root)
    model.mkdir(parents=True, exist_ok=False)
    (model / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 1920 1080 1000 1000 960 540\n",
        encoding="utf-8",
    )

    image_lines = [
        "# Image list",
        (
            f"# Number of images: {len(registered_names)}, "
            "mean observations per image: 0"
        ),
    ]
    for image_id, name in enumerate(registered_names, start=1):
        translation = "nan 0 0" if invalid_pose and image_id == 1 else "0 0 0"
        image_lines.extend(
            (
                f"{image_id} 1 0 0 0 {translation} 1 {name}",
                "",
            )
        )
    (model / "images.txt").write_text(
        "\n".join(image_lines) + "\n",
        encoding="utf-8",
    )

    image_count = max(1, len(registered_names))
    point_lines = ["# 3D point list"]
    for point_id, (error, track_length) in enumerate(
        zip(point_errors, track_lengths, strict=True),
        start=1,
    ):
        if track_length < 0:
            raise ValueError("track lengths must be non-negative")
        track = " ".join(
            f"{1 + (track_index % image_count)} {track_index}"
            for track_index in range(track_length)
        )
        suffix = f" {track}" if track else ""
        point_lines.append(
            f"{point_id} {point_id * 0.1:.6f} 0 1 128 128 128 {error}{suffix}"
        )
    (model / "points3D.txt").write_text(
        "\n".join(point_lines) + "\n",
        encoding="utf-8",
    )
    return model


@dataclass
class FakeMediaBackend:
    timeline: tuple[TimelineFrame, ...]
    probe: VideoProbe = VideoProbe(
        width=1920,
        height=1080,
        avg_fps=30.0,
        rotation_degrees=0,
    )
    fail_after_writes: int | None = None
    calls: list[tuple[object, ...]] = field(default_factory=list)
    _writes: int = 0

    def probe_video(self, path: Path) -> VideoProbe:
        self.calls.append(("probe_video", path))
        return self.probe

    def video_timeline(
        self, path: Path, fps_limit: int | None
    ) -> tuple[TimelineFrame, ...]:
        self.calls.append(("video_timeline", path, fps_limit))
        if fps_limit is None or not self.timeline or self.probe.avg_fps < fps_limit:
            return self.timeline
        stride = max(1, math.ceil(self.probe.avg_fps / fps_limit))
        bounded = list(self.timeline[::stride])
        if bounded[-1] != self.timeline[-1]:
            bounded.append(self.timeline[-1])
        return tuple(bounded)

    def _before_write(self) -> None:
        if (
            self.fail_after_writes is not None
            and self._writes >= self.fail_after_writes
        ):
            raise RuntimeError("injected media failure")
        self._writes += 1

    def extract_video_indices(
        self,
        path: Path,
        indices: tuple[int, ...],
        destination: Path,
        long_edge_cap: int | None,
    ) -> None:
        self.calls.append(
            ("extract_video_indices", path, indices, destination, long_edge_cap)
        )
        destination.mkdir(parents=True, exist_ok=True)
        width, height = self.probe.width, self.probe.height
        if self.probe.rotation_degrees in {90, 270}:
            width, height = height, width
        if long_edge_cap is not None and max(width, height) > long_edge_cap:
            scale = long_edge_cap / max(width, height)
            width = max(1, round(width * scale))
            height = max(1, round(height * scale))
        for output_index, source_index in enumerate(indices):
            self._before_write()
            Image.new(
                "RGB",
                (width, height),
                (source_index % 255, 32, 64),
            ).save(destination / f"frame_{output_index:06d}.png")

    def copy_photo(
        self,
        path: Path,
        destination: Path,
        long_edge_cap: int | None,
    ) -> None:
        self.calls.append(("copy_photo", path, destination, long_edge_cap))
        self._before_write()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            if long_edge_cap is not None and max(image.size) > long_edge_cap:
                image.thumbnail(
                    (long_edge_cap, long_edge_cap), Image.Resampling.LANCZOS
                )
            image.save(destination)


@pytest.fixture
def video_inventory(tmp_path: Path) -> SourceInventory:
    root = tmp_path / "video-input"
    root.mkdir()
    (root / "room.mov").write_bytes(b"video")
    return discover_source(root)


@pytest.fixture
def photo_inventory(tmp_path: Path) -> SourceInventory:
    root = tmp_path / "photo-input"
    _write_image(root / "IMG_0001.JPG")
    _write_image(root / "IMG_0002.JPG", value=96)
    return discover_source(root)


@pytest.fixture
def large_photo_inventory(tmp_path: Path) -> SourceInventory:
    root = tmp_path / "large-photo-input"
    for index in range(305):
        _write_image(root / f"IMG_{index:04d}.png", size=(2, 2), value=index % 255)
    return discover_source(root)


@pytest.fixture
def fake_media_backend() -> FakeMediaBackend:
    timeline = tuple(
        TimelineFrame(
            source_index=index,
            source_pts=index * 1_000,
            timestamp_s=index / 12.0,
        )
        for index in range(49)
    )
    return FakeMediaBackend(timeline=timeline)

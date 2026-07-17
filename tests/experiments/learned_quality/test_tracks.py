from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from PIL import Image

from experiments.learned_quality.contracts import FrameArtifact
from experiments.learned_quality.tracks import (
    TRACK_AUDIT_SCHEMA_VERSION,
    TrackQualificationError,
    TrackQualificationPolicy,
    qualify_colmap_static_tracks,
)


def _frame(
    tmp_path: Path, index: int, *, image_name: str | None = None
) -> FrameArtifact:
    name = image_name or f"frame_{index:06d}.png"
    path = tmp_path / f"selected-{index:06d}.png"
    Image.new("RGB", (8, 6), (index % 255, 0, 0)).save(path)
    return FrameArtifact(
        image_name=name,
        frame_id=f"frame-{index:06d}",
        path=path,
        sha256=f"{index:064x}",
    )


def _write_colmap_model(
    tmp_path: Path,
    *,
    image_rows: tuple[tuple[int, str, tuple[tuple[float, float, int], ...]], ...],
    point_rows: tuple[
        tuple[int, tuple[float, float, float], float, tuple[tuple[int, int], ...]], ...
    ],
) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    images_lines: list[str] = []
    for image_id, image_name, points in image_rows:
        images_lines.append(f"{image_id} 1 0 0 0 0 0 0 1 {image_name}")
        images_lines.append(
            " ".join(f"{x!r} {y!r} {point_id}" for x, y, point_id in points)
        )
    (model / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")
    point_lines = []
    for point_id, xyz, error, observations in point_rows:
        track = " ".join(
            f"{image_id} {point_index}" for image_id, point_index in observations
        )
        point_lines.append(
            f"{point_id} {xyz[0]!r} {xyz[1]!r} {xyz[2]!r} "
            f"255 255 255 {error!r} {track}"
        )
    (model / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")
    return model


def _single_track_fixture(
    tmp_path: Path,
    coordinates: tuple[tuple[float, float], ...],
    *,
    xyz: tuple[float, float, float] = (0.0, 0.0, 1.0),
    error: float = 0.1,
    point_ids: tuple[int, ...] | None = None,
) -> tuple[Path, tuple[FrameArtifact, ...]]:
    frames = tuple(_frame(tmp_path, index) for index in range(len(coordinates)))
    ids = point_ids or (7,) * len(coordinates)
    image_rows = tuple(
        (index + 1, frame.image_name, ((x, y, ids[index]),))
        for index, (frame, (x, y)) in enumerate(zip(frames, coordinates, strict=True))
    )
    model = _write_colmap_model(
        tmp_path,
        image_rows=image_rows,
        point_rows=(
            (
                7,
                xyz,
                error,
                tuple((index + 1, 0) for index in range(len(coordinates))),
            ),
        ),
    )
    return model, frames


@pytest.mark.parametrize(
    ("x", "y", "reason"),
    (
        (float("nan"), 2.0, "non_finite"),
        (float("inf"), 2.0, "non_finite"),
        (-0.01, 2.0, "out_of_bounds"),
        (8.0, 2.0, "out_of_bounds"),
        (2.0, 6.0, "out_of_bounds"),
    ),
)
def test_qualification_discards_invalid_observation_without_clamping(
    tmp_path: Path, x: float, y: float, reason: str
) -> None:
    model, frames = _single_track_fixture(
        tmp_path,
        ((x, y), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)),
    )

    result = qualify_colmap_static_tracks(
        model,
        frames,
        tmp_path / "track_audit.json",
        policy=TrackQualificationPolicy(maximum_invalid_fraction=0.5),
    )

    assert len(result.tracks) == 1
    assert tuple((item.x, item.y) for item in result.tracks[0].observations) == (
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
    )
    assert result.report.rejected_observations_by_reason[reason] == 1
    assert result.audit_path.is_file()
    assert len(result.audit_sha256) == 64


def test_qualification_accepts_values_immediately_inside_bounds(
    tmp_path: Path,
) -> None:
    model, frames = _single_track_fixture(
        tmp_path,
        ((0.0, 0.0), (7.999999, 5.999999), (4.0, 3.0)),
    )

    result = qualify_colmap_static_tracks(
        model,
        frames,
        tmp_path / "track_audit.json",
        policy=TrackQualificationPolicy(),
    )

    assert len(result.tracks) == 1
    assert result.report.rejected_observation_count == 0


@pytest.mark.parametrize(
    ("xyz", "error", "reason"),
    (
        ((math.nan, 0.0, 1.0), 0.1, "invalid_geometry"),
        ((0.0, math.inf, 1.0), 0.1, "invalid_geometry"),
        ((0.0, 0.0, 1.0), math.nan, "invalid_reprojection_error"),
        ((0.0, 0.0, 1.0), -0.1, "invalid_reprojection_error"),
    ),
)
def test_invalid_track_geometry_writes_audit_then_fails(
    tmp_path: Path,
    xyz: tuple[float, float, float],
    error: float,
    reason: str,
) -> None:
    model, frames = _single_track_fixture(
        tmp_path,
        ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)),
        xyz=xyz,
        error=error,
    )
    audit_path = tmp_path / "track_audit.json"

    with pytest.raises(TrackQualificationError, match="no qualified") as caught:
        qualify_colmap_static_tracks(
            model,
            frames,
            audit_path,
            policy=TrackQualificationPolicy(),
        )

    assert audit_path.is_file()
    assert caught.value.report.rejected_tracks_by_reason[reason] == 1
    assert sum(caught.value.report.rejected_observations_by_reason.values()) == (
        caught.value.report.rejected_observation_count
    )
    assert json.loads(audit_path.read_text(encoding="utf-8"))["schema_version"] == (
        TRACK_AUDIT_SCHEMA_VERSION
    )


def test_point_id_mismatch_and_short_track_are_rejected(tmp_path: Path) -> None:
    model, frames = _single_track_fixture(
        tmp_path,
        ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)),
        point_ids=(8, 7, 7),
    )

    with pytest.raises(TrackQualificationError) as caught:
        qualify_colmap_static_tracks(
            model,
            frames,
            tmp_path / "track_audit.json",
            policy=TrackQualificationPolicy(),
        )

    assert caught.value.report.rejected_observations_by_reason["point_id_mismatch"] == 1
    assert (
        caught.value.report.rejected_tracks_by_reason["insufficient_observations"] == 1
    )


def test_missing_colmap_image_is_audited_but_three_valid_views_survive(
    tmp_path: Path,
) -> None:
    model, frames = _single_track_fixture(
        tmp_path,
        ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)),
    )
    points_path = model / "points3D.txt"
    points_path.write_text(
        points_path.read_text(encoding="utf-8").rstrip() + " 999 0\n",
        encoding="utf-8",
    )

    result = qualify_colmap_static_tracks(
        model,
        frames,
        tmp_path / "track_audit.json",
        policy=TrackQualificationPolicy(),
    )

    assert len(result.tracks) == 1
    assert result.report.rejected_observations_by_reason["missing_image"] == 1


def test_duplicate_track_ids_are_all_rejected(tmp_path: Path) -> None:
    model, frames = _single_track_fixture(
        tmp_path,
        ((1.0, 1.0), (2.0, 2.0), (3.0, 3.0)),
    )
    points_path = model / "points3D.txt"
    row = points_path.read_text(encoding="utf-8").strip()
    points_path.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(TrackQualificationError, match="no qualified") as caught:
        qualify_colmap_static_tracks(
            model,
            frames,
            tmp_path / "track_audit.json",
            policy=TrackQualificationPolicy(),
        )

    assert caught.value.report.raw_track_count == 2
    assert caught.value.report.rejected_tracks_by_reason["duplicate_track_id"] == 2


def test_duplicate_canonical_frame_observation_is_never_published(
    tmp_path: Path,
) -> None:
    frames = tuple(_frame(tmp_path, index) for index in range(3))
    model = _write_colmap_model(
        tmp_path,
        image_rows=(
            (1, frames[0].image_name, ((1.0, 1.0, 7),)),
            (2, frames[0].image_name, ((2.0, 2.0, 7),)),
            (3, frames[1].image_name, ((3.0, 3.0, 7),)),
            (4, frames[2].image_name, ((4.0, 4.0, 7),)),
        ),
        point_rows=((7, (0.0, 0.0, 1.0), 0.1, ((1, 0), (2, 0), (3, 0), (4, 0))),),
    )

    result = qualify_colmap_static_tracks(
        model,
        frames,
        tmp_path / "track_audit.json",
        policy=TrackQualificationPolicy(maximum_invalid_fraction=0.5),
    )

    assert len(result.tracks) == 1
    assert len(result.tracks[0].observations) == 3
    assert (
        result.report.rejected_observations_by_reason["duplicate_frame_observation"]
        == 1
    )


def test_systematic_coordinate_mismatch_writes_audit_then_fails(
    tmp_path: Path,
) -> None:
    frames = tuple(_frame(tmp_path, index) for index in range(4))
    image_points: list[list[tuple[float, float, int]]] = [[], [], [], []]
    point_rows = []
    for point_id in range(25):
        for image_index in range(4):
            invalid = point_id < 2 and image_index == 0
            image_points[image_index].append(
                ((8.0 if invalid else float(image_index + 1)), 2.0, point_id)
            )
        point_rows.append(
            (
                point_id,
                (float(point_id), 0.0, 1.0),
                0.1,
                tuple((image_index + 1, point_id) for image_index in range(4)),
            )
        )
    model = _write_colmap_model(
        tmp_path,
        image_rows=tuple(
            (index + 1, frames[index].image_name, tuple(image_points[index]))
            for index in range(4)
        ),
        point_rows=tuple(point_rows),
    )
    audit_path = tmp_path / "track_audit.json"

    with pytest.raises(TrackQualificationError, match="systematic") as caught:
        qualify_colmap_static_tracks(
            model,
            frames,
            audit_path,
            policy=TrackQualificationPolicy(maximum_invalid_fraction=0.01),
        )

    assert caught.value.audit_path.is_file()
    assert caught.value.report.raw_observation_count == 100
    assert caught.value.report.rejected_observation_count == 2
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["rejected_observations_by_reason"] == {"out_of_bounds": 2}
    assert "NaN" not in audit_path.read_text(encoding="utf-8")
    assert "Infinity" not in audit_path.read_text(encoding="utf-8")


def test_qualification_is_deterministic_and_sorted(tmp_path: Path) -> None:
    frames = tuple(_frame(tmp_path, index) for index in range(3))
    model = _write_colmap_model(
        tmp_path,
        image_rows=tuple(
            (
                index + 1,
                frames[index].image_name,
                ((2.0, 2.0, 9), (1.0, 1.0, 3)),
            )
            for index in range(3)
        ),
        point_rows=(
            (9, (1.0, 0.0, 1.0), 0.2, ((3, 0), (2, 0), (1, 0))),
            (3, (0.0, 0.0, 1.0), 0.1, ((3, 1), (2, 1), (1, 1))),
        ),
    )

    first = qualify_colmap_static_tracks(
        model,
        frames,
        tmp_path / "first.json",
        policy=TrackQualificationPolicy(),
    )
    second = qualify_colmap_static_tracks(
        model,
        frames,
        tmp_path / "second.json",
        policy=TrackQualificationPolicy(),
    )

    assert tuple(track.track_id for track in first.tracks) == (3, 9)
    assert tuple(
        observation.frame_id for observation in first.tracks[0].observations
    ) == tuple(frame.frame_id for frame in frames)
    assert first.audit_sha256 == second.audit_sha256
    assert first.audit_path.read_bytes() == second.audit_path.read_bytes()

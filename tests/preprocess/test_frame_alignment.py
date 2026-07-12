from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.static_pipeline.contracts import (
    FrameRecord,
    SelectionManifest,
    SelectionPolicy,
)


def _camera(marker: float = 0.0) -> dict[str, object]:
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = marker + 1.0
    w2c = np.eye(4, dtype=np.float64)
    w2c[0, 3] = marker
    return {"K": K, "w2c": w2c, "width": 640, "height": 480}


def _record(
    frame_id: str,
    output_name: str,
    *,
    timestamp_s: float | None,
    selected: bool = True,
) -> FrameRecord:
    return FrameRecord(
        frame_id=frame_id,
        source_relative_path=output_name,
        source_index=None,
        source_pts=None,
        timestamp_s=timestamp_s,
        output_name=output_name,
        sha256="0" * 64,
        selected=selected,
        metrics=None,
        selection_score=None,
        reasons=(),
    )


def _manifest(*records: FrameRecord) -> SelectionManifest:
    return SelectionManifest(
        schema_version=1,
        source_digest="1" * 64,
        effective_mode="smart",
        policy=SelectionPolicy(
            mode="smart",
            frame_budget=300,
            resolution_long_edge_cap=1280,
        ),
        frames=records,
        image_set_digest="2" * 64,
    )


def _join(*args: object, **kwargs: object) -> tuple[object, ...]:
    from backend.preprocess.frame_alignment import join_registered_frames

    return join_registered_frames(*args, **kwargs)


def test_unregistered_middle_frame_is_skipped_without_positional_pairing(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for name in ("frame_000.png", "frame_001.png", "frame_002.png"):
        (frames / name).write_bytes(name.encode())

    joined = _join(
        frames,
        {
            "frame_000.png": _camera(0.0),
            "frame_002.png": _camera(2.0),
        },
    )

    assert [item.image_name for item in joined] == [
        "frame_000.png",
        "frame_002.png",
    ]
    assert [item.image_path for item in joined] == [
        frames / "frame_000.png",
        frames / "frame_002.png",
    ]
    assert [float(item.w2c[0, 3]) for item in joined] == [0.0, 2.0]
    assert all(item.frame_id is None for item in joined)
    assert all(item.timestamp_s is None for item in joined)


def test_camera_name_requires_an_exact_physical_match(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "other.png").write_bytes(b"frame")

    with pytest.raises(FileNotFoundError, match="missing.png"):
        _join(frames, {"missing.png": _camera()})


def test_duplicate_nested_basenames_are_rejected(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    (frames / "left").mkdir(parents=True)
    (frames / "right").mkdir()
    (frames / "left" / "same.png").write_bytes(b"left")
    (frames / "right" / "same.png").write_bytes(b"right")

    with pytest.raises(ValueError, match="duplicate.*same.png"):
        _join(frames, {"same.png": _camera()})


def test_manifest_order_and_tied_timestamps_are_preserved(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for name in ("frame_1.png", "frame_2.png", "unused.png"):
        (frames / name).write_bytes(name.encode())
    manifest = _manifest(
        _record("id-two", "frame_2.png", timestamp_s=1.25),
        _record("id-unused", "unused.png", timestamp_s=1.25, selected=False),
        _record("id-one", "frame_1.png", timestamp_s=1.25),
    )

    joined = _join(
        frames,
        {
            "frame_1.png": _camera(1.0),
            "frame_2.png": _camera(2.0),
        },
        manifest=manifest,
    )

    assert [item.image_name for item in joined] == ["frame_2.png", "frame_1.png"]
    assert [item.frame_id for item in joined] == ["id-two", "id-one"]
    assert [item.timestamp_s for item in joined] == [1.25, 1.25]


def test_without_manifest_image_names_use_natural_numeric_order(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for name in ("frame_10.png", "frame_2.png", "frame_1.png"):
        (frames / name).write_bytes(name.encode())

    joined = _join(
        frames,
        {
            "frame_10.png": _camera(10.0),
            "frame_2.png": _camera(2.0),
            "frame_1.png": _camera(1.0),
        },
    )

    assert [item.image_name for item in joined] == [
        "frame_1.png",
        "frame_2.png",
        "frame_10.png",
    ]


def test_depth_is_joined_by_stem_and_missing_per_name_depth_is_none(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames"
    depth = tmp_path / "depth"
    frames.mkdir()
    depth.mkdir()
    for name in ("frame_1.png", "frame_2.png"):
        (frames / name).write_bytes(name.encode())
    expected_depth = depth / "frame_2_depth.npy"
    expected_depth.write_bytes(b"depth")

    joined = _join(
        frames,
        {
            "frame_1.png": _camera(1.0),
            "frame_2.png": _camera(2.0),
        },
        depth_dir=depth,
    )

    assert [item.depth_path for item in joined] == [None, expected_depth]


def test_empty_camera_map_and_empty_camera_name_are_rejected(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame.png").write_bytes(b"frame")

    with pytest.raises(ValueError, match="camera map"):
        _join(frames, {})
    with pytest.raises(ValueError, match="camera name"):
        _join(frames, {"": _camera()})


@pytest.mark.parametrize(
    ("camera", "message"),
    [
        ({"K": np.eye(4), "w2c": np.eye(4)}, "K"),
        ({"K": np.eye(3), "w2c": np.eye(3)}, "w2c"),
        (
            {
                "K": np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                "w2c": np.eye(4),
            },
            "K",
        ),
        (
            {
                "K": np.eye(3),
                "w2c": np.array(
                    [
                        [1.0, 0.0, 0.0, np.inf],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
            },
            "w2c",
        ),
    ],
)
def test_malformed_or_non_finite_camera_matrices_are_rejected(
    tmp_path: Path,
    camera: dict[str, object],
    message: str,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame.png").write_bytes(b"frame")

    with pytest.raises(ValueError, match=message):
        _join(frames, {"frame.png": camera})


def test_manifest_duplicate_selected_names_are_rejected(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame.png").write_bytes(b"frame")
    manifest = _manifest(
        _record("first", "frame.png", timestamp_s=0.0),
        _record("second", "frame.png", timestamp_s=1.0),
    )

    with pytest.raises(ValueError, match="duplicate.*frame.png"):
        _join(frames, {"frame.png": _camera()}, manifest=manifest)


def test_manifest_must_select_every_registered_camera(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for name in ("frame_1.png", "frame_2.png"):
        (frames / name).write_bytes(name.encode())
    manifest = _manifest(_record("one", "frame_1.png", timestamp_s=0.0))

    with pytest.raises(ValueError, match="frame_2.png"):
        _join(
            frames,
            {
                "frame_1.png": _camera(1.0),
                "frame_2.png": _camera(2.0),
            },
            manifest=manifest,
        )


def test_symlink_frame_and_depth_are_not_accepted(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    depth = tmp_path / "depth"
    physical = tmp_path / "physical"
    frames.mkdir()
    depth.mkdir()
    physical.mkdir()
    (physical / "frame.png").write_bytes(b"frame")
    (physical / "frame_depth.npy").write_bytes(b"depth")
    try:
        (frames / "frame.png").symlink_to(physical / "frame.png")
        (depth / "frame_depth.npy").symlink_to(physical / "frame_depth.npy")
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    with pytest.raises(FileNotFoundError, match="frame.png"):
        _join(
            frames,
            {"frame.png": _camera()},
            depth_dir=depth,
        )

    (frames / "regular.png").write_bytes(b"regular")
    (physical / "regular_depth.npy").write_bytes(b"depth")
    (depth / "regular_depth.npy").symlink_to(physical / "regular_depth.npy")
    joined = _join(frames, {"regular.png": _camera()}, depth_dir=depth)
    assert joined[0].depth_path is None

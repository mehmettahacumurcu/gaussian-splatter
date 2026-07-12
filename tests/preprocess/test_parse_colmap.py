from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.preprocess import parse_colmap


def _write_model(
    root: Path,
    *,
    camera_lines: tuple[str, ...] = ("7 PINHOLE 640 480 500 510 320 240",),
    image_records: tuple[tuple[str, str | None], ...] = (
        ("11 1 0 0 0 0 0 0 7 frame.png", ""),
    ),
    point_lines: tuple[str, ...] = ("1 1 2 3 10 20 30 0.5 11 0 12 1",),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cameras.txt").write_text(
        "# cameras\n" + "\n".join(camera_lines) + "\n",
        encoding="utf-8",
    )
    image_text = [
        "# images",
        f"# Number of images: {len(image_records)}, mean observations per image: 0",
    ]
    for header, observations in image_records:
        image_text.append(header)
        if observations is not None:
            image_text.append(observations)
    (root / "images.txt").write_text(
        "\n".join(image_text) + "\n",
        encoding="utf-8",
    )
    (root / "points3D.txt").write_text(
        "# points\n" + "\n".join(point_lines) + "\n",
        encoding="utf-8",
    )
    return root


def test_exact_camera_reader_preserves_empty_observation_lines_and_exact_names(
    tmp_path: Path,
) -> None:
    model = _write_model(
        tmp_path / "model",
        image_records=(
            ("11 1 0 0 0 0 0 0 7 nested/Room Çalışma 01.PNG", ""),
            ("42 1 0 0 0 1 2 3 7 second.png", "1.0 2.0 -1"),
        ),
    )

    cameras = parse_colmap.parse_cameras_from_model(model)

    assert tuple(cameras) == ("nested/Room Çalışma 01.PNG", "second.png")
    first = cameras["nested/Room Çalışma 01.PNG"]
    assert set(first) == {"image_id", "K", "R", "t", "w2c", "width", "height"}
    assert first["image_id"] == 11
    assert first["width"] == 640
    assert first["height"] == 480
    np.testing.assert_allclose(
        first["K"],
        np.array([[500, 0, 320], [0, 510, 240], [0, 0, 1]], dtype=np.float64),
    )
    np.testing.assert_allclose(first["w2c"], np.eye(4))
    np.testing.assert_allclose(cameras["second.png"]["t"], [1, 2, 3])


def test_exact_camera_reader_is_side_effect_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = _write_model(tmp_path / "model")

    parse_colmap.parse_cameras_from_model(model)

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("camera_line", "expected_k"),
    (
        ("3 PINHOLE 20 10 8 9 4 5", [[8, 0, 4], [0, 9, 5], [0, 0, 1]]),
        (
            "3 SIMPLE_PINHOLE 20 10 8 4 5",
            [[8, 0, 4], [0, 8, 5], [0, 0, 1]],
        ),
        (
            "3 SIMPLE_RADIAL 20 10 8 4 5 0.01",
            [[8, 0, 4], [0, 8, 5], [0, 0, 1]],
        ),
        (
            "3 OPENCV 20 10 8 9 4 5 0.1 0.2 0.3 0.4",
            [[8, 0, 4], [0, 9, 5], [0, 0, 1]],
        ),
    ),
)
def test_supported_camera_models_build_legacy_intrinsics(
    tmp_path: Path,
    camera_line: str,
    expected_k: list[list[int]],
) -> None:
    model = _write_model(
        tmp_path / "model",
        camera_lines=(camera_line,),
        image_records=(("9 1 0 0 0 0 0 0 3 image.png", ""),),
    )

    cameras = parse_colmap.parse_cameras_from_model(model)

    np.testing.assert_allclose(cameras["image.png"]["K"], expected_k)


def test_exact_readers_never_search_sibling_models_and_legacy_wrappers_keep_shape(
    tmp_path: Path,
) -> None:
    colmap_root = tmp_path / "colmap"
    exact = _write_model(
        colmap_root / "sparse" / "0",
        image_records=(("11 1 0 0 0 0 0 0 7 exact.png", ""),),
        point_lines=("1 1 2 3 10 20 30 0.5 11 0",),
    )
    selected_by_legacy = _write_model(
        colmap_root / "sparse" / "1",
        image_records=(("12 1 0 0 0 0 0 0 7 legacy.png", ""),),
        point_lines=("2 4 5 6 40 50 60 0.25 12 0 12 1",),
    )
    (exact / "cameras.bin").write_bytes(b"x")
    (selected_by_legacy / "cameras.bin").write_bytes(b"larger")

    exact_cameras = parse_colmap.parse_cameras_from_model(exact)
    exact_points = parse_colmap.load_points3d_from_model(exact)
    legacy_cameras = parse_colmap.parse_cameras(colmap_root)
    legacy_points = parse_colmap.load_points3d(colmap_root)
    legacy_confidence = parse_colmap.load_points3d_with_confidence(colmap_root)

    assert tuple(exact_cameras) == ("exact.png",)
    assert tuple(legacy_cameras) == ("legacy.png",)
    assert len(exact_points) == 4
    assert len(legacy_points) == 2
    assert len(legacy_confidence) == 4
    np.testing.assert_array_equal(legacy_points[0], legacy_confidence[0])
    np.testing.assert_array_equal(legacy_points[1], legacy_confidence[1])


@pytest.mark.parametrize(
    ("records", "message"),
    (
        (
            (
                ("1 1 0 0 0 0 0 0 7 same.png", ""),
                ("2 1 0 0 0 0 0 0 7 same.png", ""),
            ),
            "Duplicate image name",
        ),
        (
            (
                ("1 1 0 0 0 0 0 0 7 first.png", ""),
                ("1 1 0 0 0 0 0 0 7 second.png", ""),
            ),
            "Duplicate image ID",
        ),
    ),
)
def test_duplicate_image_identity_is_rejected_before_name_mapping(
    tmp_path: Path,
    records: tuple[tuple[str, str], ...],
    message: str,
) -> None:
    model = _write_model(tmp_path / "model", image_records=records)

    with pytest.raises(ValueError, match=message):
        parse_colmap.parse_cameras_from_model(model)


def test_duplicate_camera_id_is_rejected(tmp_path: Path) -> None:
    model = _write_model(
        tmp_path / "model",
        camera_lines=(
            "7 PINHOLE 640 480 500 510 320 240",
            "7 PINHOLE 320 240 250 250 160 120",
        ),
    )

    with pytest.raises(ValueError, match="Duplicate camera ID"):
        parse_colmap.parse_cameras_from_model(model)


def test_missing_camera_reference_is_rejected_deterministically(tmp_path: Path) -> None:
    model = _write_model(
        tmp_path / "model",
        image_records=(("11 1 0 0 0 0 0 0 99 missing.png", ""),),
    )

    with pytest.raises(ValueError, match="camera ID 99"):
        parse_colmap.parse_cameras_from_model(model)


@pytest.mark.parametrize(
    ("camera_line", "error_type", "message"),
    (
        ("7 RADIAL 640 480 500 320 240 0.1 0.2", NotImplementedError, "RADIAL"),
        ("7 PINHOLE 640 480 500 510 320", ValueError, "camera"),
        ("not-a-camera", ValueError, "camera"),
    ),
)
def test_unsupported_or_malformed_camera_rows_fail_deterministically(
    tmp_path: Path,
    camera_line: str,
    error_type: type[Exception],
    message: str,
) -> None:
    model = _write_model(tmp_path / "model", camera_lines=(camera_line,))

    with pytest.raises(error_type, match=message):
        parse_colmap.parse_cameras_from_model(model)


@pytest.mark.parametrize(
    ("header", "message"),
    (
        ("11 0 0 0 0 0 0 0 7 zero.png", "quaternion"),
        ("11 nan 0 0 0 0 0 0 7 nan.png", "quaternion"),
        ("11 1 0 0 0 0 0 0", "image header"),
    ),
)
def test_invalid_image_headers_and_quaternions_are_rejected(
    tmp_path: Path,
    header: str,
    message: str,
) -> None:
    model = _write_model(tmp_path / "model", image_records=((header, ""),))

    with pytest.raises(ValueError, match=message):
        parse_colmap.parse_cameras_from_model(model)


@pytest.mark.parametrize(
    "records",
    (
        (("11 1 0 0 0 0 0 0 7 missing.png", None),),
        (("11 1 0 0 0 0 0 0 7 truncated.png", "1.0 2.0"),),
        (("11 1 0 0 0 0 0 0 7 malformed.png", "1.0 nope -1"),),
    ),
)
def test_missing_or_malformed_points2d_record_is_rejected(
    tmp_path: Path,
    records: tuple[tuple[str, str | None], ...],
) -> None:
    model = _write_model(tmp_path / "model", image_records=records)

    with pytest.raises(ValueError, match="POINTS2D"):
        parse_colmap.parse_cameras_from_model(model)


@pytest.mark.parametrize("second_observations", ("", None))
def test_missing_points2d_cannot_consume_numeric_filename_header(
    tmp_path: Path,
    second_observations: str | None,
) -> None:
    model = _write_model(
        tmp_path / "model",
        image_records=(
            ("11 1 0 0 0 0 0 0 7 first.png", None),
            ("12 1 0 0 0 0 0 0 7 1 2 3", second_observations),
        ),
    )

    with pytest.raises(ValueError, match="POINTS2D"):
        parse_colmap.parse_cameras_from_model(model)


def test_header_shaped_integer_points2d_row_remains_valid(tmp_path: Path) -> None:
    model = _write_model(
        tmp_path / "model",
        image_records=(
            (
                "11 1 0 0 0 0 0 0 7 frame.png",
                "10 20 1 30 40 2 50 60 7 70 80 3",
            ),
        ),
    )

    cameras = parse_colmap.parse_cameras_from_model(model)

    assert tuple(cameras) == ("frame.png",)


def test_exact_camera_reader_requires_declared_image_count(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "model")
    images_file = model / "images.txt"
    images_file.write_text(
        "\n".join(
            line
            for line in images_file.read_text(encoding="utf-8").splitlines()
            if not line.startswith("# Number of images:")
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="declared image count"):
        parse_colmap.parse_cameras_from_model(model)


def test_exact_point_reader_returns_metrics_and_stable_dtypes(tmp_path: Path) -> None:
    model = _write_model(
        tmp_path / "model",
        point_lines=(
            "4 1.5 2.5 3.5 10 20 30 0.25 11 0 12 4",
            "9 -1 -2 -3 40 50 60 1.5",
        ),
    )

    xyz, rgb, tracks, errors = parse_colmap.load_points3d_from_model(model)

    assert xyz.dtype == np.float32
    assert rgb.dtype == np.uint8
    assert tracks.dtype == np.int32
    assert errors.dtype == np.float32
    np.testing.assert_allclose(xyz, [[1.5, 2.5, 3.5], [-1, -2, -3]])
    np.testing.assert_array_equal(rgb, [[10, 20, 30], [40, 50, 60]])
    np.testing.assert_array_equal(tracks, [2, 0])
    np.testing.assert_allclose(errors, [0.25, 1.5])


@pytest.mark.parametrize(
    ("point_lines", "message"),
    (
        (("1 1 2 3 10 20 30",), "point"),
        (("1 1 2 3 10 20 30 0.5 11",), "track"),
        (("1 1 2 3 10 20 30 0.5 11 nope",), "point"),
        (
            (
                "1 1 2 3 10 20 30 0.5",
                "1 4 5 6 40 50 60 1.0",
            ),
            "Duplicate point ID",
        ),
    ),
)
def test_malformed_point_rows_are_rejected(
    tmp_path: Path,
    point_lines: tuple[str, ...],
    message: str,
) -> None:
    model = _write_model(tmp_path / "model", point_lines=point_lines)

    with pytest.raises(ValueError, match=message):
        parse_colmap.load_points3d_from_model(model)


def test_empty_points_file_fails_deterministically(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "model", point_lines=())

    with pytest.raises(RuntimeError, match="points3D.txt.*empty"):
        parse_colmap.load_points3d_from_model(model)


@pytest.mark.parametrize("missing_name", ("cameras.txt", "images.txt"))
def test_exact_camera_reader_requires_regular_text_files(
    tmp_path: Path,
    missing_name: str,
) -> None:
    model = _write_model(tmp_path / "model")
    (model / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match="Incomplete COLMAP text model"):
        parse_colmap.parse_cameras_from_model(model)


def test_exact_point_reader_requires_regular_points_file(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "model")
    (model / "points3D.txt").unlink()

    with pytest.raises(FileNotFoundError, match="points3D.txt"):
        parse_colmap.load_points3d_from_model(model)


@pytest.mark.parametrize(
    ("linked_name", "reader"),
    (
        ("cameras.txt", parse_colmap.parse_cameras_from_model),
        ("images.txt", parse_colmap.parse_cameras_from_model),
        ("points3D.txt", parse_colmap.load_points3d_from_model),
    ),
)
def test_exact_readers_reject_symlinked_model_text_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    linked_name: str,
    reader: object,
) -> None:
    model = _write_model(tmp_path / "model")
    linked_path = model / linked_name
    real_is_symlink = Path.is_symlink

    def report_link(path: Path) -> bool:
        return path == linked_path or real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_link)

    with pytest.raises(FileNotFoundError):
        reader(model)  # type: ignore[operator]


def test_legacy_quaternion_zero_behavior_is_unchanged() -> None:
    np.testing.assert_array_equal(parse_colmap.quat_to_rotmat(0, 0, 0, 0), np.eye(3))

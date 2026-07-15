from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import experiments.learned_quality.da3 as da3_module
from experiments.learned_quality.da3 import (
    ANCHOR_PROCESS_RESOLUTION,
    AnchorInferenceResult,
    AnchorStageFailure,
    DA3Frame,
    FinalPoseCamera,
    FinalPoseStageFailure,
    FrameChunk,
    MetricSkyResult,
    PinholeCamera,
    PoseConditionedDepthResult,
    normalize_w2c,
    robust_shared_pinhole,
    run_anchor_inference,
    run_metric_sky,
    run_pose_conditioned_depth,
    schedule_overlapping_chunks,
    select_anchor_indices,
)
from experiments.learned_quality.lifecycle import VramReleaseRecord


@pytest.mark.parametrize(
    ("vram_gb", "expected_count"),
    ((40.0, 96), (69.9, 96), (70.0, 120), (80, 120)),
)
def test_select_anchor_indices_uses_locked_budget_and_endpoints(
    vram_gb: float,
    expected_count: int,
) -> None:
    indices = select_anchor_indices(frame_count=800, vram_gb=vram_gb)

    assert len(indices) == expected_count
    assert indices[0] == 0
    assert indices[-1] == 799
    assert indices == tuple(sorted(set(indices)))
    gaps = tuple(second - first for first, second in zip(indices, indices[1:]))
    assert max(gaps) - min(gaps) <= 1


@pytest.mark.parametrize("frame_count", (1, 2, 17, 96))
def test_select_anchor_indices_caps_only_at_frame_count(frame_count: int) -> None:
    assert select_anchor_indices(frame_count, 40.0) == tuple(range(frame_count))


@pytest.mark.parametrize("frame_count", (True, False, 0, -1, 1.0, "4"))
def test_select_anchor_indices_rejects_nonpositive_or_non_plain_frame_count(
    frame_count: object,
) -> None:
    with pytest.raises(ValueError, match="frame_count"):
        select_anchor_indices(frame_count, 80.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "vram_gb",
    (True, False, -0.1, float("nan"), float("inf"), "80"),
)
def test_select_anchor_indices_rejects_invalid_vram(vram_gb: object) -> None:
    with pytest.raises(ValueError, match="vram_gb"):
        select_anchor_indices(800, vram_gb)  # type: ignore[arg-type]


def test_normalize_w2c_appends_affine_row_without_inverting_pose() -> None:
    source = np.array(
        [
            [0, -1, 0, 3],
            [1, 0, 0, -2],
            [0, 0, 1, 5],
        ],
        dtype=np.int64,
    )

    normalized = normalize_w2c(source)

    np.testing.assert_array_equal(normalized[:3], source)
    np.testing.assert_array_equal(normalized[3], (0.0, 0.0, 0.0, 1.0))
    assert normalized.shape == (4, 4)
    assert normalized.dtype == np.float64
    assert normalized.flags.writeable is False


def test_normalize_w2c_accepts_exact_affine_4x4_and_returns_copy() -> None:
    source = np.eye(4, dtype=np.float32)
    normalized = normalize_w2c(source)

    np.testing.assert_array_equal(normalized, source)
    assert normalized is not source
    assert normalized.dtype == np.float64


@pytest.mark.parametrize(
    "matrix",
    (
        np.eye(3),
        np.zeros((4, 3)),
        np.zeros((4, 5)),
        np.array([["x"] * 4] * 4),
        np.eye(4, dtype=np.complex128),
        np.eye(4, dtype=bool),
    ),
)
def test_normalize_w2c_rejects_malformed_or_nonnumeric_matrix(
    matrix: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="W2C"):
        normalize_w2c(matrix)


@pytest.mark.parametrize(
    "matrix",
    (
        np.array(
            [[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.000000000001]]
        ),
        np.array(
            [[-1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        np.array(
            [[2.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        np.array(
            [[1.0, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
        np.array(
            [[1.0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ),
    ),
)
def test_normalize_w2c_rejects_invalid_affine_or_rotation(
    matrix: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="W2C"):
        normalize_w2c(matrix)


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf"), 1.0e12))
def test_normalize_w2c_rejects_nonfinite_or_implausible_values(
    bad_value: float,
) -> None:
    matrix = np.eye(4)
    matrix[0, 3] = bad_value

    with pytest.raises(ValueError, match="W2C"):
        normalize_w2c(matrix)


@pytest.mark.parametrize(
    ("frame_count", "expected_windows"),
    (
        (1, ((0, 1),)),
        (48, ((0, 48),)),
        (49, ((0, 48), (1, 49))),
        (72, ((0, 48), (24, 72))),
        (73, ((0, 48), (24, 72), (25, 73))),
        (100, ((0, 48), (24, 72), (48, 96), (52, 100))),
    ),
)
def test_schedule_overlapping_chunks_has_exact_48_24_coverage(
    frame_count: int,
    expected_windows: tuple[tuple[int, int], ...],
) -> None:
    chunks = schedule_overlapping_chunks(frame_count)

    assert tuple((chunk.start, chunk.stop) for chunk in chunks) == expected_windows
    assert tuple(chunk.index for chunk in chunks) == tuple(range(len(chunks)))
    assert all(chunk.frame_indices == tuple(range(chunk.start, chunk.stop)) for chunk in chunks)
    assert all(chunk.stop - chunk.start <= 48 for chunk in chunks)
    assert set().union(*(set(chunk.frame_indices) for chunk in chunks)) == set(
        range(frame_count)
    )
    assert len({(chunk.start, chunk.stop) for chunk in chunks}) == len(chunks)
    assert chunks[-1].stop == frame_count


def test_frame_chunk_is_frozen() -> None:
    chunk = schedule_overlapping_chunks(3)[0]

    assert isinstance(chunk, FrameChunk)
    with pytest.raises(FrozenInstanceError):
        chunk.start = 1  # type: ignore[misc]


@pytest.mark.parametrize("frame_count", (True, False, 0, -1, 1.0, "48"))
def test_schedule_overlapping_chunks_rejects_invalid_frame_count(
    frame_count: object,
) -> None:
    with pytest.raises(ValueError, match="frame_count"):
        schedule_overlapping_chunks(frame_count)  # type: ignore[arg-type]


def _intrinsic(fx: float, fy: float, cx: float = 960.0, cy: float = 540.0) -> np.ndarray:
    return np.array(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)))


def test_robust_shared_pinhole_preserves_fx_fy_and_ignores_one_outlier() -> None:
    intrinsics = np.stack(
        (
            _intrinsic(1200.0, 1180.0),
            _intrinsic(1200.0, 1180.0),
            _intrinsic(1200.0, 1180.0),
            _intrinsic(9000.0, 7000.0),
        )
    )

    camera = robust_shared_pinhole(
        intrinsics,
        ((1920, 1080),) * len(intrinsics),
    )

    assert camera == PinholeCamera(
        model="PINHOLE",
        width=1920,
        height=1080,
        fx=1200.0,
        fy=1180.0,
        cx=960.0,
        cy=540.0,
    )
    with pytest.raises(FrozenInstanceError):
        camera.fx = 1.0  # type: ignore[misc]


def test_robust_shared_pinhole_accepts_one_intrinsic_and_one_video_size() -> None:
    camera = robust_shared_pinhole(
        np.stack((_intrinsic(800.0, 820.0, 320.0, 240.0),)),
        (640, 480),
    )

    assert camera.model == "PINHOLE"
    assert camera.matrix == (
        (800.0, 0.0, 320.0),
        (0.0, 820.0, 240.0),
        (0.0, 0.0, 1.0),
    )


@pytest.mark.parametrize(
    "intrinsics",
    (
        np.empty((0, 3, 3)),
        np.eye(3),
        np.zeros((2, 4, 4)),
        np.array([[['x'] * 3] * 3]),
        np.array([np.eye(3, dtype=np.complex128)]),
        np.array([np.eye(3, dtype=bool)]),
    ),
)
def test_robust_shared_pinhole_rejects_malformed_intrinsics(
    intrinsics: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="intrinsics"):
        robust_shared_pinhole(intrinsics, (640, 480))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda matrix: matrix.__setitem__((0, 0, 0), 0.0),
        lambda matrix: matrix.__setitem__((0, 1, 1), -1.0),
        lambda matrix: matrix.__setitem__((0, 0, 1), 0.01),
        lambda matrix: matrix.__setitem__((0, 1, 0), 0.01),
        lambda matrix: matrix.__setitem__((0, 2, 2), 1.000000000001),
        lambda matrix: matrix.__setitem__((0, 0, 2), 640.0),
        lambda matrix: matrix.__setitem__((0, 1, 2), 480.0),
        lambda matrix: matrix.__setitem__((0, 0, 0), float("nan")),
        lambda matrix: matrix.__setitem__((0, 0, 0), float("inf")),
        lambda matrix: matrix.__setitem__((0, 0, 0), 1.0e12),
    ),
)
def test_robust_shared_pinhole_rejects_nonphysical_calibration(mutate: object) -> None:
    intrinsics = np.stack((_intrinsic(800.0, 820.0, 320.0, 240.0),))
    mutate(intrinsics)  # type: ignore[operator]

    with pytest.raises(ValueError, match="intrinsics"):
        robust_shared_pinhole(intrinsics, (640, 480))


@pytest.mark.parametrize(
    "image_sizes",
    (
        (0, 480),
        (640, -1),
        (640.0, 480),
        (True, 480),
        ((640, 480),),
        ((640, 480), (800, 600), (640, 480)),
        "640x480",
    ),
)
def test_robust_shared_pinhole_rejects_invalid_or_disagreeing_image_sizes(
    image_sizes: object,
) -> None:
    intrinsics = np.stack((_intrinsic(800, 820, 320, 240),) * 3)

    with pytest.raises(ValueError, match="image_sizes"):
        robust_shared_pinhole(intrinsics, image_sizes)


def test_robust_shared_pinhole_rejects_severe_focal_disagreement() -> None:
    intrinsics = np.stack(
        (
            _intrinsic(400.0, 450.0),
            _intrinsic(800.0, 900.0),
            _intrinsic(1600.0, 1800.0),
            _intrinsic(3200.0, 3600.0),
        )
    )

    with pytest.raises(ValueError, match="disagree"):
        robust_shared_pinhole(intrinsics, (1920, 1080))


def _frames(root: Path, count: int) -> tuple[DA3Frame, ...]:
    root.mkdir(parents=True)
    result = []
    for index in range(count):
        path = root / f"image-{index:03d}.png"
        path.write_bytes(f"image {index}".encode())
        result.append(
            DA3Frame(
                image_name=path.name,
                frame_id=f"frame-{index:03d}",
                path=path,
            )
        )
    return tuple(result)


def _prediction(
    count: int,
    *,
    extrinsic_columns: int = 4,
    include_conf: bool = True,
    include_sky: bool = False,
) -> SimpleNamespace:
    depth = np.arange(count * 2 * 3, dtype=np.float64).reshape(count, 2, 3) + 1.0
    extrinsics_4x4 = np.repeat(np.eye(4)[None], count, axis=0)
    extrinsics_4x4[:, 0, 3] = np.arange(count)
    extrinsics = (
        extrinsics_4x4[:, :3, :]
        if extrinsic_columns == 3
        else extrinsics_4x4
    )
    intrinsics = np.repeat(_intrinsic(2.0, 2.5, 1.0, 1.0)[None], count, axis=0)
    attributes: dict[str, object] = {
        "depth": depth,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }
    if include_conf:
        attributes["conf"] = np.full_like(depth, 0.75)
    if include_sky:
        attributes["sky"] = np.zeros_like(depth, dtype=bool)
    return SimpleNamespace(**attributes)


class _RecordingModel:
    def __init__(self, prediction: object) -> None:
        self.prediction = prediction
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def inference(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return self.prediction


def test_run_anchor_inference_uses_exact_official_call_and_publishes_mapping(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 3)
    prediction = _prediction(3, extrinsic_columns=3, include_sky=True)
    model = _RecordingModel(prediction)
    output = tmp_path / "anchor-artifacts"

    result = run_anchor_inference(
        model,
        frames,
        output,
        vram_gb=40.0,
    )

    assert model.calls == [
        (
            (),
            {
                "image": [str(frame.path) for frame in frames],
                "process_res": 504,
                "process_res_method": "upper_bound_resize",
                "use_ray_pose": True,
                "ref_view_strategy": "middle",
            },
        )
    ]
    assert isinstance(result, AnchorInferenceResult)
    assert result.anchor_indices == (0, 1, 2)
    assert tuple(artifact.image_name for artifact in result.artifacts) == tuple(
        frame.image_name for frame in frames
    )
    assert tuple(artifact.frame_id for artifact in result.artifacts) == tuple(
        frame.frame_id for frame in frames
    )
    assert tuple(artifact.depth_path.name for artifact in result.artifacts) == (
        "frame-000--image-000.png.depth.npy",
        "frame-001--image-001.png.depth.npy",
        "frame-002--image-002.png.depth.npy",
    )
    assert all(artifact.confidence_path is not None for artifact in result.artifacts)
    assert all(artifact.sky_path is None for artifact in result.artifacts)
    np.testing.assert_array_equal(
        np.load(result.artifacts[1].depth_path, allow_pickle=False),
        prediction.depth[1].astype(np.float32),
    )
    np.testing.assert_array_equal(
        np.load(result.artifacts[1].confidence_path, allow_pickle=False),
        prediction.conf[1].astype(np.float32),
    )
    assert result.shared_camera == PinholeCamera(
        model="PINHOLE",
        width=3,
        height=2,
        fx=2.0,
        fy=2.5,
        cx=1.0,
        cy=1.0,
    )
    assert result.cameras[2].image_name == frames[2].image_name
    assert result.cameras[2].w2c[0][3] == 2.0
    assert result.metadata_path == output / "metadata.json"
    payload_bytes = result.metadata_path.read_bytes()
    payload = json.loads(payload_bytes)
    assert payload_bytes == (
        json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    assert payload["schema_version"] == 1
    assert payload["stage"] == "da3_anchor"
    assert payload["anchor_indices"] == [0, 1, 2]


def test_run_anchor_inference_selects_locked_120_anchors_in_exact_order(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 130)
    expected_indices = select_anchor_indices(130, 70.0)
    model = _RecordingModel(_prediction(len(expected_indices)))

    result = run_anchor_inference(
        model,
        frames,
        tmp_path / "output",
        vram_gb=70.0,
    )

    assert result.anchor_indices == expected_indices
    assert model.calls[0][1]["image"] == [
        str(frames[index].path) for index in expected_indices
    ]
    assert tuple(artifact.image_name for artifact in result.artifacts) == tuple(
        frames[index].image_name for index in expected_indices
    )


def test_run_anchor_inference_accepts_batched_4x4_prediction_extrinsics(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 2)

    result = run_anchor_inference(
        _RecordingModel(_prediction(2, extrinsic_columns=4)),
        frames,
        tmp_path / "output",
        vram_gb=40,
    )

    assert result.cameras[1].w2c[0][3] == 1.0
    assert result.cameras[1].w2c[3] == (0.0, 0.0, 0.0, 1.0)


def test_run_anchor_inference_does_not_fabricate_optional_prediction_fields(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 1)

    result = run_anchor_inference(
        _RecordingModel(_prediction(1, include_conf=False, include_sky=False)),
        frames,
        tmp_path / "output",
        vram_gb=0,
    )

    assert result.artifacts[0].confidence_path is None
    assert result.artifacts[0].sky_path is None


def _set_value(array: np.ndarray, index: tuple[int, ...], value: float) -> None:
    array[index] = value


@pytest.mark.parametrize(
    "mutate",
    (
        lambda prediction: setattr(prediction, "depth", prediction.depth[:-1]),
        lambda prediction: _set_value(prediction.depth, (0, 0, 0), float("nan")),
        lambda prediction: _set_value(prediction.depth, (0, 0, 0), 0.0),
        lambda prediction: setattr(prediction, "conf", prediction.conf[:, :, :-1]),
        lambda prediction: _set_value(prediction.conf, (0, 0, 0), -0.1),
        lambda prediction: setattr(prediction, "sky", np.zeros((2, 2, 2), dtype=bool)),
        lambda prediction: setattr(
            prediction,
            "sky",
            np.full(prediction.depth.shape, 0.25, dtype=np.float32),
        ),
        lambda prediction: setattr(
            prediction,
            "sky",
            np.full(prediction.depth.shape, float("nan")),
        ),
        lambda prediction: setattr(
            prediction,
            "extrinsics",
            prediction.extrinsics[:-1],
        ),
        lambda prediction: _set_value(prediction.extrinsics, (0, 0, 0), -1.0),
        lambda prediction: setattr(
            prediction,
            "intrinsics",
            prediction.intrinsics[:-1],
        ),
        lambda prediction: _set_value(prediction.intrinsics, (0, 0, 1), 0.1),
        lambda prediction: delattr(prediction, "extrinsics"),
        lambda prediction: delattr(prediction, "intrinsics"),
    ),
)
def test_run_anchor_inference_rejects_malformed_prediction_before_publish(
    tmp_path: Path,
    mutate: object,
) -> None:
    frames = _frames(tmp_path / "input", 2)
    prediction = _prediction(2, include_sky=True)
    mutate(prediction)  # type: ignore[operator]
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="prediction"):
        run_anchor_inference(
            _RecordingModel(prediction),
            frames,
            output,
            vram_gb=40.0,
        )

    assert not output.exists()
    assert not tuple(tmp_path.glob(".output.staging-*"))


def test_run_anchor_inference_rejects_probability_sky_instead_of_boolean_mask(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 1)
    prediction = _prediction(1)
    prediction.sky = np.zeros(prediction.depth.shape, dtype=np.float32)

    with pytest.raises(ValueError, match="prediction sky.*boolean"):
        run_anchor_inference(
            _RecordingModel(prediction),
            frames,
            tmp_path / "output",
            vram_gb=40,
        )


@pytest.mark.parametrize("field", ("depth", "conf"))
def test_run_anchor_inference_rejects_float32_artifact_overflow_before_staging(
    tmp_path: Path,
    field: str,
) -> None:
    frames = _frames(tmp_path / "input", 1)
    prediction = _prediction(1)
    setattr(prediction, field, np.full((1, 2, 3), 1.0e40, dtype=np.float64))

    with pytest.raises(ValueError, match=f"prediction {field}.*finite float32"):
        run_anchor_inference(
            _RecordingModel(prediction),
            frames,
            tmp_path / "output",
            vram_gb=40,
        )

    assert not (tmp_path / "output").exists()
    assert not tuple(tmp_path.glob(".output.staging-*"))


@pytest.mark.parametrize(
    ("image_name", "frame_id"),
    (("../image.png", "frame-0"), ("image.png", "../frame-0"), ("image.png", "a/b")),
)
def test_run_anchor_inference_rejects_path_traversal_names_before_model_call(
    tmp_path: Path,
    image_name: str,
    frame_id: str,
) -> None:
    safe = _frames(tmp_path / "input", 1)[0]
    frame = DA3Frame(image_name=image_name, frame_id=frame_id, path=safe.path)
    model = _RecordingModel(_prediction(1))

    with pytest.raises(ValueError, match="frame"):
        run_anchor_inference(model, (frame,), tmp_path / "output", vram_gb=40)

    assert model.calls == []


def test_run_anchor_inference_refuses_existing_final_directory_before_model_call(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 1)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "accepted.txt"
    marker.write_text("accepted", encoding="utf-8")
    model = _RecordingModel(_prediction(1))

    with pytest.raises(FileExistsError, match="output"):
        run_anchor_inference(model, frames, output, vram_gb=40)

    assert marker.read_text(encoding="utf-8") == "accepted"
    assert model.calls == []


def test_run_anchor_inference_promotion_race_cannot_partially_replace_final(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 1)
    output = tmp_path / "output"

    def lose_race(staging: Path, target: Path) -> None:
        assert staging.parent == target.parent
        target.mkdir()
        (target / "accepted.txt").write_text("accepted", encoding="utf-8")
        raise FileExistsError("promotion target already exists")

    with pytest.raises(FileExistsError, match="promotion"):
        run_anchor_inference(
            _RecordingModel(_prediction(1)),
            frames,
            output,
            vram_gb=40,
            promote=lose_race,
        )

    assert tuple(path.name for path in output.iterdir()) == ("accepted.txt",)
    assert (output / "accepted.txt").read_text(encoding="utf-8") == "accepted"
    assert len(tuple(tmp_path.glob(".output.staging-*"))) == 1


def test_run_anchor_inference_is_byte_deterministic_for_identical_prediction(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 2)
    prediction = _prediction(2, include_sky=True)

    first = run_anchor_inference(
        _RecordingModel(prediction),
        frames,
        tmp_path / "first",
        vram_gb=40,
    )
    second = run_anchor_inference(
        _RecordingModel(prediction),
        frames,
        tmp_path / "second",
        vram_gb=40,
    )

    assert first.metadata_path.read_bytes() == second.metadata_path.read_bytes()
    assert tuple(artifact.depth_path.read_bytes() for artifact in first.artifacts) == tuple(
        artifact.depth_path.read_bytes() for artifact in second.artifacts
    )
    assert tuple(
        artifact.confidence_path.read_bytes() for artifact in first.artifacts
    ) == tuple(artifact.confidence_path.read_bytes() for artifact in second.artifacts)


class _FakeCudaOom(RuntimeError):
    pass


class _FakeTorch:
    OutOfMemoryError = _FakeCudaOom

    class cuda:
        OutOfMemoryError = _FakeCudaOom


class _FailingModel:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.call_count = 0

    def inference(self, **kwargs: object) -> object:
        self.call_count += 1
        raise self.error


def test_run_anchor_inference_wraps_only_cuda_oom_without_retry(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 3)
    model = _FailingModel(_FakeCudaOom("CUDA out of memory"))

    with pytest.raises(AnchorStageFailure) as raised:
        run_anchor_inference(
            model,
            frames,
            tmp_path / "output",
            vram_gb=40,
            torch_module=_FakeTorch,
        )

    assert raised.value.locked_anchor_count == 3
    assert raised.value.process_resolution == ANCHOR_PROCESS_RESOLUTION == 504
    assert isinstance(raised.value.__cause__, _FakeCudaOom)
    assert model.call_count == 1
    assert not (tmp_path / "output").exists()


def test_run_anchor_inference_propagates_non_oom_error_unchanged(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 1)
    error = RuntimeError("decoder failed")
    model = _FailingModel(error)

    with pytest.raises(RuntimeError) as raised:
        run_anchor_inference(
            model,
            frames,
            tmp_path / "output",
            vram_gb=40,
            torch_module=_FakeTorch,
        )

    assert raised.value is error
    assert model.call_count == 1


def _processed_camera() -> PinholeCamera:
    return PinholeCamera(
        model="PINHOLE",
        width=3,
        height=2,
        fx=600.0,
        fy=900.0,
        cx=1.0,
        cy=1.0,
    )


def _metric_prediction(
    count: int,
    *,
    network_depth: float = 2.0,
    confidence: float | None = None,
    include_sky: bool = True,
) -> SimpleNamespace:
    attributes: dict[str, object] = {
        "depth": np.full((count, 2, 3), network_depth, dtype=np.float64),
    }
    if confidence is not None:
        attributes["conf"] = np.full((count, 2, 3), confidence, dtype=np.float64)
    if include_sky:
        sky = np.zeros((count, 2, 3), dtype=bool)
        sky[:, 0, 0] = True
        attributes["sky"] = sky
    return SimpleNamespace(**attributes)


class _MetricModel:
    def __init__(
        self,
        *,
        error: BaseException | None = None,
        confidence: float | None = None,
        include_sky: bool = True,
    ) -> None:
        self.error = error
        self.confidence = confidence
        self.include_sky = include_sky
        self.calls: list[dict[str, object]] = []

    def inference(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        image = kwargs["image"]
        return _metric_prediction(
            len(image),  # type: ignore[arg-type]
            confidence=self.confidence,
            include_sky=self.include_sky,
        )


def _vram_release_record() -> VramReleaseRecord:
    return VramReleaseRecord(
        cuda_available=True,
        allocated_before_bytes=10,
        reserved_before_bytes=20,
        allocated_after_bytes=0,
        reserved_after_bytes=0,
        moved_to_cpu=True,
        gc_ran=True,
        cache_cleared=True,
    )


def test_run_metric_sky_uses_exact_batches_and_converts_network_depth_to_meters(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 5)
    model = _MetricModel(confidence=2.5)
    output = tmp_path / "metric"
    release_calls: list[object] = []
    factory_calls: list[str] = []

    result = run_metric_sky(
        model,
        frames,
        _processed_camera(),
        output,
        initial_batch_size=3,
        retry_batch_size=2,
        release_model=lambda failed: release_calls.append(failed) or _vram_release_record(),
        retry_model_factory=lambda: factory_calls.append("factory") or _MetricModel(),
        torch_module=_FakeTorch,
    )

    assert isinstance(result, MetricSkyResult)
    assert model.calls == [
        {
            "image": [str(frame.path) for frame in frames[:3]],
            "process_res": 504,
            "process_res_method": "upper_bound_resize",
        },
        {
            "image": [str(frame.path) for frame in frames[3:]],
            "process_res": 504,
            "process_res_method": "upper_bound_resize",
        },
    ]
    assert release_calls == []
    assert factory_calls == []
    assert tuple((chunk.start, chunk.stop, chunk.requested_size) for chunk in result.chunks) == (
        (0, 3, 3),
        (3, 5, 3),
    )
    assert result.processed_camera == _processed_camera()
    assert result.attempts[0].size == 3
    assert result.attempts[0].outcome == "succeeded"
    assert result.release_record is None
    assert result.retry_model_release_record is None
    assert tuple(artifact.depth_path.name for artifact in result.artifacts) == (
        "frame-000--image-000.png.metric_depth.npy",
        "frame-001--image-001.png.metric_depth.npy",
        "frame-002--image-002.png.metric_depth.npy",
        "frame-003--image-003.png.metric_depth.npy",
        "frame-004--image-004.png.metric_depth.npy",
    )
    assert all(artifact.sky_path is not None for artifact in result.artifacts)
    assert all(artifact.confidence_path is not None for artifact in result.artifacts)
    np.testing.assert_array_equal(
        np.load(result.artifacts[0].depth_path, allow_pickle=False),
        np.full((2, 3), 5.0, dtype=np.float32),
    )
    assert np.load(result.artifacts[0].sky_path, allow_pickle=False).dtype == np.bool_
    np.testing.assert_array_equal(
        np.load(result.artifacts[0].confidence_path, allow_pickle=False),
        np.full((2, 3), 2.5, dtype=np.float32),
    )
    metadata = json.loads(result.metadata_path.read_bytes())
    assert metadata["depth"] == {
        "conversion": "mean_processed_focal_times_network_depth_div_300",
        "processed_height": 2,
        "processed_width": 3,
        "units": "meters",
    }
    assert metadata["processed_camera"] == {
        "cx": 1.0,
        "cy": 1.0,
        "fx": 600.0,
        "fy": 900.0,
        "height": 2,
        "model": "PINHOLE",
        "width": 3,
    }


def test_run_metric_sky_releases_and_recreates_model_for_one_cuda_oom_retry(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 5)
    failed_model = _MetricModel(error=_FakeCudaOom("CUDA out of memory"))
    retry_model = _MetricModel()
    events: list[object] = []

    def release_model(model: object) -> VramReleaseRecord:
        events.append(("release", model))
        return _vram_release_record()

    def retry_model_factory() -> object:
        events.append("factory")
        return retry_model

    result = run_metric_sky(
        failed_model,
        frames,
        _processed_camera(),
        tmp_path / "metric",
        initial_batch_size=3,
        retry_batch_size=2,
        release_model=release_model,
        retry_model_factory=retry_model_factory,
        torch_module=_FakeTorch,
    )

    assert len(failed_model.calls) == 1
    assert [len(call["image"]) for call in retry_model.calls] == [2, 2, 1]
    assert events == [
        ("release", failed_model),
        "factory",
        ("release", retry_model),
    ]
    assert tuple((attempt.size, attempt.outcome) for attempt in result.attempts) == (
        (3, "cuda_oom"),
        (2, "succeeded"),
    )
    assert result.release_record == _vram_release_record()
    assert result.retry_model_release_record == _vram_release_record()
    assert tuple(chunk.requested_size for chunk in result.chunks) == (2, 2, 2)
    metadata = json.loads(result.metadata_path.read_bytes())
    assert metadata["retry_model_release_record"] == {
        "allocated_after_bytes": 0,
        "allocated_before_bytes": 10,
        "cache_cleared": True,
        "cuda_available": True,
        "gc_ran": True,
        "moved_to_cpu": True,
        "reserved_after_bytes": 0,
        "reserved_before_bytes": 20,
    }


def test_run_metric_sky_propagates_non_oom_without_release_or_recreation(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 1)
    error = RuntimeError("metric decoder failed")
    model = _MetricModel(error=error)
    release_calls: list[object] = []
    factory_calls: list[str] = []

    with pytest.raises(RuntimeError) as raised:
        run_metric_sky(
            model,
            frames,
            _processed_camera(),
            tmp_path / "metric",
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=lambda failed: release_calls.append(failed) or _vram_release_record(),
            retry_model_factory=lambda: factory_calls.append("factory") or _MetricModel(),
            torch_module=_FakeTorch,
        )

    assert raised.value is error
    assert release_calls == []
    assert factory_calls == []
    assert not (tmp_path / "metric").exists()


def test_run_metric_sky_performs_at_most_one_smaller_batch_retry(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 2)
    first = _MetricModel(error=_FakeCudaOom("first OOM"))
    second_error = _FakeCudaOom("retry OOM")
    second = _MetricModel(error=second_error)
    released: list[object] = []

    def release_model(model: object) -> VramReleaseRecord:
        released.append(model)
        return _vram_release_record()

    with pytest.raises(_FakeCudaOom) as raised:
        run_metric_sky(
            first,
            frames,
            _processed_camera(),
            tmp_path / "metric",
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=release_model,
            retry_model_factory=lambda: second,
            torch_module=_FakeTorch,
        )

    assert raised.value is second_error
    assert len(first.calls) == 1
    assert len(second.calls) == 1
    assert released == [first, second]
    assert tuple(
        (attempt.size, attempt.outcome)
        for attempt in raised.value.batch_retry_attempts
    ) == ((2, "cuda_oom"), (1, "cuda_oom"))
    assert raised.value.metric_retry_model_release_record == _vram_release_record()


def test_run_metric_sky_releases_replacement_after_ordinary_retry_failure(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 2)
    first = _MetricModel(error=_FakeCudaOom("first OOM"))
    retry_error = ValueError("retry output failed")
    second = _MetricModel(error=retry_error)
    released: list[object] = []

    def release_model(model: object) -> VramReleaseRecord:
        released.append(model)
        return _vram_release_record()

    with pytest.raises(ValueError) as raised:
        run_metric_sky(
            first,
            frames,
            _processed_camera(),
            tmp_path / "metric",
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=release_model,
            retry_model_factory=lambda: second,
            torch_module=_FakeTorch,
        )

    assert raised.value is retry_error
    assert released == [first, second]
    assert raised.value.metric_retry_model_release_record == _vram_release_record()
    assert not (tmp_path / "metric").exists()


def test_run_metric_sky_cleanup_failure_never_masks_retry_failure(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 2)
    first = _MetricModel(error=_FakeCudaOom("first OOM"))
    retry_error = ValueError("primary retry failure")
    cleanup_error = RuntimeError("replacement cleanup failed")
    second = _MetricModel(error=retry_error)
    released: list[object] = []

    def release_model(model: object) -> VramReleaseRecord:
        released.append(model)
        if model is second:
            raise cleanup_error
        return _vram_release_record()

    with pytest.raises(ValueError) as raised:
        run_metric_sky(
            first,
            frames,
            _processed_camera(),
            tmp_path / "metric",
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=release_model,
            retry_model_factory=lambda: second,
            torch_module=_FakeTorch,
        )

    assert raised.value is retry_error
    assert released == [first, second]
    assert raised.value.metric_retry_model_release_error is cleanup_error


def test_run_metric_sky_rejects_factory_reusing_released_initial_model(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 2)
    initial_error = _FakeCudaOom("first OOM")
    model = _MetricModel(error=initial_error)
    released: list[object] = []

    def release_model(value: object) -> VramReleaseRecord:
        released.append(value)
        return _vram_release_record()

    with pytest.raises(ValueError, match="new model object") as raised:
        run_metric_sky(
            model,
            frames,
            _processed_camera(),
            tmp_path / "metric",
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=release_model,
            retry_model_factory=lambda: model,
            torch_module=_FakeTorch,
        )

    assert len(model.calls) == 1
    assert released == [model]
    assert raised.value.__cause__ is initial_error
    assert raised.value.metric_initial_model_release_record == _vram_release_record()


@pytest.mark.parametrize(
    "prediction",
    (
        _metric_prediction(1, include_sky=False),
        SimpleNamespace(
            depth=np.ones((1, 2, 3)),
            sky=np.zeros((1, 2, 3), dtype=np.float32),
        ),
        SimpleNamespace(
            depth=np.ones((1, 3, 3)),
            sky=np.zeros((1, 3, 3), dtype=bool),
        ),
    ),
)
def test_run_metric_sky_rejects_missing_nonboolean_or_wrong_processed_shape(
    tmp_path: Path,
    prediction: object,
) -> None:
    frames = _frames(tmp_path / "input", 1)

    with pytest.raises(ValueError, match="metric prediction|processed"):
        run_metric_sky(
            _RecordingModel(prediction),
            frames,
            _processed_camera(),
            tmp_path / "metric",
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=lambda failed: _vram_release_record(),
            retry_model_factory=lambda: _RecordingModel(prediction),
            torch_module=_FakeTorch,
        )

    assert not (tmp_path / "metric").exists()


def test_run_metric_sky_requires_physical_processed_pinhole_before_model_call(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 1)
    model = _MetricModel()
    invalid = PinholeCamera(
        model="PINHOLE",
        width=3,
        height=2,
        fx=0.0,
        fy=900.0,
        cx=1.0,
        cy=1.0,
    )

    with pytest.raises(ValueError, match="processed_camera"):
        run_metric_sky(
            model,
            frames,
            invalid,
            tmp_path / "metric",
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=lambda failed: _vram_release_record(),
            retry_model_factory=lambda: _MetricModel(),
            torch_module=_FakeTorch,
        )

    assert model.calls == []


def test_run_metric_sky_rejects_focal_scaling_overflow_before_staging(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 1)
    prediction = _metric_prediction(1, network_depth=1.0e38)
    camera = PinholeCamera(
        model="PINHOLE",
        width=3,
        height=2,
        fx=1.0e9,
        fy=1.0e9,
        cx=1.0,
        cy=1.0,
    )

    with pytest.raises(ValueError, match="metric depth.*finite float32"):
        run_metric_sky(
            _RecordingModel(prediction),
            frames,
            camera,
            tmp_path / "metric",
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=lambda failed: _vram_release_record(),
            retry_model_factory=lambda: _RecordingModel(prediction),
            torch_module=_FakeTorch,
        )

    assert not (tmp_path / "metric").exists()
    assert not tuple(tmp_path.glob(".metric.staging-*"))


def _pose_cameras(frames: tuple[DA3Frame, ...]) -> tuple[FinalPoseCamera, ...]:
    result = []
    for index, frame in enumerate(frames):
        w2c = np.eye(4)
        w2c[0, 3] = index
        result.append(
            FinalPoseCamera(
                image_name=frame.image_name,
                frame_id=frame.frame_id,
                width=3,
                height=2,
                w2c=tuple(tuple(float(value) for value in row) for row in w2c),
                intrinsics=(
                    (2.0, 0.0, 1.0),
                    (0.0, 2.5, 1.0),
                    (0.0, 0.0, 1.0),
                ),
            )
        )
    return tuple(result)


class _FinalPoseModel:
    def __init__(
        self,
        *,
        error_call: int | None = None,
        error: BaseException | None = None,
        include_confidence: bool = True,
        bad_extrinsics: bool = False,
        bad_intrinsics: bool = False,
        float32_camera_roundtrip: bool = False,
        extrinsic_drift: float = 0.0,
    ) -> None:
        self.error_call = error_call
        self.error = error
        self.include_confidence = include_confidence
        self.bad_extrinsics = bad_extrinsics
        self.bad_intrinsics = bad_intrinsics
        self.float32_camera_roundtrip = float32_camera_roundtrip
        self.extrinsic_drift = extrinsic_drift
        self.calls: list[dict[str, object]] = []

    def inference(self, **kwargs: object) -> object:
        call_index = len(self.calls)
        self.calls.append(kwargs)
        if self.error_call == call_index and self.error is not None:
            raise self.error
        count = len(kwargs["image"])  # type: ignore[arg-type]
        depth_value = float(10 * (call_index + 1))
        attributes: dict[str, object] = {
            "depth": np.full((count, 2, 3), depth_value, dtype=np.float64),
            "extrinsics": np.array(
                kwargs["extrinsics"],
                dtype=(np.float32 if self.float32_camera_roundtrip else np.float64),
                copy=True,
            ),
            "intrinsics": np.array(
                kwargs["intrinsics"],
                dtype=(np.float32 if self.float32_camera_roundtrip else np.float64),
                copy=True,
            ),
        }
        if self.include_confidence:
            attributes["conf"] = np.full(
                (count, 2, 3),
                float(2 * call_index + 1),
                dtype=np.float64,
            )
        if self.bad_extrinsics:
            attributes["extrinsics"][0, 0, 3] += 1.0  # type: ignore[index]
        if self.extrinsic_drift:
            attributes["extrinsics"][0, 0, 3] += self.extrinsic_drift  # type: ignore[index]
        if self.bad_intrinsics:
            attributes["intrinsics"][0, 0, 1] = 0.1  # type: ignore[index]
        return SimpleNamespace(**attributes)


def test_run_pose_conditioned_depth_uses_exact_official_conditioning_call(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 3)
    cameras = _pose_cameras(frames)
    model = _FinalPoseModel()

    result = run_pose_conditioned_depth(
        model,
        frames,
        cameras,
        tmp_path / "final-depth",
        torch_module=_FakeTorch,
    )

    assert isinstance(result, PoseConditionedDepthResult)
    assert len(model.calls) == 1
    call = model.calls[0]
    assert set(call) == {
        "image",
        "extrinsics",
        "intrinsics",
        "align_to_input_ext_scale",
        "process_res",
        "process_res_method",
    }
    assert call["image"] == [str(frame.path) for frame in frames]
    assert call["align_to_input_ext_scale"] is True
    assert call["process_res"] == 504
    assert call["process_res_method"] == "upper_bound_resize"
    np.testing.assert_array_equal(
        call["extrinsics"],
        np.asarray([camera.w2c for camera in cameras]),
    )
    np.testing.assert_array_equal(
        call["intrinsics"],
        np.asarray([camera.intrinsics for camera in cameras]),
    )
    assert tuple(camera.image_name for camera in result.cameras) == tuple(
        frame.image_name for frame in frames
    )
    assert result.processed_camera == PinholeCamera(
        model="PINHOLE",
        width=3,
        height=2,
        fx=2.0,
        fy=2.5,
        cx=1.0,
        cy=1.0,
    )
    assert result.contributions[0].chunk_indices == (0,)
    assert result.artifacts[0].depth_path.name.endswith(".final_depth.npy")
    assert result.artifacts[0].confidence_path is not None
    assert result.artifacts[0].sky_path is None


def test_run_pose_conditioned_depth_accepts_faithful_float32_camera_roundtrip(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 1)
    camera = _pose_cameras(frames)[0]
    w2c = np.asarray(camera.w2c).copy()
    w2c[0, 3] = 1234.56789
    camera = FinalPoseCamera(
        image_name=camera.image_name,
        frame_id=camera.frame_id,
        width=camera.width,
        height=camera.height,
        w2c=tuple(tuple(float(value) for value in row) for row in w2c),
        intrinsics=camera.intrinsics,
    )

    result = run_pose_conditioned_depth(
        _FinalPoseModel(float32_camera_roundtrip=True),
        frames,
        (camera,),
        tmp_path / "final-depth",
        torch_module=_FakeTorch,
    )

    assert result.cameras[0].w2c[0][3] == 1234.56789


def test_run_pose_conditioned_depth_rejects_drift_after_float32_roundtrip(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 1)
    camera = _pose_cameras(frames)[0]
    w2c = np.asarray(camera.w2c).copy()
    w2c[0, 3] = 1234.56789
    camera = FinalPoseCamera(
        image_name=camera.image_name,
        frame_id=camera.frame_id,
        width=camera.width,
        height=camera.height,
        w2c=tuple(tuple(float(value) for value in row) for row in w2c),
        intrinsics=camera.intrinsics,
    )

    with pytest.raises(ValueError, match="extrinsics drifted"):
        run_pose_conditioned_depth(
            _FinalPoseModel(
                float32_camera_roundtrip=True,
                extrinsic_drift=0.01,
            ),
            frames,
            (camera,),
            tmp_path / "final-depth",
            torch_module=_FakeTorch,
        )


def test_run_pose_conditioned_depth_fuses_differing_overlap_by_confidence(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 73)
    model = _FinalPoseModel()

    result = run_pose_conditioned_depth(
        model,
        frames,
        _pose_cameras(frames),
        tmp_path / "final-depth",
        torch_module=_FakeTorch,
    )

    assert [len(call["image"]) for call in model.calls] == [48, 48, 48]
    assert model.calls[0]["image"] == [str(frame.path) for frame in frames[0:48]]
    assert model.calls[1]["image"] == [str(frame.path) for frame in frames[24:72]]
    assert model.calls[2]["image"] == [str(frame.path) for frame in frames[25:73]]
    assert tuple((chunk.start, chunk.stop) for chunk in result.chunks) == (
        (0, 48),
        (24, 72),
        (25, 73),
    )
    assert result.contributions[24].chunk_indices == (0, 1)
    assert result.contributions[25].chunk_indices == (0, 1, 2)
    np.testing.assert_allclose(
        np.load(result.artifacts[24].depth_path, allow_pickle=False),
        np.full((2, 3), 17.5, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.load(result.artifacts[24].confidence_path, allow_pickle=False),
        np.full((2, 3), 3.0, dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.load(result.artifacts[25].depth_path, allow_pickle=False),
        np.full((2, 3), 220.0 / 9.0, dtype=np.float32),
        rtol=1e-6,
    )
    metadata = json.loads(result.metadata_path.read_bytes())
    assert metadata["fusion"] == {
        "confidence": "maximum_contributing_confidence",
        "depth": "confidence_weighted_mean_in_final_colmap_scale",
    }
    assert metadata["contributions"][24]["chunk_indices"] == [0, 1]


def test_run_pose_conditioned_depth_uses_unweighted_overlap_without_confidence(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 49)

    result = run_pose_conditioned_depth(
        _FinalPoseModel(include_confidence=False),
        frames,
        _pose_cameras(frames),
        tmp_path / "final-depth",
        torch_module=_FakeTorch,
    )

    assert result.artifacts[1].confidence_path is None
    np.testing.assert_array_equal(
        np.load(result.artifacts[1].depth_path, allow_pickle=False),
        np.full((2, 3), 15.0, dtype=np.float32),
    )


def test_run_pose_conditioned_depth_aborts_on_cuda_oom_without_smaller_context(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 73)
    model = _FinalPoseModel(
        error_call=1,
        error=_FakeCudaOom("CUDA out of memory"),
    )

    with pytest.raises(FinalPoseStageFailure) as raised:
        run_pose_conditioned_depth(
            model,
            frames,
            _pose_cameras(frames),
            tmp_path / "final-depth",
            torch_module=_FakeTorch,
        )

    assert raised.value.chunk_index == 1
    assert raised.value.chunk_start == 24
    assert raised.value.chunk_stop == 72
    assert raised.value.actual_chunk_size == 48
    assert raised.value.locked_context_size == 48
    assert raised.value.process_resolution == 504
    assert isinstance(raised.value.__cause__, _FakeCudaOom)
    assert len(model.calls) == 2
    assert not (tmp_path / "final-depth").exists()


def test_run_pose_conditioned_depth_propagates_non_oom_unchanged(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 2)
    error = RuntimeError("base decoder failed")
    model = _FinalPoseModel(error_call=0, error=error)

    with pytest.raises(RuntimeError) as raised:
        run_pose_conditioned_depth(
            model,
            frames,
            _pose_cameras(frames),
            tmp_path / "final-depth",
            torch_module=_FakeTorch,
        )

    assert raised.value is error
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "model",
    (
        _FinalPoseModel(bad_extrinsics=True),
        _FinalPoseModel(bad_intrinsics=True),
    ),
)
def test_run_pose_conditioned_depth_rejects_camera_drift_before_publish(
    tmp_path: Path,
    model: _FinalPoseModel,
) -> None:
    frames = _frames(tmp_path / "input", 2)

    with pytest.raises(ValueError, match="final prediction"):
        run_pose_conditioned_depth(
            model,
            frames,
            _pose_cameras(frames),
            tmp_path / "final-depth",
            torch_module=_FakeTorch,
        )

    assert not (tmp_path / "final-depth").exists()


def test_run_pose_conditioned_depth_rejects_mismatched_camera_mapping_before_call(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 2)
    cameras = list(_pose_cameras(frames))
    cameras[1] = FinalPoseCamera(
        image_name="wrong.png",
        frame_id=cameras[1].frame_id,
        width=cameras[1].width,
        height=cameras[1].height,
        w2c=cameras[1].w2c,
        intrinsics=cameras[1].intrinsics,
    )
    model = _FinalPoseModel()

    with pytest.raises(ValueError, match="camera.*mapping"):
        run_pose_conditioned_depth(
            model,
            frames,
            tuple(cameras),
            tmp_path / "final-depth",
            torch_module=_FakeTorch,
        )

    assert model.calls == []


def test_run_pose_conditioned_depth_promotion_failure_keeps_final_private(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 1)
    output = tmp_path / "final-depth"

    def lose_race(staging: Path, target: Path) -> None:
        target.mkdir()
        (target / "accepted.txt").write_text("accepted", encoding="utf-8")
        raise FileExistsError("promotion target already exists")

    with pytest.raises(FileExistsError, match="promotion"):
        run_pose_conditioned_depth(
            _FinalPoseModel(),
            frames,
            _pose_cameras(frames),
            output,
            torch_module=_FakeTorch,
            promote=lose_race,
        )

    assert tuple(path.name for path in output.iterdir()) == ("accepted.txt",)
    assert len(tuple(tmp_path.glob(".final-depth.staging-*"))) == 1


def test_run_pose_conditioned_depth_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    frames = _frames(tmp_path / "input", 49)
    cameras = _pose_cameras(frames)

    first = run_pose_conditioned_depth(
        _FinalPoseModel(),
        frames,
        cameras,
        tmp_path / "first",
        torch_module=_FakeTorch,
    )
    second = run_pose_conditioned_depth(
        _FinalPoseModel(),
        frames,
        cameras,
        tmp_path / "second",
        torch_module=_FakeTorch,
    )

    assert first.metadata_path.read_bytes() == second.metadata_path.read_bytes()
    assert tuple(artifact.depth_path.read_bytes() for artifact in first.artifacts) == tuple(
        artifact.depth_path.read_bytes() for artifact in second.artifacts
    )


def test_da3_module_imports_without_torch_da3_or_exporter_modules() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    script = "\n".join(
        (
            "import json",
            "import sys",
            f"sys.path.insert(0, {str(repository_root)!r})",
            "import experiments.learned_quality.da3",
            "blocked = sorted(name for name in sys.modules if "
            "name == 'torch' or name.startswith('torch.') or "
            "name == 'depth_anything_3' or name.startswith('depth_anything_3.'))",
            "print(json.dumps(blocked, separators=(',', ':')))",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_da3_adapter_contains_no_official_colmap_or_gaussian_exporter_call() -> None:
    source = Path(da3_module.__file__).read_text(encoding="utf-8")

    assert "export_dir" not in source
    assert "export_format" not in source
    assert "infer_gs" not in source

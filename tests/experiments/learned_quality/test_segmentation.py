from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import MISSING, FrozenInstanceError, fields, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from experiments.learned_quality.contracts import FrameArtifact
from experiments.learned_quality.lifecycle import VramReleaseRecord
from experiments.learned_quality.segmentation import (
    GROUNDING_DINO_MODEL_REF,
    SAM_MODEL_REF,
    TRANSIENT_PROMPT,
    DetectionBox,
    SamBoxPrompt,
    SemanticEvidence,
    SemanticFrameEvidence,
    SemanticPolicy,
    run_semantic_evidence,
)


EXPECTED_PROMPT = (
    "person. child. hand. dog. cat. animal. bicycle. motorcycle. "
    "scooter. car. truck. bus."
)


def _release_record() -> VramReleaseRecord:
    return VramReleaseRecord(
        cuda_available=False,
        allocated_before_bytes=None,
        reserved_before_bytes=None,
        allocated_after_bytes=None,
        reserved_after_bytes=None,
        moved_to_cpu=True,
        gc_ran=True,
        cache_cleared=False,
    )


def _write_frame(
    root: Path,
    index: int,
    *,
    width: int = 8,
    height: int = 6,
) -> FrameArtifact:
    path = root / f"frame_{index:06d}.png"
    pixels = np.full((height, width, 3), 20 + index, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")
    return FrameArtifact(
        image_name=path.name,
        frame_id=f"frame-{index:06d}",
        path=path,
        sha256=sha256(path.read_bytes()).hexdigest(),
    )


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


class FakeDetector:
    model_ref = GROUNDING_DINO_MODEL_REF

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str, float, float]] = []

    def detect_batch(
        self,
        frames: tuple[FrameArtifact, ...],
        *,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Mapping[str, tuple[DetectionBox, ...]]:
        self.calls.append(
            (
                tuple(frame.frame_id for frame in frames),
                prompt,
                box_threshold,
                text_threshold,
            )
        )
        result: dict[str, tuple[DetectionBox, ...]] = {
            frame.frame_id: () for frame in frames
        }
        for frame in frames:
            if frame.frame_id == "frame-000000":
                result[frame.frame_id] = (
                    DetectionBox(
                        box_id="person-0",
                        xyxy=(1.0, 1.0, 4.0, 4.0),
                        score=0.95,
                        phrase="person",
                    ),
                    DetectionBox(
                        box_id="dog-0",
                        xyxy=(4.0, 2.0, 7.0, 5.0),
                        score=0.90,
                        phrase="dog",
                    ),
                )
        return result


class FakeSam:
    model_ref = SAM_MODEL_REF

    def __init__(self, *, width: int = 8, height: int = 6) -> None:
        self.width = width
        self.height = height
        self.refine_calls: list[tuple[tuple[str, str], ...]] = []
        self.propagate_calls: list[tuple[str, ...]] = []

    def refine_batch(
        self,
        prompts: tuple[SamBoxPrompt, ...],
    ) -> Mapping[tuple[str, str], object]:
        keys = tuple((prompt.frame.frame_id, prompt.box.box_id) for prompt in prompts)
        self.refine_calls.append(keys)
        masks: dict[tuple[str, str], object] = {}
        for frame_id, box_id in keys:
            mask = np.zeros((self.height, self.width), dtype=np.bool_)
            if box_id == "person-0":
                mask[1:4, 1:4] = True
            elif box_id == "dog-0":
                mask[2:5, 4:7] = True
            masks[(frame_id, box_id)] = mask
        return masks

    def propagate(
        self,
        frames: tuple[FrameArtifact, ...],
        direct_masks: Mapping[str, object],
        *,
        clip_frames: int,
    ) -> Mapping[str, tuple[object, tuple[str, ...]]]:
        assert clip_frames == 3
        assert tuple(direct_masks) == tuple(frame.frame_id for frame in frames)
        self.propagate_calls.append(tuple(frame.frame_id for frame in frames))
        return {
            frame.frame_id: (
                np.zeros((self.height, self.width), dtype=np.bool_),
                (),
            )
            for frame in frames
        }


class EmptyDetector(FakeDetector):
    def detect_batch(
        self,
        frames: tuple[FrameArtifact, ...],
        *,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Mapping[str, tuple[DetectionBox, ...]]:
        self.calls.append(
            (
                tuple(frame.frame_id for frame in frames),
                prompt,
                box_threshold,
                text_threshold,
            )
        )
        return {frame.frame_id: () for frame in frames}


class PropagationSam(FakeSam):
    def __init__(self, support_ids: tuple[str, ...]) -> None:
        super().__init__()
        self.support_ids = support_ids

    def propagate(
        self,
        frames: tuple[FrameArtifact, ...],
        direct_masks: Mapping[str, object],
        *,
        clip_frames: int,
    ) -> Mapping[str, tuple[object, tuple[str, ...]]]:
        assert clip_frames == 3
        self.propagate_calls.append(tuple(frame.frame_id for frame in frames))
        result: dict[str, tuple[object, tuple[str, ...]]] = {}
        for frame in frames:
            candidate = np.zeros((self.height, self.width), dtype=np.bool_)
            support_ids: tuple[str, ...] = ()
            if frame.frame_id == "frame-000001":
                candidate[1:3, 2:5] = True
                support_ids = self.support_ids
            result[frame.frame_id] = (candidate, support_ids)
        return result


class FakeCudaOOM(RuntimeError):
    pass


class FailingDetector(FakeDetector):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def detect_batch(
        self,
        frames: tuple[FrameArtifact, ...],
        *,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Mapping[str, tuple[DetectionBox, ...]]:
        self.calls.append(
            (
                tuple(frame.frame_id for frame in frames),
                prompt,
                box_threshold,
                text_threshold,
            )
        )
        raise self.error


class FailingSam(FakeSam):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    def refine_batch(
        self,
        prompts: tuple[SamBoxPrompt, ...],
    ) -> Mapping[tuple[str, str], object]:
        keys = tuple((prompt.frame.frame_id, prompt.box.box_id) for prompt in prompts)
        self.refine_calls.append(keys)
        raise self.error


def _install_fake_typed_cuda_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        OutOfMemoryError=FakeCudaOOM,
        cuda=SimpleNamespace(OutOfMemoryError=FakeCudaOOM),
    )
    monkeypatch.setattr(
        "experiments.learned_quality.lifecycle.importlib.import_module",
        lambda name: fake_torch,
    )


def test_prompt_policy_and_public_records_are_exact_frozen_and_default_free() -> None:
    assert TRANSIENT_PROMPT == EXPECTED_PROMPT
    assert TRANSIENT_PROMPT.encode("utf-8") == EXPECTED_PROMPT.encode("utf-8")
    for record_type in (
        SemanticPolicy,
        DetectionBox,
        SamBoxPrompt,
        SemanticFrameEvidence,
        SemanticEvidence,
    ):
        assert record_type.__dataclass_params__.frozen is True
        assert all(
            item.default is MISSING and item.default_factory is MISSING
            for item in fields(record_type)
        )

    policy = SemanticPolicy(0.25, 0.20, 3, 1)
    with pytest.raises(FrozenInstanceError):
        policy.dino_box_threshold = 0.30  # type: ignore[misc]


def test_direct_detection_refines_every_box_and_publishes_canonical_maps(
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))
    detector = FakeDetector()
    sam = FakeSam()
    releases: list[object] = []

    def release_model(model: object) -> VramReleaseRecord:
        releases.append(model)
        return _release_record()

    result = run_semantic_evidence(
        frames,
        tmp_path / "semantic",
        policy=SemanticPolicy(0.25, 0.20, 3, 1),
        detector_factory=lambda: detector,
        sam_factory=lambda: sam,
        initial_batch_size=2,
        retry_batch_size=1,
        release_model=release_model,
    )

    assert isinstance(result, SemanticEvidence)
    assert result.prompt == EXPECTED_PROMPT
    assert tuple(item.frame for item in result.frames) == frames
    assert detector.calls == [
        (
            ("frame-000000", "frame-000001"),
            EXPECTED_PROMPT,
            0.25,
            0.20,
        )
    ]
    assert sam.refine_calls == [
        (("frame-000000", "person-0"), ("frame-000000", "dog-0"))
    ]
    assert sam.propagate_calls == [("frame-000000", "frame-000001")]
    assert releases == [detector, sam]

    first, second = result.frames
    expected_direct = np.zeros((6, 8), dtype=np.uint8)
    expected_direct[1:4, 1:4] = 255
    expected_direct[2:5, 4:7] = 255
    np.testing.assert_array_equal(_read_mask(first.direct_confirmed_path), expected_direct)
    np.testing.assert_array_equal(
        _read_mask(second.direct_confirmed_path),
        np.zeros((6, 8), dtype=np.uint8),
    )
    for item in result.frames:
        np.testing.assert_array_equal(
            _read_mask(item.propagated_candidate_path),
            np.zeros((6, 8), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            _read_mask(item.direct_support_path),
            np.zeros((6, 8), dtype=np.uint8),
        )
        assert item.width == 8
        assert item.height == 6
        for path, digest in (
            (item.direct_confirmed_path, item.direct_confirmed_sha256),
            (item.propagated_candidate_path, item.propagated_candidate_sha256),
            (item.direct_support_path, item.direct_support_sha256),
        ):
            assert path.is_absolute()
            assert path.is_file()
            assert sha256(path.read_bytes()).hexdigest() == digest

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["prompt"] == EXPECTED_PROMPT
    assert manifest["policy"] == {
        "dino_box_threshold": 0.25,
        "dino_text_threshold": 0.20,
        "propagation_clip_frames": 3,
        "propagation_neighbor_frames": 1,
    }
    assert manifest["models"] == {
        "grounding_dino": {
            "code_commit": GROUNDING_DINO_MODEL_REF.code_commit,
            "license_id": GROUNDING_DINO_MODEL_REF.license_id,
            "repo_id": GROUNDING_DINO_MODEL_REF.repo_id,
            "revision": GROUNDING_DINO_MODEL_REF.revision,
        },
        "sam": {
            "code_commit": SAM_MODEL_REF.code_commit,
            "license_id": SAM_MODEL_REF.license_id,
            "repo_id": SAM_MODEL_REF.repo_id,
            "revision": SAM_MODEL_REF.revision,
        },
    }
    assert tuple(record.stage_id for record in result.stage_records) == (
        "semantic_dino",
        "semantic_sam",
    )
    assert result.manifest_path.is_file()
    assert sha256(result.manifest_path.read_bytes()).hexdigest() == result.manifest_sha256


def test_empty_detections_still_emit_three_zero_maps_per_frame(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))
    detector = EmptyDetector()
    sam = FakeSam()
    releases: list[object] = []

    result = run_semantic_evidence(
        frames,
        tmp_path / "semantic",
        policy=SemanticPolicy(0.25, 0.20, 3, 1),
        detector_factory=lambda: detector,
        sam_factory=lambda: sam,
        initial_batch_size=2,
        retry_batch_size=1,
        release_model=lambda model: releases.append(model) or _release_record(),
    )

    assert detector.calls == [
        (("frame-000000", "frame-000001"), EXPECTED_PROMPT, 0.25, 0.20)
    ]
    assert sam.refine_calls == []
    assert sam.propagate_calls == [("frame-000000", "frame-000001")]
    assert releases == [detector, sam]
    for frame in result.frames:
        for path in (
            frame.direct_confirmed_path,
            frame.propagated_candidate_path,
            frame.direct_support_path,
        ):
            np.testing.assert_array_equal(
                _read_mask(path),
                np.zeros((6, 8), dtype=np.uint8),
            )


def test_propagation_remains_candidate_until_nearby_direct_support_exists(
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))
    unsupported = run_semantic_evidence(
        frames,
        tmp_path / "unsupported",
        policy=SemanticPolicy(0.25, 0.20, 3, 1),
        detector_factory=FakeDetector,
        sam_factory=lambda: PropagationSam(()),
        initial_batch_size=2,
        retry_batch_size=1,
        release_model=lambda model: _release_record(),
    )
    supported = run_semantic_evidence(
        frames,
        tmp_path / "supported",
        policy=SemanticPolicy(0.25, 0.20, 3, 1),
        detector_factory=FakeDetector,
        sam_factory=lambda: PropagationSam(("frame-000000",)),
        initial_batch_size=2,
        retry_batch_size=1,
        release_model=lambda model: _release_record(),
    )

    expected_candidate = np.zeros((6, 8), dtype=np.uint8)
    expected_candidate[1:3, 2:5] = 255
    for result in (unsupported, supported):
        np.testing.assert_array_equal(
            _read_mask(result.frames[1].direct_confirmed_path),
            np.zeros((6, 8), dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            _read_mask(result.frames[1].propagated_candidate_path),
            expected_candidate,
        )
    np.testing.assert_array_equal(
        _read_mask(unsupported.frames[1].direct_support_path),
        np.zeros((6, 8), dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        _read_mask(supported.frames[1].direct_support_path),
        expected_candidate,
    )


def test_propagation_cannot_claim_its_own_candidate_as_direct_support(
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))
    detector = FakeDetector()
    sam = PropagationSam(("frame-000001",))
    releases: list[object] = []
    output = tmp_path / "semantic"

    with pytest.raises(ValueError, match="source has no direct evidence"):
        run_semantic_evidence(
            frames,
            output,
            policy=SemanticPolicy(0.25, 0.20, 3, 1),
            detector_factory=lambda: detector,
            sam_factory=lambda: sam,
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=lambda model: releases.append(model) or _release_record(),
        )

    assert releases == [detector, sam]
    assert not output.exists()


@pytest.mark.parametrize("mode", ["missing", "extra", "reordered", "bad_box"])
def test_detector_results_require_exact_ordered_frame_and_box_joins(
    tmp_path: Path,
    mode: str,
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))

    class BadDetector(FakeDetector):
        def detect_batch(
            self,
            batch: tuple[FrameArtifact, ...],
            *,
            prompt: str,
            box_threshold: float,
            text_threshold: float,
        ) -> Mapping[str, tuple[DetectionBox, ...]]:
            result = dict(
                super().detect_batch(
                    batch,
                    prompt=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            )
            if mode == "missing":
                result.pop(batch[-1].frame_id)
            elif mode == "extra":
                result["frame-extra"] = ()
            elif mode == "reordered":
                result = dict(reversed(tuple(result.items())))
            else:
                result[batch[0].frame_id] = (
                    DetectionBox(
                        box_id="below-threshold",
                        xyxy=(1.0, 1.0, 4.0, 4.0),
                        score=0.24,
                        phrase="person",
                    ),
                )
            return result

    detector = BadDetector()
    releases: list[object] = []
    sam_factory_calls: list[bool] = []
    output = tmp_path / "semantic"

    with pytest.raises(ValueError, match="detector"):
        run_semantic_evidence(
            frames,
            output,
            policy=SemanticPolicy(0.25, 0.20, 3, 1),
            detector_factory=lambda: detector,
            sam_factory=lambda: sam_factory_calls.append(True) or FakeSam(),
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=lambda model: releases.append(model) or _release_record(),
        )

    assert releases == [detector]
    assert sam_factory_calls == []
    assert not output.exists()


@pytest.mark.parametrize("mode", ["box_join", "mask_shape", "propagation_order"])
def test_sam_results_require_exact_box_frame_order_and_shapes(
    tmp_path: Path,
    mode: str,
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))

    class BadSam(FakeSam):
        def refine_batch(
            self,
            prompts: tuple[SamBoxPrompt, ...],
        ) -> Mapping[tuple[str, str], object]:
            result = dict(super().refine_batch(prompts))
            if mode == "box_join":
                result.pop(next(reversed(result)))
            elif mode == "mask_shape":
                result[next(iter(result))] = np.zeros((1, 1), dtype=np.bool_)
            return result

        def propagate(
            self,
            selected_frames: tuple[FrameArtifact, ...],
            direct_masks: Mapping[str, object],
            *,
            clip_frames: int,
        ) -> Mapping[str, tuple[object, tuple[str, ...]]]:
            result = dict(
                super().propagate(
                    selected_frames,
                    direct_masks,
                    clip_frames=clip_frames,
                )
            )
            if mode == "propagation_order":
                result = dict(reversed(tuple(result.items())))
            return result

    detector = FakeDetector()
    sam = BadSam()
    releases: list[object] = []
    output = tmp_path / "semantic"

    with pytest.raises(ValueError, match="SAM"):
        run_semantic_evidence(
            frames,
            output,
            policy=SemanticPolicy(0.25, 0.20, 3, 1),
            detector_factory=lambda: detector,
            sam_factory=lambda: sam,
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=lambda model: releases.append(model) or _release_record(),
        )

    assert releases == [detector, sam]
    assert not output.exists()


def test_preflight_rejects_malformed_policy_frames_and_output_paths(
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))
    existing_output = tmp_path / "existing"
    existing_output.mkdir()
    drifted = _write_frame(frames_dir, 2)
    drifted.path.write_bytes(b"digest drift")
    calls: list[str] = []

    cases = (
        (
            frames,
            tmp_path / "bad-policy",
            SemanticPolicy(float("nan"), 0.20, 3, 1),
        ),
        (
            (replace(frames[0], image_name="../frame.png"), frames[1]),
            tmp_path / "unsafe-name",
            SemanticPolicy(0.25, 0.20, 3, 1),
        ),
        (
            (frames[0], replace(frames[1], frame_id=frames[0].frame_id)),
            tmp_path / "duplicate-id",
            SemanticPolicy(0.25, 0.20, 3, 1),
        ),
        (
            (drifted,),
            tmp_path / "digest-drift",
            SemanticPolicy(0.25, 0.20, 3, 1),
        ),
        (frames, Path("relative-output"), SemanticPolicy(0.25, 0.20, 3, 1)),
        (frames, existing_output, SemanticPolicy(0.25, 0.20, 3, 1)),
    )
    for selected_frames, output, policy in cases:
        with pytest.raises(ValueError):
            run_semantic_evidence(
                selected_frames,
                output,
                policy=policy,
                detector_factory=lambda: calls.append("detector") or FakeDetector(),
                sam_factory=lambda: calls.append("sam") or FakeSam(),
                initial_batch_size=2,
                retry_batch_size=1,
                release_model=lambda model: _release_record(),
            )

    assert calls == []


def test_typed_detector_cuda_oom_releases_and_recreates_exact_model_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_typed_cuda_oom(monkeypatch)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))
    failed = FailingDetector(FakeCudaOOM("typed detector OOM"))
    recreated = FakeDetector()
    detector_models = [failed, recreated]
    detector_factory_calls: list[object] = []
    sam = FakeSam()
    releases: list[object] = []

    def detector_factory() -> FakeDetector:
        model = detector_models[len(detector_factory_calls)]
        detector_factory_calls.append(model)
        return model

    result = run_semantic_evidence(
        frames,
        tmp_path / "semantic",
        policy=SemanticPolicy(0.25, 0.20, 3, 1),
        detector_factory=detector_factory,
        sam_factory=lambda: sam,
        initial_batch_size=2,
        retry_batch_size=1,
        release_model=lambda model: releases.append(model) or _release_record(),
    )

    assert detector_factory_calls == [failed, recreated]
    assert releases == [failed, recreated, sam]
    assert [len(call[0]) for call in failed.calls + recreated.calls] == [2, 1, 1]
    assert all(call[1:] == (EXPECTED_PROMPT, 0.25, 0.20) for call in failed.calls)
    assert all(
        call[1:] == (EXPECTED_PROMPT, 0.25, 0.20) for call in recreated.calls
    )
    dino_record = result.stage_records[0]
    assert dino_record.status == "fallback"
    assert set(dino_record.details) == {
        "attempts",
        "model",
        "release",
        "retry_release",
    }
    assert [
        (attempt["size"], attempt["outcome"])
        for attempt in dino_record.details["attempts"]  # type: ignore[index, union-attr]
    ] == [(2, "cuda_oom"), (1, "succeeded")]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage_records"][0]["details"]["attempts"][0][
        "error_type"
    ] == "FakeCudaOOM"


def test_typed_sam_cuda_oom_releases_and_recreates_exact_model_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_typed_cuda_oom(monkeypatch)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))
    detector = FakeDetector()
    failed = FailingSam(FakeCudaOOM("typed SAM OOM"))
    recreated = FakeSam()
    sam_models = [failed, recreated]
    sam_factory_calls: list[object] = []
    releases: list[object] = []

    def sam_factory() -> FakeSam:
        model = sam_models[len(sam_factory_calls)]
        sam_factory_calls.append(model)
        return model

    result = run_semantic_evidence(
        frames,
        tmp_path / "semantic",
        policy=SemanticPolicy(0.25, 0.20, 3, 1),
        detector_factory=lambda: detector,
        sam_factory=sam_factory,
        initial_batch_size=2,
        retry_batch_size=1,
        release_model=lambda model: releases.append(model) or _release_record(),
    )

    assert sam_factory_calls == [failed, recreated]
    assert releases == [detector, failed, recreated]
    assert [len(call) for call in failed.refine_calls + recreated.refine_calls] == [
        2,
        1,
        1,
    ]
    assert recreated.propagate_calls == [("frame-000000", "frame-000001")]
    sam_record = result.stage_records[1]
    assert sam_record.status == "fallback"
    assert [
        (attempt["size"], attempt["outcome"])
        for attempt in sam_record.details["attempts"]  # type: ignore[index, union-attr]
    ] == [(2, "cuda_oom"), (1, "succeeded")]


def test_second_cuda_oom_is_raised_without_a_third_factory_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_typed_cuda_oom(monkeypatch)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))
    first = FailingDetector(FakeCudaOOM("first OOM"))
    second_error = FakeCudaOOM("second OOM")
    second = FailingDetector(second_error)
    models = [first, second]
    factory_calls: list[object] = []
    releases: list[object] = []
    sam_calls: list[bool] = []
    output = tmp_path / "semantic"

    def detector_factory() -> FakeDetector:
        model = models[len(factory_calls)]
        factory_calls.append(model)
        return model

    with pytest.raises(FakeCudaOOM) as caught:
        run_semantic_evidence(
            frames,
            output,
            policy=SemanticPolicy(0.25, 0.20, 3, 1),
            detector_factory=detector_factory,
            sam_factory=lambda: sam_calls.append(True) or FakeSam(),
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=lambda model: releases.append(model) or _release_record(),
        )

    assert caught.value is second_error
    assert factory_calls == [first, second]
    assert releases == [first, second]
    assert sam_calls == []
    assert [attempt.outcome for attempt in second_error.batch_retry_attempts] == [
        "cuda_oom",
        "cuda_oom",
    ]
    assert not output.exists()


def test_non_oom_failure_keeps_identity_releases_once_and_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_typed_cuda_oom(monkeypatch)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))
    error = RuntimeError("detector download failed")
    detector = FailingDetector(error)
    factory_calls: list[bool] = []
    releases: list[object] = []
    sam_calls: list[bool] = []
    output = tmp_path / "semantic"

    with pytest.raises(RuntimeError) as caught:
        run_semantic_evidence(
            frames,
            output,
            policy=SemanticPolicy(0.25, 0.20, 3, 1),
            detector_factory=lambda: factory_calls.append(True) or detector,
            sam_factory=lambda: sam_calls.append(True) or FakeSam(),
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=lambda model: releases.append(model) or _release_record(),
        )

    assert caught.value is error
    assert factory_calls == [True]
    assert releases == [detector]
    assert sam_calls == []
    assert not output.exists()


def test_failed_atomic_promotion_removes_staging_and_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))
    output = tmp_path / "semantic"
    promotion_error = OSError("atomic promotion failed")
    original_replace = Path.replace

    def fail_final_promotion(path: Path, target: Path) -> Path:
        if Path(target) == output and path.name.endswith(".staging"):
            raise promotion_error
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_final_promotion)
    with pytest.raises(OSError) as caught:
        run_semantic_evidence(
            frames,
            output,
            policy=SemanticPolicy(0.25, 0.20, 3, 1),
            detector_factory=FakeDetector,
            sam_factory=FakeSam,
            initial_batch_size=2,
            retry_batch_size=1,
            release_model=lambda model: _release_record(),
        )

    assert caught.value is promotion_error
    assert not output.exists()
    assert tuple(tmp_path.glob(".semantic.*.staging")) == ()


def test_same_inputs_policy_and_fake_outputs_are_byte_deterministic(
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = (_write_frame(frames_dir, 0), _write_frame(frames_dir, 1))
    output = tmp_path / "semantic"

    first = run_semantic_evidence(
        frames,
        output,
        policy=SemanticPolicy(0.25, 0.20, 3, 1),
        detector_factory=FakeDetector,
        sam_factory=FakeSam,
        initial_batch_size=2,
        retry_batch_size=1,
        release_model=lambda model: _release_record(),
    )
    first_files = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    first_manifest_sha256 = first.manifest_sha256
    shutil.rmtree(output)

    second = run_semantic_evidence(
        frames,
        output,
        policy=SemanticPolicy(0.25, 0.20, 3, 1),
        detector_factory=FakeDetector,
        sam_factory=FakeSam,
        initial_batch_size=2,
        retry_batch_size=1,
        release_model=lambda model: _release_record(),
    )
    second_files = {
        path.relative_to(output): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    assert second_files == first_files
    assert second.manifest_sha256 == first_manifest_sha256

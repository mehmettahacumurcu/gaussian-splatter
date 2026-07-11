from __future__ import annotations

import math

import cv2
import numpy as np

from backend.static_pipeline.contracts import FrameMetrics
from backend.static_pipeline.selection_metrics import (
    CandidateFrame,
    ScoredCandidate,
    choose_smart_candidates,
    pairwise_overlap,
    score_candidates,
)


def _candidate(index: int, rgb: np.ndarray) -> CandidateFrame:
    return CandidateFrame(
        frame_id=f"candidate-{index}",
        source_relative_path="room.mov",
        source_index=index,
        source_pts=index * 100,
        timestamp_s=float(index),
        ordinal=index,
        rgb=rgb,
    )


def _scored(index: int, score: float, *, rejected: bool = False) -> ScoredCandidate:
    frame = _candidate(index, np.zeros((16, 16, 3), dtype=np.uint8))
    return ScoredCandidate(
        frame=frame,
        metrics=FrameMetrics(
            sharpness=score,
            exposure_score=0.8,
            duplicate_similarity=0.0,
            overlap_score=0.5,
        ),
        sharpness_normalized=score,
        overlap_normalized=0.5,
        total_score=score,
        reasons=("bad_exposure",) if rejected else (),
        hard_rejected=rejected,
        gray_64=np.zeros((64, 64), dtype=np.uint8),
        descriptors=None,
        keypoint_count=0,
    )


def test_metric_scoring_is_finite_deterministic_and_uses_previous_candidate() -> None:
    rng = np.random.default_rng(7)
    textured = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    different_flat = np.full((96, 96, 3), 180, dtype=np.uint8)
    candidates = (
        _candidate(0, textured),
        _candidate(1, textured.copy()),
        _candidate(2, different_flat),
    )

    first = score_candidates(candidates)
    second = score_candidates(candidates)

    assert first[0].metrics.duplicate_similarity == 0.0
    assert first[1].metrics.duplicate_similarity > 0.985
    assert "near_duplicate" in first[1].reasons
    assert first[2].metrics.duplicate_similarity < 0.985
    assert "low_visual_overlap" in first[2].reasons
    assert all(math.isfinite(item.total_score) for item in first)
    assert [item.total_score for item in first] == [item.total_score for item in second]
    assert [item.reasons for item in first] == [item.reasons for item in second]


def test_constant_image_duplicate_fallback_avoids_false_flat_duplicates() -> None:
    dark = np.full((64, 64, 3), 40, dtype=np.uint8)
    bright = np.full((64, 64, 3), 180, dtype=np.uint8)
    scored = score_candidates(
        (
            _candidate(0, dark),
            _candidate(1, bright),
            _candidate(2, bright.copy()),
        )
    )

    assert scored[1].metrics.duplicate_similarity == 0.0
    assert scored[2].metrics.duplicate_similarity == 1.0


def test_equal_metric_ranges_stay_finite_and_do_not_mark_every_frame_blurred() -> None:
    flat = np.full((64, 64, 3), 127, dtype=np.uint8)
    scored = score_candidates(
        tuple(_candidate(index, flat.copy()) for index in range(4))
    )

    assert all(math.isfinite(item.sharpness_normalized) for item in scored)
    assert all(math.isfinite(item.overlap_normalized) for item in scored)
    assert not all("blurred" in item.reasons for item in scored)


def test_raw_metrics_distinguish_texture_blur_and_clipped_exposure() -> None:
    rng = np.random.default_rng(11)
    textured = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(textured, (21, 21), 0)
    black = np.zeros_like(textured)
    scored = score_candidates(
        (
            _candidate(0, textured),
            _candidate(1, blurred),
            _candidate(2, black),
        )
    )

    assert scored[0].metrics.sharpness > scored[1].metrics.sharpness
    assert "bad_exposure" in scored[2].reasons
    identical = score_candidates(
        (_candidate(3, textured), _candidate(4, textured.copy()))
    )
    assert pairwise_overlap(identical[0], identical[1]) > 0.20


def test_smart_selection_prefers_window_quality_with_temporal_coverage() -> None:
    winners = {0, 3, 6, 9, 12, 15}
    candidates = tuple(
        _scored(index, 1.0 if index in winners else 0.1) for index in range(16)
    )
    selected = choose_smart_candidates(
        candidates,
        budget=6,
        pairwise_overlap=lambda _left, _right: 0.5,
    )

    assert [item.source_index for item in selected] == [0, 3, 6, 9, 12, 15]
    assert all(
        left.timestamp_s < right.timestamp_s
        for left, right in zip(selected, selected[1:])
    )


def test_rejected_candidate_is_only_window_fallback_when_no_usable_choice() -> None:
    candidates = (
        _scored(0, 0.8),
        _scored(1, 1.0, rejected=True),
        _scored(2, 0.2),
        _scored(3, 0.9, rejected=True),
    )
    selected = choose_smart_candidates(
        candidates,
        budget=2,
        pairwise_overlap=lambda _left, _right: 0.5,
    )

    assert [item.source_index for item in selected] == [0, 2]
    assert all("temporal_window_fallback" not in item.reasons for item in selected)

    fallback = choose_smart_candidates(
        (_scored(0, 1.0, rejected=True), _scored(1, 0.5, rejected=True)),
        budget=1,
        pairwise_overlap=lambda _left, _right: 0.5,
    )
    assert "bad_exposure" in fallback[0].reasons
    assert "temporal_window_fallback" in fallback[0].reasons


def test_bridge_uses_pairwise_overlap_and_stays_within_budget() -> None:
    candidates = tuple(_scored(index, 1.0 - index * 0.05) for index in range(5))
    overlaps = {
        (0, 2): 0.05,
        (2, 4): 0.5,
        (0, 1): 0.6,
        (1, 2): 0.6,
        (1, 4): 0.4,
    }

    def overlap(left: ScoredCandidate, right: ScoredCandidate) -> float:
        key = tuple(sorted((left.ordinal, right.ordinal)))
        return overlaps.get(key, 0.5)

    selected = choose_smart_candidates(
        candidates,
        budget=3,
        pairwise_overlap=overlap,
    )

    assert len(selected) == 3
    assert selected[0].source_index == 0
    assert selected[-1].source_index == 4
    assert any("continuity_bridge" in item.reasons for item in selected)


def test_sparse_windows_add_endpoint_without_deleting_window_winner() -> None:
    candidates = (
        _scored(0, 0.1),
        _scored(1, 1.0),
        _scored(10, 0.9),
    )
    selected = choose_smart_candidates(
        candidates,
        budget=5,
        pairwise_overlap=lambda _left, _right: 0.5,
    )

    assert [item.source_index for item in selected] == [0, 1, 10]

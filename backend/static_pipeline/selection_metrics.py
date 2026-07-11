from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import cv2
import numpy as np

from .contracts import FrameMetrics


@dataclass(frozen=True)
class CandidateFrame:
    frame_id: str
    source_relative_path: str
    source_index: int | None
    source_pts: int | None
    timestamp_s: float | None
    ordinal: int
    rgb: np.ndarray = field(compare=False, repr=False)


@dataclass(frozen=True)
class ScoredCandidate:
    frame: CandidateFrame
    metrics: FrameMetrics
    sharpness_normalized: float
    overlap_normalized: float
    total_score: float
    reasons: tuple[str, ...]
    hard_rejected: bool
    gray_64: np.ndarray = field(compare=False, repr=False)
    descriptors: np.ndarray | None = field(compare=False, repr=False)
    keypoint_count: int = field(compare=False)

    @property
    def frame_id(self) -> str:
        return self.frame.frame_id

    @property
    def source_relative_path(self) -> str:
        return self.frame.source_relative_path

    @property
    def source_index(self) -> int | None:
        return self.frame.source_index

    @property
    def source_pts(self) -> int | None:
        return self.frame.source_pts

    @property
    def timestamp_s(self) -> float | None:
        return self.frame.timestamp_s

    @property
    def ordinal(self) -> int:
        return self.frame.ordinal


def _duplicate_similarity(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return 0.0
    previous_std = float(previous.std())
    current_std = float(current.std())
    if previous_std < 1e-6 or current_std < 1e-6:
        # TM_CCOEFF_NORMED is undefined for constant patches and OpenCV can
        # report unrelated flat frames as perfect duplicates. Preserve the
        # exact metric for textured frames and make the degenerate case explicit.
        return 1.0 if np.array_equal(previous, current) else 0.0
    value = float(cv2.matchTemplate(previous, current, cv2.TM_CCOEFF_NORMED)[0, 0])
    return float(np.clip(np.nan_to_num(value, nan=0.0), -1.0, 1.0))


def _descriptor_overlap(
    left_descriptors: np.ndarray | None,
    left_count: int,
    right_descriptors: np.ndarray | None,
    right_count: int,
) -> float:
    if (
        left_descriptors is None
        or right_descriptors is None
        or left_count < 12
        or right_count < 12
    ):
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    good_matches = sum(
        1
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    )
    denominator = max(min(left_count, right_count), 1)
    return float(np.clip(good_matches / denominator, 0.0, 1.0))


def pairwise_overlap(left: ScoredCandidate, right: ScoredCandidate) -> float:
    return _descriptor_overlap(
        left.descriptors,
        left.keypoint_count,
        right.descriptors,
        right.keypoint_count,
    )


def _normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    lower = float(np.percentile(array, 5, method="linear"))
    upper = float(np.percentile(array, 95, method="linear"))
    if upper - lower <= 1e-12:
        return np.full(array.shape, 0.5, dtype=np.float64)
    return np.clip((array - lower) / (upper - lower), 0.0, 1.0)


def score_candidates(
    candidates: Sequence[CandidateFrame],
) -> tuple[ScoredCandidate, ...]:
    if not candidates:
        return ()
    orb = cv2.ORB_create(nfeatures=500)
    raw_rows: list[dict[str, object]] = []
    previous_gray_64: np.ndarray | None = None
    previous_descriptors: np.ndarray | None = None
    previous_keypoint_count = 0

    for candidate in candidates:
        rgb = np.asarray(candidate.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("candidate RGB arrays must have shape HxWx3")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray_64 = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        sharpness = float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()))
        dark_fraction = float(np.mean(gray <= 8))
        bright_fraction = float(np.mean(gray >= 247))
        midtone = 1.0 - abs(float(gray.mean()) - 127.5) / 127.5
        exposure_score = float(
            np.clip(
                0.6 * (1.0 - dark_fraction - bright_fraction) + 0.4 * midtone,
                0.0,
                1.0,
            )
        )
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        keypoint_count = len(keypoints) if keypoints is not None else 0
        duplicate_similarity = _duplicate_similarity(previous_gray_64, gray_64)
        overlap_score = _descriptor_overlap(
            previous_descriptors,
            previous_keypoint_count,
            descriptors,
            keypoint_count,
        )
        raw_rows.append(
            {
                "candidate": candidate,
                "sharpness": sharpness,
                "exposure_score": exposure_score,
                "duplicate_similarity": duplicate_similarity,
                "overlap_score": overlap_score,
                "clipped_fraction": dark_fraction + bright_fraction,
                "gray_64": gray_64,
                "descriptors": descriptors,
                "keypoint_count": keypoint_count,
            }
        )
        previous_gray_64 = gray_64
        previous_descriptors = descriptors
        previous_keypoint_count = keypoint_count

    sharpness_values = [float(row["sharpness"]) for row in raw_rows]
    overlap_values = [float(row["overlap_score"]) for row in raw_rows]
    sharpness_normalized = _normalize(sharpness_values)
    overlap_normalized = _normalize(overlap_values)
    blur_threshold = float(
        np.percentile(np.asarray(sharpness_values), 10, method="linear")
    )

    scored: list[ScoredCandidate] = []
    for index, row in enumerate(raw_rows):
        reasons: list[str] = []
        if float(row["clipped_fraction"]) > 0.35:
            reasons.append("bad_exposure")
        if float(row["sharpness"]) < blur_threshold:
            reasons.append("blurred")
        if float(row["duplicate_similarity"]) > 0.985:
            reasons.append("near_duplicate")
        if int(row["keypoint_count"]) < 12 or (
            index > 0 and int(raw_rows[index - 1]["keypoint_count"]) < 12
        ):
            reasons.append("low_visual_overlap")

        total_score = float(
            0.45 * sharpness_normalized[index]
            + 0.25 * float(row["exposure_score"])
            + 0.20 * overlap_normalized[index]
            + 0.10 * (1.0 - float(row["duplicate_similarity"]))
        )
        hard_rejected = any(
            reason in {"bad_exposure", "blurred", "near_duplicate"}
            for reason in reasons
        )
        scored.append(
            ScoredCandidate(
                frame=row["candidate"],  # type: ignore[arg-type]
                metrics=FrameMetrics(
                    sharpness=float(row["sharpness"]),
                    exposure_score=float(row["exposure_score"]),
                    duplicate_similarity=float(row["duplicate_similarity"]),
                    overlap_score=float(row["overlap_score"]),
                ),
                sharpness_normalized=float(sharpness_normalized[index]),
                overlap_normalized=float(overlap_normalized[index]),
                total_score=total_score,
                reasons=tuple(reasons),
                hard_rejected=hard_rejected,
                gray_64=row["gray_64"],  # type: ignore[arg-type]
                descriptors=row["descriptors"],  # type: ignore[arg-type]
                keypoint_count=int(row["keypoint_count"]),
            )
        )
    return tuple(scored)


def _candidate_key(candidate: ScoredCandidate) -> tuple[float, int, str]:
    coordinate = (
        candidate.timestamp_s
        if candidate.timestamp_s is not None
        else float(candidate.ordinal)
    )
    return (coordinate, candidate.ordinal, candidate.frame_id)


def _window_assignments(
    candidates: Sequence[ScoredCandidate], window_count: int
) -> dict[str, int]:
    ordered = sorted(candidates, key=_candidate_key)
    timestamps = [candidate.timestamp_s for candidate in ordered]
    use_timestamps = all(value is not None for value in timestamps)
    if use_timestamps:
        start = float(timestamps[0])  # type: ignore[arg-type]
        end = float(timestamps[-1])  # type: ignore[arg-type]
        if end > start:
            return {
                candidate.frame_id: min(
                    window_count - 1,
                    int(
                        (float(candidate.timestamp_s) - start)
                        / (end - start)
                        * window_count
                    ),
                )
                for candidate in ordered
            }
    return {
        candidate.frame_id: min(window_count - 1, index * window_count // len(ordered))
        for index, candidate in enumerate(ordered)
    }


def _append_reason(candidate: ScoredCandidate, reason: str) -> ScoredCandidate:
    if reason in candidate.reasons:
        return candidate
    return replace(candidate, reasons=candidate.reasons + (reason,))


def choose_smart_candidates(
    candidates: Sequence[ScoredCandidate],
    budget: int,
    *,
    pairwise_overlap: Callable[
        [ScoredCandidate, ScoredCandidate], float
    ] = pairwise_overlap,
) -> tuple[ScoredCandidate, ...]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not candidates:
        return ()
    ordered = sorted(candidates, key=_candidate_key)
    window_count = min(budget, len(ordered))
    assignments = _window_assignments(ordered, window_count)
    windows: list[list[ScoredCandidate]] = [[] for _ in range(window_count)]
    for candidate in ordered:
        windows[assignments[candidate.frame_id]].append(candidate)

    selected: dict[str, ScoredCandidate] = {}
    for window in windows:
        if not window:
            continue
        usable = [candidate for candidate in window if not candidate.hard_rejected]
        pool = usable or window
        winner = max(
            pool, key=lambda item: (item.total_score, -item.ordinal, item.frame_id)
        )
        if not usable:
            winner = _append_reason(winner, "temporal_window_fallback")
        selected[winner.frame_id] = winner

    usable_all = [candidate for candidate in ordered if not candidate.hard_rejected]
    endpoint_candidates = (usable_all[0], usable_all[-1]) if usable_all else ()
    endpoint_ids: set[str] = set()
    for endpoint in endpoint_candidates:
        endpoint_ids.add(endpoint.frame_id)
        if endpoint.frame_id in selected:
            continue
        if len(selected) >= budget:
            window_index = assignments[endpoint.frame_id]
            current_in_window = [
                candidate
                for candidate in selected.values()
                if assignments[candidate.frame_id] == window_index
            ]
            if current_in_window:
                selected.pop(current_in_window[0].frame_id)
        selected[endpoint.frame_id] = endpoint

    initial_pairs = list(
        zip(
            sorted(selected.values(), key=_candidate_key),
            sorted(selected.values(), key=_candidate_key)[1:],
        )
    )
    for left, right in initial_pairs:
        if pairwise_overlap(left, right) >= 0.12:
            continue
        between = [
            candidate
            for candidate in ordered
            if left.ordinal < candidate.ordinal < right.ordinal
            and candidate.frame_id not in selected
            and not candidate.hard_rejected
        ]
        if not between:
            continue
        midpoint = (left.ordinal + right.ordinal) / 2.0
        bridge = max(
            between,
            key=lambda candidate: (
                min(
                    pairwise_overlap(left, candidate),
                    pairwise_overlap(candidate, right),
                ),
                candidate.total_score,
                -abs(candidate.ordinal - midpoint),
                -candidate.ordinal,
            ),
        )
        bridge = _append_reason(bridge, "continuity_bridge")
        if len(selected) < budget:
            selected[bridge.frame_id] = bridge
            continue

        tentative = sorted([*selected.values(), bridge], key=_candidate_key)
        removable: list[ScoredCandidate] = []
        for index in range(1, len(tentative) - 1):
            candidate = tentative[index]
            if (
                candidate.frame_id in endpoint_ids
                or candidate.frame_id == bridge.frame_id
                or "continuity_bridge" in candidate.reasons
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
        selected.pop(victim.frame_id, None)
        selected[bridge.frame_id] = bridge

    return tuple(sorted(selected.values(), key=_candidate_key))

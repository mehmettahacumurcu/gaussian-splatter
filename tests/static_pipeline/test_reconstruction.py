from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

import backend.static_pipeline.reconstruction as reconstruction_module
from backend.static_pipeline.contracts import (
    ColmapAttempt,
    FrameRecord,
    ReconstructionBundle,
    SelectionManifest,
    SelectionPolicy,
)
from backend.static_pipeline.reconstruction import (
    ReconstructionGateError,
    evaluate_reconstruction,
    measure_models,
    rank_models,
    reconstruct_with_gate,
)
from tests.static_pipeline.fixtures import write_colmap_text_model


def _manifest(
    count: int,
    *,
    mode: str = "smart",
    effective_mode: str | None = None,
    timestamp_step: float | None = 0.1,
    digest: str = "a" * 64,
) -> SelectionManifest:
    frames = tuple(
        FrameRecord(
            frame_id=f"{index:024x}",
            source_relative_path="capture.mov"
            if timestamp_step is not None
            else f"IMG_{index:04d}.jpg",
            source_index=index if timestamp_step is not None else None,
            source_pts=index * 1_000 if timestamp_step is not None else None,
            timestamp_s=index * timestamp_step if timestamp_step is not None else None,
            output_name=f"frame_{index:06d}.png",
            sha256=f"{index:064x}",
            selected=True,
            metrics=None,
            selection_score=None,
            reasons=(mode,),
        )
        for index in range(count)
    )
    return SelectionManifest(
        schema_version=1,
        source_digest="f" * 64,
        effective_mode=effective_mode or mode,
        policy=SelectionPolicy(
            mode=mode,
            frame_budget=count,
            resolution_long_edge_cap=1280,
        ),
        frames=frames,
        image_set_digest=digest,
    )


def _names(manifest: SelectionManifest, indices: set[int] | range) -> tuple[str, ...]:
    selected = manifest.selected_frames
    return tuple(selected[index].output_name for index in sorted(indices))


def _model(
    root: Path,
    names: tuple[str, ...],
    *,
    errors: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0),
    tracks: tuple[int, ...] = (4, 4, 4, 4),
    invalid_pose: bool = False,
) -> Path:
    return write_colmap_text_model(root, names, errors, tracks, invalid_pose)


def _colmap_attempt(attempt_dir: Path, names: tuple[str, ...]) -> ColmapAttempt:
    model = _model(attempt_dir / "sparse" / "0", names)
    database = attempt_dir / "colmap.db"
    database.write_bytes(b"db")
    return ColmapAttempt(
        root=attempt_dir,
        database_path=database,
        model_dirs=(model,),
        colmap_version="COLMAP 3.11",
        fingerprint="c" * 64,
    )


def test_fragmented_285_of_502_fails_with_exact_share_and_smart_retry(
    tmp_path: Path,
) -> None:
    manifest = _manifest(502)
    component_sizes = (285, 40, 30, 25, 20, 18, 16, 14, 12, 10, 7, 5)
    cursor = 0
    models = []
    for component_index, size in enumerate(component_sizes):
        component_names = _names(manifest, range(cursor, cursor + size))
        models.append(_model(tmp_path / str(component_index), component_names))
        cursor += size

    decision = evaluate_reconstruction(measure_models(models, manifest), manifest)

    assert decision.passed is False
    assert decision.dominant.registered_count == 285
    assert decision.dominant.registered_ratio == pytest.approx(285 / 502)
    assert decision.dominant.registered_share == pytest.approx(285 / 482)
    assert decision.retry_recommended is True
    assert decision.failures[:2] == ("registered_ratio", "dominant_component")


def test_healthy_95_of_100_passes_with_bounded_gaps(tmp_path: Path) -> None:
    manifest = _manifest(100)
    missing = {10, 30, 50, 70, 90}
    registered = set(range(100)) - missing
    model = _model(tmp_path / "healthy", _names(manifest, registered))

    decision = evaluate_reconstruction(measure_models((model,), manifest), manifest)

    assert decision.passed is True
    assert decision.failures == ()
    assert decision.warnings == ()
    assert decision.dominant.registered_ratio == pytest.approx(0.95)
    assert decision.dominant.max_interior_gap_s == pytest.approx(0.2)


def test_registration_count_beats_padded_model_size(tmp_path: Path) -> None:
    manifest = _manifest(100)
    strong = _model(tmp_path / "strong", _names(manifest, range(95)))
    weak = _model(tmp_path / "weak", _names(manifest, range(40)))
    with (weak / "points3D.txt").open("a", encoding="utf-8") as handle:
        handle.write("#" + ("padding" * 20_000) + "\n")

    ranked = rank_models(measure_models((weak, strong), manifest))

    assert ranked[0].model_dir == strong
    assert ranked[0].registered_count == 95


@pytest.mark.parametrize("corruption", ["unknown_name", "invalid_pose"])
def test_unknown_names_or_invalid_pose_fail_stable_validity_key(
    tmp_path: Path,
    corruption: str,
) -> None:
    manifest = _manifest(100)
    registered = _names(manifest, set(range(100)) - {10, 30, 50, 70, 90})
    names = registered + (("unknown.png",) if corruption == "unknown_name" else ())
    model = _model(
        tmp_path / corruption,
        names,
        invalid_pose=corruption == "invalid_pose",
    )

    decision = evaluate_reconstruction(measure_models((model,), manifest), manifest)

    assert decision.passed is False
    assert "valid_names_intrinsics_poses" in decision.failures
    if corruption == "unknown_name":
        assert decision.dominant.registered_count == 95
        assert "unknown.png" not in decision.dominant.registered_names
    else:
        assert decision.retry_recommended is False


def test_video_gap_intervals_include_start_interior_and_end(tmp_path: Path) -> None:
    manifest = _manifest(5, timestamp_step=0.5)
    model = _model(tmp_path / "video", _names(manifest, {1, 3}))

    decision = evaluate_reconstruction(measure_models((model,), manifest), manifest)

    assert [interval.kind for interval in decision.uncovered_intervals] == [
        "start",
        "interior",
        "end",
    ]
    assert [
        (interval.start_s, interval.end_s, interval.missing_frame_ids)
        for interval in decision.uncovered_intervals
    ] == [
        (0.0, 0.5, (manifest.frames[0].frame_id,)),
        (0.5, 1.5, (manifest.frames[2].frame_id,)),
        (1.5, 2.0, (manifest.frames[4].frame_id,)),
    ]
    assert all(
        interval.coverage_unit == "seconds" for interval in decision.uncovered_intervals
    )


def test_video_tied_timestamps_preserve_selected_manifest_order(
    tmp_path: Path,
) -> None:
    manifest = _manifest(3, timestamp_step=0.5)
    first, second, third = manifest.frames
    tied_second = replace(second, timestamp_s=first.timestamp_s)
    reordered = replace(manifest, frames=(tied_second, first, third))
    model = _model(
        tmp_path / "tied-order",
        (tied_second.output_name, third.output_name),
    )

    decision = evaluate_reconstruction(measure_models((model,), reordered), reordered)

    assert len(decision.uncovered_intervals) == 1
    interval = decision.uncovered_intervals[0]
    assert interval.kind == "interior"
    assert interval.missing_frame_ids == (first.frame_id,)
    assert interval.left_boundary_frame_id == tied_second.frame_id
    assert interval.right_boundary_frame_id == third.frame_id


def test_video_timestamps_must_be_nondecreasing_in_manifest_order(
    tmp_path: Path,
) -> None:
    manifest = _manifest(3, timestamp_step=0.5)
    first, second, third = manifest.frames
    reordered = replace(manifest, frames=(second, first, third))
    model = _model(tmp_path / "out-of-order", _names(manifest, range(3)))

    with pytest.raises(ValueError, match="nondecreasing"):
        measure_models((model,), reordered)


def test_photo_order_gap_intervals_use_selected_manifest_order(tmp_path: Path) -> None:
    manifest = _manifest(6, timestamp_step=None)
    model = _model(tmp_path / "photos", _names(manifest, {1, 3, 4}))

    decision = evaluate_reconstruction(measure_models((model,), manifest), manifest)

    assert decision.dominant.coverage_unit == "photo_order"
    assert decision.dominant.start_gap_s == 1.0
    assert decision.dominant.max_interior_gap_s == 2.0
    assert decision.dominant.end_gap_s == 1.0
    assert [interval.kind for interval in decision.uncovered_intervals] == [
        "start",
        "interior",
        "end",
    ]
    assert (
        decision.uncovered_intervals[1].left_boundary_frame_id
        == manifest.frames[1].frame_id
    )
    assert (
        decision.uncovered_intervals[1].right_boundary_frame_id
        == manifest.frames[3].frame_id
    )
    assert all(
        interval.coverage_unit == "photo_order"
        for interval in decision.uncovered_intervals
    )


def test_ninety_percent_registration_passes_with_only_stable_warning(
    tmp_path: Path,
) -> None:
    manifest = _manifest(100)
    missing = set(range(5, 100, 10))
    model = _model(tmp_path / "warning", _names(manifest, set(range(100)) - missing))

    decision = evaluate_reconstruction(measure_models((model,), manifest), manifest)

    assert decision.passed is True
    assert decision.failures == ()
    assert decision.warnings == ("below_95_percent_registration_target",)


def test_point_quality_failure_is_deterministic_and_does_not_retry(
    tmp_path: Path,
) -> None:
    manifest = _manifest(10)
    model = _model(tmp_path / "empty-points", _names(manifest, range(10)))
    (model / "points3D.txt").write_text("# empty\n", encoding="utf-8")

    measured = measure_models((model,), manifest)[0]
    decision = evaluate_reconstruction((measured,), manifest)

    assert math.isinf(measured.median_reprojection_error_px)
    assert math.isinf(measured.p95_reprojection_error_px)
    assert measured.median_track_length == 0.0
    assert "median_reprojection" in decision.failures
    assert "p95_reprojection" in decision.failures
    assert "median_track_length" in decision.failures
    assert decision.retry_recommended is False


def test_pose_validity_cofailure_does_not_veto_useful_smart_retry(
    tmp_path: Path,
) -> None:
    manifest = _manifest(100)
    model = _model(
        tmp_path / "coverage-and-pose",
        _names(manifest, range(80)),
        invalid_pose=True,
    )

    decision = evaluate_reconstruction(measure_models((model,), manifest), manifest)

    assert "registered_ratio" in decision.failures
    assert "valid_names_intrinsics_poses" in decision.failures
    assert decision.uncovered_intervals
    assert decision.retry_recommended is True


def test_non_numeric_point_arrays_fail_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(5)
    model = _model(tmp_path / "malformed-points", _names(manifest, range(5)))
    monkeypatch.setattr(
        reconstruction_module.colmap_parser,
        "load_points3d_from_model",
        lambda _model: ([["bad", 0, 1]], [[1, 2, 3]], [4], [0.5]),
    )

    measured = measure_models((model,), manifest)[0]

    assert math.isinf(measured.median_reprojection_error_px)
    assert math.isinf(measured.p95_reprojection_error_px)
    assert measured.median_track_length == 0.0
    assert measured.sparse_point_count == 0


def test_invalid_attempt_index_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(5)
    model = _model(tmp_path / "model", _names(manifest, range(5)))
    measured = measure_models((model,), manifest)

    with pytest.raises(ValueError, match="attempt"):
        evaluate_reconstruction(measured, manifest, attempt_index=2)


def test_reconstruct_retries_smart_once_records_decisions_and_promotes(
    tmp_path: Path,
) -> None:
    manifest = _manifest(100)
    first_missing = set(range(2, 100, 5))
    second_missing = {10, 30, 50, 70, 90}
    attempt_calls: list[int] = []

    def run_attempt(
        current: SelectionManifest, root: Path, _policy: object
    ) -> ColmapAttempt:
        attempt_calls.append(len(attempt_calls))
        missing = first_missing if len(attempt_calls) == 1 else second_missing
        return _colmap_attempt(root, _names(current, set(range(100)) - missing))

    def backfill(
        current: SelectionManifest, _intervals: tuple[object, ...]
    ) -> SelectionManifest:
        return replace(current, image_set_digest="b" * 64)

    bundle = reconstruct_with_gate(
        manifest,
        use_gpu=True,
        attempt_root=tmp_path / "attempts",
        run_attempt=run_attempt,
        materialize_backfill=backfill,
    )

    assert isinstance(bundle, ReconstructionBundle)
    assert attempt_calls == [0, 1]
    assert len(bundle.attempts) == 2
    assert len(bundle.decisions) == 2
    assert bundle.decisions[0].passed is False
    assert bundle.decisions[1].passed is True
    assert bundle.selected_manifest.image_set_digest == "b" * 64
    assert bundle.accepted_model_dir == tmp_path / "attempts" / "validated"
    assert (bundle.accepted_model_dir / "cameras.txt").is_file()


def test_validated_publication_copies_only_required_text_model_files(
    tmp_path: Path,
) -> None:
    manifest = _manifest(20)

    def run_attempt(
        current: SelectionManifest, root: Path, _policy: object
    ) -> ColmapAttempt:
        attempt = _colmap_attempt(root, _names(current, range(20)))
        model = attempt.model_dirs[0]
        (model / "ignored.bin").write_bytes(b"ignored")
        extras = model / "ignored-directory"
        extras.mkdir()
        (extras / "ignored.txt").write_bytes(b"ignored")
        return attempt

    bundle = reconstruct_with_gate(
        manifest,
        use_gpu=True,
        attempt_root=tmp_path / "attempts",
        run_attempt=run_attempt,
        materialize_backfill=lambda current, _intervals: current,
    )

    assert {path.name for path in bundle.accepted_model_dir.iterdir()} == {
        "cameras.txt",
        "images.txt",
        "points3D.txt",
    }


def test_noop_smart_backfill_does_not_launch_duplicate_attempt(tmp_path: Path) -> None:
    manifest = _manifest(20)
    calls = 0

    def run_attempt(
        current: SelectionManifest, root: Path, _policy: object
    ) -> ColmapAttempt:
        nonlocal calls
        calls += 1
        return _colmap_attempt(root, _names(current, range(10)))

    with pytest.raises(ReconstructionGateError) as captured:
        reconstruct_with_gate(
            manifest,
            use_gpu=True,
            attempt_root=tmp_path / "attempts",
            run_attempt=run_attempt,
            materialize_backfill=lambda current, _intervals: current,
        )

    assert calls == 1
    assert len(captured.value.attempts) == 1
    assert len(captured.value.decisions) == 1


def test_fixed_failure_never_calls_backfill(tmp_path: Path) -> None:
    manifest = _manifest(20, mode="fixed_fps", effective_mode="fixed_fps")
    calls = 0

    def run_attempt(
        current: SelectionManifest, root: Path, _policy: object
    ) -> ColmapAttempt:
        nonlocal calls
        calls += 1
        return _colmap_attempt(root, _names(current, range(10)))

    def unexpected_backfill(*_args: object) -> SelectionManifest:
        raise AssertionError("fixed mode requested backfill")

    with pytest.raises(ReconstructionGateError):
        reconstruct_with_gate(
            manifest,
            use_gpu=False,
            attempt_root=tmp_path / "attempts",
            run_attempt=run_attempt,
            materialize_backfill=unexpected_backfill,
        )

    assert calls == 1


def test_second_smart_failure_stops_after_exactly_two_attempts(tmp_path: Path) -> None:
    manifest = _manifest(20)
    calls = 0

    def run_attempt(
        current: SelectionManifest, root: Path, _policy: object
    ) -> ColmapAttempt:
        nonlocal calls
        calls += 1
        return _colmap_attempt(root, _names(current, range(10)))

    with pytest.raises(ReconstructionGateError) as captured:
        reconstruct_with_gate(
            manifest,
            use_gpu=True,
            attempt_root=tmp_path / "attempts",
            run_attempt=run_attempt,
            materialize_backfill=lambda current, _intervals: replace(
                current, image_set_digest="b" * 64
            ),
        )

    assert calls == 2
    assert len(captured.value.attempts) == 2
    assert len(captured.value.decisions) == 2
    assert captured.value.decision is captured.value.decisions[-1]


def test_failed_attempt_preserves_preexisting_validated_bytes(tmp_path: Path) -> None:
    manifest = _manifest(20, mode="fixed_fps", effective_mode="fixed_fps")
    attempt_root = tmp_path / "attempts"
    validated = attempt_root / "validated"
    validated.mkdir(parents=True)
    (validated / "keep.bin").write_bytes(b"previous-good")

    with pytest.raises(ReconstructionGateError):
        reconstruct_with_gate(
            manifest,
            use_gpu=False,
            attempt_root=attempt_root,
            run_attempt=lambda current, root, _policy: _colmap_attempt(
                root, _names(current, range(10))
            ),
            materialize_backfill=lambda _current, _intervals: pytest.fail(
                "fixed mode requested backfill"
            ),
        )

    assert (validated / "keep.bin").read_bytes() == b"previous-good"


def test_promotion_race_preserves_destination_and_cleans_owned_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(20)
    attempt_root = tmp_path / "attempts"
    real_promote = reconstruction_module.promote_directory

    def raced_promotion(staging: Path, target: Path) -> None:
        if target.name != "validated":
            real_promote(staging, target)
            return
        target.mkdir()
        (target / "keep.bin").write_bytes(b"raced-winner")
        raise FileExistsError(target)

    monkeypatch.setattr(reconstruction_module, "promote_directory", raced_promotion)

    with pytest.raises(FileExistsError):
        reconstruct_with_gate(
            manifest,
            use_gpu=True,
            attempt_root=attempt_root,
            run_attempt=lambda current, root, _policy: _colmap_attempt(
                root, _names(current, range(20))
            ),
            materialize_backfill=lambda current, _intervals: current,
        )

    assert (attempt_root / "validated" / "keep.bin").read_bytes() == b"raced-winner"
    assert not list(attempt_root.glob(".validated.tmp-*"))


def test_failed_model_copy_cleans_the_owned_partial_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(20)
    attempt_root = tmp_path / "attempts"

    def failed_copy(_source: Path, destination: Path, **_kwargs: object) -> None:
        destination.write_bytes(b"partial")
        raise OSError("injected copy failure")

    monkeypatch.setattr(reconstruction_module.shutil, "copy2", failed_copy)

    with pytest.raises(OSError, match="injected"):
        reconstruct_with_gate(
            manifest,
            use_gpu=True,
            attempt_root=attempt_root,
            run_attempt=lambda current, root, _policy: _colmap_attempt(
                root, _names(current, range(20))
            ),
            materialize_backfill=lambda current, _intervals: current,
        )

    assert not list(attempt_root.glob(".validated.tmp-*"))


def test_owned_stage_cleanup_never_deletes_a_replacement_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = tmp_path / ".validated.tmp-owned"
    stage.mkdir()
    (stage / "owned.bin").write_bytes(b"owned")
    owned_identity = reconstruction_module._directory_identity(stage)
    assert owned_identity is not None
    moved_owned = tmp_path / "owned-moved"
    real_identity = reconstruction_module._directory_identity
    swapped = False

    def swap_after_identity(path: Path) -> tuple[int, int] | None:
        nonlocal swapped
        identity = real_identity(path)
        if path == stage and not swapped:
            swapped = True
            stage.rename(moved_owned)
            stage.mkdir()
            (stage / "foreign.bin").write_bytes(b"foreign")
        return identity

    monkeypatch.setattr(
        reconstruction_module,
        "_directory_identity",
        swap_after_identity,
    )

    reconstruction_module._remove_owned_stage(stage, owned_identity)

    assert (stage / "foreign.bin").read_bytes() == b"foreign"
    assert (moved_owned / "owned.bin").read_bytes() == b"owned"
    assert not list(tmp_path.glob(".*.cleanup-*"))


@pytest.mark.skipif(os.name != "nt", reason="exercises Windows handle pinning")
def test_windows_cleanup_pins_claimed_quarantine_during_recursive_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = tmp_path / ".validated.tmp-owned"
    stage.mkdir()
    (stage / "owned.bin").write_bytes(b"owned")
    owned_identity = reconstruction_module._directory_identity(stage)
    assert owned_identity is not None
    moved_claim = tmp_path / "moved-claim"
    real_clear = reconstruction_module._clear_directory_contents
    rename_was_blocked = False

    def attempt_swap_while_pinned(path: Path) -> None:
        nonlocal rename_was_blocked
        try:
            path.rename(moved_claim)
        except OSError:
            rename_was_blocked = True
        else:
            raise AssertionError("claimed cleanup directory was not handle-pinned")
        real_clear(path)

    monkeypatch.setattr(
        reconstruction_module,
        "_clear_directory_contents",
        attempt_swap_while_pinned,
    )

    reconstruction_module._remove_owned_stage(stage, owned_identity)

    assert rename_was_blocked is True
    assert not stage.exists()
    assert not moved_claim.exists()
    assert not list(tmp_path.glob(".*.cleanup-*"))


@pytest.mark.skipif(os.name == "nt", reason="exercises POSIX fd-relative cleanup")
def test_posix_cleanup_clears_the_opened_inode_not_a_path_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = tmp_path / ".validated.tmp-owned"
    stage.mkdir()
    (stage / "owned.bin").write_bytes(b"owned")
    owned_identity = reconstruction_module._directory_identity(stage)
    assert owned_identity is not None
    moved_claim = tmp_path / "moved-claim"
    real_clear = reconstruction_module._clear_directory_fd
    swapped = False

    def swap_path_then_clear_opened_inode(directory_fd: int) -> None:
        nonlocal swapped
        quarantine = next(tmp_path.glob(".*.cleanup-*"))
        quarantine.rename(moved_claim)
        quarantine.mkdir()
        (quarantine / "foreign.bin").write_bytes(b"foreign")
        swapped = True
        real_clear(directory_fd)

    monkeypatch.setattr(
        reconstruction_module,
        "_clear_directory_fd",
        swap_path_then_clear_opened_inode,
    )

    reconstruction_module._remove_owned_stage(stage, owned_identity)

    assert swapped is True
    quarantine = next(tmp_path.glob(".*.cleanup-*"))
    assert (quarantine / "foreign.bin").read_bytes() == b"foreign"
    assert moved_claim.is_dir()
    assert not list(moved_claim.iterdir())

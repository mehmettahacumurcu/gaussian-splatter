from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import ExifTags, Image, ImageChops

from backend.static_pipeline import selection as selection_module
from backend.static_pipeline.contracts import (
    SelectionPolicy,
    SourceInventory,
    UncoveredInterval,
)
from backend.static_pipeline.selection import (
    FfmpegMediaBackend,
    TimelineFrame,
    VideoProbe,
    bound_selection_for_reconstruction,
    make_frame_id,
    plan_backfill,
    select_frames,
    write_selection_manifest,
)
from backend.static_pipeline.sources import discover_source
from tests.static_pipeline.fixtures import (
    FakeMediaBackend,
    _write_image,
)


def test_fixed_video_selection_is_deterministic_and_atomic(
    tmp_path: Path,
    video_inventory: SourceInventory,
    fake_media_backend: FakeMediaBackend,
) -> None:
    policy = SelectionPolicy(
        mode="fixed_fps",
        frame_budget=300,
        resolution_long_edge_cap=1280,
        fixed_fps=4,
    )
    first = select_frames(
        video_inventory,
        tmp_path / "frames",
        policy,
        media=fake_media_backend,
    )
    second = select_frames(
        video_inventory,
        tmp_path / "frames-again",
        policy,
        media=fake_media_backend,
    )
    assert first.effective_mode == "fixed_fps"
    assert first.image_set_digest == second.image_set_digest
    assert len(first.selected_frames) <= 300
    assert all(
        frame.output_name == f"frame_{index:06d}.png"
        for index, frame in enumerate(first.selected_frames)
    )


def test_photo_fixed_mode_keeps_all_images_and_records_effective_mode(
    tmp_path: Path,
    photo_inventory: SourceInventory,
    fake_media_backend: FakeMediaBackend,
) -> None:
    policy = SelectionPolicy(
        mode="fixed_fps",
        frame_budget=300,
        resolution_long_edge_cap=1280,
        fixed_fps=4,
    )
    manifest = select_frames(
        photo_inventory,
        tmp_path / "frames",
        policy,
        media=fake_media_backend,
    )
    assert manifest.effective_mode == "photo_set_all"
    assert len(manifest.selected_frames) == len(photo_inventory.media_files)


def test_large_photo_all_manifest_gets_a_recorded_colmap_guardrail(
    tmp_path: Path,
    large_photo_inventory: SourceInventory,
    fake_media_backend: FakeMediaBackend,
) -> None:
    policy = SelectionPolicy(
        mode="fixed_fps",
        frame_budget=300,
        resolution_long_edge_cap=1280,
        fixed_fps=4,
    )
    original = select_frames(
        large_photo_inventory,
        tmp_path / "all-frames",
        policy,
        media=fake_media_backend,
    )
    original_names = sorted(
        path.name for path in (tmp_path / "all-frames").glob("*.png")
    )
    bounded = bound_selection_for_reconstruction(
        large_photo_inventory,
        original,
        tmp_path / "colmap-frames",
        media=fake_media_backend,
    )
    assert len(original.selected_frames) == len(large_photo_inventory.media_files)
    assert len(bounded.selected_frames) == 300
    assert bounded.reconstruction_guardrail == "uniform_profile_cap"
    assert any(
        "colmap_budget_guardrail" in frame.reasons
        for frame in bounded.frames
        if not frame.selected
    )
    assert original_names == sorted(
        path.name for path in (tmp_path / "all-frames").glob("*.png")
    )


def test_selection_replaces_stale_tail_files(
    tmp_path: Path,
    video_inventory: SourceInventory,
    fake_media_backend: FakeMediaBackend,
) -> None:
    target = tmp_path / "frames"
    target.mkdir()
    (target / "frame_999999.png").write_bytes(b"stale")
    select_frames(
        video_inventory,
        target,
        SelectionPolicy(
            mode="fixed_fps",
            frame_budget=3,
            resolution_long_edge_cap=1280,
        ),
        media=fake_media_backend,
    )
    assert not (target / "frame_999999.png").exists()
    assert sorted(path.name for path in target.glob("*.png")) == [
        "frame_000000.png",
        "frame_000001.png",
        "frame_000002.png",
    ]


def test_smart_selection_materializes_selected_frames_and_records_all_candidates(
    tmp_path: Path,
    video_inventory: SourceInventory,
    fake_media_backend: FakeMediaBackend,
) -> None:
    policy = SelectionPolicy(
        mode="smart",
        frame_budget=6,
        resolution_long_edge_cap=1280,
        candidate_fps=12,
        candidate_long_edge=320,
    )
    first = select_frames(
        video_inventory,
        tmp_path / "smart-frames",
        policy,
        media=fake_media_backend,
    )
    second = select_frames(
        video_inventory,
        tmp_path / "smart-frames-again",
        policy,
        media=fake_media_backend,
    )

    assert first.effective_mode == "smart"
    assert len(first.frames) > len(first.selected_frames)
    assert len(first.selected_frames) <= policy.frame_budget
    assert first.image_set_digest == second.image_set_digest
    assert [frame.frame_id for frame in first.selected_frames] == [
        frame.frame_id for frame in second.selected_frames
    ]
    assert all(frame.metrics is not None for frame in first.frames)
    assert all(
        frame.output_name == "" and frame.sha256 == ""
        for frame in first.frames
        if not frame.selected
    )
    assert len(list((tmp_path / "smart-frames").glob("*.png"))) == len(
        first.selected_frames
    )
    extraction_caps = [
        call[4]
        for call in fake_media_backend.calls
        if call[0] == "extract_video_indices"
    ]
    assert extraction_caps[:2] == [320, 1280]
    assert not list(tmp_path.glob("*.candidates-*"))


def test_smart_video_materialization_preserves_timestamp_record_mapping(
    tmp_path: Path,
    video_inventory: SourceInventory,
) -> None:
    backend = FakeMediaBackend(
        timeline=(
            TimelineFrame(2, 200, 0.0),
            TimelineFrame(0, 0, 1.0),
            TimelineFrame(1, 100, 2.0),
        ),
        probe=VideoProbe(1920, 1080, 3.0, 0),
    )
    manifest = select_frames(
        video_inventory,
        tmp_path / "smart",
        SelectionPolicy(
            mode="smart",
            frame_budget=3,
            resolution_long_edge_cap=1280,
        ),
        media=backend,
    )

    assert [frame.source_index for frame in manifest.selected_frames] == [2, 0, 1]
    for frame in manifest.selected_frames:
        with Image.open(tmp_path / "smart" / frame.output_name) as image:
            assert image.getpixel((0, 0))[0] == frame.source_index


def test_backfill_adds_at_most_two_bridges_and_keeps_smart_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    video_inventory: SourceInventory,
    fake_media_backend: FakeMediaBackend,
) -> None:
    policy = SelectionPolicy(
        mode="smart",
        frame_budget=6,
        resolution_long_edge_cap=1280,
    )
    original = select_frames(
        video_inventory,
        tmp_path / "smart",
        policy,
        media=fake_media_backend,
    )
    original = replace(
        original,
        policy=replace(original.policy, frame_budget=8),
    )
    gap = UncoveredInterval(start_s=1.0, end_s=3.0, missing_frame_ids=())
    gaps = (gap, gap)
    monkeypatch.setattr(selection_module, "pairwise_overlap", lambda _left, _right: 0.5)
    backfilled = plan_backfill(
        video_inventory,
        original,
        gaps,
        tmp_path / "backfill",
        media=fake_media_backend,
        max_per_interval=99,
    )
    added = {frame.frame_id for frame in backfilled.selected_frames} - {
        frame.frame_id for frame in original.selected_frames
    }
    assert 1 <= len(added) <= 2
    assert len(backfilled.selected_frames) == len(original.selected_frames)
    assert all(
        "colmap_gap_backfill" in frame.reasons
        for frame in backfilled.selected_frames
        if frame.frame_id in added
    )
    assert len(list((tmp_path / "backfill").glob("*.png"))) == len(
        backfilled.selected_frames
    )
    with pytest.raises(ValueError, match="already"):
        plan_backfill(
            video_inventory,
            backfilled,
            gaps,
            tmp_path / "backfill-again",
            media=fake_media_backend,
        )


def test_fixed_mode_rejects_backfill_before_touching_output(
    tmp_path: Path,
    video_inventory: SourceInventory,
    fake_media_backend: FakeMediaBackend,
) -> None:
    fixed = select_frames(
        video_inventory,
        tmp_path / "fixed",
        SelectionPolicy(
            mode="fixed_fps",
            frame_budget=6,
            resolution_long_edge_cap=1280,
        ),
        media=fake_media_backend,
    )
    calls_before = len(fake_media_backend.calls)
    target = tmp_path / "backfill"

    with pytest.raises(ValueError, match="Smart"):
        plan_backfill(
            video_inventory,
            fixed,
            (),
            target,
            media=fake_media_backend,
        )

    assert not target.exists()
    assert len(fake_media_backend.calls) == calls_before


def test_backfill_rejects_output_ancestor_before_candidate_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    video_inventory: SourceInventory,
    fake_media_backend: FakeMediaBackend,
) -> None:
    original = select_frames(
        video_inventory,
        tmp_path / "smart",
        SelectionPolicy(
            mode="smart",
            frame_budget=6,
            resolution_long_edge_cap=1280,
        ),
        media=fake_media_backend,
    )

    def unexpected_analysis(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unsafe ancestor triggered candidate analysis")

    monkeypatch.setattr(
        selection_module,
        "_analyze_source_candidates",
        unexpected_analysis,
    )
    with pytest.raises(ValueError, match="source input"):
        plan_backfill(
            video_inventory,
            original,
            (UncoveredInterval(1.0, 2.0, ()),),
            video_inventory.root.parent,
            media=fake_media_backend,
        )


def test_unexpected_backend_output_aborts_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    video_inventory: SourceInventory,
) -> None:
    backend = FakeMediaBackend(
        timeline=(TimelineFrame(0, 0, 0.0), TimelineFrame(1, 1, 0.25))
    )
    real_extract = backend.extract_video_indices

    def extract_with_extra(
        path: Path,
        indices: tuple[int, ...],
        destination: Path,
        long_edge_cap: int | None,
    ) -> None:
        real_extract(path, indices, destination, long_edge_cap)
        (destination / "unexpected.txt").write_bytes(b"unexpected")

    monkeypatch.setattr(backend, "extract_video_indices", extract_with_extra)
    target = tmp_path / "frames"

    with pytest.raises(RuntimeError, match="unexpected"):
        select_frames(
            video_inventory,
            target,
            SelectionPolicy(
                mode="fixed_fps",
                frame_budget=300,
                resolution_long_edge_cap=1280,
            ),
            media=backend,
        )

    assert not target.exists()
    assert not list(tmp_path.glob("frames.tmp-*"))


def test_photo_selection_natural_sorts_and_never_upscales(tmp_path: Path) -> None:
    root = tmp_path / "photos"
    _write_image(root / "IMG_10.JPG", size=(8, 6), value=10)
    _write_image(root / "IMG_2.JPG", size=(8, 6), value=20)
    _write_image(root / "img_1.JPG", size=(8, 6), value=30)
    inventory = discover_source(root)
    backend = FakeMediaBackend(timeline=())
    manifest = select_frames(
        inventory,
        tmp_path / "frames",
        SelectionPolicy(
            mode="fixed_fps",
            frame_budget=1,
            resolution_long_edge_cap=1280,
        ),
        media=backend,
    )
    assert [frame.source_relative_path for frame in manifest.selected_frames] == [
        "img_1.JPG",
        "IMG_2.JPG",
        "IMG_10.JPG",
    ]
    with Image.open(tmp_path / "frames" / "frame_000000.png") as image:
        assert image.size == (8, 6)


def test_smart_photo_order_uses_exif_only_when_complete(tmp_path: Path) -> None:
    root = tmp_path / "photos"
    root.mkdir()
    rows = (
        ("IMG_1.JPG", "2026:01:03 10:00:00", 30),
        ("IMG_2.JPG", "2026:01:01 10:00:00", 60),
        ("IMG_10.JPG", "2026:01:02 10:00:00", 90),
    )
    for filename, captured_at, value in rows:
        exif = Image.Exif()
        exif[ExifTags.IFD.Exif] = {36867: captured_at}
        Image.new("RGB", (32, 24), (value, value, value)).save(
            root / filename,
            exif=exif,
        )
    inventory = discover_source(root)
    manifest = select_frames(
        inventory,
        tmp_path / "smart-photos",
        SelectionPolicy(
            mode="smart",
            frame_budget=2,
            resolution_long_edge_cap=1280,
        ),
        media=FakeMediaBackend(timeline=()),
    )

    assert [frame.source_relative_path for frame in manifest.frames] == [
        "IMG_2.JPG",
        "IMG_10.JPG",
        "IMG_1.JPG",
    ]
    assert all(frame.source_index is None for frame in manifest.frames)
    assert all(frame.timestamp_s is None for frame in manifest.frames)
    assert manifest.frames[0].frame_id == make_frame_id(
        inventory.digest,
        None,
        None,
        "IMG_2.JPG",
    )


def test_smart_photo_order_falls_back_wholly_when_exif_is_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "photos"
    root.mkdir()
    exif = Image.Exif()
    exif[ExifTags.IFD.Exif] = {36867: "2026:01:01 10:00:00"}
    Image.new("RGB", (24, 18), (80, 80, 80)).save(root / "IMG_10.JPG", exif=exif)
    Image.new("RGB", (24, 18), (40, 40, 40)).save(root / "IMG_2.JPG")
    inventory = discover_source(root)
    manifest = select_frames(
        inventory,
        tmp_path / "smart",
        SelectionPolicy(
            mode="smart",
            frame_budget=2,
            resolution_long_edge_cap=1280,
        ),
        media=FakeMediaBackend(timeline=()),
    )

    assert [frame.source_relative_path for frame in manifest.frames] == [
        "IMG_2.JPG",
        "IMG_10.JPG",
    ]


def test_fixed_vfr_targets_use_nearest_unique_frames_with_stable_ties(
    tmp_path: Path,
    video_inventory: SourceInventory,
) -> None:
    backend = FakeMediaBackend(
        timeline=(
            TimelineFrame(0, 100, 5.00),
            TimelineFrame(1, 110, 5.10),
            TimelineFrame(2, 120, 5.26),
            TimelineFrame(3, 130, 5.49),
            TimelineFrame(4, 140, 5.51),
        )
    )
    manifest = select_frames(
        video_inventory,
        tmp_path / "frames",
        SelectionPolicy(
            mode="fixed_fps",
            frame_budget=10,
            resolution_long_edge_cap=1280,
            fixed_fps=2,
        ),
        media=backend,
    )
    assert [frame.source_index for frame in manifest.selected_frames] == [0, 3]


def test_fixed_fps_matches_targets_against_the_full_source_timeline(
    tmp_path: Path,
    video_inventory: SourceInventory,
) -> None:
    backend = FakeMediaBackend(
        timeline=tuple(
            TimelineFrame(index, index * 1_000, index / 30.0) for index in range(31)
        )
    )
    manifest = select_frames(
        video_inventory,
        tmp_path / "frames",
        SelectionPolicy(
            mode="fixed_fps",
            frame_budget=100,
            resolution_long_edge_cap=1280,
            fixed_fps=4,
            candidate_fps=12,
        ),
        media=backend,
    )
    assert [frame.source_index for frame in manifest.selected_frames] == [
        0,
        7,
        15,
        22,
        30,
    ]
    assert ("video_timeline", video_inventory.root / "room.mov", None) in backend.calls


@pytest.mark.parametrize("count,expects_output", [(3, False), (4, True)])
def test_photo_guardrail_boundary_and_absolute_cap(
    tmp_path: Path,
    count: int,
    expects_output: bool,
) -> None:
    root = tmp_path / f"photos-{count}"
    for index in range(count):
        _write_image(root / f"IMG_{index}.png", size=(2, 2), value=index)
    inventory = discover_source(root)
    backend = FakeMediaBackend(timeline=())
    policy = SelectionPolicy(
        mode="fixed_fps",
        frame_budget=3,
        resolution_long_edge_cap=1280,
    )
    original = select_frames(
        inventory,
        tmp_path / f"all-{count}",
        policy,
        media=backend,
    )
    output = tmp_path / f"bounded-{count}"
    bounded = bound_selection_for_reconstruction(
        inventory,
        original,
        output,
        media=backend,
        max_frames=900,
    )
    assert output.exists() is expects_output
    assert len(bounded.selected_frames) == min(count, 3, 800)


def test_materialization_failure_preserves_previous_target(
    tmp_path: Path,
    photo_inventory: SourceInventory,
) -> None:
    target = tmp_path / "frames"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_bytes(b"keep")
    backend = FakeMediaBackend(timeline=(), fail_after_writes=1)

    with pytest.raises(RuntimeError, match="injected"):
        select_frames(
            photo_inventory,
            target,
            SelectionPolicy(
                mode="fixed_fps",
                frame_budget=300,
                resolution_long_edge_cap=1280,
            ),
            media=backend,
        )

    assert marker.read_bytes() == b"keep"
    assert not list(tmp_path.glob("frames.tmp-*"))


def test_smart_analysis_failure_preserves_previous_target(
    tmp_path: Path,
    photo_inventory: SourceInventory,
) -> None:
    target = tmp_path / "smart-frames"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_bytes(b"keep")
    backend = FakeMediaBackend(timeline=(), fail_after_writes=1)

    with pytest.raises(RuntimeError, match="injected"):
        select_frames(
            photo_inventory,
            target,
            SelectionPolicy(
                mode="smart",
                frame_budget=2,
                resolution_long_edge_cap=1280,
            ),
            media=backend,
        )

    assert marker.read_bytes() == b"keep"
    assert not list(tmp_path.glob("smart-frames.tmp-*"))


def test_selection_refuses_to_write_inside_immutable_source(
    photo_inventory: SourceInventory,
) -> None:
    target = photo_inventory.root / "generated-frames"

    with pytest.raises(ValueError, match="source input"):
        select_frames(
            photo_inventory,
            target,
            SelectionPolicy(
                mode="fixed_fps",
                frame_budget=300,
                resolution_long_edge_cap=1280,
            ),
            media=FakeMediaBackend(timeline=()),
        )

    assert not target.exists()


def test_selection_rejects_output_ancestor_of_immutable_source(
    monkeypatch: pytest.MonkeyPatch,
    photo_inventory: SourceInventory,
) -> None:
    def unexpected_stage(_target: Path) -> Path:
        raise AssertionError("unsafe ancestor passed output validation")

    monkeypatch.setattr(selection_module, "_fresh_stage", unexpected_stage)
    with pytest.raises(ValueError, match="source input"):
        select_frames(
            photo_inventory,
            photo_inventory.root.parent,
            SelectionPolicy(
                mode="fixed_fps",
                frame_budget=2,
                resolution_long_edge_cap=1280,
            ),
            media=FakeMediaBackend(timeline=()),
        )


@pytest.mark.parametrize(
    ("candidate_fps", "candidate_long_edge"),
    [(13, 320), (12, 321)],
)
def test_smart_analysis_rejects_values_above_quality_contract(
    tmp_path: Path,
    video_inventory: SourceInventory,
    candidate_fps: int,
    candidate_long_edge: int,
) -> None:
    with pytest.raises(ValueError, match="candidate"):
        select_frames(
            video_inventory,
            tmp_path / "smart",
            SelectionPolicy(
                mode="smart",
                frame_budget=2,
                resolution_long_edge_cap=1280,
                candidate_fps=candidate_fps,
                candidate_long_edge=candidate_long_edge,
            ),
            media=FakeMediaBackend(timeline=()),
        )


def test_fixed_mode_ignores_irrelevant_smart_candidate_caps(
    tmp_path: Path,
    video_inventory: SourceInventory,
    fake_media_backend: FakeMediaBackend,
) -> None:
    manifest = select_frames(
        video_inventory,
        tmp_path / "fixed",
        SelectionPolicy(
            mode="fixed_fps",
            frame_budget=3,
            resolution_long_edge_cap=1280,
            candidate_fps=30,
            candidate_long_edge=999,
        ),
        media=fake_media_backend,
    )
    assert manifest.effective_mode == "fixed_fps"


def test_invalid_selection_mode_is_rejected_before_io(
    tmp_path: Path,
    video_inventory: SourceInventory,
    fake_media_backend: FakeMediaBackend,
) -> None:
    calls_before = len(fake_media_backend.calls)

    with pytest.raises(ValueError, match="mode"):
        select_frames(
            video_inventory,
            tmp_path / "invalid",
            SelectionPolicy(
                mode="unsupported",  # type: ignore[arg-type]
                frame_budget=3,
                resolution_long_edge_cap=1280,
            ),
            media=fake_media_backend,
        )

    assert len(fake_media_backend.calls) == calls_before


def test_backfill_revalidates_smart_candidate_contract_before_io(
    tmp_path: Path,
    video_inventory: SourceInventory,
    fake_media_backend: FakeMediaBackend,
) -> None:
    original = select_frames(
        video_inventory,
        tmp_path / "smart",
        SelectionPolicy(
            mode="smart",
            frame_budget=6,
            resolution_long_edge_cap=1280,
        ),
        media=fake_media_backend,
    )
    invalid = replace(
        original,
        policy=replace(original.policy, candidate_fps=13),
    )
    calls_before = len(fake_media_backend.calls)

    with pytest.raises(ValueError, match="candidate_fps"):
        plan_backfill(
            video_inventory,
            invalid,
            (UncoveredInterval(1.0, 2.0, ()),),
            tmp_path / "backfill",
            media=fake_media_backend,
        )

    assert len(fake_media_backend.calls) == calls_before


def test_publish_failure_restores_previous_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    photo_inventory: SourceInventory,
) -> None:
    target = tmp_path / "frames"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_bytes(b"keep")
    backend = FakeMediaBackend(timeline=())
    real_rename = selection_module.os.rename
    rename_count = 0

    def fail_stage_promotion(source: Path, destination: Path) -> None:
        nonlocal rename_count
        rename_count += 1
        if rename_count == 2:
            raise OSError("injected publish failure")
        real_rename(source, destination)

    monkeypatch.setattr(selection_module.os, "rename", fail_stage_promotion)

    with pytest.raises(OSError, match="injected publish"):
        select_frames(
            photo_inventory,
            target,
            SelectionPolicy(
                mode="fixed_fps",
                frame_budget=300,
                resolution_long_edge_cap=1280,
            ),
            media=backend,
        )

    assert marker.read_bytes() == b"keep"
    assert not list(tmp_path.glob("frames.tmp-*"))
    assert not list(tmp_path.glob("frames.backup-*"))


def test_backup_cleanup_failure_rolls_back_published_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    photo_inventory: SourceInventory,
) -> None:
    target = tmp_path / "frames"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_bytes(b"keep")
    backend = FakeMediaBackend(timeline=())
    real_remove = selection_module._remove_path
    failed_cleanup = False

    def fail_first_backup_cleanup(path: Path) -> None:
        nonlocal failed_cleanup
        if ".backup-" in path.name and not failed_cleanup:
            failed_cleanup = True
            raise PermissionError("injected backup cleanup failure")
        real_remove(path)

    monkeypatch.setattr(selection_module, "_remove_path", fail_first_backup_cleanup)

    with pytest.raises(PermissionError, match="injected backup cleanup"):
        select_frames(
            photo_inventory,
            target,
            SelectionPolicy(
                mode="fixed_fps",
                frame_budget=300,
                resolution_long_edge_cap=1280,
            ),
            media=backend,
        )

    assert marker.read_bytes() == b"keep"
    assert not list(tmp_path.glob("frames.backup-*"))


def test_frame_id_and_manifest_bytes_are_canonical(
    tmp_path: Path,
    photo_inventory: SourceInventory,
) -> None:
    backend = FakeMediaBackend(timeline=())
    policy = SelectionPolicy(
        mode="fixed_fps",
        frame_budget=300,
        resolution_long_edge_cap=1280,
    )
    manifest = select_frames(
        photo_inventory,
        tmp_path / "frames",
        policy,
        media=backend,
    )
    expected_id = hashlib.sha256(
        f"{photo_inventory.digest}|None|None|IMG_0001.JPG".encode("utf-8")
    ).hexdigest()[:24]
    assert (
        make_frame_id(
            photo_inventory.digest,
            None,
            None,
            "IMG_0001.JPG",
        )
        == expected_id
    )
    first = write_selection_manifest(tmp_path / "first.json", manifest)
    second = write_selection_manifest(tmp_path / "second.json", manifest)
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8"))["schema_version"] == 1


@pytest.mark.parametrize(
    "rotation,expected_filter",
    [
        (0, None),
        (90, "transpose=clock"),
        (180, "rotate=PI"),
        (270, "transpose=cclock"),
    ],
)
def test_ffmpeg_argv_owns_rotation_once_and_keeps_paths_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rotation: int,
    expected_filter: str | None,
) -> None:
    captured: list[list[str]] = []
    scripts: list[str] = []
    probe_payload = {
        "streams": [
            {
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "tags": {"rotate": str(rotation)},
                "side_data_list": [],
            }
        ]
    }

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(args)
        assert kwargs.get("shell") in {None, False}
        if args[0] == "ffprobe":
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps(probe_payload),
                stderr="",
            )
        script_index = args.index("-filter_script:v") + 1
        scripts.append(Path(args[script_index]).read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = FfmpegMediaBackend()
    video = tmp_path / "folder with spaces" / "room.mov"
    destination = tmp_path / "output with spaces"
    backend.extract_video_indices(video, (0, 5), destination, 1280)

    ffmpeg_argv = captured[-1]
    assert ffmpeg_argv.index("-noautorotate") < ffmpeg_argv.index("-i")
    assert str(video) in ffmpeg_argv
    assert ffmpeg_argv[ffmpeg_argv.index("-vsync") + 1] == "0"
    assert "-start_number" in ffmpeg_argv
    assert "frame_%06d.png" in ffmpeg_argv[-1]
    assert "scale=" in scripts[-1]
    if expected_filter is None:
        assert "transpose=" not in scripts[-1] and "rotate=" not in scripts[-1]
    else:
        assert expected_filter in scripts[-1]


def test_ffprobe_prefers_display_matrix_rotation_and_scale_does_not_upsize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts: list[str] = []
    payload = {
        "streams": [
            {
                "width": 640,
                "height": 480,
                "avg_frame_rate": "30000/1001",
                "tags": {"rotate": "90"},
                "side_data_list": [{"rotation": 270}],
            }
        ]
    }

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[0] == "ffprobe":
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps(payload),
                stderr="",
            )
        scripts.append(
            Path(args[args.index("-filter_script:v") + 1]).read_text(encoding="utf-8")
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = FfmpegMediaBackend()
    probe = backend.probe_video(tmp_path / "room.mov")
    assert probe.rotation_degrees == 270
    assert probe.rotation_source == "display_matrix"
    backend.extract_video_indices(
        tmp_path / "room.mov",
        (0,),
        tmp_path / "frames",
        1280,
    )
    assert "scale=" not in scripts[-1]
    assert "transpose=clock" in scripts[-1]


def test_candidate_timeline_uses_anchored_nearest_sampling_at_twelve_fps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "streams": [
            {
                "width": 640,
                "height": 480,
                "avg_frame_rate": "30/1",
                "tags": {},
                "side_data_list": [],
            }
        ],
        "frames": [
            {
                "best_effort_timestamp": str(index),
                "best_effort_timestamp_time": str(index / 30.0),
            }
            for index in range(31)
        ],
    }

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    timeline = FfmpegMediaBackend().video_timeline(tmp_path / "room.mov", 12)
    assert [frame.source_index for frame in timeline] == [
        0,
        2,
        5,
        7,
        10,
        12,
        15,
        17,
        20,
        22,
        25,
        27,
        30,
    ]


@pytest.mark.integration
def test_display_matrix_rotation_matches_ffmpeg_autorotation_pixels(
    tmp_path: Path,
) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg binaries are unavailable")

    pattern = tmp_path / "pattern.png"
    image = Image.new("RGB", (8, 6))
    pixels = image.load()
    for y in range(6):
        for x in range(8):
            pixels[x, y] = (x * 25, y * 35, (x + y) * 15)
    image.save(pattern)
    base = tmp_path / "base.mp4"
    rotated = tmp_path / "rotated.mov"
    auto = tmp_path / "auto.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            str(pattern),
            "-t",
            "1",
            "-r",
            "1",
            "-pix_fmt",
            "yuv444p",
            str(base),
        ],
        check=True,
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-display_rotation:v:0",
                "90",
                "-i",
                str(base),
                "-c",
                "copy",
                str(rotated),
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        pytest.skip("FFmpeg does not support -display_rotation")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(rotated),
            "-frames:v",
            "1",
            str(auto),
        ],
        check=True,
    )
    manual_dir = tmp_path / "manual"
    FfmpegMediaBackend().extract_video_indices(rotated, (0,), manual_dir, None)
    with Image.open(auto).convert("RGB") as auto_image:
        with Image.open(manual_dir / "frame_000000.png").convert("RGB") as manual:
            assert ImageChops.difference(auto_image, manual).getbbox() is None

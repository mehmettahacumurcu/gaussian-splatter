from pathlib import Path

import pytest

from backend.static_pipeline.sources import copy_input_read_only, discover_source


def test_discovers_exactly_one_root_video_and_hashes_bytes(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    video = root / "room.mov"
    video.write_bytes(b"first")
    first = discover_source(root)
    video.write_bytes(b"second")
    second = discover_source(root)
    assert first.kind == "video"
    assert first.digest != second.digest


@pytest.mark.parametrize("layout", ["root_images", "images_dir"])
def test_discovers_photo_sets_in_supported_locations(tmp_path: Path, layout: str) -> None:
    root = tmp_path / layout
    image_root = root if layout == "root_images" else root / "images"
    image_root.mkdir(parents=True)
    (image_root / "IMG_0001.JPG").write_bytes(b"one")
    inventory = discover_source(root)
    assert inventory.kind == "photo_set"
    assert [item.relative_path for item in inventory.media_files] == (
        ["IMG_0001.JPG"] if layout == "root_images" else ["images/IMG_0001.JPG"]
    )


@pytest.mark.parametrize("mutator", ["multiple_videos", "mixed", "both_image_locations", "empty"])
def test_rejects_ambiguous_or_empty_layouts(tmp_path: Path, mutator: str) -> None:
    root = tmp_path / mutator
    root.mkdir()
    if mutator in {"multiple_videos", "mixed"}:
        (root / "a.mp4").write_bytes(b"a")
    if mutator == "multiple_videos":
        (root / "b.mov").write_bytes(b"b")
    if mutator in {"mixed", "both_image_locations"}:
        (root / "a.jpg").write_bytes(b"a")
    if mutator == "both_image_locations":
        (root / "images").mkdir()
        (root / "images" / "b.png").write_bytes(b"b")
    with pytest.raises(ValueError):
        discover_source(root)


def test_copy_does_not_mutate_source_and_verifies_inventory(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "frame.jpg").write_bytes(b"pixels")
    before = discover_source(source)
    copied = copy_input_read_only(source, tmp_path / "local")
    after = discover_source(source)
    assert copied.digest == before.digest == after.digest


def test_inventory_digest_covers_non_media_files(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    (root / "room.mp4").write_bytes(b"video")
    note = root / "notes.txt"
    note.write_bytes(b"first")
    first = discover_source(root)
    note.write_bytes(b"second")
    second = discover_source(root)
    assert first.digest != second.digest


@pytest.mark.parametrize(
    "relative_path",
    ["nested/frame.jpg", "images/nested/frame.png", "images/clip.mov"],
)
def test_rejects_supported_media_outside_exact_locations(
    tmp_path: Path, relative_path: str
) -> None:
    root = tmp_path / "capture"
    media = root / relative_path
    media.parent.mkdir(parents=True)
    media.write_bytes(b"media")
    with pytest.raises(ValueError):
        discover_source(root)


def test_copy_refuses_to_merge_or_replace_an_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "frame.jpg").write_bytes(b"pixels")
    target = tmp_path / "local"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_bytes(b"keep")

    with pytest.raises(FileExistsError):
        copy_input_read_only(source, target)

    assert marker.read_bytes() == b"keep"

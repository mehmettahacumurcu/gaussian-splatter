import hashlib
import json
import shutil
from pathlib import Path

import pytest

from backend.static_pipeline import sources as source_module
from backend.static_pipeline.sources import copy_input_read_only, discover_source


def _symlink_or_skip(link: Path, target: Path, *, is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=is_directory)
    except OSError:
        pytest.skip("symlink creation is unavailable")


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


def test_manifest_digest_matches_required_canonical_payload(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    (root / "room.mp4").write_bytes(b"video")
    (root / "notes.txt").write_bytes(b"notes")

    inventory = discover_source(root)
    payload = {
        "schema_version": 1,
        "kind": inventory.kind,
        "files": [
            {
                "relative_path": item.relative_path,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in inventory.all_files
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    assert inventory.digest == hashlib.sha256(encoded).hexdigest()
    json.dumps(inventory.to_manifest_dict())


def test_rejects_destination_inside_source_tree(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "frame.jpg").write_bytes(b"pixels")

    with pytest.raises(ValueError, match="inside"):
        copy_input_read_only(source, source / "local")

    assert not (source / "local").exists()


def test_copy_detects_corruption_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "frame.jpg").write_bytes(b"pixels")
    target = tmp_path / "local"
    real_copytree = shutil.copytree

    def corrupt_copy(src: Path, dst: Path, **kwargs: object) -> Path:
        copied = real_copytree(src, dst, **kwargs)
        (Path(dst) / "frame.jpg").write_bytes(b"corrupt")
        return Path(copied)

    monkeypatch.setattr(source_module.shutil, "copytree", corrupt_copy)

    with pytest.raises(ValueError, match="does not match"):
        copy_input_read_only(source, target)

    assert not target.exists()
    assert not list(tmp_path.glob("local.tmp-*"))


def test_copy_preserves_links_instead_of_following_a_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input"
    source.mkdir()
    frame = source / "frame.jpg"
    frame.write_bytes(b"pixels")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"secret")
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    else:
        probe.unlink()

    real_copytree = shutil.copytree
    observed_symlinks: list[object] = []

    def swap_and_copy(src: Path, dst: Path, **kwargs: object) -> Path:
        frame.unlink()
        frame.symlink_to(outside)
        observed_symlinks.append(kwargs.get("symlinks"))
        return Path(real_copytree(src, dst, **kwargs))

    monkeypatch.setattr(source_module.shutil, "copytree", swap_and_copy)

    with pytest.raises(ValueError, match="symlink"):
        copy_input_read_only(source, tmp_path / "local")

    assert observed_symlinks == [True]
    assert not list(tmp_path.glob("local.tmp-*"))


def test_discovery_rejects_external_symlink(tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    (root / "frame.jpg").write_bytes(b"pixels")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    _symlink_or_skip(root / "outside.txt", outside)

    with pytest.raises(ValueError, match="outside"):
        discover_source(root)


def test_discovery_rejects_broken_symlink(tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    (root / "frame.jpg").write_bytes(b"pixels")
    _symlink_or_skip(root / "broken.txt", root / "missing.txt")

    with pytest.raises(ValueError, match="broken|cyclic"):
        discover_source(root)


def test_discovery_rejects_symlink_cycle(tmp_path: Path) -> None:
    root = tmp_path / "input"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "frame.jpg").write_bytes(b"pixels")
    _symlink_or_skip(nested / "loop", root, is_directory=True)

    with pytest.raises(ValueError, match="cycle|cyclic"):
        discover_source(root)


def test_copy_rebases_an_absolute_symlink_that_stays_inside_source(tmp_path: Path) -> None:
    root = tmp_path / "input"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"pixels")
    _symlink_or_skip(root / "frame.jpg", payload.resolve())
    before = discover_source(root)

    copied = copy_input_read_only(root, tmp_path / "local")

    assert copied.digest == before.digest
    assert (copied.root / "frame.jpg").read_bytes() == b"pixels"


def test_atomic_promotion_loses_a_destination_race_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "frame.jpg").write_bytes(b"pixels")
    target = tmp_path / "local"
    real_promote = source_module._atomic_promote_no_replace

    def racing_promote(staged: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "keep.txt").write_bytes(b"keep")
        real_promote(staged, destination)

    monkeypatch.setattr(source_module, "_atomic_promote_no_replace", racing_promote)

    with pytest.raises(FileExistsError):
        copy_input_read_only(source, target)

    assert (target / "keep.txt").read_bytes() == b"keep"
    assert not list(tmp_path.glob("local.tmp-*"))

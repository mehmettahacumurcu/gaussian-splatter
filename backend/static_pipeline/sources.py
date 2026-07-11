from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Literal

from .contracts import SourceFile, SourceInventory


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
_SUPPORTED_MEDIA_SUFFIXES = VIDEO_SUFFIXES | IMAGE_SUFFIXES
_HASH_CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _iter_files(input_root: Path) -> tuple[tuple[str, Path], ...]:
    """Walk logical paths safely, following only links contained by the root."""

    root = input_root.resolve(strict=True)
    discovered: list[tuple[str, Path]] = []

    def visit(directory: Path, relative_dir: PurePosixPath, ancestors: frozenset[Path]) -> None:
        resolved_directory = directory.resolve(strict=True)
        if not _is_within(resolved_directory, root):
            raise ValueError(f"symlink resolves outside input root: {directory}")
        if resolved_directory in ancestors:
            raise ValueError(f"symlink cycle detected: {directory}")

        next_ancestors = ancestors | {resolved_directory}
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = relative_dir / entry.name
            try:
                resolved = entry.resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as exc:
                raise ValueError(f"broken or cyclic symlink: {entry}") from exc

            if entry.is_symlink() and not _is_within(resolved, root):
                raise ValueError(f"symlink resolves outside input root: {entry}")

            if entry.is_dir():
                visit(entry, relative, next_ancestors)
            elif entry.is_file():
                discovered.append((relative.as_posix(), entry))

    visit(input_root, PurePosixPath(), frozenset())
    return tuple(sorted(discovered, key=lambda item: item[0]))


def _classify_media(
    all_files: tuple[SourceFile, ...],
) -> tuple[Literal["video", "photo_set"], tuple[SourceFile, ...]]:
    root_videos: list[SourceFile] = []
    root_images: list[SourceFile] = []
    images_dir_images: list[SourceFile] = []

    for item in all_files:
        relative = PurePosixPath(item.relative_path)
        suffix = relative.suffix.lower()
        if suffix not in _SUPPORTED_MEDIA_SUFFIXES:
            continue

        parts = relative.parts
        if len(parts) == 1:
            if suffix in VIDEO_SUFFIXES:
                root_videos.append(item)
            else:
                root_images.append(item)
            continue

        if len(parts) == 2 and parts[0] == "images" and suffix in IMAGE_SUFFIXES:
            images_dir_images.append(item)
            continue

        raise ValueError(
            "supported media must be one root video, root images, or images directly under images/"
        )

    populated_layouts = sum(
        bool(items) for items in (root_videos, root_images, images_dir_images)
    )
    if populated_layouts != 1:
        raise ValueError("input must contain exactly one unambiguous supported media layout")

    if root_videos:
        if len(root_videos) != 1:
            raise ValueError("video layout requires exactly one root video")
        return "video", tuple(root_videos)

    media_files = root_images or images_dir_images
    return "photo_set", tuple(media_files)


def discover_source(input_root: Path) -> SourceInventory:
    """Validate an input layout and build a canonical, content-addressed inventory."""

    requested_root = Path(input_root)
    if not requested_root.exists():
        raise ValueError(f"input root does not exist: {requested_root}")
    if not requested_root.is_dir():
        raise ValueError(f"input root is not a directory: {requested_root}")

    root = requested_root.resolve(strict=True)
    source_files = tuple(
        SourceFile(
            relative_path=relative_path,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for relative_path, path in _iter_files(requested_root)
    )
    kind, media_files = _classify_media(source_files)

    digest_payload = {
        "schema_version": 1,
        "kind": kind,
        "files": [
            {
                "relative_path": item.relative_path,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in source_files
        ],
    }
    canonical = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return SourceInventory(
        schema_version=1,
        root=root,
        kind=kind,
        media_files=media_files,
        all_files=source_files,
        digest=digest,
    )


def _file_identity(files: tuple[SourceFile, ...]) -> tuple[tuple[str, int, str], ...]:
    return tuple((item.relative_path, item.size_bytes, item.sha256) for item in files)


def copy_input_read_only(source: Path, destination: Path) -> SourceInventory:
    """Copy a source tree transactionally and verify every copied byte."""

    source_inventory = discover_source(Path(source))
    target = Path(destination)
    target_parent = target.parent.resolve(strict=False)
    resolved_target = target_parent / target.name

    if os.path.lexists(target):
        raise FileExistsError(f"destination already exists: {target}")
    if _is_within(resolved_target, source_inventory.root):
        raise ValueError("destination must not be inside the source input")

    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    if os.path.lexists(staged):
        raise FileExistsError(f"staging destination already exists: {staged}")

    try:
        shutil.copytree(source_inventory.root, staged, copy_function=shutil.copy2)
        staged_inventory = discover_source(staged)
        if staged_inventory.kind != source_inventory.kind or _file_identity(
            staged_inventory.all_files
        ) != _file_identity(source_inventory.all_files):
            raise ValueError("copied input inventory does not match source inventory")

        if os.path.lexists(target):
            raise FileExistsError(f"destination appeared while copying: {target}")
        os.replace(staged, target)
        return discover_source(target)
    finally:
        if os.path.lexists(staged):
            shutil.rmtree(staged)

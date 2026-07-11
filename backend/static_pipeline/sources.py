from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
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


def _copy_file_no_follow(source: str, destination: str) -> str:
    """Copy a regular file without dereferencing a last-moment symlink swap."""

    return shutil.copy2(source, destination, follow_symlinks=False)


def _rebase_absolute_internal_symlinks(staged: Path, source_root: Path) -> None:
    """Point preserved absolute internal links at their staged counterparts."""

    for current_root, directory_names, file_names in os.walk(staged, followlinks=False):
        current = Path(current_root)
        for name in (*directory_names, *file_names):
            link = current / name
            if not link.is_symlink():
                continue

            raw_target = os.readlink(link)
            if not os.path.isabs(raw_target):
                continue
            if os.name == "nt" and raw_target.startswith("\\\\?\\"):
                raw_target = (
                    f"\\\\{raw_target[8:]}"
                    if raw_target.startswith("\\\\?\\UNC\\")
                    else raw_target[4:]
                )
            try:
                resolved_target = Path(raw_target).resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as exc:
                raise ValueError(f"broken or cyclic symlink: {link}") from exc
            if not _is_within(resolved_target, source_root):
                raise ValueError(f"symlink resolves outside input root: {link}")

            staged_target = staged / resolved_target.relative_to(source_root)
            if not os.path.lexists(staged_target):
                raise ValueError(f"internal symlink target was not staged: {link}")
            relative_target = os.path.relpath(staged_target, start=link.parent)
            target_is_directory = resolved_target.is_dir()
            link.unlink()
            link.symlink_to(relative_target, target_is_directory=target_is_directory)


def _atomic_promote_no_replace(staged: Path, destination: Path) -> None:
    """Atomically rename a staged directory only when destination is absent."""

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as exc:
            raise RuntimeError(
                "atomic no-replace directory promotion requires renameat2 on Linux"
            ) from exc

        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            os.fsencode(staged),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
        if result == 0:
            return

        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_number, os.strerror(error_number), str(destination)
            )
        if error_number == errno.ENOSYS:
            raise RuntimeError(
                "the Linux runtime does not support atomic no-replace promotion"
            )
        raise OSError(error_number, os.strerror(error_number), str(destination))

    try:
        # Windows rename is already no-replace. Static notebooks run on Linux;
        # this branch keeps local development deterministic on Windows as well.
        os.rename(staged, destination)
    except OSError as exc:
        if os.path.lexists(destination):
            raise FileExistsError(
                errno.EEXIST, os.strerror(errno.EEXIST), str(destination)
            ) from exc
        raise


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
        shutil.copytree(
            source_inventory.root,
            staged,
            symlinks=True,
            copy_function=_copy_file_no_follow,
        )
        _rebase_absolute_internal_symlinks(staged, source_inventory.root)
        staged_inventory = discover_source(staged)
        if staged_inventory.kind != source_inventory.kind or _file_identity(
            staged_inventory.all_files
        ) != _file_identity(source_inventory.all_files):
            raise ValueError("copied input inventory does not match source inventory")

        _atomic_promote_no_replace(staged, target)
        return discover_source(target)
    finally:
        if os.path.lexists(staged):
            shutil.rmtree(staged)

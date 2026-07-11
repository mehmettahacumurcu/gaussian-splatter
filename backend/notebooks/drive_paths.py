from __future__ import annotations

import re
from pathlib import PurePosixPath


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")


def normalize_input_folder(raw: str) -> str:
    value = raw.strip()
    if value == "MyDrive":
        raise ValueError("Choose a folder below MyDrive, not the Drive root")
    if value.startswith("MyDrive/"):
        value = value[len("MyDrive/") :]
    if not value or "\\" in value or _CONTROL.search(value):
        raise ValueError("Input must be a non-empty MyDrive-relative folder")
    if value.startswith("/") or value.endswith("/") or _DRIVE_LETTER.match(value):
        raise ValueError("Absolute runtime paths are not accepted")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Path contains an empty or traversal segment")
    if parts[-1].endswith("_result"):
        raise ValueError("Choose the input folder, not a result folder")
    canonical = PurePosixPath(*parts).as_posix()
    if canonical in {"", "."}:
        raise ValueError("Choose a folder below MyDrive")
    return canonical


def derive_result_folder(input_folder: str) -> str:
    canonical = normalize_input_folder(input_folder)
    path = PurePosixPath(canonical)
    return path.with_name(f"{path.name}_result").as_posix()

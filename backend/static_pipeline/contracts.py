from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SourceInventory:
    schema_version: int
    root: Path
    kind: Literal["video", "photo_set"]
    media_files: tuple[SourceFile, ...]
    all_files: tuple[SourceFile, ...]
    digest: str

    def to_manifest_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["root"] = str(self.root)
        return value

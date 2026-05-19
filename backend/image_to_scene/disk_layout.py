"""Indexed-file disk conventions, Python port of image-blaster's
.claude/scripts/asset-pipeline/request-metadata.mjs.

Visible files: N-slug.ext   (e.g., 0-world.ply)
Hidden sidecar: .N-slug-request.json
Multiple files in one generation share the same index N.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexedName:
    index: int
    slug: str
    extension: str  # includes leading dot, e.g., ".ply"
    hidden: bool


_VISIBLE = re.compile(r"^(\d+)-([a-z0-9][a-z0-9-]*)(\.[a-zA-Z0-9]+)$")
_HIDDEN = re.compile(r"^\.(\d+)-([a-z0-9][a-z0-9-]*)(\.[a-zA-Z0-9]+)$")
_SLUG = re.compile(r"[^a-z0-9]+")


def parse_indexed_name(name: str) -> IndexedName | None:
    """Parse a filename. Returns None if it does not match the indexed form."""
    if name.startswith("."):
        m = _HIDDEN.match(name)
        if not m:
            return None
        idx, slug, ext = m.groups()
        return IndexedName(int(idx), slug, ext, hidden=True)

    m = _VISIBLE.match(name)
    if not m:
        return None
    idx, slug, ext = m.groups()
    return IndexedName(int(idx), slug, ext, hidden=False)


def next_index(directory: Path, slug: str) -> int:
    """Return one past the highest existing index for files of the given slug."""
    if not directory.exists():
        return 0
    highest = -1
    for entry in directory.iterdir():
        parsed = parse_indexed_name(entry.name)
        if parsed is None:
            continue
        # Hidden request files have slug like "world-request"; strip suffix to match base slug.
        base_slug = parsed.slug.split("-request")[0] if parsed.hidden else parsed.slug
        if base_slug == slug:
            highest = max(highest, parsed.index)
    return highest + 1


def slug_safe(text: str) -> str:
    """Normalize a string into a stable slug. Raises ValueError if empty result."""
    lower = text.strip().lower()
    cleaned = _SLUG.sub("-", lower).strip("-")
    if not cleaned:
        raise ValueError(f"slug_safe({text!r}) produced empty string")
    return cleaned


from datetime import datetime, timezone
import json


@dataclass(frozen=True)
class WorldEnvelope:
    slug: str
    root: Path
    source: Path
    output_world: Path
    project_json: Path
    image_json: Path


def ensure_envelope(worlds_root: Path, slug: str) -> WorldEnvelope:
    """Create or open the worlds/<slug>/ tree. Idempotent."""
    slug = slug_safe(slug)
    root = worlds_root / slug
    source = root / "source"
    output_world = root / "output" / "world"
    project_json = root / "project.json"
    image_json = root / "image.json"

    for d in (root, source, output_world):
        d.mkdir(parents=True, exist_ok=True)

    if not project_json.exists():
        now = datetime.now(timezone.utc).isoformat()
        project_json.write_text(
            json.dumps({
                "slug": slug,
                "display_name": slug.replace("-", " ").title(),
                "created_at": now,
                "updated_at": now,
            }, indent=2),
            encoding="utf-8",
        )

    return WorldEnvelope(slug, root, source, output_world, project_json, image_json)


def artifact_path(directory: Path, index: int, slug: str, extension: str) -> Path:
    """Build a visible indexed artifact path: <directory>/<index>-<slug><ext>."""
    return directory / f"{index}-{slug}{extension}"


def request_path(directory: Path, index: int, slug: str) -> Path:
    """Build a hidden request metadata path: <directory>/.<index>-<slug>-request.json."""
    return directory / f".{index}-{slug}-request.json"

"""Find the splat, the COLMAP model and the photos of a training run in a folder.

Handles the layouts the app produces: a Spirula/Colab run folder synced from
Drive (``outputs/training/**/splat.ply`` + ``diagnostic_dataset/sparse/0`` +
``diagnostic_dataset/images``), a final-pipeline result folder, and a local
scene (``output/ply/*.ply`` + ``colmap/sparse/0`` + ``frames``).
"""
from __future__ import annotations

import os
from pathlib import Path

from .colmap import has_model

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_MAX_DEPTH = 6
_SKIP_DIRS = {"node_modules", ".git", "__pycache__", "coverage"}


class DetectError(ValueError):
    """Turkish message for the UI."""


def _walk(root: Path, max_depth: int = _MAX_DEPTH):
    """Directories under ``root`` (breadth-limited, hidden and tool folders skipped)."""
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS]
        if len(here.parts) - root_depth >= max_depth:
            dirnames[:] = []
        yield here, filenames


def _is_splat(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".ply") and not name.startswith("points3d") and "coverage" not in name \
        and "heat" not in name


def find_sparse(root: Path, max_depth: int = _MAX_DEPTH) -> list[Path]:
    found = [d for d, _ in _walk(root, max_depth) if has_model(d)]
    # A model in sparse/0 beats loose copies; ties go to the larger model.
    return sorted(found, key=lambda d: (d.name != "0", -_model_size(d), str(d)))


def _model_size(d: Path) -> int:
    for name in ("images.bin", "images.txt"):
        if (d / name).is_file():
            return (d / name).stat().st_size
    return 0


def find_splats(root: Path) -> list[Path]:
    plys = [d / f for d, files in _walk(root) for f in files if _is_splat(Path(f))]
    # The trained splat is by far the largest .ply in a run folder.
    return sorted(plys, key=lambda p: p.stat().st_size, reverse=True)


def find_images(sparse: Path, names: list[str]) -> Path | None:
    """The folder that holds the camera images (checked by the first image name)."""
    if not names:
        return None
    probe = Path(names[0])
    for base in [sparse, *list(sparse.parents)[:3]]:
        for sub in ("images", "frames", "input", "images_4k", ""):
            cand = base / sub if sub else base
            if (cand / probe).is_file() or (cand / probe.name).is_file():
                return cand
    return None


def detect(path: str = "", ply_path: str = "", sparse_dir: str = "") -> tuple[Path, Path]:
    """(splat .ply, COLMAP model folder) from a run folder and/or explicit paths."""
    root = Path(path).expanduser() if path.strip() else None
    if root is not None and not root.exists():
        raise DetectError(f"Klasör bulunamadı: {root}")
    if root is not None and root.is_file():
        if root.suffix.lower() != ".ply":
            raise DetectError("Bir klasör ya da .ply dosyası seç")
        ply_path = ply_path or str(root)
        root = None

    if ply_path.strip():
        ply = Path(ply_path).expanduser()
        if not ply.is_file():
            raise DetectError(f"PLY bulunamadı: {ply}")
    else:
        if root is None:
            raise DetectError("Klasör ya da PLY yolu gerekli")
        plys = find_splats(root)
        if not plys:
            raise DetectError(f"Klasörde splat .ply bulunamadı: {root}")
        ply = plys[0]

    if sparse_dir.strip():
        sparse = Path(sparse_dir).expanduser()
        if not has_model(sparse):
            raise DetectError(f"Bu klasörde COLMAP modeli (cameras/images .bin/.txt) yok: {sparse}")
    else:
        # The given folder first, then a few levels up with a shallow search: the
        # final-pipeline layout keeps the model next to the per-trainer folders
        # (ROOM2/sfm/sparse_0 + ROOM2/spirula_high/splat.ply). Shallow, so a
        # Drive root is never crawled; a model of another run is caught later by
        # the alignment check.
        sparse = None
        start = root if root else ply.parent
        bases = ([(root, _MAX_DEPTH)] if root else []) + [(p, 3) for p in [start, *start.parents][:4] if p != root]
        for base, depth in bases:
            models = find_sparse(base, depth)
            if models:
                sparse = models[0]
                break
        if sparse is None:
            raise DetectError("COLMAP modeli bulunamadı (sparse/0). Model klasörünü ayrıca belirt.")
    return ply, sparse

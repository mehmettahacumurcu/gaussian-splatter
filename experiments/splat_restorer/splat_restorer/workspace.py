"""Folder layout. Durable outputs live on Drive; scratch work on local disk.

Drive (``root``)::

    split.json                    scene split (written once, then reused)
    pairs/<scene>.tar             all pairs of one scene (atomic, = done marker)
    splats/<scene>/<recipe>.pt    degraded splats of *test* scenes
    runs/<run>/...                restorer / baseline checkpoints and logs
    results/...                   evaluation tables, contact sheets

Local (``local``)::

    raw/<scene>/                  downloaded scene, deleted after use
    pairs/                        extracted pair tars for training / eval
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Workspace:
    root: Path
    local: Path

    def __post_init__(self):
        self.root = Path(self.root)
        self.local = Path(self.local)
        for d in (self.root, self.pairs_dir, self.splats_dir, self.runs_dir, self.results_dir, self.local):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def split_path(self) -> Path:
        return self.root / "split.json"

    @property
    def pairs_dir(self) -> Path:
        return self.root / "pairs"

    @property
    def splats_dir(self) -> Path:
        return self.root / "splats"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def pair_tar(self, scene: str) -> Path:
        return self.pairs_dir / f"{scene}.tar"

    def raw_dir(self, scene: str) -> Path:
        return self.local / "raw" / scene

    @property
    def local_pairs(self) -> Path:
        return self.local / "pairs"

    def splat_path(self, scene: str, recipe: str) -> Path:
        return self.splats_dir / scene / f"{recipe}.pt"


def atomic_copy(src: Path, dst: Path) -> None:
    """Copy to ``dst`` via a temp name so a crash never leaves a half file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)

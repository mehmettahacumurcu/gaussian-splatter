"""Coverage results on disk: ``<data>/coverage/<id>/{meta.json, views.json, heat.ply, coverage.npz}``.

The trained splat itself is not copied (it can be several GB); ``meta.json``
keeps its path and the viewer streams it from there.
"""
from __future__ import annotations

import json
import re
import secrets
import shutil
from pathlib import Path

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,80}$")


class CoverageStore:
    def __init__(self, data_root: str | Path):
        self.root = Path(data_root) / "coverage"

    def new_id(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "kapsama"
        return f"{slug}-{secrets.token_hex(3)}"

    def dir(self, result_id: str) -> Path | None:
        if not _ID_RE.match(result_id):
            return None
        return self.root / result_id

    def get(self, result_id: str) -> dict | None:
        d = self.dir(result_id)
        if d is None or not (d / "meta.json").is_file():
            return None
        try:
            return json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def views(self, result_id: str) -> list[dict]:
        d = self.dir(result_id)
        try:
            return json.loads((d / "views.json").read_text(encoding="utf-8")) if d else []
        except (OSError, json.JSONDecodeError):
            return []

    def list(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        metas = [m for d in self.root.iterdir() if d.is_dir() and (m := self.get(d.name))]
        return sorted(metas, key=lambda m: m.get("created_at", 0), reverse=True)

    def heat_path(self, result_id: str) -> Path | None:
        d = self.dir(result_id)
        return d / "heat.ply" if d and (d / "heat.ply").is_file() else None

    def delete(self, result_id: str) -> bool:
        d = self.dir(result_id)
        if d is None or not d.is_dir():
            return False
        shutil.rmtree(d)
        return True

    def tmp_dir(self, result_id: str) -> Path:
        tmp = self.root / f".{result_id}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        return tmp

    def publish(self, result_id: str, tmp: Path) -> None:
        final = self.root / result_id
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)

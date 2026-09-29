"""On-disk layout for composer assets, scene documents and exports.

    <data_root>/compose/assets/<asset_id>.{ply,glb}  + <asset_id>.json (Asset)
    <data_root>/compose/scenes/<scene_id>.json       (SceneDoc)
    <data_root>/compose/exports/<scene_id>/          (merged.ply, meshes/, scene.json)
    <data_root>/compose/exports/<scene_id>.zip

Pipeline results (<data_root>/<scene>/output/ply/*.ply) appear as read-only
assets with id ``scene__<scene>``. Ids are generated here and checked against
``ID_PATTERN`` before they touch a path, so user input never becomes a path.
"""
from __future__ import annotations

import os
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import BinaryIO

from .glb import GlbFormatError, validate_glb
from .models import ID_PATTERN, Asset, SceneDoc, SceneObject, SceneSummary
from .plyio import PlyFormatError, validate_ply

_ID_RE = re.compile(ID_PATTERN)
PIPELINE_PREFIX = "scene__"
_KIND_BY_EXT = {".ply": "splat", ".glb": "mesh"}
_EXT_BY_KIND = {"splat": ".ply", "mesh": ".glb"}


class AssetError(ValueError):
    """Upload rejected (wrong type or unreadable content)."""


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class ComposeStore:
    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        self.root = self.data_root / "compose"
        self.assets_dir = self.root / "assets"
        self.scenes_dir = self.root / "scenes"
        self.exports_dir = self.root / "exports"

    def _ensure_dirs(self) -> None:
        for d in (self.assets_dir, self.scenes_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- assets ---------------------------------------------------------
    def save_upload(self, filename: str, src: BinaryIO) -> Asset:
        ext = Path(filename).suffix.lower()
        kind = _KIND_BY_EXT.get(ext)
        if kind is None:
            raise AssetError("Only .ply (splat) and .glb (mesh) files are supported")
        self._ensure_dirs()
        asset_id = new_id("a")
        final = self.assets_dir / f"{asset_id}{ext}"
        part = self.assets_dir / f"{asset_id}{ext}.part"
        try:
            with open(part, "wb") as fh:
                shutil.copyfileobj(src, fh, 1024 * 1024)
            if kind == "splat":
                validate_ply(part)
            else:
                validate_glb(part)
        except (PlyFormatError, GlbFormatError) as exc:
            part.unlink(missing_ok=True)
            raise AssetError(str(exc)) from exc
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        os.replace(part, final)
        asset = Asset(
            id=asset_id,
            kind=kind,
            name=Path(filename.replace("\\", "/")).stem[:128] or asset_id,
            size_bytes=final.stat().st_size,
            source="upload",
            created_ts=time.time(),
        )
        _atomic_write_text(self.assets_dir / f"{asset_id}.json", asset.model_dump_json())
        return asset

    def _upload_path(self, asset: Asset) -> Path:
        return self.assets_dir / f"{asset.id}{_EXT_BY_KIND[asset.kind]}"

    def _pipeline_ply(self, scene_name: str) -> Path | None:
        ply_dir = self.data_root / scene_name / "output" / "ply"
        plys = sorted(ply_dir.glob("*.ply")) if ply_dir.is_dir() else []
        return plys[-1] if plys else None

    def _pipeline_asset(self, scene_name: str) -> Asset | None:
        ply = self._pipeline_ply(scene_name)
        if ply is None:
            return None
        st = ply.stat()
        return Asset(id=PIPELINE_PREFIX + scene_name, kind="splat", name=scene_name,
                     size_bytes=st.st_size, source="pipeline", created_ts=st.st_mtime)

    def list_assets(self) -> list[Asset]:
        out: list[Asset] = []
        if self.assets_dir.is_dir():
            for meta in sorted(self.assets_dir.glob("*.json")):
                asset = self.get_asset(meta.stem)
                if asset is not None:
                    out.append(asset)
        if self.data_root.is_dir():
            for d in sorted(self.data_root.iterdir()):
                if d.is_dir() and d.name != "compose" and _ID_RE.match(PIPELINE_PREFIX + d.name) \
                        and _ID_RE.match(d.name):
                    asset = self._pipeline_asset(d.name)
                    if asset is not None:
                        out.append(asset)
        return out

    def get_asset(self, asset_id: str) -> Asset | None:
        if not _ID_RE.match(asset_id):
            return None
        if asset_id.startswith(PIPELINE_PREFIX):
            name = asset_id[len(PIPELINE_PREFIX):]
            return self._pipeline_asset(name) if _ID_RE.match(name) else None
        meta = self.assets_dir / f"{asset_id}.json"
        if not meta.is_file():
            return None
        try:
            asset = Asset.model_validate_json(meta.read_text(encoding="utf-8"))
        except ValueError:
            return None
        return asset if self._upload_path(asset).is_file() else None

    def asset_path(self, asset_id: str) -> Path | None:
        asset = self.get_asset(asset_id)
        if asset is None:
            return None
        if asset.source == "pipeline":
            return self._pipeline_ply(asset.name)
        return self._upload_path(asset)

    # -- scenes ---------------------------------------------------------
    def create_scene(self, name: str, base: Asset) -> SceneDoc:
        doc = SceneDoc(
            id=new_id("s"),
            name=name,
            objects=[SceneObject(id=new_id("o"), kind="splat", asset=base.id, name=base.name, role="base")],
        )
        self.save_scene(doc)
        return doc

    def save_scene(self, doc: SceneDoc) -> None:
        self._ensure_dirs()
        _atomic_write_text(self.scenes_dir / f"{doc.id}.json", doc.model_dump_json(indent=2))

    def load_scene(self, scene_id: str) -> SceneDoc | None:
        if not _ID_RE.match(scene_id):
            return None
        path = self.scenes_dir / f"{scene_id}.json"
        if not path.is_file():
            return None
        return SceneDoc.model_validate_json(path.read_text(encoding="utf-8"))

    def list_scenes(self) -> list[SceneSummary]:
        out: list[SceneSummary] = []
        if not self.scenes_dir.is_dir():
            return out
        for path in self.scenes_dir.glob("*.json"):
            try:
                doc = SceneDoc.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out.append(SceneSummary(id=doc.id, name=doc.name, updated_ts=path.stat().st_mtime,
                                    object_count=len(doc.objects)))
        out.sort(key=lambda s: s.updated_ts, reverse=True)
        return out

    # -- exports --------------------------------------------------------
    def export_dir(self, scene_id: str) -> Path:
        return self.exports_dir / scene_id

    def export_zip(self, scene_id: str) -> Path:
        return self.exports_dir / f"{scene_id}.zip"

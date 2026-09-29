"""Export job body: bake a saved SceneDoc into merged.ply + meshes + zip."""
from __future__ import annotations

import re
import secrets
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .bake import ColorAdjust, Crop, Placement, merge_clouds, transform_cloud
from .fsutil import replace_with_retry
from .glb import placement_matrix, wrap_with_transform
from .models import SceneDoc, SceneObject
from .plyio import GaussianCloud, read_ply, write_ply
from .store import ComposeStore

ProgressFn = Callable[..., None]


def placement_for(obj: SceneObject) -> Placement:
    t = obj.transform
    crop = None
    if obj.crop is not None:
        crop = Crop(center=obj.crop.center, half_size=obj.crop.halfSize,
                    quaternion_xyzw=obj.crop.quaternion)
    color = None
    if obj.color is not None:
        color = ColorAdjust(exposure=obj.color.exposure, tint=obj.color.tint,
                            saturation=obj.color.saturation)
    return Placement(position=t.position, quaternion_xyzw=t.quaternion, scale=t.scale,
                     crop=crop, color=color)


def missing_assets(doc: SceneDoc, store: ComposeStore) -> list[str]:
    return [o.asset for o in doc.objects if o.visible and store.asset_path(o.asset) is None]


def _build_zip(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(src_dir).as_posix())


def _clean_stale_exports(store: ComposeStore, scene_id: str) -> None:
    """Remove leftovers of killed export runs for ``scene_id``.

    Deletes ``<id>.<token>.tmp`` dirs and ``<id>.<token>.zip.tmp`` files. If the
    export dir is missing but ``<id>.<token>.old`` backups exist (killed between the
    two swap steps), the newest backup is restored; any other backup is deleted.

    Assumes a single backend process with the single-worker JobManager: it cannot
    tell leftovers from a live export, so it is not safe with multiple uvicorn
    workers sharing DATA_ROOT.
    """
    pattern = re.compile(rf"^{re.escape(scene_id)}\.[0-9a-f]{{8}}\.(tmp|zip\.tmp|old)$")
    stale = [p for p in store.exports_dir.iterdir() if pattern.match(p.name)]
    backups = sorted((p for p in stale if p.name.endswith(".old") and p.is_dir()),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if backups and not store.export_dir(scene_id).exists():
        replace_with_retry(backups[0], store.export_dir(scene_id))
    for p in stale:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)


def run_export(
    doc: SceneDoc,
    store: ComposeStore,
    on_progress: ProgressFn,
    cancel_check: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Bake ``doc`` into ``exports/<id>/`` and ``exports/<id>.zip``.

    Everything is built under unique temp names first (including the zip); only
    when that all succeeded are the previous outputs swapped out, so a failure
    or cancellation never leaves a partial export or damages the previous one.
    """

    def check_cancel() -> None:
        if cancel_check():
            raise RuntimeError("Export cancelled")

    missing = missing_assets(doc, store)
    if missing:
        raise FileNotFoundError(f"Missing assets: {', '.join(missing)}")
    visible = [o for o in doc.objects if o.visible]
    splats = [o for o in visible if o.kind == "splat"]
    meshes = [o for o in visible if o.kind == "mesh"]
    on_progress("compose_load", 1.0, f"{len(visible)} obje hazır",
                {"splats": len(splats), "meshes": len(meshes)})

    store.exports_dir.mkdir(parents=True, exist_ok=True)
    _clean_stale_exports(store, doc.id)
    token = secrets.token_hex(4)
    tmp = store.exports_dir / f"{doc.id}.{token}.tmp"
    zip_tmp = store.exports_dir / f"{doc.id}.{token}.zip.tmp"
    backup = store.exports_dir / f"{doc.id}.{token}.old"
    final = store.export_dir(doc.id)
    zip_path = store.export_zip(doc.id)
    tmp.mkdir(parents=True)
    try:
        clouds = []
        # Several objects may share one asset (duplicates): read each PLY once and
        # keep it only until its last user is baked (transform_cloud never
        # mutates its input).
        uses_left = Counter(o.asset for o in splats)
        sources: dict[str, GaussianCloud] = {}
        for i, obj in enumerate(splats):
            check_cancel()
            on_progress("compose_bake", i / max(len(splats), 1), f"{obj.name} işleniyor", {})
            src = sources.get(obj.asset)
            if src is None:
                src = read_ply(store.asset_path(obj.asset))
            uses_left[obj.asset] -= 1
            if uses_left[obj.asset] > 0:
                sources[obj.asset] = src
            else:
                sources.pop(obj.asset, None)
            clouds.append(transform_cloud(src, placement_for(obj)))
            del src
        sources.clear()
        merged = merge_clouds(clouds)
        del clouds
        n_gaussians = merged.count
        check_cancel()
        on_progress("compose_write", 0.0, "merged.ply yazılıyor", {"gaussians": n_gaussians})
        write_ply(merged, tmp / "merged.ply")
        del merged
        for obj in meshes:
            t = obj.transform
            wrap_with_transform(store.asset_path(obj.asset), tmp / "meshes" / f"{obj.id}.glb",
                                placement_matrix(t.position, t.quaternion, t.scale))
        (tmp / "scene.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        on_progress("compose_write", 0.6, "zip hazırlanıyor", {})
        _build_zip(tmp, zip_tmp)

        # Commit: swap the directory (keeping the old one until the zip is in place too).
        had_previous = final.exists()
        if had_previous:
            replace_with_retry(final, backup)
        try:
            replace_with_retry(tmp, final)
            replace_with_retry(zip_tmp, zip_path)
            shutil.rmtree(backup, ignore_errors=True)
        except BaseException:
            shutil.rmtree(final, ignore_errors=True)
            if had_previous:
                replace_with_retry(backup, final)
            raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        zip_tmp.unlink(missing_ok=True)

    on_progress("compose_write", 1.0, "Export hazır", {})
    return {
        "scene_id": doc.id,
        "gaussians": n_gaussians,
        "meshes": len(meshes),
        "download_url": f"/compose/exports/{doc.id}/download",
    }

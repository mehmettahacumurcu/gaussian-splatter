"""Export job body: bake a saved SceneDoc into merged.ply + meshes + zip."""
from __future__ import annotations

import os
import shutil
import zipfile
from typing import Any, Callable

from .bake import ColorAdjust, Crop, Placement, merge_clouds, transform_cloud
from .glb import placement_matrix, wrap_with_transform
from .models import SceneDoc, SceneObject
from .plyio import read_ply, write_ply
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


def run_export(doc: SceneDoc, store: ComposeStore, on_progress: ProgressFn) -> dict[str, Any]:
    missing = missing_assets(doc, store)
    if missing:
        raise FileNotFoundError(f"Missing assets: {', '.join(missing)}")
    visible = [o for o in doc.objects if o.visible]
    splats = [o for o in visible if o.kind == "splat"]
    meshes = [o for o in visible if o.kind == "mesh"]
    on_progress("compose_load", 1.0, f"{len(visible)} obje hazır",
                {"splats": len(splats), "meshes": len(meshes)})

    store.exports_dir.mkdir(parents=True, exist_ok=True)
    tmp = store.exports_dir / f"{doc.id}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    final = store.export_dir(doc.id)
    try:
        clouds = []
        for i, obj in enumerate(splats):
            on_progress("compose_bake", i / max(len(splats), 1), f"{obj.name} işleniyor", {})
            clouds.append(transform_cloud(read_ply(store.asset_path(obj.asset)), placement_for(obj)))
        merged = merge_clouds(clouds)
        del clouds
        on_progress("compose_write", 0.0, "merged.ply yazılıyor", {"gaussians": merged.count})
        write_ply(merged, tmp / "merged.ply")
        for obj in meshes:
            t = obj.transform
            wrap_with_transform(store.asset_path(obj.asset), tmp / "meshes" / f"{obj.id}.glb",
                                placement_matrix(t.position, t.quaternion, t.scale))
        (tmp / "scene.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    on_progress("compose_write", 0.6, "zip hazırlanıyor", {})
    zip_path = store.export_zip(doc.id)
    zip_tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(zip_tmp, "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(final.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(final).as_posix())
    os.replace(zip_tmp, zip_path)
    on_progress("compose_write", 1.0, "Export hazır", {})
    return {
        "scene_id": doc.id,
        "gaussians": merged.count,
        "meshes": len(meshes),
        "download_url": f"/compose/exports/{doc.id}/download",
    }

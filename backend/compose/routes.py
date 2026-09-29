"""/compose API: assets, scene documents and export jobs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .exporter import missing_assets, run_export
from .models import Asset, CreateSceneRequest, ExportResponse, SceneDoc, SceneSummary
from .store import AssetError, ComposeStore

_MEDIA_TYPES = {"splat": "application/octet-stream", "mesh": "model/gltf-binary"}


def _kind_errors(doc: SceneDoc, store: ComposeStore) -> list[str]:
    errors = []
    for obj in doc.objects:
        asset = store.get_asset(obj.asset)
        if asset is not None and asset.kind != obj.kind:
            errors.append(f"object {obj.id} is a {obj.kind} but asset {obj.asset} is a {asset.kind}")
    return errors


def build_compose_router(data_root: str | Path, get_manager: Callable[[], Any]) -> APIRouter:
    store = ComposeStore(data_root)
    router = APIRouter(prefix="/compose", tags=["compose"])

    @router.get("/assets", response_model=list[Asset])
    def list_assets() -> list[Asset]:
        return store.list_assets()

    @router.post("/assets", response_model=Asset, status_code=201)
    def upload_asset(file: UploadFile = File(...)) -> Asset:
        try:
            return store.save_upload(file.filename or "", file.file)
        except AssetError as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.get("/assets/{asset_id}/file")
    def asset_file(asset_id: str) -> FileResponse:
        asset = store.get_asset(asset_id)
        path = store.asset_path(asset_id) if asset else None
        if asset is None or path is None:
            raise HTTPException(404, f"Asset not found: {asset_id}")
        return FileResponse(path, media_type=_MEDIA_TYPES[asset.kind],
                            filename=f"{asset.name}{path.suffix}")

    @router.get("/scenes", response_model=list[SceneSummary])
    def list_scenes() -> list[SceneSummary]:
        return store.list_scenes()

    @router.post("/scenes", response_model=SceneDoc, status_code=201)
    def create_scene(req: CreateSceneRequest) -> SceneDoc:
        base = store.get_asset(req.base_asset)
        if base is None:
            raise HTTPException(400, f"Base asset not found: {req.base_asset}")
        if base.kind != "splat":
            raise HTTPException(400, "Base asset must be a splat (.ply)")
        return store.create_scene(req.name, base)

    @router.get("/scenes/{scene_id}", response_model=SceneDoc)
    def get_scene(scene_id: str) -> SceneDoc:
        doc = store.load_scene(scene_id)
        if doc is None:
            raise HTTPException(404, f"Scene not found: {scene_id}")
        return doc

    @router.put("/scenes/{scene_id}", response_model=SceneDoc)
    def save_scene(scene_id: str, doc: SceneDoc) -> SceneDoc:
        if doc.id != scene_id:
            raise HTTPException(400, "Scene id in body does not match the URL")
        errors = _kind_errors(doc, store)
        if errors:
            raise HTTPException(400, "; ".join(errors))
        store.save_scene(doc)
        return doc

    @router.post("/scenes/{scene_id}/export", response_model=ExportResponse, status_code=202)
    def export_scene(scene_id: str) -> ExportResponse:
        doc = store.load_scene(scene_id)
        if doc is None:
            raise HTTPException(404, f"Scene not found: {scene_id}")
        missing = missing_assets(doc, store)
        if missing:
            raise HTTPException(400, f"Missing assets: {', '.join(missing)}")
        if not any(o.visible and o.kind == "splat" for o in doc.objects):
            raise HTTPException(400, "No visible splat objects to export")
        manager = get_manager()
        job = manager.create(scene=f"compose-{scene_id}")
        manager.submit(job.id, lambda cb: run_export(doc, store, cb))
        return ExportResponse(job_id=job.id)

    @router.get("/exports/{scene_id}/download")
    def download_export(scene_id: str) -> FileResponse:
        doc = store.load_scene(scene_id)
        zip_path = store.export_zip(scene_id) if doc else None
        if doc is None or zip_path is None or not zip_path.is_file():
            raise HTTPException(404, f"No export for scene: {scene_id}")
        return FileResponse(zip_path, media_type="application/zip", filename=f"{doc.name}.zip")

    return router

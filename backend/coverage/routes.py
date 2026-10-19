"""/coverage API: start a coverage scan, list results, stream the splats and photos."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .detect import IMAGE_SUFFIXES, DetectError, detect
from .runner import run_coverage
from .store import CoverageStore


class CoverageRequest(BaseModel):
    path: str = ""        # run folder (or a .ply inside it)
    ply_path: str = ""    # optional explicit splat
    sparse_dir: str = ""  # optional explicit COLMAP model folder
    name: str = ""


class CoverageStarted(BaseModel):
    job_id: str
    coverage_id: str
    ply_path: str
    sparse_dir: str


def build_coverage_router(data_root: str | Path, get_manager: Callable[[], Any]) -> APIRouter:
    store = CoverageStore(data_root)
    router = APIRouter(prefix="/coverage", tags=["coverage"])

    def _meta(result_id: str) -> dict:
        meta = store.get(result_id)
        if meta is None:
            raise HTTPException(404, f"Kapsama sonucu bulunamadı: {result_id}")
        return meta

    @router.get("")
    def list_results() -> list[dict]:
        return store.list()

    @router.post("", response_model=CoverageStarted, status_code=202)
    def start(req: CoverageRequest) -> CoverageStarted:
        try:
            ply, sparse = detect(req.path, req.ply_path, req.sparse_dir)
        except DetectError as exc:
            raise HTTPException(400, str(exc)) from exc
        name = req.name.strip() or (Path(req.path).name if req.path.strip() else ply.stem)
        result_id = store.new_id(name)
        manager = get_manager()
        job = manager.create(scene=f"coverage-{result_id}")
        job_id = job.id

        def cancelled() -> bool:
            return bool(getattr(manager, "cancel_requested", lambda _id: False)(job_id))

        manager.submit(job_id, lambda cb: run_coverage(store, result_id, name, ply, sparse, cb, cancelled))
        return CoverageStarted(job_id=job_id, coverage_id=result_id, ply_path=str(ply), sparse_dir=str(sparse))

    @router.get("/{result_id}")
    def get_result(result_id: str) -> dict:
        return {**_meta(result_id), "views": store.views(result_id)}

    @router.delete("/{result_id}")
    def delete_result(result_id: str) -> dict:
        if not store.delete(result_id):
            raise HTTPException(404, f"Kapsama sonucu bulunamadı: {result_id}")
        return {"ok": True}

    @router.get("/{result_id}/splat/{which}")
    def splat_file(result_id: str, which: str) -> FileResponse:
        meta = _meta(result_id)
        if which == "heat":
            path = store.heat_path(result_id)
        elif which == "normal":
            path = Path(meta["ply_path"])
        else:
            raise HTTPException(404, "which: normal | heat")
        if path is None or not path.is_file():
            raise HTTPException(404, f"Splat dosyası yok: {path}")
        return FileResponse(path, media_type="application/octet-stream", filename=f"{result_id}-{which}.ply")

    @router.get("/{result_id}/photo/{index}")
    def photo(result_id: str, index: int) -> FileResponse:
        meta = _meta(result_id)
        views = store.views(result_id)
        if not meta.get("images_dir") or not 0 <= index < len(views):
            raise HTTPException(404, "Fotoğraf yok")
        base = Path(meta["images_dir"])
        name = Path(views[index]["name"])
        for cand in (base / name, base / name.name):
            if cand.is_file() and cand.suffix.lower() in IMAGE_SUFFIXES:
                return FileResponse(cand)
        raise HTTPException(404, "Fotoğraf yok")

    return router

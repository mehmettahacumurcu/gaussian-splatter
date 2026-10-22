"""/spirula API: local runs, live cost estimates and GPU status for the "Yeni Splat" page."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import estimate as E
from . import settings as S
from .gpu import gpu_status
from .runner import IMAGE_SUFFIXES, VIEWER_PORT, LocalRunRequest, RunError, binary_version, find_binary, run_local

GPU_CACHE_SECONDS = 3.0


class EstimateRequest(BaseModel):
    settings: S.SpirulaSettings = Field(default_factory=S.SpirulaSettings)
    image_width: int = Field(default=1920, ge=64, le=16384)
    image_height: int = Field(default=1080, ge=64, le=16384)


class ProbeRequest(BaseModel):
    input_mode: Literal["video", "images", "dataset"]
    input_path: str
    fps: float = Field(default=4.0, gt=0, le=30)


class LocalStartRequest(BaseModel):
    input_mode: Literal["video", "images", "dataset"]
    input_path: str
    settings: S.SpirulaSettings = Field(default_factory=S.SpirulaSettings)
    name: str = ""


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:40] or "run"


def probe_input(mode: str, path: Path, fps: float) -> dict:
    """Frame count and size, so the estimate uses the real capture instead of 1080p defaults."""
    if mode == "video":
        if not path.is_file():
            raise HTTPException(400, f"Video bulunamadı: {path}")
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return {"kind": "video", "frames": None, "width": None, "height": None, "warning": "ffprobe yok"}
        out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height,avg_frame_rate:stream_side_data=rotation:format=duration",
                              "-of", "json", str(path)], capture_output=True, text=True, timeout=60)
        info = json.loads(out.stdout or "{}")
        stream = (info.get("streams") or [{}])[0]
        w, h = stream.get("width"), stream.get("height")
        rotation = next((abs(int(d.get("rotation", 0))) for d in stream.get("side_data_list", []) if "rotation" in d), 0)
        if rotation in (90, 270):
            w, h = h, w
        duration = float(info.get("format", {}).get("duration") or 0)
        return {"kind": "video", "frames": int(duration * fps) or None, "width": w, "height": h,
                "duration": round(duration, 1)}
    folder = path / "images" if (path / "images").is_dir() else path
    if not folder.is_dir():
        raise HTTPException(400, f"Klasör bulunamadı: {path}")
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    w = h = None
    if images:
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(images[0]) as im:
                w, h = im.size
        except (OSError, UnidentifiedImageError):
            pass
    has_model = any((path / "sparse").glob("*/images.bin")) or (path / "sparse" / "images.bin").is_file()
    return {"kind": mode, "frames": len(images), "width": w, "height": h, "has_sfm": has_model}


def build_spirula_router(data_root: str | Path, get_manager: Callable[[], Any]) -> APIRouter:
    root = Path(data_root) / "spirula"
    router = APIRouter(tags=["spirula"])
    gpu_cache: dict = {"at": 0.0, "value": None}

    def cached_gpu() -> dict:
        if gpu_cache["value"] is None or time.time() - gpu_cache["at"] > GPU_CACHE_SECONDS:
            gpu_cache.update(at=time.time(), value=gpu_status())
        return gpu_cache["value"]

    def manifest(run_id: str) -> dict:
        path = root / run_id / "manifest.json"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or not path.is_file():
            raise HTTPException(404, f"Koşu bulunamadı: {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    @router.get("/system/gpu")
    def system_gpu() -> dict:
        return cached_gpu()

    @router.get("/spirula/info")
    def info() -> dict:
        try:
            binary = find_binary()
            version, error = binary_version(binary), None
        except (RunError, OSError, subprocess.TimeoutExpired) as exc:
            binary, version, error = None, None, str(exc)
        return {"available": binary is not None, "binary": str(binary) if binary else None, "version": version,
                "error": error, "presets": S.PRESETS, "defaults": S.SpirulaSettings().model_dump(),
                "gpu_profiles": E.GPU_PROFILES, "viewer_port": VIEWER_PORT}

    @router.post("/spirula/estimate")
    def estimate(req: EstimateRequest) -> dict:
        gpu = cached_gpu()
        local = gpu["gpus"][0] if gpu.get("gpus") else {}
        out = []
        for name in E.GPU_PROFILES:
            est = E.estimate(req.settings, req.image_width, req.image_height, gpu=name,
                             local_total_mib=local.get("total_mib"), local_used_mib=local.get("used_mib"))
            row = est.as_dict()
            if name == "local" and local:
                row["gpu_label"] = f"Bu PC · {local['name'].replace('NVIDIA GeForce ', '')}"
            out.append(row)
        return {"resolved": S.resolve(req.settings).as_dict(), "estimates": out, "gpu": gpu}

    @router.post("/spirula/probe")
    def probe(req: ProbeRequest) -> dict:
        return probe_input(req.input_mode, Path(req.input_path.strip().strip('"')), req.fps)

    @router.post("/spirula/local", status_code=202)
    def start_local(req: LocalStartRequest) -> dict:
        path = Path(req.input_path.strip().strip('"'))
        if not path.exists():
            raise HTTPException(400, f"Girdi bulunamadı: {path}")
        try:
            find_binary()
        except RunError as exc:
            raise HTTPException(400, str(exc)) from exc
        name = req.name.strip() or path.stem
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(name)}"
        manager = get_manager()
        job = manager.create(scene=f"spirula-{run_id}")
        job_id = job.id
        local_req = LocalRunRequest(run_id=run_id, input_mode=req.input_mode, input_path=path,
                                    settings=req.settings, name=name)

        def cancelled() -> bool:
            return bool(getattr(manager, "cancel_requested", lambda _id: False)(job_id))

        def runner(cb):
            try:
                result = run_local(local_req, data_root, cb, cancelled)
            except Exception as exc:
                _mark_manifest(root / run_id, "failed", str(exc))
                raise
            return {**result, "run_id": run_id, "download_url": f"/spirula/runs/{run_id}/splat"}

        manager.submit(job_id, runner)
        return {"job_id": job_id, "run_id": run_id}

    @router.get("/spirula/runs")
    def list_runs() -> list[dict]:
        if not root.is_dir():
            return []
        runs = []
        for path in sorted(root.glob("*/manifest.json"), reverse=True):
            try:
                m = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            runs.append({k: m.get(k) for k in ("run_id", "name", "status", "input_mode", "input_path", "settings",
                                               "started_at", "finished_at", "splat", "error")})
        return runs

    @router.get("/spirula/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return manifest(run_id)

    @router.get("/spirula/runs/{run_id}/log")
    def run_log(run_id: str, tail: int = 200) -> dict:
        manifest(run_id)
        logs = sorted((root / run_id / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime)
        if not logs:
            return {"label": None, "lines": []}
        latest = logs[-1]
        # Spirula redraws its progress line with \r; keep only the last state of each line.
        text = latest.read_text(encoding="utf-8", errors="replace")
        lines = [ln.rsplit("\r", 1)[-1] for ln in text.splitlines()]
        return {"label": latest.stem, "labels": [p.stem for p in logs], "lines": lines[-max(1, min(tail, 2000)):]}

    @router.get("/spirula/runs/{run_id}/splat")
    def run_splat(run_id: str) -> FileResponse:
        m = manifest(run_id)
        splat = Path(m.get("splat") or "")
        if not splat.is_file():
            raise HTTPException(404, "Bu koşunun splat dosyası yok")
        return FileResponse(splat, media_type="application/octet-stream", filename=f"{run_id}.ply")

    return router


def _mark_manifest(workspace: Path, status: str, error: str) -> None:
    path = workspace / "manifest.json"
    try:
        m = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        m.update(status="cancelled" if error == "İptal edildi" else status, error=error, finished_at=time.time())
        path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, ValueError):
        pass

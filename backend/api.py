"""Faz 7 — FastAPI backend.

Video upload → background pipeline → .ply indirme.

Çalıştırma:
    uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload

Swagger UI:
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations
import io
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .api_models import (
    HealthResponse,
    Job,
    JobListResponse,
    JobStatus,
    ProcessResponse,
)
from .config import default_config, cloud_config, scene_paths
from .job_manager import JobManager, get_manager
from .pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Uygulama yaşam döngüsü
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown hook'ları."""
    manager = get_manager()
    app.state.manager = manager
    print("[api] JobManager hazır (max_workers=1, GPU sıralaması aktif)")
    try:
        yield
    finally:
        print("[api] Shutdown — executor kapatılıyor (running job'lar bitene kadar beklenir)")
        manager.shutdown(wait=True)


app = FastAPI(
    title="4DGS Studio Backend",
    description="Video → 4D Gaussian Splatting pipeline (Faz 7 HTTP katmanı).",
    version="0.1.0",
    lifespan=lifespan,
)


# CORS — Tauri / localhost web client'ları için
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:*",
        "http://127.0.0.1:*",
        "tauri://localhost",
        "https://tauri.localhost",
    ],
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|tauri://localhost|https://tauri\.localhost)$",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Servis sağlığı + GPU durumu."""
    manager: JobManager = app.state.manager
    gpu_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_ok else None
    return HealthResponse(
        gpu_available=gpu_ok,
        gpu_name=gpu_name,
        active_jobs=manager.active_count(),
    )


# ---------------------------------------------------------------------------
# Process — yeni job oluştur
# ---------------------------------------------------------------------------
@app.post("/process", response_model=ProcessResponse, status_code=202, tags=["jobs"])
async def process_video(
    video: UploadFile = File(..., description="Girdi video dosyası (mp4/mov)"),
    scene: str = Form("unnamed_scene", description="Sahne ismi — data/<scene>/ altında çalışılır"),
    smoke_test: bool = Form(False, description="True: hızlı preset (500 iter, 480x270, 10 ts)"),
    cloud: bool = Form(False, description="True: cloud_config (1920x1080, 60k iter)"),
    skip_foundation: bool = Form(True, description="Foundation modelleri atla"),
) -> ProcessResponse:
    """
    Video'yu upload et ve pipeline'ı kuyruğa al.

    Response: 202 Accepted + job_id. İlerleme için /status/{job_id}'yi poll et.
    """
    # Sahne adı sanitize — path traversal engelle
    safe_scene = _safe_scene_name(scene)

    # Video'yu diske yaz
    paths = scene_paths(safe_scene)
    paths["base"].mkdir(parents=True, exist_ok=True)
    video_path = paths["base"] / "video.mp4"

    # Chunked write (büyük dosya support)
    try:
        with open(video_path, "wb") as f:
            while chunk := await video.read(1024 * 1024):  # 1 MB
                f.write(chunk)
    except Exception as e:
        raise HTTPException(500, f"Video yazılamadı: {e}")

    # Job kaydını oluştur
    manager: JobManager = app.state.manager
    job = manager.create(scene=safe_scene, smoke_test=smoke_test)

    # Runner kapanı — pipeline.run_pipeline'ı callback'le çağırır
    def _runner(on_progress: Any) -> dict:
        cfg = cloud_config() if cloud else default_config()
        if smoke_test:
            cfg.train.n_iters = 500
            cfg.train.image_resolution = (480, 270)
            cfg.train.ckpt_interval = 500
            cfg.train.log_interval = 25
            cfg.train.density_start_iter = 100
            cfg.train.density_end_iter = 400
            cfg.train.density_interval = 50
            cfg.export.num_timestamps = 10

        return run_pipeline(
            str(video_path),
            safe_scene,
            cfg,
            skip_foundation=skip_foundation,
            progress_callback=on_progress,
        )

    manager.submit(job.id, _runner)

    return ProcessResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"/status/{job.id}",
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
@app.get("/status/{job_id}", response_model=Job, tags=["jobs"])
def get_status(job_id: str) -> Job:
    """Belirli bir job'un durumunu döndür."""
    manager: JobManager = app.state.manager
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job bulunamadı: {job_id}")
    return job


# ---------------------------------------------------------------------------
# Liste
# ---------------------------------------------------------------------------
@app.get("/jobs", response_model=JobListResponse, tags=["jobs"])
def list_jobs() -> JobListResponse:
    """Tüm job'ları (en yeni önce) listele."""
    manager: JobManager = app.state.manager
    jobs = sorted(manager.list_all(), key=lambda j: j.created_at, reverse=True)
    return JobListResponse(jobs=jobs, total=len(jobs))


# ---------------------------------------------------------------------------
# Download — .ply'ları zip olarak stream et
# ---------------------------------------------------------------------------
@app.get("/download/{job_id}", tags=["jobs"])
def download_result(job_id: str) -> StreamingResponse:
    """Bitmiş bir job'un .ply çıktılarını ZIP olarak indir."""
    manager: JobManager = app.state.manager
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job bulunamadı: {job_id}")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, f"Job henüz hazır değil (status: {job.status.value})")

    ply_dir = Path(job.ply_dir) if job.ply_dir else scene_paths(job.scene)["output"] / "ply"
    if not ply_dir.exists():
        raise HTTPException(404, f"Çıktı klasörü bulunamadı: {ply_dir}")

    ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        raise HTTPException(404, f"{ply_dir} içinde .ply yok")

    # Memory stream'de zip oluştur ve stream'le gönder
    def _zip_iter():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in ply_files:
                zf.write(p, arcname=p.name)
        buf.seek(0)
        while chunk := buf.read(64 * 1024):
            yield chunk

    return StreamingResponse(
        _zip_iter(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job.scene}_{job_id[:8]}.zip"'},
    )


# ---------------------------------------------------------------------------
# Splat - viewer icin bireysel .ply erisimi (Faz 8a)
# Hem in-memory job_id'den hem de diskteki scene name'den ply bulabilir.
# ---------------------------------------------------------------------------
class SplatInfo(BaseModel):
    """Bir job/sahne'nin splat ciktilarinin metadata'si (viewer icin)."""
    job_id: str           # gelen identifier (job_id ya da scene name)
    scene: str            # cozulmus sahne adi
    num_frames: int
    frame_urls: list[str]
    total_size_bytes: int
    status: JobStatus
    source: str           # "registry" ya da "disk"


class SceneListItem(BaseModel):
    """/scenes listesinin bir elemani."""
    name: str
    num_frames: int
    total_size_bytes: int
    modified_ts: float


class SceneListResponse(BaseModel):
    scenes: list[SceneListItem]
    total: int


def _resolve_ply_dir(identifier: str) -> tuple[Path, str, JobStatus, str]:
    """
    Bir identifier'i (job_id veya scene name) ply klasorune cevir.
    Returns (ply_dir, scene_name, status, source).
    source: "registry" | "disk"
    """
    # 1) Once in-memory job registry
    manager: JobManager = app.state.manager
    job = manager.get(identifier)
    if job is not None:
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                409, f"Job henuz hazir degil (status: {job.status.value})"
            )
        ply_dir = Path(job.ply_dir) if job.ply_dir else scene_paths(job.scene)["output"] / "ply"
        return ply_dir, job.scene, job.status, "registry"

    # 2) Scene name olarak dene (data/<name>/output/ply)
    safe = _safe_scene_name(identifier)
    candidate = scene_paths(safe)["output"] / "ply"
    if candidate.exists() and any(candidate.glob("*.ply")):
        return candidate, safe, JobStatus.COMPLETED, "disk"

    raise HTTPException(
        404,
        f"Job veya sahne bulunamadi: {identifier} "
        f"(ne registry'de ne data/{safe}/output/ply altinda)",
    )


@app.get("/scenes", response_model=SceneListResponse, tags=["disk"])
def list_disk_scenes() -> SceneListResponse:
    """
    data/ altinda output/ply iceren tum sahneleri listele.
    Server restart olsa bile diskteki ciktilar burada gorunur.
    """
    data_root = Path("data")
    items: list[SceneListItem] = []
    if data_root.exists():
        for scene_dir in sorted(data_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            ply_dir = scene_dir / "output" / "ply"
            if not ply_dir.exists():
                continue
            ply_files = sorted(ply_dir.glob("*.ply"))
            if not ply_files:
                continue
            total = sum(p.stat().st_size for p in ply_files)
            items.append(SceneListItem(
                name=scene_dir.name,
                num_frames=len(ply_files),
                total_size_bytes=total,
                modified_ts=ply_dir.stat().st_mtime,
            ))
    # En yeni once
    items.sort(key=lambda s: s.modified_ts, reverse=True)
    return SceneListResponse(scenes=items, total=len(items))


@app.get("/splat/{job_id}/info", response_model=SplatInfo, tags=["splat"])
def splat_info(job_id: str) -> SplatInfo:
    """
    Viewer metadata: kac frame, hangi URL'lerden cekilecek, toplam boyut.
    `job_id` parametresi hem UUID hem sahne adi olabilir.
    """
    ply_dir, scene, status, source = _resolve_ply_dir(job_id)
    ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        raise HTTPException(404, f"{ply_dir} icinde .ply yok")

    total = sum(p.stat().st_size for p in ply_files)
    frame_urls = [f"/splat/{job_id}/frame/{i}" for i in range(len(ply_files))]
    return SplatInfo(
        job_id=job_id,
        scene=scene,
        num_frames=len(ply_files),
        frame_urls=frame_urls,
        total_size_bytes=total,
        status=status,
        source=source,
    )


@app.get("/splat/{job_id}/frame/{idx}", tags=["splat"])
def splat_frame(job_id: str, idx: int) -> FileResponse:
    """
    Belirli bir timestamp icin .ply dosyasini dondur.
    `job_id` hem UUID hem sahne adi olabilir.
    """
    ply_dir, _scene, _status, _source = _resolve_ply_dir(job_id)
    ply_files = sorted(ply_dir.glob("*.ply"))
    if idx < 0 or idx >= len(ply_files):
        raise HTTPException(
            404,
            f"Frame index gecersiz: {idx} (mevcut: 0..{len(ply_files)-1})",
        )

    return FileResponse(
        ply_files[idx],
        media_type="application/octet-stream",
        filename=ply_files[idx].name,
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------
def _safe_scene_name(raw: str) -> str:
    """Path traversal + weird chars sanitize. Sadece alfanumerik, _, -, nokta kabul."""
    safe = "".join(c if (c.isalnum() or c in "_-.") else "_" for c in raw)
    safe = safe.strip("._") or "unnamed_scene"
    return safe[:64]

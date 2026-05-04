"""Faz 7 — FastAPI için Pydantic veri modelleri."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Bir job'un yaşam döngüsündeki dört durum."""
    QUEUED = "queued"         # Kuyruğa girdi, henüz başlamadı
    RUNNING = "running"       # Aktif işleniyor
    COMPLETED = "completed"   # Başarıyla bitti, indirmeye hazır
    FAILED = "failed"         # Hata aldı, error alanında detay var


class PhaseProgress(BaseModel):
    """Tek bir faz için ilerleme bilgisi."""
    name: str = Field(..., description="Faz adı, örn. 'colmap', 'training', 'export'")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="0.0–1.0 arası ilerleme oranı")
    message: str = Field("", description="O an görülebilir kısa mesaj (iter sayacı, loss vb.)")
    details: dict[str, Any] = Field(default_factory=dict, description="Faza özel ek veri")


class Job(BaseModel):
    """API'da job kaydının tam şeması."""
    id: str
    scene: str
    status: JobStatus
    smoke_test: bool = False

    # İlerleme
    phase: PhaseProgress = Field(default_factory=lambda: PhaseProgress(name="queued"))
    overall_progress: float = Field(0.0, ge=0.0, le=1.0, description="Tüm pipeline'a göre toplam ilerleme")

    # Zamanlama
    created_at: float = Field(..., description="Unix timestamp (job oluşturulma)")
    started_at: float | None = None
    finished_at: float | None = None

    # Sonuç / hata
    result: dict[str, Any] | None = Field(
        None, description="Başarılı ise run_pipeline'ın döndürdüğü status dict'i"
    )
    error: str | None = Field(None, description="Hata mesajı (FAILED durumunda)")

    # Dosya yolları (indirme için)
    ply_dir: str | None = None
    download_url: str | None = None

    # Cooperative cancel — set by JobManager.cancel() for RUNNING jobs.
    # Runners poll this via JobManager.cancel_requested(job_id) and stop
    # cleanly at the next phase/iter checkpoint.
    cancel_requested: bool = False


class ProcessResponse(BaseModel):
    """POST /process cevabı."""
    job_id: str
    status: JobStatus
    status_url: str
    message: str = "Job kuyruğa alındı"


class JobListResponse(BaseModel):
    """GET /jobs cevabı."""
    jobs: list[Job]
    total: int


class HealthResponse(BaseModel):
    """GET / cevabı — basit sağlık kontrolü."""
    service: str = "4DGS Studio Backend"
    version: str = "0.1.0"
    gpu_available: bool
    gpu_name: str | None = None
    active_jobs: int = 0


class EditJobRequest(BaseModel):
    """POST /process body when mode='edit'. Comes through the
    multipart form path with these fields as form-encoded values."""
    scene: str = Field(..., description="Scene directory under data/")
    source_ckpt: str = Field(
        ...,
        description="Source checkpoint, relative to data/<scene>/ "
                    "(e.g. 'output/ckpt/ckpt_final.pt')",
    )
    frame_idx: int = Field(..., ge=0, description="Frame index user clicked on")
    click_x: float = Field(..., ge=0.0, le=1.0, description="Normalized x click")
    click_y: float = Field(..., ge=0.0, le=1.0, description="Normalized y click")
    quality_mode: str = Field(..., description="'A' (LaMa preview) or 'B' (SD quality)")

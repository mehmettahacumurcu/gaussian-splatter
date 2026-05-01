"""Faz 7 — Thread-safe job registry + single-worker executor.

Tek GPU'muz var, bu yüzden aynı anda tek bir job çalışmalı. Birden fazla
request gelirse ThreadPoolExecutor(max_workers=1) onları sırayla işler.
State in-memory tutuluyor — server restart'ında kaybolur.
"""
from __future__ import annotations
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Callable

from .api_models import Job, JobStatus, PhaseProgress


# ---------------------------------------------------------------------------
# Faz ağırlıkları — toplam ilerleme hesabı için.
# Gerçek süreleri tam bilemesek de kaba bir oran yeter ki kullanıcı bir şey görsün.
# Değerler toplamda 1.0'a normalize edilir.
# ---------------------------------------------------------------------------
PHASE_WEIGHTS = {
    "frames":     0.05,   # ffmpeg hızlı
    "colmap":     0.15,   # birkaç dakika
    "foundation": 0.10,   # skip ediliyorsa 0 olur
    "init":       0.02,   # GaussianModel init
    "training":   0.60,   # loopun büyük çoğunluğu
    "export":     0.08,   # .ply yazma
}


def _phase_order() -> list[str]:
    return list(PHASE_WEIGHTS.keys())


def _compute_overall(phase_name: str, phase_progress: float,
                     skip_foundation: bool = True) -> float:
    """Hangi fazdayız + o faz ne kadar tamamlandı → toplam ilerleme (0-1)."""
    order = _phase_order()
    weights = dict(PHASE_WEIGHTS)
    if skip_foundation:
        weights["foundation"] = 0.0

    total = sum(weights.values()) or 1.0
    # Normalize
    weights = {k: v / total for k, v in weights.items()}

    done = 0.0
    for name in order:
        if name == phase_name:
            done += weights[name] * max(0.0, min(1.0, phase_progress))
            break
        done += weights[name]
    return min(done, 1.0)


class JobManager:
    """Job registry + sıralı executor. Thread-safe."""

    def __init__(self, data_dir: Path | str = "data"):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gs4d-job")
        self._futures: dict[str, Future] = {}
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Job yaşam döngüsü
    # ------------------------------------------------------------------
    def create(self, scene: str, smoke_test: bool = False) -> Job:
        """Yeni job kaydı oluştur (henüz çalıştırılmaz)."""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            scene=scene,
            status=JobStatus.QUEUED,
            smoke_test=smoke_test,
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def submit(self, job_id: str, runner: Callable[[Callable[[str, float, str, dict], None]], dict]) -> None:
        """
        Runner'ı executor'a gönder. Runner'a bir progress_callback geçilir:
            callback(phase_name, progress_0_1, message, details_dict)
        Runner başarıyla biterse status COMPLETED, exception fırlatırsa FAILED olur.
        """
        def _on_progress(phase: str, progress: float, message: str = "",
                         details: dict[str, Any] | None = None) -> None:
            self._update_phase(job_id, phase, progress, message, details or {})

        def _wrapped() -> None:
            job = self.get(job_id)
            if job is None:
                return
            self._mark_started(job_id)
            try:
                result = runner(_on_progress)
                self._mark_completed(job_id, result=result)
            except Exception as e:  # noqa: BLE001
                import traceback
                tb = traceback.format_exc()
                self._mark_failed(job_id, error=f"{e}\n\n{tb}")

        future = self._executor.submit(_wrapped)
        with self._lock:
            self._futures[job_id] = future

    # ------------------------------------------------------------------
    # İç state güncelleyiciler
    # ------------------------------------------------------------------
    def _mark_started(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.RUNNING
            job.started_at = time.time()

    def _mark_completed(self, job_id: str, result: dict | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.COMPLETED
            job.finished_at = time.time()
            job.overall_progress = 1.0
            job.phase = PhaseProgress(name="done", progress=1.0, message="Tamamlandı")
            if result is not None:
                job.result = result
            # İndirilebilir URL'i set et
            ply_dir = self.data_dir / job.scene / "output" / "ply"
            if ply_dir.exists():
                job.ply_dir = str(ply_dir)
                job.download_url = f"/download/{job_id}"

    def _mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = JobStatus.FAILED
            job.finished_at = time.time()
            job.error = error

    def _update_phase(self, job_id: str, phase: str, progress: float,
                      message: str, details: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.phase = PhaseProgress(
                name=phase, progress=progress, message=message, details=details,
            )
            job.overall_progress = _compute_overall(phase, progress)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def list_all(self) -> list[Job]:
        with self._lock:
            return [j.model_copy(deep=True) for j in self._jobs.values()]

    def active_count(self) -> int:
        with self._lock:
            return sum(
                1 for j in self._jobs.values()
                if j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            )

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------
    def cancel(self, job_id: str) -> tuple[bool, str]:
        """Cancel a job. Returns (success, message).
        - QUEUED jobs: Future cancelled, status set to FAILED. Reliable.
        - RUNNING jobs: best-effort. We can't actually interrupt subprocess.run
          calls (ffmpeg, colmap) running in the executor thread. We mark the
          job as cancellation-requested but it will still run to completion
          or natural failure.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False, "Job not found"
            future = self._futures.get(job_id)
            if job.status == JobStatus.QUEUED:
                if future is not None and future.cancel():
                    job.status = JobStatus.FAILED
                    job.finished_at = time.time()
                    job.error = "Cancelled by user (queued)"
                    return True, "Cancelled queued job"
                # Future raced into running between .get() and .cancel() — fall through
            if job.status == JobStatus.RUNNING:
                # Can't actually stop a running job from outside. Mark a flag
                # the pipeline could check, but for now the best we can do is
                # let it finish or hang. Don't lie to the user.
                return False, (
                    "Job is already running; cancellation not supported once "
                    "subprocess phase begins. Stop the pod to abort."
                )
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                return False, f"Job already {job.status.value}"
            return False, f"Unknown status: {job.status}"

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def shutdown(self, wait: bool = False) -> None:
        """Executor'ı kapat. wait=True ise bekleyen job'lar bitinceye kadar bekler."""
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


# Modül-düzeyinde tekil instance (FastAPI app bunu kullanır)
_default_manager: JobManager | None = None


def get_manager() -> JobManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = JobManager()
    return _default_manager

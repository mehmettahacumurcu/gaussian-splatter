"""Unit test: JobManager.cancel() sets a flag readable by runners.

Run: python scripts/test_edit_cancel_flag.py
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.job_manager import JobManager


def test_cancel_requested_reads_flag():
    mgr = JobManager()
    job_id = "test_job_1"
    # simulate a running job by direct insert (bypass _executor)
    mgr._jobs[job_id] = {
        "id": job_id, "scene": "x", "status": "running",
        "phase": {"name": "training", "progress": 0.5, "message": "", "details": {}},
        "created_at": 0.0, "started_at": 0.0,
        "cancel_requested": False,
    }
    assert mgr.cancel_requested(job_id) is False
    mgr.cancel(job_id)  # marks the flag for RUNNING jobs
    assert mgr.cancel_requested(job_id) is True


if __name__ == "__main__":
    test_cancel_requested_reads_flag()
    print("ok: test_cancel_requested_reads_flag")

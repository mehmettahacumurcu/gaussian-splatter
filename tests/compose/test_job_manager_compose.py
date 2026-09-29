from __future__ import annotations

import pytest

from backend.job_manager import JobManager, _compute_overall


def test_compose_phases_have_their_own_progress_scale():
    assert _compute_overall("compose_load", 1.0) == pytest.approx(0.05)
    assert _compute_overall("compose_bake", 0.5) == pytest.approx(0.40)
    assert _compute_overall("compose_write", 1.0) == pytest.approx(1.0)


def test_known_pipeline_phases_are_unchanged():
    assert _compute_overall("training", 0.0) == pytest.approx((0.05 + 0.15 + 0.02) / 0.90)


def test_unknown_phase_returns_none():
    assert _compute_overall("stage_input", 0.3) is None


def test_unknown_phase_never_jumps_to_done_or_goes_backwards(tmp_path):
    manager = JobManager(data_dir=tmp_path)
    fresh = manager.create(scene="img")
    manager._update_phase(fresh.id, "stage_input", 0.3, "", {})
    assert manager.get(fresh.id).overall_progress == pytest.approx(0.3)

    trained = manager.create(scene="vid")
    manager._update_phase(trained.id, "export", 1.0, "", {})
    manager._update_phase(trained.id, "eval", 0.1, "", {})
    assert manager.get(trained.id).overall_progress == pytest.approx(1.0)
    manager.shutdown(wait=False)


def test_completed_job_uses_result_download_url(tmp_path):
    manager = JobManager(data_dir=tmp_path)
    job = manager.create(scene="compose-s_1")
    manager._mark_completed(job.id, result={"download_url": "/compose/exports/s_1/download"})
    assert manager.get(job.id).download_url == "/compose/exports/s_1/download"
    manager.shutdown(wait=False)


def test_pipeline_job_keeps_legacy_download_url_and_ply_dir(tmp_path):
    (tmp_path / "vid" / "output" / "ply").mkdir(parents=True)
    manager = JobManager(data_dir=tmp_path)
    job = manager.create(scene="vid")
    manager._mark_completed(job.id, result={"scene_name": "vid"})
    done = manager.get(job.id)
    assert done.download_url == f"/download/{job.id}"
    assert done.ply_dir == str(tmp_path / "vid" / "output" / "ply")
    manager.shutdown(wait=False)


def test_job_without_ply_dir_or_download_url_has_no_download(tmp_path):
    manager = JobManager(data_dir=tmp_path)
    job = manager.create(scene="vid")
    manager._mark_completed(job.id, result=None)
    assert manager.get(job.id).download_url is None
    manager.shutdown(wait=False)

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.job_manager import _compute_overall
from backend.spirula_run import estimate as E
from backend.spirula_run import settings as S
from backend.spirula_run.gpu import parse_smi
from backend.spirula_run.routes import build_spirula_router
from backend.spirula_run.runner import LocalRunRequest, RunError, parse_step, run_local

FAKE = Path(__file__).with_name("fake_spirula.py")


@pytest.fixture
def fake_binary(tmp_path, monkeypatch):
    """A spirula.exe stand-in the runner can Popen directly."""
    if os.name == "nt":
        exe = tmp_path / "spirula.cmd"
        exe.write_text(f'@"{sys.executable}" "{FAKE}" %*\r\n', encoding="utf-8")
    else:
        exe = tmp_path / "spirula"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    monkeypatch.setenv("SPIRULA_EXE", str(exe))
    return exe


def _images(folder: Path, n: int = 12) -> Path:
    from PIL import Image

    folder.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (64, 48)).save(folder / f"{i:05d}.jpg")
    return folder


# --- settings / parsing -------------------------------------------------------

def test_preset_defaults_keep_the_final_pipeline_command():
    run = S.resolve(S.SpirulaSettings())
    assert (run.quality, run.iterations, run.cap_max, run.sfm_quality) == ("high", 50000, 3_000_000, "extreme")
    args = S.train_args("spirula", "ds", "sparse/0", "out", run)
    assert "--use-bilateral-grid" not in args and "--floater-suppression" not in args
    assert args[args.index("--disable-viewer") + 1] == "true"


def test_overrides_reach_the_train_command():
    run = S.resolve(S.SpirulaSettings(preset="baseline", cap_max=2_000_000, appearance=False,
                                      floater_suppression="strong", resolution_divisor=2))
    args = S.train_args("spirula", "ds", "sparse/0", "out", run, viewer_port=7010)
    pairs = dict(zip(args[::1], args[1::1]))
    assert pairs["--cap-max"] == "2000000" and pairs["--quality"] == "medium"
    assert pairs["--use-bilateral-grid"] == "false" and pairs["--use-ppisp"] == "false"
    assert pairs["--floater-suppression"] == "strong" and pairs["--train-resolution-divisor"] == "2"
    assert pairs["--viewer-port"] == "7010" and pairs["--disable-viewer"] == "false"


def test_settings_reject_unknown_fields():
    with pytest.raises(ValueError):
        S.SpirulaSettings(max_gaussians=5)


def test_parse_step_line():
    line = ("step   1200/50000 (  2%)  splats 412345  [elapsed 00:01:02.300 | ETA 00:40:00.000]  "
            "rgb_loss=0.03  ssim=0.9  psnr=21.25")
    assert parse_step(line) == {"step": 1200, "total": 50000, "splats": 412345, "eta": "00:40:00", "psnr": 21.25}
    assert parse_step("Cameras: 270") is None


def test_parse_smi():
    gpus = parse_smi("NVIDIA GeForce RTX 3060 Ti, 8192, 2100, 5900, 7\n")
    assert gpus == [{"name": "NVIDIA GeForce RTX 3060 Ti", "total_mib": 8192, "used_mib": 2100,
                     "free_mib": 5900, "util_pct": 7}]


# --- estimate -----------------------------------------------------------------

def test_estimate_grows_with_splats_and_resolution():
    small = E.estimate(S.SpirulaSettings(cap_max=1_000_000))
    big = E.estimate(S.SpirulaSettings(cap_max=50_000_000))
    half = E.estimate(S.SpirulaSettings(cap_max=1_000_000, resolution_divisor=2))
    assert 1200 < small.vram_mib < 1800          # measured 1458 MiB on the 3060 Ti
    assert big.vram_mib > 40_000
    assert half.vram_mib < small.vram_mib
    assert big.train_seconds > small.train_seconds


def test_estimate_fit_uses_free_local_memory():
    est = E.estimate(S.SpirulaSettings(cap_max=10_000_000), local_total_mib=8192, local_used_mib=2100)
    assert est.available_mib == 6092 and est.fits is False
    assert 4_000_000 < est.max_splats < 6_000_000
    assert E.estimate(S.SpirulaSettings(cap_max=10_000_000), gpu="A100").fits is True
    assert E.estimate(S.SpirulaSettings(), gpu="local").fits is None   # no nvidia-smi reading


def test_spirula_phases_drive_overall_progress():
    assert _compute_overall("spirula_frames", 0.0) == 0.0
    assert 0.2 < _compute_overall("spirula_train", 0.0) < 0.35
    assert _compute_overall("spirula_export", 1.0) == 1.0


# --- runner ---------------------------------------------------------------------

def test_run_local_images_end_to_end(tmp_path, fake_binary):
    images = _images(tmp_path / "capture")
    events = []
    result = run_local(LocalRunRequest("r1", "images", images, S.SpirulaSettings(iterations=1000), "oda"),
                       tmp_path / "data", lambda *a: events.append(a))
    splat = Path(result["splat_path"])
    assert splat.is_file() and splat.name == "splat.ply"
    assert result["sfm"]["registered_images"] == 12 and result["sfm"]["coverage"] == 1.0
    train = [e for e in events if e[0] == "spirula_train" and e[3].get("step")]
    assert train[-1][1] == 1.0 and train[-1][3]["splats"] == 10000 and train[-1][3]["viewer_url"].startswith("http")
    ws = tmp_path / "data" / "spirula" / "r1"
    assert (ws / "logs" / "train.log").is_file() and (ws / "manifest.json").is_file()
    args = (ws / "outputs" / "args.txt").read_text()
    assert "--viewer-port" in args and "--cap-max 3000000" in args
    assert all(0.0 <= e[1] <= 1.0 for e in events)


def test_run_local_stops_on_low_sfm_coverage(tmp_path, fake_binary, monkeypatch):
    monkeypatch.setenv("FAKE_SFM_SHARE", "0.5")
    images = _images(tmp_path / "capture")
    with pytest.raises(RunError, match="SfM kapsamı yetersiz"):
        run_local(LocalRunRequest("r2", "images", images, S.SpirulaSettings()), tmp_path / "data", lambda *a: None)
    # ...unless partial models are allowed.
    out = run_local(LocalRunRequest("r3", "images", images, S.SpirulaSettings(allow_partial=True, iterations=1000)),
                    tmp_path / "data", lambda *a: None)
    assert out["sfm"]["registered_images"] == 6


def test_run_local_needs_enough_images(tmp_path, fake_binary):
    images = _images(tmp_path / "capture", n=3)
    with pytest.raises(RunError, match="En az 10"):
        run_local(LocalRunRequest("r4", "images", images, S.SpirulaSettings()), tmp_path / "data", lambda *a: None)


# --- routes ---------------------------------------------------------------------

class FakeManager:
    def __init__(self):
        self.results, self.errors = {}, {}

    def create(self, scene: str, smoke_test: bool = False):
        return SimpleNamespace(id=f"job{len(self.results) + len(self.errors)}", scene=scene)

    def submit(self, job_id, runner):
        try:
            self.results[job_id] = runner(lambda *a, **k: None)
        except Exception as exc:  # noqa: BLE001
            self.errors[job_id] = exc


def _client(tmp_path):
    manager = FakeManager()
    app = FastAPI()
    app.include_router(build_spirula_router(tmp_path / "data", lambda: manager))
    return TestClient(app), manager


def test_estimate_route_lists_every_gpu(tmp_path, monkeypatch):
    from backend.spirula_run import routes
    monkeypatch.setattr(routes, "gpu_status", lambda: {"available": True, "error": None, "gpus": [
        {"name": "NVIDIA GeForce RTX 3060 Ti", "total_mib": 8192, "used_mib": 2000, "free_mib": 6192, "util_pct": 0}]})
    client, _ = _client(tmp_path)
    body = client.post("/spirula/estimate", json={"settings": {"cap_max": 50_000_000}}).json()
    rows = {r["gpu"]: r for r in body["estimates"]}
    assert set(rows) == set(E.GPU_PROFILES)
    assert rows["local"]["gpu_label"] == "Bu PC · RTX 3060 Ti" and rows["local"]["fits"] is False
    assert body["resolved"]["cap_max"] == 50_000_000
    assert client.post("/spirula/estimate", json={"settings": {"preset": "nope"}}).status_code == 422


def test_probe_and_local_run_routes(tmp_path, fake_binary):
    client, manager = _client(tmp_path)
    images = _images(tmp_path / "capture")
    probe = client.post("/spirula/probe", json={"input_mode": "images", "input_path": str(images)})
    assert probe.status_code == 200 and probe.json()["frames"] == 12 and probe.json()["width"] == 64
    started = client.post("/spirula/local", json={"input_mode": "images", "input_path": str(images),
                                                  "settings": {"iterations": 1000}, "name": "Oda 2"})
    assert started.status_code == 202
    run_id = started.json()["run_id"]
    assert run_id.endswith("Oda_2") and manager.results[started.json()["job_id"]]["download_url"].endswith("/splat")
    assert client.get("/spirula/runs").json()[0]["status"] == "completed"
    log = client.get(f"/spirula/runs/{run_id}/log").json()
    assert log["label"] == "train" and any("splats" in ln for ln in log["lines"])
    assert client.get(f"/spirula/runs/{run_id}/splat").content == b"ply"
    assert client.get("/spirula/runs/..%2Fx").status_code == 404


def test_failed_run_is_marked_in_manifest(tmp_path, fake_binary):
    client, manager = _client(tmp_path)
    images = _images(tmp_path / "capture", n=3)
    run_id = client.post("/spirula/local", json={"input_mode": "images", "input_path": str(images)}).json()["run_id"]
    assert isinstance(manager.errors["job0"], RunError)
    run = client.get(f"/spirula/runs/{run_id}").json()
    assert run["status"] == "failed" and "En az 10" in run["error"]


def test_local_start_rejects_missing_input(tmp_path, fake_binary):
    client, _ = _client(tmp_path)
    assert client.post("/spirula/local", json={"input_mode": "video", "input_path": str(tmp_path / "x.mov")}
                       ).status_code == 400

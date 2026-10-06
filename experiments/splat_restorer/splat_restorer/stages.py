"""Notebook-facing stages. Every stage is resumable: re-running skips work
whose durable output already exists on Drive."""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
from pathlib import Path

from . import dl3dv
from .pairs import extract_pairs, generate_scene_pairs
from .workspace import Workspace, atomic_write_json


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- preflight
def preflight(ws: Workspace) -> dict:
    import torch

    info = {}
    assert torch.cuda.is_available(), "No CUDA GPU. Runtime -> Change runtime type -> GPU."
    p = torch.cuda.get_device_properties(0)
    info["gpu"] = f"{p.name}, {p.total_memory / 2**30:.0f} GB, sm_{p.major}{p.minor}"
    info["torch"] = f"{torch.__version__} (CUDA {torch.version.cuda})"
    info["bf16"] = torch.cuda.is_bf16_supported()

    import gsplat

    from .gs_trainer import render

    # Tiny render: forces the gsplat CUDA build now rather than mid-run.
    params = {
        "means": torch.randn(64, 3, device="cuda") * 0.1 + torch.tensor([0, 0, 2.0], device="cuda"),
        "scales": torch.full((64, 3), -3.0, device="cuda"),
        "quats": torch.tensor([[1.0, 0, 0, 0]], device="cuda").repeat(64, 1),
        "opacities": torch.zeros(64, device="cuda"),
        "sh0": torch.rand(64, 1, 3, device="cuda"),
        "shN": torch.zeros(64, 15, 3, device="cuda"),
    }
    K = torch.tensor([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]], device="cuda")
    img, _ = render(params, torch.eye(4, device="cuda"), K, 128, 128, 3)
    assert img.shape == (128, 128, 3) and torch.isfinite(img).all()
    info["gsplat"] = f"{gsplat.__version__} (render OK)"

    import diffusers, peft, transformers

    info["diffusers"] = diffusers.__version__
    info["peft"] = peft.__version__
    info["transformers"] = transformers.__version__

    from huggingface_hub import HfApi, auth_check

    info["hf_user"] = HfApi().whoami()["name"]
    auth_check(dl3dv.REPO, repo_type="dataset")
    info["dl3dv_access"] = "OK"

    info["local_free_gb"] = round(shutil.disk_usage(ws.local).free / 2**30, 1)
    info["drive_free_gb"] = round(shutil.disk_usage(ws.root).free / 2**30, 1)
    for k, v in info.items():
        log(f"  {k:14s} {v}")
    if info["drive_free_gb"] < 30:
        log("  WARNING: under 30 GB free on Drive; pairs + checkpoints peak around 25-30 GB.")
    return info


# ----------------------------------------------------------------------------- split
def prepare_split(ws: Workspace, n_test: int, seed: int = 1234) -> dict:
    if ws.split_path.exists():
        split = json.loads(ws.split_path.read_text(encoding="utf-8"))
        log(f"split: reusing {ws.split_path} ({len(split['test'])} test, {len(split['train_pool'])} train pool)")
        return split
    scenes = dl3dv.list_scenes()
    labels = dl3dv.scene_labels()
    split = dl3dv.make_split(scenes, n_test=n_test, seed=seed, labels=labels)
    atomic_write_json(ws.split_path, split)
    log(f"split: {len(scenes)} scenes -> {len(split['test'])} test, {len(split['train_pool'])} train pool")
    return split


def done_scenes(ws: Workspace, scenes: list[str]) -> list[str]:
    return [s for s in scenes if ws.pair_tar(s).exists()]


def train_scenes(ws: Workspace, split: dict, n: int) -> list[str]:
    """First n train-pool scenes whose pairs exist (failed scenes are skipped)."""
    failed = set(_failures(ws))
    pool = [s for s in split["train_pool"] if s not in failed]
    return done_scenes(ws, pool)[:n]


def _failures(ws: Workspace) -> dict:
    p = ws.root / "failed_scenes.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ----------------------------------------------------------------------------- pairs
def generate_pairs(
    ws: Workspace,
    split: dict,
    n_train: int,
    recipes: list[str],
    max_pairs_per_recipe: int = 100,
    only_test: bool = False,
    time_budget_h: float | None = None,
) -> None:
    """Make pairs for all test scenes, then train-pool scenes until n_train succeed.

    Downloads the next scene in a background thread while the GPU trains on
    the current one.
    """
    t_start = time.time()
    test = set(split["test"])
    failed = _failures(ws)
    queue = [s for s in split["test"] if s not in failed]
    if not only_test:
        queue += [s for s in split["train_pool"] if s not in failed]

    def need_more_train() -> bool:
        return len(done_scenes(ws, [s for s in split["train_pool"] if s not in failed])) < n_train

    def wanted(s: str) -> bool:
        return not ws.pair_tar(s).exists() and (s in test or need_more_train())

    pending = [s for s in queue if wanted(s)]
    log(f"pairs: {len(done_scenes(ws, list(test)))}/{len(test)} test done, "
        f"{len(done_scenes(ws, split['train_pool']))} train done (target {n_train}), {len(pending)} candidates queued")

    prefetch: dict[str, dict] = {}

    def start_download(s: str):
        slot = {"done": threading.Event(), "dir": None, "error": None}

        def run():
            try:
                slot["dir"] = dl3dv.download_scene(s, ws.raw_dir(s))
            except Exception as e:  # noqa: BLE001 - reported below
                slot["error"] = e
            finally:
                slot["done"].set()

        threading.Thread(target=run, daemon=True).start()
        prefetch[s] = slot

    order = [s for s in pending]
    for k, scene in enumerate(order):
        if not wanted(scene):
            continue
        if time_budget_h and (time.time() - t_start) / 3600 > time_budget_h:
            log(f"pairs: time budget of {time_budget_h} h reached, stopping (re-run to continue)")
            break
        if scene not in prefetch:
            start_download(scene)
        nxt = next((s for s in order[k + 1 :] if wanted(s)), None)
        if nxt and nxt not in prefetch:
            start_download(nxt)
        slot = prefetch.pop(scene)
        slot["done"].wait()
        is_test = scene in test
        try:
            if slot["error"]:
                raise slot["error"]
            generate_scene_pairs(
                ws, scene, slot["dir"], recipes, is_test=is_test, max_pairs_per_recipe=max_pairs_per_recipe, log=log
            )
        except Exception as e:  # noqa: BLE001 - one bad scene must not stop the run
            failed[scene] = f"{type(e).__name__}: {e}"
            atomic_write_json(ws.root / "failed_scenes.json", failed)
            log(f"  {scene[:12]} FAILED: {failed[scene]}")
            traceback.print_exc()
        finally:
            dl3dv.remove_download(ws.raw_dir(scene))
        done_train = len(done_scenes(ws, split["train_pool"]))
        log(f"pairs progress: test {len(done_scenes(ws, list(test)))}/{len(test)}, train {done_train}/{n_train}, "
            f"elapsed {(time.time() - t_start) / 60:.0f} min")
        if not only_test and not need_more_train() and all(ws.pair_tar(s).exists() or s in failed for s in test):
            break
    # Do not leave half-downloaded prefetches around.
    for s in list(prefetch):
        prefetch[s]["done"].wait()
        dl3dv.remove_download(ws.raw_dir(s))


def generate_one(ws: Workspace, scene: str, recipes: list[str], is_test: bool, max_pairs_per_recipe: int = 100) -> Path:
    if ws.pair_tar(scene).exists():
        log(f"pairs for {scene[:12]} already on Drive")
        return ws.pair_tar(scene)
    try:
        scene_dir = dl3dv.download_scene(scene, ws.raw_dir(scene))
        return generate_scene_pairs(ws, scene, scene_dir, recipes, is_test=is_test,
                                    max_pairs_per_recipe=max_pairs_per_recipe, log=log)
    finally:
        dl3dv.remove_download(ws.raw_dir(scene))


def sanity_overfit(ws: Workspace, scene: str, steps: int = 300) -> bool:
    """Train the restorer briefly on ONE scene and check it beats its input there.

    This proves the plumbing (data, model, loss, optimiser) works before any
    long run. Outputs stay on local disk.
    """
    import csv as _csv

    from .train_restorer import RestorerConfig, train_restorer

    root = extract(ws, [scene])
    local_ws = Workspace(root=ws.local / "sanity", local=ws.local / "sanity_local")
    shutil.rmtree(local_ws.runs_dir / "sanity", ignore_errors=True)
    cfg = RestorerConfig(
        run_name="sanity", scenes=[scene], steps=steps, sample_every=100, log_every=10,
        local_ckpt_every=10**9, gram_warmup_steps=10**9, warmup_steps=50, num_workers=4,
    )
    train_restorer(local_ws, cfg, root, log=log)
    rows = [r for r in _csv.DictReader(open(local_ws.runs_dir / "sanity" / "log.csv")) if r["val_psnr"]]
    last = rows[-1]
    gain = float(last["val_psnr"]) - float(last["val_psnr_input"])
    ok = gain > 0.5
    log(f"SANITY {'PASSED' if ok else 'FAILED'}: val PSNR {float(last['val_psnr']):.2f} vs input "
        f"{float(last['val_psnr_input']):.2f} ({gain:+.2f} dB) after {steps} steps")
    (local_ws.runs_dir / "sanity" / "model_final.pt").unlink(missing_ok=True)
    return ok


def extract(ws: Workspace, scenes: list[str]) -> Path:
    t = time.time()
    root = extract_pairs(ws, scenes)
    log(f"extracted {len(scenes)} scenes to {root} in {time.time() - t:.0f}s")
    return root


# ----------------------------------------------------------------------------- reports
def show_markdown(path: Path) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 - plain text fallback outside notebooks
        print(path.read_text(encoding="utf-8"))


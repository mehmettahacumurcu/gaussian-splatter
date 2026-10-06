"""DL3DV-Benchmark access: scene listing, deterministic split, per-scene download."""
from __future__ import annotations

import csv
import io
import random
import shutil
from pathlib import Path

REPO = "DL3DV/DL3DV-Benchmark"
SCENE_SUBDIR = "gaussian_splat"  # undistorted images + COLMAP model
IMAGES = "images_4"  # 960x540


def list_scenes() -> list[str]:
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(REPO, repo_type="dataset", recursive=False)
    return sorted(e.path for e in entries if len(e.path) == 64 and all(c in "0123456789abcdef" for c in e.path))


def scene_labels() -> dict[str, str]:
    """scene hash -> coarse label (e.g. indoor/outdoor) from benchmark-meta.csv, if available."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(REPO, "benchmark-meta.csv", repo_type="dataset")
    except Exception:
        return {}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return {}
    hash_key = next((k for k in rows[0] if "hash" in k.lower()), None)
    # "environment" is bounded / unbounded (roughly indoor-ish vs. open scenes).
    label_key = next((k for k in rows[0] if k.lower() in ("environment", "env", "label")), None)
    if hash_key is None:
        return {}
    return {r[hash_key].strip(): (r.get(label_key) or "").strip() if label_key else "" for r in rows}


def make_split(scenes: list[str], n_test: int, seed: int = 1234, labels: dict[str, str] | None = None) -> dict:
    """Shuffle once; n_test scenes become test, the rest an ordered train pool.

    Test scenes are drawn round-robin over ``labels`` (e.g. bounded/unbounded)
    so both kinds are represented. Train runs with N scenes always take the
    first N of the pool, so a 10-scene run is a subset of the 25-scene run
    (clean scaling curve).
    """
    rng = random.Random(seed)
    order = list(scenes)
    rng.shuffle(order)
    labels = labels or {}
    groups: dict[str, list[str]] = {}
    for s in order:
        groups.setdefault(labels.get(s, ""), []).append(s)
    test: list[str] = []
    keys = sorted(groups)
    while len(test) < min(n_test, len(order)):
        for k in keys:
            if groups[k] and len(test) < n_test:
                test.append(groups[k].pop(0))
    test_set = set(test)
    return {
        "seed": seed,
        "test": test,
        "train_pool": [s for s in order if s not in test_set],
        "labels": {s: labels.get(s, "") for s in order},
    }


def download_scene(scene: str, dest: Path) -> Path:
    """Download images_4 + sparse model of one scene. Returns the scene dir."""
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=[f"{scene}/{SCENE_SUBDIR}/sparse/0/*", f"{scene}/{SCENE_SUBDIR}/{IMAGES}/*"],
        max_workers=16,
    )
    scene_dir = dest / scene / SCENE_SUBDIR
    if not (scene_dir / IMAGES).is_dir():
        raise FileNotFoundError(f"Download of {scene} produced no {IMAGES} folder")
    return scene_dir


def remove_download(dest: Path) -> None:
    shutil.rmtree(dest, ignore_errors=True)

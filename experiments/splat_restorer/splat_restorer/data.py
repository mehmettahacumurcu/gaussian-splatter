"""Pair datasets read from extracted pair tars (see pairs.py)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def collect_items(pairs_root: Path, scenes: list[str], recipes: list[str] | None = None) -> list[dict]:
    items = []
    for s in scenes:
        index = json.loads((pairs_root / s / "index.json").read_text(encoding="utf-8"))
        for rname, rec in index["recipes"].items():
            if recipes and rname not in recipes:
                continue
            for it in rec["items"]:
                items.append(
                    {
                        "scene": s,
                        "recipe": rname,
                        "frame": it["frame"],
                        "deg": str(pairs_root / s / it["deg"]),
                        "gt": str(pairs_root / s / it["gt"]),
                        "ref": str(pairs_root / s / it["ref_path"]),
                    }
                )
    return items


def _load(path: str) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def eval_crop_box(h: int, w: int, multiple: int = 64) -> tuple[int, int, int, int]:
    """Centred crop whose sides are multiples of ``multiple`` (top, left, h, w)."""
    ch, cw = h - h % multiple, w - w % multiple
    return (h - ch) // 2, (w - cw) // 2, ch, cw


class PairDataset(torch.utils.data.Dataset):
    """Returns deg / gt / ref tensors in [0, 1], shape (3, H, W).

    Training: one random ``crop`` x ``crop`` window (shared by all three images,
    the reference is a nearby view so the window roughly corresponds) plus a
    random horizontal flip. Evaluation: deterministic centred crop to multiples
    of 64 (540x960 -> 512x960).
    """

    def __init__(self, items: list[dict], train: bool, crop: int = 512):
        self.items = items
        self.train = train
        self.crop = crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        deg, gt, ref = _load(it["deg"]), _load(it["gt"]), _load(it["ref"])
        _, h, w = deg.shape
        if self.train:
            c = min(self.crop, h - h % 8, w - w % 8)
            top = random.randint(0, h - c)
            left = random.randint(0, w - c)
            sl = (slice(None), slice(top, top + c), slice(left, left + c))
            deg, gt = deg[sl], gt[sl]
            rh, rw = ref.shape[1:]
            rt, rl = min(top, rh - c), min(left, rw - c)
            ref = ref[:, rt : rt + c, rl : rl + c]
            if random.random() < 0.5:
                deg, gt, ref = deg.flip(-1), gt.flip(-1), ref.flip(-1)
        else:
            top, left, ch, cw = eval_crop_box(h, w)
            sl = (slice(None), slice(top, top + ch), slice(left, left + cw))
            deg, gt, ref = deg[sl], gt[sl], ref[sl]
        return {"deg": deg, "gt": gt, "ref": ref, "idx": idx}


def split_holdout(items: list[dict], frac: float = 0.02, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Small validation slice for loss curves (same scenes; not the real test)."""
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    n_val = max(1, int(len(items) * frac))
    val = [items[i] for i in order[:n_val]]
    train = [items[i] for i in order[n_val:]]
    return train, val

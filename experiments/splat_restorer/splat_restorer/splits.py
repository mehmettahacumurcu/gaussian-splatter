"""Frame splits for degraded reconstructions.

Every scene reserves a fixed set of *eval* frames (every ``eval_every``-th
frame) that no recipe ever trains on, so the distill-back test can measure the
splat on frames nobody has seen. Each recipe then picks its own sparse set of
*train* frames; everything else becomes a *pair* frame (render vs. photo).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Recipe:
    name: str
    train_stride: int  # keep every N-th non-eval frame for training
    steps: int  # optimisation steps for the degraded splat


# Two artefact regimes: very sparse (floaters, holes, smeared geometry) and
# moderately sparse (blur and view-dependent errors between views).
RECIPES: dict[str, Recipe] = {
    "sparse_hard": Recipe("sparse_hard", train_stride=12, steps=4000),
    "sparse_mild": Recipe("sparse_mild", train_stride=4, steps=4000),
}


@dataclass
class FrameSplit:
    train: list[int]
    pair: list[int]
    eval: list[int]


def eval_frames(num_frames: int, eval_every: int = 8) -> list[int]:
    return list(range(eval_every // 2, num_frames, eval_every))


def make_split(num_frames: int, recipe: Recipe, eval_every: int = 8) -> FrameSplit:
    ev = eval_frames(num_frames, eval_every)
    ev_set = set(ev)
    candidates = [i for i in range(num_frames) if i not in ev_set]
    train = candidates[:: recipe.train_stride]
    # Always include the last candidate so the trajectory end is covered.
    if candidates and candidates[-1] not in train:
        train.append(candidates[-1])
    train_set = set(train)
    pair = [i for i in candidates if i not in train_set]
    return FrameSplit(train=sorted(train), pair=pair, eval=ev)


def subsample(indices: list[int], max_count: int) -> list[int]:
    """Evenly spaced subset (deterministic) of at most max_count indices."""
    if len(indices) <= max_count:
        return list(indices)
    pos = np.linspace(0, len(indices) - 1, max_count).round().astype(int)
    return [indices[p] for p in sorted(set(pos.tolist()))]


def nearest_reference(
    query: int,
    candidates: list[int],
    centers: np.ndarray,
    forwards: np.ndarray,
    angle_weight: float = 0.5,
) -> int:
    """Training frame whose camera is closest in position and viewing direction."""
    c = np.asarray(candidates)
    d_pos = np.linalg.norm(centers[c] - centers[query], axis=1)
    d_pos = d_pos / (d_pos.max() + 1e-8)
    cos = np.clip(forwards[c] @ forwards[query], -1.0, 1.0)
    d_ang = (1.0 - cos) / 2.0
    return int(c[np.argmin(d_pos + angle_weight * d_ang)])

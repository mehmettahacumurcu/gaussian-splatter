"""Unit test: depth-warp transports pixels from anchor view to target view
using known depth + relative pose. Synthetic 2-cam fronto-parallel plane.

Run: python scripts/test_edit_warp.py
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from backend.edit.warp import warp_anchor_to_target


def test_zero_baseline_identity():
    """Anchor and target are the same camera → warp returns the anchor RGB."""
    H, W = 32, 32
    K = torch.tensor([[20.0, 0, 16], [0, 20, 16], [0, 0, 1]], dtype=torch.float32)
    w2c = torch.eye(4, dtype=torch.float32)

    torch.manual_seed(0)
    anchor_rgb = torch.rand(3, H, W)

    depth = torch.ones(H, W) * 1.0

    target_rgb = warp_anchor_to_target(
        anchor_rgb=anchor_rgb,
        target_depth=depth,
        K_anchor=K, K_target=K,
        w2c_anchor=w2c, w2c_target=w2c,
    )

    diff = (target_rgb - anchor_rgb).abs().mean().item()
    assert diff < 1e-3, f"Expected near-identity, got mean diff {diff}"


if __name__ == "__main__":
    test_zero_baseline_identity()
    print("ok: test_zero_baseline_identity")

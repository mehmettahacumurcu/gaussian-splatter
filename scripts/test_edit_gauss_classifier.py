"""Unit test: Gaussian classifier projects centers, counts mask-hit fraction,
thresholds. Synthetic 2-camera scene with one Gaussian on each side of a divider.

Run: python scripts/test_edit_gauss_classifier.py
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from backend.edit.gauss_classifier import classify_gaussians_for_deletion


def test_two_gauss_one_in_mask():
    H, W = 100, 100
    K = torch.tensor([[80.0, 0, 50], [0, 80, 50], [0, 0, 1]], dtype=torch.float32)

    w2c_a = torch.eye(4, dtype=torch.float32)
    w2c_b = torch.eye(4, dtype=torch.float32)
    w2c_b[0, 3] = 0.1

    w2cs = [w2c_a, w2c_b]

    means = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 1.0],
    ], dtype=torch.float32)

    mask_a = torch.zeros(H, W, dtype=torch.bool)
    mask_a[:, :50] = True
    mask_b = mask_a.clone()
    masks = torch.stack([mask_a, mask_b])

    flags = classify_gaussians_for_deletion(
        means=means, K=K, w2cs=w2cs, masks=masks, threshold=0.5,
        image_size=(W, H),
    )

    # Center pixel (50, 50) is NOT in mask (mask covers cols 0..49). Both gauss should be False.
    assert flags[0].item() is False, f"Expected False for center gaussian, got {flags[0]}"
    assert flags[1].item() is False, f"Expected False for right gaussian, got {flags[1]}"


def test_gauss_clearly_in_mask():
    H, W = 100, 100
    K = torch.tensor([[80.0, 0, 50], [0, 80, 50], [0, 0, 1]], dtype=torch.float32)
    w2c_a = torch.eye(4, dtype=torch.float32)
    w2cs = [w2c_a, w2c_a.clone()]

    # Gauss at (-0.4, 0, 1) — projects to ~(50 - 0.4*80, 50) = (18, 50) — left side
    means = torch.tensor([[-0.4, 0.0, 1.0]], dtype=torch.float32)

    mask_a = torch.zeros(H, W, dtype=torch.bool)
    mask_a[:, :50] = True
    masks = torch.stack([mask_a, mask_a.clone()])

    flags = classify_gaussians_for_deletion(
        means=means, K=K, w2cs=w2cs, masks=masks, threshold=0.5,
        image_size=(W, H),
    )
    assert flags[0].item() is True, f"Expected True for left-side gaussian, got {flags[0]}"


if __name__ == "__main__":
    test_two_gauss_one_in_mask()
    print("ok: test_two_gauss_one_in_mask")
    test_gauss_clearly_in_mask()
    print("ok: test_gauss_clearly_in_mask")

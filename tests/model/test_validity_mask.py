from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import pytest
import torch

from backend.model.losses_perceptual import LPIPSLoss
from backend.model.trainer import (
    Trainer4DGS,
    _masked_depth_loss,
    _prepare_validity_masks,
    _resize_validity_mask,
    compute_masked_recon_loss,
    compute_recon_loss,
    masked_psnr,
)


def test_none_keeps_the_legacy_reconstruction_expression() -> None:
    rendered = torch.zeros((11, 11, 3))
    target = torch.full_like(rendered, 0.25)

    expected = (rendered - target).abs().mean()
    actual = compute_recon_loss(rendered, target, lambda_ssim=0.0)

    assert torch.equal(actual, expected)
    assert (
        inspect.signature(Trainer4DGS.train).parameters["validity_mask"].default is None
    )


def test_masked_l1_is_normalized_by_valid_weight() -> None:
    rendered = torch.tensor(
        [[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]]
    )
    target = torch.zeros_like(rendered)
    validity = torch.tensor([[1.0, 0.0], [0.5, 0.0]])

    loss, skipped = compute_masked_recon_loss(
        rendered,
        target,
        lambda_ssim=0.0,
        validity=validity,
    )

    assert loss.item() == pytest.approx((3.0 + 0.75) / (1.5 * 3.0))
    assert skipped == ()


def test_invalid_pixels_have_zero_reconstruction_gradient() -> None:
    rendered = torch.ones((3, 3, 3), requires_grad=True)
    target = torch.zeros_like(rendered)
    validity = torch.ones((3, 3))
    validity[1, 1] = 0.0

    loss, _ = compute_masked_recon_loss(
        rendered,
        target,
        lambda_ssim=0.2,
        validity=validity,
    )
    loss.backward()

    assert torch.equal(rendered.grad[1, 1], torch.zeros(3))
    assert torch.any(rendered.grad[0, 0] != 0)


def test_empty_support_returns_connected_zero_and_records_skips() -> None:
    rendered = torch.ones((2, 2, 3), requires_grad=True)
    target = torch.zeros_like(rendered)

    loss, skipped = compute_masked_recon_loss(
        rendered,
        target,
        lambda_ssim=0.2,
        validity=torch.zeros((2, 2)),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert skipped == ("recon", "ssim")
    assert torch.equal(rendered.grad, torch.zeros_like(rendered))


def test_conservative_resize_never_restores_an_invalid_pixel() -> None:
    validity = torch.ones((4, 4))
    validity[1, 1] = 0.0

    down = _resize_validity_mask(validity, (2, 2))
    up = _resize_validity_mask(down, (4, 4))

    assert torch.equal(down, torch.tensor([[0.0, 1.0], [1.0, 1.0]]))
    assert torch.all(up[:2, :2] == 0.0)


@pytest.mark.parametrize(
    ("masks", "match"),
    [
        ((torch.ones((2, 2)),), "length"),
        ((torch.ones((2, 2, 1)), torch.ones((2, 2))), "shape"),
        ((torch.tensor([[1.1]]), torch.ones((1, 1))), "range"),
        ((torch.tensor([[float("nan")]]), torch.ones((1, 1))), "finite"),
    ],
)
def test_invalid_validity_inputs_fail_closed(
    masks: tuple[torch.Tensor, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _prepare_validity_masks(
            masks,
            frame_paths=(Path("a.png"), Path("b.png")),
            is_multiview=False,
            motion_masks_active=False,
        )


def test_validity_rejects_multiview_and_existing_motion_weighting() -> None:
    masks = (torch.ones((2, 2)), torch.ones((2, 2)))
    frames = (Path("a.png"), Path("b.png"))

    with pytest.raises(ValueError, match="multi-view"):
        _prepare_validity_masks(
            masks,
            frame_paths=frames,
            is_multiview=True,
            motion_masks_active=False,
        )
    with pytest.raises(ValueError, match="motion"):
        _prepare_validity_masks(
            masks,
            frame_paths=frames,
            is_multiview=False,
            motion_masks_active=True,
        )


def test_masked_psnr_ignores_invalid_error() -> None:
    target = torch.zeros((2, 2, 3))
    rendered = target.clone()
    rendered[0, 0] = 1.0
    validity = torch.ones((2, 2))
    validity[0, 0] = 0.0

    assert masked_psnr(rendered, target, validity) == float("inf")


def test_depth_loss_combines_soft_validity_and_edit_mask() -> None:
    rendered = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
    target = torch.ones((2, 2))
    validity = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
    edit = torch.tensor([[False, True], [False, False]])

    value, supported = _masked_depth_loss(rendered, target, validity, edit)

    assert supported
    assert value.item() == pytest.approx(1.0)


def test_spatial_lpips_returns_a_map_without_changing_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    class FakeModel(torch.nn.Module):
        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return (pred - target).abs().mean(dim=1, keepdim=True)

    def build(*, net: str, verbose: bool, spatial: bool = False) -> FakeModel:
        del net, verbose
        calls.append(spatial)
        return FakeModel()

    monkeypatch.setitem(sys.modules, "lpips", types.SimpleNamespace(LPIPS=build))
    pred = torch.zeros((3, 4, 5))
    target = torch.ones_like(pred)

    scalar = LPIPSLoss()(pred, target)
    spatial = LPIPSLoss(spatial=True)(pred, target)

    assert scalar.ndim == 0
    assert spatial.shape == (1, 1, 4, 5)
    assert calls == [False, True]

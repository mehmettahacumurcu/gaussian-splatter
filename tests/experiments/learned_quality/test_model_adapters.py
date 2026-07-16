from __future__ import annotations

from types import SimpleNamespace

import torch

from experiments.learned_quality.model_adapters import SeaRaftTorchAdapter


def test_sea_raft_honors_pinned_half_scale_and_restores_pixel_flow() -> None:
    seen_shapes = []

    class Model:
        def __call__(self, first, second, *, iters, test_mode):
            del second, iters, test_mode
            seen_shapes.append(tuple(first.shape))
            batch, _, height, width = first.shape
            return {
                "flow": [torch.ones((batch, 2, height, width))],
                "info": [torch.zeros((batch, 4, height, width))],
            }

    adapter = SeaRaftTorchAdapter.__new__(SeaRaftTorchAdapter)
    adapter.args = SimpleNamespace(scale=-1, iters=4, var_min=0, var_max=10)
    adapter.model = Model()
    adapter.device = "cpu"
    first = torch.zeros((2, 3, 8, 12))

    flow, uncertainty = adapter._infer(first, first)

    assert seen_shapes == [(2, 3, 4, 6)]
    assert flow.shape == (2, 2, 8, 12)
    assert uncertainty.shape == (2, 8, 12)
    torch.testing.assert_close(flow, torch.full_like(flow, 2.0))
    torch.testing.assert_close(uncertainty, torch.ones_like(uncertainty))

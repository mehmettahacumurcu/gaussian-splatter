from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from backend.export.to_splat import write_ply
from backend.model.trainer import compute_masked_recon_loss


@pytest.mark.gpu
@pytest.mark.integration
def test_masked_training_gpu_smoke_exports_portable_ply(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    targets = torch.rand((3, 16, 16, 3), device="cuda")
    validity = torch.ones((3, 16, 16), device="cuda")
    validity[:, 4:8, 5:10] = 0.0
    prediction = torch.nn.Parameter(torch.zeros_like(targets))
    optimizer = torch.optim.Adam((prediction,), lr=0.02)

    for step in range(100):
        index = step % len(targets)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = compute_masked_recon_loss(
            prediction[index], targets[index], 0.2, validity[index]
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.count_nonzero(prediction.grad[index, 4:8, 5:10]) == 0
        optimizer.step()

    ply = write_ply(
        means=np.zeros((1, 3), dtype=np.float32),
        log_scales=np.zeros((1, 3), dtype=np.float32),
        quats=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities_logit=np.zeros((1,), dtype=np.float32),
        sh_dc=np.zeros((1, 3), dtype=np.float32),
        sh_rest=np.zeros((1, 0, 3), dtype=np.float32),
        output_path=tmp_path / "frame_0000.ply",
    )
    payload = ply.read_bytes()
    assert payload.startswith((b"ply\n", b"ply\r\n"))
    assert b"element vertex 1" in payload[:1024]

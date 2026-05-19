"""Quick sanity check for the MSVC-wrapped runtime: does gsplat compile + run?"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Apply the shim BEFORE importing gsplat so its JIT picks up the patched flags.
from backend.image_to_scene import _gsplat_msvc_shim  # noqa: F401

import torch

print(f"torch: {torch.__version__}  cuda: {torch.cuda.is_available()}")

print("Importing gsplat...")
import gsplat
print(f"gsplat: {gsplat.__version__}")

print("Calling rasterization (this triggers JIT compile on first run; can take 2-5 min)...")
from gsplat import rasterization

N = 100
device = "cuda"
means = torch.randn(N, 3, device=device) * 0.3
means[:, 2] += 3.0
quats = torch.zeros(N, 4, device=device)
quats[:, 0] = 1.0
scales = torch.full((N, 3), 0.05, device=device)
opacities = torch.full((N,), 0.5, device=device)
colors = torch.rand(N, 3, device=device)
K = torch.tensor([[200., 0., 32.], [0., 200., 32.], [0., 0., 1.]], device=device)
w2c = torch.eye(4, device=device).unsqueeze(0)

out, alpha, info = rasterization(
    means=means, quats=quats, scales=scales, opacities=opacities,
    colors=colors,
    viewmats=w2c, Ks=K.unsqueeze(0), width=64, height=64,
    sh_degree=None,
)
print(f"rasterization OK; out shape: {out.shape}  alpha shape: {alpha.shape}")
print("DONE: gsplat works.")

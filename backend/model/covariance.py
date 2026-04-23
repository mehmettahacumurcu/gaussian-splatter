"""Faz 4c — 3D kovaryans inşası ve EWA splatting projeksiyonu.

3D kovaryans:  Σ = R · S · S^T · R^T   (PSD garantili)
2D projeksiyon: J · W · Σ · W^T · J^T   (EWA approximation)
"""
from __future__ import annotations
import torch


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    (N, 4) quat (w, x, y, z) → (N, 3, 3) rotasyon. Otomatik normalize.
    """
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def build_covariance_3d(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """
    (N, 3) scales (linear, > 0) + (N, 4) quat → (N, 3, 3) kovaryans.
    Σ = R S S^T R^T → her zaman PSD.
    """
    S = torch.diag_embed(scales)               # (N, 3, 3)
    R = quat_to_rotmat(quats)                  # (N, 3, 3)
    RS = torch.bmm(R, S)
    return torch.bmm(RS, RS.transpose(1, 2))


def project_gaussians(
    means: torch.Tensor,        # (N, 3) world
    covs3d: torch.Tensor,       # (N, 3, 3)
    K: torch.Tensor,            # (3, 3) intrinsics
    W: torch.Tensor,            # (4, 4) world-to-camera
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    EWA splatting: 3D Gaussian → 2D Gaussian.

    Returns:
        means2d: (N, 2)  pixel coords
        covs2d:  (N, 2, 2)
        z:       (N,)    kamera-uzayı derinliği
    """
    N = means.shape[0]
    device = means.device

    # World → camera
    means_h = torch.cat([means, torch.ones(N, 1, device=device)], dim=-1)  # (N, 4)
    means_cam = (W @ means_h.T).T[:, :3]                                   # (N, 3)
    z = means_cam[:, 2].clamp(min=0.01)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pinhole projection
    means2d = torch.stack([
        fx * means_cam[:, 0] / z + cx,
        fy * means_cam[:, 1] / z + cy,
    ], dim=-1)

    # Jacobian d(pixel)/d(cam)  (N, 2, 3)
    J = torch.zeros(N, 2, 3, device=device)
    J[:, 0, 0] = fx / z
    J[:, 1, 1] = fy / z
    J[:, 0, 2] = -fx * means_cam[:, 0] / (z * z)
    J[:, 1, 2] = -fy * means_cam[:, 1] / (z * z)

    # W'nin sadece rotasyon kısmı (3x3)
    W_rot = W[:3, :3].unsqueeze(0)                                        # (1, 3, 3)
    JW = torch.bmm(J, W_rot.expand(N, -1, -1))                            # (N, 2, 3)

    covs2d = torch.bmm(torch.bmm(JW, covs3d), JW.transpose(1, 2))        # (N, 2, 2)
    return means2d, covs2d, z


# ---------------------------------------------------------------------------
# Hızlı sağlık testi
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    N = 100
    scales = torch.rand(N, 3) * 0.1 + 0.01
    quats = torch.randn(N, 4)
    cov3d = build_covariance_3d(scales, quats)

    # PSD kontrolü: tüm eigenvalue'lar ≥ 0
    eigs = torch.linalg.eigvalsh(cov3d)
    print(f"3D cov shape: {cov3d.shape}, min eig = {eigs.min():.6f} (≥ 0 olmalı)")

    means = torch.randn(N, 3) * 2 + torch.tensor([0., 0., 5.])
    K = torch.tensor([[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]])
    W = torch.eye(4)

    m2d, c2d, z = project_gaussians(means, cov3d, K, W)
    print(f"2D means: {m2d.shape}, mean depth: {z.mean():.3f}")
    print(f"2D cov:   {c2d.shape}")

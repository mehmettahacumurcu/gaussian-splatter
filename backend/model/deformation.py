"""Faz 4b — HexPlane + MLP deformation field.

Statik Gaussian'lar için zaman-bağımlı (Δpos, Δquat, Δscale) sapması öğrenir.
HexPlane (4DGaussians paper) 4D uzayı 6 düzleme ayırarak O(n²) bellek kullanır.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class HexPlane(nn.Module):
    """6 düzlem: XY, XZ, YZ, XT, YT, ZT — feature dim'i çarp veya topla."""

    def __init__(self, resolution: int = 64, feat_dim: int = 32):
        super().__init__()
        self.resolution = resolution
        self.feat_dim = feat_dim
        self.planes = nn.ParameterList([
            nn.Parameter(torch.randn(1, feat_dim, resolution, resolution) * 0.01)
            for _ in range(6)
        ])

    @staticmethod
    def _sample_plane(plane: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        """
        Bilinear sample (N, feat_dim).
        c1, c2: (N,) ∈ [-1, 1]
        """
        grid = torch.stack([c1, c2], dim=-1).view(1, 1, -1, 2)
        feat = F.grid_sample(plane, grid, mode='bilinear',
                             align_corners=True, padding_mode='border')
        return feat.squeeze(0).squeeze(1).T   # (N, feat_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, y, z, t: (N,) ∈ [-1, 1]
        Returns:
            (N, feat_dim * 6) — 6 düzlemden örneklenmiş feature'lar concat
        """
        feats = [
            self._sample_plane(self.planes[0], x, y),  # XY
            self._sample_plane(self.planes[1], x, z),  # XZ
            self._sample_plane(self.planes[2], y, z),  # YZ
            self._sample_plane(self.planes[3], x, t),  # XT
            self._sample_plane(self.planes[4], y, t),  # YT
            self._sample_plane(self.planes[5], z, t),  # ZT
        ]
        return torch.cat(feats, dim=-1)


class DeformationField(nn.Module):
    """HexPlane + MLP → (Δpos, Δquat, Δscale)."""

    def __init__(self, resolution: int = 64, feat_dim: int = 32, mlp_width: int = 256):
        super().__init__()
        self.hexplane = HexPlane(resolution, feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 6, mlp_width), nn.SiLU(),
            nn.Linear(mlp_width, mlp_width),    nn.SiLU(),
            nn.Linear(mlp_width, 10),           # Δpos (3) + Δquat (4) + Δscale (3)
        )
        # Son katmanı sıfır ile başlat → eğitim başında identity deformation
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _normalize_means(means: torch.Tensor, scene_extent: float) -> torch.Tensor:
        """Means'i [-1, 1]'e normalize et (sahne kapsamına göre)."""
        return torch.clamp(means / max(scene_extent, 1e-6), -1.0, 1.0)

    def forward(
        self,
        means: torch.Tensor,        # (N, 3) — orijinal ya da normalize edilmiş
        t: float,                   # ∈ [0, 1]
        scene_extent: float = 1.0,  # means henüz normalize değilse
        already_normalized: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta_pos:   (N, 3)
            delta_quat:  (N, 4)
            delta_scale: (N, 3)  — log-space (yani eklenir, sonra exp alınır)
        """
        if not already_normalized:
            means = self._normalize_means(means, scene_extent)

        x, y, z = means[:, 0], means[:, 1], means[:, 2]
        t_vec = torch.full_like(x, t * 2.0 - 1.0)   # [0,1] → [-1,1]

        feat = self.hexplane(x, y, z, t_vec)        # (N, feat_dim * 6)
        delta = self.mlp(feat)                      # (N, 10)

        return delta[:, :3], delta[:, 3:7], delta[:, 7:10]


# ---------------------------------------------------------------------------
# Hızlı sağlık testi
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    field = DeformationField(resolution=32, feat_dim=16, mlp_width=64)
    means = torch.randn(500, 3)
    dpos, dquat, dscale = field(means, t=0.5, scene_extent=2.0)
    print(f"Δpos:   {dpos.shape}, max abs = {dpos.abs().max():.6f} (zero-init → 0 olmalı)")
    print(f"Δquat:  {dquat.shape}")
    print(f"Δscale: {dscale.shape}")
    # Param sayısı
    n_params = sum(p.numel() for p in field.parameters())
    print(f"Toplam parametre: {n_params:,}")

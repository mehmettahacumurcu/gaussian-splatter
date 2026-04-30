"""Faz 4b — HexPlane + MLP deformation field.

Statik Gaussian'lar için zaman-bağımlı (Δpos, Δquat, Δscale) sapması öğrenir.
HexPlane (4DGaussians paper) 4D uzayı 6 düzleme ayırarak O(n²) bellek kullanır.

v2 — genişletilmiş mimari:
  - Varsayılan mlp_width 256 → 512, mlp_depth 2 → 4 hidden layer
  - HexPlane resolution 64 → 96, feat_dim 32 → 48
  - Zaman t için Fourier positional encoding (num_time_freqs varsayılan 6)
    Bu MLP'nin zaman boyutunu daha iyi ayırt etmesini sağlar —
    linear t ile MLP genelde zamansız davranır.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HexPlane(nn.Module):
    """6 düzlem: XY, XZ, YZ, XT, YT, ZT — feature dim'i concat."""

    def __init__(self, resolution: int = 96, feat_dim: int = 48):
        super().__init__()
        self.resolution = resolution
        self.feat_dim = feat_dim
        self.planes = nn.ParameterList([
            nn.Parameter(torch.randn(1, feat_dim, resolution, resolution) * 0.01)
            for _ in range(6)
        ])

    @staticmethod
    def _sample_plane(plane: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        grid = torch.stack([c1, c2], dim=-1).view(1, 1, -1, 2)
        feat = F.grid_sample(plane, grid, mode="bilinear",
                             align_corners=True, padding_mode="border")
        return feat.squeeze(0).squeeze(1).T   # (N, feat_dim)

    def forward(self, x, y, z, t):
        feats = [
            self._sample_plane(self.planes[0], x, y),  # XY
            self._sample_plane(self.planes[1], x, z),  # XZ
            self._sample_plane(self.planes[2], y, z),  # YZ
            self._sample_plane(self.planes[3], x, t),  # XT
            self._sample_plane(self.planes[4], y, t),  # YT
            self._sample_plane(self.planes[5], z, t),  # ZT
        ]
        return torch.cat(feats, dim=-1)


class MultiResHexPlane(nn.Module):
    """Phase 2.5 — Multi-resolution HexPlane stack (4DGaussians-style).

    Multiple HexPlane scale'lerinde sample edip feature'lari concat eder.
    Coarse plane düşük frekansli motion'i ogrenir, fine plane yüksek detay.

    Args:
        resolutions: list of int, orn [12, 24, 48, 96]
        feat_dim:    her resolution icin feat dim (toplam = sum * 6 plane)
    """

    def __init__(self, resolutions: list[int] = (24, 48, 96), feat_dim: int = 24):
        super().__init__()
        self.resolutions = list(resolutions)
        self.feat_dim = feat_dim
        self.planes_list = nn.ModuleList([
            HexPlane(resolution=r, feat_dim=feat_dim) for r in self.resolutions
        ])

    @property
    def total_feat_dim(self) -> int:
        # Her HexPlane.forward 6 plane'i concat -> feat_dim * 6
        # Multi-res: ayrica scale sayisi kadar
        return self.feat_dim * 6 * len(self.resolutions)

    def forward(self, x, y, z, t):
        feats = [hp(x, y, z, t) for hp in self.planes_list]
        return torch.cat(feats, dim=-1)


def fourier_encode_scalar(value: float, num_freqs: int, device) -> torch.Tensor:
    """
    [t, sin(πt), cos(πt), sin(2πt), cos(2πt), ..., sin(2^(L-1)·πt), cos(...)].
    Size: 1 + 2*num_freqs.
    """
    base = torch.tensor([value], device=device)
    encs = [base]
    for i in range(num_freqs):
        freq = (2.0 ** i) * math.pi
        encs.append(torch.sin(freq * base))
        encs.append(torch.cos(freq * base))
    return torch.cat(encs, dim=0)


def decode_fourier_trajectory(
    coeffs: torch.Tensor,   # (N, K, 2, 3) — sin/cos × xyz
    t: float,               # ∈ [0, 1]
) -> torch.Tensor:
    """
    Per-gaussian Fourier trajectory decode.
    4DGS paper (Yang et al 2024) yaklaşımı: her gaussian kendi trajectory'sini öğrenir.

    Δpos(i, t) = Σ_{k=1}^{K} [A_{ik} · sin(2πkt) + B_{ik} · cos(2πkt)]

    Ama DC koşulu için: t=0'da Δpos=0 olsun istiyoruz (tüm B_k = 0 başta, öğrenilir).
    Yani B term'leri t=0'da toplamı 0 olmayabilir ama bu OK, sadece motion relative.

    Args:
        coeffs: (N, K, 2, 3) tensor. son axes: [sin_coef, cos_coef] × [x, y, z]
        t: zaman değeri [0, 1]

    Returns:
        (N, 3) Δpos tensor
    """
    if coeffs is None or coeffs.shape[1] == 0:
        return torch.zeros(coeffs.shape[0] if coeffs is not None else 0, 3,
                           device=coeffs.device if coeffs is not None else "cpu")

    N, K, _, _ = coeffs.shape
    device = coeffs.device
    # Frekanslar: 2π·1, 2π·2, ..., 2π·K
    freqs = torch.arange(1, K + 1, device=device, dtype=coeffs.dtype)  # (K,)
    phases = 2.0 * math.pi * freqs * t                                 # (K,)
    sin_vals = torch.sin(phases)                                       # (K,)
    cos_vals = torch.cos(phases)                                       # (K,)

    # coeffs[:, :, 0, :] → A (sin coef), shape (N, K, 3)
    # coeffs[:, :, 1, :] → B (cos coef), shape (N, K, 3)
    # Çıkış: Σ_k A_k·sin(2πkt) + B_k·cos(2πkt)  → (N, 3)
    delta = (
        (coeffs[:, :, 0, :] * sin_vals[None, :, None]).sum(dim=1) +
        (coeffs[:, :, 1, :] * cos_vals[None, :, None]).sum(dim=1)
    )
    return delta


class DeformationField(nn.Module):
    """HexPlane + MLP → (Δpos, Δquat, Δscale). Fourier-encoded t."""

    def __init__(
        self,
        resolution: int = 96,
        feat_dim: int = 48,
        mlp_width: int = 512,
        mlp_depth: int = 4,         # hidden layer sayısı (önceki: implicit 2)
        num_time_freqs: int = 6,    # Fourier frekans sayısı (0 = kapalı)
        # Phase 2.5 — Multi-resolution HexPlane (default off, single-res)
        multires_resolutions: list | None = None,
        multires_feat_dim: int | None = None,
    ):
        super().__init__()
        self.num_time_freqs = num_time_freqs

        # Phase 2.5 — Multi-res toggle
        if multires_resolutions:
            mr_feat = multires_feat_dim if multires_feat_dim else feat_dim
            self.hexplane = MultiResHexPlane(
                resolutions=list(multires_resolutions), feat_dim=mr_feat,
            )
            input_hexplane_dim = self.hexplane.total_feat_dim
            print(f"[deformation] MultiResHexPlane: resolutions={multires_resolutions} "
                  f"feat_dim={mr_feat} total={input_hexplane_dim}")
        else:
            self.hexplane = HexPlane(resolution, feat_dim)
            input_hexplane_dim = feat_dim * 6

        # Girdi: HexPlane features + Fourier time (1 + 2*L)
        t_dim = 1 + 2 * num_time_freqs
        input_dim = input_hexplane_dim + t_dim

        # MLP: input → width → width → ... (mlp_depth katman) → 10
        layers = []
        prev = input_dim
        for _ in range(mlp_depth):
            layers.append(nn.Linear(prev, mlp_width))
            layers.append(nn.SiLU())
            prev = mlp_width
        layers.append(nn.Linear(prev, 10))      # Δpos(3) + Δquat(4) + Δscale(3)
        self.mlp = nn.Sequential(*layers)

        # Son katmanı sıfır ile başlat → eğitim başında identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _normalize_means(means: torch.Tensor, scene_extent: float) -> torch.Tensor:
        return torch.clamp(means / max(scene_extent, 1e-6), -1.0, 1.0)

    def forward(
        self,
        means: torch.Tensor,        # (N, 3)
        t: float,                   # ∈ [0, 1]
        scene_extent: float = 1.0,
        already_normalized: bool = False,
    ):
        if not already_normalized:
            means = self._normalize_means(means, scene_extent)

        x, y, z = means[:, 0], means[:, 1], means[:, 2]
        t_norm = t * 2.0 - 1.0  # → [-1, 1]
        t_vec = torch.full_like(x, t_norm)

        # HexPlane feat (N, feat_dim*6)
        hex_feat = self.hexplane(x, y, z, t_vec)

        # Fourier time encoding — aynı değer her gaussian için (N kopyası)
        t_enc = fourier_encode_scalar(t_norm, self.num_time_freqs, means.device)
        t_enc_broadcast = t_enc.unsqueeze(0).expand(means.shape[0], -1)

        feat = torch.cat([hex_feat, t_enc_broadcast], dim=-1)
        delta = self.mlp(feat)

        return delta[:, :3], delta[:, 3:7], delta[:, 7:10]


# ---------------------------------------------------------------------------
# Hızlı sağlık testi
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    field = DeformationField()
    means = torch.randn(500, 3)
    dpos, dquat, dscale = field(means, t=0.5, scene_extent=2.0)
    n_params = sum(p.numel() for p in field.parameters())
    print(f"Default config params: {n_params:,}")
    print(f"Δpos max abs (zero-init): {dpos.abs().max():.6f}")

    field_small = DeformationField(resolution=64, feat_dim=32, mlp_width=256, mlp_depth=2)
    n_small = sum(p.numel() for p in field_small.parameters())
    print(f"Previous default config params: {n_small:,}")

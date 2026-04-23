"""Faz 4a — Eğitilebilir 3D Gaussian model.

Her Gaussian'ın parametreleri:
  - means (N, 3)        — 3D pozisyon
  - scales (N, 3)       — log-space, exp() ile pozitif olur
  - quats (N, 4)        — quaternion (w, x, y, z), forward'da normalize
  - opacities (N, 1)    — logit-space, sigmoid ile [0, 1]
  - sh_dc (N, 1, 3)     — Spherical Harmonics DC (sabit renk bileşeni)
  - sh_rest (N, K, 3)   — SH yüksek frekans (K = (deg+1)² - 1)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


def _rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """[0, 1] RGB → SH DC bileşeni (degree 0)."""
    C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
    return (rgb - 0.5) / C0


class GaussianModel(nn.Module):
    """Eğitilebilir 3D Gaussian Splatting modeli."""

    def __init__(
        self,
        init_points: torch.Tensor,
        init_colors: torch.Tensor | None = None,
        sh_degree: int = 3,
    ):
        """
        Args:
            init_points: (N, 3) — COLMAP sparse point cloud
            init_colors: (N, 3) — [0, 1] başlangıç renkleri (None → gri)
            sh_degree:   Spherical Harmonics derecesi (0–3)
        """
        super().__init__()
        N = len(init_points)
        if N == 0:
            raise ValueError("init_points boş — COLMAP rekonstrüksiyonu başarısız mı?")

        self.sh_degree = sh_degree
        # Toplam SH katsayı sayısı (deg+1)²; rest = total - 1
        self.num_sh_rest = (sh_degree + 1) ** 2 - 1

        # ----- Pozisyon -----
        self.means = nn.Parameter(init_points.float())                                # (N, 3)

        # ----- Scale (log-space) -----
        # En yakın komşu mesafesinin yarısı ile başlat
        with torch.no_grad():
            init_scale = self._estimate_init_scale(init_points)
        log_scale = torch.log(init_scale).unsqueeze(-1).expand(-1, 3).clone()         # (N, 3)
        self.scales = nn.Parameter(log_scale)

        # ----- Rotasyon (identity quaternion) -----
        quats = torch.zeros(N, 4)
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats)                                              # (N, 4)

        # ----- Opasite (logit-space; sigmoid(-2) ≈ 0.119) -----
        self.opacities = nn.Parameter(torch.full((N, 1), -2.0))                       # (N, 1)

        # ----- Renk: SH -----
        if init_colors is None:
            init_colors = torch.full((N, 3), 0.5)
        sh_dc = _rgb_to_sh_dc(init_colors).unsqueeze(1)                               # (N, 1, 3)
        self.sh_dc = nn.Parameter(sh_dc)
        self.sh_rest = nn.Parameter(torch.zeros(N, self.num_sh_rest, 3))              # (N, K, 3)

    # ------------------------------------------------------------------
    # Yardımcı: başlangıç scale'ini KNN ile tahmin et
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_init_scale(points: torch.Tensor, k: int = 3) -> torch.Tensor:
        """En yakın k komşu ortalama mesafesi → scale (N,)."""
        N = points.shape[0]
        if N <= k + 1:
            return torch.full((N,), 0.01)

        # Bellek dostu chunked nearest-neighbor (büyük N için)
        chunk = 4096
        dists = []
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            d = torch.cdist(points[s:e], points)        # (chunk, N)
            d, _ = torch.topk(d, k=k + 1, largest=False)  # +1: kendisi
            dists.append(d[:, 1:].mean(dim=1))           # kendini at
        mean_dist = torch.cat(dists, dim=0)              # (N,)
        # Scale = mesafenin yarısı (komşu Gaussian'larla hafif örtüşme)
        return torch.clamp(mean_dist * 0.5, min=1e-6, max=1.0)

    # ------------------------------------------------------------------
    # Aktive edilmiş paramlara erişim
    # ------------------------------------------------------------------
    @property
    def get_scales(self) -> torch.Tensor:
        return torch.exp(self.scales)              # (N, 3) > 0

    @property
    def get_opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.opacities)       # (N, 1) ∈ [0, 1]

    @property
    def get_quats(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.quats, dim=-1)

    @property
    def get_colors(self) -> torch.Tensor:
        """SH katsayıları (N, K_total, 3)."""
        return torch.cat([self.sh_dc, self.sh_rest], dim=1)

    @property
    def num_points(self) -> int:
        return int(self.means.shape[0])

    # ------------------------------------------------------------------
    # Density control: prune
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prune_gaussians(self, min_opacity: float = 0.005, max_scale: float = 0.1) -> int:
        """Düşük opasite veya çok büyük Gaussian'ları sil. Silinen sayıyı döner."""
        keep = (
            (self.get_opacities.squeeze(-1) > min_opacity) &
            (self.get_scales.max(dim=-1).values < max_scale)
        )
        n_pruned = int((~keep).sum().item())
        if n_pruned > 0:
            self._apply_mask(keep)
        return n_pruned

    @torch.no_grad()
    def _apply_mask(self, mask: torch.Tensor) -> None:
        """Bool maskesi ile tüm parametreleri filtrele (in-place yeniden bağla)."""
        self.means     = nn.Parameter(self.means[mask].contiguous())
        self.scales    = nn.Parameter(self.scales[mask].contiguous())
        self.quats     = nn.Parameter(self.quats[mask].contiguous())
        self.opacities = nn.Parameter(self.opacities[mask].contiguous())
        self.sh_dc     = nn.Parameter(self.sh_dc[mask].contiguous())
        self.sh_rest   = nn.Parameter(self.sh_rest[mask].contiguous())

    @torch.no_grad()
    def append_gaussians(
        self,
        new_means: torch.Tensor,
        new_scales: torch.Tensor,
        new_quats: torch.Tensor,
        new_opacities: torch.Tensor,
        new_sh_dc: torch.Tensor,
        new_sh_rest: torch.Tensor,
    ) -> None:
        """Yeni Gaussian'ları (clone/split sonucu) listenin sonuna ekle."""
        device = self.means.device
        cat = lambda a, b: torch.cat([a, b.to(device)], dim=0).contiguous()
        self.means     = nn.Parameter(cat(self.means, new_means))
        self.scales    = nn.Parameter(cat(self.scales, new_scales))
        self.quats     = nn.Parameter(cat(self.quats, new_quats))
        self.opacities = nn.Parameter(cat(self.opacities, new_opacities))
        self.sh_dc     = nn.Parameter(cat(self.sh_dc, new_sh_dc))
        self.sh_rest   = nn.Parameter(cat(self.sh_rest, new_sh_rest))

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def state_for_save(self) -> dict:
        """Checkpoint için (CPU) sözlük."""
        return {k: v.detach().cpu() for k, v in self.state_dict().items()} | {
            "sh_degree": self.sh_degree,
            "num_points": self.num_points,
        }

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "GaussianModel":
        """Boş bir model oluştur ve ağırlıkları yükle."""
        N = int(ckpt["num_points"])
        sh_deg = int(ckpt.get("sh_degree", 3))
        # Dummy noktalar — sonra state_dict ile üzerine yazılacak
        dummy = torch.zeros(N, 3)
        m = cls(dummy, sh_degree=sh_deg)
        # state_dict yalnızca tensor key'leri içersin
        sd = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        m.load_state_dict(sd, strict=True)
        return m


# ---------------------------------------------------------------------------
# Hızlı sağlık testi (CPU'da çalışır)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pts = torch.randn(1000, 3)
    rgb = torch.rand(1000, 3)
    m = GaussianModel(pts, init_colors=rgb, sh_degree=3)
    print(f"Means:     {m.means.shape}")
    print(f"Scales:    {m.get_scales.shape}, range=[{m.get_scales.min():.4f}, {m.get_scales.max():.4f}]")
    print(f"Opacities: {m.get_opacities.shape}, mean={m.get_opacities.mean():.4f}")
    print(f"SH dc:     {m.sh_dc.shape}")
    print(f"SH rest:   {m.sh_rest.shape}  (K = (deg+1)²-1 = {m.num_sh_rest})")
    # Prune test
    n_pruned = m.prune_gaussians(min_opacity=0.5)
    print(f"Prune (min_opacity=0.5) → {n_pruned} silindi, kalan: {m.num_points}")

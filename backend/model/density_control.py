"""Faz 5b — Adaptive Density Control (ADC).

Orijinal 3DGS makalesindeki yaklaşıma yakın:
  - prune: opasite çok düşük ya da scale çok büyük → sil
  - clone:  küçük + yüksek pozisyon gradyanı → tıpkısını ekle
  - split:  büyük + yüksek pozisyon gradyanı → ikiye böl, scale'i küçült
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .gaussian_model import GaussianModel


class DensityController:
    """Pozisyon gradyanlarını biriktiren ve clone/split/prune kararı veren yardımcı."""

    def __init__(
        self,
        grad_threshold: float = 2e-4,
        scale_split_threshold: float = 0.01,
        min_opacity: float = 0.005,
        max_scale: float = 0.1,
        max_gaussians: int = 0,   # 0 = sınırsız; >0 ise N >= cap iken split kapalı
    ):
        self.grad_threshold = grad_threshold
        self.scale_split_threshold = scale_split_threshold
        self.min_opacity = min_opacity
        self.max_scale = max_scale
        self.max_gaussians = int(max_gaussians)

        self._grad_accum: torch.Tensor | None = None
        self._n_obs: torch.Tensor | None = None

    def reset(self, num_points: int, device: torch.device) -> None:
        self._grad_accum = torch.zeros(num_points, device=device)
        self._n_obs = torch.zeros(num_points, device=device)

    def accumulate(self, gs: GaussianModel) -> None:
        """Bir backward'dan sonra çağır — pozisyon gradyanını biriktir."""
        if gs.means.grad is None:
            return
        if self._grad_accum is None or len(self._grad_accum) != gs.num_points:
            self.reset(gs.num_points, gs.means.device)

        grad_norm = gs.means.grad.norm(dim=-1)
        self._grad_accum += grad_norm
        self._n_obs += 1

    @torch.no_grad()
    def step(self, gs: GaussianModel) -> dict:
        """Clone + split + prune. İstatistikleri döner."""
        stats = {"cloned": 0, "split": 0, "pruned": 0, "before": gs.num_points}

        if self._grad_accum is not None and self._n_obs is not None:
            obs = self._n_obs.clamp(min=1)
            avg_grad = self._grad_accum / obs

            high_grad = avg_grad > self.grad_threshold
            scale_max = gs.get_scales.max(dim=-1).values
            small = scale_max <= self.scale_split_threshold
            large = scale_max > self.scale_split_threshold

            clone_mask = high_grad & small
            split_mask = high_grad & large

            # v3.7.2: Hard cap on N — eğer cap'i aştıysak split/clone kapa,
            # sadece prune yapsın. Banana ultra'da N=164k oldu, render çöktü.
            if self.max_gaussians > 0 and gs.num_points >= self.max_gaussians:
                clone_mask = torch.zeros_like(clone_mask)
                split_mask = torch.zeros_like(split_mask)

            # Clone: aynı parametrelerle kopyala
            n_cloned = int(clone_mask.sum().item())
            if n_cloned > 0:
                gs.append_gaussians(
                    new_means     = gs.means[clone_mask].clone(),
                    new_scales    = gs.scales[clone_mask].clone(),
                    new_quats     = gs.quats[clone_mask].clone(),
                    new_opacities = gs.opacities[clone_mask].clone(),
                    new_sh_dc     = gs.sh_dc[clone_mask].clone(),
                    new_sh_rest   = gs.sh_rest[clone_mask].clone(),
                )
                stats["cloned"] = n_cloned

            # Split: aynı bölgede biraz kaymış 2 yeni Gaussian, scale küçült
            n_split = int(split_mask.sum().item())
            if n_split > 0:
                idx = split_mask.nonzero(as_tuple=False).squeeze(-1)
                # Her Gaussian'dan 2 tane üret
                noise = torch.randn(n_split * 2, 3, device=gs.means.device)
                base_scale = gs.get_scales[idx].repeat(2, 1)
                offset = noise * base_scale * 0.5

                new_means     = gs.means[idx].repeat(2, 1) + offset
                new_scales    = (gs.scales[idx].repeat(2, 1) - torch.log(torch.tensor(1.6, device=gs.means.device)))
                new_quats     = gs.quats[idx].repeat(2, 1)
                new_opacities = gs.opacities[idx].repeat(2, 1)
                new_sh_dc     = gs.sh_dc[idx].repeat(2, 1, 1)
                new_sh_rest   = gs.sh_rest[idx].repeat(2, 1, 1)

                # Orijinalleri sil, sonra yeni iki tanesini ekle.
                # ÖNEMLİ: clone aşamasında tensor büyümüş olabilir (1529 → 1532),
                # bu yüzden keep mask'ini mevcut tensor boyutunda oluşturuyoruz.
                # split_mask'ten gelen idx değerleri orijinal prefix'te kaldığı için
                # büyüyen tensor'da da geçerli (clone'lar sona eklenir, prefix bozulmaz).
                current_n = gs.num_points
                keep = torch.ones(current_n, dtype=torch.bool, device=gs.means.device)
                keep[idx] = False
                gs._apply_mask(keep)
                gs.append_gaussians(new_means, new_scales, new_quats,
                                    new_opacities, new_sh_dc, new_sh_rest)
                stats["split"] = n_split

        # Prune
        n_pruned = gs.prune_gaussians(self.min_opacity, self.max_scale)
        stats["pruned"] = n_pruned
        stats["after"] = gs.num_points

        # Buffer'ları sıfırla
        self.reset(gs.num_points, gs.means.device)
        return stats

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
        fourier_K: int = 8,
    ):
        """
        Args:
            init_points: (N, 3) — COLMAP sparse point cloud
            init_colors: (N, 3) — [0, 1] başlangıç renkleri (None → gri)
            sh_degree:   Spherical Harmonics derecesi (0–3)
            fourier_K:   Per-gaussian Fourier trajectory frekans sayısı (0 = kapalı).
                         Default 8 → K frekans × 2 (sin/cos) × 3 axis = 48 float/gaussian.
                         4DGS paper (Yang et al 2024) SOTA mimari.
        """
        super().__init__()
        N = len(init_points)
        if N == 0:
            raise ValueError("init_points boş — COLMAP rekonstrüksiyonu başarısız mı?")

        self.sh_degree = sh_degree
        # Toplam SH katsayı sayısı (deg+1)²; rest = total - 1
        self.num_sh_rest = (sh_degree + 1) ** 2 - 1
        self.fourier_K = int(fourier_K)

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

        # ----- Per-gaussian Fourier trajectory (v3.6 / Yol C) -----
        # Δpos(gaussian_i, t) = Σ_{k=1}^{K} [A_{ik} sin(2πkt) + B_{ik} cos(2πkt)]
        # Shape: (N, K, 2, 3) — son axes: [sin_coef, cos_coef] × [x, y, z]
        # Zero-init → başlangıçta motion yok, MLP'ninki gibi identity davranış.
        if self.fourier_K > 0:
            self.fourier_pos_coeffs = nn.Parameter(
                torch.zeros(N, self.fourier_K, 2, 3)
            )
        else:
            # Dummy placeholder — kapalı mod için (backward compat)
            self.register_parameter("fourier_pos_coeffs", None)

        # ----- Phase 2.1 — Static/Dynamic explicit split -----
        # is_static=True olan Gaussian'lar deformation BYPASS edilir
        # (renderda d_means=means, d_quats=quats, d_scales=get_scales).
        # Init: hepsi static (mask-based selection sonra promote eder).
        # register_buffer: nn.Module persistent ama gradient yok.
        self.register_buffer("is_static", torch.ones(N, dtype=torch.bool))

        # ----- Phase 2.4 — Background flag (distance-based split) -----
        # is_background=True olan Gaussian'lar uzakta sayilir; deformation kapali,
        # densify pruning farkli (more permissive). flag_background_by_distance()
        # ile init sonrasi set edilir. Default: hepsi foreground (False).
        self.register_buffer("is_background", torch.zeros(N, dtype=torch.bool))

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
    # Phase 2.1 — Static/Dynamic API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def promote_dynamic(self, dynamic_mask: torch.Tensor) -> int:
        """Verilen bool mask'i (N,) True olan Gaussian'lari dynamic flag'le.

        Returns:
            promoted: dynamic'e gecirilen Gaussian sayisi
        """
        if not hasattr(self, "is_static"):
            return 0
        if dynamic_mask.shape[0] != self.num_points:
            raise ValueError(
                f"dynamic_mask boyutu ({dynamic_mask.shape[0]}) num_points ({self.num_points}) ile uyumsuz"
            )
        prev_static = self.is_static.clone()
        self.is_static = self.is_static & ~dynamic_mask.to(self.is_static.device)
        promoted = int((prev_static & ~self.is_static).sum().item())
        return promoted

    @property
    def num_static(self) -> int:
        return int(self.is_static.sum().item()) if hasattr(self, "is_static") else self.num_points

    @property
    def num_dynamic(self) -> int:
        return self.num_points - self.num_static

    # ------------------------------------------------------------------
    # Phase 2.4 — Background API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def flag_background_by_distance(
        self, scene_center: torch.Tensor, scene_extent: float, ratio: float = 2.0
    ) -> int:
        """Scene merkezinden 'ratio * scene_extent' uzakta olanlari background flag'le.

        Args:
            scene_center: (3,) world-space scene center
            scene_extent: scene scale (look-points'ten tahmin)
            ratio:        threshold = ratio * scene_extent (default 2.0)

        Returns:
            n_bg: background olarak isaretlenen Gaussian sayisi
        """
        if not hasattr(self, "is_background"):
            return 0
        sc = scene_center.to(self.means.device).reshape(1, 3)
        d = (self.means.detach() - sc).norm(dim=-1)  # (N,)
        threshold = ratio * scene_extent
        bg_mask = d > threshold
        self.is_background = bg_mask.to(self.is_background.device)
        return int(bg_mask.sum().item())

    @property
    def num_background(self) -> int:
        return int(self.is_background.sum().item()) if hasattr(self, "is_background") else 0

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
        # v3.6: fourier trajectory senkronu
        if self.fourier_pos_coeffs is not None:
            self.fourier_pos_coeffs = nn.Parameter(
                self.fourier_pos_coeffs[mask].contiguous()
            )
        # Phase 2.1 — is_static buffer senkronu (prune)
        if hasattr(self, "is_static"):
            self.is_static = self.is_static[mask].contiguous()
        # Phase 2.4 — is_background buffer senkronu
        if hasattr(self, "is_background"):
            self.is_background = self.is_background[mask].contiguous()

    # ------------------------------------------------------------------
    # Optimizer-aware density ops (T4 — perf fix)
    # ------------------------------------------------------------------
    # Per-gauss parametrelerin listesi. _apply_mask_keep_optimizer ve
    # _append_keep_optimizer bu listeyi tarar. Fourier dahil edilirse runtime
    # eklenir.
    _PER_GAUSS_ATTRS = ("means", "scales", "quats", "opacities", "sh_dc", "sh_rest")

    def _per_gauss_attrs(self) -> tuple[str, ...]:
        attrs = list(self._PER_GAUSS_ATTRS)
        if self.fourier_pos_coeffs is not None:
            attrs.append("fourier_pos_coeffs")
        return tuple(attrs)

    @staticmethod
    def _swap_param_in_optimizer(optimizer, old_p: nn.Parameter, new_p: nn.Parameter) -> dict | None:
        """Pop optimizer state for `old_p`, return it; replace old_p with new_p in
        param_groups. Caller is responsible for re-attaching the (possibly mutated)
        state under `optimizer.state[new_p]` after mutating buffer shapes."""
        state = optimizer.state.pop(old_p, None)
        for group in optimizer.param_groups:
            for i, p in enumerate(group["params"]):
                if p is old_p:
                    group["params"][i] = new_p
        return state

    @torch.no_grad()
    def _apply_mask_keep_optimizer(self, mask: torch.Tensor, optimizer) -> None:
        """Per-gauss param'ları mask ile filtrele AND optimizer Adam state'ini
        ayni mask ile filtrele.

        INRIA 3DGS reference impl pattern: density step'lerde Adam'ın
        exp_avg / exp_avg_sq buffer'ları parametre boyutuyla in-sync kalmali
        ki momentum kaybolmasin. Onceki kod _build_optimizer() ile her density
        step'te optimizer'i sifirdan kuruyordu — Adam momentum 200+ kez
        sifirlaniyor → convergence ciddi yavasliyordu.
        """
        for attr in self._per_gauss_attrs():
            old_p = getattr(self, attr)
            new_data = old_p.data[mask].contiguous()
            new_p = nn.Parameter(new_data)
            state = self._swap_param_in_optimizer(optimizer, old_p, new_p)
            if state is not None:
                if "exp_avg" in state:
                    state["exp_avg"] = state["exp_avg"][mask].contiguous()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"] = state["exp_avg_sq"][mask].contiguous()
                optimizer.state[new_p] = state
            setattr(self, attr, new_p)

        # Buffer'lar (gradient yok ama N ile sync olmali)
        if hasattr(self, "is_static"):
            self.is_static = self.is_static[mask].contiguous()
        if hasattr(self, "is_background"):
            self.is_background = self.is_background[mask].contiguous()

    @torch.no_grad()
    def _append_keep_optimizer(
        self,
        optimizer,
        new_means: torch.Tensor,
        new_scales: torch.Tensor,
        new_quats: torch.Tensor,
        new_opacities: torch.Tensor,
        new_sh_dc: torch.Tensor,
        new_sh_rest: torch.Tensor,
        new_fourier_coeffs: torch.Tensor | None = None,
    ) -> None:
        """Yeni gauss'ları concat AND optimizer Adam state'i icin sifir-padded uzat."""
        device = self.means.device
        new_data_map = {
            "means": new_means,
            "scales": new_scales,
            "quats": new_quats,
            "opacities": new_opacities,
            "sh_dc": new_sh_dc,
            "sh_rest": new_sh_rest,
        }
        if self.fourier_pos_coeffs is not None:
            if new_fourier_coeffs is None:
                n_new = new_means.shape[0]
                new_fourier_coeffs = torch.zeros(
                    n_new, self.fourier_K, 2, 3, device=device
                )
            new_data_map["fourier_pos_coeffs"] = new_fourier_coeffs

        for attr in self._per_gauss_attrs():
            old_p = getattr(self, attr)
            new_chunk = new_data_map[attr].to(device)
            new_data = torch.cat([old_p.data, new_chunk], dim=0).contiguous()
            new_p = nn.Parameter(new_data)
            state = self._swap_param_in_optimizer(optimizer, old_p, new_p)
            if state is not None:
                n_new = new_chunk.shape[0]
                if "exp_avg" in state:
                    ea = state["exp_avg"]
                    pad = torch.zeros(
                        (n_new,) + ea.shape[1:], device=ea.device, dtype=ea.dtype
                    )
                    state["exp_avg"] = torch.cat([ea, pad], dim=0).contiguous()
                if "exp_avg_sq" in state:
                    es = state["exp_avg_sq"]
                    pad = torch.zeros(
                        (n_new,) + es.shape[1:], device=es.device, dtype=es.dtype
                    )
                    state["exp_avg_sq"] = torch.cat([es, pad], dim=0).contiguous()
                optimizer.state[new_p] = state
            setattr(self, attr, new_p)

        # Buffer'lar
        n_new = new_means.shape[0]
        if hasattr(self, "is_static"):
            new_static = torch.ones(n_new, dtype=torch.bool, device=device)
            self.is_static = torch.cat([self.is_static, new_static], dim=0).contiguous()
        if hasattr(self, "is_background"):
            new_bg = torch.zeros(n_new, dtype=torch.bool, device=device)
            self.is_background = torch.cat([self.is_background, new_bg], dim=0).contiguous()

    @torch.no_grad()
    def prune_gaussians_keep_optimizer(
        self, optimizer, min_opacity: float = 0.005, max_scale: float = 0.1
    ) -> int:
        """Optimizer-aware prune. Bkz. _apply_mask_keep_optimizer."""
        keep = (
            (self.get_opacities.squeeze(-1) > min_opacity) &
            (self.get_scales.max(dim=-1).values < max_scale)
        )
        n_pruned = int((~keep).sum().item())
        if n_pruned > 0:
            self._apply_mask_keep_optimizer(keep, optimizer)
        return n_pruned

    @torch.no_grad()
    def append_gaussians(
        self,
        new_means: torch.Tensor,
        new_scales: torch.Tensor,
        new_quats: torch.Tensor,
        new_opacities: torch.Tensor,
        new_sh_dc: torch.Tensor,
        new_sh_rest: torch.Tensor,
        new_fourier_coeffs: torch.Tensor | None = None,
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
        # v3.6: fourier trajectory senkronu
        if self.fourier_pos_coeffs is not None:
            n_new = new_means.shape[0]
            if new_fourier_coeffs is None:
                # Yeni gaussian için trajectory zero-init (motion yok, öğrensin)
                new_fourier_coeffs = torch.zeros(
                    n_new, self.fourier_K, 2, 3, device=device
                )
            self.fourier_pos_coeffs = nn.Parameter(
                cat(self.fourier_pos_coeffs, new_fourier_coeffs)
            )
        # Phase 2.1 — is_static senkronu (split/clone yeni Gaussian'lar parent'tan inherit)
        if hasattr(self, "is_static"):
            n_new = new_means.shape[0]
            # Default: yeni Gaussian static (caller motion mask'ten promote etmeli)
            new_static = torch.ones(n_new, dtype=torch.bool, device=device)
            self.is_static = torch.cat([self.is_static, new_static], dim=0).contiguous()
        # Phase 2.4 — is_background senkronu (yeni gauss default foreground)
        if hasattr(self, "is_background"):
            n_new = new_means.shape[0]
            new_bg = torch.zeros(n_new, dtype=torch.bool, device=device)
            self.is_background = torch.cat([self.is_background, new_bg], dim=0).contiguous()

    # ------------------------------------------------------------------
    # Edit / refit helpers (Task C2 — object deletion)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def filter_in_place(self, keep: torch.Tensor) -> None:
        """Drop Gaussians where *keep* is False. Modifies all parameter tensors.

        Wraps the existing ``_apply_mask`` helper which already handles the
        full attribute list (``means``, ``scales``, ``quats``, ``opacities``,
        ``sh_dc``, ``sh_rest``, ``fourier_pos_coeffs``, ``is_static``,
        ``is_background``).

        After this call the optimizer state for these params is stale — the
        caller is responsible for rebuilding the optimizer.  The edit-refit
        flow always rebuilds via ``Trainer4DGS.__init__``, so this is fine.
        """
        assert keep.dtype == torch.bool, "keep must be a bool tensor"
        assert keep.shape[0] == self.num_points, (
            f"keep length {keep.shape[0]} != num_points {self.num_points}"
        )
        if (~keep).any():
            self._apply_mask(keep)

    def set_freeze_mask(self, freeze_mask: torch.Tensor) -> None:
        """Mark Gaussians as frozen (no gradient updates this iter).

        Used by edit-refit to restrict gradient updates to the affected zone.
        Stored as a plain attribute (not register_buffer) since GaussianModel
        is an nn.Module but the mask is ephemeral and should not be
        checkpointed.  Trainer4DGS reads ``gs._freeze_mask`` just before
        ``optimizer.step()`` and zeros the gradients for frozen Gaussians.
        """
        assert freeze_mask.dtype == torch.bool, "freeze_mask must be a bool tensor"
        assert freeze_mask.shape[0] == self.num_points, (
            f"freeze_mask length {freeze_mask.shape[0]} != num_points {self.num_points}"
        )
        self._freeze_mask: torch.Tensor = freeze_mask

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def state_for_save(self) -> dict:
        """Checkpoint için (CPU) sözlük."""
        return {k: v.detach().cpu() for k, v in self.state_dict().items()} | {
            "sh_degree": self.sh_degree,
            "num_points": self.num_points,
            "fourier_K": self.fourier_K,
        }

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "GaussianModel":
        """Boş bir model oluştur ve ağırlıkları yükle."""
        N = int(ckpt["num_points"])
        sh_deg = int(ckpt.get("sh_degree", 3))
        fourier_K = int(ckpt.get("fourier_K", 0))  # Eski ckpt'lerde yok → 0
        # Dummy noktalar — sonra state_dict ile üzerine yazılacak
        dummy = torch.zeros(N, 3)
        m = cls(dummy, sh_degree=sh_deg, fourier_K=fourier_K)
        # state_dict yalnızca tensor key'leri içersin
        sd = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        # Backward compat: eski ckpt'lerde fourier_pos_coeffs yok
        m.load_state_dict(sd, strict=False)
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

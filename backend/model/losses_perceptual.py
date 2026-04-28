"""Phase 1.6 — LPIPS perceptual loss wrapper.

Lazy-loaded lpips module. `pip install lpips` gerekli; yoksa loss 0 doner
(graceful fallback, training crash etmez).

Kullanim:
    from .losses_perceptual import LPIPSLoss
    lpips_fn = LPIPSLoss(net="alex")  # ilk forward'da yuklenir
    loss = lpips_fn(rendered_rgb, gt_rgb)  # rendered/gt: [3, H, W] in [0, 1]
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss with lazy import + graceful fallback.

    Args:
        net: "alex" (hizli, ~24 MB) | "vgg" (kaliteli, ~110 MB) | "squeeze"
        device: forward'da otomatik tespit (input device).
    """

    def __init__(self, net: str = "alex"):
        super().__init__()
        self.net = net
        self._model: Optional[nn.Module] = None
        self._available: Optional[bool] = None  # None = belirsiz, True/False = test edildi

    def _try_load(self, device: torch.device) -> bool:
        if self._available is False:
            return False
        if self._model is not None:
            return True
        try:
            import lpips  # noqa
        except ImportError:
            print(f"⚠ LPIPS paketi yok (`pip install lpips`). Loss 0 dondurulecek.")
            self._available = False
            return False
        try:
            self._model = lpips.LPIPS(net=self.net, verbose=False).to(device)
            for p in self._model.parameters():
                p.requires_grad_(False)
            self._available = True
            print(f"✓ LPIPS yuklendi (net={self.net}, device={device})")
            return True
        except Exception as e:
            print(f"⚠ LPIPS yukleme hatasi: {e}")
            self._available = False
            return False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """LPIPS distance(pred, target).

        Args:
            pred:   [3, H, W] or [B, 3, H, W] in [0, 1]
            target: ayni shape, [0, 1]

        Returns:
            scalar tensor (0 if LPIPS unavailable)
        """
        if not self._try_load(pred.device):
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        # lpips expects [B, 3, H, W] in [-1, 1]
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_n = pred.clamp(0, 1) * 2.0 - 1.0
        target_n = target.clamp(0, 1) * 2.0 - 1.0
        # Eval mode (BN/dropout disabled), but allow gradient w.r.t. input
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            d = self._model(pred_n.float(), target_n.float())
        return d.mean()

"""Cheap baselines: identity, unsharp mask, and a small residual U-Net.

If the diffusion restorer cannot beat the small U-Net trained on the same
pairs, its pretrained image prior is not earning its cost.
"""
from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PairDataset, collect_items, split_holdout
from .workspace import Workspace, atomic_copy, atomic_write_json


def unsharp(img: torch.Tensor, amount: float = 0.6, sigma: float = 1.2) -> torch.Tensor:
    """img: (B, 3, H, W) in [0, 1]."""
    k = int(2 * math.ceil(3 * sigma) + 1)
    x = torch.arange(k, device=img.device, dtype=img.dtype) - k // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    blur = F.conv2d(F.pad(img, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    blur = F.conv2d(blur, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
    return (img + amount * (img - blur)).clamp(0, 1)


class _Block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.GELU(), nn.Conv2d(c_out, c_out, 3, padding=1), nn.GELU()
        )
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class SmallUNet(nn.Module):
    """~8M-parameter residual U-Net; optional reference image as extra input."""

    def __init__(self, use_ref: bool = True, width: int = 48):
        super().__init__()
        c_in = 6 if use_ref else 3
        w = width
        self.use_ref = use_ref
        self.inc = _Block(c_in, w)
        self.d1 = _Block(w, 2 * w)
        self.d2 = _Block(2 * w, 4 * w)
        self.d3 = _Block(4 * w, 8 * w)
        self.u2 = _Block(8 * w + 4 * w, 4 * w)
        self.u1 = _Block(4 * w + 2 * w, 2 * w)
        self.u0 = _Block(2 * w + w, w)
        self.out = nn.Conv2d(w, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, deg, ref=None):
        x = torch.cat([deg, ref], 1) if self.use_ref else deg
        e0 = self.inc(x)
        e1 = self.d1(F.avg_pool2d(e0, 2))
        e2 = self.d2(F.avg_pool2d(e1, 2))
        e3 = self.d3(F.avg_pool2d(e2, 2))
        up = lambda t, ref_t: F.interpolate(t, size=ref_t.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.u2(torch.cat([up(e3, e2), e2], 1))
        d1 = self.u1(torch.cat([up(d2, e1), e1], 1))
        d0 = self.u0(torch.cat([up(d1, e0), e0], 1))
        return (deg + self.out(d0)).clamp(0, 1)

    @torch.no_grad()
    def restore(self, deg, ref=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.forward(deg, ref).float().clamp(0, 1)


@dataclass
class BaselineConfig:
    run_name: str
    scenes: list[str]
    use_ref: bool = True
    steps: int = 20_000
    batch_size: int = 16
    lr: float = 3e-4
    crop: int = 256
    lambda_lpips: float = 0.2
    num_workers: int = 8
    seed: int = 0


def train_baseline(ws: Workspace, cfg: BaselineConfig, pairs_root: Path, device="cuda", log=print) -> Path:
    import lpips

    run_dir = ws.runs_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "model_final.pt"
    if final_path.exists():
        log(f"[{cfg.run_name}] already finished -> {final_path}")
        return final_path
    atomic_write_json(run_dir / "config.json", asdict(cfg))
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    items = collect_items(pairs_root, cfg.scenes)
    train_items, _ = split_holdout(items, frac=0.02, seed=cfg.seed)
    loader = torch.utils.data.DataLoader(
        PairDataset(train_items, train=True, crop=cfg.crop), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True, pin_memory=True, persistent_workers=cfg.num_workers > 0,
    )
    net = SmallUNet(use_ref=cfg.use_ref).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    net_lpips = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    log(f"[{cfg.run_name}] {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M params, {len(train_items)} pairs")

    log_f = open(run_dir / "log.csv", "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(["step", "loss", "sec_per_step"])
    step, t_last = 0, time.time()
    while step < cfg.steps:
        for batch in loader:
            deg, gt, ref = (batch[k].to(device, non_blocking=True) for k in ("deg", "gt", "ref"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(deg, ref)
                loss = (out.float() - gt).abs().mean()
                loss = loss + cfg.lambda_lpips * net_lpips(out.float() * 2 - 1, gt * 2 - 1).float().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                dt = (time.time() - t_last) / 200
                t_last = time.time()
                writer.writerow([step, f"{loss.item():.5f}", f"{dt:.3f}"])
                log_f.flush()
                if step % 2000 == 0:
                    log(f"[{cfg.run_name}] step {step}/{cfg.steps} loss {loss.item():.4f}")
            if step >= cfg.steps:
                break
    log_f.close()
    tmp = ws.local / f"{cfg.run_name}_final.pt"
    torch.save({"state_dict": net.state_dict(), "use_ref": cfg.use_ref}, tmp)
    atomic_copy(tmp, final_path)
    atomic_write_json(run_dir / "DONE.json", {"steps": step})
    log(f"[{cfg.run_name}] finished -> {final_path}")
    return final_path


def load_baseline(path: Path, device="cuda") -> SmallUNet:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    net = SmallUNet(use_ref=sd["use_ref"]).to(device)
    net.load_state_dict(sd["state_dict"])
    return net.eval()

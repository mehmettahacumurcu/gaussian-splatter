"""Train the diffusion restorer on extracted pairs. Resumable from Drive."""
from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import PairDataset, collect_items, split_holdout
from .workspace import Workspace, atomic_copy, atomic_write_json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRAM_LAYERS = {3: 1.0 / 2.6, 8: 1.0 / 4.8, 15: 1.0 / 3.7, 22: 1.0 / 5.6, 29: 10.0 / 1.5}


@dataclass
class RestorerConfig:
    run_name: str
    scenes: list[str]
    use_ref: bool = True
    steps: int = 10_000
    batch_size: int = 4
    lr: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 1e-2
    lambda_l2: float = 1.0
    lambda_lpips: float = 1.0
    lambda_gram: float = 1.0
    gram_warmup_steps: int = 2000
    crop: int = 512
    timestep: int = 199
    lora_rank_vae: int = 4
    grad_checkpointing: bool = False
    num_workers: int = 8
    log_every: int = 50
    sample_every: int = 500
    local_ckpt_every: int = 500
    drive_ckpt_every: int = 2000
    val_items: int = 64
    seed: int = 0
    recipes: list[str] = field(default_factory=list)


def gram_loss(pred: torch.Tensor, target: torch.Tensor, vgg: torch.nn.Module) -> torch.Tensor:
    """Difix-style style loss on VGG16 feature Gram matrices (inputs in [0, 1])."""
    mean = torch.tensor(IMAGENET_MEAN, device=pred.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=pred.device).view(1, 3, 1, 1)
    x, y = (pred - mean) / std, (target - mean) / std
    total = pred.new_zeros(())
    for i, layer in enumerate(vgg):
        x, y = layer(x), layer(y)
        if i in GRAM_LAYERS:
            b, d, h, w = x.shape
            fx, fy = x.reshape(b, d, h * w).float(), y.reshape(b, d, h * w).float()
            gx = fx @ fx.transpose(1, 2)
            gy = fy @ fy.transpose(1, 2)
            total = total + GRAM_LAYERS[i] * ((gx - gy) ** 2).mean() / (d * h * w)
        if i >= max(GRAM_LAYERS):
            break
    return total


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def _to_views(batch, use_ref: bool, device) -> tuple[torch.Tensor, torch.Tensor]:
    deg, gt, ref = (batch[k].to(device, non_blocking=True) for k in ("deg", "gt", "ref"))
    if use_ref:
        x = torch.stack([deg, ref], dim=1)
        y = torch.stack([gt, ref], dim=1)
    else:
        x, y = deg[:, None], gt[:, None]
    return x * 2 - 1, y * 2 - 1


def _save_grid(path: Path, rows: list[list[torch.Tensor]]) -> None:
    from PIL import Image

    lines = [torch.cat([t.clamp(0, 1) for t in row], dim=2) for row in rows]
    grid = torch.cat(lines, dim=1)
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=90)


@torch.no_grad()
def validate(model, val_ds, use_ref: bool, net_lpips, device, max_items: int) -> dict:
    model.set_eval()
    l2s, lps, psnrs, base_psnrs = [], [], [], []
    for i in range(min(len(val_ds), max_items)):
        s = val_ds[i]
        deg, gt, ref = (s[k][None].to(device) for k in ("deg", "gt", "ref"))
        out = model.restore(deg, ref if use_ref else None)
        mse = F.mse_loss(out, gt).item()
        l2s.append(mse)
        psnrs.append(10 * math.log10(1 / max(mse, 1e-12)))
        base_psnrs.append(10 * math.log10(1 / max(F.mse_loss(deg, gt).item(), 1e-12)))
        lps.append(net_lpips(out * 2 - 1, gt * 2 - 1).mean().item())
    model.set_train()
    return {
        "val_psnr": float(np.mean(psnrs)),
        "val_psnr_input": float(np.mean(base_psnrs)),
        "val_lpips": float(np.mean(lps)),
        "val_l2": float(np.mean(l2s)),
    }


def train_restorer(ws: Workspace, cfg: RestorerConfig, pairs_root: Path, device: str = "cuda", log=print) -> Path:
    import lpips
    import torchvision

    from .restorer_model import Restorer

    run_dir = ws.runs_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "model_final.pt"
    if final_path.exists():
        log(f"[{cfg.run_name}] already finished -> {final_path}")
        return final_path
    local_dir = ws.local / "runs" / cfg.run_name
    local_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "config.json", asdict(cfg))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    items = collect_items(pairs_root, cfg.scenes, cfg.recipes or None)
    train_items, val_items = split_holdout(items, frac=0.02, seed=cfg.seed)
    log(f"[{cfg.run_name}] {len(cfg.scenes)} scenes, {len(train_items)} train pairs, {len(val_items)} val pairs, ref={cfg.use_ref}")
    train_ds = PairDataset(train_items, train=True, crop=cfg.crop)
    val_ds = PairDataset(val_items, train=False)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=cfg.num_workers > 0,
    )

    model = Restorer(timestep=cfg.timestep, lora_rank_vae=cfg.lora_rank_vae, device=device)
    model.set_train()
    if cfg.grad_checkpointing:
        model.unet.enable_gradient_checkpointing()
    params = list(model.unet.parameters()) + model.vae_trainable_parameters()
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(cfg.warmup_steps, 1)))

    net_lpips = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:30].to(device).eval().requires_grad_(False)

    # Resume: prefer the newer of local / Drive checkpoints.
    step = 0
    candidates = [p for p in (local_dir / "ckpt.pt", run_dir / "ckpt.pt") if p.exists()]
    if candidates:
        ck_path = max(candidates, key=lambda p: torch.load(p, map_location="cpu", weights_only=False, mmap=True)["step"])
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = int(ck["step"])
        log(f"[{cfg.run_name}] resumed from {ck_path} at step {step}")
        del ck

    log_path = run_dir / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    writer = csv.writer(log_f)
    if new_log:
        writer.writerow(["step", "loss", "l2", "lpips", "gram", "lr", "sec_per_step", "val_psnr", "val_psnr_input", "val_lpips"])

    def save_ckpt(to_drive: bool):
        state = {"model": model.trainable_state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}
        tmp = local_dir / "ckpt.pt.partial"
        torch.save(state, tmp)
        tmp.replace(local_dir / "ckpt.pt")
        if to_drive:
            atomic_copy(local_dir / "ckpt.pt", run_dir / "ckpt.pt")

    it = _infinite(loader)
    t_last = time.time()
    while step < cfg.steps:
        batch = next(it)
        x, y = _to_views(batch, cfg.use_ref, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x)
        pred_f = pred.float().flatten(0, 1)
        y_f = y.flatten(0, 1)
        l2 = F.mse_loss(pred_f, y_f) * cfg.lambda_l2
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = net_lpips(pred_f, y_f).float().mean() * cfg.lambda_lpips
        loss = l2 + lp
        gram = torch.zeros((), device=device)
        if cfg.lambda_gram > 0 and step >= cfg.gram_warmup_steps:
            h, w = pred_f.shape[-2:]
            c = min(400, h, w)
            top, left = random.randint(0, h - c), random.randint(0, w - c)
            p_c = pred_f[..., top : top + c, left : left + c] * 0.5 + 0.5
            y_c = y_f[..., top : top + c, left : left + c] * 0.5 + 0.5
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gram = gram_loss(p_c, y_c, vgg) * cfg.lambda_gram
            loss = loss + gram
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1

        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"[{cfg.run_name}] non-finite loss at step {step}")

        val = {}
        if step % cfg.sample_every == 0 or step == cfg.steps:
            val = validate(model, val_ds, cfg.use_ref, net_lpips, device, cfg.val_items)
            rows = []
            for i in range(min(4, len(val_ds))):
                s = val_ds[i]
                deg, gt, ref = (s[k][None].to(device) for k in ("deg", "gt", "ref"))
                model.set_eval()
                out = model.restore(deg, ref if cfg.use_ref else None)
                model.set_train()
                rows.append([deg[0], out[0], gt[0], ref[0]])
            _save_grid(run_dir / "samples" / f"step_{step:06d}.jpg", rows)
            log(
                f"[{cfg.run_name}] step {step}: val PSNR {val['val_psnr']:.2f} "
                f"(input {val['val_psnr_input']:.2f}), LPIPS {val['val_lpips']:.4f}"
            )
        if step % cfg.log_every == 0:
            dt = (time.time() - t_last) / cfg.log_every
            t_last = time.time()
            writer.writerow([
                step, f"{loss.item():.5f}", f"{l2.item():.5f}", f"{lp.item():.5f}", f"{gram.item():.5f}",
                f"{sched.get_last_lr()[0]:.2e}", f"{dt:.3f}",
                val.get("val_psnr", ""), val.get("val_psnr_input", ""), val.get("val_lpips", ""),
            ])
            log_f.flush()
            if step % (cfg.log_every * 10) == 0:
                log(f"[{cfg.run_name}] step {step}/{cfg.steps} loss {loss.item():.4f} ({dt:.2f}s/step)")
        if step % cfg.local_ckpt_every == 0 and step < cfg.steps:
            save_ckpt(to_drive=step % cfg.drive_ckpt_every == 0)

    log_f.close()
    final = model.trainable_state_dict()
    final["unet"] = {k: v.half() for k, v in final["unet"].items()}
    final["use_ref"] = cfg.use_ref
    final["lora_rank_vae"] = cfg.lora_rank_vae
    tmp = local_dir / "model_final.pt"
    torch.save(final, tmp)
    atomic_copy(tmp, final_path)
    # The resume checkpoint is large; drop it once the final weights are safe.
    for p in (run_dir / "ckpt.pt", local_dir / "ckpt.pt"):
        p.unlink(missing_ok=True)
    atomic_write_json(run_dir / "DONE.json", {"steps": step, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    log(f"[{cfg.run_name}] finished -> {final_path}")
    return final_path


def load_restorer(path: Path, device: str = "cuda"):
    from .restorer_model import Restorer

    sd = torch.load(path, map_location="cpu", weights_only=False)
    model = Restorer(timestep=int(sd.get("timestep", 199)), lora_rank_vae=int(sd.get("lora_rank_vae", 4)), device=device)
    sd["unet"] = {k: v.float() for k, v in sd["unet"].items()}
    model.load_trainable_state_dict(sd)
    model.set_eval()
    return model, bool(sd.get("use_ref", False))

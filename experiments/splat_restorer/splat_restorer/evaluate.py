"""Image-level evaluation on held-out *test scenes* (never seen in training).

Metrics: PSNR, SSIM, LPIPS-Alex (training used LPIPS-VGG, so the evaluation
metric is not the one the models optimised). A method only "works" if it
improves LPIPS *without* lowering PSNR much: diffusion models can hallucinate
plausible texture that looks sharper yet is wrong.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import PairDataset, collect_items
from .gs_trainer import ssim
from .workspace import Workspace


def _metrics(out: torch.Tensor, gt: torch.Tensor, net_lpips) -> dict:
    mse = F.mse_loss(out, gt).item()
    return {
        "psnr": 10 * math.log10(1 / max(mse, 1e-12)),
        "ssim": ssim(out[0].permute(1, 2, 0), gt[0].permute(1, 2, 0)).item(),
        "lpips": net_lpips(out * 2 - 1, gt * 2 - 1).mean().item(),
    }


def build_methods(ws: Workspace, restorer_runs: list[str], baseline_runs: list[str], device="cuda") -> dict:
    """name -> zero-arg loader returning callable(deg, ref) -> out (all [0, 1], Bx3xHxW)."""
    from .baseline import load_baseline, unsharp

    methods = {
        "input": lambda: (lambda d, r: d),
        "unsharp": lambda: (lambda d, r: unsharp(d)),
    }
    for run in baseline_runs:
        path = ws.runs_dir / run / "model_final.pt"
        if path.exists():
            methods[run] = (lambda p=path: (lambda net: (lambda d, r: net.restore(d, r if net.use_ref else None)))(load_baseline(p, device)))
    for run in restorer_runs:
        path = ws.runs_dir / run / "model_final.pt"
        if path.exists():
            def loader(p=path):
                from .train_restorer import load_restorer

                model, use_ref = load_restorer(p, device)
                return lambda d, r: model.restore(d, r if use_ref else None)

            methods[run] = loader
    return methods


@torch.no_grad()
def evaluate(
    ws: Workspace,
    pairs_root: Path,
    test_scenes: list[str],
    methods: dict,
    tag: str,
    max_items_per_scene: int = 40,
    n_sheets: int = 8,
    device="cuda",
    log=print,
) -> Path:
    import lpips

    net_lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    items = []
    for s in test_scenes:
        its = collect_items(pairs_root, [s])
        random.Random(0).shuffle(its)
        items += its[:max_items_per_scene]
    ds = PairDataset(items, train=False)
    log(f"[eval {tag}] {len(items)} pairs from {len(test_scenes)} test scenes, methods: {list(methods)}")

    out_dir = ws.results_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_idx = sorted(random.Random(1).sample(range(len(ds)), min(n_sheets, len(ds))))
    sheet_cols: dict[int, list] = defaultdict(list)
    rows = []
    for name, loader in methods.items():
        fn = loader()
        for i in range(len(ds)):
            s = ds[i]
            deg, gt, ref = (s[k][None].to(device) for k in ("deg", "gt", "ref"))
            out = fn(deg, ref).float().clamp(0, 1)
            m = _metrics(out, gt, net_lpips)
            it = items[i]
            rows.append({"method": name, "scene": it["scene"], "recipe": it["recipe"], "frame": it["frame"], **m})
            if i in sheet_idx:
                if not sheet_cols[i]:
                    sheet_cols[i].append(("ref", ref[0].cpu()))
                sheet_cols[i].append((name, out[0].cpu()))
                if name == list(methods)[-1]:
                    sheet_cols[i].append(("gt", gt[0].cpu()))
        del fn
        torch.cuda.empty_cache()
        log(f"[eval {tag}] {name} done")

    with open(out_dir / "per_item.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = _summarise(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    _contact_sheets(out_dir, sheet_cols)
    log(_markdown(summary))
    return out_dir


def _summarise(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["recipe"])].append(r)
        groups[(r["method"], "all")].append(r)
    out = {}
    for (method, recipe), rs in groups.items():
        out.setdefault(method, {})[recipe] = {
            k: float(np.mean([r[k] for r in rs])) for k in ("psnr", "ssim", "lpips")
        } | {"n": len(rs)}
    return out


def _markdown(summary: dict) -> str:
    recipes = sorted({rec for m in summary.values() for rec in m}, key=lambda r: (r != "all", r))
    base = summary.get("input", {})
    lines = []
    for rec in recipes:
        lines.append(f"\n### {rec}\n")
        lines.append("| method | PSNR ↑ | ΔPSNR | SSIM ↑ | LPIPS ↓ | ΔLPIPS |")
        lines.append("|---|---|---|---|---|---|")
        for method, by_rec in summary.items():
            if rec not in by_rec:
                continue
            m = by_rec[rec]
            b = base.get(rec, m)
            lines.append(
                f"| {method} | {m['psnr']:.2f} | {m['psnr'] - b['psnr']:+.2f} | {m['ssim']:.4f} | "
                f"{m['lpips']:.4f} | {m['lpips'] - b['lpips']:+.4f} |"
            )
    return "\n".join(lines) + "\n"


def _contact_sheets(out_dir: Path, sheet_cols: dict) -> None:
    from PIL import Image, ImageDraw

    for i, cols in sheet_cols.items():
        tiles = []
        for name, t in cols:
            arr = (t.permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
            img = Image.fromarray(arr)
            ImageDraw.Draw(img).text((8, 8), name, fill=(255, 255, 0))
            tiles.append(img)
        w, h = tiles[0].size
        cols_n = 3
        rows_n = math.ceil(len(tiles) / cols_n)
        sheet = Image.new("RGB", (w * cols_n, h * rows_n))
        for k, tile in enumerate(tiles):
            sheet.paste(tile, ((k % cols_n) * w, (k // cols_n) * h))
        sheet.save(out_dir / f"sheet_{i:04d}.jpg", quality=88)

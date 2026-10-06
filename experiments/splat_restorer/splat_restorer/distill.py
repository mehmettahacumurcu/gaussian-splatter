"""Distill-back test: does restoring novel views actually improve the *splat*?

For each test scene and recipe, starting from the same weak splat:

* control: fine-tune on the real training photos only;
* <method>: fine-tune on the training photos plus restored renders of the
  pair-frame poses (pseudo ground truth, Difix3D-style).

Both are scored on the scene's eval frames, which no splat ever trained on.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .gs_trainer import TrainConfig, View, params_from_state, render, ssim, train_gaussians
from .splits import subsample
from .workspace import Workspace


def _load_photo(path: Path, device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0).to(device)


def pad_to_multiple(x: torch.Tensor, m: int = 64) -> tuple[torch.Tensor, tuple[int, int]]:
    """x: (B, 3, H, W). Reflect-pad bottom/right to multiples of m."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    return F.pad(x, (0, pw, 0, ph), mode="reflect"), (h, w)


def restore_full(fn, deg_hwc: torch.Tensor, ref_hwc: torch.Tensor) -> torch.Tensor:
    deg = deg_hwc.permute(2, 0, 1)[None]
    ref = ref_hwc.permute(2, 0, 1)[None]
    deg_p, (h, w) = pad_to_multiple(deg)
    ref_p, _ = pad_to_multiple(ref)
    out = fn(deg_p, ref_p).float()[..., :h, :w]
    return out[0].permute(1, 2, 0).clamp(0, 1)


@torch.no_grad()
def _score(params, frames, photos, Ks, w2c, width, height, net_lpips) -> dict:
    ps, ss, ls = [], [], []
    for i in frames:
        img, _ = render(params, w2c[i], Ks[i], width, height, 3)
        gt = photos[i]
        ps.append(10 * math.log10(1 / max(F.mse_loss(img, gt).item(), 1e-12)))
        ss.append(ssim(img, gt).item())
        ls.append(net_lpips(img.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).item())
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "lpips": float(np.mean(ls))}


def distill_scene(
    ws: Workspace,
    pairs_root: Path,
    scene: str,
    recipe: str,
    methods: dict,
    finetune_steps: int = 1500,
    max_pseudo: int = 60,
    pseudo_share: float = 0.5,
    device="cuda",
    net_lpips=None,
    log=print,
) -> dict:
    index = json.loads((pairs_root / scene / "index.json").read_text(encoding="utf-8"))
    rec = index["recipes"][recipe]
    width, height = index["width"], index["height"]
    Ks = torch.tensor(index["cameras"]["K"], device=device)
    w2c = torch.tensor(index["cameras"]["w2c"], device=device)
    root = pairs_root / scene
    eval_photos = {i: _load_photo(root / "eval" / f"{i:05d}.jpg", device) for i in rec["eval"]}
    train_photos = {i: _load_photo(root / recipe / "ref" / f"{i:05d}.jpg", device) for i in rec["train"]}
    start = torch.load(ws.splat_path(scene, recipe), map_location="cpu", weights_only=False)
    ft_cfg = TrainConfig(steps=finetune_steps, densify=False, lr_means=1.6e-5, lr_means_final_frac=0.1, seed=0)
    score = lambda p: _score(p, rec["eval"], eval_photos, Ks, w2c, width, height, net_lpips)

    results = {"start": score(params_from_state(start, device))}
    train_views = [View(train_photos[i], w2c[i], Ks[i]) for i in rec["train"]]

    params = train_gaussians(train_views, width, height, ft_cfg, params=params_from_state(start, device))
    results["control"] = score(params)
    del params

    pseudo_frames = subsample(rec["pair"], max_pseudo)
    from .splits import nearest_reference

    w2c_np = np.array(index["cameras"]["w2c"])
    centers = -np.einsum("fij,fi->fj", w2c_np[:, :3, :3], w2c_np[:, :3, 3])
    forwards = w2c_np[:, 2, :3]
    base = params_from_state(start, device)
    with torch.no_grad():
        renders = {i: render(base, w2c[i], Ks[i], width, height, 3)[0] for i in pseudo_frames}
    del base

    for name, loader in methods.items():
        fn = loader()
        with torch.no_grad():
            pseudo = {}
            for i in pseudo_frames:
                ref = train_photos[nearest_reference(i, rec["train"], centers, forwards)]
                pseudo[i] = restore_full(fn, renders[i], ref)
        del fn
        torch.cuda.empty_cache()
        # Pseudo views together get ``pseudo_share`` of the samples.
        w_pseudo = pseudo_share / (1 - pseudo_share) * len(train_views) / max(len(pseudo), 1)
        views = train_views + [View(pseudo[i], w2c[i], Ks[i], weight=w_pseudo) for i in pseudo_frames]
        params = train_gaussians(views, width, height, ft_cfg, params=params_from_state(start, device))
        results[name] = score(params)
        del params, pseudo
        torch.cuda.empty_cache()
    log(f"  {scene[:12]} {recipe}: " + ", ".join(f"{k} {v['psnr']:.2f}/{v['lpips']:.3f}" for k, v in results.items()))
    return results


def run_distill(
    ws: Workspace, pairs_root: Path, test_scenes: list[str], recipes: list[str], methods: dict, tag: str,
    finetune_steps: int = 1500, device="cuda", log=print,
) -> Path:
    import lpips

    net_lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    out_dir = ws.results_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "per_scene.json"
    all_res = json.loads(partial.read_text(encoding="utf-8")) if partial.exists() else {}
    for scene in test_scenes:
        for recipe in recipes:
            key = f"{scene}/{recipe}"
            if key in all_res:
                continue
            if not ws.splat_path(scene, recipe).exists():
                log(f"  skip {key}: no saved splat")
                continue
            all_res[key] = distill_scene(
                ws, pairs_root, scene, recipe, methods, finetune_steps=finetune_steps, device=device,
                net_lpips=net_lpips, log=log,
            )
            partial.write_text(json.dumps(all_res, indent=2), encoding="utf-8")

    lines = []
    for recipe in recipes:
        keys = [k for k in all_res if k.endswith("/" + recipe)]
        if not keys:
            continue
        names = list(all_res[keys[0]])
        lines += [f"\n### {recipe} ({len(keys)} scenes)\n", "| condition | PSNR ↑ | Δ vs control | SSIM ↑ | LPIPS ↓ | Δ vs control | wins vs control |", "|---|---|---|---|---|---|---|"]
        ctrl = {m: np.mean([all_res[k]["control"][m] for k in keys]) for m in ("psnr", "ssim", "lpips")}
        for n in names:
            mean = {m: np.mean([all_res[k][n][m] for k in keys]) for m in ("psnr", "ssim", "lpips")}
            wins = sum(all_res[k][n]["psnr"] > all_res[k]["control"]["psnr"] for k in keys)
            lines.append(
                f"| {n} | {mean['psnr']:.2f} | {mean['psnr'] - ctrl['psnr']:+.2f} | {mean['ssim']:.4f} | "
                f"{mean['lpips']:.4f} | {mean['lpips'] - ctrl['lpips']:+.4f} | {wins}/{len(keys)} |"
            )
    md = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    log(md)
    return out_dir

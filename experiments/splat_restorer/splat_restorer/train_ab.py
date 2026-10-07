"""A/B training variants of the app's static trainer on one captured scene.

Every variant trains the same Trainer4DGS setup the hybrid notebook uses
("Spirula dataset + our trainer"), on the same frames and seed, and is scored
on the same held-out photos. The trainer classes are passed in, so this module
does not depend on the app's source layout.

Variants change one thing at a time:

* ``lr_decay``: exponential decay of the Gaussian position learning rate
  (standard 3DGS; the app's trainer keeps it constant). Applied from the
  trainer's progress callback, so the trainer code is unchanged.
* ``clean``: training frames are undistorted at full resolution, area-averaged
  down (2x2 averaging halves sensor noise) and lightly denoised. Held-out
  photos are never trained on; they are scored both as captured and cleaned.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .colmap_io import read_sparse_model
from .own_splat import SplatData, _k_and_dist, _scores, _to_u8, _w2c_t, comparison_image, load_ply
from .gs_trainer import render


@dataclass
class Variant:
    name: str
    steps: int = 30000
    cap: int = 1_000_000
    lr_decay: bool = False
    lr_final_frac: float = 0.01
    clean: bool = False
    note: str = ""


VARIANTS = {
    "baseline": Variant("baseline", note="28 Eylül eğitimiyle aynı ayarlar"),
    "lr_decay": Variant("lr_decay", lr_decay=True, note="konum öğrenme oranı 100x azalır"),
    "lr_decay_clean": Variant("lr_decay_clean", lr_decay=True, clean=True, note="+ gürültüsü azaltılmış eğitim kareleri"),
    "lr_decay_60k": Variant("lr_decay_60k", steps=60000, lr_decay=True, note="+ 60k adım"),
}


# ----------------------------------------------------------------------------- clean frames


def clean_frames(source: Path, data: SplatData, out: Path, model_name: str = "0", nlm_h: float = 3.0,
                 workers: int = 8, log=print) -> Path:
    """Frames on the same cameras as ``data``, area-downsampled from full resolution and denoised."""
    import cv2

    out = Path(out)
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    model = read_sparse_model(Path(source) / "sparse" / model_name)
    cam = next(iter(model.cameras.values()))
    K, dist = _k_and_dist(cam)
    w, h = cam.width, cam.height
    sx, sy = w / data.width, h / data.height
    # Same output camera as data.K, expressed at full resolution (pixel-centre convention).
    k_full = data.K.copy()
    k_full[0, 0] *= sx; k_full[1, 1] *= sy
    k_full[0, 2] = (data.K[0, 2] + 0.5) * sx - 0.5
    k_full[1, 2] = (data.K[1, 2] + 0.5) * sy - 0.5
    mx, my = cv2.initUndistortRectifyMap(K, dist, None, k_full, (w, h), cv2.CV_32FC1)

    def convert(i):
        dest = frames / data.frames[i]
        if dest.exists():
            return
        img = cv2.imread(str(Path(source) / "images" / data.names[i]))
        full = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        small = cv2.resize(full, (data.width, data.height), interpolation=cv2.INTER_AREA)
        if nlm_h > 0:
            small = cv2.fastNlMeansDenoisingColored(small, None, nlm_h, nlm_h, 7, 21)
        tmp = dest.with_suffix(".tmp.png")
        cv2.imwrite(str(tmp), small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        tmp.replace(dest)

    todo = [i for i in range(data.num_frames) if not (frames / data.frames[i]).exists()]
    if todo:
        log(f"[clean] {len(todo)} frames: full-res undistort -> area downsample -> NLM h={nlm_h}")
        with ThreadPoolExecutor(workers) as pool:
            for n, _ in enumerate(pool.map(convert, todo), 1):
                if n % 200 == 0 or n == len(todo):
                    log(f"[clean] {n}/{len(todo)}")
    return frames


# ----------------------------------------------------------------------------- training


def init_points(source: Path, model_name: str, seed: int, max_points: int = 150_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COLMAP points, subsampled exactly like the hybrid notebook."""
    model = read_sparse_model(Path(source) / "sparse" / model_name)
    xyz, rgb = model.points_xyz, model.points_rgb
    ids = np.arange(len(xyz))
    if len(ids) > max_points:
        ids = np.sort(np.random.default_rng(seed).choice(len(xyz), max_points, replace=False))
    return xyz, rgb, ids


def train_variant(
    v: Variant,
    data: SplatData,
    frames_dir: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    init_ids: np.ndarray,
    holdout: list[int],
    work: Path,
    app,  # namespace with GaussianModel, DeformationField, Trainer4DGS, export_to_ply, scene_extent
    seed: int = 42,
    log_interval: int = 200,
    log=print,
) -> tuple[Path, dict]:
    """Train one variant with the app's trainer; returns (exported PLY, info)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gs = app.GaussianModel(torch.from_numpy(xyz[init_ids]).float(),
                           torch.from_numpy(rgb[init_ids].astype("float32") / 255), fourier_K=0)
    deform = app.DeformationField(resolution=8, feat_dim=4, mlp_width=32, mlp_depth=1, num_time_freqs=2)
    for p in deform.parameters():
        p.requires_grad_(False)
    trainer = app.Trainer4DGS(
        gs, deform, scene_extent=app.scene_extent(xyz), max_gaussians=v.cap,
        density_end_iter=min(25000, v.steps // 2), deform_pos_mode="mlp",
        lambda_deform_reg=0, lambda_smoothness=0, lambda_rigidity=0, lambda_fourier_reg=0, lambda_mask_motion=0,
    )
    lr0 = trainer.lr_specs["means"]
    curve = []

    def progress(it, total, loss, psnr, n):
        curve.append({"iter": it, "loss": loss, "train_psnr": psnr, "splats": n})
        if v.lr_decay:
            lr = lr0 * v.lr_final_frac ** (min(it, v.steps) / v.steps)
            for g in trainer.optimizer.param_groups:
                if any(p is trainer.gs.means for p in g["params"]):
                    g["lr"] = lr

    t0 = time.time()
    trainer.train(
        [frames_dir / f for f in data.frames], torch.from_numpy(data.K).float(),
        [torch.from_numpy(w).float() for w in data.w2c],
        n_iters=v.steps, image_size=(data.width, data.height), static_mode=True, preload_to_ram=True,
        log_interval=log_interval, ckpt_dir=work / "ckpt", ckpt_interval=v.steps,
        sh_progressive_schedule=True, holdout_indices=holdout, progress_callback=progress,
    )
    torch.cuda.synchronize()
    seconds = time.time() - t0
    ply = app.export_to_ply(gs, None, work / "export", num_timestamps=1)[0]
    info = {"variant": asdict(v), "train_seconds": seconds, "splats": int(gs.num_points), "curve": curve}
    del trainer, gs, deform
    torch.cuda.empty_cache()
    return Path(ply), info


# ----------------------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate_ply(ply: Path, data: SplatData, holdout: list[int], clean_dir: Path | None, net_lpips,
                 render_dir: Path, device="cuda") -> dict:
    """Score held-out frames against the captured photo and (if given) its cleaned version."""
    params, deg = load_ply(ply, device)
    w2c, K = _w2c_t(data, device)
    render_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for i in holdout:
        img, _ = render(params, w2c[i], K, data.width, data.height, deg)
        row = _scores(img, data.photo(i, device), net_lpips)
        if clean_dir is not None:
            clean = torch.from_numpy(np.asarray(Image.open(clean_dir / data.frames[i]).convert("RGB")).copy())
            row["clean"] = _scores(img, clean.to(device).float() / 255, net_lpips)
        rows[str(i)] = row
        Image.fromarray(_to_u8(img)).save(render_dir / f"{i:06d}.jpg", quality=92)
    del params
    torch.cuda.empty_cache()
    return rows


def summarise(results: dict) -> dict:
    out = {}
    for name, r in results.items():
        rows = list(r["frames"].values())
        m = {k: float(np.mean([x[k] for x in rows])) for k in ("psnr", "ssim", "lpips")}
        if rows and "clean" in rows[0]:
            m.update({f"clean_{k}": float(np.mean([x["clean"][k] for x in rows])) for k in ("psnr", "ssim", "lpips")})
        m.update(splats=r["info"]["splats"], minutes=r["info"]["train_seconds"] / 60)
        out[name] = m
    return out


def markdown(summary: dict, n_frames: int) -> str:
    base = summary.get("baseline") or next(iter(summary.values()))
    has_clean = "clean_psnr" in base
    head = "| varyant | PSNR ↑ | Δ | SSIM ↑ | LPIPS ↓ | Δ |" + (" PSNR (temiz foto) | Δ |" if has_clean else "") + " splat | dk |"
    sep = "|---" * (head.count("|") - 1) + "|"
    lines = [f"## Eğitim varyantları: {n_frames} saklanan kare (hiçbiri eğitimde kullanılmadı)", "", head, sep]
    for name, m in summary.items():
        row = (f"| {name} | {m['psnr']:.2f} | {m['psnr'] - base['psnr']:+.2f} | {m['ssim']:.4f} | "
               f"{m['lpips']:.4f} | {m['lpips'] - base['lpips']:+.4f} |")
        if has_clean:
            row += f" {m['clean_psnr']:.2f} | {m['clean_psnr'] - base['clean_psnr']:+.2f} |"
        row += f" {m['splats']:,} | {m['minutes']:.0f} |"
        lines.append(row)
    lines += ["", "Δ: ilk satıra (baseline) göre. 'Temiz foto' sütunu, sensör gürültüsü azaltılmış fotoğrafa karşı ölçer; "
              "gürültüsüz eğitilen varyant orijinal fotoğrafın gürültüsünü taklit etmediği için orada daha adil karşılaştırılır."]
    return "\n".join(lines)


def comparison_sheets(results: dict, data: SplatData, out_dir: Path, render_root: Path, n_worst: int = 4, n_even: int = 4) -> list[Path]:
    names = list(results)
    first = results[names[0]]["frames"]
    ids = sorted(int(i) for i in first)
    worst = sorted(ids, key=lambda i: first[str(i)]["psnr"])[:n_worst]
    even = [ids[k] for k in np.linspace(0, len(ids) - 1, n_even + 2).round().astype(int)[1:-1]]
    picks = list(dict.fromkeys(worst + even))
    paths = []
    for i in picks:
        tiles = {"foto": _to_u8(data.photo(i, "cpu"))}
        for n in names:
            tiles[n] = np.asarray(Image.open(render_root / n / "renders" / f"{i:06d}.jpg").convert("RGB"))
        p = out_dir / f"compare_{i:06d}.jpg"
        comparison_image(tiles, p, tile_width=560, box_from=names[0])
        paths.append(p)
    return paths


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)

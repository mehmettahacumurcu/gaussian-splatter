"""Capacity and "best combination" runs with gsplat's reference trainer.

Earlier tests changed one thing at a time on 1M splats and the original SfM and
plateaued around 27.3 dB on pose-clean frames. This module combines the winners
(re-done SfM, MCMC) and raises the splat budget, keeping the comparison fair:

* held-out photos are fixed BY FILE NAME (a re-done SfM registers a different
  set of frames, so index-based splits would compare different photos);
* every SfM is scored on the held-out names it registered, and the table's
  main columns use the names registered by ALL compared SfMs;
* training never sees a held-out name (excluded at the dataset level, the same
  wrapper the gsplat testbed uses).

Fly-through videos follow the capture's own camera path (smoothed) with a slow
sideways drift, so floaters in front of the cameras become visible; PSNR on
held-out frames barely registers them.
"""
from __future__ import annotations

import fnmatch
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .gs_trainer import render
from .gsplat_testbed import GsVariant
from .own_splat import SplatData, _label, _to_u8


@dataclass
class ComboVariant:
    name: str
    sfm: str  # key into the notebook's SFM_SOURCES ("original", "extreme", ...)
    gs: GsVariant
    steps: int = 30000
    note: str = ""


def _mcmc(cap: int, *extra: str) -> GsVariant:
    return GsVariant(f"mcmc_{cap // 1_000_000}m", preset="mcmc", args=["--strategy.cap-max", str(cap), *extra])


VARIANTS = {
    # Ordered by priority: a cut-short run still answers the main questions first.
    "ext_default": ComboVariant("ext_default", "extreme", GsVariant("default"),
                                note="yeni SfM (extreme) + gsplat varsayılanı; sadece SfM'in etkisi"),
    "ext_mcmc_3m": ComboVariant("ext_mcmc_3m", "extreme", _mcmc(3_000_000),
                                note="yeni SfM + MCMC 3M splat"),
    "orig_mcmc_3m": ComboVariant("orig_mcmc_3m", "original", _mcmc(3_000_000),
                                 note="eski SfM + MCMC 3M; kapasite etkisini SfM'den ayırır"),
    "ext_mcmc_1m": ComboVariant("ext_mcmc_1m", "extreme", _mcmc(1_000_000),
                                note="yeni SfM + MCMC 1M; 3M ile kıyas için"),
    "ext_mcmc_5m": ComboVariant("ext_mcmc_5m", "extreme", _mcmc(5_000_000),
                                note="yeni SfM + MCMC 5M splat (büyük GPU)"),
    "ext_mcmc_3m_60k": ComboVariant("ext_mcmc_3m_60k", "extreme", _mcmc(3_000_000), steps=60000,
                                    note="yeni SfM + MCMC 3M, 60k adım"),
    "ext_mcmc_3m_bilagrid": ComboVariant("ext_mcmc_3m_bilagrid", "extreme", _mcmc(3_000_000, "--use-bilateral-grid"),
                                         note="yeni SfM + MCMC 3M + kare başına pozlama grid'i (ölçüm grid'siz PLY ile)"),
}


def variant_dict(v: ComboVariant) -> dict:
    return asdict(v)


# ----------------------------------------------------------------------------- held-out photos by name


def holdout_names(names_sorted: list[str], every: int, pattern: str = "") -> list[str]:
    """Fixed held-out photo names: every N-th name of the reference SfM (sorted), plus any
    name matching ``pattern`` (glob, e.g. "test_*" for a separate test walk)."""
    chosen = set(names_sorted[::every]) if every > 0 else set()
    if pattern:
        chosen |= {n for n in names_sorted if fnmatch.fnmatch(n, pattern)}
    return sorted(chosen)


def common_names(per_sfm: dict[str, list[str]]) -> list[str]:
    sets = [set(v) for v in per_sfm.values()]
    return sorted(set.intersection(*sets)) if sets else []


# ----------------------------------------------------------------------------- reporting


def summarise(results: dict, common: list[str], bad_names: list[str]) -> dict:
    """results[name] = {"frames": {image_name: {psnr, ssim, lpips}}, "info": {...}}."""
    bad_s = set(bad_names)
    out = {}
    for name, r in results.items():
        rows = r["frames"]
        sel = [rows[n] for n in common if n in rows]
        ok = [rows[n] for n in common if n in rows and n not in bad_s]
        m = {k: float(np.mean([x[k] for x in sel])) for k in ("psnr", "ssim", "lpips")}
        m.update({f"{k}_ok": float(np.mean([x[k] for x in ok])) for k in ("psnr", "ssim", "lpips")})
        own = list(rows.values())
        info = r.get("info", {})
        m.update(n=len(sel), n_own=len(own), psnr_own=float(np.mean([x["psnr"] for x in own])),
                 splats=info.get("splats"), minutes=(info.get("train_seconds") or 0) / 60,
                 sfm=r.get("sfm", "?"), note=r.get("note", ""))
        out[name] = m
    return out


def markdown(summary: dict, n_common: int, bad_names: list[str], missing: dict[str, list[str]]) -> str:
    base = summary.get("ref_orig_default") or next(iter(summary.values()))
    lines = [
        f"## Kapasite + en iyi kombinasyon: {n_common} ortak saklanan fotoğraf",
        "",
        "Saklanan fotoğraflar **dosya adıyla** sabit; bütün SfM'lerin kaydettiği ortak isimler üzerinden ölçülür "
        "(hiçbiri hiçbir varyantta eğitimde kullanılmadı). 'Kötü kareler hariç' sütunu, eski SfM'de çöken "
        f"{len(bad_names)} saklanan kareyi çıkarır.",
        "",
        "| varyant | SfM | PSNR ↑ | Δ | PSNR (kötü hariç) | Δ | SSIM ↑ | LPIPS ↓ | Δ | LPIPS (kötü hariç) | splat | dk |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in summary.items():
        splats = f"{m['splats']:,}" if m.get("splats") else "-"
        lines.append(
            f"| {name} | {m['sfm']} | {m['psnr']:.2f} | {m['psnr'] - base['psnr']:+.2f} | {m['psnr_ok']:.2f} | "
            f"{m['psnr_ok'] - base['psnr_ok']:+.2f} | {m['ssim']:.4f} | {m['lpips']:.4f} | "
            f"{m['lpips'] - base['lpips']:+.4f} | {m['lpips_ok']:.4f} | {splats} | {m['minutes']:.0f} |")
    lines += ["", "Δ: `ref_orig_default` (eski SfM, gsplat varsayılanı, 1M) satırına göre. ~0,3 dB altındaki PSNR farkları "
              "koşudan koşuya oynayabilir; LPIPS'te ~0,01."]
    for sfm, names in missing.items():
        if names:
            lines.append(f"* `{sfm}` SfM'i {len(names)} saklanan fotoğrafı kaydedemedi (ortalamalara girmez): "
                         + ", ".join(names[:12]) + (" ..." if len(names) > 12 else ""))
    lines += ["* Asıl görsel karar için `flythrough_*.mp4` videolarına bak: floater ve titremeyi PSNR göstermez."]
    return "\n".join(lines)


def save_json(path: Path, obj) -> None:
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


# ----------------------------------------------------------------------------- fly-through


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """wxyz quaternion of a rotation matrix."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def flythrough_path(data: SplatData, names: list[str], n_frames: int = 450, smooth: int = 15,
                    drift_frac: float = 0.04) -> list[np.ndarray]:
    """Camera-to-world poses along the capture path (cameras ``names`` in order), smoothed,
    resampled to ``n_frames`` and drifted sideways by up to ``drift_frac`` of the path extent."""
    index = {n: i for i, n in enumerate(data.names)}
    ids = [index[n] for n in names if n in index]
    if len(ids) < 2:
        raise ValueError("fly-through needs at least two registered cameras")
    w2c = data.w2c[ids]
    R_c2w = np.transpose(w2c[:, :3, :3], (0, 2, 1))
    centers = -np.einsum("nij,nj->ni", R_c2w, w2c[:, :3, 3])
    quats = np.array([_rot_to_quat(R) for R in R_c2w])
    for i in range(1, len(quats)):  # keep the quaternion hemisphere continuous before averaging
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    k = max(1, smooth)
    kernel = np.ones(k) / k
    pad = lambda a: np.pad(a, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
    c_s = np.stack([np.convolve(pad(centers)[:, d], kernel, mode="valid") for d in range(3)], 1)
    q_s = np.stack([np.convolve(pad(quats)[:, d], kernel, mode="valid") for d in range(4)], 1)
    q_s /= np.linalg.norm(q_s, axis=1, keepdims=True)
    t_src = np.linspace(0, 1, len(ids))
    t = np.linspace(0, 1, n_frames)
    c_t = np.stack([np.interp(t, t_src, c_s[:, d]) for d in range(3)], 1)
    q_t = np.stack([np.interp(t, t_src, q_s[:, d]) for d in range(4)], 1)
    extent = float(np.linalg.norm(centers.max(0) - centers.min(0)))
    poses = []
    for j in range(n_frames):
        R = _quat_to_rot(q_t[j])
        right = R[:, 0]
        c = c_t[j] + right * drift_frac * extent * math.sin(2 * math.pi * 3 * t[j])
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, c
        poses.append(M)
    return poses


@torch.no_grad()
def render_flythrough(params, deg: int, data: SplatData, poses: list[np.ndarray], out_dir: Path, scale: float = 0.5,
                      rasterize_mode: str = "classic", device="cuda", **raster_kwargs) -> Path:
    """Render JPEG frames (resumable) for ``poses`` at ``scale`` of the evaluation resolution."""
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = int(round(data.width * scale / 2) * 2), int(round(data.height * scale / 2) * 2)
    K = torch.tensor(data.K, dtype=torch.float32, device=device).clone()
    K[0] *= w / data.width
    K[1] *= h / data.height
    for j, c2w in enumerate(poses):
        p = out_dir / f"{j:05d}.jpg"
        if p.exists():
            continue
        w2c = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device=device)
        img, _ = render(params, w2c, K, w, h, deg, rasterize_mode, **raster_kwargs)
        Image.fromarray(_to_u8(img)).save(p, quality=90)
    return out_dir


def write_video(frame_dirs: dict[str, Path], out: Path, fps: int = 30, cols: int = 2) -> Path:
    """MP4 of one frame folder, or a labelled grid of several (same frame count)."""
    import imageio.v2 as imageio

    names = list(frame_dirs)
    files = {n: sorted(Path(d).glob("*.jpg")) for n, d in frame_dirs.items()}
    count = min(len(f) for f in files.values())
    rows = math.ceil(len(names) / cols) if len(names) > 1 else 1
    with imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=2) as writer:
        for j in range(count):
            tiles = [_label(Image.open(files[n][j]).convert("RGB"), n) if len(names) > 1 else Image.open(files[n][j]).convert("RGB")
                     for n in names]
            if len(tiles) == 1:
                writer.append_data(np.asarray(tiles[0]))
                continue
            tw, th = tiles[0].size
            canvas = Image.new("RGB", (tw * cols, th * rows), (16, 16, 16))
            for k, tile in enumerate(tiles):
                canvas.paste(tile, ((k % cols) * tw, (k // cols) * th))
            writer.append_data(np.asarray(canvas))
    return out

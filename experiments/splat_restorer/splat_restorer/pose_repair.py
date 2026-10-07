"""Pose-repair experiment: does a better SfM fix the frames the splat cannot fit?

On IMG_5966 a few short video segments render at 11-14 dB even on frames the
splat was trained on: their camera poses are wrong, so they feed inconsistent
supervision to every trainer. This module compares SfM reconstructions of the
same images end to end:

1. ``sfm_stats``: registration, reprojection error and track statistics, and
   whether the known-bad frames were registered at all.
2. Each reconstruction trains the app's trainer (``train_ab``) with the SAME
   held-out photos, matched by file name (each SfM uses its own cameras, so
   frame indices and coordinate frames differ between reconstructions).
3. ``frame_fit``: render PSNR of every registered frame, so the bad segments
   can be compared before/after even though most of them are training frames.
"""
from __future__ import annotations

import json
import math
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .colmap_io import read_sparse_model
from .gs_trainer import render
from .own_splat import SplatData, _to_u8, _w2c_t, comparison_image, load_ply

# Common `spirula sfm auto` flags: the capture is one phone video, one lens, frames in file order
# (`--sequence .` as the app's preprocessing runner passes for a single video; the current SfM used it too).
SFM_COMMON = ["--data-type", "video", "--camera-model", "opencv", "--camera-mode", "single",
              "--sequence", ".", "--no-masks"]

# Extra flags per reconstruction; None = the dataset's existing SfM (Spirula, quality high, SIFT).
SFM_VARIANTS: dict[str, list[str] | None] = {
    "current": None,
    "extreme_sift": ["--quality", "extreme"],
    "extreme_aliked": ["--quality", "extreme", "--features", "aliked-n16rot", "--matcher", "lightglue"],
    "high_aliked_bottomup": ["--quality", "high", "--features", "aliked-n16rot", "--matcher", "lightglue",
                             "--mapper", "bottom-up"],
}

# Help-text tokens each variant needs, so an unsupported option skips the variant instead of failing the run.
SFM_REQUIRES = {
    "extreme_sift": ["extreme"],
    "extreme_aliked": ["extreme", "aliked-n16rot", "lightglue"],
    "high_aliked_bottomup": ["aliked-n16rot", "lightglue", "bottom-up"],
}


def sfm_args(binary: str, images: Path, workspace: Path, extra: list[str], device: str) -> list[str]:
    return [str(binary), "sfm", "auto", str(images), "--output", str(workspace), *SFM_COMMON, *extra,
            "--device", device, "--lang", "en"]


# ----------------------------------------------------------------------------- COLMAP statistics


def _read(f, fmt: str):
    size = struct.calcsize("<" + fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("Truncated COLMAP binary file")
    return struct.unpack("<" + fmt, data)


def _image_observations(path: Path) -> dict[str, tuple]:
    """Per image name: (qvec, tvec, camera_id, xy of observations that have a 3D point, their point ids)."""
    out = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            props = _read(f, "idddddddi")
            name = bytearray()
            while (c := f.read(1)) != b"\x00":
                if not c:
                    raise ValueError("Truncated image name")
                name += c
            (n2d,) = _read(f, "Q")
            obs = np.frombuffer(f.read(24 * n2d), dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
            keep = obs["id"] >= 0
            out[name.decode("utf-8")] = (np.array(props[1:5]), np.array(props[5:8]), props[8],
                                         np.stack([obs["x"][keep], obs["y"][keep]], 1), obs["id"][keep])
    return out


def _point_stats(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(point ids, xyz, track length) per 3D point. (Spirula leaves the stored error field at 0.)"""
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        ids = np.empty(n, np.int64)
        xyz = np.empty((n, 3), np.float64)
        track = np.empty(n, np.int64)
        for i in range(n):
            props = _read(f, "QdddBBBd")
            ids[i] = props[0]
            xyz[i] = props[1:4]
            (t,) = _read(f, "Q")
            track[i] = t
            f.seek(8 * t, 1)
    return ids, xyz, track


def _project(cam, pts_cam: np.ndarray) -> np.ndarray:
    """Camera-frame points -> pixels with the COLMAP camera model (NaN for unsupported models)."""
    p = cam.params
    x, y = pts_cam[:, 0] / pts_cam[:, 2], pts_cam[:, 1] / pts_cam[:, 2]
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = p[0], p[1], p[2]
        k1 = p[3] if len(p) > 3 else 0.0
        k2 = p[4] if cam.model == "RADIAL" else 0.0
        r2 = x * x + y * y
        d = 1 + k1 * r2 + k2 * r2 * r2
        return np.stack([f * x * d + cx, f * y * d + cy], 1)
    if cam.model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = p[:4]
        if cam.model == "OPENCV":
            k1, k2, p1, p2 = p[4:8]
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2 * r2
            x, y = (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                    y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
        return np.stack([fx * x + cx, fy * y + cy], 1)
    return np.full((len(pts_cam), 2), np.nan)


def reprojection_errors(sparse_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per image name: pixel reprojection error of each observation; and track lengths."""
    from .colmap_io import qvec_to_rotmat

    sparse_dir = Path(sparse_dir)
    cams = read_sparse_model(sparse_dir).cameras
    pid, xyz, track = _point_stats(sparse_dir / "points3D.bin")
    order = np.argsort(pid)
    pid, xyz = pid[order], xyz[order]
    errors = {}
    for name, (q, t, cam_id, xy, ids) in _image_observations(sparse_dir / "images.bin").items():
        if not len(ids):
            errors[name] = np.empty(0)
            continue
        pos = np.searchsorted(pid, ids)
        pts = xyz[np.clip(pos, 0, len(pid) - 1)] @ qvec_to_rotmat(q).T + t
        errors[name] = np.linalg.norm(_project(cams[cam_id], pts) - xy, axis=1)
    return errors, track


def components(workspace: Path) -> list[tuple[Path, int]]:
    """sparse/N models in a Spirula/COLMAP workspace, largest first."""
    found = []
    for d in sorted((Path(workspace) / "sparse").iterdir()):
        if (d / "images.bin").is_file() and (d / "cameras.bin").is_file() and (d / "points3D.bin").is_file():
            with open(d / "images.bin", "rb") as f:
                found.append((d, _read(f, "Q")[0]))
    return sorted(found, key=lambda x: -x[1])


def sfm_stats(sparse_dir: Path, bad_names: list[str], holdout_names: list[str], n_input: int) -> dict:
    errors, track = reprojection_errors(sparse_dir)
    cams = read_sparse_model(Path(sparse_dir)).cameras
    all_err = np.concatenate([e for e in errors.values() if len(e)]) if errors else np.empty(0)
    image_err = {n: float(np.median(e)) for n, e in errors.items() if len(e)}
    bad_in = [n for n in bad_names if n in errors]
    return {
        "registered": len(errors),
        "coverage": len(errors) / max(n_input, 1),
        "cameras": len(cams),
        "points": int(len(track)),
        "reproj_error_px": float(np.mean(all_err)) if len(all_err) else float("nan"),
        "reproj_error_median_px": float(np.median(all_err)) if len(all_err) else float("nan"),
        "track_length": float(track.mean()) if len(track) else float("nan"),
        "obs_per_image_median": float(np.median([len(e) for e in errors.values()])) if errors else 0.0,
        "bad_registered": len(bad_in),
        "bad_total": len(bad_names),
        "bad_obs_median": float(np.median([len(errors[n]) for n in bad_in])) if bad_in else 0.0,
        "bad_reproj_median_px": float(np.median([image_err[n] for n in bad_in if n in image_err])) if bad_in else float("nan"),
        "holdout_registered": sum(n in errors for n in holdout_names),
        "holdout_total": len(holdout_names),
    }


def make_dataset(images: Path, sparse_dir: Path, out: Path) -> Path:
    """Folder with images/ (symlink, or copy where symlinks are unavailable) and sparse/0."""
    out = Path(out)
    (out / "sparse").mkdir(parents=True, exist_ok=True)
    target = out / "sparse" / "0"
    if not target.exists():
        shutil.copytree(sparse_dir, target)
    link = out / "images"
    if not link.exists():
        try:
            link.symlink_to(Path(images).resolve(), target_is_directory=True)
        except OSError:
            shutil.copytree(images, link)
    return out


def holdout_indices(data: SplatData, holdout_names: list[str]) -> tuple[list[int], list[str]]:
    """Indices of the fixed held-out photos in this reconstruction, and the names it did not register."""
    index = {n: i for i, n in enumerate(data.names)}
    return [index[n] for n in holdout_names if n in index], [n for n in holdout_names if n not in index]


# ----------------------------------------------------------------------------- per-frame fit


@torch.no_grad()
def frame_fit(ply: Path, data: SplatData, keep_renders: set[str], render_dir: Path, device="cuda") -> dict[str, float]:
    """Render PSNR of every registered frame (train and held-out), keyed by image name."""
    params, deg = load_ply(ply, device)
    w2c, K = _w2c_t(data, device)
    render_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, name in enumerate(data.names):
        img, _ = render(params, w2c[i], K, data.width, data.height, deg)
        out[name] = 10 * math.log10(1.0 / max(F.mse_loss(img, data.photo(i, device)).item(), 1e-12))
        if name in keep_renders:
            Image.fromarray(_to_u8(img)).save(render_dir / f"{Path(name).stem}.jpg", quality=92)
    del params
    torch.cuda.empty_cache()
    return out


def summarise_variant(name: str, stats: dict | None, frames: dict, fit: dict[str, float], data: SplatData,
                      holdout_names: list[str], bad_names: list[str], info: dict) -> dict:
    """frames: held-out metrics keyed by this reconstruction's frame index (as train_ab returns them)."""
    by_name = {data.names[int(i)]: m for i, m in frames.items()}
    bad = set(bad_names)

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    all_rows = [by_name[n] for n in holdout_names if n in by_name]
    good_rows = [by_name[n] for n in holdout_names if n in by_name and n not in bad]
    bad_rows = [by_name[n] for n in holdout_names if n in by_name and n in bad]
    holdout = set(holdout_names)
    train_fit = [v for n, v in fit.items() if n not in holdout]
    bad_fit = [fit[n] for n in bad_names if n in fit]
    return {
        "name": name,
        "sfm": stats,
        "holdout_n": len(all_rows),
        "psnr": mean(all_rows, "psnr"), "ssim": mean(all_rows, "ssim"), "lpips": mean(all_rows, "lpips"),
        "psnr_posefree": mean(good_rows, "psnr"), "lpips_posefree": mean(good_rows, "lpips"),
        "psnr_bad_holdout": mean(bad_rows, "psnr"), "bad_holdout_n": len(bad_rows),
        "train_fit_psnr": float(np.mean(train_fit)) if train_fit else float("nan"),
        "bad_fit_psnr": float(np.mean(bad_fit)) if bad_fit else float("nan"),
        "bad_fit_n": len(bad_fit),
        "frames_below_18db": int(sum(v < 18 for v in fit.values())),
        "splats": info.get("splats"),
        "minutes": info.get("train_seconds", 0) / 60,
    }


def markdown(rows: list[dict], n_holdout: int, n_bad: int) -> str:
    base = rows[0]
    lines = [
        f"## Poz onarımı: aynı {n_holdout} saklanan fotoğraf (isimle eşlenir), app eğiticisi + lr_decay, 30k",
        "",
        "| SfM | kayıtlı | yeniden proj. hata (px) | iz uzunluğu | gözlem/kare (tümü / kötü) | PSNR ↑ | Δ | LPIPS ↓ | "
        f"PSNR (poz hatası hariç) | kötü bölüm eğitim-karesi PSNR ({n_bad} kare) | <18 dB kare | splat |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["sfm"] or {}
        reg = f"{s.get('registered', '?')}" + (f" ({s['bad_registered']}/{s['bad_total']} kötü)" if s else "")
        lines.append(
            f"| {r['name']} | {reg} | {s.get('reproj_error_px', float('nan')):.3f} | {s.get('track_length', float('nan')):.2f} | "
            f"{s.get('obs_per_image_median', 0):.0f} / {s.get('bad_obs_median', 0):.0f} | "
            f"{r['psnr']:.2f} | {r['psnr'] - base['psnr']:+.2f} | {r['lpips']:.4f} | {r['psnr_posefree']:.2f} | "
            f"{r['bad_fit_psnr']:.2f} ({r['bad_fit_n']}) | {r['frames_below_18db']} | {r['splats']:,} |"
        )
    missing = [f"{r['name']}: {n_holdout - r['holdout_n']}" for r in rows if r["holdout_n"] < n_holdout]
    lines += [
        "",
        "* **kötü bölüm eğitim-karesi PSNR**: mevcut SfM'de splat'in kendi eğitim karelerini bile tutturamadığı "
        "bölümler (sabit isim listesi). Onarım işe yaradıysa bu sütun belirgin yükselir.",
        "* Her SfM kendi kameralarını kullanır; saklanan fotoğraflar isimle aynıdır ama her SfM'in lens düzeltmesiyle açılır.",
        "* **gözlem/kare**: bir karede 3D noktaya bağlanan öznitelik sayısının medyanı. Mevcut SfM'de kötü kareler ~125 gözlemle "
        "zar zor bağlı (normal kareler ~840); onarım bu sayıyı da yükseltmeli.",
        "* Yeniden projeksiyon hatasını notebook kendisi hesaplar (Spirula dosyadaki hata alanını 0 bırakıyor). Tek başına kalite "
        "ölçüsü değildir; daha çok/zor kareyi kaydeden SfM'in hatası biraz artabilir.",
    ]
    if missing:
        lines.append("* Kayıt edilemeyen saklanan fotoğraflar (ortalamaya girmez): " + ", ".join(missing))
    return "\n".join(lines)


def strip_plot(fits: dict[str, dict[str, float]], order: list[str], bad_names: list[str], out: Path) -> Path:
    """One row per reconstruction: per-frame PSNR along the video (red = fit failure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(fits), 1, figsize=(16, 1.4 * len(fits) + 0.6), sharex=True, squeeze=False)
    x = np.arange(len(order))
    for ax, (name, fit) in zip(axes[:, 0], fits.items()):
        y = np.array([fit.get(n, np.nan) for n in order])
        ax.scatter(x, y, s=3, c=np.where(y < 18, "tab:red", "tab:blue"))
        ax.axhline(18, color="0.6", lw=0.6)
        missing = np.isnan(y)
        if missing.any():
            ax.scatter(x[missing], np.full(missing.sum(), 10.5), s=3, c="0.4", marker="|")
        ax.set_ylim(10, 34)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        for n in bad_names:
            if n in order:
                ax.axvspan(order.index(n) - 0.5, order.index(n) + 0.5, color="orange", alpha=0.15, lw=0)
    axes[-1, 0].set_xlabel("video sırası (gri çizgi: o SfM'de kayıtsız kare; turuncu: mevcut SfM'in kötü bölümleri)")
    fig.suptitle("Kare başına render PSNR (eğitim + saklanan)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def focus_sheets(render_root: Path, variants: list[str], photos: dict[str, np.ndarray], out_dir: Path) -> list[Path]:
    """photo | each reconstruction's render, for the fixed focus frames (by name)."""
    paths = []
    for name, photo in photos.items():
        stem = Path(name).stem
        tiles = {"foto": photo}
        for v in variants:
            p = render_root / v / "focus" / f"{stem}.jpg"
            if p.exists():
                img = np.asarray(Image.open(p).convert("RGB"))
                if img.shape == photo.shape:
                    tiles[v] = img
                else:  # different undistortion crop -> resize for display only
                    tiles[v] = np.asarray(Image.fromarray(img).resize((photo.shape[1], photo.shape[0]), Image.LANCZOS))
        if len(tiles) > 1:
            out = out_dir / f"focus_{stem}.jpg"
            comparison_image(tiles, out, tile_width=480, box_from=next(k for k in tiles if k != "foto"))
            paths.append(out)
    return paths


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)

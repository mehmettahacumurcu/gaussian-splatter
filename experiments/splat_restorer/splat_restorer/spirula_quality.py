"""Spirula Studio training variants, scored on the same held-out photos as our other trainers.

Spirula (v2026.9.24) holds out every 10th image of the name-sorted set with
``eval_mode=interval, eval_interval=10``, i.e. the same 93 frames our own
trainers keep out. Its PLY stays in the input COLMAP frame, so it is rendered
with our cameras through ``own_splat`` exactly like every other splat.

Two scores per frame:

* ``raw``: the exported PLY as a viewer shows it (primary).
* ``cc``: after a per-image affine colour map fitted on one half of the image
  and scored on the other half (and swapped). Spirula trains per-image
  appearance (bilateral grid + PPISP) that is not exported into the PLY; this
  diagnostic shows how much of the raw gap is colour rather than geometry.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .gs_trainer import render, ssim
from .own_splat import SplatData, _to_u8, _w2c_t, comparison_image, load_ply

# Frames where our in-house IMG_5966 splat cannot fit even its TRAINING photos
# (render PSNR < 18 dB, merged and padded by 2): misregistered close-ups. Fixed
# in advance from that splat's diagnostics, never from a candidate's scores.
POSE_FAIL_SEGMENTS = [(485, 496), (518, 522), (557, 563), (702, 706), (777, 784), (789, 793)]


@dataclass
class Variant:
    name: str
    quality: str = "medium"
    iterations: int = 30000
    cap: int = 1_000_000
    flags: dict = field(default_factory=dict)
    note: str = ""
    minutes_rtx6000: int = 15  # rough wall time on an RTX PRO 6000 (96 GB)
    min_gib: float = 14


# Same settings as the 28 Sep IMG_5966 Spirula run (medium, 4K, MoGe-2 normals and
# depth at 0.01) unless a variant says otherwise.
VARIANTS = {
    "medium_holdout": Variant("medium_holdout", note="28 Eylül ayarları + saklanan kareler"),
    "medium_no_appearance": Variant(
        "medium_no_appearance", flags={"use_bilateral_grid": "false", "use_ppisp": "false"},
        note="kare başına renk telafisi kapalı (PLY renkleri dürüst)"),
    "medium_floater_mild": Variant("medium_floater_mild", flags={"floater_suppression": "mild"},
                                   note="floater bastırma: mild"),
    "medium_depth_x5": Variant("medium_depth_x5", flags={"depth_supervision_weight": "0.05"},
                               note="derinlik desteği 5x (0.01 -> 0.05)"),
    "high_native": Variant("high_native", quality="high", iterations=50000, cap=3_000_000,
                           note="Spirula high: 3M / 50k", minutes_rtx6000=45, min_gib=22),
    "ultra": Variant("ultra", quality="ultra", iterations=80000, cap=10_000_000,
                     note="Spirula ultra: 10M / 80k", minutes_rtx6000=150, min_gib=38),
}


def train_args(binary: Path, dataset: Path, recon_dir: str, out_prefix: Path, v: Variant,
               device: str, depth_maps: bool = True) -> list[str]:
    """CLI for one variant; mirrors the app's Spirula notebook plus the held-out split."""
    args = [str(binary), "train", "3dgs", "--data", str(dataset), "--colmap-recon-dir", str(recon_dir),
            "--output-dir-prefix", str(out_prefix), "--output-dir-name", "training",
            "--quality", v.quality, "--num-iterations", str(v.iterations), "--max-steps", str(v.iterations),
            "--cap-max", str(v.cap), "--steps-per-save", str(v.iterations),
            "--save-full-checkpoint", "false", "--save-only-latest-checkpoint", "true",
            "--train-resolution-divisor", "1", "--sh-degree", "3", "--cache-images", "disk",
            "--disable-viewer", "true", "--keep-viewer-alive", "false",
            "--load-normals", "true", "--normal-supervision-weight", "0.01",
            "--load-depths", str(depth_maps).lower(), "--depth-supervision-weight", "0.01" if depth_maps else "0",
            "--eval-mode", "interval", "--eval-interval", "10", "--validation-fraction", "0",
            "--save-eval-images", "true",
            "--device", device, "--lang", "en"]
    for key, value in v.flags.items():
        flag = "--" + key.replace("_", "-")
        if flag in args:  # a variant overrides a base value
            args[args.index(flag) + 1] = value
        else:
            args += [flag, value]
    return args


def find_outputs(out_prefix: Path) -> dict:
    """splat.ply, metrics.json and eval images Spirula wrote under ``out_prefix``."""
    plys = sorted(out_prefix.rglob("splat.ply"))
    metrics = sorted(out_prefix.rglob("metrics.json"))
    configs = sorted(out_prefix.rglob("config.json"))
    return {
        "ply": plys[-1] if plys else None,
        "metrics": metrics[-1] if metrics else None,
        "config": configs[-1] if configs else None,
        "eval_gt": sorted(out_prefix.rglob("eval-gt-*.png")),
        "eval_render": sorted(out_prefix.rglob("eval-render-*.png")),
    }


def _thumb(path: Path) -> np.ndarray:
    im = Image.open(path)
    im.draft("RGB", (240, 135))  # fast JPEG decode; no-op for PNG
    return np.asarray(im.convert("L").resize((96, 54), Image.BILINEAR), np.float32) / 255


def match_eval_slots(eval_gt: list[Path], data: SplatData, holdout: list[int]) -> tuple[list[int], list[float]]:
    """Frame id of each Spirula eval slot, found by image content.

    Spirula holds out the requested frames but writes eval-gt-NNNNN in its own
    order (not sorted by name), so slot k is not holdout[k]. Returns the frame
    per slot and the thumbnail PSNR of each match. Spirula saves the uncorrected
    original while our photos are undistorted, so a true match scores ~27-31 dB and a
    wrong frame ~10 dB."""
    if len(eval_gt) != len(holdout):
        raise AssertionError(f"Spirula held out {len(eval_gt)} images, we hold out {len(holdout)}")
    ours = np.stack([_thumb(data.frames_dir / data.frames[i]) for i in holdout])
    frames, psnrs = [], []
    for path in eval_gt:
        err = ((ours - _thumb(path)) ** 2).mean(axis=(1, 2))
        j = int(err.argmin())
        frames.append(holdout[j])
        psnrs.append(float(10 * np.log10(1 / max(float(err[j]), 1e-12))))
    return frames, psnrs


def native_metrics(path: Path | None) -> dict:
    """Spirula's own metrics.json, flattened to mean values (format may vary: keep numbers only)."""
    if path is None or not Path(path).exists():
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    flat = {}

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            flat[prefix[:-1]] = float(obj)
        elif isinstance(obj, list) and obj and all(isinstance(x, (int, float)) for x in obj):
            flat[prefix[:-1] + "(mean)"] = float(np.mean(obj))

    walk(raw)
    return flat


# ----------------------------------------------------------------------------- scoring


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return 10 * math.log10(1.0 / max(F.mse_loss(a, b).item(), 1e-12))


def _affine_fit(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    a = src.reshape(-1, 3).double()
    A = torch.cat([a, torch.ones_like(a[:, :1])], 1)
    return torch.linalg.lstsq(A, dst.reshape(-1, 3).double()).solution


def _apply(img: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    a = img.reshape(-1, 3).double()
    return (torch.cat([a, torch.ones_like(a[:, :1])], 1) @ M).float().reshape(img.shape).clamp(0, 1)


def colour_corrected(img: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Half-image cross-fit: map for the left half is fitted on the right half and vice versa."""
    w = img.shape[1] // 2
    out = img.clone()
    out[:, :w] = _apply(img[:, :w], _affine_fit(img[:, w:], gt[:, w:]))
    out[:, w:] = _apply(img[:, w:], _affine_fit(img[:, :w], gt[:, :w]))
    return out


def _scores(img, gt, net_lpips) -> dict:
    return {"psnr": _psnr(img, gt), "ssim": ssim(img, gt).item(),
            "lpips": net_lpips(img.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).item()}


@torch.no_grad()
def evaluate_ply(ply: Path, data: SplatData, holdout: list[int], net_lpips, render_dir: Path, device="cuda") -> dict:
    params, deg = load_ply(Path(ply), device)
    w2c, K = _w2c_t(data, device)
    render_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for i in holdout:
        img, _ = render(params, w2c[i], K, data.width, data.height, deg)
        gt = data.photo(i, device)
        rows[str(i)] = {"raw": _scores(img, gt, net_lpips), "cc": _scores(colour_corrected(img, gt), gt, net_lpips)}
        Image.fromarray(_to_u8(img)).save(render_dir / f"{i:06d}.jpg", quality=92)
    n = int(params["means"].shape[0])
    del params
    torch.cuda.empty_cache()
    return {"frames": rows, "splats": n}


def is_pose_fail(i: int) -> bool:
    return any(a <= i <= b for a, b in POSE_FAIL_SEGMENTS)


def summarise(results: dict) -> dict:
    out = {}
    for name, r in results.items():
        frames = r["frames"]
        keep = [k for k in frames if not is_pose_fail(int(k))]
        m = {"splats": r.get("splats"), "minutes": r.get("minutes"), "note": r.get("note", "")}
        for kind in ("raw", "cc"):
            for metric in ("psnr", "ssim", "lpips"):
                m[f"{kind}_{metric}"] = float(np.mean([frames[k][kind][metric] for k in frames]))
                m[f"{kind}_{metric}_clean"] = float(np.mean([frames[k][kind][metric] for k in keep]))
        m["native"] = r.get("native", {})
        out[name] = m
    return out


def markdown(summary: dict, n_frames: int, n_excluded: int, reference: str | None) -> str:
    base = summary.get(reference) if reference else None
    base = base or next(iter(summary.values()))
    lines = [
        f"## Spirula varyantları: {n_frames} saklanan kare (eğitimde hiç kullanılmadı)",
        "",
        "| varyant | PSNR (PLY) | Δ | LPIPS (PLY) | Δ | PSNR, poz hatası hariç | PSNR renk eşitlemeli | splat | dk | not |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in summary.items():
        lines.append(
            f"| {name} | {m['raw_psnr']:.2f} | {m['raw_psnr'] - base['raw_psnr']:+.2f} | {m['raw_lpips']:.4f} | "
            f"{m['raw_lpips'] - base['raw_lpips']:+.4f} | {m['raw_psnr_clean']:.2f} | {m['cc_psnr']:.2f} | "
            f"{(m['splats'] or 0):,} | {m['minutes'] or 0:.0f} | {m['note']} |")
    lines += [
        "",
        f"Δ: `{reference or next(iter(summary))}` satırına göre. **PLY** sütunları dışa aktarılan splat'in görüntüleyicide "
        "görüneceği hali (asıl karar ölçütü).",
        f"**Poz hatası hariç:** kamera pozu bozuk {n_excluded} kare (sabit liste, adaylardan bağımsız) çıkarılmış ortalama.",
        "**Renk eşitlemeli:** her karede görüntünün bir yarısından öğrenilen renk dönüşümü diğer yarıda ölçülür. Spirula "
        "eğitimde kare başına renk telafisi (bilateral grid + PPISP) öğrenir ve bu PLY'ye aktarılmaz; ham ile renk "
        "eşitlemeli arasındaki fark bunun etkisidir. Sadece teşhis içindir.",
    ]
    natives = {n: m["native"] for n, m in summary.items() if m.get("native")}
    if natives:
        lines += ["", "**Spirula'nın kendi ölçümü** (4K, kendi render'ı; ham sayılar):", ""]
        for n, nat in natives.items():
            keys = [k for k in nat if "psnr" in k.lower() or "ssim" in k.lower()][:6]
            lines.append(f"- `{n}`: " + ", ".join(f"{k} {nat[k]:.3f}" for k in keys))
    return "\n".join(lines)


def comparison_sheets(results: dict, data: SplatData, out_dir: Path, render_root: Path, holdout: list[int],
                      n_worst: int = 3, n_even: int = 4) -> list[Path]:
    names = list(results)
    first = results[names[0]]["frames"]
    ids = [i for i in holdout if not is_pose_fail(i)]
    worst = sorted(ids, key=lambda i: first[str(i)]["raw"]["psnr"])[:n_worst]
    even = [ids[k] for k in np.linspace(0, len(ids) - 1, n_even + 2).round().astype(int)[1:-1]]
    paths = []
    for i in dict.fromkeys(worst + even):
        tiles = {"foto": _to_u8(data.photo(i, "cpu"))}
        for n in names:
            p = render_root / n / "renders" / f"{i:06d}.jpg"
            if p.exists():
                tiles[n] = np.asarray(Image.open(p).convert("RGB"))
        p = out_dir / f"compare_{i:06d}.jpg"
        comparison_image(tiles, p, tile_width=480, box_from=names[0])
        paths.append(p)
    return paths


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


def variant_dict(v: Variant) -> dict:
    return asdict(v)

"""Train gsplat's reference trainer (examples/simple_trainer.py, v1.5.3) variants and
score them on the same held-out photos as our other experiments.

Each variant changes one setting of gsplat's maintained trainer, so research
features (MCMC, AbsGS, antialiasing, pose refinement, full-resolution input)
are compared on one footing. Splats are read straight from the trainer's
checkpoint and rendered with our cameras through ``own_splat``'s evaluation,
so numbers line up with the Difix and A/B tests. Frames where even nearby
*training* views fail (bad SfM poses in close-up segments) are flagged and a
second average without them is reported.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .gs_trainer import params_from_state, render
from .own_splat import SplatData, _psnr, _w2c_t


@dataclass
class GsVariant:
    name: str
    preset: str = "default"  # simple_trainer config: "default" or "mcmc"
    factor: int = 2  # data_factor: 2 -> 1920x1080 from 4K, 1 -> native 4K
    args: list[str] = field(default_factory=list)
    rasterize_mode: str = "classic"
    note: str = ""


VARIANTS = {
    "default": GsVariant("default", note="gsplat varsayılanı (orijinal 3DGS yoğunlaştırma)"),
    "mcmc": GsVariant("mcmc", preset="mcmc", args=["--strategy.cap-max", "1000000"], note="3DGS-MCMC, 1M splat"),
    "absgrad": GsVariant("absgrad", args=["--strategy.absgrad", "--strategy.grow-grad2d", "0.0006"],
                         note="AbsGS (mutlak gradyan; gsplat benchmark eşiği 0.0006)"),
    "antialiased": GsVariant("antialiased", args=["--antialiased"], rasterize_mode="antialiased",
                             note="Mip-Splatting'in 2D filtresi"),
    "pose_opt": GsVariant("pose_opt", args=["--pose-opt"], note="eğitim kamera pozlarını iyileştirir"),
    "default_4k": GsVariant("default_4k", factor=1, note="4K tam çözünürlükte eğitim"),
}


# ----------------------------------------------------------------------------- data


def prepare_factor_images(data_dir: Path, factor: int, workers: int = 8, log=print) -> None:
    """``images_<factor>`` as area-downsampled PNGs (gsplat requires the folder to exist;
    given JPGs it would re-create them with bicubic resizing instead)."""
    import cv2

    if factor <= 1:
        return
    src, dst = Path(data_dir) / "images", Path(data_dir) / f"images_{factor}"
    dst.mkdir(exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.is_file())

    def convert(p: Path):
        out = dst / p.relative_to(src).with_suffix(".png")
        if out.exists():
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(round(w / factor)), int(round(h / factor))), interpolation=cv2.INTER_AREA)
        tmp = out.with_suffix(".tmp.png")
        cv2.imwrite(str(tmp), small, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        tmp.replace(out)

    todo = [p for p in files if not (dst / p.relative_to(src).with_suffix(".png")).exists()]
    if todo:
        log(f"[images_{factor}] {len(todo)} images, area downsample x{factor}")
        with ThreadPoolExecutor(workers) as pool:
            for n, _ in enumerate(pool.map(convert, todo), 1):
                if n % 200 == 0 or n == len(todo):
                    log(f"[images_{factor}] {n}/{len(todo)}")


# ----------------------------------------------------------------------------- training


def trainer_command(v: GsVariant, data_dir: Path, result_dir: Path, steps: int, test_every: int) -> list[str]:
    return [
        sys.executable, "simple_trainer.py", v.preset,
        "--data-dir", str(data_dir), "--data-factor", str(v.factor), "--test-every", str(test_every),
        "--no-normalize-world-space", "--result-dir", str(result_dir),
        "--max-steps", str(steps), "--eval-steps", str(steps), "--save-steps", str(steps),
        "--disable-viewer", "--disable-video", *v.args,
    ]


def run_variant(v: GsVariant, data_dir: Path, examples_dir: Path, result_dir: Path, steps: int,
                test_every: int, log_every_s: float = 60.0, log=print) -> float:
    """Run simple_trainer in a subprocess; full output goes to result_dir/train.log. Returns seconds."""
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = trainer_command(v, data_dir, result_dir, steps, test_every)
    log("[train] " + " ".join(cmd[1:]))
    t0, last = time.time(), 0.0
    with open(result_dir / "train.log", "w", encoding="utf-8") as f, subprocess.Popen(
        cmd, cwd=examples_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1,
    ) as proc:
        for line in proc.stdout:
            f.write(line)
            line = line.rstrip()
            is_progress = "it/s]" in line or "s/it]" in line
            if not line or (is_progress and time.time() - last < log_every_s):
                continue
            if is_progress:
                last = time.time()
            log("  " + line[-160:])
        code = proc.wait()
    if code != 0:
        tail = (result_dir / "train.log").read_text(encoding="utf-8").splitlines()[-25:]
        raise RuntimeError(f"{v.name} failed (exit {code}):\n" + "\n".join(tail))
    return time.time() - t0


def load_checkpoint(result_dir: Path, device="cuda") -> tuple[torch.nn.ParameterDict, int]:
    """Splats from simple_trainer's last checkpoint; same layout as gs_trainer params."""
    ckpts = sorted(Path(result_dir, "ckpts").glob("ckpt_*_rank0.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint in {result_dir}/ckpts")
    splats = torch.load(ckpts[-1], map_location="cpu", weights_only=False)["splats"]
    state = {k: splats[k].float() for k in ("means", "scales", "quats", "opacities", "sh0", "shN")}
    sh_degree = int(round(math.sqrt(state["shN"].shape[1] + 1))) - 1
    return params_from_state(state, device), sh_degree


def save_ply(params, path: Path) -> None:
    """Standard 3DGS PLY (log scales, logit opacity, SH channel-major f_rest)."""
    p = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in params.items()}
    n, k = p["means"].shape[0], p["shN"].shape[1]
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(3 * k)] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    cols = [p["means"], np.zeros((n, 3), np.float32), p["sh0"].reshape(n, 3),
            p["shN"].transpose(0, 2, 1).reshape(n, 3 * k), p["opacities"].reshape(n, 1), p["scales"], p["quats"]]
    data = np.concatenate(cols, axis=1).astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n%send_header\n" % (
        n, "".join(f"property float {nm}\n" for nm in names))
    tmp = Path(path).with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    tmp.replace(path)


# ----------------------------------------------------------------------------- pose-failure frames


@torch.no_grad()
def frame_psnr(params, deg: int, data: SplatData, ids: list[int], rasterize_mode="classic", device="cuda") -> dict[int, float]:
    w2c, K = _w2c_t(data, device)
    out = {}
    for i in ids:
        img, _ = render(params, w2c[i], K, data.width, data.height, deg, rasterize_mode)
        out[i] = _psnr(img, data.photo(i, device))
    return out


def pose_failure_holdouts(train_psnr: dict[int, float], holdout: list[int], k_mad: float = 4.0,
                          neighbours: int = 2) -> tuple[list[int], list[int], float]:
    """Held-out frames whose neighbouring *training* frames the splat also fails to reproduce.

    Training frames far below the typical training PSNR (median - k*MAD) cannot be fitted
    even though they were trained on: usually wrong SfM poses (close-up segments) or
    heavy blur. Returns (flagged held-out frames, flagged training frames, threshold dB)."""
    v = np.array(list(train_psnr.values()))
    med = float(np.median(v))
    thr = med - k_mad * 1.4826 * float(np.median(np.abs(v - med)))
    bad_train = sorted(i for i, p in train_psnr.items() if p < thr)
    bad_set = set(bad_train)
    flagged = [h for h in holdout if any((h + d) in bad_set for d in range(-neighbours, neighbours + 1) if d)]
    return flagged, bad_train, thr


# ----------------------------------------------------------------------------- reporting


def summarise(results: dict, keep: list[int]) -> dict:
    out = {}
    keep_s = {str(i) for i in keep}
    for name, r in results.items():
        rows = r["frames"]
        allv = list(rows.values())
        kept = [x for i, x in rows.items() if i in keep_s]
        m = {f"{k}": float(np.mean([x[k] for x in allv])) for k in ("psnr", "ssim", "lpips")}
        m.update({f"{k}_ok": float(np.mean([x[k] for x in kept])) for k in ("psnr", "ssim", "lpips")})
        info = r.get("info", {})
        m.update(splats=info.get("splats"), minutes=info.get("train_seconds", 0) / 60, source=r.get("source", "gsplat"))
        out[name] = m
    return out


def markdown(summary: dict, n_frames: int, flagged: list[int], thr: float) -> str:
    base = summary.get("default") or next(iter(summary.values()))
    lines = [
        f"## gsplat test zemini: {n_frames} saklanan kare (hiçbiri eğitimde kullanılmadı)",
        "",
        f"'Poz hatasız' sütunları, yakınındaki eğitim kareleri bile {thr:.1f} dB altında kalan "
        f"{len(flagged)} kareyi hariç tutar (yanlış SfM pozu / yakın plan çökmesi): {flagged}",
        "",
        "| varyant | PSNR ↑ | Δ | PSNR (poz hatasız) | Δ | SSIM ↑ | LPIPS ↓ | LPIPS (poz hatasız) | splat | dk |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in summary.items():
        splats = f"{m['splats']:,}" if m.get("splats") else "-"
        lines.append(
            f"| {name} | {m['psnr']:.2f} | {m['psnr'] - base['psnr']:+.2f} | {m['psnr_ok']:.2f} | "
            f"{m['psnr_ok'] - base['psnr_ok']:+.2f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['lpips_ok']:.4f} | "
            f"{splats} | {m['minutes']:.0f} |"
        )
    lines += ["", "Δ: `default` satırına göre. 'app:' ile başlayan satırlar app'in kendi eğiticisiyle yapılan "
              "A/B testinden (aynı kareler, aynı ölçüm). +0,3 dB altındaki farklar koşudan koşuya oynayabilir."]
    return "\n".join(lines)


def save_json(path: Path, obj) -> None:
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)

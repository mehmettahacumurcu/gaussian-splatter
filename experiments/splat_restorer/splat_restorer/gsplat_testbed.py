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
    # Extra gsplat rasterization options needed to render this splat as it was trained (3DGUT).
    render: dict = field(default_factory=dict)
    # Also score with the plain rasterizer, i.e. how a standard 3DGS viewer would show the PLY.
    portable_eval: bool = False
    # Leave training frames in pose-failure segments out of training (held-out set unchanged).
    exclude_bad_frames: bool = False
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
    # Appended later; earlier variants keep their names so a resumed run skips them.
    # 3DGUT has no 2D mean gradients for the default densifier, so gsplat runs it with MCMC:
    # compare it against the `mcmc` row.
    "with_ut": GsVariant("with_ut", preset="mcmc", args=["--strategy.cap-max", "1000000", "--with-ut", "--with-eval3d"],
                         render={"with_ut": True, "with_eval3d": True}, portable_eval=True,
                         note="3DGUT + MCMC (unscented transform, 3D değerlendirme); `mcmc` ile karşılaştır"),
    "bilateral_grid": GsVariant("bilateral_grid", args=["--use-bilateral-grid"],
                                note="kare başına renk/pozlama grid'i (ölçüm grid'siz PLY ile)"),
    "depth_loss": GsVariant("depth_loss", args=["--depth-loss"], note="SfM noktalarından derinlik kaybı"),
    "mcmc_pose_opt": GsVariant("mcmc_pose_opt", preset="mcmc", args=["--strategy.cap-max", "1000000", "--pose-opt"],
                               note="MCMC + poz iyileştirme"),
    "absgrad_aa": GsVariant("absgrad_aa", args=["--strategy.absgrad", "--strategy.grow-grad2d", "0.0006", "--antialiased"],
                            rasterize_mode="antialiased",
                            note="AbsGS + anti-aliasing (gsplat'in en iyi varsayılan kombinasyonu)"),
    "exclude_bad_frames": GsVariant("exclude_bad_frames", exclude_bad_frames=True,
                                    note="poz hatalı bölümlerdeki eğitim kareleri çıkarılmış"),
}

# Wrapper around simple_trainer: optionally drops training images listed in GS_EXCLUDE_NAMES
# (a JSON list of image names) at the dataset level, so the train/val split (index % test_every)
# and therefore the held-out frames stay exactly the same.
WRAPPER = """import json, os, runpy, sys
import numpy as np
import datasets.colmap as dc
_exclude = set(json.load(open(os.environ["GS_EXCLUDE_NAMES"]))) if os.environ.get("GS_EXCLUDE_NAMES") else set()
_init = dc.Dataset.__init__
def _filtered_init(self, parser, split="train", *args, **kwargs):
    _init(self, parser, split, *args, **kwargs)
    if split == "train" and _exclude:
        before = len(self.indices)
        self.indices = np.array([i for i in self.indices if parser.image_names[i] not in _exclude],
                                dtype=self.indices.dtype)
        print(f"[tb_train] excluded {before - len(self.indices)} training images", flush=True)
dc.Dataset.__init__ = _filtered_init
sys.argv = ["simple_trainer.py", *sys.argv[1:]]
runpy.run_path("simple_trainer.py", run_name="__main__")
"""


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
        sys.executable, "tb_train.py", v.preset,
        "--data-dir", str(data_dir), "--data-factor", str(v.factor), "--test-every", str(test_every),
        "--no-normalize-world-space", "--result-dir", str(result_dir),
        "--max-steps", str(steps), "--eval-steps", str(steps), "--save-steps", str(steps),
        "--disable-viewer", "--disable-video", *v.args,
    ]


def run_variant(v: GsVariant, data_dir: Path, examples_dir: Path, result_dir: Path, steps: int,
                test_every: int, exclude_names: list[str] | None = None, log_every_s: float = 60.0,
                log=print) -> float:
    """Run simple_trainer (through WRAPPER) in a subprocess; full output goes to
    result_dir/train.log. Returns seconds."""
    import os

    result_dir.mkdir(parents=True, exist_ok=True)
    (Path(examples_dir) / "tb_train.py").write_text(WRAPPER, encoding="utf-8")
    env = dict(os.environ)
    env.pop("GS_EXCLUDE_NAMES", None)
    if exclude_names:
        (result_dir / "exclude_names.json").write_text(json.dumps(exclude_names), encoding="utf-8")
        env["GS_EXCLUDE_NAMES"] = str(result_dir / "exclude_names.json")
    cmd = trainer_command(v, data_dir, result_dir, steps, test_every)
    log("[train] " + " ".join(cmd[1:]))
    t0, last = time.time(), 0.0
    with open(result_dir / "train.log", "w", encoding="utf-8") as f, subprocess.Popen(
        cmd, cwd=examples_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
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
def frame_psnr(params, deg: int, data: SplatData, ids: list[int], rasterize_mode="classic", device="cuda",
               **raster_kwargs) -> dict[int, float]:
    w2c, K = _w2c_t(data, device)
    out = {}
    for i in ids:
        img, _ = render(params, w2c[i], K, data.width, data.height, deg, rasterize_mode, **raster_kwargs)
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
        m.update(splats=info.get("splats"), minutes=info.get("train_seconds", 0) / 60, source=r.get("source", "gsplat"),
                 gsplat_val_psnr=(r.get("gsplat_val") or {}).get("psnr"))
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
        "| varyant | PSNR ↑ | Δ | PSNR (poz hatasız) | Δ | SSIM ↑ | LPIPS ↓ | LPIPS (poz hatasız) | gsplat val PSNR "
        "| splat | dk |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in summary.items():
        splats = f"{m['splats']:,}" if m.get("splats") else "-"
        gv = f"{m['gsplat_val_psnr']:.2f}" if m.get("gsplat_val_psnr") else "-"
        lines.append(
            f"| {name} | {m['psnr']:.2f} | {m['psnr'] - base['psnr']:+.2f} | {m['psnr_ok']:.2f} | "
            f"{m['psnr_ok'] - base['psnr_ok']:+.2f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['lpips_ok']:.4f} | "
            f"{gv} | {splats} | {m['minutes']:.0f} |"
        )
    lines += ["", "Δ: `default` satırına göre. 'app:' ile başlayan satırlar app'in kendi eğiticisiyle yapılan "
              "A/B testinden (aynı kareler, aynı ölçüm). +0,3 dB altındaki farklar koşudan koşuya oynayabilir.",
              "'gsplat val PSNR': gsplat'in kendi değerlendirmesi (kendi lens düzeltmesi/kırpması ve, varsa, "
              "bilateral grid ile); kendi içinde karşılaştırılır, bizim sütunlarla birebir aynı ölçek değildir.",
              "'(klasik)' satırları, splat'in standart bir 3DGS görüntüleyicide (3DGUT olmadan) nasıl göründüğüdür."]
    return "\n".join(lines)


def save_json(path: Path, obj) -> None:
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


def gsplat_val(stats_dir: Path) -> dict | None:
    """gsplat's own validation metrics (last val_step*.json) from a run or Drive folder."""
    files = sorted(Path(stats_dir).glob("*val_step*.json"), key=lambda p: int("".join(c for c in p.stem.split("step")[-1] if c.isdigit()) or 0))
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def fix_gsplat_install(examples_dir: Path, python: str = sys.executable) -> None:
    """Make gsplat v1.5.3's example trainer importable on Colab. Idempotent.

    1. ``examples/datasets`` has no ``__init__.py``, so it is a namespace package and loses to
       the regular Hugging Face ``datasets`` package that Colab preinstalls.
    2. The pinned pycolmap fork defines ``np.uint64(-1)``, an OverflowError under numpy 2.
    3. MCMC relocate/sample_add pass unclamped split counts to the relocation kernel, which
       indexes a 51x51 binomial table with them: once one Gaussian is sampled more than 50
       times (mass opacity death, seen with cap 3M+), the kernel reads out of bounds and dies
       with "illegal memory access". The reference MCMC code clamps to N_max; so do we.
    Then imports both exactly as the trainer does, in a subprocess from ``examples_dir``.
    """
    import importlib.util
    import re

    init = Path(examples_dir) / "datasets" / "__init__.py"
    if not init.exists():
        init.write_text("", encoding="utf-8")
    gs_spec = importlib.util.find_spec("gsplat")
    if gs_spec and gs_spec.origin:
        ops = Path(gs_spec.origin).parent / "strategy" / "ops.py"
        text = ops.read_text(encoding="utf-8")
        fixed = text.replace("ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,",
                             "ratios=(torch.bincount(sampled_idxs)[sampled_idxs] + 1).clamp(max=binoms.shape[0]),")
        if fixed != text:
            ops.write_text(fixed, encoding="utf-8")
        if ".clamp(max=binoms.shape[0])" not in fixed:
            print("UYARI: gsplat MCMC yaması uygulanamadı (kod değişmiş); 3M+ MCMC çökebilir.")
    spec = importlib.util.find_spec("pycolmap")  # locate without importing (import would fail)
    if spec and spec.submodule_search_locations:
        sm = Path(next(iter(spec.submodule_search_locations))) / "scene_manager.py"
        if sm.exists():
            text = sm.read_text(encoding="utf-8")
            fixed = re.sub(r"np\.uint64\(\s*-1\s*\)", "np.uint64(np.iinfo(np.uint64).max)", text)
            if fixed != text:
                sm.write_text(fixed, encoding="utf-8")
    check = ("import numpy, pycolmap; from datasets.colmap import Parser; import datasets.colmap as d; "
             "print('ok numpy', numpy.__version__, d.__file__)")
    out = subprocess.run([python, "-c", check], cwd=examples_dir, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("gsplat trainer imports still fail:\n" + out.stderr[-2000:])
    print(out.stdout.strip())

"""Final pipeline test: one capture with a separate test walk, every remaining candidate.

The training video and the test walk are sampled into ONE flat image folder,
``train_NNNNN.png`` and ``eval_NNNNN.png``, so a single extreme SfM registers
both into the same frame. Spirula then trains with ``--eval-mode filename``
(names containing "train" train, "eval" are held out) and gsplat excludes the
eval names, so no candidate ever sees a test-walk photo. Every splat is scored
with our renderer on the registered eval frames, keyed by image name.

Frames: from every window of ``source_fps / target_fps`` video frames the
sharpest one (Laplacian variance at half resolution) is kept, at full
resolution, in one ffmpeg decode pass.
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

TRAIN_PREFIX, EVAL_PREFIX = "train_", "eval_"
FILENAME_SPLIT = ["--eval-mode", "filename"]


# ----------------------------------------------------------------------------- frames


def video_info(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate,nb_frames:stream_side_data=rotation:format=duration",
                          "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    rot = next((int(sd.get("rotation", 0)) for sd in s.get("side_data_list", []) if "rotation" in sd), 0)
    w, h = int(s["width"]), int(s["height"])
    if abs(rot) % 180 == 90:  # ffmpeg auto-rotates on decode
        w, h = h, w
    fps = float(Fraction(s["avg_frame_rate"]))
    return {"width": w, "height": h, "fps": fps, "duration": float(d["format"]["duration"]),
            "frames": int(s.get("nb_frames") or 0)}


def _passthrough_flag() -> list[str]:
    """Keep every decoded frame once (no CFR duplication); flag name changed in ffmpeg 5.1."""
    first = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n", 1)[0]
    m = re.search(r"version n?(\d+)\.(\d+)", first)
    new = m is None or (int(m.group(1)), int(m.group(2))) >= (5, 1)
    return ["-fps_mode", "passthrough"] if new else ["-vsync", "0"]


def frame_sharpness(video: Path, width: int = 1920, max_seconds: float = 0, log=print, tag="") -> list[float]:
    """Laplacian variance of every frame, on a ``width``-wide grey copy (ffmpeg does the scaling)."""
    import cv2

    info = video_info(video)
    h = round(info["height"] * width / info["width"] / 2) * 2
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-map", "0:v:0", *_passthrough_flag()]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-vf", f"scale={width}:{h}:flags=area,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=width * h * 4)
    scores, t0 = [], time.time()
    while True:
        buf = proc.stdout.read(width * h)
        if len(buf) < width * h:
            break
        scores.append(float(cv2.Laplacian(np.frombuffer(buf, np.uint8).reshape(h, width), cv2.CV_64F).var()))
        if len(scores) % 2000 == 0:
            log(f"[{tag}keskinlik] {len(scores)} kare ({len(scores) / (time.time() - t0):.0f} kare/sn)")
    if proc.wait() != 0 or not scores:
        raise RuntimeError(f"ffmpeg failed on {video} ({len(scores)} frames)")
    return scores


def pick_sharpest(scores: list[float], window: int) -> list[int]:
    """Index of the sharpest frame in each window of ``window`` frames (a short tail window is dropped)."""
    picked = []
    for a in range(0, len(scores), window):
        chunk = scores[a:a + window]
        if len(chunk) >= max(1, window // 2):
            picked.append(a + int(np.argmax(chunk)))
    return picked


def write_frames(video: Path, picked: list[int], out_dir: Path, prefix: str, max_seconds: float = 0,
                 log=print, workers: int = 8) -> None:
    """Decode once and write frames ``picked`` (in order) as ``<prefix>NNNNN.png`` at full resolution.
    (ffmpeg's select filter would avoid the pipe but ran ~6x slower than decoding in tests.)"""
    import cv2

    info = video_info(video)
    w, h = info["width"], info["height"]
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-map", "0:v:0", *_passthrough_flag()]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    # Phone/Blackmagic video is BT.709 limited range; swscale would otherwise assume BT.601.
    cmd += ["-vf", "scale=in_color_matrix=bt709:in_range=tv,format=bgr24", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    frame_bytes = w * h * 3
    order = {n: k for k, n in enumerate(picked)}
    pending = threading.Semaphore(workers * 2)  # bounds frames held in memory while PNGs are written

    def write(buf, k):
        try:
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            tmp = out_dir / f"{prefix}{k:05d}.tmp.png"
            cv2.imwrite(str(tmp), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            tmp.replace(out_dir / f"{prefix}{k:05d}.png")
        finally:
            pending.release()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes)
    t0, n, futures = time.time(), 0, []
    with ThreadPoolExecutor(workers) as pool:
        while n <= picked[-1]:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if n in order:
                pending.acquire()
                futures.append(pool.submit(write, buf, order[n]))
            n += 1
            if n % 2000 == 0:
                log(f"[{prefix}yazım] {n} kare çözüldü, {len(futures)}/{len(picked)} yazıldı "
                    f"({n / (time.time() - t0):.0f} kare/sn)")
        for f in futures:
            f.result()
    proc.kill()  # frames after the last pick are not needed
    proc.wait()
    if len(futures) != len(picked):
        raise RuntimeError(f"{video}: {len(picked)} frames picked but only {len(futures)} decoded")


def extract_sharpest(video: Path, out_dir: Path, prefix: str, target_fps: float, score_width: int = 1920,
                     max_seconds: float = 0, log=print) -> dict:
    """Write the sharpest frame of every window as ``<prefix>NNNNN.png`` (full resolution). Resumable:
    a finished extraction leaves ``.<prefix>frames.json`` (picked frame numbers + scores) and is skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / f".{prefix}frames.json"
    if done.exists():
        return json.loads(done.read_text())
    for old in out_dir.glob(f"{prefix}*.png"):
        old.unlink()
    t0 = time.time()
    info = video_info(video)
    window = max(1, round(info["fps"] / target_fps))
    all_scores = frame_sharpness(video, score_width, max_seconds, log, prefix)
    picked = pick_sharpest(all_scores, window)
    write_frames(video, picked, out_dir, prefix, max_seconds, log)
    scores = [all_scores[i] for i in picked]
    med = float(np.median(all_scores))
    rel = np.array(scores) / med
    # How much sharper the pick is than the frame a fixed stride would have taken (window start).
    stride = np.array([all_scores[i - i % window] for i in picked])
    res = {"video": str(video), "source_fps": info["fps"], "target_fps": target_fps, "window": window,
           "decoded_frames": len(all_scores), "picked": picked, "scores": scores, "median_sharpness": med,
           "picked_vs_stride_gain": float(np.median(np.array(scores) / np.maximum(stride, 1e-9))),
           "blurry_picked": int((rel < 0.35).sum()), "seconds": time.time() - t0}
    done.write_text(json.dumps(res), encoding="utf-8")
    log(f"[{prefix}frames] {len(picked)} kare ({window} karede bir en keskini), {res['seconds'] / 60:.1f} dk; "
        f"sabit aralığa göre medyan keskinlik x{res['picked_vs_stride_gain']:.2f}; "
        f"çok bulanık (medyanın %35'i altı) {res['blurry_picked']}")
    return res


def sharpness_plot(results: dict[str, dict], out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 1, figsize=(14, 2.6 * len(results)), squeeze=False)
    for ax, (name, r) in zip(axes[:, 0], results.items()):
        t = np.array(r["picked"]) / r["source_fps"]
        ax.plot(t, np.array(r["scores"]) / r["median_sharpness"], lw=0.8)
        ax.axhline(0.35, color="r", lw=0.8, ls="--")
        ax.set_title(f"{name}: seçilen karelerin keskinliği / videonun medyanı (kırmızı: çok bulanık eşiği)")
        ax.set_xlabel("saniye")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- split


def eval_indices(data) -> list[int]:
    return [i for i, n in enumerate(data.names) if Path(n).name.startswith(EVAL_PREFIX)]


def train_names(data) -> list[str]:
    return [n for n in data.names if Path(n).name.startswith(TRAIN_PREFIX)]


def registered_split(images_dir: Path, data) -> dict:
    files = sorted(p.name for p in Path(images_dir).glob("*.png"))
    n_train = sum(f.startswith(TRAIN_PREFIX) for f in files)
    n_eval = sum(f.startswith(EVAL_PREFIX) for f in files)
    return {"train_input": n_train, "eval_input": n_eval,
            "train_registered": len(train_names(data)), "eval_registered": len(eval_indices(data))}


# ----------------------------------------------------------------------------- candidates


@dataclass
class Candidate:
    name: str
    trainer: str  # "spirula" or "gsplat"
    note: str
    spirula: object = None  # spirula_quality.Variant
    gs: object = None  # gsplat_testbed.GsVariant
    steps: int = 30000  # gsplat only


def candidates() -> dict[str, Candidate]:
    from .gsplat_testbed import VARIANTS as GS
    from .spirula_quality import VARIANTS as SQ, Variant

    high = SQ["high_native"]
    return {c.name: c for c in [
        Candidate("spirula_high", "spirula", "Spirula high 3M/50k, renk telafisi açık (mevcut finalist, taban)",
                  spirula=dataclasses.replace(high, name="spirula_high")),
        Candidate("gsplat_default", "gsplat", "gsplat varsayılanı 30k, 1920 (referans eğitici)", gs=GS["default"]),
        Candidate("spirula_high_floater", "spirula", "spirula_high + floater bastırma mild",
                  spirula=Variant("spirula_high_floater", quality="high", iterations=50000, cap=3_000_000,
                                  flags={"floater_suppression": "mild"}, note="high + floater mild",
                                  minutes_rtx6000=50, min_gib=22)),
        Candidate("spirula_ultra", "spirula", "Spirula ultra 10M/80k, renk telafisi açık",
                  spirula=dataclasses.replace(SQ["ultra"], name="spirula_ultra")),
        Candidate("spirula_high_no_appearance_floater", "spirula",
                  "high + renk telafisi kapalı + floater mild (eski listedeki varyant)",
                  spirula=dataclasses.replace(SQ["high_no_appearance_floater"],
                                              name="spirula_high_no_appearance_floater")),
    ]}


def candidate_dict(c: Candidate) -> dict:
    d = {"name": c.name, "trainer": c.trainer, "note": c.note, "steps": c.steps}
    for key in ("spirula", "gs"):
        v = getattr(c, key)
        if v is not None:
            d[key] = dataclasses.asdict(v)
    return d


# ----------------------------------------------------------------------------- scoring


def evaluate_ply(ply: Path, data, ids: list[int], net_lpips, render_dir: Path | None,
                 modes: tuple[str, ...] = ("classic",), device="cuda") -> dict:
    """Per eval image name: raw + colour-corrected scores for each rasterize mode.
    Renders of the first mode are kept as JPEGs for comparison sheets."""
    import torch
    from PIL import Image

    from .gs_trainer import render
    from .own_splat import _to_u8, _w2c_t, load_ply
    from .spirula_quality import _scores, colour_corrected

    params, deg = load_ply(Path(ply), device)
    w2c, K = _w2c_t(data, device)
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)
    out = {"splats": int(params["means"].shape[0]), "modes": {}}
    with torch.no_grad():
        for m_i, mode in enumerate(modes):
            rows = {}
            for i in ids:
                img, _ = render(params, w2c[i], K, data.width, data.height, deg, mode)
                gt = data.photo(i, device)
                rows[data.names[i]] = {"raw": _scores(img, gt, net_lpips),
                                       "cc": _scores(colour_corrected(img, gt), gt, net_lpips)}
                if m_i == 0 and render_dir is not None:
                    Image.fromarray(_to_u8(img)).save(render_dir / f"{Path(data.names[i]).stem}.jpg", quality=92)
            out["modes"][mode] = rows
    del params
    torch.cuda.empty_cache()
    return out


def _mean(rows: dict, kind: str, metric: str) -> float:
    return float(np.mean([r[kind][metric] for r in rows.values()]))


def summarise(results: dict) -> dict:
    out = {}
    for name, r in results.items():
        m = {"splats": r.get("splats"), "minutes": r.get("minutes"), "note": r.get("note", ""),
             "native": r.get("native", {}), "file_mb": r.get("file_mb")}
        for mode, rows in r["modes"].items():
            for kind in ("raw", "cc"):
                for metric in ("psnr", "ssim", "lpips"):
                    m[f"{mode}_{kind}_{metric}"] = _mean(rows, kind, metric)
        m["frames"] = len(next(iter(r["modes"].values())))
        out[name] = m
    return out


def best_by_lpips(summary: dict, names: list[str]) -> str:
    return min(names, key=lambda n: summary[n]["classic_raw_lpips"])


def _native_line(nat: dict) -> str:
    keys = [k for k in nat if any(t in k.lower() for t in ("psnr", "ssim", "lpips"))][:6]
    return ", ".join(f"{k} {nat[k]:.3f}" for k in keys)


def markdown(summary: dict, split: dict, baseline: str, post: dict | None = None) -> str:
    base = summary[baseline]
    lines = [
        f"## ROOM2 final pipeline: test yürüyüşünden {split['eval_registered']} kare "
        f"(eğitime hiç girmedi; {split['eval_input']} karenin kaydedileni)",
        "",
        f"SfM: eğitim {split['train_registered']}/{split['train_input']}, test {split['eval_registered']}/{split['eval_input']} kare kaydedildi.",
        "",
        "| aday | LPIPS ↓ | Δ | PSNR | Δ | SSIM | PSNR renk eşitlemeli | splat | dk | not |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in summary.items():
        lines.append(
            f"| {name} | {m['classic_raw_lpips']:.4f} | {m['classic_raw_lpips'] - base['classic_raw_lpips']:+.4f} | "
            f"{m['classic_raw_psnr']:.2f} | {m['classic_raw_psnr'] - base['classic_raw_psnr']:+.2f} | "
            f"{m['classic_raw_ssim']:.4f} | {m['classic_cc_psnr']:.2f} | {(m['splats'] or 0):,} | "
            f"{m['minutes'] or 0:.0f} | {m['note']} |")
    lines += ["", f"Δ: `{baseline}` satırına göre. Karar ölçütü **LPIPS** + fly-through videosu (bizim ölçücü Spirula "
              "PLY'lerinde PSNR'ı ~5 dB düşük gösteriyordu; LPIPS ve göz daha güvenilir)."]
    aa = {n: m for n, m in summary.items() if "antialiased_raw_psnr" in m}
    if aa:
        lines += ["", "**Teşhis: antialiased render** (Spirula PLY'lerinde 'puslu' görünüm ve düşük PSNR'ın kaynağı "
                  "2D filtre olabilir; bu satır yükseliyorsa sorun ölçücüde):", "",
                  "| aday | classic PSNR / LPIPS | antialiased PSNR / LPIPS |", "|---|---|---|"]
        for n, m in aa.items():
            lines.append(f"| {n} | {m['classic_raw_psnr']:.2f} / {m['classic_raw_lpips']:.4f} | "
                         f"{m['antialiased_raw_psnr']:.2f} / {m['antialiased_raw_lpips']:.4f} |")
    natives = {n: m["native"] for n, m in summary.items() if m.get("native")}
    if natives:
        lines += ["", "**Spirula'nın kendi ölçümü** (aynı test kareleri, kendi render'ı, 4K):", ""]
        lines += [f"- `{n}`: {_native_line(nat)}" for n, nat in natives.items()]
    if post:
        lines += ["", f"### Son işlem: `{post['source']}`", "",
                  "| sürüm | LPIPS | PSNR | splat | dosya MB |", "|---|---|---|---|---|"]
        for n, m in post["rows"].items():
            lines.append(f"| {n} | {m['lpips']:.4f} | {m['psnr']:.2f} | {m['splats']:,} | {m['file_mb']:.1f} |")
        if post.get("notes"):
            lines += [""] + [f"- {t}" for t in post["notes"]]
    return "\n".join(lines)


def comparison_sheets(results: dict, data, ids: list[int], render_root: Path, out_dir: Path, baseline: str,
                      n_worst: int = 3, n_even: int = 4) -> list[Path]:
    """Photo | each candidate on the worst frames of the baseline (LPIPS) and evenly spaced frames."""
    from PIL import Image

    from .own_splat import _to_u8, comparison_image

    rows = results[baseline]["modes"]["classic"]
    by_name = {data.names[i]: i for i in ids}
    names = [n for n in by_name if n in rows]
    worst = sorted(names, key=lambda n: -rows[n]["raw"]["lpips"])[:n_worst]
    even = [names[k] for k in np.linspace(0, len(names) - 1, n_even + 2).round().astype(int)[1:-1]]
    paths = []
    for n in dict.fromkeys(worst + even):
        tiles = {"foto": _to_u8(data.photo(by_name[n], "cpu"))}
        for cand in results:
            p = render_root / cand / "renders" / f"{Path(n).stem}.jpg"
            if p.exists():
                tiles[cand] = np.asarray(Image.open(p).convert("RGB"))
        p = out_dir / f"compare_{Path(n).stem}.jpg"
        comparison_image(tiles, p, tile_width=480, box_from=baseline)
        paths.append(p)
    return paths


# ----------------------------------------------------------------------------- post-processing


def scene_extent(ply: Path) -> float:
    """Robust diagonal (2nd-98th percentile box) of the splat centres, in scene units."""
    from .own_splat import load_ply

    params, _ = load_ply(Path(ply), "cpu")
    xyz = params["means"].detach().numpy()
    lo, hi = np.percentile(xyz, 2, axis=0), np.percentile(xyz, 98, axis=0)
    return float(np.linalg.norm(hi - lo))


def splat_transform(args: list[str], log_path: Path, timeout: int = 3600) -> tuple[bool, str]:
    cmd = ["splat-transform", "-w", *[str(a) for a in args]]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        text, ok = p.stdout + p.stderr, p.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        text, ok = repr(e), False
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n" + text + "\n")
    return ok, text[-1500:]


def floater_args(extent: float, opacity: float = 0.1) -> str:
    """-F size,opacity,min with the voxel scaled to the scene: splat-transform's default 0.05 is
    for metric scenes (~a 5 m room = 1/140 of its diagonal); COLMAP units are arbitrary.
    ``opacity``: accumulated opacity at which a voxel counts as solid (default 0.1)."""
    return f"{extent / 140:.5f},{opacity},0.004"


def file_mb(path: Path) -> float:
    return Path(path).stat().st_size / 2**20


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

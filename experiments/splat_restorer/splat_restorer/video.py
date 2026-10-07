"""Run a restorer over every frame of a splat render video.

There is no ground-truth photo for a render video, so the outputs are for
looking at (side-by-side video, zoomed comparison sheet) plus three rough
numbers: how much the model changed each frame, how much sharper it got, and
whether frame-to-frame flicker went up (per-frame restoration is not
temporally consistent).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    s = info["streams"][0]
    num, den = (int(x) for x in s["r_frame_rate"].split("/"))
    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": num / den if den else 30.0,
        "duration": float(info.get("format", {}).get("duration", 0) or 0),
    }


def target_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Scale so the long edge is ``long_edge`` (never upscale); both sides multiples of 8."""
    s = min(1.0, long_edge / max(width, height))
    return max(8, round(width * s / 8) * 8), max(8, round(height * s / 8) * 8)


def extract_frames(video: Path, out_dir: Path, size: tuple[int, int], stride: int = 1, max_seconds: float = 0) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"scale={size[0]}:{size[1]}:flags=lanczos"
    if stride > 1:
        vf = f"select='not(mod(n\\,{stride}))',{vf}"
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video)]
    if max_seconds > 0:
        cmd += ["-t", str(max_seconds)]
    out = str(out_dir / "%05d.png")
    # -fps_mode needs ffmpeg 5.1+; Colab ships 4.4, which only knows -vsync.
    try:
        subprocess.run(cmd + ["-vf", vf, "-fps_mode", "vfr", out], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(cmd + ["-vf", vf, "-vsync", "vfr", out], check=True)
    return len(list(out_dir.glob("*.png")))


def _load(path: Path, device) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def _save(t: torch.Tensor, path: Path) -> None:
    arr = (t[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def load_ref(path: Path, size: tuple[int, int], device) -> torch.Tensor:
    """Reference photo, centre-cropped to the frame aspect and resized to the frame size."""
    img = Image.open(path).convert("RGB")
    w, h = size
    s = max(w / img.width, h / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    img = img.crop((left, top, left + w, top + h))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)


def restore_frames(model, in_dir: Path, out_dir: Path, ref: torch.Tensor | None = None, device="cuda", log=print) -> float:
    """Restore every frame not already in ``out_dir``. Returns seconds per frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(in_dir.glob("*.png")) if not (out_dir / p.name).exists()]
    if not todo:
        return 0.0
    log(f"[video] restoring {len(todo)} frames")
    t0, n = time.time(), 0
    for i, p in enumerate(todo):
        x = _load(p, device)
        h, w = x.shape[-2:]
        # The UNet wants latent sides divisible by 8, i.e. image sides divisible by 64.
        ph, pw = (-h) % 64, (-w) % 64
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if ph or pw else x
        rp = F.pad(ref, (0, pw, 0, ph), mode="reflect") if ref is not None and (ph or pw) else ref
        y = model.restore(xp, rp)[..., :h, :w]
        _save(y, out_dir / p.name)
        n += 1
        if i == 0 or (i + 1) % 25 == 0 or i + 1 == len(todo):
            log(f"[video] {i + 1}/{len(todo)}  {(time.time() - t0) / n:.2f} s/frame")
    return (time.time() - t0) / max(n, 1)


def encode(frames_dir: Path, fps: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:.6f}", "-i", str(frames_dir / "%05d.png"),
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )


def _label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    pad = max(4, img.height // 120)
    box = d.textbbox((0, 0), text)
    d.rectangle((0, 0, box[2] + 2 * pad, box[3] + 2 * pad), fill=(0, 0, 0))
    d.text((pad, pad), text, fill=(255, 255, 255))
    return img


def side_by_side(before_dir: Path, after_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(after_dir.glob("*.png")):
        if (out_dir / p.name).exists():
            continue
        a = _label(Image.open(before_dir / p.name).convert("RGB"), "ONCE (render)")
        b = _label(Image.open(p).convert("RGB"), "SONRA (Difix)")
        canvas = Image.new("RGB", (a.width + b.width, a.height))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        canvas.save(out_dir / p.name)


def _most_changed_box(a: np.ndarray, b: np.ndarray, size: int) -> tuple[int, int]:
    diff = torch.from_numpy(np.abs(a - b).mean(-1))[None, None]
    pooled = F.avg_pool2d(diff, size, stride=max(1, size // 4))
    idx = int(pooled.flatten().argmax())
    ny, nx = pooled.shape[-2:]
    step = max(1, size // 4)
    return (idx % nx) * step, (idx // nx) * step


def comparison_sheet(before_dir: Path, after_dir: Path, out: Path, rows: int = 4) -> None:
    """Rows of: before | after | zoomed before | zoomed after, zoomed where the model changed most."""
    names = sorted(p.name for p in after_dir.glob("*.png"))
    picks = [names[i] for i in np.linspace(0, len(names) - 1, min(rows, len(names))).round().astype(int)]
    tiles = []
    for name in picks:
        a = Image.open(before_dir / name).convert("RGB")
        b = Image.open(after_dir / name).convert("RGB")
        crop = max(32, min(a.width, a.height) // 3)
        x, y = _most_changed_box(np.asarray(a, np.float32), np.asarray(b, np.float32), crop)
        za = a.crop((x, y, x + crop, y + crop)).resize((a.height, a.height), Image.NEAREST)
        zb = b.crop((x, y, x + crop, y + crop)).resize((a.height, a.height), Image.NEAREST)
        d = ImageDraw.Draw(a)
        d.rectangle((x, y, x + crop, y + crop), outline=(255, 0, 0), width=max(2, a.height // 200))
        row = [_label(a, "once"), _label(b, "sonra"), _label(za, "once (zoom)"), _label(zb, "sonra (zoom)")]
        canvas = Image.new("RGB", (sum(t.width for t in row), a.height))
        xo = 0
        for t in row:
            canvas.paste(t, (xo, 0))
            xo += t.width
        tiles.append(canvas)
    sheet = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)))
    yo = 0
    for t in tiles:
        sheet.paste(t, (0, yo))
        yo += t.height
    if sheet.width > 3200:
        sheet = sheet.resize((3200, round(sheet.height * 3200 / sheet.width)), Image.LANCZOS)
    sheet.save(out, quality=92)


def _gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _laplacian_var(g: np.ndarray) -> float:
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def metrics(before_dir: Path, after_dir: Path) -> dict:
    names = sorted(p.name for p in after_dir.glob("*.png"))
    change_psnr, sharp_ratio, flick_b, flick_a = [], [], [], []
    prev_b = prev_a = None
    for name in names:
        gb, ga = _gray(before_dir / name), _gray(after_dir / name)
        mse = float(((gb - ga) ** 2).mean())
        change_psnr.append(10 * np.log10(1.0 / max(mse, 1e-10)))
        sharp_ratio.append(_laplacian_var(ga) / max(_laplacian_var(gb), 1e-10))
        if prev_b is not None:
            flick_b.append(float(np.abs(gb - prev_b).mean()))
            flick_a.append(float(np.abs(ga - prev_a).mean()))
        prev_b, prev_a = gb, ga
    fb, fa = float(np.mean(flick_b)) if flick_b else 0.0, float(np.mean(flick_a)) if flick_a else 0.0
    return {
        "frames": len(names),
        "change_psnr_db": float(np.mean(change_psnr)),
        "sharpness_ratio": float(np.median(sharp_ratio)),
        "flicker_before": fb,
        "flicker_after": fa,
        "flicker_ratio": fa / fb if fb > 0 else float("nan"),
    }


def summary_markdown(m: dict, info: dict) -> str:
    fr = m["flicker_ratio"]
    return "\n".join([
        f"## Difix video testi: {info['video']}",
        "",
        f"* Model: `{info['repo']}`" + (f", referans: `{info['ref']}`" if info.get("ref") else ""),
        f"* İşlenen kare: {m['frames']} ({info['size'][0]}x{info['size'][1]}), {info['sec_per_frame']:.2f} s/kare",
        "",
        "| ölçüm | değer | nasıl okunur |",
        "|---|---|---|",
        f"| değişim (önce↔sonra PSNR) | {m['change_psnr_db']:.1f} dB | 40+ dB: neredeyse dokunmamış, 25-35: belirgin düzeltme, <22: çok şey uydurmuş olabilir |",
        f"| keskinlik oranı | {m['sharpness_ratio']:.2f}x | >1 daha keskin; tek başına kalite demek değil (gürültü de keskinlik sayılır) |",
        f"| titreme oranı | {fr:.2f}x | ~1 iyi; 1.2+ ise kareler arası titreme artmış (kare kare işlemenin bilinen yan etkisi) |",
        "",
        "Gerçek fotoğraf olmadığı için bu sayılar doğruluk ölçmez; asıl karar `side_by_side.mp4` ve "
        "`compare.jpg`'deki zoom'lara bakarak verilir (kırmızı kutu: modelin en çok değiştirdiği bölge).",
    ])


def run(
    video: Path,
    out_dir: Path,
    model,
    repo: str,
    work_dir: Path = Path("/content/difix_video"),
    ref_image: Path | None = None,
    long_edge: int = 1024,
    stride: int = 1,
    max_seconds: float = 0,
    device: str = "cuda",
    log=print,
) -> dict:
    video, out_dir = Path(video), Path(out_dir)
    info = probe(video)
    size = target_size(info["width"], info["height"], long_edge)
    work = Path(work_dir) / f"{video.stem}_{size[0]}x{size[1]}_s{stride}_t{max_seconds:g}"
    before, after = work / "before", work / f"after_{repo.split('/')[-1]}"
    if not any(before.glob("*.png")):
        n = extract_frames(video, before, size, stride, max_seconds)
        log(f"[video] {info['width']}x{info['height']} @ {info['fps']:.2f} fps -> {n} frames at {size[0]}x{size[1]}")
    ref = load_ref(Path(ref_image), size, device) if ref_image else None
    spf = restore_frames(model, before, after, ref=ref, device=device, log=log)
    timing = after.parent / f"{after.name}_timing.json"
    if spf:
        timing.write_text(json.dumps({"sec_per_frame": spf}), encoding="utf-8")
    elif timing.exists():
        spf = json.loads(timing.read_text(encoding="utf-8"))["sec_per_frame"]

    out_dir.mkdir(parents=True, exist_ok=True)
    fps = info["fps"] / stride
    encode(after, fps, out_dir / "restored.mp4")
    sbs = work / "side_by_side"
    side_by_side(before, after, sbs)
    encode(sbs, fps, out_dir / "side_by_side.mp4")
    comparison_sheet(before, after, out_dir / "compare.jpg")
    m = metrics(before, after)
    meta = {"video": video.name, "repo": repo, "ref": Path(ref_image).name if ref_image else None,
            "size": size, "stride": stride, "sec_per_frame": spf, **m}
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    md = summary_markdown(m, meta)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    log(md)
    return {"out_dir": out_dir, "work": work, "metrics": meta}


def cleanup(work: Path) -> None:
    shutil.rmtree(work, ignore_errors=True)

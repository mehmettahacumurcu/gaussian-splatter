"""Test a restorer on one of our own trained splats.

Inputs: the COLMAP dataset the splat was trained on (``images/`` +
``sparse/<model>/``), the exported PLY and the list of held-out frames the
trainer never saw. Two questions:

1. ``evaluate_holdout``: on held-out photos, is the restored render closer to
   the real photo than the raw render (PSNR / SSIM / LPIPS)?
2. ``distill_back``: if the splat is fine-tuned with restored renders of new
   viewpoints (Difix3D-style pseudo views), do the held-out frames improve,
   compared with fine-tuning on the real photos alone?

``prepare_dataset`` reproduces the importer the splat was trained with
(undistort to PINHOLE with alpha=0, resize to the long edge, frames in sorted
name order), so frame indices and held-out photos match the training run.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .colmap_io import read_sparse_model
from .distill import restore_full
from .gs_trainer import TrainConfig, View, params_from_state, params_to_cpu_state, render, ssim, train_gaussians
from .splits import nearest_reference, subsample

# ----------------------------------------------------------------------------- data


@dataclass
class SplatData:
    frames_dir: Path
    frames: list[str]  # frame file names, index = frame id
    names: list[str]  # original image names
    width: int
    height: int
    K: np.ndarray  # (3, 3)
    w2c: np.ndarray  # (N, 4, 4)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def centers(self) -> np.ndarray:
        R, t = self.w2c[:, :3, :3], self.w2c[:, :3, 3]
        return -np.einsum("nji,nj->ni", R, t)

    @property
    def forwards(self) -> np.ndarray:
        return self.w2c[:, 2, :3]

    def photo_u8(self, i: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(Image.open(self.frames_dir / self.frames[i]).convert("RGB")).copy())

    def photo(self, i: int, device) -> torch.Tensor:
        return self.photo_u8(i).to(device).float() / 255.0


def _k_and_dist(cam) -> tuple[np.ndarray, np.ndarray]:
    p = cam.params
    if cam.model == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1:3]; dist = [0.0] * 4
    elif cam.model == "SIMPLE_RADIAL":
        fx = fy = p[0]; cx, cy = p[1:3]; dist = [p[3], 0.0, 0.0, 0.0]
    elif cam.model == "PINHOLE":
        fx, fy, cx, cy = p[:4]; dist = [0.0] * 4
    elif cam.model == "OPENCV":
        fx, fy, cx, cy = p[:4]; dist = list(p[4:8])
    else:
        raise ValueError(f"Unsupported camera model {cam.model} (fisheye/360 not handled)")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float64)
    return K, np.array(dist, dtype=np.float64)


def prepare_dataset(source: Path, out: Path, long_edge: int = 1920, model_name: str = "0",
                    workers: int = 8, log=print) -> SplatData:
    """Undistorted, resized frames + cameras, like the training run's importer. Resumable."""
    import cv2

    source, out = Path(source), Path(out)
    meta_path = out / "cameras.json"
    if meta_path.exists():
        return load_prepared(out)
    model = read_sparse_model(source / "sparse" / model_name)
    if len(model.cameras) != 1:
        raise ValueError("Only single-camera datasets are supported.")
    cam = next(iter(model.cameras.values()))
    K, dist = _k_and_dist(cam)
    w, h = cam.width, cam.height
    s = min(1.0, long_edge / max(w, h))
    ow, oh = max(1, round(w * s)), max(1, round(h * s))
    if np.any(dist):
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (ow, oh))
    else:
        newK = K.copy(); newK[0] *= ow / w; newK[1] *= oh / h
    mx, my = cv2.initUndistortRectifyMap(K, dist, None, newK, (ow, oh), cv2.CV_32FC1)

    images = sorted(model.images.values(), key=lambda im: im.name)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    def convert(item):
        i, im = item
        dest = frames_dir / f"frame_{i:06d}.png"
        if dest.exists():
            return
        img = cv2.imread(str(source / "images" / im.name))
        if img is None or img.shape[:2] != (h, w):
            raise ValueError(f"Bad image {im.name}")
        tmp = frames_dir / f"frame_{i:06d}.tmp.png"
        cv2.imwrite(str(tmp), cv2.remap(img, mx, my, cv2.INTER_LINEAR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        tmp.replace(dest)

    log(f"[data] {len(images)} images {w}x{h} {cam.model} -> {ow}x{oh} PINHOLE")
    with ThreadPoolExecutor(workers) as pool:
        for n, _ in enumerate(pool.map(convert, enumerate(images)), 1):
            if n % 100 == 0 or n == len(images):
                log(f"[data] {n}/{len(images)}")
    meta = {
        "width": ow, "height": oh, "K": newK.tolist(),
        "names": [im.name for im in images],
        "frames": [f"frame_{i:06d}.png" for i in range(len(images))],
        "w2c": [im.world_to_camera().tolist() for im in images],
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return load_prepared(out)


def load_prepared(out: Path) -> SplatData:
    meta = json.loads((Path(out) / "cameras.json").read_text(encoding="utf-8"))
    return SplatData(Path(out) / "frames", meta["frames"], meta["names"], meta["width"], meta["height"],
                     np.array(meta["K"]), np.array(meta["w2c"]))


_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "short": "<i2", "ushort": "<u2", "int": "<i4", "int32": "<i4",
              "uint": "<u4", "uint32": "<u4"}


def load_ply(path: Path, device) -> tuple[torch.nn.ParameterDict, int]:
    """Standard 3DGS PLY (log scales, logit opacity, SH) -> gs_trainer params, sh_degree."""
    with open(path, "rb") as f:
        props, n, in_vertex, elements = [], 0, False, []
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                assert "binary_little_endian" in line, f"Unsupported PLY format: {line}"
            elif line.startswith("element"):
                _, name, count = line.split()
                elements.append(name)
                in_vertex = name == "vertex"
                if in_vertex:
                    n = int(count)
            elif line.startswith("property") and in_vertex:
                _, typ, name = line.split()
                props.append((name, _PLY_TYPES[typ]))
            elif line == "end_header":
                break
        assert elements and elements[0] == "vertex", f"Expected vertex data first, got {elements}"
        arr = np.fromfile(f, dtype=np.dtype(props), count=n)
    names = arr.dtype.names
    cols = lambda prefix, k: np.stack([arr[f"{prefix}{i}"] for i in range(k)], axis=1).astype(np.float32)
    n_rest = sum(1 for p in names if p.startswith("f_rest_"))
    k = n_rest // 3
    sh_degree = int(round(math.sqrt(k + 1))) - 1
    shN = cols("f_rest_", n_rest).reshape(n, 3, k).transpose(0, 2, 1) if k else np.zeros((n, 0, 3), np.float32)
    t = lambda a: torch.nn.Parameter(torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device))
    params = torch.nn.ParameterDict({
        "means": t(np.stack([arr["x"], arr["y"], arr["z"]], axis=1)),
        "scales": t(cols("scale_", 3)),
        "quats": t(cols("rot_", 4)),
        "opacities": t(arr["opacity"].astype(np.float32)),
        "sh0": t(cols("f_dc_", 3)[:, None, :]),
        "shN": t(shN),
    })
    return params, sh_degree


# ----------------------------------------------------------------------------- metrics


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return 10 * math.log10(1.0 / max(F.mse_loss(a, b).item(), 1e-12))


def _scores(img: torch.Tensor, gt: torch.Tensor, net_lpips) -> dict:
    """img, gt: (H, W, 3) in [0, 1] on the same device."""
    return {
        "psnr": _psnr(img, gt),
        "ssim": ssim(img, gt).item(),
        "lpips": net_lpips(img.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).item(),
    }


def _w2c_t(data: SplatData, device) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor(data.w2c, dtype=torch.float32, device=device),
            torch.tensor(data.K, dtype=torch.float32, device=device))


@torch.no_grad()
def score_splat(params, sh_degree: int, data: SplatData, ids: list[int], net_lpips, device="cuda") -> dict:
    w2c, K = _w2c_t(data, device)
    rows = []
    for i in ids:
        img, _ = render(params, w2c[i], K, data.width, data.height, sh_degree)
        rows.append(_scores(img, data.photo(i, device), net_lpips))
    return {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips")}


@torch.no_grad()
def reproduce_check(params, sh_degree: int, data: SplatData, original_metrics: dict, device="cuda") -> list[dict]:
    """Re-render the frames of the training run's own evaluation and compare PSNR."""
    w2c, K = _w2c_t(data, device)
    index = {f: i for i, f in enumerate(data.frames)}
    rows = []
    for v in original_metrics["views"]:
        i = index[v["frame"]]
        img, _ = render(params, w2c[i], K, data.width, data.height, sh_degree)
        rows.append({"frame": v["frame"], "original": v["psnr"], "now": _psnr(img, data.photo(i, device))})
    return rows


# ----------------------------------------------------------------------------- restoration


def make_restore_fn(model, use_ref: bool, long_edge: int = 0):
    """fn(deg, ref) on (1, 3, H, W) tensors padded to multiples of 64.

    ``long_edge`` > 0 runs the model at that size and resizes back (the model
    was trained around 1024 px)."""

    def fn(deg, ref):
        h, w = deg.shape[-2:]
        if long_edge and max(h, w) > long_edge:
            s = long_edge / max(h, w)
            size = (max(64, round(h * s / 64) * 64), max(64, round(w * s / 64) * 64))
            d = F.interpolate(deg, size=size, mode="area")
            r = F.interpolate(ref, size=size, mode="area") if use_ref else None
            out = model.restore(d, r)
            return F.interpolate(out, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
        return model.restore(deg, ref if use_ref else None)

    return fn


def _to_u8(img: torch.Tensor) -> np.ndarray:
    return (img.float().clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def _label(img: Image.Image, text: str) -> Image.Image:
    from PIL import ImageFont

    try:  # Pillow >= 10.1 can size the built-in font
        font = ImageFont.load_default(size=max(14, img.width // 28))
    except TypeError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(img)
    pad = max(4, img.width // 120)
    box = d.textbbox((pad, pad), text, font=font)
    d.rectangle((0, 0, box[2] + pad, box[3] + pad), fill=(0, 0, 0))
    d.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return img


def _worst_box(err: np.ndarray, size: int) -> tuple[int, int]:
    t = torch.from_numpy(err)[None, None]
    step = max(1, size // 4)
    pooled = F.avg_pool2d(t, size, stride=step)
    idx = int(pooled.flatten().argmax())
    nx = pooled.shape[-1]
    return (idx % nx) * step, (idx // nx) * step


def comparison_image(tiles: dict[str, np.ndarray], out: Path, tile_width: int = 640) -> None:
    """One column per image: full frame on top, 3x zoom below where the raw render
    is furthest from the photo (red box)."""
    gt, rnd = tiles["foto"].astype(np.float32), tiles["render"].astype(np.float32)
    h, w = gt.shape[:2]
    crop = max(48, min(h, w) // 4)
    x, y = _worst_box(np.abs(gt - rnd).mean(-1), crop)
    s = tile_width / w
    top_h = round(h * s)
    canvas = Image.new("RGB", (tile_width * len(tiles), top_h + tile_width), (20, 20, 20))
    for j, (name, arr) in enumerate(tiles.items()):
        im = Image.fromarray(arr)
        small = im.resize((tile_width, top_h), Image.LANCZOS)
        ImageDraw.Draw(small).rectangle((x * s, y * s, (x + crop) * s, (y + crop) * s), outline=(255, 0, 0), width=3)
        zoom = im.crop((x, y, x + crop, y + crop)).resize((tile_width, tile_width), Image.LANCZOS)
        canvas.paste(_label(small, name), (j * tile_width, 0))
        canvas.paste(_label(zoom, f"{name} (zoom)"), (j * tile_width, top_h))
    canvas.save(out, quality=92)


def evaluate_holdout(
    params,
    sh_degree: int,
    data: SplatData,
    holdout: list[int],
    train_ids: list[int],
    methods: dict,
    out_dir: Path,
    net_lpips,
    n_images: int = 12,
    restore_long_edge: int = 0,
    device="cuda",
    log=print,
) -> dict:
    """methods: name -> (loader() -> Restorer, use_ref). Scores render and each method vs the held-out photos."""
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "per_frame.json"
    res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    w2c, K = _w2c_t(data, device)
    centers, forwards = data.centers, data.forwards

    with torch.no_grad():
        renders = {}
        for i in holdout:
            img, _ = render(params, w2c[i], K, data.width, data.height, sh_degree)
            renders[i] = img.half().cpu()
            res.setdefault(str(i), {})["render"] = _scores(img, data.photo(i, device), net_lpips)
    by_render = sorted(holdout, key=lambda i: res[str(i)]["render"]["psnr"])
    shown = sorted(set(by_render[: n_images // 2]) | set(subsample(holdout, n_images - n_images // 2)))
    log(f"[eval] {len(holdout)} held-out frames; render PSNR "
        f"{np.mean([res[str(i)]['render']['psnr'] for i in holdout]):.2f}")
    refs = {i: nearest_reference(i, train_ids, centers, forwards) for i in holdout}

    for name, (loader, use_ref) in methods.items():
        todo = [i for i in holdout if name not in res[str(i)] or (i in shown and not (img_dir / f"{name}_{i:06d}.jpg").exists())]
        if not todo:
            continue
        model = loader()
        fn = make_restore_fn(model, use_ref, restore_long_edge)
        with torch.no_grad():
            for n, i in enumerate(todo, 1):
                deg = renders[i].to(device).float()
                ref = data.photo(refs[i], device)
                out = restore_full(fn, deg, ref)
                res[str(i)][name] = _scores(out, data.photo(i, device), net_lpips)
                if i in shown:
                    Image.fromarray(_to_u8(out)).save(img_dir / f"{name}_{i:06d}.jpg", quality=95)
                if n % 20 == 0 or n == len(todo):
                    log(f"[eval] {name}: {n}/{len(todo)}")
        res_path.write_text(json.dumps(res), encoding="utf-8")
        del model, fn
        torch.cuda.empty_cache()
    res_path.write_text(json.dumps(res), encoding="utf-8")

    for i in shown:
        tiles = {"foto": _to_u8(data.photo(i, "cpu")), "render": _to_u8(renders[i])}
        for name in methods:
            p = img_dir / f"{name}_{i:06d}.jpg"
            if p.exists():
                tiles[name] = np.asarray(Image.open(p).convert("RGB"))
        tiles["referans"] = _to_u8(data.photo(refs[i], "cpu"))
        comparison_image(tiles, out_dir / f"compare_{i:06d}.jpg")

    summary = summarise(res, holdout, ["render", *methods])
    summary["shown_frames"] = shown
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = eval_markdown(summary)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    log(md)
    return summary


def summarise(res: dict, ids: list[int], names: list[str]) -> dict:
    out = {"frames": len(ids), "methods": {}}
    for name in names:
        rows = [res[str(i)][name] for i in ids if name in res[str(i)]]
        if not rows:
            continue
        base = [res[str(i)]["render"] for i in ids if name in res[str(i)]]
        out["methods"][name] = {
            "psnr": float(np.mean([r["psnr"] for r in rows])),
            "ssim": float(np.mean([r["ssim"] for r in rows])),
            "lpips": float(np.mean([r["lpips"] for r in rows])),
            "lpips_wins": int(sum(r["lpips"] < b["lpips"] for r, b in zip(rows, base))),
            "psnr_wins": int(sum(r["psnr"] > b["psnr"] for r, b in zip(rows, base))),
            "n": len(rows),
        }
    worst = sorted(ids, key=lambda i: res[str(i)]["render"]["psnr"])[:8]
    out["worst"] = [{"frame": i, **{n: res[str(i)][n] for n in names if n in res[str(i)]}} for i in worst]
    return out


def eval_markdown(s: dict) -> str:
    r = s["methods"]["render"]
    lines = [
        f"## Saklanan kareler: render vs Difix ({s['frames']} kare, splat bu fotoğrafları hiç görmedi)",
        "",
        "| yöntem | PSNR ↑ | Δ | SSIM ↑ | LPIPS ↓ | Δ | LPIPS'i iyileşen kare | PSNR'ı iyileşen kare |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in s["methods"].items():
        lines.append(
            f"| {name} | {m['psnr']:.2f} | {m['psnr'] - r['psnr']:+.2f} | {m['ssim']:.4f} | {m['lpips']:.4f} | "
            f"{m['lpips'] - r['lpips']:+.4f} | {m['lpips_wins']}/{m['n']} | {m['psnr_wins']}/{m['n']} |"
        )
    names = [n for n in s["methods"] if n != "render"]
    lines += ["", "**En kötü 8 kare (render PSNR'a göre), PSNR / LPIPS:**", "",
              "| kare | render | " + " | ".join(names) + " |", "|---|---" + "|---" * len(names) + "|"]
    for w in s["worst"]:
        cells = [f"{w['render']['psnr']:.2f} / {w['render']['lpips']:.3f}"]
        cells += [f"{w[n]['psnr']:.2f} / {w[n]['lpips']:.3f}" if n in w else "-" for n in names]
        lines.append(f"| {w['frame']} | " + " | ".join(cells) + " |")
    lines += ["", "Okuma: LPIPS (algısal fark) düşüyorsa ve PSNR belirgin düşmüyorsa düzeltme gerçek. "
              "PSNR düşüp LPIPS iyileşiyorsa model keskin ama uydurma detay ekliyor olabilir; zoom'lara bak."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- distill back


def pseudo_poses(data: SplatData, train_ids: list[int], n: int, shift_frac: float) -> list[tuple[int, np.ndarray]]:
    """New viewpoints: training cameras moved sideways / up by ``shift_frac`` of the camera path size.

    Returns (source training frame, w2c) pairs; the source photo is the restorer's reference."""
    centers = data.centers
    extent = float(np.linalg.norm(centers[train_ids].max(0) - centers[train_ids].min(0)))
    offset = shift_frac * extent
    dirs = [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)]  # camera right / left / down / up
    out = []
    for j, i in enumerate(subsample(train_ids, n)):
        axis, sign = dirs[j % len(dirs)]
        w2c = data.w2c[i].copy()
        R = w2c[:3, :3]
        c = centers[i] + sign * offset * R[axis]
        w2c[:3, 3] = -R @ c
        out.append((i, w2c))
    return out


def distill_back(
    params,
    sh_degree: int,
    data: SplatData,
    holdout: list[int],
    train_ids: list[int],
    methods: dict,
    out_dir: Path,
    net_lpips,
    steps: int = 3000,
    n_pseudo: int = 120,
    shift_frac: float = 0.03,
    pseudo_share: float = 0.3,
    train_stride: int = 2,
    restore_long_edge: int = 0,
    device="cuda",
    log=print,
) -> dict:
    """Fine-tune the splat: real photos only (control) vs photos + restored pseudo views."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.json"
    results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    start_state = params_to_cpu_state(params)
    w2c_t, K = _w2c_t(data, device)
    scene_scale = 1.1 * float(np.linalg.norm(data.centers[train_ids] - data.centers[train_ids].mean(0), axis=1).max())
    cfg = TrainConfig(steps=steps, sh_degree=sh_degree, densify=False, lr_means=1.6e-5, lr_means_final_frac=0.1, seed=0)
    score = lambda p: score_splat(p, sh_degree, data, holdout, net_lpips, device)

    if "start" not in results:
        results["start"] = score(params)
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    ft_ids = train_ids[::train_stride]
    log(f"[distill] loading {len(ft_ids)} training photos (every {train_stride}.)")
    train_views = [View(data.photo_u8(i), w2c_t[i], K) for i in ft_ids]

    if "control" not in results:
        p = train_gaussians(train_views, data.width, data.height, cfg, scene_scale=scene_scale,
                            params=params_from_state(start_state, device), log=log)
        results["control"] = score(p)
        del p
        torch.cuda.empty_cache()
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log(f"[distill] control: {results['control']}")

    poses = pseudo_poses(data, train_ids, n_pseudo, shift_frac)
    base = params_from_state(start_state, device)
    with torch.no_grad():
        renders = [render(base, torch.tensor(w, dtype=torch.float32, device=device), K,
                          data.width, data.height, sh_degree)[0].half().cpu() for _, w in poses]
    del base
    for name, (loader, use_ref) in methods.items():
        if name in results:
            continue
        model = loader()
        fn = make_restore_fn(model, use_ref, restore_long_edge)
        pseudo = []
        with torch.no_grad():
            for (src, w), rnd in zip(poses, renders):
                out = restore_full(fn, rnd.to(device).float(), data.photo(src, device))
                pseudo.append(torch.from_numpy(_to_u8(out)))
        del model, fn
        torch.cuda.empty_cache()
        ex_dir = out_dir / f"pseudo_{name}"
        ex_dir.mkdir(exist_ok=True)
        for k in subsample(list(range(len(poses))), 6):
            tiles = {"render (yeni kamera)": _to_u8(renders[k]), name: pseudo[k].numpy(),
                     "kaynak foto": _to_u8(data.photo(poses[k][0], "cpu"))}
            tw = 800
            row = [_label(Image.fromarray(a).resize((tw, round(tw * data.height / data.width)), Image.LANCZOS), t)
                   for t, a in tiles.items()]
            canvas = Image.new("RGB", (tw * len(row), row[0].height))
            for j, im in enumerate(row):
                canvas.paste(im, (tw * j, 0))
            canvas.save(ex_dir / f"pseudo_{k:03d}.jpg", quality=90)
        w_pseudo = pseudo_share / (1 - pseudo_share) * len(train_views) / max(len(pseudo), 1)
        views = train_views + [View(img, torch.tensor(w, dtype=torch.float32, device=device), K, weight=w_pseudo)
                               for (_, w), img in zip(poses, pseudo)]
        p = train_gaussians(views, data.width, data.height, cfg, scene_scale=scene_scale,
                            params=params_from_state(start_state, device), log=log)
        results[name] = score(p)
        del p, pseudo, views
        torch.cuda.empty_cache()
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log(f"[distill] {name}: {results[name]}")

    meta = {"steps": steps, "n_pseudo": n_pseudo, "shift_frac": shift_frac, "pseudo_share": pseudo_share,
            "train_photos": len(ft_ids), "holdout": len(holdout)}
    md = distill_markdown(results, meta)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    log(md)
    return results


def distill_markdown(results: dict, meta: dict) -> str:
    c = results.get("control", results["start"])
    lines = [
        f"## Splat'e geri besleme ({meta['steps']} adım ince ayar, {meta['n_pseudo']} yeni bakış, "
        f"kaydırma %{meta['shift_frac'] * 100:.0f})",
        "",
        f"Ölçüm: {meta['holdout']} saklanan kare (hiçbir koşulda eğitimde kullanılmadı).",
        "",
        "| koşul | PSNR ↑ | Δ control | SSIM ↑ | LPIPS ↓ | Δ control |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(f"| {name} | {m['psnr']:.2f} | {m['psnr'] - c['psnr']:+.2f} | {m['ssim']:.4f} | "
                     f"{m['lpips']:.4f} | {m['lpips'] - c['lpips']:+.4f} |")
    lines += ["", "* `start`: mevcut splat. `control`: aynı ince ayar, sadece gerçek fotoğraflar.",
              "* Difix koşulu `control`'ü geçiyorsa düzeltilmiş yeni bakışlar splat'in kendisini iyileştirmiş demektir."]
    return "\n".join(lines)

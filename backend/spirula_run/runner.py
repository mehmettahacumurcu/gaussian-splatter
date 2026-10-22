"""Run the Spirula pipeline on this PC with the same steps as the Colab notebook.

frames (``sam extract --adaptive``) -> SfM (``sfm auto``) -> quality gate
(largest model, coverage threshold) -> geometry (MoGe-2 normals/depth) ->
``spirula train`` with the web viewer on, so the app can show the splat
growing. Everything lives in ``<data>/spirula/<run_id>/``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Literal

from . import settings as S

ProgressFn = Callable[[str, float, str, dict], None]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
DEFAULT_BINARY = Path(r"E:\spirula.exe")
VIEWER_PORT = 7007
BELOW_NORMAL = 0x00004000
# "step   1200/50000 (  2%)  splats 412345  [elapsed 00:01:02.3 | ETA 00:40:00.0]  ... psnr=21.3"
STEP_RE = re.compile(r"step\s+(\d+)\s*/\s*(\d+).*?splats\s+(\d+)(?:.*?ETA\s+([\d:.]+))?(?:.*?psnr=([\d.]+))?")
# Generic "12/381" counters printed by sfm/geometry; only a rough in-phase estimate.
FRACTION_RE = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")


class RunError(RuntimeError):
    """Turkish message for the UI."""


class Cancelled(RunError):
    pass


def find_binary() -> Path:
    """spirula.exe: $SPIRULA_EXE, then E:\\spirula.exe, then PATH."""
    candidates = [os.environ.get("SPIRULA_EXE"), str(DEFAULT_BINARY), shutil.which("spirula")]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    raise RunError("spirula.exe bulunamadı. SPIRULA_EXE ortam değişkenine yolunu yaz.")


def binary_version(binary: Path) -> str:
    out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60,
                         encoding="utf-8", errors="replace")
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "?"


def free_port(preferred: int = VIEWER_PORT) -> int:
    for port in [preferred, *range(preferred + 1, preferred + 20)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RunError("Spirula görüntüleyicisi için boş port bulunamadı")


def parse_step(line: str) -> dict | None:
    m = STEP_RE.search(line)
    if not m:
        return None
    out = {"step": int(m.group(1)), "total": int(m.group(2)), "splats": int(m.group(3))}
    if m.group(4):
        out["eta"] = m.group(4).split(".")[0]
    if m.group(5):
        out["psnr"] = float(m.group(5))
    return out


def registered_names(images_bin: Path) -> list[str]:
    names = []
    with images_bin.open("rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            f.read(64)
            name = bytearray()
            while (c := f.read(1)) not in (b"\x00", b""):
                name += c
            names.append(name.decode("utf-8"))
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.seek(24 * n2d, 1)
    return names


def link_dir(link: Path, target: Path) -> None:
    """Directory link without admin rights: a junction on Windows, a symlink elsewhere."""
    if link.exists() or link.is_symlink():
        return
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target.resolve()), str(link))
    else:
        link.symlink_to(target.resolve(), target_is_directory=True)


@dataclass
class LocalRunRequest:
    run_id: str
    input_mode: Literal["video", "images", "dataset"]
    input_path: Path
    settings: S.SpirulaSettings
    name: str = ""


@dataclass
class _Ctx:
    binary: Path
    workspace: Path
    on_progress: ProgressFn
    cancelled: Callable[[], bool]
    manifest: dict = field(default_factory=dict)
    phase_progress: dict = field(default_factory=dict)

    @property
    def logs(self) -> Path:
        return self.workspace / "logs"

    def save(self) -> None:
        (self.workspace / "manifest.json").write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")

    def run(self, label: str, args: list[str], phase: str, accepted=(0,),
            on_line: Callable[[str], None] | None = None) -> tuple[int, str]:
        """Run one Spirula command, streaming its output to logs/<label>.log; returns (code, log text)."""
        self.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.logs / f"{label}.log"
        self.manifest.setdefault("commands", {})[label] = args
        self.save()
        last_msg = [0.0]
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + subprocess.list2cmdline(args) + "\n")
            proc = subprocess.Popen(args, cwd=self.workspace, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                                    creationflags=BELOW_NORMAL if os.name == "nt" else 0)
            stop = threading.Event()

            def watchdog():
                while not stop.wait(1.0):
                    if self.cancelled() and proc.poll() is None:
                        proc.terminate()

            threading.Thread(target=watchdog, daemon=True).start()
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    if on_line:
                        on_line(line)
                    else:
                        m = FRACTION_RE.search(line)
                        if m and 0 < int(m.group(2)) and int(m.group(1)) <= int(m.group(2)):
                            frac = min(0.99, int(m.group(1)) / int(m.group(2)))
                            self.phase_progress[phase] = max(self.phase_progress.get(phase, 0.0), frac)
                        if time.time() - last_msg[0] > 2 and line.strip():
                            self.on_progress(phase, self.phase_progress.get(phase, 0.0), line.strip()[:160], {})
                            last_msg[0] = time.time()
                code = proc.wait()
            finally:
                stop.set()
                if proc.poll() is None:
                    proc.kill()
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if self.cancelled():
            raise Cancelled("İptal edildi")
        if code not in accepted:
            tail = "\n".join(text.splitlines()[-25:])
            raise RunError(f"{label} {code} koduyla bitti. Log: {log_path}\n{tail}")
        return code, text


def _video_fps(video: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RunError("Video kare hızı için ffprobe gerekli (ffmpeg kur).")
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(video)],
                         capture_output=True, text=True, timeout=120)
    stream = json.loads(out.stdout)["streams"][0]
    return float(Fraction(stream.get("avg_frame_rate") or stream["r_frame_rate"]))


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _pick_model(dataset: Path, image_names: set[str]) -> tuple[Path, dict]:
    models = sorted(p for p in (dataset / "sparse").glob("*") if (p / "images.bin").is_file())
    if not models:
        raise RunError(f"SfM model üretmedi: {dataset / 'sparse'}")
    counts = {m: len(set(registered_names(m / "images.bin")) & image_names) for m in models}
    best = max(counts, key=counts.get)
    total = len(set().union(*(set(registered_names(m / "images.bin")) for m in models)) & image_names)
    report = {"input_images": len(image_names), "registered_images": counts[best],
              "coverage": counts[best] / max(len(image_names), 1),
              "dominant_component_share": counts[best] / max(total, 1),
              "component_count": len(models), "selected_reconstruction": str(best)}
    return best, report


def run_local(req: LocalRunRequest, data_root: Path, on_progress: ProgressFn,
              cancelled: Callable[[], bool] = lambda: False, binary: Path | None = None) -> dict:
    run = S.resolve(req.settings)
    ws = Path(data_root) / "spirula" / req.run_id
    ws.mkdir(parents=True, exist_ok=True)
    ctx = _Ctx(binary or find_binary(), ws, on_progress, cancelled,
               {"run_id": req.run_id, "name": req.name, "input_mode": req.input_mode,
                "input_path": str(req.input_path), "settings": run.as_dict(), "status": "running",
                "started_at": time.time()})
    ctx.save()
    b = str(ctx.binary)
    dataset = ws / "dataset"
    dataset.mkdir(exist_ok=True)

    # 1. Images
    on_progress("spirula_frames", 0.0, "Kareler hazırlanıyor", {})
    if req.input_mode == "video":
        if not req.input_path.is_file():
            raise RunError(f"Video bulunamadı: {req.input_path}")
        image_dir = dataset / "images"
        image_dir.mkdir(exist_ok=True)
        if not _images(image_dir):
            skip = S.frame_skip(_video_fps(req.input_path), run.fps)
            ctx.manifest["frame_skip"] = skip
            ctx.run("frame_extract", S.extract_args(b, str(req.input_path), str(image_dir), skip), "spirula_frames")
        capture_kind = "video"
    else:
        src = req.input_path
        if not src.is_dir():
            raise RunError(f"Klasör bulunamadı: {src}")
        image_dir = src / "images" if (src / "images").is_dir() else src
        capture_kind = "individual"
        link_dir(dataset / "images", image_dir)
        if req.input_mode == "dataset" and (src / "sparse").is_dir() and not (dataset / "sparse").exists():
            link_dir(dataset / "sparse", src / "sparse")
    images = _images(image_dir)
    if len(images) < 10:
        raise RunError(f"En az 10 görüntü gerekli; bulunan: {len(images)}")
    image_names = {p.relative_to(image_dir).as_posix() for p in images}
    on_progress("spirula_frames", 1.0, f"{len(images)} kare", {"images": len(images)})

    # 2. SfM (video always; folders only without an existing model)
    has_model = any((dataset / "sparse").glob("*/images.bin"))
    if req.input_mode != "dataset" or not has_model:
        if not has_model:
            on_progress("spirula_sfm", 0.0, f"SfM ({run.sfm_quality}) başlıyor", {})
            _, text = ctx.run("sfm", S.sfm_args(b, str(image_dir), str(dataset), run, capture_kind), "spirula_sfm",
                              accepted=(0, 3))
            if "RESULT:" not in text:
                raise RunError("SfM sonucu doğrulanamadı; logs/sfm.log'a bak.")
    best, report = _pick_model(dataset, image_names)
    ctx.manifest["sfm"] = report
    ctx.save()
    if (report["coverage"] < S.MIN_REGISTERED_COVERAGE or report["dominant_component_share"] < S.MIN_DOMINANT_COMPONENT_SHARE) \
            and not run.allow_partial:
        raise RunError(f"SfM kapsamı yetersiz: {report['coverage']:.0%} kare kayıtlı, en büyük parça "
                       f"%{report['dominant_component_share'] * 100:.0f}. 'Kısmi modele izin ver' ile devam edebilirsin.")
    on_progress("spirula_sfm", 1.0, f"SfM: {report['registered_images']}/{report['input_images']} kare kayıtlı", report)

    # Geometry + training read the chosen model as sparse/0 (links, so nothing is copied).
    train_ds = ws / "train_dataset"
    (train_ds / "sparse").mkdir(parents=True, exist_ok=True)
    link_dir(train_ds / "images", image_dir)
    link_dir(train_ds / "sparse" / "0", best)

    # 3. Geometry
    on_progress("spirula_geometry", 0.0, f"Geometri ({run.geometry_model})", {})
    try:
        ctx.run("geometry", S.geometry_args(b, str(train_ds), run), "spirula_geometry")
    except Cancelled:
        raise
    except RunError:
        # Known Spirula crash after all maps are written (seen as -11 on Colab): accept if complete.
        normals = list((train_ds / "normals").rglob("*.png")) if (train_ds / "normals").is_dir() else []
        if len(normals) < report["registered_images"]:
            raise
        ctx.manifest["geometry_warning"] = "geometry süreci çöktü ama tüm normal haritaları yazılmış"
    on_progress("spirula_geometry", 1.0, "Geometri hazır", {})

    # 4. Training with the live viewer
    port = free_port()
    viewer_url = f"http://127.0.0.1:{port}"
    out_prefix = ws / "outputs"
    info = {"viewer_url": viewer_url, "total": run.iterations}
    on_progress("spirula_train", 0.0, "Eğitim başlıyor", info)

    def on_line(line: str) -> None:
        st = parse_step(line)
        if st:
            on_progress("spirula_train", st["step"] / max(st["total"], 1),
                        f"{st['step']:,}/{st['total']:,} adım · {st['splats']:,} splat", {**info, **st})

    ctx.run("train", S.train_args(b, str(train_ds), "sparse/0", str(out_prefix), run, viewer_port=port),
            "spirula_train", on_line=on_line)
    plys = sorted((out_prefix / "training").rglob("splat.ply"), key=lambda p: p.stat().st_mtime)
    if not plys:
        raise RunError("Eğitim bitti ama splat.ply bulunamadı")
    splat = plys[-1]
    ctx.manifest.update({"status": "completed", "splat": str(splat), "finished_at": time.time()})
    ctx.save()
    on_progress("spirula_export", 1.0, "Splat hazır", {"splat": str(splat)})
    return {"splat_path": str(splat), "workspace": str(ws), "sfm": report, "settings": run.as_dict()}

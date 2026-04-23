"""Faz 2a — ffmpeg ile videodan PNG kareler çıkar."""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import List


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg PATH'te bulunamadı. Conda ile kur: conda install -c conda-forge ffmpeg"
        )


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: int = 10,
    resize_long_edge: int | None = None,
    overwrite: bool = False,
) -> List[Path]:
    """
    Video → numaralandırılmış PNG kareler.

    Args:
        video_path: Giriş video dosyası (mp4, mov, avi, ...)
        output_dir: Karelerin yazılacağı klasör
        fps: Saniyede kaç kare çıkarılsın (10 → başlangıç için yeterli)
        resize_long_edge: Uzun kenar bu piksele küçültülür (None = orijinal)
        overwrite: True → mevcut kareleri sil ve yeniden çıkar

    Returns:
        Çıkarılan PNG dosyalarının sıralı listesi
    """
    _check_ffmpeg()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")

    out = Path(output_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    existing = sorted(out.glob("frame_*.png"))
    if existing and not overwrite:
        print(f"⚠ {len(existing)} mevcut frame bulundu, atlanıyor (overwrite=True ile yeniden çıkar)")
        return existing

    # Video filter zinciri
    vf_parts = [f"fps={fps}"]
    if resize_long_edge:
        # Uzun kenarı resize_long_edge'e ölçekle, oran koru, çift sayıya yuvarla
        vf_parts.append(
            f"scale='if(gt(iw,ih),{resize_long_edge},-2)':'if(gt(iw,ih),-2,{resize_long_edge})'"
        )
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", vf,
        "-q:v", "1",
        str(out / "frame_%04d.png"),
    ]
    print(f"→ ffmpeg: fps={fps}, resize_long_edge={resize_long_edge}")
    subprocess.run(cmd, check=True)

    frames = sorted(out.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("ffmpeg hiç kare çıkarmadı (codec problemi olabilir)")
    print(f"✓ {len(frames)} frame çıkarıldı → {out}")
    return frames


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Videodan PNG kareler çıkar")
    p.add_argument("video", type=str, help="Giriş video dosyası")
    p.add_argument("output_dir", type=str, help="Çıktı klasörü")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--resize", type=int, default=None, help="Uzun kenarı bu piksele ölçekle")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    extract_frames(args.video, args.output_dir, args.fps, args.resize, args.overwrite)

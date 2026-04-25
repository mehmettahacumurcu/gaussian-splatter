#!/usr/bin/env python3
"""
HyperNeRF frame sequence -> MP4 dönüştürücüsü.

HyperNeRF (ve Nerfies) formatı:
    <dataset>/
        rgb/
            2x/     000001.png ... NNNNNN.png   (half res)
            4x/     ...
            8x/     ...
            16x/    ...
        camera/     000001.json ... (our pipeline bunu kullanmiyor)
        dataset.json  (frame listesi)
        scene.json    (scale vb metadata)

Bu script seçili bir çözünürlük tier'indaki PNG'leri ffmpeg ile
MP4'e concat eder. Çıktı doğrudan backend pipeline'ına verilebilir:

    python -m backend.pipeline data/<scene>/video.mp4 --scene <scene>

Örnek:
    python scripts/hypernerf_to_mp4.py \
        data/hypernerf_chickchicken/raw/chickchicken \
        --output data/hypernerf_chickchicken/video.mp4 \
        --res 2x --fps 30
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="HyperNeRF frames -> MP4")
    p.add_argument(
        "dataset_dir",
        help="HyperNeRF dataset klasörü (içinde rgb/, camera/, dataset.json olan)",
    )
    p.add_argument(
        "--output", required=True,
        help="Çıktı MP4 yolu (ör. data/hypernerf_chickchicken/video.mp4)",
    )
    p.add_argument(
        "--res", default="2x", choices=["1x", "2x", "4x", "8x", "16x"],
        help="Çözünürlük tier'i (1x=full, 2x=yarı, 4x=çeyrek ... 16x=on altıda bir)",
    )
    p.add_argument("--fps", type=int, default=30, help="Çıktı video FPS (default 30)")
    p.add_argument("--start", type=int, default=1,
                   help="Başlangıç frame index (1-based, default 1)")
    p.add_argument("--count", type=int, default=None,
                   help="İşlenecek frame sayısı (default: tümü)")
    p.add_argument("--crf", type=int, default=18,
                   help="H.264 quality (düşük=iyi, 18=lossless-like, 23=default)")
    args = p.parse_args()

    ds = Path(args.dataset_dir).resolve()
    rgb_dir = ds / "rgb" / args.res
    if not rgb_dir.exists():
        sys.exit(f"HATA: {rgb_dir} bulunamadı. --dataset_dir yanlış mı?")

    # Mevcut tüm PNG'leri bul (HyperNeRF 000001.png, 000002.png ...)
    all_frames = sorted(rgb_dir.glob("*.png"))
    if not all_frames:
        sys.exit(f"HATA: {rgb_dir} altında PNG yok")

    # Slice (1-based start)
    frames = all_frames[args.start - 1:]
    if args.count is not None:
        frames = frames[:args.count]
    if not frames:
        sys.exit(f"HATA: start={args.start} + count={args.count} ile 0 frame kaldı")

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg'e concat demuxer ile file list geç — en güvenli yol,
    # filename pattern'den bağımsız, sadece seçili frame'leri kullanır.
    list_file = out.parent / f".{out.stem}_ffmpeg_list.txt"
    duration = 1.0 / args.fps
    with open(list_file, "w", encoding="utf-8") as f:
        for fr in frames:
            # ffmpeg concat demuxer: "file '<path>'" + "duration <sec>"
            # Windows path'lerinde backslash'ları forward-slash'a çevir
            path_str = str(fr).replace("\\", "/")
            f.write(f"file '{path_str}'\n")
            f.write(f"duration {duration:.6f}\n")
        # concat demuxer son frame için duplicate entry ister
        last_str = str(frames[-1]).replace("\\", "/")
        f.write(f"file '{last_str}'\n")

    print(f"[info] {len(frames)} frame (çözünürlük {args.res}, {args.fps} fps) -> {out}")
    print(f"[info] beklenen süre: {len(frames) / args.fps:.1f} saniye")

    # -fps_mode cfr: constant frame rate zorla. VFR çıktı, downstream
    # ffmpeg -vf fps=10 filtresini yanıltıyordu (beklenenin 3x'i frame).
    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-loglevel", "warning",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-fps_mode", "cfr",
        "-r", str(args.fps),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", str(args.crf),
        "-movflags", "+faststart",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("HATA: ffmpeg bulunamadı. Conda env aktif mi? (conda activate gs4d)")
    except subprocess.CalledProcessError as e:
        sys.exit(f"HATA: ffmpeg exit {e.returncode}")
    finally:
        try:
            list_file.unlink()
        except OSError:
            pass

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"[done] {out} ({size_mb:.1f} MB)")
    print()
    print("Sonraki adım — pipeline'a gönder:")
    print(f'  curl -F "video=@{out}" -F "scene={out.parent.name}" \\')
    print(f'       -F "smoke_test=true" http://127.0.0.1:8000/process')


if __name__ == "__main__":
    main()

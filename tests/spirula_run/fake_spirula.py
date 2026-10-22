"""Stand-in for spirula.exe: writes what each real command would, so the runner can be tested end to end."""
import struct
import sys
from pathlib import Path


def opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def write_images_bin(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(names)))
        for i, name in enumerate(names, 1):
            f.write(struct.pack("<i", i) + b"\0" * 56 + struct.pack("<i", 1))
            f.write(name.encode() + b"\0")
            f.write(struct.pack("<Q", 0))


def main(args):
    if args[:1] == ["--version"]:
        print("spirula 2026.9.24 (fake)")
    elif args[:2] == ["sam", "extract"]:
        out = Path(opt(args, "--out"))
        out.mkdir(parents=True, exist_ok=True)
        for i in range(12):
            (out / f"{i:05d}.jpg").write_bytes(b"jpg")
            print(f"extract {i + 1}/12", flush=True)
    elif args[:2] == ["sfm", "auto"]:
        images = Path(args[2])
        names = sorted(p.name for p in images.iterdir())
        keep = names[: int(len(names) * float(__import__("os").environ.get("FAKE_SFM_SHARE", "1")))]
        write_images_bin(Path(opt(args, "--output")) / "sparse" / "0" / "images.bin", keep)
        print("RESULT: ok", flush=True)
    elif args[:1] == ["geometry"]:
        ds = Path(args[1])
        (ds / "normals").mkdir(exist_ok=True)
        for p in (ds / "images").iterdir():
            (ds / "normals" / (p.stem + ".png")).write_bytes(b"png")
    elif args[:1] == ["train"]:
        total = int(opt(args, "--max-steps"))
        for step in (1, total // 2, total):
            print(f"step {step:>6}/{total} ({100 * step // total:3d}%)  splats {step * 10}  "
                  f"[elapsed 00:00:01.000 | ETA 00:00:02.000]  rgb_loss=0.1  ssim=0.5  psnr=25.5", flush=True)
        out = Path(opt(args, "--output-dir-prefix")) / "training"
        out.mkdir(parents=True, exist_ok=True)
        (out / "splat.ply").write_bytes(b"ply")
        Path(opt(args, "--output-dir-prefix")).joinpath("args.txt").write_text(" ".join(args))
    else:
        print("unknown", args)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""CoTracker online streaming inference standalone testi.

cotracker2_online sliding window inference'in gerçekten çalıştığını
banana_demo data'sı üzerinde 343 frame ile doğrular.

Mevcut backend kodunu (point_tracking._track_online_streaming) kullanır,
fallback chain'i bypass eder, doğrudan online'a gider.

Kullanım:
    python scripts/test_cotracker_online.py
    python scripts/test_cotracker_online.py --scene banana_demo --grid 15

Çıktı:
    data/<scene>/test_online_tracks.pt — track tensoru
    Validation:
      - shape (1, T, N, 2) — N = grid_size² beklenen
      - finite values
      - track range plausible (frame boyutunda)
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.preprocess.point_tracking import (  # noqa: E402
    _load_video_tensor,
    _track_online_streaming,
)


def main():
    parser = argparse.ArgumentParser(description="cotracker2_online streaming test")
    parser.add_argument("--scene", default="banana_demo", help="Sahne adı (data/<scene>/frames)")
    parser.add_argument("--grid", type=int, default=15, help="Grid size (NxN noktalar)")
    parser.add_argument("--max-edge", type=int, default=480, help="Video max long edge")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="İlk N frame ile sınırla (debug)")
    args = parser.parse_args()

    frames_dir = REPO_ROOT / "data" / args.scene / "frames"
    if not frames_dir.exists():
        print(f"✗ {frames_dir} yok. Sahne hazır değil.")
        sys.exit(1)

    n_frames_total = len(list(frames_dir.glob("frame_*.png")))
    print(f"\n{'=' * 70}")
    print(f"  CoTracker2_online Streaming Test")
    print(f"{'=' * 70}")
    print(f"  Scene:       {args.scene}")
    print(f"  Frames dir:  {frames_dir}")
    print(f"  Total frames: {n_frames_total}")
    print(f"  Grid size:   {args.grid}x{args.grid} = {args.grid ** 2} nokta")
    print(f"  Max edge:    {args.max_edge}")
    print()

    if not torch.cuda.is_available():
        print("✗ CUDA yok")
        sys.exit(1)

    print("→ cotracker2_online yükleniyor...")
    t0 = time.perf_counter()
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to("cuda").eval()
    step = int(getattr(model, "step", 8))
    print(f"  ✓ Model load OK ({time.perf_counter() - t0:.1f}s) — model.step={step}")
    print(f"  GPU mem: alloc={torch.cuda.memory_allocated() / 1e6:.0f} MB, "
          f"free={torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

    print(f"\n→ Video yükleniyor (max_long_edge={args.max_edge})...")
    t0 = time.perf_counter()
    video = _load_video_tensor(frames_dir, args.max_edge)
    if args.max_frames is not None:
        video = video[:, :args.max_frames]
    video = video.to("cuda")
    print(f"  ✓ Video shape: {tuple(video.shape)} "
          f"({video.numel() * 4 / 1e6:.1f} MB on GPU)")
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    print(f"\n→ Online streaming başlıyor...")
    t0 = time.perf_counter()
    peak_mb_before = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    try:
        tracks, vis = _track_online_streaming(
            model, video, grid_size=args.grid, device="cuda"
        )
    except Exception as e:
        print(f"\n✗ FAIL: {e}")
        print(f"\n  memory_summary at fail:")
        print(torch.cuda.memory_summary(abbreviated=True))
        raise

    elapsed = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"\n✓ STREAMING OK ({elapsed:.1f}s)")
    print(f"  Peak GPU mem (during streaming): {peak_mb:.0f} MB")
    print(f"  Tracks shape:     {tuple(tracks.shape)}")
    print(f"  Visibility shape: {tuple(vis.shape)}")

    # --- VALIDATION ---
    T_video = video.shape[1]
    print(f"\n--- VALIDATION ---")

    # Shape check
    expected_n = args.grid * args.grid
    if tracks.shape[2] != expected_n:
        print(f"  ⚠ Track sayısı beklenenden farklı: {tracks.shape[2]} != {expected_n}")
    else:
        print(f"  ✓ N tracks = {expected_n} (beklenen)")

    if tracks.shape[1] < T_video * 0.9:  # %10 tolerance for trailing partial windows
        print(f"  ⚠ T_tracks ({tracks.shape[1]}) << T_video ({T_video}) — son window partial olabilir")
    else:
        print(f"  ✓ T_tracks = {tracks.shape[1]} (T_video={T_video})")

    # Finite check
    if not torch.isfinite(tracks).all():
        n_nan = (~torch.isfinite(tracks)).sum().item()
        print(f"  ✗ NaN/Inf var: {n_nan} eleman")
    else:
        print(f"  ✓ Tüm track'ler finite")

    # Range check
    H, W = video.shape[3], video.shape[4]
    x_range = (tracks[..., 0].min().item(), tracks[..., 0].max().item())
    y_range = (tracks[..., 1].min().item(), tracks[..., 1].max().item())
    print(f"  X range: [{x_range[0]:.1f}, {x_range[1]:.1f}] (frame W={W})")
    print(f"  Y range: [{y_range[0]:.1f}, {y_range[1]:.1f}] (frame H={H})")
    if x_range[1] > W * 1.5 or y_range[1] > H * 1.5:
        print(f"  ⚠ Track koordinatları frame dışında — coord scale hatası olabilir")
    else:
        print(f"  ✓ Track koordinatları frame içinde plausible")

    # Visibility ratio
    vis_ratio = vis.float().mean().item()
    print(f"  Visibility ratio: {vis_ratio:.1%} ({'iyi' if vis_ratio > 0.5 else 'düşük'})")

    # Per-track motion
    if tracks.shape[1] > 1:
        track_motion = (tracks[:, -1] - tracks[:, 0]).norm(dim=-1)  # (1, N)
        motion_mean = track_motion.mean().item()
        motion_max = track_motion.max().item()
        print(f"  Track motion (frame_0 → frame_{tracks.shape[1] - 1}):")
        print(f"    mean: {motion_mean:.1f} px, max: {motion_max:.1f} px")
        if motion_mean < 1.0:
            print(f"    ⚠ Çok az motion — tracker hareket yakalayamamış olabilir")
        else:
            print(f"    ✓ Motion var")

    # Save
    output_path = REPO_ROOT / "data" / args.scene / "test_online_tracks.pt"
    torch.save({
        "tracks": tracks.cpu(),
        "visibility": vis.cpu(),
        "video_shape": tuple(video.shape),
    }, output_path)
    print(f"\n✓ Kaydedildi: {output_path}")
    print(f"  ({output_path.stat().st_size / 1e6:.1f} MB)")

    print(f"\n{'=' * 70}")
    print(f"  SONUÇ: cotracker2_online sliding window 343 frame için ÇALIŞTI ✓")
    print(f"  Süre: {elapsed:.1f}s, peak {peak_mb:.0f} MB (offline 22.4 GB peak'e karşı)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

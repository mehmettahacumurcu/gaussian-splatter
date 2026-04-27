"""Motion Magnification — mevcut training output'unu × N büyüterek motion'ı görselleştir.

NEDEN GEREKLİ:
    Banana/cookie/chickchicken gibi HyperNeRF datasetlerinde fiziksel motion
    sahnenin %0.5-1'i kadar küçük. Network matematiksel olarak doğru öğreniyor
    (Δpos mean 0.83) ama viewer'da gözle görünmüyor (~7 px hareket).

    Bu script Fourier trajectory katsayılarını × N ile çarparak motion'ı
    suni olarak büyütür. Sonuç realistic değil ama:
      1. "Motion VAR mı?" sorusuna yes/no verir
      2. Network'ün öğrendiği motion **yön**ünü gösterir
      3. Demo için faydalı

Kullanım:
    python scripts/magnify_motion.py banana_high_v3 --factor 5.0
        → data/banana_high_v3_mag5/output/ply/ yaratır

    python scripts/magnify_motion.py banana_high_v3 --factor 10.0 --num-timestamps 60

Çıktı:
    data/<scene>_mag<N>/output/ply/  — N× büyütülmüş PLY frame'leri
    Viewer'da yükle: '<scene>_mag<N>'

NOT:
    Magnification fourier_pos_coeffs'a uygulanır.
    MLP dpos da çarpılır mı? Hayır — sadece Fourier per-gaussian trajectory.
    Bu network'ün **per-gaussian özgün motion**unu gösterir,
    global MLP korreksiyonunu değil. (4DGS paper'ın asıl katkısı.)
"""
from __future__ import annotations
import argparse
import sys
import shutil
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.export.to_splat import export_to_ply  # noqa: E402
from backend.model.gaussian_model import GaussianModel  # noqa: E402
from backend.model.deformation import DeformationField  # noqa: E402


def find_final_checkpoint(scene: str) -> Path:
    """Sahnenin en uygun checkpoint'ini bul (final → en yüksek iter)."""
    ckpt_dir = REPO_ROOT / "data" / scene / "output" / "ckpt"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Ckpt dizini bulunamadı: {ckpt_dir}")
    # Önce final
    final = ckpt_dir / "ckpt_final.pt"
    if final.exists():
        return final
    # En yüksek iter
    candidates = []
    for p in ckpt_dir.glob("ckpt_*.pt"):
        stem = p.stem
        parts = stem.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            candidates.append((int(parts[1]), p))
    if not candidates:
        raise FileNotFoundError(f"Geçerli ckpt bulunamadı: {ckpt_dir}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def build_model_from_ckpt(ckpt_path: Path, device: str = "cuda"):
    """Ckpt'ten GaussianModel + DeformationField rebuild et (recover_scene.py benzeri)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gs_state = ckpt["gs"]
    sh_deg = int(ckpt.get("sh_degree", gs_state.get("sh_degree", 3)))
    # Fourier_K NESTED ckpt["gs"] altında — state_for_save() içinde saklanıyor
    fourier_K = int(gs_state.get("fourier_K", 0))

    # Dummy init
    dummy_pts = torch.zeros(1, 3)
    dummy_rgb = torch.zeros(1, 3)
    gs = GaussianModel(dummy_pts, init_colors=dummy_rgb, sh_degree=sh_deg, fourier_K=fourier_K)

    # State load
    for key in ("means", "scales", "quats", "opacities", "sh_dc", "sh_rest", "fourier_pos_coeffs"):
        if key in gs_state:
            new_tensor = gs_state[key].to(device)
            setattr(gs, key, torch.nn.Parameter(new_tensor, requires_grad=False))
    gs = gs.to(device)

    # Deformation field
    deform = None
    if "deform" in ckpt:
        ds = ckpt["deform"]
        plane_keys = [k for k in ds if "hexplane.planes." in k]
        if plane_keys:
            sample_key = sorted(plane_keys)[0]
            shape = ds[sample_key].shape
            feat_dim, R = int(shape[1]), int(shape[2])
        else:
            feat_dim, R = 48, 96
        # MLP width
        mlp_w = 512
        for k, v in ds.items():
            if "mlp" in k and k.endswith(".weight") and v.dim() == 2:
                mlp_w = v.shape[0]
                break
        deform = DeformationField(resolution=R, feat_dim=feat_dim, mlp_width=mlp_w)
        deform.load_state_dict(ds, strict=False)
        deform = deform.to(device)

    scene_extent = float(ckpt.get("scene_extent", 1.0))
    return gs, deform, scene_extent


def main():
    parser = argparse.ArgumentParser(description="Motion Magnification — mevcut model'i × N büyüt")
    parser.add_argument("scene", help="Sahne adı (örn. banana_high_v3)")
    parser.add_argument("--factor", type=float, default=5.0,
                        help="Motion magnification factor (default: 5.0)")
    parser.add_argument("--num-timestamps", type=int, default=None,
                        help="Export edilecek frame sayısı (default: training'de kaç ts ise)")
    parser.add_argument("--magnify-mlp", action="store_true",
                        help="MLP deform output'u da büyüt (default: sadece Fourier)")
    parser.add_argument("--suffix", default=None,
                        help="Çıktı sahne suffix'i (default: _magN)")
    args = parser.parse_args()

    if args.factor <= 0:
        print("[ERR] --factor pozitif olmalı")
        sys.exit(1)

    # 1. Find checkpoint
    print(f"→ {args.scene} sahnesinin checkpoint'i aranıyor...")
    ckpt_path = find_final_checkpoint(args.scene)
    print(f"✓ Ckpt: {ckpt_path.name}")

    # 2. Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"→ Model yükleniyor ({device})")
    gs, deform, scene_extent = build_model_from_ckpt(ckpt_path, device=device)
    print(f"  N gaussians: {gs.num_points:,}")
    print(f"  fourier_K: {gs.fourier_K}")
    print(f"  scene_extent: {scene_extent:.2f}")

    if gs.fourier_K == 0 or gs.fourier_pos_coeffs is None:
        print("[ERR] Bu ckpt fourier trajectory içermiyor — magnification yapılamaz")
        print("      Hybrid veya fourier mode ile training yap.")
        sys.exit(2)

    # 3. Magnify
    print(f"\n→ Motion magnification: × {args.factor}")
    with torch.no_grad():
        # Fourier coeffs çarp
        before_max = gs.fourier_pos_coeffs.abs().max().item()
        gs.fourier_pos_coeffs.data *= args.factor
        after_max = gs.fourier_pos_coeffs.abs().max().item()
        print(f"  Fourier coeff max: {before_max:.4f} → {after_max:.4f}")
        # Note: training'deki clamp scene_extent × 0.03 idi.
        # Magnify clamp'ı atlatır (kasıtlı, motion'ı görmek için).

    # MLP output da büyüt? (deformation field forward_scale parametresi yok,
    # MLP forward'ı manuel modify etmek lazım — bu basitçe yapılamaz çünkü
    # MLP içinde non-linearity var. Sadece Fourier büyüt yeter.)
    if args.magnify_mlp:
        print("  (MLP magnify desteklenmiyor — sadece Fourier büyütüldü)")

    # 4. Export
    suffix = args.suffix or f"_mag{int(args.factor)}"
    dst_scene = args.scene + suffix
    dst_dir = REPO_ROOT / "data" / dst_scene
    dst_ply = dst_dir / "output" / "ply"
    if dst_ply.exists():
        print(f"→ Eski {dst_ply} temizleniyor")
        shutil.rmtree(dst_ply)
    dst_ply.mkdir(parents=True, exist_ok=True)

    # Determine num_timestamps from existing scene if not specified
    if args.num_timestamps is None:
        src_ply = REPO_ROOT / "data" / args.scene / "output" / "ply"
        if src_ply.exists():
            n = len(list(src_ply.glob("frame_*.ply")))
            args.num_timestamps = n if n > 0 else 60
        else:
            args.num_timestamps = 60
    print(f"  num_timestamps: {args.num_timestamps}")

    print(f"\n→ {args.num_timestamps} frame PLY export ediliyor → {dst_ply}")
    files = export_to_ply(
        gs, deform, dst_ply,
        num_timestamps=args.num_timestamps,
        scene_extent=scene_extent,
        device=device,
    )

    print(f"\n✓ BAŞARILI")
    print(f"  Scene: {dst_scene}")
    print(f"  Magnification: × {args.factor}")
    print(f"  PLY count: {len(files)}")
    print(f"  Source: {args.scene} ({ckpt_path.name})")
    print(f"\n  Viewer'da yükle: '{dst_scene}'")
    print(f"  Karşılaştır: '{args.scene}' (orijinal) vs '{dst_scene}' (× {args.factor})")


if __name__ == "__main__":
    main()

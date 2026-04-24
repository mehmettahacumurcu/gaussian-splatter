"""Recovery — sahne NaN'lı PLY'lere diverge ettiyse son valid checkpoint'ten re-export.

Kullanım:
    python scripts/recover_scene.py cutlemon_full

Ne yapar:
    1. data/<scene>/output/ckpt/*.pt içindeki tüm checkpoint'leri tarar
    2. En büyük iter'den başlayarak geriye gider, ilk NaN-içermeyen ckpt'yi bulur
    3. O ckpt'ten 60 frame PLY export eder → data/<scene>_recovered/output/ply/
    4. Frontend viewer'da "<scene>_recovered" ismiyle yüklenebilir

Çıktı:
    ✓ Recovered scene: <scene>_recovered (from ckpt_018000.pt)
"""
from __future__ import annotations
import sys
import shutil
from pathlib import Path

import torch
import numpy as np

# Proje root'u path'e ekle
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.export.to_splat import export_to_ply  # noqa: E402
from backend.model.gaussian_model import GaussianModel  # noqa: E402
from backend.model.deformation import DeformationField  # noqa: E402


def scan_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Ckpt dizinindeki tüm ckpt_XXXXXX.pt dosyalarını (iter, path) olarak sırala (iter DESC)."""
    found = []
    for p in ckpt_dir.glob("ckpt_*.pt"):
        stem = p.stem  # ckpt_012000
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        try:
            it = int(parts[1])
        except ValueError:
            continue
        found.append((it, p))
    # Final ckpt varsa onu da dahil et (iter olarak son iter+1 değil kendi adını koruyalım)
    final = ckpt_dir / "ckpt_final.pt"
    if final.exists() and not any(p.name == "ckpt_final.pt" for _, p in found):
        # final'i en son yere koy, iter=+inf
        found.append((10**9, final))
    found.sort(key=lambda x: x[0], reverse=True)  # büyük iter önce
    return found


def is_ckpt_valid(path: Path) -> tuple[bool, dict]:
    """Ckpt'yi yükle, NaN kontrolü yap. (valid, metrics)."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, {"error": f"load failed: {e}"}

    gs_state = ckpt.get("gs", {})
    metrics = {}
    all_valid = True
    for key in ("means", "scales", "quats", "opacities", "sh_dc", "sh_rest"):
        t = gs_state.get(key)
        if t is None:
            continue
        n_nan = int(torch.isnan(t).sum().item())
        n_inf = int(torch.isinf(t).sum().item())
        pct = (n_nan + n_inf) / max(t.numel(), 1) * 100
        metrics[key] = f"NaN={n_nan} Inf={n_inf} ({pct:.1f}%)"
        if n_nan > 0 or n_inf > 0:
            all_valid = False

    deform_state = ckpt.get("deform", {})
    deform_nan = 0
    for v in deform_state.values():
        if isinstance(v, torch.Tensor):
            deform_nan += int(torch.isnan(v).sum().item())
    metrics["deform_total_nan"] = deform_nan
    if deform_nan > 0:
        all_valid = False

    return all_valid, metrics


def build_model_from_ckpt(ckpt_path: Path, device: str = "cuda"):
    """Ckpt'ten GaussianModel + DeformationField rebuild et."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gs_state = ckpt["gs"]
    sh_deg = int(ckpt.get("sh_degree", 3))

    # GaussianModel'i initialize et (dummy pts + init_colors), sonra state yükle
    dummy_pts = torch.zeros(1, 3)
    dummy_rgb = torch.zeros(1, 3)
    gs = GaussianModel(dummy_pts, init_colors=dummy_rgb, sh_degree=sh_deg)
    # state'i params olarak kopyala
    for key in ("means", "scales", "quats", "opacities", "sh_dc", "sh_rest"):
        if key in gs_state:
            param = getattr(gs, key)
            new_tensor = gs_state[key].to(device)
            # Parameter shape match gerekmez — yeni bir Parameter yaratıp atanır
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
        # load_state_dict strict=False çünkü bazı key'ler eksik olabilir
        deform.load_state_dict(ds, strict=False)
        deform = deform.to(device)

    scene_extent = float(ckpt.get("scene_extent", 1.0))
    return gs, deform, scene_extent


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/recover_scene.py <scene_name> [num_timestamps=60]")
        sys.exit(1)

    scene = sys.argv[1]
    num_ts = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    data_root = REPO_ROOT / "data"
    src = data_root / scene
    ckpt_dir = src / "output" / "ckpt"
    if not ckpt_dir.exists():
        print(f"[ERR] {ckpt_dir} yok")
        sys.exit(2)

    print(f"→ {ckpt_dir} taranıyor...")
    ckpts = scan_checkpoints(ckpt_dir)
    if not ckpts:
        print("[ERR] Hiç checkpoint bulunamadı")
        sys.exit(3)

    # Son ckpt'ten geriye dön, ilk valid olanı bul
    chosen = None
    for it, path in ckpts:
        label = f"ckpt_{it:06d}" if it != 10**9 else "ckpt_final"
        print(f"  {label}.pt kontrol ediliyor...", end=" ")
        ok, metrics = is_ckpt_valid(path)
        if ok:
            print("✓ VALID")
            chosen = (it, path, metrics)
            break
        else:
            key_issues = ", ".join(
                f"{k}={v}" for k, v in metrics.items()
                if ("NaN=" in str(v) and "NaN=0" not in str(v)) or k == "deform_total_nan" and v > 0
            )
            print(f"✗ NaN/Inf: {key_issues}")

    if chosen is None:
        print("\n[ERR] Hiçbir ckpt valid değil — model tamamen NaN. Yeniden training gerek.")
        sys.exit(4)

    it, path, metrics = chosen
    label = f"ckpt_{it:06d}" if it != 10**9 else "ckpt_final"
    print(f"\n✓ Seçilen: {label}.pt")

    # Target dizin
    dst_scene = f"{scene}_recovered"
    dst = data_root / dst_scene
    dst_output = dst / "output"
    dst_ply = dst_output / "ply"
    if dst_ply.exists():
        print(f"→ Eski {dst_ply} temizleniyor")
        shutil.rmtree(dst_ply)
    dst_ply.mkdir(parents=True, exist_ok=True)

    # Model build + export
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"→ Model yükleniyor ({device})")
    gs, deform, scene_extent = build_model_from_ckpt(path, device=device)

    print(f"→ {num_ts} frame PLY export ediliyor → {dst_ply}")
    files = export_to_ply(
        gs, deform, dst_ply,
        num_timestamps=num_ts,
        scene_extent=scene_extent,
        device=device,
    )

    print(f"\n✓ BAŞARILI")
    print(f"  Scene: {dst_scene}")
    print(f"  Kaynak: {label}.pt (from {scene})")
    print(f"  PLY sayısı: {len(files)}")
    print(f"  Viewer'da yükle: '{dst_scene}' sahne adıyla")


if __name__ == "__main__":
    main()

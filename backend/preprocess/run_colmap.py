"""Faz 2b — COLMAP Structure-from-Motion ile kamera pozları."""
from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path


def _resolve_colmap_exe(explicit: str | Path | None = None) -> str:
    """
    COLMAP exe yolunu şu sırayla bul:
      1) Açıkça verilen argüman (explicit)
      2) COLMAP_EXE ortam değişkeni
      3) PATH'teki 'colmap' / 'colmap.exe'
      4) Windows'ta tipik kurulum yolları
    """
    # 1) Açıkça verildi mi?
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Verilen COLMAP yolu mevcut değil: {p}")

    # 2) Ortam değişkeni
    env_path = os.environ.get("COLMAP_EXE")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return str(p)
        print(f"⚠ COLMAP_EXE={env_path} ama dosya yok, PATH'e düşülüyor")

    # 3) PATH
    found = shutil.which("colmap") or shutil.which("colmap.exe")
    if found:
        return found

    # 4) Windows tipik kurulum yolları (son çare)
    candidates = [
        r"E:\colmap\bin\colmap.exe",
        r"C:\colmap\bin\colmap.exe",
        r"C:\Program Files\COLMAP\bin\colmap.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c

    raise RuntimeError(
        "colmap bulunamadı. Şu seçeneklerden biri:\n"
        "  1) 'colmap' PATH'te olsun (https://github.com/colmap/colmap/releases)\n"
        "  2) COLMAP_EXE ortam değişkenine tam yolu ver:\n"
        "     Windows:  setx COLMAP_EXE \"E:\\colmap\\bin\\colmap.exe\"\n"
        "     (yeni terminal açman gerekir)\n"
        "  3) run_colmap(..., colmap_exe='...') parametresi ile doğrudan geç"
    )


def run_colmap(
    frames_dir: str | Path,
    output_dir: str | Path,
    camera_model: str = "PINHOLE",
    use_gpu: bool = True,
    sequential: bool = True,
    colmap_exe: str | Path | None = None,
) -> Path:
    """
    COLMAP pipeline:
      1) feature_extractor (SIFT)
      2) sequential_matcher (ya da exhaustive_matcher)
      3) mapper (sparse reconstruction)
      4) model_converter → cameras.txt, images.txt, points3D.txt

    Args:
        frames_dir: PNG karelerin bulunduğu klasör
        output_dir: COLMAP çıktısının yazılacağı klasör
        camera_model: PINHOLE | SIMPLE_RADIAL | OPENCV
        use_gpu: SIFT için GPU kullan
        sequential: True → sequential_matcher (video kareleri ardışık),
                    False → exhaustive_matcher (yavaş ama daha sağlam)

    Returns:
        Sparse rekonstrüksiyon klasörü (sparse/0)
    """
    colmap = _resolve_colmap_exe(colmap_exe)
    print(f"→ COLMAP: {colmap}")

    frames_dir = Path(frames_dir)
    if not any(frames_dir.glob("*.png")):
        raise FileNotFoundError(f"{frames_dir} altında PNG bulunamadı (önce extract_frames çalıştır)")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "colmap.db"
    sparse = out / "sparse"
    sparse.mkdir(exist_ok=True)

    # COLMAP 4.x'te `--SiftExtraction.use_gpu` / `--SiftMatching.use_gpu`
    # argümanları kaldırıldı. CUDA ile derlenmişse GPU otomatik kullanılır.
    # CPU'ya zorlamak için `gpu_index = -1` geçilir.
    extra_feat: list[str] = []
    extra_match: list[str] = []
    if not use_gpu:
        extra_feat += ["--SiftExtraction.gpu_index", "-1"]
        extra_match += ["--SiftMatching.gpu_index", "-1"]

    # 1) Feature extraction
    print("→ COLMAP: feature_extractor")
    subprocess.run([
        colmap, "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(frames_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",
        *extra_feat,
    ], check=True)

    # 2) Matching
    matcher = "sequential_matcher" if sequential else "exhaustive_matcher"
    print(f"→ COLMAP: {matcher}")
    subprocess.run([
        colmap, matcher,
        "--database_path", str(db),
        *extra_match,
    ], check=True)

    # 3) Sparse reconstruction
    print("→ COLMAP: mapper")
    subprocess.run([
        colmap, "mapper",
        "--database_path", str(db),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse),
    ], check=True)

    # COLMAP genelde sparse/0 oluşturur; parçalı rekonstrüksiyonda sparse/1, sparse/2 ... olabilir
    all_subs = [p for p in sparse.iterdir() if p.is_dir()]
    if not all_subs:
        raise RuntimeError("COLMAP rekonstrüksiyonu başarısız (sparse/ boş)")

    # Rekonstrüksiyon büyüklüğünü toplam BIN boyutundan tahmin et ve ona göre sırala
    def _model_size(model_dir: Path) -> int:
        total = 0
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            f = model_dir / name
            if f.exists():
                total += f.stat().st_size
        return total

    sub_models = sorted(all_subs, key=_model_size, reverse=True)

    if len(sub_models) > 1:
        print(f"⚠ COLMAP {len(sub_models)} ayrı parça oluşturdu "
              "(tüm kareler tek bir modele bağlanamadı):")
        for m in sub_models:
            size_mb = _model_size(m) / (1024 * 1024)
            print(f"    {m.name}: {size_mb:.2f} MB (cameras+images+points3D)")
        print(f"  → En büyük parçayı kullanıyoruz: {sub_models[0].name}")
        print("  (İdeal olan: video daha yavaş/yumuşak hareket içersin ya da "
              "--exhaustive matcher denensin)")
    sparse_0 = sub_models[0]

    # 4) Binary → text format dönüşümü (parse_colmap için)
    print("→ COLMAP: model_converter (BIN → TXT)")
    subprocess.run([
        colmap, "model_converter",
        "--input_path", str(sparse_0),
        "--output_path", str(sparse_0),
        "--output_type", "TXT",
    ], check=True)

    print(f"✓ COLMAP tamamlandı → {sparse_0}")
    return sparse_0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="COLMAP SfM pipeline")
    p.add_argument("frames_dir", type=str)
    p.add_argument("output_dir", type=str)
    p.add_argument("--camera-model", default="PINHOLE")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--exhaustive", action="store_true",
                   help="Exhaustive matcher kullan (yavaş ama sağlam)")
    p.add_argument("--colmap-exe", default=None,
                   help="COLMAP exe tam yolu (PATH'te yoksa). "
                        "Alternatif: COLMAP_EXE ortam değişkeni.")
    args = p.parse_args()

    run_colmap(
        args.frames_dir, args.output_dir,
        camera_model=args.camera_model,
        use_gpu=not args.no_gpu,
        sequential=not args.exhaustive,
        colmap_exe=args.colmap_exe,
    )

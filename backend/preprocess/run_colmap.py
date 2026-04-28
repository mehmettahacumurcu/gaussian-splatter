"""Faz 2b — COLMAP Structure-from-Motion ile kamera pozları.

Progress hook'lu sürüm — her COLMAP subprocess'inin stdout'u line-by-line
parse edilip on_progress callback'e (0-1 fraction, message) yansıtılır.
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional


# on_progress imzası: (fraction_0_1, message_str) -> None
ColmapProgressCallback = Callable[[float, str], None]


# ---- Regex patterns (COLMAP v3.7+ log formatları) ---------------------------
_FEAT_PAT = re.compile(r"Processed file \[(\d+)/(\d+)\]")
_MATCH_SEQ_PAT = re.compile(r"Matching block \[(\d+)/(\d+)\]")
_MATCH_EX_PAT = re.compile(r"Matching block \[(\d+)/(\d+), (\d+)/(\d+)\]")
_MAP_REG_PAT = re.compile(r"num_reg_frames=(\d+)")
_MAP_BA_PAT = re.compile(r"Retriangulation and Global bundle adjustment")


def _resolve_colmap_exe(explicit: str | Path | None = None) -> str:
    """COLMAP exe yolunu çöz — eski mantık aynı."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Verilen COLMAP yolu mevcut değil: {p}")
    env_path = os.environ.get("COLMAP_EXE")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return str(p)
        print(f"⚠ COLMAP_EXE={env_path} ama dosya yok, PATH'e düşülüyor")
    found = shutil.which("colmap") or shutil.which("colmap.exe")
    if found:
        return found
    candidates = [
        r"E:\colmap\bin\colmap.exe",
        r"C:\colmap\bin\colmap.exe",
        r"C:\Program Files\COLMAP\bin\colmap.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise RuntimeError(
        "colmap bulunamadı. COLMAP_EXE env var'a tam yolu ver ya da "
        "PATH'e ekle (https://github.com/colmap/colmap/releases)."
    )


def _stream_subprocess(
    cmd: list[str],
    parser: Callable[[str], Optional[tuple[float, str]]],
    on_progress: ColmapProgressCallback | None,
    phase_start: float,
    phase_end: float,
    phase_name: str,
) -> None:
    """
    Subprocess'i başlat, stdout'u satır satır stream et.
    parser(line) → (frac_in_phase 0-1, msg) veya None.
    on_progress çağrılırken fraction [phase_start, phase_end] aralığına map'lenir.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    last_reported = -1.0
    assert proc.stdout is not None
    for line in proc.stdout:
        # Echo to our stdout too (kullanıcı backend log'unda görebilsin)
        print(line, end="", flush=True)
        if on_progress is None:
            continue
        try:
            parsed = parser(line)
        except Exception as e:  # parser hatası progress'i bozmasın
            print(f"[stream parser err] {e}")
            parsed = None
        if parsed is None:
            continue
        frac, msg = parsed
        frac = max(0.0, min(1.0, frac))
        overall = phase_start + frac * (phase_end - phase_start)
        # Aşırı callback gürültüsü olmasın — en az %0.5 değişimde rapor et
        if overall - last_reported >= 0.005:
            on_progress(overall, f"{phase_name}: {msg}")
            last_reported = overall
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_colmap(
    frames_dir: str | Path,
    output_dir: str | Path,
    camera_model: str = "PINHOLE",
    use_gpu: bool = True,
    sequential: bool = True,
    colmap_exe: str | Path | None = None,
    on_progress: ColmapProgressCallback | None = None,
    single_camera: str = "yes",                    # "yes" | "no" | "per_folder"
    extra_feat_args: list[str] | None = None,      # SIFT/ImageReader extra
    extra_match_args: list[str] | None = None,     # matcher extra
    extra_mapper_args: list[str] | None = None,    # mapper/BA extra
    glob_pattern: str = "*.png",                   # multi-folder icin "**/*.png"
) -> Path:
    """
    COLMAP pipeline:
      1) feature_extractor    (0.00 – 0.15 progress)
      2) sequential_matcher   (0.15 – 0.40)
      3) mapper               (0.40 – 0.95)
      4) model_converter TXT  (0.95 – 1.00)

    v5.0.1: Premium multi-view destegi:
      - single_camera="per_folder" → her cam folder'inin kendi K'si
      - extra_feat_args ile max_num_features, distortion params
      - extra_mapper_args ile BA refinement secenekleri
    """
    colmap = _resolve_colmap_exe(colmap_exe)
    print(f"→ COLMAP: {colmap}")

    frames_dir = Path(frames_dir)
    frames = sorted(frames_dir.glob(glob_pattern))
    if not frames:
        raise FileNotFoundError(f"{frames_dir} altında pattern '{glob_pattern}' icin dosya yok")
    n_frames = len(frames)
    print(f"  {n_frames} frame bulundu")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "colmap.db"
    sparse = out / "sparse"
    sparse.mkdir(exist_ok=True)

    extra_feat: list[str] = list(extra_feat_args or [])
    extra_match: list[str] = list(extra_match_args or [])
    extra_mapper: list[str] = list(extra_mapper_args or [])
    if not use_gpu:
        extra_feat += ["--SiftExtraction.gpu_index", "-1"]
        extra_match += ["--SiftMatching.gpu_index", "-1"]

    # single_camera modu: "yes" → tek K (default), "no" → her image kendi K'si,
    # "per_folder" → her subfolder kendi K'si (multi-cam ideal)
    if single_camera == "per_folder":
        cam_args = ["--ImageReader.single_camera_per_folder", "1"]
    elif single_camera == "no":
        cam_args = ["--ImageReader.single_camera", "0"]
    else:
        cam_args = ["--ImageReader.single_camera", "1"]

    # ---------------- 1) Feature extraction (0.00 → 0.15) ----------------
    print("→ COLMAP: feature_extractor")
    if on_progress:
        on_progress(0.0, "feature_extractor: başlıyor")

    def _feat_parser(line: str):
        m = _FEAT_PAT.search(line)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            return cur / max(tot, 1), f"SIFT {cur}/{tot}"
        return None

    _stream_subprocess(
        [colmap, "feature_extractor",
         "--database_path", str(db),
         "--image_path", str(frames_dir),
         "--ImageReader.camera_model", camera_model,
         *cam_args,
         *extra_feat],
        _feat_parser, on_progress, 0.0, 0.15, "feature_extractor",
    )

    # ---------------- 2) Matching (0.15 → 0.40) ----------------
    matcher = "sequential_matcher" if sequential else "exhaustive_matcher"
    print(f"→ COLMAP: {matcher}")

    def _match_parser(line: str):
        m = _MATCH_EX_PAT.search(line)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            return cur / max(tot, 1), f"match {cur}/{tot}"
        m = _MATCH_SEQ_PAT.search(line)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            return cur / max(tot, 1), f"match {cur}/{tot}"
        return None

    _stream_subprocess(
        [colmap, matcher,
         "--database_path", str(db),
         *extra_match],
        _match_parser, on_progress, 0.15, 0.40, matcher,
    )

    # ---------------- 3) Mapper (0.40 → 0.95) ----------------
    print("→ COLMAP: mapper")
    if on_progress:
        on_progress(0.40, "mapper: başlıyor")

    def _map_parser(line: str):
        m = _MAP_REG_PAT.search(line)
        if m:
            cur = int(m.group(1))
            # 0-1 arası frac_in_phase — %95'i cap, kalan 5% BA için
            return min(cur / max(n_frames, 1), 0.95), f"registered {cur}/{n_frames}"
        if _MAP_BA_PAT.search(line):
            # Bundle adjustment sırasında frac değişmez, ama mesaj değişir
            return None  # don't override fraction, let it sit
        return None

    _stream_subprocess(
        [colmap, "mapper",
         "--database_path", str(db),
         "--image_path", str(frames_dir),
         "--output_path", str(sparse),
         *extra_mapper],
        _map_parser, on_progress, 0.40, 0.95, "mapper",
    )

    # ---------------- Sub-model seçimi ----------------
    all_subs = [p for p in sparse.iterdir() if p.is_dir()]
    if not all_subs:
        raise RuntimeError("COLMAP rekonstrüksiyonu başarısız (sparse/ boş)")

    def _model_size(model_dir: Path) -> int:
        total = 0
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            f = model_dir / name
            if f.exists():
                total += f.stat().st_size
        return total

    sub_models = sorted(all_subs, key=_model_size, reverse=True)
    if len(sub_models) > 1:
        print(f"⚠ COLMAP {len(sub_models)} ayrı parça oluşturdu:")
        for m in sub_models:
            size_mb = _model_size(m) / (1024 * 1024)
            print(f"    {m.name}: {size_mb:.2f} MB")
        print(f"  → En büyük parçayı kullanıyoruz: {sub_models[0].name}")
    sparse_0 = sub_models[0]

    # ---------------- 4) Model converter (0.95 → 1.0) ----------------
    print("→ COLMAP: model_converter (BIN → TXT)")
    if on_progress:
        on_progress(0.97, "model_converter: BIN → TXT")
    subprocess.run([
        colmap, "model_converter",
        "--input_path", str(sparse_0),
        "--output_path", str(sparse_0),
        "--output_type", "TXT",
    ], check=True)
    if on_progress:
        on_progress(1.0, f"done: {sparse_0.name}")

    print(f"✓ COLMAP tamamlandı → {sparse_0}")
    return sparse_0


def run_mvs_dense_reconstruction(
    image_dir: str | Path,
    sparse_dir: str | Path,  # sparse/0/
    dense_dir: str | Path,
    colmap_exe: str | Path | None = None,
    on_progress: ColmapProgressCallback | None = None,
    geom_consistency: bool = True,
    max_image_size: int = 2000,  # downsample large images for MVS speed
) -> Path:
    """COLMAP MVS Dense Reconstruction — sparse → dense point cloud.

    Pipeline:
      1) image_undistorter   — undistort images, prepare for MVS
      2) patch_match_stereo  — per-image dense depth maps via PatchMatchMVS
      3) stereo_fusion       — fuse depth maps to dense point cloud

    Output: dense_dir/fused.ply with potentially 500k-2M+ dense points.
    Ideal for 4DGS init — orders of magnitude better than sparse SfM.

    Time cost: 30-90 min for ~500 images on RTX 3060 Ti.
    """
    colmap = _resolve_colmap_exe(colmap_exe)
    image_dir = Path(image_dir)
    sparse_dir = Path(sparse_dir)
    dense_dir = Path(dense_dir)
    dense_dir.mkdir(parents=True, exist_ok=True)

    # 1) image_undistorter (0.0 → 0.10)
    print(f"→ COLMAP MVS: image_undistorter")
    if on_progress:
        on_progress(0.0, "MVS image_undistorter")
    subprocess.run([
        colmap, "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ], check=True)
    if on_progress:
        on_progress(0.10, "MVS undistort done")

    # 2) patch_match_stereo (0.10 → 0.85) — per-image dense depth
    print(f"→ COLMAP MVS: patch_match_stereo (geom_consistency={geom_consistency})")
    if on_progress:
        on_progress(0.10, "MVS patch_match_stereo")
    pms_args = [
        colmap, "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "true" if geom_consistency else "false",
    ]
    subprocess.run(pms_args, check=True)
    if on_progress:
        on_progress(0.85, "MVS depth maps done")

    # 3) stereo_fusion (0.85 → 1.0) — fuse depth maps to point cloud
    print(f"→ COLMAP MVS: stereo_fusion")
    if on_progress:
        on_progress(0.85, "MVS stereo_fusion")
    fused_path = dense_dir / "fused.ply"
    subprocess.run([
        colmap, "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric" if geom_consistency else "photometric",
        "--output_path", str(fused_path),
    ], check=True)
    if on_progress:
        on_progress(1.0, f"MVS done: {fused_path.name}")

    print(f"✓ MVS dense reconstruction tamamlandı → {fused_path}")
    return fused_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="COLMAP SfM pipeline")
    p.add_argument("frames_dir", type=str)
    p.add_argument("output_dir", type=str)
    p.add_argument("--camera-model", default="PINHOLE")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--exhaustive", action="store_true",
                   help="Exhaustive matcher kullan (yavaş ama sağlam)")
    p.add_argument("--colmap-exe", default=None)
    args = p.parse_args()

    def _cli_progress(frac, msg):
        print(f"[{frac*100:5.1f}%] {msg}")

    run_colmap(
        args.frames_dir, args.output_dir,
        camera_model=args.camera_model,
        use_gpu=not args.no_gpu,
        sequential=not args.exhaustive,
        colmap_exe=args.colmap_exe,
        on_progress=_cli_progress,
    )

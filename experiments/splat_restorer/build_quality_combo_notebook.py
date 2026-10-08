"""Build colab/quality_combo_test.ipynb: capacity + best-combination gsplat runs, fly-through videos.

    python experiments/splat_restorer/build_quality_combo_notebook.py
"""
from __future__ import annotations

import json

from build_gsplat_testbed_notebook import FUSED_SSIM, GSPLAT_TAG, NERFVIEW, PYCOLMAP, sources
from build_notebook import PKG, code, md

OUT = PKG.parents[2] / "colab" / "quality_combo_test.ipynb"

CELLS = [
    md("""
# Kapasite + en iyi kombinasyon

Önceki testler her şeyi **tek tek** ve **1M splat** sınırıyla denedi; poz hatasız karelerde ~27,3 dB'de tıkandık.
Bu notebook kazananları **birleştirir** ve splat bütçesini büyütür:

| varyant | ne var |
|---|---|
| `ext_default` | poz onarımındaki **yeni SfM** (extreme) + gsplat varsayılanı: sadece SfM'in etkisi |
| `ext_mcmc_3m` | yeni SfM + MCMC, **3M splat** |
| `orig_mcmc_3m` | eski SfM + MCMC 3M: kapasitenin etkisini SfM'den ayırır |
| `ext_mcmc_1m` | yeni SfM + MCMC 1M |
| `ext_mcmc_5m` | yeni SfM + MCMC **5M** (büyük GPU) |
| `ext_mcmc_3m_60k` | yeni SfM + MCMC 3M, **60k adım** |
| `ext_mcmc_3m_bilagrid` | yeni SfM + MCMC 3M + kare başına pozlama telafisi (ölçüm telafisiz PLY ile) |

Daha önce gsplat test zemininde eğitilen `default` ve `mcmc` (eski SfM, 1M) yeniden eğitilmez; Drive'dan okunup
`ref_orig_*` satırları olarak tabloya girer.

**Adil ölçüm:** Saklanan fotoğraflar **dosya adıyla** sabittir (yeni SfM farklı sayıda kare kaydettiği için sıraya göre
bölmek farklı fotoğrafları karşılaştırırdı). Tablo, bütün SfM'lerin kaydettiği ortak saklanan fotoğraflar üzerinden hesaplanır.

**Fly-through videoları:** Her varyant için kameranın kendi yolundan, hafif yana kayarak 15 sn'lik bir video; sonunda en iyi
varyantlarla 2x2 karşılaştırma videosu. Floater ve titremeyi PSNR göstermez, asıl görsel karar bu videolarla verilir.

**Kullanım:** A100 80 GB / H100 / RTX PRO 6000 (5M varyantı için 40 GB+ önerilir), **High-RAM** → **Run all**.
Varyant başına ~15-45 dk; hepsi ~3-4 saat. Her varyant bitince Drive'a yazılır; kopma olursa Run all biten varyantları atlar.

**Yeni çekim için:** `DATASET_ZIP`'i yeni dataset'e çevir, `EXTRA_SFM`'i boş bırak (veya yeni SfM yolunu ver),
ayrı bir test yürüyüşü çektiysen onun dosya adı desenini `HOLDOUT_PATTERN`'e yaz (ör. `test_*`).
"""),
    code('''
DATASET_ZIP = "/content/drive/MyDrive/GaussianTests/inputs/IMG_5966_dataset.zip"  # @param {type:"string"}
EXTRA_SFM = "/content/drive/MyDrive/GaussianTests/pose_repair/IMG_5966/extreme_sift/sparse_selected"  # @param {type:"string"}
OUT_DIR = "/content/drive/MyDrive/GaussianTests/quality_combo/IMG_5966"  # @param {type:"string"}
REFERENCE_DIR = "/content/drive/MyDrive/GaussianTests/gsplat_testbed/IMG_5966"  # @param {type:"string"}
VARIANTS = "ext_default,ext_mcmc_3m,orig_mcmc_3m,ext_mcmc_1m,ext_mcmc_5m,ext_mcmc_3m_60k,ext_mcmc_3m_bilagrid"  # @param {type:"string"}
HOLDOUT_EVERY = 10     # @param {type:"integer"}
HOLDOUT_PATTERN = ""   # @param {type:"string"}
FLYTHROUGH = True      # @param {type:"boolean"}
FLY_FRAMES = 450       # @param {type:"integer"}
STEPS_SCALE = 1.0      # @param {type:"number"}
FACTOR_OVERRIDE = 0    # @param {type:"integer"}
# EXTRA_SFM: dataset zip'indekinden farklı ikinci bir SfM modeli (cameras/images/points3D.bin klasörü) → "extreme".
#   Boşsa sadece dataset'teki SfM ("original") kullanılır ve 'ext_*' varyantları atlanır.
# HOLDOUT_PATTERN: ayrı test yürüyüşü karelerinin dosya adı deseni (glob); her N. kareye ek olarak saklanır.
# STEPS_SCALE / FACTOR_OVERRIDE: hızlı deneme için (ör. 0.1 ve 4). REFERENCE_DIR: önceki gsplat sonuçları (yoksa boş bırak).
VARIANTS = [v.strip() for v in VARIANTS.split(",") if v.strip()]
''', title="Ayarlar"),
    code(f'''
import os, subprocess, sys
os.environ["MAX_JOBS"] = "4"
if not os.path.exists("/content/gsplat_src"):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", "{GSPLAT_TAG}",
                    "https://github.com/nerfstudio-project/gsplat.git", "/content/gsplat_src"], check=True)
# requirements.txt'teki numpy<2 pini kullanılmıyor (Colab'da kernel yeniden başlatma ister).
!pip install -q "gsplat=={GSPLAT_TAG[1:]}" ninja tyro viser "imageio[ffmpeg]" scikit-learn tqdm "torchmetrics[image]" tensorboard tensorly pyyaml matplotlib splines lpips "{PYCOLMAP}" "{NERFVIEW}"
# fused-ssim derlenirken torch'u görmeli: build isolation kapalı.
!pip install -q --no-build-isolation "{FUSED_SSIM}"
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
''', title="Kurulum (gsplat eğiticisi + bağımlılıklar, ~5-10 dk)"),
    code(f'''
import json
from pathlib import Path

SOURCES = json.loads({sources()!r})
root = Path("/content/tb_src")
for name, text in SOURCES.items():
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
if str(root / "lib") not in sys.path:
    sys.path.insert(0, str(root / "lib"))
from splat_restorer import own_splat, train_ab, pose_repair, gsplat_testbed as tb, quality_combo as qc
# Colab uyumluluğu: HF datasets paketiyle isim çakışması + pycolmap fork'unun numpy-2 hatası.
tb.fix_gsplat_install(Path("/content/gsplat_src/examples"))
print(f"{{len(SOURCES)}} dosya yazıldı")
''', title="Kodu yaz (gömülü kaynaklar)"),
    code('''
import shutil, zipfile
from google.colab import drive
drive.mount("/content/drive")
out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
workers = os.cpu_count() or 4

raw = Path("/content/own_dataset")
if not (raw / ".unzipped").exists():
    print("Zip açılıyor...")
    shutil.rmtree(raw, ignore_errors=True)
    with zipfile.ZipFile(DATASET_ZIP) as z:
        z.extractall(raw)
    (raw / ".unzipped").touch()
cams_bin = next(raw.rglob("sparse/*/cameras.bin"))
src = cams_bin.parents[2]
sfm_dirs = {"original": cams_bin.parent}
if EXTRA_SFM:
    assert Path(EXTRA_SFM, "images.bin").exists(), f"EXTRA_SFM içinde images.bin yok: {EXTRA_SFM}"
    sfm_dirs["extreme"] = Path(EXTRA_SFM)
VARIANTS = [v for v in VARIANTS if qc.VARIANTS[v].sfm in sfm_dirs]

factors = sorted({FACTOR_OVERRIDE or qc.VARIANTS[v].gs.factor for v in VARIANTS})
for f in factors:
    tb.prepare_factor_images(src, f, workers=workers)   # bir kez; her SfM klasörüne bağlanır
datasets, data = {}, {}
for key, sdir in sfm_dirs.items():
    ds = pose_repair.make_dataset(src / "images", sdir, Path(f"/content/combo_ds_{key}"))
    for f in factors:
        link = ds / f"images_{f}"
        if f > 1 and not link.exists():
            try:
                link.symlink_to((src / f"images_{f}").resolve(), target_is_directory=True)
            except OSError:
                shutil.copytree(src / f"images_{f}", link)
    datasets[key] = ds
    # Ölçüm fotoğrafları: her SfM kendi lens düzeltmesiyle, önceki testlerle aynı hazırlık (1920 uzun kenar).
    data[key] = own_splat.prepare_dataset(ds, Path(f"/content/combo_prep_{key}"), long_edge=1920,
                                          model_name="0", workers=workers)

names = qc.holdout_names(sorted(data["original"].names), HOLDOUT_EVERY, HOLDOUT_PATTERN)
per_sfm, missing, holdout = {}, {}, {}
for key, d in data.items():
    holdout[key], missing[key] = pose_repair.holdout_indices(d, names)
    per_sfm[key] = [d.names[i] for i in holdout[key]]
common = qc.common_names(per_sfm)
bad_names = []
flags_path = Path(REFERENCE_DIR, "pose_flags.json") if REFERENCE_DIR else None
if flags_path and flags_path.exists():
    flagged = json.loads(flags_path.read_text())["flagged_holdout"]
    bad_names = [data["original"].names[i] for i in flagged]
for key in data:
    print(f"{key}: {data[key].num_frames} kayıtlı kare, saklanan {len(holdout[key])}/{len(names)}"
          + (f", kaydedilemeyen {len(missing[key])}" if missing[key] else ""))
print(f"Ortak saklanan fotoğraf: {len(common)}; eski SfM'de çöken (ikinci ortalamada hariç): {len(bad_names)}")
# Fly-through yolu: bütün SfM'lerin kaydettiği kareler, video sırasıyla (isim sırası).
fly_names = qc.common_names({k: list(d.names) for k, d in data.items()})
''', title="Drive + veri (+ SfM'ler, isimle sabit saklanan fotoğraflar)"),
    code('''
import dataclasses, lpips, torch
assert torch.cuda.is_available(), "GPU yok: Runtime → Change runtime type → GPU"
net_lpips = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
results, failed, fly_dirs = {}, [], {}

def by_name(frames: dict, d) -> dict:
    return {d.names[int(i)]: v for i, v in frames.items()}

def flythrough(name, params, deg, d, v=None):
    if not FLYTHROUGH:
        return
    mp4 = out_dir / f"flythrough_{name}.mp4"
    frames_dir = Path("/content/fly") / name
    poses = qc.flythrough_path(d, fly_names, n_frames=FLY_FRAMES)
    kw = dict(v.gs.render) if v else {}
    qc.render_flythrough(params, deg, d, poses, frames_dir, rasterize_mode=v.gs.rasterize_mode if v else "classic", **kw)
    fly_dirs[name] = frames_dir
    if not mp4.exists():
        qc.write_video({name: frames_dir}, mp4)
    print(f"  fly-through: {mp4.name}")

# Önceki gsplat test zemininden referanslar (eski SfM, 1M): yeniden eğitilmez.
for ref, rname in (("default", "ref_orig_default"), ("mcmc", "ref_orig_mcmc_1m")):
    rp = Path(REFERENCE_DIR, ref, "result.json") if REFERENCE_DIR else None
    if rp and rp.exists():
        r = json.loads(rp.read_text())
        results[rname] = {"sfm": "original", "note": f"gsplat test zemini '{ref}' (yeniden eğitilmedi)",
                          "info": r.get("info", {}), "frames": by_name(r["frames"], data["original"])}
        ply = Path(REFERENCE_DIR, ref, "final.ply")
        if FLYTHROUGH and ply.exists():   # kareler ve video varsa yeniden render edilmez
            params, deg = own_splat.load_ply(ply, "cuda")
            flythrough(rname, params, deg, data["original"])
            del params; torch.cuda.empty_cache()

for name in VARIANTS:
    v = qc.VARIANTS[name]
    vdir = out_dir / name
    res_path = vdir / "result.json"
    d = data[v.sfm]
    if res_path.exists():
        results[name] = json.loads(res_path.read_text())
        if FLYTHROUGH and (vdir / "final.ply").exists():
            params, deg = own_splat.load_ply(vdir / "final.ply", "cuda")
            flythrough(name, params, deg, d, v)
            del params; torch.cuda.empty_cache()
        print(f"{name}: daha önce bitmiş, atlandı")
        continue
    gs = dataclasses.replace(v.gs, factor=FACTOR_OVERRIDE) if FACTOR_OVERRIDE else v.gs
    steps = max(100, int(v.steps * STEPS_SCALE))
    print(f"\\n===== {name}: {v.note} ({steps:,} adım)", flush=True)
    run_dir = Path("/content/gs_runs") / name
    shutil.rmtree(run_dir, ignore_errors=True)
    exclude = [d.names[i] for i in holdout[v.sfm]]   # saklanan fotoğraflar hiçbir varyantta eğitime girmez
    try:
        # test_every çok büyük: gsplat'in kendi bölmesi devre dışı, saklama isimle yapılır.
        seconds = tb.run_variant(gs, datasets[v.sfm], Path("/content/gsplat_src/examples"), run_dir, steps,
                                 1_000_000, exclude)
    except RuntimeError as e:   # bir varyant çökerse diğerleri yine koşsun
        print(f"!!! {name} BAŞARISIZ, atlanıyor:\\n{e}")
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "failed.txt").write_text(str(e), encoding="utf-8")
        if (run_dir / "train.log").exists():   # tam log: çöküşten önceki relocate/add sayıları
            shutil.copy2(run_dir / "train.log", vdir / "train.log")
        failed.append(name)
        continue
    params, deg = tb.load_checkpoint(run_dir)
    vdir.mkdir(parents=True, exist_ok=True)
    frames = train_ab.evaluate_params(params, deg, d, holdout[v.sfm], None, net_lpips, vdir / "renders",
                                      rasterize_mode=gs.rasterize_mode, **gs.render)
    tb.save_ply(params, vdir / "final.ply")
    shutil.copy2(run_dir / "train.log", vdir / "train.log")
    info = {"variant": qc.variant_dict(v), "train_seconds": seconds, "steps": steps,
            "splats": int(params["means"].shape[0]), "excluded_train_images": len(exclude)}
    results[name] = {"sfm": v.sfm, "note": v.note, "info": info, "frames": by_name(frames, d)}
    tb.save_json(res_path, results[name])
    flythrough(name, params, deg, d, v)
    del params
    torch.cuda.empty_cache()
    m = qc.summarise({name: results[name]}, common, bad_names)[name]
    print(f"{name}: PSNR {m['psnr']:.2f}  LPIPS {m['lpips']:.4f}  kötü-hariç {m['psnr_ok']:.2f}/{m['lpips_ok']:.4f}  "
          f"{m['splats']:,} splat  {m['minutes']:.0f} dk")
''', title="Eğit + ölç + fly-through (varyant başına ~15-45 dk)"),
    code('''
import base64
from IPython.display import HTML, Markdown, display
if failed:
    print("Başarısız varyantlar (ayrıntı Drive'da <varyant>/failed.txt):", failed)
summary = qc.summarise(results, common, bad_names)
md_text = qc.markdown(summary, len(common), bad_names, missing)
(out_dir / "summary.md").write_text(md_text, encoding="utf-8")
qc.save_json(out_dir / "summary.json", summary)
display(Markdown(md_text))
if FLYTHROUGH and fly_dirs:
    ranked = sorted((n for n in summary if n in fly_dirs and not n.startswith("ref_")), key=lambda n: summary[n]["lpips_ok"])
    grid = (["ref_orig_default"] if "ref_orig_default" in fly_dirs else []) + ranked[:3]
    grid_mp4 = qc.write_video({n: fly_dirs[n] for n in grid}, out_dir / "flythrough_compare.mp4")
    print("Karşılaştırma videosu:", grid_mp4, "→", grid)
    if grid_mp4.stat().st_size < 40 * 2**20:
        b64 = base64.b64encode(grid_mp4.read_bytes()).decode()
        display(HTML(f'<video controls loop width="1400" src="data:video/mp4;base64,{b64}"></video>'))
''', title="Sonuçlar + karşılaştırma videosu"),
    md("""
## Nasıl okunur?

* **Δ** `ref_orig_default` (eski SfM, gsplat varsayılanı, 1M) satırına göre. PSNR'da ~0,3 dB, LPIPS'te ~0,01 altındaki farklar
  koşudan koşuya oynayabilir.
* `ext_default` vs `ref_orig_default` → **SfM'in** etkisi; `orig_mcmc_3m` vs `ref_orig_mcmc_1m` → **kapasitenin** etkisi;
  `ext_mcmc_3m` ikisi birlikte; `_5m`, `_60k`, `_bilagrid` üstüne eklenenler.
* Asıl görsel karar: `flythrough_compare.mp4` (sol üst eski referans). Floater, titreme ve bulanıklığa bak.

Bana `summary.md` ve `flythrough_compare.mp4` yeterli (Drive'da `OUT_DIR` altında).
"""),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

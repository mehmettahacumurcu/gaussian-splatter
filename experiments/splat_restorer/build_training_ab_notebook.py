"""Build colab/training_ab_test.ipynb: A/B training variants with the app's trainer.

The app's trainer files are read from git (APP_REF), not from the working
tree, so the notebook trains exactly what the app ships:

    python experiments/splat_restorer/build_training_ab_notebook.py
"""
from __future__ import annotations

import json
import subprocess

from build_notebook import PKG, code, md

REPO = PKG.parents[2]
OUT = REPO / "colab" / "training_ab_test.ipynb"
APP_REF = "feature/scene-composer"
APP_FILES = [
    "backend/model/gaussian_model.py", "backend/model/deformation.py", "backend/model/renderer.py",
    "backend/model/density_control.py", "backend/model/trainer.py", "backend/preprocess/parse_colmap.py",
    "backend/export/to_splat.py",
]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout


def sources() -> tuple[str, str]:
    files = {f"app/{p}": git("show", f"{APP_REF}:{p}") for p in APP_FILES}
    for pkg in ("backend", "backend/model", "backend/preprocess", "backend/export"):
        files[f"app/{pkg}/__init__.py"] = ""
    for p in sorted(PKG.iterdir()):
        if p.suffix in (".py", ".txt"):
            files[f"lib/splat_restorer/{p.name}"] = p.read_text(encoding="utf-8")
    return json.dumps(files, ensure_ascii=False), git("rev-parse", "--short", APP_REF).strip()


SOURCES, APP_SHA = sources()

CELLS = [
    md(f"""
# Eğitim A/B testi: aynı sahne, aynı eğitici, tek tek değişiklik

App'teki **Spirula dataset + bizim trainer** yolunun eğiticisi (`{APP_REF}` @ `{APP_SHA}`) birkaç varyantla eğitilir;
hepsi aynı kareler, aynı tohum ve **aynı saklanan fotoğraflarla** (her 10. kare, eğitimde hiç kullanılmaz) ölçülür.

| varyant | değişiklik |
|---|---|
| `baseline` | 28 Eylül IMG_5966 eğitimiyle aynı ayarlar (doğrulama: ~25,3 dB beklenir) |
| `lr_decay` | Gaussian konumlarının öğrenme oranı eğitim boyunca 100 kat azalır (standart 3DGS; eğiticimizde sabit) |
| `lr_decay_clean` | + eğitim kareleri 4K'dan 2x2 ortalamayla küçültülür ve hafifçe gürültüden arındırılır |
| `lr_decay_60k` | + 60k adım |

**Kullanım:** GPU (A100 önerilir, **High-RAM** açık) → **Run all**. Varyant başına ~10-30 dk.
Her varyant bitince sonucu Drive'a yazılır; kopma olursa Run all biten varyantları atlar.
"""),
    code('''
DATASET_ZIP = "/content/drive/MyDrive/GaussianTests/inputs/IMG_5966_dataset.zip"  # @param {type:"string"}
OUT_DIR = "/content/drive/MyDrive/GaussianTests/training_ab/IMG_5966"  # @param {type:"string"}
VARIANTS = "baseline,lr_decay,lr_decay_clean,lr_decay_60k"  # @param {type:"string"}
HOLDOUT_EVERY = 10   # @param {type:"integer"}
LONG_EDGE = 1920     # @param {type:"integer"}
NLM_H = 3.0          # @param {type:"number"}
STEPS_OVERRIDE = 0   # @param {type:"integer"}
SEED = 42
# STEPS_OVERRIDE > 0: hızlı deneme için her varyantın adım sayısı (60k varyantı bunun 2 katı).
# NLM_H: gürültü temizleme gücü (0 = sadece 2x2 ortalama).
VARIANTS = [v.strip() for v in VARIANTS.split(",") if v.strip()]
''', title="Ayarlar"),
    code('''
import os
os.environ["MAX_JOBS"] = "4"
!pip install -q "gsplat==1.5.3" "pytorch-msssim==1.0.0" plyfile ninja lpips psutil
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
''', title="Kurulum"),
    code(f'''
import json, sys, types
from pathlib import Path

SOURCES = json.loads({SOURCES!r})
root = Path("/content/ab_src")
for name, text in SOURCES.items():
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
for sub in ("app", "lib"):
    if str(root / sub) not in sys.path:
        sys.path.insert(0, str(root / sub))

from backend.model.gaussian_model import GaussianModel
from backend.model.deformation import DeformationField
from backend.model.trainer import Trainer4DGS
from backend.model.renderer import render_view
from backend.export.to_splat import export_to_ply
from backend.preprocess.parse_colmap import scene_extent
from splat_restorer import own_splat, train_ab
app = types.SimpleNamespace(GaussianModel=GaussianModel, DeformationField=DeformationField, Trainer4DGS=Trainer4DGS,
                            export_to_ply=export_to_ply, scene_extent=scene_extent)
print(f"{{len(SOURCES)}} dosya yazıldı; app eğiticisi {APP_REF} @ {APP_SHA}")
''', title="Kodu yaz (app eğiticisi + test kodu)"),
    code('''
import shutil, zipfile, psutil
from google.colab import drive
drive.mount("/content/drive")
out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

raw = Path("/content/own_dataset")
if not (raw / ".unzipped").exists():
    print("Zip açılıyor...")
    shutil.rmtree(raw, ignore_errors=True)
    with zipfile.ZipFile(DATASET_ZIP) as z:
        z.extractall(raw)
    (raw / ".unzipped").touch()
cams_bin = next(raw.rglob("sparse/*/cameras.bin"))
src, model_name = cams_bin.parents[2], cams_bin.parent.name
data = own_splat.prepare_dataset(src, Path(f"/content/own_prepared_{LONG_EDGE}"), long_edge=LONG_EDGE,
                                 model_name=model_name, workers=os.cpu_count() or 4)
clean_dir = train_ab.clean_frames(src, data, Path(f"/content/own_clean_{LONG_EDGE}_h{NLM_H:g}"), model_name, nlm_h=NLM_H,
                                  workers=os.cpu_count() or 4)
holdout = list(range(0, data.num_frames, HOLDOUT_EVERY))
xyz, rgb, init_ids = train_ab.init_points(src, model_name, SEED)
need = data.num_frames * data.width * data.height * 3 / 2**30
print(f"{data.num_frames} kare {data.width}x{data.height}, saklanan {len(holdout)}; {len(xyz):,} nokta")
print(f"Eğitim RAM önbelleği ~{need:.1f} GiB; boş RAM {psutil.virtual_memory().available / 2**30:.1f} GiB")
assert need < psutil.virtual_memory().available / 2**30 * 0.6, "RAM yetersiz: Runtime → Change runtime type → High-RAM"
''', title="Drive + veri (+ temiz kareler)"),
    code('''
import time, torch
assert torch.cuda.is_available(), "GPU yok: Runtime → Change runtime type → GPU"
t0 = time.time()
m = torch.tensor([[0., 0., 2.]], device="cuda", requires_grad=True)
r, _, _ = render_view(m, torch.tensor([[1., 0, 0, 0]], device="cuda"), torch.full((1, 3), .1, device="cuda"),
                      torch.tensor([.5], device="cuda"), torch.full((1, 3), .5, device="cuda"),
                      torch.tensor([[50., 0, 32], [0, 50., 32], [0, 0, 1.]], device="cuda"), torch.eye(4, device="cuda"),
                      64, 64, sh_degree=0)
r.sum().backward()
assert m.grad is not None and torch.isfinite(m.grad).all()
print(f"GPU render/backward OK ({time.time() - t0:.0f} s, ilk derleme dahil)")
''', title="GPU kontrolü (ilk seferde gsplat derlenir, ~3-5 dk)"),
    code('''
import dataclasses, lpips
from IPython.display import Markdown, display
net_lpips = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
results = {}
for name in VARIANTS:
    res_path = out_dir / name / "result.json"
    if res_path.exists():
        results[name] = json.loads(res_path.read_text())
        print(f"{name}: daha önce bitmiş, atlandı")
        continue
    v = train_ab.VARIANTS[name]
    if STEPS_OVERRIDE > 0:
        v = dataclasses.replace(v, steps=STEPS_OVERRIDE * (2 if v.steps > 30000 else 1))
    print(f"\\n===== {name}: {v.note} ({v.steps:,} adım)")
    work = Path("/content/ab_work") / name
    shutil.rmtree(work, ignore_errors=True)
    frames_dir = clean_dir if v.clean else data.frames_dir
    ply, info = train_ab.train_variant(v, data, frames_dir, xyz, rgb, init_ids, holdout, work, app, seed=SEED)
    (out_dir / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply, out_dir / name / "final.ply")
    frames = train_ab.evaluate_ply(ply, data, holdout, clean_dir, net_lpips, out_dir / name / "renders")
    results[name] = {"info": info, "frames": frames}
    train_ab.save_json(res_path, results[name])
    s = train_ab.summarise({name: results[name]})[name]
    print(f"{name}: PSNR {s['psnr']:.2f}  LPIPS {s['lpips']:.4f}  temiz-foto PSNR {s['clean_psnr']:.2f}  "
          f"{s['splats']:,} splat  {s['minutes']:.0f} dk")
''', title="Eğit + ölç (varyant başına ~10-30 dk)"),
    code('''
from IPython.display import Image as IPImage
summary = train_ab.summarise(results)
md_text = train_ab.markdown(summary, len(holdout))
(out_dir / "summary.md").write_text(md_text, encoding="utf-8")
train_ab.save_json(out_dir / "summary.json", summary)
display(Markdown(md_text))
sheets = train_ab.comparison_sheets(results, data, out_dir, out_dir)
print("Sütunlar: foto | " + " | ".join(results) + "; alt sıra: kırmızı kutunun zoom'u")
for p in sheets[:6]:
    display(IPImage(filename=str(p), width=1600))
''', title="Sonuçlar"),
    md("""
## Nasıl okunur?

* **PSNR / LPIPS** saklanan fotoğraflara karşıdır; **Δ** baseline'a göre farktır. +0,3 dB ve üstü belirgin bir iyileşmedir.
* **Temiz foto PSNR:** fotoğrafların gürültüsü azaltılmış hâline karşı. `lr_decay_clean` orijinal fotoğrafa karşı
  gürültüyü taklit etmediği için orada geride görünebilir; bu sütunda adil karşılaştırılır.
* `baseline` ~25,3 dB değilse kurulumda bir fark var demektir; diğer sonuçlara güvenmeden önce bana haber ver.

Bana `summary.md` ve birkaç `compare_*.jpg` getirmen yeterli (Drive'da `OUT_DIR` altında).
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
            "app_trainer": {"ref": APP_REF, "commit": APP_SHA},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB), app trainer {APP_REF} @ {APP_SHA}")


if __name__ == "__main__":
    main()

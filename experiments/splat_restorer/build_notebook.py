"""Build colab/splat_restorer_pilot.ipynb with the package sources embedded.

Run after editing anything under splat_restorer/:

    python experiments/splat_restorer/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE / "splat_restorer"
OUT = HERE.parents[1] / "colab" / "splat_restorer_pilot.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str, title: str | None = None) -> dict:
    meta = {"cellView": "form"} if title else {}
    src = (f"# @title {title}\n" if title else "") + text.strip("\n")
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


def embedded_sources() -> str:
    files = {p.name: p.read_text(encoding="utf-8") for p in sorted(PKG.iterdir()) if p.suffix in (".py", ".txt")}
    return json.dumps(files, ensure_ascii=False)


CELLS = [
    md("""
# Splat Restorer — pilot

Bu notebook, splat render'larındaki hataları (floater, bulanıklık, uzak açı bozulmaları) temizleyen
tek adımlı bir diffusion modelinin (Difix3D mimarisi, SD-Turbo tabanlı) **işe yarayıp yaramadığını** test eder.

**Akış**
1. **Sanity:** Tek sahnede kısa eğitim yapılır; model girdisini geçemiyorsa kodda sorun var demektir.
2. **Çiftler:** DL3DV sahnelerinde bilerek zayıf splat'ler eğitilir; render'lar gerçek fotoğraflarla eşlenir.
3. **Eğitim:** 10 / 25 / 50 sahneyle restorer (ölçekleme eğrisi) + referanssız ablation + küçük CNN baseline.
4. **Görüntü değerlendirmesi:** Eğitimde **hiç görülmemiş** test sahnelerinde PSNR / SSIM / LPIPS ölçülür.
5. **Splat'e geri besleme:** Düzeltilmiş render'larla splat ince ayarlanır; kör karelerde splat'in kendisi iyileşiyor mu bakılır.

**Gerekenler**
* GPU runtime (A100 / H100 / daha büyüğü). *Runtime → Change runtime type.*
* Colab **Secrets** (soldaki 🔑) içinde `HF_TOKEN` adında, DL3DV erişimi olan Hugging Face token'ı; "Notebook access" açık olmalı.
* Google Drive'da ~30 GB boş alan.

**Kopma / yeniden başlatma:** Her aşama kaldığı yerden devam eder. Oturum koparsa notebook'u
baştan **Run all** ile çalıştır; Drive'da biten işler atlanır, eğitimler son checkpoint'ten devam eder.

Lisans notu: model kodu NVIDIA Difix3D'den uyarlanmıştır (yalnızca ticari olmayan araştırma kullanımı).
DL3DV verisi kendi kullanım koşullarına tabidir.
"""),
    code('''
DRIVE_DIR = "splat_restorer"        # @param {type:"string"}
N_TEST_SCENES = 12                  # @param {type:"integer"}
N_TRAIN_SCENES = 50                 # @param {type:"integer"}
SCALING = "10,25,50"                # @param {type:"string"}
RECIPES = "sparse_hard,sparse_mild" # @param {type:"string"}
MAX_PAIRS_PER_RECIPE = 100          # @param {type:"integer"}
RESTORER_STEPS = 10000              # @param {type:"integer"}
RESTORER_BATCH = 4                  # @param {type:"integer"}
RUN_NOREF_ABLATION = True           # @param {type:"boolean"}
RUN_CNN_BASELINE = True             # @param {type:"boolean"}
PAIRS_TIME_BUDGET_H = 0             # @param {type:"number"}
# PAIRS_TIME_BUDGET_H > 0 stops pair generation after that many hours (re-run to continue).

SCALING = [int(x) for x in SCALING.split(",") if x.strip()]
RECIPES = [x.strip() for x in RECIPES.split(",") if x.strip()]
assert max(SCALING) <= N_TRAIN_SCENES, "SCALING içindeki en büyük sayı N_TRAIN_SCENES'i geçemez"
''', title="Ayarlar"),
    code('''
!pip install -q "gsplat==1.5.3" ninja "diffusers==0.32.2" "transformers==4.47.1" "peft==0.14.0" "accelerate==1.2.1" lpips
''', title="Kurulum (sabitlenmiş sürümler)"),
    code(f'''
import json, os, sys
from pathlib import Path

SOURCES = json.loads({embedded_sources()!r})
pkg = Path("/content/splat_restorer_src/splat_restorer")
pkg.mkdir(parents=True, exist_ok=True)
for name, text in SOURCES.items():
    (pkg / name).write_text(text, encoding="utf-8")
if "/content/splat_restorer_src" not in sys.path:
    sys.path.insert(0, "/content/splat_restorer_src")
print(f"{{len(SOURCES)}} files written to {{pkg}}")
''', title="Kodu yaz (gömülü kaynaklar)"),
    code('''
from google.colab import drive, userdata
drive.mount("/content/drive")
token = userdata.get("HF_TOKEN")
assert token, "Colab Secrets içinde HF_TOKEN bulunamadı (soldaki 🔑 menüsü)."
os.environ["HF_TOKEN"] = token
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from splat_restorer.workspace import Workspace
from splat_restorer import stages
ws = Workspace(root=Path("/content/drive/MyDrive") / DRIVE_DIR, local=Path("/content/sr_work"))
print("Drive klasörü:", ws.root)
info = stages.preflight(ws)   # ilk seferde gsplat CUDA derlemesi ~2-5 dk sürebilir
split = stages.prepare_split(ws, n_test=N_TEST_SCENES)
''', title="Drive, token, ön kontrol, sahne ayrımı"),
    md("""
## 1. Sanity — tek sahnede kısa eğitim
Uzun koşulardan önce kodun uçtan uca çalıştığını doğrular (~15-25 dk: sahne indirme + 2 zayıf splat + 300 adım eğitim).
`SANITY PASSED` görmelisin. Görmezsen devam etme, çıktıyı bana getir.
"""),
    code('''
from IPython.display import Image as IPImage, display
sanity_scene = split["train_pool"][0]
stages.generate_one(ws, sanity_scene, RECIPES, is_test=False, max_pairs_per_recipe=MAX_PAIRS_PER_RECIPE)
ok = stages.sanity_overfit(ws, sanity_scene, steps=300)
samples = sorted((ws.local / "sanity" / "runs" / "sanity" / "samples").glob("*.jpg"))
if samples:
    print("Satırlar: girdi | model | gerçek | referans")
    display(IPImage(filename=str(samples[-1]), width=1200))
''', title="Sanity"),
    md("""
## 2. Çift üretimi
Önce test sahneleri, sonra eğitim sahneleri (hedef `N_TRAIN_SCENES`). Sahne başına birkaç dakika; 60+ sahne için
saatler sürebilir. Bozuk sahneler `failed_scenes.json`'a yazılır ve atlanır. Kopma olursa tekrar çalıştır.
"""),
    code('''
stages.generate_pairs(ws, split, n_train=N_TRAIN_SCENES, recipes=RECIPES,
                      max_pairs_per_recipe=MAX_PAIRS_PER_RECIPE,
                      time_budget_h=PAIRS_TIME_BUDGET_H or None)
test_scenes = stages.done_scenes(ws, split["test"])
train_all = stages.train_scenes(ws, split, N_TRAIN_SCENES)
print(f"hazır: {len(test_scenes)} test, {len(train_all)} eğitim sahnesi")
''', title="Çiftleri üret"),
    md("""
## 3. Eğitimler
Her koşu kaldığı yerden devam eder; biten koşular atlanır. `runs/<koşu>/samples/` altında ara örnekler,
`log.csv` içinde kayıp ve doğrulama değerleri var. İlk log satırlarındaki `s/step` ile toplam süreyi tahmin edebilirsin.
"""),
    code('''
from splat_restorer.train_restorer import RestorerConfig, train_restorer
from splat_restorer.baseline import BaselineConfig, train_baseline

assert len(train_all) >= max(SCALING), f"Sadece {len(train_all)} eğitim sahnesi hazır; önce çiftleri tamamla."
pairs_root = stages.extract(ws, test_scenes + train_all)

restorer_runs, baseline_runs = [], []
for n in SCALING:
    name = f"restorer_ref_n{n}"
    train_restorer(ws, RestorerConfig(run_name=name, scenes=train_all[:n], use_ref=True,
                                      steps=RESTORER_STEPS, batch_size=RESTORER_BATCH), pairs_root, log=stages.log)
    restorer_runs.append(name)
N = max(SCALING)
if RUN_NOREF_ABLATION:
    name = f"restorer_noref_n{N}"
    train_restorer(ws, RestorerConfig(run_name=name, scenes=train_all[:N], use_ref=False,
                                      steps=RESTORER_STEPS, batch_size=RESTORER_BATCH), pairs_root, log=stages.log)
    restorer_runs.append(name)
if RUN_CNN_BASELINE:
    for use_ref in (True, False):
        name = f"cnn_{'ref' if use_ref else 'noref'}_n{N}"
        train_baseline(ws, BaselineConfig(run_name=name, scenes=train_all[:N], use_ref=use_ref), pairs_root, log=stages.log)
        baseline_runs.append(name)
print("restorer:", restorer_runs, "| baseline:", baseline_runs)
''', title="Restorer ve baseline eğitimleri"),
    md("""
## 4. Görüntü değerlendirmesi (görülmemiş test sahneleri)
Okuma: bir yöntem ancak **LPIPS'i düşürürken PSNR'ı belirgin düşürmüyorsa** gerçekten düzeltiyordur.
Ölçekleme: `restorer_ref_n10 → n25 → n50` sırasıyla iyileşiyorsa daha fazla sahneye değer.
`ref` ile `noref` farkı referans görüntünün katkısını; `cnn` satırları diffusion'ın gerekip gerekmediğini gösterir.
"""),
    code('''
from splat_restorer.evaluate import build_methods, evaluate
methods = build_methods(ws, restorer_runs, baseline_runs)
eval_dir = evaluate(ws, pairs_root, test_scenes, methods, tag="eval_pilot", log=stages.log)
stages.show_markdown(eval_dir / "summary.md")
for p in sorted(eval_dir.glob("sheet_*.jpg"))[:4]:
    display(IPImage(filename=str(p), width=1400))
''', title="Değerlendir"),
    md("""
## 5. Splat'e geri besleme testi
Aynı zayıf splat'ten başlayarak: `control` sadece gerçek eğitim fotoğraflarıyla, diğerleri ek olarak düzeltilmiş
render'larla ince ayarlanır. Hepsi hiçbir splat'in görmediği karelerde ölçülür. **Asıl karar bu tablodan çıkar:**
`Δ vs control` pozitif PSNR / negatif LPIPS ve çoğu sahnede "win" görmek istiyoruz.
"""),
    code('''
from splat_restorer.distill import run_distill
best = f"restorer_ref_n{max(SCALING)}"
pick = [m for m in (best, f"restorer_noref_n{max(SCALING)}", f"cnn_ref_n{max(SCALING)}") if m in methods]
distill_dir = run_distill(ws, pairs_root, test_scenes, RECIPES, {m: methods[m] for m in pick},
                          tag="distill_pilot", log=stages.log)
stages.show_markdown(distill_dir / "summary.md")
''', title="Geri besleme testi"),
    md("""
## Sonuçlar nerede?
`MyDrive/<DRIVE_DIR>/results/` altında `eval_pilot/summary.md`, `distill_pilot/summary.md` ve görsel karşılaştırmalar.
Bu iki `summary.md` dosyasını ve birkaç `sheet_*.jpg`'yi bana getirmen yeterli.
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

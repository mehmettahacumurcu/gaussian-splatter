"""Build colab/gsplat_testbed.ipynb: gsplat's reference trainer variants on our held-out frames.

    python experiments/splat_restorer/build_gsplat_testbed_notebook.py
"""
from __future__ import annotations

import json

from build_notebook import PKG, code, md

OUT = PKG.parents[2] / "colab" / "gsplat_testbed.ipynb"
GSPLAT_TAG = "v1.5.3"
PYCOLMAP = "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"
NERFVIEW = "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
FUSED_SSIM = "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"


def sources() -> str:
    files = {f"lib/splat_restorer/{p.name}": p.read_text(encoding="utf-8")
             for p in sorted(PKG.iterdir()) if p.suffix in (".py", ".txt")}
    return json.dumps(files, ensure_ascii=False)


CELLS = [
    md(f"""
# gsplat test zemini: araştırma yöntemleri aynı kareler üzerinde

gsplat'in bakımlı referans eğiticisi (`examples/simple_trainer.py`, {GSPLAT_TAG}) her seferinde **tek bir ayar**
değiştirilerek eğitilir; hepsi önceki testlerle **aynı 93 saklanan fotoğrafta** (her 10. kare) aynı kodla ölçülür.

| varyant | ne değişir |
|---|---|
| `default` | gsplat varsayılanı (orijinal 3DGS yoğunlaştırma), 1920x1080 |
| `mcmc` | 3DGS-MCMC yoğunlaştırma, 1M splat sınırı |
| `absgrad` | AbsGS: mutlak gradyanla bölme (ince detay) |
| `antialiased` | Mip-Splatting'in 2D anti-aliasing filtresi |
| `pose_opt` | eğitim sırasında kamera pozlarını iyileştirme |
| `default_4k` | 4K tam çözünürlükte eğitim (en yavaşı, en sona konuldu) |

Ayrıca: yakınındaki **eğitim** kareleri bile çöken (yanlış SfM pozu) saklanan kareler otomatik bulunur ve
"poz hatasız" ikinci bir ortalama raporlanır; app eğiticisiyle yapılan A/B sonuçları da aynı tabloya eklenir.

**Kullanım:** GPU (A100 önerilir) → **Run all**. 1080p varyant başına ~15-25 dk, 4K ~45-90 dk.
Her varyant bitince Drive'a yazılır; kopma olursa Run all biten varyantları atlar.
"""),
    code('''
DATASET_ZIP = "/content/drive/MyDrive/GaussianTests/inputs/IMG_5966_dataset.zip"  # @param {type:"string"}
OUT_DIR = "/content/drive/MyDrive/GaussianTests/gsplat_testbed/IMG_5966"  # @param {type:"string"}
APP_AB_DIR = "/content/drive/MyDrive/GaussianTests/training_ab/IMG_5966"  # @param {type:"string"}
VARIANTS = "default,mcmc,absgrad,antialiased,pose_opt,default_4k"  # @param {type:"string"}
STEPS = 30000          # @param {type:"integer"}
TEST_EVERY = 10        # @param {type:"integer"}
SAVE_PLY = True        # @param {type:"boolean"}
FACTOR_OVERRIDE = 0    # @param {type:"integer"}
# FACTOR_OVERRIDE > 0: hızlı deneme için bütün varyantlarda çözünürlük bölücüsü (ör. 4).
# APP_AB_DIR: app eğiticisinin A/B sonuçları (varsa tabloya 'app:' satırları olarak eklenir; yoksa boş bırak).
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
from splat_restorer import own_splat, train_ab, gsplat_testbed as tb
print(f"{{len(SOURCES)}} dosya yazıldı")
''', title="Kodu yaz (gömülü kaynaklar)"),
    code('''
import shutil, zipfile
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
assert model_name == "0", "gsplat eğiticisi sparse/0 bekliyor"
# Ölçüm için gerçek fotoğraflar: önceki testlerle aynı hazırlık (1920 uzun kenar).
data = own_splat.prepare_dataset(src, Path("/content/own_prepared_1920"), long_edge=1920,
                                 model_name=model_name, workers=os.cpu_count() or 4)
holdout = list(range(0, data.num_frames, TEST_EVERY))   # gsplat'in val kümesiyle aynı (sıralı isimler, i % N == 0)
train_ids = [i for i in range(data.num_frames) if i not in set(holdout)]
factors = sorted({FACTOR_OVERRIDE or tb.VARIANTS[v].factor for v in VARIANTS})
for f in factors:
    tb.prepare_factor_images(src, f, workers=os.cpu_count() or 4)
print(f"{data.num_frames} kare, saklanan {len(holdout)}; gsplat çözünürlük bölücüleri {factors}")
''', title="Drive + veri"),
    code('''
import dataclasses, time, lpips, torch
from IPython.display import Markdown, display
assert torch.cuda.is_available(), "GPU yok: Runtime → Change runtime type → GPU"
net_lpips = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
flags_path = out_dir / "pose_flags.json"
results = {}
for name in VARIANTS:
    res_path = out_dir / name / "result.json"
    if res_path.exists():
        results[name] = json.loads(res_path.read_text())
        print(f"{name}: daha önce bitmiş, atlandı")
        continue
    v = tb.VARIANTS[name]
    if FACTOR_OVERRIDE:
        v = dataclasses.replace(v, factor=FACTOR_OVERRIDE)
    print(f"\\n===== {name}: {v.note}")
    run_dir = Path("/content/gs_runs") / name
    shutil.rmtree(run_dir, ignore_errors=True)
    seconds = tb.run_variant(v, src, Path("/content/gsplat_src/examples"), run_dir, STEPS, TEST_EVERY)
    params, deg = tb.load_checkpoint(run_dir)
    (out_dir / name).mkdir(parents=True, exist_ok=True)
    frames = train_ab.evaluate_params(params, deg, data, holdout, None, net_lpips, out_dir / name / "renders",
                                      rasterize_mode=v.rasterize_mode)
    if not flags_path.exists():
        print("Eğitim karelerinde poz hatası taranıyor...")
        tp = tb.frame_psnr(params, deg, data, train_ids, v.rasterize_mode)
        flagged, bad_train, thr = tb.pose_failure_holdouts(tp, holdout)
        tb.save_json(flags_path, {"reference_variant": name, "threshold_db": thr, "flagged_holdout": flagged,
                                  "bad_train": bad_train, "train_psnr": {str(k): x for k, x in tp.items()}})
        print(f"Çöken eğitim kareleri ({len(bad_train)}): {bad_train}")
        print(f"Hariç tutulacak saklanan kareler: {flagged}")
    if SAVE_PLY:
        tb.save_ply(params, out_dir / name / "final.ply")
    for stats in (run_dir / "stats").glob("val_step*.json"):
        shutil.copy2(stats, out_dir / name / ("gsplat_" + stats.name))
    results[name] = {"source": "gsplat", "info": {"variant": dataclasses.asdict(v), "train_seconds": seconds,
                     "splats": int(params["means"].shape[0])}, "frames": frames}
    tb.save_json(res_path, results[name])
    del params
    torch.cuda.empty_cache()
    m = tb.summarise({name: results[name]}, holdout)[name]
    print(f"{name}: PSNR {m['psnr']:.2f}  LPIPS {m['lpips']:.4f}  {m['splats']:,} splat  {m['minutes']:.0f} dk")
''', title="Eğit + ölç (varyant başına ~15-90 dk)"),
    code('''
from IPython.display import Image as IPImage
flags = json.loads(flags_path.read_text())
keep = [h for h in holdout if h not in set(flags["flagged_holdout"])]
table = dict(results)
if APP_AB_DIR and Path(APP_AB_DIR).exists():   # app eğiticisinin A/B sonuçları, aynı kareler
    for p in sorted(Path(APP_AB_DIR).glob("*/result.json")):
        table["app:" + p.parent.name] = json.loads(p.read_text())
summary = tb.summarise(table, keep)
md_text = tb.markdown(summary, len(holdout), flags["flagged_holdout"], flags["threshold_db"])
(out_dir / "summary.md").write_text(md_text, encoding="utf-8")
tb.save_json(out_dir / "summary.json", summary)
display(Markdown(md_text))
sheets = train_ab.comparison_sheets(results, data, out_dir, out_dir)
print("Sütunlar: foto | " + " | ".join(results) + "; alt sıra: kırmızı kutunun zoom'u")
for p in sheets[:6]:
    display(IPImage(filename=str(p), width=1600))
''', title="Sonuçlar"),
    md("""
## Nasıl okunur?

* **PSNR / LPIPS:** saklanan fotoğraflara karşı; **Δ** `default`'a göre. ~0,3 dB altındaki farklar koşudan koşuya oynayabilir.
* **Poz hatasız sütunlar:** yanlış kamera pozu yüzünden hiçbir yöntemin düzeltemeyeceği kareler çıkarılmış hâli;
  yöntemleri karşılaştırmak için daha güvenilir.
* `app:` satırları app'in kendi eğiticisi (önceki A/B testi). gsplat `default` ile `app:lr_decay` yakınsa, app eğiticisi
  öğrenme oranı düzeltmesiyle referans seviyesine gelmiş demektir.

Bana `summary.md` ve birkaç `compare_*.jpg` getirmen yeterli (Drive'da `OUT_DIR` altında).
"""),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
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

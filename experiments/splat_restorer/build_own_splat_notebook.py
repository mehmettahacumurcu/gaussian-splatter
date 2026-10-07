"""Build colab/difix_own_splat_test.ipynb: pretrained Difix on one of our own splats.

Run after editing anything under splat_restorer/:

    python experiments/splat_restorer/build_own_splat_notebook.py
"""
from __future__ import annotations

from build_notebook import PKG, code, embedded_sources, md

OUT = PKG.parents[2] / "colab" / "difix_own_splat_test.ipynb"

CELLS = [
    md("""
# Difix testi: kendi splat'imizde kaliteyi artırıyor mu?

NVIDIA'nın eğitilmiş **Difix** modelini bizim eğittiğimiz bir splat üzerinde gerçek fotoğraflarla ölçer.
Eğitim yok; varsayılan ayarlar IMG_5966 splat'ine göre doldurulmuştur.

**Akış**
1. **Veri:** Dataset zip'i açılır ve splat'in eğitildiği şekilde hazırlanır (lens düzeltme, 1920x1080, aynı kare sırası).
2. **Doğrulama:** Splat yüklenir; eğitim notebook'unun kendi değerlendirmesi yeniden render edilir.
   PSNR'lar tutuyorsa (±0,1 dB) kurulum doğrudur.
3. **Saklanan kareler:** Splat'in eğitimde **hiç görmediği** her 10. fotoğrafta (93 kare)
   ham render, Difix ve Difix-ref (en yakın eğitim fotoğrafını referans alır) gerçek fotoğrafla karşılaştırılır.
4. **Splat'e geri besleme:** Splat, Difix'in düzelttiği yeni kamera açılarıyla kısa bir ince ayardan geçer.
   Saklanan karelerde splat'in kendisi iyileşiyor mu, sadece gerçek fotoğraflarla yapılan ince ayara (`control`) göre ölçülür.

**Kullanım:** GPU runtime (A100 veya L4 önerilir) → **Runtime → Run all**. Toplam ~30-60 dk.
Sonuçlar Drive'da `OUT_DIR` altına yazılır; kopma olursa tekrar Run all, bitmiş adımlar atlanır.

Lisans: Difix kodu ve ağırlıkları NVIDIA lisansıyla yalnızca ticari olmayan araştırma/değerlendirme içindir.
"""),
    code('''
DATASET_ZIP = "/content/drive/MyDrive/GaussianTests/inputs/IMG_5966_dataset.zip"  # @param {type:"string"}
RUN_DIR = "/content/drive/MyDrive/GaussianTests/results/scene_20260928-173733_631d09"  # @param {type:"string"}
OUT_DIR = "/content/drive/MyDrive/GaussianTests/difix_test/IMG_5966"  # @param {type:"string"}
MODELS = "difix,difix_ref"   # @param {type:"string"}
RESTORE_LONG_EDGE = 0        # @param {type:"integer"}
RUN_DISTILL = True           # @param {type:"boolean"}
DISTILL_MODEL = "difix_ref"  # @param ["difix_ref", "difix"]
DISTILL_STEPS = 3000         # @param {type:"integer"}
N_PSEUDO = 120               # @param {type:"integer"}
PSEUDO_SHIFT = 0.03          # @param {type:"number"}
# RUN_DIR: eğitim çıktısı (final_*.ply, run_manifest.json, evaluation/metrics.json).
# RESTORE_LONG_EDGE: 0 = Difix tam çözünürlükte (1920) çalışır; 1024 = küçültüp çalıştırıp geri büyütür.
# PSEUDO_SHIFT: yeni kamera açıları, kamera yolunun boyutunun bu oranı kadar sağa/sola/yukarı/aşağı kaydırılır.
MODELS = [m.strip() for m in MODELS.split(",") if m.strip()]
''', title="Ayarlar"),
    code('''
import os
os.environ["MAX_JOBS"] = "4"
!pip install -q "gsplat==1.5.3" ninja "diffusers==0.32.2" "transformers==4.47.1" "peft==0.14.0" "accelerate==1.2.1" lpips
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
''', title="Kurulum (sabitlenmiş sürümler)"),
    code(f'''
import json, sys
from pathlib import Path

SOURCES = json.loads({embedded_sources()!r})
pkg = Path("/content/splat_restorer_src/splat_restorer")
pkg.mkdir(parents=True, exist_ok=True)
for name, text in SOURCES.items():
    (pkg / name).write_text(text, encoding="utf-8")
if "/content/splat_restorer_src" not in sys.path:
    sys.path.insert(0, "/content/splat_restorer_src")
print(f"{{len(SOURCES)}} dosya yazıldı: {{pkg}}")
''', title="Kodu yaz (gömülü kaynaklar)"),
    code('''
from google.colab import drive
drive.mount("/content/drive")
try:  # nvidia/difix açık bir model; token sadece varsa kullanılır.
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass

run_dir, out_dir = Path(RUN_DIR), Path(OUT_DIR)
plys = sorted(run_dir.glob("final_*.ply"))
assert Path(DATASET_ZIP).exists(), f"Dataset zip bulunamadı: {DATASET_ZIP}"
assert plys, f"{run_dir} içinde final_*.ply yok"
manifest = json.loads((run_dir / "run_manifest.json").read_text())
ply_path = plys[-1]
out_dir.mkdir(parents=True, exist_ok=True)
print("splat:", ply_path.name, "| eğitim:", manifest.get("config"), "| saklanan kare:", len(manifest["holdout_indices"]))
''', title="Drive + girdiler"),
    code('''
import shutil, zipfile
from splat_restorer import own_splat

raw = Path("/content/own_dataset")
if not (raw / ".unzipped").exists():
    print("Zip açılıyor (Drive'dan okuma birkaç dakika sürebilir)...")
    shutil.rmtree(raw, ignore_errors=True)
    with zipfile.ZipFile(DATASET_ZIP) as z:
        z.extractall(raw)
    (raw / ".unzipped").touch()
src = next(p.parent.parent for p in raw.rglob("sparse/*/cameras.bin"))
data = own_splat.prepare_dataset(src, Path("/content/own_prepared"), long_edge=manifest["config"].get("long_edge", 1920),
                                 workers=os.cpu_count() or 4)
report_path = run_dir / "import_report.json"
if report_path.exists():
    report = json.loads(report_path.read_text())
    same = dict(zip(data.names, data.frames)) == report["mapping"]
    print("Kare sırası eğitimle aynı:", same)
    assert same, "Kare eşleşmesi eğitim koşusuyla tutmuyor; saklanan kareler yanlış olur."
holdout = manifest["holdout_indices"]
train_ids = [i for i in range(data.num_frames) if i not in set(holdout)]
print(f"{data.num_frames} kare {data.width}x{data.height}; eğitim {len(train_ids)}, saklanan {len(holdout)}")
''', title="Veriyi hazırla"),
    code('''
import numpy as np, torch
assert torch.cuda.is_available(), "GPU yok: Runtime → Change runtime type → GPU"
params, sh_degree = own_splat.load_ply(ply_path, "cuda")  # ilk render'da gsplat derlenir (~2-5 dk)
print(f"{len(params['means']):,} splat, SH derecesi {sh_degree}")
orig_path = run_dir / "evaluation" / "metrics.json"
if orig_path.exists():
    rows = own_splat.reproduce_check(params, sh_degree, data, json.loads(orig_path.read_text()))
    for r in rows:
        print(f"  {r['frame']}: eğitimdeki {r['original']:.2f} dB, şimdi {r['now']:.2f} dB")
    gap = np.mean([r["now"] - r["original"] for r in rows])
    print(f"Ortalama fark: {gap:+.3f} dB")
    assert abs(gap) < 0.3, "Render eğitimdekiyle tutmuyor; sonuçlar güvenilmez olur (kurulum/kamera sorunu)."
''', title="Splat'i yükle + doğrula"),
    code('''
import lpips
from IPython.display import Image as IPImage, Markdown, display
from splat_restorer.restorer_model import Restorer

net_lpips = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
loader = lambda repo: (lambda: Restorer.from_difix(repo, device="cuda"))
methods = {m: (loader(f"nvidia/{m}"), m.endswith("_ref")) for m in MODELS}
summary = own_splat.evaluate_holdout(params, sh_degree, data, holdout, train_ids, methods, out_dir / "eval", net_lpips,
                                     restore_long_edge=RESTORE_LONG_EDGE)
display(Markdown((out_dir / "eval" / "summary.md").read_text()))
print("Sütunlar: foto | render | " + " | ".join(MODELS) + " | referans; alt sıra: kırmızı kutunun zoom'u")
worst = [w["frame"] for w in summary["worst"]][:3]
for i in worst + [i for i in summary["shown_frames"] if i not in worst][:2]:
    display(IPImage(filename=str(out_dir / "eval" / f"compare_{i:06d}.jpg"), width=1600))
''', title="Saklanan karelerde ölçüm (her model ~5 dk)"),
    code('''
if RUN_DISTILL:
    dm = {DISTILL_MODEL: (loader(f"nvidia/{DISTILL_MODEL}"), DISTILL_MODEL.endswith("_ref"))}
    own_splat.distill_back(params, sh_degree, data, holdout, train_ids, dm, out_dir / "distill", net_lpips,
                           steps=DISTILL_STEPS, n_pseudo=N_PSEUDO, shift_frac=PSEUDO_SHIFT,
                           restore_long_edge=RESTORE_LONG_EDGE)
    display(Markdown((out_dir / "distill" / "summary.md").read_text()))
    print("Yeni kamera açılarından örnekler: ham render | Difix | kaynak fotoğraf")
    for p in sorted((out_dir / "distill" / f"pseudo_{DISTILL_MODEL}").glob("*.jpg"))[:3]:
        display(IPImage(filename=str(p), width=1600))
''', title="Splat'e geri besleme (~15-30 dk)"),
    md("""
## Nasıl okunur?

* **Saklanan kareler tablosu:** Difix satırında **LPIPS düşüyor** ve PSNR belirgin düşmüyorsa render'ı gerçekten
  fotoğrafa yaklaştırıyor. PSNR düşüp LPIPS iyileşiyorsa keskin ama uydurma detay ekliyor olabilir; zoom'lara bak.
* **En kötü kareler:** Splat'in zayıf olduğu yerler (ör. 13-18 dB'lik kareler); Difix'in asıl fark yaratması gereken yer.
* **Geri besleme tablosu:** Difix satırı `control`'ü geçiyorsa düzeltilmiş görüntüler splat'in kendisini iyileştirmiş demektir.

Bana `eval/summary.md`, `distill/summary.md` ve birkaç `eval/compare_*.jpg` getirmen yeterli.
"""),
]


def main() -> None:
    import json

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

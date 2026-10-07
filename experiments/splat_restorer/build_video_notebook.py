"""Build colab/difix_video_test.ipynb: NVIDIA's pretrained Difix on a splat render video.

Run after editing restorer_model.py or video.py:

    python experiments/splat_restorer/build_video_notebook.py
"""
from __future__ import annotations

import json

from build_notebook import PKG, code, md

OUT = PKG.parents[2] / "colab" / "difix_video_test.ipynb"
EMBED = ("__init__.py", "restorer_model.py", "video.py", "LICENSE-DIFIX3D.txt")


def embedded_sources() -> str:
    return json.dumps({name: (PKG / name).read_text(encoding="utf-8") for name in EMBED}, ensure_ascii=False)


CELLS = [
    md("""
# Difix video testi: hazır model splat render'ını iyileştiriyor mu?

NVIDIA'nın eğitilmiş **Difix** modeli (`nvidia/difix`), splat render videosunun her karesine uygulanır.
Eğitim yok; sadece hazır modelin bizim splat'imizde ne yaptığına bakıyoruz.

**Çıktılar** (Drive'da `MyDrive/<OUT_DIR>/<video adı>/`):
* `side_by_side.mp4`: solda orijinal render, sağda Difix
* `compare.jpg`: 4 kare; modelin en çok değiştirdiği bölgeye zoom (kırmızı kutu)
* `restored.mp4`: sadece düzeltilmiş video
* `summary.md` / `metrics.json`: değişim miktarı, keskinlik, titreme

**Kullanım:** `VIDEO_PATH`'e Drive'daki videonun yolunu yaz (boş bırakırsan dosya yükleme penceresi açılır),
sonra **Runtime → Run all**. GPU runtime gerekli; A100/L4 hızlı, T4 de çalışır (daha yavaş).

**Bilinmesi gerekenler**
* Gerçek fotoğraf olmadığı için doğruluk ölçülemez; Difix bazen detay **uydurur**. Karar zoom'lara bakarak verilir.
* Kareler tek tek işlendiği için videoda hafif titreme olabilir; `summary.md` bunu ayrıca ölçer.
* `difix_ref` modeli için `REF_IMAGE`'e sahnenin gerçek bir fotoğrafını ver (splat'in eğitildiği videodan bir kare).
* Lisans: Difix kodu ve ağırlıkları NVIDIA lisansıyla yalnızca ticari olmayan araştırma/değerlendirme içindir.
"""),
    code('''
VIDEO_PATH = ""            # @param {type:"string"}
MODEL = "difix"            # @param ["difix", "difix_ref"]
REF_IMAGE = ""             # @param {type:"string"}
LONG_EDGE = 1024           # @param {type:"integer"}
FRAME_STRIDE = 1           # @param {type:"integer"}
MAX_SECONDS = 0            # @param {type:"number"}
OUT_DIR = "difix_video_test"  # @param {type:"string"}
# VIDEO_PATH örn. "/content/drive/MyDrive/videolar/son_splat.mp4"; boşsa yükleme penceresi açılır.
# LONG_EDGE: işleme çözünürlüğünün uzun kenarı (Difix 1024x576 civarında eğitildi; büyütmez, sadece küçültür).
# FRAME_STRIDE: 2 = her 2. kare (hızlı deneme). MAX_SECONDS: 0 = videonun tamamı.
assert MODEL != "difix_ref" or REF_IMAGE, "difix_ref için REF_IMAGE (gerçek fotoğraf) gerekli; yoksa MODEL='difix' seç."
''', title="Ayarlar"),
    code('''
!pip install -q "diffusers==0.32.2" "transformers==4.47.1" "peft==0.14.0" "accelerate==1.2.1"
!ffmpeg -version | head -1
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
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
print(f"{{len(SOURCES)}} dosya yazıldı: {{pkg}}")
''', title="Kodu yaz (gömülü kaynaklar)"),
    code('''
from google.colab import drive
drive.mount("/content/drive")
try:  # nvidia/difix açık bir model; token sadece varsa kullanılır (indirme limiti için).
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass

if VIDEO_PATH:
    video_path = Path(VIDEO_PATH)
    assert video_path.exists(), f"Video bulunamadı: {video_path} (Drive'daki tam yolu kontrol et)"
else:
    from google.colab import files
    up = Path("/content/upload"); up.mkdir(exist_ok=True)
    print("Videoyu seç (büyük dosyalarda Drive'a koyup VIDEO_PATH kullanmak daha hızlı):")
    uploaded = files.upload()
    name = next(iter(uploaded))
    video_path = up / name
    video_path.write_bytes(uploaded[name])
ref_path = Path(REF_IMAGE) if REF_IMAGE else None
assert ref_path is None or ref_path.exists(), f"Referans görüntü bulunamadı: {ref_path}"

from splat_restorer import video
info = video.probe(video_path)
print(f"{video_path.name}: {info['width']}x{info['height']}, {info['fps']:.2f} fps, {info['duration']:.1f} s")
''', title="Drive + video"),
    code('''
import torch
from splat_restorer.restorer_model import Restorer
assert torch.cuda.is_available(), "GPU yok: Runtime → Change runtime type → GPU"
model = Restorer.from_difix(f"nvidia/{MODEL}", device="cuda")
print(f"nvidia/{MODEL} yüklendi")
''', title="Modeli yükle (ilk seferde ~5 GB indirme, birkaç dakika)"),
    code('''
out_dir = Path("/content/drive/MyDrive") / OUT_DIR / f"{video_path.stem}_{MODEL}"
result = video.run(video_path, out_dir, model, repo=f"nvidia/{MODEL}", ref_image=ref_path,
                   long_edge=LONG_EDGE, stride=FRAME_STRIDE, max_seconds=MAX_SECONDS)
print("Çıktılar:", out_dir)
''', title="Videoyu işle"),
    code('''
import base64
from IPython.display import HTML, Image as IPImage, Markdown, display
display(Markdown((out_dir / "summary.md").read_text(encoding="utf-8")))
display(IPImage(filename=str(out_dir / "compare.jpg"), width=1400))
sbs = out_dir / "side_by_side.mp4"
if sbs.stat().st_size < 40 * 2**20:
    b64 = base64.b64encode(sbs.read_bytes()).decode()
    display(HTML(f'<video controls loop width="1400" src="data:video/mp4;base64,{b64}"></video>'))
else:
    print(f"Video notebook'ta gösterilemeyecek kadar büyük; Drive'dan aç: {sbs}")
''', title="Sonuçlar"),
    md("""
## Nasıl karar veririz?

* **Zoom'larda** floater'lar, bulanık bölgeler, splat "lekeleri" temizlenmiş ve detay gerçekçi görünüyorsa: umut verici.
* Detay **uydurulmuş** görünüyorsa (olmayan doku, şekil değişimi) ya da video belirgin titriyorsa: hazır model yetmiyor.
* `summary.md`, `compare.jpg` ve `side_by_side.mp4`'ten birkaç saniyelik bir kesit bana getirmen yeterli.
"""),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "L4"},
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

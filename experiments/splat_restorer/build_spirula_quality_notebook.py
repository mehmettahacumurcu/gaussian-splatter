"""Build colab/spirula_quality_test.ipynb: Spirula variants scored on our held-out photos.

The app's Spirula recipe (binary download, Vulkan recovery, dataset import,
MoGe-2 geometry) is taken unchanged from git (APP_REF); only its settings are
filled in and its single training cell is replaced by a variant loop plus our
evaluation:

    python experiments/splat_restorer/build_spirula_quality_notebook.py
"""
from __future__ import annotations

import ast
import json
import subprocess

from build_notebook import PKG, code, md

REPO = PKG.parents[2]
OUT = REPO / "colab" / "spirula_quality_test.ipynb"
APP_REF = "feature/scene-composer"
TEMPLATE = "backend/notebooks/templates/spirula.ipynb"

SETTINGS = {  # values for the template's own settings cell
    "INPUT_MODE": "dataset_zip",
    "INPUT_DATASET_ZIP": "/content/drive/MyDrive/GaussianTests/inputs/IMG_5966_dataset.zip",
    "GEOMETRY_MODEL": "moge2-vitb",  # what the 28 Sep run used
    "GENERATE_DEPTH": True,
    "ALLOW_PARTIAL_RECONSTRUCTION": False,
    "RESUME_CHECKPOINT": "",
}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout


def replace_assignments(source: str, values: dict) -> str:
    """Replace named top-level assignments with literals (same rule as the app's generator)."""
    lines = source.splitlines(keepends=True)
    edits = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in values:
                edits.append((node.lineno - 1, node.end_lineno, f"{name} = {values[name]!r}\n"))
    missing = set(values) - {ast.parse(t).body[0].targets[0].id for _, _, t in edits}
    if missing:
        raise ValueError(f"Template drift, settings not found: {missing}")
    for start, end, text in reversed(edits):
        lines[start:end] = [text]
    return "".join(lines)


def package_sources() -> str:
    files = {p.name: p.read_text(encoding="utf-8") for p in sorted(PKG.iterdir()) if p.suffix in (".py", ".txt")}
    return json.dumps(files, ensure_ascii=False)


def template_cells() -> tuple[list[dict], list[dict], str]:
    nb = json.loads(git("show", f"{APP_REF}:{TEMPLATE}"))
    sha = git("rev-parse", "--short", APP_REF).strip()
    cells = nb["cells"]
    train_idx = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "'train', '3dgs'" in "".join(c["source"]))
    settings_idx = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "INPUT_MODE =" in "".join(c["source"]))
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    s = replace_assignments("".join(cells[settings_idx]["source"]), SETTINGS)
    s += "\nTRAIN_QUALITY = 'medium'\nSFM_QUALITY = 'high'  # unused: the dataset's existing SfM model is reused\n"
    cells[settings_idx]["source"] = s.splitlines(keepends=True)
    # Keep everything up to the training cell (binary, Vulkan, data, geometry), drop
    # the template's intro and its single-run training/export cells.
    end = train_idx
    while cells[end - 1]["cell_type"] == "markdown":  # the template's heading for its training step
        end -= 1
    return cells[1:settings_idx + 1], cells[settings_idx + 1:end], sha


def build() -> dict:
    head, middle, sha = template_cells()
    intro = md(f"""
# Spirula kalite testi: aynı 93 saklanan kare, varyant varyant

App'teki Spirula Colab tarifi (`{APP_REF}` @ `{sha}`: resmi v2026.9.24 ikilisi, Vulkan kurtarma, veri içe aktarma,
MoGe-2 geometri) **aynen** kullanılır; eğitim adımı aşağıdaki varyantları sırayla çalıştırır. Her varyant her 10. kareyi
saklar (`eval_mode=interval`, 0,10,...,920 = diğer deneylerimizdeki 93 kare). Dışa aktarılan PLY bizim
kameralarımızla render edilip **aynı ölçümle** puanlanır; referans satırı bizim eğiticinin en iyi sonucudur (lr_decay).

| varyant | değişiklik | süre (RTX PRO 6000) |
|---|---|---|
| `medium_holdout` | 28 Eylül ayarları (medium 1M/30k, 4K, normal+derinlik 0,01) | ~15 dk |
| `medium_no_appearance` | kare başına renk telafisi (bilateral grid + PPISP) kapalı | ~15 dk |
| `medium_floater_mild` | `floater_suppression=mild` | ~15 dk |
| `medium_depth_x5` | derinlik desteği 0,05 | ~15 dk |
| `high_native` | Spirula high: 3M / 50k | ~45 dk |
| `high_no_appearance` | high + renk telafisi kapalı: PLY renkleri dürüst (final adayı) | ~60 dk |
| `high_no_appearance_floater` | üstüne floater bastırma mild (isteğe bağlı) | ~60 dk |
| `ultra` *(varsayılan kapalı)* | 10M / 80k, 80 GB GPU önerilir | ~2,5 sa |

**Kullanım:** Runtime → **RTX PRO 6000** (yoksa A100 80 GB; `ultra` için şart), **High-RAM** → **Run all**.
Önce ikili + Vulkan + geometri (~15-20 dk), sonra varyantlar. Her varyant bitince sonucu Drive'a yazılır;
kopma olursa Run all biten varyantları atlar (yarım kalan varyant baştan başlar).

Colab'a özgü kısımlar (Linux Vulkan ikilisi, `/content` yolları, Drive) yerelde çalıştırılamaz; değerlendirme kodu
yerelde arşivdeki Spirula PLY'siyle doğrulandı.
""")
    settings = code('''
OUT_DIR = "/content/drive/MyDrive/GaussianTests/spirula_quality/IMG_5966"  # @param {type:"string"}
VARIANTS = "medium_holdout,medium_no_appearance,medium_floater_mild,medium_depth_x5,high_native,high_no_appearance"  # @param {type:"string"}
REFERENCE_PLY = "/content/drive/MyDrive/GaussianTests/training_ab/IMG_5966/lr_decay/final.ply"  # @param {type:"string"}
# VARIANTS'a "ultra" eklenebilir (10M / 80k, ~2,5 saat, 80 GB GPU).
VARIANTS = [v.strip() for v in VARIANTS.split(",") if v.strip()]
''', title="Test ayarları")
    sources = code(f'''
import json, sys
from pathlib import Path
SOURCES = json.loads({package_sources()!r})
_pkg = Path("/content/sq_src/splat_restorer")
_pkg.mkdir(parents=True, exist_ok=True)
for _name, _text in SOURCES.items():
    (_pkg / _name).write_text(_text, encoding="utf-8")
if "/content/sq_src" not in sys.path:
    sys.path.insert(0, "/content/sq_src")
from splat_restorer import spirula_quality as sq
_unknown = [v for v in VARIANTS if v not in sq.VARIANTS]
assert not _unknown, f"Bilinmeyen varyant: {{_unknown}}; seçenekler: {{list(sq.VARIANTS)}}"
# Şablonun bellek ön kontrolü en ağır seçili varyanta göre yapılsın.
TRAIN_ITERATIONS = max(sq.VARIANTS[v].iterations for v in VARIANTS)
TRAIN_CAP_MAX = max(sq.VARIANTS[v].cap for v in VARIANTS)
MIN_TRAIN_VRAM_GIB = max(sq.VARIANTS[v].min_gib for v in VARIANTS)
print("Varyantlar:", VARIANTS, "| en ağır:", TRAIN_ITERATIONS, "adım,", TRAIN_CAP_MAX, "splat,", MIN_TRAIN_VRAM_GIB, "GiB")
''', title="Test kodunu yaz")
    data_prep = code('''
!pip install -q "gsplat==1.5.3" lpips plyfile
import os, shutil, zipfile
from splat_restorer import own_splat
own_raw = Path("/content/own_dataset")
if not (own_raw / ".unzipped").exists():
    shutil.rmtree(own_raw, ignore_errors=True)
    with zipfile.ZipFile(INPUT_DATASET_ZIP) as z:
        z.extractall(own_raw)
    (own_raw / ".unzipped").touch()
_cams = next(own_raw.rglob("sparse/*/cameras.bin"))
data = own_splat.prepare_dataset(_cams.parents[2], Path("/content/own_prepared_1920"), long_edge=1920,
                                 model_name=_cams.parent.name, workers=os.cpu_count() or 4)
holdout = list(range(0, data.num_frames, 10))
out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
print(f"Ölçüm: {data.num_frames} kare {data.width}x{data.height}, saklanan {len(holdout)}; "
      f"poz hatası nedeniyle ikinci ortalamada hariç: {[i for i in holdout if sq.is_pose_fail(i)]}")
''', title="Ölçüm verisini hazırla (diğer deneylerle aynı)")
    loop = code('''
import lpips, re, subprocess, time, torch
net_lpips = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
results = {}

def evaluate(name, ply, note, minutes):
    r = sq.evaluate_ply(ply, data, holdout, net_lpips, out_dir / name / "renders")
    r.update(note=note, minutes=minutes)
    return r

if REFERENCE_PLY and Path(REFERENCE_PLY).exists():
    ref_path = out_dir / "reference_lr_decay" / "result.json"
    if not ref_path.exists():
        sq.save_json(ref_path, evaluate("reference_lr_decay", Path(REFERENCE_PLY), "bizim eğitici, lr_decay (referans)", None))
    results["reference_lr_decay"] = json.loads(ref_path.read_text())

for name in VARIANTS:
    v = sq.VARIANTS[name]
    vdir = out_dir / name
    res_path = vdir / "result.json"
    if res_path.exists():
        results[name] = json.loads(res_path.read_text())
        print(f"{name}: daha önce bitmiş, atlandı")
        continue
    print(f"\\n===== {name}: {v.note} ({v.quality}, {v.iterations:,} adım, {v.cap:,} splat)", flush=True)
    prefix = OUT / name
    shutil.rmtree(prefix, ignore_errors=True)
    args = sq.train_args(BINARY, TRAIN_DATASET, TRAIN_RECON_DIR, prefix, v, GPU_SELECTOR, GENERATE_DEPTH)
    vdir.mkdir(parents=True, exist_ok=True)
    log_path = vdir / "train.log"
    t0 = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("COMMAND: " + repr(args) + "\\n")
        proc = subprocess.Popen(args, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        last = -1
        for line in proc.stdout:
            log.write(line)
            m = re.search(r"\\b(\\d+)\\s*/\\s*" + str(v.iterations) + r"\\b", line)
            if m and int(m.group(1)) // 2000 != last:  # her 2000 adımda bir satır
                last = int(m.group(1)) // 2000
                print(line.rstrip()[:160], flush=True)
            elif not m and re.search(r"error|warn|eval|psnr", line, re.I):
                print(line.rstrip()[:200], flush=True)
        code_ = proc.wait()
    minutes = (time.monotonic() - t0) / 60
    assert code_ == 0, f"{name}: Spirula çıkış kodu {code_}; günlük: {log_path}"
    found = sq.find_outputs(prefix)
    assert found["ply"] is not None, f"{name}: splat.ply bulunamadı ({prefix})"
    shutil.copy2(found["ply"], vdir / "spirula.ply")
    for key in ("metrics", "config"):
        if found[key] is not None:
            shutil.copy2(found[key], vdir / f"spirula_{key}.json")
    slot_frames = None
    if found["eval_gt"]:
        # Spirula yazdığı eval-gt sırası bizim sıramız değil: kareleri içeriğe göre eşle.
        slot_frames, match_psnr = sq.match_eval_slots(found["eval_gt"], data, holdout)
        # Spirula GT lens düzeltmesiz orijinal; bizim foto düzeltilmiş → aynı kare ~27-31 dB, farklı kare ~10.
        print(f"Saklanan kare eşleşmesi: en kötü {min(match_psnr):.1f} dB, "
              f"{len(set(slot_frames))}/{len(holdout)} farklı kare", flush=True)
        ok = min(match_psnr) > 20 and sorted(slot_frames) == sorted(holdout)
        assert ok, "Spirula'nın sakladığı kareler bizimkilerle aynı değil!"
    r = evaluate(name, vdir / "spirula.ply", v.note, minutes)
    r["native"] = sq.native_metrics(found["metrics"])
    r["spirula_eval_slot_frames"] = slot_frames
    r["variant"] = sq.variant_dict(v)
    sq.save_json(res_path, r)
    results[name] = r
    s = sq.summarise({name: r})[name]
    print(f"{name}: PLY PSNR {s['raw_psnr']:.2f}  LPIPS {s['raw_lpips']:.4f}  poz-hatası-hariç {s['raw_psnr_clean']:.2f}  "
          f"renk-eşitlemeli {s['cc_psnr']:.2f}  {r['splats']:,} splat  {minutes:.0f} dk")
    shutil.rmtree(prefix, ignore_errors=True)  # yerel diski boşalt; PLY ve ölçümler Drive'da
''', title="Varyantları eğit + ölç")
    results_cell = code('''
from IPython.display import Image as IPImage, Markdown, display
summary = sq.summarise(results)
excluded = sum(sq.is_pose_fail(i) for i in holdout)
md_text = sq.markdown(summary, len(holdout), excluded, "reference_lr_decay" if "reference_lr_decay" in results else None)
(out_dir / "summary.md").write_text(md_text, encoding="utf-8")
sq.save_json(out_dir / "summary.json", summary)
display(Markdown(md_text))
sheets = sq.comparison_sheets(results, data, out_dir, out_dir, holdout)
print("Sütunlar: foto | " + " | ".join(results) + "; alt sıra: kırmızı kutunun zoom'u")
for p in sheets:
    display(IPImage(filename=str(p), width=1600))
''', title="Sonuçlar")
    closing = md("""
## Nasıl okunur?

* **PSNR / LPIPS (PLY):** dışa aktarılan splat'in görüntüleyicide görüneceği hali, asıl karar ölçütü.
* **Poz hatası hariç:** kamera pozu bozuk 5 kare çıkarılmış hali; eğiticiler arasındaki gerçek fark burada daha net.
* **Renk eşitlemeli** ile ham PSNR arasındaki büyük fark, renk telafisinin PLY'ye taşınmadığını gösterir
  (`medium_no_appearance` ile karşılaştır).
* Bana `summary.md` ve birkaç `compare_*.jpg` getirmen yeterli (Drive'da `OUT_DIR` altında).
""")
    cells = [intro, *head, settings, sources, *middle, data_prep, loop, results_cell, closing]
    for c in cells:
        if c["cell_type"] == "code":
            ast.parse("\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%"))))
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "spirula_template": {"ref": APP_REF, "commit": sha, "path": TEMPLATE},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    nb = build()
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB), {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()

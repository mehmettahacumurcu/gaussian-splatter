"""Build colab/pose_repair_test.ipynb: do better SfM poses fix the frames no trainer can fit?

Spirula setup cells (pinned release, GLIBC check, NVIDIA Vulkan recovery,
logging) are taken from the app's maintained template with the same drift
checks the app's preprocessing notebook uses; the trainer and the evaluation
are the A/B notebook's (app trainer from APP_REF + lr_decay):

    python experiments/splat_restorer/build_pose_repair_notebook.py
"""
from __future__ import annotations

import ast
import json

from build_notebook import code, md
from build_training_ab_notebook import APP_REF, REPO, git, sources

OUT = REPO / "colab" / "pose_repair_test.ipynb"
SPIRULA_REF = "feature/spirula-preprocessing-notebook"
TEMPLATE = "backend/notebooks/templates/spirula.ipynb"

# Frames of the current IMG_5966 SfM whose OWN training render is below 18 dB with the 28 Sep splat
# (rendered for all 923 frames; mostly close-ups, about half next to gaps of unregistered video frames).
# Fixed by name, so no candidate is ever scored on a subset chosen from its own results.
BAD_NAMES = ["08217.jpg", "08248.jpg", "08263.jpg", "08278.jpg", "08292.jpg", "08309.jpg", "08368.jpg",
             "08862.jpg", "09492.jpg", "09507.jpg", "09522.jpg", "12044.jpg", "13227.jpg", "13244.jpg",
             "13258.jpg", "13273.jpg", "13407.jpg"]
# Two well-fitting frames as a sanity reference in the side-by-side sheets.
GOOD_NAMES = ["02714.jpg", "15553.jpg"]


def template_cells() -> list[str]:
    nb = json.loads(git("show", f"{SPIRULA_REF}:{TEMPLATE}"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def template_cell(cells: list[str], marker: str) -> str:
    found = [c for c in cells if marker in c]
    if len(found) != 1:
        raise ValueError(f"Spirula template drift: {marker}")
    return found[0]


def shared_bootstrap(cells: list[str]) -> str:
    """Pinned release constants and logging helpers, without the training setup (as the app's preprocess notebook)."""
    original = ast.parse(template_cell(cells, "def run_logged("))
    constants = {"RELEASE_TAG", "VERSION", "COMMIT", "ASSET_NAME", "ASSET_URL", "ASSET_SIZE", "EXPECTED_SHA256", "RELEASE_API"}
    functions = {"save_manifest", "_copy_file_to_drive", "_copy_tree_to_drive", "fail", "require", "run_logged"}
    picked = []
    for node in original.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not (isinstance(node, ast.ImportFrom) and node.module == "google.colab"):
            picked.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            picked.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            picked.append(node)
    if {n.name for n in picked if isinstance(n, ast.FunctionDef)} != functions:
        raise ValueError("Spirula logging template drift")
    text = ast.unparse(ast.Module(body=picked, type_ignores=[]))
    marker = "if proc.returncode not in accepted_exit_codes:"
    if text.count(marker) != 1:
        raise ValueError("Spirula process runner drift")
    # Record tolerated exit codes (SfM exit 3 = partial but valid model) in the log as well.
    return text.replace(marker, "log.write(f'\\nEXIT_CODE: {proc.returncode}\\n')\n            " + marker)


def installer(cells: list[str]) -> str:
    text = template_cell(cells, "def select_release_asset(")
    marker = "help_output = run_logged('train_help'"
    if text.count(marker) != 1:
        raise ValueError("Spirula installer template drift")
    return text.split(marker)[0] + (
        "\nSFM_HELP = run_logged('sfm_help', [str(BINARY), 'sfm', 'auto', '--help'], cwd=WORK, timeout=60)\n"
        "require('--quality' in SFM_HELP and '--features' in SFM_HELP, 'Spirula sfm auto --help beklenen seçenekleri listelemiyor.', 'sfm_help')\n"
        "print('Spirula', VERSION, 'hazır:', BINARY)\n"
    )


SOURCES, APP_SHA = sources()
CELLS_T = template_cells()
SPIRULA_SHA = git("rev-parse", "--short", SPIRULA_REF).strip()

BOOTSTRAP = shared_bootstrap(CELLS_T) + '''

from google.colab import drive
drive.mount('/content/drive')

WORK = Path('/content/pose_repair_work')
LOGS = WORK / 'logs'
MANIFEST_PATH = WORK / 'manifest.json'
BINARY = WORK / 'spirula'
LOGS.mkdir(parents=True, exist_ok=True)
DRIVE_OUT = Path(OUT_DIR)
DRIVE_OUT.mkdir(parents=True, exist_ok=True)
manifest = {'run': 'pose_repair', 'source_commit': COMMIT, 'variants': SFM_VARIANTS, 'status': 'starting'}


def retry_drive_export():
    """Copy local Spirula logs to Drive (also called by fail())."""
    ok, error = _copy_tree_to_drive(LOGS, DRIVE_OUT / 'spirula_logs', 'logs')
    save_manifest()
    _copy_file_to_drive(MANIFEST_PATH, DRIVE_OUT / 'spirula_manifest.json', 'manifest')
    return [] if ok else [error]


save_manifest()
print('Yerel çalışma:', WORK, '| Drive:', DRIVE_OUT)
'''

CELLS = [
    md(f"""
# Poz onarımı testi: daha iyi SfM, splat'in tutturamadığı kareleri düzeltiyor mu?

IMG_5966'da birkaç kısa bölümde (çoğu yakın plan) splat **eğitildiği kareleri bile** 11-14 dB'de render ediyor:
o karelerin kamera pozları yanlış, ve bu her eğiticiye tutarsız bilgi veriyor. Bu notebook aynı 1039 kareden
yeni SfM'ler kurar ve hepsini aynı şekilde eğitip karşılaştırır:

| SfM | ne değişiyor |
|---|---|
| `current` | dataset'teki mevcut SfM (Spirula, quality high, SIFT) — referans |
| `extreme_sift` | Spirula `--quality extreme` (daha yüksek çözünürlük, daha çok öznitelik ve çift) |
| `extreme_aliked` | + öğrenilmiş öznitelik/eşleştirici: ALIKED + LightGlue |
| `high_aliked_bottomup` | (isteğe bağlı) ALIKED + LightGlue + `--mapper bottom-up` (küçük parçalardan birleştirme) |

Her SfM için: kayıt sayısı, yeniden projeksiyon hatası, iz uzunluğu; sonra app eğiticisi (`{APP_REF}` @ `{APP_SHA}`)
+ `lr_decay` ile 30k eğitim; **aynı 93 saklanan fotoğraf** (isimle eşlenir) üzerinde PSNR/SSIM/LPIPS, ve
mevcut SfM'in kötü bölümlerindeki **eğitim karelerinin** uyumu (onarım işe yaradı mı?).

**Kullanım:** GPU runtime (Spirula için Ubuntu 24.04 tabanlı yeni Colab imajı; A100/L4/RTX PRO 6000), **High-RAM** açık
→ **Run all**. Süre: SfM başına ~30-90 dk (extreme ve ALIKED yavaştır), eğitim başına ~15 dk; toplam ~2,5-4 saat.
Her SfM ve her eğitim bitince Drive'a yazılır; kopma olursa Run all biten adımları atlar.

Spirula Studio v2026.9.24 resmi ikilisi kullanılır (kurulum hücreleri app şablonundan, `{SPIRULA_REF}` @ `{SPIRULA_SHA}`).
"""),
    code(f'''
DATASET_ZIP = "/content/drive/MyDrive/GaussianTests/inputs/IMG_5966_dataset.zip"  # @param {{type:"string"}}
OUT_DIR = "/content/drive/MyDrive/GaussianTests/pose_repair/IMG_5966"  # @param {{type:"string"}}
SFM_VARIANTS = "current,extreme_sift,extreme_aliked"  # @param {{type:"string"}}
TRAIN_STEPS = 30000  # @param {{type:"integer"}}
LONG_EDGE = 1920     # @param {{type:"integer"}}
SEED = 42
# SFM_VARIANTS'a "high_aliked_bottomup" eklenebilir (bir SfM + bir eğitim daha, ~1 saat).
SFM_VARIANTS = [v.strip() for v in SFM_VARIANTS.split(",") if v.strip()]
BAD_NAMES = {BAD_NAMES!r}
GOOD_NAMES = {GOOD_NAMES!r}
''', title="Ayarlar"),
    code(BOOTSTRAP, title="Drive + günlükler"),
    code(template_cell(CELLS_T, "libc_name, libc_version ="), title="Çalışma ortamı kontrolü (GLIBC, NVIDIA)"),
    code(installer(CELLS_T), title="Spirula kurulumu (sabitlenmiş sürüm)"),
    code(template_cell(CELLS_T, "def recover_matching_nvidia_userspace("), title="NVIDIA Vulkan kontrolü"),
    code('''
import os
os.environ["MAX_JOBS"] = "4"
!pip install -q "gsplat==1.5.3" "pytorch-msssim==1.0.0" plyfile ninja lpips psutil matplotlib
''', title="Eğitim bağımlılıkları"),
    code(f'''
import sys, types

SOURCES = json.loads({SOURCES!r})
root = Path("/content/pr_src")
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
from backend.export.to_splat import export_to_ply
from backend.preprocess.parse_colmap import scene_extent
from splat_restorer import own_splat, train_ab, pose_repair
app = types.SimpleNamespace(GaussianModel=GaussianModel, DeformationField=DeformationField, Trainer4DGS=Trainer4DGS,
                            export_to_ply=export_to_ply, scene_extent=scene_extent)
unknown = [v for v in SFM_VARIANTS if v not in pose_repair.SFM_VARIANTS]
assert not unknown, f"Bilinmeyen SfM varyantı: {{unknown}}"
print(f"{{len(SOURCES)}} dosya yazıldı; app eğiticisi {APP_REF} @ {APP_SHA}")
''', title="Kodu yaz (app eğiticisi + test kodu)"),
    code('''
import zipfile
raw = Path("/content/own_dataset")
if not (raw / ".unzipped").exists():
    print("Zip açılıyor...")
    shutil.rmtree(raw, ignore_errors=True)
    with zipfile.ZipFile(DATASET_ZIP) as z:
        z.extractall(raw)
    (raw / ".unzipped").touch()
cams_bin = next(raw.rglob("sparse/*/cameras.bin"))
SRC, CURRENT_MODEL = cams_bin.parents[2], cams_bin.parent.name
IMAGES = SRC / "images"
N_INPUT = sum(1 for p in IMAGES.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
current = own_splat.prepare_dataset(SRC, Path(f"/content/pr_prep/current_{LONG_EDGE}"), long_edge=LONG_EDGE,
                                    model_name=CURRENT_MODEL, workers=os.cpu_count() or 4)
# The fixed held-out photos: every 10th frame of the CURRENT reconstruction (the A/B notebook's 93 frames).
HOLDOUT_NAMES = current.names[::10]
FOCUS = BAD_NAMES + GOOD_NAMES
print(f"{N_INPUT} görüntü; mevcut SfM {current.num_frames} kayıtlı; saklanan {len(HOLDOUT_NAMES)}; kötü bölüm {len(BAD_NAMES)} kare")
''', title="Dataset (zip → yerel disk)"),
    code('''
SPARSE = {"current": SRC / "sparse" / CURRENT_MODEL}
SFM_STATS = {}
for name in SFM_VARIANTS:
    if name == "current":
        SFM_STATS[name] = pose_repair.sfm_stats(SPARSE[name], BAD_NAMES, HOLDOUT_NAMES, N_INPUT)
        continue
    saved = DRIVE_OUT / name / "sparse_selected"
    if (saved / "images.bin").is_file():
        SPARSE[name] = saved
        SFM_STATS[name] = json.loads((DRIVE_OUT / name / "sfm_report.json").read_text())["stats"]
        print(f"{name}: SfM daha önce bitmiş, Drive'dan alındı")
        continue
    missing = [t for t in pose_repair.SFM_REQUIRES.get(name, []) if t not in SFM_HELP]
    if missing:
        print(f"{name}: bu Spirula sürümü {missing} seçeneklerini listelemiyor; atlandı")
        continue
    ws = WORK / "sfm" / name
    shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)
    args = pose_repair.sfm_args(BINARY, IMAGES, ws, pose_repair.SFM_VARIANTS[name], GPU_SELECTOR)
    print(f"\\n===== {name}: SfM başlıyor ({' '.join(args[3:])})")
    t0 = time.monotonic()
    try:
        sfm_log = run_logged(f"sfm_{name}", args, cwd=ws, timeout=8 * 3600, accepted_exit_codes=(0, 3))
    except Exception as exc:  # one failed reconstruction must not stop the others
        print(f"{name}: SfM başarısız — {exc}")
        retry_drive_export()
        continue
    comps = pose_repair.components(ws)
    if not comps:
        print(f"{name}: SfM model üretmedi; günlük: {LOGS / ('sfm_' + name + '.log')}")
        continue
    selected = comps[0][0]
    shutil.copytree(selected, saved, dirs_exist_ok=True)
    SPARSE[name] = saved
    SFM_STATS[name] = pose_repair.sfm_stats(saved, BAD_NAMES, HOLDOUT_NAMES, N_INPUT)
    reported = re.search(r"Reprojection error:\\s*mean\\s+([\\d.]+)\\s*px", sfm_log or "")
    SFM_STATS[name]["spirula_reported_reproj_px"] = float(reported[1]) if reported else None
    SFM_STATS[name]["partial_exit3"] = "EXIT_CODE: 3" in (sfm_log or "")
    pose_repair.save_json(DRIVE_OUT / name / "sfm_report.json", {
        "args": [str(a) for a in args], "minutes": (time.monotonic() - t0) / 60,
        "components": [[c.name, n] for c, n in comps], "stats": SFM_STATS[name]})
    retry_drive_export()
    s = SFM_STATS[name]
    print(f"{name}: {(time.monotonic() - t0) / 60:.0f} dk, {s['registered']}/{N_INPUT} kayıtlı ({len(comps)} bileşen), "
          f"hata {s['reproj_error_px']:.3f} px, kötü bölümden {s['bad_registered']}/{s['bad_total']} kayıtlı")
for name, s in SFM_STATS.items():
    print(f"{name:22s} kayıtlı {s['registered']:5d}  hata {s['reproj_error_px']:.3f} px  iz {s['track_length']:.2f}  "
          f"saklanan {s['holdout_registered']}/{s['holdout_total']}")
''', title="Yeni SfM'ler (her biri ~30-90 dk)"),
    code('''
import dataclasses, lpips, torch
assert torch.cuda.is_available(), "GPU yok: Runtime → Change runtime type → GPU"
net_lpips = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
RESULTS = {}
lr_decay = dataclasses.replace(train_ab.VARIANTS["lr_decay"], steps=TRAIN_STEPS)
for name in SFM_VARIANTS:
    if name not in SPARSE:
        continue
    res_path = DRIVE_OUT / name / "result.json"
    if res_path.exists():
        RESULTS[name] = json.loads(res_path.read_text())
        print(f"{name}: eğitim daha önce bitmiş, atlandı")
        continue
    if name == "current":
        ds, data = SRC, current
        model_name = CURRENT_MODEL
    else:
        ds = pose_repair.make_dataset(IMAGES, SPARSE[name], Path("/content/pr_ds") / name)
        data = own_splat.prepare_dataset(ds, Path(f"/content/pr_prep/{name}_{LONG_EDGE}"), long_edge=LONG_EDGE,
                                         model_name="0", workers=os.cpu_count() or 4)
        model_name = "0"
    holdout, missing = pose_repair.holdout_indices(data, HOLDOUT_NAMES)
    xyz, rgb, init_ids = train_ab.init_points(ds, model_name, SEED)
    print(f"\\n===== {name}: {data.num_frames} kare, saklanan {len(holdout)} (kayıtsız {len(missing)}), {lr_decay.steps:,} adım")
    work = Path("/content/pr_work") / name
    shutil.rmtree(work, ignore_errors=True)
    ply, info = train_ab.train_variant(lr_decay, data, data.frames_dir, xyz, rgb, init_ids, holdout, work, app, seed=SEED)
    (DRIVE_OUT / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ply, DRIVE_OUT / name / "final.ply")
    frames = train_ab.evaluate_ply(ply, data, holdout, None, net_lpips, work / "renders")
    fit = pose_repair.frame_fit(ply, data, set(FOCUS), DRIVE_OUT / name / "focus")
    RESULTS[name] = {"frames": frames, "names": data.names, "fit": fit, "info": info,
                     "holdout_missing": missing, "stats": SFM_STATS[name]}
    pose_repair.save_json(res_path, RESULTS[name])
    row = pose_repair.summarise_variant(name, SFM_STATS[name], frames, fit,
                                        types.SimpleNamespace(names=data.names), HOLDOUT_NAMES, BAD_NAMES, info)
    print(f"{name}: PSNR {row['psnr']:.2f}  LPIPS {row['lpips']:.4f}  kötü bölüm eğitim-karesi {row['bad_fit_psnr']:.2f} dB  "
          f"<18 dB kare {row['frames_below_18db']}")
    if name != "current":  # free local disk; the current set is kept for the photo tiles below
        shutil.rmtree(Path(f"/content/pr_prep/{name}_{LONG_EDGE}"), ignore_errors=True)
''', title="Eğit + ölç (her SfM için ~15 dk)"),
    code('''
from IPython.display import Image as IPImage, Markdown, display
rows = [pose_repair.summarise_variant(n, r["stats"], r["frames"], r["fit"], types.SimpleNamespace(names=r["names"]),
                                      HOLDOUT_NAMES, BAD_NAMES, r["info"]) for n, r in RESULTS.items()]
md_text = pose_repair.markdown(rows, len(HOLDOUT_NAMES), len(BAD_NAMES))
(DRIVE_OUT / "summary.md").write_text(md_text, encoding="utf-8")
pose_repair.save_json(DRIVE_OUT / "summary.json", rows)
display(Markdown(md_text))
order = sorted(p.name for p in IMAGES.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
plot = pose_repair.strip_plot({n: r["fit"] for n, r in RESULTS.items()}, order, BAD_NAMES, DRIVE_OUT / "frame_fit.png")
display(IPImage(filename=str(plot), width=1600))
index = {n: i for i, n in enumerate(current.names)}
photos = {n: own_splat._to_u8(current.photo(index[n], "cpu")) for n in FOCUS if n in index}
sheets = pose_repair.focus_sheets(DRIVE_OUT, list(RESULTS), photos, DRIVE_OUT)
print("Sütunlar: foto | " + " | ".join(RESULTS) + "  (her SfM kendi kamerasıyla; alt sıra zoom)")
for p in sheets[:6]:
    display(IPImage(filename=str(p), width=1600))
''', title="Sonuçlar"),
    md("""
## Nasıl okunur?

* **kötü bölüm eğitim-karesi PSNR:** `current` satırında ~12-15 dB beklenir. Yeni bir SfM bunu belirgin yükseltiyorsa
  (ör. 22+ dB) pozlar onarılmış demektir; şerit grafikte kırmızı noktalar kaybolur.
* **PSNR / LPIPS (93 saklanan kare):** sahnenin geneli. Poz onarımı kötü bölümlerdeki tutarsız bilgiyi kaldırırsa
  genel kalite de artabilir (floater ve bulanıklık azalır).
* **kayıtlı:** kayıtlı kare sayısı ve kötü bölümdeki 17 karenin kaçının kaydedildiği. Kaydedilemeyen saklanan
  fotoğraflar ortalamaya girmez; tabloda ayrıca yazılır.

Bana `summary.md`, `frame_fit.png` ve birkaç `focus_*.jpg` getirmen yeterli (Drive'da `OUT_DIR` altında).
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
            "spirula_template": {"ref": SPIRULA_REF, "commit": SPIRULA_SHA},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    for cell in CELLS:
        if cell["cell_type"] == "code":
            ast.parse("\n".join(l for l in "".join(cell["source"]).splitlines() if not l.lstrip().startswith("!")))
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB); app {APP_REF} @ {APP_SHA}, spirula template @ {SPIRULA_SHA}")


if __name__ == "__main__":
    main()

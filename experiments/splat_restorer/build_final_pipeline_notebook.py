"""Build colab/final_pipeline_test.ipynb: every remaining pipeline experiment on one new capture.

Two videos (a training walk and a separate test walk) become one image folder;
the app's Spirula recipe (APP_REF) runs its extreme SfM and geometry on it
unchanged, then the candidates train without the test walk and are scored on it:

    python experiments/splat_restorer/build_final_pipeline_notebook.py
"""
from __future__ import annotations

import ast
import json

from build_gsplat_testbed_notebook import FUSED_SSIM, GSPLAT_TAG, NERFVIEW, PYCOLMAP
from build_notebook import PKG, code, md
from build_spirula_quality_notebook import APP_REF, TEMPLATE, git, package_sources, replace_assignments

REPO = PKG.parents[2]
OUT = REPO / "colab" / "final_pipeline_test.ipynb"
DATASET_FOLDER = "/content/room2_dataset"
NODE = "v22.12.0"
SPLAT_TRANSFORM = "@playcanvas/splat-transform@3.10.1"

SETTINGS = {  # values for the template's own settings cell
    "INPUT_MODE": "dataset_folder",
    "INPUT_DATASET_FOLDER": DATASET_FOLDER,
    "CAPTURE_KIND": "video",
    "GENERATE_DEPTH": True,
    "ALLOW_PARTIAL_RECONSTRUCTION": False,
    "RESUME_CHECKPOINT": "",
}


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
    s += "\nTRAIN_QUALITY = 'high'\nSFM_QUALITY = 'extreme'  # dataset klasöründe SfM yoksa; Drive'da kayıtlı SfM varsa o kullanılır\n"
    cells[settings_idx]["source"] = s.splitlines(keepends=True)
    middle = cells[settings_idx + 1:train_idx]
    while middle[-1]["cell_type"] == "markdown":  # the template's heading for its training step
        middle.pop()
    prep = "".join(middle[0]["source"])
    assert "DATASET_FOLDER.is_dir()" in prep and "drive.mount" in prep, "Template drift: setup cell moved"
    return cells[1:settings_idx + 1], middle, sha


def build() -> dict:
    head, middle, sha = template_cells()
    intro = md(f"""
# Final pipeline testi: ROOM2, tüm adaylar aynı test yürüyüşünde

Eğitim videosu ve **ayrı çekilen test yürüyüşü** tek klasöre örneklenir (`train_*.png`, `eval_*.png`); her pencereden
**en keskin kare** seçilir. App'in Spirula tarifi (`{APP_REF}` @ `{sha}`: v2026.9.24 ikilisi, Vulkan, **extreme SfM**,
MoGe-2 geometri) aynen çalışır, iki yürüyüş aynı SfM'e girer. Adaylar test karelerini **hiç görmeden** eğitilir
(Spirula `--eval-mode filename`, gsplat isimle hariç tutar) ve hepsi aynı test karelerinde puanlanır.

| aday | ne | süre (RTX PRO 6000) |
|---|---|---|
| `spirula_high` | Spirula high 3M / 50k, renk telafisi açık: **mevcut finalist, taban** | ~50 dk |
| `gsplat_default` | gsplat varsayılanı 30k, 1920 (referans eğitici) | ~20 dk |
| `spirula_high_floater` | high + floater bastırma mild (renk telafisi açık, tek değişiklik) | ~50 dk |
| `spirula_ultra` | Spirula ultra 10M / 80k | ~2,5-3 sa |
| `spirula_high_no_appearance_floater` *(varsayılan kapalı)* | high + renk telafisi kapalı + floater mild | ~60 dk |

Sonra **kazanan** (en düşük LPIPS ya da `POSTPROCESS_VARIANT`) için: splat-transform ile **floater temizliği**, **SPZ** ve
**SOG** sıkıştırma; her sürüm aynı test karelerinde ölçülür ve fly-through videosu alınır. Dosyalar SuperSplat'te açılabilir.

**Kullanım:** Runtime → **RTX PRO 6000** (yoksa **A100 80 GB**; ultra 40 GB+ ister), **High-RAM** → **Run all**.
Toplam ~7-9 saat. Her aday bitince Drive'a yazılır; kopma olursa Run all biten adayları atlar, SfM'i Drive'dan geri yükler.
""")
    settings = code(f'''
TRAIN_VIDEO = "/content/drive/MyDrive/GaussianTests/inputs/ROOM2_train.mov"  # @param {{type:"string"}}
TEST_VIDEO = "/content/drive/MyDrive/GaussianTests/inputs/ROOM2_test.mov"  # @param {{type:"string"}}
OUT_DIR = "/content/drive/MyDrive/GaussianTests/final_pipeline/ROOM2"  # @param {{type:"string"}}
VARIANTS = "spirula_high,gsplat_default,spirula_high_floater,spirula_ultra"  # @param {{type:"string"}}
TRAIN_FPS = 4.0   # @param {{type:"number"}}
TEST_FPS = 2.0    # @param {{type:"number"}}
POSTPROCESS_VARIANT = ""  # @param {{type:"string"}}
FLY_FRAMES = 450  # @param {{type:"integer"}}
# TRAIN_FPS / TEST_FPS: saniyede kaç kare; her 30/fps karelik pencereden en keskini alınır.
# POSTPROCESS_VARIANT: boşsa en düşük LPIPS'li aday temizlenir/sıkıştırılır.
# "spirula_high_no_appearance_floater" VARIANTS'a eklenebilir (~60 dk).
VARIANTS = [v.strip() for v in VARIANTS.split(",") if v.strip()]
DATASET_FOLDER_LOCAL = "{DATASET_FOLDER}"
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
from splat_restorer import final_pipeline as fp, spirula_quality as sq
CANDIDATES = fp.candidates()
_unknown = [v for v in VARIANTS if v not in CANDIDATES]
assert not _unknown, f"Bilinmeyen aday: {{_unknown}}; seçenekler: {{list(CANDIDATES)}}"
_spirula = [CANDIDATES[v].spirula for v in VARIANTS if CANDIDATES[v].trainer == "spirula"]
# Şablonun bellek ön kontrolü en ağır seçili Spirula adayına göre yapılsın.
TRAIN_ITERATIONS = max((v.iterations for v in _spirula), default=50000)
TRAIN_CAP_MAX = max((v.cap for v in _spirula), default=3_000_000)
MIN_TRAIN_VRAM_GIB = max((v.min_gib for v in _spirula), default=22)
print("Adaylar:", VARIANTS, "| en ağır Spirula:", TRAIN_ITERATIONS, "adım,", TRAIN_CAP_MAX, "splat,", MIN_TRAIN_VRAM_GIB, "GiB")
''', title="Test kodunu yaz")
    install = code(f'''
import os, re, shutil, subprocess, torch
assert torch.cuda.is_available(), "GPU yok: Runtime → Change runtime type → GPU"
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), "torch", torch.__version__, torch.version.cuda)
# Blackwell (sm_120) için gsplat'in JIT derlemesi CUDA 12.8+ nvcc ister; Colab'ın nvcc'si eskiyse 12.8 kurulur.
def _nvcc_version():
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    try:
        m = re.search(r"release (\\d+)\\.(\\d+)", subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    except FileNotFoundError:
        return (0, 0)
if torch.cuda.get_device_capability()[0] >= 12 and _nvcc_version() < (12, 8):
    if not os.path.exists("/usr/local/cuda-12.8/bin/nvcc"):
        _ubuntu = subprocess.run(["bash", "-c", ". /etc/os-release && echo ubuntu${{VERSION_ID//./}}"], capture_output=True, text=True).stdout.strip()
        !wget -q https://developer.download.nvidia.com/compute/cuda/repos/{{_ubuntu}}/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        !dpkg -i /tmp/cuda-keyring.deb > /dev/null && apt-get update -qq && apt-get install -y -qq cuda-toolkit-12-8 > /dev/null
    os.environ["CUDA_HOME"] = "/usr/local/cuda-12.8"
    os.environ["PATH"] = "/usr/local/cuda-12.8/bin:" + os.environ["PATH"]
print("nvcc", _nvcc_version())
os.environ["MAX_JOBS"] = "4"
!pip install -q "gsplat=={GSPLAT_TAG[1:]}" ninja lpips plyfile "imageio[ffmpeg]"
if "gsplat_default" in VARIANTS:
    if not os.path.exists("/content/gsplat_src"):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", "{GSPLAT_TAG}",
                        "https://github.com/nerfstudio-project/gsplat.git", "/content/gsplat_src"], check=True)
    !pip install -q tyro viser scikit-learn tqdm "torchmetrics[image]" tensorboard tensorly pyyaml matplotlib splines "{PYCOLMAP}" "{NERFVIEW}"
    !pip install -q --no-build-isolation "{FUSED_SSIM}"
    from splat_restorer import gsplat_testbed as tb
    tb.fix_gsplat_install(Path("/content/gsplat_src/examples"))
# gsplat'i şimdi derle: hata saatler sonra ölçüm sırasında değil burada çıksın.
from gsplat import rasterization
_d = "cuda"
_img, _, _ = rasterization(torch.zeros(1, 3, device=_d), torch.tensor([[1., 0, 0, 0]], device=_d), torch.full((1, 3), .1, device=_d),
                           torch.ones(1, device=_d), torch.ones(1, 3, device=_d), torch.eye(4, device=_d)[None],
                           torch.tensor([[[100., 0, 32], [0, 100., 32], [0, 0, 1]]], device=_d), 64, 64)
print("gsplat derlendi ve çalışıyor")
# splat-transform (floater temizliği + SPZ/SOG) Node 22+ ister.
def _node_major():
    try:
        return int(subprocess.run(["node", "-v"], capture_output=True, text=True).stdout.strip().lstrip("v").split(".")[0])
    except (FileNotFoundError, ValueError):
        return 0
if _node_major() < 22:
    if not os.path.exists("/content/node22/bin/node"):
        !wget -q https://nodejs.org/dist/{NODE}/node-{NODE}-linux-x64.tar.xz -O /tmp/node.tar.xz
        !mkdir -p /content/node22 && tar -xJf /tmp/node.tar.xz -C /content/node22 --strip-components=1
    os.environ["PATH"] = "/content/node22/bin:" + os.environ["PATH"]
!npm install -g -s {SPLAT_TRANSFORM}
!splat-transform --version
''', title="Kurulum: gsplat (ölçüm + referans eğitici), splat-transform (~5-10 dk)")
    extract = code('''
import shutil, time
from google.colab import drive
drive.mount("/content/drive")
out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
images_local = Path(DATASET_FOLDER_LOCAL) / "images"
local_videos = Path("/content/videos"); local_videos.mkdir(exist_ok=True)
frame_info = {}
for prefix, video, fps in ((fp.TRAIN_PREFIX, TRAIN_VIDEO, TRAIN_FPS), (fp.EVAL_PREFIX, TEST_VIDEO, TEST_FPS)):
    local = local_videos / Path(video).name
    if not local.exists() or local.stat().st_size != Path(video).stat().st_size:
        print("Video yerel diske kopyalanıyor:", video, flush=True)
        shutil.copy2(video, local)
    frame_info[prefix] = fp.extract_sharpest(local, images_local, prefix, fps)
fp.save_json(out_dir / "frames.json", frame_info)
display_plot = fp.sharpness_plot({"eğitim": frame_info[fp.TRAIN_PREFIX], "test": frame_info[fp.EVAL_PREFIX]},
                                  out_dir / "sharpness.png")
# Önceki oturumun SfM'i Drive'daysa ve aynı kareler seçildiyse geri yükle: şablon onu kullanır (extreme SfM tekrar koşmaz).
saved_sfm = out_dir / "sfm" / "sparse_0"
saved_frames = out_dir / "sfm" / "frames.json"
if (saved_sfm / "images.bin").exists() and saved_frames.exists():
    same = all(json.loads(saved_frames.read_text())[k]["picked"] == frame_info[k]["picked"] for k in frame_info)
    if same and not (Path(DATASET_FOLDER_LOCAL) / "sparse" / "0" / "images.bin").exists():
        shutil.copytree(saved_sfm, Path(DATASET_FOLDER_LOCAL) / "sparse" / "0", dirs_exist_ok=True)
    print("Drive'daki SfM", "geri yüklendi" if same else "farklı karelere ait, KULLANILMIYOR (yeniden SfM)")
INPUT_DATASET_FOLDER = DATASET_FOLDER_LOCAL
print(f"Kareler: eğitim {len(frame_info[fp.TRAIN_PREFIX]['picked'])}, test {len(frame_info[fp.EVAL_PREFIX]['picked'])} → {images_local}")
''', title="Kareleri çıkar: her pencerenin en keskini (eğitim + test yürüyüşü)")
    save_sfm = code('''
# Şablonun seçtiği SfM modelini Drive'a sakla (kopmadan sonra Run all extreme SfM'i tekrar koşmasın).
if not SFM_REUSED:
    fp.copy_tree(RECON_DIR, out_dir / "sfm" / "sparse_0")
    fp.save_json(out_dir / "sfm" / "frames.json", frame_info)
    shutil.copy2(DATASET / "sfm_report.json", out_dir / "sfm" / "sfm_report.json")
    print("SfM Drive'a kaydedildi:", out_dir / "sfm")
''', title="SfM'i Drive'a kaydet")
    data_prep = code('''
from splat_restorer import own_splat, pose_repair, quality_combo as qc
data = own_splat.prepare_dataset(GEOMETRY_DATASET, Path("/content/fp_prepared_1920"), long_edge=1920,
                                 model_name="0", workers=os.cpu_count() or 4)
eval_ids = fp.eval_indices(data)
split = fp.registered_split(IMAGE_DIR, data)
fp.save_json(out_dir / "split.json", split)
print(f"Ölçüm: {data.width}x{data.height}; {split}")
assert len(eval_ids) >= 20, "Test yürüyüşünden çok az kare kaydedildi: SfM raporuna bak"
fly_names = fp.train_names(data)   # fly-through: eğitim yürüyüşünün yolu, video sırasıyla
fly_poses = qc.flythrough_path(data, fly_names, n_frames=FLY_FRAMES)
if "gsplat_default" in VARIANTS:
    gs_dataset = pose_repair.make_dataset(IMAGE_DIR, RECON_DIR, Path("/content/fp_gs_dataset"))
    tb.prepare_factor_images(gs_dataset, CANDIDATES["gsplat_default"].gs.factor, workers=os.cpu_count() or 8)
''', title="Ölçüm verisi: test kareleri (isimle)")
    loop = code('''
import lpips, re, subprocess, time, traceback
net_lpips = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
results, failed, fly_dirs = {}, [], {}
eval_names = [data.names[i] for i in eval_ids]

def flythrough(name, ply):
    frames_dir = Path("/content/fly") / name
    params, deg = own_splat.load_ply(ply, "cuda")
    qc.render_flythrough(params, deg, data, fly_poses, frames_dir)
    del params; torch.cuda.empty_cache()
    fly_dirs[name] = frames_dir
    mp4 = out_dir / f"flythrough_{name}.mp4"
    if not mp4.exists():
        qc.write_video({name: frames_dir}, mp4)

def train_spirula(c, vdir):
    v = c.spirula
    prefix = OUT / c.name
    shutil.rmtree(prefix, ignore_errors=True)
    args = sq.train_args(BINARY, TRAIN_DATASET, TRAIN_RECON_DIR, prefix, v, GPU_SELECTOR, GENERATE_DEPTH,
                         split=fp.FILENAME_SPLIT)
    log_path = vdir / "train.log"
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
            elif not m and re.search(r"error|warn|eval|psnr|split|held", line, re.I):
                print(line.rstrip()[:200], flush=True)
        code_ = proc.wait()
    if code_ != 0:
        raise RuntimeError(f"Spirula çıkış kodu {code_}; günlük: {log_path}\\n" + "".join(open(log_path).readlines()[-25:]))
    found = sq.find_outputs(prefix)
    if found["ply"] is None:
        raise RuntimeError(f"splat.ply bulunamadı ({prefix})")
    shutil.copy2(found["ply"], vdir / "splat.ply")
    for key in ("metrics", "config"):
        if found[key] is not None:
            shutil.copy2(found[key], vdir / f"spirula_{key}.json")
    # Spirula'nın sakladığı kare sayısı = kaydedilen test kareleri olmalı (filename bölmesi çalıştı mı?).
    n_gt = len(found["eval_gt"])
    print(f"Spirula'nın sakladığı kare: {n_gt} (kaydedilen test karesi {len(eval_ids)})"
          + ("" if n_gt == len(eval_ids) else "  !!! UYUŞMUYOR, train.log'a bak"), flush=True)
    if n_gt == 0:
        raise RuntimeError("Spirula hiç kare saklamadı: test kareleri eğitime girmiş olabilir (--eval-mode filename)")
    native = sq.native_metrics(found["metrics"])  # klasör silinmeden önce oku
    shutil.rmtree(prefix, ignore_errors=True)  # yerel diski boşalt
    return {"native": native, "spirula_held_out": n_gt}

# Uzun eğitimlerden önce birkaç dakikalık deneme: Spirula test karelerini isimle gerçekten dışarıda tutuyor mu?
check_path = out_dir / "split_check.json"
if any(CANDIDATES[n].trainer == "spirula" for n in VARIANTS) and not check_path.exists():
    print("Bölme denemesi (300 adım, 1/4 çözünürlük)...", flush=True)
    _probe = fp.Candidate("split_check", "spirula", "bölme denemesi", spirula=sq.Variant(
        "split_check", quality="medium", iterations=300, cap=100_000, flags={"train_resolution_divisor": "4"}))
    (out_dir / "split_check").mkdir(exist_ok=True)
    _r = train_spirula(_probe, out_dir / "split_check")
    fp.save_json(check_path, {"held_out": _r["spirula_held_out"], "eval_registered": len(eval_ids)})
    print("Bölme denemesi geçti.", flush=True)

def train_gsplat(c, vdir):
    run_dir = Path("/content/gs_runs") / c.name
    shutil.rmtree(run_dir, ignore_errors=True)
    # test_every çok büyük: gsplat'in kendi bölmesi kapalı, test kareleri isimle hariç.
    tb.run_variant(c.gs, gs_dataset, Path("/content/gsplat_src/examples"), run_dir, c.steps, 1_000_000, eval_names)
    params, deg = tb.load_checkpoint(run_dir)
    tb.save_ply(params, vdir / "splat.ply")
    del params; torch.cuda.empty_cache()
    shutil.copy2(run_dir / "train.log", vdir / "train.log")
    shutil.rmtree(run_dir, ignore_errors=True)
    return {}

for name in VARIANTS:
    c = CANDIDATES[name]
    vdir = out_dir / name
    res_path = vdir / "result.json"
    if res_path.exists():
        results[name] = json.loads(res_path.read_text())
        if (vdir / "splat.ply").exists():
            flythrough(name, vdir / "splat.ply")
        print(f"{name}: daha önce bitmiş, atlandı")
        continue
    print(f"\\n===== {name}: {c.note}", flush=True)
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "failed.txt").unlink(missing_ok=True)
    t0 = time.monotonic()
    try:
        extra = train_spirula(c, vdir) if c.trainer == "spirula" else train_gsplat(c, vdir)
    except Exception as e:   # bir aday çökerse diğerleri yine koşsun
        print(f"!!! {name} BAŞARISIZ, atlanıyor:\\n{e}")
        (vdir / "failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
        failed.append(name)
        continue
    minutes = (time.monotonic() - t0) / 60
    # Spirula PLY'leri ayrıca antialiased render ile de ölçülür (ölçücü teşhisi).
    modes = ("classic", "antialiased") if c.trainer == "spirula" else ("classic",)
    r = fp.evaluate_ply(vdir / "splat.ply", data, eval_ids, net_lpips, vdir / "renders", modes)
    r.update(extra, note=c.note, minutes=minutes, candidate=fp.candidate_dict(c), file_mb=fp.file_mb(vdir / "splat.ply"))
    fp.save_json(res_path, r)
    results[name] = r
    s = fp.summarise({name: r})[name]
    print(f"{name}: LPIPS {s['classic_raw_lpips']:.4f}  PSNR {s['classic_raw_psnr']:.2f}  renk-eşitlemeli "
          f"{s['classic_cc_psnr']:.2f}  {r['splats']:,} splat  {minutes:.0f} dk", flush=True)
    flythrough(name, vdir / "splat.ply")
''', title="Adayları eğit + ölç + fly-through")
    post = code('''
summary = fp.summarise(results)
trained = [n for n in VARIANTS if n in results]
assert trained, "Hiçbir aday bitmedi: <aday>/failed.txt dosyalarına bak"
winner = POSTPROCESS_VARIANT or fp.best_by_lpips(summary, trained)
src = out_dir / winner / "splat.ply"
pdir = out_dir / "postprocess" / winner
pdir.mkdir(parents=True, exist_ok=True)
plog = pdir / "splat_transform.log"
post_path = pdir / "post.json"
post = json.loads(post_path.read_text()) if post_path.exists() else None
if post is None:
    print(f"Son işlem kaynağı: {winner}", flush=True)
    notes, versions = [], {"orijinal": (src, src)}   # ad -> (ölçülecek PLY, boyutu raporlanan dosya)
    extent = fp.scene_extent(src)
    for vname, opacity in (("temiz", 0.1), ("temiz_sıkı", 0.5)):   # sıkı: voksel ancak opaklık 0,5'te "dolu"
        f_arg = fp.floater_args(extent, opacity)
        clean_v = pdir / ("clean.ply" if vname == "temiz" else "clean_strict.ply")
        for gpu in ([], ["-g", "cpu"]):   # önce GPU (WebGPU/Vulkan), olmazsa CPU
            ok, tail = fp.splat_transform([*gpu, src, "-F", f_arg, clean_v], plog)
            if ok:
                notes.append(f"{vname}: -F {f_arg} (sahne köşegeni {extent:.2f} birim){' [cpu]' if gpu else ''}")
                versions[vname] = (clean_v, clean_v)
                break
        if not ok:
            notes.append(f"{vname} ÇALIŞMADI (splat_transform.log): " + tail[-300:])
    clean = pdir / "clean.ply"
    ok = "temiz" in versions
    base_name, base = ("temiz", clean) if ok else ("orijinal", src)
    for fmt in ("spz", "sog"):
        packed, back = pdir / f"{base.stem}.{fmt}", pdir / f"{base.stem}_{fmt}.ply"
        done = False
        for gpu in ([], ["-g", "cpu"]):
            if fp.splat_transform([*gpu, base, packed], plog)[0] and fp.splat_transform([*gpu, packed, back], plog)[0]:
                versions[f"{base_name}+{fmt}"] = (back, packed)
                done = True
                break
        if not done:
            notes.append(f"{fmt.upper()} dönüşümü ÇALIŞMADI (splat_transform.log)")
    rows = {}
    for vname, (ply, sized) in versions.items():
        r = fp.evaluate_ply(ply, data, eval_ids, net_lpips, pdir / "renders" / vname, ("classic",))
        rows[vname] = {"lpips": fp._mean(r["modes"]["classic"], "raw", "lpips"),
                       "psnr": fp._mean(r["modes"]["classic"], "raw", "psnr"),
                       "splats": r["splats"], "file_mb": fp.file_mb(sized), "ply": str(ply)}
        print(f"  {vname}: LPIPS {rows[vname]['lpips']:.4f}  PSNR {rows[vname]['psnr']:.2f}  "
              f"{r['splats']:,} splat  {rows[vname]['file_mb']:.1f} MB", flush=True)
    post = {"source": winner, "rows": rows, "notes": notes}
    fp.save_json(post_path, post)
for vname, row in post["rows"].items():
    if vname != "orijinal" and Path(row["ply"]).exists():
        flythrough(f"{winner}__{vname}", Path(row["ply"]))
''', title="Kazanan: floater temizliği + SPZ/SOG, aynı karelerde ölç")
    results_cell = code('''
import base64
from IPython.display import HTML, Image as IPImage, Markdown, display
if failed:
    print("Başarısız adaylar (ayrıntı Drive'da <aday>/failed.txt):", failed)
for n in trained:   # eski koşularda native boş kalmıştı: Drive'daki kopyadan doldur
    if not summary[n]["native"] and (out_dir / n / "spirula_metrics.json").exists():
        summary[n]["native"] = sq.native_metrics(out_dir / n / "spirula_metrics.json")
baseline = "spirula_high" if "spirula_high" in summary else trained[0]
md_text = fp.markdown(summary, split, baseline, post)
(out_dir / "summary.md").write_text(md_text, encoding="utf-8")
fp.save_json(out_dir / "summary.json", summary)
display(Markdown(md_text))
display(IPImage(filename=str(out_dir / "sharpness.png"), width=1400))
sheets = fp.comparison_sheets({n: results[n] for n in trained}, data, eval_ids, out_dir, out_dir, baseline)
print("Sütunlar: foto | " + " | ".join(trained) + "; alt sıra: kırmızı kutunun zoom'u")
for p in sheets:
    display(IPImage(filename=str(p), width=1600))
videos = {"flythrough_compare.mp4": [n for n in trained if n in fly_dirs][:4],
          "flythrough_postprocess.mp4": [n for n in fly_dirs if n == winner or n.startswith(winner + "__")][:4]}
for fname, names in videos.items():
    if len(names) < 2:
        continue
    mp4 = qc.write_video({n: fly_dirs[n] for n in names}, out_dir / fname)
    print(fname, "→", names)
    if mp4.stat().st_size < 40 * 2**20:
        b64 = base64.b64encode(mp4.read_bytes()).decode()
        display(HTML(f'<video controls loop width="1400" src="data:video/mp4;base64,{b64}"></video>'))
''', title="Sonuçlar + karşılaştırma videoları")
    closing = md("""
## Nasıl okunur?

* **LPIPS** (düşük iyi) asıl ölçüt; ~0,005 altı farklar gürültü sayılabilir. **PSNR** yardımcı; Spirula PLY'lerinde bizim
  ölçücü PSNR'ı düşük gösteriyordu, **antialiased teşhis** tablosu bunun nedenini ayırır.
* `spirula_ultra` vs `spirula_high` → kapasitenin etkisi; `spirula_high_floater` vs `spirula_high` → floater bastırma;
  `gsplat_default` → Spirula'nın bizim referans eğiticiye üstünlüğü.
* **Son işlem** tablosu: temizlik ve sıkıştırmanın kaliteye maliyeti ve dosya boyutu. SuperSplat'te bakmak için
  `postprocess/<kazanan>/clean.ply`, `.spz`, `.sog` dosyaları.
* Görsel karar: `flythrough_compare.mp4` (adaylar) ve `flythrough_postprocess.mp4` (kazananın sürümleri).

Bana haber vermen yeterli; `summary.md`, kareler ve videolar Drive'da `OUT_DIR` altında.
""")
    cells = [intro, *head, settings, sources, install, extract, *middle, save_sfm, data_prep, loop, post,
             results_cell, closing]
    for c in cells:
        if c["cell_type"] == "code":
            lines = "".join(c["source"]).splitlines()  # shell lines -> `pass` at the same indent
            ast.parse("\n".join(l[:len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith(("!", "%")) else l
                                for l in lines))
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

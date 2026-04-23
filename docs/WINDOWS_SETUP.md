# Windows Setup Guide

Bu doküman, 4DGS Studio'yu Windows'ta sıfırdan kurarken karşılaşılan gerçek sorunları ve çözümlerini belgeler. Linux'ta çoğu şey "just works" — Windows'ta değil.

Doğrulanmış çalışan konfigürasyon:

- **OS:** Windows 10/11
- **GPU:** RTX 3060 Ti 8GB (compute capability 8.6)
- **CUDA Toolkit:** 12.6 (nvcc)
- **Python:** 3.10 (conda env: `gs4d`)
- **PyTorch:** 2.5.1+cu124
- **gsplat:** 1.5.3
- **Visual Studio:** 2022 Build Tools (MSVC 19.38)

---

## 1. Ön koşullar

### 1.1 Visual Studio 2022 Build Tools

gsplat'in CUDA extension'ı Windows'ta **JIT compile** ile derlenir. Bunun için `cl.exe` (MSVC compiler) gerekli.

[Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) indir ve kurarken **"Desktop development with C++"** workload'ını seç (en azından MSVC v143 + Windows SDK). Full VS IDE gerekmez, sadece Build Tools yeterli.

Kurulum sonrası `cl.exe` yolu şuna benzer:

```
C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64\cl.exe
```

### 1.2 CUDA Toolkit 12.6

NVIDIA'nın [CUDA Toolkit 12.6 installer](https://developer.nvidia.com/cuda-12-6-0-download-archive)'ı. Kurulum sonrası:

```cmd
nvcc --version
```

`Cuda compilation tools, release 12.6` görmeli.

### 1.3 Conda / Miniconda

[Miniconda](https://docs.conda.io/en/latest/miniconda.html) veya Anaconda.

### 1.4 COLMAP

Conda ortamı kurulumu sırasında conda-forge üzerinden otomatik gelir (`environment.yml`).

---

## 2. Kurulum adımları

### 2.1 Conda ortamı

```cmd
conda env create -f environment.yml
conda activate gs4d
```

### 2.2 PyTorch (ayrı kurulum)

**KRİTİK:** torch versiyonu ve CUDA wheel'i çok titiz seçilmeli.

```cmd
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

Neden `cu124` ama `nvcc 12.6`?
CUDA wheel'leri **forward-compatible**: cu124 binary'si cu126 runtime'ında sorunsuz çalışır. PyTorch 2.5.x henüz cu126 wheel'i yayınlamadı (bu dokümanın yazıldığı tarih itibarıyla).

**Yapma:**
- `--pre` flag'i ile pre-release torch yükleme (`torch 2.11.0+cu126` gsplat 1.5.3 ile uyumsuz; `CUDACachingAllocator.h` parse hatası verir)
- Eski cu118 wheel'i (gsplat 1.5.3 modern CUDA header'larla compile oluyor, cu118 uyumlu değil)

Doğrulama:

```cmd
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Beklenen: `2.5.1+cu124 12.4 True`

### 2.3 Geri kalan paketler

```cmd
pip install -r requirements.txt
```

---

## 3. gsplat CUDA extension — JIT compile

gsplat 1.5.3'ün Windows wheel'inde prebuilt CUDA extension yok. İlk `import gsplat` çağrısında `torch.utils.cpp_extension.load()` ile **JIT compile** yapar. Bu ~2-3 dakika sürer.

### 3.1 gsplat'i import ederken "x64 Native Tools Command Prompt" KULLANMA ZORUNLULUĞU

Normal `cmd.exe` ya da PowerShell'de `cl.exe` PATH'te olmadığı için compile fail eder. İki yol var:

**Yol A (önerilen):** Start menüsünden **"x64 Native Tools Command Prompt for VS 2022"**'yi aç. Bu shell `cl.exe`'yi, Windows SDK include'larını, CUDA PATH'ini otomatik ayarlar. Conda env'i bu shell'de aktive et:

```cmd
conda activate gs4d
```

**Yol B:** Normal cmd'de `vcvars64.bat`'i source'la (tek seferlik per shell):

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
conda activate gs4d
```

### 3.2 Bilinen sorun: `-Wno-attributes` bug'ı (gsplat 1.5.3)

gsplat 1.5.3'ün `cuda/_backend.py` dosyası C++ compile flag'lerine `-Wno-attributes` ekliyor — bu bir **GCC flag'i**, MSVC tanımıyor. MSVC şu hatayı verir:

```
cl: Komut satırı error D8021: geçersiz sayısal bağımsız değişken '/Wno-attributes'
```

**Fix (tek satır patch):**

```cmd
python -c "p=r'E:\anaconda3\envs\gs4d\lib\site-packages\gsplat\cuda\_backend.py'; s=open(p,encoding='utf-8').read(); s2=s.replace('extra_cflags = [opt_level, \"-Wno-attributes\"]', 'extra_cflags = [opt_level]'); open(p,'w',encoding='utf-8').write(s2); print('Patched:', s!=s2)"
```

(Yukardaki `E:\anaconda3\envs\gs4d\...` yolunu kendi kurulumuna göre güncelle.)

Bulmak için:

```cmd
findstr /S /N "Wno-attributes" %CONDA_PREFIX%\lib\site-packages\gsplat\*.py
```

Patch sonrası cache'i temizle (yoksa eski bytecode'u kullanır):

```cmd
del /F /Q %CONDA_PREFIX%\lib\site-packages\gsplat\cuda\__pycache__\_backend.cpython-310.pyc
rmdir /S /Q "%LOCALAPPDATA%\torch_extensions" 2>nul
rmdir /S /Q "%USERPROFILE%\.cache\torch_extensions" 2>nul
```

İlk pipeline çalıştırması gsplat'i compile edecek (`gsplat: CUDA extension has been set up successfully in ~150 seconds`). Sonraki çalıştırmalar cache'ten anında yükler.

---

## 4. Türkçe karakter / stdout redirect sorunu

Pipeline çıktısında `→ ✓ ⚠` gibi Unicode karakterler var. Windows default code page'i `cp1254` (Turkish Windows) ya da `cp1252` — bu karakterleri encode edemez. Normal konsol çıktısında Windows tolerans gösterir, ama **stdout'u dosyaya redirect ederken** (örn. `> build_log.txt`) şu hatayı verir:

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position ...
```

Fix — her shell oturumunda:

```cmd
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
```

Kalıcı yapmak için sistem environment variable'larına ekle.

---

## 5. COLMAP sub-model seçimi

Bazı video'larda (yavaş hareket, az feature) COLMAP incremental mapper **2+ ayrı model** üretir. Pipeline'ımız bunu tespit edip **en büyük** parçayı seçer (dosya boyutuna göre: cameras.bin + images.bin + points3D.bin toplam MB):

```
⚠ COLMAP 2 ayrı parça oluşturdu:
    0: 1.29 MB
    1: 1.19 MB
  → En büyük parçayı kullanıyoruz: 0
```

İdeal değil. Daha iyi bir sahne yakalamak için:

- Video daha yavaş/yumuşak pan yapsın
- `--exhaustive` matcher'ını dene (`backend/preprocess/run_colmap.py` içinde geçici değişiklik)
- Frame'leri daha yavaş çıkar (`--fps 5` ile daha az ama daha dağılmış frame)

---

## 6. Hızlı doğrulama

Tüm kurulum tamamlandıktan sonra:

```cmd
conda activate gs4d
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import gsplat; print('gsplat OK')"
colmap -h
```

Test video'sunu koy ve **smoke test** çalıştır (500 iter, 480x270, 10 timestamp — ~2 dk):

```cmd
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
python -m backend.pipeline "data\test_scene\video.mp4" --scene test_scene --smoke-test
```

Beklenen çıktı:

```
=== Pipeline tamamlandı (~30-60s) ===
  frames: ok
  colmap: NN kamera, NNNN nokta
  training: 500 iter, son loss=~0.10
  export: 10 timestamp
```

`data/test_scene/output/ply/frame_0000.ply` dosyasını [antimatter15 viewer](https://antimatter15.com/splat/)'a sürükle-bırak. Bulanık da olsa sahnenin silueti görünmeli.

---

## 7. Yaygın hata mesajları ve çözümleri

| Hata | Sebep | Çözüm |
|------|-------|-------|
| `ImportError: cannot import name 'csrc' from 'gsplat'` | JIT compile başarısız | Bkz. §3.2 (-Wno-attributes patch) |
| `_jit_compile() missing 2 required positional arguments` | PyTorch API değişti | torch 2.5.1+cu124 kullan (bkz §2.2) |
| `where cl` failed / `cl.exe not found` | Build Tools PATH'te değil | x64 Native Tools Command Prompt kullan (§3.1) |
| `UnicodeEncodeError: 'charmap' codec can't encode` | Windows cp1254 + stdout redirect | `set PYTHONIOENCODING=utf-8` (§4) |
| `CUDACachingAllocator.h(105): error: invalid combination...` | torch 2.11+ pre-release + gsplat 1.5.3 | torch 2.5.1+cu124'e düşür (§2.2) |
| `cl: error D8021: invalid numeric argument '/Wno-attributes'` | gsplat GCC flag'i MSVC'ye gönderiyor | Bkz. §3.2 |
| `IndexError: The shape of the mask [N] ... tensor [M, 3]` | Density control bug (artık düzeltildi) | En son `density_control.py`'yi çek |
| `COLMAP mapper başarısız / 0 kamera` | Az feature / az hareket | Daha iyi video; `--exhaustive` matcher |

---

## 8. Bilgi: Ninja / torch_extensions cache

İlk gsplat compile sonrası CUDA `.o` dosyaları şurada cache'lenir:

```
%LOCALAPPDATA%\torch_extensions\Cache\py310_cu124\gsplat_cuda\
```

PyTorch versiyonunu değiştirdiğinde (örn. 2.5.1 → 2.6) bu cache **geçersiz** olur ama otomatik temizlenmez. Manuel sil:

```cmd
rmdir /S /Q "%LOCALAPPDATA%\torch_extensions"
```

---

## 9. Referans: Test edilmiş tam sürüm kombinasyonu

`pip freeze` çıktısının kritik satırları:

```
torch==2.5.1+cu124
torchvision==0.20.1+cu124
gsplat==1.5.3
numpy==1.26.4
opencv-python==4.x
plyfile==1.x
```

CUDA runtime: 12.6.85 (nvcc), driver 538+ (`nvidia-smi`).

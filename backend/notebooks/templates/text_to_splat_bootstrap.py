"""Standard-library host setup; never changes Colab's kernel Python/torch."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

if __package__:
    from .text_to_splat_settings import estimate_requirements, model_provenance, require_mesh_memory, validate_settings
else:
    from text_to_splat_settings import estimate_requirements, model_provenance, require_mesh_memory, validate_settings

PYTHON_VERSION = '3.11.11'
UV_VERSION = '0.8.22'
TRELLIS_COMMIT = '442aa1e1afb9014e80681d3bf604e8d728a86ee7'
DINO_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'

# Direct wheels only; this second private environment has no Torch 2.4 pin.
IMAGE_REQUIREMENTS = '''torch==2.7.1+cu128
torchvision==0.22.1+cu128
diffusers==0.37.0
transformers==4.57.6
accelerate==1.12.0
huggingface-hub==0.36.2
numpy==1.26.4
Pillow==11.3.0
sentencepiece==0.2.1
protobuf==4.25.8
safetensors==0.6.2
rembg==2.0.59
onnxruntime==1.19.2
timm==1.0.22
kornia==0.8.2
einops==0.8.1
psutil==7.2.2
'''


def validate_config(config):
    config.update(validate_settings(config))
    for name, maximum in [('prompt', 2000), ('negative_prompt', 2000), ('style', 500)]:
        value = config[name]
        if not isinstance(value, str) or len(value) > maximum or '\x00' in value:
            raise ValueError(f'{name}: en çok {maximum} karakterlik metin kullanın; NUL karakteri olamaz.')
    if not config['prompt'].strip():
        raise ValueError('PROMPT boş olamaz; örneğin a red sports car yazın.')
    seed = config['seed']
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError('SEED 0–2147483647 arasında tam sayı olmalı.')
    count = config['target_splat_count']
    if count is not None and (type(count) is not int or not 1000 <= count <= 2000000):
        raise ValueError('TARGET_SPLAT_COUNT None veya 1000–2000000 arasında tam sayı olmalı.')
    drive = Path('/content/drive/MyDrive').resolve()
    output = Path(config['output_dir']).resolve()
    if not output.is_relative_to(drive) or output == drive:
        raise ValueError('OUTPUT_DIR /content/drive/MyDrive/ altında bir alt klasör olmalı.')


def host_preflight(config):
    validate_config(config)
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or not Path('/content').is_dir():
        raise RuntimeError('Bu notebook Linux x86_64 Google Colab içindir. Yerelde model indirmeyin; L4/A100 Colab oturumu açın.')
    if sys.version_info < (3, 10):
        raise RuntimeError('Colab çekirdeği Python 3.10+ olmalı. Model adımları ayrıca sabit Python 3.11 ortamında çalışır.')
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=20)
        gpu, memory, driver = result.stdout.strip().splitlines()[0].split(',')
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise RuntimeError('GPU bulunamadı. Colab: Çalışma zamanı → Çalışma zamanı türünü değiştir → L4 veya A100.') from exc
    gib = float(memory) / 1024
    requirements = estimate_requirements(config)
    if config['reconstruction_model'] == 'trellis2':
        require_mesh_memory(config, gib, gib)
    if gib < requirements['min_vram']:
        raise RuntimeError(f'Yetersiz VRAM: {gpu.strip()} {gib:.1f} GiB. Seçilen modeller/ayarlar en az {requirements["min_vram"]} GiB ister. Daha küçük model veya çözünürlük seçin; T4 desteklenmez.')
    if int(driver.strip().split('.')[0]) < 570:
        raise RuntimeError('Modern görüntü ortamı CUDA 12.8 için NVIDIA 570+ sürücüsü ister. Güncel Colab GPU oturumu seçin.')
    # Original TRELLIS wheels were built before Blackwell. Fail before install
    # or downloads instead of presenting a large-memory preset as compatible.
    if config['reconstruction_model'] == 'trellis' and any(word in gpu.lower() for word in ('blackwell', 'rtx pro 6000', 'b100', 'b200', 'rtx 50')):
        raise RuntimeError('Bu Blackwell GPU, kararlı TRELLIS Torch 2.4 cu121 çekirdekleriyle uyumlu değil. Kararlı yol için uyumlu bir GPU oturumu seçin veya deneysel TRELLIS.2 mesh → splat yolunu açıkça seçin.')
    if config['reconstruction_model'] == 'trellis2':
        _mesh_module().compiler_preflight()
    if shutil.disk_usage('/content').free < requirements['disk_gib'] * 1024**3:
        raise RuntimeError(f'Yerel Colab diskinde en az {requirements["disk_gib"]} GiB boş yer gerekli (ortam + seçilen modeller).')
    ram_required = requirements['ram_gib']
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available_ram = int(meminfo['MemAvailable'].split()[0]) / 1024**2
    if available_ram < ram_required:
        raise RuntimeError(f'CPU offload için boş sistem RAM yetersiz: {available_ram:.1f} GiB; {config["image_model"]} yaklaşık {ram_required} GiB boş RAM ister. Colab yüksek RAM oturumu veya küçük model seçin.')
    load_hf_token()
    gated = [model['id'] for model in model_provenance(config).values() if model.get('gated')]
    if gated and not os.environ.get('HF_TOKEN'):
        raise RuntimeError('Seçilen modeller erişim onayı gerektirir: ' + ', '.join(gated)
            + '. Hugging Face model sayfalarında lisansı kabul edin (DINOv3 için Meta onayı bekleyin); Colab Secrets içinde HF_TOKEN tanımlayın ve notebook erişimini açın.')
    print(f'✓ {gpu.strip()} · {gib:.1f} GiB · sürücü {driver.strip()} · çekirdek Python {platform.python_version()}', flush=True)
    print('Görüntü: ayrı Python 3.11 / Torch 2.7.1 cu128. TRELLIS: ayrı Torch 2.4 cu121; TRELLIS.2: kendi ortamı. Colab çekirdeği değiştirilmez.', flush=True)


def load_hf_token():
    """Secrets remain only in the process environment, never config/logs."""
    if os.environ.get('HF_TOKEN'):
        return
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
    except Exception:
        return
    if isinstance(token, str) and token.strip():
        os.environ['HF_TOKEN'] = token.strip()


def _mesh_module():
    return importlib.import_module((__package__ + '.' if __package__ else '') + 'text_to_splat_mesh')


def run_logged(command, log_path, *, env=None, cwd=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf8') as log:
        process = subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf8', errors='replace', env=env, cwd=cwd)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f'İşlem başarısız (kod {code}). Günlük: {log_path}. Görüntü/TRELLIS wheel sürümlerini kontrol edin; deneysel TRELLIS.2 yolunda derleyici günlüğünü inceleyin.')


def runtime_environment(work):
    work = Path(work)
    env = dict(os.environ)
    env.update({'PYTHONPATH': str(work) + os.pathsep + str(work / 'TRELLIS'),
        'PYTHONNOUSERSITE': '1', 'PYTHONUNBUFFERED': '1',
        'HF_HOME': str(work / 'cache' / 'huggingface'), 'TORCH_HOME': str(work / 'cache' / 'torch'),
        'HF_MODULES_CACHE': str(work / 'cache' / 'huggingface' / 'modules'),
        'U2NET_HOME': str(work / 'cache' / 'rembg'), 'NUMBA_CACHE_DIR': str(work / 'cache' / 'numba'),
        'ATTN_BACKEND': 'xformers', 'SPARSE_ATTN_BACKEND': 'xformers', 'SPARSE_BACKEND': 'spconv',
        'SPCONV_ALGO': 'native', 'SPCONV_DISABLE_JIT': '1',
        'UV_CACHE_DIR': str(work / 'cache' / 'uv'), 'UV_PYTHON_INSTALL_DIR': str(work / 'python'),
        'UV_LINK_MODE': 'copy', 'PIP_DISABLE_PIP_VERSION_CHECK': '1'})
    return env


def prepare_environment(config):
    host_preflight(config)
    work = Path(config['work_dir'])
    work.mkdir(parents=True, exist_ok=True)
    env = runtime_environment(work)
    log = work / 'logs' / 'setup.log'
    uv_dir = work / 'uv-tool'
    if not (uv_dir / 'uv').exists():
        run_logged([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--only-binary=:all:',
                    '--target', uv_dir, f'uv=={UV_VERSION}'], log)
    uv_env = dict(env, PYTHONPATH=str(uv_dir))
    uv = [sys.executable, '-m', 'uv']
    image_python = work / 'image-env' / 'bin' / 'python'
    if not image_python.exists():
        run_logged(uv + ['venv', '--python', PYTHON_VERSION, '--python-preference', 'only-managed', str(work / 'image-env')], log, env=uv_env)
    image_lock = work / 'image-requirements.txt'
    image_lock.write_text(IMAGE_REQUIREMENTS, encoding='utf8')
    image_digest = hashlib.sha256(IMAGE_REQUIREMENTS.encode()).hexdigest()
    image_marker = work / 'image-environment.sha256'
    if not image_marker.exists() or image_marker.read_text() != image_digest:
        run_logged(uv + ['pip', 'install', '--python', str(image_python), '--only-binary=:all:',
            '--index-strategy', 'unsafe-best-match', '--extra-index-url', 'https://download.pytorch.org/whl/cu128',
            '-r', str(image_lock)], log, env=uv_env)
        run_logged(uv + ['pip', 'check', '--python', str(image_python)], log, env=uv_env)
        image_marker.write_text(image_digest)
    run_stage(config, 'check-image')
    if config['reconstruction_model'] == 'trellis2':
        _mesh_module().prepare_environment(config, uv, uv_env, log, run_logged)
        run_stage(config, 'check')
        return work / 'mesh-env' / 'bin' / 'python'
    python = work / 'env' / 'bin' / 'python'
    if not python.exists():
        run_logged(uv + ['venv', '--python', PYTHON_VERSION, '--python-preference', 'only-managed', str(work / 'env')], log, env=uv_env)
    lock = work / 'text_to_splat_requirements.txt'
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    marker = work / 'environment.sha256'
    if not marker.exists() or marker.read_text() != digest:
        run_logged(uv + ['pip', 'sync', '--python', str(python), '--only-binary=:all:',
            '--index-strategy', 'unsafe-best-match', '--extra-index-url', 'https://download.pytorch.org/whl/cu121',
            str(lock)], log, env=uv_env)
        run_logged(uv + ['pip', 'check', '--python', str(python)], log, env=uv_env)
        marker.write_text(digest)
    # Avoid importing models before the private environment is ready.
    repo = work / 'TRELLIS'
    if not (repo / '.git').exists():
        run_logged(['git', 'init', str(repo)], log)
        run_logged(['git', 'remote', 'add', 'origin', 'https://github.com/microsoft/TRELLIS.git'], log, cwd=repo)
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True, capture_output=True)
    if result.returncode:
        run_logged(['git', 'fetch', '--depth', '1', 'origin', TRELLIS_COMMIT], log, cwd=repo)
        run_logged(['git', 'checkout', '--detach', TRELLIS_COMMIT], log, cwd=repo)
    elif result.stdout.strip() != TRELLIS_COMMIT:
        raise RuntimeError('TRELLIS klasöründe farklı commit var. Temiz Colab oturumunda yeniden çalıştırın.')
    prepare_trellis_sources(repo)
    run_stage(config, 'check')
    print('✓ Ortam hazır. Henüz model ağırlıkları indirilmedi.', flush=True)
    return python


def run_stage(config, stage):
    validate_config(config)
    if stage not in {'image', 'check-image', 'gaussian', 'export', 'check'}:
        raise ValueError('Bilinmeyen notebook aşaması.')
    work = Path(config['work_dir'])
    if stage in ('image', 'check-image'):
        python = work / 'image-env' / 'bin' / 'python'
        script = work / 'text_to_splat_image.py'
        command_stage = 'check' if stage == 'check-image' else stage
    elif config['reconstruction_model'] == 'trellis2':
        python = work / 'mesh-env' / 'bin' / 'python'
        script = work / 'text_to_splat_mesh.py'
        command_stage = stage
    else:
        python = work / 'env' / 'bin' / 'python'
        script = work / 'text_to_splat_runtime.py'
        command_stage = stage
    if not python.exists():
        raise RuntimeError('Önce ortam kurulum hücresini çalıştırın.')
    config_path = work / 'run-config.json'
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding='utf8')
    log = work / 'logs' / f'{stage}.log'
    env = runtime_environment(work)
    if config['reconstruction_model'] == 'trellis2' and stage not in ('image', 'check-image'):
        env.update(_mesh_module().mesh_environment(work))
    try:
        run_logged([python, script, command_stage, '--config', config_path],
                   log, env=env, cwd=work)
    finally:
        # Local log is always retained if Drive disconnects during the run.
        try:
            logs = Path(config['output_dir']) / 'logs'
            logs.mkdir(parents=True, exist_ok=True)
            for item in (work / 'logs').glob('*.log'):
                shutil.copy2(item, logs / item.name)
        except OSError as exc:
            print(f'Drive günlük kopyası başarısız; yerel günlükler {work / "logs"}: {exc}', flush=True)


def prepare_trellis_sources(repo_dir):
    """Make pinned TRELLIS Gaussian-only imports lazy; reject upstream drift.

    No model computation is modified. Unused save_ply/load_ply retain their
    optional utils3d import; the notebook exports activated fields itself.
    """
    root = Path(repo_dir)
    patches = [
        ("trellis/__init__.py", "from . import renderers\n", "# Gaussian-only notebook: renderers are intentionally not imported.\n"),
        ("trellis/pipelines/__init__.py", "from .trellis_text_to_3d import TrellisTextTo3DPipeline\n", "# Gaussian-only notebook: no direct text pipeline or Open3D dependency.\n"),
        (
            "trellis/representations/__init__.py",
            "from .radiance_field import Strivec\nfrom .octree import DfsOctree as Octree\nfrom .gaussian import Gaussian\nfrom .mesh import MeshExtractResult\n",
            "from .gaussian import Gaussian\n# Gaussian-only notebook: no mesh/radiance-field imports.\n",
        ),
        (
            "trellis/models/structured_latent_vae/__init__.py",
            "from .encoder import SLatEncoder, ElasticSLatEncoder\nfrom .decoder_gs import SLatGaussianDecoder, ElasticSLatGaussianDecoder\nfrom .decoder_rf import SLatRadianceFieldDecoder, ElasticSLatRadianceFieldDecoder\nfrom .decoder_mesh import SLatMeshDecoder, ElasticSLatMeshDecoder\n",
            "from .decoder_gs import SLatGaussianDecoder, ElasticSLatGaussianDecoder\n# Gaussian-only notebook: only its trained Gaussian decoder is loaded.\n",
        ),
        (
            "trellis/representations/gaussian/gaussian_model.py",
            "import utils3d\n",
            "# utils3d is optional and imported only by the unused legacy PLY methods.\n",
        ),
        (
            "trellis/representations/gaussian/gaussian_model.py",
            "    def save_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):\n",
            "    def save_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):\n        import utils3d\n",
        ),
        (
            "trellis/representations/gaussian/gaussian_model.py",
            "    def load_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):\n",
            "    def load_ply(self, path, transform=[[1, 0, 0], [0, 0, -1], [0, 1, 0]]):\n        import utils3d\n",
        ),
        (
            "trellis/pipelines/trellis_image_to_3d.py",
            "torch.hub.load('facebookresearch/dinov2', name, pretrained=True)",
            "torch.hub.load('facebookresearch/dinov2:" + DINO_COMMIT + "', name, pretrained=True, trust_repo=True, skip_validation=True)",
        ),
    ]
    sources = {}
    for relative, original, replacement in patches:
        path = root / relative
        source = sources.get(path)
        if source is None:
            source = path.read_text(encoding="utf-8")
        # Check the replacement before original because some insertions contain
        # the original signature as a substring. Reject duplicates in both cases.
        if source.count(replacement) == 1:
            continue
        if source.count(original) != 1:
            raise RuntimeError("TRELLIS kaynak sürümü beklenenden farklı: " + relative)
        sources[path] = source.replace(original, replacement, 1)
    # Validate all patches before writing any changes.
    for path, source in sources.items():
        compile(source, str(path), "exec")
    for path, source in sources.items():
        path.write_text(source, encoding="utf-8")

"""Executed by the notebook's private Python, one process per GPU stage."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

if __package__:
    from .text_to_splat_helpers import StageCache, atomic_json, file_sha256, normalize_gaussians, render_turntable, write_ply
    from .text_to_splat_settings import estimate_requirements, model_provenance, validate_settings
else:
    from text_to_splat_helpers import StageCache, atomic_json, file_sha256, normalize_gaussians, render_turntable, write_ply
    from text_to_splat_settings import estimate_requirements, model_provenance, validate_settings

PINS = {
    'trellis_repo': 'microsoft/TRELLIS', 'trellis_commit': '442aa1e1afb9014e80681d3bf604e8d728a86ee7',
    'trellis_model': 'microsoft/TRELLIS-image-large', 'trellis_revision': '25e0d31ffbebe4b5a97464dd851910efc3002d96',
    'dino_repo': 'facebookresearch/dinov2', 'dino_commit': '7764ea0f912e53c92e82eb78a2a1631e92725fc8',
    'dino_model': 'dinov2_vitl14_reg',
    'dino_weights': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth',
}


def provenance_for(config):
    """Identical cache identity across the image, Gaussian and mesh environments."""
    return {'models': model_provenance(config), 'recipe_sha256': config['recipe_sha256']}


def gpu_preflight(min_vram):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Yanlış Python: model hücreleri özel Python 3.11 ortamında çalıştırılmalı. Kurulum hücresini tekrar çalıştırın.')
    import torch
    if torch.__version__.split('+')[0] != '2.4.0' or torch.version.cuda != '12.1':
        raise RuntimeError('Torch/CUDA sürümü uyuşmuyor; sabitlenmiş kurulum hücresini yeniden çalıştırın.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU bulunamadı. Colab: Çalışma zamanı → Çalışma zamanı türünü değiştir → L4/A100.')
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = torch.cuda.mem_get_info()[0] / 1024**3
    if memory < min_vram or free < min_vram - 2:
        raise RuntimeError(f'Yetersiz VRAM: toplam {memory:.1f}, boş {free:.1f} GiB; bu ön ayar en az {min_vram} GiB ister. L4/A100 veya daha düşük ön ayar seçin.')
    return {'name': torch.cuda.get_device_name(0), 'total_gib': memory, 'free_gib_at_start': free}


def gaussian_model_directory(work):
    """Fetch four pinned Gaussian-only model pairs; never mutate the HF cache."""
    from huggingface_hub import hf_hub_download
    model_id, revision = PINS['trellis_model'], PINS['trellis_revision']
    source = hf_hub_download(model_id, 'pipeline.json', revision=revision)
    config = json.loads(Path(source).read_text())
    selected = {'sparse_structure_decoder', 'sparse_structure_flow_model', 'slat_flow_model', 'slat_decoder_gs'}
    if not selected <= config['args']['models'].keys():
        raise RuntimeError('Sabitlenmiş TRELLIS model tanımı değişti; Gaussian model alanları eksik.')
    config['args']['models'] = {k: v for k, v in config['args']['models'].items() if k in selected}
    model_dir = work / 'gaussian-model'
    model_dir.mkdir(exist_ok=True)
    for prefix in config['args']['models'].values():
        if not prefix.startswith('ckpts/') or '..' in prefix or '\\' in prefix:
            raise RuntimeError('Beklenmeyen model yolu; indirme durduruldu.')
        for suffix in ('.json', '.safetensors'):
            cached = hf_hub_download(model_id, prefix + suffix, revision=revision)
            dest = model_dir / (prefix + suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(cached)
    atomic_json(model_dir / 'pipeline.json', config)
    return model_dir


def generate_gaussians(output, config, work):
    import numpy as np
    import torch
    from PIL import Image
    from trellis.pipelines import TrellisImageTo3DPipeline

    model_dir = gaussian_model_directory(work)
    print('TRELLIS Gaussian modeli yükleniyor; mesh ve radyans alanı yüklenmez.', flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(model_dir))
    pipeline.cuda()
    with Image.open(output / 'cutout.png') as image:
        with torch.inference_mode():
            result = pipeline.run(image, seed=config['trellis_seed'], formats=['gaussian'],
                sparse_structure_sampler_params={'steps': config['sparse_steps'], 'cfg_strength': config['sparse_cfg']},
                slat_sampler_params={'steps': config['slat_steps'], 'cfg_strength': config['slat_cfg']})
    gs = result['gaussian'][0]
    if gs.sh_degree != 0:
        raise RuntimeError('Beklenmeyen SH derecesi; koordinat dönüşümü için reçete güncellenmeli.')
    def cpu(value):
        return value.detach().float().cpu().numpy()
    values = dict(xyz=cpu(gs.get_xyz), scales=cpu(gs.get_scaling), rotations=cpu(gs.get_rotation),
                  opacity=cpu(gs.get_opacity), dc=cpu(gs._features_dc))
    # Validate activated fields before committing the expensive stage.
    _, normalization = normalize_gaussians(**values)
    temp = output / 'gaussians_raw.npz.partial'
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    os.replace(temp, output / 'gaussians_raw.npz')
    checkpoint = Path(torch.hub.get_dir()) / 'checkpoints' / 'dinov2_vitl14_reg4_pretrain.pth'
    return {'native_splat_count': normalization['input_count'], 'dino_weights_sha256': file_sha256(checkpoint)}


def export_gaussians(output, config):
    import numpy as np
    with np.load(output / 'gaussians_raw.npz', allow_pickle=False) as archive:
        cloud, normalization = normalize_gaussians(**dict(archive),
            target_count=config['target_splat_count'], seed=config['seed'])
    write_ply(output / 'target.ply', cloud)
    preview = render_turntable(output / 'turntable.gif', cloud)
    print(f"✓ target.ply: {len(cloud['means']):,} Gaussian · +Y yukarı · merkez orijin", flush=True)
    return {'normalization': normalization, 'preview': preview}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['image', 'gaussian', 'export', 'check'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = validate_settings(json.loads(Path(args.config).read_text(encoding='utf8')))
    if args.stage == 'image':
        raise RuntimeError('Görüntü adımı için ayrı image-env ortamını kullanın; kurulum hücresini yeniden çalıştırın.')
    gpu = gpu_preflight(estimate_requirements(config)['geometry_vram'])
    import torch
    # Avoid TF32 differences between GPU generations; full bitwise reproducibility is not promised.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == 'check':
        import xformers.ops
        import spconv.pytorch as spconv
        from trellis.pipelines import TrellisImageTo3DPipeline
        # The registry imports decoders lazily; force every selected class now
        # so optional mesh/renderer dependencies cannot fail after downloads.
        from trellis.models import (SparseStructureDecoder, SparseStructureFlowModel,
                                    SLatGaussianDecoder, SLatFlowModel)
        # Run tiny real kernels BEFORE multi-GB model downloads. No JIT build fallback.
        q = torch.randn(1, 16, 2, 32, device='cuda', dtype=torch.float16)
        xformers.ops.memory_efficient_attention(q, q, q)
        features = torch.randn(2, 4, device='cuda')
        indices = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32, device='cuda')
        sparse = spconv.SparseConvTensor(features, indices, [4, 4, 4], 1)
        spconv.SubMConv3d(4, 4, 3, padding=1, algo=spconv.ConvAlgo.Native).cuda()(sparse)
        torch.cuda.synchronize()
        print('✓ Python, Torch, xformers, spconv CUDA çekirdekleri ve Gaussian import kontrolü başarılı.', flush=True)
        return
    work = Path(config['work_dir'])
    cache = StageCache(config['output_dir'], config, provenance_for(config))
    cache.data['environment'] = {'python': platform.python_version(), 'torch_cuda': torch.version.cuda,
        'gpu': gpu, 'packages': {p: importlib.metadata.version(p) for p in (
            'torch', 'torchvision', 'xformers', 'spconv-cu120', 'cumm-cu120', 'numpy', 'diffusers',
            'transformers', 'huggingface-hub', 'rembg', 'onnxruntime', 'Pillow')}}
    cache.data.setdefault('environments', {})['trellis'] = cache.data['environment']
    atomic_json(cache.path, cache.data)
    if args.stage == 'gaussian':
        cache.run('gaussian', ['gaussians_raw.npz'], lambda: generate_gaussians(cache.output, config, work), ['image'])
    else:
        cache.run('export', ['target.ply', 'turntable.gif'], lambda: export_gaussians(cache.output, config), ['image', 'gaussian'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'İşlem durdu: {exc}\nAyrıntılar günlükte. Aynı ayarlarla ilgili hücreyi yeniden çalıştırabilirsiniz; ayarlar değişirse yeni OUTPUT_DIR seçin.', file=sys.stderr, flush=True)
        raise

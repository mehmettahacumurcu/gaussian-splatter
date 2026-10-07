"""Executed by the notebook's private Python, one process per GPU stage."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from text_to_splat_helpers import StageCache, atomic_json, file_sha256, normalize_gaussians, render_turntable, write_ply

PINS = {
    'trellis_repo': 'microsoft/TRELLIS', 'trellis_commit': '442aa1e1afb9014e80681d3bf604e8d728a86ee7',
    'trellis_model': 'microsoft/TRELLIS-image-large', 'trellis_revision': '25e0d31ffbebe4b5a97464dd851910efc3002d96',
    'image_model': 'stabilityai/stable-diffusion-xl-base-1.0', 'image_revision': '462165984030d82259a11f4367a4eed129e94a7b',
    'dino_repo': 'facebookresearch/dinov2', 'dino_commit': '7764ea0f912e53c92e82eb78a2a1631e92725fc8',
    'dino_model': 'dinov2_vitl14_reg',
    'dino_weights': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth',
    'rembg_model': 'u2net', 'rembg_weights': 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx',
    'rembg_md5': '60024c5c889badc19c04ad937298a77b',
}


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


def generate_image(output, config):
    import torch
    from diffusers import StableDiffusionXLPipeline
    from PIL import Image
    import numpy as np
    import rembg

    pipe = StableDiffusionXLPipeline.from_pretrained(PINS['image_model'], revision=PINS['image_revision'],
        variant='fp16', use_safetensors=True, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    prompt = config['prompt'] + '. ' + config['style']
    prompt += '. Single isolated object, full object visible, three-quarter view, plain white background, studio lighting, no text.'
    negative = config['negative_prompt']
    framing_negative = 'cropped, out of frame, multiple objects, text, watermark, blurry, flat illustration'
    negative = ', '.join(filter(None, (negative, framing_negative)))
    print('Görüntü üretiliyor. Kısa, tek nesneli İngilizce istemler önerilir; CLIP uzun metni kırpabilir.', flush=True)
    image = pipe(prompt=prompt, negative_prompt=negative, num_inference_steps=config['quality']['image_steps'],
        guidance_scale=7.0, height=1024, width=1024,
        generator=torch.Generator(device='cpu').manual_seed(config['seed'])).images[0]
    image.save(output / 'input.png')
    del pipe
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    session = rembg.new_session('u2net', providers=['CPUExecutionProvider'])
    cutout = rembg.remove(image, session=session).convert('RGBA')
    alpha = np.asarray(cutout)[:, :, 3]
    coords = np.argwhere(alpha > 204)
    if len(coords) < 64 or np.any(np.ptp(coords, axis=0) < 8):
        raise RuntimeError('Arka plan temizlemede nesne bulunamadı. İstemi sadeleştirin ve yeni OUTPUT_DIR seçin.')
    # Ensure an alpha channel is used by TRELLIS, including unusually full masks.
    if np.all(alpha == 255):
        raise RuntimeError('Arka plan maskesi tüm görüntüyü kaplıyor. Tek nesne/beyaz arka plan ile yeniden deneyin.')
    cutout.save(output / 'cutout.png')
    with Image.open(output / 'input.png') as check:
        check.verify()
    weights = Path(os.environ['U2NET_HOME']) / 'u2net.onnx'
    return {'effective_prompt': prompt, 'effective_negative_prompt': negative,
            'rembg_weights_sha256': file_sha256(weights)}


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
    quality = config['quality']
    with Image.open(output / 'cutout.png') as image:
        with torch.inference_mode():
            result = pipeline.run(image, seed=config['seed'], formats=['gaussian'],
                sparse_structure_sampler_params={'steps': quality['sparse_steps'], 'cfg_strength': 7.5},
                slat_sampler_params={'steps': quality['slat_steps'], 'cfg_strength': 3.0})
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
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    gpu = gpu_preflight(config['quality']['min_vram'])
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
    provenance = {'models': PINS, 'recipe_sha256': config['recipe_sha256'],
        'licenses': {'TRELLIS': 'MIT', 'SDXL': 'CreativeML Open RAIL++-M', 'DINOv2': 'Apache-2.0', 'U2Net': 'Apache-2.0'}}
    cache = StageCache(config['output_dir'], config, provenance)
    cache.data['environment'] = {'python': platform.python_version(), 'torch_cuda': torch.version.cuda,
        'gpu': gpu, 'packages': {p: importlib.metadata.version(p) for p in (
            'torch', 'torchvision', 'xformers', 'spconv-cu120', 'cumm-cu120', 'numpy', 'diffusers',
            'transformers', 'huggingface-hub', 'rembg', 'onnxruntime', 'Pillow')}}
    atomic_json(cache.path, cache.data)
    if args.stage == 'image':
        cache.run('image', ['input.png', 'cutout.png'], lambda: generate_image(cache.output, config))
    elif args.stage == 'gaussian':
        cache.run('gaussian', ['gaussians_raw.npz'], lambda: generate_gaussians(cache.output, config, work), ['image'])
    else:
        cache.run('export', ['target.ply', 'turntable.gif'], lambda: export_gaussians(cache.output, config), ['image', 'gaussian'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'İşlem durdu: {exc}\nAyrıntılar günlükte. Aynı ayarlarla ilgili hücreyi yeniden çalıştırabilirsiniz; ayarlar değişirse yeni OUTPUT_DIR seçin.', file=sys.stderr, flush=True)
        raise

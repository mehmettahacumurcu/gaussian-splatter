"""Shared, dependency-free allow-lists embedded in the notebook (metadata checked 2026-10-07).

VRAM figures are conservative recipe admission thresholds, not measured peak promises.
Never accept a repository ID, revision, Python expression or installation URL from a user.
"""
from __future__ import annotations

import math

IMAGE_MODELS = {
    'sdxl': dict(id='stabilityai/stable-diffusion-xl-base-1.0', revision='462165984030d82259a11f4367a4eed129e94a7b', license='CreativeML Open RAIL++-M', gated=False, negative_prompt=True, min_vram=12, steps=25, guidance=7.0, disk_gib=35, ram_gib=12),
    'flux1_dev': dict(id='black-forest-labs/FLUX.1-dev', revision='3de623fc3c33e44ffbe2bad470d0f45bccf2eb21', license='FLUX.1-dev Non-Commercial License', gated=True, negative_prompt=True, min_vram=36, steps=28, guidance=3.5, disk_gib=60, ram_gib=40),
    'flux1_schnell': dict(id='black-forest-labs/FLUX.1-schnell', revision='741f7c3ce8b383c54771c7003378a50191e9efe9', license='Apache-2.0', gated=True, negative_prompt=False, min_vram=36, steps=4, guidance=0.0, disk_gib=60, ram_gib=40),
    'flux2_klein_4b': dict(id='black-forest-labs/FLUX.2-klein-4B', revision='e7b7dc27f91deacad38e78976d1f2b499d76a294', license='Apache-2.0', gated=False, negative_prompt=False, min_vram=16, steps=4, guidance=1.0, disk_gib=40, ram_gib=20),
    'qwen_image': dict(id='Qwen/Qwen-Image', revision='75e0b4be04f60ec59a75f475837eced720f823b6', license='Apache-2.0', gated=False, negative_prompt=True, min_vram=60, steps=40, guidance=4.0, disk_gib=90, ram_gib=56),
}

BACKGROUND_MODELS = {
    'u2net': dict(id='xuebinqin/U-2-Net:u2net', revision='md5:60024c5c889badc19c04ad937298a77b', license='ONNX weights: unverified; U-2-Net code: Apache-2.0; rembg code: MIT', gated=False,
                  url='https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx'),
    'birefnet': dict(id='ZhengPeng7/BiRefNet', revision='e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4', license='MIT', gated=False),
}
RECONSTRUCTION_MODELS = {
    'trellis': dict(id='microsoft/TRELLIS-image-large', revision='25e0d31ffbebe4b5a97464dd851910efc3002d96', license='MIT', gated=False),
    'trellis2': dict(id='microsoft/TRELLIS.2-4B', revision='af44b45f2e35a493886929c6d786e563ec68364d', license='MIT', gated=False),
    'hunyuan3d': dict(id='tencent/Hunyuan3D-2.1', revision='0b94677654c57bb9a6b6845cd7b704ccf551d327', license='Tencent Hunyuan 3D 2.1 Community License', gated=False, implemented=False),
}

GPU_PRESETS = {
    'l4': dict(image_model='sdxl', background_model='u2net', reconstruction_model='trellis', image_steps=25, image_guidance=7.0, image_resolution=1024, sparse_steps=12, sparse_cfg=7.5, slat_steps=12, slat_cfg=3.0, mesh_views=24, mesh_fit_iterations=1500, mesh_splat_cap=50000),
    'a100': dict(image_model='flux1_dev', background_model='birefnet', reconstruction_model='trellis', image_steps=28, image_guidance=3.5, image_resolution=1024, sparse_steps=20, sparse_cfg=7.5, slat_steps=20, slat_cfg=3.0, mesh_views=48, mesh_fit_iterations=3000, mesh_splat_cap=100000),
    'h100': dict(image_model='qwen_image', background_model='birefnet', reconstruction_model='trellis2', image_steps=40, image_guidance=4.0, image_resolution=1024, sparse_steps=24, sparse_cfg=7.5, slat_steps=24, slat_cfg=3.0, mesh_views=64, mesh_fit_iterations=4000, mesh_splat_cap=150000),
    'rtx_pro_6000': dict(image_model='qwen_image', background_model='birefnet', reconstruction_model='trellis2', image_steps=40, image_guidance=4.0, image_resolution=1024, sparse_steps=24, sparse_cfg=7.5, slat_steps=24, slat_cfg=3.0, mesh_views=72, mesh_fit_iterations=5000, mesh_splat_cap=200000),
}
SETTING_FIELDS = set(GPU_PRESETS['l4']) | {'gpu_preset', 'trellis_seed'}
INTEGER_RANGES = {'image_steps': (1, 100), 'trellis_seed': (0, 2147483647),
                  'sparse_steps': (1, 100), 'slat_steps': (1, 100),
                  'mesh_views': (12, 120), 'mesh_fit_iterations': (100, 10000),
                  'mesh_splat_cap': (1000, 300000)}


def resolve_settings(config):
    """Explicit values win; otherwise GPU + image model + legacy quality defaults apply."""
    result = dict(config)
    gpu = result.get('gpu_preset') or 'l4'
    if gpu not in GPU_PRESETS:
        raise ValueError('GPU_PRESET izin verilen seçeneklerden biri olmalı.')
    defaults = dict(GPU_PRESETS[gpu])
    model = result.get('image_model') or defaults['image_model']
    if model not in IMAGE_MODELS:
        raise ValueError('IMAGE_MODEL izin verilen seçeneklerden biri olmalı.')
    if model != defaults['image_model']:
        defaults.update(image_steps=IMAGE_MODELS[model]['steps'], image_guidance=IMAGE_MODELS[model]['guidance'])
    preset = result.get('quality_preset', result.get('preset', 'baseline'))
    if preset not in ('baseline', 'quality', 'ultra'):
        raise ValueError('QUALITY_PRESET baseline, quality veya ultra olmalı.')
    if preset != 'baseline':
        defaults.update(sparse_steps=20 if preset == 'quality' else 25, slat_steps=20 if preset == 'quality' else 25)
        if model == 'sdxl':
            defaults['image_steps'] = 40 if preset == 'quality' else 50
    for key, value in defaults.items():
        if result.get(key) is None:
            result[key] = value
    result['gpu_preset'] = gpu
    if result.get('trellis_seed') is None:
        result['trellis_seed'] = result.get('seed', 42)
    return result


def validate_settings(config):
    result = resolve_settings(config)
    for key, allowed in [('image_model', IMAGE_MODELS), ('background_model', BACKGROUND_MODELS),
                         ('reconstruction_model', RECONSTRUCTION_MODELS)]:
        if result[key] not in allowed:
            raise ValueError(f'{key}: izin verilen model seçeneklerinden birini seçin.')
    if result['reconstruction_model'] == 'hunyuan3d':
        raise ValueError('Hunyuan3D deneysel/sonra: doku bağımlılıkları ve özel lisans henüz bu Colab tarifine entegre edilmedi.')
    for key, (low, high) in INTEGER_RANGES.items():
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f'{key}: {low}–{high} arasında tam sayı olmalı.')
    if type(result['image_resolution']) is not int or result['image_resolution'] not in (512, 768, 1024):
        raise ValueError('IMAGE_RESOLUTION 512, 768 veya 1024 olmalı.')
    for key in ('image_guidance', 'sparse_cfg', 'slat_cfg'):
        if type(result[key]) not in (float, int) or not math.isfinite(result[key]) or not 0 <= result[key] <= 20:
            raise ValueError(f'{key}: 0–20 arasında sonlu sayı olmalı.')
    if not IMAGE_MODELS[result['image_model']]['negative_prompt'] and result.get('negative_prompt', '').strip():
        raise ValueError('Seçilen görüntü modeli negatif istem kullanmaz; NEGATIVE_PROMPT alanını boşaltın.')
    if result['image_model'] == 'flux1_schnell' and (result['image_steps'] > 4 or result['image_guidance'] != 0):
        raise ValueError('FLUX.1-schnell için 1–4 görüntü adımı ve guidance=0 kullanın.')
    if result['image_model'] == 'flux2_klein_4b' and result['image_guidance'] != 1:
        raise ValueError('Damıtılmış FLUX.2-klein-4B için guidance=1 kullanın.')
    return result


def estimate_requirements(config):
    settings = validate_settings(config)
    image = IMAGE_MODELS[settings['image_model']]
    mesh = settings['reconstruction_model'] == 'trellis2'
    geometry = (28 + (4 if settings['mesh_splat_cap'] > 150000 else 0)) if mesh else 20
    if settings.get('quality_preset', settings.get('preset')) == 'ultra':
        geometry = max(geometry, 38)
    return dict(min_vram=max(image['min_vram'], geometry), image_vram=image['min_vram'],
                geometry_vram=geometry, disk_gib=image['disk_gib'] + (40 if mesh else 0), ram_gib=image['ram_gib'])


def model_provenance(config):
    settings = validate_settings(config)
    records = {name: dict(registry[settings[field]]) for name, registry, field in (
        ('image', IMAGE_MODELS, 'image_model'), ('background', BACKGROUND_MODELS, 'background_model'),
        ('reconstruction', RECONSTRUCTION_MODELS, 'reconstruction_model'))}
    if settings['reconstruction_model'] == 'trellis':
        records['encoder'] = dict(id='facebookresearch/dinov2:dinov2_vitl14_reg',
            revision='7764ea0f912e53c92e82eb78a2a1631e92725fc8', license='Apache-2.0',
            url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth')
    else:
        records['encoder'] = dict(DINO3_MODEL)
        records['sparse_structure_decoder'] = dict(RECONSTRUCTION_MODELS['trellis'])
    return records


# The TRELLIS.2 encoder has its own access/license requirement; no BRIA model is loaded.
DINO3_MODEL = dict(id='facebook/dinov3-vitl16-pretrain-lvd1689m', revision='ea8dc2863c51be0a264bab82070e3e8836b02d51',
                   license='DINOv3 License', gated=True, access='Meta manual approval + HF_TOKEN')

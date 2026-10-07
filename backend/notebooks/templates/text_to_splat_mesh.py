"""Experimental TRELLIS.2 -> coloured mesh -> known cameras -> gsplat 1.5.3.

All CUDA builds/downloads run only in the Colab private environment. This module
can be imported by offline CPU tests without torch or any upstream installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from text_to_splat_settings import DINO3_MODEL, RECONSTRUCTION_MODELS, estimate_requirements, validate_settings
except ModuleNotFoundError:
    from .text_to_splat_settings import DINO3_MODEL, RECONSTRUCTION_MODELS, estimate_requirements, validate_settings

SOURCE_PINS = {
    'TRELLIS.2': ('https://github.com/microsoft/TRELLIS.2.git', '75fbf0183001ed9876c8dbb35de6b68552ee08bd'),
    'CuMesh': ('https://github.com/JeffreyXiang/CuMesh.git', '12289e1062f0603f2f0d0771b02e1395d247f26f'),
    'FlexGEMM': ('https://github.com/JeffreyXiang/FlexGEMM.git', '6dd94a859c26ee8246888502eada3dd8ad85532e'),
    'nvdiffrast': ('https://github.com/NVlabs/nvdiffrast.git', '253ac4fcea7de5f396371124af597e6cc957bfae'),
    'utils3d': ('https://github.com/EasternJournalist/utils3d.git', '9a4eb15e4021b67b12c460c7057d642626897ec8'),
}
REQUIREMENTS = [
    'torch==2.7.1+cu128', 'torchvision==0.22.1+cu128', 'xformers==0.0.31',
    'numpy==1.26.4', 'transformers==4.57.6', 'huggingface-hub==0.36.0',
    'safetensors==0.6.2', 'pillow==11.3.0', 'scipy==1.15.3', 'trimesh==4.11.3',
    'easydict==1.13', 'tqdm==4.67.1', 'opencv-python-headless==4.11.0.86',
    'imageio==2.37.0', 'imageio-ffmpeg==0.6.0', 'ninja==1.11.1.4',
    'packaging==25.0', 'setuptools==80.9.0', 'wheel==0.45.1',
    'kornia==0.8.1', 'timm==1.0.22', 'einops==0.8.1', 'zstandard==0.25.0',
    'moderngl==5.12.0', 'plyfile==1.1.2',
]


def mesh_environment(work):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(work) + os.pathsep + str(Path(work) / 'TRELLIS.2'),
               ATTN_BACKEND='xformers', SPARSE_ATTN_BACKEND='xformers', SPARSE_CONV_BACKEND='flex_gemm',
               PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', MAX_JOBS='2',
               HF_HOME=str(Path(work) / 'cache' / 'huggingface'),
               TORCH_EXTENSIONS_DIR=str(Path(work) / 'cache' / 'extensions'),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    return env


def compiler_preflight():
    nvcc = shutil.which('nvcc')
    if not nvcc or not shutil.which('g++'):
        raise RuntimeError('TRELLIS.2 deneysel yolu CUDA nvcc ve g++ derleyicisi ister. CUDA toolkit bulunan Colab oturumu seçin veya TRELLIS Gaussian yoluna dönün.')
    version = subprocess.run([nvcc, '--version'], capture_output=True, text=True, check=True).stdout
    match = re.search(r'release (\d+)\.(\d+)', version)
    # One cu128 recipe across A100/H100/Blackwell; no global toolkit installation.
    if not match or int(match[1]) != 12 or int(match[2]) < 8:
        raise RuntimeError('TRELLIS.2 özel cu128 ortamı için CUDA toolkit 12.8/12.9 gerekir; derleyici sürümü uyumsuz. Güncel Colab imajı seçin. Sistem CUDA kurulumu değiştirilmez.')


def _checkout(work, name, log, run_logged):
    url, revision = SOURCE_PINS[name]
    repo = work / name
    if not (repo / '.git').exists():
        run_logged(['git', 'init', repo], log)
        run_logged(['git', 'remote', 'add', 'origin', url], log, cwd=repo)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True)
    if head.returncode:
        run_logged(['git', 'fetch', '--depth', '1', 'origin', revision], log, cwd=repo)
        run_logged(['git', 'checkout', '--detach', revision], log, cwd=repo)
        run_logged(['git', 'submodule', 'update', '--init', '--recursive'], log, cwd=repo)
    elif head.stdout.strip() != revision:
        raise RuntimeError(f'{name} kaynak revizyonu uyuşmuyor; temiz Colab oturumu seçin.')
    return repo


def patch_pipeline_sources(repo):
    """Disable the unrequested BRIA model; pinned local DINO is passed in config."""
    path = Path(repo) / 'trellis2/pipelines/trellis2_image_to_3d.py'
    source = path.read_text(encoding='utf8')
    original = "pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])"
    replacement = 'pipeline.rembg_model = None  # External, selected RGBA mask; never load BRIA.'
    if source.count(replacement) == 1 and source.count(original) == 0:
        return
    else:
        if source.count(original) != 1 or replacement in source:
            raise RuntimeError('TRELLIS.2 kaynak sürümü değişti; arka plan modeli güvenle devre dışı bırakılamadı.')
        source = source.replace(original, replacement)
        compile(source, str(path), 'exec')
        path.write_text(source, encoding='utf8')


def prepare_environment(config, uv, uv_env, log, run_logged):
    compiler_preflight()
    work = Path(config['work_dir'])
    python = work / 'mesh-env/bin/python'
    env = mesh_environment(work)
    env.update({key: value for key, value in uv_env.items() if key.startswith('UV_')})
    if not python.exists():
        run_logged(uv + ['venv', '--python', '3.11.11', '--python-preference', 'only-managed', str(work / 'mesh-env')], log, env=uv_env)
    identity = hashlib.sha256(json.dumps([REQUIREMENTS, SOURCE_PINS, 'gsplat==1.5.3', 1], sort_keys=True).encode()).hexdigest()
    marker = work / 'mesh-environment.sha256'
    if not marker.exists() or marker.read_text() != identity:
        run_logged(uv + ['pip', 'install', '--python', str(python), '--index-strategy', 'unsafe-best-match',
            '--extra-index-url', 'https://download.pytorch.org/whl/cu128', *REQUIREMENTS], log, env=uv_env)
        # Builds are explicitly part of this experimental route, confined to its venv.
        for name in SOURCE_PINS:
            repo = _checkout(work, name, log, run_logged)
            if name == 'TRELLIS.2':
                patch_pipeline_sources(repo)
                continue
            run_logged(uv + ['pip', 'install', '--python', str(python), '--no-build-isolation', '--no-deps', str(repo)], log,
                       env=dict(env, PYTHONPATH=uv_env['PYTHONPATH']))
        run_logged(uv + ['pip', 'install', '--python', str(python), '--no-build-isolation', '--no-deps',
            str(work / 'TRELLIS.2/o-voxel')], log, env=dict(env, PYTHONPATH=uv_env['PYTHONPATH']))
        run_logged(uv + ['pip', 'install', '--python', str(python), '--no-build-isolation', 'gsplat==1.5.3'], log, env=uv_env)
        run_logged(uv + ['pip', 'check', '--python', str(python)], log, env=uv_env)
        marker.write_text(identity)
    return python


def camera_matrices(count, size=512):
    """OpenCV world-to-camera (+Z forward, +Y down), uniformly spread around +Z."""
    import numpy as np
    views = []
    for index in range(count):
        z = 1 - 2 * (index + .5) / count
        yaw = index * math.pi * (3 - math.sqrt(5))
        eye = 2.2 * np.array([math.sqrt(1-z*z)*math.cos(yaw), math.sqrt(1-z*z)*math.sin(yaw), z])
        forward = -eye / np.linalg.norm(eye)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])
        view = np.eye(4, dtype=np.float32)
        view[:3, :3], view[:3, 3] = rotation, -rotation @ eye
        views.append(view)
    focal = .5 * size / math.tan(math.radians(40) / 2)
    intrinsic = np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], np.float32)
    return np.stack(views), intrinsic


def clip_vertices(vertices, view, intrinsic, size):
    """OpenCV projection to nvdiffrast bottom-up clip coordinates; flip output rows."""
    import torch
    cam = vertices @ view[:3, :3].T + view[:3, 3]
    z = cam[:, 2]
    x = 2 * intrinsic[0, 0] / size * cam[:, 0]
    y = -2 * intrinsic[1, 1] / size * cam[:, 1]
    near, far = .01, 10.
    depth = (far+near)/(far-near)*z - 2*far*near/(far-near)
    return torch.stack([x, y, depth, z], dim=-1).unsqueeze(0)


def pinned_model_directory(work):
    from huggingface_hub import hf_hub_download, snapshot_download
    from text_to_splat_helpers import atomic_json
    model = RECONSTRUCTION_MODELS['trellis2']
    path = hf_hub_download(model['id'], 'pipeline.json', revision=model['revision'])
    pipeline = json.loads(Path(path).read_text())
    args = pipeline['args']
    model_dir = work / 'trellis2-model'
    model_dir.mkdir(exist_ok=True)
    # Only 512 path in this first mesh recipe, reducing downloads/VRAM and build risk.
    args['models'] = {key: value for key, value in args['models'].items() if not key.endswith('_1024')}
    for key, prefix in list(args['models'].items()):
        selected = model
        if key == 'sparse_structure_decoder':
            selected = RECONSTRUCTION_MODELS['trellis']
            required_prefix = selected['id'] + '/'
            if not prefix.startswith(required_prefix):
                raise RuntimeError('TRELLIS.2 sparse decoder kimliği değişti.')
            prefix = prefix[len(required_prefix):]
        if not prefix.startswith('ckpts/') or '..' in prefix or '\\' in prefix:
            raise RuntimeError('Sabitlenmiş model dosya yolu geçersiz.')
        dest_prefix = model_dir / 'ckpts' / key
        for ext in ('.json', '.safetensors'):
            cached = hf_hub_download(selected['id'], prefix + ext, revision=selected['revision'])
            dest = Path(str(dest_prefix) + ext)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.symlink_to(cached)
        args['models'][key] = str(dest_prefix)
    dino = snapshot_download(DINO3_MODEL['id'], revision=DINO3_MODEL['revision'],
                             allow_patterns=['*.json', '*.safetensors'])
    args['image_cond_model']['args']['model_name'] = dino
    args['rembg_model'] = None
    args['low_vram'] = True
    args['default_pipeline_type'] = '512'
    atomic_json(model_dir / 'pipeline.json', pipeline)
    return model_dir


def generate_mesh(output, config):
    import numpy as np
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(pinned_model_directory(Path(config['work_dir']))))
    pipeline.cuda()
    with Image.open(output / 'cutout.png') as image, torch.inference_mode():
        mesh = pipeline.run(image, seed=config['trellis_seed'], pipeline_type='512',
            sparse_structure_sampler_params={'steps': config['sparse_steps'], 'guidance_strength': config['sparse_cfg']},
            shape_slat_sampler_params={'steps': config['slat_steps'], 'guidance_strength': config['slat_cfg']},
            tex_slat_sampler_params={'steps': config['slat_steps'], 'guidance_strength': 1.0})[0]
        mesh.simplify(500000)
        attrs = mesh.query_vertex_attrs()
        values = dict(vertices=mesh.vertices.cpu().numpy(), faces=mesh.faces.cpu().numpy(),
                      colors=attrs[:, :3].clamp(0, 1).cpu().numpy(), alpha=attrs[:, 5:6].clamp(0, 1).cpu().numpy())
    if len(values['vertices']) < 4 or len(values['faces']) < 2:
        raise RuntimeError('TRELLIS.2 boş mesh üretti; yeni istem/seed deneyin.')
    save_npz(output / 'mesh.npz', **values)
    return {'vertices': len(values['vertices']), 'faces': len(values['faces']), 'pipeline_type': '512',
            'texture_guidance': 1.0, 'appearance': 'vertex base color + alpha; metallic/roughness not fitted'}


def save_npz(path, **values):
    import numpy as np
    partial = Path(str(path) + '.partial')
    with partial.open('wb') as stream:
        np.savez_compressed(stream, **values)
    os.replace(partial, path)


def render_views(output, config):
    import numpy as np
    import torch
    import nvdiffrast.torch as dr
    size = 512
    with np.load(output / 'mesh.npz', allow_pickle=False) as mesh:
        vertices = torch.tensor(mesh['vertices'], device='cuda', dtype=torch.float32)
        faces = torch.tensor(mesh['faces'], device='cuda', dtype=torch.int32)
        attrs = torch.tensor(np.concatenate([mesh['colors'], mesh['alpha']], -1), device='cuda', dtype=torch.float32)
    views, intrinsic = camera_matrices(config['mesh_views'], size)
    images, masks = [], []
    context = dr.RasterizeCudaContext()
    with torch.no_grad():
        for view in views:
            clip = clip_vertices(vertices, torch.tensor(view, device='cuda'), torch.tensor(intrinsic, device='cuda'), size)
            rast, _ = dr.rasterize(context, clip, faces, [size, size])
            rgba, _ = dr.interpolate(attrs.unsqueeze(0), rast, faces)
            mask = (rast[..., 3:] > 0).float() * rgba[..., 3:].clamp(0, 1)
            premultiplied = torch.cat([rgba[..., :3] * mask, mask], -1)
            aa = dr.antialias(premultiplied, rast, clip, faces)[0].flip(0).clamp(0, 1)
            image = (aa[..., :3] + 1 - aa[..., 3:]).clamp(0, 1)
            images.append((image.cpu().numpy() * 255).round().astype(np.uint8))
            masks.append((aa[..., 3:].cpu().numpy() * 255).round().astype(np.uint8))
    save_npz(output / 'mesh_views.npz', images=np.stack(images), masks=np.stack(masks), viewmats=views, K=intrinsic)
    return {'views': len(views), 'resolution': size, 'camera_convention': 'OpenCV world-to-camera, +Z forward, +Y down',
            'renderer': 'nvdiffrast 0.4.0, unlit vertex color, white background'}


def sample_surface(vertices, faces, colors, count, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) / 2
    if not np.isfinite(areas).all() or areas.sum() <= 1e-12:
        raise ValueError('Mesh yüzey alanı geçersiz.')
    selected = rng.choice(len(faces), count, p=areas/areas.sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, :1])
    weights = np.concatenate([1-root, root*(1-uv[:, 1:]), root*uv[:, 1:]], -1)
    points = (triangles[selected] * weights[:, :, None]).sum(1)
    rgb = (colors[faces[selected]] * weights[:, :, None]).sum(1)
    spacing = max(math.sqrt(float(areas.sum()) / count), 1e-4)
    return points.astype(np.float32), rgb.astype(np.float32), spacing


def fit_gaussians(output, config):
    import numpy as np
    import torch
    from gsplat import rasterization
    torch.manual_seed(config['trellis_seed'])
    with np.load(output / 'mesh.npz', allow_pickle=False) as mesh:
        xyz, rgb, spacing = sample_surface(mesh['vertices'], mesh['faces'], mesh['colors'],
                                           config['mesh_splat_cap'], config['trellis_seed'])
    # Fixed population enforces a hard cap; no hidden densification or unbounded growth.
    n = len(xyz)
    def parameter(value):
        return torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float32, device='cuda'))
    means = parameter(xyz)
    scales = parameter(np.full((n, 3), math.log(spacing), np.float32))
    quats = parameter(np.tile([1., 0, 0, 0], (n, 1)))
    alpha = parameter(np.full(n, 2., np.float32))
    color = parameter(np.log(np.clip(rgb, .001, .999) / np.clip(1-rgb, .001, .999)))
    params = [means, scales, quats, alpha, color]
    optimizer = torch.optim.Adam([{'params': [value], 'lr': lr} for value, lr in
                                  zip(params, [1e-4, .003, .001, .02, .01])], eps=1e-15)
    with np.load(output / 'mesh_views.npz', allow_pickle=False) as dataset:
        # Targets stay on CPU, one view per iteration, VRAM independent of view count.
        targets, masks = dataset['images'].copy(), dataset['masks'].copy()
        views = torch.tensor(dataset['viewmats'], device='cuda')
        intrinsic = torch.tensor(dataset['K'], device='cuda')[None]
    size = targets.shape[1]
    rng = np.random.default_rng(config['trellis_seed'])
    for step in range(config['mesh_fit_iterations']):
        index = int(rng.integers(len(targets)))
        target = torch.tensor(targets[index], device='cuda', dtype=torch.float32)[None] / 255
        mask = torch.tensor(masks[index], device='cuda', dtype=torch.float32)[None] / 255
        rendered, rendered_alpha, _ = rasterization(means, quats, scales.exp(), alpha.sigmoid(), color.sigmoid(),
            views[index:index+1], intrinsic, size, size, packed=True, backgrounds=torch.ones((1, 3), device='cuda'))
        loss = (rendered-target).abs().mean() + .2*(rendered_alpha-mask).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('3DGS fitting sonlu olmayan kayıp üretti; daha küçük splat bütçesi deneyin.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            means.clamp_(-1.5, 1.5)
            scales.clamp_(math.log(1e-5), math.log(.1))
            quats.copy_(torch.nn.functional.normalize(quats, dim=-1))
            alpha.clamp_(-10, 10)
            color.clamp_(-10, 10)
        if step % 100 == 0 or step + 1 == config['mesh_fit_iterations']:
            print(f'3DGS {step+1}/{config["mesh_fit_iterations"]} · L1+alpha {loss.item():.5f}', flush=True)
    def cpu(value):
        return value.detach().float().cpu().numpy()
    save_npz(output / 'gaussians_raw.npz', xyz=cpu(means), scales=cpu(scales.exp()), rotations=cpu(quats),
             opacity=cpu(alpha.sigmoid()), dc=cpu((color.sigmoid()-.5)/.28209479177387814))
    return {'native_splat_count': n, 'fit_iterations': config['mesh_fit_iterations'], 'final_loss': loss.item(),
            'trainer': 'gsplat 1.5.3 fixed-population Adam, SH0; no densification', 'source_up': '+Z'}


def check_kernels():
    import torch
    import xformers.ops
    import nvdiffrast.torch as dr
    import cumesh
    import flex_gemm
    import o_voxel
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from gsplat import rasterization
    q = torch.randn(1, 16, 2, 32, device='cuda', dtype=torch.float16)
    xformers.ops.memory_efficient_attention(q, q, q)
    vertices = torch.tensor([[[-.5, -.5, 0, 1], [.5, -.5, 0, 1], [0, .5, 0, 1]]], device='cuda')
    dr.rasterize(dr.RasterizeCudaContext(), vertices, torch.tensor([[0, 1, 2]], device='cuda', dtype=torch.int32), [16, 16])
    means = torch.tensor([[0., 0., 2.]], device='cuda', requires_grad=True)
    colors, alpha, _ = rasterization(means, torch.tensor([[1., 0, 0, 0]], device='cuda'),
        torch.full((1, 3), .1, device='cuda'), torch.full((1,), .8, device='cuda'), torch.ones((1, 3), device='cuda'),
        torch.eye(4, device='cuda')[None], torch.tensor([[[16., 0, 8], [0, 16., 8], [0, 0, 1]]], device='cuda'), 16, 16)
    (colors.sum()+alpha.sum()).backward()
    torch.cuda.synchronize()
    print('✓ TRELLIS.2 import, xformers/nvdiffrast/gsplat ileri-geri CUDA kontrolleri; henüz model indirilmedi.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['check', 'gaussian', 'mesh', 'views', 'fit', 'export'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = validate_settings(json.loads(Path(args.config).read_text(encoding='utf8')))
    if args.stage == 'gaussian':
        # Release the complete model process before rasterizing and training.
        for stage in ('mesh', 'views', 'fit'):
            subprocess.run([sys.executable, __file__, stage, '--config', args.config], check=True)
        return
    import torch
    if sys.version_info[:2] != (3, 11) or torch.__version__.split('+')[0] != '2.7.1' or torch.version.cuda != '12.8':
        raise RuntimeError('TRELLIS.2 ortamı Python 3.11 / Torch 2.7.1 cu128 olmalı; kurulum hücresini tekrar çalıştırın.')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] / 1024**3 < estimate_requirements(config)['geometry_vram'] - 2:
        raise RuntimeError('TRELLIS.2 için boş GPU belleği yetersiz; başka işlemleri kapatın veya daha büyük Colab GPU seçin.')
    if args.stage == 'check':
        check_kernels()
        return
    from text_to_splat_helpers import StageCache, atomic_json
    from text_to_splat_runtime import provenance_for, export_gaussians
    cache = StageCache(config['output_dir'], config, provenance_for(config))
    import importlib.metadata
    cache.data.setdefault('environments', {})['mesh'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'sources': SOURCE_PINS,
        'source_licenses': {'TRELLIS.2': 'MIT', 'CuMesh': 'MIT', 'FlexGEMM': 'MIT', 'utils3d': 'MIT',
                            'nvdiffrast': 'NVIDIA Source Code License (1-Way Commercial)', 'gsplat': 'Apache-2.0'},
        'packages': {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions() if dist.metadata['Name']}}
    atomic_json(cache.path, cache.data)
    if args.stage == 'mesh':
        cache.run('mesh', ['mesh.npz'], lambda: generate_mesh(cache.output, config), ['image'])
    elif args.stage == 'views':
        cache.run('views', ['mesh_views.npz'], lambda: render_views(cache.output, config), ['image', 'mesh'])
    elif args.stage == 'fit':
        cache.run('gaussian', ['gaussians_raw.npz'], lambda: fit_gaussians(cache.output, config), ['image', 'mesh', 'views'])
    else:
        cache.run('export', ['target.ply', 'turntable.gif'], lambda: export_gaussians(cache.output, config), ['image', 'mesh', 'views', 'gaussian'])


if __name__ == '__main__':
    main()

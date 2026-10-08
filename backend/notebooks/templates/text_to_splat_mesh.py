"""Experimental TRELLIS.2 -> textured mesh -> known cameras -> gsplat 1.5.3.

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
    from text_to_splat_settings import DINO3_MODEL, RECONSTRUCTION_MODELS, require_mesh_memory, validate_settings
except ModuleNotFoundError:
    from .text_to_splat_settings import DINO3_MODEL, RECONSTRUCTION_MODELS, require_mesh_memory, validate_settings

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


def pinned_model_directory(work, pipeline_type='512'):
    from huggingface_hub import hf_hub_download, snapshot_download
    from text_to_splat_helpers import atomic_json
    model = RECONSTRUCTION_MODELS['trellis2']
    path = hf_hub_download(model['id'], 'pipeline.json', revision=model['revision'])
    pipeline = json.loads(Path(path).read_text())
    args = pipeline['args']
    model_dir = work / 'trellis2-model'
    model_dir.mkdir(exist_ok=True)
    # Both cascades use 512 shape + 1024 shape/texture; no 1536 weights exist.
    if pipeline_type == '512':
        args['models'] = {key: value for key, value in args['models'].items() if not key.endswith('_1024')}
    else:
        args['models'].pop('tex_slat_flow_model_512', None)
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
    args['default_pipeline_type'] = pipeline_type
    atomic_json(model_dir / 'pipeline.json', pipeline)
    return model_dir


def mesh_arrays(textured_mesh):
    """to_glb returns +Y-up trimesh; undo that rotation for our +Z-up cameras.

    PIL texture rows are top-down; trimesh UV v is bottom-up. nvdiffrast samples
    row zero at v=0, so invert v here (independent of the output image flip).
    """
    import numpy as np
    visual = textured_mesh.visual
    material = getattr(visual, 'material', None)
    texture = getattr(material, 'baseColorTexture', None)
    uv = getattr(visual, 'uv', None)
    vertices = np.asarray(textured_mesh.vertices, dtype=np.float32)
    values = dict(vertices=vertices[:, [0, 2, 1]] * [1, -1, 1],
                  faces=np.asarray(textured_mesh.faces, dtype=np.int32))
    if texture is not None and uv is not None:
        rgba = np.asarray(texture.convert('RGBA'), dtype=np.uint8)
        factor = getattr(material, 'baseColorFactor', None)
        if factor is not None:
            factor = np.asarray(factor, dtype=np.float32)
            if factor.max() > 1:
                factor /= 255
            rgba = np.rint(rgba.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
        uv = np.asarray(uv, dtype=np.float32).copy()
        uv[:, 1] = 1 - uv[:, 1]
        # Vertex texture samples only seed the fit; targets always sample UVs per pixel.
        ij = np.rint(uv.clip(0, 1) * [rgba.shape[1]-1, rgba.shape[0]-1]).astype(int)
        colors = rgba[ij[:, 1], ij[:, 0]].astype(np.float32) / 255
        values.update(uv=uv, texture=rgba)
    else:
        colors = np.asarray(visual.to_color().vertex_colors if hasattr(visual, 'to_color')
                            else visual.vertex_colors, dtype=np.float32) / 255
    values.update(colors=colors[:, :3], alpha=colors[:, 3:4])
    return values


def generate_mesh(output, config):
    import torch
    import o_voxel
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline_type = config['trellis2_pipeline_type']
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        str(pinned_model_directory(Path(config['work_dir']), pipeline_type)))
    pipeline.cuda()
    with Image.open(output / 'cutout.png') as image, torch.inference_mode():
        # The RGBA stage already crops/composites foreground; avoid optional BRIA
        # on an opaque mask while retaining upstream's normal preprocessing.
        if image.mode != 'RGBA' or image.getextrema()[3] == (255, 255):
            raise RuntimeError('TRELLIS.2 için saydam arka planlı cutout.png gerekli.')
        mesh = pipeline.run(image, seed=config['trellis_seed'], pipeline_type=pipeline_type,
            max_num_tokens=49152,
            sparse_structure_sampler_params={'steps': config['sparse_steps'], 'guidance_strength': config['sparse_cfg']},
            shape_slat_sampler_params={'steps': config['slat_steps'], 'guidance_strength': config['slat_cfg']},
            tex_slat_sampler_params={'steps': config['slat_steps'], 'guidance_strength': 1.0})[0]
        # Pinned example.py and o-voxel/o_voxel/postprocess.py: bake voxel PBR
        # attributes into a UV atlas. Preserve fine geometry; no coarse 500k pass.
        mesh.simplify(16777216)
        textured = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=[[-.5, -.5, -.5], [.5, .5, .5]], decimation_target=1000000,
            texture_size=config['mesh_texture_size'], remesh=False, verbose=True)
        textured.export(str(output / 'mesh.glb'))  # embedded PNG, broad GLB compatibility
        values = mesh_arrays(textured)
        effective_resolution = round(1 / float(mesh.voxel_size))
    if len(values['vertices']) < 4 or len(values['faces']) < 2:
        raise RuntimeError('TRELLIS.2 boş mesh üretti; yeni istem/seed deneyin.')
    save_npz(output / 'mesh.npz', **values)
    return {'vertices': len(values['vertices']), 'faces': len(values['faces']), 'pipeline_type': pipeline_type,
            'effective_resolution': effective_resolution, 'max_num_tokens': 49152,
            'texture_size': config['mesh_texture_size'], 'texture_guidance': 1.0,
            'appearance': 'UV PBR base-color + alpha; unlit targets, metallic/roughness retained in GLB only'}


def save_npz(path, **values):
    import numpy as np
    partial = Path(str(path) + '.partial')
    with partial.open('wb') as stream:
        np.savez_compressed(stream, **values)
    os.replace(partial, path)


def split_view_indices(count):
    import numpy as np
    indices = np.arange(count)
    return indices[indices % 12 != 0], indices[indices % 12 == 0]


def render_views(output, config):
    import numpy as np
    import torch
    import nvdiffrast.torch as dr
    from PIL import Image
    size, count = config['mesh_render_resolution'], config['mesh_views']
    with np.load(output / 'mesh.npz', allow_pickle=False) as mesh:
        vertices = torch.tensor(mesh['vertices'], device='cuda', dtype=torch.float32)
        faces = torch.tensor(mesh['faces'], device='cuda', dtype=torch.int32)
        attrs = torch.tensor(np.concatenate([mesh['colors'], mesh['alpha']], -1), device='cuda', dtype=torch.float32)
        uv = torch.tensor(mesh['uv'], device='cuda') if 'texture' in mesh else None
        texture = torch.tensor(mesh['texture'], device='cuda', dtype=torch.float32)[None] / 255 if uv is not None else None
    views, intrinsic = camera_matrices(count, size)
    # Disk-backed arrays: neither GPU nor CPU RAM grows with view count.
    images = np.lib.format.open_memmap(output / 'mesh_images.npy', mode='w+', dtype=np.uint8, shape=(count, size, size, 3))
    masks = np.lib.format.open_memmap(output / 'mesh_masks.npy', mode='w+', dtype=np.uint8, shape=(count, size, size, 1))
    preview = Image.new('RGB', (4 * 256, 4 * 256), 'white')
    preview_ids = list(np.linspace(0, count-1, min(count, 16), dtype=int))
    context = dr.RasterizeCudaContext()
    with torch.no_grad():
        for index, view in enumerate(views):
            clip = clip_vertices(vertices, torch.tensor(view, device='cuda'), torch.tensor(intrinsic, device='cuda'), size)
            rast, _ = dr.rasterize(context, clip, faces, [size, size])
            if texture is not None:
                texcoord, _ = dr.interpolate(uv[None], rast, faces)
                rgba = dr.texture(texture, texcoord, filter_mode='linear', boundary_mode='clamp')
            else:
                rgba, _ = dr.interpolate(attrs.unsqueeze(0), rast, faces)
            mask = (rast[..., 3:] > 0).float() * rgba[..., 3:].clamp(0, 1)
            premultiplied = torch.cat([rgba[..., :3] * mask, mask], -1)
            aa = dr.antialias(premultiplied, rast, clip, faces)[0].flip(0).clamp(0, 1)
            image = (aa[..., :3] + 1 - aa[..., 3:]).clamp(0, 1)
            images[index] = (image.cpu().numpy() * 255).round().astype(np.uint8)
            masks[index] = (aa[..., 3:].cpu().numpy() * 255).round().astype(np.uint8)
            if index in preview_ids:
                tile = preview_ids.index(index)
                preview.paste(Image.fromarray(images[index]).resize((256, 256), Image.Resampling.LANCZOS),
                              ((tile % 4) * 256, (tile // 4) * 256))
    images.flush()
    masks.flush()
    del images, masks
    train, heldout = split_view_indices(count)
    save_npz(output / 'mesh_views.npz', viewmats=views, K=intrinsic, train_indices=train, eval_indices=heldout)
    preview.save(output / 'views_preview.jpg', quality=92)
    return {'views': count, 'resolution': size, 'train_indices': train.tolist(), 'eval_indices': heldout.tolist(),
            'camera_convention': 'OpenCV world-to-camera, +Z forward, +Y down',
            'renderer': 'nvdiffrast 0.4.0, unlit UV base-color (vertex fallback), white background'}


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


def capped_strategy(cap, iterations):
    """gsplat 1.5.3 DefaultStrategy with pre-allocation growth budgeting.

    Each duplicate or two-way split adds exactly one Gaussian. Keep upstream
    optimizer/state surgery, pruning and reset; bound candidates BEFORE growth.
    No MCMC relocation or its binomial-table kernel is used.
    """
    import torch
    from gsplat.strategy import DefaultStrategy
    from gsplat.strategy.ops import duplicate, split

    class CappedDefaultStrategy(DefaultStrategy):
        @torch.no_grad()
        def _grow_gs(self, params, optimizers, state, step):
            n = len(params['means'])
            room = min(cap - n, max(1, n // 10))  # at most 10% growth per refinement
            if room <= 0:
                return 0, 0
            grads = state['grad2d'] / state['count'].clamp_min(1)
            candidates = torch.where(grads > self.grow_grad2d)[0]
            if len(candidates) > room:
                candidates = candidates[torch.topk(grads[candidates], room).indices]
            chosen = torch.zeros(n, dtype=torch.bool, device=grads.device)
            chosen[candidates] = True
            small = params['scales'].exp().max(dim=-1).values <= self.grow_scale3d * state['scene_scale']
            dupli, splitting = chosen & small, chosen & ~small
            nd, ns = int(dupli.sum()), int(splitting.sum())
            if nd:
                duplicate(params=params, optimizers=optimizers, state=state, mask=dupli)
            if ns:
                splitting = torch.cat([splitting, torch.zeros(nd, dtype=torch.bool, device=grads.device)])
                split(params=params, optimizers=optimizers, state=state, mask=splitting,
                      revised_opacity=self.revised_opacity)
            assert len(params['means']) <= cap
            return nd, ns

    return CappedDefaultStrategy(absgrad=True, grow_grad2d=.0008,
        refine_start_iter=min(500, iterations // 10), refine_stop_iter=min(15000, iterations * 3 // 4),
        refine_every=100, reset_every=3000, prune_opa=.005)


def image_ssim(prediction, target):
    """RGB NHWC [0,1], 11px Gaussian (sigma 1.5), valid-window mean SSIM."""
    import torch
    import torch.nn.functional as F
    x, y = prediction.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)
    axis = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    weights = torch.exp(-axis.square() / (2 * 1.5**2))
    weights /= weights.sum()
    horizontal = weights.reshape(1, 1, 1, 11).expand(3, 1, 1, 11)
    vertical = weights.reshape(1, 1, 11, 1).expand(3, 1, 11, 1)
    def blur(value):
        return F.conv2d(F.conv2d(value, horizontal, groups=3), vertical, groups=3)
    mx, my = blur(x), blur(y)
    vx, vy, cov = blur(x*x)-mx*mx, blur(y*y)-my*my, blur(x*y)-mx*my
    return (((2*mx*my + .01**2) * (2*cov + .03**2)) /
            ((mx*mx + my*my + .01**2) * (vx + vy + .03**2))).mean()


def fit_gaussians(output, config):
    import numpy as np
    import torch
    from gsplat import rasterization
    torch.manual_seed(config['trellis_seed'])
    cap, iterations, degree = config['mesh_splat_cap'], config['mesh_fit_iterations'], config['mesh_sh_degree']
    initial = min(100000, max(1000, cap // 5), cap)
    with np.load(output / 'mesh.npz', allow_pickle=False) as mesh:
        xyz, rgb, spacing = sample_surface(mesh['vertices'], mesh['faces'], mesh['colors'],
                                           initial, config['trellis_seed'])
    n = len(xyz)
    scene_scale = 2.2  # camera sphere radius, matching simple_trainer's camera extent
    def parameter(value):
        return torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float32, device='cuda'))
    params = {
        'means': parameter(xyz), 'scales': parameter(np.full((n, 3), math.log(spacing), np.float32)),
        'quats': parameter(np.tile([1., 0, 0, 0], (n, 1))),
        'opacities': parameter(np.full(n, math.log(.1/.9), np.float32)),
        'sh0': parameter(((rgb-.5)/.28209479177387814)[:, None]),
        'shN': parameter(np.zeros((n, (degree+1)**2-1, 3), np.float32)),
    }
    rates = dict(means=1.6e-4*scene_scale, scales=5e-3, quats=1e-3,
                 opacities=5e-2, sh0=2.5e-3, shN=2.5e-3/20)
    optimizers = {key: torch.optim.Adam([value], lr=rates[key], eps=1e-15) for key, value in params.items()}
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers['means'], gamma=.01**(1/iterations))
    strategy = capped_strategy(cap, iterations)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    targets = np.load(output / 'mesh_images.npy', mmap_mode='r', allow_pickle=False)
    masks = np.load(output / 'mesh_masks.npy', mmap_mode='r', allow_pickle=False)
    with np.load(output / 'mesh_views.npz', allow_pickle=False) as dataset:
        views = torch.tensor(dataset['viewmats'], device='cuda')
        intrinsic = torch.tensor(dataset['K'], device='cuda')[None]
        train, heldout = dataset['train_indices'], dataset['eval_indices']
    size = targets.shape[1]
    rng = np.random.default_rng(config['trellis_seed'])
    def render(index, sh_degree):
        return rasterization(params['means'], params['quats'], params['scales'].exp(),
            params['opacities'].sigmoid(), torch.cat([params['sh0'], params['shN']], 1),
            views[index:index+1], intrinsic, size, size, sh_degree=sh_degree, packed=True,
            absgrad=True, near_plane=.01, far_plane=10., backgrounds=torch.ones((1, 3), device='cuda'))
    for step in range(iterations):
        index = int(rng.choice(train))  # held-out views are never optimized
        target = torch.tensor(targets[index], device='cuda', dtype=torch.float32)[None] / 255
        mask = torch.tensor(masks[index], device='cuda', dtype=torch.float32)[None] / 255
        rendered, rendered_alpha, info = render(index, min(step // 1000, degree))
        strategy.step_pre_backward(params, optimizers, state, step, info)
        loss = (rendered-target).abs().mean() + .2*(rendered_alpha-mask).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('3DGS fitting sonlu olmayan kayıp üretti; daha küçük splat bütçesi deneyin.')
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        # Upstream simple_trainer calls strategy AFTER optimizer updates.
        strategy.step_post_backward(params, optimizers, state, step, info, packed=True)
        if not len(params['means']):
            raise RuntimeError('3DGS budaması tüm Gaussianları sildi; farklı seed veya mesh deneyin.')
        with torch.no_grad():
            params['means'].clamp_(-1.5, 1.5)
            params['scales'].clamp_(math.log(1e-5), math.log(.1))
            params['quats'].copy_(torch.nn.functional.normalize(params['quats'], dim=-1))
            params['opacities'].clamp_(-10, 10)
        if step % 100 == 0 or step + 1 == iterations:
            print(f'3DGS {step+1}/{iterations} · {len(params["means"]):,}/{cap:,} splat · L1+alpha {loss.item():.5f}', flush=True)
        del rendered, rendered_alpha, info, target, mask
    scores = []
    with torch.no_grad():
        for index in heldout:
            predicted, _, _ = render(int(index), degree)
            predicted = predicted.clamp(0, 1)
            target = torch.tensor(targets[index], device='cuda', dtype=torch.float32)[None] / 255
            psnr = -10 * torch.log10((predicted-target).square().mean().clamp_min(1e-10))
            scores.append((float(psnr), float(image_ssim(predicted, target))))
    metrics = dict(psnr=float(np.mean(scores, axis=0)[0]), ssim=float(np.mean(scores, axis=0)[1]),
                   view_count=len(heldout), indices=heldout.tolist(),
                   protocol='held-out every 12th Fibonacci view; full image RGB [0,1], white background; SSIM 11px sigma1.5 valid')
    print(f'Kontrol görüşleri: PSNR {metrics["psnr"]:.2f} dB · SSIM {metrics["ssim"]:.4f}', flush=True)
    def cpu(value):
        return value.detach().float().cpu().numpy()
    save_npz(output / 'gaussians_raw.npz', xyz=cpu(params['means']), scales=cpu(params['scales'].exp()),
             rotations=cpu(params['quats']), opacity=cpu(params['opacities'].sigmoid()),
             dc=cpu(params['sh0']), sh_rest=cpu(params['shN']))
    return {'native_splat_count': len(params['means']), 'initial_splat_count': n,
            'fit_iterations': iterations, 'final_loss': loss.item(), 'evaluation': metrics,
            'trainer': 'gsplat 1.5.3 capped DefaultStrategy, packed absgrad, progressive SH',
            'sh_degree': degree, 'scene_scale': scene_scale, 'learning_rates': rates, 'source_up': '+Z'}


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
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU bulunamadı; uygun Colab GPU seçin.')
    require_mesh_memory(config, torch.cuda.get_device_properties(0).total_memory / 1024**3,
                        torch.cuda.mem_get_info()[0] / 1024**3)
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
        cache.run('mesh', ['mesh.npz', 'mesh.glb'], lambda: generate_mesh(cache.output, config), ['image'])
    elif args.stage == 'views':
        cache.run('views', ['mesh_views.npz', 'mesh_images.npy', 'mesh_masks.npy', 'views_preview.jpg'], lambda: render_views(cache.output, config), ['image', 'mesh'])
    elif args.stage == 'fit':
        cache.run('gaussian', ['gaussians_raw.npz'], lambda: fit_gaussians(cache.output, config), ['image', 'mesh', 'views'])
    else:
        cache.run('export', ['target.ply', 'turntable.gif'], lambda: export_gaussians(cache.output, config), ['image', 'mesh', 'views', 'gaussian'])


if __name__ == '__main__':
    main()

"""Portable, CPU-only 3DGS conversion and approximate turntable utilities.

This file is embedded in the generated notebook. No application imports.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    os.replace(temp, path)


def sh_basis(directions, degree):
    """Real SH basis/order/signs used by original 3DGS and gsplat, through SH3."""
    d = np.asarray(directions, dtype=np.float64)
    x, y, z = (d / np.linalg.norm(d, axis=-1, keepdims=True)).T
    basis = [np.full_like(x, .28209479177387814)]
    if degree >= 1:
        c = .4886025119029199
        basis += [-c*y, c*z, -c*x]
    if degree >= 2:
        basis += [1.0925484305920792*x*y, -1.0925484305920792*y*z,
                  .31539156525252005*(2*z*z-x*x-y*y), -1.0925484305920792*x*z,
                  .5462742152960396*(x*x-y*y)]
    if degree >= 3:
        basis += [-.5900435899266435*y*(3*x*x-y*y), 2.890611442640554*x*y*z,
                  -.4570457994644658*y*(4*z*z-x*x-y*y), .3731763325901154*z*(2*z*z-3*x*x-3*y*y),
                  -.4570457994644658*x*(4*z*z-x*x-y*y), 1.445305721320277*z*(x*x-y*y),
                  -.5900435899266435*x*(x*x-3*y*y)]
    return np.stack(basis, axis=-1)


def rotate_sh_to_y_up(rest):
    """Rotate appearance with geometry: f_out(d) = f_source(R.T @ d).

    Solve the small basis change in float64; apply in float32 chunks so a 3M
    cloud never materializes a full double-precision SH coefficient array.
    """
    rest = np.asarray(rest, dtype=np.float32)
    if rest.ndim != 3 or rest.shape[2] != 3 or rest.shape[1] not in (0, 3, 8, 15):
        raise ValueError('SH rest boyutları geçersiz.')
    if not np.isfinite(rest).all():
        raise ValueError('SH rest sonlu sayılar olmalı.')
    if not rest.shape[1]:
        return rest.copy()
    degree = int(math.sqrt(rest.shape[1] + 1)) - 1
    i = np.arange(64)
    z = 1 - 2*(i+.5)/64
    phi = i * math.pi * (3-math.sqrt(5))
    directions = np.stack([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z], -1)
    source_directions = directions[:, [0, 2, 1]] * [1, -1, 1]
    transform = np.linalg.lstsq(sh_basis(directions, degree)[:, 1:],
                               sh_basis(source_directions, degree)[:, 1:], rcond=None)[0].astype(np.float32)
    output = np.empty_like(rest)
    for start in range(0, len(rest), 65536):
        output[start:start+65536] = np.einsum('ij,njc->nic', transform, rest[start:start+65536])
    return output


def normalize_gaussians(xyz, scales, rotations, opacity, dc, target_count=None, seed=0, sh_rest=None):
    """Activated TRELLIS values -> standard 3DGS, +Y up, longest centre span 1.

    Rotation is a proper rotation (x,y,z)->(x,z,-y), NOT a reflection. TRELLIS
    emits SH0; its higher coefficients remain zero padded. Mesh SH is rotated.
    target_count is a deterministic upper cap, never fabricated extra splats.
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in (xyz, scales, rotations, opacity, dc)]
    xyz, scales, rotations, opacity, dc = arrays
    n = len(xyz)
    if opacity.shape == (n, 1):
        opacity = opacity[:, 0]
    if dc.shape == (n, 1, 3):
        dc = dc[:, 0]
    if n == 0 or any(a.shape != shape for a, shape in zip(
        (xyz, scales, rotations, opacity, dc), ((n, 3), (n, 3), (n, 4), (n,), (n, 3))
    )):
        raise ValueError('Gaussian dizilerinin boyutları geçersiz veya boş.')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Gaussian verisinde NaN veya sonsuz değer var.')
    if np.any(scales <= 0) or np.any((opacity < 0) | (opacity > 1)):
        raise ValueError('Ölçek pozitif, etkin opaklık 0–1 arasında olmalı.')
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('Sıfır uzunluklu quaternion geçersiz.')
    rotations = rotations / norms
    lo, hi = xyz.min(0), xyz.max(0)
    span = float(np.max(hi - lo))
    if span <= 1e-10:
        raise ValueError('Nesne boyutu sıfır; normalizasyon yapılamadı.')
    center = (lo + hi) / 2
    factor = 1.0 / span
    xyz = (xyz - center) * factor
    xyz = xyz[:, [0, 2, 1]] * [1, 1, -1]
    # Left-multiply by (sqrt(1/2), -sqrt(1/2), 0, 0), wxyz.
    w, x, y, z = rotations.T
    rotations = np.stack((w + x, x - w, y + z, z - y), axis=1) / math.sqrt(2)
    indices = np.arange(n)
    if target_count is not None:
        if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 1:
            raise ValueError('Hedef Gaussian sayısı pozitif tam sayı olmalı.')
        if target_count < n:
            weights = np.maximum(opacity * np.max(scales, axis=1) ** 2, 1e-20)
            indices = np.sort(np.random.default_rng(seed).choice(n, target_count, replace=False, p=weights / weights.sum()))
    if sh_rest is not None and (np.ndim(sh_rest) != 3 or np.shape(sh_rest)[0] != n):
        raise ValueError('SH rest Gaussian sayısı uyuşmuyor.')
    rest = (np.zeros((len(indices), 15, 3), dtype=np.float32) if sh_rest is None
            else rotate_sh_to_y_up(np.asarray(sh_rest)[indices]))
    alpha = np.clip(opacity[indices], 1e-6, 1 - 1e-6)
    cloud = {
        'means': xyz[indices].astype(np.float32),
        'log_scales': np.log(scales[indices] * factor).astype(np.float32),
        'quats': rotations[indices].astype(np.float32),
        'opacities': (np.log(alpha) - np.log1p(-alpha)).astype(np.float32),
        'sh_dc': dc[indices].astype(np.float32),
        'sh_rest': rest,
    }
    if not all(np.isfinite(a).all() for a in cloud.values()):
        raise ValueError('Normalizasyon sonucu float32 sınırlarını aştı.')
    metadata = {
        'source_up': '+Z', 'output_up': '+Y', 'handedness': 'right',
        'rotation_source_to_output': [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        'source_bbox_center': center.tolist(), 'uniform_scale': factor,
        'units': 'arbitrary; longest original Gaussian-centre AABB side = 1, not metres',
        'input_count': n, 'output_count': len(indices), 'target_splat_count': target_count,
        'count_policy': 'seeded opacity-times-largest-scale-squared thinning; upper cap only',
        'source_sh_degree': 0 if sh_rest is None else int(math.sqrt(rest.shape[1]+1))-1,
        'export_sh_degree': int(math.sqrt(rest.shape[1]+1))-1,
        'higher_sh': 'zero padded' if sh_rest is None else 'rotated with geometry',
    }
    return cloud, metadata


def write_ply(path, cloud):
    """Binary little-endian INRIA PLY; channel-major SH, log scales/logit alpha."""
    n = len(cloud['means'])
    rest = cloud['sh_rest']
    k = rest.shape[1]
    if k not in (0, 3, 8, 15) or rest.shape != (n, k, 3) or n < 1:
        raise ValueError('SH dizisi veya Gaussian sayısı geçersiz.')
    shapes = {'means': (n, 3), 'log_scales': (n, 3), 'quats': (n, 4),
              'opacities': (n,), 'sh_dc': (n, 3)}
    if any(np.shape(cloud[key]) != shape for key, shape in shapes.items()):
        raise ValueError('PLY alan boyutları uyuşmuyor.')
    if not all(np.isfinite(a).all() for a in cloud.values()):
        raise ValueError('PLY alanları sonlu sayılar olmalı.')
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
    names += [f'f_rest_{i}' for i in range(3 * k)] + ['opacity']
    names += [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    header = 'ply\nformat binary_little_endian 1.0\ncomment text_to_splat +Y up; arbitrary units\n'
    header += f'element vertex {n}\n' + ''.join(f'property float {name}\n' for name in names) + 'end_header\n'
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    with temp.open('wb') as stream:
        stream.write(header.encode('ascii'))
        # Stream blocks: a 3M SH3 cloud must not allocate another giant float64
        # interleaved array plus float32 and bytes copies just to serialize it.
        for start in range(0, n, 65536):
            end = min(start+65536, n)
            values = np.column_stack((cloud['means'][start:end], np.zeros((end-start, 3), np.float32),
                cloud['sh_dc'][start:end], rest[start:end].transpose(0, 2, 1).reshape(end-start, 3*k),
                cloud['opacities'][start:end], cloud['log_scales'][start:end], cloud['quats'][start:end]))
            stream.write(values.astype('<f4', copy=False).tobytes())
    os.replace(temp, path)
    return path


def quaternion_matrices(quats):
    q = np.asarray(quats, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=1).reshape(-1, 3, 3)


def render_turntable(path, cloud, frames=36, size=256, max_splats=12000):
    """CPU orthographic, depth-sorted anisotropic Gaussian/SH0 GIF.

    Bounded preview only: selected high-importance splats, 3-sigma cut-off,
    minimum pixel filter. Does not depend on a CUDA rasterizer or mesh decoder.
    """
    from PIL import Image
    n = len(cloud['means'])
    scales = np.exp(cloud['log_scales'])
    alpha = 1 / (1 + np.exp(-np.clip(cloud['opacities'], -30, 30)))
    importance = alpha * np.max(scales, axis=1) ** 2
    ids = np.argsort(importance, kind='stable')[-max_splats:]
    xyz, scales, alpha = cloud['means'][ids], scales[ids], alpha[ids]
    color = np.clip(0.5 + 0.28209479177387814 * cloud['sh_dc'][ids], 0, 1)
    rot = quaternion_matrices(cloud['quats'][ids])
    cov = (rot * scales[:, None, :] ** 2) @ rot.transpose(0, 2, 1)
    extent = max(0.65, float(np.linalg.norm(xyz, axis=1).max()) + float(scales.max()) * 3)
    zoom = size * 0.43 / extent
    pictures = []
    elevation = math.radians(15)
    for angle in np.linspace(0, 2 * math.pi, frames, endpoint=False):
        right = np.array([math.cos(angle), 0, -math.sin(angle)])
        forward = np.array([math.sin(angle)*math.cos(elevation), math.sin(elevation), math.cos(angle)*math.cos(elevation)])
        up = np.cross(forward, right)
        view = np.stack((right, -up))
        uv = xyz @ view.T * zoom + size / 2
        cov2 = np.einsum('ai,nij,bj->nab', view, cov, view) * zoom ** 2 + np.eye(2)[None] * 0.3
        radii = np.ceil(3 * np.sqrt(np.linalg.eigvalsh(cov2)[:, -1])).astype(int)
        inv = np.linalg.inv(cov2)
        canvas = np.full((size, size, 3), 0.08, dtype=np.float64)
        for i in np.argsort(xyz @ forward):  # far -> near, camera is at +forward
            u, v = uv[i]
            r = min(int(radii[i]), size)
            x0, x1 = max(0, int(u)-r), min(size, int(u)+r+1)
            y0, y1 = max(0, int(v)-r), min(size, int(v)+r+1)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            dx, dy = xx + 0.5 - u, yy + 0.5 - v
            exponent = inv[i, 0, 0]*dx*dx + 2*inv[i, 0, 1]*dx*dy + inv[i, 1, 1]*dy*dy
            a = np.where(exponent <= 9, np.exp(-0.5*exponent)*alpha[i], 0)[..., None]
            canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1]*(1-a) + color[i]*a
        pictures.append(Image.fromarray(np.uint8(np.clip(canvas, 0, 1) * 255)))
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    pictures[0].save(temp, format='GIF', save_all=True, append_images=pictures[1:], duration=100,
                     loop=0, optimize=False, disposal=2)
    os.replace(temp, path)
    return {'renderer': 'CPU orthographic anisotropic Gaussian SH0 approximation',
            'frames': frames, 'size': size, 'preview_count': min(n, max_splats), 'full_count': n}


class StageCache:
    """Checksummed, configuration-bound Drive stages; manifest committed last."""
    def __init__(self, output_dir, config, provenance):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / 'manifest.json'
        identity = {'config': config, 'provenance': provenance}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf8'))
            if self.data.get('fingerprint') != fingerprint:
                raise ValueError('Bu çıktı klasörü farklı ayarlara ait. OUTPUT_DIR için yeni bir klasör seçin.')
        else:
            existing = [p for p in self.output.iterdir() if p.name != 'logs']
            if existing:
                raise ValueError('Çıktı klasöründe manifest olmadan dosyalar var. Yeni bir OUTPUT_DIR seçin.')
            self.data = {'schema_version': 1, 'fingerprint': fingerprint, **identity, 'stages': {}, 'status': 'new'}
            atomic_json(self.path, self.data)

    def valid(self, name):
        stage = self.data['stages'].get(name, {})
        return stage.get('status') == 'complete' and bool(stage.get('files')) and all(
            (self.output / f).is_file() and file_sha256(self.output / f) == checksum
            for f, checksum in stage['files'].items())

    def run(self, name, filenames, function, dependencies=()):
        if any(not self.valid(dep) for dep in dependencies):
            raise ValueError('Önce önceki hücreyi yeniden çalıştırın; eksik veya bozuk ara çıktı var.')
        if self.valid(name):
            print(f'✓ {name}: doğrulanmış çıktı bulundu; tekrar kullanılacak.', flush=True)
            return
        # Any downstream stages are invalid after a stage rerun, even if files remain.
        order = ['image', 'mesh', 'views', 'gaussian', 'export']
        if name in order:
            for downstream in order[order.index(name) + 1:]:
                self.data['stages'].pop(downstream, None)
        self.data['status'] = 'running'
        self.data['stages'][name] = {'status': 'running'}
        atomic_json(self.path, self.data)
        start = time.monotonic()
        try:
            extra = function() or {}
            checksums = {f: file_sha256(self.output / f) for f in filenames}
            self.data['stages'][name] = {'status': 'complete', 'seconds': time.monotonic()-start,
                                         'files': checksums, **extra}
            self.data['status'] = 'complete' if name == 'export' else 'partial'
        except Exception as exc:
            self.data['stages'][name] = {'status': 'failed', 'seconds': time.monotonic()-start, 'error': str(exc)}
            self.data['status'] = 'failed'
            raise
        finally:
            atomic_json(self.path, self.data)

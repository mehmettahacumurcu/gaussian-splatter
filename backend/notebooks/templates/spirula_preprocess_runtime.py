"""Standalone preprocessing recipe embedded verbatim in generated notebooks.

No project imports, training code or shell interpolation. The caller supplies
the existing notebook log runner; heavy stages have no automatic time cutoff.
"""
import hashlib
import json
import math
import re
import shutil
import struct
import urllib.request
import zipfile
from fractions import Fraction
from pathlib import Path, PurePosixPath

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
SAM_MODELS = {
    'sam3-q4_0': (706606590, '5dafc790c8493319f542f575e718084f4e0a452fd7e483c64853d33ffe3f1889'),
    'sam3-f16': (1837873470, '1c8cef822a6f0f0908c8e7c51139c1061a842ebfbc88e8b0bb1d8e342ec50f8e'),
}


def sha256_stream(stream):
    # hashlib.file_digest requires Python 3.11; keep offline checks usable on 3.10.
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def image_files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def safe_image_name(name):
    relative = PurePosixPath(name.replace('\\', '/'))
    if not name or relative.is_absolute() or '..' in relative.parts or ':' in name:
        raise ValueError(f'Unsafe COLMAP image name: {name}')
    return relative


def registered_names(path, cameras=None, image_ids=None):
    """Read the complete binary image table, rejecting truncated/unsafe records."""
    names = []
    size = path.stat().st_size
    with path.open('rb') as f:
        def read(n):
            data = f.read(n)
            if len(data) != n:
                raise ValueError(f'Truncated COLMAP images: {path}')
            return data
        count, = struct.unpack('<Q', read(8))
        if not 0 < count <= 1000000:
            raise ValueError(f'Invalid COLMAP image count: {count}')
        for _ in range(count):
            record = struct.unpack('<i7di', read(64))
            if cameras is not None and record[-1] not in cameras:
                raise ValueError(f'COLMAP image references missing camera: {record[-1]}')
            if not all(math.isfinite(x) for x in record[1:8]) or sum(x * x for x in record[1:5]) < 1e-12:
                raise ValueError('Invalid COLMAP image pose')
            if image_ids is not None:
                if record[0] in image_ids:
                    raise ValueError('Duplicate COLMAP image ID')
                image_ids.add(record[0])
            name = bytearray()
            while len(name) < 4096:
                char = read(1)
                if char == b'\0':
                    break
                name.extend(char)
            else:
                raise ValueError('COLMAP image name too long')
            name = name.decode('utf8')
            safe_image_name(name)
            names.append(name)
            n, = struct.unpack('<Q', read(8))
            if f.tell() + n * 24 > size:
                raise ValueError(f'Truncated COLMAP image observations: {path}')
            f.seek(n * 24, 1)
        if f.tell() != size or len(set(names)) != len(names):
            raise ValueError(f'Invalid/duplicate COLMAP image table: {path}')
    return names


def validate_model(folder):
    # These are the COLMAP IDs and parameter counts written by pinned Spirula.
    parameters = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12, 17: 2}
    def read(f, n):
        data = f.read(n)
        if len(data) != n:
            raise ValueError(f'Truncated COLMAP model: {f.name}')
        return data
    camera_ids = set()
    with (folder / 'cameras.bin').open('rb') as f:
        count, = struct.unpack('<Q', read(f, 8))
        if not 0 < count <= 1000000:
            raise ValueError('Invalid COLMAP camera count')
        for _ in range(count):
            camera, kind, width, height = struct.unpack('<iiQQ', read(f, 24))
            if kind not in parameters or not width or not height or camera in camera_ids:
                raise ValueError('Invalid COLMAP camera record')
            values = struct.unpack('<' + 'd' * parameters[kind], read(f, parameters[kind] * 8))
            if not all(math.isfinite(v) for v in values) or values[0] <= 0:
                raise ValueError('Invalid COLMAP camera calibration')
            camera_ids.add(camera)
        if f.read(1):
            raise ValueError('Trailing COLMAP camera data')
    ids = set()
    names = registered_names(folder / 'images.bin', cameras=camera_ids, image_ids=ids)
    point_ids = set()
    with (folder / 'points3D.bin').open('rb') as f:
        count, = struct.unpack('<Q', read(f, 8))
        if count > (folder / 'points3D.bin').stat().st_size // 51:
            raise ValueError('Invalid COLMAP point count')
        for _ in range(count):
            point, x, y, z, r, g, b, error, tracks = struct.unpack('<Q3d3BdQ', read(f, 51))
            if point in point_ids or not all(math.isfinite(v) for v in (x, y, z, error)):
                raise ValueError('Invalid COLMAP point record')
            point_ids.add(point)
            if tracks > (folder / 'points3D.bin').stat().st_size // 8:
                raise ValueError('Invalid COLMAP point track count')
            for _ in range(tracks):
                image, observation = struct.unpack('<ii', read(f, 8))
                if image not in ids or observation < 0:
                    raise ValueError('Invalid COLMAP point track')
        if f.read(1):
            raise ValueError('Trailing COLMAP point data')
    return names


def validate_maps(root, names, depth):
    import cv2
    import numpy as np
    counts = {'normals': 0, 'depths': 0}
    for name in names:
        relative = Path(safe_image_name(name)).with_suffix('.png')
        normal_shape = None
        for kind in ('normals', 'depths') if depth else ('normals',):
            path = root / kind / relative
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
            if img is None or img.size == 0 or not np.isfinite(img).all():
                raise ValueError(f'Missing/unreadable geometry map: {path}')
            if kind == 'normals':
                if img.ndim != 3 or img.shape[2] not in (3, 4):
                    raise ValueError(f'Invalid normal map: {path}')
                normal_shape = img.shape[:2]
            elif img.ndim != 2 or img.dtype != np.uint16 or not np.any(img > 0) or img.shape != normal_shape:
                raise ValueError(f'Invalid depth map: {path}')
            counts[kind] += 1
    return counts


class Preprocessor:
    def __init__(self, config, work, binary, device, run, drive_root='/content/drive/MyDrive'):
        self.config = config
        self.s = config['settings']
        self.work = Path(work)
        self.dataset = self.work / 'dataset'
        self.reports = self.work / 'reports'
        self.reports.mkdir(parents=True, exist_ok=True)
        self.binary, self.device, self.run = str(binary), device, run
        self.drive_root = Path(drive_root)
        self.report = {}
        path = self.reports / 'sfm_report.json'
        if path.exists():
            self.report = json.loads(path.read_text(encoding='utf8'))
        self.sources = []

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
        temp.replace(path)

    def stage(self, name):
        status_path = self.reports / 'stages.json'
        states = json.loads(status_path.read_text()) if status_path.exists() else {}
        states[name] = {'status': 'running'}
        self.write_json(status_path, states)
        try:
            result = getattr(self, name)()
        except BaseException as exc:
            states[name] = {'status': 'failed', 'error': str(exc)}
            self.write_json(status_path, states)
            raise
        states[name] = {'status': 'complete'}
        self.write_json(status_path, states)
        return result

    def command(self, name, args, accepted=(0,)):
        args = [self.binary, *map(str, args), '--device', self.device, '--lang', 'en']
        return self.run(name, args, cwd=self.work, timeout=math.inf, accepted_exit_codes=accepted)

    def sam_model(self):
        key = self.s['mask_model']
        size, expected = SAM_MODELS[key]
        target = self.work / (key + '.ggml')
        def valid():
            if not target.is_file() or target.stat().st_size != size:
                return False
            with target.open('rb') as stream:
                return sha256_stream(stream) == expected
        if not valid():
            partial = target.with_suffix('.download')
            url = f'https://huggingface.co/PABannier/sam3.cpp/resolve/main/{key}.ggml'
            print('Maskeleme modeli indiriliyor:', key, flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as out:
                shutil.copyfileobj(response, out, 1024 * 1024)
            partial.replace(target)
            if not valid():
                raise ValueError('SAM model size/SHA-256 verification failed')
        return target

    def mask_options(self):
        return ['--model', self.sam_model(), '--text', self.s['mask_prompt'],
                '--neg-text', self.s['mask_negative_prompt'], '--threshold', self.s['mask_threshold'],
                '--dilate-ratio', self.s['mask_dilate'], '--max-size', self.s['mask_max_size']]

    def prepare(self):
        settings_path = self.reports / 'preprocess_settings.json'
        if settings_path.exists() and json.loads(settings_path.read_text(encoding='utf8')) != self.config:
            raise ValueError('Ayarlar bu çalışma klasörüyle uyuşmuyor. Yeni ayarlarla yeni bir notebook çalışması başlat.')
        self.write_json(settings_path, self.config)
        image_root = self.dataset / 'images'
        image_root.mkdir(parents=True, exist_ok=True)
        self.sources = []
        extraction = []
        for index, entry in enumerate(self.config['inputs']):
            prefix = '' if len(self.config['inputs']) == 1 else f'input_{index + 1:02d}'
            source = self.drive_root / entry['path']
            dest = image_root / prefix
            dest.mkdir(parents=True, exist_ok=True)
            masks = self.dataset / 'masks' / prefix
            if entry['kind'] == 'video':
                local = self.work / 'sources' / (str(index) + source.suffix)
                local.parent.mkdir(exist_ok=True)
                shutil.copy2(source, local)
                probe = self.run(f'video_info_{index}', ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                    '-show_entries', 'stream=avg_frame_rate', '-of', 'json', str(local)], cwd=self.work, timeout=60)
                # run_logged includes its command and exit marker; find the JSON payload.
                payload = '\n'.join(line for line in probe.splitlines() if not line.startswith(('COMMAND:', 'EXIT_CODE:')))
                info = json.JSONDecoder().raw_decode(payload[payload.index('{'):])[0]
                rate = float(Fraction(info['streams'][0]['avg_frame_rate']))
                if not math.isfinite(rate) or rate <= 0:
                    raise ValueError('Video kare hızı okunamadı')
                skip = max(1, math.floor(rate / entry['fps'] + 0.5))
                args = ['sam', 'extract', local, '--out', dest, '--skip', skip,
                        '--keep', self.s['sharpness_window'], '--quality', 101]
                if self.s['adaptive_fps']:
                    args += ['--adaptive']
                if self.s['mask_objects']:
                    args += [*self.mask_options(), '--mask-mode', 'image', '--mask-out', masks]
                self.command(f'frames_{index}', args)
                self.sources.append({'prefix': prefix, 'telemetry': str(local), 'fps': rate})
                extraction.append({'input': entry['path'], 'source_fps': rate, 'skip': skip,
                                   'target_fps': entry['fps'], 'nominal_fps': rate / skip,
                                   'adaptive': self.s['adaptive_fps']})
            else:
                if any(p.suffix.lower() == '.exr' for p in source.rglob('*') if p.is_file()):
                    raise ValueError('Bu notebook fotoğraf girdisinde EXR desteklemiyor; JPG, PNG veya BMP kullan.')
                files = image_files(source)
                if not files:
                    raise ValueError(f'Fotoğraf klasöründe desteklenen görüntü yok: {source}')
                for file in files:
                    rel = file.relative_to(source)
                    if any(part.lower() in {'masks', 'normals', 'depths', 'outputs', 'sparse'} for part in rel.parts[:-1]):
                        continue
                    target = dest / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, target)
                if self.s['mask_objects']:
                    # This CLI's track defaults to temporal memory, unlike the GUI's
                    # photo mode. Isolate each photo to prevent unrelated views from
                    # contaminating each other's masks. It reloads the model each time.
                    model_options = self.mask_options()
                    for file_index, file in enumerate(image_files(dest)):
                        temp = self.work / 'photo_masks' / str(index) / str(file_index)
                        single = temp / 'input'
                        single.mkdir(parents=True, exist_ok=True)
                        link = single / file.name
                        if not link.exists():
                            link.symlink_to(file.resolve())
                        self.command(f'photo_masks_{index}_{file_index}', ['sam', 'track', '--frames', single,
                            '--out', temp, '--max-frames', 1, *model_options])
                        target = masks / file.relative_to(dest).with_suffix('.png')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(temp / 'frame_00000.png', target)
            if not image_files(dest):
                raise ValueError(f'Girdi görüntü üretmedi: {source}')
        # Geometry/mask outputs use stems, so reject a.jpg + a.png collisions.
        names = [p.relative_to(image_root).with_suffix('').as_posix() for p in image_files(image_root)]
        if len(names) != len(set(names)):
            raise ValueError('Aynı klasörde aynı kök ada sahip görüntüler var; adları ayır.')
        if self.s['static_mask']:
            args = ['sam', 'mask', image_root, '--out', self.dataset / 'masks']
            if self.s['static_shapes']:
                args += ['--shape', self.s['static_shapes']]
            self.command('static_masks', args)
        if self.s['mask_objects'] or self.s['static_mask']:
            self.validate_masks()
        self.write_json(self.reports / 'extraction.json', extraction)
        self.write_json(self.work / 'capture_manifest.json', {'captures': self.sources})
        print('SfM girişi:', len(names), 'görüntü')

    def validate_masks(self):
        from PIL import Image
        for image in image_files(self.dataset / 'images'):
            rel = image.relative_to(self.dataset / 'images')
            mask = self.dataset / 'masks' / rel.with_suffix('.png')
            with Image.open(image) as original, Image.open(mask) as generated:
                generated.load()
                if generated.size != original.size:
                    raise ValueError(f'Mask/image size mismatch: {mask}')

    def sfm_args(self):
        s = self.s
        args = ['sfm', 'auto', self.dataset / 'images', '--output', self.dataset,
                '--quality', s['quality'], '--data-type', s['capture_kind'], '--pairs', s['pairs'],
                '--camera-model', s['camera_model'], '--camera-mode', s['camera_mode'], '--overlap', s['overlap'],
                '--sensor-gauge', s['sensor_gauge'], '--metric-gps', s['metric_gps'], '--exif-attitude', s['exif_attitude']]
        for i, entry in enumerate(self.config['inputs']):
            if entry['kind'] == 'video' or s['capture_kind'] == 'video':
                prefix = '.' if len(self.config['inputs']) == 1 else f'input_{i + 1:02d}'
                # Each camera subfolder is a separate sequence; no false transitions
                # between clips or lens streams are introduced.
                folders = sorted({p.parent for p in image_files(self.dataset / 'images' / prefix)})
                for folder in folders:
                    args += ['--sequence', folder.relative_to(self.dataset / 'images').as_posix()]
        capture = self.work / 'capture_manifest.json'
        if capture.exists() and s['sensor_gauge'] != 'none':
            args += ['--manifest', capture]
        for key, flag in [('focal', '--focal'), ('distortion', '--distortion'),
                          ('max_features', '--max-features'), ('max_image_size', '--max-image-size')]:
            if s[key]:
                args += [flag, s[key]]
        if not s['loop_closure']:
            args += ['--no-loop-closure']
        args += ['--prefilter-sequential' if s['prefilter_sequential'] else '--no-prefilter-sequential']
        if s['distortion_refine'] in ('final', 'fixed'):
            args += ['--no-refine-extra-params']
        if s['distortion_refine'] == 'fixed':
            args += ['--no-final-extra-params']
        for key, flag in [('per_image_intrinsics', '--final-per-image-intrinsics'), ('final_free_rig', '--final-free-rig')]:
            if s[key]:
                args += [flag]
        if not (s['mask_objects'] or s['static_mask']):
            args += ['--no-masks']
        return args

    def reconstruct(self):
        self.report = {}
        self.write_json(self.reports / 'sfm_report.json', self.report)
        log = self.command('sfm', self.sfm_args(), accepted=(0, 3))
        components = []
        all_names = set()
        input_names = {p.relative_to(self.dataset / 'images').as_posix() for p in image_files(self.dataset / 'images')}
        for folder in sorted((self.dataset / 'sparse').glob('*')):
            if not folder.is_dir() or not folder.name.isdigit():
                continue
            for file in ('cameras.bin', 'images.bin', 'points3D.bin'):
                if not (folder / file).is_file() or (folder / file).stat().st_size < 8:
                    raise ValueError(f'Incomplete COLMAP model: {folder / file}')
            names = validate_model(folder)
            if set(names) - input_names:
                raise ValueError('COLMAP model references missing input images')
            all_names.update(names)
            components.append({'model': folder.name, 'images': len(names)})
        if not components:
            raise ValueError('SfM kullanılabilir bir model yazmadı. Günlük ve kareler yerel diskte korunuyor.')
        components.sort(key=lambda c: (-c['images'], int(c['model'])))
        total = len(input_names)
        selected = components[0]
        error = re.search(r'Reprojection error:\s*mean\s+([\d.]+)\s*px', log)
        self.report = {'input_images': total, 'registered_union': len(all_names),
            'coverage': len(all_names) / total, 'components': components,
            'selected_model': selected['model'], 'selected_images': selected['images'],
            'selected_coverage': selected['images'] / total,
            'dominant_share': selected['images'] / len(all_names),
            'mean_reprojection_px': float(error[1]) if error else None,
            'partial': 'EXIT_CODE: 3' in log or selected['images'] < total,
            'warnings': []}
        if self.report['selected_coverage'] < 0.8 or len(components) > 1:
            self.report['warnings'].append('Çekim eksik veya birden çok bileşen var. Eğitimden önce bileşenleri incele.')
        self.write_json(self.reports / 'sfm_report.json', self.report)
        return self.report

    def geometry(self):
        if not self.s['generate_geometry']:
            print('Geometri haritaları kapalı; atlandı.')
            return
        if not self.report:
            raise ValueError('Önce SfM ve kalite raporu hücresini çalıştır.')
        self.report.pop('geometry', None)
        self.write_json(self.reports / 'sfm_report.json', self.report)
        selected = self.dataset / 'sparse' / self.report['selected_model']
        names = registered_names(selected / 'images.bin')
        root = self.work / 'geometry_dataset'
        (root / 'sparse').mkdir(parents=True, exist_ok=True)
        # Only symlink within the notebook's private working directory.
        for link, target in [(root / 'images', self.dataset / 'images'), (root / 'sparse/0', selected)]:
            if not link.exists():
                link.symlink_to(target.resolve(), target_is_directory=True)
            elif link.resolve() != target.resolve():
                raise ValueError(f'Selected geometry dataset changed: {link}')
        args = ['geometry', root, '--model', self.s['geometry_model'], '--max-size', self.s['geometry_max_size'],
                '--num-tokens', self.s['geometry_tokens'], '--normal-format', 'png', '--overwrite']
        if self.s['generate_depth']:
            args += ['--depth', '--depth-units', 'relative']
        log = self.command('geometry', args, accepted=(0, -11))
        counts = validate_maps(root, names, self.s['generate_depth'])
        if 'EXIT_CODE: -11' in log:
            if not re.search(r'done:\s*\d+ written', log):
                raise ValueError('Geometry crashed without completion marker')
            self.report['warnings'].append('Geometri işlemi çıkışta çöktü (-11). Tüm seçili görüntülerin haritaları okunup doğrulandı; neden henüz bilinmiyor.')
        for kind in ('normals', 'depths') if self.s['generate_depth'] else ('normals',):
            shutil.copytree(root / kind, self.dataset / kind, dirs_exist_ok=True)
        self.report['geometry'] = {'model': self.s['geometry_model'], 'component': self.report['selected_model'], **counts}
        self.write_json(self.reports / 'sfm_report.json', self.report)

    def package(self):
        if not self.report:
            raise ValueError('Önce SfM raporunu üret.')
        if self.s['generate_geometry'] and 'geometry' not in self.report:
            raise ValueError('Geometri hücresi tamamlanmadı; eksik dataset paketlenmedi.')
        output = self.work / 'outputs/dataset.zip'
        output.parent.mkdir(exist_ok=True)
        self.write_json(self.reports / 'preprocess_settings.json', self.config)
        partial = output.with_suffix('.partial')
        folders = ['images', 'sparse', 'normals', 'depths', 'masks']
        if self.s['keep_intermediates']:
            folders.append('features')
        with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for folder in folders:
                for file in sorted((self.dataset / folder).rglob('*')):
                    if file.is_file():
                        archive.write(file, file.relative_to(self.dataset).as_posix())
            if self.s['keep_intermediates'] and (self.dataset / 'matches.bin').is_file():
                archive.write(self.dataset / 'matches.bin', 'matches.bin')
            for file in self.reports.glob('*.json'):
                archive.write(file, file.name)
        partial.replace(output)
        with output.open('rb') as stream:
            checksum = sha256_stream(stream)
        self.write_json(self.reports / 'archive.json', {'bytes': output.stat().st_size, 'sha256': checksum})
        print('Dataset ZIP hazır:', output)
        return output


def verify_cli(prep):
    # Fail before copying a capture if the pinned binary changes its contract.
    for name, args, flags in [
        ('sfm_help', ['sfm', 'auto', '--help'], ['--camera-model', '--camera-mode', '--quality', '--pairs', '--sequence', '--manifest']),
        ('extract_help', ['sam', 'extract', '--help'], ['--skip', '--keep', '--adaptive', '--mask-mode']),
        ('geometry_help', ['geometry', '--help'], ['--model', '--num-tokens', '--normal-format', '--depth']),
        ('sam_help', ['sam', '--help'], ['--shape', '--frames', '--text']),
    ]:
        text = prep.command(name, args)
        missing = [flag for flag in flags if flag not in text]
        if missing:
            raise ValueError(f'Spirula CLI değişti: {name}: {missing}')

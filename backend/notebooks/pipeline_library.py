"""Versioned notebook recipes; generation never installs software or starts training."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import re
import zipfile
from pathlib import Path
from typing import Literal

import nbformat
from pydantic import Field, field_validator, model_validator

from .builder import build_static_notebook
from .drive_paths import normalize_input_folder
from .models import StrictModel, StaticNotebookRunSpec
from .source import NotebookSource

ASSETS = Path(__file__).parent / 'templates'
ROOT = Path(__file__).resolve().parents[2]
PRESETS = {
    'hybrid': {
        'baseline': dict(iterations=30000, max_gaussians=1000000, min_vram=20, recipe='baseline_1m'),
        'quality': dict(iterations=50000, max_gaussians=3000000, min_vram=38, recipe='quality_3m'),
        'ultra': dict(iterations=60000, max_gaussians=5000000, min_vram=70, recipe='stress_5m'),
    },
    'spirula': {
        'baseline': dict(iterations=30000, max_gaussians=1000000, min_vram=14, recipe='medium'),
        'quality': dict(iterations=60000, max_gaussians=6000000, min_vram=22, recipe='high'),
        'ultra': dict(iterations=80000, max_gaussians=10000000, min_vram=38, recipe='ultra'),
    },
}


class PipelineNotebookSpec(StrictModel):
    pipeline: Literal['native', 'hybrid', 'spirula']
    input_mode: Literal['folder', 'video', 'dataset_zip', 'dataset_folder']
    input_path: str
    preset: Literal['baseline', 'quality', 'ultra'] = 'baseline'
    iterations: int | None = Field(default=None, ge=1000, le=120000)
    max_gaussians: int | None = Field(default=None, ge=50000, le=20000000)
    fps: int = Field(default=4, ge=1, le=30)
    model_name: str = Field(default='0', pattern=r'^\d+$')
    allow_partial: bool = False
    generate_depth: bool = True
    geometry_model: Literal['moge2-vits', 'moge2-vitb', 'moge2-vitl'] = 'moge2-vitb'
    sfm_quality: Literal['high', 'extreme'] = 'high'

    @field_validator('input_path')
    @classmethod
    def input_is_drive_relative(cls, value: str) -> str:
        return normalize_input_folder(value)

    @model_validator(mode='after')
    def check_mode(self):
        allowed = {'native': {'folder'}, 'hybrid': {'dataset_zip', 'dataset_folder'},
                   'spirula': {'video', 'dataset_zip', 'dataset_folder'}}
        if self.input_mode not in allowed[self.pipeline]:
            raise ValueError(f'{self.pipeline} supports: {sorted(allowed[self.pipeline])}')
        if self.pipeline == 'native' and self.max_gaussians and self.max_gaussians > 6000000:
            raise ValueError('Native generator supports at most 6,000,000 Gaussians')
        return self


def replace_assignments(source: str, values: dict) -> str:
    """Replace only named top-level assignments; values are Python literals, not code."""
    lines = source.splitlines(keepends=True)
    edits = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in values:
                edits.append((node.lineno - 1, node.end_lineno, f'{name} = {values[name]!r}\n'))
    found = {ast.parse(text).body[0].targets[0].id for _, _, text in edits}
    if found != set(values):
        raise ValueError(f'Template configuration drift: {set(values) - found}')
    for start, end, text in reversed(edits):
        lines[start:end] = [text]
    return ''.join(lines)


def _training_bundle() -> bytes:
    paths = ['backend/model/gaussian_model.py', 'backend/model/deformation.py',
             'backend/model/renderer.py', 'backend/model/density_control.py',
             'backend/model/trainer.py', 'backend/preprocess/parse_colmap.py', 'backend/export/to_splat.py']
    files = {p: (ROOT / p).read_bytes() for p in paths}
    for package in ['backend', 'backend/model', 'backend/preprocess', 'backend/export']:
        files[f'{package}/__init__.py'] = b''
    files['spirula_import.py'] = (ASSETS / 'spirula_import.py').read_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buffer.getvalue()


def build_pipeline_notebook(spec: PipelineNotebookSpec, *, source: NotebookSource):
    if spec.pipeline == 'native':
        native = StaticNotebookRunSpec(input_folder=spec.input_path,
            frame_selection={'mode': 'fixed_fps', 'fixed_fps': spec.fps},
            quality={'profile': {'baseline': 'balanced_l4', 'quality': 'high', 'ultra': 'ultra'}[spec.preset],
                     'n_iters': spec.iterations, 'max_gaussians': spec.max_gaussians})
        notebook = build_static_notebook(native, source=source)
    else:
        notebook = nbformat.read(ASSETS / f'{spec.pipeline}.ipynb', as_version=4)
        cfg = PRESETS[spec.pipeline][spec.preset]
        steps = spec.iterations or cfg['iterations']
        cap = spec.max_gaussians or cfg['max_gaussians']
        path = '/content/drive/MyDrive/' + spec.input_path
        # Custom caps retain at least the next known memory tier; never silently lower quality.
        tiers = list(PRESETS[spec.pipeline].values())
        cap_tier = next((p for p in tiers if cap <= p['max_gaussians']), tiers[-1])
        vram = max(cfg['min_vram'], cap_tier['min_vram'])
        notebook.cells[0].source = (
            f'# {"Spirula dataset → bizim trainer" if spec.pipeline == "hybrid" else "Spirula preprocessing + training"}\n\n'
            f'Preset: **{spec.preset}** · {steps:,} adım · {cap:,} Gaussian bütçesi.\n\n'
            'GPU seç, ayar hücresini kontrol et ve hücreleri sırayla çalıştır. Her koşu ayrı sonuç klasörüne yazılır. '
            'Daha büyük bütçe daha iyi kalite garantisi değildir. Bellek eşiği bir ön kontroldür, OOM garantisi değildir.\n\n'
            + ('Hazır dataset içinde images/ ve sparse/ olmalı. Bu yol yeniden SfM yapmaz; tek kamera kalibrasyonu destekler. '
               'normals/depths bu trainer tarafından kullanılmaz. Nominal Gaussian sınırı büyüme adımında aşılabilir; '
               'checkpoint tam optimizer devamı sağlamaz. Sonuçlar GaussianTests/results altında. '
               'İsteğe bağlı kurtarma hücresini yalnızca güvenilir checkpoint ile çalıştır.\n'
               if spec.pipeline == 'hybrid' else
               'Spirula v2026.9.24 resmi Linux ikilisi kullanılır; kaynak derlenmez. '
               'SfM kapsamı düşükse uzun eğitimden önce durur. Sonuçlar GaussianTests/spirula_cloud_tests altında. '
               'Vulkan ve Drive kurtarma adımları dahildir. Dataset girişinde mevcut sparse model kullanılır, geometri haritaları yeniden üretilir.\n'))
        if spec.pipeline == 'hybrid':
            cell = next(c for c in notebook.cells if 'DATASET_PATH =' in c.source)
            cell.source = replace_assignments(cell.source, {'DATASET_PATH': path, 'SCENE_NAME': 'scene', 'MODEL_NAME': spec.model_name, 'PRESET': cfg['recipe']})
            cell.source += f'\nITERATIONS_OVERRIDE = {steps}\nGAUSSIANS_OVERRIDE = {cap}\nMIN_VRAM_OVERRIDE = {vram}\n'
            bundle = _training_bundle()
            cell = next(c for c in notebook.cells if 'BUNDLE_B64 =' in c.source)
            cell.source = replace_assignments(cell.source, {'BUNDLE_B64': base64.b64encode(bundle).decode(), 'BUNDLE_SHA256': hashlib.sha256(bundle).hexdigest()})
        else:
            cell = next(c for c in notebook.cells if 'INPUT_MODE =' in c.source)
            cell.source = replace_assignments(cell.source, {
                'INPUT_MODE': spec.input_mode, 'INPUT_VIDEO': path if spec.input_mode == 'video' else '',
                'INPUT_DATASET_ZIP': path if spec.input_mode == 'dataset_zip' else '',
                'INPUT_DATASET_FOLDER': path if spec.input_mode == 'dataset_folder' else '',
                'TARGET_FPS': spec.fps, 'TRAIN_ITERATIONS': steps, 'TRAIN_CAP_MAX': cap,
                'MIN_TRAIN_VRAM_GIB': vram, 'GEOMETRY_MODEL': spec.geometry_model,
                'ALLOW_PARTIAL_RECONSTRUCTION': spec.allow_partial, 'GENERATE_DEPTH': spec.generate_depth})
            cell.source += f'\nTRAIN_QUALITY = {cfg["recipe"]!r}\nSFM_QUALITY = {spec.sfm_quality!r}\n'
    notebook.metadata['pipeline'] = {'id': spec.pipeline, 'template_version': 1, 'spec': spec.model_dump(),
                                    'generator_commit': source.commit_sha}
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            ast.parse(cell.source)
    nbformat.validate(notebook)
    return notebook


def pipeline_filename(spec: PipelineNotebookSpec) -> str:
    leaf = Path(spec.input_path).stem
    leaf = re.sub(r'[^A-Za-z0-9_-]', '_', leaf)[:60] or 'scene'
    return f'{leaf}_{spec.pipeline}_{spec.preset}.ipynb'

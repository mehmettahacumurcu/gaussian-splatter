"""Versioned notebook recipes; generation never installs software or starts training."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import re
import subprocess
import unicodedata
import zipfile
from pathlib import Path
from typing import Literal

import nbformat
from pydantic import Field, field_validator, model_validator

from .builder import build_static_notebook
from .drive_paths import normalize_input_folder
from .models import StrictModel, StaticNotebookRunSpec
from .source import (
    GENERATOR_ID, GENERATOR_VERSION, REPOSITORY_URL, NotebookSource, NotebookSourceError,
)
from .templates.text_to_splat_settings import IMAGE_MODELS, MAX_DETAIL, SETTING_FIELDS, model_provenance, validate_settings

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
    'text_to_splat': {
        'baseline': dict(min_vram=20, recipe='sdxl_trellis', image_steps=25, sparse_steps=12, slat_steps=12, resolution=1024),
        'quality': dict(min_vram=20, recipe='sdxl_trellis', image_steps=40, sparse_steps=20, slat_steps=20, resolution=1024),
        'ultra': dict(min_vram=38, recipe='sdxl_trellis', image_steps=50, sparse_steps=25, slat_steps=25, resolution=1024),
        'max_detail': dict(min_vram=60, recipe='trellis2_max_detail', image_steps=25, resolution=1024, **MAX_DETAIL),
    },
}


def text_prompt_slug(prompt: str) -> str:
    """A stable ASCII basename, including a digest for non-ASCII/colliding prompts."""
    ascii_prompt = unicodedata.normalize('NFKD', prompt).encode('ascii', 'ignore').decode()
    slug = re.sub(r'[^a-z0-9]+', '-', ascii_prompt.lower()).strip('-')[:48].rstrip('-') or 'object'
    return f'{slug}-{hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]}'


def resolve_embedded_notebook_source() -> NotebookSource:
    """Record local provenance; this recipe embeds its runtime and never clones HEAD."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise NotebookSourceError('Gömülü notebook için yerel Git HEAD okunamadı') from exc
    sha = result.stdout.strip()
    if not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise NotebookSourceError('Gömülü notebook kaynağı 40 karakterli Git SHA gerektirir')
    return NotebookSource(REPOSITORY_URL, sha, GENERATOR_ID, GENERATOR_VERSION)


class PipelineNotebookSpec(StrictModel):
    pipeline: Literal['native', 'hybrid', 'spirula', 'text_to_splat']
    input_mode: Literal['folder', 'video', 'dataset_zip', 'dataset_folder', 'text'] | None = None
    input_path: str = ''
    preset: Literal['baseline', 'quality', 'ultra', 'max_detail'] = 'baseline'
    iterations: int | None = Field(default=None, ge=1000, le=120000)
    max_gaussians: int | None = Field(default=None, ge=50000, le=20000000)
    fps: int = Field(default=4, ge=1, le=30)
    model_name: str = Field(default='0', pattern=r'^\d+$')
    allow_partial: bool = False
    generate_depth: bool = True
    geometry_model: Literal['moge2-vits', 'moge2-vitb', 'moge2-vitl'] = 'moge2-vitb'
    sfm_quality: Literal['high', 'extreme'] = 'high'
    prompt: str = Field(default='', max_length=2000)
    negative_prompt: str = Field(default='', max_length=2000)
    style: str = Field(default='', max_length=500)
    seed: int = Field(default=42, ge=0, le=2147483647, strict=True)
    output_dir: str | None = None
    target_splat_count: int | None = Field(default=None, ge=1000, le=2000000, strict=True)
    gpu_preset: Literal['l4', 'a100', 'h100', 'rtx_pro_6000'] = 'l4'
    image_model: Literal['sdxl', 'flux1_dev', 'flux1_schnell', 'flux2_klein_4b', 'qwen_image'] | None = None
    background_model: Literal['u2net', 'birefnet'] | None = None
    reconstruction_model: Literal['trellis', 'trellis2', 'hunyuan3d'] | None = None
    image_steps: int | None = Field(default=None, ge=1, le=100, strict=True)
    image_guidance: float | None = Field(default=None, ge=0, le=20, allow_inf_nan=False, strict=True)
    image_resolution: Literal[512, 768, 1024] | None = None
    trellis_seed: int | None = Field(default=None, ge=0, le=2147483647, strict=True)
    sparse_steps: int | None = Field(default=None, ge=1, le=100, strict=True)
    sparse_cfg: float | None = Field(default=None, ge=0, le=20, allow_inf_nan=False, strict=True)
    slat_steps: int | None = Field(default=None, ge=1, le=100, strict=True)
    slat_cfg: float | None = Field(default=None, ge=0, le=20, allow_inf_nan=False, strict=True)
    mesh_views: int | None = Field(default=None, ge=12, le=200, strict=True)
    mesh_fit_iterations: int | None = Field(default=None, ge=100, le=30000, strict=True)
    mesh_splat_cap: int | None = Field(default=None, ge=1000, le=3000000, strict=True)

    trellis2_pipeline_type: Literal['512', '1024_cascade', '1536_cascade'] | None = None
    mesh_render_resolution: int | None = Field(default=None, ge=512, le=2048, strict=True)
    mesh_sh_degree: int | None = Field(default=None, ge=0, le=3, strict=True)
    mesh_texture_size: Literal[2048, 4096] | None = None

    @field_validator('prompt', 'negative_prompt', 'style')
    @classmethod
    def text_is_literal(cls, value: str) -> str:
        if '\x00' in value:
            raise ValueError('Metin alanları NUL karakteri içeremez')
        return value.strip()

    @field_validator('output_dir')
    @classmethod
    def output_is_drive_relative(cls, value: str | None) -> str | None:
        return normalize_input_folder(value, allow_result_folder=True) if value and value.strip() else None

    @model_validator(mode='after')
    def check_mode(self):
        allowed = {'native': {'folder'}, 'hybrid': {'dataset_zip', 'dataset_folder'},
                   'spirula': {'video', 'dataset_zip', 'dataset_folder'}, 'text_to_splat': {'text'}}
        if self.pipeline == 'text_to_splat' and self.input_mode is None:
            self.input_mode = 'text'
        if self.input_mode not in allowed[self.pipeline]:
            raise ValueError(f'{self.pipeline} supports: {sorted(allowed[self.pipeline])}')
        if self.pipeline == 'text_to_splat':
            if not self.prompt:
                raise ValueError('Nesne açıklaması 1–2000 karakter olmalı')
            if self.input_path:
                raise ValueError('Metin → splat tarifi dosya girişi kullanmaz; input_path boş olmalı')
            if self.iterations is not None or self.max_gaussians is not None:
                raise ValueError('Metin → splat için eğitim ayarları yerine preset ve target_splat_count kullanın')
            self.output_dir = self.output_dir or f'GaussianTests/text_to_splat/{text_prompt_slug(self.prompt)}'
            resolved = validate_settings(self.model_dump())
            for field in SETTING_FIELDS:
                setattr(self, field, resolved[field])
        else:
            if self.preset == 'max_detail':
                raise ValueError('Max detay yalnızca text_to_splat tarifi için kullanılabilir')
            if any(getattr(self, field) != type(self).model_fields[field].default
                   for field in self.model_fields_set & SETTING_FIELDS):
                raise ValueError('Model ayarları yalnızca text_to_splat tarifi için kullanılabilir')
            self.input_path = normalize_input_folder(self.input_path)
            if self.prompt or self.negative_prompt or self.style or self.output_dir or self.target_splat_count is not None:
                raise ValueError('Metin alanları yalnızca text_to_splat tarifi için kullanılabilir')
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
                # Preserve trusted Colab form annotations, never user-supplied code.
                tail = lines[node.end_lineno - 1].encode('utf-8')[node.end_col_offset:].decode('utf-8')
                annotation = tail[tail.index('#@param'):].rstrip() if '#@param' in tail else ''
                edits.append((node.lineno - 1, node.end_lineno, f'{name} = {values[name]!r}' +
                              (f' {annotation}' if annotation else '') + '\n'))
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
    elif spec.pipeline == 'text_to_splat':
        notebook = nbformat.read(ASSETS / 'text_to_splat.ipynb', as_version=4)
        config = next(c for c in notebook.cells if c.cell_type == 'code' and 'PROMPT =' in c.source)
        config.source = replace_assignments(config.source, {
            'PROMPT': spec.prompt, 'NEGATIVE_PROMPT': spec.negative_prompt, 'STYLE': spec.style,
            'SEED': spec.seed, 'QUALITY_PRESET': spec.preset,
            'OUTPUT_DIR': '/content/drive/MyDrive/' + spec.output_dir,
            'TARGET_SPLAT_COUNT': spec.target_splat_count,
            **{field.upper(): getattr(spec, field) for field in SETTING_FIELDS},
        })
        if not IMAGE_MODELS[spec.image_model]['negative_prompt']:
            config.source = '\n'.join(
                "NEGATIVE_PROMPT = '' # Bu modelde negatif istem devre dışı." if line.startswith('NEGATIVE_PROMPT =') else line
                for line in config.source.splitlines()) + '\n'
        runtime_files = {
            name: (ASSETS / name).read_text(encoding='utf-8')
            for name in ('text_to_splat_bootstrap.py', 'text_to_splat_runtime.py', 'text_to_splat_helpers.py',
                         'text_to_splat_requirements.txt', 'text_to_splat_settings.py',
                         'text_to_splat_image.py', 'text_to_splat_mesh.py')
        }
        bundle = next(c for c in notebook.cells if c.cell_type == 'code' and 'RUNTIME_FILES =' in c.source)
        bundle.source = replace_assignments(bundle.source, {
            'RUNTIME_FILES': runtime_files, 'QUALITY_PRESETS': PRESETS['text_to_splat'],
        })
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
    if spec.pipeline == 'text_to_splat':
        notebook.metadata['pipeline']['template_version'] = 2
        notebook.metadata['pipeline']['models'] = model_provenance(spec.model_dump())
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            ast.parse(cell.source)
    nbformat.validate(notebook)
    return notebook


def pipeline_filename(spec: PipelineNotebookSpec) -> str:
    leaf = text_prompt_slug(spec.prompt) if spec.pipeline == 'text_to_splat' else Path(spec.input_path).stem
    leaf = re.sub(r'[^A-Za-z0-9_-]', '_', leaf)[:60] or 'scene'
    return f'{leaf}_{spec.pipeline}_{spec.preset}.ipynb'

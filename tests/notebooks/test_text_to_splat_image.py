"""Offline checks for model API routing, authentication and private environments."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.notebooks.templates import text_to_splat_bootstrap as bootstrap
from backend.notebooks.templates import text_to_splat_image as image
from backend.notebooks.templates.text_to_splat_runtime import provenance_for
from backend.notebooks.templates.text_to_splat_settings import DINO3_MODEL, IMAGE_MODELS, validate_settings


def configuration(**overrides):
    return validate_settings(dict(prompt='a toy', negative_prompt='', style='', seed=42,
        target_splat_count=None, output_dir='/content/drive/MyDrive/test',
        recipe_sha256='offline-test', **overrides))


@pytest.mark.parametrize('model', list(IMAGE_MODELS))
def test_image_arguments_use_each_models_real_guidance_api(model):
    config = configuration(image_model=model)
    negative = 'blur' if IMAGE_MODELS[model]['negative_prompt'] else ''
    args = image.inference_arguments(config, 'a toy', negative)
    assert args['width'] == args['height'] == config['image_resolution']
    assert args['num_inference_steps'] == config['image_steps']
    if model == 'qwen_image':
        assert args['true_cfg_scale'] == config['image_guidance']
        assert 'guidance_scale' not in args
        assert args['negative_prompt'] == 'blur'
    elif model == 'flux1_dev':
        assert args['true_cfg_scale'] == args['guidance_scale'] == config['image_guidance']
        assert args['negative_prompt'] == 'blur'
        assert 'true_cfg_scale' not in image.inference_arguments(config, 'a toy', '')
    elif model == 'flux1_schnell':
        assert args['guidance_scale'] == 0 and args['max_sequence_length'] == 256
        assert 'negative_prompt' not in args
    elif model == 'flux2_klein_4b':
        assert args['guidance_scale'] == 1 and 'negative_prompt' not in args
    else:
        assert args['negative_prompt'] == 'blur'


def mock_hub(monkeypatch, metadata):
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        hf_hub_url=lambda repo, filename, revision: f'https://huggingface.co/{repo}/resolve/{revision}/{filename}',
        get_hf_file_metadata=metadata))


def test_access_preflight_includes_gated_dinov3_without_downloading(monkeypatch):
    calls = []
    mock_hub(monkeypatch, lambda url, **kwargs: calls.append((url, kwargs)))
    monkeypatch.setenv('HF_TOKEN', 'unit-test-only')
    config = configuration(reconstruction_model='trellis2')
    image.check_model_access(config)
    assert len(calls) == 2
    assert calls[0][0].endswith('/model_index.json')
    assert DINO3_MODEL['id'] in calls[1][0]
    assert DINO3_MODEL['revision'] in calls[1][0]
    assert calls[1][0].endswith('/model.safetensors')
    assert calls[1][1] == {'token': 'unit-test-only'}


def test_missing_dino_token_and_rejected_access_are_clear_and_redacted(monkeypatch):
    mock_hub(monkeypatch, lambda *_args, **_kwargs: None)
    monkeypatch.delenv('HF_TOKEN', raising=False)
    with pytest.raises(RuntimeError, match='DINOv3|dinov3'):
        image.check_model_access(configuration(reconstruction_model='trellis2'))
    monkeypatch.setenv('HF_TOKEN', 'unit-test-only')
    def rejected(*_args, **_kwargs):
        raise PermissionError('request header included unit-test-only')
    mock_hub(monkeypatch, rejected)
    with pytest.raises(RuntimeError, match='erişimi doğrulanamadı') as error:
        image.check_model_access(configuration(image_model='flux1_dev'))
    assert 'unit-test-only' not in str(error.value)


@pytest.mark.parametrize('route,stage,environment,script,command_stage', [
    ('trellis', 'image', 'image-env', 'text_to_splat_image.py', 'image'),
    ('trellis2', 'check-image', 'image-env', 'text_to_splat_image.py', 'check'),
    ('trellis', 'gaussian', 'env', 'text_to_splat_runtime.py', 'gaussian'),
    ('trellis2', 'gaussian', 'mesh-env', 'text_to_splat_mesh.py', 'gaussian'),
    ('trellis2', 'export', 'mesh-env', 'text_to_splat_mesh.py', 'export'),
])
def test_bootstrap_dispatches_private_environment_and_safe_json(
    tmp_path, monkeypatch, route, stage, environment, script, command_stage,
):
    config = configuration(reconstruction_model=route, work_dir=str(tmp_path))
    config['output_dir'] = str(tmp_path / 'output')
    config['prompt'] = "object'; __import__('os').system('must-not-run'); #\n"
    (tmp_path / environment / 'bin').mkdir(parents=True)
    (tmp_path / environment / 'bin/python').touch()
    # Output path policy is tested separately; this test never writes outside tmp_path.
    monkeypatch.setattr(bootstrap, 'validate_config', lambda _config: None)
    monkeypatch.setattr(bootstrap, '_mesh_module', lambda: SimpleNamespace(
        mesh_environment=lambda work: {'PYTHONPATH': str(work / 'TRELLIS.2'), 'SPARSE_CONV_BACKEND': 'flex_gemm'}))
    calls = []
    monkeypatch.setattr(bootstrap, 'run_logged', lambda *args, **kwargs: calls.append((args, kwargs)))
    bootstrap.run_stage(config, stage)
    command = calls[0][0][0]
    assert command[:3] == [tmp_path / environment / 'bin/python', tmp_path / script, command_stage]
    saved = json.loads((tmp_path / 'run-config.json').read_text(encoding='utf8'))
    assert saved['prompt'] == config['prompt']
    assert 'HF_TOKEN' not in saved
    env = calls[0][1]['env']
    if environment == 'mesh-env':
        assert env['PYTHONPATH'].endswith('TRELLIS.2') and env['SPARSE_CONV_BACKEND'] == 'flex_gemm'
    else:
        assert env['SPARSE_BACKEND'] == 'spconv'


def test_host_gates_dino_and_checks_compiler_before_environment_install(monkeypatch):
    config = configuration(reconstruction_model='trellis2')
    monkeypatch.setattr(bootstrap.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(bootstrap.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(bootstrap.Path, 'is_dir', lambda _path: True)
    monkeypatch.setattr(bootstrap.subprocess, 'run', lambda *_args, **_kwargs:
        SimpleNamespace(stdout='NVIDIA H100, 81920, 580.82.07'))
    monkeypatch.setattr(bootstrap.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(bootstrap.Path, 'read_text', lambda *_args, **_kwargs: 'MemAvailable: 100000000 kB\n')
    compiler_calls = []
    monkeypatch.setattr(bootstrap, '_mesh_module', lambda: SimpleNamespace(
        compiler_preflight=lambda: compiler_calls.append(True)))
    monkeypatch.setattr(bootstrap, 'load_hf_token', lambda: None)
    monkeypatch.delenv('HF_TOKEN', raising=False)
    with pytest.raises(RuntimeError, match='dinov3'):
        bootstrap.host_preflight(config)
    assert compiler_calls == [True]


def test_model_provenance_records_the_selected_pipeline_and_no_token(monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'unit-test-only')
    config = configuration(image_model='qwen_image', background_model='birefnet', reconstruction_model='trellis2')
    manifest = provenance_for(config)
    assert manifest == image.provenance_for(config)
    assert manifest['models']['image']['id'] == 'Qwen/Qwen-Image'
    assert manifest['models']['background']['license'] == 'MIT'
    assert manifest['models']['encoder']['id'] == DINO3_MODEL['id']
    assert all(record['id'] and record['revision'] and record['license'] for record in manifest['models'].values())
    assert 'unit-test-only' not in json.dumps(manifest)

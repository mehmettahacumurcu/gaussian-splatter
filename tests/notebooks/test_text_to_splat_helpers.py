import base64
import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend.compose.plyio import read_ply, validate_ply
from backend.notebooks.templates import text_to_splat_bootstrap as bootstrap
from backend.notebooks.templates.text_to_splat_helpers import (
    StageCache, normalize_gaussians, quaternion_matrices, render_turntable, write_ply,
)


def sample():
    return dict(
        xyz=np.array([[1, 2, 3], [5, 6, 7], [3, 4, 4]], dtype=np.float32),
        scales=np.array([[.1, .2, .3], [.3, .1, .2], [.2, .3, .1]], dtype=np.float32),
        rotations=np.array([[1, 0, 0, 0], [math.sqrt(.5), 0, 0, math.sqrt(.5)], [1, 1, 1, 1]], dtype=np.float32),
        opacity=np.array([[.2], [.8], [.5]], dtype=np.float32),
        dc=np.array([[[.5, -.5, 1]], [[1, 0, .3]], [[-.4, .6, 0]]], dtype=np.float32),
    )


def test_activated_values_and_coordinate_covariance_round_trip(tmp_path):
    source = sample()
    cloud, info = normalize_gaussians(**source)
    np.testing.assert_allclose(cloud['means'], [[-.5, -.5, .5], [.5, .5, -.5], [0, -.25, 0]])
    np.testing.assert_allclose(np.exp(cloud['log_scales']), source['scales'] / 4, rtol=1e-6)
    np.testing.assert_allclose(1/(1+np.exp(-cloud['opacities'])), source['opacity'][:, 0], rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(cloud['quats'], axis=1), 1, rtol=1e-6)
    transform = np.array(info['rotation_source_to_output'])
    np.testing.assert_allclose(quaternion_matrices(cloud['quats']),
                               transform @ quaternion_matrices(source['rotations']), atol=1e-7)
    assert info['output_up'] == '+Y' and info['uniform_scale'] == .25
    assert info['source_bbox_center'] == [3, 4, 5]
    assert np.count_nonzero(cloud['sh_rest']) == 0
    path = write_ply(tmp_path / 'target.ply', cloud)
    assert validate_ply(path) == 3
    loaded = read_ply(path)
    assert loaded.sh_degree == 3
    for key in cloud:
        np.testing.assert_array_equal(getattr(loaded, key), cloud[key])


def test_writer_preserves_channel_major_higher_sh_and_little_endian(tmp_path):
    cloud, _ = normalize_gaussians(**sample())
    cloud['sh_rest'] = np.arange(135, dtype=np.float32).reshape(3, 15, 3)
    path = write_ply(tmp_path / 'sh.ply', cloud)
    data = path.read_bytes()
    assert b'format binary_little_endian 1.0' in data
    loaded = read_ply(path)
    np.testing.assert_array_equal(loaded.sh_rest, cloud['sh_rest'])


@pytest.mark.parametrize('field,value', [
    ('xyz', [[0, 0, 0]] * 3), ('xyz', [[float('nan'), 0, 0]] * 3),
    ('scales', [[0, 1, 1]] * 3), ('opacity', [-.1, .2, .3]),
    ('rotations', [[0, 0, 0, 0]] * 3), ('dc', [[1, 2, 3, 4]] * 3),
])
def test_invalid_gaussian_values_rejected(field, value):
    source = sample()
    source[field] = np.asarray(value)
    with pytest.raises(ValueError):
        normalize_gaussians(**source)


def test_alpha_endpoints_remain_finite_and_count_is_upper_cap():
    source = sample()
    source['opacity'] = np.array([0, 1, .5])
    full, _ = normalize_gaussians(**source, target_count=100)
    assert len(full['means']) == 3
    assert np.isfinite(full['opacities']).all()
    a, info = normalize_gaussians(**source, target_count=2, seed=42)
    b, _ = normalize_gaussians(**source, target_count=2, seed=42)
    assert info['input_count'] == 3 and info['output_count'] == 2
    np.testing.assert_array_equal(a['means'], b['means'])
    assert np.max(np.ptp(a['means'], axis=0)) <= 1


def test_preview_is_decodable_animated_gif(tmp_path):
    cloud, _ = normalize_gaussians(**sample())
    target = tmp_path / 'preview.gif'
    result = render_turntable(target, cloud, frames=4, size=64, max_splats=2)
    assert result['preview_count'] == 2 and result['full_count'] == 3
    with Image.open(target) as gif:
        assert gif.size == (64, 64) and gif.n_frames == 4
        pixels = []
        for i in range(gif.n_frames):
            gif.seek(i)
            pixels.append(np.asarray(gif.convert('RGB')).copy())
        assert np.std(pixels[0]) > 1
        assert not np.array_equal(pixels[0], pixels[1])


def test_cache_resume_corruption_and_downstream_invalidation(tmp_path):
    cache = StageCache(tmp_path, {'prompt': 'test'}, {'model': 'pinned'})
    calls = []
    def write(name):
        calls.append(name)
        (tmp_path / name).write_text('ok')
        return {'detail': 'verified'}
    cache.run('image', ['input.png'], lambda: write('input.png'))
    cache.run('gaussian', ['raw.npz'], lambda: write('raw.npz'), ['image'])
    resumed = StageCache(tmp_path, {'prompt': 'test'}, {'model': 'pinned'})
    resumed.run('image', ['input.png'], lambda: pytest.fail('cached stage reran'))
    (tmp_path / 'input.png').write_text('corrupt')
    with pytest.raises(ValueError, match='Önce'):
        resumed.run('gaussian', ['raw.npz'], lambda: None, ['image'])
    resumed.run('image', ['input.png'], lambda: write('input.png'))
    assert not resumed.valid('gaussian')
    assert len(calls) == 3
    with pytest.raises(ValueError, match='farklı'):
        StageCache(tmp_path, {'prompt': 'changed'}, {'model': 'pinned'})


def test_cache_failed_stage_is_retryable_and_not_complete(tmp_path):
    cache = StageCache(tmp_path, {}, {})
    def fail():
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        cache.run('image', ['input.png'], fail)
    assert cache.data['status'] == 'failed' and not cache.valid('image')
    cache.run('image', ['input.png'], lambda: (tmp_path / 'input.png').write_text('ok') and {})
    assert cache.valid('image')
    assert cache.data['stages']['image']['seconds'] >= 0


def test_cache_does_not_overwrite_unowned_directory(tmp_path):
    (tmp_path / 'target.ply').write_text('old')
    with pytest.raises(ValueError, match='manifest'):
        StageCache(tmp_path, {}, {})
    assert (tmp_path / 'target.ply').read_text() == 'old'


@pytest.mark.parametrize('key,value', [
    ('prompt', ' '), ('prompt', 'x' * 2001), ('style', 'x' * 501), ('seed', True),
    ('seed', -1), ('target_splat_count', 10), ('output_dir', '/content/drive/MyDrive/../../x'),
])
def test_notebook_manually_edited_config_is_validated(key, value):
    config = dict(prompt='car', negative_prompt='', style='', seed=42,
                  target_splat_count=None, output_dir='/content/drive/MyDrive/car')
    config[key] = value
    with pytest.raises(ValueError):
        bootstrap.validate_config(config)


def test_frontend_fixture_is_the_actual_helper_output(tmp_path):
    fixture = Path(__file__).parents[2] / 'frontend/src/notebook/__tests__/fixtures/text-target.json'
    saved = json.loads(fixture.read_text(encoding='utf8'))
    cloud, _ = normalize_gaussians(**sample())
    path = write_ply(tmp_path / 'target.ply', cloud)
    assert base64.b64decode(saved['base64']) == path.read_bytes()


@pytest.mark.parametrize('response,expected', [
    (None, 'GPU bulunamadı'),
    ('Tesla T4, 15360, 580.82.07', 'Yetersiz VRAM'),
    ('NVIDIA L4, 23034, 510.00', 'sürücüsü'),
])
def test_gpu_host_checks_fail_before_any_installs(monkeypatch, response, expected):
    config = dict(prompt='car', negative_prompt='', style='', seed=42, quality={'min_vram': 20},
                  target_splat_count=None, output_dir='/content/drive/MyDrive/car')
    monkeypatch.setattr(bootstrap.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(bootstrap.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(bootstrap.Path, 'is_dir', lambda _: True)
    def query(command, **kwargs):
        assert command[0] == 'nvidia-smi'
        if response is None:
            raise FileNotFoundError('no GPU')
        return SimpleNamespace(stdout=response)
    monkeypatch.setattr(bootstrap.subprocess, 'run', query)
    with pytest.raises(RuntimeError, match=expected):
        bootstrap.host_preflight(config)


def test_runtime_wrong_python_fails_before_torch_import(monkeypatch):
    from backend.notebooks.templates import text_to_splat_helpers
    monkeypatch.setitem(sys.modules, 'text_to_splat_helpers', text_to_splat_helpers)
    from backend.notebooks.templates import text_to_splat_runtime
    monkeypatch.setattr(text_to_splat_runtime.sys, 'version_info', (3, 13, 0))
    with pytest.raises(RuntimeError, match='Yanlış Python'):
        text_to_splat_runtime.gpu_preflight(20)


def test_runtime_pins_attention_backends_and_disables_build_fallback(tmp_path):
    env = bootstrap.runtime_environment(tmp_path)
    assert env['ATTN_BACKEND'] == env['SPARSE_ATTN_BACKEND'] == 'xformers'
    assert env['SPARSE_BACKEND'] == 'spconv'
    assert env['SPCONV_ALGO'] == 'native' and env['SPCONV_DISABLE_JIT'] == '1'
    assert str(tmp_path) in env['HF_HOME'] and env['PYTHONNOUSERSITE'] == '1'

"""CPU-only geometry, provenance and recovery checks; no model/CUDA downloads."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.notebooks.templates import text_to_splat_helpers as helpers
from backend.notebooks.templates import text_to_splat_mesh as mesh


@pytest.mark.parametrize('count', [12, 48, 120, 160, 200])
def test_cameras_are_right_handed_centered_and_keep_object_in_front(count):
    size = 512
    views, intrinsic = mesh.camera_matrices(count, size)
    assert views.shape == (count, 4, 4)
    # Corners of the normalized unit object, including both hemispheres.
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    rotation = views[:, :3, :3]
    np.testing.assert_allclose(rotation @ rotation.transpose(0, 2, 1),
                               np.broadcast_to(np.eye(3), (count, 3, 3)), atol=2e-7)
    np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=2e-7)
    np.testing.assert_allclose(views[:, 3], np.tile([0, 0, 0, 1], (count, 1)))
    eye = -(rotation.transpose(0, 2, 1) @ views[:, :3, 3, None]).squeeze(-1)
    np.testing.assert_allclose(np.linalg.norm(eye, axis=1), 2.2, atol=3e-7)
    assert eye[:, 2].min() < -1 and eye[:, 2].max() > 1
    center_camera = views[:, :3, 3]
    center_pixels = center_camera @ intrinsic.T
    center_pixels = center_pixels[:, :2] / center_pixels[:, 2:]
    np.testing.assert_allclose(center_pixels, size / 2, atol=1e-5)
    camera_corners = np.einsum('nij,kj->nki', rotation, corners) + views[:, None, :3, 3]
    assert camera_corners[:, :, 2].min() > 1.3
    # +Z is world up; its projection must move upward in the +Y-down images.
    world_up_camera = np.einsum('nij,j->ni', rotation, [0, 0, .1]) + center_camera
    up_pixels = world_up_camera @ intrinsic.T
    assert np.all(up_pixels[:, 1] / up_pixels[:, 2] < size / 2)


def separated_triangles():
    # Disjoint right triangles with areas .5 and 2: sample probabilities 1:4.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                         [10, 0, 0], [12, 0, 0], [10, 2, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    colors = np.array([[1, 0, 0]] * 3 + [[0, 0, 1]] * 3, dtype=np.float32)
    return vertices, faces, colors


def test_surface_sampling_is_deterministic_area_weighted_and_capped():
    vertices, faces, colors = separated_triangles()
    points, sampled_colors, spacing = mesh.sample_surface(vertices, faces, colors, 10000, 123)
    repeated = mesh.sample_surface(vertices, faces, colors, 10000, 123)
    np.testing.assert_array_equal(points, repeated[0])
    np.testing.assert_array_equal(sampled_colors, repeated[1])
    assert points.shape == sampled_colors.shape == (10000, 3)
    assert points.dtype == sampled_colors.dtype == np.float32
    assert spacing == pytest.approx(np.sqrt(2.5 / 10000))
    large = points[:, 0] > 5
    assert .78 < large.mean() < .82
    assert np.all(points[:, 2] == 0)
    assert np.all(points[~large, :2] >= 0)
    assert np.all(points[~large, :2].sum(1) <= 1.000001)
    local_large = points[large, :2] - [10, 0]
    assert np.all(local_large >= 0) and np.all(local_large.sum(1) <= 2.000001)
    np.testing.assert_allclose(sampled_colors[~large], np.tile([1, 0, 0], ((~large).sum(), 1)), atol=1e-6)
    np.testing.assert_allclose(sampled_colors[large], np.tile([0, 0, 1], (large.sum(), 1)), atol=1e-6)
    tiny = mesh.sample_surface(vertices, faces, colors, 7, 124)
    assert tiny[0].shape == (7, 3)
    assert not np.array_equal(points[:7], tiny[0])


@pytest.mark.parametrize('bad_vertices', [
    [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
    [[0, 0, 0], [float('nan'), 0, 0], [0, 1, 0]],
])
def test_degenerate_or_nonfinite_mesh_cannot_seed_gaussians(bad_vertices):
    with pytest.raises(ValueError, match='yüzey alanı'):
        mesh.sample_surface(np.asarray(bad_vertices), np.array([[0, 1, 2]]), np.ones((3, 3)), 20, 1)


@pytest.mark.parametrize('corrupted', ['image', 'mesh', 'views', 'gaussian'])
def test_mesh_cache_recovery_invalidates_entire_downstream_chain(tmp_path, corrupted):
    order = ['image', 'mesh', 'views', 'gaussian', 'export']
    cache = helpers.StageCache(tmp_path, {'route': 'trellis2'}, {'revision': 'pinned'})
    def write(stage):
        (tmp_path / (stage + '.data')).write_text('valid ' + stage, encoding='utf8')
        return {'verified': stage}
    for index, stage in enumerate(order):
        cache.run(stage, [stage + '.data'], lambda stage=stage: write(stage), order[:index])
    assert cache.data['status'] == 'complete'
    cache = helpers.StageCache(tmp_path, {'route': 'trellis2'}, {'revision': 'pinned'})
    (tmp_path / (corrupted + '.data')).write_text('damaged', encoding='utf8')
    index = order.index(corrupted)
    next_stage = order[index + 1]
    with pytest.raises(ValueError, match='Önce'):
        cache.run(next_stage, [next_stage + '.data'], lambda: pytest.fail('used corrupt dependency'), order[:index+1])
    cache.run(corrupted, [corrupted + '.data'], lambda: write(corrupted), order[:index])
    assert all(cache.valid(stage) for stage in order[:index+1])
    assert all(not cache.valid(stage) for stage in order[index+1:])
    assert all(stage not in cache.data['stages'] for stage in order[index+1:])
    assert cache.data['status'] == 'partial'


ORIGINAL_BRIA = "pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])"
PATCHED_BRIA = 'pipeline.rembg_model = None  # External, selected RGBA mask; never load BRIA.'


def pipeline_source(tmp_path, statements):
    path = tmp_path / 'trellis2/pipelines/trellis2_image_to_3d.py'
    path.parent.mkdir(parents=True)
    path.write_text('def build(pipeline, args, rembg):\n' + ''.join('    ' + line + '\n' for line in statements), encoding='utf8')
    return path


def test_remove_bria_patch_is_idempotent_and_executable(tmp_path):
    path = pipeline_source(tmp_path, [ORIGINAL_BRIA, 'return pipeline'])
    mesh.patch_pipeline_sources(tmp_path)
    first = path.read_text(encoding='utf8')
    assert ORIGINAL_BRIA not in first and first.count(PATCHED_BRIA) == 1
    mesh.patch_pipeline_sources(tmp_path)
    assert path.read_text(encoding='utf8') == first
    namespace = {}
    exec(compile(first, str(path), 'exec'), namespace)
    result = namespace['build'](SimpleNamespace(), {}, None)
    assert result.rembg_model is None


@pytest.mark.parametrize('statements', [
    ['pipeline.rembg_model = changed_api()'], [ORIGINAL_BRIA, ORIGINAL_BRIA],
    [PATCHED_BRIA, ORIGINAL_BRIA], [PATCHED_BRIA, PATCHED_BRIA],
])
def test_remove_bria_patch_rejects_missing_duplicate_or_mixed_source(tmp_path, statements):
    path = pipeline_source(tmp_path, statements)
    original = path.read_text(encoding='utf8')
    with pytest.raises(RuntimeError, match='kaynak sürümü'):
        mesh.patch_pipeline_sources(tmp_path)
    assert path.read_text(encoding='utf8') == original


@pytest.mark.parametrize('release,accepted', [('12.8', True), ('12.9', True), ('12.7', False), ('13.0', False)])
def test_compiler_matches_cu128_major_without_installing(monkeypatch, release, accepted):
    monkeypatch.setattr(mesh.shutil, 'which', lambda program: '/mock/bin/' + program)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f'Cuda compilation tools, release {release}, V{release}.0')
    monkeypatch.setattr(mesh.subprocess, 'run', run)
    if accepted:
        mesh.compiler_preflight()
    else:
        with pytest.raises(RuntimeError, match='CUDA toolkit'):
            mesh.compiler_preflight()
    assert calls == [['/mock/bin/nvcc', '--version']]


def test_compiler_missing_fails_without_running_commands(monkeypatch):
    monkeypatch.setattr(mesh.shutil, 'which', lambda _program: None)
    monkeypatch.setattr(mesh.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('missing compiler was run'))
    with pytest.raises(RuntimeError, match='nvcc'):
        mesh.compiler_preflight()


def fake_model_hub(tmp_path, monkeypatch, decoder=None):
    upstream = {'args': {'models': {
        'sparse_structure_decoder': decoder or mesh.RECONSTRUCTION_MODELS['trellis']['id'] + '/ckpts/ss_decoder',
        'sparse_structure_flow_model': 'ckpts/ss_flow',
        'shape_slat_flow_model_512': 'ckpts/shape_512',
        'shape_slat_flow_model_1024': 'ckpts/shape_1024',
        'tex_slat_flow_model_1024': 'ckpts/tex_1024',
    }, 'image_cond_model': {'args': {'model_name': 'unpinned-dino'}},
       'rembg_model': {'name': 'BRIA', 'args': {'model_name': 'briaai/RMBG-2.0'}}}}
    metadata = tmp_path / 'upstream-pipeline.json'
    metadata.write_text(json.dumps(upstream), encoding='utf8')
    downloads, snapshots, links = [], [], []
    def download(repo, filename, *, revision):
        downloads.append((repo, filename, revision))
        if filename == 'pipeline.json':
            return str(metadata)
        # Tiny local fixtures; never call an actual HF function or fetch weights.
        fixture = tmp_path / f'fixture-{len(downloads)}.data'
        fixture.write_bytes(b'offline fixture')
        return str(fixture)
    def snapshot(repo, *, revision, allow_patterns):
        snapshots.append((repo, revision, allow_patterns))
        return str(tmp_path / 'dino-snapshot')
    def link(path, target, **_kwargs):
        # Native Windows symlinks may require elevated rights; check the link
        # contract and emulate its content without changing any outside paths.
        assert path.is_relative_to(tmp_path) and Path(target).is_relative_to(tmp_path)
        links.append((path, Path(target)))
        path.write_bytes(Path(target).read_bytes())
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download, snapshot_download=snapshot))
    monkeypatch.setitem(sys.modules, 'text_to_splat_helpers', helpers)
    monkeypatch.setattr(Path, 'symlink_to', link)
    return upstream, metadata, downloads, snapshots, links


def test_model_directory_pins_every_source_and_omits_bria_and_1024_weights(tmp_path, monkeypatch):
    original, metadata, downloads, snapshots, links = fake_model_hub(tmp_path, monkeypatch)
    directory = mesh.pinned_model_directory(tmp_path)
    result = json.loads((directory / 'pipeline.json').read_text(encoding='utf8'))['args']
    assert json.loads(metadata.read_text(encoding='utf8')) == original
    assert result['rembg_model'] is None and result['low_vram'] is True
    assert result['default_pipeline_type'] == '512'
    assert result['image_cond_model']['args']['model_name'] == str(tmp_path / 'dino-snapshot')
    assert len(links) == 6
    assert len(downloads) == 7
    for repo, filename, revision in downloads:
        selected = mesh.RECONSTRUCTION_MODELS['trellis'] if filename.startswith('ckpts/ss_decoder.') else mesh.RECONSTRUCTION_MODELS['trellis2']
        assert repo == selected['id'] and revision == selected['revision']
        assert len(revision) == 40 and '1024' not in filename and 'bria' not in repo.lower()
    assert snapshots == [(mesh.DINO3_MODEL['id'], mesh.DINO3_MODEL['revision'], ['*.json', '*.safetensors'])]
    assert all(not key.endswith('_1024') for key in result['models'])
    assert all(Path(prefix).is_relative_to(directory) for prefix in result['models'].values())


@pytest.mark.parametrize('decoder', [
    'untrusted/model/ckpts/ss_decoder',
    'microsoft/TRELLIS-image-large/ckpts/../escape',
    'microsoft/TRELLIS-image-large/ckpts/bad\\path',
])
def test_model_directory_rejects_upstream_path_drift_before_weight_fetch(tmp_path, monkeypatch, decoder):
    _, _, downloads, snapshots, links = fake_model_hub(tmp_path, monkeypatch, decoder)
    with pytest.raises(RuntimeError, match='kimliği|dosya yolu'):
        mesh.pinned_model_directory(tmp_path)
    assert len(downloads) == 1 and downloads[0][1] == 'pipeline.json'
    assert snapshots == links == []


@pytest.mark.parametrize('pipeline_type', ['1024_cascade', '1536_cascade'])
def test_cascades_fetch_pinned_1024_shape_and_texture(tmp_path, monkeypatch, pipeline_type):
    _, _, downloads, _, _ = fake_model_hub(tmp_path, monkeypatch)
    directory = mesh.pinned_model_directory(tmp_path, pipeline_type)
    config = json.loads((directory / 'pipeline.json').read_text())['args']
    assert config['default_pipeline_type'] == pipeline_type
    assert 'shape_slat_flow_model_512' in config['models']
    assert 'shape_slat_flow_model_1024' in config['models']
    assert 'tex_slat_flow_model_1024' in config['models']
    assert 'tex_slat_flow_model_512' not in config['models']
    assert any(filename == 'ckpts/shape_1024.safetensors' for _, filename, _ in downloads)
    assert any(filename == 'ckpts/tex_1024.safetensors' for _, filename, _ in downloads)


@pytest.mark.parametrize('count', [12, 24, 160, 200])
def test_held_out_views_disjoint_complete_and_deterministic(count):
    train, evaluation = mesh.split_view_indices(count)
    np.testing.assert_array_equal(evaluation, np.arange(0, count, 12))
    assert not set(train) & set(evaluation)
    assert sorted([*train, *evaluation]) == list(range(count))
    assert .08 <= len(evaluation)/count <= .09


def test_glb_texture_preferred_over_vertex_color_and_axes_uv_restored():
    from PIL import Image
    rgba = np.array([[[255, 0, 0, 255], [0, 255, 0, 128]],
                     [[0, 0, 255, 255], [255, 255, 255, 255]]], dtype=np.uint8)
    model = SimpleNamespace(vertices=np.array([[1, 3, -2], [2, 4, -3], [3, 5, -4]]), faces=[[0, 1, 2]],
        visual=SimpleNamespace(uv=[[0, 1], [1, 1], [0, 0]], vertex_colors=np.zeros((3, 4)),
            material=SimpleNamespace(baseColorTexture=Image.fromarray(rgba), baseColorFactor=[255]*4)))
    arrays = mesh.mesh_arrays(model)
    np.testing.assert_array_equal(arrays['vertices'], [[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    np.testing.assert_array_equal(arrays['texture'], rgba)
    np.testing.assert_array_equal(arrays['uv'], [[0, 0], [1, 0], [0, 1]])
    np.testing.assert_array_equal(arrays['colors'], np.eye(3))
    assert arrays['alpha'][1, 0] == pytest.approx(128/255)
    model.visual = SimpleNamespace(vertex_colors=np.tile([128, 64, 32, 255], (3, 1)))
    fallback = mesh.mesh_arrays(model)
    assert 'texture' not in fallback
    np.testing.assert_allclose(fallback['colors'][0], np.array([128, 64, 32])/255)


def test_ssim_cpu_identity_and_degradation():
    import torch
    # CPU tensors only. No real gsplat/TRELLIS module or CUDA kernel is imported.
    x = torch.linspace(0, 1, 32*32*3, device='cpu').reshape(1, 32, 32, 3)
    assert mesh.image_ssim(x, x).item() == pytest.approx(1, abs=1e-6)
    assert mesh.image_ssim(x, 1-x).item() < .5


@pytest.mark.parametrize('cap,n', [(20, 20), (21, 20), (22, 20), (100, 20)])
def test_growth_budget_before_allocation_with_mock_gsplat(monkeypatch, cap, n):
    import torch
    class FakeDefault:
        grow_scale3d = .01
        revised_opacity = False
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    calls = []
    def mutate(*, params, optimizers, state, mask, revised_opacity=None):
        assert mask.device.type == 'cpu'
        added = int(mask.sum())
        assert len(params['means']) + added <= cap
        calls.append(added)
        params['means'] = torch.cat([params['means'], torch.zeros(added, 3)])
    monkeypatch.setitem(sys.modules, 'gsplat.strategy', SimpleNamespace(DefaultStrategy=FakeDefault))
    monkeypatch.setitem(sys.modules, 'gsplat.strategy.ops', SimpleNamespace(duplicate=mutate, split=mutate))
    strategy = mesh.capped_strategy(cap, 30000)
    scales = torch.full((n, 3), np.log(.001), device='cpu')
    scales[-1] = np.log(.1)  # highest gradient is a split; next is a duplicate
    params = {'means': torch.zeros(n, 3, device='cpu'), 'scales': scales}
    state = {'grad2d': torch.arange(1, n+1, device='cpu').float(),
             'count': torch.ones(n, device='cpu'), 'scene_scale': 1.}
    nd, ns = strategy._grow_gs(params, {}, state, 600)
    assert nd + ns == min(cap-n, n//10)
    assert len(params['means']) <= cap
    assert sum(calls) == nd + ns

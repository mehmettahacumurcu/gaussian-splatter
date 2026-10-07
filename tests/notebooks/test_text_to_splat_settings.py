import ast
import math
import re

import nbformat
import pytest
from pydantic import ValidationError

from backend.notebooks.pipeline_library import PipelineNotebookSpec, build_pipeline_notebook, replace_assignments
from backend.notebooks.source import NotebookSource, REPOSITORY_URL
from backend.notebooks.templates.text_to_splat_settings import (
    GPU_PRESETS, IMAGE_MODELS, SETTING_FIELDS, estimate_requirements, model_provenance, validate_settings,
)
from scripts import generate_pipeline_notebook

SOURCE = NotebookSource(REPOSITORY_URL, 'a' * 40, 'test', 1)


def spec(**kwargs):
    return PipelineNotebookSpec(pipeline='text_to_splat', prompt='a car', **kwargs)


@pytest.mark.parametrize('gpu', list(GPU_PRESETS))
def test_gpu_presets_resolve_models_and_knobs_but_preserve_explicit_overrides(gpu):
    selected = spec(gpu_preset=gpu)
    for field, value in GPU_PRESETS[gpu].items():
        assert getattr(selected, field) == value
    overridden = spec(gpu_preset=gpu, image_model='sdxl', image_steps=17, sparse_cfg=4.5, trellis_seed=101)
    assert (overridden.image_model, overridden.image_steps, overridden.image_guidance) == ('sdxl', 17, 7)
    assert overridden.sparse_cfg == 4.5 and overridden.trellis_seed == 101


@pytest.mark.parametrize('image', list(IMAGE_MODELS))
@pytest.mark.parametrize('route', ['trellis', 'trellis2'])
def test_all_model_route_notebooks_validate_compile_and_record_real_pins(image, route):
    selected = spec(image_model=image, reconstruction_model=route, background_model='birefnet')
    notebook = build_pipeline_notebook(selected, source=SOURCE)
    nbformat.validate(notebook)
    assert notebook.metadata.pipeline.template_version == 2
    records = notebook.metadata.pipeline.models
    assert records.image.id == IMAGE_MODELS[image]['id']
    for record in records.values():
        assert record['id'] and record['license']
        assert re.fullmatch('[a-f0-9]{40}', record['revision'])
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'<cell-{index}>', 'exec')
    cell = next(c for c in notebook.cells if c.metadata.get('stage') == 'config')
    names = {}
    for node in ast.parse(cell.source).body:
        assert isinstance(node, ast.Assign)  # Config remains literal assignments only.
        names[node.targets[0].id] = ast.literal_eval(node.value)
    assert {field: names[field.upper()] for field in SETTING_FIELDS} == {
        field: getattr(selected, field) for field in SETTING_FIELDS}
    negative_line = next(line for line in cell.source.splitlines() if line.startswith('NEGATIVE_PROMPT ='))
    assert ('#@param' in negative_line) == IMAGE_MODELS[image]['negative_prompt']
    assert 'MODEL SETTINGS' in cell.source and 'görüntü' in cell.source.lower()
    bundle = next(c for c in notebook.cells if c.metadata.get('stage') == 'bundle')
    first = ast.parse(bundle.source).body[0]
    for name, source in ast.literal_eval(first.value).items():
        if name.endswith('.py'):
            compile(source, name, 'exec')


@pytest.mark.parametrize('fields', [
    {'gpu_preset': 'A100; evil'}, {'image_model': 'custom/repo'}, {'background_model': "u2net'; evil()"},
    {'reconstruction_model': 'hunyuan3d'}, {'reconstruction_model': 'anything'},
    {'image_steps': 0}, {'image_steps': 101}, {'image_steps': True}, {'image_steps': '25'},
    {'image_guidance': float('nan')}, {'image_guidance': float('inf')}, {'image_guidance': True},
    {'image_resolution': 1536}, {'trellis_seed': -1}, {'trellis_seed': True},
    {'sparse_steps': 0}, {'sparse_cfg': -1}, {'slat_steps': 101}, {'slat_cfg': 20.1},
    {'mesh_views': 11}, {'mesh_views': 121}, {'mesh_fit_iterations': 99}, {'mesh_fit_iterations': 10001},
    {'mesh_splat_cap': 999}, {'mesh_splat_cap': 300001}, {'mesh_splat_cap': 1.1},
    {'image_model': 'flux1_schnell', 'negative_prompt': 'blurry'},
    {'image_model': 'flux2_klein_4b', 'negative_prompt': 'blurry'},
    {'image_model': 'flux1_schnell', 'image_steps': 5},
    {'image_model': 'flux1_schnell', 'image_guidance': 1},
    {'image_model': 'flux2_klein_4b', 'image_guidance': 3},
])
def test_new_settings_reject_invalid_or_injected_values(fields):
    with pytest.raises(ValidationError):
        spec(**fields)


@pytest.mark.parametrize('image', ['sdxl', 'flux1_dev', 'qwen_image'])
def test_supported_negatives_are_literal_data_including_comment_syntax(image):
    prompt = 'car\n#@param __import__("os").system("never")\n車'
    selected = PipelineNotebookSpec(pipeline='text_to_splat', prompt=prompt, image_model=image, negative_prompt=prompt)
    notebook = build_pipeline_notebook(selected, source=SOURCE)
    cell = next(c for c in notebook.cells if c.metadata.get('stage') == 'config')
    namespace = {'__builtins__': {}}
    exec(compile(cell.source, '<literal-config>', 'exec'), namespace)
    assert namespace['PROMPT'] == namespace['NEGATIVE_PROMPT'] == prompt
    assert sum(line.startswith('PROMPT =') for line in cell.source.splitlines()) == 1


def test_form_annotation_survives_unicode_byte_offsets_and_is_never_copied_from_value():
    result = replace_assignments("PROMPT = 'kırmızı' #@param {\"type\":\"string\"}\n", {'PROMPT': "x\n#@param 'bad'"})
    assert result.count('\n') == 1
    assert result.endswith(' #@param {"type":"string"}\n')
    assert ast.literal_eval(ast.parse(result).body[0].value) == "x\n#@param 'bad'"


def test_stage_thresholds_use_peak_not_sum_and_mesh_cap_adds_headroom():
    assert estimate_requirements(spec().model_dump())['min_vram'] == 20
    flux = estimate_requirements(spec(image_model='flux1_dev').model_dump())
    assert flux['min_vram'] == 36
    qwen = estimate_requirements(spec(image_model='qwen_image', reconstruction_model='trellis2').model_dump())
    assert qwen['min_vram'] == 60 and qwen['disk_gib'] == 130
    small = spec(reconstruction_model='trellis2', mesh_splat_cap=150000).model_dump()
    large = spec(reconstruction_model='trellis2', mesh_splat_cap=150001).model_dump()
    assert estimate_requirements(small)['min_vram'] == 28
    assert estimate_requirements(large)['min_vram'] == 32
    assert estimate_requirements(spec(preset='ultra').model_dump())['min_vram'] == 38


def test_manifest_lists_gated_auxiliary_without_unselected_or_unverified_models():
    records = model_provenance(spec(reconstruction_model='trellis2', background_model='birefnet').model_dump())
    assert records['encoder']['gated'] and 'manual' in records['encoder']['access']
    assert records['sparse_structure_decoder']['id'] == 'microsoft/TRELLIS-image-large'
    assert 'briaai' not in repr(records) and 'hunyuan' not in repr(records).lower()
    assert records['background']['license'] == 'MIT'
    u2net = model_provenance(spec().model_dump())['background']
    assert 'unverified' in u2net['license'] and u2net['revision'].startswith('md5:')


def test_form_runtime_validation_matches_api_and_does_not_mutate_input():
    request = dict(prompt='car', seed=17, image_model='flux1_schnell', gpu_preset='a100')
    resolved = validate_settings(request)
    assert resolved['image_steps'] == 4 and resolved['image_guidance'] == 0
    assert resolved['trellis_seed'] == 17 and 'image_steps' not in request
    with pytest.raises(ValueError, match='Hunyuan3D'):
        validate_settings(dict(request, reconstruction_model='hunyuan3d'))
    with pytest.raises(ValueError):
        validate_settings(dict(request, sparse_cfg=math.inf))


def test_other_recipes_reject_text_model_knobs():
    with pytest.raises(ValidationError, match='Model ayarları'):
        PipelineNotebookSpec(pipeline='native', input_mode='folder', input_path='capture', image_model='sdxl')


def test_other_recipe_metadata_round_trips_with_unset_model_defaults():
    original = PipelineNotebookSpec(pipeline='native', input_mode='folder', input_path='capture')
    assert PipelineNotebookSpec(**original.model_dump()) == original


def test_cli_exposes_new_selectors_and_custom_knobs(tmp_path):
    output = tmp_path / 'model-settings.ipynb'
    generate_pipeline_notebook.main(['--pipeline', 'text_to_splat', '--prompt', 'toy',
        '--gpu-preset', 'h100', '--image-model', 'flux2_klein_4b', '--background-model', 'birefnet',
        '--reconstruction-model', 'trellis2', '--image-steps', '4', '--image-guidance', '1',
        '--trellis-seed', '19', '--mesh-views', '36', '--mesh-fit-iterations', '200',
        '--mesh-splat-cap', '10000', '--output', str(output)])
    result = nbformat.read(output, 4)
    nbformat.validate(result)
    assert result.metadata.pipeline.spec.image_model == 'flux2_klein_4b'
    assert result.metadata.pipeline.spec.mesh_views == 36
    assert result.metadata.pipeline.spec.trellis_seed == 19

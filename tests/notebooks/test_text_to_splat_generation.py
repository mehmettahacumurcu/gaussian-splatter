from __future__ import annotations

import ast
import re
from types import SimpleNamespace

import nbformat
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.notebooks.pipeline_library import (
    ASSETS, PRESETS, PipelineNotebookSpec, build_pipeline_notebook, pipeline_filename,
    replace_assignments, resolve_embedded_notebook_source,
)
from backend.notebooks.routes import static_notebook_router
from backend.notebooks.source import NotebookSource, NotebookSourceError, REPOSITORY_URL
from scripts import generate_pipeline_notebook


SOURCE = NotebookSource(REPOSITORY_URL, 'a' * 40, '4dgs-studio.static-notebook', 1)


def _literals(notebook):
    values = {}
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    values[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


def test_text_defaults_normalize_prompt_and_derive_safe_stable_output():
    spec = PipelineNotebookSpec(pipeline='text_to_splat', prompt='  kırmızı yarış arabası  ')
    assert spec.prompt == 'kırmızı yarış arabası'
    assert spec.input_mode == 'text'
    assert spec.input_path == ''
    assert spec.seed == 42
    assert spec.target_splat_count is None
    assert re.fullmatch(r'GaussianTests/text_to_splat/[a-z0-9-]+-[a-f0-9]{8}', spec.output_dir)
    assert spec.output_dir == PipelineNotebookSpec(pipeline='text_to_splat', prompt=spec.prompt).output_dir
    assert spec.output_dir != PipelineNotebookSpec(pipeline='text_to_splat', prompt='kirmizi yaris arabasi').output_dir
    assert '/' not in pipeline_filename(spec)
    assert pipeline_filename(spec).endswith('_text_to_splat_baseline.ipynb')


def test_text_output_can_be_named_result_without_relaxing_input_rules():
    spec = PipelineNotebookSpec(pipeline='text_to_splat', prompt='car',
                                output_dir='MyDrive/GaussianTests/car_result')
    assert spec.output_dir == 'GaussianTests/car_result'
    with pytest.raises(ValidationError):
        PipelineNotebookSpec(pipeline='native', input_mode='folder', input_path=spec.output_dir)


@pytest.mark.parametrize('preset', ['baseline', 'quality', 'ultra', 'max_detail'])
def test_text_generation_embeds_portable_runtime_and_compiles_every_cell(preset):
    spec = PipelineNotebookSpec(pipeline='text_to_splat', prompt='a red sports car', preset=preset)
    notebook = build_pipeline_notebook(spec, source=SOURCE)
    nbformat.validate(notebook)
    assert notebook.metadata.pipeline.id == 'text_to_splat'
    assert notebook.metadata.pipeline.generator_commit == SOURCE.commit_sha
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            compile(cell.source, f'<cell-{index}>', 'exec')
            assert cell.outputs == [] and cell.execution_count is None
    literals = _literals(notebook)
    assert literals['QUALITY_PRESET'] == preset
    assert literals['OUTPUT_DIR'] == '/content/drive/MyDrive/' + spec.output_dir
    assert literals['QUALITY_PRESETS'] == PRESETS['text_to_splat']
    assert set(literals['RUNTIME_FILES']) == {
        'text_to_splat_bootstrap.py', 'text_to_splat_runtime.py', 'text_to_splat_helpers.py',
        'text_to_splat_requirements.txt',
        'text_to_splat_settings.py', 'text_to_splat_image.py', 'text_to_splat_mesh.py',
    }
    for name, source in literals['RUNTIME_FILES'].items():
        assert source == (ASSETS / name).read_text(encoding='utf-8')
        if name.endswith('.py'):
            compile(source, name, 'exec')
    assert all('C:/Users/TAHA' not in cell.source for cell in notebook.cells)


@pytest.mark.parametrize('prompt', [
    'a "red" sports car with a \'glossy\' roof',
    'kırmızı yarış arabası 🚗\n白い背景\narka üç çeyrek görünüm',
    "car'; __import__('os').system('do-not-execute'); #\nPROMPT = 'injected",
    r'a toy with literal \\newlines and {braces} and $shell syntax',
])
def test_prompt_quotes_newlines_unicode_remain_data(prompt):
    spec = PipelineNotebookSpec(
        pipeline='text_to_splat', input_mode='text', prompt=prompt,
        negative_prompt='watermark, "blurry"\nextra wheels', style='ürün fotoğrafı',
        seed=2147483647, output_dir='MyDrive/GaussianTests/custom target', target_splat_count=45000,
    )
    notebook = build_pipeline_notebook(spec, source=SOURCE)
    literals = _literals(notebook)
    assert literals['PROMPT'] == prompt
    assert literals['NEGATIVE_PROMPT'] == spec.negative_prompt
    assert literals['STYLE'] == spec.style
    assert literals['SEED'] == 2147483647
    assert literals['TARGET_SPLAT_COUNT'] == 45000
    assert literals['OUTPUT_DIR'] == '/content/drive/MyDrive/GaussianTests/custom target'
    # Executing only the substituted assignments cannot execute prompt content.
    config = next(cell for cell in notebook.cells if cell.cell_type == 'code' and 'PROMPT =' in cell.source)
    statements = [node for node in ast.parse(config.source).body
                  if isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {
                      'PROMPT', 'NEGATIVE_PROMPT', 'STYLE', 'SEED', 'QUALITY_PRESET', 'OUTPUT_DIR', 'TARGET_SPLAT_COUNT',
                  }]
    assert len(statements) == 7
    namespace = {'__builtins__': {}}
    exec(compile(ast.Module(body=statements, type_ignores=[]), '<config>', 'exec'), namespace)
    assert namespace['PROMPT'] == prompt


@pytest.mark.parametrize('overrides', [
    {'prompt': ''}, {'prompt': ' \n\t '}, {'prompt': 'x' * 2001}, {'prompt': 'bad\x00text'},
    {'negative_prompt': 'x' * 2001}, {'style': 'x' * 501},
    {'seed': -1}, {'seed': 2147483648}, {'seed': 1.5}, {'seed': True}, {'seed': '42'},
    {'target_splat_count': 999}, {'target_splat_count': 2000001}, {'target_splat_count': 1000.5},
    {'target_splat_count': True}, {'preset': 'unknown'},
    {'output_dir': '../escape'}, {'output_dir': '/content/drive/MyDrive/out'},
    {'output_dir': 'C:/out'}, {'output_dir': 'MyDrive'}, {'output_dir': 'folder\\out'},
    {'output_dir': 'folder\nnext'}, {'output_dir': 'folder/../out'},
    {'input_mode': 'folder'}, {'input_path': 'captures/room'},
    {'iterations': 10000}, {'max_gaussians': 100000}, {'unknown_field': True},
])
def test_invalid_text_requests_fail_before_generation(overrides):
    with pytest.raises(ValidationError):
        PipelineNotebookSpec(**{'pipeline': 'text_to_splat', 'prompt': 'a red sports car', **overrides})


@pytest.mark.parametrize('kwargs', [
    {'pipeline': 'native'},
    {'pipeline': 'native', 'input_mode': 'folder'},
    {'pipeline': 'hybrid', 'input_mode': 'dataset_zip', 'input_path': ''},
    {'pipeline': 'spirula', 'input_mode': 'text', 'prompt': 'car'},
    {'pipeline': 'spirula', 'input_mode': 'video', 'input_path': '../escape'},
    {'pipeline': 'native', 'input_mode': 'folder', 'input_path': 'captures/room', 'prompt': 'car'},
])
def test_existing_pipeline_input_requirements_are_preserved(kwargs):
    with pytest.raises(ValidationError):
        PipelineNotebookSpec(**kwargs)


def test_literal_replacement_fails_closed_on_template_drift():
    with pytest.raises(ValueError, match='drift'):
        replace_assignments('PROMPT = "x"\n', {'PROMPT': 'safe', 'STYLE': 'missing'})


def test_embedded_source_records_local_head_without_upstream_or_release_override(monkeypatch):
    calls = []

    def fake_git(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='b' * 40 + '\n')

    monkeypatch.setenv('STATIC_NOTEBOOK_COMMIT_SHA', 'c' * 40)
    monkeypatch.setattr('backend.notebooks.pipeline_library.subprocess.run', fake_git)
    source = resolve_embedded_notebook_source()
    assert source.commit_sha == 'b' * 40
    assert calls == [['git', 'rev-parse', 'HEAD']]


def test_embedded_source_rejects_bad_provenance(monkeypatch):
    monkeypatch.setattr('backend.notebooks.pipeline_library.subprocess.run',
                        lambda *args, **kwargs: SimpleNamespace(stdout='not-a-commit'))
    with pytest.raises(NotebookSourceError):
        resolve_embedded_notebook_source()


def test_text_route_download_works_without_remotely_reachable_app_source(monkeypatch):
    def unavailable_remote():
        raise NotebookSourceError('local feature commit is unpushed')

    monkeypatch.setattr('backend.notebooks.routes.resolve_notebook_source', unavailable_remote)
    app = FastAPI()
    app.include_router(static_notebook_router)
    client = TestClient(app)
    catalog = client.get('/notebooks/static/pipelines').json()
    assert catalog['presets']['text_to_splat'] == PRESETS['text_to_splat']
    response = client.post('/notebooks/static/pipeline', json={
        'pipeline': 'text_to_splat', 'prompt': 'a red sports car', 'seed': 123,
    })
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('application/x-ipynb+json')
    assert '_text_to_splat_baseline.ipynb' in response.headers['content-disposition']
    notebook = nbformat.reads(response.text, 4)
    nbformat.validate(notebook)
    assert _literals(notebook)['SEED'] == 123
    assert re.fullmatch('[a-f0-9]{40}', notebook.metadata.pipeline.generator_commit)
    rejected = client.post('/notebooks/static/pipeline', json={'pipeline': 'text_to_splat', 'prompt': ''})
    assert rejected.status_code == 422


def test_text_cli_is_portable_without_pushing_and_does_not_overwrite(tmp_path, monkeypatch):
    def unavailable_remote():
        raise NotebookSourceError('local feature commit is unpushed')

    monkeypatch.setattr(generate_pipeline_notebook, 'resolve_notebook_source', unavailable_remote)
    output = tmp_path / 'red_car.ipynb'
    args = ['--pipeline', 'text_to_splat', '--prompt', 'a red sports car', '--preset', 'quality',
            '--seed', '123', '--target-splat-count', '25000', '--output', str(output)]
    generate_pipeline_notebook.main(args)
    notebook = nbformat.read(output, 4)
    nbformat.validate(notebook)
    assert notebook.metadata.pipeline.spec.input_mode == 'text'
    assert _literals(notebook)['TARGET_SPLAT_COUNT'] == 25000
    with pytest.raises(FileExistsError):
        generate_pipeline_notebook.main(args)

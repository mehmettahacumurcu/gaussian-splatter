import ast
import base64
import hashlib
import io
import zipfile

import nbformat
import pytest
from pydantic import ValidationError

from backend.notebooks.pipeline_library import PipelineNotebookSpec, build_pipeline_notebook
from backend.notebooks.source import NotebookSource, REPOSITORY_URL

SOURCE = NotebookSource(REPOSITORY_URL, 'a' * 40, '4dgs-studio.static-notebook', 1)


@pytest.mark.parametrize('pipeline,mode', [('native', 'folder'), ('hybrid', 'dataset_zip'), ('spirula', 'video'), ('spirula', 'dataset_folder')])
def test_all_pipelines_generate_clean_parseable_notebooks(pipeline, mode):
    spec = PipelineNotebookSpec(pipeline=pipeline, input_mode=mode, input_path='captures/room')
    notebook = build_pipeline_notebook(spec, source=SOURCE)
    nbformat.validate(notebook)
    assert notebook.metadata['pipeline']['id'] == pipeline
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.outputs == [] and cell.execution_count is None


def test_hybrid_parameters_and_bundle_are_portable():
    path = "captures/room'; raise RuntimeError('injection'); #.zip"
    nb = build_pipeline_notebook(PipelineNotebookSpec(pipeline='hybrid', input_mode='dataset_zip', input_path=path, iterations=61000, max_gaussians=4000000), source=SOURCE)
    sources = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    assignments = {n.targets[0].id: n.value for n in ast.walk(ast.parse(sources)) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert ast.literal_eval(assignments['DATASET_PATH']) == '/content/drive/MyDrive/' + path
    assert ast.literal_eval(assignments['ITERATIONS_OVERRIDE']) == 61000
    raw = base64.b64decode(ast.literal_eval(assignments['BUNDLE_B64']))
    assert hashlib.sha256(raw).hexdigest() == ast.literal_eval(assignments['BUNDLE_SHA256'])
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert 'spirula_import.py' in archive.namelist()
        assert 'backend/model/trainer.py' in archive.namelist()
    assert 'E:\\' not in sources and 'C:/Users/TAHA' not in sources


def test_spirula_configuration_and_recovery():
    nb = build_pipeline_notebook(PipelineNotebookSpec(pipeline='spirula', input_mode='video', input_path='captures/test.MOV', preset='quality', fps=8), source=SOURCE)
    src = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    tree = ast.parse(src)
    vals = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert ast.literal_eval(vals['TRAIN_ITERATIONS']) == 60000
    assert ast.literal_eval(vals['TRAIN_CAP_MAX']) == 6000000
    assert ast.literal_eval(vals['TARGET_FPS']) == 8
    assert 'accepted_exit_codes=(0, 3)' in src
    assert 'validate_geometry_maps' in src
    assert "'--quality', TRAIN_QUALITY" in src


@pytest.mark.parametrize('kwargs', [
    dict(pipeline='hybrid', input_mode='video', input_path='x'),
    dict(pipeline='native', input_mode='dataset_zip', input_path='x'),
    dict(pipeline='spirula', input_mode='video', input_path='../bad'),
    dict(pipeline='spirula', input_mode='video', input_path='D:/video.mov'),
    dict(pipeline='spirula', input_mode='video', input_path='x', fps=0),
])
def test_invalid_requests_fail_before_generation(kwargs):
    with pytest.raises(ValidationError):
        PipelineNotebookSpec(**kwargs)

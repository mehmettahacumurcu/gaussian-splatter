import ast
import json
import struct
import zipfile
from pathlib import Path

import nbformat
import pytest
from pydantic import ValidationError

from backend.notebooks.spirula_preprocess import PreprocessSpec, build_preprocess_notebook
from backend.notebooks.templates.spirula_preprocess_runtime import Preprocessor
from backend.notebooks.source import NotebookSource, REPOSITORY_URL

SOURCE = NotebookSource(REPOSITORY_URL, 'a' * 40, '4dgs-studio.static-notebook', 1)


def model(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'cameras.bin').write_bytes(struct.pack('<QiiQQ4d', 1, 1, 1, 8, 6, 4, 4, 4, 3))
    (folder / 'points3D.bin').write_bytes(struct.pack('<Q', 0))
    with (folder / 'images.bin').open('wb') as f:
        f.write(struct.pack('<Q', len(names)))
        for i, name in enumerate(names):
            f.write(struct.pack('<i7di', i + 1, 1, 0, 0, 0, 0, 0, 0, 1))
            f.write(name.encode() + b'\0' + struct.pack('<Q', 0))


def spec(**settings):
    return PreprocessSpec(inputs=[{'kind': 'video', 'path': 'captures/room.MOV', 'fps': 4}], settings=settings)


def test_generated_recipe_is_portable_and_never_invokes_training(tmp_path):
    cfg = spec(generate_geometry=False, quality='extreme', pairs='sequential', overlap=18)
    nb = build_preprocess_notebook(cfg, source=SOURCE)
    nbformat.validate(nb)
    ns = {}
    for cell in nb.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert not cell.outputs and cell.execution_count is None
            if cell.metadata.get('stage') in ('config', 'recipe'):
                exec(cell.source, ns)
    assert ns['CONFIG'] == cfg.model_dump()
    commands = []
    def run(name, args, **kw):
        commands.append(args)
        if 'sfm' in args:
            model(tmp_path / 'dataset/sparse/0', ['a.jpg'])
            model(tmp_path / 'dataset/sparse/1', ['b.jpg'])
            return 'RESULT: PARTIAL\nReprojection error: mean 0.8 px\nEXIT_CODE: 3'
        raise AssertionError(f'Unexpected command: {args}')
    prep = ns['Preprocessor'](ns['CONFIG'], tmp_path, 'spirula', 'uuid:123', run)
    images = tmp_path / 'dataset/images'
    images.mkdir(parents=True)
    for name in ('a.jpg', 'b.jpg', 'c.jpg'):
        (images / name).write_bytes(b'image')
    prep.reconstruct()
    prep.geometry()
    archive = prep.package()
    assert len(commands) == 1
    assert commands[0][1:3] == ['sfm', 'auto']
    assert commands[0][commands[0].index('--overlap') + 1] == '18'
    assert prep.report['registered_union'] == 2
    assert prep.report['selected_images'] == 1
    assert prep.report['coverage'] == pytest.approx(2 / 3)
    assert prep.report['selected_coverage'] == pytest.approx(1 / 3)
    with zipfile.ZipFile(archive) as z:
        assert {'images/a.jpg', 'sparse/0/images.bin', 'sparse/1/images.bin', 'sfm_report.json', 'preprocess_settings.json'} <= set(z.namelist())
        assert not any('training' in n for n in z.namelist())


def test_options_reach_actual_video_command_without_shell_interpolation(tmp_path):
    path = "captures/a{room}'; echo bad.MOV"
    cfg = PreprocessSpec(inputs=[{'kind': 'video', 'path': path, 'fps': 6}], settings={
        'adaptive_fps': True, 'sharpness_window': 5, 'generate_geometry': False})
    drive = tmp_path / 'drive'
    source = drive / path
    source.parent.mkdir(parents=True)
    source.write_bytes(b'video')
    commands = []
    def run(name, args, **kw):
        commands.append(args)
        if args[0] == 'ffprobe':
            return 'COMMAND: ' + repr(args) + '\n' + json.dumps({'streams': [{'avg_frame_rate': '30/1'}]}) + '\nEXIT_CODE: 0'
        if 'extract' in args:
            out = Path(args[args.index('--out') + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / '00000.png').write_bytes(b'image')
            return ''
        raise AssertionError(args)
    prep = Preprocessor(cfg.model_dump(), tmp_path / 'work', 'spirula', 'uuid:123', run, drive_root=drive)
    prep.prepare()
    cmd = commands[-1]
    assert cmd[1:3] == ['sam', 'extract']
    assert cmd[cmd.index('--skip') + 1] == '5'
    assert cmd[cmd.index('--keep') + 1] == '5'
    assert '--adaptive' in cmd
    assert Path(cmd[3]).read_bytes() == b'video'


@pytest.mark.parametrize('change', [
    {'inputs': [{'kind': 'video', 'path': '../escape', 'fps': 4}]},
    {'settings': {'quality': 'imaginary'}},
    {'settings': {'max_features': -1}},
    {'settings': {'mask_objects': True, 'mask_prompt': ''}},
    {'settings': {'distortion': '1; arbitrary'}},
    {'output_path': '/content/drive/MyDrive/output'},
])
def test_invalid_preprocessing_request_is_rejected(change):
    data = spec().model_dump()
    data.update(change)
    with pytest.raises(ValidationError):
        PreprocessSpec.model_validate(data)


def test_geometry_disabled_never_downloads_or_runs_model(tmp_path):
    def forbidden(*args, **kw):
        raise AssertionError('Geometry should be skipped')
    prep = Preprocessor(spec(generate_geometry=False).model_dump(), tmp_path, 'spirula', 'auto', forbidden)
    prep.geometry()


def test_model_report_rejects_path_traversal(tmp_path):
    def run(*args, **kw):
        model(tmp_path / 'dataset/sparse/0', ['../outside.png'])
        return 'EXIT_CODE: 0'
    prep = Preprocessor(spec().model_dump(), tmp_path, 'spirula', 'auto', run)
    (prep.dataset / 'images').mkdir(parents=True)
    (prep.dataset / 'images/a.jpg').write_bytes(b'image')
    with pytest.raises(ValueError, match='image'):
        prep.reconstruct()


def test_photo_masks_are_renamed_for_each_input_and_nested_camera(tmp_path, monkeypatch):
    from PIL import Image
    for folder in ('drive/one/cam', 'drive/two'):
        path = tmp_path / folder
        path.mkdir(parents=True)
        for name in ('a.jpg', 'b.jpg'):
            Image.new('RGB', (8, 6)).save(path / name)
    cfg = PreprocessSpec(inputs=[{'kind': 'photos', 'path': 'one'}, {'kind': 'photos', 'path': 'two'}],
                         settings={'mask_objects': True, 'generate_geometry': False, 'capture_kind': 'individual'})
    commands = []
    def run(name, args, **kw):
        commands.append(args)
        assert args[1:3] == ['sam', 'track']
        assert len(list(Path(args[args.index('--frames') + 1]).iterdir())) == 1
        out = Path(args[args.index('--out') + 1])
        out.mkdir(parents=True, exist_ok=True)
        Image.new('L', (8, 6), 255).save(out / 'frame_00000.png')
        return 'EXIT_CODE: 0'
    monkeypatch.setattr(Preprocessor, 'sam_model', lambda self: tmp_path / 'sam3.ggml')
    prep = Preprocessor(cfg.model_dump(), tmp_path / 'work', 'spirula', 'auto', run, drive_root=tmp_path / 'drive')
    prep.prepare()
    assert len(commands) == 4
    assert (prep.dataset / 'masks/input_01/cam/a.png').exists()
    assert (prep.dataset / 'masks/input_02/a.png').exists()
    assert (prep.dataset / 'masks/input_01/cam/b.png').exists()
    assert '--sequence' not in prep.sfm_args()


@pytest.mark.parametrize('missing', [False, True])
def test_geometry_crash_requires_every_selected_map_before_packaging(tmp_path, missing):
    from PIL import Image
    import numpy as np
    def run(name, args, **kw):
        assert kw['accepted_exit_codes'] == (0, -11)
        assert '--overwrite' in args
        root = Path(args[2])
        (root / 'normals').mkdir(parents=True)
        (root / 'depths').mkdir()
        Image.new('RGB', (8, 6)).save(root / 'normals/a.png')
        if not missing:
            Image.fromarray(np.full((6, 8), 500, dtype=np.uint16)).save(root / 'depths/a.png')
        return 'done: 1 written, 0 already there\nEXIT_CODE: -11'
    prep = Preprocessor(spec().model_dump(), tmp_path, 'spirula', 'auto', run)
    model(prep.dataset / 'sparse/0', ['a.jpg'])
    (prep.dataset / 'images').mkdir()
    prep.report = {'selected_model': '0', 'warnings': [], 'geometry': {'old': True}}
    if missing:
        with pytest.raises(ValueError, match='map'):
            prep.stage('geometry')
        with pytest.raises(ValueError, match='Geometri'):
            prep.package()
        assert json.loads((prep.reports / 'stages.json').read_text())['geometry']['status'] == 'failed'
    else:
        prep.geometry()
        assert prep.report['geometry']['depths'] == 1
        assert '-11' in prep.report['warnings'][0]
        assert (prep.dataset / 'depths/a.png').exists()


def test_bootstrap_drive_outage_preserves_zip_and_can_retry(tmp_path, monkeypatch):
    import shutil
    nb = build_preprocess_notebook(spec(), source=SOURCE)
    bootstrap = next(c.source for c in nb.cells if c.cell_type == 'code' and 'def retry_drive_export' in c.source)
    functions = [n for n in ast.parse(bootstrap).body if isinstance(n, ast.FunctionDef)]
    work = tmp_path / 'work'
    work.mkdir()
    output = work / 'outputs'
    output.mkdir()
    archive = output / 'dataset.zip'
    archive.write_bytes(b'complete dataset')
    manifest = {'preprocessing_status': 'complete'}
    ns = {'Path': Path, 'WORK': work, 'OUT': output, 'LOGS': work / 'logs', 'DRIVE_OUT': tmp_path / 'drive',
          'MANIFEST_PATH': work / 'manifest.json', 'manifest': manifest, 'json': json, 'shutil': shutil}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<bootstrap>', 'exec'), ns)
    original = shutil.copy2
    def unavailable(*args, **kw):
        raise OSError(107, 'Transport endpoint is not connected')
    monkeypatch.setattr(shutil, 'copy2', unavailable)
    assert ns['retry_drive_export']()
    assert manifest['status'] == 'export_pending'
    assert archive.read_bytes() == b'complete dataset'
    monkeypatch.setattr(shutil, 'copy2', original)
    assert ns['retry_drive_export']() == []
    assert manifest['status'] == 'complete'
    assert (tmp_path / 'drive/dataset.zip').read_bytes() == archive.read_bytes()


@pytest.mark.parametrize('broken_file', ['cameras.bin', 'points3D.bin'])
def test_truncated_colmap_component_never_becomes_a_dataset(tmp_path, broken_file):
    def run(*args, **kw):
        folder = tmp_path / 'dataset/sparse/0'
        model(folder, ['a.jpg'])
        (folder / broken_file).write_bytes(struct.pack('<Q', 1))
        return 'EXIT_CODE: 3'
    prep = Preprocessor(spec().model_dump(), tmp_path, 'spirula', 'auto', run)
    (prep.dataset / 'images').mkdir(parents=True)
    (prep.dataset / 'images/a.jpg').write_bytes(b'image')
    with pytest.raises(ValueError, match='COLMAP'):
        prep.reconstruct()

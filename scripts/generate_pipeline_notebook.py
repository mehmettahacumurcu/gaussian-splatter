"""Generate a portable notebook without running training.

python -m scripts.generate_pipeline_notebook --pipeline spirula --input-mode video \
    --input-path GaussianTests/inputs/capture.MOV --preset quality --output room.ipynb
"""
import argparse
from pathlib import Path

from backend.notebooks.builder import serialize_notebook
from backend.notebooks.pipeline_library import (
    PipelineNotebookSpec, build_pipeline_notebook, resolve_embedded_notebook_source,
)
from backend.notebooks.source import resolve_notebook_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', choices=['native', 'hybrid', 'spirula', 'text_to_splat'], required=True)
    parser.add_argument('--input-mode', choices=['folder', 'video', 'dataset_zip', 'dataset_folder', 'text'])
    parser.add_argument('--input-path', default='', help='Path relative to MyDrive; required except for text_to_splat')
    parser.add_argument('--preset', choices=['baseline', 'quality', 'ultra'], default='baseline')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--max-gaussians', type=int)
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--model-name', default='0')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--geometry-model', choices=['moge2-vits', 'moge2-vitb', 'moge2-vitl'], default='moge2-vitb')
    parser.add_argument('--sfm-quality', choices=['high', 'extreme'], default='high')
    parser.add_argument('--no-depth', action='store_true')
    parser.add_argument('--prompt', default='', help='Object description (1..2000 characters) for text_to_splat')
    parser.add_argument('--negative-prompt', default='')
    parser.add_argument('--style', default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', help='Text splat output folder relative to MyDrive (not the notebook destination)')
    parser.add_argument('--target-splat-count', type=int)
    for flag, choices in {
        'gpu-preset': ['l4', 'a100', 'h100', 'rtx_pro_6000'],
        'image-model': ['sdxl', 'flux1_dev', 'flux1_schnell', 'flux2_klein_4b', 'qwen_image'],
        'background-model': ['u2net', 'birefnet'],
        'reconstruction-model': ['trellis', 'trellis2', 'hunyuan3d'],
    }.items():
        parser.add_argument('--' + flag, choices=choices, default=argparse.SUPPRESS)
    for flag in ('image-steps', 'image-resolution', 'trellis-seed', 'sparse-steps', 'slat-steps',
                 'mesh-views', 'mesh-fit-iterations', 'mesh-splat-cap'):
        parser.add_argument('--' + flag, type=int, default=argparse.SUPPRESS)
    for flag in ('image-guidance', 'sparse-cfg', 'slat-cfg'):
        parser.add_argument('--' + flag, type=float, default=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args(argv))
    output = args.pop('output')
    args['generate_depth'] = not args.pop('no_depth')
    spec = PipelineNotebookSpec(**args)
    source = resolve_embedded_notebook_source() if spec.pipeline == 'text_to_splat' else resolve_notebook_source()
    raw = serialize_notebook(build_pipeline_notebook(spec, source=source))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite a previous experiment notebook.
    with output.open('xb') as stream:
        stream.write(raw)
    print(output.resolve())


if __name__ == '__main__':
    main()

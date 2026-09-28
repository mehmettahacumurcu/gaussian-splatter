"""Generate a portable notebook without running training.

python -m scripts.generate_pipeline_notebook --pipeline spirula --input-mode video \
    --input-path GaussianTests/inputs/capture.MOV --preset quality --output room.ipynb
"""
import argparse
from pathlib import Path

from backend.notebooks.builder import serialize_notebook
from backend.notebooks.pipeline_library import PipelineNotebookSpec, build_pipeline_notebook
from backend.notebooks.source import resolve_notebook_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', choices=['native', 'hybrid', 'spirula'], required=True)
    parser.add_argument('--input-mode', choices=['folder', 'video', 'dataset_zip', 'dataset_folder'], required=True)
    parser.add_argument('--input-path', required=True, help='Path relative to MyDrive')
    parser.add_argument('--preset', choices=['baseline', 'quality', 'ultra'], default='baseline')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--max-gaussians', type=int)
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--model-name', default='0')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--geometry-model', choices=['moge2-vits', 'moge2-vitb', 'moge2-vitl'], default='moge2-vitb')
    parser.add_argument('--sfm-quality', choices=['high', 'extreme'], default='high')
    parser.add_argument('--no-depth', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop('output')
    args['generate_depth'] = not args.pop('no_depth')
    spec = PipelineNotebookSpec(**args)
    raw = serialize_notebook(build_pipeline_notebook(spec, source=resolve_notebook_source()))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite a previous experiment notebook.
    with output.open('xb') as stream:
        stream.write(raw)
    print(output.resolve())


if __name__ == '__main__':
    main()

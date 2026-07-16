from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat


REPOSITORY_URL = "https://github.com/mehmettahacumurcu/gaussian-splatter.git"
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def _tag(name: str) -> dict[str, list[str]]:
    return {"tags": [name]}


def build_learned_quality_notebook(
    commit_sha: str,
    *,
    repository_url: str = REPOSITORY_URL,
) -> nbformat.NotebookNode:
    if _FULL_SHA.fullmatch(commit_sha) is None:
        raise ValueError("notebook source requires a full commit SHA")
    if repository_url != REPOSITORY_URL:
        raise ValueError("notebook repository URL is not approved")
    cells = [
        nbformat.v4.new_markdown_cell(
            "# A100 learned-quality Gaussian splat test\n\n"
            "Choose an **A100 High-RAM** runtime and use **Runtime → Run all**. "
            "This isolated experiment keeps the legacy result untouched, publishes "
            "`<input>_learned_test_result`, and includes diagnostics—not a web viewer.",
            metadata=_tag("title"),
        ),
        nbformat.v4.new_code_cell(
            'INPUT_FOLDER = ""  # @param {type:"string"}\n',
            metadata=_tag("config"),
        ),
        nbformat.v4.new_code_cell(
            "import json, shutil, subprocess\n"
            "gpu_line = subprocess.run([\n"
            "    'nvidia-smi', '--query-gpu=name,memory.total',\n"
            "    '--format=csv,noheader,nounits',\n"
            "], check=True, capture_output=True, text=True).stdout.splitlines()[0]\n"
            "gpu_name, memory_mib = (part.strip() for part in gpu_line.rsplit(',', 1))\n"
            "vram_gib = float(memory_mib) / 1024.0\n"
            "assert 'A100' in gpu_name.upper(), f'A100 required; detected {gpu_name}'\n"
            "assert vram_gib >= 39.0, f'At least 39 GiB VRAM required; detected {vram_gib:.1f}'\n"
            "disk_gib = shutil.disk_usage('/content').free / (1024 ** 3)\n"
            "assert disk_gib >= 80.0, f'At least 80 GiB local disk required; detected {disk_gib:.1f}'\n"
            "print(json.dumps({'gpu': gpu_name, 'vram_gib': round(vram_gib, 1), "
            "'disk_free_gib': round(disk_gib, 1)}, sort_keys=True))\n",
            metadata=_tag("preflight"),
        ),
        nbformat.v4.new_code_cell(
            "from google.colab import drive\n"
            "from pathlib import Path\n"
            "drive.mount('/content/drive')\n"
            "DRIVE_ROOT = Path('/content/drive/MyDrive').resolve()\n",
            metadata=_tag("drive"),
        ),
        nbformat.v4.new_code_cell(
            "import json, unicodedata\n"
            "from pathlib import Path, PurePosixPath\n"
            "raw_folder = INPUT_FOLDER.strip()\n"
            "if not raw_folder:\n"
            "    raw_folder = input('MyDrive-relative input folder: ').strip()\n"
            "assert raw_folder and '\\\\' not in raw_folder, 'Use a MyDrive-relative POSIX path'\n"
            "assert not any(unicodedata.category(ch) == 'Cc' for ch in raw_folder)\n"
            "folder = PurePosixPath(raw_folder)\n"
            "assert not folder.is_absolute() and folder.parts\n"
            "assert all(part not in {'', '.', '..'} for part in folder.parts)\n"
            "assert not raw_folder.endswith(('_result', '_learned_test_result', "
            "'_learned_test_diagnostics'))\n"
            "INPUT_PATH = DRIVE_ROOT.joinpath(*folder.parts).resolve()\n"
            "INPUT_PATH.relative_to(DRIVE_ROOT)\n"
            "assert INPUT_PATH.is_dir(), f'Input folder does not exist: {INPUT_PATH}'\n"
            "RUN_SPEC = {'schema_version': 1, 'input_folder': folder.as_posix(), "
            "'publish': {'replace_owned_result': True}}\n"
            "SPEC_PATH = Path('/content/learned_spec.json')\n"
            "with SPEC_PATH.open('w', encoding='utf-8') as handle:\n"
            "    json.dump(RUN_SPEC, handle, sort_keys=True, separators=(',', ':'))\n"
            "print(f'Input: {INPUT_PATH}')\n"
            "print(f'Result: {INPUT_PATH.with_name(INPUT_PATH.name + \"_learned_test_result\")}')\n",
            metadata=_tag("path-spec"),
        ),
        nbformat.v4.new_code_cell(
            "import shutil, subprocess\n"
            "from pathlib import Path\n"
            "SOURCE_ROOT = Path('/content/gaussian-splatter-src')\n"
            "if SOURCE_ROOT.exists():\n"
            "    shutil.rmtree(SOURCE_ROOT)\n"
            f"REPOSITORY_URL = {repository_url!r}\n"
            f"COMMIT_SHA = {commit_sha!r}\n"
            "subprocess.run(['git', 'clone', '--no-checkout', REPOSITORY_URL, "
            "str(SOURCE_ROOT)], check=True)\n"
            "subprocess.run(['git', '-C', str(SOURCE_ROOT), 'checkout', '--detach', "
            "COMMIT_SHA], check=True)\n"
            "actual = subprocess.run(['git', '-C', str(SOURCE_ROOT), 'rev-parse', "
            "'HEAD'], check=True, capture_output=True, text=True).stdout.strip()\n"
            "assert actual == COMMIT_SHA, 'Immutable source checkout mismatch'\n",
            metadata=_tag("checkout"),
        ),
        nbformat.v4.new_code_cell(
            "import subprocess\n"
            "subprocess.run(['bash', 'colab/static_notebook_bootstrap.sh'], "
            "cwd=SOURCE_ROOT, check=True)\n",
            metadata=_tag("bootstrap"),
        ),
        nbformat.v4.new_code_cell(
            "import sys\n"
            "sys.path.insert(0, str(SOURCE_ROOT))\n"
            "from huggingface_hub import snapshot_download\n"
            "from experiments.learned_quality.dependencies import (\n"
            "    install_learned_environment, materialize_pinned_assets,\n"
            ")\n"
            "def resolved_revision(local_path):\n"
            "    metadata_root = Path(local_path) / '.cache' / 'huggingface' / 'download'\n"
            "    revisions = set()\n"
            "    for metadata in metadata_root.rglob('*.metadata'):\n"
            "        first = metadata.read_text(encoding='utf-8').splitlines()[0].strip()\n"
            "        if len(first) == 40:\n"
            "            revisions.add(first)\n"
            "    assert len(revisions) == 1, f'Cannot authenticate checkpoint: {local_path}'\n"
            "    return revisions.pop()\n"
            "LEARNED_ENV = install_learned_environment(Path(sys.executable).resolve())\n"
            "LEARNED_ASSETS = materialize_pinned_assets(\n"
            "    LEARNED_ENV, downloader=snapshot_download, resolve_revision=resolved_revision,\n"
            ")\n",
            metadata=_tag("learned-dependencies"),
        ),
        nbformat.v4.new_code_cell(
            "from experiments.learned_quality.dependencies import verify_learned_environment\n"
            "MODEL_MANIFEST = verify_learned_environment(LEARNED_ENV, LEARNED_ASSETS)\n"
            "assert MODEL_MANIFEST.path == Path('/content/learned-env/model_manifest.json')\n"
            "print(f'Verified model manifest: {MODEL_MANIFEST.path}')\n",
            metadata=_tag("verify"),
        ),
        nbformat.v4.new_code_cell(
            "import os, subprocess\n"
            "environment = dict(os.environ)\n"
            "environment['LEARNED_MODEL_MANIFEST'] = str(MODEL_MANIFEST.path)\n"
            "subprocess.run(\n"
            "    [\n"
            '        "/content/learned-env/bin/python",\n'
            '        "scripts/learned_quality_run.py",\n'
            '        "--spec",\n'
            '        "/content/learned_spec.json",\n'
            "    ],\n"
            "    cwd=SOURCE_ROOT,\n"
            "    env=environment,\n"
            "    check=True,\n"
            ")\n",
            metadata=_tag("execute"),
        ),
        nbformat.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "receipt = json.loads(Path('/content/learned_run_result.json').read_text(encoding='utf-8'))\n"
            "assert receipt.get('status') == 'success', receipt\n"
            "expected = INPUT_PATH.with_name(INPUT_PATH.name + '_learned_test_result').resolve()\n"
            "actual = Path(receipt['final_path']).resolve()\n"
            "assert actual == expected, f'Unexpected result path: {actual}'\n"
            "success = json.loads((actual / '_SUCCESS').read_text(encoding='utf-8'))\n"
            "assert success['run_id'] == receipt['run_id']\n"
            "required = (\n"
            "    'splat.ply', 'quality_report.json', 'experiment_report.json',\n"
            "    'diagnostics/masks_contact_sheet.png',\n"
            "    'diagnostics/depth_contact_sheet.png',\n"
            "    'diagnostics/geometry_contact_sheet.png',\n"
            "    'diagnostics/final_render_contact_sheet.png',\n"
            ")\n"
            "for relative in required:\n"
            "    assert (actual / relative).is_file(), relative\n"
            "RESULT_FOLDER = actual\n",
            metadata=_tag("validate"),
        ),
        nbformat.v4.new_code_cell(
            "from IPython.display import Image as DisplayImage, display\n"
            "CONTACT_SHEETS = (\n"
            "    'masks_contact_sheet.png', 'depth_contact_sheet.png',\n"
            "    'geometry_contact_sheet.png', 'final_render_contact_sheet.png',\n"
            ")\n"
            "print(f'Result folder: {RESULT_FOLDER}')\n"
            "print(f'Splat: {RESULT_FOLDER / \"splat.ply\"}')\n"
            "print(f'Experiment report: {RESULT_FOLDER / \"experiment_report.json\"}')\n"
            "for name in CONTACT_SHEETS:\n"
            "    path = RESULT_FOLDER / 'diagnostics' / name\n"
            "    print(path)\n"
            "    display(DisplayImage(filename=str(path)))\n"
            "print('Use splat.ply in SuperSplat or the existing 4DGS Studio viewer.')\n",
            metadata=_tag("summary"),
        ),
    ]
    for cell in cells:
        cell["id"] = cell.metadata["tags"][0]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "generator": {
                "id": "4dgs-studio.learned-quality-a100-notebook",
                "version": 1,
                "commit_sha": commit_sha,
            },
        },
    )
    nbformat.validate(notebook)
    return notebook


def write_learned_quality_notebook(path: Path, commit_sha: str) -> Path:
    notebook = build_learned_quality_notebook(commit_sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path, version=4)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the pinned A100 experiment")
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("colab/learned_quality_a100_experiment.ipynb"),
    )
    args = parser.parse_args()
    write_learned_quality_notebook(args.output, args.commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

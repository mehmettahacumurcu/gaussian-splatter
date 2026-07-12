from __future__ import annotations

import json

import nbformat

from .models import StaticNotebookRunSpec
from .source import NotebookSource


def _cell_tag(name: str) -> dict[str, list[str]]:
    return {"tags": [name]}


def build_static_cells(
    spec: StaticNotebookRunSpec,
    source: NotebookSource,
) -> list[nbformat.NotebookNode]:
    canonical_json = json.dumps(
        spec.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    run_spec_source = (
        "import json\n"
        f"RUN_SPEC_JSON = {canonical_json!r}\n"
        "RUN_SPEC = json.loads(RUN_SPEC_JSON)\n"
        "with open('/content/run_spec.json', 'w', encoding='utf-8') as handle:\n"
        "    json.dump(RUN_SPEC, handle, sort_keys=True, separators=(',', ':'))\n"
    )
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Static Gaussian Splat — quality-first Run All\n\n"
            "Select a GPU runtime, then use **Runtime → Run all**. The notebook "
            "mounts Drive and writes the portable result beside your capture folder. "
            "It does not bundle a web viewer.",
            metadata=_cell_tag("title"),
        ),
        nbformat.v4.new_code_cell(
            run_spec_source,
            metadata=_cell_tag("run-spec"),
        ),
        nbformat.v4.new_code_cell(
            "import json, shutil, subprocess, sys\n"
            "from pathlib import Path\n"
            "gpu_line = subprocess.run([\n"
            "    'nvidia-smi', '--query-gpu=name,memory.total',\n"
            "    '--format=csv,noheader,nounits'\n"
            "], check=True, capture_output=True, text=True).stdout.splitlines()[0]\n"
            "gpu_name, memory_mib = (part.strip() for part in gpu_line.rsplit(',', 1))\n"
            "vram_gb = float(memory_mib) / 1024\n"
            "profile = RUN_SPEC['quality']['profile']\n"
            "minimum_vram = {'balanced_l4': 22, 'high': 30, 'premium': 46, 'ultra': 75}\n"
            "assert vram_gb >= minimum_vram[profile], (\n"
            "    f'{profile} needs {minimum_vram[profile]} GB VRAM; found {vram_gb:.1f} GB'\n"
            ")\n"
            "disk_gb = shutil.disk_usage('/content').free / (1024 ** 3)\n"
            "python_version = subprocess.run([sys.executable, '--version'], check=True, "
            "capture_output=True, text=True).stdout.strip()\n"
            "assert disk_gb >= 25, f'At least 25 GB free disk is required; found {disk_gb:.1f}'\n"
            "print(json.dumps({'gpu': gpu_name, 'vram_gb': round(vram_gb, 1), "
            "'disk_free_gb': round(disk_gb, 1), 'profile': profile, "
            "'python': python_version}, sort_keys=True))\n",
            metadata=_cell_tag("preflight"),
        ),
        nbformat.v4.new_code_cell(
            "from google.colab import drive\n"
            "from pathlib import Path\n"
            "drive.mount('/content/drive')\n"
            "DRIVE_ROOT = Path('/content/drive/MyDrive').resolve()\n"
            "INPUT_FOLDER = DRIVE_ROOT.joinpath(*RUN_SPEC['input_folder'].split('/'))\n"
            "print(f'Input: {INPUT_FOLDER}')\n"
            "print(f'Result: {INPUT_FOLDER.with_name(INPUT_FOLDER.name + \"_result\")}')\n",
            metadata=_cell_tag("drive"),
        ),
        nbformat.v4.new_code_cell(
            "import shutil, subprocess\n"
            "from pathlib import Path\n"
            "SOURCE_ROOT = Path('/content/gaussian-splatter-src')\n"
            "if SOURCE_ROOT.exists():\n"
            "    shutil.rmtree(SOURCE_ROOT)\n"
            f"REPOSITORY_URL = {source.repo_url!r}\n"
            f"COMMIT_SHA = {source.commit_sha!r}\n"
            "subprocess.run(['git', 'clone', '--no-checkout', REPOSITORY_URL, "
            "str(SOURCE_ROOT)], check=True)\n"
            "subprocess.run(['git', '-C', str(SOURCE_ROOT), 'checkout', '--detach', "
            "COMMIT_SHA], check=True)\n"
            "actual = subprocess.run(['git', '-C', str(SOURCE_ROOT), 'rev-parse', "
            "'HEAD'], check=True, capture_output=True, text=True).stdout.strip()\n"
            "assert actual == COMMIT_SHA, 'Immutable source checkout mismatch'\n",
            metadata=_cell_tag("checkout"),
        ),
        nbformat.v4.new_code_cell(
            "import subprocess\n"
            "subprocess.run(['bash', 'colab/static_notebook_bootstrap.sh'], "
            "cwd=SOURCE_ROOT, check=True)\n",
            metadata=_cell_tag("bootstrap"),
        ),
        nbformat.v4.new_code_cell(
            "import subprocess, sys\n"
            "subprocess.run([sys.executable, 'scripts/notebook_static_run.py', "
            "'--spec', '/content/run_spec.json'], cwd=SOURCE_ROOT, check=True)\n",
            metadata=_cell_tag("execute"),
        ),
        nbformat.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "receipt = json.loads(Path('/content/run_result.json').read_text(encoding='utf-8'))\n"
            "assert receipt.get('status') == 'success', receipt\n"
            "expected = INPUT_FOLDER.with_name(INPUT_FOLDER.name + '_result').resolve()\n"
            "actual = Path(receipt['final_path']).resolve()\n"
            "assert actual == expected, f'Unexpected result path: {actual}'\n"
            "success = json.loads((actual / '_SUCCESS').read_text(encoding='utf-8'))\n"
            "assert success['run_id'] == receipt['run_id']\n"
            "for required in ('splat.ply', 'preview.png', 'quality_report.json', "
            "'scene_metadata.json'):\n"
            "    assert (actual / required).is_file(), required\n"
            "RESULT_FOLDER = actual\n",
            metadata=_cell_tag("validate"),
        ),
        nbformat.v4.new_code_cell(
            "print(f'Result folder: {RESULT_FOLDER}')\n"
            "print(f'Splat: {RESULT_FOLDER / \"splat.ply\"}')\n"
            "print(f'Preview: {RESULT_FOLDER / \"preview.png\"}')\n"
            "print(f'Quality report: {RESULT_FOLDER / \"quality_report.json\"}')\n"
            "print('Open splat.ply in SuperSplat or the 4DGS Studio viewer.')\n",
            metadata=_cell_tag("summary"),
        ),
    ]
    return cells

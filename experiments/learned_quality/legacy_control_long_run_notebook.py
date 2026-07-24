from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat


REPOSITORY_URL = "https://github.com/mehmettahacumurcu/gaussian-splatter.git"
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def _tag(name: str) -> dict[str, list[str]]:
    return {"tags": [name]}


def build_legacy_control_long_run_notebook(
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
            "# Native 1080p legacy-control 30K local run\n\n"
            "This A100 High-RAM notebook restores the verified 800-frame "
            "lineage once, then runs one continuous native 1920x1080 trainer "
            "through 30K. It preserves raw PLY and resumable optimizer "
            "checkpoints at every 5K boundary under `/content`. Nothing is "
            "published back to Drive, and the runtime stays connected after "
            "the run so the files can be inspected and downloaded.",
            metadata=_tag("title"),
        ),
        nbformat.v4.new_code_cell(
            'INPUT_FOLDER = ""  # @param {type:"string"}\n'
            'RUNTIME_PROFILE = "a100_legacy_control_native_1080p_30k_local"\n',
            metadata=_tag("config"),
        ),
        nbformat.v4.new_code_cell(
            "import json, os, re, shutil, subprocess\n"
            "gpu_line = subprocess.run([\n"
            "    'nvidia-smi', '--query-gpu=name,memory.total',\n"
            "    '--format=csv,noheader,nounits',\n"
            "], check=True, capture_output=True, text=True).stdout.splitlines()[0]\n"
            "gpu_name, memory_mib = (part.strip() for part in gpu_line.rsplit(',', 1))\n"
            "vram_gib = float(memory_mib) / 1024.0\n"
            "is_a100 = re.search(r'\\bA100\\b', gpu_name, flags=re.IGNORECASE) is not None\n"
            "assert is_a100 and vram_gib >= 75.0, (\n"
            "    f'A100 with at least 75 GiB VRAM required; detected '\n"
            "    f'{gpu_name} ({vram_gib:.1f} GiB)'\n"
            ")\n"
            "disk_gib = shutil.disk_usage('/content').free / (1024 ** 3)\n"
            "assert disk_gib >= 120.0, (\n"
            "    f'At least 120 GiB local disk required; detected {disk_gib:.1f}'\n"
            ")\n"
            "host_ram_gib = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / (1024 ** 3)\n"
            "assert host_ram_gib >= 100.0, (\n"
            "    f'At least 100 GiB host RAM required; detected {host_ram_gib:.1f}'\n"
            ")\n"
            "print(json.dumps({'runtime_profile': RUNTIME_PROFILE, 'gpu': gpu_name, "
            "'vram_gib': round(vram_gib, 1), 'host_ram_gib': round(host_ram_gib, 1), "
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
            "assert not raw_folder.endswith((\n"
            "    '_legacy_control_5k_result', '_training_ablation',\n"
            "    '_floor_recovery_diagnostic', '_learned_test_result',\n"
            "    '_learned_test_diagnostics', '_learned_test_cache',\n"
            "))\n"
            "INPUT_PATH = DRIVE_ROOT.joinpath(*folder.parts).resolve()\n"
            "INPUT_PATH.relative_to(DRIVE_ROOT)\n"
            "assert INPUT_PATH.is_dir(), f'Input folder does not exist: {INPUT_PATH}'\n"
            "CACHE_PATH = INPUT_PATH.with_name(INPUT_PATH.name + '_learned_test_cache')\n"
            "REFERENCE_PATH = INPUT_PATH.with_name(INPUT_PATH.name + '_training_ablation')\n"
            "assert CACHE_PATH.is_dir(), 'Verified cache is missing: ' + str(CACHE_PATH)\n"
            "assert (REFERENCE_PATH / '_SUCCESS.json').is_file(), (\n"
            "    'Completed A100 diagnostic is required: ' + str(REFERENCE_PATH)\n"
            ")\n"
            "RUN_SPEC = {\n"
            "    'schema_version': 1,\n"
            "    'input_folder': folder.as_posix(),\n"
            "    'runtime_profile': RUNTIME_PROFILE,\n"
            "}\n"
            "SPEC_PATH = Path('/content/legacy_control_long_spec.json')\n"
            "with SPEC_PATH.open('w', encoding='utf-8') as handle:\n"
            "    json.dump(RUN_SPEC, handle, sort_keys=True, separators=(',', ':'))\n"
            "print(f'Input: {INPUT_PATH}')\n"
            "print(f'Verified cache: {CACHE_PATH}')\n"
            "print(f'A100 reference: {REFERENCE_PATH}')\n"
            "print('Local outputs: /content/legacy_control_1080p_30k/<run-id>')\n",
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
            "import sys, time\n"
            "sys.path.insert(0, str(SOURCE_ROOT))\n"
            "from huggingface_hub import snapshot_download\n"
            "from experiments.learned_quality.dependencies import (\n"
            "    install_learned_environment, materialize_pinned_assets,\n"
            ")\n"
            "def download_with_backoff(**kwargs):\n"
            "    kwargs['max_workers'] = 1\n"
            "    for attempt in range(6):\n"
            "        try:\n"
            "            return snapshot_download(**kwargs)\n"
            "        except Exception as error:\n"
            "            if attempt == 5 or '429' not in str(error):\n"
            "                raise\n"
            "            delay = 240 + 60 * attempt\n"
            "            print(f'Hugging Face rate limited; retrying in {delay}s...')\n"
            "            time.sleep(delay)\n"
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
            "    LEARNED_ENV, downloader=download_with_backoff,\n"
            "    resolve_revision=resolved_revision,\n"
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
            "environment['PYTHONUNBUFFERED'] = '1'\n"
            "subprocess.run(\n"
            "    [\n"
            "        '/content/learned-env/bin/python', '-u', '-m',\n"
            '        "scripts.learned_quality_legacy_control_long_run",\n'
            '        "--spec", str(SPEC_PATH),\n'
            '        "--source-revision", COMMIT_SHA,\n'
            '        "--model-manifest", str(MODEL_MANIFEST.path),\n'
            "    ],\n"
            "    cwd=SOURCE_ROOT,\n"
            "    env=environment,\n"
            "    check=True,\n"
            ")\n"
            "print('The local run finished. This Colab session remains connected.')\n",
            metadata=_tag("execute"),
        ),
        nbformat.v4.new_code_cell(
            "import subprocess\n"
            "from pathlib import Path\n"
            "root = Path('/content/legacy_control_1080p_30k')\n"
            "runs = sorted(\n"
            "    (path for path in root.iterdir() if path.is_dir()),\n"
            "    key=lambda path: path.stat().st_mtime,\n"
            "    reverse=True,\n"
            ") if root.is_dir() else []\n"
            "assert runs, 'No local run exists yet.'\n"
            "active = runs[0]\n"
            "print(f'LOCAL_RUN_ROOT={active}')\n"
            "metrics = sorted(active.rglob('metrics.jsonl'), key=lambda p: p.stat().st_mtime)\n"
            "if metrics:\n"
            "    print('\\nLatest training metrics:')\n"
            "    print('\\n'.join(metrics[-1].read_text(encoding='utf-8').splitlines()[-3:]))\n"
            "print('\\nSaved boundaries:')\n"
            "for iteration in (5000, 10000, 15000, 20000, 25000, 30000):\n"
            "    ply = active / 'ply' / f'legacy_control_{iteration:06d}.ply'\n"
            "    checkpoint = active / 'checkpoints' / f'legacy_control_{iteration:06d}.pt'\n"
            "    print(f'{iteration:6d}: PLY={ply.is_file()} CHECKPOINT={checkpoint.is_file()}')\n"
            "subprocess.run([\n"
            "    'nvidia-smi',\n"
            "    '--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw',\n"
            "    '--format=csv,noheader',\n"
            "], check=False)\n"
            "print('\\nExpected final files include legacy_control_005000.ply and '\n"
            "      'legacy_control_030000.pt.')\n",
            metadata=_tag("monitor"),
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
                "id": "4dgs-studio.legacy-control-long-run-notebook",
                "version": 1,
                "commit_sha": commit_sha,
            },
        },
    )
    nbformat.validate(notebook)
    return notebook


def write_legacy_control_long_run_notebook(path: Path, commit_sha: str) -> Path:
    notebook = build_legacy_control_long_run_notebook(commit_sha=commit_sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path, version=4)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the native-1080p local 30K legacy-control notebook"
    )
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    write_legacy_control_long_run_notebook(args.output, args.commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

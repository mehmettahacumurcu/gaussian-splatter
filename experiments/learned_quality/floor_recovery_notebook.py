from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat


REPOSITORY_URL = "https://github.com/mehmettahacumurcu/gaussian-splatter.git"
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def _tag(name: str) -> dict[str, list[str]]:
    return {"tags": [name]}


def build_floor_recovery_notebook(
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
            "# Automatic floor-hole recovery diagnostic\n\n"
            "This is the final bounded floor-recovery test. It requires the passing "
            "CPU track audit, verified final pre-training cache, and the completed "
            "A100 training diagnostic for the same input. The notebook restores that "
            "lineage once, fits a reliable floor plane, identifies empty floor cells, "
            "adds at most 150,000 low-opacity motion/sky-filtered seeds, and runs one "
            "deterministic 5K legacy-density arm. It publishes a diagnostic candidate "
            "only when available. It never starts a full 120K run and never overwrites "
            "the learned or legacy production result.",
            metadata=_tag("title"),
        ),
        nbformat.v4.new_code_cell(
            'INPUT_FOLDER = ""  # @param {type:"string"}\n'
            'RUNTIME_PROFILE = "a100_floor_recovery"\n',
            metadata=_tag("config"),
        ),
        nbformat.v4.new_code_cell(
            "import json, re, shutil, subprocess\n"
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
            "assert disk_gib >= 80.0, f'At least 80 GiB local disk required; detected {disk_gib:.1f}'\n"
            "print(json.dumps({'runtime_profile': RUNTIME_PROFILE, 'gpu': gpu_name, "
            "'vram_gib': round(vram_gib, 1), 'disk_free_gib': round(disk_gib, 1)}, "
            "sort_keys=True))\n",
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
            "    '_floor_recovery_diagnostic', '_training_ablation',\n"
            "    '_learned_test_result', '_learned_test_diagnostics',\n"
            "    '_learned_test_cache',\n"
            "))\n"
            "INPUT_PATH = DRIVE_ROOT.joinpath(*folder.parts).resolve()\n"
            "INPUT_PATH.relative_to(DRIVE_ROOT)\n"
            "assert INPUT_PATH.is_dir(), f'Input folder does not exist: {INPUT_PATH}'\n"
            "CACHE_PATH = INPUT_PATH.with_name(INPUT_PATH.name + '_learned_test_cache')\n"
            "REFERENCE_PATH = INPUT_PATH.with_name(INPUT_PATH.name + '_training_ablation')\n"
            "RESULT_PATH = INPUT_PATH.with_name(INPUT_PATH.name + '_floor_recovery_diagnostic')\n"
            "assert (REFERENCE_PATH / '_SUCCESS.json').is_file(), (\n"
            "    'Completed A100 training diagnostic is required: ' + str(REFERENCE_PATH)\n"
            ")\n"
            "RUN_SPEC = {'schema_version': 1, 'input_folder': folder.as_posix(), "
            "'runtime_profile': RUNTIME_PROFILE, "
            "'publish': {'replace_owned_result': True}}\n"
            "SPEC_PATH = Path('/content/learned_floor_recovery_spec.json')\n"
            "with SPEC_PATH.open('w', encoding='utf-8') as handle:\n"
            "    json.dump(RUN_SPEC, handle, sort_keys=True, separators=(',', ':'))\n"
            "print(f'Input: {INPUT_PATH}')\n"
            "print(f'Verified cache: {CACHE_PATH}')\n"
            "print(f'A100 legacy reference: {REFERENCE_PATH}')\n"
            "print(f'Floor diagnostic result: {RESULT_PATH}')\n",
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
            "    LEARNED_ENV, downloader=download_with_backoff, "
            "resolve_revision=resolved_revision,\n"
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
            "import json, os, subprocess, traceback\n"
            "from pathlib import Path\n"
            "from google.colab import drive, runtime\n"
            "environment = dict(os.environ)\n"
            "environment['LEARNED_MODEL_MANIFEST'] = str(MODEL_MANIFEST.path)\n"
            "environment['PYTHONUNBUFFERED'] = '1'\n"
            "failure = None\n"
            "try:\n"
            "    completed = subprocess.run(\n"
            "        [\n"
            '            "/content/learned-env/bin/python",\n'
            '            "-u", "-m", "scripts.learned_quality_floor_recovery_run",\n'
            '            "--spec", "/content/learned_floor_recovery_spec.json",\n'
            '            "--source-revision", COMMIT_SHA,\n'
            '            "--model-manifest", str(MODEL_MANIFEST.path),\n'
            "        ],\n"
            "        cwd=SOURCE_ROOT, env=environment, check=False,\n"
            "    )\n"
            "    receipt_path = Path('/content/learned_floor_recovery_result.json')\n"
            "    if not receipt_path.is_file():\n"
            "        raise RuntimeError('Floor recovery ended without a run receipt')\n"
            "    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))\n"
            "    if completed.returncode != 0 or receipt.get('status') not in {'success', 'rejected'}:\n"
            "        raise RuntimeError(f'Floor recovery infrastructure failed: {receipt}')\n"
            "    actual = Path(receipt['final_path']).resolve()\n"
            "    if actual != RESULT_PATH.resolve():\n"
            "        raise RuntimeError(f'Unexpected floor diagnostic path: {actual}')\n"
            "    success = json.loads((actual / '_SUCCESS.json').read_text(encoding='utf-8'))\n"
            "    if success.get('run_id') != receipt.get('run_id'):\n"
            "        raise RuntimeError('Floor diagnostic marker does not match this run')\n"
            "    required = (\n"
            "        'decision.json', 'comparison.json', 'diagnostic_summary.md',\n"
            "        'legacy_reference/diagnostic_matrix.json',\n"
            "    )\n"
            "    missing = [name for name in required if not (actual / name).is_file()]\n"
            "    if missing:\n"
            "        raise RuntimeError(f'Published floor diagnostic is incomplete: {missing}')\n"
            "    decision = receipt.get('decision') or {}\n"
            "    print(f'Floor diagnostic: {actual}')\n"
            "    print(f'Decision: {decision.get(\"reason\")}')\n"
            "    print(f'Failures: {decision.get(\"failures\", [])}')\n"
            "    candidate = actual / 'candidate.ply'\n"
            "    if candidate.is_file():\n"
            "        print(f'Bounded 5K candidate PLY: {candidate}')\n"
            "    print('No full 120K training was started.')\n"
            "except BaseException as exc:\n"
            "    failure = exc\n"
            "    print(f'Run ended with {type(exc).__name__}: {exc}')\n"
            "    traceback.print_exception(type(exc), exc, exc.__traceback__)\n"
            "finally:\n"
            "    print('Flushing outstanding Google Drive writes...')\n"
            "    try:\n"
            "        drive.flush_and_unmount()\n"
            "    except BaseException as flush_error:\n"
            "        print(f'Drive flush/unmount failed: {flush_error}')\n"
            "    print('Releasing the Colab runtime now.')\n"
            "    try:\n"
            "        runtime.unassign()\n"
            "    except BaseException as release_error:\n"
            "        print(f'Runtime release request failed: {release_error}')\n"
            "        if failure is None:\n"
            "            failure = release_error\n"
            "if failure is not None:\n"
            "    raise failure\n",
            metadata=_tag("execute"),
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
                "id": "4dgs-studio.floor-recovery-notebook",
                "version": 1,
                "commit_sha": commit_sha,
            },
        },
    )
    nbformat.validate(notebook)
    return notebook


def write_floor_recovery_notebook(path: Path, commit_sha: str) -> Path:
    notebook = build_floor_recovery_notebook(commit_sha=commit_sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path, version=4)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the floor recovery notebook")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    write_floor_recovery_notebook(args.output, args.commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

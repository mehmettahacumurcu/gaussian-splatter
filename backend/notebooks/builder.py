from __future__ import annotations

import re
import unicodedata

import nbformat

from .drive_paths import normalize_input_folder
from .models import StaticNotebookRunSpec
from .source import NotebookSource, REPOSITORY_URL
from .static_template import build_static_cells

_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def build_static_notebook(
    spec: StaticNotebookRunSpec,
    *,
    source: NotebookSource,
) -> nbformat.NotebookNode:
    if _FULL_SHA.fullmatch(source.commit_sha) is None:
        raise ValueError("Notebook source requires a full 40-character commit SHA")
    if source.repo_url != REPOSITORY_URL:
        raise ValueError("Notebook source repository is not approved")
    if source.generator_id != "4dgs-studio.static-notebook":
        raise ValueError("Notebook source generator_id is unsupported")
    if source.generator_version != 1:
        raise ValueError("Notebook source generator_version is unsupported")
    notebook = nbformat.v4.new_notebook(
        cells=build_static_cells(spec, source),
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "generator": {
                "id": source.generator_id,
                "version": source.generator_version,
                "commit_sha": source.commit_sha,
            },
        },
    )
    nbformat.validate(notebook)
    return notebook


def serialize_notebook(notebook: nbformat.NotebookNode) -> bytes:
    nbformat.validate(notebook)
    return nbformat.writes(notebook, version=4).encode("utf-8")


def notebook_download_filename(input_folder: str) -> str:
    canonical = normalize_input_folder(input_folder)
    leaf = canonical.rsplit("/", 1)[-1]
    ascii_leaf = unicodedata.normalize("NFKD", leaf).encode("ascii", "ignore").decode()
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_leaf).strip("_-") or "scene"
    return f"{safe[:80]}_static_splat.ipynb"

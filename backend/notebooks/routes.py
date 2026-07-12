from __future__ import annotations

from fastapi import APIRouter, Response

from .builder import (
    build_static_notebook,
    notebook_download_filename,
    serialize_notebook,
)
from .models import StaticNotebookRunSpec
from .presets import get_static_preset_manifest
from .source import resolve_notebook_source

static_notebook_router = APIRouter(prefix="/notebooks/static", tags=["notebooks"])


@static_notebook_router.get("/presets")
def static_notebook_presets():
    return get_static_preset_manifest()


@static_notebook_router.post("")
def generate_static_notebook(spec: StaticNotebookRunSpec) -> Response:
    source = resolve_notebook_source()
    content = serialize_notebook(build_static_notebook(spec, source=source))
    filename = notebook_download_filename(spec.input_folder)
    return Response(
        content=content,
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

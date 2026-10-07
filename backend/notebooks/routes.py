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
from .pipeline_library import (
    PRESETS, PipelineNotebookSpec, build_pipeline_notebook, pipeline_filename,
    resolve_embedded_notebook_source,
)
from .spirula_preprocess import (
    PreprocessSpec, build_preprocess_notebook, preprocess_catalog, preprocess_filename,
)

static_notebook_router = APIRouter(prefix="/notebooks/static", tags=["notebooks"])


@static_notebook_router.get('/preprocess')
def spirula_preprocess_catalog():
    return preprocess_catalog()


@static_notebook_router.post('/preprocess/validate')
def validate_spirula_preprocess_preset(spec: PreprocessSpec):
    return spec.model_dump()


@static_notebook_router.post('/preprocess')
def generate_spirula_preprocess_notebook(spec: PreprocessSpec) -> Response:
    notebook = build_preprocess_notebook(spec, source=resolve_notebook_source())
    return Response(content=serialize_notebook(notebook), media_type='application/x-ipynb+json',
                    headers={'Content-Disposition': f'attachment; filename="{preprocess_filename(spec)}"'})


@static_notebook_router.get("/pipelines")
def pipeline_notebook_presets():
    return {"template_version": 1, "presets": PRESETS}


@static_notebook_router.post("/pipeline")
def generate_pipeline_notebook(spec: PipelineNotebookSpec) -> Response:
    source = resolve_embedded_notebook_source() if spec.pipeline == 'text_to_splat' else resolve_notebook_source()
    notebook = build_pipeline_notebook(spec, source=source)
    return Response(
        content=serialize_notebook(notebook),
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f'attachment; filename="{pipeline_filename(spec)}"'},
    )


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

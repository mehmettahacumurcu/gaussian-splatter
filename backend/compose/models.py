"""Pydantic models for the scene composer (SceneDoc v1)."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Safe for use in file names: no separators, no leading dot.
ID_PATTERN = r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}$"
# Reference to an asset: uploaded ids plus ``scene__<name>`` pipeline ids, whose
# names may contain Unicode letters. Still no leading dot and no separators.
ASSET_REF_PATTERN = r"^[\w-][\w.-]{0,79}$"

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # three.js order: x, y, z, w


def _unit_quat(q: Quat) -> Quat:
    norm = math.sqrt(sum(c * c for c in q))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"quaternion must be unit length (norm={norm:.4f})")
    return tuple(c / norm for c in q)  # type: ignore[return-value]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Transform(_Strict):
    position: Vec3 = (0.0, 0.0, 0.0)
    quaternion: Quat = (0.0, 0.0, 0.0, 1.0)
    scale: float = Field(1.0, gt=0)

    @field_validator("quaternion")
    @classmethod
    def _unit(cls, v: Quat) -> Quat:
        return _unit_quat(v)

    def is_identity(self, tol: float = 1e-6) -> bool:
        return (
            all(abs(c) <= tol for c in self.position)
            and all(abs(a - b) <= tol for a, b in zip(self.quaternion, (0.0, 0.0, 0.0, 1.0)))
            and abs(self.scale - 1.0) <= tol
        )


class CropBox(_Strict):
    center: Vec3 = (0.0, 0.0, 0.0)
    halfSize: Vec3
    quaternion: Quat = (0.0, 0.0, 0.0, 1.0)

    @field_validator("quaternion")
    @classmethod
    def _unit(cls, v: Quat) -> Quat:
        return _unit_quat(v)

    @field_validator("halfSize")
    @classmethod
    def _positive(cls, v: Vec3) -> Vec3:
        if any(c <= 0 for c in v):
            raise ValueError("crop halfSize must be > 0 on every axis")
        return v


class ColorAdjust(_Strict):
    exposure: float = Field(0.0, ge=-3.0, le=3.0)
    tint: Vec3 = (1.0, 1.0, 1.0)
    saturation: float = Field(1.0, ge=0.0, le=2.0)

    @field_validator("tint")
    @classmethod
    def _tint_range(cls, v: Vec3) -> Vec3:
        if any(c < 0 or c > 2 for c in v):
            raise ValueError("tint channels must be within 0..2")
        return v


class SceneObject(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    kind: Literal["splat", "mesh"]
    asset: str = Field(pattern=ASSET_REF_PATTERN)
    name: str = Field(min_length=1, max_length=128)
    role: Literal["base", "object"] = "object"
    visible: bool = True
    transform: Transform = Field(default_factory=Transform)
    crop: CropBox | None = None
    color: ColorAdjust | None = None

    @model_validator(mode="after")
    def _mesh_has_no_splat_edits(self) -> "SceneObject":
        if self.kind == "mesh" and (self.crop is not None or self.color is not None):
            raise ValueError(f"object {self.id}: crop and colour are only supported on splats")
        return self


class SceneDoc(_Strict):
    version: Literal[1] = 1
    id: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1, max_length=128)
    viewUp: Literal["y", "-y"] = "y"
    # Measured world "up" of the base capture (unit vector); None = assume +Y.
    up: Vec3 | None = None
    objects: list[SceneObject] = Field(default_factory=list)

    @field_validator("up")
    @classmethod
    def _unit_up(cls, v: Vec3 | None) -> Vec3 | None:
        if v is None:
            return None
        norm = math.sqrt(sum(c * c for c in v))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
            raise ValueError(f"up must be a unit vector (norm={norm:.4f})")
        return tuple(c / norm for c in v)  # type: ignore[return-value]

    @model_validator(mode="after")
    def _check_objects(self) -> "SceneDoc":
        ids = [o.id for o in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object ids must be unique")
        bases = [o for o in self.objects if o.role == "base"]
        if len(bases) != 1:
            raise ValueError("scene must have exactly one base object")
        base = bases[0]
        if base.kind != "splat":
            raise ValueError("base object must be a splat")
        if not base.transform.is_identity():
            raise ValueError("base object transform must be identity")
        return self


class Asset(_Strict):
    id: str = Field(pattern=ASSET_REF_PATTERN)
    kind: Literal["splat", "mesh"]
    name: str
    size_bytes: int
    source: Literal["upload", "pipeline"]
    created_ts: float


class SceneSummary(BaseModel):
    id: str
    name: str
    updated_ts: float
    object_count: int


class CreateSceneRequest(_Strict):
    name: str = Field(min_length=1, max_length=128)
    base_asset: str = Field(pattern=ASSET_REF_PATTERN)


class ExportResponse(BaseModel):
    job_id: str

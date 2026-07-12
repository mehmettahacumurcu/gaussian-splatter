"""Conservative, regression-guarded polish for static INRIA 3DGS PLY files."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .contracts import (
    PolishPolicy,
    PolishReport,
    ReconstructionBundle,
    RenderViewMetric,
)
from .sources import _atomic_promote_no_replace

if TYPE_CHECKING:
    from backend.preprocess.frame_alignment import RegisteredFrame


RenderEvaluator = Callable[
    [Path, tuple["RegisteredFrame", ...]],
    tuple[RenderViewMetric, ...],
]


_REASON_ORDER = (
    "candidate_empty",
    "candidate_validation",
    "removed_fraction",
    "opacity_mass_loss",
    "render_evaluation",
    "render_metrics_incomplete",
    "render_metrics_nonfinite",
    "mean_psnr_drop",
    "mean_ssim_drop",
    "single_view_psnr_drop",
)
_REASON_RANK = {reason: index for index, reason in enumerate(_REASON_ORDER)}
_STANDARD_VERTEX_PREFIX = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
)
_STANDARD_VERTEX_SUFFIX = (
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


@dataclass(frozen=True)
class _ValidatedStaticPly:
    path: Path
    vertices: np.ndarray
    sh_degree: int

    @property
    def count(self) -> int:
        return int(len(self.vertices))


@dataclass(frozen=True)
class _FileBinding:
    path: Path
    device: int
    inode: int
    size: int
    modified_ns: int
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_file_binding(path: Path) -> _FileBinding:
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"integrity requires a regular non-symlink file: {path}")
    digest = _sha256_file(path)
    after = os.lstat(path)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or not stat.S_ISREG(after.st_mode):
        raise RuntimeError(f"file identity changed while binding integrity: {path}")
    return _FileBinding(
        path=path,
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        modified_ns=after.st_mtime_ns,
        sha256=digest,
    )


def _require_unchanged(binding: _FileBinding) -> None:
    current = _capture_file_binding(binding.path)
    if current != binding:
        raise RuntimeError(f"file integrity changed during evaluation: {binding.path}")


def _require_all_unchanged(bindings: Sequence[_FileBinding]) -> None:
    for binding in bindings:
        _require_unchanged(binding)


def _canonical_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    unknown = sorted(set(reasons).difference(_REASON_RANK))
    if unknown:
        raise ValueError(f"unknown polish rejection reason(s): {', '.join(unknown)}")
    return tuple(reason for reason in _REASON_ORDER if reason in reasons)


def _exceeds(value: float, limit: float) -> bool:
    return value > limit and not math.isclose(
        value,
        limit,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _metric_payload(metric: object) -> dict[str, object]:
    name = getattr(metric, "image_name", None)
    psnr = getattr(metric, "psnr_db", None)
    ssim = getattr(metric, "ssim", None)
    return {
        "image_name": name if isinstance(name, str) else None,
        "psnr_db": float(psnr)
        if isinstance(psnr, (int, float))
        and not isinstance(psnr, bool)
        and math.isfinite(float(psnr))
        else None,
        "ssim": float(ssim)
        if isinstance(ssim, (int, float))
        and not isinstance(ssim, bool)
        and math.isfinite(float(ssim))
        else None,
    }


def _metrics_are_complete(
    raw_metrics: Sequence[RenderViewMetric],
    candidate_metrics: Sequence[RenderViewMetric],
) -> bool:
    if not raw_metrics or len(raw_metrics) != len(candidate_metrics):
        return False
    if len(raw_metrics) > 12:
        return False
    raw_names = tuple(getattr(metric, "image_name", None) for metric in raw_metrics)
    candidate_names = tuple(
        getattr(metric, "image_name", None) for metric in candidate_metrics
    )
    return (
        raw_names == candidate_names
        and all(isinstance(name, str) and bool(name.strip()) for name in raw_names)
        and len(set(raw_names)) == len(raw_names)
    )


def _metrics_are_finite(
    raw_metrics: Sequence[RenderViewMetric],
    candidate_metrics: Sequence[RenderViewMetric],
) -> bool:
    for metric in (*raw_metrics, *candidate_metrics):
        psnr = getattr(metric, "psnr_db", None)
        ssim = getattr(metric, "ssim", None)
        if (
            isinstance(psnr, bool)
            or not isinstance(psnr, (int, float))
            or not math.isfinite(float(psnr))
            or isinstance(ssim, bool)
            or not isinstance(ssim, (int, float))
            or not math.isfinite(float(ssim))
            or not -1.0 <= float(ssim) <= 1.0
        ):
            return False
    return True


def _build_polish_report(
    *,
    raw_path: Path,
    candidate_path: Path | None,
    original_count: int,
    kept_count: int,
    raw_opacity_mass: float,
    candidate_opacity_mass: float,
    raw_metrics: Sequence[RenderViewMetric],
    candidate_metrics: Sequence[RenderViewMetric],
    policy: PolishPolicy,
    fallback_reasons: Sequence[str] = (),
) -> PolishReport:
    """Build a deterministic decision report from geometry and render metrics."""

    if isinstance(raw_opacity_mass, bool) or not isinstance(
        raw_opacity_mass, (int, float)
    ):
        raise TypeError("raw_opacity_mass must be a finite number")
    if isinstance(candidate_opacity_mass, bool) or not isinstance(
        candidate_opacity_mass, (int, float)
    ):
        raise TypeError("candidate_opacity_mass must be a finite number")
    if not math.isfinite(float(raw_opacity_mass)) or raw_opacity_mass <= 0.0:
        raise ValueError("raw_opacity_mass must be finite and positive")
    if not math.isfinite(float(candidate_opacity_mass)) or candidate_opacity_mass < 0.0:
        raise ValueError("candidate_opacity_mass must be finite and non-negative")

    removed_fraction = (original_count - kept_count) / original_count
    opacity_mass_loss = max(
        0.0,
        min(1.0, 1.0 - candidate_opacity_mass / raw_opacity_mass),
    )
    reasons = list(fallback_reasons)
    if kept_count == 0:
        reasons.append("candidate_empty")
    if _exceeds(removed_fraction, policy.max_removed_fraction):
        reasons.append("removed_fraction")
    if _exceeds(opacity_mass_loss, policy.max_opacity_mass_loss):
        reasons.append("opacity_mass_loss")

    complete = _metrics_are_complete(raw_metrics, candidate_metrics)
    finite = complete and _metrics_are_finite(raw_metrics, candidate_metrics)
    skip_render_decision = any(
        reason
        in {
            "candidate_empty",
            "candidate_validation",
            "render_evaluation",
            "render_metrics_incomplete",
            "render_metrics_nonfinite",
        }
        for reason in reasons
    )
    if not skip_render_decision:
        if not complete:
            reasons.append("render_metrics_incomplete")
        elif not finite:
            reasons.append("render_metrics_nonfinite")

    raw_payload = [_metric_payload(metric) for metric in raw_metrics]
    candidate_payload = [_metric_payload(metric) for metric in candidate_metrics]
    views: list[dict[str, object]] = []
    mean_psnr_drop: float | None = None
    mean_ssim_drop: float | None = None
    max_single_psnr_drop: float | None = None

    if complete and finite and not skip_render_decision:
        psnr_drops: list[float] = []
        ssim_drops: list[float] = []
        for raw_metric, candidate_metric in zip(
            raw_metrics, candidate_metrics, strict=True
        ):
            psnr_drop = float(raw_metric.psnr_db - candidate_metric.psnr_db)
            ssim_drop = float(raw_metric.ssim - candidate_metric.ssim)
            psnr_drops.append(psnr_drop)
            ssim_drops.append(ssim_drop)
            views.append(
                {
                    "image_name": raw_metric.image_name,
                    "raw_psnr_db": float(raw_metric.psnr_db),
                    "candidate_psnr_db": float(candidate_metric.psnr_db),
                    "psnr_drop_db": psnr_drop,
                    "raw_ssim": float(raw_metric.ssim),
                    "candidate_ssim": float(candidate_metric.ssim),
                    "ssim_drop": ssim_drop,
                }
            )
        mean_psnr_drop = float(sum(psnr_drops) / len(psnr_drops))
        mean_ssim_drop = float(sum(ssim_drops) / len(ssim_drops))
        max_single_psnr_drop = float(max(psnr_drops))
        if _exceeds(mean_psnr_drop, policy.max_mean_psnr_drop_db):
            reasons.append("mean_psnr_drop")
        if _exceeds(mean_ssim_drop, policy.max_mean_ssim_drop):
            reasons.append("mean_ssim_drop")
        if _exceeds(
            max_single_psnr_drop,
            policy.max_single_view_psnr_drop_db,
        ):
            reasons.append("single_view_psnr_drop")

    ordered_reasons = _canonical_reasons(reasons)
    accepted = not ordered_reasons and candidate_path is not None
    if not accepted and not ordered_reasons:
        ordered_reasons = ("candidate_validation",)

    return PolishReport(
        accepted=accepted,
        raw_path=Path(raw_path),
        candidate_path=Path(candidate_path) if candidate_path is not None else None,
        selected_path=Path(candidate_path) if accepted else Path(raw_path),
        original_count=original_count,
        kept_count=kept_count,
        opacity_mass_loss=opacity_mass_loss,
        render_metrics={
            "kind": "registered_training_views",
            "sampled_view_count": len(raw_metrics) if complete else 0,
            "raw": raw_payload,
            "candidate": candidate_payload,
            "views": views,
            "mean_psnr_drop_db": mean_psnr_drop,
            "mean_ssim_drop": mean_ssim_drop,
            "max_single_view_psnr_drop_db": max_single_psnr_drop,
        },
        reasons=ordered_reasons,
    )


def validate_static_ply(path: str | Path) -> _ValidatedStaticPly:
    """Strictly validate and load one degree-zero-through-three INRIA PLY."""

    ply_path = Path(path)
    if ply_path.is_symlink() or not ply_path.is_file():
        raise ValueError(f"PLY must be a regular non-symlink file: {ply_path}")

    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("PLY validation requires the 'plyfile' package") from exc

    try:
        ply = PlyData.read(str(ply_path), mmap=False)
    except Exception as exc:
        raise ValueError(f"invalid PLY file: {ply_path}") from exc

    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise ValueError("static PLY must contain exactly one vertex element")
    vertices = ply.elements[0].data
    if len(vertices) == 0:
        raise ValueError("static PLY vertex element must not be empty")
    names = tuple(vertices.dtype.names or ())
    suffix_count = len(_STANDARD_VERTEX_SUFFIX)
    if (
        names[: len(_STANDARD_VERTEX_PREFIX)] != _STANDARD_VERTEX_PREFIX
        or names[-suffix_count:] != _STANDARD_VERTEX_SUFFIX
    ):
        raise ValueError("static PLY must use the exact standard INRIA vertex layout")
    rest_names = names[len(_STANDARD_VERTEX_PREFIX) : -suffix_count]
    rest_count_to_degree = {0: 0, 9: 1, 24: 2, 45: 3}
    sh_degree = rest_count_to_degree.get(len(rest_names))
    expected_rest = tuple(f"f_rest_{index}" for index in range(len(rest_names)))
    if sh_degree is None or rest_names != expected_rest:
        raise ValueError(
            "f_rest fields must be contiguous spherical-harmonic coefficients "
            "for SH degree 0, 1, 2, or 3"
        )
    for name in names:
        dtype = vertices.dtype.fields[name][0]
        if dtype.kind != "f" or dtype.itemsize != 4:
            raise ValueError(f"PLY field {name!r} must be a float32 scalar")
        if not np.isfinite(np.asarray(vertices[name])).all():
            raise ValueError("PLY vertex fields must all be finite")

    log_scales = np.column_stack([vertices[f"scale_{index}"] for index in range(3)])
    with np.errstate(over="ignore", invalid="ignore"):
        activated_scales = np.exp(log_scales)
    if not np.isfinite(activated_scales).all() or np.any(activated_scales <= 0.0):
        raise ValueError("activated PLY scales must be finite and positive")

    quaternions = np.column_stack(
        [vertices[f"rot_{index}"] for index in range(4)]
    ).astype(np.float64, copy=False)
    quaternion_norms = np.linalg.norm(quaternions, axis=1)
    if not np.isfinite(quaternion_norms).all() or np.any(
        quaternion_norms <= np.finfo(np.float32).tiny
    ):
        raise ValueError("PLY rotations must contain finite nonzero quaternions")

    return _ValidatedStaticPly(
        path=ply_path,
        vertices=np.array(vertices, copy=True),
        sh_degree=sh_degree,
    )


def _renderer_arrays(
    validated: _ValidatedStaticPly,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = validated.vertices
    means = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
    log_scales = np.column_stack([vertices[f"scale_{index}"] for index in range(3)])
    scales = np.exp(log_scales)
    quaternions = np.column_stack([vertices[f"rot_{index}"] for index in range(4)])
    quaternions = quaternions / np.linalg.norm(
        quaternions.astype(np.float64),
        axis=1,
        keepdims=True,
    )
    opacities = _sigmoid(vertices["opacity"])
    dc = np.column_stack([vertices[f"f_dc_{index}"] for index in range(3)])[:, None, :]
    rest_basis_count = (validated.sh_degree + 1) ** 2 - 1
    if rest_basis_count:
        flat_rest = np.column_stack(
            [vertices[f"f_rest_{index}"] for index in range(rest_basis_count * 3)]
        )
        rest = flat_rest.reshape(len(vertices), 3, rest_basis_count).transpose(
            0,
            2,
            1,
        )
        colors = np.concatenate((dc, rest), axis=1)
    else:
        colors = dc
    return means, quaternions, scales, opacities, colors


def evaluate_static_ply_views(
    ply_path: str | Path,
    views: tuple["RegisteredFrame", ...],
    *,
    device: str = "cuda",
) -> tuple[RenderViewMetric, ...]:
    """Render one PLY against exact registered training views."""

    import torch
    from PIL import Image

    from backend.eval.nvs_eval import _psnr, _scale_K, _ssim
    from backend.model.renderer import render_view

    validated = validate_static_ply(ply_path)
    arrays = _renderer_arrays(validated)

    def tensor(value: np.ndarray) -> torch.Tensor:
        contiguous = np.ascontiguousarray(value, dtype=np.float32)
        return torch.from_numpy(contiguous).to(device)

    means, quaternions, scales, opacities, colors = map(tensor, arrays)
    metrics: list[RenderViewMetric] = []
    with torch.no_grad():
        for view in views:
            try:
                with Image.open(view.image_path) as opened:
                    ground_truth_array = np.array(
                        opened.convert("RGB"),
                        dtype=np.float32,
                        copy=True,
                    ) / np.float32(255.0)
            except Exception as exc:
                raise ValueError(
                    f"registered ground truth cannot be decoded: {view.image_name}"
                ) from exc
            height, width = ground_truth_array.shape[:2]
            ground_truth = tensor(ground_truth_array)
            intrinsic = tensor(np.asarray(view.K, dtype=np.float32))
            intrinsic = _scale_K(intrinsic, width, height, width, height)
            world_to_camera = tensor(np.asarray(view.w2c, dtype=np.float32))
            rendered, _, _ = render_view(
                means=means,
                quats=quaternions,
                scales=scales,
                opacities=opacities,
                colors=colors,
                K=intrinsic,
                w2c=world_to_camera,
                width=width,
                height=height,
                sh_degree=validated.sh_degree,
            )
            rendered = rendered.clamp(0.0, 1.0)
            metrics.append(
                RenderViewMetric(
                    image_name=view.image_name,
                    psnr_db=_psnr(rendered, ground_truth),
                    ssim=_ssim(rendered, ground_truth),
                )
            )
    return tuple(metrics)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative_exp = np.exp(logits[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def _opacity_mass(vertices: np.ndarray) -> float:
    mass = float(np.sum(_sigmoid(vertices["opacity"]), dtype=np.float64))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("static PLY must have finite positive opacity mass")
    return mass


def _write_normalized_candidate(
    raw: _ValidatedStaticPly,
    candidate_path: Path,
    keep_mask: np.ndarray,
    fade_factors: np.ndarray,
) -> _ValidatedStaticPly:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("PLY polish requires the 'plyfile' package") from exc

    vertices = np.array(raw.vertices[keep_mask], copy=True)
    quaternion_fields = [f"rot_{index}" for index in range(4)]
    quaternions = np.column_stack([vertices[name] for name in quaternion_fields])
    norms = np.linalg.norm(quaternions.astype(np.float64), axis=1, keepdims=True)
    normalized = quaternions / norms
    for index, name in enumerate(quaternion_fields):
        vertices[name] = normalized[:, index]

    alpha = _sigmoid(vertices["opacity"]) * fade_factors[keep_mask]
    minimum_alpha = float(np.finfo(np.float32).tiny)
    maximum_alpha = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
    alpha = np.clip(alpha, minimum_alpha, maximum_alpha)
    vertices["opacity"] = np.log(alpha / (1.0 - alpha))

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    staged = candidate_path.with_name(f"{candidate_path.name}.tmp-{uuid.uuid4().hex}")
    if os.path.lexists(staged):
        raise FileExistsError(f"candidate staging path already exists: {staged}")
    try:
        PlyData([PlyElement.describe(vertices, "vertex")]).write(str(staged))
        validate_static_ply(staged)
        _atomic_promote_no_replace(staged, candidate_path)
    finally:
        if staged.is_symlink() or staged.is_file():
            staged.unlink(missing_ok=True)
    return validate_static_ply(candidate_path)


def _visible_in_any_frustum(
    vertices: np.ndarray,
    views: tuple["RegisteredFrame", ...],
) -> np.ndarray:
    from PIL import Image

    positions = np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(
        np.float64, copy=False
    )
    visible = np.zeros(len(positions), dtype=bool)
    chunk_size = 100_000
    for view in views:
        intrinsic = np.asarray(view.K, dtype=np.float64)
        world_to_camera = np.asarray(view.w2c, dtype=np.float64)
        if (
            intrinsic.shape != (3, 3)
            or world_to_camera.shape != (4, 4)
            or not np.isfinite(intrinsic).all()
            or not np.isfinite(world_to_camera).all()
        ):
            raise ValueError(f"registered camera is invalid: {view.image_name}")
        try:
            with Image.open(view.image_path) as image:
                width, height = image.size
        except Exception as exc:
            raise ValueError(
                f"registered frame cannot be decoded: {view.image_name}"
            ) from exc
        if width <= 0 or height <= 0:
            raise ValueError(
                f"registered frame has invalid dimensions: {view.image_name}"
            )

        rotation = world_to_camera[:3, :3]
        translation = world_to_camera[:3, 3]
        for start in range(0, len(positions), chunk_size):
            stop = min(start + chunk_size, len(positions))
            camera_xyz = positions[start:stop] @ rotation.T + translation
            depth = camera_xyz[:, 2]
            in_front = depth > 1e-8
            safe_depth = np.where(in_front, depth, 1.0)
            projected_x = (
                intrinsic[0, 0] * camera_xyz[:, 0] / safe_depth + intrinsic[0, 2]
            )
            projected_y = (
                intrinsic[1, 1] * camera_xyz[:, 1] / safe_depth + intrinsic[1, 2]
            )
            visible[start:stop] |= (
                in_front
                & (projected_x >= 0.0)
                & (projected_x < width)
                & (projected_y >= 0.0)
                & (projected_y < height)
            )
    return visible


def _geometry_keep_mask(
    raw: _ValidatedStaticPly,
    reconstruction: ReconstructionBundle,
    views: tuple["RegisteredFrame", ...],
    policy: PolishPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    from backend.preprocess.parse_colmap import load_points3d_from_model

    positions = np.column_stack(
        [raw.vertices[axis] for axis in ("x", "y", "z")]
    ).astype(np.float64, copy=False)
    center = np.median(positions, axis=0)
    radius = np.linalg.norm(positions - center, axis=1)
    robust_extent = float(np.percentile(radius, 99.5))
    if not math.isfinite(robust_extent) or robust_extent <= 0.0:
        raise ValueError("static PLY must have positive robust spatial extent")

    alpha = _sigmoid(raw.vertices["opacity"])
    log_scales = np.column_stack(
        [raw.vertices[f"scale_{index}"] for index in range(3)]
    ).astype(np.float64, copy=False)
    scales = np.exp(log_scales)
    maximum_scale = np.max(scales, axis=1)
    minimum_scale = np.min(scales, axis=1)
    anisotropy = maximum_scale / minimum_scale

    sparse_xyz, _, _, _ = load_points3d_from_model(reconstruction.accepted_model_dir)
    sparse_xyz = np.asarray(sparse_xyz, dtype=np.float64)
    if (
        sparse_xyz.ndim != 2
        or sparse_xyz.shape[1] != 3
        or len(sparse_xyz) == 0
        or not np.isfinite(sparse_xyz).all()
    ):
        raise ValueError("accepted reconstruction has invalid sparse points")
    core_lower, core_upper = np.percentile(sparse_xyz, (0.5, 99.5), axis=0)
    margin = (core_upper - core_lower) * policy.crop_margin_fraction
    expanded_lower = core_lower - margin
    expanded_upper = core_upper + margin
    inside_sparse_bounds = np.all(
        (positions >= expanded_lower) & (positions <= expanded_upper),
        axis=1,
    )

    distance_to_core = np.ones(len(positions), dtype=np.float64)
    for axis in range(3):
        axis_margin = margin[axis]
        if axis_margin <= 0.0:
            continue
        axis_factor = np.ones(len(positions), dtype=np.float64)
        below_core = positions[:, axis] < core_lower[axis]
        above_core = positions[:, axis] > core_upper[axis]
        axis_factor[below_core] = (
            positions[below_core, axis] - expanded_lower[axis]
        ) / axis_margin
        axis_factor[above_core] = (
            expanded_upper[axis] - positions[above_core, axis]
        ) / axis_margin
        distance_to_core = np.minimum(distance_to_core, axis_factor)
    distance_to_core = np.clip(distance_to_core, 0.0, 1.0)
    fade_factors = distance_to_core**2 * (3.0 - 2.0 * distance_to_core)

    keep_mask = (
        (alpha >= policy.min_opacity)
        & (maximum_scale <= policy.max_relative_scale * robust_extent)
        & (anisotropy <= policy.max_anisotropy)
        & inside_sparse_bounds
        & _visible_in_any_frustum(raw.vertices, views)
    )
    return keep_mask, fade_factors


def _uniform_sample_views(
    views: tuple["RegisteredFrame", ...],
    limit: int = 12,
) -> tuple["RegisteredFrame", ...]:
    if len(views) <= limit:
        return views
    indices = tuple(
        position * (len(views) - 1) // (limit - 1) for position in range(limit)
    )
    return tuple(views[index] for index in indices)


def _validate_registered_frames(
    reconstruction: ReconstructionBundle,
    views: tuple["RegisteredFrame", ...],
) -> tuple[_FileBinding, ...]:
    manifest_records = {
        record.output_name: record
        for record in reconstruction.selected_manifest.selected_frames
    }
    registered_names = reconstruction.decision.dominant.registered_names
    expected_order = tuple(
        record.output_name
        for record in reconstruction.selected_manifest.selected_frames
        if record.output_name in registered_names
    )
    actual_order = tuple(view.image_name for view in views)
    if actual_order != expected_order or set(actual_order) != set(registered_names):
        raise ValueError("registered frames do not match reconstruction manifest order")
    if len(set(actual_order)) != len(actual_order):
        raise ValueError("registered frame names must be unique")

    bindings: list[_FileBinding] = []
    for view in views:
        record = manifest_records.get(view.image_name)
        if (
            record is None
            or view.frame_id != record.frame_id
            or view.image_path.name != view.image_name
        ):
            raise ValueError(
                f"registered ground truth does not match manifest: {view.image_name}"
            )
        try:
            binding = _capture_file_binding(view.image_path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"registered ground truth is missing or unsafe: {view.image_name}"
            ) from exc
        if binding.sha256 != record.sha256:
            raise ValueError(
                f"registered ground truth hash differs from manifest: {view.image_name}"
            )
        bindings.append(binding)
    return tuple(bindings)


def polish_static_ply(
    raw_path: str | Path,
    candidate_path: str | Path,
    reconstruction: ReconstructionBundle,
    registered_frames: Sequence["RegisteredFrame"],
    *,
    policy: PolishPolicy = PolishPolicy(),
    render_evaluator: RenderEvaluator | None = None,
) -> PolishReport:
    """Build and regression-check a conservative static PLY candidate."""

    raw_target = Path(raw_path)
    candidate_target = Path(candidate_path)
    resolved_candidate = (
        candidate_target.parent.resolve(strict=False) / candidate_target.name
    )
    if raw_target.resolve(strict=True) == resolved_candidate:
        raise ValueError("candidate path must be distinct from the raw PLY")
    if os.path.lexists(candidate_target):
        raise FileExistsError(f"candidate path already exists: {candidate_target}")
    if not reconstruction.decision.passed or reconstruction.decision.failures:
        raise ValueError("PLY polish requires a passing reconstruction")
    views = tuple(registered_frames)
    if not views:
        raise ValueError("PLY polish requires at least one registered frame")
    ground_truth_bindings = _validate_registered_frames(reconstruction, views)
    evaluator = render_evaluator or evaluate_static_ply_views

    sampled_views = _uniform_sample_views(views)
    raw = validate_static_ply(raw_target)
    raw_binding = _capture_file_binding(raw.path)
    sparse_points_binding = _capture_file_binding(
        reconstruction.accepted_model_dir / "points3D.txt"
    )
    keep_mask, fade_factors = _geometry_keep_mask(
        raw,
        reconstruction,
        sampled_views,
        policy,
    )
    _require_unchanged(sparse_points_binding)
    if not np.any(keep_mask):
        report = _build_polish_report(
            raw_path=raw.path,
            candidate_path=None,
            original_count=raw.count,
            kept_count=0,
            raw_opacity_mass=_opacity_mass(raw.vertices),
            candidate_opacity_mass=0.0,
            raw_metrics=(),
            candidate_metrics=(),
            policy=policy,
            fallback_reasons=("candidate_empty",),
        )
        validate_static_ply(report.selected_path)
        _require_unchanged(raw_binding)
        _require_unchanged(sparse_points_binding)
        _require_all_unchanged(ground_truth_bindings)
        return report
    try:
        candidate = _write_normalized_candidate(
            raw,
            candidate_target,
            keep_mask,
            fade_factors,
        )
    except ValueError:
        candidate_alpha = (
            _sigmoid(raw.vertices["opacity"])[keep_mask] * fade_factors[keep_mask]
        )
        report = _build_polish_report(
            raw_path=raw.path,
            candidate_path=None,
            original_count=raw.count,
            kept_count=int(np.count_nonzero(keep_mask)),
            raw_opacity_mass=_opacity_mass(raw.vertices),
            candidate_opacity_mass=float(np.sum(candidate_alpha, dtype=np.float64)),
            raw_metrics=(),
            candidate_metrics=(),
            policy=policy,
            fallback_reasons=("candidate_validation",),
        )
        validate_static_ply(report.selected_path)
        _require_unchanged(raw_binding)
        _require_unchanged(sparse_points_binding)
        _require_all_unchanged(ground_truth_bindings)
        return report
    candidate_binding = _capture_file_binding(candidate.path)
    _require_unchanged(sparse_points_binding)
    _require_all_unchanged(ground_truth_bindings)
    fallback_reasons: tuple[str, ...] = ()
    try:
        raw_metrics = tuple(evaluator(raw.path, sampled_views))
        _require_unchanged(raw_binding)
        _require_unchanged(candidate_binding)
        _require_unchanged(sparse_points_binding)
        _require_all_unchanged(ground_truth_bindings)
        candidate_metrics = tuple(evaluator(candidate.path, sampled_views))
        _require_unchanged(raw_binding)
        _require_unchanged(candidate_binding)
        _require_unchanged(sparse_points_binding)
        _require_all_unchanged(ground_truth_bindings)
        expected_names = tuple(view.image_name for view in sampled_views)
        if (
            tuple(metric.image_name for metric in raw_metrics) != expected_names
            or tuple(metric.image_name for metric in candidate_metrics)
            != expected_names
        ):
            fallback_reasons = ("render_metrics_incomplete",)
    except Exception:
        _require_unchanged(raw_binding)
        _require_unchanged(candidate_binding)
        _require_unchanged(sparse_points_binding)
        _require_all_unchanged(ground_truth_bindings)
        raw_metrics = ()
        candidate_metrics = ()
        fallback_reasons = ("render_evaluation",)
    report = _build_polish_report(
        raw_path=raw.path,
        candidate_path=candidate.path,
        original_count=raw.count,
        kept_count=candidate.count,
        raw_opacity_mass=_opacity_mass(raw.vertices),
        candidate_opacity_mass=_opacity_mass(candidate.vertices),
        raw_metrics=raw_metrics,
        candidate_metrics=candidate_metrics,
        policy=policy,
        fallback_reasons=fallback_reasons,
    )
    validate_static_ply(report.selected_path)
    _require_unchanged(raw_binding)
    _require_unchanged(candidate_binding)
    _require_unchanged(sparse_points_binding)
    _require_all_unchanged(ground_truth_bindings)
    return report

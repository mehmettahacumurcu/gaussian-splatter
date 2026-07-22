from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.model.trainer import (
    _resize_validity_mask,
    _ssim_loss_map,
    load_frame_tensor,
    masked_psnr,
    psnr,
)

from .ablation import (
    AblationCheckpoint,
    AblationVariant,
    GateDecision,
    StructuralMetrics,
    evaluate_checkpoint,
)
from .contracts import FrameArtifact, LearnedArtifacts
from .density import AdaptiveDensityController
from .training import (
    _atomic_copy,
    _augment_points3d,
    _coverage_indices,
    _make_quality_probe,
    _validate_evidence,
    _write_json_atomic,
)

if TYPE_CHECKING:
    from backend.notebooks.models import StaticNotebookRunSpec
    from backend.static_pipeline.training import PreparedTrainingInput, TrainingResult

    from .contracts import LearnedReconstructionOutput


@dataclass(frozen=True)
class PreparedAblationScene:
    variant: AblationVariant
    scene_dir: Path
    original_frames: tuple[FrameArtifact, ...]
    camera_frames: tuple[Path, ...]
    validity_mask: tuple[torch.Tensor, ...] | None
    quality_validity: tuple[torch.Tensor, ...]
    depth_dir: Path | None
    adaptive_density: bool
    training_rgb_digest: str


@dataclass(frozen=True)
class StructuralSnapshot:
    metrics: StructuralMetrics
    robust_scene_extent: float
    opacity_quantiles: Mapping[str, float]
    scale_quantiles: Mapping[str, float]
    sh_dc_abs_quantiles: Mapping[str, float]
    sh_rest_abs_quantiles: Mapping[str, float]
    max_anisotropy: float
    centroid: tuple[float, float, float]
    covariance_eigenvalues: tuple[float, float, float]


class AblationGateFailure(RuntimeError):
    def __init__(self, checkpoint: AblationCheckpoint, decision: GateDecision) -> None:
        self.checkpoint = checkpoint
        self.decision = decision
        super().__init__(
            f"ablation gate stopped {checkpoint.experiment_id} at "
            f"iteration {checkpoint.iteration}: {', '.join(decision.reasons)}"
        )


@dataclass(frozen=True)
class AblationExperimentResult:
    experiment_id: str
    passed: bool
    stopped_early: bool
    stop_reasons: tuple[str, ...]
    checkpoints: tuple[AblationCheckpoint, ...]
    raw_ply_path: Path | None
    metrics_path: Path | None
    run_manifest_path: Path | None
    status: Mapping[str, object]


def make_ablation_static_spec(
    base_spec: StaticNotebookRunSpec,
    variant: AblationVariant,
) -> StaticNotebookRunSpec:
    """Resolve a short, fixed-720p diagnostic without changing core losses."""

    uses_depth = "depth" in variant.features
    density_start = 500 if variant.density_events else variant.n_iterations
    density_end = (
        max(600, variant.n_iterations - 500)
        if variant.density_events
        else variant.n_iterations
    )
    advanced = base_spec.quality.advanced.model_copy(
        update={
            "run_eval": False,
            # Keep the profile's depth-loss defaults resolvable for depth
            # variants. The ablation pipeline runner still forces
            # skip_foundation=True because verified depth was staged once.
            "foundation": True,
            "lambda_depth": None if uses_depth else 0.0,
            "resolution_long_edge_cap": 1_280,
            "density_start_iter": density_start,
            "density_end_iter": density_end,
            "opacity_reset_interval": (
                base_spec.quality.advanced.opacity_reset_interval
                if variant.density_events
                else 0
            ),
            "multires_schedule": [(0, 720)],
        }
    )
    quality = base_spec.quality.model_copy(
        update={
            "n_iters": variant.n_iterations,
            "advanced": advanced,
        }
    )
    return base_spec.model_copy(update={"quality": quality})


def prepare_ablation_scene(
    artifacts: LearnedArtifacts,
    *,
    accepted_model_dir: Path,
    scene_dir: Path,
    variant: AblationVariant,
) -> PreparedAblationScene:
    scene = Path(scene_dir)
    evidence = _validate_evidence(artifacts, Path(accepted_model_dir), scene)

    # RGB is deliberately held constant across every ablation. Photometric
    # normalization is not one of the four tested variables.
    for frame in evidence.original_frames:
        destination = scene / "frames" / frame.image_name
        if destination.resolve(strict=False) != frame.path.resolve(strict=True):
            _atomic_copy(frame.path, destination)

    depth_dir: Path | None = None
    if "depth" in variant.features:
        depth_dir = scene / "depth"
        if os.path.lexists(depth_dir):
            raise FileExistsError("ablation depth destination must be fresh")
        depth_dir.mkdir()
        for frame, (source, _digest) in zip(
            evidence.original_frames,
            evidence.depth_sources,
            strict=True,
        ):
            _atomic_copy(
                source,
                depth_dir / f"{Path(frame.image_name).stem}_depth.npy",
            )

    if "dense_seeds" in variant.features:
        _augment_points3d(
            scene / "colmap" / "sparse" / "0" / "points3D.txt",
            evidence.seed_xyz,
            evidence.seed_rgb,
        )

    validity = evidence.validity if "masks" in variant.features else None
    return PreparedAblationScene(
        variant=variant,
        scene_dir=scene,
        original_frames=evidence.original_frames,
        camera_frames=tuple(
            scene / "frames" / frame.image_name for frame in evidence.original_frames
        ),
        validity_mask=validity,
        quality_validity=evidence.validity,
        depth_dir=depth_dir,
        adaptive_density="adaptive_density" in variant.features,
        training_rgb_digest=evidence.training_rgb_digest,
    )


def _finite_quantiles(values: torch.Tensor) -> dict[str, float]:
    flattened = values.detach().float().reshape(-1).cpu()
    finite = flattened[torch.isfinite(flattened)]
    if finite.numel() == 0:
        return {
            "q0": 0.0,
            "q25": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q95": 0.0,
            "q99": 0.0,
            "q100": 0.0,
        }
    levels = (0.0, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0)
    if finite.numel() <= 1 << 24:
        result = torch.quantile(
            finite,
            torch.tensor(levels, dtype=finite.dtype),
        ).tolist()
    else:
        # torch.quantile rejects inputs above 2**24 elements. SH-rest has 45
        # values per Gaussian, so otherwise healthy ablations can cross that
        # boundary long before the configured Gaussian ceiling. The filtered
        # tensor is already an owned CPU buffer, making an exact in-place NumPy
        # quantile both bounded and equivalent to Torch's linear interpolation.
        result = np.quantile(
            finite.numpy(),
            levels,
            method="linear",
            overwrite_input=True,
        ).tolist()
    return {
        name: float(value)
        for name, value in zip(
            ("q0", "q25", "q50", "q75", "q95", "q99", "q100"),
            result,
            strict=True,
        )
    }


def collect_structural_snapshot(trainer: Any) -> StructuralSnapshot:
    gs = trainer.gs
    means = gs.means.detach().float()
    scales = gs.get_scales.detach().float()
    opacities = gs.get_opacities.detach().float().reshape(-1)
    sh_dc = gs.sh_dc.detach().float()
    sh_rest = gs.sh_rest.detach().float()
    tensors = (means, scales, opacities, sh_dc, sh_rest)
    nonfinite_count = sum(
        int((~torch.isfinite(tensor)).sum().item()) for tensor in tensors
    )

    finite_means = means[torch.isfinite(means).all(dim=-1)]
    if finite_means.numel() == 0:
        robust_extent = 0.0
        center = torch.zeros(3, dtype=torch.float32)
        covariance_eigenvalues = torch.zeros(3, dtype=torch.float32)
        out_of_bounds_fraction = 0.0
    else:
        center = torch.median(finite_means, dim=0).values
        centered = finite_means - center
        radii = torch.linalg.vector_norm(centered, dim=-1)
        robust_extent = float(torch.quantile(radii, 0.995).item())
        covariance = centered.T @ centered / max(int(centered.shape[0]), 1)
        covariance_eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        scene_extent = float(getattr(trainer, "scene_extent", robust_extent))
        bound = max(scene_extent, 1e-12)
        out_of_bounds_fraction = float((radii > bound).float().mean().item())

    finite_rows = (
        torch.isfinite(scales).all(dim=-1)
        & torch.isfinite(opacities)
        & torch.isfinite(sh_dc).all(dim=(-1, -2))
    )
    visible = finite_rows & (opacities >= 0.10)
    visible_count = int(visible.sum().item())
    denominator = max(visible_count, 1)
    safe_scales = torch.where(torch.isfinite(scales), scales, torch.ones_like(scales))
    anisotropy = safe_scales.max(dim=-1).values / safe_scales.min(
        dim=-1
    ).values.clamp_min(1e-12)
    high_anisotropy = int((visible & (anisotropy > 30.0)).sum().item())
    oversize_limit = 0.03 * robust_extent
    oversized = int(
        (visible & (safe_scales.max(dim=-1).values > oversize_limit)).sum().item()
    )
    c0 = 0.28209479177387814
    dc_rgb = sh_dc[:, 0, :] * c0 + 0.5
    pure_white = int((visible & (dc_rgb >= 0.99).all(dim=-1)).sum().item())

    return StructuralSnapshot(
        metrics=StructuralMetrics(
            visible_white_fraction=pure_white / denominator,
            high_anisotropy_fraction=high_anisotropy / denominator,
            oversized_fraction=oversized / denominator,
            nonfinite_count=nonfinite_count,
            out_of_bounds_fraction=out_of_bounds_fraction,
        ),
        robust_scene_extent=robust_extent,
        opacity_quantiles=_finite_quantiles(opacities),
        scale_quantiles=_finite_quantiles(scales),
        sh_dc_abs_quantiles=_finite_quantiles(sh_dc.abs()),
        sh_rest_abs_quantiles=_finite_quantiles(sh_rest.abs()),
        max_anisotropy=float(_finite_quantiles(anisotropy)["q100"]),
        centroid=tuple(float(value) for value in center.tolist()),
        covariance_eigenvalues=tuple(
            float(value) for value in covariance_eigenvalues.tolist()
        ),
    )


def summarize_render_metrics(
    rendered: torch.Tensor,
    ground_truth: torch.Tensor,
    validity: torch.Tensor,
) -> dict[str, float]:
    if rendered.shape != ground_truth.shape or rendered.ndim != 3:
        raise ValueError("rendered and ground_truth must be matching HxWxC tensors")
    if validity.shape != rendered.shape[:2]:
        raise ValueError("validity must match the render dimensions")
    mask = validity.to(device=rendered.device, dtype=rendered.dtype).clamp(0.0, 1.0)
    unmasked_l1 = float((rendered - ground_truth).abs().mean().item())
    per_pixel_l1 = (rendered - ground_truth).abs().mean(dim=-1)
    support = float(mask.sum().item())
    masked_l1 = (
        float((per_pixel_l1 * mask).sum().item() / support) if support > 0 else 0.0
    )
    unmasked_psnr = psnr(rendered, ground_truth)
    masked_psnr_value = masked_psnr(rendered, ground_truth, mask)
    composited = rendered * mask.unsqueeze(-1) + ground_truth * (
        1.0 - mask.unsqueeze(-1)
    )
    unmasked_ssim = 1.0 - float(_ssim_loss_map(rendered, ground_truth).mean().item())
    masked_ssim = 1.0 - float(_ssim_loss_map(composited, ground_truth).mean().item())

    def bounded(value: float) -> float:
        return min(1.0, max(0.0, value))

    rendered_gray = rendered.mean(dim=-1)
    ground_truth_gray = ground_truth.mean(dim=-1)

    def edge_map(value: torch.Tensor) -> torch.Tensor:
        horizontal = F.pad((value[:, 1:] - value[:, :-1]).abs(), (0, 1, 0, 0))
        vertical = F.pad((value[1:, :] - value[:-1, :]).abs(), (0, 0, 0, 1))
        return torch.sqrt(horizontal.square() + vertical.square() + 1e-12)

    rendered_edges = edge_map(rendered_gray)
    target_edges = edge_map(ground_truth_gray)
    edge_l1 = float((rendered_edges - target_edges).abs().mean().item())
    rendered_centered = rendered_edges.flatten() - rendered_edges.mean()
    target_centered = target_edges.flatten() - target_edges.mean()
    denominator = torch.linalg.vector_norm(rendered_centered) * torch.linalg.vector_norm(
        target_centered
    )
    if float(denominator.item()) <= 1e-12:
        edge_correlation = 1.0 if edge_l1 <= 1e-6 else 0.0
    else:
        raw_correlation = float(
            torch.dot(rendered_centered, target_centered).item() / denominator.item()
        )
        edge_correlation = min(1.0, max(0.0, (raw_correlation + 1.0) / 2.0))

    return {
        "psnr_unmasked": 100.0 if math.isinf(unmasked_psnr) else unmasked_psnr,
        "psnr_masked": (
            100.0 if math.isinf(masked_psnr_value) else masked_psnr_value
        ),
        "ssim_unmasked": bounded(unmasked_ssim),
        "ssim_masked": bounded(masked_ssim),
        "l1_unmasked": unmasked_l1,
        "l1_masked": masked_l1,
        "edge_l1": edge_l1,
        "edge_correlation": edge_correlation,
    }


def _tensor_image(value: torch.Tensor, *, width: int = 320) -> Image.Image:
    array = (
        value.detach().float().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    )
    image = Image.fromarray(array, mode="RGB")
    if image.width > width:
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


@dataclass(frozen=True)
class DiagnosticRenderView:
    """Fixed and perturbed render tensors for a single verified probe camera."""

    K: torch.Tensor
    w2c: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    perturbed_w2c: torch.Tensor
    perturbed_alpha: torch.Tensor
    perturbed_depth: torch.Tensor


class AblationDiagnosticCollector:
    def __init__(
        self,
        *,
        scene_dir: Path,
        frames: tuple[FrameArtifact, ...],
        validity: tuple[torch.Tensor, ...],
        variant: AblationVariant,
        output_root: Path,
        control_checkpoints: Mapping[int, AblationCheckpoint] | None = None,
        checkpoint_observer: Callable[
            [Any, int, tuple[int, int], int, tuple[DiagnosticRenderView, ...]],
            Mapping[str, object],
        ]
        | None = None,
        stop_on_quality: bool = True,
    ) -> None:
        from backend.preprocess.frame_alignment import join_registered_frames
        from backend.preprocess.parse_colmap import parse_cameras_from_model

        self.scene_dir = Path(scene_dir)
        self.variant = variant
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        model_dir = self.scene_dir / "colmap" / "sparse" / "0"
        cameras = parse_cameras_from_model(model_dir)
        registered = join_registered_frames(self.scene_dir / "frames", cameras)
        by_name = {
            frame.image_name: mask
            for frame, mask in zip(frames, validity, strict=True)
        }
        if any(frame.image_name not in by_name for frame in registered):
            raise ValueError("diagnostic cameras require an exact frame join")
        self.registered = registered
        self.validity = tuple(by_name[frame.image_name] for frame in registered)
        self.probe_indices = _coverage_indices(len(registered))
        self.control_checkpoints = dict(control_checkpoints or {})
        self.checkpoint_observer = checkpoint_observer
        self.stop_on_quality = bool(stop_on_quality)
        self.checkpoints: list[AblationCheckpoint] = []
        self.decisions: list[GateDecision] = []
        self.extra_checkpoints: list[Mapping[str, object]] = []
        self._consecutive_psnr_deficits = 0
        self._structural_streaks: dict[str, int] = {}
        self._lpips_metric: Any | None = None
        self._lpips_checked = False

    @staticmethod
    def _depth_image(depth: torch.Tensor, alpha: torch.Tensor) -> Image.Image:
        valid = torch.isfinite(depth) & (depth > 0.0) & (alpha > 0.01)
        normalized = torch.zeros_like(depth, dtype=torch.float32)
        if bool(valid.any()):
            values = depth[valid].float()
            low = torch.quantile(values, 0.02)
            high = torch.quantile(values, 0.98)
            span = (high - low).clamp_min(1e-6)
            normalized[valid] = ((depth[valid].float() - low) / span).clamp(0.0, 1.0)
        return _tensor_image(normalized.unsqueeze(-1).repeat(1, 1, 3))

    def _lpips_distance(
        self,
        rendered: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> float | None:
        if not self._lpips_checked:
            from backend.model.losses_perceptual import LPIPSLoss

            self._lpips_metric = LPIPSLoss(net="alex").to(rendered.device)
            self._lpips_checked = True
        if self._lpips_metric is None:
            return None
        value = self._lpips_metric(
            rendered.permute(2, 0, 1),
            ground_truth.permute(2, 0, 1),
        )
        if getattr(self._lpips_metric, "_available", False) is not True:
            return None
        result = float(value.item())
        return result if math.isfinite(result) else None

    def _render_fixed_views(
        self,
        trainer: Any,
        resolution: tuple[int, int],
        sh_degree: int,
        iteration: int,
    ) -> tuple[dict[str, float | None], tuple[DiagnosticRenderView, ...]]:
        from backend.model.renderer import render_view

        width, height = resolution
        aggregate: dict[str, list[float]] = {
            key: []
            for key in (
                "psnr_unmasked",
                "psnr_masked",
                "ssim_unmasked",
                "ssim_masked",
                "l1_unmasked",
                "l1_masked",
                "edge_l1",
                "edge_correlation",
                "alpha_coverage",
                "depth_finite_fraction",
                "depth_median",
                "perturbed_alpha_coverage",
                "perturbed_depth_finite_fraction",
                "perturbed_depth_median",
            )
        }
        lpips_values: list[float] = []
        fixed_rows: list[tuple[Image.Image, ...]] = []
        perturbed_rows: list[Image.Image] = []
        render_views: list[DiagnosticRenderView] = []
        with torch.no_grad():
            for index in self.probe_indices:
                frame = self.registered[index]
                gt = load_frame_tensor(frame.image_path, (width, height)).to(
                    trainer.device
                )
                validity = _resize_validity_mask(
                    self.validity[index].to(trainer.device),
                    (height, width),
                )
                source_height, source_width = self.validity[index].shape
                K = torch.from_numpy(frame.K).float().to(trainer.device)
                K = K.clone()
                K[0, :] *= width / source_width
                K[1, :] *= height / source_height
                w2c = torch.from_numpy(frame.w2c).float().to(trainer.device)
                if getattr(trainer, "static_mode", False):
                    means = trainer.gs.means
                    quats = F.normalize(trainer.gs.quats, dim=-1)
                    scales = trainer.gs.get_scales
                else:
                    timestamp = index / max(len(self.registered) - 1, 1)
                    means, quats, scales = trainer._apply_deformation(timestamp)
                rendered, alpha, _ = render_view(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=trainer.gs.get_opacities,
                    colors=trainer.gs.get_colors,
                    K=K,
                    w2c=w2c,
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                    with_depth=True,
                )
                rgb = rendered[..., :3]
                depth = rendered[..., 3]
                metrics = summarize_render_metrics(rgb, gt, validity)
                for name, value in metrics.items():
                    aggregate[name].append(value)
                alpha_plane = alpha[..., 0]
                depth_valid = (
                    torch.isfinite(depth) & (depth > 0.0) & (alpha_plane > 0.01)
                )
                aggregate["alpha_coverage"].append(
                    float((alpha_plane > 0.01).float().mean().item())
                )
                aggregate["depth_finite_fraction"].append(
                    float(depth_valid.float().mean().item())
                )
                aggregate["depth_median"].append(
                    float(depth[depth_valid].median().item())
                    if bool(depth_valid.any())
                    else 0.0
                )
                lpips_value = self._lpips_distance(rgb, gt)
                if lpips_value is not None:
                    lpips_values.append(lpips_value)

                fixed_rows.append(
                    (
                        _tensor_image(gt),
                        _tensor_image(rgb),
                        _tensor_image((rgb - gt).abs()),
                        self._depth_image(depth, alpha_plane),
                    )
                )

                perturbed_w2c = w2c.clone()
                perturbed_w2c[0, 3] += 0.02 * max(
                    float(getattr(trainer, "scene_extent", 1.0)),
                    1e-3,
                )
                perturbed, perturbed_alpha, _ = render_view(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=trainer.gs.get_opacities,
                    colors=trainer.gs.get_colors,
                    K=K,
                    w2c=perturbed_w2c,
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                    with_depth=True,
                )
                perturbed_rgb = perturbed[..., :3]
                perturbed_depth = perturbed[..., 3]
                perturbed_alpha_plane = perturbed_alpha[..., 0]
                perturbed_valid = (
                    torch.isfinite(perturbed_depth)
                    & (perturbed_depth > 0.0)
                    & (perturbed_alpha_plane > 0.01)
                )
                aggregate["perturbed_alpha_coverage"].append(
                    float((perturbed_alpha_plane > 0.01).float().mean().item())
                )
                aggregate["perturbed_depth_finite_fraction"].append(
                    float(perturbed_valid.float().mean().item())
                )
                aggregate["perturbed_depth_median"].append(
                    float(perturbed_depth[perturbed_valid].median().item())
                    if bool(perturbed_valid.any())
                    else 0.0
                )
                perturbed_rows.append(_tensor_image(perturbed_rgb))
                render_views.append(
                    DiagnosticRenderView(
                        K=K.detach(),
                        w2c=w2c.detach(),
                        alpha=alpha_plane.detach(),
                        depth=depth.detach(),
                        perturbed_w2c=perturbed_w2c.detach(),
                        perturbed_alpha=perturbed_alpha_plane.detach(),
                        perturbed_depth=perturbed_depth.detach(),
                    )
                )

        if not fixed_rows:
            raise ValueError("diagnostic fixed-view set is empty")
        row_height = max(image.height for row in fixed_rows for image in row)
        column_width = max(image.width for row in fixed_rows for image in row)
        fixed_contact = Image.new(
            "RGB",
            (column_width * 4, row_height * len(fixed_rows)),
        )
        for row_index, row in enumerate(fixed_rows):
            for column_index, image in enumerate(row):
                fixed_contact.paste(
                    image,
                    (column_index * column_width, row_index * row_height),
                )
        fixed_contact.save(
            self.output_root / f"contact_{iteration:06d}_fixed.png"
        )
        fixed_contact.save(self.output_root / f"contact_{iteration:06d}.png")

        perturbed_contact = Image.new(
            "RGB",
            (column_width, row_height * len(perturbed_rows)),
        )
        for row_index, image in enumerate(perturbed_rows):
            perturbed_contact.paste(image, (0, row_index * row_height))
        perturbed_contact.save(
            self.output_root / f"contact_{iteration:06d}_perturbed.png"
        )

        result: dict[str, float | None] = {
            name: float(sum(values) / len(values))
            for name, values in aggregate.items()
        }
        result["lpips_unmasked"] = (
            float(sum(lpips_values) / len(lpips_values)) if lpips_values else None
        )
        return result, tuple(render_views)

    def __call__(
        self,
        trainer: Any,
        iteration: int,
        resolution: tuple[int, int],
        sh_degree: int,
    ) -> None:
        metrics, render_views = self._render_fixed_views(
            trainer,
            resolution,
            sh_degree,
            iteration,
        )
        structural = collect_structural_snapshot(trainer)
        checkpoint = AblationCheckpoint(
            experiment_id=self.variant.experiment_id,
            iteration=iteration,
            psnr_unmasked=metrics["psnr_unmasked"],
            psnr_masked=metrics["psnr_masked"],
            ssim_unmasked=metrics["ssim_unmasked"],
            ssim_masked=metrics["ssim_masked"],
            l1_unmasked=metrics["l1_unmasked"],
            l1_masked=metrics["l1_masked"],
            gaussian_count=int(trainer.gs.num_points),
            structural=structural.metrics,
            edge_l1=float(metrics["edge_l1"]),
            edge_correlation=float(metrics["edge_correlation"]),
            lpips_unmasked=(
                float(metrics["lpips_unmasked"])
                if metrics["lpips_unmasked"] is not None
                else None
            ),
            alpha_coverage=float(metrics["alpha_coverage"]),
            depth_finite_fraction=float(metrics["depth_finite_fraction"]),
            depth_median=float(metrics["depth_median"]),
            perturbed_alpha_coverage=float(metrics["perturbed_alpha_coverage"]),
            perturbed_depth_finite_fraction=float(
                metrics["perturbed_depth_finite_fraction"]
            ),
            perturbed_depth_median=float(metrics["perturbed_depth_median"]),
        )
        control = self.control_checkpoints.get(iteration)
        if self.stop_on_quality:
            decision = evaluate_checkpoint(
                checkpoint,
                control=control,
                previous_psnr_deficits=self._consecutive_psnr_deficits,
                previous_structural_streaks=self._structural_streaks,
            )
        else:
            nonfinite = structural.metrics.nonfinite_count > 0
            decision = GateDecision(
                stop=nonfinite,
                reasons=("nonfinite_values",) if nonfinite else (),
            )
        extra: Mapping[str, object] | None = None
        if self.checkpoint_observer is not None:
            observed = self.checkpoint_observer(
                trainer,
                iteration,
                resolution,
                sh_degree,
                render_views,
            )
            if not isinstance(observed, Mapping):
                raise TypeError("checkpoint observer must return a mapping")
            extra = dict(observed)
            self.extra_checkpoints.append(extra)
        self._consecutive_psnr_deficits = decision.consecutive_psnr_deficits
        self._structural_streaks = dict(decision.structural_streaks)
        self.checkpoints.append(checkpoint)
        self.decisions.append(decision)
        _write_json_atomic(
            self.output_root / f"checkpoint_{iteration:06d}.json",
            {
                "checkpoint": asdict(checkpoint),
                "gate": asdict(decision),
                "structural": asdict(structural),
                "extra": extra,
            },
        )
        if decision.stop:
            raise AblationGateFailure(checkpoint, decision)


class AblationPipelineRunner:
    def __init__(
        self,
        artifacts: LearnedArtifacts,
        *,
        accepted_model_dir: Path,
        variant: AblationVariant,
        pipeline_runner: Callable[..., Mapping[str, object]],
        diagnostic_root: Path,
        seed: int,
        control_checkpoints: Mapping[int, AblationCheckpoint] | None = None,
        density_factory: Callable[[], Any] = AdaptiveDensityController,
    ) -> None:
        self.artifacts = artifacts
        self.accepted_model_dir = Path(accepted_model_dir)
        self.variant = variant
        self.pipeline_runner = pipeline_runner
        self.diagnostic_root = Path(diagnostic_root)
        self.seed = int(seed)
        self.control_checkpoints = dict(control_checkpoints or {})
        self.density_factory = density_factory
        self.collector: AblationDiagnosticCollector | None = None

    def __call__(self, **kwargs: object) -> Mapping[str, object]:
        if "video_path" not in kwargs:
            raise ValueError("ablation pipeline runner requires video_path")
        if "trainer_customizer" in kwargs or "trainer_train_kwargs" in kwargs:
            raise ValueError("ablation trainer injection cannot be overridden")
        scene_dir = Path(kwargs["video_path"]).parent
        prepared = prepare_ablation_scene(
            self.artifacts,
            accepted_model_dir=self.accepted_model_dir,
            scene_dir=scene_dir,
            variant=self.variant,
        )
        quality_probe, registered_validity = _make_quality_probe(
            scene_dir,
            prepared.original_frames,
            prepared.quality_validity,
        )
        self.collector = AblationDiagnosticCollector(
            scene_dir=scene_dir,
            frames=prepared.original_frames,
            validity=prepared.quality_validity,
            variant=self.variant,
            output_root=self.diagnostic_root,
            control_checkpoints=self.control_checkpoints,
        )

        def customize(trainer: Any) -> None:
            if prepared.adaptive_density:
                trainer.density = self.density_factory()

        train_kwargs: dict[str, object] = {
            "camera_generator": torch.Generator(device="cpu").manual_seed(self.seed),
            "diagnostic_iterations": self.variant.checkpoints,
            "diagnostic_callback": self.collector,
        }
        if prepared.validity_mask is not None:
            train_kwargs["validity_mask"] = registered_validity
        if prepared.adaptive_density:
            train_kwargs["density_quality_probe"] = quality_probe

        forwarded = dict(kwargs)
        forwarded["skip_foundation"] = True
        forwarded["trainer_customizer"] = customize
        forwarded["trainer_train_kwargs"] = train_kwargs
        status = self.pipeline_runner(**forwarded)
        if not isinstance(status, Mapping):
            raise TypeError("ablation pipeline runner must return a status mapping")
        return dict(status)


def run_ablation_experiment(
    prepared: PreparedTrainingInput,
    base_spec: StaticNotebookRunSpec,
    reconstruction: LearnedReconstructionOutput,
    variant: AblationVariant,
    *,
    diagnostic_root: Path,
    seed: int = 1701,
    source_long_edge: int | None = None,
    control_checkpoints: Mapping[int, AblationCheckpoint] | None = None,
    pipeline_runner: Callable[..., Mapping[str, object]] | None = None,
    validated_training_runner: Callable[..., TrainingResult] | None = None,
) -> AblationExperimentResult:
    """Run one isolated training variant and convert gate stops into receipts."""

    if prepared.reconstruction != reconstruction.bundle:
        raise ValueError("prepared and learned reconstruction contracts disagree")
    if pipeline_runner is None:
        from backend.pipeline import run_pipeline

        pipeline_runner = run_pipeline
    if validated_training_runner is None:
        from backend.static_pipeline.training import run_validated_training

        validated_training_runner = run_validated_training

    experiment_runner = AblationPipelineRunner(
        reconstruction.artifacts,
        accepted_model_dir=reconstruction.accepted_model_dir,
        variant=variant,
        pipeline_runner=pipeline_runner,
        diagnostic_root=Path(diagnostic_root),
        seed=seed,
        control_checkpoints=control_checkpoints,
    )
    training_kwargs: dict[str, object] = {
        "source_long_edge": source_long_edge,
        "pipeline_runner": experiment_runner,
    }
    acceptance = getattr(reconstruction, "acceptance", None)
    if acceptance is not None and acceptance.mode == "best_effort":
        from .training import make_output_first_reconstruction_validator

        training_kwargs["reconstruction_validator"] = (
            make_output_first_reconstruction_validator(reconstruction)
        )

    spec = make_ablation_static_spec(base_spec, variant)
    try:
        training = validated_training_runner(prepared, spec, **training_kwargs)
    except AblationGateFailure as failure:
        checkpoints = (
            tuple(experiment_runner.collector.checkpoints)
            if experiment_runner.collector is not None
            else (failure.checkpoint,)
        )
        return AblationExperimentResult(
            experiment_id=variant.experiment_id,
            passed=False,
            stopped_early=True,
            stop_reasons=tuple(failure.decision.reasons),
            checkpoints=checkpoints,
            raw_ply_path=None,
            metrics_path=None,
            run_manifest_path=None,
            status={"gate": "stopped"},
        )

    collector = experiment_runner.collector
    if collector is None:
        raise RuntimeError("ablation training did not install its diagnostics")
    checkpoints = tuple(collector.checkpoints)
    observed = tuple(checkpoint.iteration for checkpoint in checkpoints)
    if observed != variant.checkpoints:
        raise RuntimeError(
            "ablation training did not publish every diagnostic checkpoint: "
            f"expected {variant.checkpoints}, observed {observed}"
        )
    raw_ply_path = Path(training.raw_ply_path)
    metrics_path = raw_ply_path.parent.parent / "logs" / "metrics.jsonl"
    if not metrics_path.is_file():
        metrics_path = None
    return AblationExperimentResult(
        experiment_id=variant.experiment_id,
        passed=True,
        stopped_early=False,
        stop_reasons=(),
        checkpoints=checkpoints,
        raw_ply_path=raw_ply_path,
        metrics_path=metrics_path,
        run_manifest_path=Path(training.run_manifest_path),
        status=dict(training.status),
    )

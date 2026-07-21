from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

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

    density_end = max(600, variant.n_iterations - 500)
    uses_depth = "depth" in variant.features
    advanced = base_spec.quality.advanced.model_copy(
        update={
            "run_eval": False,
            # Keep the profile's depth-loss defaults resolvable for depth
            # variants. The ablation pipeline runner still forces
            # skip_foundation=True because verified depth was staged once.
            "foundation": True,
            "lambda_depth": None if uses_depth else 0.0,
            "resolution_long_edge_cap": 1_280,
            "density_start_iter": 500,
            "density_end_iter": density_end,
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
    levels = torch.tensor(
        [0.0, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0],
        dtype=finite.dtype,
    )
    result = torch.quantile(finite, levels)
    return {
        name: float(value)
        for name, value in zip(
            ("q0", "q25", "q50", "q75", "q95", "q99", "q100"),
            result.tolist(),
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
    else:
        center = torch.median(finite_means, dim=0).values
        radii = torch.linalg.vector_norm(finite_means - center, dim=-1)
        robust_extent = float(torch.quantile(radii, 0.995).item())

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
        ),
        robust_scene_extent=robust_extent,
        opacity_quantiles=_finite_quantiles(opacities),
        scale_quantiles=_finite_quantiles(scales),
        sh_dc_abs_quantiles=_finite_quantiles(sh_dc.abs()),
        sh_rest_abs_quantiles=_finite_quantiles(sh_rest.abs()),
        max_anisotropy=float(_finite_quantiles(anisotropy)["q100"]),
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

    return {
        "psnr_unmasked": 100.0 if math.isinf(unmasked_psnr) else unmasked_psnr,
        "psnr_masked": (
            100.0 if math.isinf(masked_psnr_value) else masked_psnr_value
        ),
        "ssim_unmasked": bounded(unmasked_ssim),
        "ssim_masked": bounded(masked_ssim),
        "l1_unmasked": unmasked_l1,
        "l1_masked": masked_l1,
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
        self.checkpoints: list[AblationCheckpoint] = []
        self.decisions: list[GateDecision] = []
        self._consecutive_psnr_deficits = 0

    def _render_fixed_views(
        self,
        trainer: Any,
        resolution: tuple[int, int],
        sh_degree: int,
        iteration: int,
    ) -> dict[str, float]:
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
            )
        }
        rows: list[tuple[Image.Image, Image.Image]] = []
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
                rendered, _, _ = render_view(
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
                    with_depth=False,
                )
                rgb = rendered[..., :3]
                metrics = summarize_render_metrics(rgb, gt, validity)
                for name, value in metrics.items():
                    aggregate[name].append(value)
                rows.append((_tensor_image(gt), _tensor_image(rgb)))

        if not rows:
            raise ValueError("diagnostic fixed-view set is empty")
        row_height = max(image.height for row in rows for image in row)
        column_width = max(image.width for row in rows for image in row)
        contact = Image.new("RGB", (column_width * 2, row_height * len(rows)))
        for row_index, (ground_truth, rendered) in enumerate(rows):
            contact.paste(ground_truth, (0, row_index * row_height))
            contact.paste(rendered, (column_width, row_index * row_height))
        contact.save(self.output_root / f"contact_{iteration:06d}.png")
        return {
            name: float(sum(values) / len(values))
            for name, values in aggregate.items()
        }

    def __call__(
        self,
        trainer: Any,
        iteration: int,
        resolution: tuple[int, int],
        sh_degree: int,
    ) -> None:
        metrics = self._render_fixed_views(
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
        )
        control = self.control_checkpoints.get(iteration)
        decision = evaluate_checkpoint(
            checkpoint,
            control=control,
            previous_psnr_deficits=self._consecutive_psnr_deficits,
        )
        self._consecutive_psnr_deficits = decision.consecutive_psnr_deficits
        self.checkpoints.append(checkpoint)
        self.decisions.append(decision)
        _write_json_atomic(
            self.output_root / f"checkpoint_{iteration:06d}.json",
            {
                "checkpoint": asdict(checkpoint),
                "gate": asdict(decision),
                "structural": asdict(structural),
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

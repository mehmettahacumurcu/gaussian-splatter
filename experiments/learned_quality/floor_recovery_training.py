from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np
import torch
from scipy.spatial import cKDTree

from .ablation import AblationCheckpoint, AblationVariant
from .ablation_training import (
    AblationDiagnosticCollector,
    make_ablation_static_spec,
)
from .contracts import FrameArtifact, LearnedArtifacts
from .floor_recovery import FLOOR_RECOVERY_SCHEMA, FloorSeedArtifact
from .training import _atomic_copy, _augment_points3d, _validate_evidence

if TYPE_CHECKING:
    from backend.notebooks.models import StaticNotebookRunSpec
    from backend.static_pipeline.training import PreparedTrainingInput, TrainingResult

    from .contracts import LearnedReconstructionOutput


FLOOR_RECOVERY_CHECKPOINTS = (0, 100, 499, 500, 600, 1_000, 2_500, 5_000)
FLOOR_RECOVERY_VARIANT = AblationVariant(
    experiment_id="floor_recovery",
    features=frozenset({"depth"}),
    n_iterations=5_000,
    checkpoints=FLOOR_RECOVERY_CHECKPOINTS,
    density_events=True,
)


@dataclass(frozen=True)
class FloorCheckpointMetrics:
    iteration: int
    floor_alpha_coverage: float
    residual_hole_fraction: float
    plane_depth_relative_error: float
    perturbed_depth_disagreement_ratio: float
    occupied_hole_fraction: float


@dataclass(frozen=True)
class FloorTrainingResult:
    checkpoints: tuple[AblationCheckpoint, ...]
    floor_checkpoints: tuple[FloorCheckpointMetrics, ...]
    raw_ply_path: Path
    metrics_path: Path | None
    run_manifest_path: Path
    status: Mapping[str, object]


@dataclass(frozen=True)
class PreparedFloorRecoveryScene:
    scene_dir: Path
    original_frames: tuple[FrameArtifact, ...]
    quality_validity: tuple[torch.Tensor, ...]
    depth_dir: Path
    original_point_count: int
    floor_xyz: np.ndarray
    floor_rgb: np.ndarray
    maximum_scale: float
    plane_normal: np.ndarray
    plane_offset: float
    plane_tolerance: float
    training_rgb_digest: str
    validity_mask: None = None
    adaptive_density: bool = False


def make_floor_recovery_static_spec(
    base_spec: StaticNotebookRunSpec,
) -> StaticNotebookRunSpec:
    """Build the bounded depth-plus-legacy-density training profile."""

    return make_ablation_static_spec(base_spec, FLOOR_RECOVERY_VARIANT)


def _validated_seed_arrays(
    floor_xyz: np.ndarray,
    floor_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(floor_xyz)
    rgb = np.asarray(floor_rgb)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) == 0:
        raise ValueError("floor_xyz must have shape (N, 3)")
    if rgb.shape != xyz.shape:
        raise ValueError("floor_rgb must align with floor_xyz")
    if not np.issubdtype(xyz.dtype, np.floating) or not np.isfinite(xyz).all():
        raise ValueError("floor_xyz must be finite floating point")
    if rgb.dtype != np.uint8:
        raise ValueError("floor_rgb must be uint8")
    return np.asarray(xyz, dtype=np.float32), rgb


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _validated_floor_artifact(
    artifact: FloorSeedArtifact,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    for path, label in (
        (artifact.npz_path, "floor seed archive"),
        (artifact.metadata_path, "floor seed metadata"),
        (artifact.plane_path, "floor plane metadata"),
        (artifact.holes_path, "floor hole metadata"),
    ):
        if not Path(path).is_file() or Path(path).is_symlink():
            raise ValueError(f"{label} must be a regular file")
    metadata = _read_json(artifact.metadata_path, "floor seed metadata")
    plane = _read_json(artifact.plane_path, "floor plane metadata")
    holes = _read_json(artifact.holes_path, "floor hole metadata")
    if metadata.get("schema") != FLOOR_RECOVERY_SCHEMA:
        raise ValueError("floor seed metadata schema mismatch")
    if plane.get("schema") != FLOOR_RECOVERY_SCHEMA:
        raise ValueError("floor plane metadata schema mismatch")
    if holes.get("schema") != FLOOR_RECOVERY_SCHEMA:
        raise ValueError("floor hole metadata schema mismatch")
    fingerprint = metadata.get("content_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or fingerprint != artifact.content_fingerprint
    ):
        raise ValueError("floor seed fingerprint mismatch")
    if metadata.get("npz_sha256") != _sha256(artifact.npz_path):
        raise ValueError("floor seed archive digest mismatch")
    try:
        with np.load(artifact.npz_path, allow_pickle=False) as archive:
            expected = {
                "xyz",
                "rgb",
                "confidence",
                "view_support",
                "source_frame_index",
                "hole_cell_id",
            }
            if set(archive.files) != expected:
                raise ValueError("floor seed archive keys mismatch")
            xyz = np.array(archive["xyz"], copy=True)
            rgb = np.array(archive["rgb"], copy=True)
            support = np.array(archive["view_support"], copy=True)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError) and "keys mismatch" in str(exc):
            raise
        raise ValueError("floor seed archive is invalid") from exc
    xyz, rgb = _validated_seed_arrays(xyz, rgb)
    if (
        len(xyz) != artifact.point_count
        or metadata.get("point_count") != artifact.point_count
        or support.dtype != np.uint16
        or support.shape != (len(xyz),)
        or np.any(support < 3)
    ):
        raise ValueError("floor seed archive violates its support contract")
    normal = np.asarray(plane.get("normal"), dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("floor plane normal is invalid")
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("floor plane normal is degenerate")
    normal /= norm
    try:
        offset = float(plane["offset"])
        tolerance = float(plane["inlier_tolerance"])
        cell_width = float(holes["cell_width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("floor geometry metadata is incomplete") from exc
    if not all(math.isfinite(value) for value in (offset, tolerance, cell_width)):
        raise ValueError("floor geometry metadata must be finite")
    if tolerance <= 0.0 or cell_width <= 0.0:
        raise ValueError("floor geometry scales must be positive")
    return xyz, rgb, normal, offset, tolerance, min(0.025, 0.5 * cell_width)


def _point_count(path: Path) -> int:
    count = 0
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            count += 1
    if count < 1:
        raise ValueError("accepted sparse model contains no points")
    return count


def prepare_floor_recovery_scene(
    artifacts: LearnedArtifacts,
    *,
    accepted_model_dir: Path,
    scene_dir: Path,
    floor_artifact: FloorSeedArtifact,
) -> PreparedFloorRecoveryScene:
    """Stage legacy RGB/depth and append only the verified floor seed rows."""

    scene = Path(scene_dir)
    evidence = _validate_evidence(artifacts, Path(accepted_model_dir), scene)
    xyz, rgb, normal, offset, tolerance, maximum_scale = (
        _validated_floor_artifact(floor_artifact)
    )
    for frame in evidence.original_frames:
        destination = scene / "frames" / frame.image_name
        if destination.resolve(strict=False) != frame.path.resolve(strict=True):
            _atomic_copy(frame.path, destination)
    depth_dir = scene / "depth"
    if os.path.lexists(depth_dir):
        raise FileExistsError("floor recovery depth destination must be fresh")
    depth_dir.mkdir()
    for frame, (source, _digest) in zip(
        evidence.original_frames,
        evidence.depth_sources,
        strict=True,
    ):
        _atomic_copy(source, depth_dir / f"{Path(frame.image_name).stem}_depth.npy")
    points_path = scene / "colmap" / "sparse" / "0" / "points3D.txt"
    original_count = _point_count(points_path)
    _augment_points3d(points_path, xyz, rgb)
    if _point_count(points_path) != original_count + len(xyz):
        raise RuntimeError("floor seed augmentation count mismatch")
    return PreparedFloorRecoveryScene(
        scene_dir=scene,
        original_frames=evidence.original_frames,
        quality_validity=evidence.validity,
        depth_dir=depth_dir,
        original_point_count=original_count,
        floor_xyz=xyz,
        floor_rgb=rgb,
        maximum_scale=maximum_scale,
        plane_normal=normal,
        plane_offset=offset,
        plane_tolerance=tolerance,
        training_rgb_digest=evidence.training_rgb_digest,
    )


def initialize_floor_seed_slice(
    trainer: Any,
    *,
    original_count: int,
    floor_xyz: np.ndarray,
    floor_rgb: np.ndarray,
    maximum_scale: float,
) -> None:
    """Verify and safely initialize only the appended floor Gaussian slice."""

    if type(original_count) is not int or original_count < 1:
        raise ValueError("original_count must be a positive plain integer")
    if (
        isinstance(maximum_scale, bool)
        or not isinstance(maximum_scale, (int, float))
        or not math.isfinite(float(maximum_scale))
        or float(maximum_scale) <= 0.0
    ):
        raise ValueError("maximum_scale must be finite and positive")
    xyz, rgb = _validated_seed_arrays(floor_xyz, floor_rgb)
    gs = getattr(trainer, "gs", None)
    required = ("means", "scales", "quats", "opacities", "sh_dc", "sh_rest")
    if gs is None or any(not hasattr(gs, name) for name in required):
        raise TypeError("trainer must expose a complete Gaussian model")
    expected_count = original_count + len(xyz)
    if int(gs.num_points) != expected_count:
        raise RuntimeError(
            "floor seed initialization count mismatch: "
            f"{gs.num_points} != {expected_count}"
        )
    seed_slice = slice(original_count, expected_count)
    actual_xyz = gs.means.detach()[seed_slice].float().cpu().numpy()
    if not np.allclose(actual_xyz, xyz, rtol=0.0, atol=1e-5):
        raise RuntimeError("floor seed slice does not match the verified artifact")
    c0 = 0.28209479177387814
    actual_rgb = (
        gs.sh_dc.detach()[seed_slice, 0, :].float().cpu().numpy() * c0 + 0.5
    )
    expected_rgb = rgb.astype(np.float32) / 255.0
    if not np.allclose(actual_rgb, expected_rgb, rtol=0.0, atol=1e-5):
        raise RuntimeError("floor seed colors do not match the verified artifact")
    tensors = tuple(getattr(gs, name) for name in required)
    if any(not bool(torch.isfinite(value.detach()).all()) for value in tensors):
        raise RuntimeError("Gaussian initialization contains non-finite values")
    with torch.no_grad():
        gs.opacities[seed_slice].fill_(-4.0)
        gs.scales[seed_slice].clamp_(max=math.log(float(maximum_scale)))
        gs.quats[seed_slice].zero_()
        gs.quats[seed_slice, 0] = 1.0
        gs.sh_rest[seed_slice].zero_()


class FloorDiagnosticProbe:
    """Measure survival and plane consistency over a bounded seed sample."""

    def __init__(
        self,
        *,
        floor_xyz: np.ndarray,
        plane_normal: np.ndarray,
        plane_offset: float,
        plane_tolerance: float,
        maximum_scale: float,
        sample_budget: int = 4_096,
    ) -> None:
        xyz, _ = _validated_seed_arrays(
            floor_xyz,
            np.zeros(np.asarray(floor_xyz).shape, dtype=np.uint8),
        )
        if len(xyz) > sample_budget:
            indices = np.linspace(0, len(xyz) - 1, sample_budget).round().astype(int)
            xyz = xyz[indices]
        self.floor_xyz = np.asarray(xyz, dtype=np.float64)
        self.normal = np.asarray(plane_normal, dtype=np.float64)
        self.offset = float(plane_offset)
        self.tolerance = float(plane_tolerance)
        self.radius = max(float(maximum_scale) * 2.0, self.tolerance)
        self.checkpoints: list[FloorCheckpointMetrics] = []

    def __call__(
        self,
        trainer: Any,
        iteration: int,
        _resolution: tuple[int, int],
        _sh_degree: int,
    ) -> Mapping[str, object]:
        means = trainer.gs.means.detach().float().cpu().numpy()
        opacities = trainer.gs.get_opacities.detach().float().cpu().numpy().reshape(-1)
        if means.ndim != 2 or means.shape[1:] != (3,) or len(means) != len(opacities):
            raise RuntimeError("floor diagnostic received an invalid Gaussian model")
        if not np.isfinite(means).all() or not np.isfinite(opacities).all():
            raise RuntimeError("floor diagnostic received non-finite Gaussians")
        distances, indices = cKDTree(means).query(self.floor_xyz, k=1)
        occupied = distances <= self.radius
        # The verified seeds intentionally start at sigmoid(-4) ~= 0.018.
        # They are spatially present at iteration zero but must not count as a
        # recovered floor until training raises them to useful opacity.
        visible = occupied & (opacities[indices] > 0.10)
        alpha_coverage = float(np.mean(visible))
        plane_distance = np.abs(means[indices] @ self.normal + self.offset)
        plane_error = (
            float(np.median(plane_distance[occupied]) / self.tolerance)
            if bool(np.any(occupied))
            else float("inf")
        )
        metric = FloorCheckpointMetrics(
            iteration=int(iteration),
            floor_alpha_coverage=alpha_coverage,
            residual_hole_fraction=1.0 - alpha_coverage,
            plane_depth_relative_error=plane_error,
            perturbed_depth_disagreement_ratio=1.0,
            occupied_hole_fraction=float(np.mean(occupied)),
        )
        self.checkpoints.append(metric)
        return asdict(metric)


class FloorRecoveryPipelineRunner:
    def __init__(
        self,
        artifacts: LearnedArtifacts,
        *,
        accepted_model_dir: Path,
        floor_artifact: FloorSeedArtifact,
        pipeline_runner: Callable[..., Mapping[str, object]],
        diagnostic_root: Path,
        seed: int,
        control_checkpoints: Mapping[int, AblationCheckpoint] | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.accepted_model_dir = Path(accepted_model_dir)
        self.floor_artifact = floor_artifact
        self.pipeline_runner = pipeline_runner
        self.diagnostic_root = Path(diagnostic_root)
        self.seed = int(seed)
        self.control_checkpoints = dict(control_checkpoints or {})
        self.collector: AblationDiagnosticCollector | None = None
        self.floor_probe: FloorDiagnosticProbe | None = None

    def __call__(self, **kwargs: object) -> Mapping[str, object]:
        if "video_path" not in kwargs:
            raise ValueError("floor recovery pipeline runner requires video_path")
        if "trainer_customizer" in kwargs or "trainer_train_kwargs" in kwargs:
            raise ValueError("floor recovery trainer injection cannot be overridden")
        scene_dir = Path(kwargs["video_path"]).parent
        prepared = prepare_floor_recovery_scene(
            self.artifacts,
            accepted_model_dir=self.accepted_model_dir,
            scene_dir=scene_dir,
            floor_artifact=self.floor_artifact,
        )
        self.floor_probe = FloorDiagnosticProbe(
            floor_xyz=prepared.floor_xyz,
            plane_normal=prepared.plane_normal,
            plane_offset=prepared.plane_offset,
            plane_tolerance=prepared.plane_tolerance,
            maximum_scale=prepared.maximum_scale,
        )
        self.collector = AblationDiagnosticCollector(
            scene_dir=scene_dir,
            frames=prepared.original_frames,
            validity=prepared.quality_validity,
            variant=FLOOR_RECOVERY_VARIANT,
            output_root=self.diagnostic_root,
            control_checkpoints=self.control_checkpoints,
            checkpoint_observer=self.floor_probe,
            stop_on_quality=False,
        )

        def customize(trainer: Any) -> None:
            initialize_floor_seed_slice(
                trainer,
                original_count=prepared.original_point_count,
                floor_xyz=prepared.floor_xyz,
                floor_rgb=prepared.floor_rgb,
                maximum_scale=prepared.maximum_scale,
            )

        train_kwargs: dict[str, object] = {
            "camera_generator": torch.Generator(device="cpu").manual_seed(self.seed),
            "diagnostic_iterations": FLOOR_RECOVERY_CHECKPOINTS,
            "diagnostic_callback": self.collector,
        }
        forwarded = dict(kwargs)
        forwarded["skip_foundation"] = True
        forwarded["trainer_customizer"] = customize
        forwarded["trainer_train_kwargs"] = train_kwargs
        status = self.pipeline_runner(**forwarded)
        if not isinstance(status, Mapping):
            raise TypeError("floor recovery pipeline runner must return a status mapping")
        return dict(status)


def run_floor_recovery_training(
    prepared: PreparedTrainingInput,
    base_spec: StaticNotebookRunSpec,
    reconstruction: LearnedReconstructionOutput,
    floor_artifact: FloorSeedArtifact,
    *,
    diagnostic_root: Path,
    seed: int = 1701,
    source_long_edge: int | None = None,
    control_checkpoints: Mapping[int, AblationCheckpoint] | None = None,
    pipeline_runner: Callable[..., Mapping[str, object]] | None = None,
    validated_training_runner: Callable[..., TrainingResult] | None = None,
) -> FloorTrainingResult:
    """Run one bounded floor-only recovery arm with the legacy trainer."""

    if prepared.reconstruction != reconstruction.bundle:
        raise ValueError("prepared and learned reconstruction contracts disagree")
    if pipeline_runner is None:
        from backend.pipeline import run_pipeline

        pipeline_runner = run_pipeline
    if validated_training_runner is None:
        from backend.static_pipeline.training import run_validated_training

        validated_training_runner = run_validated_training
    experiment_runner = FloorRecoveryPipelineRunner(
        reconstruction.artifacts,
        accepted_model_dir=reconstruction.accepted_model_dir,
        floor_artifact=floor_artifact,
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
    training = validated_training_runner(
        prepared,
        make_floor_recovery_static_spec(base_spec),
        **training_kwargs,
    )
    collector = experiment_runner.collector
    probe = experiment_runner.floor_probe
    if collector is None or probe is None:
        raise RuntimeError("floor recovery training did not install diagnostics")
    checkpoints = tuple(collector.checkpoints)
    floor_checkpoints = tuple(
        replace(
            floor_checkpoint,
            perturbed_depth_disagreement_ratio=(
                max(checkpoint.depth_median, checkpoint.perturbed_depth_median)
                / min(checkpoint.depth_median, checkpoint.perturbed_depth_median)
                if min(checkpoint.depth_median, checkpoint.perturbed_depth_median)
                > 0.0
                else 1_000_000.0
            ),
        )
        for checkpoint, floor_checkpoint in zip(
            collector.checkpoints,
            probe.checkpoints,
            strict=True,
        )
    )
    observed = tuple(checkpoint.iteration for checkpoint in checkpoints)
    floor_observed = tuple(checkpoint.iteration for checkpoint in floor_checkpoints)
    if observed != FLOOR_RECOVERY_CHECKPOINTS or floor_observed != observed:
        raise RuntimeError(
            "floor recovery training did not publish every checkpoint: "
            f"expected {FLOOR_RECOVERY_CHECKPOINTS}, observed {observed}, "
            f"floor {floor_observed}"
        )
    raw_ply_path = Path(training.raw_ply_path)
    metrics_path = raw_ply_path.parent.parent / "logs" / "metrics.jsonl"
    return FloorTrainingResult(
        checkpoints=checkpoints,
        floor_checkpoints=floor_checkpoints,
        raw_ply_path=raw_ply_path,
        metrics_path=metrics_path if metrics_path.is_file() else None,
        run_manifest_path=Path(training.run_manifest_path),
        status=dict(training.status),
    )

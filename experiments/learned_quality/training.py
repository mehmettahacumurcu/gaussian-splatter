from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.static_pipeline.training import (
    PipelineRunner,
    PreparedTrainingInput,
    TrainingResult,
    run_validated_training,
)

from .contracts import (
    FrameArtifact,
    LearnedArtifacts,
    LearnedReconstructionOutput,
    LearnedTrainingOutput,
)
from .density import AdaptiveDensityController


_MODEL_FILES = ("cameras.txt", "images.txt", "points3D.txt")
_SEED_FIELDS = ("xyz", "rgb", "confidence", "view_support")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


def _verify_digest(path: Path, expected: str, label: str) -> None:
    _require_file(path, label)
    if _sha256(path) != expected:
        raise ValueError(f"{label} hash mismatch: {path.name}")


def _frame_key(frame: FrameArtifact) -> tuple[str, str, str]:
    return frame.frame_id, frame.image_name, frame.sha256


@dataclass(frozen=True)
class _ValidatedEvidence:
    original_frames: tuple[FrameArtifact, ...]
    training_frames: tuple[FrameArtifact, ...]
    validity: tuple[torch.Tensor, ...]
    depth_sources: tuple[tuple[Path, str], ...]
    seed_xyz: np.ndarray
    seed_rgb: np.ndarray
    training_rgb_digest: str
    fallbacks: tuple[str, ...]
    source_model_snapshot: tuple[tuple[Path, str], ...]


def _validate_dense_seeds(dense: object) -> tuple[np.ndarray, np.ndarray]:
    path = Path(getattr(dense, "npz_path"))
    _verify_digest(path, getattr(dense, "npz_sha256"), "dense seed archive")
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != _SEED_FIELDS:
            raise ValueError("dense seed archive fields are not exact")
        xyz = np.asarray(archive["xyz"])
        rgb = np.asarray(archive["rgb"])
        confidence = np.asarray(archive["confidence"])
        support = np.asarray(archive["view_support"])
    count = int(getattr(dense, "point_count"))
    if (
        xyz.dtype != np.float32
        or xyz.shape != (count, 3)
        or rgb.dtype != np.uint8
        or rgb.shape != (count, 3)
        or confidence.dtype != np.float32
        or confidence.shape != (count,)
        or support.dtype != np.uint16
        or support.shape != (count,)
        or not np.isfinite(xyz).all()
        or not np.isfinite(confidence).all()
        or np.any(confidence < 0.0)
        or np.any(support < 3)
    ):
        raise ValueError("dense seed archive violates its typed contract")
    return np.array(xyz, copy=True), np.array(rgb, copy=True)


def _validate_evidence(
    artifacts: LearnedArtifacts,
    accepted_model_dir: Path,
    scene_dir: Path,
) -> _ValidatedEvidence:
    photometric = artifacts.photometric
    masks = artifacts.masks
    depth = artifacts.depth
    dense = artifacts.dense_seeds
    if photometric is None or masks is None or depth is None or dense is None:
        raise ValueError(
            "training requires photometric, mask, depth, and dense evidence"
        )
    if getattr(depth, "dense_seeds", None) is not dense:
        depth_dense = getattr(depth, "dense_seeds", None)
        if depth_dense is None or (
            Path(getattr(depth_dense, "npz_path")) != Path(getattr(dense, "npz_path"))
            or getattr(depth_dense, "npz_sha256") != getattr(dense, "npz_sha256")
        ):
            raise ValueError("depth and learned artifacts disagree on dense seeds")

    original_frames = tuple(getattr(photometric, "original_frames"))
    training_frames = tuple(getattr(photometric, "training_frames"))
    mask_frames = tuple(getattr(masks, "frames"))
    depth_frames = tuple(getattr(depth, "frames"))
    if not original_frames or not (
        len(original_frames)
        == len(training_frames)
        == len(mask_frames)
        == len(depth_frames)
    ):
        raise ValueError("learned evidence frame counts must match and be non-empty")
    if len({frame.image_name for frame in original_frames}) != len(original_frames):
        raise ValueError("learned evidence contains duplicate frame names")

    decision = getattr(photometric, "decision")
    if decision not in {"accepted", "rejected"}:
        raise ValueError("photometric decision must be accepted or rejected")
    fallbacks = () if decision == "accepted" else ("photometric_rejected_original_rgb",)

    scene_frames = scene_dir / "frames"
    validity: list[torch.Tensor] = []
    depths: list[tuple[Path, str]] = []
    for index, original in enumerate(original_frames):
        if not isinstance(original, FrameArtifact):
            raise ValueError("photometric original frame contract is invalid")
        _verify_digest(original.path, original.sha256, "original RGB")
        copied = scene_frames / original.image_name
        _verify_digest(copied, original.sha256, "copied original RGB")

        training = training_frames[index]
        fused = mask_frames[index]
        validated_depth = depth_frames[index]
        if (
            _frame_key(training)[:2] != _frame_key(original)[:2]
            or _frame_key(getattr(fused, "frame")) != _frame_key(original)
            or _frame_key(getattr(validated_depth, "frame")) != _frame_key(original)
        ):
            raise ValueError("learned artifacts require an exact frame join")
        _verify_digest(training.path, training.sha256, "training RGB")

        validity_path = Path(getattr(fused, "training_validity_path"))
        _verify_digest(
            validity_path,
            getattr(fused, "training_validity_sha256"),
            "training validity",
        )
        with Image.open(validity_path) as opened:
            if opened.mode != "L":
                raise ValueError("training validity must be an 8-bit grayscale PNG")
            validity_array = np.asarray(opened, dtype=np.uint8)
        with Image.open(original.path) as opened:
            expected_shape = (opened.height, opened.width)
        if validity_array.shape != expected_shape:
            raise ValueError("training validity dimensions must match RGB")
        validity.append(torch.from_numpy(validity_array.copy()).float().div_(255.0))

        depth_path = Path(getattr(validated_depth, "depth_path"))
        depth_digest = getattr(validated_depth, "depth_sha256")
        _verify_digest(depth_path, depth_digest, "validated depth")
        depth_array = np.load(depth_path, allow_pickle=False)
        if (
            depth_array.dtype != np.float32
            or depth_array.shape != expected_shape
            or not np.isfinite(depth_array).all()
            or np.any(depth_array < 0.0)
        ):
            raise ValueError("validated depth violates its array contract")
        depths.append((depth_path, depth_digest))

    if decision == "rejected" and tuple(map(_frame_key, training_frames)) != tuple(
        map(_frame_key, original_frames)
    ):
        raise ValueError("rejected photometric evidence must reuse original RGB")

    source_snapshot: list[tuple[Path, str]] = []
    copied_model = scene_dir / "colmap" / "sparse" / "0"
    for name in _MODEL_FILES:
        source = accepted_model_dir / name
        copied = copied_model / name
        _require_file(source, f"accepted model {name}")
        digest = _sha256(source)
        _verify_digest(copied, digest, f"copied accepted model {name}")
        source_snapshot.append((source, digest))
    snapshotted_paths = {path for path, _ in source_snapshot}
    for frame in (*original_frames, *training_frames):
        if frame.path not in snapshotted_paths:
            source_snapshot.append((frame.path, frame.sha256))
            snapshotted_paths.add(frame.path)

    xyz, rgb = _validate_dense_seeds(dense)
    training_digest = getattr(photometric, "training_rgb_digest")
    if not isinstance(training_digest, str) or len(training_digest) != 64:
        raise ValueError("training RGB digest is invalid")
    return _ValidatedEvidence(
        original_frames=original_frames,
        training_frames=training_frames,
        validity=tuple(validity),
        depth_sources=tuple(depths),
        seed_xyz=xyz,
        seed_rgb=rgb,
        training_rgb_digest=training_digest,
        fallbacks=fallbacks,
        source_model_snapshot=tuple(source_snapshot),
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary, follow_symlinks=False)
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _augment_points3d(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    original = path.read_bytes()
    maximum_id = 0
    for raw_line in original.decode("utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            maximum_id = max(maximum_id, int(line.split()[0]))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(original)
            if original and not original.endswith((b"\n", b"\r")):
                stream.write(b"\n")
            for offset, (point, color) in enumerate(
                zip(xyz, rgb, strict=True), start=1
            ):
                row = (
                    f"{maximum_id + offset} "
                    f"{float(point[0]):.9g} {float(point[1]):.9g} {float(point[2]):.9g} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
                )
                stream.write(row.encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    if not path.read_bytes().startswith(original):
        raise RuntimeError("dense augmentation did not preserve sparse points")


def _coverage_indices(count: int, budget: int = 8) -> tuple[int, ...]:
    if count <= budget:
        return tuple(range(count))
    return tuple(
        sorted({round(index * (count - 1) / (budget - 1)) for index in range(budget)})
    )


def _make_quality_probe(
    scene_dir: Path,
    frames: tuple[FrameArtifact, ...],
    validity: tuple[torch.Tensor, ...],
) -> Callable[[Any, int, tuple[int, int], int], float]:
    from backend.preprocess.frame_alignment import join_registered_frames
    from backend.preprocess.parse_colmap import parse_cameras_from_model

    model_dir = scene_dir / "colmap" / "sparse" / "0"
    cameras = parse_cameras_from_model(model_dir)
    registered = join_registered_frames(scene_dir / "frames", cameras)
    by_name = {
        frame.image_name: mask for frame, mask in zip(frames, validity, strict=True)
    }
    if tuple(frame.image_name for frame in registered) != tuple(
        frame.image_name for frame in frames
    ):
        raise ValueError("quality probe cameras require an exact frame join")
    probe_indices = _coverage_indices(len(registered))

    def probe(
        trainer: Any,
        iteration: int,
        resolution: tuple[int, int],
        sh_degree: int,
    ) -> float:
        del iteration
        from backend.model.renderer import render_view
        from backend.model.trainer import (
            _resize_validity_mask,
            load_frame_tensor,
            masked_psnr,
        )

        width, height = resolution
        scores: list[float] = []
        with torch.no_grad():
            for index in probe_indices:
                registered_frame = registered[index]
                gt = load_frame_tensor(registered_frame.image_path, (width, height)).to(
                    trainer.device
                )
                mask = _resize_validity_mask(
                    by_name[registered_frame.image_name].to(trainer.device),
                    (height, width),
                )
                source_height, source_width = by_name[registered_frame.image_name].shape
                K = torch.from_numpy(registered_frame.K).float().to(trainer.device)
                K = K.clone()
                K[0, :] *= width / source_width
                K[1, :] *= height / source_height
                w2c = torch.from_numpy(registered_frame.w2c).float().to(trainer.device)
                if getattr(trainer, "static_mode", False):
                    means = trainer.gs.means
                    quats = F.normalize(trainer.gs.quats, dim=-1)
                    scales = trainer.gs.get_scales
                else:
                    timestamp = index / max(len(registered) - 1, 1)
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
                value = masked_psnr(rendered[..., :3], gt, mask)
                scores.append(100.0 if math.isinf(value) else value)
        if not scores or not all(math.isfinite(value) for value in scores):
            raise ValueError("fixed-view quality probe produced invalid PSNR")
        return float(sum(scores) / len(scores))

    return probe


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


class ExperimentPipelineRunner:
    def __init__(
        self,
        artifacts: LearnedArtifacts,
        *,
        accepted_model_dir: Path,
        pipeline_runner: PipelineRunner,
        density_factory: Callable[[], Any],
    ) -> None:
        self.artifacts = artifacts
        self.accepted_model_dir = Path(accepted_model_dir)
        self.pipeline_runner = pipeline_runner
        self.density_factory = density_factory
        self.density_history_path: Path | None = None
        self.final_gaussian_count = 0
        self.training_rgb_digest = ""
        self.fallbacks: tuple[str, ...] = ()

    def __call__(self, **kwargs: object) -> Mapping[str, object]:
        if "video_path" not in kwargs:
            raise ValueError("experiment pipeline runner requires video_path")
        if "trainer_customizer" in kwargs or "trainer_train_kwargs" in kwargs:
            raise ValueError("experiment trainer injection cannot be overridden")
        scene_dir = Path(kwargs["video_path"]).parent
        evidence = _validate_evidence(
            self.artifacts, self.accepted_model_dir, scene_dir
        )

        for frame in evidence.training_frames:
            destination = scene_dir / "frames" / frame.image_name
            if frame.path != destination:
                _atomic_copy(frame.path, destination)
            _verify_digest(destination, frame.sha256, "installed training RGB")
        depth_dir = scene_dir / "depth"
        if os.path.lexists(depth_dir):
            raise FileExistsError("learned depth destination must be fresh")
        depth_dir.mkdir()
        for frame, (source, digest) in zip(
            evidence.original_frames, evidence.depth_sources, strict=True
        ):
            destination = depth_dir / f"{Path(frame.image_name).stem}_depth.npy"
            _atomic_copy(source, destination)
            _verify_digest(destination, digest, "installed validated depth")

        points_path = scene_dir / "colmap" / "sparse" / "0" / "points3D.txt"
        _augment_points3d(points_path, evidence.seed_xyz, evidence.seed_rgb)
        quality_probe = _make_quality_probe(
            scene_dir, evidence.original_frames, evidence.validity
        )
        state: dict[str, object] = {}

        def customize(trainer: Any) -> None:
            density = self.density_factory()
            if not hasattr(density, "accumulate_view") or not hasattr(
                density, "step_at"
            ):
                raise TypeError("density_factory must return an experiment controller")
            trainer.density = density
            state["trainer"] = trainer
            state["density"] = density

        forwarded = dict(kwargs)
        forwarded["skip_foundation"] = True
        forwarded["trainer_customizer"] = customize
        forwarded["trainer_train_kwargs"] = {
            "validity_mask": evidence.validity,
            "density_quality_probe": quality_probe,
        }
        try:
            status = self.pipeline_runner(**forwarded)
        finally:
            for source, digest in evidence.source_model_snapshot:
                _verify_digest(source, digest, "learned source artifact after training")
        if not isinstance(status, Mapping):
            raise TypeError("experiment pipeline runner must return a status mapping")
        if "trainer" not in state or "density" not in state:
            raise RuntimeError(
                "pipeline did not apply the experiment trainer customizer"
            )
        trainer = state["trainer"]
        density = state["density"]
        self.final_gaussian_count = int(trainer.gs.num_points)
        self.training_rgb_digest = evidence.training_rgb_digest
        self.fallbacks = evidence.fallbacks
        self.density_history_path = (
            scene_dir / "output" / "diagnostics" / "density_history.json"
        )
        _write_json_atomic(self.density_history_path, density.history)
        return dict(status)


def make_experiment_pipeline_runner(
    artifacts: LearnedArtifacts,
    *,
    accepted_model_dir: Path,
    pipeline_runner: PipelineRunner | None = None,
    density_factory: Callable[[], Any] = AdaptiveDensityController,
) -> ExperimentPipelineRunner:
    if pipeline_runner is None:
        from backend.pipeline import run_pipeline

        pipeline_runner = run_pipeline
    return ExperimentPipelineRunner(
        artifacts,
        accepted_model_dir=accepted_model_dir,
        pipeline_runner=pipeline_runner,
        density_factory=density_factory,
    )


def run_learned_training(
    prepared: PreparedTrainingInput,
    spec: object,
    reconstruction: LearnedReconstructionOutput,
    *,
    source_long_edge: int | None = None,
    pipeline_runner: PipelineRunner | None = None,
    density_factory: Callable[[], Any] = AdaptiveDensityController,
    validated_training_runner: Callable[..., TrainingResult] = run_validated_training,
) -> LearnedTrainingOutput:
    if prepared.reconstruction != reconstruction.bundle:
        raise ValueError("prepared and learned reconstruction contracts disagree")
    experiment_runner = make_experiment_pipeline_runner(
        reconstruction.artifacts,
        accepted_model_dir=reconstruction.accepted_model_dir,
        pipeline_runner=pipeline_runner,
        density_factory=density_factory,
    )
    result = validated_training_runner(
        prepared,
        spec,
        source_long_edge=source_long_edge,
        pipeline_runner=experiment_runner,
    )
    if experiment_runner.density_history_path is None:
        raise RuntimeError("learned density history was not persisted")
    return LearnedTrainingOutput(
        raw_ply_path=result.raw_ply_path,
        status=result.status,
        resolved_config=result.resolved_config,
        run_manifest_path=result.run_manifest_path,
        density_history_path=experiment_runner.density_history_path,
        final_gaussian_count=experiment_runner.final_gaussian_count,
        training_rgb_digest=experiment_runner.training_rgb_digest,
        fallbacks=experiment_runner.fallbacks,
    )

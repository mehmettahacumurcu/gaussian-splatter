from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from backend.preprocess.parse_colmap import parse_cameras_from_model
from backend.static_pipeline.reconstruction import measure_models
from backend.static_pipeline.runner import HardwareInfo, SelectionOutput
from backend.static_pipeline.selection import plan_backfill

from .contracts import (
    FrameArtifact,
    LearnedArtifacts,
    LearnedReconstructionOutput,
    StageRecord,
)
from .da3 import (
    DA3Frame,
    FinalPoseCamera,
    run_anchor_inference,
    run_metric_sky,
    run_pose_conditioned_depth,
)
from .depth import DenseSeedPolicy, validate_depth_and_fuse_seeds
from .flow import (
    FlowGatePolicy,
    RigidFrameEvidence,
    RigidSceneEvidence,
    StaticTrack,
    TrackObservation,
    run_motion_evidence,
)
from .geometry import (
    HybridGeometryInputs,
    geometry_frame_set_digest,
    make_classical_candidate_runner,
    make_hybrid_candidate_runner,
    run_geometry_comparison,
)
from .lifecycle import release_cuda_model
from .masks import MaskFusionPolicy, fuse_evidence_masks
from .model_adapters import (
    Sam2ImageAdapter,
    SeaRaftTorchAdapter,
    TransformersGroundingDinoAdapter,
    load_da3_model,
)
from .photometric import PhotometricPolicy, fit_and_validate_photometric_transforms
from .segmentation import SemanticPolicy, run_semantic_evidence


_SOURCE_ROOT = Path("/content/learned-sources")
_CHECKPOINT_ROOT = Path("/content/learned-checkpoints")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_rows(rows: object) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _checkpoint(repo_id: str) -> Path:
    return _CHECKPOINT_ROOT / repo_id.replace("/", "--")


def _validate_model_manifest() -> Path:
    path = Path(
        os.environ.get(
            "LEARNED_MODEL_MANIFEST", "/content/learned-env/model_manifest.json"
        )
    ).resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("verified learned model_manifest.json is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("learned model manifest schema is unsupported")
    expected_sources = {
        "da3": _SOURCE_ROOT / "da3",
        "sam2": _SOURCE_ROOT / "sam2",
        "sea-raft": _SOURCE_ROOT / "sea-raft",
    }
    actual_sources = {
        row.get("name"): Path(row.get("path", ""))
        for row in payload.get("sources", ())
        if isinstance(row, dict)
    }
    if actual_sources != expected_sources:
        raise ValueError("learned model manifest source roots are not exact")
    return path


def _frame_artifacts(selection: SelectionOutput) -> tuple[FrameArtifact, ...]:
    result = []
    for record in selection.manifest.selected_frames:
        path = (selection.frames_dir / record.output_name).resolve()
        if not path.is_file() or path.is_symlink() or _sha256(path) != record.sha256:
            raise ValueError(f"selected frame changed: {record.output_name}")
        result.append(
            FrameArtifact(
                image_name=record.output_name,
                frame_id=record.frame_id,
                path=path,
                sha256=record.sha256,
            )
        )
    return tuple(result)


def _resize_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(values, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def _materialize_metric(
    frames: tuple[FrameArtifact, ...], metric: object, output_root: Path
) -> tuple[tuple[Path, str], tuple[FrameArtifact, ...]]:
    output_root.mkdir(parents=True, exist_ok=False)
    depth_rows: list[tuple[Path, str]] = []
    sky_rows: list[FrameArtifact] = []
    for frame, prediction in zip(frames, metric.artifacts, strict=True):
        with Image.open(frame.path) as opened:
            size = opened.size
        depth = np.load(prediction.depth_path, allow_pickle=False)
        if depth.shape != (size[1], size[0]):
            depth = _resize_float(depth, size)
        depth_path = output_root / f"{Path(frame.image_name).stem}.depth.npy"
        np.save(depth_path, np.asarray(depth, dtype=np.float32), allow_pickle=False)
        depth_rows.append((depth_path, _sha256(depth_path)))
        if prediction.sky_path is None:
            raise ValueError("DA3Metric omitted a sky proposal")
        sky = np.load(prediction.sky_path, allow_pickle=False)
        if sky.shape != (size[1], size[0]):
            sky = np.asarray(
                Image.fromarray(np.asarray(sky, dtype=np.uint8), mode="L").resize(
                    size, Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            )
        sky_path = output_root / f"{Path(frame.image_name).stem}.sky.npy"
        np.save(sky_path, np.asarray(sky, dtype=np.bool_), allow_pickle=False)
        sky_rows.append(
            FrameArtifact(
                image_name=frame.image_name,
                frame_id=frame.frame_id,
                path=sky_path,
                sha256=_sha256(sky_path),
            )
        )
    return tuple(depth_rows), tuple(sky_rows)


def _image_point_rows(model_dir: Path):
    rows = (model_dir / "images.txt").read_text(encoding="utf-8").splitlines()
    content = [line.strip() for line in rows if not line.startswith("#")]
    content = [line for line in content if line or line == ""]
    result = {}
    index = 0
    while index < len(content):
        if not content[index]:
            index += 1
            continue
        header = content[index].split(maxsplit=9)
        points = content[index + 1].split() if index + 1 < len(content) else []
        image_id = int(header[0])
        coordinates = {}
        for point_index in range(0, len(points), 3):
            point_id = int(points[point_index + 2])
            if point_id >= 0:
                coordinates[point_index // 3] = (
                    float(points[point_index]),
                    float(points[point_index + 1]),
                    point_id,
                )
        result[image_id] = (header[9], coordinates)
        index += 2
    return result


def _static_tracks(
    model_dir: Path, frames: tuple[FrameArtifact, ...]
) -> tuple[StaticTrack, ...]:
    by_name = {frame.image_name: frame for frame in frames}
    images = _image_point_rows(model_dir)
    tracks = []
    for raw in (model_dir / "points3D.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        point_id = int(parts[0])
        error = float(parts[7])
        observations = []
        observed_frame_ids: set[str] = set()
        ambiguous_track = False
        for offset in range(8, len(parts), 2):
            image_id = int(parts[offset])
            point_index = int(parts[offset + 1])
            image = images.get(image_id)
            if image is None or image[0] not in by_name:
                continue
            coordinate = image[1].get(point_index)
            if coordinate is None or coordinate[2] != point_id:
                continue
            frame_id = by_name[image[0]].frame_id
            if frame_id in observed_frame_ids:
                ambiguous_track = True
                break
            observed_frame_ids.add(frame_id)
            observations.append(
                TrackObservation(
                    frame_id=frame_id,
                    x=coordinate[0],
                    y=coordinate[1],
                )
            )
        if not ambiguous_track and len(observations) >= 2:
            tracks.append(
                StaticTrack(
                    track_id=point_id,
                    xyz=tuple(float(value) for value in parts[1:4]),
                    mean_reprojection_error=error,
                    observations=tuple(observations),
                )
            )
    return tuple(tracks)


def _rigid_scene(
    frames: tuple[FrameArtifact, ...],
    model_dir: Path,
    depths: tuple[tuple[Path, str], ...],
) -> RigidSceneEvidence:
    cameras = parse_cameras_from_model(model_dir)
    records = []
    geometry_rows = []
    for frame, (depth_path, depth_digest) in zip(frames, depths, strict=True):
        with Image.open(frame.path) as opened:
            width, height = opened.size
        camera = cameras.get(frame.image_name)
        if camera is None:
            records.append(
                RigidFrameEvidence(frame, width, height, False, None, None, None, None)
            )
            geometry_rows.append((frame.frame_id, False))
            continue
        K = np.asarray(camera["K"], dtype=np.float64).copy()
        K[0, :] *= width / int(camera["width"])
        K[1, :] *= height / int(camera["height"])
        w2c = np.asarray(camera["w2c"], dtype=np.float64)
        records.append(
            RigidFrameEvidence(
                frame=frame,
                width=width,
                height=height,
                registered=True,
                w2c_4x4=tuple(float(value) for value in w2c.reshape(-1)),
                pinhole_fx_fy_cx_cy=(
                    float(K[0, 0]),
                    float(K[1, 1]),
                    float(K[0, 2]),
                    float(K[1, 2]),
                ),
                depth_path=depth_path,
                depth_sha256=depth_digest,
            )
        )
        geometry_rows.append((frame.frame_id, tuple(float(v) for v in w2c.reshape(-1))))
    model_hashes = tuple(
        (name, _sha256(model_dir / name))
        for name in ("cameras.txt", "images.txt", "points3D.txt")
    )
    return RigidSceneEvidence(
        frames=tuple(records),
        static_tracks=_static_tracks(model_dir, frames),
        geometry_digest=_digest_rows((model_hashes, geometry_rows)),
        depth_digest=_digest_rows(tuple(digest for _, digest in depths)),
    )


def _final_cameras(
    frames: tuple[FrameArtifact, ...], model_dir: Path
) -> tuple[FinalPoseCamera, ...]:
    cameras = parse_cameras_from_model(model_dir)
    available = [
        index for index, frame in enumerate(frames) if frame.image_name in cameras
    ]
    if not available:
        raise ValueError("winning geometry registered no final-pose cameras")
    result = []
    for index, frame in enumerate(frames):
        source_index = (
            index
            if index in available
            else min(available, key=lambda item: abs(item - index))
        )
        camera = cameras[frames[source_index].image_name]
        with Image.open(frame.path) as opened:
            width, height = opened.size
        K = np.asarray(camera["K"], dtype=np.float64).copy()
        K[0, :] *= width / int(camera["width"])
        K[1, :] *= height / int(camera["height"])
        result.append(
            FinalPoseCamera(
                image_name=frame.image_name,
                frame_id=frame.frame_id,
                width=width,
                height=height,
                w2c=tuple(
                    tuple(float(value) for value in row)
                    for row in np.asarray(camera["w2c"], dtype=np.float64)
                ),
                intrinsics=tuple(tuple(float(value) for value in row) for row in K),
            )
        )
    return tuple(result)


def _run_evidence_cycle(
    selection: SelectionOutput,
    frames: tuple[FrameArtifact, ...],
    hardware: HardwareInfo,
    output_root: Path,
) -> LearnedArtifacts:
    output_root.mkdir(parents=True, exist_ok=False)
    da3_frames = tuple(
        DA3Frame(frame.image_name, frame.frame_id, frame.path) for frame in frames
    )
    base = load_da3_model(_SOURCE_ROOT / "da3", _checkpoint("depth-anything/DA3-BASE"))
    try:
        anchors = run_anchor_inference(
            base, da3_frames, output_root / "da3-anchor", vram_gb=hardware.vram_gb
        )
    finally:
        release_cuda_model(base)
        del base

    classical_runner = make_classical_candidate_runner(
        use_gpu=hardware.colmap_gpu_sift is True
    )
    pre_attempt = classical_runner(
        selection.manifest,
        frames,
        output_root / "classical-prepass",
        0,
        geometry_frame_set_digest(selection.manifest, frames),
    )
    measured = tuple(measure_models(pre_attempt.model_dirs, selection.manifest))
    if not measured:
        raise RuntimeError("classical prepass produced no readable geometry")
    pre_model = max(
        measured, key=lambda item: (item.registered_count, item.sparse_point_count)
    ).model_dir

    metric_model = load_da3_model(
        _SOURCE_ROOT / "da3", _checkpoint("depth-anything/DA3METRIC-LARGE")
    )
    try:
        metric = run_metric_sky(
            metric_model,
            da3_frames,
            anchors.shared_camera,
            output_root / "da3-metric",
            initial_batch_size=12,
            retry_batch_size=6,
            release_model=release_cuda_model,
            retry_model_factory=lambda: load_da3_model(
                _SOURCE_ROOT / "da3", _checkpoint("depth-anything/DA3METRIC-LARGE")
            ),
        )
    finally:
        release_cuda_model(metric_model)
        del metric_model
    depths, sky = _materialize_metric(frames, metric, output_root / "metric-native")
    scene = _rigid_scene(frames, pre_model, depths)

    semantic = run_semantic_evidence(
        frames,
        output_root / "semantic",
        policy=SemanticPolicy(0.30, 0.25, 32, 3),
        detector_factory=lambda: TransformersGroundingDinoAdapter(
            _checkpoint("IDEA-Research/grounding-dino-tiny")
        ),
        sam_factory=lambda: Sam2ImageAdapter(
            _SOURCE_ROOT / "sam2", _checkpoint("facebook/sam2.1-hiera-large")
        ),
        initial_batch_size=8,
        retry_batch_size=4,
        release_model=release_cuda_model,
    )
    flow = run_motion_evidence(
        frames,
        scene,
        output_root / "flow",
        policy=FlowGatePolicy(0.03, 0.005, 1.5, 0.01, 3.0, 3, 1.5, 3.0),
        model_factory=lambda: SeaRaftTorchAdapter(
            _SOURCE_ROOT / "sea-raft",
            _checkpoint("MemorySlices/Tartan-C-T-TSKH-spring540x960-M"),
        ),
        initial_pair_batch_size=2,
        retry_pair_batch_size=1,
        release_model=release_cuda_model,
    )
    masks = fuse_evidence_masks(
        frames,
        semantic,
        flow,
        sky,
        scene,
        output_root / "masks",
        policy=MaskFusionPolicy(
            0.003, 0.001, 0.0002, 0.20, 0.45, 0.80, 0.003, 0.02, 0.01
        ),
    )
    stages = (
        StageRecord(
            "da3_anchor", "accepted", details={"anchors": len(anchors.anchor_indices)}
        ),
        StageRecord(
            "classical_prepass",
            "accepted",
            details={"registered": max(item.registered_count for item in measured)},
        ),
        StageRecord(
            "da3_metric_sky", "accepted", details={"frames": len(metric.artifacts)}
        ),
        *semantic.stage_records,
        *flow.stage_records,
        StageRecord(
            "mask_fusion", "accepted", details={"digest": masks.mask_set_digest}
        ),
    )
    return LearnedArtifacts(
        da3=anchors,
        semantic=semantic,
        flow=flow,
        masks=masks,
        model_manifest_path=_validate_model_manifest(),
        stage_records=stages,
    )


def run_learned_reconstruction(
    selection: SelectionOutput,
    *,
    spec: object,
    hardware: HardwareInfo,
    output_root: Path,
) -> LearnedReconstructionOutput:
    del spec
    _validate_model_manifest()
    frames = _frame_artifacts(selection)
    active = {
        "selection": selection,
        "frames": frames,
        "artifacts": _run_evidence_cycle(
            selection, frames, hardware, output_root.parent / "learned-evidence-0"
        ),
    }
    classical = make_classical_candidate_runner(
        use_gpu=hardware.colmap_gpu_sift is True
    )

    def hybrid(manifest, candidate_frames, attempt_dir, attempt_index, frame_digest):
        runner = make_hybrid_candidate_runner(
            HybridGeometryInputs(
                anchors=active["artifacts"].da3,
                masks=active["artifacts"].masks,
            ),
            use_gpu=hardware.colmap_gpu_sift is True,
        )
        return runner(
            manifest, candidate_frames, attempt_dir, attempt_index, frame_digest
        )

    def backfill(manifest, uncovered, *, max_per_interval):
        backfill_root = output_root.parent / "selection-backfill"
        new_manifest = plan_backfill(
            selection.inventory,
            manifest,
            uncovered,
            backfill_root,
            max_per_interval=max_per_interval,
        )
        new_selection = SelectionOutput(
            inventory=selection.inventory,
            manifest=new_manifest,
            frames_dir=backfill_root,
            source_manifest_path=selection.source_manifest_path,
        )
        new_frames = _frame_artifacts(new_selection)
        active.update(
            selection=new_selection,
            frames=new_frames,
            artifacts=_run_evidence_cycle(
                new_selection,
                new_frames,
                hardware,
                output_root.parent / "learned-evidence-1",
            ),
        )
        return new_manifest, new_frames

    comparison = run_geometry_comparison(
        selection.manifest,
        frames,
        output_root=output_root,
        artifacts=active["artifacts"],
        classical_runner=classical,
        hybrid_runner=hybrid,
        materialize_backfill=backfill,
    )
    final_frames = active["frames"]
    artifacts = active["artifacts"]
    metric_artifacts = artifacts.da3
    # Use metric-native depth already bound into the motion evidence scene.
    rigid_frames = tuple(artifacts.flow.frames)
    del metric_artifacts, rigid_frames
    metric_dir = (
        output_root.parent
        / (
            "learned-evidence-1"
            if comparison.selected_manifest != selection.manifest
            else "learned-evidence-0"
        )
        / "metric-native"
    )
    depths = tuple(
        (metric_dir / f"{Path(frame.image_name).stem}.depth.npy", "")
        for frame in final_frames
    )
    depths = tuple((path, _sha256(path)) for path, _ in depths)
    final_scene = _rigid_scene(final_frames, comparison.accepted_model_dir, depths)
    photometric = fit_and_validate_photometric_transforms(
        final_frames,
        final_scene,
        artifacts.masks,
        output_root.parent / "photometric",
        reference_frame_id=final_frames[len(final_frames) // 2].frame_id,
        policy=PhotometricPolicy(0.03, 0.10, 5),
    )
    base = load_da3_model(_SOURCE_ROOT / "da3", _checkpoint("depth-anything/DA3-BASE"))
    try:
        final_depth = run_pose_conditioned_depth(
            base,
            tuple(
                DA3Frame(frame.image_name, frame.frame_id, frame.path)
                for frame in final_frames
            ),
            _final_cameras(final_frames, comparison.accepted_model_dir),
            output_root.parent / "final-pose-depth",
        )
    finally:
        release_cuda_model(base)
        del base
    validated_depth = validate_depth_and_fuse_seeds(
        final_frames,
        final_depth,
        artifacts.masks,
        comparison.accepted_model_dir,
        output_root.parent / "validated-depth",
        policy=DenseSeedPolicy(0.20, 0.05, 0.003, 0.001, 2),
    )
    final_artifacts = replace(
        artifacts,
        photometric=photometric,
        depth=validated_depth,
        dense_seeds=validated_depth.dense_seeds,
        stage_records=(
            *artifacts.stage_records,
            StageRecord(
                "geometry_comparison",
                "accepted",
                details={"candidates": len(comparison.geometry_candidates)},
            ),
            StageRecord(
                "photometric_validation",
                photometric.decision,
                details={"training_rgb_digest": photometric.training_rgb_digest},
            ),
            StageRecord(
                "final_pose_depth",
                "accepted",
                details={"frames": len(final_depth.artifacts)},
            ),
            StageRecord(
                "dense_seed_fusion",
                "accepted",
                details={"points": validated_depth.dense_seeds.point_count},
            ),
        ),
    )
    return LearnedReconstructionOutput(
        bundle=comparison.bundle,
        frames_dir=comparison.frames_dir,
        artifacts=final_artifacts,
        geometry_candidates=comparison.geometry_candidates,
    )

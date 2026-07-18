from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from backend.preprocess.parse_colmap import parse_cameras_from_model
from backend.static_pipeline.colmap import (
    _colmap_version_from_help,
    _resolve_colmap_executable,
)
from backend.static_pipeline.progress import current_stage_reporter
from backend.static_pipeline.reconstruction import measure_models
from backend.static_pipeline.runner import HardwareInfo, SelectionOutput
from backend.static_pipeline.selection import plan_backfill

from .contracts import (
    FrameArtifact,
    LearnedArtifacts,
    LearnedReconstructionOutput,
    StageRecord,
)
from .cache import (
    CheckpointInputs,
    CheckpointKind,
    LearnedCheckpointStore,
    checkpoint_fingerprint,
    producer_code_digest,
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
from .milestones import (
    BaseEvidenceState,
    FinalPretrainingState,
    GeometryMilestoneState,
    LEGACY_COLMAP_PRODUCER_DIGESTS,
    MasksMilestoneState,
    MilestoneRef,
    MilestoneSession,
    MilestoneState,
    MotionMilestoneState,
    SemanticMilestoneState,
    assemble_learned_reconstruction,
    compatible_colmap_fingerprints,
)
from .model_adapters import (
    Sam2ImageAdapter,
    SeaRaftTorchAdapter,
    TransformersGroundingDinoAdapter,
    load_da3_model,
)
from .photometric import PhotometricPolicy, fit_and_validate_photometric_transforms
from .segmentation import SemanticPolicy, run_semantic_evidence
from .tracks import TrackQualificationPolicy, qualify_colmap_static_tracks


_SOURCE_ROOT = Path("/content/learned-sources")
_CHECKPOINT_ROOT = Path("/content/learned-checkpoints")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_COLMAP_PRODUCER_PATHS = (
    "backend/static_pipeline/colmap.py",
    "experiments/learned_quality/geometry.py",
    "experiments/learned_quality/runtime.py",
)
_BASE_EVIDENCE_PRODUCER_PATHS = (
    "backend/preprocess/parse_colmap.py",
    "experiments/learned_quality/da3.py",
    "experiments/learned_quality/runtime.py",
    "experiments/learned_quality/tracks.py",
)
_SEMANTIC_PRODUCER_PATHS = (
    "experiments/learned_quality/model_adapters.py",
    "experiments/learned_quality/segmentation.py",
)
_MOTION_PRODUCER_PATHS = (
    "experiments/learned_quality/flow.py",
    "experiments/learned_quality/lifecycle.py",
    "experiments/learned_quality/model_adapters.py",
)
_MASKS_PRODUCER_PATHS = ("experiments/learned_quality/masks.py",)
_BACKFILL_SELECTION_PRODUCER_PATHS = (
    "backend/static_pipeline/selection.py",
    "experiments/learned_quality/runtime.py",
)
_GEOMETRY_PRODUCER_PATHS = (
    "experiments/learned_quality/geometry.py",
    "experiments/learned_quality/runtime.py",
)
_FINAL_PRETRAINING_PRODUCER_PATHS = (
    "experiments/learned_quality/depth.py",
    "experiments/learned_quality/photometric.py",
    "experiments/learned_quality/runtime.py",
)


def _learned_stage(stage_id: str):
    reporter = current_stage_reporter()
    if reporter is None:
        return nullcontext()
    return reporter.stage(stage_id)


def _probe_colmap_version() -> str:
    executable = _resolve_colmap_executable(None)
    completed = subprocess.run(
        (executable, "-h"),
        check=True,
        capture_output=True,
        text=True,
    )
    return _colmap_version_from_help(completed)


def _colmap_cache_fingerprint(
    selection: SelectionOutput,
    hardware: HardwareInfo,
    model_manifest_path: Path,
    *,
    version_probe: Callable[[], str] = _probe_colmap_version,
) -> str:
    return checkpoint_fingerprint(
        CheckpointKind.COLMAP,
        _colmap_cache_inputs(
            selection,
            hardware,
            model_manifest_path,
            version_probe=version_probe,
        ),
    )


def _colmap_cache_inputs(
    selection: SelectionOutput,
    hardware: HardwareInfo,
    model_manifest_path: Path,
    *,
    version_probe: Callable[[], str] = _probe_colmap_version,
) -> CheckpointInputs:
    return CheckpointInputs(
        source_digest=selection.inventory.digest,
        settings={
            "selection_digest": selection.manifest.image_set_digest,
            "use_gpu": hardware.colmap_gpu_sift is True,
        },
        model_manifest_sha256=_sha256(model_manifest_path),
        tool_versions={"colmap": version_probe()},
        producer_code_sha256=producer_code_digest(
            _REPOSITORY_ROOT,
            _COLMAP_PRODUCER_PATHS,
        ),
    )


def _colmap_cache_fingerprints(
    selection: SelectionOutput,
    hardware: HardwareInfo,
    model_manifest_path: Path,
    *,
    version_probe: Callable[[], str] = _probe_colmap_version,
) -> tuple[str, ...]:
    return compatible_colmap_fingerprints(
        _colmap_cache_inputs(
            selection,
            hardware,
            model_manifest_path,
            version_probe=version_probe,
        ),
        legacy_producer_digests=LEGACY_COLMAP_PRODUCER_DIGESTS,
    )


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


def _rigid_scene(
    frames: tuple[FrameArtifact, ...],
    model_dir: Path,
    depths: tuple[tuple[Path, str], ...],
    *,
    static_tracks: tuple[StaticTrack, ...],
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
        static_tracks=static_tracks,
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


def _evidence_milestone_refs(
    session: MilestoneSession,
    *,
    colmap_fingerprint: str,
) -> dict[CheckpointKind, MilestoneRef]:
    base_upstream = {
        CheckpointKind.SELECTION: session.selection_ref.fingerprint,
        CheckpointKind.COLMAP: colmap_fingerprint,
    }
    base = session.make_ref(
        CheckpointKind.BASE_EVIDENCE,
        upstream=base_upstream,
        settings={
            "track_policy": {
                "minimum_observations": 3,
                "maximum_invalid_fraction": 0.01,
                "maximum_examples": 32,
            },
            "anchor_vram_policy": "a100-80gb",
            "metric_batches": [12, 6],
        },
        producer_paths=_BASE_EVIDENCE_PRODUCER_PATHS,
    )
    semantic = session.make_ref(
        CheckpointKind.SEMANTIC,
        upstream={CheckpointKind.BASE_EVIDENCE: base.fingerprint},
        settings={
            "policy": [0.30, 0.25, 32, 3],
            "batches": [8, 4],
        },
        producer_paths=_SEMANTIC_PRODUCER_PATHS,
    )
    motion = session.make_ref(
        CheckpointKind.MOTION,
        upstream={CheckpointKind.BASE_EVIDENCE: base.fingerprint},
        settings={
            "policy": [0.03, 0.005, 1.5, 0.01, 3.0, 3, 1.5, 3.0],
            "pair_batches": [2, 1],
        },
        producer_paths=_MOTION_PRODUCER_PATHS,
    )
    masks = session.make_ref(
        CheckpointKind.MASKS,
        upstream={
            CheckpointKind.BASE_EVIDENCE: base.fingerprint,
            CheckpointKind.SEMANTIC: semantic.fingerprint,
            CheckpointKind.MOTION: motion.fingerprint,
        },
        settings={
            "policy": [
                0.003,
                0.001,
                0.0002,
                0.20,
                0.45,
                0.80,
                0.003,
                0.02,
                0.01,
            ]
        },
        producer_paths=_MASKS_PRODUCER_PATHS,
    )
    return {
        CheckpointKind.BASE_EVIDENCE: base,
        CheckpointKind.SEMANTIC: semantic,
        CheckpointKind.MOTION: motion,
        CheckpointKind.MASKS: masks,
    }


def _refs_for_artifacts(
    session: MilestoneSession,
    artifacts: LearnedArtifacts,
) -> dict[CheckpointKind, MilestoneRef]:
    base = artifacts.base_evidence
    if not isinstance(base, BaseEvidenceState):
        raise ValueError("learned artifacts are missing typed base evidence")
    return _evidence_milestone_refs(session, colmap_fingerprint=base.colmap_ref)


def _geometry_milestone_ref(
    session: MilestoneSession,
    masks_ref: MilestoneRef,
    hardware: HardwareInfo,
) -> MilestoneRef:
    return session.make_ref(
        CheckpointKind.GEOMETRY,
        upstream={
            CheckpointKind.SELECTION: session.selection_ref.fingerprint,
            CheckpointKind.MASKS: masks_ref.fingerprint,
        },
        settings={
            "candidate_rounds": 2,
            "backfill_max_per_interval": 2,
            "use_gpu": hardware.colmap_gpu_sift is True,
        },
        producer_paths=_GEOMETRY_PRODUCER_PATHS,
    )


def _final_pretraining_milestone_ref(
    session: MilestoneSession,
    geometry_ref: MilestoneRef,
    masks_ref: MilestoneRef,
) -> MilestoneRef:
    return session.make_ref(
        CheckpointKind.PRETRAINING,
        upstream={
            CheckpointKind.GEOMETRY: geometry_ref.fingerprint,
            CheckpointKind.MASKS: masks_ref.fingerprint,
        },
        settings={
            "photometric": [0.03, 0.10, 5],
            "dense_seed": [0.20, 0.05, 0.003, 0.001, 2],
        },
        producer_paths=_FINAL_PRETRAINING_PRODUCER_PATHS,
    )


def _publish_backfill_selection(
    session: MilestoneSession,
    selection: SelectionOutput,
    *,
    parent_selection_ref: MilestoneRef,
) -> MilestoneSession:
    selection_ref = session.make_ref(
        CheckpointKind.SELECTION,
        upstream={},
        settings={
            "mode": "geometry_backfill",
            "parent_selection": parent_selection_ref.fingerprint,
            "max_per_interval": 2,
        },
        producer_paths=_BACKFILL_SELECTION_PRODUCER_PATHS,
        selection_digest=selection.manifest.image_set_digest,
    )
    generation = session.store.publish_milestone(
        MilestoneState(
            ref=selection_ref,
            upstream={},
            value=selection,
            artifact_roots={"selection": selection.frames_dir},
        ),
        run_id=f"{session.run_id}-selection-backfill",
    )
    reporter = current_stage_reporter()
    if reporter is not None:
        reporter.cache_event("save", "backfill selection milestone", str(generation))
    return session.branch(
        selection_digest=selection.manifest.image_set_digest,
        selection_ref=selection_ref,
        selection_root=selection.frames_dir,
    )


def _restore_stage_state(
    session: MilestoneSession | None,
    ref: MilestoneRef | None,
    destination: Path,
    expected_type: type[object],
) -> object | None:
    if session is None or ref is None:
        return None
    restored = session.restore(ref, destination)
    if restored is None:
        return None
    if not isinstance(restored.value, expected_type):
        raise ValueError(
            f"restored {ref.kind.value} milestone has the wrong state type"
        )
    reporter = current_stage_reporter()
    if reporter is not None:
        reporter.cache_event("hit", f"{ref.kind.value} milestone", str(destination))
    return restored.value


def _run_evidence_cycle(
    selection: SelectionOutput,
    frames: tuple[FrameArtifact, ...],
    hardware: HardwareInfo,
    output_root: Path,
    *,
    checkpoint_store: LearnedCheckpointStore | None = None,
    milestone_session: MilestoneSession | None = None,
) -> LearnedArtifacts:
    output_root.mkdir(parents=True, exist_ok=False)
    reporter = current_stage_reporter()
    da3_frames = tuple(
        DA3Frame(frame.image_name, frame.frame_id, frame.path) for frame in frames
    )
    model_manifest_path: Path | None = None
    cache_fingerprints: tuple[str, ...] | None = None
    milestone_refs: dict[CheckpointKind, MilestoneRef] = {}
    if milestone_session is not None:
        model_manifest_path = _validate_model_manifest()
        cache_fingerprints = _colmap_cache_fingerprints(
            selection,
            hardware,
            model_manifest_path,
        )
        milestone_refs = _evidence_milestone_refs(
            milestone_session,
            colmap_fingerprint=cache_fingerprints[0],
        )

    base_state = _restore_stage_state(
        milestone_session,
        milestone_refs.get(CheckpointKind.BASE_EVIDENCE),
        output_root / "restored-base-evidence",
        BaseEvidenceState,
    )
    if isinstance(base_state, BaseEvidenceState):
        anchors = base_state.anchors
        depths = base_state.depths
        sky = base_state.sky
        scene = base_state.scene
        if reporter is not None:
            for stage_id in ("da3_anchor", "classical_colmap", "da3_metric_sky"):
                reporter.skip(stage_id, "restored from base_evidence milestone")
    else:
        if milestone_session is not None and reporter is not None:
            reporter.cache_event(
                "miss",
                "base_evidence milestone",
                str(milestone_session.store.cache_root),
            )
        with _learned_stage("da3_anchor"):
            base = load_da3_model(
                _SOURCE_ROOT / "da3", _checkpoint("depth-anything/DA3-BASE")
            )
            try:
                anchors = run_anchor_inference(
                    base,
                    da3_frames,
                    output_root / "da3-anchor",
                    vram_gb=hardware.vram_gb,
                )
            finally:
                release_cuda_model(base)
                del base

        cache_fingerprint: str | None = None
        pending_colmap_publish = False
        with _learned_stage("classical_colmap"):
            pre_attempt = None
            if checkpoint_store is not None:
                if model_manifest_path is None:
                    model_manifest_path = _validate_model_manifest()
                if cache_fingerprints is None:
                    cache_fingerprints = _colmap_cache_fingerprints(
                        selection,
                        hardware,
                        model_manifest_path,
                    )
                cache_fingerprint = cache_fingerprints[0]
                for index, candidate_fingerprint in enumerate(cache_fingerprints):
                    suffix = "" if index == 0 else f"-legacy-{index}"
                    pre_attempt = checkpoint_store.restore_colmap(
                        candidate_fingerprint,
                        destination=output_root / f"classical-prepass-restored{suffix}",
                    )
                    if pre_attempt is not None:
                        pending_colmap_publish = index > 0
                        break
                if reporter is not None:
                    reporter.cache_event(
                        "hit" if pre_attempt is not None else "miss",
                        (
                            "COLMAP checkpoint"
                            if not pending_colmap_publish
                            else "COLMAP checkpoint (compatible legacy)"
                        ),
                        str(checkpoint_store.cache_root),
                    )
            if pre_attempt is None:
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
                pending_colmap_publish = checkpoint_store is not None
            measured = tuple(measure_models(pre_attempt.model_dirs, selection.manifest))
            if not measured:
                raise RuntimeError("classical prepass produced no readable geometry")
            pre_model = max(
                measured,
                key=lambda item: (item.registered_count, item.sparse_point_count),
            ).model_dir
            qualified_tracks = qualify_colmap_static_tracks(
                pre_model,
                frames,
                output_root / "track_audit.json",
                policy=TrackQualificationPolicy(),
            )

        with _learned_stage("da3_metric_sky"):
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
                        _SOURCE_ROOT / "da3",
                        _checkpoint("depth-anything/DA3METRIC-LARGE"),
                    ),
                )
            finally:
                release_cuda_model(metric_model)
                del metric_model
            depths, sky = _materialize_metric(
                frames, metric, output_root / "metric-native"
            )
            scene = _rigid_scene(
                frames,
                pre_model,
                depths,
                static_tracks=qualified_tracks.tracks,
            )
            if (
                pending_colmap_publish
                and checkpoint_store is not None
                and cache_fingerprint is not None
            ):
                generation = checkpoint_store.publish_colmap(
                    pre_attempt,
                    fingerprint=cache_fingerprint,
                    run_id=f"colmap-{uuid.uuid4().hex}",
                )
                if reporter is not None:
                    reporter.cache_event(
                        "save",
                        "COLMAP checkpoint",
                        str(generation),
                    )
        current_colmap_ref = (
            cache_fingerprints[0]
            if cache_fingerprints is not None
            else cache_fingerprint
        )
        if current_colmap_ref is None and milestone_session is not None:
            raise RuntimeError("base evidence cannot identify its COLMAP checkpoint")
        base_state = BaseEvidenceState(
            anchors=anchors,
            depths=depths,
            sky=sky,
            scene=scene,
            track_audit=qualified_tracks,
            colmap_ref=current_colmap_ref or "0" * 64,
        )
        if milestone_session is not None:
            base_ref = milestone_refs[CheckpointKind.BASE_EVIDENCE]
            generation = milestone_session.publish(
                base_ref,
                upstream={
                    CheckpointKind.SELECTION: (
                        milestone_session.selection_ref.fingerprint
                    ),
                    CheckpointKind.COLMAP: base_state.colmap_ref,
                },
                value=base_state,
                artifact_roots={"base": output_root},
            )
            if reporter is not None:
                reporter.cache_event("save", "base_evidence milestone", str(generation))

    semantic_state = _restore_stage_state(
        milestone_session,
        milestone_refs.get(CheckpointKind.SEMANTIC),
        output_root / "restored-semantic",
        SemanticMilestoneState,
    )
    if isinstance(semantic_state, SemanticMilestoneState):
        semantic = semantic_state.semantic
        if reporter is not None:
            reporter.skip("semantic_masks", "restored from semantic milestone")
    else:
        with _learned_stage("semantic_masks"):
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
        if milestone_session is not None:
            semantic_ref = milestone_refs[CheckpointKind.SEMANTIC]
            generation = milestone_session.publish(
                semantic_ref,
                upstream={
                    CheckpointKind.BASE_EVIDENCE: milestone_refs[
                        CheckpointKind.BASE_EVIDENCE
                    ].fingerprint
                },
                value=SemanticMilestoneState(semantic),
                artifact_roots={"semantic": output_root / "semantic"},
            )
            if reporter is not None:
                reporter.cache_event("save", "semantic milestone", str(generation))

    motion_state = _restore_stage_state(
        milestone_session,
        milestone_refs.get(CheckpointKind.MOTION),
        output_root / "restored-motion",
        MotionMilestoneState,
    )
    if isinstance(motion_state, MotionMilestoneState):
        flow = motion_state.motion
        if reporter is not None:
            reporter.skip("optical_flow", "restored from motion milestone")
    else:
        with _learned_stage("optical_flow"):
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
        if milestone_session is not None:
            motion_ref = milestone_refs[CheckpointKind.MOTION]
            generation = milestone_session.publish(
                motion_ref,
                upstream={
                    CheckpointKind.BASE_EVIDENCE: milestone_refs[
                        CheckpointKind.BASE_EVIDENCE
                    ].fingerprint
                },
                value=MotionMilestoneState(flow),
                artifact_roots={"motion": output_root / "flow"},
            )
            if reporter is not None:
                reporter.cache_event("save", "motion milestone", str(generation))

    masks_state = _restore_stage_state(
        milestone_session,
        milestone_refs.get(CheckpointKind.MASKS),
        output_root / "restored-masks",
        MasksMilestoneState,
    )
    if isinstance(masks_state, MasksMilestoneState):
        masks = masks_state.masks
        if reporter is not None:
            reporter.skip("mask_fusion", "restored from masks milestone")
    else:
        with _learned_stage("mask_fusion"):
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
        if milestone_session is not None:
            masks_ref = milestone_refs[CheckpointKind.MASKS]
            generation = milestone_session.publish(
                masks_ref,
                upstream={
                    CheckpointKind.BASE_EVIDENCE: milestone_refs[
                        CheckpointKind.BASE_EVIDENCE
                    ].fingerprint,
                    CheckpointKind.SEMANTIC: milestone_refs[
                        CheckpointKind.SEMANTIC
                    ].fingerprint,
                    CheckpointKind.MOTION: milestone_refs[
                        CheckpointKind.MOTION
                    ].fingerprint,
                },
                value=MasksMilestoneState(masks),
                artifact_roots={"masks": output_root / "masks"},
            )
            if reporter is not None:
                reporter.cache_event("save", "masks milestone", str(generation))

    registered_count = sum(
        1 for frame in getattr(scene, "frames", ()) if frame.registered
    )
    stages = (
        StageRecord(
            "da3_anchor", "accepted", details={"anchors": len(anchors.anchor_indices)}
        ),
        StageRecord(
            "classical_prepass",
            "accepted",
            details={"registered": registered_count},
        ),
        StageRecord("da3_metric_sky", "accepted", details={"frames": len(depths)}),
        *semantic.stage_records,
        *flow.stage_records,
        StageRecord(
            "mask_fusion", "accepted", details={"digest": masks.mask_set_digest}
        ),
    )
    return LearnedArtifacts(
        da3=anchors,
        base_evidence=base_state,
        semantic=semantic,
        flow=flow,
        masks=masks,
        model_manifest_path=model_manifest_path or _validate_model_manifest(),
        stage_records=stages,
    )


def run_learned_reconstruction(
    selection: SelectionOutput,
    *,
    spec: object,
    hardware: HardwareInfo,
    output_root: Path,
    checkpoint_store: LearnedCheckpointStore | None = None,
    milestone_session: MilestoneSession | None = None,
) -> LearnedReconstructionOutput:
    del spec
    _validate_model_manifest()
    frames = _frame_artifacts(selection)
    evidence_kwargs: dict[str, object] = {"checkpoint_store": checkpoint_store}
    if milestone_session is not None:
        evidence_kwargs["milestone_session"] = milestone_session
    active = {
        "selection": selection,
        "frames": frames,
        "artifacts": _run_evidence_cycle(
            selection,
            frames,
            hardware,
            output_root.parent / "learned-evidence-0",
            **evidence_kwargs,
        ),
        "milestone_session": milestone_session,
    }
    if milestone_session is not None:
        active["milestone_refs"] = _refs_for_artifacts(
            milestone_session,
            active["artifacts"],
        )
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
            source_manifest_path=backfill_root / "selection_manifest.json",
        )
        new_frames = _frame_artifacts(new_selection)
        branch_session = active["milestone_session"]
        if branch_session is not None:
            branch_session = _publish_backfill_selection(
                branch_session,
                new_selection,
                parent_selection_ref=branch_session.selection_ref,
            )
        branch_kwargs: dict[str, object] = {"checkpoint_store": checkpoint_store}
        if branch_session is not None:
            branch_kwargs["milestone_session"] = branch_session
        new_artifacts = _run_evidence_cycle(
            new_selection,
            new_frames,
            hardware,
            output_root.parent / "learned-evidence-1",
            **branch_kwargs,
        )
        active.update(
            selection=new_selection,
            frames=new_frames,
            artifacts=new_artifacts,
            milestone_session=branch_session,
        )
        if branch_session is not None:
            active["milestone_refs"] = _refs_for_artifacts(
                branch_session,
                new_artifacts,
            )
        return new_manifest, new_frames

    geometry_state = None
    geometry_ref = None
    if active["milestone_session"] is not None:
        geometry_ref = _geometry_milestone_ref(
            active["milestone_session"],
            active["milestone_refs"][CheckpointKind.MASKS],
            hardware,
        )
        geometry_state = _restore_stage_state(
            active["milestone_session"],
            geometry_ref,
            output_root,
            GeometryMilestoneState,
        )
    if isinstance(geometry_state, GeometryMilestoneState):
        comparison = LearnedReconstructionOutput(
            bundle=geometry_state.bundle,
            frames_dir=geometry_state.frames_dir,
            artifacts=active["artifacts"],
            geometry_candidates=geometry_state.geometry_candidates,
        )
        reporter = current_stage_reporter()
        if reporter is not None:
            reporter.skip("geometry_comparison", "restored from geometry milestone")
    else:
        with _learned_stage("geometry_comparison"):
            comparison = run_geometry_comparison(
                selection.manifest,
                frames,
                output_root=output_root,
                artifacts=active["artifacts"],
                classical_runner=classical,
                hybrid_runner=hybrid,
                materialize_backfill=backfill,
            )
        geometry_state = GeometryMilestoneState(
            bundle=comparison.bundle,
            frames_dir=comparison.frames_dir,
            geometry_candidates=comparison.geometry_candidates,
        )
        if active["milestone_session"] is not None:
            geometry_ref = _geometry_milestone_ref(
                active["milestone_session"],
                active["milestone_refs"][CheckpointKind.MASKS],
                hardware,
            )
            generation = active["milestone_session"].publish(
                geometry_ref,
                upstream={
                    CheckpointKind.SELECTION: active[
                        "milestone_session"
                    ].selection_ref.fingerprint,
                    CheckpointKind.MASKS: active["milestone_refs"][
                        CheckpointKind.MASKS
                    ].fingerprint,
                },
                value=geometry_state,
                artifact_roots={"geometry": output_root},
            )
            reporter = current_stage_reporter()
            if reporter is not None:
                reporter.cache_event("save", "geometry milestone", str(generation))
    final_frames = active["frames"]
    artifacts = active["artifacts"]
    final_ref = None
    final_state = None
    if active["milestone_session"] is not None and isinstance(
        geometry_ref, MilestoneRef
    ):
        final_ref = _final_pretraining_milestone_ref(
            active["milestone_session"],
            geometry_ref,
            active["milestone_refs"][CheckpointKind.MASKS],
        )
        final_state = _restore_stage_state(
            active["milestone_session"],
            final_ref,
            output_root.parent / "restored-final-pretraining",
            FinalPretrainingState,
        )
    if isinstance(final_state, FinalPretrainingState):
        reporter = current_stage_reporter()
        if reporter is not None:
            for stage_id in (
                "photometric_validation",
                "final_pose_depth",
                "dense_seed_fusion",
            ):
                reporter.skip(stage_id, "restored from final pretraining milestone")
        return assemble_learned_reconstruction(
            geometry=geometry_state,
            base=artifacts.base_evidence,
            semantic=SemanticMilestoneState(artifacts.semantic),
            motion=MotionMilestoneState(artifacts.flow),
            masks=MasksMilestoneState(artifacts.masks),
            evidence_stage_records=artifacts.stage_records,
            final=final_state,
        )
    metric_artifacts = artifacts.da3
    # Use metric-native depth already bound into the motion evidence scene.
    rigid_frames = tuple(artifacts.flow.frames)
    del metric_artifacts, rigid_frames
    if isinstance(artifacts.base_evidence, BaseEvidenceState):
        depths = artifacts.base_evidence.depths
    else:
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
    if len(depths) != len(final_frames):
        raise ValueError("base-evidence depth count differs from final frames")
    with _learned_stage("photometric_validation"):
        final_tracks = qualify_colmap_static_tracks(
            comparison.accepted_model_dir,
            final_frames,
            output_root.parent / "final_track_audit.json",
            policy=TrackQualificationPolicy(),
        )
        final_scene = _rigid_scene(
            final_frames,
            comparison.accepted_model_dir,
            depths,
            static_tracks=final_tracks.tracks,
        )
        photometric = fit_and_validate_photometric_transforms(
            final_frames,
            final_scene,
            artifacts.masks,
            output_root.parent / "photometric",
            reference_frame_id=final_frames[len(final_frames) // 2].frame_id,
            policy=PhotometricPolicy(0.03, 0.10, 5),
        )
    with _learned_stage("final_pose_depth"):
        base = load_da3_model(
            _SOURCE_ROOT / "da3", _checkpoint("depth-anything/DA3-BASE")
        )
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
    with _learned_stage("dense_seed_fusion"):
        validated_depth = validate_depth_and_fuse_seeds(
            final_frames,
            final_depth,
            artifacts.masks,
            comparison.accepted_model_dir,
            output_root.parent / "validated-depth",
            policy=DenseSeedPolicy(0.20, 0.05, 0.003, 0.001, 2),
        )
    final_stage_records = (
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
    )
    manifest_root = output_root.parent / "final-pretraining"
    manifest_root.mkdir(parents=True, exist_ok=False)
    manifest_copy = manifest_root / "model_manifest.json"
    manifest_source = artifacts.model_manifest_path or _validate_model_manifest()
    shutil.copy2(manifest_source, manifest_copy)
    final_state = FinalPretrainingState(
        photometric=photometric,
        depth=validated_depth,
        dense_seeds=validated_depth.dense_seeds,
        final_stage_records=final_stage_records,
        model_manifest_path=manifest_copy.resolve(strict=True),
    )
    if active["milestone_session"] is not None and isinstance(final_ref, MilestoneRef):
        generation = active["milestone_session"].publish(
            final_ref,
            upstream={
                CheckpointKind.GEOMETRY: geometry_ref.fingerprint,
                CheckpointKind.MASKS: active["milestone_refs"][
                    CheckpointKind.MASKS
                ].fingerprint,
            },
            value=final_state,
            artifact_roots={
                "photometric": output_root.parent / "photometric",
                "validated_depth": output_root.parent / "validated-depth",
                "final_metadata": manifest_root,
            },
        )
        reporter = current_stage_reporter()
        if reporter is not None:
            reporter.cache_event("save", "final pretraining milestone", str(generation))
    if isinstance(artifacts.base_evidence, BaseEvidenceState):
        return assemble_learned_reconstruction(
            geometry=geometry_state,
            base=artifacts.base_evidence,
            semantic=SemanticMilestoneState(artifacts.semantic),
            motion=MotionMilestoneState(artifacts.flow),
            masks=MasksMilestoneState(artifacts.masks),
            evidence_stage_records=artifacts.stage_records,
            final=final_state,
        )
    return LearnedReconstructionOutput(
        bundle=geometry_state.bundle,
        frames_dir=geometry_state.frames_dir,
        artifacts=replace(
            artifacts,
            photometric=photometric,
            depth=validated_depth,
            dense_seeds=validated_depth.dense_seeds,
            model_manifest_path=manifest_copy.resolve(strict=True),
            stage_records=(*artifacts.stage_records, *final_stage_records),
        ),
        geometry_candidates=geometry_state.geometry_candidates,
    )

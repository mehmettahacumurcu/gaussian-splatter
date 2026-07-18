from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from .cache import (
    CheckpointKind,
    LearnedCacheOwnershipError,
    LearnedCheckpointStore,
    producer_code_digest,
)


AUDIT_SCHEMA_VERSION = 1
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "fingerprint",
        "input_digest",
        "selection_digest",
        "colmap_fingerprint",
        "qualification_policy_sha256",
        "audit_producer_code_sha256",
        "track_audit_sha256",
        "accepted_track_count",
        "passed",
    }
)


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_strict_json(path: Path) -> object:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


@dataclass(frozen=True)
class AuditInputs:
    input_digest: str
    selection_digest: str
    colmap_fingerprint: str
    qualification_policy_sha256: str
    audit_producer_code_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "input_digest",
            "selection_digest",
            "colmap_fingerprint",
            "qualification_policy_sha256",
            "audit_producer_code_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )


def audit_fingerprint(inputs: AuditInputs) -> str:
    if not isinstance(inputs, AuditInputs):
        raise TypeError("inputs must be AuditInputs")
    return hashlib.sha256(
        _strict_bytes(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                **asdict(inputs),
            }
        )
    ).hexdigest()


def qualification_policy_digest(policy: object) -> str:
    from .tracks import TRACK_AUDIT_SCHEMA_VERSION, TrackQualificationPolicy

    if not isinstance(policy, TrackQualificationPolicy):
        raise TypeError("policy must be a TrackQualificationPolicy")
    return hashlib.sha256(
        _strict_bytes(
            {
                "track_audit_schema_version": TRACK_AUDIT_SCHEMA_VERSION,
                "policy": asdict(policy),
            }
        )
    ).hexdigest()


def audit_producer_digest(repository_root: Path | None = None) -> str:
    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    return producer_code_digest(
        root,
        (
            "backend/static_pipeline/reconstruction.py",
            "experiments/learned_quality/audit.py",
            "experiments/learned_quality/tracks.py",
        ),
    )


def make_audit_inputs(
    *,
    input_digest: str,
    selection_digest: str,
    colmap_fingerprint: str,
    policy: object,
    repository_root: Path | None = None,
) -> AuditInputs:
    return AuditInputs(
        input_digest=input_digest,
        selection_digest=selection_digest,
        colmap_fingerprint=colmap_fingerprint,
        qualification_policy_sha256=qualification_policy_digest(policy),
        audit_producer_code_sha256=audit_producer_digest(repository_root),
    )


@dataclass(frozen=True)
class AuditReceipt:
    schema_version: int
    fingerprint: str
    input_digest: str
    selection_digest: str
    colmap_fingerprint: str
    qualification_policy_sha256: str
    audit_producer_code_sha256: str
    track_audit_sha256: str
    accepted_track_count: int
    passed: bool

    def __post_init__(self) -> None:
        if self.schema_version != AUDIT_SCHEMA_VERSION:
            raise ValueError("audit receipt schema is unsupported")
        for field_name in (
            "fingerprint",
            "input_digest",
            "selection_digest",
            "colmap_fingerprint",
            "qualification_policy_sha256",
            "audit_producer_code_sha256",
            "track_audit_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )
        if type(self.accepted_track_count) is not int or self.accepted_track_count < 1:
            raise ValueError("accepted_track_count must be a positive integer")
        if self.passed is not True:
            raise ValueError("only passing audit receipts are publishable")


def _receipt_from_payload(payload: object) -> AuditReceipt:
    if not isinstance(payload, dict) or frozenset(payload) != _RECEIPT_KEYS:
        raise ValueError("audit receipt fields are malformed")
    return AuditReceipt(**payload)


def publish_audit_receipt(
    cache_root: Path,
    *,
    inputs: AuditInputs,
    track_audit_path: Path,
    accepted_track_count: int,
) -> Path:
    if not isinstance(inputs, AuditInputs):
        raise TypeError("inputs must be AuditInputs")
    audit_path = Path(track_audit_path).resolve(strict=True)
    if audit_path.is_symlink() or not audit_path.is_file():
        raise ValueError("track_audit_path must be a regular file")
    store = LearnedCheckpointStore(
        Path(cache_root),
        input_identity=inputs.input_digest,
    )
    store._ensure_owned_root(create=True)
    fingerprint = audit_fingerprint(inputs)
    target = store.cache_root / "audits" / fingerprint
    if os.path.lexists(target):
        try:
            require_audit_receipt(store.cache_root, expected=inputs)
            return target
        except RuntimeError:
            if target.is_symlink() or not target.is_dir():
                raise ValueError("audit receipt target is unsafe")
            shutil.rmtree(target)
    target.mkdir(parents=True)
    try:
        copied_audit = target / "track_audit.json"
        shutil.copy2(audit_path, copied_audit)
        receipt = AuditReceipt(
            schema_version=AUDIT_SCHEMA_VERSION,
            fingerprint=fingerprint,
            input_digest=inputs.input_digest,
            selection_digest=inputs.selection_digest,
            colmap_fingerprint=inputs.colmap_fingerprint,
            qualification_policy_sha256=inputs.qualification_policy_sha256,
            audit_producer_code_sha256=inputs.audit_producer_code_sha256,
            track_audit_sha256=_sha256(copied_audit),
            accepted_track_count=accepted_track_count,
            passed=True,
        )
        success = target / "_SUCCESS.json"
        with success.open("xb") as stream:
            stream.write(_strict_bytes(asdict(receipt)))
            stream.flush()
            os.fsync(stream.fileno())
        require_audit_receipt(store.cache_root, expected=inputs)
        return target
    except BaseException:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        raise


def require_audit_receipt(
    cache_root: Path,
    *,
    expected: AuditInputs,
) -> AuditReceipt:
    message = (
        "A matching CPU cache audit receipt is required; run the CPU cache audit "
        "notebook before starting an A100 session"
    )
    try:
        if not isinstance(expected, AuditInputs):
            raise TypeError("expected must be AuditInputs")
        store = LearnedCheckpointStore(
            Path(cache_root),
            input_identity=expected.input_digest,
        )
        if not store._ensure_owned_root(create=False):
            raise ValueError("cache root is missing")
        target = store.cache_root / "audits" / audit_fingerprint(expected)
        if target.is_symlink() or not target.is_dir():
            raise ValueError("audit receipt directory is missing")
        success = target / "_SUCCESS.json"
        track_audit = target / "track_audit.json"
        if (
            success.is_symlink()
            or not success.is_file()
            or track_audit.is_symlink()
            or not track_audit.is_file()
        ):
            raise ValueError("audit receipt is incomplete")
        receipt = _receipt_from_payload(_read_strict_json(success))
        if receipt.fingerprint != audit_fingerprint(expected):
            raise ValueError("audit fingerprint differs")
        for field_name, value in asdict(expected).items():
            if getattr(receipt, field_name) != value:
                raise ValueError(f"audit {field_name} differs")
        if receipt.track_audit_sha256 != _sha256(track_audit):
            raise ValueError("track audit hash differs")
        if not math.isfinite(float(receipt.accepted_track_count)):
            raise ValueError("accepted track count is invalid")
        return receipt
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        LearnedCacheOwnershipError,
    ) as error:
        raise RuntimeError(message) from error


def _selection_frames(selection: object) -> tuple[object, ...]:
    from backend.static_pipeline.runner import SelectionOutput

    from .contracts import FrameArtifact

    if not isinstance(selection, SelectionOutput):
        raise TypeError("selection must be a SelectionOutput")
    frames = []
    for record in selection.manifest.selected_frames:
        path = (selection.frames_dir / record.output_name).resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"selected frame is not a regular file: {path}")
        observed = _sha256(path)
        if observed != record.sha256:
            raise ValueError(f"selected frame hash differs: {record.output_name}")
        frames.append(
            FrameArtifact(
                image_name=record.output_name,
                frame_id=record.frame_id,
                path=path,
                sha256=observed,
            )
        )
    if not frames:
        raise ValueError("selection contains no frames")
    return tuple(frames)


def _colmap_fingerprints(store: LearnedCheckpointStore) -> tuple[str, ...]:
    parent = store.cache_root / "colmap"
    if parent.is_symlink() or not parent.is_dir():
        return ()
    result = []
    for candidate in sorted(parent.iterdir(), key=lambda path: path.name):
        try:
            fingerprint = _digest(candidate.name, "COLMAP fingerprint")
        except ValueError:
            continue
        if store.find_generation(CheckpointKind.COLMAP, fingerprint) is not None:
            result.append(fingerprint)
    return tuple(result)


def run_cache_audit(
    *,
    input_path: Path,
    cache_root: Path,
    local_root: Path,
    spec: object,
    repository_root: Path | None = None,
) -> tuple[AuditReceipt, ...]:
    """Restore/create selection, audit every reusable COLMAP generation, publish receipts."""
    from backend.static_pipeline.reconstruction import measure_models
    from backend.static_pipeline.runner import (
        HardwareInfo,
        SelectionOutput,
        _selection_adapter,
    )
    from backend.static_pipeline.sources import copy_input_read_only, discover_source

    from .milestones import MilestoneState
    from .runner import _selection_milestone_ref
    from .tracks import TrackQualificationPolicy, qualify_colmap_static_tracks

    source = discover_source(Path(input_path))
    store = LearnedCheckpointStore(Path(cache_root), input_identity=source.digest)
    store.probe_drive_publication(run_id="cpu-audit")
    work = Path(local_root).resolve(strict=False)
    if os.path.lexists(work):
        raise FileExistsError(f"audit local_root already exists: {work}")
    work.mkdir(parents=True)
    hardware = HardwareInfo(
        "CPU audit",
        0.0,
        False,
        shutil.disk_usage(work).free / (1024**3),
        colmap_gpu_sift=True,
    )
    selection_ref = _selection_milestone_ref(
        source,
        spec,
        hardware,
        Path("/unused/model_manifest.json"),
    )
    restored = store.restore_milestone(
        selection_ref,
        destination=work / "selection-restored",
        source_inventory=source,
    )
    if restored is None:
        copied = copy_input_read_only(Path(input_path), work / "input")
        if copied.digest != source.digest:
            raise ValueError("local input copy differs from Drive source")
        selection = _selection_adapter(
            copied,
            spec=spec,
            output_root=work / "selection",
        )
        store.publish_milestone(
            MilestoneState(
                ref=selection_ref,
                upstream={},
                value=selection,
                artifact_roots={"selection": selection.frames_dir},
            ),
            run_id="cpu-audit-selection",
        )
    else:
        selection = restored.value
    if not isinstance(selection, SelectionOutput):
        raise ValueError("selection checkpoint restored the wrong state type")
    frames = _selection_frames(selection)
    policy = TrackQualificationPolicy()
    receipts: list[AuditReceipt] = []
    failures: list[str] = []
    for index, fingerprint in enumerate(_colmap_fingerprints(store)):
        destination = work / "colmap" / f"{index:03d}-{fingerprint}"
        attempt = store.restore_colmap(fingerprint, destination=destination)
        if attempt is None:
            continue
        try:
            measured = tuple(measure_models(attempt.model_dirs, selection.manifest))
            if not measured:
                raise ValueError("COLMAP checkpoint contains no measurable model")
            model = max(
                measured,
                key=lambda item: (item.registered_count, item.sparse_point_count),
            ).model_dir
            audit_path = work / "track-audits" / fingerprint / "track_audit.json"
            audit_path.parent.mkdir(parents=True)
            qualified = qualify_colmap_static_tracks(
                model,
                frames,
                audit_path,
                policy=policy,
            )
            inputs = make_audit_inputs(
                input_digest=source.digest,
                selection_digest=selection.manifest.image_set_digest,
                colmap_fingerprint=fingerprint,
                policy=policy,
                repository_root=repository_root,
            )
            publish_audit_receipt(
                store.cache_root,
                inputs=inputs,
                track_audit_path=qualified.audit_path,
                accepted_track_count=qualified.report.accepted_track_count,
            )
            receipts.append(require_audit_receipt(store.cache_root, expected=inputs))
        except (OSError, RuntimeError, ValueError) as error:
            failures.append(f"{fingerprint}: {type(error).__name__}: {error}")
    if not receipts:
        detail = "; ".join(failures) if failures else "no valid COLMAP checkpoint"
        raise RuntimeError(f"CPU cache audit found no compatible geometry: {detail}")
    return tuple(receipts)

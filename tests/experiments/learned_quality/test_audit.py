from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.static_pipeline.contracts import (
    ColmapAttempt,
    FrameRecord,
    SelectionManifest,
    SelectionPolicy,
    SourceFile,
    SourceInventory,
)
from backend.static_pipeline.runner import SelectionOutput
from experiments.learned_quality.contracts import (
    LearnedQualityRunSpec,
    to_static_run_spec,
)

from experiments.learned_quality.audit import (
    AuditInputs,
    find_compatible_audit_receipts,
    make_audit_inputs,
    publish_audit_receipt,
    require_audit_receipt,
    run_cache_audit,
)
from experiments.learned_quality.cache import LearnedCheckpointStore
from experiments.learned_quality.tracks import (
    QualifiedStaticTracks,
    TrackAuditReport,
    TrackQualificationPolicy,
)


_KNOWN_ROOM_COLMAP_FINGERPRINT = (
    "684800f437e38a019e9d96b6928b6a5d3995b93cc6c925eccd620a926756ec82"
)


def _inputs() -> AuditInputs:
    return AuditInputs(
        input_digest="a" * 64,
        selection_digest="b" * 64,
        colmap_fingerprint="c" * 64,
        qualification_policy_sha256="d" * 64,
        audit_producer_code_sha256="e" * 64,
    )


def _owned_cache(tmp_path: Path) -> Path:
    root = tmp_path / "room_learned_test_cache"
    LearnedCheckpointStore(root, input_identity="a" * 64).probe_drive_publication(
        run_id="audit-probe"
    )
    return root


def test_audit_receipt_round_trip_is_strict_and_hash_verified(tmp_path: Path) -> None:
    cache_root = _owned_cache(tmp_path)
    audit = tmp_path / "track_audit.json"
    audit.write_text('{"accepted_track_count":42}\n', encoding="utf-8")

    published = publish_audit_receipt(
        cache_root,
        inputs=_inputs(),
        track_audit_path=audit,
        accepted_track_count=42,
    )
    receipt = require_audit_receipt(cache_root, expected=_inputs())

    assert receipt.passed is True
    assert receipt.accepted_track_count == 42
    assert published == cache_root / "audits" / receipt.fingerprint
    assert (published / "track_audit.json").read_bytes() == audit.read_bytes()
    assert (
        json.loads((published / "_SUCCESS.json").read_text(encoding="utf-8"))[
            "track_audit_sha256"
        ]
        == receipt.track_audit_sha256
    )


def test_audit_module_imports_no_learned_framework_or_model_adapter() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import experiments.learned_quality.audit; "
            "print(','.join(sorted(name for name in sys.modules if "
            "name == 'torch' or name == 'transformers' or "
            "name.endswith('.model_adapters'))))",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert completed.stdout.strip() == ""


@pytest.mark.parametrize(
    "field",
    (
        "input_digest",
        "selection_digest",
        "colmap_fingerprint",
        "qualification_policy_sha256",
        "audit_producer_code_sha256",
    ),
)
def test_audit_receipt_rejects_every_stale_identity_field(
    tmp_path: Path,
    field: str,
) -> None:
    cache_root = _owned_cache(tmp_path)
    audit = tmp_path / "track_audit.json"
    audit.write_text("{}\n", encoding="utf-8")
    publish_audit_receipt(
        cache_root,
        inputs=_inputs(),
        track_audit_path=audit,
        accepted_track_count=1,
    )

    with pytest.raises(RuntimeError, match="CPU cache audit"):
        require_audit_receipt(
            cache_root,
            expected=replace(_inputs(), **{field: "f" * 64}),
        )


def test_audit_receipt_rejects_corrupt_track_audit(tmp_path: Path) -> None:
    cache_root = _owned_cache(tmp_path)
    audit = tmp_path / "track_audit.json"
    audit.write_text("{}\n", encoding="utf-8")
    published = publish_audit_receipt(
        cache_root,
        inputs=_inputs(),
        track_audit_path=audit,
        accepted_track_count=1,
    )
    (published / "track_audit.json").write_text("corrupt\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="CPU cache audit"):
        require_audit_receipt(cache_root, expected=_inputs())


def test_cpu_audit_materializes_selection_and_publishes_colmap_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "room"
    input_path.mkdir()
    video = input_path / "room.mov"
    video.write_bytes(b"video")
    source_file = SourceFile("room.mov", 5, "1" * 64)
    inventory = SourceInventory(
        schema_version=1,
        root=input_path,
        kind="video",
        media_files=(source_file,),
        all_files=(source_file,),
        digest="a" * 64,
    )
    cache_root = tmp_path / "room_learned_test_cache"
    store = LearnedCheckpointStore(cache_root, input_identity=inventory.digest)
    attempt_root = tmp_path / "source-colmap"
    model = attempt_root / "sparse" / "0"
    model.mkdir(parents=True)
    database = attempt_root / "colmap.db"
    database.write_bytes(b"database")
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        (model / filename).write_text(filename, encoding="utf-8")
    store.publish_colmap(
        ColmapAttempt(
            root=attempt_root,
            database_path=database,
            model_dirs=(model,),
            colmap_version="COLMAP 3.11.1",
            fingerprint="2" * 64,
        ),
        fingerprint="c" * 64,
        run_id="room-colmap",
    )

    monkeypatch.setattr(
        "backend.static_pipeline.sources.discover_source",
        lambda _path: inventory,
    )
    monkeypatch.setattr(
        "backend.static_pipeline.sources.copy_input_read_only",
        lambda _source, _target: inventory,
    )

    def select(_inventory: object, *, output_root: Path, **_kwargs: object):
        frames = output_root / "selected"
        frames.mkdir(parents=True)
        frame_path = frames / "frame_000000.png"
        frame_path.write_bytes(b"frame")
        manifest_path = frames / "selection_manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        manifest = SelectionManifest(
            schema_version=1,
            source_digest=inventory.digest,
            effective_mode="smart",
            policy=SelectionPolicy("smart", 800, 1920),
            frames=(
                FrameRecord(
                    "frame-0",
                    "room.mov",
                    0,
                    0,
                    0.0,
                    "frame_000000.png",
                    __import__("hashlib").sha256(b"frame").hexdigest(),
                    True,
                    None,
                    1.0,
                    ("selected",),
                ),
            ),
            image_set_digest="b" * 64,
        )
        return SelectionOutput(inventory, manifest, frames, manifest_path)

    monkeypatch.setattr("backend.static_pipeline.runner._selection_adapter", select)
    monkeypatch.setattr(
        "backend.static_pipeline.reconstruction.measure_models",
        lambda model_dirs, _manifest: (
            SimpleNamespace(
                model_dir=model_dirs[0], registered_count=1, sparse_point_count=10
            ),
        ),
    )

    def qualify(_model: Path, _frames: object, audit_path: Path, **_kwargs: object):
        audit_path.write_text('{"accepted_track_count":1}\n', encoding="utf-8")
        report = TrackAuditReport(1, 1, 1, 0, 3, 3, 0, {}, {}, (), "d" * 64)
        return QualifiedStaticTracks((), report, audit_path, "e" * 64)

    monkeypatch.setattr(
        "experiments.learned_quality.tracks.qualify_colmap_static_tracks",
        qualify,
    )
    spec = to_static_run_spec(LearnedQualityRunSpec(input_folder="room"))

    receipts = run_cache_audit(
        input_path=input_path,
        cache_root=cache_root,
        local_root=tmp_path / "audit-work",
        spec=spec,
        repository_root=Path(__file__).resolve().parents[3],
    )

    assert len(receipts) == 1
    assert receipts[0].colmap_fingerprint == "c" * 64
    assert receipts[0].selection_digest == "b" * 64
    assert (cache_root / "selection").is_dir()


def test_compatible_audit_receipt_recovers_known_room_colmap_generation(
    tmp_path: Path,
) -> None:
    cache_root = _owned_cache(tmp_path)
    store = LearnedCheckpointStore(cache_root, input_identity="a" * 64)
    attempt_root = tmp_path / "known-room-colmap"
    model = attempt_root / "sparse" / "0"
    model.mkdir(parents=True)
    database = attempt_root / "colmap.db"
    database.write_bytes(b"database")
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        (model / filename).write_text(filename, encoding="utf-8")
    store.publish_colmap(
        ColmapAttempt(
            root=attempt_root,
            database_path=database,
            model_dirs=(model,),
            colmap_version="COLMAP 3.11.1",
            fingerprint="f" * 64,
        ),
        fingerprint=_KNOWN_ROOM_COLMAP_FINGERPRINT,
        run_id="known-room",
    )
    policy = TrackQualificationPolicy()
    inputs = make_audit_inputs(
        input_digest="a" * 64,
        selection_digest="b" * 64,
        colmap_fingerprint=_KNOWN_ROOM_COLMAP_FINGERPRINT,
        policy=policy,
        repository_root=Path(__file__).resolve().parents[3],
    )
    audit = tmp_path / "known-room-track-audit.json"
    audit.write_text('{"accepted_track_count":1}\n', encoding="utf-8")
    publish_audit_receipt(
        cache_root,
        inputs=inputs,
        track_audit_path=audit,
        accepted_track_count=1,
    )

    receipts = find_compatible_audit_receipts(
        cache_root,
        input_digest="a" * 64,
        selection_digest="b" * 64,
        policy=policy,
        repository_root=Path(__file__).resolve().parents[3],
    )

    assert tuple(item.colmap_fingerprint for item in receipts) == (
        _KNOWN_ROOM_COLMAP_FINGERPRINT,
    )

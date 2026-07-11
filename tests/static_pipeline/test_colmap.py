from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import backend.static_pipeline.colmap as colmap_module
from backend.static_pipeline.colmap import (
    build_colmap_commands,
    choose_colmap_policy,
    probe_colmap_gpu_support,
    run_colmap_attempt,
)
from backend.static_pipeline.contracts import (
    FrameMetrics,
    FrameRecord,
    SelectionManifest,
    SelectionPolicy,
)
from backend.static_pipeline.selection import write_selection_manifest
from backend.static_pipeline.stage_cache import stage_fingerprint


@pytest.mark.parametrize(
    ("use_gpu", "count", "attempt", "matcher", "overlap"),
    [
        (True, 300, 0, "exhaustive", 0),
        (True, 800, 1, "exhaustive", 0),
        (False, 300, 0, "sequential", 20),
        (False, 300, 1, "exhaustive", 0),
        (False, 301, 1, "sequential", 40),
    ],
)
def test_matcher_policy(
    use_gpu: bool,
    count: int,
    attempt: int,
    matcher: str,
    overlap: int,
) -> None:
    policy = choose_colmap_policy(
        use_gpu=use_gpu,
        selected_count=count,
        attempt_index=attempt,
    )

    assert policy.matcher == matcher
    assert policy.sequential_overlap == overlap


@pytest.mark.parametrize(
    ("use_gpu", "count", "attempt"),
    [
        (True, 0, 0),
        (True, 801, 0),
        (True, 300, -1),
        (True, 300, 2),
    ],
)
def test_policy_rejects_unbounded_sets_and_extra_attempts(
    use_gpu: bool,
    count: int,
    attempt: int,
) -> None:
    with pytest.raises(ValueError):
        choose_colmap_policy(
            use_gpu=use_gpu,
            selected_count=count,
            attempt_index=attempt,
        )


def test_commands_use_argv_and_forward_cpu_overlap(tmp_path: Path) -> None:
    frames = tmp_path / "frames with spaces"
    frames.mkdir()
    (frames / "frame_000000.png").write_bytes(b"png")
    attempt = tmp_path / "attempt with spaces"
    policy = choose_colmap_policy(
        use_gpu=False,
        selected_count=1,
        attempt_index=0,
    )

    commands = build_colmap_commands(frames, attempt, "colmap", policy)

    assert len(commands) == 3
    assert all(isinstance(command, tuple) for command in commands)
    assert all(str(frames) in command for command in (commands[0], commands[2]))
    feature = next(command for command in commands if "feature_extractor" in command)
    matcher = next(command for command in commands if "sequential_matcher" in command)
    assert feature[-2:] == ("--SiftExtraction.use_gpu", "0")
    assert matcher[-4:] == (
        "--SiftMatching.use_gpu",
        "0",
        "--SequentialMatching.overlap",
        "20",
    )


def test_gpu_commands_use_exhaustive_without_cpu_flags(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    policy = choose_colmap_policy(
        use_gpu=True,
        selected_count=800,
        attempt_index=1,
    )

    commands = build_colmap_commands(frames, tmp_path / "attempt", "colmap", policy)
    flattened = tuple(argument for command in commands for argument in command)

    assert "exhaustive_matcher" in flattened
    assert "--SiftExtraction.use_gpu" not in flattened
    assert "--SiftMatching.use_gpu" not in flattened
    assert "--SequentialMatching.overlap" not in flattened


@pytest.mark.parametrize("existing_kind", ["directory", "file"])
def test_attempt_refuses_existing_directory_or_file(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    attempt = tmp_path / "attempt"
    if existing_kind == "directory":
        attempt.mkdir()
    else:
        attempt.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        run_colmap_attempt(
            tmp_path / "missing-frames",
            attempt,
            choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
        )


def test_attempt_refuses_dangling_symlink(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    try:
        attempt.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    assert os.path.lexists(attempt)
    with pytest.raises(FileExistsError):
        run_colmap_attempt(
            tmp_path / "missing-frames",
            attempt,
            choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
        )


def _frame_id(
    source_digest: str,
    source_index: int | None,
    timestamp_s: float | None,
    source_relative_path: str,
) -> str:
    return hashlib.sha256(
        (f"{source_digest}|{source_index}|{timestamp_s}|{source_relative_path}").encode(
            "utf-8"
        )
    ).hexdigest()[:24]


def _records_digest(records: tuple[FrameRecord, ...]) -> str:
    selected_payload = [
        [record.frame_id, record.output_name, record.sha256]
        for record in records
        if record.selected
    ]
    return hashlib.sha256(
        json.dumps(
            selected_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_contract_manifest(
    root: Path,
    *,
    source_digest: str,
    effective_mode: str,
    policy: SelectionPolicy,
    records: tuple[FrameRecord, ...],
    reconstruction_guardrail: str | None = None,
) -> str:
    digest = _records_digest(records)
    write_selection_manifest(
        root / "selection_manifest.json",
        SelectionManifest(
            schema_version=1,
            source_digest=source_digest,
            effective_mode=effective_mode,
            policy=policy,
            frames=records,
            image_set_digest=digest,
            reconstruction_guardrail=reconstruction_guardrail,
        ),
    )
    return digest


def _write_selected_frames(root: Path, frame_bytes: bytes = b"png") -> Path:
    root.mkdir()
    output_name = "frame_000000.png"
    source_digest = "b" * 64
    source_relative_path = "capture.mp4"
    source_index = 0
    timestamp_s = 0.0
    frame_id = _frame_id(
        source_digest,
        source_index,
        timestamp_s,
        source_relative_path,
    )
    frame_sha256 = hashlib.sha256(frame_bytes).hexdigest()
    (root / output_name).write_bytes(frame_bytes)
    selected_payload = [[frame_id, output_name, frame_sha256]]
    image_set_digest = hashlib.sha256(
        json.dumps(
            selected_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    (root / "selection_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_digest": source_digest,
                "effective_mode": "smart",
                "policy": {
                    "mode": "smart",
                    "frame_budget": 1,
                    "resolution_long_edge_cap": 1280,
                    "fixed_fps": 4,
                    "candidate_fps": 12,
                    "candidate_long_edge": 320,
                    "version": "selection-v1",
                },
                "frames": [
                    {
                        "frame_id": frame_id,
                        "source_relative_path": source_relative_path,
                        "source_index": source_index,
                        "source_pts": 0,
                        "timestamp_s": timestamp_s,
                        "output_name": output_name,
                        "sha256": frame_sha256,
                        "selected": True,
                        "metrics": {
                            "sharpness": 1.0,
                            "exposure_score": 0.9,
                            "duplicate_similarity": 0.0,
                            "overlap_score": 0.8,
                        },
                        "selection_score": 1.0,
                        "reasons": [],
                    }
                ],
                "image_set_digest": image_set_digest,
                "reconstruction_guardrail": None,
            }
        ),
        encoding="utf-8",
    )
    return root


def _selection_digest(frames_dir: Path) -> str:
    payload = json.loads(
        (frames_dir / "selection_manifest.json").read_text(encoding="utf-8")
    )
    return str(payload["image_set_digest"])


def test_selection_contract_accepts_serialized_fixed_fps_manifest(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "fixed-frames"
    frames.mkdir()
    source_digest = "c" * 64
    frame_bytes = b"fixed"
    frame_hash = hashlib.sha256(frame_bytes).hexdigest()
    (frames / "frame_000000.png").write_bytes(frame_bytes)
    records = (
        FrameRecord(
            frame_id=_frame_id(source_digest, 7, -0.25, "capture.mp4"),
            source_relative_path="capture.mp4",
            source_index=7,
            source_pts=-2,
            timestamp_s=-0.25,
            output_name="frame_000000.png",
            sha256=frame_hash,
            selected=True,
            metrics=None,
            selection_score=None,
            reasons=("fixed_fps",),
        ),
    )
    digest = _write_contract_manifest(
        frames,
        source_digest=source_digest,
        effective_mode="fixed_fps",
        policy=SelectionPolicy(
            mode="fixed_fps",
            frame_budget=1,
            resolution_long_edge_cap=1280,
        ),
        records=records,
    )

    assert colmap_module._read_selection_digest(frames) == digest


def test_selection_contract_accepts_serialized_smart_photo_manifest(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "smart-photo-frames"
    frames.mkdir()
    source_digest = "d" * 64
    frame_bytes = b"smart-photo"
    frame_hash = hashlib.sha256(frame_bytes).hexdigest()
    (frames / "frame_000000.png").write_bytes(frame_bytes)
    records = (
        FrameRecord(
            frame_id=_frame_id(source_digest, None, None, "photos/object 01.jpg"),
            source_relative_path="photos/object 01.jpg",
            source_index=None,
            source_pts=None,
            timestamp_s=None,
            output_name="frame_000000.png",
            sha256=frame_hash,
            selected=True,
            metrics=FrameMetrics(1.0, 0.9, 0.0, 0.8),
            selection_score=1.0,
            reasons=(),
        ),
    )
    digest = _write_contract_manifest(
        frames,
        source_digest=source_digest,
        effective_mode="smart",
        policy=SelectionPolicy(
            mode="smart",
            frame_budget=1,
            resolution_long_edge_cap=1280,
        ),
        records=records,
    )

    assert colmap_module._read_selection_digest(frames) == digest


def test_selection_contract_accepts_serialized_bounded_photo_manifest(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "bounded-photo-frames"
    frames.mkdir()
    source_digest = "e" * 64
    frame_bytes = b"bounded-photo"
    frame_hash = hashlib.sha256(frame_bytes).hexdigest()
    (frames / "frame_000000.png").write_bytes(frame_bytes)
    records = (
        FrameRecord(
            frame_id=_frame_id(source_digest, None, None, "photo_1.jpg"),
            source_relative_path="photo_1.jpg",
            source_index=None,
            source_pts=None,
            timestamp_s=None,
            output_name="frame_000000.png",
            sha256=frame_hash,
            selected=True,
            metrics=None,
            selection_score=None,
            reasons=("photo_set_all", "uniform_profile_cap"),
        ),
        FrameRecord(
            frame_id=_frame_id(source_digest, None, None, "photo_2.jpg"),
            source_relative_path="photo_2.jpg",
            source_index=None,
            source_pts=None,
            timestamp_s=None,
            output_name="",
            sha256="",
            selected=False,
            metrics=None,
            selection_score=None,
            reasons=("photo_set_all", "colmap_budget_guardrail"),
        ),
    )
    digest = _write_contract_manifest(
        frames,
        source_digest=source_digest,
        effective_mode="photo_set_all",
        policy=SelectionPolicy(
            mode="fixed_fps",
            frame_budget=1,
            resolution_long_edge_cap=1280,
        ),
        records=records,
        reconstruction_guardrail="uniform_profile_cap",
    )

    assert colmap_module._read_selection_digest(frames) == digest


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_manifest",
        "malformed_manifest",
        "wrong_schema",
        "invalid_digest",
        "modified_png",
        "missing_png",
        "extra_png",
        "extra_supported_image",
        "extra_directory",
        "extra_manifest_field",
        "missing_manifest_field",
        "unhashable_effective_mode",
        "invalid_policy",
        "extra_policy_field",
        "missing_policy_field",
        "policy_mode_mismatch",
        "unhashable_policy_mode",
        "invalid_policy_version",
        "invalid_frame_metadata",
        "extra_frame_field",
        "missing_frame_field",
        "unsafe_source_path",
        "inconsistent_source_metadata",
        "invalid_metrics_schema",
        "nonfinite_metric",
        "invalid_selection_score",
        "duplicate_reasons",
        "invalid_guardrail",
        "record_hash_mismatch",
        "image_set_digest_mismatch",
    ],
)
def test_attempt_rejects_invalid_selection_contract_before_subprocess_or_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    manifest_path = frames / "selection_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corruption == "missing_manifest":
        manifest_path.unlink()
    elif corruption == "malformed_manifest":
        manifest_path.write_text("not-json", encoding="utf-8")
    elif corruption == "wrong_schema":
        payload["schema_version"] = True
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "invalid_digest":
        payload["image_set_digest"] = "not-a-digest"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "modified_png":
        (frames / "frame_000000.png").write_bytes(b"modified")
    elif corruption == "missing_png":
        (frames / "frame_000000.png").unlink()
    elif corruption == "extra_png":
        (frames / "frame_999999.png").write_bytes(b"extra")
    elif corruption == "extra_supported_image":
        (frames / "untracked.jpg").write_bytes(b"extra")
    elif corruption == "extra_directory":
        (frames / "untracked-directory").mkdir()
    elif corruption == "extra_manifest_field":
        payload["unexpected"] = None
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "missing_manifest_field":
        payload.pop("source_digest")
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "unhashable_effective_mode":
        payload["effective_mode"] = []
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "invalid_policy":
        payload["policy"]["frame_budget"] = True
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "extra_policy_field":
        payload["policy"]["unexpected"] = None
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "missing_policy_field":
        payload["policy"].pop("candidate_fps")
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "policy_mode_mismatch":
        payload["policy"]["mode"] = "fixed_fps"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "unhashable_policy_mode":
        payload["policy"]["mode"] = []
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "invalid_policy_version":
        payload["policy"]["version"] = True
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "invalid_frame_metadata":
        payload["frames"][0]["reasons"] = "not-a-list"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "extra_frame_field":
        payload["frames"][0]["unexpected"] = None
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "missing_frame_field":
        payload["frames"][0].pop("source_pts")
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "unsafe_source_path":
        payload["frames"][0]["source_relative_path"] = "../capture.mp4"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "inconsistent_source_metadata":
        payload["frames"][0]["source_index"] = None
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "invalid_metrics_schema":
        payload["frames"][0]["metrics"].pop("overlap_score")
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "nonfinite_metric":
        payload["frames"][0]["metrics"]["sharpness"] = float("nan")
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "invalid_selection_score":
        payload["frames"][0]["selection_score"] = None
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "duplicate_reasons":
        payload["frames"][0]["reasons"] = ["blurred", "blurred"]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "invalid_guardrail":
        payload["reconstruction_guardrail"] = "unknown"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "record_hash_mismatch":
        payload["frames"][0]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif corruption == "image_set_digest_mismatch":
        payload["image_set_digest"] = "0" * 64
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(corruption)

    subprocess_calls = 0

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        nonlocal subprocess_calls
        subprocess_calls += 1

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    attempt = tmp_path / "attempt"

    with pytest.raises((FileNotFoundError, ValueError)):
        run_colmap_attempt(
            frames,
            attempt,
            choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
            colmap_exe="colmap",
        )

    assert subprocess_calls == 0
    assert not os.path.lexists(attempt)


def test_attempt_rejects_symlinked_subtree_in_selected_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    try:
        (frames / "foreign-subtree").symlink_to(
            tmp_path / "missing-subtree",
            target_is_directory=True,
        )
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    calls = 0

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(subprocess, "run", unexpected_run)

    with pytest.raises(ValueError):
        run_colmap_attempt(
            frames,
            tmp_path / "attempt",
            choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
            colmap_exe="colmap",
        )
    assert calls == 0


def test_attempt_runs_fresh_commands_converts_every_model_and_fingerprints_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    selection_digest = _selection_digest(frames)
    attempt = tmp_path / "attempt"
    commands: list[tuple[str, ...]] = []
    progress: list[tuple[float, str]] = []

    def fake_run(
        command: tuple[str, ...],
        *,
        check: bool,
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert isinstance(command, tuple)
        commands.append(command)
        if command[1] == "--version":
            assert kwargs == {"capture_output": True, "text": True}
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="COLMAP 3.11.1\n",
                stderr="",
            )
        assert kwargs == {}
        if command[1] == "feature_extractor":
            (attempt / "colmap.db").write_bytes(b"database")
        if command[1] == "mapper":
            assert (attempt / "sparse").is_dir()
            (attempt / "sparse" / "1").mkdir(parents=True)
            (attempt / "sparse" / "0").mkdir()
        if command[1] == "model_converter":
            model_dir = Path(command[command.index("--input_path") + 1])
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                (model_dir / name).write_text("# converted\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    policy = choose_colmap_policy(
        use_gpu=False,
        selected_count=1,
        attempt_index=0,
    )

    result = run_colmap_attempt(
        frames,
        attempt,
        policy,
        colmap_exe="colmap",
        on_progress=lambda fraction, message: progress.append((fraction, message)),
    )

    assert result.root == attempt
    assert result.database_path == attempt / "colmap.db"
    assert [model.name for model in result.model_dirs] == ["0", "1"]
    assert result.colmap_version == "COLMAP 3.11.1"
    assert result.fingerprint == stage_fingerprint(
        "colmap",
        inputs={"selection": selection_digest},
        settings=asdict(policy),
        policy_version=policy.version,
        tools={"colmap": "COLMAP 3.11.1"},
    )
    converters = [command for command in commands if "model_converter" in command]
    assert len(converters) == 2
    assert all(command[-2:] == ("--output_type", "TXT") for command in converters)
    assert [command[1] for command in commands[:4]] == [
        "--version",
        "feature_extractor",
        "sequential_matcher",
        "mapper",
    ]
    assert [fraction for fraction, _message in progress] == sorted(
        fraction for fraction, _message in progress
    )
    assert progress[-1][0] == 1.0


def test_attempt_fingerprint_changes_with_selection_policy_and_colmap_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_policy = choose_colmap_policy(
        use_gpu=True,
        selected_count=1,
        attempt_index=0,
    )

    def run(
        name: str,
        *,
        frame_bytes: bytes,
        policy=base_policy,
        version: str = "COLMAP 3.11\n",
    ) -> str:
        frames = _write_selected_frames(tmp_path / f"frames-{name}", frame_bytes)
        attempt = tmp_path / f"attempt-{name}"
        monkeypatch.setattr(
            subprocess,
            "run",
            _successful_attempt_run(attempt, version_stdout=version),
        )
        return run_colmap_attempt(
            frames,
            attempt,
            policy,
            colmap_exe="colmap",
        ).fingerprint

    base = run("base", frame_bytes=b"one")
    changed_selection = run("selection", frame_bytes=b"two")
    changed_policy = run(
        "policy",
        frame_bytes=b"one",
        policy=replace(base_policy, camera_model="OPENCV"),
    )
    changed_version = run(
        "version",
        frame_bytes=b"one",
        version="COLMAP 3.12\n",
    )

    assert len({base, changed_selection, changed_policy, changed_version}) == 4


def test_progress_stays_monotonic_with_many_sparse_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    attempt = tmp_path / "attempt"
    progress: list[float] = []

    def fake_run(
        command: tuple[str, ...],
        *,
        check: bool,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="COLMAP 3.11")
        if command[1] == "feature_extractor":
            (attempt / "colmap.db").write_bytes(b"database")
        if command[1] == "mapper":
            for index in range(5):
                (attempt / "sparse" / str(index)).mkdir(parents=True)
        if command[1] == "model_converter":
            model = Path(command[command.index("--input_path") + 1])
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                (model / name).write_text("# converted\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_colmap_attempt(
        frames,
        attempt,
        choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
        colmap_exe="colmap",
        on_progress=lambda fraction, _message: progress.append(fraction),
    )

    assert progress == sorted(progress)
    assert progress[-1] == 1.0


def test_attempt_fails_when_colmap_produces_no_sparse_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")

    def fake_run(
        command: tuple[str, ...],
        *,
        check: bool,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        if command[1] == "feature_extractor":
            (tmp_path / "attempt" / "colmap.db").write_bytes(b"database")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="COLMAP 3.11\n" if command[1] == "--version" else "",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="no sparse model"):
        run_colmap_attempt(
            frames,
            tmp_path / "attempt",
            choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
            colmap_exe="colmap",
        )


def _successful_attempt_run(
    attempt: Path,
    *,
    version_stdout: str = "COLMAP 3.11\n",
    version_stderr: str = "",
    create_database: bool = True,
    write_text_models: bool = True,
):
    def fake_run(
        command: tuple[str, ...],
        *,
        check: bool,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        if command[1] == "--version":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=version_stdout,
                stderr=version_stderr,
            )
        if command[1] == "feature_extractor" and create_database:
            (attempt / "colmap.db").write_bytes(b"database")
        if command[1] == "mapper":
            (attempt / "sparse" / "0").mkdir(parents=True)
        if command[1] == "model_converter" and write_text_models:
            model = Path(command[command.index("--input_path") + 1])
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                (model / name).write_text("# converted\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return fake_run


def test_version_uses_trimmed_stderr_when_stdout_is_whitespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    attempt = tmp_path / "attempt"
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_attempt_run(
            attempt,
            version_stdout="  \n",
            version_stderr="COLMAP 3.12 stderr\n",
        ),
    )

    result = run_colmap_attempt(
        frames,
        attempt,
        choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
        colmap_exe="colmap",
    )

    assert result.colmap_version == "COLMAP 3.12 stderr"


@pytest.mark.parametrize("failure", ["missing_database", "missing_text_model"])
def test_attempt_rejects_incomplete_colmap_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    attempt = tmp_path / "attempt"
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_attempt_run(
            attempt,
            create_database=failure != "missing_database",
            write_text_models=failure != "missing_text_model",
        ),
    )

    with pytest.raises(RuntimeError):
        run_colmap_attempt(
            frames,
            attempt,
            choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
            colmap_exe="colmap",
        )


@pytest.mark.parametrize("empty_version", [True, False])
def test_version_failure_does_not_create_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_version: bool,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    attempt = tmp_path / "attempt"

    def fail_version(
        command: tuple[str, ...],
        *,
        check: bool,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        if not empty_version:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout=" \n", stderr="\t")

    monkeypatch.setattr(subprocess, "run", fail_version)

    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        run_colmap_attempt(
            frames,
            attempt,
            choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
            colmap_exe="colmap",
        )

    assert not os.path.lexists(attempt)


def test_failed_attempt_is_preserved_and_cannot_be_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    attempt = tmp_path / "attempt"
    calls = 0

    def fail_matcher(
        command: tuple[str, ...],
        *,
        check: bool,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        assert check is True
        calls += 1
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="COLMAP 3.11")
        if command[1] == "feature_extractor":
            (attempt / "colmap.db").write_bytes(b"database")
        if command[1].endswith("_matcher"):
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fail_matcher)
    policy = choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0)

    with pytest.raises(subprocess.CalledProcessError):
        run_colmap_attempt(frames, attempt, policy, colmap_exe="colmap")
    assert attempt.is_dir()
    calls_after_failure = calls

    with pytest.raises(FileExistsError):
        run_colmap_attempt(frames, attempt, policy, colmap_exe="colmap")
    assert calls == calls_after_failure


def test_attempt_detects_post_creation_root_replacement_before_colmap_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    attempt = tmp_path / "attempt"
    displaced = tmp_path / "displaced-owned-attempt"
    foreign_marker = attempt / "foreign-marker"
    original_builder = colmap_module.build_colmap_commands
    subprocess_calls = 0

    def replace_attempt_root(*args, **kwargs):
        commands = original_builder(*args, **kwargs)
        attempt.rename(displaced)
        attempt.mkdir()
        foreign_marker.write_bytes(b"foreign")
        return commands

    def version_only(
        command: tuple[str, ...],
        *,
        check: bool,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal subprocess_calls
        assert check is True
        subprocess_calls += 1
        return subprocess.CompletedProcess(command, 0, stdout="COLMAP 3.11")

    monkeypatch.setattr(colmap_module, "build_colmap_commands", replace_attempt_root)
    monkeypatch.setattr(subprocess, "run", version_only)

    with pytest.raises(RuntimeError, match="identity"):
        run_colmap_attempt(
            frames,
            attempt,
            choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
            colmap_exe="colmap",
        )

    assert subprocess_calls == 1
    assert foreign_marker.read_bytes() == b"foreign"


def test_attempt_rechecks_identity_after_progress_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    attempt = tmp_path / "attempt"
    displaced = tmp_path / "displaced-owned-attempt"
    foreign_marker = attempt / "foreign-marker"
    subprocess_calls = 0

    def version_only(
        command: tuple[str, ...],
        *,
        check: bool,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal subprocess_calls
        assert check is True
        subprocess_calls += 1
        return subprocess.CompletedProcess(command, 0, stdout="COLMAP 3.11")

    def replace_attempt_root(_fraction: float, _message: str) -> None:
        attempt.rename(displaced)
        attempt.mkdir()
        foreign_marker.write_bytes(b"foreign")

    monkeypatch.setattr(subprocess, "run", version_only)

    with pytest.raises(RuntimeError, match="identity"):
        run_colmap_attempt(
            frames,
            attempt,
            choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
            colmap_exe="colmap",
            on_progress=replace_attempt_root,
        )

    assert subprocess_calls == 1
    assert foreign_marker.read_bytes() == b"foreign"


def test_attempt_revalidates_selected_frames_after_progress_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    displaced = tmp_path / "displaced-frames"
    attempt = tmp_path / "attempt"
    original_frame = (frames / "frame_000000.png").read_bytes()
    original_manifest = (frames / "selection_manifest.json").read_bytes()
    mutated = False
    subprocess_calls = 0

    def successful_run(
        command: tuple[str, ...],
        *,
        check: bool,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal subprocess_calls
        assert check is True
        subprocess_calls += 1
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="COLMAP 3.11")
        if command[1] == "feature_extractor":
            image_root = Path(command[command.index("--image_path") + 1])
            assert (image_root / "untracked.jpg").is_file()
            (attempt / "colmap.db").write_bytes(b"database")
        if command[1] == "mapper":
            (attempt / "sparse" / "0").mkdir()
        if command[1] == "model_converter":
            model = Path(command[command.index("--input_path") + 1])
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                (model / name).write_text("# converted\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    def replace_selected_frames(_fraction: float, _message: str) -> None:
        nonlocal mutated
        if mutated:
            return
        frames.rename(displaced)
        frames.mkdir()
        (frames / "frame_000000.png").write_bytes(original_frame)
        (frames / "selection_manifest.json").write_bytes(original_manifest)
        (frames / "untracked.jpg").write_bytes(b"foreign")
        mutated = True

    monkeypatch.setattr(subprocess, "run", successful_run)

    with pytest.raises(ValueError):
        run_colmap_attempt(
            frames,
            attempt,
            choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
            colmap_exe="colmap",
            on_progress=replace_selected_frames,
        )

    assert subprocess_calls == 1
    assert (frames / "untracked.jpg").read_bytes() == b"foreign"


def test_attempt_rejects_database_injected_by_progress_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    attempt = tmp_path / "attempt"
    injected = False
    subprocess_calls = 0

    def successful_run(
        command: tuple[str, ...],
        *,
        check: bool,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal subprocess_calls
        assert check is True
        subprocess_calls += 1
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="COLMAP 3.11")
        if command[1] == "feature_extractor":
            assert (attempt / "colmap.db").read_bytes() == b"stale-database"
            (attempt / "colmap.db").write_bytes(b"database")
        if command[1] == "mapper":
            (attempt / "sparse" / "0").mkdir()
        if command[1] == "model_converter":
            model = Path(command[command.index("--input_path") + 1])
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                (model / name).write_text("# converted\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    def inject_database(_fraction: float, _message: str) -> None:
        nonlocal injected
        if injected:
            return
        (attempt / "colmap.db").write_bytes(b"stale-database")
        injected = True

    monkeypatch.setattr(subprocess, "run", successful_run)

    with pytest.raises(RuntimeError, match="fresh|unexpected"):
        run_colmap_attempt(
            frames,
            attempt,
            choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
            colmap_exe="colmap",
            on_progress=inject_database,
        )

    assert subprocess_calls == 1
    assert (attempt / "colmap.db").read_bytes() == b"stale-database"


def test_attempt_rejects_in_place_database_overwrite_before_matcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _write_selected_frames(tmp_path / "frames")
    attempt = tmp_path / "attempt"
    subprocess_calls = 0

    def successful_run(
        command: tuple[str, ...],
        *,
        check: bool,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal subprocess_calls
        assert check is True
        subprocess_calls += 1
        if command[1] == "--version":
            return subprocess.CompletedProcess(command, 0, stdout="COLMAP 3.11")
        if command[1] == "feature_extractor":
            (attempt / "colmap.db").write_bytes(b"feature-database")
        if command[1].endswith("_matcher"):
            assert (attempt / "colmap.db").read_bytes() == b"stale-database"
        if command[1] == "mapper":
            (attempt / "sparse" / "0").mkdir()
        if command[1] == "model_converter":
            model = Path(command[command.index("--input_path") + 1])
            for name in ("cameras.txt", "images.txt", "points3D.txt"):
                (model / name).write_text("# converted\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    def overwrite_before_matcher(_fraction: float, message: str) -> None:
        if message.endswith("_matcher"):
            (attempt / "colmap.db").write_bytes(b"stale-database")

    monkeypatch.setattr(subprocess, "run", successful_run)

    with pytest.raises(RuntimeError, match="database.*changed"):
        run_colmap_attempt(
            frames,
            attempt,
            choose_colmap_policy(use_gpu=True, selected_count=1, attempt_index=0),
            colmap_exe="colmap",
            on_progress=overwrite_before_matcher,
        )

    assert subprocess_calls == 2
    assert (attempt / "colmap.db").read_bytes() == b"stale-database"


def test_gpu_probe_falls_back_to_cpu_and_removes_failed_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = tmp_path / "probe"

    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["colmap"])

    monkeypatch.setattr(subprocess, "run", fail_run)

    assert probe_colmap_gpu_support("colmap", probe) is False
    assert not probe.exists()


def test_gpu_probe_cleans_a_partially_written_checkerboard_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = tmp_path / "probe"
    subprocess_calls = 0

    def partial_write(images_dir: Path) -> tuple[Path, ...]:
        (images_dir / "checkerboard_0.png").write_bytes(b"partial")
        raise OSError("simulated image write failure")

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        nonlocal subprocess_calls
        subprocess_calls += 1

    monkeypatch.setattr(colmap_module, "_write_checkerboards", partial_write)
    monkeypatch.setattr(subprocess, "run", unexpected_run)

    assert probe_colmap_gpu_support("colmap", probe) is False
    assert subprocess_calls == 0
    assert not os.path.lexists(probe)


def test_gpu_probe_requires_successful_database_and_uses_checked_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, ...]] = []
    checkerboard_payloads: list[tuple[bytes, ...]] = []

    def fake_run(
        command: tuple[str, ...],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert isinstance(command, tuple)
        seen.append(command)
        images = Path(command[command.index("--image_path") + 1])
        checkerboard_payloads.append(
            tuple(path.read_bytes() for path in sorted(images.glob("*.png")))
        )
        database = Path(command[command.index("--database_path") + 1])
        database.write_bytes(b"database")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    first_probe = tmp_path / "probe-first"
    second_probe = tmp_path / "probe-second"
    assert probe_colmap_gpu_support("colmap", first_probe) is True
    assert probe_colmap_gpu_support("colmap", second_probe) is True
    assert len(seen) == 2
    assert seen[0][1] == "feature_extractor"
    assert seen[0][-2:] == ("--SiftExtraction.use_gpu", "1")
    assert len(checkerboard_payloads[0]) == 2
    assert checkerboard_payloads[0] == checkerboard_payloads[1]
    assert not first_probe.exists()
    assert not second_probe.exists()


def test_gpu_probe_treats_missing_database_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = tmp_path / "probe"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, check: subprocess.CompletedProcess(command, 0),
    )

    assert probe_colmap_gpu_support("colmap", probe) is False
    assert not probe.exists()


def test_gpu_probe_does_not_delete_raced_in_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = tmp_path / "probe"
    marker = probe / "foreign-marker"
    original_mkdir = Path.mkdir

    def raced_mkdir(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if path == probe:
            original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)
            marker.write_bytes(b"foreign")
            raise FileExistsError(path)
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", raced_mkdir)

    assert probe_colmap_gpu_support("colmap", probe) is False
    assert marker.read_bytes() == b"foreign"


def test_gpu_probe_does_not_delete_post_creation_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = tmp_path / "probe"
    displaced = tmp_path / "displaced-owned-probe"
    marker = probe / "foreign-marker"

    def replace_probe_root(
        command: tuple[str, ...],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        probe.rename(displaced)
        probe.mkdir()
        marker.write_bytes(b"foreign")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", replace_probe_root)

    assert probe_colmap_gpu_support("colmap", probe) is False
    assert marker.read_bytes() == b"foreign"


def test_gpu_probe_rejects_preexisting_path_without_touching_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    probe = tmp_path / "probe"
    probe.mkdir()
    marker = probe / "marker"
    marker.write_bytes(b"preserve")
    calls = 0

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    caplog.set_level("WARNING")

    assert probe_colmap_gpu_support("colmap", probe) is False

    assert calls == 0
    assert marker.read_bytes() == b"preserve"
    assert "already exists" in caplog.text

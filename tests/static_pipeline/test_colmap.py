from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from backend.static_pipeline.colmap import (
    build_colmap_commands,
    choose_colmap_policy,
    probe_colmap_gpu_support,
    run_colmap_attempt,
)
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


def _write_selected_frames(root: Path, frame_bytes: bytes = b"png") -> Path:
    root.mkdir()
    output_name = "frame_000000.png"
    frame_id = "a" * 24
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
                "source_digest": "b" * 64,
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
                        "source_relative_path": "capture.mp4",
                        "source_index": 0,
                        "source_pts": 0,
                        "timestamp_s": 0.0,
                        "output_name": output_name,
                        "sha256": frame_sha256,
                        "selected": True,
                        "metrics": None,
                        "selection_score": 1.0,
                        "reasons": ["smart_window"],
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


def test_gpu_probe_rejects_preexisting_path_without_touching_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    with pytest.raises(FileExistsError):
        probe_colmap_gpu_support("colmap", probe)

    assert calls == 0
    assert marker.read_bytes() == b"preserve"

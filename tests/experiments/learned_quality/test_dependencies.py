from __future__ import annotations

import json
import os
from hashlib import sha256
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from typing import Mapping, cast

import pytest

from experiments.learned_quality.contracts import ModelRef
from experiments.learned_quality.dependencies import (
    CHECKPOINT_MODEL_REFS,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_ENV_ROOT,
    DEFAULT_LOCK_PATH,
    DEFAULT_SOURCE_ROOT,
    PROTECTED_PACKAGES,
    SOURCE_REPOSITORY_REFS,
    LearnedAssets,
    LearnedEnvironment,
    ModelManifest,
    PinnedModelSnapshot,
    ProtectedRuntimeError,
    ProtectedRuntimeSnapshot,
    RuntimePackage,
    SourceCheckout,
    assert_no_protected_changes,
    capture_protected_runtime,
    install_learned_environment,
    materialize_pinned_assets,
    snapshot_pinned_model,
    verify_learned_environment,
)


LOCK_LINES = (
    "transformers==4.57.6",
    "huggingface-hub==0.36.2",
    "tokenizers==0.22.1",
    "safetensors==0.6.2",
    "pycolmap==3.12.6",
    "omegaconf==2.3.0",
    "hydra-core==1.3.2",
    "iopath==0.1.10",
    "portalocker==3.2.0",
    "addict==2.4.0",
    "moviepy==1.0.3",
    "trimesh==4.7.4",
    "evo==1.36.5",
)


def _runtime_rows() -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "version": f"1.0.{index}",
            "module_path": f"/runtime/{name}/__init__.py",
        }
        for index, name in enumerate(PROTECTED_PACKAGES)
    ]


def _runtime_stdout(rows: list[dict[str, object]] | None = None) -> str:
    return json.dumps({"packages": rows if rows is not None else _runtime_rows()})


def _snapshot(
    *,
    versions: Mapping[str, str] | None = None,
    paths: Mapping[str, str] | None = None,
) -> ProtectedRuntimeSnapshot:
    versions = versions or {}
    paths = paths or {}
    return ProtectedRuntimeSnapshot(
        python_exe=Path("C:/runtime/python"),
        packages={
            name: RuntimePackage(
                name=name,
                version=versions.get(name, f"1.0.{index}"),
                module_path=paths.get(name, f"/runtime/{name}/__init__.py"),
            )
            for index, name in enumerate(PROTECTED_PACKAGES)
        },
    )


def _write_lock(root: Path, lines: tuple[str, ...] = LOCK_LINES) -> Path:
    lock_path = root / "requirements-lock.txt"
    lock_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lock_path


def _learned_environment(root: Path) -> LearnedEnvironment:
    snapshot = _snapshot()
    return LearnedEnvironment(
        root=root,
        python_exe=root / "bin" / "python",
        lock_path=root.parent / "requirements-lock.txt",
        pip_report_path=root / "pip-dry-run-report.json",
        main_runtime_before=snapshot,
        main_runtime_after=snapshot,
    )


_SOURCE_NAMES = ("da3", "sam2", "sea-raft")
_SOURCE_BY_NAME = dict(zip(_SOURCE_NAMES, SOURCE_REPOSITORY_REFS, strict=True))


class _GitRunner:
    def __init__(
        self,
        source_root: Path,
        *,
        outputs: Mapping[tuple[str, str], object] | None = None,
        fail_step: str | None = None,
    ) -> None:
        self.source_root = source_root
        self.outputs = dict(outputs or {})
        self.fail_step = fail_step
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(
        self,
        argv: list[str],
        **kwargs: object,
    ) -> CompletedProcess[str]:
        self.calls.append((list(argv), dict(kwargs)))
        if argv[:4] == ["git", "clone", "--no-checkout", "--filter=blob:none"]:
            destination = Path(argv[-1])
            destination.mkdir(parents=False, exist_ok=False)
            if self.fail_step == "clone":
                raise CalledProcessError(1, argv, stderr="clone failed")
            return CompletedProcess(argv, 0, stdout="", stderr="")

        if len(argv) < 5 or argv[:2] != ["git", "-C"]:
            raise AssertionError(f"unexpected subprocess argv: {argv!r}")
        destination = Path(argv[2])
        name = destination.name
        source = _SOURCE_BY_NAME[name]
        command = argv[3:]
        if command[:2] == ["checkout", "--detach"]:
            if self.fail_step == "checkout":
                raise CalledProcessError(1, argv, stderr="checkout failed")
            stdout: object = ""
        elif command == ["remote", "get-url", "origin"]:
            stdout = self.outputs.get((name, "remote"), f"{source.repo_id}\n")
        elif command == ["rev-parse", "HEAD"]:
            stdout = self.outputs.get((name, "head"), f"{source.revision}\n")
        elif command == ["status", "--porcelain"]:
            stdout = self.outputs.get((name, "status"), "")
        else:
            raise AssertionError(f"unexpected subprocess argv: {argv!r}")
        return CompletedProcess(argv, 0, stdout=cast(str, stdout), stderr="")


class _InstallRunner:
    def __init__(
        self,
        root: Path,
        *,
        report_payload: str = '{"install":[]}',
        report_mode: str = "write",
        venv_report_kind: str | None = None,
        fail_stage: str | None = None,
        before_stdout: str | None = None,
        after_stdout: str | None = None,
    ) -> None:
        self.root = root
        self.report_payload = report_payload
        self.report_mode = report_mode
        self.venv_report_kind = venv_report_kind
        self.fail_stage = fail_stage
        self.before_stdout = before_stdout or _runtime_stdout()
        self.after_stdout = after_stdout or self.before_stdout
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.capture_count = 0
        self.report_existed_before_dry_run: bool | None = None

    def __call__(
        self,
        argv: list[str],
        **kwargs: object,
    ) -> CompletedProcess[str]:
        self.calls.append((list(argv), dict(kwargs)))
        if len(argv) >= 4 and argv[1:3] == ["-I", "-c"]:
            stdout = (
                self.before_stdout if self.capture_count == 0 else self.after_stdout
            )
            self.capture_count += 1
            return CompletedProcess(argv, 0, stdout=stdout, stderr="")

        if argv[1:5] == ["-m", "venv", "--system-site-packages", "--clear"]:
            if self.fail_stage == "venv":
                raise CalledProcessError(1, argv, stderr="venv failed")
            self.root.mkdir(parents=True, exist_ok=True)
            worker_bin = self.root / "bin"
            worker_bin.mkdir(exist_ok=True)
            (worker_bin / "python").write_text("worker", encoding="utf-8")
            report = self.root / "pip-dry-run-report.json"
            if self.venv_report_kind == "regular":
                report.write_text("stale", encoding="utf-8")
            elif self.venv_report_kind == "directory":
                report.mkdir()
            elif self.venv_report_kind == "symlink":
                target = self.root / "report-target.json"
                target.write_text("target", encoding="utf-8")
                report.symlink_to(target)
            return CompletedProcess(argv, 0, stdout="", stderr="")

        if "--dry-run" in argv:
            if self.fail_stage == "dry-run":
                raise CalledProcessError(1, argv, stderr="dry-run failed")
            report = Path(argv[argv.index("--report") + 1])
            self.report_existed_before_dry_run = report.exists()
            if self.report_mode == "write":
                report.write_text(self.report_payload, encoding="utf-8")
            return CompletedProcess(argv, 0, stdout="", stderr="")

        if "--no-deps" in argv:
            if self.fail_stage == "install":
                raise CalledProcessError(1, argv, stderr="install failed")
            return CompletedProcess(argv, 0, stdout="", stderr="")

        raise AssertionError(f"unexpected subprocess argv: {argv!r}")


def test_dependency_lock_is_exact_and_has_no_extra_entries() -> None:
    root = Path(__file__).parents[3]
    lock = root / "experiments" / "learned_quality" / "requirements-lock.txt"

    assert lock.read_text(encoding="utf-8") == "\n".join(LOCK_LINES) + "\n"


def test_protected_package_names_are_exact() -> None:
    assert PROTECTED_PACKAGES == ("torch", "torchvision", "numpy", "gsplat")


def test_source_repository_refs_are_exact_and_immutable() -> None:
    assert SOURCE_REPOSITORY_REFS == (
        ModelRef(
            repo_id="https://github.com/ByteDance-Seed/Depth-Anything-3.git",
            revision="3fe327a6abe2e5db95b54444ea95463dbfef5610",
            code_commit=None,
            license_id="Apache-2.0",
        ),
        ModelRef(
            repo_id="https://github.com/facebookresearch/sam2.git",
            revision="2b90b9f5ceec907a1c18123530e92e794ad901a4",
            code_commit=None,
            license_id="Apache-2.0",
        ),
        ModelRef(
            repo_id="https://github.com/MemorySlices/SEA-RAFT.git",
            revision="9137517ba24e628442aec097d3afe71d03503b75",
            code_commit=None,
            license_id="BSD-3-Clause",
        ),
    )
    assert isinstance(SOURCE_REPOSITORY_REFS, tuple)


def test_checkpoint_model_refs_are_exact_and_immutable() -> None:
    da3_commit = "3fe327a6abe2e5db95b54444ea95463dbfef5610"
    sam2_commit = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
    sea_raft_commit = "9137517ba24e628442aec097d3afe71d03503b75"
    assert CHECKPOINT_MODEL_REFS == (
        ModelRef(
            repo_id="depth-anything/DA3-BASE",
            revision="f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
            code_commit=da3_commit,
            license_id="Apache-2.0",
        ),
        ModelRef(
            repo_id="depth-anything/DA3METRIC-LARGE",
            revision="4010e39f3634a45bc60553321fb49fb760bd594e",
            code_commit=da3_commit,
            license_id="Apache-2.0",
        ),
        ModelRef(
            repo_id="IDEA-Research/grounding-dino-tiny",
            revision="a2bb814dd30d776dcf7e30523b00659f4f141c71",
            code_commit=None,
            license_id="Apache-2.0",
        ),
        ModelRef(
            repo_id="facebook/sam2.1-hiera-large",
            revision="665f8e2ad61cf5f53d65644ff27c8ee525124610",
            code_commit=sam2_commit,
            license_id="Apache-2.0",
        ),
        ModelRef(
            repo_id="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
            revision="eb97ef34ba5d856c3fa2cdcd073150c057ac8b69",
            code_commit=sea_raft_commit,
            license_id="BSD-3-Clause",
        ),
    )
    assert isinstance(CHECKPOINT_MODEL_REFS, tuple)


@pytest.mark.parametrize(
    ("supplied_name", "protected_name"),
    [
        ("Torch", "torch"),
        ("torch-vision", "torchvision"),
        ("torch_vision", "torchvision"),
        ("Num_Py", "numpy"),
        ("g-splat", "gsplat"),
    ],
)
def test_protected_package_in_install_plan_is_rejected(
    supplied_name: str,
    protected_name: str,
) -> None:
    report = {
        "install": [{"metadata": {"name": supplied_name, "version": "9.9.9"}}]
    }

    with pytest.raises(ProtectedRuntimeError, match=protected_name):
        assert_no_protected_changes(report)


@pytest.mark.parametrize(
    ("collection", "entry", "protected_name"),
    [
        ("remove", {"name": "NUMPY"}, "numpy"),
        ("uninstall", {"metadata": {"name": "g_splat"}}, "gsplat"),
    ],
)
def test_protected_package_in_removal_plan_is_rejected(
    collection: str,
    entry: dict[str, object],
    protected_name: str,
) -> None:
    with pytest.raises(ProtectedRuntimeError, match=protected_name):
        assert_no_protected_changes({collection: [entry]})


@pytest.mark.parametrize(
    "report",
    [
        {"install": {}},
        {"install": [None]},
        {"install": [{}]},
        {"install": [{"metadata": []}]},
        {"install": [{"metadata": {"name": ""}}]},
        {"remove": [{"name": 123}]},
    ],
)
def test_malformed_pip_plan_is_rejected(report: dict[str, object]) -> None:
    with pytest.raises(ProtectedRuntimeError, match="report"):
        assert_no_protected_changes(report)


def test_unprotected_plan_and_identical_snapshots_are_accepted() -> None:
    before = _snapshot()
    after = _snapshot()
    report = {
        "install": [{"metadata": {"name": "transformers"}}],
        "remove": [{"name": "old-helper"}],
        "uninstall": [],
    }

    assert assert_no_protected_changes(report, before=before, after=after) is None


@pytest.mark.parametrize(
    ("package_name", "versions", "paths"),
    [
        ("torch", {"torch": "2.7.0"}, {}),
        ("numpy", {}, {"numpy": "/other/numpy/__init__.py"}),
    ],
)
def test_runtime_version_or_module_path_change_is_rejected(
    package_name: str,
    versions: dict[str, str],
    paths: dict[str, str],
) -> None:
    before = _snapshot()
    after = _snapshot(versions=versions, paths=paths)

    with pytest.raises(ProtectedRuntimeError, match=package_name):
        assert_no_protected_changes({}, before=before, after=after)


def test_one_sided_runtime_comparison_is_rejected() -> None:
    with pytest.raises(ProtectedRuntimeError, match="before.*after"):
        assert_no_protected_changes({}, before=_snapshot())


def test_capture_protected_runtime_uses_one_safe_fresh_subprocess() -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(argv: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append((argv, kwargs))
        return CompletedProcess(
            args=argv,
            returncode=0,
            stdout=_runtime_stdout(),
            stderr="",
        )

    python_exe = Path("/runtime/bin/python")
    snapshot = capture_protected_runtime(python_exe, runner=runner)

    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert isinstance(argv, list)
    assert argv[0] == str(python_exe)
    assert "-c" in argv
    assert kwargs == {"check": True, "capture_output": True, "text": True}
    assert "shell" not in kwargs
    assert snapshot.python_exe == python_exe
    assert tuple(snapshot.packages) == PROTECTED_PACKAGES
    assert snapshot.packages["torch"].module_path == "/runtime/torch/__init__.py"


@pytest.mark.parametrize(
    "malformation",
    ["missing", "duplicate", "empty-version", "non-string-path", "extra"],
)
def test_capture_rejects_missing_duplicate_empty_or_non_string_package_data(
    malformation: str,
) -> None:
    rows = _runtime_rows()
    if malformation == "missing":
        rows.pop()
    elif malformation == "duplicate":
        rows.append(dict(rows[0]))
    elif malformation == "empty-version":
        rows[0]["version"] = ""
    elif malformation == "non-string-path":
        rows[0]["module_path"] = None
    else:
        rows.append(
            {"name": "transformers", "version": "1", "module_path": "/extra"}
        )

    def runner(argv: list[str], **_kwargs: object) -> CompletedProcess[str]:
        return CompletedProcess(
            args=argv,
            returncode=0,
            stdout=_runtime_stdout(rows),
            stderr="",
        )

    with pytest.raises(ProtectedRuntimeError, match="protected runtime"):
        capture_protected_runtime(Path("/runtime/python"), runner=runner)


@pytest.mark.parametrize("stdout", ["not-json", "[]", '{"packages":null}'])
def test_capture_rejects_malformed_json_payload(stdout: str) -> None:
    def runner(argv: list[str], **_kwargs: object) -> CompletedProcess[str]:
        return CompletedProcess(args=argv, returncode=0, stdout=stdout, stderr="")

    with pytest.raises(ProtectedRuntimeError, match="protected runtime"):
        capture_protected_runtime(Path("/runtime/python"), runner=runner)


@pytest.mark.parametrize(
    "revision",
    ["a" * 39, "A" * 40, "g" * 40],
)
def test_snapshot_rejects_invalid_requested_revision_before_download(
    tmp_path: Path,
    revision: str,
) -> None:
    called = False

    def downloader(**_kwargs: object) -> Path:
        nonlocal called
        called = True
        return tmp_path / "download"

    model = ModelRef(
        repo_id="example/model",
        revision=revision,
        code_commit=None,
        license_id="Apache-2.0",
    )

    with pytest.raises(ValueError, match="40-character lowercase hex"):
        snapshot_pinned_model(
            model,
            tmp_path / "destination",
            downloader=downloader,
            resolve_revision=lambda _path: "a" * 40,
        )
    assert called is False


@pytest.mark.parametrize(
    "resolved_revision",
    ["b" * 39, "B" * 40, "g" * 40, "b" * 40],
)
def test_snapshot_rejects_invalid_or_mismatched_resolved_revision(
    tmp_path: Path,
    resolved_revision: str,
) -> None:
    model = ModelRef(
        repo_id="example/model",
        revision="a" * 40,
        code_commit=None,
        license_id="Apache-2.0",
    )

    with pytest.raises(ValueError, match="resolved revision"):
        snapshot_pinned_model(
            model,
            tmp_path / "destination",
            downloader=lambda **_kwargs: tmp_path / "download",
            resolve_revision=lambda _path: resolved_revision,
        )


def test_snapshot_pinned_model_passes_exact_pin_and_returns_frozen_record(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    revision = "a" * 40
    model = ModelRef(
        repo_id="example/model",
        revision=revision,
        code_commit="b" * 40,
        license_id="Apache-2.0",
    )
    downloaded = tmp_path / "downloaded"

    def downloader(**kwargs: object) -> Path:
        calls.append(kwargs)
        return downloaded

    snapshot = snapshot_pinned_model(
        model,
        tmp_path / "destination",
        downloader=downloader,
        resolve_revision=lambda path: revision if path == downloaded else "",
    )

    assert calls == [
        {
            "repo_id": "example/model",
            "revision": revision,
            "local_dir": tmp_path / "destination",
        }
    ]
    assert snapshot == PinnedModelSnapshot(
        model=model,
        local_path=downloaded,
        resolved_revision=revision,
    )
    with pytest.raises(FrozenInstanceError):
        snapshot.resolved_revision = "b" * 40  # type: ignore[misc]


def test_runtime_records_are_frozen() -> None:
    package = RuntimePackage(name="torch", version="2.6.0", module_path="/torch")

    with pytest.raises(FrozenInstanceError):
        package.version = "2.7.0"  # type: ignore[misc]


def test_non_mapping_report_is_rejected() -> None:
    report = cast(Mapping[str, object], [])

    with pytest.raises(ProtectedRuntimeError, match="report"):
        assert_no_protected_changes(report)


def test_learned_environment_defaults_are_exact() -> None:
    assert DEFAULT_ENV_ROOT == Path("/content/learned-env")
    assert DEFAULT_LOCK_PATH == Path(
        __file__
    ).parents[3] / "experiments" / "learned_quality" / "requirements-lock.txt"


@pytest.mark.parametrize("relative_field", ["main_python", "root", "lock_path"])
def test_install_rejects_relative_paths_before_subprocess(
    tmp_path: Path,
    relative_field: str,
) -> None:
    main_python = tmp_path / "main-python"
    root = tmp_path / "worker"
    lock_path = _write_lock(tmp_path)
    if relative_field == "main_python":
        main_python = Path("main-python")
    elif relative_field == "root":
        root = Path("worker")
    else:
        lock_path = Path("requirements-lock.txt")
    calls: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: object) -> CompletedProcess[str]:
        calls.append(argv)
        return CompletedProcess(argv, 0, stdout="", stderr="")

    with pytest.raises(ValueError, match="absolute"):
        install_learned_environment(
            main_python,
            root=root,
            lock_path=lock_path,
            runner=runner,
        )
    assert calls == []


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param(LOCK_LINES + ("",), id="blank"),
        pytest.param(("# comment", *LOCK_LINES), id="comment"),
        pytest.param((*LOCK_LINES, LOCK_LINES[0]), id="duplicate"),
        pytest.param((*LOCK_LINES, "extra==1.0"), id="extra"),
        pytest.param((*LOCK_LINES, "torch==2.7.0"), id="protected"),
        pytest.param((*LOCK_LINES, "xformers==0.0.30"), id="xformers"),
        pytest.param(
            (LOCK_LINES[1], LOCK_LINES[0], *LOCK_LINES[2:]),
            id="reordered",
        ),
    ],
)
def test_install_rejects_tampered_lock_before_subprocess(
    tmp_path: Path,
    lines: tuple[str, ...],
) -> None:
    lock_path = _write_lock(tmp_path, lines)
    calls: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: object) -> CompletedProcess[str]:
        calls.append(argv)
        return CompletedProcess(argv, 0, stdout="", stderr="")

    with pytest.raises(ProtectedRuntimeError, match="lock"):
        install_learned_environment(
            tmp_path / "main-python",
            root=tmp_path / "worker",
            lock_path=lock_path,
            runner=runner,
        )
    assert calls == []


def test_install_rejects_symlink_or_non_file_lock(tmp_path: Path) -> None:
    target = _write_lock(tmp_path)
    symlink = tmp_path / "lock-link.txt"
    symlink.symlink_to(target)

    with pytest.raises(ProtectedRuntimeError, match="lock"):
        install_learned_environment(
            tmp_path / "main-python",
            root=tmp_path / "worker",
            lock_path=symlink,
            runner=lambda *_args, **_kwargs: pytest.fail("runner must not execute"),
        )

    directory = tmp_path / "lock-directory"
    directory.mkdir()
    with pytest.raises(ProtectedRuntimeError, match="lock"):
        install_learned_environment(
            tmp_path / "main-python",
            root=tmp_path / "worker",
            lock_path=directory,
            runner=lambda *_args, **_kwargs: pytest.fail("runner must not execute"),
        )


def test_install_rejects_lock_beneath_environment_root_before_subprocess(
    tmp_path: Path,
) -> None:
    root = tmp_path / "worker"
    root.mkdir()
    lock_path = _write_lock(root)
    original_lock = lock_path.read_bytes()
    calls: list[list[str]] = []

    def runner(argv: list[str], **_kwargs: object) -> CompletedProcess[str]:
        calls.append(argv)
        pytest.fail("runner must not execute")

    with pytest.raises(ProtectedRuntimeError, match="lock"):
        install_learned_environment(
            tmp_path / "main-python",
            root=root,
            lock_path=lock_path,
            runner=runner,
        )

    assert calls == []
    assert lock_path.read_bytes() == original_lock


def test_install_rejects_symlink_environment_root_after_runtime_capture(
    tmp_path: Path,
) -> None:
    lock_path = _write_lock(tmp_path)
    target = tmp_path / "worker-target"
    target.mkdir()
    root = tmp_path / "worker-link"
    root.symlink_to(target, target_is_directory=True)
    runner = _InstallRunner(root)

    with pytest.raises(ProtectedRuntimeError, match="environment root"):
        install_learned_environment(
            tmp_path / "main-python",
            root=root,
            lock_path=lock_path,
            runner=runner,
        )

    assert runner.capture_count == 2
    assert all(call[0][1:3] == ["-I", "-c"] for call in runner.calls)


def test_install_success_uses_exact_safe_argv_and_returns_frozen_record(
    tmp_path: Path,
) -> None:
    main_python = tmp_path / "main-python"
    root = tmp_path / "worker"
    lock_path = _write_lock(tmp_path)
    original_lock = lock_path.read_bytes()
    outside = tmp_path / "outside-sentinel"
    outside.write_text("keep", encoding="utf-8")
    runner = _InstallRunner(root, venv_report_kind="regular")

    environment = install_learned_environment(
        main_python,
        root=root,
        lock_path=lock_path,
        runner=runner,
    )

    worker_python = root / "bin" / "python"
    report_path = root / "pip-dry-run-report.json"
    assert len(runner.calls) == 5
    assert runner.calls[0][0][0:3] == [str(main_python), "-I", "-c"]
    assert runner.calls[1][0] == [
        str(main_python),
        "-m",
        "venv",
        "--system-site-packages",
        "--clear",
        str(root),
    ]
    assert runner.calls[2][0] == [
        str(worker_python),
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--report",
        str(report_path),
        "-r",
        str(lock_path),
    ]
    assert runner.calls[3][0] == [
        str(worker_python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "-r",
        str(lock_path),
    ]
    assert runner.calls[4][0][0:3] == [str(main_python), "-I", "-c"]
    assert all(
        kwargs == {"check": True, "capture_output": True, "text": True}
        for _argv, kwargs in runner.calls
    )
    assert all(isinstance(argv, list) for argv, _kwargs in runner.calls)
    assert all("shell" not in kwargs for _argv, kwargs in runner.calls)
    assert runner.report_existed_before_dry_run is False
    assert environment == LearnedEnvironment(
        root=root,
        python_exe=worker_python,
        lock_path=lock_path,
        pip_report_path=report_path,
        main_runtime_before=environment.main_runtime_before,
        main_runtime_after=environment.main_runtime_after,
    )
    assert environment.main_runtime_before == environment.main_runtime_after
    assert report_path.is_file()
    assert lock_path.read_bytes() == original_lock
    assert outside.read_text(encoding="utf-8") == "keep"
    with pytest.raises(FrozenInstanceError):
        environment.root = tmp_path / "other"  # type: ignore[misc]


@pytest.mark.parametrize("report_kind", ["symlink", "directory"])
def test_install_rejects_unsafe_preexisting_worker_report(
    tmp_path: Path,
    report_kind: str,
) -> None:
    root = tmp_path / "worker"
    runner = _InstallRunner(root, venv_report_kind=report_kind)

    with pytest.raises(ProtectedRuntimeError, match="pip report"):
        install_learned_environment(
            tmp_path / "main-python",
            root=root,
            lock_path=_write_lock(tmp_path),
            runner=runner,
        )

    assert runner.capture_count == 2
    assert not any("--dry-run" in argv for argv, _kwargs in runner.calls)
    assert not any("--no-deps" in argv for argv, _kwargs in runner.calls)


@pytest.mark.parametrize(
    ("report_mode", "report_payload"),
    [
        pytest.param("missing", "", id="missing"),
        pytest.param("write", "not-json", id="malformed"),
        pytest.param(
            "write",
            '{"install":[],"install":[]}',
            id="duplicate-key",
        ),
        pytest.param("write", '{"install":[],"value":NaN}', id="nan"),
        pytest.param("write", '{"install":[],"value":Infinity}', id="infinity"),
    ],
)
def test_install_rejects_missing_or_non_strict_pip_report(
    tmp_path: Path,
    report_mode: str,
    report_payload: str,
) -> None:
    root = tmp_path / "worker"
    runner = _InstallRunner(
        root,
        report_mode=report_mode,
        report_payload=report_payload,
    )

    with pytest.raises(ProtectedRuntimeError, match="pip report"):
        install_learned_environment(
            tmp_path / "main-python",
            root=root,
            lock_path=_write_lock(tmp_path),
            runner=runner,
        )

    assert runner.capture_count == 2
    assert not any("--no-deps" in argv for argv, _kwargs in runner.calls)


def test_protected_dry_run_plan_stops_before_install(tmp_path: Path) -> None:
    root = tmp_path / "worker"
    runner = _InstallRunner(
        root,
        report_payload='{ "install": [{"metadata": {"name": "torch"}}] }',
    )

    with pytest.raises(ProtectedRuntimeError, match="torch"):
        install_learned_environment(
            tmp_path / "main-python",
            root=root,
            lock_path=_write_lock(tmp_path),
            runner=runner,
        )

    assert runner.capture_count == 2
    assert not any("--no-deps" in argv for argv, _kwargs in runner.calls)


@pytest.mark.parametrize("fail_stage", ["venv", "dry-run", "install"])
def test_subprocess_failure_still_runs_mandatory_after_snapshot(
    tmp_path: Path,
    fail_stage: str,
) -> None:
    root = tmp_path / "worker"
    runner = _InstallRunner(root, fail_stage=fail_stage)

    with pytest.raises(CalledProcessError):
        install_learned_environment(
            tmp_path / "main-python",
            root=root,
            lock_path=_write_lock(tmp_path),
            runner=runner,
        )

    assert runner.capture_count == 2
    assert runner.calls[-1][0][1:3] == ["-I", "-c"]


def test_protected_runtime_drift_takes_precedence_over_install_failure(
    tmp_path: Path,
) -> None:
    rows = _runtime_rows()
    rows[0]["version"] = "9.9.9"
    root = tmp_path / "worker"
    runner = _InstallRunner(
        root,
        fail_stage="install",
        after_stdout=_runtime_stdout(rows),
    )

    with pytest.raises(ProtectedRuntimeError, match="torch") as error:
        install_learned_environment(
            tmp_path / "main-python",
            root=root,
            lock_path=_write_lock(tmp_path),
            runner=runner,
        )

    assert isinstance(error.value.__cause__, CalledProcessError)
    assert runner.capture_count == 2


def test_learned_asset_defaults_are_exact() -> None:
    assert DEFAULT_SOURCE_ROOT == Path("/content/learned-sources")
    assert DEFAULT_CHECKPOINT_ROOT == Path("/content/learned-checkpoints")


def test_materialize_missing_assets_uses_exact_pins_and_safe_argv(
    tmp_path: Path,
) -> None:
    environment = _learned_environment(tmp_path / "venv")
    source_root = tmp_path / "sources"
    checkpoint_root = tmp_path / "checkpoints"
    runner = _GitRunner(source_root)
    download_calls: list[dict[str, object]] = []
    resolved: dict[Path, str] = {}

    def downloader(**kwargs: object) -> Path:
        download_calls.append(dict(kwargs))
        local_dir = Path(cast(Path, kwargs["local_dir"]))
        local_dir.mkdir()
        resolved[local_dir] = cast(str, kwargs["revision"])
        return local_dir

    assets = materialize_pinned_assets(
        environment,
        source_root=source_root,
        checkpoint_root=checkpoint_root,
        runner=runner,
        downloader=downloader,
        resolve_revision=resolved.__getitem__,
    )

    expected_git_calls: list[list[str]] = []
    expected_sources: list[SourceCheckout] = []
    for name, source in zip(_SOURCE_NAMES, SOURCE_REPOSITORY_REFS, strict=True):
        destination = source_root / name
        expected_git_calls.extend(
            [
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    "--filter=blob:none",
                    source.repo_id,
                    str(destination),
                ],
                [
                    "git",
                    "-C",
                    str(destination),
                    "checkout",
                    "--detach",
                    source.revision,
                ],
                [
                    "git",
                    "-C",
                    str(destination),
                    "remote",
                    "get-url",
                    "origin",
                ],
                ["git", "-C", str(destination), "rev-parse", "HEAD"],
                ["git", "-C", str(destination), "status", "--porcelain"],
            ]
        )
        expected_sources.append(
            SourceCheckout(
                name=name,
                repo_url=source.repo_id,
                path=destination,
                requested_commit=source.revision,
                resolved_commit=source.revision,
                license_id=source.license_id,
            )
        )

    checkpoint_destinations = tuple(
        checkpoint_root / model.repo_id.replace("/", "--")
        for model in CHECKPOINT_MODEL_REFS
    )
    assert [argv for argv, _kwargs in runner.calls] == expected_git_calls
    assert all(
        kwargs == {"check": True, "capture_output": True, "text": True}
        for _argv, kwargs in runner.calls
    )
    assert all("shell" not in kwargs for _argv, kwargs in runner.calls)
    assert all(
        forbidden not in " ".join(argv).casefold()
        for argv, _kwargs in runner.calls
        for forbidden in ("pip install", "setup.py", "pyproject.toml")
    )
    assert download_calls == [
        {
            "repo_id": model.repo_id,
            "revision": model.revision,
            "local_dir": destination,
        }
        for model, destination in zip(
            CHECKPOINT_MODEL_REFS,
            checkpoint_destinations,
            strict=True,
        )
    ]
    assert assets == LearnedAssets(
        sources=tuple(expected_sources),
        checkpoints=tuple(
            PinnedModelSnapshot(
                model=model,
                local_path=destination,
                resolved_revision=model.revision,
            )
            for model, destination in zip(
                CHECKPOINT_MODEL_REFS,
                checkpoint_destinations,
                strict=True,
            )
        ),
    )
    with pytest.raises(FrozenInstanceError):
        assets.sources = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        assets.sources[0].resolved_commit = "a" * 40  # type: ignore[misc]


def test_materialize_reuses_only_clean_exact_source_directories(
    tmp_path: Path,
) -> None:
    environment = _learned_environment(tmp_path / "venv")
    source_root = tmp_path / "sources"
    checkpoint_root = tmp_path / "checkpoints"
    source_root.mkdir()
    checkpoint_root.mkdir()
    for name in _SOURCE_NAMES:
        (source_root / name).mkdir()
    for model in CHECKPOINT_MODEL_REFS:
        (checkpoint_root / model.repo_id.replace("/", "--")).mkdir()
    runner = _GitRunner(source_root)

    assets = materialize_pinned_assets(
        environment,
        source_root=source_root,
        checkpoint_root=checkpoint_root,
        runner=runner,
        downloader=lambda **kwargs: cast(Path, kwargs["local_dir"]),
        resolve_revision=lambda path: next(
            model.revision
            for model in CHECKPOINT_MODEL_REFS
            if path.name == model.repo_id.replace("/", "--")
        ),
    )

    assert len(assets.sources) == len(SOURCE_REPOSITORY_REFS)
    assert len(runner.calls) == 9
    assert all("clone" not in argv and "checkout" not in argv for argv, _ in runner.calls)
    assert [argv[3:] for argv, _kwargs in runner.calls] == [
        command
        for _name in _SOURCE_NAMES
        for command in (
            ["remote", "get-url", "origin"],
            ["rev-parse", "HEAD"],
            ["status", "--porcelain"],
        )
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("remote", "https://example.invalid/wrong.git\n", "remote"),
        ("head", f"{'b' * 40}\n", "HEAD"),
        ("status", " M setup.py\n", "dirty"),
        (
            "remote",
            f"{SOURCE_REPOSITORY_REFS[0].repo_id}\nextra\n",
            "remote",
        ),
        ("head", f"{'A' * 40}\n", "HEAD"),
        ("head", None, "HEAD"),
    ],
)
def test_materialize_rejects_untrusted_or_malformed_existing_source(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "da3").mkdir()
    runner = _GitRunner(source_root, outputs={("da3", field): value})

    with pytest.raises(ValueError, match=message):
        materialize_pinned_assets(
            _learned_environment(tmp_path / "venv"),
            source_root=source_root,
            checkpoint_root=tmp_path / "checkpoints",
            runner=runner,
            downloader=lambda **_kwargs: pytest.fail("download must not execute"),
            resolve_revision=lambda _path: pytest.fail("resolver must not execute"),
        )

    assert all("clone" not in argv for argv, _kwargs in runner.calls)


@pytest.mark.parametrize("fail_step", ["clone", "checkout"])
def test_clone_failure_leaves_destination_but_never_trusts_it_without_checks(
    tmp_path: Path,
    fail_step: str,
) -> None:
    source_root = tmp_path / "sources"
    checkpoint_root = tmp_path / "checkpoints"
    failing_runner = _GitRunner(source_root, fail_step=fail_step)

    with pytest.raises(CalledProcessError):
        materialize_pinned_assets(
            _learned_environment(tmp_path / "venv"),
            source_root=source_root,
            checkpoint_root=checkpoint_root,
            runner=failing_runner,
            downloader=lambda **_kwargs: pytest.fail("download must not execute"),
            resolve_revision=lambda _path: pytest.fail("resolver must not execute"),
        )

    destination = source_root / "da3"
    assert destination.is_dir()
    reuse_runner = _GitRunner(
        source_root,
        outputs={("da3", "head"): f"{'b' * 40}\n"},
    )
    with pytest.raises(ValueError, match="HEAD"):
        materialize_pinned_assets(
            _learned_environment(tmp_path / "venv"),
            source_root=source_root,
            checkpoint_root=checkpoint_root,
            runner=reuse_runner,
            downloader=lambda **_kwargs: pytest.fail("download must not execute"),
            resolve_revision=lambda _path: pytest.fail("resolver must not execute"),
        )
    assert all("clone" not in argv and "checkout" not in argv for argv, _ in reuse_runner.calls)


@pytest.mark.parametrize(
    "relative_field",
    ["environment.root", "environment.python_exe", "source_root", "checkpoint_root"],
)
def test_materialize_rejects_relative_paths_before_external_calls(
    tmp_path: Path,
    relative_field: str,
) -> None:
    environment = _learned_environment(tmp_path / "venv")
    source_root = tmp_path / "sources"
    checkpoint_root = tmp_path / "checkpoints"
    if relative_field == "environment.root":
        environment = replace(environment, root=Path("venv"))
    elif relative_field == "environment.python_exe":
        environment = replace(environment, python_exe=Path("venv/bin/python"))
    elif relative_field == "source_root":
        source_root = Path("sources")
    else:
        checkpoint_root = Path("checkpoints")
    calls: list[list[str]] = []

    with pytest.raises(ValueError, match="absolute"):
        materialize_pinned_assets(
            environment,
            source_root=source_root,
            checkpoint_root=checkpoint_root,
            runner=lambda argv, **_kwargs: calls.append(argv),  # type: ignore[arg-type]
            downloader=lambda **_kwargs: pytest.fail("download must not execute"),
            resolve_revision=lambda _path: pytest.fail("resolver must not execute"),
        )
    assert calls == []


@pytest.mark.parametrize(
    "case",
    [
        "source-in-env",
        "source-contains-env",
        "checkpoint-in-env",
        "checkpoint-contains-env",
        "same-asset-roots",
        "source-contains-checkpoint",
        "checkpoint-contains-source",
    ],
)
def test_materialize_rejects_overlapping_resolved_roots_before_external_calls(
    tmp_path: Path,
    case: str,
) -> None:
    environment_root = tmp_path / "sandbox" / "venv"
    source_root = tmp_path / "sources"
    checkpoint_root = tmp_path / "checkpoints"
    if case == "source-in-env":
        source_root = environment_root / "sources"
    elif case == "source-contains-env":
        source_root = environment_root.parent
    elif case == "checkpoint-in-env":
        checkpoint_root = environment_root / "checkpoints"
    elif case == "checkpoint-contains-env":
        checkpoint_root = environment_root.parent
    elif case == "same-asset-roots":
        checkpoint_root = source_root
    elif case == "source-contains-checkpoint":
        checkpoint_root = source_root / "checkpoints"
    else:
        source_root = checkpoint_root / "sources"
    calls: list[list[str]] = []

    with pytest.raises(ValueError, match="root"):
        materialize_pinned_assets(
            _learned_environment(environment_root),
            source_root=source_root,
            checkpoint_root=checkpoint_root,
            runner=lambda argv, **_kwargs: calls.append(argv),  # type: ignore[arg-type]
            downloader=lambda **_kwargs: pytest.fail("download must not execute"),
            resolve_revision=lambda _path: pytest.fail("resolver must not execute"),
        )
    assert calls == []


@pytest.mark.parametrize("root_kind", ["source", "checkpoint"])
def test_materialize_rejects_existing_symlink_asset_root(
    tmp_path: Path,
    root_kind: str,
) -> None:
    target = tmp_path / f"{root_kind}-target"
    target.mkdir()
    symlink = tmp_path / f"{root_kind}-link"
    symlink.symlink_to(target, target_is_directory=True)
    source_root = symlink if root_kind == "source" else tmp_path / "sources"
    checkpoint_root = symlink if root_kind == "checkpoint" else tmp_path / "checkpoints"

    with pytest.raises(ValueError, match="symlink"):
        materialize_pinned_assets(
            _learned_environment(tmp_path / "venv"),
            source_root=source_root,
            checkpoint_root=checkpoint_root,
            runner=lambda *_args, **_kwargs: pytest.fail("runner must not execute"),
            downloader=lambda **_kwargs: pytest.fail("download must not execute"),
            resolve_revision=lambda _path: pytest.fail("resolver must not execute"),
        )


@pytest.mark.parametrize("destination_kind", ["file", "symlink"])
def test_materialize_rejects_unsafe_existing_source_destination(
    tmp_path: Path,
    destination_kind: str,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    destination = source_root / "da3"
    target = tmp_path / "source-target"
    if destination_kind == "file":
        destination.write_bytes(b"do-not-overwrite")
    else:
        target.mkdir()
        destination.symlink_to(target, target_is_directory=True)
    original = destination.read_bytes() if destination_kind == "file" else None

    with pytest.raises(ValueError, match="source destination"):
        materialize_pinned_assets(
            _learned_environment(tmp_path / "venv"),
            source_root=source_root,
            checkpoint_root=tmp_path / "checkpoints",
            runner=lambda *_args, **_kwargs: pytest.fail("runner must not execute"),
            downloader=lambda **_kwargs: pytest.fail("download must not execute"),
            resolve_revision=lambda _path: pytest.fail("resolver must not execute"),
        )
    if original is not None:
        assert destination.read_bytes() == original
    else:
        assert destination.is_symlink()


@pytest.mark.parametrize("destination_kind", ["file", "symlink"])
def test_materialize_rejects_unsafe_existing_checkpoint_destination(
    tmp_path: Path,
    destination_kind: str,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    destination = checkpoint_root / CHECKPOINT_MODEL_REFS[0].repo_id.replace("/", "--")
    target = tmp_path / "checkpoint-target"
    if destination_kind == "file":
        destination.write_bytes(b"do-not-overwrite")
    else:
        target.mkdir()
        destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="checkpoint destination"):
        materialize_pinned_assets(
            _learned_environment(tmp_path / "venv"),
            source_root=tmp_path / "sources",
            checkpoint_root=checkpoint_root,
            runner=lambda *_args, **_kwargs: pytest.fail("runner must not execute"),
            downloader=lambda **_kwargs: pytest.fail("download must not execute"),
            resolve_revision=lambda _path: pytest.fail("resolver must not execute"),
        )
    if destination_kind == "symlink":
        assert destination.is_symlink()
    else:
        assert destination.read_bytes() == b"do-not-overwrite"


def test_materialize_propagates_checkpoint_revision_mismatch(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    checkpoint_root = tmp_path / "checkpoints"
    runner = _GitRunner(source_root)
    download_calls: list[dict[str, object]] = []

    def downloader(**kwargs: object) -> Path:
        download_calls.append(dict(kwargs))
        local_dir = Path(cast(Path, kwargs["local_dir"]))
        local_dir.mkdir()
        return local_dir

    with pytest.raises(ValueError, match="resolved revision"):
        materialize_pinned_assets(
            _learned_environment(tmp_path / "venv"),
            source_root=source_root,
            checkpoint_root=checkpoint_root,
            runner=runner,
            downloader=downloader,
            resolve_revision=lambda _path: "b" * 40,
        )

    assert download_calls == [
        {
            "repo_id": CHECKPOINT_MODEL_REFS[0].repo_id,
            "revision": CHECKPOINT_MODEL_REFS[0].revision,
            "local_dir": checkpoint_root
            / CHECKPOINT_MODEL_REFS[0].repo_id.replace("/", "--"),
        }
    ]


_LEARNED_IMPORT_MODULES = (
    "transformers",
    "huggingface_hub",
    "tokenizers",
    "safetensors",
    "pycolmap",
    "omegaconf",
    "hydra",
    "iopath",
    "portalocker",
    "addict",
    "moviepy.editor",
    "trimesh",
    "evo",
    "depth_anything_3",
    "sam2",
    "raft",
)


def _verified_assets(tmp_path: Path) -> tuple[LearnedEnvironment, LearnedAssets]:
    environment = _learned_environment(tmp_path / "venv")
    environment.root.mkdir()
    environment.python_exe.parent.mkdir()
    environment.python_exe.write_text("worker", encoding="utf-8")
    environment.lock_path.write_text("lock", encoding="utf-8")
    source_root = tmp_path / "sources"
    checkpoint_root = tmp_path / "checkpoints"
    source_root.mkdir()
    checkpoint_root.mkdir()

    sources = []
    for name, source in zip(_SOURCE_NAMES, SOURCE_REPOSITORY_REFS, strict=True):
        path = source_root / name
        path.mkdir()
        sources.append(
            SourceCheckout(
                name=name,
                repo_url=source.repo_id,
                path=path,
                requested_commit=source.revision,
                resolved_commit=source.revision,
                license_id=source.license_id,
            )
        )

    checkpoints = []
    for model in CHECKPOINT_MODEL_REFS:
        path = checkpoint_root / model.repo_id.replace("/", "--")
        path.mkdir()
        checkpoints.append(
            PinnedModelSnapshot(
                model=model,
                local_path=path,
                resolved_revision=model.revision,
            )
        )
    return environment, LearnedAssets(tuple(sources), tuple(checkpoints))


class _VerifyRunner:
    def __init__(
        self,
        environment: LearnedEnvironment,
        *,
        worker_stdout: object | None = None,
        freeze_stdout: object | None = None,
        host_stdout: str | None = None,
        git_outputs: Mapping[tuple[str, str], object] | None = None,
        fail_worker: bool = False,
    ) -> None:
        self.environment = environment
        self.worker_stdout = worker_stdout
        self.freeze_stdout = freeze_stdout
        self.host_stdout = host_stdout or _runtime_stdout()
        self.git_outputs = dict(git_outputs or {})
        self.fail_worker = fail_worker
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(
        self,
        argv: list[str],
        **kwargs: object,
    ) -> CompletedProcess[str]:
        self.calls.append((list(argv), dict(kwargs)))
        if argv[:2] == ["git", "-C"]:
            source = _SOURCE_BY_NAME[Path(argv[2]).name]
            command = argv[3:]
            if command == ["remote", "get-url", "origin"]:
                stdout = self.git_outputs.get(
                    (Path(argv[2]).name, "remote"),
                    f"{source.repo_id}\n",
                )
            elif command == ["rev-parse", "HEAD"]:
                stdout = self.git_outputs.get(
                    (Path(argv[2]).name, "head"),
                    f"{source.revision}\n",
                )
            elif command == ["status", "--porcelain"]:
                stdout = self.git_outputs.get((Path(argv[2]).name, "status"), "")
            else:
                raise AssertionError(f"unexpected git argv: {argv!r}")
            return CompletedProcess(argv, 0, stdout=cast(str, stdout), stderr="")
        if argv[:4] == [
            str(self.environment.python_exe),
            "-m",
            "pip",
            "freeze",
        ]:
            return CompletedProcess(
                argv,
                0,
                stdout=cast(
                    str,
                    self.freeze_stdout
                    if self.freeze_stdout is not None
                    else "transformers==4.57.6\nAddict==2.4.0\n",
                ),
                stderr="",
            )
        if argv[0] == str(self.environment.python_exe) and "-c" in argv:
            if self.fail_worker:
                raise CalledProcessError(1, argv, stderr="worker failed")
            return CompletedProcess(
                argv,
                0,
                stdout=cast(
                    str,
                    self.worker_stdout
                    if self.worker_stdout is not None
                    else json.dumps(
                        {"modules": list(_LEARNED_IMPORT_MODULES), "ok": True},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n",
                ),
                stderr="",
            )
        if argv[0] == str(self.environment.main_runtime_before.python_exe):
            return CompletedProcess(argv, 0, stdout=self.host_stdout, stderr="")
        raise AssertionError(f"unexpected subprocess argv: {argv!r}")


def test_verify_writes_deterministic_manifest_with_safe_exact_worker_argv(
    tmp_path: Path,
) -> None:
    environment, assets = _verified_assets(tmp_path)
    runner = _VerifyRunner(environment)

    first = verify_learned_environment(environment, assets, runner=runner)
    first_bytes = first.path.read_bytes()
    second = verify_learned_environment(environment, assets, runner=runner)

    assert isinstance(first, ModelManifest)
    assert first.path == environment.root / "model_manifest.json"
    assert first.sha256 == sha256(first_bytes).hexdigest()
    assert second.path.read_bytes() == first_bytes
    assert second.sha256 == first.sha256
    assert first.sources == assets.sources
    assert first.checkpoints == assets.checkpoints
    assert first.worker_pip_freeze == ("Addict==2.4.0", "transformers==4.57.6")
    assert first.main_runtime == environment.main_runtime_before

    payload = json.loads(first_bytes)
    assert first_bytes.endswith(b"\n")
    assert payload["schema_version"] == 1
    assert payload["worker_python"] == str(environment.python_exe)
    assert payload["lock_path"] == str(environment.lock_path)
    assert payload["asset_roots"] == {
        "checkpoints": str(assets.checkpoints[0].local_path.parent),
        "sources": str(assets.sources[0].path.parent),
    }
    assert payload["worker_pip_freeze"] == [
        "Addict==2.4.0",
        "transformers==4.57.6",
    ]

    worker_calls = [
        argv
        for argv, _kwargs in runner.calls
        if argv[0] == str(environment.python_exe) and "-c" in argv
    ]
    assert len(worker_calls) == 2
    assert worker_calls[0][1:3] == ["-I", "-c"]
    assert len(worker_calls[0]) == 5
    assert worker_calls[0][3] == worker_calls[1][3]
    assert all(module in worker_calls[0][3] for module in _LEARNED_IMPORT_MODULES)
    assert all(str(source.path) not in worker_calls[0][3] for source in assets.sources)
    assert json.loads(worker_calls[0][4]) == {
        source.name: str(source.path) for source in assets.sources
    }
    assert all(
        kwargs == {"check": True, "capture_output": True, "text": True}
        and "shell" not in kwargs
        for _argv, kwargs in runner.calls
    )
    assert all(
        forbidden not in " ".join(argv).casefold()
        for argv, _kwargs in runner.calls
        for forbidden in ("clone", "checkout", "install", "download", "setup")
    )
    assert list(environment.root.glob(".model_manifest.*.tmp")) == []
    with pytest.raises(FrozenInstanceError):
        first.sha256 = "0" * 64  # type: ignore[misc]


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "not-json",
        "[]",
        '{"modules":[],"ok":true,"ok":true}',
        '{"modules":[],"ok":NaN}',
        json.dumps({"modules": list(_LEARNED_IMPORT_MODULES), "ok": False}),
        json.dumps({"modules": [*_LEARNED_IMPORT_MODULES, "extra"], "ok": True}),
    ],
)
def test_verify_rejects_malformed_or_unsuccessful_worker_output_after_host_check(
    tmp_path: Path,
    stdout: str,
) -> None:
    environment, assets = _verified_assets(tmp_path)
    runner = _VerifyRunner(environment, worker_stdout=stdout)

    with pytest.raises(ProtectedRuntimeError, match="worker verification"):
        verify_learned_environment(environment, assets, runner=runner)

    assert any(
        argv[0] == str(environment.main_runtime_before.python_exe)
        for argv, _kwargs in runner.calls
    )
    assert not (environment.root / "model_manifest.json").exists()


def test_verify_propagates_failed_worker_only_after_host_check(tmp_path: Path) -> None:
    environment, assets = _verified_assets(tmp_path)
    runner = _VerifyRunner(environment, fail_worker=True)

    with pytest.raises(CalledProcessError) as error:
        verify_learned_environment(environment, assets, runner=runner)

    assert error.value.stderr == "worker failed"
    assert runner.calls[-1][0][0] == str(
        environment.main_runtime_before.python_exe
    )


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "\n",
        "transformers==4.57.6\n\nAddict==2.4.0\n",
        "transformers==4.57.6\r\nAddict==2.4.0\r\n",
        "transformers==4.57.6\nAddict\t==2.4.0\n",
        "not a freeze entry\n",
        "Foo_Bar==1\nfoo-bar==2\n",
        "Foo.Bar==1\nfoo-bar==2\n",
    ],
)
def test_verify_rejects_malformed_or_duplicate_freeze_entries(
    tmp_path: Path,
    stdout: str,
) -> None:
    environment, assets = _verified_assets(tmp_path)
    runner = _VerifyRunner(environment, freeze_stdout=stdout)

    with pytest.raises(ProtectedRuntimeError, match="pip freeze"):
        verify_learned_environment(environment, assets, runner=runner)

    assert not (environment.root / "model_manifest.json").exists()


def test_host_drift_has_precedence_over_worker_failure(tmp_path: Path) -> None:
    environment, assets = _verified_assets(tmp_path)
    rows = _runtime_rows()
    rows[0] = {**rows[0], "version": "9.9.9"}
    runner = _VerifyRunner(
        environment,
        worker_stdout="not-json",
        host_stdout=_runtime_stdout(rows),
    )

    with pytest.raises(ProtectedRuntimeError, match="torch") as error:
        verify_learned_environment(environment, assets, runner=runner)

    assert isinstance(error.value.__cause__, ProtectedRuntimeError)
    assert "worker verification" in str(error.value.__cause__)
    assert not (environment.root / "model_manifest.json").exists()


def test_verify_requires_before_and_after_snapshots_from_same_main_python(
    tmp_path: Path,
) -> None:
    environment, assets = _verified_assets(tmp_path)
    environment = replace(
        environment,
        main_runtime_after=replace(
            environment.main_runtime_after,
            python_exe=Path("C:/different/python"),
        ),
    )
    calls: list[list[str]] = []

    with pytest.raises(ProtectedRuntimeError, match="main Python"):
        verify_learned_environment(
            environment,
            assets,
            runner=lambda argv, **_kwargs: calls.append(argv),  # type: ignore[arg-type]
        )

    assert calls == []


@pytest.mark.parametrize(
    ("git_field", "stdout", "message"),
    [
        ("remote", "https://example.invalid/tampered.git\n", "remote"),
        ("head", f"{'b' * 40}\n", "HEAD"),
        ("status", " M core/raft.py\n", "dirty"),
    ],
)
def test_verify_freshly_rejects_source_git_tampering_before_worker_imports(
    tmp_path: Path,
    git_field: str,
    stdout: str,
    message: str,
) -> None:
    environment, assets = _verified_assets(tmp_path)
    runner = _VerifyRunner(
        environment,
        git_outputs={("da3", git_field): stdout},
    )

    with pytest.raises(ValueError, match=message):
        verify_learned_environment(environment, assets, runner=runner)

    assert all(argv[0] == "git" for argv, _kwargs in runner.calls)
    assert not (environment.root / "model_manifest.json").exists()


@pytest.mark.parametrize(
    "tamper",
    [
        "source-missing",
        "source-extra",
        "source-duplicate",
        "source-order",
        "source-url",
        "source-requested",
        "source-resolved",
        "source-license",
        "source-path-escape",
        "checkpoint-missing",
        "checkpoint-extra",
        "checkpoint-duplicate",
        "checkpoint-order",
        "checkpoint-repo",
        "checkpoint-requested",
        "checkpoint-resolved",
        "checkpoint-code",
        "checkpoint-license",
        "checkpoint-path-escape",
        "cross-kind-overlap",
    ],
)
def test_verify_rejects_tampered_asset_records_before_subprocess(
    tmp_path: Path,
    tamper: str,
) -> None:
    environment, assets = _verified_assets(tmp_path)
    sources = list(assets.sources)
    checkpoints = list(assets.checkpoints)
    escape = tmp_path / "escape"
    escape.mkdir()

    if tamper == "source-missing":
        sources.pop()
    elif tamper == "source-extra":
        sources.append(sources[-1])
    elif tamper == "source-duplicate":
        sources[1] = sources[0]
    elif tamper == "source-order":
        sources[0], sources[1] = sources[1], sources[0]
    elif tamper == "source-url":
        sources[0] = replace(sources[0], repo_url="https://example.invalid/x.git")
    elif tamper == "source-requested":
        sources[0] = replace(sources[0], requested_commit="b" * 40)
    elif tamper == "source-resolved":
        sources[0] = replace(sources[0], resolved_commit="b" * 40)
    elif tamper == "source-license":
        sources[0] = replace(sources[0], license_id="MIT")
    elif tamper == "source-path-escape":
        escaped_path = escape / sources[1].name
        escaped_path.mkdir()
        sources[1] = replace(sources[1], path=escaped_path)
    elif tamper == "checkpoint-missing":
        checkpoints.pop()
    elif tamper == "checkpoint-extra":
        checkpoints.append(checkpoints[-1])
    elif tamper == "checkpoint-duplicate":
        checkpoints[1] = checkpoints[0]
    elif tamper == "checkpoint-order":
        checkpoints[0], checkpoints[1] = checkpoints[1], checkpoints[0]
    elif tamper == "checkpoint-repo":
        checkpoints[0] = replace(
            checkpoints[0],
            model=replace(checkpoints[0].model, repo_id="other/model"),
        )
    elif tamper == "checkpoint-requested":
        checkpoints[0] = replace(
            checkpoints[0],
            model=replace(checkpoints[0].model, revision="b" * 40),
        )
    elif tamper == "checkpoint-resolved":
        checkpoints[0] = replace(checkpoints[0], resolved_revision="b" * 40)
    elif tamper == "checkpoint-code":
        checkpoints[0] = replace(
            checkpoints[0],
            model=replace(checkpoints[0].model, code_commit="b" * 40),
        )
    elif tamper == "checkpoint-license":
        checkpoints[0] = replace(
            checkpoints[0],
            model=replace(checkpoints[0].model, license_id="MIT"),
        )
    elif tamper == "checkpoint-path-escape":
        escaped_path = escape / checkpoints[1].local_path.name
        escaped_path.mkdir()
        checkpoints[1] = replace(checkpoints[1], local_path=escaped_path)
    else:
        checkpoints[0] = replace(checkpoints[0], local_path=sources[0].path)

    calls: list[list[str]] = []
    with pytest.raises(ValueError):
        verify_learned_environment(
            environment,
            LearnedAssets(tuple(sources), tuple(checkpoints)),
            runner=lambda argv, **_kwargs: calls.append(argv),  # type: ignore[arg-type]
        )
    assert calls == []
    assert not (environment.root / "model_manifest.json").exists()


@pytest.mark.parametrize("asset_kind", ["source", "checkpoint"])
def test_verify_rejects_symlink_asset_directory_before_subprocess(
    tmp_path: Path,
    asset_kind: str,
) -> None:
    environment, assets = _verified_assets(tmp_path)
    if asset_kind == "source":
        original = assets.sources[1].path
        target = tmp_path / "source-target"
        target.mkdir()
        original.rmdir()
        original.symlink_to(target, target_is_directory=True)
    else:
        original = assets.checkpoints[1].local_path
        target = tmp_path / "checkpoint-target"
        target.mkdir()
        original.rmdir()
        original.symlink_to(target, target_is_directory=True)
    calls: list[list[str]] = []

    with pytest.raises(ValueError, match="symlink"):
        verify_learned_environment(
            environment,
            assets,
            runner=lambda argv, **_kwargs: calls.append(argv),  # type: ignore[arg-type]
        )
    assert calls == []


@pytest.mark.parametrize("existing_kind", ["directory", "symlink"])
def test_verify_never_overwrites_unsafe_manifest_path(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    environment, assets = _verified_assets(tmp_path)
    manifest = environment.root / "model_manifest.json"
    target = tmp_path / "manifest-target.json"
    if existing_kind == "directory":
        manifest.mkdir()
    else:
        target.write_bytes(b"preserve")
        manifest.symlink_to(target)
    calls: list[list[str]] = []

    with pytest.raises(ProtectedRuntimeError, match="manifest path"):
        verify_learned_environment(
            environment,
            assets,
            runner=lambda argv, **_kwargs: calls.append(argv),  # type: ignore[arg-type]
        )

    assert calls == []
    if existing_kind == "symlink":
        assert manifest.is_symlink()
        assert target.read_bytes() == b"preserve"


def test_verify_cleans_temporary_file_when_atomic_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, assets = _verified_assets(tmp_path)
    runner = _VerifyRunner(environment)
    manifest = environment.root / "model_manifest.json"

    def fail_replace(_self: Path, _target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        verify_learned_environment(environment, assets, runner=runner)

    assert not manifest.exists()
    assert list(environment.root.glob(".model_manifest.*.tmp")) == []


def test_verify_cleans_temporary_file_when_manifest_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, assets = _verified_assets(tmp_path)
    runner = _VerifyRunner(environment)
    real_fdopen = os.fdopen

    class _FailingStream:
        def __init__(self, descriptor: int) -> None:
            self.descriptor = descriptor

        def __enter__(self) -> _FailingStream:
            return self

        def __exit__(self, *_args: object) -> None:
            os.close(self.descriptor)

        def write(self, _data: bytes) -> int:
            raise OSError("write failed")

    def fail_fdopen(descriptor: int, mode: str) -> object:
        assert mode == "wb"
        return _FailingStream(descriptor)

    monkeypatch.setattr(os, "fdopen", fail_fdopen)
    try:
        with pytest.raises(OSError, match="write failed"):
            verify_learned_environment(environment, assets, runner=runner)
    finally:
        monkeypatch.setattr(os, "fdopen", real_fdopen)

    assert not (environment.root / "model_manifest.json").exists()
    assert list(environment.root.glob(".model_manifest.*.tmp")) == []

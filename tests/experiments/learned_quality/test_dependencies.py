from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from subprocess import CompletedProcess
from typing import Mapping, cast

import pytest

from experiments.learned_quality.contracts import ModelRef
from experiments.learned_quality.dependencies import (
    CHECKPOINT_MODEL_REFS,
    PROTECTED_PACKAGES,
    SOURCE_REPOSITORY_REFS,
    PinnedModelSnapshot,
    ProtectedRuntimeError,
    ProtectedRuntimeSnapshot,
    RuntimePackage,
    assert_no_protected_changes,
    capture_protected_runtime,
    snapshot_pinned_model,
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
        python_exe=Path("/runtime/python"),
        packages={
            name: RuntimePackage(
                name=name,
                version=versions.get(name, f"1.0.{index}"),
                module_path=paths.get(name, f"/runtime/{name}/__init__.py"),
            )
            for index, name in enumerate(PROTECTED_PACKAGES)
        },
    )


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

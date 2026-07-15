from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unicodedata
from hashlib import sha256
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from subprocess import CompletedProcess

from .contracts import ModelRef


PROTECTED_PACKAGES = ("torch", "torchvision", "numpy", "gsplat")
DEFAULT_ENV_ROOT = Path("/content/learned-env")
DEFAULT_LOCK_PATH = Path(__file__).with_name("requirements-lock.txt")
DEFAULT_SOURCE_ROOT = Path("/content/learned-sources")
DEFAULT_CHECKPOINT_ROOT = Path("/content/learned-checkpoints")

_EXPECTED_LOCK_LINES = (
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

_DA3_COMMIT = "3fe327a6abe2e5db95b54444ea95463dbfef5610"
_SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
_SEA_RAFT_COMMIT = "9137517ba24e628442aec097d3afe71d03503b75"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


SOURCE_REPOSITORY_REFS = (
    ModelRef(
        repo_id="https://github.com/ByteDance-Seed/Depth-Anything-3.git",
        revision=_DA3_COMMIT,
        code_commit=None,
        license_id="Apache-2.0",
    ),
    ModelRef(
        repo_id="https://github.com/facebookresearch/sam2.git",
        revision=_SAM2_COMMIT,
        code_commit=None,
        license_id="Apache-2.0",
    ),
    ModelRef(
        repo_id="https://github.com/MemorySlices/SEA-RAFT.git",
        revision=_SEA_RAFT_COMMIT,
        code_commit=None,
        license_id="BSD-3-Clause",
    ),
)


CHECKPOINT_MODEL_REFS = (
    ModelRef(
        repo_id="depth-anything/DA3-BASE",
        revision="f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
        code_commit=_DA3_COMMIT,
        license_id="Apache-2.0",
    ),
    ModelRef(
        repo_id="depth-anything/DA3METRIC-LARGE",
        revision="4010e39f3634a45bc60553321fb49fb760bd594e",
        code_commit=_DA3_COMMIT,
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
        code_commit=_SAM2_COMMIT,
        license_id="Apache-2.0",
    ),
    ModelRef(
        repo_id="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
        revision="eb97ef34ba5d856c3fa2cdcd073150c057ac8b69",
        code_commit=_SEA_RAFT_COMMIT,
        license_id="BSD-3-Clause",
    ),
)


_CAPTURE_SCRIPT = """\
import importlib
import importlib.metadata
import json
from pathlib import Path

rows = []
for name in ("torch", "torchvision", "numpy", "gsplat"):
    module = importlib.import_module(name)
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        locations = tuple(getattr(module, "__path__", ()))
        if len(locations) != 1:
            raise RuntimeError(f"cannot resolve module path for {name}")
        module_path = locations[0]
    rows.append(
        {
            "name": name,
            "version": importlib.metadata.version(name),
            "module_path": str(Path(module_path).resolve()),
        }
    )
print(json.dumps({"packages": rows}, allow_nan=False, separators=(",", ":")))
"""

LEARNED_IMPORT_MODULES = (
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

_VERIFY_SCRIPT = """\
import importlib
import json
import sys
from pathlib import Path

def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value

def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")

paths = json.loads(
    sys.argv[1],
    object_pairs_hook=unique_object,
    parse_constant=reject_constant,
)
if set(paths) != {"da3", "sam2", "sea-raft"}:
    raise ValueError("source path payload is malformed")
if any(not isinstance(value, str) or not value for value in paths.values()):
    raise ValueError("source path payload is malformed")
da3 = Path(paths["da3"])
sea_raft = Path(paths["sea-raft"])
sys.path[:0] = [
    str(da3 / "src"),
    paths["sam2"],
    str(sea_raft),
    str(sea_raft / "core"),
]
modules = (
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
for module in modules:
    importlib.import_module(module)
print(
    json.dumps(
        {
            "modules": modules,
            "ok": True,
            "prefix": str(Path(sys.prefix).resolve()),
        },
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
)
"""


@dataclass(frozen=True)
class RuntimePackage:
    name: str
    version: str
    module_path: str


@dataclass(frozen=True)
class ProtectedRuntimeSnapshot:
    python_exe: Path
    packages: Mapping[str, RuntimePackage]


@dataclass(frozen=True)
class LearnedEnvironment:
    root: Path
    python_exe: Path
    lock_path: Path
    pip_report_path: Path
    main_runtime_before: ProtectedRuntimeSnapshot
    main_runtime_after: ProtectedRuntimeSnapshot


class ProtectedRuntimeError(RuntimeError):
    """A dependency operation would mutate the protected host runtime."""


def _canonical_package_name(name: str) -> str:
    return (
        name.strip()
        .casefold()
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
    )


def _is_absolute_module_path(value: str) -> bool:
    return (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    )


_PROTECTED_BY_CANONICAL = {
    _canonical_package_name(name): name for name in PROTECTED_PACKAGES
}


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse_runtime_payload(stdout: str) -> Mapping[str, RuntimePackage]:
    try:
        payload = json.loads(
            stdout,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtectedRuntimeError(
            "protected runtime snapshot is not strict JSON"
        ) from exc
    if not isinstance(payload, Mapping) or set(payload) != {"packages"}:
        raise ProtectedRuntimeError("protected runtime snapshot payload is malformed")
    rows = payload["packages"]
    if not isinstance(rows, list):
        raise ProtectedRuntimeError("protected runtime package data is malformed")

    captured: dict[str, RuntimePackage] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "version",
            "module_path",
        }:
            raise ProtectedRuntimeError("protected runtime package data is malformed")
        name = row["name"]
        version = row["version"]
        module_path = row["module_path"]
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(version, str)
            or not version.strip()
            or not isinstance(module_path, str)
            or not module_path.strip()
        ):
            raise ProtectedRuntimeError("protected runtime package data is malformed")
        if not _is_absolute_module_path(module_path):
            raise ProtectedRuntimeError(
                f"protected runtime package {name} module path must be absolute"
            )
        canonical = _canonical_package_name(name)
        protected_name = _PROTECTED_BY_CANONICAL.get(canonical)
        if protected_name is None or protected_name in captured:
            raise ProtectedRuntimeError("protected runtime package data is malformed")
        captured[protected_name] = RuntimePackage(
            name=protected_name,
            version=version,
            module_path=module_path,
        )

    missing = [name for name in PROTECTED_PACKAGES if name not in captured]
    if missing:
        raise ProtectedRuntimeError(
            f"protected runtime snapshot is missing package {missing[0]}"
        )
    return {name: captured[name] for name in PROTECTED_PACKAGES}


def capture_protected_runtime(
    python_exe: Path,
    *,
    runner: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> ProtectedRuntimeSnapshot:
    python_path = Path(python_exe)
    completed = runner(
        [str(python_path), "-I", "-c", _CAPTURE_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
    )
    return ProtectedRuntimeSnapshot(
        python_exe=python_path,
        packages=_parse_runtime_payload(completed.stdout),
    )


def _plan_entry_name(entry: object, collection: str) -> str:
    if not isinstance(entry, Mapping):
        raise ProtectedRuntimeError(f"pip report {collection} entry is malformed")

    metadata = entry.get("metadata")
    if collection == "install":
        if not isinstance(metadata, Mapping):
            raise ProtectedRuntimeError(
                f"pip report {collection} entry metadata is malformed"
            )
        name = metadata.get("name")
    elif metadata is not None:
        if not isinstance(metadata, Mapping):
            raise ProtectedRuntimeError(
                f"pip report {collection} entry metadata is malformed"
            )
        name = metadata.get("name")
    else:
        name = entry.get("name")

    if not isinstance(name, str) or not name.strip():
        raise ProtectedRuntimeError(f"pip report {collection} entry name is malformed")
    return name


def _validated_snapshot_packages(
    snapshot: ProtectedRuntimeSnapshot,
    label: str,
) -> Mapping[str, RuntimePackage]:
    if not isinstance(snapshot, ProtectedRuntimeSnapshot) or not isinstance(
        snapshot.packages, Mapping
    ):
        raise ProtectedRuntimeError(f"{label} protected runtime snapshot is malformed")

    captured: dict[str, RuntimePackage] = {}
    for key, package in snapshot.packages.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(package, RuntimePackage)
        ):
            raise ProtectedRuntimeError(
                f"{label} protected runtime snapshot is malformed"
            )
        canonical = _canonical_package_name(key)
        protected_name = _PROTECTED_BY_CANONICAL.get(canonical)
        if (
            protected_name is None
            or protected_name in captured
            or _canonical_package_name(package.name) != canonical
            or not isinstance(package.version, str)
            or not package.version.strip()
            or not isinstance(package.module_path, str)
            or not package.module_path.strip()
        ):
            named = protected_name or key
            raise ProtectedRuntimeError(
                f"{label} protected runtime package {named} is malformed"
            )
        if not _is_absolute_module_path(package.module_path):
            raise ProtectedRuntimeError(
                f"{label} protected runtime package {protected_name} "
                "module path must be absolute"
            )
        captured[protected_name] = package

    missing = [name for name in PROTECTED_PACKAGES if name not in captured]
    if missing:
        raise ProtectedRuntimeError(
            f"{label} protected runtime snapshot is missing package {missing[0]}"
        )
    return captured


def assert_no_protected_changes(
    report: Mapping[str, object],
    *,
    before: ProtectedRuntimeSnapshot | None = None,
    after: ProtectedRuntimeSnapshot | None = None,
) -> None:
    if not isinstance(report, Mapping):
        raise ProtectedRuntimeError("pip report must be a mapping")

    for collection in ("install", "remove", "uninstall"):
        entries = report.get(collection, [])
        if not isinstance(entries, list):
            raise ProtectedRuntimeError(
                f"pip report {collection} collection is malformed"
            )
        for entry in entries:
            supplied_name = _plan_entry_name(entry, collection)
            protected_name = _PROTECTED_BY_CANONICAL.get(
                _canonical_package_name(supplied_name)
            )
            if protected_name is not None:
                raise ProtectedRuntimeError(
                    f"pip {collection} plan changes protected package {protected_name}"
                )

    if (before is None) != (after is None):
        raise ProtectedRuntimeError(
            "before and after protected runtime snapshots must be provided together"
        )
    if before is None or after is None:
        return

    before_packages = _validated_snapshot_packages(before, "before")
    after_packages = _validated_snapshot_packages(after, "after")
    for protected_name in PROTECTED_PACKAGES:
        previous = before_packages[protected_name]
        current = after_packages[protected_name]
        if previous.version != current.version:
            raise ProtectedRuntimeError(
                f"protected package {protected_name} version changed"
            )
        if previous.module_path != current.module_path:
            raise ProtectedRuntimeError(
                f"protected package {protected_name} module path changed"
            )


def _require_absolute_path(value: Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return path


def _validate_lock(lock_path: Path) -> None:
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ProtectedRuntimeError(
            "learned dependency lock must be a regular non-symlink file"
        )
    try:
        text = lock_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProtectedRuntimeError("learned dependency lock is unreadable") from exc
    if tuple(text.splitlines()) != _EXPECTED_LOCK_LINES:
        raise ProtectedRuntimeError(
            "learned dependency lock does not match the approved exact lock"
        )


def _assert_lock_outside_environment_root(root: Path, lock_path: Path) -> None:
    resolved_root = root.resolve(strict=False)
    resolved_lock = lock_path.resolve(strict=False)
    if resolved_lock == resolved_root or resolved_root in resolved_lock.parents:
        raise ProtectedRuntimeError(
            "learned dependency lock cannot be inside the environment root"
        )


def _run_checked(
    runner: Callable[..., CompletedProcess[str]],
    argv: list[str],
) -> CompletedProcess[str]:
    return runner(
        argv,
        check=True,
        capture_output=True,
        text=True,
    )


def _prepare_report_path(report_path: Path) -> None:
    if report_path.is_symlink():
        raise ProtectedRuntimeError("pip report path cannot be a symlink")
    if report_path.exists():
        if not report_path.is_file():
            raise ProtectedRuntimeError("pip report path must be a regular file")
        report_path.unlink()


def _load_strict_pip_report(report_path: Path) -> Mapping[str, object]:
    if report_path.is_symlink() or not report_path.is_file():
        raise ProtectedRuntimeError(
            "pip report was not created as a regular non-symlink file"
        )
    try:
        text = report_path.read_text(encoding="utf-8")
        report = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        raise ProtectedRuntimeError("pip report is not strict JSON") from exc
    if not isinstance(report, Mapping):
        raise ProtectedRuntimeError("pip report root must be a mapping")
    return report


def _capture_and_compare_runtime(
    main_python: Path,
    before: ProtectedRuntimeSnapshot,
    runner: Callable[..., CompletedProcess[str]],
) -> ProtectedRuntimeSnapshot:
    try:
        after = capture_protected_runtime(main_python, runner=runner)
        assert_no_protected_changes({}, before=before, after=after)
    except ProtectedRuntimeError:
        raise
    except Exception as exc:
        raise ProtectedRuntimeError(
            "protected runtime after-snapshot comparison failed"
        ) from exc
    return after


def install_learned_environment(
    main_python: Path,
    *,
    root: Path = DEFAULT_ENV_ROOT,
    lock_path: Path = DEFAULT_LOCK_PATH,
    runner: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> LearnedEnvironment:
    main_python_path = _require_absolute_path(main_python, "main_python")
    root_path = _require_absolute_path(root, "root")
    approved_lock_path = _require_absolute_path(lock_path, "lock_path")
    _validate_lock(approved_lock_path)
    _assert_lock_outside_environment_root(root_path, approved_lock_path)

    before = capture_protected_runtime(main_python_path, runner=runner)
    worker_python = root_path / "bin" / "python"
    report_path = root_path / "pip-dry-run-report.json"

    try:
        if root_path.is_symlink():
            raise ProtectedRuntimeError("learned environment root cannot be a symlink")
        _run_checked(
            runner,
            [
                str(main_python_path),
                "-m",
                "venv",
                "--system-site-packages",
                "--clear",
                str(root_path),
            ],
        )
        _prepare_report_path(report_path)
        _run_checked(
            runner,
            [
                str(worker_python),
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--report",
                str(report_path),
                "-r",
                str(approved_lock_path),
            ],
        )
        report = _load_strict_pip_report(report_path)
        assert_no_protected_changes(report)
        _run_checked(
            runner,
            [
                str(worker_python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "-r",
                str(approved_lock_path),
            ],
        )
    except Exception as primary_error:
        try:
            _capture_and_compare_runtime(main_python_path, before, runner)
        except ProtectedRuntimeError as drift_error:
            raise drift_error from primary_error
        raise

    after = _capture_and_compare_runtime(main_python_path, before, runner)
    return LearnedEnvironment(
        root=root_path,
        python_exe=worker_python,
        lock_path=approved_lock_path,
        pip_report_path=report_path,
        main_runtime_before=before,
        main_runtime_after=after,
    )


@dataclass(frozen=True)
class PinnedModelSnapshot:
    model: ModelRef
    local_path: Path
    resolved_revision: str


@dataclass(frozen=True)
class SourceCheckout:
    name: str
    repo_url: str
    path: Path
    requested_commit: str
    resolved_commit: str
    license_id: str


@dataclass(frozen=True)
class LearnedAssets:
    sources: tuple[SourceCheckout, ...]
    checkpoints: tuple[PinnedModelSnapshot, ...]


@dataclass(frozen=True)
class ModelManifest:
    path: Path
    sha256: str
    sources: tuple[SourceCheckout, ...]
    checkpoints: tuple[PinnedModelSnapshot, ...]
    worker_pip_freeze: tuple[str, ...]
    main_runtime: ProtectedRuntimeSnapshot


def _require_revision(value: object, label: str) -> str:
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 40-character lowercase hex revision")
    return value


def snapshot_pinned_model(
    model: ModelRef,
    destination: Path,
    *,
    downloader: Callable[..., str | Path],
    resolve_revision: Callable[[Path], str],
) -> PinnedModelSnapshot:
    requested_revision = _require_revision(model.revision, "requested revision")
    destination_path = Path(destination)
    local_path = Path(
        downloader(
            repo_id=model.repo_id,
            revision=requested_revision,
            local_dir=destination_path,
        )
    )
    resolved_revision = _require_revision(
        resolve_revision(local_path),
        "resolved revision",
    )
    if resolved_revision != requested_revision:
        raise ValueError(
            f"resolved revision for {model.repo_id} does not match requested revision"
        )
    return PinnedModelSnapshot(
        model=model,
        local_path=local_path,
        resolved_revision=resolved_revision,
    )


_SOURCE_NAMES = ("da3", "sam2", "sea-raft")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_asset_root(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    if path.exists() and not path.is_dir():
        raise ValueError(f"{label} must be a directory")


def _validate_asset_destination(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    if path.exists() and not path.is_dir():
        raise ValueError(f"{label} must be a directory")


def _single_git_output(stdout: object, label: str) -> str:
    if not isinstance(stdout, str):
        raise ValueError(f"{label} output is malformed")
    lines = stdout.splitlines()
    if len(lines) != 1:
        raise ValueError(f"{label} output is malformed")
    return lines[0]


def _verify_source_checkout(
    destination: Path,
    source: ModelRef,
    runner: Callable[..., CompletedProcess[str]],
) -> str:
    remote = _single_git_output(
        _run_checked(
            runner,
            [
                "git",
                "-C",
                str(destination),
                "remote",
                "get-url",
                "origin",
            ],
        ).stdout,
        "source remote",
    )
    if remote != source.repo_id:
        raise ValueError("source remote does not match the approved repository")

    head = _single_git_output(
        _run_checked(
            runner,
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
        ).stdout,
        "source HEAD",
    )
    _require_revision(head, "source HEAD")
    if head != source.revision:
        raise ValueError("source HEAD does not match the requested commit")

    status = _run_checked(
        runner,
        ["git", "-C", str(destination), "status", "--porcelain"],
    ).stdout
    if not isinstance(status, str):
        raise ValueError("source status output is malformed")
    if status != "":
        raise ValueError("source checkout is dirty")
    return head


def _checkpoint_destination(checkpoint_root: Path, model: ModelRef) -> Path:
    return checkpoint_root / model.repo_id.replace("/", "--")


def materialize_pinned_assets(
    environment: LearnedEnvironment,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    runner: Callable[..., CompletedProcess[str]] = subprocess.run,
    downloader: Callable[..., str | Path],
    resolve_revision: Callable[[Path], str],
) -> LearnedAssets:
    environment_root = _require_absolute_path(environment.root, "environment.root")
    _require_absolute_path(environment.python_exe, "environment.python_exe")
    source_root_path = _require_absolute_path(source_root, "source_root")
    checkpoint_root_path = _require_absolute_path(
        checkpoint_root,
        "checkpoint_root",
    )

    _validate_asset_root(environment_root, "environment root")
    _validate_asset_root(source_root_path, "source root")
    _validate_asset_root(checkpoint_root_path, "checkpoint root")
    resolved_environment_root = environment_root.resolve(strict=False)
    resolved_source_root = source_root_path.resolve(strict=False)
    resolved_checkpoint_root = checkpoint_root_path.resolve(strict=False)
    if _paths_overlap(resolved_environment_root, resolved_source_root):
        raise ValueError("source root must be outside the environment root")
    if _paths_overlap(resolved_environment_root, resolved_checkpoint_root):
        raise ValueError("checkpoint root must be outside the environment root")
    if _paths_overlap(resolved_source_root, resolved_checkpoint_root):
        raise ValueError("source and checkpoint roots cannot overlap")

    source_destinations = tuple(
        source_root_path / name for name in _SOURCE_NAMES
    )
    checkpoint_destinations = tuple(
        _checkpoint_destination(checkpoint_root_path, model)
        for model in CHECKPOINT_MODEL_REFS
    )
    for destination in source_destinations:
        _validate_asset_destination(destination, "source destination")
    for destination in checkpoint_destinations:
        _validate_asset_destination(destination, "checkpoint destination")

    source_root_path.mkdir(parents=True, exist_ok=True)
    checkpoint_root_path.mkdir(parents=True, exist_ok=True)
    _validate_asset_root(source_root_path, "source root")
    _validate_asset_root(checkpoint_root_path, "checkpoint root")

    sources: list[SourceCheckout] = []
    for name, source, destination in zip(
        _SOURCE_NAMES,
        SOURCE_REPOSITORY_REFS,
        source_destinations,
        strict=True,
    ):
        destination_exists = destination.exists()
        _validate_asset_destination(destination, "source destination")
        if not destination_exists:
            _run_checked(
                runner,
                [
                    "git",
                    "clone",
                    "--no-checkout",
                    "--filter=blob:none",
                    source.repo_id,
                    str(destination),
                ],
            )
            _validate_asset_destination(destination, "source destination")
            _run_checked(
                runner,
                [
                    "git",
                    "-C",
                    str(destination),
                    "checkout",
                    "--detach",
                    source.revision,
                ],
            )
        resolved_commit = _verify_source_checkout(destination, source, runner)
        sources.append(
            SourceCheckout(
                name=name,
                repo_url=source.repo_id,
                path=destination,
                requested_commit=source.revision,
                resolved_commit=resolved_commit,
                license_id=source.license_id,
            )
        )

    checkpoints: list[PinnedModelSnapshot] = []
    for model, destination in zip(
        CHECKPOINT_MODEL_REFS,
        checkpoint_destinations,
        strict=True,
    ):
        _validate_asset_destination(destination, "checkpoint destination")
        checkpoints.append(
            snapshot_pinned_model(
                model,
                destination,
                downloader=downloader,
                resolve_revision=resolve_revision,
            )
        )

    return LearnedAssets(
        sources=tuple(sources),
        checkpoints=tuple(checkpoints),
    )


def _require_plain_directory(path: Path, label: str) -> Path:
    candidate = _require_absolute_path(path, label)
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError(f"{label} must be a non-symlink directory")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} cannot be resolved") from exc
    if resolved != candidate:
        raise ValueError(f"{label} cannot traverse a symlink")
    return candidate


def _validate_assets_for_verification(
    environment: LearnedEnvironment,
    assets: LearnedAssets,
) -> tuple[Path, Path]:
    if not isinstance(environment, LearnedEnvironment):
        raise ValueError("learned environment is malformed")
    environment_root = _require_plain_directory(
        environment.root,
        "environment root",
    )
    worker_python = _require_absolute_path(
        environment.python_exe,
        "environment python",
    )
    if worker_python != environment_root / "bin" / "python":
        raise ValueError("environment python must match the exact installer path")
    if not worker_python.is_file():
        raise ValueError("environment python must be a usable file")
    lock_path = _require_absolute_path(environment.lock_path, "environment lock")
    if lock_path.is_symlink() or not lock_path.is_file():
        raise ValueError("environment lock must be a non-symlink file")
    if not isinstance(
        environment.main_runtime_before,
        ProtectedRuntimeSnapshot,
    ) or not isinstance(environment.main_runtime_after, ProtectedRuntimeSnapshot):
        raise ProtectedRuntimeError("protected main runtime snapshots are malformed")
    before_python = _require_absolute_path(
        environment.main_runtime_before.python_exe,
        "protected main Python",
    )
    after_python = _require_absolute_path(
        environment.main_runtime_after.python_exe,
        "protected main Python",
    )
    if before_python != after_python:
        raise ProtectedRuntimeError(
            "protected snapshots must use the same main Python"
        )
    assert_no_protected_changes(
        {},
        before=environment.main_runtime_before,
        after=environment.main_runtime_after,
    )
    if not isinstance(assets, LearnedAssets):
        raise ValueError("learned assets are malformed")
    if not isinstance(assets.sources, tuple) or not isinstance(
        assets.checkpoints,
        tuple,
    ):
        raise ValueError("learned assets must use immutable tuples")
    if len(assets.sources) != len(SOURCE_REPOSITORY_REFS):
        raise ValueError("source records must exactly match the approved set")
    if len(assets.checkpoints) != len(CHECKPOINT_MODEL_REFS):
        raise ValueError("checkpoint records must exactly match the approved set")

    source_names = [source.name for source in assets.sources]
    if len(set(source_names)) != len(source_names):
        raise ValueError("duplicate source record")
    checkpoint_ids = [checkpoint.model.repo_id for checkpoint in assets.checkpoints]
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError("duplicate checkpoint record")

    first_source = assets.sources[0]
    if not isinstance(first_source, SourceCheckout):
        raise ValueError("source record is malformed")
    source_root = _require_plain_directory(first_source.path.parent, "source root")
    source_paths: list[Path] = []
    for name, approved, source in zip(
        _SOURCE_NAMES,
        SOURCE_REPOSITORY_REFS,
        assets.sources,
        strict=True,
    ):
        if not isinstance(source, SourceCheckout):
            raise ValueError("source record is malformed")
        if (
            source.name != name
            or source.repo_url != approved.repo_id
            or source.requested_commit != approved.revision
            or source.resolved_commit != approved.revision
            or source.license_id != approved.license_id
        ):
            raise ValueError(f"source record {name} does not match the approved pin")
        _require_revision(source.requested_commit, f"source {name} requested commit")
        _require_revision(source.resolved_commit, f"source {name} resolved commit")
        expected_path = source_root / name
        if source.path != expected_path:
            raise ValueError(f"source path {name} is outside its approved root")
        source_paths.append(_require_plain_directory(source.path, f"source {name}"))

    first_checkpoint = assets.checkpoints[0]
    if not isinstance(first_checkpoint, PinnedModelSnapshot):
        raise ValueError("checkpoint record is malformed")
    checkpoint_root = _require_plain_directory(
        first_checkpoint.local_path.parent,
        "checkpoint root",
    )
    checkpoint_paths: list[Path] = []
    for approved, checkpoint in zip(
        CHECKPOINT_MODEL_REFS,
        assets.checkpoints,
        strict=True,
    ):
        if not isinstance(checkpoint, PinnedModelSnapshot):
            raise ValueError("checkpoint record is malformed")
        if checkpoint.model != approved:
            raise ValueError(
                f"checkpoint record {approved.repo_id} does not match the approved ref"
            )
        requested = _require_revision(
            checkpoint.model.revision,
            f"checkpoint {approved.repo_id} requested revision",
        )
        resolved = _require_revision(
            checkpoint.resolved_revision,
            f"checkpoint {approved.repo_id} resolved revision",
        )
        if requested != resolved or resolved != approved.revision:
            raise ValueError(
                f"checkpoint {approved.repo_id} revision does not match approved ref"
            )
        expected_path = _checkpoint_destination(checkpoint_root, approved)
        if checkpoint.local_path != expected_path:
            raise ValueError(
                f"checkpoint path {approved.repo_id} is outside its approved root"
            )
        checkpoint_paths.append(
            _require_plain_directory(
                checkpoint.local_path,
                f"checkpoint {approved.repo_id}",
            )
        )

    all_paths = source_paths + checkpoint_paths
    resolved_paths = [path.resolve(strict=True) for path in all_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("duplicate source or checkpoint path")
    for index, first in enumerate(resolved_paths):
        for second in resolved_paths[index + 1 :]:
            if _paths_overlap(first, second):
                raise ValueError("source and checkpoint paths cannot overlap")
    if _paths_overlap(environment_root.resolve(), source_root.resolve()):
        raise ValueError("source root must not overlap the environment root")
    if _paths_overlap(environment_root.resolve(), checkpoint_root.resolve()):
        raise ValueError("checkpoint root must not overlap the environment root")
    if _paths_overlap(source_root.resolve(), checkpoint_root.resolve()):
        raise ValueError("source and checkpoint roots cannot overlap")
    return source_root, checkpoint_root


def _parse_worker_verification(stdout: object, expected_prefix: Path) -> None:
    if not isinstance(stdout, str) or not stdout:
        raise ProtectedRuntimeError("worker verification output is missing")
    try:
        payload = json.loads(
            stdout,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError) as exc:
        raise ProtectedRuntimeError(
            "worker verification output is not strict JSON"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"modules", "ok", "prefix"}
        or payload["ok"] is not True
        or payload["modules"] != list(LEARNED_IMPORT_MODULES)
    ):
        raise ProtectedRuntimeError("worker verification was unsuccessful")
    prefix = payload["prefix"]
    if not isinstance(prefix, str) or Path(prefix) != expected_prefix:
        raise ProtectedRuntimeError(
            "worker prefix does not match the learned environment root"
        )


_FREEZE_NAME_PATTERN = re.compile(
    r"^([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"(?:===|==|\s+@\s+).+$"
)


def _canonical_freeze_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def _parse_worker_freeze(stdout: object) -> tuple[str, ...]:
    if not isinstance(stdout, str) or not stdout:
        raise ProtectedRuntimeError("worker pip freeze output is malformed")
    lines = stdout.split("\n")
    if lines[-1] == "":
        lines.pop()
    if not lines:
        raise ProtectedRuntimeError("worker pip freeze output is malformed")
    seen: set[str] = set()
    for line in lines:
        if not line or any(unicodedata.category(character) == "Cc" for character in line):
            raise ProtectedRuntimeError("worker pip freeze entry is malformed")
        match = _FREEZE_NAME_PATTERN.fullmatch(line)
        if match is None:
            raise ProtectedRuntimeError("worker pip freeze entry is malformed")
        canonical = _canonical_freeze_name(match.group(1))
        if canonical in seen:
            raise ProtectedRuntimeError(
                f"worker pip freeze contains duplicate package {match.group(1)}"
            )
        seen.add(canonical)
    return tuple(sorted(lines))


def _runtime_manifest_payload(
    snapshot: ProtectedRuntimeSnapshot,
    label: str,
) -> dict[str, object]:
    packages = _validated_snapshot_packages(snapshot, label)
    return {
        "python_exe": str(snapshot.python_exe),
        "packages": [
            {
                "name": packages[name].name,
                "version": packages[name].version,
                "module_path": packages[name].module_path,
            }
            for name in PROTECTED_PACKAGES
        ],
    }


def _manifest_payload(
    environment: LearnedEnvironment,
    assets: LearnedAssets,
    source_root: Path,
    checkpoint_root: Path,
    freeze: tuple[str, ...],
    verified_runtime: ProtectedRuntimeSnapshot,
) -> dict[str, object]:
    return {
        "asset_roots": {
            "checkpoints": str(checkpoint_root),
            "sources": str(source_root),
        },
        "checkpoints": [
            {
                "code_commit": checkpoint.model.code_commit,
                "license_id": checkpoint.model.license_id,
                "local_path": str(checkpoint.local_path),
                "repo_id": checkpoint.model.repo_id,
                "requested_revision": checkpoint.model.revision,
                "resolved_revision": checkpoint.resolved_revision,
            }
            for checkpoint in assets.checkpoints
        ],
        "lock_path": str(environment.lock_path),
        "protected_host": {
            "after_install": _runtime_manifest_payload(
                environment.main_runtime_after,
                "after install",
            ),
            "before_install": _runtime_manifest_payload(
                environment.main_runtime_before,
                "before install",
            ),
            "verified": _runtime_manifest_payload(
                verified_runtime,
                "verified",
            ),
        },
        "schema_version": 1,
        "sources": [
            {
                "license_id": source.license_id,
                "name": source.name,
                "path": str(source.path),
                "repo_url": source.repo_url,
                "requested_commit": source.requested_commit,
                "resolved_commit": source.resolved_commit,
            }
            for source in assets.sources
        ],
        "worker_pip_freeze": list(freeze),
        "worker_python": str(environment.python_exe),
    }


def _write_model_manifest(path: Path, payload: Mapping[str, object]) -> str:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ProtectedRuntimeError(
            "model manifest path must be a regular non-symlink file"
        )
    try:
        data = (
            json.dumps(
                payload,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtectedRuntimeError("model manifest is not strict JSON") from exc

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=".model_manifest.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise
    return sha256(data).hexdigest()


def verify_learned_environment(
    environment: LearnedEnvironment,
    assets: LearnedAssets,
    *,
    runner: Callable[..., CompletedProcess[str]] = subprocess.run,
) -> ModelManifest:
    source_root, checkpoint_root = _validate_assets_for_verification(
        environment,
        assets,
    )
    manifest_path = environment.root / "model_manifest.json"
    if manifest_path.is_symlink() or (
        manifest_path.exists() and not manifest_path.is_file()
    ):
        raise ProtectedRuntimeError(
            "model manifest path must be a regular non-symlink file"
        )

    for source, approved in zip(
        assets.sources,
        SOURCE_REPOSITORY_REFS,
        strict=True,
    ):
        _verify_source_checkout(source.path, approved, runner)

    worker_error: Exception | None = None
    freeze: tuple[str, ...] | None = None
    try:
        source_payload = json.dumps(
            {source.name: str(source.path) for source in assets.sources},
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        completed = _run_checked(
            runner,
            [
                str(environment.python_exe),
                "-I",
                "-B",
                "-c",
                _VERIFY_SCRIPT,
                source_payload,
            ],
        )
        _parse_worker_verification(completed.stdout, environment.root)
        freeze = _parse_worker_freeze(
            _run_checked(
                runner,
                [str(environment.python_exe), "-m", "pip", "freeze"],
            ).stdout
        )
    except Exception as exc:
        worker_error = exc

    try:
        verified_runtime = _capture_and_compare_runtime(
            environment.main_runtime_before.python_exe,
            environment.main_runtime_before,
            runner,
        )
        assert_no_protected_changes(
            {},
            before=environment.main_runtime_after,
            after=verified_runtime,
        )
    except Exception as drift_error:
        if isinstance(drift_error, ProtectedRuntimeError):
            raise drift_error from worker_error
        raise ProtectedRuntimeError(
            "protected host runtime verification failed"
        ) from drift_error

    if worker_error is not None:
        raise worker_error
    if freeze is None:
        raise ProtectedRuntimeError("worker pip freeze output is missing")

    digest = _write_model_manifest(
        manifest_path,
        _manifest_payload(
            environment,
            assets,
            source_root,
            checkpoint_root,
            freeze,
            verified_runtime,
        ),
    )
    return ModelManifest(
        path=manifest_path,
        sha256=digest,
        sources=assets.sources,
        checkpoints=assets.checkpoints,
        worker_pip_freeze=freeze,
        main_runtime=verified_runtime,
    )

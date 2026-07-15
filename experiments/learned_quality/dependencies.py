from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess

from .contracts import ModelRef


PROTECTED_PACKAGES = ("torch", "torchvision", "numpy", "gsplat")
DEFAULT_ENV_ROOT = Path("/content/learned-env")
DEFAULT_LOCK_PATH = Path(__file__).with_name("requirements-lock.txt")

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
    return name.strip().casefold().replace("-", "").replace("_", "")


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

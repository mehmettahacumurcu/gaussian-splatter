from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

from backend.static_pipeline.contracts import ArtifactRecord

from .contracts import GENERATOR_ID


MANIFEST_SCHEMA_VERSION = 1
_CONTACT_SHEETS = frozenset(
    {
        "masks_contact_sheet.png",
        "depth_contact_sheet.png",
        "geometry_contact_sheet.png",
        "final_render_contact_sheet.png",
    }
)
_REQUIRED_FILES = frozenset(
    {
        "splat.ply",
        "preview.png",
        "scene_metadata.json",
        "quality_report.json",
        "run_manifest.json",
        "selection_manifest.json",
        "experiment_report.json",
        "geometry_candidates.json",
        "model_manifest.json",
        "diagnostics/masks_contact_sheet.png",
        "diagnostics/depth_contact_sheet.png",
        "diagnostics/geometry_contact_sheet.png",
        "diagnostics/final_render_contact_sheet.png",
        "diagnostics/density_history.json",
        "diagnostics/photometric_report.json",
        "logs/pipeline.log",
    }
)
_WEB_SUFFIXES = frozenset({".html", ".htm", ".js", ".mjs", ".wasm"})
_GEOMETRY_ACCEPTANCE_KEYS = (
    "geometry_acceptance_mode",
    "geometry_policy_version",
    "geometry_strict_failures",
    "geometry_guarded_metrics",
    "geometry_colmap_fingerprint",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(path: Path, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")


def _plain_tree(root: Path) -> None:
    try:
        metadata = os.lstat(root)
    except OSError as exc:
        raise ValueError("bundle must be an existing directory") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("bundle root must be a regular non-symlink directory")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"bundle contains a symlink: {path.relative_to(root)}")


def _safe_relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact path must be portable and relative")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("artifact path escapes the bundle")
    return relative


def _strict_read_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid constant {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON file: {path.name}") from exc


def _json_object(path: Path) -> dict[str, object]:
    value = _strict_read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path.name}")
    return value


def _json_safe(value: object) -> object:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.write_text(encoded + "\n", encoding="utf-8", newline="\n")


def inventory_learned_bundle(root: Path) -> tuple[ArtifactRecord, ...]:
    root = Path(root)
    _plain_tree(root)
    records: list[ArtifactRecord] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"run_manifest.json", "_SUCCESS"}:
            continue
        _safe_relative(relative)
        records.append(ArtifactRecord(relative, path.stat().st_size, _sha256(path)))
    return tuple(records)


def _validate_ply(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(64 * 1024).replace(b"\r\n", b"\n")
    end = header.find(b"end_header\n")
    if (
        not header.startswith(b"ply\n")
        or end < 0
        or b"format " not in header[:end]
        or b"element vertex " not in header[:end]
        or any(
            token not in header[:end]
            for token in (b"property float x", b"property float y", b"property float z")
        )
    ):
        raise ValueError("splat.ply has an invalid PLY header")


def _manifest_inventory(value: object) -> tuple[ArtifactRecord, ...]:
    if not isinstance(value, list):
        raise ValueError("run manifest artifacts must be a list")
    records: list[ArtifactRecord] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("run manifest artifact record is invalid")
        relative = item["relative_path"]
        _safe_relative(relative)
        size = item["size_bytes"]
        digest = item["sha256"]
        if (
            relative in seen
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("run manifest artifact record is invalid")
        seen.add(relative)
        records.append(ArtifactRecord(relative, size, digest))
    return tuple(sorted(records, key=lambda record: record.relative_path))


def validate_learned_bundle(
    root: Path,
    *,
    run_id: str | None = None,
) -> tuple[ArtifactRecord, ...]:
    root = Path(root)
    _plain_tree(root)
    existing_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    missing = sorted(_REQUIRED_FILES - existing_files)
    if missing:
        raise ValueError(
            f"learned bundle is missing required files: {', '.join(missing)}"
        )
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.casefold() in _WEB_SUFFIXES:
            raise ValueError(f"learned bundle contains a web asset: {path.name}")
    for path in root.rglob("*.json"):
        _strict_read_json(path)
    _validate_ply(root / "splat.ply")
    for relative in (
        "preview.png",
        *tuple(f"diagnostics/{name}" for name in sorted(_CONTACT_SHEETS)),
    ):
        if not (root / relative).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"{relative} has an invalid PNG signature")

    manifest = _json_object(root / "run_manifest.json")
    if manifest.get("generator_id") != GENERATOR_ID:
        raise ValueError("run manifest has an unexpected generator_id")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("run manifest schema is unsupported")
    manifest_run_id = manifest.get("run_id")
    if not isinstance(manifest_run_id, str) or not manifest_run_id:
        raise ValueError("run manifest run_id is invalid")
    if run_id is not None and manifest_run_id != run_id:
        raise ValueError("run manifest does not match the requested run_id")
    if manifest.get("status") != "success":
        raise ValueError("run manifest status must be success")
    expected = _manifest_inventory(manifest.get("artifacts"))
    actual = inventory_learned_bundle(root)
    if expected != actual:
        raise ValueError(
            "run manifest artifact inventory does not match bundle contents"
        )
    experiment = _json_object(root / "experiment_report.json")
    quality = _json_object(root / "quality_report.json")
    learned_quality = quality.get("learned_quality")
    if not isinstance(learned_quality, dict):
        raise ValueError("quality report learned_quality metadata is required")
    geometry_metadata = {
        key: manifest.get(key) for key in _GEOMETRY_ACCEPTANCE_KEYS
    }
    if geometry_metadata != {
        key: experiment.get(key) for key in _GEOMETRY_ACCEPTANCE_KEYS
    } or geometry_metadata != {
        key: learned_quality.get(key) for key in _GEOMETRY_ACCEPTANCE_KEYS
    }:
        raise ValueError("geometry acceptance metadata differs across reports")
    mode = geometry_metadata["geometry_acceptance_mode"]
    policy = geometry_metadata["geometry_policy_version"]
    failures = geometry_metadata["geometry_strict_failures"]
    metrics = geometry_metadata["geometry_guarded_metrics"]
    fingerprint = geometry_metadata["geometry_colmap_fingerprint"]
    if mode not in {"strict", "best_effort"}:
        raise ValueError("geometry acceptance mode is invalid")
    if policy not in {"strict-v1", "output-first-v1"}:
        raise ValueError("geometry policy version is invalid")
    if not isinstance(failures, list) or any(
        not isinstance(failure, str) or not failure for failure in failures
    ):
        raise ValueError("geometry strict failures are invalid")
    if mode == "strict":
        if policy != "strict-v1" or metrics is not None or fingerprint is not None:
            raise ValueError("strict geometry contains recovery-only claims")
    elif (
        policy != "output-first-v1"
        or not failures
        or not isinstance(metrics, dict)
        or not isinstance(fingerprint, str)
        or len(fingerprint) != 64
    ):
        raise ValueError("best-effort geometry metadata is incomplete")
    source = manifest.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("digest"), str)
        or len(source["digest"]) != 64
    ):
        raise ValueError("run manifest source digest is invalid")
    if (
        not isinstance(manifest.get("repository_commit"), str)
        or not manifest["repository_commit"]
    ):
        raise ValueError("run manifest repository_commit is required")
    for key in ("tool_versions", "timing", "hardware"):
        if not isinstance(manifest.get(key), dict) or not manifest[key]:
            raise ValueError(f"run manifest {key} must be a non-empty object")
    return actual


def finalize_learned_bundle(
    bundle_root: Path,
    *,
    run_id: str,
    experiment_report: Mapping[str, object],
    geometry_candidates: object,
    model_manifest_path: Path,
    contact_sheets: Mapping[str, Path],
    density_history_path: Path,
    photometric_report_path: Path,
) -> Path:
    bundle_root = Path(bundle_root)
    _plain_tree(bundle_root)
    if set(contact_sheets) != _CONTACT_SHEETS:
        raise ValueError("all four exact contact sheets are required")
    for path, label in (
        (Path(model_manifest_path), "model manifest"),
        (Path(density_history_path), "density history"),
        (Path(photometric_report_path), "photometric report"),
        *((Path(path), name) for name, path in contact_sheets.items()),
    ):
        _plain_file(path, label)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be non-empty")

    diagnostics = bundle_root / "diagnostics"
    diagnostics.mkdir(exist_ok=False)
    shutil.copyfile(model_manifest_path, bundle_root / "model_manifest.json")
    shutil.copyfile(density_history_path, diagnostics / "density_history.json")
    shutil.copyfile(photometric_report_path, diagnostics / "photometric_report.json")
    for name, source in contact_sheets.items():
        shutil.copyfile(source, diagnostics / name)
    report = dict(experiment_report)
    geometry_metadata = {
        "geometry_acceptance_mode": report.get(
            "geometry_acceptance_mode", "strict"
        ),
        "geometry_policy_version": report.get(
            "geometry_policy_version", "strict-v1"
        ),
        "geometry_strict_failures": report.get("geometry_strict_failures", []),
        "geometry_guarded_metrics": report.get("geometry_guarded_metrics"),
        "geometry_colmap_fingerprint": report.get(
            "geometry_colmap_fingerprint"
        ),
    }
    if geometry_metadata["geometry_acceptance_mode"] == "best_effort":
        report["status"] = "best_effort"
    report.update(geometry_metadata)
    report.update({"generator_id": GENERATOR_ID, "run_id": run_id})
    _write_json(bundle_root / "experiment_report.json", report)
    _write_json(bundle_root / "geometry_candidates.json", geometry_candidates)

    quality = _json_object(bundle_root / "quality_report.json")
    quality["learned_quality"] = {
        "experiment_report": "experiment_report.json",
        "geometry_candidates": "geometry_candidates.json",
        "model_manifest": "model_manifest.json",
        "contact_sheets": [f"diagnostics/{name}" for name in sorted(_CONTACT_SHEETS)],
        **geometry_metadata,
    }
    _write_json(bundle_root / "quality_report.json", quality)

    manifest = _json_object(bundle_root / "run_manifest.json")
    if manifest.get("run_id") != run_id:
        raise ValueError("base bundle run_id does not match learned run")
    manifest["generator_id"] = GENERATOR_ID
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["status"] = "success"
    manifest.update(geometry_metadata)
    manifest["artifacts"] = [
        asdict(record) for record in inventory_learned_bundle(bundle_root)
    ]
    _write_json(bundle_root / "run_manifest.json", manifest)
    validate_learned_bundle(bundle_root, run_id=run_id)
    return bundle_root

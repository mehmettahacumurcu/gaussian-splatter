from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


RESULT_PATH = Path("/content/legacy_control_export_result.json")
FAILURE_DRIVE_ROOT = Path("/content/drive/MyDrive")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _source_revision(value: str) -> str:
    revision = value.strip().lower()
    if _GIT_SHA.fullmatch(revision) is None:
        raise argparse.ArgumentTypeError("source revision must be a full Git SHA")
    return revision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce one verified legacy-control 5K training and publish its "
            "raw and production-polish candidate PLYs."
        )
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--source-revision", required=True, type=_source_revision)
    parser.add_argument("--model-manifest", type=Path)
    return parser


def _load_runner() -> Callable[..., object]:
    from experiments.learned_quality.legacy_control_export import (
        run_legacy_control_export,
    )

    return run_legacy_control_export


def _write_receipt(payload: dict[str, object]) -> None:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_PATH.with_name(f".{RESULT_PATH.name}.tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, RESULT_PATH)
    print(encoded, flush=True)


def _result_payload(result: Any) -> dict[str, object]:
    final_path = Path(result.final_path).resolve()
    raw = Path(result.raw_ply_path).resolve()
    polished = Path(result.polished_ply_path).resolve()
    expected = {
        final_path / "raw_legacy_control_5k.ply",
        final_path / "polished_legacy_control_5k.ply",
    }
    actual = {raw, polished}
    if actual != expected or not raw.is_file() or not polished.is_file():
        raise RuntimeError("published PLY is missing or outside the result folder")
    return {
        "status": "success",
        "run_id": str(result.run_id),
        "final_path": str(final_path),
        "raw_ply_path": str(raw),
        "polished_ply_path": str(polished),
        "polish_accepted": bool(result.polish_accepted),
        "polish_reasons": list(result.polish_reasons),
        "full_120k_training_started": False,
    }


def _publish_durable_failure(
    spec: object,
    error: Exception,
    *,
    source_revision: str,
) -> Path:
    input_folder = str(getattr(spec, "input_folder"))
    input_name = Path(input_folder).name
    destination = (
        FAILURE_DRIVE_ROOT
        / f"{input_name}_legacy_control_5k_failures"
        / f"{uuid.uuid4().hex}.json"
    )
    payload = {
        "schema_version": 1,
        "status": "failed",
        "input_folder": input_folder,
        "source_revision": source_revision,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "full_120k_training_started": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from experiments.learned_quality.legacy_control_export import (
            LegacyControlExportRunSpec,
        )

        spec = LegacyControlExportRunSpec.model_validate_json(args.spec.read_bytes())
        raw_manifest = args.model_manifest or os.environ.get(
            "LEARNED_MODEL_MANIFEST"
        )
        if raw_manifest is None:
            raise ValueError(
                "model manifest is required via --model-manifest or "
                "LEARNED_MODEL_MANIFEST"
            )
        model_manifest = Path(raw_manifest).resolve()
    except Exception as error:
        _write_receipt(
            {
                "status": "invalid_spec",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        return 2

    try:
        result = _load_runner()(
            spec,
            model_manifest_path=model_manifest,
            expected_source_revision=args.source_revision,
        )
        payload = _result_payload(result)
    except Exception as error:
        payload = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "full_120k_training_started": False,
        }
        try:
            payload["durable_failure_path"] = str(
                _publish_durable_failure(
                    spec,
                    error,
                    source_revision=args.source_revision,
                )
            )
        except Exception as publication_error:
            payload["durable_failure_error"] = (
                f"{type(publication_error).__name__}: {publication_error}"
            )
        _write_receipt(payload)
        return 1

    _write_receipt(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

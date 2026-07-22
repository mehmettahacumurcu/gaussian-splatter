from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


RESULT_PATH = Path("/content/learned_floor_recovery_result.json")
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
            "Run one verified 5K automatic floor-hole recovery diagnostic "
            "against the preserved A100 legacy control."
        )
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--source-revision", required=True, type=_source_revision)
    parser.add_argument("--model-manifest", type=Path)
    return parser


def _load_runner() -> Callable[..., object]:
    from experiments.learned_quality.floor_recovery_runner import (
        run_floor_recovery_diagnostic,
    )

    return run_floor_recovery_diagnostic


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
    decision = getattr(result, "decision", None)
    return {
        "status": str(result.status),
        "run_id": str(result.run_id),
        "final_path": str(result.final_path) if result.final_path is not None else None,
        "decision": asdict(decision) if decision is not None else None,
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
        / f"{input_name}_floor_recovery_diagnostic_failures"
        / f"{uuid.uuid4().hex}.json"
    )
    payload = {
        "schema_version": 1,
        "status": "failed",
        "input_folder": input_folder,
        "source_revision": source_revision,
        "completed_stage": getattr(
            error, "floor_recovery_completed_stage", None
        ),
        "input_fingerprints": getattr(
            error, "floor_recovery_input_fingerprints", {}
        ),
        "evidence_paths": list(
            getattr(error, "floor_recovery_evidence_paths", ())
        ),
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
        from experiments.learned_quality.floor_recovery_runner import (
            FloorRecoveryRunSpec,
        )

        spec = FloorRecoveryRunSpec.model_validate_json(args.spec.read_bytes())
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
    except Exception as error:
        payload: dict[str, object] = {
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

    payload = _result_payload(result)
    _write_receipt(payload)
    return 0 if result.status in {"success", "rejected"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

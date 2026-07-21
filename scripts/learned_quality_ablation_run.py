from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


RESULT_PATH = Path("/content/learned_ablation_result.json")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _source_revision(value: str) -> str:
    revision = value.strip().lower()
    if _GIT_SHA.fullmatch(revision) is None:
        raise argparse.ArgumentTypeError("source revision must be a full Git SHA")
    return revision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone A100 learned-training ablation matrix from one "
            "verified local cache snapshot."
        )
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--source-revision", required=True, type=_source_revision)
    parser.add_argument("--model-manifest", type=Path)
    return parser


def _load_runner() -> Callable[..., object]:
    from experiments.learned_quality.ablation_runner import run_training_ablation

    return run_training_ablation


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
    matrix = result.matrix
    diagnosis = matrix.diagnosis
    return {
        "status": "success" if result.complete else "partial",
        "run_id": result.run_id,
        "final_path": str(result.final_path),
        "complete": bool(result.complete),
        "diagnosis": diagnosis.kind,
        "causes": [list(cause) for cause in diagnosis.causes],
        "experiments": sorted(matrix.results),
        "errors": dict(matrix.errors),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from experiments.learned_quality.ablation_runner import AblationRunSpec

        spec = AblationRunSpec.model_validate_json(args.spec.read_bytes())
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
        _write_receipt(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        return 1

    payload = _result_payload(result)
    _write_receipt(payload)
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

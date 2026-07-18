from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated A100 learned-quality Gaussian experiment."
    )
    parser.add_argument("--spec", required=True, type=Path)
    return parser


def _load_runner() -> Callable[[Any], object]:
    from experiments.learned_quality.runner import run_learned_quality_notebook

    return run_learned_quality_notebook


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec_path = args.spec.resolve()
    receipt_path = spec_path.with_name("learned_run_result.json")
    try:
        raw_spec = spec_path.read_bytes()
        from experiments.learned_quality.contracts import parse_learned_spec_json

        spec = parse_learned_spec_json(raw_spec)
    except Exception as exc:
        _write_receipt(
            receipt_path,
            {
                "status": "invalid_spec",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        return 2

    try:
        result = _load_runner()(spec)
    except Exception as exc:
        diagnostics = getattr(exc, "diagnostics_path", None)
        _write_receipt(
            receipt_path,
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "diagnostics_path": str(diagnostics) if diagnostics else None,
                "durable_milestone_kind": getattr(exc, "durable_milestone_kind", None),
                "durable_milestone_fingerprint": getattr(
                    exc, "durable_milestone_fingerprint", None
                ),
                "next_stage_id": getattr(exc, "next_stage_id", None),
            },
        )
        return 1

    _write_receipt(
        receipt_path,
        {
            "status": "success",
            "run_id": result.run_id,
            "final_path": str(result.final_path),
            "local_bundle": str(result.local_bundle),
            "quality_report_path": str(result.quality_report_path),
            "manifest_path": str(result.manifest_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

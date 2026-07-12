from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a validated static Gaussian-splat notebook specification."
    )
    parser.add_argument("--spec", required=True, type=Path)
    return parser


def _load_runner() -> Callable[[Any], object]:
    from backend.static_pipeline.runner import run_static_notebook

    return run_static_notebook


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec_path = args.spec.resolve()
    receipt_path = spec_path.with_name("run_result.json")
    try:
        raw_spec = spec_path.read_bytes()
        from backend.notebooks.models import parse_run_spec_json

        spec = parse_run_spec_json(raw_spec)
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
                "diagnostics_path": (
                    str(diagnostics) if diagnostics is not None else None
                ),
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

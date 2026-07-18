from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

from experiments.learned_quality.audit import run_cache_audit
from experiments.learned_quality.contracts import (
    LearnedQualityRunSpec,
    derive_learned_cache_root,
    to_static_run_spec,
)


RESULT_PATH = Path("/content/learned_audit_result.json")


def _write_result(payload: object) -> None:
    RESULT_PATH.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CPU-only learned-quality selection/COLMAP/track audit"
    )
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec = LearnedQualityRunSpec.model_validate_json(
            args.spec.read_text(encoding="utf-8")
        )
        drive_root = Path(
            os.environ.get("FOURDGS_DRIVE_ROOT", "/content/drive/MyDrive")
        ).resolve(strict=True)
        input_path = drive_root.joinpath(*spec.input_folder.split("/")).resolve(
            strict=True
        )
        input_path.relative_to(drive_root)
        cache_root = derive_learned_cache_root(input_path)
        local_root = Path("/content/4dgs-audits") / uuid.uuid4().hex
        print(f"[CPU AUDIT] Input inventory: {input_path}", flush=True)
        print(f"[CPU AUDIT] Drive cache: {cache_root}", flush=True)
        receipts = run_cache_audit(
            input_path=input_path,
            cache_root=cache_root,
            local_root=local_root,
            spec=to_static_run_spec(spec),
        )
        payload = {
            "status": "success",
            "input_path": str(input_path),
            "cache_root": str(cache_root),
            "receipts": [asdict(receipt) for receipt in receipts],
        }
        _write_result(payload)
        for receipt in receipts:
            print(
                "TRACK AUDIT PASSED - "
                f"COLMAP {receipt.colmap_fingerprint} - "
                f"{receipt.accepted_track_count} qualified tracks",
                flush=True,
            )
        return 0
    except BaseException as error:
        _write_result(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

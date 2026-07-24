from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path


_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _source_revision(value: str) -> str:
    revision = value.strip().lower()
    if _GIT_SHA.fullmatch(revision) is None:
        raise argparse.ArgumentTypeError("source revision must be a full Git SHA")
    return revision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one local native-1080p legacy-control training through 30K "
            "and preserve local PLY and optimizer checkpoints."
        )
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--source-revision", required=True, type=_source_revision)
    parser.add_argument("--model-manifest", type=Path)
    return parser


def _load_runner() -> Callable[..., object]:
    from experiments.learned_quality.legacy_control_long_run import (
        run_legacy_control_long,
    )

    return run_legacy_control_long


def _validate_local_result(result: object) -> tuple[Path, tuple[Path, ...], tuple[Path, ...]]:
    root = Path(getattr(result, "local_root")).resolve(strict=True)
    plys = tuple(Path(path).resolve(strict=True) for path in getattr(result, "ply_paths"))
    checkpoints = tuple(
        Path(path).resolve(strict=True)
        for path in getattr(result, "checkpoint_paths")
    )
    if len(plys) != 6 or len(checkpoints) != 6:
        raise RuntimeError("local snapshot set is incomplete")
    for path in (*plys, *checkpoints):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"local snapshot is missing: {path}")
        if not path.is_relative_to(root):
            raise RuntimeError(f"local snapshot escaped its run root: {path}")
        if "drive" in {part.casefold() for part in path.parts}:
            raise RuntimeError(f"local snapshot unexpectedly targets Drive: {path}")
    return root, plys, checkpoints


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from experiments.learned_quality.legacy_control_long_run import (
            LegacyControlLongRunSpec,
        )

        spec = LegacyControlLongRunSpec.model_validate_json(args.spec.read_bytes())
        raw_manifest = args.model_manifest or os.environ.get(
            "LEARNED_MODEL_MANIFEST"
        )
        if raw_manifest is None:
            raise ValueError(
                "model manifest is required via --model-manifest or "
                "LEARNED_MODEL_MANIFEST"
            )
        model_manifest = Path(raw_manifest).resolve(strict=True)
    except Exception as error:
        print(
            f"Invalid local-run configuration: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    try:
        result = _load_runner()(
            spec,
            model_manifest_path=model_manifest,
            expected_source_revision=args.source_revision,
        )
        root, plys, checkpoints = _validate_local_result(result)
    except Exception as error:
        print(
            f"Local training failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exception(type(error), error, error.__traceback__)
        return 1

    print(f"LOCAL_RUN_ROOT={root}", flush=True)
    for path in plys:
        print(f"PLY={path}", flush=True)
    for path in checkpoints:
        print(f"CHECKPOINT={path}", flush=True)
    print("Training finished. The Colab session remains assigned.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

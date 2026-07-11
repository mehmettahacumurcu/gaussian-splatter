from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.static_pipeline.stage_cache import (
    cache_matches,
    promote_directory,
    stage_fingerprint,
)


def test_source_or_selection_change_invalidates_stage() -> None:
    first = stage_fingerprint(
        "frames",
        inputs={"source": "a", "selection": "one"},
        settings={"cap": 1280},
        policy_version="v1",
    )
    changed_source = stage_fingerprint(
        "frames",
        inputs={"source": "b", "selection": "one"},
        settings={"cap": 1280},
        policy_version="v1",
    )
    changed_selection = stage_fingerprint(
        "frames",
        inputs={"source": "a", "selection": "two"},
        settings={"cap": 1280},
        policy_version="v1",
    )
    changed_policy = stage_fingerprint(
        "frames",
        inputs={"source": "a", "selection": "one"},
        settings={"cap": 1280},
        policy_version="v2",
    )

    assert first == "6a2c32714ecd06d125dbda940f0fcc5548c765be8768ff4250a18949ecf9bb18"
    assert len({first, changed_source, changed_selection, changed_policy}) == 4


def test_upstream_fingerprint_changes_invalidate_every_downstream_stage() -> None:
    def chain(source_digest: str) -> tuple[str, str]:
        selection = stage_fingerprint(
            "selection",
            inputs={"source": source_digest},
            settings={"mode": "smart"},
            policy_version="selection-v1",
        )
        colmap = stage_fingerprint(
            "colmap",
            inputs={"selection": selection},
            settings={"matcher": "sequential"},
            policy_version="reconstruction-v1",
        )
        return selection, colmap

    first_selection, first_colmap = chain("source-a")
    changed_selection, changed_colmap = chain("source-b")

    assert first_selection != changed_selection
    assert first_colmap != changed_colmap


def test_stage_fingerprint_is_canonical_and_includes_tools() -> None:
    first = stage_fingerprint(
        "colmap",
        inputs={"selection": "two", "source": "one"},
        settings={"nested": {"z": 2, "a": [1, True, None]}},
        policy_version="v2",
        tools={"ffmpeg": "7", "colmap": "3.11"},
    )
    reordered = stage_fingerprint(
        "colmap",
        inputs={"source": "one", "selection": "two"},
        settings={"nested": {"a": [1, True, None], "z": 2}},
        policy_version="v2",
        tools={"colmap": "3.11", "ffmpeg": "7"},
    )
    changed_tool = stage_fingerprint(
        "colmap",
        inputs={"source": "one", "selection": "two"},
        settings={"nested": {"a": [1, True, None], "z": 2}},
        policy_version="v2",
        tools={"colmap": "3.12", "ffmpeg": "7"},
    )

    assert first == reordered
    assert first != changed_tool
    assert len(first) == 64


@pytest.mark.parametrize(
    "settings,expected_exception",
    [
        ({"path": Path("not-json")}, TypeError),
        ({"bad": float("nan")}, ValueError),
        ({"bad": float("inf")}, ValueError),
    ],
)
def test_stage_fingerprint_rejects_non_json_settings(
    settings: dict[str, object],
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception):
        stage_fingerprint(
            "frames",
            inputs={"source": "digest"},
            settings=settings,
            policy_version="v1",
        )


def test_cache_matches_requires_schema_fingerprint_and_all_paths(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "marker.json"
    required_file = tmp_path / "frames" / "frame_000001.png"
    required_file.parent.mkdir()
    required_file.write_bytes(b"frame")
    marker.write_text(
        json.dumps({"schema_version": 1, "fingerprint": "expected"}),
        encoding="utf-8",
    )

    assert cache_matches(marker, "expected", (required_file,))
    assert not cache_matches(marker, "wrong", (required_file,))
    assert not cache_matches(marker, "expected", (tmp_path / "missing",))

    marker.write_text(
        json.dumps({"schema_version": 2, "fingerprint": "expected"}),
        encoding="utf-8",
    )
    assert not cache_matches(marker, "expected", (required_file,))

    marker.write_text("not-json", encoding="utf-8")
    assert not cache_matches(marker, "expected", (required_file,))


@pytest.mark.parametrize("schema_version", [True, 1.0, "1", None])
def test_cache_matches_rejects_non_integer_schema_one(
    tmp_path: Path,
    schema_version: object,
) -> None:
    marker = tmp_path / "marker.json"
    required_file = tmp_path / "frame.png"
    required_file.write_bytes(b"frame")
    marker.write_text(
        json.dumps({"schema_version": schema_version, "fingerprint": "expected"}),
        encoding="utf-8",
    )

    assert not cache_matches(marker, "expected", (required_file,))


def test_promote_directory_requires_same_parent_and_absent_target(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "frames.staging"
    staging.mkdir()
    (staging / "frame.png").write_bytes(b"frame")
    target = tmp_path / "frames"

    promote_directory(staging, target)

    assert not staging.exists()
    assert (target / "frame.png").read_bytes() == b"frame"

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    with pytest.raises(FileExistsError):
        promote_directory(replacement, target)

    other_parent = tmp_path / "other"
    other_parent.mkdir()
    with pytest.raises(ValueError, match="same parent"):
        promote_directory(replacement, other_parent / "target")

    with pytest.raises(FileNotFoundError):
        promote_directory(tmp_path / "missing", tmp_path / "new-target")

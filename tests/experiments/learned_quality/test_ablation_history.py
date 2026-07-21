from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import pytest

from experiments.learned_quality.ablation_history import analyze_historical_run


def _write_binary_ply(path: Path) -> None:
    properties = (
        "x",
        "y",
        "z",
        "scale_0",
        "scale_1",
        "scale_2",
        "opacity",
    )
    header = [
        "ply",
        "format binary_little_endian 1.0",
        "element vertex 3",
        *(f"property float {name}" for name in properties),
        "end_header",
        "",
    ]
    rows = (
        (0.0, 0.0, 0.0, math.log(0.01), math.log(0.02), math.log(0.03), 0.0),
        (1.0, 2.0, 3.0, math.log(0.10), math.log(0.20), math.log(0.30), 2.0),
        (-1.0, -2.0, -3.0, math.log(1.00), math.log(2.00), math.log(3.00), -2.0),
    )
    with path.open("wb") as stream:
        stream.write("\n".join(header).encode("ascii"))
        for row in rows:
            stream.write(struct.pack("<7f", *row))


def test_historical_analyzer_reads_ply_reports_and_density_events(tmp_path: Path) -> None:
    root = tmp_path / "old-result"
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(parents=True)
    _write_binary_ply(root / "splat.ply")
    (root / "run_manifest.json").write_text(
        json.dumps({"resolved_config": {"n_iters": 120_000}}),
        encoding="utf-8",
    )
    (root / "experiment_report.json").write_text(
        json.dumps({"final_gaussian_count": 550_295}),
        encoding="utf-8",
    )
    (diagnostics / "density_history.json").write_text(
        json.dumps(
            [
                {"event": "density", "iter": 18_000, "before": 10, "after": 8},
                {"event": "opacity_reset", "iter": 18_000},
                {"event": "density", "iter": 18_100, "before": 8, "after": 12},
            ]
        ),
        encoding="utf-8",
    )

    summary = analyze_historical_run(root)

    assert summary["available"] is True
    assert summary["configured_iterations"] == 120_000
    assert summary["reported_final_gaussian_count"] == 550_295
    assert summary["ply"]["vertex_count"] == 3
    assert summary["ply"]["position_bounds"]["min"] == pytest.approx([-1, -2, -3])
    assert summary["ply"]["position_bounds"]["max"] == pytest.approx([1, 2, 3])
    assert summary["ply"]["activated_scale_quantiles"]["q100"] == pytest.approx(3.0)
    assert summary["density_history"]["density_events"] == 2
    assert summary["density_history"]["opacity_resets"] == 1


def test_historical_analyzer_is_nonfatal_when_result_is_absent(tmp_path: Path) -> None:
    summary = analyze_historical_run(tmp_path / "missing")

    assert summary == {
        "available": False,
        "reason": "historical_result_missing",
    }

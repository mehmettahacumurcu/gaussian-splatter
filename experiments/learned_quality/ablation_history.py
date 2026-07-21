from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_PLY_TYPES: Mapping[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def _load_json(path: Path) -> object | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {name: 0.0 for name in ("q0", "q50", "q95", "q99", "q100")}
    result = np.quantile(finite, (0.0, 0.5, 0.95, 0.99, 1.0))
    return {
        name: float(value)
        for name, value in zip(
            ("q0", "q50", "q95", "q99", "q100"),
            result,
            strict=True,
        )
    }


def _read_ply_header(path: Path) -> tuple[str, int, np.dtype[Any], int]:
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    active_element: str | None = None
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("historical PLY is missing its magic header")
        format_name: str | None = None
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("historical PLY header is incomplete")
            line = raw.decode("ascii").strip()
            parts = line.split()
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "format":
                format_name = parts[1]
            elif parts[0] == "element":
                active_element = parts[1]
                if active_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and active_element == "vertex":
                if parts[1] == "list":
                    raise ValueError("historical vertex list properties are unsupported")
                if parts[1] not in _PLY_TYPES:
                    raise ValueError(f"unsupported historical PLY type: {parts[1]}")
                vertex_properties.append((parts[2], _PLY_TYPES[parts[1]]))
            elif parts[0] == "end_header":
                offset = stream.tell()
                break
    if format_name not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"unsupported historical PLY format: {format_name}")
    if vertex_count is None or vertex_count < 0 or not vertex_properties:
        raise ValueError("historical PLY has no valid vertex element")
    return format_name, vertex_count, np.dtype(vertex_properties), offset


def _read_ply_vertices(
    path: Path,
    format_name: str,
    count: int,
    dtype: np.dtype[Any],
    offset: int,
) -> np.ndarray:
    if format_name == "binary_little_endian":
        return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(count,))
    rows = np.empty(count, dtype=dtype)
    names = dtype.names or ()
    with path.open("rb") as stream:
        stream.seek(offset)
        for index in range(count):
            values = stream.readline().decode("ascii").split()
            if len(values) < len(names):
                raise ValueError("historical ASCII PLY ended before all vertices")
            rows[index] = tuple(values[: len(names)])
    return rows


def analyze_gaussian_ply(path: Path) -> dict[str, object]:
    ply_path = Path(path)
    if not ply_path.is_file():
        return {"available": False, "reason": "ply_missing"}
    try:
        format_name, count, dtype, offset = _read_ply_header(ply_path)
        rows = _read_ply_vertices(ply_path, format_name, count, dtype, offset)
        names = set(dtype.names or ())
        result: dict[str, object] = {
            "available": True,
            "format": format_name,
            "vertex_count": count,
            "properties": sorted(names),
        }
        if {"x", "y", "z"}.issubset(names):
            positions = np.column_stack(
                [np.asarray(rows[name], dtype=np.float64) for name in ("x", "y", "z")]
            )
            finite = positions[np.isfinite(positions).all(axis=1)]
            if finite.size:
                centered = finite - np.median(finite, axis=0)
                covariance = centered.T @ centered / max(len(centered), 1)
                result["position_bounds"] = {
                    "min": finite.min(axis=0).tolist(),
                    "max": finite.max(axis=0).tolist(),
                }
                result["position_covariance_eigenvalues"] = np.linalg.eigvalsh(
                    covariance
                ).clip(min=0.0).tolist()
        scale_names = tuple(name for name in ("scale_0", "scale_1", "scale_2") if name in names)
        if scale_names:
            raw_scales = np.column_stack(
                [np.asarray(rows[name], dtype=np.float64) for name in scale_names]
            )
            result["raw_scale_quantiles"] = _quantiles(raw_scales)
            result["activated_scale_quantiles"] = _quantiles(
                np.exp(np.clip(raw_scales, -50.0, 50.0))
            )
        if "opacity" in names:
            logits = np.asarray(rows["opacity"], dtype=np.float64)
            activated = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
            result["activated_opacity_quantiles"] = _quantiles(activated)
        return result
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        return {
            "available": False,
            "reason": "ply_analysis_failed",
            "error": f"{type(error).__name__}: {error}",
        }


def _density_rows(payload: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(payload, list):
        return tuple(row for row in payload if isinstance(row, Mapping))
    if isinstance(payload, Mapping):
        for key in ("events", "history", "records", "density_history"):
            value = payload.get(key)
            if isinstance(value, list):
                return tuple(row for row in value if isinstance(row, Mapping))
    return ()


def _density_summary(payload: object) -> dict[str, object]:
    rows = _density_rows(payload)
    density = tuple(
        row
        for row in rows
        if row.get("event") == "density"
        or any(key in row for key in ("cloned", "split", "pruned", "before", "after"))
    )
    resets = tuple(row for row in rows if row.get("event") == "opacity_reset")
    return {
        "available": bool(rows),
        "records": len(rows),
        "density_events": len(density),
        "opacity_resets": len(resets),
        "largest_prune": max((int(row.get("pruned", 0)) for row in density), default=0),
        "peak_gaussian_count": max(
            (
                int(value)
                for row in density
                for value in (row.get("before", 0), row.get("after", 0))
            ),
            default=0,
        ),
    }


def analyze_historical_run(root: Path) -> dict[str, object]:
    result_root = Path(root)
    if not result_root.is_dir():
        return {"available": False, "reason": "historical_result_missing"}
    manifest = _load_json(result_root / "run_manifest.json")
    experiment = _load_json(result_root / "experiment_report.json")
    quality = _load_json(result_root / "quality_report.json")
    density = _load_json(result_root / "diagnostics" / "density_history.json")
    manifest_map = manifest if isinstance(manifest, Mapping) else {}
    experiment_map = experiment if isinstance(experiment, Mapping) else {}
    resolved = manifest_map.get("resolved_config", {})
    configured_iterations = (
        resolved.get("n_iters") if isinstance(resolved, Mapping) else None
    )
    return {
        "available": True,
        "configured_iterations": configured_iterations,
        "reported_final_gaussian_count": experiment_map.get("final_gaussian_count"),
        "ply": analyze_gaussian_ply(result_root / "splat.ply"),
        "density_history": _density_summary(density),
        "run_manifest": manifest,
        "experiment_report": experiment,
        "quality_report": quality,
    }

"""Run logger — training / pipeline metrics ve events'leri disk'e yazar.

Her sahne için `data/<scene>/output/logs/` altında 3 dosya:

  - metrics.jsonl : Per-iter training metrics (loss components, PSNR, N, Δpos, timing)
                    JSON Lines formatında, frontend'de chart çizmek için ideal.
  - events.log    : Insan-okunabilir event log (phase changes, density ops,
                    opacity resets, warnings). Debug / post-mortem için.
  - summary.json  : Run sonunda writelansa toplu özet — config snapshot,
                    phase durations, final metrics, PLY stats.

Kullanım:
    logger = RunLogger(output_dir=paths["output"] / "logs", scene="cutlemon_v3")
    logger.log_event("phase:start", phase="colmap")
    logger.log_metric(iter=100, loss=0.15, psnr=18.2, n_points=5000, ...)
    logger.save_summary({...})
    logger.close()
"""
from __future__ import annotations
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    """Dataclass, Path, tuple/list nested → JSON-serializable."""
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


class RunLogger:
    """
    Thread-unsafe (tek training thread varsayımı). Her çağrıda flush eder
    ki crash durumunda veri kaybolmasın.
    """

    def __init__(self, output_dir: str | Path, scene: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scene = scene
        self.t_start = time.time()
        self.phase_starts: dict[str, float] = {}

        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.events_path  = self.output_dir / "events.log"
        self.summary_path = self.output_dir / "summary.json"

        # Handles
        self._metrics_f = open(self.metrics_path, "w", encoding="utf-8", buffering=1)
        self._events_f  = open(self.events_path,  "w", encoding="utf-8", buffering=1)

        self.log_event("run:start", scene=scene, timestamp=self.t_start)

    # ------------------------------------------------------------------
    # Metrics (per-iter, JSONL)
    # ------------------------------------------------------------------
    def log_metric(self, **kwargs: Any) -> None:
        """Bir iter metric entry'si yaz."""
        record = {"t": round(time.time() - self.t_start, 3)}
        record.update(kwargs)
        self._metrics_f.write(json.dumps(_to_jsonable(record), ensure_ascii=False))
        self._metrics_f.write("\n")

    # ------------------------------------------------------------------
    # Events (human-readable)
    # ------------------------------------------------------------------
    def log_event(self, event: str, **kwargs: Any) -> None:
        """Bir event yaz — phase start/end, density op, warning vs."""
        dt = round(time.time() - self.t_start, 2)
        parts = [f"[{dt:>7.2f}s]", f"event={event}"]
        for k, v in kwargs.items():
            parts.append(f"{k}={v}")
        self._events_f.write(" ".join(parts) + "\n")

    # ------------------------------------------------------------------
    # Phase timing helpers
    # ------------------------------------------------------------------
    def phase_start(self, phase: str) -> None:
        self.phase_starts[phase] = time.time()
        self.log_event("phase:start", phase=phase)

    def phase_end(self, phase: str, **extra: Any) -> float:
        start = self.phase_starts.get(phase, self.t_start)
        dur = time.time() - start
        self.log_event("phase:end", phase=phase, duration=round(dur, 2), **extra)
        return dur

    # ------------------------------------------------------------------
    # Warnings / errors
    # ------------------------------------------------------------------
    def warn(self, msg: str, **kwargs: Any) -> None:
        self.log_event("warn", msg=msg, **kwargs)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    def save_summary(self, data: dict[str, Any]) -> None:
        data = dict(data)
        data["total_duration_sec"] = round(time.time() - self.t_start, 2)
        data["scene"] = self.scene
        self.summary_path.write_text(
            json.dumps(_to_jsonable(data), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.log_event("run:summary_saved", path=str(self.summary_path))

    # ------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._metrics_f.close()
        except Exception:
            pass
        try:
            self.log_event("run:end", total_sec=round(time.time() - self.t_start, 2))
            self._events_f.close()
        except Exception:
            pass

    # Context manager convenience
    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

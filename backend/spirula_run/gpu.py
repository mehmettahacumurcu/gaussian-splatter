"""Live GPU memory via nvidia-smi (Spirula is Vulkan, so torch can't see its usage)."""
from __future__ import annotations

import shutil
import subprocess

QUERY = "name,memory.total,memory.used,memory.free,utilization.gpu"


def parse_smi(text: str) -> list[dict]:
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        name, total, used, free, util = parts
        try:
            gpus.append({"name": name, "total_mib": int(total), "used_mib": int(used),
                         "free_mib": int(free), "util_pct": int(util) if util.isdigit() else None})
        except ValueError:
            continue
    return gpus


def gpu_status() -> dict:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"available": False, "gpus": [], "error": "nvidia-smi bulunamadı"}
    try:
        out = subprocess.run([smi, f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "gpus": [], "error": str(exc)}
    gpus = parse_smi(out.stdout)
    return {"available": bool(gpus), "gpus": gpus, "error": None if gpus else out.stderr.strip() or "GPU yok"}

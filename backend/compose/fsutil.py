"""Small filesystem helpers shared by the composer store, exporter and caches."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def replace_with_retry(src: str | Path, dst: str | Path, tries: int = 5, delay: float = 0.2) -> None:
    """``os.replace`` that tolerates transient Windows locks (antivirus, open handles)."""
    for attempt in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(delay)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file in the same directory + replace."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        replace_with_retry(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise

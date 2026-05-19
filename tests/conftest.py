"""Shared pytest fixtures for the 4dgs-studio backend test suite."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest


# Make the backend module importable without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def tmp_worlds_dir(tmp_path) -> Path:
    """A temporary `worlds/` root for disk-layout tests."""
    root = tmp_path / "worlds"
    root.mkdir()
    return root

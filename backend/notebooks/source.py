from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

REPOSITORY_URL = "https://github.com/mehmettahacumurcu/gaussian-splatter.git"
GENERATOR_ID = "4dgs-studio.static-notebook"
GENERATOR_VERSION = 1
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


class NotebookSourceError(RuntimeError):
    """A generated notebook cannot be pinned to reachable immutable source."""


@dataclass(frozen=True)
class NotebookSource:
    repo_url: str
    commit_sha: str
    generator_id: str
    generator_version: int


class GitResolver(Protocol):
    @property
    def head(self) -> str: ...

    @property
    def upstream(self) -> str: ...

    def contains(self, sha: str, upstream: str) -> bool: ...


class _SubprocessGit:
    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def _run(self, *args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=self.cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise NotebookSourceError(f"git {' '.join(args)} failed") from exc
        return completed.stdout.strip()

    @property
    def head(self) -> str:
        return self._run("rev-parse", "HEAD")

    @property
    def upstream(self) -> str:
        return self._run("rev-parse", "--abbrev-ref", "@{u}")

    def contains(self, sha: str, upstream: str) -> bool:
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", sha, upstream],
                cwd=self.cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return True
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 1:
                return False
            raise NotebookSourceError("git reachability check failed") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise NotebookSourceError("git reachability check failed") from exc


def _attribute_value(target: object, name: str) -> object:
    value = getattr(target, name)
    return value() if callable(value) else value


def _upstream_contains(git: object, sha: str, upstream: str) -> bool:
    if hasattr(git, "upstream_contains_head"):
        value = getattr(git, "upstream_contains_head")
        return bool(value(sha, upstream) if callable(value) else value)
    contains = getattr(git, "contains")
    return bool(contains(sha, upstream))


def resolve_notebook_source(git: GitResolver | object | None = None) -> NotebookSource:
    resolver = git or _SubprocessGit(Path(__file__).resolve().parents[2])
    configured = os.environ.get("STATIC_NOTEBOOK_COMMIT_SHA", "").strip()
    sha_value = configured or _attribute_value(resolver, "head")
    if not isinstance(sha_value, str) or _FULL_SHA.fullmatch(sha_value) is None:
        raise NotebookSourceError(
            "Notebook source requires a lowercase full 40-character commit SHA"
        )
    if configured:
        return NotebookSource(
            repo_url=REPOSITORY_URL,
            commit_sha=sha_value,
            generator_id=GENERATOR_ID,
            generator_version=GENERATOR_VERSION,
        )
    upstream_value = _attribute_value(resolver, "upstream")
    if not isinstance(upstream_value, str) or not upstream_value:
        raise NotebookSourceError("Notebook source requires a configured git upstream")
    if not _upstream_contains(resolver, sha_value, upstream_value):
        raise NotebookSourceError(
            "Notebook source is not remotely reachable; push the commit or configure "
            "STATIC_NOTEBOOK_COMMIT_SHA to a reachable release SHA"
        )
    return NotebookSource(
        repo_url=REPOSITORY_URL,
        commit_sha=sha_value,
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
    )

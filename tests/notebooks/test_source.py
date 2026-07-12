from __future__ import annotations

import pytest

from backend.notebooks.source import NotebookSourceError, resolve_notebook_source


class FakeGit:
    def __init__(self, *, head: str, contains: bool) -> None:
        self.head = head
        self.upstream = "origin/feature/static"
        self.upstream_contains_head = contains


def test_source_uses_full_head_only_when_upstream_contains_it(monkeypatch) -> None:
    monkeypatch.delenv("STATIC_NOTEBOOK_COMMIT_SHA", raising=False)
    source = resolve_notebook_source(git=FakeGit(head="b" * 40, contains=True))
    assert source.commit_sha == "b" * 40
    assert source.repo_url == (
        "https://github.com/mehmettahacumurcu/gaussian-splatter.git"
    )


def test_source_rejects_unpushed_head(monkeypatch) -> None:
    monkeypatch.delenv("STATIC_NOTEBOOK_COMMIT_SHA", raising=False)
    with pytest.raises(NotebookSourceError, match="push"):
        resolve_notebook_source(git=FakeGit(head="b" * 40, contains=False))


def test_source_rejects_non_full_or_uppercase_sha(monkeypatch) -> None:
    monkeypatch.delenv("STATIC_NOTEBOOK_COMMIT_SHA", raising=False)
    for value in ("main", "A" * 40, "a" * 39):
        with pytest.raises(NotebookSourceError, match="40-character"):
            resolve_notebook_source(git=FakeGit(head=value, contains=True))


def test_deployment_override_does_not_depend_on_local_git_state(monkeypatch) -> None:
    monkeypatch.setenv("STATIC_NOTEBOOK_COMMIT_SHA", "c" * 40)
    source = resolve_notebook_source(git=FakeGit(head="invalid", contains=False))
    assert source.commit_sha == "c" * 40

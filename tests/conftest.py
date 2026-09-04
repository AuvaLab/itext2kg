"""Shared pytest fixtures and path setup for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT

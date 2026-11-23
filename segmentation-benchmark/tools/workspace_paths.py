"""Relative paths for metadata produced by the photo masks workspace."""

import os
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def resolve_reference(value: str | Path) -> Path:
    """Resolve a metadata path against the directory that contains tools."""
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def record_path(path: str | Path) -> str:
    """Store paths relative to the workspace when both are on one volume."""
    absolute = Path(path).resolve()
    try:
        return Path(os.path.relpath(absolute, WORKSPACE_ROOT)).as_posix()
    except ValueError:
        return str(absolute)

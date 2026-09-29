"""Lineage metadata attached to reports and MLflow runs."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .config import PROJECT_ROOT


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lineage(data_path: Path) -> dict[str, str | bool | None]:
    """Code and data identity; `git_dirty` flags runs that the SHA alone cannot reproduce."""

    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "data_sha256": file_sha256(data_path),
    }


def mlflow_tags(values: dict[str, str | bool | None]) -> dict[str, str]:
    return {key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in values.items() if value is not None}

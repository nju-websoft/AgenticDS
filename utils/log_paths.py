from __future__ import annotations

from pathlib import Path


LOGS_DIR = Path("logs")


def logs_mirror_path(path: str | Path) -> Path:
    target = Path(path)
    if target.is_absolute():
        target = target.relative_to(Path.cwd())
    parts = list(target.parts)
    if parts and parts[0] == "output":
        parts = parts[1:]
    return LOGS_DIR.joinpath(*parts)

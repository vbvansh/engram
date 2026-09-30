"""Load a .env file into os.environ.

PowerShell has no `source .env`, so every fresh terminal starts without the
keys and the harness dies with "set ENGRAM_ADMIN_KEY". Rather than making the
operator hand-load the file each time, the entry points call `load_env_file()`
on startup.

Real environment variables always win — this only fills in what is missing, so
an explicitly exported value (or a CI secret) is never clobbered.
"""

from __future__ import annotations

import os
from pathlib import Path

# Searched in order, relative to the current working directory. The local
# canonical stack keeps its secrets in the repo-root .env.local.
_CANDIDATES = (".env.local", ".env", "../.env", "../../.env")


def find_env_file(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for c in _CANDIDATES:
        p = Path(c)
        if p.exists():
            return p
    return None


def load_env_file(explicit: str | None = None, *, override: bool = False) -> Path | None:
    """Populate os.environ from a .env file. Returns the file used, if any."""
    path = find_env_file(explicit)
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if not name:
            continue
        if override or name not in os.environ:
            os.environ[name] = value
    return path

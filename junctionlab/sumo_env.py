"""Locate and run the SUMO binaries shipped with the eclipse-sumo pip package."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path


class SumoNotFoundError(RuntimeError):
    pass


@cache
def sumo_home() -> Path:
    """SUMO_HOME of the pip package, falling back to the SUMO_HOME environment variable."""
    try:
        import sumo  # provided by eclipse-sumo
    except ImportError:
        sumo = None
    if sumo is not None and getattr(sumo, "SUMO_HOME", None):
        return Path(sumo.SUMO_HOME)
    if os.environ.get("SUMO_HOME"):
        return Path(os.environ["SUMO_HOME"])
    raise SumoNotFoundError("SUMO not found: install eclipse-sumo or set SUMO_HOME")


def binary(name: str) -> Path:
    """Path to a SUMO tool such as 'sumo' or 'netconvert'."""
    exe = name + (".exe" if sys.platform == "win32" else "")
    candidate = sumo_home() / "bin" / exe
    if candidate.is_file():
        return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise SumoNotFoundError(f"SUMO binary '{name}' not found in {candidate.parent} or on PATH")


def run(
    name: str, args: list[str], cwd: Path | str | None = None, timeout: float = 600
) -> subprocess.CompletedProcess[str]:
    """Run a SUMO tool headless. Raises CalledProcessError with SUMO's output on failure."""
    env = {**os.environ, "SUMO_HOME": str(sumo_home())}
    return subprocess.run(
        [str(binary(name)), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def version() -> str:
    """First line of `sumo --version`, e.g. 'Eclipse SUMO sumo Version 1.28.0'."""
    return run("sumo", ["--version"]).stdout.splitlines()[0].strip()

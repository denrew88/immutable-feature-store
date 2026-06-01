"""Helpers for using sibling packages from a source checkout.

The repository keeps some public packages under ``packages/*/src`` while the
legacy ``fs`` namespace still lives under ``python/fs``.  When users run code
directly from a checkout, those packages are not necessarily installed as
wheels.  This helper makes the local package source importable before wrapper
modules import it.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_package_source(package_dir_name: str) -> None:
    """Put ``packages/<package_dir_name>/src`` on ``sys.path`` when it exists."""

    repo_root = Path(__file__).resolve().parents[2]
    package_src = repo_root / "packages" / package_dir_name / "src"
    if not package_src.is_dir():
        return

    package_src_str = str(package_src)
    if package_src_str in sys.path:
        return
    sys.path.insert(0, package_src_str)

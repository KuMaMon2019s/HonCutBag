#!/usr/bin/env python3
"""Fail when a project command escapes HonCut's locked uv interpreter."""

from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIN_FILE = PROJECT_ROOT / ".python-version"


def _expected_environment() -> Path:
    configured = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if configured:
        path = Path(configured)
        return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    return (PROJECT_ROOT / ".venv").resolve()


def _fail(message: str) -> None:
    raise SystemExit(f"HonCut Python environment check failed: {message}")


def main() -> None:
    expected_environment = _expected_environment()
    executable = Path(sys.executable).absolute()
    prefix = Path(sys.prefix).resolve()
    pinned = PIN_FILE.read_text(encoding="utf-8").strip()
    pinned_version = tuple(int(part) for part in pinned.split(".")[:3])

    if prefix != expected_environment:
        _fail(
            f"expected sys.prefix={expected_environment}, got {prefix}; "
            "run through `make test` or `uv run --locked python -m ...`"
        )
    if sys.version_info[:3] != pinned_version:
        _fail(
            f"expected Python {pinned}, got "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

    missing = []
    versions = {}
    for distribution in ("honcut-pipeline", "openai", "pytest"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        _fail("missing locked distributions: " + ", ".join(missing))

    print(f"HonCut environment: {prefix}")
    print(f"HonCut interpreter: {executable}")
    print(f"Python: {pinned}")
    print(
        "Locked distributions: "
        + ", ".join(f"{name}={version}" for name, version in versions.items())
    )


if __name__ == "__main__":
    main()

"""Startup dependency checks for the HonCut pipeline."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable


REQUIRED_MODULES = ("openai", "langgraph", "langchain_core")


def check_dependencies(required: Iterable[str] = REQUIRED_MODULES) -> None:
    """Raise an actionable error when required pipeline modules are missing."""
    missing: list[str] = []
    for module in required:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)

    if missing:
        install_command = f'"{sys.executable}" -m pip install -r requirements.txt'
        raise ImportError(
            "Missing pipeline dependencies: "
            f"{', '.join(missing)}\n"
            f"Python interpreter: {sys.executable}\n"
            f"Install them with: {install_command}"
        )

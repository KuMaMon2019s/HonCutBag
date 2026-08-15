"""Startup dependency checks for the HonCut pipeline."""

from __future__ import annotations

import importlib
import shutil
import sys
from collections.abc import Iterable

REQUIRED_MODULES = (
    "openai",
    "requests",
    "yaml",
    "ffmpeg",
    "pydantic",
    "pydantic_settings",
    "numpy",
    "PIL",
    "dotenv",
    "langgraph",
    "langgraph.checkpoint.sqlite",
    "langchain_core",
    "websockets",
    "arq",
    "scenedetect",
    "cv2",
)
REQUIRED_COMMANDS = ("ffmpeg", "ffprobe")


def check_dependencies(
    required: Iterable[str] = REQUIRED_MODULES,
    required_commands: Iterable[str] = REQUIRED_COMMANDS,
) -> None:
    """Raise an actionable error when required pipeline modules are missing."""
    missing: list[str] = []
    for module in required:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)

    missing_commands = [command for command in required_commands if shutil.which(command) is None]
    if missing or missing_commands:
        install_command = f'uv pip install --python "{sys.executable}" -e .'
        missing_details = []
        if missing:
            missing_details.append("Python modules: " + ", ".join(missing))
        if missing_commands:
            missing_details.append("system commands: " + ", ".join(missing_commands))
        raise ImportError(
            "Missing pipeline dependencies: " + "; ".join(missing_details) + "\n"
            f"Python interpreter: {sys.executable}\n"
            f"Install Python packages with: {install_command}\n"
            "Install FFmpeg commands on macOS with: brew install ffmpeg"
        )

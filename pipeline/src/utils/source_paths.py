"""Canonical repository paths used by pipeline runtime adapters."""

from pathlib import Path


PIPELINE_SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PIPELINE_SRC_DIR.parent.parent
OM_TOOLS_DIR = PROJECT_ROOT / "vendor" / "video_tools"


__all__ = [
    "OM_TOOLS_DIR",
    "PIPELINE_SRC_DIR",
    "PROJECT_ROOT",
]

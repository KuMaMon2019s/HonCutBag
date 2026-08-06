#!/usr/bin/env python3
"""HonCut pipeline CLI and backward-compatible orchestration facade.

Phase entry points live in :mod:`phases`; the implementation core is kept as
one module so the existing LangGraph nodes and monkeypatch-based integrations
continue to share a single module namespace.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.deps import check_dependencies

check_dependencies()

from phases import pipeline_core as _core

# Preserve the historical ``import pipeline_runner`` module identity. This is
# important to callers that patch phase functions before invoking run_pipeline.
sys.modules[__name__] = _core

if __name__ == "__main__":
    _core.main()

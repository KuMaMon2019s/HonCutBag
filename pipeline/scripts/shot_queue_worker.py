#!/usr/bin/env python3
"""Run the HonCut storyboard shot worker as an independent process."""

import sys
from pathlib import Path

from arq import run_worker

# Direct script execution puts pipeline/scripts (not the repository root) on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SRC = REPO_ROOT / "pipeline" / "src"
for import_root in (REPO_ROOT, PIPELINE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from pipeline.src.utils.shot_queue import WorkerSettings


if __name__ == "__main__":
    run_worker(WorkerSettings)

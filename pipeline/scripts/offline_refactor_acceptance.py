#!/usr/bin/env python3
"""Run the no-provider dry-run/resume release matrix."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime.pipeline_execution import run_pipeline


ALL_PHASES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5]


def run_matrix(
    *,
    rounds: int = 10,
    root: str | Path | None = None,
    verbose: bool = False,
) -> dict:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    temporary = None
    if root is None:
        temporary = tempfile.TemporaryDirectory(prefix="honcut-refactor-acceptance-")
        matrix_root = Path(temporary.name)
    else:
        matrix_root = Path(root).resolve()
        matrix_root.mkdir(parents=True, exist_ok=True)
    receipts = []
    try:
        for index in range(1, rounds + 1):
            output_dir = matrix_root / f"round-{index:02d}"
            common = {
                "text": "Offline refactor acceptance fixture.",
                "duration": 12,
                "shot_duration": 4,
                "dry_run": True,
                "skip_phase": ALL_PHASES,
                "output_dir": str(output_dir),
                "project_id": "refactor-acceptance",
                "enable_reshoot": False,
            }
            if verbose:
                initial = run_pipeline(**common)
                resumed = run_pipeline(**common, resume=True)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    initial = run_pipeline(**common)
                    resumed = run_pipeline(**common, resume=True)
            if initial.get("status") != "completed" or resumed.get("status") != (
                "completed"
            ):
                raise RuntimeError(
                    f"offline acceptance round {index} failed: "
                    f"initial={initial.get('status')}, resumed={resumed.get('status')}"
                )
            manifest = json.loads(
                (output_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8")
            )
            receipts.append(
                {
                    "round": index,
                    "initial": initial["status"],
                    "resume": resumed["status"],
                    "run_id": manifest["run_fingerprint"],
                    "project_id": manifest["resolved_config"]["project_id"],
                    "provider_requests": 0,
                }
            )
        return {"status": "passed", "rounds": rounds, "receipts": receipts}
    finally:
        if temporary is not None:
            temporary.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HonCut's fully offline dry-run/resume release matrix"
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--root")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = run_matrix(rounds=args.rounds, root=args.root, verbose=args.verbose)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

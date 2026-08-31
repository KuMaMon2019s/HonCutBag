"""Phase 1 director planning entry point."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from prompt.speech_pacing import annotate_shot_pacing
from quality.delivery_promise import classify_from_brief
from runtime.phase_timing import _banner, _elapsed, _now


class DirectorPlanningError(RuntimeError):
    """Director evidence is missing or invalid, so Phase 1 must stop."""


def _dialogue_by_sequence(events: list[dict[str, Any]]) -> dict[str, str]:
    dialogue: dict[str, list[str]] = defaultdict(list)
    for event in events:
        sequence_id = str(event.get("sequence_id") or "").strip()
        if not sequence_id:
            continue
        raw_lines = event.get("lines") or []
        if isinstance(raw_lines, dict):
            raw_lines = [raw_lines]
        if isinstance(raw_lines, str):
            raw_lines = [raw_lines]
        for raw_line in raw_lines if isinstance(raw_lines, list) else []:
            line = (
                raw_line.get("line")
                if isinstance(raw_line, dict)
                else raw_line
            )
            text = str(line or "").strip()
            if text:
                dialogue[sequence_id].append(text)
    return {
        sequence_id: "\n".join(lines)
        for sequence_id, lines in dialogue.items()
    }


def run_phase1_director(
    events: list[dict[str, Any]],
    output_dir: Path,
    dry_run: bool,
) -> dict:
    """Plan sequence-level intent after Event Extractor has run."""
    _banner("1", 9, "导演规划 (Director Planner)", dry_run)
    start = _now()
    plan_path = Path(output_dir) / "director_plan.json"
    reconciliation_path = (
        Path(output_dir) / "director_plan_reconciliation.json"
    )
    try:
        from phases.phase1.director_planner import plan_director
        result = plan_director(events, output_dir, dry_run)
        status = result.get("status")
        if status != "done" and not (dry_run and status == "skipped"):
            detail = result.get("error") or result.get("reason") or "missing success evidence"
            raise RuntimeError(f"director planning returned {status}: {detail}")
        # Lock the intended production medium before providers can downgrade it.
        delivery_promise = classify_from_brief("cinematic", {}).to_dict()
        result.setdefault("delivery_promise", delivery_promise)
        plan = result.get("plan")
        if isinstance(plan, dict):
            plan.setdefault("delivery_promise", delivery_promise)
            sequences = plan.get("sequences", [])
            dialogue = _dialogue_by_sequence(events)
            pacing_inputs = [
                {
                    "dialogue": dialogue.get(sequence.get("sequence_id"), ""),
                    "emotion": sequence.get("emotion_arc", ""),
                }
                for sequence in sequences
            ]
            pacing = annotate_shot_pacing(pacing_inputs)
            for sequence, annotation in zip(sequences, pacing):
                sequence["speech_pacing"] = {
                    "duration_s": annotation["speech_duration_s"],
                    "emotion": annotation["emotion"],
                }
            result_path = Path(result.get("output", plan_path))
            result_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        result["duration_s"] = _elapsed(start)
        return result
    except Exception as exc:
        plan_path.unlink(missing_ok=True)
        reconciliation_path.unlink(missing_ok=True)
        if isinstance(exc, DirectorPlanningError):
            raise
        raise DirectorPlanningError(str(exc)) from exc

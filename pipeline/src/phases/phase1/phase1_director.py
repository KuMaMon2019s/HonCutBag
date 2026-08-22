"""Phase 1 director planning entry point."""

from __future__ import annotations

import json
from pathlib import Path

from prompt.speech_pacing import annotate_shot_pacing
from quality.delivery_promise import classify_from_brief
from runtime.phase_timing import _banner, _elapsed, _now


def run_phase1_director(text: str, output_dir: Path, dry_run: bool) -> dict:
    """Phase 1: 导演规划（M1 增量模块）"""
    _banner("1", 9, "导演规划 (Director Planner)", dry_run)
    start = _now()
    plan_path = Path(output_dir) / "director_plan.json"
    try:
        from phases.phase1.director_planner import plan_director
        result = plan_director(text, output_dir, dry_run)
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
            scenes = plan.get("scenes", [])
            pacing_inputs = []
            for scene in scenes:
                dialogue = scene.get("dialogue") or scene.get("lines")
                if not dialogue and scene.get("dialogue_words"):
                    dialogue = "字" * int(scene["dialogue_words"])
                pacing_inputs.append({
                    "dialogue": dialogue or "",
                    "emotion": scene.get("emotion_arc", ""),
                })
            pacing = annotate_shot_pacing(pacing_inputs)
            for scene, annotation in zip(scenes, pacing):
                scene["speech_pacing"] = {
                    "duration_s": annotation["speech_duration_s"],
                    "emotion": annotation["emotion"],
                }
            result_path = Path(result.get("output", plan_path))
            result_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        result["duration_s"] = _elapsed(start)
        return result
    except Exception:
        plan_path.unlink(missing_ok=True)
        raise

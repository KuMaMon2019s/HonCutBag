"""Phase 7.3 duration measurement and reshoot planning."""

from __future__ import annotations

import json
from pathlib import Path

from .frame_analysis import probe_duration


def build_reshoot_list(shots_dir: Path, required_gap_s: float, round_number: int) -> dict:
    deficits: list[dict] = []
    for shot_dir in sorted(Path(shots_dir).iterdir()) if Path(shots_dir).is_dir() else []:
        video = shot_dir / "output.mp4"
        meta_path = shot_dir / "SHOT_META.json"
        if not video.is_file() or not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            requested = float(meta.get("duration") or meta.get("requested_duration") or 0)
            actual = probe_duration(video)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"  ⚠ [7.3] 无法评估 {shot_dir.name} 时长短板: {exc}", flush=True)
            continue
        gap = max(0.0, requested - actual)
        if gap > 0:
            deficits.append({
                "shot_id": shot_dir.name,
                "requested_s": round(requested, 3),
                "actual_s": round(actual, 3),
                "gap_s": round(gap, 3),
            })
    deficits.sort(key=lambda item: item["gap_s"], reverse=True)
    selected: list[dict] = []
    covered = 0.0
    for item in deficits:
        selected.append(item)
        covered += item["gap_s"]
        if covered >= required_gap_s:
            break
    return {"shots": selected, "round": round_number}


def evaluate_duration_gate(
    output_dir: Path,
    target_duration: float | None,
    round_number: int = 0,
    reshoots: list[dict] | None = None,
) -> tuple[dict, dict | None]:
    output_dir = Path(output_dir)
    actual = probe_duration(output_dir / "raw_assembly.mp4")
    history = list(reshoots or [])
    if target_duration is None:
        gate = {
            "target_s": None,
            "actual_s": round(actual, 3),
            "gap_s": None,
            "passed": True,
            "reshoots": history,
            "skipped_reason": "target_duration is None",
        }
        reshoot_plan = None
    else:
        target = float(target_duration)
        gap = max(0.0, target - actual)
        passed = actual >= target * 0.9
        gate = {
            "target_s": round(target, 3),
            "actual_s": round(actual, 3),
            "gap_s": round(gap, 3),
            "passed": passed,
            "reshoots": history,
        }
        reshoot_plan = None if passed else build_reshoot_list(output_dir / "shots", gap, round_number + 1)
        if reshoot_plan is not None:
            (output_dir / "reshoot_list.json").write_text(
                json.dumps(reshoot_plan, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    (output_dir / "duration_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return gate, reshoot_plan

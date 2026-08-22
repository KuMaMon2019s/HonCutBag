"""Phase 7 handoff into pixel-level Phase 8 QA."""

from __future__ import annotations

import json
from pathlib import Path

from runtime.phase_timing import _banner, _elapsed, _now


def run_phase7(output_dir: Path, dry_run: bool, storyboard_data: dict = None) -> dict:
    """Validate the pre-paid QA receipt before Phase 8 reviews real pixels.

    Prompt/metadata checks live in Phase 5. Phase 8 is the single owner of
    per-shot video sampling, reference-image identity review, and recoverable
    selective reshoots. Keeping that decision in one place prevents a text-
    only check from masquerading as generated-video consistency QA.
    """
    _banner(7, 9, "视频质检交接 (Phase 8 owns pixel QA)", dry_run)
    start = _now()
    output_dir = Path(output_dir)

    if dry_run:
        print("  ⊘ dry-run 模式，跳过视频质检交接")
        return {"status": "skipped", "reason": "dry-run", "duration_s": _elapsed(start)}

    try:
        gate = json.loads(
            (output_dir / "storyboard_qa_report.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "error",
            "error": f"Phase 7 requires the Phase 5 QA receipt: {exc}",
            "duration_s": _elapsed(start),
        }
    if gate.get("gate_passed") is not True:
        return {
            "status": "error",
            "error": "Phase 7 refused a non-passing Phase 5 QA receipt",
            "duration_s": _elapsed(start),
        }

    print("  ✓ Phase 7 交接完成；逐镜头像素/VLM 检查由 Phase 8 执行")
    return {
        "status": "done",
        "duration_s": _elapsed(start),
        "outputs": [
            name
            for name in (
                "storyboard_qa_report.json",
                "variation_report.json",
                "slideshow_risk_report.json",
            )
            if (output_dir / name).is_file()
        ],
        "variation_score": float(gate.get("variation_score", 5.0)),
        "slideshow_risk": float(gate.get("slideshow_risk", 0.0)),
        "video_quality_owner": "phase8",
    }

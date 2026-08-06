"""Score six structural signals that make a video feel like a slideshow.

Adapted from OpenMontage ``lib/slideshow_risk.py`` for HonCut's shot lists.
Lower scores are better.
"""

from collections import Counter
from typing import Any


def score_slideshow_risk(
    scenes: list[dict[str, Any]],
    edit_decisions: dict[str, Any] | None = None,
    renderer_family: str | None = None,
    render_runtime: str | None = None,
) -> dict[str, Any]:
    if not scenes:
        return {"average": 5.0, "verdict": "fail", "dimensions": {}, "render_runtime": render_runtime}
    dimensions = {
        "repetition": _score_repetition(scenes),
        "decorative_visuals": _score_decorative(scenes),
        "weak_motion": _score_weak_motion(scenes),
        "weak_shot_intent": _score_weak_intent(scenes),
        "typography_overreliance": _score_typography(scenes),
        "unsupported_cinematic_claims": _score_cinematic_claims(scenes, renderer_family),
    }
    average = sum(item["score"] for item in dimensions.values()) / len(dimensions)
    verdict = "strong" if average < 2 else "acceptable" if average < 3 else "revise" if average < 4 else "fail"
    return {"average": round(average, 2), "verdict": verdict, "dimensions": dimensions, "render_runtime": render_runtime}


def _score_repetition(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    if len(scenes) < 3:
        return {"score": 0.0, "reason": "Too few scenes to assess repetition"}
    types = Counter(s.get("type", "unknown") for s in scenes)
    common_type, common_count = types.most_common(1)[0]
    descriptions = [s.get("description", "").lower()[:50] for s in scenes]
    sizes = [s.get("shot_language", {}).get("shot_size", "none") for s in scenes]
    score, reasons = 0.0, []
    if common_count / len(scenes) > 0.7:
        score += 2.0; reasons.append(f"Scene type '{common_type}' dominates at {common_count / len(scenes):.0%}")
    if len(set(descriptions)) / len(descriptions) < 0.6:
        score += 1.5; reasons.append("Descriptions repeat")
    size_ratio = Counter(sizes).most_common(1)[0][1] / len(scenes)
    if size_ratio > 0.6:
        score += 1.5; reasons.append(f"Same shot size in {size_ratio:.0%} of scenes")
    return {"score": min(5.0, score), "reason": "; ".join(reasons) or "Good variety"}


def _score_decorative(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(not any(s.get(k) for k in ("information_role", "narrative_role", "shot_intent")) for s in scenes)
    ratio = count / len(scenes)
    reason = f"{count}/{len(scenes)} scenes have no stated purpose" if ratio > .5 else f"{count}/{len(scenes)} scenes lack stated purpose" if ratio > .2 else "Most scenes have clear communicative purpose"
    return {"score": round(min(5.0, ratio * 5), 1), "reason": reason}


def _score_weak_motion(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    moving = [s for s in scenes if s.get("shot_language", {}).get("camera_movement", "static") not in ("static", "unspecified", None)]
    if not moving:
        return {"score": 1.5, "reason": "No camera movement defined (may be intentional for static style)"}
    purposeless = sum(not s.get("shot_intent") for s in moving)
    ratio = purposeless / len(moving)
    return {"score": round(min(5.0, ratio * 4), 1), "reason": f"{purposeless}/{len(moving)} moving shots lack shot_intent" if ratio > .5 else "Camera movement appears purposeful"}


def _score_weak_intent(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(bool(s.get("shot_intent")) for s in scenes)
    ratio = count / len(scenes)
    reason = f"Only {count}/{len(scenes)} scenes have shot_intent — most shots lack purpose" if ratio < .3 else f"{count}/{len(scenes)} scenes have shot_intent" if ratio < .6 else "Strong shot intent coverage"
    return {"score": round((1 - ratio) * 5, 1), "reason": reason}


def _score_typography(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(s.get("type") in ("text_card", "stat_card", "kpi_grid") for s in scenes)
    ratio = count / len(scenes)
    if ratio > .6: return {"score": 4.0, "reason": f"{count}/{len(scenes)} scenes are text/stat cards — video feels like animated slides"}
    if ratio > .4: return {"score": 2.5, "reason": f"{count}/{len(scenes)} scenes are text-based — consider balancing with visual scenes"}
    if ratio > .2: return {"score": 1.0, "reason": "Balanced text and visual content"}
    return {"score": 0.0, "reason": "Visual-first approach"}


def _score_cinematic_claims(scenes: list[dict[str, Any]], renderer_family: str | None) -> dict[str, Any]:
    if not renderer_family or "cinematic" not in renderer_family.lower():
        return {"score": 0.0, "reason": "Not claiming cinematic treatment"}
    issues = []
    if not any(s.get("hero_moment") for s in scenes): issues.append("Claims cinematic but has no hero_moment defined")
    moving = sum(s.get("shot_language", {}).get("camera_movement", "static") != "static" for s in scenes)
    if moving < len(scenes) * .3: issues.append(f"Claims cinematic but only {moving}/{len(scenes)} scenes have camera movement")
    lit = sum(bool(s.get("shot_language", {}).get("lighting_key")) for s in scenes)
    if lit < len(scenes) * .3: issues.append(f"Claims cinematic but only {lit}/{len(scenes)} scenes define lighting")
    return {"score": round(min(5.0, len(issues) * 1.8), 1), "reason": "; ".join(issues) or "Cinematic claims supported by structure"}

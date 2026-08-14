"""Conservative shot-boundary classification shared by Phase 1 and Phase 4."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CONTINUOUS_VALUES = {"continuous", "continue", "extend", "native_extend", "连续", "延长"}
CUT_VALUES = {"cut", "scene_cut", "fresh", "new_scene", "硬切", "换场"}
_CONTINUATION_CUES = (
    "承接上镜",
    "本镜由此延续",
    "继续",
    "延续",
    "紧接",
    "direct continuation",
    "continues from",
)
_TIME_JUMP_CUES = (
    "与此同时",
    "另一边",
    "后来",
    "次日",
    "翌日",
    "多年后",
    "几小时后",
    "回忆",
    "梦境",
    "meanwhile",
    "later",
    "next day",
    "years later",
    "flashback",
)
_SCENE_TRANSITIONS = {"dissolve", "fade", "fade_to_black", "wipe", "match_cut"}


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _subjects(shot: Mapping[str, Any]) -> set[str]:
    raw = shot.get("who") or shot.get("characters") or []
    if not isinstance(raw, (list, tuple, set)):
        raw = [raw]
    subjects = {_text(value) for value in raw if _text(value)}
    if subjects:
        return subjects
    fallback = _text(
        shot.get("continuity_subject")
        or shot.get("tracking_prompt")
        or shot.get("subject_description")
    )
    return {fallback} if fallback else set()


def classify_boundary(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    *,
    index: int,
) -> tuple[str, str]:
    """Return ``(cut|continuous, reason)`` without guessing across weak evidence."""
    if index <= 1 or previous is None:
        return "cut", "first shot starts a new generation group"

    explicit = _text(current.get("boundary_before") or current.get("continuity_boundary"))
    if explicit in CUT_VALUES:
        return "cut", _text(current.get("continuity_reason")) or "storyboard explicitly starts fresh"
    if explicit in CONTINUOUS_VALUES:
        return (
            "continuous",
            _text(current.get("continuity_reason"))
            or "storyboard explicitly continues the previous moving state",
        )

    previous_transition = _text(previous.get("transition_to_next"))
    if previous_transition in _SCENE_TRANSITIONS:
        return "cut", f"previous shot requests a scene transition ({previous_transition})"

    previous_place = _text(previous.get("where") or previous.get("scene"))
    current_place = _text(current.get("where") or current.get("scene"))
    if not previous_place or not current_place or previous_place != current_place:
        return "cut", "location is missing or changes across the boundary"

    combined = " ".join(
        _text(current.get(field)) for field in ("visual", "what", "action_description")
    )
    if any(cue in combined for cue in _TIME_JUMP_CUES):
        return "cut", "the next shot contains a temporal or narrative jump"

    previous_subjects = _subjects(previous)
    current_subjects = _subjects(current)
    if not previous_subjects or not current_subjects or previous_subjects.isdisjoint(current_subjects):
        return "cut", "no stable subject is shared across the boundary"

    if not any(cue in combined for cue in _CONTINUATION_CUES):
        return "cut", "no explicit action-state continuation cue is present"

    return "continuous", "same place and subject with an explicit action-state continuation cue"


def annotate_boundaries(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill normalized boundary metadata while preserving explicit LLM decisions."""
    previous: dict[str, Any] | None = None
    for index, shot in enumerate(shots, 1):
        boundary, reason = classify_boundary(previous, shot, index=index)
        shot["boundary_before"] = boundary
        if not str(shot.get("continuity_reason") or "").strip():
            shot["continuity_reason"] = reason
        if boundary == "continuous" and not str(shot.get("continuity_subject") or "").strip():
            shared = sorted(_subjects(previous or {}) & _subjects(shot))
            if shared:
                shot["continuity_subject"] = ", ".join(shared)
        previous = shot
    return shots


__all__ = ["annotate_boundaries", "classify_boundary"]

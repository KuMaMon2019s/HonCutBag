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
_SCENE_TRANSITIONS = {
    "cross_dissolve",
    "crossfade",
    "dissolve",
    "fade",
    "fade_in",
    "fade_out",
    "fade_to_black",
    "flash_cut",
    "iris",
    "match_cut",
    "scene_transition",
    "smash_cut",
    "wipe",
    "叠化",
    "淡入",
    "淡出",
    "淡黑",
    "闪切",
    "划变",
    "转场",
}


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _transition(value: Any) -> str:
    return _text(value).replace("-", "_").replace(" ", "_")


def _is_scene_transition(value: Any) -> bool:
    normalized = _transition(value)
    if normalized in _SCENE_TRANSITIONS:
        return True
    return any(
        marker in normalized
        for marker in ("dissolve", "crossfade", "fade_", "wipe", "iris", "转场", "叠化")
    )


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
    allow_unverified_explicit: bool = False,
) -> tuple[str, str]:
    """Return ``(cut|continuous, reason)`` without guessing across weak evidence."""
    if index <= 1 or previous is None:
        return "cut", "first shot starts a new generation group"

    explicit = _text(current.get("boundary_before") or current.get("continuity_boundary"))
    if explicit in CUT_VALUES:
        return "cut", _text(current.get("continuity_reason")) or "storyboard explicitly starts fresh"

    # A declared scene transition is a hard boundary even when the next shot
    # was accidentally labelled continuous by an upstream model.  Bridge-video
    # planning must never interpolate across dissolves, fades, wipes or jumps.
    previous_transition = _transition(previous.get("transition_to_next"))
    if _is_scene_transition(previous_transition):
        return "cut", f"previous shot requests a scene transition ({previous_transition})"

    previous_place = _text(previous.get("where") or previous.get("scene"))
    current_place = _text(current.get("where") or current.get("scene"))
    if previous_place and current_place and previous_place != current_place:
        return "cut", "location changes across the boundary"

    combined = " ".join(
        _text(current.get(field))
        for field in ("visual", "what", "action_description", "continuity_reason")
    )
    if any(cue in combined for cue in _TIME_JUMP_CUES):
        return "cut", "the next shot contains a temporal or narrative jump"

    previous_subjects = _subjects(previous)
    current_subjects = _subjects(current)
    if previous_subjects and current_subjects and previous_subjects.isdisjoint(current_subjects):
        return "cut", "no stable subject is shared across the boundary"

    previous_sequence_values = previous.get("source_sequence_ids") or []
    current_sequence_values = current.get("source_sequence_ids") or []
    if not isinstance(previous_sequence_values, (list, tuple, set)):
        previous_sequence_values = [previous_sequence_values]
    if not isinstance(current_sequence_values, (list, tuple, set)):
        current_sequence_values = [current_sequence_values]
    previous_sequences = {
        _text(value) for value in previous_sequence_values if _text(value)
    }
    current_sequences = {
        _text(value) for value in current_sequence_values if _text(value)
    }
    if previous_sequences and current_sequences and previous_sequences.isdisjoint(current_sequences):
        return "cut", "screenplay sequence changes across the boundary"

    same_place = bool(previous_place and current_place and previous_place == current_place)
    shared_subject = bool(previous_subjects and current_subjects and previous_subjects & current_subjects)
    same_sequence = bool(previous_sequences and current_sequences and previous_sequences & current_sequences)
    previous_end = _text(previous.get("end_state"))
    current_start = _text(current.get("start_state") or current.get("prev_shot_context"))
    same_state = bool(previous_end and current_start and previous_end == current_start)
    has_continuation_cue = any(cue in combined for cue in _CONTINUATION_CUES)

    if explicit in CONTINUOUS_VALUES:
        if not (
            (same_place and shared_subject)
            or same_sequence
            or same_state
            or (shared_subject and has_continuation_cue)
        ):
            if allow_unverified_explicit:
                return (
                    "continuous",
                    _text(current.get("continuity_reason"))
                    or "legacy storyboard explicitly continues the previous shot",
                )
            return "cut", "continuous label lacks matching place/subject/sequence/state evidence"
        return (
            "continuous",
            _text(current.get("continuity_reason"))
            or "storyboard explicitly continues a verified moving state",
        )

    if not previous_place or not current_place:
        return "cut", "location evidence is missing across the boundary"
    if not previous_subjects or not current_subjects:
        return "cut", "subject evidence is missing across the boundary"

    if not has_continuation_cue:
        return "cut", "no explicit action-state continuation cue is present"

    return "continuous", "same place and subject with an explicit action-state continuation cue"


def annotate_boundaries(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill normalized boundary metadata while preserving explicit LLM decisions."""
    previous: dict[str, Any] | None = None
    for index, shot in enumerate(shots, 1):
        boundary, reason = classify_boundary(previous, shot, index=index)
        shot["boundary_before"] = boundary
        # Keep the explanation consistent with the normalized decision; stale
        # model prose must not claim continuity after code has forced a cut.
        shot["continuity_reason"] = reason
        if boundary == "continuous" and not str(shot.get("continuity_subject") or "").strip():
            shared = sorted(_subjects(previous or {}) & _subjects(shot))
            if shared:
                shot["continuity_subject"] = ", ".join(shared)
        previous = shot
    return shots


__all__ = ["annotate_boundaries", "classify_boundary"]

"""Strict contracts for model-produced text and visual understanding.

JSON syntax alone is not a production boundary.  These models are used both
to generate a native Provider JSON Schema and to validate the returned object
before any value can drive identity, ordering, QA, or retry decisions.
"""

from __future__ import annotations

import json
import re
from typing import Literal, TypeVar

from json_repair import repair_json
from pydantic import BaseModel, ConfigDict, Field


class StrictUnderstandingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpokenLineUnderstanding(StrictUnderstandingModel):
    speaker: str
    line: str
    confidence: float = Field(ge=0, le=1)
    evidence: str


class BodyActionUnderstanding(StrictUnderstandingModel):
    micro_action_index: int = Field(ge=1)
    performer: str
    technique: str
    side: str
    limbs: list[str]
    footwork: str
    torso: str
    weight_shift: str
    direction: str
    contact: str
    end_pose: str


class EventUnderstanding(StrictUnderstandingModel):
    who: list[str]
    where: str
    what: str
    emotion: str
    visual: str
    time: str
    action_type: str
    event_role: Literal[
        "scene_setup",
        "character_state",
        "dialogue",
        "action_chain",
        "reaction",
        "consequence",
        "turning_point",
        "transition",
    ] = "scene_setup"
    source_excerpt: str = ""
    micro_actions: list[str] = Field(default_factory=list)
    body_action_choreography: list[BodyActionUnderstanding] = Field(
        default_factory=list
    )
    generation_motion_mode: Literal["none", "atomic", "composite"] = "none"
    action_phase: Literal[
        "none",
        "setup",
        "attack",
        "counter",
        "impact",
        "recovery",
        "consequence",
    ] = "none"
    start_state: str = ""
    end_state: str = ""
    causal_link: str = ""
    continuity_before: Literal["cut", "continuous"] = "cut"
    continuity_subject: str = ""
    dramatic_turn: bool = False
    lines: list[SpokenLineUnderstanding] = Field(default_factory=list)


class EventUnderstandingBatch(StrictUnderstandingModel):
    events: list[EventUnderstanding]


class DirectorSequenceIntentUnderstanding(StrictUnderstandingModel):
    sequence_id: str
    scene_goal: str
    emotion_arc: str
    visual_focus: str
    spatial_intent: str
    transition_intent: str


class DirectorPlanUnderstanding(StrictUnderstandingModel):
    director_schema: Literal["honcut.director-plan.v1"] = Field(
        alias="schema",
        serialization_alias="schema",
    )
    sequences: list[DirectorSequenceIntentUnderstanding]


class DurationScaledActionSelectionUnderstanding(StrictUnderstandingModel):
    source_event_id: int = Field(ge=1)
    selected_source_generation_unit_indexes: list[int]
    narrative_purpose: str
    emotional_beat: str
    director_alignment: str


class DurationScaledActionSelectionBatch(StrictUnderstandingModel):
    selection_schema: Literal[
        "honcut.duration-scaled-action-selection.v1"
    ] = Field(alias="schema", serialization_alias="schema")
    events: list[DurationScaledActionSelectionUnderstanding]


class CharacterVariantUnderstanding(StrictUnderstandingModel):
    state_name: str
    description: str


class IdentityPropUnderstanding(StrictUnderstandingModel):
    id: str
    name: str
    description: str
    attachment_mode: Literal["body_attached", "isolated_handheld"]
    persistence: Literal["always", "role_active"]
    reference_required: bool


class CharacterAppearanceUnderstanding(StrictUnderstandingModel):
    gender: Literal["male", "female", "nonbinary", "unknown"]
    age_range: str
    height: str = ""
    build: str = ""
    hair: str
    face: str
    clothing: str
    interaction_props: list[str] = Field(default_factory=list)
    identity_props: list[IdentityPropUnderstanding] = Field(default_factory=list)
    distinguishing: str = ""
    summary: str
    variants: list[CharacterVariantUnderstanding] = Field(default_factory=list)


class CharacterPersonalityUnderstanding(StrictUnderstandingModel):
    traits: list[str] = Field(default_factory=list)
    speech_style: str = ""
    motivation: str = ""


class CharacterRelationshipUnderstanding(StrictUnderstandingModel):
    target_id: str
    type: str
    description: str


class CharacterUnderstanding(StrictUnderstandingModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    role: Literal["protagonist", "antagonist", "supporting", "extra"]
    appearance: CharacterAppearanceUnderstanding
    personality: CharacterPersonalityUnderstanding = Field(
        default_factory=CharacterPersonalityUnderstanding
    )
    style: str = ""
    negative: str = ""
    size: str = "2K"
    first_appearance: int = Field(default=0, ge=0)
    appearance_count: int = Field(default=0, ge=0)
    relationships: list[CharacterRelationshipUnderstanding] = Field(
        default_factory=list
    )


class CharacterUnderstandingBatch(StrictUnderstandingModel):
    characters: list[CharacterUnderstanding]


class StoryboardPromptUnderstanding(StrictUnderstandingModel):
    prompt: str
    caption: str


class SemanticMachineSemantics(StrictUnderstandingModel):
    entity_type: Literal["character"] = "character"
    gender: Literal["male", "female", "nonbinary", "unknown"] = "unknown"
    role: Literal[
        "protagonist",
        "antagonist",
        "supporting",
        "extra",
        "unknown",
    ] = "unknown"


class SemanticEntityRecord(StrictUnderstandingModel):
    character_id: str
    display_name: str
    source_identity_ref_ids: list[str]
    machine_semantics: SemanticMachineSemantics


class SemanticSourceMentionRecord(StrictUnderstandingModel):
    ref_id: str
    text: str
    language: Literal["zh", "en", "mixed", "und"]
    character_id: str


class SemanticEventRecord(StrictUnderstandingModel):
    event_id: int = Field(ge=1)
    action_unit_id: str
    participant_ref_ids: list[str]
    character_ids: list[str]


class SemanticUnderstandingLedger(StrictUnderstandingModel):
    semantic_schema: Literal["honcut.semantic-understanding.v2"] = Field(
        alias="schema",
        serialization_alias="schema",
    )
    entities: list[SemanticEntityRecord]
    source_mentions: list[SemanticSourceMentionRecord]
    events: list[SemanticEventRecord]


class StoryOrderUnderstanding(StrictUnderstandingModel):
    suggested_order: list[str]
    narrative_consistent: bool
    issues: list[str]


class ShotSemanticReview(StrictUnderstandingModel):
    verdict: Literal["pass", "reshoot", "revise", "fail"]
    issues: list[str]
    confidence: float = Field(ge=0, le=1)


class FrameEvidence(StrictUnderstandingModel):
    frame_id: str
    observed: str


class FirstFrameIssue(StrictUnderstandingModel):
    code: Literal["ANNOTATION_CONTAMINATION", "STYLE_MISMATCH"]
    severity: Literal["severe", "moderate", "minor"]
    frame_ids: list[str]
    message: str
    expected: str
    observed: str
    confidence: float = Field(ge=0, le=1)
    frame_evidence: list[FrameEvidence]


class FirstFrameUnderstanding(StrictUnderstandingModel):
    issues: list[FirstFrameIssue]


class PanelEvidence(StrictUnderstandingModel):
    shot_id: str
    observed: str


class CharacterEvidence(StrictUnderstandingModel):
    character_id: str
    reference_input_indices: list[int]
    expected: str
    observed: str
    storyboard_ids: list[str]


class StoryboardVisualIssue(StrictUnderstandingModel):
    red_line: Literal["R1", "R2", "R3", "R4"]
    severity: Literal["severe", "moderate", "minor"]
    mismatch_type: Literal[
        "identity",
        "gender",
        "clothing_color",
        "lighting",
        "action",
        "end_state",
        "other",
    ]
    shot_ids: list[str]
    message: str
    reference_input_indices: list[int]
    expected: str
    observed: str
    confidence: float = Field(ge=0, le=1)
    character_evidence: list[CharacterEvidence]
    panel_evidence: list[PanelEvidence]


class StoryboardVisualUnderstanding(StrictUnderstandingModel):
    issues: list[StoryboardVisualIssue]


class CharacterReferenceViewUnderstanding(StrictUnderstandingModel):
    passed: bool
    view_match: bool
    framing_match: bool
    neutral_pose: bool
    hands_empty: bool
    plain_background: bool
    single_character: bool
    face_visible: bool
    both_eyes_visible: bool
    issues: list[str]


class CharacterCrossViewUnderstanding(StrictUnderstandingModel):
    passed: bool
    identity_consistent: bool
    outfit_consistent: bool
    body_proportions_consistent: bool
    issues: list[str]


class CharacterReferenceUnderstanding(StrictUnderstandingModel):
    views: dict[str, CharacterReferenceViewUnderstanding]
    cross_view: CharacterCrossViewUnderstanding
    failed_views: list[str]
    summary: str = ""


class IdentityDetailUnderstanding(StrictUnderstandingModel):
    passed: bool
    character_identity_consistent: bool
    declared_items_present: bool
    item_geometry_consistent: bool
    colors_materials_consistent: bool
    attachment_modes_correct: bool
    undeclared_items_absent: bool
    issues: list[str]


UnderstandingT = TypeVar("UnderstandingT", bound=BaseModel)


def _json_text(raw: str) -> str:
    """Remove one complete Markdown fence without scanning for JSON fragments."""

    text = str(raw).strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*\n?(.*?)\n?```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fenced.group(1).strip() if fenced else text


def _raise_document_error(message: str, text: str, position: int = 0) -> None:
    raise json.JSONDecodeError(message, text, max(0, min(position, len(text))))


def _validate_repair_candidate(text: str) -> None:
    """Allow syntax repair for exactly one bounded JSON document.

    ``json-repair`` can intentionally recover stray prose, multiple top-level
    fragments, and value-level truncation.  Those behaviours are unsafe at a
    production contract boundary because they can turn an incomplete QA list
    into an empty passing list.  HonCut permits punctuation/delimiter repair,
    but never document scanning or incomplete-value salvage.
    """

    if not text:
        _raise_document_error("empty structured response", text)

    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(text)
    except json.JSONDecodeError:
        pass
    else:
        if text[end:].strip():
            _raise_document_error(
                "structured response contains trailing content",
                text,
                end,
            )
        return

    if text[0] not in "[{":
        _raise_document_error(
            "structured response must start with one JSON container",
            text,
        )

    stack: list[tuple[str, int]] = []
    in_string = False
    escaped = False
    matching_opener = {"}": "{", "]": "["}
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char in "[{":
            stack.append((char, index))
            continue
        if char not in "]}":
            continue
        opener = matching_opener[char]
        if not any(value == opener for value, _ in stack):
            _raise_document_error(
                "structured response has an unmatched closing delimiter",
                text,
                index,
            )
        while stack and stack[-1][0] != opener:
            _, missing_closer_position = stack.pop()
            completed_value = text[missing_closer_position + 1 : index].strip()
            if not completed_value or completed_value[-1] in "{[:,":
                _raise_document_error(
                    "structured response ends before a nested value is complete",
                    text,
                    index,
                )
        stack.pop()
        if not stack and text[index + 1 :].strip():
            _raise_document_error(
                "structured response contains multiple documents or trailing prose",
                text,
                index + 1,
            )

    if in_string or escaped:
        _raise_document_error(
            "structured response ends inside a string",
            text,
            len(text),
        )
    if text[-1] in "{[:," or re.search(
        r"(?:tru|fals|nul|[-+.eE])$",
        text,
        flags=re.IGNORECASE,
    ):
        _raise_document_error(
            "structured response ends before a value is complete",
            text,
            len(text),
        )


def parse_structured_output(
    raw: str,
    response_model: type[UnderstandingT],
) -> UnderstandingT:
    """Repair bounded JSON syntax, then validate the strict business DTO."""

    text = _json_text(raw)
    _validate_repair_candidate(text)
    try:
        payload = repair_json(
            text,
            return_objects=True,
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise json.JSONDecodeError(
            "structured response JSON repair failed",
            text,
            0,
        ) from exc
    return response_model.model_validate(payload)


def native_json_schema_format(
    response_model: type[BaseModel],
) -> dict[str, object]:
    """Build the Responses API ``text.format`` structured-output contract."""

    name = re.sub(r"[^A-Za-z0-9_-]+", "_", response_model.__name__).strip("_")
    schema = response_model.model_json_schema()

    def strictify(value: object) -> None:
        if isinstance(value, dict):
            value.pop("default", None)
            if value.get("type") == "object" or "properties" in value:
                properties = value.get("properties")
                if isinstance(properties, dict):
                    value["additionalProperties"] = False
                    value["required"] = list(properties)
            for nested in value.values():
                strictify(nested)
        elif isinstance(value, list):
            for nested in value:
                strictify(nested)

    strictify(schema)
    return {
        "type": "json_schema",
        "name": name or "honcut_understanding",
        "strict": True,
        "schema": schema,
    }


def native_chat_json_schema_format(
    response_model: type[BaseModel],
) -> dict[str, object]:
    """Build the Chat Completions ``response_format`` equivalent."""

    response_format = native_json_schema_format(response_model)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_format["name"],
            "strict": response_format["strict"],
            "schema": response_format["schema"],
        },
    }


__all__ = [
    "CharacterUnderstandingBatch",
    "CharacterReferenceUnderstanding",
    "DirectorPlanUnderstanding",
    "DurationScaledActionSelectionBatch",
    "FirstFrameUnderstanding",
    "IdentityDetailUnderstanding",
    "EventUnderstandingBatch",
    "ShotSemanticReview",
    "StoryboardPromptUnderstanding",
    "StoryboardVisualUnderstanding",
    "StoryOrderUnderstanding",
    "native_json_schema_format",
    "native_chat_json_schema_format",
    "parse_structured_output",
]

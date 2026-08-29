"""Run-local Phase 3 performance boards derived from canonical Pxx action lineage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw, ImageFilter

from prompt.seedream_image_prompt import (
    bind_reference_roles,
    image_request_fingerprint,
)
from quality.character_performance_qa import (
    CHARACTER_PERFORMANCE_CELL_QA_SCHEMA,
    CHARACTER_PERFORMANCE_QA_SCHEMA,
    CharacterPerformanceQAError,
    combine_character_performance_qa,
    review_character_performance_board,
    review_character_performance_cell,
)
from quality.character_reference_qa import file_sha256


CHARACTER_PERFORMANCE_BOARD_SCHEMA = "honcut.character-performance-board.v2"
CHARACTER_PERFORMANCE_GUIDE_SCHEMA = "honcut.character-performance-guide.v1"
CHARACTER_PERFORMANCE_CELL_SCHEMA = "honcut.character-performance-cell.v2"
CHARACTER_PERFORMANCE_POSE_GUIDE_SCHEMA = "honcut.character-performance-pose-guide.v2"
CHARACTER_PERFORMANCE_POSE_CONSTRAINTS_SCHEMA = (
    "honcut.character-performance-pose-constraints.v1"
)
PERFORMANCE_PROMPT_OPTIMIZATION_SCHEMA = (
    "honcut.character-performance-prompt-optimization.v2"
)
PERFORMANCE_PROMPT_TEMPLATE_ID = "honcut.character-performance-board-prompt.v2"
PERFORMANCE_PROMPT_GUIDANCE_URL = (
    "https://ark.volcengine.com/region:cn-beijing/docs/82379/1824121?lang=zh"
)
PERFORMANCE_BOARD_FILENAME = "performance_reference_board.png"
PERFORMANCE_BOARD_RECEIPT = "performance_reference_board.json"
PERFORMANCE_BOARD_QA_RECEIPT = "performance_reference_board_qa.json"
PERFORMANCE_BOARD_SIZE = "3072x2048"
PERFORMANCE_BOARD_PIXEL_SIZE = (3072, 2048)
PERFORMANCE_CELL_SIZE = "2048x2048"
PERFORMANCE_CELL_PIXEL_SIZE = (2048, 2048)
PERFORMANCE_POSE_GUIDE_PIXEL_SIZE = (1024, 1024)
PERFORMANCE_COMPOSITION_MODE = "locally_feathered_2x3_v2"
PERFORMANCE_MAX_CELL_CORRECTION_ROUNDS = 1
PERFORMANCE_CELL_IDS = tuple(f"A{index:02d}" for index in range(1, 7))
PERFORMANCE_POSE_VOCABULARY = (
    "combat_ready",
    "attack",
    "evade",
    "block",
    "prop_hold",
    "prop_use",
)
PERFORMANCE_KEY_POSE_PHASES = (
    "recognizable authored key pose",
    "peak force, displacement or contact",
    "authored end pose without a later result",
)
PERFORMANCE_PROMPT_EVALUATION_DIMENSIONS = (
    "synthetic_recognizability",
    "aesthetic_quality",
    "identity_consistency",
    "pose_clarity",
    "prop_accuracy",
    "no_clones",
    "no_layout_pollution",
)
_PERFORMANCE_PROMPT_CANDIDATES = (
    {
        "candidate_id": "identity_only_baseline_v1",
        "covered_dimensions": (
            "synthetic_recognizability",
            "aesthetic_quality",
            "identity_consistency",
        ),
    },
    {
        "candidate_id": "action_first_v1",
        "covered_dimensions": (
            "pose_clarity",
            "prop_accuracy",
            "no_clones",
            "no_layout_pollution",
        ),
    },
    {
        "candidate_id": "lineage_first_synthetic_v1",
        "covered_dimensions": PERFORMANCE_PROMPT_EVALUATION_DIMENSIONS,
    },
)
_BEAT_ID_RE = re.compile(r"^S\d+_P\d+$")
_SOURCE_ACTION_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_ACTION_MARKERS = (
    "战斗", "攻击", "挥", "砍", "刺", "踢", "打", "冲", "闪避", "躲", "后仰",
    "格挡", "防御", "抵挡", "持", "握", "拿", "举", "使用", "操作", "投掷",
    "fight", "attack", "strike", "kick", "dodge", "evade", "block", "guard",
    "hold", "wield", "use", "operate", "throw",
)


class PerformanceBoardImageClient(Protocol):
    model: str

    def image_to_image(
        self,
        prompt: str,
        ref_image: str | list[str],
        output_path: str,
        size: str,
    ) -> str: ...


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def performance_prompt_optimization_contract() -> dict[str, Any]:
    """Return the frozen, zero-request prompt candidate decision.

    Ark's debug/batch/scoring workflow is used as the design pattern, while the
    production Runtime never calls a prompt-optimization service. The scores
    below are deterministic contract-coverage scores, not model quality claims.
    """
    dimensions = list(PERFORMANCE_PROMPT_EVALUATION_DIMENSIONS)
    candidates = []
    for candidate in _PERFORMANCE_PROMPT_CANDIDATES:
        covered = list(candidate["covered_dimensions"])
        candidates.append({
            "candidate_id": candidate["candidate_id"],
            "covered_dimensions": covered,
            "contract_coverage_score": len(covered),
            "maximum_contract_coverage_score": len(dimensions),
        })
    selected = "lineage_first_synthetic_v1"
    template_contract = {
        "template_id": PERFORMANCE_PROMPT_TEMPLATE_ID,
        "selected_candidate_id": selected,
        "instruction_order": [
            "identity_and_synthetic_makeup_lock",
            "canonical_pxx_action_lineage",
            "recognizable_action_key_pose",
            "prop_ownership",
            "pixel_and_story_boundary_prohibitions",
        ],
        "evaluation_dimensions": dimensions,
    }
    return {
        "schema": PERFORMANCE_PROMPT_OPTIMIZATION_SCHEMA,
        "status": "selected",
        "method": "offline_contract_candidate_comparison",
        "guidance_source": PERFORMANCE_PROMPT_GUIDANCE_URL,
        "score_kind": "deterministic_contract_coverage",
        "evaluation_dimensions": dimensions,
        "candidates": candidates,
        "selected_candidate_id": selected,
        "selected_template_id": PERFORMANCE_PROMPT_TEMPLATE_ID,
        "selected_template_sha256": _canonical_hash(template_contract),
        "production_auto_optimization": False,
        "provider_request_count": 0,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}.",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.", suffix=".png", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _character_props(character: Mapping[str, Any]) -> list[dict[str, Any]]:
    appearance = character.get("appearance")
    props = appearance.get("identity_props") if isinstance(appearance, Mapping) else []
    return [dict(item) for item in props or [] if isinstance(item, Mapping)]


def _prop_ids_for_action(
    props: list[dict[str, Any]],
    action_text: str,
) -> list[str]:
    folded = action_text.casefold()
    matched = []
    for index, prop in enumerate(props, 1):
        prop_id = str(prop.get("id") or f"prop_{index}").strip()
        prop_name = str(prop.get("name") or "").strip()
        if (
            prop_id.casefold() in folded
            or (prop_name and prop_name.casefold() in folded)
            or prop.get("persistence") == "always"
            or prop.get("attachment_mode") == "body_attached"
        ):
            matched.append(prop_id)
    if not matched and len(props) == 1 and any(marker in folded for marker in _ACTION_MARKERS):
        matched.append(str(props[0].get("id") or "prop_1"))
    return matched


def _generation_units_for_source(
    beat: Mapping[str, Any], source_action_unit_id: str
) -> list[dict[str, Any]]:
    return [
        dict(unit)
        for unit in beat.get("generation_action_units") or []
        if isinstance(unit, Mapping)
        and str(unit.get("source_action_unit_id") or "").strip()
        == source_action_unit_id
    ]


def _source_bindings_for_beat(beat: Mapping[str, Any], beat_id: str) -> list[dict[str, Any]]:
    units = [
        dict(unit)
        for unit in beat.get("generation_action_units") or []
        if isinstance(unit, Mapping)
    ]
    unit_ids = [str(unit.get("unit_id") or "").strip() for unit in units]
    if (
        not units
        or any(not _SOURCE_ACTION_ID_RE.fullmatch(unit_id) for unit_id in unit_ids)
        or len(set(unit_ids)) != len(unit_ids)
    ):
        raise ValueError(f"{beat_id} lacks canonical generation action-unit lineage")

    declared_source_ids = list(dict.fromkeys(
        str(value).strip()
        for value in beat.get("source_action_unit_ids") or []
        if str(value).strip()
    ))
    discovered_source_ids = list(dict.fromkeys(
        str(unit.get("source_action_unit_id") or "").strip()
        for unit in units
        if str(unit.get("source_action_unit_id") or "").strip()
    ))
    if declared_source_ids or discovered_source_ids:
        if (
            declared_source_ids != discovered_source_ids
            or any(not _SOURCE_ACTION_ID_RE.fullmatch(value) for value in declared_source_ids)
        ):
            raise ValueError(f"{beat_id} has inconsistent source action-unit lineage")
        return [
            {
                "source_action_unit_id": source_id,
                "source_lineage_kind": "source_action_unit",
                "units": _generation_units_for_source(beat, source_id),
            }
            for source_id in declared_source_ids
        ]

    assignments = [
        dict(item)
        for item in beat.get("timeline_assignments") or []
        if isinstance(item, Mapping)
    ]
    assignment_ids = [
        str(value).strip()
        for value in beat.get("timeline_assignment_ids") or []
        if str(value).strip()
    ]
    embedded_assignment_ids = [
        str(item.get("assignment_id") or "").strip() for item in assignments
    ]
    if (
        len(assignments) != len(units)
        or assignment_ids != embedded_assignment_ids
        or any(not _SOURCE_ACTION_ID_RE.fullmatch(value) for value in assignment_ids)
        or len(set(assignment_ids)) != len(assignment_ids)
    ):
        raise ValueError(f"{beat_id} lacks canonical source action-unit lineage")
    bindings = []
    for assignment_id, assignment, unit in zip(
        assignment_ids, assignments, units, strict=True
    ):
        if (
            assignment.get("source_event_id") != unit.get("source_event_id")
            or assignment.get("source_generation_unit_indexes")
            != unit.get("source_generation_unit_indexes")
        ):
            raise ValueError(f"{beat_id} timeline assignment lineage is inconsistent")
        bindings.append({
            "source_action_unit_id": assignment_id,
            "source_lineage_kind": "timeline_assignment",
            "units": [unit],
        })
    return bindings


def _unit_action_text(units: list[dict[str, Any]], beat: Mapping[str, Any]) -> str:
    """Return canonical source facts without repeating generated prose."""
    parts: list[str] = []
    for unit in units:
        values = unit.get("source_fact_echoes") or []
        if not values:
            values = unit.get("actions") or []
        if isinstance(values, str):
            values = [values]
        parts.extend(str(value).strip() for value in values if str(value).strip())
    if not parts:
        parts.append(str(beat.get("action") or "").strip())
    seen: set[str] = set()
    return "；".join(part for part in parts if part and not (part in seen or seen.add(part)))


def _unit_source_fact_text(units: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for unit in units:
        values = unit.get("source_fact_echoes") or []
        if isinstance(values, str):
            values = [values]
        parts.extend(str(value).strip() for value in values if str(value).strip())
    seen: set[str] = set()
    return "；".join(part for part in parts if not (part in seen or seen.add(part)))


def _pose_category(action_text: str, prop_ids: list[str]) -> str:
    folded = action_text.casefold()
    classifications = (
        ("prop_use", ("使用", "操作", "启动", "发射", "use", "operate", "activate", "fire")),
        ("block", ("格挡", "抵挡", "防御", "架住", "block", "guard", "parry")),
        (
            "evade",
            (
                "闪避", "躲", "后仰", "后倾", "侧身", "侧滑", "降低重心",
                "evade", "dodge", "avoid",
            ),
        ),
        ("attack", ("攻击", "挥", "砍", "刺", "踢", "击", "attack", "strike", "kick", "swing")),
        (
            "combat_ready",
            (
                "戒备", "准备", "对峙", "步架", "架势", "站稳",
                "ready", "stance", "footwork base",
            ),
        ),
        ("prop_hold", ("持", "握", "拿", "举", "hold", "wield", "carry")),
    )
    for category, markers in classifications:
        if any(marker in folded for marker in markers):
            return category
    return "prop_hold" if prop_ids else "combat_ready"


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    folded = text.casefold()
    return any(marker.casefold() in folded for marker in markers)


def _pose_constraints(action_text: str, pose_category: str) -> dict[str, str]:
    """Project authored facts into a compact, zero-provider pose contract."""
    text = str(action_text).strip()
    lead_foot = "unspecified"
    if _contains_any(text, (
        "左脚向前", "左脚前跨", "左脚在前", "左腿在前",
        "left foot forward", "lead with the left foot", "left leg forward",
    )):
        lead_foot = "left"
    elif _contains_any(text, (
        "右脚向前", "右脚前跨", "右脚在前", "右腿在前",
        "right foot forward", "lead with the right foot", "right leg forward",
    )):
        lead_foot = "right"

    moving_foot = "unspecified"
    if _contains_any(text, ("左脚向", "左脚滑", "左脚跨", "收回左脚", "left foot")):
        moving_foot = "left"
    elif _contains_any(text, ("右脚向", "右脚滑", "右脚跨", "收回右脚", "right foot")):
        moving_foot = "right"

    movement_direction = "unspecified"
    if _contains_any(text, ("向左侧", "向左方", "向左滑", "leftward", "to the left")):
        movement_direction = "left"
    elif _contains_any(text, ("向右侧", "向右方", "向右滑", "rightward", "to the right")):
        movement_direction = "right"
    elif _contains_any(text, ("向前", "前跨", "前进", "forward")):
        movement_direction = "forward"
    elif _contains_any(text, ("向后", "后撤", "后退", "backward", "back step")):
        movement_direction = "backward"

    stance = "neutral"
    if _contains_any(text, ("双脚前后", "前后步", "步架", "弓步", "staggered", "split stance")):
        stance = "staggered"
    elif _contains_any(text, ("侧滑", "侧移", "向左侧", "向右侧", "lateral", "side step")):
        stance = "lateral"
    elif pose_category in {"attack", "evade", "block", "combat_ready"}:
        stance = "action"

    knees = (
        "bent"
        if _contains_any(text, ("屈膝", "膝微屈", "弯膝", "bent knee", "knees bent"))
        else "unspecified"
    )
    center_of_gravity = (
        "lowered"
        if _contains_any(text, (
            "重心下沉", "降低身体重心", "降低重心", "低重心", "沉髋", "压低重心",
            "lower center", "lowered center", "drop the center",
        ))
        else "unspecified"
    )
    torso_lean = "unspecified"
    if _contains_any(text, ("后倾", "后仰", "lean back", "lean backward")):
        torso_lean = "backward"
    elif _contains_any(text, ("向左倾", "左倾", "lean left")):
        torso_lean = "left"
    elif _contains_any(text, ("向右倾", "右倾", "lean right")):
        torso_lean = "right"
    elif _contains_any(text, ("中正", "直立", "upright")):
        torso_lean = "upright"

    prop_orientation = "unspecified"
    if _contains_any(text, ("横握", "横向", "水平", "horizontal")):
        prop_orientation = "horizontal"
    elif _contains_any(text, ("竖直", "垂直", "vertical")):
        prop_orientation = "vertical"
    elif _contains_any(text, ("斜", "对角", "diagonal")):
        prop_orientation = "diagonal"

    prop_start = "unspecified"
    prop_end = "unspecified"
    direction_pairs = (
        ("right_lower", "left_upper", ("右下方向左上", "右下至左上", "lower right to upper left")),
        ("left_lower", "right_upper", ("左下方向右上", "左下至右上", "lower left to upper right")),
        ("right_upper", "left_lower", ("右上方向左下", "右上至左下", "upper right to lower left")),
        ("left_upper", "right_lower", ("左上方向右下", "左上至右下", "upper left to lower right")),
    )
    for start, end, markers in direction_pairs:
        if _contains_any(text, markers):
            prop_start, prop_end = start, end
            prop_orientation = "diagonal"
            break

    prop_side = "unspecified"
    if _contains_any(text, ("身体左前", "身前左侧", "character's left", "body left")):
        prop_side = "left"
    elif _contains_any(text, ("身体右前", "身前右侧", "character's right", "body right")):
        prop_side = "right"
    elif _contains_any(text, ("身前", "in front of the body")):
        prop_side = "front"

    return {
        "schema": CHARACTER_PERFORMANCE_POSE_CONSTRAINTS_SCHEMA,
        "stance": stance,
        "knees": knees,
        "center_of_gravity": center_of_gravity,
        "torso_lean": torso_lean,
        "lead_foot": lead_foot,
        "moving_foot": moving_foot,
        "movement_direction": movement_direction,
        "prop_orientation": prop_orientation,
        "prop_side": prop_side,
        "prop_start": prop_start,
        "prop_end": prop_end,
    }


def _eligible_beat(shot: Mapping[str, Any], beat: Mapping[str, Any], props: list[dict[str, Any]]) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            beat.get("action"), beat.get("start_state"), beat.get("end_state"),
            shot.get("what"), shot.get("visual"),
        )
    ).casefold()
    body_contract = beat.get("body_action_contract")
    return bool(
        str(shot.get("shot_intent") or "").casefold() == "action"
        or (isinstance(body_contract, Mapping) and body_contract.get("required") is True)
        or any(marker in text for marker in _ACTION_MARKERS)
        or _prop_ids_for_action(props, text)
    )


def _canonical_shot_id(shot: Mapping[str, Any], index: int) -> str:
    """Use the same canonical Sxx identity written by the storyboard owner."""
    raw = shot.get("shot_id") or shot.get("id") or shot.get("shot_order") or index
    text = str(raw).strip()
    if re.fullmatch(r"[Ss]?\d+", text):
        return f"S{int(text.lstrip('Ss')):02d}"
    return text or f"S{index:02d}"


def build_character_performance_plan(
    storyboard: Mapping[str, Any],
    character: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Bind six pose cells to existing canonical Pxx/action-unit facts only."""
    character_id = str(character.get("id") or "").strip()
    if not character_id:
        raise ValueError("performance board character ID is missing")
    props = _character_props(character)
    beat_bindings: list[dict[str, Any]] = []
    for shot_index, shot in enumerate(storyboard.get("shots") or [], start=1):
        if not isinstance(shot, Mapping):
            continue
        shot_id = _canonical_shot_id(shot, shot_index)
        for beat in shot.get("storyboard_beats") or []:
            if not isinstance(beat, Mapping):
                continue
            if character_id not in [str(value) for value in beat.get("character_ids") or []]:
                continue
            if not _eligible_beat(shot, beat, props):
                continue
            beat_id = str(beat.get("beat_id") or "").strip()
            if not _BEAT_ID_RE.fullmatch(beat_id) or not beat_id.startswith(f"{shot_id}_"):
                raise ValueError(f"invalid canonical performance beat ID: {beat_id!r}")
            source_bindings = []
            for lineage in _source_bindings_for_beat(beat, beat_id):
                source_id = lineage["source_action_unit_id"]
                units = lineage["units"]
                action_text = _unit_action_text(units, beat)
                if not action_text:
                    raise ValueError(f"{beat_id} has an empty authored action")
                category_text = _unit_source_fact_text(units) or action_text
                source_bindings.append({
                    "source_action_unit_id": source_id,
                    "source_lineage_kind": lineage["source_lineage_kind"],
                    "generation_action_unit_ids": [
                        str(unit.get("unit_id") or "").strip()
                        for unit in units
                        if str(unit.get("unit_id") or "").strip()
                    ],
                    "action_description": action_text,
                    "prop_ids": _prop_ids_for_action(props, action_text),
                    "pose_category": _pose_category(
                        category_text,
                        _prop_ids_for_action(props, action_text),
                    ),
                })
            beat_bindings.append({
                "beat_id": beat_id,
                "parent_shot_id": shot_id,
                "sources": source_bindings,
            })
    if not beat_bindings:
        return None
    if len(beat_bindings) > len(PERFORMANCE_CELL_IDS):
        raise ValueError(
            f"{character_id} needs {len(beat_bindings)} Pxx performance bindings; "
            "one v1 board can carry at most six"
        )

    candidates = [
        {**source, "beat_id": beat["beat_id"], "parent_shot_id": beat["parent_shot_id"]}
        for beat in beat_bindings
        for source in beat["sources"]
    ]
    selected: list[dict[str, Any]] = []
    for beat in beat_bindings:
        selected.append({
            **beat["sources"][0],
            "beat_id": beat["beat_id"],
            "parent_shot_id": beat["parent_shot_id"],
        })
    selected_keys = {
        (binding["beat_id"], binding["source_action_unit_id"])
        for binding in selected
    }
    for candidate in candidates:
        key = (candidate["beat_id"], candidate["source_action_unit_id"])
        if len(selected) >= len(PERFORMANCE_CELL_IDS):
            break
        if key not in selected_keys:
            selected.append(dict(candidate))
            selected_keys.add(key)

    # Specialize duplicate prop-bearing action families without changing their
    # canonical source-action lineage or inventing a new story action.
    seen_categories: set[str] = set()
    for index, binding in enumerate(selected):
        category = str(binding["pose_category"])
        if category not in seen_categories:
            seen_categories.add(category)
            continue
        if not binding["prop_ids"]:
            continue
        specialized = ""
        if (
            category in {"combat_ready", "prop_hold"}
            and "prop_hold" not in seen_categories
        ):
            specialized = "prop_hold"
        elif (
            category in {"attack", "block", "prop_use"}
            and "prop_use" not in seen_categories
        ):
            specialized = "prop_use"
        if specialized:
            selected[index] = {**binding, "pose_category": specialized}
            seen_categories.add(specialized)
    cursor = 0
    while len(selected) < len(PERFORMANCE_CELL_IDS):
        missing_categories = [
            category
            for category in PERFORMANCE_POSE_VOCABULARY
            if category not in {binding["pose_category"] for binding in selected}
        ]
        candidate: dict[str, Any] | None = None
        for missing_category in missing_categories:
            if missing_category == "prop_hold":
                compatible = [
                    item
                    for item in candidates
                    if item["prop_ids"]
                    and item["pose_category"] in {
                        "prop_hold", "combat_ready", "block", "attack",
                    }
                ]
            elif missing_category == "prop_use":
                prop_actions = [item for item in candidates if item["prop_ids"]]
                compatible = [
                    item
                    for item in prop_actions
                    if _contains_any(
                        item["action_description"],
                        (
                            "使用", "操作", "启动", "发射", "挥", "砍", "刺", "击",
                            "use", "operate", "activate", "fire", "swing", "strike",
                        ),
                    )
                ] or [
                    item
                    for item in prop_actions
                    if item["pose_category"] in {"prop_use", "attack", "block"}
                ]
            else:
                compatible = [
                    item
                    for item in candidates
                    if item["pose_category"] == missing_category
                ]
            if compatible:
                candidate = {**compatible[0], "pose_category": missing_category}
                break
        if candidate is None:
            candidate = dict(candidates[cursor % len(candidates)])
        selected.append(dict(candidate))
        cursor += 1

    cells = []
    occurrence_by_binding: dict[tuple[str, str], int] = {}
    for index, (cell_id, binding) in enumerate(
        zip(PERFORMANCE_CELL_IDS, selected, strict=True)
    ):
        binding_key = (binding["beat_id"], binding["source_action_unit_id"])
        occurrence = occurrence_by_binding.get(binding_key, 0)
        occurrence_by_binding[binding_key] = occurrence + 1
        cells.append({
            "cell_id": cell_id,
            "grid_position": {"row": index // 3 + 1, "column": index % 3 + 1},
            "character_id": character_id,
            "parent_shot_id": binding["parent_shot_id"],
            "beat_id": binding["beat_id"],
            "source_action_unit_id": binding["source_action_unit_id"],
            "source_lineage_kind": binding["source_lineage_kind"],
            "generation_action_unit_ids": binding["generation_action_unit_ids"],
            "prop_ids": binding["prop_ids"],
            "pose_category": binding["pose_category"],
            "pose_focus": PERFORMANCE_KEY_POSE_PHASES[
                occurrence % len(PERFORMANCE_KEY_POSE_PHASES)
            ],
            "action_description": binding["action_description"],
            "pose_constraints": _pose_constraints(
                binding["action_description"], binding["pose_category"]
            ),
        })
    return {
        "schema": CHARACTER_PERFORMANCE_BOARD_SCHEMA,
        "character_id": character_id,
        "usage": "run_local_video_motion_reference_only",
        "pose_vocabulary": list(PERFORMANCE_POSE_VOCABULARY),
        "layout": {"rows": 2, "columns": 3, "cell_order": list(PERFORMANCE_CELL_IDS)},
        "cells": cells,
    }


def build_character_performance_prompt(
    character: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> str:
    appearance = character.get("appearance")
    styling = appearance.get("synthetic_styling") if isinstance(appearance, Mapping) else None
    aesthetic_contract = _compact_performance_styling_contract(styling)
    positions = (
        "top-left", "top-center", "top-right",
        "bottom-left", "bottom-center", "bottom-right",
    )
    binding_ids: dict[tuple[str, str], str] = {}
    binding_facts: list[str] = []
    for cell in plan["cells"]:
        key = (str(cell["beat_id"]), str(cell["source_action_unit_id"]))
        if key in binding_ids:
            continue
        binding_id = f"B{len(binding_ids) + 1:02d}"
        binding_ids[key] = binding_id
        binding_facts.append(f"- {binding_id}: {cell['action_description']}")
    cell_instructions = "\n".join(
        (
            f"- {position} (internal {cell['cell_id']}; never print the ID): "
            f"action={binding_ids[(str(cell['beat_id']), str(cell['source_action_unit_id']))]}; "
            f"role={cell['pose_category']}; key_pose={cell['pose_focus']}; "
            f"constraints={_compact_pose_constraints_text(cell['pose_constraints'])}."
        )
        for position, cell in zip(positions, plan["cells"], strict=True)
    )
    return f"""Create one clean 3:2 character performance reference image containing six evenly
spaced full-body poses of the same single character, arranged conceptually as 2 rows x 3 columns.
There are no visible cell dividers: use one seamless neutral light-gray studio background.

Identity is locked by Image 1. Preserve the exact face, pearl bio-ceramic synthetic porcelain
makeup, narrow temple-to-cheek iridescent circuit stripe, luminous iris ring, hair, body
proportions, outfit, colors and character-specific makeup design across all six poses.
Image 2, when present, supplies declared prop geometry/material/color only.
{aesthetic_contract}

Canonical authored action facts:
{chr(10).join(binding_facts)}

Follow these six positions exactly:
{cell_instructions}

Every position must visibly perform its exact action, not a generic standing guard. An evade must
show the specified displaced foot, lowered center of gravity and torso lean. An attack must show the
specified stepping foot and exact prop swing direction. A block must show the declared defensive
prop placement. Keep the entire body and prop visible with enough empty space to read the silhouette.
Do not invent an attack, outcome, injury, wet clothing, torn clothing, dirt or later story state.
A declared prop may appear only in cells whose prop_ids include it and must remain owned by this
character.

Pixel prohibitions: no text, no letters, no numbers, no Axx labels, no arrows, no captions, no UI,
no panel borders, no grid lines. Do not add another character. This is a pose reference sheet, not a
storyboard and not a finished cinematic frame.
"""


def _compact_performance_styling_contract(styling: Any) -> str:
    """Keep identity anchors without repeating the complete video policy."""
    if not isinstance(styling, Mapping):
        return ""
    anchors = [
        str(value).strip()
        for value in styling.get("visible_anchors") or []
        if str(value).strip()
    ]
    design_id = str(styling.get("makeup_design_id") or "").strip()
    material = str(styling.get("non_human_material") or "").strip()
    pieces = [
        f"makeup_design_id={design_id}" if design_id else "",
        f"material={material}" if material else "",
        "anchors=" + " | ".join(anchors) if anchors else "",
    ]
    identity = "; ".join(piece for piece in pieces if piece)
    return (
        "Synthetic identity lock: " + identity + ". Keep a warm, healthy, elegant "
        "complexion with clear pupils and catchlights; never corpse-like, "
        "horror-styled, masked or photoreal."
    )


def _compact_pose_constraints_text(constraints: Any) -> str:
    if not isinstance(constraints, Mapping):
        raise CharacterPerformanceQAError("performance prompt lacks pose constraints")
    ordered_keys = (
        "stance", "knees", "center_of_gravity", "torso_lean", "lead_foot",
        "moving_foot", "movement_direction", "prop_orientation", "prop_side",
        "prop_start", "prop_end",
    )
    declared = [
        f"{key}={constraints.get(key)}"
        for key in ordered_keys
        if str(constraints.get(key) or "unspecified") not in {"unspecified", "neutral"}
    ]
    return ",".join(declared) or "no additional directional fact"


def build_character_performance_cell_prompt(
    character: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> str:
    appearance = character.get("appearance")
    styling = appearance.get("synthetic_styling") if isinstance(appearance, Mapping) else None
    aesthetic_contract = _compact_performance_styling_contract(styling)
    return f"""Create one square full-body character action reference on a seamless neutral
light-gray studio background. Show exactly one character and exactly one clearly readable pose.

Identity is locked by Image 1. Preserve the exact face, pearl bio-ceramic synthetic porcelain
makeup, circuit stripe, luminous iris ring, hair, proportions, outfit and colors. Image 2, when
present, supplies the declared prop geometry/material/color only. The final action-pose schematic
reference supplies body-joint, anatomical-side and prop-line geometry only. Match its silhouette
and limb topology, but render the finished character rather than diagram lines or colored joints.
{aesthetic_contract}

Exact authored action: {cell['action_description']}
Required action role: {cell['pose_category']}
Required key pose: {cell['pose_focus']}
Deterministic pose constraints: {_compact_pose_constraints_text(cell['pose_constraints'])}

The pose must visibly perform the exact authored action, not a generic guard. Preserve every stated
left/right foot placement, center-of-gravity change, torso lean, prop orientation and swing direction.
Keep the entire body, both feet, both hands and the complete prop visible with clear negative space.
Do not add a second person, a later action, an outcome, injury, wet clothing, torn clothing or dirt.
No text, letters, numbers, labels, arrows, captions, UI, borders or grid lines.
"""


def build_character_performance_cell_correction_prompt(
    character: Mapping[str, Any],
    cell: Mapping[str, Any],
    qa_feedback: list[str],
) -> str:
    """Project one failed cell into a single bounded, QA-guided redraw prompt."""
    if not qa_feedback or any(not str(item).strip() for item in qa_feedback):
        raise CharacterPerformanceQAError("performance cell correction requires QA feedback")
    feedback = "\n".join(f"- {str(item).strip()}" for item in qa_feedback)
    role = str(cell.get("pose_category") or "").strip()
    pose_directive = {
        "attack": (
            "Freeze at the authored attack peak, with the named stepping foot visibly advanced "
            "and the complete weapon frozen on the authored diagonal swing path."
        ),
        "block": (
            "Freeze only after the authored foot has retracted and place the complete weapon "
            "at the exact declared defensive side and orientation."
        ),
        "evade": (
            "Freeze after the authored lateral displacement with the named foot moved, center "
            "of gravity lowered and torso visibly leaning in the authored direction."
        ),
        "prop_hold": "Freeze a readable authored hold with both hands and the full prop visible.",
        "prop_use": "Freeze the authored prop-use peak; do not replace it with a neutral hold.",
        "combat_ready": "Freeze the exact authored ready stance, not a neutral standing portrait.",
    }.get(role, "Freeze the exact authored action at its most readable key pose.")
    return f"""{build_character_performance_cell_prompt(character, cell)}

This is the one allowed correction redraw for internal cell {cell['cell_id']}. The previous image
failed strict action QA for these exact reasons:
{feedback}

Correct every listed failure and nothing else. {pose_directive}
Use the deterministic pose constraints and the final schematic as the geometry authority.
Anatomical left/right belongs to the character, not the viewer; never infer it from a camera
label. Make the major action, weight shift and prop relationship unmistakable. Do not print this
feedback or the internal cell ID in the image.
"""


def _pose_guide_geometry(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Project one authored action into a deterministic front-facing pose skeleton."""
    role = str(cell.get("pose_category") or "combat_ready")
    constraints = cell.get("pose_constraints")
    if (
        not isinstance(constraints, Mapping)
        or constraints.get("schema") != CHARACTER_PERFORMANCE_POSE_CONSTRAINTS_SCHEMA
    ):
        raise CharacterPerformanceQAError("performance cell lacks current pose constraints")
    geometry: dict[str, Any] = {
        "head": (512, 150),
        "neck": (512, 225),
        "left_shoulder": (610, 275),
        "right_shoulder": (414, 275),
        "left_elbow": (650, 430),
        "right_elbow": (374, 430),
        "left_hand": (620, 545),
        "right_hand": (404, 545),
        "left_hip": (570, 530),
        "right_hip": (454, 530),
        "left_knee": (620, 710),
        "right_knee": (404, 710),
        "left_foot": (660, 900),
        "right_foot": (364, 900),
        "prop": ((405, 570), (725, 300)),
        "emphasis": (),
    }
    if role == "evade":
        geometry.update({
            "head": (600, 190),
            "neck": (570, 255),
            "left_shoulder": (650, 300),
            "right_shoulder": (470, 330),
            "left_elbow": (715, 450),
            "right_elbow": (410, 455),
            "left_hand": (680, 565),
            "right_hand": (375, 560),
            "left_hip": (590, 555),
            "right_hip": (480, 575),
            "left_knee": (660, 720),
            "right_knee": (340, 735),
            "left_foot": (700, 885),
            "right_foot": (225, 900),
            "prop": ((370, 575), (690, 420)),
            "emphasis": ("right_hip", "right_knee", "right_foot"),
        })
    elif role in {"attack", "prop_use"}:
        geometry.update({
            "left_elbow": (590, 410),
            "right_elbow": (465, 465),
            "left_hand": (555, 500),
            "right_hand": (475, 535),
            "left_knee": (645, 725),
            "right_knee": (410, 700),
            "left_foot": (700, 925),
            "right_foot": (360, 855),
            "prop": ((330, 720), (760, 255)),
            "emphasis": ("left_hip", "left_knee", "left_foot"),
        })
    elif role == "block":
        geometry.update({
            "left_elbow": (650, 390),
            "right_elbow": (500, 435),
            "left_hand": (690, 470),
            "right_hand": (650, 555),
            "left_knee": (600, 690),
            "right_knee": (420, 725),
            "left_foot": (610, 835),
            "right_foot": (375, 915),
            "prop": ((705, 715), (705, 235)),
            "emphasis": ("left_hand", "left_knee", "left_foot"),
        })
    elif role == "prop_hold":
        geometry.update({
            "left_hand": (585, 500),
            "right_hand": (455, 500),
            "prop": ((360, 520), (690, 520)),
            "emphasis": ("left_hand", "right_hand"),
        })

    # Anatomical left appears on viewer-right in this front-facing schematic.
    if constraints.get("stance") == "staggered":
        geometry.update({
            "left_knee": (585, 715),
            "right_knee": (430, 675),
            "left_foot": (620, 930),
            "right_foot": (385, 840),
        })
    if constraints.get("lead_foot") == "left":
        geometry.update({
            "left_knee": (620, 730),
            "right_knee": (420, 675),
            "left_foot": (675, 930),
            "right_foot": (385, 835),
        })
    elif constraints.get("lead_foot") == "right":
        geometry.update({
            "left_knee": (600, 675),
            "right_knee": (400, 730),
            "left_foot": (640, 835),
            "right_foot": (350, 930),
        })
    if (
        constraints.get("movement_direction") == "right"
        and constraints.get("moving_foot") == "right"
    ):
        geometry.update({"right_knee": (335, 735), "right_foot": (205, 910)})
    elif (
        constraints.get("movement_direction") == "left"
        and constraints.get("moving_foot") == "left"
    ):
        geometry.update({"left_knee": (690, 735), "left_foot": (820, 910)})
    if constraints.get("knees") == "bent":
        geometry.update({
            "left_knee": (
                geometry["left_knee"][0] + 25,
                min(770, geometry["left_knee"][1] + 25),
            ),
            "right_knee": (
                geometry["right_knee"][0] - 25,
                min(770, geometry["right_knee"][1] + 25),
            ),
        })

    prop_points = {
        "right_lower": (330, 740),
        "left_lower": (694, 740),
        "right_upper": (330, 245),
        "left_upper": (694, 245),
    }
    prop_start = str(constraints.get("prop_start") or "unspecified")
    prop_end = str(constraints.get("prop_end") or "unspecified")
    if prop_start in prop_points and prop_end in prop_points:
        geometry["prop"] = (prop_points[prop_start], prop_points[prop_end])
    elif constraints.get("prop_orientation") == "horizontal":
        geometry["prop"] = ((330, 520), (694, 520))
    elif constraints.get("prop_orientation") == "vertical":
        prop_x = 700 if constraints.get("prop_side") == "left" else 324
        geometry["prop"] = ((prop_x, 740), (prop_x, 245))
    return geometry


def _ensure_performance_pose_guide(
    output_dir: Path,
    character_id: str,
    cell: Mapping[str, Any],
) -> dict[str, str]:
    """Create a text-free, zero-provider pose schematic and strict hash receipt."""
    cell_id = str(cell.get("cell_id") or "")
    guide_dir = output_dir / "characters" / character_id / "performance_pose_guides"
    guide_path = guide_dir / f"{cell_id}.png"
    receipt_path = guide_dir / f"{cell_id}.json"
    cell_hash = _canonical_hash(cell)
    geometry = _pose_guide_geometry(cell)
    geometry_hash = _canonical_hash(geometry)
    expected = {
        "schema": CHARACTER_PERFORMANCE_POSE_GUIDE_SCHEMA,
        "status": "done",
        "character_id": character_id,
        "cell_id": cell_id,
        "cell_sha256": cell_hash,
        "pose_constraints_sha256": _canonical_hash(cell["pose_constraints"]),
        "geometry_sha256": geometry_hash,
        "generator": "honcut.front-facing-action-skeleton.v2",
        "image": guide_path.relative_to(output_dir).as_posix(),
        "provider_requests": 0,
    }
    cached = _load_json(receipt_path)
    if receipt_path.is_file() and cached is None:
        raise CharacterPerformanceQAError(f"{character_id} {cell_id} pose receipt is corrupt")
    if cached is not None and cached.get("schema") != CHARACTER_PERFORMANCE_POSE_GUIDE_SCHEMA:
        raise CharacterPerformanceQAError(f"{character_id} {cell_id} pose receipt schema is unknown")
    if (
        cached is not None
        and all(cached.get(key) == value for key, value in expected.items())
        and guide_path.is_file()
        and cached.get("image_sha256") == file_sha256(guide_path)
    ):
        try:
            with Image.open(guide_path) as image:
                image.verify()
                if image.size == PERFORMANCE_POSE_GUIDE_PIXEL_SIZE:
                    return {
                        "path": expected["image"],
                        "sha256": str(cached["image_sha256"]),
                        "kind": "action_pose_schematic",
                    }
        except (OSError, ValueError):
            pass

    canvas = Image.new("RGB", PERFORMANCE_POSE_GUIDE_PIXEL_SIZE, (24, 28, 36))
    draw = ImageDraw.Draw(canvas)
    line_color = (226, 232, 240)
    joint_color = (255, 184, 76)
    prop_color = (70, 220, 255)
    width = 24
    head_x, head_y = geometry["head"]
    draw.ellipse(
        (head_x - 60, head_y - 60, head_x + 60, head_y + 60),
        outline=line_color,
        width=width,
    )
    pairs = (
        ("neck", "left_shoulder"), ("neck", "right_shoulder"),
        ("neck", "left_hip"), ("neck", "right_hip"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_hand"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_hand"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_foot"),
        ("right_hip", "right_knee"), ("right_knee", "right_foot"),
    )
    for start, end in pairs:
        draw.line((geometry[start], geometry[end]), fill=line_color, width=width)
    for key in (
        "neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_hand", "right_hand", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_foot", "right_foot",
    ):
        x, y = geometry[key]
        radius = 18 if key not in geometry["emphasis"] else 28
        color = joint_color if key in geometry["emphasis"] else line_color
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.line(geometry["prop"], fill=prop_color, width=34)
    _atomic_png(guide_path, canvas)
    receipt = {**expected, "image_sha256": file_sha256(guide_path)}
    _atomic_json(receipt_path, receipt)
    return {
        "path": expected["image"],
        "sha256": str(receipt["image_sha256"]),
        "kind": "action_pose_schematic",
    }


def _reference_records(output_dir: Path, character: Mapping[str, Any]) -> list[dict[str, str]]:
    character_id = str(character.get("id") or "").strip()
    character_dir = output_dir / "characters" / character_id
    paths = [character_dir / "reference_board.png"]
    prop_board = character_dir / "prop_detail_board.png"
    if prop_board.is_file():
        paths.append(prop_board)
    records = []
    for path in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"performance board reference missing: {path}")
        records.append({
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": file_sha256(path),
            "kind": (
                "character_identity_board" if path.name == "reference_board.png"
                else "prop_detail_board"
            ),
        })
    return records


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_character_performance_board(
    output_dir: Path,
    character_id: str,
    *,
    expected_plan: Mapping[str, Any] | None = None,
    expected_prompt_sha256: str | None = None,
    expected_model: str | None = None,
    expected_references: list[dict[str, str]] | None = None,
) -> bool:
    output_dir = Path(output_dir)
    character_dir = output_dir / "characters" / character_id
    image_path = character_dir / PERFORMANCE_BOARD_FILENAME
    receipt = _load_json(character_dir / PERFORMANCE_BOARD_RECEIPT)
    qa = _load_json(character_dir / PERFORMANCE_BOARD_QA_RECEIPT)
    if receipt is None or qa is None:
        return False
    if (
        receipt.get("schema") != CHARACTER_PERFORMANCE_BOARD_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("character_id") != character_id
        or receipt.get("image") != PERFORMANCE_BOARD_FILENAME
        or qa.get("schema") != CHARACTER_PERFORMANCE_QA_SCHEMA
        or qa.get("status") != "passed"
        or qa.get("character_id") != character_id
        or qa.get("passed") is not True
        or receipt.get("composition_mode") != PERFORMANCE_COMPOSITION_MODE
    ):
        return False
    if receipt.get("prompt_optimization") != performance_prompt_optimization_contract():
        return False
    plan = receipt.get("plan")
    if not isinstance(plan, dict) or plan.get("schema") != CHARACTER_PERFORMANCE_BOARD_SCHEMA:
        return False
    cells = plan.get("cells")
    if (
        not isinstance(cells, list)
        or any(not isinstance(cell, dict) for cell in cells)
        or [cell.get("cell_id") for cell in cells]
        != list(PERFORMANCE_CELL_IDS)
        or any(cell.get("character_id") != character_id for cell in cells)
        or any(
            not isinstance(cell.get("pose_constraints"), dict)
            or cell["pose_constraints"].get("schema")
            != CHARACTER_PERFORMANCE_POSE_CONSTRAINTS_SCHEMA
            for cell in cells
        )
    ):
        return False
    plan_hash = _canonical_hash(plan)
    if receipt.get("plan_sha256") != plan_hash or qa.get("plan_sha256") != plan_hash:
        return False
    if expected_plan is not None and dict(expected_plan) != plan:
        return False
    if expected_prompt_sha256 is not None and receipt.get("prompt_sha256") != expected_prompt_sha256:
        return False
    if expected_model is not None and receipt.get("model") != expected_model:
        return False
    references = receipt.get("references")
    if expected_references is not None and references != expected_references:
        return False
    if not isinstance(references, list):
        return False
    for record in references:
        if not isinstance(record, dict):
            return False
        source = output_dir / str(record.get("path") or "")
        if not source.is_file() or file_sha256(source) != record.get("sha256"):
            return False
    if not image_path.is_file():
        return False
    image_hash = file_sha256(image_path)
    if receipt.get("image_sha256") != image_hash or qa.get("image_sha256") != image_hash:
        return False
    try:
        with Image.open(image_path) as image:
            image.verify()
            if image.size != PERFORMANCE_BOARD_PIXEL_SIZE:
                return False
    except (OSError, ValueError):
        return False
    components = receipt.get("component_cells") or []
    if components:
        if (
            not isinstance(components, list)
            or [item.get("cell_id") for item in components if isinstance(item, dict)]
            != list(PERFORMANCE_CELL_IDS)
            or qa.get("action_verdict_source") != "isolated_persisted_cells"
            or qa.get("board_verdict_source") != "whole_board_global_fields_only"
        ):
            return False
        cells_by_id = {str(cell["cell_id"]): cell for cell in cells}
        for component in components:
            if not isinstance(component, dict):
                return False
            cell_id = str(component.get("cell_id") or "")
            component_path = output_dir / str(component.get("image") or "")
            component_qa = _load_json(component_path.with_suffix(".qa.json"))
            component_references = component.get("references")
            if (
                component.get("schema") != CHARACTER_PERFORMANCE_CELL_SCHEMA
                or component.get("status") != "passed"
                or component.get("character_id") != character_id
                or component.get("cell_sha256")
                != _canonical_hash(cells_by_id.get(cell_id))
                or not component_path.is_file()
                or component.get("image_sha256") != file_sha256(component_path)
                or not isinstance(component_references, list)
                or not any(
                    item.get("kind") == "action_pose_schematic"
                    for item in component_references
                    if isinstance(item, dict)
                )
                or component_qa is None
                or component_qa.get("schema")
                != CHARACTER_PERFORMANCE_CELL_QA_SCHEMA
                or component_qa.get("status") != "passed"
                or component_qa.get("passed") is not True
                or component_qa.get("image_sha256") != file_sha256(component_path)
            ):
                return False
            for reference in component_references:
                if not isinstance(reference, dict):
                    return False
                reference_path = output_dir / str(reference.get("path") or "")
                if (
                    not reference_path.is_file()
                    or reference.get("sha256") != file_sha256(reference_path)
                ):
                    return False
    return True


def _materialize_performance_guides(
    output_dir: Path,
    character_id: str,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    character_dir = output_dir / "characters" / character_id
    board_path = character_dir / PERFORMANCE_BOARD_FILENAME
    board_receipt_path = character_dir / PERFORMANCE_BOARD_RECEIPT
    board_hash = file_sha256(board_path)
    board_receipt_hash = file_sha256(board_receipt_path)
    cells = [dict(cell) for cell in plan["cells"]]
    beat_ids = list(dict.fromkeys(cell["beat_id"] for cell in cells))
    results = []
    with Image.open(board_path) as raw:
        board = raw.convert("RGB")
        cell_width = board.width // 3
        cell_height = board.height // 2
        for beat_id in beat_ids:
            selected = [cell for cell in cells if cell["beat_id"] == beat_id]
            destination_dir = output_dir / "performance_guides" / beat_id
            destination = destination_dir / f"{character_id}.png"
            receipt_path = destination.with_suffix(".json")
            guide = Image.new("RGB", (cell_width * len(selected), cell_height))
            for offset, cell in enumerate(selected):
                position = cell["grid_position"]
                left = (int(position["column"]) - 1) * cell_width
                upper = (int(position["row"]) - 1) * cell_height
                crop = board.crop((left, upper, left + cell_width, upper + cell_height))
                guide.paste(crop, (offset * cell_width, 0))
            _atomic_png(destination, guide)
            receipt = {
                "schema": CHARACTER_PERFORMANCE_GUIDE_SCHEMA,
                "status": "done",
                "usage": "current_pxx_motion_reference_only",
                "character_id": character_id,
                "beat_id": beat_id,
                "image": destination.relative_to(output_dir).as_posix(),
                "image_sha256": file_sha256(destination),
                "source_board": board_path.relative_to(output_dir).as_posix(),
                "source_board_sha256": board_hash,
                "source_board_receipt": board_receipt_path.relative_to(output_dir).as_posix(),
                "source_board_receipt_sha256": board_receipt_hash,
                "cell_ids": [cell["cell_id"] for cell in selected],
                "source_action_unit_ids": list(dict.fromkeys(
                    cell["source_action_unit_id"] for cell in selected
                )),
                "prop_ids": list(dict.fromkeys(
                    prop_id for cell in selected for prop_id in cell["prop_ids"]
                )),
                "provider_requests": 0,
            }
            _atomic_json(receipt_path, receipt)
            results.append(receipt)
    return results


def validate_character_performance_guide(
    output_dir: Path,
    character_id: str,
    beat_id: str,
) -> dict[str, Any] | None:
    output_dir = Path(output_dir)
    receipt_path = output_dir / "performance_guides" / beat_id / f"{character_id}.json"
    receipt = _load_json(receipt_path)
    if receipt is None:
        return None
    image_path = output_dir / str(receipt.get("image") or "")
    board_path = output_dir / str(receipt.get("source_board") or "")
    board_receipt_path = output_dir / str(receipt.get("source_board_receipt") or "")
    if (
        receipt.get("schema") != CHARACTER_PERFORMANCE_GUIDE_SCHEMA
        or receipt.get("status") != "done"
        or receipt.get("usage") != "current_pxx_motion_reference_only"
        or receipt.get("character_id") != character_id
        or receipt.get("beat_id") != beat_id
        or receipt.get("provider_requests") != 0
        or not isinstance(receipt.get("cell_ids"), list)
        or not receipt["cell_ids"]
        or any(cell_id not in PERFORMANCE_CELL_IDS for cell_id in receipt["cell_ids"])
        or not image_path.is_file()
        or not board_path.is_file()
        or not board_receipt_path.is_file()
        or receipt.get("image_sha256") != file_sha256(image_path)
        or receipt.get("source_board_sha256") != file_sha256(board_path)
        or receipt.get("source_board_receipt_sha256") != file_sha256(board_receipt_path)
        or not validate_character_performance_board(output_dir, character_id)
    ):
        return None
    return receipt


def _generate_performance_cell_components(
    output_dir: Path,
    character: Mapping[str, Any],
    plan: Mapping[str, Any],
    references: list[dict[str, str]],
    *,
    image_client: PerformanceBoardImageClient,
    resolved_model: str,
    correction_round: int = 0,
    correction_feedback: Mapping[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    character_id = str(character.get("id") or "").strip()
    component_dir = output_dir / "characters" / character_id / "performance_cells"
    component_dir.mkdir(parents=True, exist_ok=True)
    base_roles = [
        "character_identity_board_only",
        *(["character_prop_detail_only"] if len(references) > 1 else []),
    ]
    results: list[dict[str, Any]] = []
    provider_requests = 0
    feedback_by_cell = dict(correction_feedback or {})
    if correction_round not in {0, 1}:
        raise CharacterPerformanceQAError("performance cell correction round is out of bounds")
    if correction_round == 0 and feedback_by_cell:
        raise CharacterPerformanceQAError("base performance cells cannot carry correction feedback")
    if correction_round > 0 and not feedback_by_cell:
        raise CharacterPerformanceQAError("performance correction requires failed cell feedback")
    unknown_feedback = set(feedback_by_cell).difference(PERFORMANCE_CELL_IDS)
    if unknown_feedback:
        raise CharacterPerformanceQAError("performance correction references unknown cells")
    for cell in plan["cells"]:
        cell_id = str(cell["cell_id"])
        pose_guide = _ensure_performance_pose_guide(output_dir, character_id, cell)
        cell_references = [*references, pose_guide]
        reference_paths = [str(output_dir / record["path"]) for record in cell_references]
        roles = [*base_roles, "action_pose_schematic_only"]
        is_correction_target = correction_round > 0 and cell_id in feedback_by_cell
        if correction_round > 0 and not is_correction_target:
            image_path = component_dir / f"{cell_id}.png"
            receipt_path = component_dir / f"{cell_id}.json"
            base_prompt = bind_reference_roles(
                build_character_performance_cell_prompt(character, cell),
                roles,
            )
            base_prompt_hash = hashlib.sha256(base_prompt.encode("utf-8")).hexdigest()
            base_request_fingerprint = image_request_fingerprint(
                prompt=base_prompt,
                model=resolved_model,
                size=PERFORMANCE_CELL_SIZE,
                reference_image_sha256=[record["sha256"] for record in cell_references],
            )
            base_expected = {
                "schema": CHARACTER_PERFORMANCE_CELL_SCHEMA,
                "status": "passed",
                "character_id": character_id,
                "cell_id": cell_id,
                "cell_sha256": _canonical_hash(cell),
                "model": resolved_model,
                "size": PERFORMANCE_CELL_SIZE,
                "prompt_sha256": base_prompt_hash,
                "request_fingerprint": base_request_fingerprint,
                "references": cell_references,
                "image": image_path.relative_to(output_dir).as_posix(),
            }
            cached = _load_json(receipt_path)
            if (
                cached is None
                or any(cached.get(key) != value for key, value in base_expected.items())
                or not image_path.is_file()
                or cached.get("image_sha256") != file_sha256(image_path)
            ):
                raise CharacterPerformanceQAError(
                    f"{character_id} {cell_id} base component cannot support correction"
                )
            try:
                with Image.open(image_path) as image:
                    image.verify()
                    if image.size != PERFORMANCE_CELL_PIXEL_SIZE:
                        raise CharacterPerformanceQAError(
                            f"{character_id} {cell_id} base component has invalid size"
                        )
            except (OSError, ValueError) as exc:
                raise CharacterPerformanceQAError(
                    f"{character_id} {cell_id} base component is not a valid image"
                ) from exc
            results.append(dict(cached))
            continue
        if is_correction_target:
            correction_dir = component_dir / "corrections" / f"round_{correction_round:02d}"
            image_path = correction_dir / f"{cell_id}.png"
            receipt_path = correction_dir / f"{cell_id}.json"
            raw_prompt = build_character_performance_cell_correction_prompt(
                character,
                cell,
                feedback_by_cell[cell_id],
            )
        else:
            image_path = component_dir / f"{cell_id}.png"
            receipt_path = component_dir / f"{cell_id}.json"
            raw_prompt = build_character_performance_cell_prompt(character, cell)
        prompt = bind_reference_roles(raw_prompt, roles)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        request_fingerprint = image_request_fingerprint(
            prompt=prompt,
            model=resolved_model,
            size=PERFORMANCE_CELL_SIZE,
            reference_image_sha256=[record["sha256"] for record in cell_references],
        )
        expected = {
            "schema": CHARACTER_PERFORMANCE_CELL_SCHEMA,
            "status": "passed",
            "character_id": character_id,
            "cell_id": cell_id,
            "cell_sha256": _canonical_hash(cell),
            "model": resolved_model,
            "size": PERFORMANCE_CELL_SIZE,
            "prompt_sha256": prompt_hash,
            "request_fingerprint": request_fingerprint,
            "references": cell_references,
            "image": image_path.relative_to(output_dir).as_posix(),
        }
        if is_correction_target:
            expected.update({
                "correction_round": correction_round,
                "qa_feedback": feedback_by_cell[cell_id],
            })
        cached = _load_json(receipt_path)
        cache_valid = bool(
            cached is not None
            and all(cached.get(key) == value for key, value in expected.items())
            and image_path.is_file()
            and cached.get("image_sha256") == file_sha256(image_path)
        )
        if cache_valid:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                    cache_valid = image.size == PERFORMANCE_CELL_PIXEL_SIZE
            except (OSError, ValueError):
                cache_valid = False
        if not cache_valid:
            _atomic_json(receipt_path, {**expected, "status": "pending"})
            image_client.image_to_image(
                prompt=prompt,
                ref_image=reference_paths,
                output_path=str(image_path),
                size=PERFORMANCE_CELL_SIZE,
            )
            provider_requests += 1
            try:
                with Image.open(image_path) as image:
                    image.verify()
                    if image.size != PERFORMANCE_CELL_PIXEL_SIZE:
                        raise CharacterPerformanceQAError(
                            f"{character_id} {cell_id} component has invalid size {image.size}"
                        )
            except (OSError, ValueError) as exc:
                raise CharacterPerformanceQAError(
                    f"{character_id} {cell_id} component is not a valid image"
                ) from exc
            cached = {**expected, "image_sha256": file_sha256(image_path)}
            _atomic_json(receipt_path, cached)
        results.append(dict(cached))
    return results, provider_requests


def _compose_performance_cell_components(
    output_dir: Path,
    components: list[dict[str, Any]],
    destination: Path,
) -> None:
    if [item.get("cell_id") for item in components] != list(PERFORMANCE_CELL_IDS):
        raise CharacterPerformanceQAError("performance components are incomplete or unordered")
    board = Image.new("RGB", PERFORMANCE_BOARD_PIXEL_SIZE, (232, 235, 238))
    pose_size = 940
    pose_inset = (1024 - pose_size) // 2
    feather_mask = Image.new("L", (pose_size, pose_size), 0)
    ImageDraw.Draw(feather_mask).rectangle(
        (48, 48, pose_size - 49, pose_size - 49),
        fill=255,
    )
    feather_mask = feather_mask.filter(ImageFilter.GaussianBlur(radius=32))
    for index, component in enumerate(components):
        image_path = output_dir / str(component["image"])
        with Image.open(image_path) as raw:
            pose = raw.convert("RGB").resize(
                (pose_size, pose_size),
                Image.Resampling.LANCZOS,
            )
        board.paste(
            pose,
            (
                (index % 3) * 1024 + pose_inset,
                (index // 3) * 1024 + pose_inset,
            ),
            feather_mask,
        )
    _atomic_png(destination, board)


def _failed_performance_cell_feedback(
    qa_result: Mapping[str, Any],
) -> dict[str, list[str]]:
    """Return only cell-scoped failures eligible for one bounded redraw."""
    cells = qa_result.get("cells")
    if not isinstance(cells, list):
        raise CharacterPerformanceQAError("performance QA omitted cell verdicts")
    by_id = {
        str(item.get("cell_id") or ""): item
        for item in cells
        if isinstance(item, Mapping)
    }
    if set(by_id) != set(PERFORMANCE_CELL_IDS):
        raise CharacterPerformanceQAError("performance QA cell verdicts are incomplete")
    feedback: dict[str, list[str]] = {}
    boolean_checks = (
        "same_character",
        "action_semantics_match",
        "pose_distinct",
        "clothing_consistent",
        "makeup_consistent",
        "healthy_beautiful_synthetic_styling",
        "no_uncanny_or_corpse_like_styling",
        "prop_ownership_correct",
        "no_extra_character",
        "no_text_or_layout_marks",
    )
    for cell_id in PERFORMANCE_CELL_IDS:
        verdict = by_id[cell_id]
        failed_checks = [key for key in boolean_checks if verdict.get(key) is not True]
        if not failed_checks:
            continue
        issues = [
            str(item).strip()
            for item in verdict.get("issues") or []
            if str(item).strip()
        ]
        if not issues:
            issues = ["Failed strict checks: " + ", ".join(failed_checks)]
        feedback[cell_id] = issues
    return feedback


def _review_performance_cell_components(
    output_dir: Path,
    character_id: str,
    plan: Mapping[str, Any],
    components: list[dict[str, Any]],
    references: list[dict[str, str]],
    *,
    review_client: Any,
    synthetic_styling: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Persist one immutable action verdict per component image hash."""
    cells = {
        str(cell.get("cell_id") or ""): dict(cell)
        for cell in plan.get("cells") or []
        if isinstance(cell, Mapping)
    }
    if [item.get("cell_id") for item in components] != list(PERFORMANCE_CELL_IDS):
        raise CharacterPerformanceQAError("performance components are incomplete for cell QA")
    identity_record = next(
        (item for item in references if item.get("kind") == "character_identity_board"),
        None,
    )
    prop_record = next(
        (item for item in references if item.get("kind") == "prop_detail_board"),
        None,
    )
    if identity_record is None:
        raise CharacterPerformanceQAError("performance cell QA has no identity reference")
    identity_path = output_dir / identity_record["path"]
    prop_path = output_dir / prop_record["path"] if prop_record is not None else None
    styling_hash = _canonical_hash(synthetic_styling or {})
    results: list[dict[str, Any]] = []
    verdict_fields = (
        "cell_id",
        "same_character",
        "action_semantics_match",
        "fine_direction_match",
        "pose_distinct",
        "clothing_consistent",
        "makeup_consistent",
        "healthy_beautiful_synthetic_styling",
        "no_uncanny_or_corpse_like_styling",
        "prop_ownership_correct",
        "no_extra_character",
        "no_text_or_layout_marks",
        "issues",
    )
    for component in components:
        cell_id = str(component["cell_id"])
        cell = cells.get(cell_id)
        if cell is None:
            raise CharacterPerformanceQAError(f"performance cell QA cannot resolve {cell_id}")
        image_path = output_dir / str(component["image"])
        qa_path = image_path.with_suffix(".qa.json")
        expected = {
            "schema": CHARACTER_PERFORMANCE_CELL_QA_SCHEMA,
            "character_id": character_id,
            "cell_id": cell_id,
            "cell_sha256": _canonical_hash(cell),
            "image": image_path.relative_to(output_dir).as_posix(),
            "image_sha256": file_sha256(image_path),
            "identity_reference_sha256": identity_record["sha256"],
            "prop_reference_sha256": (
                prop_record["sha256"] if prop_record is not None else None
            ),
            "synthetic_styling_sha256": styling_hash,
        }
        cached = _load_json(qa_path)
        if qa_path.is_file() and cached is None:
            raise CharacterPerformanceQAError(f"{character_id} {cell_id} QA receipt is corrupt")
        if cached is not None and cached.get("schema") != CHARACTER_PERFORMANCE_CELL_QA_SCHEMA:
            raise CharacterPerformanceQAError(
                f"{character_id} {cell_id} QA receipt schema is unknown"
            )
        cache_valid = bool(
            cached is not None
            and all(cached.get(key) == value for key, value in expected.items())
            and cached.get("status") in {"passed", "failed"}
            and isinstance(cached.get("passed"), bool)
            and all(field in cached for field in verdict_fields)
        )
        if cache_valid:
            results.append({
                "schema": CHARACTER_PERFORMANCE_CELL_QA_SCHEMA,
                "passed": bool(cached["passed"]),
                **{field: cached[field] for field in verdict_fields},
            })
            continue
        qa_result = review_character_performance_cell(
            review_client,
            image_path,
            identity_path=identity_path,
            prop_path=prop_path,
            character_id=character_id,
            cell=cell,
            synthetic_styling=synthetic_styling,
        )
        receipt = {
            **expected,
            **qa_result,
            "status": "passed" if qa_result["passed"] else "failed",
        }
        _atomic_json(qa_path, receipt)
        results.append(qa_result)
    return results


def generate_character_performance_board(
    output_dir: Path,
    storyboard: Mapping[str, Any],
    character: Mapping[str, Any],
    *,
    image_client: PerformanceBoardImageClient,
    review_client: Any,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Generate or exactly reuse one performance board, then derive Pxx guides locally."""
    output_dir = Path(output_dir)
    plan = build_character_performance_plan(storyboard, character)
    if plan is None:
        return None
    character_id = str(character.get("id") or "").strip()
    character_dir = output_dir / "characters" / character_id
    character_dir.mkdir(parents=True, exist_ok=True)
    references = _reference_records(output_dir, character)
    prompt = build_character_performance_prompt(character, plan)
    roles = [
        "character_identity_board_only",
        *(["character_prop_detail_only"] if len(references) > 1 else []),
    ]
    bound_prompt = bind_reference_roles(prompt, roles)
    prompt_hash = hashlib.sha256(bound_prompt.encode("utf-8")).hexdigest()
    resolved_model = str(model or getattr(image_client, "model", "") or "doubao-seedream-5.0-lite")
    request_fingerprint = image_request_fingerprint(
        prompt=bound_prompt,
        model=resolved_model,
        size=PERFORMANCE_BOARD_SIZE,
        reference_image_sha256=[record["sha256"] for record in references],
    )
    if validate_character_performance_board(
        output_dir,
        character_id,
        expected_plan=plan,
        expected_prompt_sha256=prompt_hash,
        expected_model=resolved_model,
        expected_references=references,
    ):
        guides = _materialize_performance_guides(output_dir, character_id, plan)
        return {
            "character_id": character_id,
            "reused": True,
            "board": (character_dir / PERFORMANCE_BOARD_FILENAME).relative_to(output_dir).as_posix(),
            "guides": guides,
            "provider_requests": 0,
        }

    image_path = character_dir / PERFORMANCE_BOARD_FILENAME
    receipt_path = character_dir / PERFORMANCE_BOARD_RECEIPT
    qa_path = character_dir / PERFORMANCE_BOARD_QA_RECEIPT
    plan_hash = _canonical_hash(plan)
    previous_receipt = _load_json(receipt_path)
    previous_qa = _load_json(qa_path)
    previous_exact_contract = bool(
        previous_receipt is not None
        and previous_receipt.get("plan") == plan
        and previous_receipt.get("plan_sha256") == plan_hash
        and previous_receipt.get("model") == resolved_model
        and previous_receipt.get("prompt_sha256") == prompt_hash
        and previous_receipt.get("references") == references
    )
    previous_exact_failure = bool(
        previous_exact_contract and previous_receipt.get("status") == "failed"
    )
    previous_mode = (
        str(previous_receipt.get("generation_mode") or "whole_board")
        if previous_receipt is not None
        else "whole_board"
    )
    previous_cell_fallback = bool(
        previous_exact_contract
        and previous_mode == "per_cell_fallback"
        and isinstance(previous_receipt.get("component_cells"), list)
    )
    if (
        previous_exact_failure
        and previous_mode == "per_cell_correction"
        and previous_receipt.get("composition_mode") == PERFORMANCE_COMPOSITION_MODE
    ):
        raise CharacterPerformanceQAError(
            f"{character_id} bounded performance cell correction already failed blocking QA"
        )
    attempts = [
        dict(item)
        for item in (previous_receipt or {}).get("attempts") or []
        if isinstance(item, Mapping)
    ]
    correction_attempts = [
        dict(item)
        for item in (previous_receipt or {}).get("correction_attempts") or []
        if isinstance(item, Mapping)
    ]
    if previous_exact_failure and previous_mode == "whole_board" and not attempts:
        if previous_qa is not None:
            archived_qa = character_dir / "performance_reference_board_qa.whole_board.json"
            _atomic_json(archived_qa, previous_qa)
            attempts.append({
                "mode": "whole_board",
                "status": "failed",
                "provider_requests": 1,
                "image_sha256": previous_receipt.get("image_sha256"),
                "qa_receipt": archived_qa.name,
                "qa_receipt_sha256": file_sha256(archived_qa),
            })
    pending = {
        "schema": CHARACTER_PERFORMANCE_BOARD_SCHEMA,
        "status": "pending",
        "character_id": character_id,
        "plan": plan,
        "plan_sha256": plan_hash,
        "model": resolved_model,
        "size": PERFORMANCE_BOARD_SIZE,
        "prompt_sha256": prompt_hash,
        "prompt_optimization": performance_prompt_optimization_contract(),
        "request_fingerprint": request_fingerprint,
        "references": references,
        "image": PERFORMANCE_BOARD_FILENAME,
        "composition_mode": PERFORMANCE_COMPOSITION_MODE,
        "generation_mode": (
            "per_cell_fallback"
            if previous_exact_failure or previous_cell_fallback
            else "whole_board"
        ),
        "attempts": attempts,
        "correction_attempts": correction_attempts,
    }
    _atomic_json(receipt_path, pending)
    appearance = character.get("appearance")
    styling = appearance.get("synthetic_styling") if isinstance(appearance, Mapping) else None
    current_provider_requests = 0

    def review_current_board(
        component_records: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], str]:
        try:
            with Image.open(image_path) as generated:
                generated.verify()
                if generated.size != PERFORMANCE_BOARD_PIXEL_SIZE:
                    raise CharacterPerformanceQAError(
                        f"{character_id} performance board has invalid size {generated.size}"
                    )
        except (OSError, ValueError) as exc:
            raise CharacterPerformanceQAError(
                f"{character_id} performance board is not a valid image"
            ) from exc
        board_result = review_character_performance_board(
            review_client,
            image_path,
            character_id=character_id,
            cells=plan["cells"],
            synthetic_styling=styling if isinstance(styling, dict) else None,
        )
        if component_records:
            cell_results = _review_performance_cell_components(
                output_dir,
                character_id,
                plan,
                component_records,
                references,
                review_client=review_client,
                synthetic_styling=styling if isinstance(styling, dict) else None,
            )
            qa_result = combine_character_performance_qa(board_result, cell_results)
        else:
            qa_result = board_result
        image_hash = file_sha256(image_path)
        qa_receipt = {
            **qa_result,
            "status": "passed" if qa_result["passed"] else "failed",
            "character_id": character_id,
            "image": PERFORMANCE_BOARD_FILENAME,
            "image_sha256": image_hash,
            "plan_sha256": plan_hash,
        }
        _atomic_json(qa_path, qa_receipt)
        return qa_result, image_hash

    use_cell_fallback = previous_exact_failure or previous_cell_fallback
    if not use_cell_fallback:
        image_client.image_to_image(
            prompt=bound_prompt,
            ref_image=[str(output_dir / record["path"]) for record in references],
            output_path=str(image_path),
            size=PERFORMANCE_BOARD_SIZE,
        )
        current_provider_requests += 1
        qa_result, image_hash = review_current_board()
        if not qa_result["passed"]:
            archived_qa = character_dir / "performance_reference_board_qa.whole_board.json"
            _atomic_json(archived_qa, _load_json(qa_path) or {})
            attempts.append({
                "mode": "whole_board",
                "status": "failed",
                "provider_requests": 1,
                "image_sha256": image_hash,
                "qa_receipt": archived_qa.name,
                "qa_receipt_sha256": file_sha256(archived_qa),
            })
            use_cell_fallback = True

    components: list[dict[str, Any]] = []
    if use_cell_fallback:
        pending = {
            **pending,
            "generation_mode": "per_cell_fallback",
            "attempts": attempts,
        }
        _atomic_json(receipt_path, pending)
        components, component_requests = _generate_performance_cell_components(
            output_dir,
            character,
            plan,
            references,
            image_client=image_client,
            resolved_model=resolved_model,
        )
        current_provider_requests += component_requests
        _compose_performance_cell_components(output_dir, components, image_path)
        resume_failed_fallback = bool(
            previous_exact_failure
            and previous_mode == "per_cell_fallback"
            and previous_qa is not None
            and previous_receipt is not None
            and previous_receipt.get("image_sha256") == file_sha256(image_path)
            and previous_qa.get("image_sha256") == file_sha256(image_path)
        )
        if resume_failed_fallback:
            qa_result = dict(previous_qa)
            image_hash = file_sha256(image_path)
        else:
            qa_result, image_hash = review_current_board(components)
        if not qa_result["passed"]:
            feedback = _failed_performance_cell_feedback(qa_result)
            archived_qa = character_dir / "performance_reference_board_qa.per_cell_fallback.json"
            _atomic_json(archived_qa, _load_json(qa_path) or dict(qa_result))
            archived_board = character_dir / "performance_reference_board.per_cell_fallback.png"
            with Image.open(image_path) as failed_board:
                _atomic_png(archived_board, failed_board.convert("RGB"))
            if not any(
                item.get("mode") == "per_cell_fallback"
                for item in attempts
            ):
                attempts.append({
                    "mode": "per_cell_fallback",
                    "status": "failed",
                    "provider_requests": len(components),
                    "image_sha256": image_hash,
                    "image": archived_board.name,
                    "qa_receipt": archived_qa.name,
                    "qa_receipt_sha256": file_sha256(archived_qa),
                })
            if not feedback:
                failed = {
                    **pending,
                    "status": "failed",
                    "generation_mode": "per_cell_fallback",
                    "attempts": attempts,
                    "image_sha256": image_hash,
                    "component_cells": components,
                    "provider_request_count": sum(
                        int(item.get("provider_requests") or 0) for item in attempts
                    ),
                }
                _atomic_json(receipt_path, failed)
                raise CharacterPerformanceQAError(
                    f"{character_id} performance board has no cell-scoped correction target"
                )
            if correction_attempts:
                raise CharacterPerformanceQAError(
                    f"{character_id} performance cell correction history is already consumed"
                )
            correction_components, correction_requests = (
                _generate_performance_cell_components(
                    output_dir,
                    character,
                    plan,
                    references,
                    image_client=image_client,
                    resolved_model=resolved_model,
                    correction_round=PERFORMANCE_MAX_CELL_CORRECTION_ROUNDS,
                    correction_feedback=feedback,
                )
            )
            current_provider_requests += correction_requests
            _compose_performance_cell_components(
                output_dir,
                correction_components,
                image_path,
            )
            corrected_qa, corrected_hash = review_current_board(correction_components)
            correction_attempt = {
                "round": PERFORMANCE_MAX_CELL_CORRECTION_ROUNDS,
                "status": "passed" if corrected_qa["passed"] else "failed",
                "target_cell_ids": list(feedback),
                "qa_feedback": feedback,
                "provider_requests": len(feedback),
                "image_sha256": corrected_hash,
                "qa_receipt": PERFORMANCE_BOARD_QA_RECEIPT,
                "qa_receipt_sha256": file_sha256(qa_path),
            }
            correction_attempts.append(correction_attempt)
            components = correction_components
            qa_result = corrected_qa
            image_hash = corrected_hash
            pending = {
                **pending,
                "generation_mode": "per_cell_correction",
                "attempts": attempts,
                "correction_attempts": correction_attempts,
            }
            if not qa_result["passed"]:
                failed = {
                    **pending,
                    "status": "failed",
                    "image_sha256": image_hash,
                    "component_cells": components,
                    "provider_request_count": sum(
                        int(item.get("provider_requests") or 0) for item in attempts
                    ) + sum(
                        int(item.get("provider_requests") or 0)
                        for item in correction_attempts
                    ),
                }
                _atomic_json(receipt_path, failed)
                raise CharacterPerformanceQAError(
                    f"{character_id} performance board failed bounded correction QA"
                )
    complete = {
        **pending,
        "status": "passed",
        "image_sha256": image_hash,
        "qa_receipt": PERFORMANCE_BOARD_QA_RECEIPT,
        "qa_receipt_sha256": file_sha256(qa_path),
        "component_cells": components,
        "provider_request_count": sum(
            int(item.get("provider_requests") or 0) for item in attempts
        ) + sum(
            int(item.get("provider_requests") or 0)
            for item in correction_attempts
        ) + (
            len(components)
            if components
            and not any(item.get("mode") == "per_cell_fallback" for item in attempts)
            else 1 if not components and not attempts else 0
        ),
    }
    _atomic_json(receipt_path, complete)
    guides = _materialize_performance_guides(output_dir, character_id, plan)
    return {
        "character_id": character_id,
        "reused": False,
        "board": image_path.relative_to(output_dir).as_posix(),
        "guides": guides,
        "provider_requests": current_provider_requests,
    }


def generate_performance_reference_boards(
    output_dir: Path,
    storyboard: Mapping[str, Any],
    characters: list[Mapping[str, Any]],
    *,
    image_client: PerformanceBoardImageClient,
    review_client: Any,
) -> list[dict[str, Any]]:
    results = []
    for character in characters:
        result = generate_character_performance_board(
            output_dir,
            storyboard,
            character,
            image_client=image_client,
            review_client=review_client,
        )
        if result is not None:
            results.append(result)
    return results


def attach_performance_guides_to_storyboard(
    storyboard: dict[str, Any],
    generated_boards: list[dict[str, Any]],
) -> None:
    """Persist canonical run-local guide provenance on its owning Pxx beat."""
    by_beat: dict[str, list[dict[str, Any]]] = {}
    for board in generated_boards:
        for receipt in board.get("guides") or []:
            if not isinstance(receipt, dict):
                raise ValueError("performance guide receipt must be a mapping")
            beat_id = str(receipt.get("beat_id") or "").strip()
            by_beat.setdefault(beat_id, []).append({
                "kind": CHARACTER_PERFORMANCE_GUIDE_SCHEMA,
                "usage": "current_pxx_motion_reference_only",
                "character_id": receipt["character_id"],
                "beat_id": beat_id,
                "image": receipt["image"],
                "image_sha256": receipt["image_sha256"],
                "receipt": str(Path(receipt["image"]).with_suffix(".json")),
                "cell_ids": list(receipt["cell_ids"]),
                "source_action_unit_ids": list(
                    receipt["source_action_unit_ids"]
                ),
                "prop_ids": list(receipt["prop_ids"]),
                "source_board": receipt["source_board"],
                "source_board_sha256": receipt["source_board_sha256"],
                "source_board_receipt": receipt["source_board_receipt"],
                "source_board_receipt_sha256": (
                    receipt["source_board_receipt_sha256"]
                ),
            })
    observed_beats: set[str] = set()
    for shot in storyboard.get("shots") or []:
        if not isinstance(shot, dict):
            continue
        for beat in shot.get("storyboard_beats") or []:
            if not isinstance(beat, dict):
                continue
            beat_id = str(beat.get("beat_id") or "").strip()
            guides = sorted(
                by_beat.get(beat_id, []),
                key=lambda item: item["character_id"],
            )
            beat["character_performance_required"] = bool(guides)
            beat["character_performance_guides"] = guides
            if guides:
                observed_beats.add(beat_id)
    unknown = sorted(set(by_beat) - observed_beats)
    if unknown:
        raise ValueError(
            "performance guides reference unknown Pxx beats: " + ", ".join(unknown)
        )

"""Executable body choreography contracts for dance and combat shots.

The video model must not receive labels such as ``complex dance`` or
``continuous fight`` as if they were executable movement.  This module keeps
the authored plot outcome intact while requiring an ordered, observable body
mechanics score: performer, side, limbs, footwork, torso/weight transfer,
direction/contact and the resulting pose.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Iterable

BODY_ACTION_CONTRACT_SCHEMA = "honcut.body-action-choreography.v1"

_CHOREOGRAPHY_DOMAIN = re.compile(
    r"舞蹈|舞步|街舞|齐舞|群舞|breaking|breakdance|hip[ -]?hop|dance|"
    r"格斗|搏斗|打斗|对打|武打|功夫|武术|搏击|拳击|combat|fight|"
    r"kung\s*fu|martial\s*arts|托马斯|Thomas|铁山靠",
    re.IGNORECASE,
)
_NAMED_TECHNIQUE = re.compile(
    r"托马斯(?:全旋)?|Thomas(?:\s+flare)?|铁山靠|乌龙绞柱|风车|大回环|"
    r"头转|背转|六步|三步|toprock|downrock|flare|windmill|airflare|"
    r"pop(?:ping)?|lock(?:ing)?|wave|moonwalk|扫堂腿|侧踢|回旋踢|"
    r"摆拳|直拳|勾拳|膝撞|肘击|抱摔|过肩摔|咏春摊手|膀手|伏手",
    re.IGNORECASE,
)
_SIDE_OR_VECTOR = re.compile(
    r"左|右|双|前|后|向左|向右|左侧|右侧|顺时针|逆时针|"
    r"left|right|both|forward|backward|clockwise|counterclockwise",
    re.IGNORECASE,
)
_BODY_PART = re.compile(
    r"手|臂|肘|拳|掌|肩|胸|背|腰|躯干|胯|髋|膝|腿|脚|足|头|颈|"
    r"重心|支撑腿|摆动腿|hand|arm|elbow|fist|palm|shoulder|chest|"
    r"torso|hip|knee|leg|foot|head|weight",
    re.IGNORECASE,
)
_KINETIC_VERB = re.compile(
    r"挡|格挡|闪|闪避|侧身|下潜|转|旋|撑|蹬|跨|滑|扫|踢|靠|撞|"
    r"推|拉|抓|扣|锁|摔|跃|跳|落|收|撤|换步|移步|压低|抬高|"
    r"block|parry|dodge|slip|duck|pivot|spin|plant|kick|strike|"
    r"push|pull|grab|lock|throw|jump|land|shift|lean",
    re.IGNORECASE,
)
_UNAMBIGUOUS_BODY_EXECUTION = re.compile(
    r"挡|格挡|闪避|(?<!灯光)闪(?!烁)|侧身|下潜|转身|旋转|撑|蹬|跨步|滑步|扫腿|踢|"
    r"抓|扣|锁|摔|跃|跳|落地|换步|移步|压低|抬腿|突袭|"
    r"block|parry|dodge|slip|duck|pivot|spin|plant|kick|strike|"
    r"grab|lock|throw|jump|land|shift|lean|lunge",
    re.IGNORECASE,
)
_NON_BODY_EFFECT_ACTION = re.compile(
    r"(?:武器|刀刃|能量刃|弹丸|子弹|冲击波|气流|能量|电流|电弧|火花|"
    r"雨滴|碎片|列车|车辆|车门|玻璃|灯光|地面|墙(?:面|壁)?|"
    r"weapon|blade|projectile|bullet|shockwave|airflow|energy|electric|"
    r"spark|raindrop|debris|train|vehicle|door|glass|light|floor|wall)"
    r"[^，。；,.]{0,40}"
    r"(?:撞击|击中|爆发|扩散|席卷|卷起|震动|闪烁|飞散|"
    r"impact|hit|burst|spread|sweep|vibrate|flicker|scatter)",
    re.IGNORECASE,
)
_VAGUE_ACTION = re.compile(
    r"复杂(?:复核|复合)?动作|高难度动作|连续动作|一套动作|完成动作|"
    r"进行(?:舞蹈|格斗|打斗|功夫|武术)|参与舞蹈|跳舞|舞动|齐舞|"
    r"激烈(?:格斗|打斗|搏斗)|连续(?:格斗|打斗|攻击)|双方(?:格斗|打斗|搏斗)|"
    r"格斗动作|功夫动作|武术动作|街舞动作|舞蹈动作|"
    r"complex (?:review )?movement|complex action|dance sequence|fight sequence",
    re.IGNORECASE,
)
_PLACEHOLDER_MECHANICS = re.compile(
    r"^(?:未明确|未指定|未指明|不明确|未知|无|不适用|unspecified|unknown|none|n/?a|-)$",
    re.IGNORECASE,
)
_PLACEHOLDER_MECHANICS_FRAGMENT = re.compile(
    r"未明确|未指定|未指明|不明确|未知|\bunspecified\b|\bunknown\b",
    re.IGNORECASE,
)
_STRUCTURED_MECHANICS_FIELDS = (
    "performer",
    "technique",
    "side",
    "limbs",
    "footwork",
    "torso",
    "weight_shift",
    "direction",
    "contact",
    "end_pose",
)


def _text_values(record: dict[str, Any], fields: Iterable[str]) -> str:
    values: list[str] = []
    for field in fields:
        value = record.get(field)
        if isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value)
        elif isinstance(value, dict):
            values.extend(str(item) for item in value.values())
        elif value not in (None, ""):
            values.append(str(value))
    return " ".join(values)


def requires_explicit_body_choreography(record: dict[str, Any]) -> bool:
    """Return whether a shot/event is a dance, fight, or martial-arts passage."""
    raw_choreography = record.get("body_action_choreography")
    if raw_choreography in (None, "", []):
        raw_choreography = record.get("action_choreography")
    if raw_choreography not in (None, "", []):
        # Once a producer/model asserts a structured body score, every field
        # is authoritative and must be validated even when the surrounding
        # prose omits generic words such as "fight" or "dance".
        return True
    action_ledger = _string_list(record.get("micro_actions"))
    if not action_ledger:
        action_ledger = _string_list(record.get("generation_actions"))
    if action_ledger:
        # Canonical current actions outrank contextual ``what``/``visual``.
        # A fight may remain in the background while this beat only depicts a
        # shield, shockwave or another non-body result.
        if _CHOREOGRAPHY_DOMAIN.search(str(record.get("action_type") or "")):
            return True
        action_text = " ".join(action_ledger)
        if (
            _CHOREOGRAPHY_DOMAIN.search(action_text)
            or _NAMED_TECHNIQUE.search(action_text)
            or _VAGUE_ACTION.search(action_text)
        ):
            return True
        contextual_text = _text_values(record, ("what", "visual"))
        return bool(
            _CHOREOGRAPHY_DOMAIN.search(contextual_text)
            and any(
                _UNAMBIGUOUS_BODY_EXECUTION.search(action)
                and not _NON_BODY_EFFECT_ACTION.search(action)
                for action in action_ledger
            )
        )
    else:
        text = _text_values(
            record,
            (
                "action_type",
                "what",
                "visual",
                "action",
                "action_description",
                "source_excerpt",
            ),
        )
    return bool(_CHOREOGRAPHY_DOMAIN.search(text))


def _action_requires_body_beat(action: str) -> bool:
    """Distinguish performer mechanics from prop/environment consequences."""
    if _NAMED_TECHNIQUE.search(action) or _CHOREOGRAPHY_DOMAIN.search(action):
        return True
    if _VAGUE_ACTION.search(action):
        return True
    if (
        _NON_BODY_EFFECT_ACTION.search(action)
        and not _UNAMBIGUOUS_BODY_EXECUTION.search(action)
    ):
        return False
    return bool(
        _UNAMBIGUOUS_BODY_EXECUTION.search(action)
        or (_BODY_PART.search(action) and _KINETIC_VERB.search(action))
    )


def is_mechanically_specific_action(value: Any) -> bool:
    """Return whether text gives the model observable body mechanics."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text or _VAGUE_ACTION.fullmatch(text):
        return False
    if _NAMED_TECHNIQUE.search(text):
        return True
    # ``左挡`` / ``右闪`` are intentionally valid concise beats.  Longer
    # descriptions can instead specify a limb plus its kinetic operation.
    if _SIDE_OR_VECTOR.search(text) and _KINETIC_VERB.search(text):
        return True
    return bool(_BODY_PART.search(text) and _KINETIC_VERB.search(text))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _missing_structured_mechanics(beat: dict[str, Any]) -> list[str]:
    """Return fields whose values cannot drive an executable body beat."""
    missing: list[str] = []
    for field in _STRUCTURED_MECHANICS_FIELDS:
        value = beat.get(field)
        values = value if isinstance(value, list) else [value]
        normalized = [str(item or "").strip() for item in values]
        if not normalized or any(
            not item
            or _PLACEHOLDER_MECHANICS.fullmatch(item)
            or _PLACEHOLDER_MECHANICS_FRAGMENT.search(item)
            for item in normalized
        ):
            missing.append(field)
    return missing


def _matching_choreography_beats(
    beats: list[dict[str, Any]],
    action: str,
    position: int,
) -> list[dict[str, Any]]:
    normalized_action = re.sub(r"\s+", " ", action).strip()
    matches: list[dict[str, Any]] = []
    for beat in beats:
        beat_action = re.sub(
            r"\s+",
            " ",
            str(beat.get("micro_action") or "").strip(),
        )
        if beat_action and beat_action == normalized_action:
            matches.append(beat)
            continue
        if not beat_action and int(beat.get("micro_action_index") or 0) == position:
            matches.append(beat)
    return matches


def normalize_body_action_choreography(
    value: Any,
    *,
    micro_actions: Iterable[Any] = (),
) -> list[dict[str, Any]]:
    """Normalize LLM/user choreography while preserving concrete language."""
    actions = [str(item).strip() for item in micro_actions if str(item).strip()]
    if isinstance(value, dict):
        value = value.get("beats") or value.get("actions") or []
    if not isinstance(value, list):
        value = []
    beats: list[dict[str, Any]] = []
    for position, raw in enumerate(value, 1):
        if isinstance(raw, str):
            raw = {"description": raw}
        if not isinstance(raw, dict):
            continue
        try:
            action_index = int(raw.get("micro_action_index") or position)
        except (TypeError, ValueError):
            action_index = position
        action_index = max(1, action_index)
        micro_action = str(raw.get("micro_action") or "").strip()
        if not micro_action and action_index <= len(actions):
            micro_action = actions[action_index - 1]
        beat = {
            "beat": len(beats) + 1,
            "micro_action_index": action_index,
            "micro_action": micro_action,
            "performer": str(raw.get("performer") or raw.get("actor") or "").strip(),
            "technique": str(raw.get("technique") or raw.get("move") or "").strip(),
            "side": str(raw.get("side") or raw.get("lead_side") or "").strip(),
            "limbs": _string_list(raw.get("limbs") or raw.get("body_parts")),
            "footwork": str(raw.get("footwork") or "").strip(),
            "torso": str(raw.get("torso") or "").strip(),
            "weight_shift": str(raw.get("weight_shift") or "").strip(),
            "direction": str(raw.get("direction") or "").strip(),
            "contact": str(raw.get("contact") or raw.get("target_contact") or "").strip(),
            "end_pose": str(raw.get("end_pose") or raw.get("result") or "").strip(),
            "description": str(raw.get("description") or "").strip(),
        }
        if any(
            value not in ("", [])
            for key, value in beat.items()
            if key not in {"beat", "micro_action_index"}
        ):
            beats.append(beat)
    return beats


def _beat_prompt(beat: dict[str, Any]) -> str:
    details = []
    labels = (
        ("执行者", "performer"),
        ("招式", "technique"),
        ("侧别", "side"),
        ("肢体", "limbs"),
        ("步法", "footwork"),
        ("躯干", "torso"),
        ("重心", "weight_shift"),
        ("方向", "direction"),
        ("接触", "contact"),
        ("终态", "end_pose"),
    )
    for label, key in labels:
        value = beat.get(key)
        if isinstance(value, list):
            value = "、".join(str(item) for item in value if str(item).strip())
        if str(value or "").strip():
            details.append(f"{label}={value}")
    action = str(beat.get("micro_action") or beat.get("description") or "").strip()
    prefix = f"第{int(beat.get('beat') or 1)}拍"
    return f"{prefix}：{action}" + (f"；{'；'.join(details)}" if details else "")


def build_body_action_contract(record: dict[str, Any]) -> dict[str, Any] | None:
    """Build a durable contract and fail-closed diagnostics for one event/shot."""
    micro_actions = _string_list(record.get("micro_actions"))
    generation_actions = _string_list(record.get("generation_actions"))
    raw = record.get("body_action_choreography")
    if raw in (None, "", []):
        raw = record.get("action_choreography")
    has_structured_choreography = raw not in (None, "", [])
    beats = normalize_body_action_choreography(raw, micro_actions=micro_actions)
    if not beats:
        # Legacy explicit action strings remain usable and auditable. Vague
        # dance/fight summaries deliberately do not get upgraded by guessing.
        beats = normalize_body_action_choreography(
            [
                {"micro_action_index": index, "micro_action": action, "description": action}
                for index, action in enumerate(micro_actions, 1)
                if is_mechanically_specific_action(action)
            ],
            micro_actions=micro_actions,
        )
    required = requires_explicit_body_choreography(record)
    # ``generation_actions`` may combine multiple source actions into one
    # compact arrow-delimited prompt.  Validate the canonical source ledger
    # whenever it exists so prompt compaction cannot create a false failure.
    executable_actions = micro_actions or generation_actions
    incomplete_beats: list[dict[str, Any]] = []
    uncovered_actions: list[str] = []
    vague_actions: list[str] = []
    if has_structured_choreography:
        for beat in beats:
            missing = _missing_structured_mechanics(beat)
            if missing:
                incomplete_beats.append({
                    "beat": int(beat.get("beat") or 0),
                    "micro_action_index": int(
                        beat.get("micro_action_index") or 0
                    ),
                    "micro_action": str(beat.get("micro_action") or ""),
                    "missing_fields": missing,
                })
        for position, action in enumerate(executable_actions, 1):
            matches = _matching_choreography_beats(beats, action, position)
            action_requires_beat = _action_requires_body_beat(action)
            if required and action_requires_beat and not matches:
                uncovered_actions.append(action)
            if matches and not any(
                not _missing_structured_mechanics(beat) for beat in matches
            ):
                vague_actions.append(action)
    else:
        vague_actions = [
            action
            for action in executable_actions
            if _action_requires_body_beat(action)
            and not is_mechanically_specific_action(action)
        ]
    errors: list[dict[str, Any]] = []
    if required and not executable_actions:
        errors.append({
            "code": "body_choreography_actions_missing",
            "message": "dance/combat/martial-arts passage has no ordered executable body actions",
        })
    if required and vague_actions:
        errors.append({
            "code": "body_choreography_vague_action",
            "message": (
                "dance/combat/martial-arts actions must name the side, limbs, technique, "
                "footwork/weight transfer, direction/contact and resulting pose"
            ),
            "actions": vague_actions,
        })
    if required and incomplete_beats:
        errors.append({
            "code": "body_choreography_incomplete_beat",
            "message": (
                "structured body beats must provide executable non-placeholder "
                "performer, technique, side, limbs, footwork, torso, weight shift, "
                "direction, contact and end pose"
            ),
            "beats": incomplete_beats,
        })
    if required and uncovered_actions:
        errors.append({
            "code": "body_choreography_action_uncovered",
            "message": "ordered body actions must each map to a structured choreography beat",
            "actions": uncovered_actions,
        })
    if required and not beats:
        errors.append({
            "code": "body_choreography_beats_missing",
            "message": "no per-beat body mechanics score was supplied",
        })
    if not required and not beats:
        return None
    return {
        "schema": BODY_ACTION_CONTRACT_SCHEMA,
        "required": required,
        "plot_boundary": (
            "choreography may detail movement but must not invent a new prop, injury, winner, "
            "location change, relationship change, or plot outcome"
        ),
        "beats": beats,
        "prompt": " → ".join(_beat_prompt(beat) for beat in beats),
        "forbidden": [
            "复杂动作/连续格斗/跳舞等抽象占位词",
            "只写情绪、速度、镜头运动或最终姿势而不写身体执行过程",
            "左右侧、攻守方、接触点、重心转移或招式顺序漂移",
        ],
        "errors": errors,
        "valid": not errors,
    }


def apply_body_action_contract(record: dict[str, Any]) -> dict[str, Any] | None:
    """Attach a normalized contract without mutating caller-owned nested data."""
    declared_choreography = record.get("body_action_choreography")
    if declared_choreography in (None, "", []):
        declared_choreography = record.get("action_choreography")
    has_declared_choreography = declared_choreography not in (None, "", [])
    contract = build_body_action_contract(record)
    if contract is None:
        record.pop("body_action_contract", None)
        return None
    if has_declared_choreography:
        record["body_action_choreography"] = copy.deepcopy(contract["beats"])
    else:
        # A mechanically specific legacy action may yield a prompt-only
        # compatibility beat with empty typed fields.  Persisting it as if it
        # were authored structured choreography makes a later validation pass
        # mistake compatibility text for an authoritative DTO.
        record.pop("body_action_choreography", None)
    record["body_action_contract"] = contract
    return contract


def body_action_contract_errors(record: dict[str, Any]) -> list[dict[str, Any]]:
    contract = build_body_action_contract(record)
    return list(contract.get("errors") or []) if contract else []


def body_action_prompt(record: dict[str, Any]) -> str:
    contract = record.get("body_action_contract")
    if not isinstance(contract, dict):
        contract = build_body_action_contract(record)
    if not contract:
        return ""
    prompt = str(contract.get("prompt") or "").strip()
    if not prompt:
        return ""
    return (
        "[逐拍肢体动作谱｜不可摘要] "
        f"{prompt}。逐拍保持执行者、左右侧、四肢、步法、躯干、重心、方向、接触点和终态；"
        "禁止用‘复杂动作’‘连续格斗’‘跳舞’或镜头运动替代身体执行。"
    )


def body_action_qa_instruction(record: dict[str, Any]) -> str:
    prompt = body_action_prompt(record)
    if not prompt:
        return ""
    return (
        f"{prompt} QA must verify every beat in order across sampled frames, including lead side, "
        "limb path, foot placement, torso/weight transfer, partner contact and resulting pose. "
        "A generic movement, omitted beat, mirrored side, swapped attacker/defender, or camera-only "
        "motion is a reshoot/revise failure."
    )

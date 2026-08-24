#!/usr/bin/env python3
"""Sequence-aware director intent planning for Phase 1."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from runtime.llm_policy import LLMStreamPolicy
from schemas.understanding import (
    DirectorPlanUnderstanding,
    native_chat_json_schema_format,
    parse_structured_output,
)
from utils.ark_llm import call_llm_stream, create_ark_client
from utils.config import DEFAULT_TEXT_MODEL, get_api_key

DIRECTOR_LLM_POLICY = LLMStreamPolicy.long_structured_output(max_tokens=8000)
# Compatibility aliases for integrations that inspect the Phase 1 limits.
LLM_WALL_TIMEOUT = DIRECTOR_LLM_POLICY.wall_timeout_seconds
LLM_IDLE_TIMEOUT = DIRECTOR_LLM_POLICY.idle_timeout_seconds

DIRECTOR_PLAN_SCHEMA = "honcut.director-plan.v1"
DIRECTOR_INTENT_FIELDS = (
    "scene_goal",
    "emotion_arc",
    "visual_focus",
    "spatial_intent",
    "transition_intent",
)
MAX_SCHEMA_CORRECTIONS = 1

SYSTEM_PROMPT = (
    "你是资深影视导演。事件账本已经完成剧本理解与分段。"
    "你只为每个既有 sequence_id 定义五项导演意图：scene_goal、emotion_arc、"
    "visual_focus、spatial_intent、transition_intent。"
    "不要重新分场，不决定镜头数量、景别、机位角度、运镜、焦段、光影或时长；"
    "这些具体拍摄决策由下游 adaptation engine 负责。"
    "输出严格 JSON，不要输出任何解释文字。"
    "必须逐一覆盖输入中的全部 sequence_id，保持原顺序，不新增、不合并、不遗漏。"
)

USER_PROMPT_TEMPLATE = (
    "请读取以下 canonical 事件账本，为既有 sequence 做导演意图规划。"
    "不要重新分场，也不要输出具体 shot 字段。\n\n"
    "事件账本：\n{events_json}\n\n"
    "输出格式：\n"
    "{{\n"
    f'  "schema": "{DIRECTOR_PLAN_SCHEMA}",\n'
    '  "sequences": [\n'
    "    {{\n"
    '      "sequence_id": "SEQ001",\n'
    '      "scene_goal": "这一段为什么存在",\n'
    '      "emotion_arc": "情绪X→情绪Y",\n'
    '      "visual_focus": "观众应该重点看什么",\n'
    '      "spatial_intent": "人物与环境的空间关系",\n'
    '      "transition_intent": "如何进入下一段或如何收束"\n'
    "    }}\n"
    "  ]\n"
    "}}\n"
)


def _sequence_ids(events: list[dict[str, Any]]) -> list[str]:
    sequence_ids: list[str] = []
    for index, event in enumerate(events, 1):
        if not isinstance(event, dict):
            raise ValueError(f"director event {index} must be an object")
        sequence_id = str(event.get("sequence_id") or "").strip()
        if not sequence_id:
            raise ValueError(f"director event {index} is missing sequence_id")
        if sequence_id not in sequence_ids:
            sequence_ids.append(sequence_id)
    if not sequence_ids:
        raise ValueError("director planning requires at least one sequence")
    return sequence_ids


def validate_director_plan(
    plan: object,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate exact event-ledger coverage and the narrow intent contract."""
    plan = DirectorPlanUnderstanding.model_validate(plan).model_dump(by_alias=True)
    sequences = plan.get("sequences")

    expected_ids = _sequence_ids(events)
    actual_ids: list[str] = []
    for index, sequence in enumerate(sequences, 1):
        sequence_id = str(sequence.get("sequence_id") or "").strip()
        if not sequence_id:
            raise ValueError(f"director sequence {index} has empty sequence_id")
        if sequence_id in actual_ids:
            raise ValueError(f"director sequence_id is duplicated: {sequence_id}")
        actual_ids.append(sequence_id)
        for field in DIRECTOR_INTENT_FIELDS:
            value = sequence.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"director sequence {sequence_id} has empty {field}"
                )

    if actual_ids != expected_ids:
        missing = [value for value in expected_ids if value not in actual_ids]
        unknown = [value for value in actual_ids if value not in expected_ids]
        raise ValueError(
            "director sequence coverage/order mismatch; "
            f"expected={expected_ids}, actual={actual_ids}, "
            f"missing={missing}, unknown={unknown}"
        )
    return plan


def plan_director(
    events: list[dict[str, Any]],
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Phase 1: 导演规划

    Args:
        events: Event Extractor 产出的 canonical 事件账本
        output_dir: 输出目录
        dry_run: dry-run 模式

    Returns:
        director_plan dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "director_plan.json"

    if dry_run:
        print("  ⊘ dry-run 模式，跳过导演规划")
        return {"status": "skipped", "reason": "dry-run"}

    # 调用 LLM。生产导演规划是 Phase 1 的必需证据，失败必须向上冒泡。
    try:
        api_key = get_api_key("ARK_AGENT_API_KEY")
        if not api_key:
            raise RuntimeError("director planning requires ARK_AGENT_API_KEY")

        client = create_ark_client(read_timeout=LLM_IDLE_TIMEOUT)
        _sequence_ids(events)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            events_json=json.dumps(events, ensure_ascii=False, indent=2)
        )

        correction = ""
        plan = None
        for attempt in range(MAX_SCHEMA_CORRECTIONS + 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt + correction,
                },
            ]
            content = call_llm_stream(
                messages=messages,
                model=DEFAULT_TEXT_MODEL,
                max_tokens=DIRECTOR_LLM_POLICY.max_tokens,
                wall_timeout=DIRECTOR_LLM_POLICY.wall_timeout_seconds,
                idle_timeout=DIRECTOR_LLM_POLICY.idle_timeout_seconds,
                response_format=native_chat_json_schema_format(
                    DirectorPlanUnderstanding
                ),
                _client=client,
            )
            if not content:
                raise ValueError("LLM 返回空内容")
            try:
                parsed = parse_structured_output(
                    content,
                    DirectorPlanUnderstanding,
                ).model_dump(by_alias=True)
                plan = validate_director_plan(parsed, events)
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                if attempt >= MAX_SCHEMA_CORRECTIONS:
                    raise
                correction = (
                    "\n\n上次输出未通过导演计划业务校验："
                    f"{exc}。请重新输出完整 JSON，严格覆盖全部 sequence_id，"
                    "保持原顺序且不要增加任何 shot 字段。"
                )
        if plan is None:
            raise ValueError("director planning did not produce a validated plan")

        # 写入文件
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

        sequences = plan["sequences"]
        print(f"  ✓ [M1] 导演意图完成: {len(sequences)} 个 sequence")
        for sequence in sequences:
            print(
                f"    {sequence['sequence_id']}: {sequence['scene_goal']} "
                f"({sequence['emotion_arc']})"
            )

        return {"status": "done", "plan": plan, "output": str(plan_path)}

    except Exception as exc:
        plan_path.unlink(missing_ok=True)
        raise RuntimeError(f"director planning failed: {exc}") from exc

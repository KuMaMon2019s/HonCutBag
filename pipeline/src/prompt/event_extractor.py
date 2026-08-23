#!/usr/bin/env python3
"""
事件提取器 - Phase 1 编剧引擎的事件提取模块

从 text_parser.py 输出的 segments 中提取结构化事件。
每个事件包含：谁(who)、在哪(where)、做什么(what)、情绪(emotion)、视觉描述(visual)。

输入：text_parser.py 的输出 JSON（segments 列表），或直接传入文本
输出：JSON 格式的结构化事件列表

逻辑：
1. 接收 segments（从 text_parser 输出或直接文本）
2. 对每个 segment，调用 LLM 提取事件
3. 一个 segment 可能产生多个事件
4. 解析或瞬时流错误重试 1 次，仍失败则终止，禁止静默丢事件
5. 合并所有 segment 的事件，重新编号
6. 输出 JSON
"""

import json
import sys
import os
import argparse
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from utils.action_units import annotate_event_motion_modes
from utils.body_action_contracts import (
    apply_body_action_contract,
    normalize_body_action_choreography,
)
from utils.ark_llm import (
    LLMConnectTimeout,
    LLMIdleTimeout,
    LLMReadTimeout,
    LLMStreamError,
    LLMWallTimeout,
    call_llm_stream,
    create_ark_client,
)


# ─── LLM 配置 ───────────────────────────────────────────────────────────────

GENERAL_SYSTEM_PROMPT = (
    "你是影视编剧与连续性编辑。从文本中提取可拍摄的叙事事件。"
    "事件不是镜头：不要把每句话或每个招式机械拆成一个事件，镜头划分由下游导演完成。"
    "你必须保留动作的起始状态、结束状态、因果关系和无署名对白的可靠归属。输出严格 JSON 数组。"
    "不要输出任何解释文字，只输出 JSON。"
)

ACTION_SYSTEM_PROMPT = (
    "你是动作影视编剧与连续性编辑。从动作型文本中提取可拍摄的因果动作单元。"
    "事件不是镜头：不要把每句话或每个招式机械拆成一个事件，镜头划分由下游导演完成。"
    "micro_actions 是按时间先后生成的可执行动作阶段，不是‘复杂动作’‘跳舞’‘连续格斗’等抽象标签；"
    "同一时刻并行完成的复合动作必须合成一条。"
    "舞蹈、格斗、功夫或武术段落必须进一步输出逐拍 body_action_choreography，明确左右侧、"
    "执行肢体、步法、躯干、重心、方向、接触点和终态。"
    "你必须保留动作的起始状态、结束状态、因果关系和无署名对白的可靠归属。输出严格 JSON 数组。"
    "不要输出任何解释文字，只输出 JSON。"
)

# Backward-compatible export for integrations that imported the old constant.
SYSTEM_PROMPT = GENERAL_SYSTEM_PROMPT

USER_PROMPT_TEMPLATE = (
    "文档类型：{format_hint}\n\n"
    "以下前后文只用于判断人物、对白归属和动作承接，严禁从前后文重复提取事件：\n"
    "<context_before>\n{context_before}\n</context_before>\n"
    "<target>\n{content}\n</target>\n"
    "<context_after>\n{context_after}\n</context_after>\n\n"
    "只从 <target> 提取事件。输出 JSON 数组，每个元素包含：\n"
    "- who: 数组，只写可跨事件复用的稳定身份标签；服装、年龄、伤势、动作、站位和地点修饰不得进入 who，"
    "应写入 visual/start_state/end_state。相同人物在 target 与上下文中必须沿用同一标签；"
    "不得把多词姓名截短，也不得用主角/他/她替换已有标签\n"
    "- where: 字符串，地点\n"
    "- what: 字符串，发生了什么\n"
    "- emotion: 字符串，情绪氛围\n"
    "- visual: 字符串，描述画面（用于生成视频镜头）\n"
    "- time: 字符串，时间/季节\n"
    "- action_type: 字符串，事件类型（discovery/conflict/resolution/transition 等）\n"
    "- event_role: 字符串，只能是 scene_setup/character_state/dialogue/action_chain/reaction/consequence/turning_point/transition\n"
    "- source_excerpt: 字符串，逐字摘录 <target> 中支撑本事件的连续原文\n"
    "- micro_actions: 字符串数组，按发生顺序列出本动作单元中的可见动作；非动作事件为 []\n"
    "- body_action_choreography: 数组；仅舞蹈/格斗/功夫/武术段落必填，每项必须包含 "
    "micro_action_index、performer、technique、side、limbs（数组）、footwork、torso、"
    "weight_shift、direction、contact、end_pose；非此类事件为 []\n"
    "- generation_motion_mode: 字符串，只能是 none/atomic/composite；micro_actions=[] 时为 none；"
    "需按先后执行的动作阶段为 atomic；原文明确说明同一时刻并行完成、"
    "融为一个整体且不是逐个执行时才为 composite\n"
    "- action_phase: 字符串，只能是 none/setup/attack/counter/impact/recovery/consequence\n"
    "- start_state: 字符串，本单元开始时人物、武器、空间与运动状态\n"
    "- end_state: 字符串，本单元结束时可供下一段承接的定格状态\n"
    "- causal_link: 字符串，说明本单元由上一事件的什么动作或决定引发；无则为空字符串\n"
    "- continuity_before: 字符串，cut/continuous；只有同一时空且状态直接承接才为 continuous\n"
    "- continuity_subject: 字符串，continuous 时跨单元跟踪的主要人物或物体，否则为空字符串\n"
    "- dramatic_turn: 布尔值，只允许 turning_point 为 true；其他 event_role 必须为 false\n"
    "- lines: 数组，本事件中角色说出的台词原文，每条为 "
    "{{\"speaker\": \"角色名或未知\", \"line\": \"逐字台词\", "
    "\"confidence\": 0到1, \"evidence\": \"归属依据\"}}；无台词时为空数组 []\n"
    "line 必须逐字保留剧本原文，禁止改写、摘要或翻译。\n"
    "剧本对白可能写作 角色名：\"台词\" 或 角色名:\"台词\"，全角/半角冒号与引号均可能出现。"
    "只有证据充分才填写角色名；若只能猜测，speaker 写‘未知’并降低 confidence，禁止为了完整而编造。\n\n"
    "{format_contract}"
)

GENERAL_PROSE_CONTRACT = (
    "【通用叙事规则】\n"
    "1. 场景建立、人物状态、对白、行为、反应、后果与关系变化按叙事功能划分 event_role。\n"
    "2. 只在原文明确描述可见行为时填写 micro_actions；氛围、说明与内心信息不得虚构肢体动作。\n"
    "3. 对人物、物体或空间造成的持久变化必须进入 end_state，供后续事件承接。\n"
    "4. 目标、关系、认知或处境发生转折时单列 turning_point，并设置 dramatic_turn=true；"
    "所有非 turning_point 事件必须设置 dramatic_turn=false。\n"
    "5. 同一时空的状态直接承接才使用 continuous；换场、跳时或独立叙事段落使用 cut。\n"
    "6. who 只放可作为角色资产的具名个体；群体与背景参与者写入 visual，不得写入 who。\n"
    "7. who 的每个值必须是稳定身份标签，不得包含服装、年龄、伤势、动作、站位或地点修饰；"
    "同一人物必须沿用 target/上下文中已有的最短无歧义标签。"
)

ACTION_SCREENPLAY_CONTRACT = (
    "【动作型叙事规则】\n"
    "1. 场景建立、人物当前状态、对白、动作链、反应、后果和叙事转折是不同 event_role。\n"
    "2. micro_actions 只表示视频模型需要按时间先后完成的可见动作阶段，"
    "不得写‘复杂动作’‘复合动作’‘跳舞’‘激烈格斗’‘连续攻击’等不可执行占位词。"
    "原文明确说明多个贡献在同一时刻并行完成、"
    "融为一个整体且并非逐个执行时，必须合成一条复合 micro_action 并设为 composite；"
    "‘连贯衔接’或‘一气呵成’本身不代表同时发生；原文明示先、随后、逐渐、最终等状态变化时仍须拆成多条。\n"
    "3. 动作造成的人物、物体、空间、朝向、速度或受力状态变化必须写入 end_state。\n"
    "4. 目标、立场、关系或局势发生变化时单列 turning_point，并设置 dramatic_turn=true；"
    "所有非 turning_point 事件必须设置 dramatic_turn=false。\n"
    "5. 相邻事件的位置、朝向、速度和受力状态直接延续时使用 continuous；"
    "换场、跳时或独立叙事段落使用 cut。\n"
    "6. who 只放可作为角色资产的具名个体；群体与背景参与者写入 visual，不得写入 who。\n"
    "7. who 的每个值必须是稳定身份标签，不得包含服装、年龄、伤势、动作、站位或地点修饰；"
    "同一人物必须沿用 target/上下文中已有的最短无歧义标签。\n"
    "8. 舞蹈、格斗、功夫、武术或搏击段落必须逐拍填写 body_action_choreography："
    "每拍明确执行者、左右侧、肢体路径、步法、躯干旋转、重心转移、运动方向、接触点和终态。"
    "原文点名的招式（例如街舞托马斯、铁山靠）必须逐字保留；原文只写泛化表演或交手时，"
    "允许在不改变人物、道具、伤亡、胜负、地点和剧情结果的边界内补足可拍摄编舞，例如左挡、"
    "右闪、换步、支撑腿、摆动腿和受力终态。禁止只写动作难度、速度、情绪或镜头效果。\n"
    "9. 风格说明、摄影约束、负面约束、角色一致性要求和对前文剧情的总结不是新的时间线动作或事件；"
    "不得为它们输出 scene_setup/character_state/transition，也不得把已发生的剧情再提取一遍。"
)

LLM_TIMEOUT = 900  # 健康大段落长流可超过 300s；空闲停滞仍由 75s 独立阈值处理
LLM_IDLE_TIMEOUT = 75
MAX_RETRIES = 1  # 解析失败重试次数


class EventExtractionError(RuntimeError):
    """A non-empty source segment could not be extracted completely."""


def _get_client() -> OpenAI:
    """
    创建 OpenAI 客户端

    API Key: ARK_AGENT_API_KEY (火山方舟 Agent Plan)

    Returns:
        OpenAI 客户端实例

    Raises:
        SystemExit: 如果所有 API key 环境变量均未设置
    """
    return create_ark_client(read_timeout=LLM_IDLE_TIMEOUT)


def _call_llm(prompt: str, system_prompt: str = GENERAL_SYSTEM_PROMPT) -> str:
    """
    调用 LLM 并返回原始响应文本

    Args:
        prompt: 用户 prompt（segment 内容）

    Returns:
        LLM 原始响应字符串

    Raises:
        Exception: API 调用失败时抛出
    """
    client = _get_client()

    return call_llm_stream(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=16000,
        wall_timeout=LLM_TIMEOUT,
        idle_timeout=LLM_IDLE_TIMEOUT,
        _client=client,
    )


_EVENT_ROLES = {
    "scene_setup", "character_state", "dialogue", "action_chain", "reaction",
    "consequence", "turning_point", "transition",
}
_ACTION_PHASES = {"none", "setup", "attack", "counter", "impact", "recovery", "consequence"}
_GROUP_PARTICIPANT_RE = re.compile(
    r"(?:数[十百千]|机械(?:单位|身影|部队|群)|群众|人群|群体|军队|部队|居民群|敌群)"
)
_GLOBAL_DIRECTIVE_SCOPE_RE = re.compile(
    r"(?:全程|始终|所有镜头|每(?:个|一)镜头|throughout|every\s+shot)",
    re.IGNORECASE,
)
_GLOBAL_DIRECTIVE_IMPERATIVE_RE = re.compile(
    r"(?:只使用|保持|禁止|不得|必须|避免|不要|不允许|must|avoid|forbid|without|\bno\b)",
    re.IGNORECASE,
)
_GLOBAL_DIRECTIVE_PRODUCTION_RE = re.compile(
    r"(?:摄影机|运镜|镜头|画面|字幕|水印|logo|风格|质感|人物变形|脸部变化|"
    r"手指|肢体错误|颜色漂移|同一张脸|同一发型|身材比例|character\s+consistency|"
    r"camera|subtitle|watermark)",
    re.IGNORECASE,
)
_NARRATIVE_JUMP_CUES = (
    "与此同时", "另一边", "次日", "翌日", "后来", "数小时后", "多年后", "回忆",
    "梦境", "转场", "来到", "抵达", "离开当前", "meanwhile", "later", "next day",
)
# Cache-invalidation contract: this version is mixed into every segment_hash
# (see below), and cached files are keyed by that hash. Bumping this value
# therefore invalidates ALL existing phase1_event_segments/ caches — including
# caches carried over from earlier run directories — and forces every Phase 1
# event extraction to re-run (a paid LLM pass). Never bump casually, and never
# reuse cross-run segment caches from a run produced under a different value.
EVENT_FLOW_SCHEMA_VERSION = "12.0"


def is_global_production_directive_text(evidence: str) -> bool:
    """Classify project-wide visual rules without inventing a timeline event."""
    return bool(
        _GLOBAL_DIRECTIVE_SCOPE_RE.search(evidence)
        and _GLOBAL_DIRECTIVE_IMPERATIVE_RE.search(evidence)
        and _GLOBAL_DIRECTIVE_PRODUCTION_RE.search(evidence)
    )


def _is_non_narrative_directive_candidate(event: Dict[str, Any]) -> bool:
    """Whether an extracted record has no action/dialogue story-clock evidence."""
    if event.get("micro_actions") or event.get("lines"):
        return False
    return event.get("event_role") in {
        "scene_setup", "character_state", "transition"
    }


def _is_global_production_directive(event: Dict[str, Any]) -> bool:
    """Return true for project-wide visual rules that have no story-clock beat."""
    if not _is_non_narrative_directive_candidate(event):
        return False
    evidence = str(event.get("source_excerpt") or "")
    return is_global_production_directive_text(evidence)


def _normalize_event(event: Dict[str, Any], source_content: str = "") -> Dict[str, Any]:
    who = event.get("who", [])
    if isinstance(who, str):
        who = [who] if who.strip() else []
    normalized_who = [str(name).strip() for name in who if str(name).strip()] if isinstance(who, list) else []
    background_groups = [name for name in normalized_who if _GROUP_PARTICIPANT_RE.search(name)]
    event["who"] = [name for name in normalized_who if name not in background_groups]
    if background_groups:
        event["background_groups"] = background_groups

    role = str(event.get("event_role") or "").strip().lower()
    if role not in _EVENT_ROLES:
        action_type = str(event.get("action_type") or "").lower()
        if action_type in {"conflict", "action", "fight", "chase"}:
            role = "action_chain"
        elif action_type in {"resolution", "reversal"}:
            role = "turning_point"
        elif action_type == "transition":
            role = "transition"
        else:
            role = "scene_setup"
    event["event_role"] = role

    micro_actions = event.get("micro_actions", [])
    if isinstance(micro_actions, str):
        micro_actions = [micro_actions] if micro_actions.strip() else []
    event["micro_actions"] = [str(item).strip() for item in micro_actions if str(item).strip()] if isinstance(micro_actions, list) else []
    event["body_action_choreography"] = normalize_body_action_choreography(
        event.get("body_action_choreography") or event.get("action_choreography"),
        micro_actions=event["micro_actions"],
    )
    apply_body_action_contract(event)
    motion_mode = str(event.get("generation_motion_mode") or "").strip().lower()
    if not event["micro_actions"]:
        event["generation_motion_mode"] = "none"
    else:
        event["generation_motion_mode"] = (
            motion_mode if motion_mode in {"atomic", "composite"} else "atomic"
        )
    phase = str(event.get("action_phase") or "none").strip().lower()
    event["action_phase"] = phase if phase in _ACTION_PHASES else "none"
    for field in ("start_state", "end_state", "causal_link", "continuity_subject", "source_excerpt"):
        event[field] = str(event.get(field) or "").strip()
    boundary = str(event.get("continuity_before") or "cut").strip().lower()
    event["continuity_before"] = boundary if boundary in {"cut", "continuous"} else "cut"
    dramatic_turn = event.get("dramatic_turn", role == "turning_point")
    if not isinstance(dramatic_turn, bool):
        raise ValueError("dramatic_turn 必须是 JSON 布尔值")
    expected_turn = role == "turning_point"
    if dramatic_turn is not expected_turn:
        raise ValueError(
            "event_role 与 dramatic_turn 冲突：只有 turning_point 可以为 true"
        )
    event["dramatic_turn"] = dramatic_turn

    lines = event.get("lines", [])
    normalized_lines = []
    if isinstance(lines, list):
        for raw in lines:
            if not isinstance(raw, dict) or not str(raw.get("line") or "").strip():
                continue
            line = str(raw["line"]).strip()
            if source_content and line not in source_content:
                raise ValueError(f"台词不是 target 中的逐字原文: {line}")
            try:
                confidence = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            speaker = str(raw.get("speaker") or "未知").strip() or "未知"
            normalized_lines.append({
                "speaker": speaker,
                "line": line,
                "confidence": confidence,
                "evidence": str(raw.get("evidence") or "").strip(),
            })
    event["lines"] = normalized_lines
    return event


def _parse_events(response: str, source_content: str = "") -> List[Dict[str, Any]]:
    """
    解析 LLM 响应为事件列表

    尝试从响应中提取 JSON 数组。支持：
    - 纯 JSON 数组
    - 被 ```json ... ``` 包裹的 JSON
    - 被 ``` ... ``` 包裹的 JSON

    Args:
        response: LLM 原始响应字符串

    Returns:
        事件字典列表

    Raises:
        ValueError: 无法解析为有效 JSON 数组
    """
    text = response.strip()

    # 尝试提取 ```json ... ``` 代码块
    if "```" in text:
        # 匹配 ```json 或 ``` 包裹的内容
        import re
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # 尝试解析 JSON
    parsed = json.loads(text)

    # 验证是数组
    if not isinstance(parsed, list):
        raise ValueError(f"期望 JSON 数组，得到 {type(parsed).__name__}")

    # 验证每个元素的基本结构
    required_fields = {"who", "where", "what", "emotion", "visual", "time", "action_type"}
    for i, event in enumerate(parsed):
        if not isinstance(event, dict):
            raise ValueError(f"第 {i+1} 个事件不是字典")
        missing = required_fields - set(event.keys())
        if missing:
            raise ValueError(f"第 {i+1} 个事件缺少字段: {missing}")
        _normalize_event(event, source_content)

    # A model may quote only a small visual fragment from a paragraph that is
    # wholly a project-wide production directive.  The fragment then loses the
    # scope/imperative tokens required by the local classifier.  Drop the whole
    # response only when the complete source paragraph is a directive and every
    # extracted record is action/dialogue-free; a mixed narrative paragraph is
    # preserved instead of being guessed away.
    if (
        parsed
        and is_global_production_directive_text(source_content)
        and all(_is_non_narrative_directive_candidate(event) for event in parsed)
    ):
        return []
    return [event for event in parsed if not _is_global_production_directive(event)]


def _extract_events_from_segment(segment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从单个 segment 中提取事件

    调用 LLM 提取事件，解析失败则重试 1 次。

    Args:
        segment: text_parser 输出的单个 segment 字典

    Returns:
        事件列表（有效响应可以为空）

    Raises:
        EventExtractionError: 非空 segment 在重试后仍无法完整提取
    """
    content = segment.get("content", "")
    if not content.strip():
        return []

    format_hint = str(segment.get("format_hint") or "general_prose")
    is_action_format = format_hint == "prose_action_screenplay"
    prompt = USER_PROMPT_TEMPLATE.format(
        content=content,
        format_hint=format_hint,
        context_before=segment.get("context_before", ""),
        context_after=segment.get("context_after", ""),
        format_contract=(
            ACTION_SCREENPLAY_CONTRACT if is_action_format else GENERAL_PROSE_CONTRACT
        ),
    )
    system_prompt = ACTION_SYSTEM_PROMPT if is_action_format else GENERAL_SYSTEM_PROMPT

    last_error = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            attempt_prompt = prompt
            if attempt and last_error:
                attempt_prompt += (
                    "\n\n【重试纠错】上次响应未通过 canonical event schema："
                    f"{last_error}。event_role 与 dramatic_turn 必须严格对应："
                    "turning_point=true，其他 event_role=false。"
                )
            response = _call_llm(attempt_prompt, system_prompt=system_prompt)
            events = _parse_events(response, content)
            return events
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"  JSON 解析失败，重试 ({attempt+1}/{MAX_RETRIES}): {e}", file=sys.stderr)
                time.sleep(1)
            continue
        except (
            LLMConnectTimeout,
            LLMReadTimeout,
            LLMIdleTimeout,
            LLMWallTimeout,
            LLMStreamError,
        ) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"  LLM 流中断，重试 ({attempt+1}/{MAX_RETRIES}): {e}", file=sys.stderr)
                time.sleep(1)
                continue
            break
        except Exception as e:
            raise EventExtractionError(
                f"segment {segment.get('id', '?')} LLM 调用失败: {e}"
            ) from e

    raise EventExtractionError(
        f"segment {segment.get('id', '?')} 事件提取失败（已重试 {MAX_RETRIES} 次）: {last_error}"
    ) from last_error


def extract_events(
    segments: list[dict[str, Any]],
    checkpoint_dir: str | Path | None = None,
    *,
    continuity_mode: str | None = None,
) -> Dict[str, Any]:
    """
    核心函数：从 segments 列表中提取所有事件

    对每个 segment 调用 LLM 提取事件，合并后重新编号。

    Args:
        segments: text_parser 输出的 segments 列表

    Returns:
        包含 total_events 和 events 的字典
    """
    if not segments:
        return {"total_events": 0, "events": []}

    all_events = []
    event_id = 1

    segment_cache_dir = Path(checkpoint_dir) / "phase1_event_segments" if checkpoint_dir else None
    if segment_cache_dir is not None:
        segment_cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_one(segment):
        segment_id = segment.get("id", 0)
        segment_hash = hashlib.sha256(
            json.dumps(
                {
                    "event_extraction_schema_version": EVENT_FLOW_SCHEMA_VERSION,
                    "segment": segment,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cache_path = (
            segment_cache_dir / f"segment_{segment_id}_{segment_hash[:16]}.json"
            if segment_cache_dir is not None else None
        )
        if cache_path is not None and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                events = _parse_events(
                    json.dumps(cached.get("events", []), ensure_ascii=False),
                    str(segment.get("content", "")),
                )
                print(f"复用 segment {segment_id} 事件缓存...", file=sys.stderr)
                return segment_id, events
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                pass
        print(f"处理 segment {segment_id}...", file=sys.stderr)
        events = _extract_events_from_segment(segment)
        if cache_path is not None:
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps({"segment_hash": segment_hash, "events": events}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, cache_path)
        return segment_id, events

    source_segments = [segment for segment in segments if str(segment.get("content", "")).strip()]
    with ThreadPoolExecutor(max_workers=3) as executor:
        ordered_results = list(executor.map(extract_one, source_segments))

    for segment_id, events in ordered_results:

        for event in events:
            event["id"] = event_id
            event["segment_id"] = segment_id
            all_events.append(event)
            event_id += 1

    _annotate_global_event_flow(all_events, continuity_mode=continuity_mode)

    source_hash = hashlib.sha256(
        json.dumps(source_segments, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": EVENT_FLOW_SCHEMA_VERSION,
        "continuity_mode": continuity_mode,
        "document_format": next(
            (str(segment.get("format_hint")) for segment in source_segments if segment.get("format_hint")),
            "general_prose",
        ),
        "source_segments_hash": source_hash,
        "source_segment_count": len(source_segments),
        "covered_segment_ids": [segment_id for segment_id, _events in ordered_results],
        "total_events": len(all_events),
        "events": all_events,
    }


def _annotate_global_event_flow(
    events: list[dict[str, Any]],
    *,
    continuity_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Assign deterministic global sequence/action/dialogue identities in source order.

    An explicit one-take direction is a source-level continuity fact. Segment-local
    extraction may still call later paragraphs a ``cut`` because their location is
    paraphrased (``street``, ``road ahead``, ``sidewalk``). In one-take mode those
    paragraph boundaries must not become unrelated screenplay sequences.
    """
    preserve_one_take = str(continuity_mode or "").strip().lower() in {
        "one_take",
        "single_take",
        "oner",
    }
    annotate_event_motion_modes(events)
    sequence_number = 0
    action_number = 0
    dialogue_number = 0
    previous: Dict[str, Any] | None = None
    current_sequence = ""
    for index, event in enumerate(events, 1):
        boundary = str(event.get("continuity_before") or "cut").lower()
        exact_same_place = bool(
            previous
            and str(previous.get("where") or "").strip()
            and str(previous.get("where") or "").strip() == str(event.get("where") or "").strip()
        )
        compatible_place = bool(previous and _locations_compatible(previous, event))
        if previous and preserve_one_take and not _has_narrative_jump(event):
            if boundary != "continuous":
                event["model_continuity_before"] = boundary
                event["continuity_repair_reason"] = (
                    "explicit one-take source keeps all events in one screenplay sequence"
                )
            boundary = "continuous"
        elif previous and boundary == "cut" and _should_repair_cross_segment_boundary(previous, event):
            event["model_continuity_before"] = "cut"
            event["continuity_repair_reason"] = (
                "cross-segment causal action state continues despite location wording drift"
            )
            boundary = "continuous"
        if index == 1 or boundary != "continuous" or (
            not preserve_one_take and not (exact_same_place or compatible_place)
        ):
            boundary = "cut"
            sequence_number += 1
            current_sequence = f"SEQ{sequence_number:03d}"
        event["continuity_before"] = boundary
        event["sequence_id"] = current_sequence

        if event.get("event_role") in {"action_chain", "reaction", "consequence", "turning_point"} and (
            event.get("micro_actions") or event.get("action_phase") != "none"
        ):
            action_number += 1
            event["action_unit_id"] = f"AU{action_number:03d}"
        else:
            event["action_unit_id"] = None
        for line in event.get("lines", []):
            dialogue_number += 1
            line["dialogue_id"] = f"D{dialogue_number:03d}"
        previous = event
    return events


def _locations_compatible(previous: Dict[str, Any], current: Dict[str, Any]) -> bool:
    previous_where = str(previous.get("where") or "")
    current_where = str(current.get("where") or "")

    def tokens(value: str) -> set[str]:
        normalized = re.sub(r"\s+", "", value.casefold())
        word_tokens = set(re.findall(r"[a-z0-9_]{3,}", normalized))
        cjk_runs = re.findall(r"[\u3400-\u9fff]{2,}", normalized)
        cjk_bigrams = {
            run[index:index + 2]
            for run in cjk_runs
            for index in range(len(run) - 1)
        }
        return word_tokens | cjk_bigrams

    previous_tokens = tokens(previous_where)
    current_tokens = tokens(current_where)
    return bool(previous_tokens & current_tokens)


def _should_repair_cross_segment_boundary(
    previous: Dict[str, Any], current: Dict[str, Any]
) -> bool:
    if previous.get("segment_id") == current.get("segment_id"):
        return False
    if current.get("event_role") in {"scene_setup", "transition"}:
        return False
    if _has_narrative_jump(current):
        return False
    previous_who = set(previous.get("who") or [])
    current_who = set(current.get("who") or [])
    shared_subject = bool(previous_who & current_who)
    causal = bool(str(current.get("causal_link") or "").strip())
    return shared_subject and causal and _locations_compatible(previous, current)


def _has_narrative_jump(event: dict[str, Any]) -> bool:
    combined = " ".join(
        str(event.get(field) or "")
        for field in ("what", "start_state", "causal_link", "source_excerpt")
    )
    for cue in _NARRATIVE_JUMP_CUES:
        start = 0
        while True:
            position = combined.find(cue, start)
            if position < 0:
                break
            prefix = combined[max(0, position - 16):position]
            if not re.search(
                r"(?:不|不要|不得|禁止|避免|没有|并无|无)\s*"
                r"(?:发生|进行|使用|出现|允许)?\s*$",
                prefix,
            ):
                return True
            start = position + len(cue)
    return False


def extract_events_from_text(text: str) -> Dict[str, Any]:
    """
    从原始文本提取事件（内部调用 text_parser）

    Args:
        text: 原始文本

    Returns:
        事件提取结果字典
    """
    # 导入 text_parser
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prompt.text_parser import parse_text

    parsed = parse_text(text)
    segments = parsed.get("segments", [])

    return extract_events(segments)


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="事件提取器 - 从文本/segments 中提取结构化事件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从 text_parser 输出文件读取
  python event_extractor.py --input parsed.json --output events.json

  # 从文本直接提取（内部调用 text_parser）
  python event_extractor.py --text "艾米在雪地里找到了一只受伤的小狼" --output events.json

  # 管道模式
  python text_parser.py --text "..." | python event_extractor.py --output events.json
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入文件路径（text_parser 输出的 JSON）",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出 JSON 文件路径（可选，默认打印到 stdout）",
    )

    parser.add_argument(
        "--text", "-t",
        type=str,
        help="直接传入文本（内部调用 text_parser）",
    )

    args = parser.parse_args()

    # 确定输入来源
    segments = None

    if args.text:
        # 从文本直接提取
        result = extract_events_from_text(args.text)
    elif args.input:
        # 从文件读取 segments
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"错误：文件不存在 - {args.input}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"错误：JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        # 支持直接传入 segments 列表或 parse_text 的完整输出
        if isinstance(data, list):
            segments = data
        elif isinstance(data, dict) and "segments" in data:
            segments = data["segments"]
        else:
            print("错误：输入格式不支持，期望 segments 列表或包含 segments 的字典", file=sys.stderr)
            sys.exit(1)

        result = extract_events(segments)
    elif not sys.stdin.isatty():
        # 从 stdin 读取（管道模式）
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"错误：stdin JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        if isinstance(data, list):
            segments = data
        elif isinstance(data, dict) and "segments" in data:
            segments = data["segments"]
        else:
            print("错误：输入格式不支持", file=sys.stderr)
            sys.exit(1)

        result = extract_events(segments)
    else:
        parser.print_help()
        sys.exit(1)

    # 输出 JSON
    json_output = json.dumps(result, ensure_ascii=False, indent=2)
    print(json_output)

    # 写入文件（如果指定）
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"\n已写入文件：{args.output}", file=sys.stderr)
        except IOError as e:
            print(f"错误：无法写入文件 - {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

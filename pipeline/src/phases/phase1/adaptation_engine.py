#!/usr/bin/env python3
"""
改编引擎 - Phase 1 编剧引擎的影视化改编模块

将事件列表改编为适合视频制作的 shot 列表。
决定哪些事件保留、哪些合并、哪些删减、顺序如何。

输入：
- events JSON（event_extractor.py 输出）
- characters JSON（character_discoverer.py 输出，可选）
- 目标时长（默认 60 秒）
- 每镜时长（默认 12 秒）

输出：
- shots JSON（包含 shot_order, source_events, action, reason, who, where,
  what, emotion, visual, suggested_duration, transition_to_next）

逻辑：
1. 计算可用时长：target_duration / avg_shot_duration = 最大 shot 数
2. 调用 LLM 做改编决策：
   - keep：重要事件保留
   - merge：相似/连续事件合并
   - drop：重复/不重要事件删减
   - expand：关键情感时刻可扩展为多镜
3. 为每个 shot 分配时长（总计不超过 target_duration）
4. 建议转场方式（cut/dissolve/fade）
5. 输出排序后的 shot 列表
"""

import json
import sys
import os
import argparse
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI, APITimeoutError
from utils.ark_llm import (
    LLMConnectTimeout,
    LLMIdleTimeout,
    LLMReadTimeout,
    LLMStreamError,
    call_llm_stream,
    create_ark_client,
)


# ─── LLM 配置 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是影视导演兼剪辑师。将故事事件改编为视频分镜。"
    "考虑：节奏感、情感弧线、视觉变化、时长限制。"
    "输出严格 JSON，不要输出任何解释文字。"
)

USER_PROMPT_TEMPLATE = (
    "目标时长：{target_duration}秒，每镜约{shot_duration}秒，最多{max_shots}个镜头。"
    "每镜只完成一个明确情节。\n\n"
    "事件列表：\n{events_json}\n\n"
    "角色列表：\n{characters_summary}\n\n"
    "请输出一个 JSON 对象，包含：\n"
    "- strategy: 字符串，改编策略概述（一句话说明你的改编思路）\n"
    "- shots: JSON 数组，每个 shot 包含：\n"
    "  - shot_order: 整数，镜头序号（从 1 开始）\n"
    "  - source_events: 整数数组，来源事件编号\n"
    "  - action: 字符串，keep/merge/drop/expand\n"
    "  - reason: 字符串，改编理由\n"
    "  - who: 字符串数组，出场角色名（空数组 [] 表示纯风景/无角色镜头）\n"
    "  - where: 字符串，地点\n"
    "  - what: 字符串，发生了什么\n"
    "  - emotion: 字符串，情绪/情感\n"
    "  - visual: 字符串，画面描述（用于视频生成）\n"
    "  - suggested_duration: 整数，建议时长（秒）\n"
    "  - boundary_before: 字符串，cut/continuous；只有同一时空、同一主体且动作状态直接承接时才能为 continuous\n"
    "  - continuity_reason: 字符串，说明为何延长上一视频或为何重新起镜\n"
    "  - continuity_subject: 字符串，continuous 时需要跨镜跟踪的主要人物或物体\n"
    "  - transition_to_next: 字符串，转场方式 cut/dissolve/fade\n"
    "  - associate_assets: 字符串数组，该镜头涉及的资产ID（格式 'char:角色id' 或 'scene:场景名'）\n"
    "  - shot_size: 字符串，景别（extreme_wide/wide/medium_wide/medium/medium_close/close_up/extreme_close_up/over_shoulder/insert/establishing）\n"
    "  - camera_movement: 字符串，摄影机运动（static/pan_left/pan_right/tilt_up/tilt_down/dolly_in/dolly_out/tracking_left/tracking_right/crane_up/crane_down/handheld/steadicam/orbital/zoom_in/zoom_out）\n"
    "  - lighting_key: 字符串，光影基调（high_key/low_key/natural/golden_hour/blue_hour/tungsten_warm/neon/silhouette/rim_lit/volumetric/overcast_soft）\n"
    "  - shot_intent: 字符串，镜头叙事意图（establishing/reveal/reaction/dialogue/action/transition/atmosphere/detail）\n\n"
    "  - dialogue: 对象或 null；有角色在本镜头说话时为 {{\"speaker\": \"角色名\", \"line\": \"剧本台词原文\"}}，无对白时必须为 null\n"
    "  - gen_strategy: 字符串，视频生成策略（flf2v/phantom/i2v）；最终值会由确定性规则校正\n\n"
    "【对白铁律】\n"
    "dialogue.line 必须逐字来自上方事件列表中的剧本原文，禁止改写、摘要或编造台词。\n"
    "本镜头无人真正说话时 dialogue 必须输出 null，不得把 visual、what 或旁白当作角色对白。\n\n"
    "【镜头连贯性规则】\n"
    "每个镜头（除第一个外）必须在 visual 描述开头加入「承接上镜」段：\n"
    "- 格式：'承接上镜：上镜定格于{{角色名}}{{位置/姿态/朝向}}，{{最后动作的终态}}——本镜由此延续'\n"
    "- 目的：让视频生成时画面自然衔接，不跳跃\n"
    "- 第一个镜头无需承接\n"
    "- 跨场景切换时不写承接（硬切）\n\n"
    "【视频延长判定】\n"
    "boundary_before=continuous 表示本镜必须使用上一镜视频作为素材延长生成；仅在同一时空、"
    "同一主要主体、动作方向/速度/姿态连续且没有叙事时间跳跃时使用。景别或机位需要硬切、"
    "地点/时间变化、主体切换、回忆/梦境/与此同时等情况必须为 cut。第一镜必须为 cut。\n\n"
    "【小说化动作剧本】\n"
    "事件中的 event_role、sequence_id、action_unit_id、micro_actions、start_state、end_state、"
    "causal_link、continuity_before 是连续性事实，不是可自由改写的文案。\n"
    "- 一个 action_unit 是完整的攻防/反应/结果单元，不得为了逐句对应而拆成互不相干的硬切。\n"
    "- 同一 sequence_id 中相邻动作单元若 continuity_before=continuous，应优先设计成延长视频；"
    "只有明确换机位、插入反应镜或生成能力无法容纳时才使用 cut，并在 continuity_reason 说明。\n"
    "- visual 必须按 micro_actions 的原始顺序描述，保留 start_state→动作→end_state 和 causal_link，"
    "禁止概括成‘双方激烈打斗’。\n"
    "- turning_point/dramatic_turn 必须保留为独立叙事节拍，不得与普通交锋合并掉。\n"
    "- lines 中 speaker/confidence/evidence 是对白归属证据；低置信度时不得擅自换成另一角色。\n\n"
    "【片段间过渡规则】\n"
    "相邻片段之间必须设计过渡桥梁，消灭跳跃感：\n"
    "1. 动作桥梁：前段结尾=动作起始态，后段首镜=进行时/完成时\n"
    "2. 情绪接力：前段结尾用反应镜/微表情铺垫，后段承接强化\n"
    "3. 空间视线：场景切换时用空镜+视线引导+声音延续\n"
    "4. 台词黏合：前段末尾声音延续到后段首镜\n\n"
    "【铁律优先级】\n"
    "台词零删改 > 出场人物完整 > 只描述动作状态 > 长台词拆镜\n\n"
    "【HonCut 分镜铁律】\n"
    "0. who 只能逐字引用上方角色列表中的主名，别名必须改写为对应主名；"
    "群体/群众/背景元素（如‘数百名机械居民’‘企业执法机械体群’）不得写入 who，只能写入 visual。\n"
    "1. 每片段时长≤15秒，超过必须拆分\n"
    "2. 单镜台词>20字必须拆镜（台词4字/秒计算：20字=5秒）\n"
    "3. 在场人物不消失：同场景内角色不能无故离场，必须交代去向\n"
    "4. 人物外观不进提示词：发型/服装/体态由角色参考图承载，visual只写动作和表情\n"
    "5. 声音只写环境音+音效，禁止写配乐/BGM/背景音乐\n"
    "6. 群演不抢戏：群演只做背景动作，不给特写和台词\n"
    "7. 景别视角错开：相邻镜头不应使用相同景别和角度\n\n"
    "【HonCut Identity Anchor】\n"
    "每个镜头的 visual 字段必须以「角色名 — 3-6个视觉特征」开头。\n"
    "特征从角色列表的 appearance.summary 提取，逐字复述，不省略不用代词。\n"
    "错误示例：'她站在门口' / '该角色转身'\n"
    "正确示例：'林夏 — 黑色长直发及肩, 白色修身衬衫, 深蓝西装裤 — 站在便利店门口'\n\n"
    "【HonCut 资产绑定（associateAssetsIds）】\n"
    "每个镜头必须声明 associate_assets，列出画面中可见的角色和场景：\n"
    "- 角色出现即引用：格式 'char:角色id'（如 'char:lin_xia'）\n"
    "- 场景必选：格式 'scene:场景名'（如 'scene:便利店门口'）\n"
    "- 示例：[\"char:lin_xia\", \"char:shen_yu\", \"scene:便利店门口\"]\n\n"
    "【HonCut 空间位置基准】\n"
    "如果提供了 director_plan 中的 spatial_positions，每个镜头的 visual 必须按基准表标注角色位置：\n"
    "- 格式：'角色名在画面左前/右前/居中，面朝左/右/镜头'\n"
    "- 同场景内角色位置不应无故跳变（左前突然变右前）\n"
    "- 位置变化必须有动作交代（如'林夏从左侧走到右侧'）\n\n"
    "注意：所有 shot 的 suggested_duration 总和应接近 target_duration（允许 ±10% 偏差）。"
)


# Stage 2 only: its duration budget is local to one beat batch.  Keep this
# independent from USER_PROMPT_TEMPLATE, whose global contract is still used by
# the LEGACY single-call path.
BATCH_EXPAND_PROMPT = (
    "本批目标时长：{batch_target}秒，每镜约{shot_duration}秒，恰好输出{max_shots}个镜头。"
    "每镜只完成一个明确情节。\n\n"
    "本批来源事件：\n{events_json}\n\n角色列表：\n{characters_summary}\n\n"
    "输出严格 JSON 对象，包含 strategy 和 shots。每个 shot 必须包含以下字段：\n"
    "beat_order（整数，必须等于该镜展开自哪个 beat 的 beat_order）、shot_order、"
    "source_events、action、reason、who、where、what、emotion、visual、"
    "suggested_duration、boundary_before、continuity_reason、continuity_subject、"
    "transition_to_next、associate_assets、shot_size、"
    "camera_movement、lighting_key、shot_intent、dialogue、gen_strategy。\n"
    "JSON 示例：{{\"strategy\":\"本批策略\",\"shots\":[{{\"beat_order\":1,"
    "\"shot_order\":1,\"source_events\":[1],\"action\":\"keep\","
    "\"reason\":\"理由\",\"who\":[\"角色主名\"],\"where\":\"地点\","
    "\"what\":\"事件\",\"emotion\":\"情绪\",\"visual\":\"画面\","
    "\"suggested_duration\":12,\"boundary_before\":\"cut\","
    "\"continuity_reason\":\"新场景\",\"continuity_subject\":\"\","
    "\"transition_to_next\":\"cut\","
    "\"associate_assets\":[\"char:id\",\"scene:地点\"],\"shot_size\":\"medium\","
    "\"camera_movement\":\"static\",\"lighting_key\":\"natural\","
    "\"shot_intent\":\"action\",\"dialogue\":null,\"gen_strategy\":\"phantom\"}}]}}\n\n"
    "【对白铁律】dialogue 有对白时为 {{\"speaker\":\"角色名\",\"line\":\"剧本台词原文\"}}，"
    "line 必须逐字来自来源事件，禁止改写、摘要或编造；无人真正说话时必须为 null，"
    "不得把 visual、what 或旁白当对白。\n\n"
    "【镜头连贯性规则】除第一镜和跨场景硬切外，visual 开头必须写"
    "「承接上镜：上镜定格于{{角色名}}{{位置/姿态/朝向}}，{{最后动作的终态}}——本镜由此延续」。\n"
    "【视频延长判定】第一镜 boundary_before 必须为 cut。只有同一时空、同一主要主体、"
    "动作方向/速度/姿态直接连续且无时间跳跃时，下一镜 boundary_before 才能为 continuous，"
    "并填写 continuity_reason 与 continuity_subject；否则必须为 cut。\n"
    "【小说化动作剧本】严格继承来源事件的 sequence_id/action_unit_id/micro_actions/start_state/"
    "end_state/causal_link/continuity_before；按动作原顺序展开，禁止用‘激烈打斗’代替具体招式与结果。"
    "同一 sequence_id 的连续动作优先 boundary_before=continuous；turning_point 必须独立保留。\n"
    "【片段间过渡规则】相邻片段用动作桥梁、情绪接力、空间视线或台词黏合消灭跳跃感。\n"
    "【铁律优先级】台词零删改 > 出场人物完整 > 只描述动作状态 > 长台词拆镜。\n\n"
    "【HonCut 分镜铁律】who 只能逐字引用角色主名，别名改主名，群众只进 visual；"
    "每片段≤15秒；单镜台词>20字必须拆镜（按4字/秒）；同场景人物不得无故消失；"
    "人物外观不进提示词；声音只写环境音和音效，禁止配乐/BGM/背景音乐；"
    "群演只做背景动作；相邻镜头景别和角度必须错开。\n\n"
    "【HonCut Identity Anchor】每镜 visual 必须以「角色名 — 3-6个视觉特征」开头，"
    "特征逐字取自 appearance.summary，不用代词。\n\n"
    "【HonCut 资产绑定（associateAssetsIds）】每镜 associate_assets 必须列出可见角色"
    "（char:角色id）和必选场景（scene:场景名）。\n\n"
    "【HonCut 空间位置基准】若 director_plan 提供 spatial_positions，visual 必须标注角色"
    "在画面左前/右前/居中及朝向；同场景位置不得无故跳变，变化必须有动作交代。\n\n"
    "本批所有 shot 的 suggested_duration 总和应接近 {batch_target} 秒（允许 ±10% 偏差）。"
)


_ACTION_VERBS = (
    "抬手", "举手", "挥手", "走来", "走向", "走到", "坐下", "站起", "起身",
    "转身", "拥抱", "抱住", "牵手", "拉手", "奔跑", "跑来", "跳起", "跪下",
    "推开", "拉开", "打开", "关上", "递给", "接过", "弯腰", "回头",
    "冲出", "冲来", "追击", "挥刀", "拔刀", "举刀", "横扫", "斩", "劈", "刺击",
    "格挡", "挡住刀锋", "抽刀", "旋身", "踹", "抬膝", "扫腿", "翻滚", "撞出",
    "撑住钢梁", "扣住手腕", "刀锋", "两刃碰撞",
    "raises her hand", "raises his hand", "walks over", "walks toward", "sits down",
    "stands up", "turns around", "embraces", "hugs", "holds hands", "runs toward",
)
_DIALOGUE_EMOTION_MARKERS = (
    "对话", "交谈", "说话", "说道", "询问", "回答", "低语", "耳语", "台词",
    "凝视", "对视", "注视", "微笑", "落泪", "流泪", "皱眉", "表情", "神情",
    "情绪", "反应", "沉默", "脸部", "面部", "dialogue", "speaks", "talking",
    "expression", "emotion", "reaction", "close-up", "close up", "facing each other",
    "eye contact", "smiles", "tears",
)


def determine_gen_strategy(shot: Dict[str, Any]) -> str:
    """Choose the local video route with action > character > safe I2V precedence.

    Clear body movement uses FLF2V. Every other character shot uses Phantom so
    character reference images constrain appearance consistently. Scenery and
    ambient shots without characters use single-image I2V.
    """
    searchable = " ".join(
        str(shot.get(field, ""))
        for field in ("visual", "what", "prompt", "description")
    ).lower()
    if any(verb in searchable for verb in _ACTION_VERBS):
        return "flf2v"

    who = shot.get("who", [])
    has_characters = bool(who)
    if has_characters:
        return "phantom"
    return "i2v"

LLM_TIMEOUT = 900  # 135 个事件的健康流实测超过 240 秒；保留 15 分钟绝对安全上限
LLM_IDLE_TIMEOUT = 75  # 只在流连续 75 秒没有任何 chunk 时判定停滞
MAX_RETRIES = 1  # 解析失败重试次数
NETWORK_RETRIES = 2  # 网络超时自动重试次数（2026-08-09 R7: 70事件大prompt一次超时即死太脆）
AVG_SHOT_DURATION = 12  # 默认每镜时长（秒）
MIN_SHOT_DURATION = 4  # 单镜头最小时长（秒）
MAX_SHOT_DURATION = 15  # 单镜头最大时长（秒）
CHARS_PER_SECOND = 4  # 中文剧本预估：约 4 字/秒（范围 3-5）
DEFAULT_TARGET_DURATION = 60  # 默认目标时长（用户未指定时使用）


def estimate_duration_from_text(text: str) -> int:
    """
    根据剧本字数预估视频时长（秒）。

    中文阅读/表演节奏约 3-5 字/秒，取中值 4 字/秒。
    结果限制在 [15, 300] 秒范围内（15秒~5分钟）。

    Args:
        text: 剧本文本

    Returns:
        预估时长（秒）
    """
    if not text:
        return DEFAULT_TARGET_DURATION

    # 去除空白后计算有效字符数
    clean_text = re.sub(r'\s+', '', text)
    char_count = len(clean_text)

    if char_count == 0:
        return DEFAULT_TARGET_DURATION

    # 预估时长 = 字数 / 字速
    estimated = char_count / CHARS_PER_SECOND

    # 限制范围：最短 15 秒，最长 300 秒（5 分钟）
    estimated = max(15, min(300, estimated))

    return int(estimated)


def estimate_shot_count(target_duration: int, shot_duration: int = AVG_SHOT_DURATION) -> int:
    """
    根据目标时长和单镜时长计算合理的镜头数量。

    单镜时长限制在 [MIN_SHOT_DURATION, MAX_SHOT_DURATION] 范围内。

    Args:
        target_duration: 目标总时长（秒）
        shot_duration: 每镜平均时长（秒）

    Returns:
        建议的最大镜头数
    """
    # 将 shot_duration 限制在合理范围内
    shot_duration = max(MIN_SHOT_DURATION, min(MAX_SHOT_DURATION, shot_duration))

    max_shots = max(1, (target_duration + shot_duration - 1) // shot_duration)
    return max_shots


# ─── LLM 客户端 ─────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    """
    创建 OpenAI 客户端

    API Key: ARK_AGENT_API_KEY (火山方舟 Agent Plan)
    """
    return create_ark_client(read_timeout=LLM_IDLE_TIMEOUT)


def _call_llm(user_prompt: str, max_tokens: int = 32000) -> str:
    """
    调用 LLM 并返回原始响应文本

    Args:
        user_prompt: 用户 prompt

    Returns:
        LLM 原始响应字符串

    Raises:
        Exception: API 调用失败时抛出
    """
    client = _get_client()

    # 2026-08-09 R8 教训：70事件→15镜大JSON非流式调用，turbo生成完整响应
    # 远超 240s timeout（R7/R8 共 6 次超时实锤）。改流式后 timeout 语义变为
    # "等下一个数据块"，turbo 推理模型先吐 reasoning 再吐 content，数据流不断即不断连。
    # 2026-08-09 R9 教训：不设 max_tokens 时用默认输出上限，15镜×18字段 JSON
    # 在 char 8354/9433 被截断（JSONDecodeError: Unterminated string）。
    # 探针实锤 max_tokens=16000/32000 均被 Agent Plan 端点接受（HTTP 200），
    # 取 32000 留足 reasoning + 15 镜 JSON 余量。
    return call_llm_stream(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        wall_timeout=LLM_TIMEOUT,
        idle_timeout=LLM_IDLE_TIMEOUT,
        _client=client,
    )


def _call_llm_with_timeout_retry(user_prompt: str, max_tokens: int = 32000) -> str:
    """
    网络超时自动重试包装（2026-08-09 R7 教训：70 事件大 prompt 一次 ReadTimeout 即死）

    只对 APITimeoutError 重试，其他异常原样上抛由调用方处理。

    Args:
        user_prompt: 用户 prompt

    Returns:
        LLM 原始响应字符串

    Raises:
        RuntimeError: 连续超时超过 NETWORK_RETRIES 次
    """
    for net_attempt in range(NETWORK_RETRIES + 1):
        try:
            return _call_llm(user_prompt, max_tokens=max_tokens)
        except (
            APITimeoutError,
            LLMConnectTimeout,
            LLMReadTimeout,
            LLMIdleTimeout,
            LLMStreamError,
        ) as e:
            if net_attempt < NETWORK_RETRIES:
                wait = 15 * (net_attempt + 1)
                print(f"  ⚠ LLM 网络超时，{wait}s 后重试 ({net_attempt + 1}/{NETWORK_RETRIES})...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"LLM 调用失败: 连续 {NETWORK_RETRIES} 次网络超时: {e}"
                ) from e
    raise RuntimeError("LLM 调用失败: 意外退出重试循环")  # 不可达，满足类型检查


def _validate_shots(shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and normalize a shot list shared by both adaptation modes."""
    if not isinstance(shots, list):
        raise ValueError(f"'shots' 应为数组，得到 {type(shots).__name__}")

    required_fields = {"shot_order", "source_events", "action", "who", "where", "what", "visual", "suggested_duration"}
    for i, shot in enumerate(shots):
        if not isinstance(shot, dict):
            raise ValueError(f"第 {i+1} 个 shot 不是字典")
        missing = required_fields - set(shot.keys())
        if missing:
            raise ValueError(f"第 {i+1} 个 shot 缺少字段: {missing}")

    # 规范化结构化字段（shot_size/camera_movement/lighting_key/shot_intent）
    # 如果 LLM 没返回，给默认值而不是缺失
    _VALID_SHOT_SIZES = {
        "extreme_wide", "wide", "medium_wide", "medium", "medium_close",
        "close_up", "extreme_close_up", "over_shoulder", "insert", "establishing",
    }
    _VALID_CAMERA_MOVEMENTS = {
        "static", "pan_left", "pan_right", "tilt_up", "tilt_down",
        "dolly_in", "dolly_out", "tracking_left", "tracking_right",
        "crane_up", "crane_down", "handheld", "steadicam", "orbital",
        "zoom_in", "zoom_out", "rack_focus",
    }
    _VALID_LIGHTING_KEYS = {
        "high_key", "low_key", "natural", "golden_hour", "blue_hour",
        "tungsten_warm", "neon", "silhouette", "rim_lit", "volumetric", "overcast_soft",
    }
    _VALID_SHOT_INTENTS = {
        "establishing", "reveal", "reaction", "dialogue", "action",
        "transition", "atmosphere", "detail",
    }
    for shot in shots:
        ss = shot.get("shot_size", "")
        if ss not in _VALID_SHOT_SIZES:
            shot["shot_size"] = "wide"
        cm = shot.get("camera_movement", "")
        if cm not in _VALID_CAMERA_MOVEMENTS:
            shot["camera_movement"] = "static"
        lk = shot.get("lighting_key", "")
        if lk not in _VALID_LIGHTING_KEYS:
            shot["lighting_key"] = "natural"
        si = shot.get("shot_intent", "")
        if si not in _VALID_SHOT_INTENTS:
            shot["shot_intent"] = "atmosphere"
        who = shot.get("who", [])
        if isinstance(who, str):
            shot["who"] = [who] if who else []
        elif not isinstance(who, list):
            shot["who"] = []
        aa = shot.get("associate_assets", [])
        if isinstance(aa, str):
            shot["associate_assets"] = [aa] if aa else []
        elif not isinstance(aa, list):
            shot["associate_assets"] = []
        dialogue = shot.get("dialogue")
        if not (
            (isinstance(dialogue, str) and dialogue.strip())
            or (isinstance(dialogue, list) and dialogue)
            or (isinstance(dialogue, dict) and dialogue)
        ):
            shot["dialogue"] = None
        shot["gen_strategy"] = determine_gen_strategy(shot)
    return shots


def _parse_response(response: str) -> Dict[str, Any]:
    """
    解析 LLM 响应为结构化字典

    期望格式：{"strategy": "...", "shots": [...]}

    Args:
        response: LLM 原始响应字符串

    Returns:
        包含 strategy 和 shots 的字典

    Raises:
        ValueError: 无法解析为有效 JSON 或缺少必要字段
    """
    text = response.strip()

    # 尝试提取 ```json ... ``` 代码块
    if "```" in text:
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # 解析 JSON
    parsed = json.loads(text)

    if not isinstance(parsed, dict):
        raise ValueError(f"期望 JSON 对象，得到 {type(parsed).__name__}")

    if "shots" not in parsed:
        raise ValueError("缺少 'shots' 字段")

    if "strategy" not in parsed:
        parsed["strategy"] = ""

    parsed["shots"] = _validate_shots(parsed["shots"])

    return parsed


def _build_characters_summary(characters: Optional[List[Dict[str, Any]]]) -> str:
    """
    将角色列表构建为 LLM 可读的摘要文本

    Args:
        characters: 角色列表（character_discoverer.py 输出）

    Returns:
        角色摘要字符串
    """
    if not characters:
        return "（无角色信息）"

    lines = []
    for char in characters:
        name = char.get("name", "未知")
        role = char.get("role", "unknown")
        aliases = char.get("aliases", [])
        appearance = char.get("appearance", {})
        clothing = appearance.get("clothing", "") if isinstance(appearance, dict) else ""

        line = f"- {name}"
        if aliases:
            line += f"（别名：{', '.join(aliases)}）"
        line += f"，角色定位：{role}"
        if clothing:
            line += f"，穿着：{clothing}"
        lines.append(line)

    return "\n".join(lines)


def _build_events_json(events: List[Dict[str, Any]]) -> str:
    """
    将事件列表格式化为 LLM 可读的 JSON 字符串

    为每个事件添加编号，方便 LLM 引用 source_events。

    Args:
        events: 事件列表

    Returns:
        格式化的事件 JSON 字符串
    """
    # 为每个事件添加编号
    numbered_events = []
    for i, event in enumerate(events, 1):
        event_copy = dict(event)
        event_copy["event_id"] = i
        numbered_events.append(event_copy)

    return json.dumps(numbered_events, ensure_ascii=False, indent=2)


def _inherit_event_semantics(
    shots: List[Dict[str, Any]], events: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Carry source screenplay evidence into shots after the LLM adaptation pass.

    The model still chooses framing and whether a camera cut is useful, while
    source ordering, exact excerpts, action units, and speaker evidence remain
    deterministic and auditable downstream.
    """
    event_by_id = {index: event for index, event in enumerate(events, 1)}
    previous_sequence_ids: List[str] = []
    for shot_index, shot in enumerate(shots):
        raw_ids = shot.get("source_events", [])
        source_ids = raw_ids if isinstance(raw_ids, list) else []
        details = [event_by_id[event_id] for event_id in source_ids if event_id in event_by_id]

        excerpts = [str(event.get("source_excerpt") or "").strip() for event in details]
        excerpts = [excerpt for excerpt in excerpts if excerpt]
        if excerpts:
            shot["source_excerpt"] = "\n".join(dict.fromkeys(excerpts))

        sequence_ids = [str(event.get("sequence_id")) for event in details if event.get("sequence_id")]
        sequence_ids = list(dict.fromkeys(sequence_ids))
        action_unit_ids = [str(event.get("action_unit_id")) for event in details if event.get("action_unit_id")]
        micro_actions = [
            str(action)
            for event in details
            for action in (event.get("micro_actions") or [])
            if str(action).strip()
        ]
        roles = [str(event.get("event_role")) for event in details if event.get("event_role")]
        shot["source_sequence_ids"] = sequence_ids
        shot["source_action_unit_ids"] = list(dict.fromkeys(action_unit_ids))
        shot["source_event_roles"] = list(dict.fromkeys(roles))
        shot["micro_actions"] = micro_actions

        speaker_evidence = [
            dict(line)
            for event in details
            for line in (event.get("lines") or [])
            if isinstance(line, dict) and line.get("line")
        ]
        if speaker_evidence:
            shot["speaker_attribution"] = speaker_evidence
            dialogue = shot.get("dialogue")
            if isinstance(dialogue, dict):
                exact = next(
                    (line for line in speaker_evidence if line.get("line") == dialogue.get("line")),
                    None,
                )
                if exact is None:
                    # A generated/paraphrased line violates the screenplay contract.
                    shot["dialogue"] = None
                else:
                    shot["dialogue"] = {
                        "speaker": exact.get("speaker", "未知"),
                        "line": exact["line"],
                        "confidence": exact.get("confidence", 0.0),
                        "evidence": exact.get("evidence", ""),
                        "dialogue_id": exact.get("dialogue_id"),
                    }

        first_source = details[0] if details else {}
        source_boundary = str(first_source.get("continuity_before") or "").lower()
        source_subject = str(first_source.get("continuity_subject") or "").strip()
        same_sequence = bool(sequence_ids and previous_sequence_ids and sequence_ids[0] in previous_sequence_ids)
        if shot_index == 0:
            shot["boundary_before"] = "cut"
        elif not str(shot.get("boundary_before") or "").strip() and source_boundary in {"cut", "continuous"}:
            shot["boundary_before"] = source_boundary if same_sequence else "cut"
        if shot.get("boundary_before") == "continuous":
            if source_subject and not str(shot.get("continuity_subject") or "").strip():
                shot["continuity_subject"] = source_subject
            shot.setdefault(
                "continuity_reason",
                "source action unit directly continues within the same screenplay sequence",
            )
        previous_sequence_ids = sequence_ids
    return shots


BEAT_SKELETON_PROMPT = (
    "目标时长：{target_duration}秒，每镜约{shot_duration}秒。请把全部事件压缩为恰好{beat_count}个 beat。\n\n"
    "事件列表：\n{events_json}\n\n角色列表：\n{characters_summary}\n\n"
    "只做全局改编决策，不要输出 visual、镜头语言、景别或摄影细节。输出严格 JSON 对象："
    '{{"strategy":"一句话改编策略","beats":[{{"beat_order":1,"source_events":[1],'
    '"action":"keep/merge/drop","reason":"一句话理由","who":["角色主名"],'
    '"where":"地点","what":"一句话事件","suggested_duration":12}}]}}。\n'
    "【全局铁律】\n"
    "1. beats 数量必须恰好等于 {beat_count}，总建议时长应接近 {target_duration} 秒（±10%）。\n"
    "2. 每个输入事件编号必须且至少被某个 beat 的 source_events 引用；删减事件也必须放入 action=drop 的 beat 显式声明。\n"
    "3. keep 保留关键因果/情感节点，merge 合并连续或重复事件，drop 只删不影响因果链的内容。\n"
    "4. 台词归属必须忠于原事件；who 只能使用角色列表主名，别名改为主名，群众不得写入 who。\n"
    "5. action_unit_id 是已经按因果闭环分组的动作单元：不得按句号或单个招式再次打碎；"
    "也不得把不同 turning_point 合并掉。长动作可以与相邻单元合并为 beat，但必须保持 micro_actions 原顺序。\n"
    "6. sequence_id 与 continuity_before 是生成连续性依据。同一 sequence 的连续单元尽量落在相邻 beat，"
    "换场/跳时/关系转折不得为了省镜头而错误连拍。\n"
    "7. 只输出骨架决策，禁止展开对白、visual、Identity Anchor 或任何镜头生成细节。"
)


def _parse_beat_skeleton(response: str, expected_count: int, event_count: int) -> Dict[str, Any]:
    """Parse and validate the bounded Stage 1 beat table."""
    text = response.strip()
    if "```" in text:
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("beats"), list):
        raise ValueError("骨架响应必须是包含 beats 数组的 JSON 对象")
    beats = parsed["beats"]
    if len(beats) != expected_count:
        raise ValueError(f"beat 数量应为 {expected_count}，实际为 {len(beats)}")
    required = {"beat_order", "source_events", "action", "reason", "who", "where", "what", "suggested_duration"}
    covered = set()
    for i, beat in enumerate(beats, 1):
        if not isinstance(beat, dict):
            raise ValueError(f"第 {i} 个 beat 不是字典")
        missing = required - set(beat)
        if missing:
            raise ValueError(f"第 {i} 个 beat 缺少字段: {missing}")
        if beat["action"] not in {"keep", "merge", "drop"}:
            raise ValueError(f"第 {i} 个 beat action 无效: {beat['action']}")
        if not isinstance(beat["source_events"], list):
            raise ValueError(f"第 {i} 个 beat source_events 必须是数组")
        for event_id in beat["source_events"]:
            if not isinstance(event_id, int) or event_id < 1 or event_id > event_count:
                raise ValueError(f"第 {i} 个 beat 引用了无效事件编号: {event_id}")
            covered.add(event_id)
    missing_events = set(range(1, event_count + 1)) - covered
    if missing_events:
        raise ValueError(f"beat 未覆盖事件编号: {sorted(missing_events)}")
    parsed.setdefault("strategy", "")
    return parsed


def _build_beat_skeleton(
    events: List[Dict[str, Any]],
    characters_summary: str,
    target_duration: int,
    shot_duration: int,
) -> Dict[str, Any]:
    """Build a globally informed, bounded beat table (Stage 1)."""
    beat_count = estimate_shot_count(target_duration, shot_duration)
    prompt = BEAT_SKELETON_PROMPT.format(
        target_duration=target_duration,
        shot_duration=shot_duration,
        beat_count=beat_count,
        events_json=_build_events_json(events),
        characters_summary=characters_summary,
    )
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = _call_llm_with_timeout_retry(prompt, max_tokens=8000)
            skeleton = _parse_beat_skeleton(response, beat_count, len(events))
            event_by_id = {i: dict(event, event_id=i) for i, event in enumerate(events, 1)}
            for beat in skeleton["beats"]:
                beat["_source_event_details"] = [event_by_id[event_id] for event_id in beat["source_events"]]
            return skeleton
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < MAX_RETRIES:
                print(f"骨架解析失败，重试中（{attempt + 1}/{MAX_RETRIES}）: {e}", file=sys.stderr)
                time.sleep(1)
            else:
                raise RuntimeError(f"骨架响应解析失败（已重试 {MAX_RETRIES} 次）: {e}") from e
        except Exception as e:
            raise RuntimeError(f"骨架 LLM 调用失败: {e}") from e
    raise RuntimeError("骨架 LLM 调用失败：未获得有效响应")


def _batch_prompt(
    batch: List[Dict[str, Any]],
    characters_summary: str,
    target_duration: int,
    shot_duration: int,
    offset: int,
    relay: Optional[Dict[str, str]],
) -> str:
    public_beats = [{k: v for k, v in beat.items() if not k.startswith("_")} for beat in batch]
    event_details = [detail for beat in batch for detail in beat.get("_source_event_details", [])]
    batch_target = sum(beat.get("suggested_duration", 0) for beat in batch)
    base = BATCH_EXPAND_PROMPT.format(
        batch_target=batch_target,
        shot_duration=shot_duration,
        max_shots=len(batch),
        events_json=json.dumps(event_details, ensure_ascii=False, indent=2),
        characters_summary=characters_summary,
    )
    if relay is None:
        relay_text = "这是第一镜，无需承接上镜。"
    else:
        relay_text = (
            "上一批最后一镜接力上下文（只据此保持连续）："
            f"who={json.dumps(relay['who'], ensure_ascii=False)}；where={relay['where']}；"
        )
        visual = relay.get("visual")
        if isinstance(visual, str) and visual:
            relay_text += f"visual末尾={visual[-100:]}"
        else:
            relay_text += "上一镜无可用 visual，仅按 who/where 承接"
    return (
        f"{base}\n\n【本批展开任务】\n本批 beat：\n"
        f"{json.dumps(public_beats, ensure_ascii=False, indent=2)}\n"
        f"{relay_text}\n本批只输出 {len(batch)} 个完整 shot；第一个 shot_order 必须为 {offset + 1}，"
        "随后连续递增。每个 beat 恰好展开为一个 shot，source_events/action 必须忠于 beat。"
    )


def _expand_beats_to_shots(
    beats: List[Dict[str, Any]],
    characters_summary: str,
    target_duration: int,
    shot_duration: int,
    output_dir: Optional[Path] = None,
    resumed_shots: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Expand three beats at a time, relaying only the previous final shot."""
    shots: List[Dict[str, Any]] = list(resumed_shots or [])
    relay: Optional[Dict[str, str]] = None
    if shots:
        last = shots[-1]
        relay = {
            "who": last.get("who", []),
            "where": str(last.get("where", "")),
            "visual": last.get("visual"),
        }
    first_missing = len(shots)
    for start in range(first_missing, len(beats), 3):
        batch = beats[start:start + 3]
        prompt = _batch_prompt(batch, characters_summary, target_duration, shot_duration, len(shots), relay)
        parsed = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = _call_llm_with_timeout_retry(prompt, max_tokens=16000)
                parsed = _parse_response(response)
                if len(parsed["shots"]) != len(batch):
                    raise ValueError(f"本批应输出 {len(batch)} 镜，实际为 {len(parsed['shots'])} 镜")
                expanded_fields = {
                    "shot_order", "source_events", "action", "reason", "who", "where",
                    "what", "emotion", "visual", "suggested_duration", "transition_to_next",
                    "associate_assets", "shot_size", "camera_movement", "lighting_key",
                    "shot_intent", "dialogue", "gen_strategy",
                }
                for index, shot in enumerate(parsed["shots"], 1):
                    missing = expanded_fields - set(shot)
                    if missing:
                        raise ValueError(f"本批第 {index} 镜缺少完整字段: {missing}")
                    beat = batch[index - 1]
                    if shot.get("beat_order") != beat["beat_order"]:
                        raise ValueError(
                            f"本批第 {index} 镜 beat_order 错配: "
                            f"应为 {beat['beat_order']}，实际为 {shot.get('beat_order')}"
                        )
                    returned_sources = shot.get("source_events")
                    if not isinstance(returned_sources, list) or set(returned_sources) != set(beat["source_events"]):
                        raise ValueError(f"本批第 {index} 镜 source_events 与 beat 不一致")
                break
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < MAX_RETRIES:
                    print(f"第 {start // 3 + 1} 批解析失败，重试中（{attempt + 1}/{MAX_RETRIES}）: {e}", file=sys.stderr)
                    time.sleep(1)
                else:
                    raise RuntimeError(
                        f"第 {start // 3 + 1} 批响应解析失败（已重试 {MAX_RETRIES} 次）: {e}"
                    ) from e
            except Exception as e:
                raise RuntimeError(f"第 {start // 3 + 1} 批 LLM 调用失败: {e}") from e
        if parsed is None:
            raise RuntimeError(f"第 {start // 3 + 1} 批未获得有效响应")
        for shot, beat in zip(parsed["shots"], batch):
            # One beat maps to one shot; keep coverage/drop decisions deterministic
            # even when the expansion model drifts from its input skeleton.
            shot["source_events"] = list(beat["source_events"])
            shot["action"] = beat["action"]
            shot["shot_order"] = len(shots) + 1
            shot.pop("beat_order", None)
            shots.append(shot)
        if output_dir is not None:
            _atomic_write_json(
                output_dir / "shots_partial.json",
                {
                    "completed_batches": list(range(1, (len(shots) + 2) // 3 + 1)),
                    "shots": shots,
                },
            )
        last = shots[-1]
        relay = {
            "who": last.get("who", []),
            "where": str(last.get("where", "")),
            "visual": last.get("visual"),
        }
    return shots


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON without ever exposing a partially written checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_layered_checkpoints(
    output_dir: Path,
    events: List[Dict[str, Any]],
    expected_beats: int,
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load only valid, contiguous layered checkpoints."""
    skeleton = None
    skeleton_path = output_dir / "beat_skeleton.json"
    if skeleton_path.exists():
        try:
            candidate = json.loads(skeleton_path.read_text(encoding="utf-8"))
            _parse_beat_skeleton(json.dumps(candidate, ensure_ascii=False), expected_beats, len(events))
            event_by_id = {i: dict(event, event_id=i) for i, event in enumerate(events, 1)}
            for beat in candidate["beats"]:
                beat["_source_event_details"] = [event_by_id[event_id] for event_id in beat["source_events"]]
            skeleton = candidate
            print(f"  ↺ Reusing layered checkpoint: {skeleton_path}")
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            skeleton = None

    shots: List[Dict[str, Any]] = []
    partial_path = output_dir / "shots_partial.json"
    if skeleton is not None and partial_path.exists():
        try:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            candidate_shots = partial.get("shots", [])
            completed = partial.get("completed_batches", [])
            if not isinstance(candidate_shots, list) or not isinstance(completed, list):
                raise ValueError("invalid partial checkpoint structure")
            contiguous = list(range(1, len(completed) + 1))
            expected_count = min(len(skeleton["beats"]), len(completed) * 3)
            if completed != contiguous or len(candidate_shots) != expected_count:
                raise ValueError("partial checkpoint is not batch-contiguous")
            _validate_shots(candidate_shots)
            shots = candidate_shots
            print(f"  ↺ Reusing {len(completed)} completed layered batch(es): {partial_path}")
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            shots = []
    return skeleton, shots


# ─── 核心函数 ────────────────────────────────────────────────────────────────

def adapt_events(
    events: List[Dict[str, Any]],
    characters: Optional[List[Dict[str, Any]]] = None,
    target_duration: Optional[int] = None,
    shot_duration: int = AVG_SHOT_DURATION,
    source_text: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    将事件列表改编为 shot 列表

    Args:
        events: 事件列表（event_extractor.py 输出）
        characters: 角色列表（character_discoverer.py 输出，可选）
        target_duration: 目标总时长（秒），默认 None（根据剧本长度智能预估）
        shot_duration: 每镜平均时长（秒），默认 12
        source_text: 原始剧本文本（用于智能预估时长）

    Returns:
        包含 target_duration, estimated_shots, strategy, shots 的字典

    Raises:
        ValueError: 事件为空或时长不合理
        RuntimeError: LLM 调用失败
    """
    # ── 输入验证 ──────────────────────────────────────────────────────────
    if not events:
        raise ValueError("事件列表为空，无法进行改编")

    # ── 智能时长预估 ─────────────────────────────────────────────────────
    if target_duration is None:
        if source_text:
            target_duration = estimate_duration_from_text(source_text)
            print(f"  ℹ 根据剧本长度智能预估时长: {target_duration}秒")
        else:
            target_duration = DEFAULT_TARGET_DURATION
            print(f"  ℹ 使用默认时长: {target_duration}秒")

    if target_duration < 10:
        raise ValueError(f"目标时长不合理：{target_duration}秒（最少 10 秒）")

    if not MIN_SHOT_DURATION <= shot_duration <= MAX_SHOT_DURATION:
        raise ValueError(
            f"每镜时长不合理：{shot_duration}秒"
            f"（应在 {MIN_SHOT_DURATION}-{MAX_SHOT_DURATION} 秒）"
        )

    # ── 计算最大 shot 数（智能分镜）──────────────────────────────────────
    max_shots = estimate_shot_count(target_duration, shot_duration)

    # ── 构建 prompt ───────────────────────────────────────────────────────
    events_json = _build_events_json(events)
    characters_summary = _build_characters_summary(characters)

    requested_mode = os.getenv("HONCUT_ADAPT_MODE", "layered").strip().lower()
    use_layered = requested_mode != "single" and len(events) > 10
    if use_layered:
        checkpoint_dir = Path(output_dir) if output_dir is not None else None
        skeleton = None
        resumed_shots: List[Dict[str, Any]] = []
        if checkpoint_dir is not None:
            skeleton, resumed_shots = _load_layered_checkpoints(checkpoint_dir, events, max_shots)
        if skeleton is None:
            skeleton = _build_beat_skeleton(
                events, characters_summary, target_duration, shot_duration
            )
            if checkpoint_dir is not None:
                _atomic_write_json(checkpoint_dir / "beat_skeleton.json", skeleton)
        shots = _expand_beats_to_shots(
            skeleton["beats"], characters_summary, target_duration, shot_duration,
            output_dir=checkpoint_dir, resumed_shots=resumed_shots,
        )
        _validate_shots(shots)

        # Defensive assembly: batch responses may ignore their requested offset.
        for i, shot in enumerate(shots, 1):
            shot["shot_order"] = i

        _inherit_event_semantics(shots, events)

        for i, shot in enumerate(shots):
            if i > 0:
                prev = shots[i - 1]
                prev_visual = prev.get("visual", "")
                if shot.get("where") == prev.get("where"):
                    shot["prev_shot_context"] = (
                        f"承接上镜：{prev_visual[-80:]}"
                        if len(prev_visual) > 80
                        else f"承接上镜：{prev_visual}"
                    )
                else:
                    shot["prev_shot_context"] = ""
            else:
                shot["prev_shot_context"] = ""

        total_duration = sum(shot.get("suggested_duration", 0) for shot in shots)
        from quality.shot_continuity import annotate_boundaries

        annotate_boundaries(shots)
        if abs(total_duration - target_duration) > target_duration * 0.10:
            print(
                f"  ⚠ 分镜建议总时长 {total_duration}秒与目标 {target_duration}秒偏差超过 10%",
                file=sys.stderr,
            )
        return {
            "target_duration": target_duration,
            "estimated_shots": len(shots),
            "total_duration": total_duration,
            "strategy": skeleton.get("strategy", ""),
            "shots": shots,
        }

    # [LEGACY-KEEP layered-adapt] 原单次调用路径；显式 single 或事件数 <= 10 时使用。
    user_prompt = USER_PROMPT_TEMPLATE.format(
        target_duration=target_duration,
        shot_duration=shot_duration,
        max_shots=max_shots,
        events_json=events_json,
        characters_summary=characters_summary,
    )

    # ── 调用 LLM（带重试）─────────────────────────────────────────────────
    parsed = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = _call_llm_with_timeout_retry(user_prompt)
            parsed = _parse_response(response)
            break
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < MAX_RETRIES:
                print(f"解析失败，重试中（{attempt + 1}/{MAX_RETRIES}）: {e}", file=sys.stderr)
                time.sleep(1)
            else:
                raise RuntimeError(f"LLM 响应解析失败（已重试 {MAX_RETRIES} 次）: {e}") from e
        except Exception as e:
            raise RuntimeError(f"LLM 调用失败: {e}") from e
    
    if parsed is None:
        raise RuntimeError("LLM 调用失败：未获得有效响应")

    # ── 组装输出 ──────────────────────────────────────────────────────────
    shots = parsed["shots"]

    # 确保 shot_order 连续
    for i, shot in enumerate(shots, 1):
        shot["shot_order"] = i

    _inherit_event_semantics(shots, events)

    # Add continuity context between shots (镜头连贯性)
    for i, shot in enumerate(shots):
        if i > 0:
            prev = shots[i - 1]
            prev_visual = prev.get("visual", "")
            # Only add continuity if same scene (same 'where')
            if shot.get("where") == prev.get("where"):
                shot["prev_shot_context"] = (
                    f"承接上镜：{prev_visual[-80:]}"
                    if len(prev_visual) > 80
                    else f"承接上镜：{prev_visual}"
                )
            else:
                shot["prev_shot_context"] = ""  # Scene change, hard cut
        else:
            shot["prev_shot_context"] = ""  # First shot

    # 计算总时长
    from quality.shot_continuity import annotate_boundaries

    annotate_boundaries(shots)
    total_duration = sum(shot.get("suggested_duration", 0) for shot in shots)

    result = {
        "target_duration": target_duration,
        "estimated_shots": len(shots),
        "total_duration": total_duration,
        "strategy": parsed.get("strategy", ""),
        "shots": shots,
    }

    return result


# ─── CLI 入口 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="改编引擎 - 将事件列表改编为视频分镜 shot 列表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python adaptation_engine.py --events events.json --duration 60 --output shots.json
  python adaptation_engine.py --events events.json --characters characters.json --duration 30
  cat events.json | python adaptation_engine.py --duration 60
        """,
    )

    parser.add_argument(
        "--events", "-e",
        type=str,
        help="事件 JSON 文件路径（event_extractor.py 输出）",
    )

    parser.add_argument(
        "--characters", "-c",
        type=str,
        help="角色 JSON 文件路径（character_discoverer.py 输出，可选）",
    )

    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="目标时长（秒），默认 60",
    )

    parser.add_argument(
        "--shot-duration",
        type=int,
        default=AVG_SHOT_DURATION,
        help=f"每镜平均时长（秒），默认 {AVG_SHOT_DURATION}",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出 JSON 文件路径（可选，默认打印到 stdout）",
    )

    args = parser.parse_args()

    # ── 读取事件 ──────────────────────────────────────────────────────────
    events = None

    if args.events:
        try:
            with open(args.events, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"错误：文件不存在 - {args.events}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"错误：JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        # 支持直接传入 events 列表或包含 events 的字典
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict) and "events" in data:
            events = data["events"]
        else:
            print("错误：输入格式不支持，期望 events 列表或包含 events 的字典", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty():
        # 从 stdin 读取（管道模式）
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"错误：stdin JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        if isinstance(data, list):
            events = data
        elif isinstance(data, dict) and "events" in data:
            events = data["events"]
        else:
            print("错误：输入格式不支持", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # ── 读取角色（可选）────────────────────────────────────────────────────
    characters = None
    if args.characters:
        try:
            with open(args.characters, "r", encoding="utf-8") as f:
                char_data = json.load(f)
        except FileNotFoundError:
            print(f"错误：文件不存在 - {args.characters}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"错误：JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        # 支持直接传入角色列表或包含 characters 的字典
        if isinstance(char_data, list):
            characters = char_data
        elif isinstance(char_data, dict) and "characters" in char_data:
            characters = char_data["characters"]
        else:
            print("警告：角色文件格式不支持，忽略角色信息", file=sys.stderr)

    # ── 执行改编 ──────────────────────────────────────────────────────────
    try:
        result = adapt_events(
            events=events,
            characters=characters,
            target_duration=args.duration,
            shot_duration=args.shot_duration,
            output_dir=Path(args.output).parent if args.output else None,
        )
    except (ValueError, RuntimeError) as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    # ── 输出 JSON ─────────────────────────────────────────────────────────
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

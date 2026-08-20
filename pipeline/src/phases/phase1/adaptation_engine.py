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
1. 计算 sequence 装箱下限与最多 1.3× 的剪辑前素材预算
2. 调用 LLM 做改编决策：
   - keep：重要事件保留
   - merge：相似/连续事件合并
   - drop：重复/不重要事件删减
   - expand：关键情感时刻可扩展为多镜
3. 为每个 shot 分配可执行素材时长；Phase 8 再按交付时长精确剪辑
4. 建议转场方式（cut/dissolve/fade）
5. 输出排序后的 shot 列表
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import APITimeoutError, OpenAI

from utils.action_units import (
    normalize_action_units,
    normalize_event_action_units,
    normalized_action_unit_count,
)
from utils.ark_llm import (
    LLMConnectTimeout,
    LLMIdleTimeout,
    LLMReadTimeout,
    LLMStreamError,
    call_llm_stream,
    create_ark_client,
)
from utils.character_identity import (
    normalize_character_reference,
    resolve_character_name,
)
from utils.camera_motion_contracts import (
    CAMERA_MOTION_PLANNING_INSTRUCTIONS,
    CAMERA_MOVEMENT_VALUES,
    apply_camera_motion_contract,
)
from utils.temporal_visual_contracts import apply_temporal_visual_contract
from utils.video_capabilities import (
    MAX_CONTENT_BEATS_PER_PRIMARY_SHOT,
    VideoModelCapabilities,
    capabilities_for,
    get_video_capabilities,
    max_primary_story_duration,
    min_primary_story_duration,
)

# ─── LLM 配置 ───────────────────────────────────────────────────────────────

_SHOT_SIZE_VALUES = (
    "extreme_wide", "wide", "medium_wide", "medium", "medium_close",
    "close_up", "extreme_close_up", "over_shoulder", "insert", "establishing",
)
_CAMERA_MOVEMENT_VALUES = CAMERA_MOVEMENT_VALUES
_LIGHTING_KEY_VALUES = (
    "high_key", "low_key", "natural", "golden_hour", "blue_hour",
    "tungsten_warm", "neon", "silhouette", "rim_lit", "volumetric",
    "overcast_soft",
)
_SHOT_INTENT_VALUES = (
    "establishing", "reveal", "reaction", "dialogue", "action",
    "transition", "atmosphere", "detail",
)
_VALID_SHOT_SIZES = frozenset(_SHOT_SIZE_VALUES)
_VALID_CAMERA_MOVEMENTS = frozenset(_CAMERA_MOVEMENT_VALUES)
_VALID_LIGHTING_KEYS = frozenset(_LIGHTING_KEY_VALUES)
_VALID_SHOT_INTENTS = frozenset(_SHOT_INTENT_VALUES)

_SHOT_LANGUAGE_ENUM_CONTRACT = (
    "  - shot_size: 字符串，景别（" + "/".join(_SHOT_SIZE_VALUES) + "）\n"
    "  - camera_movement: 字符串，摄影机运动（"
    + "/".join(_CAMERA_MOVEMENT_VALUES)
    + "）\n"
    "  - lighting_key: 字符串，光影基调（"
    + "/".join(_LIGHTING_KEY_VALUES)
    + "）\n"
    "  - shot_intent: 字符串，镜头叙事意图（"
    + "/".join(_SHOT_INTENT_VALUES)
    + "）\n"
)

SYSTEM_PROMPT = (
    "你是影视导演兼剪辑师。将故事事件改编为视频分镜。"
    "考虑：节奏感、情感弧线、视觉变化、时长限制。"
    "输出严格 JSON，不要输出任何解释文字。"
)

USER_PROMPT_TEMPLATE = (
    "目标时长：{target_duration}秒，每镜约{shot_duration}秒，必须输出恰好{max_shots}个镜头。"
    "每镜只完成一个明确情节，并遵守当前视频模型的 generation_action_units 上限。\n\n"
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
    + _SHOT_LANGUAGE_ENUM_CONTRACT
    + CAMERA_MOTION_PLANNING_INSTRUCTIONS
    + "\n"
    + "  - hero_moment: 布尔值，是否为全片视觉峰值；4 镜以上至少且通常恰好一个为 true\n"
    "  - texture_keywords: 2–4 个具体环境材质/光影纹理关键词组成的字符串数组\n\n"
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
    "【导演镜头边界】\n"
    "boundary_before=continuous 只表示叙事与空间连续，供导演总览和剪辑判断；每个 Sxx 仍从自己的"
    "P01 以角色图+本格故事图等多图生成 8–15 秒第一段。只有当 P01 的最大叙事时长或动作容量不足以完整"
    "覆盖当前 Sxx 的全部剧情细节时，才生成 6–10 秒容量延长格，截取前段视频末段后延长；可按容量增加"
    "至多两个延长格，但不得仅因"
    "动作激烈、人物多或运镜复杂就强行增加延长格。若下一 Sxx 的 boundary_before=continuous，当前"
    "Sxx 只声明独立的 4–6 秒后置桥接任务，不占用 Pxx 和一级镜头剧情时长。所有一级视频完成后，"
    "以前一一级成片真实尾帧作首帧、下一一级成片真实首帧作尾帧生成桥接。下一 Sxx 为 cut，或存在换场、跳时、"
    "主体切换、回忆/梦境/与此同时、fade/dissolve/wipe 等转场时，绝不生成桥接，由 Phase 8 添加转场特效。第一镜必须为 cut。\n\n"
    "时长分配必须给上述结构留足当前视频模型可执行的最小时长，并服从其时长量化粒度；"
    "若目标总时长不足，优先调整一级分镜边界/时长，不得生成不可执行的伪 Pxx。\n\n"
    "【小说化动作剧本】\n"
    "事件中的 event_role、sequence_id、action_unit_id、micro_actions、start_state、end_state、"
    "causal_link、continuity_before 是连续性事实，不是可自由改写的文案。\n"
    "- 一个导演级镜头可以容纳同一 sequence_id 中多个连续 action_unit；必须保留全部动作单元，"
    "后续由编剧引擎按单段内容承载能力与相邻一级分镜边界自动拆成 1–3 个内部 Pxx；拆分只能重排时长，"
    "必须逐项覆盖当前 Sxx 的 micro_actions，保持原顺序和 start_state→end_state 因果，不得新增剧情。\n"
    "- 不同 sequence_id、明确换场/跳时不得为了减少镜头而错误合并。\n"
    "- visual 必须按 micro_actions 的原始顺序描述，保留 start_state→动作→end_state 和 causal_link，"
    "禁止概括成‘双方激烈打斗’。\n"
    "- turning_point/dramatic_turn 必须在 what/visual 中明确保留；同一 sequence 内可与紧邻因果动作"
    "共用一个导演级镜头，但不得被概括或吞掉。\n"
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
    "群体/群众/背景元素不得写入 who，只能写入 visual；who=[] 是无人物硬合同。\n"
    "1. 每个一级片段必须为15–30秒，并由一个8–15秒基础段加零至两个6–10秒容量延长段完整承载，超过必须拆分\n"
    "2. 单镜台词>20字必须拆镜（台词4字/秒计算：20字=5秒）\n"
    "3. 在场人物不消失：同场景内角色不能无故离场，必须交代去向\n"
    "4. 人物外观不进提示词：发型/服装/体态由角色参考图承载，visual只写动作和表情\n"
    "5. 声音只写环境音+音效，禁止写配乐/BGM/背景音乐\n"
    "6. 群演不抢戏：群演只做背景动作，不给特写和台词\n"
    "7. 景别视角错开：相邻镜头不应使用相同景别和角度\n\n"
    "【镜头语言容量闸门】\n"
    "shot_size、camera_movement、lighting_key、shot_intent、hero_moment、texture_keywords "
    "都是必填生成合同，不得省略或全部套用默认值。相邻镜头景别必须形成节奏差异；动作镜头不得"
    "全部 static；4 镜以上必须指定视觉峰值并为每镜提供具体纹理。连续长镜头可以保持同一真实"
    "光源，但应依据剧本允许的构图距离、运镜和环境纹理形成变化，不得虚构时空跳变。\n\n"
    "【HonCut Identity Anchor】\n"
    "身份只通过 who 与 associate_assets 结构化绑定；visual 只描述动作、表情、站位和环境，"
    "不得重复人物外貌。who=[] 时 visual 和 associate_assets 都不得引入任何角色。\n\n"
    "【HonCut 资产绑定（associateAssetsIds）】\n"
    "每个镜头必须声明 associate_assets，列出画面中可见的角色和场景：\n"
    "- 角色出现即引用：格式 'char:CHARACTER_ID'\n"
    "- 场景必选：格式 'scene:LOCATION_ID'\n"
    "- who=[] 时只能绑定场景资产，不得包含 char: 资产\n\n"
    "【HonCut 空间位置基准】\n"
    "如果提供了 director_plan 中的 spatial_positions，每个镜头的 visual 必须按基准表标注角色位置：\n"
    "- 格式：'角色名在画面左前/右前/居中，面朝左/右/镜头'\n"
    "- 同场景内角色位置不应无故跳变（左前突然变右前）\n"
    "- 位置变化必须有可见动作交代\n\n"
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
    "camera_movement、lighting_key、shot_intent、hero_moment、texture_keywords、"
    "dialogue、gen_strategy。\n"
    "JSON 示例：{{\"strategy\":\"本批策略\",\"shots\":[{{\"beat_order\":1,"
    "\"shot_order\":1,\"source_events\":[1],\"action\":\"keep\","
    "\"reason\":\"理由\",\"who\":[\"角色主名\"],\"where\":\"地点\","
    "\"what\":\"事件\",\"emotion\":\"情绪\",\"visual\":\"画面\","
    "\"suggested_duration\":15,\"boundary_before\":\"cut\","
    "\"continuity_reason\":\"新场景\",\"continuity_subject\":\"\","
    "\"transition_to_next\":\"cut\","
    "\"associate_assets\":[\"char:id\",\"scene:地点\"],\"shot_size\":\"medium\","
    "\"camera_movement\":\"dolly_in\",\"lighting_key\":\"natural\","
    "\"shot_intent\":\"action\",\"hero_moment\":false,"
    "\"texture_keywords\":[\"场景中的具体材质\",\"场景中的具体光影\"],"
    "\"dialogue\":null,\"gen_strategy\":\"phantom\"}}]}}\n\n"
    "【对白铁律】dialogue 有对白时为 {{\"speaker\":\"角色名\",\"line\":\"剧本台词原文\"}}，"
    "line 必须逐字来自来源事件，禁止改写、摘要或编造；无人真正说话时必须为 null，"
    "不得把 visual、what 或旁白当对白。\n\n"
    "【镜头连贯性规则】除第一镜和跨场景硬切外，visual 开头必须写"
    "「承接上镜：上镜定格于{{角色名}}{{位置/姿态/朝向}}，{{最后动作的终态}}——本镜由此延续」。\n"
    "【导演镜头边界】第一镜 boundary_before 必须为 cut。同一时空和动作因果连续时，下一镜可标"
    "continuous 并填写 continuity_reason；这只描述剪辑连续性，不表示跨 Sxx 延长视频。"
    "每个 Sxx 总时长 15–30 秒；P01 用角色图+本格故事图等多图生成 8–15 秒。只有 P01 最大叙事时长"
    "或动作容量不能覆盖当前 Sxx 全部剧情细节时，才生成一个或两个 6–10 秒容量延长格并截取前段"
    "视频末段后延长；不得仅凭视觉复杂度增加延长格。只有下一 Sxx 明确 continuous 时，才声明独立"
    "4–6 秒后置桥接任务：所有一级视频完成后，以当前一级成片真实尾帧和下一一级成片真实首帧生成；"
    "cut、换场、跳时、主体切换及 fade/dissolve/wipe 等转场绝不生成桥接，由 Phase 8 添加转场特效。\n"
    "时长必须匹配当前视频模型的最小时长、最大时长和时长量化粒度；不足时调整一级分镜时长或边界，"
    "不得输出不可执行的伪 Pxx。\n"
    "【小说化动作剧本】严格继承来源事件的 sequence_id/action_unit_id/micro_actions/start_state/"
    "end_state/causal_link/continuity_before；按动作原顺序展开，禁止用‘激烈打斗’代替具体招式与结果。"
    "同一 sequence_id 的相邻 action_unit 可以归入同一个导演级镜头，必须保留全部 micro_actions，"
    "供后续编剧引擎按内容承载能力与相邻边界拆成 1–3 个 Pxx 顺序执行；剧情格必须逐项覆盖当前 Sxx 的原始动作，"
    "不得改序、遗漏、新增剧情或提前执行下一 Sxx；跨 sequence 必须拆开，turning_point 必须明确保留。"
    "每个事件在 shots 中的 source_events 引用次数不得少于输入中的 minimum_primary_beat_occurrences；"
    "重复引用只分担该事件尚未表现的后续动作。\n"
    "【片段间过渡规则】相邻片段用动作桥梁、情绪接力、空间视线或台词黏合消灭跳跃感。\n"
    "【铁律优先级】台词零删改 > 出场人物完整 > 只描述动作状态 > 长台词拆镜。\n\n"
    "【HonCut 分镜铁律】who 只能逐字引用角色主名，别名改主名，群众只进 visual；"
    "每片段必须在 15–30 秒一级镜头及其 8–15/6–10 秒二级片段承载范围内；单镜台词>20字必须拆镜（按4字/秒）；同场景人物不得无故消失；"
    "人物外观不进提示词；声音只写环境音和音效，禁止配乐/BGM/背景音乐；"
    "群演只做背景动作；相邻镜头景别和角度必须错开。\n\n"
    "【镜头语言继承】本批每个 shot 的 shot_size、camera_movement、lighting_key、shot_intent、"
    "hero_moment、texture_keywords 必须逐字复制对应 beat 的全局镜头语言合同，不得重新规划或"
    "退回 wide/static/natural 默认组合。\n\n"
    + CAMERA_MOTION_PLANNING_INSTRUCTIONS
    + "\n\n"
    "【HonCut Identity Anchor】身份只通过 who 与 associate_assets 结构化绑定；visual 不重复外貌。"
    "who=[] 时不得写角色或绑定 char: 资产。\n\n"
    "【HonCut 资产绑定（associateAssetsIds）】每镜 associate_assets 必须列出可见角色"
    "（char:CHARACTER_ID）和必选场景（scene:LOCATION_ID）。\n\n"
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
    "撑住钢梁", "扣住手腕", "两刃碰撞",
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
    # Prefer authored action fields, while retaining compatibility with older
    # storyboards that wrote real movement only into ``visual``/``what``.
    # Noun-only tokens such as ``刀锋`` must not appear in _ACTION_VERBS: a
    # blade lying on a table is not evidence that FLF2V is required.
    authored_actions = shot.get("generation_actions") or []
    if isinstance(authored_actions, str):
        authored_actions = [authored_actions]
    searchable = " ".join(
        [str(value) for value in authored_actions]
        + [
            str(shot.get("action_description") or ""),
            str(shot.get("visual") or ""),
            str(shot.get("what") or ""),
        ]
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
CHARS_PER_SECOND = 4  # 中文剧本预估：约 4 字/秒（范围 3-5）
DEFAULT_TARGET_DURATION = 60  # 默认目标时长（用户未指定时使用）

# Phase 1 plans source material; Phase 8 owns the frame-exact delivery trim.
# Keep the delivery duration unchanged downstream while permitting at most 30%
# pre-edit coverage for redundant motion, continuity work and transition handles.
MAX_PRE_EDIT_DURATION_RATIO = 1.3


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


def estimate_shot_count(
    target_duration: int,
    shot_duration: int = AVG_SHOT_DURATION,
    capabilities: VideoModelCapabilities | None = None,
) -> int:
    """
    根据目标时长和单镜时长计算合理的镜头数量。

    单镜时长由所选视频模型及“一次基础段+有界延长段”合同共同限制。

    Args:
        target_duration: 目标总时长（秒）
        shot_duration: 每镜平均时长（秒）

    Returns:
        建议的最大镜头数
    """
    profile = capabilities or get_video_capabilities()
    semantic_maximum = max_primary_story_duration(profile)
    shot_duration = max(
        min_primary_story_duration(profile),
        min(semantic_maximum, shot_duration),
    )

    max_shots = max(1, (target_duration + shot_duration - 1) // shot_duration)
    return max_shots


def estimate_action_aware_shot_count(
    events: List[Dict[str, Any]],
    target_duration: int,
    requested_shot_duration: int,
) -> int:
    """Estimate editorial shots, leaving paid clip capacity to inner beats.

    An editorial shot is a director-level story unit, not one provider call.
    Dense action is expanded later into ``storyboard_beats`` where the first
    beat starts from an image and subsequent beats extend its video.

    Capacity math consumes normalized generation action units (see
    ``utils.action_units``), never the raw micro_actions ledger: sequential
    plot actions cost one unit each (deduplicated across events through a
    shared seen set), simultaneous composite motions merge into one unit,
    sustained states and camera constraints cost nothing.  The full ledger is
    preserved for audit and beat partitioning.
    """
    return _estimate_action_capacity_plan(
        events,
        target_duration,
        requested_shot_duration,
    )["primary_shots"]


def _minimum_primary_duration_for_units(
    generation_units: int,
    capabilities: VideoModelCapabilities,
) -> int:
    """Return executable story time for one base clip plus its extensions."""
    content_beats = max(
        1,
        math.ceil(generation_units / capabilities.max_micro_actions_per_beat),
    )
    if content_beats > MAX_CONTENT_BEATS_PER_PRIMARY_SHOT:
        raise ValueError(
            f"{generation_units} generation action units exceed one primary shot's "
            f"{MAX_CONTENT_BEATS_PER_PRIMARY_SHOT}-clip capacity"
        )
    first_minimum, _ = capabilities.effective_duration_bounds("multi_image")
    tail_minimum, _ = capabilities.effective_duration_bounds("tail_video_extend")
    return math.ceil(max(
        min_primary_story_duration(capabilities),
        first_minimum + max(0, content_beats - 1) * tail_minimum,
    ))


def _sequence_material_cost(
    generation_units: int,
    primary_shots: int,
    capabilities: VideoModelCapabilities,
    *,
    maximize: bool = False,
) -> float:
    """Find an executable allocation bound for one isolated sequence."""
    if primary_shots < 1:
        return math.inf
    unit_capacity = (
        capabilities.max_micro_actions_per_beat
        * MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    )
    if generation_units > primary_shots * unit_capacity:
        return math.inf

    costs: dict[int, float] = {0: 0.0}
    for _ in range(primary_shots):
        next_costs: dict[int, float] = {}
        for assigned, current_cost in costs.items():
            remaining = generation_units - assigned
            for shot_units in range(min(unit_capacity, remaining) + 1):
                total = assigned + shot_units
                candidate = current_cost + _minimum_primary_duration_for_units(
                    shot_units,
                    capabilities,
                )
                previous = next_costs.get(total)
                if previous is None or (
                    candidate > previous if maximize else candidate < previous
                ):
                    next_costs[total] = candidate
        costs = next_costs
    return costs.get(generation_units, math.inf)


def _sequence_generation_unit_counts(
    events: list[dict[str, Any]],
) -> list[tuple[str, int]]:
    """Aggregate normalized action units by non-mergeable screenplay sequence."""
    unit_counts = _event_generation_action_unit_counts(events)
    ordered: dict[str, int] = {}
    for event_id, event in enumerate(events, 1):
        sequence = str(event.get("sequence_id") or "").strip() or "__unspecified__"
        ordered.setdefault(sequence, 0)
        ordered[sequence] += unit_counts[event_id]
    return list(ordered.items())


def _material_duration_bound(
    sequence_units: list[tuple[str, int]],
    primary_shots: int,
    capabilities: VideoModelCapabilities,
    *,
    maximize: bool = False,
) -> float:
    """Return a layout-duration bound with every shot owned by one sequence."""
    allocations: dict[int, float] = {0: 0.0}
    unit_capacity = (
        capabilities.max_micro_actions_per_beat
        * MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    )
    for _sequence, units in sequence_units:
        minimum_shots = max(1, math.ceil(units / unit_capacity))
        next_allocations: dict[int, float] = {}
        for used, current_cost in allocations.items():
            for count in range(minimum_shots, primary_shots - used + 1):
                sequence_cost = _sequence_material_cost(
                    units,
                    count,
                    capabilities,
                    maximize=maximize,
                )
                if math.isinf(sequence_cost):
                    continue
                total_shots = used + count
                candidate = current_cost + sequence_cost
                previous = next_allocations.get(total_shots)
                if previous is None or (
                    candidate > previous if maximize else candidate < previous
                ):
                    next_allocations[total_shots] = candidate
        allocations = next_allocations
    return allocations.get(primary_shots, math.inf)


def _estimate_action_capacity_plan(
    events: list[dict[str, Any]],
    delivery_duration: int,
    requested_shot_duration: int,
) -> dict[str, Any]:
    """Plan truthful sequence packing inside the bounded pre-edit allowance."""
    profile = get_video_capabilities()
    baseline = estimate_shot_count(
        delivery_duration,
        requested_shot_duration,
        profile,
    )
    authored_events = [
        event
        for event in events
        if str(event.get("event_role") or "") != "drop"
    ]
    sequence_units = _sequence_generation_unit_counts(authored_events)
    if not sequence_units:
        sequence_units = [("__unspecified__", 0)]
    unit_capacity = (
        profile.max_micro_actions_per_beat * MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    )
    structural_shots = sum(
        max(1, math.ceil(units / unit_capacity))
        for _sequence, units in sequence_units
    )
    primary_shots = max(1, baseline, structural_shots)
    minimum_material_duration = _material_duration_bound(
        sequence_units,
        primary_shots,
        profile,
    )
    worst_case_material_duration = _material_duration_bound(
        sequence_units,
        primary_shots,
        profile,
        maximize=True,
    )
    maximum_material_duration = math.floor(
        float(delivery_duration) * MAX_PRE_EDIT_DURATION_RATIO + 1e-6
    )
    if minimum_material_duration > maximum_material_duration + 1e-6:
        sequence_summary = ", ".join(
            f"{sequence}={units}u"
            for sequence, units in sequence_units
        ) or "no authored sequences"
        raise ValueError(
            "cannot carry the authored action detail: "
            f"{delivery_duration}s delivery allows at most "
            f"{maximum_material_duration}s of pre-edit material (1.3x), but the "
            f"authored structure needs at least {minimum_material_duration:g}s across "
            f"{primary_shots} sequence-isolated primary shots for {profile.name}; "
            f"packing: {sequence_summary}"
        )
    return {
        "primary_shots": primary_shots,
        "structural_shots": structural_shots,
        "generation_action_units": sum(units for _sequence, units in sequence_units),
        "sequence_generation_action_units": dict(sequence_units),
        "minimum_material_duration": math.ceil(minimum_material_duration),
        "worst_case_material_duration": math.ceil(worst_case_material_duration),
        "maximum_material_duration": maximum_material_duration,
        "pre_edit_duration_ratio_limit": MAX_PRE_EDIT_DURATION_RATIO,
        "material_duration": max(
            int(delivery_duration),
            min(
                maximum_material_duration,
                math.ceil(worst_case_material_duration),
            ),
        ),
        "delivery_duration": int(delivery_duration),
        "capability_profile": profile.name,
    }


def _event_primary_occurrence_requirement(
    event: Dict[str, Any],
    capabilities: VideoModelCapabilities,
    seen: Optional[set] = None,
) -> int:
    """Return how many primary shots an event needs under the one-extension rule."""
    content_beats = _event_content_beat_requirement(event, capabilities, seen=seen)
    return max(
        1,
        math.ceil(content_beats / MAX_CONTENT_BEATS_PER_PRIMARY_SHOT),
    )


def _event_content_beat_requirement(
    event: Dict[str, Any],
    capabilities: VideoModelCapabilities,
    seen: Optional[set] = None,
) -> int:
    """Return content beats from normalized generation action units.

    Sustained states, camera constraints and cross-event duplicates cost
    nothing; simultaneous composite motions merge into one unit.  Pass a
    shared ``seen`` set across events in one gate pass to deduplicate
    repeated sequential actions globally.
    """
    actions = event.get("micro_actions") or []
    if isinstance(actions, str):
        actions = [actions]
    if not actions:
        return 0
    units = normalize_event_action_units(event, actions=actions, seen=seen)["units"]
    if units == 0:
        return 0
    return max(
        1,
        math.ceil(units / capabilities.max_micro_actions_per_beat),
    )


def _event_generation_action_unit_counts(
    events: List[Dict[str, Any]],
) -> Dict[int, int]:
    """Count normalized units for an ordered ledger with cross-event dedupe."""

    seen: set = set()
    return {
        event_id: normalize_event_action_units(event, seen=seen)["units"]
        for event_id, event in enumerate(events, 1)
    }


def _event_content_beat_requirements(
    events: List[Dict[str, Any]],
    capabilities: VideoModelCapabilities,
) -> Dict[int, int]:
    """Normalize an ordered event ledger once, including cross-event dedupe."""

    unit_counts = _event_generation_action_unit_counts(events)
    return {
        event_id: (
            math.ceil(units / capabilities.max_micro_actions_per_beat)
            if units else 0
        )
        for event_id, units in unit_counts.items()
    }


def _event_primary_occurrence_requirements(
    events: List[Dict[str, Any]],
    capabilities: VideoModelCapabilities,
) -> Dict[int, int]:
    content = _event_content_beat_requirements(events, capabilities)
    return {
        # Even a zero-cost state/camera event still needs one explicit primary
        # occurrence in the source ledger; zero only means it consumes no
        # generation action capacity.
        event_id: max(
            1,
            math.ceil(beats / MAX_CONTENT_BEATS_PER_PRIMARY_SHOT),
        )
        for event_id, beats in content.items()
    }


def select_generation_actions(
    micro_actions: List[str],
    limit: int | None = None,
    duration_seconds: float | None = None,
    capabilities: VideoModelCapabilities | None = None,
) -> List[str]:
    """Select a duration-bounded ordered motion contract for a model.

    ``micro_actions`` remains the complete screenplay ledger. The generated
    prompt receives a bounded representative sequence, because asking a video
    model to perform eight to twenty-six atomic movements in one clip causes it
    to hold the reference pose and omit the action altogether. Four-to-five
    second clips deliberately receive one visible authored action; the other
    source actions remain in the audit ledger for adaptation decisions.
    """
    profile = capabilities or get_video_capabilities()
    if limit is None:
        limit = profile.action_limit(None)
    if duration_seconds is not None:
        limit = min(limit, profile.action_limit(duration_seconds))
    actions = [str(value).strip() for value in micro_actions if str(value).strip()]
    if len(actions) <= limit:
        return actions
    if limit <= 1:
        return actions[:1]
    indices = [round(index * (len(actions) - 1) / (limit - 1)) for index in range(limit)]
    return [actions[index] for index in dict.fromkeys(indices)]


def generation_action_limit(
    duration_seconds: float | int | None,
    *,
    model: str | None = None,
    provider: str | None = None,
    capabilities: VideoModelCapabilities | None = None,
) -> int:
    """Return the action budget owned by the selected video-model profile."""

    profile = capabilities or get_video_capabilities(model=model, provider=provider)
    return profile.action_limit(duration_seconds)


def normalize_shot_durations(
    shots: List[Dict[str, Any]],
    target_duration: int,
    capabilities: VideoModelCapabilities | None = None,
) -> List[Dict[str, Any]]:
    """Assign exact seconds, giving dense action/dialogue enough Pxx capacity."""
    if not shots:
        return shots
    profile = capabilities or capabilities_for(shots[0])
    primary_minimum = math.ceil(min_primary_story_duration(profile))
    primary_maximum = int(max_primary_story_duration(profile))

    def requirements(
        shot: Dict[str, Any],
        *,
        bridge_required: bool,
    ) -> tuple[float, int, int]:
        actions = shot.get("micro_actions") or []
        if isinstance(actions, str):
            actions = [actions]
        details = shot.get("_source_event_details") or []
        if details:
            detail_actions = [
                action
                for event in details if isinstance(event, dict)
                for action in (event.get("micro_actions") or [])
                if str(action).strip()
            ]
            actions = actions or detail_actions
        if "generation_action_units" in shot:
            raw_generation_units = shot.get("generation_action_units") or []
            generation_unit_count = len(raw_generation_units)
        else:
            generation_unit_count = normalized_action_unit_count(actions)
        spoken = float(shot.get("speech_duration_s") or 0)
        action_beats = math.ceil(
            generation_unit_count / profile.max_micro_actions_per_beat
        ) if generation_unit_count else 1
        spoken_beats = math.ceil(spoken / profile.max_unique_beat_s) if spoken else 1
        content_beats = max(1, action_beats, spoken_beats)
        if content_beats > MAX_CONTENT_BEATS_PER_PRIMARY_SHOT:
            raise ValueError(
                f"a primary shot requires {content_beats} story-bearing clips for "
                f"{profile.name}; split the source event before duration allocation"
            )
        first_minimum, _first_maximum = profile.effective_duration_bounds(
            "multi_image"
        )
        tail_minimum, _tail_maximum = profile.effective_duration_bounds(
            "tail_video_extend"
        )
        lower = math.ceil(max(
            primary_minimum,
            first_minimum + max(0, content_beats - 1) * tail_minimum,
            spoken,
        ))
        upper = primary_maximum
        weight = float(max(content_beats, spoken / max(profile.min_unique_beat_s, 1)))
        return weight, lower, upper

    constraints = []
    for index, shot in enumerate(shots):
        bridge_required = (
            index + 1 < len(shots)
            and str(shots[index + 1].get("boundary_before") or "").lower()
            == "continuous"
        )
        constraints.append(requirements(shot, bridge_required=bridge_required))
    weights = [item[0] for item in constraints]
    lower_bounds = [item[1] for item in constraints]
    upper_bounds = [item[2] for item in constraints]
    if sum(lower_bounds) > target_duration:
        raise ValueError(
            f"{len(shots)} shots need at least {sum(lower_bounds)}s to preserve all "
            f"actions across {profile.name}'s 15-30s primary-shot contract, "
            f"above the {target_duration}s target"
        )
    if sum(upper_bounds) < target_duration:
        raise ValueError(
            f"{len(shots)} shots can carry at most {sum(upper_bounds)}s under the "
            f"bounded-extension {profile.name} contract, below the {target_duration}s target"
        )

    allocations = list(lower_bounds)
    remaining = int(target_duration) - sum(allocations)
    # Weighted fair allocation preserves exact total duration and provider caps.
    while remaining:
        candidates = [
            index for index, value in enumerate(allocations)
            if value < upper_bounds[index]
        ]
        if not candidates:
            raise ValueError("duration allocation exhausted provider capacity")
        selected = max(
            candidates,
            key=lambda index: (weights[index] / allocations[index], -index),
        )
        allocations[selected] += 1
        remaining -= 1
    for index, (shot, duration) in enumerate(zip(shots, allocations, strict=True)):
        shot["suggested_duration"] = duration
        shot["duration_allocation"] = {
            "method": "semantic_weighted_provider_bounded",
            "complexity_weight": weights[index],
            "capability_profile": profile.name,
            "minimum_seconds": lower_bounds[index],
            "maximum_seconds": upper_bounds[index],
        }
    return shots


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


_SHOT_LANGUAGE_FIELDS = (
    "shot_size",
    "camera_movement",
    "lighting_key",
    "shot_intent",
    "hero_moment",
    "texture_keywords",
)


def _validate_authored_shot_language(
    shots: List[Dict[str, Any]], *, label: str
) -> None:
    """Require explicit shot language where the planner owns the contract."""
    for index, shot in enumerate(shots, 1):
        missing = set(_SHOT_LANGUAGE_FIELDS) - set(shot)
        if missing:
            raise ValueError(f"第 {index} 个 {label} 缺少镜头语言字段: {missing}")
        if shot["shot_size"] not in _VALID_SHOT_SIZES:
            raise ValueError(
                f"第 {index} 个 {label} shot_size 无效: {shot['shot_size']}; "
                f"合法值: {', '.join(_SHOT_SIZE_VALUES)}"
            )
        if shot["camera_movement"] not in _VALID_CAMERA_MOVEMENTS:
            raise ValueError(
                f"第 {index} 个 {label} camera_movement 无效: "
                f"{shot['camera_movement']}; 合法值: "
                f"{', '.join(_CAMERA_MOVEMENT_VALUES)}"
            )
        if shot["lighting_key"] not in _VALID_LIGHTING_KEYS:
            raise ValueError(
                f"第 {index} 个 {label} lighting_key 无效: {shot['lighting_key']}; "
                f"合法值: {', '.join(_LIGHTING_KEY_VALUES)}"
            )
        if shot["shot_intent"] not in _VALID_SHOT_INTENTS:
            raise ValueError(
                f"第 {index} 个 {label} shot_intent 无效: {shot['shot_intent']}; "
                f"合法值: {', '.join(_SHOT_INTENT_VALUES)}"
            )
        if not isinstance(shot["hero_moment"], bool):
            raise ValueError(f"第 {index} 个 {label} hero_moment 必须是布尔值")
        textures = shot["texture_keywords"]
        if (
            not isinstance(textures, list)
            or not 2 <= len(textures) <= 4
            or any(not isinstance(value, str) or not value.strip() for value in textures)
        ):
            raise ValueError(
                f"第 {index} 个 {label} texture_keywords 必须包含 2–4 个非空字符串"
            )


def _validate_shot_language_variation(
    shots: List[Dict[str, Any]], *, minimum_quality: float = 3.0
) -> Dict[str, Any]:
    """Fail in Phase 1 before paid storyboard/video work can start."""
    from quality.variation_checker import check_scene_variation

    report = check_scene_variation(shots)
    quality = round(5.0 - float(report.get("score", 5.0)), 2)
    report = {**report, "quality": quality}
    if quality < minimum_quality:
        violations = "; ".join(str(value) for value in report["violations"])
        raise ValueError(
            f"shot-language variation quality {quality:g}/5 is below "
            f"{minimum_quality:g} before paid storyboard generation: {violations}"
        )
    return report


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

    # 兼容旧单次调用/检查点；全局骨架路径会在此前严格要求显式字段。
    # 如果 LLM 没返回，给默认值而不是缺失
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
        if not isinstance(shot.get("hero_moment"), bool):
            shot["hero_moment"] = False
        textures = shot.get("texture_keywords")
        if isinstance(textures, str):
            textures = [textures]
        if not isinstance(textures, list):
            textures = []
        shot["texture_keywords"] = [
            str(value).strip() for value in textures if str(value).strip()
        ][:4]
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
        apply_camera_motion_contract(shot)
    return shots


def _parse_response(
    response: str, *, require_authored_shot_language: bool = False
) -> Dict[str, Any]:
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

    if require_authored_shot_language:
        _validate_authored_shot_language(parsed["shots"], label="shot")
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


def _normalize_character_reference(value: Any) -> str:
    """Backward-compatible wrapper around the shared identity normalizer."""
    return normalize_character_reference(value)


def _canonical_character_name(
    value: Any,
    characters: Optional[List[Dict[str, Any]]],
) -> str:
    """Resolve a source mention or qualified description to one character name."""
    original = str(value or "").strip()
    return resolve_character_name(original, characters) or original


def _canonicalize_shot_characters(
    shots: List[Dict[str, Any]],
    characters: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Canonicalize ``who`` while retaining original source mentions for audit."""
    if not characters:
        return shots
    for shot in shots:
        raw_who = shot.get("who") or []
        if isinstance(raw_who, str):
            raw_who = [raw_who]
        original = [str(name).strip() for name in raw_who if str(name).strip()]
        canonical = list(
            dict.fromkeys(
                _canonical_character_name(name, characters) for name in original
            )
        )
        canonical = [name for name in canonical if name]
        if canonical != original:
            shot["source_character_mentions"] = original
        shot["who"] = canonical
    return shots


def _inherit_event_semantics(
    shots: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    characters: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Carry source screenplay evidence into shots after the LLM adaptation pass.

    The model still chooses framing and whether a camera cut is useful, while
    source ordering, exact excerpts, action units, and speaker evidence remain
    deterministic and auditable downstream.
    """
    event_by_id = {index: event for index, event in enumerate(events, 1)}
    event_occurrences: Dict[int, List[int]] = {}
    for shot_index, shot in enumerate(shots):
        raw_ids = shot.get("source_events", [])
        source_ids = raw_ids if isinstance(raw_ids, list) else []
        for event_id in dict.fromkeys(source_ids):
            if event_id in event_by_id:
                event_occurrences.setdefault(event_id, []).append(shot_index)

    event_slices: Dict[tuple[int, int], Dict[str, Any]] = {}
    for event_id, occurrence_shots in event_occurrences.items():
        event = event_by_id[event_id]
        actions = [
            str(action).strip()
            for action in (event.get("micro_actions") or [])
            if str(action).strip()
        ]
        base, remainder = divmod(len(actions), len(occurrence_shots))
        cursor = 0
        previous_state = str(event.get("start_state") or "").strip()
        for occurrence, shot_index in enumerate(occurrence_shots):
            size = base + (1 if occurrence < remainder else 0)
            action_slice = actions[cursor : cursor + size]
            cursor += size
            is_last = occurrence == len(occurrence_shots) - 1
            if is_last:
                end_state = str(event.get("end_state") or "").strip()
            elif action_slice:
                end_state = f"已完成动作：{action_slice[-1]}"
            else:
                end_state = previous_state
            event_slices[(event_id, shot_index)] = {
                "micro_actions": action_slice,
                "start_state": previous_state,
                "end_state": end_state,
                "occurrence": occurrence + 1,
                "occurrence_count": len(occurrence_shots),
            }
            previous_state = end_state

    # Capture the actionable keys owned by earlier source events once.  Every
    # slice of the same event receives the same snapshot, so a deliberate
    # repeated action split across two shots is preserved while a later event's
    # exact repeat is still classified as a cross-event duplicate.
    seen_before_event: Dict[int, set[str]] = {}
    preceding_event_keys: set[str] = set()
    for event_id, event in event_by_id.items():
        seen_before_event[event_id] = set(preceding_event_keys)
        event_actions = event.get("micro_actions") or []
        if isinstance(event_actions, str):
            event_actions = [event_actions]
        normalize_event_action_units(
            event,
            actions=[str(action).strip() for action in event_actions if str(action).strip()],
            seen=preceding_event_keys,
        )

    previous_sequence_ids: List[str] = []
    for shot_index, shot in enumerate(shots):
        raw_ids = shot.get("source_events", [])
        source_ids = list(dict.fromkeys(raw_ids)) if isinstance(raw_ids, list) else []
        details = [event_by_id[event_id] for event_id in source_ids if event_id in event_by_id]
        slices = [
            (event_id, event_slices[(event_id, shot_index)])
            for event_id in source_ids
            if (event_id, shot_index) in event_slices
        ]

        canonical_who = list(dict.fromkeys(
            str(name).strip()
            for event in details
            for name in (event.get("who") or [])
            if str(name).strip()
        ))
        if details:
            # Character identity is a source-ledger contract. A model synonym
            # would break downstream reference lookup and falsely report a
            # disappearance, so source labels are restored before resolution.
            shot["who"] = canonical_who

        excerpts = [str(event.get("source_excerpt") or "").strip() for event in details]
        excerpts = [excerpt for excerpt in excerpts if excerpt]
        if excerpts:
            shot["source_excerpt"] = "\n".join(dict.fromkeys(excerpts))

        sequence_ids = [str(event.get("sequence_id")) for event in details if event.get("sequence_id")]
        sequence_ids = list(dict.fromkeys(sequence_ids))
        action_unit_ids = [str(event.get("action_unit_id")) for event in details if event.get("action_unit_id")]
        micro_actions = [
            action
            for _event_id, event_slice in slices
            for action in event_slice["micro_actions"]
        ]
        roles = [str(event.get("event_role")) for event in details if event.get("event_role")]
        shot["source_sequence_ids"] = sequence_ids
        shot["source_action_unit_ids"] = list(dict.fromkeys(action_unit_ids))
        shot["source_event_roles"] = list(dict.fromkeys(roles))
        shot["source_event_slices"] = [
            {
                "event_id": event_id,
                "occurrence": event_slice["occurrence"],
                "occurrence_count": event_slice["occurrence_count"],
                "micro_actions": list(event_slice["micro_actions"]),
            }
            for event_id, event_slice in slices
        ]
        shot["micro_actions"] = micro_actions
        generation_units: List[Dict[str, Any]] = []
        generation_categories: List[str] = []
        ledger_offset = 0
        for event_id, event_slice in slices:
            slice_actions = list(event_slice["micro_actions"])
            normalized = normalize_event_action_units(
                event_by_id[event_id],
                actions=slice_actions,
                seen=set(seen_before_event[event_id]),
            )
            generation_categories.extend(normalized["categories"])
            for unit in normalized["generation_action_units"]:
                serialized = dict(unit)
                serialized["unit_id"] = f"GAU{len(generation_units) + 1:03d}"
                serialized["ledger_indexes"] = [
                    ledger_offset + int(index)
                    for index in unit.get("ledger_indexes", [])
                ]
                serialized["source_event_id"] = event_id
                source_action_unit_id = str(
                    event_by_id[event_id].get("action_unit_id") or ""
                ).strip()
                if source_action_unit_id:
                    serialized["source_action_unit_id"] = source_action_unit_id
                generation_units.append(serialized)
            ledger_offset += len(slice_actions)
        if not slices and micro_actions:
            normalized = normalize_action_units(micro_actions)
            generation_categories = list(normalized["categories"])
            generation_units = [
                dict(unit) for unit in normalized["generation_action_units"]
            ]
        shot["generation_action_units"] = generation_units
        shot["generation_action_categories"] = generation_categories
        generation_actions = select_generation_actions(
            micro_actions,
            duration_seconds=shot.get("suggested_duration") or shot.get("duration"),
        )
        shot["generation_actions"] = generation_actions
        shot["generation_load"] = {
            "source_action_units": len(set(shot["source_action_unit_ids"])),
            "source_micro_actions": len(micro_actions),
            "generation_action_units": len(generation_units),
            "prompted_actions": len(generation_actions),
            "compression": "representative" if len(generation_actions) < len(micro_actions) else "full",
        }
        if generation_actions:
            shot["action_description"] = " → ".join(generation_actions)
            shot["gen_strategy"] = determine_gen_strategy(shot)

        first_source = details[0] if details else {}
        apply_temporal_visual_contract(
            shot,
            source_times=(event.get("time") for event in details),
        )
        if slices and slices[0][1].get("start_state"):
            shot["start_state"] = str(slices[0][1]["start_state"])
        if slices and slices[-1][1].get("end_state"):
            shot["end_state"] = str(slices[-1][1]["end_state"])
        causal_links = [
            str(event.get("causal_link") or "").strip()
            for event in details
            if str(event.get("causal_link") or "").strip()
        ]
        if causal_links:
            shot["causal_link"] = "；".join(dict.fromkeys(causal_links))

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
    return _canonicalize_shot_characters(shots, characters)


BEAT_SKELETON_PROMPT = (
    "目标时长：{target_duration}秒，每镜约{shot_duration}秒。请把全部事件压缩为恰好{beat_count}个 beat。\n\n"
    "事件列表：\n{events_json}\n\n角色列表：\n{characters_summary}\n\n"
    "只做全局改编与镜头语言规划，不要展开 visual 或人物外貌。输出严格 JSON 对象："
    '{{"strategy":"一句话改编策略","beats":[{{"beat_order":1,"source_events":[1],'
    '"action":"keep/merge/drop","reason":"一句话理由","who":["角色主名"],'
    '"where":"地点","what":"一句话事件","suggested_duration":15,'
    '"shot_size":"medium_wide","camera_movement":"dolly_in",'
    '"lighting_key":"natural","shot_intent":"establishing",'
    '"hero_moment":false,"texture_keywords":["场景中的具体材质","场景中的具体光影"]}}]}}。\n'
    + "【镜头语言合法词表】以下四个枚举字段只能逐字选用所列值，禁止发明组合值或方向后缀：\n"
    + _SHOT_LANGUAGE_ENUM_CONTRACT
    + CAMERA_MOTION_PLANNING_INSTRUCTIONS
    + "\n"
    + "hero_moment 必须为 JSON 布尔值；texture_keywords 必须为 2–4 个非空字符串。\n"
    + "【全局铁律】\n"
    "1. beats 数量必须恰好等于 {beat_count}，总建议时长应接近 {target_duration} 秒（±10%）；"
    "每个 beat 的 generation_action_unit_count 合计不得超过 "
    "{max_generation_action_units_per_beat}。\n"
    "2. 每个输入事件编号必须且至少被某个 beat 的 source_events 引用；删减事件也必须放入 action=drop 的 beat 显式声明。"
    "同一事件若分配给多个 beat，后一个 beat 必须承接前一个，只负责该事件后续尚未表现的动作，禁止重放整段事件。\n"
    "3. keep 保留关键因果/情感节点，merge 合并连续或重复事件，drop 只删不影响因果链的内容。\n"
    "4. 台词归属必须忠于原事件；who 只能使用角色列表主名，别名改为主名，群众不得写入 who。\n"
    "5. beat 是导演级叙事镜头，不是单次视频调用。同一 sequence_id 的连续 action_unit 可以合并，"
    "但必须完整保留 source_events 与 micro_actions 原顺序，后续会拆成 P01/P02…；不同 sequence_id、"
    "换场/跳时不得错误合并。turning_point 可与同 sequence 中紧邻的因果动作共用 beat，但必须明确"
    "保留转折。每个事件在 beats 中的引用次数不得少于输入中的 "
    "minimum_primary_beat_occurrences；同一事件的后续引用只承载尚未表现的动作，不得重放。\n"
    "6. sequence_id 与 continuity_before 是生成连续性依据。同一 sequence 的连续单元尽量落在相邻 beat，"
    "换场/跳时/关系转折不得为了省镜头而错误连拍。\n"
    "7. shot_size、camera_movement、lighting_key、shot_intent、hero_moment、texture_keywords "
    "是骨架的全局结构字段，全部必填。相邻 beat 景别必须形成差异，动作 beat 不得全部 static；"
    "4 个及以上 beat 必须至少一个 hero_moment=true，每个 beat 给出 2–4 个具体纹理关键词。"
    "一镜到底可以保持同一真实光源，但镜头变化必须来自源文本允许的构图距离与运镜；禁止为了"
    "追求差异虚构转场、跳时、摇臂、无人机、环绕或剧本明令禁止的运镜。禁止展开对白、visual、"
    "Identity Anchor 或人物外貌。"
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
    required = {
        "beat_order",
        "source_events",
        "action",
        "reason",
        "who",
        "where",
        "what",
        "suggested_duration",
        *_SHOT_LANGUAGE_FIELDS,
    }
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
    _validate_authored_shot_language(beats, label="beat")
    for beat in beats:
        apply_camera_motion_contract(beat)
    parsed.setdefault("strategy", "")
    return parsed


def _validate_beat_action_capacity(
    beats: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    capabilities: VideoModelCapabilities | None = None,
) -> None:
    """Allow inner Pxx expansion while rejecting unrelated narrative merges."""
    profile = capabilities or get_video_capabilities()
    event_by_id = {i: event for i, event in enumerate(events, 1)}
    occurrence_requirements = _event_primary_occurrence_requirements(events, profile)
    for beat in beats:
        if beat.get("action") == "drop":
            continue
        details = [
            event_by_id[event_id]
            for event_id in beat.get("source_events", [])
            if event_id in event_by_id
        ]
        sequences = {
            str(event.get("sequence_id"))
            for event in details
            if str(event.get("sequence_id") or "").strip()
        }
        if len(sequences) > 1:
            raise ValueError(
                f"beat {beat.get('beat_order')} merges unrelated sequences "
                f"{sorted(sequences)}"
            )
    content_loads = _beat_content_loads(beats, events, profile)
    for index, content_beats in enumerate(content_loads, 1):
        if content_beats > MAX_CONTENT_BEATS_PER_PRIMARY_SHOT:
            raise ValueError(
                f"beat {index} requires {content_beats} story-bearing clips for "
                f"{profile.name}; maximum is {MAX_CONTENT_BEATS_PER_PRIMARY_SHOT}"
            )
    for event_id, event in event_by_id.items():
        observed = sum(
            event_id in beat.get("source_events", [])
            for beat in beats
            if beat.get("action") != "drop"
        )
        required = occurrence_requirements[event_id]
        if required > 1 and observed < required:
            actions = event.get("micro_actions") or []
            if isinstance(actions, str):
                actions = [actions]
            raise ValueError(
                f"event {event_id} requires at least {required} primary beats to carry "
                f"its normalized generation action units while preserving all "
                f"{len(actions)} micro-actions; observed {observed}"
            )


def _validate_beat_material_duration(
    beats: list[dict[str, Any]],
    events: list[dict[str, Any]],
    material_duration: int,
    capabilities: VideoModelCapabilities,
) -> None:
    """Reject a capacity-valid layout whose clip minima exceed its material clock."""
    content_loads = _beat_content_loads(beats, events, capabilities)
    first_minimum, _ = capabilities.effective_duration_bounds("multi_image")
    tail_minimum, _ = capabilities.effective_duration_bounds("tail_video_extend")
    lower_bounds = [
        math.ceil(max(
            min_primary_story_duration(capabilities),
            first_minimum + max(0, load - 1) * tail_minimum,
        ))
        for load in content_loads
    ]
    required = sum(lower_bounds)
    if required > material_duration:
        raise ValueError(
            f"beat layout needs at least {required}s of provider-executable material; "
            f"budget is {material_duration}s (per-beat minima: {lower_bounds})"
        )


def _beat_content_loads(
    beats: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    capabilities: VideoModelCapabilities,
) -> List[int]:
    """Calculate the eventual per-shot action load after event occurrence slicing."""
    event_by_id = {index: event for index, event in enumerate(events, 1)}
    positions: Dict[int, List[int]] = {}
    for beat_index, beat in enumerate(beats):
        if beat.get("action") == "drop":
            continue
        raw_ids = beat.get("source_events") or []
        if not isinstance(raw_ids, list):
            raise ValueError(f"beat {beat_index + 1} source_events must be an array")
        for event_id in dict.fromkeys(raw_ids):
            if event_id not in event_by_id:
                raise ValueError(
                    f"beat {beat_index + 1} references invalid event {event_id}"
                )
            positions.setdefault(event_id, []).append(beat_index)

    generation_units_by_beat: List[int] = [0 for _ in beats]
    generation_unit_counts = _event_generation_action_unit_counts(events)
    for event_id, occurrence_positions in positions.items():
        generation_unit_count = generation_unit_counts[event_id]
        base, remainder = divmod(generation_unit_count, len(occurrence_positions))
        for occurrence, beat_index in enumerate(occurrence_positions):
            size = base + (1 if occurrence < remainder else 0)
            generation_units_by_beat[beat_index] += size

    loads = []
    for generation_unit_count in generation_units_by_beat:
        action_beats = (
            math.ceil(
                generation_unit_count / capabilities.max_micro_actions_per_beat
            )
            if generation_unit_count else 1
        )
        loads.append(max(1, action_beats))
    return loads


def _repair_beat_action_capacity(
    beats: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    capabilities: VideoModelCapabilities | None = None,
) -> List[Dict[str, Any]]:
    """Deterministically add missing dense-event occurrences to adjacent beats.

    The LLM still chooses editorial framing. This repair owns the non-negotiable
    source ledger: event order, sequence boundaries, turning points, and the
    provider's per-primary-shot content capacity.
    """
    profile = capabilities or get_video_capabilities()
    repaired = [dict(beat) for beat in beats]
    for beat in repaired:
        raw_ids = beat.get("source_events") or []
        if not isinstance(raw_ids, list):
            raise ValueError("source_events must be an array before capacity repair")
        beat["source_events"] = list(dict.fromkeys(raw_ids))

    event_by_id = {index: event for index, event in enumerate(events, 1)}
    occurrence_requirements = _event_primary_occurrence_requirements(events, profile)

    def legal_merge(event_id: int, beat: Dict[str, Any]) -> bool:
        if beat.get("action") == "drop":
            return False
        other_ids = [
            value for value in beat.get("source_events", []) if value != event_id
        ]
        if not other_ids:
            return True
        event = event_by_id[event_id]
        details = [event_by_id[value] for value in other_ids if value in event_by_id]
        sequence = str(event.get("sequence_id") or "").strip()
        other_sequences = {
            str(detail.get("sequence_id") or "").strip()
            for detail in details
            if str(detail.get("sequence_id") or "").strip()
        }
        if sequence and other_sequences and other_sequences != {sequence}:
            return False
        return True

    for event_id, event in event_by_id.items():
        required = occurrence_requirements[event_id]
        if required <= 1:
            continue
        while True:
            current = [
                index for index, beat in enumerate(repaired)
                if beat.get("action") != "drop"
                and event_id in beat.get("source_events", [])
            ]
            if len(current) >= required:
                break
            lower = max(
                (
                    index for index, beat in enumerate(repaired)
                    if any(value < event_id for value in beat.get("source_events", []))
                ),
                default=0,
            )
            upper = min(
                (
                    index for index, beat in enumerate(repaired)
                    if any(value > event_id for value in beat.get("source_events", []))
                ),
                default=len(repaired) - 1,
            )
            candidates = []
            for candidate in range(lower, upper + 1):
                if candidate in current or not legal_merge(event_id, repaired[candidate]):
                    continue
                trial = [dict(beat) for beat in repaired]
                trial[candidate]["source_events"] = sorted(
                    [*trial[candidate].get("source_events", []), event_id]
                )
                loads = _beat_content_loads(trial, events, profile)
                if max(loads, default=1) > MAX_CONTENT_BEATS_PER_PRIMARY_SHOT:
                    continue
                distance = min((abs(candidate - value) for value in current), default=0)
                candidates.append(
                    (loads[candidate], distance, len(trial[candidate]["source_events"]), candidate)
                )
            if not candidates:
                raise ValueError(
                    f"event {event_id} needs {required} primary beats, but no adjacent "
                    "beat can accept another occurrence without crossing a sequence "
                    "or exceeding provider capacity"
                )
            candidate = min(candidates)[-1]
            repaired[candidate]["source_events"] = sorted(
                [*repaired[candidate].get("source_events", []), event_id]
            )
            repaired[candidate]["capacity_repair"] = {
                "reason": "dense_event_occurrence_added",
                "event_id": event_id,
                "minimum_occurrences": required,
            }

    for beat in repaired:
        details = [
            event_by_id[event_id]
            for event_id in beat.get("source_events", [])
            if event_id in event_by_id
        ]
        if not details or "capacity_repair" not in beat:
            continue
        beat["action"] = "merge" if len(details) > 1 else "keep"
        beat["who"] = list(dict.fromkeys(
            str(name)
            for detail in details
            for name in (detail.get("who") or [])
            if str(name).strip()
        ))
        locations = list(dict.fromkeys(
            str(detail.get("where") or "").strip()
            for detail in details
            if str(detail.get("where") or "").strip()
        ))
        if len(locations) == 1:
            beat["where"] = locations[0]
        descriptions = [
            str(detail.get("what") or "").strip()
            for detail in details
            if str(detail.get("what") or "").strip()
        ]
        visuals = [
            str(detail.get("visual") or "").strip()
            for detail in details
            if str(detail.get("visual") or "").strip()
        ]
        if descriptions:
            beat["what"] = "；随后".join(descriptions)
        if visuals:
            beat["visual"] = "按来源剧情顺序：" + "；随后".join(visuals)
        existing_reason = str(beat.get("reason") or "").strip()
        beat["reason"] = "；".join(filter(None, [
            existing_reason,
            "代码按视频容量补齐稠密事件的相邻一级分镜出现次数",
        ]))

    _validate_beat_action_capacity(repaired, events, profile)
    return repaired


def _build_beat_skeleton(
    events: List[Dict[str, Any]],
    characters_summary: str,
    target_duration: int,
    shot_duration: int,
    beat_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a globally informed, bounded beat table (Stage 1)."""
    profile = get_video_capabilities()
    beat_count = beat_count or estimate_shot_count(
        target_duration,
        shot_duration,
        profile,
    )
    occurrence_requirements = _event_primary_occurrence_requirements(events, profile)
    generation_unit_counts = _event_generation_action_unit_counts(events)
    prompt_events = []
    for event_id, event in enumerate(events, 1):
        prompt_event = dict(event)
        prompt_event["minimum_primary_beat_occurrences"] = occurrence_requirements[
            event_id
        ]
        prompt_event["generation_action_unit_count"] = generation_unit_counts[event_id]
        prompt_events.append(prompt_event)
    prompt = BEAT_SKELETON_PROMPT.format(
        target_duration=target_duration,
        shot_duration=shot_duration,
        beat_count=beat_count,
        max_generation_action_units_per_beat=(
            profile.max_micro_actions_per_beat
            * MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
        ),
        events_json=_build_events_json(prompt_events),
        characters_summary=characters_summary,
    )
    last_validation_error = ""
    for attempt in range(1 + MAX_RETRIES):
        try:
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\n\n【重试纠错】上次骨架合并了不相关叙事。同一 sequence_id 的连续"
                    "action_unit 可以合并并保留完整顺序；跨 sequence 必须拆开；turning_point"
                    "必须明确保留但可以与同 sequence 的紧邻因果动作共用 beat。"
                    f"每个 beat 最多承载 {profile.max_micro_actions_per_beat * MAX_CONTENT_BEATS_PER_PRIMARY_SHOT}"
                    "个 generation_action_units。"
                    "每个事件在 beats 中出现的次数不得少于它的 "
                    "minimum_primary_beat_occurrences；重复引用只分担尚未表现的后续动作。"
                )
                if last_validation_error:
                    attempt_prompt += (
                        "\n上次响应的具体失败原因："
                        f"{last_validation_error}。必须修正该结构问题后再输出。"
                    )
            response = _call_llm_with_timeout_retry(attempt_prompt, max_tokens=8000)
            skeleton = _parse_beat_skeleton(response, beat_count, len(events))
            skeleton["beats"] = _repair_beat_action_capacity(
                skeleton["beats"], events, profile
            )
            skeleton["shot_language_plan"] = _validate_shot_language_variation(
                skeleton["beats"]
            )
            _validate_beat_action_capacity(skeleton["beats"], events, profile)
            if target_duration >= len(skeleton["beats"]) * math.ceil(
                min_primary_story_duration(profile)
            ):
                _validate_beat_material_duration(
                    skeleton["beats"],
                    events,
                    target_duration,
                    profile,
                )
            event_by_id = {i: dict(event, event_id=i) for i, event in enumerate(events, 1)}
            for beat in skeleton["beats"]:
                beat["_source_event_details"] = [event_by_id[event_id] for event_id in beat["source_events"]]
            return skeleton
        except (json.JSONDecodeError, ValueError) as e:
            last_validation_error = str(e)
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
    checkpoint_fingerprint: Optional[str] = None,
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
                    "shot_intent", "hero_moment", "texture_keywords", "dialogue",
                    "gen_strategy",
                }
                for index, shot in enumerate(parsed["shots"], 1):
                    beat = batch[index - 1]
                    for field in _SHOT_LANGUAGE_FIELDS:
                        value = beat[field]
                        shot[field] = list(value) if isinstance(value, list) else value
                    apply_camera_motion_contract(shot)
                    missing = expanded_fields - set(shot)
                    if missing:
                        raise ValueError(f"本批第 {index} 镜缺少完整字段: {missing}")
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
                    "_checkpoint": {
                        "schema": LAYERED_CHECKPOINT_SCHEMA,
                        "input_fingerprint": checkpoint_fingerprint,
                    },
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


LAYERED_CHECKPOINT_SCHEMA = "honcut.layered-adaptation.v5"


def _layered_input_fingerprint(
    events: List[Dict[str, Any]],
    characters_summary: str,
    target_duration: int,
    shot_duration: int,
    expected_beats: int,
) -> str:
    """Bind layered checkpoints to the complete semantic adaptation input."""
    contract = {
        "schema": LAYERED_CHECKPOINT_SCHEMA,
        "events": events,
        "characters_summary": characters_summary,
        "target_duration": target_duration,
        "shot_duration": shot_duration,
        "expected_beats": expected_beats,
    }
    encoded = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_matches(value: Any, input_fingerprint: str) -> bool:
    metadata = value.get("_checkpoint") if isinstance(value, dict) else None
    return bool(
        isinstance(metadata, dict)
        and metadata.get("schema") == LAYERED_CHECKPOINT_SCHEMA
        and metadata.get("input_fingerprint") == input_fingerprint
    )


def _load_layered_checkpoints(
    output_dir: Path,
    events: List[Dict[str, Any]],
    expected_beats: int,
    input_fingerprint: str,
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load only valid, contiguous layered checkpoints."""
    skeleton = None
    skeleton_path = output_dir / "beat_skeleton.json"
    if skeleton_path.exists():
        try:
            candidate = json.loads(skeleton_path.read_text(encoding="utf-8"))
            if not _checkpoint_matches(candidate, input_fingerprint):
                raise ValueError("layered skeleton belongs to a different input")
            _parse_beat_skeleton(json.dumps(candidate, ensure_ascii=False), expected_beats, len(events))
            _validate_beat_action_capacity(candidate["beats"], events)
            candidate["shot_language_plan"] = _validate_shot_language_variation(
                candidate["beats"]
            )
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
            if not _checkpoint_matches(partial, input_fingerprint):
                raise ValueError("partial shots belong to a different input")
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
        target_duration: 最终交付时长（秒），默认 None（根据剧本长度智能预估）
        shot_duration: 每镜平均时长（秒），默认 12
        source_text: 原始剧本文本（用于智能预估时长）

    Returns:
        包含交付时长、剪辑前素材时长、容量计划与 shots 的字典

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

    capability_profile = get_video_capabilities()
    if shot_duration <= 0:
        raise ValueError(f"每镜时长不合理：{shot_duration}秒（必须大于 0）")

    # ── 分离交付时长与 Phase 8 剪辑前的素材时长 ───────────────────────────
    capacity_plan = _estimate_action_capacity_plan(
        events,
        target_duration,
        shot_duration,
    )
    max_shots = capacity_plan["primary_shots"]
    material_duration = capacity_plan["material_duration"]
    effective_shot_duration = max(
        min_primary_story_duration(capability_profile),
        min(
            max_primary_story_duration(capability_profile),
            round(material_duration / max_shots),
        ),
    )

    # ── 构建 prompt ───────────────────────────────────────────────────────
    occurrence_requirements = _event_primary_occurrence_requirements(
        events, capability_profile
    )
    generation_unit_counts = _event_generation_action_unit_counts(events)
    prompt_events = []
    for event_id, event in enumerate(events, 1):
        prompt_event = dict(event)
        prompt_event["minimum_primary_beat_occurrences"] = occurrence_requirements[
            event_id
        ]
        prompt_event["generation_action_unit_count"] = generation_unit_counts[event_id]
        prompt_events.append(prompt_event)
    events_json = _build_events_json(prompt_events)
    characters_summary = _build_characters_summary(characters)

    requested_mode = os.getenv("HONCUT_ADAPT_MODE", "layered").strip().lower()
    use_layered = requested_mode != "single" and len(events) > 10
    if use_layered:
        checkpoint_dir = Path(output_dir) if output_dir is not None else None
        layered_fingerprint = _layered_input_fingerprint(
            events,
            characters_summary,
            material_duration,
            effective_shot_duration,
            max_shots,
        )
        skeleton = None
        resumed_shots: List[Dict[str, Any]] = []
        if checkpoint_dir is not None:
            skeleton, resumed_shots = _load_layered_checkpoints(
                checkpoint_dir,
                events,
                max_shots,
                layered_fingerprint,
            )
        if skeleton is None:
            skeleton = _build_beat_skeleton(
                events, characters_summary, material_duration, effective_shot_duration,
                max_shots,
            )
            skeleton["_checkpoint"] = {
                "schema": LAYERED_CHECKPOINT_SCHEMA,
                "input_fingerprint": layered_fingerprint,
            }
            if checkpoint_dir is not None:
                _atomic_write_json(checkpoint_dir / "beat_skeleton.json", skeleton)
        shots = _expand_beats_to_shots(
            skeleton["beats"], characters_summary, material_duration, effective_shot_duration,
            output_dir=checkpoint_dir,
            resumed_shots=resumed_shots,
            checkpoint_fingerprint=layered_fingerprint,
        )
        _validate_shots(shots)
        shot_language_plan = _validate_shot_language_variation(shots)

        # Defensive assembly: batch responses may ignore their requested offset.
        for i, shot in enumerate(shots, 1):
            shot["shot_order"] = i

        _inherit_event_semantics(shots, events, characters)
        for shot in shots:
            apply_camera_motion_contract(shot)
        from quality.shot_continuity import annotate_boundaries

        annotate_boundaries(shots)
        normalize_shot_durations(shots, material_duration, capability_profile)

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
        if abs(total_duration - material_duration) > material_duration * 0.10:
            print(
                f"  ⚠ 分镜建议总时长 {total_duration}秒与素材目标 "
                f"{material_duration}秒偏差超过 10%",
                file=sys.stderr,
            )
        return {
            "target_duration": target_duration,
            "delivery_target_duration": target_duration,
            "material_duration": material_duration,
            "capacity_plan": capacity_plan,
            "estimated_shots": len(shots),
            "requested_shot_duration": shot_duration,
            "effective_shot_duration": effective_shot_duration,
            "total_duration": total_duration,
            "shot_language_plan": shot_language_plan,
            "strategy": skeleton.get("strategy", ""),
            "shots": shots,
        }

    # [LEGACY-KEEP layered-adapt] 原单次调用路径；显式 single 或事件数 <= 10 时使用。
    user_prompt = USER_PROMPT_TEMPLATE.format(
        target_duration=material_duration,
        shot_duration=effective_shot_duration,
        max_shots=max_shots,
        max_generation_action_units_per_shot=(
            capability_profile.max_micro_actions_per_beat
            * MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
        ),
        events_json=events_json,
        characters_summary=characters_summary,
    )

    # ── 调用 LLM（带重试）─────────────────────────────────────────────────
    parsed = None
    shot_language_plan: Optional[Dict[str, Any]] = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            attempt_prompt = user_prompt
            if attempt:
                attempt_prompt += (
                    "\n\n【重试纠错】必须输出恰好"
                    f"{max_shots}个镜头，并满足每个事件的 "
                    "minimum_primary_beat_occurrences；重复引用同一事件时只推进"
                    "尚未表现的后续动作，不得重放。"
                )
            response = _call_llm_with_timeout_retry(attempt_prompt)
            parsed = _parse_response(
                response, require_authored_shot_language=True
            )
            if len(parsed["shots"]) != max_shots:
                raise ValueError(
                    f"必须输出 {max_shots} 个镜头，实际为 {len(parsed['shots'])}"
                )
            parsed["shots"] = _repair_beat_action_capacity(
                parsed["shots"], events, capability_profile
            )
            _validate_beat_action_capacity(
                parsed["shots"],
                events,
                capability_profile,
            )
            _validate_beat_material_duration(
                parsed["shots"],
                events,
                material_duration,
                capability_profile,
            )
            shot_language_plan = _validate_shot_language_variation(parsed["shots"])
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

    _inherit_event_semantics(shots, events, characters)
    for shot in shots:
        apply_camera_motion_contract(shot)
    from quality.shot_continuity import annotate_boundaries

    annotate_boundaries(shots)
    normalize_shot_durations(shots, material_duration, capability_profile)

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
    total_duration = sum(shot.get("suggested_duration", 0) for shot in shots)

    result = {
        "target_duration": target_duration,
        "delivery_target_duration": target_duration,
        "material_duration": material_duration,
        "capacity_plan": capacity_plan,
        "estimated_shots": len(shots),
        "requested_shot_duration": shot_duration,
        "effective_shot_duration": effective_shot_duration,
        "total_duration": total_duration,
        "shot_language_plan": shot_language_plan,
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

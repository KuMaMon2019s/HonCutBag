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
1. 先按交付时长规划一级/二级分镜故事时钟，并单独记录动作容量压力
2. 调用 LLM 做改编决策：
   - keep：重要事件保留
   - merge：相似/连续事件合并
   - drop：重复/不重要事件删减
   - expand：关键情感时刻可扩展为多镜
3. 为每个 shot 分配不超过交付时长的故事时钟；桥接生成开销另记账
4. 建议转场方式（cut/dissolve/fade）
5. 输出排序后的 shot 列表
"""

import argparse
import copy
import functools
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import APITimeoutError, OpenAI
from pydantic import ValidationError

from phases.phase1.director_planner import (
    DIRECTOR_INTENT_FIELDS,
    DIRECTOR_PLAN_SCHEMA,
)
from runtime.llm_policy import LLMStreamPolicy
from runtime.provider_attempt_policy import effective_provider_retries
from schemas.understanding import (
    DurationScaledActionSelectionBatch,
    SourceIndexedScreenplayRewriteBatch,
    native_chat_json_schema_format,
    parse_structured_output,
)
from utils.action_units import (
    ACTION_TIMELINE_SCHEMA,
    SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA,
    build_action_timeline,
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
    resolve_character_id,
    resolve_character_name,
)
from utils.material_budget import MAX_CONTENT_PROVIDER_PADDING_LOSS_RATE
from utils.camera_angle_contracts import (
    CAMERA_ANGLE_PLANNING_INSTRUCTIONS,
    CAMERA_ANGLE_VALUES,
)

SHOT_POLICY_CONTINUITY = "continuity"
SHOT_POLICY_BALANCED = "balanced"
SHOT_POLICY_CUT_DRIVEN = "cut-driven"
SHOT_POLICIES = (
    SHOT_POLICY_CONTINUITY,
    SHOT_POLICY_BALANCED,
    SHOT_POLICY_CUT_DRIVEN,
)

SOURCE_INDEXED_REWRITE_RECONCILIATION_SCHEMA = (
    "honcut.source-indexed-screenplay-rewrite-reconciliation.v1"
)
SOURCE_INDEXED_REWRITE_RECONCILIATION_POLICY = (
    "honcut.source-indexed-screenplay-rewrite-reconciliation-policy.v1"
)


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SOURCE_INDEXED_REWRITE_RECONCILIATION_POLICY_SHA256 = (
    _canonical_json_sha256({
        "policy": SOURCE_INDEXED_REWRITE_RECONCILIATION_POLICY,
        "duplicate_scope": "adjacent_only",
        "authority": [
            "source_event_id",
            "production_action_index",
            "source_micro_action_indexes",
        ],
        "retention": "first_observation",
        "prose_merge": "forbidden",
    })
)
DEFAULT_SHOT_POLICY = SHOT_POLICY_CONTINUITY
PRIMARY_SHOT_LAYOUT_SCHEMA = "honcut.primary-shot-layout.v2"
MAX_CONTINUITY_CONTENT_BEATS_PER_PRIMARY_SHOT = 4
from utils.camera_motion_contracts import (
    CAMERA_MOTION_PLANNING_INSTRUCTIONS,
    CAMERA_MOVEMENT_VALUES,
    apply_camera_motion_contract,
)
from utils.body_action_contracts import apply_body_action_contract
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
_CAMERA_ANGLE_VALUES = CAMERA_ANGLE_VALUES
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
_VALID_CAMERA_ANGLES = frozenset(_CAMERA_ANGLE_VALUES)
_VALID_LIGHTING_KEYS = frozenset(_LIGHTING_KEY_VALUES)
_VALID_SHOT_INTENTS = frozenset(_SHOT_INTENT_VALUES)

_SHOT_LANGUAGE_ENUM_CONTRACT = (
    "  - shot_size: 字符串，景别（" + "/".join(_SHOT_SIZE_VALUES) + "）\n"
    "  - camera_angle: 字符串，机位角度（"
    + "/".join(_CAMERA_ANGLE_VALUES)
    + "）\n"
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
    "每个 Sxx 承载一个连续因果段，可包含多个有序动作单元，并遵守当前视频模型的 "
    "generation_action_units 上限。shot_duration 仅是软偏好；同一时空、主体和因果链应优先留在"
    "同一 Sxx，只有换场、跳时、主体切换或真实导演硬切才新增 Sxx。\n\n"
    "事件列表：\n{events_json}\n\n"
    "角色列表：\n{characters_summary}\n\n"
    "请输出一个 JSON 对象，包含：\n"
    "- strategy: 字符串，改编策略概述（一句话说明你的改编思路）\n"
    "- shots: JSON 数组，每个 shot 包含：\n"
    "  - shot_order: 整数，镜头序号（从 1 开始）\n"
    "  - source_events: 整数数组，本镜实际保留并生成的来源事件编号\n"
    "  - dropped_source_events: 整数数组，本轮因时长容量明确删减的来源事件编号\n"
    "  - action: 字符串，keep/merge/expand\n"
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
    + CAMERA_ANGLE_PLANNING_INSTRUCTIONS
    + "\n"
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
    "- 按当前 Sxx 时长，每镜最多承载 {max_generation_action_units_per_shot} 个"
    "generation_action_units；超出时必须拆到另一保留镜头，或把非关键重复事件显式列入"
    "dropped_source_events。\n"
    "- 一个导演级镜头可以容纳同一 sequence_id 中多个连续 action_unit；对 source_events 中明确保留"
    "的事件必须保留全部动作单元，"
    "后续由编剧引擎按单段内容承载能力与相邻一级分镜边界自动拆成 1–3 个内部 Pxx；拆分只能重排时长，"
    "必须逐项覆盖当前 Sxx 的 micro_actions，保持原顺序和 start_state→end_state 因果，不得新增剧情。\n"
    "- 不同 sequence_id、明确换场/跳时不得为了减少镜头而错误合并。\n"
    "- visual 必须按 micro_actions 的原始顺序描述，保留 start_state→动作→end_state 和 causal_link，"
    "禁止概括成‘双方激烈打斗’。\n"
    "- turning_point/dramatic_turn 必须在 what/visual 中明确保留；同一 sequence 内可与紧邻因果动作"
    "共用一个导演级镜头，但不得被概括或吞掉。\n"
    "- 时长不足时，可把非关键重复动作放入 dropped_source_events；这些事件不得同时出现在"
    "source_events，也不得进入 visual/micro_actions/生成提示词。scene_setup、turning_point、"
    "dramatic_turn、consequence 不得删减。\n"
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
    "shot_size、camera_angle、camera_movement、lighting_key、shot_intent、hero_moment、texture_keywords "
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


# Stage 2 only: its duration budget is local to one beat batch. Keep this
# independent from the compatibility-only USER_PROMPT_TEMPLATE; production
# adaptation has no single-call route.
BATCH_EXPAND_PROMPT = (
    "本批目标时长：{batch_target}秒，每镜约{shot_duration}秒，恰好输出{max_shots}个镜头。"
    "每个 Sxx 承载一个连续因果段，可包含多个有序动作单元；同一时空、主体和因果链优先保留在"
    "同一 Sxx，只有换场、跳时、主体切换或真实导演硬切才新增 Sxx。\n\n"
    "本批来源事件：\n{events_json}\n\n角色列表：\n{characters_summary}\n\n"
    "输出严格 JSON 对象，包含 strategy 和 shots。每个 shot 必须包含以下字段：\n"
    "beat_order（整数，必须等于该镜展开自哪个 beat 的 beat_order）、shot_order、"
    "source_events、action、reason、who、where、what、emotion、visual、"
    "suggested_duration、boundary_before、continuity_reason、continuity_subject、"
    "transition_to_next、associate_assets、shot_size、camera_angle、"
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
    "\"camera_angle\":\"eye_level\","
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
    "同一 sequence_id 的相邻 action_unit 可以归入同一个导演级镜头；本批只接收骨架明确保留的"
    "source_events，必须保留它们的全部 micro_actions，"
    "供后续编剧引擎按内容承载能力与相邻边界拆成 1–3 个 Pxx 顺序执行；剧情格必须逐项覆盖当前 Sxx 的原始动作，"
    "不得改序、遗漏、新增剧情或提前执行下一 Sxx；跨 sequence 必须拆开，turning_point 必须明确保留。"
    "每个保留事件在 shots 中的 source_events 引用次数不得少于输入中的 minimum_kept_primary_beat_occurrences；"
    "重复引用只分担该事件尚未表现的后续动作。\n"
    "【片段间过渡规则】相邻片段用动作桥梁、情绪接力、空间视线或台词黏合消灭跳跃感。\n"
    "【铁律优先级】台词零删改 > 出场人物完整 > 只描述动作状态 > 长台词拆镜。\n\n"
    "【HonCut 分镜铁律】who 只能逐字引用角色主名，别名改主名，群众只进 visual；"
    "每片段必须在 15–30 秒一级镜头及其 8–15/6–10 秒二级片段承载范围内；单镜台词>20字必须拆镜（按4字/秒）；同场景人物不得无故消失；"
    "人物外观不进提示词；声音只写环境音和音效，禁止配乐/BGM/背景音乐；"
    "群演只做背景动作；相邻镜头景别和角度必须错开。\n\n"
    "【镜头语言继承】本批每个 shot 的 shot_size、camera_angle、camera_movement、lighting_key、shot_intent、"
    "hero_moment、texture_keywords 必须逐字复制对应 beat 的全局镜头语言合同，不得重新规划或"
    "退回 wide/static/natural 默认组合。\n\n"
    + CAMERA_ANGLE_PLANNING_INSTRUCTIONS
    + "\n"
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

ADAPTATION_LLM_POLICY = LLMStreamPolicy.adaptation_structured_output(
    max_tokens=32000,
)
# Compatibility aliases for integrations that inspect the Phase 1 limits.
LLM_TIMEOUT = ADAPTATION_LLM_POLICY.wall_timeout_seconds
LLM_IDLE_TIMEOUT = ADAPTATION_LLM_POLICY.idle_timeout_seconds
LLM_TRANSPORT_READ_TIMEOUT = (
    ADAPTATION_LLM_POLICY.transport_read_timeout_seconds
)
MAX_RETRIES = 1  # 解析失败重试次数
NETWORK_RETRIES = 2  # 网络超时自动重试次数（2026-08-09 R7: 70事件大prompt一次超时即死太脆）
AVG_SHOT_DURATION = 12  # 默认每镜时长（秒）
CHARS_PER_SECOND = 4  # 中文剧本预估：约 4 字/秒（范围 3-5）
DEFAULT_TARGET_DURATION = 60  # 默认目标时长（用户未指定时使用）

# The historical 1.3x value is retained only as an advisory cost reference.
# Story-bearing Sxx/Pxx duration is planned against the delivery clock; bridge
# generation is additive and receives its own ledger after boundary planning.
GENERATED_DURATION_RATIO_REFERENCE = 1.3


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
    *,
    shot_policy: str = DEFAULT_SHOT_POLICY,
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
        shot_policy=shot_policy,
    )["primary_shots"]


def _minimum_primary_duration_for_units(
    generation_units: int,
    capabilities: VideoModelCapabilities,
    *,
    max_content_beats: int = MAX_CONTENT_BEATS_PER_PRIMARY_SHOT,
) -> int:
    """Return executable story time for one base clip plus its extensions."""
    content_beats = max(
        1,
        math.ceil(generation_units / capabilities.temporal_slice_limit),
    )
    if content_beats > max_content_beats:
        raise ValueError(
            f"{generation_units} generation action units exceed one primary shot's "
            f"{max_content_beats}-clip capacity"
        )
    first_minimum, _ = capabilities.effective_duration_bounds("multi_image")
    tail_minimum, _ = capabilities.effective_duration_bounds("tail_video_extend")
    return math.ceil(max(
        min_primary_story_duration(capabilities),
        first_minimum + max(0, content_beats - 1) * tail_minimum,
    ))


def _generation_unit_capacity_for_story_duration(
    story_duration: float,
    capabilities: VideoModelCapabilities,
    *,
    max_content_beats: int = MAX_CONTENT_BEATS_PER_PRIMARY_SHOT,
) -> int:
    """Return the action-unit capacity that fits one Sxx story allocation."""
    content_beats = 0
    for candidate in range(1, max_content_beats + 1):
        if _minimum_primary_duration_for_units(
            candidate * capabilities.temporal_slice_limit,
            capabilities,
            max_content_beats=max_content_beats,
        ) <= story_duration + 1e-6:
            content_beats = candidate
    return max(1, content_beats) * capabilities.temporal_slice_limit


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
        capabilities.temporal_slice_limit
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


def _non_mergeable_primary_segment_count(
    events: list[dict[str, Any]],
) -> int:
    """Count sequence changes and explicit director hard-cut boundaries."""
    segments = 0
    previous_sequence: str | None = None
    for event in events:
        if str(event.get("event_role") or "") == "drop":
            continue
        sequence = str(event.get("sequence_id") or "").strip() or "__unspecified__"
        boundary = str(
            event.get("continuity_before")
            or event.get("boundary_before")
            or ""
        ).strip().lower()
        if (
            segments == 0
            or sequence != previous_sequence
            or boundary in {"cut", "hard_cut", "hard-cut"}
        ):
            segments += 1
        previous_sequence = sequence
    return max(1, segments)


def _maximum_generation_units_for_story_clock(
    primary_shots: int,
    story_duration: int,
    capabilities: VideoModelCapabilities,
) -> int:
    """Return the largest normalized action ledger executable in the clock."""
    unit_capacity = (
        capabilities.temporal_slice_limit
        * MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    )
    for generation_units in range(primary_shots * unit_capacity, -1, -1):
        if (
            _sequence_material_cost(
                generation_units,
                primary_shots,
                capabilities,
            )
            <= story_duration + 1e-6
        ):
            return generation_units
    return 0


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
        capabilities.temporal_slice_limit
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


def _estimate_cut_driven_action_capacity_plan(
    events: list[dict[str, Any]],
    delivery_duration: int,
    requested_shot_duration: int,
) -> dict[str, Any]:
    """Plan the delivery story clock and report authored action pressure.

    The authored ledger may require more provider-executable time than the
    requested delivery.  That is an adaptation signal, not permission to grow
    Sxx/Pxx story time.  The screenwriter must explicitly merge or drop
    non-essential events while preserving mandatory turns; bridge generation
    is accounted after the primary storyboard exists.
    """
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
        profile.temporal_slice_limit * MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    )
    structural_shots = sum(
        max(1, math.ceil(units / unit_capacity))
        for _sequence, units in sequence_units
    )
    minimum_primary_duration = math.ceil(min_primary_story_duration(profile))
    if delivery_duration < minimum_primary_duration:
        raise ValueError(
            f"{delivery_duration}s delivery is below {profile.name}'s "
            f"{minimum_primary_duration}s minimum primary story duration"
        )
    maximum_story_shots = max(1, int(delivery_duration) // minimum_primary_duration)
    minimum_material_duration = _material_duration_bound(
        sequence_units,
        structural_shots,
        profile,
    )
    worst_case_material_duration = _material_duration_bound(
        sequence_units,
        structural_shots,
        profile,
        maximize=True,
    )
    action_capacity_fits = (
        structural_shots <= maximum_story_shots
        and minimum_material_duration <= delivery_duration + 1e-6
    )
    # The requested average shot duration is an editorial preference, not a
    # semantic-capacity ceiling.  Preserve every sequence slot when the full
    # ledger fits.  Under unavoidable compression, grow only until another
    # primary shot no longer increases executable action coverage; this avoids
    # both artificial event deletion and needless rapid cutting.
    if action_capacity_fits:
        primary_shots = min(
            max(baseline, structural_shots),
            maximum_story_shots,
        )
    else:
        search_end = min(
            maximum_story_shots,
            max(baseline, structural_shots + 1),
        )
        capacities = {
            shot_count: _maximum_generation_units_for_story_clock(
                shot_count,
                delivery_duration,
                profile,
            )
            for shot_count in range(baseline, search_end + 1)
        }
        best_capacity = max(capacities.values())
        first_capacity_maximizer = min(
            shot_count
            for shot_count, capacity in capacities.items()
            if capacity == best_capacity
        )
        # One additional slot on the same capacity plateau gives ordered,
        # indivisible event ranges room to share a boundary without forcing a
        # temporal jump.  It does not enlarge the story clock or theoretical
        # action budget, and is bounded by the structural search range.
        primary_shots = min(first_capacity_maximizer + 1, search_end)
    mandatory_event_ids = _mandatory_adaptation_event_ids(authored_events)
    mandatory_sequences = list(dict.fromkeys(
        str(event.get("sequence_id") or "").strip() or "__unspecified__"
        for event_id, event in enumerate(authored_events, 1)
        if event_id in mandatory_event_ids
    ))
    if len(mandatory_sequences) > maximum_story_shots:
        raise ValueError(
            f"mandatory source facts span {len(mandatory_sequences)} sequences; "
            f"the {delivery_duration}s story clock can carry at most "
            f"{maximum_story_shots} sequence-isolated beats"
        )
    primary_shots = max(primary_shots, len(mandatory_sequences))
    return {
        "primary_shots": primary_shots,
        "structural_shots": structural_shots,
        "maximum_story_shots": maximum_story_shots,
        "mandatory_sequence_count": len(mandatory_sequences),
        "generation_action_units": sum(units for _sequence, units in sequence_units),
        "sequence_generation_action_units": dict(sequence_units),
        "minimum_material_duration": math.ceil(minimum_material_duration),
        "worst_case_material_duration": math.ceil(worst_case_material_duration),
        "storyboard_duration_limit": int(delivery_duration),
        "material_duration": int(delivery_duration),
        "action_capacity_status": (
            "fits_story_clock"
            if action_capacity_fits
            else "screenplay_compression_required"
        ),
        "action_capacity_pressure_ratio": round(
            float(minimum_material_duration) / float(delivery_duration),
            6,
        ),
        "generated_duration_ratio_reference": GENERATED_DURATION_RATIO_REFERENCE,
        "generated_duration_ratio_reference_is_advisory": True,
        "delivery_duration": int(delivery_duration),
        "requested_shot_duration": int(requested_shot_duration),
        "capability_profile": profile.name,
    }


def _validate_shot_policy(shot_policy: str) -> str:
    normalized = str(shot_policy or "").strip().lower()
    if normalized not in SHOT_POLICIES:
        raise ValueError(
            f"shot_policy must be one of {', '.join(SHOT_POLICIES)}"
        )
    return normalized


def _balanced_bounded_allocations(
    total: int,
    lower_bounds: list[int],
    upper_bounds: list[int],
) -> list[int] | None:
    """Allocate an integer clock as evenly as provider bounds permit."""
    if len(lower_bounds) != len(upper_bounds) or not lower_bounds:
        return None
    if any(lower > upper for lower, upper in zip(lower_bounds, upper_bounds)):
        return None
    if total < sum(lower_bounds) or total > sum(upper_bounds):
        return None
    allocations = list(lower_bounds)
    remaining = int(total) - sum(allocations)
    while remaining:
        candidates = [
            index
            for index, value in enumerate(allocations)
            if value < upper_bounds[index]
        ]
        if not candidates:
            return None
        selected = min(candidates, key=lambda index: (allocations[index], index))
        allocations[selected] += 1
        remaining -= 1
    return allocations


def _balanced_content_beat_counts(
    primary_shots: int,
    total_content_beats: int,
) -> list[int]:
    base, remainder = divmod(total_content_beats, primary_shots)
    return [
        base + (1 if index < remainder else 0)
        for index in range(primary_shots)
    ]


def _build_primary_shot_layout_candidate(
    *,
    target_duration: int,
    primary_shots: int,
    total_content_beats: int,
    capabilities: VideoModelCapabilities,
    shot_policy: str,
    source_action_units: int,
    maximum_production_action_units: int,
    maximum_padding_loss_rate: float,
    maximum_content_beats_per_primary_shot: int = (
        MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    ),
) -> dict[str, Any] | None:
    beat_counts = _balanced_content_beat_counts(
        primary_shots,
        total_content_beats,
    )
    if any(
        count < 1 or count > maximum_content_beats_per_primary_shot
        for count in beat_counts
    ):
        return None

    base_minimum, base_maximum = capabilities.effective_duration_bounds(
        "multi_image"
    )
    tail_minimum, tail_maximum = capabilities.effective_duration_bounds(
        "tail_video_extend"
    )
    primary_minimum = math.ceil(min_primary_story_duration(capabilities))
    primary_maximum = math.floor(max_primary_story_duration(capabilities))
    primary_lower_bounds = [
        max(
            primary_minimum,
            math.ceil(base_minimum + (count - 1) * tail_minimum),
        )
        for count in beat_counts
    ]
    primary_upper_bounds = [
        min(
            primary_maximum,
            math.floor(base_maximum + (count - 1) * tail_maximum),
        )
        for count in beat_counts
    ]
    primary_allocations = _balanced_bounded_allocations(
        int(target_duration),
        primary_lower_bounds,
        primary_upper_bounds,
    )
    if primary_allocations is None:
        return None

    effective_beat_durations: list[list[int]] = []
    request_durations: list[list[float]] = []
    for primary_duration, content_beats in zip(
        primary_allocations,
        beat_counts,
    ):
        beat_lower_bounds = [math.ceil(base_minimum)] + [
            math.ceil(tail_minimum)
        ] * (content_beats - 1)
        beat_upper_bounds = [math.floor(base_maximum)] + [
            math.floor(tail_maximum)
        ] * (content_beats - 1)
        beat_allocations = _balanced_bounded_allocations(
            primary_duration,
            beat_lower_bounds,
            beat_upper_bounds,
        )
        if beat_allocations is None:
            return None
        strategies = ["multi_image"] + [
            "tail_video_extend"
        ] * (content_beats - 1)
        try:
            primary_requests = [
                capabilities.request_duration_for_effective_story(
                    duration,
                    strategy,
                )
                for duration, strategy in zip(beat_allocations, strategies)
            ]
        except ValueError:
            return None
        effective_beat_durations.append(beat_allocations)
        request_durations.append(primary_requests)

    total_request = sum(sum(items) for items in request_durations)
    padding = total_request - float(target_duration)
    loss_rate = padding / total_request if total_request else 0.0
    if (
        total_content_beats > 1
        and loss_rate > maximum_padding_loss_rate + 1e-6
    ):
        return None

    action_capacities = [
        count * capabilities.temporal_slice_limit
        for count in beat_counts
    ]
    total_action_capacity = sum(action_capacities)
    return {
        "schema": PRIMARY_SHOT_LAYOUT_SCHEMA,
        "shot_policy": shot_policy,
        "primary_shots": primary_shots,
        "story_duration_allocations_s": primary_allocations,
        "content_beat_counts": beat_counts,
        "effective_story_durations_s": effective_beat_durations,
        "provider_request_durations_s": request_durations,
        "generation_action_unit_capacities": action_capacities,
        "temporal_slice_capacities": action_capacities,
        "max_temporal_slices_per_content_beat": (
            capabilities.temporal_slice_limit
        ),
        "max_motion_contributions_per_slice": (
            capabilities.motion_contribution_limit
        ),
        "max_generation_action_units_per_primary_shot": max(action_capacities),
        "max_content_beats_per_primary_shot": int(
            maximum_content_beats_per_primary_shot
        ),
        "total_generation_action_unit_capacity": total_action_capacity,
        "production_action_unit_target": min(
            int(source_action_units),
            int(maximum_production_action_units),
            total_action_capacity,
        ),
        "cross_sxx_boundary_count": max(0, primary_shots - 1),
        "projected_content_provider_request_duration_s": round(
            total_request,
            6,
        ),
        "projected_content_provider_padding_duration_s": round(padding, 6),
        "projected_padding_loss_rate": round(loss_rate, 6),
        "maximum_padding_loss_rate": float(maximum_padding_loss_rate),
        "capability_profile": capabilities.name,
    }


def _solve_primary_shot_layout(
    *,
    target_duration: int,
    requested_shot_duration: int,
    capabilities: VideoModelCapabilities,
    shot_policy: str,
    source_action_units: int,
    minimum_primary_shots: int,
    maximum_padding_loss_rate: float = MAX_CONTENT_PROVIDER_PADDING_LOSS_RATE,
) -> dict[str, Any]:
    """Enumerate executable Sxx/Pxx layouts and apply one policy objective."""
    policy = _validate_shot_policy(shot_policy)
    if policy == SHOT_POLICY_CUT_DRIVEN:
        raise ValueError("cut-driven layouts must use the legacy planner")
    minimum_story = math.ceil(min_primary_story_duration(capabilities))
    maximum_story = math.floor(max_primary_story_duration(capabilities))
    minimum_shots = max(
        1,
        int(minimum_primary_shots),
        math.ceil(float(target_duration) / maximum_story),
    )
    maximum_shots = math.floor(float(target_duration) / minimum_story)
    candidates: list[dict[str, Any]] = []
    maximum_content_beats = (
        MAX_CONTINUITY_CONTENT_BEATS_PER_PRIMARY_SHOT
        if policy == SHOT_POLICY_CONTINUITY
        else MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    )
    # Continuity repacks the complete executable content ledger into longer
    # Sxx containers.  Balanced retains the established story-clock cap; its
    # requested shot count remains the editorial tie-breaker.
    maximum_production_action_units = (
        max(1, int(source_action_units))
        if policy == SHOT_POLICY_CONTINUITY
        else max(
            1,
            math.floor(float(target_duration) / capabilities.min_unique_beat_s),
        )
    )
    for primary_shots in range(minimum_shots, maximum_shots + 1):
        for total_content_beats in range(
            primary_shots,
            primary_shots * maximum_content_beats + 1,
        ):
            candidate = _build_primary_shot_layout_candidate(
                target_duration=target_duration,
                primary_shots=primary_shots,
                total_content_beats=total_content_beats,
                capabilities=capabilities,
                shot_policy=policy,
                source_action_units=source_action_units,
                maximum_production_action_units=maximum_production_action_units,
                maximum_padding_loss_rate=maximum_padding_loss_rate,
                maximum_content_beats_per_primary_shot=maximum_content_beats,
            )
            if candidate is not None:
                candidates.append(candidate)
    if not candidates:
        raise ValueError(
            "no primary-shot layout can satisfy story clock, sequence isolation, "
            "Provider capacity, and padding constraints"
        )

    preferred_shots = max(
        1,
        math.ceil(float(target_duration) / requested_shot_duration),
    )
    if policy == SHOT_POLICY_CONTINUITY:
        objective_order = [
            "maximum_production_action_unit_target",
            "minimum_primary_shots",
            "minimum_cross_sxx_boundary_count",
            "minimum_provider_padding",
        ]
        selected = min(
            candidates,
            key=lambda candidate: (
                -candidate["production_action_unit_target"],
                candidate["primary_shots"],
                candidate["cross_sxx_boundary_count"],
                candidate["projected_content_provider_padding_duration_s"],
                candidate["total_generation_action_unit_capacity"],
            ),
        )
    else:
        objective_order = [
            "maximum_production_action_unit_target",
            "nearest_requested_shot_count",
            "minimum_primary_shots",
            "minimum_provider_padding",
        ]
        selected = min(
            candidates,
            key=lambda candidate: (
                -candidate["production_action_unit_target"],
                abs(candidate["primary_shots"] - preferred_shots),
                candidate["primary_shots"],
                candidate["projected_content_provider_padding_duration_s"],
                candidate["total_generation_action_unit_capacity"],
            ),
        )
    selected = copy.deepcopy(selected)
    selected["objective_order"] = objective_order
    selected["objective_decision"] = {
        "candidate_count": len(candidates),
        "requested_shot_duration_s": int(requested_shot_duration),
        "preferred_primary_shot_count": preferred_shots,
        "selected_primary_shot_count": selected["primary_shots"],
        "selected_production_action_unit_target": selected[
            "production_action_unit_target"
        ],
        "story_clock_action_unit_limit": selected[
            "total_generation_action_unit_capacity"
        ],
        "source_action_unit_count": int(source_action_units),
    }
    return selected


def _legacy_primary_shot_layout(
    capacity_plan: dict[str, Any],
    *,
    target_duration: int,
    capabilities: VideoModelCapabilities,
) -> dict[str, Any]:
    primary_shots = int(capacity_plan["primary_shots"])
    base, remainder = divmod(int(target_duration), primary_shots)
    allocations = [
        base + (1 if index < remainder else 0)
        for index in range(primary_shots)
    ]
    capacities = [
        _generation_unit_capacity_for_story_duration(duration, capabilities)
        for duration in allocations
    ]
    content_beats = sum(
        math.ceil(capacity / capabilities.temporal_slice_limit)
        for capacity in capacities
    )
    source_action_units = int(
        capacity_plan.get("generation_action_units") or 0
    )
    layout = _build_primary_shot_layout_candidate(
        target_duration=target_duration,
        primary_shots=primary_shots,
        total_content_beats=content_beats,
        capabilities=capabilities,
        shot_policy=SHOT_POLICY_CUT_DRIVEN,
        source_action_units=source_action_units,
        maximum_production_action_units=source_action_units,
        # Legacy selection is intentionally preserved even when its audited
        # request padding is above the current Phase 5 safety threshold.
        maximum_padding_loss_rate=1.0,
    )
    if layout is None:
        raise ValueError("legacy cut-driven layout is not Provider-executable")
    layout["maximum_padding_loss_rate"] = (
        MAX_CONTENT_PROVIDER_PADDING_LOSS_RATE
    )
    layout["objective_order"] = ["legacy_cut_driven_algorithm"]
    layout["objective_decision"] = {
        "selected_primary_shot_count": primary_shots,
    }
    return layout


def _estimate_action_capacity_plan(
    events: list[dict[str, Any]],
    delivery_duration: int,
    requested_shot_duration: int,
    *,
    shot_policy: str = DEFAULT_SHOT_POLICY,
    max_material_padding_ratio: float = MAX_CONTENT_PROVIDER_PADDING_LOSS_RATE,
    delivery_overrun_ratio: float = 0.0,
) -> dict[str, Any]:
    policy = _validate_shot_policy(shot_policy)
    for field, raw_value in (
        ("max_material_padding_ratio", max_material_padding_ratio),
        ("delivery_overrun_ratio", delivery_overrun_ratio),
    ):
        if isinstance(raw_value, bool):
            raise ValueError(f"{field} must be numeric")
        value = float(raw_value)
        if not 0 <= value <= 0.25:
            raise ValueError(f"{field} must be between 0 and 0.25")
    max_material_padding_ratio = float(max_material_padding_ratio)
    delivery_overrun_ratio = float(delivery_overrun_ratio)
    profile = get_video_capabilities()
    nominal_duration = int(delivery_duration)
    ceiling_duration = math.floor(
        nominal_duration * (1.0 + delivery_overrun_ratio) + 1e-9
    )
    candidates: list[dict[str, Any]] = []
    first_error: ValueError | None = None
    for planned_duration in range(nominal_duration, ceiling_duration + 1):
        try:
            candidate = _estimate_cut_driven_action_capacity_plan(
                events,
                planned_duration,
                requested_shot_duration,
            )
            minimum_primary_segments = _non_mergeable_primary_segment_count(events)
            if policy == SHOT_POLICY_CUT_DRIVEN:
                candidate["shot_policy"] = policy
                candidate["minimum_primary_segment_count"] = (
                    minimum_primary_segments
                )
                layout = _legacy_primary_shot_layout(
                    candidate,
                    target_duration=planned_duration,
                    capabilities=profile,
                )
            else:
                sequence_count = len(
                    _sequence_generation_unit_counts([
                        event
                        for event in events
                        if str(event.get("event_role") or "") != "drop"
                    ])
                )
                layout = _solve_primary_shot_layout(
                    target_duration=planned_duration,
                    requested_shot_duration=requested_shot_duration,
                    capabilities=profile,
                    shot_policy=policy,
                    source_action_units=int(candidate["generation_action_units"]),
                    minimum_primary_shots=max(
                        1,
                        sequence_count,
                        minimum_primary_segments,
                        int(candidate["mandatory_sequence_count"]),
                    ),
                    maximum_padding_loss_rate=max_material_padding_ratio,
                )
                candidate["shot_policy"] = policy
                candidate["primary_shots"] = layout["primary_shots"]
                candidate["minimum_primary_segment_count"] = (
                    minimum_primary_segments
                )
            candidate["primary_shot_layout"] = layout
            layout["nominal_delivery_duration_s"] = nominal_duration
            layout["delivery_ceiling_duration_s"] = ceiling_duration
            layout["planned_delivery_duration_s"] = planned_duration
            layout["delivery_overrun_ratio"] = delivery_overrun_ratio
            layout["maximum_padding_loss_rate"] = (
                max_material_padding_ratio
            )
            candidate["action_capacity_status"] = (
                "fits_story_clock"
                if layout["production_action_unit_target"]
                >= candidate["generation_action_units"]
                else "screenplay_compression_required"
            )
            candidate["nominal_delivery_duration"] = nominal_duration
            candidate["delivery_ceiling_duration"] = ceiling_duration
            candidate["planned_delivery_duration"] = planned_duration
            candidate["delivery_overrun_ratio"] = delivery_overrun_ratio
            candidate["max_material_padding_ratio"] = (
                max_material_padding_ratio
            )
            candidates.append(candidate)
        except ValueError as exc:
            if first_error is None:
                first_error = exc
    if not candidates:
        if first_error is not None:
            raise first_error
        raise ValueError("no delivery duration has an executable shot layout")
    source_units = max(
        int(candidate["generation_action_units"]) for candidate in candidates
    )
    selected = min(
        candidates,
        key=lambda candidate: (
            -min(
                source_units,
                int(candidate["primary_shot_layout"][
                    "production_action_unit_target"
                ]),
            ),
            int(candidate["planned_delivery_duration"]) - nominal_duration,
            int(candidate["primary_shot_layout"]["primary_shots"]),
            float(candidate["primary_shot_layout"].get(
                "projected_content_provider_padding_duration_s"
            ) or 0.0),
        ),
    )
    selected["material_duration"] = int(selected["planned_delivery_duration"])
    selected["storyboard_duration_limit"] = int(
        selected["planned_delivery_duration"]
    )
    selected["delivery_duration"] = nominal_duration
    return selected


def _resolve_padding_rewrite_layout(
    capacity_plan: Dict[str, Any],
    *,
    target_duration: int,
    capabilities: VideoModelCapabilities,
    rewrite_request: Dict[str, Any],
    shot_policy: str = DEFAULT_SHOT_POLICY,
) -> Dict[str, Any]:
    """Choose the least-loss single-request layout for one Phase 5 rewrite."""

    from schemas.replanning import (
        PADDING_LOSS_ERROR_CODE,
        SCREENPLAY_REWRITE_REQUEST_SCHEMA,
    )

    if rewrite_request.get("schema") != SCREENPLAY_REWRITE_REQUEST_SCHEMA:
        raise ValueError("unsupported screenplay rewrite request schema")
    if rewrite_request.get("reason_code") != PADDING_LOSS_ERROR_CODE:
        raise ValueError("unsupported screenplay rewrite reason")
    if rewrite_request.get("attempt") != 1:
        raise ValueError("screenplay padding rewrite attempt must equal 1")
    raw_limit = rewrite_request.get("maximum_padding_loss_rate")
    if isinstance(raw_limit, bool):
        raise ValueError("screenplay padding rewrite limit must be numeric")
    try:
        maximum_padding_loss_rate = float(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "screenplay padding rewrite limit must be numeric"
        ) from exc
    if not 0 < maximum_padding_loss_rate < 1:
        raise ValueError("screenplay padding rewrite limit must be between 0 and 1")

    policy = _validate_shot_policy(shot_policy)
    if policy != SHOT_POLICY_CUT_DRIVEN:
        layout = _solve_primary_shot_layout(
            target_duration=target_duration,
            requested_shot_duration=max(
                1,
                int(
                    capacity_plan.get("requested_shot_duration")
                    or round(
                        float(target_duration)
                        / max(1, int(capacity_plan.get("primary_shots") or 1))
                    )
                ),
            ),
            capabilities=capabilities,
            shot_policy=policy,
            source_action_units=int(
                capacity_plan.get("generation_action_units") or 0
            ),
            minimum_primary_shots=max(
                1,
                int(capacity_plan.get("mandatory_sequence_count") or 1),
                int(capacity_plan.get("minimum_primary_segment_count") or 1),
                len(capacity_plan.get("sequence_generation_action_units") or {}),
            ),
            maximum_padding_loss_rate=maximum_padding_loss_rate,
        )
        layout["reason_code"] = PADDING_LOSS_ERROR_CODE
        layout["rewrite_attempt"] = 1
        layout["objective_decision"]["rewrite_replanned_with_shared_solver"] = True
        return layout

    current_shots = int(capacity_plan.get("primary_shots") or 0)
    mandatory_sequences = int(
        capacity_plan.get("mandatory_sequence_count") or 1
    )
    if current_shots < 2:
        raise ValueError("screenplay padding rewrite has no smaller shot layout")
    effective_minimum, effective_maximum = capabilities.effective_duration_bounds(
        "multi_image"
    )
    minimum_shots = max(
        1,
        mandatory_sequences,
        math.ceil(float(target_duration) / effective_maximum),
    )
    maximum_shots = min(
        current_shots - 1,
        math.floor(float(target_duration) / effective_minimum),
    )

    for primary_shots in range(maximum_shots, minimum_shots - 1, -1):
        base, remainder = divmod(int(target_duration), primary_shots)
        allocations = [
            base + (1 if index < remainder else 0)
            for index in range(primary_shots)
        ]
        try:
            request_durations = [
                capabilities.request_duration_for_effective_story(
                    duration,
                    "multi_image",
                )
                for duration in allocations
            ]
        except ValueError:
            continue
        total_request = sum(request_durations)
        padding = total_request - float(target_duration)
        loss_rate = padding / total_request if total_request else 0.0
        if loss_rate <= maximum_padding_loss_rate + 1e-6:
            action_capacities = [
                capabilities.temporal_slice_limit
            ] * primary_shots
            return {
                "schema": PRIMARY_SHOT_LAYOUT_SCHEMA,
                "shot_policy": policy,
                "reason_code": PADDING_LOSS_ERROR_CODE,
                "rewrite_attempt": 1,
                "maximum_padding_loss_rate": maximum_padding_loss_rate,
                "primary_shots": primary_shots,
                "story_duration_allocations_s": allocations,
                "content_beat_counts": [1] * primary_shots,
                "max_content_beats_per_primary_shot": (
                    MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
                ),
                "effective_story_durations_s": [
                    [duration] for duration in allocations
                ],
                "provider_request_durations_s": [
                    [duration] for duration in request_durations
                ],
                "generation_action_unit_capacities": action_capacities,
                "total_generation_action_unit_capacity": sum(action_capacities),
                "production_action_unit_target": min(
                    int(capacity_plan.get("generation_action_units") or 0),
                    sum(action_capacities),
                ),
                "cross_sxx_boundary_count": max(0, primary_shots - 1),
                "effective_shot_duration_s": round(
                    float(target_duration) / primary_shots
                ),
                "max_generation_action_units_per_primary_shot": (
                    capabilities.temporal_slice_limit
                ),
                "projected_content_provider_request_duration_s": round(
                    total_request, 6
                ),
                "projected_content_provider_padding_duration_s": round(
                    padding, 6
                ),
                "projected_padding_loss_rate": round(loss_rate, 6),
                "capability_profile": capabilities.name,
                "objective_order": ["legacy_cut_driven_padding_rewrite"],
                "objective_decision": {
                    "selected_primary_shot_count": primary_shots,
                },
            }
    raise ValueError(
        "no screenplay layout can satisfy the Phase 5 Provider padding limit "
        "while preserving mandatory sequence isolation"
    )


def _event_primary_occurrence_requirement(
    event: Dict[str, Any],
    capabilities: VideoModelCapabilities,
    seen: Optional[set] = None,
    *,
    max_content_beats_per_primary_shot: int = (
        MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    ),
) -> int:
    """Return how many primary shots an event needs under the one-extension rule."""
    content_beats = _event_content_beat_requirement(event, capabilities, seen=seen)
    return max(
        1,
        math.ceil(content_beats / max_content_beats_per_primary_shot),
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
    units = normalize_event_action_units(
        event,
        actions=actions,
        seen=seen,
        max_motion_contributions_per_slice=(
            capabilities.motion_contribution_limit
        ),
    )["units"]
    if units == 0:
        return 0
    return max(
        1,
        math.ceil(units / capabilities.temporal_slice_limit),
    )


def _event_generation_action_unit_counts(
    events: List[Dict[str, Any]],
    capabilities: VideoModelCapabilities | None = None,
) -> Dict[int, int]:
    """Count Provider execution units with cross-event source dedupe."""

    profile = capabilities or get_video_capabilities()
    seen: set = set()
    return {
        event_id: normalize_event_action_units(
            event,
            seen=seen,
            max_motion_contributions_per_slice=(
                profile.motion_contribution_limit
            ),
        )["units"]
        for event_id, event in enumerate(events, 1)
    }


def _event_content_beat_requirements(
    events: List[Dict[str, Any]],
    capabilities: VideoModelCapabilities,
) -> Dict[int, int]:
    """Normalize an ordered event ledger once, including cross-event dedupe."""

    unit_counts = _event_generation_action_unit_counts(
        events,
        capabilities=capabilities,
    )
    return {
        event_id: (
            math.ceil(units / capabilities.temporal_slice_limit)
            if units else 0
        )
        for event_id, units in unit_counts.items()
    }


def _event_primary_occurrence_requirements(
    events: List[Dict[str, Any]],
    capabilities: VideoModelCapabilities,
    *,
    max_content_beats_per_primary_shot: int = (
        MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    ),
) -> Dict[int, int]:
    content = _event_content_beat_requirements(events, capabilities)
    return {
        # Even a zero-cost state/camera event still needs one explicit primary
        # occurrence in the source ledger; zero only means it consumes no
        # generation action capacity.
        event_id: max(
            1,
            math.ceil(beats / max_content_beats_per_primary_shot),
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
    *,
    max_content_beats_per_primary_shot: int = (
        MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    ),
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
            generation_unit_count / profile.temporal_slice_limit
        ) if generation_unit_count else 1
        spoken_beats = math.ceil(spoken / profile.max_unique_beat_s) if spoken else 1
        content_beats = max(1, action_beats, spoken_beats)
        if content_beats > max_content_beats_per_primary_shot:
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

def _get_client(policy: LLMStreamPolicy = ADAPTATION_LLM_POLICY) -> OpenAI:
    """
    创建 OpenAI 客户端

    API Key: ARK_AGENT_API_KEY (火山方舟 Agent Plan)
    """
    return create_ark_client(
        read_timeout=policy.transport_read_timeout_seconds,
    )


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
    policy = LLMStreamPolicy.adaptation_structured_output(
        max_tokens=max_tokens,
    )
    client = _get_client(policy)

    # 2026-08-09 R8 教训：70事件→15镜大JSON非流式调用，turbo生成完整响应
    # 远超 240s timeout（R7/R8 共 6 次超时实锤）。流式响应仍可能在模型推理时
    # 长时间没有 content chunk，因此由 Runtime profile 分离 idle、wall 与 SDK read。
    # 2026-08-09 R9 教训：不设 max_tokens 时用默认输出上限，15镜×18字段 JSON
    # 在 char 8354/9433 被截断（JSONDecodeError: Unterminated string）。
    # 探针实锤 max_tokens=16000/32000 均被 Agent Plan 端点接受（HTTP 200），
    # 取 32000 留足 reasoning + 15 镜 JSON 余量。
    return call_llm_stream(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=policy.max_tokens,
        wall_timeout=policy.wall_timeout_seconds,
        read_timeout=policy.transport_read_timeout_seconds,
        idle_timeout=policy.idle_timeout_seconds,
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
    network_retry_limit = effective_provider_retries(NETWORK_RETRIES)
    for net_attempt in range(network_retry_limit + 1):
        try:
            return _call_llm(user_prompt, max_tokens=max_tokens)
        except (
            APITimeoutError,
            LLMConnectTimeout,
            LLMReadTimeout,
            LLMIdleTimeout,
            LLMStreamError,
        ) as e:
            if net_attempt < network_retry_limit:
                wait = 15 * (net_attempt + 1)
                print(f"  ⚠ LLM 网络超时，{wait}s 后重试 ({net_attempt + 1}/{network_retry_limit})...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"LLM 调用失败: 连续 {network_retry_limit} 次网络超时: {e}"
                ) from e
    raise RuntimeError("LLM 调用失败: 意外退出重试循环")  # 不可达，满足类型检查


_SHOT_LANGUAGE_FIELDS = (
    "shot_size",
    "camera_angle",
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
        if shot["camera_angle"] not in _VALID_CAMERA_ANGLES:
            raise ValueError(
                f"第 {index} 个 {label} camera_angle 无效: "
                f"{shot['camera_angle']}; 合法值: "
                f"{', '.join(_CAMERA_ANGLE_VALUES)}"
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
        dropped_source_events = shot.get("dropped_source_events", [])
        if not isinstance(dropped_source_events, list):
            dropped_source_events = []
        shot["dropped_source_events"] = list(dict.fromkeys(
            event_id
            for event_id in dropped_source_events
            if isinstance(event_id, int)
        ))
        ss = shot.get("shot_size", "")
        if ss not in _VALID_SHOT_SIZES:
            shot["shot_size"] = "wide"
        ca = shot.get("camera_angle", "")
        if ca not in _VALID_CAMERA_ANGLES:
            shot["camera_angle"] = "eye_level"
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


_EVENT_LLM_FIELDS = (
    "who",
    "background_groups",
    "where",
    "what",
    "emotion",
    "visual",
    "time",
    "action_type",
    "event_role",
    "micro_actions",
    "body_action_choreography",
    "generation_motion_mode",
    "action_phase",
    "start_state",
    "end_state",
    "causal_link",
    "continuity_before",
    "continuity_subject",
    "dramatic_turn",
    "lines",
    "sequence_id",
    "action_unit_id",
    "minimum_kept_primary_beat_occurrences",
    "generation_action_unit_count",
)


def _event_llm_view(event: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only authoritative event fields needed by the Phase 1 adapter LLM.

    Derived contracts such as ``body_action_contract`` duplicate choreography
    and add prompt/errors/forbidden payloads. Source excerpts and normalized
    generation units remain in the audit event but are reconstructed or read
    from that source after adaptation, so they are intentionally omitted here.
    """
    return {
        field: copy.deepcopy(event[field])
        for field in _EVENT_LLM_FIELDS
        if field in event
    }


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
        event_copy = _event_llm_view(event)
        event_copy["event_id"] = i
        numbered_events.append(event_copy)

    return json.dumps(numbered_events, ensure_ascii=False, indent=2)


def _build_event_details_json(events: List[Dict[str, Any]]) -> str:
    """Render batched event details through the same compact provider view.

    Stage 2 receives the globally assigned event ids from Stage 1, so unlike
    ``_build_events_json`` this helper preserves those ids rather than
    renumbering the current batch from one.
    """
    compact_events = []
    for event in events:
        event_copy = _event_llm_view(event)
        if event.get("event_id") is not None:
            event_copy["event_id"] = event["event_id"]
        compact_events.append(event_copy)
    return json.dumps(compact_events, ensure_ascii=False, indent=2)


def _director_intents_by_sequence(
    director_plan: Optional[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """Validate and reduce the Director artifact to adaptation-owned inputs."""
    if director_plan is None:
        return {}
    if not isinstance(director_plan, dict):
        raise ValueError("director_plan must be an object")
    if director_plan.get("schema") != DIRECTOR_PLAN_SCHEMA:
        raise ValueError(
            f"director_plan schema must be {DIRECTOR_PLAN_SCHEMA}"
        )
    sequences = director_plan.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("director_plan sequences must be an array")

    expected_ids = list(
        dict.fromkeys(
            str(event.get("sequence_id") or "").strip()
            for event in events
        )
    )
    if not expected_ids or any(not value for value in expected_ids):
        raise ValueError("adaptation events must all have sequence_id")

    intents: Dict[str, Dict[str, str]] = {}
    actual_ids: List[str] = []
    for index, sequence in enumerate(sequences, 1):
        if not isinstance(sequence, dict):
            raise ValueError(f"director_plan sequence {index} must be an object")
        sequence_id = str(sequence.get("sequence_id") or "").strip()
        if not sequence_id or sequence_id in intents:
            raise ValueError(
                f"director_plan has invalid or duplicate sequence_id: {sequence_id}"
            )
        intent: Dict[str, str] = {"sequence_id": sequence_id}
        for field in DIRECTOR_INTENT_FIELDS:
            value = sequence.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"director_plan sequence {sequence_id} has empty {field}"
                )
            intent[field] = value.strip()
        intents[sequence_id] = intent
        actual_ids.append(sequence_id)
    if actual_ids != expected_ids:
        raise ValueError(
            "director_plan sequence coverage/order does not match events; "
            f"expected={expected_ids}, actual={actual_ids}"
        )
    return intents


PRODUCTION_DIRECTOR_INTENT_SCHEMA = "honcut.production-director-intent.v1"


def _duration_scaled_visible_actions(
    event: Dict[str, Any],
) -> List[str] | None:
    """Return selected actions when intra-event omissions narrow visible facts."""
    selection = event.get("production_action_selection")
    if not isinstance(selection, dict):
        return None
    selected = selection.get("selected_source_micro_action_indexes")
    omitted = selection.get("omitted_source_micro_action_indexes")
    if not isinstance(selected, list) or not isinstance(omitted, list):
        raise ValueError("production action selection has invalid index arrays")
    if not omitted:
        return None
    raw_actions = event.get("micro_actions") or []
    if isinstance(raw_actions, str):
        raw_actions = [raw_actions]
    actions = [
        str(action).strip() for action in raw_actions if str(action).strip()
    ]
    if len(actions) != len(selected):
        raise ValueError(
            "duration-scaled event actions do not match selected source indexes"
        )
    return actions


def _source_event_narrative_fact(
    event: Dict[str, Any],
    event_id: int,
) -> str:
    selected_actions = _duration_scaled_visible_actions(event)
    if selected_actions is not None:
        if selected_actions:
            return " → ".join(selected_actions)
        states = [
            str(event.get(field) or "").strip()
            for field in ("start_state", "end_state")
            if str(event.get(field) or "").strip()
        ]
        return (
            " → ".join(dict.fromkeys(states))
            if states
            else f"canonical source event {event_id} retained state"
        )
    for field in ("what", "summary", "source_excerpt"):
        value = str(event.get(field) or "").strip()
        if value:
            return value
    raw_actions = event.get("micro_actions") or []
    if isinstance(raw_actions, str):
        raw_actions = [raw_actions]
    actions = [
        str(action).strip()
        for action in raw_actions
        if str(action).strip()
    ]
    if actions:
        return " → ".join(actions)
    states = [
        str(event.get(field) or "").strip()
        for field in ("start_state", "end_state")
        if str(event.get(field) or "").strip()
    ]
    if states:
        return " → ".join(dict.fromkeys(states))
    return f"canonical source event {event_id}"


def _source_event_visual_fact(
    event: Dict[str, Any],
    event_id: int,
) -> str:
    """Keep visual prose only when duration scaling did not omit its actions."""
    if _duration_scaled_visible_actions(event) is not None:
        return _source_event_narrative_fact(event, event_id)
    return (
        str(event.get("visual") or "").strip()
        or _source_event_narrative_fact(event, event_id)
    )


def _ground_production_beat_text_fields(
    beat: Dict[str, Any],
    selected_events: List[Dict[str, Any]],
) -> None:
    """Replace plot-bearing model prose with selected source-event facts."""
    source_event_ids = beat.get("source_events")
    if (
        not isinstance(source_event_ids, list)
        or not source_event_ids
        or len(source_event_ids) != len(selected_events)
    ):
        raise ValueError(
            "production beat grounding requires aligned source event details"
        )

    who = list(dict.fromkeys(
        str(name).strip()
        for event in selected_events
        for name in (event.get("who") or [])
        if str(name).strip()
    ))
    locations = list(dict.fromkeys(
        str(event.get("where") or "").strip()
        for event in selected_events
        if str(event.get("where") or "").strip()
    ))
    narrative_facts = list(dict.fromkeys(
        _source_event_narrative_fact(event, event_id)
        for event_id, event in zip(
            source_event_ids,
            selected_events,
            strict=True,
        )
    ))
    visual_facts = list(dict.fromkeys(
        _source_event_visual_fact(event, event_id)
        for event_id, event in zip(
            source_event_ids,
            selected_events,
            strict=True,
        )
    ))
    texture_candidates = list(dict.fromkeys(
        str(value).strip()
        for event_id, event in zip(
            source_event_ids,
            selected_events,
            strict=True,
        )
        for value in (
            _source_event_visual_fact(event, event_id),
            event.get("where"),
            _source_event_narrative_fact(event, event_id),
        )
        if str(value or "").strip()
    ))
    while len(texture_candidates) < 2:
        texture_candidates.append(
            f"canonical source event {source_event_ids[len(texture_candidates) % len(source_event_ids)]}"
        )

    beat["who"] = who
    beat["where"] = (
        locations[0]
        if len(locations) == 1
        else " / ".join(locations)
        if locations
        else "source-event location unspecified"
    )
    beat["what"] = "；随后".join(narrative_facts)
    beat["visual"] = "；随后".join(visual_facts)
    beat["reason"] = (
        "source-grounded production mapping for events "
        + ",".join(str(event_id) for event_id in source_event_ids)
    )
    beat["texture_keywords"] = texture_candidates[:4]


def _build_production_director_intent(
    source_intent: Dict[str, str],
    selected_events: List[Dict[str, Any]],
    *,
    source_event_ids: List[int],
    shot: Dict[str, Any],
) -> Dict[str, Any]:
    """Project sequence direction onto one source-grounded production shot.

    A sequence-level Director plan is authored before duration compression and
    may therefore describe facts that the production ledger later omits.  Only
    its identity/hash cross the production boundary; every plot-bearing field
    below is rebuilt from the shot's selected canonical events.
    """
    if not selected_events or len(selected_events) != len(source_event_ids):
        raise ValueError(
            "production director intent requires aligned selected source events"
        )
    if any(not isinstance(event_id, int) or event_id < 1 for event_id in source_event_ids):
        raise ValueError("production director intent has invalid source event ids")
    sequence_id = str(source_intent.get("sequence_id") or "").strip()
    event_sequences = {
        str(event.get("sequence_id") or "").strip()
        for event in selected_events
    }
    if not sequence_id or event_sequences != {sequence_id}:
        raise ValueError(
            "production director intent source sequence does not match selected events"
        )

    def joined(field: str, separator: str = "；") -> str:
        values = [
            str(event.get(field) or "").strip()
            for event in selected_events
            if str(event.get(field) or "").strip()
        ]
        return separator.join(dict.fromkeys(values))

    scene_goal = "；".join(dict.fromkeys(
        _source_event_narrative_fact(event, event_id)
        for event_id, event in zip(
            source_event_ids,
            selected_events,
            strict=True,
        )
    ))
    emotion_arc = joined("emotion", " → ") or "preserve source-event emotional state"
    visual_focus = "；".join(dict.fromkeys(
        _source_event_visual_fact(event, event_id)
        for event_id, event in zip(
            source_event_ids,
            selected_events,
            strict=True,
        )
    )) or scene_goal
    locations = joined("where", " → ") or "preserve source-event location"
    start_state = str(selected_events[0].get("start_state") or "").strip()
    end_state = str(selected_events[-1].get("end_state") or "").strip()
    spatial_intent = locations
    if start_state or end_state:
        spatial_intent += (
            f"；state progression: {start_state or 'source start'}"
            f" → {end_state or 'source end'}"
        )
    boundary = (
        str(selected_events[0].get("continuity_before") or "cut").strip().lower()
        or "cut"
    )
    if boundary not in {"cut", "continuous"}:
        raise ValueError(
            f"production director intent has invalid continuity boundary: {boundary}"
        )
    movement = str(shot.get("camera_movement") or "static").strip() or "static"
    transition_intent = (
        f"{boundary} entry into selected source events; preserve source order "
        f"with camera movement {movement}"
    )
    source_intent_sha256 = hashlib.sha256(
        json.dumps(
            source_intent,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": PRODUCTION_DIRECTOR_INTENT_SCHEMA,
        "source_director_plan_schema": DIRECTOR_PLAN_SCHEMA,
        "source_director_intent_sha256": source_intent_sha256,
        "sequence_id": sequence_id,
        "source_event_ids": list(source_event_ids),
        "scene_goal": scene_goal,
        "emotion_arc": emotion_arc,
        "visual_focus": visual_focus,
        "spatial_intent": spatial_intent,
        "transition_intent": transition_intent,
    }


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
    """Canonicalize ``who`` while retaining original source mentions for audit.

    New Phase 1 runs bind every source mention to instance IDs before
    Adaptation.  Those IDs are the authority: a group display mention can map
    to several instances and therefore cannot be reconstructed from an
    instance-level compatibility projection by matching display names again.
    Name resolution remains only as a legacy path for inputs that have not yet
    been bound by the semantic ledger.
    """
    if not characters:
        for shot in shots:
            shot.pop("_semantic_character_binding_verified", None)
        return shots
    characters_by_id: Dict[str, Dict[str, Any]] = {}
    for character in characters:
        character_id = str(character.get("id") or "").strip()
        if not character_id or character_id in characters_by_id:
            raise ValueError(
                "canonical characters require unique non-empty instance ids"
            )
        characters_by_id[character_id] = character

    for shot in shots:
        raw_who = shot.get("who") or []
        if isinstance(raw_who, str):
            raw_who = [raw_who]
        original = [str(name).strip() for name in raw_who if str(name).strip()]

        semantic_binding_verified = bool(
            shot.pop("_semantic_character_binding_verified", False)
        )
        prebound = shot.get("character_ids")
        if semantic_binding_verified:
            if not isinstance(prebound, list):
                raise ValueError("shot canonical character_ids must be an array")
            character_ids = list(dict.fromkeys(
                str(value).strip() for value in prebound if str(value).strip()
            ))
            unknown_ids = [
                character_id
                for character_id in character_ids
                if character_id not in characters_by_id
            ]
            if unknown_ids:
                raise ValueError(
                    "shot contains an unknown canonical character id: "
                    f"{unknown_ids}"
                )
            canonical_character_names = [
                str(characters_by_id[character_id].get("name") or "").strip()
                for character_id in character_ids
            ]
            if any(not name for name in canonical_character_names):
                raise ValueError(
                    "canonical character instance is missing its display name"
                )

            source_mentions: List[str] = []
            for reference in (shot.get("participant_refs") or []):
                if not isinstance(reference, dict):
                    continue
                instance_id = str(
                    reference.get("instance_id")
                    or reference.get("character_id")
                    or ""
                ).strip()
                if instance_id not in characters_by_id:
                    raise ValueError(
                        "shot participant reference contains an unknown "
                        f"canonical character id: {instance_id or '<missing>'}"
                    )
                if instance_id not in character_ids:
                    raise ValueError(
                        "shot participant reference is outside the canonical cast: "
                        f"{instance_id}"
                    )
                mention = str(reference.get("mention") or "").strip()
                if mention and mention not in source_mentions:
                    source_mentions.append(mention)

            non_character_participants = list(dict.fromkeys(
                str(value).strip()
                for value in (shot.get("non_character_participants") or [])
                if str(value).strip()
            ))
            canonical = list(dict.fromkeys([
                *canonical_character_names,
                *non_character_participants,
            ]))
            if canonical != original:
                if source_mentions:
                    shot["source_character_mentions"] = source_mentions
            shot["who"] = canonical
            shot["character_ids"] = character_ids
            continue

        canonical = list(
            dict.fromkeys(
                _canonical_character_name(name, characters) for name in original
            )
        )
        canonical = [name for name in canonical if name]
        if canonical != original:
            shot["source_character_mentions"] = original
        shot["who"] = canonical
        character_ids = [
            resolve_character_id(name, characters) for name in canonical
        ]
        if any(character_id is None for character_id in character_ids):
            raise ValueError(
                "shot contains a participant without a canonical character id: "
                f"{canonical}"
            )
        shot["character_ids"] = list(dict.fromkeys(character_ids))
    return shots


def _inherit_event_semantics(
    shots: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    characters: Optional[List[Dict[str, Any]]] = None,
    director_plan: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Carry source screenplay evidence into shots after the LLM adaptation pass.

    The model still chooses framing and whether a camera cut is useful, while
    source ordering, exact excerpts, action units, and speaker evidence remain
    deterministic and auditable downstream.
    """
    event_by_id = {index: event for index, event in enumerate(events, 1)}
    director_intents = _director_intents_by_sequence(director_plan, events)
    event_occurrences: Dict[int, List[int]] = {}
    for shot_index, shot in enumerate(shots):
        raw_ids = shot.get("source_events", [])
        source_ids = raw_ids if isinstance(raw_ids, list) else []
        for event_id in dict.fromkeys(source_ids):
            if event_id in event_by_id:
                event_occurrences.setdefault(event_id, []).append(shot_index)

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
            actions=[
                str(action).strip()
                for action in event_actions
                if str(action).strip()
            ],
            seen=preceding_event_keys,
        )

    event_slices: Dict[tuple[int, int], Dict[str, Any]] = {}
    for event_id, occurrence_shots in event_occurrences.items():
        event = event_by_id[event_id]
        actions = [
            str(action).strip()
            for action in (event.get("micro_actions") or [])
            if str(action).strip()
        ]
        normalized_event = normalize_event_action_units(
            event,
            actions=actions,
            seen=set(seen_before_event[event_id]),
            max_motion_contributions_per_slice=(
                get_video_capabilities().motion_contribution_limit
            ),
        )
        generation_units = normalized_event["generation_action_units"]
        declared_counts = []
        has_declared_counts = False
        for shot_index in occurrence_shots:
            raw_ledger = shots[shot_index].get(
                "source_event_generation_unit_counts"
            )
            ledger = raw_ledger if isinstance(raw_ledger, dict) else {}
            declared = ledger.get(str(event_id))
            if declared is not None:
                has_declared_counts = True
            declared_counts.append(declared)
        if has_declared_counts:
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in declared_counts
            ) or sum(declared_counts) != len(generation_units):
                raise ValueError(
                    f"event {event_id} has an invalid inherited generation-unit ledger"
                )
            unit_sizes = declared_counts
            declared_action_indexes: list[list[int]] = []
            declared_units_per_occurrence: list[list[dict[str, Any]]] = []
            unit_offset = 0
            for size in unit_sizes:
                selected_units = generation_units[
                    unit_offset : unit_offset + size
                ]
                unit_offset += size
                declared_units_per_occurrence.append(
                    [copy.deepcopy(unit) for unit in selected_units]
                )
                declared_action_indexes.append(sorted({
                    int(index)
                    for unit in selected_units
                    for index in (unit.get("ledger_indexes") or [])
                }))
            owned_indexes = {
                index
                for indexes in declared_action_indexes
                for index in indexes
            }
            # Sustained/duplicate constraints cost zero generation units, but
            # remain authored semantic detail.  Attach each to the next
            # visible action slice (or the final slice) without consuming Pxx
            # capacity.
            for action_index in range(len(actions)):
                if action_index in owned_indexes:
                    continue
                owner = next(
                    (
                        occurrence
                        for occurrence, indexes in enumerate(
                            declared_action_indexes
                        )
                        if any(index >= action_index for index in indexes)
                    ),
                    len(declared_action_indexes) - 1,
                )
                declared_action_indexes[owner].append(action_index)
            declared_action_indexes = [
                sorted(indexes) for indexes in declared_action_indexes
            ]
        else:
            base, remainder = divmod(len(actions), len(occurrence_shots))
            unit_sizes = [None] * len(occurrence_shots)
            declared_units_per_occurrence = []
        action_cursor = 0
        previous_state = str(event.get("start_state") or "").strip()
        for occurrence, shot_index in enumerate(occurrence_shots):
            if has_declared_counts:
                selected_action_indexes = declared_action_indexes[occurrence]
                action_slice = [
                    actions[index]
                    for index in selected_action_indexes
                    if 0 <= index < len(actions)
                ]
                local_index_by_event_index = {
                    event_action_index: local_index
                    for local_index, event_action_index in enumerate(
                        selected_action_indexes
                    )
                }
                scoped_generation_units: list[dict[str, Any]] = []
                for unit in declared_units_per_occurrence[occurrence]:
                    scoped = copy.deepcopy(unit)
                    for index_field in (
                        "ledger_indexes",
                        "contribution_ledger_indexes",
                        "effect_ledger_indexes",
                        "sustained_ledger_indexes",
                    ):
                        scoped[index_field] = [
                            local_index_by_event_index[index]
                            for index in unit.get(index_field) or []
                            if index in local_index_by_event_index
                        ]
                    scoped_generation_units.append(scoped)
                scoped_categories = [
                    normalized_event["categories"][index]
                    for index in selected_action_indexes
                ]
            else:
                size = base + (
                    1 if occurrence < remainder else 0
                )
                action_slice = actions[action_cursor : action_cursor + size]
                action_cursor += size
                scoped_generation_units = []
                scoped_categories = []
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
                "generation_action_units": scoped_generation_units,
                "generation_action_categories": scoped_categories,
            }
            previous_state = end_state

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
            _ground_production_beat_text_fields(shot, details)
            shot["who"] = canonical_who
            shot["character_ids"] = list(dict.fromkeys(
                str(character_id).strip()
                for event in details
                for character_id in (event.get("character_ids") or [])
                if str(character_id).strip()
            ))
            shot["source_event_casts"] = [
                {
                    "source_event_id": event_id,
                    "character_ids": list(dict.fromkeys(
                        str(character_id).strip()
                        for character_id in (event.get("character_ids") or [])
                        if str(character_id).strip()
                    )),
                }
                for event_id, event in zip(source_ids, details, strict=True)
            ]
            participant_refs = [
                dict(reference)
                for event in details
                for reference in (event.get("participant_refs") or [])
                if isinstance(reference, dict)
                and str(reference.get("ref_id") or "").strip()
            ]
            shot["participant_refs"] = list({
                str(reference["ref_id"]): reference
                for reference in participant_refs
            }.values())
            non_character_participants = list(dict.fromkeys(
                str(value).strip()
                for event in details
                for value in (event.get("non_character_participants") or [])
                if str(value).strip()
            ))
            if non_character_participants:
                shot["non_character_participants"] = non_character_participants
            else:
                shot.pop("non_character_participants", None)
            shot["_semantic_character_binding_verified"] = all(
                isinstance(event.get("character_ids"), list)
                and event.get("character_instance_ids")
                == event.get("character_ids")
                and isinstance(event.get("character_entity_ids"), list)
                and isinstance(event.get("participant_refs"), list)
                for event in details
            )

        excerpts = [str(event.get("source_excerpt") or "").strip() for event in details]
        excerpts = [excerpt for excerpt in excerpts if excerpt]
        if excerpts:
            shot["source_excerpt"] = "\n".join(dict.fromkeys(excerpts))

        sequence_ids = [str(event.get("sequence_id")) for event in details if event.get("sequence_id")]
        sequence_ids = list(dict.fromkeys(sequence_ids))
        source_action_unit_refs = [
            {
                "source_event_id": event_id,
                "action_unit_id": str(event_by_id[event_id]["action_unit_id"]),
            }
            for event_id in source_ids
            if event_id in event_by_id and event_by_id[event_id].get("action_unit_id")
        ]
        action_unit_ids = [
            reference["action_unit_id"] for reference in source_action_unit_refs
        ]
        micro_actions = [
            action
            for _event_id, event_slice in slices
            for action in event_slice["micro_actions"]
        ]
        roles = [str(event.get("event_role")) for event in details if event.get("event_role")]
        shot["source_sequence_ids"] = sequence_ids
        if director_intents:
            if len(sequence_ids) != 1 or sequence_ids[0] not in director_intents:
                raise ValueError(
                    "adapted shot cannot bind one director intent: "
                    f"source_sequence_ids={sequence_ids}"
                )
            shot["director_intent"] = _build_production_director_intent(
                director_intents[sequence_ids[0]],
                details,
                source_event_ids=source_ids,
                shot=shot,
            )
        shot["source_action_unit_ids"] = list(dict.fromkeys(action_unit_ids))
        shot["source_action_unit_refs"] = source_action_unit_refs
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
        shot["production_action_refs"] = [
            copy.deepcopy(event["production_action_selection"])
            for event in details
            if isinstance(event.get("production_action_selection"), dict)
        ]
        shot["micro_actions"] = micro_actions
        generation_units: List[Dict[str, Any]] = []
        generation_categories: List[str] = []
        ledger_offset = 0
        for event_id, event_slice in slices:
            slice_actions = list(event_slice["micro_actions"])
            precomputed_units = event_slice.get("generation_action_units")
            if precomputed_units:
                normalized_categories = list(
                    event_slice.get("generation_action_categories") or []
                )
                normalized_units = precomputed_units
            else:
                normalized = normalize_event_action_units(
                    event_by_id[event_id],
                    actions=slice_actions,
                    seen=set(seen_before_event[event_id]),
                    max_motion_contributions_per_slice=(
                        get_video_capabilities().motion_contribution_limit
                    ),
                )
                normalized_categories = list(normalized["categories"])
                normalized_units = normalized["generation_action_units"]
            generation_categories.extend(normalized_categories)
            for unit in normalized_units:
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
        included_actions = {str(action).strip() for action in micro_actions if str(action).strip()}
        choreography: List[Dict[str, Any]] = []
        for event in details:
            for raw_beat in event.get("body_action_choreography") or []:
                if not isinstance(raw_beat, dict):
                    continue
                beat_action = str(raw_beat.get("micro_action") or "").strip()
                if beat_action and beat_action not in included_actions:
                    continue
                serialized_beat = dict(raw_beat)
                serialized_beat["beat"] = len(choreography) + 1
                choreography.append(serialized_beat)
        if choreography:
            shot["body_action_choreography"] = choreography
        apply_body_action_contract(shot)
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
    "逐 sequence 导演意图（只规定为什么这样拍，不替你决定具体镜头字段）：\n"
    "{director_intents_json}\n\n"
    "固定 beat/sequence 槽位：\n{sequence_beat_plan}\n\n"
    "只做全局改编与镜头语言规划，不要展开 visual 或人物外貌。输出严格 JSON 对象："
    '{{"strategy":"一句话改编策略","beats":[{{"beat_order":1,"source_events":[1],'
    '"dropped_source_events":[],"action":"keep/merge","reason":"一句话理由","who":["角色主名"],'
    '"where":"地点","what":"一句话事件","suggested_duration":15,'
    '"shot_size":"medium_wide","camera_angle":"eye_level","camera_movement":"dolly_in",'
    '"lighting_key":"natural","shot_intent":"establishing",'
    '"hero_moment":false,"texture_keywords":["场景中的具体材质","场景中的具体光影"]}}]}}。\n'
    + "【镜头语言合法词表】以下枚举字段只能逐字选用所列值，禁止发明组合值或方向后缀：\n"
    + _SHOT_LANGUAGE_ENUM_CONTRACT
    + CAMERA_ANGLE_PLANNING_INSTRUCTIONS
    + "\n"
    + CAMERA_MOTION_PLANNING_INSTRUCTIONS
    + "\n"
    + "hero_moment 必须为 JSON 布尔值；texture_keywords 必须为 2–4 个非空字符串。\n"
    + "【全局铁律】\n"
    "1. beats 数量必须恰好等于 {beat_count}，总建议时长应接近 {target_duration} 秒（±10%）；"
    "每个 beat 的 generation_action_unit_count 合计不得超过 "
    "{max_generation_action_units_per_beat}，且固定槽位中的 max_generation_action_units 是该 beat 的更精确硬上限。\n"
    "2. 每个输入事件编号必须进入 source_events 或 dropped_source_events；两者不得重叠。"
    "每个 beat 至少保留一个 source_event。非关键重复动作可显式删减，但 scene_setup、turning_point、"
    "dramatic_turn、consequence 必须保留。\n"
    "2a. 输入事件中的 micro_actions 是经过故事时钟缩放后的唯一生产动作合同；"
    "production_action_selection 记录其来源索引。what/source_excerpt 只保留事件事实与结果，"
    "不得据此恢复 omitted_source_micro_action_indexes 指向的动作，也不得另造替代动作。\n"
    "3. keep 保留关键因果/情感节点，merge 合并连续事件；dropped_source_events 只放不影响因果链的删减。\n"
    "4. 台词归属必须忠于原事件；who 只能使用角色列表主名，别名改为主名，群众不得写入 who。\n"
    "5. beat 是导演级叙事镜头，不是单次视频调用。一个 beat 应承载一个内容完整的连续因果段，"
    "可以包含多个有序 action_unit；同一 sequence_id、同一时空、同一主体和因果链应优先合并，"
    "但必须完整保留 source_events 与 micro_actions 原顺序，后续会拆成 P01/P02…；不同 sequence_id、"
    "减少一级 beat 数只允许把原有内容段依次装入这些 Pxx，禁止只保留前几个段落并裁掉后续剧情；"
    "换场/跳时、主体切换或真实导演硬切才新增 beat。turning_point 可与同 sequence 中紧邻的因果动作共用 beat，但必须明确"
    "保留转折。被保留事件在 beats 中的引用次数不得少于输入中的 "
    "minimum_kept_primary_beat_occurrences；同一事件的后续引用只承载尚未表现的动作，不得重放。\n"
    "6. sequence_id 与 continuity_before 是生成连续性依据。每个 beat 只能引用固定槽位指定 sequence 的事件；"
    "同一 sequence 的连续单元落在相邻 beat，换场/跳时/关系转折不得为了省镜头而错误连拍。\n"
    "6a. 每个 beat 必须服从对应 sequence 的 director intent：scene_goal 决定叙事目的，emotion_arc 决定"
    "情绪推进，visual_focus 决定观众注意点，spatial_intent 决定空间关系，transition_intent 决定边界设计；"
    "Director 不替你决定景别、机位、运镜、焦段、光影、镜头数或时长。\n"
    "7. shot_size、camera_angle、camera_movement、lighting_key、shot_intent、hero_moment、texture_keywords "
    "是骨架的全局结构字段，全部必填。相邻 beat 景别必须形成差异，动作 beat 不得全部 static；"
    "4 个及以上 beat 必须至少一个 hero_moment=true，每个 beat 给出 2–4 个具体纹理关键词。"
    "一镜到底可以保持同一真实光源，但镜头变化必须来自源文本允许的构图距离与运镜；禁止为了"
    "追求差异虚构转场、跳时、摇臂、无人机、环绕或剧本明令禁止的运镜。禁止展开对白、visual、"
    "Identity Anchor 或人物外貌。"
)


CANONICAL_BEAT_LANGUAGE_PROMPT = (
    "目标时长：{target_duration}秒，共 {beat_count} 个一级镜头。代码已经固定每个镜头的 "
    "sequence、来源事件、执行子片、时长与容量；你不得重新分配、删减、重排或补写这些账本。\n\n"
    "当前生产事件（只用于理解每个固定编号的内容）：\n{events_json}\n\n"
    "canonical 角色列表：\n{characters_summary}\n\n"
    "不可变 beat 合同：\n{canonical_beat_contracts}\n\n"
    "逐 sequence 导演意图（只说明为什么这样拍）：\n{director_intents_json}\n\n"
    "你只负责摄影语言、构图、光影和视觉节奏。输出严格 JSON 对象："
    '{{"strategy":"一句话摄影策略","beats":[{{"beat_order":1,'
    '"shot_size":"medium_wide","camera_angle":"eye_level",'
    '"camera_movement":"dolly_in","lighting_key":"natural",'
    '"shot_intent":"establishing","hero_moment":false,'
    '"texture_keywords":["具体材质","具体光影"]}}]}}。\n'
    + "【镜头语言合法词表】以下枚举字段只能逐字选用所列值，禁止发明组合值或方向后缀：\n"
    + _SHOT_LANGUAGE_ENUM_CONTRACT
    + CAMERA_ANGLE_PLANNING_INSTRUCTIONS
    + "\n"
    + CAMERA_MOTION_PLANNING_INSTRUCTIONS
    + "\n"
    + "hero_moment 必须为 JSON 布尔值；texture_keywords 必须为 2–4 个非空字符串。\n"
    "beats 数量必须恰好为 {beat_count}，beat_order 从 1 连续递增。"
    "禁止输出 source_events、dropped_source_events、sequence_id、action、"
    "source_event_generation_unit_counts、timeline_assignment_ids、who、where、what、"
    "suggested_duration 或任何动作文本。相邻镜头景别应形成有意义差异；动作镜头不能全部 static；"
    "只有来源允许时才能使用环绕、摇臂、无人机或跳切，不得为追求变化虚构时空边界。"
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
        "dropped_source_events",
        "action",
        "reason",
        "who",
        "where",
        "what",
        "suggested_duration",
        *_SHOT_LANGUAGE_FIELDS,
    }
    kept_events: set[int] = set()
    dropped_events: set[int] = set()
    for i, beat in enumerate(beats, 1):
        if not isinstance(beat, dict):
            raise ValueError(f"第 {i} 个 beat 不是字典")
        missing = required - set(beat)
        if missing:
            raise ValueError(f"第 {i} 个 beat 缺少字段: {missing}")
        if beat["action"] not in {"keep", "merge"}:
            raise ValueError(f"第 {i} 个 beat action 无效: {beat['action']}")
        for field in ("source_events", "dropped_source_events"):
            if not isinstance(beat[field], list):
                raise ValueError(f"第 {i} 个 beat {field} 必须是数组")
            if len(beat[field]) != len(set(beat[field])):
                raise ValueError(f"第 {i} 个 beat {field} 不得包含重复编号")
            for event_id in beat[field]:
                if not isinstance(event_id, int) or not 1 <= event_id <= event_count:
                    raise ValueError(f"第 {i} 个 beat 引用了无效事件编号: {event_id}")
        if not beat["source_events"]:
            raise ValueError(f"第 {i} 个 beat 至少需要一个保留的 source_event")
        overlap = set(beat["source_events"]) & set(beat["dropped_source_events"])
        if overlap:
            raise ValueError(f"第 {i} 个 beat 同时保留并删减事件: {sorted(overlap)}")
        repeated_drops = dropped_events & set(beat["dropped_source_events"])
        if repeated_drops:
            raise ValueError(f"删减事件只能记录一次: {sorted(repeated_drops)}")
        kept_events.update(beat["source_events"])
        dropped_events.update(beat["dropped_source_events"])
    overlap = kept_events & dropped_events
    if overlap:
        raise ValueError(f"事件不得跨 beat 同时保留并删减: {sorted(overlap)}")
    missing_events = set(range(1, event_count + 1)) - kept_events - dropped_events
    if missing_events:
        raise ValueError(f"beat 未覆盖事件编号: {sorted(missing_events)}")
    _validate_authored_shot_language(beats, label="beat")
    for beat in beats:
        apply_camera_motion_contract(beat)
    parsed.setdefault("strategy", "")
    return parsed


def _parse_canonical_beat_language(
    response: str,
    expected_count: int,
) -> Dict[str, Any]:
    """Parse only the model-owned camera-language projection."""
    text = response.strip()
    if "```" in text:
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("beats"), list):
        raise ValueError("镜头语言响应必须是包含 beats 数组的 JSON 对象")
    beats = parsed["beats"]
    if len(beats) != expected_count:
        raise ValueError(
            f"镜头语言 beat 数量应为 {expected_count}，实际为 {len(beats)}"
        )
    protected_fields = {
        "source_events",
        "dropped_source_events",
        "sequence_id",
        "action",
        "source_event_generation_unit_counts",
        "timeline_assignment_ids",
        "who",
        "where",
        "what",
        "suggested_duration",
    }
    for beat_order, beat in enumerate(beats, 1):
        if not isinstance(beat, dict):
            raise ValueError(f"第 {beat_order} 个镜头语言 beat 不是字典")
        if beat.get("beat_order") != beat_order:
            raise ValueError("镜头语言 beat_order 必须从 1 连续递增")
        leaked = protected_fields & set(beat)
        if leaked:
            raise ValueError(
                "模型不得改写 canonical beat 账本字段: "
                f"{sorted(leaked)}"
            )
    _validate_authored_shot_language(beats, label="镜头语言 beat")
    for beat in beats:
        apply_camera_motion_contract(beat)
    parsed.setdefault("strategy", "")
    return parsed


def _canonical_beat_contracts(
    events: List[Dict[str, Any]],
    duration_scaled_event_plan: Dict[str, Any],
    timeline_layout_binding: Dict[str, Any],
) -> list[dict[str, Any]]:
    """Project the persisted duration vector into immutable Sxx contracts."""
    if duration_scaled_event_plan.get("schema") != DURATION_SCALED_EVENT_PLAN_SCHEMA:
        raise ValueError("canonical beat contracts require duration plan v4")
    if timeline_layout_binding.get("schema") != ACTION_TIMELINE_SCHEMA:
        raise ValueError("canonical beat contracts require action timeline v1")
    if timeline_layout_binding.get("duration_plan_schema") != (
        DURATION_SCALED_EVENT_PLAN_SCHEMA
    ):
        raise ValueError("timeline binding does not match the duration plan")

    sequence_plan = duration_scaled_event_plan.get("sequence_beat_plan")
    capacities = duration_scaled_event_plan.get(
        "generation_action_unit_capacities_per_beat"
    )
    event_counts_per_beat = duration_scaled_event_plan.get(
        "source_event_generation_unit_counts_per_beat"
    )
    binding_sxx = timeline_layout_binding.get("sxx")
    assignments = timeline_layout_binding.get("assignments")
    zero_attachments = timeline_layout_binding.get(
        "zero_story_time_attachments"
    )
    values = (
        sequence_plan,
        capacities,
        event_counts_per_beat,
        binding_sxx,
    )
    if any(not isinstance(value, list) for value in values):
        raise ValueError("canonical beat ledgers must be arrays")
    beat_count = len(sequence_plan)
    if (
        beat_count < 1
        or len(capacities) != beat_count
        or len(event_counts_per_beat) != beat_count
        or len(binding_sxx) != beat_count
    ):
        raise ValueError("canonical beat ledgers have inconsistent lengths")
    if not isinstance(assignments, list) or not isinstance(zero_attachments, list):
        raise ValueError("canonical timeline binding is incomplete")

    generation_counts = _event_generation_action_unit_counts(events)
    attachment_by_shot: dict[str, list[int]] = {}
    attached_zero_events: set[int] = set()
    for attachment in zero_attachments:
        if not isinstance(attachment, dict):
            raise ValueError("zero-story-time attachment must be an object")
        event_id = attachment.get("source_event_id")
        sxx_id = str(attachment.get("sxx_id") or "")
        if (
            not isinstance(event_id, int)
            or event_id not in generation_counts
            or generation_counts[event_id] != 0
            or event_id in attached_zero_events
            or not sxx_id
            or attachment.get("consumes_temporal_capacity") is not False
        ):
            raise ValueError("zero-story-time attachment is invalid")
        attached_zero_events.add(event_id)
        attachment_by_shot.setdefault(sxx_id, []).append(event_id)

    assignment_counts: dict[tuple[str, int], int] = {}
    assignment_ids: set[str] = set()
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("timeline assignment must be an object")
        assignment_id = str(assignment.get("assignment_id") or "")
        event_id = assignment.get("source_event_id")
        sxx_id = str(assignment.get("sxx_id") or "")
        if (
            not assignment_id
            or assignment_id in assignment_ids
            or not isinstance(event_id, int)
            or event_id not in generation_counts
            or not sxx_id
        ):
            raise ValueError("timeline assignment identity is invalid")
        assignment_ids.add(assignment_id)
        assignment_counts[(sxx_id, event_id)] = (
            assignment_counts.get((sxx_id, event_id), 0) + 1
        )

    declared_totals = {event_id: 0 for event_id in generation_counts}
    contracts: list[dict[str, Any]] = []
    for beat_order, (sequence_item, capacity, raw_counts, sxx) in enumerate(
        zip(
            sequence_plan,
            capacities,
            event_counts_per_beat,
            binding_sxx,
            strict=True,
        ),
        1,
    ):
        if (
            not isinstance(sequence_item, dict)
            or sequence_item.get("beat_order") != beat_order
            or not isinstance(capacity, int)
            or isinstance(capacity, bool)
            or capacity < 1
            or not isinstance(raw_counts, dict)
            or not isinstance(sxx, dict)
        ):
            raise ValueError("canonical beat vector contains invalid entries")
        sequence_id = str(sequence_item.get("sequence_id") or "").strip()
        sxx_id = f"S{beat_order:03d}"
        if (
            not sequence_id
            or sxx.get("sxx_order") != beat_order
            or sxx.get("sxx_id") != sxx_id
            or str(sxx.get("sequence_id") or "").strip() != sequence_id
        ):
            raise ValueError("canonical beat sequence and timeline binding drifted")

        normalized_counts: dict[str, int] = {}
        for raw_event_id, raw_count in raw_counts.items():
            try:
                event_id = int(raw_event_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("canonical beat references an invalid event") from exc
            if (
                event_id not in generation_counts
                or isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count <= 0
            ):
                raise ValueError("canonical beat has an invalid event count")
            event_sequence = (
                str(events[event_id - 1].get("sequence_id") or "").strip()
                or "__unspecified__"
            )
            if event_sequence != sequence_id:
                raise ValueError("canonical beat crosses a sequence boundary")
            if assignment_counts.get((sxx_id, event_id), 0) != raw_count:
                raise ValueError("timeline assignment count drifted from duration plan")
            normalized_counts[str(event_id)] = raw_count
            declared_totals[event_id] += raw_count

        zero_event_ids = attachment_by_shot.get(sxx_id, [])
        for event_id in zero_event_ids:
            event_sequence = (
                str(events[event_id - 1].get("sequence_id") or "").strip()
                or "__unspecified__"
            )
            if event_sequence != sequence_id:
                raise ValueError("zero-story-time fact crossed a sequence boundary")
            normalized_counts[str(event_id)] = 0

        source_event_ids = sorted(int(value) for value in normalized_counts)
        if not source_event_ids:
            raise ValueError("canonical Sxx cannot be detached from source facts")
        pxx = sxx.get("pxx")
        if not isinstance(pxx, list) or not pxx:
            raise ValueError("canonical Sxx requires Pxx timeline records")
        pxx_assignment_ids: list[str] = []
        story_duration = 0.0
        for pxx_record in pxx:
            if not isinstance(pxx_record, dict):
                raise ValueError("canonical Pxx timeline record is invalid")
            raw_assignment_ids = pxx_record.get("assignment_ids")
            duration = pxx_record.get("effective_story_duration_s")
            if (
                not isinstance(raw_assignment_ids, list)
                or isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or float(duration) <= 0
            ):
                raise ValueError("canonical Pxx duration/assignment ledger is invalid")
            pxx_assignment_ids.extend(str(value) for value in raw_assignment_ids)
            story_duration += float(duration)
        expected_assignment_ids = {
            str(item.get("assignment_id") or "")
            for item in assignments
            if isinstance(item, dict) and item.get("sxx_id") == sxx_id
        }
        if set(pxx_assignment_ids) != expected_assignment_ids or len(
            pxx_assignment_ids
        ) != len(expected_assignment_ids):
            raise ValueError("canonical Pxx assignment ledger is incomplete")
        if sum(normalized_counts.values()) != len(pxx_assignment_ids):
            raise ValueError("canonical Sxx count and execution subslices differ")
        if sum(normalized_counts.values()) > capacity:
            raise ValueError("canonical Sxx exceeds its vector capacity")
        contracts.append({
            "beat_order": beat_order,
            "sxx_id": sxx_id,
            "sequence_id": sequence_id,
            "source_events": source_event_ids,
            "dropped_source_events": [],
            "source_event_generation_unit_counts": normalized_counts,
            "max_generation_action_units": capacity,
            "execution_subslice_count": len(pxx_assignment_ids),
            "timeline_assignment_ids": pxx_assignment_ids,
            "zero_story_time_source_event_ids": list(zero_event_ids),
            "suggested_duration": story_duration,
        })

    for event_id, expected_count in generation_counts.items():
        if declared_totals[event_id] != expected_count:
            raise ValueError(
                f"canonical beat ledger changed event {event_id} capacity from "
                f"{expected_count} to {declared_totals[event_id]}"
            )
        if expected_count == 0 and event_id not in attached_zero_events:
            raise ValueError(
                f"zero-story-time event {event_id} is not attached to an Sxx"
            )
        if expected_count > 0 and event_id in attached_zero_events:
            raise ValueError("execution event cannot use zero-story-time attachment")
    return contracts


def _validate_beats_match_canonical_contracts(
    beats: List[Dict[str, Any]],
    contracts: list[dict[str, Any]],
) -> None:
    """Reject checkpoint/model drift in every code-owned beat field."""
    if len(beats) != len(contracts):
        raise ValueError("canonical beat count changed after duration planning")
    protected_fields = (
        "beat_order",
        "sxx_id",
        "sequence_id",
        "source_events",
        "dropped_source_events",
        "source_event_generation_unit_counts",
        "max_generation_action_units",
        "execution_subslice_count",
        "timeline_assignment_ids",
        "zero_story_time_source_event_ids",
        "suggested_duration",
    )
    for beat_order, (beat, contract) in enumerate(
        zip(beats, contracts, strict=True),
        1,
    ):
        for field in protected_fields:
            if beat.get(field) != contract.get(field):
                raise ValueError(
                    f"canonical beat {beat_order} field {field} drifted from "
                    "the duration/timeline contract"
                )


def _validate_shots_match_beat_ledgers(
    shots: List[Dict[str, Any]],
    beats: List[Dict[str, Any]],
) -> None:
    """Keep resumed/expanded shots on the exact code-owned beat mapping."""
    if len(shots) > len(beats):
        raise ValueError("expanded shots exceed the canonical beat ledger")
    protected_fields = (
        "source_events",
        "dropped_source_events",
        "source_event_generation_unit_counts",
        "sxx_id",
        "sequence_id",
        "max_generation_action_units",
        "execution_subslice_count",
        "timeline_assignment_ids",
        "zero_story_time_source_event_ids",
        "suggested_duration",
    )
    for shot_order, (shot, beat) in enumerate(
        zip(shots, beats, strict=False),
        1,
    ):
        for field in protected_fields:
            if field in beat and shot.get(field) != beat.get(field):
                raise ValueError(
                    f"expanded shot {shot_order} field {field} drifted from "
                    "its canonical beat"
                )


_MANDATORY_ADAPTATION_EVENT_ROLES = frozenset({
    "scene_setup",
    "turning_point",
    "dramatic_turn",
    "consequence",
})


def _event_is_mandatory_for_adaptation(event: Dict[str, Any]) -> bool:
    role = str(event.get("event_role") or "").strip().lower()
    return role in _MANDATORY_ADAPTATION_EVENT_ROLES or bool(event.get("dramatic_turn"))


def terminal_outcome_event_ids(events: List[Dict[str, Any]]) -> set[int]:
    """Protect the final authored narrative fact even when its role drifts.

    Event extraction already removes production-only directives.  Walking
    backward still avoids anchoring a malformed empty record while preserving
    a terminal transition or resolution that has visible state but no atomic
    micro-action.
    """
    for event_id in range(len(events), 0, -1):
        event = events[event_id - 1]
        has_visible_fact = any(
            str(event.get(field) or "").strip()
            for field in (
                "what",
                "visual",
                "source_excerpt",
                "start_state",
                "end_state",
            )
        ) or bool(event.get("micro_actions") or event.get("lines"))
        if has_visible_fact:
            return {event_id}
    return set()


def _base_mandatory_adaptation_event_ids(
    events: List[Dict[str, Any]],
) -> set[int]:
    structural_ids = {
        event_id
        for event_id, event in enumerate(events, 1)
        if _event_is_mandatory_for_adaptation(event)
    }
    return structural_ids | terminal_outcome_event_ids(events)


def _continuous_predecessor_event_ids(
    events: List[Dict[str, Any]],
    seed_event_ids: set[int],
) -> set[int]:
    """Return seeds plus structured predecessors up to each cut boundary.

    ``continuity_before=continuous`` is an authored temporal contract: keeping
    the current event while deleting its previous same-sequence event would
    create an unexplained state jump.
    """
    invalid_ids = seed_event_ids - set(range(1, len(events) + 1))
    if invalid_ids:
        raise ValueError(
            f"continuous predecessor closure received unknown events: "
            f"{sorted(invalid_ids)}"
        )
    required = set(seed_event_ids)
    previous_in_sequence: Dict[int, int] = {}
    last_event_by_sequence: Dict[str, int] = {}
    for event_id, event in enumerate(events, 1):
        sequence_id = (
            str(event.get("sequence_id") or "").strip() or "__unspecified__"
        )
        previous_event_id = last_event_by_sequence.get(sequence_id)
        if previous_event_id is not None:
            previous_in_sequence[event_id] = previous_event_id
        last_event_by_sequence[sequence_id] = event_id

    pending = list(required)
    while pending:
        event_id = pending.pop()
        event = events[event_id - 1]
        boundary = str(event.get("continuity_before") or "").strip().lower()
        if boundary != "continuous":
            continue
        previous_event_id = previous_in_sequence.get(event_id)
        if previous_event_id is None:
            sequence_id = (
                str(event.get("sequence_id") or "").strip()
                or "__unspecified__"
            )
            raise ValueError(
                f"event {event_id} declares continuous entry for sequence "
                f"{sequence_id!r} without a predecessor"
            )
        if previous_event_id not in required:
            required.add(previous_event_id)
            pending.append(previous_event_id)
    return required


def _mandatory_adaptation_event_ids(
    events: List[Dict[str, Any]],
) -> set[int]:
    """Protect mandatory facts and their structured continuous causes."""
    return _continuous_predecessor_event_ids(
        events,
        _base_mandatory_adaptation_event_ids(events),
    )


def _dropped_source_event_ids(beats: List[Dict[str, Any]]) -> set[int]:
    return {
        event_id
        for beat in beats
        for event_id in (beat.get("dropped_source_events") or [])
        if isinstance(event_id, int)
    }


def _validate_beat_event_order(
    beats: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> None:
    """Require each sequence's source-event ranges to be contiguous and monotonic."""
    event_by_id = {event_id: event for event_id, event in enumerate(events, 1)}
    positions: Dict[int, List[int]] = {}
    for beat_index, beat in enumerate(beats):
        raw_ids = beat.get("source_events") or []
        if not isinstance(raw_ids, list):
            raise ValueError(f"beat {beat_index + 1} source_events must be an array")
        for event_id in dict.fromkeys(raw_ids):
            if event_id in event_by_id:
                positions.setdefault(event_id, []).append(beat_index)

    for event_id, event_positions in positions.items():
        expected = list(range(event_positions[0], event_positions[-1] + 1))
        if event_positions != expected:
            raise ValueError(
                f"event order jumps backward: event {event_id} occupies "
                f"non-contiguous beats {event_positions}"
            )

    ordered_sequences: Dict[str, List[int]] = {}
    for event_id, event in event_by_id.items():
        if event_id not in positions:
            continue
        sequence = str(event.get("sequence_id") or "").strip() or "__unspecified__"
        ordered_sequences.setdefault(sequence, []).append(event_id)
    for sequence, event_ids in ordered_sequences.items():
        previous_event_id: int | None = None
        previous_positions: List[int] | None = None
        for event_id in event_ids:
            current_positions = positions[event_id]
            if (
                previous_positions is not None
                and previous_positions[-1] > current_positions[0]
            ):
                raise ValueError(
                    "event order jumps backward: "
                    f"sequence {sequence} event {event_id} starts at beat "
                    f"{current_positions[0] + 1} before event {previous_event_id} "
                    f"finishes at beat {previous_positions[-1] + 1}"
                )
            previous_event_id = event_id
            previous_positions = current_positions


def _validate_beat_action_capacity(
    beats: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    capabilities: VideoModelCapabilities | None = None,
    *,
    max_content_beats_per_primary_shot: int = (
        MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    ),
) -> None:
    """Allow inner Pxx expansion while rejecting unrelated narrative merges."""
    profile = capabilities or get_video_capabilities()
    event_by_id = {i: event for i, event in enumerate(events, 1)}
    occurrence_requirements = _event_primary_occurrence_requirements(
        events,
        profile,
        max_content_beats_per_primary_shot=(
            max_content_beats_per_primary_shot
        ),
    )
    dropped_event_ids = _dropped_source_event_ids(beats)
    kept_event_ids = {
        event_id
        for beat in beats
        for event_id in (beat.get("source_events") or [])
        if isinstance(event_id, int)
    }
    overlap = kept_event_ids & dropped_event_ids
    if overlap:
        raise ValueError(
            f"events cannot be both kept and dropped: {sorted(overlap)}"
        )
    invalid_dropped = dropped_event_ids - set(event_by_id)
    if invalid_dropped:
        raise ValueError(f"dropped events are invalid: {sorted(invalid_dropped)}")
    missing_event_ids = set(event_by_id) - kept_event_ids - dropped_event_ids
    if missing_event_ids:
        raise ValueError(
            f"events must be explicitly kept or dropped: {sorted(missing_event_ids)}"
        )
    mandatory_event_ids = _mandatory_adaptation_event_ids(events)
    missing_continuous_predecessors = (
        _continuous_predecessor_event_ids(events, kept_event_ids)
        - kept_event_ids
    )
    if missing_continuous_predecessors:
        raise ValueError(
            "kept events omit continuous predecessors: "
            f"{sorted(missing_continuous_predecessors)}"
        )
    _validate_beat_event_order(beats, events)
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
        if content_beats > max_content_beats_per_primary_shot:
            raise ValueError(
                f"beat {index} requires {content_beats} story-bearing clips for "
                f"{profile.name}; maximum is "
                f"{max_content_beats_per_primary_shot}"
            )
    for event_id, event in event_by_id.items():
        observed = sum(
            event_id in beat.get("source_events", [])
            for beat in beats
            if beat.get("action") != "drop"
        )
        if observed == 0 and event_id in dropped_event_ids:
            if event_id in mandatory_event_ids:
                raise ValueError(f"mandatory event {event_id} cannot be dropped")
            continue
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


def _beat_generation_unit_loads(
    beats: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> List[int]:
    """Distribute each kept event's normalized units across its beat occurrences."""
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
        declared = []
        has_declared = False
        for beat_index in occurrence_positions:
            raw_ledger = beats[beat_index].get(
                "source_event_generation_unit_counts"
            )
            ledger = raw_ledger if isinstance(raw_ledger, dict) else {}
            value = ledger.get(str(event_id))
            if value is not None:
                has_declared = True
            declared.append(value)
        if has_declared:
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in declared
            ) or sum(declared) != generation_unit_count:
                raise ValueError(
                    f"event {event_id} has an invalid per-shot generation-unit ledger"
                )
            sizes = declared
        else:
            base, remainder = divmod(
                generation_unit_count,
                len(occurrence_positions),
            )
            sizes = [
                base + (1 if occurrence < remainder else 0)
                for occurrence in range(len(occurrence_positions))
            ]
        for beat_index, size in zip(occurrence_positions, sizes, strict=True):
            generation_units_by_beat[beat_index] += size

    return generation_units_by_beat


def _beat_content_loads(
    beats: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    capabilities: VideoModelCapabilities,
) -> List[int]:
    """Calculate the eventual per-shot clip load after event occurrence slicing."""
    loads = []
    for generation_unit_count in _beat_generation_unit_loads(beats, events):
        action_beats = (
            math.ceil(
                generation_unit_count / capabilities.temporal_slice_limit
            )
            if generation_unit_count else 1
        )
        loads.append(max(1, action_beats))
    return loads


def _sequence_beat_plan(
    events: List[Dict[str, Any]],
    beat_count: int,
    max_generation_units_per_beat: int,
) -> List[str]:
    """Allocate legacy scalar beat slots for historical artifact migration.

    Current runs use ``_vector_sequence_beat_allocation`` and must never call
    this scalar estimator after a duration plan has been persisted.  Keeping
    the function isolated preserves deterministic recovery for old artifacts
    without allowing their packing semantics back into production planning.
    """
    if beat_count < 1 or max_generation_units_per_beat < 1:
        raise ValueError("beat count and per-beat generation capacity must be positive")
    ordered_sequences: List[str] = []
    for event in events:
        sequence = str(event.get("sequence_id") or "").strip() or "__unspecified__"
        if sequence not in ordered_sequences:
            ordered_sequences.append(sequence)
    if not ordered_sequences:
        raise ValueError("cannot plan beat sequences without source events")
    generation_units = _event_generation_action_unit_counts(events)
    total_units = {sequence: 0 for sequence in ordered_sequences}
    mandatory_units = {sequence: 0 for sequence in ordered_sequences}
    mandatory_event_units = {sequence: [] for sequence in ordered_sequences}
    mandatory_sequences: set[str] = set()
    mandatory_event_ids = _mandatory_adaptation_event_ids(events)
    for event_id, event in enumerate(events, 1):
        sequence = str(event.get("sequence_id") or "").strip() or "__unspecified__"
        units = generation_units[event_id]
        total_units[sequence] += units
        if event_id in mandatory_event_ids:
            mandatory_sequences.add(sequence)
            mandatory_units[sequence] += units
            mandatory_event_units[sequence].append(units)

    def ordered_mandatory_slots(sequence: str) -> int:
        slots = 0
        last_load = 0
        for units in mandatory_event_units[sequence]:
            occurrence_count = max(1, math.ceil(units / max_generation_units_per_beat))
            base, remainder = divmod(units, occurrence_count)
            chunks = [
                base + (1 if occurrence < remainder else 0)
                for occurrence in range(occurrence_count)
            ]
            first, *tail = chunks
            if slots == 0:
                slots = 1
                last_load = first
            elif last_load + first <= max_generation_units_per_beat:
                last_load += first
            else:
                slots += 1
                last_load = first
            for chunk in tail:
                slots += 1
                last_load = chunk
        return slots

    allocations = {
        sequence: (
            max(
                1,
                math.ceil(
                    mandatory_units[sequence] / max_generation_units_per_beat
                ),
                ordered_mandatory_slots(sequence),
            )
            if sequence in mandatory_sequences
            else 0
        )
        for sequence in ordered_sequences
    }
    required = sum(allocations.values())
    if required > beat_count:
        raise ValueError(
            f"mandatory sequence content needs {required} beats; only {beat_count} available"
        )
    while sum(allocations.values()) < beat_count:
        sequence = max(
            ordered_sequences,
            key=lambda value: (
                total_units[value]
                - allocations[value] * max_generation_units_per_beat,
                total_units[value],
                -ordered_sequences.index(value),
            ),
        )
        allocations[sequence] += 1
    return [
        sequence
        for sequence in ordered_sequences
        for _ in range(allocations[sequence])
        if allocations[sequence] > 0
    ]


def _vector_sequence_beat_allocation(
    events: List[Dict[str, Any]],
    generation_unit_capacities_per_beat: list[int],
) -> tuple[List[str], list[dict[str, int]]]:
    """Allocate ordered sequences and selected event slices to exact capacities.

    Unlike the legacy scalar estimator above, this is a production allocation:
    an event may continue into the next adjacent beat, capacities may be
    asymmetric, and the returned event matrix is the evidence consumed by
    downstream screenplay projection.  Non-mergeable sequences still own
    contiguous, disjoint beat intervals.
    """

    capacities = list(generation_unit_capacities_per_beat)
    if not capacities or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in capacities
    ):
        raise ValueError("generation-unit capacity ledger must be positive")
    if not events:
        raise ValueError("cannot allocate beat sequences without source events")

    ordered_sequences: list[str] = []
    events_by_sequence: dict[str, list[int]] = {}
    for event_id, event in enumerate(events, 1):
        sequence = str(event.get("sequence_id") or "").strip() or "__unspecified__"
        if sequence not in events_by_sequence:
            ordered_sequences.append(sequence)
            events_by_sequence[sequence] = []
        events_by_sequence[sequence].append(event_id)

    generation_units = _event_generation_action_unit_counts(events)
    mandatory_event_ids = _mandatory_adaptation_event_ids(events)
    sequence_contracts = []
    for sequence in ordered_sequences:
        event_ids = events_by_sequence[sequence]
        mandatory_ids = [
            event_id for event_id in event_ids if event_id in mandatory_event_ids
        ]
        selected_ids = [
            event_id for event_id in event_ids if generation_units[event_id] > 0
        ]
        sequence_contracts.append({
            "sequence_id": sequence,
            "event_ids": event_ids,
            "mandatory_event_ids": mandatory_ids,
            "selected_event_ids": selected_ids,
            # A zero-story-time posture, environment, or sustained-state fact
            # still belongs to its authored sequence.  The sequence therefore
            # needs one Sxx to which that fact can be attached even though the
            # fact does not consume Provider temporal capacity.
            "mandatory": bool(event_ids),
            "mandatory_units": sum(
                generation_units[event_id] for event_id in mandatory_ids
            ),
            "total_units": sum(
                generation_units[event_id] for event_id in event_ids
            ),
        })

    beat_count = len(capacities)
    # State is the number of beats already consumed.  The score first keeps as
    # much source capacity as possible, then minimizes unused Provider capacity,
    # then uses a stable earlier-sequence tie break through the length tuple.
    states: dict[int, tuple[tuple[int, int, tuple[int, ...]], tuple[int, ...]]] = {
        0: ((0, 0, ()), ())
    }
    for sequence_index, contract in enumerate(sequence_contracts):
        next_states: dict[
            int,
            tuple[tuple[int, int, tuple[int, ...]], tuple[int, ...]],
        ] = {}
        remaining_mandatory = sum(
            1
            for later in sequence_contracts[sequence_index + 1 :]
            if later["mandatory"]
        )
        for consumed, (score, lengths) in states.items():
            minimum = 1 if contract["mandatory"] else 0
            maximum = beat_count - consumed - remaining_mandatory
            for length in range(minimum, maximum + 1):
                segment_capacity = sum(capacities[consumed : consumed + length])
                if segment_capacity < contract["total_units"]:
                    continue
                served = min(contract["total_units"], segment_capacity)
                unused = max(0, segment_capacity - served)
                candidate_lengths = lengths + (length,)
                candidate_score = (
                    score[0] + served,
                    score[1] - unused,
                    tuple(-value for value in candidate_lengths),
                )
                key = consumed + length
                incumbent = next_states.get(key)
                if incumbent is None or candidate_score > incumbent[0]:
                    next_states[key] = (candidate_score, candidate_lengths)
        states = next_states
        if not states:
            break

    selected = states.get(beat_count)
    if selected is None:
        mandatory_sequences = sum(
            1 for contract in sequence_contracts if contract["mandatory"]
        )
        raise ValueError(
            "mandatory sequence content cannot fit the exact generation-unit "
            f"capacity vector {capacities}; mandatory_sequences={mandatory_sequences}"
        )
    _score, selected_lengths = selected
    sequence_plan = [
        contract["sequence_id"]
        for contract, length in zip(
            sequence_contracts,
            selected_lengths,
            strict=True,
        )
        for _ in range(length)
    ]
    if len(sequence_plan) != beat_count:
        raise ValueError("vector sequence allocation did not cover every beat")

    event_counts_per_beat: list[dict[str, int]] = [
        {} for _ in range(beat_count)
    ]
    beat_offset = 0
    for contract, length in zip(
        sequence_contracts,
        selected_lengths,
        strict=True,
    ):
        if length == 0:
            continue
        segment_end = beat_offset + length
        cursor = beat_offset
        loads = [0] * length
        # The duration plan has already selected the complete production
        # ledger. Every selected event slice is bound here; downstream prompt
        # generation may not silently drop or repack it.
        for event_id in contract["selected_event_ids"]:
            remaining = generation_units[event_id]
            while remaining:
                while (
                    cursor < segment_end
                    and loads[cursor - beat_offset] >= capacities[cursor]
                ):
                    cursor += 1
                if cursor >= segment_end:
                    raise ValueError(
                        "mandatory event allocation exceeded its sequence segment"
                    )
                room = capacities[cursor] - loads[cursor - beat_offset]
                assigned = min(remaining, room)
                event_counts_per_beat[cursor][str(event_id)] = (
                    event_counts_per_beat[cursor].get(str(event_id), 0)
                    + assigned
                )
                loads[cursor - beat_offset] += assigned
                remaining -= assigned
        beat_offset = segment_end

    return sequence_plan, event_counts_per_beat


DURATION_SCALED_EVENT_PLAN_SCHEMA = "honcut.duration-scaled-event-plan.v4"
DURATION_SCALED_ACTION_SELECTION_SCHEMA = (
    "honcut.duration-scaled-action-selection.v1"
)


def _source_event_generation_contracts(
    events: List[Dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize the immutable source ledger once for index-based selection."""
    contracts: list[dict[str, Any]] = []
    source_seen: set[str] = set()
    capabilities = get_video_capabilities()
    for event_id, event in enumerate(events, 1):
        raw_actions = event.get("micro_actions") or []
        if isinstance(raw_actions, str):
            raw_actions = [raw_actions]
        actions = [
            str(action).strip()
            for action in raw_actions
            if str(action).strip()
        ]
        normalized = normalize_event_action_units(
            event,
            actions=actions,
            seen=source_seen,
            max_motion_contributions_per_slice=(
                capabilities.motion_contribution_limit
            ),
        )
        contracts.append({
            "event_id": event_id,
            "actions": actions,
            "categories": list(normalized["categories"]),
            "generation_units": [
                dict(unit) for unit in normalized["generation_action_units"]
            ],
        })
    return contracts


def _representative_generation_unit_indexes(
    unit_count: int,
    target_count: int,
    *,
    event_role: str,
) -> list[int]:
    """Select ordered source-unit indexes without inventing choreography."""
    if target_count >= unit_count:
        return list(range(unit_count))
    if target_count <= 0 or unit_count <= 0:
        return []
    if target_count == 1:
        # Establishing facts are most legible at their authored onset; turns
        # and consequences are defined by the resulting visible state.
        return [0 if event_role == "scene_setup" else unit_count - 1]
    return [
        round(position * (unit_count - 1) / (target_count - 1))
        for position in range(target_count)
    ]


def _materialize_production_event(
    source_event: Dict[str, Any],
    contract: Dict[str, Any],
    selected_unit_indexes: list[int],
    *,
    preserve_duplicate_actions: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Compress complete source slices into fixed contiguous production groups.

    ``selected_unit_indexes`` historically meant "surviving" source units.
    That severed causal dependencies and silently removed the intervening
    screenplay.  Its length now supplies only the solver's production-unit
    target.  Every source action remains in exactly one contiguous rewrite
    group, and the persisted group ledger is the authority for the later real
    screenplay rewrite.
    """
    del preserve_duplicate_actions
    source_units = [
        copy.deepcopy(unit) for unit in contract["generation_units"]
    ]
    target_count = len(selected_unit_indexes)
    if source_units and not 1 <= target_count <= len(source_units):
        raise ValueError(
            "source-indexed screenplay rewrite target must retain at least "
            "one production action per non-empty event"
        )
    if not source_units and target_count:
        raise ValueError("zero-unit source event cannot gain production actions")

    if not source_units:
        source_hash = hashlib.sha256(
            json.dumps(
                contract["actions"],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        production_event = copy.deepcopy(source_event)
        production_event["micro_actions"] = list(contract["actions"])
        rewrite = {
            "schema": SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA,
            "source_event_id": int(contract["event_id"]),
            "source_micro_actions_sha256": source_hash,
            "source_micro_action_count": len(contract["actions"]),
            "source_generation_action_unit_count": 0,
            "production_generation_action_unit_count": 0,
            "groups": [],
            "static_source_facts": list(contract["actions"]),
            "omitted_source_micro_action_indexes": [],
        }
        production_event["production_action_rewrite"] = rewrite
        selection = {
            "schema": DURATION_SCALED_EVENT_PLAN_SCHEMA,
            "source_event_id": contract["event_id"],
            "selected_source_micro_action_indexes": list(
                range(1, len(contract["actions"]) + 1)
            ),
            "omitted_source_micro_action_indexes": [],
            "source_micro_actions_sha256": source_hash,
            "production_action_rewrite_schema": (
                SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA
            ),
            "production_action_groups": [],
        }
        production_event["production_action_selection"] = selection
        return production_event, selection

    unit_groups: list[list[dict[str, Any]]] = []
    unit_group_indexes: list[list[int]] = []
    if target_count:
        quotient, remainder = divmod(len(source_units), target_count)
        cursor = 0
        for group_index in range(target_count):
            group_size = quotient + (1 if group_index < remainder else 0)
            unit_group_indexes.append(
                list(range(cursor + 1, cursor + group_size + 1))
            )
            unit_groups.append(source_units[cursor:cursor + group_size])
            cursor += group_size
        if cursor != len(source_units) or any(not group for group in unit_groups):
            raise ValueError("source-indexed screenplay rewrite grouping failed")

    grouped_action_indexes: list[list[int]] = []
    assigned_indexes: set[int] = set()
    for unit_group in unit_groups:
        indexes = sorted({
            int(action_index)
            for unit in unit_group
            for action_index in unit.get("ledger_indexes") or []
            if 0 <= int(action_index) < len(contract["actions"])
        })
        grouped_action_indexes.append(indexes)
        assigned_indexes.update(indexes)
    unassigned_indexes = [
        index
        for index in range(len(contract["actions"]))
        if index not in assigned_indexes
    ]
    for source_index in unassigned_indexes:
        if not grouped_action_indexes:
            break
        destination = len(grouped_action_indexes) - 1
        for group_index, indexes in enumerate(grouped_action_indexes):
            if indexes and source_index <= indexes[-1]:
                destination = group_index
                break
        grouped_action_indexes[destination].append(source_index)
        grouped_action_indexes[destination].sort()
        assigned_indexes.add(source_index)

    if source_units and assigned_indexes != set(range(len(contract["actions"]))):
        raise ValueError("source-indexed screenplay rewrite lost source facts")
    source_hash = hashlib.sha256(
        json.dumps(
            contract["actions"],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    production_event = copy.deepcopy(source_event)
    groups: list[dict[str, Any]] = []
    source_index_to_group: dict[int, int] = {}
    production_actions: list[str] = []
    for production_index, (unit_group, unit_indexes, action_indexes) in enumerate(
        zip(
            unit_groups,
            unit_group_indexes,
            grouped_action_indexes,
            strict=True,
        ),
        1,
    ):
        source_actions = [
            contract["actions"][index] for index in action_indexes
        ]
        rewritten_action = (
            source_actions[0]
            if len(source_actions) == 1
            else "连续因果动作：" + " → ".join(source_actions)
        )
        production_actions.append(rewritten_action)
        for source_index in action_indexes:
            source_index_to_group[source_index + 1] = production_index
        groups.append({
            "production_action_index": production_index,
            "source_generation_unit_indexes": list(unit_indexes),
            "source_micro_action_indexes": [
                index + 1 for index in action_indexes
            ],
            "source_actions": source_actions,
            "source_actions_sha256": hashlib.sha256(
                json.dumps(
                    source_actions,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "rewritten_micro_action": rewritten_action,
            "maximum_motion_load": max(
                (int(unit.get("motion_load") or 1) for unit in unit_group),
                default=0,
            ),
            "pace_weight": sum(
                int(unit.get("pace_weight") or 1) for unit in unit_group
            ),
            "performers": list(dict.fromkeys(
                performer
                for unit in unit_group
                for performer in unit.get("performers") or []
            )),
            "targets": list(dict.fromkeys(
                target
                for unit in unit_group
                for target in unit.get("targets") or []
            )),
            "state_reads": list(dict.fromkeys(
                value
                for unit in unit_group
                for value in unit.get("state_reads") or []
            )),
            "state_writes": list(dict.fromkeys(
                value
                for unit in unit_group
                for value in unit.get("state_writes") or []
            )),
            "start_state": str(unit_group[0].get("start_state") or ""),
            "end_state": str(unit_group[-1].get("end_state") or ""),
        })
    production_event["micro_actions"] = production_actions
    choreography = source_event.get("body_action_choreography") or []
    if isinstance(choreography, list):
        rewritten_choreography = []
        for beat in choreography:
            if not isinstance(beat, dict):
                continue
            source_index = beat.get("micro_action_index")
            if source_index not in source_index_to_group:
                continue
            migrated_beat = copy.deepcopy(beat)
            migrated_beat["source_micro_action_index"] = source_index
            migrated_beat["micro_action_index"] = source_index_to_group[
                source_index
            ]
            rewritten_choreography.append(migrated_beat)
        production_event["body_action_choreography"] = rewritten_choreography
    production_event.pop("body_action_contract", None)
    production_event["action_temporal_relations"] = [
        {
            "micro_action_index": group_index,
            "performers": list(group["performers"]),
            "targets": list(group["targets"]),
            "action_kind": "state_change",
            "temporal_relation": "root" if group_index == 1 else "after",
            "reference_action_indexes": [] if group_index == 1 else [group_index - 1],
            "pace": (
                "slow" if int(group["pace_weight"]) >= 3 else (
                    "normal" if int(group["pace_weight"]) >= 2 else "fast"
                )
            ),
            "state_reads": list(group["state_reads"]),
            "state_writes": list(group["state_writes"]),
        }
        for group_index, group in enumerate(groups, 1)
    ]
    production_event["generation_motion_mode"] = (
        "atomic" if production_actions else "none"
    )
    rewrite = {
        "schema": SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA,
        "source_event_id": int(contract["event_id"]),
        "source_micro_actions_sha256": source_hash,
        "source_micro_action_count": len(contract["actions"]),
        "source_generation_action_unit_count": len(source_units),
        "production_generation_action_unit_count": target_count,
        "groups": groups,
        "omitted_source_micro_action_indexes": [],
    }
    production_event["production_action_rewrite"] = rewrite
    selection = {
        "schema": DURATION_SCALED_EVENT_PLAN_SCHEMA,
        "source_event_id": contract["event_id"],
        "selected_source_micro_action_indexes": list(
            range(1, len(contract["actions"]) + 1)
        ),
        "omitted_source_micro_action_indexes": [],
        "source_micro_actions_sha256": source_hash,
        "production_action_rewrite_schema": (
            SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA
        ),
        "production_action_groups": copy.deepcopy(groups),
    }
    production_event["production_action_selection"] = selection
    return production_event, selection


def _build_duration_scaled_event_plan(
    events: List[Dict[str, Any]],
    *,
    target_duration: int,
    beat_count: int,
    effective_shot_duration: int,
    capabilities: VideoModelCapabilities | None = None,
    max_generation_units_per_beat: int | None = None,
    maximum_total_generation_units: int | None = None,
    generation_unit_capacities_per_beat: list[int] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Create a source-linked production event ledger that fits beat slots.

    Source events remain immutable.  When mandatory event choreography cannot
    fit the requested story clock, this owner selects an ordered representative
    subset of source generation units.  The event fact, outcome, sequence and
    source indexes remain intact and auditable; no action text is synthesized.
    Whole optional-event omission remains the later skeleton owner's decision.
    """
    if not events:
        raise ValueError("duration-scaled event planning requires source events")
    if target_duration < 1 or beat_count < 1 or effective_shot_duration < 1:
        raise ValueError("duration-scaled event planning requires positive timing")
    profile = capabilities or get_video_capabilities()
    layout_content_beat_limit = MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    if (
        maximum_total_generation_units is not None
        and max_generation_units_per_beat is not None
    ):
        layout_content_beat_limit = max(
            MAX_CONTENT_BEATS_PER_PRIMARY_SHOT,
            math.ceil(
                int(max_generation_units_per_beat)
                / profile.temporal_slice_limit
            ),
        )
    per_beat_capacity = _generation_unit_capacity_for_story_duration(
        effective_shot_duration,
        profile,
        max_content_beats=layout_content_beat_limit,
    )
    if max_generation_units_per_beat is not None:
        if (
            isinstance(max_generation_units_per_beat, bool)
            or int(max_generation_units_per_beat) < 1
        ):
            raise ValueError("generation-unit rewrite capacity must be positive")
        per_beat_capacity = min(
            per_beat_capacity,
            int(max_generation_units_per_beat),
        )
    if generation_unit_capacities_per_beat is None:
        per_beat_capacities = [per_beat_capacity] * beat_count
    else:
        if (
            len(generation_unit_capacities_per_beat) != beat_count
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > per_beat_capacity
                for value in generation_unit_capacities_per_beat
            )
        ):
            raise ValueError(
                "generation-unit capacity ledger must match the primary shots"
            )
        per_beat_capacities = list(generation_unit_capacities_per_beat)
    if maximum_total_generation_units is not None:
        if (
            isinstance(maximum_total_generation_units, bool)
            or int(maximum_total_generation_units) < 0
        ):
            raise ValueError("total generation-unit capacity cannot be negative")
        maximum_total_generation_units = min(
            int(maximum_total_generation_units),
            sum(per_beat_capacities),
        )

    source_contracts = _source_event_generation_contracts(events)

    structural_mandatory_event_ids = {
        event_id
        for event_id, event in enumerate(events, 1)
        if _event_is_mandatory_for_adaptation(event)
    }
    terminal_event_ids = terminal_outcome_event_ids(events)
    base_mandatory_event_ids = (
        structural_mandatory_event_ids | terminal_event_ids
    )
    mandatory_event_ids = sorted(_mandatory_adaptation_event_ids(events))
    source_counts = {
        contract["event_id"]: len(contract["generation_units"])
        for contract in source_contracts
    }
    planned_event_ids = list(range(1, len(events) + 1))

    def build_candidate(
        selected_targets: tuple[int, ...],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        targets = dict(zip(planned_event_ids, selected_targets, strict=True))
        compression_active = any(
            targets[event_id] < source_counts[event_id]
            for event_id in planned_event_ids
        )
        production_events: List[Dict[str, Any]] = []
        records: List[Dict[str, Any]] = []
        for contract, source_event in zip(source_contracts, events, strict=True):
            event_id = contract["event_id"]
            units = contract["generation_units"]
            target_count = targets[event_id]
            role = str(source_event.get("event_role") or "").strip().lower()
            selected_unit_indexes = _representative_generation_unit_indexes(
                len(units),
                target_count,
                event_role=role,
            )
            production_event, selection = _materialize_production_event(
                source_event,
                contract,
                selected_unit_indexes,
                preserve_duplicate_actions=not compression_active,
            )
            production_events.append(production_event)
            records.append({
                "source_event_id": event_id,
                "sequence_id": (
                    str(source_event.get("sequence_id") or "").strip()
                    or "__unspecified__"
                ),
                "mandatory": event_id in mandatory_event_ids,
                "mandatory_reason": (
                    "structural"
                    if event_id in structural_mandatory_event_ids
                    else (
                        "terminal_outcome"
                        if event_id in terminal_event_ids
                        else (
                            "continuous_predecessor"
                            if event_id in mandatory_event_ids
                            else "optional"
                        )
                    )
                ),
                "source_micro_action_count": len(contract["actions"]),
                "source_generation_action_units": len(units),
                "selected_source_generation_unit_indexes": list(
                    range(1, len(units) + 1)
                ),
                "production_generation_action_unit_target": target_count,
                "selected_source_micro_action_indexes": list(
                    selection["selected_source_micro_action_indexes"]
                ),
                "omitted_source_micro_action_indexes": list(
                    selection["omitted_source_micro_action_indexes"]
                ),
                "source_micro_actions_sha256": selection[
                    "source_micro_actions_sha256"
                ],
            })

        production_counts = _event_generation_action_unit_counts(production_events)
        for record in records:
            event_id = record["source_event_id"]
            planned_event = production_events[event_id - 1]
            record["production_micro_action_count"] = len(
                planned_event.get("micro_actions") or []
            )
            record["production_generation_action_units"] = production_counts[
                event_id
            ]
            rewrite_groups = (
                planned_event.get("production_action_rewrite", {}).get(
                    "groups"
                )
                or []
            )
            record["scaling"] = (
                "rewrite"
                if (
                    record["production_generation_action_units"]
                    < record["source_generation_action_units"]
                    or any(
                        len(group.get("source_micro_action_indexes") or []) > 1
                        for group in rewrite_groups
                        if isinstance(group, dict)
                    )
                )
                else "full"
            )
        return production_events, records

    # Search by the fewest omitted normalized units.  Semantic weights only
    # break equal-loss ties, protecting turns and consequences over setup.
    role_weights = {
        "scene_setup": 2,
        "consequence": 6,
        "turning_point": 8,
        "dramatic_turn": 8,
    }

    # The old best-first search enumerated the Cartesian product of every
    # event target.  A long continuous chain could exhaust 50,000 states before
    # reaching a valid low-unit plan even though the final beat capacity was
    # tiny.  Dynamic programming keeps only the best semantic-loss candidate
    # for each observable packing state: occupied beats and trailing load.
    # Its state space is bounded by ``beat_count * per_beat_capacity`` rather
    # than by the number of action-count combinations.
    ScoreAndTargets = tuple[int, int, int, tuple[int, ...]]
    packing_states: dict[tuple[int, int, int], ScoreAndTargets] = {
        (0, 0, 0): (0, 0, 0, ())
    }
    previous_sequence: str | None = None
    for event_id in planned_event_ids:
        event = events[event_id - 1]
        sequence = str(event.get("sequence_id") or "").strip() or "__unspecified__"
        starts_sequence = sequence != previous_sequence
        source_count = source_counts[event_id]
        # Source facts are immutable for current runs.  Capacity pressure may
        # compress a non-empty event into one source-indexed production action,
        # but may not silently erase the event because it was classified as
        # optional by an upstream narrative heuristic.
        minimum_target = 1 if source_count > 0 else 0
        role_weight = role_weights.get(
            str(event.get("event_role") or "").strip().lower(),
            5,
        )
        next_states: dict[tuple[int, int, int], ScoreAndTargets] = {}
        for (
            occupied_beats,
            trailing_load,
            selected_so_far,
        ), state in packing_states.items():
            removed_so_far, weighted_so_far, relative_so_far, targets_so_far = state
            for target in range(source_count, minimum_target - 1, -1):
                candidate_selected = selected_so_far + target
                if (
                    maximum_total_generation_units is not None
                    and candidate_selected > maximum_total_generation_units
                ):
                    continue
                candidate_beats = occupied_beats
                candidate_trailing = trailing_load
                if starts_sequence or candidate_beats == 0:
                    candidate_beats += 1
                    candidate_trailing = 0
                remaining_target = target
                packing_invalid = candidate_beats > beat_count
                while remaining_target and not packing_invalid:
                    room = (
                        per_beat_capacities[candidate_beats - 1]
                        - candidate_trailing
                    )
                    if room <= 0:
                        candidate_beats += 1
                        candidate_trailing = 0
                        if candidate_beats > beat_count:
                            packing_invalid = True
                            break
                        continue
                    assigned = min(remaining_target, room)
                    candidate_trailing += assigned
                    remaining_target -= assigned
                    if remaining_target:
                        candidate_beats += 1
                        candidate_trailing = 0
                        if candidate_beats > beat_count:
                            packing_invalid = True
                            break
                if packing_invalid:
                    continue
                if target == 0 and candidate_beats > beat_count:
                    continue
                removed = source_count - target
                candidate: ScoreAndTargets = (
                    removed_so_far + removed,
                    weighted_so_far + removed * role_weight,
                    relative_so_far
                    + round(10000 * removed / max(source_count, 1)),
                    targets_so_far + (target,),
                )
                key = (
                    candidate_beats,
                    candidate_trailing,
                    candidate_selected,
                )
                incumbent = next_states.get(key)
                if incumbent is None or candidate < incumbent:
                    next_states[key] = candidate
        packing_states = next_states
        previous_sequence = sequence
        if not packing_states:
            break

    selected: tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[str],
        list[dict[str, int]],
    ] | None = None
    if packing_states:
        _removed, _weighted, _relative, targets = min(packing_states.values())
        production_events, records = build_candidate(targets)
        try:
            (
                sequence_plan,
                event_counts_per_beat,
            ) = _vector_sequence_beat_allocation(
                production_events,
                per_beat_capacities,
            )
        except ValueError:
            sequence_plan = []
            event_counts_per_beat = []
        if sequence_plan:
            selected = (
                production_events,
                records,
                sequence_plan,
                event_counts_per_beat,
            )

    if selected is None:
        raise ValueError(
            "mandatory sequence facts cannot fit the available story-beat slots "
            "even after bounded intra-event action scaling"
        )
    (
        production_events,
        records,
        sequence_plan,
        event_counts_per_beat,
    ) = selected
    production_counts = _event_generation_action_unit_counts(production_events)
    source_generation_units = sum(source_counts.values())
    production_generation_units = sum(production_counts.values())
    plan = {
        "schema": DURATION_SCALED_EVENT_PLAN_SCHEMA,
        "target_duration_s": int(target_duration),
        "beat_count": int(beat_count),
        "effective_shot_duration_s": int(effective_shot_duration),
        "max_generation_action_units_per_beat": int(per_beat_capacity),
        "generation_action_unit_capacities_per_beat": list(
            per_beat_capacities
        ),
        "maximum_total_generation_action_units": (
            int(maximum_total_generation_units)
            if maximum_total_generation_units is not None
            else beat_count * per_beat_capacity
        ),
        "source_generation_action_units": source_generation_units,
        "production_generation_action_units": production_generation_units,
        "intra_event_scaling_applied": any(
            record["scaling"] == "rewrite" for record in records
        ),
        "base_mandatory_source_event_ids": sorted(base_mandatory_event_ids),
        "terminal_outcome_source_event_ids": sorted(terminal_event_ids),
        "mandatory_source_event_ids": mandatory_event_ids,
        "causal_predecessor_source_event_ids": sorted(
            set(mandatory_event_ids) - base_mandatory_event_ids
        ),
        "sequence_beat_plan": [
            {"beat_order": index, "sequence_id": sequence}
            for index, sequence in enumerate(sequence_plan, 1)
        ],
        "source_event_generation_unit_counts_per_beat": event_counts_per_beat,
        "events": records,
    }
    return production_events, plan


def _bind_action_timeline_to_primary_layout(
    production_events: List[Dict[str, Any]],
    duration_scaled_event_plan: Dict[str, Any],
    primary_shot_layout: Dict[str, Any],
    capabilities: VideoModelCapabilities,
) -> dict[str, Any]:
    """Bind canonical temporal slices to one exact Sxx/Pxx capacity matrix.

    Duration planning owns how many slices from each event enter each Sxx.
    This function is the single deterministic handoff from that vector ledger
    into Provider-sized Pxx slots; it never recomputes a scalar beat plan.
    """

    def bucket_counts(
        units: list[dict[str, Any]],
        durations: list[float],
        limit: int,
    ) -> tuple[int, ...]:
        """Balance contiguous temporal slices against canonical Pxx time."""
        unit_count = len(units)
        beat_count = len(durations)
        minimum = 1 if unit_count >= beat_count else 0
        candidates: list[tuple[tuple[float, ...], tuple[int, ...]]] = []

        def visit(prefix: tuple[int, ...], remaining: int) -> None:
            position = len(prefix)
            if position == beat_count:
                if remaining:
                    return
                cursor = 0
                weights = []
                for count in prefix:
                    weights.append(sum(
                        int(unit.get("pace_weight") or 1)
                        for unit in units[cursor:cursor + count]
                    ))
                    cursor += count
                total_weight = sum(weights)
                total_duration = sum(durations)
                imbalance = sum(
                    abs(
                        weight
                        - total_weight * float(duration) / total_duration
                    )
                    for weight, duration in zip(weights, durations, strict=True)
                ) if total_duration else float("inf")
                empty_between_content = sum(
                    1
                    for index, count in enumerate(prefix)
                    if count == 0
                    and any(value > 0 for value in prefix[:index])
                    and any(value > 0 for value in prefix[index + 1:])
                )
                leading_empty = 0
                for count in prefix:
                    if count:
                        break
                    leading_empty += 1
                trailing_empty = 0
                for count in reversed(prefix):
                    if count:
                        break
                    trailing_empty += 1
                candidates.append((
                    (
                        float(leading_empty + empty_between_content),
                        round(imbalance, 9),
                        float(trailing_empty),
                        *tuple(float(-count) for count in prefix),
                    ),
                    prefix,
                ))
                return
            slots_left = beat_count - position - 1
            lower = max(minimum, remaining - slots_left * limit)
            upper = min(limit, remaining - slots_left * minimum)
            for count in range(lower, upper + 1):
                visit((*prefix, count), remaining - count)

        visit((), unit_count)
        if not candidates:
            raise ValueError("timeline slices cannot fit the canonical Pxx vector")
        return min(candidates, key=lambda item: item[0])[1]

    if duration_scaled_event_plan.get("schema") != DURATION_SCALED_EVENT_PLAN_SCHEMA:
        raise ValueError("timeline binding requires the current duration plan")
    if primary_shot_layout.get("schema") != PRIMARY_SHOT_LAYOUT_SCHEMA:
        raise ValueError("timeline binding requires the current primary-shot layout")
    event_counts_per_shot = duration_scaled_event_plan.get(
        "source_event_generation_unit_counts_per_beat"
    )
    sequence_beat_plan = duration_scaled_event_plan.get("sequence_beat_plan")
    content_beat_counts = primary_shot_layout.get("content_beat_counts")
    effective_story_durations = primary_shot_layout.get(
        "effective_story_durations_s"
    )
    if (
        not isinstance(event_counts_per_shot, list)
        or not isinstance(sequence_beat_plan, list)
        or not isinstance(content_beat_counts, list)
        or not isinstance(effective_story_durations, list)
        or len(event_counts_per_shot) != len(content_beat_counts)
        or len(sequence_beat_plan) != len(content_beat_counts)
        or len(effective_story_durations) != len(content_beat_counts)
    ):
        raise ValueError("timeline binding ledgers do not match primary shots")
    sequence_ids: list[str] = []
    for shot_index, item in enumerate(sequence_beat_plan, 1):
        if (
            not isinstance(item, dict)
            or item.get("beat_order") != shot_index
            or not str(item.get("sequence_id") or "").strip()
        ):
            raise ValueError("timeline binding has an invalid sequence beat plan")
        sequence_ids.append(str(item["sequence_id"]).strip())

    event_units: dict[int, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for event_id, event in enumerate(production_events, 1):
        normalized = normalize_event_action_units(
            event,
            seen=seen,
            max_motion_contributions_per_slice=(
                capabilities.motion_contribution_limit
            ),
        )
        units = [
            copy.deepcopy(unit)
            for unit in normalized.get("generation_action_units") or []
            if isinstance(unit, dict)
        ]
        for unit_order, unit in enumerate(units, 1):
            unit["source_event_id"] = event_id
            unit["event_temporal_slice_order"] = unit_order
        event_units[event_id] = units

    cursors = {event_id: 0 for event_id in event_units}
    assignments: list[dict[str, Any]] = []
    sxx_records: list[dict[str, Any]] = []
    global_pxx_order = 0
    slice_limit = capabilities.temporal_slice_limit
    motion_limit = capabilities.motion_contribution_limit
    for shot_index, (raw_counts, pxx_count, pxx_durations, sequence_id) in enumerate(
        zip(
            event_counts_per_shot,
            content_beat_counts,
            effective_story_durations,
            sequence_ids,
            strict=True,
        ),
        1,
    ):
        if not isinstance(raw_counts, dict) or (
            isinstance(pxx_count, bool)
            or not isinstance(pxx_count, int)
            or pxx_count < 1
        ):
            raise ValueError("timeline binding contains an invalid Sxx ledger")
        if (
            not isinstance(pxx_durations, list)
            or len(pxx_durations) != pxx_count
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) <= 0
                for value in pxx_durations
            )
        ):
            raise ValueError("timeline binding has invalid Pxx story durations")
        shot_units: list[dict[str, Any]] = []
        for raw_event_id, raw_count in raw_counts.items():
            try:
                event_id = int(raw_event_id)
                count = int(raw_count)
            except (TypeError, ValueError) as exc:
                raise ValueError("timeline binding has invalid event counts") from exc
            if count < 0 or event_id not in event_units:
                raise ValueError("timeline binding references an invalid event")
            start = cursors[event_id]
            end = start + count
            if end > len(event_units[event_id]):
                raise ValueError("timeline binding exceeds an event slice ledger")
            shot_units.extend(event_units[event_id][start:end])
            cursors[event_id] = end
        if len(shot_units) > pxx_count * slice_limit:
            raise ValueError("timeline binding exceeds Sxx temporal capacity")

        pxx_records: list[dict[str, Any]] = []
        temporal_bucket_counts = bucket_counts(
            shot_units,
            [float(value) for value in pxx_durations],
            slice_limit,
        )
        shot_unit_cursor = 0
        for local_pxx_order in range(1, pxx_count + 1):
            global_pxx_order += 1
            temporal_bucket_count = temporal_bucket_counts[
                local_pxx_order - 1
            ]
            pxx_units = shot_units[
                shot_unit_cursor:shot_unit_cursor + temporal_bucket_count
            ]
            shot_unit_cursor += temporal_bucket_count
            pxx_id = f"P{global_pxx_order:03d}"
            pxx_assignment_ids: list[str] = []
            for unit in pxx_units:
                motion_load = int(unit.get("motion_load") or 1)
                if motion_load > motion_limit:
                    raise ValueError(
                        "timeline slice exceeds Provider motion contribution capacity"
                    )
                assignment_id = f"TA{len(assignments) + 1:03d}"
                pxx_assignment_ids.append(assignment_id)
                assignments.append({
                    "assignment_id": assignment_id,
                    "source_event_id": int(unit["source_event_id"]),
                    "event_temporal_slice_order": int(
                        unit["event_temporal_slice_order"]
                    ),
                    "source_temporal_slice_id": str(
                        unit.get("temporal_slice_id") or ""
                    ),
                    "semantic_temporal_slice_id": str(
                        unit.get("semantic_temporal_slice_id")
                        or unit.get("temporal_slice_id")
                        or ""
                    ),
                    "semantic_motion_load": int(
                        unit.get("semantic_motion_load")
                        or unit.get("motion_load")
                        or 1
                    ),
                    "execution_subslice_id": str(
                        unit.get("execution_subslice_id") or ""
                    ),
                    "execution_subslice_order": int(
                        unit.get("execution_subslice_order") or 1
                    ),
                    "execution_subslice_count": int(
                        unit.get("execution_subslice_count") or 1
                    ),
                    "provider_capacity_staging": str(
                        unit.get("provider_capacity_staging")
                        or "not_required"
                    ),
                    "source_micro_action_indexes": [
                        int(index)
                        for index in (
                            unit.get("source_micro_action_indexes")
                            or [
                                int(value) + 1
                                for value in unit.get("ledger_indexes") or []
                            ]
                        )
                    ],
                    "contribution_micro_action_indexes": [
                        int(index) + 1
                        for index in unit.get(
                            "contribution_ledger_indexes"
                        ) or unit.get("ledger_indexes") or []
                    ],
                    "effect_micro_action_indexes": [
                        int(index) + 1
                        for index in unit.get("effect_ledger_indexes") or []
                    ],
                    "sustained_micro_action_indexes": [
                        int(index) + 1
                        for index in unit.get("sustained_ledger_indexes") or []
                    ],
                    "motion_load": motion_load,
                    "pace_weight": int(unit.get("pace_weight") or 1),
                    "performers": list(unit.get("performers") or []),
                    "targets": list(unit.get("targets") or []),
                    "state_reads": list(unit.get("state_reads") or []),
                    "state_writes": list(unit.get("state_writes") or []),
                    "start_state": str(unit.get("start_state") or ""),
                    "end_state": str(unit.get("end_state") or ""),
                    "source_fact_echoes": list(
                        unit.get("source_fact_echoes") or unit.get("actions") or []
                    ),
                    "source_generation_unit_indexes": list(
                        unit.get("source_generation_unit_indexes") or []
                    ),
                    "source_actions_sha256": str(
                        unit.get("source_actions_sha256") or ""
                    ),
                    "screenplay_rewrite_schema": (
                        SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA
                        if unit.get("source_fact_echoes")
                        else None
                    ),
                    "sxx_id": f"S{shot_index:03d}",
                    "pxx_id": pxx_id,
                })
            pxx_records.append({
                "pxx_id": pxx_id,
                "pxx_order_within_sxx": local_pxx_order,
                "temporal_slice_capacity": slice_limit,
                "effective_story_duration_s": float(
                    pxx_durations[local_pxx_order - 1]
                ),
                "assigned_pace_weight": sum(
                    int(unit.get("pace_weight") or 1) for unit in pxx_units
                ),
                "assignment_ids": pxx_assignment_ids,
            })
        sxx_records.append({
            "sxx_id": f"S{shot_index:03d}",
            "sxx_order": shot_index,
            "sequence_id": sequence_id,
            "temporal_slice_capacity": pxx_count * slice_limit,
            "assigned_temporal_slice_count": len(shot_units),
            "pxx": pxx_records,
        })

    unassigned = {
        event_id: len(units) - cursors[event_id]
        for event_id, units in event_units.items()
        if len(units) != cursors[event_id]
    }
    if unassigned:
        raise ValueError(
            f"timeline binding left production slices unassigned: {unassigned}"
        )

    positive_positions: dict[int, list[int]] = {}
    for shot_index, raw_counts in enumerate(event_counts_per_shot, 1):
        for raw_event_id, raw_count in raw_counts.items():
            event_id = int(raw_event_id)
            if int(raw_count) > 0:
                positive_positions.setdefault(event_id, []).append(shot_index)
    zero_story_time_attachments: list[dict[str, Any]] = []
    zero_event_ids_by_sequence: dict[str, list[int]] = {}
    for event_id, units in event_units.items():
        if units:
            continue
        event_sequence = (
            str(production_events[event_id - 1].get("sequence_id") or "").strip()
            or "__unspecified__"
        )
        zero_event_ids_by_sequence.setdefault(event_sequence, []).append(event_id)
    for event_id, units in event_units.items():
        if units:
            continue
        event_sequence = (
            str(production_events[event_id - 1].get("sequence_id") or "").strip()
            or "__unspecified__"
        )
        candidate_shots = [
            shot_index
            for shot_index, sequence_id in enumerate(sequence_ids, 1)
            if sequence_id == event_sequence
        ]
        if not candidate_shots:
            raise ValueError(
                "zero-story-time source fact has no Sxx in its authored "
                f"sequence: event={event_id}, sequence={event_sequence}"
            )
        next_event_id = next(
            (
                candidate_id
                for candidate_id in range(event_id + 1, len(production_events) + 1)
                if positive_positions.get(candidate_id)
                and (
                    str(
                        production_events[candidate_id - 1].get("sequence_id")
                        or ""
                    ).strip()
                    or "__unspecified__"
                ) == event_sequence
            ),
            None,
        )
        previous_event_id = next(
            (
                candidate_id
                for candidate_id in range(event_id - 1, 0, -1)
                if positive_positions.get(candidate_id)
                and (
                    str(
                        production_events[candidate_id - 1].get("sequence_id")
                        or ""
                    ).strip()
                    or "__unspecified__"
                ) == event_sequence
            ),
            None,
        )
        if next_event_id is not None:
            attached_shot = positive_positions[next_event_id][0]
            anchor_event_id = next_event_id
            attachment_rule = "nearest_next_execution_event"
        elif previous_event_id is not None:
            attached_shot = positive_positions[previous_event_id][-1]
            anchor_event_id = previous_event_id
            attachment_rule = "nearest_previous_execution_event"
        else:
            sequence_zero_events = zero_event_ids_by_sequence[event_sequence]
            zero_event_offset = sequence_zero_events.index(event_id)
            attached_shot = candidate_shots[
                min(
                    len(candidate_shots) - 1,
                    (
                        zero_event_offset * len(candidate_shots)
                        // len(sequence_zero_events)
                    ),
                )
            ]
            anchor_event_id = None
            attachment_rule = "sequence_source_order_distribution"
        sxx_records[attached_shot - 1].setdefault(
            "zero_story_time_source_event_ids", []
        ).append(event_id)
        zero_story_time_attachments.append({
            "source_event_id": event_id,
            "sequence_id": event_sequence,
            "sxx_id": f"S{attached_shot:03d}",
            "anchor_source_event_id": anchor_event_id,
            "attachment_rule": attachment_rule,
            "consumes_temporal_capacity": False,
        })
    return {
        "schema": ACTION_TIMELINE_SCHEMA,
        "layout_schema": PRIMARY_SHOT_LAYOUT_SCHEMA,
        "duration_plan_schema": DURATION_SCALED_EVENT_PLAN_SCHEMA,
        "max_temporal_slices_per_content_beat": slice_limit,
        "max_motion_contributions_per_slice": motion_limit,
        "sxx": sxx_records,
        "assignments": assignments,
        "zero_story_time_attachments": zero_story_time_attachments,
    }


def _validate_source_indexed_rewrite_observation(
    item: dict[str, Any],
    expected_groups: dict[int, list[dict[str, Any]]],
) -> str:
    event_id = item["source_event_id"]
    groups = expected_groups.get(event_id)
    if groups is None:
        raise ValueError(
            "source-indexed rewrite event coverage/order mismatch; "
            f"unexpected source_event_id={event_id}"
        )
    actions = item["production_actions"]
    if len(actions) != len(groups):
        raise ValueError(
            f"source-indexed rewrite event {event_id} changed group count"
        )
    structural_actions: list[dict[str, Any]] = []
    for action, group in zip(actions, groups, strict=True):
        action_index = action["production_action_index"]
        source_indexes = action["source_micro_action_indexes"]
        if (
            action_index != group["production_action_index"]
            or source_indexes != group["source_micro_action_indexes"]
            or not str(action["rewritten_micro_action"]).strip()
        ):
            raise ValueError(
                f"source-indexed rewrite event {event_id} changed lineage"
            )
        structural_actions.append({
            "production_action_index": action_index,
            "source_micro_action_indexes": source_indexes,
        })
    for field in (
        "narrative_purpose",
        "emotional_beat",
        "director_alignment",
    ):
        if not str(item.get(field) or "").strip():
            raise ValueError(
                f"source-indexed rewrite event {event_id} has empty {field}"
            )
    return _canonical_json_sha256({
        "source_event_id": event_id,
        "production_actions": structural_actions,
    })


def _reconcile_source_indexed_rewrite_observations(
    returned: list[dict[str, Any]],
    expected_ids: list[int],
    expected_groups: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse only adjacent duplicates with identical authority lineage."""
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("source-indexed rewrite expected ids are not unique")
    original_ids = [item["source_event_id"] for item in returned]
    reconciled: list[dict[str, Any]] = []
    reconciled_signatures: list[str] = []
    reconciled_positions: list[int] = []
    duplicates: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for position, item in enumerate(returned, start=1):
        event_id = item["source_event_id"]
        signature = _validate_source_indexed_rewrite_observation(
            item,
            expected_groups,
        )
        if reconciled and reconciled[-1]["source_event_id"] == event_id:
            if reconciled_signatures[-1] != signature:
                raise ValueError(
                    "source-indexed rewrite adjacent duplicate changed lineage; "
                    f"source_event_id={event_id}"
                )
            duplicates.append({
                "source_event_id": event_id,
                "retained_position": reconciled_positions[-1],
                "dropped_position": position,
                "structural_signature_sha256": signature,
                "retained_observation_sha256": _canonical_json_sha256(
                    reconciled[-1]
                ),
                "dropped_observation_sha256": _canonical_json_sha256(item),
            })
            continue
        if event_id in seen_ids:
            raise ValueError(
                "source-indexed rewrite event coverage/order mismatch; "
                f"non-adjacent duplicate source_event_id={event_id}"
            )
        seen_ids.add(event_id)
        reconciled.append(item)
        reconciled_signatures.append(signature)
        reconciled_positions.append(position)
    reconciled_ids = [item["source_event_id"] for item in reconciled]
    if reconciled_ids != expected_ids:
        raise ValueError(
            "source-indexed rewrite event coverage/order mismatch; "
            f"expected={expected_ids}, actual={original_ids}, "
            f"reconciled={reconciled_ids}"
        )
    receipt = {
        "schema": SOURCE_INDEXED_REWRITE_RECONCILIATION_SCHEMA,
        "policy": SOURCE_INDEXED_REWRITE_RECONCILIATION_POLICY,
        "policy_sha256": (
            SOURCE_INDEXED_REWRITE_RECONCILIATION_POLICY_SHA256
        ),
        "expected_source_event_ids": expected_ids,
        "original_source_event_ids": original_ids,
        "reconciled_source_event_ids": reconciled_ids,
        "original_event_count": len(returned),
        "reconciled_event_count": len(reconciled),
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "source_fact_loss_count": 0,
        "provider_request_count": 0,
    }
    return reconciled, receipt


def _apply_source_indexed_screenplay_rewrite(
    source_events: List[Dict[str, Any]],
    production_events: List[Dict[str, Any]],
    duration_scaled_event_plan: Dict[str, Any],
    director_plan: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Perform one real, fixed-lineage screenplay compression request.

    The deterministic solver owns event counts and contiguous source groups.
    The model may only rewrite each fixed group into one richer production
    action.  Exact source actions remain in the rewrite ledger and later Pxx
    prompts, so prose quality can change without changing facts or capacity.
    """
    plan = copy.deepcopy(duration_scaled_event_plan)
    records = plan.get("events")
    if not isinstance(records, list) or len(records) != len(source_events):
        raise ValueError("source-indexed rewrite requires aligned event records")
    rewrite_records = [
        record for record in records if record.get("scaling") == "rewrite"
    ]
    if not rewrite_records:
        plan["semantic_selection_status"] = "not_required"
        return copy.deepcopy(production_events), plan

    intents = _director_intents_by_sequence(director_plan, source_events)
    inputs: list[dict[str, Any]] = []
    expected_groups: dict[int, list[dict[str, Any]]] = {}
    for record in rewrite_records:
        event_id = record.get("source_event_id")
        if not isinstance(event_id, int) or not 1 <= event_id <= len(source_events):
            raise ValueError("source-indexed rewrite has an invalid event id")
        production_event = production_events[event_id - 1]
        rewrite = production_event.get("production_action_rewrite")
        if (
            not isinstance(rewrite, dict)
            or rewrite.get("schema") != SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA
            or rewrite.get("omitted_source_micro_action_indexes") != []
        ):
            raise ValueError("source-indexed rewrite ledger is missing or invalid")
        groups = rewrite.get("groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("source-indexed rewrite requires fixed groups")
        expected_groups[event_id] = copy.deepcopy(groups)
        source_event = source_events[event_id - 1]
        sequence_id = (
            str(source_event.get("sequence_id") or "").strip()
            or "__unspecified__"
        )
        inputs.append({
            "source_event_id": event_id,
            "sequence_id": sequence_id,
            "event_role": source_event.get("event_role"),
            "what": source_event.get("what"),
            "start_state": source_event.get("start_state"),
            "end_state": source_event.get("end_state"),
            "causal_link": source_event.get("causal_link"),
            "fixed_production_actions": [
                {
                    "production_action_index": group[
                        "production_action_index"
                    ],
                    "source_micro_action_indexes": group[
                        "source_micro_action_indexes"
                    ],
                    "source_actions": group["source_actions"],
                    "performers": group.get("performers") or [],
                    "targets": group.get("targets") or [],
                    "start_state": group.get("start_state") or "",
                    "end_state": group.get("end_state") or "",
                }
                for group in groups
            ],
            "director_intent": intents[sequence_id],
        })

    system_prompt = (
        "你是影视编剧，执行一次来源索引受约束的时长压缩重写。代码已经固定事件、"
        "生产动作数量、每个生产动作覆盖的来源动作索引和顺序。你只能把同组来源动作写成"
        "一个具体、连续、有重量和因果承接的生产动作描述；不得删减、调序、跨组移动、"
        "新增剧情、改变人物/道具/胜负/终态，也不得输出镜头字段。输出严格 JSON。"
    )
    base_prompt = (
        "请重写下列事件。必须逐事件、逐 production_action_index 完整返回；"
        "source_micro_action_indexes 必须与输入逐项完全一致。rewritten_micro_action 要按"
        "source_actions 原顺序写出全部事实，明确执行者、目标、起止状态、力度、重心与惯性；"
        "并行攻击/反应/效果写在同一故事时刻，串行动作保持先后。每组只能得到一个连续动作段。\n\n"
        f"输入：\n{json.dumps(inputs, ensure_ascii=False, indent=2)}"
    )
    stream_policy = LLMStreamPolicy.adaptation_structured_output(
        max_tokens=12000,
    )
    client = create_ark_client(
        read_timeout=stream_policy.transport_read_timeout_seconds,
    )
    parsed: dict[str, Any] | None = None
    correction = ""
    expected_ids = [record["source_event_id"] for record in rewrite_records]
    retry_limit = effective_provider_retries(MAX_RETRIES)
    for attempt in range(retry_limit + 1):
        content = call_llm_stream(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": base_prompt + correction},
            ],
            max_tokens=stream_policy.max_tokens,
            wall_timeout=stream_policy.wall_timeout_seconds,
            read_timeout=stream_policy.transport_read_timeout_seconds,
            idle_timeout=stream_policy.idle_timeout_seconds,
            response_format=native_chat_json_schema_format(
                SourceIndexedScreenplayRewriteBatch
            ),
            _client=client,
        )
        try:
            candidate = parse_structured_output(
                content,
                SourceIndexedScreenplayRewriteBatch,
            ).model_dump(by_alias=True)
            returned = candidate["events"]
            returned, reconciliation = (
                _reconcile_source_indexed_rewrite_observations(
                    returned,
                    expected_ids,
                    expected_groups,
                )
            )
            candidate["events"] = returned
            plan["source_indexed_rewrite_reconciliation"] = reconciliation
            parsed = candidate
            break
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            if attempt >= retry_limit:
                raise ValueError(
                    f"source-indexed screenplay rewrite failed validation: {exc}"
                ) from exc
            correction = (
                "\n\n上次输出未通过固定来源索引校验："
                f"{exc}。请完整重写 JSON，严禁改变事件、组数或来源索引。"
            )
    if parsed is None:
        raise ValueError("source-indexed screenplay rewrite returned no valid plan")

    rewritten_events = copy.deepcopy(production_events)
    semantic_by_event = {
        item["source_event_id"]: item for item in parsed["events"]
    }
    records_by_event = {
        record["source_event_id"]: record for record in records
    }
    for event_id, item in semantic_by_event.items():
        event = rewritten_events[event_id - 1]
        rewrite = event["production_action_rewrite"]
        rewritten_actions = [
            str(action["rewritten_micro_action"]).strip()
            for action in item["production_actions"]
        ]
        event["micro_actions"] = rewritten_actions
        for group, action in zip(
            rewrite["groups"],
            item["production_actions"],
            strict=True,
        ):
            group["rewritten_micro_action"] = str(
                action["rewritten_micro_action"]
            ).strip()
        selection = event.get("production_action_selection")
        if isinstance(selection, dict):
            selection["production_action_groups"] = copy.deepcopy(
                rewrite["groups"]
            )
        record = records_by_event[event_id]
        for field in (
            "narrative_purpose",
            "emotional_beat",
            "director_alignment",
        ):
            record[field] = item[field]
        record["screenplay_rewrite_attempt"] = 1

    production_counts = _event_generation_action_unit_counts(rewritten_events)
    for record in records:
        event_id = record["source_event_id"]
        target = int(record.get("production_generation_action_unit_target") or 0)
        if production_counts[event_id] != target:
            raise ValueError(
                f"source-indexed rewrite changed event {event_id} capacity from "
                f"{target} to {production_counts[event_id]}"
            )
        record["production_generation_action_units"] = production_counts[event_id]

    per_beat_capacities = [
        int(value)
        for value in plan["generation_action_unit_capacities_per_beat"]
    ]
    sequence_plan, event_counts_per_beat = _vector_sequence_beat_allocation(
        rewritten_events,
        per_beat_capacities,
    )
    if sequence_plan != [
        item["sequence_id"] for item in plan["sequence_beat_plan"]
    ]:
        raise ValueError("source-indexed rewrite changed sequence ownership")
    plan["source_event_generation_unit_counts_per_beat"] = (
        event_counts_per_beat
    )
    plan["production_generation_action_units"] = sum(
        production_counts.values()
    )
    plan["semantic_selection_schema"] = (
        SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA
    )
    plan["semantic_selection_status"] = "source_indexed_rewrite"
    plan["screenplay_rewrite_attempt"] = 1
    plan["director_plan_schema"] = director_plan.get("schema")
    return rewritten_events, plan


def _apply_director_action_selection(
    source_events: List[Dict[str, Any]],
    production_events: List[Dict[str, Any]],
    duration_scaled_event_plan: Dict[str, Any],
    director_plan: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Let Director intent choose *which* bounded source units survive.

    Deterministic capacity planning has already fixed the number of units per
    mandatory event.  The model may only choose 1-based indexes from the
    source action-unit ledger and explain their purpose/emotional alignment;
    it cannot change counts, text, event identity, sequence, or timing.
    """
    if duration_scaled_event_plan.get("schema") != (
        DURATION_SCALED_EVENT_PLAN_SCHEMA
    ):
        raise ValueError("unsupported duration-scaled event plan schema")
    if len(source_events) != len(production_events):
        raise ValueError("semantic action selection requires aligned event ledgers")
    plan = copy.deepcopy(duration_scaled_event_plan)
    records = plan.get("events")
    if not isinstance(records, list) or len(records) != len(source_events):
        raise ValueError("duration-scaled event plan has incomplete event records")
    if any(record.get("scaling") == "rewrite" for record in records):
        if any(record.get("scaling") == "representative" for record in records):
            raise ValueError("mixed duration rewrite generations are unsupported")
        return _apply_source_indexed_screenplay_rewrite(
            source_events,
            production_events,
            duration_scaled_event_plan,
            director_plan,
        )
    scaled_records = [
        record
        for record in records
        if record.get("mandatory") and record.get("scaling") == "representative"
    ]
    if not scaled_records:
        plan["semantic_selection_status"] = "not_required"
        return copy.deepcopy(production_events), plan

    contracts = _source_event_generation_contracts(source_events)
    intents = _director_intents_by_sequence(director_plan, source_events)
    selection_inputs = []
    for record in scaled_records:
        event_id = record.get("source_event_id")
        if not isinstance(event_id, int) or not 1 <= event_id <= len(source_events):
            raise ValueError("duration-scaled event record has invalid source id")
        source_event = source_events[event_id - 1]
        contract = contracts[event_id - 1]
        sequence_id = (
            str(source_event.get("sequence_id") or "").strip()
            or "__unspecified__"
        )
        target_count = len(
            record.get("selected_source_generation_unit_indexes") or []
        )
        selection_inputs.append({
            "source_event_id": event_id,
            "sequence_id": sequence_id,
            "event_role": source_event.get("event_role"),
            "what": source_event.get("what"),
            "start_state": source_event.get("start_state"),
            "end_state": source_event.get("end_state"),
            "causal_link": source_event.get("causal_link"),
            "target_generation_action_unit_count": target_count,
            "source_generation_action_units": [
                {
                    "source_generation_unit_index": unit_index,
                    "source_micro_action_indexes": [
                        int(index) + 1
                        for index in unit.get("ledger_indexes", [])
                    ],
                    "actions": list(unit.get("actions") or []),
                }
                for unit_index, unit in enumerate(
                    contract["generation_units"],
                    1,
                )
            ],
            "director_intent": intents[sequence_id],
        })

    system_prompt = (
        "你是时长受限的影视编剧，与导演意图协作。代码已经决定每个来源事件能保留多少个动作单元。"
        "你只选择来源 generation unit 的索引，不生成镜头字段，不改写动作，不新增动作，不改变事件顺序。"
        "选择必须保留事件因果结果，并以 scene_goal、emotion_arc、visual_focus、spatial_intent、"
        "transition_intent 解释叙事目的与情绪节拍。输出严格 JSON。"
    )
    base_prompt = (
        "请为下列需要缩放的强制事件选择生产动作索引。每个事件必须返回且仅返回"
        " target_generation_action_unit_count 个互不重复、严格递增、范围合法的索引。"
        "narrative_purpose 说明该事件为何必须被观众理解；emotional_beat 说明保留动作承载的情绪变化；"
        "director_alignment 说明它如何服务既有五项导演意图。\n\n"
        f"输入：\n{json.dumps(selection_inputs, ensure_ascii=False, indent=2)}"
    )
    stream_policy = LLMStreamPolicy.adaptation_structured_output(
        max_tokens=8000,
    )
    client = create_ark_client(
        read_timeout=stream_policy.transport_read_timeout_seconds,
    )
    parsed: dict[str, Any] | None = None
    correction = ""
    expected_ids = [record["source_event_id"] for record in scaled_records]
    records_by_id = {
        record["source_event_id"]: record for record in scaled_records
    }
    retry_limit = effective_provider_retries(MAX_RETRIES)
    for attempt in range(retry_limit + 1):
        content = call_llm_stream(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": base_prompt + correction},
            ],
            max_tokens=stream_policy.max_tokens,
            wall_timeout=stream_policy.wall_timeout_seconds,
            read_timeout=stream_policy.transport_read_timeout_seconds,
            idle_timeout=stream_policy.idle_timeout_seconds,
            response_format=native_chat_json_schema_format(
                DurationScaledActionSelectionBatch
            ),
            _client=client,
        )
        try:
            candidate = parse_structured_output(
                content,
                DurationScaledActionSelectionBatch,
            ).model_dump(by_alias=True)
            returned = candidate["events"]
            returned_ids = [item["source_event_id"] for item in returned]
            if returned_ids != expected_ids:
                raise ValueError(
                    "duration action selection event coverage/order mismatch; "
                    f"expected={expected_ids}, actual={returned_ids}"
                )
            for item in returned:
                event_id = item["source_event_id"]
                selected_indexes = item[
                    "selected_source_generation_unit_indexes"
                ]
                expected_count = len(
                    records_by_id[event_id].get(
                        "selected_source_generation_unit_indexes"
                    ) or []
                )
                source_unit_count = len(
                    contracts[event_id - 1]["generation_units"]
                )
                if (
                    len(selected_indexes) != expected_count
                    or selected_indexes != sorted(set(selected_indexes))
                    or any(
                        not 1 <= index <= source_unit_count
                        for index in selected_indexes
                    )
                ):
                    raise ValueError(
                        f"duration action selection for event {event_id} must "
                        f"contain exactly {expected_count} ordered source indexes"
                    )
                for field in (
                    "narrative_purpose",
                    "emotional_beat",
                    "director_alignment",
                ):
                    if not str(item.get(field) or "").strip():
                        raise ValueError(
                            f"duration action selection event {event_id} has empty {field}"
                        )
            parsed = candidate
            break
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            if attempt >= retry_limit:
                raise ValueError(
                    f"director-aligned action selection failed validation: {exc}"
                ) from exc
            correction = (
                "\n\n上次输出未通过结构或来源索引校验："
                f"{exc}。请重新输出完整 JSON；不得改变事件、目标数量或来源动作。"
            )
    if parsed is None:
        raise ValueError("director-aligned action selection returned no valid plan")

    semantic_by_event = {
        item["source_event_id"]: item for item in parsed["events"]
    }
    selected_units_by_event = {
        record["source_event_id"]: [
            int(index) - 1
            for index in record.get(
                "selected_source_generation_unit_indexes"
            ) or []
        ]
        for record in records
    }
    for event_id, item in semantic_by_event.items():
        selected_units_by_event[event_id] = [
            int(index) - 1
            for index in item["selected_source_generation_unit_indexes"]
        ]

    selected_production_events: list[dict[str, Any]] = []
    for contract, source_event, record in zip(
        contracts,
        source_events,
        records,
        strict=True,
    ):
        event_id = contract["event_id"]
        selected_unit_indexes = selected_units_by_event[event_id]
        production_event, selection = _materialize_production_event(
            source_event,
            contract,
            selected_unit_indexes,
            preserve_duplicate_actions=False,
        )
        semantic = semantic_by_event.get(event_id)
        if semantic is not None:
            for field in (
                "narrative_purpose",
                "emotional_beat",
                "director_alignment",
            ):
                selection[field] = semantic[field]
                record[field] = semantic[field]
            record["selected_source_generation_unit_indexes"] = list(
                semantic["selected_source_generation_unit_indexes"]
            )
        record["selected_source_micro_action_indexes"] = list(
            selection["selected_source_micro_action_indexes"]
        )
        record["omitted_source_micro_action_indexes"] = list(
            selection["omitted_source_micro_action_indexes"]
        )
        record["production_micro_action_count"] = len(
            production_event["micro_actions"]
        )
        production_event["production_action_selection"] = selection
        selected_production_events.append(production_event)

    production_counts = _event_generation_action_unit_counts(
        selected_production_events
    )
    for record in records:
        event_id = record["source_event_id"]
        selected_count = len(
            record.get("selected_source_generation_unit_indexes") or []
        )
        if production_counts[event_id] != selected_count:
            raise ValueError(
                f"director selection changed event {event_id} capacity from "
                f"{selected_count} to {production_counts[event_id]}"
            )
        record["production_generation_action_units"] = production_counts[
            event_id
        ]

    per_beat_capacities = [
        int(value)
        for value in plan["generation_action_unit_capacities_per_beat"]
    ]
    sequence_plan, event_counts_per_beat = _vector_sequence_beat_allocation(
        selected_production_events,
        per_beat_capacities,
    )
    if sequence_plan != [
        item["sequence_id"] for item in plan["sequence_beat_plan"]
    ]:
        raise ValueError("director action selection changed sequence beat ownership")
    plan["source_event_generation_unit_counts_per_beat"] = (
        event_counts_per_beat
    )
    plan["production_generation_action_units"] = sum(
        production_counts.values()
    )
    plan["semantic_selection_schema"] = (
        DURATION_SCALED_ACTION_SELECTION_SCHEMA
    )
    plan["semantic_selection_status"] = "director_aligned"
    plan["director_plan_schema"] = director_plan.get("schema")
    return selected_production_events, plan


def _repair_bounded_single_sequence_order(
    beats: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    capabilities: VideoModelCapabilities,
    *,
    unit_capacity: int,
    material_duration: int | None,
    generation_unit_capacities_per_beat: list[int] | None = None,
    max_content_beats_per_primary_shot: int = (
        MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    ),
) -> List[Dict[str, Any]] | None:
    """Find an exact chronological layout for a bounded single sequence.

    This path is intentionally bounded.  It handles the common dense one-take
    case exactly, while larger or multi-sequence ledgers continue through the
    general deterministic repair and still face the same fail-closed order
    gate.
    """
    if material_duration is None or not beats or not events:
        return None
    sequences = {
        str(event.get("sequence_id") or "").strip() or "__unspecified__"
        for event in events
    }
    if len(sequences) != 1:
        return None
    generation_units = _event_generation_action_unit_counts(events)
    if max(generation_units.values(), default=0) > unit_capacity:
        return None
    mandatory_ids = _mandatory_adaptation_event_ids(events)
    protected_zero_ids = {
        event_id
        for event_id, units in generation_units.items()
        if units == 0
    }
    optional_ids = [
        event_id
        for event_id in generation_units
        if event_id not in mandatory_ids | protected_zero_ids
    ]
    if len(optional_ids) > 18:
        return None

    model_dropped = _dropped_source_event_ids(beats)
    drop_candidates = []
    for drop_count in range(len(optional_ids) + 1):
        for dropped in itertools.combinations(optional_ids, drop_count):
            drop_candidates.append((
                sum(generation_units[event_id] for event_id in dropped),
                drop_count,
                sum(event_id not in model_dropped for event_id in dropped),
                dropped,
            ))
    drop_candidates.sort()
    occurrence_requirements = _event_primary_occurrence_requirements(
        events,
        capabilities,
        max_content_beats_per_primary_shot=(
            max_content_beats_per_primary_shot
        ),
    )
    beat_count = len(beats)
    explicit_capacity_ledger = generation_unit_capacities_per_beat is not None
    beat_capacities = (
        list(generation_unit_capacities_per_beat)
        if generation_unit_capacities_per_beat is not None
        else [unit_capacity] * beat_count
    )
    if len(beat_capacities) != beat_count:
        raise ValueError("generation-unit capacity ledger must match repair beats")
    all_occupied = (1 << beat_count) - 1

    def duration_cost(loads: tuple[int, ...]) -> int:
        return sum(
            _minimum_primary_duration_for_units(
                load,
                capabilities,
                max_content_beats=max_content_beats_per_primary_shot,
            )
            for load in loads
        )

    def find_placements(
        kept_event_ids: tuple[int, ...],
    ) -> Dict[int, tuple[int, ...]] | None:
        @functools.lru_cache(maxsize=None)
        def search(
            event_offset: int,
            previous_end: int,
            loads: tuple[int, ...],
            occupied: int,
        ) -> tuple[tuple[int, ...], ...] | None:
            if event_offset == len(kept_event_ids):
                if occupied != all_occupied or duration_cost(loads) > material_duration:
                    return None
                return ()
            event_id = kept_event_ids[event_offset]
            units = generation_units[event_id]
            minimum = occurrence_requirements[event_id]
            maximum = min(beat_count, minimum + (1 if units else 0))
            for count in range(minimum, maximum + 1):
                for start in range(previous_end, beat_count - count + 1):
                    positions = tuple(range(start, start + count))
                    next_loads = list(loads)
                    next_occupied = occupied
                    if explicit_capacity_ledger:
                        remaining_units = units
                        additions = []
                        for position in positions:
                            assigned = min(
                                remaining_units,
                                max(
                                    0,
                                    beat_capacities[position]
                                    - next_loads[position],
                                ),
                            )
                            additions.append(assigned)
                            remaining_units -= assigned
                    else:
                        base, remainder = divmod(units, count)
                        additions = [
                            base + (1 if occurrence < remainder else 0)
                            for occurrence in range(count)
                        ]
                        remaining_units = 0
                    for position, assigned in zip(
                        positions,
                        additions,
                        strict=True,
                    ):
                        next_loads[position] += assigned
                        next_occupied |= 1 << position
                    if remaining_units or any(
                        next_loads[position] > beat_capacities[position]
                        for position in positions
                    ):
                        continue
                    serialized_loads = tuple(next_loads)
                    if duration_cost(serialized_loads) > material_duration:
                        continue
                    suffix = search(
                        event_offset + 1,
                        positions[-1],
                        serialized_loads,
                        next_occupied,
                    )
                    if suffix is not None:
                        return (positions, *suffix)
            return None

        result = search(0, 0, (0,) * beat_count, 0)
        if result is None:
            return None
        return dict(zip(kept_event_ids, result, strict=True))

    selected_placements: Dict[int, tuple[int, ...]] | None = None
    selected_dropped: set[int] = set()
    all_event_ids = set(range(1, len(events) + 1))
    for _units, _count, _model_penalty, dropped in drop_candidates:
        kept_set = all_event_ids - set(dropped)
        if _continuous_predecessor_event_ids(events, kept_set) != kept_set:
            continue
        kept = tuple(sorted(kept_set))
        selected_placements = find_placements(kept)
        if selected_placements is not None:
            selected_dropped = set(dropped)
            break
    if selected_placements is None:
        return None

    repaired = [dict(beat) for beat in beats]
    event_by_id = {event_id: event for event_id, event in enumerate(events, 1)}
    sequence = next(iter(sequences))
    selected_allocations: dict[int, dict[int, int]] = {}
    selected_loads = [0] * beat_count
    for event_id, positions in selected_placements.items():
        remaining_units = generation_units[event_id]
        selected_allocations[event_id] = {}
        if explicit_capacity_ledger:
            additions = []
            for position in positions:
                assigned = min(
                    remaining_units,
                    max(
                        0,
                        beat_capacities[position] - selected_loads[position],
                    ),
                )
                additions.append(assigned)
                remaining_units -= assigned
        else:
            base, remainder = divmod(remaining_units, len(positions))
            additions = [
                base + (1 if occurrence < remainder else 0)
                for occurrence in range(len(positions))
            ]
            remaining_units = 0
        for position, assigned in zip(positions, additions, strict=True):
            selected_allocations[event_id][position] = assigned
            selected_loads[position] += assigned
        if remaining_units:
            raise ValueError("selected event placement lost generation units")
    for beat_index, beat in enumerate(repaired):
        source_events = [
            event_id
            for event_id, positions in selected_placements.items()
            if beat_index in positions
        ]
        beat["source_events"] = source_events
        beat["dropped_source_events"] = (
            sorted(selected_dropped) if beat_index == 0 else []
        )
        beat["sequence_id"] = sequence
        beat["action"] = "merge" if len(source_events) > 1 else "keep"
        beat["source_event_generation_unit_counts"] = {
            str(event_id): selected_allocations[event_id][beat_index]
            for event_id in source_events
        }
        details = [event_by_id[event_id] for event_id in source_events]
        beat["capacity_repair"] = {
            "reason": "ordered_single_sequence_story_capacity_rebalanced",
            "sequence_id": sequence,
            "max_generation_units": unit_capacity,
        }
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
        if locations:
            beat["where"] = locations[0] if len(locations) == 1 else " / ".join(locations)
        descriptions = [
            str(detail.get("what") or "").strip()
            for detail in details
            if str(detail.get("what") or "").strip()
        ]
        if descriptions:
            beat["what"] = "；随后".join(descriptions)
        beat["reason"] = "；".join(filter(None, [
            str(beat.get("reason") or "").strip(),
            "代码按单一 sequence、事件顺序与故事时钟容量重建账本",
        ]))

    _validate_beat_action_capacity(
        repaired,
        events,
        capabilities,
        max_content_beats_per_primary_shot=(
            max_content_beats_per_primary_shot
        ),
    )
    _validate_beat_material_duration(
        repaired,
        events,
        material_duration,
        capabilities,
    )
    return repaired


def _repair_beat_action_capacity(
    beats: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    capabilities: VideoModelCapabilities | None = None,
    *,
    max_generation_units_per_beat: int | None = None,
    material_duration: int | None = None,
    generation_unit_capacities_per_beat: list[int] | None = None,
) -> List[Dict[str, Any]]:
    """Repair the source ledger without crossing sequences or story capacity.

    The model still owns shot language and editorial intent.  Code owns the
    auditable source ledger: contiguous sequence slots, mandatory-event
    retention, explicit non-key drops, and deterministic action-unit slicing.
    """
    profile = capabilities or get_video_capabilities()
    max_content_beats_per_primary_shot = max(
        MAX_CONTENT_BEATS_PER_PRIMARY_SHOT,
        math.ceil(
            int(
                max_generation_units_per_beat
                or (
                    profile.temporal_slice_limit
                    * MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
                )
            )
            / profile.temporal_slice_limit
        ),
    )
    hard_capacity = (
        profile.temporal_slice_limit
        * max_content_beats_per_primary_shot
    )
    unit_capacity = min(
        hard_capacity,
        max_generation_units_per_beat or hard_capacity,
    )
    repaired = [dict(beat) for beat in beats]
    explicit_capacity_ledger = generation_unit_capacities_per_beat is not None
    beat_capacities = (
        list(generation_unit_capacities_per_beat)
        if generation_unit_capacities_per_beat is not None
        else [unit_capacity] * len(repaired)
    )
    if (
        len(beat_capacities) != len(repaired)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            or value > unit_capacity
            for value in beat_capacities
        )
    ):
        raise ValueError("generation-unit capacity ledger must match repair beats")
    for beat in repaired:
        raw_ids = beat.get("source_events") or []
        raw_dropped_ids = beat.get("dropped_source_events") or []
        if not isinstance(raw_ids, list):
            raise ValueError("source_events must be an array before capacity repair")
        if not isinstance(raw_dropped_ids, list):
            raise ValueError(
                "dropped_source_events must be an array before capacity repair"
            )
        beat["source_events"] = list(dict.fromkeys(raw_ids))
        beat["dropped_source_events"] = list(dict.fromkeys(raw_dropped_ids))

    def material_fits(candidate: List[Dict[str, Any]]) -> bool:
        if material_duration is None:
            return True
        try:
            _validate_beat_material_duration(
                candidate,
                events,
                material_duration,
                profile,
            )
        except ValueError:
            return False
        return True

    try:
        _validate_beat_action_capacity(
            repaired,
            events,
            profile,
            max_content_beats_per_primary_shot=(
                max_content_beats_per_primary_shot
            ),
        )
        if (
            all(
                load <= beat_capacities[index]
                for index, load in enumerate(
                    _beat_generation_unit_loads(repaired, events)
                )
            )
            and material_fits(repaired)
        ):
            for beat in repaired:
                sequences = {
                    str(events[event_id - 1].get("sequence_id") or "").strip()
                    or "__unspecified__"
                    for event_id in beat["source_events"]
                }
                if len(sequences) == 1:
                    beat["sequence_id"] = next(iter(sequences))
            return repaired
    except ValueError:
        pass

    ordered_repair = _repair_bounded_single_sequence_order(
        repaired,
        events,
        profile,
        unit_capacity=unit_capacity,
        material_duration=material_duration,
        generation_unit_capacities_per_beat=(
            generation_unit_capacities_per_beat
        ),
        max_content_beats_per_primary_shot=(
            max_content_beats_per_primary_shot
        ),
    )
    if ordered_repair is not None:
        return ordered_repair

    sequence_plan = _sequence_beat_plan(events, len(repaired), unit_capacity)
    event_by_id = {index: event for index, event in enumerate(events, 1)}
    event_sequences = {
        event_id: str(event.get("sequence_id") or "").strip() or "__unspecified__"
        for event_id, event in event_by_id.items()
    }
    sequence_slots = {
        sequence: [
            index for index, planned in enumerate(sequence_plan) if planned == sequence
        ]
        for sequence in dict.fromkeys(sequence_plan)
    }
    sequence_event_ids = {
        sequence: [
            event_id
            for event_id, event_sequence in event_sequences.items()
            if event_sequence == sequence
        ]
        for sequence in sequence_slots
    }
    generation_units = _event_generation_action_unit_counts(events)
    occurrence_requirements = _event_primary_occurrence_requirements(
        events,
        profile,
        max_content_beats_per_primary_shot=(
            max_content_beats_per_primary_shot
        ),
    )
    model_kept = {
        event_id
        for beat in repaired
        for event_id in beat["source_events"]
        if event_id in event_by_id
        and event_sequences[event_id] in sequence_slots
    }
    mandatory_ids = _mandatory_adaptation_event_ids(events)
    requested_kept = _continuous_predecessor_event_ids(
        events,
        model_kept | mandatory_ids,
    )
    placements: Dict[int, tuple[int, ...]] = {}

    def generation_allocation(
        candidate: Dict[int, tuple[int, ...]],
    ) -> tuple[List[int], Dict[int, Dict[int, int]], bool]:
        loads = [0 for _ in repaired]
        allocations: Dict[int, Dict[int, int]] = {}
        for event_id in sorted(candidate):
            positions = candidate[event_id]
            remaining_units = generation_units[event_id]
            allocations[event_id] = {}
            if explicit_capacity_ledger:
                additions = []
                for beat_index in positions:
                    assigned = min(
                        remaining_units,
                        max(
                            0,
                            beat_capacities[beat_index] - loads[beat_index],
                        ),
                    )
                    additions.append(assigned)
                    remaining_units -= assigned
            else:
                base, remainder = divmod(remaining_units, len(positions))
                additions = [
                    base + (1 if occurrence < remainder else 0)
                    for occurrence in range(len(positions))
                ]
                remaining_units = 0
            for beat_index, assigned in zip(
                positions,
                additions,
                strict=True,
            ):
                allocations[event_id][beat_index] = assigned
                loads[beat_index] += assigned
            if remaining_units or any(
                loads[beat_index] > beat_capacities[beat_index]
                for beat_index in positions
            ):
                return loads, allocations, False
        return loads, allocations, True

    def generation_loads(candidate: Dict[int, tuple[int, ...]]) -> List[int]:
        loads, _allocations, _valid = generation_allocation(candidate)
        return loads

    def placement_fits(candidate: Dict[int, tuple[int, ...]]) -> bool:
        for positions in candidate.values():
            if tuple(positions) != tuple(range(positions[0], positions[-1] + 1)):
                return False
        for sequence, event_ids in sequence_event_ids.items():
            placed_ids = [event_id for event_id in event_ids if event_id in candidate]
            for previous_id, current_id in zip(placed_ids, placed_ids[1:]):
                if candidate[previous_id][-1] > candidate[current_id][0]:
                    return False
        loads, _allocations, allocation_valid = generation_allocation(candidate)
        if not allocation_valid:
            return False
        if any(
            load > beat_capacities[index]
            for index, load in enumerate(loads)
        ):
            return False
        if material_duration is None:
            return True
        material_cost = sum(
            _minimum_primary_duration_for_units(
                load,
                profile,
                max_content_beats=max_content_beats_per_primary_shot,
            )
            for load in loads
        )
        return material_cost <= material_duration + 1e-6

    def preferred_slot(event_id: int) -> int:
        sequence = event_sequences[event_id]
        slots = sequence_slots[sequence]
        source_ids = sequence_event_ids[sequence]
        position = source_ids.index(event_id)
        if len(slots) == 1 or len(source_ids) == 1:
            return slots[0]
        relative = position * (len(slots) - 1) / (len(source_ids) - 1)
        return slots[round(relative)]

    def placement_options(event_id: int) -> List[tuple[int, ...]]:
        slots = sequence_slots[event_sequences[event_id]]
        minimum = occurrence_requirements[event_id]
        if minimum > len(slots):
            return []
        maximum = len(slots) if generation_units[event_id] else minimum
        preferred = preferred_slot(event_id)
        options = [
            option
            for count in range(minimum, maximum + 1)
            for option in itertools.combinations(slots, count)
        ]
        return sorted(
            options,
            key=lambda option: (
                len(option) - minimum,
                sum(abs(slot - preferred) for slot in option),
                option,
            ),
        )

    mandatory_order = sorted(mandatory_ids)

    def place_mandatory(position: int) -> bool:
        if position >= len(mandatory_order):
            return True
        event_id = mandatory_order[position]
        for option in placement_options(event_id):
            placements[event_id] = option
            if placement_fits(placements):
                if place_mandatory(position + 1):
                    return True
            placements.pop(event_id, None)
        return False

    if not place_mandatory(0):
        raise ValueError(
            "mandatory events cannot fit the sequence-isolated story-beat capacity"
        )

    for event_id in sorted(requested_kept - mandatory_ids):
        event_predecessors = (
            _continuous_predecessor_event_ids(events, {event_id}) - {event_id}
        )
        if not event_predecessors <= set(placements):
            continue
        candidates = []
        for option in placement_options(event_id):
            trial = dict(placements)
            trial[event_id] = option
            loads = generation_loads(trial)
            if not placement_fits(trial):
                continue
            occupied = {
                beat_index
                for positions in placements.values()
                for beat_index in positions
            }
            candidates.append((
                sum(
                    _minimum_primary_duration_for_units(
                        load,
                        profile,
                        max_content_beats=max_content_beats_per_primary_shot,
                    )
                    for load in loads
                ),
                sum(abs(slot - preferred_slot(event_id)) for slot in option),
                -sum(slot not in occupied for slot in option),
                sum(load * load for load in loads),
                option,
            ))
        if candidates:
            placements[event_id] = min(candidates)[-1]

    for sequence, slots in sequence_slots.items():
        if any(
            event_sequences[event_id] == sequence for event_id in placements
        ):
            continue
        fallback_event = sequence_event_ids[sequence][0]
        placements[fallback_event] = (slots[0],)
        if not placement_fits(placements):
            raise ValueError(f"sequence {sequence} has no event that fits its beat slots")

    while True:
        occupied = {
            beat_index
            for positions in placements.values()
            for beat_index in positions
        }
        empty_slots = [index for index in range(len(repaired)) if index not in occupied]
        if not empty_slots:
            break
        empty = empty_slots[0]
        sequence = sequence_plan[empty]
        candidates = []
        for event_id, positions in placements.items():
            if event_sequences[event_id] != sequence or empty in positions:
                continue
            trial = dict(placements)
            trial[event_id] = tuple(sorted((*positions, empty)))
            loads = generation_loads(trial)
            if placement_fits(trial):
                candidates.append((
                    sum(
                        _minimum_primary_duration_for_units(
                            load,
                            profile,
                            max_content_beats=(
                                max_content_beats_per_primary_shot
                            ),
                        )
                        for load in loads
                    ),
                    min(abs(empty - position) for position in positions),
                    -generation_units[event_id],
                    event_id,
                    trial[event_id],
                ))
        if not candidates:
            raise ValueError(f"sequence {sequence} cannot populate beat {empty + 1}")
        _material_cost, _distance, _negative_units, event_id, positions = min(candidates)
        placements[event_id] = positions

    sources_by_beat: List[List[int]] = [[] for _ in repaired]
    for event_id, positions in placements.items():
        for beat_index in positions:
            sources_by_beat[beat_index].append(event_id)
    dropped_ids = set(event_by_id) - set(placements)
    _final_loads, final_allocations, allocation_valid = generation_allocation(
        placements
    )
    if not allocation_valid:
        raise ValueError("repaired event placement lost generation units")
    dropped_by_beat: List[List[int]] = [[] for _ in repaired]
    ordered_event_sequences = list(dict.fromkeys(event_sequences.values()))

    def dropped_audit_slot(sequence: str) -> int:
        if sequence in sequence_slots:
            return sequence_slots[sequence][0]
        sequence_index = ordered_event_sequences.index(sequence)
        for adjacent in ordered_event_sequences[sequence_index + 1:]:
            if adjacent in sequence_slots:
                return sequence_slots[adjacent][0]
        for adjacent in reversed(ordered_event_sequences[:sequence_index]):
            if adjacent in sequence_slots:
                return sequence_slots[adjacent][-1]
        raise ValueError(f"dropped sequence {sequence} has no auditable beat slot")

    for event_id in sorted(dropped_ids):
        dropped_by_beat[dropped_audit_slot(event_sequences[event_id])].append(event_id)

    for index, beat in enumerate(repaired):
        source_events = sorted(sources_by_beat[index])
        dropped_events = sorted(dropped_by_beat[index])
        changed = (
            source_events != sorted(beat.get("source_events") or [])
            or dropped_events != sorted(beat.get("dropped_source_events") or [])
        )
        beat["source_events"] = source_events
        beat["dropped_source_events"] = dropped_events
        beat["sequence_id"] = sequence_plan[index]
        details = [event_by_id[event_id] for event_id in source_events]
        beat["action"] = "merge" if len(details) > 1 else "keep"
        beat["source_event_generation_unit_counts"] = {
            str(event_id): final_allocations[event_id][index]
            for event_id in source_events
        }
        if not changed:
            continue
        beat["capacity_repair"] = {
            "reason": "sequence_and_story_capacity_rebalanced",
            "sequence_id": sequence_plan[index],
            "max_generation_units": unit_capacity,
        }
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
        if locations:
            beat["where"] = locations[0] if len(locations) == 1 else " / ".join(locations)
        descriptions = [
            str(detail.get("what") or "").strip()
            for detail in details
            if str(detail.get("what") or "").strip()
        ]
        if descriptions:
            beat["what"] = "；随后".join(descriptions)
        existing_reason = str(beat.get("reason") or "").strip()
        beat["reason"] = "；".join(filter(None, [
            existing_reason,
            "代码按 sequence 与故事时钟容量重建可审计事件账本",
        ]))

    _validate_beat_action_capacity(
        repaired,
        events,
        profile,
        max_content_beats_per_primary_shot=(
            max_content_beats_per_primary_shot
        ),
    )
    if any(
        load > beat_capacities[index]
        for index, load in enumerate(
            _beat_generation_unit_loads(repaired, events)
        )
    ):
        raise ValueError("repaired beat ledger still exceeds story capacity")
    if material_duration is not None:
        _validate_beat_material_duration(
            repaired,
            events,
            material_duration,
            profile,
        )
    return repaired


def _restore_redundantly_dropped_events(
    beats: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    *,
    material_duration: int,
    capabilities: VideoModelCapabilities | None = None,
    max_generation_units_per_beat: int | None = None,
    generation_unit_capacities_per_beat: list[int] | None = None,
) -> List[Dict[str, Any]]:
    """Restore whole events that still fit the structured story clock.

    The model may rank optional events, but it may not manufacture compression
    pressure by leaving usable action capacity empty.  Starting from its kept
    ledger, try dropped events in descending normalized action coverage and
    accept only candidates that retain every previously kept event while
    satisfying sequence, per-beat and material-duration contracts.
    """
    profile = capabilities or get_video_capabilities()
    repaired = _repair_beat_action_capacity(
        beats,
        events,
        profile,
        max_generation_units_per_beat=max_generation_units_per_beat,
        material_duration=material_duration,
        generation_unit_capacities_per_beat=(
            generation_unit_capacities_per_beat
        ),
    )
    generation_units = _event_generation_action_unit_counts(events)

    def kept_ids(candidate: List[Dict[str, Any]]) -> set[int]:
        return {
            event_id
            for beat in candidate
            for event_id in (beat.get("source_events") or [])
            if isinstance(event_id, int)
        }

    dropped_ids = {
        event_id
        for beat in repaired
        for event_id in (beat.get("dropped_source_events") or [])
        if isinstance(event_id, int)
    }
    restore_order = sorted(
        dropped_ids,
        key=lambda event_id: (-generation_units.get(event_id, 0), event_id),
    )
    for event_id in restore_order:
        previous_kept = kept_ids(repaired)
        event_sequence = (
            str(events[event_id - 1].get("sequence_id") or "").strip()
            or "__unspecified__"
        )
        candidate = [dict(beat) for beat in repaired]
        for beat in candidate:
            beat["source_events"] = list(beat.get("source_events") or [])
            beat["dropped_source_events"] = [
                dropped
                for dropped in (beat.get("dropped_source_events") or [])
                if dropped != event_id
            ]
        destination = next(
            (
                beat
                for beat in candidate
                if str(beat.get("sequence_id") or "").strip() == event_sequence
            ),
            None,
        )
        if destination is None:
            continue
        destination["source_events"] = list(dict.fromkeys(
            [*destination["source_events"], event_id]
        ))
        try:
            candidate = _repair_beat_action_capacity(
                candidate,
                events,
                profile,
                max_generation_units_per_beat=max_generation_units_per_beat,
                material_duration=material_duration,
                generation_unit_capacities_per_beat=(
                    generation_unit_capacities_per_beat
                ),
            )
        except ValueError:
            continue
        if not previous_kept | {event_id} <= kept_ids(candidate):
            continue
        repaired = candidate
    return repaired


def _build_canonical_beat_skeleton(
    events: List[Dict[str, Any]],
    characters_summary: str,
    target_duration: int,
    director_plan: Optional[Dict[str, Any]],
    duration_scaled_event_plan: Dict[str, Any],
    timeline_layout_binding: Dict[str, Any],
) -> Dict[str, Any]:
    """Build Sxx from the persisted vector while the model authors cameras."""
    profile = get_video_capabilities()
    contracts = _canonical_beat_contracts(
        events,
        duration_scaled_event_plan,
        timeline_layout_binding,
    )
    director_intents = _director_intents_by_sequence(director_plan, events)
    public_contracts = [
        {
            "beat_order": contract["beat_order"],
            "sxx_id": contract["sxx_id"],
            "sequence_id": contract["sequence_id"],
            "source_events": contract["source_events"],
            "source_event_generation_unit_counts": contract[
                "source_event_generation_unit_counts"
            ],
            "execution_subslice_count": contract["execution_subslice_count"],
            "zero_story_time_source_event_ids": contract[
                "zero_story_time_source_event_ids"
            ],
            "suggested_duration": contract["suggested_duration"],
        }
        for contract in contracts
    ]
    prompt = CANONICAL_BEAT_LANGUAGE_PROMPT.format(
        target_duration=target_duration,
        beat_count=len(contracts),
        events_json=_build_events_json(events),
        characters_summary=characters_summary,
        canonical_beat_contracts=json.dumps(
            public_contracts,
            ensure_ascii=False,
            indent=2,
        ),
        director_intents_json=json.dumps(
            list(director_intents.values()),
            ensure_ascii=False,
            indent=2,
        ),
    )
    retry_limit = effective_provider_retries(MAX_RETRIES)
    last_validation_error = ""
    for attempt in range(1 + retry_limit):
        try:
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\n\n【重试纠错】上次响应越过了摄影语言边界或字段不合法。"
                    "只返回 beat_order 与合法镜头语言字段，不得返回来源、动作、时长或容量账本。"
                )
                if last_validation_error:
                    attempt_prompt += (
                        "\n上次响应的具体失败原因："
                        f"{last_validation_error}。请修正后输出完整 JSON。"
                    )
            response = _call_llm_with_timeout_retry(
                attempt_prompt,
                max_tokens=8000,
            )
            language_plan = _parse_canonical_beat_language(
                response,
                len(contracts),
            )
            beats: list[dict[str, Any]] = []
            event_by_id = {
                event_id: dict(event, event_id=event_id)
                for event_id, event in enumerate(events, 1)
            }
            for contract, authored in zip(
                contracts,
                language_plan["beats"],
                strict=True,
            ):
                beat = copy.deepcopy(contract)
                for field in _SHOT_LANGUAGE_FIELDS:
                    value = authored[field]
                    beat[field] = (
                        list(value) if isinstance(value, list) else value
                    )
                beat["action"] = (
                    "merge" if len(beat["source_events"]) > 1 else "keep"
                )
                beat["reason"] = "canonical duration-plan vector projection"
                beat["who"] = []
                beat["where"] = ""
                beat["what"] = ""
                beat["_source_event_details"] = [
                    event_by_id[event_id]
                    for event_id in beat["source_events"]
                ]
                _ground_production_beat_text_fields(
                    beat,
                    beat["_source_event_details"],
                )
                if director_intents:
                    sequence_id = beat["sequence_id"]
                    if sequence_id not in director_intents:
                        raise ValueError(
                            "canonical beat cannot bind director intent for "
                            f"sequence {sequence_id}"
                        )
                    beat["director_intent"] = _build_production_director_intent(
                        director_intents[sequence_id],
                        beat["_source_event_details"],
                        source_event_ids=list(beat["source_events"]),
                        shot=beat,
                    )
                beats.append(beat)

            _validate_beats_match_canonical_contracts(beats, contracts)
            expected_loads = [
                sum(contract["source_event_generation_unit_counts"].values())
                for contract in contracts
            ]
            actual_loads = _beat_generation_unit_loads(beats, events)
            if actual_loads != expected_loads:
                raise ValueError(
                    "canonical beat loads drifted from the duration vector: "
                    f"expected={expected_loads}, actual={actual_loads}"
                )
            if any(
                load > contract["max_generation_action_units"]
                for load, contract in zip(actual_loads, contracts, strict=True)
            ):
                raise ValueError("canonical beat exceeds its persisted capacity")
            max_content_beats = max(
                MAX_CONTENT_BEATS_PER_PRIMARY_SHOT,
                max(
                    math.ceil(
                        contract["max_generation_action_units"]
                        / profile.temporal_slice_limit
                    )
                    for contract in contracts
                ),
            )
            _validate_beat_action_capacity(
                beats,
                events,
                profile,
                max_content_beats_per_primary_shot=max_content_beats,
            )
            _validate_beat_material_duration(
                beats,
                events,
                target_duration,
                profile,
            )
            strategy = str(language_plan.get("strategy") or "").strip()
            skeleton = {
                "strategy": strategy,
                "beats": beats,
                "canonical_layout_source": DURATION_SCALED_EVENT_PLAN_SCHEMA,
                "shot_language_plan": _validate_shot_language_variation(beats),
            }
            return skeleton
        except (json.JSONDecodeError, ValueError) as exc:
            last_validation_error = str(exc)
            if attempt < retry_limit:
                print(
                    "canonical 骨架镜头语言解析失败，重试中"
                    f"（{attempt + 1}/{retry_limit}）: {exc}",
                    file=sys.stderr,
                )
                time.sleep(1)
            else:
                raise RuntimeError(
                    "canonical 骨架镜头语言响应解析失败"
                    f"（已重试 {retry_limit} 次）: {exc}"
                ) from exc
        except Exception as exc:
            raise RuntimeError(f"canonical 骨架 LLM 调用失败: {exc}") from exc
    raise RuntimeError("canonical 骨架 LLM 调用失败：未获得有效响应")


def _build_beat_skeleton(
    events: List[Dict[str, Any]],
    characters_summary: str,
    target_duration: int,
    shot_duration: int,
    beat_count: Optional[int] = None,
    director_plan: Optional[Dict[str, Any]] = None,
    max_generation_units_per_beat: int | None = None,
    generation_unit_capacities_per_beat: list[int] | None = None,
    duration_scaled_event_plan: Dict[str, Any] | None = None,
    timeline_layout_binding: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a globally informed, bounded beat table (Stage 1)."""
    profile = get_video_capabilities()
    beat_count = beat_count or estimate_shot_count(
        target_duration,
        shot_duration,
        profile,
    )
    if duration_scaled_event_plan is not None or timeline_layout_binding is not None:
        if duration_scaled_event_plan is None or timeline_layout_binding is None:
            raise ValueError(
                "canonical skeleton requires both duration plan and timeline binding"
            )
        if duration_scaled_event_plan.get("beat_count") != beat_count:
            raise ValueError("canonical duration plan beat count drifted")
        canonical_capacities = duration_scaled_event_plan.get(
            "generation_action_unit_capacities_per_beat"
        )
        if (
            generation_unit_capacities_per_beat is not None
            and list(generation_unit_capacities_per_beat)
            != canonical_capacities
        ):
            raise ValueError("caller capacity vector drifted from duration plan")
        if (
            max_generation_units_per_beat is not None
            and (
                not isinstance(canonical_capacities, list)
                or not canonical_capacities
                or max(canonical_capacities) != max_generation_units_per_beat
            )
        ):
            raise ValueError("caller maximum capacity drifted from duration plan")
        return _build_canonical_beat_skeleton(
            events,
            characters_summary,
            target_duration,
            director_plan,
            duration_scaled_event_plan,
            timeline_layout_binding,
        )
    if generation_unit_capacities_per_beat is not None and (
        len(generation_unit_capacities_per_beat) != beat_count
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in generation_unit_capacities_per_beat
        )
    ):
        raise ValueError(
            "generation-unit capacity ledger must match the screenplay beats"
        )
    generation_unit_counts = _event_generation_action_unit_counts(events)
    max_content_beats_per_primary_shot = max(
        MAX_CONTENT_BEATS_PER_PRIMARY_SHOT,
        math.ceil(
            int(
                max_generation_units_per_beat
                or (
                    profile.temporal_slice_limit
                    * MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
                )
            )
            / profile.temporal_slice_limit
        ),
    )
    occurrence_requirements = _event_primary_occurrence_requirements(
        events,
        profile,
        max_content_beats_per_primary_shot=(
            max_content_beats_per_primary_shot
        ),
    )
    per_beat_generation_unit_capacity = _generation_unit_capacity_for_story_duration(
        shot_duration,
        profile,
        max_content_beats=max_content_beats_per_primary_shot,
    )
    if max_generation_units_per_beat is not None:
        per_beat_generation_unit_capacity = min(
            per_beat_generation_unit_capacity,
            int(max_generation_units_per_beat),
        )
    sequence_beat_plan = [
        {
            "beat_order": index,
            "sequence_id": sequence,
            "max_generation_action_units": (
                generation_unit_capacities_per_beat[index - 1]
                if generation_unit_capacities_per_beat is not None
                else per_beat_generation_unit_capacity
            ),
        }
        for index, sequence in enumerate(
            _sequence_beat_plan(
                events,
                beat_count,
                per_beat_generation_unit_capacity,
            ),
            1,
        )
    ]
    director_intents = _director_intents_by_sequence(director_plan, events)
    prompt_events = []
    for event_id, event in enumerate(events, 1):
        prompt_event = dict(event)
        prompt_event["minimum_kept_primary_beat_occurrences"] = (
            occurrence_requirements[event_id]
        )
        prompt_event["generation_action_unit_count"] = generation_unit_counts[event_id]
        prompt_events.append(prompt_event)
    prompt = BEAT_SKELETON_PROMPT.format(
        target_duration=target_duration,
        shot_duration=shot_duration,
        beat_count=beat_count,
        max_generation_action_units_per_beat=per_beat_generation_unit_capacity,
        events_json=_build_events_json(prompt_events),
        characters_summary=characters_summary,
        sequence_beat_plan=json.dumps(sequence_beat_plan, ensure_ascii=False),
        director_intents_json=json.dumps(
            list(director_intents.values()),
            ensure_ascii=False,
            indent=2,
        ),
    )
    retry_limit = effective_provider_retries(MAX_RETRIES)
    last_validation_error = ""
    for attempt in range(1 + retry_limit):
        try:
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\n\n【重试纠错】上次骨架合并了不相关叙事。同一 sequence_id 的连续"
                    "action_unit 可以合并并保留完整顺序；跨 sequence 必须拆开；turning_point"
                    "必须明确保留但可以与同 sequence 的紧邻因果动作共用 beat。"
                    f"每个 beat 最多承载 {per_beat_generation_unit_capacity}"
                    "个 generation_action_units。"
                    "每个被保留事件在 beats 中出现的次数不得少于它的 "
                    "minimum_kept_primary_beat_occurrences；非关键重复事件可放入 "
                    "dropped_source_events；重复引用只分担尚未表现的后续动作。"
                )
                if last_validation_error:
                    attempt_prompt += (
                        "\n上次响应的具体失败原因："
                        f"{last_validation_error}。必须修正该结构问题后再输出。"
                    )
            response = _call_llm_with_timeout_retry(attempt_prompt, max_tokens=8000)
            skeleton = _parse_beat_skeleton(response, beat_count, len(events))
            skeleton["beats"] = _repair_beat_action_capacity(
                skeleton["beats"],
                events,
                profile,
                max_generation_units_per_beat=per_beat_generation_unit_capacity,
                material_duration=target_duration,
                generation_unit_capacities_per_beat=(
                    generation_unit_capacities_per_beat
                ),
            )
            skeleton["beats"] = _restore_redundantly_dropped_events(
                skeleton["beats"],
                events,
                material_duration=target_duration,
                capabilities=profile,
                max_generation_units_per_beat=per_beat_generation_unit_capacity,
                generation_unit_capacities_per_beat=(
                    generation_unit_capacities_per_beat
                ),
            )
            skeleton["shot_language_plan"] = _validate_shot_language_variation(
                skeleton["beats"]
            )
            _validate_beat_action_capacity(
                skeleton["beats"],
                events,
                profile,
                max_content_beats_per_primary_shot=(
                    max_content_beats_per_primary_shot
                ),
            )
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
                _ground_production_beat_text_fields(
                    beat,
                    beat["_source_event_details"],
                )
                if director_intents:
                    beat_sequences = list(
                        dict.fromkeys(
                            str(
                                event_by_id[event_id].get("sequence_id") or ""
                            ).strip()
                            for event_id in beat["source_events"]
                        )
                    )
                    if (
                        len(beat_sequences) != 1
                        or beat_sequences[0] not in director_intents
                    ):
                        raise ValueError(
                            "beat cannot bind one director intent: "
                            f"source_events={beat['source_events']}, "
                            f"sequence_ids={beat_sequences}"
                        )
                    beat["director_intent"] = _build_production_director_intent(
                        director_intents[beat_sequences[0]],
                        beat["_source_event_details"],
                        source_event_ids=list(beat["source_events"]),
                        shot=beat,
                    )
            return skeleton
        except (json.JSONDecodeError, ValueError) as e:
            last_validation_error = str(e)
            if attempt < retry_limit:
                print(f"骨架解析失败，重试中（{attempt + 1}/{retry_limit}）: {e}", file=sys.stderr)
                time.sleep(1)
            else:
                raise RuntimeError(f"骨架响应解析失败（已重试 {retry_limit} 次）: {e}") from e
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
        events_json=_build_event_details_json(event_details),
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
    retry_limit = effective_provider_retries(MAX_RETRIES)
    for start in range(first_missing, len(beats), 3):
        batch = beats[start:start + 3]
        prompt = _batch_prompt(batch, characters_summary, target_duration, shot_duration, len(shots), relay)
        parsed = None
        for attempt in range(1 + retry_limit):
            try:
                response = _call_llm_with_timeout_retry(prompt, max_tokens=16000)
                parsed = _parse_response(response)
                if len(parsed["shots"]) != len(batch):
                    raise ValueError(f"本批应输出 {len(batch)} 镜，实际为 {len(parsed['shots'])} 镜")
                expanded_fields = {
                    "shot_order", "source_events", "action", "reason", "who", "where",
                    "what", "emotion", "visual", "suggested_duration", "transition_to_next",
                    "associate_assets", "shot_size", "camera_angle", "camera_movement", "lighting_key",
                    "shot_intent", "hero_moment", "texture_keywords", "dialogue",
                    "gen_strategy",
                }
                for index, shot in enumerate(parsed["shots"], 1):
                    beat = batch[index - 1]
                    for field in _SHOT_LANGUAGE_FIELDS:
                        value = beat[field]
                        shot[field] = list(value) if isinstance(value, list) else value
                    if beat.get("director_intent"):
                        shot["director_intent"] = copy.deepcopy(
                            beat["director_intent"]
                        )
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
                if attempt < retry_limit:
                    print(f"第 {start // 3 + 1} 批解析失败，重试中（{attempt + 1}/{retry_limit}）: {e}", file=sys.stderr)
                    time.sleep(1)
                else:
                    raise RuntimeError(
                        f"第 {start // 3 + 1} 批响应解析失败（已重试 {retry_limit} 次）: {e}"
                    ) from e
            except Exception as e:
                raise RuntimeError(f"第 {start // 3 + 1} 批 LLM 调用失败: {e}") from e
        if parsed is None:
            raise RuntimeError(f"第 {start // 3 + 1} 批未获得有效响应")
        for shot, beat in zip(parsed["shots"], batch):
            # One beat maps to one shot; keep coverage/drop decisions deterministic
            # even when the expansion model drifts from its input skeleton.
            shot["source_events"] = list(beat["source_events"])
            shot["dropped_source_events"] = list(
                beat.get("dropped_source_events") or []
            )
            if "source_event_generation_unit_counts" in beat:
                shot["source_event_generation_unit_counts"] = copy.deepcopy(
                    beat["source_event_generation_unit_counts"]
                )
            else:
                shot.pop("source_event_generation_unit_counts", None)
            for field in (
                "sxx_id",
                "sequence_id",
                "max_generation_action_units",
                "execution_subslice_count",
                "timeline_assignment_ids",
                "zero_story_time_source_event_ids",
            ):
                if field in beat:
                    shot[field] = copy.deepcopy(beat[field])
                else:
                    shot.pop(field, None)
            shot["action"] = beat["action"]
            shot["suggested_duration"] = beat["suggested_duration"]
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
    _validate_shots_match_beat_ledgers(shots, beats)
    return shots


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON without ever exposing a partially written checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


SCREENPLAY_PLAN_SCHEMA = "honcut.screenplay-plan.v7"
LEGACY_SCREENPLAY_PLAN_SCHEMAS = frozenset({
    "honcut.screenplay-plan.v1",
    "honcut.screenplay-plan.v5",
    "honcut.screenplay-plan.v6",
})
LAYERED_CHECKPOINT_SCHEMA = "honcut.layered-adaptation.v19"
LEGACY_LAYERED_CHECKPOINT_SCHEMA = "honcut.layered-adaptation.v18"
LAYERED_CHECKPOINT_MIGRATION_SCHEMA = (
    "honcut.layered-adaptation-checkpoint-migration.v1"
)


def migrate_screenplay_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade the last production screenplay plan without inventing evidence."""
    if not isinstance(plan, dict):
        raise ValueError("screenplay plan must be an object")
    schema = str(plan.get("schema") or "").strip()
    if schema == SCREENPLAY_PLAN_SCHEMA:
        event_scaling = plan.get("event_action_scaling")
        records = (
            event_scaling.get("events")
            if isinstance(event_scaling, dict)
            else None
        )
        for record in records or []:
            if not isinstance(record, dict):
                continue
            if (
                record.get("scaling") == "rewrite"
                and int(record.get("source_generation_action_units") or 0)
                == 0
            ):
                raise ValueError(
                    "screenplay plan marks a zero-action event as rewrite"
                )
        return copy.deepcopy(plan)
    match = re.fullmatch(r"honcut\.screenplay-plan\.v(\d+)", schema)
    if match and int(match.group(1)) > int(SCREENPLAY_PLAN_SCHEMA.rsplit("v", 1)[1]):
        raise ValueError(
            f"screenplay plan schema {schema} is newer than supported version "
            f"{SCREENPLAY_PLAN_SCHEMA}"
        )
    if schema not in LEGACY_SCREENPLAY_PLAN_SCHEMAS:
        raise ValueError(f"unsupported screenplay plan schema: {schema or '<missing>'}")
    migrated = copy.deepcopy(plan)
    beats = [beat for beat in migrated.get("beats") or [] if isinstance(beat, dict)]
    durations = [
        beat.get("duration_s")
        for beat in beats
    ]
    if (
        not beats
        or any(
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or float(duration) <= 0
            for duration in durations
        )
    ):
        raise ValueError("legacy screenplay plan has invalid beat durations")
    profile = get_video_capabilities()
    action_capacities = [profile.temporal_slice_limit] * len(beats)
    migrated["schema"] = SCREENPLAY_PLAN_SCHEMA
    migrated["primary_shot_layout"] = {
        "schema": PRIMARY_SHOT_LAYOUT_SCHEMA,
        "shot_policy": SHOT_POLICY_CUT_DRIVEN,
        "primary_shots": len(beats),
        "story_duration_allocations_s": durations,
        "content_beat_counts": [1] * len(beats),
        "max_content_beats_per_primary_shot": (
            MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
        ),
        "generation_action_unit_capacities": action_capacities,
        "temporal_slice_capacities": action_capacities,
        "max_temporal_slices_per_content_beat": (
            profile.temporal_slice_limit
        ),
        "max_motion_contributions_per_slice": (
            profile.motion_contribution_limit
        ),
        "max_generation_action_units_per_primary_shot": (
            profile.temporal_slice_limit
        ),
        "total_generation_action_unit_capacity": sum(action_capacities),
        "cross_sxx_boundary_count": max(0, len(beats) - 1),
        "provider_request_durations_s": None,
        "projected_content_provider_request_duration_s": None,
        "projected_content_provider_padding_duration_s": None,
        "projected_padding_loss_rate": None,
        "capability_profile": profile.name,
        "objective_order": ["legacy_cut_driven_algorithm"],
        "objective_decision": {
            "migration_preserves_historical_layout": True,
        },
        "migration_source": schema,
        "provider_ledger_source": "storyboard_material_budget",
    }
    migrated["migration"] = {
        "source_schema": schema,
        "target_schema": SCREENPLAY_PLAN_SCHEMA,
        "deterministic": True,
    }
    return migrated


def _build_screenplay_plan(
    events: List[Dict[str, Any]],
    shots: List[Dict[str, Any]],
    source_capacity_plan: Dict[str, Any],
    *,
    target_duration: int,
    source_events_hash: str | None = None,
    capabilities: VideoModelCapabilities | None = None,
    production_events: List[Dict[str, Any]] | None = None,
    duration_scaled_event_plan: Dict[str, Any] | None = None,
    primary_shot_layout: Dict[str, Any] | None = None,
    source_action_timeline: Dict[str, Any] | None = None,
    production_action_timeline: Dict[str, Any] | None = None,
    timeline_layout_binding: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Separate the complete source ledger from the fitted production ledger.

    ``_estimate_action_capacity_plan`` deliberately measures the complete
    authored event ledger.  Once Adaptation has produced and validated a
    duration-scaled shot plan, carrying that input-pressure status forward as
    the final capacity status is incorrect accounting.  This function records
    both ledgers and reconciles the public capacity result only after exact
    production durations and source references exist.
    """
    if not shots:
        raise ValueError("screenplay plan requires at least one production beat")
    profile = capabilities or get_video_capabilities()
    resolved_primary_layout = copy.deepcopy(
        primary_shot_layout or source_capacity_plan.get("primary_shot_layout")
    )
    if not isinstance(resolved_primary_layout, dict) or (
        resolved_primary_layout.get("schema") != PRIMARY_SHOT_LAYOUT_SCHEMA
    ):
        raise ValueError("screenplay plan requires a current primary-shot layout")
    if resolved_primary_layout.get("shot_policy") not in SHOT_POLICIES:
        raise ValueError("primary-shot layout has an invalid shot policy")
    if int(resolved_primary_layout.get("primary_shots") or 0) != len(shots):
        raise ValueError("primary-shot layout count does not match production shots")
    max_content_beats_per_primary_shot = int(
        resolved_primary_layout.get("max_content_beats_per_primary_shot")
        or MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    )
    layout_durations = resolved_primary_layout.get(
        "story_duration_allocations_s"
    )
    shot_durations = [shot.get("suggested_duration") for shot in shots]
    if layout_durations != shot_durations:
        raise ValueError("primary-shot layout duration ledger does not match shots")
    adapted_events = production_events or events
    if len(adapted_events) != len(events):
        raise ValueError("production event ledger must preserve source event cardinality")
    for event_id, (source_event, production_event) in enumerate(
        zip(events, adapted_events, strict=True),
        1,
    ):
        source_sequence = (
            str(source_event.get("sequence_id") or "").strip()
            or "__unspecified__"
        )
        production_sequence = (
            str(production_event.get("sequence_id") or "").strip()
            or "__unspecified__"
        )
        if source_sequence != production_sequence:
            raise ValueError(
                f"production event {event_id} changed source sequence identity"
            )
    _validate_beat_action_capacity(
        shots,
        adapted_events,
        profile,
        max_content_beats_per_primary_shot=(
            max_content_beats_per_primary_shot
        ),
    )
    _validate_beat_material_duration(
        shots,
        adapted_events,
        target_duration,
        profile,
    )

    action_scaling_records: list[dict[str, Any]] = []
    if duration_scaled_event_plan is not None:
        if duration_scaled_event_plan.get("schema") != (
            DURATION_SCALED_EVENT_PLAN_SCHEMA
        ):
            raise ValueError("unsupported duration-scaled event plan schema")
        raw_records = duration_scaled_event_plan.get("events")
        if not isinstance(raw_records, list):
            raise ValueError("duration-scaled event plan requires events")
        record_ids = [record.get("source_event_id") for record in raw_records]
        if record_ids != list(range(1, len(events) + 1)):
            raise ValueError(
                "duration-scaled event plan must cover source events in order"
            )
        for event_id, (record, source_event, production_event) in enumerate(
            zip(raw_records, events, adapted_events, strict=True),
            1,
        ):
            selected = record.get("selected_source_micro_action_indexes")
            omitted = record.get("omitted_source_micro_action_indexes")
            if (
                not isinstance(selected, list)
                or not isinstance(omitted, list)
                or any(not isinstance(index, int) for index in [*selected, *omitted])
            ):
                raise ValueError(
                    f"duration-scaled event {event_id} has invalid action indexes"
                )
            raw_actions = source_event.get("micro_actions") or []
            if isinstance(raw_actions, str):
                raw_actions = [raw_actions]
            source_actions = [
                str(action).strip()
                for action in raw_actions
                if str(action).strip()
            ]
            expected_indexes = set(range(1, len(source_actions) + 1))
            if (
                set(selected) & set(omitted)
                or set(selected) | set(omitted) != expected_indexes
            ):
                raise ValueError(
                    f"duration-scaled event {event_id} action lineage is incomplete"
                )
            actual_actions = production_event.get("micro_actions") or []
            if isinstance(actual_actions, str):
                actual_actions = [actual_actions]
            if record.get("scaling") == "rewrite":
                rewrite = production_event.get("production_action_rewrite")
                if (
                    selected != list(range(1, len(source_actions) + 1))
                    or omitted
                    or not isinstance(rewrite, dict)
                    or rewrite.get("schema")
                    != SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA
                    or rewrite.get("omitted_source_micro_action_indexes") != []
                ):
                    raise ValueError(
                        f"production event {event_id} has an invalid rewrite ledger"
                    )
                groups = rewrite.get("groups")
                if not isinstance(groups, list):
                    raise ValueError(
                        f"production event {event_id} rewrite groups are missing"
                    )
                rewritten_actions = [
                    str(group.get("rewritten_micro_action") or "").strip()
                    for group in groups
                    if isinstance(group, dict)
                ]
                source_actions_by_index = {
                    int(index): str(action)
                    for group in groups
                    for index, action in zip(
                        group.get("source_micro_action_indexes") or [],
                        group.get("source_actions") or [],
                        strict=True,
                    )
                }
                if (
                    rewritten_actions != list(actual_actions)
                    or set(source_actions_by_index) != expected_indexes
                    or [
                        source_actions_by_index[index]
                        for index in range(1, len(source_actions) + 1)
                    ] != source_actions
                ):
                    raise ValueError(
                        f"production event {event_id} rewrite lineage is incomplete"
                    )
                normalize_event_action_units(production_event)
            else:
                expected_actions = [
                    source_actions[index - 1] for index in selected
                ]
                if expected_actions != list(actual_actions):
                    raise ValueError(
                        f"production event {event_id} actions do not match source lineage"
                    )
            action_scaling_records.append(copy.deepcopy(record))
    else:
        mandatory_ids = _mandatory_adaptation_event_ids(events)
        structural_mandatory_ids = {
            event_id
            for event_id, event in enumerate(events, 1)
            if _event_is_mandatory_for_adaptation(event)
        }
        terminal_event_ids = terminal_outcome_event_ids(events)
        for event_id, event in enumerate(events, 1):
            raw_actions = event.get("micro_actions") or []
            if isinstance(raw_actions, str):
                raw_actions = [raw_actions]
            indexes = list(range(1, len([
                action for action in raw_actions if str(action).strip()
            ]) + 1))
            action_scaling_records.append({
                "source_event_id": event_id,
                "sequence_id": (
                    str(event.get("sequence_id") or "").strip()
                    or "__unspecified__"
                ),
                "mandatory": event_id in mandatory_ids,
                "mandatory_reason": (
                    "structural"
                    if event_id in structural_mandatory_ids
                    else (
                        "terminal_outcome"
                        if event_id in terminal_event_ids
                        else (
                            "continuous_predecessor"
                            if event_id in mandatory_ids
                            else "optional"
                        )
                    )
                ),
                "selected_source_micro_action_indexes": indexes,
                "omitted_source_micro_action_indexes": [],
                "scaling": "full",
            })
    action_scaling_by_event = {
        record["source_event_id"]: record for record in action_scaling_records
    }

    event_ids = set(range(1, len(events) + 1))
    kept_ids: set[int] = set()
    omitted_ids: set[int] = set()
    omitted_occurrences: list[int] = []
    beats: list[dict[str, Any]] = []
    total_duration = 0.0
    production_generation_units = 0
    director_projection_schemas: set[str] = set()

    for beat_order, shot in enumerate(shots, 1):
        source_refs = shot.get("source_events") or []
        omitted_refs = shot.get("dropped_source_events") or []
        if (
            not isinstance(source_refs, list)
            or not source_refs
            or any(not isinstance(event_id, int) for event_id in source_refs)
        ):
            raise ValueError(
                f"production beat {beat_order} has invalid source event references"
            )
        if (
            not isinstance(omitted_refs, list)
            or any(not isinstance(event_id, int) for event_id in omitted_refs)
        ):
            raise ValueError(
                f"production beat {beat_order} has invalid omitted event references"
            )
        if set(source_refs) & set(omitted_refs):
            raise ValueError(
                f"production beat {beat_order} both keeps and omits a source event"
            )

        sequence_ids = [
            str(value).strip()
            for value in (shot.get("source_sequence_ids") or [])
            if str(value).strip()
        ]
        sequence_ids = list(dict.fromkeys(sequence_ids))
        source_sequences = {
            str(events[event_id - 1].get("sequence_id") or "").strip()
            or "__unspecified__"
            for event_id in source_refs
            if event_id in event_ids
        }
        if not sequence_ids and source_sequences == {"__unspecified__"}:
            sequence_ids = ["__unspecified__"]
        if len(sequence_ids) != 1:
            raise ValueError(
                f"production beat {beat_order} must bind exactly one sequence"
            )
        if source_sequences != {sequence_ids[0]}:
            raise ValueError(
                f"production beat {beat_order} source refs do not match "
                f"sequence {sequence_ids[0]}"
            )
        director_intent = shot.get("director_intent")
        if director_intent is not None:
            if not isinstance(director_intent, dict):
                raise ValueError(
                    f"production beat {beat_order} director_intent must be an object"
                )
            intent_schema = str(director_intent.get("schema") or "").strip()
            if intent_schema != PRODUCTION_DIRECTOR_INTENT_SCHEMA:
                raise ValueError(
                    f"production beat {beat_order} has unsupported director intent "
                    f"schema: {intent_schema or '<missing>'}"
                )
            if director_intent.get("source_event_ids") != list(
                dict.fromkeys(source_refs)
            ):
                raise ValueError(
                    f"production beat {beat_order} director intent lineage does "
                    "not match source events"
                )
            if director_intent.get("sequence_id") != sequence_ids[0]:
                raise ValueError(
                    f"production beat {beat_order} director intent sequence mismatch"
                )
            director_projection_schemas.add(intent_schema)

        raw_duration = shot.get("suggested_duration")
        if isinstance(raw_duration, bool):
            raise ValueError(f"production beat {beat_order} has invalid duration")
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"production beat {beat_order} has invalid duration"
            ) from exc
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"production beat {beat_order} has invalid duration")
        total_duration += duration

        generation_units = shot.get("generation_action_units") or []
        if not isinstance(generation_units, list):
            raise ValueError(
                f"production beat {beat_order} generation_action_units must be a list"
            )
        production_generation_units += len(generation_units)
        kept_ids.update(source_refs)
        omitted_ids.update(omitted_refs)
        omitted_occurrences.extend(omitted_refs)
        beats.append(
            {
                "beat_id": f"SPB{beat_order:03d}",
                "beat_order": beat_order,
                "sequence_id": sequence_ids[0],
                "duration_s": int(duration) if duration.is_integer() else duration,
                "source_refs": list(dict.fromkeys(source_refs)),
                "omitted_source_refs": list(dict.fromkeys(omitted_refs)),
                "adaptation_action": str(shot.get("action") or "keep"),
                "narrative_summary": str(shot.get("what") or "").strip(),
                "director_intent": copy.deepcopy(director_intent),
                "production_action_refs": [
                    {
                        "source_event_id": event_id,
                        "selected_source_micro_action_indexes": list(
                            action_scaling_by_event[event_id].get(
                                "selected_source_micro_action_indexes"
                            ) or []
                        ),
                        "omitted_source_micro_action_indexes": list(
                            action_scaling_by_event[event_id].get(
                                "omitted_source_micro_action_indexes"
                            ) or []
                        ),
                        "production_action_rewrite_schema": (
                            SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA
                            if action_scaling_by_event[event_id].get("scaling")
                            == "rewrite"
                            else None
                        ),
                        "source_micro_action_groups": [
                            list(group.get("source_micro_action_indexes") or [])
                            for group in (
                                adapted_events[event_id - 1].get(
                                    "production_action_rewrite", {}
                                ).get("groups")
                                or []
                            )
                        ],
                    }
                    for event_id in dict.fromkeys(source_refs)
                ],
            }
        )

    if len(omitted_occurrences) != len(set(omitted_occurrences)):
        raise ValueError("an omitted source event may be recorded only once")
    if kept_ids & omitted_ids:
        raise ValueError("source events cannot be both kept and omitted")
    invalid_ids = (kept_ids | omitted_ids) - event_ids
    if invalid_ids:
        raise ValueError(f"screenplay plan references unknown events: {sorted(invalid_ids)}")
    missing_ids = event_ids - kept_ids - omitted_ids
    if missing_ids:
        raise ValueError(f"screenplay plan does not account for events: {sorted(missing_ids)}")
    mandatory_ids = _mandatory_adaptation_event_ids(events)
    missing_mandatory = mandatory_ids - kept_ids
    if missing_mandatory:
        raise ValueError(
            f"screenplay plan omits mandatory events: {sorted(missing_mandatory)}"
        )
    missing_continuous_predecessors = (
        _continuous_predecessor_event_ids(events, kept_ids) - kept_ids
    )
    if missing_continuous_predecessors:
        raise ValueError(
            "screenplay plan keeps events without continuous predecessors: "
            f"{sorted(missing_continuous_predecessors)}"
        )
    if not math.isclose(total_duration, float(target_duration), abs_tol=1e-6):
        raise ValueError(
            f"production screenplay duration {total_duration:g}s does not equal "
            f"the {target_duration}s delivery target"
        )

    source_status = str(
        source_capacity_plan.get("action_capacity_status") or ""
    )
    if source_status not in {
        "fits_story_clock",
        "screenplay_compression_required",
    }:
        raise ValueError(f"unsupported source capacity status: {source_status!r}")
    intra_event_scaling_applied = any(
        record.get("scaling") == "rewrite"
        or record.get("omitted_source_micro_action_indexes")
        for record in action_scaling_records
    )
    duration_scaling_status = (
        "applied"
        if (
            omitted_ids
            or intra_event_scaling_applied
            or source_status == "screenplay_compression_required"
        )
        else "not_required"
    )
    serialized_action_scaling = []
    for record in action_scaling_records:
        serialized = copy.deepcopy(record)
        event_id = serialized["source_event_id"]
        serialized["production_status"] = (
            "whole_event_omitted" if event_id in omitted_ids else "kept"
        )
        serialized_action_scaling.append(serialized)
    serialized_events = json.dumps(
        events,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    screenplay_plan = {
        "schema": SCREENPLAY_PLAN_SCHEMA,
        "target_duration_s": int(target_duration),
        "primary_shot_layout": resolved_primary_layout,
        "source_ledger": {
            "artifact": "phase1_events.json",
            "action_timeline_artifact": "ACTION_TIMELINE.json",
            "action_timeline_schema": ACTION_TIMELINE_SCHEMA,
            "event_count": len(events),
            "generation_action_units": int(
                source_capacity_plan.get("generation_action_units") or 0
            ),
            "minimum_material_duration_s": int(
                source_capacity_plan.get("minimum_material_duration") or 0
            ),
            "capacity_status": source_status,
            "capacity_pressure_ratio": float(
                source_capacity_plan.get("action_capacity_pressure_ratio") or 0
            ),
        },
        "production_ledger": {
            "capacity_status": "fits_story_clock",
            "duration_scaling_status": duration_scaling_status,
            "event_action_scaling_schema": DURATION_SCALED_EVENT_PLAN_SCHEMA,
            "intra_event_scaling_applied": intra_event_scaling_applied,
            "intra_event_omitted_micro_action_count": sum(
                len(record.get("omitted_source_micro_action_indexes") or [])
                for record in action_scaling_records
                if record["source_event_id"] in kept_ids
            ),
            "source_fact_loss_count": 0,
            "source_indexed_rewrite_schema": (
                SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA
                if any(
                    record.get("scaling") == "rewrite"
                    for record in action_scaling_records
                    if record["source_event_id"] in kept_ids
                )
                else None
            ),
            "source_indexed_rewrite_event_count": sum(
                1
                for record in action_scaling_records
                if record["source_event_id"] in kept_ids
                and record.get("scaling") == "rewrite"
            ),
            "event_count": len(kept_ids),
            "generation_action_units": production_generation_units,
            "effective_story_duration_s": (
                int(total_duration) if total_duration.is_integer() else total_duration
            ),
            "kept_source_event_ids": sorted(kept_ids),
            "omitted_source_event_ids": sorted(omitted_ids),
            "base_mandatory_source_event_ids": sorted(
                _base_mandatory_adaptation_event_ids(events)
            ),
            "terminal_outcome_source_event_ids": sorted(
                terminal_outcome_event_ids(events)
            ),
            "mandatory_source_event_ids": sorted(mandatory_ids),
            "causal_predecessor_source_event_ids": sorted(
                mandatory_ids - _base_mandatory_adaptation_event_ids(events)
            ),
        },
        "event_action_scaling": {
            "schema": DURATION_SCALED_EVENT_PLAN_SCHEMA,
            "events": serialized_action_scaling,
        },
        "source_action_timeline": copy.deepcopy(source_action_timeline),
        "production_action_timeline": copy.deepcopy(
            production_action_timeline
        ),
        "timeline_layout_binding": copy.deepcopy(timeline_layout_binding),
        "beats": beats,
        "lineage": {
            "source_events_sha256": hashlib.sha256(serialized_events).hexdigest(),
            "source_checkpoint_input_hash": source_events_hash,
        },
    }
    if director_projection_schemas:
        if director_projection_schemas != {PRODUCTION_DIRECTOR_INTENT_SCHEMA}:
            raise ValueError("production beats use inconsistent director projections")
        screenplay_plan["production_ledger"][
            "production_director_intent_schema"
        ] = PRODUCTION_DIRECTOR_INTENT_SCHEMA
    if duration_scaled_event_plan is not None:
        semantic_status = str(
            duration_scaled_event_plan.get("semantic_selection_status")
            or "not_recorded"
        )
        screenplay_plan["production_ledger"][
            "semantic_action_selection_status"
        ] = semantic_status
        screenplay_plan["event_action_scaling"][
            "semantic_selection_status"
        ] = semantic_status
        semantic_schema = duration_scaled_event_plan.get(
            "semantic_selection_schema"
        )
        if semantic_schema:
            screenplay_plan["production_ledger"][
                "semantic_action_selection_schema"
            ] = semantic_schema
            screenplay_plan["event_action_scaling"][
                "semantic_selection_schema"
            ] = semantic_schema
        if duration_scaled_event_plan.get("director_plan_schema"):
            screenplay_plan["event_action_scaling"]["director_plan_schema"] = (
                duration_scaled_event_plan["director_plan_schema"]
            )
    reconciled_capacity = dict(source_capacity_plan)
    reconciled_capacity.update(
        {
            "source_action_capacity_status": source_status,
            "source_generation_action_units": int(
                source_capacity_plan.get("generation_action_units") or 0
            ),
            "action_capacity_status": "fits_story_clock",
            "duration_scaling_status": duration_scaling_status,
            "intra_event_scaling_applied": intra_event_scaling_applied,
            "production_generation_action_units": production_generation_units,
            "production_event_count": len(kept_ids),
            "omitted_source_event_count": len(omitted_ids),
            "screenplay_plan_schema": SCREENPLAY_PLAN_SCHEMA,
        }
    )
    return screenplay_plan, reconciled_capacity


def _layered_input_fingerprint(
    events: List[Dict[str, Any]],
    characters_summary: str,
    target_duration: int,
    shot_duration: int,
    expected_beats: int,
    director_plan: Optional[Dict[str, Any]] = None,
    screenplay_rewrite_request: Optional[Dict[str, Any]] = None,
    shot_policy: str = DEFAULT_SHOT_POLICY,
    primary_shot_layout: Optional[Dict[str, Any]] = None,
    source_action_timeline: Optional[Dict[str, Any]] = None,
    production_action_timeline: Optional[Dict[str, Any]] = None,
    timeline_layout_binding: Optional[Dict[str, Any]] = None,
    checkpoint_schema: str = LAYERED_CHECKPOINT_SCHEMA,
) -> str:
    """Bind layered checkpoints to the complete semantic adaptation input."""
    contract = {
        "schema": checkpoint_schema,
        "events": events,
        "characters_summary": characters_summary,
        "target_duration": target_duration,
        "shot_duration": shot_duration,
        "expected_beats": expected_beats,
        "director_intents": list(
            _director_intents_by_sequence(director_plan, events).values()
        ),
        "screenplay_rewrite_request": screenplay_rewrite_request,
        "shot_policy": _validate_shot_policy(shot_policy),
        "primary_shot_layout": primary_shot_layout,
        "source_action_timeline": source_action_timeline,
        "production_action_timeline": production_action_timeline,
        "timeline_layout_binding": timeline_layout_binding,
    }
    if checkpoint_schema == LAYERED_CHECKPOINT_SCHEMA:
        contract[
            "source_indexed_rewrite_reconciliation_policy_sha256"
        ] = SOURCE_INDEXED_REWRITE_RECONCILIATION_POLICY_SHA256
    encoded = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class UnsupportedLayeredCheckpointSchemaError(ValueError):
    """A future layered checkpoint must never trigger a silent rebuild."""


def _checkpoint_match_kind(
    value: Any,
    input_fingerprint: str,
    legacy_input_fingerprint: str | None = None,
) -> str:
    metadata = value.get("_checkpoint") if isinstance(value, dict) else None
    if not isinstance(metadata, dict):
        return "invalid"
    schema = str(metadata.get("schema") or "").strip()
    fingerprint = str(metadata.get("input_fingerprint") or "").strip()
    if schema == LAYERED_CHECKPOINT_SCHEMA:
        return "current" if fingerprint == input_fingerprint else "invalid"
    if schema == LEGACY_LAYERED_CHECKPOINT_SCHEMA:
        if legacy_input_fingerprint and fingerprint == legacy_input_fingerprint:
            return "legacy_v18"
        return "legacy_v18_audit_only"
    version_match = re.fullmatch(r"honcut\.layered-adaptation\.v(\d+)", schema)
    if version_match and int(version_match.group(1)) > 19:
        raise UnsupportedLayeredCheckpointSchemaError(
            f"layered checkpoint schema {schema} is newer than supported "
            f"version {LAYERED_CHECKPOINT_SCHEMA}"
        )
    return "invalid"


def _checkpoint_matches(value: Any, input_fingerprint: str) -> bool:
    return _checkpoint_match_kind(value, input_fingerprint) == "current"


def _write_layered_checkpoint_migration_receipt(
    output_dir: Path,
    *,
    status: str,
    input_fingerprint: str,
    legacy_input_fingerprint: str | None,
    artifacts: list[dict[str, Any]],
    reason: str | None = None,
) -> None:
    receipt = {
        "schema": LAYERED_CHECKPOINT_MIGRATION_SCHEMA,
        "status": status,
        "from_schema": LEGACY_LAYERED_CHECKPOINT_SCHEMA,
        "to_schema": LAYERED_CHECKPOINT_SCHEMA,
        "input_fingerprint": input_fingerprint,
        "legacy_input_fingerprint": legacy_input_fingerprint,
        "artifacts": artifacts,
        "provider_request_count": 0,
    }
    if reason:
        receipt["reason"] = reason
    _atomic_write_json(
        output_dir / "layered_checkpoint_migration_v18_to_v19.json",
        receipt,
    )


def _load_layered_checkpoints(
    output_dir: Path,
    events: List[Dict[str, Any]],
    expected_beats: int,
    input_fingerprint: str,
    *,
    legacy_input_fingerprint: str | None = None,
    max_content_beats_per_primary_shot: int = (
        MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
    ),
    generation_unit_capacities_per_beat: list[int] | None = None,
    duration_scaled_event_plan: Dict[str, Any] | None = None,
    timeline_layout_binding: Dict[str, Any] | None = None,
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load only valid, contiguous layered checkpoints."""
    skeleton = None
    migrated_artifacts: list[dict[str, Any]] = []
    skeleton_path = output_dir / "beat_skeleton.json"
    if skeleton_path.exists():
        try:
            candidate = json.loads(skeleton_path.read_text(encoding="utf-8"))
            match_kind = _checkpoint_match_kind(
                candidate,
                input_fingerprint,
                legacy_input_fingerprint,
            )
            if match_kind == "legacy_v18_audit_only":
                _write_layered_checkpoint_migration_receipt(
                    output_dir,
                    status="audit_only",
                    input_fingerprint=input_fingerprint,
                    legacy_input_fingerprint=legacy_input_fingerprint,
                    artifacts=[{
                        "path": skeleton_path.name,
                        "sha256": hashlib.sha256(
                            skeleton_path.read_bytes()
                        ).hexdigest(),
                    }],
                    reason="legacy checkpoint fingerprint or lineage mismatch",
                )
                raise ValueError("legacy layered skeleton is audit-only")
            if match_kind not in {"current", "legacy_v18"}:
                raise ValueError("layered skeleton belongs to a different input")
            _parse_beat_skeleton(json.dumps(candidate, ensure_ascii=False), expected_beats, len(events))
            _validate_beat_action_capacity(
                candidate["beats"],
                events,
                max_content_beats_per_primary_shot=(
                    max_content_beats_per_primary_shot
                ),
            )
            if generation_unit_capacities_per_beat is not None:
                loads = _beat_generation_unit_loads(candidate["beats"], events)
                if (
                    len(generation_unit_capacities_per_beat) != len(loads)
                    or any(
                        load > generation_unit_capacities_per_beat[index]
                        for index, load in enumerate(loads)
                    )
                ):
                    raise ValueError(
                        "layered skeleton exceeds the primary-shot capacity ledger"
                    )
            if duration_scaled_event_plan is not None or timeline_layout_binding is not None:
                if duration_scaled_event_plan is None or timeline_layout_binding is None:
                    raise ValueError("canonical checkpoint validation is incomplete")
                _validate_beats_match_canonical_contracts(
                    candidate["beats"],
                    _canonical_beat_contracts(
                        events,
                        duration_scaled_event_plan,
                        timeline_layout_binding,
                    ),
                )
            candidate["shot_language_plan"] = _validate_shot_language_variation(
                candidate["beats"]
            )
            event_by_id = {i: dict(event, event_id=i) for i, event in enumerate(events, 1)}
            for beat in candidate["beats"]:
                beat["_source_event_details"] = [event_by_id[event_id] for event_id in beat["source_events"]]
            if match_kind == "legacy_v18":
                migrated_artifacts.append({
                    "path": skeleton_path.name,
                    "sha256": hashlib.sha256(
                        skeleton_path.read_bytes()
                    ).hexdigest(),
                })
                candidate["_checkpoint"] = {
                    "schema": LAYERED_CHECKPOINT_SCHEMA,
                    "input_fingerprint": input_fingerprint,
                }
            skeleton = candidate
            print(f"  ↺ Reusing layered checkpoint: {skeleton_path}")
        except UnsupportedLayeredCheckpointSchemaError:
            raise
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            skeleton = None

    shots: List[Dict[str, Any]] = []
    partial_path = output_dir / "shots_partial.json"
    if skeleton is not None and partial_path.exists():
        try:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            match_kind = _checkpoint_match_kind(
                partial,
                input_fingerprint,
                legacy_input_fingerprint,
            )
            if match_kind == "legacy_v18_audit_only":
                raise ValueError("legacy partial checkpoint is audit-only")
            if match_kind not in {"current", "legacy_v18"}:
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
            _validate_shots_match_beat_ledgers(
                candidate_shots,
                skeleton["beats"],
            )
            if match_kind == "legacy_v18":
                migrated_artifacts.append({
                    "path": partial_path.name,
                    "sha256": hashlib.sha256(
                        partial_path.read_bytes()
                    ).hexdigest(),
                })
            shots = candidate_shots
            print(f"  ↺ Reusing {len(completed)} completed layered batch(es): {partial_path}")
        except UnsupportedLayeredCheckpointSchemaError:
            raise
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            shots = []
    if migrated_artifacts:
        _write_layered_checkpoint_migration_receipt(
            output_dir,
            status="migrated",
            input_fingerprint=input_fingerprint,
            legacy_input_fingerprint=legacy_input_fingerprint,
            artifacts=migrated_artifacts,
        )
    return skeleton, shots


# ─── 核心函数 ────────────────────────────────────────────────────────────────

def adapt_events(
    events: List[Dict[str, Any]],
    characters: Optional[List[Dict[str, Any]]] = None,
    target_duration: Optional[int] = None,
    shot_duration: int = AVG_SHOT_DURATION,
    source_text: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    director_plan: Optional[Dict[str, Any]] = None,
    source_events_hash: Optional[str] = None,
    screenplay_rewrite_request: Optional[Dict[str, Any]] = None,
    shot_policy: str = DEFAULT_SHOT_POLICY,
    max_material_padding_ratio: float = MAX_CONTENT_PROVIDER_PADDING_LOSS_RATE,
    delivery_overrun_ratio: float = 0.0,
) -> Dict[str, Any]:
    """
    将事件列表改编为 shot 列表

    Args:
        events: 事件列表（event_extractor.py 输出）
        characters: 角色列表（character_discoverer.py 输出，可选）
        target_duration: 最终交付时长（秒），默认 None（根据剧本长度智能预估）
        shot_duration: 每镜平均时长（秒），默认 12
        source_text: 原始剧本文本（用于智能预估时长）
        director_plan: Event Extractor 之后生成的 sequence 导演意图
        source_events_hash: ``phase1_events.json`` 的输入血缘哈希

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
    source_action_timeline = build_action_timeline(
        events,
        max_motion_contributions_per_slice=(
            capability_profile.motion_contribution_limit
        ),
    )
    if output_dir is not None:
        _atomic_write_json(
            Path(output_dir) / "ACTION_TIMELINE.json",
            source_action_timeline,
        )

    # ── 分离交付时长与 Phase 8 剪辑前的素材时长 ───────────────────────────
    capacity_plan = _estimate_action_capacity_plan(
        events,
        target_duration,
        shot_duration,
        shot_policy=shot_policy,
        max_material_padding_ratio=max_material_padding_ratio,
        delivery_overrun_ratio=delivery_overrun_ratio,
    )
    rewrite_layout = None
    if screenplay_rewrite_request is not None:
        rewrite_story_duration = int(
            capacity_plan.get("planned_delivery_duration")
            or target_duration
        )
        rewrite_layout = _resolve_padding_rewrite_layout(
            capacity_plan,
            target_duration=rewrite_story_duration,
            capabilities=capability_profile,
            rewrite_request=screenplay_rewrite_request,
            shot_policy=shot_policy,
        )
        rewrite_layout["nominal_delivery_duration_s"] = int(
            capacity_plan.get("nominal_delivery_duration") or target_duration
        )
        rewrite_layout["delivery_ceiling_duration_s"] = int(
            capacity_plan.get("delivery_ceiling_duration") or target_duration
        )
        rewrite_layout["planned_delivery_duration_s"] = rewrite_story_duration
        rewrite_layout["delivery_overrun_ratio"] = float(
            capacity_plan.get("delivery_overrun_ratio") or 0.0
        )
        capacity_plan = {
            **capacity_plan,
            "primary_shots": rewrite_layout["primary_shots"],
            "screenplay_rewrite": rewrite_layout,
            "primary_shot_layout": rewrite_layout,
        }
    primary_shot_layout = (
        rewrite_layout or capacity_plan["primary_shot_layout"]
    )
    max_content_beats_per_primary_shot = int(
        primary_shot_layout.get("max_content_beats_per_primary_shot")
        or MAX_CONTENT_BEATS_PER_PRIMARY_SHOT
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
    production_events, duration_scaled_event_plan = (
        _build_duration_scaled_event_plan(
            events,
            target_duration=material_duration,
            beat_count=max_shots,
            effective_shot_duration=effective_shot_duration,
            capabilities=capability_profile,
            max_generation_units_per_beat=(
                primary_shot_layout[
                    "max_generation_action_units_per_primary_shot"
                ]
            ),
            maximum_total_generation_units=primary_shot_layout[
                "production_action_unit_target"
            ],
            generation_unit_capacities_per_beat=list(
                primary_shot_layout["generation_action_unit_capacities"]
            ),
        )
    )
    if duration_scaled_event_plan["intra_event_scaling_applied"]:
        if director_plan is None:
            duration_scaled_event_plan["semantic_selection_status"] = (
                "deterministic_without_director"
            )
        else:
            production_events, duration_scaled_event_plan = (
                _apply_director_action_selection(
                    events,
                    production_events,
                    duration_scaled_event_plan,
                    director_plan,
                )
            )
    else:
        duration_scaled_event_plan["semantic_selection_status"] = "not_required"

    production_action_timeline = build_action_timeline(
        production_events,
        max_motion_contributions_per_slice=(
            capability_profile.motion_contribution_limit
        ),
    )
    timeline_layout_binding = _bind_action_timeline_to_primary_layout(
        production_events,
        duration_scaled_event_plan,
        primary_shot_layout,
        capability_profile,
    )
    duration_scaled_event_plan["timeline_layout_binding"] = copy.deepcopy(
        timeline_layout_binding
    )
    primary_shot_layout["timeline_assignment_count"] = len(
        timeline_layout_binding["assignments"]
    )

    characters_summary = _build_characters_summary(characters)

    def _run_layered_adaptation() -> Dict[str, Any]:
        checkpoint_dir = Path(output_dir) if output_dir is not None else None
        fingerprint_kwargs = {}
        if screenplay_rewrite_request is not None:
            fingerprint_kwargs["screenplay_rewrite_request"] = (
                screenplay_rewrite_request
            )
        layered_fingerprint = _layered_input_fingerprint(
            production_events,
            characters_summary,
            material_duration,
            effective_shot_duration,
            max_shots,
            director_plan,
            shot_policy=shot_policy,
            primary_shot_layout=primary_shot_layout,
            source_action_timeline=source_action_timeline,
            production_action_timeline=production_action_timeline,
            timeline_layout_binding=timeline_layout_binding,
            **fingerprint_kwargs,
        )
        legacy_layered_fingerprint = _layered_input_fingerprint(
            production_events,
            characters_summary,
            material_duration,
            effective_shot_duration,
            max_shots,
            director_plan,
            shot_policy=shot_policy,
            primary_shot_layout=primary_shot_layout,
            source_action_timeline=source_action_timeline,
            production_action_timeline=production_action_timeline,
            timeline_layout_binding=timeline_layout_binding,
            checkpoint_schema=LEGACY_LAYERED_CHECKPOINT_SCHEMA,
            **fingerprint_kwargs,
        )
        skeleton = None
        resumed_shots: List[Dict[str, Any]] = []
        if checkpoint_dir is not None:
            skeleton, resumed_shots = _load_layered_checkpoints(
                checkpoint_dir,
                production_events,
                max_shots,
                layered_fingerprint,
                legacy_input_fingerprint=legacy_layered_fingerprint,
                max_content_beats_per_primary_shot=(
                    max_content_beats_per_primary_shot
                ),
                generation_unit_capacities_per_beat=list(
                    primary_shot_layout[
                        "generation_action_unit_capacities"
                    ]
                ),
                duration_scaled_event_plan=duration_scaled_event_plan,
                timeline_layout_binding=timeline_layout_binding,
            )
        if skeleton is None:
            skeleton_kwargs = {
                "max_generation_units_per_beat": primary_shot_layout[
                    "max_generation_action_units_per_primary_shot"
                ],
                "generation_unit_capacities_per_beat": list(
                    primary_shot_layout[
                        "generation_action_unit_capacities"
                    ]
                ),
                "duration_scaled_event_plan": duration_scaled_event_plan,
                "timeline_layout_binding": timeline_layout_binding,
            }
            skeleton = _build_beat_skeleton(
                production_events,
                characters_summary,
                material_duration,
                effective_shot_duration,
                max_shots,
                director_plan,
                **skeleton_kwargs,
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

        _inherit_event_semantics(
            shots,
            production_events,
            characters,
            director_plan,
        )
        for shot in shots:
            apply_camera_motion_contract(shot)
        from quality.shot_continuity import annotate_boundaries

        annotate_boundaries(shots)
        normalize_shot_durations(
            shots,
            material_duration,
            capability_profile,
            max_content_beats_per_primary_shot=(
                max_content_beats_per_primary_shot
            ),
        )
        planned_allocations = primary_shot_layout.get(
            "story_duration_allocations_s"
        )
        if (
            not isinstance(planned_allocations, list)
            or len(planned_allocations) != len(shots)
        ):
            raise ValueError("primary-shot layout does not match production shots")
        for shot, duration in zip(shots, planned_allocations, strict=True):
            shot["suggested_duration"] = duration
            shot["duration_allocation"] = {
                "method": "primary_shot_layout",
                "layout_schema": PRIMARY_SHOT_LAYOUT_SCHEMA,
                "shot_policy": shot_policy,
            }
        _validate_beat_material_duration(
            shots,
            production_events,
            material_duration,
            capability_profile,
        )

        screenplay_plan, reconciled_capacity_plan = _build_screenplay_plan(
            events,
            shots,
            capacity_plan,
            target_duration=material_duration,
            source_events_hash=source_events_hash,
            capabilities=capability_profile,
            production_events=production_events,
            duration_scaled_event_plan=duration_scaled_event_plan,
            primary_shot_layout=primary_shot_layout,
            source_action_timeline=source_action_timeline,
            production_action_timeline=production_action_timeline,
            timeline_layout_binding=timeline_layout_binding,
        )
        screenplay_plan_sha256 = hashlib.sha256(
            json.dumps(
                screenplay_plan,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if checkpoint_dir is not None:
            _atomic_write_json(
                checkpoint_dir / "SCREENPLAY_PLAN.json",
                screenplay_plan,
            )

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
            "planned_delivery_duration": material_duration,
            "delivery_ceiling_duration": capacity_plan[
                "delivery_ceiling_duration"
            ],
            "delivery_overrun_ratio": delivery_overrun_ratio,
            "max_material_padding_ratio": max_material_padding_ratio,
            "material_duration": material_duration,
            "capacity_plan": reconciled_capacity_plan,
            "screenplay_rewrite": rewrite_layout,
            "shot_policy": shot_policy,
            "primary_shot_layout": primary_shot_layout,
            "source_action_timeline": source_action_timeline,
            "production_action_timeline": production_action_timeline,
            "timeline_layout_binding": timeline_layout_binding,
            "screenplay_plan": screenplay_plan,
            "duration_scaled_event_plan": duration_scaled_event_plan,
            "screenplay_plan_sha256": screenplay_plan_sha256,
            "estimated_shots": len(shots),
            "requested_shot_duration": shot_duration,
            "effective_shot_duration": effective_shot_duration,
            "total_duration": total_duration,
            "shot_language_plan": shot_language_plan,
            "strategy": skeleton.get("strategy", ""),
            "shots": shots,
        }

    return _run_layered_adaptation()


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
        "--shot-policy",
        choices=SHOT_POLICIES,
        default=DEFAULT_SHOT_POLICY,
        help="一级分镜策略，默认 continuity",
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
            shot_policy=args.shot_policy,
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

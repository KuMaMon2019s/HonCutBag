#!/usr/bin/env python3
"""
分镜生成器 - Phase 1 编剧引擎的最后一个模块

将 adaptation_engine.py 输出的 shot 列表转化为 orchestrator.py 可消费的 STORYBOARD.json。
这是 Phase 1 → Phase 4 的结构化分镜桥梁。

输入：
- shots JSON（adaptation_engine.py 输出）
- characters JSON（character_discoverer.py 输出，用于 first_frame 路径）

输出：
- STORYBOARD.json（与 orchestrator.py 的 load_storyboard() / parse_shots() 兼容）

逻辑：
1. 读取 shots + characters
2. 对每个 shot，调用 LLM：
   - 将中文 visual 描述翻译/转化为英文 Seedance prompt（加电影感关键词）
   - 生成中文字幕（简短，≤15字）
3. 计算 caption_frames（30fps，字幕占中间 80%）
4. 组装 STORYBOARD.json
5. 验证：确保 orchestrator.py 的 parse_shots() 能正确解析
"""

import json
import sys
import os
import argparse
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import APIConnectionError, OpenAI

from prompt.eight_layer_summary import build_subject_summary
from utils.config import ToolPaths
from utils.pipeline_config import load_config
from utils.visual_style_spec import VisualStyle, parse_visual_style
from utils.ark_llm import call_llm_stream, create_ark_client


def _load_default_visual_style(
    visual_style_path: Optional[str] = None,
) -> VisualStyle:
    """Load an override or the default HonCut visual style."""
    default_path = ToolPaths.PROMPTS_DIR / "default_visual_style.md"
    # ToolPaths historically points at pipeline/src/prompts; the portable
    # prompt assets live in pipeline/prompts.
    bundled_path = Path(__file__).resolve().parents[3] / "prompts" / "default_visual_style.md"
    style_path = Path(visual_style_path) if visual_style_path else default_path
    if not visual_style_path and not style_path.exists():
        style_path = bundled_path
    if style_path.exists():
        return parse_visual_style(style_path.read_text(encoding="utf-8"))
    return VisualStyle(
        name="fallback",
        style_prompt_full="cinematic, warm tones",
    )


# ─── LLM 配置 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是 AI 视频生成 prompt 专家。{style_directive}\n"
    "将中文场景描述转化为英文 Seedance 视频生成 prompt。\n\n"
    "【输出结构】三段式，严格按以下顺序：\n"
    "1. 【画面】（主干，最长）：完整保留场景描述中的所有视觉元素——主体、动作、空间关系、物件细节。\n"
    "   角色外貌不写具体描述，改为参考图绑定语句：\n"
    "   'Based on the reference image of {角色名}, maintain consistent: face features, hairstyle, costume details.'\n"
    "2. 【光影】（独立段）：光源方向、色调倾向、明暗关系。\n"
    "   雨天/阴天：漫射冷光，无主光源，灰蓝主调，空气潮湿感，低饱和度。\n"
    "   傍晚：暖调侧逆光，斜射余晖，长影拉伸。\n"
    "   夜间：窗光冷蓝，室内暖点光源，冷暖对比。\n"
    "3. 【风格】（最短）：固定锚定词 + 画质锁定。\n"
    "   {quality_anchor}\n"
    "   画质：Ultra-sharp 4K, high detail, natural sharpness, no subtitles, no watermark.\n\n"
    "【情绪→面容映射】\n"
    "心动/欣喜：嘴角微扬，眼底笑意，眼神明亮 → subtle smile, bright eyes\n"
    "悲伤/失落：面色沉静，眼神黯淡 → calm face, dim eyes\n"
    "温柔/深情：神情柔和，眉目温润 → gentle expression, warm gaze\n"
    "惊讶/震惊：神情微愣，目光骤聚 → stunned expression, wide eyes\n"
    "紧张/慌乱：眼神飘忽，目光四顾 → nervous gaze, restless eyes\n"
    "隐忍/克制：神情内敛，唇线收紧 → restrained expression, tight lips\n\n"
    "同时生成一句中文字幕（≤15字）。\n"
    "输出严格 JSON：{\"prompt\": \"英文prompt\", \"caption\": \"中文字幕\"}\n"
    "不要输出任何解释文字。\n\n"
    "【Identity Anchor 规则】\n"
    "角色身份只由 who 和角色参考图绑定；不要把发型、服装、体态重复写入 visual。\n"
    "不要使用代词代替出场角色主名；who=[] 时不得引入人物或人形主体。\n"
)


def _render_system_prompt(visual_style_text: Optional[str] = None) -> str:
    """Render the style branch while preserving the original output contract."""
    if visual_style_text:
        style_directive = f"严格遵循项目美术风格：{visual_style_text}，禁止替换为真人写实摄影风格。"
        quality_anchor = f"必加：{visual_style_text}；cinematic quality, ultra-fine detail, strong contrast."
    else:
        # [LEGACY-KEEP 2026-08-09] 无项目风格时保留原真人写实默认行为。
        style_directive = "专精真人写实摄影风格。"
        quality_anchor = (
            "必加：Photorealistic cinematography, cinematic quality, ultra-fine detail, "
            "strong contrast, delicate skin texture, detailed facial rendering, "
            "strand-by-strand hair detail, modern urban aesthetic, oriental temperament."
        )
    return SYSTEM_PROMPT.replace("{style_directive}", style_directive).replace(
        "{quality_anchor}", quality_anchor
    )

USER_PROMPT_TEMPLATE = (
    "场景：{visual}\n"
    "角色：{who}\n"
    "情绪：{emotion}\n"
    "地点：{where}\n"
    "角色参考图绑定：{ref_binding}\n"
    "风格/情绪/光影提示：{style_suffix}\n\n"
    "要求：\n"
    "0. 角色字段只允许引用 CHARACTERS.json 中已有的角色主名；群体、群众和背景元素不是角色条目\n"
    "1. 【画面】段完整保留场景描述的所有视觉元素，角色用参考图绑定语句替代外貌描述\n"
    "2. 【光影】段根据地点和时间推导光影（雨天=漫射冷光灰蓝调，傍晚=暖调侧逆光）\n"
    "3. 【风格】段使用固定锚定词\n"
    "4. 三段合并为一个连贯的英文 prompt，不用中文标签\n\n"
    '输出 JSON：{{"prompt": "英文视频生成prompt", "caption": "中文字幕"}}'
)

LLM_TIMEOUT = 360  # 健康长流可能超过 180s；空闲停滞由独立阈值处理
LLM_IDLE_TIMEOUT = 75
MAX_RETRIES = 3  # 解析失败重试次数（从 1 提高到 3）
SHOT_WALL_CLOCK_S = 900  # 可容纳两次完整长流调用、退避及解析开销
FPS = 30  # 帧率

IDENTITY_LOCK_PHRASES = [
    "the same character",
    "consistent across all shots",
    "maintain exact appearance from reference image",
    "no deformation, no drift, no face morph",
    "Do not alter clothing category or primary color",
]

CAMERA_OPENERS = {
    "establishing": "Wide establishing shot, slow cinematic push-in, authored lighting and rendering style.",
    "close_up": "Medium close-up, subtle handheld motion, shallow depth of field in the authored rendering style.",
    "action": "Dynamic tracking shot, authored lighting, crisp subject detail.",
    "reaction": "One continuous shot, natural head movement, no cuts, no zoom.",
    "transition": "Slow pan across scene, authored lighting and atmosphere.",
    "atmosphere": "Wide aerial shot, slow drift, authored time-of-day lighting.",
}

CAMERA_NEGATIONS = {
    "static": "no camera movement, locked tripod, no pan, no tilt, no zoom",
    "slow_pan": "no zoom, no cuts, smooth slow pan only",
    "tracking": "no zoom, no cuts, smooth tracking only",
    "handheld": "no zoom, no cuts, natural handheld movement",
}

INTENT_TO_CAMERA = {
    "establishing": "slow_pan",
    "transition": "slow_pan",
    "reveal": "tracking",
    "emotional": "static",
    "action": "tracking",
    "atmosphere": "slow_pan",
    "reaction": "handheld",
}

SHOT_SIZE_MAP = {
    "establishing": "Establishing wide shot",
    "extreme_wide": "Extreme wide shot",
    "wide": "Wide shot",
    "full": "Full shot",
    "medium_wide": "Medium wide shot",
    "medium": "Medium shot",
    "medium_close_up": "Medium close-up",
    "medium_close": "Medium close-up",
    "close_up": "Close-up",
    "extreme_close_up": "Extreme close-up",
}

CAMERA_TERMS = {
    "dolly_in": "推进(dolly in)",
    "dolly_out": "拉出(dolly out)",
    "pan_left": "左摇(pan left)",
    "pan_right": "右摇(pan right)",
    "slow_pan": "左摇(pan left)",
    "tracking": "跟拍(tracking shot)",
    "tracking_shot": "跟拍(tracking shot)",
    "tracking_left": "向左跟拍(tracking left)",
    "tracking_right": "向右跟拍(tracking right)",
    "orbit": "环绕(orbit)",
    "handheld": "手持(handheld)",
    "static": "固定(fixed/locked)",
    "fixed": "固定(fixed/locked)",
    "crane_up": "上升(crane up)",
    "crane_down": "下降(crane down)",
    "push_in": "推入(push in)",
    "whip_pan": "甩镜(whip-pan)",
    "rack_focus": "焦点转移(rack focus)",
    "steadicam": "稳定器跟拍(steadicam tracking)",
}

QUALITY_GUARDRAILS = (
    "变形扭曲(warping)，形态渐变(morphing)，面部扭曲(distorted faces)，"
    "多余手指(extra fingers)，模糊纹理(blurry textures)，"
    "抖动运动(jittery motion)，伪影(artifacts)"
)


def _remove_fast_motion_words(text: str) -> str:
    """Remove blur-prone style phrases without slowing narrative actions."""
    cleaned = re.sub(r"(?i)\bfast\s+motion\b", "clear continuous motion", str(text))
    return cleaned.replace("快速运动镜头", "连续清晰运动镜头")


# ─── LLM 客户端 ─────────────────────────────────────────────────────────────

def estimate_shot_duration(word_count: int) -> float:
    """Estimate shot duration from word count using video-toolkit formula."""
    return math.ceil(word_count / 2.5) + 2


def _get_client() -> OpenAI:
    """
    创建 OpenAI 客户端

    API Key: ARK_AGENT_API_KEY (火山方舟 Agent Plan)
    """
    return create_ark_client(read_timeout=LLM_IDLE_TIMEOUT)


def _call_llm(user_prompt: str, visual_style_text: Optional[str] = None) -> str:
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

    return call_llm_stream(
        messages=[
            {"role": "system", "content": _render_system_prompt(visual_style_text)},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=16000,
        wall_timeout=LLM_TIMEOUT,
        idle_timeout=LLM_IDLE_TIMEOUT,
        _client=client,
    )


def _parse_llm_response(response: str) -> Dict[str, str]:
    """
    解析 LLM 响应为 {"prompt": "...", "caption": "..."}

    Args:
        response: LLM 原始响应字符串

    Returns:
        包含 prompt 和 caption 的字典

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

    if "prompt" not in parsed:
        raise ValueError("缺少 'prompt' 字段")
    if "caption" not in parsed:
        raise ValueError("缺少 'caption' 字段")

    return {"prompt": parsed["prompt"], "caption": parsed["caption"]}


# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def _calculate_caption_frames(duration: int) -> str:
    """
    计算字幕帧范围（30fps，字幕占中间 80%）

    Args:
        duration: 镜头时长（秒）

    Returns:
        帧范围字符串，如 "30-180"
    """
    total_frames = duration * FPS
    # 字幕占中间 80%
    start_frame = int(total_frames * 0.1)
    end_frame = int(total_frames * 0.9)
    # 确保至少 1 帧
    start_frame = max(1, start_frame)
    end_frame = max(start_frame + 1, end_frame)
    return f"{start_frame}-{end_frame}"


def _build_characters_map(characters: Optional[List[Dict[str, Any]]]) -> Dict[str, str]:
    """
    构建角色名 → first_frame 路径的映射

    Args:
        characters: 角色列表（character_discoverer.py 输出）

    Returns:
        字典：角色名 → 角色卡声明的人脸参考图路径
    """
    if not characters:
        return {}

    char_map = {}
    for char in characters:
        char_id = char.get("id", "")
        name = char.get("name", "")
        face_reference = char.get("face_reference") or "face_closeup.png"
        # 优先使用 id，其次使用 name
        if char_id:
            reference_path = f"characters/{char_id}/{face_reference}"
            char_map[name] = reference_path
            # 也把别名映射上
            for alias in char.get("aliases", []):
                char_map[alias] = reference_path
        elif name:
            # 没有 id 时用 name 做路径
            safe_name = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '_', name.lower())
            char_map[name] = f"characters/{safe_name}/{face_reference}"

    return char_map


def _get_first_frame_for_shot(
    shot: Dict[str, Any],
    characters_map: Dict[str, str],
) -> Optional[str]:
    """
    为 shot 确定 first_frame 路径

    如果 shot 中有 who 字段（角色列表），取第一个角色的人脸参考图。

    Args:
        shot: shot 字典（含 who 字段）
        characters_map: 角色名 → 路径映射

    Returns:
        first_frame 路径字符串，或 None
    """
    who = shot.get("who", [])
    if not who or not characters_map:
        return None

    # 取第一个角色
    first_char = who[0] if isinstance(who, list) else str(who)
    return characters_map.get(first_char)


def _build_shot_prompt_legacy(
    shot: Dict[str, Any],
    characters: Optional[List[Dict[str, Any]]] = None,
    scene_style_map: Optional[Dict[str, str]] = None,
    prev_shot: Optional[Dict[str, Any]] = None,
    visual_style_path: Optional[str] = None,
) -> str:
    """
    为单个 shot 构建 LLM user prompt

    Args:
        shot: shot 字典
        characters: 角色列表（可选）
        scene_style_map: 场景级风格后缀映射（可选，P1-B 同场景共享视觉参数）

    Returns:
        格式化后的 user prompt
    """
    visual = shot.get("visual", "")
    who_list = shot.get("who", [])
    who = ", ".join(who_list) if isinstance(who_list, list) else str(who_list)
    emotion = shot.get("emotion", "")
    where = shot.get("where", "")

    # Select deterministic camera language and persist it for STORYBOARD.json.
    intent = str(shot.get("shot_intent") or "establishing").lower()
    camera = INTENT_TO_CAMERA.get(intent, "slow_pan")
    previous_camera = prev_shot.get("camera_movement") if prev_shot else None
    if previous_camera == camera:
        camera = next(
            alternative
            for alternative in ("slow_pan", "tracking", "static")
            if alternative != camera
        )
    shot["camera_movement"] = camera

    shot_size = str(shot.get("shot_size") or "medium").lower()
    framing = SHOT_SIZE_MAP.get(shot_size, "Medium shot")
    opener_key = intent if intent in CAMERA_OPENERS else "establishing"
    if shot_size in {"medium_close_up", "close_up", "extreme_close_up"} and intent not in {
        "action", "reaction"
    }:
        opener_key = "close_up"
    camera_desc = CAMERA_OPENERS[opener_key]
    camera_negation = CAMERA_NEGATIONS[camera]

    # Build one verbatim identity-lock block per on-screen character. Aliases
    # resolve to the canonical character record, but never use appearance.summary.
    characters_map: Dict[str, Dict[str, Any]] = {}
    for character in characters or []:
        name = character.get("name", "")
        if name:
            characters_map[name] = character
            for alias in character.get("aliases", []):
                characters_map[alias] = character

    identity_blocks = []
    shot_who = who_list if isinstance(who_list, list) else [who_list]
    for requested_name in filter(None, shot_who):
        character = characters_map.get(requested_name)
        if not character:
            continue
        appearance = character.get("appearance") or {}
        features = [
            str(appearance.get(field, "")).strip()
            for field in ("hair", "face", "clothing", "distinguishing")
            if appearance.get(field)
        ]
        if not features:
            continue
        canonical_name = character.get("name") or requested_name
        char_id = character.get("id")
        face_reference = character.get("face_reference") or "face_closeup.png"
        reference_path = (
            f"characters/{char_id}/{face_reference}"
            if char_id
            else f"{canonical_name}_ref.png"
        )
        identity_blocks.append(
            f"[reference_image: {reference_path}]\n"
            "[identity_lock]\n"
            f"{canonical_name}: {IDENTITY_LOCK_PHRASES[0]} — {', '.join(features)} — "
            f"{IDENTITY_LOCK_PHRASES[1]}; {IDENTITY_LOCK_PHRASES[2]}; "
            f"{IDENTITY_LOCK_PHRASES[3]}. {IDENTITY_LOCK_PHRASES[4]}."
        )
    identity_block = "\n".join(identity_blocks)

    # --- P1-B2: 追加场景级共享视觉参数（同场景镜头共享 Layer 3-5）---
    if scene_style_map:
        scene_suffix = scene_style_map.get(shot.get("where", ""), "")
        if scene_suffix and scene_suffix not in visual:
            visual = visual + " " + scene_suffix

    # Build reference binding for characters in this shot
    ref_parts = []
    if characters:
        for char in characters:
            char_name = char.get("name", "")
            if char_name in who or any(alias in who for alias in char.get("aliases", [])):
                ref_parts.append(
                    f"Based on the reference image of {char_name}, "
                    f"maintain consistent: face features, hairstyle, costume details."
                )
    ref_binding = " ".join(ref_parts) if ref_parts else "No character reference."
    
    visual_style = _load_default_visual_style(visual_style_path)

    # Add emotion and style enrichment. The historical emotion helper owns a
    # photorealistic anchor, so a declared project style must replace that
    # anchor instead of being appended after a contradictory instruction.
    try:
        from prompt.emotion_mapping import build_style_suffix
        style_suffix = build_style_suffix(emotion=emotion, scene=where)
    except ImportError:
        style_suffix = ""
    if visual_style_path and (
        visual_style.style_prompt_short or visual_style.style_prompt_full
    ):
        style_suffix = (
            visual_style.style_prompt_short or visual_style.style_prompt_full
        )
    
    scene_suffix = scene_style_map.get(where, "") if scene_style_map else ""
    lighting = shot.get("lighting_key") or "natural"
    action = shot.get("what") or visual
    style = (
        visual_style.style_prompt_short
        or visual_style.style_prompt_full
        or "cinematic narrative rendering"
    )
    audio = "Ambient natural sound, no music."
    eight_part_prompt = (
        f"{framing}. {camera_desc} Camera movement: {camera}; {camera_negation}.\n"
        f"{identity_block + chr(10) if identity_block else ''}"
        f"Action: {action}.\nSetting: {where}. {scene_suffix}\n"
        f"Lighting: {lighting} lighting.\nStyle: {style}\nAudio: {audio}"
    )

    prompt = USER_PROMPT_TEMPLATE.format(
        visual=eight_part_prompt,
        who=who,
        emotion=emotion,
        where=where,
        ref_binding=ref_binding,
        style_suffix=style_suffix,
    )

    # Append the portable design system after identity-lock and camera rules.
    if visual_style.style_prompt_full:
        prompt = f"{prompt}\n\nVisual style: {visual_style.style_prompt_full}"
    shot.setdefault("speech_duration_s", estimate_shot_duration(len(prompt.split())))
    return prompt


def _concrete_subject_description(
    shot: Dict[str, Any], characters: Optional[List[Dict[str, Any]]]
) -> str:
    """Return names plus at most three stable, drawable appearance traits."""
    requested = shot.get("who", [])
    requested = requested if isinstance(requested, list) else [requested]
    descriptions = []
    for requested_name in filter(None, requested):
        match = next(
            (
                char for char in characters or []
                if requested_name == char.get("name")
                or requested_name in char.get("aliases", [])
            ),
            None,
        )
        appearance = (match or {}).get("appearance", {})
        traits = [
            str(appearance.get(key, "")).strip()
            for key in ("hair", "face", "clothing", "distinguishing")
            if appearance.get(key)
        ][:3]
        descriptions.append(f"{requested_name}—{'，'.join(traits)}" if traits else str(requested_name))
    return "；".join(descriptions) or str(shot.get("subject_description") or "场景主体")


def _specific_lighting(
    shot: Dict[str, Any], where: str, visual_style: VisualStyle
) -> str:
    lighting = str(shot.get("lighting_description") or shot.get("lighting_key") or "").strip()
    if lighting and any(token in lighting for token in ("左", "右", "上", "下", "逆光", "侧光")):
        if any(token in lighting.upper() for token in ("K", "暖", "冷")):
            if not any(token in lighting for token in ("气氛", "氛围", "雾", "雨", "尘", "颗粒", "潮湿")):
                lighting += "，空气颗粒轻微可见，气氛与剧情情绪一致"
            return lighting
    if any(token in where for token in ("夜", "月")):
        return "冷蓝月光从镜头右上方射入，色温5600K，暖橙环境光轻微补亮轮廓，气氛克制"
    if any(token in where for token in ("室内", "房", "店", "办公室")):
        return "暖白LED主光从镜头左上方照射，色温4200K，右侧冷色窗光勾勒轮廓，气氛沉静"

    style_text = " ".join(filter(None, (
        visual_style.style_prompt_short,
        visual_style.style_prompt_full,
        " ".join(visual_style.mood_keywords),
        " ".join(visual_style.tags),
    ))).lower()
    has_rain = any(token in style_text for token in ("雨", "rain", "storm", "暴风"))
    has_night = any(token in style_text for token in ("夜", "night", "moon", "月"))
    if has_rain and has_night:
        return "冷蓝雨夜光从镜头右上方漫射照入，湿润表面反射微光，低饱和度，气氛湿冷压抑"
    if has_night:
        return "冷蓝夜间主光从镜头右上方照入，微弱环境光勾勒轮廓，气氛与全片美术风格一致"
    if has_rain or any(token in style_text for token in ("阴", "overcast", "cloudy")):
        return "阴雨天漫射冷光从上方均匀落下，低饱和度，空气潮湿，气氛与全片美术风格一致"
    if any(token in style_text for token in ("黄金时段", "golden hour", "日落", "sunset", "黄昏", "dusk")):
        return "黄金时段暖调主光从镜头左上方侧逆光照射，斜射余晖拉出长影，气氛与全片美术风格一致"
    if any(token in style_text for token in ("黎明", "清晨", "dawn", "sunrise")):
        return "清晨柔和自然光从镜头左上方斜射，薄雾中轮廓清晰，气氛与全片美术风格一致"
    return "与全片美术风格一致的自然光照，主光从镜头左上方照射，明暗关系真实克制"


def _build_eight_layer_prompt(
    shot: Dict[str, Any],
    characters: Optional[List[Dict[str, Any]]] = None,
    scene_style_map: Optional[Dict[str, str]] = None,
    prev_shot: Optional[Dict[str, Any]] = None,
    visual_style_path: Optional[str] = None,
) -> str:
    """Build the deterministic eight-layer blueprint consumed by the LLM."""
    shot_number = shot.get("shot_number") or shot.get("shot_order") or shot.get("id") or 1
    try:
        shot_number = int(str(shot_number).lstrip("Ss"))
    except (TypeError, ValueError):
        shot_number = 1

    requested = shot.get("who", [])
    requested = requested if isinstance(requested, list) else [requested]
    references = []
    for char in characters or []:
        if char.get("name") not in requested and not set(char.get("aliases", [])).intersection(requested):
            continue
        char_id = char.get("id") or char.get("name")
        ref = char.get("face_reference") or f"characters/{char_id}/face_closeup.png"
        # 文件名只保留在注释中供 Phase 1 人工排查，不进入 LLM 提示词。
        shot.setdefault("reference_debug_files", []).append(str(ref))
        references.append(f"参考{{图片N}}中的{char.get('name')}作为主体，保持身份与服装一致")

    intent = str(shot.get("shot_intent") or "establishing").lower()
    generation_actions = [
        str(value).strip()
        for value in (shot.get("generation_actions") or [])
        if str(value).strip()
    ]
    duration = float(shot.get("suggested_duration", shot.get("duration", 5)))
    aspect_ratio = str(shot.get("aspect_ratio") or "16:9")
    declared_camera = str(shot.get("camera_movement") or "").strip().lower()
    camera_key = declared_camera or INTENT_TO_CAMERA.get(intent, "slow_pan")
    if declared_camera in {"", "unspecified"}:
        if generation_actions or intent == "action":
            camera_key = "steadicam"
        elif duration >= 4:
            camera_key = "dolly_in"
    shot["camera_movement"] = camera_key
    camera = CAMERA_TERMS.get(camera_key, "固定(fixed/locked)")
    framing = SHOT_SIZE_MAP.get(str(shot.get("shot_size") or "medium").lower(), "Medium shot")
    subject = _concrete_subject_description(shot, characters)
    emotion = str(shot.get("emotion") or "")
    source_action = (
        " → ".join(generation_actions)
        or shot.get("action_description")
        or shot.get("script_excerpt")
        or shot.get("source_excerpt")
    )
    if source_action and len(str(source_action)) > 20:
        action = str(source_action)[:120]
        supplemental = str(shot.get("what") or "").strip()
        # generation_actions is already the paid model's complete executable
        # contract. Re-appending the broad plot summary reintroduces omitted
        # choreography ("continuous attacks", "fight fiercely") and defeats
        # the duration budget.
        if not generation_actions and supplemental and supplemental not in action:
            action = f"{action}；补充：{supplemental}"
    else:
        action = str(source_action or shot.get("what") or shot.get("visual") or "保持自然姿态")
    where = str(shot.get("where") or "当前场景")
    scene_suffix = (scene_style_map or {}).get(where, "")
    visual_style = _load_default_visual_style(visual_style_path)
    lighting = _specific_lighting(shot, where, visual_style)
    texture_keywords = [
        str(value).strip()
        for value in (shot.get("texture_keywords") or [])
        if str(value).strip()
    ]
    audio = str(shot.get("audio") or shot.get("sound") or "环境底噪与动作同期声")
    style_anchor = visual_style.style_prompt_short or visual_style.style_prompt_full or "电影叙事风格，35mm胶片质感"
    motion_contract = (
        "动作执行契约：整镜连续发生可见位移；按起始动作→核心接触/变化→明确结果依次完成。"
        "人物不得在首帧姿态原地停留，不得只让头发、雨水、蒸汽或镜头轻微漂移来冒充动作。"
        "保持指定景别，不得无动机切成特写或让交互角色离开画面；双人动作的起始与接触时刻必须同时看见双方。"
        if generation_actions or intent == "action"
        else
        "动态画面契约：主体保持自然表演，环境和摄影机持续产生可见变化；"
        "禁止把首帧做成仅有雨水、头发或呼吸轻动的静态动图。"
    )

    layers = []
    if references:
        layers.append("元素参考声明：" + "；".join(references))
    subject_layers = [
        ("景别与主体：", f"{framing}，{subject}"),
        ("运镜：", camera),
        ("场景与光影：", f"{where}，{scene_suffix}，{lighting}".replace("，，", "，")),
    ]
    if texture_keywords:
        subject_layers.append(("材质纹理：", "、".join(texture_keywords)))
    subject_summary = build_subject_summary(subject_layers)
    layers.extend([
        f"镜头{shot_number}：",
        f"主体总结：{subject_summary}",
        f"动作：{action}",
        f"运动契约：{motion_contract}",
    ])
    if shot.get("hero_moment"):
        layers.append(
            "视觉峰值：这是全片 hero moment，构图、动作结果与环境层次必须形成清晰视觉峰值。"
        )
    layers.extend([
        f"音效：{audio}",
        f"全局收尾：{style_anchor}；约束词：{QUALITY_GUARDRAILS}；4K，{aspect_ratio}，{duration}秒",
    ])
    blueprint = "\n".join(layers)
    blueprint = _remove_fast_motion_words(blueprint)
    shot["eight_layer_prompt"] = blueprint
    shot.setdefault("speech_duration_s", estimate_shot_duration(len(blueprint.split())))
    return USER_PROMPT_TEMPLATE.format(
        visual=blueprint,
        who=", ".join(map(str, requested)),
        emotion=emotion,
        where=where,
        ref_binding="；".join(references) if references else "No character reference.",
        style_suffix=style_anchor,
    )


def _build_shot_prompt(
    shot: Dict[str, Any],
    characters: Optional[List[Dict[str, Any]]] = None,
    scene_style_map: Optional[Dict[str, str]] = None,
    prev_shot: Optional[Dict[str, Any]] = None,
    visual_style_path: Optional[str] = None,
) -> str:
    """Build shot prompt using the eight-layer framework."""
    return _build_eight_layer_prompt(
        shot, characters, scene_style_map, prev_shot, visual_style_path
    )


# ─── 核心函数 ────────────────────────────────────────────────────────────────

def _generate_single_shot(
    shot: Dict[str, Any],
    index: int,
    total: int,
    characters: Optional[List[Dict[str, Any]]] = None,
    visual_style_text: Optional[str] = None,
    scene_style_map: Optional[Dict[str, str]] = None,
    previous_shot: Optional[Dict[str, Any]] = None,
    visual_style_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate and assemble one shot; shared by serial and arq paths."""
    print(f"Shot {index}/{total} 开始生成...")
    shot_deadline = time.monotonic() + SHOT_WALL_CLOCK_S
    if shot.get("source_excerpt") and not shot.get("action_description"):
        shot["action_description"] = shot["source_excerpt"]
    duration = shot.get("suggested_duration", 5)
    visual_style = _load_default_visual_style(visual_style_path)
    fallback_style = (
        visual_style.style_prompt_short
        or visual_style.style_prompt_full
        or "cinematic narrative rendering"
    )
    user_prompt = _build_shot_prompt(
        shot, characters, scene_style_map=scene_style_map,
        prev_shot=previous_shot, visual_style_path=visual_style_path,
    )
    llm_result = None
    last_error = "unknown error"
    for attempt in range(1 + MAX_RETRIES):
        if time.monotonic() + LLM_TIMEOUT > shot_deadline:
            print(f"Shot {index} 总时限 {SHOT_WALL_CLOCK_S}s 到，使用降级方案", file=sys.stderr)
            llm_result = {
                "prompt": f"{fallback_style}, {shot.get('visual', 'scene')}, natural lighting",
                "caption": shot.get("what", ""),
            }
            break
        try:
            prompt = user_prompt if attempt == 0 else user_prompt + f"\n\n[重试反馈] 上次失败原因: {last_error}。请确保输出有效的 JSON 格式。"
            llm_result = _parse_llm_response(_call_llm(prompt, visual_style_text))
            break
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                print(f"Shot {index} LLM 解析失败，重试中（{attempt + 1}/{MAX_RETRIES}）: {exc}", file=sys.stderr)
                time.sleep(1)
            else:
                print(f"Shot {index} LLM 解析失败，使用降级方案: {exc}", file=sys.stderr)
        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                print(f"Shot {index} LLM 调用失败，重试中（{attempt + 1}/{MAX_RETRIES}）: {exc}", file=sys.stderr)
                time.sleep(min(15 * (attempt + 1), 60) if isinstance(exc, (APIConnectionError, TimeoutError, ConnectionError)) else 1)
            else:
                print(f"Shot {index} LLM 调用失败，使用降级方案: {exc}", file=sys.stderr)
        if attempt == MAX_RETRIES:
            llm_result = {
                "prompt": f"{fallback_style}, {shot.get('visual', 'scene')}, natural lighting",
                "caption": shot.get("what", ""),
            }

    if llm_result is None:
        llm_result = {
            "prompt": f"{fallback_style}, {shot.get('visual', 'scene')}, natural lighting",
            "caption": shot.get("what", ""),
        }
    prompt_blueprint = user_prompt.partition("场景：")[2].partition("\n角色：")[0].strip()
    if prompt_blueprint:
        llm_result["prompt"] = f"{prompt_blueprint}\n{llm_result['prompt']}".strip()
    if visual_style.style_prompt_full and visual_style.style_prompt_full not in llm_result["prompt"]:
        llm_result["prompt"] = f"{llm_result['prompt']}\n\nVisual style: {visual_style.style_prompt_full}"
    llm_result["prompt"] = _remove_fast_motion_words(llm_result["prompt"])
    print(f"Shot {index}/{total} ✅ 完成")

    characters_map = _build_characters_map(characters)
    first_frame = _get_first_frame_for_shot(shot, characters_map)
    shot_name = shot.get("what", f"镜头 {index}")
    if len(shot_name) > 30:
        shot_name = shot_name[:30] + "..."
    result = {
        "id": index, "name": shot_name, "duration": duration,
        "prompt": llm_result["prompt"], "caption": llm_result["caption"],
        "caption_frames": _calculate_caption_frames(duration),
        "dialogue": shot.get("dialogue"), "visual": shot.get("visual", ""),
        "what": shot.get("what", ""),
    }
    for field in ("narration", "voiceover"):
        if field in shot:
            result[field] = shot[field]
    if first_frame:
        result["first_frame"] = first_frame
    who = shot.get("who", [])
    result["who"] = who if isinstance(who, list) else ([str(who)] if who else [])
    defaults = {
        "shot_type": shot.get("shot_type") or shot.get("shot_size"),
        "subject_description": _concrete_subject_description(shot, characters),
        "action_description": shot.get("action_description") or shot.get("what") or shot.get("visual"),
        "camera_movement_en": str(shot.get("camera_movement") or "fixed").replace("_", " "),
        "lighting_description": _specific_lighting(shot, str(shot.get("where") or ""), visual_style),
    }
    for field, value in defaults.items():
        if value:
            result[field] = value
    for field in (
        "shot_size", "camera_movement", "lighting_key", "shot_intent",
        "hero_moment", "texture_keywords",
        "gen_strategy", "where", "audio", "sound", "emotion",
        "transition_to_next", "boundary_before", "continuity_reason",
        "continuity_subject", "screen_direction", "tracking_prompt",
        "source_excerpt", "source_sequence_ids", "source_action_unit_ids",
        "source_events", "source_event_slices", "source_event_roles",
        "micro_actions", "generation_actions", "generation_action_units",
        "generation_action_categories",
        "generation_load", "start_state", "end_state", "causal_link",
        "speaker_attribution",
        "aspect_ratio", "width", "height",
    ):
        if field in {
            "generation_action_units",
            "generation_action_categories",
            "hero_moment",
            "texture_keywords",
        }:
            if field in shot:
                result[field] = shot[field]
        elif shot.get(field):
            result[field] = shot[field]
    assets = shot.get("associate_assets", [])
    if assets:
        result["associate_assets"] = assets if isinstance(assets, list) else [assets]
    return result

def generate_storyboard(
    shots: List[Dict[str, Any]],
    characters: Optional[List[Dict[str, Any]]] = None,
    title: str = "未命名项目",
    visual_style_path: Optional[str] = None,
    visual_style_text: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    audio_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    将 shot 列表转化为 STORYBOARD.json 格式

    Args:
        shots: shot 列表（adaptation_engine.py 输出）
        characters: 角色列表（character_discoverer.py 输出，可选）
        title: 项目标题
        config: 管线配置；支持 audio_enabled 或 audio.enabled
        config_path: 可选 YAML/JSON 管线配置路径
        audio_enabled: 显式音频开关，优先于配置

    Returns:
        STORYBOARD.json 格式的字典

    Raises:
        RuntimeError: LLM 调用失败
        ValueError: 输入无效
    """
    if not shots:
        raise ValueError("shot 列表为空，无法生成分镜")

    pipeline_config = config if config is not None else load_config(config_path)
    if not isinstance(pipeline_config, dict):
        pipeline_config = {}
    layout_config = pipeline_config.get("layout", {})
    if not isinstance(layout_config, dict):
        layout_config = {}
    configured_aspect = str(
        pipeline_config.get("aspect_ratio")
        or layout_config.get("aspect_ratio")
        or ""
    ).strip()
    configured_width = int(
        pipeline_config.get("width") or layout_config.get("width") or 0
    )
    configured_height = int(
        pipeline_config.get("height") or layout_config.get("height") or 0
    )
    if configured_aspect:
        for shot in shots:
            shot.setdefault("aspect_ratio", configured_aspect)
            if configured_width > 0:
                shot.setdefault("width", configured_width)
            if configured_height > 0:
                shot.setdefault("height", configured_height)

    # --- P1-B1: 同场景共享视觉参数（参考 HonCut 五层镜头构建）---
    # 同场景镜头共享 Layer 3-5（Subject/Lighting/Style），只有 Layer 1-2（Camera/Movement）随镜头变
    scene_style_map = {}  # {where: style_suffix}
    for shot in shots:
        where = shot.get("where", "")
        if where and where not in scene_style_map:
            emotion = shot.get("emotion", "")
            # 基于场景首个镜头的情绪生成场景级风格后缀
            try:
                from prompt.emotion_mapping import build_style_suffix
                scene_style_map[where] = build_style_suffix(emotion=emotion, scene=where)
            except Exception:
                scene_style_map[where] = ""

    # 计算总时长
    total_duration = sum(shot.get("suggested_duration", 5) for shot in shots)

    total = len(shots)
    if os.environ.get("HONCUT_SHOT_QUEUE") == "1":
        from utils.shot_queue import make_payload, run_shot_queue

        inferred_dir = Path(visual_style_path).parent if visual_style_path else (Path(config_path).parent if config_path else Path.cwd())
        output_dir = Path(pipeline_config.get("output_dir", inferred_dir))
        run_tag = str(pipeline_config.get("run_tag") or output_dir.name)
        payloads = [
            make_payload(
                shot, i, total, characters=characters,
                visual_style_text=visual_style_text, scene_style_map=scene_style_map,
                previous_shot=shots[i - 2] if i > 1 else None,
                visual_style_path=visual_style_path,
            )
            for i, shot in enumerate(shots, 1)
        ]
        storyboard_shots = run_shot_queue(
            payloads, run_tag=run_tag, partial_path=output_dir / "shots_partial.json"
        )
    else:
        def generate_one(item):
            i, shot = item
            return _generate_single_shot(
                shot, i, total, characters, visual_style_text, scene_style_map,
                shots[i - 2] if i > 1 else None, visual_style_path,
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            storyboard_shots = list(executor.map(generate_one, enumerate(shots, 1)))

    # 组装完整 STORYBOARD
    audio_config = pipeline_config.get("audio", {})
    if not isinstance(audio_config, dict):
        audio_config = {}
    configured_enabled = audio_config.get(
        "enabled", pipeline_config.get("audio_enabled", True)
    )
    enabled = audio_enabled if audio_enabled is not None else configured_enabled
    if not isinstance(enabled, bool):
        enabled = True
    tts_enabled = audio_config.get("tts", True)
    if not isinstance(tts_enabled, bool):
        tts_enabled = True

    storyboard = {
        "title": title,
        "total_shots": len(storyboard_shots),
        "target_duration": total_duration,
        "aspect_ratio": configured_aspect or str(
            next((shot.get("aspect_ratio") for shot in storyboard_shots if shot.get("aspect_ratio")), "16:9")
        ),
        "width": configured_width or next(
            (shot.get("width") for shot in storyboard_shots if shot.get("width")), None
        ),
        "height": configured_height or next(
            (shot.get("height") for shot in storyboard_shots if shot.get("height")), None
        ),
        "static_reference_images": {},
        "audio": {
            "enabled": bool(enabled),
            "tts": tts_enabled,
        },
        "shots": storyboard_shots,
    }

    return storyboard


def validate_storyboard(storyboard: Dict[str, Any]) -> List[str]:
    """
    验证 STORYBOARD 是否兼容 orchestrator.py 的 parse_shots()

    Args:
        storyboard: STORYBOARD 字典

    Returns:
        错误列表（空列表表示通过验证）
    """
    errors = []

    # 检查顶层结构
    if "shots" not in storyboard:
        errors.append("缺少 'shots' 数组")
        return errors

    if not isinstance(storyboard["shots"], list):
        errors.append("'shots' 应为数组")
        return errors

    # 检查每个 shot 的必要字段（与 orchestrator.parse_shots() 对齐）
    required_fields = {"id", "name", "prompt"}
    for i, shot in enumerate(storyboard["shots"]):
        if not isinstance(shot, dict):
            errors.append(f"第 {i+1} 个 shot 不是字典")
            continue
        missing = required_fields - set(shot.keys())
        if missing:
            errors.append(f"第 {i+1} 个 shot 缺少字段: {missing}")

        # 检查 id 是否为整数
        if "id" in shot and not isinstance(shot["id"], int):
            errors.append(f"第 {i+1} 个 shot 的 id 应为整数，得到 {type(shot['id']).__name__}")

        # 检查 duration 是否为正数
        if "duration" in shot:
            dur = shot["duration"]
            if not isinstance(dur, (int, float)) or dur <= 0:
                errors.append(f"第 {i+1} 个 shot 的 duration 应为正数，得到 {dur}")

        # 检查 prompt 是否为非空字符串
        if "prompt" in shot and not isinstance(shot["prompt"], str):
            errors.append(f"第 {i+1} 个 shot 的 prompt 应为字符串")

    return errors


# ─── CLI 入口 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="分镜生成器 - 将 shot 列表转化为 STORYBOARD.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python storyboard_generator.py --shots shots.json --characters characters.json --output STORYBOARD.json
  python storyboard_generator.py --shots shots.json --output STORYBOARD.json  # 无角色也行
  cat shots.json | python storyboard_generator.py --output STORYBOARD.json
        """,
    )

    parser.add_argument(
        "--shots", "-s",
        type=str,
        help="shots JSON 文件路径（adaptation_engine.py 输出）",
    )

    parser.add_argument(
        "--characters", "-c",
        type=str,
        help="角色 JSON 文件路径（character_discoverer.py 输出，可选）",
    )

    parser.add_argument(
        "--title", "-t",
        type=str,
        default="未命名项目",
        help="项目标题（默认：未命名项目）",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出 JSON 文件路径（可选，默认打印到 stdout）",
    )

    parser.add_argument(
        "--visual-style",
        type=str,
        help="visual-style.md 覆盖路径（可选）",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="管线 YAML/JSON 配置路径（可选）",
    )

    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="禁用 Phase 9 音频处理",
    )

    args = parser.parse_args()

    # ── 读取 shots ────────────────────────────────────────────────────────
    shots_data = None

    if args.shots:
        try:
            with open(args.shots, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"错误：文件不存在 - {args.shots}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"错误：JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        # 支持直接传入 shots 列表或包含 shots 的字典
        if isinstance(data, list):
            shots_data = data
        elif isinstance(data, dict) and "shots" in data:
            shots_data = data["shots"]
            # 如果字典中有 target_duration，用它作为 title 的参考
            if "strategy" in data and args.title == "未命名项目":
                args.title = f"视频项目 ({len(shots_data)} 镜头)"
        else:
            print("错误：输入格式不支持，期望 shots 列表或包含 shots 的字典", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty():
        # 从 stdin 读取（管道模式）
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"错误：stdin JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        if isinstance(data, list):
            shots_data = data
        elif isinstance(data, dict) and "shots" in data:
            shots_data = data["shots"]
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

    # ── 生成分镜 ──────────────────────────────────────────────────────────
    print(f"🎬 分镜生成器 — 处理 {len(shots_data)} 个镜头...", file=sys.stderr)

    try:
        storyboard = generate_storyboard(
            shots=shots_data,
            characters=characters,
            title=args.title,
            visual_style_path=args.visual_style,
            config_path=args.config,
            audio_enabled=False if args.no_audio else None,
        )
    except (ValueError, RuntimeError) as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    # ── 验证 ──────────────────────────────────────────────────────────────
    errors = validate_storyboard(storyboard)
    if errors:
        print("⚠️  验证发现以下问题：", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print("输出仍会生成，但下游可能解析失败。", file=sys.stderr)
    else:
        print("✅ 验证通过：与 orchestrator.py parse_shots() 兼容", file=sys.stderr)

    # ── 输出 JSON ─────────────────────────────────────────────────────────
    json_output = json.dumps(storyboard, ensure_ascii=False, indent=2)
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

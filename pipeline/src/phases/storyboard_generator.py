#!/usr/bin/env python3
"""
分镜生成器 - Phase 2 事件图谱引擎的最后一个模块

将 adaptation_engine.py 输出的 shot 列表转化为 orchestrator.py 可消费的 STORYBOARD.json。
这是 Phase 2 → Phase 4 的桥梁。

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
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI

from utils.config import ToolPaths
from utils.visual_style_spec import VisualStyle, parse_visual_style


def _load_default_visual_style(
    visual_style_path: Optional[str] = None,
) -> VisualStyle:
    """Load an override or the default HonCut visual style."""
    default_path = ToolPaths.PROMPTS_DIR / "default_visual_style.md"
    # ToolPaths historically points at pipeline/src/prompts; the portable
    # prompt assets live in pipeline/prompts.
    bundled_path = Path(__file__).resolve().parents[2] / "prompts" / "default_visual_style.md"
    style_path = Path(visual_style_path) if visual_style_path else default_path
    if not visual_style_path and not style_path.exists():
        style_path = bundled_path
    if style_path.exists():
        return parse_visual_style(style_path.read_text(encoding="utf-8"))
    return VisualStyle(
        name="fallback",
        style_prompt_full="cinematic, warm tones, 16:9",
    )


# ─── LLM 配置 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是 AI 视频生成 prompt 专家，专精真人写实摄影风格。\n"
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
    "   必加：Photorealistic cinematography, cinematic quality, ultra-fine detail, "
    "strong contrast, delicate skin texture, detailed facial rendering, "
    "strand-by-strand hair detail, modern urban aesthetic, oriental temperament.\n"
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
    "每个镜头的 visual 描述中，角色必须用「角色名 — 视觉特征」格式开头。\n"
    "禁止使用代词（he/she/该角色/同上/此人），必须每次写出角色名+特征。\n"
    "示例：'林夏 — 黑色长直发及肩, 白色修身衬衫, 深蓝西装裤 — 站在便利店门口...'\n"
)

USER_PROMPT_TEMPLATE = (
    "场景：{visual}\n"
    "角色：{who}\n"
    "情绪：{emotion}\n"
    "地点：{where}\n"
    "角色参考图绑定：{ref_binding}\n"
    "风格/情绪/光影提示：{style_suffix}\n\n"
    "要求：\n"
    "1. 【画面】段完整保留场景描述的所有视觉元素，角色用参考图绑定语句替代外貌描述\n"
    "2. 【光影】段根据地点和时间推导光影（雨天=漫射冷光灰蓝调，傍晚=暖调侧逆光）\n"
    "3. 【风格】段使用固定锚定词\n"
    "4. 三段合并为一个连贯的英文 prompt，不用中文标签\n\n"
    '输出 JSON：{{"prompt": "英文视频生成prompt", "caption": "中文字幕"}}'
)

LLM_TIMEOUT = 60  # 秒
MAX_RETRIES = 3  # 解析失败重试次数（从 1 提高到 3）
FPS = 30  # 帧率

IDENTITY_LOCK_PHRASES = [
    "the same character",
    "consistent across all shots",
    "maintain exact appearance from reference image",
    "no deformation, no drift, no face morph",
    "Do not alter clothing category or primary color",
]

CAMERA_OPENERS = {
    "establishing": "Wide establishing shot, slow cinematic push-in, cinematic lighting, photorealistic, 35mm film quality.",
    "close_up": "Medium close-up, subtle handheld motion, shallow depth of field, photorealistic, 35mm film.",
    "action": "Dynamic tracking shot, cinematic lighting, photorealistic, 35mm film quality, crisp subject detail.",
    "reaction": "One continuous shot, natural head movement, photorealistic, 35mm film grain, no cuts, no zoom.",
    "transition": "Slow pan across scene, cinematic lighting, photorealistic, volumetric haze, 35mm film.",
    "atmosphere": "Wide aerial shot, slow drift, golden hour lighting, photorealistic, 35mm film quality.",
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
    "extreme_wide": "Extreme wide shot",
    "wide": "Wide shot",
    "full": "Full shot",
    "medium_wide": "Medium wide shot",
    "medium": "Medium shot",
    "medium_close_up": "Medium close-up",
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
    "orbit": "环绕(orbit)",
    "handheld": "手持(handheld)",
    "static": "固定(fixed/locked)",
    "fixed": "固定(fixed/locked)",
    "crane_up": "上升(crane up)",
    "crane_down": "下降(crane down)",
    "push_in": "推入(push in)",
    "whip_pan": "甩镜(whip-pan)",
    "rack_focus": "焦点转移(rack focus)",
}

EMOTION_ACTIONS = {
    "悲伤": "低头，肩膀微颤，眼眶泛红，手指攥紧衣角",
    "喜悦": "嘴角上扬，眉眼舒展，脚步轻快",
    "紧张": "频繁看手表，手指敲击桌面，呼吸急促，眼神闪躲",
    "愤怒": "双拳紧握，下颌紧绷，胸口起伏",
}

QUALITY_GUARDRAILS = (
    "变形扭曲(warping)，形态渐变(morphing)，面部扭曲(distorted faces)，"
    "多余手指(extra fingers)，模糊纹理(blurry textures)，"
    "抖动运动(jittery motion)，伪影(artifacts)"
)


def _remove_fast_motion_words(text: str) -> str:
    """Remove blur-prone speed words without corrupting words like breakfast."""
    return re.sub(r"(?i)\bfast\b", "smooth", str(text)).replace("快速", "平稳")


# ─── LLM 客户端 ─────────────────────────────────────────────────────────────

def estimate_shot_duration(word_count: int) -> float:
    """Estimate shot duration from word count using video-toolkit formula."""
    return math.ceil(word_count / 2.5) + 2


def _get_client() -> OpenAI:
    """
    创建 OpenAI 客户端

    API Key: ARK_AGENT_API_KEY (火山方舟 Agent Plan)
    """
    api_key = (
        os.environ.get("ARK_AGENT_API_KEY")
        
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        print("错误：环境变量 ARK_AGENT_API_KEY 未设置（火山方舟 Agent Plan）", file=sys.stderr)
        sys.exit(1)

    return OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/plan/v3",
    )


def _call_llm(user_prompt: str) -> str:
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

    response = client.chat.completions.create(
        model="doubao-seed-2.0-lite",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        timeout=LLM_TIMEOUT,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM 返回空内容")
    return content


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
        字典：角色名 → "characters/{id}/front.png"
    """
    if not characters:
        return {}

    char_map = {}
    for char in characters:
        char_id = char.get("id", "")
        name = char.get("name", "")
        # 优先使用 id，其次使用 name
        if char_id:
            char_map[name] = f"characters/{char_id}/front.png"
            # 也把别名映射上
            for alias in char.get("aliases", []):
                char_map[alias] = f"characters/{char_id}/front.png"
        elif name:
            # 没有 id 时用 name 做路径
            safe_name = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '_', name.lower())
            char_map[name] = f"characters/{safe_name}/front.png"

    return char_map


def _get_first_frame_for_shot(
    shot: Dict[str, Any],
    characters_map: Dict[str, str],
) -> Optional[str]:
    """
    为 shot 确定 first_frame 路径

    如果 shot 中有 who 字段（角色列表），取第一个角色的 front.png。

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
        reference_path = f"characters/{char_id}/front.png" if char_id else f"{canonical_name}_ref.png"
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
    
    # Add emotion and style enrichment
    try:
        from prompt.emotion_mapping import build_style_suffix
        style_suffix = build_style_suffix(emotion=emotion, scene=where)
    except ImportError:
        style_suffix = ""
    
    scene_suffix = scene_style_map.get(where, "") if scene_style_map else ""
    lighting = shot.get("lighting_key") or "natural"
    action = shot.get("what") or visual
    style = "Photorealistic, cinematic, 35mm film quality, no 3D, no cartoon, no VFX aesthetic."
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
    visual_style = _load_default_visual_style(visual_style_path)
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


def _specific_lighting(shot: Dict[str, Any], where: str) -> str:
    lighting = str(shot.get("lighting_description") or shot.get("lighting_key") or "").strip()
    if lighting and any(token in lighting for token in ("左", "右", "上", "下", "逆光", "侧光")):
        if any(token in lighting.upper() for token in ("K", "暖", "冷")):
            return lighting
    if any(token in where for token in ("夜", "月", "室外")):
        return "冷蓝月光从镜头右上方射入，色温5600K，暖橙环境光轻微补亮轮廓，气氛克制"
    if any(token in where for token in ("室内", "房", "店", "办公室")):
        return "暖白LED主光从镜头左上方照射，色温4200K，右侧冷色窗光勾勒轮廓，气氛沉静"
    return "黄金时段主光从镜头左上方侧逆光照射，色温4800K，空气颗粒可见，气氛真实"


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
        references.append(f"参考图{ref}中的{char.get('name')}作为主体，保持身份与服装一致")

    intent = str(shot.get("shot_intent") or "establishing").lower()
    camera_key = str(shot.get("camera_movement") or INTENT_TO_CAMERA.get(intent, "slow_pan")).lower()
    if prev_shot and prev_shot.get("camera_movement") == camera_key:
        camera_key = "fixed"
    shot["camera_movement"] = camera_key
    camera = CAMERA_TERMS.get(camera_key, "固定(fixed/locked)")
    framing = SHOT_SIZE_MAP.get(str(shot.get("shot_size") or "medium").lower(), "Medium shot")
    subject = _concrete_subject_description(shot, characters)
    emotion = str(shot.get("emotion") or "")
    action = str(shot.get("action_description") or shot.get("what") or shot.get("visual") or "保持自然姿态")
    externalized = next((value for key, value in EMOTION_ACTIONS.items() if key in emotion), "")
    if externalized and externalized not in action:
        action = f"{action}，{externalized}"
    where = str(shot.get("where") or "当前场景")
    scene_suffix = (scene_style_map or {}).get(where, "")
    lighting = _specific_lighting(shot, where)
    audio = str(shot.get("audio") or shot.get("sound") or "环境底噪与动作同期声")
    visual_style = _load_default_visual_style(visual_style_path)
    style_anchor = visual_style.style_prompt_short or visual_style.style_prompt_full or "电影叙事风格，35mm胶片质感"
    duration = shot.get("duration", 5)

    layers = []
    if references:
        layers.append("元素参考声明：" + "；".join(references))
    layers.extend([
        f"镜头{shot_number}：",
        f"{framing}，{subject}",
        f"动作：{action}",
        f"运镜：{camera}",
        f"场景与光影：{where}，{scene_suffix}，{lighting}".replace("，，", "，"),
        f"音效：{audio}",
        f"全局收尾：{style_anchor}；约束词：{QUALITY_GUARDRAILS}；4K，16:9，{duration}秒",
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

def generate_storyboard(
    shots: List[Dict[str, Any]],
    characters: Optional[List[Dict[str, Any]]] = None,
    title: str = "未命名项目",
    visual_style_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    将 shot 列表转化为 STORYBOARD.json 格式

    Args:
        shots: shot 列表（adaptation_engine.py 输出）
        characters: 角色列表（character_discoverer.py 输出，可选）
        title: 项目标题

    Returns:
        STORYBOARD.json 格式的字典

    Raises:
        RuntimeError: LLM 调用失败
        ValueError: 输入无效
    """
    if not shots:
        raise ValueError("shot 列表为空，无法生成分镜")

    # 构建角色映射
    characters_map = _build_characters_map(characters)

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

    # 处理每个 shot
    storyboard_shots = []
    for i, shot in enumerate(shots, 1):
        duration = shot.get("suggested_duration", 5)

        # 调用 LLM 生成英文 prompt 和中文 caption
        previous_shot = shots[i - 2] if i > 1 else None
        user_prompt = _build_shot_prompt(
            shot,
            characters,
            scene_style_map=scene_style_map,
            prev_shot=previous_shot,
            visual_style_path=visual_style_path,
        )
        llm_result = None

        for attempt in range(1 + MAX_RETRIES):
            try:
                # 如果是重试，注入质量反馈到 prompt
                if attempt > 0:
                    feedback_prompt = user_prompt + f"\n\n[重试反馈] 上次失败原因: {last_error}。请确保输出有效的 JSON 格式。"
                    response = _call_llm(feedback_prompt)
                else:
                    response = _call_llm(user_prompt)
                
                llm_result = _parse_llm_response(response)
                break
            except (json.JSONDecodeError, ValueError) as e:
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    print(f"Shot {i} LLM 解析失败，重试中（{attempt + 1}/{MAX_RETRIES}）: {e}", file=sys.stderr)
                    time.sleep(1)
                else:
                    print(f"Shot {i} LLM 解析失败，使用降级方案: {e}", file=sys.stderr)
                    # 降级：直接用 visual 作为 prompt，空 caption
                    llm_result = {
                        "prompt": f"Cinematic shot, {shot.get('visual', 'scene')}, natural lighting, film grain",
                        "caption": shot.get("what", ""),
                    }
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    print(f"Shot {i} LLM 调用失败，重试中（{attempt + 1}/{MAX_RETRIES}）: {e}", file=sys.stderr)
                    time.sleep(1)
                else:
                    print(f"Shot {i} LLM 调用失败，使用降级方案: {e}", file=sys.stderr)
                    llm_result = {
                        "prompt": f"Cinematic shot, {shot.get('visual', 'scene')}, natural lighting, film grain",
                        "caption": shot.get("what", ""),
                    }

        if llm_result is None:
            llm_result = {
                "prompt": f"Cinematic shot, {shot.get('visual', 'scene')}, natural lighting, film grain",
                "caption": shot.get("what", ""),
            }

        # Identity and camera constraints are deterministic safety rails. Keep
        # them in the final Seedance prompt even if the translation model omits
        # or paraphrases part of the requested structure.
        prompt_blueprint = user_prompt.partition("场景：")[2].partition("\n角色：")[0].strip()
        if prompt_blueprint:
            llm_result["prompt"] = f"{prompt_blueprint}\n{llm_result['prompt']}".strip()

        # Preserve the king field verbatim in the final Seedance prompt even
        # when the translation model paraphrases or omits style instructions.
        visual_style = _load_default_visual_style(visual_style_path)
        if (
            visual_style.style_prompt_full
            and visual_style.style_prompt_full not in llm_result["prompt"]
        ):
            llm_result["prompt"] = (
                f"{llm_result['prompt']}\n\nVisual style: "
                f"{visual_style.style_prompt_full}"
            )
        llm_result["prompt"] = _remove_fast_motion_words(llm_result["prompt"])

        # 确定 first_frame
        first_frame = _get_first_frame_for_shot(shot, characters_map)

        # 计算 caption_frames
        caption_frames = _calculate_caption_frames(duration)

        # 构建 shot 名称
        shot_name = shot.get("what", f"镜头 {i}")
        if len(shot_name) > 30:
            shot_name = shot_name[:30] + "..."

        # 组装 storyboard shot
        storyboard_shot = {
            "id": i,
            "name": shot_name,
            "duration": duration,
            "prompt": llm_result["prompt"],
            "caption": llm_result["caption"],
            "caption_frames": caption_frames,
            # Retained for deterministic route diagnostics and FLF2V end-state prompts.
            "visual": shot.get("visual", ""),
            "what": shot.get("what", ""),
        }

        # 透传结构化字段（who/shot_size/camera_movement/lighting_key/shot_intent/associate_assets）
        # 这些字段从 adaptation_engine 的 LLM 输出传递到 STORYBOARD.json，供 M2 和 Phase 6 消费
        if first_frame:
            storyboard_shot["first_frame"] = first_frame

        # who: 出场角色列表（空数组 = 纯风景镜头）
        who_list = shot.get("who", [])
        if isinstance(who_list, list):
            storyboard_shot["who"] = who_list
        else:
            storyboard_shot["who"] = [str(who_list)] if who_list else []

        # Structured fields used by review and generation routing.
        structured_defaults = {
            "shot_type": shot.get("shot_type") or shot.get("shot_size"),
            "subject_description": _concrete_subject_description(shot, characters),
            "action_description": shot.get("action_description") or shot.get("what") or shot.get("visual"),
            "camera_movement_en": str(shot.get("camera_movement") or "fixed").replace("_", " "),
            "lighting_description": _specific_lighting(shot, str(shot.get("where") or "")),
        }
        for field, value in structured_defaults.items():
            if value:
                storyboard_shot[field] = value
        for field in ("shot_size", "camera_movement", "lighting_key", "shot_intent", "gen_strategy", "where", "audio", "sound"):
            val = shot.get(field)
            if val:
                storyboard_shot[field] = val

        # associate_assets: 资产绑定
        aa = shot.get("associate_assets", [])
        if aa:
            storyboard_shot["associate_assets"] = aa if isinstance(aa, list) else [aa]

        storyboard_shots.append(storyboard_shot)

    # 组装完整 STORYBOARD
    storyboard = {
        "title": title,
        "total_shots": len(storyboard_shots),
        "target_duration": total_duration,
        "static_reference_images": {},
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

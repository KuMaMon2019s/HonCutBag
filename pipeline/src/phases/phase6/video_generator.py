"""Asset-aware Phase 6 video generation routing."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from clients.video_client import VideoClient
from prompt.eight_layer_summary import build_subject_summary
from utils.style_slices import get_slice


BASE_NEGATIVE_PROMPT = (
    "变形扭曲(warping), 形态渐变(morphing), 面部扭曲(distorted faces), "
    "多余手指(extra fingers), 模糊纹理(blurry textures), "
    "抖动运动(jittery motion), 伪影(artifacts)"
)


def _time_continuity_contract(*values: object) -> tuple[str, str]:
    """Return a positive continuity lock and its contradictory visual guardrails."""
    text = " ".join(str(value) for value in values if value).lower()
    is_night = any(token in text for token in ("夜", "night", "moon", "月光"))
    is_day = any(token in text for token in ("白天", "daytime", "daylight", "midday", "正午"))
    if is_night and not is_day:
        return (
            "整个镜头从第一帧到最后一帧始终保持深夜，不得渐变为白天或黎明，"
            "天空和环境不得出现日光",
            "白天(daytime), 日光(daylight), 晴空(clear sky), 明亮天空(bright sky), "
            "灰白日间天空(overcast daylight), 清晨(dawn), 日出(sunrise)",
        )
    if is_day and not is_night:
        return (
            "整个镜头从第一帧到最后一帧始终保持日间光照，不得渐变为夜景",
            "深夜(deep night), 月光(moonlight), 纯夜景(night scene)",
        )
    return "", ""

CAMERA_MOVEMENTS = {
    "dolly_in": "推进(dolly in)", "dolly_out": "拉出(dolly out)",
    "pan_left": "左摇(pan left)", "pan_right": "右摇(pan right)",
    "slow_pan": "左摇(pan left)", "tracking": "跟拍(tracking shot)",
    "tracking_shot": "跟拍(tracking shot)", "orbit": "环绕(orbit)",
    "tracking_left": "向左跟拍(tracking left)",
    "tracking_right": "向右跟拍(tracking right)",
    "handheld": "手持(handheld)", "static": "固定(fixed/locked)",
    "fixed": "固定(fixed/locked)", "crane_up": "上升(crane up)",
    "crane_down": "下降(crane down)", "push_in": "推入(push in)",
    "whip_pan": "甩镜(whip-pan)", "rack_focus": "焦点转移(rack focus)",
}


def _characters_list(characters: Any) -> list[dict[str, Any]]:
    if isinstance(characters, dict):
        return list(characters.get("characters", []))
    return list(characters or [])


def _shot_number(shot_meta: dict[str, Any]) -> int:
    raw = (
        shot_meta.get("shot_number")
        or shot_meta.get("shot_order")
        or shot_meta.get("id")
        or shot_meta.get("shot_id")
        or 1
    )
    try:
        return int(str(raw).lstrip("Ss"))
    except (TypeError, ValueError):
        return 1


def build_video_prompt(
    shot_meta: dict[str, Any],
    characters: Any,
    scene_consistency: dict[str, Any],
    model: str,
) -> str | dict[str, str]:
    """Assemble the eight layers and apply Seedance/Kling guardrail routing."""
    number = _shot_number(shot_meta)
    scene = scene_consistency.get("shots", {}).get(f"S{number:02d}", {})
    requested_declared = "who" in shot_meta or "characters" in shot_meta
    requested = shot_meta.get("who") or shot_meta.get("characters") or []
    requested = requested if isinstance(requested, list) else [requested]
    explicit_scenery = requested_declared and not requested
    selected = []
    for character in _characters_list(characters):
        matches_requested = (
            character.get("name") in requested
            or character.get("id") in requested
            or bool(set(character.get("aliases", [])).intersection(requested))
        )
        # An explicit who=[] is a hard scenery contract, not shorthand for
        # "select every known character". Legacy metadata with no character
        # field at all retains the historical all-character fallback.
        if matches_requested or (not requested_declared and not requested):
            selected.append(character)

    parts = []
    definitions = [str(char.get("prompt_definition", "")).strip() for char in selected]
    definitions = [definition for definition in definitions if definition]
    if definitions:
        parts.append("元素参考声明：" + "；".join(definitions))
    parts.append(f"镜头{number}：")
    shot_type = shot_meta.get("shot_type") or shot_meta.get("shot_size") or "中景"
    subject = shot_meta.get("subject_description")
    if not subject:
        traits = []
        for character in selected:
            appearance = character.get("appearance", {})
            stable = [str(appearance.get(key, "")).strip() for key in ("hair", "face", "clothing") if appearance.get(key)][:3]
            traits.append(f"{character.get('name')}—{'，'.join(stable)}")
        subject = "；".join(traits) or shot_meta.get("visual") or "场景主体"
    action = str(shot_meta.get("action_description") or shot_meta.get("what") or "保持自然姿态")
    camera_key = str(shot_meta.get("camera_movement") or "fixed").lower()
    camera = CAMERA_MOVEMENTS.get(camera_key, str(shot_meta.get("camera_movement") or "固定(fixed/locked)"))
    layout = scene.get("spatial_layout", {})
    setting = scene.get("scene_description") or shot_meta.get("where") or "当前场景"
    lighting = scene.get("lighting_description") or scene.get("lighting_note") or scene_consistency.get("global_lighting") or "与全片美术风格一致的自然光照，明暗关系真实克制"
    scene_and_lighting = f"{setting}，{layout.get('subject', '')}，{lighting}".replace("，，", "，")
    subject_summary = build_subject_summary([
        ("景别与主体：", f"{shot_type}，{subject}"),
        ("动作：", action),
        ("运镜：", camera),
        ("场景与光影：", scene_and_lighting),
    ])
    parts.append(f"主体总结：{subject_summary}")
    audio = shot_meta.get("audio") or shot_meta.get("sound")
    if audio:
        parts.append(f"音效：{audio}")
    # The per-shot storyboard frame already carries the project's visual style.
    # Project-level summaries often contain plot nouns (characters, palaces,
    # props, future locations). Repeating that prose in every prompt caused a
    # cloud-only opening shot to invent a fairy and a palace.
    style = (
        "仅继承该镜头参考分镜帧的渲染方式、色彩科学、光影质感、材质真实度、"
        "镜头语言与氛围；不得从项目级风格描述引入本镜头动作与视觉契约未明确列出的"
        "人物、建筑、地点、道具或剧情元素"
    )
    style = get_slice(style, "video")
    quality = scene.get("quality_suffix") or f"4K, 16:9, {shot_meta.get('duration', 5)}秒"
    time_lock, time_negative = _time_continuity_contract(
        shot_meta.get("time_of_day"),
        shot_meta.get("time"),
        lighting,
        style,
        scene_consistency.get("global_style_lock"),
    )
    if time_lock:
        # This is intentionally outside build_subject_summary's character budget.
        parts.append(f"时空连续性硬约束：{time_lock}")
    # Layer 8 is appended after the bounded summary and is never truncated.
    parts.append(f"全局收尾：{style}；{quality}")

    negatives = [BASE_NEGATIVE_PROMPT, str(scene.get("negative_prompt", "")).strip()]
    if explicit_scenery:
        negatives.append(
            "人物(people), 人形主体(humanoid figures), 角色(characters), "
            "服装(costumes), 未明确列出的建筑或道具(unlisted architecture or props)"
        )
    if time_negative:
        negatives.append(time_negative)
    negatives.extend(str(char.get("negative_guardrails", "")).strip() for char in selected)
    negative_prompt = ", ".join(dict.fromkeys(item for item in negatives if item))
    prompt = "。".join(parts)
    prompt = re.sub(r"(?i)\bfast\b", "smooth", prompt).replace("快速", "平稳")
    if explicit_scenery:
        scenery_lock = (
            "纯环境镜头硬约束：画面中保持零人物、零人形主体、零服装与零角色道具，"
            "只呈现本镜头主体总结和动作中明确列出的环境元素"
        )
        return f"{prompt}。{scenery_lock}。约束条件：{negative_prompt}"
    fictional_decl = "虚拟形象声明：片中角色均为 AI 生成的虚构角色，非真实人物"
    if "kling" in model.lower():
        return {"prompt": f"{fictional_decl}。{prompt}", "negative_prompt": negative_prompt}
    identity_lock = "identity-lock：保持参考图中的面部骨骼、发型、服装类别与主色不变"
    return f"{prompt}。{fictional_decl}。{identity_lock}。约束条件：{negative_prompt}"


def _load_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _read_state(state: Any, name: str, default: Any = None) -> Any:
    if isinstance(state, MutableMapping):
        return state.get(name, default)
    return getattr(state, name, default)


def _write_state(state: Any, name: str, value: Any) -> None:
    if isinstance(state, MutableMapping):
        state[name] = value
    else:
        setattr(state, name, value)


class VideoGenerator:
    """Generate Phase 6 video, preferring Bridge assets when available."""

    def __init__(self, video_client: VideoClient | None = None) -> None:
        self.video_client = video_client or VideoClient()

    async def run(self, state: Any) -> Any:
        asset_id = _read_state(state, "character_asset_id") or _read_state(state, "asset_id")
        prompt = _read_state(state, "prompt", "")
        negative_prompt = None
        output_dir = Path(_read_state(state, "output_dir", "."))
        shot_meta = _read_state(state, "shot_meta") or _read_state(state, "shot")
        characters = _read_state(state, "characters") or _load_json(output_dir / "CHARACTERS.json", {})
        scene_contract = _read_state(state, "scene_consistency") or _load_json(output_dir / "SCENE_CONSISTENCY.json", {})
        model = str(_read_state(state, "model") or "seedance")
        if shot_meta and scene_contract:
            routed = build_video_prompt(shot_meta, characters, scene_contract, model)
            if isinstance(routed, dict):
                prompt = routed["prompt"]
                negative_prompt = routed["negative_prompt"]
            else:
                prompt = routed
            _write_state(state, "assembled_prompt", routed)
        if asset_id:
            result = await self.video_client.generate_with_assets(
                asset_id=asset_id,
                asset_type="character",
                image_index=0,
                model=_read_state(state, "model") or "wan22",
                prompt=prompt,
            )
        else:
            generation_kwargs = {
                "reference_images": _read_state(state, "reference_images", []),
            }
            if negative_prompt:
                generation_kwargs["negative_prompt"] = negative_prompt
            result = self.video_client.generate(
                prompt=prompt,
                **generation_kwargs,
            )
            if inspect.isawaitable(result):
                result = await result
        _write_state(state, "video_result", result)
        return state

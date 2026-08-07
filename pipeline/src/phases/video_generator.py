"""Asset-aware Phase 5 video generation routing."""

from __future__ import annotations

import inspect
import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from clients.video_client import VideoClient


BASE_NEGATIVE_PROMPT = (
    "变形扭曲(warping), 形态渐变(morphing), 面部扭曲(distorted faces), "
    "多余手指(extra fingers), 模糊纹理(blurry textures), "
    "抖动运动(jittery motion), 伪影(artifacts)"
)

CAMERA_MOVEMENTS = {
    "dolly_in": "推进(dolly in)", "dolly_out": "拉出(dolly out)",
    "pan_left": "左摇(pan left)", "pan_right": "右摇(pan right)",
    "slow_pan": "左摇(pan left)", "tracking": "跟拍(tracking shot)",
    "tracking_shot": "跟拍(tracking shot)", "orbit": "环绕(orbit)",
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
    raw = shot_meta.get("shot_number") or shot_meta.get("shot_order") or shot_meta.get("id") or 1
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
    requested = shot_meta.get("who") or shot_meta.get("characters") or []
    requested = requested if isinstance(requested, list) else [requested]
    selected = []
    for character in _characters_list(characters):
        if not requested or character.get("name") in requested or character.get("id") in requested \
                or set(character.get("aliases", [])).intersection(requested):
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
    parts.append(f"{shot_type}，{subject}")
    parts.append("动作：" + str(shot_meta.get("action_description") or shot_meta.get("what") or "保持自然姿态"))
    camera_key = str(shot_meta.get("camera_movement") or "fixed").lower()
    camera = CAMERA_MOVEMENTS.get(camera_key, str(shot_meta.get("camera_movement") or "固定(fixed/locked)"))
    parts.append(f"运镜：{camera}")
    layout = scene.get("spatial_layout", {})
    setting = scene.get("scene_description") or shot_meta.get("where") or "当前场景"
    lighting = scene.get("lighting_description") or scene.get("lighting_note") or "主光从镜头左上方照射，色温4800K"
    parts.append(f"场景与光影：{setting}，{layout.get('subject', '')}，{lighting}".replace("，，", "，"))
    audio = shot_meta.get("audio") or shot_meta.get("sound")
    if audio:
        parts.append(f"音效：{audio}")
    style = scene.get("style_anchor") or scene.get("style_suffix") or scene_consistency.get("global_style_lock") or "电影叙事风格"
    quality = scene.get("quality_suffix") or f"4K, 16:9, {shot_meta.get('duration', 5)}秒"
    parts.append(f"全局收尾：{style}；{quality}")

    negatives = [BASE_NEGATIVE_PROMPT, str(scene.get("negative_prompt", "")).strip()]
    negatives.extend(str(char.get("negative_guardrails", "")).strip() for char in selected)
    negative_prompt = ", ".join(dict.fromkeys(item for item in negatives if item))
    prompt = "。".join(parts)
    prompt = prompt.replace("快速", "平稳").replace("fast", "smooth").replace("Fast", "Smooth")
    if "kling" in model.lower():
        return {"prompt": prompt, "negative_prompt": negative_prompt}
    identity_lock = "identity-lock：保持参考图中的面部骨骼、发型、服装类别与主色不变"
    return f"{prompt}。{identity_lock}。约束条件：{negative_prompt}"


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
    """Generate Phase 5 video, preferring Bridge assets when available."""

    def __init__(self, video_client: VideoClient | None = None) -> None:
        self.video_client = video_client or VideoClient()

    async def run(self, state: Any) -> Any:
        asset_id = _read_state(state, "character_asset_id") or _read_state(state, "asset_id")
        prompt = _read_state(state, "prompt", "")
        negative_prompt = None
        try:
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
        except Exception as exc:
            _write_state(state, "prompt_fallback_reason", str(exc))
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

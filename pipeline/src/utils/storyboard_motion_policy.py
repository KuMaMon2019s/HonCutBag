"""Shared prompt contract for storyboard motion notation."""

from __future__ import annotations

STORYBOARD_MOTION_POLICY_MARKER = "[storyboard-motion-notation]"
STORYBOARD_MOTION_POLICY = (
    f"{STORYBOARD_MOTION_POLICY_MARKER} "
    "分镜参考图中的主体动作箭头、运动轨迹线和摄影机运动箭头，都是非叙事性的制作标注。"
    "必须读取并遵循它们的运动语义：主体箭头控制主体的运动方向、路径和速度趋势，"
    "摄影机箭头控制机位的移动方向和轨迹；在不改变文字剧情动作合同及首尾状态的前提下，"
    "按这些标注完成可见运动。所有箭头和标注只用于控制生成，不属于场景。"
    "最终视频的任何一帧都不得出现或残留箭头、轨迹线、辅助线、文字提示、Pxx/Sxx 编号、"
    "分格边框、色块标记或水印，也不得把它们转化成光效、道具、HUD、UI 或字幕。"
    "Treat storyboard arrows and motion paths as control notation only: follow their motion semantics, "
    "but never render the notation itself."
)


def apply_storyboard_motion_policy(prompt: object) -> str:
    """Prepend the storyboard notation contract exactly once."""
    text = str(prompt or "").strip()
    if STORYBOARD_MOTION_POLICY_MARKER in text:
        return text
    return f"{STORYBOARD_MOTION_POLICY}\n{text}".strip()

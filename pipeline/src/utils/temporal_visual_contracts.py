"""Deterministic time-of-day contracts for image, video, and visual QA.

Broad labels such as ``day`` are weak generation instructions: a rainy or neon
style can easily pull them toward night.  This module turns authored time text
into a local clock window plus observable light requirements and contradictory
visual cues.  The clock window is a production visual lock, not a claim about
astronomical sunrise for a particular latitude or date.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping
from typing import Any


TEMPORAL_VISUAL_CONTRACT_SCHEMA = "honcut.temporal-visual.v1"

_PERIODS: dict[str, dict[str, Any]] = {
    "dawn": {
        "aliases": ("黎明", "破晓", "拂晓", "dawn", "sunrise"),
        "start": "05:30",
        "end": "07:00",
        "label": "黎明",
        "requirements": (
            "太阳接近地平线，天空由冷蓝向低角度暖光过渡",
            "环境仍可辨识且不是纯黑夜空",
        ),
        "forbidden": ("深夜黑空", "正午顶光", "霓虹独占主照明"),
    },
    "morning": {
        "aliases": ("早晨", "上午", "清晨", "morning", "early day"),
        "start": "08:00",
        "end": "10:30",
        "label": "上午",
        "requirements": (
            "太阳已升起，天空和环境保持清晰的自然日光亮度",
            "阴雨或室内场景仍须由可见窗光或日间环境光主导曝光",
        ),
        "forbidden": ("黑色或藏蓝夜空", "月亮或星空", "霓虹独占主照明"),
    },
    "midday": {
        "aliases": ("正午", "中午", "午间", "midday", "noon"),
        "start": "11:30",
        "end": "14:00",
        "label": "正午",
        "requirements": (
            "太阳明显高于地平线，环境曝光由强而稳定的日光主导",
            "天空明亮；阴雨只降低对比度，不得把环境变成夜景",
        ),
        "forbidden": ("黑色或藏蓝夜空", "月亮或星空", "夜间欠曝光", "霓虹独占主照明"),
    },
    "afternoon": {
        "aliases": ("下午", "午后", "afternoon"),
        "start": "14:00",
        "end": "17:00",
        "label": "下午",
        "requirements": (
            "太阳仍明显高于地平线，环境保持完整日光曝光",
            "可有斜射光但不得出现日落后的暗蓝夜空",
        ),
        "forbidden": ("黑色或藏蓝夜空", "月亮或星空", "纯夜景", "霓虹独占主照明"),
    },
    "golden_hour": {
        "aliases": ("黄金时段", "黄金时间", "日落前", "golden hour"),
        "start": "17:00",
        "end": "18:30",
        "label": "日落前黄金时段",
        "requirements": (
            "低角度暖色太阳仍可见或其直射余晖仍主导曝光",
            "天空保持有日光的暖亮渐变而不是夜空",
        ),
        "forbidden": ("深夜黑空", "星空", "正午顶光", "霓虹独占主照明"),
    },
    "dusk": {
        "aliases": ("黄昏", "傍晚", "暮色", "蓝调时刻", "dusk", "twilight", "blue hour", "sunset"),
        "start": "18:00",
        "end": "19:30",
        "label": "黄昏",
        "requirements": (
            "地平线保留可见暮光，天空不是完全黑色",
            "环境光由暮色与实景灯共同塑形且保持可辨识层次",
        ),
        "forbidden": ("正午顶光", "无暮光的纯黑深夜", "完全由霓虹替代天空环境光"),
    },
    "night": {
        "aliases": ("夜间", "夜晚", "深夜", "午夜", "雨夜", "夜景", "night", "midnight", "moonlight"),
        "start": "20:00",
        "end": "04:30",
        "label": "夜间",
        "requirements": (
            "太阳已落下，环境由月光、城市实景灯或人工光源主导",
            "天空与环境不得出现日间高亮曝光",
        ),
        "forbidden": (
            "白天(daytime)",
            "日光(daylight)",
            "晴空(clear sky)",
            "明亮天空(bright sky)",
            "灰白日间天空(overcast daylight)",
            "清晨(dawn)",
            "日出(sunrise)",
        ),
    },
    "day": {
        "aliases": ("日间", "白天", "昼间", "daytime", "daylight", "broad daylight"),
        "start": "10:00",
        "end": "16:00",
        "label": "日间",
        "requirements": (
            "太阳明显高于地平线，天空保持明亮的日间亮度",
            "主环境曝光必须由日光主导；阴雨只形成灰白漫射日光，不得变成夜景",
            "霓虹、警示灯和室内实景灯只能作局部辅光，不能取代日光主照明",
        ),
        "forbidden": (
            "黑色或藏蓝夜空",
            "月亮或星空",
            "纯夜景",
            "夜间欠曝光",
            "霓虹独占主照明",
        ),
    },
}

_PERIOD_PRIORITY = (
    "golden_hour",
    "dawn",
    "morning",
    "midday",
    "afternoon",
    "dusk",
    "night",
    "day",
)
_CLOCK_RANGE = re.compile(
    r"(?<!\d)([01]?\d|2[0-3])(?:[:：]([0-5]\d)|\s*[时点])"
    r"\s*(?:-|–|—|~|～|至|到)\s*"
    r"([01]?\d|2[0-3])(?:[:：]([0-5]\d)|\s*[时点])"
)


def _contains_alias(text: str, alias: str) -> bool:
    if any("\u4e00" <= char <= "\u9fff" for char in alias):
        return alias in text
    return re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text) is not None


def _period_from_text(text: str) -> str | None:
    normalized = str(text or "").casefold()
    for period in _PERIOD_PRIORITY:
        if any(_contains_alias(normalized, alias) for alias in _PERIODS[period]["aliases"]):
            return period
    return None


def _clock_range(text: str) -> tuple[str, str] | None:
    match = _CLOCK_RANGE.search(str(text or ""))
    if match is None:
        return None
    start_hour, start_minute, end_hour, end_minute = match.groups()
    return (
        f"{int(start_hour):02d}:{int(start_minute or 0):02d}",
        f"{int(end_hour):02d}:{int(end_minute or 0):02d}",
    )


def _period_from_clock(start: str, end: str) -> str:
    start_minutes = int(start[:2]) * 60 + int(start[3:])
    end_minutes = int(end[:2]) * 60 + int(end[3:])
    if end_minutes <= start_minutes:
        return "night"
    midpoint = (start_minutes + end_minutes) / 2
    if midpoint < 7 * 60:
        return "dawn"
    if midpoint < 11 * 60:
        return "morning"
    if midpoint < 14 * 60:
        return "midday"
    if midpoint < 17 * 60:
        return "afternoon"
    if midpoint < 18.5 * 60:
        return "golden_hour"
    if midpoint < 20 * 60:
        return "dusk"
    return "night"


def build_temporal_visual_contract(
    *,
    source_time: object = "",
    time_of_day: object = "",
    time_window: object = "",
    visual_context: object = "",
    lighting_context: object = "",
) -> dict[str, Any] | None:
    """Build a structured contract, prioritizing authored time over visual style."""
    authored = " ".join(
        str(value).strip()
        for value in (source_time, time_of_day, time_window)
        if str(value or "").strip()
    )
    inferred = " ".join(
        str(value).strip()
        for value in (visual_context, lighting_context)
        if str(value or "").strip()
    )
    period = _period_from_text(authored)
    source_kind = "authored_time"
    if period is None:
        period = _period_from_text(inferred)
        source_kind = "visual_context_inference"
    explicit_range = _clock_range(f"{time_window} {source_time} {time_of_day}")
    if period is None and explicit_range is not None:
        period = _period_from_clock(*explicit_range)
        source_kind = "authored_clock_window"
    if period is None:
        return None

    definition = _PERIODS[period]
    start, end = explicit_range or (definition["start"], definition["end"])
    return {
        "schema": TEMPORAL_VISUAL_CONTRACT_SCHEMA,
        "period": period,
        "label": definition["label"],
        "source_time": str(source_time or time_of_day or time_window or "").strip(),
        "source_kind": source_kind,
        "local_clock_window": {
            "start": start,
            "end": end,
            "basis": "local_scene_time_visual_lock",
        },
        "visible_light_requirements": list(definition["requirements"]),
        "forbidden_visual_cues": list(definition["forbidden"]),
        "continuity": "first_frame_through_last_frame",
    }


def normalized_temporal_visual_contract(value: object) -> dict[str, Any] | None:
    """Return a defensive copy only for a complete supported contract."""
    if not isinstance(value, Mapping):
        return None
    period = str(value.get("period") or "").strip()
    window = value.get("local_clock_window")
    if period not in _PERIODS or not isinstance(window, Mapping):
        return None
    start = str(window.get("start") or "").strip()
    end = str(window.get("end") or "").strip()
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", start) or not re.fullmatch(
        r"(?:[01]\d|2[0-3]):[0-5]\d", end
    ):
        return None
    contract = copy.deepcopy(dict(value))
    definition = _PERIODS[period]
    contract["schema"] = TEMPORAL_VISUAL_CONTRACT_SCHEMA
    contract.setdefault("label", definition["label"])
    contract.setdefault("visible_light_requirements", list(definition["requirements"]))
    contract.setdefault("forbidden_visual_cues", list(definition["forbidden"]))
    contract.setdefault("continuity", "first_frame_through_last_frame")
    return contract


def apply_temporal_visual_contract(
    shot: dict[str, Any],
    *,
    source_times: Iterable[object] = (),
) -> dict[str, Any] | None:
    """Attach a canonical time contract to one shot in place."""
    existing = normalized_temporal_visual_contract(shot.get("temporal_visual_contract"))
    if existing is not None:
        shot["temporal_visual_contract"] = existing
        return existing

    unique_sources = list(
        dict.fromkeys(
            str(value).strip() for value in source_times if str(value or "").strip()
        )
    )
    source_time = " / ".join(unique_sources) or shot.get("time") or ""
    contract = build_temporal_visual_contract(
        source_time=source_time,
        time_of_day=shot.get("time_of_day") or "",
        time_window=shot.get("time_window") or "",
        visual_context=" ".join(
            str(shot.get(key) or "") for key in ("where", "visual", "scene_description")
        ),
        lighting_context=" ".join(
            str(shot.get(key) or "") for key in ("lighting_description", "lighting_key")
        ),
    )
    if contract is None:
        return None
    shot["temporal_visual_contract"] = contract
    if not str(shot.get("time_of_day") or "").strip():
        shot["time_of_day"] = contract["period"]
    window = contract["local_clock_window"]
    shot["time_window"] = f"{window['start']}-{window['end']}"
    if unique_sources:
        shot["time"] = unique_sources[0]
        shot["source_time_values"] = unique_sources
    return contract


def temporal_visual_prompt(contract: object) -> str:
    """Render the positive, model-facing hard constraint."""
    normalized = normalized_temporal_visual_contract(contract)
    if normalized is None:
        return ""
    window = normalized["local_clock_window"]
    requirements = "；".join(normalized["visible_light_requirements"])
    continuity_prefix = (
        "整个镜头从第一帧到最后一帧始终保持深夜，不得渐变为白天或黎明；"
        if normalized["period"] == "night"
        else (
            "整个镜头从第一帧到最后一帧始终保持日间光照，不得渐变为夜景；"
            if normalized["period"] in {"day", "morning", "midday", "afternoon"}
            else ""
        )
    )
    return (
        f"{continuity_prefix}本地场景钟点锁定为 {window['start']}–{window['end']}（{normalized['label']}）；"
        "从第一帧到最后一帧不得跳出该时间段，也不得因雨、冷色、霓虹或剧情情绪变成另一时段；"
        f"可见光照证据：{requirements}"
    )


def temporal_visual_negative_prompt(contract: object) -> str:
    """Render cues that would visibly contradict the authored clock window."""
    normalized = normalized_temporal_visual_contract(contract)
    if normalized is None:
        return ""
    return "时间段冲突禁止项：" + "，".join(normalized["forbidden_visual_cues"])


def temporal_visual_qa_instruction(contract: object) -> str:
    """Render a reviewer instruction with an explicit fail condition."""
    normalized = normalized_temporal_visual_contract(contract)
    if normalized is None:
        return ""
    window = normalized["local_clock_window"]
    return (
        f"核验画面是否始终可判定为本地 {window['start']}–{window['end']}（{normalized['label']}）。"
        "若出现禁止视觉线索、日夜颠倒或首尾时间段漂移，必须判为 reshoot/revise，"
        "不能用冷色调、暴雨、霓虹或气氛风格解释为合格。"
    )

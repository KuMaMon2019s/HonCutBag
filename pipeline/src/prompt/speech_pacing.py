"""Emotion-aware Chinese dialogue timing adapted from Toonflow's storyboard rules."""
import math
import re
from typing import Any

SPEECH_RATES = {"angry": 4.0, "normal": 3.0, "sad": 2.0}
EMOTION_ALIASES = {
    "angry": ("angry", "anger", "愤怒", "急促", "争吵", "怒斥", "惊慌"),
    "sad": ("sad", "悲伤", "深情", "沉思", "低语", "虚弱", "临终", "哀悼", "回忆"),
}

def pacing_tier(emotion: str | None) -> str:
    normalized = (emotion or "").lower()
    for tier, aliases in EMOTION_ALIASES.items():
        if any(alias in normalized for alias in aliases): return tier
    return "normal"

def count_spoken_characters(dialogue: str) -> int:
    return len(re.findall(r"[\w\u3400-\u9fff]", dialogue, flags=re.UNICODE))

def estimate_speech_duration(dialogue: str, emotion: str | None = None, pause_seconds: float = .4, safety_seconds: float = 1.0) -> int:
    """Return Toonflow's ceil(chars/rate + punctuation pauses + safety)."""
    if not dialogue or dialogue.strip() in {"无台词", "none", "No dialogue"}: return 0
    rate = SPEECH_RATES[pacing_tier(emotion)]
    pauses = len(re.findall(r"[，。！？、；：,.!?;:…—]", dialogue)) * pause_seconds
    return math.ceil(count_spoken_characters(dialogue) / rate + pauses + safety_seconds)

def annotate_shot_pacing(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for shot in shots:
        dialogue = shot.get("dialogue") or shot.get("lines") or ""
        shot["speech_duration_s"] = estimate_speech_duration(str(dialogue), shot.get("emotion"))
    return shots

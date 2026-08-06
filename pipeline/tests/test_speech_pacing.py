import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from speech_pacing import annotate_shot_pacing, estimate_speech_duration, pacing_tier

def test_emotion_tiers_map_to_expected_rates():
    assert [pacing_tier(x) for x in ("愤怒", "正常对话", "悲伤沉思")] == ["angry", "normal", "sad"]

def test_slower_emotion_needs_more_time():
    assert estimate_speech_duration("一二三四五六七八", "悲伤", 0, 0) == 4
    assert estimate_speech_duration("一二三四五六七八", "愤怒", 0, 0) == 2

def test_annotation_preserves_shots_and_adds_duration():
    shots = [{"dialogue": "你好，世界。", "emotion": "正常"}, {"lines": "无台词"}]
    assert annotate_shot_pacing(shots) is shots
    assert shots[0]["speech_duration_s"] >= 3 and shots[1]["speech_duration_s"] == 0

import json
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import phases.phase1.adaptation_engine as adaptation_engine
from phases.phase9.audio_mixer import AudioMixer
from prompt import event_extractor


def _event_response(lines=None):
    return json.dumps({"events": [{
        "who": ["凛"],
        "where": "雨夜轨道",
        "what": "凛质问烬",
        "emotion": "紧张",
        "visual": "两人隔着轨道对峙",
        "time": "雨夜",
        "action_type": "conflict",
        **({"lines": lines} if lines is not None else {}),
    }]}, ensure_ascii=False)


def test_extracted_dialogue_lines_reach_adaptation_prompt(monkeypatch):
    expected_lines = [{
        "speaker": "凛",
        "line": "你为什么要拦我？",
        "confidence": 1.0,
        "evidence": "凛质问：\"你为什么要拦我？\"",
    }]
    monkeypatch.setattr(
        event_extractor,
        "_call_llm",
        lambda _prompt, **_kwargs: _event_response(expected_lines),
    )
    events = event_extractor.extract_events([
        {"id": 1, "content": "凛质问：\"你为什么要拦我？\""}
    ])["events"]
    prompts = []

    def fake_adaptation_call(prompt, max_tokens=32000):
        prompts.append(prompt)
        if "不可变 beat 合同" in prompt:
            return json.dumps({"strategy": "保留对白", "beats": [{
                "beat_order": 1,
                "shot_size": "medium",
                "camera_angle": "over_shoulder",
                "camera_movement": "dolly_in",
                "lighting_key": "low_key",
                "shot_intent": "dialogue",
                "hero_moment": False,
                "texture_keywords": ["雨水反光", "轨道金属"],
            }]}, ensure_ascii=False)
        return json.dumps({"strategy": "保留对白", "shots": [{
            "beat_order": 1,
            "shot_order": 1,
            "source_events": [1],
            "action": "keep",
            "reason": "保留原台词",
            "who": ["凛"],
            "where": "雨夜轨道",
            "what": "凛质问烬",
            "emotion": "紧张",
            "visual": "两人隔着轨道对峙",
            "suggested_duration": 15,
            "boundary_before": "cut",
            "continuity_reason": "第一镜",
            "continuity_subject": "",
            "transition_to_next": "cut",
            "associate_assets": ["char:凛", "scene:雨夜轨道"],
            "shot_size": "medium",
            "camera_angle": "over_shoulder",
            "camera_movement": "dolly_in",
            "lighting_key": "low_key",
            "shot_intent": "dialogue",
            "hero_moment": False,
            "texture_keywords": ["雨水反光", "轨道金属"],
            "dialogue": expected_lines[0],
            "gen_strategy": "phantom",
        }]}, ensure_ascii=False)

    monkeypatch.setattr(adaptation_engine, "_call_llm_with_timeout_retry", fake_adaptation_call)
    adaptation_engine.adapt_events(events, target_duration=15, shot_duration=15)

    assert len(prompts) == 2
    assert all(expected_lines[0]["line"] in prompt for prompt in prompts)
    assert all('"lines"' in prompt for prompt in prompts)


def test_event_without_dialogue_gets_empty_lines():
    events = event_extractor._parse_events(_event_response())

    assert events[0]["lines"] == []


def test_audio_mixer_extracts_structured_dialogue_line():
    assert AudioMixer._spoken_text({"speaker": "烬", "line": "别再往前了。"}) == "别再往前了。"

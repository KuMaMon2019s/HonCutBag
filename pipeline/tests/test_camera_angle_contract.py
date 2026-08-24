"""Camera angle is authored once in adaptation and preserved downstream."""

from __future__ import annotations

import json

import pytest

from phases.phase1.adaptation_engine import _parse_beat_skeleton
from phases.phase1.storyboard_generator import _build_eight_layer_prompt
from utils.camera_angle_contracts import (
    CAMERA_ANGLE_VALUES,
    camera_angle_description,
    canonical_camera_angle,
)


def _skeleton_response(camera_angle: str) -> str:
    return json.dumps(
        {
            "strategy": "建立压迫关系",
            "beats": [
                {
                    "beat_order": 1,
                    "source_events": [1],
                    "dropped_source_events": [],
                    "action": "keep",
                    "reason": "保留人物与空间关系",
                    "who": ["林夏"],
                    "where": "控制室",
                    "what": "林夏发现异常信号",
                    "suggested_duration": 15,
                    "shot_size": "medium_wide",
                    "camera_angle": camera_angle,
                    "camera_movement": "dolly_in",
                    "lighting_key": "low_key",
                    "shot_intent": "reveal",
                    "hero_moment": False,
                    "texture_keywords": ["磨砂控制台", "冷色屏幕反光"],
                }
            ],
        },
        ensure_ascii=False,
    )


def test_adaptation_requires_one_controlled_camera_angle():
    parsed = _parse_beat_skeleton(
        _skeleton_response("over_shoulder"),
        expected_count=1,
        event_count=1,
    )

    assert parsed["beats"][0]["camera_angle"] == "over_shoulder"
    with pytest.raises(ValueError, match="camera_angle 无效: heroic_low"):
        _parse_beat_skeleton(
            _skeleton_response("heroic_low"),
            expected_count=1,
            event_count=1,
        )


def test_camera_angle_compatibility_normalizes_only_at_downstream_boundary():
    assert CAMERA_ANGLE_VALUES == (
        "eye_level",
        "low",
        "high",
        "dutch",
        "over_shoulder",
        "aerial",
        "bird",
        "worm",
    )
    assert canonical_camera_angle("over-shoulder") == "over_shoulder"
    assert canonical_camera_angle("unknown legacy angle") == "eye_level"
    assert camera_angle_description("low") == "低机位仰拍"


def test_storyboard_prompt_translates_without_replanning_camera_angle():
    shot = {
        "id": 1,
        "who": [],
        "where": "空旷控制室",
        "what": "屏幕突然显示异常面孔",
        "suggested_duration": 5,
        "shot_size": "medium_wide",
        "camera_angle": "low",
        "camera_movement": "dolly_in",
        "lighting_key": "low_key",
        "shot_intent": "reveal",
        "hero_moment": False,
        "texture_keywords": ["磨砂控制台", "冷色屏幕反光"],
    }

    prompt = _build_eight_layer_prompt(shot)

    assert shot["camera_angle"] == "low"
    assert "机位角度：低机位仰拍" in prompt

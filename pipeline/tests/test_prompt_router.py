#!/usr/bin/env python3
"""M4 prompt_router 单元测试：4 种路由模式。"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prompt_router import route_prompt


def test_seedance2_multi_shot_returns_chinese_structured():
    """seedance-2-0 + multi_shot → 中文结构化 prompt，包含 '分镜'。"""
    shot_data = {
        "style": "真人写实",
        "shots": [
            {
                "duration": 5,
                "time": "白天",
                "where": "咖啡厅",
                "camera": "中景",
                "visual": "女孩低头搅拌咖啡",
                "who": ["林夏"],
            }
        ],
    }
    assets = [{"name": "林夏", "description": "黑色长发，白衬衫"}]

    result = route_prompt("seedance-2-0", "multi_shot", shot_data, assets)

    assert "分镜" in result, f"seedance-2-0 multi_shot 应包含 '分镜'，实际：{result[:100]}"
    assert "林夏" in result
    assert "咖啡厅" in result


def test_seedance2_single_shot_returns_english_photorealistic():
    """seedance-2-0 + single_shot → 英文 prompt，包含 'Photorealistic'。"""
    shot_data = {
        "visual": "a girl stirring coffee",
        "emotion": "calm",
        "where": "cafe",
        "camera": "medium shot",
    }
    result = route_prompt("seedance-2-0", "single_shot", shot_data, [])

    assert "Photorealistic" in result, f"seedance-2-0 single_shot 应包含 'Photorealistic'，实际：{result[:100]}"
    assert "Scene:" in result or "Shot:" in result


def test_wan26_returns_cinematic_narrative():
    """wan2.6 → 叙事式英文 prompt，包含 'Cinematic'。"""
    shot_data = {
        "visual": "a boy running in the rain",
        "emotion": "tense",
        "where": "city street",
    }
    result = route_prompt("wan2.6", "single_shot", shot_data, [])

    assert "Cinematic" in result, f"wan2.6 应包含 'Cinematic'，实际：{result[:100]}"
    assert "Subject:" in result
    assert "Lighting:" in result


def test_generic_model_returns_five_dimension():
    """其他模型 → 五维度 prompt，包含 '[Visual]'。"""
    shot_data = {
        "visual": "一个男人走进房间",
        "emotion": "neutral",
        "where": "室内",
    }
    result = route_prompt("some-other-model", "single_shot", shot_data, [])

    assert "[Visual]" in result, f"generic 应包含 '[Visual]'，实际：{result[:100]}"
    assert "[Motion]" in result
    assert "[Camera]" in result
    assert "[Audio]" in result
    assert "[Narrative]" in result


def test_seedance2_multi_shot_without_assets():
    """seedance-2-0 multi_shot 即使没有 assets 也能正常生成。"""
    shot_data = {
        "shots": [
            {
                "duration": 4,
                "time": "夜晚",
                "where": "街道",
                "camera": "远景",
                "visual": "路灯下的人影",
                "who": [],
            }
        ],
    }
    result = route_prompt("seedance-2-0", "multi_shot", shot_data, [])

    assert "分镜" in result
    assert "街道" in result


def test_multi_ref_mode():
    """multi_ref 模式 → 包含 '[References]' 和 '[Instruction]'。"""
    shot_data = {
        "visual": "a girl walking in the park",
        "where": "park",
        "emotion": "happy",
        "camera": "medium shot",
    }
    assets = [
        {"name": "Alice", "description": "blonde hair, blue dress"},
        {"name": "Bob", "description": "brown hair, red shirt"},
    ]
    result = route_prompt("any-model", "multi_ref", shot_data, assets)

    assert "[References]" in result, f"multi_ref 应包含 '[References]'，实际：{result[:100]}"
    assert "[Instruction]" in result, f"multi_ref 应包含 '[Instruction]'，实际：{result[:100]}"
    assert "Alice" in result
    assert "Bob" in result
    assert "park" in result


def test_seedance_multi_12dims():
    """seedance-2-0 multi_shot 包含 12 维编码字段（动作、表情等）。"""
    shot_data = {
        "shots": [
            {
                "duration": 5,
                "time": "白天",
                "where": "咖啡厅",
                "camera": "中景",
                "visual": "女孩低头搅拌咖啡",
                "who": ["林夏"],
                "action": "搅拌咖啡",
                "expression": "专注",
                "lighting": "柔和侧光",
            }
        ],
    }
    result = route_prompt("seedance-2-0", "multi_shot", shot_data, [])

    assert "动作" in result, f"seedance-2-0 multi_shot 应包含 '动作'，实际：{result[:200]}"
    assert "表情" in result, f"seedance-2-0 multi_shot 应包含 '表情'，实际：{result[:200]}"
    assert "光影" in result, f"seedance-2-0 multi_shot 应包含 '光影'，实际：{result[:200]}"
    assert "搅拌咖啡" in result
    assert "专注" in result

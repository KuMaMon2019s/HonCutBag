#!/usr/bin/env python3
"""M5 quality_gate.run_storyboard_review 单元测试。"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quality_gate import run_storyboard_review


CHARACTERS = [
    {"id": "c1", "name": "林夏"},
    {"id": "c2", "name": "陈默"},
]


def _good_storyboard():
    """构造一个合格的分镜数据。"""
    return {
        "shots": [
            {
                "shot_id": "S01",
                "who": ["林夏"],
                "visual": "林夏低头搅拌咖啡，勺子碰杯壁发出清脆声响",
                "what": "搅拌咖啡",
            },
            {
                "shot_id": "S02",
                "who": ["陈默"],
                "visual": "陈默推门而入，门轴发出吱呀声",
                "what": "推门进入",
            },
        ]
    }


def test_normal_storyboard_gets_grade_a():
    """正常分镜 → A 级。"""
    result = run_storyboard_review(_good_storyboard(), "测试剧本", CHARACTERS)

    assert result["grade"] == "A", f"正常分镜应为 A 级，实际 {result['grade']}"
    assert result["severe"] == 0
    assert result["total_shots"] == 2


def test_abstract_words_deduct_points():
    """包含抽象词（如 '美丽的'）→ 扣分（moderate 增加）。"""
    storyboard = {
        "shots": [
            {
                "shot_id": "S01",
                "who": ["林夏"],
                "visual": "美丽的风景让人心情愉悦，非常好看",
                "what": "看风景",
            }
        ]
    }
    result = run_storyboard_review(storyboard, "测试剧本", CHARACTERS)

    assert result["moderate"] > 0, "抽象词应触发 moderate 问题"
    # 至少包含 '美丽的' 和 '非常' 两个抽象词
    assert any("美丽的" in issue for issue in result["issues"])


def test_empty_visual_deducts_points():
    """visual 为空 → 扣分（moderate 增加）。"""
    storyboard = {
        "shots": [
            {
                "shot_id": "S01",
                "who": ["林夏"],
                "visual": "",
                "what": "某动作",
            }
        ]
    }
    result = run_storyboard_review(storyboard, "测试剧本", CHARACTERS)

    assert result["moderate"] > 0, "空 visual 应触发 moderate 问题"
    assert any("visual" in issue.lower() or "R2" in issue for issue in result["issues"])


def test_unknown_character_triggers_r1_critical():
    """角色不在 characters 列表 → R1 严重问题。"""
    storyboard = {
        "shots": [
            {
                "shot_id": "S01",
                "who": ["张三"],  # 不在 CHARACTERS 中
                "visual": "张三走进房间，脚步声清晰可闻",
                "what": "进入",
            }
        ]
    }
    result = run_storyboard_review(storyboard, "测试剧本", CHARACTERS)

    assert result["severe"] >= 1, "未知角色应触发 R1 严重问题"
    assert any("R1" in issue and "张三" in issue for issue in result["issues"])


def test_three_plus_critical_issues_gets_grade_d():
    """3+ 严重问题 → D 级。"""
    storyboard = {
        "shots": [
            {"shot_id": "S01", "who": ["张三"], "visual": "张三走进房间，脚步声清晰", "what": "进入"},
            {"shot_id": "S02", "who": ["李四"], "visual": "李四转身离开，门声清脆", "what": "离开"},
            {"shot_id": "S03", "who": ["王五"], "visual": "王五坐下，椅子吱呀作响", "what": "坐下"},
        ]
    }
    result = run_storyboard_review(storyboard, "测试剧本", CHARACTERS)

    assert result["severe"] >= 3, f"应有 3+ 严重问题，实际 {result['severe']}"
    assert result["grade"] == "D", f"3+ 严重问题应为 D 级，实际 {result['grade']}"


def test_duration_over_15s():
    """片段时长 >15s → 严重问题。"""
    storyboard = {
        "shots": [
            {
                "shot_id": "S01",
                "who": ["林夏"],
                "visual": "林夏低头搅拌咖啡，勺子碰杯壁发出清脆声响",
                "what": "搅拌咖啡",
                "suggested_duration": 20,
            }
        ]
    }
    result = run_storyboard_review(storyboard, "测试剧本", CHARACTERS)

    assert result["severe"] >= 1, f"时长超限应触发严重问题，实际 severe={result['severe']}"
    assert any("时长" in issue and "20s" in issue for issue in result["issues"])


def test_long_dialogue():
    """台词 >20字 → 中等问题（建议拆镜）。"""
    storyboard = {
        "shots": [
            {
                "shot_id": "S01",
                "who": ["林夏"],
                "visual": "林夏低头搅拌咖啡，勺子碰杯壁发出清脆声响",
                "what": "搅拌咖啡",
                "dialogue": "\"这是一段超过二十个字的台词内容，需要被拆分成多个镜头来表现\"",
            }
        ]
    }
    result = run_storyboard_review(storyboard, "测试剧本", CHARACTERS)

    assert result["moderate"] >= 1, f"长台词应触发中等问题，实际 moderate={result['moderate']}"
    assert any("拆镜" in issue for issue in result["issues"])


def test_character_disappear():
    """同场景内人物消失 → 中等问题。"""
    storyboard = {
        "shots": [
            {
                "shot_id": "S01",
                "who": ["林夏", "陈默"],
                "where": "咖啡馆",
                "visual": "林夏和陈默在咖啡馆聊天",
                "what": "聊天",
            },
            {
                "shot_id": "S02",
                "who": ["林夏"],  # 陈默消失
                "where": "咖啡馆",  # 同场景
                "visual": "林夏独自思考",
                "what": "思考",
            }
        ]
    }
    result = run_storyboard_review(storyboard, "测试剧本", CHARACTERS)

    assert result["moderate"] >= 1, f"人物消失应触发中等问题，实际 moderate={result['moderate']}"
    assert any("消失" in issue and "陈默" in issue for issue in result["issues"])


def test_banned_visual_words():
    """visual 含禁词（如"色调"）→ 中等问题。"""
    storyboard = {
        "shots": [
            {
                "shot_id": "S01",
                "who": ["林夏"],
                "visual": "林夏低头搅拌咖啡，色调温暖，光影柔和",
                "what": "搅拌咖啡",
            }
        ]
    }
    result = run_storyboard_review(storyboard, "测试剧本", CHARACTERS)

    assert result["moderate"] >= 1, f"禁词应触发中等问题，实际 moderate={result['moderate']}"
    assert any("禁词" in issue and ("色调" in issue or "光影" in issue) for issue in result["issues"])


def test_same_camera():
    """相邻镜头同景别 → 中等问题。"""
    storyboard = {
        "shots": [
            {
                "shot_id": "S01",
                "who": ["林夏"],
                "visual": "林夏低头搅拌咖啡，勺子碰杯壁发出清脆声响",
                "what": "搅拌咖啡",
                "camera": "特写",
            },
            {
                "shot_id": "S02",
                "who": ["林夏"],
                "visual": "林夏抬头看向窗外",
                "what": "看向窗外",
                "camera": "特写",  # 同景别
            }
        ]
    }
    result = run_storyboard_review(storyboard, "测试剧本", CHARACTERS)

    assert result["moderate"] >= 1, f"同景别应触发中等问题，实际 moderate={result['moderate']}"
    assert any("景别" in issue and "特写" in issue for issue in result["issues"])


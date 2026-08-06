#!/usr/bin/env python3
"""
测试角色过滤器 - 验证非人物角色被正确过滤
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from phases.character_discoverer import _is_human_character, _filter_non_human_characters


def test_is_human_character_filtered():
    """测试应该被过滤掉的非人物角色"""
    test_cases = [
        ("冷空气", False, "天气现象"),
        ("鸡", False, "动物"),
        ("打探人员", False, "抽象指代"),
        ("说话者", False, "抽象指代"),
        ("试验者", False, "抽象指代"),
        ("行走者", False, "抽象指代"),
        ("观察者", False, "抽象指代"),
        ("记录者", False, "抽象指代"),
        ("思考者", False, "抽象指代"),
        ("保安们", False, "复数群体"),
    ]
    
    for name, expected, description in test_cases:
        result = _is_human_character(name)
        assert result == expected, f"{name} ({description}) should be {expected}, got {result}"


def test_is_human_character_kept():
    """测试应该保留的人物角色"""
    test_cases = [
        ("他", True, "主角"),
        ("母亲", True, "人物"),
        ("保安", True, "人物"),
        ("记者", True, "职业"),
        ("医生", True, "职业"),
        ("艾米", True, "人名"),
        ("小女孩", True, "人物"),
    ]
    
    for name, expected, description in test_cases:
        result = _is_human_character(name)
        assert result == expected, f"{name} ({description}) should be {expected}, got {result}"


def test_filter_non_human_characters():
    """测试批量过滤功能"""
    test_stats = {
        "冷空气": {"events": [1], "contexts": ["事件1: 冷空气来袭"]},
        "鸡": {"events": [2], "contexts": ["事件2: 鸡在叫"]},
        "他": {"events": [1, 2, 3], "contexts": ["事件1: 他走进房间"]},
        "母亲": {"events": [1, 2], "contexts": ["事件1: 母亲在厨房"]},
        "保安": {"events": [3], "contexts": ["事件3: 保安巡逻"]},
        "说话者": {"events": [1], "contexts": ["事件1: 说话者发言"]},
        "保安们": {"events": [3], "contexts": ["事件3: 保安们集合"]},
    }
    
    filtered = _filter_non_human_characters(test_stats)
    
    # 验证过滤结果
    assert len(filtered) <= 5, f"Should keep at most 5 characters, got {len(filtered)}"
    assert "冷空气" not in filtered, "冷空气 should be filtered out"
    assert "鸡" not in filtered, "鸡 should be filtered out"
    assert "说话者" not in filtered, "说话者 should be filtered out"
    assert "保安们" not in filtered, "保安们 should be filtered out"
    
    # 验证保留了正确的人物
    assert "他" in filtered, "他 should be kept"
    assert "母亲" in filtered, "母亲 should be kept"
    assert "保安" in filtered, "保安 should be kept"

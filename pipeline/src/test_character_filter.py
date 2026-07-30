#!/usr/bin/env python3
"""
测试角色过滤器 - 验证非人物角色被正确过滤
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from character_discoverer import _is_human_character, _filter_non_human_characters

# 测试用例
test_cases = [
    # 应该被过滤掉的（返回 False）
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
    
    # 应该保留的（返回 True）
    ("他", True, "主角"),
    ("母亲", True, "人物"),
    ("保安", True, "人物"),
    ("记者", True, "职业"),
    ("医生", True, "职业"),
    ("艾米", True, "人名"),
    ("小女孩", True, "人物"),
]

print("测试 _is_human_character 函数：\n")
passed = 0
failed = 0

for name, expected, description in test_cases:
    result = _is_human_character(name)
    status = "✓" if result == expected else "✗"
    if result == expected:
        passed += 1
    else:
        failed += 1
    
    print(f"{status} {name:15} ({description:10}) -> {result} (期望: {expected})")

print(f"\n测试结果：{passed} 通过，{failed} 失败")

# 测试批量过滤
print("\n\n测试 _filter_non_human_characters 函数：\n")
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
print(f"\n过滤前: {list(test_stats.keys())}")
print(f"过滤后: {list(filtered.keys())}")
print(f"保留数量: {len(filtered)}")

if len(filtered) <= 5 and "冷空气" not in filtered and "鸡" not in filtered:
    print("\n✓ 批量过滤测试通过")
else:
    print("\n✗ 批量过滤测试失败")

sys.exit(0 if failed == 0 else 1)

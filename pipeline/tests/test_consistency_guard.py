"""
test_consistency_guard.py — consistency_guard 模块的单元测试
"""

import pytest
from consistency_guard import (
    check_character_consistency,
    DEFAULT_THRESHOLD,
)


class TestCheckCharacterConsistency:
    """测试角色一致性检查主函数"""

    def test_perfect_consistency(self):
        """所有特征在 prompt 中都能匹配"""
        characters_data = {
            "characters": [
                {
                    "name": "主角",
                    "appearance": {
                        "hair": "黑色短发",
                        "clothing": "休闲装",
                        "gender": "male",
                    },
                }
            ]
        }
        shots_data = {
            "shots": [
                {
                    "id": 1,
                    "prompt": "A young man with 黑色短发 in 休闲装 walks",
                }
            ]
        }
        result = check_character_consistency(characters_data, shots_data)
        assert result["consistency_score"] == 100
        assert result["passed"] is True
        assert result["inconsistent_characters"] == []

    def test_inconsistent_character(self):
        """特征在部分 shot 中缺失"""
        characters_data = {
            "characters": [
                {
                    "name": "主角",
                    "appearance": {
                        "hair": "黑色短发",
                        "clothing": "休闲装",
                    },
                }
            ]
        }
        shots_data = {
            "shots": [
                {"id": 1, "prompt": "A man with 黑色短发 in 休闲装"},
                {"id": 2, "prompt": "A man in formal suit"},  # 两个特征都缺失
                {"id": 3, "prompt": "A man with 黑色短发"},  # 只匹配一个
            ]
        }
        result = check_character_consistency(characters_data, shots_data)
        # shot 2 完全没有匹配，shot 3 匹配了一个特征
        assert result["consistency_score"] < 100

    def test_threshold_pass(self):
        """阈值通过"""
        characters_data = {
            "characters": [
                {
                    "name": "主角",
                    "appearance": {"hair": "黑色短发"},
                }
            ]
        }
        shots_data = {
            "shots": [
                {"id": 1, "prompt": "A man with 黑色短发 walks"},
                {"id": 2, "prompt": "A man runs"},  # 缺失
            ]
        }
        # 50% 一致性，阈值 40 应该通过
        result = check_character_consistency(characters_data, shots_data, threshold=40)
        assert result["passed"] is True

    def test_threshold_fail(self):
        """阈值失败"""
        characters_data = {
            "characters": [
                {
                    "name": "主角",
                    "appearance": {"hair": "黑色短发"},
                }
            ]
        }
        shots_data = {
            "shots": [
                {"id": 1, "prompt": "A man with 黑色短发 walks"},
                {"id": 2, "prompt": "A man runs"},  # 缺失
            ]
        }
        # 50% 一致性，阈值 80 应该失败
        result = check_character_consistency(characters_data, shots_data, threshold=80)
        assert result["passed"] is False

    def test_no_characters_returns_zero(self):
        """无角色数据返回 0 分"""
        characters_data = {"characters": []}
        shots_data = {"shots": [{"id": 1, "prompt": "A man walks"}]}
        result = check_character_consistency(characters_data, shots_data)
        assert result["consistency_score"] == 0
        assert result["passed"] is False

    def test_no_shots_returns_zero(self):
        """无镜头数据返回 0 分"""
        characters_data = {
            "characters": [
                {"name": "主角", "appearance": {"hair": "黑色短发"}}
            ]
        }
        shots_data = {"shots": []}
        result = check_character_consistency(characters_data, shots_data)
        assert result["consistency_score"] == 0
        assert result["passed"] is False

    def test_multiple_characters(self):
        """多角色情况"""
        characters_data = {
            "characters": [
                {
                    "name": "主角",
                    "appearance": {"hair": "黑色短发"},
                },
                {
                    "name": "配角",
                    "appearance": {"hair": "棕色长发"},
                },
            ]
        }
        shots_data = {
            "shots": [
                {"id": 1, "prompt": "A man with 黑色短发 and a woman with 棕色长发 talk"},
            ]
        }
        result = check_character_consistency(characters_data, shots_data)
        assert result["consistency_score"] == 100
        assert result["inconsistent_characters"] == []

    def test_output_format(self):
        """输出格式符合规范"""
        characters_data = {
            "characters": [
                {
                    "name": "主角",
                    "appearance": {"hair": "黑色短发"},
                }
            ]
        }
        shots_data = {
            "shots": [
                {"id": 1, "prompt": "A man runs"},  # 缺少黑色短发
            ]
        }
        result = check_character_consistency(characters_data, shots_data)

        # 检查必需的键
        assert "consistency_score" in result
        assert "passed" in result
        assert "inconsistent_characters" in result

        # 检查类型
        assert isinstance(result["consistency_score"], int)
        assert isinstance(result["passed"], bool)
        assert isinstance(result["inconsistent_characters"], list)

    def test_shot_id_formatting(self):
        """shot ID 格式化为 S01, S02 等"""
        characters_data = {
            "characters": [
                {
                    "name": "主角",
                    "appearance": {"hair": "黑色短发"},
                }
            ]
        }
        shots_data = {
            "shots": [
                {"id": 1, "prompt": "missing"},
                {"id": 2, "prompt": "missing"},
                {"id": 3, "prompt": "missing"},
            ]
        }
        result = check_character_consistency(characters_data, shots_data)
        if result["inconsistent_characters"]:
            missing = result["inconsistent_characters"][0]["missing_in_shots"]
            assert all(s.startswith("S") for s in missing)

    def test_case_insensitive_matching(self):
        """匹配大小写不敏感"""
        characters_data = {
            "characters": [
                {
                    "name": "主角",
                    "appearance": {"hair": "BlackHair"},
                }
            ]
        }
        shots_data = {
            "shots": [
                {"id": 1, "prompt": "A man with BLACKHAIR walks"},
            ]
        }
        result = check_character_consistency(characters_data, shots_data)
        assert result["consistency_score"] == 100


class TestDefaultThreshold:
    """测试默认阈值"""

    def test_default_threshold_value(self):
        """默认阈值为 70"""
        assert DEFAULT_THRESHOLD == 70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

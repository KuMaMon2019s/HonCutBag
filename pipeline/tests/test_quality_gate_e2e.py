#!/usr/bin/env python3
"""
端到端测试：质检机制验证
"""

import json
import tempfile
from pathlib import Path
from quality.consistency_guard import check_character_consistency, run_consistency_check

def test_consistency_guard():
    """测试角色一致性检查"""
    print("=" * 60)
    print("测试 1: 角色一致性检查")
    print("=" * 60)
    
    # 测试数据：角色特征完整，但部分镜头缺失
    characters_data = {
        'characters': [
            {
                'name': '主角',
                'appearance': {
                    'hair': '黑色短发',
                    'clothing': '休闲装',
                    'face': '坚毅的面容'
                }
            }
        ]
    }
    
    shots_data = {
        'shots': [
            {'id': 1, 'prompt': '黑色短发的男人走进房间'},
            {'id': 2, 'prompt': '休闲装的男人坐在椅子上'},
            {'id': 3, 'prompt': '坚毅面容的男人看着窗外'},
            {'id': 4, 'prompt': '一个男人站在窗前'},  # 缺失特征
        ]
    }
    
    result = check_character_consistency(characters_data, shots_data, threshold=70)
    
    print(f"一致性分数: {result['consistency_score']}")
    print(f"是否通过: {result['passed']}")
    print(f"检查总数: {result['total_checks']}")
    print(f"通过数量: {result['passed_checks']}")
    
    if result['inconsistent_characters']:
        print("\n不一致的角色:")
        for char in result['inconsistent_characters']:
            print(f"  - {char['name']}")
            print(f"    期望特征: {', '.join(char['expected_features'])}")
            print(f"    缺失镜头: {', '.join(char['missing_in_shots'])}")
    
    assert result['consistency_score'] == 75, f"期望分数 75，实际 {result['consistency_score']}"
    assert result['passed'] == True, f"期望通过，实际未通过"
    print("\n✓ 测试 1 通过")
def test_quality_gate():
    """测试质检门逻辑"""
    print("\n" + "=" * 60)
    print("测试 2: 质检门逻辑")
    print("=" * 60)
    
    # 模拟 pipeline 中的质检门判断
    def quality_gate_check(consistency_score, threshold=70):
        """质检门判断逻辑"""
        return consistency_score >= threshold
    
    # 测试用例
    test_cases = [
        (85, True, "高分通过"),
        (70, True, "临界值通过"),
        (69, False, "临界值未通过"),
        (50, False, "低分未通过"),
    ]
    
    for score, expected, desc in test_cases:
        result = quality_gate_check(score)
        status = "✓" if result == expected else "✗"
        print(f"{status} 分数 {score}: {desc} (期望 {expected}, 实际 {result})")
        assert result == expected, f"测试失败: {desc}"
    
    print("\n✓ 测试 2 通过")
def test_file_based_check():
    """测试基于文件的一致性检查"""
    print("\n" + "=" * 60)
    print("测试 3: 基于文件的一致性检查")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # 创建测试文件
        characters_file = tmpdir / "CHARACTERS.json"
        storyboard_file = tmpdir / "STORYBOARD.json"
        
        characters_data = {
            'characters': [
                {
                    'name': '主角',
                    'appearance': {
                        'hair': '黑色短发',
                        'clothing': '休闲装'
                    }
                }
            ]
        }
        
        storyboard_data = {
            'shots': [
                {'id': 1, 'prompt': '黑色短发的男人走进房间'},
                {'id': 2, 'prompt': '休闲装的男人坐在椅子上'},
            ]
        }
        
        characters_file.write_text(json.dumps(characters_data, ensure_ascii=False))
        storyboard_file.write_text(json.dumps(storyboard_data, ensure_ascii=False))
        
        # 运行基于文件的检查
        result = run_consistency_check(tmpdir, threshold=70)
        
        print(f"一致性分数: {result['consistency_score']}")
        print(f"是否通过: {result['passed']}")
        
        # 检查是否生成了报告文件
        report_file = tmpdir / "consistency_report.json"
        assert report_file.exists(), "未生成 consistency_report.json"
        print(f"✓ 报告文件已生成: {report_file}")
        
        assert result['consistency_score'] == 100, f"期望分数 100，实际 {result['consistency_score']}"
        assert result['passed'] == True, f"期望通过，实际未通过"
    
    print("\n✓ 测试 3 通过")
def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("质检机制端到端测试")
    print("=" * 60 + "\n")
    
    tests = [
        test_consistency_guard,
        test_quality_gate,
        test_file_based_check,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60 + "\n")
    
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

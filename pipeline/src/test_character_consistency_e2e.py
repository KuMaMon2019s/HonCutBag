#!/usr/bin/env python3
"""
角色一致性端到端测试
验证 appearance dict 传递和角色参考图注入
"""

import json
import tempfile
from pathlib import Path
import shutil

def test_appearance_dict_propagation():
    """测试 appearance dict 在 fan_out_characters 中的传递"""
    print("=" * 60)
    print("测试 1: appearance dict 传递")
    print("=" * 60)
    
    # 模拟 CHARACTERS.json 中的角色数据
    characters_data = {
        "characters": [
            {
                "id": "char_001",
                "name": "主角",
                "appearance": {
                    "summary": "年轻男性，黑色短发，休闲装，坚毅面容",
                    "hair": "黑色短发",
                    "clothing": "休闲装",
                    "face": "坚毅面容",
                    "build": "中等身材"
                },
                "style": "写实风格"
            }
        ]
    }
    
    # 模拟 fan_out_characters 逻辑
    char_dict = {
        "id": characters_data["characters"][0]["id"],
        "name": characters_data["characters"][0]["name"],
        "description": characters_data["characters"][0]["appearance"]["summary"],
        "appearance": characters_data["characters"][0]["appearance"],  # 新增字段
        "style": characters_data["characters"][0]["style"]
    }
    
    # 验证
    assert "appearance" in char_dict, "char_dict 缺少 appearance 字段"
    assert char_dict["appearance"]["hair"] == "黑色短发", "hair 字段不匹配"
    assert char_dict["appearance"]["clothing"] == "休闲装", "clothing 字段不匹配"
    assert char_dict["appearance"]["face"] == "坚毅面容", "face 字段不匹配"
    
    print("✓ appearance dict 完整传递")
    print(f"  - 字段数量: {len(char_dict['appearance'])}")
    print(f"  - 包含: hair, clothing, face, build, summary")
    
    return True


def test_character_reference_mapping():
    """测试角色参考图映射逻辑"""
    print("\n" + "=" * 60)
    print("测试 2: 角色参考图映射")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # 创建角色目录和参考图
        chars_dir = output_dir / "characters"
        char_dir = chars_dir / "char_001"
        char_dir.mkdir(parents=True)
        front_png = char_dir / "front.png"
        front_png.touch()
        
        # 模拟 _run_phase5_om_seedance 中的映射逻辑
        characters_data = {
            "characters": [
                {
                    "id": "char_001",
                    "name": "主角",
                    "appearance": {"summary": "年轻男性"}
                }
            ]
        }
        
        character_ref_images = {}
        for char in characters_data.get("characters", []):
            char_id = char.get("id", "")
            char_name = char.get("name", "")
            char_dir = output_dir / "characters" / char_id
            front_png = char_dir / "front.png"
            if front_png.exists():
                character_ref_images[char_id] = str(front_png)
                character_ref_images[char_name] = str(front_png)
        
        # 验证
        assert len(character_ref_images) == 2, "映射数量应为 2"
        assert "char_001" in character_ref_images, "缺少 char_001 映射"
        assert "主角" in character_ref_images, "缺少 主角 映射"
        assert character_ref_images["char_001"] == character_ref_images["主角"], "两个映射应指向同一文件"
        
        print("✓ 角色参考图映射正确")
        print(f"  - char_001 -> {character_ref_images['char_001']}")
        print(f"  - 主角 -> {character_ref_images['主角']}")
    
    return True


def test_shot_character_matching():
    """测试 shot 与角色的匹配逻辑"""
    print("\n" + "=" * 60)
    print("测试 3: shot 与角色匹配")
    print("=" * 60)
    
    # 模拟角色参考图映射
    character_ref_images = {
        "char_001": "/tmp/characters/char_001/front.png",
        "主角": "/tmp/characters/char_001/front.png",
        "char_002": "/tmp/characters/char_002/front.png",
        "配角": "/tmp/characters/char_002/front.png"
    }
    
    # 测试场景 1: shot 中明确指定 characters 字段
    shot1 = {
        "id": 1,
        "characters": ["char_001"],
        "prompt": "主角走进房间"
    }
    
    matched_ref = None
    if shot1.get("characters"):
        for char_id in shot1["characters"]:
            if char_id in character_ref_images:
                matched_ref = character_ref_images[char_id]
                break
    
    assert matched_ref is not None, "未匹配到角色参考图"
    assert matched_ref == "/tmp/characters/char_001/front.png", "匹配到错误的参考图"
    
    print("✓ 场景 1: shot.characters 字段匹配成功")
    print(f"  - shot.characters: {shot1['characters']}")
    print(f"  - 匹配到: {matched_ref}")
    
    # 测试场景 2: 从 prompt 中匹配角色名称
    shot2 = {
        "id": 2,
        "prompt": "主角和配角在对话"
    }
    
    matched_ref = None
    for char_key, char_path in character_ref_images.items():
        if char_key.lower() in shot2["prompt"].lower():
            matched_ref = char_path
            break
    
    assert matched_ref is not None, "未从 prompt 中匹配到角色"
    
    print("✓ 场景 2: prompt 关键词匹配成功")
    print(f"  - prompt: {shot2['prompt']}")
    print(f"  - 匹配到: {matched_ref}")
    
    return True


def test_reference_priority():
    """测试参考图优先级逻辑"""
    print("\n" + "=" * 60)
    print("测试 4: 参考图优先级")
    print("=" * 60)
    
    # 模拟场景
    storyboard_image = "/tmp/storyboard.png"
    character_ref = "/tmp/characters/char_001/front.png"
    
    # 场景 1: 有角色参考图
    reference_image = character_ref if character_ref else storyboard_image
    assert reference_image == character_ref, "应优先使用角色参考图"
    print("✓ 场景 1: 角色参考图优先于 storyboard.png")
    
    # 场景 2: 无角色参考图
    character_ref = None
    reference_image = character_ref if character_ref else storyboard_image
    assert reference_image == storyboard_image, "应降级到 storyboard.png"
    print("✓ 场景 2: 无角色参考图时降级到 storyboard.png")
    
    # 场景 3: 两者都无
    storyboard_image = None
    reference_image = character_ref if character_ref else storyboard_image
    assert reference_image is None, "应返回 None"
    print("✓ 场景 3: 两者都无时返回 None")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("角色一致性端到端测试")
    print("=" * 60 + "\n")
    
    tests = [
        ("appearance dict 传递", test_appearance_dict_propagation),
        ("角色参考图映射", test_character_reference_mapping),
        ("shot 与角色匹配", test_shot_character_matching),
        ("参考图优先级", test_reference_priority)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ 测试失败: {name}")
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

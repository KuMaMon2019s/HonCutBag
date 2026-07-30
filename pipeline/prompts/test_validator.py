#!/usr/bin/env python3
"""测试提示词验证脚本"""

from prompt_validator import validate_prompt

# v1 简化版提示词（应该失败）
v1_prompt = """
角色设计，三视图，可爱卡通风格
正面视图，侧面视图，背面视图
白色背景，清晰线条
"""

# v2 完整版提示词（应该通过）
v2_prompt = """
【宏观描述】画面风格：真人写实风格，照片级渲染，细节超高清。
根据以下角色描述，生成一张纯白背景的角色三视图设定表，
清晰展示角色的正面、侧面、背面标准正交视图。
【微观描述】
1. 角色描述：20多岁清秀纤细的都市白领女性，黑色长直发，穿白色衬衫
2. 画面要求：纯白背景，无阴影。
3. 正面全身站立像：角色正面朝向镜头，面部五官清晰，表情自然。
质感十足，高质量，震撼的视觉效果。
"""

print("=" * 60)
print("测试 v1 简化版提示词（预期：失败）")
print("=" * 60)
is_valid_v1, missing_v1 = validate_prompt(v1_prompt, "three_view")
print(f"验证结果: {'✅ 通过' if is_valid_v1 else '❌ 失败'}")
print(f"缺失关键词: {missing_v1}")
print()

print("=" * 60)
print("测试 v2 完整版提示词（预期：通过）")
print("=" * 60)
is_valid_v2, missing_v2 = validate_prompt(v2_prompt, "three_view")
print(f"验证结果: {'✅ 通过' if is_valid_v2 else '❌ 失败'}")
print(f"缺失关键词: {missing_v2}")
print()

print("=" * 60)
print("测试总结")
print("=" * 60)
print(f"v1 简化版: {'❌ 失败（符合预期）' if not is_valid_v1 else '✅ 通过（不符合预期）'}")
print(f"v2 完整版: {'✅ 通过（符合预期）' if is_valid_v2 else '❌ 失败（不符合预期）'}")

# 验证测试用例是否正确
assert not is_valid_v1, "v1 应该失败"
assert is_valid_v2, "v2 应该通过"
print("\n✅ 所有测试用例验证通过！")

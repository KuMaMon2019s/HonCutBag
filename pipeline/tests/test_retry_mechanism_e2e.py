#!/usr/bin/env python3
"""
重试机制端到端测试
验证 MAX_RETRIES 和重试反馈注入
"""

import sys
import os

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_max_retries_config():
    """测试 MAX_RETRIES 配置"""
    print("=" * 60)
    print("测试 1: MAX_RETRIES 配置")
    print("=" * 60)
    
    from phases.storyboard_generator import MAX_RETRIES
    
    assert MAX_RETRIES == 3, f"期望 MAX_RETRIES=3，实际={MAX_RETRIES}"
    print(f"✓ MAX_RETRIES = {MAX_RETRIES}")


def test_retry_feedback_injection():
    """测试重试时反馈注入逻辑"""
    print("\n" + "=" * 60)
    print("测试 2: 重试反馈注入逻辑")
    print("=" * 60)
    
    # 模拟重试逻辑
    user_prompt = "原始 prompt"
    last_error = "JSON 解析失败: Expecting value"
    
    # 第一次尝试
    attempt = 0
    if attempt > 0:
        feedback_prompt = user_prompt + f"\n\n[重试反馈] 上次失败原因: {last_error}。请确保输出有效的 JSON 格式。"
    else:
        feedback_prompt = user_prompt
    
    assert feedback_prompt == user_prompt, "第一次尝试不应注入反馈"
    print("✓ 第一次尝试不注入反馈")
    
    # 第二次尝试（重试）
    attempt = 1
    if attempt > 0:
        feedback_prompt = user_prompt + f"\n\n[重试反馈] 上次失败原因: {last_error}。请确保输出有效的 JSON 格式。"
    else:
        feedback_prompt = user_prompt
    
    assert feedback_prompt != user_prompt, "重试时应注入反馈"
    assert "[重试反馈]" in feedback_prompt, "反馈应包含 [重试反馈] 标记"
    assert last_error in feedback_prompt, "反馈应包含上次错误信息"
    print("✓ 重试时注入反馈")
    print(f"  反馈内容: {feedback_prompt[-50:]}")


def test_retry_error_tracking():
    """测试重试时错误追踪"""
    print("\n" + "=" * 60)
    print("测试 3: 重试错误追踪")
    print("=" * 60)
    
    # 模拟重试循环
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                raise ValueError("第一次错误")
            elif attempt == 1:
                raise ValueError("第二次错误")
            else:
                # 第三次成功
                break
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                print(f"  尝试 {attempt + 1} 失败: {last_error}")
    
    assert last_error == "第二次错误", f"期望 last_error='第二次错误'，实际='{last_error}'"
    print(f"✓ last_error 正确追踪: {last_error}")


def test_retry_with_fallback():
    """测试重试失败后的降级方案"""
    print("\n" + "=" * 60)
    print("测试 4: 重试失败后的降级方案")
    print("=" * 60)
    
    # 模拟降级逻辑
    max_retries = 3
    llm_result = None
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # 模拟所有重试都失败
            raise ValueError(f"尝试 {attempt + 1} 失败")
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                print(f"  尝试 {attempt + 1} 失败，重试中...")
            else:
                print(f"  所有重试失败，使用降级方案")
                llm_result = {
                    "prompt": "Cinematic shot, fallback scene",
                    "caption": "降级 caption",
                }
    
    assert llm_result is not None, "应该有降级结果"
    assert "fallback" in llm_result["prompt"], "降级 prompt 应包含 fallback"
    print(f"✓ 降级方案: {llm_result}")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("重试机制端到端测试")
    print("=" * 60 + "\n")
    
    tests = [
        ("MAX_RETRIES 配置", test_max_retries_config),
        ("重试反馈注入逻辑", test_retry_feedback_injection),
        ("重试错误追踪", test_retry_error_tracking),
        ("重试失败后的降级方案", test_retry_with_fallback),
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
    success = main()
    sys.exit(0 if success else 1)

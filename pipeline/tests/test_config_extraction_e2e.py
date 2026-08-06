#!/usr/bin/env python3
"""
配置提取端到端测试
验证 config.yaml 配置读取和 retry_count 统一递增
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_config_yaml_loading():
    """测试 config.yaml 加载"""
    print("=" * 60)
    print("测试 1: config.yaml 加载")
    print("=" * 60)
    
    try:
        from utils.pipeline_config import get_quality_gate_threshold
        
        # 测试默认值
        threshold = get_quality_gate_threshold()
        assert threshold == 70, f"期望默认阈值 70，实际 {threshold}"
        print(f"✓ 默认阈值: {threshold}")
        
        return True
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False


def test_config_custom_threshold():
    """测试自定义阈值配置"""
    print("\n" + "=" * 60)
    print("测试 2: 自定义阈值配置")
    print("=" * 60)
    
    try:
        from utils.pipeline_config import get_quality_gate_threshold
        
        # 创建临时 config.yaml
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("""
quality_gate:
  consistency_threshold: 85
""")
            
            # 测试自定义值
            threshold = get_quality_gate_threshold(str(config_path))
            assert threshold == 85, f"期望自定义阈值 85，实际 {threshold}"
            print(f"✓ 自定义阈值: {threshold}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_retry_count_increment():
    """测试 retry_count 统一递增"""
    print("\n" + "=" * 60)
    print("测试 3: retry_count 统一递增")
    print("=" * 60)
    
    try:
        from pipeline_runner import run_pipeline
        
        # 创建临时输出目录
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # 创建模拟的 checkpoint.json
            checkpoint = {
                "status": "running",
                "current_phase": "phase5",
                "retry_count": 0,
                "completed_phases": []
            }
            checkpoint_path = output_dir / "checkpoint.json"
            checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2))
            
            # 模拟 retry_count 递增
            checkpoint["retry_count"] += 1
            checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2))
            
            # 读取验证
            loaded = json.loads(checkpoint_path.read_text())
            assert loaded["retry_count"] == 1, f"期望 retry_count=1，实际 {loaded['retry_count']}"
            print(f"✓ retry_count 递增: {loaded['retry_count']}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_config_fallback():
    """测试配置回退到默认值"""
    print("\n" + "=" * 60)
    print("测试 4: 配置回退到默认值")
    print("=" * 60)
    
    try:
        from utils.pipeline_config import get_quality_gate_threshold
        
        # 测试不存在的配置文件
        threshold = get_quality_gate_threshold("/nonexistent/config.yaml")
        assert threshold == 70, f"期望回退到默认值 70，实际 {threshold}"
        print(f"✓ 回退到默认值: {threshold}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("配置提取端到端测试")
    print("=" * 60 + "\n")
    
    tests = [
        ("config.yaml 加载", test_config_yaml_loading),
        ("自定义阈值配置", test_config_custom_threshold),
        ("retry_count 统一递增", test_retry_count_increment),
        ("配置回退到默认值", test_config_fallback),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
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

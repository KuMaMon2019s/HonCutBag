#!/usr/bin/env python3
"""
pipeline_config.py — 管线配置管理模块

从 config.yaml 读取可配置参数，支持回退到默认值。
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional


# 默认配置
DEFAULT_CONFIG = {
    "quality_gate": {
        "consistency_threshold": 70,
        "min_shot_quality": 60,
    },
    "retry": {
        "max_retries": 3,
        "backoff_factor": 2.0,
    },
    "consistency_guard": {
        "enabled": True,
        "check_interval": 5,
    },
}


def _find_config_file() -> Optional[Path]:
    """查找 config.yaml 文件"""
    # 优先查找当前目录
    current_dir = Path.cwd()
    config_path = current_dir / "config.yaml"
    if config_path.exists():
        return config_path
    
    # 查找 scripts 目录
    scripts_dir = Path(__file__).parent
    config_path = scripts_dir / "config.yaml"
    if config_path.exists():
        return config_path
    
    # 查找项目根目录
    project_root = scripts_dir.parent
    config_path = project_root / "config.yaml"
    if config_path.exists():
        return config_path
    
    return None


def load_config(config_path: Optional[str] = None) -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径（可选）
    
    Returns:
        配置字典
    """
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    
    # 自动查找配置文件
    path = _find_config_file()
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    return {}


def get_config_value(key_path: str, default: Any = None, config_path: Optional[str] = None) -> Any:
    """
    获取配置值（支持点分隔路径）
    
    Args:
        key_path: 配置键路径，如 "quality_gate.consistency_threshold"
        default: 默认值
        config_path: 配置文件路径（可选）
    
    Returns:
        配置值
    """
    config = load_config(config_path)
    
    # 解析点分隔路径
    keys = key_path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def get_quality_gate_threshold(config_path: Optional[str] = None) -> int:
    """获取质检门阈值"""
    return get_config_value(
        "quality_gate.consistency_threshold",
        default=DEFAULT_CONFIG["quality_gate"]["consistency_threshold"],
        config_path=config_path
    )


def get_max_retries(config_path: Optional[str] = None) -> int:
    """获取最大重试次数"""
    return get_config_value(
        "retry.max_retries",
        default=DEFAULT_CONFIG["retry"]["max_retries"],
        config_path=config_path
    )


def get_backoff_factor(config_path: Optional[str] = None) -> float:
    """获取重试退避因子"""
    return get_config_value(
        "retry.backoff_factor",
        default=DEFAULT_CONFIG["retry"]["backoff_factor"],
        config_path=config_path
    )


def is_consistency_guard_enabled(config_path: Optional[str] = None) -> bool:
    """检查一致性守卫是否启用"""
    return get_config_value(
        "consistency_guard.enabled",
        default=DEFAULT_CONFIG["consistency_guard"]["enabled"],
        config_path=config_path
    )


if __name__ == "__main__":
    # 测试
    print("配置管理模块测试")
    print("-" * 60)
    
    print(f"质检门阈值: {get_quality_gate_threshold()}")
    print(f"最大重试次数: {get_max_retries()}")
    print(f"重试退避因子: {get_backoff_factor()}")
    print(f"一致性守卫启用: {is_consistency_guard_enabled()}")

#!/usr/bin/env python3
"""
提示词验证脚本 - 在调用 API 前检查必需关键词

v1 vs v2 对比发现，简化提示词会遗漏关键细节（质感十足、震撼视觉效果、
正面朝向、高质量、布局位置标注、明确色值）。本脚本确保 prompt 包含
所有必需关键词。
"""

import argparse
import sys
from typing import List, Tuple

# 必需关键词配置
REQUIRED_KEYWORDS = {
    "three_view": [
        "质感十足",
        "震撼的视觉效果",
        "正面",
        "高质量",
        "角色描述",
        "画面要求",
        "纯白背景"
    ]
}


def validate_prompt(prompt: str, template_type: str = "three_view") -> Tuple[bool, List[str]]:
    """
    验证提示词是否包含所有必需关键词
    
    Args:
        prompt: 待验证的提示词文本
        template_type: 模板类型，默认 "three_view"
    
    Returns:
        tuple[bool, list[str]]: (是否通过验证, 缺失的关键词列表)
    
    Examples:
        >>> is_valid, missing = validate_prompt("测试提示词", "three_view")
        >>> if not is_valid:
        ...     raise ValueError(f"提示词缺少必需关键词: {missing}")
    """
    if template_type not in REQUIRED_KEYWORDS:
        raise ValueError(f"未知的模板类型: {template_type}。支持的类型: {list(REQUIRED_KEYWORDS.keys())}")
    
    required = REQUIRED_KEYWORDS[template_type]
    missing = [keyword for keyword in required if keyword not in prompt]
    
    is_valid = len(missing) == 0
    return is_valid, missing


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="验证提示词是否包含必需关键词",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python prompt_validator.py --prompt "质感十足的震撼视觉效果..." --template three_view
  python prompt_validator.py --prompt "测试" --template three_view
        """
    )
    
    parser.add_argument(
        "--prompt",
        required=True,
        help="待验证的提示词文本"
    )
    parser.add_argument(
        "--template",
        default="three_view",
        choices=list(REQUIRED_KEYWORDS.keys()),
        help="模板类型 (默认: three_view)"
    )
    
    args = parser.parse_args()
    
    is_valid, missing = validate_prompt(args.prompt, args.template)
    
    if is_valid:
        print("✅ 提示词验证通过")
        sys.exit(0)
    else:
        print(f"❌ 缺少必需关键词: {missing}")
        sys.exit(1)


if __name__ == "__main__":
    main()

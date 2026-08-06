#!/usr/bin/env python3
"""M1 director_planner 单元测试：dry_run + 静态内容。

注意：不实际调用 LLM API，只测 dry_run 模式和静态 prompt 内容。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phases.director_planner import plan_director, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def test_plan_director_dry_run_returns_skipped(tmp_path):
    """plan_director(dry_run=True) 返回 {'status': 'skipped'}，不调 API。"""
    result = plan_director("测试剧本", tmp_path, dry_run=True)

    assert isinstance(result, dict)
    assert result.get("status") == "skipped"
    # dry_run 不应生成 director_plan.json
    assert not (tmp_path / "director_plan.json").exists()


def test_system_prompt_exists_and_non_empty():
    """SYSTEM_PROMPT 存在且非空。"""
    assert SYSTEM_PROMPT, "SYSTEM_PROMPT 不应为空"
    assert len(SYSTEM_PROMPT) > 20, "SYSTEM_PROMPT 过短"
    assert "导演" in SYSTEM_PROMPT


def test_user_prompt_template_exists_and_non_empty():
    """USER_PROMPT_TEMPLATE 存在且非空。"""
    assert USER_PROMPT_TEMPLATE, "USER_PROMPT_TEMPLATE 不应为空"
    assert "{script_text}" in USER_PROMPT_TEMPLATE, "模板应包含 {script_text} 占位符"


def test_user_prompt_template_contains_four_bridges():
    """USER_PROMPT_TEMPLATE 包含 4 种桥梁关键词。"""
    bridges = ["动作桥梁", "情绪接力", "空间视线", "台词黏合"]
    for bridge in bridges:
        assert bridge in USER_PROMPT_TEMPLATE, (
            f"USER_PROMPT_TEMPLATE 应包含 '{bridge}'"
        )

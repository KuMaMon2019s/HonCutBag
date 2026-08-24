#!/usr/bin/env python3
"""
director_planner.py — M1: HonCut 导演规划层
作为 Phase 1 的导演子步骤，产出结构化的导演规划。
只做四件事：拆分场、台词统计、情绪分析、过渡设计。
"""

import json
from pathlib import Path

from runtime.llm_policy import LLMStreamPolicy
from utils.ark_llm import call_llm_stream, create_ark_client
from utils.config import DEFAULT_TEXT_MODEL, get_api_key

DIRECTOR_LLM_POLICY = LLMStreamPolicy.long_structured_output(max_tokens=8000)
# Compatibility aliases for integrations that inspect the Phase 1 limits.
LLM_WALL_TIMEOUT = DIRECTOR_LLM_POLICY.wall_timeout_seconds
LLM_IDLE_TIMEOUT = DIRECTOR_LLM_POLICY.idle_timeout_seconds

SYSTEM_PROMPT = (
    "你是资深影视导演。对剧本做导演级规划分析。"
    "只做四件事：拆分场、台词统计、情绪分析、过渡设计。"
    "不规划光影、色调、配乐。"
    "输出严格 JSON，不要输出任何解释文字。"
    # --- P2-5e: M1 流程约束 ---
    "严格按5步线性执行，不回退：1.通读全文→2.拆分场→3.台词统计→4.情绪分析→5.过渡设计。"
    "方法论不外泄，只输出结构化JSON结果。"
)

USER_PROMPT_TEMPLATE = (
    "请对以下剧本做导演规划，输出 JSON：\n\n"
    "剧本：\n{script_text}\n\n"
    "输出格式：\n"
    "{{\n"
    '  "scenes": [\n'
    "    {{\n"
    '      "scene_id": "Sc1",\n'
    '      "scene_name": "地点·时间概况",\n'
    '      "dialogue_count": 3,\n'
    '      "dialogue_words": 86,\n'
    '      "emotion_intensity": 4,\n'
    '      "emotion_arc": "情绪X→情绪Y",\n'
    '      "notes": {{\n'
    '        "emotional_peak": "关键情感砸点",\n'
    '        "consistency_anchors": ["角色: 外貌服装锚点"],\n'
    '        "spatial": "空间与距离描述",\n'
    '        "ambient_sound": "环境音提示",\n'
    '        "pitfall": "易错提示"\n'
    "      }}\n"
    "    }}\n"
    "  ],\n"
    '  "scene_transitions": [\n'
    "    {{\n"
    '      "from": "Sc1",\n'
    '      "to": "Sc2",\n'
    '      "type": "动作桥梁|情绪接力|空间视线|台词黏合",\n'
    '      "description": "过渡描述"\n'
    "    }}\n"
    "  ],\n"
    '  "spatial_positions": [\n'
    "    {{\n"
    '      "character": "角色名",\n'
    '      "default_position": "左前/右前/居中/左后/右后",\n'
    '      "default_facing": "面朝左/面朝右/面朝镜头/背对镜头"\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "分场原则：一个场=同一时空下一段连续戏，以地点变更/时间跳变/戏剧单元收束为切点。\n"
    "情绪分析：逐场给情绪浓度0~10 + 场内情绪推进X→Y。\n"
    "场间过渡4种桥梁：\n"
    "1. 动作桥梁：同一组人物连续动作，前段结尾=动作起始态，后段首镜=进行时/完成时\n"
    "2. 情绪接力：对话/冲突情绪延续，前段结尾用反应镜/微表情铺垫，后段承接强化\n"
    "3. 空间视线：场景切换/视线转移，空镜+视线引导+声音延续\n"
    "4. 台词黏合：台词/音效需画面回应，前段末尾声音延续到后段首镜\n"
    "空间位置基准：为每个角色建立默认站位和朝向，后续分镜必须按此基准标注，确保跨镜位置一致。\n"
)


def plan_director(script_text: str, output_dir: Path, dry_run: bool = False) -> dict:
    """
    Phase 1: 导演规划

    Args:
        script_text: 剧本文本
        output_dir: 输出目录
        dry_run: dry-run 模式

    Returns:
        director_plan dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "director_plan.json"

    if dry_run:
        print("  ⊘ dry-run 模式，跳过导演规划")
        return {"status": "skipped", "reason": "dry-run"}

    # 调用 LLM。生产导演规划是 Phase 1 的必需证据，失败必须向上冒泡。
    try:
        api_key = get_api_key("ARK_AGENT_API_KEY")
        if not api_key:
            raise RuntimeError("director planning requires ARK_AGENT_API_KEY")

        client = create_ark_client(read_timeout=LLM_IDLE_TIMEOUT)
        user_prompt = USER_PROMPT_TEMPLATE.format(script_text=script_text[:8000])

        content = call_llm_stream(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=DEFAULT_TEXT_MODEL,
            max_tokens=DIRECTOR_LLM_POLICY.max_tokens,
            wall_timeout=DIRECTOR_LLM_POLICY.wall_timeout_seconds,
            idle_timeout=DIRECTOR_LLM_POLICY.idle_timeout_seconds,
            _client=client,
        )

        if not content:
            raise ValueError("LLM 返回空内容")

        # 解析 JSON（兼容 markdown 代码块包裹）
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        plan = json.loads(text)

        # 写入文件
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

        scenes = plan.get("scenes", [])
        transitions = plan.get("scene_transitions", [])
        print(f"  ✓ [M1] 导演规划完成: {len(scenes)} 场, {len(transitions)} 个过渡")
        for s in scenes:
            print(f"    {s.get('scene_id', '?')}: {s.get('scene_name', '?')} "
                  f"(情绪{s.get('emotion_intensity', '?')}/10, {s.get('emotion_arc', '?')})")

        return {"status": "done", "plan": plan, "output": str(plan_path)}

    except Exception as exc:
        plan_path.unlink(missing_ok=True)
        raise RuntimeError(f"director planning failed: {exc}") from exc

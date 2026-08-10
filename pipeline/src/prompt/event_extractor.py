#!/usr/bin/env python3
"""
事件提取器 - Phase 2 事件图谱引擎的第二个模块

从 text_parser.py 输出的 segments 中提取结构化事件。
每个事件包含：谁(who)、在哪(where)、做什么(what)、情绪(emotion)、视觉描述(visual)。

输入：text_parser.py 的输出 JSON（segments 列表），或直接传入文本
输出：JSON 格式的结构化事件列表

逻辑：
1. 接收 segments（从 text_parser 输出或直接文本）
2. 对每个 segment，调用 LLM 提取事件
3. 一个 segment 可能产生多个事件
4. 解析失败则重试 1 次，仍失败则标记 error 跳过
5. 合并所有 segment 的事件，重新编号
6. 输出 JSON
"""

import json
import sys
import os
import argparse
import time
from typing import List, Dict, Any, Optional

from openai import OpenAI


# ─── LLM 配置 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一个影视编剧助手。从文本中提取可视化事件。"
    "每个事件必须能转化为一个视频镜头。输出严格 JSON 数组。"
    "不要输出任何解释文字，只输出 JSON。"
)

USER_PROMPT_TEMPLATE = (
    "从以下文本中提取事件：\n\n"
    "{content}\n\n"
    "输出 JSON 数组，每个元素包含：\n"
    "- who: 数组，参与者列表\n"
    "- where: 字符串，地点\n"
    "- what: 字符串，发生了什么\n"
    "- emotion: 字符串，情绪氛围\n"
    "- visual: 字符串，描述画面（用于生成视频镜头）\n"
    "- time: 字符串，时间/季节\n"
    "- action_type: 字符串，事件类型（discovery/conflict/resolution/transition 等）\n"
    "- lines: 数组，本事件中角色说出的台词原文，每条为 "
    "{{\"speaker\": \"角色名\", \"line\": \"逐字台词\"}}；无台词时为空数组 []\n"
    "line 必须逐字保留剧本原文，禁止改写、摘要或翻译。\n"
    "剧本对白可能写作 角色名：\"台词\" 或 角色名:\"台词\"，全角/半角冒号与引号均可能出现。"
)

LLM_TIMEOUT = 180  # 秒（2026-08-09: turbo 推理模型比 lite 慢，60s 实测 8/19 段超时）
MAX_RETRIES = 1  # 解析失败重试次数


def _get_client() -> OpenAI:
    """
    创建 OpenAI 客户端

    API Key: ARK_AGENT_API_KEY (火山方舟 Agent Plan)

    Returns:
        OpenAI 客户端实例

    Raises:
        SystemExit: 如果所有 API key 环境变量均未设置
    """
    # 尝试多个环境变量名
    api_key = (
        os.environ.get("ARK_AGENT_API_KEY")
        
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        print("错误：环境变量 ARK_AGENT_API_KEY 未设置（火山方舟 Agent Plan）", file=sys.stderr)
        sys.exit(1)

    return OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/plan/v3",
    )


def _call_llm(prompt: str) -> str:
    """
    调用 LLM 并返回原始响应文本

    Args:
        prompt: 用户 prompt（segment 内容）

    Returns:
        LLM 原始响应字符串

    Raises:
        Exception: API 调用失败时抛出
    """
    client = _get_client()

    response = client.chat.completions.create(
        model="doubao-seed-2.1-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        timeout=LLM_TIMEOUT,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM 返回空内容")
    return content


def _parse_events(response: str) -> List[Dict[str, Any]]:
    """
    解析 LLM 响应为事件列表

    尝试从响应中提取 JSON 数组。支持：
    - 纯 JSON 数组
    - 被 ```json ... ``` 包裹的 JSON
    - 被 ``` ... ``` 包裹的 JSON

    Args:
        response: LLM 原始响应字符串

    Returns:
        事件字典列表

    Raises:
        ValueError: 无法解析为有效 JSON 数组
    """
    text = response.strip()

    # 尝试提取 ```json ... ``` 代码块
    if "```" in text:
        # 匹配 ```json 或 ``` 包裹的内容
        import re
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # 尝试解析 JSON
    parsed = json.loads(text)

    # 验证是数组
    if not isinstance(parsed, list):
        raise ValueError(f"期望 JSON 数组，得到 {type(parsed).__name__}")

    # 验证每个元素的基本结构
    required_fields = {"who", "where", "what", "emotion", "visual", "time", "action_type"}
    for i, event in enumerate(parsed):
        if not isinstance(event, dict):
            raise ValueError(f"第 {i+1} 个事件不是字典")
        missing = required_fields - set(event.keys())
        if missing:
            raise ValueError(f"第 {i+1} 个事件缺少字段: {missing}")
        event.setdefault("lines", [])

    return parsed


def _extract_events_from_segment(segment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从单个 segment 中提取事件

    调用 LLM 提取事件，解析失败则重试 1 次。

    Args:
        segment: text_parser 输出的单个 segment 字典

    Returns:
        事件列表（可能为空）
    """
    content = segment.get("content", "")
    if not content.strip():
        return []

    prompt = USER_PROMPT_TEMPLATE.format(content=content)

    last_error = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = _call_llm(prompt)
            events = _parse_events(response)
            return events
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"  JSON 解析失败，重试 ({attempt+1}/{MAX_RETRIES}): {e}", file=sys.stderr)
                time.sleep(1)
            continue
        except Exception as e:
            last_error = e
            print(f"  LLM 调用失败: {e}", file=sys.stderr)
            break

    # 所有重试都失败
    print(f"  事件提取失败 (segment {segment.get('id', '?')}): {last_error}", file=sys.stderr)
    return []


def extract_events(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    核心函数：从 segments 列表中提取所有事件

    对每个 segment 调用 LLM 提取事件，合并后重新编号。

    Args:
        segments: text_parser 输出的 segments 列表

    Returns:
        包含 total_events 和 events 的字典
    """
    if not segments:
        return {"total_events": 0, "events": []}

    all_events = []
    event_id = 1

    for segment in segments:
        segment_id = segment.get("id", 0)
        print(f"处理 segment {segment_id}...", file=sys.stderr)

        events = _extract_events_from_segment(segment)

        for event in events:
            event["id"] = event_id
            event["segment_id"] = segment_id
            all_events.append(event)
            event_id += 1

    return {
        "total_events": len(all_events),
        "events": all_events,
    }


def extract_events_from_text(text: str) -> Dict[str, Any]:
    """
    从原始文本提取事件（内部调用 text_parser）

    Args:
        text: 原始文本

    Returns:
        事件提取结果字典
    """
    # 导入 text_parser
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prompt.text_parser import parse_text

    parsed = parse_text(text)
    segments = parsed.get("segments", [])

    return extract_events(segments)


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="事件提取器 - 从文本/segments 中提取结构化事件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从 text_parser 输出文件读取
  python event_extractor.py --input parsed.json --output events.json

  # 从文本直接提取（内部调用 text_parser）
  python event_extractor.py --text "艾米在雪地里找到了一只受伤的小狼" --output events.json

  # 管道模式
  python text_parser.py --text "..." | python event_extractor.py --output events.json
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入文件路径（text_parser 输出的 JSON）",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出 JSON 文件路径（可选，默认打印到 stdout）",
    )

    parser.add_argument(
        "--text", "-t",
        type=str,
        help="直接传入文本（内部调用 text_parser）",
    )

    args = parser.parse_args()

    # 确定输入来源
    segments = None

    if args.text:
        # 从文本直接提取
        result = extract_events_from_text(args.text)
    elif args.input:
        # 从文件读取 segments
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"错误：文件不存在 - {args.input}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"错误：JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        # 支持直接传入 segments 列表或 parse_text 的完整输出
        if isinstance(data, list):
            segments = data
        elif isinstance(data, dict) and "segments" in data:
            segments = data["segments"]
        else:
            print("错误：输入格式不支持，期望 segments 列表或包含 segments 的字典", file=sys.stderr)
            sys.exit(1)

        result = extract_events(segments)
    elif not sys.stdin.isatty():
        # 从 stdin 读取（管道模式）
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"错误：stdin JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        if isinstance(data, list):
            segments = data
        elif isinstance(data, dict) and "segments" in data:
            segments = data["segments"]
        else:
            print("错误：输入格式不支持", file=sys.stderr)
            sys.exit(1)

        result = extract_events(segments)
    else:
        parser.print_help()
        sys.exit(1)

    # 输出 JSON
    json_output = json.dumps(result, ensure_ascii=False, indent=2)
    print(json_output)

    # 写入文件（如果指定）
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(json_output)
            print(f"\n已写入文件：{args.output}", file=sys.stderr)
        except IOError as e:
            print(f"错误：无法写入文件 - {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

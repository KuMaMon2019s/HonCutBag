#!/usr/bin/env python3
"""
文本解析器 - Phase 1 编剧引擎的文本解析模块

将任意文本输入解析为结构化段落列表，供后续 event_extractor.py 消费。

输入：任意文本（一句话、描述、大纲、长文）
输出：JSON 格式的结构化段落列表

逻辑：
- short（<500字）：不拆分，整个文本作为一个 segment
- medium（500-5000字）：按段落拆分（双换行 \n\n 或单换行）
- long（>5000字）：先按章节标题拆（第X章/Chapter X/数字编号），再按段落拆

不使用 LLM — 纯规则解析（正则 + 字符串分割），速度快、零成本
"""

import re
import json
import sys
import argparse
from typing import List, Dict, Any


def detect_input_type(text: str) -> str:
    """
    判断输入规模类型
    
    Args:
        text: 输入文本
        
    Returns:
        "short" | "medium" | "long"
    """
    char_count = len(text)
    if char_count < 500:
        return "short"
    elif char_count <= 5000:
        return "medium"
    else:
        return "long"


def extract_chapter_title(line: str) -> str:
    """
    从行中提取章节标题
    
    支持的格式：
    - 第X章 标题
    - Chapter X 标题
    - 数字编号. 标题
    - 纯数字编号 标题
    
    Args:
        line: 单行文本
        
    Returns:
        提取的标题，如果不是章节标题则返回空字符串
    """
    line = line.strip()
    
    # 第X章 标题
    match = re.match(r'^第[一二三四五六七八九十百千\d]+章\s+(.+)$', line)
    if match:
        return match.group(1)
    
    # Chapter X 标题
    match = re.match(r'^Chapter\s+\d+\s+(.+)$', line, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # 数字编号. 标题 (1. 标题)
    match = re.match(r'^(\d+)\.\s+(.+)$', line)
    if match:
        return match.group(2)
    
    # 纯数字编号 标题 (1 标题)
    match = re.match(r'^(\d+)\s+(.+)$', line)
    if match:
        return match.group(2)
    
    return ""


def generate_title(content: str, max_length: int = 20) -> str:
    """
    为段落生成标题（取前 N 字）
    
    Args:
        content: 段落内容
        max_length: 最大长度
        
    Returns:
        生成的标题
    """
    # 取第一行
    first_line = content.split('\n')[0].strip()
    
    # 截取
    if len(first_line) <= max_length:
        return first_line
    else:
        return first_line[:max_length] + "..."


def split_by_chapters(text: str) -> List[Dict[str, str]]:
    """
    按章节拆分长文本
    
    Args:
        text: 输入文本
        
    Returns:
        章节列表，每个元素包含 title 和 content
    """
    lines = text.split('\n')
    chapters = []
    current_chapter = {"title": "", "content": []}
    
    for line in lines:
        chapter_title = extract_chapter_title(line)
        
        if chapter_title:
            # 保存之前的章节
            if current_chapter["content"]:
                chapters.append({
                    "title": current_chapter["title"],
                    "content": '\n'.join(current_chapter["content"]).strip()
                })
            
            # 开始新章节
            current_chapter = {"title": chapter_title, "content": []}
        else:
            current_chapter["content"].append(line)
    
    # 保存最后一个章节
    if current_chapter["content"]:
        chapters.append({
            "title": current_chapter["title"],
            "content": '\n'.join(current_chapter["content"]).strip()
        })
    
    return [ch for ch in chapters if ch["content"]]


def split_by_paragraphs(text: str) -> List[str]:
    """
    按段落拆分文本
    
    优先使用双换行 \n\n 分割，如果没有则使用单换行 \n
    
    Args:
        text: 输入文本
        
    Returns:
        段落列表
    """
    # 尝试双换行分割
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # 如果只有一个段落且有单换行，尝试单换行分割
    if len(paragraphs) == 1 and '\n' in text:
        paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def parse_text(text: str) -> Dict[str, Any]:
    """
    核心解析函数：将文本解析为结构化段落列表
    
    Args:
        text: 输入文本
        
    Returns:
        解析结果字典，包含：
        - input_type: "short" | "medium" | "long"
        - total_chars: 总字符数
        - segments: 段落列表，每个段落包含 id, title, content, char_count
    """
    # 错误处理：空输入
    if not text or not text.strip():
        return {
            "input_type": "short",
            "total_chars": 0,
            "segments": []
        }
    
    text = text.strip()
    input_type = detect_input_type(text)
    total_chars = len(text)
    
    segments = []
    segment_id = 1
    
    if input_type == "short":
        # 不拆分，整个文本作为一个 segment
        segments.append({
            "id": segment_id,
            "title": generate_title(text),
            "content": text,
            "char_count": total_chars
        })
    
    elif input_type == "medium":
        # 按段落拆分
        paragraphs = split_by_paragraphs(text)
        for para in paragraphs:
            segments.append({
                "id": segment_id,
                "title": generate_title(para),
                "content": para,
                "char_count": len(para)
            })
            segment_id += 1
    
    else:  # long
        # 先按章节拆，再按段落拆
        chapters = split_by_chapters(text)
        
        # 如果没有检测到章节，按段落拆
        if not chapters:
            chapters = [{"title": "", "content": text}]
        
        for chapter in chapters:
            if chapter["title"]:
                # 有章节标题，作为独立 segment
                segments.append({
                    "id": segment_id,
                    "title": chapter["title"],
                    "content": chapter["content"],
                    "char_count": len(chapter["content"])
                })
                segment_id += 1
            else:
                # 无章节标题，按段落拆
                paragraphs = split_by_paragraphs(chapter["content"])
                for para in paragraphs:
                    segments.append({
                        "id": segment_id,
                        "title": generate_title(para),
                        "content": para,
                        "char_count": len(para)
                    })
                    segment_id += 1
    
    return {
        "input_type": input_type,
        "total_chars": total_chars,
        "segments": segments
    }


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="文本解析器 - 将任意文本解析为结构化段落列表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从文件读取
  python text_parser.py --input story.txt --output parsed.json
  
  # 从 stdin 读取
  echo "艾米在雪地里找到了一只受伤的小狼" | python text_parser.py --output parsed.json
  
  # 直接传文本
  python text_parser.py --text "艾米在雪地里找到了一只受伤的小狼"
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入文件路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出 JSON 文件路径（可选，默认只打印到 stdout）"
    )
    
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="直接传入文本"
    )
    
    args = parser.parse_args()
    
    # 获取输入文本
    text = None
    
    if args.text:
        text = args.text
    elif args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"错误：文件不存在 - {args.input}", file=sys.stderr)
            sys.exit(1)
        except UnicodeDecodeError as e:
            print(f"错误：文件编码问题 - {e}", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty():
        # 从 stdin 读取
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(1)
    
    # 解析文本
    result = parse_text(text)
    
    # 输出 JSON
    json_output = json.dumps(result, ensure_ascii=False, indent=2)
    
    # 打印到 stdout
    print(json_output)
    
    # 写入文件（如果指定）
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"\n已写入文件：{args.output}", file=sys.stderr)
        except IOError as e:
            print(f"错误：无法写入文件 - {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

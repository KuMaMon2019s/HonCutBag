#!/usr/bin/env python3
"""
分镜生成器 - Phase 2 事件图谱引擎的最后一个模块

将 adaptation_engine.py 输出的 shot 列表转化为 orchestrator.py 可消费的 STORYBOARD.json。
这是 Phase 2 → Phase 4 的桥梁。

输入：
- shots JSON（adaptation_engine.py 输出）
- characters JSON（character_discoverer.py 输出，用于 first_frame 路径）

输出：
- STORYBOARD.json（与 orchestrator.py 的 load_storyboard() / parse_shots() 兼容）

逻辑：
1. 读取 shots + characters
2. 对每个 shot，调用 LLM：
   - 将中文 visual 描述翻译/转化为英文 Seedance prompt（加电影感关键词）
   - 生成中文字幕（简短，≤15字）
3. 计算 caption_frames（30fps，字幕占中间 80%）
4. 组装 STORYBOARD.json
5. 验证：确保 orchestrator.py 的 parse_shots() 能正确解析
"""

import json
import sys
import os
import argparse
import re
import time
from typing import List, Dict, Any, Optional

from openai import OpenAI


# ─── LLM 配置 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是 AI 视频生成 prompt 专家，专精真人写实摄影风格。\n"
    "将中文场景描述转化为英文 Seedance 视频生成 prompt。\n\n"
    "【输出结构】三段式，严格按以下顺序：\n"
    "1. 【画面】（主干，最长）：完整保留场景描述中的所有视觉元素——主体、动作、空间关系、物件细节。\n"
    "   角色外貌不写具体描述，改为参考图绑定语句：\n"
    "   'Based on the reference image of {角色名}, maintain consistent: face features, hairstyle, costume details.'\n"
    "2. 【光影】（独立段）：光源方向、色调倾向、明暗关系。\n"
    "   雨天/阴天：漫射冷光，无主光源，灰蓝主调，空气潮湿感，低饱和度。\n"
    "   傍晚：暖调侧逆光，斜射余晖，长影拉伸。\n"
    "   夜间：窗光冷蓝，室内暖点光源，冷暖对比。\n"
    "3. 【风格】（最短）：固定锚定词 + 画质锁定。\n"
    "   必加：Photorealistic cinematography, cinematic quality, ultra-fine detail, "
    "strong contrast, delicate skin texture, detailed facial rendering, "
    "strand-by-strand hair detail, modern urban aesthetic, oriental temperament.\n"
    "   画质：Ultra-sharp 4K, high detail, natural sharpness, no subtitles, no watermark.\n\n"
    "【情绪→面容映射】\n"
    "心动/欣喜：嘴角微扬，眼底笑意，眼神明亮 → subtle smile, bright eyes\n"
    "悲伤/失落：面色沉静，眼神黯淡 → calm face, dim eyes\n"
    "温柔/深情：神情柔和，眉目温润 → gentle expression, warm gaze\n"
    "惊讶/震惊：神情微愣，目光骤聚 → stunned expression, wide eyes\n"
    "紧张/慌乱：眼神飘忽，目光四顾 → nervous gaze, restless eyes\n"
    "隐忍/克制：神情内敛，唇线收紧 → restrained expression, tight lips\n\n"
    "同时生成一句中文字幕（≤15字）。\n"
    "输出严格 JSON：{\"prompt\": \"英文prompt\", \"caption\": \"中文字幕\"}\n"
    "不要输出任何解释文字。"
)

USER_PROMPT_TEMPLATE = (
    "场景：{visual}\n"
    "角色：{who}\n"
    "情绪：{emotion}\n"
    "地点：{where}\n"
    "角色参考图绑定：{ref_binding}\n"
    "风格/情绪/光影提示：{style_suffix}\n\n"
    "要求：\n"
    "1. 【画面】段完整保留场景描述的所有视觉元素，角色用参考图绑定语句替代外貌描述\n"
    "2. 【光影】段根据地点和时间推导光影（雨天=漫射冷光灰蓝调，傍晚=暖调侧逆光）\n"
    "3. 【风格】段使用固定锚定词\n"
    "4. 三段合并为一个连贯的英文 prompt，不用中文标签\n\n"
    '输出 JSON：{{"prompt": "英文视频生成prompt", "caption": "中文字幕"}}'
)

LLM_TIMEOUT = 60  # 秒
MAX_RETRIES = 3  # 解析失败重试次数（从 1 提高到 3）
FPS = 30  # 帧率


# ─── LLM 客户端 ─────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    """
    创建 OpenAI 客户端

    API Key: ARK_AGENT_API_KEY (火山方舟 Agent Plan)
    """
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


def _call_llm(user_prompt: str) -> str:
    """
    调用 LLM 并返回原始响应文本

    Args:
        user_prompt: 用户 prompt

    Returns:
        LLM 原始响应字符串

    Raises:
        Exception: API 调用失败时抛出
    """
    client = _get_client()

    response = client.chat.completions.create(
        model="doubao-seed-2.0-lite",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        timeout=LLM_TIMEOUT,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM 返回空内容")
    return content


def _parse_llm_response(response: str) -> Dict[str, str]:
    """
    解析 LLM 响应为 {"prompt": "...", "caption": "..."}

    Args:
        response: LLM 原始响应字符串

    Returns:
        包含 prompt 和 caption 的字典

    Raises:
        ValueError: 无法解析为有效 JSON 或缺少必要字段
    """
    text = response.strip()

    # 尝试提取 ```json ... ``` 代码块
    if "```" in text:
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # 解析 JSON
    parsed = json.loads(text)

    if not isinstance(parsed, dict):
        raise ValueError(f"期望 JSON 对象，得到 {type(parsed).__name__}")

    if "prompt" not in parsed:
        raise ValueError("缺少 'prompt' 字段")
    if "caption" not in parsed:
        raise ValueError("缺少 'caption' 字段")

    return {"prompt": parsed["prompt"], "caption": parsed["caption"]}


# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def _calculate_caption_frames(duration: int) -> str:
    """
    计算字幕帧范围（30fps，字幕占中间 80%）

    Args:
        duration: 镜头时长（秒）

    Returns:
        帧范围字符串，如 "30-180"
    """
    total_frames = duration * FPS
    # 字幕占中间 80%
    start_frame = int(total_frames * 0.1)
    end_frame = int(total_frames * 0.9)
    # 确保至少 1 帧
    start_frame = max(1, start_frame)
    end_frame = max(start_frame + 1, end_frame)
    return f"{start_frame}-{end_frame}"


def _build_characters_map(characters: Optional[List[Dict[str, Any]]]) -> Dict[str, str]:
    """
    构建角色名 → first_frame 路径的映射

    Args:
        characters: 角色列表（character_discoverer.py 输出）

    Returns:
        字典：角色名 → "characters/{id}/front.png"
    """
    if not characters:
        return {}

    char_map = {}
    for char in characters:
        char_id = char.get("id", "")
        name = char.get("name", "")
        # 优先使用 id，其次使用 name
        if char_id:
            char_map[name] = f"characters/{char_id}/front.png"
            # 也把别名映射上
            for alias in char.get("aliases", []):
                char_map[alias] = f"characters/{char_id}/front.png"
        elif name:
            # 没有 id 时用 name 做路径
            safe_name = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '_', name.lower())
            char_map[name] = f"characters/{safe_name}/front.png"

    return char_map


def _get_first_frame_for_shot(
    shot: Dict[str, Any],
    characters_map: Dict[str, str],
) -> Optional[str]:
    """
    为 shot 确定 first_frame 路径

    如果 shot 中有 who 字段（角色列表），取第一个角色的 front.png。

    Args:
        shot: shot 字典（含 who 字段）
        characters_map: 角色名 → 路径映射

    Returns:
        first_frame 路径字符串，或 None
    """
    who = shot.get("who", [])
    if not who or not characters_map:
        return None

    # 取第一个角色
    first_char = who[0] if isinstance(who, list) else str(who)
    return characters_map.get(first_char)


def _build_shot_prompt(shot: Dict[str, Any], characters: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    为单个 shot 构建 LLM user prompt

    Args:
        shot: shot 字典
        characters: 角色列表（可选）

    Returns:
        格式化后的 user prompt
    """
    visual = shot.get("visual", "")
    who_list = shot.get("who", [])
    who = ", ".join(who_list) if isinstance(who_list, list) else str(who_list)
    emotion = shot.get("emotion", "")
    where = shot.get("where", "")
    
    # Build reference binding for characters in this shot
    ref_parts = []
    if characters:
        for char in characters:
            char_name = char.get("name", "")
            if char_name in who or any(alias in who for alias in char.get("aliases", [])):
                ref_parts.append(
                    f"Based on the reference image of {char_name}, "
                    f"maintain consistent: face features, hairstyle, costume details."
                )
    ref_binding = " ".join(ref_parts) if ref_parts else "No character reference."
    
    # Add emotion and style enrichment
    try:
        from emotion_mapping import build_style_suffix
        style_suffix = build_style_suffix(emotion=emotion, scene=where)
    except ImportError:
        style_suffix = ""
    
    return USER_PROMPT_TEMPLATE.format(
        visual=visual,
        who=who,
        emotion=emotion,
        where=where,
        ref_binding=ref_binding,
        style_suffix=style_suffix,
    )


# ─── 核心函数 ────────────────────────────────────────────────────────────────

def generate_storyboard(
    shots: List[Dict[str, Any]],
    characters: Optional[List[Dict[str, Any]]] = None,
    title: str = "未命名项目",
) -> Dict[str, Any]:
    """
    将 shot 列表转化为 STORYBOARD.json 格式

    Args:
        shots: shot 列表（adaptation_engine.py 输出）
        characters: 角色列表（character_discoverer.py 输出，可选）
        title: 项目标题

    Returns:
        STORYBOARD.json 格式的字典

    Raises:
        RuntimeError: LLM 调用失败
        ValueError: 输入无效
    """
    if not shots:
        raise ValueError("shot 列表为空，无法生成分镜")

    # 构建角色映射
    characters_map = _build_characters_map(characters)

    # 计算总时长
    total_duration = sum(shot.get("suggested_duration", 5) for shot in shots)

    # 处理每个 shot
    storyboard_shots = []
    for i, shot in enumerate(shots, 1):
        duration = shot.get("suggested_duration", 5)

        # 调用 LLM 生成英文 prompt 和中文 caption
        user_prompt = _build_shot_prompt(shot, characters)
        llm_result = None

        for attempt in range(1 + MAX_RETRIES):
            try:
                # 如果是重试，注入质量反馈到 prompt
                if attempt > 0:
                    feedback_prompt = user_prompt + f"\n\n[重试反馈] 上次失败原因: {last_error}。请确保输出有效的 JSON 格式。"
                    response = _call_llm(feedback_prompt)
                else:
                    response = _call_llm(user_prompt)
                
                llm_result = _parse_llm_response(response)
                break
            except (json.JSONDecodeError, ValueError) as e:
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    print(f"Shot {i} LLM 解析失败，重试中（{attempt + 1}/{MAX_RETRIES}）: {e}", file=sys.stderr)
                    time.sleep(1)
                else:
                    print(f"Shot {i} LLM 解析失败，使用降级方案: {e}", file=sys.stderr)
                    # 降级：直接用 visual 作为 prompt，空 caption
                    llm_result = {
                        "prompt": f"Cinematic shot, {shot.get('visual', 'scene')}, natural lighting, film grain",
                        "caption": shot.get("what", ""),
                    }
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES:
                    print(f"Shot {i} LLM 调用失败，重试中（{attempt + 1}/{MAX_RETRIES}）: {e}", file=sys.stderr)
                    time.sleep(1)
                else:
                    print(f"Shot {i} LLM 调用失败，使用降级方案: {e}", file=sys.stderr)
                    llm_result = {
                        "prompt": f"Cinematic shot, {shot.get('visual', 'scene')}, natural lighting, film grain",
                        "caption": shot.get("what", ""),
                    }

        if llm_result is None:
            llm_result = {
                "prompt": f"Cinematic shot, {shot.get('visual', 'scene')}, natural lighting, film grain",
                "caption": shot.get("what", ""),
            }

        # 确定 first_frame
        first_frame = _get_first_frame_for_shot(shot, characters_map)

        # 计算 caption_frames
        caption_frames = _calculate_caption_frames(duration)

        # 构建 shot 名称
        shot_name = shot.get("what", f"镜头 {i}")
        if len(shot_name) > 30:
            shot_name = shot_name[:30] + "..."

        # 组装 storyboard shot
        storyboard_shot = {
            "id": i,
            "name": shot_name,
            "duration": duration,
            "prompt": llm_result["prompt"],
            "caption": llm_result["caption"],
            "caption_frames": caption_frames,
        }

        # 仅在有 first_frame 时添加该字段
        if first_frame:
            storyboard_shot["first_frame"] = first_frame

        storyboard_shots.append(storyboard_shot)

    # 组装完整 STORYBOARD
    storyboard = {
        "title": title,
        "total_shots": len(storyboard_shots),
        "target_duration": total_duration,
        "static_reference_images": {},
        "shots": storyboard_shots,
    }

    return storyboard


def validate_storyboard(storyboard: Dict[str, Any]) -> List[str]:
    """
    验证 STORYBOARD 是否兼容 orchestrator.py 的 parse_shots()

    Args:
        storyboard: STORYBOARD 字典

    Returns:
        错误列表（空列表表示通过验证）
    """
    errors = []

    # 检查顶层结构
    if "shots" not in storyboard:
        errors.append("缺少 'shots' 数组")
        return errors

    if not isinstance(storyboard["shots"], list):
        errors.append("'shots' 应为数组")
        return errors

    # 检查每个 shot 的必要字段（与 orchestrator.parse_shots() 对齐）
    required_fields = {"id", "name", "prompt"}
    for i, shot in enumerate(storyboard["shots"]):
        if not isinstance(shot, dict):
            errors.append(f"第 {i+1} 个 shot 不是字典")
            continue
        missing = required_fields - set(shot.keys())
        if missing:
            errors.append(f"第 {i+1} 个 shot 缺少字段: {missing}")

        # 检查 id 是否为整数
        if "id" in shot and not isinstance(shot["id"], int):
            errors.append(f"第 {i+1} 个 shot 的 id 应为整数，得到 {type(shot['id']).__name__}")

        # 检查 duration 是否为正数
        if "duration" in shot:
            dur = shot["duration"]
            if not isinstance(dur, (int, float)) or dur <= 0:
                errors.append(f"第 {i+1} 个 shot 的 duration 应为正数，得到 {dur}")

        # 检查 prompt 是否为非空字符串
        if "prompt" in shot and not isinstance(shot["prompt"], str):
            errors.append(f"第 {i+1} 个 shot 的 prompt 应为字符串")

    return errors


# ─── CLI 入口 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="分镜生成器 - 将 shot 列表转化为 STORYBOARD.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python storyboard_generator.py --shots shots.json --characters characters.json --output STORYBOARD.json
  python storyboard_generator.py --shots shots.json --output STORYBOARD.json  # 无角色也行
  cat shots.json | python storyboard_generator.py --output STORYBOARD.json
        """,
    )

    parser.add_argument(
        "--shots", "-s",
        type=str,
        help="shots JSON 文件路径（adaptation_engine.py 输出）",
    )

    parser.add_argument(
        "--characters", "-c",
        type=str,
        help="角色 JSON 文件路径（character_discoverer.py 输出，可选）",
    )

    parser.add_argument(
        "--title", "-t",
        type=str,
        default="未命名项目",
        help="项目标题（默认：未命名项目）",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出 JSON 文件路径（可选，默认打印到 stdout）",
    )

    args = parser.parse_args()

    # ── 读取 shots ────────────────────────────────────────────────────────
    shots_data = None

    if args.shots:
        try:
            with open(args.shots, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"错误：文件不存在 - {args.shots}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"错误：JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        # 支持直接传入 shots 列表或包含 shots 的字典
        if isinstance(data, list):
            shots_data = data
        elif isinstance(data, dict) and "shots" in data:
            shots_data = data["shots"]
            # 如果字典中有 target_duration，用它作为 title 的参考
            if "strategy" in data and args.title == "未命名项目":
                args.title = f"视频项目 ({len(shots_data)} 镜头)"
        else:
            print("错误：输入格式不支持，期望 shots 列表或包含 shots 的字典", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty():
        # 从 stdin 读取（管道模式）
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"错误：stdin JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        if isinstance(data, list):
            shots_data = data
        elif isinstance(data, dict) and "shots" in data:
            shots_data = data["shots"]
        else:
            print("错误：输入格式不支持", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # ── 读取角色（可选）────────────────────────────────────────────────────
    characters = None
    if args.characters:
        try:
            with open(args.characters, "r", encoding="utf-8") as f:
                char_data = json.load(f)
        except FileNotFoundError:
            print(f"错误：文件不存在 - {args.characters}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"错误：JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        # 支持直接传入角色列表或包含 characters 的字典
        if isinstance(char_data, list):
            characters = char_data
        elif isinstance(char_data, dict) and "characters" in char_data:
            characters = char_data["characters"]
        else:
            print("警告：角色文件格式不支持，忽略角色信息", file=sys.stderr)

    # ── 生成分镜 ──────────────────────────────────────────────────────────
    print(f"🎬 分镜生成器 — 处理 {len(shots_data)} 个镜头...", file=sys.stderr)

    try:
        storyboard = generate_storyboard(
            shots=shots_data,
            characters=characters,
            title=args.title,
        )
    except (ValueError, RuntimeError) as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    # ── 验证 ──────────────────────────────────────────────────────────────
    errors = validate_storyboard(storyboard)
    if errors:
        print("⚠️  验证发现以下问题：", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print("输出仍会生成，但下游可能解析失败。", file=sys.stderr)
    else:
        print("✅ 验证通过：与 orchestrator.py parse_shots() 兼容", file=sys.stderr)

    # ── 输出 JSON ─────────────────────────────────────────────────────────
    json_output = json.dumps(storyboard, ensure_ascii=False, indent=2)
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

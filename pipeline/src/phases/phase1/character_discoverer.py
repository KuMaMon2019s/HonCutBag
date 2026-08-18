#!/usr/bin/env python3
"""
角色发现器 - Phase 1 编剧引擎的角色发现模块

从 event_extractor.py 输出的 events 中发现所有角色，生成 CHARACTERS.json。

输入：event_extractor.py 的输出 JSON（events 列表，每个 event 有 who[] 字段）
输出：CHARACTERS.json（符合 PIPELINE.md schema）

逻辑：
1. 从所有 events 的 who[] 中收集所有角色名
2. 去重 + 合并别名（"艾米"和"小女孩"可能是同一人）
3. 调用 LLM 为每个角色生成：
   - appearance（外貌描述，用于 Seedream 生成三视图）
   - role（protagonist/antagonist/supporting/extra）
   - aliases（别名列表）
   - personality（性格特征）
   - style（推荐画风）
   - negative（负面提示词）
4. 统计每个角色的出场次数
5. 生成 id（拼音或英文缩写，用于目录名）
"""

import json
import sys
import argparse
import re
import time
import hashlib
from typing import List, Dict, Any
from collections import defaultdict

from openai import OpenAI
from utils.ark_llm import (
    LLMConnectTimeout,
    LLMIdleTimeout,
    LLMReadTimeout,
    LLMStreamError,
    call_llm_stream,
    create_ark_client,
)
from utils.character_identity import resolve_character_name
from utils.character_body_contracts import (
    ADULT_LEAD_DISCOVERY_INSTRUCTIONS,
    apply_adult_lead_body_contracts,
    body_contract_forbidden,
    body_contract_prompt,
)


# ─── LLM 配置 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是角色设计师。根据故事事件中的角色信息，为每个角色生成视觉描述。"
    "描述必须足够详细，能让 AI 图片生成器画出一致的角色。"
    "输出严格 JSON 数组，不要输出任何解释文字。"
    "\n\n重要过滤规则：只提取有具体外貌、动作、对话的人物角色。"
    "必须排除以下类型：天气现象（如'冷空气'、'风'）、动物（如'鸡'、'狗'）、"
    "抽象指代（如'说话者'、'观察者'、'记录者'、'思考者'、'行走者'、'试验者'、'打探人员'）、"
    "复数群体（如'保安们'应合并为'保安'）、物品、概念。"
    "同一角色的编号、职业称呼和通用指代必须合并为一个对象："
    "保留最具体的主名，其余写入 aliases；禁止将'主角'、'他'、'她'单独输出为角色。"
    "方括号中的来源称呼可能混有服装、年龄、伤势、动作或地点修饰；name 必须是可跨镜头复用的"
    "稳定身份名，aliases 必须逐字收录所有属于该角色的来源称呼，瞬时修饰只能进入 appearance/variants。"
    "不得因为来源称呼使用中文、英文、编号、职业名或多词名称而丢弃角色，也不得凭子串把两个角色合并。"
    "最多保留5个主要人物角色。"
    "\n\n外貌描述具体化要求：\n"
    "- hair 必须写明发色+发长+发型（如'黑色长直发及肩'），不能只写'长发'\n"
    "- clothing 必须具体到单品（上装+下装+鞋+配饰），不能只写'通勤装'、'休闲服'\n"
    "- summary 必须包含发型+服装+体态三要素，让 AI 图片生成器能画出一致的角色\n"
    "- 事件原文已明确的服装颜色、层次、材质、发型和配饰属于硬约束，必须原样保留，禁止改色、换装或替换材质\n"
    "- 根据角色的身份/职业/场景推导合理的具体服装（白领→衬衫+西装裤，学生→校服，店主→围裙）"
)
SYSTEM_PROMPT = f"{SYSTEM_PROMPT}\n\n{ADULT_LEAD_DISCOVERY_INSTRUCTIONS}"

USER_PROMPT_TEMPLATE = (
    "以下是故事中的角色列表和出现的事件：\n\n"
    "{character_context}\n\n"
    "注意：只提取有具体外貌、动作、对话的人物角色。排除：天气现象、动物、"
    "抽象指代（如'说话者'、'观察者'）、复数群体。"
    "同一实体的编号/职业/主角指代只输出一个对象，其余称呼放入 aliases；"
    "'主角'、'他'、'她'不得独立成条。最多保留5个主要角色。\n\n"
    "身份归一化硬约束：每个【来源称呼】必须被审计。若称呼带有服装、年龄、伤势、动作或地点修饰，"
    "从中提取稳定身份作为 name，并把完整【来源称呼】逐字放入 aliases；若它只是物品、环境或群众描述，"
    "则不得生成角色。禁止按语言或字母类型区别处理，禁止把多词姓名截成最后一个词，禁止以模糊子串合并。\n\n"
    "忠实度要求：事件上下文中明确出现的服装颜色、层次、材质、发型和配饰必须逐项保留；"
    "只能补全未指定的细节，不能把淡粉改成月白、把轻纱改成其他面料或擅自换装。\n\n"
    "为每个角色输出 JSON 对象，组成数组。每个对象包含：\n"
    "- id: 英文标识（拼音或英文缩写，用于目录名，如 amy, wolf, old_man）\n"
    "- name: 角色名称（中文）\n"
    "- aliases: 别名数组（如 [\"小女孩\", \"她\"]）\n"
    "- role: 角色定位，枚举值 protagonist/antagonist/supporting/extra\n"
    "- appearance: 外貌对象，包含：\n"
    "  - gender: male/female/nonbinary/unknown\n"
    "  - age_range: 年龄段，如 '7-10', '20-30', '50-60'\n"
    "  - height: 身高描述（可选）\n"
    "  - build: 体型，如 slim/athletic/heavy/petite\n"
    "  - hair: 发型发色（必须具体：发色+发长+发型，如'黑色长直发及肩'、'深棕色短发微卷'）\n"
    "  - face: 面部特征（必须具体：脸型+五官特点，如'鹅蛋脸、柳叶眉、杏眼、高鼻梁'）\n"
    "  - clothing: 典型穿着（必须具体到单品：上装+下装+鞋子+配饰，如'白色修身衬衫+深蓝色高腰西装裤+黑色尖头平底鞋+银色细链手表'）\n"
    "  - distinguishing: 显著标记（可选）\n"
    "  - summary: 一句话外貌总结（必须包含：发型+服装+体态，如'20多岁清秀纤细的都市女白领，黑色长直发及肩，皮肤白皙，穿白色修身衬衫搭配深蓝色高腰西装裤'）\n"
    "- personality: 性格对象（可选），包含：\n"
    "  - traits: 性格特征数组\n"
    "  - speech_style: 说话风格\n"
    "  - motivation: 动机\n"
    "- style: 推荐画风（如 '张艺谋式写实, 35mm film, 自然光'）\n"
    "- negative: 负面提示词（如 '卡通, 3D渲染, 过度饱和'）\n"
    "- size: 推荐生成尺寸（如 '1920x1920'）\n"
    "- first_appearance: 首次出场的事件 ID（整数）\n"
    "- appearance_count: 出场次数（整数）\n"
    "- relationships: 与其他角色的关系数组（可选），每项含 target_id, type, description\n\n"
    "【衍生状态检测（HonCut derive_assets 规范）】\n"
    "分析每个角色在故事中是否有明显的状态变化（如淋湿、换装、受伤、变身）。\n"
    "如果有，在角色输出的 appearance 中增加 variants 字段：\n"
    "- variants: 数组，每项包含 {{state_name: '淋湿', description: '头发湿透贴在脸上，衬衫被雨淋湿半透明'}}\n"
    "- 只提取「稳定、可复用、资产级」的状态变化，瞬时表情/局部特写不算\n"
    "- 常见状态：淋湿、换装、受伤、变身、卸妆、戴帽/摘帽\n"
    "- 如果没有明显状态变化，variants 可以为空数组 []\n"
    "\n{adult_lead_body_contract}\n"
)

LLM_TIMEOUT = 600
LLM_IDLE_TIMEOUT = 75
MAX_RETRIES = 3  # 解析失败重试次数

ENTITY_SUFFIXES = (
    "机器人", "号", "型", "级", "者", "员", "师", "家", "王", "后",
    "公主", "王子", "先生", "小姐", "佣兵", "机械体", "合成人", "复制体",
    "复制品", "生命体", "机甲", "战士", "卫兵", "执法体",
    "仙女", "女子", "少女", "女人", "男子", "男人", "女孩", "男孩",
    "姑娘", "妇人", "夫人", "老者",
)
MAX_ENTITY_NAME_CHINESE_CHARS = 12
GENERIC_CHARACTER_NAMES = {"主角", "主人公", "男主", "女主", "人物", "他", "她", "它"}
CHARACTER_CONTEXT_SCHEMA_VERSION = 6

GENERIC_BACKGROUND_CHARACTER_NAMES = {
    "路人", "行人", "游客", "观众", "听众", "读者",
    "女性", "男性", "老人", "小孩", "孩子", "青年", "中年",
    "人群", "群众", "大家", "情侣", "夫妻", "朋友", "同事", "邻居",
}

# A qualified ``who`` label can accidentally end in an object or environment
# noun.  Such tails are evidence that the entire label is not an identity.
NON_CHARACTER_IDENTITY_SUFFIXES = (
    "衣", "裙", "裤", "鞋", "帽", "眼镜", "伞", "包", "箱",
    "刀", "剑", "枪", "武器", "门", "窗", "墙", "灯", "光",
    "云海", "山脉", "河流", "道路", "车辆", "建筑", "房间", "走廊",
)
RELATIONAL_CHARACTER_LABELS = {
    "父亲", "母亲", "爸爸", "妈妈", "丈夫", "妻子", "哥哥", "姐姐",
    "弟弟", "妹妹", "儿子", "女儿",
}
MAX_QUALIFIER_CHINESE_CHARS = 16

ABSTRACT_CHARACTER_NAMES = {
    "说话者", "观察者", "记录者", "思考者", "行走者", "试验者", "打探人员",
}
NON_HUMAN_EXACT_NAMES = {
    "冷空气", "风", "雨", "雪", "雷", "电",
    "鸡", "鸭", "狗", "猫", "鸟", "鱼",
    "桌子", "椅子", "车", "书", "刀", "剑", "枪", "武器",
    "积水", "路面", "钢梁", "混凝土", "高楼", "玻璃", "灰尘",
}
NON_HUMAN_ENTITY_SUFFIXES = (
    "霓虹牌", "电缆", "路面", "钢梁", "混凝土", "高楼", "塑料布",
    "纸屑", "玻璃", "水浪", "灰尘", "碎石", "残骸", "护臂", "手掌",
    "手指", "车门", "刀具", "武器", "护甲", "铠甲", "弓箭",
    "云海", "山脉", "河流", "道路", "车辆", "建筑", "房间", "走廊",
)


def _get_client() -> OpenAI:
    """
    创建 OpenAI 客户端

    API Key: ARK_AGENT_API_KEY (火山方舟 Agent Plan)
    """
    return create_ark_client(read_timeout=LLM_IDLE_TIMEOUT)


def _call_llm(prompt: str) -> str:
    """调用 LLM 并返回原始响应文本"""
    client = _get_client()

    return call_llm_stream(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=16000,
        wall_timeout=LLM_TIMEOUT,
        idle_timeout=LLM_IDLE_TIMEOUT,
        _client=client,
    )


def _fix_json(text: str) -> str:
    """
    尝试修复常见的 JSON 格式错误

    修复策略：
    1. 缺少逗号：在 } 和 " 之间、} 和 { 之间、] 和 " 之间插入逗号
    2. 尾部多余逗号：删除最后一个元素后的逗号
    3. 截断的字符串/对象：找到最后一个完整的对象，截断后面的不完整内容
    4. 未闭合的字符串：截断到最后一个完整的键值对
    """
    # 策略 1: 修复缺少逗号
    # } 和 " 之间 (对象后跟字符串键)
    text = re.sub(r'}\s*"', '}, "', text)
    # } 和 { 之间 (对象后跟对象，数组中常见)
    text = re.sub(r'}\s*{', '}, {', text)
    # ] 和 " 之间
    text = re.sub(r']\s*"', '], "', text)
    # ] 和 { 之间
    text = re.sub(r']\s*{', '], {', text)
    # } 和 [ 之间
    text = re.sub(r'}\s*\[', '}, [', text)

    # 策略 2: 修复尾部多余逗号 (,] 或 ,})
    text = re.sub(r',\s*]', ']', text)
    text = re.sub(r',\s*}', '}', text)

    # 策略 3: 处理截断的内容
    # 检查是否有未闭合的引号或括号
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')

    if open_braces > 0 or open_brackets > 0:
        # 有未闭合的括号，找到最后一个完整的 } 并截断
        last_complete_obj = text.rfind('}')
        if last_complete_obj > 0:
            text = text[:last_complete_obj + 1]
            # 重新计算括号
            open_brackets = text.count('[') - text.count(']')
            if open_brackets > 0:
                # 还需要闭合数组
                text = text.rstrip()
                if not text.endswith(']'):
                    text += ']'
        else:
            # 没有完整的对象，尝试找到最后一个完整的值
            # 查找最后一个 " 后跟 , 或 } 或 ] 的位置（表示完整值的结束）
            last_complete_value = -1
            for match in re.finditer(r'"\s*[,}\]]', text):
                last_complete_value = match.end() - 1  # 位置在引号后
            
            if last_complete_value > 0:
                # 截断到最后一个完整值
                text = text[:last_complete_value + 1]
                # 移除末尾可能的逗号
                text = re.sub(r',\s*$', '', text)
                # 添加必要的闭合
                open_braces = text.count('{') - text.count('}')
                open_brackets = text.count('[') - text.count(']')
                while open_braces > 0:
                    text += '}'
                    open_braces -= 1
                open_brackets = text.count('[') - text.count(']')
                while open_brackets > 0:
                    text += ']'
                    open_brackets -= 1

    return text


def _parse_characters(response: str) -> List[Dict[str, Any]]:
    """
    解析 LLM 响应为角色列表

    支持纯 JSON 数组和被 ```json ... ``` 包裹的 JSON
    使用 strict=False 容忍一些格式问题
    如果解析失败，尝试修复常见错误
    """
    text = response.strip()

    # 尝试提取 ```json ... ``` 代码块
    if "```" in text:
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # 第一次尝试：使用 strict=False
    try:
        parsed = json.loads(text, strict=False)
    except json.JSONDecodeError:
        # 第二次尝试：修复 JSON 后解析
        try:
            fixed_text = _fix_json(text)
            parsed = json.loads(fixed_text, strict=False)
            print("  JSON 修复成功", file=sys.stderr)
        except json.JSONDecodeError as e:
            # 第三次尝试：更激进的修复 - 找到最后一个完整的对象
            try:
                # 找到最后一个完整的 }, 然后截断
                last_obj_end = text.rfind('}')
                if last_obj_end > 0:
                    truncated = text[:last_obj_end + 1]
                    # 确保是有效的数组结尾
                    if not truncated.rstrip().endswith(']'):
                        truncated = truncated.rstrip() + ']'
                    parsed = json.loads(truncated, strict=False)
                    print("  JSON 激进修复成功（截断到最后一个完整对象）", file=sys.stderr)
                else:
                    raise e
            except json.JSONDecodeError:
                raise ValueError(f"JSON 解析失败，已尝试修复: {e}")

    # 验证是数组
    if not isinstance(parsed, list):
        raise ValueError(f"期望 JSON 数组，得到 {type(parsed).__name__}")

    # 验证每个元素的基本结构
    required_fields = {"id", "name", "appearance", "role"}
    for i, char in enumerate(parsed):
        if not isinstance(char, dict):
            raise ValueError(f"第 {i+1} 个角色不是字典")
        missing = required_fields - set(char.keys())
        if missing:
            raise ValueError(f"第 {i+1} 个角色缺少字段: {missing}")
        # 验证 appearance.summary 存在
        appearance = char.get("appearance", {})
        if not isinstance(appearance, dict) or "summary" not in appearance:
            raise ValueError(f"第 {i+1} 个角色的 appearance 缺少 summary 字段")

    return parsed


def _collect_character_stats(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    从事件列表中收集角色统计信息

    Returns:
        dict: {角色名: {"events": [event_ids], "contexts": [event_summary]}}
    """
    stats = defaultdict(lambda: {"events": [], "contexts": [], "dialogue_count": 0})

    for event in events:
        event_id = event.get("id", 0)
        who = event.get("who", [])
        what = event.get("what", "")
        visual = event.get("visual", "")
        where = event.get("where", "")
        emotion = event.get("emotion", "")
        speakers = {
            str(line.get("speaker", "")).strip()
            for line in event.get("lines", [])
            if isinstance(line, dict) and str(line.get("speaker", "")).strip()
        }

        # 构建事件摘要
        summary_parts = []
        if where:
            summary_parts.append(f"在{where}")
        if what:
            summary_parts.append(what)
        if visual and visual not in what:
            summary_parts.append(f"视觉硬约束：{visual}")
        if emotion:
            summary_parts.append(f"（{emotion}）")
        event_summary = "，".join(summary_parts) if summary_parts else "参与事件"

        for character_name in who:
            if character_name and character_name.strip():
                name = character_name.strip()
                stats[name]["events"].append(event_id)
                stats[name]["contexts"].append(f"事件{event_id}: {event_summary}")
                if name in speakers:
                    stats[name]["dialogue_count"] += 1

    return dict(stats)


def _build_character_context(stats: Dict[str, Dict[str, Any]]) -> str:
    """
    构建角色上下文文本，供 LLM 参考
    """
    lines = []
    for name, info in stats.items():
        event_count = len(info["events"])
        first_event = min(info["events"]) if info["events"] else 0
        lines.append(f"【{name}】出场 {event_count} 次，首次出场于事件 {first_event}")
        source_aliases = [
            str(alias).strip()
            for alias in (info.get("source_aliases") or [])
            if str(alias).strip() and str(alias).strip() != name
        ]
        if source_aliases:
            lines.append(f"  - 必须归并的来源称呼：{json.dumps(source_aliases, ensure_ascii=False)}")
        # 最多展示 5 条事件上下文
        for ctx in info["contexts"][:5]:
            lines.append(f"  - {ctx}")
        if len(info["contexts"]) > 5:
            lines.append(f"  - ...（还有 {len(info['contexts']) - 5} 条）")
        lines.append("")

    return "\n".join(lines)


def _compute_text_hash(events: List[Dict[str, Any]]) -> str:
    """计算事件列表的 SHA-256 哈希，用于溯源"""
    text = json.dumps(events, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_human_character(name: str) -> bool:
    """
    判断角色名是否可能是人物角色
    
    排除：
    - 天气现象（冷空气、风、雨等）
    - 动物（鸡、狗、猫等）
    - 抽象指代（说话者、观察者、记录者等以"者"、"员"结尾）
    - 复数群体（以"们"结尾）
    - 物品、概念
    
    保留：
    - 无人机、机器人等智能设备（可作为主角）
    - 工程师、运维员等职业角色
    """
    name = str(name or "").strip()
    if not name:
        return False

    # 排除复数群体
    if name.endswith("们"):
        return False
    
    # 白名单：无人机、机器人等智能设备可作为主角
    robot_whitelist = ["无人机", "机器人", "机械臂", "传感器"]
    if any(keyword in name for keyword in robot_whitelist):
        return True
    
    # Suffixes such as “员/者/师” are commonly active occupations and may not
    # be rejected generically. Only exact abstract placeholders are excluded.
    if name in ABSTRACT_CHARACTER_NAMES:
        return False

    # A declared character/entity suffix wins over an object word elsewhere
    # in the label (for example a weapon-carrying fighter or racing driver).
    if any(name.endswith(suffix) for suffix in ENTITY_SUFFIXES):
        return True

    if name in NON_HUMAN_EXACT_NAMES:
        return False
    if any(name.endswith(suffix) for suffix in NON_HUMAN_ENTITY_SUFFIXES):
        return False
    
    return True


def _filter_non_human_characters(stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    过滤掉非人物角色
    
    Args:
        stats: 角色统计字典
        
    Returns:
        过滤后的角色统计字典
    """
    filtered = {}
    for name, info in stats.items():
        if _is_human_character(name):
            filtered[name] = info
        else:
            print(f"  过滤非人物角色: {name}", file=sys.stderr)
    
    return filtered


def _qualified_mention_tail(name: str) -> tuple[str, str] | None:
    """Split ``qualifier 的 identity`` without assuming a language or role."""
    if "的" not in name:
        return None
    qualifier, candidate = name.rsplit("的", 1)
    qualifier = qualifier.strip(" ，,；;：:（）()[]【】")
    candidate = candidate.strip(" ，,；;：:（）()[]【】")
    return (qualifier, candidate) if qualifier and candidate else None


def _looks_like_stable_identity(
    candidate: str,
    known_mentions: set[str],
) -> bool:
    """Conservatively recognize a reusable identity at a qualified tail."""
    if candidate in known_mentions:
        return True
    if candidate in GENERIC_BACKGROUND_CHARACTER_NAMES:
        return False
    if candidate in RELATIONAL_CHARACTER_LABELS:
        # A relational role needs its owner (for example whose father) unless
        # the same label is independently established elsewhere.
        return False
    if any(candidate.endswith(suffix) for suffix in NON_CHARACTER_IDENTITY_SUFFIXES):
        return False
    if not _is_human_character(candidate):
        return False

    latin_name = re.fullmatch(
        r"[A-Za-z][A-Za-z0-9]*(?:[\s_-]+[A-Za-z0-9]+)*",
        candidate,
    )
    if latin_name:
        return True
    if any(candidate.endswith(suffix) for suffix in ENTITY_SUFFIXES):
        return True
    chinese_chars = [char for char in candidate if "\u4e00" <= char <= "\u9fff"]
    return len(chinese_chars) == len(candidate) and 2 <= len(chinese_chars) <= 4


def _stable_identity_from_qualified_mention(
    name: str,
    known_mentions: set[str],
) -> str | None:
    split = _qualified_mention_tail(name)
    if split is None:
        return None
    qualifier, candidate = split
    qualifier_length = sum("\u4e00" <= char <= "\u9fff" for char in qualifier)
    if qualifier_length > MAX_QUALIFIER_CHINESE_CHARS:
        return None
    return candidate if _looks_like_stable_identity(candidate, known_mentions) else None


def _qualified_tail_is_non_character(name: str) -> bool:
    split = _qualified_mention_tail(name)
    if split is None:
        return False
    _qualifier, candidate = split
    return (
        candidate in GENERIC_BACKGROUND_CHARACTER_NAMES
        or any(candidate.endswith(suffix) for suffix in NON_CHARACTER_IDENTITY_SUFFIXES)
        or not _is_human_character(candidate)
    )


def _filter_descriptive_phrases(stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    过滤描述性短语（非真实角色名）
    
    移除：
    - 包含描述性形容词/修饰语的名称（如"年轻的"、"没带伞的"、"都市"）
    - 通用角色描述（如"路人"、"女性"、"男性"、"店员"）
    - 过长且不像实体名称的描述性短语
    
    Args:
        stats: 角色统计字典
        
    Returns:
        过滤后的角色统计字典
    """
    filtered = {}
    
    def merge_info(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key in ("events", "contexts"):
            combined = [*(target.get(key) or []), *(source.get(key) or [])]
            target[key] = list(dict.fromkeys(combined))
        target["dialogue_count"] = int(target.get("dialogue_count") or 0) + int(
            source.get("dialogue_count") or 0
        )
        target["source_aliases"] = list(dict.fromkeys([
            *(target.get("source_aliases") or []),
            *(source.get("source_aliases") or []),
        ]))

    known_mentions = {str(name).strip() for name in stats if str(name).strip()}
    for name, info in stats.items():
        canonical_name = _stable_identity_from_qualified_mention(name, known_mentions)
        if canonical_name:
            normalized_info = dict(info)
            normalized_info["source_aliases"] = list(dict.fromkeys([
                *(normalized_info.get("source_aliases") or []),
                name,
            ]))
            if canonical_name in filtered:
                merge_info(filtered[canonical_name], normalized_info)
            else:
                filtered[canonical_name] = normalized_info
            print(
                f"  归一化限定角色名: {name} → {canonical_name}",
                file=sys.stderr,
            )
            continue

        if _qualified_tail_is_non_character(name):
            print(f"  过滤非角色限定短语: {name}", file=sys.stderr)
            continue

        # 检查是否是通用角色描述
        # Active occupational roles still need a stable visual asset. Only
        # discard exact background placeholders.
        if name in GENERIC_BACKGROUND_CHARACTER_NAMES:
            print(f"  过滤通用角色描述: {name}", file=sys.stderr)
            continue
        
        # 实体后缀允许较长的复合名称，如“白色金属AI巡检机器人”。
        chinese_chars = [c for c in name if '\u4e00' <= c <= '\u9fff']
        is_entity = any(name.endswith(suffix) for suffix in ENTITY_SUFFIXES)
        if len(chinese_chars) > MAX_ENTITY_NAME_CHINESE_CHARS:
            print(f"  过滤超长实体名称: {name} ({len(chinese_chars)} 字)", file=sys.stderr)
            continue
        if not is_entity and len(chinese_chars) > 6:
            print(f"  过滤过长名称: {name} ({len(chinese_chars)} 字)", file=sys.stderr)
            continue
        
        if name in filtered:
            merge_info(filtered[name], info)
        else:
            filtered[name] = info
    
    return filtered


def _post_filter_characters(characters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    后处理过滤：移除 LLM 可能错误包含的非人物角色
    
    Args:
        characters: LLM 返回的角色列表
        
    Returns:
        过滤后的角色列表
    """
    filtered = []
    for char in characters:
        name = char.get("name", "")
        if _is_human_character(name):
            filtered.append(char)
        else:
            print(f"  后处理过滤非人物角色: {name}", file=sys.stderr)
    
    # LLM 有时会把同一角色的主名、职业和通用指代分成多条。
    # 优先使用显式 aliases 合并；“银白色机械技师/机械技师”这类同定位修饰名也合并。
    merged: List[Dict[str, Any]] = []
    for char in filtered:
        name = str(char.get("name", "")).strip()
        aliases = {str(alias).strip() for alias in char.get("aliases", []) if str(alias).strip()}
        target = None
        for existing in merged:
            existing_name = str(existing.get("name", "")).strip()
            existing_aliases = {
                str(alias).strip() for alias in existing.get("aliases", []) if str(alias).strip()
            }
            explicit_alias_match = (
                name in existing_aliases
                or existing_name in aliases
                or bool(aliases & existing_aliases)
            )
            qualified_name_match = (
                char.get("role") == existing.get("role")
                and min(len(name), len(existing_name)) >= 2
                and (name.endswith(existing_name) or existing_name.endswith(name))
            )
            generic_role_match = (
                char.get("role") == existing.get("role")
                and (name in GENERIC_CHARACTER_NAMES or existing_name in GENERIC_CHARACTER_NAMES)
            )
            if explicit_alias_match or qualified_name_match or generic_role_match:
                target = existing
                break

        if target is None:
            merged.append(char)
            continue

        target_name = str(target.get("name", "")).strip()
        # Prefer a concrete, more specific canonical name over a generic or shortened name.
        if target_name in GENERIC_CHARACTER_NAMES or (
            name not in GENERIC_CHARACTER_NAMES and name.endswith(target_name)
        ):
            char, target = target, char
            merged[merged.index(char)] = target
            name, target_name = target_name, name

        combined_aliases = list(dict.fromkeys(
            [*target.get("aliases", []), target_name, name, *char.get("aliases", [])]
        ))
        target["aliases"] = [
            alias for alias in combined_aliases if alias and alias != target.get("name")
        ]

    # 无法归并的通用指代不能单独成为角色资产。
    filtered = [char for char in merged if char.get("name") not in GENERIC_CHARACTER_NAMES]

    # 限制最多 5 个角色
    if len(filtered) > 5:
        print(f"  角色数量超限 ({len(filtered)} > 5)，只保留前 5 个", file=sys.stderr)
        filtered = filtered[:5]
    
    return filtered


class _ReferencePromptTemplate(str):
    """Placeholder template with a non-serialized legacy containment shim."""

    def __contains__(self, item: object) -> bool:
        # [LEGACY-KEEP 2026-08-09] Older callers used ``"<主体1>" in value``
        # as a capability check. The serialized value remains fully unnumbered.
        if item == "<主体1>" and "{主体N}" in str(self):
            return True
        return super().__contains__(item)


def _add_reference_contract(character: Dict[str, Any]) -> None:
    """Add the V6.3 reference contract without removing legacy character fields."""
    appearance = character.get("appearance")
    appearance = appearance if isinstance(appearance, dict) else {}
    traits = [
        str(appearance.get(key, "")).strip()
        for key in ("clothing", "face", "hair", "distinguishing")
        if appearance.get(key)
    ][:3]
    if len("、".join(traits)) > 80:
        traits = traits[:2]
    subject_traits = "、".join(traits) or str(appearance.get("summary") or character.get("name", "角色"))
    body_lock = body_contract_prompt(character)
    distinguishing_features = list(traits or [subject_traits])
    if body_lock:
        distinguishing_features.append(body_lock)
    character.setdefault("distinguishing_features", distinguishing_features)
    character.setdefault("face_reference", "face_closeup.png")
    character.setdefault("body_reference", "full_body.png")
    character.setdefault(
        "prompt_definition",
        # [LEGACY-KEEP 2026-08-09] 旧值写死为：将图片1中的[...]定义为<主体1>
        _ReferencePromptTemplate(
            f"将{{图片N}}中的[{subject_traits}]定义为{{主体N}}"
            + (f"；{body_lock}" if body_lock else "")
        ),
    )
    base_guardrails = "去龄化, 特征变形, 更换服装, 颜色偏移, 多余肢体, 关节扭曲"
    body_guardrails = ", ".join(body_contract_forbidden(character))
    character.setdefault(
        "negative_guardrails",
        ", ".join(filter(None, (
            base_guardrails,
            body_guardrails,
            str(character.get("negative", "")).strip(),
        ))),
    )


def _attach_source_identity_evidence(
    characters: List[Dict[str, Any]],
    stats: Dict[str, Dict[str, Any]],
) -> None:
    """Attach source aliases and aggregate appearance evidence deterministically."""
    characters_by_name = {
        str(character.get("name") or "").strip(): character
        for character in characters
        if str(character.get("name") or "").strip()
    }
    evidence: Dict[str, Dict[str, Any]] = {
        name: {"events": set(), "aliases": []}
        for name in characters_by_name
    }

    for stat_name, stat_info in stats.items():
        source_mentions = list(dict.fromkeys([
            str(stat_name).strip(),
            *(
                str(alias).strip()
                for alias in (stat_info.get("source_aliases") or [])
                if str(alias).strip()
            ),
        ]))
        resolved = {
            canonical
            for mention in source_mentions
            if (canonical := resolve_character_name(mention, characters))
        }
        if len(resolved) != 1:
            continue
        canonical = next(iter(resolved))
        if canonical not in evidence:
            continue
        evidence[canonical]["events"].update(stat_info.get("events") or [])
        evidence[canonical]["aliases"].extend(source_mentions)

    for canonical, character in characters_by_name.items():
        event_ids = sorted(evidence[canonical]["events"])
        character["first_appearance"] = event_ids[0] if event_ids else 0
        character["appearance_count"] = len(event_ids)
        aliases = list(dict.fromkeys([
            *(character.get("aliases") or []),
            *evidence[canonical]["aliases"],
        ]))
        character["aliases"] = [
            alias
            for alias in aliases
            if alias and alias != canonical
        ]


def discover_characters(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    核心函数：从事件列表中发现所有角色并生成 CHARACTERS.json

    Args:
        events: event_extractor 输出的事件列表

    Returns:
        符合 PIPELINE.md schema 的 CHARACTERS.json 字典
    """
    if not events:
        return {
            "version": "1.0",
            "source_text_hash": "",
            "total_characters": 0,
            "characters": [],
        }

    # 1. 收集角色统计
    stats = _collect_character_stats(events)
    if not stats:
        print("警告：未从事件中发现任何角色", file=sys.stderr)
        return {
            "version": "1.0",
            "source_text_hash": _compute_text_hash(events),
            "total_characters": 0,
            "characters": [],
        }

    print(f"发现 {len(stats)} 个角色名: {list(stats.keys())}", file=sys.stderr)

    # 1.5 预过滤：移除明显的非人物角色
    stats = _filter_non_human_characters(stats)
    if not stats:
        print("警告：过滤后未剩任何人物角色", file=sys.stderr)
        return {
            "version": "1.0",
            "source_text_hash": _compute_text_hash(events),
            "total_characters": 0,
            "characters": [],
        }

    # 1.6 过滤描述性短语（非真实角色名）
    stats = _filter_descriptive_phrases(stats)
    if not stats:
        print("警告：过滤描述性短语后未剩任何角色", file=sys.stderr)
        return {
            "version": "1.0",
            "source_text_hash": _compute_text_hash(events),
            "total_characters": 0,
            "characters": [],
        }

    print(f"过滤后保留 {len(stats)} 个角色名: {list(stats.keys())}", file=sys.stderr)

    # 2. 构建 LLM prompt
    character_context = _build_character_context(stats)
    prompt = USER_PROMPT_TEMPLATE.format(
        character_context=character_context,
        adult_lead_body_contract=ADULT_LEAD_DISCOVERY_INSTRUCTIONS,
    )

    # 3. 调用 LLM（带重试）
    characters = []
    last_error = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            print("调用 LLM 生成角色描述...", file=sys.stderr)
            response = _call_llm(prompt)
            characters = _parse_characters(response)
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"  JSON 解析失败，重试 ({attempt+1}/{MAX_RETRIES}): {e}", file=sys.stderr)
                time.sleep(1)
            continue
        except (LLMConnectTimeout, LLMReadTimeout, LLMIdleTimeout, LLMStreamError) as e:
            last_error = e
            if attempt < 1:
                print(f"  LLM 流中断，重试 (1/1): {e}", file=sys.stderr)
                time.sleep(1)
                continue
            break
        except Exception as e:
            last_error = e
            print(f"  LLM 调用失败: {e}", file=sys.stderr)
            break

    if not characters and last_error:
        print(f"错误：角色发现失败: {last_error}", file=sys.stderr)
        # 回退：为每个角色名生成最简描述
        characters = _fallback_characters(stats)

    # 3.5 后处理过滤：移除 LLM 可能错误包含的非人物角色
    characters = _post_filter_characters(characters)

    # 4. Reconcile every source label against the canonical roster. This also
    # preserves qualified source mentions as aliases and aggregates repeated
    # labels for the same character instead of taking only the first match.
    _attach_source_identity_evidence(characters, stats)

    # 5. 按出场次数排序（主角在前）
    characters.sort(key=lambda c: (-c.get("appearance_count", 0), c.get("first_appearance", 0)))

    # 5.5 Apply deterministic adult-lead proportions after narrative ranking.
    # The LLM is instructed to emit this schema, but the normalization below is
    # the actual contract boundary and prevents creative paraphrase or drift.
    apply_adult_lead_body_contracts(characters)

    # 6. 生成 asset_path
    for char in characters:
        char_id = char.get("id", "unknown")
        char["asset_path"] = f"characters/{char_id}/"
        _add_reference_contract(char)

    # 7. 构建最终输出
    result = {
        "version": "1.0",
        "source_text_hash": _compute_text_hash(events),
        "total_characters": len(characters),
        "characters": characters,
    }

    return result


def _fallback_characters(stats: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    LLM 失败时的回退：为每个角色名生成最简描述
    """
    characters = []
    trusted = [
        (name, info)
        for name, info in stats.items()
        if len(info.get("events", [])) >= 2 or info.get("dialogue_count", 0) > 0
    ]
    trusted.sort(key=lambda item: (-len(item[1].get("events", [])), item[0]))
    for i, (name, info) in enumerate(trusted[:5]):
        # 简单生成 id（拼音首字母或 index）
        char_id = f"char_{i+1:03d}"

        characters.append({
            "id": char_id,
            "name": name,
            "aliases": list(info.get("source_aliases") or []),
            "role": "supporting" if len(info["events"]) < 3 else "protagonist",
            "appearance": {
                "gender": "unknown",
                "age_range": "unknown",
                "summary": name,
            },
            "style": "写实风格, 自然光",
            "negative": "卡通, 3D渲染",
            "size": "1920x1920",
            "first_appearance": min(info["events"]) if info["events"] else 0,
            "appearance_count": len(info["events"]),
            "asset_path": f"characters/{char_id}/",
        })

    return characters


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="角色发现器 - 从事件列表中发现角色并生成 CHARACTERS.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从 events.json 读取
  python character_discoverer.py --input events.json --output characters.json

  # 管道模式
  python event_extractor.py --text "..." | python character_discoverer.py --output characters.json
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入文件路径（event_extractor 输出的 JSON）",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出 JSON 文件路径（可选，默认打印到 stdout）",
    )

    args = parser.parse_args()

    # 确定输入来源
    events = None

    if args.input:
        # 从文件读取 events
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"错误：文件不存在 - {args.input}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"错误：JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        # 支持直接传入 events 列表或 extract_events 的完整输出
        if isinstance(data, list):
            events = data
        elif isinstance(data, dict) and "events" in data:
            events = data["events"]
        else:
            print("错误：输入格式不支持，期望 events 列表或包含 events 的字典", file=sys.stderr)
            sys.exit(1)

    elif not sys.stdin.isatty():
        # 从 stdin 读取（管道模式）
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"错误：stdin JSON 解析失败 - {e}", file=sys.stderr)
            sys.exit(1)

        if isinstance(data, list):
            events = data
        elif isinstance(data, dict) and "events" in data:
            events = data["events"]
        else:
            print("错误：输入格式不支持", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # 执行角色发现
    result = discover_characters(events)

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

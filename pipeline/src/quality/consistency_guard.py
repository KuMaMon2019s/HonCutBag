#!/usr/bin/env python3
"""
consistency_guard.py — 角色一致性检查模块

检查三视图生成后，角色特征是否在视频生成的 shot prompts 中保持一致。
对比 CHARACTERS.json 中的 appearance 字段与 STORYBOARD.json 中的 shot prompts。
"""

import json
import re
from pathlib import Path
from typing import Optional


# 默认一致性阈值
DEFAULT_THRESHOLD = 70

# 需要从 appearance 中提取用于匹配的关键字段
APPEARANCE_KEY_FIELDS = [
    "hair",
    "clothing",
    "face",
    "build",
    "gender",
    "age_range",
]


def _extract_character_features(character: dict) -> list:
    """
    从角色数据中提取关键特征
    
    Args:
        character: 角色数据 dict
    
    Returns:
        特征列表，如 ["黑色短发", "休闲装"]
    """
    appearance = character.get("appearance", {})
    features = []
    
    for field in APPEARANCE_KEY_FIELDS:
        value = appearance.get(field)
        if value and isinstance(value, str):
            features.append(value)
    
    return features


def _fuzzy_match(feature: str, prompt: str) -> bool:
    """
    模糊匹配：去除常见虚词后匹配
    
    Args:
        feature: 角色特征，如 "坚毅的面容"
        prompt: 镜头描述，如 "坚毅面容的男人看着窗外"
    
    Returns:
        是否匹配
    """
    # 精确匹配
    if feature.lower() in prompt.lower():
        return True
    
    # 去除常见虚词后匹配
    # 移除 "的"、"了"、"在"、"是" 等虚词
    feature_clean = re.sub(r'[的了在是]', '', feature).lower()
    prompt_clean = re.sub(r'[的了在是]', '', prompt).lower()
    
    if feature_clean in prompt_clean:
        return True
    
    # 关键词匹配：提取特征中的核心词汇
    # 例如 "坚毅的面容" -> ["坚毅", "面容"]
    keywords = re.findall(r'[\u4e00-\u9fa5]+', feature)
    if keywords:
        # 只要有一个关键词在 prompt 中就认为匹配
        for keyword in keywords:
            if keyword in prompt:
                return True
    
    return False


def _shot_id(shot: dict) -> str:
    """Return a stable display id for both numeric and prefixed shot ids."""
    raw_id = shot.get("id", "?")
    if isinstance(raw_id, int):
        return f"S{raw_id:02d}"
    text = str(raw_id)
    return text if text.upper().startswith("S") else f"S{text.zfill(2)}"


def _character_is_in_shot(character_name: str, shot: dict) -> bool:
    """Use explicit cast metadata when present; otherwise keep legacy behavior."""
    # Presence of ``who`` is authoritative, including an explicitly empty list.
    # Treating ``who: []`` as missing metadata makes every scenery-only shot a
    # character shot and unfairly lowers the consistency score.
    if "who" in shot:
        who = shot.get("who")
        if isinstance(who, list):
            return character_name in {str(item) for item in who}
        if isinstance(who, str):
            return bool(who.strip()) and character_name in who
        if who is None:
            return False

    assets = shot.get("associate_assets")
    if isinstance(assets, list) and assets:
        character_assets = {
            str(item).split(":", 1)[1]
            for item in assets
            if str(item).startswith("char:")
        }
        if character_assets:
            return character_name in character_assets
    return True


def _shot_character_contract(shot: dict) -> str:
    """Collect all prompt-contract fields that describe on-screen identity."""
    fields = (
        "prompt",
        "subject_description",
        "visual",
        "action_description",
        "what",
    )
    return "\n".join(str(shot.get(field, "")) for field in fields)


def check_character_consistency(
    characters_data: dict,
    shots_data: dict,
    threshold: int = DEFAULT_THRESHOLD,
) -> dict:
    """
    检查角色一致性
    
    Args:
        characters_data: CHARACTERS.json 的内容
        shots_data: STORYBOARD.json 的内容
        threshold: 一致性阈值（0-100），默认 70
    
    Returns:
        {
            "consistency_score": 85,
            "passed": True,
            "inconsistent_characters": [
                {
                    "name": "主角",
                    "expected_features": ["黑色短发", "休闲装"],
                    "missing_in_shots": ["S03", "S07"]
                }
            ]
        }
    """
    characters = characters_data.get("characters", [])
    shots = shots_data.get("shots", [])
    
    if not characters or not shots:
        return {
            "consistency_score": 0,
            "passed": False,
            "inconsistent_characters": [],
            "error": "No characters or shots data"
        }
    
    inconsistent_characters = []
    total_checks = 0
    passed_checks = 0
    
    # 对每个角色进行检查
    for character in characters:
        name = character.get("name", "Unknown")
        
        # 提取角色的关键特征
        expected_features = _extract_character_features(character)
        
        if not expected_features:
            continue
        
        # 只检查角色实际出场的镜头，并覆盖新旧两套分镜字段契约。
        missing_in_shots = []
        relevant_shots = []
        for shot in shots:
            if not _character_is_in_shot(name, shot):
                continue
            relevant_shots.append(shot)
            prompt = _shot_character_contract(shot)
            
            # 检查是否有任何特征在 prompt 中缺失
            feature_found = False
            for feature in expected_features:
                # 使用模糊匹配
                if _fuzzy_match(feature, prompt):
                    feature_found = True
                    break
            
            if not feature_found:
                missing_in_shots.append(_shot_id(shot))
            
            total_checks += 1
            if feature_found:
                passed_checks += 1
        
        # 如果角色在超过 30% 的 shots 中缺失特征，标记为不一致
        if relevant_shots and len(missing_in_shots) > len(relevant_shots) * 0.3:
            inconsistent_characters.append({
                "name": name,
                "expected_features": expected_features,
                "missing_in_shots": missing_in_shots[:5]  # 只显示前 5 个
            })
    
    # 计算一致性分数
    consistency_score = int((passed_checks / total_checks * 100)) if total_checks > 0 else 0
    passed = consistency_score >= threshold
    
    return {
        "consistency_score": consistency_score,
        "passed": passed,
        "inconsistent_characters": inconsistent_characters,
        "total_checks": total_checks,
        "passed_checks": passed_checks
    }


def run_consistency_check(
    output_dir: Path,
    threshold: int = DEFAULT_THRESHOLD,
) -> dict:
    """
    从文件读取数据并运行一致性检查
    
    Args:
        output_dir: 输出目录
        threshold: 一致性阈值
    
    Returns:
        检查结果 dict
    """
    characters_path = output_dir / "CHARACTERS.json"
    storyboard_path = output_dir / "STORYBOARD.json"
    
    if not characters_path.exists():
        return {
            "consistency_score": 0,
            "passed": False,
            "error": f"CHARACTERS.json not found at {characters_path}"
        }
    
    if not storyboard_path.exists():
        return {
            "consistency_score": 0,
            "passed": False,
            "error": f"STORYBOARD.json not found at {storyboard_path}"
        }
    
    try:
        characters_data = json.loads(characters_path.read_text(encoding="utf-8"))
        storyboard_data = json.loads(storyboard_path.read_text(encoding="utf-8"))
        
        result = check_character_consistency(
            characters_data,
            storyboard_data,
            threshold
        )
        
        # 写出结果
        result_path = output_dir / "consistency_report.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        return result
        
    except Exception as e:
        return {
            "consistency_score": 0,
            "passed": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # 测试
    import sys
    
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    result = run_consistency_check(output_dir)
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result.get("passed") else 1)

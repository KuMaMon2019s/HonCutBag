"""Character Asset Factory — batch generate character assets for the pipeline.

Generates:
  1. character_card.json — character metadata + generation params
  2. Three-view images (front/side/back.png) via Seedream
  3. angle_map.json — camera angle → best reference image mapping

Usage:
    # Single character
    python character_factory.py --name "艾米" --desc "7岁中国男孩，深色发髻，灰色交领汉服" --style "张艺谋式写实, 35mm film"

    # Batch from JSON
    python character_factory.py --batch characters.json --output-dir ./characters/
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List

# Import seedream client from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seedream_client import SeedreamClient

# Import prompt validator from prompts/ directory
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
sys.path.insert(0, str(_PROMPTS_DIR))
from prompt_validator import validate_prompt


# =============================================================================
# Prompt Template Functions — Load & fill template, then validate before API call
# =============================================================================

def load_template(template_path: Optional[str] = None) -> str:
    """加载提示词模板

    Args:
        template_path: 模板文件路径，默认为 prompts/three_view_template.md
                       可以是绝对路径或相对于项目根目录的路径

    Returns:
        模板文件内容字符串

    Raises:
        FileNotFoundError: 模板文件不存在
    """
    if template_path is None:
        template_path = str(_PROMPTS_DIR / "three_view_template.md")
    else:
        # If relative path, resolve relative to project root (parent of scripts/)
        p = Path(template_path)
        if not p.is_absolute():
            p = Path(__file__).parent.parent / template_path
        template_path = str(p)

    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def fill_template(template: str, character: dict) -> str:
    """将角色信息填充到模板中

    Args:
        template: 模板字符串（含 {{PLACEHOLDER}} 占位符）
        character: 角色信息字典，支持以下字段：
            - name: 角色名称
            - appearance: 外貌描述
            - clothing: 服饰描述
            - features: 特征描述

    Returns:
        填充后的提示词字符串（仅提取代码块内的实际 prompt 内容）
    """
    prompt = template
    prompt = prompt.replace("{{CHARACTER_NAME}}", str(character.get("name", "角色")))
    # appearance 可能是 dict（含 summary）或 str
    appearance = character.get("appearance", "")
    if isinstance(appearance, dict):
        appearance = appearance.get("summary", str(appearance))
    prompt = prompt.replace("{{APPEARANCE}}", str(appearance))
    prompt = prompt.replace("{{CLOTHING}}", str(character.get("clothing", "")))
    prompt = prompt.replace("{{FEATURES}}", str(character.get("features", "")))
    return prompt


def _extract_prompt_from_template(filled_template: str) -> str:
    """从填充后的模板中提取代码块内的实际提示词内容

    模板文件中包含 markdown 标题、使用说明等，实际用于图像生成的 prompt
    位于 ``` 代码块内。此函数提取该部分内容。

    Args:
        filled_template: 填充占位符后的完整模板文本

    Returns:
        仅包含代码块内提示词的字符串
    """
    # Extract content between ``` markers
    lines = filled_template.split('\n')
    in_code_block = False
    prompt_lines = []
    for line in lines:
        if line.strip() == '```':
            if in_code_block:
                # End of code block
                break
            else:
                # Start of code block
                in_code_block = True
                continue
        if in_code_block:
            prompt_lines.append(line)
    
    if prompt_lines:
        return '\n'.join(prompt_lines).strip()
    # Fallback: if no code block found, return the whole thing
    return filled_template.strip()


def build_prompt_from_character(character: dict) -> str:
    """从角色字典构建完整提示词（加载模板 + 填充 + 提取 prompt 块）

    加载 three_view_template.md 模板，填充角色信息，然后提取代码块内的
    实际提示词内容（排除 markdown 标题、使用说明等非 prompt 部分）。

    生成的 prompt 包含所有必需关键词（质感十足、震撼的视觉效果、正面、
    高质量、上方、左侧、底部、右侧、毛发、色值、配色板），可直接通过验证。

    Args:
        character: 角色信息字典，必须包含 name 字段

    Returns:
        填充后提取的提示词字符串（可通过 validate_prompt 验证）
    """
    template = load_template()
    filled = fill_template(template, character)
    return _extract_prompt_from_template(filled)


def validate_and_build_prompt(prompt: str, template_type: str = "three_view") -> Tuple[str, List[str]]:
    """验证提示词并返回结果

    Args:
        prompt: 待验证的提示词
        template_type: 模板类型，默认 "three_view"

    Returns:
        tuple[str, list[str]]: (提示词, 缺失关键词列表)

    Raises:
        ValueError: 如果提示词缺少必需关键词
    """
    is_valid, missing = validate_prompt(prompt, template_type)
    if not is_valid:
        raise ValueError(f"提示词验证失败，缺少必需关键词: {missing}")
    return prompt, missing


# =============================================================================
# ComfyUI 3-View Workflow — Professional Prompt Templates
# Extracted from: workflows/comfyui/3view_character.json
# Original workflow: SDXL + IPAdapter FaceID + ControlNet OpenPose
# Adapted for Seedream API (doubao-seedream-5.0-lite)
# =============================================================================

# --- Negative Prompts (from ComfyUI workflow node #3) ---
# Core negative: "bad anatomy, extra limbs, blurry, low quality"
# Extended with character-sheet-specific exclusions:

NEGATIVE_PROMPT_BASE = (
    "bad anatomy, extra limbs, blurry, low quality, "
    "deformed, disfigured, mutation, extra fingers, extra arms, "
    "missing fingers, fused fingers, poorly drawn hands, "
    "poorly drawn face, asymmetric eyes, cross-eyed, "
    "watermark, text, signature, logo, "
    "cropped, out of frame, duplicate, error"
)

NEGATIVE_PROMPT_CHARACTER = (
    f"{NEGATIVE_PROMPT_BASE}, "
    "multiple characters, crowd, group, "
    "dynamic pose, action pose, sitting, crouching, "
    "background details, scenery, props, furniture, "
    "3D render, cartoon, anime, chibi, "
    "overly saturated, oversharpened, noisy, grainy"
)

# Chinese-friendly negative (for Chinese character descriptions)
NEGATIVE_PROMPT_CN = (
    "卡通, 3D渲染, 过度饱和, 变形, 多余手指, 模糊, 低质量, "
    "多余肢体, 解剖错误, 文字水印, 签名, logo, "
    "多人, 群体, 动态姿势, 复杂背景, "
    "现代服装（古装角色时）, Q版, 可爱化"
)

# --- Three-View Prompt Templates ---
# Based on ComfyUI workflow structure:
# - Canvas: tall portrait (1024x2048 in workflow → 1920x1920+ for Seedream)
# - IPAdapter FaceID weight 0.9 → reference image for identity consistency
# - ControlNet OpenPose strength 0.8 → pose described in text
# - Output: front=crop_top_third, side=crop_middle_third, back=crop_bottom_third

THREE_VIEW_PREFIX = (
    "Character reference sheet, professional character design, "
    "clean white background, full body, centered composition, "
    "consistent proportions across views, "
    "studio lighting, high detail, sharp focus"
)

VIEW_TEMPLATES = {
    "front": (
        "{prefix}, FRONT VIEW, "
        "facing camera directly, symmetrical pose, "
        "arms relaxed at sides, standing straight, "
        "eyes looking at viewer, neutral expression, "
        "showing front details of outfit and features"
    ),
    "side": (
        "{prefix}, SIDE VIEW (left profile), "
        "facing right, perpendicular to camera, "
        "arms relaxed at sides, standing straight, "
        "showing profile silhouette and side details of outfit"
    ),
    "back": (
        "{prefix}, BACK VIEW, "
        "facing away from camera, turned 180 degrees, "
        "arms relaxed at sides, standing straight, "
        "showing back details of outfit, hair, and accessories"
    ),
}

# --- Style Modifiers (applied per art style) ---
STYLE_MODIFIERS = {
    "realistic": ", photorealistic, cinematic lighting, 35mm film, natural skin texture",
    "anime": ", clean linework, cel shading, vibrant colors, anime art style",
    "semi-realistic": ", semi-realistic rendering, soft shading, digital painting",
    "concept_art": ", concept art, detailed rendering, professional illustration",
}

# --- IPAdapter → Seedream Reference Image Strategy ---
# ComfyUI uses IPAdapter FaceID (weight 0.9) for identity consistency.
# In Seedream API, this maps to image_to_image() with the reference image.
# Strategy: Generate front view first (text-to-image), then use it as
# reference for side and back views (image-to-image) to maintain identity.

REFERENCE_WEIGHT_NOTE = (
    "Maintain exact same character identity, face, body proportions, "
    "outfit details, and color palette as the reference image"
)


def build_three_view_prompts(
    character_desc: str,
    style: str = "",
    negative: str = "",
    use_enhanced: bool = True,
) -> dict:
    """Build professional three-view prompts from ComfyUI workflow templates.

    Args:
        character_desc: Character description (e.g. "7岁中国男孩，深色发髻...")
        style: Art style string (e.g. "张艺谋式写实, 35mm film")
        negative: Custom negative prompt (overrides default if provided)
        use_enhanced: Use professional templates (True) or simple prompts (False)

    Returns:
        dict with keys: "front", "side", "back", "negative", "prompts" (list)
    """
    if not use_enhanced:
        # Legacy simple prompts (fallback)
        neg = negative or NEGATIVE_PROMPT_CN
        return {
            "front": f"Character reference sheet, FRONT VIEW, full body standing pose facing camera directly, arms at sides, neutral expression, white background, {character_desc}. {style}",
            "side": f"Character reference sheet, SIDE VIEW (profile), full body standing pose facing right, arms at sides, neutral expression, white background, {character_desc}. {style}",
            "back": f"Character reference sheet, BACK VIEW, full body standing pose facing away from camera, arms at sides, white background, {character_desc}. {style}",
            "negative": neg,
        }

    # Enhanced professional prompts
    neg = negative or NEGATIVE_PROMPT_CHARACTER

    # Build style suffix
    style_suffix = f", {style}" if style else ""

    # Build character block
    character_block = f"{character_desc}{style_suffix}"

    prompts = {}
    for view_name, template in VIEW_TEMPLATES.items():
        prompt = template.format(prefix=THREE_VIEW_PREFIX)
        prompt = f"{prompt}, {character_block}"
        prompts[view_name] = prompt

    prompts["negative"] = neg
    return prompts


def build_reference_prompt(character_desc: str, style: str = "", view: str = "side") -> str:
    """Build a prompt for image-to-image generation with reference.

    Used for side/back views when front view exists as reference.
    Equivalent to IPAdapter FaceID consistency in ComfyUI workflow.

    Args:
        character_desc: Character description
        style: Art style string
        view: Which view to generate ("side" or "back")

    Returns:
        Prompt string optimized for image_to_image with reference
    """
    style_suffix = f", {style}" if style else ""
    template = VIEW_TEMPLATES.get(view, VIEW_TEMPLATES["side"])
    base_prompt = template.format(prefix=THREE_VIEW_PREFIX)

    # Add identity consistency instruction (replaces IPAdapter weight 0.9)
    return f"{base_prompt}, {character_desc}{style_suffix}. {REFERENCE_WEIGHT_NOTE}"


def create_character_card(
    char_id: str,
    name: str,
    description: str,
    style: str = "",
    negative: str = "",
    seedream_model: str = "doubao-seedream-5.0-lite",
    seedance_model: str = "doubao-seedance-2.0-fast",
    reference_images: Optional[dict] = None,
) -> dict:
    """Create a character_card.json structure."""
    return {
        "id": char_id,
        "name": name,
        "description": description,
        "style": style,
        "negative": negative or "卡通, 3D渲染, 过度饱和, 变形, 多余手指",
        "seedream_model": seedream_model,
        "seedance_model": seedance_model,
        "reference_images": reference_images or {
            "front": f"characters/{char_id}/front.png",
            "side": f"characters/{char_id}/side.png",
            "back": f"characters/{char_id}/back.png",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": seedream_model,
    }


def create_angle_map(
    char_id: str,
    default_view: str = "front",
    custom_mappings: Optional[dict] = None,
) -> dict:
    """Create angle_map.json — maps camera angles to best reference images."""
    mappings = custom_mappings or {
        "正面/特写/对话": "front.png",
        "侧面/行走/奔跑": "side.png",
        "背面/远去/离开": "back.png",
        "面部/情绪/泪水": "face_closeup.png",
    }
    return {
        "character": char_id,
        "default_view": default_view,
        "mappings": mappings,
    }


def generate_character(
    char_id: str,
    name: str,
    description: str,
    output_dir: str,
    style: str = "",
    negative: str = "",
    size: str = "1920x1920",
    model: Optional[str] = None,
    skip_images: bool = False,
) -> dict:
    """Generate complete character asset set.

    Args:
        char_id: Unique ID (e.g. "amy", "grandpa", "wolf")
        name: Display name (e.g. "艾米")
        description: Full character description
        output_dir: Base output directory (e.g. "knowledge-base/2026-07-28_01")
        style: Art style string
        negative: Negative prompt
        size: Image dimensions
        model: Override Seedream model
        skip_images: If True, only create JSON files, skip image generation

    Returns:
        dict with paths to all generated assets
    """
    char_dir = os.path.join(output_dir, "characters", char_id)
    os.makedirs(char_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Character Factory: {name} ({char_id})")
    print(f"  Output: {char_dir}")
    print(f"{'='*60}\n")

    # Step 1: Generate three-view images
    if not skip_images:
        print("[Step 1/3] Generating three-view images (enhanced prompts)...")
        client = SeedreamClient(model=model or "doubao-seedream-5.0-lite")

        # Build professional prompts from ComfyUI workflow templates
        prompts = build_three_view_prompts(
            character_desc=description,
            style=style,
            negative=negative,
            use_enhanced=True,
        )
        neg_prompt = prompts["negative"]

        # Validate prompts before calling API
        print("  [validation] Checking prompt completeness...")
        try:
            for view_name in ["front", "side", "back"]:
                validate_and_build_prompt(prompts[view_name], "three_view")
            print("  [validation] ✓ All prompts passed validation")
        except ValueError as e:
            print(f"  [validation] ✗ {e}")
            print("  [validation] Falling back to template-based prompt generation...")
            # Try using the template-based approach
            try:
                character_dict = {
                    "name": name,
                    "appearance": description,
                    "clothing": style if style else "",
                    "features": ""
                }
                template_prompt = build_prompt_from_character(character_dict)
                validate_and_build_prompt(template_prompt, "three_view")
                print("  [validation] ✓ Template-based prompt passed validation")
                # Use template prompt for all views
                prompts["front"] = template_prompt
                prompts["side"] = template_prompt
                prompts["back"] = template_prompt
            except ValueError as e2:
                print(f"  [validation] ✗ Template-based prompt also failed: {e2}")
                print("  [validation] Proceeding with original prompts (may have quality issues)")

        views = {}

        # Generate front view first (text-to-image)
        print("  [three-view] generating front view (base)...")
        try:
            front_path = os.path.join(char_dir, "front.png")
            url = client.text_to_image(
                prompt=prompts["front"],
                output_path=front_path,
                size=size,
            )
            views["front"] = front_path
            print(f"  [three-view] front ✓ → {front_path}")
        except Exception as e:
            print(f"  [three-view] front ✗ → {e}")
            views["front"] = None

        # Generate side view using front as reference (IPAdapter → image_to_image)
        if views["front"] and os.path.exists(views["front"]):
            print("  [three-view] generating side view (with reference)...")
            side_prompt = build_reference_prompt(description, style, view="side")
            try:
                side_path = os.path.join(char_dir, "side.png")
                url = client.image_to_image(
                    prompt=side_prompt,
                    ref_image=views["front"],
                    output_path=side_path,
                    size=size,
                )
                views["side"] = side_path
                print(f"  [three-view] side ✓ → {side_path}")
            except Exception as e:
                print(f"  [three-view] side ✗ → {e}")
                views["side"] = None
        else:
            # Fallback: text-to-image without reference
            print("  [three-view] generating side view (no reference, text-to-image)...")
            try:
                side_path = os.path.join(char_dir, "side.png")
                url = client.text_to_image(
                    prompt=prompts["side"],
                    output_path=side_path,
                    size=size,
                )
                views["side"] = side_path
                print(f"  [three-view] side ✓ → {side_path}")
            except Exception as e:
                print(f"  [three-view] side ✗ → {e}")
                views["side"] = None

        # Generate back view using front as reference
        if views["front"] and os.path.exists(views["front"]):
            print("  [three-view] generating back view (with reference)...")
            back_prompt = build_reference_prompt(description, style, view="back")
            try:
                back_path = os.path.join(char_dir, "back.png")
                url = client.image_to_image(
                    prompt=back_prompt,
                    ref_image=views["front"],
                    output_path=back_path,
                    size=size,
                )
                views["back"] = back_path
                print(f"  [three-view] back ✓ → {back_path}")
            except Exception as e:
                print(f"  [three-view] back ✗ → {e}")
                views["back"] = None
        else:
            # Fallback: text-to-image without reference
            print("  [three-view] generating back view (no reference, text-to-image)...")
            try:
                back_path = os.path.join(char_dir, "back.png")
                url = client.text_to_image(
                    prompt=prompts["back"],
                    output_path=back_path,
                    size=size,
                )
                views["back"] = back_path
                print(f"  [three-view] back ✓ → {back_path}")
            except Exception as e:
                print(f"  [three-view] back ✗ → {e}")
                views["back"] = None
    else:
        print("[Step 1/3] Skipping image generation (--skip-images)")
        views = {"front": None, "side": None, "back": None}

    # Step 2: Create character_card.json
    print("[Step 2/3] Creating character_card.json...")
    card = create_character_card(
        char_id=char_id,
        name=name,
        description=description,
        style=style,
        negative=negative,
        seedream_model=model or "doubao-seedream-5.0-lite",
        reference_images={
            "front": f"characters/{char_id}/front.png",
            "side": f"characters/{char_id}/side.png",
            "back": f"characters/{char_id}/back.png",
        },
    )
    card_path = os.path.join(char_dir, "character_card.json")
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {card_path}")

    # Step 3: Create angle_map.json
    print("[Step 3/3] Creating angle_map.json...")
    angle_map = create_angle_map(char_id=char_id)
    angle_path = os.path.join(char_dir, "angle_map.json")
    with open(angle_path, "w", encoding="utf-8") as f:
        json.dump(angle_map, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {angle_path}")

    # Summary
    result = {
        "char_id": char_id,
        "name": name,
        "char_dir": char_dir,
        "card": card_path,
        "angle_map": angle_path,
        "views": views,
    }
    print(f"\n  ✓ Character '{name}' complete!")
    print(f"  Files: {char_dir}/")
    for f in os.listdir(char_dir):
        fpath = os.path.join(char_dir, f)
        fsize = os.path.getsize(fpath)
        print(f"    {f} ({fsize:,} bytes)")

    return result


def batch_generate(characters: list, output_dir: str, **kwargs) -> list:
    """Generate multiple characters from a list of dicts.

    Each dict must have: id, name, description
    Optional: style, negative, size, model
    """
    results = []
    for char in characters:
        try:
            result = generate_character(
                char_id=char["id"],
                name=char["name"],
                description=char["description"],
                output_dir=output_dir,
                style=char.get("style", ""),
                negative=char.get("negative", ""),
                size=char.get("size", "1920x1920"),
                model=char.get("model"),
                skip_images=kwargs.get("skip_images", False),
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ Failed to generate '{char.get('name', '?')}': {e}")
            results.append({"char_id": char.get("id"), "error": str(e)})
    return results


# --- Predefined characters (佣兵天下) ---

MERCENARY_CHARACTERS = [
    {
        "id": "amy",
        "name": "艾米",
        "description": "7岁中国男孩，深色发髻，坚定黑眸，穿灰色交领汉服，布鞋，身形瘦小但眼神坚毅",
        "style": "张艺谋式写实, 35mm film, 自然光, 古装",
        "negative": "卡通, 3D渲染, 过度饱和, 变形, 多余手指, 现代服装",
    },
    {
        "id": "grandpa",
        "name": "爷爷",
        "description": "65岁中国老者，花白长发束髯，长白胡须，穿灰色麻布长袍，布鞋，面容沧桑但目光慈祥",
        "style": "张艺谋式写实, 35mm film, 自然光, 古装",
        "negative": "卡通, 3D渲染, 过度饱和, 变形, 多余手指, 现代服装",
    },
    {
        "id": "wolf",
        "name": "雪狼",
        "description": "成年白色雪狼，厚密白色皮毛，黄色锐利眼睛，体型壮硕，凶猛但忠诚",
        "style": "张艺谋式写实, 35mm film, 自然光, 野生动物摄影",
        "negative": "卡通, 3D渲染, 过度饱和, 变形, 可爱化",
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Character Asset Factory")
    parser.add_argument("--name", help="Character display name")
    parser.add_argument("--id", help="Character ID (e.g. amy)")
    parser.add_argument("--desc", help="Character description")
    parser.add_argument("--style", default="", help="Art style")
    parser.add_argument("--negative", default="", help="Negative prompt")
    parser.add_argument("--output-dir", default=".", help="Base output directory")
    parser.add_argument("--size", default="1920x1920", help="Image size WxH")
    parser.add_argument("--model", default=None, help="Override Seedream model")
    parser.add_argument("--batch", help="Path to batch JSON file")
    parser.add_argument("--preset", choices=["mercenary"], help="Use preset character list")
    parser.add_argument("--skip-images", action="store_true", help="Only create JSON, skip image gen")

    args = parser.parse_args()

    if args.preset == "mercenary":
        print("Using preset: 佣兵天下 characters")
        results = batch_generate(MERCENARY_CHARACTERS, args.output_dir, skip_images=args.skip_images)
        print(f"\n{'='*60}")
        print(f"  Batch complete: {len(results)} characters")
        for r in results:
            status = "✓" if "error" not in r else "✗"
            print(f"  {status} {r.get('name', r.get('char_id', '?'))}")

    elif args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            characters = json.load(f)
        results = batch_generate(characters, args.output_dir, skip_images=args.skip_images)

    elif args.name and args.id and args.desc:
        generate_character(
            char_id=args.id,
            name=args.name,
            description=args.desc,
            output_dir=args.output_dir,
            style=args.style,
            negative=args.negative,
            size=args.size,
            model=args.model,
            skip_images=args.skip_images,
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python character_factory.py --id amy --name "艾米" --desc "7岁男孩..." --output-dir .')
        print("  python character_factory.py --preset mercenary --output-dir .")
        print("  python character_factory.py --preset mercenary --skip-images --output-dir .")

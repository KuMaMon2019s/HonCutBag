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
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List

# Import seedream client from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clients.seedream_client import SeedreamClient

# Import prompt validator from prompts/ directory
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
sys.path.insert(0, str(_PROMPTS_DIR))
from prompt.prompt_validator import validate_prompt


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
        # If relative path, resolve relative to the pipeline prompt directory.
        p = Path(template_path)
        if not p.is_absolute():
            p = _PROMPTS_DIR / template_path
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
    "【宏观描述】画面风格：真人写实风格，照片级渲染，细节超高清。"
    "根据以下角色描述，生成一张纯白背景的角色三视图设定表，"
    "清晰展示角色的正面、侧面、背面标准正交视图。"
    "要求服装、发型、配饰等所有细节在三个视角中完全一致。"
)

VIEW_TEMPLATES = {
    "front": (
        "{prefix}\n"
        "【微观描述】\n"
        "1. 角色描述：{character_desc}\n"
        "2. 画面要求：纯白背景，无阴影。\n"
        "3. 正面全身站立像：角色正面朝向镜头，面部五官清晰，表情自然，"
        "展示完整正面轮廓、服装正面剪裁、发型正面形态。"
        "质感十足，高质量，震撼的视觉效果。"
    ),
    "side": (
        "{prefix}\n"
        "【微观描述】\n"
        "1. 角色描述：{character_desc}\n"
        "2. 画面要求：纯白背景，无阴影。\n"
        "3. 90度侧面全身像：角色左侧面朝向镜头，展示侧面轮廓、"
        "发型侧面层次、服装侧面剪裁和身体比例。"
        "与正面视图保持完全一致的服装、发型、配饰细节。"
        "质感十足，高质量，震撼的视觉效果。"
    ),
    "back": (
        "{prefix}\n"
        "【微观描述】\n"
        "1. 角色描述：{character_desc}\n"
        "2. 画面要求：纯白背景，无阴影。\n"
        "3. 背面全身像：角色背面朝向镜头，展示背部轮廓、"
        "发型后部层次、服装背面设计（拉链、纽扣、褶皱）。"
        "与正面视图保持完全一致的服装、发型、配饰细节。"
        "质感十足，高质量，震撼的视觉效果。"
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

SOURCE_IMAGE_RULES = (
    "avoid strong shadows, front-facing or three-quarter view, "
    "solid-color or simple fabric background, face occupies at least 60 percent"
)

FULL_BODY_IMAGE_RULES = (
    "vertical 9:16 character reference, camera pulled far back, straight-on standing pose, "
    "entire body visible from the top of the hair to the soles of both shoes, both feet fully "
    "inside the frame, generous empty margin above the hair and below the shoes, character "
    "occupies no more than 75 percent of canvas height, plain neutral background, no scenery, "
    "no props, no crop, no close-up, no medium shot, no knees or feet outside frame"
)

FULL_BODY_REFERENCE_SIZE = "1440x2560"


def _reference_rendering_clause(style: str) -> str:
    """Keep character references in the project's declared visual medium.

    Character Factory historically forced every reference to be photorealistic,
    even when ``style`` explicitly requested animation or illustration.  That
    creates an identity/style mismatch which then propagates into every video
    shot.  Preserve the legacy default only when no non-photographic medium is
    declared.
    """
    normalized = str(style or "").lower()
    non_photographic_markers = (
        "二维", "动画", "动漫", "赛璐璐", "厚涂", "插画", "手绘", "漫画",
        "2d", "animation", "anime", "cel shading", "illustration", "painted",
    )
    if any(marker in normalized for marker in non_photographic_markers):
        return (
            "2D cinematic character illustration in the exact declared project style, "
            "painted cel shading, deliberately designed fictional facial features, "
            "consistent linework and color blocks, no photoreal skin, no live-action person"
        )
    return "Photorealistic, natural skin texture"


def build_model_reference_prompts(
    character_desc: str, style: str = "", target_model: str = "seedance"
) -> dict:
    """Build separated reference prompts for Seedance or Kling."""
    suffix = f", {style}" if style else ""
    fictional_decl = (
        "This is a fully fictional AI-generated character (virtual avatar), "
        "not a real person; the face is a synthetic digital creation"
    )
    rendering = _reference_rendering_clause(style)
    identity = f"{fictional_decl}. {character_desc}{suffix}. {rendering}, neutral expression"
    if "kling" in target_model.lower():
        return {
            "front": f"{identity}, {SOURCE_IMAGE_RULES}, front portrait, identity reference",
            "side": f"{identity}, {SOURCE_IMAGE_RULES}, strict side profile, identity reference",
            "three_quarter": f"{identity}, {SOURCE_IMAGE_RULES}, three-quarter portrait, identity reference",
            "detail": f"{identity}, {SOURCE_IMAGE_RULES}, facial detail close-up, face occupies 70 percent",
        }
    return {
        "face_closeup": f"{identity}, {SOURCE_IMAGE_RULES}, head-and-shoulders close-up, face occupies 70 percent",
        "full_body": (
            f"{identity}, {FULL_BODY_IMAGE_RULES}, separate full-body standing reference, complete outfit and footwear visible, "
            "same identity and clothing as the face reference"
        ),
    }


def build_combined_sheet_prompt(
    character_desc: str,
    style: str = "",
) -> str:
    """Build a single prompt for a combined character sheet (all views in one image).
    
    Generates ONE image with all 4 views side-by-side (closeup, front, side, back).
    This ensures character consistency across all views since they're generated together.
    
    Args:
        character_desc: Character description (e.g. "7岁中国男孩，深色发髻...")
        style: Art style string (e.g. "张艺谋式写实, 35mm film")
    
    Returns:
        Single prompt string for combined character sheet generation
    """
    style_suffix = f"，{style}" if style else ""
    full_desc = f"{character_desc}{style_suffix}"
    
    rendering = _reference_rendering_clause(style)
    prompt = (
        "【宏观描述】所有角色均为 AI 生成的虚拟形象，非真实人物。"
        f"画面媒介：{rendering}。"
        "根据以下角色描述，生成一张纯白背景的角色四视图设定表。"
        "要求服装、发型、配饰等所有细节在四个视角中完全一致。\n"
        "【微观描述】\n"
        f"1. 角色描述：{full_desc}\n"
        "2. 画面要求：纯净中性灰背景 #E8E8E8，无阴影，均匀柔光。\n"
        "3. 同一画面按2×2网格展示四个视图：\n"
        "   - 左上：人像特写（头顶至锁骨，面部占60%+，五官清晰）\n"
        "   - 右上：正面全身站立像（面对镜头，从头顶到脚底完整）\n"
        "   - 左下：90度侧面全身站立像（纯侧面轮廓，从头顶到脚底完整）\n"
        "   - 右下：背面全身站立像（后脑/背部/发尾清晰，从头顶到脚底完整）\n"
        "4. 自然站立，双臂自然下垂，双脚平行微分。\n"
        "5. 四视图身份、设计语言、线条和色块完全一致；面部与发型细节清晰，但不得改变既定画面媒介。\n"
        "6. 画面比例 1:1 正方形，2×2网格布局。图中不要有任何文字。\n"
        "质感十足，高质量，震撼的视觉效果。"
    )
    return prompt


def crop_character_sheet(
    sheet_path: str,
    output_dir: str,
    num_views: int = 4,
) -> dict:
    """Crop a combined character sheet (2×2 grid) into individual view images.
    
    Layout: 左上=closeup, 右上=front, 左下=side, 右下=back
    
    Args:
        sheet_path: Path to the combined character sheet image (square, e.g. 1920x1920)
        output_dir: Directory to save cropped view images
        num_views: Number of views to crop (default 4)
    
    Returns:
        dict mapping view names to file paths
    """
    # 2×2 grid positions: (col, row) — col 0=left, 1=right; row 0=top, 1=bottom
    view_layout = [
        ("closeup", 0, 0),  # 左上
        ("front",   1, 0),  # 右上
        ("side",    0, 1),  # 左下
        ("back",    1, 1),  # 右下
    ][:num_views]
    views = {}
    
    for view_name, col, row in view_layout:
        view_path = os.path.join(output_dir, f"{view_name}.png")
        # Crop each quadrant: width=iw/2, height=ih/2, x=col*iw/2, y=row*ih/2
        crop_cmd = [
            "ffmpeg", "-y", "-i", sheet_path,
            "-vf", f"crop=iw/2:ih/2:{col}*iw/2:{row}*ih/2",
            view_path
        ]
        try:
            subprocess.run(crop_cmd, capture_output=True, check=True)
            views[view_name] = view_path
            print(f"  [crop] {view_name} ✓ → {view_path}")
        except subprocess.CalledProcessError as e:
            print(f"  [crop] {view_name} ✗ → {e}")
            views[view_name] = None
    
    return views


def build_three_view_prompts(
    character_desc: str,
    style: str = "",
    negative: str = "",
    use_enhanced: bool = True,
) -> dict:
    """Build professional three-view prompts from ComfyUI workflow templates.
    
    DEPRECATED: This function is kept for backward compatibility but is no longer used.
    The new approach uses build_combined_sheet_prompt() to generate all views in one image.

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
    style_suffix = f"，{style}" if style else ""
    full_desc = f"{character_desc}{style_suffix}"

    prompts = {}
    for view_name, template in VIEW_TEMPLATES.items():
        prompt = template.format(prefix=THREE_VIEW_PREFIX, character_desc=full_desc)
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
    style_suffix = f"，{style}" if style else ""
    full_desc = f"{character_desc}{style_suffix}"
    template = VIEW_TEMPLATES.get(view, VIEW_TEMPLATES["side"])
    return template.format(prefix=THREE_VIEW_PREFIX, character_desc=full_desc)


def create_character_card(
    char_id: str,
    name: str,
    description: str,
    style: str = "",
    negative: str = "",
    seedream_model: str = "doubao-seedream-5.0-lite",
    seedance_model: str = "doubao-seedance-2.0-mini",
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
            "face_closeup": f"characters/{char_id}/face_closeup.png",
            "full_body": f"characters/{char_id}/full_body.png",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": seedream_model,
    }


def create_angle_map(
    char_id: str,
    default_view: str = "face_closeup",
    custom_mappings: Optional[dict] = None,
) -> dict:
    """Create angle_map.json — maps camera angles to best reference images."""
    mappings = custom_mappings or {
        "正面/特写/对话": "face_closeup.png",
        "侧面/行走/奔跑": "full_body.png",
        "背面/远去/离开": "full_body.png",
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
    variants: Optional[list] = None,
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

    # Initialize client (may be None if skip_images=True)
    client = None

    # Step 1: Generate separated references. The legacy combined sheet remains
    # available as a graceful fallback for providers that reject this route.
    target_model = "kling" if model and "kling" in model.lower() else "seedance"
    seedream_model = (
        model if model and "seedream" in model.lower()
        else "doubao-seedream-5.0-lite"
    )
    if not skip_images:
        print(f"[Step 1/3] Generating separated {target_model} references...")
        client = SeedreamClient(model=seedream_model)
        reference_prompts = build_model_reference_prompts(description, style, target_model)
        views = {}
        try:
            first_path = None
            for index, (view_name, reference_prompt) in enumerate(reference_prompts.items()):
                view_path = os.path.join(char_dir, f"{view_name}.png")
                view_size = FULL_BODY_REFERENCE_SIZE if view_name == "full_body" else size
                if index and first_path and hasattr(client, "image_to_image"):
                    client.image_to_image(
                        prompt=f"{REFERENCE_WEIGHT_NOTE}. {reference_prompt}",
                        ref_image=first_path,
                        output_path=view_path,
                        size=view_size,
                    )
                else:
                    client.text_to_image(prompt=reference_prompt, output_path=view_path, size=view_size)
                first_path = first_path or view_path
                views[view_name] = view_path
                print(f"  [{view_name}] ✓ → {view_path}")
        except Exception as exc:
            print(f"  ⚠ separated references failed ({exc}); using legacy combined sheet fallback")
            sheet_path = os.path.join(char_dir, "character_sheet.png")
            client.text_to_image(
                prompt=build_combined_sheet_prompt(description, style),
                output_path=sheet_path,
                size="1920x1920",
            )
            legacy_views = crop_character_sheet(sheet_path, char_dir, num_views=4)
            views = legacy_views
    else:
        print("[Step 1/3] Skipping image generation (--skip-images)")
        views = {name: None for name in build_model_reference_prompts(description, style, target_model)}

    # Step 2: Create character_card.json
    print("[Step 2/3] Creating character_card.json...")
    card = create_character_card(
        char_id=char_id,
        name=name,
        description=description,
        style=style,
        negative=negative,
        seedream_model=seedream_model,
        reference_images={name: f"characters/{char_id}/{name}.png" for name in views},
    )
    card["face_reference"] = f"characters/{char_id}/face_closeup.png"
    card["body_reference"] = f"characters/{char_id}/full_body.png"
    card["reference_strategy"] = "kling_four_views" if "kling" in target_model.lower() else "seedance_face_and_body"
    card["source_image_rules"] = SOURCE_IMAGE_RULES
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

    # Step 4: Generate variant images (P1-A3: 衍生参考图)
    variant_paths = {}
    if variants and not skip_images and client is not None:
        print(f"\n[Step 4] Generating {len(variants)} variant reference(s)...")
        for variant in variants:
            state_name = variant.get("state_name", "unknown")
            variant_desc = variant.get("description", "")
            if not variant_desc:
                print(f"  [variant] Skipping {state_name}: no description")
                continue
            
            variant_filename = f"variant_{state_name}.png"
            variant_path = os.path.join(char_dir, variant_filename)
            
            # Skip if already exists
            if os.path.exists(variant_path):
                print(f"  [variant] {variant_filename} already exists, skipping")
                variant_paths[state_name] = variant_path
                continue
            
            try:
                # Build variant prompt: base appearance + state change
                # Emphasize face must remain identical
                rendering = _reference_rendering_clause(style)
                variant_prompt = (
                    f"Character reference sheet, same person as base reference. "
                    f"State change: {variant_desc}. "
                    f"CRITICAL: facial features, bone structure, and identity must remain "
                    f"100% identical to the base character. Only modify clothing, hair condition, "
                    f"or add props as described in the state change. "
                    f"{rendering}, front view, full body, white background, consistent lighting."
                )
                
                print(f"  [variant] Generating {variant_filename}...")
                url = client.text_to_image(
                    prompt=variant_prompt,
                    output_path=variant_path,
                    size=size,
                )
                print(f"  [variant] ✓ {variant_filename}")
                variant_paths[state_name] = variant_path
                
            except Exception as e:
                print(f"  [variant] ✗ Failed to generate {variant_filename}: {e}")
                # Continue with other variants (graceful degradation)
                continue
    elif skip_images:
        print("\n[Step 4] Skipping variant generation (--skip-images)")
    else:
        print("\n[Step 4] No variants to generate")

    # Summary
    result = {
        "char_id": char_id,
        "name": name,
        "char_dir": char_dir,
        "card": card_path,
        "angle_map": angle_path,
        "views": views,
        "variants": variant_paths,
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
    Optional: style, negative, size, model, variants
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
                variants=char.get("appearance", {}).get("variants", []),
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ Failed to generate '{char.get('name', '?')}': {e}")
            results.append({"char_id": char.get("id"), "error": str(e)})
    return results


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
    parser.add_argument("--skip-images", action="store_true", help="Only create JSON, skip image gen")

    args = parser.parse_args()

    if args.batch:
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
        print("  python character_factory.py --batch characters.json --output-dir .")
        print("  python character_factory.py --batch characters.json --skip-images --output-dir .")

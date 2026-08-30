"""Character Asset Factory — batch generate character assets for the pipeline.

Generates:
  1. character_card.json — character metadata + generation params
  2. Four-view identity images (face/front/side/back) via Seedream
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
import hashlib
import argparse
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, List

# Import seedream client from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC_DIR))
from clients.seedream_client import SeedreamClient
from tools.character_reference_board import ensure_character_reference_board

# Import prompt validator from prompts/ directory
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
sys.path.insert(0, str(_PROMPTS_DIR))
from prompt.prompt_validator import validate_prompt
from prompt.seedream_image_prompt import bind_reference_roles
from quality.character_reference_qa import (
    CHARACTER_REFERENCE_QA_SCHEMA,
    CharacterReferenceQAError,
    build_character_reference_qa_receipt,
    file_sha256,
    review_character_reference_pack,
    review_identity_detail_reference,
    validate_character_reference_qa_receipt,
)
from utils.character_reference_contracts import (
    IDENTITY_DETAIL_ASSET_POLICY,
    STATIC_REFERENCE_ASSET_POLICY,
    identity_detail_prompt_items,
    normalize_identity_props,
)
from utils.character_body_contracts import character_visual_description
from utils.camera_motion_contracts import (
    HUMAN_PERSPECTIVE_NEGATIVE,
)


class CharacterReferenceGenerationPaused(RuntimeError):
    """A durable acceptance limit stopped after one new identity image."""

    def __init__(
        self,
        *,
        character_id: str,
        view_name: str,
        view_path: Path,
        character_description: str,
        synthetic_styling: dict[str, Any] | None,
    ) -> None:
        self.character_id = character_id
        self.view_name = view_name
        self.view_path = view_path
        self.character_description = character_description
        self.synthetic_styling = synthetic_styling
        super().__init__(
            "character reference generation paused after one durable image: "
            f"{character_id}/{view_name}"
        )


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
    prompt = prompt.replace("{{APPEARANCE}}", character_visual_description(character))
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
# Strategy: Generate a neutral face close-up first (text-to-image), then use
# only that pose-free identity anchor for each body view (image-to-image).

REFERENCE_WEIGHT_NOTE = (
    "Use the supplied image only for the fictional identity: face, hair, proportions, "
    "outfit and colors. Ignore its pose, camera, framing, background, text, props and "
    "people; the requested view wins"
)

SOURCE_IMAGE_RULES = (
    "one character; strict straight-on head-and-shoulders view from crown through "
    "clavicles; face fills 60–75%; centered neutral expression; plain gray studio; even "
    "light; no full body, action, scene, crowd, text, logo or hand-operated object; "
    "50–85mm natural perspective; no wide-angle distortion"
)

FULL_BODY_IMAGE_RULES = (
    "vertical 9:16 full-body identity reference; hair top and both shoe soles visible with "
    "clear margins; subject uses at most 75% of canvas height; plain gray studio; upright "
    "neutral stance; arms down, hands open and empty, feet parallel and hip-width, balanced "
    "weight; no held item, action, scene, crowd, extra person, text, logo or crop; 50–85mm "
    "natural perspective; no wide-angle distortion"
)

FULL_BODY_REFERENCE_SIZE = "2K"
REFERENCE_CONTRACT_VERSION = 7
REFERENCE_GENERATION_CONTRACT_SCHEMA = "honcut.character-reference-generation.v1"
PROP_DETAIL_QA_SCHEMA = "honcut.prop-detail-board-qa.v1"


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_reference_generation_contract(
    *,
    prompts: dict[str, str],
    model: str,
    synthetic_styling: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the exact cache input for canonical character-reference pixels."""
    return {
        "schema": REFERENCE_GENERATION_CONTRACT_SCHEMA,
        "reference_contract_version": REFERENCE_CONTRACT_VERSION,
        "model": model,
        "prompt_sha256": {
            name: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for name, prompt in prompts.items()
        },
        "synthetic_styling_sha256": (
            _canonical_json_sha256(synthetic_styling)
            if synthetic_styling
            else None
        ),
    }


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
        "三维动画", "风格化三维", "合成人", "机器人", "机械面甲",
        "2d", "3d cgi", "stylized 3d", "cgi animation", "animation", "anime",
        "cel shading", "illustration", "painted", "synthetic", "android", "robot",
    )
    if any(marker in normalized for marker in non_photographic_markers):
        return (
            "High-end stylized CGI character design in the exact declared project medium, "
            "deliberately synthetic materials and designed digital geometry, consistent "
            "silhouette, declared veil/makeup/tattoo/mechanical texture anchors and color blocks, "
            "no photoreal skin or untreated natural human face, "
            "no live-action person and no likeness of a real person"
        )
    return "Photorealistic, natural skin texture"


def build_model_reference_prompts(
    character_desc: str,
    style: str = "",
    target_model: str = "seedance",
    synthetic_styling: dict[str, Any] | None = None,
) -> dict:
    """Build separated reference prompts for Seedance or Kling."""
    aesthetic_contract = ""
    if synthetic_styling:
        from utils.privacy_visual_policy import (
            is_current_synthetic_styling,
            synthetic_makeup_reference_prompt_contract,
        )

        if not is_current_synthetic_styling(synthetic_styling):
            raise ValueError(
                "synthetic character reference generation requires the current aesthetic profile"
            )
        aesthetic_contract = synthetic_makeup_reference_prompt_contract(
            synthetic_styling
        )
    fictional_decl = (
        "This is a fully fictional AI-generated character (virtual avatar), "
        "not a real person; the entire visual identity is a designed digital creation "
        "with no real-person likeness"
    )
    rendering = _reference_rendering_clause(style)
    asset_boundary = (
        ""
        if STATIC_REFERENCE_ASSET_POLICY in character_desc
        else f"Asset boundary: {STATIC_REFERENCE_ASSET_POLICY} "
    )
    identity = (
        f"{aesthetic_contract}\n{fictional_decl}. "
        f"Static identity facts only: {character_desc}. {asset_boundary}"
        f"Rendering medium only: {rendering}. Neutral expression. Use only explicitly "
        "declared identity facts; never invent a collar, opening, seam, ornament, prop, "
        "location, crowd, camera movement, pose or action. "
        f"Avoid: {HUMAN_PERSPECTIVE_NEGATIVE}"
    )
    if "kling" in target_model.lower():
        return {
            "front": f"{identity}, {SOURCE_IMAGE_RULES}, front portrait, identity reference",
            "side": f"{identity}, {SOURCE_IMAGE_RULES}, strict side profile, identity reference",
            "three_quarter": f"{identity}, {SOURCE_IMAGE_RULES}, three-quarter portrait, identity reference",
            "detail": f"{identity}, {SOURCE_IMAGE_RULES}, facial detail close-up, face occupies 70 percent",
        }
    return {
        "face_closeup": (
            f"{identity}. VIEW CONTRACT — FACE CLOSE-UP: {SOURCE_IMAGE_RULES}. "
            "This must be a true identity mugshot-style close-up, never a scene still"
        ),
        "full_body": (
            f"{identity}. VIEW CONTRACT — FRONT FULL BODY: {FULL_BODY_IMAGE_RULES}. "
            "Face, chest, knees and toes face camera; show the complete matching outfit and footwear"
        ),
        "side": (
            f"{identity}. VIEW CONTRACT — STRICT 90-DEGREE LEFT SIDE: {FULL_BODY_IMAGE_RULES}. "
            "Nose, torso, hips, knees and toes point left; one eye visible; no head turn; show "
            "the complete matching identity, proportions, outfit and footwear"
        ),
        "back": (
            f"{identity}. VIEW CONTRACT — STRICT 180-DEGREE BACK: {FULL_BODY_IMAGE_RULES}. "
            "Show rear hair, shoulders, spine, outfit, legs and heels; face and front torso "
            "fully hidden; no head turn; match identity, proportions and outfit"
        ),
    }


def build_identity_detail_prompt(
    character_desc: str,
    identity_props: list[dict[str, Any]],
    style: str = "",
    correction: str = "",
) -> str:
    """Build a four-view-derived detail board without polluting neutral poses."""
    rendering = _reference_rendering_clause(style)
    correction_clause = (
        f"CORRECTION — the previous detail board failed: {correction}. "
        if correction
        else ""
    )
    return (
        f"{correction_clause}Create one professional identity-detail reference board for the exact "
        "same fictional character shown in the supplied approved face and full-body references. "
        f"Static identity: {character_desc}. Rendering medium: {rendering}. "
        f"Declared identity items: {identity_detail_prompt_items(identity_props)}. "
        f"Policy: {IDENTITY_DETAIL_ASSET_POLICY} "
        "Use a clean 2x2 detail-board layout on neutral gray #E8E8E8 with even studio light. "
        "For body_attached items, show a close crop on the same attachment point plus one isolated "
        "material/color detail. For isolated_handheld items, show the item alone from front, side, "
        "and three-quarter angles at a stable scale; no hand touches it and the character does not "
        "operate it. Preserve exact authored primary/secondary colors, material finish, geometry, "
        "markings, straps and left/right orientation. Do not add a scene, action pose, another person, "
        "an undeclared object, captions, labels, watermark, border text or logo."
    )


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
    rendering = _reference_rendering_clause(style)
    prompt = (
        "【宏观描述】所有角色均为 AI 生成的虚拟形象，非真实人物。"
        f"画面媒介：{rendering}。"
        "根据以下角色描述，生成一张纯白背景的角色四视图设定表。"
        "要求服装、发型、配饰等所有细节在四个视角中完全一致。\n"
        "【微观描述】\n"
        f"1. 静态角色身份描述：{character_desc}\n"
        "2. 画面要求：纯净中性灰背景 #E8E8E8，无阴影，均匀柔光。\n"
        "3. 同一画面按2×2网格展示四个视图：\n"
        "   - 左上：人像特写（头顶至锁骨，面部占60%+，五官清晰）\n"
        "   - 右上：正面全身站立像（面对镜头，从头顶到脚底完整）\n"
        "   - 左下：90度侧面全身站立像（纯侧面轮廓，从头顶到脚底完整）\n"
        "   - 右下：背面全身站立像（后脑/背部/发尾清晰，从头顶到脚底完整）\n"
        "4. 自然站立，双臂自然下垂，双脚平行微分；禁止舞蹈、表演和动作姿势。\n"
        f"   静态资产边界：{STATIC_REFERENCE_ASSET_POLICY}\n"
        "5. 四视图身份、设计语言、线条和色块完全一致；面部与发型细节清晰，但不得改变既定画面媒介。\n"
        "6. 画面比例 1:1 正方形，2×2网格布局。图中不要有任何文字、街景、店铺、"
        "人群、道具或项目剧情元素；项目风格只决定媒介，不得带入场景与动作。\n"
        "质感十足，高质量，震撼的视觉效果。"
    )
    return prompt


def crop_character_sheet(
    sheet_path: str,
    output_dir: str,
    num_views: int = 4,
) -> dict:
    """Crop a combined character sheet (2×2 grid) into individual view images.
    
    Layout: 左上=face_closeup, 右上=full_body, 左下=side, 右下=back
    
    Args:
        sheet_path: Path to the combined character sheet image (square, e.g. 1920x1920)
        output_dir: Directory to save cropped view images
        num_views: Number of views to crop (default 4)
    
    Returns:
        dict mapping view names to file paths
    """
    # 2×2 grid positions: (col, row) — col 0=left, 1=right; row 0=top, 1=bottom
    view_layout = [
        ("face_closeup", 0, 0),  # 左上
        ("full_body", 1, 0),  # 右上
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
    
    DEPRECATED: kept for backward compatibility. Production generates four
    separated references and uses the combined sheet only as a provider-error fallback.

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
            "face_closeup": f"characters/{char_id}/face_closeup.png",
            "full_body": f"characters/{char_id}/full_body.png",
            "side": f"characters/{char_id}/side.png",
            "back": f"characters/{char_id}/back.png",
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
        "正面/全身/站立": "full_body.png",
        "侧面/行走/奔跑": "side.png",
        "背面/远去/离开": "back.png",
        "面部/情绪/泪水": "face_closeup.png",
    }
    return {
        "character": char_id,
        "default_view": default_view,
        "mappings": mappings,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reference_view_size(view_name: str, default_size: str) -> str:
    return (
        FULL_BODY_REFERENCE_SIZE
        if view_name in {"full_body", "side", "back"}
        else default_size
    )


def _generate_reference_view(
    client: Any,
    *,
    view_name: str,
    prompt: str,
    output_path: Path,
    size: str,
    identity_anchor: Path | None,
    correction: str = "",
    before_provider_request: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Generate one view atomically without letting an anchor dictate its pose."""
    final_prompt = prompt
    if correction:
        final_prompt = (
            f"CORRECTION — the previous {view_name} failed blocking view QA: "
            f"{correction}. Regenerate from scratch and obey the view contract exactly. "
            f"{prompt}"
        )
    temporary = output_path.with_name(f".{output_path.stem}.generating{output_path.suffix}")
    try:
        if identity_anchor is not None and hasattr(client, "image_to_image"):
            provider_prompt = bind_reference_roles(
                f"{REFERENCE_WEIGHT_NOTE}. {final_prompt}",
                ["character_identity_only"],
            )
            if before_provider_request is not None:
                before_provider_request({
                    "provider_family": "seedream_image",
                    "request_mode": "image_to_image",
                    "view_name": view_name,
                    "model": str(getattr(client, "model", "unknown-image-model")),
                    "size": size,
                    "prompt_sha256": hashlib.sha256(
                        provider_prompt.encode("utf-8")
                    ).hexdigest(),
                    "reference_sha256": file_sha256(identity_anchor),
                })
            client.image_to_image(
                prompt=provider_prompt,
                ref_image=str(identity_anchor),
                output_path=str(temporary),
                size=size,
            )
        else:
            if before_provider_request is not None:
                before_provider_request({
                    "provider_family": "seedream_image",
                    "request_mode": "text_to_image",
                    "view_name": view_name,
                    "model": str(getattr(client, "model", "unknown-image-model")),
                    "size": size,
                    "prompt_sha256": hashlib.sha256(
                        final_prompt.encode("utf-8")
                    ).hexdigest(),
                    "reference_sha256": None,
                })
            client.text_to_image(
                prompt=final_prompt,
                output_path=str(temporary),
                size=size,
            )
        if not temporary.is_file():
            raise RuntimeError(f"reference generator did not write {view_name}")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _generate_identity_detail(
    client: Any,
    *,
    character_description: str,
    identity_props: list[dict[str, Any]],
    style: str,
    canonical_paths: list[Path],
    output_path: Path,
    correction: str = "",
) -> None:
    """Generate one detail board from approved canonical identity references."""
    temporary = output_path.with_name(
        f".{output_path.stem}.generating{output_path.suffix}"
    )
    try:
        client.image_to_image(
            prompt=bind_reference_roles(
                build_identity_detail_prompt(
                    character_description,
                    identity_props,
                    style,
                    correction,
                ),
                ["character_face_identity_only", "character_body_identity_only"],
            ),
            ref_image=[str(path) for path in canonical_paths],
            output_path=str(temporary),
            size="2K",
        )
        if not temporary.is_file():
            raise RuntimeError("identity-detail generator did not write its output")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _quality_control_identity_detail(
    *,
    char_id: str,
    character_description: str,
    identity_props: list[dict[str, Any]],
    style: str,
    char_dir: Path,
    canonical_paths: list[Path],
    detail_path: Path,
    image_client: Any,
    review_client: Any,
    max_retries: int,
) -> dict[str, Any]:
    """Block Phase 3 until the supplemental item board matches the four views."""
    report_path = char_dir / "prop_detail_board_qa.json"
    input_contract = {
        "schema": "honcut.prop-detail-board-input.v1",
        "character_description_sha256": hashlib.sha256(
            character_description.encode("utf-8")
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            build_identity_detail_prompt(
                character_description,
                identity_props,
                style,
            ).encode("utf-8")
        ).hexdigest(),
        "identity_props_sha256": _canonical_json_sha256(identity_props),
        "canonical_references": [
            {"path": path.name, "sha256": file_sha256(path)}
            for path in canonical_paths
        ],
    }
    try:
        cached = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None
    if (
        isinstance(cached, dict)
        and cached.get("schema") == PROP_DETAIL_QA_SCHEMA
        and cached.get("status") == "passed"
        and cached.get("input_contract") == input_contract
        and detail_path.is_file()
        and cached.get("inputs", {}).get("prop_detail_board", {}).get("sha256")
        == file_sha256(detail_path)
    ):
        print(f"  [prop-detail-qa] {char_id} ✓ exact cache, zero Provider requests")
        return cached
    attempts: list[dict[str, Any]] = []
    _generate_identity_detail(
        image_client,
        character_description=character_description,
        identity_props=identity_props,
        style=style,
        canonical_paths=canonical_paths,
        output_path=detail_path,
    )
    for attempt in range(1, max_retries + 2):
        result = review_identity_detail_reference(
            review_client,
            canonical_paths,
            detail_path,
            identity_props,
        )
        attempts.append({"attempt": attempt, **result})
        receipt = {
            "schema": PROP_DETAIL_QA_SCHEMA,
            "character_id": char_id,
            "status": "passed" if result["passed"] else "failed",
            "identity_props": identity_props,
            "input_contract": input_contract,
            "inputs": {
                "canonical_references": input_contract["canonical_references"],
                "prop_detail_board": {
                    "path": detail_path.name,
                    "sha256": file_sha256(detail_path),
                },
            },
            "attempts": attempts,
        }
        _write_json_atomic(report_path, receipt)
        if result["passed"]:
            print(f"  [prop-detail-qa] {char_id} ✓ attempt {attempt}")
            return receipt
        if attempt > max_retries:
            raise CharacterReferenceQAError(
                f"{char_id} identity detail failed after {attempt} QA attempt(s): "
                + "; ".join(result.get("issues") or ["detail contract violation"])
            )
        archive = char_dir / "prop_detail_board_qa_attempts" / f"attempt_{attempt:02d}"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(detail_path, archive / detail_path.name)
        _generate_identity_detail(
            image_client,
            character_description=character_description,
            identity_props=identity_props,
            style=style,
            canonical_paths=canonical_paths,
            output_path=detail_path,
            correction="; ".join(result.get("issues") or ["item or identity mismatch"]),
        )
    raise AssertionError("unreachable prop-detail QA state")


def _archive_reference_attempt(
    char_dir: Path,
    view_paths: dict[str, Path],
    attempt: int,
) -> None:
    archive = char_dir / "reference_qa_attempts" / f"attempt_{attempt:02d}"
    archive.mkdir(parents=True, exist_ok=True)
    for name, path in view_paths.items():
        if path.is_file():
            shutil.copy2(path, archive / f"{name}.png")


def _view_correction(
    review: dict[str, Any],
    view_name: str,
    *,
    require_synthetic: bool,
) -> str:
    """Render only evidence-backed correction directives.

    Reviewer prose is diagnostic and may invent details that are absent from
    the authored contract.  Mapping failed evidence fields to fixed directives
    prevents those hallucinations from being promoted into the next paid image
    prompt.
    """
    view = review.get("views", {}).get(view_name, {})
    directives: list[str] = []
    field_directives = {
        "view_match": "obey the exact requested camera angle",
        "framing_match": "obey the exact requested framing and show every required body edge",
        "neutral_pose": "use the neutral anatomical identity stance, not an action pose",
        "plain_background": "use one plain neutral gray studio background",
        "single_character": "show exactly one character with no duplicate or extra person",
        "declared_identity_match": "match the declared face, hair, proportions, colors and identity marks",
        "declared_outfit_match": "match only the explicitly declared outfit and footwear",
    }
    for field, directive in field_directives.items():
        if view.get(field) is not True:
            directives.append(directive)
    if view_name in {"full_body", "side", "back"} and view.get("hands_empty") is not True:
        directives.append("keep both hands open and empty with no held or operated item")
    if view_name == "side" and view.get("both_eyes_visible") is not False:
        directives.append("show a strict left profile with exactly one eye visible")
    if view_name == "back" and (
        view.get("face_visible") is not False
        or view.get("both_eyes_visible") is not False
    ):
        directives.append("show a strict rear view with the entire face invisible")
    if require_synthetic and view_name != "back":
        synthetic_directives = {
            "synthetic_makeup_visible": "make the declared synthetic porcelain makeup clearly visible",
            "synthetic_material_anchor_match": "show the exact declared warm pearl-ceramic face material",
            "circuit_makeup_anchor_match": "place the exact fine circuit stripes from both temples to upper cheekbones, away from eyelids",
            "iris_ring_anchor_match": "put a thin luminous ring inside each iris around its dark pupil, separate from eyeliner",
            "face_unobscured": "keep the complete face unobscured",
            "makeup_clean_and_harmonious": "keep the makeup clean, harmonious and cosmetic",
            "no_grotesque_damage": "remove cracks, scars, seams and grotesque mechanical damage",
            "healthy_warm_complexion": "restore a warm healthy complexion with living cheek color",
            "lively_eyes_with_catchlights": "show clear pupils, layered irises and bright catchlights",
            "living_color_in_cheeks_and_lips": "restore coordinated warm cheek and coral lip color",
            "no_uncanny_or_corpse_like_styling": "remove corpse-like, waxy or horror styling",
        }
        for field, directive in synthetic_directives.items():
            if view.get(field) is not True:
                directives.append(directive)
    cross = review.get("cross_view", {})
    cross_directives = {
        "identity_consistent": "match the same identity across all four views",
        "outfit_consistent": "match the same declared outfit across all four views",
        "body_proportions_consistent": "match the same body proportions across all four views",
    }
    if require_synthetic:
        cross_directives["synthetic_makeup_consistent"] = (
            "match the same synthetic makeup geometry and colors across all face-visible views"
        )
    for field, directive in cross_directives.items():
        if cross.get(field) is not True:
            directives.append(directive)
    if not directives:
        directives.append(
            "wrong view angle, framing, neutral stance, studio background, or cross-view identity"
        )
    return "; ".join(dict.fromkeys(directives))


def _quality_control_reference_views(
    *,
    char_id: str,
    character_description: str,
    char_dir: Path,
    prompts: dict[str, str],
    view_paths: dict[str, Path],
    image_client: Any,
    review_client: Any,
    default_size: str,
    max_retries: int,
    review_max_retries: int,
    synthetic_styling: dict[str, Any] | None,
    generation_contract: dict[str, Any],
) -> dict[str, Any]:
    """Review all views together and regenerate only the implicated views."""
    if max_retries < 0 or max_retries > 2:
        raise ValueError("character reference QA retries must be between 0 and 2")
    if review_max_retries < 0 or review_max_retries > 2:
        raise ValueError("character reference review retries must be between 0 and 2")
    report_path = char_dir / "character_reference_qa.json"
    attempts: list[dict[str, Any]] = []
    ordered_names = tuple(view_paths)
    anchor_name = ordered_names[0]

    for attempt in range(1, max_retries + 2):
        result: dict[str, Any] | None = None
        successful_review_attempt = 0
        for review_attempt in range(1, review_max_retries + 2):
            try:
                result = review_character_reference_pack(
                    review_client,
                    view_paths,
                    character_description,
                    synthetic_styling,
                )
                successful_review_attempt = review_attempt
                break
            except Exception as exc:
                failed = {
                    "attempt": attempt,
                    "review_attempt": review_attempt,
                    "attempt_kind": "review_error",
                    "passed": False,
                    "views": {},
                    "cross_view": {"passed": False, "issues": [str(exc)]},
                    "failed_views": list(ordered_names),
                    "summary": f"review failed: {exc}",
                }
                attempts.append(failed)
                receipt = build_character_reference_qa_receipt(
                    char_id=char_id,
                    view_paths=view_paths,
                    attempts=attempts,
                    synthetic_styling=synthetic_styling,
                    generation_contract=generation_contract,
                )
                _write_json_atomic(report_path, receipt)
                if review_attempt <= review_max_retries:
                    print(
                        f"  [reference-qa] {char_id} review response invalid "
                        f"({review_attempt}/{review_max_retries + 1}): {exc}; "
                        "re-reviewing the same images"
                    )
                    continue
                raise CharacterReferenceQAError(
                    f"{char_id} reference review could not produce valid evidence after "
                    f"{review_attempt} attempt(s): {exc}"
                ) from exc

        if result is None:
            raise AssertionError("character reference review retry loop returned no result")
        if result.get("qa_verdict") == "manual_review":
            raise CharacterReferenceQAError(
                f"{char_id} reference QA requires manual review; no image retry was submitted"
            )
        attempts.append({
            "attempt": attempt,
            "review_attempt": successful_review_attempt,
            "attempt_kind": "semantic_review",
            **result,
        })
        receipt = build_character_reference_qa_receipt(
            char_id=char_id,
            view_paths=view_paths,
            attempts=attempts,
            synthetic_styling=synthetic_styling,
            generation_contract=generation_contract,
        )
        _write_json_atomic(report_path, receipt)
        if result["passed"]:
            print(f"  [reference-qa] {char_id} ✓ attempt {attempt}")
            return receipt
        if attempt > max_retries:
            raise CharacterReferenceQAError(
                f"{char_id} reference views failed after {attempt} QA attempt(s): "
                f"{result['failed_views']} — {result.get('summary') or 'view contract violation'}"
            )

        _archive_reference_attempt(char_dir, view_paths, attempt)
        failed_views = set(result.get("failed_views") or ordered_names)
        # If the identity anchor itself changes, every dependent view must be
        # regenerated from the new anchor to avoid a mixed-identity pack.
        if anchor_name in failed_views:
            failed_views.update(ordered_names)
        print(
            f"  [reference-qa] attempt {attempt} rejected; regenerating "
            f"{', '.join(name for name in ordered_names if name in failed_views)}"
        )
        for name in ordered_names:
            if name not in failed_views:
                continue
            identity_anchor = None if name == anchor_name else view_paths[anchor_name]
            _generate_reference_view(
                image_client,
                view_name=name,
                prompt=prompts[name],
                output_path=view_paths[name],
                size=_reference_view_size(name, default_size),
                identity_anchor=identity_anchor,
                correction=_view_correction(
                    result,
                    name,
                    require_synthetic=bool(synthetic_styling),
                ),
            )

    raise AssertionError("unreachable character reference QA state")


def generate_character(
    char_id: str,
    name: str,
    description: str,
    output_dir: str,
    style: str = "",
    negative: str = "",
    size: str = "2K",
    model: Optional[str] = None,
    skip_images: bool = False,
    variants: Optional[list] = None,
    identity_props: Optional[list] = None,
    synthetic_styling: Optional[dict[str, Any]] = None,
    review_client: Any | None = None,
    view_qa_max_retries: int = 2,
    review_qa_max_retries: int = 2,
    max_new_image_requests: int | None = None,
    before_provider_request: Callable[[dict[str, Any]], None] | None = None,
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
    normalized_identity_props = normalize_identity_props(identity_props or [])
    prop_detail_board_path: Path | None = None
    prop_detail_board_receipt: dict[str, Any] | None = None
    reference_board_path: Path | None = None
    if (
        max_new_image_requests is not None
        and (
            isinstance(max_new_image_requests, bool)
            or not isinstance(max_new_image_requests, int)
            or max_new_image_requests < 1
        )
    ):
        raise ValueError(
            "max_new_image_requests must be a positive integer or None"
        )

    print(f"\n{'='*60}")
    print(f"  Character Factory: {name} ({char_id})")
    print(f"  Output: {char_dir}")
    print(f"{'='*60}\n")

    # Initialize client (may be None if skip_images=True)
    client = None

    # Step 1: Generate separated references.  A failed request must remain a
    # failed request; silently switching to the retired combined sheet changes
    # the paid payload, geometry contract, and cache lineage.
    target_model = "kling" if model and "kling" in model.lower() else "seedance"
    seedream_model = (
        model if model and "seedream" in model.lower()
        else "doubao-seedream-5.0-lite"
    )
    if not skip_images:
        reference_prompts = build_model_reference_prompts(
            description,
            style,
            target_model,
            synthetic_styling=synthetic_styling,
        )
        generation_contract = build_reference_generation_contract(
            prompts=reference_prompts,
            model=seedream_model,
            synthetic_styling=synthetic_styling,
        )
        expected_views = {
            view_name: Path(char_dir) / f"{view_name}.png"
            for view_name in reference_prompts
        }
        report_path = Path(char_dir) / "character_reference_qa.json"
        existing_pack_complete = validate_character_reference_qa_receipt(
            report_path,
            expected_views,
            synthetic_styling=synthetic_styling,
            generation_contract=generation_contract,
        )
        if existing_pack_complete:
            print(
                f"[Step 1/3] Reusing exact {target_model} reference contract; "
                "zero image and review Provider requests..."
            )
        else:
            print(f"[Step 1/3] Generating separated {target_model} references...")
            try:
                pending_receipt = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                pending_receipt = None
            pending_is_current = bool(
                isinstance(pending_receipt, dict)
                and pending_receipt.get("schema")
                == CHARACTER_REFERENCE_QA_SCHEMA
                and pending_receipt.get("status") == "pending"
                and pending_receipt.get("synthetic_styling")
                == synthetic_styling
                and pending_receipt.get("generation_contract")
                == generation_contract
                and isinstance(pending_receipt.get("inputs"), dict)
            )
            if not pending_is_current:
                pending_receipt = {
                    "schema": CHARACTER_REFERENCE_QA_SCHEMA,
                    "character_id": char_id,
                    "status": "pending",
                    "inputs": {},
                    "synthetic_styling": synthetic_styling,
                    "generation_contract": generation_contract,
                    "attempts": [],
                }
                _write_json_atomic(report_path, pending_receipt)
            client = SeedreamClient(model=seedream_model)
        views = {name: str(path) for name, path in expected_views.items()}
        if not existing_pack_complete:
            views = {}
            identity_anchor = None
            new_image_requests = 0
            for view_name, reference_prompt in reference_prompts.items():
                view_path = expected_views[view_name]
                pending_input = (pending_receipt.get("inputs") or {}).get(
                    view_name
                )
                if (
                    isinstance(pending_input, dict)
                    and pending_input.get("path") == view_path.name
                    and view_path.is_file()
                    and pending_input.get("sha256") == file_sha256(view_path)
                ):
                    views[view_name] = str(view_path)
                    identity_anchor = identity_anchor or view_path
                    print(f"  [{view_name}] ♻ durable partial reference reused")
                    continue
                if (
                    max_new_image_requests is not None
                    and new_image_requests >= max_new_image_requests
                ):
                    raise RuntimeError(
                        "character reference generation crossed its configured "
                        "new-image request limit"
                    )
                _generate_reference_view(
                    client,
                    view_name=view_name,
                    prompt=reference_prompt,
                    output_path=view_path,
                    size=_reference_view_size(view_name, size),
                    identity_anchor=identity_anchor,
                    before_provider_request=before_provider_request,
                )
                identity_anchor = identity_anchor or view_path
                views[view_name] = str(view_path)
                new_image_requests += 1
                pending_receipt["inputs"][view_name] = {
                    "path": view_path.name,
                    "sha256": file_sha256(view_path),
                }
                _write_json_atomic(report_path, pending_receipt)
                print(f"  [{view_name}] ✓ → {view_path}")
                if (
                    max_new_image_requests is not None
                    and new_image_requests >= max_new_image_requests
                ):
                    raise CharacterReferenceGenerationPaused(
                        character_id=char_id,
                        view_name=view_name,
                        view_path=view_path,
                        character_description=description,
                        synthetic_styling=synthetic_styling,
                    )

        if not all(isinstance(path, str) and Path(path).is_file() for path in views.values()):
            raise RuntimeError(f"{char_id} reference generation produced an incomplete pack")
        if existing_pack_complete:
            qa_receipt = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            if review_client is None:
                from clients.ark_multimodal_client import ArkMultimodalClient

                review_client = ArkMultimodalClient()
            qa_receipt = _quality_control_reference_views(
                char_id=char_id,
                character_description=description,
                char_dir=Path(char_dir),
                prompts=reference_prompts,
                view_paths={name: Path(path) for name, path in views.items()},
                image_client=client,
                review_client=review_client,
                default_size=size,
                max_retries=view_qa_max_retries,
                review_max_retries=review_qa_max_retries,
                synthetic_styling=synthetic_styling,
                generation_contract=generation_contract,
            )
        reference_board_path = ensure_character_reference_board(
            Path(char_dir),
            character_id=char_id,
        )
        if normalized_identity_props:
            if client is None:
                client = SeedreamClient(model=seedream_model)
            if review_client is None:
                from clients.ark_multimodal_client import ArkMultimodalClient

                review_client = ArkMultimodalClient()
            prop_detail_board_path = Path(char_dir) / "prop_detail_board.png"
            detail_face_view = (
                "face_closeup"
                if "face_closeup" in views
                else "detail"
                if "detail" in views
                else "front"
            )
            detail_body_view = "full_body" if "full_body" in views else "front"
            canonical_paths = [
                Path(views[detail_face_view]),
                Path(views[detail_body_view]),
            ]
            prop_detail_board_receipt = _quality_control_identity_detail(
                char_id=char_id,
                character_description=description,
                identity_props=normalized_identity_props,
                style=style,
                char_dir=Path(char_dir),
                canonical_paths=canonical_paths,
                detail_path=prop_detail_board_path,
                image_client=client,
                review_client=review_client,
                max_retries=view_qa_max_retries,
            )
    else:
        print("[Step 1/3] Skipping image generation (--skip-images)")
        views = {
            name: None
            for name in build_model_reference_prompts(
                description,
                style,
                target_model,
                synthetic_styling=synthetic_styling,
            )
        }
        qa_receipt = None

    # Step 2: Create character_card.json
    print("[Step 2/3] Creating character_card.json...")
    reference_images = {
        name: f"characters/{char_id}/{name}.png" for name in views
    }
    card = create_character_card(
        char_id=char_id,
        name=name,
        description=description,
        style=style,
        negative=negative,
        seedream_model=seedream_model,
        reference_images=reference_images,
    )
    face_view = (
        "face_closeup"
        if "face_closeup" in views
        else "detail"
        if "detail" in views
        else "front"
    )
    body_view = "full_body" if "full_body" in views else "front"
    card["face_reference"] = f"characters/{char_id}/{face_view}.png"
    card["body_reference"] = f"characters/{char_id}/{body_view}.png"
    card["reference_strategy"] = (
        "kling_four_views"
        if "kling" in target_model.lower()
        else "seedance_four_views"
    )
    card["source_image_rules"] = SOURCE_IMAGE_RULES
    card["reference_contract_version"] = REFERENCE_CONTRACT_VERSION
    card["reference_generation_contract"] = (
        generation_contract if not skip_images else None
    )
    card["synthetic_styling"] = synthetic_styling
    card["reference_board"] = (
        f"characters/{char_id}/reference_board.png"
        if reference_board_path is not None
        else None
    )
    card["reference_board_receipt"] = (
        f"characters/{char_id}/reference_board.json"
        if reference_board_path is not None
        else None
    )
    card["identity_props"] = normalized_identity_props
    card["prop_detail_board"] = (
        f"characters/{char_id}/prop_detail_board.png"
        if normalized_identity_props
        else None
    )
    card["prop_detail_board_qa_report"] = (
        f"characters/{char_id}/prop_detail_board_qa.json"
        if prop_detail_board_receipt is not None
        else None
    )
    card["reference_qa_report"] = (
        f"characters/{char_id}/character_reference_qa.json"
        if qa_receipt is not None
        else None
    )
    card_path = os.path.join(char_dir, "character_card.json")
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {card_path}")

    # Step 3: Create angle_map.json
    print("[Step 3/3] Creating angle_map.json...")
    angle_map = create_angle_map(
        char_id=char_id,
        default_view=face_view,
        custom_mappings={
            "正面/特写/对话": f"{face_view}.png",
            "正面/全身/站立": f"{body_view}.png",
            "侧面/行走/奔跑": "side.png",
            "背面/远去/离开": "back.png",
            "面部/情绪/泪水": f"{face_view}.png",
        },
    )
    angle_path = os.path.join(char_dir, "angle_map.json")
    with open(angle_path, "w", encoding="utf-8") as f:
        json.dump(angle_map, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {angle_path}")

    # Historical state-specific image files are intentionally untouched for audit only. Plot
    # state such as wet clothing, dirt, or damage remains in Pxx start/end state.
    print("\n[Step 4] Legacy state variants disabled; existing files remain audit-only")

    # Summary
    result = {
        "char_id": char_id,
        "name": name,
        "char_dir": char_dir,
        "card": card_path,
        "angle_map": angle_path,
        "views": views,
        "prop_detail_board": (
            str(prop_detail_board_path) if prop_detail_board_path else None
        ),
        "variants": {},
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
                size=char.get("size", "2K"),
                model=char.get("model"),
                skip_images=kwargs.get("skip_images", False),
                variants=char.get("appearance", {}).get("variants", []),
                identity_props=char.get("appearance", {}).get("identity_props", []),
                synthetic_styling=char.get("appearance", {}).get("synthetic_styling"),
                review_client=kwargs.get("review_client"),
                view_qa_max_retries=kwargs.get("view_qa_max_retries", 2),
                review_qa_max_retries=kwargs.get("review_qa_max_retries", 2),
                max_new_image_requests=kwargs.get("max_new_image_requests"),
                before_provider_request=kwargs.get("before_provider_request"),
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ Failed to generate '{char.get('name', '?')}': {e}")
            if kwargs.get("raise_on_error", False):
                raise
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
    parser.add_argument("--size", default="2K", help="Seedream size tier or WxH")
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

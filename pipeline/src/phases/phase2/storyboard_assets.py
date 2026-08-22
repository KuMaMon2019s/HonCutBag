"""Storyboard templates, keyframes, end frames, and image artifact validation."""

from __future__ import annotations

import json
import math
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from prompt.shot_prompt_builder import build_batch_prompts
from quality.composition_validator import validate_composition
from utils.character_body_contracts import character_visual_description
from utils.file_integrity import _file_sha256
from utils.source_paths import PIPELINE_SRC_DIR as SCRIPT_DIR
from utils.storyboard_geometry import (
    SEEDREAM_MIN_PIXELS,
    _storyboard_canvas,
    _storyboard_image_size,
)


def load_storyboard_prompt_techniques() -> str:
    """加载 HonCut 分镜提示词技巧。

    返回精简版提示词技巧文本，追加到分镜生成 prompt 中，
    增强镜头语言、构图规则和画质控制。
    """
    techniques_path = SCRIPT_DIR.parent / "prompts" / "storyboard_prompt_techniques.md"
    if not techniques_path.exists():
        return ""
    try:
        return techniques_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def fill_storyboard_template(template: str, storyboard_data: dict, characters_data: dict) -> str:
    """填充故事板提示词模板

    从 STORYBOARD.json 提取镜头描述，从 CHARACTERS.json 提取角色描述，
    替换模板中的占位符。
    集成 HonCut 分镜提示词技巧（镜头语言、构图规则、画质控制）。

    注意：只提取代码块内的提示词部分，忽略文档说明。
    """
    # 提取代码块内的提示词（忽略文档说明）
    import re
    code_block_match = re.search(r'```\n(.*?)```', template, re.DOTALL)
    if code_block_match:
        prompt_template = code_block_match.group(1).strip()
    else:
        # 如果没有代码块，使用整个模板
        prompt_template = template.strip()

    # 提取镜头描述
    shots = storyboard_data.get("shots", [])
    storyboard_lines = []
    for i, shot in enumerate(shots, 1):
        scene = (
            shot.get("visual")
            or shot.get("scene")
            or shot.get("description")
            or shot.get("prompt")
            or ""
        )
        action = shot.get("action_description") or shot.get("action") or shot.get("what") or ""
        camera = (
            shot.get("camera_movement")
            or shot.get("camera_movement_en")
            or shot.get("camera")
            or shot.get("shot_type")
            or ""
        )
        scene = str(scene).strip()
        action = str(action).strip()
        camera = str(camera).strip()
        if not scene and not action:
            raise ValueError(f"storyboard shot {i} has no visual content")
        line = f"面板 {i}: {scene or action}"
        if action:
            line += f" — 动作: {action}"
        if camera:
            line += f" — 镜头: {camera}"
        storyboard_lines.append(line)
    storyboard_content = "\n".join(storyboard_lines) if storyboard_lines else "无分镜内容"

    # 提取角色描述
    characters = characters_data.get("characters", [])
    char_lines = []
    for c in characters:
        name = c.get("name", "未知角色")
        summary = character_visual_description(c)
        if summary:
            char_lines.append(f"- {name}: {summary}")
    character_reference = "\n".join(char_lines) if char_lines else "无角色描述"

    # 面板数量
    panel_count = len(shots) if shots else 12

    # 风格（默认值）
    style = "粗铅笔线条，细节最少，快速手势绘画能量"

    # 替换占位符
    prompt = prompt_template.replace("{{STORYBOARD_CONTENT}}", storyboard_content)
    prompt = prompt.replace("{{CHARACTER_REFERENCE}}", character_reference)
    prompt = prompt.replace("{{PANEL_COUNT}}", str(panel_count))
    prompt = prompt.replace("{{STYLE}}", style)

    # 追加 HonCut 分镜提示词技巧参考
    techniques = load_storyboard_prompt_techniques()
    if techniques:
        prompt += "\n\n---\n# 分镜提示词技巧参考\n\n"
        # 提取核心段落（跳过 YAML frontmatter 和主标题）
        tech_lines = techniques.split("\n")
        core_lines = []
        in_core = False
        for line in tech_lines:
            if line.startswith("## 核心原则"):
                in_core = True
            if in_core:
                core_lines.append(line)
        # 限制长度避免 prompt 过长（取前60行核心内容）
        prompt += "\n".join(core_lines[:60])

    return prompt


def _normalize_shot_id(shot_item: dict) -> Optional[str]:
    """Return a zero-padded shot ID, or ``None`` when no usable ID exists."""
    raw = (
        shot_item.get("shot_id")
        or shot_item.get("id")
        or shot_item.get("shot_order", 0)
    )
    if not raw:
        return None
    if isinstance(raw, int):
        return f"S{raw:02d}"

    raw_str = str(raw)
    # Legacy storyboards already store IDs such as ``S01``.
    if raw_str.upper().startswith("S"):
        raw_str = raw_str[1:]
    return f"S{raw_str.zfill(2)}"


def _shot_storyboard_reference(output_dir: Path, shot_id: Any) -> Optional[Path]:
    """Return a single-shot reference; never return the overview contact sheet."""
    normalized = _normalize_shot_id({"id": shot_id})
    if normalized is None:
        return None
    path = Path(output_dir) / "storyboard_images" / f"{normalized}.png"
    if path.is_file() and path.stat().st_size > 1024:
        return path
    return None


_ACTION_END_STATES = {
    # Chinese
    "抬手": "hand raised to its highest point, arm extended",
    "抬手拂发": "hand lowered after brushing hair aside, hair now clear of the face",
    "走来": "has arrived at the destination, standing steadily",
    "坐下": "seated steadily on the chair, posture relaxed",
    "转身": "has completed the turn, now facing the new direction",
    "拥抱": "arms wrapped around each other in a warm embrace",
    "牵手": "hands clasped together, fingers interlocked",
    "回头": "head turned to look back over the shoulder",
    "起身": "standing upright, fully risen from the seated position",
    "挥手": "hand raised in a waving gesture, arm extended",
    # English (base forms)
    "raise hand": "hand raised to its highest point, arm extended",
    "walk over": "has arrived at the destination, standing steadily",
    "sit down": "seated steadily on the chair, posture relaxed",
    "turn around": "has completed the turn, now facing the new direction",
    "embrace": "arms wrapped around each other in a warm embrace",
    "hold hands": "hands clasped together, fingers interlocked",
    "look back": "head turned to look back over the shoulder",
    "stand up": "standing upright, fully risen from the seated position",
    "wave": "hand raised in a waving gesture, arm extended",
    # English conjugated variants. Do not infer an unmentioned purpose.
    "raises her hand": "hand raised in the explicitly described gesture",
    "raises his hand": "hand raised in the explicitly described gesture",
    "brush away": "hand lowered after brushing, action completed",
    "brushes away": "hand lowered after brushing, action completed",
    "walks over": "has arrived at the destination, standing steadily",
    "walks toward": "has arrived at the destination, standing steadily",
    "sits down": "seated steadily on the chair, posture relaxed",
    "stands up": "standing upright, fully risen from the seated position",
    "turns around": "has completed the turn, now facing the new direction",
    "embraces": "arms wrapped around each other in a warm embrace",
    "hugs": "arms wrapped around each other in a warm embrace",
    "holds hands": "hands clasped together, fingers interlocked",
    "runs toward": "has arrived at the destination, standing steadily",
    "looks back": "head turned to look back over the shoulder",
    "waves": "hand raised in a waving gesture, arm extended",
}


def _derive_end_state(shot: dict) -> str:
    """Derive explicit end-state description from action verbs.

    Returns a concrete end-state sentence for known action verbs,
    or a generic fallback for unknown actions.
    """
    micro_actions = shot.get("micro_actions") or []
    if isinstance(micro_actions, list):
        explicit_end = next(
            (str(item).strip() for item in reversed(micro_actions) if str(item).strip()),
            "",
        )
        if explicit_end:
            return explicit_end

    action_text = " ".join(
        str(shot.get(field, "")) for field in ("visual", "what", "description")
        if shot.get(field)
    ).lower()

    # Check for known action verbs (longest match first to avoid partial matches)
    sorted_verbs = sorted(_ACTION_END_STATES.keys(), key=len, reverse=True)
    for verb in sorted_verbs:
        if verb.lower() in action_text:
            return _ACTION_END_STATES[verb]

    # Generic fallback
    return "the action described is fully completed, natural resting pose afterwards"


def build_end_frame_prompt(shot: dict) -> str:
    """Build rich t2i prompt for end frame generation.

    M3 fix: switched from i2i (copies reference) to t2i with rich description.
    Includes: end-state pose, scene context, character appearance, style anchors.
    """
    prompt = shot.get("prompt", shot.get("visual", ""))
    end_state = _derive_end_state(shot)
    subjects = shot.get("who") or []
    subjects = subjects if isinstance(subjects, list) else [subjects]
    subjects = [str(subject).strip() for subject in subjects if str(subject).strip()]
    subject_contract = ""
    if subjects:
        subject_contract = (
            f"The frame must contain exactly {len(subjects)} principal character(s): "
            f"{', '.join(subjects)}. Every named principal character must remain visible; "
            "do not omit, merge, duplicate, or replace anyone."
        )
    identity = str(shot.get("subject_description") or "").strip()

    # Extract character appearance from prompt (simple heuristic)
    char_desc = ""
    if "少女" in prompt or "girl" in prompt.lower():
        char_desc = "young woman in traditional attire"
    elif "少年" in prompt or "boy" in prompt.lower():
        char_desc = "young man"

    contract_parts = [f"Exact final micro-action: {end_state}."]
    if subject_contract:
        contract_parts.append(subject_contract)
    if identity:
        contract_parts.append(f"Identity and costume contract: {identity}.")
    return "\n\n".join(contract_parts) + "\n\n" + (
        f"Scene: {prompt}.\n\n"
        f"Character: {char_desc if char_desc else 'the character'}.\n\n"
        f"Style: maintain the same artistic style, lighting, and composition as the start frame.\n\n"
        f"The action has just completed; the character is in the final resting position.\n\n"
        f"Background, camera angle, and environment must match the start frame exactly.\n\n"
        "No text, no letters, no captions, no subtitles, and no speech bubbles."
    )


def fit_to_aspect(image_path: Path, target_w: int, target_h: int, output_path: Path) -> Path:
    """Resize image to exact target dimensions WITHOUT stretching.

    M5: If image already matches target aspect ratio (within 1% tolerance):
    simple high-quality resize. Otherwise: resize to COVER target aspect,
    then CENTER-CROP to exact dimensions. NEVER stretch/distort.

    Args:
        image_path: Source image path
        target_w: Target width in pixels
        target_h: Target height in pixels
        output_path: Where to save result (PNG)

    Returns:
        output_path (for chaining)
    """
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert('RGB')
        src_w, src_h = img.size

        src_aspect = src_w / src_h
        target_aspect = target_w / target_h
        aspect_tolerance = 0.01  # 1% tolerance

        if abs(src_aspect - target_aspect) / target_aspect <= aspect_tolerance:
            # Already matches aspect ratio — simple resize
            resized = img.resize((target_w, target_h), Image.LANCZOS)
        else:
            # Need to cover + center-crop
            scale_w = target_w / src_w
            scale_h = target_h / src_h
            scale = max(scale_w, scale_h)  # COVER = use larger scale

            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            resized_cover = img.resize((new_w, new_h), Image.LANCZOS)

            # Center-crop to exact target dimensions
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
            resized = resized_cover.crop((left, top, left + target_w, top + target_h))

        resized.save(str(output_path), 'PNG')

    return output_path


SEEDREAM_MIN_PIXELS = 3686400


FLF2V_SIMILARITY_LOW: float = 0.3


FLF2V_SIMILARITY_HIGH: float = 0.97


FLF2V_SHARPNESS_RATIO: float = 0.15


def _validate_end_frame(
    first_frame_path: Path,
    end_frame_path: Path,
    similarity_low: float = FLF2V_SIMILARITY_LOW,
    similarity_high: float = FLF2V_SIMILARITY_HIGH,
    sharpness_floor_ratio: float = FLF2V_SHARPNESS_RATIO,
    brightness_range: tuple = (15, 240),
) -> dict:
    """Validate end frame against first frame using metric-based checks.

    Returns dict with keys: passed, similarity, sharpness_ok, brightness_ok,
    resolution_ok, reason (if failed).

    Thresholds (M4, t2i-adapted):
      similarity_low=0.3   — scene drift floor (unchanged from M2)
      similarity_high=0.97 — catch true copies (0.99+), allow changed staging that
                             retains the same subject and luminous background
      sharpness_floor_ratio=0.15 — t2i is softer than i2i; 0.20× ratio is acceptable

    No VLM required — deterministic metric checks:
    1. Resolution identical to first frame
    2. Non-black, non-blank (mean brightness in range)
    3. Sharpness: Laplacian variance above floor (first_frame_variance × ratio)
    4. Similarity: perceptual distance must be in band [low, high]
       - Too similar (> high) → copy of first frame (no action progress)
       - Too different (< low) → scene/camera drifted
    """
    from PIL import Image
    import numpy as np

    result = {
        "passed": False,
        "similarity": None,
        "sharpness_ok": False,
        "brightness_ok": False,
        "resolution_ok": False,
        "reason": None,
        "thresholds": {
            "similarity_low": similarity_low,
            "similarity_high": similarity_high,
            "sharpness_ratio": sharpness_floor_ratio,
        },
    }

    try:
        first_img = Image.open(first_frame_path).convert("RGB")
        end_img = Image.open(end_frame_path).convert("RGB")
    except Exception as e:
        result["reason"] = f"cannot open images: {e}"
        return result

    # 1. Resolution normalization (M8): if sizes differ, normalize first frame
    #    to end frame dimensions via fit_to_aspect (COVER + center-crop, no stretch).
    #    This handles legacy square first frames (1920×1920) vs new 16:9 end frames (2560×1440).
    end_w, end_h = end_img.size
    if first_img.size != end_img.size:
        import tempfile
        tmp_first = Path(tempfile.mktemp(suffix=".png"))
        try:
            fit_to_aspect(first_frame_path, end_w, end_h, tmp_first)
            first_img = Image.open(tmp_first).convert("RGB")
        finally:
            tmp_first.unlink(missing_ok=True)
    result["resolution_ok"] = True

    # Convert to numpy arrays for metric computation
    first_arr = np.array(first_img)
    end_arr = np.array(end_img)

    # 2. Brightness check (non-black, non-blank)
    mean_brightness = float(np.mean(end_arr))
    result["brightness_ok"] = brightness_range[0] <= mean_brightness <= brightness_range[1]
    if not result["brightness_ok"]:
        result["reason"] = f"brightness out of range: {mean_brightness:.1f} (expected {brightness_range})"
        return result

    # 3. Sharpness check (Laplacian variance)
    def _laplacian_variance(arr):
        """Compute Laplacian variance as sharpness proxy."""
        gray = np.mean(arr, axis=2)  # RGB → grayscale
        # Simple 3×3 Laplacian kernel via array slicing
        lap = (
            gray[:-2, 1:-1] + gray[2:, 1:-1] +
            gray[1:-1, :-2] + gray[1:-1, 2:] -
            4 * gray[1:-1, 1:-1]
        )
        return float(np.var(lap))

    first_sharpness = _laplacian_variance(first_arr)
    end_sharpness = _laplacian_variance(end_arr)
    sharpness_floor = first_sharpness * sharpness_floor_ratio
    result["sharpness_ok"] = end_sharpness >= sharpness_floor
    if not result["sharpness_ok"]:
        result["reason"] = (
            f"too blurry: sharpness={end_sharpness:.1f} < floor={sharpness_floor:.1f} "
            f"(first_frame={first_sharpness:.1f} × {sharpness_floor_ratio})"
        )
        return result

    # 4. Similarity check (downsampled grayscale MSE → normalized similarity)
    def _downsample_gray(arr, target_size=64):
        """Downsample to small grayscale image for perceptual comparison."""
        gray = np.mean(arr, axis=2)
        # Simple box downsample
        h, w = gray.shape
        step_h = max(1, h // target_size)
        step_w = max(1, w // target_size)
        small = gray[::step_h, ::step_w][:target_size, :target_size]
        return small.astype(np.float64)

    first_small = _downsample_gray(first_arr)
    end_small = _downsample_gray(end_arr)

    # Pad to same shape if needed
    min_h = min(first_small.shape[0], end_small.shape[0])
    min_w = min(first_small.shape[1], end_small.shape[1])
    first_small = first_small[:min_h, :min_w]
    end_small = end_small[:min_h, :min_w]

    # MSE → similarity (1.0 = identical, 0.0 = maximally different)
    mse = float(np.mean((first_small - end_small) ** 2))
    # Normalize: max possible MSE for 8-bit images is 255^2 = 65025
    similarity = max(0.0, 1.0 - mse / 65025.0)
    result["similarity"] = round(similarity, 4)

    if similarity < similarity_low:
        result["reason"] = (
            f"too different (scene drift): similarity={similarity:.4f} < {similarity_low}"
        )
        return result

    if similarity > similarity_high:
        result["reason"] = (
            f"too similar (no action progress): similarity={similarity:.4f} > {similarity_high}"
        )
        return result

    result["passed"] = True
    return result


def _end_frame_sidecar_path(end_frame_path: Path) -> Path:
    """Return the sidecar meta JSON path for an end frame."""
    return end_frame_path.with_name(end_frame_path.stem + "_end.meta.json")


def _write_end_frame_sidecar(
    end_frame_path: Path,
    first_frame_sha: str,
    prompt_sha: str,
    validation: dict,
):
    """Write cache sidecar for end frame."""
    sidecar = _end_frame_sidecar_path(end_frame_path)
    sidecar.write_text(json.dumps({
        "first_frame_sha256": first_frame_sha,
        "prompt_sha256": prompt_sha,
        "validation": validation,
    }, indent=2))


def _read_end_frame_sidecar(end_frame_path: Path) -> Optional[dict]:
    """Read cache sidecar, or None if missing/invalid."""
    sidecar = _end_frame_sidecar_path(end_frame_path)
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text())
    except Exception:
        return None


def _generate_flf2v_end_frame(
    shot_item: dict,
    shot_id: str,
    first_frame_path: Path,
    ref_image_path: Optional[Path | list[Path]],
) -> bool:
    """Generate one idempotent Seedream end frame for an FLF2V shot.

    Prefer character identity references over unconstrained text-to-image.  Pure
    t2i can pass coarse similarity checks while silently dropping a character
    or changing identity.  The start frame remains validation-only so the end
    pose is not encouraged to become a near-copy.

    Cache: uses sidecar meta JSON with first_frame_sha + prompt_sha + validation.
    Validation: metric-based checks (resolution, brightness, sharpness, similarity band).
    """
    if shot_item.get("gen_strategy", "i2v") != "flf2v":
        return False

    # First frame MUST exist — used for validation and size reference
    if not first_frame_path.exists():
        raise FileNotFoundError(
            f"[FLF2V] {shot_id}: first frame {first_frame_path} not found. "
            "Cannot generate end frame without the start frame as reference."
        )

    end_path = first_frame_path.with_name(f"{shot_id}_end.png")
    prompt = build_end_frame_prompt(shot_item)
    first_frame_sha = _file_sha256(first_frame_path)
    import hashlib as _hashlib
    prompt_sha = _hashlib.sha256(f"identity_refs_v2_micro_end\n{prompt}".encode()).hexdigest()

    # Check cache sidecar
    sidecar = _read_end_frame_sidecar(end_path)
    if (
        sidecar is not None
        and end_path.exists()
        and sidecar.get("first_frame_sha256") == first_frame_sha
        and sidecar.get("prompt_sha256") == prompt_sha
        and sidecar.get("validation", {}).get("passed")
    ):
        print(f"    ⏭ [FLF2V] {end_path.name} cached+validated, skipping")
        return False

    # Generate end frame with all declared character identity references.  Do
    # not include the start frame in the reference set: that previously caused
    # near-identical copies with no action progress.
    from clients.seedream_client import SeedreamClient
    client = SeedreamClient()
    # M5: use video target aspect ratio (16:9), not first frame's dimensions
    video_w = shot_item.get("width", 1280)
    video_h = shot_item.get("height", 720)
    size = _storyboard_image_size(video_width=video_w, video_height=video_h)

    identity_refs = []
    if isinstance(ref_image_path, list):
        identity_refs = [Path(path) for path in ref_image_path if Path(path).exists()]
    elif ref_image_path is not None and Path(ref_image_path).exists():
        identity_refs = [Path(ref_image_path)]

    try:
        if identity_refs:
            client.image_to_image(
                prompt=prompt,
                ref_image=[str(path) for path in identity_refs],
                output_path=str(end_path),
                size=size,
            )
            print(
                f"    [FLF2V] 终帧 {end_path.name} ✓ "
                f"(identity refs: {len(identity_refs)})"
            )
        else:
            client.text_to_image(
                prompt=prompt,
                output_path=str(end_path),
                size=size,
            )
            print(f"    [FLF2V] 终帧 {end_path.name} ✓ (t2i fallback: no identity refs)")
    except Exception as e:
        # Last-resort scene-preserving fallback.
        print(f"    ⚠ [FLF2V] identity generation failed ({e}), falling back to start-frame i2i")
        client.image_to_image(
            prompt=prompt,
            ref_image=str(first_frame_path),
            output_path=str(end_path),
            size=size,
        )
        print(f"    [FLF2V] 终帧 {end_path.name} ✓ (i2i fallback)")

    # Validate the generated end frame
    validation = _validate_end_frame(first_frame_path, end_path)
    _write_end_frame_sidecar(end_path, first_frame_sha, prompt_sha, validation)

    if not validation["passed"]:
        reason = validation.get("reason", "unknown")
        print(f"    ⚠ [FLF2V] {end_path.name} validation FAILED: {reason}")
        # Retry once with the same identity-locked route.
        print(f"    [FLF2V] retrying {end_path.name}...")
        if identity_refs:
            client.image_to_image(
                prompt=prompt,
                ref_image=[str(path) for path in identity_refs],
                output_path=str(end_path),
                size=size,
            )
        else:
            client.text_to_image(
                prompt=prompt,
                output_path=str(end_path),
                size=size,
            )
        validation = _validate_end_frame(first_frame_path, end_path)
        _write_end_frame_sidecar(end_path, first_frame_sha, prompt_sha, validation)

        if not validation["passed"]:
            reason = validation.get("reason", "unknown")
            print(f"    ✗ [FLF2V] {end_path.name} retry FAILED: {reason}")
            raise RuntimeError(
                f"[FLF2V] {shot_id}: end frame validation failed after retry: {reason}"
            )

    sim = validation.get("similarity", "N/A")
    print(f"    ✓ [FLF2V] {end_path.name} validated (similarity={sim})")
    return True


def _storyboard_keyframe_description(shot: dict) -> str:
    """Build an identity-locked, single-moment prompt for one storyboard frame."""
    who_declared = "who" in shot
    who = shot.get("who") or []
    identity = str(shot.get("subject_description") or "").strip()
    action = str(
        shot.get("action_description") or shot.get("what") or ""
    ).strip()
    # Dialogue belongs to the later audio/subtitle track, not the visual
    # keyframe.  Leaving quoted lines inside a Seedream prompt frequently
    # produces comic speech bubbles which then become baked into the video.
    action = re.sub(r"[“\"](?:(?![”\"]).){1,120}[”\"]", "", action)
    action = re.sub(r"\s+", " ", action).strip()
    staging = str(shot.get("visual") or "").strip()
    generation_actions = shot.get("generation_actions") or []
    is_action_shot = bool(generation_actions) or str(shot.get("shot_intent") or "").lower() == "action"
    start_state = str(shot.get("start_state") or "").strip()
    # Only explicit who=[] is an environment contract. Legacy storyboards may
    # omit ``who`` while carrying character identity and action in the older
    # subject/action fields.
    if who_declared and not who:
        parts = [
            "Environment-only cinematic keyframe.",
            "Depict exclusively the described clouds, landscape, architecture, light, and atmosphere.",
            f"Exact environment contract: {action}." if action else "",
            f"Visual staging: {staging}." if staging else "",
            "The frame is uninhabited: zero people, zero humanoid figures, and zero unrelated objects.",
        ]
    else:
        parts = [
            "Single decisive cinematic keyframe.",
            (
                "Character identity lock (gender, hair, face, clothing, and body proportions "
                f"must remain exact): {identity}."
                if identity
                else ""
            ),
            (
                f"Exact starting-state contract before any action begins: {start_state}."
                if is_action_shot and start_state
                else (
                    f"Show the poised starting pose immediately before this first action: "
                    f"{generation_actions[0]}."
                    if is_action_shot and generation_actions
                    else f"Exact action contract: {action}." if action else ""
                )
            ),
            (
                "This is frame zero: the first attack, impact, and result have not happened yet. "
                "Do not depict contact, sparks, damage, or the final pose."
                if is_action_shot
                else "Show one decisive final pose of that exact action contract."
            ),
            (
                f"Scene and composition: {shot.get('where', '')}."
                if is_action_shot
                else f"Visual staging: {staging}." if staging else ""
            ),
            "Depict only subjects, props, and actions explicitly named in the contract.",
            "No exposed midriff unless the identity contract explicitly requires it.",
            "No text, no letters, no captions, no subtitles, and no speech bubbles; dialogue is audio-only.",
        ]
    return " ".join(part for part in parts if part)


def _generate_shot_images(
    output_dir: Path,
    storyboard_data: dict,
    regenerate_shot_ids: set[str] | None = None,
) -> int:
    """Generate storyboard images for each shot (M2 task).

    Args:
        output_dir: Project output directory
        storyboard_data: Storyboard data with shots list

    Returns:
        Number of successfully generated images
    """
    try:
        video_width, video_height, _aspect_ratio = _storyboard_canvas(storyboard_data)
        storyboard_images_dir = output_dir / "storyboard_images"
        storyboard_images_dir.mkdir(exist_ok=True)
        shots = storyboard_data.get("shots", [])
        prompt_scenes = []
        for shot in shots:
            prompt_scene = dict(shot)
            prompt_scene["description"] = _storyboard_keyframe_description(shot)
            prompt_scene.setdefault("shot_language", {
                "shot_size": shot.get("shot_size"),
                "camera_movement": shot.get("camera_movement"),
                "lighting_key": shot.get("lighting_key"),
            })
            prompt_scenes.append(prompt_scene)
        # Each shot's visual/action contract already contains its intended style.
        # A project-level LLM style summary may mention plot objects (palace,
        # character, props); injecting it into every shot leaks future content.
        batch_prompts = build_batch_prompts(
            prompt_scenes,
            None,
        )
        prompt_by_id = {str(item["scene_id"]): item["prompt"] for item in batch_prompts}

        # --- P0-1a: Load character reference images (for shot image consistency) ---
        char_ref_map = {}  # {char_name_lower: preferred_reference_path}
        protagonist_ref = None
        chars_path = output_dir / "CHARACTERS.json"
        if chars_path.exists():
            try:
                chars_data = json.loads(chars_path.read_text())
                for char in chars_data.get("characters", []):
                    reference_path = None
                    for char_dir in (
                        output_dir / "characters" / char["id"],
                        output_dir / "characters" / "characters" / char["id"],
                    ):
                        candidates = [
                            char_dir / "face_closeup.png",
                            char_dir / "full_body.png",
                            *sorted(char_dir.glob("variant_*.png")),
                            char_dir / "front.png",  # legacy fallback
                        ]
                        reference_path = next(
                            (path for path in candidates if path.exists()), None
                        )
                        if reference_path is not None:
                            break
                    if reference_path is not None:
                        char_ref_map[char["name"].lower()] = reference_path
                        char_ref_map[char["id"].lower()] = reference_path
                        if protagonist_ref is None:
                            protagonist_ref = reference_path
                if char_ref_map:
                    print(f"  → [P0-1] 已加载 {len(char_ref_map)//2} 个角色参考图")
            except Exception as e:
                print(f"  ⚠ [P0-1] 角色参考图加载失败: {e}")

        generated_count = 0

        # Phase 2 runs before the character factory. Character shots must wait
        # until Phase 3 has produced references instead of silently using t2i.
        has_character_shots = any(shot.get("who") for shot in shots)
        if has_character_shots and not char_ref_map:
            print(
                "  → [M2] 角色参考图尚未生成；逐镜分镜图延后到 Phase 3",
                flush=True,
            )
            return 0

        # --- P2-5d: HonCut concurrent shot image generation ---
        def _gen_shot_image(shot_item):
            """Single shot image generation logic (for concurrent calls)"""
            shot_id = _normalize_shot_id(shot_item)
            if shot_id is None:
                print("    ⚠ [M2] 分镜缺少有效 shot_id/id/shot_order，跳过")
                return None
            shot_prompt = prompt_by_id.get(str(shot_item.get("id", ""))) or shot_item.get("prompt", shot_item.get("visual", ""))
            if not shot_prompt:
                return None
            shot_image_path = storyboard_images_dir / f"{shot_id}.png"

            # --- P0-1c: Match character reference image ---
            # Support structured who[] from storyboard:
            # - Empty who [] → pure landscape/no_character → NO reference injection
            # - Single character → use that character's preferred reference
            # - Multiple characters → use first character's preferred reference
            ref_image_paths = []
            shot_who = shot_item.get("who", [])
            if not isinstance(shot_who, list):
                shot_who = [shot_who] if shot_who else []

            if len(shot_who) == 0:
                # Pure landscape / no_character shot — do NOT inject any character reference
                print(f"    [M2] {shot_id}: 纯风景镜头(who=[]), 不注入角色参考")
            else:
                # Match first available character reference
                for name in shot_who:
                    reference = char_ref_map.get(str(name).lower())
                    if reference is not None and reference not in ref_image_paths:
                        ref_image_paths.append(reference)
                # Fallback to protagonist only if who[] is non-empty but no match found
                if not ref_image_paths and protagonist_ref:
                    ref_image_paths.append(protagonist_ref)

            ref_image_path = ref_image_paths[0] if ref_image_paths else None

            stale_for_references = bool(
                shot_image_path.exists()
                and ref_image_paths
                and any(
                    reference.stat().st_mtime > shot_image_path.stat().st_mtime
                    for reference in ref_image_paths
                )
            )
            force_regenerate = bool(
                regenerate_shot_ids and shot_id in regenerate_shot_ids
            )
            if (
                shot_image_path.exists()
                and not stale_for_references
                and not force_regenerate
            ):
                _generate_flf2v_end_frame(
                    shot_item, shot_id, shot_image_path, ref_image_paths
                )
                return shot_id
            if stale_for_references:
                print(
                    f"    [M2] {shot_id}: 角色参考图较新，刷新旧分镜图",
                    flush=True,
                )
            elif force_regenerate:
                print(f"    [M2] {shot_id}: 按质检清单定向重绘", flush=True)

            # --- 429 retry with exponential backoff ---
            import time as _time
            _m2_max_retries = 3
            _m2_wait_times = [120, 240, 480]
            for _m2_attempt in range(1, _m2_max_retries + 1):
                try:
                    _m2_size = _storyboard_image_size(
                        video_width=video_width,
                        video_height=video_height,
                    )
                    if ref_image_paths and all(path.exists() for path in ref_image_paths):
                        # P0-1c: Use image_to_image mode (with reference image)
                        from clients.seedream_client import SeedreamClient
                        client = SeedreamClient()
                        client.image_to_image(
                            prompt=shot_prompt,
                            ref_image=[str(path) for path in ref_image_paths],
                            output_path=str(shot_image_path),
                            size=_m2_size,
                        )
                        refs = ", ".join(path.name for path in ref_image_paths)
                        print(f"    [M2] 分镜图 {shot_id}.png ✓ (refs: {refs})")
                    else:
                        # No reference image, pure text-to-image
                        from clients.seedream_client import text_to_image
                        text_to_image(prompt=shot_prompt, output_path=str(shot_image_path), size=_m2_size)
                    print(f"    [M2] 分镜图 {shot_id}.png ✓")
                    _generate_flf2v_end_frame(
                        shot_item, shot_id, shot_image_path, ref_image_paths
                    )
                    return shot_id
                except Exception as e:
                    _err_str = str(e)
                    _is_429 = (
                        "429" in _err_str
                        or "Too Many Requests" in _err_str
                        or (hasattr(e, "response") and getattr(getattr(e, "response", None), "status_code", None) == 429)
                    )
                    if _is_429 and _m2_attempt < _m2_max_retries:
                        _wait = _m2_wait_times[_m2_attempt - 1]
                        print(f"    [M2] {shot_id} retry {_m2_attempt}/{_m2_max_retries} (429, wait {_wait}s)...")
                        _time.sleep(_wait)
                        continue
                    else:
                        # Non-429 or retries exhausted → raise
                        print(f"    [M2] {shot_id}.png ✗ → {e}")
                        raise
            return None

        # Concurrent execution (max_workers=3)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_gen_shot_image, s): s for s in shots}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        generated_count += 1
                except Exception as e:
                    shot = futures[future]
                    shot_id = _normalize_shot_id(shot) or "<missing>"
                    print(f"    [M2] 分镜图 {shot_id}.png 并发失败（降级跳过）: {e}")
        print(f"  → [M2] 分镜图序列: {generated_count}/{len(shots)} 张")
        return generated_count
    except Exception as e:
        print(f"  ⚠ [M2] 分镜图序列生成失败（降级跳过）: {e}")
        return 0


def _validate_storyboard_image_composition(output_dir: Path, storyboard_data: dict) -> dict:
    """Validate that every generated storyboard cut has its required image asset."""
    cuts = []
    cursor = 0.0
    for shot in storyboard_data.get("shots", []):
        shot_id = _normalize_shot_id(shot)
        if shot_id is None:
            continue
        duration = float(shot.get("duration", 5))
        cuts.append({
            "id": shot_id,
            "source": f"storyboard_images/{shot_id}.png",
            "in_seconds": cursor,
            "out_seconds": cursor + duration,
        })
        cursor += duration
    report = validate_composition({"cuts": cuts}, output_dir)
    (output_dir / "storyboard_composition_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report

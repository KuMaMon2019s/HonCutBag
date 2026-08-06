"""Emotion → facial expression / eye / lighting mapping for video prompts.

Supports HonCut's realpeople_urban_modern storyboard style.
Used by storyboard_generator.py to enrich prompts with emotion-specific details.
"""

# Emotion → (face_words_en, eye_words_en, lighting_words_en)
EMOTION_MAP = {
    "心动": ("subtle smile, cheeks slightly flushed", "bright eyes, focused gaze", "soft backlight, warm scattered light"),
    "欣喜": ("subtle smile, cheeks slightly flushed", "bright eyes, focused gaze", "soft backlight, warm scattered light"),
    "悲伤": ("calm face, subdued expression", "dim eyes, wandering gaze", "diffused cold light, low-key lighting"),
    "失落": ("calm face, subdued expression", "dim eyes, wandering gaze", "diffused cold light, low-key lighting"),
    "愤怒": ("sharp brows, cold stern expression", "intense piercing gaze", "hard side light, high contrast"),
    "压迫": ("sharp brows, cold stern expression", "intense piercing gaze", "hard side light, high contrast"),
    "温柔": ("gentle expression, warm brows", "focused tender gaze", "soft backlight, shallow depth of field"),
    "深情": ("gentle expression, warm brows", "focused tender gaze", "soft backlight, shallow depth of field"),
    "坚定": ("serious composed expression", "steady resolute gaze", "neutral even lighting"),
    "决绝": ("serious composed expression", "steady resolute gaze", "neutral even lighting"),
    "惊讶": ("slightly stunned expression", "wide eyes, suddenly focused gaze", "ambient light unchanged"),
    "震惊": ("slightly stunned expression", "wide eyes, suddenly focused gaze", "ambient light unchanged"),
    "冷漠": ("cold indifferent expression", "distant empty gaze", "cold blue side light"),
    "疏离": ("cold indifferent expression", "distant empty gaze", "cold blue side light"),
    "喜悦": ("vivid expression, radiant smile", "bright lively eyes", "warm natural light"),
    "雀跃": ("vivid expression, radiant smile", "bright lively eyes", "warm natural light"),
    "紧张": ("slightly dazed expression", "restless wandering gaze", "ambient light unchanged"),
    "慌乱": ("slightly dazed expression", "restless wandering gaze", "ambient light unchanged"),
    "隐忍": ("restrained composed expression", "deep suppressed emotion in eyes", "low-key lighting"),
    "克制": ("restrained composed expression", "deep suppressed emotion in eyes", "low-key lighting"),
    "暧昧": ("subtle knowing smile", "warm lingering gaze", "soft warm ambient, gentle contrast"),
    "羞涩": ("shy averted expression, flushed ears", "darting shy gaze", "soft diffused warm light"),
}

# Style anchor words (mandatory in every prompt)
STYLE_ANCHOR = (
    "Photorealistic cinematography, cinematic quality, ultra-fine detail, "
    "strong contrast, delicate skin texture, detailed facial rendering, "
    "strand-by-strand hair detail, modern urban aesthetic, oriental temperament. "
    "Ultra-sharp 4K, high detail, natural sharpness, no subtitles, no watermark."
)

# Scene → lighting defaults (when no specific emotion)
SCENE_LIGHTING = {
    "雨天": "diffused cold light, no key light, grey-blue tone, humid atmosphere, low saturation",
    "阴天": "diffused cold light, no key light, grey-blue tone, low saturation",
    "傍晚": "warm backlight, oblique golden hour glow, long shadows stretching",
    "黄昏": "warm backlight, oblique golden hour glow, long shadows stretching",
    "夜间": "cold blue window light, warm interior point light, cold-warm contrast",
    "夜晚": "cold blue window light, warm interior point light, cold-warm contrast",
    "室内": "overhead ambient light, neutral grey tone, soft even illumination",
    "办公室": "overhead ambient light, neutral grey tone, soft even illumination",
}


def get_emotion_words(emotion: str) -> tuple:
    """Get (face, eye, lighting) words for an emotion.
    
    Tries exact match first, then partial match.
    Returns empty strings if no match.
    """
    if not emotion:
        return ("", "", "")
    
    # Exact match
    if emotion in EMOTION_MAP:
        return EMOTION_MAP[emotion]
    
    # Partial match (emotion might be '心动/期待' or '紧张又期待')
    for key, value in EMOTION_MAP.items():
        if key in emotion:
            return value
    
    return ("", "", "")


def get_scene_lighting(scene: str) -> str:
    """Get default lighting for a scene type."""
    if not scene:
        return ""
    for key, value in SCENE_LIGHTING.items():
        if key in scene:
            return value
    return ""


def build_style_suffix(emotion: str = "", scene: str = "") -> str:
    """Build the style/lighting suffix for a prompt.
    
    Combines emotion-specific face/eye words + scene lighting + style anchor.
    """
    parts = []
    
    face, eye, lighting = get_emotion_words(emotion)
    if face:
        parts.append(face)
    if eye:
        parts.append(eye)
    
    # Lighting: emotion-specific overrides scene default
    if lighting and lighting != "ambient light unchanged":
        parts.append(lighting)
    else:
        scene_light = get_scene_lighting(scene)
        if scene_light:
            parts.append(scene_light)
    
    # Always append style anchor
    parts.append(STYLE_ANCHOR)
    
    return ", ".join(parts)

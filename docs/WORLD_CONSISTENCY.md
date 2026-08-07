# World Consistency & Continuity Error Prevention

> Root cause analysis of visual inconsistencies in HonCut V5 test output

## Problem Statement

V5 test (westlake_seedance_v5) produced 22.67s video with 3 major issues:
1. **Character hairstyle inconsistency** — hairstyle changed between shots
2. **Subtitles are narration, not dialogue** — captions describe scenes instead of showing character speech
3. **Camera movement fragmentation** — all 5 shots use `static` camera, jarring transitions

---

## P0-A: Identity Anchor (Character Consistency)

### Root Cause
`storyboard_generator.py` line 287-288 uses `appearance.summary` (natural language sentence) instead of verbatim repeating structured visual attributes.

### Current Code
```python
summary = appearance.get("summary", "")
if summary:
    identity_prefix += f"{char_name} — {summary} — "
```

### Problem
`summary` is a Chinese sentence like "20多岁纤细的年轻女生，黑色长直发及肩..." — Seedance doesn't understand this well.

### Fix: Verbatim Structured Attributes
```python
# Extract structured fields and build English visual feature list
hair = appearance.get("hair", "")           # "黑色长直发及肩"
face = appearance.get("face", "")           # "鹅蛋脸，细柳叶眉，柔和杏仁眼，小巧鼻梁，淡粉色唇"
clothing = appearance.get("clothing", "")   # "浅米色棉麻短袖衬衫+浅蓝色A字牛仔半身裙..."
distinguishing = appearance.get("distinguishing", "")

# Build English prompt fragment (translate or use pre-translated)
identity_features = f"hair: {hair}, face: {face}, clothing: {clothing}"
if distinguishing:
    identity_features += f", distinguishing: {distinguishing}"
```

### Reference: OpenMontage Iron Law
> "Repeat identity verbatim across every shot. 'the same character' / pronouns / 'Aang again' **do not work**. Repeat the 3-6 disambiguating visual attributes verbatim in every shot block."

### Toonflow Reference
- **File**: `toonflow/pipelines/seedance_pipeline.py` line ~300-340
- **Approach**: Uses `@图N:角色名参考图` syntax to bind reference images

### Acceptance Criteria
- Each shot prompt containing a character includes specific visual features
- Example: "Lin Xiao — black long straight hair to shoulders, oval face, willow-leaf eyebrows, gentle almond eyes, small nose bridge, light pink lips, light beige cotton linen short-sleeve shirt, light blue A-line denim half skirt, white low-top canvas sneakers, beige woven crossbody bag"

---

## P0-B: Dialogue Subtitles (Not Narration)

### Root Cause
1. `storyboard_generator.py` only generates `caption` (scene description)
2. Original script has no dialogue — Phase 2 (编剧引擎) doesn't generate character speech
3. `speech_duration_s: 0` for all shots

### Current Data Structure
```json
{
    "caption": "西湖边静坐看晚霞",  // ← narration/scene description
    "caption_frames": "18-162",
    "speech_duration_s": 0          // ← no dialogue!
}
```

### Fix: Add Dialogue Generation
1. **Phase 2 (adaptation_engine.py)**: Generate `dialogue` field for each shot
2. **storyboard_generator.py**: Output both `dialogue` and `narration` fields
3. **subtitle_burn.py**: Different styles for dialogue vs narration

### Proposed Data Structure
```json
{
    "dialogue": "这里的晚霞真美...",  // ← character speech (if any)
    "narration": "西湖边静坐看晚霞",  // ← scene description fallback
    "caption_frames": "18-162",
    "speech_duration_s": 3.5,
    "speaker": "林晓"
}
```

### Subtitle Style Differentiation
| Type | Style | Position |
|------|-------|----------|
| Dialogue | White text, black outline | Bottom center |
| Narration | Yellow text, no outline | Top center |

### Reference: Toonflow Dialogue System
- **File**: `toonflow/pipelines/script_pipeline.py` line ~200-250
- **Approach**: LLM generates dialogue based on character personality and scene context

### Acceptance Criteria
- When character speaks: show dialogue subtitle
- When no dialogue: fallback to narration subtitle
- Different visual styles for dialogue vs narration

---

## P1-A: Camera Movement Coherence

### Root Cause
All 5 shots have `camera_movement: "static"` — no variation, jarring transitions.

### Current Data
| Shot | shot_size | camera_movement | shot_intent |
|------|-----------|-----------------|-------------|
| S1 | establishing | static | establishing |
| S2 | close_up | static | transition |
| S3 | medium_close | static | reveal |
| S4 | wide | static | atmosphere |
| S5 | medium_close | static | emotional |

### Fix: Camera Movement Constraints
1. Adjacent shots should not both be `static`
2. Same-scene shots should share visual parameters (lighting, color tone)
3. Camera movement should follow cinematic logic

### Proposed Rules
```python
# Rule 1: No consecutive static shots
if prev_shot.camera_movement == "static":
    next_shot.camera_movement = random.choice(["slow_pan", "slow_zoom", "tilt"])

# Rule 2: Same scene = same lighting
if next_shot.where == prev_shot.where:
    next_shot.lighting_key = prev_shot.lighting_key
    next_shot.color_tone = prev_shot.color_tone
```

### Reference: OpenMontage shot_language
- **File**: `openmontage/lib/shot_prompt_builder.py` line ~150-180
- **Approach**: Enumerated camera movements with transition logic

### Acceptance Criteria
- No two consecutive shots use `static` camera
- Same-scene shots share lighting/color parameters
- Camera movement follows shot intent (e.g., `reveal` → `slow_zoom`)

---

## Implementation Priority

| Task | Priority | Files | Effort |
|------|----------|-------|--------|
| P0-A: Identity Anchor | 🔴 P0 | `storyboard_generator.py` | Small |
| P0-B: Dialogue Subtitles | 🔴 P0 | `storyboard_generator.py`, `subtitle_burn.py` | Medium |
| P1-A: Camera Coherence | 🟡 P1 | `storyboard_generator.py`, `prompt_router.py` | Medium |

---

## Minimal Change Approach

Modify only `_build_shot_prompt()` in `storyboard_generator.py`:

1. **Identity Anchor**: Read `hair/face/clothing/distinguishing` from CHARACTERS.json, repeat verbatim in each shot prompt
2. **Dialogue Generation**: Add `dialogue` field generation (if character speaks) or `narration`
3. **Camera Variation**: Ensure adjacent shots don't all use `static`

### Pseudocode
```python
def _build_shot_prompt(shot, characters, scene_style_map):
    # 1. Identity Anchor (verbatim structured attributes)
    identity = build_identity_features(shot["who"], characters)
    
    # 2. Dialogue or narration
    text = shot.get("dialogue") or shot.get("narration") or shot.get("visual")
    
    # 3. Camera movement (ensure variation)
    camera = ensure_camera_variation(shot, prev_shot)
    
    return format_prompt(identity, text, camera, shot["lighting_key"])
```

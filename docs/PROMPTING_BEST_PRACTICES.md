# HonCut Prompting Best Practices

This guide adapts the OpenMontage FLUX prompting reference for HonCut's storyboard and Seedance workflow. Treat these rules as defaults, then refine one variable at a time from generated results.

## Core Principles

### 1. Prose Style Over Keywords

Write a compact visual description with relationships between subject, action, setting, and light.

**Wrong**: `woman, portrait, beautiful, blonde, studio, 8k`

**Right**: `A professional studio portrait of a blonde woman in her thirties, captured with soft studio lighting that accentuates her features.`

Keyword lists leave composition and relationships ambiguous. Natural prose gives the model a coherent scene to render.

### 2. Front-Load Important Elements

FLUX prioritizes elements that appear earlier in the prompt. Seedance prompts also benefit from establishing the main subject and continuity constraints before secondary scene detail.

**Wrong**: `A forest background where a knight stands.`

**Right**: `A knight in shining armor stands in a forest, illuminated by soft dappled light.`

**HonCut application**: Identity-lock blocks belong before action, setting, and style in `_build_shot_prompt()`. This makes character continuity a primary instruction rather than an afterthought.

### 3. Always Specify Lighting

Lighting has the single greatest impact on image quality. Name the source, direction or quality, color, and mood when they matter.

Natural lighting:

- Golden hour: warm, soft, directional
- Overcast: soft, diffused, even
- Harsh midday: high contrast, strong shadows
- Dappled forest light: organic pools of light and shadow

Studio lighting:

- Softbox: even and professional
- Rim light: edge definition and subject separation
- Butterfly lighting: beauty and glamour
- Rembrandt lighting: dramatic, classic portraiture

Atmospheric lighting:

- Volumetric fog: depth and mystery
- Neon glow: urban and cyberpunk
- Candlelight: warm and intimate
- Low-key lighting: moody and tense

**HonCut application**: Set `lighting_key` for each shot and retain the visual-style lighting language. Avoid leaving a generic `natural lighting` fallback when the scene establishes a time of day or mood.

### 4. Aim for 30–80 Words

- Too short: generic results with insufficient direction
- Too long: competing details can dilute the focal subject
- Sweet spot: enough detail to guide composition without confusing priorities

Identity-lock text may push HonCut prompts beyond 80 words. Continuity is more important than hitting the range exactly; keep the remaining scene description concise.

### 5. Use Positive Descriptions

FLUX does not support negative prompts. Describe the desired replacement instead of focusing attention on an unwanted element.

**Wrong**: `A portrait, no glasses, no hat.`

**Right**: `A portrait with a clear, unobstructed gaze and visible wind-swept hair.`

Useful replacements include:

| Avoid | Prefer |
|---|---|
| no people | empty, deserted, solitary |
| no bright colors | muted earth tones, subdued palette |
| no blur | tack-sharp focus, crisp detail |
| no clutter | clean, minimal composition |
| no CGI look | photorealistic, natural, organic |

Provider-level safety rails may still require fixed exclusion phrases. Keep those separate from the positive creative description so they do not dominate the scene.

## HonCut Shot Prompt Structure

Use this order:

1. **Identity lock** — character name, reference, and stable appearance
2. **Shot and framing** — composition, size, and camera angle
3. **Camera movement** — one clear motion declaration
4. **Action** — observable behavior or change
5. **Setting** — environment and spatial context
6. **Lighting** — source, direction, color, and mood
7. **Style and technical detail** — medium, lens language, palette, quality
8. **Audio** — ambient sound or quoted dialogue when the provider supports it

This is a video-oriented adaptation of the FLUX formula: Subject → Action → Style → Context → Lighting → Technical. HonCut moves identity and camera constraints forward because cross-shot continuity and motion are production requirements.

## Specificity Checklist

Before accepting a shot prompt, confirm that it answers:

- Who or what is the focal subject?
- What visible action occurs during the shot?
- Where is the subject, and what spatial relationship matters?
- What is the shot size and camera behavior?
- What is the light source, quality, and color?
- Which appearance details must remain constant?
- Which style or technical details materially affect the result?

Prefer concrete details such as `flowing auburn hair`, `cream linen dress`, `85mm portrait perspective`, or `cool window light from camera left` over vague praise such as `beautiful`, `epic`, or `high quality`.

## Examples

### Good Shot Prompt

```text
[identity_lock] Lin Xiao, a 25-year-old woman with flowing auburn hair and
emerald-green eyes, wearing a cream linen dress; maintain exact appearance
from the reference image. Wide establishing shot with a slow cinematic
push-in. Lin Xiao walks through a rose garden and pauses beside one bloom.
Soft diffused moonlight creates a subtle rim light, with shallow depth of
field and a warm cinematic film palette. Ambient crickets and soft footsteps.
```

The identity is front-loaded, the action is observable, the camera instruction is singular, and the lighting is explicit. At roughly 70 words, it remains focused.

### Bad Shot Prompt

```text
beautiful woman, garden, night, walking, cinematic, 8k, detailed
```

It uses disconnected keywords, provides no identity anchor, leaves lighting and camera behavior undefined, and relies on vague quality terms.

## Iteration Strategy

1. Start with subject, identity, and action.
2. Add framing and one camera movement.
3. Specify lighting and atmosphere.
4. Add only the style and technical details that change the result.
5. Generate and evaluate.
6. Change one variable at a time so the effect is measurable.

For dialogue-driven Seedance shots, quote the exact spoken line and keep visible action compatible with the available duration. For landscape-only shots, spend more of the prompt budget on composition, depth, weather, and light.

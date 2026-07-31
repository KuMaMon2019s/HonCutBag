# HonCut AI Video Pipeline — PIPELINE Master Document

> **Version**: v1.0.0  
> **Updated**: 2026-07-28  
> **Status**: Architecture definition phase

---

## 1. Data Flow Overview (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     HonCut AI Video Pipeline · 9-Phase Architecture          │
└─────────────────────────────────────────────────────────────────────────────┘

                          ┌──────────────────┐
                          │  Arbitrary Text   │
                          │ (sentence/outline │
                          │   /long-form)     │
                          └────────┬─────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 1: Director Planning  │
                    │  (M1 - scene/emotion/trans.) │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 2: Screenwriter Engine│
                    │  text → structured narrative │
                    └──────┬───────────────┬──────┘
                           │               │
              STORYBOARD.json      CHARACTERS.json
                           │               │
                           │    ┌──────────▼──────────┐
                           │    │  Phase 3: Character  │
                           │    │  Assets (3-views +   │
                           │    │  character cards)    │
                           │    └──────────┬──────────┘
                           │               │
                           │    characters/*/front|side|back.png
                           │    character_card.json / angle_map.json
                           │               │
                    ┌──────▼───────────────▼──────┐
                    │  Phase 4: Smart Routing      │
                    │  (M4 - model-based routing)  │
                    └──────────────┬──────────────┘
                                   │
                    shots/S*/SHOT_META.json
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 5: Video Generation   │
                    │  Seedance async generation   │
                    └──────────────┬──────────────┘
                                   │
                    shots/S*/output.mp4 + frames/
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 6: Quality Gate       │
                    │  (M5 - supervision + grading)│
                    └──┬─────────────────────┬────┘
                       │                     │
                  ✅ ≥0.7              🔴 <0.7
                       │                     │
                       │              ┌──────▼──────┐
                       │              │ Back to P5   │
                       │              │ re-gen       │
                       │              └─────────────┘
                       │
                    ┌──▼───────────────────────────┐
                    │  Phase 7: Rough Assembly      │
                    │  (M3 - transition bridges)    │
                    └──────────────┬───────────────┘
                                   │
                          stitched.mp4
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 8: Post-Production    │
                    │  Audio/quality/rhythm/       │
                    │  transitions                 │
                    └──────────────┬──────────────┘
                                   │
                          polished.mp4
                                   │
                    ┌──────────────▼──────────────┐
                    │  Phase 9: Full Integration   │
                    │  Go backend + n8n trigger    │
                    │  + E2E validation            │
                    └─────────────────────────────┘
```

---

## 2. Phase Dependencies

```
Phase 1 (Director Planning)
  └── Depended on by all phases (runtime foundation + planning input)

Phase 2 (Screenwriter) ──depends──▶ Phase 1
  ├── outputs STORYBOARD.json ──▶ Phase 4
  └── outputs CHARACTERS.json ──▶ Phase 3

Phase 3 (Character Assets) ──depends──▶ Phase 2 (CHARACTERS.json)
  └── outputs character assets ──▶ Phase 4

Phase 4 (Smart Routing) ──depends──▶ Phase 2 + Phase 3
  └── outputs SHOT_META.json ──▶ Phase 5

Phase 5 (Video Generation) ──depends──▶ Phase 4
  ├── outputs output.mp4 ──▶ Phase 6, Phase 7
  └── outputs frames/ ──▶ Phase 6

Phase 6 (Quality Gate) ──depends──▶ Phase 5
  ├── ✅ Pass ──▶ Phase 7
  └── 🔴 Fail ──▶ Phase 5 (closed-loop retry)

Phase 7 (Rough Assembly) ──depends──▶ Phase 5 (all output.mp4)
  └── outputs stitched.mp4 ──▶ Phase 8

Phase 8 (Post-Production) ──depends──▶ Phase 7
  └── outputs polished.mp4 ──▶ Phase 9 / delivery

Phase 9 (Full Integration) ──depends──▶ Phases 1-8
  └── End-to-end orchestration
```

---

## 3. Phase Definitions

### Phase 1: Director Planning ✅

> **One-liner**: Produce structured director planning — scene breakdown, emotion arcs, and transition design — before storyboarding begins.

| Item | Description |
|------|-------------|
| **Input** | Script text (segments from text_parser), optional character list |
| **Output** | `director_plan.json` (scenes, emotion arcs, scene transitions) |
| **Tool** | `director_planner.py` (LLM call: doubao-seed-2.0-lite) |
| **Status** | ✅ Implemented (M1 module) |

**Key behaviors**:
- Splits script into scenes by location/time/dramatic unit boundaries
- Assigns emotion intensity (0–10) and emotion arc (X→Y) per scene
- Designs inter-scene transitions using 4 bridge types
- Outputs pure structured JSON, no creative prose

---

### Phase 2: Screenwriter Engine ❌

> **One-liner**: Parse arbitrary text into a structured event graph and character list.

| Item | Description |
|------|-------------|
| **Input** | Arbitrary text (sentence / paragraph / outline / long-form) |
| **Output** | `STORYBOARD.json` + `CHARACTERS.json` |
| **Toolchain** | `text_parser.py` → `event_extractor.py` → `character_discoverer.py` → `adaptation_engine.py` → `storyboard_generator.py` |
| **Status** | ❌ Not implemented |

**Toolchain details**:

| Tool | Responsibility |
|------|---------------|
| `text_parser.py` | Auto-detect input scale: short text → direct extraction; long text → split by paragraph/chapter |
| `event_extractor.py` | Extract events (who/what/where/when/why) from text blocks |
| `character_discoverer.py` | Discover characters from events and extract attributes |
| `adaptation_engine.py` | Adapt raw events into visualizable shot language |
| `storyboard_generator.py` | Generate final storyboard script |

**Key constraints**:
- Input format is unrestricted; text_parser handles adaptation
- Output must strictly conform to JSON Schema below

---

### Phase 2.5: Storyboard Sequence (M2)

> **One-liner**: Generate one storyboard image per shot for visual composition reference.

| Item | Description |
|------|-------------|
| **Input** | `STORYBOARD.json` shot prompts |
| **Output** | `storyboard_images/S01.png`, `S02.png`, ... |
| **Tool** | `seedream_client.py` (extended per-shot calls) |
| **Status** | ✅ Implemented (M2 module) |

---

### Phase 3: Character Assets ✅ (needs enhancement)

> **One-liner**: Generate three-view images and character cards for each character.

| Item | Description |
|------|-------------|
| **Input** | `CHARACTERS.json` (from Phase 2) |
| **Output** | `characters/{name}/front.png` · `side.png` · `back.png` · `character_card.json` · `angle_map.json` |
| **Tool** | `character_factory.py` (calls Seedream 5.0-lite `/images/generations`) |
| **Status** | ✅ Implemented, needs enhancement |

**Enhancement directions**:
- Reference ComfyUI three-view workflow prompt templates and negative prompts
- Convert IPAdapter / ControlNet concepts to Seedream API parameters (e.g., reference_image, control_strength)
- Unify character consistency (appearance anchoring across shots)

**Output directory structure**:
```
characters/
├── {character_name}/
│   ├── front.png          # Front view
│   ├── side.png           # Side view
│   ├── back.png           # Back view
│   ├── character_card.json # Character attribute card
│   └── angle_map.json     # Angle mapping table
```

---

### Phase 4: Smart Routing ✅ (M4)

> **One-liner**: Select the optimal generation tool and prompt format for each shot.

| Item | Description |
|------|-------------|
| **Input** | `STORYBOARD.json` + character three-view paths |
| **Output** | `shots/S{N}/SHOT_META.json` (with routing decision) |
| **Tool** | `orchestrator.py` + `tool_router.py` + `prompt_router.py` (M4) |
| **Status** | ✅ Implemented |

**Routing dimensions**:
- Shot type (wide / medium / close-up / motion)
- Character count (single / multi / none)
- Action complexity (static / dynamic / interactive)
- Emotional tone (determines style parameters)

**M4 Model Routing** (`prompt_router.py`):

| Mode | Trigger | Prompt Format |
|------|---------|---------------|
| Seedance 2.0 multi-shot | model=seedance-2.0 + multi-shot | Chinese structured 12-dim encoding + @image refs + ms durations |
| Seedance 2.0 single-shot | model=seedance-2.0 + single-shot | reference_image + English prompt |
| Generic first/last-frame | Other models + first/last-frame | [Visual][Motion][Camera][Audio][Narrative] 5 dimensions |
| Wan 2.6 | model=wan2.6 | Narrative English (style → subject → lighting → camera) |

**Output directory structure**:
```
shots/
├── S1/
│   └── SHOT_META.json
├── S2/
│   └── SHOT_META.json
└── ...
```

---

### Phase 5: Video Generation ✅

> **One-liner**: Call Seedance API to asynchronously generate video clips for each shot.

| Item | Description |
|------|-------------|
| **Input** | `shots/S{N}/SHOT_META.json` |
| **Output** | `shots/S{N}/output.mp4` + `shots/S{N}/frames/` |
| **Tool** | `seedance_client.py` (Agent Plan async submit/poll) |
| **Status** | ✅ Implemented |

**API constraints**:
- Endpoint: `/api/plan/v3/` (Agent Plan)
- Model: `doubao-seedance-2.0-fast`
- All parameters at top level, `watermark: false`
- Async mode: submit → poll → download

---

### Phase 6: Quality Gate ✅ (M5)

> **One-liner**: Check character consistency and visual quality; reject and re-generate if below threshold.

| Item | Description |
|------|-------------|
| **Input** | `shots/S{N}/frames/` (sampled frames) |
| **Output** | `consistency_report.json` |
| **Tool** | `consistency_guard.py` (embedding comparison) + `quality_gate.py` (`run_storyboard_review` — M5) |
| **Status** | ✅ Implemented, needs enhancement |

**M5 Supervision Layer** (`run_storyboard_review`):

4 red-line checks:
- R1: Asset references valid (characters/scenes must exist)
- R2: Script fidelity (dialogue verbatim, no omissions or additions)
- R3: Concreteness (no abstract/vague descriptions)
- R4: Parent-child assets correct (derived states use derived IDs)

Grading: A (0 critical, ≤2 moderate) / B (0 critical, ≤5 moderate) / C (1–2 critical) / D (≥3 critical)

**Enhancement directions**:
- Replace hand-written ffmpeg frame extraction with OM `frame_sampler`
- Add `composition_validator` (composition checking)
- Add `face_tracker` (face tracking consistency)

**Closed-loop mechanism**:
- Consistency score ≥ 0.7 → ✅ Pass, proceed to Phase 7
- Consistency score < 0.7 → 🔴 Fail, return to Phase 5 re-gen (max 3 retries)

---

### Phase 7: Rough Assembly ✅ (M3)

> **One-liner**: Stitch all shot clips into a rough-cut video with transition bridges.

| Item | Description |
|------|-------------|
| **Input** | All `shots/S{N}/output.mp4` |
| **Output** | `stitched.mp4` (rough cut) |
| **Tool** | `assembly_engine.py` (OM `video_stitch` + `remotion_caption_burn`) |
| **Status** | ✅ Implemented, needs enhancement |

**M3 Transition Bridges** (4 types injected into adaptation prompts):

| Bridge | Trigger | Method |
|--------|---------|--------|
| Action bridge | Same characters, continuous action | End of previous = action start state; first shot of next = in-progress/completed |
| Emotion relay | Dialogue/conflict emotion continues | End of previous uses reaction shot / micro-expression; next inherits and amplifies |
| Spatial / gaze | Scene change / gaze shift | Empty shot + gaze guidance + sound continuation |
| Dialogue glue | Dialogue/SFX needs visual response | Sound from end of previous carries into first shot of next |

**Enhancement directions**:
- OM `video_trimmer`: Remove dead frames (black/blurry frames at start/end of each shot)
- OM `silence_cutter`: Remove silent segments (if audio track present)

---

### Phase 8: Post-Production ❌

> **One-liner**: Fine-tune rough-cut video with audio, visual quality, rhythm, and transitions to produce final deliverable.

| Item | Description |
|------|-------------|
| **Input** | `stitched.mp4` |
| **Output** | `polished.mp4` (final deliverable) |
| **Tool** | All from OpenMontage (see table below) |
| **Status** | ❌ Not implemented |

**Tool list (all from OM)**:

| Category | Tool | Responsibility |
|----------|------|---------------|
| **Audio** | `audio_mixer` | Multi-track mixing (BGM + SFX + dialogue) |
| | `music_gen` | AI-generated background music |
| | `doubao_tts` | Doubao TTS for narration/dialogue |
| | `audio_enhance` | Audio noise reduction/enhancement |
| **Quality** | `enhancement/` | Super-resolution / denoising |
| **Aspect** | `auto_reframe` | Auto-crop for different ratios (16:9 / 9:16 / 1:1) |
| **Graphics** | `graphics/` | Title cards / end cards / subtitle cards |
| **Rhythm** | Speed ramp / beat-sync | Adjust shot speed to music rhythm |
| **Transitions** | Transition refinement | Select transition type by emotion (cut/dissolve/wipe/zoom) |

---

### Phase 9: Full Integration ✅

> **One-liner**: Orchestrate all phases via Go backend with n8n trigger and end-to-end validation.

| Item | Description |
|------|-------------|
| **Input** | User request (text + parameters) |
| **Output** | `polished.mp4` (final deliverable) |
| **Tool** | Go backend + n8n webhook trigger + E2E tests |
| **Status** | ✅ Implemented |

**Integration architecture**:
```
User request → n8n webhook → Go backend → Phase 1-8 sequential execution → deliver polished.mp4
```

---

## 4. Key JSON Interface Specifications

### 4.1 CHARACTERS.json Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CHARACTERS",
  "type": "object",
  "required": ["version", "characters"],
  "properties": {
    "version": { "type": "string", "const": "1.0" },
    "source_text_hash": { "type": "string", "description": "SHA-256 of input text for traceability" },
    "characters": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "name", "appearance", "role"],
        "properties": {
          "id": { "type": "string", "description": "Unique character ID, e.g. char_001" },
          "name": { "type": "string", "description": "Character name" },
          "role": { "type": "string", "enum": ["protagonist", "antagonist", "supporting", "extra"], "description": "Character role" },
          "appearance": {
            "type": "object",
            "required": ["gender", "age_range", "summary"],
            "properties": {
              "gender": { "type": "string", "enum": ["male", "female", "nonbinary", "unknown"] },
              "age_range": { "type": "string", "description": "e.g. '20-30'" },
              "height": { "type": "string", "description": "e.g. '175cm'" },
              "build": { "type": "string", "description": "Body type, e.g. 'slim', 'athletic', 'heavy'" },
              "hair": { "type": "string", "description": "Hair style and color" },
              "face": { "type": "string", "description": "Facial features" },
              "clothing": { "type": "string", "description": "Typical attire" },
              "distinguishing": { "type": "string", "description": "Distinguishing marks" },
              "summary": { "type": "string", "description": "One-line appearance summary for prompt generation" }
            }
          },
          "personality": {
            "type": "object",
            "properties": {
              "traits": { "type": "array", "items": { "type": "string" } },
              "speech_style": { "type": "string" },
              "motivation": { "type": "string" }
            }
          },
          "relationships": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "target_id": { "type": "string" },
                "type": { "type": "string", "description": "e.g. 'friend', 'rival', 'mentor'" },
                "description": { "type": "string" }
              }
            }
          },
          "asset_path": { "type": "string", "description": "Character asset directory, e.g. 'characters/char_001/'" }
        }
      }
    }
  }
}
```

---

### 4.2 STORYBOARD.json Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "STORYBOARD",
  "type": "object",
  "required": ["version", "title", "shots"],
  "properties": {
    "version": { "type": "string", "const": "1.0" },
    "title": { "type": "string", "description": "Work title" },
    "genre": { "type": "string", "description": "Genre, e.g. 'sci-fi', 'romance', 'thriller'" },
    "tone": { "type": "string", "description": "Overall tone, e.g. 'dark', 'uplifting', 'suspenseful'" },
    "total_duration_target": { "type": "number", "description": "Target total duration (seconds)" },
    "synopsis": { "type": "string", "description": "Story synopsis (3-5 sentences)" },
    "shots": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["shot_id", "sequence", "description", "characters"],
        "properties": {
          "shot_id": { "type": "string", "description": "Shot ID, e.g. 'S1', 'S2'" },
          "sequence": { "type": "integer", "description": "Shot sequence number (starting from 1)" },
          "description": { "type": "string", "description": "Shot content description (natural language)" },
          "scene": { "type": "string", "description": "Scene/location" },
          "time_of_day": { "type": "string", "enum": ["dawn", "morning", "noon", "afternoon", "dusk", "night"] },
          "camera": {
            "type": "object",
            "properties": {
              "shot_type": { "type": "string", "enum": ["wide", "medium", "close-up", "extreme-close-up", "over-shoulder", "pov", "aerial"] },
              "angle": { "type": "string", "enum": ["eye-level", "low-angle", "high-angle", "dutch", "birds-eye"] },
              "movement": { "type": "string", "enum": ["static", "pan-left", "pan-right", "tilt-up", "tilt-down", "dolly-in", "dolly-out", "tracking", "crane", "handheld"] },
              "lens": { "type": "string", "description": "e.g. '50mm', 'wide-angle', 'telephoto'" }
            }
          },
          "characters": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "character_id": { "type": "string" },
                "action": { "type": "string", "description": "Character action in this shot" },
                "emotion": { "type": "string", "description": "Emotional state" },
                "position": { "type": "string", "description": "Position in frame, e.g. 'left', 'center', 'right'" }
              }
            }
          },
          "dialogue": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "character_id": { "type": "string" },
                "line": { "type": "string" },
                "delivery": { "type": "string", "description": "Delivery style, e.g. 'whisper', 'shout', 'calm'" }
              }
            }
          },
          "mood": { "type": "string", "description": "Shot mood, e.g. 'tense', 'peaceful', 'joyful'" },
          "duration_hint": { "type": "number", "description": "Suggested duration (seconds)" },
          "transition_to_next": { "type": "string", "enum": ["cut", "dissolve", "fade", "wipe", "match-cut"], "description": "Transition to next shot" },
          "notes": { "type": "string", "description": "Director notes / special requirements" }
        }
      }
    }
  }
}
```

---

### 4.3 SHOT_META.json Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SHOT_META",
  "type": "object",
  "required": ["shot_id", "route", "prompt", "parameters"],
  "properties": {
    "shot_id": { "type": "string", "description": "Shot ID, e.g. 'S1'" },
    "source_storyboard": { "type": "string", "description": "Source shot_id from STORYBOARD.json" },
    "route": {
      "type": "object",
      "required": ["tool", "model"],
      "properties": {
        "tool": { "type": "string", "description": "Tool used, e.g. 'seedance_client'" },
        "model": { "type": "string", "description": "Model name, e.g. 'doubao-seedance-2.0-fast'" },
        "reason": { "type": "string", "description": "Routing decision reason" },
        "fallback": { "type": "string", "description": "Fallback tool/model" }
      }
    },
    "prompt": {
      "type": "object",
      "required": ["text"],
      "properties": {
        "text": { "type": "string", "description": "Final prompt sent to generation model" },
        "negative_prompt": { "type": "string", "description": "Negative prompt" },
        "style_prefix": { "type": "string", "description": "Style prefix, e.g. 'cinematic, 4K'" }
      }
    },
    "parameters": {
      "type": "object",
      "properties": {
        "duration": { "type": "number", "description": "Video duration (seconds)" },
        "resolution": { "type": "string", "description": "e.g. '1280x720', '720x1280'" },
        "fps": { "type": "integer", "description": "Frame rate" },
        "aspect_ratio": { "type": "string", "enum": ["16:9", "9:16", "1:1", "4:3"] },
        "seed": { "type": "integer", "description": "Random seed (reproducible)" },
        "cfg_scale": { "type": "number", "description": "Guidance strength" },
        "watermark": { "type": "boolean", "const": false }
      }
    },
    "reference_images": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "character_id": { "type": "string" },
          "image_path": { "type": "string", "description": "Reference image path" },
          "role": { "type": "string", "enum": ["face_ref", "pose_ref", "style_ref", "full_body_ref"] },
          "weight": { "type": "number", "description": "Reference weight 0.0-1.0" }
        }
      }
    },
    "status": { "type": "string", "enum": ["pending", "generating", "completed", "failed", "retrying"] },
    "retry_count": { "type": "integer", "default": 0 },
    "output_path": { "type": "string", "description": "Generation result path" },
    "task_id": { "type": "string", "description": "Agent Plan async task ID" }
  }
}
```

---

### 4.4 consistency_report.json Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CONSISTENCY_REPORT",
  "type": "object",
  "required": ["shot_id", "overall_score", "checks", "verdict"],
  "properties": {
    "shot_id": { "type": "string" },
    "timestamp": { "type": "string", "format": "date-time" },
    "overall_score": { "type": "number", "minimum": 0, "maximum": 1, "description": "Overall consistency score" },
    "verdict": { "type": "string", "enum": ["pass", "fail"], "description": "pass: ≥0.7, fail: <0.7" },
    "checks": {
      "type": "object",
      "properties": {
        "character_consistency": {
          "type": "object",
          "properties": {
            "score": { "type": "number", "minimum": 0, "maximum": 1 },
            "method": { "type": "string", "description": "e.g. 'embedding_cosine_similarity'" },
            "details": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "character_id": { "type": "string" },
                  "reference_image": { "type": "string" },
                  "frame_scores": {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "properties": {
                        "frame_index": { "type": "integer" },
                        "score": { "type": "number" },
                        "face_detected": { "type": "boolean" }
                      }
                    }
                  }
                }
              }
            }
          }
        },
        "composition": {
          "type": "object",
          "properties": {
            "score": { "type": "number" },
            "method": { "type": "string", "description": "e.g. 'rule_of_thirds_analysis'" },
            "issues": { "type": "array", "items": { "type": "string" } }
          }
        },
        "temporal_coherence": {
          "type": "object",
          "properties": {
            "score": { "type": "number" },
            "method": { "type": "string", "description": "e.g. 'frame_to_frame_embedding_delta'" },
            "max_delta": { "type": "number" },
            "avg_delta": { "type": "number" }
          }
        },
        "face_tracking": {
          "type": "object",
          "properties": {
            "score": { "type": "number" },
            "faces_tracked": { "type": "integer" },
            "tracking_loss_frames": { "type": "array", "items": { "type": "integer" } }
          }
        }
      }
    },
    "retry_recommendation": {
      "type": "object",
      "properties": {
        "should_retry": { "type": "boolean" },
        "retry_reason": { "type": "string" },
        "suggested_parameter_changes": {
          "type": "object",
          "description": "Suggested generation parameter adjustments"
        }
      }
    }
  }
}
```

---

## 5. Status Summary

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| 1 | Director Planning (M1) | ✅ Implemented | — |
| 2 | Screenwriter Engine | ❌ Not implemented | Core screenwriting module |
| 2.5 | Storyboard Sequence (M2) | ✅ Implemented | Per-shot storyboard images |
| 3 | Character Assets | ✅ Needs enhancement | ComfyUI prompt migration |
| 4 | Smart Routing (M4) | ✅ Implemented | Model-based prompt routing |
| 5 | Video Generation | ✅ Implemented | — |
| 6 | Quality Gate (M5) | ✅ Needs enhancement | OM frame_sampler replacement |
| 7 | Rough Assembly (M3) | ✅ Needs enhancement | OM video_trimmer + silence_cutter |
| 8 | Post-Production | ❌ Not implemented | Full OM post-production toolkit |
| 9 | Full Integration | ✅ Implemented | — |

**Implementation progress**: 8/10 phases available (3 need enhancement), 2 not implemented (Phase 2, Phase 8).

---

## 6. Constraints and Conventions

### 6.1 API Constraints
- **Agent Plan Endpoint**: `/api/plan/v3/`
- **Seedream**: `doubao-seedream-5.0-lite`, sync interface `/images/generations`
- **Seedance**: `doubao-seedance-2.0-fast`, async interface `/contents/generations/tasks`
- **Seedance parameters**: All at top level, `watermark: false`

### 6.2 Hardware Constraints
- M4 Mac 16GB, no local model inference
- ComfyUI referenced for workflow/prompt design only, no local inference

### 6.3 Architecture Conventions
- Input is "arbitrary text", not specifically novels/chapters
- OCC (OpenChatCut) archived; OM (OpenMontage) is the core engine
- All phase outputs written to working directory for traceability and debugging

### 6.4 Directory Structure Convention
```
{project_root}/
├── characters/          # Phase 3 output
├── shots/               # Phase 4/5/6 output
│   ├── S1/
│   │   ├── SHOT_META.json
│   │   ├── output.mp4
│   │   └── frames/
│   └── ...
├── STORYBOARD.json      # Phase 2 output
├── CHARACTERS.json      # Phase 2 output
├── director_plan.json   # Phase 1 output (M1)
├── storyboard_images/   # Phase 2.5 output (M2)
├── consistency_report.json  # Phase 6 output
├── stitched.mp4         # Phase 7 output
└── polished.mp4         # Phase 8 output (final deliverable)
```

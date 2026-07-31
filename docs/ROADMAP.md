# HonCut Refactoring Roadmap (Integrating Toonflow + OpenMontage Best Practices)

> Generated: 2026-07-31  
> Principle: **Add only, don't delete; extend only, don't rewrite** — layer capabilities on top of existing 8 phases without restructuring or rewriting

---

## I. Existing Architecture (Unchanged)

```
Phase 2   Screenwriter Engine     text_parser → event_extractor → character_discoverer → adaptation_engine → storyboard_generator
Phase 2.5 Storyboard              seedream_client → storyboard.png
Phase 3   Character Factory       character_factory → 2×2 grid → crop → front/side/back.png
Phase 4   Orchestrator            orchestrator → shots/S01-S10/SHOT_META.json
Phase 5   Video Generation        seedance_client (TOS + reference_image + 429 backoff)
Phase 6   Consistency Guard       consistency_guard + quality_gate
Phase 7   Assembly Engine         edit_decisions (trim + normalize + smart_transition + xfade)
Phase 8   Post-Production         audio_pipeline → visual_post → rhythm_editor → final_encode
```

**All existing modules (25 files, ~9600 lines) are retained without any deletions or modifications.**

---

## II. Refactoring Overview (6 Incremental Modules)

| # | Module | Learn From | Insertion Point | Method |
|---|--------|-----------|-----------------|--------|
| M1 | Director Planning Layer | Toonflow Stage 1 | New Phase 1 before Phase 2 | Add `director_planner.py` |
| M2 | Storyboard Image Sequence | Toonflow Stage 6 | Extend Phase 2.5 | Extend `seedream_client` calls |
| M3 | Inter-Clip Transition Bridges | Toonflow storyboard table | Extend Phase 2 adaptation_engine | Extend LLM prompt |
| M4 | Model Routing | Toonflow video prompts | New routing layer before Phase 5 | Add `prompt_router.py` |
| M5 | Supervision Layer Review | Toonflow supervision layer | Extend Phase 6 | Extend `quality_gate.py` |
| M6 | Artifact Chain + Checkpoint | OpenMontage | Overlay on full pipeline | Add `artifact_chain.py` |

---

## III. M1: Director Planning Layer (Learn from Toonflow Stage 1)

### Goal
Add Phase 1 before Phase 2 to produce structured director planning, giving downstream storyboard phases emotional grounding and consistency anchors.

### New File
`pipeline/src/director_planner.py`

### Input
- Script text (segments from text_parser)
- Character list (optional, if available)

### Output
`director_plan.json`:
```json
{
  "scenes": [
    {
      "scene_id": "Sc1",
      "scene_name": "Convenience store entrance · heavy rain evening",
      "dialogue_count": 3,
      "dialogue_words": 86,
      "emotion_intensity": 4,
      "emotion_arc": "boredom → curiosity → attraction",
      "notes": {
        "emotional_peak": "Shen Yu appears with umbrella, eyes meet",
        "consistency_anchors": ["Lin Xia: white shirt + dark blue pants + black straight hair", "Shen Yu: black umbrella + rolled sleeves + mechanical watch"],
        "spatial": "Lin Xia stands left-front facing right, Shen Yu enters from right",
        "ambient_sound": "rain hitting glass, convenience store door bell",
        "pitfall": "Watch umbrella occlusion when both characters in frame"
      }
    }
  ],
  "scene_transitions": [
    {
      "from": "Sc1",
      "to": "Sc2",
      "type": "action bridge",
      "description": "Shen Yu tilts umbrella → two walk side-by-side into rain"
    }
  ]
}
```

### Implementation
- Call LLM (doubao-seed-2.0-lite), prompt follows Toonflow director planning
- Does only 4 things: scene splitting, dialogue counting, emotion analysis, transition design
- Does not plan lighting/color grading/music
- Outputs pure structured JSON, no creative prose

### Relationship to Existing Code
- **Does not modify** Phase 2 logic in `pipeline_runner.py`
- Inserts `run_phase1()` call before Phase 2 in `run_pipeline()`
- `director_plan.json` passed as additional input to Phase 2 adaptation_engine

### Toonflow Original Best Practices (Direct Reuse)

**Scene Splitting Rules:**
- One scene = one continuous dramatic unit in same time-space
- Cut points: location change / time jump / dramatic unit closure
- If script has scene markers → preserve original fidelity

**Emotion Analysis:**
- Assign emotion intensity 0~10 + one-line emotional tone per scene
- Mark intra-scene emotion progression as X→Y

**4 Types of Inter-Scene Transition Bridges:**

| Bridge | Trigger | Method |
|--------|---------|--------|
| Action bridge | Same characters, continuous action | End of previous = action start state; first shot of next = in-progress/completed |
| Emotion relay | Dialogue/conflict emotion continues | End of previous uses reaction shot / micro-expression; next inherits and amplifies |
| Spatial / gaze | Scene change / gaze shift | Empty shot + gaze guidance + sound continuation |
| Dialogue glue | Dialogue/SFX needs visual response | Sound from end of previous carries into first shot of next |

---

## IV. M2: Storyboard Image Sequence (Learn from Toonflow Stage 6)

### Goal
Extend Phase 2.5 from generating 1 storyboard.png to one storyboard image per shot.

### Modification Method
**Does not modify** `seedream_client.py`, extends calls in `pipeline_runner.py` `run_phase2_5()`:

```python
# Current: generate 1 image
seedream_client.text_to_image(prompt=storyboard_prompt, output_path="storyboard.png")

# Extended: one per shot (appended after existing call)
for shot in storyboard_data["shots"]:
    shot_prompt = shot["prompt"]  # each shot already has independent prompt
    shot_image_path = output_dir / "storyboard_images" / f"{shot['shot_id']}.png"
    seedream_client.text_to_image(prompt=shot_prompt, output_path=str(shot_image_path))
```

### Output
```
output/
  storyboard.png          ← retained (overview)
  storyboard_images/
    S01.png               ← new (one per shot)
    S02.png
    ...
    S10.png
```

### Integration with Phase 5
During Phase 5 video generation, corresponding storyboard image can serve as composition reference (reference_image):
```python
# In _run_phase5_fallback, appended after existing logic
shot_image = output_dir / "storyboard_images" / f"{shot_id}.png"
if shot_image.exists():
    # Use storyboard image as composition reference (lower priority than character reference)
    if first_frame_b64 is None:
        first_frame_b64 = base64.b64encode(shot_image.read_bytes()).decode()
```

---

## V. M3: Inter-Clip Transition Bridges (Learn from Toonflow Storyboard Table)

### Goal
Add inter-clip transition design rules to adaptation_engine LLM prompt.

### Modification Method
**Does not modify** `adaptation_engine.py` code logic, only extends LLM prompt content:

Append transition rules to `USER_PROMPT_TEMPLATE` (from Toonflow original):

```
[Inter-Clip Transition Rules]
Adjacent clips must design transition bridges to eliminate jump-cut feel:
1. Action bridge: end of previous = action start state; first shot of next = in-progress/completed
2. Emotion relay: end of previous uses reaction shot / micro-expression; next inherits and amplifies
3. Spatial / gaze: use empty shot + gaze guidance + sound continuation during scene changes
4. Dialogue glue: sound from end of previous carries into first shot of next

Each shot's visual description must begin with "continuation from previous shot" paragraph (except first shot).
```

### Relationship to Existing Code
- Existing `adaptation_engine.py` already has continuation rules (line 70), only need to extend prompt content
- **Does not modify code logic**, only modifies prompt string

---

## VI. M4: Model Routing (Learn from Toonflow 4 Prompt Modes)

### Goal
Add prompt routing layer before Phase 5, auto-match prompt format by model name.

### New File
`pipeline/src/prompt_router.py`

### 4 Modes (from Toonflow original)

| Mode | Trigger | Prompt Format |
|------|---------|---------------|
| Seedance 2.0 multi-shot | model=seedance-2.0 + multi-shot | Chinese structured 12-dim encoding + @imageN references + ms durations |
| Seedance 2.0 single-shot | model=seedance-2.0 + single-shot | reference_image + English prompt |
| Generic first/last-frame | Other models + first/last-frame | [Visual][Motion][Camera][Audio][Narrative] 5 dimensions |
| Wan 2.6 | model=wan2.6 | Narrative English (style → subject → lighting → camera) |

### Routing Logic
```python
def route_prompt(model_name: str, mode: str, shot_data: dict, assets: list) -> str:
    model_lower = model_name.lower()
    
    if "seedance" in model_lower and "2" in model_lower:
        if mode == "multi_shot":
            return _build_seedance2_multi(shot_data, assets)
        else:
            return _build_seedance2_single(shot_data, assets)
    elif "wan" in model_lower and "2.6" in model_lower:
        return _build_wan26_narrative(shot_data)
    else:
        return _build_generic_first_last_frame(shot_data)
```

### Seedance 2.0 Multi-Shot Format (from Toonflow original)
```
Visual style and type: realistic, cinematic, warm urban tone

Image definitions:
@Image1: Lin Xia, black straight hair to shoulders, white fitted shirt + dark blue suit pants
@Image2: Shen Yu, short clean hair, casual business attire, holding black umbrella

Generate a video composed of the following N shots:

Shot 1 6s: Time: evening, Scene: convenience store entrance, Camera: wide shot, static,
  Lin Xia stands at convenience store entrance, watching heavy rain, frowning...

Shot 2 6s: Time: evening, Scene: convenience store entrance, Camera: medium shot,
  Shen Yu appears with black umbrella, "Together?"...
```

### Relationship to Existing Code
- **Does not modify** `seedance_client.py`
- In `pipeline_runner.py` `_run_phase5_fallback()`, call `prompt_router.route_prompt()` before building prompt
- Existing style_prefix and reference_image injection logic retained

---

## VII. M5: Supervision Layer Review (Learn from Toonflow Supervision Layer)

### Goal
Extend `quality_gate.py` to add Toonflow's 4 red-line reviews and A/B/C/D grading.

### Modification Method
**Does not modify** existing `quality_gate.py` `run_quality_check()`, adds new function:

```python
# New function, does not modify existing code
def run_storyboard_review(storyboard_data: dict, script_text: str, characters: list) -> dict:
    """Learn from Toonflow supervision layer: review storyboard quality.
    
    4 red lines (violation = serious):
    R1: Asset references valid (characters/scenes must exist in characters)
    R2: Script fidelity (dialogue verbatim, no omissions or additions)
    R3: Concreteness (no abstract/vague descriptions)
    R4: Parent-child assets correct (derived states use derived IDs)
    
    Grading: A(0 critical ≤2 moderate) / B(0 critical ≤5 moderate) / C(1-2 critical) / D(≥3 critical)
    """
```

### Relationship to Existing Code
- Existing `run_quality_check()` checks file existence (phase-level)
- New `run_storyboard_review()` checks content quality (storyboard-level)
- Two functions complement, don't conflict
- Call `run_storyboard_review()` at end of `run_phase2()`

---

## VIII. M6: Artifact Chain + Checkpoint (Learn from OpenMontage)

### Goal
Each phase produces structured JSON artifacts, supports recovery from any phase.

### New File
`pipeline/src/artifact_chain.py`

### Artifact Chain Definition
```python
ARTIFACT_CHAIN = {
    "phase1":  {"produces": "director_plan.json",     "requires": []},
    "phase2":  {"produces": "events.json + characters.json + storyboard.json", "requires": ["director_plan.json"]},
    "phase2_5": {"produces": "storyboard_images/",     "requires": ["storyboard.json"]},
    "phase3":  {"produces": "characters/",             "requires": ["characters.json"]},
    "phase4":  {"produces": "shots/",                  "requires": ["storyboard.json"]},
    "phase5":  {"produces": "shots/*/output.mp4",      "requires": ["shots/"]},
    "phase6":  {"produces": "quality_report.json",     "requires": ["shots/*/output.mp4"]},
    "phase7":  {"produces": "edit_decisions.json + raw_assembly.mp4", "requires": ["shots/*/output.mp4"]},
    "phase8":  {"produces": "polished.mp4 + render_report.json", "requires": ["raw_assembly.mp4"]},
}
```

### Checkpoint Enhancement
```python
def save_checkpoint(phase: str, output_dir: Path, artifacts: dict):
    """Write checkpoint after each phase."""
    checkpoint = {
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "artifacts": artifacts,
        "status": "done",
    }
    checkpoint_path = output_dir / f"checkpoint_{phase}.json"
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False))

def can_resume_from(phase: str, output_dir: Path) -> bool:
    """Check if can resume from specified phase."""
    required = ARTIFACT_CHAIN[phase]["requires"]
    for artifact in required:
        if not (output_dir / artifact).exists():
            return False
    return True
```

### Relationship to Existing Code
- Existing LangGraph checkpoint retained
- New file-level checkpoint (one JSON per phase)
- Call `save_checkpoint()` after each phase in `run_pipeline()`
- Support `--resume-from phase5` to recover from any phase

---

## IX. Implementation Sequence

| Batch | Module | Dependencies | Estimate |
|-------|--------|--------------|----------|
| Batch 1 | M2 Storyboard Sequence | None | Low (extend Phase 2.5 calls) |
| Batch 1 | M3 Transition Bridges | None | Low (extend prompt string) |
| Batch 2 | M1 Director Planning | None | Medium (new director_planner.py) |
| Batch 2 | M5 Supervision Review | None | Medium (extend quality_gate.py) |
| Batch 3 | M4 Model Routing | M2 | High (new prompt_router.py) |
| Batch 3 | M6 Artifact Chain | M1 | Medium (new artifact_chain.py) |

### Iron Rules
1. **Do not delete any existing code**
2. **Do not rewrite any existing functions**
3. **Only add new files or append logic at end of existing functions**
4. **New modules degrade gracefully via try/except, failures don't affect existing flow**
5. **After each batch, full re-run validation to confirm no breakage**

---

## X. Reference Sources

| Source | File | What We Learn |
|--------|------|---------------|
| Toonflow director planning | `data/skills/production_execution_director_plan.md` | Scene splitting/emotion/transition/pitfalls |
| Toonflow storyboard table | `data/skills/production_execution_storyboard_table.md` | Iron rules/shot splitting/transition bridges |
| Toonflow supervision layer | `data/skills/production_agent_supervision.md` | 4 red lines/A-D grading/review dimensions |
| Toonflow video prompts | `src/lib/fixDB.ts` (videoPromptGeneration) | 4-mode routing/Seedance 2.0 format |
| Toonflow storyboard generation | `src/routes/production/storyboard/batchGenerateImage.ts` | One per shot/concurrent batching |
| OpenMontage cinematic | `pipeline_defs/cinematic.yaml` | 8-phase artifact chain/checkpoint/approval gates |
| OpenMontage VideoCompose | `tools/video/video_compose.py` | edit_decisions/frame-precise/normalization/multi-engine |
| OpenMontage pipeline_loader | `lib/pipeline_loader.py` | Phase ordering/sub-phases/conditional activation |

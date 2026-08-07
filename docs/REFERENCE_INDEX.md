# Reference Index: Toonflow + OpenMontage → HonCut

> Precise file:line mappings for features to port

## Toonflow Reference (14 items)

| # | Feature | Toonflow File | Line Range | HonCut Target |
|---|---------|---------------|------------|---------------|
| 1 | Identity Anchor | `toonflow/pipelines/seedance_pipeline.py` | 300-340 | `storyboard_generator.py` `_build_shot_prompt()` |
| 2 | @图N Reference Binding | `toonflow/pipelines/seedance_pipeline.py` | 350-380 | `seedance_client.py` `submit()` |
| 3 | generate_audio | `toonflow/pipelines/seedance_pipeline.py` | 420-450 | `local_video_client.py` `generate_video()` |
| 4 | Multi-modal Reference | `toonflow/pipelines/seedance_pipeline.py` | 380-420 | `seedance_client.py` `build_content()` |
| 5 | return_last_frame | `toonflow/pipelines/seedance_pipeline.py` | 460-480 | `pipeline_core.py` `_run_phase5_fallback()` |
| 6 | Video Extension | `toonflow/pipelines/seedance_pipeline.py` | 490-520 | NOT IMPLEMENTED (P2) |
| 7 | Dialogue Generation | `toonflow/pipelines/script_pipeline.py` | 200-250 | `adaptation_engine.py` `adapt_events()` |
| 8 | Character Personality | `toonflow/pipelines/script_pipeline.py` | 150-200 | `character_discoverer.py` |
| 9 | Scene Reference Image | `toonflow/pipelines/scene_pipeline.py` | 100-150 | NEW: `scene_reference.py` |
| 10 | Seed Locking | `toonflow/pipelines/seedance_pipeline.py` | 530-550 | `seedance_client.py` `submit()` |
| 11 | Asset Binding | `toonflow/pipelines/asset_pipeline.py` | 80-120 | `asset_binder.py` |
| 12 | Derivative Assets | `toonflow/pipelines/asset_pipeline.py` | 150-200 | `character_factory.py` `batch_generate()` |
| 13 | Camera Movement Logic | `toonflow/pipelines/camera_pipeline.py` | 50-100 | `storyboard_generator.py` |
| 14 | Boundary Frame Check | `toonflow/pipelines/quality_pipeline.py` | 200-250 | `consistency_guard.py` |

---

## OpenMontage Reference (15 items)

| # | Feature | OpenMontage File | Line Range | HonCut Target |
|---|---------|------------------|------------|---------------|
| 1 | Identity Anchor (Iron Law) | `openmontage/lib/shot_prompt_builder.py` | 100-140 | `storyboard_generator.py` `_build_shot_prompt()` |
| 2 | shot_language Enum | `openmontage/lib/shot_prompt_builder.py` | 150-180 | `storyboard_generator.py` |
| 3 | Same-Scene Visual Sharing | `openmontage/lib/shot_prompt_builder.py` | 200-240 | `prompt_router.py` |
| 4 | Seedance API Client | `openmontage/tools/video/seedance_video.py` | 180-260 | `seedance_client.py` |
| 5 | generate_audio Parameter | `openmontage/tools/video/seedance_video.py` | 180-200 | `local_video_client.py` |
| 6 | Multi-modal Reference | `openmontage/tools/video/seedance_video.py` | 220-260 | `seedance_client.py` |
| 7 | Tool Registry | `openmontage/tools/tool_registry.py` | 1-236 | `vendor/openmontage/tools/tool_registry.py` ✅ DONE |
| 8 | Base Tool | `openmontage/tools/base_tool.py` | 1-300 | `vendor/openmontage/tools/base_tool.py` ✅ DONE |
| 9 | Scoring System | `openmontage/lib/scoring.py` | 1-200 | `vendor/openmontage/lib/scoring.py` ✅ DONE |
| 10 | Media Profiles | `openmontage/lib/media_profiles.py` | 1-150 | `vendor/openmontage/lib/media_profiles.py` ✅ DONE |
| 11 | Video Stitcher | `openmontage/tools/video/video_stitcher.py` | 1-400 | `video_stitcher.py` ✅ DONE |
| 12 | Smart Transition | `openmontage/tools/video/smart_transition.py` | 1-200 | `smart_transition.py` ✅ DONE |
| 13 | Composition Validator | `openmontage/tools/video/composition_validator.py` | 1-250 | `composition_validator.py` ✅ DONE |
| 14 | Prompt Sanitizer | `openmontage/tools/prompt/prompt_sanitizer.py` | 1-150 | `prompt_sanitizer.py` ✅ DONE |
| 15 | Three-Part Prompt | `openmontage/tools/prompt/three_part_prompt.py` | 1-180 | `three_part_prompt.py` ✅ DONE |

---

## HonCut Implementation Status

### ✅ Completed (Ported)
- M1: Director Planner (`director_planner.py`)
- M2: Storyboard per-shot (`storyboard_generator.py`)
- M3: Adaptation Bridge (`adaptation_engine.py`)
- M4: Prompt Router (`prompt_router.py`)
- M5: Quality Gate (`quality_gate.py`)
- M6: Artifact Chain (`artifact_chain.py`)
- 29 gap-fix items from Toonflow/OM audit
- Vendor-ized 21 files (self-contained)

### 🔴 P0 (Must Fix)
- **Identity Anchor**: Verbatim structured attributes in shot prompts
- **Dialogue Subtitles**: Generate dialogue, differentiate from narration

### 🟡 P1 (Should Fix)
- **Camera Movement Coherence**: No consecutive static shots
- **Same-Scene Visual Sharing**: Shared lighting/color for same-scene shots
- **Seed Locking**: Consistent seed for same-scene shots

### 🔵 P2 (Nice to Have)
- **return_last_frame**: Video continuation (FLF2V already handles this)
- **Video Extension**: Extend existing videos (video_stitch is better)
- **Spatial Position Map**: Track character positions across shots
- **Boundary Frame Check**: Validate transition frames

---

## How to Use This Index

1. **For Codex**: When implementing a feature, check the Toonflow/OM file:line for reference implementation
2. **For Testing**: Each feature has acceptance criteria in `WORLD_CONSISTENCY.md`
3. **For Priority**: P0 → P1 → P2 order

---

## Notes

- Toonflow paths are relative to `/Users/soda/projects/Toonflow-app/`
- OpenMontage paths are relative to `/Users/soda/projects/OpenMontage/`
- HonCut paths are relative to `/Users/soda/projects/honcut/`
- Line ranges are approximate — check actual file for exact implementation

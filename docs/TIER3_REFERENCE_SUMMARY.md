# Tier 3 Reference Summary

Tier 3 OpenMontage skills are reference material, not tools to port. Their value to HonCut is a shared prompting method, a provider-selection model, and reliable scene/audio composition rules that improve use of the existing pipeline.

## 1. Prompting Methodology (from flux-best-practices)

### Core Principles

- Write descriptive prose instead of comma-separated keyword lists.
- Front-load the subject and other non-negotiable elements.
- Always specify lighting; it has the greatest impact on perceived quality.
- Aim for 30–80 words, while allowing critical identity-lock text to exceed that range.
- Do not use negative prompts for FLUX; describe the desired positive result.
- Prefer concrete attributes and spatial relationships over vague adjectives.
- Refine iteratively, changing one prompt variable at a time.

The baseline structure is:

```text
Subject → Action/Pose → Style/Medium → Context/Setting → Lighting → Camera/Technical
```

For HonCut video generation, identity continuity and camera instructions are production constraints, so the practical ordering becomes:

```text
Identity → Framing → Camera movement → Action → Setting → Lighting → Style → Audio
```

### HonCut Application

- Phase 2's `storyboard_generator` should use these principles when preparing the Seedance prompt blueprint.
- `_build_shot_prompt()` already uses prose-oriented labeled sentences rather than a bare keyword list. ✅
- Identity-lock blocks are placed before action, setting, and style details. ✅
- The portable visual style is appended consistently, and each blueprint has an explicit lighting field. ✅
- Deterministic camera language protects continuity even if the translation model omits part of the requested structure. ✅

Some current fixed safety-rail phrases use exclusions such as `no cuts` or `no zoom`. Those are motion constraints for the video provider, not FLUX creative prompting. Keep the creative description positive and isolate fixed provider constraints from it.

### Key Takeaways for HonCut

1. Use Subject → Action → Style → Context → Lighting → Technical as the semantic checklist.
2. Always give each shot a meaningful lighting specification, not only a generic quality marker.
3. Specificity beats vagueness: use `flowing auburn hair`, not merely `beautiful woman`.
4. Put character identity and stable appearance before scene description.
5. Keep a single focal action and a single camera movement per short shot.
6. Treat 30–80 words as guidance, not a reason to remove necessary identity locks.

See [PROMPTING_BEST_PRACTICES.md](PROMPTING_BEST_PRACTICES.md) for examples and a review checklist.

## 2. Video Provider Comparison (from ai-video-gen)

### Provider Matrix

Provider pricing, availability, and limits can change. The cost labels below are relative planning tiers distilled from the reference, not live quotes.

| Provider | Best For | Cost | Speed | Native Audio | Lip-Sync |
|---|---|---:|---|---|---|
| Seedance 2.0 | Cinematic, trailers, dialogue, multi-shot | High | Slow | Yes | Yes |
| VEO 3.1 | Photoreal landscapes | Medium | Medium | No | No |
| Kling Pro | Anime and stylized motion | Medium | Medium | No | No |
| Sora V2 | Creative or abstract footage | High | Slow | No | No |
| Runway Gen-4 | Product shots and fast iterations | Medium | Fast | No | No |
| Gemini Omni | Conversational editing of existing clips | Low | Fast | No | No |

The source describes four gateway families: fal.ai, HeyGen, Kling's official API, and Gemini. A provider name and gateway are separate decisions; for example, fal.ai Kling and Kling Official have different authentication and request behavior.

### HonCut Current State

- **Primary**: Seedance 2.0 through the local Bridge and Windows ComfyUI path.
- **Fallback**: The configured provider matrix documents candidates, but this task does not add or alter fallback execution logic.
- **Recommendation**: Keep Seedance as the default for cinematic and dialogue-driven content because native synchronized audio and lip-sync are decisive capabilities.
- **Selection interface**: Continue routing generation decisions through `video_selector` instead of coupling pipeline logic directly to a provider tool.

### Selection Guidance

| Need | Preferred Choice | Reason |
|---|---|---|
| Dialogue or visible speaking | Seedance 2.0 | Native synchronized audio and quoted-dialogue lip-sync |
| Cinematic trailer or multi-shot clip | Seedance 2.0 | Camera control and multi-shot generation |
| Landscape with no dialogue | VEO 3.1 | Strong photoreal landscape fit without paying for audio capability |
| Anime or strongly stylized motion | Kling Pro | Better stylistic fit in the reference comparison |
| Product-focused shot | Runway Gen-4 | Fast product-shot workflow |
| Creative or abstract concept | Sora V2 | Strong fit for exploratory imagery |
| Iterative edit to an existing clip | Gemini Omni | Stateful conversational editing is the differentiator |

### Operational Pattern

The multi-gateway reference recommends submitting an asynchronous job, retaining its execution ID, polling at 10-second intervals, and retrieving the returned video URL after completion. It presents a bounded wait in its polling example. HonCut's Bridge client already defaults to 10-second polling, but uses progress-aware queue and stall windows rather than a single five-minute timeout. Preserve HonCut's more defensive behavior unless operational evidence supports changing it.

### Key Takeaways for HonCut

1. Seedance 2.0 is the right primary choice for dialogue-driven work.
2. Consider VEO for landscape-only scenes where synchronized speech is irrelevant.
3. Separate provider capability selection from gateway availability and credentials.
4. Keep asynchronous polling bounded and observable.
5. Continue using the `video_selector` pattern for future selection and fallback logic.
6. Treat the config matrix as documentation for future routing, not an automatic fallback implementation.

## 3. Composition Patterns (from video-toolkit)

### Scene-Based Timing

The initial word-budget formula is:

```text
durationSeconds = ceil(word_count / 2.5) + 2
```

The inverse planning budget is:

```text
word_count = (durationSeconds - 2) × 2.5
```

Example: 17 words requires `ceil(17 / 2.5) + 2 = 9` seconds.

This estimate assumes roughly 2.5 spoken words per second, one second before audio begins, and one second of trailing padding. It is appropriate for planning, not final synchronization.

### Audio Sync Pattern

1. Generate one audio file per scene or shot, not one long voiceover file.
2. Measure each generated file's actual duration with `ffprobe`.
3. Set the scene duration to `ceil(actual_audio_duration + 2)`.
4. Start scene audio after a one-second delay.
5. Reserve the other second for trailing padding and transitions.
6. Review representative still frames before the final render.

Example: an audio file measuring 6.8 seconds produces a 9-second scene because `ceil(6.8 + 2) = 9`.

### HonCut Application

- Phase 2 can use the word-budget formula for an initial `speech_duration_s` estimate.
- HonCut's existing speech pacing annotation remains authoritative when it has already populated `speech_duration_s`; the new fallback must not overwrite it.
- `doubao_tts` exposes timestamp alignment that can support subtitle and sync decisions. ✅
- HonCut already has `ffprobe`-based duration utilities in its audio/media pipeline. ✅
- Phase 8 should prefer measured media duration over text-derived estimates whenever generated audio is available.

### Key Takeaways for HonCut

1. Per-scene or per-shot audio makes local timing corrections possible.
2. Word-budget estimation improves early storyboard planning.
3. Audio-first synchronization is more reliable than text-only estimation.
4. Preserve an existing speech-duration annotation instead of replacing it with a generic estimate.
5. Use actual duration plus explicit delay/padding when composing the final timeline.

## 4. Actionable Integrations

### 4a. Word Budget Estimation in Phase 2

`storyboard_generator.py` now provides:

```python
def estimate_shot_duration(word_count: int) -> float:
    """Estimate shot duration from word count using video-toolkit formula."""
    return math.ceil(word_count / 2.5) + 2
```

After `_build_shot_prompt()` finishes assembling the prompt, it counts words and initializes `shot["speech_duration_s"]` with this estimate only when the field is absent. Existing emotion-aware speech pacing data therefore wins.

This is deliberately an initial planning signal. Once TTS exists, final composition should probe actual audio and recalculate timing.

### 4b. Prompting Best Practices

`docs/PROMPTING_BEST_PRACTICES.md` records:

- prose-style examples;
- lighting patterns;
- front-loading strategy;
- identity-lock positioning;
- positive alternatives to negative prompts;
- an HonCut-specific prompt structure and checklist.

### 4c. Provider Comparison in Config

`VIDEO_PROVIDER_MATRIX` in `utils/config.py` records the subset most relevant to future HonCut routing:

- Seedance for cinematic, dialogue, lip-sync, and multi-shot work;
- VEO for photoreal landscapes;
- Kling for anime and stylized content;
- Sora for creative and abstract content.

The constant is descriptive. It does not change existing provider execution, availability checks, or fallback behavior.

## 5. Integration Boundaries

This Tier 3 integration intentionally does not:

- port the OpenMontage skills as executable tools;
- modify existing video provider implementations;
- change `pipeline_core.py`;
- add a fallback provider or new gateway;
- replace measured audio timing with a word-count estimate.

Future provider selection can consume `VIDEO_PROVIDER_MATRIX`, but should validate live availability, credentials, model limits, and pricing before making an automated choice.

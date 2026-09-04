## Context

See [proposal.md](proposal.md) for motivation. Serena inspection located the defect in Phase 2: the grid contract persists `stage`, `visible_fact`, `camera_movement` and a subject count, while the current local renderer always draws the same standing skeleton. The text hash only flips direction and changes a small vertical offset. Graphify confirms the direct chain is `generate_shot_storyboards` / guide migration → local guide derivation → renderer, with the resulting versioned fields projected by Phase 4 and validated/packaged for Phase 6.

The run-17 `S01_P01` evidence contains three materially different cells but identical skeleton geometry. It also shows why a body-contract-only solution is insufficient: canonical `generation_action_units` carry performers, targets, source IDs and ordered actions even when a non-required `body_action_contract` has empty mechanics fields.

## Goals / Non-Goals

**Goals:**

- Make each Gxx pose a deterministic projection of existing canonical action facts and optional body mechanics.
- Keep Phase 2 as the only guide producer and keep derivation local, identity-neutral and zero Provider.
- Make pose lineage, geometry and migration independently verifiable before Phase 6 submission.
- Preserve Graph/sequential parity and the current Phase 6 media role/order.

**Non-Goals:**

- Do not ask an LLM/VLM to interpret or review guide poses.
- Do not add new plot facts or repair upstream action-unit lineage.
- Do not reuse character pixels or turn the guide into a rendered storyboard.
- Do not change Phase 3 performance-board ownership or fix run-17's separate performance lineage failure.

## Decisions

### 1. Add a Phase 2 pose-semantics compiler before rendering

Phase 2 will compile a versioned `honcut.storyboard-guide-pose-contract` for each assigned Gxx. The compiler consumes, in authority order:

1. current Pxx `generation_action_units` and their `unit_id`, `source_action_unit_id`, source indexes, performers, targets and ordered actions;
2. matching valid `body_action_contract.beats` fields (`performer`, `technique`, `side`, `limbs`, `footwork`, `torso`, `weight_shift`, `direction`, `contact`, `end_pose`);
3. the Gxx stage and current camera-motion contract.

The compiler partitions ordered generation units across the Gxx allocated to that Pxx. When a unit spans multiple cells, stage/progress produces preparation, execution and recovery/terminal variants of the same action. When units outnumber cells, a cell may bind an ordered group, but the source IDs remain explicit and no unit is silently dropped. When cells outnumber units, the unit is expanded into monotonic phase samples rather than duplicated as identical geometry.

The P01 first unit receives a special timing role only when controlled pose classification resolves it to the pure `ready` family and at least one later unit resolves to a dynamic family. In that case Phase 2 emits exactly one completed-pose cell with `timing_role=initial_anchor` and `story_time_weight=0`; it is a t=0 state already established by the Phase 4 cinematic first frame. The remaining cells are reallocated to later dynamic units. This does not remove the source action or change ordering. A lone ready action, P02+ without a cinematic first frame, and other low-motion families such as prop hold or spatial state remain ordinary timed actions. Phase 4 transports the explicit timing metadata and Phase 6 tells the Provider to begin the next Gxx immediately; neither downstream owner reclassifies the action.

Controlled lexical classification over the canonical action strings is allowed only to select a generic pose family when typed mechanics are absent. It is versioned, deterministic, multilingual and auditable; it does not create source IDs, characters, props or plot outcomes. This is preferable to hashing prose (which has no semantic meaning) and to adding a Provider call (which would be probabilistic and costly).

Lexical evidence is field-aware and polarity-aware. Positive technique, footwork, torso, weight-shift, end-pose and canonical-action evidence outranks contact prose. Local Chinese and English negation scopes reject matches such as “无实际格挡”, “未击中” and “without blocking”, and the rejected evidence remains in the contract for audit. A body-action beat may only modify units whose source micro-action indexes match; the old single-beat fallback is permitted only when the unit has no source index at all.

Alternative considered: require every upstream beat to have a fully typed body contract. Rejected because non-combat and existing valid Pxx frequently carry usable structured action units without full body mechanics, and forcing upstream regeneration would broaden this owner and add probabilistic failure.

### 2. Render a controlled joint skeleton, not a single pose with jitter

Introduce a Phase 2-local pure geometry module, expected at `pipeline/src/phases/phase2/storyboard_guide_pose.py`. It will contain:

- a finite pose-family vocabulary such as neutral/spatial, locomotion, enter/exit, ready, strike, kick, evade/lean, block, grab/control, throw, fall/land/recover, hold/use-prop and reveal;
- normalized joint coordinates for head, neck, shoulders, elbows, hands, hips, knees and feet;
- deterministic transforms for side, direction, action phase, torso lean, weight shift and contact;
- multi-subject layout driven by bound performers/targets, using generic actor slots rather than character identity;
- abstract object/space glyphs for non-human performers and non-body events.

Each actor geometry and each cell receive a canonical JSON pose fingerprint. Repeated cells for one action receive monotonic progress samples that drive root translation and joint interpolation. A following action interpolates from the preceding canonical action's terminal joints and accumulated root position instead of resetting to the neutral template; the transition origin action-unit IDs are persisted and validated. Typed mechanics additionally modify center drop, torso lean, stance width, lead step and two-hand hold geometry. The raster renderer draws only from that geometry and structural validation rejects progress samples whose body displacement remains below the deterministic minimum. Red action arrows derive from the same resolved movement/contact vector; blue camera arrows derive only from the camera-motion contract. Randomness and prose hashes are prohibited from pose or arrow direction.

Alternative considered: import the Phase 3 performance-board renderer. Rejected because that module owns character-specific run-local reference assets and its dependency direction would couple Phase 2 production to a later Phase. Common concepts may be mirrored as a small pure vocabulary, but Phase 2 retains guide ownership.

### 3. Version the complete observable contract

Upgrade the timing-bearing pose contract and enclosing guide/shot-storyboard manifests together while retaining renderer `honcut.identity-neutral-story-guide-renderer.v2`. The semantic payload and receipt add:

- `pose_contract_schema` and `pose_policy_sha256`;
- per-cell ordered action bindings and source lineage;
- actor/object role slots, pose family, phase and normalized joint geometry;
- per-actor and per-cell pose fingerprints;
- resolved action-vector and camera-vector fields.
- per-cell `timing_role` and `story_time_weight`, plus the ordered zero-time anchor cell IDs propagated to Phase 6.

The shot-storyboard manifest version is bumped because it embeds the new guide contract. `STORYBOARD.json`, the continuity DTO, Phase 4 projection, asset packager and Phase 6 request fingerprint carry the current guide kind/renderer plus pose-contract hash and ordered pose fingerprints. Existing authority/non-authority roles and `source_pixel_usage=none` remain unchanged.

Alternative considered: keep guide v2 and change pixels only. Rejected because old receipts would validate against semantically weaker payloads and recovery could silently reuse identical-pose artifacts.

### 4. Fail closed on semantic/geometry drift while avoiding aesthetic QA

Validation is purely deterministic. It checks schema/version, Gxx→Pxx coverage, action-unit lineage, pose-policy hash, semantic-payload hash, joint bounds, actor count/roles, pose fingerprints, source-board hash and output image hash. It also checks that cells with different bound action/phase semantics do not collapse to the same pose fingerprint unless both explicitly resolve to the same non-body spatial state.

No VLM verdict, confidence threshold or aesthetic score is introduced. This directly verifies the renderer contract and avoids another probabilistic QA gate.

### 5. Migrate by redraw, never by blessing old pixels

Guide v2 remains audit-only. A zero-call migration may compile and render v3 beside the old files only when the current storyboard beat, complete Gxx assignment, source board hash, canonical action units and source action lineage all validate. Missing or contradictory lineage requires normal Phase 2 rebuild; unknown future versions fail closed. Migration does not overwrite v2 files or reuse their pixels.

### 6. Keep downstream responsibilities unchanged

Phase 4 copies the new fields into `GenerationChunk`; the asset packager verifies them; Phase 6 includes them in media validation, prompt indexing and task fingerprint. Phase 5 may verify artifact structure through the shared Phase 2 validator but does not reinterpret pose quality. Cross-Sxx bridges continue to use only cinematic first/last frames.

## Risks / Trade-offs

- [Lexical fallback can classify a rare action too broadly] → Keep a controlled, versioned taxonomy; persist matched evidence and fall back to an auditable neutral transition rather than inventing a precise move.
- [Too many performers can make a small guide unreadable] → Use deterministic slot compaction and relation glyphs, preserve the full logical performer/target list in the receipt, and test the supported visual slot limit.
- [Two stages can legitimately share a similar silhouette] → Compare normalized joint fingerprints and require variation only when bound action/phase semantics require visible motion; record an explicit `static_spatial_state` exception.
- [Contract version bump invalidates caches] → Provide verified local redraw migration and make cache invalidation explicit; no Provider work is needed.
- [Scope could absorb the run-17 Phase 3 failure] → Keep performance-board lineage out of this change and record it as a separate unresolved owner.

## Migration Plan

1. Land the pose contract/compiler and deterministic geometry tests without changing downstream consumption.
2. Switch Phase 2 generation and migration to guide v3/renderer v2; retain old assets as audit-only.
3. Propagate and validate the new hashes/fingerprints through continuity and Phase 6 fingerprinting.
4. Run provider-deny Phase 2→6 integration and ten-round recovery tests, then full regression/lint/diff checks.
5. Rollback is a revert of the versioned change; newly written v3 files remain harmless audit artifacts to older code, while no v2 file is overwritten.

## Open Questions

None. The implementation must derive poses from the currently persisted canonical action facts and may not expand the task to upstream semantic regeneration.

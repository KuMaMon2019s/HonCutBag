## 1. Baseline and Regression Fixtures

- [x] 1.1 Create a clean `codex/phase2-storyboard-guide-pose-fix` worktree from the run-17 candidate baseline `2807f4612d81216a9a241ed731548019aa290ef0`, import only this OpenSpec change, and verify the existing dirty infrastructure workspace and `prototypes/` are untouched.
- [x] 1.2 Repeat Serena preflight in the implementation worktree for the Phase 2 guide compiler/renderer, continuity DTO, Phase 4 projection, asset packager, Phase 6 consumer and tests; record definitions, references, callers and types before code edits.
- [x] 1.3 Repeat focused Graphify impact analysis for the renderer and guide contract, verify every reported `source_location` against Serena/source, and record the owner, upstream/downstream chain and non-applicable API/DB/frontend impacts.
- [x] 1.4 Add an anonymized run-17-style fixture with environment motion, one-to-many reveal, locomotion, evade, block, strike, grab/throw and prop use; verify it contains no production plot constants.
- [x] 1.5 Add failing regression tests proving the current renderer collapses materially different Gxx action/phase semantics to identical pose fingerprints while only arrows change.

## 2. Source-Bound Pose Contract

- [x] 2.1 Implement the versioned Phase 2 pose-semantics compiler and verify every Gxx records Sxx/Pxx, ordered generation/source action-unit IDs, performers, targets, stage and matched body-action beats.
- [x] 2.2 Implement deterministic Gxx partition/expansion over ordered action units and verify no unit is lost, duplicated across unrelated Pxx, reordered or inferred from another beat.
- [x] 2.3 Add a controlled multilingual pose-family fallback for canonical action text when typed mechanics are absent; verify matched evidence and policy hash are persisted and unknown actions resolve to an auditable neutral transition.
- [x] 2.4 Add strict lineage validation and verify missing IDs, conflicting source indexes, future schemas and cross-Pxx bindings fail before image/video Provider submission.

## 3. Deterministic Identity-Neutral Renderer

- [x] 3.1 Add the pure Phase 2 joint-geometry renderer with controlled pose families, direction, stage, torso, limb and weight-shift transforms; verify representative poses produce bounded, distinct normalized joint coordinates.
- [x] 3.2 Add deterministic multi-subject role slots and contact/spatial layout; verify performer/target count and interaction are represented without character identity pixels or unintended clones.
- [x] 3.3 Add abstract non-human/object/spatial glyph handling and verify environment/vehicle-only cells do not fabricate a standing person.
- [x] 3.4 Derive red action arrows from the resolved action vector and blue arrows only from the camera-motion contract; verify arrows and skeleton orientation remain coherent for opposite directions.
- [x] 3.5 Persist per-actor/per-cell pose fingerprints and verify start/progress/end cells vary when required, identical inputs render byte-identically, and no prose hash or randomness controls geometry.

## 4. Guide v3 Contract and Downstream Propagation

- [x] 4.1 Upgrade the narrative guide to v3, renderer to v2 and embedded shot-storyboard contract version; verify semantic payload, pose policy, action lineage, joint geometry, pose fingerprints and existing source-pixel/authority fields are hashed together.
- [x] 4.2 Extend the continuity DTO and Phase 4 projection with the pose-contract hash and ordered pose fingerprints; verify Graph and sequential execution produce identical JSON-safe chunks.
- [x] 4.3 Extend asset packaging and Phase 6 validation/fingerprinting; verify only the current Pxx guide is accepted, media ordering is unchanged, bridge requests exclude guides and mismatched versions/hashes fail before submission.
- [x] 4.4 Update structural validation to detect semantically distinct cells collapsing to identical poses except explicit static-spatial states; verify this check is deterministic and does not invoke VLM/LLM QA.

## 5. Migration and Legacy Isolation

- [x] 5.1 Implement side-by-side zero-call v2→v3 redraw for fully verified source boards, Gxx assignments and canonical action lineage; verify old files are not overwritten and the migration receipt binds old/new hashes.
- [x] 5.2 Mark incomplete/corrupt v2 guides audit-only and reject unknown future versions; verify no fallback blesses existing identical-pose pixels or reconstructs missing lineage from free prose.
- [x] 5.3 Add source guards confirming no production reference to `pipeline_core.py`, no Provider call in guide derivation/migration and no Phase 3 import from Phase 2; verify guards run in the target test suite.

## 6. Verification and Knowledge Sync

- [x] 6.1 Run focused pose/compiler/guide/continuity/Phase 6 tests and verify all action, multi-subject, non-body, tamper and version scenarios from the spec pass.
- [x] 6.2 Run Phase 2→6 provider-deny integration through Graph and sequential paths plus ten recovery rounds; verify Provider request count remains zero and guide/task IDs, pose fingerprints and hashes stay stable.
- [x] 6.3 Run the applicable Phase 1→9 zero-Provider regression, `make lint`, `git diff --check` and `make test`; verify `pipeline_core.py` is unchanged from the selected baseline.
- [x] 6.4 Update `docs/HONCUT_ARCHITECTURE.md` and relevant README contract references for guide v3/renderer v2, then verify no competing owner or architecture fact is introduced.
- [x] 6.5 Perform Serena post-change symbol/reference/type/diagnostics validation and verify no caller, DTO field or invalid reference is omitted.
- [x] 6.6 Run `openspec verify align-storyboard-guide-pose-semantics`, resolve every spec/task mismatch, and leave the change unarchived until all implementation and regression evidence is complete.
- [x] 6.7 Run `graphify update .`, focused `graphify affected` checks, save the verified result without secrets, and refresh reflections; verify Graphify hooks and community labels remain intact.
- [x] 6.8 Write a regression receipt bound to the candidate commit with the provider-deny evidence. Mark the live gate `pending_live_acceptance`; do not run a paid Phase 6 acceptance without a new explicit authorization.
- [x] 6.9 Remove the production-reachable unversioned test compatibility lineage, update affected fixtures to canonical generation action units, rerun verification, and replace the regression receipt with evidence bound to the corrected candidate commit.

## 7. Live-evidence Follow-up: Negation and Motion Fidelity

- [x] 7.1 Reproduce the live run misclassification where a negated contact phrase overrides positive evade mechanics; add Chinese and English regression cases.
- [x] 7.2 Make pose-family classification field- and polarity-aware, persist rejected negated matches, and prevent unmatched body-action beats from drifting across action units.
- [x] 7.3 Add monotonic per-action Gxx progress plus deterministic root, torso, center-of-gravity, stance and two-hand-hold geometry; reject arrow-only progress.
- [ ] 7.4 Upgrade the pose contract version, redraw the saved live structure with zero Provider calls, run focused/full regression and post-change Serena/Graphify/OpenSpec validation, then replace the regression evidence.

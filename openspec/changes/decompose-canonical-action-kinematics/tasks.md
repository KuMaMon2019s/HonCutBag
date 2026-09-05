## 1. Mandatory Preflight and Scope Freeze

- [x] 1.1 Re-read `docs/HONCUT_ARCHITECTURE.md`, run `graphify reflect --if-stale`, and verify Serena is activated on the implementation worktree; record the exact clean baseline and confirm `pipeline_core.py` is unchanged.
- [x] 1.2 Use Serena to locate the body-action contract symbols, definitions, references, implementations, DTOs and tests; verify the call chain from Event Extractor through Adaptation/Storyboard to Phase 2, Phase 3 and Phase 6.
- [x] 1.3 Run full Graphify impact analysis for `build_body_action_contract`, `compile_pose_contracts`, performance-board planning and action-execution brief; verify State/Artifact/cache/test/doc impacts and that no frontend, API or database schema is involved.
- [x] 1.4 Publish the implementation target and expected file list before editing; verify the owner remains Phase 1 body-action contract and the change is classified as an Architecture Change.

## 2. Canonical Kinematics Schema and Compiler

- [x] 2.1 Add the versioned internal canonical kinematics DTO/enums for source beats, final GAU/Pxx projections, actor-local axes, ordered phase windows, channel states, orientation and spatial transforms; verify strict serialization rejects extra fields, illegal enums and non-monotonic phases, while the Provider-facing `BodyActionUnderstanding` JSON schema remains unchanged.
- [x] 2.2 Implement ordered `actor_tracks` for every canonical performer and the pure story-neutral primitive compiler for root, waist/torso, head, bilateral arm/hand and bilateral leg/foot channels; verify every phase of every actor track contains the complete stable channel inventory.
- [x] 2.3 Implement separate leg-versus-foot and arm-versus-hand mechanics plus left/right mirroring; verify asymmetric fixtures never swap sides and distal contact does not collapse into the proximal joint chain.
- [x] 2.4 Implement Pxx-local yaw anchors and optional camera-relative projection without mutating actor-local orientation; verify a camera orbit changes front/profile/back projection while relative actor yaw remains constant and no absolute world-facing fact is invented.
- [x] 2.5 Implement explicit turn/spin/flip transforms with axis, direction, amount, support release, airborne interval and landing state; verify camera rotation, ordinary dodge and view mirroring cannot create a flip or spin.
- [x] 2.6 Implement load/drive/apex/contact/follow-through/settle phase selection and adjacent-action terminal-state inheritance; verify multiple action groups never reset to neutral or add an idle guard between moves.
- [x] 2.7 Add source-basis/evidence hashes and a deterministic policy hash to every compiled beat; verify identical input is byte-semantically stable and changing a source fact changes the kinematics hash.
- [x] 2.8 Add fixed-precision normalized joint/root translations and rotations, activation/support/contact states and generic amplitude-class minima; verify active full-body actions move root, waist and a major limb chain by deterministic floors while still/guard actions are not artificially enlarged.

## 3. Phase 1 Ownership and Persistence

- [x] 3.1 Attach source-beat kinematics keyed by `micro_action_index` only inside the existing body-action contract builder and advance its schema; verify Event Extractor, Director completion and Adaptation all call the same owner without modifying `BodyActionUnderstanding` or issuing another LLM request.
- [x] 3.2 After generation action units and Pxx are finalized, build strict per-GAU `kinematics_projection` records from ordered `source_micro_action_indexes`; verify each Pxx has complete, non-overlapping coverage, no foreign lineage, and staged execution sub-slices remain children of the same GAU.
- [x] 3.3 Preserve source and projected kinematics through production-event rewrite, shot inheritance and storyboard-beat partitioning; verify source event/action/Pxx lineage and hashes remain exact across every projection.
- [x] 3.4 Ensure kinematic subphases do not affect story action count, timeline slices, Pxx capacity or material duration; verify capacity tests report the same generation-action-unit totals before and after decomposition.
- [x] 3.5 Advance Event Flow, layered Adaptation and secondary Storyboard cache identities required by serialized contract changes; verify old fingerprints cannot silently reuse incompatible data and future versions fail closed.
- [x] 3.6 Add zero/one/multiple actor and non-body event coverage, including one simultaneous GAU with attack and defence performers; verify non-body actions receive no invented human channels and each real performer gets exactly one independent actor track.

## 4. Phase 2 Pose and Atlas Projection

- [x] 4.1 Advance the pose-contract schema and replace body-action pose-family target interpolation with direct sampling from the final GAU/Pxx kinematics projection; verify non-body/spatial geometry retains existing behavior.
- [x] 4.2 Project waist, head, bilateral limb/foot channels and root displacement into actual skeleton joints; verify each active channel produces measurable geometry changes rather than arrow-only differences.
- [x] 4.3 Project Pxx-local relative yaw into front/back/left-profile/right-profile and three-quarter camera relations only when the camera contract supports it; verify camera orbit changes view fingerprints without mutating actor yaw and missing camera evidence remains controlled unknown.
- [x] 4.4 Preserve terminal-to-initial channel state across action-group boundaries and compress phase windows for fast cadence; verify compound actions stay continuous and complete without prolonged setup or terminal hold.
- [x] 4.5 Bind kinematics schema/hash to every Gxx, pose sample, guide/atlas receipt and Artifact fingerprint; verify another Pxx, stale hash or future schema is rejected before media generation.
- [x] 4.6 Add pixel-level tests for left/right limbs, leg-versus-foot, waist/head rotation, front/back/profile, flip and spin sequences plus simultaneous multi-actor tracks; verify each actual raster skeleton changes by deterministic minimum distances and performers are never collapsed together.

## 5. Phase 3 and Phase 6 Consumers

- [x] 5.1 Advance Phase 3 performance pose constraints/guide schema and derive them from final per-GAU actor tracks instead of reparsing action prose; verify reference-board cell plans retain the same source-action and performer lineage.
- [x] 5.2 Render local Phase 3 pose guides from sampled kinematic channel states and update compact image prompts; verify prompts contain no invented move, printed layout marker or full raw contract dump.
- [x] 5.3 Advance the Phase 6 action-execution brief to include ordered phase IDs, per-performer active channels, Pxx-local orientation/transform summary and kinematics hash; verify final media ordering, media roles and nine-image budget remain unchanged.
- [x] 5.4 Update Phase 6 prompt projection to express fast continuous body coordination from the brief while preserving canonical action order; verify no new action, flip, spin, actor or target appears.
- [x] 5.5 Verify Phase 4 first-frame, continuity and Phase 5/8 QA remain consumers of the same body-action hash without taking kinematics ownership or adding retries.

## 6. Motion Blueprint Convergence and Legacy Isolation

- [x] 6.1 Replace the isolated acceptance motion-blueprint compiler's independent technique motion authority with canonical kinematic phases; verify the blueprint and production pose pipeline share the same kinematics hash and phase fingerprints while ordinary Phase 6 never adds it as `reference_video` or another production media role.
- [x] 6.2 Quarantine the old independent blueprint technique registry as immutable audit-only compatibility evidence; verify it cannot satisfy current preflight or paid admission.
- [x] 6.3 Add a provider-deny compound-action replay covering large amplitude and fast cadence; verify root, waist/head and multiple major limb chains move, all canonical groups execute in order and no Provider boundary is reached.

## 7. Migration, Recovery and Regression

- [x] 7.1 Implement strict sidecar migration for evidence-complete legacy body-action contracts embedded in Event/Storyboard parent Artifacts; verify original parents remain unchanged/audit-only and the migration receipt binds parent Artifact, old/new schema, policy, final GAU/Pxx lineage and hashes.
- [x] 7.2 Mark incomplete, ambiguous, corrupted or future-version parent contracts and every downstream pose asset stale/audit-only; verify no prose guessing, partial refresh or Provider request occurs.
- [x] 7.3 Add Graph and sequential-path equivalence tests; verify both paths produce identical kinematics, pose, prompt and task fingerprints from the same Phase owner.
- [x] 7.4 Run ten recovery rounds with all Provider transports denied; verify task counts, action-group counts, contract hashes, pose/media hashes and fingerprints remain stable with zero requests.
- [x] 7.5 Run targeted body-action, storyboard-pose, performance-board, Phase 6 prompt, blueprint and migration tests; verify every scenario in the delta spec has regression coverage.
- [x] 7.6 Add a source/schema guard proving the Provider-facing `BodyActionUnderstanding` fields and model request count are unchanged; verify the new internal contract cannot leak into the LLM response schema.

## 8. Documentation and Final Validation

- [x] 8.1 Update `docs/HONCUT_ARCHITECTURE.md` and applicable README/workflow documentation with the canonical kinematics owner, coordinate system, consumer boundaries and migration semantics; verify no competing owner is documented.
- [x] 8.2 Run Serena post-change validation for modified symbols, references, callers, interfaces/types and diagnostics; verify no stale reference or parallel implementation remains.
- [x] 8.3 Run `make lint`, `git diff --check`, `make test` and the required zero-Provider offline acceptance; verify all gates pass and `pipeline_core.py` has no diff from the baseline.
- [x] 8.4 Run strict OpenSpec validation and OpenSpec Verify; verify implementation, tests and docs satisfy every requirement and task before marking the change complete.
- [x] 8.5 Run `graphify update .`, affected-path review, `graphify save-result` and `graphify reflect --if-stale`; verify the graph records the new owner/consumer paths without storing secrets or Provider evidence.
- [x] 8.6 Update Serena Memory only if the final verified owner/schema/consumer boundary is durable project knowledge; verify no one-off bug, Prompt, media or temporary path is stored.
- [x] 8.7 Write a regression receipt bound to the final commit and mark live acceptance `pending_live_acceptance`; verify no paid request occurred and any later real Provider gate requires a separate explicit fee authorization.

## 9. Falsification Follow-up

- [x] 9.1 Tighten source and projection validation so phase IDs, source indexes, yaw, camera relation, performer identity, GAU identity and child hashes are internally consistent even when an input recomputes its outer hash; add negative tests for every rejected shape.
- [x] 9.2 Require complete canonical-kinematics coverage for every body-action GAU in a Phase 2 cell group; reject mixed complete/missing projections and bind sampled projection lineage to every action binding before rendering.
- [x] 9.3 Connect the strict legacy sidecar compiler through the existing Artifact migration boundary without adding a new service; validate parent content/lineage evidence, preserve immutable parents, and prove the supported entrypoint is reachable outside its unit test.
- [x] 9.4 Run the falsification suite, target and full regression, Serena post-validation, strict OpenSpec Verify and Graphify sync; preserve the previous receipt as superseded audit evidence and write a new receipt bound to the corrected implementation commit with live status `pending_live_acceptance`.

## 1. Preflight and Contract Baseline

- [x] 1.1 Re-run Serena symbol/reference/implementation analysis for the Phase 2 pose compiler, guide renderer, camera-motion contract, `GenerationChunk`, Phase 6 prompt/media assembly and tests; record the minimal edit set before touching production code.
- [x] 1.2 Re-run Graphify query/affected/path analysis against the current worktree graph, verify every inferred edge against source, and record Phase 2/4/5/6, migration and acceptance impacts.
- [x] 1.3 Add failing contract tests for 4/7/10/15-second timing, action-group/pose-sample separation, semantic terminal hold and deterministic hashes; verify they fail for the expected missing behavior.

## 2. Duration and Camera Capability Contracts

- [x] 2.1 Extend the versioned video capability profile with pose-sample density, reliable action-group limits, atlas page options and terminal-hold bounds; verify boundary tests for 4–15 seconds and invalid values.
- [x] 2.2 Add a pure timing-contract builder for `initial_anchor`, `story_action` completion window and `terminal_hold`; verify 7 seconds yields a finite dynamic budget and acceptable tail without hard-coded plot terms.
- [x] 2.3 Extend the existing camera-motion contract with height, translation/zoom/pan/tilt/segmentation parameters and a deterministic minimum-duration validator; verify 90 degrees at 10 degrees/second rejects a 7-second Pxx before Provider work.
- [x] 2.4 Verify Adaptation remains the only camera-technique selector and persists the resolved contract/hash; add tests proving Phase 2 cannot silently alter an infeasible camera path.

## 3. Phase 2 Action Groups and Adaptive Atlas

- [x] 3.1 Version and implement ordered action groups over canonical generation/source action units; verify no action is added, reordered, dropped or moved across Pxx.
- [x] 3.2 Allocate one or more monotonic pose samples per action group while preserving zero-time initial anchors, transition origin and cumulative geometry; verify distinct high-amplitude stages produce distinct fingerprints.
- [x] 3.3 Project each pose sample through the single canonical camera path so front/profile/rear-three-quarter samples change joint projection and occlusion, not only the blue arrow; verify deterministic geometry snapshots.
- [x] 3.4 Implement deterministic Phase 2 `single_atlas` and applicable `paged_atlas` candidates from one canonical pose payload; verify complete ordered page allocation with no overlap, omissions or semantic drift.
- [x] 3.5 Upgrade the identity-neutral renderer for 9/18/27/36-cell layouts within Provider dimensions/aspect ratio, retaining Gxx and controlled arrows while copying zero source pixels; verify image hashes and pixel isolation.
- [x] 3.6 Upgrade guide and shot-storyboard manifests with page order, action groups, pose samples, timing/camera hashes, renderer identity and `source_pixel_usage=none`; verify tamper and future-schema failures are closed.

## 4. Continuity and Phase 6 Consumption

- [x] 4.1 Upgrade `GenerationChunk` and continuity-plan construction with JSON-safe atlas pages, action groups, pose samples, completion window, terminal mode and camera-contract hash; verify Graph and sequential construction serialize identically.
- [x] 4.2 Update the asset packager to freeze authoritative media before selecting one pre-rendered atlas candidate, prefer paged guides when at least two slots remain, fall back to a single dense atlas when exactly one remains, and fail closed when required media exceeds nine images.
- [x] 4.3 Update Phase 6 prompt assembly to reference the final page/media indexes, treat atlas cells as an ordered motion envelope, start after the zero-time anchor, finish within the completion window and hold the canonical terminal semantics without replaying the initial pose.
- [x] 4.4 Add optional `exact_pose` terminal-reference consumption and budget/fingerprint coverage while keeping `semantic_hold` as the no-extra-media default; verify semantic guard variation passes and missing exact evidence blocks before submission.
- [x] 4.5 Include all page, action-group, timing, camera, terminal and final-media hashes in generation task fingerprints/receipts; verify cold start and resume preserve task IDs and never submit a duplicate.

## 5. Migration, Regression and Documentation

- [x] 5.1 Implement zero-Provider side-by-side migration for fully verifiable legacy single-grid contracts; mark incomplete/corrupt legacy assets audit-only and reject unknown future versions without overwriting old files.
- [x] 5.2 Add regression tests for Phase 2 correction and migration, Phase 3 character-lock refresh, Phase 5 consumers, Phase 6 media order, cross-Sxx bridge exclusion and Graph/sequential owner parity.
- [x] 5.3 Run the relevant Phase 1–9 provider-deny acceptance and ten recovery rounds; verify Provider request count is zero and pages, observations, tasks and media hashes remain stable.
- [x] 5.4 Update `docs/HONCUT_ARCHITECTURE.md`, README migration entry and source guards with the versioned timing/atlas/camera contracts; verify `pipeline_core.py` is unchanged and has no new production references.
- [x] 5.5 Run Serena post-change symbol/reference/type/diagnostics validation and resolve every introduced stale reference or contract mismatch.
- [x] 5.6 Run target pytest, `make lint`, `git diff --check` and `make test`; bind the zero-request regression receipt to the final candidate commit.
- [x] 5.7 Run `openspec validate add-adaptive-storyboard-pose-atlas --strict` and OpenSpec Verify; leave any incomplete requirement/task unchecked and record unresolved risks.
- [x] 5.8 Run `graphify update .`, inspect affected owners at depth 2, save the verified result and refresh lessons; update Serena Memory only if a durable architectural rule changed.

## 6. Paid Admission Boundary

- [x] 6.1 After all zero-cost gates pass, prepare a no-submit seven-second acceptance payload with frozen model, media order, atlas strategy, timing/camera contract, fingerprint and one-request hard limit; verify no Provider call occurs.
- [x] 6.2 Keep status `pending_live_acceptance` until a new explicit fee authorization is provided; any later live run must allow at most one Seedance submission, no retry/reshoot/budget expansion, and must record call-chain and human visual verdict separately.

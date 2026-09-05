## 1. Baseline and mandatory preflight

- [x] 1.1 Create a clean `codex/seedance-motion-blueprint-gate` worktree from the explicitly selected production commit, preserve the current dirty workspace untouched, and verify the baseline commit plus `pipeline_core.py` hash are recorded.
- [x] 1.2 Complete Serena preflight for the Phase 2 action/pose owners, Phase 6 media/prompt builder, Seedance adapter, TOS uploader, `GenerationTaskStore`, and related tests; record symbols, definitions, references, implementations, types, callers, and the minimum safe change set.
- [x] 1.3 Complete Graphify impact analysis for the Phase 2→Phase 6 media path, upload path, task lifecycle, Artifact lineage, acceptance scripts, and affected tests; verify every selected graph node against source and save the result.
- [x] 1.4 Reconcile the preflight findings with this OpenSpec design and stop for review if the verified production call chain or owner boundaries conflict with the planned acceptance-only approach.

## 2. Versioned motion-blueprint contracts

- [x] 2.1 Add strict JSON-safe schemas for the temporal blueprint, versioned generic motion policy, actor/event tracks, camera track, source lineage, renderer identity, semantic frame hash, and media hash; verify unknown future versions and incomplete lineage fail closed.
- [x] 2.2 Define a small data-driven registry of supported normalized motion primitives without story-specific names or prose and verify unsupported actions are reported before any media upload.
- [x] 2.3 Add fixtures derived from canonical action contracts that cover locomotion, large torso/limb changes, ordered compound actions, prop/contact timing, camera movement, unknown actions, and zero-actor inputs.

## 3. CPU-compatible deterministic compiler

- [x] 3.1 Implement deterministic joint/root interpolation at a supported Seedance frame rate using existing lightweight image/media dependencies and verify compilation requires neither CUDA nor a local generative model.
- [x] 3.2 Render identity-neutral actors, optional neutral prop/contact geometry, and camera transforms without faces, textures, costume details, labels, grids, arrows, subtitles, logos, or watermarks; verify pixel-content guard tests.
- [x] 3.3 Encode a Seedance-compatible H.264 MP4 with pinned parameters and verify repeated compilation on the supported host produces identical semantic and media hashes.
- [x] 3.4 Implement local blueprint measurements for ordered event intervals, joint/root displacement, action onset, terminal hold, duration, FPS, codec, dimensions, ratio, and size; verify static and low-amplitude inputs fail before upload.

## 4. Seedance-only acceptance projection

- [x] 4.1 Add a dedicated no-submit acceptance entry point that reads only hash-verified canonical evidence, compiles or validates one blueprint, and writes an isolated preflight receipt; verify normal CLI, Lifecycle, Graph, and checkpoint discovery cannot invoke it.
- [x] 4.2 Reuse the verified production Phase 6 request builder so the frozen identity, initial composition, duration, output profile, current-Pxx semantics, final media order, and one-based indices match production while the blueprint has the explicit `reference_video` motion role.
- [x] 4.3 Add a single-variable equivalence checker and verify changes to identity pixels, start frame, duration, profile, unrelated prompt semantics, or extra experimental media block the request.
- [x] 4.4 Validate the configured Seedance 2.0 capability and official image/video/request limits, freeze the exact model, payload/media hashes, TOS PUT ceiling, one video-submission ceiling, and task fingerprint, and verify an unsupported model or unbounded budget stops at zero requests.
- [x] 4.5 Add a Seedance-only transport guard that rejects every alternate Provider configuration or fallback path and verify non-Seedance credentials and availability cannot influence the gate.
- [x] 4.6 Route the optional live submit through the existing TOS owner, `GenerationTaskStore`, Runtime no-retry scope, and raw Seedance POST guard; verify `SubmissionAttempted`/`submission_uncertain` is durable before transport and the same fingerprint cannot submit twice.

## 5. Falsifiable verdict and regression

- [x] 5.1 Implement separate call-chain and business-motion verdict records with deterministic output measurements plus an explicit human-verdict slot; verify Provider acceptance/download alone cannot mark the capability passed.
- [x] 5.2 Implement terminal outcomes `pending_live_acceptance`, `capability_gate_passed`, `capability_route_paused`, and `submission_uncertain`; verify any conclusive motion failure pauses the route without retry, redraw, reshoot, budget increase, or alternate Provider call.
- [x] 5.3 Add tests proving a single-actor pass is scoped only to single-actor evidence and cannot authorize multi-actor choreography or production activation.
- [x] 5.4 Run the target tests and Provider-deny replay, including ten identical resumes with zero TOS/Seedance submissions and stable blueprint, task, and receipt hashes.
- [x] 5.5 Run Phase 1–9 zero-Provider regression where applicable, `make lint`, `git diff --check`, and `make test`; verify `pipeline_core.py`, Graph topology, ordinary production media contracts, and unrelated user files are unchanged.

## 6. Validation, synchronization, and live-gate preparation

- [x] 6.1 Complete Serena post-change validation for all changed symbols, references, callers, interfaces, types, and diagnostics; verify no duplicate or stale implementation remains.
- [x] 6.2 Run OpenSpec strict validation and verify every requirement/scenario is covered by tests or a clearly pending live/human step.
- [x] 6.3 Run `graphify update .`, inspect the affected owner at depth two, save the sanitized result, and refresh reflections; update Serena Memory and architecture documentation only if verified durable ownership or production contracts changed.
- [x] 6.4 Generate a fresh isolated Seedance 2.0 no-submit preflight bound to the final candidate commit and regression receipt, verify it records zero Provider submissions and `pending_live_acceptance`, and report the exact finite paid scope to the user.
- [ ] 6.5 After a separate explicit fee authorization, perform at most one Seedance 2.0 video submission, persist its terminal call-chain evidence, and stop without retry on failure, rejection, timeout, uncertainty, or inadequate motion.
- [ ] 6.6 Obtain and record the explicit human motion verdict; on pass prepare a separate production-integration OpenSpec proposal, and on failure record `capability_route_paused` with no production activation.

## 7. Perceptual-amplitude correction after local falsification

- [x] 7.1 Preserve the first nine-second blueprint and receipts as audit-only evidence, record its slow-drift measurements, and ensure the revised paid-admission policy rejects it without upload.
- [x] 7.2 Upgrade the motion policy and renderer contracts to v2, freeze a four-second single-action gate, cap setup anchors at 0.15 seconds, and compile dynamic primitives as deterministic anticipation/peak/recovery/terminal phase curves without inventing canonical actions.
- [x] 7.3 Replace endpoint-only admission with semantic peak-speed, apex-timing, multi-major-joint, and perceptible-onset checks plus decoded foreground occupancy, centroid travel, transition-activity, and terminal-hold checks.
- [x] 7.4 Add regression fixtures for slow endpoint drift, single-joint false positives, setup-heavy clips, insufficient actor occupancy, deterministic four-second encoding, and a visibly large single-action blueprint; run a ten-resume Provider-deny replay with stable hashes.
- [x] 7.5 Complete Serena post-validation, target and full tests, lint, diff check, OpenSpec strict verification, Graphify refresh, and a fresh no-submit receipt bound to the corrected final commit; leave tasks 6.5 and 6.6 pending.

## 8. Technique-specific choreography correction

- [x] 8.1 Re-run Serena and Graphify preflight for the acceptance compiler, request projection, legacy audit path, and tests; confirm the change remains acceptance-only and does not alter production Graph, Phase owners, Provider policy, or `pipeline_core.py`.
- [x] 8.2 Upgrade the blueprint, policy, renderer, and gate receipt contracts to v3 and add a deterministic technique registry hash plus per-event technique ID, ordered phase IDs, contact phases, and keyframe fingerprint.
- [x] 8.3 Replace the shared dynamic phase curve with technique-specific key poses and interpolation for every supported dynamic primitive, keeping setup anchors separate and rejecting incomplete registry entries before rendering.
- [x] 8.4 Add deterministic technique assertions and regression coverage proving distinct primitives have distinct temporal/geometric signatures, contact is phase-local, and representative evade/kick/block/strike biomechanics are visible in semantic frames and rendered pixels.
- [x] 8.5 Remove choreography tuning language from the Seedance prompt projection, retain only media authority/non-authority instructions, and prove prompt wording cannot change the compiled blueprint contract or measurements.
- [ ] 8.6 Preserve v1/v2 blueprints as audit-only, run ten Provider-deny resumes, target/full tests, lint, diff check, Serena validation, strict OpenSpec verification, Graphify incremental sync, and generate a fresh v3 no-submit receipt in a new isolated directory; leave paid/human tasks 6.5 and 6.6 pending.

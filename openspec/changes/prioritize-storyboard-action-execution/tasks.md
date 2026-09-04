## 1. Preflight and Failure Fixture

- [x] 1.1 Re-run Serena symbol/reference/type investigation on the exact implementation baseline for Phase 6 prompt construction, continuity chunking, media packaging, fingerprints, and live acceptance; verify all active callers and implementations are recorded before editing.
- [x] 1.2 Re-run narrow Graphify queries for the Phase 6 prompt owner and affected depth-two paths; verify source locations against Serena and source code and record that this is not an architecture change.
- [x] 1.3 Import the run-02 prompt, receipt, media hashes, ordered action groups, and visual verdict as immutable test evidence without copying secrets, Provider URLs, or response bodies; verify every imported evidence file matches its recorded SHA-256.
- [x] 1.4 Add a failing Provider-deny regression proving the current request places action priority at the tail and permits the static-opening-pose failure; verify no Provider client can be reached by the fixture.

## 2. Versioned Action-Execution Brief

- [x] 2.1 Add the JSON-safe `honcut.action-execution-brief.v1` DTO and deterministic canonical serialization; verify identical inputs yield identical schema payload and hash.
- [x] 2.2 Compile the brief from only current-Pxx action groups, source action-unit lineage, shot timing, terminal state, resolved media indexes, and the primary camera technique; verify later-Pxx actions and cells are rejected or excluded.
- [x] 2.3 Render ordered observable body displacement, contact, weight transfer, prop interaction, immediate-start, completion-window, and semantic-terminal instructions; verify every canonical group appears exactly once and in order.
- [x] 2.4 Add deterministic validation for missing/duplicate groups, broken lineage, unresolved media indexes, impossible timing, and future schema; verify each fails before task submission.
- [x] 2.5 Commit the independent brief contract as `feat: add phase6 action execution brief` after its focused tests pass.

## 3. Action-First Prompt Projection

- [x] 3.1 Resolve and validate the final media list before rendering Provider prose, preserving the established media order and authority roles; verify Graph and sequential paths produce identical media indexes and hashes.
- [x] 3.2 Render the media-role preamble followed immediately by the complete action brief, explicitly separating identity/composition references from motion authority; verify the action brief begins before identity, continuity, style, negative, and camera support sections.
- [x] 3.3 Add deterministic compact projections for canonical identity, continuity, spatial/prop locks, camera, style, and negative constraints while retaining their complete source contracts in fingerprint/receipt metadata; verify exact-one identity and distinguishing signatures remain present.
- [x] 3.4 Deduplicate camera and motion instructions so the submitted prompt contains one primary camera instruction and one motion authority; verify camera support does not precede or contradict ordered body action.
- [x] 3.5 Remove the late duplicate motion-priority/action-window suffix from the final request path; verify equivalent markers cannot appear twice in the Provider payload.
- [x] 3.6 Commit prompt ordering and media-role isolation as `fix: prioritize current-shot action execution` after prompt, continuity, and packaging tests pass.

## 4. Prompt Budget, Fingerprints, and Recovery

- [x] 4.1 Add named deterministic section budgets derived from the existing Provider capability limit, reserving mandatory action and identity content before optional support clauses; verify no arbitrary tail truncation occurs.
- [x] 4.2 Fail closed before submission when the media preamble plus mandatory action/identity projection cannot fit; verify GenerationTaskStore has no `SubmissionAttempted` event for budget failure.
- [x] 4.3 Add action brief schema/hash, ordered group IDs, source lineage, media-role indexes/hashes, projection policy hash, final prompt hash, and canonical hash to task identity and receipts; verify each relevant input change invalidates the fingerprint.
- [x] 4.4 Verify cold start plus ten recoveries preserve task count, prompt hash, brief hash, media hashes, and Provider request count zero.
- [ ] 4.5 Commit budgeting and task identity as `fix: make phase6 prompt projection deterministic` after recovery tests pass.

## 5. Zero-Cost Replay and Regression

- [ ] 5.1 Add a Provider-deny run-02 replay that reconstructs the new request from hash-verified persisted evidence and writes a separate replay receipt; verify the original run and receipts remain byte-for-byte unchanged.
- [ ] 5.2 Assert the replay prompt is within the capability limit, contains all action groups near the front, has no duplicate camera/motion contract, and preserves all media/canonical hashes; verify Provider request count is zero.
- [ ] 5.3 Add parameterized tests for one/many action groups, contact/no-contact, prop/no-prop, P01/P02+, short/long durations, optional detail overflow, and mandatory content overflow.
- [ ] 5.4 Verify Phase 6 Graph and sequential execution produce identical briefs, prompts, fingerprints, media order, and fail-closed behavior.
- [ ] 5.5 Verify `pipeline_core.py`, Phase 2 pose-atlas output/hashes, Provider transport, Runtime retry policy, Graph topology, and database schemas are unchanged.
- [ ] 5.6 Commit replay and regression coverage as `test: enforce action-first provider requests` after focused tests pass.

## 6. Validation and Knowledge Sync

- [ ] 6.1 Run all focused Phase 6 prompt, continuity, packaging, fingerprint, recovery, and live-acceptance tests and verify Provider requests remain zero.
- [ ] 6.2 Run the Phase 1–9 zero-Provider acceptance, Graph/sequential parity, and recovery matrix; verify no paid client is reachable.
- [ ] 6.3 Run `make lint`, `git diff --check`, and `make test`; verify all required gates pass without modifying unrelated files.
- [ ] 6.4 Perform Serena post-change validation on every modified symbol, caller, implementation, type, and file diagnostic; verify no stale reference or parallel prompt path remains.
- [ ] 6.5 Run `openspec verify prioritize-storyboard-action-execution` (or the installed equivalent) and strict OpenSpec validation; verify implementation and acceptance evidence satisfy every scenario.
- [ ] 6.6 Run `graphify update .`, query affected depth two, save a sanitized useful result, and refresh reflections if stale; verify the graph resolves all changed source locations.
- [ ] 6.7 Update `docs/HONCUT_ARCHITECTURE.md` only for the verified Provider-facing projection and replay semantics, and update Serena memory only if the completed change creates durable architectural knowledge.

## 7. Paid Admission and Future One-Shot Gate

- [ ] 7.1 Write a zero-submit paid-admission receipt bound to the final candidate commit, regression receipt, prompt policy hash, exact media list, model, duration, and a hard maximum of one video submission; verify no Provider request is made.
- [ ] 7.2 Stop and request a new explicit fee authorization for the new candidate/run; verify prior run-02 authorization is not reused.
- [ ] 7.3 After authorization, execute at most one live video submission with atomic `submission_uncertain` recording and no retry/reshoot/compensation path; verify task events and request accounting equal one logical submission.
- [ ] 7.4 Record call-chain and business verdicts separately, including ordered action completion, meaningful body displacement/contact, semantic terminal state, identity stability, and absence of grid/text/clone contamination; verify any failure remains final and audit-only.
- [ ] 7.5 Mark the change live-accepted only when regression, replay, one-shot call chain, and business verdict all pass; otherwise persist `live_acceptance_failed` without expanding cost or scope.

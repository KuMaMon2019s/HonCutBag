## Context

See `proposal.md` for motivation and `specs/phase6-action-execution-priority/spec.md` for the behavioral contract.

The run-02 request is a concrete counterexample: its 15,952-character prompt placed the authoritative beat and motion-priority instructions after approximately 14,900 characters of media, canonical identity, spatial, prop, and camera material. A second camera execution section alone occupied roughly 2,976 characters. The output preserved identity and avoided grid contamination, but it largely held the transverse guard pose and did not clearly execute the canonical block-to-evade action groups.

Serena tracing identifies the active chain as Phase 6 prompt construction → continuity chunk augmentation → media packaging. Graphify confirms the downstream consumers are the Provider request builder, full-chain task freezer, Phase 6 live acceptance, and their tests. Phase 2 has already produced a hash-verified v7 atlas and is not the failed owner.

Constraints:

- Preserve dependency direction and the shared Phase 6 owner used by Graph and sequential execution.
- Preserve complete canonical contracts and fingerprints; only their Provider-facing prose projection may be compacted.
- Preserve Runtime/transport retry and submission semantics.
- Do not modify `pipeline_core.py`, Phase 2 renderer, Provider clients, Graph topology, or database schemas.

## Goals / Non-Goals

**Goals:**

- Make action semantics the earliest substantive instruction after media mapping.
- Remove prompt competition without weakening identity or provenance.
- Make prompt fitting deterministic and fail closed before cost.
- Reproduce request composition offline from persisted evidence.
- Evaluate visible action completion separately from technical Provider success.

**Non-Goals:**

- Improving the pose-atlas renderer, changing action extraction, or inventing choreography.
- Guaranteeing that a stochastic video model always follows the prompt.
- Retrying, reshooting, or automatically tuning prompts after a live failure.
- Replacing human visual review with a brittle semantic QA gate.

## Decisions

### 1. Compile one narrow action-execution brief inside the Phase 6 owner

Introduce a pure, versioned compiler owned by Phase 6. It consumes only the current chunk's canonical action groups, source lineage, duration/timing, terminal state, camera primary technique, and resolved media indexes. It emits structured JSON-safe data plus a deterministic compact rendering.

The rendering order is:

1. current shot and allowed action-group IDs;
2. immediate-start instruction;
3. ordered observable actions with body displacement/contact/weight transfer;
4. action completion window and semantic terminal state;
5. one supporting camera instruction;
6. prohibitions on opening-pose hold, future actions, clones, and guide-layout leakage.

Alternative considered: move the existing free-form motion-priority suffix earlier. Rejected because it remains derived from already-expanded prose, cannot prove group completeness, and has no independent hash or schema.

### 2. Use a two-layer prompt model

The internal task retains complete canonical visual, identity, continuity, spatial, prop, camera, and action contracts. The Provider transport receives deterministic projections:

- **Action projection:** lossless for current-Pxx action groups and timing.
- **Identity projection:** compact but lossless for character count and distinguishing visual signatures.
- **Continuity projection:** current start state plus only the immediately relevant prior anchor.
- **Camera projection:** one primary technique and only parameters that affect the current performance.
- **Style/negative projection:** deduplicated controlled clauses.

Full structured contracts and hashes remain in the task fingerprint and receipt. This separates audit completeness from attention-limited transport text.

Alternative considered: simply increase the prompt limit. Rejected because the Provider/model attention limit is not solved by accepting more text, and longer prompts worsen instruction competition.

### 3. Resolve media first, then compile the brief and prompt

The current chain constructs most prose before final atlas metadata and indexes are known. The revised composition first validates and orders required media, then creates the media-role preamble, compiles the brief using exact image/video indexes, and finally appends projected supporting contracts.

This changes composition order, not media ownership or upload order. Identity boards and first frames remain authoritative for their existing concerns; the prompt explicitly says those assets do not authorize a static hold. Motion references remain non-authority for identity and composition.

Alternative considered: move the atlas to the first media position. Rejected because it would violate the established identity-first media contract and could increase cloning/layout leakage.

### 4. Budget named prompt sections deterministically

Use the existing Provider capability profile to select a hard character limit. Reserve space in this order:

1. media-role preamble;
2. full action-execution brief;
3. mandatory compact identity cardinality/signatures;
4. required continuity and prop locks;
5. primary camera instruction;
6. deduplicated style and negative clauses.

Every section has a stable schema/version/hash and a deterministic renderer. Optional supporting clauses are omitted only by declared priority; arbitrary string-tail truncation is forbidden. If mandatory sections do not fit, preflight fails before task submission.

Alternative considered: token-count-based LLM summarization. Rejected because it adds cost, nondeterminism, another owner, and a new failure mode.

### 5. Version fingerprints without changing storage schemas

Add action-brief and prompt-projection fields to existing JSON-safe task metadata and receipts. Task identity includes all full source hashes and the rendered transport prompt hash, so a policy refresh creates a new task fingerprint rather than silently reusing an older request.

Existing run-02 remains immutable and audit-only. It is used only as a hash-verified Provider-deny replay input; no old file or receipt is rewritten.

### 6. Keep visual business verdict intentionally narrow

The live verdict asks whether the ordered action groups are visibly executed with meaningful whole-body/root displacement, required contact/prop interaction, and the semantic terminal state. It separately checks identity stability and absence of grid/text/clone contamination. It does not demand frame-identical timing or exact prose reproduction.

The verdict can be recorded by human review or a diagnostic observation, but no probabilistic model boolean may initiate retries or override deterministic lineage/budget failures.

## Risks / Trade-offs

- **[Risk] Compact identity prose could weaken visual consistency** → Preserve exact-one cardinality and distinguishing signatures, retain identity-board authority, and compare identity stability in offline and live acceptance.
- **[Risk] Action-first wording may produce excessive motion** → Require physically observable but source-bounded actions, preserve continuity start/end state, and forbid invented or future actions.
- **[Risk] One camera instruction may reduce cinematic variety** → Keep the canonical camera contract in audit metadata and select its primary technique deterministically; visual variety remains owned by shot planning, not repeated prose.
- **[Risk] Provider behavior remains stochastic** → Treat the change as a reliability improvement, require one-shot business acceptance, and never hide a failed motion verdict behind QA relaxation.
- **[Risk] Fingerprint changes prevent reuse of old tasks** → Mark old requests audit-only and require a fresh task identity; this avoids replaying known-bad prompt policy.

## Migration Plan

1. Add the versioned action-execution brief and prompt projection with focused unit tests.
2. Integrate it into both sequential/direct and continuity request composition through the existing Phase 6 owner.
3. Reconstruct run-02 with Provider submission denied and persist a new replay receipt that references, but does not alter, the original evidence.
4. Run focused, full, Graph/sequential parity, recovery, lint, and OpenSpec verification gates.
5. If all zero-cost gates pass, write a paid-admission preflight. A future live test requires a new explicit fee authorization and a fresh run directory.

Rollback is code-only: revert the independent commits. Existing tasks and receipts remain immutable and continue to identify their prompt policy by hash.

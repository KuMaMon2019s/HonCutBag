## Why

Sixteen paid full-chain acceptance runs consumed 188 recorded Provider requests without producing one accepted 36-second delivery. The latest failure is not a generation failure: Phase 3 asked Seedream to show one declared prop from several angles, then a non-ledgered VLM boolean gate interpreted those views as several logical props and stopped the run. HonCut needs a bounded reliability change before any further paid full-chain attempt.

## What Changes

- Version the Phase 3 prop-detail contract so a logical item count is distinct from the number of visual depictions used for a turnaround.
- Replace the prop-detail raw `passed` boolean hard gate with a typed Observation, append-only QA Ledger Decision, confidence, concrete evidence, and the existing tolerant visual QA policy.
- Treat schema, hash, lineage, declared item identity, and media-role corruption as deterministic blockers; treat ambiguous visual interpretation as diagnostic, acceptable deviation, or manual review according to policy.
- Add a bounded audit and source guard for Phase 1–5 VLM paths that can still turn model booleans or prose directly into blocking State without the canonical ledger/policy owner.
- Add provider-deny replay acceptance using saved media and saved structured observations so the failed run-16 asset can be re-evaluated without regenerating images or submitting another paid request.
- Prevent a new paid 36-second run until the replay gate, targeted regression, full regression, and recovery checks pass on the exact candidate commit.
- Preserve failed runs and v1 receipts as audit-only evidence; do not rewrite them as successful or use them to trigger retries.

### Scope

- Phase 3 prop-detail generation/QA contract and its direct Phase 3 consumers.
- Shared QA Ledger and visual policy integration only where required by this path.
- A repository-wide inventory guard for equivalent Phase 1–5 non-ledgered probabilistic hard gates.
- Offline replay, migration, recovery, and paid-run admission checks.

### Non-goals

- No Graph topology, Phase ownership, API, database architecture, frontend/backend, or Provider retry-policy change.
- No change to `pipeline_core.py`, Phase 6 media ordering, video generation, story semantics, or character roster rules.
- No automatic correction, paid redraw, paid re-review, reshoot, or retry.
- No new full-chain paid run as part of implementation; a later run requires a separately reviewed preflight and explicit authorization.

## Capabilities

### New Capabilities

- `visual-qa-reliability`: Defines typed, ledgered, confidence-aware visual QA behavior and the logical-item-versus-view contract for Phase 3 prop-detail boards.
- `acceptance-replay-gate`: Defines zero-Provider replay, recovery, evidence migration, and paid-run admission requirements for reliability fixes.

### Modified Capabilities

None. No main OpenSpec capabilities currently exist.

## Impact

- **Pipeline:** Phase 3 Character Factory and character reference QA; Phase 1–5 hard-gate inventory tests; run acceptance tooling.
- **Schemas/contracts:** A new prop-detail input/QA receipt version and a typed structured Observation version. Known v1 evidence becomes audit-only or is deterministically re-evaluated; unknown future versions remain fail closed.
- **Database:** No schema redesign. Existing append-only `qa_observations` and `qa_decisions` tables are reused.
- **Provider clients:** No transport or retry changes. Replay is provider-deny; any later live gate remains separately authorized and at most one request.
- **API/frontend/backend:** Not applicable.
- **Configuration:** No new production flag. Paid admission is an acceptance-tool policy, not a normal CLI bypass.
- **Architecture:** No owner or dependency-direction change. Phase 3 remains the owner; Graphify impact analysis identified `generate_character`, `batch_generate`, and `run_phase3` as direct consumers.

## Context

See `proposal.md` for the reliability problem. Serena traced the current failure through Phase 3 Character Factory into character reference QA and its strict understanding DTO. Graphify confirmed the direct impact path is `run_phase3 -> batch_generate -> generate_character -> prop-detail generation/QA`; downstream registry and quality gates consume the resulting board and receipt, while Graph topology, Phase 6 media, and `pipeline_core.py` are outside the owner.

The current generator explicitly asks for isolated handheld props from front, side, and three-quarter angles. The current reviewer has no representation for “one logical item, several depictions,” trusts aggregate booleans, and raises directly. Unlike four-view identity and performance-board QA, this path does not persist its Observation/Decision in `runtime.db`.

Implementation must start from the exact run-16 source commit `2702b6dda00f2d17ce3b9c6a30d97ecac514caff` on a clean `codex/phase3-prop-detail-qa-ledger-fix` branch. The current checkout and infrastructure changes are not a safe implementation baseline.

## Goals / Non-Goals

**Goals:**

- Make the prop-detail contract semantically consistent and recoverable.
- Reuse the existing append-only QA Ledger and tolerant visual QA policy.
- Prove the fix with saved production evidence before another paid run.
- Detect equivalent raw probabilistic hard gates before merge.
- Keep strict deterministic integrity checks intact.

**Non-Goals:**

- Do not change Phase ownership, Graph topology, Provider transports, Runtime retry policy, video media ordering, or story semantics.
- Do not introduce a general service/manager or a second QA framework.
- Do not make all visual defects pass; high-confidence evidence-backed defects still block.
- Do not mutate or resume run-16.

## Decisions

### 1. Version content identity separately from board presentation

Introduce a current prop-detail contract with two independent dimensions:

- `logical_items`: declared canonical item IDs and topology authority.
- `depictions`: one or more observed views associated with a declared logical item.

For each item, the typed Observation records whether the logical identity is present, whether all depictions are mutually consistent, depiction count, topology/material/color findings, confidence, and evidence. A visible count of three depictions is therefore not compared with canonical `component_count=1`.

**Alternative considered:** Require the image model to draw the prop only once. Rejected because the board is explicitly a geometry reference and needs turnaround views; it would reduce downstream usefulness and still leave the QA model unable to distinguish duplicates from views.

### 2. Make aggregate model verdicts diagnostic

The structured model may still return a summary for observability, but the Phase owner recomputes authority from per-item typed fields. The current visual QA policy decides `pass`, `acceptable_deviation`, `block`, or `manual_review`; the model cannot write Phase State directly.

**Alternative considered:** Add an exception string for “three angles.” Rejected because it hard-codes one layout and leaves the same failure for two, four, or mixed body-attached depictions.

### 3. Reuse QALedger without changing its database schema

Build the Observation fingerprint from board and canonical reference hashes, canonical contract hash, reviewer model, Prompt hash, and current DTO schema. Record the Observation once, then record the policy Decision using the existing policy hash and supersession rules.

Deterministic errors are supplied separately to policy: invalid schema, missing/unknown item IDs, hash/lineage mismatch, media-role mismatch, or budget violations. Semantic findings contain controlled category, confidence, and concrete evidence.

**Alternative considered:** Add prop-detail-specific database tables. Rejected because the existing ledger already provides append-only identity, dedupe, and policy supersession.

### 4. Do not regenerate run-16 media to prove the fix

Add an isolated provider-deny replay fixture/tool that reads a copied or externally referenced hash manifest for the run-16 board and canonical evidence. It creates a new current-version replay receipt and Decision without editing the failed run. If a new VLM Observation is strictly necessary, that belongs to a later separately authorized single-request live gate; it is not part of zero-request implementation acceptance.

The zero-request replay must support two modes:

1. Recompute policy from a stored typed fixture derived from the preserved evidence.
2. Reuse a matching existing ledger Observation when one exists.

Neither mode may claim fresh visual observation or mutate the historical result.

### 5. Guard the class of bug, not a single function

Add a bounded source/feature guard for production Phase 1–5 VLM paths. The guard identifies model review results that can reach a blocking exception or State patch without passing through a typed parser plus ledger/policy owner. It explicitly permits deterministic schema/hash/lineage/budget failures and migration adapters.

The initial inventory is reviewed with Serena references and Graphify impact evidence. Each finding is either migrated in this change when it is the same visual-QA class, or recorded as a separate owner/change when migration would expand scope.

**Alternative considered:** Rewrite all QA code in one change. Rejected because it would mix owners and make rollback and live acceptance unsafe.

### 6. Paid admission is a separate acceptance decision

Implementation completion produces `pending_live_acceptance`, not full acceptance. No new 36-second run is allowed until the exact candidate commit has:

- targeted and full regression passes;
- lint and diff checks;
- provider-deny replay of preserved evidence;
- ten recovery rounds with zero Provider requests;
- a completed hard-gate inventory;
- a no-submit preflight with finite limits.

A later live run remains separately authorized and cannot retry, redraw, reshoot, or expand scope automatically.

## Risks / Trade-offs

- **[Typed visual evidence can still be wrong]** -> Require controlled categories, explicit evidence, confidence thresholds, and manual review below threshold; preserve deterministic blockers.
- **[Accepting several depictions could hide real duplicates]** -> Evaluate same-item consistency and undeclared logical-item evidence instead of raw visible count.
- **[Legacy v1 data lacks new fields]** -> Preserve it audit-only and re-evaluate from verified media; never fabricate missing confidence or evidence.
- **[Source guard produces false positives]** -> Keep the allowlist limited to verified deterministic validators and test it against known ledgered QA owners.
- **[Replay passes but a new Provider output differs]** -> Replay is an admission gate, not proof of future image quality; the later narrow live gate remains required.
- **[Branch divergence reintroduces fixed Phase 1 failures]** -> Base implementation on run-16's exact commit and bind all receipts to the candidate commit.

## Migration Plan

1. Create the clean implementation branch from `2702b6d`; preserve current infrastructure changes separately.
2. Add current typed schemas, Prompt contract, parser, and unit fixtures before changing the owner.
3. Integrate the existing ledger/policy into the prop-detail QA owner and retain deterministic failures.
4. Mark v1 prop-detail receipts audit-only; add a separate replay/migration receipt without overwriting historical files.
5. Add the hard-gate inventory and source guard, then address only findings within the same Phase 3 visual-QA owner.
6. Run targeted tests, full tests, lint, diff check, provider-deny replay, and recovery matrix.
7. Update architecture documentation and Graphify only if implementation changes public schema or documented recovery behavior.
8. Roll back by reverting the independent commits; v1 receipts and failed runs remain unchanged throughout.

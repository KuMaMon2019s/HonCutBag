## 1. Freeze the implementation baseline and evidence

- [x] 1.1 Create `codex/phase3-prop-detail-qa-ledger-fix` from clean commit `2702b6dda00f2d17ce3b9c6a30d97ecac514caff`, verify the worktree has no unrelated changes, and record the baseline plus unchanged `pipeline_core.py` hash.
- [x] 1.2 Re-run the AGENTS.md Serena preflight for the exact baseline and verify definitions, callers, DTOs, tests, and direct consumers of the Phase 3 prop-detail QA owner.
- [x] 1.3 Run the narrow Graphify impact analysis for the exact baseline and verify the owner path remains `run_phase3 -> batch_generate -> generate_character -> prop-detail QA`, with Graph topology, Phase 6 media, Provider transport, and `pipeline_core.py` out of scope.
- [x] 1.4 Create a hash-only manifest for the preserved run-16 board, canonical contract, QA receipt, source run identity, and failed acceptance receipt; verify no media, database, receipt, or status in run-16 is modified.

## 2. Version the prop-detail contract

- [x] 2.1 Add the current prop-detail input/receipt schema that separates declared `logical_items` from their associated `depictions`, and verify one logical item can legally bind two, three or more consistent views.
- [x] 2.2 Extend the typed visual understanding DTO with per-item identity, depiction count and consistency, topology/material/color findings, controlled categories, confidence, and concrete evidence; verify unknown future schema and incomplete item coverage fail closed.
- [x] 2.3 Update the prop-detail generation and QA prompts to declare logical item IDs and explain that multiple views are depictions rather than duplicate props; verify prompt fingerprints change deterministically without adding story-specific constants.
- [x] 2.4 Update parsing and deterministic validation so invented IDs, missing declared IDs, invalid hashes, lineage mismatch, wrong media roles, and malformed schema block before policy evaluation.

## 3. Route Phase 3 visual QA through the ledger owner

- [x] 3.1 Replace the prop-detail aggregate model `passed` hard gate with a typed Observation persisted through the existing `QALedger`; verify the Observation fingerprint includes evidence, canonical contract, reviewer model, prompt, and schema hashes.
- [x] 3.2 Compute the authoritative Decision with the existing visual QA policy, preserving `>=0.65` pass/acceptable-deviation behavior and requiring `>=0.85`, a controlled category, and concrete evidence for semantic blocking.
- [x] 3.3 Preserve deterministic validation errors as unconditional blockers and verify ambiguous or low-confidence findings can only become diagnostics, acceptable deviation, or `manual_review`.
- [x] 3.4 Ensure `manual_review`, failure, timeout, and uncertainty never trigger redraw, re-review, retry, reshoot, or budget expansion; verify the Phase owner stops with persisted evidence.
- [x] 3.5 Verify `generate_character`, `batch_generate`, sequential Phase 3, and Graph Phase 3 share the same owner and cannot execute a parallel raw-boolean QA path.

## 4. Isolate legacy evidence and add zero-Provider replay

- [x] 4.1 Mark known v1 prop-detail QA receipts audit-only and add a deterministic re-evaluation input projection for hash-valid evidence; verify the historical receipt is never overwritten and future versions fail closed.
- [x] 4.2 Add a provider-deny replay entry point that reads only hash-verified saved media, artifacts, observations, decisions, and receipts; verify any Ark, Seedream, VLM, Seedance, or authoritative TOS boundary fails immediately with zero successful submissions.
- [x] 4.3 Replay the preserved run-16 board using a stored typed fixture or matching ledger Observation and write a distinct current-version replay receipt bound to the candidate commit, regression receipt, source run, hashes, policy, and Decision IDs.
- [x] 4.4 Repeat replay/recovery ten times and verify Provider counts stay zero while task counts, Observation IDs, Decision IDs, and source media hashes remain unchanged.
- [x] 4.5 Verify run-16 remains `live_acceptance_failed` and audit-only even if the isolated current policy accepts its preserved board.

## 5. Prevent equivalent probabilistic hard gates

- [x] 5.1 Inventory Phase 1-5 model/VLM review paths with Serena references and Graphify evidence, and classify each blocker as deterministic validation, ledgered policy, or an out-of-contract raw probabilistic gate.
- [x] 5.2 Add a bounded source/feature guard that rejects direct probabilistic `passed` or prose-to-exception/State paths while permitting verified schema, hash, lineage, role, budget, and migration validators.
- [x] 5.3 Migrate only same-owner Phase 3 visual-QA findings discovered by the inventory; record other owners as separate OpenSpec changes and verify no scope expansion occurs in this branch.

## 6. Regression and recovery verification

- [x] 6.1 Add regression tests proving one declared prop shown from front, side, and three-quarter views is one logical item and does not block solely because three depictions are visible.
- [x] 6.2 Add negative tests proving an undeclared distinct prop with high-confidence evidence blocks, while low-confidence item-count concerns remain diagnostic.
- [x] 6.3 Add deterministic failure tests for malformed/current/future schema, missing or invented item IDs, canonical/evidence hash mismatch, broken lineage, wrong media role, and budget overflow.
- [x] 6.4 Add ledger tests proving unchanged evidence is reviewed once, ten recoveries issue zero Provider requests, and a policy change appends a superseding Decision without a new review.
- [x] 6.5 Add Graph/sequential parity and caller coverage tests, plus a guard that verifies `pipeline_core.py` is unchanged and has no new production references.
- [x] 6.6 Run targeted pytest, the Phase 1-9 zero-Provider acceptance and recovery matrix, `make lint`, `git diff --check`, and `make test`; verify every result is bound to the same candidate commit.

## 7. Post-change validation and knowledge sync

- [x] 7.1 Use Serena to re-check every changed symbol, callers, interfaces, DTOs, diagnostics, and stale references; verify there is no duplicate or parallel QA implementation.
- [x] 7.2 Run `openspec verify stabilize-visual-qa-gates` (or the installed CLI's equivalent) and verify implementation, tests, replay receipts, and task completion comply with both capability specs.
- [x] 7.3 Run `graphify update .` and a depth-2 affected query for the Phase 3 QA owner; verify the incremental graph matches the actual changed path and save only durable, sanitized conclusions.
- [x] 7.4 Update `docs/HONCUT_ARCHITECTURE.md` and relevant feature tests only if the public schema or documented recovery behavior changed; otherwise record architecture docs and Serena Memory as not required.

## 8. Paid-run admission stop-loss

- [x] 8.1 Generate a signed/hash-bound `paid_admission` receipt only after the inventory, targeted/full regression, lint/diff checks, provider-deny replay, and ten-round recovery all pass on the exact candidate commit.
- [x] 8.2 If any zero-request gate fails, record `paid_admission_blocked` with the first failure signature and stop without requesting or submitting a new paid full-chain run.
- [x] 8.3 After all zero-request gates pass, prepare only a no-submit preflight with finite Provider-family limits; verify a future full-chain run still requires a separate explicit authorization and cannot retry, redraw, reshoot, or expand its frozen budget.

# Stabilize Phase 5 optional supervision gate

## Why

The Phase 1-5 visual-gate inventory found that the optional storyboard supervision owner can, when `supervision_blocking` is explicitly enabled, convert its model-level verdict directly into an exception without a typed QA Ledger Observation and deterministic policy Decision.

## Scope

- Define a strict supervision Observation DTO.
- Persist model evidence through the existing `QALedger`.
- Make the current aggregate verdict diagnostic and move authority to a deterministic policy.
- Preserve deterministic schema, source projection, hash, lineage, and budget blockers.

## Non-goals

- Do not change Phase 3 prop-detail QA in this change.
- Do not change the default non-blocking supervision configuration.
- Do not change Graph topology, Provider transport, retries, or correction budgets.

## Status

Recorded follow-up only. No implementation is authorized by the Phase 3 stop-loss change.

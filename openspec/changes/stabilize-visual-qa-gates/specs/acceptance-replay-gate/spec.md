## Purpose

Defines a zero-Provider replay and admission gate so reliability fixes are proven against preserved production evidence before HonCut is allowed to spend money on another full-chain acceptance run.

## ADDED Requirements

### Requirement: Reliability replay is provider-deny
The replay acceptance SHALL load hash-verified persisted media, structured artifacts, Observations, Decisions, and receipts while installing guards that fail on any text, image, VLM, video, or authoritative TOS submission attempt.

#### Scenario: Replay of run-16 prop-detail evidence
- **WHEN** the run-16 board, canonical contract, and v1 QA receipt hashes are valid
- **THEN** the replay evaluates the current contract without regenerating the board, uploading media, or submitting a Provider request

#### Scenario: Replay attempts network access
- **WHEN** any replay path reaches a Provider or authoritative upload boundary
- **THEN** the replay fails immediately and records the attempted family with zero successful submissions

### Requirement: Replay is bound to immutable evidence and code
Each replay receipt SHALL bind the candidate Git commit, regression receipt, source run identity, canonical contract hash, media hashes, input receipt hashes, policy hash, and resulting Decision IDs.

#### Scenario: Evidence changes after preflight
- **WHEN** any bound media or receipt hash changes between preflight and replay
- **THEN** replay fails closed and does not evaluate the changed evidence

#### Scenario: Candidate commit changes
- **WHEN** implementation code changes after regression evidence is produced
- **THEN** paid admission is invalid until regression and replay evidence are regenerated for the new commit

### Requirement: Recovery is side-effect stable
Repeated replay or recovery from the same boundary SHALL preserve task counts, Provider submission counts, Observation IDs, Decision IDs, and source media hashes.

#### Scenario: Ten recovery rounds
- **WHEN** replay is repeated ten times from the same Phase 3 boundary
- **THEN** Provider submission counts remain zero and all stable IDs and hashes remain unchanged

### Requirement: Failed historical runs remain audit-only
The system SHALL NOT resume, mutate, retry, or mark a failed paid run successful as part of replay. New replay receipts SHALL be stored outside the failed run's authoritative production receipts or as append-only audit evidence with distinct schema and identity.

#### Scenario: run-16 remains failed
- **WHEN** current policy accepts the preserved run-16 board during isolated replay
- **THEN** run-16 remains `live_acceptance_failed` and only the new replay receipt reports the isolated current-version verdict

### Requirement: Paid full-chain admission requires all zero-request gates
A new 36-second paid run SHALL be prohibited until targeted regression, full regression, lint/diff checks, the visual hard-gate inventory, provider-deny replay, and recovery matrix all pass on the exact candidate commit.

#### Scenario: Targeted tests pass but replay fails
- **WHEN** unit tests pass but the stored production evidence cannot pass provider-deny replay
- **THEN** the system records `paid_admission_blocked` and does not submit a new paid full-chain run

#### Scenario: All admission gates pass
- **WHEN** every required zero-request gate passes on the same commit and the later paid scope receives explicit user authorization
- **THEN** the acceptance tool may produce a no-submit preflight with finite hard limits; it still may not submit without that authorization

### Requirement: Paid admission never authorizes automatic expansion
Paid admission SHALL NOT authorize automatic retry, redraw, re-review, correction, reshoot, task-list expansion, or budget increase.

#### Scenario: A later live gate fails or becomes uncertain
- **WHEN** a later authorized request fails, times out, or remains submission-uncertain
- **THEN** execution stops, preserves the event, and requires a new decision rather than retrying or expanding scope


## Purpose

Defines reliable visual-quality decisions that preserve strict structural safety while preventing probabilistic model interpretations from contradicting the media contract or repeatedly blocking otherwise valid production assets.

## ADDED Requirements

### Requirement: Logical items are distinct from their visual depictions
The system SHALL represent the number of declared logical identity items separately from the number of angles, crops, or turnaround depictions used to show each item. Multiple consistent depictions of one declared item MUST NOT be treated as multiple logical items.

#### Scenario: One prop shown from three angles
- **WHEN** one declared handheld prop is shown in front, side, and three-quarter depictions on the same detail board
- **THEN** the system records one logical item with three depictions and does not block solely because three prop images are visible

#### Scenario: Undeclared distinct prop is present
- **WHEN** the board contains a visually distinct logical prop that cannot be reconciled to any declared item and the finding has concrete evidence at blocking confidence
- **THEN** the system blocks the board as an undeclared logical item

### Requirement: Visual observations are typed and non-authoritative
Every visual review that can advance or block Phase 3 SHALL produce a strict typed Observation with per-item identity, depiction consistency, topology, confidence, and concrete evidence. A model-supplied aggregate `passed` value or prose issue SHALL NOT directly determine production State.

#### Scenario: Aggregate verdict contradicts typed evidence
- **WHEN** the model aggregate verdict is negative but the typed per-item evidence confirms the declared item and only reports multiple consistent depictions
- **THEN** the Phase owner ignores the aggregate verdict for authority and computes the Decision from typed evidence and policy

#### Scenario: Structured observation is invalid
- **WHEN** the response is missing required typed fields, contains an unknown future schema, or cannot be parsed as one complete document
- **THEN** the system records or reports a deterministic schema failure and does not reinterpret prose as valid evidence

### Requirement: Visual decisions use the canonical tolerant policy
Visual Observations SHALL be persisted before a pure policy creates a Decision. A semantic score at or above 0.65 SHALL permit `pass` or `acceptable_deviation`. A negative semantic finding SHALL block only when confidence is at least 0.85, a controlled blocking category is present, and concrete evidence identifies the mismatch.

#### Scenario: Low-confidence item-count concern
- **WHEN** a reviewer expresses an item-count concern below 0.85 confidence or without concrete evidence distinguishing logical items from views
- **THEN** the concern remains diagnostic and does not authorize failure or paid correction

#### Scenario: High-confidence topology mismatch
- **WHEN** a reviewer identifies a different logical item topology with confidence at least 0.85, a valid blocking category, and concrete evidence
- **THEN** the policy produces a blocking Decision

### Requirement: Deterministic contract errors remain strict
Schema version, content hash, canonical contract hash, lineage, declared item ID coverage, media role, and request budget errors SHALL remain deterministic blockers independent of visual confidence thresholds.

#### Scenario: Board hash changes after observation
- **WHEN** the board content hash no longer matches the evidence hash bound to its Observation
- **THEN** the system fails closed before reuse or downstream consumption

#### Scenario: Declared item ID is absent from the observation contract
- **WHEN** a typed Observation omits a declared item ID or invents an unknown ID
- **THEN** the system blocks on deterministic coverage rather than guessing from names or prose

### Requirement: Observations and Decisions are recoverable without duplicate review
The system SHALL reuse an existing Observation when evidence, canonical contract, evaluator, Prompt, and schema fingerprints are unchanged. Policy changes SHALL append a superseding Decision without repeating the Provider review.

#### Scenario: Repeated recovery
- **WHEN** the same Phase 3 evidence is recovered ten times with an unchanged Observation fingerprint
- **THEN** no new Provider request occurs and the same Observation is reused

#### Scenario: Policy threshold changes
- **WHEN** the evidence and Observation remain unchanged but the policy fingerprint changes
- **THEN** the system appends a new Decision linked to the previous Decision and does not call the reviewer again

### Requirement: Paid correction is never inferred from ambiguous QA
An ambiguous, low-confidence, or manual-review visual Decision SHALL NOT trigger automatic redraw, re-review, retry, reshoot, or budget expansion.

#### Scenario: Manual review decision
- **WHEN** a valid Observation lacks enough confidence for pass or block
- **THEN** the run pauses with `manual_review` and preserves evidence without any automatic Provider request

### Requirement: Equivalent non-ledgered hard gates are prohibited
Phase 1–5 production code SHALL NOT convert a probabilistic model boolean or prose finding directly into blocking State outside the designated ledger and policy owner. Source guards and feature tests SHALL cover the allowed migration and deterministic validation boundaries.

#### Scenario: New raw VLM boolean gate is introduced
- **WHEN** production code adds a direct model `passed` check that can block a Phase without a ledgered Observation and policy Decision
- **THEN** the source guard or feature test fails before merge

#### Scenario: Deterministic parser rejects malformed data
- **WHEN** code rejects malformed schema, invalid hash, or incomplete lineage without consulting a probabilistic reviewer
- **THEN** the guard permits that deterministic failure path

### Requirement: Legacy visual QA evidence cannot silently satisfy the new contract
Known legacy prop-detail QA receipts SHALL remain immutable and SHALL be either marked audit-only or deterministically projected into a new re-evaluation input. Unknown future versions SHALL fail closed.

#### Scenario: Known v1 failed receipt with valid media
- **WHEN** a v1 receipt and its media hashes are intact
- **THEN** the system preserves the v1 receipt, creates separate current-version evidence for re-evaluation, and never rewrites the v1 result

#### Scenario: Future receipt version
- **WHEN** a receipt uses an unknown future schema
- **THEN** the system refuses production reuse and requires explicit migration support


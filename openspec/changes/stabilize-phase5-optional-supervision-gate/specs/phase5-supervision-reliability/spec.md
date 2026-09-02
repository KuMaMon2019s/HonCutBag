## Purpose

Prevent optional storyboard supervision from turning an unpersisted probabilistic summary into blocking authority.

## ADDED Requirements

### Requirement: Supervision verdicts are ledgered
When blocking supervision is enabled, the model SHALL produce a strict typed Observation that is persisted before a deterministic Phase 5 policy creates the authoritative Decision. Aggregate verdict text SHALL remain diagnostic.

#### Scenario: Model aggregate block lacks evidence
- **WHEN** the model returns `block` without a controlled category, sufficient confidence, and concrete evidence
- **THEN** Phase 5 does not treat the aggregate value as blocking authority

### Requirement: Existing structural blockers remain strict
Schema, source-projection hash, lineage, and budget failures SHALL remain deterministic blockers independent of model confidence.

#### Scenario: Projection hash changes
- **WHEN** the supervision input projection changes after its Observation is recorded
- **THEN** Phase 5 refuses reuse before any policy Decision

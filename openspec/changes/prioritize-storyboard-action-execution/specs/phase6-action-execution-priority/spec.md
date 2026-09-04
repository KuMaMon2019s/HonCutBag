## Purpose

Ensure every Phase 6 video request gives the current shot's complete canonical action sequence enough prompt priority and unambiguous media authority to produce visible movement without weakening identity, provenance, recovery, or cost controls.

## ADDED Requirements

### Requirement: Current-shot action execution brief
Before a video request can be submitted, the system SHALL deterministically compile one action-execution brief from the current Pxx canonical action groups, their source action-unit lineage, the shot time budget, the required terminal state, and the final media-role indexes.

The brief SHALL preserve every current-Pxx action group exactly once and in canonical order, SHALL identify observable body displacement, contact, weight transfer, and prop interaction when those facts exist, and SHALL exclude all later-Pxx actions.

#### Scenario: Complete ordered action sequence
- **WHEN** the current Pxx contains multiple canonical action groups
- **THEN** the action-execution brief contains every group exactly once in the same order with its source lineage intact

#### Scenario: Future action isolation
- **WHEN** the next Pxx contains additional actions
- **THEN** no next-Pxx action or atlas cell is included in the current request's action-execution brief

#### Scenario: Immediate motion after reference frame
- **WHEN** a first frame depicts the opening pose and the current shot requires subsequent movement
- **THEN** the brief requires movement to begin immediately after that frame rather than holding or returning to the opening pose for unused time

### Requirement: Action-first transport projection
The submitted Provider prompt SHALL place the complete action-execution brief immediately after the media index and media-role preamble and before verbose identity, continuity, style, negative, or camera material.

The system SHALL NOT truncate, summarize away, or move any canonical action group behind optional prompt material. Identity constraints SHALL continue to govern appearance, but SHALL NOT be expressed as a requirement to freeze the reference pose.

#### Scenario: Action appears before supporting contracts
- **WHEN** a Phase 6 prompt contains identity, continuity, camera, style, and action material
- **THEN** the media-role preamble and complete action-execution brief appear before the supporting contracts

#### Scenario: Optional detail exceeds budget
- **WHEN** the unprojected supporting contracts would exceed the Provider prompt budget
- **THEN** optional repetition is deterministically compacted while the complete action-execution brief remains unchanged

#### Scenario: Required action cannot fit
- **WHEN** the media preamble plus complete action-execution brief alone cannot fit within the Provider capability limit
- **THEN** the request fails before Provider submission with a deterministic budget error

### Requirement: Single authority per concern
The Provider-facing request SHALL express one motion authority, one concise identity projection, and one primary camera instruction. Full canonical identity, camera, lineage, and authority contracts SHALL remain persisted and fingerprinted even when their transport prose is compacted.

Camera guidance SHALL support the ordered body actions and SHALL NOT replace, delay, or contradict them.

#### Scenario: Duplicate camera contracts
- **WHEN** upstream prompt layers provide equivalent camera instructions more than once
- **THEN** the submitted prompt contains one deterministic primary camera instruction and the full source contract remains present in the receipt metadata

#### Scenario: Identity remains authoritative
- **WHEN** identity prose is compacted for transport
- **THEN** character count, stable identity, face, hair, body, outfit, and required prop signatures remain enforced and the canonical contract hash is unchanged

### Requirement: Explicit media-role isolation
The final media-role instructions SHALL state that identity boards govern identity, the first frame governs initial composition, current action crops/boards and the current pose atlas govern motion, and continuity anchors govern prior state.

The instructions SHALL prohibit copying grid layout, labels, arrows, borders, multiple reference bodies, or the opening pose hold into the video.

#### Scenario: Static identity references and dynamic atlas
- **WHEN** the identity board and first frame show a guard pose but the current atlas specifies block and evade actions
- **THEN** the request treats the guard pose only as identity/initial-state evidence and treats the atlas/action brief as motion authority

#### Scenario: Reference layout isolation
- **WHEN** a motion reference contains multiple poses, cells, arrows, or labels
- **THEN** the request requires one continuous character performance without clones, split panels, grids, arrows, or text in the output

### Requirement: Deterministic request identity and audit
The generation task fingerprint and receipt SHALL include the action-execution brief schema and hash, ordered action-group IDs, source action-unit lineage, final media-role indexes and hashes, prompt projection policy hash, final prompt hash, and canonical visual contract hash.

Reconstructing the same accepted inputs SHALL yield the same brief, prompt, fingerprint, and task count.

#### Scenario: Provider-deny replay
- **WHEN** a completed or failed request is replayed from hash-verified persisted evidence with all Provider submissions denied
- **THEN** the reconstructed prompt and fingerprint match deterministically and the Provider request count remains zero

#### Scenario: Changed action contract
- **WHEN** an ordered action group or its source lineage changes
- **THEN** the action brief hash and task fingerprint change before any Provider submission

### Requirement: Bounded action acceptance
Regression acceptance SHALL verify contract completeness, ordering, prompt placement, media authority, budget behavior, Graph/sequential parity, and Provider-deny replay before any paid test is admissible.

A paid live acceptance, when separately authorized, SHALL allow at most one video submission and SHALL record call-chain and business verdicts separately. Failure, timeout, uncertainty, or insufficient visible movement SHALL NOT trigger an automatic retry, reshoot, compensation request, or budget expansion.

#### Scenario: Visible action succeeds
- **WHEN** the single live output visibly completes the ordered current-Pxx action groups with meaningful body displacement and reaches the semantic terminal state without forbidden reference artifacts
- **THEN** both the call-chain and business verdicts may pass

#### Scenario: Static-pose output
- **WHEN** the output preserves identity but mostly holds or returns to the opening pose instead of visibly completing the action groups
- **THEN** the business verdict fails, the artifact remains audit-only, and no automatic Provider request follows

#### Scenario: Regression gate fails
- **WHEN** any zero-request regression or replay requirement fails
- **THEN** paid admission is blocked before submission

## Purpose

Define a bounded, reproducible way to test whether Seedance 2.0 can transfer large, ordered choreography from an identity-neutral temporal reference video before HonCut adopts that control route in production.

## ADDED Requirements

### Requirement: Deterministic temporal motion blueprint
The system SHALL compile a selected, hash-verified canonical Pxx action and camera contract into an identity-neutral motion blueprint video without invoking any Provider. The same inputs and compiler version MUST produce the same semantic blueprint fingerprint and media hash.

#### Scenario: Repeatable local compilation
- **WHEN** the same canonical Pxx contract, motion policy, duration, frame rate, and renderer version are compiled twice
- **THEN** both results contain the same actor tracks, event timing, camera track, semantic fingerprint, and byte-identical encoded media

#### Scenario: Unsupported or incomplete action contract
- **WHEN** an action cannot be mapped without inventing an actor, target, ordering, contact, or camera fact
- **THEN** compilation fails before any upload or Provider request and records the unsupported canonical references

### Requirement: CPU-compatible and identity-neutral generation
Blueprint generation SHALL run on the supported Mac mini environment without CUDA or local generative-model inference. It MUST use neutral geometry and MUST NOT contain a face, character texture, costume detail, final-scene styling, subtitle, logo, watermark, storyboard grid, or reusable identity pixel.

#### Scenario: Mac mini preflight
- **WHEN** the gate is prepared on the current Apple Silicon host without a CUDA device
- **THEN** local blueprint compilation and validation complete without downloading or loading a video-generation model

#### Scenario: Identity contamination
- **WHEN** the blueprint or its manifest contains character identity pixels or an identity authority role
- **THEN** preflight blocks the live gate

### Requirement: Observable large-motion evidence
Before a live request, the system MUST verify from the blueprint itself that every required dynamic action event has an ordered active interval, perceptible onset, a multi-phase motion curve, sufficient apex pose distance, sufficient peak root/joint speed, participation by the configured number of major joints, and an apex within the configured completion window. The encoded video MUST independently satisfy foreground occupancy, centroid travel, transition activity, and terminal-hold limits. Setup anchors MUST NOT satisfy a dynamic-action requirement. Thresholds MUST be versioned policy values rather than story-specific constants.

#### Scenario: Meaningful temporal action
- **WHEN** the compiled blueprint satisfies all versioned motion-amplitude, ordering, onset, and terminal-hold thresholds
- **THEN** preflight records the measurements and may admit the request projection

#### Scenario: Static or low-amplitude blueprint
- **WHEN** actor poses differ only by arrows, labels, camera motion, or sub-threshold joint movement
- **THEN** preflight fails with zero uploads and zero Provider submissions

#### Scenario: Slow endpoint drift
- **WHEN** a joint or actor root eventually crosses the displacement threshold but the movement is spread across the clip below the peak-speed or apex-timing threshold
- **THEN** preflight fails with zero uploads and zero Provider submissions

#### Scenario: Single-joint false positive
- **WHEN** only one distal joint moves materially while the torso and required major-joint set remain static
- **THEN** preflight fails with zero uploads and zero Provider submissions

### Requirement: Four-second single-action capability window
The initial capability gate SHALL compile and project exactly four seconds for one canonical dynamic action. A setup pose MAY appear only as a zero-story-time anchor capped by policy and MUST NOT consume a material share of the action window. The action phases MUST complete within the configured apex/recovery window and terminal hold MUST remain bounded.

#### Scenario: One canonical dynamic action
- **WHEN** the selected Pxx contains one supported dynamic action plus setup state
- **THEN** the gate compiles one four-second blueprint whose setup anchor is at most 0.15 seconds and whose remaining motion contains anticipation, peak, recovery, and terminal pose phases

#### Scenario: Additional action invention
- **WHEN** the compiler would need to add a strike, kick, contact, actor, or target absent from canonical lineage to make the clip appear busier
- **THEN** the gate rejects that mapping instead of inventing choreography

### Requirement: Seedance media-contract preflight
The gate MUST validate the blueprint against the current Seedance 2.0 reference-video limits and MUST freeze the exact model, duration, resolution, prompt hash, media order, media roles, source hashes, upload budget, submission budget, and task fingerprint before authorization can be consumed.

#### Scenario: Valid no-submit projection
- **WHEN** all canonical, lineage, media, TOS, model-capability, and budget checks pass
- **THEN** the gate writes a no-submit receipt with `pending_live_acceptance`, zero Provider submissions, and a maximum of one future video-generation submission

#### Scenario: Capability or budget mismatch
- **WHEN** the configured Seedance model does not accept a reference video or any frozen limit cannot be proven
- **THEN** the gate stops without silently converting the video to images, removing media, changing the model, or submitting a request

### Requirement: Single-variable production-equivalent request
The live request SHALL preserve the approved identity, initial composition, frozen four-second experimental duration, output profile, and current-Pxx semantic prompt while replacing the failed static motion-control responsibility with exactly one `reference_video` motion-blueprint input. The local control and candidate projections MUST use the same four-second duration. The final prompt media indices MUST match the actual submitted media sequence.

#### Scenario: Production-equivalent projection
- **WHEN** the live payload is built from the approved gate inputs
- **THEN** its production request builder, media roles, TOS upload path, GenerationTaskStore transition path, and fingerprint rules match the corresponding HonCut Phase 6 boundary

#### Scenario: More than one experimental variable
- **WHEN** the request also changes identity pixels, starting composition, duration, output profile, or unrelated prompt semantics
- **THEN** the gate rejects the comparison before submission

### Requirement: At-most-once paid execution
The live gate SHALL require a separate, current user fee authorization and SHALL permit at most one Seedance video-generation submission. Before transport it MUST atomically record `SubmissionAttempted` and `submission_uncertain`; failure, timeout, interruption, uncertain outcome, or Provider rejection MUST NOT trigger retry, redraw, reshoot, fallback, or budget expansion.

#### Scenario: Successful submission
- **WHEN** the Provider accepts the one authorized request
- **THEN** the existing task ledger records the Provider job ID and the gate may only poll and download that job

#### Scenario: Uncertain submission
- **WHEN** transport does not prove whether the Provider accepted the request
- **THEN** the gate remains `submission_uncertain`, performs no second submission, and requires manual adjudication

#### Scenario: Missing fee authorization
- **WHEN** the no-submit preflight passes but no current explicit fee authorization exists
- **THEN** the gate remains `pending_live_acceptance` and performs no Provider request

### Requirement: Falsifiable motion-transfer verdict
The gate MUST keep call-chain success separate from business motion transfer. A pass requires the generated video to satisfy the frozen, generic motion-transfer metrics and an explicit human verdict; successful submission or video download alone MUST NOT pass the capability.

#### Scenario: Material motion transfer
- **WHEN** the output preserves acceptable identity and composition while reproducing the required ordered events, large actor displacement, bounded onset, and bounded terminal hold without control-artifact contamination
- **THEN** the capability receipt records `capability_gate_passed`

#### Scenario: Valid video with inadequate motion
- **WHEN** the output is technically valid but remains mostly static, omits required action, performs it out of order, or spends too much time in an idle/guard pose
- **THEN** the receipt records `capability_route_paused` and the same route MUST NOT be retried automatically

#### Scenario: Single-actor result does not prove multi-actor control
- **WHEN** a one-actor gate passes
- **THEN** the receipt limits its evidence scope to one-actor choreography and does not claim multi-actor combat capability

### Requirement: No production activation from an unproven gate
The experimental blueprint and gate receipts MUST remain outside normal Graph/Lifecycle recovery and MUST NOT become ordinary Phase 6 media until this gate passes and a separate production-integration change is specified, implemented, and verified.

#### Scenario: Failed or pending gate
- **WHEN** the gate is pending, failed, uncertain, or paused
- **THEN** normal HonCut production behavior and existing media contracts remain unchanged

#### Scenario: Passed gate
- **WHEN** the gate passes
- **THEN** the result authorizes planning of a separate production-integration change but does not itself activate the route

### Requirement: Seedance-only provider scope
The capability gate and any later integration derived from its evidence SHALL use only the configured Seedance 2.0 Provider. The system MUST NOT submit to, fall back to, benchmark, or require credentials for another video-generation Provider.

#### Scenario: Seedance capability failure
- **WHEN** the one Seedance 2.0 result fails the frozen motion-transfer criteria
- **THEN** the route is paused without submitting the input to another Provider

#### Scenario: Alternate Provider is unavailable or configured
- **WHEN** another Provider is absent or present in the host configuration
- **THEN** that Provider has no effect on preflight, request projection, execution, or verdict

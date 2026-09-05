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

### Requirement: Four-second capability window
The capability gate SHALL compile and project exactly four seconds. A setup pose MAY appear only as a zero-story-time anchor capped by policy and MUST NOT consume a material share of the action window. Dynamic phases MUST complete within their configured contiguous windows and terminal hold MUST remain bounded.

#### Scenario: Canonical action combination
- **WHEN** the selected Pxx contains an eligible ordered action combination plus optional setup state
- **THEN** the gate compiles one four-second blueprint whose setup anchor is at most 0.15 seconds and whose dynamic actions execute consecutively without idle gaps

#### Scenario: Additional action invention
- **WHEN** the compiler would need to add a strike, kick, contact, actor, or target absent from canonical lineage to make the clip appear busier
- **THEN** the gate rejects that mapping instead of inventing choreography

### Requirement: Technique-specific code-owned choreography
Every supported dynamic primitive SHALL be compiled from a versioned code registry containing an ordered, technique-specific sequence of key poses, root/weight transfer, optional contact window, follow-through, and terminal recovery. A single generic interpolation curve with only different endpoint poses MUST NOT satisfy this requirement. The manifest MUST record the technique identifier, ordered phase identifiers, contact phases, registry hash, and deterministic keyframe fingerprint for every dynamic event.

#### Scenario: Distinct action techniques
- **WHEN** two different supported primitives are compiled from otherwise equivalent actor and timing inputs
- **THEN** their ordered phase identifiers and keyframe pose fingerprints differ, and each satisfies its own deterministic biomechanics assertions

#### Scenario: Contact is phase-local
- **WHEN** a canonical action declares prop or target contact
- **THEN** contact is visible only in the technique's declared contact phase or phases rather than throughout the whole event

#### Scenario: Prompt cannot substitute for choreography
- **WHEN** the Seedance request is projected
- **THEN** the prompt declares the blueprint's media responsibility and non-authority boundaries but contains no setup-duration, amplitude, peak, overshoot, or recovery tuning instruction
- **AND** removing those tuning phrases does not change the compiled blueprint, technique registry hash, or semantic measurements

#### Scenario: Legacy common-curve blueprint
- **WHEN** a v1 or v2 blueprint lacks the current technique registry and per-event keyframe evidence
- **THEN** it remains immutable audit-only evidence and cannot satisfy paid admission

### Requirement: Canonical combination density
The paid capability gate SHALL require at least three ordered dynamic actions, including at least two distinct techniques, from one hash-verified canonical Pxx. Setup anchors SHALL NOT count as actions. Dynamic intervals MUST be contiguous, each action duration MUST NOT exceed 1.25 seconds, the combination MUST sustain at least 0.75 actions per second, and the compiler MUST NOT invent, repeat, or stretch an action to fill time.

#### Scenario: Ordered combination candidate
- **WHEN** one canonical Pxx contains at least three ordered dynamic action groups with valid source-action lineage and at least two distinct techniques
- **THEN** the blueprint preserves their order, assigns contiguous bounded windows, records zero inter-action gap and combination density, and may proceed to request-equivalence preflight

#### Scenario: Insufficient dynamic cadence
- **WHEN** the receipt-bound Pxx contains fewer than three dynamic actions plus any setup anchors
- **THEN** it is classified `combination_ineligible` and cannot satisfy paid admission, regardless of its amplitude

#### Scenario: Another canonical Pxx is eligible
- **WHEN** the receipt-bound Pxx is ineligible but another Pxx in the same verified continuity plan contains a valid combination
- **THEN** the system may compile that Pxx as zero-provider local evidence but MUST stop at `pending_source_request_projection` until a production-equivalent request receipt for that exact Pxx exists

#### Scenario: Artificial combination
- **WHEN** a combination would require duplicating one action, inventing another primitive, or borrowing an action from another Pxx
- **THEN** the gate fails before upload and records no paid-admission projection

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

## Context

See `proposal.md` for motivation. HonCut currently sends static identity, first-frame/continuity, performance-guide, and neutral storyboard-guide media to Seedance 2.0. A production-equivalent gate has shown that a static pose atlas can be transported correctly yet fail to cause material choreography. The host is a Mac mini M4 and is not a supported environment for local CUDA video-model inference. The user has paid for Seedance 2.0 only, so no alternate cloud or local generation model is in scope.

Seedance 2.0 already accepts `reference_video` inputs through HonCut's TOS and Phase 6 transport path. The experiment therefore needs no local generative model and must not introduce a second task ledger, upload client, retry loop, or Graph path.

## Goals / Non-Goals

**Goals:**

- Test temporal video conditioning as a different control channel, not as another pose-atlas tuning attempt.
- Produce a deterministic, inspectable driving clip cheaply on Apple Silicon.
- Make the paid result falsifiable after one request.
- Ensure a failed experiment leaves production behavior untouched.

**Non-Goals:**

- Photorealistic local rendering or local diffusion inference.
- General text-to-motion research, arbitrary mocap generation, or a comprehensive animation library.
- Proving multi-character contact choreography with a single-character gate.
- Integrating a new cloud Provider or changing existing Phase ownership in this change.
- Evaluating, calling, or falling back to any non-Seedance video model.
- Automatically promoting a successful experimental artifact into a production run.

## Decisions

### 1. Compile a video from the canonical temporal contract, not from atlas pixels

The local compiler will consume one current Pxx's canonical actor/action ordering, duration, camera intent, and already-versioned pose constraints. It will map supported normalized action classes to a small data-driven motion-primitive registry, interpolate joints and actor roots at 24 fps, and render neutral silhouettes with an optional neutral prop line and contact markers. Camera transforms will be applied to the rendered coordinate system.

The compiler will not animate the existing guide bitmap or infer motion from red/blue arrows. This avoids treating annotation pixels as temporal evidence and makes joint/root displacement measurable before payment.

Alternative considered: interpolate the nine atlas cells. Rejected because the cells have already exhibited insufficient pose variation and do not encode timing or contact.

### 2. Keep the first experiment acceptance-only

The compiler and live entry point will be reachable only from a dedicated acceptance script. They will consume hash-verified external/canonical evidence and write to an isolated output directory. Normal CLI, Lifecycle, Graph nodes, checkpoints, and Phase 6 execution will not discover these artifacts.

This preserves the current production contract until capability is demonstrated. If the gate passes, a second OpenSpec change will decide whether the production owner belongs under Phase 2's previs domain with Phase 6 as a strict consumer.

Alternative considered: immediately add `motion_blueprint` to every Phase 6 request. Rejected because it would commit the architecture before the Provider capability is proven.

### 3. Reuse existing transport and persistence owners

The request projection will call the same Phase 6 prompt/media builder and Seedance adapter used by production, upload inputs through the existing TOS owner, and account for the video-generation request through `GenerationTaskStore`. A narrow acceptance-only dependency injection may select the frozen blueprint as a `reference_video`; it must not create a parallel Provider client or expose a general fake-Provider switch.

The exact symbols and minimum change set will be finalized during the required Serena and Graphify preflight before implementation.

### 4. Use a two-stage verdict

Stage A is deterministic: validate source lineage, blueprint encoding, actor/event coverage, frame timing, root/joint displacement, onset, terminal hold, media limits, request equivalence, and one-request accounting. A Stage A failure costs zero Provider requests.

Stage B is the one-request capability result. Call-chain status records acceptance/download/validation. Business status records measured output motion plus explicit human review. The implementation may use local pose tracking as diagnostic evidence, but an LLM/VLM score will not directly control the verdict.

### 5. Freeze generic motion thresholds in a versioned policy

Thresholds will be expressed in normalized actor/body or frame coordinates so they generalize across aspect ratios and stories. The initial policy will bound action onset and terminal idle fraction and require per-event joint/root deltas. Exact values will be selected from the prior accepted/failed evidence during implementation preflight and frozen in the regression fixture; production narrative strings will not be used as constants.

### 6. Stop after the first conclusive failure

The same fingerprint may produce at most one paid Seedance 2.0 submission. A technically valid output that fails motion transfer pauses this route. A timeout or uncertain transport state is not a capability result, but it still forbids resubmission under the same authorization and fingerprint. No alternate Provider is attempted after any outcome.

### 7. Treat the first local blueprint as a failed admission candidate

The first renderer allocated almost the entire nine-second duration to one smooth transition from `ready` to an `evade` endpoint. Its endpoint displacement passed, but the contact sheet and decoded video showed low perceived velocity and little full-body articulation. The original blueprint, manifest, regression receipt, and no-submit receipt remain immutable audit evidence. They cannot satisfy the revised paid-admission policy.

The corrected single-action gate uses a four-second experiment window. This is an acceptance projection, not a rewrite of the source Pxx story clock. The equivalence checker compares a frozen four-second control projection and candidate projection that are identical except for replacing the static motion-control atlas with one blueprint `reference_video`. Identity, initial composition, output profile, prompt semantics, and all unrelated media remain unchanged.

### 8. Compile each dynamic primitive as a phase curve

Setup primitives (`ready`, `prop_hold`) are zero-story-time anchors capped at 0.15 seconds and do not satisfy the dynamic-motion requirement. Every dynamic primitive is represented by deterministic normalized phases:

1. anticipation/counter-motion;
2. explosive peak with amplified root and joint excursion;
3. overshoot or counterbalance;
4. canonical terminal pose.

The phase coefficients are generic, versioned policy data. They may amplify an existing canonical action class but cannot invent another action, actor, target, contact, or order. A source contract with one dynamic action remains a single-action capability test.

### 9. Admit perceptible kinetics, not endpoint distance

Policy v2 requires all of the following for every dynamic event: a perceptible onset threshold, minimum peak root or joint speed, minimum count of participating major joints, minimum apex pose distance, apex before the configured completion fraction, and a bounded terminal hold. Setup anchors are measured separately and cannot rescue a failed dynamic event.

The encoded blueprint is decoded again before admission. The gate measures foreground occupancy, foreground-centroid travel, frame-difference activity, and a high-percentile transition magnitude. Semantic and rendered measurements must both pass. This prevents a normalized joint value, camera motion, or one moving wrist from passing a visually static blueprint.

### 10. Make action technique a versioned code contract

The v2 amplitude correction still applies the same normalized anticipation/peak/recovery curve to every dynamic primitive. Different endpoint poses produce larger motion, but they do not prove that an evade, kick, strike, block, grapple, throw, or locomotion sequence is being demonstrated as a recognizable technique.

Policy v3 replaces that shared curve with a data-driven technique registry. Each dynamic primitive owns an ordered set of named key phases with normalized time, root translation, pose progression, joint-specific offsets, and optional contact state. The compiler interpolates between those immutable key phases and records the registry hash plus each event's ordered phase IDs and keyframe fingerprint. Generic thresholds remain a separate admission layer; they cannot stand in for technique correctness.

The registry may only articulate the canonical primitive already selected by upstream evidence. It cannot create another action, actor, target, prop, contact relationship, or event order. Setup primitives remain bounded anchors and do not become dynamic techniques.

### 11. Keep prompt text declarative

Seedance prompt text will identify the reference video as the sole authority for current-Pxx motion timing, body kinematics, contact timing, and camera trajectory, while denying identity, costume, scene-pixel, or annotation authority. It will not prescribe numeric setup timing, explosive amplitude, anticipation, overshoot, recovery, or similar tuning language. Those facts belong to the v3 compiler, manifest, and pixels. This makes behavior reproducible from code and prevents conversational wording changes from acting as an undocumented control plane.

## Risks / Trade-offs

- **[Seedance treats a neutral rig video as loose inspiration]** → The single request directly tests this; failure pauses the route rather than spawning prompt iterations.
- **[Primitive mapping invents choreography]** → Only supported normalized action classes may compile; unknown or incomplete contracts fail before upload.
- **[Motion transfers but identity degrades]** → Identity/composition checks remain part of the business verdict and separate from motion measurements.
- **[One actor passes but combat still fails]** → Evidence scope is explicitly single actor; multi-actor motion requires its own later gate.
- **[Local encoding varies across FFmpeg builds]** → Fingerprint semantic frames separately and pin encoding parameters; byte identity is required within the supported test environment.
- **[Dirty current workspace contaminates implementation]** → Apply work must use a clean worktree based on an explicitly recorded commit and must not mix unrelated infrastructure or prototype files.

## Migration Plan

1. Implement and verify the acceptance-only compiler and no-submit gate without production activation.
2. Generate a fresh isolated gate directory and freeze the request projection.
3. Stop at `pending_live_acceptance` until the user explicitly authorizes the one paid request.
4. On failure or uncertainty, persist the receipt and leave production unchanged.
5. On pass, retain the evidence and open a separate production-integration OpenSpec change; rollback of this experiment is deletion of the acceptance-only code and tests, with receipts retained as audit evidence.

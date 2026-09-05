## Why

HonCut has now falsified the assumption that a single static pose-atlas image can reliably make Seedance execute large, ordered body choreography: the production-equivalent request was accepted, but the required motion did not transfer. The next bounded experiment must test the materially different control channel that Seedance 2.0 documents for complex action and camera reference—a temporal reference video—without committing the production pipeline to another unproven route.

The first zero-provider blueprint candidate also falsified its own admission gate: it stretched one `evade` endpoint across most of nine seconds and passed because one joint or the actor root crossed a cumulative displacement threshold. Its average dynamic speed was visually small. That artifact is audit-only and MUST NOT consume a fee authorization. The corrected gate must prove perceptible, full-body, time-localized motion before any upload.

The later four-second v3 candidate exposed a second selection flaw: the gate remained bound to the prior P01 receipt even though that Pxx contained only one dynamic action. The compiler could make that action larger, but it could not demonstrate a combination and filled nearly the whole clip with one move. A combination capability gate must select canonical multi-action evidence, preserve its order and lineage, and reject single-action evidence rather than stretching or inventing choreography.

## What Changes

- Introduce a provider-denied local compiler that turns an existing canonical Pxx action/camera contract into a short, identity-neutral `motion_blueprint.mp4` with explicit temporal ordering, large pose deltas, root travel, contact markers, and camera motion.
- Require the paid-admission candidate to contain at least three ordered dynamic action groups, including at least two distinct techniques, backed by canonical lineage. Setup anchors do not count; one- or two-action Pxx evidence remains locally inspectable but is ineligible for the fast-combination gate.
- Allocate the four-second active window across the canonical combination with no inter-action idle gap, a maximum per-action duration of 1.25 seconds, early technique apexes, and a bounded terminal hold. The compiler must never stretch one move merely to fill the clip.
- Freeze a four-second experimental window, limit setup to a brief zero-story-time anchor, and compile every dynamic primitive as anticipation → explosive peak → overshoot/recovery → terminal pose rather than one slow endpoint interpolation; paid admission additionally requires a canonical combination.
- Replace the shared generic phase curve with a versioned, code-owned technique registry. Each supported action primitive must define its own ordered key poses, weight transfer, contact window, follow-through, and recovery so the blueprint visibly demonstrates the action technique rather than merely enlarging a common movement envelope.
- Remove timing/amplitude tuning prose from the Seedance request. The prompt may only declare media authority and contamination constraints; choreography timing and technique evidence must come from the compiled blueprint contract and pixels.
- Replace cumulative endpoint-only admission with versioned perceptual kinetics: perceptible onset, peak root/joint speed, multi-joint participation, apex timing, screen-space displacement, actor occupancy, and bounded terminal hold must all pass.
- Introduce a dedicated Seedance 2.0 motion-blueprint capability gate with a no-submit preflight, immutable request projection, one-request hard limit, `submission_uncertain` accounting, and no retry, redraw, reshoot, or budget expansion.
- Compare the generated video against the frozen motion blueprint using deterministic measurements plus a separately recorded human business verdict; model-generated semantic prose cannot directly pass the gate.
- Pause the Seedance motion-blueprint route if the single production-equivalent request does not demonstrate material temporal transfer. Production integration is permitted only after the capability gate passes.
- If the gate passes, define the later production contract in which the static storyboard guide remains an audit/semantic artifact while the verified temporal blueprint becomes a distinct Phase 6 motion-reference role with full lineage and fingerprinting.
- Preserve existing Phase ownership and execution topology: no production implementation is part of the capability-gate proposal itself, no retry policy changes, and no changes to `pipeline_core.py`.
- **Non-goals:** inventing or repeating story actions merely to fill time; claiming that one canonical action proves combination choreography; local video-model inference; CUDA/GPU deployment on the Mac mini; evaluating or integrating any non-Seedance Provider; replacing character identity boards or cinematic first frames; solving multi-actor choreography from a single-actor result; running a 36-second acceptance; automatic provider fallback; or claiming general production support from a failed/inconclusive gate.

## Capabilities

### New Capabilities

- `seedance-motion-blueprint-gate`: Deterministic temporal-blueprint compilation and a one-request, falsifiable Seedance 2.0 capability gate for large ordered motion.

### Modified Capabilities

None. Existing production requirements remain unchanged until the new capability gate passes and a separate integration change is proposed.

## Impact

- **Pipeline:** adds an isolated Phase 2/Phase 6 acceptance projection; does not alter Graph topology or normal production behavior in this change.
- **API/provider:** exercises only the existing paid Seedance 2.0 `reference_video` transport contract with one new controlled media responsibility; no public API change and no alternate Provider path.
- **State/artifacts:** adds versioned, JSON-safe motion-blueprint and capability-gate receipts outside normal production recovery; no database schema change is expected for the experiment.
- **Configuration:** requires the existing Seedance/TOS configuration only at live-gate time; preflight remains zero-provider.
- **Frontend/backend:** no frontend and no Route/Controller/Service/Repository impact.
- **Database:** no schema change; the existing `GenerationTaskStore` and transport guard remain the submission-accounting owners.
- **Architecture:** the experiment respects current owners. A successful result would justify a separate architecture change for production consumption; a failed result pauses the route without production changes.

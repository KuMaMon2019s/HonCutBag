## Why

The current Phase 6 request can contain a valid storyboard pose atlas and complete action lineage yet still produce a mostly static guard pose. The run-02 evidence shows why: the executable action timing appears near the 16,000-character transport limit, after duplicated identity and camera prose, while the identity board and first frame visually reinforce the opening pose. Continuing to change Phase 2 drawings or relax QA would treat the symptom and spend more Provider calls without improving execution reliability.

## What Changes

- Add a deterministic Phase 6 action-execution brief compiled only from the current Pxx canonical action groups, timing window, terminal state, and media-role indexes.
- Place the complete action brief immediately after the media-role preamble so it cannot be displaced or truncated by identity, continuity, or camera detail.
- Project canonical identity and camera contracts into concise Provider-facing forms while retaining their complete structured contracts, hashes, authority roles, and lineage in task fingerprints and receipts.
- Deduplicate camera instructions and make character action execution higher priority than camera flourish; camera guidance must support, not replace, observable body displacement and contact.
- Make identity-board and first-frame roles explicit: they govern identity/composition but do not authorize holding the opening pose. The current pose atlas remains the motion authority.
- Add a deterministic prompt-budget policy that preserves the full current-Pxx action brief and fails before submission if required action semantics cannot fit.
- Add Provider-deny replay of the failed run-02 request and regression checks for prompt ordering, action-group coverage, media hashes, Graph/sequential parity, and zero Provider requests.
- Add a bounded future live acceptance contract that evaluates both call-chain success and visible action completion without automatic retry, reshoot, or budget expansion.

### Scope

- Phase 6 Provider request composition, prompt projection, media-role instructions, task fingerprinting/receipts, and their regression/live-acceptance tooling.
- Existing canonical action groups, pose-atlas provenance, continuity state, and media assets are inputs; their ownership is unchanged.

### Non-goals

- Do not modify Phase 2 pose-atlas rendering, action extraction, or Gxx/Axx allocation.
- Do not change Graph topology, Provider transport/retry policy, GenerationTaskStore state transitions, or Phase ownership.
- Do not change image-generation assets, loosen deterministic validation, add story-specific constants, or modify `pipeline_core.py`.
- Do not issue a paid Provider request as part of this planning change.

## Capabilities

### New Capabilities

- `phase6-action-execution-priority`: Defines how Phase 6 preserves and prioritizes current-shot action execution across prompt composition, media authority, budgeting, replay, and acceptance.

### Modified Capabilities

None.

## Impact

- **Pipeline:** Phase 6 prompt construction and packaging in `video_generator.py`, `continuity_provider.py`, and `asset_packager.py`; the exact implementation boundary will be kept within the existing Phase 6/domain and Runtime ownership rules.
- **Artifacts and persistence:** No database schema change. Existing complete canonical/action contracts remain authoritative; request fingerprints and receipts gain deterministic action-brief/projection metadata.
- **Provider/API:** No Provider API or transport change. The submitted textual projection becomes shorter and action-first; media count and upload behavior remain unchanged.
- **Frontend/backend:** Not applicable; HonCut is a local pipeline and this change has no frontend contract.
- **Configuration:** A capability-owned deterministic prompt budget may use the existing model capability profile; no user-facing retry or QA relaxation switch is introduced.
- **Architecture:** Not an architecture change. Owners, dependency direction, Graph composition, and recovery semantics remain unchanged.
- **Tests/docs:** Phase 6 prompt, continuity, packaging, replay, live-acceptance, Graph/sequential parity, and architecture behavior documentation are affected.

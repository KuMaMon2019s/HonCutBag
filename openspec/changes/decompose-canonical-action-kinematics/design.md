## Context

See `proposal.md` for motivation. The current production chain already has one upstream body-action owner in `utils/body_action_contracts.py`, called by Event Extractor, Adaptation, Storyboard planning and chunk scoping. Its v1 contract retains complete prose mechanics but no joint/channel timeline. Phase 2 therefore classifies that prose into a small pose family and interpolates from one generic origin pose to one generic target pose; Phase 3 independently extracts a smaller set of pose constraints; Phase 6 again summarizes the Phase 2 result. The acceptance motion-blueprint compiler contains a richer but separate technique registry, so it can demonstrate motion that production does not actually own.

This is a schema and cross-Phase consumer change. It must preserve Phase 1 action ownership, Pxx/action capacity, Graph topology, Provider policies and the existing JSON-safe State boundary.

## Goals / Non-Goals

**Goals:**

- Give every source body-action beat one deterministic, hash-bound kinematic timeline, then project it exactly onto final GAU/Pxx units after production grouping is known.
- Preserve independent performer motion through ordered `actor_tracks` for simultaneous attack, defence and contact actions.
- Make Phase 2, Phase 3, Phase 6 and the local blueprint compiler consume that same timeline.
- Preserve source/action/Pxx lineage and keep all compilation/migration zero-Provider.
- Make compound and fast actions visibly use the whole body without turning kinematic subphases into new story actions.

**Non-Goals:**

- No new LLM, image or video request and no Provider-side retry or Prompt-tuning loop.
- No inverse-kinematics solver, physics engine, GPU model or character-specific motion capture system.
- No change to story selection, action ordering, duration layout, actor identity, camera ownership or Graph topology.
- No attempt to infer an authored flip, spin, contact result or exact side when existing canonical facts do not support it.
- No activation of the experimental motion blueprint as a normal Phase 6 reference video or production media role.

## Decisions

### 1. Extend the existing body-action owner with two explicit compilation stages

`body_action_contracts` remains the single owner. A narrow pure module may hold typed enums, fixed-point geometry primitives and the deterministic projector, but it is invoked only by that owner.

The first stage embeds one source `kinematics` record per body-action beat, keyed by `micro_action_index`, because final generation action units and Pxx do not yet exist at Event Extractor time. After Adaptation/Storyboard has finalized GAU/Pxx grouping, the second stage builds one production `kinematics_projection` per GAU. The projection lists the exact ordered `source_micro_action_indexes`, Pxx, time slice and hashes it covers. Within a Pxx, these source-index sets must be disjoint and their union must equal the final action-unit lineage. A source beat record can never masquerade as a final GAU projection.

Alternative considered: compile richer motion only in Phase 2 or the acceptance blueprint. Rejected because this recreates the present divergence and lets each consumer invent different motion.

### 2. Use ordered actor tracks, an actor-local coordinate system and complete channel inventory

Each source record and production projection contains `actor_tracks` sorted by stable performer ID. A simultaneous attack/defence, grapple or contact action therefore has separate tracks for every canonical performer instead of compressing both bodies into one skeleton. Each track uses actor-local axes: +X actor-right, +Y up, +Z actor-forward. It stores these channels in stable order:

1. root
2. waist/torso
3. head
4. left arm
5. left hand
6. right arm
7. right hand
8. left leg
9. left foot
10. right leg
11. right foot

Arm/leg and hand/foot remain separate because joint-chain movement and distal orientation/contact are different facts. Every phase of every actor track contains all channels. A channel stores both its semantic role and a deterministic render state: fixed-precision normalized translation/rotation or joint targets, activation, amplitude, support and contact. A channel that is not actively moving is explicitly `inherit`, `stabilize`, `support`, or `balance`; `unspecified` semantic evidence still resolves to one of those non-inventive render states, so absence is never interpreted as hidden motion.

Generic amplitude classes carry story-neutral numerical floors for root displacement, waist rotation and active major joint chains. These floors apply only when the existing body mechanics already classify the action as a full-body attack, dodge, turn, flip or translation. They do not enlarge an ordinary still/guard beat.

Alternative considered: store only moving joints. Rejected because omitted channels then become ambiguous during interpolation and make left/right stability impossible to validate.

### 3. Compile generic biomechanical phases from the unchanged Provider DTO, with no extra model call

The Provider-facing `BodyActionUnderstanding` DTO and JSON schema remain unchanged. The compiler consumes its existing strict fields `technique`, `side`, `limbs`, `footwork`, `torso`, `weight_shift`, `direction`, `contact` and `end_pose`. It maps them through a versioned, story-neutral primitive vocabulary such as plant/release, flex/extend, reach/retract, shift, rotate, translate, contact, follow-through and settle. Dynamic actions use the applicable ordered phases from load → drive → apex/contact → follow-through → settle; phases that do not apply are omitted.

The compiler records, per value, whether it came from explicit structured mechanics or deterministic biomechanical projection plus the evidence hash. It may complete balance/support behavior required to make the existing movement executable, but it may not invent another attack, target, prop, flip, outcome or plot fact. Semantic QA remains diagnostic; malformed IDs, enums, hashes or phase order remain strict.

Alternative considered: require the Event Extractor LLM to return a verbose joint timeline. Rejected because it expands the longest structured response, increases stochastic schema failure and repeats information already present in the body-mechanics score.

### 4. Model Pxx-local orientation and transform separately from camera projection

There is no reliable absolute world-facing frame in the current persisted artifacts, so the contract does not invent one. Each actor track anchors its Pxx start pose at relative yaw `0` and stores continuous relative actor rotation from that anchor. When a verified canonical camera path is available, consumers derive front, back, left profile, right profile and three-quarter view relations; otherwise camera relation remains controlled `unspecified`. Camera motion changes only this derived view projection and never mutates actor-local body channels.

Spatial transforms are a separate optional record:

- kind: none / turn / spin / flip
- axis: yaw / pitch / roll
- direction: positive/negative or clockwise/counterclockwise in the declared coordinate space
- amount: bounded angle or turn count
- root trajectory and airborne/support intervals
- Pxx-local start/end orientation and landing/support state

An exact angle or turn count is authoritative only when an existing canonical fact supplies it. Otherwise an explicitly known transform kind may carry a controlled range or `unspecified` amount; it may not invent precision. Flip/spin is emitted only when existing canonical fields explicitly contain that movement. Plain camera rotation, view mirroring, dodge or ordinary turn cannot promote it.

### 5. Preserve semantic action capacity while adding intra-action timing

Kinematic phases are children of one generation action unit. They receive normalized progress and relative weights inside that unit, not independent story-clock or Provider-capacity slots. Pxx timing assigns the existing action group a duration, and the consumer scales the normalized phase windows into that duration. Adjacent action groups inherit the previous terminal channel state rather than restarting from neutral/guard.

This separates “how one move is physically executed” from “how many story actions fit in the clip”. A fast sequence shortens phase windows while retaining load/apex/contact/follow-through landmarks; it does not delete actions or create idle padding.

### 6. Make downstream consumers projections, not secondary interpreters

- Phase 2 consumes the final GAU/Pxx projection, samples each actor track at Gxx/atlas progress points and projects actor-local joints through the canonical camera path. Body-action rows no longer use `_POSES` as their movement authority; non-body/spatial rows may retain existing geometry.
- Phase 3 builds pose constraints and skeleton guides from the same sampled channel states. Prompt text is a compact projection and includes the kinematics hash rather than re-parsing the action prose.
- Phase 6 action-execution brief carries ordered phase IDs, active channels, orientation/transform summaries and the kinematics hash. It still references the same canonical action groups and media roles.
- The local motion-blueprint gate uses these canonical phases directly only inside its isolated acceptance path. Its independent technique registry becomes legacy audit-only compatibility code and cannot satisfy current admission. Ordinary Phase 6 media ordering and roles remain unchanged: this change does not add the blueprint as `reference_video` or activate the still-unproven production route.

Alternative considered: pass all raw channel JSON into every Provider Prompt. Rejected due prompt bloat and because media plus compact visible mechanics are the Provider-facing projection; the full contract remains local authority and fingerprint evidence.

### 7. Version every affected persisted boundary and migrate only provable evidence

Implementation advances the body-action contract, Event Flow cache identity, secondary storyboard/pose contract, performance pose constraints, Phase 6 action brief and blueprint manifest versions as required by actual serialized fields. Fingerprints include the kinematics schema, policy hash and contract hash.

The v1 body-action contract is embedded in Event/Storyboard artifacts rather than stored as an independent Artifact. Known old data may therefore be compiled only into a sidecar bound to its containing parent Artifact, and only when all structured mechanics, final action/Pxx lineage and source hashes verify. The parent source artifact is never overwritten and becomes audit-only; every downstream asset derived from it becomes stale/audit-only. Ambiguous or incomplete mechanics cannot be upgraded by prose guessing and must be rebuilt from the owning Phase. Unknown future versions fail closed.

This versioning does not alter `BodyActionUnderstanding` or require reissuing its model request when verified legacy structured evidence is sufficient.

### 8. Keep strict validation structural and deterministic

Validation blocks only objective corruption: missing/foreign action IDs, bad source/Pxx lineage, illegal controlled values, non-monotonic phase windows, impossible support/airborne contradictions, left/right channel conflicts and hash drift. Aesthetic naturalness and unspecified exact joint angles do not block; they remain diagnostics or stable default support states. This follows HonCut's stop-loss rule against stochastic semantic self-rejection.

## Risks / Trade-offs

- **Generic primitives may not reproduce expert choreography perfectly** → preserve all authored prose alongside the typed projection, expose `unspecified` instead of pretending precision, and make later vocabulary expansion versioned and regression-driven.
- **A single GAU may contain multiple simultaneous performers** → keep one ordered actor track per performer, validate contact counterparts and render/test each skeleton independently.
- **A full channel timeline increases artifact size** → use controlled tokens, normalized scalars and hashes; Provider prompts receive only compact projections.
- **Coordinate mistakes can mirror left/right or confuse actor and camera rotation** → define actor-local axes and the Pxx-relative yaw anchor once, add asymmetric left/right and camera-orbit tests, and validate front/back/profile projections from actual rendered pixels.
- **Version bumps can invalidate prior runs** → support only evidence-complete zero-request side-by-side migration; keep old artifacts immutable and audit-only otherwise.
- **Fast timing can produce physically implausible interpolation** → require monotonic phase ordering, support-release/landing constraints and minimum apex separation, while leaving final semantic video quality to the existing QA/acceptance boundary.

## Migration Plan

1. Add the source-beat schema/compiler and unit tests without changing the existing Provider DTO or downstream output.
2. Attach source kinematics at the existing Phase 1 `apply_body_action_contract` boundary, then add the strict projection after final GAU/Pxx construction and bump cache/fingerprint versions.
3. Switch Phase 2, then Phase 3 and Phase 6 projections to the final production projection with provider-deny regression tests at each boundary.
4. Switch only the isolated motion-blueprint acceptance compiler to the same contract and quarantine its old independent registry; keep ordinary Phase 6 media unchanged.
5. Add strict parent-Artifact sidecar migration, audit-only/stale receipts and ten-round zero-request recovery tests.
6. Update architecture documentation, run OpenSpec Verify, full tests, Serena post-validation and Graphify refresh. Rollback removes the new consumer versions together; old immutable artifacts remain available for audit.

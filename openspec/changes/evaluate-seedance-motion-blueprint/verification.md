# Zero-cost implementation verification

## Scope

This verification covers every zero-cost task through 8.6. Tasks 6.5 and 6.6
remain intentionally pending because they require a separate fee authorization,
one real Seedance 2.0 submission, and an explicit human motion verdict.

## Scorecard

| Dimension | Result |
| --- | --- |
| Completeness | 37/39 tasks; only paid submission and human verdict pending |
| Correctness | 11/11 requirements implemented or explicitly awaiting live evidence |
| Coherence | v3 remains acceptance-only and follows the approved ownership boundary |

## Evidence

- The acceptance owner is isolated under `pipeline/src/acceptance` and the
  dedicated script. Production CLI, Lifecycle, Graph, and checkpoint discovery
  do not import it.
- The compiler is deterministic, CPU-compatible, identity-neutral, H.264, and
  rejects unsupported, static, low-amplitude, future-schema, and incomplete
  lineage inputs before upload.
- The v3 technique registry gives every supported action primitive its own
  ordered code-owned key phases, weight transfer, joint articulation, optional
  contact window, follow-through, recovery, and stable fingerprint. The old
  shared phase curve is not used by v3.
- Semantic and decoded-pixel tests prove that evade, kick, block, and strike
  have distinct temporal/geometric signatures and that contact is phase-local.
- The request prompt is declarative. Numeric setup timing, amplitude, explosive
  peak, anticipation, overshoot, and recovery tuning are absent; those facts are
  bound by the compiler, registry hash, semantic-frame hash, and pixels.
- The no-submit projection is derived from hash-verified persisted Phase 6
  evidence and changes only the motion-control medium: one static pose atlas is
  replaced by one `reference_video` blueprint.
- The optional submit path is guarded by explicit authorization, a passing
  regression receipt, the existing TOS uploader, `GenerationTaskStore`, the
  Runtime zero-retry scope, and a one-POST Seedance-only guard.
- Provider completion cannot pass the capability gate. A separate explicit
  human verdict is required, and a conclusive failure pauses the route without
  retries, redraws, reshoots, alternate providers, or budget expansion.

## Zero-cost results

- Target tests: 40 passed.
- Complete suite: 1512 passed, 1 skipped, 2 warnings.
- Provider-deny recovery replay: 10 rounds, zero Provider/TOS requests, stable
  blueprint SHA-256 `49e9263c8da80eb6cb6eb2df5ecb98df2ff8d1081102077e02780a372312ac0b`
  and task fingerprint
  `8aad583fac2089c8930afb9177a095fb58a59bd0166334e0879d8d99e083e01d`.
- Lint: passed.
- Diff check: passed.
- OpenSpec strict validation: passed.
- `pipeline_core.py`: byte-identical to the selected baseline.

Serena diagnostics were executed for every changed Python file. Actionable
source and test typing diagnostics introduced by v3 were corrected. Remaining
diagnostics are unresolved third-party and project-root imports in Serena's
standalone Pyright environment; the locked project environment resolves those
imports and passed the complete test and lint suites.

The fresh no-submit evidence is isolated at
`/Users/soda/Documents/Codex/2026-09-04/honcut-seedance-motion-blueprint-gate-03`.
Its v3 gate receipt is `pending_live_acceptance`, records zero Provider requests
and zero TOS PUTs, and classifies the v2 common-curve artifact as audit-only.

## Pending evidence

- One real Seedance 2.0 submission (task 6.5).
- Explicit human assessment of transferred motion (task 6.6).

Neither pending item is represented as passed, and the zero-cost receipt must
remain `pending_live_acceptance`.

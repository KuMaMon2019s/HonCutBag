# Verification Report

## Summary

- Change: `decompose-canonical-action-kinematics`
- Result: PASS for implementation, deterministic regression, migration, documentation, and zero-Provider recovery.
- Paid live acceptance: intentionally not executed; a later run requires separate explicit fee authorization.

## Specification Compliance

| Requirement | Evidence | Result |
| --- | --- | --- |
| Source-bound and final GAU/Pxx projection | `action_kinematics.py`, body-action owner integration, projection lineage tests | PASS |
| Complete bilateral body channels | Channel compiler, Phase 2 raster projection, asymmetric and distal-joint tests | PASS |
| Pxx-local yaw independent of camera | Orientation projection and camera-orbit tests | PASS |
| Controlled flip/spin transforms | Strict transform DTO validation and flip/spin raster tests | PASS |
| Continuous executable compound motion | Terminal-state inheritance, compressed phase windows, recovery tests | PASS |
| Shared downstream kinematic authority | Phase 2, Phase 3, Phase 6 and motion-blueprint hash/fingerprint tests | PASS |
| Provider DTO and request count unchanged | Source/schema guard and Provider-deny tests | PASS |
| Strict structural validation without probabilistic gate | Invalid schema/hash/lineage tests and controlled unknown handling | PASS |
| Strict local legacy migration | Sidecar migration, immutable parent, stale downstream and future-schema tests | PASS |

## Validation Evidence

- Targeted kinematics/pose/performance/prompt/blueprint/migration suite: PASS.
- Full suite: `1544 passed, 1 skipped`.
- Lint: PASS.
- `git diff --check`: PASS.
- `pipeline_core.py` versus baseline `eca457698de925a35446a36fbef129654cf16fe3`: unchanged.
- Serena post-change symbol/reference/type/diagnostic review: PASS for changed symbols; unrelated environment/baseline diagnostics excluded.
- Graphify incremental update, affected-path review, saved result and reflection: PASS.
- Provider request count: `0`.

## Remaining Gate

The live paid Provider gate is not part of this Apply run. The regression result permits a later no-submit preflight, but no real request may be issued without a separate explicit fee authorization.

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

## Falsification Follow-up

- Strict validators now reject internally inconsistent phase IDs, source indexes, yaw values, camera relations, performer/unit identity and aggregate child hashes even if an input recomputes its outer hash.
- Phase 2 now requires every body-action GAU in a sampled cell group to carry a valid projection and binds each action row to its exact projection hash before rendering.
- Legacy compilation is reachable through the existing `ArtifactManifestStore` boundary and a supported module entrypoint; it verifies parent bytes, authority and downstream ancestry, writes sidecar artifacts atomically and never mutates the parent.
- Relevant suite: `98 passed`; Phase 3/6/blueprint suite: `110 passed`; full suite: `1556 passed, 1 skipped`; lint and diff checks: PASS; Provider request count: `0`.
- The earlier regression receipt remains immutable audit evidence and is superseded by a new external receipt bound to the corrected implementation commit. Live acceptance remains `pending_live_acceptance`.

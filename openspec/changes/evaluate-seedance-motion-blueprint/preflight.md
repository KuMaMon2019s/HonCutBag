# Implementation Preflight

## Baseline

- Worktree: `/private/tmp/honcut-seedance-motion-blueprint-gate`
- Branch: `codex/seedance-motion-blueprint-gate`
- Production baseline: `2773492c0fd600ed51a54cf7281897f25fd2e913`
- `pipeline/src/phases/pipeline_core.py` SHA-256: `2cd41fb3ea7c77d1b29d41488e9608fa0abe2f20403c975a4131d2a5d6cbdd5a`
- The dirty primary worktree was not used for implementation.

## Serena semantic analysis

The clean worktree was activated as Serena project `honcut-seedance-motion-blueprint-gate`.

Verified owners and symbols:

- Phase 2 evidence: `_derive_narrative_guides`, `generate_shot_storyboards`, `migrate_shot_storyboard_narrative_guides`, and the pose contracts persisted in `ContinuityPlan`.
- Phase 6 prompt/media: `build_video_prompt`, `_base_content`, `_provider_content`, `_media_index_manifest`, `_provider_prompt_metadata`, and `_task_payload`.
- Transport: `seedance_client.submit_content`, `_validate_content_media_roles`, `upload_media_file_required`, and `execute_seedance_video_task`.
- Persistence: `GenerationTask`, `GenerationTaskEvent`, and `GenerationTaskStore`, especially `enqueue`, `claim`, `reserve_submission_attempt`, `confirm_provider_job`, `mark_submission_uncertain`, and `submission_attempt_count`.
- Callers: direct generation and continuity execution share `execute_seedance_video_task`; the existing Phase 6 live acceptance is a dedicated non-production caller.
- Tests: `test_continuity_foundation.py`, `test_direct_ark_routing.py`, `test_generation_task_history.py`, `test_audit_regressions.py`, and the dedicated new gate tests.

The language server does not implement LSP `textDocument/implementation` for the Python protocol, so concrete implementations were verified with Serena symbol definitions/references and then against source. No implementation relationship was inferred from a filename.

## Graphify impact analysis

Graphify placed Phase 2 guide derivation in the `shot_storyboards.py` subsystem, Phase 6 prompt construction under `build_video_prompt`, continuity request assembly under `continuity_provider.py`, and paid execution under `execute_seedance_video_task` / `GenerationTaskStore`. Reverse impact for `execute_seedance_video_task` identified direct generation, continuity execution, existing live acceptances, and task-recovery tests. Every selected node was checked against the clean-baseline source.

The analysis was saved with Graphify as a sanitized useful result. Frontend, public API, database schema, Graph topology, and normal checkpoint discovery are not affected. TOS and Seedance are affected only by the optional acceptance submit path.

## Reconciliation and minimum safe change set

The verified call chain agrees with the OpenSpec design. The minimum safe implementation is:

1. an acceptance-only, CPU-compatible deterministic compiler;
2. one dedicated no-submit/live gate script;
3. focused tests and OpenSpec evidence.

The implementation must not be imported by `pipeline_runner.py`, Lifecycle, Graph, or production Phase owners. It must not modify `pipeline_core.py`, Graph topology, production media order, Provider retry policy, or database schema. A successful experiment still cannot activate production; that requires another OpenSpec change.

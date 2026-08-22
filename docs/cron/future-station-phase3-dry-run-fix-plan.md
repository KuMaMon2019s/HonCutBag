# Future Station Cron: Phase 3 dry-run blocker plan

## Reproduction

Fixture: `pipeline/tests/fixtures/future_station_cron.txt`

```bash
uv run --locked --managed-python python pipeline/src/pipeline_runner.py \
  --input pipeline/tests/fixtures/future_station_cron.txt \
  --duration 60 \
  --shot-duration 6 \
  --dry-run \
  --skip-phase 6 7 8 9 9.5 \
  --output-dir /tmp/honcut-future-station \
  --project-id cron-future-station \
  --disable-reshoot \
  --allow-real-person \
  --auto-approve
```

Observed on 2026-08-22 from `bf60c81`:

- Phase 1 completed with deterministic mock artifacts.
- Phase 2 skipped Provider work as expected.
- Phase 3 created character cards with `skip_images=True`.
- Phase 3 then ran the production reference-image quality gate and failed because
  the intentionally absent four-view images and semantic QA receipts are mandatory.
- The CLI exited with code 1 before Phase 4, Phase 5, or resume validation.
- No image or video Provider request was submitted.

Failure signature:

```text
Phase 3 质检未通过: C — 角色四视图缺失、语义视角错误或审核凭证已过期，不能继续
```

## Root cause

`phases.phase3.phase3_character.run_phase3` correctly passes
`skip_images=dry_run` to the character factory, but it does not give the remaining
Phase 3 path a dry-run boundary. The same invocation then:

1. calls the production `run_quality_check("phase3", output_dir)`, which must fail
   closed when four-view image assets are absent; and
2. would refresh character-locked Pxx storyboards after the gate, even though that
   work is not part of a zero-Provider dry-run.

The production quality rules are correct. The defect is that simulated artifacts
are passed into production-only gates and downstream asset generation.

## Planned fix

1. Add an explicit Phase 3 dry-run completion path after deterministic character
   cards are built and before the production image QA gate.
2. Write a small versioned dry-run receipt containing the character IDs, required
   reference-view names, skipped Provider operations, and source artifact hashes.
   Do not fabricate image files or a passing production QA receipt.
3. Return `status="done"` with an explicit `dry_run=true` marker and skip both the
   production reference-image gate and character-locked Pxx regeneration.
4. Keep the non-dry-run path unchanged and fail closed for missing, stale, or
   semantically invalid character references.
5. Add regression tests that make every image/video client raise if called, prove
   the dry-run finishes Phase 3 without Provider submissions, and prove the same
   missing artifacts still fail in production mode.
6. Run the Future Station Phase 1–5 command, repeat it with `--resume`, inspect the
   report and task database for zero submissions, then run targeted pytest, lint,
   and the full test suite.
7. If the now-reachable Phase 4 or Phase 5 exposes another independent dry-run
   blocker, record it separately before changing that owner.

## Acceptance criteria

- The reproduction command exits 0 and reaches Phase 5.
- A second invocation with `--resume` exits 0 and reuses safe checkpoints.
- No Provider submit call or persistent generation task is created.
- No placeholder image is accepted as a production reference.
- Non-dry-run Phase 3 continues to reject missing images and missing/stale semantic
  QA receipts.
- Targeted tests, `make lint`, `git diff --check`, and `make test` pass.

## Commit and rollback boundary

Use one implementation commit:

```text
fix(cron): keep Phase 3 dry-run outside production image gates
```

Rollback by reverting that commit. The fixture and this plan can remain as a
reproduction record.

# Future Station Cron: Phase 4–5 dry-run blocker plan

## Baseline and reproduction

This plan starts from Phase 3 fix `19645e0`. Run the fixed Future Station
fixture without Provider credentials:

```bash
env -u ARK_AGENT_API_KEY -u ARK_API_KEY -u SEEDREAM_API_KEY \
  -u OPENAI_API_KEY -u VOLCENGINE_API_KEY \
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

Phase 3 now completes with
`honcut.phase3-dry-run-receipt.v1`, no `runtime.db`, and zero generation
tasks. The next two independent blockers belong to Phase 4 and Phase 5.

## Blocker 1: Phase 4 legacy shot metadata

Failure signature:

```text
vendor/legacy/orchestrator.py:64 in parse_shots
    "name": s["name"]
KeyError: 'name'
```

The Phase 1 dry-run storyboard is valid under the current canonical schema but
does not include the legacy-only `name` field. Phase 4 passes that document
directly to the legacy subprocess.

Planned repair:

1. Normalize subprocess input at the Phase 4 ownership boundary; do not weaken
   the canonical storyboard schema or add legacy aliases to production writers.
2. Derive a deterministic display name from existing shot metadata, falling
   back to the normalized shot ID, and write the adapted document inside the run
   directory.
3. Preserve every semantic field and keep subprocess arguments as an array.
4. Add a characterization test for canonical shots without `name`, plus a
   compatibility test proving an existing authored name is preserved.
5. Keep live Phase 4 artifact and subprocess failure checks unchanged.

Implementation commit:

```text
fix(cron): normalize Phase 4 dry-run shot metadata
```

## Blocker 2: Phase 5 dry-run pixel and supervision gates

After Phase 4 is bypassed for diagnosis, Phase 5 reports six severe
`storyboard_beat_image_missing` findings because Phase 2 dry-run intentionally
does not generate Pxx pixels. Its correction loop can also enter image redraw,
and independent supervision can call a text LLM when credentials are present.

Planned repair:

1. Add an explicit Phase 5 dry-run path that runs deterministic metadata checks
   only: L1 contracts, capacity, variation, and slideshow risk.
2. Do not execute L2–L4 pixel review, image correction, embeddings, multimodal
   clients, or independent LLM supervision in dry-run.
3. Write an atomic `honcut.phase5-dry-run-receipt.v1` containing input hashes,
   structural findings, skipped operations, and the final structural verdict.
4. Keep production Phase 5 fail closed for missing Pxx images, unavailable
   required reviews, failed corrections, and blocking supervision.
5. Thread the existing `dry_run` value through both sequential and Graph Phase 5
   adapters without changing CLI flags or production defaults.

Implementation commit:

```text
fix(cron): separate Phase 5 dry-run from pixel QA
```

## Acceptance

- The Future Station Phase 1–5 command exits 0, and a second identical invocation
  with `--resume` exits 0.
- Phase 4 produces deterministic shot directories and continuity artifacts.
- Phase 5 produces the structural dry-run receipt without pixel or Provider work.
- Provider constructors and submit methods are guarded by tests that raise if
  called; `runtime.db` is absent or contains zero generation tasks.
- Production Phase 4/5 characterization tests, targeted pytest, `make lint`,
  `git diff --check`, and `make test` pass.
- Each owner is changed in its own commit. No push or PR is created.

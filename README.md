# HonCut — AI Video Generation Pipeline

HonCut is an end-to-end AI video generation pipeline that transforms arbitrary text input into polished video output. It combines LLM-driven storytelling, character asset generation, multi-model video synthesis, and automated post-production into a single reproducible pipeline.

## Architecture

```
Text Input → Director Planning → Screenwriter Engine → Character Factory → Orchestrator → Video Generation → Assembly Engine → Post-Production → Video Output
```

### 9-Phase Pipeline

| Phase | Name | Description |
|-------|------|-------------|
| Phase 1 | Director Planning | Scene breakdown, emotion analysis, and transition design before storyboarding (M1) |
| Phase 2 | Screenwriter Engine | Text parsing → event extraction → character discovery → adaptation → storyboard generation |
| Phase 2.5 | Storyboard Sequence | Per-shot storyboard images for visual reference (M2) |
| Phase 3 | Character Factory | Three-view generation (front/side/back) + character cards |
| Phase 4 | Orchestrator | Shot scheduling, timeline planning, and model routing (M4) |
| Phase 5 | Video Generation | Async video clip generation via Seedance/Wan APIs |
| Phase 6 | Quality Gate | Consistency checks, red-line supervision, and A/B/C/D grading (M5) |
| Phase 7 | Assembly Engine | Clip stitching with transition bridges (M3) |
| Phase 8 | Post-Production | Audio mixing, visual enhancement, rhythm editing, final encode |

### Incremental Modules (M1–M6)

| Module | Component | Purpose |
|--------|-----------|---------|
| M1 | `director_planner.py` | Scene breakdown, emotion analysis, transition design — runs before storyboarding to give downstream phases emotional grounding and consistency anchors |
| M2 | Storyboard Sequence | Generates one storyboard image per shot in Phase 2.5, used as composition reference during video generation |
| M3 | Transition Bridges | Four bridge types (action / emotion / spatial / dialogue) injected into adaptation prompts to eliminate jump-cut feel between shots |
| M4 | `prompt_router.py` | Auto-routes prompt format by target model — Seedance 2.0 multi-shot, Seedance 2.0 single-shot, Wan 2.6 narrative, or generic first/last-frame |
| M5 | `quality_gate.py` (`run_storyboard_review`) | Supervision layer with 4 red-line checks (asset validity, script fidelity, concreteness, parent-child assets) + A/B/C/D grading |
| M6 | `artifact_chain.py` | Per-phase checkpoint files + `--resume-from` recovery to restart from any phase without re-running completed work |

## Project Structure

```
honcut/
├── pipeline/           # Python video pipeline
│   ├── src/           # Source modules
│   ├── tests/         # Test suite
│   └── requirements.txt
├── docker/            # Docker Compose (Qdrant, MinIO, n8n)
├── data/              # Input/output data
├── docs/              # Documentation
├── scripts/           # Utility scripts
├── Makefile           # Common commands
├── pyproject.toml     # Python project config
└── environment.yml    # Conda environment
```

## Development

### Code Quality

The project uses **Black** for formatting and **Ruff** for linting:

```bash
pip install -e ".[dev]"
black pipeline/src/ pipeline/tests/
ruff check pipeline/src/ pipeline/tests/
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Testing

```bash
make test
# or
pytest pipeline/tests/ -v --cov=pipeline/src
```

## Dependencies

- Python 3.11+
- Docker & Docker Compose
- FFmpeg
- Volcano Ark API (Seedance, Seedream)

## License

MIT

# HonCut — AI Video Generation Pipeline

HonCut is an end-to-end AI video generation pipeline that transforms arbitrary text input into polished video output. It combines LLM-driven storytelling, character asset generation, multi-model video synthesis, and automated post-production into a single reproducible pipeline.

## Architecture

```
Text Input → Director Planning → Screenwriter Engine → Character Factory → Scene Consistency → Video Generation → Quality Gate → Assembly Engine → Post-Production → Video Output
```

### Pipeline Phases

| Phase | Name | Description |
|-------|------|-------------|
| Phase 1 | Director Planning | Scene breakdown, emotion analysis, and transition design before storyboarding |
| Phase 2 | Screenwriter Engine | Text parsing → event extraction → character discovery → adaptation → storyboard generation with eight-layer prompt framework |
| Phase 2.5 | Storyboard Sequence | Per-shot storyboard images for visual reference |
| Phase 3 | Character Factory | Character reference assets (face close-up + full-body + variants) + character cards |
| Phase 4 | Orchestrator | Shot scheduling, timeline planning, scene consistency contracts, and model routing |
| Phase 5 | Video Generation | Video clip generation via Seedance (online) with Wan2.2 local fallback |
| Phase 6 | Quality Gate | Consistency checks, red-line supervision, and A/B/C/D grading |
| Phase 7 | Assembly Engine | Clip stitching with smart transition analysis |
| Phase 8 | Post-Production | Audio mixing, color grading, rhythm editing, final encode |

## Key Capabilities

- **Eight-layer prompt framework** — every shot prompt is assembled from eight structured layers (element reference, shot summary, audio, style anchor, quality suffix, negative guardrails), balancing concise shot descriptions with full constraint coverage.
- **Seedance-first with graceful fallback** — shots are generated via Seedance online API by default; on timeout or stall the pipeline falls back to local Wan2.2 generation with explicit duration-loss logging.
- **Chain mode (last-frame relay)** — optional serial generation where each shot's last frame becomes the next shot's first frame, physically inheriting character appearance, lighting, and scene continuity across shots.
- **Segmentation-aware shot duration** — shot length follows video-model segmentation best practice (medium-form video: fewer, longer shots, one clear plot beat per shot) instead of many short fragments.
- **Fictional-character declaration** — reference-image and video prompts declare AI-generated fictional characters to reduce real-person content moderation friction.
- **Checkpoint & resume** — per-phase checkpoint files with `--resume-from` recovery restart from any phase without re-running completed work.
- **Quality supervision** — four red-line checks (asset validity, script fidelity, concreteness, parent-child assets) with A/B/C/D grading gate before assembly.

## Project Structure

```
honcut/
├── pipeline/           # Python video pipeline
│   ├── src/            # Source modules (phases, clients, tools, quality)
│   ├── scripts/        # Orchestrator entry points
│   ├── prompts/        # Prompt templates
│   └── requirements.txt
├── docker/             # Docker Compose (Qdrant, MinIO, n8n)
├── data/               # Input/output data
├── scripts/            # Utility scripts
├── Makefile            # Common commands
├── pyproject.toml      # Python project config
└── environment.yml     # Conda environment
```

## Development

### Code Quality

The project uses **Black** for formatting and **Ruff** for linting:

```bash
pip install -e ".[dev]"
black pipeline/src/ pipeline/tests/
ruff check pipeline/src/ pipeline/tests/
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

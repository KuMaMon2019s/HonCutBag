# HonCut Quick Start Guide

## Requirements

- Python 3.11+
- Conda or venv
- FFmpeg
- Docker (optional, for service deployment)

## Quick Installation

### 1. Clone Project

```bash
cd /Users/soda/projects/honcut
```

### 2. Create Virtual Environment

Using Conda (recommended):
```bash
conda create -n honcut python=3.11 -y
conda activate honcut
```

Or using venv:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
make install
```

This installs all Python dependencies and configures the project.

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` file and fill in required API keys:
- `ARK_AGENT_API_KEY`: Volcano Ark API key (required)
- `ARK_BASE_URL`: Volcano Ark API URL (pre-configured by default)

## Run Pipeline

### Basic Usage

```bash
make run INPUT_FILE=your_script.txt
```

### Direct Invocation

```bash
python pipeline/src/pipeline_runner.py \
  --input your_script.txt \
  --duration 60 \
  --output-dir data/output \
  --auto-approve
```

### Parameters

- `--input`: Input script file path
- `--duration`: Target video duration in seconds, default 60
- `--output-dir`: Output directory, default `data/output`
- `--auto-approve`: Auto-approve manual review nodes (for CI/testing)
- `--dry-run`: Validate pipeline only, no video generation
- `--transition`: Transition mode (crossfade/fade/cut)
- `--media-profile`: Encoding profile (1080p/480p, etc.)

## Pipeline Phases

HonCut pipeline includes 9 phases:

1. **Phase 1**: Director Planning (scene breakdown, emotion analysis, transition design)
2. **Phase 2**: Screenwriter Engine (text parsing + storyboard generation)
3. **Phase 2.5**: Storyboard image generation (per-shot visual reference)
4. **Phase 3**: Character Factory (three-view generation)
5. **Phase 4**: Orchestrator (shot scheduling + model routing)
6. **Phase 5**: Video Generation (Seedance API)
7. **Phase 6**: Quality Gate (consistency checks + supervision)
8. **Phase 7**: Assembly Engine (video stitching with transition bridges)
9. **Phase 8**: Post-Production (subtitles + transitions + final encode)

## Testing

Run all tests:
```bash
make test
```

Run single test:
```bash
pytest pipeline/tests/test_consistency_guard.py -v
```

## FAQ

### Q: How to skip a phase?

```bash
python pipeline/src/pipeline_runner.py \
  --input your_script.txt \
  --skip-phase 5 6 7 8
```

### Q: How to resume from checkpoint?

```bash
python pipeline/src/pipeline_runner.py \
  --input your_script.txt \
  --resume
```

### Q: What if video generation fails?

1. Check if `ARK_AGENT_API_KEY` is configured correctly
2. Check `data/output/pipeline_report.json` for failed phase
3. Use `--dry-run` to validate pipeline
4. Check log files for detailed errors

### Q: How to view progress?

The pipeline outputs real-time progress information including:
- Current phase
- Processing progress
- Generated file list

## Output Files

After successful run, `data/output/` directory contains:

- `polished.mp4`: Final video
- `STORYBOARD.json`: Storyboard data
- `CHARACTERS.json`: Character data
- `consistency_report.json`: Consistency check report
- `pipeline_report.json`: Pipeline execution report
- `shots/`: Intermediate files for each shot

## Next Steps

- Read [API Documentation](API.md) for detailed interfaces
- Check [Usage Examples](EXAMPLES.md) for practical usage
- Refer to [Pipeline Architecture](PIPELINE.md) for design details

## Need Help?

If you encounter issues:
1. Check log files
2. Run `make test` to verify environment
3. Check `docs/MIGRATION.md` for project structure

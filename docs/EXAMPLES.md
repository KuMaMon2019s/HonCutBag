# HonCut Usage Examples

## Example 1: Basic Video Generation

Simplest usage, generate 60-second video from script.

```bash
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/basic_example \
  --auto-approve
```

**Output**:
- `data/output/basic_example/polished.mp4` - Final video
- `data/output/basic_example/STORYBOARD.json` - Storyboard data
- `data/output/basic_example/CHARACTERS.json` - Character data

---

## Example 2: Dry Run Test

Validate pipeline only, no actual video generation (fast test).

```bash
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 30 \
  --output-dir data/output/dry_run_test \
  --dry-run
```

**Purpose**: Quickly check script format and configuration without consuming API quota.

---

## Example 3: Skip Specific Phases

Skip phases that are already completed.

```bash
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/skip_example \
  --skip-phase 2 3 4 \
  --auto-approve
```

**Purpose**: Regenerate video (Phase 5+), skip screenwriting and character generation.

---

## Example 4: Resume from Checkpoint

Resume pipeline if it was interrupted.

```bash
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/resume_example \
  --resume
```

**Purpose**: Resume interrupted pipeline execution, avoid repeating completed work.

---

## Example 5: Custom Transition and Encoding

Specify transition effect and encoding profile.

```bash
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/custom_encode \
  --transition crossfade \
  --media-profile 1080p \
  --auto-approve
```

**Available Transitions**:
- `crossfade` - Cross fade (default)
- `fade` - Fade in/out
- `cut` - Hard cut

**Available Encoding Profiles**:
- `1080p` - 1920x1080 @ 30fps
- `480p` - 854x480 @ 30fps
- `720p` - 1280x720 @ 30fps
- `cinematic` - 2048x858 @ 24fps
- `youtube_shorts` - 1080x1920 @ 30fps (vertical)

---

## Example 6: Python API Call

Call pipeline in Python code.

```python
from pipeline.src.pipeline_runner import run_pipeline

result = run_pipeline(
    input_file="scripts/sample_story.txt",
    duration=60,
    output_dir="data/output/python_api",
    dry_run=False,
    auto_approve=True,
    transition="crossfade",
    media_profile="1080p"
)

print(f"Execution status: {result['status']}")
print(f"Total duration: {result['total_duration']:.2f}s")
print(f"Output directory: {result['output_dir']}")
```

---

## Example 7: Batch Process Multiple Scripts

Use script to batch process multiple scripts.

```bash
#!/bin/bash
# batch_process.sh

for story in scripts/stories/*.txt; do
    output_name=$(basename "$story" .txt)
    echo "Processing: $story"
    
    python pipeline/src/pipeline_runner.py \
        --input "$story" \
        --duration 60 \
        --output-dir "data/output/batch/$output_name" \
        --auto-approve
    
    echo "Completed: $output_name"
done
```

**Usage**:
```bash
chmod +x batch_process.sh
./batch_process.sh
```

---

## Example 8: Check Consistency Report

Check character consistency after video generation.

```python
from pipeline.src.consistency_guard import run_consistency_check

report = run_consistency_check(
    output_dir="data/output/basic_example",
    threshold=70
)

print(f"Overall consistency: {report['overall_score']:.2f}")
print(f"Passed: {report['passed']}")

for char_id, details in report['details'].items():
    print(f"  {char_id}: {details['score']:.2f}")
    if details['issues']:
        print(f"    Issues: {details['issues']}")
```

---

## Example 9: Custom Prompt Generation

Manually modify storyboard prompts and regenerate.

```bash
# 1. Generate storyboard first
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/custom_prompt \
  --skip-phase 5 6 7 8 \
  --auto-approve

# 2. Edit STORYBOARD.json
# Modify prompts field, add custom descriptions

# 3. Continue from Phase 5
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/custom_prompt \
  --resume \
  --auto-approve
```

---

## Example 10: Monitor Pipeline Progress

Real-time monitor pipeline execution progress.

```bash
# Monitor in another terminal
tail -f data/output/monitor_example/pipeline.log

# View current phase
watch -n 1 'cat data/output/monitor_example/progress.json | jq .current_phase'
```

---

## Example 11: Using Makefile Commands

Use predefined Makefile commands to simplify operations.

```bash
# Install dependencies
make install

# Run tests
make test

# Run pipeline
make run INPUT_FILE=scripts/sample_story.txt

# Clean output
make clean

# Start Docker services
make docker-up

# Stop Docker services
make docker-down
```

---

## Example 12: Debug Mode

Enable verbose logging for debugging.

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Run pipeline
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/debug_example \
  --auto-approve 2>&1 | tee debug.log
```

---

## Example 13: Generate Different Style Videos

Use different media profiles to generate different style videos.

```bash
# Cinematic style
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --output-dir data/output/cinematic \
  --media-profile cinematic \
  --transition fade \
  --auto-approve

# YouTube Shorts (vertical)
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --output-dir data/output/shorts \
  --media-profile youtube_shorts \
  --transition crossfade \
  --auto-approve
```

---

## Example 14: Check Pipeline Report

View pipeline execution report.

```python
import json

with open("data/output/basic_example/pipeline_report.json") as f:
    report = json.load(f)

print("=== Pipeline Execution Report ===")
print(f"Status: {report['status']}")
print(f"Total duration: {report['total_duration']:.2f}s")
print(f"Number of phases: {len(report['phases'])}")

for phase_name, phase_data in report['phases'].items():
    print(f"\n{phase_name}:")
    print(f"  Status: {phase_data['status']}")
    print(f"  Duration: {phase_data['duration']:.2f}s")
    if phase_data.get('error'):
        print(f"  Error: {phase_data['error']}")
```

---

## Example 15: Environment Variable Configuration

Use environment variables to configure API and other settings.

```bash
# Configure API key
export ARK_AGENT_API_KEY=your_api_key_here

# Configure API URL
export ARK_BASE_URL=https://api.volcengine.com

# Configure log level
export LOG_LEVEL=INFO

# Run pipeline
python pipeline/src/pipeline_runner.py \
  --input scripts/sample_story.txt \
  --duration 60 \
  --output-dir data/output/env_config \
  --auto-approve
```

---

## FAQ

### Q: How to view API call details?

Set `LOG_LEVEL=DEBUG` and check log files.

### Q: How to interrupt a running pipeline?

Press `Ctrl+C`, pipeline will save checkpoint, next time you can use `--resume` to recover.

### Q: How to regenerate a specific shot?

Delete corresponding `shots/SXX/` directory, then use `--resume` to rerun.

### Q: How to modify character appearance?

Edit character description in `CHARACTERS.json`, then use `--resume` to regenerate.

### Q: How to export intermediate results?

All intermediate files are in `output_dir/` directory, including:
- `STORYBOARD.json` - Storyboard data
- `CHARACTERS.json` - Character data
- `shots/` - Videos and images for each shot

---

## Next Steps

- Read [QUICKSTART.md](QUICKSTART.md) for quick start
- Read [API.md](API.md) for detailed interfaces
- Read [PIPELINE.md](PIPELINE.md) for pipeline architecture

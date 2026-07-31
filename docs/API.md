# HonCut API Documentation

## Core Modules

### pipeline_runner.py

Main pipeline executor that orchestrates all phases.

#### Main Functions

```python
def run_pipeline(
    input_file: str,
    duration: int = 60,
    output_dir: str = "data/output",
    dry_run: bool = False,
    auto_approve: bool = False,
    skip_phase: List[int] = None,
    transition: str = "crossfade",
    media_profile: str = "1080p"
) -> dict
```

**Parameters**:
- `input_file`: Path to input script file
- `duration`: Target video duration in seconds
- `output_dir`: Output directory
- `dry_run`: Validate pipeline only, no video generation
- `auto_approve`: Auto-approve manual review nodes
- `skip_phase`: List of phases to skip
- `transition`: Transition mode (crossfade/fade/cut)
- `media_profile`: Encoding profile

**Returns**: Execution report dictionary

---

### character_factory.py

Character factory that generates three-view character assets.

#### Main Functions

```python
def generate_character_views(
    character: dict,
    output_dir: str
) -> dict
```

**Parameters**:
- `character`: Character data (from CHARACTERS.json)
- `output_dir`: Output directory

**Returns**: Dictionary containing three-view image paths

---

### seedance_client.py

Seedance API client for video generation.

#### Main Functions

```python
def submit_video_generation(
    prompt: str,
    image_url: str = None,
    duration: int = 5,
    **kwargs
) -> dict
```

**Parameters**:
- `prompt`: Video generation prompt
- `image_url`: Reference image URL (optional)
- `duration`: Video duration
- `**kwargs`: Additional parameters

**Returns**: Task submission result

```python
def poll_video_status(
    task_id: str,
    timeout: int = 300
) -> dict
```

**Parameters**:
- `task_id`: Task ID
- `timeout`: Timeout in seconds

**Returns**: Video generation status and result

---

### seedream_client.py

Seedream API client for image generation.

#### Main Functions

```python
def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    **kwargs
) -> dict
```

**Parameters**:
- `prompt`: Image generation prompt
- `width`: Image width
- `height`: Image height
- `**kwargs`: Additional parameters

**Returns**: Image generation result

---

### consistency_guard.py

Consistency guard that checks character consistency across shots.

#### Main Functions

```python
def run_consistency_check(
    output_dir: str,
    threshold: int = 70
) -> dict
```

**Parameters**:
- `output_dir`: Output directory
- `threshold`: Consistency threshold (0-100)

**Returns**: Consistency check report

---

### storyboard_generator.py

Storyboard generator that produces STORYBOARD.json.

#### Main Functions

```python
def generate_storyboard(
    script_text: str,
    duration: int = 60
) -> dict
```

**Parameters**:
- `script_text`: Script text
- `duration`: Target duration

**Returns**: Storyboard data

---

## Data Structures

### CHARACTERS.json

```json
{
  "characters": [
    {
      "id": "char_001",
      "name": "Protagonist",
      "description": "Character description",
      "appearance": {
        "hair": "Short black hair",
        "clothing": "Casual wear",
        "face": "Determined expression",
        "build": "Medium build",
        "gender": "male",
        "age_range": "25-35"
      },
      "reference_images": {
        "front": "path/to/front.png",
        "side": "path/to/side.png",
        "back": "path/to/back.png"
      }
    }
  ]
}
```

### STORYBOARD.json

```json
{
  "shots": [
    {
      "id": "S01",
      "prompt": "Shot description",
      "duration": 5,
      "characters": ["char_001"],
      "scene": "Scene description",
      "camera": "Camera movement"
    }
  ]
}
```

### consistency_report.json

```json
{
  "overall_score": 85,
  "passed": true,
  "details": {
    "char_001": {
      "score": 90,
      "issues": []
    }
  }
}
```

---

## Configuration Options

### config.yaml

```yaml
pipeline:
  default_duration: 60
  default_transition: crossfade
  default_media_profile: 1080p

api:
  ark_base_url: https://api.volcengine.com
  timeout: 300

consistency:
  threshold: 70
  max_retries: 3

generation:
  max_concurrent: 3
  retry_delay: 5
```

---

## Error Handling

### Common Error Codes

- `API_KEY_MISSING`: API key not configured
- `API_TIMEOUT`: API call timeout
- `GENERATION_FAILED`: Video/image generation failed
- `CONSISTENCY_LOW`: Consistency check failed
- `PHASE_FAILED`: Phase execution failed

### Error Handling Example

```python
try:
    result = run_pipeline(...)
except APIKeyError as e:
    print(f"API key error: {e}")
except TimeoutError as e:
    print(f"API call timeout: {e}")
except PipelineError as e:
    print(f"Pipeline execution error: {e}")
```

---

## Extension Development

### Adding a New Phase

1. Create new module in `pipeline/src/`
2. Implement `execute()` function
3. Register phase in `pipeline_runner.py`
4. Add tests to `pipeline/tests/`

### Custom Transition Effects

1. Modify `phase7_assembly.py`
2. Implement new transition function
3. Register in configuration file

### Integrating New APIs

1. Create new client module
2. Implement standard interface (submit/poll)
3. Add API configuration to config file

# Seedance 2.0 Advanced Capabilities

> Reference: Toonflow + OpenMontage implementation analysis

## 1. generate_audio (Native Audio Generation)

### Capability
Seedance 2.0 supports native audio generation alongside video. The model generates synchronized audio (dialogue, sound effects, ambient) based on the visual content and prompt.

### Toonflow Implementation
- **File**: `toonflow/pipelines/seedance_pipeline.py` line ~420-450
- **Parameter**: `generate_audio: bool = True` in API request
- **Usage**: Always enabled by default for cinematic quality

### OpenMontage Implementation
- **File**: `openmontage/tools/video/seedance_video.py` line ~180-200
- **Parameter**: Passed in payload as `"generate_audio": true`
- **Note**: "This is the moat" — native audio sync is a key differentiator

### HonCut Implementation Plan
```python
# In local_video_client.py or seedance_client.py
payload = {
    "prompt": prompt,
    "generate_audio": True,  # Enable native audio
    # ... other params
}
```

### Pitfalls
- Audio generation adds ~20% to generation time
- Some prompts may trigger privacy filters that disable audio
- Fallback: Generate video-only, then add audio in post-production (Phase 8)

---

## 2. Multi-Modal Combination Reference

### Capability
Seedance 2.0 supports combining multiple reference types in a single generation:
- Images: up to 9 reference images
- Videos: up to 3 reference videos
- Audio: up to 3 reference audio clips

### Toonflow Implementation
- **File**: `toonflow/pipelines/seedance_pipeline.py` line ~380-420
- **Structure**: `content[]` array with mixed `type` fields
```python
content = [
    {"type": "text", "text": prompt},
    {"type": "image_url", "image_url": {"url": ref_image_1}},
    {"type": "image_url", "image_url": {"url": ref_image_2}},
    {"type": "video_url", "video_url": {"url": ref_video_1}},
]
```

### OpenMontage Implementation
- **File**: `openmontage/tools/video/seedance_video.py` line ~220-260
- **Note**: Uses `reference_image_urls` + `reference_video_urls` separately

### HonCut Implementation Plan
```python
# Build content array with multiple reference types
content = []
content.append({"type": "text", "text": prompt})

# Add character reference images
for img_path in character_refs:
    content.append({"type": "image_url", "image_url": {"url": img_path}})

# Add scene reference video (if available)
if scene_video_ref:
    content.append({"type": "video_url", "video_url": {"url": scene_video_ref}})
```

### Pitfalls
- Total payload size limited — compress images before sending
- Video references must be < 10 seconds
- Audio references not yet tested in HonCut

---

## 3. return_last_frame (Video Continuation)

### Capability
Seedance 2.0 can return the last frame of a generated video, enabling video continuation/chaining.

### Toonflow Implementation
- **File**: `toonflow/pipelines/seedance_pipeline.py` line ~460-480
- **Parameter**: `return_last_frame: bool = True`
- **Usage**: Used for multi-shot continuity — last frame of shot N becomes first frame reference of shot N+1

### OpenMontage Implementation
- **Status**: Not implemented (noted as TODO in comments)

### HonCut Implementation Plan
```python
# Generate shot with last frame return
response = seedance_api.generate(
    prompt=prompt,
    return_last_frame=True,
    # ... other params
)

# Extract last frame for next shot
last_frame = response.last_frame_base64
next_shot_content = [
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{last_frame}"}},
    {"type": "text", "text": next_prompt},
]
```

### Pitfalls
- Adds ~5% to generation time
- Last frame quality may be lower than middle frames
- Alternative: Use FLF2V (first-last frame to video) for explicit control

---

## 4. Video Extension / Editing

### Capability
Seedance 2.0 supports extending an existing video by generating additional frames.

### Toonflow Implementation
- **File**: `toonflow/pipelines/seedance_pipeline.py` line ~490-520
- **Parameter**: `video_extension: bool = True, extension_duration: float = 5.0`
- **Status**: Experimental, not widely used

### OpenMontage Implementation
- **Status**: Not implemented

### HonCut Implementation Plan
- **Priority**: P2 (low priority)
- **Alternative**: Use video_stitch for multi-shot assembly instead of extension

---

## Summary: Priority for HonCut

| Feature | Priority | Effort | Impact |
|---------|----------|--------|--------|
| generate_audio | 🔴 P0 | Small | High — native audio sync |
| Multi-modal reference | 🟡 P1 | Medium | Medium — better character consistency |
| return_last_frame | 🔵 P2 | Small | Low — FLF2V already handles this |
| Video extension | 🔵 P2 | Medium | Low — video_stitch is better |

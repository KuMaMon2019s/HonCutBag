#!/usr/bin/env python3
"""S05 FLF2V smoke test v3: M3 t2i end frame → resize 1280x720 → TOS → Bridge flf2v → poll → download → verify."""
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
from PIL import Image

OUT = Path('../output/westlake_evening_v8')
SHOT = 'S05'
TARGET_W, TARGET_H = 1280, 720  # match video dimensions (Minister's recommendation)
shot_dir = OUT / 'shots' / SHOT
meta = json.loads((shot_dir / 'SHOT_META.json').read_text())
meta['gen_strategy'] = 'flf2v'
meta['duration'] = 2  # smoke: fast 49-frame render

first_frame = OUT / 'storyboard_images' / f'{SHOT}.png'
end_frame = OUT / 'storyboard_images' / f'{SHOT}_end.png'
print(f'[smoke] first frame: {first_frame} exists={first_frame.exists()}')

# Step 0: re-validate existing end frame with CURRENT thresholds (don't trust stale sidecar)
from pipeline_runner import _read_end_frame_sidecar, _generate_flf2v_end_frame, _validate_end_frame, _write_end_frame_sidecar, _file_sha256
import hashlib as _hashlib
need_regen = True
if end_frame.exists() and end_frame.stat().st_size > 1024:
    # Re-run validation with current (M4) thresholds instead of trusting stored passed flag
    fresh_validation = _validate_end_frame(first_frame, end_frame)
    if fresh_validation.get("passed"):
        print(f'[smoke] existing end frame RE-VALIDATED with current thresholds '
              f'(similarity={fresh_validation.get("similarity")}), reusing — no Seedream quota spent')
        need_regen = False
    else:
        reason = fresh_validation.get('reason', 'unknown')
        print(f'[smoke] existing end frame fails current validation (reason: {reason}), removing for regeneration')
        end_frame.unlink()
        for orphan in OUT.glob('storyboard_images/S05_end*.meta.json'):
            orphan.unlink()
            print(f'[smoke] removed orphan sidecar: {orphan.name}')

if need_regen:
    print('[smoke] generating end frame via M3 t2i (rate-limited, ~2 min)...')
    ok = _generate_flf2v_end_frame(meta, SHOT, first_frame, None)
    print(f'[smoke] end frame generated: {ok}, exists={end_frame.exists()}')
    if end_frame.exists():
        sc = _read_end_frame_sidecar(end_frame)
        if sc:
            print(f'[smoke] validation: {json.dumps(sc.get("validation", {}), ensure_ascii=False)}')

if not end_frame.exists() or end_frame.stat().st_size < 1024:
    print('[smoke] FATAL: no end frame, aborting')
    sys.exit(1)

# Step 1: resize both frames to video dimensions (1280x720) for FLF2V stability
resized = {}
for name, src in [('start', first_frame), ('end', end_frame)]:
    img = Image.open(src).convert('RGB')
    if img.size != (TARGET_W, TARGET_H):
        img = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        print(f'[smoke] {name}: resized {Image.open(src).size} -> {TARGET_W}x{TARGET_H}')
    tmp = Path(f'/tmp/flf2v_smoke_{name}.png')
    img.save(tmp)
    resized[name] = tmp

# Step 2: upload to TOS
import tos_uploader
url_start = tos_uploader.upload_image(resized['start'].read_bytes(), 'image/png')
url_end = tos_uploader.upload_image(resized['end'].read_bytes(), 'image/png')
print(f'[smoke] TOS start: {str(url_start)[:70]}...')
print(f'[smoke] TOS end:   {str(url_end)[:70]}...')
if not url_start or not url_end:
    print('[smoke] FATAL: TOS upload failed')
    sys.exit(1)

# Step 3: submit flf2v
import local_video_client
task_id = local_video_client.submit(
    prompt=meta.get('prompt', ''),
    content=[
        {'type': 'text', 'text': meta.get('prompt', '')},
        {'type': 'image_url', 'image_url': {'url': url_start}, 'role': 'first_frame', 'priority': 'high'},
        {'type': 'image_url', 'image_url': {'url': url_end}, 'role': 'last_frame', 'priority': 'high'},
    ],
    model='flf2v',
    num_frames=49,
    width=TARGET_W, height=TARGET_H, fps=24,
)
print(f'[smoke] submitted: task_id={task_id}')

# Step 4: poll (wide stall window for coarse progress)
result = local_video_client.poll(task_id, max_attempts=90)
print(f'[smoke] poll result: {result}')

# Step 5: download + verify
out_path = shot_dir / 'output_flf2v_smoke.mp4'
local_video_client.download(task_id, str(out_path), expected_duration=2.04, expected_width=TARGET_W, expected_height=TARGET_H)
print(f'[smoke] downloaded: {out_path} ({out_path.stat().st_size}B)')
print('[smoke] ✓✓✓ SMOKE COMPLETE — extract frames to verify interpolation')

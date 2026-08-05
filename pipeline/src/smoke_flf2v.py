#!/usr/bin/env python3
"""S05 FLF2V smoke test: end-frame gen → TOS upload → Bridge flf2v submit → poll → download → verify."""
import sys, json, os
sys.path.insert(0, '.')
from pathlib import Path

OUT = Path('../output/westlake_evening_v8')
SHOT = 'S05'
shot_dir = OUT / 'shots' / SHOT
meta = json.loads((shot_dir / 'SHOT_META.json').read_text())
meta['gen_strategy'] = 'flf2v'  # old SHOT_META lacks the field; force for smoke
meta['duration'] = 2  # smoke: fast 49-frame render

first_frame = OUT / 'storyboard_images' / f'{SHOT}.png'
print(f'[smoke] first frame: {first_frame} exists={first_frame.exists()}')

# Step 1: generate end frame (first frame as primary reference, M2 logic)
from pipeline_runner import _generate_flf2v_end_frame, _read_end_frame_sidecar
end_frame = OUT / 'storyboard_images' / f'{SHOT}_end.png'
if end_frame.exists() and end_frame.stat().st_size > 1024:
    sc = _read_end_frame_sidecar(end_frame)
    print(f'[smoke] end frame exists ({end_frame.stat().st_size}B), sidecar={bool(sc)}')
else:
    print('[smoke] generating end frame via Seedream (rate-limited, may take ~2 min)...')
    ok = _generate_flf2v_end_frame(meta, SHOT, first_frame, None)
    print(f'[smoke] end frame generated: {ok}, exists={end_frame.exists()}')
    if end_frame.exists():
        sc = _read_end_frame_sidecar(end_frame)
        print(f'[smoke] validation: {json.dumps(sc.get("validation", {}), ensure_ascii=False) if sc else "no sidecar"}')

if not end_frame.exists() or end_frame.stat().st_size < 1024:
    print('[smoke] FATAL: no end frame, aborting')
    sys.exit(1)

# Step 2: upload both frames to TOS
import tos_uploader
url_start = tos_uploader.upload_image(first_frame.read_bytes(), 'image/png')
url_end = tos_uploader.upload_image(end_frame.read_bytes(), 'image/png')
print(f'[smoke] TOS start: {str(url_start)[:80]}...')
print(f'[smoke] TOS end:   {str(url_end)[:80]}...')
if not url_start or not url_end:
    print('[smoke] FATAL: TOS upload failed')
    sys.exit(1)

# Step 3: submit to Bridge with model=flf2v
import local_video_client
task_id = local_video_client.submit(
    prompt=meta.get('prompt', ''),
    content=[
        {'type': 'text', 'text': meta.get('prompt', '')},
        {'type': 'image_url', 'image_url': {'url': url_start}, 'role': 'first_frame', 'priority': 'high'},
        {'type': 'image_url', 'image_url': {'url': url_end}, 'role': 'last_frame', 'priority': 'high'},
    ],
    model='flf2v',
    num_frames=49,  # smoke: fast
    width=1280, height=720, fps=24,
)
print(f'[smoke] submitted: task_id={task_id}')

# Step 4: poll (long stall window — FLF2V coarse progress)
result = local_video_client.poll(task_id, max_attempts=90)
print(f'[smoke] poll result: {result}')

# Step 5: download + verify
out_path = shot_dir / 'output_flf2v_smoke.mp4'
local_video_client.download(task_id, str(out_path), expected_duration=2.04, expected_width=1280, expected_height=720)
print(f'[smoke] downloaded: {out_path} ({out_path.stat().st_size}B)')
print('[smoke] SMOKE COMPLETE — verify frames manually')

"""Seedance API client — submit, poll, download.

Critical: ALL params at TOP LEVEL of the JSON body.
Never nest in "parameters". watermark=false MUST be included.
"""

import os
import time
import requests
from typing import Optional
from ip_blacklist import sanitize_prompt


BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
SUBMIT_ENDPOINT = f"{BASE_URL}/contents/generations/tasks"
POLL_ENDPOINT = f"{BASE_URL}/contents/generations/tasks/{{task_id}}"


def submit(
    prompt: str,
    api_key: str,
    model: str = "doubao-seedance-2.0-mini",
    duration: int = 7,
    ratio: str = "16:9",
    first_frame_base64: Optional[str] = None,
    reference_image_base64: Optional[str] = None,
    generate_audio: str = "true",  # P0-D: 默认开启（学 OpenMontage: "sync audio is the moat"）
) -> str:
    """Submit a Seedance generation task. Returns task_id."""
    # Sanitize prompt to remove IP risks
    sanitized_prompt, filtered_terms = sanitize_prompt(prompt)
    if filtered_terms:
        print(f"  [seedance] IP filter: removed {filtered_terms}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Build content — always array format
    if reference_image_base64:
        # Try TOS upload first (avoids PrivacyInformation detection)
        image_url = None
        try:
            from tos_uploader import base64_to_signed_url
            image_url = base64_to_signed_url(reference_image_base64)
        except Exception as e:
            print(f"  [seedance] TOS upload failed: {e}")
        
        if image_url is None:
            # Fallback to base64 data URL
            image_url = f"data:image/png;base64,{reference_image_base64}"
        
        # Character identity anchor (no privacy detection, maintains consistency)
        content = [
            {"type": "text", "text": sanitized_prompt},
            {
                "type": "image_url",
                "image_url": {"url": image_url},
                "role": "reference_image",
            },
        ]
    elif first_frame_base64:
        # img2vid: video starts FROM this image (strict privacy detection)
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{first_frame_base64}"},
                "role": "first_frame",
            },
            {"type": "text", "text": sanitized_prompt},
        ]
    else:
        # txt2vid: content is array with text object
        content = [{"type": "text", "text": sanitized_prompt}]

    # ALL params at top level — CRITICAL
    payload = {
        "model": model,
        "content": content,
        "generate_audio": generate_audio,
        "ratio": ratio,
        "duration": duration,
        "watermark": False,  # MUST be included at top level
    }

    resp = requests.post(SUBMIT_ENDPOINT, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Seedance API {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    task_id = data.get("id")
    if not task_id:
        raise RuntimeError(f"No task_id in response: {data}")
    return task_id


def poll(
    task_id: str,
    api_key: str,
    max_attempts: int = 40,
    interval: int = 15,
) -> str:
    """Poll task until done. Returns video_url or raises."""
    headers = {"Authorization": f"Bearer {api_key}"}
    url = POLL_ENDPOINT.format(task_id=task_id)

    for attempt in range(1, max_attempts + 1):
        time.sleep(interval)
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")

        if status == "succeeded":
            video_url = (
                data.get("content", {}).get("video_url")
                or data.get("output", {}).get("video_url")
            )
            if not video_url:
                raise RuntimeError(f"Succeeded but no video_url: {data}")
            return video_url
        elif status in ("failed", "error"):
            raise RuntimeError(f"Task {task_id} failed: {data}")
        else:
            print(f"  [poll {attempt}/{max_attempts}] status={status}")

    raise TimeoutError(f"Task {task_id} timed out after {max_attempts * interval}s")


def download(url: str, output_path: str) -> str:
    """Download video to output_path. Returns the path."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  [download] saved {output_path} ({os.path.getsize(output_path)} bytes)")
    return output_path

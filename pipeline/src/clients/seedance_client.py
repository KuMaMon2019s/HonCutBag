"""Seedance API client — submit, poll, download.

Critical: ALL params at TOP LEVEL of the JSON body.
Never nest in "parameters". watermark=false MUST be included.
"""

import os
import tempfile
import time
from typing import Optional

import requests

from clients.video_client import VideoClient
from runtime.execution_errors import ProviderJobFailedError
from runtime.provider_responses import parse_seedance_task, parse_video_submission
from runtime.security_boundaries import redact_text
from utils.config import ARK_BASE_URL
from utils.ip_blacklist import sanitize_prompt
from utils.prompt_budget import enforce_prompt_budget


BASE_URL = ARK_BASE_URL.rstrip("/")
TASKS_ENDPOINT = f"{BASE_URL}/contents/generations/tasks"
TASK_ENDPOINT = f"{TASKS_ENDPOINT}/{{task_id}}"
# Compatibility names retained for existing submit/resume call sites.
SUBMIT_ENDPOINT = TASKS_ENDPOINT
POLL_ENDPOINT = TASK_ENDPOINT


def _authorization_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _response_json(response: requests.Response, operation: str) -> dict:
    if response.status_code != 200:
        raise RuntimeError(
            f"Seedance {operation} API {response.status_code}: "
            f"{redact_text(response.text[:500])}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Seedance {operation} returned a non-object response")
    return payload


def _validate_content_media_roles(content: list[dict]) -> None:
    """Reject Seedance media-role combinations that Ark cannot accept."""
    frame_roles = {
        item.get("role")
        for item in content
        if item.get("type") == "image_url"
        and item.get("role") in {"first_frame", "last_frame"}
    }
    reference_roles = {
        item.get("role")
        for item in content
        if item.get("role") in {"reference_image", "reference_video"}
    }
    if frame_roles and reference_roles:
        raise ValueError(
            "Seedance content cannot mix first/last frame control with reference "
            f"media (frame_roles={sorted(frame_roles)}, "
            f"reference_roles={sorted(reference_roles)})"
        )


def get_task(task_id: str, *, api_key: str, timeout: float = 30) -> dict:
    """Query one video task through the Agent Plan task endpoint."""
    if not task_id.strip():
        raise ValueError("task_id must not be empty")
    response = requests.get(
        TASK_ENDPOINT.format(task_id=task_id),
        headers=_authorization_headers(api_key),
        timeout=timeout,
    )
    return _response_json(response, "get task")


def list_tasks(
    *,
    api_key: str,
    page_num: int = 1,
    page_size: int = 20,
    filter_status: str | None = None,
    task_ids: list[str] | None = None,
    model: str | None = None,
    timeout: int = 30,
) -> dict:
    """List video tasks using Ark's documented dotted filter parameters."""
    if page_num < 1 or page_size < 1:
        raise ValueError("page_num and page_size must be positive")
    params: dict[str, int | str] = {"page_num": page_num, "page_size": page_size}
    if filter_status:
        params["filter.status"] = filter_status
    if task_ids:
        params["filter.task_ids"] = ",".join(task_ids)
    if model:
        params["filter.model"] = model
    response = requests.get(
        TASKS_ENDPOINT,
        headers=_authorization_headers(api_key),
        params=params,
        timeout=timeout,
    )
    return _response_json(response, "list tasks")


def cancel_or_delete_task(task_id: str, *, api_key: str, timeout: int = 30) -> dict:
    """Cancel or delete one video task; the provider decides by current task state."""
    if not task_id.strip():
        raise ValueError("task_id must not be empty")
    response = requests.delete(
        TASK_ENDPOINT.format(task_id=task_id),
        headers=_authorization_headers(api_key),
        timeout=timeout,
    )
    if response.status_code == 204:
        return {}
    return _response_json(response, "cancel/delete task")


def _submit_direct(
    prompt: str,
    api_key: str,
    model: str = None,
    duration: int = 7,
    ratio: str = "16:9",
    first_frame_base64: Optional[str] = None,
    reference_image_base64: Optional[str] = None,
    generate_audio: Optional[str] = None,  # P0-D: Agent Plan 不支持此参数，仅按量计费可用
    seed: int = None,  # P1-C: Seed Locking（同场景同 seed）
    reference_video_base64: Optional[str] = None,  # P1-D: 多模态组合参考
) -> str:
    """Submit a Seedance generation task. Returns task_id."""
    # 从 config 读取默认模型（如果未传入）
    if model is None:
        try:
            from utils.config import SEEDANCE_MODEL
            model = SEEDANCE_MODEL
        except ImportError:
            model = "doubao-seedance-2.0-mini"
    
    # Sanitize prompt to remove IP risks
    sanitized_prompt, filtered_terms = sanitize_prompt(prompt)
    if filtered_terms:
        print(f"  [seedance] IP filter: removed {filtered_terms}")
    enforce_prompt_budget(
        sanitized_prompt,
        provider="seedance",
        model=model,
        purpose="video_generation",
    )
    
    headers = _authorization_headers(api_key)

    # Build content — always array format
    if reference_image_base64:
        # Try TOS upload first (avoids PrivacyInformation detection)
        image_url = None
        try:
            from clients.tos_uploader import base64_to_signed_url
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

    # --- P1-D: 多模态组合参考 — 视频参考追加到 content ---
    if reference_video_base64:
        video_url = None
        try:
            from clients.tos_uploader import base64_video_to_signed_url

            video_url = base64_video_to_signed_url(reference_video_base64)
        except Exception as e:
            print(f"  [seedance] Video TOS upload failed: {e}")
        if video_url:
            content.append({
                "type": "video_url",
                "video_url": {"url": video_url},
                "role": "reference_video",
            })

    _validate_content_media_roles(content)

    # ALL params at top level — CRITICAL
    payload = {
        "model": model,
        "content": content,
        # generate_audio: Agent Plan 不支持，仅按量计费可用
        **({"generate_audio": generate_audio} if generate_audio is not None else {}),
        "ratio": ratio,
        "duration": duration,
        "watermark": False,  # MUST be included at top level
    }
    # P1-C: Seed Locking（Agent Plan API 参数必须在顶层）
    if seed is not None:
        payload["seed"] = seed

    resp = requests.post(SUBMIT_ENDPOINT, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Seedance API {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    return parse_video_submission(data, provider_id="seedance").task_id


def submit_content(
    content: list,
    api_key: str,
    model: str,
    duration: int,
    ratio: str = "16:9",
    seed: int = None,
    generate_audio: Optional[str] = None,
    timeout: float = 30,
) -> str:
    """Submit a preassembled ARK Agent Plan content array. Returns task_id."""
    _validate_content_media_roles(content)
    prompt = "\n".join(
        str(item.get("text") or "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    )
    enforce_prompt_budget(
        prompt,
        provider="seedance",
        model=model,
        purpose="video_generation",
    )
    headers = _authorization_headers(api_key)
    payload = {
        "model": model,
        "content": content,
        **({"generate_audio": generate_audio} if generate_audio is not None else {}),
        "ratio": ratio,
        "duration": duration,
        "watermark": False,
    }
    if seed is not None:
        payload["seed"] = seed

    resp = requests.post(
        SUBMIT_ENDPOINT,
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Seedance API {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    return parse_video_submission(data, provider_id="seedance").task_id


def build_video_extension_content(
    prompt: str,
    reference_video_url: str,
    *,
    reference_image_urls: list[str] | None = None,
) -> list[dict]:
    """Build Seedance content for extending one prior video with stable anchors."""
    sanitized_prompt, filtered_terms = sanitize_prompt(prompt)
    if filtered_terms:
        print(f"  [seedance] IP filter: removed {filtered_terms}")
    if not reference_video_url:
        raise ValueError("reference_video_url is required for video extension")
    content: list[dict] = [{"type": "text", "text": sanitized_prompt}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": url},
            "role": "reference_image",
        }
        for url in (reference_image_urls or [])
        if url
    )
    content.append({
        "type": "video_url",
        "video_url": {"url": reference_video_url},
        "role": "reference_video",
    })
    return content


def submit_video_extension(
    prompt: str,
    reference_video_path: str,
    *,
    api_key: str,
    model: str,
    duration: int,
    ratio: str = "16:9",
    reference_image_urls: list[str] | None = None,
    seed: int | None = None,
    generate_audio: str | None = None,
) -> str:
    """Upload a prior chunk as video and submit an explicit Seedance extension task."""
    from clients.tos_uploader import upload_media_file

    video_url = upload_media_file(reference_video_path, prefix="volcengine/video")
    if not video_url:
        raise RuntimeError(f"failed to upload reference video: {reference_video_path}")
    content = build_video_extension_content(
        prompt,
        video_url,
        reference_image_urls=reference_image_urls,
    )
    return submit_content(
        content,
        api_key=api_key,
        model=model,
        duration=duration,
        ratio=ratio,
        seed=seed,
        generate_audio=generate_audio,
    )


def submit(
    prompt: str,
    api_key: str,
    model: str = None,
    duration: int = 7,
    ratio: str = "16:9",
    first_frame_base64: Optional[str] = None,
    reference_image_base64: Optional[str] = None,
    generate_audio: Optional[str] = None,
    seed: int = None,
    reference_video_base64: Optional[str] = None,
) -> str:
    """Submit through Bridge by default, retaining Ark as direct fallback."""
    client = VideoClient(provider="seedance", direct_generator=_submit_direct)
    result = client.generate(
        prompt,
        api_key=api_key,
        model=model,
        duration=duration,
        ratio=ratio,
        first_frame_base64=first_frame_base64,
        reference_image_base64=reference_image_base64,
        generate_audio=generate_audio,
        seed=seed,
        reference_video_base64=reference_video_base64,
    )
    return str(result.value)


def poll(
    task_id: str,
    api_key: str,
    max_attempts: int = 40,
    interval: int = 15,
    request_timeout: float = 30,
) -> str:
    """Poll task until done. Returns video_url or raises."""
    for attempt in range(1, max_attempts + 1):
        time.sleep(interval)
        data = get_task(task_id, api_key=api_key, timeout=request_timeout)
        task = parse_seedance_task(data)
        status = task.status

        if status == "succeeded":
            video_url = (
                task.content.get("video_url")
                or task.output.get("video_url")
            )
            if not video_url:
                raise RuntimeError(f"Succeeded but no video_url: {data}")
            return video_url
        elif status in ("failed", "error"):
            raise ProviderJobFailedError(f"Task {task_id} failed: {data}")
        else:
            print(f"  [poll {attempt}/{max_attempts}] status={status}")

    raise TimeoutError(f"Task {task_id} timed out after {max_attempts * interval}s")


def download(url: str, output_path: str) -> str:
    """Download atomically so an interrupted response never looks complete."""
    destination = os.path.abspath(output_path)
    destination_dir = os.path.dirname(destination) or "."
    os.makedirs(destination_dir, exist_ok=True)
    temporary_path = None
    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{os.path.basename(destination)}.",
                suffix=".part",
                dir=destination_dir,
                delete=False,
            ) as target:
                temporary_path = target.name
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        if not temporary_path or os.path.getsize(temporary_path) <= 0:
            raise RuntimeError("Seedance download returned an empty file")
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    print(f"  [download] saved {destination} ({os.path.getsize(destination)} bytes)")
    return destination

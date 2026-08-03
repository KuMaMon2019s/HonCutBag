"""Local video generation API client — HonCutBag ComfyUI backend.

Connects to a local ComfyUI-based video generation API at http://192.168.31.221:9100
Supports I2V (image-to-video) and T2V (text-to-video) generation.

API Spec (HonCutBag_API_v3.1):
- POST /generate: {prompt, image_urls[], image_base64_list[], steps, num_frames, cfg, seed, width, height, fps}
- GET /status/{task_id}: {status: "queued|running|completed|failed", progress: 0-100}
- GET /download/{task_id}: returns mp4 file
"""

import os
import time
import requests
from typing import Optional, List


# Default local API URL (can be overridden via config or env)
DEFAULT_API_URL = "http://192.168.31.221:9100"

# Bypass proxy for local network requests
_NO_PROXY_ENV = {
    "NO_PROXY": "192.168.*,10.*,172.16.*,localhost,127.0.0.1",
    "no_proxy": "192.168.*,10.*,172.16.*,localhost,127.0.0.1",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "http_proxy": "",
    "https_proxy": "",
}


def _get_api_url() -> str:
    """Get the local video API URL from config or env."""
    try:
        from config import LOCAL_VIDEO_API_URL
        return LOCAL_VIDEO_API_URL
    except ImportError:
        return os.environ.get("LOCAL_VIDEO_API_URL", DEFAULT_API_URL)


def _request_session() -> requests.Session:
    """Create a requests session that bypasses proxy for local network."""
    session = requests.Session()
    # Explicitly disable proxies for this session
    session.proxies = {
        "http": None,
        "https": None,
    }
    session.trust_env = False  # Don't use system proxy
    return session


def is_available(timeout: float = 3.0) -> bool:
    """Check if the local video API is reachable.
    
    Returns True if the API responds, False otherwise.
    """
    api_url = _get_api_url()
    try:
        session = _request_session()
        resp = session.get(f"{api_url}/status", timeout=timeout)
        return resp.status_code in (200, 404)  # 404 means server is up, endpoint just different
    except Exception:
        # Try root endpoint as fallback health check
        try:
            session = _request_session()
            resp = session.get(api_url, timeout=timeout)
            return resp.status_code < 500
        except Exception:
            return False


def submit(
    prompt: str,
    image_base64_list: Optional[List[str]] = None,
    image_urls: Optional[List[str]] = None,
    steps: int = 20,
    num_frames: int = 73,  # ~3 seconds at 24fps
    cfg: float = 7.0,
    seed: int = -1,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    timeout: int = 30,
) -> str:
    """Submit a video generation task to the local API.
    
    Args:
        prompt: Text description of the video to generate
        image_base64_list: List of base64-encoded images for I2V (optional)
        image_urls: List of image URLs for I2V (optional)
        steps: Number of diffusion steps
        num_frames: Number of frames to generate
        cfg: Classifier-free guidance scale
        seed: Random seed (-1 for random)
        width: Output video width
        height: Output video height
        fps: Output video framerate
        timeout: Request timeout in seconds
    
    Returns:
        task_id: Unique task identifier for polling
    
    Raises:
        RuntimeError: If the API is unreachable or returns an error
    """
    api_url = _get_api_url()
    session = _request_session()
    
    payload = {
        "prompt": prompt,
        "steps": steps,
        "num_frames": num_frames,
        "cfg": cfg,
        "seed": seed,
        "width": width,
        "height": height,
        "fps": fps,
    }
    
    if image_base64_list:
        payload["image_base64_list"] = image_base64_list
    if image_urls:
        payload["image_urls"] = image_urls
    
    try:
        resp = session.post(
            f"{api_url}/generate",
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Local video API unreachable at {api_url}: {e}")
    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"Local video API timeout: {e}")
    
    if resp.status_code != 200:
        raise RuntimeError(f"Local video API error {resp.status_code}: {resp.text[:500]}")
    
    data = resp.json()
    task_id = data.get("task_id") or data.get("id")
    if not task_id:
        raise RuntimeError(f"No task_id in local API response: {data}")
    
    return task_id


def poll(
    task_id: str,
    max_attempts: int = 60,
    interval: int = 10,
) -> dict:
    """Poll task until done.
    
    Returns:
        dict with keys: status ("completed"|"failed"|"running"|"queued"), progress (0-100)
    
    Raises:
        TimeoutError: If polling exceeds max_attempts * interval
        RuntimeError: If the task fails
    """
    api_url = _get_api_url()
    session = _request_session()
    url = f"{api_url}/status/{task_id}"
    
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [local_poll {attempt}/{max_attempts}] error: {e}")
            time.sleep(interval)
            continue
        
        # Check for error responses (e.g., {"error": "task not found"})
        if "error" in data and data["error"] is not None:
            error_msg = data["error"]
            if "not found" in error_msg.lower():
                raise RuntimeError(f"Local video task {task_id} not found in Bridge database")
            # Other errors
            raise RuntimeError(f"Local video task {task_id} error: {error_msg}")
        
        status = data.get("status", "unknown")
        progress = data.get("progress", 0)
        
        if status == "completed":
            return {"status": "completed", "progress": 100}
        elif status == "failed":
            error_msg = data.get("error", "unknown error")
            raise RuntimeError(f"Local video task {task_id} failed: {error_msg}")
        elif status == "unknown":
            # Unknown status is suspicious, log warning but continue
            print(f"  [local_poll {attempt}/{max_attempts}] WARNING: status=unknown, continuing...")
        else:
            print(f"  [local_poll {attempt}/{max_attempts}] status={status} progress={progress}%")
        
        time.sleep(interval)
    
    raise TimeoutError(f"Local video task {task_id} timed out after {max_attempts * interval}s")


def download(task_id: str, output_path: str, timeout: int = 120) -> str:
    """Download the generated video to output_path.
    
    Returns:
        The output_path on success.
    
    Raises:
        RuntimeError: If download fails
    """
    api_url = _get_api_url()
    session = _request_session()
    url = f"{api_url}/download/{task_id}"
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    try:
        resp = session.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Local video API download failed: {e}")
    
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    
    file_size = os.path.getsize(output_path)
    print(f"  [local_download] saved {output_path} ({file_size} bytes)")
    return output_path


def generate_video(
    prompt: str,
    output_path: str,
    reference_image_base64: Optional[str] = None,
    seed: int = -1,
    duration: int = 5,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
) -> str:
    """High-level function: submit + poll + download in one call.
    
    Args:
        prompt: Video description
        output_path: Where to save the output mp4
        reference_image_base64: Optional reference image for I2V
        seed: Random seed (-1 for random)
        duration: Desired video duration in seconds
        width: Output width
        height: Output height
        fps: Output framerate
    
    Returns:
        output_path on success
    
    Raises:
        RuntimeError: If any step fails
    """
    # Calculate num_frames from duration
    num_frames = int(duration * fps) + 1  # +1 for inclusive frame count
    
    # Prepare image list
    image_base64_list = None
    if reference_image_base64:
        image_base64_list = [reference_image_base64]
    
    # Submit
    print(f"  [local_video] submitting ({width}x{height}, {duration}s, {num_frames} frames)...")
    task_id = submit(
        prompt=prompt,
        image_base64_list=image_base64_list,
        num_frames=num_frames,
        seed=seed,
        width=width,
        height=height,
        fps=fps,
    )
    print(f"  [local_video] task_id={task_id}")
    
    # Poll
    result = poll(task_id)
    print(f"  [local_video] completed!")
    
    # Download
    download(task_id, output_path)
    return output_path

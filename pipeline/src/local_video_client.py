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
    max_attempts: int = 30,  # 语义改为：连续无进度轮询次数上限（默认 30 × 10s = 300s 无进度增长才判定卡死）
    interval: int = 10,
) -> dict:
    """Poll task until done, with progress-based timeout.

    只要任务还在 running/queued 且 progress 在增长，就一直等下去（适合 8GB 显存的慢机器，
    单任务可能 20-60 分钟）。只有当 progress 连续 max_attempts 次轮询都没有任何增长，
    且 status 仍是 running/queued 时，才判定卡死并 raise TimeoutError。

    Bridge v3.2 quirk: 任务完成后 status 可能永远卡在 "running" + progress=100，
    但 GET /download/{task_id} 能正常下载。当检测到 progress>=100 且 status=running 时，
    主动探测 /download 端点：连续 3 轮(30s)探测成功 → 判定完成；连续 10 轮(100s)探测失败 → 判定卡死。

    Args:
        task_id: 任务 ID
        max_attempts: 连续无进度增长的轮询次数上限。默认 30（即 30 × interval 秒无进度增长则超时）。
                      参数名保留以向后兼容，语义已从"总轮询次数"改为"无进度超时轮数"。
        interval: 轮询间隔（秒）

    Returns:
        dict with keys: status ("completed"), progress (100)

    Raises:
        TimeoutError: 如果 progress 连续 max_attempts * interval 秒没有增长
        RuntimeError: 如果任务失败或返回 error 字段
    """
    api_url = _get_api_url()
    session = _request_session()
    url = f"{api_url}/status/{task_id}"

    last_progress: float = -1.0
    stale_count: int = 0          # 连续无进度增长的轮询次数
    total_polls: int = 0          # 总轮询次数（仅用于日志）
    last_progress_change_poll: int = 0  # 上次 progress 变化时的 total_polls 序号

    # Bridge running/100 quirk tracking
    at_100_count: int = 0         # progress>=100 且 status=running 的连续轮数
    download_probe_success: int = 0  # /download 探测连续成功轮数
    download_probe_fail: int = 0     # /download 探测连续失败轮数

    while True:
        total_polls += 1
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [local_poll #{total_polls}] network error: {e}")
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
        progress = data.get("progress", 0) or 0  # guard None

        if status == "completed":
            print(f"  [local_poll #{total_polls}] ✓ completed (progress={progress}%)")
            return {"status": "completed", "progress": 100}
        elif status == "failed":
            error_msg = data.get("error", "unknown error")
            raise RuntimeError(f"Local video task {task_id} failed: {error_msg}")

        # --- Bridge v3.2 quirk: running + progress=100 ---
        if progress >= 100 and status == "running":
            at_100_count += 1
            # Probe /download endpoint
            probe_ok = _probe_download(session, api_url, task_id)
            if probe_ok:
                download_probe_success += 1
                download_probe_fail = 0
                print(f"  [local_poll #{total_polls}] download-probe OK ({download_probe_success}/3 consecutive)")
                if download_probe_success >= 3:
                    print(f"  [local_poll #{total_polls}] ✓ Bridge running/100 quirk: download probe succeeded 3x → treating as completed")
                    return {"status": "completed", "progress": 100}
            else:
                download_probe_fail += 1
                download_probe_success = 0
                print(f"  [local_poll #{total_polls}] download-probe FAIL ({download_probe_fail}/10 consecutive)")
                if download_probe_fail >= 10:
                    raise TimeoutError(
                        f"Local video task {task_id} stalled: progress=100% status=running, "
                        f"download probe failed {download_probe_fail} consecutive times"
                    )
            time.sleep(interval)
            continue

        # Reset quirk counters when not in the 100/running state
        at_100_count = 0
        download_probe_success = 0
        download_probe_fail = 0

        # Progress-based stall detection (unchanged for progress < 100)
        if progress > last_progress:
            if stale_count > 0:
                print(f"  [local_poll #{total_polls}] ↗ progress resumed: {last_progress}% → {progress}%")
            last_progress = progress
            stale_count = 0
            last_progress_change_poll = total_polls
        else:
            stale_count += 1

        stale_seconds = stale_count * interval
        if status == "unknown":
            print(f"  [local_poll #{total_polls}] WARNING: status=unknown progress={progress}% (no-progress wait {stale_seconds}s / {max_attempts * interval}s)")
        else:
            print(f"  [local_poll #{total_polls}] status={status} progress={progress}% (no-progress wait {stale_seconds}s / {max_attempts * interval}s)")

        if stale_count >= max_attempts and status in ("running", "queued", "unknown"):
            raise TimeoutError(
                f"Local video task {task_id} stalled: progress stuck at {last_progress}% "
                f"for {stale_count} polls ({stale_seconds}s) with status={status}"
            )

        time.sleep(interval)


def _probe_download(session, api_url: str, task_id: str) -> bool:
    """Probe /download/{task_id} to check if video is ready (Bridge running/100 quirk).

    Uses stream=True and reads only headers + first few bytes to avoid downloading
    the entire file. Returns True if HTTP 200 with reasonable content-type/length.
    """
    url = f"{api_url}/download/{task_id}"
    try:
        resp = session.get(url, stream=True, timeout=10)
        if resp.status_code == 200:
            # Check content-type looks like video
            ct = resp.headers.get("content-type", "")
            cl = resp.headers.get("content-length", "0")
            # Read a small chunk to verify it's real data
            chunk = next(resp.iter_content(chunk_size=1024), b"")
            resp.close()
            if chunk and len(chunk) > 0:
                return True
        else:
            resp.close()
        return False
    except Exception as e:
        print(f"  [download-probe] exception: {e}")
        return False


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

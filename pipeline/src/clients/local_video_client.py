"""Local video generation API client — HonCutBag ComfyUI backend.

Connects to a configurable local ComfyUI-based video generation API.
Supports I2V (image-to-video) and T2V (text-to-video) generation.

API Spec (HonCutBag_API_v3.1):
- POST /generate: {prompt, image_urls[], image_base64_list[], steps, num_frames, cfg, seed, width, height, fps}
- GET /status/{task_id}: {status: "queued|running|completed|failed", progress: 0-100}
- GET /download/{task_id}: returns mp4 file
"""

import base64
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import requests

from runtime.provider_responses import parse_video_submission

from utils.prompt_budget import enforce_prompt_budget

# Default local API URL (can be overridden via config or env)
DEFAULT_API_URL = "http://127.0.0.1:9100"

# Model-specific generation constraints. Additional Bridge routes can use the
# same ``fps``/``valid_frames`` shape when their safe frame sets are verified.
MODEL_PROFILES = {
    "wan22": {
        "fps": 24,
        "valid_frames": [49, 97, 145],
    },
}

# Bypass proxy for local network requests
_NO_PROXY_ENV = {
    "NO_PROXY": "192.168.*,10.*,172.16.*,localhost,127.0.0.1",
    "no_proxy": "192.168.*,10.*,172.16.*,localhost,127.0.0.1",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "http_proxy": "",
    "https_proxy": "",
}


def _valid_frames_for_model(model: str | None) -> tuple[int, list[int]]:
    """Return the model fps and legal frame counts, applying the env override."""
    model_name = model or "wan22"
    if model_name == "seedance":
        return 24, [120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360]
    profile = MODEL_PROFILES.get(model_name, MODEL_PROFILES["wan22"])
    fps = int(profile["fps"])
    configured = os.environ.get("LOCAL_VIDEO_VALID_FRAMES")
    if configured:
        try:
            valid_frames = sorted({int(value.strip()) for value in configured.split(",") if value.strip()})
        except ValueError as exc:
            raise ValueError("LOCAL_VIDEO_VALID_FRAMES must be a comma-separated list of integers") from exc
        if not valid_frames or valid_frames[0] <= 0:
            raise ValueError("LOCAL_VIDEO_VALID_FRAMES must contain positive frame counts")
    else:
        valid_frames = list(profile["valid_frames"])
    return fps, valid_frames


def snap_duration_to_frames(
    duration_s: float,
    fps: int,
    valid_frames: Sequence[int],
) -> tuple[int, float, str]:
    """Snap a requested duration to the nearest legal frame count.

    Equidistant choices select the larger count so a tie never silently
    shortens a shot.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    legal = sorted({int(value) for value in valid_frames})
    if not legal or legal[0] <= 0:
        raise ValueError("valid_frames must contain positive frame counts")
    requested_duration = float(duration_s)
    if requested_duration <= 0:
        raise ValueError("duration_s must be positive")

    target = round(requested_duration * fps) + 1
    frames = min(legal, key=lambda value: (abs(value - target), -value))
    actual_duration = frames / fps
    reason = f"target={target}, nearest legal frame count; ties prefer larger"
    return frames, actual_duration, reason


def _get_api_url() -> str:
    """Get the local video API URL from config or env."""
    try:
        from utils.config import get_bridge_api_url
        return os.environ.get("LOCAL_VIDEO_API_URL", get_bridge_api_url())
    except ImportError:
        return os.environ.get("LOCAL_VIDEO_API_URL", os.environ.get("BRIDGE_API_URL", DEFAULT_API_URL))


def get_api_url() -> str:
    """Return the active Bridge endpoint for durable task resume checks."""

    return _get_api_url()


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


def _get_ark_api_key() -> str:
    """Return the ARK credential used by Bridge-backed Seedance routes."""
    api_key = os.environ.get("ARK_AGENT_API_KEY")
    if not api_key:
        raise ValueError("ARK_AGENT_API_KEY not set")
    return api_key


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
    image_base64_list: list[str] | None = None,
    image_urls: list[str] | None = None,
    steps: int = 20,
    num_frames: int = 73,  # ~3 seconds at 24fps
    cfg: float = 7.0,
    seed: int = -1,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    timeout: int = 30,
    asset_zip_path: str | None = None,
    content: list[dict] | None = None,
    batch_id: str | None = None,
    model: str | None = None,
    return_last_frame: bool = False,
    task_dir: str | None = None,
) -> str | None:
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
        asset_zip_path: Optional path to zip file containing assets (priority over image_base64_list)
        content: Optional Bridge content[] list (highest priority, uses new contract)
        batch_id: Optional identifier shared by all shots in one pipeline run
        model: Optional Bridge model route (wan22, phantom, or flf2v)
    
    Returns:
        task_id: Unique task identifier for polling, or None if zip not supported
    
    Raises:
        RuntimeError: If the API is unreachable or returns an error
    """
    submitted_prompt = "\n".join(
        str(item.get("text") or "")
        for item in (content or [])
        if isinstance(item, dict) and item.get("type") == "text"
    ) or prompt
    enforce_prompt_budget(
        submitted_prompt,
        provider="bridge",
        model=model or "wan22",
        purpose="video_generation",
    )
    if model == "seedance":
        _get_ark_api_key()

    api_url = _get_api_url()
    session = _request_session()
    
    # --- Task-directory contract v2.0 (feature-gated by the caller) ---
    if task_dir:
        payload = {"task_dir": task_dir}
        if model is not None:
            payload["model"] = model
        if batch_id is not None:
            payload["batch_id"] = batch_id
        try:
            resp = session.post(f"{api_url}/generate", json=payload, timeout=timeout)
            if resp.status_code == 200:
                response_data = resp.json()
                return parse_video_submission(
                    response_data,
                    provider_id="bridge",
                ).task_id
            raise RuntimeError(
                f"Bridge task_dir submission failed: HTTP {resp.status_code} — {resp.text[:200]}"
            )
        except requests.exceptions.Timeout as exc:
            raise TimeoutError(f"task_dir submission timed out after {timeout}s") from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Bridge unreachable for task_dir submission: {exc}") from exc

    # [LEGACY-KEEP v2.0] content[] remains the compatibility contract.
    if content:
        try:
            payload = {
                "content": content,
                "steps": steps,
                "num_frames": num_frames,
                "cfg": cfg,
                "seed": seed,
                "width": width,
                "height": height,
                "fps": fps,
            }
            if batch_id is not None:
                payload["batch_id"] = batch_id
            if model is not None:
                payload["model"] = model
            if return_last_frame and model == "seedance":
                payload["return_last_frame"] = True
            resp = session.post(
                f"{api_url}/generate",
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                task_id = parse_video_submission(
                    data,
                    provider_id="bridge",
                ).task_id
                images_used = data.get("images_used", 0)
                warnings = data.get("warnings", [])
                if task_id:
                    print(f"  [local_submit] ✓ task_id={task_id}, images_used={images_used}")
                    if warnings:
                        print(f"  [local_submit] warnings: {warnings}")
                    return task_id
            else:
                print(f"  [local_submit] ✗ /generate with content[] failed: HTTP {resp.status_code} — {resp.text[:200]}")
                return None
        except requests.exceptions.Timeout as e:
            if model == "seedance":
                raise TimeoutError(f"Seedance submission timed out after {timeout}s") from e
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"  [local_submit] ✗ Bridge unreachable for content[]: {e}")
            return None
        except Exception as e:
            print(f"  [local_submit] ✗ content[] submission error: {e}")
            return None
    
    # Try zip upload first if provided
    if asset_zip_path:
        try:
            with open(asset_zip_path, "rb") as f:
                files = {"file": (Path(asset_zip_path).name, f, "application/zip")}
                data = {"prompt": prompt}
                resp = session.post(
                    f"{api_url}/generate_zip",
                    files=files,
                    data=data,
                    timeout=timeout,
                )
            if resp.status_code == 200:
                data = resp.json()
                return parse_video_submission(
                    data,
                    provider_id="bridge",
                ).task_id
            elif resp.status_code == 404:
                print(f"  [submit] Bridge does not support /generate_zip (404), falling back to base64")
                return None
            else:
                print(f"  [submit] /generate_zip failed with {resp.status_code}, falling back to base64")
                return None
        except requests.exceptions.ConnectionError:
            print(f"  [submit] Bridge unreachable for zip upload, falling back to base64")
            return None
        except Exception as e:
            print(f"  [submit] zip upload failed: {e}, falling back to base64")
            return None
    
    # Fallback to JSON payload
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
    if batch_id is not None:
        payload["batch_id"] = batch_id
    if model is not None:
        payload["model"] = model
    if return_last_frame and model == "seedance":
        payload["return_last_frame"] = True
    
    try:
        resp = session.post(
            f"{api_url}/generate",
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Local video API unreachable at {api_url}: {e}")
    except requests.exceptions.Timeout as e:
        if model == "seedance":
            raise TimeoutError(f"Seedance submission timed out after {timeout}s") from e
        raise RuntimeError(f"Local video API timeout: {e}")
    
    if resp.status_code != 200:
        raise RuntimeError(f"Local video API error {resp.status_code}: {resp.text[:500]}")
    
    data = resp.json()
    return parse_video_submission(data, provider_id="bridge").task_id


def poll(
    task_id: str,
    max_attempts: int | None = None,
    interval: int = 10,
    request_timeout: float = 15,
    deadline_seconds: float | None = None,
) -> dict:
    """Poll task until done, with progress-based timeout.

    Two independent timeout windows govern this loop:

    1. **Stall window** (progress < 100%): when progress fails to grow for
       ``stall_polls`` consecutive polls the task is declared stalled.
       ``stall_polls`` defaults to LOCAL_VIDEO_STALL_POLLS (env, default 60);
       an explicit *max_attempts* argument takes precedence for back-compat.

    2. **Download-probe window** (progress >= 100%, status=running): Bridge
       archives the rendered video asynchronously so /download returns 404
       briefly after progress hits 100%. We probe /download aggressively (at
       most five seconds between attempts) until VIDEO_DOWNLOAD_PROBE_TIMEOUT
       expires (default 60 seconds). LOCAL_VIDEO_DOWNLOAD_PROBE_ATTEMPTS remains
       supported as a secondary cap for backwards compatibility.

    Bridge v3.2 quirk: 任务完成后 status 可能永远卡在 "running" + progress=100，
    但 GET /download/{task_id} 能正常下载。当检测到 progress>=100 且 status=running 时，
    主动探测 /download 端点：首次探测成功即判定完成；超时后判定卡死。

    Args:
        task_id: 任务 ID
        max_attempts: 显式的连续无进度增长轮询上限。None 时使用
                      LOCAL_VIDEO_STALL_POLLS（默认 60）。参数名保留以向后兼容。
        interval: 轮询间隔（秒）

    Returns:
        dict with keys: status ("completed"), progress (100)

    Raises:
        TimeoutError: 排队超过 LOCAL_VIDEO_QUEUE_TIMEOUT，或已开始任务的 progress
            连续 max_attempts * interval 秒没有增长，或 progress=100 但
            /download 探测超过 VIDEO_DOWNLOAD_PROBE_TIMEOUT
        RuntimeError: 如果任务失败或返回 error 字段
    """
    api_url = _get_api_url()
    session = _request_session()
    url = f"{api_url}/status/{task_id}"

    last_progress: float = -1.0
    stale_count: int = 0          # 连续无进度增长的轮询次数
    total_polls: int = 0          # 总轮询次数（仅用于日志）
    last_progress_change_poll: int = 0  # 上次 progress 变化时的 total_polls 序号
    queue_started_at = time.monotonic()
    queue_timeout = (
        float(deadline_seconds)
        if deadline_seconds is not None
        else float(os.environ.get("LOCAL_VIDEO_QUEUE_TIMEOUT", "7200"))
    )
    if queue_timeout <= 0 or request_timeout <= 0:
        raise ValueError("poll deadline and request timeout must be positive")
    stall_polls = (
        max_attempts
        if max_attempts is not None
        else int(os.environ.get("LOCAL_VIDEO_STALL_POLLS", "60"))
    )
    if stall_polls <= 0:
        raise ValueError("LOCAL_VIDEO_STALL_POLLS/max_attempts must be a positive integer")

    # Download-probe window: how many consecutive /download failures at
    # progress=100% before we declare the task stalled.  Env-configurable so
    # operators can tune for slower Bridge archival without code changes.
    probe_attempts: int = int(os.environ.get("LOCAL_VIDEO_DOWNLOAD_PROBE_ATTEMPTS", "40"))
    if probe_attempts <= 0:
        raise ValueError("LOCAL_VIDEO_DOWNLOAD_PROBE_ATTEMPTS must be a positive integer")
    probe_timeout: int = int(os.environ.get("VIDEO_DOWNLOAD_PROBE_TIMEOUT", "60"))
    if probe_timeout <= 0:
        raise ValueError("VIDEO_DOWNLOAD_PROBE_TIMEOUT must be a positive integer")
    probe_interval = min(interval, 5)

    # Bridge running/100 quirk tracking
    at_100_count: int = 0         # progress>=100 且 status=running 的连续轮数
    download_probe_fail: int = 0     # /download 探测连续失败轮数
    probe_started_at: float | None = None  # monotonic time of first probe in current run

    while True:
        if time.monotonic() - queue_started_at >= queue_timeout:
            raise TimeoutError(
                f"Local video task {task_id} exceeded runtime deadline "
                f"of {queue_timeout:g}s"
            )
        total_polls += 1
        try:
            resp = session.get(url, timeout=request_timeout)
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
            content = data.get("content") if isinstance(data.get("content"), dict) else {}
            return {
                "status": "completed",
                "progress": 100,
                "last_frame_url": data.get("last_frame_url") or content.get("last_frame_url"),
            }
        elif status == "failed":
            error_msg = data.get("error", "unknown error")
            raise RuntimeError(f"Local video task {task_id} failed: {error_msg}")

        # --- Bridge v3.2 quirk: running + progress=100 ---
        if progress >= 100 and status == "running":
            at_100_count += 1
            if probe_started_at is None:
                probe_started_at = time.monotonic()
            waited_s = time.monotonic() - probe_started_at
            if download_probe_fail and waited_s >= probe_timeout:
                raise TimeoutError(
                    f"Local video task {task_id} stalled: progress=100% status=running, "
                    f"download probe failed {download_probe_fail}/{probe_attempts} times "
                    f"over {waited_s:.0f}s (timeout={probe_timeout}s)"
                )
            # Probe /download endpoint
            probe_request_timeout = min(10, max(0.1, probe_timeout - waited_s))
            probe_ok = _probe_download(
                session, api_url, task_id, timeout=probe_request_timeout
            )
            if probe_ok:
                print(f"  [local_poll #{total_polls}] ✓ download-probe OK → treating as completed")
                content = data.get("content") if isinstance(data.get("content"), dict) else {}
                return {
                    "status": "completed",
                    "progress": 100,
                    "last_frame_url": data.get("last_frame_url") or content.get("last_frame_url"),
                }
            else:
                download_probe_fail += 1
                waited_s = time.monotonic() - probe_started_at
                print(
                    f"  [local_poll #{total_polls}] download-probe FAIL "
                    f"({download_probe_fail}/{probe_attempts}) "
                    f"[{waited_s:.0f}s/{probe_timeout}s elapsed]; "
                    f"Bridge status: {data!r}"
                )
                if waited_s >= probe_timeout or download_probe_fail >= probe_attempts:
                    raise TimeoutError(
                        f"Local video task {task_id} stalled: progress=100% status=running, "
                        f"download probe failed {download_probe_fail}/{probe_attempts} times "
                        f"over {waited_s:.0f}s (timeout={probe_timeout}s)"
                    )
            time.sleep(min(probe_interval, max(0, probe_timeout - waited_s)))
            continue

        # Reset quirk counters when not in the 100/running state
        at_100_count = 0
        download_probe_fail = 0
        probe_started_at = None

        # Only an explicitly queued task receives the generous queue timeout.
        # A task reported as running at 0% has reached the execution state and
        # must be covered by normal no-progress stall detection; otherwise a
        # wedged backend can be mislabeled "queue-waiting" for two hours.
        if status == "queued":
            queue_seconds = time.monotonic() - queue_started_at
            print(
                f"  [local_poll #{total_polls}] queue-waiting: status={status} "
                f"progress={progress}% ({queue_seconds:.0f}s / {queue_timeout}s)"
            )
            if queue_seconds >= queue_timeout:
                raise TimeoutError(
                    f"Local video task {task_id} queue wait exceeded {queue_timeout}s "
                    f"with status={status} progress={progress}%"
                )
            time.sleep(interval)
            continue

        # Progress-based stall detection for tasks that have started.
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
            print(f"  [local_poll #{total_polls}] WARNING: stall-waiting status=unknown progress={progress}% (no-progress wait {stale_seconds}s / {stall_polls * interval}s)")
        else:
            print(f"  [local_poll #{total_polls}] stall-waiting: status={status} progress={progress}% (no-progress wait {stale_seconds}s / {stall_polls * interval}s)")

        if stale_count >= stall_polls and status in ("running", "queued", "unknown"):
            if progress >= 50:
                for probe_round in range(1, 4):
                    print(
                        f"  [local_poll #{total_polls}] high-progress stall: "
                        f"probing download ({probe_round}/3)"
                    )
                    if _probe_download(session, api_url, task_id):
                        print(
                            f"  [local_poll #{total_polls}] ✓ high-progress download "
                            "probe succeeded → treating as completed"
                        )
                        content = data.get("content") if isinstance(data.get("content"), dict) else {}
                        return {
                            "status": "completed",
                            "progress": 100,
                            "last_frame_url": data.get("last_frame_url") or content.get("last_frame_url"),
                        }
                    if probe_round < 3:
                        time.sleep(interval)
            raise TimeoutError(
                f"Local video task {task_id} stalled: progress stuck at {last_progress}% "
                f"for {stale_count} polls ({stale_seconds}s) with status={status}"
            )

        time.sleep(interval)


def _probe_download(
    session, api_url: str, task_id: str, timeout: float = 10
) -> bool:
    """Probe /download/{task_id} to check if video is ready (Bridge running/100 quirk).

    Uses stream=True and reads only headers + first few bytes to avoid downloading
    the entire file. Returns True if HTTP 200 with reasonable content-type/length.
    """
    url = f"{api_url}/download/{task_id}"
    try:
        resp = session.get(url, stream=True, timeout=timeout)
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


def _verify_download(
    file_path: str,
    expected_duration: float | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    require_video_stream: bool = False,
) -> dict:
    """Probe the downloaded mp4 with ffprobe and check against expectations.

    Returns:
        dict with keys: duration (float), width (int), height (int)

    Unreadable metadata is reported with ``None`` values so a successful
    download is not discarded solely because ffprobe could not inspect it.
    A RuntimeError is raised only when parsed metadata mismatches an expected
    value (duration tolerance ±1.5s; resolution must match exactly).
    """
    unavailable = {
        "duration": None,
        "width": None,
        "height": None,
        "num_frames": None,
    }
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            file_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            detail = proc.stderr.strip() or f"exit code {proc.returncode}"
            if require_video_stream:
                raise RuntimeError(
                    f"Download verification failed: ffprobe could not read video: {detail}"
                )
            print(
                f"  [local_download] WARNING: ffprobe could not read metadata "
                f"for {file_path}: {detail}",
                file=sys.stderr,
            )
            return unavailable
        info = json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        if require_video_stream:
            raise RuntimeError(
                f"Download verification failed: ffprobe could not read video: {e}"
            ) from e
        print(
            f"  [local_download] WARNING: ffprobe could not read metadata "
            f"for {file_path}: {e}",
            file=sys.stderr,
        )
        return unavailable

    streams = info.get("streams") or []
    if not any(stream.get("codec_type") == "video" for stream in streams):
        if require_video_stream:
            raise RuntimeError("Download verification failed: ffprobe found no video stream")
        print(
            f"  [local_download] WARNING: ffprobe found no video stream in {file_path}",
            file=sys.stderr,
        )
        return unavailable

    # Extract duration
    actual_duration = None
    fmt = info.get("format", {})
    if fmt.get("duration"):
        try:
            actual_duration = float(fmt["duration"])
        except (TypeError, ValueError):
            pass
    if actual_duration is None:
        for s in streams:
            if s.get("duration"):
                try:
                    actual_duration = float(s["duration"])
                    break
                except (TypeError, ValueError):
                    continue

    # Extract video stream resolution
    actual_width = None
    actual_height = None
    actual_num_frames = None
    for s in streams:
        if s.get("codec_type") == "video":
            try:
                actual_width = int(s.get("width"))
                actual_height = int(s.get("height"))
            except (TypeError, ValueError):
                pass
            try:
                if s.get("nb_frames") is not None:
                    actual_num_frames = int(s["nb_frames"])
            except (TypeError, ValueError):
                pass
            break

    actual = {
        "duration": actual_duration,
        "width": actual_width,
        "height": actual_height,
        "num_frames": actual_num_frames,
    }

    # Validate
    if expected_duration is not None and actual_duration is not None:
        if abs(actual_duration - float(expected_duration)) > 1.5:
            raise RuntimeError(
                f"Download verification failed: expected duration ~{expected_duration}s, "
                f"got {actual_duration:.2f}s (tolerance ±1.5s)"
            )
    if expected_width is not None and actual_width is not None:
        if int(actual_width) != int(expected_width):
            raise RuntimeError(
                f"Download verification failed: expected width {expected_width}, got {actual_width}"
            )
    if expected_height is not None and actual_height is not None:
        if int(actual_height) != int(expected_height):
            raise RuntimeError(
                f"Download verification failed: expected height {expected_height}, got {actual_height}"
            )

    return actual


def download(
    task_id: str,
    output_path: str,
    timeout: int = 120,
    expected_duration: float | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    verification_out: dict | None = None,
    model: str | None = None,
) -> str:
    """Download the generated video to output_path.

    Args:
        task_id: Bridge task id.
        output_path: Where to save the mp4.
        timeout: HTTP timeout in seconds.
        expected_duration: If provided, verify downloaded video duration (±1.5s tolerance).
        expected_width: If provided, verify downloaded video width (exact match).
        expected_height: If provided, verify downloaded video height (exact match).

    Returns:
        The output_path on success.

    Raises:
        RuntimeError: If download fails or verification fails. On verification
            failure the downloaded file is deleted to prevent cross-task leakage.
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

    # --- Cross-task leakage guard ---
    is_online_route = (model or "").lower() == "seedance"
    need_verify = is_online_route or any(
        v is not None for v in (expected_duration, expected_width, expected_height)
    )
    if need_verify:
        actual = None
        try:
            if is_online_route and file_size <= 1024:
                raise RuntimeError(
                    f"Download verification failed: downloaded file is too small ({file_size} bytes)"
                )
            actual = _verify_download(
                output_path,
                expected_duration=None if is_online_route else expected_duration,
                expected_width=None if is_online_route else expected_width,
                expected_height=None if is_online_route else expected_height,
                require_video_stream=is_online_route,
            )
            if all(value is None for value in actual.values()):
                print(
                    "  [local_download] WARNING: metadata unavailable; "
                    "keeping successfully downloaded file",
                    file=sys.stderr,
                )
            else:
                duration = actual.get("duration")
                duration_text = f"{duration:.2f}s" if duration is not None else "unknown"
                print(
                    f"  [local_download] ✓ verify ok: "
                    f"{actual.get('width')}x{actual.get('height')}, {duration_text}"
                )
            if verification_out is not None:
                verification_out.update(actual)
        except RuntimeError as e:
            try:
                os.remove(output_path)
                print(f"  [local_download] ✗ verification failed, deleted {output_path}")
            except OSError:
                pass
            raise RuntimeError(
                f"Download verification failed for task {task_id}: "
                f"expected {expected_width}x{expected_height}/{expected_duration}s, "
                f"got actual={actual if actual is not None else 'unknown'}"
            ) from e

    return output_path


def _save_last_frame(last_frame_value: str, output_path: str, timeout: int = 120) -> str:
    """Save a Bridge last-frame URL or base64 value next to the shot video."""
    destination = Path(output_path).parent / "last_frame.jpg"
    value = last_frame_value.strip()
    try:
        if value.startswith(("http://", "https://")):
            response = _request_session().get(value, stream=True, timeout=timeout)
            response.raise_for_status()
            with destination.open("wb") as frame_file:
                for chunk in response.iter_content(chunk_size=8192):
                    frame_file.write(chunk)
        else:
            encoded = value.split(",", 1)[1] if value.startswith("data:") else value
            destination.write_bytes(base64.b64decode(encoded, validate=True))
        if destination.stat().st_size == 0:
            raise RuntimeError("decoded last frame is empty")
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"unable to save Bridge last frame: {exc}") from exc
    print(f"  [chain] saved {destination} ({destination.stat().st_size} bytes)")
    return str(destination)


def generate_video(
    prompt: str,
    output_path: str,
    reference_image_base64: str | None = None,
    seed: int = -1,
    duration: float | None = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    asset_zip_path: str | None = None,
    image_base64_list: list[str] | None = None,
    content: list[dict] | None = None,
    batch_id: str | None = None,
    model: str | None = None,
    submit_timeout: int | None = None,
    status_timeout: float = 15,
    poll_deadline: float | None = None,
    return_last_frame: bool = False,
    task_dir: str | None = None,
    resume_task_id: str | None = None,
    on_submit_start: Callable[[], None] | None = None,
    on_submitted: Callable[[str], None] | None = None,
) -> str | dict:
    """High-level function: submit + poll + download in one call.

    Args:
        prompt: Video description
        output_path: Where to save the output mp4
        reference_image_base64: Optional reference image for I2V (single image, legacy)
        seed: Random seed (-1 for random)
        duration: Desired video duration in seconds
        width: Output width
        height: Output height
        fps: Output framerate
        asset_zip_path: Optional path to zip file containing assets (priority over base64)
        image_base64_list: Optional list of base64 images for I2V (fallback if zip not supported)
        content: Optional Bridge content[] list (highest priority, uses new contract)
        batch_id: Optional identifier shared by all shots in one pipeline run
        model: Optional Bridge model route (wan22, phantom, or flf2v)

    Returns:
        output_path on success

    Raises:
        RuntimeError: If any step fails
    """
    profile_fps, valid_frames = _valid_frames_for_model(model)
    fps = profile_fps
    frame_override = os.environ.get("LOCAL_VIDEO_NUM_FRAMES")
    shot_id = Path(output_path).parent.name
    requested_duration = float(duration) if duration is not None else None
    if frame_override is not None:
        num_frames = int(frame_override)
        expected_duration = num_frames / fps
        reason = "explicit LOCAL_VIDEO_NUM_FRAMES override"
        print(f"  [frames] override={num_frames} ({expected_duration:.2f}s)")
    elif requested_duration is not None:
        num_frames, expected_duration, reason = snap_duration_to_frames(
            requested_duration, fps, valid_frames
        )
        target = round(requested_duration * fps) + 1
        print(
            f"  [frames] {shot_id}: duration={requested_duration:.1f}s → "
            f"target={target} → snapped {num_frames} ({expected_duration:.2f}s)"
        )
    else:
        num_frames = valid_frames[len(valid_frames) // 2]
        expected_duration = num_frames / fps
        reason = "missing shot duration; middle valid frame count fallback"
        print(f"  [frames] {shot_id}: duration=missing → fallback {num_frames} ({expected_duration:.2f}s)")

    # Prepare image list (legacy single image support)
    if reference_image_base64 and not image_base64_list:
        image_base64_list = [reference_image_base64]

    task_id = resume_task_id
    if task_id:
        print(f"  [local_video] resuming task_id={task_id}")
    else:
        print(
            f"  [local_video] submitting ({width}x{height}, num_frames={num_frames}, "
            f"expected_duration={expected_duration:.2f}s)..."
        )
        if on_submit_start is not None:
            on_submit_start()
        task_id = submit(
            prompt=prompt,
            image_base64_list=image_base64_list,
            num_frames=num_frames,
            seed=seed,
            width=width,
            height=height,
            fps=fps,
            asset_zip_path=asset_zip_path,
            content=content,
            batch_id=batch_id,
            model=model,
            return_last_frame=return_last_frame,
            task_dir=task_dir,
            timeout=submit_timeout or (60 if model == "seedance" else 30),
        )

        # Handle content[] failure - fallback to zip/base64
        if task_id is None and content is not None and task_dir is None:
            print("  [local_video] content[] failed, falling back to zip/base64")
            if on_submit_start is not None:
                on_submit_start()
            task_id = submit(
                prompt=prompt,
                image_base64_list=image_base64_list,
                num_frames=num_frames,
                seed=seed,
                width=width,
                height=height,
                fps=fps,
                asset_zip_path=asset_zip_path,
                batch_id=batch_id,
                model=model,
                return_last_frame=return_last_frame,
                timeout=submit_timeout or (60 if model == "seedance" else 30),
            )

        # Handle zip upload failure - fallback to base64
        if task_id is None and asset_zip_path is not None:
            print("  [local_video] zip upload failed, falling back to base64 list")
            if on_submit_start is not None:
                on_submit_start()
            task_id = submit(
                prompt=prompt,
                image_base64_list=image_base64_list,
                num_frames=num_frames,
                seed=seed,
                width=width,
                height=height,
                fps=fps,
                batch_id=batch_id,
                model=model,
                return_last_frame=return_last_frame,
                timeout=submit_timeout or (60 if model == "seedance" else 30),
            )

        if task_id is None:
            raise RuntimeError("Failed to submit video generation task")
        if on_submitted is not None:
            on_submitted(task_id)

    print(f"  [local_video] task_id={task_id}")

    # Poll
    poll_kwargs = {}
    if status_timeout != 15:
        poll_kwargs["request_timeout"] = status_timeout
    if poll_deadline is not None:
        poll_kwargs["deadline_seconds"] = poll_deadline
    result = poll(task_id, **poll_kwargs)
    print("  [local_video] completed!")

    # Download (with cross-task leakage verification)
    verification = {}
    download(
        task_id,
        output_path,
        expected_duration=expected_duration,
        expected_width=width,
        expected_height=height,
        verification_out=verification,
        model=model,
    )

    last_frame_path = None
    if return_last_frame and model == "seedance":
        last_frame_value = result.get("last_frame_url")
        if last_frame_value:
            try:
                last_frame_path = _save_last_frame(last_frame_value, output_path)
            except RuntimeError as exc:
                print(f"    [chain] {shot_id}: 尾帧保存失败，降级为独立生成 — {exc}", flush=True)
        else:
            print(
                f"    [chain] {shot_id}: Bridge 未返回尾帧（Bridge 需升级到 v3.5+ 契约），降级为独立生成",
                flush=True,
            )

    actual_duration = verification.get("duration")
    actual_num_frames = verification.get("num_frames")
    if actual_num_frames is None and actual_duration is not None:
        actual_num_frames = round(actual_duration * fps)
    provenance = {
        "actual_model": model or "wan22",
        "requested_duration": requested_duration,
        "requested_num_frames": num_frames,
        "actual_duration": actual_duration,
        "actual_num_frames": actual_num_frames,
    }
    meta_path = Path(output_path).parent / "SHOT_META.json"
    if meta_path.exists():
        try:
            shot_meta = json.loads(meta_path.read_text())
            shot_meta.update(provenance)
            meta_path.write_text(json.dumps(shot_meta, indent=2, ensure_ascii=False))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [frames] {shot_id}: unable to update provenance: {exc}")
    print(
        f"  [frames] {shot_id}: requested_duration={requested_duration}, "
        f"requested_num_frames={num_frames}, actual_duration={actual_duration}, "
        f"actual_num_frames={actual_num_frames}; reason={reason}"
    )
    if requested_duration is not None and actual_duration is not None:
        if abs(actual_duration - requested_duration) > 1.0:
            print(
                f"  [frames] WARNING duration drift: {shot_id} requested "
                f"{requested_duration:.2f}s, actual {actual_duration:.2f}s"
            )
    if not return_last_frame:
        return output_path
    return {
        "output_path": output_path,
        "last_frame_path": last_frame_path,
        "actual_model": model or "wan22",
    }


def generate_video_with_fallback(**kwargs) -> str | dict:
    """Generate with Seedance, falling back to Wan2.2 only on a timeout.

    The same prompt and reference payload are retained because Wan2.2 accepts
    the positive identity/negative guardrails used by Seedance. Calling
    ``generate_video`` again with ``model=wan22`` deliberately re-snaps the
    requested duration to Wan2.2's legal 49/97/145 frame profile and enables
    strict duration/dimension verification.
    """
    output_path = str(kwargs["output_path"])
    shot_id = Path(output_path).parent.name
    seedance_kwargs = dict(kwargs, model="seedance")
    seedance_kwargs.setdefault("submit_timeout", 60)
    try:
        return generate_video(**seedance_kwargs)
    except TimeoutError:
        requested_duration = kwargs.get("duration")
        duration_loss = ""
        if requested_duration is not None and float(requested_duration) > 6:
            requested_label = f"{float(requested_duration):g}"
            duration_loss = (
                f" (duration {requested_label}s → 6s, Wan2.2 max ~6s)"
            )
        print(
            f"    [fallback] {shot_id}: Seedance timeout → Wan2.2{duration_loss}",
            flush=True,
        )
        wan_kwargs = dict(kwargs, model="wan22")
        wan_kwargs.pop("submit_timeout", None)
        wan_kwargs.pop("resume_task_id", None)
        return generate_video(**wan_kwargs)

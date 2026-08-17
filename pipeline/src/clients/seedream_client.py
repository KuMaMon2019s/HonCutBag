"""Seedream API client — text-to-image + image-to-image via Volcano Ark Agent Plan.

Agent Plan endpoint: POST /api/plan/v3/images/generations (synchronous)
Model: doubao-seedream-5.0-lite
Response: data[].url or data[].b64_json (no polling needed)

Usage:
    from clients.seedream_client import SeedreamClient
    client = SeedreamClient()
    # text-to-image
    url = client.text_to_image("狐耳少女弹吉他", output_path="output.png")
    # image-to-image (reference mode)
    url = client.image_to_image("same character, side view", ref_image="front.png", output_path="side.png")
"""

import base64
import io
import os
import threading
import time
from contextlib import contextmanager

import requests
from PIL import Image, ImageOps

from utils.config import ARK_BASE_URL
from utils.ip_blacklist import sanitize_prompt

# Agent Plan base URL (NOT /api/v3/ which is pay-as-you-go)
BASE_URL = ARK_BASE_URL.rstrip("/")
IMAGE_ENDPOINT = f"{BASE_URL}/images/generations"

# Agent Plan model (NOT doubao-seedream-3-0 which doesn't exist)
DEFAULT_MODEL = "doubao-seedream-5.0-lite"
DEFAULT_IMAGE_SIZE = "1920x1920"
REFERENCE_UPLOAD_MAX_EDGE = 1600
REFERENCE_UPLOAD_MAX_BYTES = 1_500_000


def _reference_data_url(reference_path: str) -> str:
    """Encode one reference without sending oversized raw PNG payloads."""
    with open(reference_path, "rb") as reference_file:
        image_bytes = reference_file.read()
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif image_bytes.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        ext = os.path.splitext(reference_path)[1].lower().lstrip(".")
        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }.get(ext, "image/png")

    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            width, height = source.size
            needs_compaction = (
                len(image_bytes) > REFERENCE_UPLOAD_MAX_BYTES
                or max(width, height) > REFERENCE_UPLOAD_MAX_EDGE
            )
            if needs_compaction:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail(
                    (REFERENCE_UPLOAD_MAX_EDGE, REFERENCE_UPLOAD_MAX_EDGE),
                    Image.Resampling.LANCZOS,
                )
                compact = io.BytesIO()
                image.save(
                    compact,
                    format="JPEG",
                    quality=90,
                    subsampling=0,
                    optimize=True,
                )
                compact_bytes = compact.getvalue()
                print(
                    f"  [seedream] compact reference {os.path.basename(reference_path)}: "
                    f"{len(image_bytes)} → {len(compact_bytes)} bytes, "
                    f"{width}x{height} → {image.width}x{image.height}",
                    flush=True,
                )
                image_bytes = compact_bytes
                mime = "image/jpeg"
    except (OSError, ValueError):
        # Preserve the provider's original validation behavior for unusual but
        # otherwise supported image formats.
        pass

    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"


class _SeedreamRateLimiter:
    """Serialize Seedream calls and space request starts across all clients."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_request_started = 0.0

    @staticmethod
    def _min_interval() -> float:
        raw_value = os.environ.get("SEEDREAM_MIN_INTERVAL", "120")
        try:
            return max(0.0, float(raw_value))
        except ValueError:
            print(
                f"  [seedream] invalid SEEDREAM_MIN_INTERVAL={raw_value!r}; using 120s",
                flush=True,
            )
            return 120.0

    @contextmanager
    def request_slot(self):
        # Keep the lock for the complete API call. This guarantees a single
        # in-flight request, not merely spaced request starts.
        with self._lock:
            wait_seconds = self._min_interval() - (
                time.monotonic() - self._last_request_started
            )
            if wait_seconds > 0:
                print(
                    f"  [seedream] rate limit: waiting {wait_seconds:.1f}s",
                    flush=True,
                )
                time.sleep(wait_seconds)
            self._last_request_started = time.monotonic()
            yield


_SEEDREAM_RATE_LIMITER = _SeedreamRateLimiter()


class AgentPlanQuotaExceededError(RuntimeError):
    """Agent Plan repeatedly returned AccountQuotaExceeded after retries."""


class SeedreamAPIError(requests.exceptions.HTTPError):
    """Seedream rejected a request and returned structured provider context."""

    def __init__(
        self,
        *,
        status_code: int,
        provider_code: str,
        provider_message: str,
        request_id: str | None,
        response: requests.Response,
    ):
        details = f"Seedream API HTTP {status_code}"
        if provider_code:
            details += f" {provider_code}"
        if provider_message:
            details += f": {provider_message}"
        if request_id:
            details += f" (request_id={request_id})"
        super().__init__(details, response=response, request=response.request)
        self.status_code = status_code
        self.provider_code = provider_code
        self.provider_message = provider_message
        self.request_id = request_id


def _provider_error_details(
    response: requests.Response,
    diagnostic_headers: dict[str, str],
) -> tuple[str, str, str | None]:
    """Extract the stable error fields callers need for bounded recovery."""
    provider_code = ""
    provider_message = ""
    try:
        payload = response.json()
    except (requests.exceptions.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            provider_code = str(error.get("code") or "").strip()
            provider_message = str(error.get("message") or "").strip()
    request_id = (
        diagnostic_headers.get("x-request-id")
        or diagnostic_headers.get("x-tt-logid")
        or None
    )
    return provider_code, provider_message, request_id


class SeedreamClient:
    """Seedream image generation client for Volcano Ark Agent Plan API."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("ARK_AGENT_API_KEY", "")
        if not self.api_key:
            raise ValueError("ARK_AGENT_API_KEY not set. Export it or pass api_key=.")
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def text_to_image(
        self,
        prompt: str,
        output_path: str = "output.png",
        size: str = DEFAULT_IMAGE_SIZE,
        timeout: int = 60,
    ) -> str:
        """Generate image from text prompt. Returns image URL.

        Agent Plan API is synchronous — no polling needed.
        """
        # Sanitize prompt to remove IP risks
        sanitized_prompt, filtered_terms = sanitize_prompt(prompt)
        if filtered_terms:
            print(f"  [seedream] IP filter: removed {filtered_terms}")

        payload = {
            "model": self.model,
            "prompt": sanitized_prompt,
            "size": size,
            "response_format": "url",
            "watermark": False,
            "sequential_image_generation": "disabled",
        }

        return self._call_and_save(payload, output_path, timeout=timeout)

    def image_to_image(
        self,
        prompt: str,
        ref_image: str | list[str],
        output_path: str = "output.png",
        size: str = DEFAULT_IMAGE_SIZE,
    ) -> str:
        """Generate image from text + reference image (i2i / reference mode).

        Args:
            prompt: Text description for generation
            ref_image: One reference path or multiple character reference paths
            output_path: Where to save result
            size: Output dimensions WxH
        """
        reference_paths = [ref_image] if isinstance(ref_image, str) else list(ref_image)
        if not reference_paths:
            raise ValueError("image_to_image requires at least one reference image")
        encoded_references = []
        for reference_path in reference_paths:
            encoded_references.append(_reference_data_url(reference_path))

        # Agent Plan i2i: use image_url in content array
        payload = {
            "model": self.model,
            "prompt": prompt,
            "image": (
                encoded_references[0]
                if len(encoded_references) == 1
                else encoded_references
            ),
            "size": size,
            "n": 1,
            "output_format": "png",
            "watermark": False,
        }

        return self._call_and_save(payload, output_path)

    def generate_three_views(
        self,
        character_desc: str,
        style: str = "",
        negative: str = "",
        output_dir: str = ".",
        size: str = DEFAULT_IMAGE_SIZE,
    ) -> dict:
        """Generate front/side/back three-view character sheets.

        Args:
            character_desc: Full character description
            style: Art style (e.g. "张艺谋式写实, 35mm film, 自然光")
            negative: Negative prompt (appended to each view)
            output_dir: Directory to save front.png, side.png, back.png
            size: Image dimensions (min 1920x1920 for Agent Plan)

        Returns:
            dict with keys "front", "side", "back" mapping to file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        views = {
            "front": f"Character reference sheet, FRONT VIEW, full body standing pose facing camera directly, arms at sides, neutral expression, white background, {character_desc}. {style}",
            "side": f"Character reference sheet, SIDE VIEW (profile), full body standing pose facing right, arms at sides, neutral expression, white background, {character_desc}. {style}",
            "back": f"Character reference sheet, BACK VIEW, full body standing pose facing away from camera, arms at sides, white background, {character_desc}. {style}",
        }

        if negative:
            for k in views:
                views[k] += f" Negative: {negative}"

        results = {}
        for view_name, prompt in views.items():
            out_path = os.path.join(output_dir, f"{view_name}.png")
            print(f"  [three-view] generating {view_name}...")
            try:
                self.text_to_image(
                    prompt=prompt,
                    output_path=out_path,
                    size=size,
                )
                results[view_name] = out_path
                print(f"  [three-view] {view_name} ✓ → {out_path}")
            except Exception as e:
                print(f"  [three-view] {view_name} ✗ → {e}")
                results[view_name] = None

        return results

    def _call_and_save(self, payload: dict, output_path: str, timeout: int = 180) -> str:
        """Call Agent Plan API (synchronous), save result. Returns image URL."""
        max_quota_retries = 3
        for quota_retry in range(max_quota_retries + 1):
            with _SEEDREAM_RATE_LIMITER.request_slot():
                print(
                    f"  [seedream] calling Agent Plan API (timeout={timeout}s)...",
                    flush=True,
                )
                resp = requests.post(
                    IMAGE_ENDPOINT,
                    json=payload,
                    headers=self.headers,
                    timeout=timeout,
                )
                status_code = getattr(resp, "status_code", 200)
                diagnostic_headers = {}
                if status_code != 200:
                    diagnostic_headers = {
                        name: value
                        for name, value in resp.headers.items()
                        if name.lower() in {
                            "retry-after",
                            "x-request-id",
                            "x-tt-logid",
                        }
                        or any(
                            marker in name.lower()
                            for marker in ("rate", "limit", "retry")
                        )
                    }
                    print(
                        f"  [seedream] ✗ HTTP {status_code} "
                        f"body={resp.text[:1000]!r} headers={diagnostic_headers}",
                        flush=True,
                    )
                    provider_code, provider_message, request_id = (
                        _provider_error_details(resp, diagnostic_headers)
                    )
                    if (
                        provider_code == "AccountQuotaExceeded"
                        or "AccountQuotaExceeded" in resp.text
                    ):
                        if quota_retry < max_quota_retries:
                            print(
                                f"  [seedream] ⚠ AccountQuotaExceeded (intermittent), "
                                f"retry {quota_retry + 1}/3 in 60s...",
                                flush=True,
                            )
                            time.sleep(60)
                            continue
                        raise AgentPlanQuotaExceededError(
                            "HTTP 429 AccountQuotaExceeded persisted after 3 retries. "
                            f"Request id: {request_id or 'N/A'}"
                        )
                    raise SeedreamAPIError(
                        status_code=status_code,
                        provider_code=provider_code,
                        provider_message=provider_message,
                        request_id=request_id,
                        response=resp,
                    )
                data = resp.json()
                break

        # Agent Plan returns data[] array with url or b64_json
        if "data" not in data or len(data["data"]) == 0:
            raise RuntimeError(f"No image data in response: {data}")

        item = data["data"][0]
        image_url = item.get("url")
        b64_json = item.get("b64_json")

        if image_url:
            # Download from URL
            self._download(image_url, output_path)
            return image_url
        elif b64_json:
            # Decode base64
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            decoded = base64.b64decode(b64_json)
            with open(output_path, "wb") as f:
                f.write(decoded)
            print(f"  [download] saved {output_path} ({len(decoded)} bytes)")
            return f"b64://{output_path}"
        else:
            raise RuntimeError(f"No url or b64_json in response: {data}")

    def _download(self, url: str, output_path: str) -> str:
        """Download image to output_path."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  [download] saved {output_path} ({os.path.getsize(output_path)} bytes)")
        return output_path


# --- Convenience functions ---

def text_to_image(prompt: str, output_path: str = "output.png", **kwargs) -> str:
    """Quick text-to-image."""
    client = SeedreamClient()
    return client.text_to_image(prompt, output_path, **kwargs)


def image_to_image(prompt: str, ref_image: str, output_path: str = "output.png", **kwargs) -> str:
    """Quick image-to-image."""
    client = SeedreamClient()
    return client.image_to_image(prompt, ref_image, output_path, **kwargs)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python seedream_client.py <prompt> [output_path]")
        sys.exit(1)

    prompt = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "output.png"
    url = text_to_image(prompt, output)
    print(f"Done! URL: {url}")

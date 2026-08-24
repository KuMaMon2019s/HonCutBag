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
import binascii
import hashlib
import io
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import requests
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from prompt.seedream_image_prompt import (
    prompt_guidance_metrics,
    single_image_request_parameters,
)
from utils.config import ARK_BASE_URL
from utils.ip_blacklist import sanitize_prompt
from utils.provider_quota import (
    FixedWindowQuotaExceededError,
    is_fixed_window_quota_exhaustion,
)

# Agent Plan base URL (NOT /api/v3/ which is pay-as-you-go)
BASE_URL = ARK_BASE_URL.rstrip("/")
IMAGE_ENDPOINT = f"{BASE_URL}/images/generations"

# Agent Plan model (NOT doubao-seedream-3-0 which doesn't exist)
DEFAULT_MODEL = "doubao-seedream-5.0-lite"
DEFAULT_IMAGE_SIZE = "2K"
AGENT_PLAN_IMAGE_MODELS = frozenset({DEFAULT_MODEL})
SEEDREAM_5_LITE_SIZE_TIERS = frozenset({"2K", "3K", "4K"})
SEEDREAM_5_LITE_MIN_PIXELS = 2560 * 1440
SEEDREAM_5_LITE_MAX_PIXELS = 4096 * 4096
SEEDREAM_5_LITE_MAX_REFERENCE_IMAGES = 14
REFERENCE_UPLOAD_MAX_EDGE = 1600
REFERENCE_UPLOAD_MAX_BYTES = 1_500_000


class _SeedreamImageItem(BaseModel):
    """Validated subset of one non-streaming image response item."""

    model_config = ConfigDict(extra="ignore")

    url: str | None = None
    b64_json: str | None = None
    size: str | None = None
    output_format: str | None = None

    @model_validator(mode="after")
    def require_exactly_one_image_source(self):
        if bool(self.url) == bool(self.b64_json):
            raise ValueError("image item must contain exactly one of url or b64_json")
        return self


class _SeedreamImagesResponse(BaseModel):
    """Fail-closed envelope for the synchronous Agent Plan response."""

    model_config = ConfigDict(extra="ignore")

    data: list[_SeedreamImageItem] = Field(min_length=1)


def _validate_image_size(size: str) -> str:
    """Validate the Agent Plan Seedream 5.0 lite size contract locally."""
    normalized = str(size).strip()
    tier = normalized.upper()
    if tier in SEEDREAM_5_LITE_SIZE_TIERS:
        return tier
    match = re.fullmatch(r"(\d+)[xX](\d+)", normalized)
    if match is None:
        raise ValueError(
            "Seedream 5.0 lite size must be 2K, 3K, 4K, or WIDTHxHEIGHT"
        )
    width, height = (int(value) for value in match.groups())
    pixels = width * height
    aspect_ratio = width / height if height else 0
    if not (
        SEEDREAM_5_LITE_MIN_PIXELS <= pixels <= SEEDREAM_5_LITE_MAX_PIXELS
        and 1 / 16 <= aspect_ratio <= 16
    ):
        raise ValueError(
            "Seedream 5.0 lite size must contain 3,686,400-16,777,216 pixels "
            "with an aspect ratio between 1:16 and 16:1"
        )
    return f"{width}x{height}"


def _prepare_prompt(prompt: str) -> str:
    """Apply the existing IP policy and emit only privacy-safe diagnostics."""
    sanitized_prompt, filtered_terms = sanitize_prompt(str(prompt))
    if not sanitized_prompt.strip():
        raise ValueError("Seedream prompt must not be empty after sanitization")
    if filtered_terms:
        print(
            f"  [seedream] IP filter removed {len(filtered_terms)} term(s)",
            flush=True,
        )
    metrics = prompt_guidance_metrics(sanitized_prompt)
    if metrics["over_recommended_length"]:
        print(
            "  [seedream] prompt exceeds official length guidance: "
            f"characters={metrics['characters']} "
            f"cjk={metrics['cjk_characters']} "
            f"english_words={metrics['english_words']} "
            f"sha256={metrics['sha256']}",
            flush=True,
        )
    return sanitized_prompt


def _single_image_payload(*, model: str, prompt: str, size: str) -> dict:
    """Build the documented non-streaming, single-image quality contract."""
    return {
        "model": model,
        "prompt": _prepare_prompt(prompt),
        **single_image_request_parameters(_validate_image_size(size)),
    }


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


class AgentPlanQuotaExceededError(FixedWindowQuotaExceededError):
    """Agent Plan returned an exhausted or repeatedly unavailable quota."""


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
        if model not in AGENT_PLAN_IMAGE_MODELS:
            raise ValueError(
                "Agent Plan image generation only supports "
                f"{DEFAULT_MODEL!r}; received {model!r}"
            )
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
        payload = _single_image_payload(
            model=self.model,
            prompt=prompt,
            size=size,
        )

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
        if len(reference_paths) > SEEDREAM_5_LITE_MAX_REFERENCE_IMAGES:
            raise ValueError(
                "Seedream 5.0 lite accepts at most 14 reference images when "
                "generating one output image"
            )
        payload = _single_image_payload(
            model=self.model,
            prompt=prompt,
            size=size,
        )
        encoded_references = []
        for reference_path in reference_paths:
            encoded_references.append(_reference_data_url(reference_path))

        payload["image"] = (
            encoded_references[0]
            if len(encoded_references) == 1
            else encoded_references
        )

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
            size: Seedream resolution tier or valid WxH dimensions

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
                    response_bytes = getattr(resp, "content", None)
                    if not isinstance(response_bytes, bytes):
                        response_bytes = str(getattr(resp, "text", "")).encode(
                            "utf-8", errors="replace"
                        )
                    print(
                        f"  [seedream] ✗ HTTP {status_code} "
                        f"body_sha256={hashlib.sha256(response_bytes).hexdigest()} "
                        f"headers={diagnostic_headers}",
                        flush=True,
                    )
                    provider_code, provider_message, request_id = (
                        _provider_error_details(resp, diagnostic_headers)
                    )
                    if (
                        provider_code == "AccountQuotaExceeded"
                        or "AccountQuotaExceeded" in resp.text
                    ):
                        if is_fixed_window_quota_exhaustion(provider_message):
                            raise AgentPlanQuotaExceededError(
                                "HTTP 429 AccountQuotaExceeded: "
                                f"{provider_message or 'fixed-window quota exhausted'} "
                                f"(request_id={request_id or 'N/A'})"
                            )
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
                try:
                    data = _SeedreamImagesResponse.model_validate(resp.json())
                except (
                    requests.exceptions.JSONDecodeError,
                    ValueError,
                    ValidationError,
                ) as exc:
                    raise RuntimeError(
                        "Seedream returned an invalid non-streaming image envelope"
                    ) from exc
                break

        # Agent Plan returns a validated data[] array with url or b64_json.
        item = data.data[0]
        image_url = item.url
        b64_json = item.b64_json

        if image_url:
            # Download from URL
            self._download(image_url, output_path)
            return image_url
        elif b64_json:
            try:
                decoded = base64.b64decode(b64_json, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise RuntimeError("Seedream returned invalid base64 image data") from exc
            self._save_validated_image(decoded, output_path)
            return f"b64://{output_path}"
        raise RuntimeError("Seedream response lost its validated image source")

    @staticmethod
    def _save_validated_image(image_bytes: bytes, output_path: str) -> str:
        """Atomically persist provider bytes only after they decode as an image."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            temporary.write_bytes(image_bytes)
            with Image.open(temporary) as image:
                image.verify()
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(f"  [download] saved {destination} ({len(image_bytes)} bytes)")
        return str(destination)

    def _download(self, url: str, output_path: str) -> str:
        """Download image to output_path."""
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with temporary.open("wb") as output_file:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        output_file.write(chunk)
            with Image.open(temporary) as image:
                image.verify()
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        print(f"  [download] saved {destination} ({destination.stat().st_size} bytes)")
        return str(destination)


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

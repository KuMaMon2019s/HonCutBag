"""Volcano TOS (Object Storage) uploader for Seedance reference images.

Ported for HonCut's volcengineSd2.ts — TOS4-HMAC-SHA256 signing.
Uploads images to TOS and returns pre-signed URLs for Seedance API.
"""

import os
import hashlib
import hmac
import base64
import json
import mimetypes
import requests
import subprocess
import tempfile
from urllib.parse import parse_qs, quote, urlparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


SEEDANCE_MAX_IMAGE_BYTES = 30 * 1024 * 1024
SEEDANCE_MAX_VIDEO_BYTES = 200 * 1024 * 1024
MULTIMODAL_MAX_IMAGE_BYTES = 10 * 1024 * 1024
MULTIMODAL_MAX_VIDEO_BYTES = 50 * 1024 * 1024
MULTIMODAL_MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
MULTIMODAL_MAX_AUDIO_BYTES = 25 * 1024 * 1024
IMAGE_TRANSPORT_METADATA = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
    "BMP": ("image/bmp", ".bmp"),
    "TIFF": ("image/tiff", ".tiff"),
    "GIF": ("image/gif", ".gif"),
}
TOS_CONTENT_SHA256_METADATA = "x-tos-meta-honcut-sha256"


# ─── .env loading (same pattern as config.py) ────────────────────────────────

def _load_env_file():
    """Load .env from project root (pipeline/.env), matching config.py behavior."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if not env_file.exists():
        # Also try pipeline/.env as fallback
        env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        return
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Strip quotes
                    if value and value[0] in ('"', "'") and value[-1] == value[0]:
                        value = value[1:-1]
                    if key and key not in os.environ:
                        os.environ[key] = value
    except Exception:
        pass


_load_env_file()

# Also try python-dotenv if available
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass


# ─── TOS config ───────────────────────────────────────────────────────────────

def _get_tos_config() -> dict:
    """Load TOS config from environment."""
    return {
        "ak": os.environ.get("TOS_ACCESS_KEY", ""),
        "sk": os.environ.get("TOS_SECRET_KEY", ""),
        "endpoint": os.environ.get("TOS_ENDPOINT", "tos-cn-beijing.volces.com"),
        "bucket": os.environ.get("TOS_BUCKET", ""),
        "region": os.environ.get("TOS_REGION", "cn-beijing"),
    }


def is_media_upload_configured() -> bool:
    """Return whether reference media can be uploaded without exposing secrets."""
    config = _get_tos_config()
    return all((config["ak"], config["sk"], config["bucket"]))


class TOSMediaUploadError(RuntimeError):
    """A required Provider media input could not be persisted to TOS."""


def require_tos_url(url: str | None, *, label: str) -> str:
    """Require a current signed URL from HonCut's configured TOS bucket."""
    normalized = str(url or "").strip()
    if not normalized:
        raise TOSMediaUploadError(f"TOS upload failed for required {label}")
    config = _get_tos_config()
    endpoint = str(config["endpoint"] or "").strip().lower().rstrip("/")
    if "://" in endpoint:
        endpoint = urlparse(endpoint).netloc
    expected_host = f"{config['bucket']}.{endpoint}".lower()
    parsed = urlparse(normalized)
    if (
        parsed.scheme != "https"
        or parsed.hostname != expected_host
        or parsed.netloc.lower() != expected_host
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.lstrip("/")
        or parsed.fragment
    ):
        raise TOSMediaUploadError(f"required {label} did not use the configured TOS origin")
    query = parse_qs(parsed.query, keep_blank_values=True)
    required_fields = {
        "X-Tos-Algorithm",
        "X-Tos-Credential",
        "X-Tos-Date",
        "X-Tos-Expires",
        "X-Tos-SignedHeaders",
        "X-Tos-Signature",
    }
    if set(query) != required_fields:
        raise TOSMediaUploadError(f"required {label} was not a signed TOS URL")
    if any(len(query[field]) != 1 for field in required_fields):
        raise TOSMediaUploadError(f"required {label} had ambiguous TOS signature fields")
    if query["X-Tos-Algorithm"] != ["TOS4-HMAC-SHA256"]:
        raise TOSMediaUploadError(f"required {label} used an invalid TOS signature")
    credential = query["X-Tos-Credential"][0]
    try:
        signed_at = datetime.strptime(query["X-Tos-Date"][0], "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
        expires = int(query["X-Tos-Expires"][0])
    except (ValueError, TypeError, IndexError) as exc:
        raise TOSMediaUploadError(f"required {label} had invalid TOS expiry metadata") from exc
    expected_scope = f"{signed_at:%Y%m%d}/{config['region']}/tos/request"
    if (
        not config["ak"]
        or credential != f"{config['ak']}/{expected_scope}"
        or query["X-Tos-SignedHeaders"] != ["host"]
    ):
        raise TOSMediaUploadError(f"required {label} did not use the configured TOS credential")
    if not 1 <= expires <= 604800:
        raise TOSMediaUploadError(f"required {label} had invalid TOS expiry")
    if datetime.now(timezone.utc).timestamp() >= signed_at.timestamp() + expires:
        raise TOSMediaUploadError(f"required {label} used an expired TOS URL")
    signature = query["X-Tos-Signature"][0]
    if len(signature) != 64 or any(char not in "0123456789abcdef" for char in signature):
        raise TOSMediaUploadError(f"required {label} used an invalid TOS signature")
    canonical_query = "&".join(
        f"{_url_encode(key)}={_url_encode(values[0])}"
        for key, values in sorted(query.items())
        if key != "X-Tos-Signature"
    )
    canonical_request = (
        f"GET\n{parsed.path}\n{canonical_query}\nhost:{expected_host}\n\nhost\nUNSIGNED-PAYLOAD"
    )
    request_hash = hashlib.sha256(canonical_request.encode()).hexdigest()
    string_to_sign = f"TOS4-HMAC-SHA256\n{query['X-Tos-Date'][0]}\n{expected_scope}\n{request_hash}"
    expected_signature = hmac.new(
        _signing_key(config["sk"], f"{signed_at:%Y%m%d}", config["region"]),
        string_to_sign.encode(),
        hashlib.sha256,
    ).hexdigest()
    if not config["sk"] or not hmac.compare_digest(signature, expected_signature):
        raise TOSMediaUploadError(f"required {label} used an invalid TOS signature")
    return normalized


# ─── TOS4-HMAC-SHA256 signing ────────────────────────────────────────────────

def _signing_key(sk: str, date: str, region: str) -> bytes:
    """TOS4-HMAC-SHA256 signing key chain: SK → date → region → 'tos' → 'request'."""
    k_date = hmac.new(sk.encode(), date.encode(), hashlib.sha256).digest()
    k_region = hmac.new(k_date, region.encode(), hashlib.sha256).digest()
    k_service = hmac.new(k_region, b"tos", hashlib.sha256).digest()
    k_signing = hmac.new(k_service, b"request", hashlib.sha256).digest()
    return k_signing


def _sign(method: str, object_key: str, headers: dict, payload_hash: str,
          timestamp: str, config: dict) -> str:
    """Generate TOS V4 Authorization header."""
    date = timestamp[:8]
    scope = f"{date}/{config['region']}/tos/request"

    host = f"{config['bucket']}.{config['endpoint']}"

    # Build canonical headers (sorted, lowercase)
    canonical_header_parts = [f"host:{host}"]
    signed_header_names = ["host"]

    for key in sorted(headers.keys()):
        if key.lower().startswith("x-tos-"):
            canonical_header_parts.append(f"{key.lower()}:{headers[key]}")
            signed_header_names.append(key.lower())

    canonical_headers = "\n".join(canonical_header_parts) + "\n"
    signed_headers = ";".join(sorted(signed_header_names))

    # Canonical request
    canonical_request = (
        f"{method}\n"
        f"/{object_key}\n"
        f"\n"                          # query string (empty for PUT)
        f"{canonical_headers}\n"
        f"{signed_headers}\n"
        f"{payload_hash}"
    )

    # String to sign
    cr_hash = hashlib.sha256(canonical_request.encode()).hexdigest()
    string_to_sign = f"TOS4-HMAC-SHA256\n{timestamp}\n{scope}\n{cr_hash}"

    # Signature
    key = _signing_key(config["sk"], date, config["region"])
    signature = hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()

    credential = f"{config['ak']}/{scope}"
    return f"TOS4-HMAC-SHA256 Credential={credential}, SignedHeaders={signed_headers}, Signature={signature}"


# ─── Pre-signed URL generation ────────────────────────────────────────────────

def _url_encode(s: str) -> str:
    """RFC 3986 percent-encoding (requests.utils.quote with safe='')."""
    return requests.utils.quote(str(s), safe="")


def get_signed_url(object_key: str, expires: int = 7200) -> str:
    """Generate pre-signed GET URL for a TOS object.

    Args:
        object_key: The object key in TOS
        expires: URL validity in seconds (default 7200 = 2 hours)

    Returns:
        Pre-signed URL string
    """
    config = _get_tos_config()
    host = f"{config['bucket']}.{config['endpoint']}"
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    date = timestamp[:8]
    scope = f"{date}/{config['region']}/tos/request"
    credential = f"{config['ak']}/{scope}"

    # Query params for pre-signed URL (sorted alphabetically)
    params = {
        "X-Tos-Algorithm": "TOS4-HMAC-SHA256",
        "X-Tos-Credential": credential,
        "X-Tos-Date": timestamp,
        "X-Tos-Expires": str(expires),
        "X-Tos-SignedHeaders": "host",
    }

    # Canonical query string (sorted by key)
    canonical_qs = "&".join(
        f"{_url_encode(k)}={_url_encode(v)}" for k, v in sorted(params.items())
    )

    # Canonical request for GET
    canonical_request = (
        f"GET\n"
        f"/{object_key}\n"
        f"{canonical_qs}\n"
        f"host:{host}\n"
        f"\n"
        f"host\n"
        f"UNSIGNED-PAYLOAD"
    )

    # String to sign
    cr_hash = hashlib.sha256(canonical_request.encode()).hexdigest()
    string_to_sign = f"TOS4-HMAC-SHA256\n{timestamp}\n{scope}\n{cr_hash}"

    # Signature
    key = _signing_key(config["sk"], date, config["region"])
    signature = hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()

    return f"https://{host}/{object_key}?{canonical_qs}&X-Tos-Signature={signature}"


# ─── Image compression (P0-E, ref: HonCut spec zipImage) ────────────────────────

def compress_image_base64(
    b64_data: str,
    max_bytes: int = SEEDANCE_MAX_IMAGE_BYTES,
) -> str:
    """Compress only when a provider limit requires it, preserving detail first."""
    import io
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return b64_data  # Pillow 未安装，跳过压缩
    
    raw = base64.b64decode(b64_data)
    if len(raw) <= max_bytes:
        return b64_data  # 已经够小
    with Image.open(io.BytesIO(raw)) as source:
        img = ImageOps.exif_transpose(source).convert("RGB")

    # Preserve the original pixel dimensions while reducing JPEG quality.
    for quality in [92, 88, 84, 78, 72, 64, 56, 48, 40]:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality, optimize=True)
        if buf.tell() <= max_bytes:
            return base64.b64encode(buf.getvalue()).decode()
    # Reduce dimensions gradually only after quality-only encodes are too large.
    resized = img
    while min(resized.size) > 300:
        scale = max(300 / min(resized.size), 0.85)
        new_size = (
            max(300, int(resized.width * scale)),
            max(300, int(resized.height * scale)),
        )
        if new_size == resized.size:
            break
        resized = resized.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=78, optimize=True)
        if buf.tell() <= max_bytes:
            return base64.b64encode(buf.getvalue()).decode()

    # Let the caller's media preflight fail closed if even the legal minimum
    # cannot satisfy the provider limit.
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=40, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ─── Upload ───────────────────────────────────────────────────────────────────

def compress_image_bytes(
    image_data: bytes,
    max_bytes: int = SEEDANCE_MAX_IMAGE_BYTES,
) -> bytes:
    """压缩图片字节到目标大小以下。内部复用 compress_image_base64。"""
    if len(image_data) <= max_bytes:
        return image_data
    b64 = base64.b64encode(image_data).decode()
    compressed_b64 = compress_image_base64(b64, max_bytes=max_bytes)
    return base64.b64decode(compressed_b64)


def _image_metadata(image_data: bytes) -> tuple[str, int, int]:
    import io

    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_data)) as image:
            image.verify()
        with Image.open(io.BytesIO(image_data)) as image:
            return str(image.format or "").upper(), int(image.width), int(image.height)
    except (ImportError, OSError, ValueError) as exc:
        raise ValueError("media preflight could not decode image bytes") from exc


def _probe_av(path: Path) -> dict:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,avg_frame_rate:format=duration,format_name",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"media preflight ffprobe failed for {path.name}") from exc
    if completed.returncode != 0:
        raise ValueError(f"media preflight ffprobe rejected {path.name}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"media preflight ffprobe returned invalid JSON for {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"media preflight ffprobe returned invalid data for {path.name}")
    return payload


def _fraction(value: str | None) -> float:
    numerator, _, denominator = str(value or "0").partition("/")
    try:
        return float(numerator) / float(denominator or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _validate_seedance_image(image_data: bytes) -> tuple[str, int, int]:
    if len(image_data) >= SEEDANCE_MAX_IMAGE_BYTES:
        raise ValueError("Seedance image must be smaller than 30 MB")
    image_format, width, height = _image_metadata(image_data)
    if image_format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"}:
        raise ValueError(f"Seedance image format is unsupported: {image_format or 'unknown'}")
    ratio = width / height if height else 0
    if not (300 <= width <= 6000 and 300 <= height <= 6000):
        raise ValueError("Seedance image dimensions must be within 300..6000 pixels")
    if not 0.4 <= ratio <= 2.5:
        raise ValueError("Seedance image aspect ratio must be within 0.4..2.5")
    return image_format, width, height


def _validate_seedance_video(path: Path) -> None:
    if path.stat().st_size > SEEDANCE_MAX_VIDEO_BYTES:
        raise ValueError("Seedance video exceeds the 200 MB input limit")
    if path.suffix.lower() not in {".mp4", ".mov"}:
        raise ValueError("Seedance video format must be mp4 or mov")
    payload = _probe_av(path)
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not isinstance(video, dict):
        raise ValueError("media preflight ffprobe found no video stream")
    codec = str(video.get("codec_name") or "").lower()
    if codec not in {"h264", "hevc"}:
        raise ValueError("Seedance video codec must be H.264 or H.265")
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    area = width * height
    if not (300 <= width <= 6000 and 300 <= height <= 6000):
        raise ValueError("Seedance video dimensions must be within 300..6000 pixels")
    if not 407696 <= area <= 8295044:
        raise ValueError("Seedance video pixel area is outside the supported range")
    fps = _fraction(video.get("avg_frame_rate"))
    if not 24 <= fps <= 60:
        raise ValueError("Seedance video frame rate must be within 24..60 fps")
    try:
        duration = float((payload.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("Seedance video duration is not measurable") from exc
    if not 2 <= duration <= 15:
        raise ValueError("Seedance reference video duration must be within 2..15 seconds")


def validate_multimodal_media_file(path: str | Path) -> str:
    """Validate one URL-backed Ark Responses input and return its media kind."""
    source = Path(path)
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"multimodal input not found or empty: {source}")
    mime_type = mimetypes.guess_type(source.name)[0] or ""
    size = source.stat().st_size
    if mime_type.startswith("image/"):
        if size >= MULTIMODAL_MAX_IMAGE_BYTES:
            raise ValueError("multimodal image must be smaller than 10 MB")
        image_format, width, height = _image_metadata(source.read_bytes())
        if image_format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF", "ICO"}:
            raise ValueError(f"multimodal image format is unsupported: {image_format}")
        if width <= 14 or height <= 14 or not 196 <= width * height <= 36_000_000:
            raise ValueError("multimodal image dimensions are outside the supported range")
        if not 1 / 150 <= width / height <= 150:
            raise ValueError("multimodal image aspect ratio is outside the supported range")
        return "image"
    if mime_type.startswith("video/"):
        if size >= MULTIMODAL_MAX_VIDEO_BYTES:
            raise ValueError("multimodal video must be smaller than 50 MB")
        if source.suffix.lower() not in {".mp4", ".avi", ".mov"}:
            raise ValueError("multimodal video format must be mp4, avi, or mov")
        payload = _probe_av(source)
        if not any(stream.get("codec_type") == "video" for stream in payload.get("streams") or []):
            raise ValueError("media preflight ffprobe found no video stream")
        return "video"
    if mime_type.startswith("audio/"):
        if size > MULTIMODAL_MAX_AUDIO_BYTES:
            raise ValueError("multimodal audio exceeds the 25 MB input limit")
        if source.suffix.lower() not in {".mp3", ".wav", ".aac", ".m4a"}:
            raise ValueError("multimodal audio format must be mp3, wav, aac, or m4a")
        payload = _probe_av(source)
        if not any(stream.get("codec_type") == "audio" for stream in payload.get("streams") or []):
            raise ValueError("media preflight ffprobe found no audio stream")
        duration = float((payload.get("format") or {}).get("duration") or 0)
        if duration <= 0 or duration > 7200:
            raise ValueError("multimodal audio duration must be within 120 minutes")
        return "audio"
    if source.suffix.lower() == ".pdf":
        if size >= MULTIMODAL_MAX_DOCUMENT_BYTES:
            raise ValueError("multimodal PDF must be smaller than 50 MB")
        with source.open("rb") as document:
            signature = document.read(5)
        if signature != b"%PDF-":
            raise ValueError("multimodal document must be a valid PDF")
        return "document"
    raise ValueError(f"unsupported multimodal input format: {source.name}")


def upload_image(image_data: bytes, content_type: str = "image/png") -> Optional[str]:
    """Upload image to TOS and return pre-signed URL.

    Args:
        image_data: Raw image bytes
        content_type: MIME type

    Returns:
        Pre-signed URL (valid 7200s) or None on failure
    """
    config = _get_tos_config()
    if not all([config["ak"], config["sk"], config["bucket"]]):
        print("  [tos] TOS config incomplete (need TOS_ACCESS_KEY, TOS_SECRET_KEY, TOS_BUCKET), skipping upload")
        return None

    # Seedance accepts images up to 30 MB. Preserve source bytes and detail
    # unless that provider limit actually requires recompression.
    image_data = compress_image_bytes(image_data)
    image_format, _width, _height = _validate_seedance_image(image_data)
    content_type, suffix = IMAGE_TRANSPORT_METADATA[image_format]

    # Object key with content hash for dedup
    content_hash = hashlib.sha256(image_data).hexdigest()
    object_key = f"volcengine/image/{content_hash}{suffix}"

    host = f"{config['bucket']}.{config['endpoint']}"
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    payload_hash = hashlib.sha256(image_data).hexdigest()

    # PUT upload with V4 signature
    headers = {
        "Content-Type": content_type,
        "Host": host,
        "x-tos-content-sha256": payload_hash,
        "x-tos-date": timestamp,
    }

    authorization = _sign("PUT", object_key, headers, payload_hash, timestamp, config)
    headers["Authorization"] = authorization

    try:
        resp = requests.put(
            f"https://{host}/{object_key}",
            data=image_data,
            headers=headers,
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            print(f"  [tos] Upload failed: HTTP {resp.status_code} — {resp.text[:200]}")
            return None
        print(f"  [tos] Uploaded: {object_key} ({len(image_data)} bytes)")
    except Exception as e:
        print(f"  [tos] Upload error: {e}")
        return None

    # Generate pre-signed GET URL (7200s = 2 hours)
    return get_signed_url(object_key, expires=7200)


def upload_file(
    image_data: bytes,
    object_key: str,
    content_type: str = "image/png",
) -> Optional[str]:
    """Upload bytes to an explicit TOS object key, preserving its directories."""
    config = _get_tos_config()
    if not all([config["ak"], config["sk"], config["bucket"]]):
        print("  [tos] TOS config incomplete (need TOS_ACCESS_KEY, TOS_SECRET_KEY, TOS_BUCKET), skipping upload")
        return None

    if content_type.startswith("image/"):
        image_data = compress_image_bytes(image_data)
        if image_data.startswith(b"\xff\xd8\xff"):
            content_type = "image/jpeg"
            object_key = str(Path(object_key).with_suffix(".jpg"))
        elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
            content_type = "image/png"
            object_key = str(Path(object_key).with_suffix(".png"))

    host = f"{config['bucket']}.{config['endpoint']}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload_hash = hashlib.sha256(image_data).hexdigest()
    encoded_key = quote(object_key, safe="/")
    if Path(object_key).stem == payload_hash:
        empty_payload_hash = hashlib.sha256(b"").hexdigest()
        head_headers = {
            "Host": host,
            "x-tos-content-sha256": empty_payload_hash,
            "x-tos-date": timestamp,
        }
        head_headers["Authorization"] = _sign(
            "HEAD",
            object_key,
            head_headers,
            empty_payload_hash,
            timestamp,
            config,
        )
        try:
            existing = requests.head(
                f"https://{host}/{encoded_key}",
                headers=head_headers,
                timeout=10,
            )
            existing_headers = {
                str(key).casefold(): str(value)
                for key, value in existing.headers.items()
            }
            declared_hash = existing_headers.get(TOS_CONTENT_SHA256_METADATA)
            declared_length = existing_headers.get("content-length")
            if (
                existing.status_code == 200
                and declared_length is not None
                and int(declared_length) == len(image_data)
                and (not declared_hash or declared_hash == payload_hash)
            ):
                print(f"  [tos] Reused: {object_key} ({len(image_data)} bytes)")
                return get_signed_url(object_key, expires=7200)
        except (OSError, ValueError, requests.RequestException):
            # A presence check is advisory. The authoritative PUT below still
            # decides whether this required input has been persisted.
            pass
    headers = {
        "Content-Type": content_type,
        "Host": host,
        "x-tos-content-sha256": payload_hash,
        "x-tos-date": timestamp,
        TOS_CONTENT_SHA256_METADATA: payload_hash,
    }
    headers["Authorization"] = _sign(
        "PUT", object_key, headers, payload_hash, timestamp, config
    )

    try:
        resp = requests.put(
            f"https://{host}/{encoded_key}",
            data=image_data,
            headers=headers,
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            print(f"  [tos] Upload failed: HTTP {resp.status_code} — {resp.text[:200]}")
            return None
        print(f"  [tos] Uploaded: {object_key} ({len(image_data)} bytes)")
    except Exception as exc:
        print(f"  [tos] Upload error for {object_key}: {exc}")
        return None
    return get_signed_url(object_key, expires=7200)


def upload_media_file(
    path: str | Path,
    *,
    prefix: str = "volcengine/media",
    contract: str = "seedance",
) -> str | None:
    """Preflight and upload one local provider input."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"media file not found: {source}")
    content_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    if contract == "seedance":
        if content_type.startswith("image/"):
            image_data = compress_image_bytes(source.read_bytes())
            image_format, _width, _height = _validate_seedance_image(image_data)
            content_type, suffix = IMAGE_TRANSPORT_METADATA[image_format]
            content_hash = hashlib.sha256(image_data).hexdigest()
            return upload_file(
                image_data,
                f"{prefix.rstrip('/')}/{content_hash}{suffix}",
                content_type,
            )
        if content_type.startswith("video/"):
            _validate_seedance_video(source)
        else:
            raise ValueError(f"unsupported Seedance media input: {source.name}")
    elif contract == "multimodal":
        validate_multimodal_media_file(source)
    else:
        raise ValueError(f"unknown media upload contract: {contract}")
    media_data = source.read_bytes()
    content_hash = hashlib.sha256(media_data).hexdigest()
    suffix = source.suffix.lower() or ".bin"
    return upload_file(
        media_data,
        f"{prefix.rstrip('/')}/{content_hash}{suffix}",
        content_type,
    )


def upload_media_file_required(
    path: str | Path,
    *,
    prefix: str = "volcengine/media",
    label: str = "media",
) -> str:
    """Upload a local image/video and require a TOS URL result."""
    return require_tos_url(
        upload_media_file(path, prefix=prefix),
        label=label,
    )


def upload_multimodal_media_file_required(
    path: str | Path,
    *,
    prefix: str = "volcengine/multimodal",
    label: str = "multimodal input",
) -> str:
    """Preflight an Ark understanding input, upload it, and require provenance."""
    return require_tos_url(
        upload_media_file(path, prefix=prefix, contract="multimodal"),
        label=label,
    )


def base64_video_to_signed_url(
    base64_data: str,
    *,
    suffix: str = ".mp4",
    content_type: str = "video/mp4",
) -> str | None:
    """Upload base64 video bytes without passing through image compression."""
    if "," in base64_data:
        header, base64_data = base64_data.split(",", 1)
        if header.startswith("data:") and ";" in header:
            declared_type = header[5:].split(";", 1)[0]
            if declared_type.startswith("video/"):
                content_type = declared_type
                guessed_suffix = mimetypes.guess_extension(declared_type)
                if guessed_suffix:
                    suffix = guessed_suffix
    try:
        video_data = base64.b64decode(base64_data, validate=True)
    except Exception as exc:
        print(f"  [tos] Video base64 decode error: {exc}")
        return None
    if not video_data:
        print("  [tos] Video base64 decode error: empty payload")
        return None
    normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(suffix=normalized_suffix) as temporary:
        temporary.write(video_data)
        temporary.flush()
        _validate_seedance_video(Path(temporary.name))
    content_hash = hashlib.sha256(video_data).hexdigest()
    return upload_file(
        video_data,
        f"volcengine/video/{content_hash}{normalized_suffix}",
        content_type,
    )


# ─── Main entry point ────────────────────────────────────────────────────────

def base64_to_signed_url(base64_data: str) -> Optional[str]:
    """Convert base64 image data to TOS signed URL.

    This is the main entry point for the Seedance pipeline.
    Strips data URL prefix if present, decodes, uploads, returns signed URL.

    Args:
        base64_data: Base64-encoded image (with or without data: prefix)

    Returns:
        Pre-signed URL string, or None if upload failed
    """
    # Strip data URL prefix if present (e.g., "data:image/png;base64,...")
    if "," in base64_data:
        base64_data = base64_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(base64_data, validate=True)
    except Exception as e:
        print(f"  [tos] Base64 decode error: {e}")
        return None

    return upload_image(image_bytes)


def base64_image_to_signed_url_required(
    base64_data: str,
    *,
    label: str = "image",
) -> str:
    """Upload a Base64 image and require a TOS URL result."""
    return require_tos_url(base64_to_signed_url(base64_data), label=label)


def base64_video_to_signed_url_required(
    base64_data: str,
    *,
    label: str = "video",
) -> str:
    """Upload a Base64 video and require a TOS URL result."""
    return require_tos_url(base64_video_to_signed_url(base64_data), label=label)

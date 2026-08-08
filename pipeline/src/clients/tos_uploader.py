"""Volcano TOS (Object Storage) uploader for Seedance reference images.

Ported for HonCut's volcengineSd2.ts — TOS4-HMAC-SHA256 signing.
Uploads images to TOS and returns pre-signed URLs for Seedance API.
"""

import os
import hashlib
import hmac
import base64
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


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

def compress_image_base64(b64_data: str, max_bytes: int = 300 * 1024) -> str:
    """压缩 base64 图片到目标大小以下（参考 HonCut 规范 zipImage）。
    
    策略：先降 quality，再降分辨率，循环直到 < max_bytes。
    """
    import io
    try:
        from PIL import Image
    except ImportError:
        return b64_data  # Pillow 未安装，跳过压缩
    
    raw = base64.b64decode(b64_data)
    if len(raw) <= max_bytes:
        return b64_data  # 已经够小
    
    img = Image.open(io.BytesIO(raw))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    # 策略1: 降 quality
    for quality in [85, 70, 55, 40]:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality, optimize=True)
        if buf.tell() <= max_bytes:
            return base64.b64encode(buf.getvalue()).decode()
    
    # 策略2: 降分辨率 + 降 quality
    for scale in [0.75, 0.5, 0.35]:
        new_size = (int(img.width * scale), int(img.height * scale))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format='JPEG', quality=60, optimize=True)
        if buf.tell() <= max_bytes:
            return base64.b64encode(buf.getvalue()).decode()
    
    # 兜底：返回最小版本
    buf = io.BytesIO()
    img.resize((int(img.width * 0.25), int(img.height * 0.25)), Image.Resampling.LANCZOS).save(
        buf, format='JPEG', quality=40, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ─── Upload ───────────────────────────────────────────────────────────────────

def compress_image_bytes(image_data: bytes, max_bytes: int = 300 * 1024) -> bytes:
    """压缩图片字节到目标大小以下。内部复用 compress_image_base64。"""
    if len(image_data) <= max_bytes:
        return image_data
    b64 = base64.b64encode(image_data).decode()
    compressed_b64 = compress_image_base64(b64, max_bytes=max_bytes)
    return base64.b64decode(compressed_b64)


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

    # --- P0-E: 上传前压缩（参考 HonCut 规范 zipImage）---
    image_data = compress_image_bytes(image_data)

    # Compression may transcode a large PNG to JPEG.  Keep the object suffix and
    # HTTP Content-Type consistent with the bytes so downstream decoders do not
    # have to guess (large full_body.png files routinely take this path).
    if image_data.startswith(b"\xff\xd8\xff"):
        content_type = "image/jpeg"
    elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        content_type = "image/png"

    # Object key with content hash for dedup
    content_hash = hashlib.sha256(image_data).hexdigest()
    ext = "png" if content_type == "image/png" else "jpg"
    object_key = f"volcengine/image/{content_hash}.{ext}"

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

    # --- P0-E: 上传前压缩（参考 HonCut 规范 zipImage）---
    base64_data = compress_image_base64(base64_data)

    try:
        image_bytes = base64.b64decode(base64_data)
    except Exception as e:
        print(f"  [tos] Base64 decode error: {e}")
        return None

    return upload_image(image_bytes)

"""Shared streaming client for Ark's OpenAI-compatible chat endpoint."""

from __future__ import annotations

import os
import random
import sys
import threading
import time
from typing import Callable, Optional

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, DefaultHttpxClient, OpenAI

from utils.config import ARK_BASE_URL, DEFAULT_TEXT_MODEL

_default_heartbeat_callback: Optional[Callable[[], None]] = None


class LLMTimeoutError(TimeoutError):
    """Base class for classified LLM timeouts."""


class LLMConnectTimeout(LLMTimeoutError):
    pass


class LLMReadTimeout(LLMTimeoutError):
    pass


class LLMIdleTimeout(LLMTimeoutError):
    """The stream produced no chunks within the configured idle window."""


class LLMWallTimeout(LLMTimeoutError):
    pass


class LLMStreamError(ConnectionError):
    """A retryable transport failure interrupted an in-progress stream."""


class LLMEmptyResponse(ValueError):
    pass


class LLMRateLimitedError(LLMStreamError):
    """Server-side burst protection / rate limiting rejected the request.

    Subclasses ``LLMStreamError`` so call sites that already retry stream
    interruptions get this case for free when the internal backoff budget
    is exhausted.
    """


_RATE_LIMIT_MARKERS = (
    "system protection triggered",
    "request burst",
    "rate limit",
    "ratelimit",
    "too many requests",
    "quota",
    "throttl",
    "slow down traffic",
)


def is_rate_limited_error(exc: BaseException) -> bool:
    """Classify server-side rate-limit / burst-protection failures."""
    from openai import RateLimitError

    if isinstance(exc, RateLimitError):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status in (429, 503):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _RATE_LIMIT_MARKERS)


_LAUNCH_STAGGER_ENV = "HONCUT_LLM_LAUNCH_STAGGER_S"
_launch_lock = threading.Lock()
_next_launch_at = 0.0


def _default_launch_stagger_s() -> float:
    raw = os.environ.get(_LAUNCH_STAGGER_ENV, "").strip()
    if not raw:
        return 1.5
    try:
        value = float(raw)
    except ValueError:
        return 1.5
    return max(0.0, value)


def _wait_for_launch_slot(launch_stagger: float) -> None:
    """Serialize request launches to avoid self-inflicted burst spikes."""
    global _next_launch_at
    if launch_stagger <= 0:
        return
    while True:
        with _launch_lock:
            now = time.monotonic()
            if now >= _next_launch_at:
                _next_launch_at = now + launch_stagger
                return
            wait_s = _next_launch_at - now
        time.sleep(wait_s)


def configure_heartbeat_callback(callback: Optional[Callable[[], None]]) -> None:
    """Set the process-level callback used by Phase 1 LLM calls."""
    global _default_heartbeat_callback
    _default_heartbeat_callback = callback


def create_ark_client(connect_timeout: float = 10.0, read_timeout: float = 60.0) -> OpenAI:
    api_key = os.environ.get("ARK_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 ARK_AGENT_API_KEY 未设置（火山方舟 Agent Plan）")
    timeout = httpx.Timeout(
        connect=connect_timeout, read=read_timeout, write=30.0, pool=10.0
    )
    # Ark is intentionally direct-routed (see utils.config).  Explicitly
    # disable ambient HTTP/SOCKS proxies at the transport boundary: relying on
    # NO_PROXY alone still makes the OpenAI client initialize a SOCKS transport
    # and fail when optional socksio is absent.
    http_client = DefaultHttpxClient(timeout=timeout, trust_env=False)
    return OpenAI(
        api_key=api_key,
        base_url=ARK_BASE_URL,
        timeout=timeout,
        max_retries=0,
        http_client=http_client,
    )


def _attempt_llm_stream(
    messages: list[dict],
    *,
    model: str = DEFAULT_TEXT_MODEL,
    max_tokens: int = 16000,
    wall_timeout: float = 180.0,
    read_timeout: float = 60.0,
    connect_timeout: float = 10.0,
    idle_timeout: Optional[float] = None,
    heartbeat_callback: Optional[Callable[[], None]] = None,
    heartbeat_interval: float = 5.0,
    _client=None,
) -> str:
    """Stream a completion with idle and hard wall-clock safety limits.

    ``idle_timeout`` is refreshed for every received chunk, so a healthy long
    response is not mistaken for a stalled request. ``wall_timeout`` remains a
    separate absolute ceiling for genuinely unbounded requests.
    """
    if wall_timeout <= 0:
        raise ValueError("wall_timeout must be positive")
    idle_timeout = read_timeout if idle_timeout is None else idle_timeout
    if idle_timeout <= 0:
        raise ValueError("idle_timeout must be positive")
    if heartbeat_interval < 0:
        raise ValueError("heartbeat_interval must be non-negative")
    client = _client or create_ark_client(connect_timeout, read_timeout)
    heartbeat_callback = heartbeat_callback or _default_heartbeat_callback
    deadline = time.monotonic() + wall_timeout
    stream = None
    wall_expired = threading.Event()
    idle_expired = threading.Event()
    stop_idle_monitor = threading.Event()
    activity_lock = threading.Lock()
    last_activity_at = time.monotonic()

    def close_stream() -> None:
        current = stream
        if current is not None:
            try:
                current.close()
            except Exception:
                pass

    def expire_wall() -> None:
        wall_expired.set()
        close_stream()

    def monitor_idle() -> None:
        check_interval = min(max(idle_timeout / 4, 0.005), 1.0)
        while not stop_idle_monitor.wait(check_interval):
            with activity_lock:
                inactive_for = time.monotonic() - last_activity_at
            if inactive_for >= idle_timeout:
                idle_expired.set()
                close_stream()
                return

    wall_timer = threading.Timer(wall_timeout, expire_wall)
    wall_timer.daemon = True
    wall_timer.start()
    idle_monitor = threading.Thread(target=monitor_idle, daemon=True)
    idle_monitor.start()
    chunks: list[str] = []
    last_heartbeat_at: Optional[float] = None
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
        )
        for chunk in stream:
            now = time.monotonic()
            if wall_expired.is_set() or now > deadline:
                try:
                    stream.close()
                finally:
                    raise LLMWallTimeout(f"LLM wall timeout after {wall_timeout}s")
            if idle_expired.is_set():
                raise LLMIdleTimeout(f"LLM stream idle timeout after {idle_timeout}s")
            with activity_lock:
                last_activity_at = now
            if (
                heartbeat_callback is not None
                and (
                    last_heartbeat_at is None
                    or heartbeat_interval == 0
                    or now - last_heartbeat_at >= heartbeat_interval
                )
            ):
                heartbeat_callback()
                last_heartbeat_at = now
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        if wall_expired.is_set():
            raise LLMWallTimeout(f"LLM wall timeout after {wall_timeout}s")
        if idle_expired.is_set():
            raise LLMIdleTimeout(f"LLM stream idle timeout after {idle_timeout}s")
    except LLMTimeoutError:
        raise
    except (httpx.ConnectTimeout,) as exc:
        raise LLMConnectTimeout(str(exc)) from exc
    except (httpx.ReadTimeout, APITimeoutError) as exc:
        if wall_expired.is_set():
            raise LLMWallTimeout(f"LLM wall timeout after {wall_timeout}s") from exc
        if idle_expired.is_set():
            raise LLMIdleTimeout(f"LLM stream idle timeout after {idle_timeout}s") from exc
        raise LLMReadTimeout(str(exc)) from exc
    except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
        if wall_expired.is_set():
            raise LLMWallTimeout(f"LLM wall timeout after {wall_timeout}s") from exc
        if idle_expired.is_set():
            raise LLMIdleTimeout(f"LLM stream idle timeout after {idle_timeout}s") from exc
        raise LLMStreamError(str(exc)) from exc
    except APIConnectionError as exc:
        cause = exc.__cause__
        if isinstance(cause, httpx.ConnectTimeout):
            raise LLMConnectTimeout(str(exc)) from exc
        if isinstance(cause, httpx.ReadTimeout):
            raise LLMReadTimeout(str(exc)) from exc
        if wall_expired.is_set():
            raise LLMWallTimeout(f"LLM wall timeout after {wall_timeout}s") from exc
        if idle_expired.is_set():
            raise LLMIdleTimeout(f"LLM stream idle timeout after {idle_timeout}s") from exc
        raise LLMStreamError(str(exc)) from exc
    except APIStatusError as exc:
        if is_rate_limited_error(exc):
            raise LLMRateLimitedError(str(exc)) from exc
        raise
    except Exception as exc:
        if wall_expired.is_set():
            raise LLMWallTimeout(f"LLM wall timeout after {wall_timeout}s") from exc
        if idle_expired.is_set():
            raise LLMIdleTimeout(f"LLM stream idle timeout after {idle_timeout}s") from exc
        if is_rate_limited_error(exc):
            raise LLMRateLimitedError(str(exc)) from exc
        raise
    finally:
        wall_timer.cancel()
        stop_idle_monitor.set()
        idle_monitor.join(timeout=1.0)
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    content = "".join(chunks)
    if not content.strip():
        raise LLMEmptyResponse("LLM 返回空内容")
    return content


def call_llm_stream(
    messages: list[dict],
    *,
    model: str = DEFAULT_TEXT_MODEL,
    max_tokens: int = 16000,
    wall_timeout: float = 180.0,
    read_timeout: float = 60.0,
    connect_timeout: float = 10.0,
    idle_timeout: Optional[float] = None,
    heartbeat_callback: Optional[Callable[[], None]] = None,
    heartbeat_interval: float = 5.0,
    rate_limit_retries: int = 3,
    rate_limit_base_wait: float = 10.0,
    rate_limit_max_wait: float = 120.0,
    rate_limit_jitter: float = 1.0,
    launch_stagger: Optional[float] = None,
    _client=None,
) -> str:
    """Public streaming entry point shared by every pipeline LLM caller.

    Wraps ``_attempt_llm_stream`` with two burst protections:

    * **Launch stagger** — concurrent callers (ThreadPoolExecutor fans in
      Phase 1) acquire serialized launch slots so requests do not volley at
      the same instant and trip server-side burst protection.
    * **Rate-limit backoff** — when the server answers with 429/503 or a
      burst-protection message, wait with exponential backoff (base×2,
      capped, plus jitter) and retry before failing the whole phase.
    """
    if rate_limit_retries < 0:
        raise ValueError("rate_limit_retries must be non-negative")
    if rate_limit_base_wait <= 0:
        raise ValueError("rate_limit_base_wait must be positive")
    stagger = _default_launch_stagger_s() if launch_stagger is None else launch_stagger
    client = _client or create_ark_client(connect_timeout, read_timeout)
    _wait_for_launch_slot(stagger)

    attempt = 0
    while True:
        try:
            return _attempt_llm_stream(
                messages,
                model=model,
                max_tokens=max_tokens,
                wall_timeout=wall_timeout,
                read_timeout=read_timeout,
                connect_timeout=connect_timeout,
                idle_timeout=idle_timeout,
                heartbeat_callback=heartbeat_callback,
                heartbeat_interval=heartbeat_interval,
                _client=client,
            )
        except LLMRateLimitedError as exc:
            if attempt >= rate_limit_retries:
                raise
            wait = min(rate_limit_base_wait * (2 ** attempt), rate_limit_max_wait)
            if rate_limit_jitter > 0:
                wait += random.uniform(0, rate_limit_jitter)
            attempt += 1
            print(
                f"  LLM 限流/burst 保护，{wait:.1f}s 后指数退避重试 "
                f"({attempt}/{rate_limit_retries}): {exc}",
                file=sys.stderr,
            )
            time.sleep(wait)

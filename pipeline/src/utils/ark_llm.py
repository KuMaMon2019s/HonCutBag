"""Shared streaming client for Ark's OpenAI-compatible chat endpoint."""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

import httpx
from openai import APIConnectionError, APITimeoutError, DefaultHttpxClient, OpenAI

from utils.config import ARK_BASE_URL

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


def call_llm_stream(
    messages: list[dict],
    *,
    model: str = "doubao-seed-2.1-turbo",
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
    except Exception as exc:
        if wall_expired.is_set():
            raise LLMWallTimeout(f"LLM wall timeout after {wall_timeout}s") from exc
        if idle_expired.is_set():
            raise LLMIdleTimeout(f"LLM stream idle timeout after {idle_timeout}s") from exc
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

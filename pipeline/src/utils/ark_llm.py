"""Shared streaming client for Ark's OpenAI-compatible chat endpoint."""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI

from utils.config import ARK_BASE_URL

_default_heartbeat_callback: Optional[Callable[[], None]] = None


class LLMTimeoutError(TimeoutError):
    """Base class for classified LLM timeouts."""


class LLMConnectTimeout(LLMTimeoutError):
    pass


class LLMReadTimeout(LLMTimeoutError):
    pass


class LLMWallTimeout(LLMTimeoutError):
    pass


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
    return OpenAI(
        api_key=api_key,
        base_url=ARK_BASE_URL,
        timeout=timeout,
        max_retries=0,
    )


def call_llm_stream(
    messages: list[dict],
    *,
    model: str = "doubao-seed-2.1-turbo",
    max_tokens: int = 16000,
    wall_timeout: float = 180.0,
    read_timeout: float = 60.0,
    connect_timeout: float = 10.0,
    heartbeat_callback: Optional[Callable[[], None]] = None,
    _client=None,
) -> str:
    """Stream a completion with classified read/connect and hard wall timeouts."""
    if wall_timeout <= 0:
        raise ValueError("wall_timeout must be positive")
    client = _client or create_ark_client(connect_timeout, read_timeout)
    heartbeat_callback = heartbeat_callback or _default_heartbeat_callback
    deadline = time.monotonic() + wall_timeout
    stream = None
    wall_expired = threading.Event()

    def force_close() -> None:
        wall_expired.set()
        current = stream
        if current is not None:
            try:
                current.close()
            except Exception:
                pass

    timer = threading.Timer(wall_timeout, force_close)
    timer.daemon = True
    timer.start()
    chunks: list[str] = []
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
        )
        for chunk in stream:
            if wall_expired.is_set() or time.monotonic() > deadline:
                try:
                    stream.close()
                finally:
                    raise LLMWallTimeout(f"LLM wall timeout after {wall_timeout}s")
            if heartbeat_callback is not None:
                heartbeat_callback()
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        if wall_expired.is_set():
            raise LLMWallTimeout(f"LLM wall timeout after {wall_timeout}s")
    except LLMTimeoutError:
        raise
    except (httpx.ConnectTimeout,) as exc:
        raise LLMConnectTimeout(str(exc)) from exc
    except (httpx.ReadTimeout, APITimeoutError) as exc:
        if wall_expired.is_set():
            raise LLMWallTimeout(f"LLM wall timeout after {wall_timeout}s") from exc
        raise LLMReadTimeout(str(exc)) from exc
    except APIConnectionError as exc:
        cause = exc.__cause__
        if isinstance(cause, httpx.ConnectTimeout):
            raise LLMConnectTimeout(str(exc)) from exc
        if isinstance(cause, httpx.ReadTimeout):
            raise LLMReadTimeout(str(exc)) from exc
        if wall_expired.is_set():
            raise LLMWallTimeout(f"LLM wall timeout after {wall_timeout}s") from exc
        raise
    except Exception as exc:
        if wall_expired.is_set():
            raise LLMWallTimeout(f"LLM wall timeout after {wall_timeout}s") from exc
        raise
    finally:
        timer.cancel()
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    content = "".join(chunks)
    if not content.strip():
        raise LLMEmptyResponse("LLM 返回空内容")
    return content

"""Lifecycle management for the optional local SAM 3 continuity service."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO
from urllib.error import URLError
from urllib.request import Request, urlopen

_VALID_MODES = {"off", "external", "managed"}


def _service_is_healthy(base_url: str) -> bool:
    request = Request(
        f"{base_url.rstrip('/')}/health",
        headers={"Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=0.75) as response:
            status = response.status
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False
    return status == 200 and payload.get("status") in {"unloaded", "ready"}


def _spawn_local_service(script: Path, log_handle: IO[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [str(script)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _mode() -> str:
    configured = os.environ.get("HONCUT_SAM3_MODE", "").strip().lower()
    if not configured:
        return "external" if os.environ.get("HONCUT_SAM3_URL", "").strip() else "off"
    if configured not in _VALID_MODES:
        choices = ", ".join(sorted(_VALID_MODES))
        raise ValueError(f"HONCUT_SAM3_MODE must be one of: {choices}")
    return configured


def _local_url() -> str:
    host = os.environ.get("SAM3_HOST", "127.0.0.1").strip() or "127.0.0.1"
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    port = int(os.environ.get("SAM3_PORT", "8001"))
    return f"http://{host}:{port}"


@contextmanager
def phase8_sam3_endpoint(output_dir: str | Path) -> Iterator[str]:
    """Yield the SAM 3 endpoint selected for one Phase 8 adjudication pass.

    ``off`` leaves object tracking disabled. ``external`` preserves the original
    URL-based integration. ``managed`` reuses a healthy configured endpoint or
    starts HonCut's local sidecar and always releases processes it owns.
    """
    mode = _mode()
    configured_url = os.environ.get("HONCUT_SAM3_URL", "").strip().rstrip("/")
    if mode == "off":
        yield ""
        return
    if mode == "external":
        if not configured_url:
            raise ValueError("HONCUT_SAM3_MODE=external requires HONCUT_SAM3_URL")
        yield configured_url
        return

    if configured_url and _service_is_healthy(configured_url):
        yield configured_url
        return

    local_url = _local_url()
    if _service_is_healthy(local_url):
        yield local_url
        return

    repo_dir = Path(__file__).resolve().parents[3]
    start_script = repo_dir / "pipeline" / "scripts" / "start_sam3.sh"
    log_path = Path(output_dir) / "logs" / "SAM3_SIDECAR.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[str] | None = None
    with log_path.open("a", encoding="utf-8") as log_handle:
        try:
            process = _spawn_local_service(start_script, log_handle)
        except OSError as exc:
            print(
                f"  ⚠ [8.15] SAM3 本地服务启动失败，降级为人工复核: {exc}",
                flush=True,
            )
            yield local_url
            return

        timeout = max(1.0, float(os.environ.get("HONCUT_SAM3_STARTUP_TIMEOUT", "30")))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _service_is_healthy(local_url):
                break
            if process.poll() is not None:
                break
            time.sleep(0.25)
        else:
            print(
                f"  ⚠ [8.15] SAM3 本地服务未在 {timeout:g}s 内就绪，" "本轮降级为人工复核",
                flush=True,
            )

        try:
            yield local_url
        finally:
            _stop_process(process)


__all__ = ["phase8_sam3_endpoint"]

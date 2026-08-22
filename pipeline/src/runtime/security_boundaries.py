"""Shared path, process, redaction, and correlation boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_SENSITIVE_KEY = re.compile(
    r"(^|_)(api_?key|authorization|credential|password|private_?key|secret|token)($|_)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_KEY_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,;]+"
)


def resolve_within_workspace(
    workspace: str | Path,
    candidate: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    root = Path(workspace).resolve(strict=True)
    requested = Path(candidate)
    if not requested.is_absolute():
        requested = root / requested
    resolved = requested.resolve(strict=must_exist)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("path escapes the configured workspace") from error
    return resolved


def validate_subprocess_args(args: Iterable[str | os.PathLike[str]]) -> list[str]:
    if isinstance(args, (str, bytes)):
        raise TypeError("subprocess commands must be argument arrays, not shell text")
    normalized = []
    for argument in args:
        value = os.fspath(argument)
        if not isinstance(value, str):
            raise TypeError("subprocess arguments must be text or path-like values")
        if "\x00" in value:
            raise ValueError("subprocess arguments must not contain NUL bytes")
        normalized.append(value)
    if not normalized or not normalized[0].strip():
        raise ValueError("subprocess command must not be empty")
    return normalized


def redact_text(value: str) -> str:
    redacted = _BEARER.sub("Bearer [REDACTED]", value)
    redacted = _KEY_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", redacted)
    for environment_value in os.environ.values():
        if len(environment_value) >= 12 and environment_value in redacted:
            redacted = redacted.replace(environment_value, "[REDACTED]")
    return redacted


def redact_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            name = str(key)
            if _SENSITIVE_KEY.search(name.replace("-", "_")):
                redacted[name] = "[REDACTED]"
            elif "prompt" in name.casefold() and isinstance(item, str):
                redacted[name] = {
                    "sha256": hashlib.sha256(item.encode("utf-8")).hexdigest(),
                    "length": len(item),
                }
            else:
                redacted[name] = redact_for_log(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact_for_log(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def safe_error_message(error: BaseException) -> str:
    return redact_text(str(error))


@dataclass(frozen=True)
class CorrelationContext:
    project_id: str
    run_id: str
    node_id: str
    task_id: str

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.project_id, self.run_id, self.node_id, self.task_id)
        ):
            raise ValueError("correlation IDs must not be empty")


def emit_runtime_event(
    event: str,
    correlation: CorrelationContext,
    **fields: Any,
) -> None:
    record = {
        "event": event,
        "project_id": correlation.project_id,
        "run_id": correlation.run_id,
        "node_id": correlation.node_id,
        "task_id": correlation.task_id,
        **redact_for_log(fields),
    }
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


__all__ = [
    "CorrelationContext",
    "emit_runtime_event",
    "redact_for_log",
    "redact_text",
    "resolve_within_workspace",
    "safe_error_message",
    "validate_subprocess_args",
]

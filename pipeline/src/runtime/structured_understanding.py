"""Runtime-owned recovery for schema-bound model understanding calls."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import ValidationError


T = TypeVar("T")
STRUCTURED_UNDERSTANDING_EXECUTION_SCHEMA = (
    "honcut.structured-understanding-execution.v1"
)
DEFAULT_STRUCTURED_UNDERSTANDING_ATTEMPTS = 2


class StructuredUnderstandingExhausted(RuntimeError):
    """A bounded schema replay could not produce one complete typed value."""

    def __init__(self, receipt: dict[str, Any], last_error: BaseException) -> None:
        self.receipt = receipt
        self.last_error = last_error
        attempts = len(receipt.get("attempts") or [])
        safe_error = str((receipt.get("attempts") or [{}])[-1].get("error") or "")
        super().__init__(
            "structured understanding exhausted "
            f"{attempts} attempt(s): {type(last_error).__name__}: {safe_error}"
        )


def _structured_error_summary(error: json.JSONDecodeError | ValidationError) -> str:
    if isinstance(error, json.JSONDecodeError):
        return (
            f"{error.msg}: line {error.lineno} column {error.colno} "
            f"(char {error.pos})"
        )
    details = [
        {
            "type": str(item.get("type") or "validation_error"),
            "loc": [str(value) for value in item.get("loc") or []],
            "msg": str(item.get("msg") or "schema validation failed"),
        }
        for item in error.errors()
    ]
    return json.dumps(details, ensure_ascii=False, separators=(",", ":"))


def execute_structured_understanding(
    operation: Callable[[], T],
    *,
    max_attempts: int = DEFAULT_STRUCTURED_UNDERSTANDING_ATTEMPTS,
) -> tuple[T, dict[str, Any]]:
    """Replay only rejected JSON/schema outputs, never transport failures.

    The operation must already request a native Provider JSON Schema and
    return the validated business DTO.  This owner does not salvage partial
    JSON or broaden parsing.  One failed structured response may be replayed
    once with the identical operation; every attempt is represented in the
    returned or raised receipt.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise ValueError("structured understanding max_attempts must be an integer")
    if max_attempts < 1 or max_attempts > DEFAULT_STRUCTURED_UNDERSTANDING_ATTEMPTS:
        raise ValueError(
            "structured understanding max_attempts must be between 1 and "
            f"{DEFAULT_STRUCTURED_UNDERSTANDING_ATTEMPTS}"
        )

    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            value = operation()
        except (json.JSONDecodeError, ValidationError) as exc:
            attempts.append({
                "attempt": attempt,
                "status": "schema_rejected",
                "error_type": type(exc).__name__,
                "error": _structured_error_summary(exc),
            })
            if attempt == max_attempts:
                receipt = {
                    "schema": STRUCTURED_UNDERSTANDING_EXECUTION_SCHEMA,
                    "status": "exhausted",
                    "max_attempts": max_attempts,
                    "attempts": attempts,
                }
                raise StructuredUnderstandingExhausted(receipt, exc) from exc
            continue
        attempts.append({"attempt": attempt, "status": "succeeded"})
        return value, {
            "schema": STRUCTURED_UNDERSTANDING_EXECUTION_SCHEMA,
            "status": "succeeded",
            "max_attempts": max_attempts,
            "attempts": attempts,
        }
    raise AssertionError("structured understanding execution exited without a result")


__all__ = [
    "DEFAULT_STRUCTURED_UNDERSTANDING_ATTEMPTS",
    "STRUCTURED_UNDERSTANDING_EXECUTION_SCHEMA",
    "StructuredUnderstandingExhausted",
    "execute_structured_understanding",
]

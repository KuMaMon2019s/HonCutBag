"""Bounded retry behavior used by concrete paid and non-paid operations."""

import random
import time


def retry_with_policy(
    func,
    max_attempts=3,
    backoff_factor=2.0,
    *args,
    non_retryable_exceptions=(),
    **kwargs,
):
    """Execute ``func`` with the existing bounded retry semantics."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if isinstance(exc, non_retryable_exceptions):
                raise
            last_error = exc
            error_text = str(exc)
            if attempt < max_attempts:
                is_429 = (
                    "429" in error_text
                    or "Too Many Requests" in error_text
                    or "QuotaExceeded" in error_text
                    or getattr(
                        getattr(exc, "response", None), "status_code", None
                    )
                    == 429
                )
                if is_429:
                    base_wait = [120, 240, 480][attempt - 1]
                    wait_time = base_wait + random.uniform(0, 30)
                else:
                    wait_time = backoff_factor ** (attempt - 1)
                print(
                    f"    ⚠ Attempt {attempt}/{max_attempts} failed: {exc}. "
                    f"Retrying in {wait_time:.1f}s...",
                    flush=True,
                )
                time.sleep(wait_time)
            else:
                print(
                    f"    ✗ All {max_attempts} attempts failed. Last error: {exc}",
                    flush=True,
                )
    raise last_error


_retry_with_policy = retry_with_policy


__all__ = ["_retry_with_policy", "retry_with_policy"]

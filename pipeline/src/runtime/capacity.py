"""Provider/media capacity configuration and in-process slot accounting."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager


class CapacityUnavailableError(RuntimeError):
    """A provider/media lane has no runnable capacity."""


def _read_positive_capacity(variable: str, fallback: int) -> int:
    raw = os.environ.get(variable)
    if raw is None:
        return fallback
    try:
        capacity = int(raw)
    except ValueError as error:
        raise ValueError(f"{variable} must be a positive integer, got {raw!r}") from error
    if capacity < 1:
        raise ValueError(f"{variable} must be at least 1, got {capacity}")
    return capacity


class CapacityTable:
    """Read-only concurrency limits keyed by provider and media type."""

    def __init__(
        self,
        limits: Mapping[tuple[str, str], int],
        defaults: Mapping[str, int],
    ) -> None:
        self._limits = dict(limits)
        self._defaults = dict(defaults)
        for key, capacity in self._limits.items():
            if capacity < 0:
                raise ValueError(f"capacity for {key!r} must not be negative")
        for media_type, capacity in self._defaults.items():
            if capacity < 0:
                raise ValueError(
                    f"default capacity for {media_type!r} must not be negative"
                )

    @classmethod
    def for_seedance_video(cls, fallback: int) -> CapacityTable:
        if fallback < 1:
            raise ValueError(
                f"video concurrency fallback must be at least 1, got {fallback}"
            )
        capacity = _read_positive_capacity(
            "HONCUT_SEEDANCE_VIDEO_CONCURRENCY", fallback
        )
        return cls({("seedance", "video"): capacity}, {"video": fallback})

    def get(self, provider_id: str, media_type: str) -> int:
        known_providers = {provider for provider, _ in self._limits}
        if provider_id in known_providers:
            return self._limits.get((provider_id, media_type), 0)
        return self._defaults.get(media_type, 0)


class SlotTable:
    """Thread-safe in-process occupancy for provider/media capacity lanes."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._occupants: dict[tuple[str, str], set[str]] = {}

    def acquire(
        self,
        provider_id: str,
        media_type: str,
        task_id: str,
        *,
        capacity: int,
    ) -> None:
        if capacity < 1:
            raise CapacityUnavailableError(
                f"no capacity for provider={provider_id!r}, media={media_type!r}"
            )
        key = (provider_id, media_type)
        with self._condition:
            while True:
                bucket = self._occupants.setdefault(key, set())
                if task_id in bucket:
                    raise RuntimeError(
                        f"task {task_id!r} already occupies provider={provider_id!r}, "
                        f"media={media_type!r}"
                    )
                if len(bucket) < capacity:
                    bucket.add(task_id)
                    return
                self._condition.wait()

    def release(self, provider_id: str, media_type: str, task_id: str) -> None:
        key = (provider_id, media_type)
        with self._condition:
            bucket = self._occupants.get(key)
            if bucket is None:
                return
            bucket.discard(task_id)
            if not bucket:
                del self._occupants[key]
            self._condition.notify_all()

    def occupied(self, provider_id: str, media_type: str) -> int:
        with self._condition:
            return len(self._occupants.get((provider_id, media_type), ()))

    @contextmanager
    def reserve(
        self,
        provider_id: str,
        media_type: str,
        task_id: str,
        *,
        capacity: int,
    ) -> Iterator[None]:
        self.acquire(provider_id, media_type, task_id, capacity=capacity)
        try:
            yield
        finally:
            self.release(provider_id, media_type, task_id)

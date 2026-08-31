"""Runtime-owned limits for healthy streamed LLM operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMStreamPolicy:
    """Bound one streamed LLM operation without confusing activity with stall."""

    wall_timeout_seconds: float
    idle_timeout_seconds: float
    max_tokens: int

    def __post_init__(self) -> None:
        if self.wall_timeout_seconds <= 0 or self.idle_timeout_seconds <= 0:
            raise ValueError("LLM stream timeouts must be greater than zero")
        if self.wall_timeout_seconds <= self.idle_timeout_seconds:
            raise ValueError("LLM wall timeout must exceed the idle timeout")
        if self.max_tokens <= 0:
            raise ValueError("LLM max_tokens must be greater than zero")

    @classmethod
    def long_structured_output(cls, *, max_tokens: int) -> "LLMStreamPolicy":
        """Policy for bounded JSON planning/extraction with an active stream.

        A 75-second idle limit detects a stalled stream. The separate 15-minute
        wall limit permits a healthy long response to finish and is consistent
        with the Phase 1 event/adaptation safety ceiling.
        """

        return cls(
            wall_timeout_seconds=900.0,
            idle_timeout_seconds=75.0,
            max_tokens=max_tokens,
        )

    @classmethod
    def adaptation_structured_output(
        cls,
        *,
        max_tokens: int,
    ) -> "LLMStreamPolicy":
        """Policy for Adaptation's bounded, reasoning-heavy JSON streams.

        Ark may pause content chunks while the model is still reasoning. A
        four-minute idle window tolerates that behavior while the existing
        fifteen-minute wall remains the absolute single-request ceiling.
        """

        return cls(
            wall_timeout_seconds=900.0,
            idle_timeout_seconds=240.0,
            max_tokens=max_tokens,
        )

    @property
    def transport_read_timeout_seconds(self) -> float:
        """Keep the SDK socket timeout behind Runtime's wall watchdog.

        The SDK transport must not race the idle/wall monitors or leak its own
        unclassified timeout first. Runtime closes the stream when either
        policy clock expires; this small grace only gives that close a chance
        to win the classification race.
        """

        return self.wall_timeout_seconds + 30.0


__all__ = ["LLMStreamPolicy"]

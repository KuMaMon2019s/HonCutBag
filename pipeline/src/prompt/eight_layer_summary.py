"""Length-safe assembly for the Layer 3-6 shot subject summary."""

from collections.abc import Sequence

SUBJECT_SUMMARY_MIN_CHARS = 40
SUBJECT_SUMMARY_MAX_CHARS = 100


def build_subject_summary(
    layers: Sequence[tuple[str, object]],
    *,
    min_chars: int = SUBJECT_SUMMARY_MIN_CHARS,
    max_chars: int = SUBJECT_SUMMARY_MAX_CHARS,
) -> str:
    """Join Layer 3-6 while applying the character budget to this text only.

    Layer labels are never removed, so a long subject or scene cannot crowd out
    action, camera, or lighting. Callers must append Layer 7/8 after this
    function returns; in particular, negative guardrails must never be passed
    through this limiter.
    """
    if min_chars < 0 or max_chars < min_chars:
        raise ValueError("invalid subject-summary character limits")
    if not layers:
        raise ValueError("subject summary requires at least one layer")

    labels = [str(label).strip() for label, _ in layers]
    values = [str(value).strip() for _, value in layers]
    if any(not label for label in labels):
        raise ValueError("subject-summary layer labels must not be empty")

    def render() -> str:
        return "；".join(
            f"{label}{value}" for label, value in zip(labels, values, strict=True)
        )

    # Preserve useful content from every layer. Longer values give up characters
    # first, with one character retained even for unusually small budgets.
    while len(render()) > max_chars:
        candidates = [index for index, value in enumerate(values) if len(value) > 1]
        if not candidates:
            raise ValueError("subject-summary limit is too small for its layer labels")
        longest = max(candidates, key=lambda index: len(values[index]))
        values[longest] = values[longest][:-1]

    summary = render()
    if len(summary) < min_chars:
        filler = "，主体动作连贯，镜头稳定，场景光影自然"
        summary = (summary + filler)[:max_chars]
    if len(summary) < min_chars:
        raise ValueError("subject-summary content cannot satisfy the minimum length")
    return summary

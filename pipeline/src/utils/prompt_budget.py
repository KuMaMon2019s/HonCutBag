"""Operational prompt budgets enforced immediately before provider submission."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SINGLETON_PROMPT_MARKERS = (
    "[storyboard-motion-notation]",
    "[honcut-video-generation-contract-v2]",
    "[逐拍肢体动作谱｜不可摘要]",
    "【非真人视觉硬约束】",
)

_SECTION_MARKER = re.compile(r"(?:(?<=\n)|(?<=。)|\A)(\[[^\]\n]{1,100}\])")


class PromptBudgetExceededError(ValueError):
    """The prompt is too large to submit safely to the selected provider/model."""


class DuplicatePromptContractError(ValueError):
    """A singleton prompt contract was accidentally rendered more than once."""


@dataclass(frozen=True)
class PromptBudget:
    provider: str
    model: str
    purpose: str
    soft_chars: int
    hard_chars: int


@dataclass(frozen=True)
class PromptBudgetReport:
    budget: PromptBudget
    total_chars: int
    section_chars: Mapping[str, int]
    marker_counts: Mapping[str, int]

    @property
    def over_soft_limit(self) -> bool:
        return self.total_chars >= self.budget.soft_chars


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _env_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_") or "UNKNOWN"


def resolve_prompt_budget(
    *,
    provider: str,
    model: str,
    purpose: str,
) -> PromptBudget:
    """Resolve conservative operational limits for one provider/model route."""
    normalized_provider = provider.strip().lower() or "generic"
    normalized_model = model.strip() or "unknown"
    normalized_purpose = purpose.strip().lower() or "generic"
    if normalized_provider == "seedance" or "seedance" in normalized_model.lower():
        prefix, defaults = "HONCUT_SEEDANCE_PROMPT", (12_000, 16_000)
    elif normalized_provider == "ark" and normalized_purpose == "multimodal_review":
        prefix, defaults = "HONCUT_ARK_MULTIMODAL_PROMPT", (30_000, 45_000)
    elif normalized_provider == "ark":
        prefix, defaults = "HONCUT_ARK_TEXT_PROMPT", (60_000, 90_000)
    else:
        prefix, defaults = "HONCUT_GENERIC_PROMPT", (16_000, 24_000)
    route_soft = _positive_env_int(f"{prefix}_SOFT_CHARS", defaults[0])
    route_hard = _positive_env_int(f"{prefix}_HARD_CHARS", defaults[1])
    model_prefix = "_".join((
        "HONCUT_PROMPT",
        _env_token(normalized_provider),
        _env_token(normalized_model),
        _env_token(normalized_purpose),
    ))
    soft = _positive_env_int(f"{model_prefix}_SOFT_CHARS", route_soft)
    hard = _positive_env_int(f"{model_prefix}_HARD_CHARS", route_hard)
    if soft >= hard:
        raise ValueError(
            f"prompt soft limit must be smaller than hard limit for {model_prefix}"
        )
    return PromptBudget(
        provider=normalized_provider,
        model=normalized_model,
        purpose=normalized_purpose,
        soft_chars=soft,
        hard_chars=hard,
    )


def prompt_section_lengths(prompt: str) -> dict[str, int]:
    """Measure prompt regions delimited by bracketed contract markers."""
    text = str(prompt or "")
    matches = list(_SECTION_MARKER.finditer(text))
    if not matches:
        return {"preamble": len(text)}
    lengths: dict[str, int] = {}
    if matches[0].start() > 0:
        lengths["preamble"] = matches[0].start()
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        marker = match.group(1)
        lengths[marker] = lengths.get(marker, 0) + end - match.start()
    return lengths


def enforce_prompt_budget(
    prompt: str,
    *,
    provider: str,
    model: str,
    purpose: str,
) -> PromptBudgetReport:
    """Log prompt sections, warn at the soft limit, and fail closed at hard limit."""
    text = str(prompt or "")
    budget = resolve_prompt_budget(provider=provider, model=model, purpose=purpose)
    marker_counts = {marker: text.count(marker) for marker in SINGLETON_PROMPT_MARKERS}
    duplicates = {
        marker: count
        for marker, count in marker_counts.items()
        if count > 1 and budget.purpose == "video_generation"
    }
    if duplicates:
        detail = ", ".join(f"{marker}={count}" for marker, count in duplicates.items())
        raise DuplicatePromptContractError(
            f"duplicate singleton prompt contract for {budget.provider}/{budget.model}: {detail}"
        )

    report = PromptBudgetReport(
        budget=budget,
        total_chars=len(text),
        section_chars=prompt_section_lengths(text),
        marker_counts=marker_counts,
    )
    context = {
        "provider": budget.provider,
        "model": budget.model,
        "purpose": budget.purpose,
        "total_chars": report.total_chars,
        "soft_chars": budget.soft_chars,
        "hard_chars": budget.hard_chars,
        "section_chars": dict(report.section_chars),
    }
    logger.info("prompt budget %s", context)
    if report.total_chars >= budget.hard_chars:
        raise PromptBudgetExceededError(
            "prompt hard limit exceeded before provider submission: "
            f"provider={budget.provider}, model={budget.model}, purpose={budget.purpose}, "
            f"chars={report.total_chars}, hard={budget.hard_chars}, "
            f"sections={dict(report.section_chars)}"
        )
    if report.over_soft_limit:
        logger.warning("prompt soft limit reached %s", context)
    return report

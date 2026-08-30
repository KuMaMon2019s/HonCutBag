"""Narrow compatibility boundary for the frozen pipeline-core facade.

New CLI and lifecycle calls pass ``character_visual_policy`` directly.  The
test compatibility facade still forwards its historical boolean, so this
module is the only runtime adapter allowed to read that legacy keyword.
"""

from __future__ import annotations

from typing import Any

from utils.canonical_visual_contracts import (
    SOURCE_DERIVED_POLICY,
    SYNTHETIC_STYLIZED_POLICY,
)


LEGACY_VISUAL_POLICY_KEY = "no_real_person"


def migrate_legacy_visual_policy_options(
    options: dict[str, Any],
    *,
    character_visual_policy: str,
) -> str:
    """Consume only the frozen facade keyword and reject ambiguous input."""
    unknown = set(options) - {LEGACY_VISUAL_POLICY_KEY}
    if unknown:
        raise TypeError(f"unexpected legacy pipeline options: {sorted(unknown)}")
    if LEGACY_VISUAL_POLICY_KEY not in options:
        return character_visual_policy
    value = options[LEGACY_VISUAL_POLICY_KEY]
    if value is None:
        return character_visual_policy
    legacy_policy = (
        SYNTHETIC_STYLIZED_POLICY if bool(value) else SOURCE_DERIVED_POLICY
    )
    if (
        character_visual_policy != SOURCE_DERIVED_POLICY
        and character_visual_policy != legacy_policy
    ):
        raise ValueError(
            "legacy visual-policy flag conflicts with character_visual_policy"
        )
    return legacy_policy

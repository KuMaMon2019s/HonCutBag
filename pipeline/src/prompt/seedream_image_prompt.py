"""Seedream image-prompt contracts shared by image-generating phases.

The provider understands multi-image requests by ordinal position.  This module
keeps that binding explicit and stable without moving phase-specific story,
character, or visual-style semantics into the transport client.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

REFERENCE_CONTRACT_TEMPLATE_ID = "honcut.seedream.reference-contract"
REFERENCE_CONTRACT_TEMPLATE_VERSION = "2"
IMAGE_REQUEST_CONTRACT_ID = "honcut.seedream.agent-plan-single-image"
IMAGE_REQUEST_CONTRACT_VERSION = "1"

_SINGLE_IMAGE_PARAMETERS = {
    "response_format": "url",
    "output_format": "png",
    "watermark": False,
    "sequential_image_generation": "disabled",
    "stream": False,
    "optimize_prompt_options": {"mode": "standard"},
}

_REFERENCE_ROLE_INSTRUCTIONS = {
    "character_identity_only": (
        "character identity only; preserve face, hair, body proportions, outfit and "
        "assigned props; ignore pose, framing, background, layout and text"
    ),
    "character_identity_board_only": (
        "one 2x2 identity board whose four cells are different views of the same single "
        "character, never four people or clones; preserve face, hair, body proportions, "
        "outfit, side/back silhouette and assigned props; ignore board layout, neutral "
        "background and cell boundaries"
    ),
    "character_face_identity_only": (
        "canonical face identity only; preserve facial geometry, skin, hairline and "
        "distinctive facial details; ignore framing, pose, background and text"
    ),
    "character_body_identity_only": (
        "canonical full-body identity only; preserve body proportions, outfit, shoes "
        "and assigned props; ignore pose, framing, background and text"
    ),
    "prior_storyboard_state": (
        "previous storyboard state; preserve camera axis, screen direction and relative "
        "spatial continuity; do not copy its pose or action progress; advance every subject "
        "to the requested new state"
    ),
    "director_single_panel_composition_only": (
        "director single panel; preserve only scene geometry, subject placement, "
        "screen direction and lighting; ignore sketch medium, grid, arrows and text"
    ),
    "prior_cinematic_state": (
        "previous finished cinematic frame; preserve identity, scene, camera axis and "
        "completed state, then advance to the requested start state"
    ),
    "bridge_source_final_state": (
        "source shot final storyboard state; preserve its completed pose, camera axis, "
        "screen direction, character identity and scene continuity"
    ),
    "bridge_target_opening_state": (
        "target shot opening storyboard state; use it only as the destination state "
        "after the requested midpoint; do not execute its new action early"
    ),
}

_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


def bind_reference_roles(prompt: str, roles: Sequence[str]) -> str:
    """Prepend exact Image-N responsibilities in provider input order.

    Seedream's official guidance asks multi-reference prompts to identify which
    image supplies each subject, style, or edit target.  Unknown roles fail
    closed so an upstream ordering change cannot silently swap identities or
    composition responsibilities.
    """
    normalized_prompt = str(prompt).strip()
    if not normalized_prompt:
        raise ValueError("Seedream prompt must not be empty")
    if not roles:
        return normalized_prompt

    bindings = []
    for index, raw_role in enumerate(roles, 1):
        role = str(raw_role).strip()
        try:
            instruction = _REFERENCE_ROLE_INSTRUCTIONS[role]
        except KeyError as exc:
            raise ValueError(f"Unknown Seedream reference role: {role!r}") from exc
        bindings.append(f"Image {index}: {instruction}.")

    return "\n".join(
        [
            (
                "[honcut-seedream-reference-contract-v"
                f"{REFERENCE_CONTRACT_TEMPLATE_VERSION}]"
            ),
            "Reference bindings are fixed in input order; never swap or merge their roles:",
            *bindings,
            "Generate the requested single image from the following contract:",
            normalized_prompt,
        ]
    )


def single_image_request_parameters(size: str) -> dict[str, object]:
    """Return a fresh copy of the documented non-streaming image parameters."""
    return {
        "size": str(size),
        **_SINGLE_IMAGE_PARAMETERS,
        "optimize_prompt_options": dict(
            _SINGLE_IMAGE_PARAMETERS["optimize_prompt_options"]
        ),
    }


def image_request_fingerprint(
    *,
    prompt: str,
    model: str,
    size: str,
    reference_image_sha256: Sequence[str] = (),
) -> str:
    """Hash every semantic Agent Plan image input without credentials.

    The fingerprint deliberately includes the request-contract version and
    ordered reference hashes.  A resolution, prompt wrapper, provider contract,
    model, or reference-order change therefore invalidates an older paid image
    instead of silently reusing it.
    """
    reference_hashes = [str(value) for value in reference_image_sha256]
    for content_hash in reference_hashes:
        if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            raise ValueError("Seedream reference fingerprint must be SHA-256")
    material = {
        "schema": "honcut.seedream-image-input.v1",
        "request_contract": {
            "id": IMAGE_REQUEST_CONTRACT_ID,
            "version": IMAGE_REQUEST_CONTRACT_VERSION,
        },
        "model": str(model),
        "prompt_sha256": hashlib.sha256(str(prompt).encode("utf-8")).hexdigest(),
        "parameters": single_image_request_parameters(size),
        "reference_image_sha256": reference_hashes,
    }
    return hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def prompt_guidance_metrics(prompt: str) -> dict[str, int | str | bool]:
    """Return privacy-safe prompt metrics for the official length guidance.

    The 300-Chinese-character / 600-English-word values are recommendations,
    not API limits.  HonCut therefore records an observable warning signal but
    never truncates a story, identity, or correction contract at transport time.
    """
    normalized_prompt = str(prompt)
    cjk_characters = sum(
        1
        for character in normalized_prompt
        if (
            "\u3400" <= character <= "\u4dbf"
            or "\u4e00" <= character <= "\u9fff"
            or "\uf900" <= character <= "\ufaff"
        )
    )
    english_words = len(_ENGLISH_WORD_RE.findall(normalized_prompt))
    return {
        "sha256": hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest(),
        "characters": len(normalized_prompt),
        "cjk_characters": cjk_characters,
        "english_words": english_words,
        "over_recommended_length": (cjk_characters > 300 or english_words > 600),
        "reference_contract_template_id": REFERENCE_CONTRACT_TEMPLATE_ID,
        "reference_contract_template_version": REFERENCE_CONTRACT_TEMPLATE_VERSION,
    }

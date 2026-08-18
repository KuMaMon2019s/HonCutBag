"""Semantic QA contract for Phase 3 character reference packs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

CHARACTER_REFERENCE_QA_SCHEMA = "honcut.character-reference-qa.v1"
SEEDANCE_REFERENCE_VIEWS = ("face_closeup", "full_body", "side", "back")


class CharacterReferenceReviewer(Protocol):
    def review(self, image_paths: list[Path], prompt: str) -> str: ...


class CharacterReferenceQAError(RuntimeError):
    """Raised when a generated character pack cannot satisfy its view contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_character_reference_qa_prompt(
    character_description: str,
    view_names: tuple[str, ...] = SEEDANCE_REFERENCE_VIEWS,
) -> str:
    """Ask a VLM to verify view semantics rather than image attractiveness."""
    requirements = {
        "face_closeup": (
            "strict straight-on head-and-shoulders close-up; crown through clavicles; "
            "face occupies 60-75% of the frame; not a full-body or action image"
        ),
        "full_body": (
            "strict front full-body reference; hair top and both shoe soles visible; "
            "upright neutral stance, arms relaxed down, feet parallel and hip-width"
        ),
        "side": (
            "strict 90-degree left side full-body profile; nose, torso, hips and toes all "
            "point left; only one eye is visible; head does not turn toward camera"
        ),
        "back": (
            "strict 180-degree rear full-body view; back of head, shoulders, spine, outfit "
            "rear and heels visible; no eyes, nose, mouth, chest or front of torso visible"
        ),
        "front": "strict straight-on identity portrait matching its requested framing",
        "three_quarter": "clear three-quarter identity portrait, neither front nor profile",
        "detail": "tight facial-detail close-up, not a full-body image",
    }
    ordered = "\n".join(
        f"- {name}: {requirements.get(name, 'match the filename view exactly')}"
        for name in view_names
    )
    return f"""You are the blocking Phase 3 character-reference inspector.
The input images are ordered and labelled by filename. Judge geometry and semantics, not beauty.

Static identity contract:
{character_description}

Per-view contracts:
{ordered}

All images must contain exactly one instance of the character on a plain neutral studio
background. No street, shop, crowd, scenery, performance, dance pose, action pose, prop not
declared in the identity, text, watermark or logo is allowed. Full-body views must use the same
neutral anatomical reference stance. Face, hair, outfit, static accessories, apparent age,
head scale and body proportions must remain the same across views.

Return one JSON object only:
{{
  "views": {{
    "<view_name>": {{
      "passed": true,
      "view_match": true,
      "framing_match": true,
      "neutral_pose": true,
      "plain_background": true,
      "single_character": true,
      "face_visible": true,
      "both_eyes_visible": false,
      "issues": []
    }}
  }},
  "cross_view": {{
    "passed": true,
    "identity_consistent": true,
    "outfit_consistent": true,
    "body_proportions_consistent": true,
    "issues": []
  }},
  "failed_views": [],
  "summary": "short factual summary"
}}

Set passed=false for any uncertain or violated requirement. For back, face_visible and
both_eyes_visible must both be false. For side, both_eyes_visible must be false. When a
cross-view mismatch is localized, list the suspect filenames in failed_views; otherwise list
all supplied filenames. Do not excuse a wrong angle because identity is consistent."""


def _json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if "{" not in text:
        raise CharacterReferenceQAError("character reference QA returned no JSON object")

    # VLMs sometimes wrap the requested object in markdown or append a prose
    # explanation (occasionally containing another JSON object).  ``json.loads``
    # over first-"{" through last-"}" turns that valid response into Extra data.
    # raw_decode consumes exactly one complete value and tells us where it ends;
    # scan candidate object starts until the QA-shaped object is found.
    decoder = json.JSONDecoder()
    first_object: dict[str, Any] | None = None
    last_error: json.JSONDecodeError | None = None
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _end = decoder.raw_decode(text, start)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(candidate, dict):
            continue
        first_object = first_object or candidate
        if isinstance(candidate.get("views"), dict) and isinstance(
            candidate.get("cross_view"), dict
        ):
            return candidate

    if first_object is not None:
        return first_object
    detail = f": {last_error}" if last_error is not None else ""
    raise CharacterReferenceQAError(
        f"character reference QA returned invalid JSON{detail}"
    )


def parse_character_reference_qa(
    raw: str,
    view_names: tuple[str, ...] = SEEDANCE_REFERENCE_VIEWS,
) -> dict[str, Any]:
    """Normalize a review and recompute the verdict from its evidence fields."""
    payload = _json_object(raw)
    raw_views = payload.get("views")
    cross = payload.get("cross_view")
    if not isinstance(raw_views, dict) or not isinstance(cross, dict):
        raise CharacterReferenceQAError(
            "character reference QA requires views and cross_view objects"
        )

    normalized_views: dict[str, dict[str, Any]] = {}
    failed: set[str] = set()
    boolean_fields = (
        "view_match",
        "framing_match",
        "neutral_pose",
        "plain_background",
        "single_character",
    )
    for name in view_names:
        evidence = raw_views.get(name)
        if not isinstance(evidence, dict):
            raise CharacterReferenceQAError(f"character reference QA omitted {name}")
        issues = evidence.get("issues")
        issues = issues if isinstance(issues, list) else [str(issues or "")]
        passed = evidence.get("passed") is True and all(
            evidence.get(field) is True for field in boolean_fields
        )
        if name == "back":
            passed = (
                passed
                and evidence.get("face_visible") is False
                and evidence.get("both_eyes_visible") is False
            )
        if name == "side":
            passed = passed and evidence.get("both_eyes_visible") is False
        if not passed:
            failed.add(name)
        normalized_views[name] = {
            "passed": passed,
            **{field: evidence.get(field) is True for field in boolean_fields},
            "face_visible": evidence.get("face_visible") is True,
            "both_eyes_visible": evidence.get("both_eyes_visible") is True,
            "issues": [str(item) for item in issues if str(item).strip()],
        }

    cross_fields = (
        "identity_consistent",
        "outfit_consistent",
        "body_proportions_consistent",
    )
    cross_passed = cross.get("passed") is True and all(
        cross.get(field) is True for field in cross_fields
    )
    declared_failed = payload.get("failed_views")
    if isinstance(declared_failed, list):
        failed.update(str(name) for name in declared_failed if name in view_names)
    if not cross_passed and not failed:
        failed.update(view_names)

    passed = not failed and cross_passed
    return {
        "passed": passed,
        "views": normalized_views,
        "cross_view": {
            "passed": cross_passed,
            **{field: cross.get(field) is True for field in cross_fields},
            "issues": [
                str(item)
                for item in (
                    cross.get("issues")
                    if isinstance(cross.get("issues"), list)
                    else [cross.get("issues") or ""]
                )
                if str(item).strip()
            ],
        },
        "failed_views": sorted(failed),
        "summary": str(payload.get("summary") or "").strip(),
    }


def review_character_reference_pack(
    reviewer: CharacterReferenceReviewer,
    view_paths: dict[str, Path],
    character_description: str,
) -> dict[str, Any]:
    view_names = tuple(view_paths)
    prompt = build_character_reference_qa_prompt(character_description, view_names)
    raw = reviewer.review([view_paths[name] for name in view_names], prompt)
    return parse_character_reference_qa(raw, view_names)


def build_character_reference_qa_receipt(
    *,
    char_id: str,
    view_paths: dict[str, Path],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    final = attempts[-1] if attempts else {"passed": False, "failed_views": list(view_paths)}
    return {
        "schema": CHARACTER_REFERENCE_QA_SCHEMA,
        "character_id": char_id,
        "status": "passed" if final.get("passed") is True else "failed",
        "inputs": {
            name: {"path": path.name, "sha256": file_sha256(path)}
            for name, path in view_paths.items()
            if path.is_file()
        },
        "attempts": attempts,
        "final": final,
    }


def validate_character_reference_qa_receipt(
    report_path: Path,
    view_paths: dict[str, Path],
) -> bool:
    """Reject missing, failed, incomplete, or stale semantic QA receipts."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        report.get("schema") != CHARACTER_REFERENCE_QA_SCHEMA
        or report.get("status") != "passed"
        or not isinstance(report.get("inputs"), dict)
    ):
        return False
    inputs = report["inputs"]
    for name, path in view_paths.items():
        item = inputs.get(name)
        if (
            not isinstance(item, dict)
            or not path.is_file()
            or item.get("sha256") != file_sha256(path)
        ):
            return False
    return True

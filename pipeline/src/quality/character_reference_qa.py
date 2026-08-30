"""Semantic QA contract for Phase 3 character reference packs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Protocol

from clients.ark_multimodal_client import review_as
from schemas.understanding import (
    CharacterReferenceUnderstanding,
    IdentityDetailUnderstanding,
    parse_structured_output,
)
from utils.character_reference_contracts import (
    IDENTITY_DETAIL_ASSET_POLICY,
    STATIC_REFERENCE_QA_POLICY,
    identity_detail_prompt_items,
)

CHARACTER_REFERENCE_QA_SCHEMA = "honcut.character-reference-qa.v5"
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
    synthetic_styling: dict[str, Any] | None = None,
) -> str:
    """Ask a VLM to verify view semantics rather than image attractiveness."""
    requirements = {
        "face_closeup": (
            "strict straight-on head-and-shoulders close-up; crown through clavicles; "
            "face occupies 60-75% of the frame; not a full-body or action image"
        ),
        "full_body": (
            "strict front full-body reference; hair top and both shoe soles visible; "
            "upright neutral stance, arms relaxed down, hands empty, feet parallel and hip-width"
        ),
        "side": (
            "strict 90-degree left side full-body profile; nose, torso, hips and toes all "
            "point left; only one eye is visible; head does not turn toward camera; no "
            "hand-supported or operated object"
        ),
        "back": (
            "strict 180-degree rear full-body view; back of head, shoulders, spine, outfit "
            "rear and heels visible; no eyes, nose, mouth, chest or front of torso visible; "
            "no hand-supported or operated object"
        ),
        "front": "strict straight-on identity portrait matching its requested framing",
        "three_quarter": "clear three-quarter identity portrait, neither front nor profile",
        "detail": "tight facial-detail close-up, not a full-body image",
    }
    ordered = "\n".join(
        f"- {name}: {requirements.get(name, 'match the filename view exactly')}"
        for name in view_names
    )
    synthetic_contract = ""
    if synthetic_styling:
        from utils.privacy_visual_policy import (
            synthetic_makeup_qa_requirements,
            synthetic_makeup_reference_qa_requirements,
        )

        synthetic_contract = f"""
Synthetic face contract (blocking):
{json.dumps(synthetic_styling, ensure_ascii=False, sort_keys=True)}
Structured aesthetic QA requirements:
{json.dumps(synthetic_makeup_qa_requirements(), ensure_ascii=False)}
Phase 3 evidence discipline:
{json.dumps(synthetic_makeup_reference_qa_requirements(), ensure_ascii=False)}
Every face-visible view must show the same declared synthetic porcelain makeup, keep the
whole face unobscured, preserve clean harmonious facial anatomy, and contain no grotesque
damage. The pearl ceramic complexion must look warm, healthy and elegant rather than gray,
blue-gray, bloodless, waxy or corpse-like. Eyes must retain clear pupils, layered irises and
bright catchlights instead of a blank solid glow; cheeks and lips must keep coordinated living
color. Circuit makeup must look like fine decorative cosmetics, never cuts, cracks or surgical
seams. Photoreal untreated human skin or a hidden face is a failure. The back view is exempt
from face visibility but must preserve the same hair and rear identity design.
For each face-visible view, report three independent anchor facts: the declared pearl-ceramic
material, the declared temple-to-upper-cheekbone circuit makeup, and the declared luminous ring
inside the iris around a dark pupil. An iris ring is not eyeliner: it must be visibly separated
from the eyelid and eyelashes. Do not collapse these facts into synthetic_profile_match.
"""
    return f"""You are the blocking Phase 3 character-reference inspector.
The input images are ordered and labelled by filename. Judge geometry and semantics, not beauty.

Static identity contract:
{character_description}

Asset-boundary contract:
{STATIC_REFERENCE_QA_POLICY}
{synthetic_contract}

Per-view contracts:
{ordered}

All images must contain exactly one instance of the character on a plain neutral studio
background. No street, shop, crowd, scenery, performance, dance pose, action pose, interaction
prop, text, watermark or logo is allowed. Full-body views must use the same neutral anatomical
reference stance. Body-worn or fastened wardrobe/accessories, face, hair, apparent age, head
scale and body proportions must remain the same across views.
Judge only the static facts explicitly written above. Never invent an undeclared collar, neckline,
opening, seam, ornament or accessory. In particular, "long-sleeved top" does not mean "high-neck
top"; when no neckline is declared, a clean round neck or high neck is not a mismatch. The model's
passed, failed_views and prose issues are diagnostic only; HonCut recomputes the verdict from the
individual evidence booleans below.

Return one JSON object only:
{{
  "views": {{
    "<view_name>": {{
      "passed": true,
      "view_match": true,
      "framing_match": true,
      "neutral_pose": true,
      "hands_empty": true,
      "plain_background": true,
      "single_character": true,
      "face_visible": true,
      "both_eyes_visible": false,
      "declared_identity_match": true,
      "declared_outfit_match": true,
      "synthetic_makeup_visible": true,
      "synthetic_profile_match": true,
      "synthetic_material_anchor_match": true,
      "circuit_makeup_anchor_match": true,
      "iris_ring_anchor_match": true,
      "face_unobscured": true,
      "makeup_clean_and_harmonious": true,
      "no_grotesque_damage": true,
      "healthy_warm_complexion": true,
      "lively_eyes_with_catchlights": true,
      "living_color_in_cheeks_and_lips": true,
      "no_uncanny_or_corpse_like_styling": true,
      "semantic_confidence": 0.95,
      "semantic_evidence": ["concrete visible fact supporting the booleans"],
      "issues": []
    }}
  }},
  "cross_view": {{
    "passed": true,
    "identity_consistent": true,
    "outfit_consistent": true,
    "body_proportions_consistent": true,
    "synthetic_makeup_consistent": true,
    "semantic_confidence": 0.95,
    "semantic_evidence": ["concrete cross-view comparison"],
    "issues": []
  }},
  "failed_views": [],
  "summary": "short factual summary"
}}

Confidence is confidence in the visible evidence, not attractiveness. A semantic match at 0.65
or above is sufficient. A negative finding can block only at 0.85 or above and must cite a
concrete visible fact in semantic_evidence. Lower-confidence negatives are diagnostic deviations.
Set passed=false for any uncertain or violated requirement. hands_empty means that no held,
hand-carried, raised, used or operated item is visible; it does not require hands to appear in
the face close-up. For back, face_visible and both_eyes_visible must both be false. For side,
both_eyes_visible must be false. When a
cross-view mismatch is localized, list the suspect filenames in failed_views; otherwise list
all supplied filenames. Do not excuse a wrong angle because identity is consistent."""


def build_identity_detail_qa_prompt(items: list[dict[str, Any]]) -> str:
    """Build the blocking review contract for a four-view-derived detail board."""
    return f"""You are the blocking Phase 3 identity-detail inspector.
Images 1 and 2 are the approved canonical face and full-body references. Image 3 is the
supplemental identity-detail board derived from them.

Declared identity-detail items:
{identity_detail_prompt_items(items)}

Detail-board policy:
{IDENTITY_DETAIL_ASSET_POLICY}

Verify that the character identity, outfit base colors, and body-worn markers match images 1-2;
every declared item is visible with its exact authored geometry, colors, materials, markings and
attachment mode; isolated_handheld items are shown detached and are not held or operated; and no
undeclared prop, location, action pose, second character, text, watermark or logo was introduced.

Return one JSON object only:
{{"passed":true,"character_identity_consistent":true,"declared_items_present":true,
"item_geometry_consistent":true,"colors_materials_consistent":true,
"attachment_modes_correct":true,"undeclared_items_absent":true,"issues":[]}}
Set passed=false whenever any required item or consistency fact is uncertain."""


def parse_identity_detail_qa(raw: str) -> dict[str, Any]:
    """Parse and recompute the identity-detail verdict from explicit evidence."""
    payload = parse_structured_output(
        raw,
        IdentityDetailUnderstanding,
    ).model_dump()
    fields = (
        "character_identity_consistent",
        "declared_items_present",
        "item_geometry_consistent",
        "colors_materials_consistent",
        "attachment_modes_correct",
        "undeclared_items_absent",
    )
    evidence = {field: payload.get(field) is True for field in fields}
    issues = payload.get("issues")
    issues = issues if isinstance(issues, list) else [str(issues or "")]
    return {
        "passed": payload.get("passed") is True and all(evidence.values()),
        **evidence,
        "issues": [str(item) for item in issues if str(item).strip()],
    }


def review_identity_detail_reference(
    reviewer: CharacterReferenceReviewer,
    canonical_paths: list[Path],
    detail_path: Path,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Review one detail board against the approved neutral identity references."""
    result = review_as(
        reviewer,
        [*canonical_paths, detail_path],
        build_identity_detail_qa_prompt(items),
        IdentityDetailUnderstanding,
    )
    return parse_identity_detail_qa(result.model_dump_json())


def parse_character_reference_qa(
    raw: str,
    view_names: tuple[str, ...] = SEEDANCE_REFERENCE_VIEWS,
    *,
    require_synthetic: bool = False,
) -> dict[str, Any]:
    """Normalize a review and recompute the verdict from its evidence fields."""
    payload = parse_structured_output(
        raw,
        CharacterReferenceUnderstanding,
    ).model_dump()
    raw_views = payload.get("views")
    cross = payload.get("cross_view")
    if not isinstance(raw_views, dict) or not isinstance(cross, dict):
        raise CharacterReferenceQAError(
            "character reference QA requires views and cross_view objects"
        )

    from quality.visual_qa_policy import decide_visual_qa

    normalized_views: dict[str, dict[str, Any]] = {}
    failed: set[str] = set()
    decisions: list[dict[str, Any]] = []
    seedance_pack = set(SEEDANCE_REFERENCE_VIEWS).issubset(view_names)
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
        authored_fields = (
            "declared_identity_match",
            "declared_outfit_match",
        )
        passed = all(
            evidence.get(field) is True
            for field in (*boolean_fields, *authored_fields)
        )
        if name == "back":
            passed = (
                passed
                and evidence.get("face_visible") is False
                and evidence.get("both_eyes_visible") is False
            )
        if name == "side":
            passed = passed and evidence.get("both_eyes_visible") is False
        if seedance_pack and name in {"full_body", "side", "back"}:
            passed = passed and evidence.get("hands_empty") is True
        synthetic_fields = (
            "synthetic_makeup_visible",
            "synthetic_profile_match",
            "synthetic_material_anchor_match",
            "circuit_makeup_anchor_match",
            "iris_ring_anchor_match",
            "face_unobscured",
            "makeup_clean_and_harmonious",
            "no_grotesque_damage",
            "healthy_warm_complexion",
            "lively_eyes_with_catchlights",
            "living_color_in_cheeks_and_lips",
            "no_uncanny_or_corpse_like_styling",
        )
        if require_synthetic and name != "back":
            passed = passed and all(
                evidence.get(field) is True for field in synthetic_fields
                if field != "synthetic_profile_match"
            )
        confidence = float(evidence.get("semantic_confidence", 1.0))
        semantic_evidence = [
            str(item).strip()
            for item in evidence.get("semantic_evidence") or issues
            if str(item).strip()
        ]
        violated = [
            field
            for field in (*boolean_fields, *authored_fields)
            if evidence.get(field) is not True
        ]
        if name == "back" and (
            evidence.get("face_visible") is not False
            or evidence.get("both_eyes_visible") is not False
        ):
            violated.append("rear_face_visibility")
        if name == "side" and evidence.get("both_eyes_visible") is not False:
            violated.append("profile_eye_visibility")
        if seedance_pack and name in {"full_body", "side", "back"} and evidence.get("hands_empty") is not True:
            violated.append("hands_empty")
        if require_synthetic and name != "back":
            violated.extend(
                field
                for field in synthetic_fields
                if field != "synthetic_profile_match" and evidence.get(field) is not True
            )
        policy = decide_visual_qa(
            semantic_score=confidence,
            findings=[
                {
                    "blocking_category": "character_reference_semantics",
                    "confidence": confidence,
                    "evidence": semantic_evidence,
                    "field": field,
                }
                for field in dict.fromkeys(violated)
            ],
        )
        accepted = policy.verdict in {"pass", "acceptable_deviation"}
        if not accepted:
            failed.add(name)
        decisions.append({"scope": name, **policy.as_dict()})
        normalized_views[name] = {
            "passed": accepted,
            **{field: evidence.get(field) is True for field in boolean_fields},
            **{field: evidence.get(field) is True for field in authored_fields},
            "face_visible": evidence.get("face_visible") is True,
            "both_eyes_visible": evidence.get("both_eyes_visible") is True,
            "hands_empty": evidence.get("hands_empty") is True,
            **{
                field: evidence.get(field) is True
                for field in synthetic_fields
            },
            "issues": [str(item) for item in issues if str(item).strip()],
            "semantic_confidence": confidence,
            "semantic_evidence": semantic_evidence,
            "qa_verdict": policy.verdict,
        }

    cross_fields = (
        "identity_consistent",
        "outfit_consistent",
        "body_proportions_consistent",
    )
    cross_semantics_match = all(
        cross.get(field) is True for field in cross_fields
    )
    if require_synthetic:
        cross_semantics_match = (
            cross_semantics_match
            and cross.get("synthetic_makeup_consistent") is True
        )
    cross_confidence = float(cross.get("semantic_confidence", 1.0))
    cross_evidence = [
        str(item).strip()
        for item in cross.get("semantic_evidence") or cross.get("issues") or []
        if str(item).strip()
    ]
    cross_policy = decide_visual_qa(
        semantic_score=cross_confidence,
        findings=(
            []
            if cross_semantics_match
            else [{
                "blocking_category": "cross_view_identity",
                "confidence": cross_confidence,
                "evidence": cross_evidence,
            }]
        ),
    )
    cross_passed = cross_policy.verdict in {"pass", "acceptable_deviation"}
    decisions.append({"scope": "cross_view", **cross_policy.as_dict()})
    if not cross_passed and not failed:
        failed.update(view_names)

    passed = not failed and cross_passed
    overall_verdict = (
        "block"
        if any(item["verdict"] == "block" for item in decisions)
        else "manual_review"
        if any(item["verdict"] == "manual_review" for item in decisions)
        else "acceptable_deviation"
        if any(item["verdict"] == "acceptable_deviation" for item in decisions)
        else "pass"
    )
    return {
        "passed": passed,
        "views": normalized_views,
        "cross_view": {
            "passed": cross_passed,
            **{field: cross.get(field) is True for field in cross_fields},
            "synthetic_makeup_consistent": (
                cross.get("synthetic_makeup_consistent") is True
            ),
            "issues": [
                str(item)
                for item in (
                    cross.get("issues")
                    if isinstance(cross.get("issues"), list)
                    else [cross.get("issues") or ""]
                )
                if str(item).strip()
            ],
            "semantic_confidence": cross_confidence,
            "semantic_evidence": cross_evidence,
            "qa_verdict": cross_policy.verdict,
        },
        "failed_views": sorted(failed),
        "summary": str(payload.get("summary") or "").strip(),
        "qa_verdict": overall_verdict,
        "qa_policy_decisions": decisions,
    }


def review_character_reference_pack(
    reviewer: CharacterReferenceReviewer,
    view_paths: dict[str, Path],
    character_description: str,
    synthetic_styling: dict[str, Any] | None = None,
    *,
    before_provider_request: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    view_names = tuple(view_paths)
    prompt = build_character_reference_qa_prompt(
        character_description,
        view_names,
        synthetic_styling,
    )
    ordered_paths = [view_paths[name] for name in view_names]
    output_dir = ordered_paths[0].parent.parent.parent
    from quality.visual_qa_policy import POLICY_ID, policy_sha256
    from runtime.qa_ledger import QALedger, observation_fingerprint
    from utils.canonical_visual_contracts import load_canonical_visual_contract

    contract = load_canonical_visual_contract(output_dir)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    evaluator_model = str(getattr(reviewer, "model", "unknown-vlm"))
    evidence = [
        {"path": path.relative_to(output_dir).as_posix(), "sha256": file_sha256(path)}
        for path in ordered_paths
    ]
    fingerprint = observation_fingerprint(
        evidence=evidence,
        canonical_contract_sha256=contract["contract_sha256"],
        evaluator_model=evaluator_model,
        prompt_sha256=prompt_sha256,
        observation_schema="CharacterReferenceUnderstanding.v2",
    )
    ledger = QALedger(output_dir / "runtime.db")
    observation = ledger.find_observation(fingerprint)
    observation_reused = observation is not None
    if observation is None:
        if before_provider_request is not None:
            before_provider_request({
                "provider_family": "multimodal_observation",
                "phase": "phase3",
                "resource_id": ordered_paths[0].parent.name,
                "model": evaluator_model,
                "observation_schema": "CharacterReferenceUnderstanding.v2",
                "evidence_fingerprint": fingerprint,
                "prompt_sha256": prompt_sha256,
                "inputs": evidence,
            })
        result = review_as(
            reviewer,
            ordered_paths,
            prompt,
            CharacterReferenceUnderstanding,
        )
        raw_observation = result.model_dump(mode="json")
        manifest_path = output_dir / "RUN_MANIFEST.json"
        try:
            run_id = str(json.loads(manifest_path.read_text(encoding="utf-8"))["run_fingerprint"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            run_id = output_dir.name
        observation, _ = ledger.record_observation(
            run_id=run_id,
            phase="phase3",
            resource_id=ordered_paths[0].parent.name,
            evidence_fingerprint=fingerprint,
            canonical_contract_sha256=contract["contract_sha256"],
            evaluator_model=evaluator_model,
            prompt_sha256=prompt_sha256,
            observation_schema="CharacterReferenceUnderstanding.v2",
            observation=raw_observation,
        )
    typed = CharacterReferenceUnderstanding.model_validate(observation.observation)
    parsed = parse_character_reference_qa(
        typed.model_dump_json(),
        view_names,
        require_synthetic=bool(synthetic_styling),
    )
    decision, decision_reused = ledger.record_decision(
        observation_id=observation.observation_id,
        phase_owner="phase3.character_reference_qa",
        policy_id=POLICY_ID,
        policy_sha256=policy_sha256(),
        verdict=parsed["qa_verdict"],
        semantic_score=min(
            [item["semantic_confidence"] for item in parsed["views"].values()]
            + [parsed["cross_view"]["semantic_confidence"]]
        ),
        decision={
            "verdict": parsed["qa_verdict"],
            "scopes": parsed["qa_policy_decisions"],
        },
    )
    parsed.update({
        "qa_observation_id": observation.observation_id,
        "qa_observation_reused": observation_reused,
        "qa_decision_id": decision.decision_id,
        "qa_decision_reused": decision_reused,
    })
    return parsed


def build_character_reference_qa_receipt(
    *,
    char_id: str,
    view_paths: dict[str, Path],
    attempts: list[dict[str, Any]],
    synthetic_styling: dict[str, Any] | None = None,
    generation_contract: dict[str, Any] | None = None,
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
        "synthetic_styling": synthetic_styling,
        "generation_contract": generation_contract,
        "attempts": attempts,
        "final": final,
    }


def validate_character_reference_qa_receipt(
    report_path: Path,
    view_paths: dict[str, Path],
    *,
    synthetic_styling: dict[str, Any] | None = None,
    generation_contract: dict[str, Any] | None = None,
) -> bool:
    """Reject missing, failed, incomplete, or stale semantic QA receipts."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if report.get("synthetic_styling") != synthetic_styling:
        return False
    if generation_contract is not None and report.get("generation_contract") != generation_contract:
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

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
)

CHARACTER_REFERENCE_QA_SCHEMA = "honcut.character-reference-qa.v5"
PROP_DETAIL_INPUT_SCHEMA = "honcut.prop-detail-board-input.v2"
PROP_DETAIL_OBSERVATION_SCHEMA = "honcut.prop-detail-observation.v2"
PROP_DETAIL_QA_SCHEMA = "honcut.prop-detail-board-qa.v2"
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


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_identity_detail_logical_items(
    output_dir: Path,
    char_id: str,
    items: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Bind authored Phase 3 items to canonical logical identities and view roles."""
    from utils.canonical_visual_contracts import load_canonical_visual_contract

    contract = load_canonical_visual_contract(output_dir)
    record = next(
        (
            candidate
            for candidate in contract["characters"]
            if char_id in {
                str(candidate["character_id"]),
                str(candidate["entity_id"]),
                *(
                    str(instance["instance_id"])
                    for instance in candidate["instances"]
                ),
            }
        ),
        None,
    )
    if record is None:
        raise CharacterReferenceQAError(
            f"{char_id} is absent from the canonical visual contract"
        )
    canonical_by_id = {
        str(item["prop_id"]): item
        for item in record["identity_props"]
    }
    authored_by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item.get("id") or "").strip()
        if not item_id or item_id in authored_by_id:
            raise CharacterReferenceQAError(
                f"{char_id} identity props require unique non-empty IDs"
            )
        authored_by_id[item_id] = item
    if set(authored_by_id) != set(canonical_by_id):
        raise CharacterReferenceQAError(
            f"{char_id} identity props disagree with canonical item IDs"
        )

    logical_items: list[dict[str, Any]] = []
    for item_id, item in authored_by_id.items():
        canonical = canonical_by_id[item_id]
        attachment_mode = str(item["attachment_mode"])
        depiction_roles = (
            ["front", "side", "three_quarter"]
            if attachment_mode == "isolated_handheld"
            else ["attachment_context", "material_detail"]
        )
        geometry = {
            key: (
                value["value"]
                if isinstance(value, dict) and "value" in value
                else value
            )
            for key, value in canonical["geometry"].items()
        }
        logical_items.append({
            "logical_item_id": item_id,
            "display_name": str(item["name"]),
            "attachment_mode": attachment_mode,
            "persistence": str(item["persistence"]),
            "depiction_roles": depiction_roles,
            "canonical_geometry": geometry,
        })
    return str(contract["contract_sha256"]), logical_items


def build_identity_detail_input_contract(
    *,
    char_id: str,
    character_description: str,
    identity_props: list[dict[str, Any]],
    canonical_contract_sha256: str,
    logical_items: list[dict[str, Any]],
    prompt_sha256: str,
    canonical_paths: list[Path],
) -> dict[str, Any]:
    """Create the deterministic authority consumed by generation, QA, and replay."""
    if not logical_items:
        raise CharacterReferenceQAError("prop detail contract requires logical items")
    expected_reference_roles = (
        "character_face_identity",
        "character_body_identity",
    )
    if len(canonical_paths) != len(expected_reference_roles):
        raise CharacterReferenceQAError(
            "prop detail contract requires face and body identity references"
        )
    references = []
    for path, media_role in zip(
        canonical_paths,
        expected_reference_roles,
        strict=True,
    ):
        if not path.is_file():
            raise CharacterReferenceQAError(
                f"canonical reference is missing: {path.name}"
            )
        references.append({
            "path": path.name,
            "sha256": file_sha256(path),
            "media_role": media_role,
        })
    depictions = [
        {
            "depiction_id": (
                f"{item['logical_item_id']}:{view_role}"
            ),
            "logical_item_id": item["logical_item_id"],
            "view_role": view_role,
        }
        for item in logical_items
        for view_role in item["depiction_roles"]
    ]
    lineage = [
        {
            "parent_role": "canonical_visual_contract",
            "sha256": canonical_contract_sha256,
        },
        *(
            {
                "parent_role": reference["media_role"],
                "sha256": reference["sha256"],
            }
            for reference in references
        ),
    ]
    return {
        "schema": PROP_DETAIL_INPUT_SCHEMA,
        "character_id": char_id,
        "canonical_visual_contract_sha256": canonical_contract_sha256,
        "character_description_sha256": hashlib.sha256(
            character_description.encode("utf-8")
        ).hexdigest(),
        "prompt_sha256": prompt_sha256,
        "identity_props_sha256": _canonical_json_sha256(identity_props),
        "logical_item_count": len(logical_items),
        "logical_items_sha256": _canonical_json_sha256(logical_items),
        "logical_items": logical_items,
        "depiction_count": len(depictions),
        "depictions_sha256": _canonical_json_sha256(depictions),
        "depictions": depictions,
        "detail_media_role": "identity_prop_geometry_reference",
        "canonical_references": references,
        "parent_lineage_sha256": _canonical_json_sha256(lineage),
        "parent_lineage": lineage,
    }


def validate_identity_detail_input_contract(
    *,
    output_dir: Path,
    canonical_paths: list[Path],
    input_contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fail closed on every deterministic v2 prop-detail authority field."""
    from utils.canonical_visual_contracts import load_canonical_visual_contract

    if input_contract.get("schema") != PROP_DETAIL_INPUT_SCHEMA:
        raise CharacterReferenceQAError("prop detail input schema is not current")
    contract = load_canonical_visual_contract(output_dir)
    if input_contract.get("canonical_visual_contract_sha256") != contract.get(
        "contract_sha256"
    ):
        raise CharacterReferenceQAError("prop detail canonical contract hash mismatch")

    logical_items = input_contract.get("logical_items")
    if (
        not isinstance(logical_items, list)
        or not logical_items
        or input_contract.get("logical_item_count") != len(logical_items)
        or input_contract.get("logical_items_sha256")
        != _canonical_json_sha256(logical_items)
    ):
        raise CharacterReferenceQAError("prop detail logical item contract is invalid")
    logical_ids = [str(item.get("logical_item_id") or "") for item in logical_items]
    if not all(logical_ids) or len(logical_ids) != len(set(logical_ids)):
        raise CharacterReferenceQAError("prop detail logical item IDs are invalid")

    char_id = str(input_contract.get("character_id") or "")
    canonical_record = next(
        (
            record
            for record in contract["characters"]
            if char_id in {
                str(record["character_id"]),
                str(record["entity_id"]),
                *(str(value["instance_id"]) for value in record["instances"]),
            }
        ),
        None,
    )
    canonical_ids = (
        [str(item["prop_id"]) for item in canonical_record["identity_props"]]
        if canonical_record is not None
        else []
    )
    if set(logical_ids) != set(canonical_ids) or len(logical_ids) != len(canonical_ids):
        raise CharacterReferenceQAError(
            "prop detail logical item IDs disagree with canonical authority"
        )

    expected_depictions = [
        {
            "depiction_id": f"{item['logical_item_id']}:{view_role}",
            "logical_item_id": item["logical_item_id"],
            "view_role": view_role,
        }
        for item in logical_items
        for view_role in item.get("depiction_roles") or []
    ]
    if (
        input_contract.get("depictions") != expected_depictions
        or input_contract.get("depiction_count") != len(expected_depictions)
        or input_contract.get("depictions_sha256")
        != _canonical_json_sha256(expected_depictions)
    ):
        raise CharacterReferenceQAError("prop detail depiction contract is invalid")
    if input_contract.get("detail_media_role") != "identity_prop_geometry_reference":
        raise CharacterReferenceQAError("prop detail media role is invalid")

    expected_roles = (
        "character_face_identity",
        "character_body_identity",
    )
    canonical_references = input_contract.get("canonical_references")
    if (
        len(canonical_paths) != len(expected_roles)
        or not isinstance(canonical_references, list)
        or len(canonical_references) != len(expected_roles)
    ):
        raise CharacterReferenceQAError("prop detail canonical references are incomplete")
    for path, expected_role, reference in zip(
        canonical_paths,
        expected_roles,
        canonical_references,
        strict=True,
    ):
        if (
            not isinstance(reference, dict)
            or not path.is_file()
            or reference.get("path") != path.name
            or reference.get("media_role") != expected_role
            or reference.get("sha256") != file_sha256(path)
        ):
            raise CharacterReferenceQAError(
                "prop detail canonical reference hash or media role mismatch"
            )
    expected_lineage = [
        {
            "parent_role": "canonical_visual_contract",
            "sha256": contract["contract_sha256"],
        },
        *(
            {
                "parent_role": reference["media_role"],
                "sha256": reference["sha256"],
            }
            for reference in canonical_references
        ),
    ]
    if (
        input_contract.get("parent_lineage") != expected_lineage
        or input_contract.get("parent_lineage_sha256")
        != _canonical_json_sha256(expected_lineage)
    ):
        raise CharacterReferenceQAError("prop detail parent lineage mismatch")
    return contract, logical_items


def build_identity_detail_qa_prompt(logical_items: list[dict[str, Any]]) -> str:
    """Build a typed review contract where logical items and views are distinct."""
    return f"""You are the blocking Phase 3 identity-detail inspector.
Images 1 and 2 are the approved canonical face and full-body references. Image 3 is the
supplemental identity-detail board derived from them.

Declared logical items and their permitted depictions:
{json.dumps(logical_items, ensure_ascii=False, sort_keys=True)}

Detail-board policy:
{IDENTITY_DETAIL_ASSET_POLICY}

Verify that the character identity, outfit base colors, and body-worn markers match images 1-2;
every declared logical item is visible with its exact canonical geometry, colors, materials and
attachment mode; and no undeclared logical item, location, action pose, second character, text,
watermark or logo was introduced. A front, side, three-quarter, crop, or material view is a
depiction of its declared logical_item_id, not another logical item. Count logical identities,
not repeated views. Multiple depictions are valid only when they are mutually consistent.

Return one JSON object only:
{{"schema":"{PROP_DETAIL_OBSERVATION_SCHEMA}","passed":true,
"character_identity_consistent":true,"character_identity_confidence":0.95,
"character_identity_evidence":["visible comparison"],"items":[{{
"logical_item_id":"declared ID","logical_identity_present":true,"depiction_count":3,
"depictions_mutually_consistent":true,"topology_consistent":true,
"colors_materials_consistent":true,"attachment_mode_correct":true,
"undeclared_logical_item_evidence":[],"semantic_confidence":0.95,
"semantic_evidence":["concrete visible item evidence"],"issues":[]}}],
"no_undeclared_logical_items":true,"undeclared_items_confidence":0.95,
"undeclared_items_evidence":["visible inventory comparison"],"issues":[]}}

Return exactly one item entry for every declared logical_item_id and no unknown IDs. The aggregate
passed value is diagnostic only. Confidence is confidence in visible evidence. Low-confidence
uncertainty must remain evidence-based; do not convert the expected depiction count into an item
count mismatch."""


def parse_identity_detail_qa(
    raw: str,
    logical_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Parse a typed Observation and compute policy inputs without trusting passed."""
    payload = parse_structured_output(
        raw,
        IdentityDetailUnderstanding,
    ).model_dump(by_alias=True)
    expected_ids = [str(item["logical_item_id"]) for item in logical_items]
    actual_ids = [str(item["logical_item_id"]) for item in payload["items"]]
    deterministic_errors: list[dict[str, Any]] = []
    if len(actual_ids) != len(set(actual_ids)):
        deterministic_errors.append({
            "category": "schema",
            "evidence": "duplicate logical_item_id in typed observation",
        })
    if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(expected_ids):
        deterministic_errors.append({
            "category": "canonical_contract",
            "evidence": {
                "expected_logical_item_ids": expected_ids,
                "observed_logical_item_ids": actual_ids,
            },
        })

    findings: list[dict[str, Any]] = []
    confidences = [
        float(payload["character_identity_confidence"]),
        float(payload["undeclared_items_confidence"]),
    ]
    if payload["character_identity_consistent"] is not True:
        findings.append({
            "blocking_category": "character_identity",
            "confidence": payload["character_identity_confidence"],
            "evidence": payload["character_identity_evidence"],
        })
    category_by_field = {
        "logical_identity_present": "logical_item_identity",
        "depictions_mutually_consistent": "logical_item_identity",
        "topology_consistent": "prop_topology",
        "colors_materials_consistent": "prop_appearance",
        "attachment_mode_correct": "attachment_mode",
    }
    for item in payload["items"]:
        confidences.append(float(item["semantic_confidence"]))
        for field, category in category_by_field.items():
            if item[field] is not True:
                findings.append({
                    "blocking_category": category,
                    "confidence": item["semantic_confidence"],
                    "evidence": item["semantic_evidence"],
                    "logical_item_id": item["logical_item_id"],
                    "field": field,
                })
        if item["depiction_count"] < 1:
            findings.append({
                "blocking_category": "logical_item_identity",
                "confidence": item["semantic_confidence"],
                "evidence": item["semantic_evidence"],
                "logical_item_id": item["logical_item_id"],
                "field": "depiction_count",
            })
        if item["undeclared_logical_item_evidence"]:
            findings.append({
                "blocking_category": "undeclared_logical_item",
                "confidence": item["semantic_confidence"],
                "evidence": item["undeclared_logical_item_evidence"],
                "logical_item_id": item["logical_item_id"],
            })
    if payload["no_undeclared_logical_items"] is not True:
        findings.append({
            "blocking_category": "undeclared_logical_item",
            "confidence": payload["undeclared_items_confidence"],
            "evidence": payload["undeclared_items_evidence"],
        })

    from quality.visual_qa_policy import decide_visual_qa

    semantic_score = min(confidences) if confidences else None
    policy = decide_visual_qa(
        semantic_score=semantic_score,
        findings=findings,
        deterministic_errors=deterministic_errors,
    )
    return {
        **payload,
        "model_passed_diagnostic": payload["passed"],
        "passed": policy.verdict in {"pass", "acceptable_deviation"},
        "qa_verdict": policy.verdict,
        "semantic_score": policy.semantic_score,
        "deterministic_errors": deterministic_errors,
        "findings": findings,
        "policy_decision": policy.as_dict(),
    }


def evaluate_identity_detail_observation(
    *,
    output_dir: Path,
    character_id: str,
    evidence: list[dict[str, Any]],
    canonical_contract_sha256: str,
    evaluator_model: str,
    prompt_sha256: str,
    logical_items: list[dict[str, Any]],
    observation_payload: dict[str, Any] | None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Persist or reuse typed evidence and apply policy without any Provider call."""
    from quality.visual_qa_policy import POLICY_ID, policy_sha256
    from runtime.qa_ledger import QALedger, observation_fingerprint

    fingerprint = observation_fingerprint(
        evidence=evidence,
        canonical_contract_sha256=canonical_contract_sha256,
        evaluator_model=evaluator_model,
        prompt_sha256=prompt_sha256,
        observation_schema=PROP_DETAIL_OBSERVATION_SCHEMA,
    )
    ledger = QALedger(output_dir / "runtime.db")
    observation = ledger.find_observation(fingerprint)
    observation_reused = observation is not None
    if observation is None:
        if observation_payload is None:
            raise CharacterReferenceQAError(
                "prop detail Observation is missing for immutable evidence"
            )
        validated = IdentityDetailUnderstanding.model_validate(observation_payload)
        observation, _ = ledger.record_observation(
            run_id=run_id or output_dir.name,
            phase="phase3",
            resource_id=character_id,
            evidence_fingerprint=fingerprint,
            canonical_contract_sha256=canonical_contract_sha256,
            evaluator_model=evaluator_model,
            prompt_sha256=prompt_sha256,
            observation_schema=PROP_DETAIL_OBSERVATION_SCHEMA,
            observation=validated.model_dump(mode="json", by_alias=True),
        )
    parsed = parse_identity_detail_qa(
        IdentityDetailUnderstanding.model_validate(
            observation.observation
        ).model_dump_json(by_alias=True),
        logical_items,
    )
    decision, decision_reused = ledger.record_decision(
        observation_id=observation.observation_id,
        phase_owner="phase3.prop_detail_qa",
        policy_id=POLICY_ID,
        policy_sha256=policy_sha256(),
        verdict=parsed["qa_verdict"],
        semantic_score=parsed["semantic_score"],
        decision=parsed["policy_decision"],
    )
    parsed.update({
        "qa_observation_id": observation.observation_id,
        "qa_observation_reused": observation_reused,
        "qa_decision_id": decision.decision_id,
        "qa_decision_reused": decision_reused,
        "qa_prompt_sha256": prompt_sha256,
    })
    return parsed


def review_identity_detail_reference(
    reviewer: CharacterReferenceReviewer,
    canonical_paths: list[Path],
    detail_path: Path,
    input_contract: dict[str, Any],
) -> dict[str, Any]:
    """Persist one typed Observation, then apply the current deterministic policy."""
    output_dir = next(
        (
            parent
            for parent in detail_path.parents
            if (parent / "CANONICAL_VISUAL_CONTRACT.json").is_file()
        ),
        None,
    )
    if output_dir is None:
        raise CharacterReferenceQAError(
            "prop detail QA cannot resolve the canonical run boundary"
        )
    contract, logical_items = validate_identity_detail_input_contract(
        output_dir=output_dir,
        canonical_paths=canonical_paths,
        input_contract=input_contract,
    )
    prompt = build_identity_detail_qa_prompt(logical_items)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    evaluator_model = str(getattr(reviewer, "model", "unknown-vlm"))
    image_paths = [*canonical_paths, detail_path]
    evidence = [
        {"path": path.relative_to(output_dir).as_posix(), "sha256": file_sha256(path)}
        for path in image_paths
    ]
    from runtime.qa_ledger import QALedger, observation_fingerprint

    fingerprint = observation_fingerprint(
        evidence=evidence,
        canonical_contract_sha256=contract["contract_sha256"],
        evaluator_model=evaluator_model,
        prompt_sha256=prompt_sha256,
        observation_schema=PROP_DETAIL_OBSERVATION_SCHEMA,
    )
    existing = QALedger(output_dir / "runtime.db").find_observation(fingerprint)
    observation_payload: dict[str, Any] | None = None
    if existing is None:
        result = review_as(
            reviewer,
            image_paths,
            prompt,
            IdentityDetailUnderstanding,
        )
        observation_payload = result.model_dump(mode="json", by_alias=True)
    try:
        run_id = str(json.loads(
            (output_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8")
        )["run_fingerprint"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        run_id = output_dir.name
    return evaluate_identity_detail_observation(
        output_dir=output_dir,
        character_id=str(input_contract["character_id"]),
        evidence=evidence,
        canonical_contract_sha256=contract["contract_sha256"],
        evaluator_model=evaluator_model,
        prompt_sha256=prompt_sha256,
        logical_items=logical_items,
        observation_payload=observation_payload,
        run_id=run_id,
    )


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

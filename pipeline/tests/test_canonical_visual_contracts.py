from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from utils.canonical_visual_contracts import (
    CANONICAL_VISUAL_CONTRACT_FILENAME,
    CanonicalVisualContractError,
    apply_character_visual_policy,
    build_canonical_visual_contract,
    persist_canonical_visual_contract,
    validate_canonical_visual_contract,
)
from utils.privacy_visual_policy import (
    is_current_synthetic_styling,
    is_synthetic_visual_identity_policy,
    synthetic_character_review_evidence,
)


def _character(
    character_id: str,
    *,
    hair: str,
    prop_description: str,
    face: str = "original fictional face",
) -> dict:
    return {
        "id": character_id,
        "name": character_id,
        "role": "protagonist",
        "appearance": {
            "gender": "male",
            "age_range": "adult",
            "height": "average",
            "build": "lean athletic",
            "hair": hair,
            "face": face,
            "clothing": "dark tailored jacket",
            "identity_props": [{
                "id": f"{character_id}_prop",
                "name": "signature tool",
                "description": prop_description,
                "attachment_mode": "isolated_handheld",
                "persistence": "role_active",
                "reference_required": True,
            }],
            "summary": "stable fictional adult character",
        },
    }


@pytest.mark.parametrize(
    ("hair", "expected_length", "prop", "expected_shape", "expected_ends"),
    [
        ("black short straight hair", "short", "transparent circular palm-sized tool", "circular", 0),
        ("silver shoulder-length wavy hair", "shoulder", "single-ended metal blade with one handle", "blade", 1),
        ("dark brown waist-length braided hair", "waist", "double-ended composite rod with two handles", "rod", 2),
    ],
)
def test_contract_projects_general_hair_and_prop_geometry(
    hair, expected_length, prop, expected_shape, expected_ends
):
    characters = {"characters": [_character("subject", hair=hair, prop_description=prop)]}
    rewritten = apply_character_visual_policy(characters, "source_derived")
    contract = build_canonical_visual_contract(
        rewritten,
        requested_policy="source_derived",
    )

    record = contract["characters"][0]
    assert record["hair"]["length_class"]["value"] == expected_length
    assert record["identity_props"][0]["geometry"]["shape_family"]["value"] == expected_shape
    assert record["identity_props"][0]["geometry"]["active_end_count"]["value"] == expected_ends
    assert validate_canonical_visual_contract(contract) == contract


def test_contract_completion_and_hash_are_repeatable(tmp_path):
    source = {
        "characters": [
            _character("stable-a", hair="unadorned hair", prop_description="signature object")
        ]
    }
    first_data, first = persist_canonical_visual_contract(
        tmp_path,
        copy.deepcopy(source),
        requested_policy="source_derived",
    )
    second_data, second = persist_canonical_visual_contract(
        tmp_path,
        copy.deepcopy(source),
        requested_policy="source_derived",
    )

    assert first == second
    assert first_data == second_data
    assert json.loads(
        (tmp_path / CANONICAL_VISUAL_CONTRACT_FILENAME).read_text(encoding="utf-8")
    ) == first
    assert first_data["canonical_visual_contract_sha256"] == first["contract_sha256"]


def test_contract_allows_a_source_with_no_characters(tmp_path):
    projected, contract = persist_canonical_visual_contract(
        tmp_path,
        {"characters": [], "total_characters": 0},
        requested_policy="source_derived",
    )

    assert contract["characters"] == []
    assert projected["characters"] == []
    assert projected["resolved_character_visual_policies"] == []
    assert validate_canonical_visual_contract(contract) == contract


@pytest.mark.parametrize("instance_count", [1, 3, 7])
def test_contract_preserves_explicit_character_instance_count(instance_count):
    character = _character(
        "group",
        hair="short hair",
        prop_description="single metal tool",
    )
    character["instance_count"] = instance_count
    rewritten = apply_character_visual_policy(
        {"characters": [character]},
        "fictional_cinematic_human_v1",
    )

    contract = build_canonical_visual_contract(
        rewritten,
        requested_policy="fictional_cinematic_human_v1",
    )

    fact = contract["characters"][0]["instance_count"]
    assert fact == {
        "value": instance_count,
        "origin": "explicit_source",
        "source_refs": ["character:group:instance_count"],
    }


def test_source_derived_policy_is_per_character():
    source = {
        "characters": [
            _character("organic", hair="black short hair", prop_description="round tool"),
            _character(
                "constructed",
                hair="designed fiber hair",
                prop_description="metal tool",
                face="explicit synthetic android face",
            ),
        ]
    }
    rewritten = apply_character_visual_policy(source, "source_derived")
    policies = {
        character["id"]: character["visual_identity_policy"]
        for character in rewritten["characters"]
    }
    assert policies == {
        "organic": "fictional_cinematic_human_v1",
        "constructed": "synthetic_stylized_character_v3",
    }
    by_id = {character["id"]: character for character in rewritten["characters"]}
    assert "synthetic_styling" not in by_id["organic"]["appearance"]
    assert is_current_synthetic_styling(
        by_id["constructed"]["appearance"]["synthetic_styling"]
    )


def test_mixed_cast_enables_qa_only_for_complete_synthetic_subset(tmp_path):
    source = {
        "characters": [
            _character("organic", hair="black short hair", prop_description="tool"),
            _character(
                "constructed",
                hair="designed fiber hair",
                prop_description="metal tool",
                face="explicit synthetic android face",
            ),
        ]
    }
    projected, _contract = persist_canonical_visual_contract(
        tmp_path,
        source,
        requested_policy="source_derived",
    )
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps(projected), encoding="utf-8"
    )

    evidence = synthetic_character_review_evidence(tmp_path)
    assert evidence["enabled"] is True
    assert evidence["synthetic_character_ids"] == ["constructed"]
    assert evidence["synthetic_character_count"] == 1
    assert evidence["identity_contract_complete"] is True


def test_contract_hash_tampering_fails_closed():
    source = {"characters": [_character("c", hair="short hair", prop_description="tool")]}
    rewritten = apply_character_visual_policy(source, "fictional_cinematic_human_v1")
    contract = build_canonical_visual_contract(
        rewritten,
        requested_policy="fictional_cinematic_human_v1",
    )
    contract["characters"][0]["hair"]["length_class"]["value"] = "long"
    with pytest.raises(CanonicalVisualContractError, match="hash mismatch"):
        validate_canonical_visual_contract(contract)


def test_legacy_synthetic_policy_cannot_satisfy_production_gate():
    assert not is_synthetic_visual_identity_policy("synthetic_stylized_character_v2")
    assert is_synthetic_visual_identity_policy("synthetic_stylized_character_v3")


def test_legacy_visual_tokens_are_isolated_to_explicit_migration_boundaries():
    source_root = Path(__file__).resolve().parents[1] / "src"
    sources = {
        path.relative_to(source_root).as_posix(): path.read_text(encoding="utf-8")
        for path in source_root.rglob("*.py")
    }

    legacy_boolean_allowed = {
        "graph/migrations.py",
        "phases/pipeline_core.py",
        "pipeline_runner.py",
        "runtime/legacy_visual_policy_migration.py",
        "runtime/run_manifest.py",
    }
    for relative, content in sources.items():
        if "no_real_person" in content or "HONCUT_NO_REAL_PERSON" in content:
            assert relative in legacy_boolean_allowed

    guide_v1_allowed = {
        "phases/phase2/shot_storyboards.py",
        "schemas/continuity.py",
    }
    for relative, content in sources.items():
        if "honcut.storyboard-narrative-guide.v1" in content:
            assert relative in guide_v1_allowed

    assert all(
        "synthetic_stylized_character_v2" not in content
        for content in sources.values()
    )
    assert all("variant_*" not in content for content in sources.values())


def test_derive_asset_compatibility_symbol_has_no_production_consumers():
    source_root = Path(__file__).resolve().parents[1] / "src"
    matches = []
    for path in source_root.rglob("*.py"):
        if "detect_derive_assets" in path.read_text(encoding="utf-8"):
            matches.append(path.relative_to(source_root).as_posix())
    assert sorted(matches) == [
        "phases/phase3/phase3_character.py",
        "phases/pipeline_core.py",
    ]

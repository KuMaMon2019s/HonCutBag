"""Character-library contracts: exact reuse, approval, and fail-closed storage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from phases.phase3.phase3_character import run_phase3
from quality.character_reference_qa import (
    PROP_DETAIL_QA_SCHEMA,
    build_character_reference_qa_receipt,
    build_identity_detail_input_contract,
    file_sha256,
    resolve_identity_detail_logical_items,
)
from runtime.character_registry import (
    CHARACTER_REGISTRY_SCHEMA_VERSION,
    CharacterRegistry,
    CharacterRegistryConflictError,
    CharacterRegistryCorruptionError,
    CharacterRegistryError,
    character_spec_fingerprint,
)
from tools.character_reference_board import ensure_character_reference_board
from utils.character_body_contracts import character_reference_identity_description

VIEWS = ("face_closeup", "full_body", "side", "back")


def _write_png(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.effect_noise((512, 512), 80 + offset).convert("RGB")
    image.save(path, format="PNG")
    assert path.stat().st_size > 10_240


def _character() -> dict:
    return {
        "id": "agent",
        "name": "特工",
        "description": "约30岁的虚构男性特工，黑色短发和清晰面部轮廓",
        "appearance": {
            "summary": "偏瘦但结实",
            "clothing": "黑色长风衣和深色高领衣",
            "identity_props": [],
            "variants": [],
        },
        "style": "cinematic photorealism",
        "negative": "identity drift",
    }


def _phase3_spec(character: dict) -> dict:
    return {
        "id": character["id"],
        "entity_id": character.get("entity_id"),
        "instance_id": character.get("instance_id"),
        "instance_ordinal": character.get("instance_ordinal"),
        "canonical_visual_contract": character.get(
            "canonical_visual_contract"
        ),
        "name": character["name"],
        "description": character_reference_identity_description(character),
        "appearance": character["appearance"],
        "style": character.get("style", ""),
        "negative": character.get("negative", ""),
        "visual_identity_policy": character.get("visual_identity_policy"),
    }


def _write_approved_pack(output_dir: Path, character: dict) -> Path:
    char_id = character["id"]
    char_dir = output_dir / "characters" / char_id
    char_dir.mkdir(parents=True, exist_ok=True)
    view_paths = {}
    for index, view in enumerate(VIEWS, start=1):
        path = char_dir / f"{view}.png"
        _write_png(path, index)
        view_paths[view] = path

    attempt = {
        "attempt": 1,
        "attempt_kind": "semantic_review",
        "passed": True,
        "views": {view: {"passed": True} for view in VIEWS},
        "cross_view": {
            "passed": True,
            "identity_consistent": True,
            "outfit_consistent": True,
            "body_proportions_consistent": True,
            "issues": [],
        },
        "failed_views": [],
        "summary": "all canonical views passed",
    }
    generation_contract = {
        "schema": "honcut.character-reference-generation.v1",
        "reference_contract_version": 7,
        "model": "doubao-seedream-5.0-lite",
        "prompt_sha256": {view: "a" * 64 for view in VIEWS},
        "synthetic_styling_sha256": None,
    }
    qa_receipt = build_character_reference_qa_receipt(
        char_id=char_id,
        view_paths=view_paths,
        attempts=[attempt],
        generation_contract=generation_contract,
    )
    (char_dir / "character_reference_qa.json").write_text(
        json.dumps(qa_receipt, ensure_ascii=False), encoding="utf-8"
    )
    ensure_character_reference_board(char_dir, character_id=char_id)
    (char_dir / "angle_map.json").write_text(
        json.dumps({"character_id": char_id, "mappings": {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    card = {
        "id": char_id,
        "name": character["name"],
        "description": character["description"],
        "reference_contract_version": 7,
        "reference_generation_contract": generation_contract,
        "synthetic_styling": None,
        "reference_images": {view: f"characters/{char_id}/{view}.png" for view in VIEWS},
        "reference_qa_report": (f"characters/{char_id}/character_reference_qa.json"),
        "reference_board": f"characters/{char_id}/reference_board.png",
        "reference_board_receipt": f"characters/{char_id}/reference_board.json",
        "identity_props": [],
        "prop_detail_board": None,
        "prop_detail_board_qa_report": None,
    }
    (char_dir / "character_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return char_dir


def _write_run_manifest(
    output_dir: Path,
    *,
    project_id: str,
    library_dir: Path,
    run_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(
            {
                "run_fingerprint": run_id,
                "resolved_config": {
                    "project_id": project_id,
                    "character_library_dir": str(library_dir.resolve()),
                },
            }
        ),
        encoding="utf-8",
    )


def test_registry_promotes_only_a_grade_and_reuses_exact_pack(tmp_path):
    character = _character()
    spec = _phase3_spec(character)
    source = tmp_path / "run-a"
    _write_approved_pack(source, spec)
    registry = CharacterRegistry(tmp_path / "library", project_id="series-a")

    with pytest.raises(CharacterRegistryError, match="grade A"):
        registry.promote_from_run(source, spec, quality_grade="B", source_run_id="run-a")

    approved = registry.promote_from_run(
        source,
        spec,
        quality_grade="A",
        source_run_id="run-a",
    )
    assert approved.status == "canonical_approved"
    assert approved.spec_fingerprint == character_spec_fingerprint(spec)
    assert registry.find_exact(spec).version_id == approved.version_id

    destination = tmp_path / "run-b"
    imported = registry.import_into_run(approved, destination)
    assert imported == destination / "characters" / "agent"
    assert (imported / "reference_board.png").is_file()
    assert (imported / "character_reference_qa.json").is_file()


def test_registry_accepts_current_prop_detail_qa_contract(
    tmp_path,
    canonical_run_contract,
):
    character = _character()
    character["appearance"]["identity_props"] = [{
        "id": "device_a",
        "name": "device",
        "description": "one black rectangular device with a silver lens",
        "attachment_mode": "isolated_handheld",
        "persistence": "role_active",
        "reference_required": True,
    }]
    spec = _phase3_spec(character)
    source = tmp_path / "source"
    canonical_run_contract(source, {"characters": [character]})
    char_dir = _write_approved_pack(source, spec)
    detail_path = char_dir / "prop_detail_board.png"
    _write_png(detail_path, 9)
    canonical_hash, logical_items = resolve_identity_detail_logical_items(
        source,
        character["id"],
        character["appearance"]["identity_props"],
    )
    canonical_paths = [
        char_dir / "face_closeup.png",
        char_dir / "full_body.png",
    ]
    input_contract = build_identity_detail_input_contract(
        char_id=character["id"],
        character_description=spec["description"],
        identity_props=character["appearance"]["identity_props"],
        canonical_contract_sha256=canonical_hash,
        logical_items=logical_items,
        prompt_sha256="b" * 64,
        canonical_paths=canonical_paths,
    )
    receipt_path = char_dir / "prop_detail_board_qa_v2.json"
    receipt_path.write_text(json.dumps({
        "schema": PROP_DETAIL_QA_SCHEMA,
        "status": "passed",
        "qa_verdict": "pass",
        "character_id": character["id"],
        "input_contract": input_contract,
        "inputs": {
            "canonical_references": input_contract["canonical_references"],
            "prop_detail_board": {
                "path": detail_path.name,
                "sha256": file_sha256(detail_path),
                "media_role": "identity_prop_geometry_reference",
            },
        },
    }), encoding="utf-8")
    card_path = char_dir / "character_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card.update({
        "identity_props": character["appearance"]["identity_props"],
        "prop_detail_board": f"characters/{character['id']}/{detail_path.name}",
        "prop_detail_board_qa_report": (
            f"characters/{character['id']}/{receipt_path.name}"
        ),
    })
    card_path.write_text(json.dumps(card), encoding="utf-8")
    registry = CharacterRegistry(tmp_path / "library", project_id="series-a")

    approved = registry.promote_from_run(
        source,
        spec,
        quality_grade="A",
        source_run_id="run-a",
    )

    assert {asset.role for asset in approved.assets} >= {
        "prop_detail_board",
        "prop_detail_board_qa",
    }


def test_registry_exact_lookup_is_project_scoped_and_spec_exact(tmp_path):
    character = _character()
    spec = _phase3_spec(character)
    source = tmp_path / "source"
    _write_approved_pack(source, spec)
    library = tmp_path / "library"
    registry = CharacterRegistry(library, project_id="series-a")
    registry.promote_from_run(source, spec, quality_grade="A", source_run_id="run-a")

    changed = {**spec, "style": "ink wash animation"}
    assert registry.find_exact(changed) is None
    assert CharacterRegistry(library, project_id="series-b").find_exact(spec) is None


def test_registry_fingerprint_separates_instances_and_canonical_lineage():
    base = _phase3_spec(_character())
    base.update({
        "id": "guards_I01",
        "entity_id": "guards",
        "instance_id": "guards_I01",
        "instance_ordinal": 1,
        "canonical_visual_contract": {"contract_sha256": "a" * 64},
    })
    second_instance = {
        **base,
        "id": "guards_I02",
        "instance_id": "guards_I02",
        "instance_ordinal": 2,
    }
    changed_contract = {
        **base,
        "canonical_visual_contract": {"contract_sha256": "b" * 64},
    }

    assert character_spec_fingerprint(base) != character_spec_fingerprint(
        second_instance
    )
    assert character_spec_fingerprint(base) != character_spec_fingerprint(
        changed_contract
    )


def test_registry_refuses_two_pixel_versions_for_one_exact_spec(tmp_path):
    spec = _phase3_spec(_character())
    first_source = tmp_path / "source-a"
    second_source = tmp_path / "source-b"
    _write_approved_pack(first_source, spec)
    _write_approved_pack(second_source, spec)
    registry = CharacterRegistry(tmp_path / "library", project_id="series-a")
    registry.promote_from_run(first_source, spec, quality_grade="A", source_run_id="run-a")

    with pytest.raises(CharacterRegistryConflictError, match="different approved pixels"):
        registry.promote_from_run(second_source, spec, quality_grade="A", source_run_id="run-b")


def test_registry_fails_closed_when_an_approved_asset_is_tampered(tmp_path):
    spec = _phase3_spec(_character())
    source = tmp_path / "source"
    _write_approved_pack(source, spec)
    registry = CharacterRegistry(tmp_path / "library", project_id="series-a")
    approved = registry.promote_from_run(source, spec, quality_grade="A", source_run_id="run-a")
    face = next(asset for asset in approved.assets if asset.role == "face_closeup")
    (registry.root / face.relative_path).write_bytes(b"tampered")

    with pytest.raises(CharacterRegistryCorruptionError, match="hash mismatch"):
        registry.find_exact(spec)


def test_registry_rejects_unknown_future_schema(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    connection = sqlite3.connect(library / "characters.db")
    connection.execute(f"PRAGMA user_version = {CHARACTER_REGISTRY_SCHEMA_VERSION + 1}")
    connection.close()

    with pytest.raises(CharacterRegistryCorruptionError, match="future schema"):
        CharacterRegistry(library, project_id="series-a")
    assert not (library / "assets").exists()


def test_registry_recovers_a_fully_written_package_after_db_commit_loss(tmp_path):
    spec = _phase3_spec(_character())
    source = tmp_path / "source"
    _write_approved_pack(source, spec)
    registry = CharacterRegistry(tmp_path / "library", project_id="series-a")
    first = registry.promote_from_run(source, spec, quality_grade="A", source_run_id="run-a")
    with sqlite3.connect(registry.db_path) as connection:
        connection.execute("DELETE FROM character_assets WHERE version_id = ?", (first.version_id,))
        connection.execute(
            "DELETE FROM character_versions WHERE version_id = ?", (first.version_id,)
        )

    recovered = registry.promote_from_run(source, spec, quality_grade="A", source_run_id="run-b")

    assert recovered.version_id == first.version_id
    assert recovered.source_run_id == "run-a"
    assert registry.find_exact(spec).version_id == first.version_id


def test_registry_rejects_db_metadata_that_disagrees_with_approval(tmp_path):
    spec = _phase3_spec(_character())
    source = tmp_path / "source"
    _write_approved_pack(source, spec)
    registry = CharacterRegistry(tmp_path / "library", project_id="series-a")
    approved = registry.promote_from_run(source, spec, quality_grade="A", source_run_id="run-a")
    with sqlite3.connect(registry.db_path) as connection:
        connection.execute(
            "UPDATE character_versions SET source_run_id = ? WHERE version_id = ?",
            ("tampered-run", approved.version_id),
        )

    with pytest.raises(CharacterRegistryCorruptionError, match="does not match"):
        registry.find_exact(spec)


def test_registry_ignores_legacy_state_metadata_and_rejects_unsafe_character_ids(tmp_path):
    spec = _phase3_spec(_character())
    source = tmp_path / "source"
    _write_approved_pack(source, spec)
    registry = CharacterRegistry(tmp_path / "library", project_id="series-a")
    variant = json.loads(json.dumps(spec))
    variant["appearance"]["variants"] = [{"id": "battle-damaged"}]

    assert registry.find_exact(variant) is None
    approved = registry.promote_from_run(
        source, variant, quality_grade="A", source_run_id="run-a"
    )
    assert registry.find_exact(variant).version_id == approved.version_id
    with pytest.raises(CharacterRegistryError, match="safe path component"):
        character_spec_fingerprint({**spec, "id": "../escape"})


def test_phase3_exact_reuse_makes_zero_generation_requests(
    tmp_path, monkeypatch, canonical_run_contract
):
    character = _character()
    library = tmp_path / "library"
    output = tmp_path / "new-run"
    _write_run_manifest(
        output,
        project_id="series-a",
        library_dir=library,
        run_id="new-run",
    )
    projected, _contract = canonical_run_contract(
        output,
        {"characters": [character]},
    )
    spec = _phase3_spec(projected["characters"][0])
    source = tmp_path / "source"
    _write_approved_pack(source, spec)
    CharacterRegistry(library, project_id="series-a").promote_from_run(
        source, spec, quality_grade="A", source_run_id="source-run"
    )
    monkeypatch.setattr(
        "phases.phase3.character_factory.batch_generate",
        lambda *_args, **_kwargs: pytest.fail("exact reuse must not call Seedream"),
    )
    result = run_phase3(output, projected, dry_run=False)

    assert result["status"] == "done"
    assert result["character_registry"]["reused"] == 1
    assert result["character_registry"]["generated"] == 0
    receipt = json.loads((output / "character_registry_receipt.json").read_text(encoding="utf-8"))
    assert receipt["registry_provider_requests"] == 0
    assert receipt["characters"][0]["action"] == "reused"
    assert hashlib.sha256(
        (output / "characters/agent/reference_board.png").read_bytes()
    ).hexdigest() == next(
        asset.sha256
        for asset in CharacterRegistry(library, project_id="series-a").find_exact(spec).assets
        if asset.role == "reference_board"
    )

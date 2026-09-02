from __future__ import annotations

import copy
import hashlib
import inspect
import json

import pytest

from phases.phase1.character_roster import (
    CHARACTER_ROSTER_FILENAME,
    CharacterRosterError,
    compile_character_roster,
    migrate_character_roster_v1,
    persist_character_roster,
    reconcile_character_observations,
    validate_character_roster,
)
from phases.phase1 import character_discoverer
from phases.phase1 import phase1_screenwriter
from tools.asset_packager import _resolve_char_ids
from utils.semantic_contracts import bind_story_semantics


def _event(
    event_id: int,
    who: list[str],
    source_excerpt: str,
    *,
    sequence_id: str = "SEQ001",
    action_unit_id: str = "",
    continuity_before: str = "continuous",
) -> dict:
    return {
        "id": event_id,
        "sequence_id": sequence_id,
        "action_unit_id": action_unit_id,
        "who": who,
        "source_excerpt": source_excerpt,
        "what": source_excerpt,
        "continuity_before": continuity_before,
    }


def _observation(name: str, character_id: str, aliases: list[str] | None = None) -> dict:
    return {
        "id": character_id,
        "name": name,
        "aliases": aliases or [],
        "role": "antagonist" if "敌" in name else "protagonist",
        "appearance": {
            "gender": "unknown",
            "age_range": "adult",
            "height": "average",
            "build": "athletic",
            "hair": "short dark hair",
            "face": "fictional balanced facial geometry",
            "clothing": "dark practical clothing",
            "interaction_props": [],
            "identity_props": [],
            "distinguishing": "",
            "summary": "stable fictional adult character",
            "variants": [],
        },
        "personality": {"traits": [], "speech_style": "", "motivation": ""},
        "style": "cinematic realism",
        "negative": "",
        "size": "2K",
        "first_appearance": 1,
        "appearance_count": 1,
        "relationships": [],
    }


def test_roster_compiles_one_group_entity_with_three_stable_instances():
    events = [
        _event(1, ["男子"], "年轻男子站在车门前。"),
        _event(
            2,
            ["男子"],
            "三名未来战斗人员突然出现，身穿相同黑色装甲。",
            action_unit_id="AU001",
        ),
        _event(3, ["男子", "第一名敌人"], "第一名敌人挥砍。", action_unit_id="AU002"),
        _event(4, ["男子", "第二名敌人"], "第二名敌人突袭。", action_unit_id="AU003"),
        _event(5, ["男子", "第三名敌人"], "第三名敌人跃下。", action_unit_id="AU004"),
    ]

    first = compile_character_roster(events)
    second = compile_character_roster(copy.deepcopy(events))

    assert first == second
    assert first["schema"] == "honcut.character-roster.v2"
    assert len(first["entities"]) == 2
    group = next(entity for entity in first["entities"] if entity["instance_count"] == 3)
    assert group["display_name"] == "未来战斗人员"
    assert group["reconciliation_origin"] == "deterministic_group_completion"
    assert [instance["ordinal"] for instance in group["instances"]] == [1, 2, 3]
    assert [instance["source_mentions"] for instance in group["instances"]] == [
        ["第一名敌人"],
        ["第二名敌人"],
        ["第三名敌人"],
    ]
    assert len({instance["instance_id"] for instance in group["instances"]}) == 3
    assert validate_character_roster(first) == first


def test_roster_reconciles_chinese_and_arabic_ordinal_notation_within_one_group():
    events = [
        _event(1, [], "三名巡查人员同时进入。", continuity_before="cut"),
        _event(2, ["队员1", "队员2", "队员3"], "三名巡查人员分散警戒。"),
        _event(3, ["第一名队员"], "第一名队员检查入口。"),
        _event(4, ["第二名队员"], "第二名队员检查通道。"),
        _event(5, ["第三名队员"], "第三名队员检查出口。"),
    ]

    first = compile_character_roster(events)
    second = compile_character_roster(copy.deepcopy(events))

    assert first == second
    assert len(first["entities"]) == 1
    group = first["entities"][0]
    assert group["instance_count"] == 3
    assert [instance["source_mentions"] for instance in group["instances"]] == [
        ["队员1", "第一名队员"],
        ["队员2", "第二名队员"],
        ["队员3", "第三名队员"],
    ]
    assert [
        reconciliation["evidence_kind"]
        for instance in group["instances"]
        for reconciliation in instance["identity_reconciliations"]
    ] == ["ordinal_notation_equivalence"] * 3


def test_roster_anonymized_mixed_ordinal_fixture_has_stable_cardinality():
    events = [
        _event(1, ["研究员"], "研究员进入大厅。", continuity_before="cut"),
        _event(2, ["研究员", "队员1", "队员2", "队员3"], "三名巡查人员同时进入。"),
        _event(3, ["研究员", "第一名队员"], "第一名队员检查入口。"),
        _event(4, ["研究员", "第二名队员"], "第二名队员检查通道。"),
        _event(5, ["研究员", "第三名队员"], "第三名队员检查出口。"),
    ]

    roster = compile_character_roster(events)

    assert len(roster["entities"]) == 2
    assert sum(entity["instance_count"] for entity in roster["entities"]) == 4
    group = next(entity for entity in roster["entities"] if entity["instance_count"] == 3)
    assert [instance["source_mentions"] for instance in group["instances"]] == [
        ["队员1", "第一名队员"],
        ["队员2", "第二名队员"],
        ["队员3", "第三名队员"],
    ]


def test_roster_reconciles_chinese_and_prefixed_arabic_ordinals():
    roster = compile_character_roster([
        _event(1, [], "两名巡查人员进入。", continuity_before="cut"),
        _event(2, ["第1名队员"], "第1名队员检查入口。"),
        _event(3, ["第一名队员"], "第一名队员继续值守。"),
        _event(4, ["第2名队员"], "第2名队员检查出口。"),
        _event(5, ["第二名队员"], "第二名队员继续值守。"),
    ])

    assert len(roster["entities"]) == 1
    assert roster["entities"][0]["instance_count"] == 2
    assert all(
        instance["identity_reconciliations"][0]["evidence_kind"]
        == "ordinal_notation_equivalence"
        for instance in roster["entities"][0]["instances"]
    )


def test_roster_accepts_complete_arabic_suffix_ordinals_for_a_counted_group():
    roster = compile_character_roster([
        _event(1, [], "三名巡查人员同时进入。", continuity_before="cut"),
        _event(2, ["队员1", "队员2", "队员3"], "三名巡查人员分散警戒。"),
    ])

    assert len(roster["entities"]) == 1
    assert roster["entities"][0]["instance_count"] == 3
    assert [instance["source_mentions"] for instance in roster["entities"][0]["instances"]] == [
        ["队员1"],
        ["队员2"],
        ["队员3"],
    ]


def test_roster_normalizes_fullwidth_arabic_suffix_ordinals():
    roster = compile_character_roster([
        _event(1, [], "两名巡查人员同时进入。", continuity_before="cut"),
        _event(2, ["队员１", "队员２"], "两名巡查人员分散警戒。"),
        _event(3, ["第一名队员"], "第一名队员检查入口。"),
        _event(4, ["第二名队员"], "第二名队员检查出口。"),
    ])

    assert len(roster["entities"]) == 1
    assert roster["entities"][0]["instance_count"] == 2


@pytest.mark.parametrize(
    ("events", "error"),
    [
        (
            [
                _event(1, [], "三名巡查人员进入，三名护卫随后进入。"),
                _event(2, ["队员1", "队员2", "队员3"], "三名人员分散警戒。"),
            ],
            "ambiguous",
        ),
        (
            [
                _event(1, [], "三名巡查人员进入。"),
                _event(2, ["队员1", "队员2"], "两名队员分散警戒。"),
            ],
            "count",
        ),
        (
            [
                _event(1, [], "三名巡查人员进入。"),
                _event(2, ["队员1", "第一名队员"], "队员1与第一名队员同时警戒。"),
                _event(3, ["队员2"], "队员2检查通道。"),
                _event(4, ["队员3"], "队员3检查出口。"),
            ],
            "co-occur",
        ),
    ],
)
def test_roster_rejects_unsafe_arabic_suffix_ordinal_merges(events, error):
    with pytest.raises(CharacterRosterError, match=error):
        compile_character_roster(events)


def test_roster_does_not_retroactively_merge_a_numbered_identity_before_group_declaration():
    roster = compile_character_roster([
        _event(1, ["队员1"], "另一名队员1单独值守。", continuity_before="cut"),
        _event(2, [], "三名巡查人员随后进入。", continuity_before="cut"),
        _event(3, ["第一名队员"], "第一名队员检查入口。"),
        _event(4, ["第二名队员"], "第二名队员检查通道。"),
        _event(5, ["第三名队员"], "第三名队员检查出口。"),
    ])

    assert sorted(entity["instance_count"] for entity in roster["entities"]) == [1, 3]


def test_roster_excludes_non_character_action_subjects_without_erasing_events():
    events = [
        _event(1, ["磁悬浮列车"], "磁悬浮列车缓缓驶入站台。"),
        _event(2, ["年轻男性"], "年轻男性站在车门前。"),
    ]

    roster = compile_character_roster(events)

    assert [entity["display_name"] for entity in roster["entities"]] == [
        "年轻男性"
    ]
    assert events[0]["who"] == ["磁悬浮列车"]
    characters, _diagnostics = reconcile_character_observations(
        [], roster, semantic_qa_enabled=False
    )
    ledger = bind_story_semantics(events, characters)
    assert ledger["events"][0]["character_ids"] == []
    assert events[0]["non_character_participants"] == ["磁悬浮列车"]


def test_roster_classifies_english_vehicle_tokens_without_substring_collisions():
    roster = compile_character_roster([
        _event(1, ["future maglev train"], "A future maglev train enters."),
        _event(2, ["Oscar"], "Oscar waits on the platform."),
    ])

    assert [entity["display_name"] for entity in roster["entities"]] == ["Oscar"]


def test_roster_allows_an_explicitly_named_nonhuman_character():
    roster = compile_character_roster([
        _event(1, ["列车"], "代号“列车”的合成人走进站台。"),
    ])

    assert [entity["display_name"] for entity in roster["entities"]] == ["列车"]


def test_roster_reconciles_unique_post_declaration_group_generic_alias():
    events = [
        _event(1, ["年轻男性"], "年轻男性站在车门前。", continuity_before="cut"),
        _event(2, ["年轻男性"], "三名未来战斗人员突然出现。"),
        _event(3, ["年轻男性", "第一名敌人"], "第一名敌人发动攻击。"),
        _event(4, ["年轻男性", "第二名敌人"], "第二名敌人从侧面突袭。"),
        _event(5, ["年轻男性", "第三名敌人"], "第三名敌人从顶部跃下。"),
        _event(6, ["敌人", "年轻男性"], "敌人释放电磁冲击，年轻男性举起芯片。"),
        _event(7, ["敌人", "年轻男性"], "年轻男性反击并控制敌人的手臂。"),
    ]

    roster = compile_character_roster(events)

    assert len(roster["entities"]) == 2
    assert sum(entity["instance_count"] for entity in roster["entities"]) == 4
    group = next(entity for entity in roster["entities"] if entity["instance_count"] == 3)
    assert all(
        "敌人" in instance["source_mentions"]
        for instance in group["instances"]
    )
    assert {
        ref
        for instance in group["instances"]
        for ref in instance["event_refs"]
    } >= {"event:6", "event:7"}

    characters, _diagnostics = reconcile_character_observations(
        [], roster, semantic_qa_enabled=False
    )
    ledger = bind_story_semantics(events, characters)
    group_entity = next(
        entity for entity in ledger["entities"] if len(entity["instance_ids"]) == 3
    )
    assert events[5]["character_instance_ids"][:3] == group_entity["instance_ids"]


def test_roster_reconciles_english_group_generic_alias_deterministically():
    events = [
        _event(1, [], "Three fighters enter the compartment."),
        _event(2, ["first enemy"], "The first enemy attacks."),
        _event(3, ["second enemy"], "The second enemy blocks the exit."),
        _event(4, ["third enemy"], "The third enemy drops from above."),
        _event(5, ["enemy"], "The enemy releases an electromagnetic pulse."),
    ]

    roster = compile_character_roster(events)

    assert len(roster["entities"]) == 1
    assert roster["entities"][0]["instance_count"] == 3
    assert all(
        "enemy" in instance["source_mentions"]
        for instance in roster["entities"][0]["instances"]
    )


@pytest.mark.parametrize("events", [
    [
        _event(1, [], "三名守卫进入，三名佣兵随后出现。"),
        _event(2, ["第一名敌人"], "第一名敌人攻击。"),
        _event(3, ["第二名敌人"], "第二名敌人攻击。"),
        _event(4, ["第三名敌人"], "第三名敌人攻击。"),
        _event(5, ["敌人"], "敌人继续逼近。"),
    ],
    [
        _event(1, [], "三名战斗人员进入。"),
        _event(2, ["第一名敌人"], "第一名敌人警戒。"),
        _event(3, ["第二名敌人"], "第二名敌人警戒。"),
        _event(4, ["第三名敌人"], "第三名敌人警戒。"),
        _event(5, ["敌人", "第一名敌人"], "敌人与第一名敌人同时出现。"),
    ],
])
def test_roster_fails_closed_on_ambiguous_group_generic_aliases(events):
    with pytest.raises(CharacterRosterError):
        compile_character_roster(events)


def test_roster_keeps_explicit_prior_individual_separate_from_later_group():
    roster = compile_character_roster([
        _event(1, ["守卫"], "另一名守卫进入。", continuity_before="cut"),
        _event(2, [], "三名守卫随后出现。"),
        _event(3, ["第一名守卫"], "第一名守卫警戒。"),
        _event(4, ["第二名守卫"], "第二名守卫警戒。"),
        _event(5, ["第三名守卫"], "第三名守卫警戒。"),
    ])

    assert sorted(entity["instance_count"] for entity in roster["entities"]) == [1, 3]


def test_roster_reconciles_one_source_proven_qualified_human_alias():
    events = [
        _event(1, ["年轻男性"], "年轻男性站在入口。", continuity_before="cut"),
        _event(2, ["年轻男性"], "三名战斗人员同时出现。"),
        _event(3, ["年轻男性", "第一名敌人"], "第一名敌人向年轻男性逼近。"),
        _event(4, ["年轻男性", "第二名敌人"], "第二名敌人突袭，男子立即格挡。"),
        _event(5, ["男子", "第三名敌人"], "第三名敌人跃下，男子连续闪避。"),
    ]

    roster = compile_character_roster(events)

    assert len(roster["entities"]) == 2
    assert sum(item["instance_count"] for item in roster["entities"]) == 4
    lead = next(item for item in roster["entities"] if item["instance_count"] == 1)
    assert lead["instances"][0]["source_mentions"] == ["年轻男性", "男子"]
    assert lead["instances"][0]["identity_reconciliations"] == [
        {
            "canonical_mention": "年轻男性",
            "source_mention": "男子",
            "sequence_id": "SEQ001",
            "event_refs": ["event:4", "event:5"],
            "evidence_kind": "continuous_source_cross_reference",
            "controlled_gender": "male",
            "evidence_sha256": lead["instances"][0]["identity_reconciliations"][0][
                "evidence_sha256"
            ],
        }
    ]


@pytest.mark.parametrize(
    ("events", "expected_entities"),
    [
        (
            [
                _event(1, ["年轻男性", "男子"], "年轻男性与男子同时出现。"),
            ],
            2,
        ),
        (
            [
                _event(1, ["年轻男性"], "年轻男性进入。", continuity_before="cut"),
                _event(2, ["高个男性"], "高个男性进入。"),
            ],
            2,
        ),
        (
            [
                _event(1, ["年轻男性"], "年轻男性进入。", continuity_before="cut"),
                _event(2, ["男子"], "另一名男子进入。", continuity_before="cut"),
            ],
            2,
        ),
    ],
)
def test_roster_does_not_guess_ambiguous_human_aliases(events, expected_entities):
    roster = compile_character_roster(events)
    assert len(roster["entities"]) == expected_entities


def test_roster_keeps_ordinal_people_independent_without_group_evidence():
    roster = compile_character_roster([
        _event(1, ["第一名守卫"], "第一名守卫进入。"),
        _event(2, ["第二名守卫"], "第二名守卫进入。"),
    ])

    assert len(roster["entities"]) == 2
    assert [entity["instance_count"] for entity in roster["entities"]] == [1, 1]
    assert all(
        entity["reconciliation_origin"] == "explicit_source"
        for entity in roster["entities"]
    )


def test_roster_rejects_group_count_conflict():
    with pytest.raises(CharacterRosterError, match="count"):
        compile_character_roster([
            _event(1, [], "三名守卫进入。"),
            _event(2, ["第一名守卫"], "第一名守卫警戒。"),
            _event(3, ["第二名守卫"], "第二名守卫警戒。"),
        ])


def test_roster_rejects_competing_groups_for_the_same_numbered_instances():
    with pytest.raises(CharacterRosterError, match="ambiguous"):
        compile_character_roster([
            _event(1, [], "三名守卫进入，三名佣兵随后出现。"),
            _event(2, ["第一名敌人"], "第一名敌人攻击。"),
            _event(3, ["第二名敌人"], "第二名敌人攻击。"),
            _event(4, ["第三名敌人"], "第三名敌人攻击。"),
        ])


def test_roster_allows_zero_characters_without_inventing_one():
    roster = compile_character_roster([
        _event(1, [], "暴雨落在空站台。"),
    ])

    assert roster["entities"] == []
    assert validate_character_roster(roster) == roster


def test_roster_hash_and_future_schema_fail_closed():
    roster = compile_character_roster([
        _event(1, ["Mira"], "Mira enters the station."),
    ])
    tampered = copy.deepcopy(roster)
    tampered["entities"][0]["display_name"] = "Other"
    with pytest.raises(CharacterRosterError, match="hash mismatch"):
        validate_character_roster(tampered)

    future = copy.deepcopy(roster)
    future["schema"] = "honcut.character-roster.v99"
    with pytest.raises(CharacterRosterError):
        validate_character_roster(future)


def test_roster_is_atomically_persisted_and_round_trips(tmp_path):
    roster = compile_character_roster([
        _event(1, ["Mira"], "Mira enters."),
    ])
    path = tmp_path / CHARACTER_ROSTER_FILENAME

    persisted = persist_character_roster(path, roster)

    assert persisted == roster
    assert json.loads(path.read_text(encoding="utf-8")) == roster
    assert not path.with_suffix(".json.tmp").exists()


def _legacy_v1(roster: dict) -> dict:
    legacy = copy.deepcopy(roster)
    legacy["schema"] = "honcut.character-roster.v1"
    for entity in legacy["entities"]:
        for instance in entity["instances"]:
            instance.pop("identity_reconciliations", None)
    legacy.pop("roster_sha256", None)
    legacy["roster_sha256"] = hashlib.sha256(
        json.dumps(
            legacy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return legacy


def test_roster_v1_migrates_from_hash_verified_original_events(tmp_path):
    events = [_event(1, ["Mira"], "Mira enters.", continuity_before="cut")]
    legacy = _legacy_v1(compile_character_roster(events))
    receipt_path = tmp_path / "CHARACTER_ROSTER_MIGRATION.json"

    migrated, receipt = migrate_character_roster_v1(
        legacy,
        events,
        receipt_path=receipt_path,
    )

    assert migrated["schema"] == "honcut.character-roster.v2"
    assert receipt["downstream_reuse_allowed"] is True
    assert receipt["legacy_artifact_preserved"] is True
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_roster_v1_migration_quarantines_changed_identity_ids():
    events = [_event(1, ["Mira"], "Mira enters.", continuity_before="cut")]
    legacy = _legacy_v1(compile_character_roster(events))
    legacy["entities"][0]["entity_id"] = "legacy_polluted_entity"
    legacy["entities"][0]["instances"][0]["instance_id"] = "legacy_polluted_instance"
    legacy.pop("roster_sha256")
    legacy["roster_sha256"] = hashlib.sha256(
        json.dumps(
            legacy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    _migrated, receipt = migrate_character_roster_v1(legacy, events)

    assert receipt["downstream_reuse_allowed"] is False


def test_roster_v1_migration_rejects_missing_source_lineage():
    events = [_event(1, ["Mira"], "Mira enters.", continuity_before="cut")]
    legacy = _legacy_v1(compile_character_roster(events))

    with pytest.raises(CharacterRosterError, match="source lineage"):
        migrate_character_roster_v1(legacy, [_event(2, ["Mira"], "Mira leaves.")])


def test_character_observation_requires_the_current_roster_hash():
    response = json.dumps({
        "schema": "honcut.character-roster-observation.v1",
        "roster_sha256": "a" * 64,
        "characters": [_observation("Mira", "Mira")],
    })

    assert character_discoverer._parse_characters(
        response,
        expected_roster_sha256="a" * 64,
    )[0]["name"] == "Mira"
    with pytest.raises(ValueError, match="hash mismatch"):
        character_discoverer._parse_characters(
            response,
            expected_roster_sha256="b" * 64,
        )


def test_discoverer_persists_roster_before_the_single_observation_call(
    tmp_path,
    monkeypatch,
):
    roster_path = tmp_path / CHARACTER_ROSTER_FILENAME
    calls = 0

    def fake_call(prompt):
        nonlocal calls
        calls += 1
        persisted = validate_character_roster(
            json.loads(roster_path.read_text(encoding="utf-8"))
        )
        return json.dumps({
            "schema": "honcut.character-roster-observation.v1",
            "roster_sha256": persisted["roster_sha256"],
            "characters": [_observation("Mira", persisted["entities"][0]["entity_id"])],
        })

    monkeypatch.setattr(character_discoverer, "_call_llm", fake_call)
    result = character_discoverer.discover_characters(
        [_event(1, ["Mira"], "Mira enters.")],
        roster_output_path=roster_path,
    )

    assert calls == 1
    assert result["character_roster_sha256"] == json.loads(
        roster_path.read_text(encoding="utf-8")
    )["roster_sha256"]


def test_default_reconciliation_repairs_missing_group_without_another_model_call():
    events = [
        _event(1, ["男子"], "年轻男子站在门前。"),
        _event(2, ["男子"], "三名未来战斗人员突然出现。"),
        _event(3, ["男子", "第一名敌人"], "第一名敌人攻击。"),
        _event(4, ["男子", "第二名敌人"], "第二名敌人攻击。"),
        _event(5, ["男子", "第三名敌人"], "第三名敌人攻击。"),
    ]
    roster = compile_character_roster(events)

    characters, diagnostics = reconcile_character_observations(
        [_observation("男子", "model_lead")],
        roster,
        semantic_qa_enabled=False,
    )

    assert len(characters) == 2
    assert sum(character["instance_count"] for character in characters) == 4
    assert all(character["entity_id"] == character["id"] for character in characters)
    group = next(character for character in characters if character["instance_count"] == 3)
    assert group["name"] == "未来战斗人员"
    assert group["appearance"]["summary"]
    assert any(item["code"] == "model_entity_missing" for item in diagnostics)


def test_strict_reconciliation_blocks_the_same_missing_entity():
    events = [
        _event(1, ["男子"], "男子站立。"),
        _event(2, [], "两名守卫进入。"),
    ]
    roster = compile_character_roster(events)

    with pytest.raises(ValueError, match="strict character roster semantic QA"):
        reconcile_character_observations(
            [_observation("男子", "model_lead")],
            roster,
            semantic_qa_enabled=True,
        )


def test_semantic_ledger_v3_binds_each_source_mention_to_one_instance():
    events = [
        _event(1, ["男子"], "男子站立。"),
        _event(2, ["男子"], "三名战斗人员出现。"),
        _event(3, ["第一名敌人"], "第一名敌人攻击。"),
        _event(4, ["第二名敌人"], "第二名敌人攻击。"),
        _event(5, ["第三名敌人"], "第三名敌人攻击。"),
    ]
    roster = compile_character_roster(events)
    characters, _diagnostics = reconcile_character_observations(
        [], roster, semantic_qa_enabled=False
    )

    ledger = bind_story_semantics(events, characters)

    assert ledger["schema"] == "honcut.semantic-understanding.v3"
    assert len(ledger["entities"]) == 2
    assert sorted(len(entity["instance_ids"]) for entity in ledger["entities"]) == [1, 3]
    enemy_instance_ids = [events[index]["character_ids"][0] for index in (2, 3, 4)]
    assert len(set(enemy_instance_ids)) == 3
    assert all(
        mention["character_id"] == mention["instance_id"]
        for mention in ledger["source_mentions"]
    )


def test_phase6_source_group_label_resolves_to_every_instance(tmp_path):
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({
            "characters": [
                {
                    "id": f"guards_I{ordinal:02d}",
                    "entity_id": "guards",
                    "instance_id": f"guards_I{ordinal:02d}",
                    "name": "守卫",
                    "aliases": [],
                }
                for ordinal in range(1, 4)
            ]
        }),
        encoding="utf-8",
    )

    assert _resolve_char_ids(tmp_path, ["守卫"]) == [
        "guards_I01",
        "guards_I02",
        "guards_I03",
    ]
    assert _resolve_char_ids(tmp_path, ["guards"]) == [
        "guards_I01",
        "guards_I02",
        "guards_I03",
    ]


def test_character_discoverer_keeps_cardinality_out_of_the_model_owner():
    prompt = (
        character_discoverer.SYSTEM_PROMPT
        + character_discoverer.USER_PROMPT_TEMPLATE
    )
    source = inspect.getsource(character_discoverer.discover_characters)

    assert "最多保留5个主要角色" not in prompt
    assert "只保留前 5 个" not in inspect.getsource(character_discoverer)
    assert "compile_character_roster" in source
    assert "roster = persist_roster(provisional_roster)" in source
    assert "reconcile_character_observations" in source
    assert reconcile_character_observations.__module__.endswith(
        "phase1.character_roster"
    )
    assert "_is_human_character" not in inspect.getsource(phase1_screenwriter)

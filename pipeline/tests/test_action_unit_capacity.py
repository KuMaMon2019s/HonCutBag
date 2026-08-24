"""Generation action unit capacity model (normalized, ledger-preserving).

Regression for the 26-08-17_03 failure: a 60s one-take flash-mob script with
99 raw micro-actions was rejected because the gate counted every authored
micro-action as paid provider work.  The capacity model now normalizes:

  sequential    ordered plot actions that cannot run in parallel -> 1 unit each
                (deduplicated across events through a shared seen-set)
  simultaneous  concurrent composite motions / group grooves -> merged to 1 unit
  sustained     sustained states and camera constraints -> 0 units
  duplicate     cross-event repeats -> 0 units

The complete micro_actions ledger is never truncated; only capacity math
consumes the normalized units.
"""

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases.phase1 import adaptation_engine as engine  # noqa: E402
from phases.phase1.storyboard_beats import plan_storyboard_beats  # noqa: E402
from phases.phase5.storyboard_qa_gate import (  # noqa: E402
    run_generation_capacity_checks,
    run_l1_checks,
)
from prompt.event_extractor import _annotate_global_event_flow  # noqa: E402
from utils.action_units import (  # noqa: E402
    annotate_event_motion_modes,
    classify_micro_action,
    event_uses_composite_motion,
    normalize_action_units,
    normalize_event_action_units,
    normalized_action_unit_count,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "flashmob_60s_events.json"
EVOLVING_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "flashmob_60s_evolving_events.json"
)

# These snapshots predate the structured extractor field.  Mark their authored
# per-event concurrency here so the regression exercises the current contract
# instead of relying on production keyword rules for one screenplay's nouns.
LEGACY_COMPOSITE_EVENTS = {2, 3, 6, 8, 13, 15, 18}
EVOLVING_COMPOSITE_EVENTS = {2, 3, 6, 8, 13, 14, 15, 16, 19, 20, 22}


def _load_capacity_fixture(path: Path, composite_events: set[int]) -> list[dict]:
    events = json.loads(path.read_text(encoding="utf-8"))
    for event_id, event in enumerate(events, 1):
        if not event.get("micro_actions"):
            event["generation_motion_mode"] = "none"
        else:
            event["generation_motion_mode"] = (
                "composite" if event_id in composite_events else "atomic"
            )
    return events


# ── classifier unit tests ───────────────────────────────────────────────────


def test_classifier_sequential_combat_actions():
    for text in ["挥拳", "肘击", "膝击", "缴械", "抓门框", "稳定", "穿门", "避让工具箱"]:
        assert classify_micro_action(text) == "sequential", text


def test_classifier_simultaneous_coordinated_actions():
    for text in [
        "两名搬运员同步抬起箱体",
        "更多参与者陆续加入",
        "队伍一同向前移动",
        "人数从1人逐渐变成3～5人",
        "多人协同调整道具位置",
        "新加入者同步程度逐渐提高",
    ]:
        assert classify_micro_action(text) == "simultaneous", text


def test_classifier_sustained_camera_and_state():
    for text in [
        "记录者以手持设备跟拍主体",
        "视频开场直接呈现表演状态，无空镜、无静止站立摆拍环节",
        "露出轻松自然的笑容",
        "与镜头保持自信的眼神交流",
        "整段动作一气呵成并向前推进",
        "观察者注意到主体的动作",
        "录制进入收尾阶段",
        "主体不停止当前活动",
    ]:
        assert classify_micro_action(text) == "sustained", text


def test_classifier_keeps_actor_actions_with_incidental_frame_language():
    for text in [
        "角色冲进画面并挥拳",
        "镜头跟随角色转身挥拳",
        "他没有犹豫，挥拳",
    ]:
        assert classify_micro_action(text) == "sequential", text
    assert classify_micro_action("镜头转向角色") == "sustained"


# ── normalization semantics ─────────────────────────────────────────────────


def test_simultaneous_actions_merge_into_one_unit():
    actions = [
        "搬运组同步抬起箱体",
        "多人协同转动箱体",
        "全体一同跨过门槛",
        "队伍同时放下箱体",
    ]
    assert normalized_action_unit_count(actions) == 1


def test_sustained_only_actions_cost_zero_units():
    actions = [
        "露出轻松自然的笑容",
        "与镜头保持自信的眼神交流",
        "记录者以手持设备跟拍主体",
    ]
    assert normalized_action_unit_count(actions) == 0


def test_cross_event_duplicate_sequential_actions_are_deduplicated():
    seen: set[str] = set()
    first = normalized_action_unit_count(["挥拳", "肘击"], seen=seen)
    second = normalized_action_unit_count(["肘击", "膝击"], seen=seen)
    assert first == 2
    assert second == 1  # 肘击 already counted in the first event


def test_cross_event_duplicate_simultaneous_summary_is_deduplicated():
    seen: set[str] = set()
    first = normalize_action_units(["人群同步挥臂。"], seen=seen)
    second = normalize_action_units(["人群 同步挥臂"], seen=seen)
    assert first["units"] == 1
    assert second["units"] == 0
    assert second["categories"] == ["duplicate"]


def test_ledger_is_preserved_with_categories():
    actions = ["挥拳", "路人从斑马线加入", "露出轻松自然的笑容", "挥拳"]
    result = normalize_action_units(actions)
    assert result["ledger"] == actions  # nothing dropped or rewritten
    assert result["categories"][0] == "sequential"
    assert result["categories"][1] == "simultaneous"
    assert result["categories"][2] == "sustained"
    assert result["categories"][3] == "sequential"
    assert result["units"] == 3  # two authored punches + one groove cluster


def test_repeated_action_inside_one_event_remains_distinct():
    result = normalize_action_units(["挥拳", "挥拳"])

    assert result["categories"] == ["sequential", "sequential"]
    assert result["units"] == 2


def test_source_authored_compound_motion_is_one_generation_unit():
    event = {
        "what": "操作员完成一个复合装配动作",
        "source_excerpt": (
            "双手对齐、压合与夹具收紧在同一时刻并行完成，"
            "融为一个整体，而不是逐个执行。"
        ),
        "micro_actions": [
            "左手对齐零件",
            "右手压合接缝",
            "夹具同步收紧",
        ],
    }

    normalized = normalize_event_action_units(event)

    assert event_uses_composite_motion(event) is True
    assert normalized["motion_mode"] == "composite"
    assert normalized["units"] == 1
    assert normalized["ledger"] == event["micro_actions"]


def test_explicit_source_composite_overrides_conflicting_model_atomic_label():
    event = {
        "what": "操作员完成复合锁定",
        "source_excerpt": (
            "同一瞬间复合动作：操作员对齐组件、压合接缝并锁紧夹具，"
            "这些贡献并行完成，不是逐项动作清单。"
        ),
        "micro_actions": ["对齐组件", "压合接缝", "锁紧夹具"],
        "generation_motion_mode": "atomic",
    }

    normalized = normalize_event_action_units(event)

    assert event_uses_composite_motion(event) is True
    assert normalized["motion_mode"] == "composite"
    assert normalized["units"] == 1


def test_composite_source_does_not_treat_last_entity_as_temporal_progression():
    event = {
        "what": "平台转向时完成并发控制动作",
        "source_excerpt": (
            "同一瞬间复合动作：平台转向，操作员贴身控制最后一名检修员的"
            "手臂，借惯性将其推向隔离门。"
        ),
        "micro_actions": [
            "平台转向",
            "贴身控制最后一名检修员的手臂",
            "借惯性将其推向隔离门",
        ],
        "generation_motion_mode": "atomic",
    }

    normalized = normalize_event_action_units(event)

    assert event_uses_composite_motion(event) is True
    assert normalized["motion_mode"] == "composite"
    assert normalized["units"] == 1


def test_sequential_fight_is_not_collapsed_without_dance_evidence():
    event = {
        "what": "两人在舞台上完成一气呵成的连续交锋",
        "source_excerpt": "舞台灯亮起，凛先挥拳，随后肘击，最后抬膝撞向腹部。",
        "micro_actions": ["挥拳", "肘击", "膝击"],
    }

    normalized = normalize_event_action_units(event)

    assert event_uses_composite_motion(event) is False
    assert normalized["motion_mode"] == "atomic"
    assert normalized["units"] == 3


def test_structured_motion_contract_preserves_real_progression():
    events = [
        {
            "what": "三人协同搬运箱体",
            "source_excerpt": "三人在同一时刻并行发力，融为一个整体。",
            "micro_actions": ["左侧抬升", "右侧抬升", "后方稳定"],
            "generation_motion_mode": "composite",
        },
        {
            "what": "操作员按顺序锁定箱体",
            "source_excerpt": "一开始对齐，随后落锁，最终松手。",
            "micro_actions": ["对齐", "落锁", "松手"],
            "generation_motion_mode": "atomic",
        },
    ]

    assert annotate_event_motion_modes(events) is True

    assert events[0]["generation_motion_mode"] == "composite"
    assert normalize_event_action_units(events[0])["units"] == 1
    assert events[1]["generation_motion_mode"] == "atomic"
    assert normalize_event_action_units(events[1])["units"] == 3


def test_same_event_repeats_survive_shot_slicing_but_later_event_deduplicates():
    events = [
        {
            "event_role": "action_chain",
            "sequence_id": "SEQ001",
            "action_unit_id": "AU001",
            "micro_actions": ["挥拳", "挥拳"],
        },
        {
            "event_role": "action_chain",
            "sequence_id": "SEQ001",
            "action_unit_id": "AU002",
            "micro_actions": ["挥拳"],
        },
    ]
    shots = [
        {"source_events": [1], "suggested_duration": 15},
        {"source_events": [1], "suggested_duration": 15},
        {"source_events": [2], "suggested_duration": 15},
    ]

    engine._inherit_event_semantics(shots, events)

    assert [shot["micro_actions"] for shot in shots] == [["挥拳"], ["挥拳"], ["挥拳"]]
    assert [shot["generation_action_categories"] for shot in shots] == [
        ["sequential"],
        ["sequential"],
        ["duplicate"],
    ]
    assert [len(shot["generation_action_units"]) for shot in shots] == [1, 1, 0]


def test_zero_cost_event_still_requires_one_primary_occurrence():
    profile = engine.get_video_capabilities()

    assert engine._event_primary_occurrence_requirement({}, profile) == 1
    assert engine._event_primary_occurrence_requirements([{}], profile) == {1: 1}


# ── gate regression: 60s flash-mob script must pass ─────────────────────────


def test_flashmob_60s_script_passes_capacity_gate():
    events = _load_capacity_fixture(FIXTURE, LEGACY_COMPOSITE_EVENTS)
    assert len(events) == 26

    _annotate_global_event_flow(events, continuity_mode="one_take")
    plan = engine._estimate_action_capacity_plan(events, 60, 12)

    assert plan["generation_action_units"] == 22
    assert plan["primary_shots"] == 5
    assert plan["minimum_material_duration"] == 33
    assert plan["material_duration"] == 60
    assert plan["storyboard_duration_limit"] == 60
    assert plan["action_capacity_status"] == "fits_story_clock"
    assert plan["generated_duration_ratio_reference"] == 1.3
    shots = engine.estimate_action_aware_shot_count(events, 60, 12)
    assert shots == 5


def test_evolving_model_flashmob_artifact_has_stable_capacity():
    events = _load_capacity_fixture(
        EVOLVING_FIXTURE, EVOLVING_COMPOSITE_EVENTS
    )
    assert len(events) == 26

    _annotate_global_event_flow(events, continuity_mode="one_take")
    plan = engine._estimate_action_capacity_plan(events, 60, 12)

    assert {event["sequence_id"] for event in events} == {"SEQ001"}
    assert plan["generation_action_units"] == 22
    assert plan["primary_shots"] == 5
    assert plan["minimum_material_duration"] == 33
    assert plan["material_duration"] == 60
    assert plan["storyboard_duration_limit"] == 60
    assert plan["action_capacity_status"] == "fits_story_clock"


def test_sequence_fragmentation_is_reported_without_expanding_story_clock():
    events = _load_capacity_fixture(FIXTURE, LEGACY_COMPOSITE_EVENTS)

    plan = engine._estimate_action_capacity_plan(events, 60, 12)

    assert plan["primary_shots"] == 8
    assert plan["structural_shots"] == 8
    assert plan["material_duration"] == 60
    assert plan["action_capacity_status"] == "fits_story_clock"


def test_dense_single_sequence_expands_structure_before_dropping_events():
    unit_counts = [1, 3, 0, 3, 5, 5, 5, 0, 4, 2, 4, 5, 4, 0]
    events = [
        {
            "sequence_id": "SEQ001",
            "event_role": (
                "scene_setup"
                if event_id == 1
                else "turning_point"
                if event_id == 4
                else "action_chain"
            ),
            "dramatic_turn": event_id == 4,
            "micro_actions": [
                f"事件{event_id}动作{action_id}"
                for action_id in range(1, unit_count + 1)
            ],
        }
        for event_id, unit_count in enumerate(unit_counts, 1)
    ]

    plan = engine._estimate_action_capacity_plan(events, 60, 12)

    assert plan["generation_action_units"] == 41
    assert plan["structural_shots"] == 7
    assert plan["primary_shots"] == 8
    assert plan["material_duration"] == 60
    assert plan["action_capacity_status"] == "screenplay_compression_required"


def test_explicitly_dropped_events_are_audited_without_consuming_shot_capacity():
    events = [
        {"event_role": "scene_setup", "sequence_id": "SEQ001"},
        {
            "event_role": "action_chain",
            "sequence_id": "SEQ001",
            "micro_actions": [f"可删动作{index}" for index in range(12)],
        },
        {
            "event_role": "turning_point",
            "sequence_id": "SEQ001",
            "micro_actions": ["护盾爆发", "敌人失衡"],
        },
    ]
    beat = {
        "beat_order": 1,
        "source_events": [1, 3],
        "dropped_source_events": [2],
        "action": "merge",
        "reason": "保留建立与转折，显式删减重复交锋",
        "who": [],
        "where": "车厢",
        "what": "护盾改变战局",
        "suggested_duration": 15,
        "shot_size": "medium_wide",
        "camera_angle": "low",
        "camera_movement": "dolly_in",
        "lighting_key": "neon",
        "shot_intent": "reveal",
        "hero_moment": True,
        "texture_keywords": ["蓝色电弧", "湿润金属"],
    }

    parsed = engine._parse_beat_skeleton(
        json.dumps({"strategy": "压缩重复动作", "beats": [beat]}, ensure_ascii=False),
        1,
        len(events),
    )
    repaired = engine._repair_beat_action_capacity(parsed["beats"], events)
    engine._validate_beat_action_capacity(repaired, events)

    assert repaired[0]["source_events"] == [1, 3]
    assert repaired[0]["dropped_source_events"] == [2]
    shots = [{**repaired[0], "shot_order": 1, "visual": "护盾爆发"}]
    engine._inherit_event_semantics(shots, events)
    assert shots[0]["dropped_source_events"] == [2]
    assert shots[0]["micro_actions"] == ["护盾爆发", "敌人失衡"]


def test_mandatory_turning_point_cannot_be_dropped_from_story_clock():
    events = [
        {"event_role": "scene_setup", "sequence_id": "SEQ001"},
        {
            "event_role": "turning_point",
            "sequence_id": "SEQ001",
            "micro_actions": ["护盾爆发"],
        },
    ]
    beats = [
        {
            "beat_order": 1,
            "source_events": [1],
            "dropped_source_events": [2],
            "action": "keep",
        }
    ]

    with pytest.raises(ValueError, match="mandatory event 2 cannot be dropped"):
        engine._validate_beat_action_capacity(beats, events)


def test_capacity_repair_restores_redundant_whole_event_drops():
    unit_counts = [1, 3, 0, 3, 5, 5, 5, 0, 4, 2, 4, 5, 4, 0]
    events = [
        {
            "sequence_id": "SEQ001",
            "event_role": (
                "scene_setup"
                if event_id == 1
                else "turning_point"
                if event_id == 4
                else "action_chain"
            ),
            "dramatic_turn": event_id == 4,
            "who": ["角色"],
            "where": "连续动作空间",
            "what": f"事件 {event_id}",
            "micro_actions": [
                f"事件{event_id}动作{action_id}"
                for action_id in range(1, unit_count + 1)
            ],
        }
        for event_id, unit_count in enumerate(unit_counts, 1)
    ]
    source_groups = [[1, 2], [3, 4], [5], [5], [8, 11], [13], [14], [14]]
    beats = [
        {
            "beat_order": index,
            "source_events": source_events,
            "dropped_source_events": (
                [6, 7, 9, 10, 12] if index == 3 else []
            ),
            "action": "merge" if len(source_events) > 1 else "keep",
            "reason": "模型过度删减",
            "who": ["角色"],
            "where": "连续动作空间",
            "what": f"段落 {index}",
        }
        for index, source_events in enumerate(source_groups, 1)
    ]
    profile = engine.get_video_capabilities()

    repaired = engine._restore_redundantly_dropped_events(
        beats,
        events,
        material_duration=60,
        capabilities=profile,
        max_generation_units_per_beat=6,
    )

    engine._validate_beat_action_capacity(repaired, events, profile)
    engine._validate_beat_material_duration(repaired, events, 60, profile)
    kept_ids = {
        event_id for beat in repaired for event_id in beat["source_events"]
    }
    dropped_ids = {
        event_id
        for beat in repaired
        for event_id in beat["dropped_source_events"]
    }
    assert kept_ids == set(range(1, 15)) - {9}
    assert dropped_ids == {9}
    assert any(beat.get("capacity_repair") for beat in repaired)
    positions = {
        event_id: [
            beat_index
            for beat_index, beat in enumerate(repaired)
            if event_id in beat["source_events"]
        ]
        for event_id in kept_ids
    }
    assert all(
        event_positions == list(range(event_positions[0], event_positions[-1] + 1))
        for event_positions in positions.values()
    )
    ordered_intervals = [positions[event_id] for event_id in sorted(positions)]
    assert all(
        previous[-1] <= current[0]
        for previous, current in zip(ordered_intervals, ordered_intervals[1:])
    )


def test_capacity_gate_rejects_event_ranges_that_jump_back_in_time():
    events = [
        {
            "sequence_id": "SEQ001",
            "event_role": "action_chain",
            "micro_actions": [f"事件{event_id}动作"],
        }
        for event_id in range(1, 5)
    ]
    beats = [
        {
            "beat_order": 1,
            "source_events": [1, 4],
            "dropped_source_events": [],
            "action": "merge",
        },
        {
            "beat_order": 2,
            "source_events": [2, 3],
            "dropped_source_events": [],
            "action": "merge",
        },
    ]

    with pytest.raises(ValueError, match="event order jumps backward"):
        engine._validate_beat_action_capacity(beats, events)


def test_capacity_repair_separates_sequences_and_rebalances_dense_station_fight():
    unit_counts = [0, 3, 0, 3, 9, 4, 6, 3, 4, 5, 2, 1, 0]
    roles = [
        "scene_setup",
        "transition",
        "character_state",
        "action_chain",
        "action_chain",
        "action_chain",
        "action_chain",
        "action_chain",
        "action_chain",
        "turning_point",
        "character_state",
        "turning_point",
        "scene_setup",
    ]
    events = [
        {
            "sequence_id": "SEQ001" if event_id < 13 else "SEQ002",
            "event_role": roles[event_id - 1],
            "dramatic_turn": event_id == 7,
            "who": ["男子"],
            "where": "未来列车",
            "what": f"事件 {event_id}",
            "micro_actions": [
                f"男子依次挥拳动作{event_id}-{action_id}"
                for action_id in range(1, unit_counts[event_id - 1] + 1)
            ],
        }
        for event_id in range(1, 14)
    ]
    source_groups = [[1, 2, 3, 4], [5, 6], [5, 7, 8, 9, 10, 11, 12], [12, 13]]
    shot_sizes = ["medium_wide", "medium", "wide", "medium_close"]
    camera_movements = ["dolly_in", "handheld", "tracking_front", "static"]
    beats = [
        {
            "beat_order": index,
            "source_events": source_events,
            "dropped_source_events": [],
            "action": "merge",
            "reason": "模型首轮密集装箱",
            "who": ["男子"],
            "where": "未来列车",
            "what": f"战斗段落 {index}",
            "suggested_duration": 15,
            "shot_size": shot_sizes[index - 1],
            "camera_angle": "eye_level",
            "camera_movement": camera_movements[index - 1],
            "lighting_key": "neon",
            "shot_intent": "reveal" if index == 4 else "action",
            "hero_moment": index == 3,
            "texture_keywords": ["湿润金属", f"蓝色电弧{index}"],
        }
        for index, source_events in enumerate(source_groups, 1)
    ]
    parsed = engine._parse_beat_skeleton(
        json.dumps({"strategy": "压缩列车战斗", "beats": beats}, ensure_ascii=False),
        4,
        len(events),
    )
    profile = engine.get_video_capabilities()
    story_capacity = engine._generation_unit_capacity_for_story_duration(
        15,
        profile,
    )

    repaired = engine._repair_beat_action_capacity(
        parsed["beats"],
        events,
        profile,
        max_generation_units_per_beat=story_capacity,
    )

    engine._validate_beat_action_capacity(repaired, events, profile)
    assert [beat["sequence_id"] for beat in repaired] == [
        "SEQ001",
        "SEQ001",
        "SEQ001",
        "SEQ002",
    ]
    assert [
        {
            events[event_id - 1]["sequence_id"]
            for event_id in beat["source_events"]
        }
        for beat in repaired
    ] == [{"SEQ001"}, {"SEQ001"}, {"SEQ001"}, {"SEQ002"}]
    mandatory_ids = {
        event_id
        for event_id, event in enumerate(events, 1)
        if engine._event_is_mandatory_for_adaptation(event)
    }
    kept_ids = {
        event_id for beat in repaired for event_id in beat["source_events"]
    }
    dropped_ids = {
        event_id
        for beat in repaired
        for event_id in beat["dropped_source_events"]
    }
    assert mandatory_ids <= kept_ids
    assert kept_ids.isdisjoint(dropped_ids)
    assert kept_ids | dropped_ids == set(range(1, 14))
    assert dropped_ids
    assert max(engine._beat_generation_unit_loads(repaired, events)) <= story_capacity
    assert max(engine._beat_content_loads(repaired, events, profile)) <= 3


def test_optional_transition_sequence_does_not_steal_mandatory_story_capacity():
    events = [
        {
            "sequence_id": "SEQ001",
            "event_role": "turning_point",
            "dramatic_turn": True,
            "what": f"必保转折 {event_id}",
            "micro_actions": [
                f"男子依次完成转折动作{event_id}-{action_id}"
                for action_id in range(1, unit_count + 1)
            ],
        }
        for event_id, unit_count in enumerate((3, 7, 1), 1)
    ]
    events.extend([
        {
            "sequence_id": "SEQ002",
            "event_role": "transition",
            "dramatic_turn": False,
            "what": "可删的过场",
            "micro_actions": [],
        },
        {
            "sequence_id": "SEQ003",
            "event_role": "scene_setup",
            "dramatic_turn": False,
            "what": "结尾新场景建立",
            "micro_actions": [],
        },
    ])
    beats = [
        {
            "beat_order": index,
            "source_events": source_events,
            "dropped_source_events": [],
            "action": "merge" if len(source_events) > 1 else "keep",
            "reason": "模型初始分配",
            "who": [],
            "where": "未来列车",
            "what": f"段落 {index}",
        }
        for index, source_events in enumerate(([1], [2], [3], [4, 5]), 1)
    ]
    profile = engine.get_video_capabilities()
    story_capacity = engine._generation_unit_capacity_for_story_duration(
        15,
        profile,
    )

    repaired = engine._repair_beat_action_capacity(
        beats,
        events,
        profile,
        max_generation_units_per_beat=story_capacity,
    )

    assert [beat["sequence_id"] for beat in repaired] == [
        "SEQ001",
        "SEQ001",
        "SEQ001",
        "SEQ003",
    ]
    assert 4 not in {
        event_id for beat in repaired for event_id in beat["source_events"]
    }
    assert [
        event_id
        for beat in repaired
        for event_id in beat["dropped_source_events"]
    ] == [4]
    assert 5 in repaired[-1]["source_events"]
    assert max(engine._beat_generation_unit_loads(repaired, events)) <= story_capacity


def test_flashmob_one_take_has_a_feasible_four_beat_skeleton(monkeypatch):
    events = _load_capacity_fixture(FIXTURE, LEGACY_COMPOSITE_EVENTS)
    _annotate_global_event_flow(events, continuity_mode="one_take")
    dropped_groups = [[5], [], [10, 13], [19]]
    source_groups = [
        [event_id for event_id in range(1, 6) if event_id not in dropped_groups[0]],
        list(range(6, 8)),
        [event_id for event_id in range(8, 19) if event_id not in dropped_groups[2]],
        [event_id for event_id in range(19, 27) if event_id not in dropped_groups[3]],
    ]
    shot_sizes = ["medium_wide", "medium", "wide", "medium_close"]
    beats = [
        {
            "beat_order": index,
            "source_events": list(source_group),
            "dropped_source_events": dropped_groups[index - 1],
            "action": "merge",
            "reason": "按连续动作容量装箱",
            "who": [],
            "where": "日本都市街头",
            "what": f"连续舞蹈段落 {index}",
            "suggested_duration": duration,
            "shot_size": shot_sizes[index - 1],
            "camera_angle": "eye_level",
            "camera_movement": "handheld",
            "lighting_key": "natural",
            "shot_intent": "reveal" if index == 4 else "action",
            "hero_moment": index == 4,
            "texture_keywords": ["城市路面", f"街头层次{index}"],
        }
        for index, (source_group, duration) in enumerate(
            zip(source_groups, [15, 15, 15, 15], strict=True),
            1,
        )
    ]
    monkeypatch.setattr(
        engine,
        "_call_llm_with_timeout_retry",
        lambda *_args, **_kwargs: json.dumps(
            {"strategy": "四段连续动作装箱", "beats": beats},
            ensure_ascii=False,
        ),
    )
    profile = engine.get_video_capabilities()

    skeleton = engine._build_beat_skeleton(events, "", 60, 15, 4)

    assert engine._beat_content_loads(skeleton["beats"], events, profile) == [3, 3, 3, 2]
    assert [
        event_id
        for beat in skeleton["beats"]
        for event_id in beat["dropped_source_events"]
    ] == []


def test_layered_expansion_preserves_skeleton_shot_language(monkeypatch):
    beat = {
        "beat_order": 1,
        "source_events": [1],
        "action": "keep",
        "reason": "保留开场",
        "who": [],
        "where": "日本都市街头",
        "what": "建立街头空间",
        "suggested_duration": 15,
        "shot_size": "medium_wide",
        "camera_angle": "eye_level",
        "camera_movement": "handheld",
        "lighting_key": "overcast_soft",
        "shot_intent": "establishing",
        "hero_moment": True,
        "texture_keywords": ["湿润路面", "手机HDR高光"],
    }
    expanded = {
        "strategy": "展开",
        "shots": [{
            "beat_order": 1,
            "shot_order": 1,
            "source_events": [1],
            "action": "keep",
            "reason": "展开",
            "who": [],
            "where": "日本都市街头",
            "what": "建立街头空间",
            "emotion": "自然",
            "visual": "手机街头画面",
            "suggested_duration": 15,
            "boundary_before": "cut",
            "continuity_reason": "开场",
            "continuity_subject": "",
            "transition_to_next": "cut",
            "associate_assets": ["scene:日本都市街头"],
            "shot_size": "wide",
            "camera_angle": "high",
            "camera_movement": "static",
            "lighting_key": "natural",
            "shot_intent": "atmosphere",
            "hero_moment": False,
            "texture_keywords": [],
            "dialogue": None,
            "gen_strategy": "phantom",
        }],
    }
    monkeypatch.setattr(
        engine,
        "_call_llm_with_timeout_retry",
        lambda *_args, **_kwargs: json.dumps(expanded, ensure_ascii=False),
    )

    shots = engine._expand_beats_to_shots([beat], "", 15, 15)

    assert {
        field: shots[0][field]
        for field in engine._SHOT_LANGUAGE_FIELDS
    } == {
        field: beat[field]
        for field in engine._SHOT_LANGUAGE_FIELDS
    }


def test_storyboard_story_clock_is_capped_but_generated_ratio_is_advisory():
    too_long = {
        "target_duration": 79,
        "delivery_target_duration": 60,
        "generated_duration_ratio_reference": 1.3,
        "shots": [{"id": "S01", "duration": 79}],
    }
    issues, _ = run_l1_checks(too_long, "")
    assert "storyboard_duration_exceeds_delivery_target" in {
        issue["code"] for issue in issues
    }

    within_story_clock = {
        "target_duration": 60,
        "delivery_target_duration": 60,
        "generated_duration_ratio_reference": 1.3,
        "shots": [{"id": "S01", "duration": 60}],
    }
    issues, _ = run_l1_checks(within_story_clock, "")
    assert not {
        issue["code"] for issue in issues
    } & {
        "storyboard_duration_exceeds_delivery_target",
        "pre_edit_material_ratio_exceeded",
    }


def test_phase5_checks_additive_bridge_ledger_and_handle_replacement():
    storyboard = {
        "continuity_mode": "one_take",
        "video_provider": "seedance",
        "delivery_target_duration": 30,
        "generated_duration_ratio_reference": 1.3,
        "shots": [
            {"id": "S01", "duration": 15, "micro_actions": ["前进"]},
            {"id": "S02", "duration": 15, "micro_actions": ["继续前进"]},
        ],
    }
    plan_storyboard_beats(storyboard)

    issues, _ = run_l1_checks(storyboard, "")
    assert not {
        "bridge_handle_budget_mismatch",
        "material_budget_ledger_stale",
        "material_budget_ledger_missing",
    } & {issue["code"] for issue in issues}
    assert storyboard["material_budget"]["story_clock_duration_s"] == 30
    assert storyboard["material_budget"]["bridge_provider_request_duration_s"] == 4
    assert storyboard["material_budget"]["total_provider_request_duration_s"] == 34
    assert storyboard["material_budget"][
        "projected_pre_edit_timeline_duration_s"
    ] == 30

    tampered = json.loads(json.dumps(storyboard))
    tampered["primary_shot_bridges"][0]["source_handle_s"] = 0.5
    issues, _ = run_l1_checks(tampered, "")
    codes = {issue["code"] for issue in issues}
    assert "bridge_handle_budget_mismatch" in codes
    assert "material_budget_ledger_stale" in codes


def test_one_take_budget_counts_only_adjacent_continuous_boundaries():
    storyboard = {
        "continuity_mode": "one_take",
        "video_provider": "seedance",
        "delivery_target_duration": 60,
        "generated_duration_ratio_reference": 1.3,
        "shots": [
            {"id": f"S{index:02d}", "duration": 15, "micro_actions": ["连续动作"]}
            for index in range(1, 5)
        ],
    }

    plan_storyboard_beats(storyboard)

    budget = storyboard["material_budget"]
    assert budget["story_clock_duration_s"] == 60
    assert budget["storyboard_duration_limit_s"] == 60
    assert budget["bridge_count"] == 3
    assert budget["bridge_provider_request_duration_s"] == 12
    assert budget["total_provider_request_duration_s"] == 72
    assert budget["total_provider_request_duration_ratio"] == 1.2
    assert budget["generated_duration_ratio_reference"] == 1.3
    assert budget["generated_duration_ratio_is_hard_limit"] is False
    assert budget["projected_pre_edit_timeline_duration_s"] == 60


def test_bridge_overhead_may_fluctuate_above_1_3_without_growing_story_clock():
    storyboard = {
        "target_duration": 60,
        "delivery_target_duration": 60,
        "generated_duration_ratio_reference": 1.3,
        "shots": [
            {"id": f"S{index:02d}", "duration": 7.5}
            for index in range(1, 9)
        ],
        "primary_shot_bridges": [
            {
                "bridge_id": f"S{index:02d}__S{index + 1:02d}",
                "generation_duration_s": 4,
                "generation_duration_range_s": [4, 6],
                "visible_duration_s": 4,
                "source_handle_s": 2,
                "target_handle_s": 2,
                "timeline_insertion_policy": "replace_boundary_handles",
            }
            for index in range(1, 8)
        ],
    }

    issues, _ = run_l1_checks(storyboard, "")
    codes = {issue["code"] for issue in issues}
    assert "storyboard_duration_exceeds_delivery_target" not in codes
    assert "pre_edit_material_ratio_exceeded" not in codes

    from utils.material_budget import build_material_budget

    ledger = build_material_budget(storyboard)
    assert ledger["story_clock_duration_s"] == 60
    assert ledger["total_provider_request_duration_s"] == 88
    assert ledger["total_provider_request_duration_ratio"] == 1.466667
    assert ledger["total_provider_request_duration_ratio_range"] == [1.466667, 1.7]
    assert ledger["generated_duration_ratio_is_hard_limit"] is False


def test_composite_motion_keeps_full_ledger_through_pxx_and_qa():
    actions = [
        "摄影师以朋友视角持iPhone跟拍女主",
        "女主保持轻松自然的笑容",
        "路人从斑马线加入",
        "更多普通行人陆续开始加入",
        "背景舞者连贯衔接肩胸律动",
        "背景舞者衔接身体波浪",
    ]
    storyboard = {
        "video_provider": "seedance",
        "shots": [{
            "id": "S01",
            "duration": 15,
            "micro_actions": actions,
            "generation_motion_mode": "composite",
            "shot_size": "medium_wide",
            "camera_angle": "eye_level",
            "camera_movement": "handheld",
            "lighting_key": "natural",
            "shot_intent": "action",
            "hero_moment": True,
            "texture_keywords": ["街道路面", "手机HDR高光"],
        }],
    }

    plan_storyboard_beats(storyboard)

    shot = storyboard["shots"][0]
    observed = [
        action
        for beat in shot["storyboard_beats"]
        for action in beat["micro_actions"]
    ]
    assert observed == actions
    assert len(shot["generation_action_units"]) == 1
    assert shot["storyboard_beat_count"] == 1
    assert {
        field: shot["storyboard_beats"][0][field]
        for field in engine._SHOT_LANGUAGE_FIELDS
    } == {
        field: shot[field]
        for field in engine._SHOT_LANGUAGE_FIELDS
    }
    assert run_generation_capacity_checks(storyboard) == []


# ── guard: genuinely dense sequential chains remain explicit pressure ──────


def test_dense_sequential_chain_uses_story_time_not_provider_padding():
    events = [{
        "event_role": "action_chain",
        "micro_actions": [
            "起势蓄力", "挥拳", "肘击", "膝击", "扫腿",
            "擒拿", "过肩摔", "地面压制", "锁技", "锁喉",
            "翻滚压制", "降服拍地",
        ],
    }]
    plan = engine._estimate_action_capacity_plan(events, 30, 12)

    assert plan["generation_action_units"] == 12
    assert plan["structural_shots"] == 2
    assert plan["primary_shots"] == 3
    assert plan["minimum_material_duration"] == 18
    assert plan["material_duration"] == 30
    assert plan["action_capacity_status"] == "fits_story_clock"


def test_provider_request_padding_does_not_consume_story_clock():
    """Provider request minima are generated cost, not authored narrative time."""
    profile = engine.get_video_capabilities()
    assert profile.request_duration_bounds("multi_image") == (8, 15)
    assert profile.request_duration_bounds("tail_video_extend") == (6, 10)
    assert profile.effective_duration_bounds("multi_image") == (3, 15)
    assert profile.effective_duration_bounds("tail_video_extend") == (3, 10)

    events = [{
        "event_role": "action_chain",
        "sequence_id": "SEQ001",
        "micro_actions": [f"有序操作{index}" for index in range(12)],
    }]
    plan = engine._estimate_action_capacity_plan(events, 30, 15)

    assert plan["generation_action_units"] == 12
    assert plan["structural_shots"] == 2
    assert plan["minimum_material_duration"] == 18
    assert plan["action_capacity_status"] == "fits_story_clock"

    storyboard = {
        "video_provider": "seedance",
        "delivery_target_duration": 3,
        "shots": [
            {
                "id": "S01",
                "duration": 3,
                "micro_actions": ["抬头确认信号"],
            }
        ],
    }
    plan_storyboard_beats(storyboard)
    beat = storyboard["shots"][0]["storyboard_beats"][0]

    assert beat["effective_story_duration_s"] == 3
    assert beat["provider_request_duration_s"] == 8
    assert beat["provider_minimum_padding_duration_s"] == 5
    assert storyboard["material_budget"]["story_clock_duration_s"] == 3
    assert storyboard["material_budget"][
        "content_provider_request_duration_s"
    ] == 8
    assert storyboard["material_budget"][
        "content_provider_padding_duration_s"
    ] == 5
    assert storyboard["material_budget"][
        "provider_request_duration_is_story_clock_limit"
    ] is False


def test_generic_dense_actions_report_screenplay_compression_pressure():
    generic = [
        {
            "event_role": "action_chain",
            "action_unit_id": f"AU{i:03d}",
            "micro_actions": [f"动作{i}-{j}" for j in range(8)],
        }
        for i in range(1, 13)
    ]
    events = [
        {"event_role": "scene_setup"},
        *generic[:4],
        {"event_role": "dialogue"},
        *generic[4:8],
        {"event_role": "dialogue"},
        *generic[8:],
        {"event_role": "scene_setup"},
    ]
    plan = engine._estimate_action_capacity_plan(events, 60, 10)

    assert plan["generation_action_units"] == 96
    assert plan["structural_shots"] == 16
    assert plan["primary_shots"] == 8
    assert plan["minimum_material_duration"] == 144
    assert plan["material_duration"] == 60
    assert plan["action_capacity_status"] == "screenplay_compression_required"


def test_60s_scaled_screenplay_reconciles_source_pressure_into_executable_ledger():
    """A fitted production screenplay must not inherit the source ledger's debt."""
    events = [
        {
            "sequence_id": "SEQ001",
            "event_role": (
                "scene_setup" if event_id == 1
                else "turning_point" if event_id == 7
                else "action_chain"
            ),
            "dramatic_turn": event_id == 7,
            "micro_actions": [
                f"事件{event_id}动作{action_id}"
                for action_id in range(1, 7)
            ],
        }
        for event_id in range(1, 8)
    ]
    source_capacity = engine._estimate_action_capacity_plan(events, 60, 12)
    assert source_capacity["action_capacity_status"] == (
        "screenplay_compression_required"
    )

    source_groups = ([1], [1], [2], [2], [3], [5], [6], [7])
    generation_unit_counts = (3, 3, 3, 3, 6, 6, 6, 6)
    durations = (6, 6, 6, 6, 9, 9, 9, 9)
    shots = [
        {
            "shot_order": shot_order,
            "source_events": list(source_events),
            "dropped_source_events": [4] if shot_order == 1 else [],
            "source_sequence_ids": ["SEQ001"],
            "suggested_duration": duration,
            "action": "keep",
            "what": f"生产节拍{shot_order}",
            "generation_action_units": [
                {"unit_id": f"GAU{unit_id:03d}"}
                for unit_id in range(1, generation_unit_count + 1)
            ],
        }
        for shot_order, (
            source_events,
            generation_unit_count,
            duration,
        ) in enumerate(
            zip(source_groups, generation_unit_counts, durations, strict=True),
            1,
        )
    ]

    screenplay_plan, reconciled = engine._build_screenplay_plan(
        events,
        shots,
        source_capacity,
        target_duration=60,
        source_events_hash="source-events-sha256",
    )

    assert screenplay_plan["schema"] == "honcut.screenplay-plan.v4"
    assert screenplay_plan["source_ledger"]["capacity_status"] == (
        "screenplay_compression_required"
    )
    assert screenplay_plan["production_ledger"] == {
        "capacity_status": "fits_story_clock",
        "duration_scaling_status": "applied",
        "event_action_scaling_schema": "honcut.duration-scaled-event-plan.v2",
        "intra_event_scaling_applied": False,
        "intra_event_omitted_micro_action_count": 0,
        "event_count": 6,
        "generation_action_units": 36,
        "effective_story_duration_s": 60,
        "kept_source_event_ids": [1, 2, 3, 5, 6, 7],
        "omitted_source_event_ids": [4],
        "base_mandatory_source_event_ids": [1, 7],
        "mandatory_source_event_ids": [1, 7],
        "causal_predecessor_source_event_ids": [],
    }
    assert sum(beat["duration_s"] for beat in screenplay_plan["beats"]) == 60
    assert reconciled["source_action_capacity_status"] == (
        "screenplay_compression_required"
    )
    assert reconciled["action_capacity_status"] == "fits_story_clock"
    assert reconciled["duration_scaling_status"] == "applied"
    assert reconciled["production_generation_action_units"] == 36


def test_duration_scaled_event_plan_fits_dense_mandatory_actions_without_mutating_source():
    """Dense mandatory events keep their facts while production choreography scales."""
    unit_counts = [3, 0, 4, 5, 5, 4, 1, 4, 3, 7, 5, 5]
    mandatory_roles = {
        1: "scene_setup",
        3: "turning_point",
        7: "scene_setup",
        10: "turning_point",
        12: "consequence",
    }
    events = [
        {
            "sequence_id": "SEQ001",
            "event_role": mandatory_roles.get(event_id, "action_chain"),
            "dramatic_turn": event_id in {3, 10},
            "what": f"来源事件 {event_id} 的因果结果",
            "micro_actions": [
                f"事件{event_id}依次执行动作{action_id}"
                for action_id in range(1, unit_count + 1)
            ],
        }
        for event_id, unit_count in enumerate(unit_counts, 1)
    ]
    source_snapshot = json.loads(json.dumps(events, ensure_ascii=False))
    source_capacity = engine._estimate_action_capacity_plan(events, 32, 12)
    profile = engine.get_video_capabilities()
    beat_count = source_capacity["primary_shots"]
    effective_shot_duration = round(32 / beat_count)

    production_events, scaling_plan = engine._build_duration_scaled_event_plan(
        events,
        target_duration=32,
        beat_count=beat_count,
        effective_shot_duration=effective_shot_duration,
        capabilities=profile,
    )

    assert events == source_snapshot
    assert scaling_plan["schema"] == "honcut.duration-scaled-event-plan.v2"
    assert scaling_plan["source_generation_action_units"] == 46
    assert scaling_plan["production_generation_action_units"] < 46
    assert scaling_plan["intra_event_scaling_applied"] is True
    assert len(production_events) == len(events)
    mandatory_ids = set(mandatory_roles)
    assert mandatory_ids <= {
        item["source_event_id"] for item in scaling_plan["events"]
    }
    assert any(
        item["source_event_id"] in mandatory_ids
        and item["omitted_source_micro_action_indexes"]
        for item in scaling_plan["events"]
    )
    per_beat_capacity = engine._generation_unit_capacity_for_story_duration(
        effective_shot_duration,
        profile,
    )
    sequence_plan = engine._sequence_beat_plan(
        production_events,
        beat_count,
        per_beat_capacity,
    )
    assert sequence_plan == ["SEQ001"] * beat_count
    assert all(
        production_events[event_id - 1]["what"] == events[event_id - 1]["what"]
        for event_id in mandatory_ids
    )


def test_mandatory_event_protects_continuous_predecessors_until_cut_boundary():
    """A kept outcome cannot survive after its same-sequence causes are omitted."""
    events = [
        {
            "sequence_id": "SEQ000",
            "event_role": "transition",
            "continuity_before": "cut",
            "what": "unrelated prologue",
            "micro_actions": ["prologue action"],
        },
        {
            "sequence_id": "SEQ001",
            "event_role": "action_chain",
            "continuity_before": "cut",
            "what": "the attack begins",
            "micro_actions": ["enemy attacks"],
        },
        {
            "sequence_id": "SEQ001",
            "event_role": "action_chain",
            "continuity_before": "continuous",
            "what": "the protagonist counters",
            "micro_actions": ["protagonist counters"],
        },
        {
            "sequence_id": "SEQ001",
            "event_role": "turning_point",
            "continuity_before": "continuous",
            "what": "the counter changes the fight",
            "micro_actions": ["enemy loses balance"],
        },
        {
            "sequence_id": "SEQ002",
            "event_role": "transition",
            "continuity_before": "cut",
            "what": "unrelated epilogue",
            "micro_actions": ["epilogue action"],
        },
    ]

    assert engine._mandatory_adaptation_event_ids(events) == {2, 3, 4}

    production_events, scaling_plan = engine._build_duration_scaled_event_plan(
        events,
        target_duration=32,
        beat_count=5,
        effective_shot_duration=6,
    )

    assert len(production_events) == len(events)
    assert scaling_plan["mandatory_source_event_ids"] == [2, 3, 4]
    assert {
        record["source_event_id"]
        for record in scaling_plan["events"]
        if record["mandatory"]
    } == {2, 3, 4}


def test_optional_kept_event_cannot_orphan_its_continuous_predecessor():
    events = [
        {
            "sequence_id": "SEQ001",
            "event_role": "action_chain",
            "continuity_before": "cut",
            "micro_actions": ["train arrives"],
        },
        {
            "sequence_id": "SEQ001",
            "event_role": "character_state",
            "continuity_before": "continuous",
            "micro_actions": ["protagonist waits at the open door"],
        },
    ]
    beats = [{
        "beat_order": 1,
        "source_events": [2],
        "dropped_source_events": [1],
        "action": "keep",
    }]

    with pytest.raises(ValueError, match="omit continuous predecessors"):
        engine._validate_beat_action_capacity(beats, events)


def test_duration_scaling_dp_fits_long_continuous_chain_without_state_explosion():
    events = [
        {
            "sequence_id": "SEQ001",
            "event_role": "turning_point" if event_id == 12 else "action_chain",
            "dramatic_turn": event_id == 12,
            "continuity_before": "cut" if event_id == 1 else "continuous",
            "what": f"causal action {event_id}",
            "micro_actions": [
                f"event {event_id} action {action_id}"
                for action_id in range(1, 5)
            ],
        }
        for event_id in range(1, 13)
    ]

    production_events, scaling_plan = engine._build_duration_scaled_event_plan(
        events,
        target_duration=32,
        beat_count=5,
        effective_shot_duration=6,
    )

    assert len(production_events) == 12
    assert scaling_plan["mandatory_source_event_ids"] == list(range(1, 13))
    assert scaling_plan["production_generation_action_units"] <= 20
    assert scaling_plan["intra_event_scaling_applied"] is True


def test_production_director_projection_excludes_unselected_sequence_plot_facts():
    selected_events = [{
        "sequence_id": "SEQ001",
        "what": "the chip projects coordinates",
        "emotion": "controlled suspense",
        "visual": "blue coordinates hover over the chip",
        "where": "inside the train carriage",
        "start_state": "the chip is dark",
        "end_state": "coordinates are visible",
        "continuity_before": "continuous",
    }]
    source_intent = {
        "sequence_id": "SEQ001",
        "scene_goal": "end with the forbidden tunnel reveal",
        "emotion_arc": "suspense to forbidden tunnel awe",
        "visual_focus": "FORBIDDEN_NEON_TUNNEL",
        "spatial_intent": "leave the carriage for the tunnel exterior",
        "transition_intent": "reveal FORBIDDEN_NEON_TUNNEL",
    }

    projection = engine._build_production_director_intent(
        source_intent,
        selected_events,
        source_event_ids=[12],
        shot={"camera_movement": "dolly_out"},
    )

    serialized = json.dumps(projection, ensure_ascii=False)
    assert projection["schema"] == "honcut.production-director-intent.v1"
    assert projection["source_event_ids"] == [12]
    assert "FORBIDDEN_NEON_TUNNEL" not in serialized
    assert "the chip projects coordinates" in projection["scene_goal"]
    assert "blue coordinates hover over the chip" in projection["visual_focus"]


def test_production_beat_text_fields_exclude_unselected_model_texture():
    selected_events = [{
        "sequence_id": "SEQ001",
        "who": ["protagonist"],
        "where": "inside the train carriage",
        "what": "the chip projects coordinates",
        "visual": "blue coordinates hover over the chip",
    }]
    beat = {
        "source_events": [12],
        "reason": "reveal FORBIDDEN_NEON_TUNNEL",
        "who": ["invented figure"],
        "where": "FORBIDDEN_NEON_TUNNEL",
        "what": "reveal FORBIDDEN_NEON_TUNNEL",
        "texture_keywords": ["FORBIDDEN_NEON_TUNNEL", "invented exterior"],
    }

    engine._ground_production_beat_text_fields(beat, selected_events)

    serialized = json.dumps(beat, ensure_ascii=False)
    assert "FORBIDDEN_NEON_TUNNEL" not in serialized
    assert beat["who"] == ["protagonist"]
    assert beat["where"] == "inside the train carriage"
    assert beat["what"] == "the chip projects coordinates"
    assert 2 <= len(beat["texture_keywords"]) <= 4


def test_screenplay_plan_binds_production_director_projection_to_source_refs():
    events = [{
        "sequence_id": "SEQ001",
        "event_role": "turning_point",
        "dramatic_turn": True,
        "continuity_before": "cut",
        "what": "the chip projects coordinates",
        "emotion": "controlled suspense",
        "visual": "blue coordinates hover over the chip",
        "where": "inside the train carriage",
        "micro_actions": ["coordinates appear"],
    }]
    source_intent = {
        "sequence_id": "SEQ001",
        "scene_goal": "resolve the sequence",
        "emotion_arc": "suspense",
        "visual_focus": "the chip",
        "spatial_intent": "inside the carriage",
        "transition_intent": "hold",
    }
    projection = engine._build_production_director_intent(
        source_intent,
        events,
        source_event_ids=[1],
        shot={"camera_movement": "dolly_out"},
    )
    source_capacity = engine._estimate_action_capacity_plan(events, 12, 12)
    shots = [{
        "shot_order": 1,
        "source_events": [1],
        "dropped_source_events": [],
        "source_sequence_ids": ["SEQ001"],
        "suggested_duration": 12,
        "action": "keep",
        "what": "the chip projects coordinates",
        "director_intent": projection,
        "generation_action_units": [{"unit_id": "GAU001"}],
    }]

    screenplay_plan, _ = engine._build_screenplay_plan(
        events,
        shots,
        source_capacity,
        target_duration=12,
    )

    assert screenplay_plan["production_ledger"][
        "production_director_intent_schema"
    ] == "honcut.production-director-intent.v1"
    assert screenplay_plan["beats"][0]["director_intent"][
        "source_event_ids"
    ] == [1]

    shots[0]["director_intent"]["source_event_ids"] = [2]
    with pytest.raises(ValueError, match="director intent lineage"):
        engine._build_screenplay_plan(
            events,
            shots,
            source_capacity,
            target_duration=12,
        )


def test_screenplay_plan_records_intra_event_action_lineage():
    events = [{
        "sequence_id": "SEQ001",
        "event_role": "turning_point",
        "dramatic_turn": True,
        "what": "角色从受压转为反制",
        "micro_actions": ["后退", "格挡", "控制手臂", "完成反制"],
    }]
    source_capacity = engine._estimate_action_capacity_plan(events, 12, 12)
    production_events, scaling_plan = engine._build_duration_scaled_event_plan(
        events,
        target_duration=12,
        beat_count=1,
        effective_shot_duration=12,
    )
    selected_indexes = scaling_plan["events"][0][
        "selected_source_micro_action_indexes"
    ]
    shots = [{
        "shot_order": 1,
        "source_events": [1],
        "dropped_source_events": [],
        "source_sequence_ids": ["SEQ001"],
        "suggested_duration": 12,
        "action": "keep",
        "what": "角色完成反制",
        "generation_action_units": normalize_event_action_units(
            production_events[0]
        )["generation_action_units"],
    }]

    screenplay_plan, _ = engine._build_screenplay_plan(
        events,
        shots,
        source_capacity,
        target_duration=12,
        production_events=production_events,
        duration_scaled_event_plan=scaling_plan,
    )

    assert screenplay_plan["schema"] == "honcut.screenplay-plan.v4"
    assert screenplay_plan["production_ledger"]["event_action_scaling_schema"] == (
        "honcut.duration-scaled-event-plan.v2"
    )
    assert screenplay_plan["beats"][0]["production_action_refs"] == [{
        "source_event_id": 1,
        "selected_source_micro_action_indexes": selected_indexes,
        "omitted_source_micro_action_indexes": scaling_plan["events"][0][
            "omitted_source_micro_action_indexes"
        ],
    }]


def test_phase5_capacity_uses_duration_scaled_production_event_ledger():
    storyboard = {
        "shots": [{
            "id": "S01",
            "duration": 5,
            "source_action_unit_ids": ["AU001", "AU003"],
            "camera_movement": "pan",
        }],
    }
    events_data = {
        "events": [
            {"action_unit_id": "AU001"},
            {"action_unit_id": "AU002"},
            {"action_unit_id": "AU003"},
        ]
    }
    screenplay_plan = {
        "schema": engine.SCREENPLAY_PLAN_SCHEMA,
        "beats": [],
        "production_ledger": {
            "base_mandatory_source_event_ids": [],
            "mandatory_source_event_ids": [],
            "causal_predecessor_source_event_ids": [],
        },
        "event_action_scaling": {
            "schema": engine.DURATION_SCALED_EVENT_PLAN_SCHEMA,
            "events": [
                {
                    "source_event_id": 1,
                    "production_status": "kept",
                    "mandatory": False,
                },
                {
                    "source_event_id": 2,
                    "production_status": "whole_event_omitted",
                    "mandatory": False,
                },
                {
                    "source_event_id": 3,
                    "production_status": "kept",
                    "mandatory": False,
                },
            ],
        },
    }

    issues = run_generation_capacity_checks(
        storyboard,
        events_data,
        screenplay_plan,
    )

    assert "action_unit_coverage_missing" not in {
        issue["code"] for issue in issues
    }


@pytest.mark.parametrize(
    "old_schema",
    ["honcut.screenplay-plan.v2", "honcut.screenplay-plan.v3"],
)
def test_phase5_rejects_pre_projection_screenplay_plan_schema(old_schema):
    issues = run_generation_capacity_checks(
        {"shots": []},
        {"events": []},
        {
            "schema": old_schema,
            "event_action_scaling": {
                "schema": "honcut.duration-scaled-event-plan.v1",
                "events": [],
            },
        },
    )

    assert "screenplay_plan_lineage_invalid" in {
        issue["code"] for issue in issues
    }


def test_phase5_rejects_misaligned_production_director_projection():
    screenplay_plan = {
        "schema": engine.SCREENPLAY_PLAN_SCHEMA,
        "production_ledger": {
            "base_mandatory_source_event_ids": [],
            "mandatory_source_event_ids": [],
            "causal_predecessor_source_event_ids": [],
            "production_director_intent_schema": (
                "honcut.production-director-intent.v1"
            ),
        },
        "event_action_scaling": {
            "schema": engine.DURATION_SCALED_EVENT_PLAN_SCHEMA,
            "events": [{
                "source_event_id": 1,
                "production_status": "kept",
                "mandatory": False,
            }],
        },
        "beats": [{
            "sequence_id": "SEQ001",
            "source_refs": [1],
            "director_intent": {
                "schema": "honcut.production-director-intent.v1",
                "sequence_id": "SEQ001",
                "source_event_ids": [2],
            },
        }],
    }

    issues = run_generation_capacity_checks(
        {"shots": []},
        {"events": [{"action_unit_id": "AU001"}]},
        screenplay_plan,
    )

    assert "screenplay_plan_lineage_invalid" in {
        issue["code"] for issue in issues
    }

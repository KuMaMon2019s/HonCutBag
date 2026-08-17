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


# ── classifier unit tests ───────────────────────────────────────────────────


def test_classifier_sequential_combat_actions():
    for text in ["挥拳", "肘击", "膝击", "缴械", "抓门框", "稳定", "穿门", "避让工具箱"]:
        assert classify_micro_action(text) == "sequential", text


def test_classifier_simultaneous_group_groove():
    for text in [
        "路人从斑马线加入",
        "更多普通行人陆续开始加入",
        "原本正常经过的人突然开始跟着音乐Groove",
        "跳舞人数从1人逐渐变成3～5人",
        "背景舞者连贯衔接肩胸律动",
        "新加入者同步程度逐渐提高",
    ]:
        assert classify_micro_action(text) == "simultaneous", text


def test_classifier_sustained_camera_and_state():
    for text in [
        "摄影师以朋友视角持iPhone跟拍女主",
        "视频开场直接呈现表演状态，无空镜、无静止站立摆拍环节",
        "露出轻松自然的笑容",
        "与镜头保持自信的眼神交流",
        "整段动作一气呵成并向前推进",
        "第一个路人注意到音乐和女主的动作",
        "音乐进入更加明显的节奏段落",
        "音乐播放至最后几个重拍节点",
        "视频录制进入收尾阶段",
        "女主不停止舞蹈，互动动作融入舞步",
        "女主与全体舞者未停止舞蹈，也未集体站住摆最终Pose",
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
        "背景舞者做快速前进Groove",
        "背景舞者连贯衔接肩胸律动",
        "背景舞者衔接身体波浪",
        "背景舞者衔接大幅挥臂",
    ]
    assert normalized_action_unit_count(actions) == 1


def test_sustained_only_actions_cost_zero_units():
    actions = [
        "露出轻松自然的笑容",
        "与镜头保持自信的眼神交流",
        "摄影师以朋友视角持iPhone跟拍女主",
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


def test_source_authored_compound_dance_is_one_generation_unit():
    event = {
        "what": "女主完成一段连贯复合舞蹈",
        "source_excerpt": (
            "肩胸隔离、胯部点缀、绕臂和脚步全部融为一段复合律动，"
            "而不是逐个执行分离动作。"
        ),
        "micro_actions": [
            "肩部做隔离",
            "胸部做隔离",
            "脚步轻快向前推进",
            "手臂完成绕臂",
        ],
    }

    normalized = normalize_event_action_units(event)

    assert event_uses_composite_motion(event) is True
    assert normalized["motion_mode"] == "composite"
    assert normalized["units"] == 1
    assert normalized["ledger"] == event["micro_actions"]


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


def test_document_compound_dance_contract_preserves_real_progression():
    events = [
        {
            "what": "定义整支短片的舞蹈语法",
            "source_excerpt": (
                "剧本中所有舞蹈描述都表示每个瞬间一个连贯的复合律动，"
                "而非一连串需要逐个执行的分离动作清单。"
            ),
            "micro_actions": [],
        },
        {
            "what": "女主与背景舞者跳出同步版本",
            "source_excerpt": "背景舞者连贯衔接肩胸律动、身体波浪与大幅挥臂。",
            "micro_actions": ["快速前进", "肩胸律动", "身体波浪", "大幅挥臂"],
        },
        {
            "what": "路人逐渐被感染并加入",
            "source_excerpt": "一开始只是点头，随后调整步伐，最终抬手模仿。",
            "micro_actions": ["点头", "调整步伐", "抬手模仿"],
        },
    ]

    assert annotate_event_motion_modes(events) is True

    assert events[1]["generation_motion_mode"] == "composite"
    assert normalize_event_action_units(events[1])["units"] == 1
    assert events[2]["generation_motion_mode"] == "atomic"
    assert normalize_event_action_units(events[2])["units"] == 3


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
    events = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(events) == 26

    _annotate_global_event_flow(events, continuity_mode="one_take")
    plan = engine._estimate_action_capacity_plan(events, 60, 12)

    assert plan["generation_action_units"] == 19
    assert plan["primary_shots"] == 4
    assert plan["minimum_material_duration"] == 70
    assert plan["material_duration"] == 75
    assert plan["maximum_material_duration"] == 78
    shots = engine.estimate_action_aware_shot_count(events, 60, 12)
    assert shots == 4


def test_evolving_model_flashmob_artifact_has_stable_capacity():
    events = json.loads(EVOLVING_FIXTURE.read_text(encoding="utf-8"))
    assert len(events) == 26

    _annotate_global_event_flow(events, continuity_mode="one_take")
    plan = engine._estimate_action_capacity_plan(events, 60, 12)

    assert {event["sequence_id"] for event in events} == {"SEQ001"}
    assert plan["generation_action_units"] == 19
    assert plan["primary_shots"] == 4
    assert plan["minimum_material_duration"] == 70
    assert plan["material_duration"] == 75
    assert plan["maximum_material_duration"] == 78


def test_sequence_fragmentation_is_rejected_before_skeleton_llm():
    events = json.loads(FIXTURE.read_text(encoding="utf-8"))

    with pytest.raises(
        ValueError,
        match=r"sequence-isolated primary shots.*packing: SEQ001=1u",
    ):
        engine.estimate_action_aware_shot_count(events, 60, 12)


def test_flashmob_one_take_has_a_feasible_four_beat_skeleton(monkeypatch):
    events = json.loads(FIXTURE.read_text(encoding="utf-8"))
    _annotate_global_event_flow(events, continuity_mode="one_take")
    source_groups = [range(1, 6), range(6, 11), range(11, 19), range(19, 27)]
    beats = [
        {
            "beat_order": index,
            "source_events": list(source_group),
            "action": "merge",
            "reason": "按连续动作容量装箱",
            "who": [],
            "where": "日本都市街头",
            "what": f"连续舞蹈段落 {index}",
            "suggested_duration": duration,
        }
        for index, (source_group, duration) in enumerate(
            zip(source_groups, [20, 20, 18, 17], strict=True),
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

    skeleton = engine._build_beat_skeleton(events, "", 75, 19, 4)

    assert engine._beat_content_loads(skeleton["beats"], events, profile) == [3, 3, 2, 2]


def test_storyboard_material_may_exceed_delivery_only_within_1_3x():
    valid = {
        "target_duration": 75,
        "delivery_target_duration": 60,
        "pre_edit_duration_ratio_limit": 1.3,
        "shots": [{"id": "S01", "duration": 75}],
    }
    issues, _ = run_l1_checks(valid, "")
    assert not {
        issue["code"] for issue in issues
    } & {
        "pre_edit_material_below_delivery_target",
        "pre_edit_material_ratio_exceeded",
    }

    too_long = {**valid, "target_duration": 79, "shots": [{"id": "S01", "duration": 79}]}
    issues, _ = run_l1_checks(too_long, "")
    assert "pre_edit_material_ratio_exceeded" in {issue["code"] for issue in issues}

    too_short = {**valid, "target_duration": 59, "shots": [{"id": "S01", "duration": 59}]}
    issues, _ = run_l1_checks(too_short, "")
    assert "pre_edit_material_below_delivery_target" in {
        issue["code"] for issue in issues
    }


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
        "shots": [{"id": "S01", "duration": 15, "micro_actions": actions}],
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
    assert run_generation_capacity_checks(storyboard) == []


# ── guard: genuinely dense sequential chains must still fail ────────────────


def test_dense_sequential_combat_chain_still_fails_when_budget_too_small():
    events = [{
        "event_role": "action_chain",
        "micro_actions": [
            "起势蓄力", "挥拳", "肘击", "膝击", "扫腿",
            "擒拿", "过肩摔", "地面压制", "锁技", "锁喉",
            "翻滚压制", "降服拍地",
        ],
    }]
    with pytest.raises(ValueError, match="delivery allows at most"):
        engine.estimate_action_aware_shot_count(events, 30, 12)


def test_generic_dense_actions_still_fail():
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
    with pytest.raises(ValueError, match="delivery allows at most"):
        engine.estimate_action_aware_shot_count(events, 60, 10)

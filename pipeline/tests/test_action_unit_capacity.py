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
)
from utils.action_units import (  # noqa: E402
    classify_micro_action,
    normalize_action_units,
    normalized_action_unit_count,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "flashmob_60s_events.json"


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
    ]:
        assert classify_micro_action(text) == "sustained", text


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
    assert result["units"] == 2  # 挥拳 + one groove cluster


# ── gate regression: 60s flash-mob script must pass ─────────────────────────


def test_flashmob_60s_script_passes_capacity_gate():
    events = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(events) == 26
    shots = engine.estimate_action_aware_shot_count(events, 60, 12)
    assert shots >= 1


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
            "擒拿", "过肩摔", "地面压制", "锁技", "降服拍地",
        ],
    }]
    with pytest.raises(ValueError, match="cannot carry the authored action detail"):
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
    with pytest.raises(ValueError, match="cannot carry the authored action detail"):
        engine.estimate_action_aware_shot_count(events, 60, 10)

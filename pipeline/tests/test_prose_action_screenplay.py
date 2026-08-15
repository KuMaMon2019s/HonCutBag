import json
import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases.phase1.adaptation_engine import (
    BATCH_EXPAND_PROMPT,
    BEAT_SKELETON_PROMPT,
    USER_PROMPT_TEMPLATE as ADAPTATION_PROMPT,
    _inherit_event_semantics,
)
from prompt.event_extractor import (
    ACTION_SCREENPLAY_CONTRACT,
    GENERAL_PROSE_CONTRACT,
    USER_PROMPT_TEMPLATE as EVENT_PROMPT,
    _annotate_global_event_flow,
    _parse_events,
)
from prompt.text_parser import SEGMENT_MAX_CHARS, parse_text
from schemas.story import StoryboardShot


PROSE_ACTION_SCRIPT = """
暴雨砸在废弃高架上，裂开的水泥路面积满黑水，远处霓虹时亮时灭，钢梁下电缆甩动。

凛握着狭长黑刃站在积水中，烬的机械左臂不断喷出白色蒸汽。

“为什么骗我？”

“我只是不想你死。”

凛骤然冲出，踏上废车引擎盖借力腾空，双手举刀从头顶劈下。

烬抬起机械臂挡住刀锋，火星迎着雨水炸开，鞋底向后滑出两道痕迹。

凛落地立刻转腰横扫，烬跳起躲开，刀锋将身后的护栏齐腰斩断。

她顺着旋转惯性用刀柄撞向烬面部，随后一拳砸向肋部，再抬膝撞向腹部。

烬挡住拳脚并抓住她的小腿向外一推，凛单手撑地旋转扫向他的膝盖。

凛重新站起由下向上斜斩，刀尖划开风衣，在胸甲上拖出一串火星。

她连续三刀追击，直刺肩膀、横切腰部，最后从下方刺向胸口。

烬扣住刀背，机械指节被高频震动磨出大片火星。

“凛，停下。”

“放手！”

凛抽刀旋身横斩，黑刃劈入混凝土立柱，裂缝迅速爬满柱体。

整根柱子断裂，上方钢梁坠落，烬冲过去抱住凛将她撞出坍塌区域。

烬半跪在地，用机械左臂撑住钢梁，护臂变形，关节迸出火花。

凛怔怔看着他，战意第一次动摇。

“为什么……”

“因为我要你活着。”

远处烟雾中传来密集金属脚步声，数十道机械身影在雨幕里靠近。

烬站到凛前方，凛沉默片刻后走到他身旁，两柄黑刃同时指向前方。
""".strip()


def _event(**overrides):
    base = {
        "who": ["凛", "烬"],
        "where": "废弃高架",
        "what": "凛进攻，烬格挡",
        "emotion": "紧张",
        "visual": "凛腾空劈下，烬用机械臂挡住",
        "time": "雨夜",
        "action_type": "conflict",
        "event_role": "action_chain",
        "source_excerpt": "凛骤然冲出，烬抬起机械臂挡住刀锋。",
        "micro_actions": ["凛冲出", "烬格挡"],
        "action_phase": "counter",
        "start_state": "凛起步，烬站定",
        "end_state": "刀锋压在机械臂上",
        "causal_link": "凛发起攻击",
        "continuity_before": "continuous",
        "continuity_subject": "凛与烬",
        "dramatic_turn": False,
        "lines": [],
    }
    base.update(overrides)
    return base


def test_parser_detects_prose_action_and_attaches_neighbor_context():
    parsed = parse_text(PROSE_ACTION_SCRIPT)

    assert parsed["document_format"] == "prose_action_screenplay"
    assert len(parsed["segments"]) >= 2
    assert max(segment["char_count"] for segment in parsed["segments"]) <= SEGMENT_MAX_CHARS
    assert all(segment["format_hint"] == "prose_action_screenplay" for segment in parsed["segments"])
    assert parsed["segments"][0]["context_before"] == ""
    assert parsed["segments"][0]["context_after"]
    assert parsed["segments"][1]["context_before"]


def test_event_prompt_uses_read_only_context_and_action_unit_contract():
    assert "严禁从前后文重复提取事件" in EVENT_PROMPT
    assert "事件不是镜头" in EVENT_PROMPT or "动作单元" in EVENT_PROMPT
    assert "2-8 个有序 micro_actions" in ACTION_SCREENPLAY_CONTRACT
    assert "氛围、说明与内心信息不得虚构肢体动作" in GENERAL_PROSE_CONTRACT
    assert "speaker 写‘未知’" in EVENT_PROMPT
    assert "continuity_before" in EVENT_PROMPT


def test_event_parser_preserves_speaker_evidence_and_action_state():
    content = "凛骤然冲出。\n“为什么骗我？”"
    payload = [_event(
        source_excerpt="凛骤然冲出。",
        lines=[{
            "speaker": "凛",
            "line": "为什么骗我？",
            "confidence": 0.92,
            "evidence": "下一段由烬回答",
        }],
    )]

    parsed = _parse_events(json.dumps(payload, ensure_ascii=False), content)

    assert parsed[0]["event_role"] == "action_chain"
    assert parsed[0]["micro_actions"] == ["凛冲出", "烬格挡"]
    assert parsed[0]["end_state"] == "刀锋压在机械臂上"
    assert parsed[0]["lines"][0]["speaker"] == "凛"
    assert parsed[0]["lines"][0]["confidence"] == 0.92


def test_event_parser_rejects_invented_dialogue():
    payload = [_event(lines=[{
        "speaker": "凛", "line": "原文里没有的台词", "confidence": 1, "evidence": "",
    }])]

    with pytest.raises(ValueError, match="逐字原文"):
        _parse_events(json.dumps(payload, ensure_ascii=False), "凛沉默地举起刀。")


def test_global_flow_assigns_stable_sequence_action_and_dialogue_ids():
    events = [
        _event(continuity_before="cut", lines=[]),
        _event(lines=[{"speaker": "凛", "line": "放手！", "confidence": 1, "evidence": "点名"}]),
        _event(where="另一处", continuity_before="continuous", micro_actions=[] , event_role="transition"),
    ]

    _annotate_global_event_flow(events)

    assert [event["sequence_id"] for event in events] == ["SEQ001", "SEQ001", "SEQ002"]
    assert [event["action_unit_id"] for event in events[:2]] == ["AU001", "AU002"]
    assert events[1]["lines"][0]["dialogue_id"] == "D001"
    assert events[2]["continuity_before"] == "cut"


def test_global_flow_repairs_cross_segment_location_wording_drift():
    events = [
        _event(
            segment_id=1,
            continuity_before="cut",
            where="废弃高架桥面，有积水与钢梁",
            end_state="凛单手撑地，烬刚后撤",
        ),
        _event(
            segment_id=2,
            continuity_before="cut",
            where="雨天废弃户外场地，地面积水",
            start_state="凛单手撑地，烬刚后撤",
            causal_link="承接上一事件的扫腿动作继续进攻",
        ),
    ]

    _annotate_global_event_flow(events)

    assert events[1]["continuity_before"] == "continuous"
    assert events[1]["sequence_id"] == events[0]["sequence_id"]
    assert events[1]["model_continuity_before"] == "cut"
    assert "cross-segment" in events[1]["continuity_repair_reason"]


def test_group_participants_are_not_promoted_to_character_assets():
    payload = [_event(who=["凛", "烬", "数十机械单位"])]

    parsed = _parse_events(json.dumps(payload, ensure_ascii=False))

    assert parsed[0]["who"] == ["凛", "烬"]
    assert parsed[0]["background_groups"] == ["数十机械单位"]


def test_adaptation_inherits_source_evidence_and_repairs_dialogue_speaker():
    events = [_event(
        continuity_before="cut",
        sequence_id="SEQ001",
        action_unit_id="AU001",
        lines=[{
            "dialogue_id": "D001", "speaker": "凛", "line": "为什么骗我？",
            "confidence": 0.93, "evidence": "相邻问答轮次",
        }],
    )]
    shots = [{
        "shot_order": 1,
        "source_events": [1],
        "dialogue": {"speaker": "烬", "line": "为什么骗我？"},
    }]

    _inherit_event_semantics(shots, events)

    assert shots[0]["boundary_before"] == "cut"
    assert shots[0]["source_sequence_ids"] == ["SEQ001"]
    assert shots[0]["source_action_unit_ids"] == ["AU001"]
    assert shots[0]["micro_actions"] == ["凛冲出", "烬格挡"]
    assert shots[0]["dialogue"]["speaker"] == "凛"
    assert shots[0]["dialogue"]["confidence"] == 0.93

    validated = StoryboardShot.model_validate(shots[0])
    assert validated.source_sequence_ids == ["SEQ001"]
    assert validated.source_action_unit_ids == ["AU001"]


def test_adaptation_prompts_preserve_action_units_and_turning_points():
    for prompt in (ADAPTATION_PROMPT, BATCH_EXPAND_PROMPT, BEAT_SKELETON_PROMPT):
        assert "action_unit" in prompt
        assert "turning_point" in prompt
    assert "同一 sequence_id" in ADAPTATION_PROMPT
    assert "micro_actions 原顺序" in BEAT_SKELETON_PROMPT

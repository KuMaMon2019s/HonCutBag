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
    _CAMERA_ANGLE_VALUES,
    _CAMERA_MOVEMENT_VALUES,
    _LIGHTING_KEY_VALUES,
    _SHOT_INTENT_VALUES,
    _SHOT_LANGUAGE_ENUM_CONTRACT,
    _SHOT_SIZE_VALUES,
    _inherit_event_semantics,
    _parse_beat_skeleton,
)
from prompt.event_extractor import (
    ACTION_SCREENPLAY_CONTRACT,
    EVENT_FLOW_SCHEMA_VERSION,
    GENERAL_PROSE_CONTRACT,
    USER_PROMPT_TEMPLATE as EVENT_PROMPT,
    _annotate_global_event_flow,
    _parse_events,
)
from prompt.text_parser import SEGMENT_MAX_CHARS, parse_text
from schemas.story import StoryboardShot
from utils.camera_motion_contracts import (
    CAMERA_MOTION_PLANNING_INSTRUCTIONS,
    CAMERA_MOVEMENT_VALUES,
    HUMAN_PERSPECTIVE_NEGATIVE,
    apply_camera_motion_contract,
    camera_motion_prompt,
)
from utils.body_action_contracts import build_body_action_contract


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
    assert "同一时刻" in ACTION_SCREENPLAY_CONTRACT
    assert "合成一条复合 micro_action" in ACTION_SCREENPLAY_CONTRACT
    assert "generation_motion_mode" in EVENT_PROMPT
    assert "一气呵成’本身不代表同时发生" in ACTION_SCREENPLAY_CONTRACT
    assert "对前文剧情的总结不是新的时间线动作" in ACTION_SCREENPLAY_CONTRACT
    assert "肩、胸、胯" not in ACTION_SCREENPLAY_CONTRACT
    assert "舞蹈词汇清单" not in ACTION_SCREENPLAY_CONTRACT
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

    parsed = _parse_events(json.dumps({"events": payload}, ensure_ascii=False), content)

    assert parsed[0]["event_role"] == "action_chain"
    assert parsed[0]["micro_actions"] == ["凛冲出", "烬格挡"]
    assert parsed[0]["generation_motion_mode"] == "atomic"
    assert parsed[0]["end_state"] == "刀锋压在机械臂上"
    assert parsed[0]["lines"][0]["speaker"] == "凛"
    assert parsed[0]["lines"][0]["confidence"] == 0.92


def test_event_parser_promotes_structured_action_performers_into_who():
    payload = [_event(
        who=["男性"],
        source_excerpt="第一名敌人挥砍，男性后仰闪避。",
        micro_actions=["第一名敌人挥砍", "男性后仰闪避"],
        body_action_choreography=[
            {
                "micro_action_index": 1,
                "performer": "第一名敌人",
                "technique": "横向挥砍",
                "side": "右",
                "limbs": ["右臂"],
                "footwork": "向前一步",
                "torso": "向左扭转",
                "weight_shift": "重心前移",
                "direction": "朝向男性",
                "contact": "刀锋擦过风衣",
                "end_pose": "右臂伸展",
            },
            {
                "micro_action_index": 2,
                "performer": "男性",
                "technique": "后仰闪避",
                "side": "后侧",
                "limbs": ["双腿", "躯干"],
                "footwork": "双脚前后错位站稳",
                "torso": "躯干向后倾斜",
                "weight_shift": "重心转移至后腿",
                "direction": "向后避开刀锋",
                "contact": "刀锋擦过风衣但未接触身体",
                "end_pose": "后腿承重的低位闪避姿态",
            },
        ],
    )]

    parsed = _parse_events(
        json.dumps({"events": payload}, ensure_ascii=False),
        "第一名敌人挥砍，男性后仰闪避。",
    )

    assert parsed[0]["who"] == ["男性", "第一名敌人"]
    assert parsed[0]["who_reconciled_from_choreography"] == ["第一名敌人"]


def test_body_choreography_structured_beat_covers_generic_action_label():
    contract = build_body_action_contract({
        "what": "二人在车厢内进行格斗",
        "micro_actions": ["男子连续闪避敌人攻击"],
        "body_action_choreography": [{
            "micro_action_index": 1,
            "micro_action": "男子连续闪避敌人攻击",
            "performer": "男子",
            "technique": "近身闪避",
            "side": "左侧",
            "limbs": ["左腿", "双臂"],
            "footwork": "左脚向侧后方撤半步",
            "torso": "躯干向左后方倾斜",
            "weight_shift": "重心转移到右腿",
            "direction": "向左后方避开横向攻击",
            "contact": "能量刃从胸前掠过但未接触身体",
            "end_pose": "右腿承重的低位警戒姿态",
        }],
    })

    assert contract is not None
    assert contract["valid"] is True
    assert contract["errors"] == []


def test_event_parser_rejects_placeholder_body_choreography_before_bad_debt():
    payload = [_event(
        what="男子在车厢内与敌人格斗",
        visual="男子连续闪避敌人的格斗攻击",
        source_excerpt="男子在车厢内连续闪避敌人攻击。",
        micro_actions=["男子连续闪避敌人攻击"],
        body_action_choreography=[{
            "micro_action_index": 1,
            "performer": "男子",
            "technique": "近身闪避",
            "side": "未明确",
            "limbs": ["双腿", "躯干", "双臂"],
            "footwork": "小幅度快速移步",
            "torso": "随攻击方向扭转",
            "weight_shift": "在双脚之间切换重心",
            "direction": "随攻击方向变换",
            "contact": "避开攻击",
            "end_pose": "低重心警戒姿态",
        }],
    )]

    with pytest.raises(ValueError, match="body choreography"):
        _parse_events(
            json.dumps({"events": payload}, ensure_ascii=False),
            "男子在车厢内连续闪避敌人攻击。",
        )


def test_event_parser_rejects_invented_dialogue():
    payload = [_event(lines=[{
        "speaker": "凛", "line": "原文里没有的台词", "confidence": 1, "evidence": "",
    }])]

    with pytest.raises(ValueError, match="逐字原文"):
        _parse_events(
            json.dumps({"events": payload}, ensure_ascii=False),
            "凛沉默地举起刀。",
        )


def test_event_parser_excludes_global_visual_directives_from_story_clock():
    directive = (
        "全程只使用同一名虚构年轻男性：同一张脸、同一发型、同一身材比例、"
        "同一套服装。摄影机保持缓慢推进，禁止人物变形、字幕、水印和 Logo。"
    )
    payload = [_event(
        who=["年轻男性"],
        where="未指定空间",
        what="建立角色一致性与摄影规范",
        visual="保持同一角色并禁止画面瑕疵",
        action_type="transition",
        event_role="scene_setup",
        source_excerpt=directive,
        micro_actions=[],
        action_phase="none",
        start_state="",
        end_state="",
        causal_link="",
        continuity_before="cut",
        continuity_subject="",
    )]

    assert _parse_events(
        json.dumps({"events": payload}, ensure_ascii=False), directive
    ) == []


def test_event_parser_excludes_fragment_from_directive_only_segment():
    directive = (
        "全程保持同一台工业设备的材质与比例。照明必须稳定，"
        "禁止标记、水印和颜色漂移。"
    )
    payload = [_event(
        who=[],
        where="未指定空间",
        what="建立稳定照明与金属表面反射",
        visual="稳定照明照亮金属表面反射",
        action_type="setup",
        event_role="scene_setup",
        source_excerpt="稳定照明与金属表面反射",
        micro_actions=[],
        action_phase="none",
        start_state="",
        end_state="",
        causal_link="",
        continuity_before="cut",
        continuity_subject="",
    )]

    assert _parse_events(
        json.dumps({"events": payload}, ensure_ascii=False), directive
    ) == []


def test_event_parser_keeps_story_action_in_mixed_directive_segment():
    source = (
        "机械臂进入工位。全程保持统一材质，禁止水印和颜色漂移。"
    )
    payload = [
        _event(
            who=["机械臂"],
            what="机械臂进入工位",
            visual="机械臂进入工位",
            event_role="action_chain",
            source_excerpt="机械臂进入工位",
            micro_actions=["机械臂进入工位"],
            generation_motion_mode="atomic",
            action_phase="setup",
        ),
        _event(
            who=[],
            what="保持统一材质",
            visual="统一材质",
            event_role="scene_setup",
            source_excerpt="统一材质",
            micro_actions=[],
            generation_motion_mode="none",
            action_phase="none",
        ),
    ]

    parsed = _parse_events(
        json.dumps({"events": payload}, ensure_ascii=False), source
    )

    assert any(event["micro_actions"] == ["机械臂进入工位"] for event in parsed)


def test_event_parser_keeps_real_scene_setup_without_visible_motion():
    setup = "深夜地下车站空无一人，冷蓝霓虹映在湿润地面。"
    payload = [_event(
        who=[],
        where="深夜地下车站",
        what="建立暴雨夜的地下车站",
        visual=setup,
        action_type="discovery",
        event_role="scene_setup",
        source_excerpt=setup,
        micro_actions=[],
        action_phase="none",
        start_state="",
        end_state="地下车站与雨夜环境建立完成",
        causal_link="",
        continuity_before="cut",
        continuity_subject="",
    )]

    parsed = _parse_events(
        json.dumps({"events": payload}, ensure_ascii=False), setup
    )

    assert len(parsed) == 1
    assert parsed[0]["event_role"] == "scene_setup"


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


def test_global_flow_keeps_explicit_one_take_in_one_sequence():
    events = [
        _event(continuity_before="cut", where="繁忙的现代日本城市街道"),
        _event(continuity_before="cut", where="镜头前方的行进路径"),
        _event(
            continuity_before="cut",
            where="街边",
            event_role="scene_setup",
            micro_actions=[],
        ),
    ]

    _annotate_global_event_flow(events, continuity_mode="one_take")

    assert [event["sequence_id"] for event in events] == ["SEQ001"] * 3
    assert [event["continuity_before"] for event in events] == [
        "cut",
        "continuous",
        "continuous",
    ]
    assert events[1]["model_continuity_before"] == "cut"
    assert "one-take" in events[1]["continuity_repair_reason"]


def test_global_flow_resolves_generic_participant_to_adjacent_specific_identity():
    events = [
        _event(
            segment_id=1,
            who=["检修员", "第一名入侵者"],
            source_excerpt="第一名入侵者被检修员推离控制台。",
            continuity_before="cut",
        ),
        _event(
            segment_id=2,
            who=["第三名入侵者", "检修员"],
            source_excerpt="第三名入侵者释放脉冲，检修员展开绝缘屏障。",
            continuity_before="continuous",
        ),
        _event(
            segment_id=3,
            who=["检修员", "入侵者"],
            source_excerpt="检修员穿过同一脉冲余波，控制最后一名入侵者的手臂。",
            causal_link="承接上一事件的脉冲余波",
            continuity_before="continuous",
        ),
    ]

    _annotate_global_event_flow(events, continuity_mode="one_take")

    assert events[2]["who"] == ["检修员", "第三名入侵者"]
    assert events[2]["model_who"] == ["检修员", "入侵者"]
    assert events[2]["who_repair_reason"] == (
        "continuous generic participant inherits the adjacent specific identity"
    )


def test_global_flow_allows_a_previously_seen_participant_to_reappear_during_coreference():
    events = [
        _event(
            who=["检修员"],
            source_excerpt="检修员进入控制舱。",
            continuity_before="cut",
        ),
        _event(
            who=["第三名入侵者"],
            source_excerpt="第三名入侵者从顶部跃下。",
            continuity_before="continuous",
        ),
        _event(
            who=["检修员", "入侵者"],
            source_excerpt="检修员闪避入侵者的重击。",
            continuity_before="continuous",
        ),
    ]

    _annotate_global_event_flow(events, continuity_mode="one_take")

    assert events[2]["who"] == ["检修员", "第三名入侵者"]
    assert events[2]["model_who"] == ["检修员", "入侵者"]
    assert events[2]["who_repair_reason"] == (
        "continuous generic participant inherits the adjacent specific identity"
    )


def test_global_flow_does_not_corefer_when_current_event_adds_an_unseen_participant():
    events = [
        _event(
            who=["第三名入侵者"],
            source_excerpt="第三名入侵者从顶部跃下。",
            continuity_before="cut",
        ),
        _event(
            who=["陌生检修员", "入侵者"],
            source_excerpt="陌生检修员进入控制舱并看向入侵者。",
            continuity_before="continuous",
        ),
    ]

    _annotate_global_event_flow(events, continuity_mode="one_take")

    assert events[1]["who"] == ["陌生检修员", "入侵者"]
    assert "model_who" not in events[1]


def test_global_flow_resolves_equivalent_human_descriptor_in_continuity():
    events = [
        _event(
            who=["男子"],
            source_excerpt="男子站在开启的车门前。",
            continuity_before="cut",
        ),
        _event(
            who=["男性", "第一名敌人"],
            source_excerpt="第一名敌人挥砍，男性后仰闪避。",
            continuity_before="continuous",
        ),
    ]

    _annotate_global_event_flow(events)

    assert events[1]["who"] == ["男子", "第一名敌人"]
    assert events[1]["model_who"] == ["男性", "第一名敌人"]
    assert events[1]["who_repair_reason"] == (
        "continuous equivalent participant inherits the adjacent identity"
    )


def test_global_flow_resolves_source_proven_forward_identity_in_continuity():
    events = [
        _event(
            who=["男性"],
            source_excerpt="年轻男性站在开启的车门前，手中握着透明芯片。",
            continuity_before="cut",
        ),
        _event(
            who=["年轻男性", "第一名战斗人员"],
            source_excerpt="第一名战斗人员冲出车厢，年轻男性后仰闪避。",
            continuity_before="continuous",
        ),
    ]

    _annotate_global_event_flow(events)

    assert events[0]["who"] == ["年轻男性"]
    assert events[0]["model_who"] == ["男性"]
    assert events[0]["who_reconciled_from_forward_continuity"] == [
        {"model_label": "男性", "source_identity": "年轻男性"}
    ]


def test_global_flow_does_not_guess_forward_identity_without_current_source_evidence():
    events = [
        _event(
            who=["男性"],
            source_excerpt="男性站在开启的车门前。",
            continuity_before="cut",
        ),
        _event(
            who=["年轻男性", "第一名战斗人员"],
            source_excerpt="年轻男性与第一名战斗人员同时进入画面。",
            continuity_before="continuous",
        ),
    ]

    _annotate_global_event_flow(events)

    assert events[0]["who"] == ["男性"]
    assert "model_who" not in events[0]


def test_global_flow_does_not_guess_ambiguous_forward_identity():
    events = [
        _event(
            who=["男性"],
            source_excerpt="年轻男性与高个男性站在开启的车门前。",
            continuity_before="cut",
        ),
        _event(
            who=["年轻男性", "高个男性"],
            source_excerpt="年轻男性与高个男性同时进入车厢。",
            continuity_before="continuous",
        ),
    ]

    _annotate_global_event_flow(events)

    assert events[0]["who"] == ["男性"]
    assert "model_who" not in events[0]


def test_global_flow_carries_exact_action_participant_from_continuity_subject():
    events = [
        _event(
            who=["男子", "第一名敌人"],
            source_excerpt="第一名敌人挥砍，男子后仰闪避。",
            continuity_before="cut",
        ),
        _event(
            who=["男性"],
            what="男性踢中第一名敌人的手腕",
            source_excerpt="男子踢中敌人的手腕。",
            continuity_before="continuous",
            continuity_subject="男性,第一名敌人",
        ),
    ]

    _annotate_global_event_flow(events)

    assert events[1]["who"] == ["男子", "第一名敌人"]
    assert events[1]["who_reconciled_from_continuity_subject"] == [
        "第一名敌人"
    ]


def test_global_flow_does_not_make_offscreen_continuity_subject_visible():
    events = [
        _event(
            who=["男子"],
            source_excerpt="男子低头查看芯片。",
            continuity_before="cut",
        ),
        _event(
            who=[],
            what="列车穿过霓虹隧道",
            source_excerpt="镜头拉远，列车穿过霓虹隧道。",
            micro_actions=[],
            action_phase="none",
            continuity_before="continuous",
            continuity_subject="男子、列车",
        ),
    ]

    _annotate_global_event_flow(events)

    assert events[1]["who"] == []
    assert "who_reconciled_from_continuity_subject" not in events[1]


def test_global_flow_does_not_merge_explicitly_new_generic_role_participant():
    events = [
        _event(
            segment_id=1,
            who=["检修员", "第一名入侵者"],
            source_excerpt="第一名入侵者被检修员推离控制台。",
            continuity_before="cut",
        ),
        _event(
            segment_id=2,
            who=["检修员", "入侵者"],
            source_excerpt="另一名入侵者从侧门进入并靠近检修员。",
            causal_link="侧门开启",
            continuity_before="continuous",
        ),
    ]

    _annotate_global_event_flow(events, continuity_mode="one_take")

    assert events[1]["who"] == ["检修员", "入侵者"]
    assert "model_who" not in events[1]


def test_one_take_mode_does_not_hide_an_explicit_time_jump():
    events = [
        _event(continuity_before="cut", where="街道"),
        _event(
            continuity_before="cut",
            where="街道",
            what="次日回到同一条街道",
        ),
    ]

    _annotate_global_event_flow(events, continuity_mode="one_take")

    assert [event["sequence_id"] for event in events] == ["SEQ001", "SEQ002"]
    assert events[1]["continuity_before"] == "cut"


def test_one_take_mode_ignores_negated_jump_constraints():
    events = [
        _event(continuity_before="cut", where="街道"),
        _event(
            continuity_before="cut",
            where="街道",
            event_role="scene_setup",
            micro_actions=[],
            what="规定一镜到底的负面约束",
            source_excerpt="不要转场。不要时间跳跃。禁止场景切换。",
        ),
    ]

    _annotate_global_event_flow(events, continuity_mode="one_take")

    assert [event["sequence_id"] for event in events] == ["SEQ001", "SEQ001"]
    assert events[1]["continuity_before"] == "continuous"


def test_segment_cache_is_invalidated_by_extractor_schema(tmp_path, monkeypatch):
    calls = 0

    def fake_call(_prompt: str, system_prompt: str) -> str:
        nonlocal calls
        calls += 1
        assert system_prompt
        return '{"events":[]}'

    segment = {
        "id": 1,
        "content": "她跃过矮墙。",
        "format_hint": "prose_action_screenplay",
        "context_before": "",
        "context_after": "",
    }
    monkeypatch.setattr("prompt.event_extractor._call_llm", fake_call)

    from prompt import event_extractor

    event_extractor.extract_events([segment], checkpoint_dir=tmp_path)
    event_extractor.extract_events([segment], checkpoint_dir=tmp_path)
    assert calls == 1

    monkeypatch.setattr(
        event_extractor,
        "EVENT_FLOW_SCHEMA_VERSION",
        EVENT_FLOW_SCHEMA_VERSION + "-next",
    )
    event_extractor.extract_events([segment], checkpoint_dir=tmp_path)

    assert calls == 2
    assert len(list((tmp_path / "phase1_event_segments").glob("*.json"))) == 2


def test_group_participants_are_not_promoted_to_character_assets():
    payload = [_event(who=["凛", "烬", "数十机械单位"])]

    parsed = _parse_events(json.dumps({"events": payload}, ensure_ascii=False))

    assert parsed[0]["who"] == ["凛", "烬"]
    assert parsed[0]["background_groups"] == ["数十机械单位"]


@pytest.mark.parametrize(
    ("event_role", "dramatic_turn"),
    [("action_chain", True), ("turning_point", False)],
)
def test_event_parser_rejects_role_and_dramatic_turn_conflicts(
    event_role,
    dramatic_turn,
):
    payload = [_event(event_role=event_role, dramatic_turn=dramatic_turn)]

    with pytest.raises(ValueError, match="event_role.*dramatic_turn"):
        _parse_events(json.dumps({"events": payload}, ensure_ascii=False))


def test_event_extractor_retries_role_and_dramatic_turn_conflict(monkeypatch):
    calls = []
    invalid = _event(event_role="action_chain", dramatic_turn=True)
    corrected = _event(event_role="action_chain", dramatic_turn=False)

    def fake_call(prompt, **_kwargs):
        calls.append(prompt)
        payload = invalid if len(calls) == 1 else corrected
        return json.dumps({"events": [payload]}, ensure_ascii=False)

    monkeypatch.setattr("prompt.event_extractor._call_llm", fake_call)
    monkeypatch.setattr("prompt.event_extractor.time.sleep", lambda _seconds: None)

    from prompt import event_extractor

    events = event_extractor._extract_events_from_segment(
        {
            "id": 1,
            "content": invalid["source_excerpt"],
            "format_hint": "prose_action_screenplay",
        }
    )

    assert len(calls) == 2
    assert "event_role 与 dramatic_turn" in calls[1]
    assert events[0]["dramatic_turn"] is False


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


def test_beat_skeleton_requires_explicit_global_shot_language():
    for field in (
        "shot_size",
        "camera_angle",
        "camera_movement",
        "lighting_key",
        "shot_intent",
        "hero_moment",
        "texture_keywords",
    ):
        assert field in BEAT_SKELETON_PROMPT

    # Both planning paths and the validator are backed by these same ordered
    # tuples. Exact contract reuse prevents their accepted vocabularies from
    # drifting apart again.
    assert _SHOT_LANGUAGE_ENUM_CONTRACT in ADAPTATION_PROMPT
    assert _SHOT_LANGUAGE_ENUM_CONTRACT in BEAT_SKELETON_PROMPT
    for values in (
        _SHOT_SIZE_VALUES,
        _CAMERA_ANGLE_VALUES,
        _CAMERA_MOVEMENT_VALUES,
        _LIGHTING_KEY_VALUES,
        _SHOT_INTENT_VALUES,
    ):
        for value in values:
            assert value in ADAPTATION_PROMPT
            assert value in BEAT_SKELETON_PROMPT

    response = json.dumps({
        "strategy": "旧格式",
        "beats": [{
            "beat_order": 1,
            "source_events": [1],
            "dropped_source_events": [],
            "action": "keep",
            "reason": "保留",
            "who": [],
            "where": "街头",
            "what": "人物前进",
            "suggested_duration": 15,
        }],
    })

    with pytest.raises(ValueError, match="缺少字段"):
        _parse_beat_skeleton(response, expected_count=1, event_count=1)


def test_beat_skeleton_enum_error_returns_actionable_legal_values():
    response = json.dumps({
        "strategy": "手持跟拍",
        "beats": [{
            "beat_order": 1,
            "source_events": [1],
            "dropped_source_events": [],
            "action": "keep",
            "reason": "保留",
            "who": [],
            "where": "街头",
            "what": "人物前进",
            "suggested_duration": 15,
            "shot_size": "medium_wide",
            "camera_angle": "eye_level",
            "camera_movement": "handheld_backward",
            "lighting_key": "natural",
            "shot_intent": "action",
            "hero_moment": False,
            "texture_keywords": ["湿润路面", "漫反射天光"],
        }],
    })

    with pytest.raises(
        ValueError,
        match=r"handheld_backward; 合法值: .*handheld.*rack_focus",
    ):
        _parse_beat_skeleton(response, expected_count=1, event_count=1)


def test_camera_motion_module_is_the_single_prompt_and_enum_contract():
    assert _CAMERA_MOVEMENT_VALUES == CAMERA_MOVEMENT_VALUES
    for value in (
        "tracking_front",
        "tracking_rear",
        "pedestal_up",
        "orbit_semicircle",
        "subtle_zoom_in",
        "dialogue_push_in",
        "dolly_in_subtle_zoom",
        "crash_zoom_in",
        "punch_in",
        "dolly_zoom_out",
        "whip_pan_left",
        "foreground_occlusion",
        "rack_focus",
    ):
        assert value in _CAMERA_MOVEMENT_VALUES
    for prompt in (ADAPTATION_PROMPT, BATCH_EXPAND_PROMPT, BEAT_SKELETON_PROMPT):
        assert CAMERA_MOTION_PLANNING_INSTRUCTIONS in prompt
        assert "每镜运动必须有稳定起点" in prompt
        assert "70% dolly + 30% zoom" in prompt


def test_camera_motion_contract_persists_physics_lens_and_negatives():
    shot = {
        "who": ["林川"],
        "shot_size": "full",
        "camera_movement": "tracking_front",
    }

    apply_camera_motion_contract(shot)

    contract = shot["camera_motion_contract"]
    assert shot["lens_mm"] == 50
    assert contract["primary_movement_count"] == 1
    assert contract["start"].startswith("stable authored full framing")
    assert "moves backward at the same natural pace" in contract["process"]
    assert "decelerate and stop" in contract["end"]
    assert "50–85mm equivalent cinematic lens" in contract["human_perspective"]
    assert HUMAN_PERSPECTIVE_NEGATIVE in contract["negative"]
    rendered = camera_motion_prompt(shot)
    assert "one primary movement only" in rendered
    assert "lens: 50mm equivalent" in rendered

    scenery = {
        "who": [],
        "shot_size": "extreme_wide",
        "camera_movement": "pan_left",
    }
    apply_camera_motion_contract(scenery)
    assert "lens_mm" not in scenery
    assert scenery["camera_motion_contract"]["human_perspective"] == ""

    combined = {
        "who": ["林川"],
        "shot_size": "medium",
        "camera_movement": "dolly_in_subtle_zoom",
    }
    apply_camera_motion_contract(combined)
    assert "70% dolly and 30% zoom" in combined["camera_motion_contract"]["process"]


def test_beat_skeleton_persists_camera_contract_before_expansion():
    response = json.dumps({
        "strategy": "前方跟拍",
        "beats": [{
            "beat_order": 1,
            "source_events": [1],
            "dropped_source_events": [],
            "action": "keep",
            "reason": "保留行走对白",
            "who": ["林川"],
            "where": "街道",
            "what": "林川迎面走来",
            "suggested_duration": 15,
            "shot_size": "wide",
            "camera_angle": "over_shoulder",
            "camera_movement": "tracking_front",
            "lighting_key": "natural",
            "shot_intent": "dialogue",
            "hero_moment": False,
            "texture_keywords": ["湿润路面", "柔和天光"],
        }],
    })

    parsed = _parse_beat_skeleton(response, expected_count=1, event_count=1)
    beat = parsed["beats"][0]

    assert beat["lens_mm"] == 50
    assert beat["camera_motion_contract"]["movement"] == "tracking_front"
    assert beat["camera_motion_contract"]["primary_movement_count"] == 1

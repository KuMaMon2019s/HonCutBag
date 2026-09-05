"""Regression coverage for executable choreography and diverse synthetic styling."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases.phase1.adaptation_engine import (
    _batch_prompt,
    _build_event_details_json,
    _build_events_json,
    _event_llm_view,
)
from phases.phase2.shot_storyboards import build_shot_storyboard_prompt
from phases.phase3 import character_factory
from phases.phase5.storyboard_qa_gate import run_generation_capacity_checks
from phases.phase6.video_generator import build_video_prompt
from quality import video_qa
from prompt.event_extractor import ACTION_SCREENPLAY_CONTRACT, USER_PROMPT_TEMPLATE, _normalize_event
from runtime.continuity_chunks import ChunkExecutionRequest
from runtime.continuity_provider import (
    _chunk_prompt,
    _chunk_scoped_shot_meta,
    _storyboard_group_prompt,
)
from schemas.continuity import GenerationChunk
from prompt.seedream_image_prompt import bind_reference_roles, prompt_guidance_metrics
from utils import privacy_visual_policy
from utils.action_kinematics import apply_generation_kinematics_projection
from utils.body_action_contracts import (
    apply_body_action_contract,
    body_action_contract_errors,
    body_action_prompt,
    build_body_action_contract,
    is_mechanically_specific_action,
)
from utils.character_body_contracts import (
    apply_adult_lead_body_contracts,
    character_reference_identity_description,
)
from utils.privacy_visual_policy import (
    SYNTHETIC_STYLIZED_CHARACTER_POLICY,
    SYNTHETIC_MAKEUP_PROFILE_ID,
    apply_synthetic_stylized_character_policy,
    synthetic_stylized_prompt_contract,
    synthetic_character_review_evidence,
    synthetic_makeup_aesthetic_profile,
    synthetic_makeup_profile_sha256,
    synthetic_makeup_reference_prompt_contract,
    synthetic_makeup_reference_qa_requirements,
)
from utils.prompt_budget import enforce_prompt_budget


def _choreography_shot() -> dict:
    shot = {
        "id": 1,
        "shot_id": "S01",
        "duration": 5,
        "where": "训练场",
        "what": "街舞与功夫对练",
        "visual": "舞者接托马斯全旋，搭档左挡右闪后以铁山靠收势",
        "who": ["舞者", "搭档"],
        "camera_movement": "tracking",
        "micro_actions": [
            "舞者左手撑地、右腿外摆启动街舞托马斯全旋",
            "搭档左臂外旋格挡后向右侧闪身",
            "搭档右脚蹬地、肩背前送完成铁山靠并沉髋收势",
        ],
        "generation_actions": [
            "舞者左手撑地、右腿外摆启动街舞托马斯全旋",
            "搭档左臂外旋格挡后向右侧闪身",
            "搭档右脚蹬地、肩背前送完成铁山靠并沉髋收势",
        ],
        "body_action_choreography": [
            {
                "micro_action_index": 1,
                "performer": "舞者",
                "technique": "街舞托马斯全旋",
                "side": "左手支撑",
                "limbs": ["左手", "右腿", "左腿"],
                "footwork": "双腿依次离地形成剪刀摆",
                "torso": "胸口下压并绕支撑肩旋转",
                "weight_shift": "双脚转移到左手与肩带",
                "direction": "逆时针",
                "contact": "左掌接触地面",
                "end_pose": "右腿扫过正前方，髋部保持离地",
            },
            {
                "micro_action_index": 2,
                "performer": "搭档",
                "technique": "左挡右闪",
                "side": "左挡、右闪",
                "limbs": ["左前臂", "双膝"],
                "footwork": "右脚向右后方撤半步",
                "torso": "躯干右倾避开来向",
                "weight_shift": "左脚转移到右脚",
                "direction": "向右后方",
                "contact": "左前臂短暂接触来势",
                "end_pose": "身体位于原轴线右侧",
            },
            {
                "micro_action_index": 3,
                "performer": "搭档",
                "technique": "铁山靠",
                "side": "右侧",
                "limbs": ["右脚", "右肩", "背部"],
                "footwork": "右脚蹬地进半步",
                "torso": "胸背整体前送而非挥臂",
                "weight_shift": "重心从后脚压到前脚",
                "direction": "向左前方",
                "contact": "右肩背训练式预接触",
                "end_pose": "沉髋屈膝稳定收势",
            },
        ],
    }
    apply_body_action_contract(shot)
    return shot


def test_action_extractor_requires_per_beat_body_mechanics_and_named_moves():
    assert "body_action_choreography" in USER_PROMPT_TEMPLATE
    assert "左右侧、肢体路径、步法、躯干旋转、重心转移" in ACTION_SCREENPLAY_CONTRACT
    assert "街舞托马斯、铁山靠" in ACTION_SCREENPLAY_CONTRACT
    assert "复杂动作" in ACTION_SCREENPLAY_CONTRACT


def test_choreography_specificity_accepts_named_or_sided_mechanics_and_rejects_placeholders():
    assert is_mechanically_specific_action("左挡")
    assert is_mechanically_specific_action("右闪")
    assert is_mechanically_specific_action("街舞托马斯全旋")
    assert is_mechanically_specific_action("铁山靠")
    assert not is_mechanically_specific_action("复杂复核动作")
    assert not is_mechanically_specific_action("双方进行激烈格斗")


def test_event_normalization_persists_structured_body_action_contract():
    event = {
        "who": ["舞者"],
        "where": "训练场",
        "what": "街舞表演",
        "emotion": "强烈",
        "visual": "舞者完成街舞托马斯",
        "time": "日间",
        "action_type": "dance",
        "event_role": "action_chain",
        "micro_actions": ["舞者左手撑地、右腿外摆启动街舞托马斯全旋"],
        "action_temporal_relations": [{
            "micro_action_index": 1,
            "performers": ["舞者"],
            "targets": [],
            "action_kind": "locomotion",
            "temporal_relation": "root",
            "reference_action_indexes": [],
            "pace": "fast",
            "state_reads": ["双脚站立"],
            "state_writes": ["左手撑地、右腿外摆"],
        }],
        "body_action_choreography": _choreography_shot()["body_action_choreography"][:1],
    }

    _normalize_event(event, "舞者完成街舞托马斯。")

    assert event["body_action_contract"]["schema"] == "honcut.body-action-choreography.v2"
    assert event["body_action_contract"]["valid"] is True
    assert event["body_action_choreography"][0]["weight_shift"]


def test_phase5_blocks_vague_dance_or_fight_placeholders_before_paid_generation():
    shot = {
        "id": 1,
        "duration": 5,
        "what": "街舞和格斗表演",
        "visual": "两人完成复杂复核动作",
        "micro_actions": ["两人完成复杂复核动作"],
        "generation_actions": ["两人完成复杂复核动作"],
        "camera_movement": "tracking",
    }

    errors = body_action_contract_errors(shot)
    issues = run_generation_capacity_checks({"shots": [shot]})

    assert {error["code"] for error in errors} >= {
        "body_choreography_vague_action",
        "body_choreography_beats_missing",
    }
    assert "body_choreography_vague_action" in {issue["code"] for issue in issues}


def test_explicit_choreography_requires_complete_mechanics_without_domain_label():
    errors = body_action_contract_errors({
        "what": "男子抓住列车扶手",
        "micro_actions": ["男子伸手抓住列车扶手"],
        "body_action_choreography": [{
            "micro_action_index": 1,
            "micro_action": "男子伸手抓住列车扶手",
            "performer": "男子",
            "technique": "抓握扶手借力",
            "side": "右侧",
            "limbs": ["手"],
            "footwork": "未明确",
            "torso": "向扶手方向转动",
            "weight_shift": "重心向扶手侧转移",
            "direction": "朝向扶手",
            "contact": "手掌接触扶手",
            "end_pose": "攥紧扶手",
        }],
    })

    assert [error["code"] for error in errors] == [
        "body_choreography_vague_action",
        "body_choreography_incomplete_beat",
    ]
    assert errors[1]["beats"][0]["missing_fields"] == ["footwork"]


def test_enriched_side_placeholder_becomes_stable_director_staging():
    record = {
        "micro_actions": ["敌人挥动能量刃横向斩击"],
        "body_action_choreography": [{
            "micro_action_index": 1,
            "micro_action": "敌人挥动能量刃横向斩击",
            "performer": "敌人",
            "technique": "横向挥砍",
            "side": "持械手臂一侧（左右原文未明确）",
            "limbs": ["持械手臂", "双腿"],
            "footwork": "前脚落地形成弓步",
            "torso": "躯干向挥砍方向转动",
            "weight_shift": "重心从后腿转移至前腿",
            "direction": "沿对手身前横向挥出",
            "contact": "刀刃擦过衣物，未命中身体",
            "end_pose": "持械手臂伸展，双脚稳定落地",
        }],
    }

    first = apply_body_action_contract(copy.deepcopy(record))
    second = apply_body_action_contract(copy.deepcopy(record))

    assert first is not None and first["valid"] is True
    assert second is not None and second["valid"] is True
    assert first["beats"][0]["side"] == second["beats"][0]["side"]
    assert first["beats"][0]["side"] in {
        "左侧（确定性导演编排）",
        "右侧（确定性导演编排）",
    }
    assert "未明确" not in first["prompt"]


def test_explicit_authored_side_is_not_replaced_by_director_staging():
    record = _choreography_shot()

    contract = apply_body_action_contract(record)

    assert contract is not None
    assert contract["beats"][0]["side"] == "左手支撑"


def test_body_contract_uses_current_action_ledger_not_fight_context():
    record = {
        "what": "近身格斗仍在持续，护盾抵御冲击波",
        "visual": "狭窄车厢内的格斗背景",
        "micro_actions": [
            "敌人释放电磁冲击",
            "男子举起透明科技芯片",
            "芯片爆发蓝色能量形成半透明护盾",
            "冲击波席卷车厢，灯光闪烁，玻璃震动，雨滴被卷起",
        ],
    }

    assert body_action_contract_errors(record) == []
    assert build_body_action_contract(record) is None


def test_non_body_effect_does_not_need_a_duplicate_choreography_beat():
    record = {
        "what": "二人在列车内格斗",
        "micro_actions": [
            "第一名敌人手中的能量武器撞击金属车壁",
            "第二名敌人从侧面突袭",
        ],
        "body_action_choreography": [{
            "micro_action_index": 2,
            "micro_action": "第二名敌人从侧面突袭",
            "performer": "第二名敌人",
            "technique": "侧面突袭",
            "side": "左侧",
            "limbs": ["双腿", "双臂"],
            "footwork": "左脚侧向突进",
            "torso": "躯干前倾",
            "weight_shift": "重心转移至前脚",
            "direction": "从左侧冲向防守者",
            "contact": "攻击被格挡，未命中身体",
            "end_pose": "攻势被拦截的前倾姿态",
        }],
    }

    assert body_action_contract_errors(record) == []


def test_non_body_rows_are_removed_from_declared_body_choreography():
    record = {
        "micro_actions": [
            "男子精准踢向敌人手腕",
            "能量武器撞击车壁处产生蓝色电弧与火花",
        ],
        "body_action_choreography": [
            {
                "micro_action_index": 1,
                "micro_action": "男子精准踢向敌人手腕",
                "performer": "男子",
                "technique": "右腿前踢",
                "side": "右侧",
                "limbs": ["右腿", "左腿"],
                "footwork": "左脚支撑，右膝抬起后伸腿",
                "torso": "躯干向左侧微倾保持平衡",
                "weight_shift": "重心完全转移至左腿",
                "direction": "右脚朝敌人持械手腕前伸",
                "contact": "右脚脚背接触敌人持械手腕",
                "end_pose": "右腿收回，左脚稳定支撑",
            },
            {
                "micro_action_index": 2,
                "micro_action": "能量武器撞击车壁处产生蓝色电弧与火花",
                "performer": "能量武器",
                "technique": "撞击车壁",
                "side": "撞击侧",
                "limbs": [],
                "footwork": "不适用",
                "torso": "不适用",
                "weight_shift": "不适用",
                "direction": "朝车壁",
                "contact": "武器接触车壁",
                "end_pose": "电弧与火花飞散",
            },
        ],
    }

    contract = apply_body_action_contract(record)

    assert contract is not None
    assert contract["valid"] is True
    assert [beat["micro_action_index"] for beat in contract["beats"]] == [1]
    assert record["micro_actions"] == [
        "男子精准踢向敌人手腕",
        "能量武器撞击车壁处产生蓝色电弧与火花",
    ]
    assert record["body_action_choreography"] == contract["beats"]


def test_non_contact_locomotion_gets_explicit_support_contact():
    record = {
        "micro_actions": ["第一名敌人高速冲刺"],
        "body_action_choreography": [{
            "micro_action_index": 1,
            "micro_action": "第一名敌人高速冲刺",
            "performer": "第一名敌人",
            "technique": "直线冲刺逼近",
            "side": "双侧",
            "limbs": ["双腿", "躯干"],
            "footwork": "双腿交替蹬地快速迈步",
            "torso": "躯干前倾保持冲刺姿态",
            "weight_shift": "重心随步伐连续向前转移",
            "direction": "沿车厢通道向前",
            "contact": "不适用",
            "end_pose": "双脚落地进入攻击距离",
        }],
    }

    contract = apply_body_action_contract(record)

    assert contract is not None
    assert contract["valid"] is True
    assert contract["beats"][0]["contact"] == "无目标接触；身体保持既有支撑接触"


def test_contact_action_still_rejects_placeholder_contact():
    record = {
        "micro_actions": ["男子右脚踢向敌人手腕"],
        "body_action_choreography": [{
            "micro_action_index": 1,
            "micro_action": "男子右脚踢向敌人手腕",
            "performer": "男子",
            "technique": "右腿前踢",
            "side": "右侧",
            "limbs": ["右腿", "左腿"],
            "footwork": "左脚支撑，右腿前伸",
            "torso": "躯干向左微倾",
            "weight_shift": "重心转移至左腿",
            "direction": "右脚朝敌人手腕前伸",
            "contact": "不适用",
            "end_pose": "右腿开始收回",
        }],
    }

    errors = body_action_contract_errors(record)

    assert [error["code"] for error in errors] == [
        "body_choreography_vague_action",
        "body_choreography_incomplete_beat",
    ]
    assert errors[1]["beats"][0]["missing_fields"] == ["contact"]


def test_legacy_action_text_is_not_promoted_to_declared_structured_choreography():
    record = {"micro_actions": ["抓住扶手"]}

    contract = apply_body_action_contract(record)

    assert contract is not None
    assert contract["required"] is False
    assert "body_action_choreography" not in record
    assert body_action_contract_errors(record) == []


def test_storyboard_and_video_prompts_keep_full_unabstracted_choreography():
    shot = _choreography_shot()
    beat = {
        "beat_id": "S01_P01",
        "position": 1,
        "duration_s": 5,
        "generation_mode": "multi_image",
        "action": shot["generation_actions"][0],
        "generation_action_units": [{
            "unit_id": "GAU001",
            "actions": [shot["generation_actions"][0]],
            "performers": ["舞者"],
            "source_event_id": 1,
            "source_micro_action_indexes": [1],
            "source_generation_unit_indexes": [1],
            "ledger_indexes": [0],
        }],
        "micro_actions": shot["micro_actions"][:1],
        "body_action_choreography": shot["body_action_choreography"][:1],
        "start_state": "舞者双脚着地",
        "end_state": "舞者左手支撑，右腿扫过正前方",
        "camera_movement": "tracking",
    }
    apply_body_action_contract(beat)
    apply_generation_kinematics_projection(beat)
    shot["storyboard_beats"] = [beat]

    board_prompt, _beats = build_shot_storyboard_prompt(
        shot, "S01", [], aspect_ratio="16:9"
    )
    video_prompt = build_video_prompt(
        shot,
        {"characters": []},
        {"shots": {"S01": {"scene_description": "训练场"}}},
        "seedance",
    )

    assert "逐拍肢体动作谱" in board_prompt
    assert "街舞托马斯全旋" in board_prompt
    assert "左手支撑" in board_prompt
    assert "重心" in body_action_prompt(shot)
    assert "[逐拍肢体动作谱｜不可摘要]" in video_prompt
    assert "铁山靠" in video_prompt


def test_chunk_prompt_renders_only_current_beat_and_keeps_audit_ledger_intact(tmp_path):
    shot = _choreography_shot()
    shot["storyboard_beats"] = []
    for position, (action, choreography) in enumerate(
        zip(
            shot["generation_actions"],
            shot["body_action_choreography"],
            strict=True,
        ),
        1,
    ):
        beat = {
            "beat_id": f"S01_P{position:02d}",
            "action": action,
            "visual": action,
            "start_state": f"第{position}拍起始",
            "end_state": f"第{position}拍终态",
            "micro_actions": [action],
            "generation_actions": [action],
            "body_action_choreography": [choreography],
        }
        apply_body_action_contract(beat)
        shot["storyboard_beats"].append(beat)
    scene = {"shots": {"S01": {"scene_description": "训练场"}}}
    full_prompt = build_video_prompt(shot, {"characters": []}, scene, "seedance")
    shot["prompt"] = full_prompt
    chunk = GenerationChunk(
        chunk_id="S01_C01",
        sequence=1,
        target_duration_s=5,
        mode="fresh",
        execution_strategy="multi_image",
        storyboard_beat_id="S01_P01",
        action_prompt=shot["generation_actions"][0],
        start_state="第1拍起始",
        end_state="第1拍终态",
    )
    request = ChunkExecutionRequest(
        resource_id="S01_C01",
        shot_id="S01",
        chunk=chunk,
        anchors={},
        output_path=tmp_path / "S01.mp4",
        previous_output_path=None,
        input_fingerprint="fixture",
        memory_context="",
    )

    scoped = _chunk_scoped_shot_meta(request, shot)
    scoped["prompt"] = build_video_prompt(
        scoped, {"characters": []}, scene, "seedance"
    )
    prompt = _chunk_prompt(request, scoped)
    budget = enforce_prompt_budget(
        prompt,
        provider="seedance",
        model="doubao-seedance-2.0-mini",
        purpose="video_generation",
    )

    assert prompt.count("[逐拍肢体动作谱｜不可摘要]") == 1
    assert prompt.count("[honcut-video-generation-contract-v2]") == 1
    assert "街舞托马斯全旋" in prompt
    assert "左挡右闪" not in prompt
    assert "铁山靠" not in prompt
    assert budget.total_chars < budget.budget.soft_chars
    assert len(scoped["body_action_choreography"]) == 1
    assert len(shot["body_action_contract"]["beats"]) == 3
    assert "左挡右闪" in full_prompt and "铁山靠" in full_prompt


def test_chunk_resolves_authoritative_beat_fields_without_leaking_hidden_ledger(tmp_path):
    shot = _choreography_shot()
    shot["subject_description"] = "舞者外观；稍后搭档执行铁山靠"
    shot["generation_action_units"] = [
        {"prompt": "街舞托马斯全旋"},
        {"prompt": "左挡右闪"},
        {"prompt": "铁山靠"},
    ]
    shot["storyboard_beats"] = []
    for position, (action, choreography) in enumerate(
        zip(
            shot["generation_actions"],
            shot["body_action_choreography"],
            strict=True,
        ),
        1,
    ):
        beat = {
            "beat_id": f"S01_P{position:02d}",
            "action": action,
            "visual": action,
            "start_state": f"第{position}拍起始",
            "end_state": f"第{position}拍终态",
            "generation_actions": [action],
            "body_action_choreography": [choreography],
        }
        apply_body_action_contract(beat)
        shot["storyboard_beats"].append(beat)
    shot["phase8_reshoot"] = {
        "round": 1,
        "issues": ["铁山靠落点错误", "当前头盔颜色漂移"],
    }
    chunk = GenerationChunk(
        chunk_id="S01_C01",
        sequence=1,
        target_duration_s=5,
        mode="fresh",
        execution_strategy="multi_image",
        storyboard_beat_id="S01_P01",
    )
    request = ChunkExecutionRequest(
        resource_id="S01_C01",
        shot_id="S01",
        chunk=chunk,
        anchors={},
        output_path=tmp_path / "S01.mp4",
        previous_output_path=None,
        input_fingerprint="fixture",
        memory_context="",
    )

    scoped = _chunk_scoped_shot_meta(request, shot)
    scoped["prompt"] = build_video_prompt(
        scoped,
        {"characters": []},
        {"shots": {"S01": {"scene_description": "训练场"}}},
        "seedance",
    )
    prompt = _chunk_prompt(request, scoped)

    assert "natural scene progression" not in prompt
    assert "第1拍起始" in prompt and "第1拍终态" in prompt
    assert "街舞托马斯全旋" in prompt
    assert "左挡右闪" not in prompt
    assert "铁山靠" not in prompt
    assert "generation_action_units" not in scoped
    assert scoped["phase8_reshoot"]["issues"] == ["当前头盔颜色漂移"]


def test_inner_beat_group_context_omits_primary_shot_end_and_later_action():
    group = {
        "group_id": "CG001",
        "beats": [{
            "shot_id": "S01",
            "start_state": "整镜起点",
            "end_state": "铁山靠收势",
            "generation_actions": ["街舞托马斯全旋", "铁山靠"],
            "storyboard_beats": [{"beat_id": "S01_P01"}, {"beat_id": "S01_P02"}],
        }],
    }
    chunk = GenerationChunk(
        chunk_id="S01_C01",
        sequence=1,
        target_duration_s=5,
        mode="fresh",
        storyboard_beat_id="S01_P01",
        start_state="左手准备撑地",
        end_state="右腿扫过正前方",
    )

    prompt = _storyboard_group_prompt(group, "S01", chunk)

    assert "左手准备撑地" in prompt
    assert "右腿扫过正前方" in prompt
    assert "铁山靠" not in prompt
    assert "整镜起点" not in prompt
    assert "only the current 1-2 actions" in prompt
    assert "No later/earlier Pxx" in prompt
    assert len(prompt) <= 300


def test_phase1_event_llm_view_excludes_derived_duplicate_contract_fields():
    event = _choreography_shot()
    event.update({
        "sequence_id": "SEQ001",
        "action_unit_id": "AU001",
        "source_excerpt": "审计原文不得重复进入 adapter prompt",
        "generation_action_units": [{"prompt": "派生动作"}],
        "prompt": "顶层派生 prompt",
        "errors": ["派生错误"],
        "forbidden": ["派生禁项"],
        "minimum_kept_primary_beat_occurrences": 2,
        "generation_action_unit_count": 3,
    })

    view = _event_llm_view(event)
    encoded = _build_events_json([event])

    assert "body_action_choreography" in view
    assert view["minimum_kept_primary_beat_occurrences"] == 2
    for duplicate in (
        "body_action_contract",
        "source_excerpt",
        "generation_action_units",
        "prompt",
        "errors",
        "forbidden",
    ):
        assert duplicate not in view
        assert f'"{duplicate}"' not in encoded


def test_phase1_batch_expansion_reuses_compact_event_view():
    event = _choreography_shot()
    event.update({
        "event_id": 7,
        "source_excerpt": "审计原文",
        "generation_action_units": [{"prompt": "派生动作"}],
        "prompt": "派生 prompt",
        "errors": ["派生错误"],
        "forbidden": ["派生禁项"],
    })
    batch = [{
        "beat_order": 1,
        "source_events": [7],
        "action": "keep",
        "suggested_duration": 5,
        "_source_event_details": [event],
    }]

    compact = _build_event_details_json([event])
    prompt = _batch_prompt(batch, "无角色", 5, 5, 0, None)

    assert '"event_id": 7' in compact
    assert '"event_id": 7' in prompt
    for duplicate in (
        "source_excerpt",
        "generation_action_units",
        '"prompt"',
        '"errors"',
        '"forbidden"',
    ):
        assert duplicate not in prompt


def test_final_video_qa_checks_every_choreography_beat_in_order():
    shot = _choreography_shot()
    captured: list[str] = []

    class FakeClient:
        def review(self, _paths, prompt):
            captured.append(prompt)
            return '{"verdict":"pass","issues":[],"confidence":0.99}'

    result = video_qa._vlm_semantic_check(
        FakeClient(),
        [video_qa.FrameSample(path="/tmp/S01_mid.jpg", timestamp=1.0, label="S01_mid")],
        {"shots": [shot]},
    )

    assert result["verdict"] == "pass"
    assert "QA must verify every beat in order" in captured[0]
    assert "街舞托马斯全旋" in captured[0]
    assert "mirrored side" in captured[0]


def test_final_video_qa_only_sends_current_batch_characters_and_props(tmp_path):
    def character(char_id, name, prop):
        return {
            "id": char_id,
            "name": name,
            "aliases": [f"{name}别名"],
            "visual_identity_policy": SYNTHETIC_STYLIZED_CHARACTER_POLICY,
            "appearance": {
                "gender": "synthetic",
                "face": f"{name}专属面部纹样",
                "clothing": f"{name}专属服装",
                "identity_props": [{"id": prop, "name": prop, "owner": char_id}],
                    "synthetic_styling": {
                        "schema": "honcut.synthetic-styling.v3",
                        "mode": "synthetic_porcelain_makeup",
                        "aesthetic_profile_id": SYNTHETIC_MAKEUP_PROFILE_ID,
                        "aesthetic_profile_sha256": synthetic_makeup_profile_sha256(),
                        "makeup_design_id": f"porcelain-{char_id}",
                    "non_human_material": "pearl bio-ceramic complexion",
                    "visible_anchors": [
                        "narrow iridescent circuit stripe from temple to cheekbone",
                        "soft luminous iris ring",
                    ],
                },
            },
        }

    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps({
            "visual_identity_policy": SYNTHETIC_STYLIZED_CHARACTER_POLICY,
            "characters": [
                character("lead", "女主", "银色发簪"),
                character("support", "支援者", "蓝色护符"),
                character("guard", "未出镜守卫", "黑曜石长枪"),
            ],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    captured: list[str] = []

    class FakeClient:
        def review(self, _paths, prompt):
            captured.append(prompt)
            return '{"verdict":"pass","issues":[],"confidence":0.99}'

    frames = [
        video_qa.FrameSample(
            path=str(tmp_path / f"S01_{label}.jpg"),
            timestamp=float(index),
            label=f"S01_{label}",
        )
        for index, label in enumerate(("first", "mid", "last"))
    ]
    result = video_qa._vlm_semantic_check(
        FakeClient(),
        frames,
        {"shots": [{
            "shot_id": "S01",
            "who": ["女主"],
            "characters": ["support"],
            "visual": "女主持银色手机向前移动",
            "interaction_props": ["银色手机"],
            "associate_assets": ["char:lead"],
        }]},
        output_dir=tmp_path,
    )

    assert result["verdict"] == "pass"
    assert "女主" in captured[0]
    assert "银色发簪" in captured[0]
    assert "支援者" in captured[0]
    assert "蓝色护符" in captured[0]
    assert "银色手机" in captured[0]
    assert "未出镜守卫" not in captured[0]
    assert "黑曜石长枪" not in captured[0]


def test_no_real_person_policy_assigns_one_persistent_porcelain_makeup_language(tmp_path):
    source = {
        "characters": [
            {"id": "lead", "name": "女主", "role": "protagonist", "appearance": {"gender": "female", "clothing": "银灰长衣"}},
            {"id": "guard", "name": "守卫", "appearance": {"gender": "male", "clothing": "黑色短衣"}},
            {"id": "dancer", "name": "舞者", "appearance": {"gender": "nonbinary", "clothing": "蓝色舞衣"}},
            {"id": "vendor", "name": "摊主", "appearance": {"gender": "male", "clothing": "红色围裙"}},
        ]
    }

    rewritten = apply_synthetic_stylized_character_policy(source)
    characters = rewritten["characters"]
    modes = [character["appearance"]["synthetic_styling"]["mode"] for character in characters]
    makeup_ids = [
        character["appearance"]["synthetic_styling"]["makeup_design_id"]
        for character in characters
    ]

    assert rewritten["visual_identity_policy"] == SYNTHETIC_STYLIZED_CHARACTER_POLICY
    assert set(modes) == {"synthetic_porcelain_makeup"}
    assert len(set(makeup_ids)) == 4
    assert all(
        len(character["appearance"]["synthetic_styling"]["visible_anchors"]) >= 2
        for character in characters
    )
    assert all("面纱" not in character["appearance"]["face"] for character in characters)
    assert all("珍珠" in character["appearance"]["face"] for character in characters)
    assert all("面颊暖意" in character["appearance"]["face"] for character in characters)
    assert all("眼神光" in character["appearance"]["face"] for character in characters)
    assert all(
        character["appearance"]["synthetic_styling"]["aesthetic_profile_id"]
        == SYNTHETIC_MAKEUP_PROFILE_ID
        for character in characters
    )
    assert all(
        character["appearance"]["synthetic_styling"]["aesthetic_profile_sha256"]
        == synthetic_makeup_profile_sha256()
        for character in characters
    )
    assert "面部必须完整可见" in synthetic_stylized_prompt_contract()
    assert "尸体般灰白" in synthetic_stylized_prompt_contract()

    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps(rewritten, ensure_ascii=False), encoding="utf-8"
    )
    evidence = synthetic_character_review_evidence(tmp_path)
    assert evidence["identity_contract_complete"] is True
    assert all(character["synthetic_styling"] for character in evidence["characters"])

    assert apply_synthetic_stylized_character_policy(rewritten) == rewritten


def test_checked_in_makeup_visual_corpus_is_structured_audit_only():
    profile = synthetic_makeup_aesthetic_profile()

    assert profile["profile_id"] == SYNTHETIC_MAKEUP_PROFILE_ID
    assert profile["instruction_boundary"] == {
        "images_are_instructions": False,
        "production_uses_structured_prompt_only": True,
        "provider_media_reference_forbidden": True,
        "identity_or_likeness_copy_forbidden": True,
        "watermark_text_logo_copy_forbidden": True,
    }
    assert len(profile["references"]) == 8
    assert len({item["sha256"] for item in profile["references"]}) == 8
    assert all(item["visual_understanding"]["usable_cues"] for item in profile["references"])
    assert all(item["visual_understanding"]["excluded_cues"] for item in profile["references"])
    assert profile["production_prompt"]["phase3_reference_priority"]
    assert profile["production_prompt"]["phase3_reference_negative"]
    assert profile["production_prompt"]["phase3_reference_qa"]
    prompt = synthetic_stylized_prompt_contract()
    assert "温润透亮" in prompt
    assert "清晰瞳孔" in prompt
    assert "reference_01" not in prompt
    assert ".png" not in prompt


def test_phase3_synthetic_reference_prompt_prioritizes_exact_face_geometry():
    source_character = {
        "id": "lead",
        "name": "主角",
        "role": "protagonist",
        "appearance": {
            "gender": "female",
            "age_range": "adult 24-30",
            "clothing": "deep gray long-sleeved training top",
        },
    }
    apply_adult_lead_body_contracts([source_character])
    character = apply_synthetic_stylized_character_policy({
        "characters": [source_character],
    })["characters"][0]
    styling = character["appearance"]["synthetic_styling"]
    contract = synthetic_makeup_reference_prompt_contract(styling)

    assert contract.startswith("[PHASE 3 SYNTHETIC IDENTITY — TOP PRIORITY]")
    assert all(anchor in contract for anchor in styling["visible_anchors"])
    assert "Inside each iris" in contract
    assert "visibly separate from eyelashes and eyeliner" in contract
    assert "长袖上衣不等于高领上衣" in " ".join(
        synthetic_makeup_reference_qa_requirements()
    )

    prompts = character_factory.build_model_reference_prompts(
        character_reference_identity_description(character),
        style="high-end stylized CGI",
        synthetic_styling=styling,
    )
    for view_name, prompt in prompts.items():
        assert prompt.index("TOP PRIORITY") < prompt.index("Static identity facts")
        assert "never invent a collar" in prompt
        assert "reference_01" not in prompt
        assert ".png" not in prompt
        transport_prompt = (
            prompt
            if view_name == "face_closeup"
            else bind_reference_roles(
                f"{character_factory.REFERENCE_WEIGHT_NOTE}. {prompt}",
                ["character_identity_only"],
            )
        )
        assert prompt_guidance_metrics(transport_prompt)[
            "over_recommended_length"
        ] is False


def test_future_makeup_visual_profile_schema_fails_closed(tmp_path, monkeypatch):
    future_profile = tmp_path / "visual_understanding.json"
    future_profile.write_text(
        json.dumps({
            "schema": "honcut.synthetic-makeup-aesthetic-profile.v99",
            "profile_id": SYNTHETIC_MAKEUP_PROFILE_ID,
        }),
        encoding="utf-8",
    )
    privacy_visual_policy._load_synthetic_makeup_aesthetic_profile.cache_clear()
    monkeypatch.setattr(
        privacy_visual_policy,
        "_SYNTHETIC_MAKEUP_PROFILE_PATH",
        future_profile,
    )

    with pytest.raises(
        privacy_visual_policy.SyntheticMakeupProfileError,
        match="unsupported",
    ):
        privacy_visual_policy.synthetic_makeup_profile_sha256()

    privacy_visual_policy._load_synthetic_makeup_aesthetic_profile.cache_clear()


def test_old_v3_makeup_profile_is_rewritten_to_current_aesthetic():
    source = {
        "visual_identity_policy": SYNTHETIC_STYLIZED_CHARACTER_POLICY,
        "characters": [{
            "id": "lead",
            "name": "主角",
            "visual_identity_policy": SYNTHETIC_STYLIZED_CHARACTER_POLICY,
            "appearance": {
                "gender": "synthetic",
                "face": "冷银无血色珍珠陶瓷皮肤",
                "clothing": "银灰长衣",
                "synthetic_styling": {
                    "schema": "honcut.synthetic-styling.v3",
                    "mode": "synthetic_porcelain_makeup",
                    "makeup_design_id": "porcelain-old",
                    "non_human_material": "冷银陶瓷",
                    "visible_anchors": ["冷银陶瓷", "电路妆纹"],
                },
            },
        }],
    }

    rewritten = apply_synthetic_stylized_character_policy(source)

    styling = rewritten["characters"][0]["appearance"]["synthetic_styling"]
    face = rewritten["characters"][0]["appearance"]["face"]
    assert styling["aesthetic_profile_id"] == SYNTHETIC_MAKEUP_PROFILE_ID
    assert styling["aesthetic_profile_sha256"] == synthetic_makeup_profile_sha256()
    assert "冷银无血色" not in face
    assert "温润透亮" in face


def test_legacy_v2_synthetic_styling_is_audit_only_not_current_identity_evidence(tmp_path):
    payload = {
        "visual_identity_policy": "synthetic_stylized_character_v2",
        "characters": [{
            "id": "lead",
            "name": "主角",
            "visual_identity_policy": "synthetic_stylized_character_v2",
            "appearance": {
                "gender": "synthetic",
                "face": "旧版机械妆",
                "clothing": "银灰长衣",
                "synthetic_styling": {
                    "schema": "honcut.synthetic-styling.v2",
                    "mode": "mechanical_makeup",
                    "non_human_material": "porcelain composite",
                    "visible_anchors": ["face tattoo", "mechanical seam"],
                },
            },
        }],
    }
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    evidence = synthetic_character_review_evidence(tmp_path)

    assert evidence["enabled"] is True
    assert evidence["identity_contract_complete"] is False


def test_current_synthetic_policy_fails_evidence_when_styling_anchors_are_missing(tmp_path):
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps(
            {
                "visual_identity_policy": SYNTHETIC_STYLIZED_CHARACTER_POLICY,
                "characters": [
                    {
                        "id": "lead",
                        "visual_identity_policy": SYNTHETIC_STYLIZED_CHARACTER_POLICY,
                        "appearance": {
                            "gender": "synthetic",
                            "face": "普通自然人脸",
                            "clothing": "银灰长衣",
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    evidence = synthetic_character_review_evidence(tmp_path)

    assert evidence["enabled"] is True
    assert evidence["identity_contract_complete"] is False

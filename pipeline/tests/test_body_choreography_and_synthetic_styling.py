"""Regression coverage for executable choreography and diverse synthetic styling."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phases.phase2.shot_storyboards import build_shot_storyboard_prompt
from phases.phase5.storyboard_qa_gate import run_generation_capacity_checks
from phases.phase6.video_generator import build_video_prompt
from quality import video_qa
from prompt.event_extractor import ACTION_SCREENPLAY_CONTRACT, USER_PROMPT_TEMPLATE, _normalize_event
from utils.body_action_contracts import (
    apply_body_action_contract,
    body_action_contract_errors,
    body_action_prompt,
    is_mechanically_specific_action,
)
from utils.privacy_visual_policy import (
    NO_REAL_PERSON_POLICY,
    apply_no_real_person_character_policy,
    no_real_person_prompt_contract,
    synthetic_character_review_evidence,
)


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
        "body_action_choreography": _choreography_shot()["body_action_choreography"][:1],
    }

    _normalize_event(event, "舞者完成街舞托马斯。")

    assert event["body_action_contract"]["schema"] == "honcut.body-action-choreography.v1"
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


def test_storyboard_and_video_prompts_keep_full_unabstracted_choreography():
    shot = _choreography_shot()
    beat = {
        "beat_id": "S01_P01",
        "position": 1,
        "duration_s": 5,
        "generation_mode": "multi_image",
        "action": shot["generation_actions"][0],
        "micro_actions": shot["micro_actions"][:1],
        "body_action_choreography": shot["body_action_choreography"][:1],
        "start_state": "舞者双脚着地",
        "end_state": "舞者左手支撑，右腿扫过正前方",
        "camera_movement": "tracking",
    }
    apply_body_action_contract(beat)
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


def test_no_real_person_policy_assigns_diverse_persistent_styling_not_uniform_helmets(tmp_path):
    source = {
        "characters": [
            {"id": "lead", "name": "女主", "role": "protagonist", "appearance": {"gender": "female", "clothing": "银灰长衣"}},
            {"id": "guard", "name": "守卫", "appearance": {"gender": "male", "clothing": "黑色短衣"}},
            {"id": "dancer", "name": "舞者", "appearance": {"gender": "nonbinary", "clothing": "蓝色舞衣"}},
            {"id": "vendor", "name": "摊主", "appearance": {"gender": "male", "clothing": "红色围裙"}},
        ]
    }

    rewritten = apply_no_real_person_character_policy(source)
    characters = rewritten["characters"]
    modes = [character["appearance"]["synthetic_styling"]["mode"] for character in characters]

    assert rewritten["visual_identity_policy"] == NO_REAL_PERSON_POLICY
    assert modes[0] == "veiled_graphic_couture"
    assert len(set(modes)) == 4
    assert all(
        len(character["appearance"]["synthetic_styling"]["visible_anchors"]) >= 2
        for character in characters
    )
    assert all("全封闭机械头盔" not in character["appearance"]["face"] for character in characters)
    assert "不得把不同角色统一改成同款头盔" in no_real_person_prompt_contract()

    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps(rewritten, ensure_ascii=False), encoding="utf-8"
    )
    evidence = synthetic_character_review_evidence(tmp_path)
    assert evidence["identity_contract_complete"] is True
    assert all(character["synthetic_styling"] for character in evidence["characters"])


def test_current_synthetic_policy_fails_evidence_when_styling_anchors_are_missing(tmp_path):
    (tmp_path / "CHARACTERS.json").write_text(
        json.dumps(
            {
                "visual_identity_policy": NO_REAL_PERSON_POLICY,
                "characters": [
                    {
                        "id": "lead",
                        "visual_identity_policy": NO_REAL_PERSON_POLICY,
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

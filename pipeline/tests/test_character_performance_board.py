"""Phase 3 run-local performance-board contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from PIL import Image

from phases.phase3.performance_reference_board import (
    CHARACTER_PERFORMANCE_BOARD_SCHEMA,
    PERFORMANCE_CELL_IDS,
    _review_performance_cell_components,
    attach_performance_guides_to_storyboard,
    build_character_performance_cell_correction_prompt,
    build_character_performance_cell_prompt,
    build_character_performance_plan,
    build_character_performance_prompt,
    generate_character_performance_board,
    performance_prompt_optimization_contract,
    validate_character_performance_board,
    validate_character_performance_guide,
)
from prompt.seedream_image_prompt import bind_reference_roles, prompt_guidance_metrics
from quality.character_performance_qa import (
    CharacterPerformanceQAError,
    review_character_performance_board,
    review_character_performance_cell,
)
from utils.canonical_visual_contracts import persist_canonical_visual_contract


def _character() -> dict:
    return {
        "id": "lead",
        "name": "领队",
        "appearance": {
            "clothing": "深蓝短外套与黑色长裤",
            "identity_props": [{
                "id": "energy_baton",
                "name": "蓝色能量短棍",
                "owner": "lead",
            }],
            "synthetic_styling": {
                "schema": "honcut.synthetic-styling.v3",
                "mode": "synthetic_porcelain_makeup",
                "makeup_design_id": "porcelain-lead",
                "non_human_material": "pearl bio-ceramic complexion",
                "visible_anchors": ["blue circuit stripe", "silver iris ring"],
            },
        },
    }


def _unit(unit_id: str, source_id: str, action: str) -> dict:
    return {
        "unit_id": unit_id,
        "source_action_unit_id": source_id,
        "source_fact_echoes": [action],
        "actions": [action],
    }


def _storyboard() -> dict:
    return {
        "shots": [{
            "id": "S01",
            "shot_intent": "action",
            "storyboard_beats": [
                {
                    "beat_id": "S01_P01",
                    "character_ids": ["lead"],
                    "action": "领队握住蓝色能量短棍格挡",
                    "source_action_unit_ids": ["AU001"],
                    "generation_action_units": [
                        _unit("GAU001", "AU001", "领队握住蓝色能量短棍格挡")
                    ],
                },
                {
                    "beat_id": "S01_P02",
                    "character_ids": ["lead"],
                    "action": "领队侧身闪避后挥动蓝色能量短棍",
                    "source_action_unit_ids": ["AU002"],
                    "generation_action_units": [
                        _unit("GAU002", "AU002", "领队侧身闪避后挥动蓝色能量短棍")
                    ],
                },
            ],
        }],
    }


def _write_reference_assets(output_dir: Path) -> None:
    projected, _contract = persist_canonical_visual_contract(
        output_dir,
        {"characters": [_character()]},
        requested_policy="synthetic_stylized_character_v3",
    )
    (output_dir / "CHARACTERS.json").write_text(
        json.dumps(projected, ensure_ascii=False),
        encoding="utf-8",
    )
    character_dir = output_dir / "characters" / "lead"
    character_dir.mkdir(parents=True)
    Image.new("RGB", (1536, 1536), (190, 200, 220)).save(
        character_dir / "reference_board.png"
    )
    Image.new("RGB", (1536, 1024), (80, 120, 210)).save(
        character_dir / "prop_detail_board.png"
    )


def _qa_payload() -> str:
    return json.dumps({
        "passed": True,
        "cells": [
            {
                "cell_id": cell_id,
                "same_character": True,
                "action_semantics_match": True,
                "action_semantics_confidence": 0.95,
                "action_semantics_evidence": [
                    "the body and prop show the declared major action relationship"
                ],
                "fine_direction_match": True,
                "pose_distinct": True,
                "clothing_consistent": True,
                "makeup_consistent": True,
                "healthy_beautiful_synthetic_styling": True,
                "no_uncanny_or_corpse_like_styling": True,
                "prop_ownership_correct": True,
                "no_extra_character": True,
                "no_text_or_layout_marks": True,
                "issues": [],
            }
            for cell_id in PERFORMANCE_CELL_IDS
        ],
        "same_single_character": True,
        "six_distinct_poses": True,
        "pose_diversity_confidence": 0.95,
        "pose_diversity_evidence": [
            "the six silhouettes use different weight shifts and prop relationships"
        ],
        "clothing_makeup_consistent": True,
        "healthy_beautiful_synthetic_styling": True,
        "props_correct": True,
        "no_extra_characters": True,
        "no_text_or_layout_marks": True,
        "issues": [],
    })


def _cell_qa_payload(
    cell_id: str,
    *,
    passed: bool = True,
    action_confidence: float = 0.95,
    action_evidence: list[str] | None = None,
    fine_direction_match: bool = True,
    pose_distinct: bool = True,
    issue: str = "",
) -> str:
    payload = {
        "cell_id": cell_id,
        "same_character": True,
        "action_semantics_match": passed,
        "action_semantics_confidence": action_confidence,
        "action_semantics_evidence": action_evidence or [
            "the visible body and prop relationship supports this action verdict"
        ],
        "fine_direction_match": fine_direction_match,
        "pose_distinct": pose_distinct,
        "clothing_consistent": True,
        "makeup_consistent": True,
        "healthy_beautiful_synthetic_styling": True,
        "no_uncanny_or_corpse_like_styling": True,
        "prop_ownership_correct": True,
        "no_extra_character": True,
        "no_text_or_layout_marks": True,
        "issues": [issue] if issue else [],
    }
    return json.dumps(payload)


def _prompt_cell_id(prompt: str) -> str:
    match = re.search(r'"cell_id":\s*"(A\d{2})"', prompt)
    assert match is not None
    return match.group(1)


class _ImageClient:
    model = "doubao-seedream-5.0-lite"

    def __init__(self) -> None:
        self.calls = []

    def image_to_image(self, *, prompt, ref_image, output_path, size):
        self.calls.append({
            "prompt": prompt,
            "ref_image": list(ref_image),
            "output_path": output_path,
            "size": size,
        })
        image = Image.new("RGB", (3072, 2048), (225, 225, 225))
        colors = [(20, 50, 90), (30, 70, 100), (40, 90, 110), (50, 110, 120), (60, 130, 130), (70, 150, 140)]
        for index, color in enumerate(colors):
            left = (index % 3) * 1024 + 160
            upper = (index // 3) * 1024 + 100
            patch = Image.new("RGB", (704, 824), color)
            image.paste(patch, (left, upper))
        image.save(output_path, format="PNG")
        return "https://provider.invalid/performance.png"


class _ReviewClient:
    def __init__(self) -> None:
        self.calls = []

    def review(self, image_paths, prompt):
        self.calls.append((list(image_paths), prompt))
        if "single performance-cell inspector" in prompt:
            return _cell_qa_payload(_prompt_cell_id(prompt))
        return _qa_payload()


class _FallbackImageClient(_ImageClient):
    def image_to_image(self, *, prompt, ref_image, output_path, size):
        self.calls.append({
            "prompt": prompt,
            "ref_image": list(ref_image),
            "output_path": output_path,
            "size": size,
        })
        dimensions = (2048, 2048) if size == "2048x2048" else (3072, 2048)
        Image.new("RGB", dimensions, (100 + len(self.calls), 130, 180)).save(
            output_path,
            format="PNG",
        )
        return "https://provider.invalid/performance.png"


class _FailThenPassReviewClient:
    def __init__(self) -> None:
        self.calls = []

    def review(self, image_paths, prompt):
        self.calls.append((list(image_paths), prompt))
        if "single performance-cell inspector" in prompt:
            return _cell_qa_payload(_prompt_cell_id(prompt))
        payload = json.loads(_qa_payload())
        if len(self.calls) == 1:
            payload["passed"] = False
            payload["cells"][0]["action_semantics_match"] = False
            payload["cells"][0]["issues"] = ["generic guard pose"]
            payload["issues"] = ["A01 action mismatch"]
        return json.dumps(payload)


class _FailTwiceThenPassReviewClient:
    def __init__(self) -> None:
        self.calls = []

    def review(self, image_paths, prompt):
        self.calls.append((list(image_paths), prompt))
        if "single performance-cell inspector" in prompt:
            cell_id = _prompt_cell_id(prompt)
            is_correction = "corrections" in str(image_paths[0])
            if cell_id in {"A02", "A03"} and not is_correction:
                return _cell_qa_payload(
                    cell_id,
                    passed=False,
                    issue="authored foot and prop direction are incorrect",
                )
            return _cell_qa_payload(cell_id)
        payload = json.loads(_qa_payload())
        if len(self.calls) == 1:
            payload["passed"] = False
            payload["cells"][0]["action_semantics_match"] = False
            payload["cells"][0]["issues"] = ["whole board action mismatch"]
            payload["issues"] = ["whole board action mismatch"]
        else:
            payload["passed"] = False
            for item in payload["cells"]:
                item["action_semantics_match"] = False
                item["issues"] = ["whole-board action verdict is intentionally unstable"]
            payload["issues"] = ["whole-board action verdict is intentionally unstable"]
        return json.dumps(payload)


class _AlwaysFailReviewClient(_FailTwiceThenPassReviewClient):
    def review(self, image_paths, prompt):
        self.calls.append((list(image_paths), prompt))
        if "single performance-cell inspector" in prompt:
            cell_id = _prompt_cell_id(prompt)
            if cell_id == "A02":
                return _cell_qa_payload(
                    cell_id,
                    passed=False,
                    issue="attack direction remains incorrect",
                )
            return _cell_qa_payload(cell_id)
        payload = json.loads(_qa_payload())
        if len(self.calls) == 1:
            payload["passed"] = False
            payload["cells"][0]["action_semantics_match"] = False
            payload["cells"][0]["issues"] = ["whole board action mismatch"]
            payload["issues"] = ["whole board action mismatch"]
        return json.dumps(payload)


class _InterruptDuringComponentQAReviewClient:
    def __init__(self) -> None:
        self.calls = []

    def review(self, image_paths, prompt):
        self.calls.append((list(image_paths), prompt))
        if "single performance-cell inspector" in prompt:
            raise RuntimeError("simulated interruption after all components were generated")
        payload = json.loads(_qa_payload())
        payload["passed"] = False
        payload["cells"][0]["action_semantics_match"] = False
        payload["cells"][0]["action_semantics_confidence"] = 0.95
        payload["cells"][0]["action_semantics_evidence"] = [
            "the visible pose is a neutral portrait"
        ]
        payload["cells"][0]["issues"] = ["whole board action mismatch"]
        payload["issues"] = ["whole board action mismatch"]
        return json.dumps(payload)


def test_plan_has_six_ordered_cells_bound_to_real_pxx_action_and_prop():
    plan = build_character_performance_plan(_storyboard(), _character())

    assert plan is not None
    assert plan["schema"] == CHARACTER_PERFORMANCE_BOARD_SCHEMA
    assert [cell["cell_id"] for cell in plan["cells"]] == list(PERFORMANCE_CELL_IDS)
    assert [cell["beat_id"] for cell in plan["cells"]] == [
        "S01_P01", "S01_P02", "S01_P01", "S01_P02", "S01_P01", "S01_P02",
    ]
    assert {cell["source_action_unit_id"] for cell in plan["cells"]} == {"AU001", "AU002"}
    assert all(cell["prop_ids"] == ["energy_baton"] for cell in plan["cells"])


def test_plan_prefers_source_facts_and_projects_generic_pose_constraints():
    storyboard = _storyboard()
    first = storyboard["shots"][0]["storyboard_beats"][0]
    first["generation_action_units"][0]["source_fact_echoes"] = [
        "领队双脚前后分开、双膝微屈站稳成战斗步架"
    ]
    first["generation_action_units"][0]["actions"] = [
        "这是一段重复同一事实并增加许多摄影、环境和结果描述的生成展开，不应进入动作板事实合同"
    ]
    second = storyboard["shots"][0]["storyboard_beats"][1]
    second["generation_action_units"][0]["source_fact_echoes"] = [
        "领队左脚向前跨出，双手将蓝色能量短棍从身体右下方向左上方挥出"
    ]

    plan = build_character_performance_plan(storyboard, _character())

    assert plan is not None
    ready = plan["cells"][0]
    attack = plan["cells"][1]
    assert ready["action_description"] == "领队双脚前后分开、双膝微屈站稳成战斗步架"
    assert "摄影" not in ready["action_description"]
    assert ready["pose_category"] == "combat_ready"
    assert ready["pose_constraints"]["stance"] == "staggered"
    assert ready["pose_constraints"]["knees"] == "bent"
    assert attack["pose_constraints"]["lead_foot"] == "left"
    assert attack["pose_constraints"]["prop_start"] == "right_lower"
    assert attack["pose_constraints"]["prop_end"] == "left_upper"


def test_six_pose_roles_specialize_only_existing_prop_action_lineage():
    storyboard = {
        "shots": [{
            "id": "S01",
            "shot_intent": "action",
            "storyboard_beats": [
                {
                    "beat_id": "S01_P01",
                    "character_ids": ["lead"],
                    "source_action_unit_ids": ["AU001", "AU002"],
                    "generation_action_units": [
                        _unit("GAU001", "AU001", "领队双脚前后分开站稳成战斗步架"),
                        _unit("GAU002", "AU002", "领队双手横握蓝色能量短棍进入戒备"),
                    ],
                },
                {
                    "beat_id": "S01_P02",
                    "character_ids": ["lead"],
                    "source_action_unit_ids": ["AU003"],
                    "generation_action_units": [
                        _unit("GAU003", "AU003", "领队右脚侧滑并后倾闪避")
                    ],
                },
                {
                    "beat_id": "S01_P03",
                    "character_ids": ["lead"],
                    "source_action_unit_ids": ["AU004", "AU005"],
                    "generation_action_units": [
                        _unit("GAU004", "AU004", "领队挥动蓝色能量短棍攻击"),
                        _unit("GAU005", "AU005", "领队竖直举棍格挡"),
                    ],
                },
            ],
        }],
    }

    plan = build_character_performance_plan(storyboard, _character())

    assert plan is not None
    assert [cell["pose_category"] for cell in plan["cells"]] == [
        "combat_ready", "evade", "attack", "prop_hold", "block", "prop_use"
    ]
    assert {cell["source_action_unit_id"] for cell in plan["cells"]} <= {
        "AU001", "AU002", "AU003", "AU004", "AU005"
    }
    assert plan["cells"][5]["source_action_unit_id"] == "AU004"
    assert plan["cells"][5]["action_description"] == "领队挥动蓝色能量短棍攻击"


def test_performance_prompts_stay_within_official_length_guidance():
    plan = build_character_performance_plan(_storyboard(), _character())
    assert plan is not None
    roles = [
        "character_identity_board_only",
        "character_prop_detail_only",
        "action_pose_schematic_only",
    ]
    board_prompt = bind_reference_roles(
        build_character_performance_prompt(_character(), plan), roles[:2]
    )
    assert prompt_guidance_metrics(board_prompt)["over_recommended_length"] is False
    for cell in plan["cells"]:
        base = bind_reference_roles(
            build_character_performance_cell_prompt(_character(), cell), roles
        )
        correction = bind_reference_roles(
            build_character_performance_cell_correction_prompt(
                _character(), cell, ["major action family is not readable"]
            ),
            roles,
        )
        assert prompt_guidance_metrics(base)["over_recommended_length"] is False
        assert prompt_guidance_metrics(correction)["over_recommended_length"] is False


def test_fine_direction_is_diagnostic_when_action_semantics_match(tmp_path):
    _write_reference_assets(tmp_path)
    pose_path = tmp_path / "pose.png"
    Image.new("RGB", (2048, 2048), (160, 170, 180)).save(pose_path)
    plan = build_character_performance_plan(_storyboard(), _character())
    assert plan is not None

    class _DiagnosticReviewer:
        def review(self, image_paths, prompt):
            return _cell_qa_payload(
                _prompt_cell_id(prompt),
                fine_direction_match=False,
                issue="camera angle makes exact anatomical side ambiguous",
            )

    result = review_character_performance_cell(
        _DiagnosticReviewer(),
        pose_path,
        identity_path=tmp_path / "characters/lead/reference_board.png",
        prop_path=tmp_path / "characters/lead/prop_detail_board.png",
        character_id="lead",
        cell=plan["cells"][0],
        synthetic_styling=_character()["appearance"]["synthetic_styling"],
    )

    assert result["passed"] is True
    assert result["action_semantics_match"] is True
    assert result["fine_direction_match"] is False


def test_low_confidence_action_mismatch_is_diagnostic_not_redraw_evidence(tmp_path):
    _write_reference_assets(tmp_path)
    pose_path = tmp_path / "pose.png"
    Image.new("RGB", (2048, 2048), (160, 170, 180)).save(pose_path)
    plan = build_character_performance_plan(_storyboard(), _character())
    assert plan is not None

    class _UncertainReviewer:
        def review(self, image_paths, prompt):
            return _cell_qa_payload(
                _prompt_cell_id(prompt),
                passed=False,
                action_confidence=0.62,
                action_evidence=["the rear hand is partly occluded in the three-quarter view"],
                issue="camera angle leaves the exact action family ambiguous",
            )

    result = review_character_performance_cell(
        _UncertainReviewer(),
        pose_path,
        identity_path=tmp_path / "characters/lead/reference_board.png",
        prop_path=tmp_path / "characters/lead/prop_detail_board.png",
        character_id="lead",
        cell=plan["cells"][0],
        synthetic_styling=_character()["appearance"]["synthetic_styling"],
    )

    assert result["passed"] is True
    assert result["action_semantics_status"] == "diagnostic_uncertain"
    assert result["blocking_fields"] == []


def test_sixty_five_percent_positive_action_confidence_is_accepted(tmp_path):
    _write_reference_assets(tmp_path)
    pose_path = tmp_path / "pose.png"
    Image.new("RGB", (2048, 2048), (160, 170, 180)).save(pose_path)
    plan = build_character_performance_plan(_storyboard(), _character())
    assert plan is not None

    class _BoundaryReviewer:
        def review(self, image_paths, prompt):
            return _cell_qa_payload(
                _prompt_cell_id(prompt),
                action_confidence=0.65,
                action_evidence=["the major body and prop relationship is recognizable"],
            )

    result = review_character_performance_cell(
        _BoundaryReviewer(),
        pose_path,
        identity_path=tmp_path / "characters/lead/reference_board.png",
        prop_path=tmp_path / "characters/lead/prop_detail_board.png",
        character_id="lead",
        cell=plan["cells"][0],
        synthetic_styling=_character()["appearance"]["synthetic_styling"],
    )

    assert result["passed"] is True
    assert result["action_semantics_status"] == "passed"
    assert result["semantic_acceptance_confidence"] == 0.65


def test_isolated_pose_distinct_is_diagnostic_because_board_owns_diversity(tmp_path):
    _write_reference_assets(tmp_path)
    pose_path = tmp_path / "pose.png"
    Image.new("RGB", (2048, 2048), (160, 170, 180)).save(pose_path)
    plan = build_character_performance_plan(_storyboard(), _character())
    assert plan is not None

    class _SingleCellReviewer:
        def review(self, image_paths, prompt):
            return _cell_qa_payload(
                _prompt_cell_id(prompt),
                pose_distinct=False,
                issue="a single isolated image cannot establish six-pose diversity",
            )

    result = review_character_performance_cell(
        _SingleCellReviewer(),
        pose_path,
        identity_path=tmp_path / "characters/lead/reference_board.png",
        prop_path=tmp_path / "characters/lead/prop_detail_board.png",
        character_id="lead",
        cell=plan["cells"][0],
        synthetic_styling=_character()["appearance"]["synthetic_styling"],
    )

    assert result["passed"] is True
    assert result["pose_distinct"] is False
    assert result["blocking_fields"] == []


def test_low_confidence_board_pose_similarity_is_diagnostic(tmp_path):
    _write_reference_assets(tmp_path)
    board_path = tmp_path / "board.png"
    Image.new("RGB", (3072, 2048), (160, 170, 180)).save(board_path)
    plan = build_character_performance_plan(_storyboard(), _character())
    assert plan is not None

    class _UncertainBoardReviewer:
        def review(self, image_paths, prompt):
            payload = json.loads(_qa_payload())
            payload["passed"] = False
            payload["six_distinct_poses"] = False
            payload["pose_diversity_confidence"] = 0.58
            payload["pose_diversity_evidence"] = [
                "two upper-body silhouettes look similar at this resolution"
            ]
            payload["issues"] = ["pose diversity is visually ambiguous"]
            return json.dumps(payload)

    result = review_character_performance_board(
        _UncertainBoardReviewer(),
        board_path,
        character_id="lead",
        cells=plan["cells"],
        synthetic_styling=_character()["appearance"]["synthetic_styling"],
    )

    assert result["passed"] is True
    assert result["pose_diversity_status"] == "diagnostic_uncertain"
    assert result["board_blocking_fields"] == []


def test_plan_normalizes_numeric_production_shot_id_to_canonical_beat_parent():
    storyboard = _storyboard()
    storyboard["shots"][0]["id"] = 1
    first = storyboard["shots"][0]["storyboard_beats"][0]
    first["source_action_unit_ids"] = []
    first["generation_action_units"][0].pop("source_action_unit_id")
    first["generation_action_units"][0].update({
        "source_event_id": 1,
        "source_generation_unit_indexes": [1],
    })
    first["timeline_assignment_ids"] = ["TA001"]
    first["timeline_assignments"] = [{
        "assignment_id": "TA001",
        "source_event_id": 1,
        "source_generation_unit_indexes": [1],
    }]

    plan = build_character_performance_plan(storyboard, _character())

    assert plan is not None
    assert {cell["parent_shot_id"] for cell in plan["cells"]} == {"S01"}
    assert plan["cells"][0]["beat_id"] == "S01_P01"
    assert plan["cells"][0]["source_action_unit_id"] == "TA001"
    assert plan["cells"][0]["source_lineage_kind"] == "timeline_assignment"


def test_plan_accepts_complete_unit_lineage_when_split_beat_summary_is_empty():
    storyboard = _storyboard()
    second = storyboard["shots"][0]["storyboard_beats"][1]
    second["source_action_unit_ids"] = []
    second["generation_action_units"] = [
        _unit("GAU002", "AU001", "领队左脚向前跨步并挥动蓝色能量短棍"),
        _unit("GAU003", "AU001", "领队竖直举起同一根蓝色能量短棍格挡"),
    ]

    plan = build_character_performance_plan(storyboard, _character())

    assert plan is not None
    p02_cells = [cell for cell in plan["cells"] if cell["beat_id"] == "S01_P02"]
    assert p02_cells
    assert {cell["source_action_unit_id"] for cell in p02_cells} == {"AU001"}
    assert {tuple(cell["generation_action_unit_ids"]) for cell in p02_cells} >= {
        ("GAU002",),
        ("GAU003",),
    }
    actions_by_unit = {
        tuple(cell["generation_action_unit_ids"]): cell["action_description"]
        for cell in p02_cells
    }
    assert "格挡" not in actions_by_unit[("GAU002",)]
    assert "挥动" not in actions_by_unit[("GAU003",)]
    assert all(
        cell["source_lineage_kind"] == "source_action_unit"
        for cell in p02_cells
    )


def test_plan_rejects_partially_missing_unit_lineage_even_without_summary():
    storyboard = _storyboard()
    second = storyboard["shots"][0]["storyboard_beats"][1]
    second["source_action_unit_ids"] = []
    second["generation_action_units"].append(
        {"unit_id": "GAU003", "source_fact_echoes": ["领队竖直举棍格挡"]}
    )

    try:
        build_character_performance_plan(storyboard, _character())
    except ValueError as exc:
        assert "inconsistent source action-unit lineage" in str(exc)
    else:
        raise AssertionError("partial unit lineage must fail closed")


def test_plan_rejects_mismatched_timeline_assignment_fallback():
    storyboard = _storyboard()
    first = storyboard["shots"][0]["storyboard_beats"][0]
    first["source_action_unit_ids"] = []
    first["generation_action_units"][0].pop("source_action_unit_id")
    first["generation_action_units"][0].update({
        "source_event_id": 1,
        "source_generation_unit_indexes": [1],
    })
    first["timeline_assignment_ids"] = ["TA001"]
    first["timeline_assignments"] = [{
        "assignment_id": "TA001",
        "source_event_id": 2,
        "source_generation_unit_indexes": [1],
    }]

    try:
        build_character_performance_plan(storyboard, _character())
    except ValueError as exc:
        assert "timeline assignment lineage is inconsistent" in str(exc)
    else:
        raise AssertionError("mismatched canonical lineage must fail closed")


def test_prompt_optimization_is_frozen_offline_and_covers_every_dimension():
    first = performance_prompt_optimization_contract()
    second = performance_prompt_optimization_contract()

    assert first == second
    assert first["schema"] == "honcut.character-performance-prompt-optimization.v3"
    assert first["method"] == "offline_contract_candidate_comparison"
    assert first["provider_request_count"] == 0
    assert first["production_auto_optimization"] is False
    assert first["selected_candidate_id"] == "lineage_first_synthetic_v1"
    selected = next(
        item
        for item in first["candidates"]
        if item["candidate_id"] == first["selected_candidate_id"]
    )
    assert selected["covered_dimensions"] == first["evaluation_dimensions"]
    assert selected["contract_coverage_score"] == 7


def test_over_guidance_board_prompt_fails_before_paid_image_call(tmp_path):
    _write_reference_assets(tmp_path)
    storyboard = _storyboard()
    storyboard["shots"][0]["storyboard_beats"][0]["generation_action_units"][0][
        "source_fact_echoes"
    ] = ["attack " * 700]
    image_client = _ImageClient()

    try:
        generate_character_performance_board(
            tmp_path,
            storyboard,
            _character(),
            image_client=image_client,
            review_client=_ReviewClient(),
        )
    except CharacterPerformanceQAError as exc:
        assert "exceeds Ark prompt guidance before Provider submission" in str(exc)
    else:
        raise AssertionError("over-guidance prompt must fail closed")

    assert image_client.calls == []


def test_generate_board_and_current_pxx_guides_are_exactly_cached(tmp_path):
    _write_reference_assets(tmp_path)
    image_client = _ImageClient()
    review_client = _ReviewClient()

    first = generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=image_client,
        review_client=review_client,
    )
    second = generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=image_client,
        review_client=review_client,
    )

    assert first is not None and first["provider_requests"] == 1
    assert second is not None and second["provider_requests"] == 0
    assert len(image_client.calls) == 1
    assert len(review_client.calls) == 1
    receipt = json.loads(
        (tmp_path / "characters/lead/performance_reference_board.json").read_text()
    )
    assert receipt["prompt_guidance"]["over_recommended_length"] is False
    assert receipt["prompt_guidance"]["sha256"] == receipt["prompt_sha256"]
    assert "no text" in image_client.calls[0]["prompt"].casefold()
    assert "wet/torn/dirty clothing" in image_client.calls[0]["prompt"]
    assert image_client.calls[0]["size"] == "3072x2048"
    assert validate_character_performance_board(tmp_path, "lead")

    p01 = validate_character_performance_guide(tmp_path, "lead", "S01_P01")
    p02 = validate_character_performance_guide(tmp_path, "lead", "S01_P02")
    assert p01 is not None and p01["cell_ids"] == ["A01", "A03", "A05"]
    assert p02 is not None and p02["cell_ids"] == ["A02", "A04", "A06"]
    assert p01["layout"] == {
        "kind": "balanced_local_repack",
        "columns": 2,
        "rows": 2,
        "cell_order": ["A01", "A03", "A05"],
        "empty_slots": 1,
    }
    assert p01["pixel_size"] == p02["pixel_size"] == [2048, 2048]
    assert p01["aspect_ratio"] == p02["aspect_ratio"] == 1.0
    assert p01["provider_requests"] == p02["provider_requests"] == 0
    assert set(p01["source_action_unit_ids"]) == {"AU001"}
    assert set(p02["source_action_unit_ids"]) == {"AU002"}
    assert len(p01["source_action_unit_ids"]) == len(set(p01["source_action_unit_ids"]))
    assert len(p02["source_action_unit_ids"]) == len(set(p02["source_action_unit_ids"]))

    storyboard = _storyboard()
    attach_performance_guides_to_storyboard(storyboard, [first])
    beats = storyboard["shots"][0]["storyboard_beats"]
    assert all(beat["character_performance_required"] is True for beat in beats)
    assert beats[0]["character_performance_guides"][0]["cell_ids"] == [
        "A01", "A03", "A05"
    ]
    assert beats[1]["character_performance_guides"][0]["cell_ids"] == [
        "A02", "A04", "A06"
    ]


def test_legacy_or_tampered_performance_guide_fails_closed(tmp_path):
    _write_reference_assets(tmp_path)
    generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=_ImageClient(),
        review_client=_ReviewClient(),
    )
    receipt_path = tmp_path / "performance_guides/S01_P01/lead.json"
    original = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutations = (
        {"schema": "honcut.character-performance-guide.v1"},
        {"layout": {**original["layout"], "columns": 3}},
        {"pixel_size": [3072, 1024]},
        {"aspect_ratio": 3.0},
        {"cell_ids": ["A01", "A01", "A05"]},
    )

    for mutation in mutations:
        receipt_path.write_text(
            json.dumps({**original, **mutation}),
            encoding="utf-8",
        )
        assert validate_character_performance_guide(
            tmp_path,
            "lead",
            "S01_P01",
        ) is None


def test_failed_whole_board_uses_six_cached_components_then_passes(tmp_path):
    _write_reference_assets(tmp_path)
    image_client = _FallbackImageClient()
    review_client = _FailThenPassReviewClient()

    generated = generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=image_client,
        review_client=review_client,
    )
    reused = generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=image_client,
        review_client=review_client,
    )

    assert generated is not None and generated["provider_requests"] == 7
    assert reused is not None and reused["provider_requests"] == 0
    assert len(image_client.calls) == 7
    assert [call["size"] for call in image_client.calls] == [
        "3072x2048",
        *("2048x2048" for _ in range(6)),
    ]
    receipt = json.loads(
        (tmp_path / "characters/lead/performance_reference_board.json").read_text()
    )
    assert receipt["generation_mode"] == "per_cell_fallback"
    assert receipt["provider_request_count"] == 7
    assert [item["cell_id"] for item in receipt["component_cells"]] == list(
        PERFORMANCE_CELL_IDS
    )
    for component in receipt["component_cells"]:
        pose_reference = next(
            item
            for item in component["references"]
            if item["kind"] == "action_pose_schematic"
        )
        pose_receipt = json.loads(
            (tmp_path / pose_reference["path"]).with_suffix(".json").read_text()
        )
        assert pose_receipt["schema"] == "honcut.character-performance-pose-guide.v2"
        assert pose_receipt["provider_requests"] == 0
        assert (tmp_path / component["image"]).with_suffix(".qa.json").is_file()


def test_acceptance_mode_stops_after_failed_whole_board_without_correction(
    tmp_path,
):
    _write_reference_assets(tmp_path)
    image_client = _FallbackImageClient()

    with pytest.raises(
        CharacterPerformanceQAError,
        match="Provider corrections are disabled",
    ):
        generate_character_performance_board(
            tmp_path,
            _storyboard(),
            _character(),
            image_client=image_client,
            review_client=_FailThenPassReviewClient(),
            allow_provider_corrections=False,
        )

    assert len(image_client.calls) == 1
    receipt = json.loads(
        (tmp_path / "characters/lead/performance_reference_board.json").read_text()
    )
    assert receipt["status"] == "failed"
    assert receipt["generation_mode"] == "whole_board"
    assert receipt["provider_request_count"] == 1
    assert receipt["provider_corrections"] == "disabled"


def test_pending_component_qa_resume_reuses_all_images_without_whole_board_replay(
    tmp_path,
):
    _write_reference_assets(tmp_path)
    image_client = _FallbackImageClient()
    try:
        generate_character_performance_board(
            tmp_path,
            _storyboard(),
            _character(),
            image_client=image_client,
            review_client=_InterruptDuringComponentQAReviewClient(),
        )
    except RuntimeError as exc:
        assert "simulated interruption" in str(exc)
    else:
        raise AssertionError("component-QA interruption must propagate")
    receipt = json.loads(
        (tmp_path / "characters/lead/performance_reference_board.json").read_text()
    )
    assert receipt["status"] == "pending"
    assert receipt["generation_mode"] == "per_cell_fallback"
    assert len(image_client.calls) == 7

    recovered = generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=image_client,
        review_client=_ReviewClient(),
    )

    assert recovered is not None
    assert recovered["provider_requests"] == 0
    assert len(image_client.calls) == 7


def test_failed_components_redraw_only_failed_cells_once_with_qa_feedback(tmp_path):
    _write_reference_assets(tmp_path)
    image_client = _FallbackImageClient()
    review_client = _FailTwiceThenPassReviewClient()

    generated = generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=image_client,
        review_client=review_client,
    )
    reused = generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=image_client,
        review_client=review_client,
    )

    assert generated is not None and generated["provider_requests"] == 9
    assert reused is not None and reused["provider_requests"] == 0
    assert len(image_client.calls) == 9
    assert len(review_client.calls) == 11
    correction_calls = image_client.calls[-2:]
    assert all("one allowed correction redraw" in call["prompt"] for call in correction_calls)
    assert all("Anatomical" in call["prompt"] for call in correction_calls)
    receipt = json.loads(
        (tmp_path / "characters/lead/performance_reference_board.json").read_text()
    )
    assert receipt["generation_mode"] == "per_cell_correction"
    assert receipt["provider_request_count"] == 9
    final_qa = json.loads(
        (tmp_path / "characters/lead/performance_reference_board_qa.json").read_text()
    )
    assert final_qa["action_verdict_source"] == "isolated_persisted_cells"
    assert final_qa["board_verdict_source"] == "whole_board_global_fields_only"
    assert final_qa["passed"] is True
    assert receipt["correction_attempts"] == [
        {
            **receipt["correction_attempts"][0],
            "round": 1,
            "status": "passed",
            "target_cell_ids": ["A02", "A03"],
            "provider_requests": 2,
        }
    ]
    by_id = {item["cell_id"]: item for item in receipt["component_cells"]}
    assert by_id["A01"]["image"].endswith("performance_cells/A01.png")
    assert by_id["A02"]["image"].endswith(
        "performance_cells/corrections/round_01/A02.png"
    )
    assert by_id["A03"]["image"].endswith(
        "performance_cells/corrections/round_01/A03.png"
    )
    assert (tmp_path / "characters/lead/performance_cells/A02.png").is_file()
    assert (
        tmp_path / "characters/lead/performance_reference_board.per_cell_fallback.png"
    ).is_file()


def test_failed_single_correction_round_is_terminal_and_never_replayed(tmp_path):
    _write_reference_assets(tmp_path)
    image_client = _FallbackImageClient()
    review_client = _AlwaysFailReviewClient()

    try:
        generate_character_performance_board(
            tmp_path,
            _storyboard(),
            _character(),
            image_client=image_client,
            review_client=review_client,
        )
    except Exception as exc:
        assert "failed bounded correction QA" in str(exc)
    else:
        raise AssertionError("failed bounded correction must block")
    request_count = len(image_client.calls)

    try:
        generate_character_performance_board(
            tmp_path,
            _storyboard(),
            _character(),
            image_client=image_client,
            review_client=review_client,
        )
    except Exception as exc:
        assert "already failed blocking QA" in str(exc)
    else:
        raise AssertionError("consumed correction must remain terminal")
    assert len(image_client.calls) == request_count == 8


def test_wet_or_damaged_state_alone_does_not_create_performance_board():
    storyboard = {
        "shots": [{
            "id": "S01",
            "shot_intent": "atmosphere",
            "storyboard_beats": [{
                "beat_id": "S01_P01",
                "character_ids": ["lead"],
                "action": "领队站在雨中，风衣已经淋湿并有泥污",
                "source_action_unit_ids": ["AU001"],
                "generation_action_units": [
                    _unit("GAU001", "AU001", "领队站在雨中，风衣已经淋湿并有泥污")
                ],
            }],
        }],
    }
    character = _character()
    character["appearance"]["identity_props"] = []

    assert build_character_performance_plan(storyboard, character) is None


def test_corrupt_or_future_board_receipt_fails_closed(tmp_path):
    _write_reference_assets(tmp_path)
    generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=_ImageClient(),
        review_client=_ReviewClient(),
    )
    receipt_path = tmp_path / "characters/lead/performance_reference_board.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema"] = "honcut.character-performance-board.v999"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert not validate_character_performance_board(tmp_path, "lead")
    assert validate_character_performance_guide(tmp_path, "lead", "S01_P01") is None


def test_old_board_and_cell_qa_schemas_fail_closed(tmp_path):
    _write_reference_assets(tmp_path)
    generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=_FallbackImageClient(),
        review_client=_FailThenPassReviewClient(),
    )
    receipt_path = tmp_path / "characters/lead/performance_reference_board.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    current_receipt = dict(receipt)
    receipt["schema"] = "honcut.character-performance-board.v1"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert not validate_character_performance_board(tmp_path, "lead")

    receipt_path.write_text(json.dumps(current_receipt), encoding="utf-8")
    first_component = current_receipt["component_cells"][0]
    qa_path = (tmp_path / first_component["image"]).with_suffix(".qa.json")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["schema"] = "honcut.character-performance-cell-qa.v1"
    qa_path.write_text(json.dumps(qa), encoding="utf-8")
    assert not validate_character_performance_board(tmp_path, "lead")


def test_known_legacy_cell_qa_is_audited_then_rereviewed_without_image_call(tmp_path):
    _write_reference_assets(tmp_path)
    image_client = _FallbackImageClient()
    generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=image_client,
        review_client=_FailThenPassReviewClient(),
    )
    board_receipt = json.loads(
        (tmp_path / "characters/lead/performance_reference_board.json").read_text()
    )
    components = board_receipt["component_cells"]
    qa_path = (tmp_path / components[0]["image"]).with_suffix(".qa.json")
    legacy = json.loads(qa_path.read_text())
    legacy["schema"] = "honcut.character-performance-cell-qa.v3"
    qa_path.write_text(json.dumps(legacy))
    review_client = _ReviewClient()
    image_call_count = len(image_client.calls)

    results = _review_performance_cell_components(
        tmp_path,
        "lead",
        board_receipt["plan"],
        components,
        board_receipt["references"],
        review_client=review_client,
        synthetic_styling=_character()["appearance"]["synthetic_styling"],
    )

    assert len(results) == 6
    assert all(result["passed"] is True for result in results)
    assert len(image_client.calls) == image_call_count
    assert len(review_client.calls) == 0
    assert results[0]["qa_observation_reused"] is True
    assert qa_path.with_name("A01.qa.audit.v3.json").is_file()

def test_missing_pose_guide_or_future_cell_qa_fails_closed(tmp_path):
    _write_reference_assets(tmp_path)
    generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=_FallbackImageClient(),
        review_client=_FailThenPassReviewClient(),
    )
    board_receipt = json.loads(
        (tmp_path / "characters/lead/performance_reference_board.json").read_text()
    )
    first_component = board_receipt["component_cells"][0]
    pose_reference = next(
        item
        for item in first_component["references"]
        if item["kind"] == "action_pose_schematic"
    )
    pose_path = tmp_path / pose_reference["path"]
    pose_path.unlink()
    assert not validate_character_performance_board(tmp_path, "lead")

    # Restore exact local pose evidence, then prove an unknown future cell-QA
    # schema still cannot satisfy the production cache.
    pose_receipt_path = pose_path.with_suffix(".json")
    pose_receipt_path.unlink()
    generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=_FallbackImageClient(),
        review_client=_ReviewClient(),
    )
    cell_qa_path = (
        tmp_path / first_component["image"]
    ).with_suffix(".qa.json")
    cell_qa = json.loads(cell_qa_path.read_text())
    cell_qa["schema"] = "honcut.character-performance-cell-qa.v999"
    cell_qa_path.write_text(json.dumps(cell_qa))
    assert not validate_character_performance_board(tmp_path, "lead")


def test_future_prompt_optimization_receipt_cannot_satisfy_cache(tmp_path):
    _write_reference_assets(tmp_path)
    generate_character_performance_board(
        tmp_path,
        _storyboard(),
        _character(),
        image_client=_ImageClient(),
        review_client=_ReviewClient(),
    )
    receipt_path = tmp_path / "characters/lead/performance_reference_board.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["prompt_optimization"]["schema"] = (
        "honcut.character-performance-prompt-optimization.v999"
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert not validate_character_performance_board(tmp_path, "lead")
    assert validate_character_performance_guide(tmp_path, "lead", "S01_P01") is None

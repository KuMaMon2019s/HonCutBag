from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import subprocess
import threading
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from clients import seedance_client, tos_uploader
from clients.seedream_client import (
    IMAGE_ENDPOINT,
    SeedreamAPIError,
    SeedreamClient,
    _reference_data_url,
)
from phases import pipeline_core
from phases.phase1.director_storyboard import (
    build_director_storyboard_prompt,
    generate_director_storyboard,
    materialize_director_panels,
)
from phases.phase1.storyboard_beats import plan_storyboard_beats
from phases.phase2.shot_storyboards import (
    generate_shot_storyboards,
    validate_shot_storyboard_artifacts,
)
from phases.phase4.continuity_plan import (
    build_continuity_plan,
    write_continuity_plan,
    write_storyboard_groups,
)
from phases.phase5.storyboard_qa_gate import run_generation_capacity_checks
from phases.phase8 import edit_decisions as edit_decision_module
from phases.phase8.continuity_adjudication import (
    SEAM_DECISIONS_KIND,
    _object_evidence_needs_retry,
    adjudicate_continuity_seams,
    decide_temporal_seam,
)
from phases.phase8.frame_analysis import decide_shot_action, measure_motion_activity
from phases.phase9.rhythm_editor import (
    _duration_preserving_speed_map,
    _probe_duration,
    _write_delivery_timeline,
    apply_speed_ramp,
)
from quality import object_trajectory as object_trajectory_module
from quality import sam3_sidecar as sam3_sidecar_module
from quality import video_qa
from quality.continuity_bridge import detect_replayed_prefix, repair_continuity_boundary
from quality.continuity_seam import (
    compare_frame_sequences,
    extract_ordered_video_frames,
    extract_video_tail_frame,
    extract_video_tail_window,
    measure_video_replay_similarity,
    measure_video_seam,
)
from quality.object_trajectory import decide_object_trajectory
from quality.seam_calibration import calibrate_seam_policy, decide_seam
from runtime.continuity_chunks import (
    ChunkExecutionRequest,
    ChunkExecutionResult,
    execute_continuity_plan,
    write_shadow_runtime_report,
)
from runtime.continuity_memory import (
    initialize_continuity_memory,
    record_recent_motion,
    render_continuity_memory_context,
    select_memory_keyframes,
)
from runtime.continuity_provider import (
    _base_content,
    _bridge_seedance_executor,
    _chunk_duration,
    _continuity_bridge_preparer,
    _direct_seedance_executor,
    _generation_seed,
    _provider_content,
    _render_privacy_safe_handle_bridge,
    _seedance_reference_image_payload,
    execute_phase6_auto_continuity,
    finalize_continuity_shot,
    materialize_continuity_shot,
    normalize_provider_minimum_padding,
    probe_continuity_frames,
)
from runtime.execution_errors import ProviderJobFailedError
from runtime.generation_tasks import GenerationTaskStore
from sam3_runtime.policy import (
    estimate_weight_bytes,
    resolve_checkpoint_path,
    resolve_runtime_policy,
)
from schemas.continuity import ContinuityPlan, GenerationChunk
from utils.artifact_chain import can_resume_from


def _write_grid_image(
    path: Path,
    *,
    size: tuple[int, int],
    vertical: list[int],
    horizontal: list[int],
) -> None:
    width, height = size
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    for position in vertical:
        pixels[:, max(0, position - 3):min(width, position + 4)] = 0
    for position in horizontal:
        pixels[max(0, position - 3):min(height, position + 4), :] = 0
    Image.fromarray(pixels).save(path)


def test_seedream_compacts_large_reference_in_memory_without_touching_source(tmp_path):
    reference = tmp_path / "large-reference.png"
    Image.new("RGB", (2560, 1440), "navy").save(reference)
    original = reference.read_bytes()

    data_url = _reference_data_url(str(reference))

    header, encoded = data_url.split(",", 1)
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as compacted:
        assert header == "data:image/jpeg;base64"
        assert max(compacted.size) == 1600
    assert reference.read_bytes() == original


def test_seedream_http_error_keeps_provider_code_and_request_id(monkeypatch, tmp_path):
    error_payload = {
        "error": {
            "code": "InputImageSensitiveContentDetected",
            "message": "The input image may contain sensitive information.",
        }
    }

    class FakeResponse:
        def __init__(self):
            self.status_code = 400
            self.headers = {"x-request-id": "request-sensitive-123"}
            self.text = json.dumps(error_payload)
            self.request = None

        @staticmethod
        def json():
            return error_payload

    monkeypatch.setenv("SEEDREAM_MIN_INTERVAL", "0")
    monkeypatch.setattr(
        "clients.seedream_client.requests.post",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    client = SeedreamClient(api_key="test-key")

    with pytest.raises(SeedreamAPIError) as captured:
        client.text_to_image(
            "synthetic storyboard",
            output_path=str(tmp_path / "unused.png"),
        )

    assert captured.value.status_code == 400
    assert captured.value.provider_code == "InputImageSensitiveContentDetected"
    assert captured.value.request_id == "request-sensitive-123"
    assert "InputImageSensitiveContentDetected" in str(captured.value)


def test_planner_keeps_short_editorial_shots_backward_compatible():
    plan = build_continuity_plan({"shots": [{"id": 1, "duration": 8, "where": "roof"}]})

    shot = plan.shots[0]
    assert shot.shot_id == "S01"
    assert shot.boundary_before == "cut"
    assert [chunk.model_dump() for chunk in shot.chunks] == [
        {
            "chunk_id": "S01_C01",
            "sequence": 1,
            "target_duration_s": 8.0,
            "requested_frames": 192,
            "expected_overlap_frames": 0,
            "expected_unique_frames": 192,
            "mode": "fresh",
            "depends_on": None,
        }
    ]


def test_planner_splits_long_shot_and_preserves_explicit_anchors(tmp_path):
    storyboard = {
        "shots": [
            {"id": "S02", "duration": 6},
            {
                "id": "S03",
                "duration": 38,
                "boundary_before": "continuous",
                "who": ["CHAR_01"],
                "where": "SCENE_02",
                "screen_direction": "left_to_right",
                "camera_movement": "tracking_right",
            },
        ]
    }
    scene_contract = {"shots": {"S03": {"style_anchor": "STYLE_MAIN"}}}

    plan = write_continuity_plan(tmp_path / "CONTINUITY_PLAN.json", storyboard, scene_contract)

    shot = plan.shots[1]
    assert shot.boundary_before == "continuous"
    assert shot.anchors.model_dump() == {
        "characters": ["CHAR_01"],
        "scene": "SCENE_02",
        "screen_direction": "left_to_right",
        "camera_motion": "tracking_right",
        "style": "STYLE_MAIN",
        "tracking_prompt": "CHAR_01",
    }
    assert [chunk.target_duration_s for chunk in shot.chunks] == [13, 13, 12]
    assert shot.continuity_group_id == "CG001"
    assert shot.extends_from_shot_id == "S02"
    assert shot.extends_from_chunk_id == "S02_C01"
    assert [chunk.mode for chunk in shot.chunks] == [
        "native_extend", "native_extend", "native_extend"
    ]
    assert [chunk.depends_on for chunk in shot.chunks] == [
        "S02_C01", "S03_C01", "S03_C02"
    ]
    persisted = json.loads((tmp_path / "CONTINUITY_PLAN.json").read_text())
    assert persisted["version"] == 1
    assert persisted["timeline_fps"] == 24
    assert persisted["shots"][1]["chunks"][2]["mode"] == "native_extend"


def test_planner_balances_a_sixteen_second_shot_without_a_one_second_tail():
    plan = build_continuity_plan({"shots": [{"id": 1, "duration": 16}]})

    assert [chunk.target_duration_s for chunk in plan.shots[0].chunks] == [8, 8]


def test_planner_carries_an_explicit_subject_prompt_into_tracking_anchors():
    plan = build_continuity_plan(
        {
            "shots": [
                {
                    "id": "S01",
                    "duration": 8,
                    "tracking_prompt": "small cobalt-blue paper boat",
                }
            ]
        }
    )

    assert plan.shots[0].anchors.tracking_prompt == "small cobalt-blue paper boat"


def test_planner_reserves_replayed_reference_frames_without_shortening_the_shot():
    plan = build_continuity_plan(
        {"shots": [{"id": "S01", "duration": 10}]},
        provider_chunk_limit_s=5,
        continuation_overlap_s=2,
    )

    shot = plan.shots[0]
    assert shot.target_frames == 240
    assert [chunk.target_duration_s for chunk in shot.chunks] == [5, 5, 4]
    assert [chunk.requested_frames for chunk in shot.chunks] == [120, 120, 96]
    assert [chunk.expected_overlap_frames for chunk in shot.chunks] == [0, 48, 48]
    assert [chunk.expected_unique_frames for chunk in shot.chunks] == [120, 72, 48]
    assert sum(chunk.expected_unique_frames or 0 for chunk in shot.chunks) == 240


def test_planner_groups_action_continuations_and_restarts_on_scene_change():
    plan = build_continuity_plan(
        {
            "shots": [
                {
                    "id": "S01",
                    "duration": 5,
                    "where": "reflecting pool",
                    "who": ["paper boat"],
                    "visual": "the paper boat drifts right",
                },
                {
                    "id": "S02",
                    "duration": 5,
                    "where": "reflecting pool",
                    "who": ["paper boat"],
                    "visual": "承接上镜：纸船保持速度向右——本镜由此延续",
                },
                {
                    "id": "S03",
                    "duration": 5,
                    "where": "train platform",
                    "who": ["traveller"],
                    "visual": "a traveller enters",
                },
            ]
        },
        continuation_overlap_s=2,
    )

    first, second, third = plan.shots
    assert [shot.boundary_before for shot in plan.shots] == ["cut", "continuous", "cut"]
    assert [shot.continuity_group_id for shot in plan.shots] == ["CG001", "CG001", "CG002"]
    assert second.extends_from_shot_id == "S01"
    assert second.chunks[0].mode == "native_extend"
    assert second.chunks[0].depends_on == first.chunks[-1].chunk_id
    assert second.chunks[0].requested_frames == 168
    assert second.chunks[0].expected_overlap_frames == 48
    assert second.chunks[0].expected_unique_frames == 120
    assert third.chunks[0].mode == "fresh"


def test_planner_starts_a_fresh_group_after_three_continuous_shots():
    storyboard = {"shots": [
        {
            "id": f"S{index:02d}", "duration": 4, "where": "roof",
            "who": ["凛"], "boundary_before": "cut" if index == 1 else "continuous",
        }
        for index in range(1, 6)
    ]}

    plan = build_continuity_plan(storyboard, continuation_overlap_s=2)

    assert [shot.continuity_group_id for shot in plan.shots] == [
        "CG001", "CG001", "CG001", "CG002", "CG002",
    ]
    assert [shot.chunks[0].mode for shot in plan.shots] == [
        "fresh", "native_extend", "native_extend", "fresh", "native_extend",
    ]
    assert plan.shots[3].boundary_before == "cut"
    assert "prevent accumulated visual and narrative drift" in plan.shots[3].continuity_reason


def test_planner_never_caps_an_explicit_one_take_continuity_group():
    storyboard = {
        "continuity_mode": "one_take",
        "shots": [
            {
                "id": f"S{index:02d}",
                "duration": 15,
                "where": "rotating corridor",
                "boundary_before": "cut" if index == 1 else "continuous",
            }
            for index in range(1, 7)
        ],
    }

    plan = build_continuity_plan(storyboard)

    assert {shot.continuity_group_id for shot in plan.shots} == {"CG001"}
    assert [shot.chunks[0].mode for shot in plan.shots] == [
        "fresh",
        "native_extend",
        "native_extend",
        "native_extend",
        "native_extend",
        "native_extend",
    ]


def test_one_take_forces_every_primary_boundary_to_emit_a_bridge():
    storyboard = {
        "continuity_mode": "one_take",
        "video_provider": "seedance",
        "shots": [
            {
                "id": "S01",
                "duration": 15,
                "where": "运输船走廊",
                "who": ["Agent"],
                "micro_actions": ["Agent沿走廊前进"],
                "transition_to_next": "dissolve",
            },
            {
                "id": "S02",
                "duration": 15,
                "where": "旋转舱门",
                "who": ["Agent", "Guard"],
                "micro_actions": ["Agent与Guard失重搏斗"],
                "boundary_before": "cut",
            },
            {
                "id": "S03",
                "duration": 15,
                "where": "观察窗",
                "who": ["Agent", "Guard"],
                "micro_actions": ["Agent将Guard推向观察窗"],
                "boundary_before": "cut",
            },
        ],
    }

    plan_storyboard_beats(storyboard)

    assert [shot["boundary_before"] for shot in storyboard["shots"]] == [
        "cut", "continuous", "continuous",
    ]
    assert [shot.get("transition_to_next") for shot in storyboard["shots"][:2]] == [
        "continuous", "continuous",
    ]
    assert [
        bridge["bridge_id"] for bridge in storyboard["primary_shot_bridges"]
    ] == ["S01__S02", "S02__S03"]
    assert all(
        bridge["generation_phase"] == "post_primary_shots"
        for bridge in storyboard["primary_shot_bridges"]
    )

    continuity = build_continuity_plan(storyboard)
    assert [shot.boundary_before for shot in continuity.shots] == [
        "cut", "continuous", "continuous",
    ]
    assert [bridge.bridge_id for bridge in continuity.bridges] == [
        "S01__S02", "S02__S03",
    ]


def test_storyboard_groups_link_fresh_group_handoffs(tmp_path):
    image_dir = tmp_path / "storyboard_images"
    image_dir.mkdir()
    storyboard = {"shots": []}
    for index in range(1, 5):
        shot_id = f"S{index:02d}"
        Image.new("RGB", (64, 36), (index * 30, 0, 0)).save(image_dir / f"{shot_id}.png")
        storyboard["shots"].append({
            "id": shot_id,
            "duration": 4,
            "where": "roof",
            "who": ["凛"],
            "boundary_before": "cut" if index == 1 else "continuous",
            "start_state": f"start-{index}",
            "generation_actions": [f"action-{index}"],
            "end_state": f"end-{index}",
        })
    plan = build_continuity_plan(storyboard, continuation_overlap_s=2)

    contract = write_storyboard_groups(tmp_path, storyboard, plan)

    first, second = contract["groups"]
    assert first["next_group_id"] == "CG002"
    assert second["previous_group_id"] == "CG001"
    assert second["handoff_from_previous"] == {
        "previous_shot_id": "S03",
        "previous_end_state": "end-3",
        "entry_shot_id": "S04",
        "entry_start_state": "start-4",
        "edit": "fresh_editorial_cut",
    }


def test_storyboard_groups_persist_plot_beats_and_render_chronological_board(tmp_path):
    image_dir = tmp_path / "storyboard_images"
    image_dir.mkdir()
    Image.new("RGB", (1280, 720), "red").save(image_dir / "S01.png")
    Image.new("RGB", (1280, 720), "blue").save(image_dir / "S02.png")
    storyboard = {
        "director_storyboard": {"image": "director_storyboard.png"},
        "shots": [
        {
            "id": "S01", "duration": 4, "where": "roof", "who": ["凛"],
            "start_state": "凛静止蓄力", "generation_actions": ["凛踩水冲出"],
            "end_state": "凛冲到烬面前",
        },
        {
            "id": "S02", "duration": 4, "where": "roof", "who": ["凛", "烬"],
            "boundary_before": "continuous", "start_state": "凛冲到烬面前",
            "generation_actions": ["刀锋撞上机械臂"], "end_state": "火星炸开",
        },
        ],
    }
    plan = build_continuity_plan(storyboard, continuation_overlap_s=2)

    contract = write_storyboard_groups(tmp_path, storyboard, plan)

    group = contract["groups"][0]
    assert group["shot_ids"] == ["S01", "S02"]
    assert group["entry_shot_id"] == "S01"
    assert group["extension_shot_ids"] == ["S02"]
    assert group["beats"][1]["generation_actions"] == ["刀锋撞上机械臂"]
    assert contract["shot_to_group"] == {"S01": "CG001", "S02": "CG001"}
    assert contract["director_storyboard"] == {"image": "director_storyboard.png"}
    assert (tmp_path / group["storyboard_board"]).is_file()
    assert json.loads((tmp_path / "STORYBOARD_GROUPS.json").read_text())["version"] == 1


def test_director_storyboard_calls_image_model_with_one_overview_contract(tmp_path):
    storyboard = {
        "title": "雨夜交锋",
        "shots": [
            {
                "id": index,
                "duration": 4,
                "who": ["凛", "烬"],
                "where": "暴雨中的废弃高架与废车",
                "generation_actions": [f"凛完成动作{index}"],
                "shot_size": "wide" if index % 2 else "medium",
                "camera_movement": "steadicam",
                "boundary_before": "cut" if index in {1, 4} else "continuous",
            }
            for index in range(1, 6)
        ],
    }

    calls = []

    class FakeImageClient:
        model = "fake-seedream"

        def text_to_image(self, prompt, output_path, size, timeout):
            calls.append({"prompt": prompt, "size": size, "timeout": timeout})
            _write_grid_image(
                Path(output_path),
                size=(2560, 1440),
                vertical=[900, 1750],
                horizontal=[690],
            )
            return "https://image.invalid/director.png"

    client = FakeImageClient()
    manifest = generate_director_storyboard(
        tmp_path,
        storyboard,
        [{"name": "凛", "description": "银白长发，暗银轻甲"}],
        client=client,
    )

    assert (tmp_path / "director_storyboard.png").is_file()
    assert manifest["kind"] == "honcut.director_storyboard.v3"
    assert manifest["status"] == "done"
    assert manifest["provider"] == "seedream"
    assert manifest["model"] == "fake-seedream"
    assert (manifest["columns"], manifest["rows"]) == (3, 2)
    assert [panel["shot_id"] for panel in manifest["panels"]] == [
        "S01", "S02", "S03", "S04", "S05",
    ]
    assert [panel["group_id"] for panel in manifest["panels"]] == [
        "DG001", "DG001", "DG001", "DG002", "DG002",
    ]
    persisted = json.loads((tmp_path / "director_storyboard.json").read_text())
    assert persisted["panels"][0]["summary"] == "凛完成动作1"
    assert all(
        abs(observed - expected) <= 5
        for observed, expected in zip(
            persisted["panel_extraction"]["vertical_dividers_px"],
            [900, 1750],
            strict=True,
        )
    )
    assert abs(persisted["panel_extraction"]["horizontal_dividers_px"][0] - 690) <= 5
    assert persisted["panels"][0]["crop"] == "director_panels/S01.png"
    assert persisted["panels"][4]["grid_row"] == 1
    assert persisted["panels"][4]["grid_column"] == 1
    assert (tmp_path / "director_panels/S01.png").is_file()
    assert (tmp_path / "director_panels/S05.png").is_file()
    assert calls[0]["size"] == "2560x1440"
    assert "严格使用 3 列 × 2 行，共 5 个面板" in calls[0]["prompt"]
    assert "必须可被机器切分的固定网格合同" in calls[0]["prompt"]
    assert "16–24 像素纯白留白槽" in calls[0]["prompt"]
    assert "禁止在行间额外重复任何 Sxx 标题" in calls[0]["prompt"]
    assert "银白长发，暗银轻甲" in calls[0]["prompt"]
    assert "S01 · 4s · WIDE · 内部1格" in calls[0]["prompt"]
    assert "S02 · 4s · MEDIUM · 内部1格" in calls[0]["prompt"]
    with Image.open(tmp_path / "director_storyboard.png") as image:
        assert image.size == tuple(manifest["size_actual"])

    cached = generate_director_storyboard(
        tmp_path,
        storyboard,
        [{"name": "凛", "description": "银白长发，暗银轻甲"}],
        client=client,
    )
    assert cached["cache_hit"] is True
    assert len(calls) == 1


def test_director_storyboard_prompt_uses_five_by_three_layout_for_15_shots():
    storyboard = {
        "shots": [
            {"id": index, "duration": 4, "generation_actions": [f"动作{index}"]}
            for index in range(1, 16)
        ],
    }

    prompt, panels, layout = build_director_storyboard_prompt(storyboard)

    assert layout == (5, 3)
    assert len(panels) == 15
    assert "共 15 个面板" in prompt
    assert panels[-1]["shot_id"] == "S15"


def test_director_panel_extraction_fails_closed_without_detectable_divider(tmp_path):
    overview = tmp_path / "director_storyboard.png"
    Image.new("RGB", (1200, 600), "white").save(overview)

    with pytest.raises(RuntimeError, match="vertical divider"):
        materialize_director_panels(
            overview,
            [{"shot_id": "S01"}, {"shot_id": "S02"}],
            2,
            1,
            tmp_path,
        )


def test_director_panel_extraction_accepts_aligned_white_gutters(tmp_path):
    overview = tmp_path / "director_storyboard.png"
    pixels = np.full((600, 1200, 3), 210, dtype=np.uint8)
    pixels[:, 394:407] = 255
    pixels[:, 794:807] = 255
    pixels[294:307, :] = 255
    Image.fromarray(pixels).save(overview)

    panels, extraction = materialize_director_panels(
        overview,
        [{"shot_id": f"S{index:02d}"} for index in range(1, 7)],
        3,
        2,
        tmp_path,
    )

    assert extraction["method"] == "aligned-rule-or-gutter-v2"
    assert extraction["vertical_dividers_px"] == [400, 800]
    assert extraction["horizontal_dividers_px"] == [300]
    assert len(panels) == 6


def test_director_storyboard_retries_rejected_layout_once(tmp_path):
    storyboard = {
        "shots": [
            {"id": index, "duration": 4, "generation_actions": [f"动作{index}"]}
            for index in range(1, 5)
        ],
    }
    calls = []

    class RetryImageClient:
        model = "fake-seedream"

        def text_to_image(self, prompt, output_path, size, timeout):
            calls.append(prompt)
            if len(calls) == 1:
                Image.new("RGB", (1200, 600), "white").save(output_path)
            else:
                _write_grid_image(
                    Path(output_path),
                    size=(1200, 600),
                    vertical=[600],
                    horizontal=[300],
                )
            return f"https://image.invalid/director-{len(calls)}.png"

    manifest = generate_director_storyboard(
        tmp_path,
        storyboard,
        client=RetryImageClient(),
        size="1200x600",
    )

    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert "版式纠错重生成（第 2/2 次）" in calls[1]
    assert "必须从空白画布重新排版" in calls[1]
    assert manifest["status"] == "done"
    assert [attempt["status"] for attempt in manifest["generation_attempts"]] == [
        "rejected_layout",
        "accepted",
    ]
    rejected = tmp_path / manifest["generation_attempts"][0]["image"]
    assert rejected.is_file()
    rejected_prompt = tmp_path / manifest["generation_attempts"][0]["prompt"]
    assert rejected_prompt.is_file()
    assert (tmp_path / "director_storyboard.png").is_file()
    assert not list(tmp_path.glob(".director_storyboard_attempt_*.png"))

    cached = generate_director_storyboard(
        tmp_path,
        storyboard,
        client=RetryImageClient(),
        size="1200x600",
    )
    assert cached["cache_hit"] is True
    assert len(calls) == 2
    accepted_prompt = (tmp_path / "director_storyboard_prompt.txt").read_text()
    assert hashlib.sha256(accepted_prompt.encode()).hexdigest() == cached["prompt_sha256"]


def test_director_storyboard_stops_after_layout_retry_limit(tmp_path):
    storyboard = {
        "shots": [
            {"id": 1, "duration": 4, "generation_actions": ["建立空间"]},
            {"id": 2, "duration": 4, "generation_actions": ["角色进入"]},
        ],
    }
    calls = []

    class InvalidImageClient:
        model = "fake-seedream"

        def text_to_image(self, prompt, output_path, size, timeout):
            calls.append(prompt)
            Image.new("RGB", (1200, 600), "white").save(output_path)
            return "https://image.invalid/invalid.png"

    with pytest.raises(RuntimeError, match="vertical divider"):
        generate_director_storyboard(
            tmp_path,
            storyboard,
            client=InvalidImageClient(),
            size="1200x600",
        )

    manifest = json.loads((tmp_path / "director_storyboard.json").read_text())
    assert len(calls) == 2
    assert manifest["status"] == "error"
    assert len(manifest["generation_attempts"]) == 2
    assert all(
        attempt["status"] == "rejected_layout"
        for attempt in manifest["generation_attempts"]
    )


def test_phase1_registers_director_storyboard_in_text_storyboard(tmp_path):
    storyboard = {
        "shots": [{
            "id": "S01", "duration": 4, "who": ["凛"],
            "generation_actions": ["凛冲出"],
        }],
    }

    class FakeImageClient:
        model = "fake-seedream"

        def text_to_image(self, prompt, output_path, size, timeout):
            Image.new("RGB", (2560, 1440), "white").save(output_path)
            return "https://image.invalid/director.png"

    pipeline_core._attach_director_storyboard(
        tmp_path,
        storyboard,
        client=FakeImageClient(),
    )

    assert storyboard["director_storyboard"] == {
        "image": "director_storyboard.png",
        "manifest": "director_storyboard.json",
        "prompt": "director_storyboard_prompt.txt",
        "status": "done",
        "provider": "seedream",
        "model": "fake-seedream",
        "panel_count": 1,
        "panel_schema": "honcut.director-panels.v1",
        "panel_dir": "director_panels",
        "preliminary_groups": ["DG001"],
    }


def test_phase2_reuses_phase1_model_director_board_without_second_overview_call(
    tmp_path,
    monkeypatch,
):
    Image.new("RGB", (2560, 1440), "white").save(
        tmp_path / "director_storyboard.png"
    )
    storyboard = {
        "director_storyboard": {
            "image": "director_storyboard.png",
            "status": "done",
            "provider": "seedream",
        },
        "shots": [],
    }
    monkeypatch.setattr(
        pipeline_core,
        "run_quality_check",
        lambda *_args, **_kwargs: SimpleNamespace(passed=True, grade="A"),
    )
    monkeypatch.setattr(
        "phases.phase2.shot_storyboards.generate_shot_storyboards",
        lambda *_args, **_kwargs: {"total_boards": 0, "total_panels": 0},
    )
    monkeypatch.setattr(
        pipeline_core.time,
        "sleep",
        lambda _seconds: pytest.fail("Phase 2 must not enter overview cooldown"),
    )

    result = pipeline_core.run_phase2(storyboard, {"characters": []}, tmp_path, False)

    assert result["status"] == "done"
    assert result["provider"] == "seedream_shot_storyboards"
    assert (tmp_path / "storyboard.png").read_bytes() == (
        tmp_path / "director_storyboard.png"
    ).read_bytes()


def test_complex_shot_maps_to_three_secondary_generation_strategies(tmp_path):
    storyboard = {
        "shots": [
            {
                "id": "S01",
                "duration": 16,
                "who": ["凛", "烬"],
                "where": "暴雨高架",
                "shot_intent": "action",
                "micro_actions": ["凛踩水冲出", "凛腾空劈刀", "烬举臂格挡", "火星炸开"],
                "boundary_before": "cut",
            },
            {
                "id": "S02",
                "duration": 15,
                "who": ["凛"],
                "where": "暴雨高架",
                "what": "凛回身望向机械部队",
                "boundary_before": "continuous",
                "continuity_reason": "凛在同一高架空间承接上镜动作",
            },
        ],
    }
    plan_storyboard_beats(storyboard)
    calls = []

    class FakeImageClient:
        model = "fake-seedream"

        def text_to_image(self, prompt, output_path, size, timeout):
            calls.append(("text_to_image", prompt, None))
            Image.new("RGB", (2560, 1440), "blue").save(output_path)
            return "https://image.invalid/panel.png"

        def image_to_image(self, prompt, ref_image, output_path, size):
            calls.append(("image_to_image", prompt, ref_image))
            Image.new("RGB", (2560, 1440), "green").save(output_path)
            return "https://image.invalid/extended-panel.png"

    contract = generate_shot_storyboards(
        tmp_path,
        storyboard,
        [],
        client=FakeImageClient(),
    )

    assert [shot["storyboard_beat_count"] for shot in storyboard["shots"]] == [2, 1]
    assert [beat["generation_mode"] for beat in storyboard["shots"][0]["storyboard_beats"]] == [
        "multi_image", "tail_video_extend",
    ]
    assert storyboard["primary_shot_bridges"][0]["target_shot_id"] == "S02"
    assert contract["total_boards"] == 2
    assert contract["total_panels"] == 3
    assert contract["total_transition_panels"] == 1
    assert [call[0] for call in calls] == [
        "text_to_image", "image_to_image", "image_to_image", "image_to_image",
    ]
    assert "S01_P01（第 1/2 格）" in calls[0][1]
    assert "S01_P02（第 2/2 格）" in calls[1][1]
    assert calls[1][2].endswith("storyboard_beats/S01_P01.png")
    second_shot_record = json.loads(
        (tmp_path / "storyboard_beats/S02_P01.json").read_text(encoding="utf-8")
    )
    assert second_shot_record["reference_images"] == [
        "storyboard_beats/S01_P02.png"
    ]
    assert calls[-1][2] == [
        str(tmp_path / "storyboard_beats/S01_P02.png"),
        str(tmp_path / "storyboard_beats/S02_P01.png"),
    ]
    assert "PREVIS 手绘过渡分镜：S01__S02" in calls[-1][1]
    assert "不得让角色保持图片1原姿势" in calls[-1][1]
    transition = storyboard["primary_shot_bridges"][0]["storyboard_transition"]
    assert transition == {
        "image": "storyboard_bridges/S01__S02.png",
        "prompt": "storyboard_bridges/S01__S02_prompt.txt",
        "reference_images": [
            "storyboard_beats/S01_P02.png",
            "storyboard_beats/S02_P01.png",
        ],
        "generation_phase": "post_primary_storyboards",
        "usage": "visual_continuity_plan_not_video_endpoint",
    }
    assert (tmp_path / "storyboard_bridges/S01__S02.png").is_file()
    assert validate_shot_storyboard_artifacts(tmp_path, storyboard) == []
    assert (tmp_path / "shot_storyboards/S01.png").is_file()
    assert (tmp_path / "storyboard_beats/S01_P01.png").is_file()
    assert (tmp_path / "storyboard_beats/S01_P02.png").is_file()
    assert (tmp_path / "storyboard_images/S01.png").is_file()

    continuity = build_continuity_plan(
        storyboard,
        continuation_overlap_s=2,
    )
    first, second = continuity.shots
    assert [chunk.mode for chunk in first.chunks] == [
        "fresh", "native_extend",
    ]
    assert [chunk.execution_strategy for chunk in first.chunks] == [
        "multi_image", "tail_video_extend",
    ]
    assert [chunk.storyboard_beat_id for chunk in first.chunks] == [
        "S01_P01", "S01_P02",
    ]
    assert [chunk.storyboard_image for chunk in first.chunks] == [
        "storyboard_beats/S01_P01.png", "storyboard_beats/S01_P02.png",
    ]
    assert [chunk.requested_frames for chunk in first.chunks] == [216, 168]
    assert [chunk.expected_unique_frames for chunk in first.chunks] == [216, 168]
    assert [chunk.expected_provider_padding_frames for chunk in first.chunks] == [0, 0]
    assert continuity.bridges[0].bridge_id == "S01__S02"
    assert continuity.bridges[0].storyboard_transition_image == (
        "storyboard_bridges/S01__S02.png"
    )
    assert continuity.bridges[0].first_frame_source == (
        "source_primary_video_tail_frame"
    )
    assert continuity.bridges[0].last_frame_source == (
        "target_primary_video_first_frame"
    )
    assert second.boundary_before == "continuous"
    assert [chunk.mode for chunk in second.chunks] == ["fresh"]
    assert second.chunks[0].execution_strategy == "multi_image"
    assert second.chunks[0].depends_on is None
    groups = write_storyboard_groups(tmp_path, storyboard, continuity)
    assert groups["groups"][0]["storyboard_board"] == "shot_storyboards/S01.png"
    assert groups["groups"][0]["beats"][0]["storyboard_beats"][1][
        "storyboard_image"
    ] == "storyboard_beats/S01_P02.png"


def test_secondary_strategies_follow_content_capacity_and_boundary_semantics():
    intense_but_single_clip = {
        "video_provider": "seedance",
        "shots": [{
            "id": "S01",
            "duration": 15,
            "who": ["agent", "guard"],
            "shot_intent": "action",
            "camera_movement": "orbital",
            "micro_actions": ["双方在失重旋转中抓住扶手"],
        }],
    }
    plan_storyboard_beats(intense_but_single_clip)
    assert [
        beat["generation_mode"]
        for beat in intense_but_single_clip["shots"][0]["storyboard_beats"]
    ] == ["multi_image"]

    continuous_boundary = {
        "video_provider": "seedance",
        "shots": [
                {
                    "id": "S01",
                    "duration": 15,
                    "where": "旋转走廊",
                    "who": ["Agent"],
                    "micro_actions": ["Agent抓住门框稳住身体"],
                    "end_state": "Agent抓住门框稳定身体",
                    "transition_to_next": "cut",
                },
                {
                    "id": "S02",
                    "duration": 15,
                    "where": "旋转走廊",
                    "who": ["Agent"],
                    "micro_actions": ["Agent顺势穿过舱门"],
                    "boundary_before": "continuous",
                    "start_state": "Agent抓住门框稳定身体",
                },
        ],
    }
    plan_storyboard_beats(continuous_boundary)
    first = continuous_boundary["shots"][0]
    assert [beat["generation_mode"] for beat in first["storyboard_beats"]] == [
        "multi_image",
    ]
    assert continuous_boundary["primary_shot_bridges"][0]["bridge_id"] == "S01__S02"
    continuity = build_continuity_plan(continuous_boundary)
    assert [
        chunk.execution_strategy for chunk in continuity.shots[0].chunks
    ] == ["multi_image"]
    assert [bridge.bridge_id for bridge in continuity.bridges] == ["S01__S02"]

    transition_boundary = {
        "video_provider": "seedance",
        "shots": [
            {
                "id": "S01",
                "duration": 16,
                "micro_actions": ["Agent离开走廊"],
                "transition_to_next": "dissolve",
            },
            {
                "id": "S02",
                "duration": 15,
                "micro_actions": ["Agent出现在观察舱"],
                "boundary_before": "continuous",
            },
        ],
    }
    plan_storyboard_beats(transition_boundary)
    assert [
        beat["generation_mode"]
        for beat in transition_boundary["shots"][0]["storyboard_beats"]
    ] == ["multi_image", "tail_video_extend"]
    assert transition_boundary["shots"][0]["secondary_storyboard_planning"][
        "bridge_required"
    ] is False
    assert run_generation_capacity_checks(transition_boundary) == []
    invalid_transition = json.loads(json.dumps(transition_boundary))
    invalid_bridge = invalid_transition["shots"][0]["storyboard_beats"][1]
    invalid_bridge.update({
        "generation_mode": "first_last_frame_bridge",
        "execution_strategy": "first_last_frame_bridge",
        "bridge_target_shot_id": "S02",
        "bridge_target_beat_id": "S02_P01",
        "bridge_target_storyboard_image": "storyboard_beats/S02_P01.png",
    })
    transition_codes = {
        issue["code"] for issue in run_generation_capacity_checks(invalid_transition)
    }
    assert "secondary_storyboard_strategy_mismatch" in transition_codes
    assert "secondary_storyboard_bridge_invalid" in transition_codes


def test_storyboard_output_safety_rejection_gets_one_non_contact_retry(tmp_path):
    storyboard = {
        "shots": [
            {
                "id": "S03",
                "duration": 5,
                "who": ["agent", "guard"],
                "where": "rotating corridor",
                "storyboard_beats": [
                    {
                        "beat_id": "S03_P01",
                        "position": 1,
                        "duration_s": 5,
                        "generation_mode": "fresh",
                        "action": "双方进行膝击攻防",
                        "start_state": "双方悬浮对峙",
                        "end_state": "完成格挡",
                    }
                ],
            }
        ]
    }
    prompts = []

    class FakeImageClient:
        model = "fake-seedream"

        def text_to_image(self, prompt, output_path, size, timeout):
            prompts.append(prompt)
            if len(prompts) == 1:
                raise RuntimeError(
                    "OutputImageSensitiveContentDetected: output image may contain sensitive information"
                )
            Image.new("RGB", (1280, 720), "blue").save(output_path)
            return "https://image.invalid/safe-panel.png"

    contract = generate_shot_storyboards(
        tmp_path,
        storyboard,
        [],
        client=FakeImageClient(),
        director_storyboard_path=tmp_path / "missing.png",
    )

    assert len(prompts) == 2
    assert prompts[1].startswith("【自动安全重生成合同｜最高优先级】")
    assert "不要画拳、肘、膝或武器真正击中身体的瞬间" in prompts[1]
    panel = contract["shots"][0]["panels"][0]
    assert panel["safety_retry"]["policy"] == "synthetic_non_contact_stunt_v1"
    assert (tmp_path / panel["safety_retry"]["prompt"]).is_file()


def test_storyboard_input_safety_rejection_keeps_known_identity_references(tmp_path):
    face = tmp_path / "characters/CHAR_A/face_closeup.png"
    body = tmp_path / "characters/CHAR_A/full_body.png"
    face.parent.mkdir(parents=True)
    Image.new("RGB", (128, 128), "orange").save(face)
    Image.new("RGB", (128, 256), "navy").save(body)
    storyboard = {
        "shots": [{
            "id": "S06",
            "who": ["CHAR_A"],
            "storyboard_beats": [
                {
                    "beat_id": "S06_P01",
                    "duration_s": 5,
                    "generation_mode": "multi_image",
                    "action": "synthetic agent stabilizes in zero gravity",
                },
                {
                    "beat_id": "S06_P02",
                    "duration_s": 4,
                    "generation_mode": "tail_video_extend",
                    "action": "synthetic agent holds the final pose",
                },
            ],
        }]
    }

    class SeedCacheClient:
        model = "fake-seedream"

        def image_to_image(self, prompt, ref_image, output_path, size):
            Image.new("RGB", (1280, 720), "green").save(output_path)
            return "https://image.invalid/seed-panel.png"

        def text_to_image(self, **_kwargs):
            pytest.fail("character shots must use image-to-image")

    generate_shot_storyboards(
        tmp_path,
        storyboard,
        [{"id": "CHAR_A", "name": "Character A"}],
        client=SeedCacheClient(),
        director_storyboard_path=tmp_path / "missing.png",
    )
    (tmp_path / "storyboard_beats/S06_P02.png").unlink()
    (tmp_path / "storyboard_beats/S06_P02.json").unlink()
    image_calls = []

    class FakeImageClient:
        model = "fake-seedream"

        def image_to_image(self, prompt, ref_image, output_path, size):
            references = [ref_image] if isinstance(ref_image, str) else list(ref_image)
            image_calls.append(references)
            previous_panel = str(tmp_path / "storyboard_beats/S06_P01.png")
            if previous_panel in references:
                raise SeedreamAPIError(
                    status_code=400,
                    provider_code="InputImageSensitiveContentDetected",
                    provider_message="The input image may contain sensitive information.",
                    request_id="request-panel-sensitive",
                    response=SimpleNamespace(request=None),
                )
            Image.new("RGB", (1280, 720), "green").save(output_path)
            return "https://image.invalid/reference-panel.png"

        def text_to_image(self, **_kwargs):
            pytest.fail("known accepted identity references should recover the request")

    contract = generate_shot_storyboards(
        tmp_path,
        storyboard,
        [{"id": "CHAR_A", "name": "Character A"}],
        client=FakeImageClient(),
        director_storyboard_path=tmp_path / "missing.png",
    )

    assert len(image_calls) == 2
    assert image_calls[1] == [str(face), str(body)]
    panel = contract["shots"][0]["panels"][1]
    assert panel["mode"] == "image_to_image"
    assert panel["used_reference_images"] == [
        "characters/CHAR_A/face_closeup.png",
        "characters/CHAR_A/full_body.png",
    ]
    assert panel["dropped_reference_images"] == [
        "storyboard_beats/S06_P01.png"
    ]
    fallback = panel["input_safety_fallback"]
    assert fallback["attempts"] == 1
    assert fallback["policy"] == "role_preserving_reference_reduction_v1"
    assert [item["status"] for item in fallback["trace"]] == [
        "rejected",
        "accepted",
    ]
    assert fallback["trace"][0]["request_id"] == "request-panel-sensitive"


def test_storyboard_input_safety_rejection_has_bounded_text_fallback(tmp_path):
    face = tmp_path / "characters/CHAR_A/face_closeup.png"
    body = tmp_path / "characters/CHAR_A/full_body.png"
    face.parent.mkdir(parents=True)
    Image.new("RGB", (128, 128), "orange").save(face)
    Image.new("RGB", (128, 256), "navy").save(body)
    storyboard = {
        "shots": [{
            "id": "S01",
            "who": ["CHAR_A"],
            "storyboard_beats": [{
                "beat_id": "S01_P01",
                "duration_s": 5,
                "generation_mode": "multi_image",
                "action": "synthetic agent enters the corridor",
            }],
        }]
    }
    calls = []

    class FakeImageClient:
        model = "fake-seedream"

        def image_to_image(self, prompt, ref_image, output_path, size):
            references = [ref_image] if isinstance(ref_image, str) else list(ref_image)
            calls.append(("image_to_image", references))
            raise RuntimeError(
                "InputImageSensitiveContentDetected: input image may contain sensitive information"
            )

        def text_to_image(self, prompt, output_path, size, timeout):
            calls.append(("text_to_image", []))
            Image.new("RGB", (1280, 720), "blue").save(output_path)
            return "https://image.invalid/text-panel.png"

    contract = generate_shot_storyboards(
        tmp_path,
        storyboard,
        [{"id": "CHAR_A", "name": "Character A"}],
        client=FakeImageClient(),
        director_storyboard_path=tmp_path / "missing.png",
    )

    assert [mode for mode, _references in calls] == [
        "image_to_image",
        "image_to_image",
        "text_to_image",
    ]
    panel = contract["shots"][0]["panels"][0]
    assert panel["mode"] == "text_to_image"
    assert panel["used_reference_images"] == []
    assert panel["dropped_reference_images"] == [
        "characters/CHAR_A/face_closeup.png",
        "characters/CHAR_A/full_body.png",
    ]
    assert panel["input_safety_fallback"]["attempts"] == 2
    assert panel["input_safety_fallback"]["final_mode"] == "text_to_image"


def test_storyboard_reference_transport_failure_retries_same_request_twice(tmp_path):
    from requests.exceptions import ConnectionError as RequestsConnectionError

    storyboard = {
        "shots": [
            {
                "id": "S04",
                "duration": 5,
                "storyboard_beats": [
                    {
                        "beat_id": "S04_P01",
                        "position": 1,
                        "duration_s": 5,
                        "generation_mode": "fresh",
                        "action": "mechanical training motion",
                    }
                ],
            }
        ]
    }
    prompts = []

    class FakeImageClient:
        model = "fake-seedream"

        def text_to_image(self, prompt, output_path, size, timeout):
            prompts.append(prompt)
            if len(prompts) < 3:
                raise RequestsConnectionError(
                    "Connection aborted: write operation timed out"
                )
            Image.new("RGB", (1280, 720), "green").save(output_path)
            return "https://image.invalid/recovered-panel.png"

    contract = generate_shot_storyboards(
        tmp_path,
        storyboard,
        [],
        client=FakeImageClient(),
        director_storyboard_path=tmp_path / "missing.png",
    )

    assert len(prompts) == 3
    assert len(set(prompts)) == 1
    retry = contract["shots"][0]["panels"][0]["transport_retry"]
    assert retry == {
        "attempts": 2,
        "policy": "same_request_bounded_transport_retry_v1",
    }


def test_phase2_uses_director_board_as_visual_reference_for_every_shot(tmp_path):
    director = tmp_path / "director_storyboard.png"
    _write_grid_image(
        director,
        size=(2560, 1440),
        vertical=[1300],
        horizontal=[],
    )
    director_panels, extraction = materialize_director_panels(
        director,
        [{"position": 1, "shot_id": "S07"}, {"position": 2, "shot_id": "S08"}],
        2,
        1,
        tmp_path,
    )
    (tmp_path / "director_storyboard.json").write_text(
        json.dumps({
            "kind": "honcut.director_storyboard.v3",
            "status": "done",
            "panels": director_panels,
            "panel_extraction": extraction,
        }),
        encoding="utf-8",
    )
    storyboard = {
        "director_storyboard": {"image": director.name, "status": "done"},
        "shots": [
            {"id": "S07", "duration": 15, "micro_actions": ["打开门", "走入房间"]},
            {"id": "S08", "duration": 15, "micro_actions": ["抬头观察"]},
        ],
    }
    plan_storyboard_beats(storyboard)
    calls = []

    class FakeImageClient:
        model = "fake-seedream"

        def text_to_image(self, **kwargs):
            pytest.fail("P01 must inherit the director overview through i2i")

        def image_to_image(self, prompt, ref_image, output_path, size):
            calls.append((prompt, ref_image, size))
            Image.new("RGB", (1440, 2560), "green").save(output_path)
            return "https://image.invalid/reference-panel.png"

    contract = generate_shot_storyboards(
        tmp_path,
        storyboard,
        [],
        client=FakeImageClient(),
        size="1440x2560",
        aspect_ratio="9:16",
        director_storyboard_path=director,
    )

    assert len(calls) == 2
    assert calls[0][1] == str(tmp_path / "director_panels/S07.png")
    assert calls[1][1] == str(tmp_path / "director_panels/S08.png")
    assert all(str(director) not in str(call[1]) for call in calls)
    assert "9:16" in calls[0][0]
    assert contract["director_storyboard"] == "director_storyboard.png"
    assert contract["director_panel_schema"] == "honcut.director-panels.v1"
    with Image.open(tmp_path / "shot_storyboards/S07.png") as board:
        assert board.height > board.width / 2


def test_phase2_refuses_an_overview_without_exact_director_crops(tmp_path):
    director = tmp_path / "director_storyboard.png"
    Image.new("RGB", (1280, 720), "white").save(director)
    storyboard = {
        "shots": [{
            "id": "S01",
            "duration": 5,
            "storyboard_beats": [{
                "beat_id": "S01_P01",
                "duration_s": 5,
                "generation_mode": "fresh",
                "action": "opens the door",
            }],
        }],
    }

    with pytest.raises(RuntimeError, match="director panel lookup requires readable"):
        generate_shot_storyboards(
            tmp_path,
            storyboard,
            [],
            client=object(),
            director_storyboard_path=director,
        )


def test_missing_pxx_blocks_validation_and_resume(tmp_path):
    storyboard = {
        "shots": [{
            "id": "S21",
            "duration": 10,
            "storyboard_beats": [
                {
                    "beat_id": "S21_P01",
                    "duration_s": 5,
                    "generation_mode": "fresh",
                    "action": "进入",
                    "storyboard_image": "storyboard_beats/S21_P01.png",
                },
                {
                    "beat_id": "S21_P02",
                    "duration_s": 5,
                    "generation_mode": "extend",
                    "action": "落座",
                    "storyboard_image": "storyboard_beats/S21_P02.png",
                },
            ],
        }],
    }
    (tmp_path / "STORYBOARD.json").write_text(
        json.dumps(storyboard), encoding="utf-8"
    )
    (tmp_path / "SHOT_STORYBOARDS.json").write_text(
        json.dumps({"status": "done", "total_panels": 2}), encoding="utf-8"
    )
    beat_dir = tmp_path / "storyboard_beats"
    beat_dir.mkdir()
    Image.new("RGB", (1280, 720), "blue").save(beat_dir / "S21_P01.png")

    errors = validate_shot_storyboard_artifacts(tmp_path, storyboard)

    assert any("S21_P02" in error for error in errors)
    assert can_resume_from("phase4", tmp_path) is False


def test_provider_uses_each_chunk_storyboard_panel_and_action(monkeypatch, tmp_path):
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "完整镜头摘要", "gen_strategy": "phantom"}),
        encoding="utf-8",
    )
    observed = {}

    def fake_build(**kwargs):
        observed.update(kwargs["shot_meta"])
        return [{"type": "text", "text": kwargs["shot_meta"]["prompt"]}]

    monkeypatch.setattr("tools.asset_packager.build_content_for_shot", fake_build)
    request = ChunkExecutionRequest(
        resource_id="S01_C02",
        shot_id="S01",
        chunk=GenerationChunk(
            chunk_id="S01_C02",
            sequence=2,
            target_duration_s=7,
            mode="native_extend",
            depends_on="S01_C01",
            storyboard_beat_id="S01_P02",
            storyboard_image="storyboard_beats/S01_P02.png",
            action_prompt="烬抬起机械臂格挡",
            start_state="凛已经冲到面前",
            end_state="火星炸开",
        ),
        anchors={"scene": "roof"},
        output_path=tmp_path / "S01_C02.mp4",
        previous_output_path=tmp_path / "S01_C01.mp4",
        input_fingerprint="fingerprint",
        memory_context="",
    )

    from runtime.continuity_provider import _base_content

    content = _base_content(tmp_path, request, json.loads((shot_dir / "SHOT_META.json").read_text()))

    assert observed["_storyboard_frame_path"] == "storyboard_beats/S01_P02.png"
    assert observed["_storyboard_beat_id"] == "S01_P02"
    assert observed["gen_strategy"] == "i2v"
    assert observed["generation_actions"] == ["烬抬起机械臂格挡"]
    assert "Execute only this visible action: 烬抬起机械臂格挡" in content[0]["text"]
    assert "摄影机箭头控制机位的移动方向和轨迹" in content[0]["text"]
    assert "背景与运镜只能辅助，不能替代主体完成动作" in content[0]["text"]
    assert "最终视频的任何一帧都不得出现或残留箭头" in content[0]["text"]


def test_secondary_first_beat_uses_multi_image_generation(monkeypatch, tmp_path):
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    meta = {"prompt": "失重走廊内开始搏斗", "gen_strategy": "i2v"}
    (shot_dir / "SHOT_META.json").write_text(json.dumps(meta), encoding="utf-8")
    observed = {}

    def fake_build(**kwargs):
        observed.update(kwargs["shot_meta"])
        return [{"type": "text", "text": kwargs["shot_meta"]["prompt"]}]

    monkeypatch.setattr("tools.asset_packager.build_content_for_shot", fake_build)
    request = ChunkExecutionRequest(
        resource_id="S01_C01",
        shot_id="S01",
        chunk=GenerationChunk(
            chunk_id="S01_C01",
            sequence=1,
            target_duration_s=4,
            mode="fresh",
            execution_strategy="multi_image",
            storyboard_beat_id="S01_P01",
            storyboard_image="storyboard_beats/S01_P01.png",
            action_prompt="Agent抓住扶手接近保安",
        ),
        anchors={"scene": "rotating corridor"},
        output_path=tmp_path / "S01_C01.mp4",
        previous_output_path=None,
        input_fingerprint="fingerprint",
        memory_context="",
    )

    content = _base_content(tmp_path, request, meta)

    assert observed["gen_strategy"] == "phantom"
    assert observed["_storyboard_frame_path"] == "storyboard_beats/S01_P01.png"
    assert "ordered multi-image" in content[0]["text"]


def test_post_primary_bridge_uses_actual_source_tail_and_target_head(
    monkeypatch,
    tmp_path,
):
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "完成当前镜头并交接到下一镜"}),
        encoding="utf-8",
    )
    target = tmp_path / "shots/S02/output.mp4"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"target-video")
    previous = tmp_path / "shots/S01/chunks/S01_C02.mp4"
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b"previous-video")

    def fake_extract(_video, output):
        Image.new("RGB", (1280, 720), "black").save(output)
        return output

    monkeypatch.setattr(
        "quality.continuity_seam.extract_video_tail_frame",
        fake_extract,
    )
    monkeypatch.setattr(
        "quality.continuity_seam.extract_video_head_frame",
        fake_extract,
    )
    uploaded = []

    def fake_upload(_data, _content_type):
        uploaded.append(len(uploaded) + 1)
        return f"https://image.test/frame-{uploaded[-1]}.jpg"

    monkeypatch.setattr("clients.tos_uploader.upload_image", fake_upload)
    monkeypatch.setattr(
        "tools.asset_packager.build_content_for_shot",
        lambda **_kwargs: pytest.fail("first/last bridge must not upload reference media"),
    )
    request = ChunkExecutionRequest(
        resource_id="S01_C03",
        shot_id="S01",
        chunk=GenerationChunk(
            chunk_id="S01_C03",
            sequence=3,
            target_duration_s=4,
            requested_frames=96,
            expected_unique_frames=96,
            mode="native_extend",
            depends_on="S01_C02",
            execution_strategy="first_last_frame_bridge",
            storyboard_beat_id="S01_P03",
            storyboard_image="storyboard_beats/S01_P03.png",
            bridge_target_shot_id="S02",
            bridge_target_beat_id="S02_P01",
            bridge_target_storyboard_image="storyboard_beats/S02_P01.png",
            action_prompt="完成当前镜头动作并进入下一镜构图",
        ),
        anchors={"scene": "rotating corridor"},
        output_path=tmp_path / "S01_C03.mp4",
        previous_output_path=previous,
        target_output_path=target,
        input_fingerprint="fingerprint",
        memory_context="",
    )

    content, _meta, _seed, duration = _provider_content(tmp_path, request)

    assert duration == 4
    assert [item.get("role") for item in content] == [
        None,
        "first_frame",
        "last_frame",
    ]
    assert content[1]["image_url"]["url"].endswith("frame-1.jpg")
    assert content[2]["image_url"]["url"].endswith("frame-2.jpg")
    assert "上一一级分镜视频真实尾帧" in content[0]["text"]
    assert "下一一级分镜视频真实首帧" in content[0]["text"]
    assert "不得新增剧情" in content[0]["text"]
    assert content[0]["text"].count("[storyboard-motion-notation]") == 1


def test_seedance_duration_separates_provider_request_from_effective_story_time(
    tmp_path,
):
    tail_request = ChunkExecutionRequest(
        resource_id="S04_C02",
        shot_id="S04",
        chunk=GenerationChunk(
            chunk_id="S04_C02",
            sequence=2,
            target_duration_s=8,
            requested_frames=192,
            expected_overlap_frames=48,
            expected_unique_frames=144,
            mode="native_extend",
            depends_on="S04_C01",
            execution_strategy="tail_video_extend",
        ),
        anchors={},
        output_path=tmp_path / "S04_C02.mp4",
        previous_output_path=tmp_path / "S04_C01.mp4",
        input_fingerprint="fingerprint",
        memory_context="",
    )

    assert _chunk_duration(tail_request) == 8

    valid_bridge = ChunkExecutionRequest(
        resource_id="S03_C03",
        shot_id="S03",
        chunk=GenerationChunk(
            chunk_id="S03_C03",
            sequence=3,
            target_duration_s=4,
            mode="native_extend",
            depends_on="S03_C02",
            execution_strategy="first_last_frame_bridge",
            bridge_target_shot_id="S04",
            bridge_target_beat_id="S04_P01",
            bridge_target_storyboard_image="storyboard_beats/S04_P01.png",
        ),
        anchors={},
        output_path=tmp_path / "S03_C03.mp4",
        previous_output_path=tmp_path / "S03_C02.mp4",
        input_fingerprint="fingerprint",
        memory_context="",
    )

    assert _chunk_duration(valid_bridge) == 4


def test_provider_prepends_no_real_person_visual_contract(monkeypatch, tmp_path):
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "写实男性特工近身搏斗", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HONCUT_NO_REAL_PERSON", "1")
    monkeypatch.setattr(
        "tools.asset_packager.build_content_for_shot",
        lambda **kwargs: [{"type": "text", "text": kwargs["shot_meta"]["prompt"]}],
    )
    request = _fresh_chunk_request(tmp_path)

    from runtime.continuity_provider import _base_content

    content = _base_content(
        tmp_path,
        request,
        json.loads((shot_dir / "SHOT_META.json").read_text()),
    )

    assert content[0]["text"].startswith("【非真人视觉硬约束】")
    assert "全封闭机械头盔" in content[0]["text"]
    assert "旧描述，一律忽略" in content[0]["text"]


def test_storyboard_beat_planner_discards_quote_only_fragments():
    storyboard = {"shots": [{
        "id": "S01",
        "duration": 16,
        "action_description": (
            "暴雨砸在高架上。凛与烬持刀对峙。"
            "“为什么骗我？”“我只是不想你死。”"
        ),
    }]}

    plan_storyboard_beats(storyboard)

    actions = [beat["action"] for beat in storyboard["shots"][0]["storyboard_beats"]]
    assert actions == ["暴雨砸在高架上", "凛与烬持刀对峙"]
    assert all(action not in {"“", "”", "\""} for action in actions)


def test_storyboard_beat_planner_is_semantic_and_rejects_impossible_density():
    storyboard = {"shots": [
        {
            "id": "S07",
            "duration": 16,
            "what": "读者查阅旧书",
            "micro_actions": ["读者翻开旧书"],
        },
        {
            "id": "S08",
            "duration": 15,
            "micro_actions": ["抬头", "发现批注", "触摸纸页", "迟疑"],
        },
    ]}

    plan_storyboard_beats(storyboard)

    long_quiet, dense_short = storyboard["shots"]
    assert long_quiet["storyboard_beat_count"] == 2
    assert "延续前格状态并推进至本镜结局：延续" not in " ".join(
        beat["action"] for beat in long_quiet["storyboard_beats"]
    )
    assert dense_short["storyboard_beat_count"] == 2
    assert [
        action
        for beat in dense_short["storyboard_beats"]
        for action in beat["micro_actions"]
    ] == ["抬头", "发现批注", "触摸纸页", "迟疑"]

    impossible = {
        "video_provider": "seedance",
        "shots": [{"id": "S09", "duration": 20, "micro_actions": list("1234567")}],
    }
    with pytest.raises(ValueError, match="cannot fit 7 micro-actions"):
        plan_storyboard_beats(impossible)


def test_phase5_blocks_secondary_plot_reordering_and_wrong_bridge_target():
    storyboard = {
        "video_provider": "seedance",
        "shots": [
            {
                "id": "S01",
                "duration": 15,
                "who": ["agent", "guard"],
                "where": "旋转走廊",
                "shot_intent": "action",
                "camera_movement": "dolly_in",
                "micro_actions": ["抓住扶手", "解除武器", "穿过舱门"],
                "start_state": "二人在旋转走廊内相持",
                "end_state": "二人穿过舱门",
            },
            {
                "id": "S02",
                "duration": 15,
                "who": ["agent", "guard"],
                "where": "旋转走廊",
                "micro_actions": ["进入下一舱段"],
                "boundary_before": "continuous",
                "start_state": "二人刚穿过舱门",
                "end_state": "进入下一舱段",
            },
        ],
    }
    plan_storyboard_beats(storyboard)

    assert run_generation_capacity_checks(storyboard) == []

    beats = storyboard["shots"][0]["storyboard_beats"]
    beats[0]["micro_actions"], beats[1]["micro_actions"] = (
        beats[1]["micro_actions"],
        beats[0]["micro_actions"],
    )
    storyboard["primary_shot_bridges"][0]["target_shot_id"] = "S99"
    codes = {
        issue["code"] for issue in run_generation_capacity_checks(storyboard)
    }

    assert "secondary_storyboard_action_order_mismatch" in codes
    assert "secondary_storyboard_bridge_invalid" in codes


def test_planner_allocates_fractional_shots_from_cumulative_frame_endpoints():
    plan = build_continuity_plan(
        {"shots": [{"id": index, "duration": 0.06} for index in range(1, 4)]},
        timeline_fps=24,
    )

    assert [shot.target_frames for shot in plan.shots] == [1, 2, 1]
    assert sum(shot.target_frames or 0 for shot in plan.shots) == round(0.18 * 24)


def test_chunk_contract_rejects_extension_without_a_dependency():
    with pytest.raises(ValidationError, match="depends_on"):
        GenerationChunk(
            chunk_id="S01_C02",
            sequence=2,
            target_duration_s=5,
            mode="native_extend",
        )


def test_plan_contract_rejects_a_chunk_above_the_provider_limit():
    with pytest.raises(ValidationError, match="provider chunk duration limit"):
        ContinuityPlan.model_validate(
            {
                "provider_chunk_limit_s": 15,
                "shots": [
                    {
                        "shot_id": "S01",
                        "target_duration_s": 16,
                        "chunks": [
                            {
                                "chunk_id": "S01_C01",
                                "sequence": 1,
                                "target_duration_s": 16,
                                "mode": "fresh",
                            }
                        ],
                    }
                ],
            }
        )


def test_upload_media_file_preserves_video_bytes_and_mime(monkeypatch, tmp_path):
    video = tmp_path / "tail.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42-video-bytes")
    observed = {}

    def fake_upload(data, object_key, content_type):
        observed.update(data=data, object_key=object_key, content_type=content_type)
        return "https://tos.test/reference.mp4"

    monkeypatch.setattr(tos_uploader, "upload_file", fake_upload)

    assert tos_uploader.upload_media_file(video) == "https://tos.test/reference.mp4"
    assert observed["data"] == video.read_bytes()
    assert observed["object_key"].endswith(".mp4")
    assert observed["content_type"] == "video/mp4"


def test_base64_video_upload_bypasses_image_compression(monkeypatch):
    video_data = b"\x00\x00\x00\x18ftypmp42-video-bytes"
    observed = {}

    def fake_upload(data, object_key, content_type):
        observed.update(data=data, object_key=object_key, content_type=content_type)
        return "https://tos.test/reference.mp4"

    monkeypatch.setattr(tos_uploader, "upload_file", fake_upload)
    payload = "data:video/mp4;base64," + base64.b64encode(video_data).decode()

    assert tos_uploader.base64_video_to_signed_url(payload) == "https://tos.test/reference.mp4"
    assert observed["data"] == video_data
    assert observed["object_key"].endswith(".mp4")
    assert observed["content_type"] == "video/mp4"


def test_video_extension_uses_reference_video_with_persistent_image_anchors(monkeypatch, tmp_path):
    video = tmp_path / "S03_C01.mp4"
    video.write_bytes(b"video")
    observed = {}

    monkeypatch.setattr(
        tos_uploader,
        "upload_media_file",
        lambda path, prefix: "https://tos.test/chunk.mp4",
    )

    def fake_submit(content, **kwargs):
        observed["content"] = content
        observed["kwargs"] = kwargs
        return "task-extension-1"

    monkeypatch.setattr(seedance_client, "submit_content", fake_submit)

    task_id = seedance_client.submit_video_extension(
        "continue walking right",
        str(video),
        api_key="test-key",
        model="doubao-seedance-2.0-mini",
        duration=8,
        reference_image_urls=["https://tos.test/character.png"],
        seed=7,
    )

    assert task_id == "task-extension-1"
    assert [item.get("role", item["type"]) for item in observed["content"]] == [
        "text",
        "reference_image",
        "reference_video",
    ]
    assert observed["content"][-1]["video_url"]["url"].endswith("chunk.mp4")
    assert observed["kwargs"]["duration"] == 8
    assert observed["kwargs"]["seed"] == 7


def test_ark_agent_plan_generation_endpoints_share_the_canonical_base_url():
    expected_base = "https://ark.cn-beijing.volces.com/api/plan/v3"

    assert seedance_client.BASE_URL == expected_base
    assert seedance_client.TASKS_ENDPOINT == (f"{expected_base}/contents/generations/tasks")
    assert seedance_client.TASK_ENDPOINT == (
        f"{expected_base}/contents/generations/tasks/{{task_id}}"
    )
    assert IMAGE_ENDPOINT == f"{expected_base}/images/generations"


def test_seedance_task_management_uses_documented_urls_and_filters(monkeypatch):
    observed = []

    class Response:
        status_code = 200
        text = "ok"

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def fake_get(url, **kwargs):
        observed.append(("GET", url, kwargs))
        return Response({"id": "task-1", "status": "running"})

    def fake_delete(url, **kwargs):
        observed.append(("DELETE", url, kwargs))
        return Response({"id": "task-1", "status": "cancelled"})

    monkeypatch.setattr(seedance_client.requests, "get", fake_get)
    monkeypatch.setattr(seedance_client.requests, "delete", fake_delete)

    task = seedance_client.get_task("task-1", api_key="test-key")
    listing = seedance_client.list_tasks(
        api_key="test-key",
        page_num=2,
        page_size=7,
        filter_status="succeeded",
        task_ids=["task-1", "task-2"],
        model="doubao-seedance-2.0-mini",
    )
    deleted = seedance_client.cancel_or_delete_task("task-1", api_key="test-key")

    assert task["status"] == "running"
    assert listing["id"] == "task-1"
    assert deleted["status"] == "cancelled"
    assert observed[0][1].endswith("/contents/generations/tasks/task-1")
    assert observed[1][1] == seedance_client.TASKS_ENDPOINT
    assert observed[1][2]["params"] == {
        "page_num": 2,
        "page_size": 7,
        "filter.status": "succeeded",
        "filter.task_ids": "task-1,task-2",
        "filter.model": "doubao-seedance-2.0-mini",
    }
    assert observed[2][0] == "DELETE"
    assert observed[2][1].endswith("/contents/generations/tasks/task-1")
    assert all(call[2]["headers"]["Authorization"] == "Bearer test-key" for call in observed)


def _materialize_test_shot(chunk_paths, output_path):
    output_path.write_bytes(b"|".join(path.read_bytes() for path in chunk_paths))


def _inspect_test_seam(previous_path, following_path, boundary_id):
    return {
        "boundary_id": boundary_id,
        "previous_size": previous_path.stat().st_size,
        "following_size": following_path.stat().st_size,
        "policy": "observe_only",
    }


def test_chunk_runtime_serializes_dependencies_but_parallelizes_shots(tmp_path):
    plan = build_continuity_plan(
        {
            "shots": [
                {"id": "S01", "duration": 16},
                {"id": "S02", "duration": 16},
            ]
        }
    )
    first_chunk_barrier = threading.Barrier(2)
    sequences = defaultdict(list)

    def execute(request):
        sequences[request.shot_id].append(request.chunk.sequence)
        if request.chunk.sequence == 1:
            assert request.previous_output_path is None
            first_chunk_barrier.wait(timeout=2)
        else:
            assert request.previous_output_path is not None
            assert request.previous_output_path.is_file()
        request.output_path.write_bytes(request.resource_id.encode())
        policy_repairs = (
            ({"attempt": 1, "policy": "original_ambient_no_music_v1"},)
            if request.resource_id == "S01_C01"
            else ()
        )
        return ChunkExecutionResult(
            request.output_path,
            f"task-{request.resource_id}",
            policy_repairs,
        )

    report = execute_continuity_plan(
        plan,
        tmp_path,
        execute_chunk=execute,
        inspect_seam=_inspect_test_seam,
        materialize_shot=_materialize_test_shot,
        max_workers=2,
    )

    assert report["status"] == "done"
    assert report["executed_chunks"] == 4
    assert report["repair_attempts"] == 1
    assert report["seam_repair_attempts"] == 0
    assert report["copyright_policy_repair_attempts"] == 1
    assert dict(sequences) == {"S01": [1, 2], "S02": [1, 2]}
    assert (tmp_path / "shots/S01/output.mp4").read_bytes() == b"S01_C01|S01_C02"
    lineage = json.loads((tmp_path / "CONTINUITY_LINEAGE.json").read_text())
    assert lineage["chunks"]["S01_C01"]["copyright_policy_repair_attempts"] == 1


def test_chunk_runtime_archives_primary_shot_bridge_videos(tmp_path):
    storyboard = {
        "continuity_mode": "one_take",
        "video_provider": "seedance",
        "shots": [
            {
                "id": "S01",
                "duration": 15,
                "where": "走廊",
                "who": ["Agent"],
                "micro_actions": ["Agent穿过旋转走廊"],
            },
            {
                "id": "S02",
                "duration": 15,
                "where": "观察窗",
                "who": ["Agent"],
                "micro_actions": ["Agent在观察窗前稳定身体"],
            },
        ],
    }
    plan_storyboard_beats(storyboard)
    plan = build_continuity_plan(storyboard)
    requests = []

    def execute(request):
        requests.append(request)
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_bytes(request.resource_id.encode())
        return ChunkExecutionResult(request.output_path)

    report = execute_continuity_plan(
        plan,
        tmp_path,
        execute_chunk=execute,
        inspect_seam=_inspect_test_seam,
        materialize_shot=_materialize_test_shot,
        normalize_chunk=lambda path, _chunk, _fps: {"output_path": str(path)},
    )

    manifest = json.loads((tmp_path / "PRIMARY_SHOT_BRIDGES.json").read_text())
    bridge_path = tmp_path / "shot_bridges/S01__S02.mp4"
    assert report["status"] == "done"
    assert report["bridge_outputs"] == ["shot_bridges/S01__S02.mp4"]
    assert bridge_path.is_file()
    assert manifest["kind"] == "honcut.primary_shot_bridges.v2"
    assert manifest["count"] == 1
    assert manifest["bridges"][0]["embedded_in_preceding_shot_output"] is False
    assert manifest["bridges"][0]["generated_after_primary_shots"] is True
    assert manifest["bridges"][0]["last_frame_source"] == (
        "target_primary_video_first_frame"
    )
    assert manifest["bridges"][0]["video_endpoint_policy"] == (
        "actual_completed_primary_frames_not_storyboard_transition"
    )
    assert [request.resource_id for request in requests] == [
        "S01_C01", "S02_C01", "S01__S02_B01",
    ]
    bridge_request = requests[-1]
    assert bridge_request.previous_output_path == tmp_path / "shots/S01/output.mp4"
    assert bridge_request.target_output_path == tmp_path / "shots/S02/output.mp4"


def test_chunk_runtime_labels_local_privacy_bridge_fallback(tmp_path):
    storyboard = {
        "continuity_mode": "one_take",
        "video_provider": "seedance",
        "shots": [
            {"id": "S01", "duration": 15, "where": "A", "micro_actions": ["move"]},
            {"id": "S02", "duration": 15, "where": "B", "micro_actions": ["move"]},
        ],
    }
    plan_storyboard_beats(storyboard)
    plan = build_continuity_plan(storyboard)

    def execute(request):
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_bytes(request.resource_id.encode())
        if request.chunk.execution_strategy == "first_last_frame_bridge":
            return ChunkExecutionResult(
                request.output_path,
                privacy_policy_repairs=({
                    "attempt": 1,
                    "policy": "required_endpoints_inseparable_local_handle_fallback_v1",
                },),
                provider_fallback={
                    "policy": "local_boundary_handle_passthrough_v1",
                    "provider_generation": False,
                },
            )
        return ChunkExecutionResult(request.output_path)

    report = execute_continuity_plan(
        plan,
        tmp_path,
        execute_chunk=execute,
        inspect_seam=_inspect_test_seam,
        materialize_shot=_materialize_test_shot,
        normalize_chunk=lambda path, _chunk, _fps: {"output_path": str(path)},
    )

    manifest = json.loads((tmp_path / "PRIMARY_SHOT_BRIDGES.json").read_text())
    bridge = manifest["bridges"][0]
    assert report["privacy_policy_repair_attempts"] == 1
    assert report["repair_attempts"] == 1
    assert bridge["provider_fallback"]["policy"] == (
        "local_boundary_handle_passthrough_v1"
    )
    assert bridge["video_endpoint_policy"] == (
        "local_actual_boundary_handle_passthrough"
    )
    assert bridge["phase8_transition_policy"] == (
        "replace_boundary_handles_with_local_passthrough"
    )


def test_chunk_runtime_returns_a_top_level_failure_summary(tmp_path):
    plan = build_continuity_plan({"shots": [{"id": "S01", "duration": 15}]})

    def fail(_request):
        raise RuntimeError("provider rejected the request")

    report = execute_continuity_plan(plan, tmp_path, execute_chunk=fail)

    assert report["status"] == "error"
    assert report["error"] == (
        "Phase 6 continuity generation failed: "
        "S01: provider rejected the request"
    )


def test_chunk_runtime_relays_previous_shot_video_inside_a_continuity_group(tmp_path):
    plan = build_continuity_plan(
        {
            "shots": [
                {"id": "S01", "duration": 5},
                {
                    "id": "S02",
                    "duration": 5,
                    "who": ["agent", "guard"],
                    "where": "旋转走廊",
                    "boundary_before": "continuous",
                    "continuity_subject": "paper boat",
                },
                {"id": "S03", "duration": 5},
            ]
        }
    )
    observed = []

    def execute(request):
        observed.append(
            (
                request.shot_id,
                request.chunk.mode,
                request.previous_output_path,
            )
        )
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_bytes(request.resource_id.encode())
        return ChunkExecutionResult(request.output_path)

    report = execute_continuity_plan(
        plan,
        tmp_path,
        execute_chunk=execute,
        materialize_shot=_materialize_test_shot,
        max_workers=2,
    )

    by_shot = {shot_id: (mode, predecessor) for shot_id, mode, predecessor in observed}
    assert report["status"] == "done"
    assert by_shot["S01"] == ("fresh", None)
    assert by_shot["S02"][0] == "native_extend"
    assert by_shot["S02"][1] == tmp_path / "shots/S01/output.mp4"
    assert by_shot["S03"] == ("fresh", None)


def test_chunk_runtime_generates_a_bounded_continuation_topup_for_large_frame_deficit(
    tmp_path,
):
    plan = build_continuity_plan(
        {"shots": [{"id": "S01", "duration": 8}]},
        provider_chunk_limit_s=5,
        continuation_overlap_s=2,
    )
    calls = []
    decoded_frames = {
        "S01_C01": 100,
        "S01_C02": 70,
        "S01_T01": 24,
        "output": 192,
    }

    def execute(request):
        calls.append((request.resource_id, request.chunk.target_duration_s))
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_bytes(request.resource_id.encode())
        return ChunkExecutionResult(request.output_path)

    def probe(path, _fps):
        return {"frames": decoded_frames[path.stem]}

    def finalize(path, target_frames, fps):
        assert target_frames == 192
        assert fps == 24
        return {"method": "exact_tail_trim", "after_frames": 192}

    report = execute_continuity_plan(
        plan,
        tmp_path,
        execute_chunk=execute,
        inspect_seam=_inspect_test_seam,
        materialize_shot=_materialize_test_shot,
        probe_frames=probe,
        finalize_shot=finalize,
    )

    timing = json.loads((tmp_path / "shots/S01/CONTINUITY_TIMING.json").read_text())
    assert report["status"] == "done"
    assert report["duration_topups"] == 1
    assert calls == [("S01_C01", 5.0), ("S01_C02", 5.0), ("S01_T01", 3.0)]
    assert [row["remaining_target_frames"] for row in timing["chunks"]] == [92, 22, -2]


def test_chunk_runtime_resumes_and_invalidates_descendants(tmp_path):
    plan = build_continuity_plan({"shots": [{"id": "S01", "duration": 16}]})
    calls = []
    variant = {"suffix": b""}

    def execute(request):
        calls.append(request.resource_id)
        request.output_path.write_bytes(request.resource_id.encode() + variant["suffix"])
        return ChunkExecutionResult(request.output_path)

    first = execute_continuity_plan(
        plan,
        tmp_path,
        execute_chunk=execute,
        inspect_seam=_inspect_test_seam,
        materialize_shot=_materialize_test_shot,
    )
    resumed = execute_continuity_plan(
        plan,
        tmp_path,
        execute_chunk=execute,
        inspect_seam=_inspect_test_seam,
        materialize_shot=_materialize_test_shot,
    )
    (tmp_path / "shots/S01/chunks/S01_C01.mp4").write_bytes(b"tampered")
    variant["suffix"] = b"-v2"
    repaired = execute_continuity_plan(
        plan,
        tmp_path,
        execute_chunk=execute,
        inspect_seam=_inspect_test_seam,
        materialize_shot=_materialize_test_shot,
    )
    changed = plan.model_copy(deep=True)
    changed.shots[0].chunks[0].target_duration_s = 7
    invalidated = execute_continuity_plan(
        changed,
        tmp_path,
        execute_chunk=execute,
        inspect_seam=_inspect_test_seam,
        materialize_shot=_materialize_test_shot,
    )

    assert first["executed_chunks"] == 2
    assert resumed["executed_chunks"] == 0
    assert resumed["skipped_chunks"] == 2
    assert repaired["executed_chunks"] == 2
    assert invalidated["executed_chunks"] == 2
    assert calls == [
        "S01_C01",
        "S01_C02",
        "S01_C01",
        "S01_C02",
        "S01_C01",
        "S01_C02",
    ]


def test_seam_metrics_keep_raw_evidence_in_observe_only_mode():
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)

    stable = compare_frame_sequences([black, black], [black, black])
    hard_cut = compare_frame_sequences([black, black], [white, white])

    assert stable["policy"] == "observe_only"
    assert stable["provisional_risk_score"] == 0
    assert hard_cut["pixel_mae"] == 1
    assert hard_cut["brightness_delta"] == 1
    assert hard_cut["provisional_risk_score"] > stable["provisional_risk_score"]


def test_seam_metrics_observe_motion_direction_reversal():
    def frame(x):
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[12:18, x : x + 6] = 255
        return pixels

    consistent = compare_frame_sequences(
        [frame(5), frame(8)],
        [frame(11), frame(14)],
    )
    reversed_motion = compare_frame_sequences(
        [frame(5), frame(8)],
        [frame(5), frame(2)],
    )

    assert consistent["motion_direction_change"] == 0
    assert reversed_motion["motion_direction_change"] == 1


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_real_video_seam_measurement_separates_stable_and_discontinuous_boundaries(tmp_path):
    def make_clip(name, color):
        path = tmp_path / name
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=96x54:d=0.5",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return path

    black_a = make_clip("black_a.mp4", "black")
    black_b = make_clip("black_b.mp4", "black")
    white = make_clip("white.mp4", "white")
    stable = measure_video_seam(
        black_a,
        black_b,
        "stable",
        evidence_dir=tmp_path / "stable_evidence",
    )
    discontinuous = measure_video_seam(
        black_a,
        white,
        "discontinuous",
        evidence_dir=tmp_path / "discontinuous_evidence",
    )

    assert stable["metrics"]["provisional_risk_score"] < 0.01
    assert discontinuous["metrics"]["provisional_risk_score"] > 0.7
    assert all(
        (tmp_path / "stable_evidence" / path.split("/")[-1]).is_file()
        for path in stable["tail_frames"]
    )


def _labelled_seam(observation_id, label, risk):
    return {
        "observation_id": observation_id,
        "label": label,
        "source": "synthetic-fixture",
        "metrics": {"provisional_risk_score": risk},
    }


def test_seam_calibration_requires_balanced_labelled_samples():
    calibration = calibrate_seam_policy(
        [
            _labelled_seam("good-1", "acceptable", 0.02),
            _labelled_seam("bad-1", "defective", 0.8),
        ]
    )

    assert calibration.status == "insufficient"
    assert calibration.accept_threshold is None
    assert decide_seam({"provisional_risk_score": 0.9}, calibration)["action"] == "observe_only"


def test_seam_calibration_creates_a_review_band_and_bounded_repair():
    calibration = calibrate_seam_policy(
        [
            _labelled_seam("good-1", "acceptable", 0.01),
            _labelled_seam("good-2", "acceptable", 0.03),
            _labelled_seam("good-3", "acceptable", 0.05),
            _labelled_seam("bad-1", "defective", 0.55),
            _labelled_seam("bad-2", "defective", 0.72),
            _labelled_seam("bad-3", "defective", 0.9),
        ]
    )

    assert calibration.status == "certified"
    assert calibration.accept_threshold == 0.05
    assert calibration.regenerate_threshold == 0.55
    assert decide_seam({"provisional_risk_score": 0.03}, calibration)["action"] == "accept"
    assert decide_seam({"provisional_risk_score": 0.3}, calibration)["action"] == "human_review"
    assert decide_seam({"provisional_risk_score": 0.8}, calibration)["action"] == "regenerate"
    exhausted = decide_seam(
        {"provisional_risk_score": 0.8},
        calibration,
        repair_attempts=1,
        max_repairs=1,
    )
    assert exhausted["action"] == "human_review"
    assert "exhausted" in exhausted["reason"]


def test_seam_calibration_refuses_overlapping_labels():
    calibration = calibrate_seam_policy(
        [
            _labelled_seam("good-1", "acceptable", 0.1),
            _labelled_seam("good-2", "acceptable", 0.2),
            _labelled_seam("good-3", "acceptable", 0.4),
            _labelled_seam("bad-1", "defective", 0.35),
            _labelled_seam("bad-2", "defective", 0.6),
            _labelled_seam("bad-3", "defective", 0.8),
        ]
    )

    assert calibration.status == "overlap"
    assert calibration.accept_threshold is None


def _certified_seam_calibration():
    return calibrate_seam_policy(
        [
            _labelled_seam("good-1", "acceptable", 0.01),
            _labelled_seam("good-2", "acceptable", 0.02),
            _labelled_seam("good-3", "acceptable", 0.03),
            _labelled_seam("bad-1", "defective", 0.7),
            _labelled_seam("bad-2", "defective", 0.8),
            _labelled_seam("bad-3", "defective", 0.9),
        ]
    )


def test_chunk_runtime_repairs_a_bad_seam_before_generating_descendants(tmp_path):
    plan = build_continuity_plan({"shots": [{"id": "S01", "duration": 32}]})
    calls = []

    def execute(request):
        calls.append(request.resource_id)
        if request.chunk.chunk_id == "S01_C03":
            assert request.previous_output_path.read_bytes() == b"good-following"
        if request.resource_id == "S01_C02":
            payload = b"bad-following"
        elif request.resource_id == "S01_C02_R01":
            payload = b"good-following"
        else:
            payload = request.resource_id.encode()
        request.output_path.write_bytes(payload)
        return ChunkExecutionResult(request.output_path, f"task-{request.resource_id}")

    def inspect(previous_path, following_path, boundary_id):
        risk = 0.8 if following_path.read_bytes() == b"bad-following" else 0.02
        return {
            "boundary_id": boundary_id,
            "metrics": {"provisional_risk_score": risk},
        }

    report = execute_continuity_plan(
        plan,
        tmp_path,
        execute_chunk=execute,
        inspect_seam=inspect,
        materialize_shot=_materialize_test_shot,
        seam_calibration=_certified_seam_calibration().model_dump(mode="json"),
        max_seam_repairs=1,
    )

    lineage = json.loads((tmp_path / "CONTINUITY_LINEAGE.json").read_text())
    repaired_seam = lineage["seams"]["S01_C01__S01_C02"]
    assert report["status"] == "done"
    assert report["repair_attempts"] == 1
    assert calls == ["S01_C01", "S01_C02", "S01_C02_R01", "S01_C03"]
    assert len(repaired_seam["attempt_history"]) == 2
    assert repaired_seam["decision"]["action"] == "accept"
    assert lineage["chunks"]["S01_C02"]["resource_id"] == "S01_C02_R01"


def test_chunk_runtime_rejects_uncertified_calibration_before_provider_execution(tmp_path):
    plan = build_continuity_plan({"shots": [{"id": "S01", "duration": 16}]})
    calibration = calibrate_seam_policy(
        [
            _labelled_seam("good-1", "acceptable", 0.02),
            _labelled_seam("bad-1", "defective", 0.8),
        ]
    )

    def unexpected_execute(request):
        raise AssertionError(f"provider must not receive {request.resource_id}")

    with pytest.raises(RuntimeError, match="requires certified"):
        execute_continuity_plan(
            plan,
            tmp_path,
            execute_chunk=unexpected_execute,
            seam_calibration=calibration.model_dump(mode="json"),
        )


def test_chunk_runtime_stops_after_repair_budget_is_exhausted(tmp_path):
    plan = build_continuity_plan({"shots": [{"id": "S01", "duration": 32}]})
    calls = []

    def execute(request):
        calls.append(request.resource_id)
        request.output_path.write_bytes(request.resource_id.encode())
        return ChunkExecutionResult(request.output_path)

    def inspect(previous_path, following_path, boundary_id):
        return {
            "boundary_id": boundary_id,
            "metrics": {"provisional_risk_score": 0.8},
        }

    report = execute_continuity_plan(
        plan,
        tmp_path,
        execute_chunk=execute,
        inspect_seam=inspect,
        materialize_shot=_materialize_test_shot,
        seam_calibration=_certified_seam_calibration().model_dump(mode="json"),
        max_seam_repairs=1,
    )

    assert report["status"] == "error"
    assert report["repair_attempts"] == 1
    assert calls == ["S01_C01", "S01_C02", "S01_C02_R01"]
    assert "human review" in report["errors"][0]["error"]


def test_replay_signal_blocks_an_otherwise_acceptable_seam(tmp_path):
    plan = build_continuity_plan({"shots": [{"id": "S01", "duration": 16}]})

    def execute(request):
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_bytes(request.resource_id.encode())
        return ChunkExecutionResult(request.output_path, f"task-{request.resource_id}")

    def inspect(_previous_path, _following_path, boundary_id):
        return {
            "boundary_id": boundary_id,
            "metrics": {"provisional_risk_score": 0.02},
            "replay": {
                "likely_replay": True,
                "policy": "human_review_only",
            },
        }

    report = execute_continuity_plan(
        plan,
        tmp_path,
        execute_chunk=execute,
        inspect_seam=inspect,
        materialize_shot=_materialize_test_shot,
        seam_calibration=_certified_seam_calibration().model_dump(mode="json"),
    )

    lineage = json.loads((tmp_path / "CONTINUITY_LINEAGE.json").read_text())
    decision = lineage["seams"]["S01_C01__S01_C02"]["decision"]
    assert report["status"] == "error"
    assert decision["action"] == "human_review"
    assert decision["replay_policy"] == "human_review_only"
    assert "replay" in decision["reason"]


def test_exact_human_review_approval_regenerates_without_repeating_original_chunks(tmp_path):
    plan = build_continuity_plan({"shots": [{"id": "S01", "duration": 16}]})
    calls = []

    def execute(request):
        calls.append(request.resource_id)
        payload = b"repaired" if request.repair_attempt else request.resource_id.encode()
        request.output_path.write_bytes(payload)
        return ChunkExecutionResult(request.output_path, f"task-{request.resource_id}")

    def inspect(previous_path, following_path, boundary_id):
        risk = 0.02 if following_path.read_bytes() == b"repaired" else 0.3
        return {
            "boundary_id": boundary_id,
            "metrics": {"provisional_risk_score": risk},
        }

    kwargs = {
        "execute_chunk": execute,
        "inspect_seam": inspect,
        "materialize_shot": _materialize_test_shot,
        "seam_calibration": _certified_seam_calibration().model_dump(mode="json"),
        "max_seam_repairs": 1,
    }
    first = execute_continuity_plan(plan, tmp_path, **kwargs)
    lineage = json.loads((tmp_path / "CONTINUITY_LINEAGE.json").read_text())
    seam = lineage["seams"]["S01_C01__S01_C02"]
    (tmp_path / "CONTINUITY_REVIEW_DECISIONS.json").write_text(
        json.dumps(
            {
                "kind": "honcut.continuity_review_decisions.v1",
                "decisions": {
                    "S01_C01__S01_C02": {
                        "action": "regenerate",
                        "approved_input_fingerprint": seam["input_fingerprint"],
                        "approved_by": "human",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    repaired = execute_continuity_plan(plan, tmp_path, **kwargs)

    lineage = json.loads((tmp_path / "CONTINUITY_LINEAGE.json").read_text())
    assert first["status"] == "error"
    assert repaired["status"] == "done"
    assert repaired["skipped_chunks"] == 2
    assert repaired["repair_attempts"] == 1
    assert calls == ["S01_C01", "S01_C02", "S01_C02_R01"]
    assert lineage["chunks"]["S01_C02"]["resource_id"] == "S01_C02_R01"
    assert lineage["seams"]["S01_C01__S01_C02"]["decision"]["action"] == "accept"


def test_chunk_runtime_resumes_the_same_repair_resource_after_provider_failure(tmp_path):
    plan = build_continuity_plan({"shots": [{"id": "S01", "duration": 32}]})
    calls = []
    fail_once = {"enabled": True}

    def execute(request):
        calls.append(request.resource_id)
        if request.resource_id == "S01_C02_R01" and fail_once["enabled"]:
            fail_once["enabled"] = False
            raise RuntimeError("temporary provider polling failure")
        payload = b"bad" if request.resource_id == "S01_C02" else b"good"
        request.output_path.write_bytes(payload)
        return ChunkExecutionResult(request.output_path)

    def inspect(previous_path, following_path, boundary_id):
        risk = 0.8 if following_path.read_bytes() == b"bad" else 0.02
        return {
            "boundary_id": boundary_id,
            "metrics": {"provisional_risk_score": risk},
        }

    kwargs = {
        "execute_chunk": execute,
        "inspect_seam": inspect,
        "materialize_shot": _materialize_test_shot,
        "seam_calibration": _certified_seam_calibration().model_dump(mode="json"),
        "max_seam_repairs": 1,
    }
    failed = execute_continuity_plan(plan, tmp_path, **kwargs)
    resumed = execute_continuity_plan(plan, tmp_path, **kwargs)

    assert failed["status"] == "error"
    assert resumed["status"] == "done"
    assert calls == ["S01_C01", "S01_C02", "S01_C02_R01", "S01_C02_R01", "S01_C03"]


def test_auto_runtime_is_default_and_defers_uncalibrated_seams_to_phase8(
    monkeypatch, tmp_path
):
    write_continuity_plan(
        tmp_path / "CONTINUITY_PLAN.json",
        {"shots": [{"id": "S01", "duration": 16}]},
    )
    monkeypatch.delenv("HONCUT_CONTINUITY_MODE", raising=False)

    report = write_shadow_runtime_report(tmp_path)

    assert report["mode"] == "auto"
    assert report["execution_enabled"] is True
    assert report["phase6_seam_policy"] == "observe_only"
    assert report["chunk_count"] == 2
    assert json.loads((tmp_path / "CONTINUITY_RUNTIME.json").read_text())["mode"] == "auto"


def test_auto_mode_fails_before_any_provider_execution(monkeypatch, tmp_path):
    monkeypatch.setenv("HONCUT_CONTINUITY_MODE", "auto")

    with pytest.raises(RuntimeError, match="seam guard"):
        write_shadow_runtime_report(tmp_path)


def test_auto_mode_requires_and_records_certified_calibration(monkeypatch, tmp_path):
    write_continuity_plan(
        tmp_path / "CONTINUITY_PLAN.json",
        {"shots": [{"id": "S01", "duration": 16}]},
    )
    calibration = _certified_seam_calibration()
    (tmp_path / "CONTINUITY_CALIBRATION.json").write_text(
        calibration.model_dump_json(indent=2),
        encoding="utf-8",
    )
    monkeypatch.setenv("HONCUT_CONTINUITY_MODE", "auto")

    report = write_shadow_runtime_report(tmp_path)

    assert report["mode"] == "auto"
    assert report["execution_enabled"] is True
    assert report["calibration_fingerprint"] == calibration.dataset_fingerprint


def test_extension_provider_content_keeps_images_as_anchors_and_adds_video(monkeypatch, tmp_path):
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "walk steadily right", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    board_dir = tmp_path / "storyboard_groups"
    board_dir.mkdir()
    (board_dir / "CG001.jpg").write_bytes(b"group-board")
    (tmp_path / "STORYBOARD_GROUPS.json").write_text(json.dumps({
        "shot_to_group": {"S01": "CG001"},
        "groups": [{
            "group_id": "CG001",
            "storyboard_board": "storyboard_groups/CG001.jpg",
            "beats": [{
                "shot_id": "S01", "start_state": "at roof edge",
                "generation_actions": ["walk steadily right"], "end_state": "at door",
            }],
        }],
    }), encoding="utf-8")
    previous = tmp_path / "previous.mp4"
    previous.write_bytes(b"video")
    monkeypatch.setattr(
        "tools.asset_packager.build_content_for_shot",
        lambda **kwargs: [
            {"type": "text", "text": kwargs["shot_meta"]["prompt"]},
            {
                "type": "image_url",
                "image_url": {"url": "https://image.test/frame.png"},
                "role": "first_frame",
                "priority": "high",
            },
        ],
    )
    monkeypatch.setattr(
        "clients.tos_uploader.upload_media_file",
        lambda path, prefix: (
            "https://video.test/tail-window.mp4"
            if prefix == "volcengine/video"
            else f"https://image.test/{Path(path).name}"
        ),
    )
    monkeypatch.setattr(
        "clients.tos_uploader.upload_image",
        lambda data, content_type: "https://image.test/CG001.jpg",
    )
    monkeypatch.setattr(
        "quality.continuity_seam.extract_video_tail_window",
        lambda _video, output, window_s: output.parent.mkdir(parents=True, exist_ok=True)
        or output.write_bytes(f"tail-{window_s}".encode())
        or output,
    )

    def fake_extract_frames(_video, outputs):
        for index, output in enumerate(outputs, 1):
            output.write_bytes(f"frame-{index}".encode())
        return tuple(outputs)

    monkeypatch.setattr(
        "quality.continuity_seam.extract_ordered_video_frames",
        fake_extract_frames,
    )
    chunk = GenerationChunk(
        chunk_id="S01_C02",
        sequence=2,
        target_duration_s=8,
        mode="native_extend",
        depends_on="S01_C01",
    )
    request = ChunkExecutionRequest(
        resource_id="S01_C02",
        shot_id="S01",
        chunk=chunk,
        anchors={"scene": "roof"},
        output_path=tmp_path / "S01_C02.mp4",
        previous_output_path=previous,
        input_fingerprint="fingerprint",
        memory_context="canonical anchors remain authoritative",
        repair_attempt=1,
    )

    content, _meta, _seed, duration = _provider_content(tmp_path, request)

    assert duration == 8
    assert [item.get("role") for item in content] == [
        None,
        "reference_image",
        "reference_image",
        "reference_image",
        "reference_image",
        "reference_image",
        "reference_video",
    ]
    assert "向后延长视频1" in content[0]["text"]
    assert "图片2、图片3、图片4" in content[0]["text"]
    assert "严格参考图片4" in content[0]["text"]
    assert "不得重播视频1中的运动轨迹" in content[0]["text"]
    assert "without a reset or cut" in content[0]["text"]
    assert "Do not skip forward in time" in content[0]["text"]
    assert "storyboard group CG001; step 1/1" in content[0]["text"]
    assert "图片5是本连续组" in content[0]["text"]
    assert "必须按当前格中的主体动作箭头和摄影机箭头" in content[0]["text"]
    assert "不得在成片中生成箭头、辅助线" in content[0]["text"]
    assert content[0]["text"].count("[storyboard-motion-notation]") == 1
    assert content[1]["image_url"]["url"] == "https://image.test/frame.png"
    assert [item["image_url"]["url"] for item in content[2:5]] == [
        f"https://image.test/{path.name}"
        for path in sorted((tmp_path / "continuity_anchors").glob("*_frame_*.jpg"))
    ]
    assert content[5]["image_url"]["url"] == "https://image.test/CG001.jpg"
    assert content[-1]["video_url"]["url"] == "https://video.test/tail-window.mp4"


@pytest.mark.parametrize(
    ("source_size", "unchanged_axis"),
    [((400, 100), "width"), ((100, 400), "height")],
)
def test_seedance_group_board_payload_pads_extreme_aspect_without_cropping(
    tmp_path,
    source_size,
    unchanged_axis,
):
    board = tmp_path / "group-board.png"
    Image.new("RGB", source_size, (240, 240, 240)).save(board)

    payload, content_type = _seedance_reference_image_payload(board)

    with Image.open(io.BytesIO(payload)) as adapted:
        ratio = adapted.width / adapted.height
        assert 0.40 <= ratio <= 2.50
        if unchanged_axis == "width":
            assert adapted.width == source_size[0]
            assert adapted.height > source_size[1]
        else:
            assert adapted.height == source_size[1]
            assert adapted.width > source_size[0]
    assert content_type == "image/png"


def test_seedance_group_board_payload_preserves_legal_image_bytes(tmp_path):
    board = tmp_path / "group-board.png"
    Image.new("RGB", (160, 90), "white").save(board)
    original = board.read_bytes()

    payload, content_type = _seedance_reference_image_payload(board)

    assert payload == original
    assert content_type == "image/png"


def test_fresh_provider_does_not_mix_first_frame_with_group_board(monkeypatch, tmp_path):
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "凛踩水冲出", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    board_dir = tmp_path / "storyboard_groups"
    board_dir.mkdir()
    (board_dir / "CG001.jpg").write_bytes(b"group-board")
    (tmp_path / "STORYBOARD_GROUPS.json").write_text(json.dumps({
        "shot_to_group": {"S01": "CG001"},
        "groups": [{
            "group_id": "CG001",
            "storyboard_board": "storyboard_groups/CG001.jpg",
            "beats": [{
                "shot_id": "S01",
                "start_state": "二人对峙",
                "generation_actions": ["凛踩水冲出"],
                "end_state": "凛逼近烬",
            }],
        }],
    }), encoding="utf-8")
    uploaded = []
    monkeypatch.setattr(
        "tools.asset_packager.build_content_for_shot",
        lambda **kwargs: [
            {"type": "text", "text": kwargs["shot_meta"]["prompt"]},
            {
                "type": "image_url",
                "image_url": {"url": "https://image.test/S01.png"},
                "role": "first_frame",
            },
        ],
    )
    monkeypatch.setattr(
        "clients.tos_uploader.upload_image",
        lambda *_args, **_kwargs: uploaded.append(True),
    )
    request = ChunkExecutionRequest(
        resource_id="S01_C01",
        shot_id="S01",
        chunk=GenerationChunk(
            chunk_id="S01_C01",
            sequence=1,
            target_duration_s=4,
            mode="fresh",
            storyboard_beat_id="S01_P01",
            storyboard_image="storyboard_beats/S01_P01.png",
            action_prompt="凛踩水冲出",
        ),
        anchors={"scene": "roof"},
        output_path=tmp_path / "S01_C01.mp4",
        previous_output_path=None,
        input_fingerprint="fingerprint",
        memory_context="anchors",
    )

    content, *_ = _provider_content(tmp_path, request)

    assert [item.get("role") for item in content] == [None, "first_frame"]
    assert uploaded == []
    assert "storyboard group CG001; step 1/1" in content[0]["text"]


def test_fresh_provider_content_uses_current_frame_and_group_storyboard(monkeypatch, tmp_path):
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "凛踩水冲出", "gen_strategy": "phantom"}),
        encoding="utf-8",
    )
    board_dir = tmp_path / "storyboard_groups"
    board_dir.mkdir()
    (board_dir / "CG001.jpg").write_bytes(b"group-board")
    (tmp_path / "STORYBOARD_GROUPS.json").write_text(json.dumps({
        "version": 1,
        "shot_to_group": {"S01": "CG001", "S02": "CG001"},
        "groups": [{
            "group_id": "CG001",
            "storyboard_board": "storyboard_groups/CG001.jpg",
            "beats": [
                {"shot_id": "S01", "start_state": "二人对峙", "generation_actions": ["凛踩水冲出"], "end_state": "凛逼近烬"},
                {"shot_id": "S02", "start_state": "凛逼近烬", "generation_actions": ["烬举臂格挡"], "end_state": "火星炸开"},
            ],
        }],
    }), encoding="utf-8")
    observed = {}

    def fake_build(**kwargs):
        observed["max_images"] = kwargs["shot_meta"].get("_max_reference_images")
        return [
            {"type": "text", "text": kwargs["shot_meta"]["prompt"]},
            {"type": "image_url", "image_url": {"url": "https://image.test/S01.png"}, "role": "reference_image"},
        ]

    monkeypatch.setattr("tools.asset_packager.build_content_for_shot", fake_build)
    monkeypatch.setattr(
        "clients.tos_uploader.upload_image",
        lambda data, content_type: "https://image.test/CG001.jpg",
    )
    request = ChunkExecutionRequest(
        resource_id="S01_C01",
        shot_id="S01",
        chunk=GenerationChunk(chunk_id="S01_C01", sequence=1, target_duration_s=4, mode="fresh"),
        anchors={"scene": "roof"},
        output_path=tmp_path / "S01_C01.mp4",
        previous_output_path=None,
        input_fingerprint="fingerprint",
        memory_context="anchors",
    )

    content, *_ = _provider_content(tmp_path, request)

    assert observed["max_images"] == 8
    assert [item.get("image_url", {}).get("url") for item in content[1:]] == [
        "https://image.test/S01.png",
        "https://image.test/CG001.jpg",
    ]
    assert "storyboard group CG001; step 1/2" in content[0]["text"]
    assert "Execute only this current shot action contract: 凛踩水冲出" in content[0]["text"]
    assert "图片2是本连续组" in content[0]["text"]
    assert "不得同时演完其他格" in content[0]["text"]


def test_extension_provider_content_never_exceeds_seedance_image_budget(monkeypatch, tmp_path):
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "continue", "gen_strategy": "phantom"}),
        encoding="utf-8",
    )
    previous = tmp_path / "previous.mp4"
    previous.write_bytes(b"video")
    observed = {}

    def fake_build(**kwargs):
        observed["max_images"] = kwargs["shot_meta"].get("_max_reference_images")
        return [
            {"type": "text", "text": kwargs["shot_meta"]["prompt"]},
            *[
                {
                    "type": "image_url",
                    "image_url": {"url": f"https://image.test/base-{index}.png"},
                    "role": "reference_image",
                }
                for index in range(8)
            ],
        ]

    monkeypatch.setattr("tools.asset_packager.build_content_for_shot", fake_build)
    monkeypatch.setattr(
        "clients.tos_uploader.upload_media_file",
        lambda path, prefix: f"https://media.test/{Path(path).name}",
    )
    monkeypatch.setattr(
        "quality.continuity_seam.extract_video_tail_window",
        lambda _video, output, window_s: output.parent.mkdir(parents=True, exist_ok=True)
        or output.write_bytes(b"tail")
        or output,
    )

    def fake_extract_frames(_video, outputs):
        for output in outputs:
            output.write_bytes(b"frame")
        return tuple(outputs)

    monkeypatch.setattr(
        "quality.continuity_seam.extract_ordered_video_frames",
        fake_extract_frames,
    )
    chunk = GenerationChunk(
        chunk_id="S01_C02",
        sequence=2,
        target_duration_s=8,
        mode="native_extend",
        depends_on="S01_C01",
    )
    request = ChunkExecutionRequest(
        resource_id="S01_C02",
        shot_id="S01",
        chunk=chunk,
        anchors={"scene": "roof"},
        output_path=tmp_path / "S01_C02.mp4",
        previous_output_path=previous,
        input_fingerprint="fingerprint",
        memory_context="anchors",
    )

    content, *_ = _provider_content(tmp_path, request)
    images = [item for item in content if item.get("type") == "image_url"]

    assert observed["max_images"] == 6
    assert len(images) == 9
    assert "图片7、图片8、图片9" in content[0]["text"]
    assert "严格参考图片9" in content[0]["text"]


def test_continuity_generation_seed_is_stable_but_differs_by_chunk_and_repair(tmp_path):
    first = _fresh_chunk_request(tmp_path)
    second = ChunkExecutionRequest(
        resource_id="S01_C02",
        shot_id="S01",
        chunk=GenerationChunk(
            chunk_id="S01_C02",
            sequence=2,
            target_duration_s=8,
            mode="native_extend",
            depends_on="S01_C01",
        ),
        anchors={"scene": "roof"},
        output_path=tmp_path / "S01_C02.mp4",
        previous_output_path=tmp_path / "S01_C01.mp4",
        input_fingerprint="fingerprint",
        memory_context="anchors",
    )
    repair = ChunkExecutionRequest(**{**second.__dict__, "repair_attempt": 1})

    assert _generation_seed(first) == _generation_seed(first)
    assert len({_generation_seed(first), _generation_seed(second), _generation_seed(repair)}) == 3


def _fresh_chunk_request(tmp_path):
    chunk = GenerationChunk(
        chunk_id="S01_C01",
        sequence=1,
        target_duration_s=8,
        mode="fresh",
    )
    return ChunkExecutionRequest(
        resource_id="S01_C01",
        shot_id="S01",
        chunk=chunk,
        anchors={"scene": "roof"},
        output_path=tmp_path / "shots/S01/chunks/S01_C01.mp4",
        previous_output_path=None,
        input_fingerprint="fingerprint",
        memory_context="canonical anchors remain authoritative",
    )


def test_direct_continuity_adapter_reuses_succeeded_paid_task(monkeypatch, tmp_path):
    monkeypatch.setattr("utils.video_validation.is_valid_video", lambda _path: True)
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "walk steadily right", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARK_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("HONCUT_CAPACITY_DB", str(tmp_path / "capacity.db"))
    monkeypatch.setenv("VIDEO_GEN_CONCURRENCY", "1")
    monkeypatch.setattr(
        "tools.asset_packager.build_content_for_shot",
        lambda **kwargs: [{"type": "text", "text": kwargs["shot_meta"]["prompt"]}],
    )
    submissions = []
    monkeypatch.setattr(
        seedance_client,
        "submit_content",
        lambda content, **kwargs: submissions.append(content) or "seedance-job-1",
    )
    monkeypatch.setattr(
        seedance_client,
        "poll",
        lambda task_id, api_key: "https://video.test/output.mp4",
    )
    monkeypatch.setattr(
        seedance_client,
        "download",
        lambda url, path: Path(path).write_bytes(b"video") or path,
    )
    request = _fresh_chunk_request(tmp_path)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    execute = _direct_seedance_executor(
        tmp_path,
        GenerationTaskStore(tmp_path / "runtime.db"),
    )

    first = execute(request)
    recovered = execute(request)

    assert first.provider_task_id == "seedance-job-1"
    assert recovered.provider_task_id == "seedance-job-1"
    assert len(submissions) == 1


def test_direct_continuity_adapter_drops_provider_rejected_privacy_images_once(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("utils.video_validation.is_valid_video", lambda _path: True)
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "continue synthetic android fight", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARK_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("HONCUT_CAPACITY_DB", str(tmp_path / "capacity.db"))
    monkeypatch.setenv("VIDEO_GEN_CONCURRENCY", "1")
    provider_content = [
        {"type": "text", "text": "synthetic androids only"},
        {"type": "image_url", "image_url": {"url": "safe-storyboard"}},
        {"type": "image_url", "image_url": {"url": "rejected-1"}},
        {"type": "image_url", "image_url": {"url": "rejected-2"}},
        {"type": "image_url", "image_url": {"url": "rejected-3"}},
        {"type": "image_url", "image_url": {"url": "safe-group-board"}},
    ]
    monkeypatch.setattr(
        "tools.asset_packager.build_content_for_shot",
        lambda **_kwargs: [dict(item) for item in provider_content],
    )
    submissions = []

    def fake_submit(content, **_kwargs):
        submissions.append(content)
        if len(submissions) == 1:
            raise RuntimeError(
                "InputImageSensitiveContentDetected.PrivacyInformation: "
                "content[2], content[3], content[4] may contain real person"
            )
        return "seedance-job-corrected"

    monkeypatch.setattr(seedance_client, "submit_content", fake_submit)
    monkeypatch.setattr(
        seedance_client,
        "poll",
        lambda task_id, api_key: "https://video.test/output.mp4",
    )
    monkeypatch.setattr(
        seedance_client,
        "download",
        lambda url, path: Path(path).write_bytes(b"video") or path,
    )
    request = _fresh_chunk_request(tmp_path)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    execute = _direct_seedance_executor(
        tmp_path,
        GenerationTaskStore(tmp_path / "runtime.db"),
    )

    result = execute(request)

    assert result.provider_task_id == "seedance-job-corrected"
    assert len(submissions) == 2
    retained_urls = [
        item["image_url"]["url"]
        for item in submissions[1]
        if item.get("type") == "image_url"
    ]
    assert retained_urls == ["safe-storyboard", "safe-group-board"]
    assert len(result.privacy_policy_repairs) == 1
    assert result.privacy_policy_repairs[0]["removed_content_indices"] == [2, 3, 4]


def test_direct_continuity_adapter_handles_two_indexed_privacy_rounds(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("utils.video_validation.is_valid_video", lambda _path: True)
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "continue authored motion", "gen_strategy": "phantom"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARK_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("HONCUT_CAPACITY_DB", str(tmp_path / "capacity.db"))
    monkeypatch.setenv("VIDEO_GEN_CONCURRENCY", "1")
    provider_content = [
        {"type": "text", "text": "authored motion"},
        {"type": "image_url", "image_url": {"url": "rejected-first"}},
        {"type": "image_url", "image_url": {"url": "rejected-second"}},
        {"type": "image_url", "image_url": {"url": "retained"}},
    ]
    monkeypatch.setattr(
        "tools.asset_packager.build_content_for_shot",
        lambda **_kwargs: [dict(item) for item in provider_content],
    )
    submissions = []

    def fake_submit(content, **_kwargs):
        submissions.append(content)
        if len(submissions) <= 2:
            raise RuntimeError(
                "InputImageSensitiveContentDetected.PrivacyInformation: "
                "content[1] may contain real person"
            )
        return "seedance-job-after-two-repairs"

    monkeypatch.setattr(seedance_client, "submit_content", fake_submit)
    monkeypatch.setattr(
        seedance_client,
        "poll",
        lambda task_id, api_key: "https://video.test/output.mp4",
    )
    monkeypatch.setattr(
        seedance_client,
        "download",
        lambda url, path: Path(path).write_bytes(b"video") or path,
    )
    request = _fresh_chunk_request(tmp_path)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)

    result = _direct_seedance_executor(
        tmp_path,
        GenerationTaskStore(tmp_path / "runtime.db"),
    )(request)

    assert result.provider_task_id == "seedance-job-after-two-repairs"
    assert len(submissions) == 3
    assert [
        item.get("image_url", {}).get("url")
        for item in submissions[-1]
        if item.get("type") == "image_url"
    ] == ["retained"]
    assert [repair["attempt"] for repair in result.privacy_policy_repairs] == [1, 2]


def test_native_extension_privacy_rejection_drops_video_but_keeps_ordered_frames(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("utils.video_validation.is_valid_video", lambda _path: True)
    shot_dir = tmp_path / "shots/S02"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "continue without reset", "gen_strategy": "phantom"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARK_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("HONCUT_CAPACITY_DB", str(tmp_path / "capacity.db"))
    monkeypatch.setenv("VIDEO_GEN_CONCURRENCY", "1")
    internal_content = [
        {"type": "text", "text": "向后延长视频1。"},
        {
            "type": "image_url",
            "image_url": {"url": "tail-frame-1"},
            "role": "reference_image",
            "_continuity_role": "ordered_tail_frame",
        },
        {
            "type": "image_url",
            "image_url": {"url": "tail-frame-2"},
            "role": "reference_image",
            "_continuity_role": "ordered_tail_frame",
        },
        {
            "type": "video_url",
            "video_url": {"url": "rejected-tail-window"},
            "role": "reference_video",
            "_continuity_role": "tail_window_video",
        },
    ]
    monkeypatch.setattr(
        "runtime.continuity_provider._provider_content",
        lambda _root, _request: ([dict(item) for item in internal_content], {}, None, 6),
    )
    submissions = []

    def fake_submit(content, **_kwargs):
        submissions.append(content)
        if len(submissions) == 1:
            raise RuntimeError(
                "InputVideoSensitiveContentDetected.PrivacyInformation: "
                "content[3] may contain real person"
            )
        return "seedance-frame-only-extension"

    monkeypatch.setattr(seedance_client, "submit_content", fake_submit)
    monkeypatch.setattr(
        seedance_client,
        "poll",
        lambda task_id, api_key: "https://video.test/output.mp4",
    )
    monkeypatch.setattr(
        seedance_client,
        "download",
        lambda url, path: Path(path).write_bytes(b"video") or path,
    )
    previous = tmp_path / "shots/S02/chunks/S02_C01.mp4"
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b"previous")
    request = ChunkExecutionRequest(
        resource_id="S02_C02",
        shot_id="S02",
        chunk=GenerationChunk(
            chunk_id="S02_C02",
            sequence=2,
            target_duration_s=6,
            mode="native_extend",
            depends_on="S02_C01",
            execution_strategy="tail_video_extend",
        ),
        anchors={},
        output_path=tmp_path / "shots/S02/chunks/S02_C02.mp4",
        previous_output_path=previous,
        input_fingerprint="privacy-video-fingerprint",
        memory_context="",
    )

    result = _direct_seedance_executor(
        tmp_path,
        GenerationTaskStore(tmp_path / "runtime.db"),
    )(request)

    assert result.provider_task_id == "seedance-frame-only-extension"
    assert len(submissions) == 2
    assert not any(item.get("type") == "video_url" for item in submissions[1])
    assert sum(item.get("type") == "image_url" for item in submissions[1]) == 2
    assert "privacy-safe frame-only continuity fallback" in submissions[1][0]["text"]
    assert not any(
        any(str(key).startswith("_") for key in item)
        for submission in submissions
        for item in submission
    )
    assert result.privacy_policy_repairs[0]["policy"] == (
        "drop_rejected_video_keep_ordered_tail_frames_v1"
    )


def test_flf2v_privacy_rejection_uses_local_handle_passthrough_without_dropping_endpoint(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("utils.video_validation.is_valid_video", lambda _path: True)
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "bridge adjacent primaries"}), encoding="utf-8"
    )
    (tmp_path / "CONTINUITY_PLAN.json").write_text(
        json.dumps({
            "timeline_fps": 24,
            "bridges": [{
                "source_shot_id": "S01",
                "target_shot_id": "S02",
                "source_handle_s": 2.0,
                "target_handle_s": 2.0,
                "visible_duration_s": 4.0,
            }],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARK_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("HONCUT_CAPACITY_DB", str(tmp_path / "capacity.db"))
    monkeypatch.setenv("VIDEO_GEN_CONCURRENCY", "1")
    endpoint_content = [
        {"type": "text", "text": "bridge"},
        {
            "type": "image_url",
            "image_url": {"url": "required-first"},
            "role": "first_frame",
            "_continuity_role": "required_first_frame",
        },
        {
            "type": "image_url",
            "image_url": {"url": "required-last"},
            "role": "last_frame",
            "_continuity_role": "required_last_frame",
        },
    ]
    monkeypatch.setattr(
        "runtime.continuity_provider._provider_content",
        lambda _root, _request: ([dict(item) for item in endpoint_content], {}, None, 4),
    )
    submissions = []

    def rejected_submit(content, **_kwargs):
        submissions.append(content)
        raise RuntimeError(
            "InputImageSensitiveContentDetected.PrivacyInformation: "
            "content[1] may contain real person"
        )

    monkeypatch.setattr(seedance_client, "submit_content", rejected_submit)
    rendered = {}

    def fake_render(source, target, output, **kwargs):
        rendered.update(source=source, target=target, **kwargs)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"local-valid-video")
        return {
            "policy": "local_boundary_handle_passthrough_v1",
            "provider_generation": False,
        }

    monkeypatch.setattr(
        "runtime.continuity_provider._render_privacy_safe_handle_bridge",
        fake_render,
    )
    source = tmp_path / "shots/S01/output.mp4"
    target = tmp_path / "shots/S02/output.mp4"
    source.write_bytes(b"source")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"target")
    request = ChunkExecutionRequest(
        resource_id="S01__S02_B01",
        shot_id="S01",
        chunk=GenerationChunk(
            chunk_id="S01__S02_B01",
            sequence=1,
            target_duration_s=4,
            requested_frames=96,
            expected_unique_frames=96,
            mode="native_extend",
            depends_on="S01_C01",
            execution_strategy="first_last_frame_bridge",
            bridge_target_shot_id="S02",
        ),
        anchors={},
        output_path=tmp_path / "shot_bridges/S01__S02.mp4",
        previous_output_path=source,
        target_output_path=target,
        input_fingerprint="bridge-privacy-fingerprint",
        memory_context="",
    )

    result = _direct_seedance_executor(
        tmp_path,
        GenerationTaskStore(tmp_path / "runtime.db"),
    )(request)

    assert len(submissions) == 1
    assert [item.get("role") for item in submissions[0][1:]] == [
        "first_frame",
        "last_frame",
    ]
    assert result.provider_task_id is None
    assert result.provider_fallback["policy"] == "local_boundary_handle_passthrough_v1"
    assert result.privacy_policy_repairs[0]["removed_content_indices"] == []
    assert rendered["source_handle_s"] == 2.0
    assert rendered["target_handle_s"] == 2.0


def test_privacy_safe_handle_bridge_has_exact_frame_budget(tmp_path):
    source = tmp_path / "source.mp4"
    target = tmp_path / "target.mp4"
    output = tmp_path / "bridge.mp4"
    for path, color in ((source, "blue"), (target, "red")):
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                f"color=c={color}:s=160x90:r=24:d=3",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
            ],
            check=True,
        )

    receipt = _render_privacy_safe_handle_bridge(
        source,
        target,
        output,
        source_handle_s=1.0,
        target_handle_s=1.0,
        timeline_fps=24,
        width=160,
        height=90,
    )

    assert probe_continuity_frames(output, 24)["frames"] == 48
    assert receipt["frames"] == 48
    assert receipt["transition_effect"] == "none_hard_cut_preserved"


def test_direct_continuity_adapter_rewrites_output_audio_policy_once(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("utils.video_validation.is_valid_video", lambda _path: True)
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "dance to music and rhythm", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARK_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("HONCUT_CAPACITY_DB", str(tmp_path / "capacity.db"))
    monkeypatch.setenv("VIDEO_GEN_CONCURRENCY", "1")
    monkeypatch.setattr(
        "tools.asset_packager.build_content_for_shot",
        lambda **kwargs: [{"type": "text", "text": kwargs["shot_meta"]["prompt"]}],
    )
    submissions: list[list[dict]] = []
    submitted_seeds: list[int | None] = []

    def fake_submit(content, **kwargs):
        submissions.append([dict(item) for item in content])
        submitted_seeds.append(kwargs.get("seed"))
        return f"seedance-job-{len(submissions)}"

    def fake_poll(task_id, api_key):
        assert api_key == "test-key"
        if task_id == "seedance-job-1":
            raise ProviderJobFailedError(
                "OutputAudioSensitiveContentDetected.PolicyViolation"
            )
        return "https://video.test/output.mp4"

    monkeypatch.setattr(seedance_client, "submit_content", fake_submit)
    monkeypatch.setattr(seedance_client, "poll", fake_poll)
    monkeypatch.setattr(
        seedance_client,
        "download",
        lambda url, path: Path(path).write_bytes(b"video") or path,
    )
    request = _fresh_chunk_request(tmp_path)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    execute = _direct_seedance_executor(
        tmp_path,
        GenerationTaskStore(tmp_path / "runtime.db"),
    )

    result = execute(request)
    recovered = execute(request)

    assert result.provider_task_id == "seedance-job-2"
    assert recovered.provider_task_id == "seedance-job-2"
    assert len(submissions) == 2
    retry_prompt = submissions[1][0]["text"]
    assert "original ambient location sounds only" in retry_prompt
    assert "No music" in retry_prompt
    assert "copyrighted audio" in retry_prompt.casefold()
    assert "dance to music and rhythm" not in retry_prompt
    assert submitted_seeds[0] != submitted_seeds[1]
    assert result.copyright_policy_repairs == ({
        "attempt": 1,
        "reason_code": "OutputAudioSensitiveContentDetected.PolicyViolation",
        "policy": "original_ambient_no_music_v1",
        "removed_content_indices": [],
    },)


def test_direct_continuity_adapter_drops_rejected_video_but_keeps_anchor_frames(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("utils.video_validation.is_valid_video", lambda _path: True)
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "continue the authored movement", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARK_AGENT_API_KEY", "test-key")
    monkeypatch.setenv("HONCUT_CAPACITY_DB", str(tmp_path / "capacity.db"))
    monkeypatch.setenv("VIDEO_GEN_CONCURRENCY", "1")
    provider_content = [
        {"type": "text", "text": "continue from ordered anchors"},
        {"type": "image_url", "image_url": {"url": "anchor-1"}},
        {"type": "image_url", "image_url": {"url": "anchor-2"}},
        {"type": "image_url", "image_url": {"url": "anchor-3"}},
        {"type": "video_url", "video_url": {"url": "rejected-tail-video"}},
    ]
    monkeypatch.setattr(
        "runtime.continuity_provider._provider_content",
        lambda *_args, **_kwargs: (
            [dict(item) for item in provider_content],
            {},
            None,
            8,
        ),
    )
    submissions: list[list[dict]] = []

    def fake_submit(content, **_kwargs):
        submissions.append([dict(item) for item in content])
        if len(submissions) == 1:
            raise RuntimeError(
                "InputVideoSensitiveContentDetected.PolicyViolation: content[4]"
            )
        return "seedance-job-frame-only"

    monkeypatch.setattr(seedance_client, "submit_content", fake_submit)
    monkeypatch.setattr(
        seedance_client,
        "poll",
        lambda task_id, api_key: "https://video.test/output.mp4",
    )
    monkeypatch.setattr(
        seedance_client,
        "download",
        lambda url, path: Path(path).write_bytes(b"video") or path,
    )
    previous = tmp_path / "shots/S01/chunks/S01_C01.mp4"
    previous.parent.mkdir(parents=True, exist_ok=True)
    previous.write_bytes(b"previous")
    request = ChunkExecutionRequest(
        resource_id="S01_C02",
        shot_id="S01",
        chunk=GenerationChunk(
            chunk_id="S01_C02",
            sequence=2,
            target_duration_s=8,
            mode="native_extend",
            depends_on="S01_C01",
            execution_strategy="tail_video_extend",
        ),
        anchors={"scene": "street"},
        output_path=tmp_path / "shots/S01/chunks/S01_C02.mp4",
        previous_output_path=previous,
        input_fingerprint="fingerprint",
        memory_context="anchors",
    )
    execute = _direct_seedance_executor(
        tmp_path,
        GenerationTaskStore(tmp_path / "runtime.db"),
    )

    result = execute(request)

    assert result.provider_task_id == "seedance-job-frame-only"
    assert len(submissions) == 2
    assert not any(item.get("type") == "video_url" for item in submissions[1])
    assert [
        item["image_url"]["url"]
        for item in submissions[1]
        if item.get("type") == "image_url"
    ] == ["anchor-1", "anchor-2", "anchor-3"]
    assert "frame-only continuity fallback" in submissions[1][0]["text"]
    assert result.copyright_policy_repairs[0]["removed_content_indices"] == [4]
    assert result.copyright_policy_repairs[0]["retained_reference_images"] == 3


def test_bridge_continuity_adapter_reuses_succeeded_paid_task(monkeypatch, tmp_path):
    monkeypatch.setattr("utils.video_validation.is_valid_video", lambda _path: True)
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "walk steadily right", "gen_strategy": "i2v"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("VIDEO_GEN_CONCURRENCY", "1")
    monkeypatch.setattr(
        "tools.asset_packager.build_content_for_shot",
        lambda **kwargs: [{"type": "text", "text": kwargs["shot_meta"]["prompt"]}],
    )
    monkeypatch.setattr(
        "clients.local_video_client.get_api_url",
        lambda: "http://bridge.test",
    )
    generations = []

    def fake_generate_video(**kwargs):
        generations.append(kwargs.get("resume_task_id"))
        kwargs["on_submit_start"]()
        kwargs["on_submitted"]("bridge-job-1")
        Path(kwargs["output_path"]).write_bytes(b"video")
        return kwargs["output_path"]

    monkeypatch.setattr(
        "clients.local_video_client.generate_video",
        fake_generate_video,
    )
    request = _fresh_chunk_request(tmp_path)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    execute = _bridge_seedance_executor(
        tmp_path,
        GenerationTaskStore(tmp_path / "runtime.db"),
    )

    first = execute(request)
    recovered = execute(request)

    assert first.provider_task_id == "bridge-job-1"
    assert recovered.provider_task_id == "bridge-job-1"
    assert generations == [None]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_extract_video_tail_frame_preserves_provider_video_geometry(tmp_path):
    video = tmp_path / "chunk.mp4"
    anchor = tmp_path / "anchors/chunk_tail.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:d=0.5",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )

    extracted = extract_video_tail_frame(video, anchor)

    image = Image.open(extracted).convert("RGB")
    assert image.size == (160, 90)
    red, green, blue = np.asarray(image).mean(axis=(0, 1))
    assert blue > red * 2
    assert blue > green * 2


def test_extract_video_tail_frame_falls_back_from_undecodable_final_timestamp(
    monkeypatch, tmp_path
):
    video = tmp_path / "chunk.mp4"
    anchor = tmp_path / "anchors/chunk_tail.jpg"
    attempts = []
    video.write_bytes(b"video")

    monkeypatch.setattr("quality.continuity_seam._video_duration", lambda _path: 5.04)

    def fake_extract(_video, timestamp, output):
        attempts.append(timestamp)
        if len(attempts) == 1:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"")
            raise RuntimeError("final timestamp has no decodable frame")
        output.write_bytes(b"jpeg")

    monkeypatch.setattr("quality.continuity_seam._extract_frame", fake_extract)

    extracted = extract_video_tail_frame(video, anchor)

    assert extracted.read_bytes() == b"jpeg"
    assert attempts == pytest.approx([4.96, 4.88])


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_tail_window_and_ordered_frames_preserve_short_temporal_context(tmp_path):
    source = tmp_path / "source.mp4"
    tail = tmp_path / "anchors/tail.mp4"
    frames = tuple(tmp_path / f"anchors/frame_{index}.jpg" for index in range(3))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=24:d=3",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )

    extract_video_tail_window(source, tail, window_s=1.5)
    extracted = extract_ordered_video_frames(tail, frames)
    duration = float(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(tail),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    )

    assert duration == pytest.approx(1.5, abs=0.05)
    assert all(Image.open(path).size == (160, 90) for path in extracted)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg is required",
)
def test_video_replay_similarity_flags_motion_replay_but_not_static_frames(tmp_path):
    moving = tmp_path / "moving.mp4"
    moving_copy = tmp_path / "moving_copy.mp4"
    static = tmp_path / "static.mp4"
    for path, source in (
        (moving, "testsrc2=s=160x90:r=24:d=2"),
        (static, "color=c=blue:s=160x90:r=24:d=2"),
    ):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                source,
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    shutil.copyfile(moving, moving_copy)

    replay = measure_video_replay_similarity(moving, moving_copy)
    static_pair = measure_video_replay_similarity(static, static)

    assert replay["likely_replay"] is True
    assert replay["motion_cosine_similarity"] == pytest.approx(1.0)
    assert static_pair["likely_replay"] is False


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_provider_minimum_padding_is_retimed_to_effective_story_frames(tmp_path):
    source = tmp_path / "S03_C03.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=24:d=4",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    chunk = GenerationChunk(
        chunk_id="S03_C03",
        sequence=3,
        target_duration_s=4,
        requested_frames=96,
        expected_provider_padding_frames=24,
        expected_unique_frames=72,
        mode="native_extend",
        depends_on="S03_C02",
        execution_strategy="first_last_frame_bridge",
        bridge_target_shot_id="S04",
        bridge_target_beat_id="S04_P01",
        bridge_target_storyboard_image="storyboard_beats/S04_P01.png",
    )

    receipt = normalize_provider_minimum_padding(source, chunk, 24)
    normalized = Path(receipt["output_path"])

    assert receipt["method"] == "provider_minimum_endpoint_preserving_retime"
    assert receipt["provider_padding_frames"] == 24
    assert normalized != source
    assert probe_continuity_frames(normalized, 24)["frames"] == 72


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_materialize_continuity_shot_concatenates_accepted_chunks(tmp_path):
    chunks = []
    for index, color in enumerate(("black", "white"), 1):
        path = tmp_path / f"chunk_{index}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=96x54:d=0.5",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        chunks.append(path)
    output = tmp_path / "output.mp4"

    materialize_continuity_shot(chunks, output)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert float(probe.stdout.strip()) == pytest.approx(1.0, abs=0.1)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_phase6_auto_runtime_repairs_real_decoded_seam_before_materializing(monkeypatch, tmp_path):
    plan = write_continuity_plan(
        tmp_path / "CONTINUITY_PLAN.json",
        {"shots": [{"id": "S01", "duration": 16}]},
    )
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "SHOT_META.json").write_text(
        json.dumps({"prompt": "a continuous locked camera shot"}),
        encoding="utf-8",
    )
    calibration = _certified_seam_calibration()
    calls = []

    def fake_executor(_root, _store):
        def execute(request):
            calls.append(request.resource_id)
            color = "white" if request.resource_id == "S01_C02" else "black"
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    (
                        f"color=c={color}:s=96x54:r=24:"
                        f"d={request.chunk.target_duration_s}"
                    ),
                    "-pix_fmt",
                    "yuv420p",
                    str(request.output_path),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            return ChunkExecutionResult(request.output_path, f"job-{request.resource_id}")

        return execute

    monkeypatch.setenv("VIDEO_PROVIDER", "seedance")
    monkeypatch.setenv("VIDEO_GEN_CONCURRENCY", "1")
    monkeypatch.setenv("HONCUT_CONTINUITY_MAX_REPAIRS", "1")
    monkeypatch.setattr(
        "runtime.continuity_provider._direct_seedance_executor",
        fake_executor,
    )
    monkeypatch.setattr(
        "clients.tos_uploader.is_media_upload_configured",
        lambda: True,
    )

    report = execute_phase6_auto_continuity(tmp_path, plan, calibration)

    lineage = json.loads((tmp_path / "CONTINUITY_LINEAGE.json").read_text())
    seam = lineage["seams"]["S01_C01__S01_C02"]
    assert report["status"] == "done"
    assert report["repair_attempts"] == 1
    assert calls == ["S01_C01", "S01_C02", "S01_C02_R01"]
    assert seam["decision"]["action"] == "accept"
    assert len(seam["attempt_history"]) == 2
    assert (tmp_path / "shots/S01/output.mp4").stat().st_size > 0
    timing = json.loads((tmp_path / "shots/S01/CONTINUITY_TIMING.json").read_text())
    assert timing["target_frames"] == 384
    assert timing["final_frames"] == 384
    assert timing["delta_frames"] == 0


def test_phase6_auto_preflights_extension_upload_before_paid_execution(monkeypatch, tmp_path):
    plan = write_continuity_plan(
        tmp_path / "CONTINUITY_PLAN.json",
        {"shots": [{"id": "S01", "duration": 10}]},
        provider_chunk_limit_s=5,
    )

    def unexpected_executor(_root, _store):
        raise AssertionError("provider executor must not initialize before preflight")

    monkeypatch.setenv("VIDEO_PROVIDER", "seedance")
    monkeypatch.setattr(
        "clients.tos_uploader.is_media_upload_configured",
        lambda: False,
    )
    monkeypatch.setattr(
        "runtime.continuity_provider._direct_seedance_executor",
        unexpected_executor,
    )

    with pytest.raises(RuntimeError, match="before any paid provider submission"):
        execute_phase6_auto_continuity(tmp_path, plan, _certified_seam_calibration())

    assert not (tmp_path / "runtime.db").exists()


def test_phase6_auto_preflights_all_chunk_durations_before_paid_execution(
    monkeypatch,
    tmp_path,
):
    plan = build_continuity_plan({"shots": [{"id": "S01", "duration": 5}]})
    stale_chunk = plan.shots[0].chunks[0]
    stale_chunk.target_duration_s = 3
    stale_chunk.requested_frames = 72
    stale_chunk.expected_unique_frames = 72

    def unexpected_executor(_root, _store):
        raise AssertionError("provider executor must not initialize before duration preflight")

    monkeypatch.setenv("VIDEO_PROVIDER", "seedance")
    monkeypatch.setattr(
        "runtime.continuity_provider._direct_seedance_executor",
        unexpected_executor,
    )

    with pytest.raises(RuntimeError, match="before any paid provider submission"):
        execute_phase6_auto_continuity(tmp_path, plan, _certified_seam_calibration())

    assert not (tmp_path / "runtime.db").exists()


def test_phase6_auto_requires_a_bridge_budgeted_plan_before_provider_init(monkeypatch, tmp_path):
    plan = write_continuity_plan(
        tmp_path / "CONTINUITY_PLAN.json",
        {"shots": [{"id": "S01", "duration": 16}]},
    )

    def unexpected_executor(_root, _store):
        raise AssertionError("provider executor must not initialize for a stale plan")

    monkeypatch.setenv("VIDEO_PROVIDER", "seedance")
    monkeypatch.setenv("HONCUT_CONTINUITY_BRIDGE", "auto")
    monkeypatch.setattr(
        "clients.tos_uploader.is_media_upload_configured",
        lambda: True,
    )
    monkeypatch.setattr(
        "runtime.continuity_provider._direct_seedance_executor",
        unexpected_executor,
    )

    with pytest.raises(RuntimeError, match="rerun Phase 4"):
        execute_phase6_auto_continuity(tmp_path, plan, _certified_seam_calibration())

    assert not (tmp_path / "runtime.db").exists()


def test_phase6_auto_accepts_zero_overlap_first_last_bridge(monkeypatch, tmp_path):
    storyboard = {
        "continuity_mode": "one_take",
        "video_provider": "seedance",
        "shots": [
            {"id": "S01", "duration": 15, "micro_actions": ["Agent穿过舱门"]},
            {"id": "S02", "duration": 15, "micro_actions": ["Agent继续前进"]},
        ],
    }
    plan_storyboard_beats(storyboard)
    plan = build_continuity_plan(storyboard)
    assert plan.shots[0].chunks[-1].execution_strategy == "multi_image"
    assert plan.bridges[0].execution_strategy == "first_last_frame_bridge"
    assert plan.bridges[0].target_duration_s == 4

    monkeypatch.setenv("VIDEO_PROVIDER", "seedance")
    monkeypatch.setenv("HONCUT_CONTINUITY_BRIDGE", "auto")
    monkeypatch.setattr(
        "runtime.continuity_provider._direct_seedance_executor",
        lambda _root, _store: lambda _request: None,
    )
    monkeypatch.setattr(
        "runtime.continuity_provider.execute_continuity_plan",
        lambda *_args, **_kwargs: {"status": "done", "errors": []},
    )

    report = execute_phase6_auto_continuity(tmp_path, plan, None)

    assert report["status"] == "done"


def test_pre_phase8_duration_closure_preserves_overlong_shot(monkeypatch, tmp_path):
    output = tmp_path / "shots/S01/output.mp4"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"provider-video")
    monkeypatch.setattr(
        "runtime.continuity_provider.probe_continuity_frames",
        lambda _path, _fps: {"frames": 250, "duration_s": 250 / 24, "source_fps": 24},
    )
    monkeypatch.setattr(
        "runtime.continuity_provider.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("overlong pre-Phase-8 shot must not be trimmed"),
    )

    receipt = finalize_continuity_shot(output, target_frames=240, timeline_fps=24)

    assert receipt == {
        "method": "deferred_phase8_excess_trim",
        "before_frames": 250,
        "target_frames": 240,
        "after_frames": 250,
        "excess_frames": 10,
        "minimum_target_met": True,
    }
    assert output.read_bytes() == b"provider-video"


def test_phase8_trims_pre_phase8_excess_per_primary_shot(monkeypatch, tmp_path):
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "output.mp4").write_bytes(b"provider-video")
    (shot_dir / "CONTINUITY_TIMING.json").write_text(
        json.dumps({
            "kind": "honcut.continuity_timing.v1",
            "internal_seams_finalized": True,
            "timeline_fps": 24,
            "target_frames": 240,
            "final_frames": 252,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        edit_decision_module,
        "probe_video",
        lambda _path: {"duration": 10.5, "has_audio": False},
    )
    monkeypatch.setattr(
        edit_decision_module,
        "detect_black_frames",
        lambda _path, **_kwargs: {"trim_start": 0.0, "trim_end": 0.0},
    )

    decisions = edit_decision_module.build_edit_decisions(str(tmp_path / "shots"))

    cut = decisions["cuts"][0]
    assert cut["in_seconds"] == 0.0
    assert cut["out_seconds"] == 10.0
    assert cut["phase8_duration_trim"]["discarded_excess_frames"] == 12
    assert decisions["metadata"]["phase8_duration_trims"][0]["shot_id"] == "S01"


def test_chunk_runtime_uses_prepared_following_without_overwriting_provider_clip(tmp_path):
    plan = build_continuity_plan({"shots": [{"id": "S01", "duration": 16}]})

    def execute(request):
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        request.output_path.write_bytes(request.resource_id.encode())
        return ChunkExecutionResult(request.output_path)

    def prepare(_previous, following, boundary_id):
        derived = tmp_path / "derived" / f"{boundary_id}.mp4"
        derived.parent.mkdir(parents=True, exist_ok=True)
        derived.write_bytes(b"prepared-" + following.read_bytes())
        return {"status": "repaired", "output_path": str(derived)}

    def inspect(_previous, following, boundary_id):
        assert boundary_id == "S01_C01__S01_C02"
        assert following.read_bytes() == b"prepared-S01_C02"
        return {"metrics": {"provisional_risk_score": 0.02}}

    def materialize(paths, output):
        output.write_bytes(b"|".join(path.read_bytes() for path in paths))

    report = execute_continuity_plan(
        plan,
        tmp_path,
        execute_chunk=execute,
        prepare_seam=prepare,
        inspect_seam=inspect,
        materialize_shot=materialize,
    )

    lineage = json.loads((tmp_path / "CONTINUITY_LINEAGE.json").read_text())
    assert report["status"] == "done"
    assert report["prepared_seams"] == 1
    assert (tmp_path / "shots/S01/chunks/S01_C02.mp4").read_bytes() == b"S01_C02"
    assert (tmp_path / "shots/S01/output.mp4").read_bytes() == (b"S01_C01|prepared-S01_C02")
    assert lineage["seams"]["S01_C01__S01_C02"]["preparation"]["status"] == "repaired"


def test_continuity_bridge_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("HONCUT_CONTINUITY_BRIDGE", raising=False)

    assert _continuity_bridge_preparer(tmp_path) is None


def test_phase8_temporal_adjudication_requires_object_or_human_corroboration():
    values = [
        0.0466, 0.0462, 0.0463, 0.0460, 0.0450, 0.0439, 0.0434,
        0.0434, 0.0432, 0.0427, 0.0422, 0.0420, 0.0419,
    ]
    candidates = [
        {
            "frames": 48 + index * 2,
            "seconds": (48 + index * 2) / 24,
            "frame_mae": value,
        }
        for index, value in enumerate(values)
    ]

    uncorroborated = decide_temporal_seam(
        candidates,
        planned_overlap_frames=48,
        timeline_fps=24,
    )
    corroborated = decide_temporal_seam(
        candidates,
        planned_overlap_frames=48,
        timeline_fps=24,
        object_trajectory_evidence={
            "verdict": "rollback",
            "confidence": 0.91,
            "repair_action": "hard_trim",
            "recommended_trim_frames": 72,
        },
    )

    assert uncorroborated["action"] == "human_review"
    assert uncorroborated["recommended_action"] == "hard_trim"
    assert corroborated["action"] == "hard_trim"
    assert corroborated["trim_frames"] == 72
    assert corroborated["additional_trim_frames"] == 24


def test_phase8_retries_only_missing_or_transient_sam3_evidence():
    assert _object_evidence_needs_retry(None) is True
    assert _object_evidence_needs_retry(
        {
            "verdict": "unavailable",
            "reason": "SAM 3 trajectory analysis failed: connection reset",
        }
    ) is True
    assert _object_evidence_needs_retry(
        {
            "verdict": "unavailable",
            "reason": "insufficient tracked subject frames across the boundary",
        }
    ) is False
    assert _object_evidence_needs_retry(
        {"verdict": "continuous", "confidence": 0.9}
    ) is False


def test_object_tracking_finds_the_subject_catchup_frame_after_rollback():
    positions = [
        (7, 0.58),
        (8, 0.64),
        (9, 0.70),
        (10, 0.20),
        (11, 0.30),
        (12, 0.40),
        (13, 0.50),
        (14, 0.60),
        (15, 0.70),
    ]
    evidence = decide_object_trajectory(
        [
            {"frame_idx": frame, "centroid": [x, 0.5], "score": 0.95}
            for frame, x in positions
        ],
        seam_frame=10,
        planned_overlap_frames=2,
        screen_direction="left_to_right",
        camera_motion="locked_off",
    )

    assert evidence["verdict"] == "rollback"
    assert evidence["repair_action"] == "hard_trim"
    assert evidence["recommended_trim_frames"] == 5
    assert evidence["confidence"] >= 0.6


def _moving_patch_frames(left_positions, *, frame_count=30):
    frames = np.zeros((frame_count, 36, 64, 3), dtype=np.uint8)
    for frame_index, left in enumerate(left_positions):
        frames[frame_index, 14:20, left : left + 8] = 80
        frames[frame_index, 14:17, left : left + 8] = 240
        frames[frame_index, 15:19, left + 2 : left + 4] = 160
    return frames


def test_sam_bbox_template_tracking_refines_to_an_original_timeline_frame(
    monkeypatch, tmp_path
):
    previous_positions = [max(4, frame_index) for frame_index in range(24)]
    following_positions = [5 + frame_index for frame_index in range(30)]
    previous_frames = _moving_patch_frames(previous_positions, frame_count=24)
    following_frames = _moving_patch_frames(following_positions, frame_count=30)
    previous = tmp_path / "previous.mp4"
    following = tmp_path / "following.mp4"

    monkeypatch.setattr(
        object_trajectory_module,
        "_decode_refinement_frames",
        lambda path: previous_frames if path == previous else following_frames,
    )
    tracked = [
        {"frame_idx": 3, "centroid": [0.30, 0.5]},
        {"frame_idx": 4, "centroid": [0.34, 0.5]},
        {
            "frame_idx": 5,
            "centroid": [0.375, 0.5],
            "bbox": [20 / 64, 14 / 36, 28 / 64, 20 / 36],
        },
        {"frame_idx": 6, "centroid": [0.18, 0.5]},
        {"frame_idx": 7, "centroid": [0.24, 0.5]},
        {"frame_idx": 8, "centroid": [0.29, 0.5]},
        {
            "frame_idx": 9,
            "centroid": [21 / 64, 0.5],
            "bbox": [17 / 64, 14 / 36, 25 / 64, 20 / 36],
        },
    ]

    refinement = object_trajectory_module.refine_object_catchup_frame(
        previous,
        following,
        tracked=tracked,
        seam_frame=6,
        coarse_trim_analysis_frames=4,
        analysis_fps=3,
        timeline_fps=12,
        planned_overlap_frames=8,
        following_frames=30,
        screen_direction="left_to_right",
    )

    assert refinement["status"] == "refined"
    assert refinement["recommended_trim_frames"] == 18
    assert refinement["remaining_frames"] == 12
    assert refinement["median_correlation"] > 0.9


def test_sam3_runtime_auto_policy_uses_stable_fp32_on_apple_silicon():
    policy = resolve_runtime_policy(
        mps_available=True,
        cuda_available=False,
        cpu_count=10,
    )

    assert policy.device == "mps"
    assert policy.precision == "fp32"
    assert policy.quantize_linear is False
    assert policy.cpu_threads == 6


def test_sam3_checkpoint_discovers_shared_sibling_without_copying(tmp_path):
    repo_root = tmp_path / "honcut"
    shared_checkpoint = tmp_path / "sam3" / "权重" / "sam3.pt"
    shared_checkpoint.parent.mkdir(parents=True)
    shared_checkpoint.write_bytes(b"checkpoint")

    assert resolve_checkpoint_path(repo_root) == shared_checkpoint


def test_sam3_explicit_checkpoint_overrides_shared_sibling(tmp_path):
    explicit = tmp_path / "selected.pt"
    shared_checkpoint = tmp_path / "sam3" / "权重" / "sam3.pt"
    shared_checkpoint.parent.mkdir(parents=True)
    shared_checkpoint.write_bytes(b"shared")

    assert resolve_checkpoint_path(
        tmp_path / "honcut",
        configured_checkpoint=str(explicit),
    ) == explicit


def test_sam3_runtime_auto_policy_uses_dynamic_int8_on_cpu():
    policy = resolve_runtime_policy(
        mps_available=False,
        cuda_available=False,
        cpu_count=4,
    )

    assert policy.device == "cpu"
    assert policy.precision == "int8_dynamic"
    assert policy.quantize_linear is True
    assert estimate_weight_bytes(
        total_parameters=100,
        linear_parameters=80,
        precision=policy.precision,
    ) == 160


def test_sam3_runtime_rejects_int8_on_mps():
    with pytest.raises(ValueError, match="only on CPU"):
        resolve_runtime_policy(
            requested_device="mps",
            requested_precision="int8_dynamic",
            mps_available=True,
        )


def test_phase8_sam3_sidecar_is_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("HONCUT_SAM3_MODE", raising=False)
    monkeypatch.delenv("HONCUT_SAM3_URL", raising=False)

    with sam3_sidecar_module.phase8_sam3_endpoint(tmp_path) as endpoint:
        assert endpoint == ""


def test_phase8_sam3_sidecar_preserves_external_url(monkeypatch, tmp_path):
    monkeypatch.setenv("HONCUT_SAM3_MODE", "external")
    monkeypatch.setenv("HONCUT_SAM3_URL", "http://sam3.internal:9000/")

    with sam3_sidecar_module.phase8_sam3_endpoint(tmp_path) as endpoint:
        assert endpoint == "http://sam3.internal:9000"


def test_phase8_managed_sam3_starts_and_releases_owned_process(monkeypatch, tmp_path):
    class FakeProcess:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            assert timeout == 10
            return 0

    process = FakeProcess()
    health_checks = iter([False, True])
    monkeypatch.setenv("HONCUT_SAM3_MODE", "managed")
    monkeypatch.delenv("HONCUT_SAM3_URL", raising=False)
    monkeypatch.setattr(
        sam3_sidecar_module,
        "_service_is_healthy",
        lambda _url: next(health_checks),
    )
    monkeypatch.setattr(
        sam3_sidecar_module,
        "_spawn_local_service",
        lambda _script, _log: process,
    )

    with sam3_sidecar_module.phase8_sam3_endpoint(tmp_path) as endpoint:
        assert endpoint == "http://127.0.0.1:8001"
        assert process.terminated is False

    assert process.terminated is True


def test_sam3_low_fps_trim_is_converted_back_to_timeline_frames(
    monkeypatch, tmp_path
):
    observed = {}
    monkeypatch.setenv("HONCUT_SAM3_ANALYSIS_FPS", "6")
    monkeypatch.setattr(
        object_trajectory_module,
        "build_tracking_clip",
        lambda *_args, **_kwargs: 12,
    )

    class StubClient:
        def __init__(self, _base_url):
            pass

        def track(self, *_args, **_kwargs):
            return [{"frame_idx": 0, "centroid": [0.5, 0.5]}]

    def decide(_frames, **kwargs):
        observed.update(kwargs)
        return {"verdict": "rollback", "recommended_trim_frames": 5}

    monkeypatch.setattr(object_trajectory_module, "Sam3TrajectoryClient", StubClient)
    monkeypatch.setattr(object_trajectory_module, "decide_object_trajectory", decide)

    result = object_trajectory_module.collect_sam3_trajectory(
        tmp_path / "previous.mp4",
        tmp_path / "following.mp4",
        boundary_id="S01_C01__S01_C02",
        evidence_dir=tmp_path,
        prompt="paper boat",
        timeline_fps=24,
        planned_overlap_frames=48,
        following_frames=120,
        screen_direction="left_to_right",
        camera_motion="locked_off",
        base_url="http://127.0.0.1:8001",
    )

    assert observed["seam_frame"] == 12
    assert observed["planned_overlap_frames"] == 12
    assert result["recommended_trim_analysis_frames"] == 5
    assert result["recommended_trim_frames"] == 20
    assert result["planned_overlap_frames"] == 48


def test_sam3_timeline_refinement_rejects_a_cut_without_safe_tail(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HONCUT_SAM3_ANALYSIS_FPS", "3")
    monkeypatch.setattr(
        object_trajectory_module,
        "build_tracking_clip",
        lambda *_args, **_kwargs: 6,
    )

    class StubClient:
        def __init__(self, _base_url):
            pass

        def track(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(object_trajectory_module, "Sam3TrajectoryClient", StubClient)
    monkeypatch.setattr(
        object_trajectory_module,
        "decide_object_trajectory",
        lambda *_args, **_kwargs: {
            "verdict": "rollback",
            "confidence": 0.9,
            "repair_action": "hard_trim",
            "recommended_trim_frames": 13,
        },
    )
    monkeypatch.setattr(
        object_trajectory_module,
        "refine_object_catchup_frame",
        lambda *_args, **_kwargs: {
            "status": "no_safe_catchup",
            "reason": "the exact catch-up leaves too little following material",
        },
    )

    evidence = object_trajectory_module.collect_sam3_trajectory(
        tmp_path / "previous.mp4",
        tmp_path / "following.mp4",
        boundary_id="S01_C01__S01_C02",
        evidence_dir=tmp_path,
        prompt="paper boat",
        timeline_fps=24,
        planned_overlap_frames=48,
        following_frames=121,
        screen_direction="left_to_right",
        camera_motion="locked_off",
        base_url="http://127.0.0.1:8001",
    )

    assert evidence["coarse_recommended_trim_frames"] == 104
    assert evidence["repair_action"] == "regenerate"
    assert evidence["timeline_refinement"]["status"] == "no_safe_catchup"


def test_object_catchup_frame_overrides_background_dominated_pixel_cut():
    values = [
        0.0466, 0.0462, 0.0463, 0.0460, 0.0450, 0.0439, 0.0434,
        0.0434, 0.0432, 0.0427, 0.0422, 0.0420, 0.0419,
    ]
    decision = decide_temporal_seam(
        [
            {
                "frames": 48 + index * 2,
                "seconds": (48 + index * 2) / 24,
                "frame_mae": value,
            }
            for index, value in enumerate(values)
        ],
        planned_overlap_frames=48,
        timeline_fps=24,
        object_trajectory_evidence={
            "verdict": "rollback",
            "confidence": 0.9,
            "repair_action": "hard_trim",
            "recommended_trim_frames": 96,
        },
    )

    assert decision["action"] == "hard_trim"
    assert decision["trim_frames"] == 96
    assert decision["trim_source"] == "object_trajectory_catchup"


def test_object_rollback_overrides_a_pixel_false_negative():
    decision = decide_temporal_seam(
        [
            {
                "frames": 48 + index * 2,
                "seconds": (48 + index * 2) / 24,
                "frame_mae": 0.04,
            }
            for index in range(13)
        ],
        planned_overlap_frames=48,
        timeline_fps=24,
        object_trajectory_evidence={
            "verdict": "rollback",
            "confidence": 0.97,
            "repair_action": "hard_trim",
            "recommended_trim_frames": 72,
        },
    )

    assert decision["action"] == "hard_trim"
    assert decision["trim_frames"] == 72
    assert decision["confidence"] == "object_trajectory_override"


def test_phase8_uses_configured_sam3_trajectory_before_requesting_topup(
    monkeypatch, tmp_path
):
    plan = build_continuity_plan(
        {
            "shots": [
                {
                    "id": "S01",
                    "duration": 8,
                    "tracking_prompt": "small blue paper boat",
                    "screen_direction": "left_to_right",
                    "camera_motion": "locked_off",
                }
            ]
        },
        provider_chunk_limit_s=5,
        continuation_overlap_s=2,
    )
    (tmp_path / "CONTINUITY_PLAN.json").write_text(
        plan.model_dump_json(indent=2),
        encoding="utf-8",
    )
    chunks = tmp_path / "shots" / "S01" / "chunks"
    chunks.mkdir(parents=True)
    (chunks / "S01_C01.mp4").write_bytes(b"first")
    (chunks / "S01_C02.mp4").write_bytes(b"second")
    (tmp_path / "shots" / "S01" / "CONTINUITY_TIMING.json").write_text(
        json.dumps(
            {
                "materialized_frames_before_closure": 194,
                "chunks": [
                    {"chunk_id": "S01_C01", "effective_unique_frames": 121},
                    {
                        "chunk_id": "S01_C02",
                        "effective_unique_frames": 73,
                        "detected_overlap_frames": 48,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    values = [
        0.0466, 0.0462, 0.0463, 0.0460, 0.0450, 0.0439, 0.0434,
        0.0434, 0.0432, 0.0427, 0.0422, 0.0420, 0.0419,
    ]
    monkeypatch.setenv("HONCUT_SAM3_URL", "http://127.0.0.1:8001")

    def collect(*_args, **kwargs):
        assert kwargs["prompt"] == "small blue paper boat"
        return {
            "verdict": "rollback",
            "confidence": 0.9,
            "repair_action": "hard_trim",
            "recommended_trim_frames": 72,
        }

    report = adjudicate_continuity_seams(
        tmp_path,
        detector=lambda *_args, **_kwargs: {
            "candidates": [
                {
                    "frames": 48 + index * 2,
                    "seconds": (48 + index * 2) / 24,
                    "frame_mae": value,
                }
                for index, value in enumerate(values)
            ]
        },
        frame_probe=lambda _path, _fps: {"frames": 121},
        sam3_collector=collect,
    )

    assert report["status"] == "topup_required"
    assert report["requires_human_review"] is False
    assert report["shots"][0]["deficit_frames"] == 22
    decisions = json.loads((tmp_path / "CONTINUITY_SEAM_DECISIONS.json").read_text())
    assert decisions["decisions"]["S01_C01__S01_C02"]["trim_frames"] == 72


def test_phase8_human_review_can_reject_appearance_only_extra_trim(
    monkeypatch, tmp_path
):
    plan = build_continuity_plan(
        {"shots": [{"id": "S01", "duration": 8}]},
        provider_chunk_limit_s=5,
        continuation_overlap_s=2,
    )
    (tmp_path / "CONTINUITY_PLAN.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    chunks = tmp_path / "shots" / "S01" / "chunks"
    chunks.mkdir(parents=True)
    previous = chunks / "S01_C01.mp4"
    following = chunks / "S01_C02.mp4"
    previous.write_bytes(b"first")
    following.write_bytes(b"second")
    (tmp_path / "shots" / "S01" / "CONTINUITY_TIMING.json").write_text(
        json.dumps(
            {
                "materialized_frames_before_closure": 241,
                "chunks": [
                    {"chunk_id": "S01_C01", "effective_unique_frames": 120},
                    {
                        "chunk_id": "S01_C02",
                        "effective_unique_frames": 121,
                        "detected_overlap_frames": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    source_fingerprint = {
        "previous_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
        "following_sha256": hashlib.sha256(following.read_bytes()).hexdigest(),
    }
    (tmp_path / "CONTINUITY_TEMPORAL_REVIEW.json").write_text(
        json.dumps(
            {
                "kind": "honcut.continuity_temporal_review.v1",
                "boundaries": {
                    "S01_C01__S01_C02": {
                        "action": "hard_trim",
                        "approved_trim_frames": 48,
                        "source_fingerprint": source_fingerprint,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    values = [
        0.112929,
        0.112,
        0.111,
        0.110,
        0.109,
        0.108,
        0.107,
        0.106,
        0.105,
        0.1048,
        0.1045,
        0.1042,
        0.10388,
    ]

    report = adjudicate_continuity_seams(
        tmp_path,
        detector=lambda *_args, **_kwargs: {
            "candidates": [
                {
                    "frames": 48 + index * 2,
                    "seconds": (48 + index * 2) / 24,
                    "frame_mae": value,
                }
                for index, value in enumerate(values)
            ]
        },
        frame_probe=lambda _path, _fps: {"frames": 121},
    )

    boundary = report["shots"][0]["boundaries"][0]
    assert report["status"] == "passed"
    assert report["requires_human_review"] is False
    assert boundary["action"] == "hard_trim"
    assert boundary["trim_frames"] == 48
    assert boundary["additional_trim_frames"] == 0
    assert boundary["confidence"] == "human_confirmed_planned_overlap"


def test_phase8_skips_sam3_when_pixel_trajectory_is_continuous(monkeypatch, tmp_path):
    plan = build_continuity_plan(
        {
            "shots": [
                {
                    "id": "S01",
                    "duration": 8,
                    "tracking_prompt": "small blue paper boat",
                }
            ]
        },
        provider_chunk_limit_s=5,
        continuation_overlap_s=2,
    )
    (tmp_path / "CONTINUITY_PLAN.json").write_text(
        plan.model_dump_json(indent=2),
        encoding="utf-8",
    )
    chunks = tmp_path / "shots" / "S01" / "chunks"
    chunks.mkdir(parents=True)
    (chunks / "S01_C01.mp4").write_bytes(b"first")
    (chunks / "S01_C02.mp4").write_bytes(b"second")
    (tmp_path / "shots" / "S01" / "CONTINUITY_TIMING.json").write_text(
        json.dumps(
            {
                "materialized_frames_before_closure": 192,
                "chunks": [
                    {"chunk_id": "S01_C01", "effective_unique_frames": 120},
                    {
                        "chunk_id": "S01_C02",
                        "effective_unique_frames": 72,
                        "detected_overlap_frames": 48,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HONCUT_SAM3_URL", "http://127.0.0.1:8001")

    def unexpected_collect(*_args, **_kwargs):
        raise AssertionError("SAM3 must not run for a continuous pixel trajectory")

    report = adjudicate_continuity_seams(
        tmp_path,
        detector=lambda *_args, **_kwargs: {
            "candidates": [
                {
                    "frames": 48 + index * 2,
                    "seconds": (48 + index * 2) / 24,
                    "frame_mae": 0.04,
                }
                for index in range(13)
            ]
        },
        frame_probe=lambda _path, _fps: {"frames": 121},
        sam3_collector=unexpected_collect,
    )

    assert report["status"] == "passed"
    assert not (tmp_path / "CONTINUITY_OBJECT_TRAJECTORIES.json").exists()


def test_phase6_applies_an_exact_phase8_hard_trim_with_bridge_disabled(
    monkeypatch, tmp_path
):
    previous = tmp_path / "previous.mp4"
    following = tmp_path / "following.mp4"
    previous.write_bytes(b"previous")
    following.write_bytes(b"following")
    boundary_id = "S01_C01__S01_C02"
    source_fingerprint = {
        "previous_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
        "following_sha256": hashlib.sha256(following.read_bytes()).hexdigest(),
    }
    (tmp_path / "CONTINUITY_SEAM_DECISIONS.json").write_text(
        json.dumps(
            {
                "kind": SEAM_DECISIONS_KIND,
                "decisions": {
                    boundary_id: {
                        "action": "hard_trim",
                        "trim_frames": 72,
                        "trim_seconds": 3,
                        "reason": "temporal rollback",
                        "source_fingerprint": source_fingerprint,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HONCUT_CONTINUITY_BRIDGE", "off")

    def render(_following, output, **kwargs):
        assert kwargs == {"trim_frames": 72, "timeline_fps": 24}
        Path(output).write_bytes(b"hard-trimmed")

    monkeypatch.setattr("runtime.continuity_provider._render_phase8_hard_trim", render)
    prepare = _continuity_bridge_preparer(tmp_path, timeline_fps=24)
    assert prepare is not None

    receipt = prepare(previous, following, boundary_id)

    assert receipt["status"] == "adjudicated_trim"
    assert receipt["selected_bridge_frames"] is None
    assert receipt["overlap"]["overlap_frames"] == 72


def test_continuity_bridge_rejects_invalid_candidate_frames_before_execution(monkeypatch, tmp_path):
    monkeypatch.setenv("HONCUT_CONTINUITY_BRIDGE", "auto")
    monkeypatch.setenv("HONCUT_CONTINUITY_BRIDGE_FRAMES", "2,8")

    with pytest.raises(ValueError, match="between 4 and 24"):
        _continuity_bridge_preparer(tmp_path)


def test_continuity_bridge_uses_a_strongly_improving_planned_overlap_trial(
    monkeypatch, tmp_path
):
    previous = tmp_path / "previous.mp4"
    following = tmp_path / "following.mp4"
    previous.write_bytes(b"previous")
    following.write_bytes(b"following")
    monkeypatch.setenv("HONCUT_CONTINUITY_BRIDGE", "auto")
    monkeypatch.setattr(
        "quality.continuity_bridge.detect_replayed_prefix",
        lambda *_args, **_kwargs: {
            "detected": False,
            "overlap_seconds": 0.0,
            "reason": "trajectory detector uncertain",
        },
    )

    def repair(_previous, _following, output, **kwargs):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"effective")
        trimmed = Path(kwargs["work_dir"]) / "trimmed_hard_cut.mp4"
        trimmed.parent.mkdir(parents=True, exist_ok=True)
        trimmed.write_bytes(b"ghost-safe-trim")
        assert kwargs["overlap_seconds"] == 2.0
        return {
            "kind": "honcut.continuity_bridge.v1",
            "status": "repaired",
            "output_path": str(output),
            "baseline_boundary_frame_mae": 0.06,
            "trimmed_boundary_frame_mae": 0.04,
            "selected_boundary_frame_mae": 0.01,
            "improved": True,
        }

    monkeypatch.setattr("quality.continuity_bridge.repair_continuity_boundary", repair)
    prepare = _continuity_bridge_preparer(tmp_path, planned_overlap_seconds=2.0)

    receipt = prepare(previous, following, "S01_C01__S01_C02")

    assert receipt["status"] == "trimmed"
    assert receipt["ghost_safe_fallback"] is True
    assert Path(receipt["output_path"]).read_bytes() == b"ghost-safe-trim"
    assert receipt["overlap"]["source"] == "phase4_planned_budget"
    assert receipt["overlap"]["overlap_seconds"] == 2.0


def test_continuity_bridge_auto_falls_back_to_provider_clip_on_local_failure(monkeypatch, tmp_path):
    previous = tmp_path / "previous.mp4"
    following = tmp_path / "following.mp4"
    previous.write_bytes(b"not-a-video")
    following.write_bytes(b"provider-video-stays-authoritative")
    monkeypatch.setenv("HONCUT_CONTINUITY_BRIDGE", "auto")

    prepare = _continuity_bridge_preparer(tmp_path)
    assert prepare is not None
    receipt = prepare(previous, following, "S01_C01__S01_C02")

    assert receipt["status"] == "fallback"
    assert receipt["output_path"] == str(following)
    assert following.read_bytes() == b"provider-video-stays-authoritative"
    assert "local continuity bridge failed" in receipt["reason"]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_detects_replayed_prefix_and_renders_a_smoother_derived_clip(tmp_path):
    master = tmp_path / "master.mkv"
    previous = tmp_path / "previous.mp4"
    following = tmp_path / "following.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=24:duration=3",
            "-c:v",
            "ffv1",
            str(master),
        ],
        check=True,
        timeout=30,
    )
    for start, output in ((0, previous), (1, following)):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                str(start),
                "-i",
                str(master),
                "-t",
                "2",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "12",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
            timeout=30,
        )

    overlap = detect_replayed_prefix(previous, following, search_seconds=1.5)
    output = tmp_path / "effective.mp4"
    receipt = repair_continuity_boundary(
        previous,
        following,
        output,
        work_dir=tmp_path / "bridge-work",
        overlap_seconds=overlap["overlap_seconds"],
        candidate_frames=(4, 6),
    )

    assert overlap["detected"] is True
    assert overlap["overlap_seconds"] == 1.0
    assert output.is_file() and output.stat().st_size > 0
    assert receipt["status"] == "repaired"
    assert receipt["selected_bridge_frames"] in {4, 6}
    assert receipt["selected_boundary_frame_mae"] < receipt["trimmed_boundary_frame_mae"]
    assert receipt["trimmed_boundary_frame_mae"] < receipt["baseline_boundary_frame_mae"]
    assert receipt["removed_replay_duration_seconds"] == pytest.approx(1.0, abs=0.05)
    assert following.stat().st_size > 0


def test_canonical_memory_is_immutable_and_generated_frames_stay_separate(tmp_path):
    canonical_dir = tmp_path / "characters/CHAR_01"
    canonical_dir.mkdir(parents=True)
    Image.new("RGB", (24, 24), "navy").save(canonical_dir / "face_closeup.png")
    plan = build_continuity_plan({"shots": [{"id": "S01", "duration": 8, "who": ["CHAR_01"]}]})

    initial = initialize_continuity_memory(tmp_path, plan)
    changed = plan.model_copy(deep=True)
    changed.shots[0].anchors.style = "rewritten by generated output"

    with pytest.raises(RuntimeError, match="canonical continuity anchors changed"):
        initialize_continuity_memory(tmp_path, changed)

    frame_a = tmp_path / "frame_a.png"
    frame_duplicate = tmp_path / "frame_duplicate.png"
    frame_b = tmp_path / "frame_b.png"
    pattern_a = np.zeros((32, 32, 3), dtype=np.uint8)
    pattern_a[:, :16] = 255
    checkerboard = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)
    pattern_b = np.repeat(checkerboard[:, :, None], 3, axis=2)
    Image.fromarray(pattern_a).save(frame_a)
    Image.fromarray(pattern_a).save(frame_duplicate)
    Image.fromarray(pattern_b).save(frame_b)

    selected = select_memory_keyframes([frame_a, frame_duplicate, frame_b])
    entry = record_recent_motion(
        tmp_path,
        plan,
        shot_id="S01",
        chunk_id="S01_C01",
        candidate_frames=[frame_a, frame_duplicate, frame_b],
    )
    persisted = json.loads((tmp_path / "CONTINUITY_MEMORY.json").read_text())
    context = render_continuity_memory_context(tmp_path, plan, "S01")

    assert len(selected) == 2
    assert len(entry["keyframes"]) == 2
    assert all(frame["role"] == "generated" for frame in entry["keyframes"])
    assert persisted["canonical"] == initial["canonical"]
    assert all(asset["role"] == "canonical" for asset in persisted["canonical"]["assets"])
    assert "canonical anchors are authoritative and immutable" in context
    assert "generated recent motion is advisory" in context


def test_recent_motion_memory_is_bounded(tmp_path):
    plan = build_continuity_plan({"shots": [{"id": "S01", "duration": 8}]})
    frame = tmp_path / "frame.png"
    Image.new("RGB", (16, 16), "gray").save(frame)

    for index in range(3):
        record_recent_motion(
            tmp_path,
            plan,
            shot_id="S01",
            chunk_id=f"S01_C0{index + 1}",
            candidate_frames=[frame],
            recent_limit=2,
        )

    persisted = json.loads((tmp_path / "CONTINUITY_MEMORY.json").read_text())
    assert [item["chunk_id"] for item in persisted["recent_motion"]] == [
        "S01_C02",
        "S01_C03",
    ]


def test_phase8_forces_continuous_boundaries_to_hard_cuts(monkeypatch, tmp_path):
    shots_dir = tmp_path / "shots"
    for shot_id in ("S01", "S02"):
        shot_dir = shots_dir / shot_id
        shot_dir.mkdir(parents=True)
        (shot_dir / "output.mp4").write_bytes(b"video")

    monkeypatch.setattr(
        edit_decision_module,
        "probe_video",
        lambda path: {"duration": 5.0, "has_audio": True},
    )
    monkeypatch.setattr(
        edit_decision_module,
        "detect_black_frames",
        lambda path: {"trim_start": 0.0, "trim_end": 0.0},
    )
    continuity_plan = build_continuity_plan(
        {
            "shots": [
                {"id": "S01", "duration": 5},
                {
                    "id": "S02",
                    "duration": 5,
                    "boundary_before": "continuous",
                    "continuity_subject": "paper boat",
                },
            ]
        }
    )

    decisions = edit_decision_module.build_edit_decisions(
        str(shots_dir),
        transition_decisions=[{"decision": "dissolve"}],
        continuity_plan=continuity_plan.model_dump(mode="json"),
    )

    assert decisions["transitions"] == [
        {
            "index": 0,
            "type": "cut",
            "duration": 0.0,
            "duration_frames": 0,
            "locked": True,
            "lock_reason": "continuous editorial boundary",
            "audio_transition": "edge_fade",
            "audio_duration": 0.35,
        }
    ]
    assert decisions["metadata"]["transition_locks"][0]["before_shot_id"] == "S02"


def test_phase8_inserts_post_primary_bridge_and_skips_transition_effects(
    monkeypatch, tmp_path
):
    shots_dir = tmp_path / "shots"
    for shot_id in ("S01", "S02"):
        shot_dir = shots_dir / shot_id
        shot_dir.mkdir(parents=True)
        (shot_dir / "output.mp4").write_bytes(b"primary-video")
    bridge_path = tmp_path / "shot_bridges/S01__S02.mp4"
    bridge_path.parent.mkdir(parents=True)
    bridge_path.write_bytes(b"bridge-video")
    (tmp_path / "PRIMARY_SHOT_BRIDGES.json").write_text(
        json.dumps(
            {
                "kind": "honcut.primary_shot_bridges.v2",
                "status": "done",
                "count": 1,
                "bridges": [
                    {
                        "boundary_id": "S01__S02",
                        "source_shot_id": "S01",
                        "target_shot_id": "S02",
                        "path": "shot_bridges/S01__S02.mp4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    storyboard = {
        "continuity_mode": "one_take",
        "video_provider": "seedance",
        "shots": [
            {"id": "S01", "duration": 15, "micro_actions": ["前进"]},
            {"id": "S02", "duration": 15, "micro_actions": ["继续前进"]},
        ],
    }
    plan_storyboard_beats(storyboard)
    plan = build_continuity_plan(storyboard)
    monkeypatch.setattr(
        edit_decision_module,
        "probe_video",
        lambda path: {
            "duration": 4.0 if "shot_bridges" in str(path) else 15.0,
            "has_audio": False,
        },
    )
    monkeypatch.setattr(
        edit_decision_module,
        "detect_black_frames",
        lambda *_args, **_kwargs: {"trim_start": 0.0, "trim_end": 0.0},
    )

    decisions = edit_decision_module.build_edit_decisions(
        str(shots_dir),
        transition_decisions=[{"decision": "dissolve"}],
        continuity_plan=plan.model_dump(mode="json"),
        target_duration=30,
    )

    assert [cut["shot_id"] for cut in decisions["cuts"]] == [
        "S01", "BRIDGE_S01__S02", "S02",
    ]
    assert [item["type"] for item in decisions["transitions"]] == ["cut", "cut"]
    assert len(decisions["metadata"]["inserted_primary_bridges"]) == 1
    assert decisions["cuts"][0]["out_seconds"] == 13.0
    assert decisions["cuts"][2]["in_seconds"] == 2.0
    replacement = decisions["metadata"]["bridge_handle_replacements"][0]
    assert replacement["source_handle_s"] == 2.0
    assert replacement["target_handle_s"] == 2.0
    assert decisions["metadata"]["projected_frames"] == 30 * 30
    assert decisions["metadata"]["pacing_normalization"] is None


def test_phase8_bridge_handles_and_bounded_pacing_hit_delivery_without_tail_loss(
    monkeypatch, tmp_path
):
    shots_dir = tmp_path / "shots"
    for shot_id in ("S01", "S02"):
        shot_dir = shots_dir / shot_id
        shot_dir.mkdir(parents=True)
        (shot_dir / "output.mp4").write_bytes(b"primary-video")
    bridge_path = tmp_path / "shot_bridges/S01__S02.mp4"
    bridge_path.parent.mkdir(parents=True)
    bridge_path.write_bytes(b"bridge-video")
    (tmp_path / "PRIMARY_SHOT_BRIDGES.json").write_text(
        json.dumps(
            {
                "kind": "honcut.primary_shot_bridges.v2",
                "status": "done",
                "count": 1,
                "bridges": [
                    {
                        "source_shot_id": "S01",
                        "target_shot_id": "S02",
                        "path": "shot_bridges/S01__S02.mp4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    storyboard = {
        "continuity_mode": "one_take",
        "video_provider": "seedance",
        "delivery_target_duration": 50,
        "pre_edit_duration_ratio_limit": 1.3,
        "shots": [
            {"id": "S01", "duration": 30, "micro_actions": ["前进"]},
            {"id": "S02", "duration": 30, "micro_actions": ["继续前进"]},
        ],
    }
    plan_storyboard_beats(storyboard)
    plan = build_continuity_plan(storyboard)
    monkeypatch.setattr(
        edit_decision_module,
        "probe_video",
        lambda path: {
            "duration": 4.0 if "shot_bridges" in str(path) else 30.0,
            "has_audio": False,
        },
    )
    monkeypatch.setattr(
        edit_decision_module,
        "detect_black_frames",
        lambda *_args, **_kwargs: {"trim_start": 0.0, "trim_end": 0.0},
    )

    decisions = edit_decision_module.build_edit_decisions(
        str(shots_dir),
        transition_decisions=[{"decision": "dissolve"}],
        continuity_plan=plan.model_dump(mode="json"),
        target_duration=50,
    )

    assert all(
        cut["speed"] == pytest.approx(1.2, abs=0.002)
        for cut in decisions["cuts"]
    )
    normalization = decisions["metadata"]["pacing_normalization"]
    assert normalization["method"] == "bounded_all_frame_pacing_normalization"
    assert normalization["speed"] == 1.2
    assert normalization["preserves_all_reviewed_frames"] is True
    # With 4s bridges and 2s handles the 30+30s material closes the 50s frame
    # budget exactly at uniform speed — no per-cut residual correction needed.
    assert normalization.get("frame_closure") is None
    assert decisions["metadata"]["projected_frames"] == 50 * 30


def test_phase8_trims_cross_shot_extension_prefix_and_keeps_scene_cuts_for_transitions(
    monkeypatch, tmp_path
):
    plan = build_continuity_plan(
        {
            "shots": [
                {"id": "S01", "duration": 5},
                {
                    "id": "S02",
                    "duration": 5,
                    "boundary_before": "continuous",
                    "continuity_subject": "paper boat",
                },
                {"id": "S03", "duration": 5},
            ]
        },
        continuation_overlap_s=2,
    )
    (tmp_path / "CONTINUITY_PLAN.json").write_text(
        plan.model_dump_json(indent=2), encoding="utf-8"
    )
    for shot in plan.shots:
        shot_dir = tmp_path / "shots" / shot.shot_id
        chunks_dir = shot_dir / "chunks"
        chunks_dir.mkdir(parents=True)
        for chunk in shot.chunks:
            (chunks_dir / f"{chunk.chunk_id}.mp4").write_bytes(chunk.chunk_id.encode())
        (shot_dir / "output.mp4").write_bytes(b"video")
    (tmp_path / "shots/S02/CONTINUITY_TIMING.json").write_text(
        json.dumps(
            {
                "materialized_frames_before_closure": 168,
                "chunks": [
                    {
                        "chunk_id": "S02_C01",
                        "effective_unique_frames": 168,
                        "detected_overlap_frames": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    sam3_calls = []

    def collect(*_args, **kwargs):
        sam3_calls.append(kwargs["boundary_id"])
        return {"verdict": "continuous", "confidence": 0.97}

    report = adjudicate_continuity_seams(
        tmp_path,
        detector=lambda *_args, **_kwargs: {
            "candidates": [
                {
                    "frames": 48 + index * 2,
                    "seconds": (48 + index * 2) / 24,
                    "frame_mae": 0.04,
                }
                for index in range(13)
            ]
        },
        frame_probe=lambda path, _fps: {
            "frames": 168 if path.stem == "S02_C01" else 120
        },
        sam3_collector=collect,
        sam3_base_url="http://127.0.0.1:8001",
    )

    assert report["status"] == "passed"
    assert sam3_calls == ["S01_C01__S02_C01"]
    persisted = json.loads((tmp_path / "CONTINUITY_SEAM_DECISIONS.json").read_text())
    cross = persisted["decisions"]["S01_C01__S02_C01"]
    assert cross["boundary_kind"] == "cross_shot"
    assert cross["trim_frames"] == 48

    monkeypatch.setattr(
        edit_decision_module,
        "probe_video",
        lambda path: {
            "duration": 7.0 if "S02" in str(path) else 5.0,
            "has_audio": False,
        },
    )
    monkeypatch.setattr(
        edit_decision_module,
        "detect_black_frames",
        lambda *_args, **_kwargs: {"trim_start": 0.0, "trim_end": 0.0},
    )
    decisions = edit_decision_module.build_edit_decisions(
        str(tmp_path / "shots"),
        transition_decisions=[{"decision": "dissolve"}, {"decision": "dissolve"}],
        continuity_plan=plan.model_dump(mode="json"),
    )

    assert decisions["cuts"][1]["in_seconds"] == 2.0
    assert decisions["cuts"][1]["out_seconds"] == 7.0
    assert decisions["transitions"][0]["type"] == "cut"
    assert decisions["transitions"][1]["type"] == "dissolve"
    assert decisions["metadata"]["continuity_trims"][0]["trim_frames"] == 48


def test_phase8_preserves_finalized_continuity_frame_budget(monkeypatch, tmp_path):
    shot_dir = tmp_path / "shots/S01"
    shot_dir.mkdir(parents=True)
    (shot_dir / "output.mp4").write_bytes(b"video")
    (shot_dir / "CONTINUITY_TIMING.json").write_text(
        json.dumps(
            {
                "kind": "honcut.continuity_timing.v1",
                "target_frames": 120,
                "final_frames": 120,
                "internal_seams_finalized": True,
            }
        ),
        encoding="utf-8",
    )
    observed = {}
    monkeypatch.setattr(
        edit_decision_module,
        "probe_video",
        lambda path: {"duration": 5.0, "has_audio": False},
    )

    def detect(path, *, trim_static_edges=True):
        observed.update(path=path, trim_static_edges=trim_static_edges)
        return {"trim_start": 0.0, "trim_end": 0.0}

    monkeypatch.setattr(edit_decision_module, "detect_black_frames", detect)

    decisions = edit_decision_module.build_edit_decisions(str(tmp_path / "shots"))

    assert observed["trim_static_edges"] is False
    assert decisions["cuts"][0]["in_seconds"] == 0.0
    assert decisions["cuts"][0]["out_seconds"] == 5.0
    assert decisions["cuts"][0]["continuity_timing"]["final_frames"] == 120


def test_phase6_shadow_keeps_the_existing_provider_route(monkeypatch, tmp_path):
    write_continuity_plan(
        tmp_path / "CONTINUITY_PLAN.json",
        {"shots": [{"id": "S01", "duration": 16}]},
    )
    observed = {"requests": 0}

    class FakeAdapter:
        def __init__(self, models):
            self.models = models

        def request(self, model, config):
            observed["requests"] += 1
            return SimpleNamespace(
                data={"status": "error", "provider": "offline-test", "error": "stopped"},
                error=None,
            )

    monkeypatch.setenv("HONCUT_CONTINUITY_MODE", "shadow")
    monkeypatch.setattr(pipeline_core, "_LocalVideoVendorAdapter", FakeAdapter)

    receipt = pipeline_core.run_phase6({"shots": []}, tmp_path, dry_run=False)

    assert observed["requests"] == 1
    assert receipt["continuity_runtime"]["mode"] == "shadow"
    assert receipt["continuity_runtime"]["execution_enabled"] is False


def test_phase6_auto_fails_before_the_provider_route(monkeypatch, tmp_path):
    class UnexpectedAdapter:
        def __init__(self, models):
            raise AssertionError("provider adapter must not initialize in guarded auto mode")

    monkeypatch.setenv("HONCUT_CONTINUITY_MODE", "auto")
    monkeypatch.setattr(pipeline_core, "_LocalVideoVendorAdapter", UnexpectedAdapter)

    receipt = pipeline_core.run_phase6({"shots": []}, tmp_path, dry_run=False)

    assert receipt["status"] == "error"
    assert "seam guard" in receipt["error"]


def test_phase6_auto_routes_only_through_continuity_runtime(monkeypatch, tmp_path):
    write_continuity_plan(
        tmp_path / "CONTINUITY_PLAN.json",
        {"shots": [{"id": "S01", "duration": 16}]},
    )
    calibration = _certified_seam_calibration()
    (tmp_path / "CONTINUITY_CALIBRATION.json").write_text(
        calibration.model_dump_json(indent=2),
        encoding="utf-8",
    )
    observed = {}

    class UnexpectedAdapter:
        def __init__(self, models):
            raise AssertionError("legacy provider route must not initialize in auto mode")

    def fake_auto(output_dir, plan, loaded_calibration):
        observed.update(
            output_dir=output_dir,
            chunk_count=sum(len(shot.chunks) for shot in plan.shots),
            calibration=loaded_calibration.dataset_fingerprint,
        )
        return {
            "status": "done",
            "outputs": ["shots/S01/output.mp4"],
            "errors": [],
            "provider": "offline-fake",
        }

    monkeypatch.setenv("HONCUT_CONTINUITY_MODE", "auto")
    monkeypatch.setattr(pipeline_core, "_LocalVideoVendorAdapter", UnexpectedAdapter)
    monkeypatch.setattr(
        "runtime.continuity_provider.execute_phase6_auto_continuity",
        fake_auto,
    )
    monkeypatch.setattr(
        pipeline_core,
        "run_quality_check",
        lambda phase, output_dir: SimpleNamespace(passed=True),
    )

    receipt = pipeline_core.run_phase6({"shots": []}, tmp_path, dry_run=False)

    assert receipt["status"] == "done"
    assert receipt["continuity_runtime"]["execution_enabled"] is True
    assert observed == {
        "output_dir": tmp_path,
        "chunk_count": 2,
        "calibration": calibration.dataset_fingerprint,
    }


def test_phase6_auto_normalizes_resumed_string_output_dir(monkeypatch, tmp_path):
    write_continuity_plan(
        tmp_path / "CONTINUITY_PLAN.json",
        {"shots": [{"id": "S01", "duration": 16}]},
    )
    observed = {}

    def fake_auto(output_dir, plan, loaded_calibration):
        observed["output_dir"] = output_dir
        observed["chunk_count"] = sum(len(shot.chunks) for shot in plan.shots)
        observed["calibration"] = loaded_calibration
        return {
            "status": "done",
            "outputs": ["shots/S01/output.mp4"],
            "errors": [],
            "provider": "offline-fake",
        }

    monkeypatch.setenv("HONCUT_CONTINUITY_MODE", "auto")
    monkeypatch.setattr(
        "runtime.continuity_provider.execute_phase6_auto_continuity",
        fake_auto,
    )
    monkeypatch.setattr(
        pipeline_core,
        "run_quality_check",
        lambda phase, output_dir: SimpleNamespace(passed=True),
    )

    receipt = pipeline_core.run_phase6({"shots": []}, str(tmp_path), dry_run=False)

    assert receipt["status"] == "done"
    assert observed == {
        "output_dir": tmp_path,
        "chunk_count": 2,
        "calibration": None,
    }


def test_rhythm_speed_map_keeps_weighted_runtime():
    boundaries = [0.0, 2.0, 5.0, 10.0]
    normalized = _duration_preserving_speed_map(
        boundaries,
        {0: 1.2, 1: 0.8, 2: 1.1},
    )

    output_duration = sum(
        (boundaries[i + 1] - boundaries[i]) / normalized[i]
        for i in range(len(boundaries) - 1)
    )

    assert output_duration == pytest.approx(10.0)
    assert normalized[0] / normalized[1] == pytest.approx(1.2 / 0.8)


def test_rhythm_speed_ramp_closes_real_av_duration(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "ramped.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(source),
        ],
        check=True,
        capture_output=True,
    )

    apply_speed_ramp(
        str(source),
        {0: 1.2, 1: 0.8, 2: 1.1},
        [1.0, 2.0, 3.0],
        str(output),
    )

    assert _probe_duration(str(output)) == pytest.approx(
        _probe_duration(str(source)), abs=2 / 24
    )


def test_phase9_duration_gate_accepts_frame_decimal_rounding():
    pipeline_core._assert_duration_conserved(
        {"video": 60.066667, "audio": 60.0},
        {"video": 60.0, "audio": 60.0},
        tolerance_s=2 / 30,
        audio_tolerance_s=0.05,
    )


def test_delivery_timeline_tracks_warped_shot_boundaries(tmp_path):
    output = tmp_path / "polished.mp4"
    timeline = {
        "shots": [
            {"shot_id": "S01", "speed": 1.0},
            {"shot_id": "S02", "speed": 1.0},
        ],
        "transitions": [{"type": "cut"}],
    }
    boundaries = [0.0, 4.0, 10.0]
    speeds = _duration_preserving_speed_map(boundaries, {0: 1.2, 1: 0.8})

    written = _write_delivery_timeline(str(output), timeline, boundaries, speeds)
    delivery = json.loads(Path(written).read_text())

    assert delivery["duration_s"] == pytest.approx(10.0)
    assert delivery["shots"][0]["output_end_s"] == pytest.approx(4.0 / speeds[0])
    assert delivery["shots"][1]["output_start_s"] == pytest.approx(
        delivery["shots"][0]["output_end_s"]
    )
    assert delivery["shots"][1]["output_end_s"] == pytest.approx(10.0)


def test_final_qa_samples_delivery_timeline_before_edit_timeline(tmp_path, monkeypatch):
    video = tmp_path / "polished.mp4"
    video.touch()
    (tmp_path / "delivery_timeline.json").write_text(json.dumps({
        "shots": [
            {"shot_id": "S01", "output_start_s": 0.0, "output_end_s": 4.0},
            {"shot_id": "S02", "output_start_s": 4.0, "output_end_s": 10.0},
        ]
    }))
    (tmp_path / "edit_timeline.json").write_text(json.dumps({
        "shots": [
            {"shot_id": "S01", "output_start_s": 0.0, "output_end_s": 5.0},
            {"shot_id": "S02", "output_start_s": 5.0, "output_end_s": 10.0},
        ]
    }))
    monkeypatch.setattr(video_qa, "_get_duration", lambda _path: 10.0)
    monkeypatch.setattr(
        video_qa,
        "_extract_frame",
        lambda _video, _directory, timestamp, label: video_qa.FrameSample(
            path=f"{label}.jpg", timestamp=timestamp, label=label
        ),
    )

    frames = video_qa._sample_frames(
        video,
        tmp_path,
        [0.0],
        {"shots": []},
        video_qa.VideoQAReport(verdict="pass", grade="A"),
    )
    s02_first = next(frame for frame in frames if frame.label == "S02_first")

    assert s02_first.timestamp == pytest.approx(4.1)


def test_animated_still_motion_metric_triggers_reshoot(tmp_path):
    frame_paths = []
    base = np.full((120, 160, 3), 80, dtype=np.uint8)
    for index in range(5):
        image = base.copy()
        image[index:index + 2, :, :] += 2  # tiny rain-like change only
        path = tmp_path / f"frame_{index}.png"
        Image.fromarray(image).save(path)
        frame_paths.append(path)

    activity = measure_motion_activity(frame_paths)
    decision = decide_shot_action(
        10.0,
        [],
        [],
        [],
        None,
        {"camera_movement": "static"},
        activity,
    )

    assert activity["median_mae"] < 3.5
    assert decision["action"] == "reshoot"
    assert "animated-still motion failure" in decision["reasons"][0]

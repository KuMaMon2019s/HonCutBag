from __future__ import annotations

import base64
import hashlib
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
from clients.seedream_client import IMAGE_ENDPOINT
from phases import pipeline_core
from phases.phase4.continuity_plan import build_continuity_plan, write_continuity_plan
from phases.phase8 import edit_decisions as edit_decision_module
from phases.phase8.continuity_adjudication import (
    SEAM_DECISIONS_KIND,
    adjudicate_continuity_seams,
    decide_temporal_seam,
)
from quality import object_trajectory as object_trajectory_module
from quality import sam3_sidecar as sam3_sidecar_module
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
    _bridge_seedance_executor,
    _continuity_bridge_preparer,
    _direct_seedance_executor,
    _generation_seed,
    _provider_content,
    execute_phase6_auto_continuity,
    materialize_continuity_shot,
)
from runtime.generation_tasks import GenerationTaskStore
from sam3_runtime.policy import (
    estimate_weight_bytes,
    resolve_checkpoint_path,
    resolve_runtime_policy,
)
from schemas.continuity import ContinuityPlan, GenerationChunk


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
        return ChunkExecutionResult(request.output_path, f"task-{request.resource_id}")

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
    assert dict(sequences) == {"S01": [1, 2], "S02": [1, 2]}
    assert (tmp_path / "shots/S01/output.mp4").read_bytes() == b"S01_C01|S01_C02"


def test_chunk_runtime_relays_previous_shot_video_inside_a_continuity_group(tmp_path):
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
        "reference_video",
    ]
    assert content[0]["text"].startswith("向后延长视频1")
    assert "图片1、图片2、图片3" in content[0]["text"]
    assert "严格参考图片3" in content[0]["text"]
    assert "不得重播视频1中的运动轨迹" in content[0]["text"]
    assert "without a reset or cut" in content[0]["text"]
    assert "Do not skip forward in time" in content[0]["text"]
    assert [item["image_url"]["url"] for item in content[1:4]] == [
        f"https://image.test/{path.name}"
        for path in sorted((tmp_path / "continuity_anchors").glob("*_frame_*.jpg"))
    ]
    assert content[-2]["image_url"]["url"] == "https://image.test/frame.png"
    assert content[-1]["video_url"]["url"] == "https://video.test/tail-window.mp4"


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


def test_bridge_continuity_adapter_reuses_succeeded_paid_task(monkeypatch, tmp_path):
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

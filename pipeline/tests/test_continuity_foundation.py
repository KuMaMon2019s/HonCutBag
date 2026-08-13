from __future__ import annotations

import base64
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
from phases import pipeline_core
from phases.phase4.continuity_plan import build_continuity_plan, write_continuity_plan
from phases.phase8 import edit_decisions as edit_decision_module
from quality.continuity_seam import compare_frame_sequences, measure_video_seam
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
    _direct_seedance_executor,
    _provider_content,
    execute_phase6_auto_continuity,
    materialize_continuity_shot,
)
from runtime.generation_tasks import GenerationTaskStore
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
    }
    assert [chunk.target_duration_s for chunk in shot.chunks] == [13, 13, 12]
    assert [chunk.depends_on for chunk in shot.chunks] == [None, "S03_C01", "S03_C02"]
    persisted = json.loads((tmp_path / "CONTINUITY_PLAN.json").read_text())
    assert persisted["version"] == 1
    assert persisted["shots"][1]["chunks"][2]["mode"] == "native_extend"


def test_planner_balances_a_sixteen_second_shot_without_a_one_second_tail():
    plan = build_continuity_plan({"shots": [{"id": 1, "duration": 16}]})

    assert [chunk.target_duration_s for chunk in plan.shots[0].chunks] == [8, 8]


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
    assert all((tmp_path / "stable_evidence" / path.split("/")[-1]).is_file() for path in stable["tail_frames"])


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
    assert decide_seam(
        {"provisional_risk_score": 0.9}, calibration
    )["action"] == "observe_only"


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


def test_shadow_runtime_records_intent_without_provider_execution(monkeypatch, tmp_path):
    write_continuity_plan(
        tmp_path / "CONTINUITY_PLAN.json",
        {"shots": [{"id": "S01", "duration": 16}]},
    )
    monkeypatch.delenv("HONCUT_CONTINUITY_MODE", raising=False)

    report = write_shadow_runtime_report(tmp_path)

    assert report["mode"] == "shadow"
    assert report["execution_enabled"] is False
    assert report["chunk_count"] == 2
    assert json.loads((tmp_path / "CONTINUITY_RUNTIME.json").read_text())["mode"] == "shadow"


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


def test_extension_provider_content_keeps_images_as_anchors_and_adds_video(
    monkeypatch, tmp_path
):
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
        lambda path, prefix: "https://video.test/previous.mp4",
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
    )

    content, _meta, _seed, duration = _provider_content(tmp_path, request)

    assert duration == 8
    assert [item.get("role") for item in content] == [
        None,
        "reference_image",
        "reference_video",
    ]
    assert "without a reset or cut" in content[0]["text"]
    assert content[-1]["video_url"]["url"] == "https://video.test/previous.mp4"


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
def test_phase6_auto_runtime_repairs_real_decoded_seam_before_materializing(
    monkeypatch, tmp_path
):
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
                    f"color=c={color}:s=96x54:d=0.5",
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

    report = execute_phase6_auto_continuity(tmp_path, plan, calibration)

    lineage = json.loads((tmp_path / "CONTINUITY_LINEAGE.json").read_text())
    seam = lineage["seams"]["S01_C01__S01_C02"]
    assert report["status"] == "done"
    assert report["repair_attempts"] == 1
    assert calls == ["S01_C01", "S01_C02", "S01_C02_R01"]
    assert seam["decision"]["action"] == "accept"
    assert len(seam["attempt_history"]) == 2
    assert (tmp_path / "shots/S01/output.mp4").stat().st_size > 0


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
                {"id": "S02", "duration": 5, "boundary_before": "continuous"},
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
            "locked": True,
            "lock_reason": "continuous editorial boundary",
            "audio_transition": "edge_fade",
            "audio_duration": 0.35,
        }
    ]
    assert decisions["metadata"]["transition_locks"][0]["before_shot_id"] == "S02"


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

    monkeypatch.delenv("HONCUT_CONTINUITY_MODE", raising=False)
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

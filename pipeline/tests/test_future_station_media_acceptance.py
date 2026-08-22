from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "pipeline" / "src"
SCRIPTS = ROOT / "pipeline" / "scripts"
for import_root in (SRC, SCRIPTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import future_station_media_acceptance as acceptance
from runtime.continuity_chunks import ChunkExecutionRequest
from runtime.generation_tasks import GenerationTaskStore
from schemas.continuity import GenerationChunk


def _request(tmp_path: Path, fingerprint: str) -> ChunkExecutionRequest:
    return ChunkExecutionRequest(
        resource_id="S01_C01",
        shot_id="S01",
        chunk=GenerationChunk(
            chunk_id="S01_C01",
            sequence=1,
            target_duration_s=1.0,
            requested_frames=30,
            expected_unique_frames=30,
            mode="fresh",
        ),
        anchors={},
        output_path=tmp_path / "shots/S01/chunks/S01_C01.mp4",
        previous_output_path=None,
        input_fingerprint=fingerprint,
        memory_context="",
    )


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_offline_executor_persists_and_reuses_exact_video(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    stats = acceptance.OfflineExecutionStats()
    execute = acceptance._offline_executor_factory(tmp_path, store, stats=stats)
    fingerprint = hashlib.sha256(b"fixture-a").hexdigest()
    request = _request(tmp_path, fingerprint)

    first = execute(request)
    first_hash = acceptance._sha256(request.output_path)
    second = execute(request)
    rows = acceptance._task_rows(tmp_path / "runtime.db")

    assert first.provider_task_id == second.provider_task_id
    assert acceptance._sha256(request.output_path) == first_hash
    assert stats.generated_chunks == 1
    assert stats.reused_chunks == 1
    assert len(rows) == 1
    assert rows[0]["provider_id"] == acceptance.OFFLINE_PROVIDER_ID
    assert rows[0]["provider_endpoint"] == acceptance.OFFLINE_PROVIDER_ENDPOINT
    assert rows[0]["status"] == "succeeded"
    assert rows[0]["test_only"] is True


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_offline_executor_refuses_corrupted_succeeded_output(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    stats = acceptance.OfflineExecutionStats()
    execute = acceptance._offline_executor_factory(tmp_path, store, stats=stats)
    request = _request(tmp_path, hashlib.sha256(b"fixture-a").hexdigest())
    execute(request)
    request.output_path.write_bytes(b"corrupt replacement")

    with pytest.raises(RuntimeError, match="no longer matches its ledger"):
        execute(request)

    assert len(acceptance._task_rows(tmp_path / "runtime.db")) == 1


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_offline_executor_changed_fingerprint_creates_new_task(tmp_path):
    store = GenerationTaskStore(tmp_path / "runtime.db")
    stats = acceptance.OfflineExecutionStats()
    execute = acceptance._offline_executor_factory(tmp_path, store, stats=stats)
    execute(_request(tmp_path, hashlib.sha256(b"fixture-a").hexdigest()))
    execute(_request(tmp_path, hashlib.sha256(b"fixture-b").hexdigest()))

    rows = acceptance._task_rows(tmp_path / "runtime.db")

    assert len(rows) == 2
    assert len({row["input_fingerprint"] for row in rows}) == 2
    assert stats.generated_chunks == 2
    assert stats.reused_chunks == 0


def test_provider_request_guard_fails_closed(monkeypatch):
    from clients import seedance_client

    stats = acceptance.OfflineExecutionStats()
    with acceptance._deny_provider_requests(stats):
        with pytest.raises(acceptance.OfflineProviderRequestError):
            seedance_client.submit_content([])

    assert stats.provider_requests == 1


def test_offline_transition_embedding_runner_returns_no_remote_evidence():
    assert acceptance._offline_transition_embedding_runner("shots", run_id="run") == {}


def test_completed_media_validation_fails_closed_on_hash_change(monkeypatch, tmp_path):
    recorded = {
        "raw_assembly.mp4": {"sha256": "raw-stable"},
        "polished.mp4": {"sha256": "polished-stable"},
    }
    current = dict(recorded)
    monkeypatch.setattr(
        acceptance,
        "_media_summary",
        lambda path: {"sha256": current[path.name]["sha256"]},
    )

    validated = acceptance._validate_completed_media(
        tmp_path, {"media": recorded}
    )

    assert validated == recorded
    current["polished.mp4"] = {"sha256": "changed"}
    with pytest.raises(RuntimeError, match="polished.mp4"):
        acceptance._validate_completed_media(tmp_path, {"media": recorded})


def test_completed_resume_reuses_final_media_without_rerender(monkeypatch, tmp_path):
    from phases.phase6 import phase6_video_gen
    from phases.phase7 import phase7_consistency
    from phases.phase8 import phase8_assembly
    from phases.phase9 import phase9_post

    for name, payload in (
        ("STORYBOARD.json", {"shots": []}),
        ("CONTINUITY_PLAN.json", {}),
        ("storyboard_qa_report.json", {"gate_passed": True}),
    ):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    media = {
        name: {"sha256": f"stable-{name}"}
        for name in ("raw_assembly.mp4", "polished.mp4")
    }
    receipt = {
        "schema": acceptance.RECEIPT_SCHEMA,
        "status": "completed",
        "fixture": {"sha256": acceptance.FIXTURE_SHA256},
        "invocations": [
            {
                "mode": "cold",
                "status": "completed",
                "started_at": "cold-start",
                "media": media,
            }
        ],
    }
    (tmp_path / acceptance.RECEIPT_NAME).write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    task = {
        "task_id": "offline-task",
        "provider_id": acceptance.OFFLINE_PROVIDER_ID,
        "provider_endpoint": acceptance.OFFLINE_PROVIDER_ENDPOINT,
        "status": "succeeded",
        "test_only": True,
    }
    monkeypatch.setattr(acceptance, "_task_rows", lambda _path: [task])
    monkeypatch.setattr(
        acceptance,
        "_media_summary",
        lambda path: media[path.name],
    )
    monkeypatch.setattr(
        phase6_video_gen,
        "run_phase6",
        lambda *_args, **_kwargs: {"status": "done"},
    )
    monkeypatch.setattr(
        phase7_consistency,
        "run_phase7",
        lambda *_args, **_kwargs: {"status": "done"},
    )

    def fail_rerender(*_args, **_kwargs):
        raise AssertionError("completed resume must not rerender final media")

    monkeypatch.setattr(phase8_assembly, "run_phase8", fail_rerender)
    monkeypatch.setattr(phase9_post, "run_phase9", fail_rerender)

    result = acceptance.run_acceptance(tmp_path, resume=True)
    latest = result["invocations"][-1]

    assert latest["status"] == "completed"
    assert latest["phase_results"]["phase8"]["mode"] == "reused_completed_media"
    assert latest["phase_results"]["phase9"]["mode"] == "reused_completed_media"
    assert latest["resume_lineage"] == {
        "task_ids_preserved": True,
        "media_hashes_preserved": True,
    }


@pytest.mark.parametrize("failed_phase", ["phase7", "phase8", "phase9"])
def test_phase7_to_phase9_failure_is_persisted(
    monkeypatch,
    tmp_path,
    failed_phase,
):
    from phases.phase6 import phase6_video_gen
    from phases.phase7 import phase7_consistency
    from phases.phase8 import phase8_assembly
    from phases.phase9 import phase9_post

    def prepare(output_dir):
        (output_dir / "STORYBOARD.json").write_text(
            json.dumps({"shots": []}), encoding="utf-8"
        )
        (output_dir / "CONTINUITY_PLAN.json").write_text("{}", encoding="utf-8")
        (output_dir / "storyboard_qa_report.json").write_text(
            json.dumps({"gate_passed": True}), encoding="utf-8"
        )
        return {"status": "completed"}

    monkeypatch.setattr(acceptance, "_prepare_phase1_to_phase5", prepare)

    def result_for(name):
        return (
            {"status": "error", "error": f"{name} acceptance blocker"}
            if name == failed_phase
            else {"status": "done", "outputs": []}
        )

    monkeypatch.setattr(
        phase6_video_gen,
        "run_phase6",
        lambda *_args, **_kwargs: result_for("phase6"),
    )
    monkeypatch.setattr(
        phase7_consistency,
        "run_phase7",
        lambda *_args, **_kwargs: result_for("phase7"),
    )
    phase8_kwargs = {}

    def run_phase8(*_args, **kwargs):
        phase8_kwargs.update(kwargs)
        return result_for("phase8")

    monkeypatch.setattr(phase8_assembly, "run_phase8", run_phase8)
    monkeypatch.setattr(
        phase9_post,
        "run_phase9",
        lambda *_args, **_kwargs: result_for("phase9"),
    )

    phase_number = failed_phase.removeprefix("phase")
    with pytest.raises(RuntimeError, match=f"Phase {phase_number} failed"):
        acceptance.run_acceptance(tmp_path)

    receipt = json.loads((tmp_path / acceptance.RECEIPT_NAME).read_text())
    invocation = receipt["invocations"][-1]
    assert receipt["status"] == "failed"
    assert invocation["status"] == "failed"
    assert invocation["phase_results"][failed_phase]["status"] == "error"
    if failed_phase == "phase8":
        assert (
            phase8_kwargs["_transition_embedding_runner"]
            is acceptance._offline_transition_embedding_runner
        )

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.run_memory import RunMemory

_runner_spec = importlib.util.spec_from_file_location("run_memory_pipeline_runner", SRC / "pipeline_runner.py")
runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(runner)
_compact_phase_record = runner._compact_phase_record
_record_run_memory = runner._record_run_memory


def _fake_summary(entries):
    return "summary: " + " | ".join(entry["content"] for entry in entries)


def test_add_and_get_three_tiers(tmp_path):
    memory = RunMemory(tmp_path, messages_per_summary=3, summarizer=_fake_summary)
    memory.add("phase", "script characters ready")
    memory.add("phase", "storyboard frames ready")
    memory.add("phase", "video shots rendered")
    memory.add("phase", "audio mix pending")

    tiers = memory.get("video shots", short_term_limit=5, summary_limit=10, rag_limit=3)

    assert [item["content"] for item in tiers["short_term"]] == ["audio mix pending"]
    assert len(tiers["summaries"]) == 1
    assert tiers["summaries"][0]["related_ids"] == [1, 2, 3]
    assert len(tiers["rag"]) == 3


def test_auto_summary_triggers_at_threshold(tmp_path):
    calls = []

    def summarizer(entries):
        calls.append(entries)
        return "compressed batch"

    memory = RunMemory(tmp_path, messages_per_summary=2, summarizer=summarizer)
    memory.add("phase", "first")
    assert calls == []
    memory.add("phase", "second")

    assert [[entry["content"] for entry in call] for call in calls] == [["first", "second"]]
    tiers = memory.get("")
    assert tiers["short_term"] == []
    assert tiers["summaries"][0]["content"] == "compressed batch"


def test_rag_ranks_most_similar_first_with_deterministic_vectors(tmp_path):
    memory = RunMemory(tmp_path, messages_per_summary=10, summarizer=_fake_summary)
    memory.add("phase", "apple apple orchard")
    memory.add("phase", "engine render video")
    memory.add("phase", "apple fruit")

    rag = memory.get("apple orchard", rag_limit=3)["rag"]

    assert rag[0]["content"] == "apple apple orchard"
    assert rag[0]["score"] > rag[1]["score"] > rag[2]["score"]


def test_deep_retrieve_expands_summary_related_ids(tmp_path):
    memory = RunMemory(tmp_path, messages_per_summary=2, summarizer=_fake_summary)
    first = memory.add("phase", "character design approved")
    second = memory.add("phase", "character reference saved")
    memory.add("phase", "audio mix complete")
    memory.add("phase", "music artifact saved")

    expanded = memory.deep_retrieve("character design", summary_limit=1)

    assert [item["id"] for item in expanded] == [first, second]


def test_database_is_valid_and_uses_wal_mode(tmp_path):
    memory = RunMemory(tmp_path, messages_per_summary=10, summarizer=_fake_summary)
    memory.add("phase", "phase record")

    with sqlite3.connect(memory.db_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        columns = [row[1] for row in connection.execute("PRAGMA table_info(memories)")]
    assert columns == [
        "id", "type", "role", "content", "embedding", "related_ids", "summarized", "created_at"
    ]


def test_disabled_config_writes_nothing(tmp_path):
    _record_run_memory(
        {"phases": {"1": {"status": "done"}}},
        tmp_path,
        {"memory_enabled": False},
    )

    assert not (tmp_path / "run_memory.db").exists()


def test_runner_records_executed_statuses_compactly(tmp_path):
    created = []

    class FakeMemory:
        def __init__(self, output_dir, messages_per_summary):
            created.append((output_dir, messages_per_summary, []))

        def add(self, role, content):
            created[0][2].append((role, json.loads(content)))

    report = {
        "phases": {
            "1": {"status": "done", "duration_s": 2.5, "outputs": ["a/b.json"]},
            "2": {"status": "error", "error": "failed"},
            "3": {"status": "skipped", "reason": "user-specified"},
        }
    }
    _record_run_memory(
        report,
        tmp_path,
        {"memory_enabled": True, "memory_messages_per_summary": 7},
        memory_factory=FakeMemory,
    )

    assert created[0][1] == 7
    records = created[0][2]
    assert [record[1]["status"] for record in records] == ["done", "error"]
    assert records[0][1]["artifacts"] == ["b.json"]
    assert all(len(_compact_phase_record(record[1]["phase"], record[1])) < 500 for record in records)


def test_runner_summarizes_locally_without_post_run_llm(monkeypatch, tmp_path):
    def unexpected_llm_call(*_args, **_kwargs):
        raise AssertionError("run-memory finalization must not call an external LLM")

    blocked_supervision = ModuleType("quality.supervision_agent")
    blocked_supervision._call_llm = unexpected_llm_call
    monkeypatch.setitem(sys.modules, "quality.supervision_agent", blocked_supervision)
    _record_run_memory(
        {
            "phases": {
                "1": {"status": "done", "duration_s": 1.0},
                "2": {"status": "done", "duration_s": 2.0},
                "3": {"status": "done", "duration_s": 3.0},
                "4": {"status": "skipped", "reason": "selected range"},
            }
        },
        tmp_path,
        {"memory_enabled": True, "memory_messages_per_summary": 3},
    )

    tiers = RunMemory(tmp_path).get("")
    assert len(tiers["summaries"]) == 1
    assert '"phase": "phase1"' in tiers["summaries"][0]["content"]
    assert '"phase": "phase3"' in tiers["summaries"][0]["content"]
    assert "phase4" not in tiers["summaries"][0]["content"]

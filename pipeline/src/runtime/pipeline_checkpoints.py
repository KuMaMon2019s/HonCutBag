"""Pipeline stage checkpoint and resume integration."""

import json
from pathlib import Path
from typing import Optional

from tools.checkpoint import write_checkpoint as write_stage_checkpoint
from utils.artifact_chain import PHASE_SEQUENCE, phase_numbers_before


PHASE_ORDER = list(PHASE_SEQUENCE)

try:
    from langgraph.checkpoint.base import empty_checkpoint
    from langgraph.checkpoint.sqlite import SqliteSaver

    LANGGRAPH_CHECKPOINTS_AVAILABLE = True
except ImportError:
    empty_checkpoint = None
    SqliteSaver = None
    LANGGRAPH_CHECKPOINTS_AVAILABLE = False


_sqlite_saver_instance = None
_sqlite_saver_path = None


def resume_skip_phases(existing_skip: list[float], resume_from: str) -> list[float]:
    return sorted(set(existing_skip).union(phase_numbers_before(resume_from)))


def checkpoint_path(output_dir: Path) -> Path:
    return Path(output_dir) / "checkpoint.json"


def record_stage_checkpoint(
    output_dir: Path,
    phase_name: str,
    phase_result: dict,
) -> Path:
    target = checkpoint_path(output_dir)
    if phase_result.get("status", "") != "done":
        return target
    safe_result = {}
    for key, value in phase_result.items():
        if key.startswith("_"):
            continue
        try:
            json.dumps(value, default=str)
            safe_result[key] = value
        except (TypeError, ValueError):
            safe_result[key] = str(value)
    run_fingerprint = None
    project_id = "local"
    manifest_path = Path(output_dir) / "RUN_MANIFEST.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            run_fingerprint = manifest.get("run_fingerprint")
            project_id = manifest.get("resolved_config", {}).get(
                "project_id",
                "local",
            )
        except (OSError, json.JSONDecodeError):
            pass
    checkpoint = write_stage_checkpoint(
        target,
        phase_name,
        safe_result,
        run_fingerprint=run_fingerprint,
        project_id=project_id,
    )
    if LANGGRAPH_CHECKPOINTS_AVAILABLE:
        try:
            save_state_to_sqlite(checkpoint, output_dir, thread_id="pipeline_run")
        except Exception as exc:
            print(f"  ⚠ SQLite checkpoint 写入失败: {exc}")
    return target


def read_checkpoint(output_dir: Path) -> Optional[dict]:
    target = checkpoint_path(output_dir)
    if not target.exists():
        return None
    try:
        with target.open(encoding="utf-8") as source:
            checkpoint = json.load(source)
        if not isinstance(checkpoint.get("completed"), list):
            return None
        manifest_path = Path(output_dir) / "RUN_MANIFEST.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if checkpoint.get("run_fingerprint") != manifest.get("run_fingerprint"):
                return None
            expected_project = manifest.get("resolved_config", {}).get(
                "project_id",
                "local",
            )
            if checkpoint.get("project_id", "local") != expected_project:
                return None
        return checkpoint
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def get_sqlite_checkpointer(output_dir: Path):
    global _sqlite_saver_instance, _sqlite_saver_path
    if not LANGGRAPH_CHECKPOINTS_AVAILABLE:
        return None
    database_path = Path(output_dir) / "checkpoint.db"
    database_path_string = str(database_path)
    if (
        _sqlite_saver_instance is not None
        and _sqlite_saver_path == database_path_string
    ):
        return _sqlite_saver_instance
    try:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        _sqlite_saver_instance = SqliteSaver.from_conn_string(database_path_string)
        _sqlite_saver_path = database_path_string
        return _sqlite_saver_instance
    except Exception as exc:
        print(f"⚠ SqliteSaver initialization failed: {exc}")
        return None


def save_state_to_sqlite(
    state: dict,
    output_dir: Path,
    thread_id: str = "default",
) -> bool:
    if not LANGGRAPH_CHECKPOINTS_AVAILABLE:
        return False
    try:
        database_path = Path(output_dir) / "checkpoint.db"
        database_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = state
        config = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }
        with SqliteSaver.from_conn_string(str(database_path)) as checkpointer:
            checkpointer.put(
                config,
                checkpoint,
                {
                    "source": "update",
                    "step": len(state.get("completed", [])),
                    "writes": state,
                },
                {},
            )
        return True
    except Exception as exc:
        print(f"⚠ Failed to save state to SQLite: {exc}")
        return False


def load_state_from_sqlite(
    output_dir: Path,
    thread_id: str = "default",
) -> Optional[dict]:
    if not LANGGRAPH_CHECKPOINTS_AVAILABLE:
        return None
    try:
        database_path = Path(output_dir) / "checkpoint.db"
        if not database_path.exists():
            return None
        with SqliteSaver.from_conn_string(str(database_path)) as checkpointer:
            config = {
                "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
            }
            checkpoint = checkpointer.get_tuple(config)
        if checkpoint and checkpoint.checkpoint:
            return checkpoint.checkpoint.get("channel_values", {})
        return None
    except Exception as exc:
        print(f"⚠ Failed to load state from SQLite: {exc}")
        return None


def get_completed_stages(output_dir: Path) -> list:
    checkpoint = read_checkpoint(output_dir)
    return [] if checkpoint is None else checkpoint.get("completed", [])


def get_next_stage(
    output_dir: Path,
    all_phases: list | None = None,
) -> Optional[str]:
    phases = PHASE_ORDER if all_phases is None else all_phases
    completed = set(get_completed_stages(output_dir))
    return next((phase for phase in phases if phase not in completed), None)


_checkpoint_path = checkpoint_path
_get_completed_stages = get_completed_stages
_get_next_stage = get_next_stage
_read_checkpoint = read_checkpoint
_record_stage_checkpoint = record_stage_checkpoint
_resume_skip_phases = resume_skip_phases


__all__ = [
    "LANGGRAPH_CHECKPOINTS_AVAILABLE",
    "PHASE_ORDER",
    "_checkpoint_path",
    "_get_completed_stages",
    "_get_next_stage",
    "_read_checkpoint",
    "_record_stage_checkpoint",
    "_resume_skip_phases",
    "get_completed_stages",
    "get_next_stage",
    "get_sqlite_checkpointer",
    "load_state_from_sqlite",
    "read_checkpoint",
    "record_stage_checkpoint",
    "resume_skip_phases",
    "save_state_to_sqlite",
]

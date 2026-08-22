from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from runtime.provider_responses import (
    ProviderResponseError,
    parse_seedance_task,
    parse_video_submission,
)
from runtime.security_boundaries import (
    CorrelationContext,
    emit_runtime_event,
    redact_for_log,
    resolve_within_workspace,
    validate_subprocess_args,
)


def test_workspace_paths_reject_traversal_absolute_escape_and_symlink(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = workspace / "shots/S01/output.mp4"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"video")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    (workspace / "escape.mp4").symlink_to(outside)

    assert resolve_within_workspace(
        workspace,
        "shots/S01/output.mp4",
        must_exist=True,
    ) == inside
    with pytest.raises(ValueError, match="escapes"):
        resolve_within_workspace(workspace, "../outside.mp4", must_exist=True)
    with pytest.raises(ValueError, match="escapes"):
        resolve_within_workspace(workspace, outside, must_exist=True)
    with pytest.raises(ValueError, match="escapes"):
        resolve_within_workspace(workspace, "escape.mp4", must_exist=True)


def test_subprocess_boundary_requires_argument_arrays_and_preserves_metacharacters():
    with pytest.raises(TypeError, match="argument arrays"):
        validate_subprocess_args("ffmpeg -i input.mp4")
    with pytest.raises(ValueError, match="NUL"):
        validate_subprocess_args(["ffmpeg", "bad\x00path"])
    assert validate_subprocess_args(["ffmpeg", "-i", "name;rm -rf ignored"]) == [
        "ffmpeg",
        "-i",
        "name;rm -rf ignored",
    ]


def test_redaction_removes_credentials_and_replaces_prompt_with_digest(monkeypatch):
    monkeypatch.setenv("ARK_AGENT_API_KEY", "environment-secret-value")
    redacted = redact_for_log(
        {
            "api_key": "direct-secret",
            "prompt": "private authored prompt",
            "nested": {
                "message": "Authorization: Bearer abc.def and environment-secret-value"
            },
        }
    )
    serialized = json.dumps(redacted, sort_keys=True)
    assert "direct-secret" not in serialized
    assert "private authored prompt" not in serialized
    assert "environment-secret-value" not in serialized
    assert redacted["prompt"]["length"] == len("private authored prompt")


def test_runtime_event_always_contains_correlation_ids_and_redacts(capsys):
    emit_runtime_event(
        "provider_task_active",
        CorrelationContext(
            project_id="project-1",
            run_id="run-1",
            node_id="phase6.video_generation",
            task_id="task-1",
        ),
        prompt="do not log this",
        token="secret-token",
    )
    event = json.loads(capsys.readouterr().out)
    assert event["project_id"] == "project-1"
    assert event["run_id"] == "run-1"
    assert event["node_id"] == "phase6.video_generation"
    assert event["task_id"] == "task-1"
    assert event["prompt"]["length"] == len("do not log this")
    assert event["token"] == "[REDACTED]"


def test_provider_response_schemas_fail_closed_without_echoing_payload():
    assert parse_video_submission(
        {"id": "job-1", "provider_extra": True},
        provider_id="seedance",
    ).task_id == "job-1"
    assert parse_seedance_task({"id": "job-1", "status": "running"}).status == (
        "running"
    )
    with pytest.raises(ProviderResponseError, match="task ID") as missing:
        parse_video_submission(
            {"prompt": "private prompt", "api_key": "secret"},
            provider_id="bridge",
        )
    assert "private prompt" not in str(missing.value)
    with pytest.raises(ProviderResponseError, match="invalid schema"):
        parse_seedance_task({"id": "job-1", "status": "invented"})


def test_repository_subprocess_calls_never_enable_shell_true():
    source_root = Path(__file__).resolve().parents[1] / "src"
    violations = []
    for source in source_root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    violations.append(str(source.relative_to(source_root)))
    assert violations == []

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clients import asr_client, local_video_client
from utils import config


def test_project_agent_key_overrides_stale_process_key_and_preserves_coding_key(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ARK_AGENT_API_KEY=project-agent-key\nARK_API_KEY=project-coding-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ARK_AGENT_API_KEY", "stale-agent-key")
    monkeypatch.setenv("ARK_API_KEY", "stale-coding-key")

    source = config.configure_ark_agent_environment(env_file)

    assert source == "project_env"
    assert config.os.environ["ARK_AGENT_API_KEY"] == "project-agent-key"
    assert config.os.environ["ARK_API_KEY"] == "stale-coding-key"


def test_exported_agent_key_is_kept_when_project_env_has_no_agent_key(
    tmp_path, monkeypatch
):
    env_file = tmp_path / ".env"
    env_file.write_text("ARK_API_KEY=ignored-coding-key\n", encoding="utf-8")
    monkeypatch.setenv("ARK_AGENT_API_KEY", "exported-agent-key")
    monkeypatch.setenv("ARK_API_KEY", "stale-coding-key")

    source = config.configure_ark_agent_environment(env_file)

    assert source == "process_env"
    assert config.os.environ["ARK_AGENT_API_KEY"] == "exported-agent-key"
    assert config.os.environ["ARK_API_KEY"] == "stale-coding-key"


def test_project_seedance_model_overrides_stale_launcher_model(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SEEDANCE_MODEL=doubao-seedance-2.0-fast\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SEEDANCE_MODEL", "doubao-seedance-2-0-fast")

    source = config.configure_seedance_model_environment(env_file)

    assert source == "project_env"
    assert config.os.environ["SEEDANCE_MODEL"] == "doubao-seedance-2.0-fast"


def test_bridge_seedance_credentials_never_fall_back_to_ark_api_key(monkeypatch):
    monkeypatch.delenv("ARK_AGENT_API_KEY", raising=False)
    monkeypatch.setenv("ARK_API_KEY", "coding-key-must-not-be-used")

    with pytest.raises(ValueError, match="ARK_AGENT_API_KEY not set"):
        local_video_client._get_ark_api_key()


def test_central_key_lookup_ignores_but_preserves_ark_api_key(monkeypatch):
    monkeypatch.delenv("ARK_AGENT_API_KEY", raising=False)
    monkeypatch.setenv("ARK_API_KEY", "coding-key-must-not-be-used")

    assert config.get_api_key("ARK_CODING") is None
    assert config.APIKeys.get("ARK_API_KEY") is None
    assert config.APIKeys.get("MISSING", fallback="ARK_API_KEY") is None
    assert config.os.environ["ARK_API_KEY"] == "coding-key-must-not-be-used"


def test_required_key_lookup_rejects_but_preserves_ark_api_key(monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "coding-key-must-not-be-used")

    with pytest.raises(ValueError, match="reserved for Honcho Coding Plan memory"):
        config.get_api_key_or_raise("ARK_CODING")

    assert config.os.environ["ARK_API_KEY"] == "coding-key-must-not-be-used"


def test_seed_asr_credentials_never_fall_back_to_ark_api_key(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"not-read-before-credential-validation")
    monkeypatch.delenv("ARK_AGENT_API_KEY", raising=False)
    monkeypatch.setenv("ARK_API_KEY", "coding-key-must-not-be-used")

    with pytest.raises(RuntimeError, match="SeedASR requires ARK_AGENT_API_KEY"):
        asr_client.transcribe_audio(str(audio_path))

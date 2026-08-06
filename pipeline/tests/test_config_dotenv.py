"""Regression tests for loading the repository-level .env file."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "pipeline" / "src"


@pytest.mark.skipif(
    not (PROJECT_ROOT / ".env").exists(),
    reason="repository .env is intentionally not present in CI",
)
def test_pipeline_runner_subprocess_loads_project_dotenv():
    env_file = PROJECT_ROOT / ".env"
    assert env_file.exists()

    assert any(
        line.strip().startswith("ARK_AGENT_API_KEY=")
        for line in env_file.read_text(encoding="utf-8").splitlines()
    )

    child_env = os.environ.copy()
    child_env.pop("ARK_AGENT_API_KEY", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pipeline_runner; from utils.config import get_api_key; "
            "print(bool(get_api_key('ARK_AGENT_API_KEY')))",
        ],
        cwd=PROJECT_ROOT,
        env={**child_env, "PYTHONPATH": str(SRC_DIR)},
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "True"

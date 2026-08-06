"""Tests for pipeline startup dependency validation."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.deps import check_dependencies


def test_dependency_check_accepts_installed_modules():
    check_dependencies(("sys", "pathlib"))


def test_dependency_check_reports_missing_modules_and_interpreter():
    def import_module(name):
        if name == "missing_package":
            raise ImportError(name)
        return object()

    with patch("utils.deps.importlib.import_module", side_effect=import_module):
        with pytest.raises(ImportError) as exc_info:
            check_dependencies(("installed_package", "missing_package"))

    message = str(exc_info.value)
    assert "missing_package" in message
    assert sys.executable in message
    assert "-m pip install -r requirements.txt" in message

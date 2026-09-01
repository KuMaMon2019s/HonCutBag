from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "pipeline" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import tos_upload_live_acceptance as acceptance


def _fixture(workspace: Path) -> Path:
    image_path = workspace / "input" / "tos-live-fixture.png"
    image_path.parent.mkdir(parents=True)
    pixels = np.random.default_rng(23).integers(
        0,
        256,
        size=(1024, 1024, 3),
        dtype=np.uint8,
    )
    Image.fromarray(pixels, "RGB").save(image_path, format="PNG")
    return image_path


def test_tos_live_preflight_is_zero_network_and_finite(monkeypatch, tmp_path):
    image_path = _fixture(tmp_path)
    regression_path = tmp_path / "tos_upload_regression.json"
    regression_path.write_text(
        json.dumps(
            {
                "schema": acceptance.REGRESSION_SCHEMA,
                "status": "passed",
                "git_commit": "test-commit",
            }
        )
    )
    monkeypatch.setattr(
        acceptance,
        "_repo_identity",
        lambda: {
            "git_commit": "test-commit",
            "tracked_worktree_clean": True,
        },
    )
    monkeypatch.setattr(acceptance, "is_media_upload_configured", lambda: True)
    monkeypatch.setattr(
        acceptance,
        "upload_multimodal_media_file_required",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight must not upload")
        ),
    )

    receipt = acceptance.build_preflight(
        tmp_path,
        image_path,
        regression_path,
    )

    assert receipt["status"] == "preflight_passed"
    assert receipt["provider_request_count"] == 0
    assert receipt["hard_limits"] == {
        "authoritative_puts": 1,
        "read_only_head_checks": 2,
        "other_provider_requests": 0,
    }
    assert receipt["fixture"]["sha256"]
    assert receipt["object_prefix"].endswith(receipt["acceptance_id"])


def test_tos_live_preflight_rejects_stale_regression(monkeypatch, tmp_path):
    image_path = _fixture(tmp_path)
    regression_path = tmp_path / "tos_upload_regression.json"
    regression_path.write_text(
        json.dumps(
            {
                "schema": acceptance.REGRESSION_SCHEMA,
                "status": "passed",
                "git_commit": "old-commit",
            }
        )
    )
    monkeypatch.setattr(
        acceptance,
        "_repo_identity",
        lambda: {
            "git_commit": "new-commit",
            "tracked_worktree_clean": True,
        },
    )

    try:
        acceptance.build_preflight(tmp_path, image_path, regression_path)
    except RuntimeError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("stale regression evidence must fail closed")

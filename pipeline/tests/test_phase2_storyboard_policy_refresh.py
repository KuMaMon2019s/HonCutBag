from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from phases.phase2 import shot_storyboards, storyboard_guide_pose


def _storyboard() -> dict:
    return {
        "shots": [
            {
                "id": "S01",
                "where": "test stage",
                "character_ids": [],
                "storyboard_beats": [
                    {
                        "beat_id": "S01_P01",
                        "planner_version": "honcut.secondary-storyboard.v16",
                        "duration_s": 5,
                        "generation_mode": "fresh",
                        "character_ids": [],
                        "action": "performer raises one arm",
                        "generation_action_units": [
                            {
                                "unit_id": "GAU001",
                                "actions": ["performer raises one arm"],
                                "performers": ["performer"],
                                "ledger_indexes": [0],
                            }
                        ],
                        "end_state": "arm raised",
                    },
                    {
                        "beat_id": "S01_P02",
                        "planner_version": "honcut.secondary-storyboard.v16",
                        "duration_s": 5,
                        "generation_mode": "extend",
                        "character_ids": [],
                        "action": "performer steps forward",
                        "generation_action_units": [
                            {
                                "unit_id": "GAU002",
                                "actions": ["performer steps forward"],
                                "performers": ["performer"],
                                "ledger_indexes": [1],
                            }
                        ],
                        "end_state": "step completed",
                    },
                ],
            }
        ]
    }


class _FixtureImageClient:
    model = "fixture-seedream"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _write(self, output_path: str) -> str:
        self.calls.append(Path(output_path).name)
        Image.effect_noise((640, 360), 30 + len(self.calls)).convert("RGB").save(
            output_path
        )
        return "https://image.invalid/fixture.png"

    def text_to_image(self, prompt, output_path, size, timeout):
        del prompt, size, timeout
        return self._write(output_path)

    def image_to_image(self, prompt, ref_image, output_path, size):
        del prompt, ref_image, size
        return self._write(output_path)


def _generated_v7(tmp_path: Path) -> tuple[dict, dict, _FixtureImageClient]:
    storyboard = _storyboard()
    client = _FixtureImageClient()
    manifest = shot_storyboards.generate_shot_storyboards(
        tmp_path,
        storyboard,
        [],
        client=client,
        director_storyboard_path=tmp_path / "missing.png",
    )
    assert manifest["kind"] == "honcut.shot_storyboards.v7"
    return storyboard, manifest, client


def test_v7_old_policy_refresh_is_local_side_by_side_and_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storyboard, source_manifest, client = _generated_v7(tmp_path)
    source_bytes = (tmp_path / "SHOT_STORYBOARDS.json").read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_assets = {
        relative_path: (tmp_path / relative_path).read_bytes()
        for record in source_manifest["shots"]
        for guide in record["narrative_guides"]
        for relative_path in [
            guide["image"],
            guide["receipt"],
            guide["pose_atlas_receipt"],
            *[
                page["image"]
                for candidate in guide["pose_atlas_candidates"]
                for page in candidate["pages"]
            ],
        ]
    }
    provider_calls = list(client.calls)
    refreshed_policy = "f" * 64
    monkeypatch.setattr(storyboard_guide_pose, "POSE_POLICY_SHA256", refreshed_policy)
    monkeypatch.setattr(shot_storyboards, "POSE_POLICY_SHA256", refreshed_policy)

    migrated = shot_storyboards.migrate_shot_storyboard_narrative_guides(
        tmp_path,
        storyboard,
    )
    repeated = shot_storyboards.migrate_shot_storyboard_narrative_guides(
        tmp_path,
        storyboard,
    )

    assert client.calls == provider_calls
    assert repeated == migrated
    assert migrated["migration"]["policy"] == "verified_v7_pose_policy_refresh_v1"
    assert migrated["migration"]["source_manifest_sha256"] == source_sha256
    assert migrated["migration"]["source_pose_policy_sha256"] != refreshed_policy
    assert migrated["migration"]["target_pose_policy_sha256"] == refreshed_policy
    assert migrated["migration"]["provider_request_count"] == 0
    assert all(
        guide["pose_policy_sha256"] == refreshed_policy
        and guide["image"].startswith(
            f"storyboard_guides/policy-refresh/{source_sha256}/"
        )
        and guide["pose_atlas_receipt"].startswith(
            f"storyboard_pose_atlases/policy-refresh/{source_sha256}/"
        )
        for record in migrated["shots"]
        for guide in record["narrative_guides"]
    )
    assert all((tmp_path / path).read_bytes() == payload for path, payload in source_assets.items())
    receipt_path = tmp_path / migrated["migration"]["receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "done"
    assert receipt["legacy_source_assets_modified"] is False
    assert receipt["provider_request_count"] == 0
    assert receipt["audit_only_assets"]
    assert all(
        (tmp_path / item["audit_path"]).is_file()
        and hashlib.sha256((tmp_path / item["audit_path"]).read_bytes()).hexdigest()
        == item["sha256"]
        for item in receipt["audit_only_assets"]
    )
    assert shot_storyboards.validate_shot_storyboard_artifacts(tmp_path, storyboard) == []


def _corrupt_grid(manifest: dict, storyboard: dict, output_dir: Path) -> None:
    del storyboard, output_dir
    manifest["shots"][0]["grid_contract"]["cells"][0]["label"] = "S01_G99"


def _corrupt_source_hash(manifest: dict, storyboard: dict, output_dir: Path) -> None:
    del storyboard, output_dir
    manifest["shots"][0]["narrative_guides"][0]["source_board_sha256"] = "0" * 64


def _corrupt_action_lineage(manifest: dict, storyboard: dict, output_dir: Path) -> None:
    del manifest, output_dir
    storyboard["shots"][0]["storyboard_beats"][0]["generation_action_units"][0][
        "ledger_indexes"
    ] = [99]


def _corrupt_atlas(manifest: dict, storyboard: dict, output_dir: Path) -> None:
    del storyboard
    page = manifest["shots"][0]["narrative_guides"][0]["pose_atlas_candidates"][0][
        "pages"
    ][0]
    (output_dir / page["image"]).write_bytes(b"corrupt atlas page")


@pytest.mark.parametrize(
    "corrupt",
    [_corrupt_grid, _corrupt_source_hash, _corrupt_action_lineage, _corrupt_atlas],
    ids=["grid", "source-hash", "action-lineage", "atlas"],
)
def test_v7_policy_refresh_fails_closed_before_replacing_old_assets(
    tmp_path: Path,
    monkeypatch,
    corrupt: Callable[[dict, dict, Path], None],
) -> None:
    storyboard, manifest, client = _generated_v7(tmp_path)
    manifest_path = tmp_path / "SHOT_STORYBOARDS.json"
    old_guide_path = tmp_path / manifest["shots"][0]["narrative_guides"][0]["image"]
    old_guide_bytes = old_guide_path.read_bytes()
    corrupt(manifest, storyboard, tmp_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_bytes = manifest_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    provider_calls = list(client.calls)
    refreshed_policy = "f" * 64
    monkeypatch.setattr(storyboard_guide_pose, "POSE_POLICY_SHA256", refreshed_policy)
    monkeypatch.setattr(shot_storyboards, "POSE_POLICY_SHA256", refreshed_policy)

    with pytest.raises(RuntimeError):
        shot_storyboards.migrate_shot_storyboard_narrative_guides(tmp_path, storyboard)

    assert client.calls == provider_calls
    assert manifest_path.read_bytes() == source_bytes
    assert old_guide_path.read_bytes() == old_guide_bytes
    assert not (tmp_path / "storyboard_guides" / "policy-refresh").exists()
    receipt = json.loads(
        (
            tmp_path
            / "storyboard_guides"
            / "migrations"
            / f"{source_sha256}.audit-only.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["kind"] == "honcut.storyboard-guide-policy-refresh.v1"
    assert receipt["status"] == "audit_only"
    assert receipt["legacy_source_assets_modified"] is False
    assert receipt["provider_request_count"] == 0

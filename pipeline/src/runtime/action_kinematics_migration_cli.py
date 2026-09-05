"""CLI entrypoint for the zero-provider action-kinematics Artifact migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.action_kinematics_migration import migrate_action_kinematics_artifact
from runtime.artifact_manifest import ArtifactManifestStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Register a deterministic action-kinematics sidecar for one Artifact."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--parent-artifact-id", required=True)
    parser.add_argument("--downstream-artifact-id", action="append", default=[])
    args = parser.parse_args()

    store = ArtifactManifestStore.from_run_directory(args.run_dir, required=True)
    assert store is not None
    result = migrate_action_kinematics_artifact(
        store,
        parent_artifact_id=args.parent_artifact_id,
        downstream_artifact_ids=tuple(args.downstream_artifact_id),
    )
    print(
        json.dumps(
            {
                "status": result["receipt"]["status"],
                "sidecar_artifact_id": result["sidecar_artifact_id"],
                "receipt_artifact_id": result["receipt_artifact_id"],
                "provider_request_count": result["provider_request_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Recoverable Phase 8 reshoot transactions and durable retry budgets."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STATE_FILE = "reshoot_state.json"
TRANSACTION_DIR = ".reshoot_transactions"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _state_path(output_dir: Path) -> Path:
    return Path(output_dir) / STATE_FILE


def read_state(output_dir: Path) -> dict[str, Any]:
    path = _state_path(output_dir)
    if not path.is_file():
        return {"status": "idle", "attempts": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "invalid", "attempts": []}
    return value if isinstance(value, dict) else {"status": "invalid", "attempts": []}


def _require_valid_state(output_dir: Path) -> dict[str, Any]:
    state = read_state(output_dir)
    valid_statuses = {"idle", "in_progress", "failed", "completed"}
    if state.get("status") not in valid_statuses or not isinstance(
        state.get("attempts"), list
    ):
        raise RuntimeError(
            "reshoot_state.json is invalid; refusing to reset the paid reshoot budget"
        )
    return state


def durable_attempt_count(output_dir: Path) -> int:
    """Return attempts from an unfinished reshoot cycle only."""
    state = _require_valid_state(output_dir)
    if state.get("status") == "completed":
        return 0
    return len(state.get("attempts", []))


def mark_cycle_completed(output_dir: Path) -> None:
    state = _require_valid_state(output_dir)
    state.update(status="completed", completed_at=_now())
    _write_json(_state_path(output_dir), state)


@dataclass
class ReshootTransaction:
    output_dir: Path
    transaction_id: str
    kind: str
    shot_ids: list[str]
    root: Path
    track_budget: bool = True

    @classmethod
    def begin(
        cls,
        output_dir: Path,
        *,
        kind: str,
        shot_ids: list[str],
        track_budget: bool = True,
    ) -> ReshootTransaction:
        output_dir = Path(output_dir)
        state = _require_valid_state(output_dir) if track_budget else None
        transaction_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        root = output_dir / TRANSACTION_DIR / transaction_id
        root.mkdir(parents=True, exist_ok=False)
        transaction = cls(
            output_dir, transaction_id, kind, list(shot_ids), root, track_budget
        )

        backed_up: list[str] = []
        for shot_id in shot_ids:
            source_dir = output_dir / "shots" / shot_id
            backup_dir = root / "shots" / shot_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("output.mp4", "SHOT_META.json", "last_frame.jpg"):
                source = source_dir / filename
                if source.is_file():
                    shutil.copy2(source, backup_dir / filename)
            if (backup_dir / "output.mp4").is_file():
                backed_up.append(shot_id)

        if backed_up != shot_ids:
            missing = sorted(set(shot_ids).difference(backed_up))
            shutil.rmtree(root, ignore_errors=True)
            raise FileNotFoundError(
                "Cannot start recoverable reshoot; source clips missing: " + ", ".join(missing)
            )

        receipt = {
            "transaction_id": transaction_id,
            "kind": kind,
            "shot_ids": shot_ids,
            "status": "prepared",
            "created_at": _now(),
        }
        _write_json(root / "receipt.json", receipt)
        if track_budget:
            assert state is not None
            if state.get("status") == "completed":
                state = {"status": "in_progress", "attempts": []}
            state.setdefault("attempts", []).append(receipt)
            state.update(status="in_progress", updated_at=_now())
            _write_json(_state_path(output_dir), state)
        return transaction

    def remove_sources(self) -> None:
        for shot_id in self.shot_ids:
            (self.output_dir / "shots" / shot_id / "output.mp4").unlink()
        self._set_status("sources_removed")

    def rollback(self, reason: str) -> None:
        for shot_id in self.shot_ids:
            live_dir = self.output_dir / "shots" / shot_id
            backup_dir = self.root / "shots" / shot_id
            live_dir.mkdir(parents=True, exist_ok=True)
            # Remove a partial replacement before restoring the known-good source.
            (live_dir / "output.mp4").unlink(missing_ok=True)
            for filename in ("output.mp4", "SHOT_META.json", "last_frame.jpg"):
                backup = backup_dir / filename
                if backup.is_file():
                    shutil.copy2(backup, live_dir / filename)
        self._set_status("rolled_back", error=reason)
        if self.track_budget:
            state = _require_valid_state(self.output_dir)
            state.update(status="failed", last_error=reason, updated_at=_now())
            self._update_attempt(state, "rolled_back", reason)
            _write_json(_state_path(self.output_dir), state)

    def commit(self) -> None:
        missing = [
            shot_id
            for shot_id in self.shot_ids
            if not (self.output_dir / "shots" / shot_id / "output.mp4").is_file()
        ]
        if missing:
            raise FileNotFoundError("Regenerated clips missing: " + ", ".join(missing))
        # Finish every fallible receipt/state write while the backup remains
        # at its rollback location. Archiving is the final same-filesystem
        # rename, so a caller can still roll back any earlier commit failure.
        self._set_status("committed")
        if self.track_budget:
            state = _require_valid_state(self.output_dir)
            state.update(status="in_progress", updated_at=_now())
            self._update_attempt(state, "committed")
            _write_json(_state_path(self.output_dir), state)
        archive = self.output_dir / "failed_artifacts" / "reshoot_replaced" / self.transaction_id
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.root), str(archive))

    def _set_status(self, status: str, *, error: str | None = None) -> None:
        receipt_path = self.root / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt.update(status=status, updated_at=_now())
        if error:
            receipt["error"] = error
        _write_json(receipt_path, receipt)

    def _update_attempt(
        self, state: dict[str, Any], status: str, error: str | None = None
    ) -> None:
        for attempt in state.get("attempts", []):
            if attempt.get("transaction_id") == self.transaction_id:
                attempt.update(status=status, updated_at=_now())
                if error:
                    attempt["error"] = error
                break

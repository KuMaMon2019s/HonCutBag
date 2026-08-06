"""Atomic HonCut stage checkpoint persistence."""
import json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class CheckpointValidationError(ValueError): pass
def validate_checkpoint(value: dict[str, Any]) -> None:
    if not isinstance(value.get("completed"), list): raise CheckpointValidationError("completed must be a list")
    if not isinstance(value.get("results"), dict): raise CheckpointValidationError("results must be an object")
def read_checkpoint(path: str|Path) -> dict[str, Any]|None:
    target=Path(path)
    if not target.exists(): return None
    value=json.loads(target.read_text(encoding="utf-8")); validate_checkpoint(value); return value
def write_checkpoint(path: str|Path, stage: str, result: dict[str, Any]) -> dict[str, Any]:
    target=Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    value=read_checkpoint(target) or {"completed": [], "results": {}, "timestamp": ""}
    if stage not in value["completed"]: value["completed"].append(stage)
    value["results"][stage]=result; value["timestamp"]=datetime.now(timezone.utc).isoformat()
    temporary=target.with_suffix(target.suffix+".tmp"); temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"); os.replace(temporary, target)
    return value
def get_next_stage(path: str|Path, stages: list[str]) -> str|None:
    value=read_checkpoint(path) or {"completed": []}; completed=set(value["completed"])
    return next((stage for stage in stages if stage not in completed), None)

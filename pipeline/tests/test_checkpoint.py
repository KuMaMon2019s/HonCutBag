import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]/"src"))
from tools.checkpoint import CheckpointValidationError, get_next_stage, read_checkpoint, validate_checkpoint, write_checkpoint
def test_missing_checkpoint_returns_none(tmp_path): assert read_checkpoint(tmp_path/"none.json") is None
def test_write_is_resumable(tmp_path):
    path=tmp_path/"cp.json"; write_checkpoint(path,"phase1",{"ok":True}); write_checkpoint(path,"phase1",{"ok":True})
    assert read_checkpoint(path)["completed"] == ["phase1"] and get_next_stage(path,["phase1","phase2"]) == "phase2"
def test_invalid_shape_rejected():
    with pytest.raises(CheckpointValidationError): validate_checkpoint({"completed": {}, "results": []})

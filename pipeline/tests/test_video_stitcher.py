import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from video_stitcher import build_stitch_plan, transition_filter
def test_cut_uses_concat(): assert transition_filter(build_stitch_plan([{"path":"a"},{"path":"b"}])) == "concat"
def test_crossfade_offsets_accumulate():
    plan=build_stitch_plan([{"path":"a","duration":3},{"path":"b","duration":4},{"path":"c","duration":2}],"crossfade",.5)
    assert plan.offsets == [2.5, 6.0]
def test_unknown_transition_rejected():
    with pytest.raises(ValueError): build_stitch_plan([{"path":"a"}],"spin")

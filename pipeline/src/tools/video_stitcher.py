"""HonCut transition planning for FFmpeg stitching."""
from dataclasses import dataclass
from typing import Any
TRANSITIONS={"cut","crossfade","fade_through_black"}
@dataclass(frozen=True)
class StitchPlan: clips:list[str]; transition:str; duration:float; offsets:list[float]
def build_stitch_plan(clips:list[dict[str,Any]],transition:str="cut",transition_duration:float=.5)->StitchPlan:
    if not clips: raise ValueError("No clips provided")
    if transition not in TRANSITIONS: raise ValueError(f"Unsupported transition: {transition}")
    if transition_duration < 0: raise ValueError("transition_duration must be non-negative")
    paths=[str(c["path"]) for c in clips]; durations=[float(c.get("duration",0)) for c in clips]
    offsets=[]; elapsed=durations[0]
    for duration in durations[1:]:
        offsets.append(round(max(0,elapsed-transition_duration),3)); elapsed += duration-transition_duration if transition!="cut" else duration
    return StitchPlan(paths,transition,transition_duration,offsets)
def transition_filter(plan:StitchPlan)->str:
    if plan.transition=="cut": return "concat"
    name="fadeblack" if plan.transition=="fade_through_black" else "fade"
    return ";".join(f"xfade=transition={name}:duration={plan.duration}:offset={offset}" for offset in plan.offsets)

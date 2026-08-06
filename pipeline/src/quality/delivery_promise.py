"""Lock and validate one of eight HonCut delivery promises."""
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

class PromiseType(Enum):
    MOTION_LED="motion_led"; SOURCE_LED="source_led"; DATA_EXPLAINER="data_explainer"; TEACHER_EXPLAINER="teacher_explainer"
    SCREEN_DEMO="screen_demo"; AVATAR_PRESENTER="avatar_presenter"; HYBRID="hybrid"; LOCALIZATION="localization"

PROMISE_RULES = {
    "motion_led": (False, True, .7), "source_led": (True, False, .3), "data_explainer": (True, False, 0),
    "teacher_explainer": (True, False, 0), "screen_demo": (True, False, 0), "avatar_presenter": (False, True, .3),
    "hybrid": (True, False, .2), "localization": (True, False, 0),
}

@dataclass
class DeliveryPromise:
    promise_type: PromiseType
    motion_required: bool
    source_required: bool
    tone_mode: str
    quality_floor: str
    approved_fallback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self); value["promise_type"] = self.promise_type.value; return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DeliveryPromise":
        return cls(PromiseType(value["promise_type"]), value.get("motion_required", False), value.get("source_required", False), value.get("tone_mode", "corporate"), value.get("quality_floor", "presentable"), value.get("approved_fallback"))

    def get_rules(self) -> dict[str, Any]:
        still, video, ratio = PROMISE_RULES[self.promise_type.value]
        return {"still_fallback_allowed": still, "requires_video_generation": video, "min_motion_ratio": ratio}

    def validate_cuts(self, cuts: list[dict[str, Any]]) -> dict[str, Any]:
        if not cuts: return {"valid": False, "violations": ["No cuts provided"], "motion_ratio": 0.0}
        slides = {"text_card", "stat_card", "chart", "bar_chart", "line_chart", "pie_chart", "kpi_grid", "comparison", "progress", "callout"}
        motion_types = {"video", "animation", "avatar"}
        motion = slide = still = 0
        for cut in cuts:
            ext = str(cut.get("source", "")).rsplit(".", 1)[-1].lower()
            kind = cut.get("type", "")
            if ext in {"mp4", "mov", "webm", "avi", "mkv"} or kind in motion_types: motion += 1
            elif kind in slides: slide += 1
            else: still += 1
        total = motion + slide + still; ratio = motion / total; rules = self.get_rules(); violations = []
        if self.motion_required and ratio < rules["min_motion_ratio"]: violations.append(f"Motion ratio {ratio:.0%} is below minimum {rules['min_motion_ratio']:.0%} for {self.promise_type.value}.")
        if not rules["still_fallback_allowed"] and slide + still > total * .5 and self.approved_fallback != "still_led": violations.append(f"{self.promise_type.value} does not allow still-led fallback; user approval is required.")
        return {"valid": not violations, "violations": violations, "motion_ratio": ratio, "motion_cuts": motion, "slide_cuts": slide, "still_cuts": still}

def classify_from_brief(pipeline_type: str, user_intent: dict[str, Any]) -> DeliveryPromise:
    defaults = {"cinematic": PromiseType.MOTION_LED, "animation": PromiseType.MOTION_LED, "animated-explainer": PromiseType.DATA_EXPLAINER, "talking-head": PromiseType.AVATAR_PRESENTER, "avatar-spokesperson": PromiseType.AVATAR_PRESENTER, "screen-demo": PromiseType.SCREEN_DEMO, "localization-dub": PromiseType.LOCALIZATION, "podcast-repurpose": PromiseType.SOURCE_LED, "clip-factory": PromiseType.SOURCE_LED}
    kind = defaults.get(pipeline_type, PromiseType.HYBRID)
    if user_intent.get("motion_required") is False and kind == PromiseType.MOTION_LED: kind = PromiseType.HYBRID
    source = user_intent.get("has_footage", False)
    if source and kind not in (PromiseType.SOURCE_LED, PromiseType.LOCALIZATION): kind = PromiseType.SOURCE_LED
    motion = user_intent.get("motion_required", kind in (PromiseType.MOTION_LED, PromiseType.AVATAR_PRESENTER))
    return DeliveryPromise(kind, motion, source, user_intent.get("tone", "corporate"), user_intent.get("quality", "presentable"))

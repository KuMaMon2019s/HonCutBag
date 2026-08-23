"""Normalized generation action unit capacity model.

The complete ``micro_actions`` screenplay ledger is never truncated.  Capacity
math instead consumes *generation action units* — a normalized count that
distinguishes what actually costs provider story-time:

  sequential    ordered plot actions that cannot run in parallel
                → 1 unit each (deduplicated across events via a shared seen set)
  simultaneous  concurrent composite motions / coordinated multi-person actions
                → merged to 1 unit per concurrent cluster
  sustained     sustained states, emotions, camera constraints
                → 0 units
  duplicate     cross-event repeats of an already-counted actionable summary
                → 0 units

Only ``sequential`` units and concurrent ``simultaneous`` clusters
consume beat capacity.  Everything else stays in the audit ledger but costs
nothing at generation time.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# ── classifier patterns ──────────────────────────────────────────────────────

# C: camera / shooting constraints — never paid provider work
_CAMERA = re.compile(
    r"镜头|摄像|摄影|拍摄|录制|机位|对焦|构图|景别|运镜|画面|空镜|摆拍|"
    r"变焦|Zoom|zoom|跟拍|视角|视频开场|视频整体|一镜到底"
)

# C: negative constraints — "don't do X" directives
_NEGATIVE = re.compile(
    r"不要|不得|禁止|避免|不能|不出现|而不是|无空镜|不突然|不露|拒绝|无静止|"
    r"不停止|不停下|不站住|不摆(?:最终)?Pose|不摆姿势|"
    r"未停止|未停下|未集体站住|未出现"
)

# C: sustained states — emotion, expression, ongoing vibe
_SUSTAINED = re.compile(
    r"持续|保持|始终|一直|不断|维持|处于|状态|氛围|情绪|笑容|眼神|自信|享受|"
    r"感染力|魅力|气质|自然|轻松|投入|表情|对口型|演唱|微笑|偶尔|"
    r"一气呵成|注意到|察觉|意识到|继续"
)

# B: simultaneous signals — concurrent, coordinated, or gradually joined work
_SIMULTANEOUS = re.compile(
    r"同时|同一时刻|同一瞬间|并行|并发|陆续|逐渐|渐渐|一起|一同|"
    r"同步|共同|协同|齐步|模仿|跟随|加入$|加入|汇聚|人群|队伍|人数|"
    r"群体|多人|全体|波浪|扩散|传播|从.{1,8}(?:加入|出现)"
)

# Source text may explicitly define several entries as one concurrent compound
# action. This deliberately uses domain-neutral temporal language: the same
# contract works for performance, sport, assembly, or crowd motion instead of
# recognizing one regression screenplay's nouns.
_COMPOSITE_MOTION_CUE = re.compile(
    r"复合(?:动作|运动|过程|表演|律动)?|"
    r"同一(?:时刻|瞬间)|同时发生|并行完成|"
    r"融为(?:一体|一个整体|一段|同一动作)|"
    r"融合为|合成为|组合成|作为一个整体|"
    r"(?:不是|并非|而非).{0,30}(?:逐个|依次|顺序|分离|清单)"
)
_TEMPORAL_PROGRESSION_CUE = re.compile(
    r"一开始|随后|然后|接着|逐步|逐渐|最终|最后|先.{0,30}再"
)

# A: sequential signals — ordered plot steps that cannot run in parallel
_SEQUENTIAL = re.compile(
    r"然后|接着|随后|之后|先.{0,6}再|再.{0,4}(?:做|完成)|完成|挥拳|肘击|膝击|击|"
    r"踢|夺|缴械|抓|推开|穿过|穿门|转向|转身|倒地|站起|开口|说道|喊道|递|"
    r"接住|稳定|避让|走向|点头"
)

# Some authored actions legitimately mention the frame/camera while describing
# paid actor motion.  These unambiguous verbs must not be zeroed merely because
# the same sentence also contains ``镜头`` or ``画面``.  Ambiguous camera verbs
# such as ``转向`` deliberately stay out so ``镜头转向角色`` remains a camera
# constraint rather than an actor action.
_ACTOR_ACTION_IN_CAMERA_TEXT = re.compile(
    r"挥拳|肘击|膝击|踢|夺|缴械|推开|穿过|穿门|倒地|站起|开口|说道|喊道|"
    r"递给|接住|避让|走向|冲进"
)
_ACTOR_CONCURRENCY_IN_CAMERA_TEXT = re.compile(
    r"同时|一起|一同|同步|齐舞|群舞|多人|全体|人群|队伍"
)


def classify_micro_action(text: str) -> str:
    """Return one of: sequential / simultaneous / sustained."""
    text = str(text).strip()
    if not text:
        return "sustained"
    if _NEGATIVE.search(text):
        return "sustained"
    if _CAMERA.search(text):
        if _ACTOR_ACTION_IN_CAMERA_TEXT.search(text):
            return (
                "simultaneous"
                if _ACTOR_CONCURRENCY_IN_CAMERA_TEXT.search(text)
                else "sequential"
            )
        return "sustained"
    if _SEQUENTIAL.search(text) and not _SIMULTANEOUS.search(text):
        return "sequential"
    if _SIMULTANEOUS.search(text):
        return "simultaneous"
    if _SUSTAINED.search(text):
        return "sustained"
    return "sequential"  # conservative default: costs a unit


def _event_motion_evidence(event: dict[str, Any]) -> str:
    return " ".join(
        str(event.get(field) or "")
        for field in ("what", "source_excerpt")
    )


def _has_local_composite_motion(event: dict[str, Any]) -> bool:
    evidence = _event_motion_evidence(event)
    return bool(
        _COMPOSITE_MOTION_CUE.search(evidence)
        and not _TEMPORAL_PROGRESSION_CUE.search(evidence)
    )


def event_uses_composite_motion(event: dict[str, Any]) -> bool:
    """Whether the source says this event is one concurrent compound motion.

    Only source evidence and the event summary participate.  ``visual`` is
    deliberately excluded because an extractor may mention background motion
    while the event itself describes a real ordered progression.
    """

    if _has_local_composite_motion(event):
        return True
    declared_mode = str(event.get("generation_motion_mode") or "").strip().lower()
    return declared_mode == "composite"


def annotate_event_motion_modes(events: list[dict[str, Any]]) -> bool:
    """Normalize explicit per-event semantics with a conservative fallback.

    The extractor owns this decision. For legacy artifacts without the field,
    only an unambiguous local concurrent-action statement is inferred; a
    document-wide keyword rule must never rewrite unrelated events.
    """

    has_composite_contract = False
    for event in events:
        actions = event.get("micro_actions") or []
        if not actions:
            continue
        evidence = _event_motion_evidence(event)
        local_contract = _has_local_composite_motion(event)
        if local_contract:
            event["generation_motion_mode"] = "composite"
            event["generation_motion_mode_reason"] = (
                "source explicitly defines this event as concurrent compound motion; "
                "source evidence overrides a conflicting model label"
            )
            has_composite_contract = True
        else:
            declared_mode = str(
                event.get("generation_motion_mode") or ""
            ).strip().lower()
            event["generation_motion_mode"] = (
                declared_mode if declared_mode in {"composite", "atomic"} else "atomic"
            )
            event["generation_motion_mode_reason"] = (
                "event extraction contract"
                if declared_mode in {"composite", "atomic"}
                else (
                    "source contains ordered state progression"
                    if _TEMPORAL_PROGRESSION_CUE.search(evidence)
                    else "no compound-motion source contract"
                )
            )
            has_composite_contract = (
                has_composite_contract
                or event["generation_motion_mode"] == "composite"
            )
    return has_composite_contract


def _dedupe_key(text: str) -> str:
    """Return a stable key for repeated screenplay summaries.

    This intentionally performs only textual normalization.  Semantic fuzzy
    matching would risk deleting two genuinely different plot actions from the
    capacity ledger.
    """

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"[\s，,。.!！？；;：:'\"“”‘’、（）()【】\[\]—…-]+", "", normalized)


def normalize_action_units(
    actions: list[str],
    *,
    seen: set[str] | None = None,
    composite_motion: bool = False,
) -> dict[str, Any]:
    """Classify every micro_action and count generation action units.

    Args:
        actions: the complete micro_actions ledger (never modified).
        seen: optional shared set for cross-event actionable-summary deduplication.
              Pass the same set across events in one gate pass.

    Returns:
        dict with keys: ledger (unchanged), categories (parallel list),
        units (int), sequential (int), simultaneous_clusters (int).
    """
    if seen is None:
        seen = set()
    # ``seen`` represents actions completed by *earlier events*.  Snapshot it
    # before this event so an intentional repeat inside the same event remains
    # a distinct sequential unit.  Publish all observed keys only after the
    # complete event has been normalized for the next event's dedupe pass.
    prior_event_keys = set(seen)
    observed_actionable_keys: set[str] = set()
    categories: list[str] = []
    generation_units: list[dict[str, Any]] = []
    sequential_units = 0
    simultaneous_clusters = 0
    active_simultaneous: dict[str, Any] | None = None

    for index, raw in enumerate(actions):
        text = str(raw).strip()
        category = classify_micro_action(text)
        if composite_motion and category == "sequential":
            category = "simultaneous"
        key = _dedupe_key(text)
        if category in {"sequential", "simultaneous"} and key in prior_event_keys:
            categories.append("duplicate")
            if category == "sequential":
                active_simultaneous = None
            continue
        if category in {"sequential", "simultaneous"}:
            observed_actionable_keys.add(key)

        if category == "sequential":
            active_simultaneous = None
            sequential_units += 1
            categories.append("sequential")
            generation_units.append({
                "kind": "sequential",
                "actions": [text],
                "ledger_indexes": [index],
            })
        elif category == "simultaneous":
            categories.append("simultaneous")
            if active_simultaneous is None:
                simultaneous_clusters += 1
                active_simultaneous = {
                    "kind": "simultaneous",
                    "actions": [],
                    "ledger_indexes": [],
                }
                generation_units.append(active_simultaneous)
            active_simultaneous["actions"].append(text)
            active_simultaneous["ledger_indexes"].append(index)
        else:
            categories.append("sustained")

    seen.update(observed_actionable_keys)

    for position, unit in enumerate(generation_units, 1):
        unit["unit_id"] = f"GAU{position:03d}"

    units = len(generation_units)
    return {
        "ledger": list(actions),
        "categories": categories,
        "units": units,
        "generation_action_units": generation_units,
        "sequential": sequential_units,
        "simultaneous_clusters": simultaneous_clusters,
        "motion_mode": "composite" if composite_motion else "atomic",
    }


def normalize_event_action_units(
    event: dict[str, Any],
    *,
    actions: list[str] | None = None,
    seen: set[str] | None = None,
) -> dict[str, Any]:
    """Normalize one event using its source-authored choreography semantics."""

    event_actions = (
        event.get("micro_actions") or []
        if actions is None
        else actions
    )
    if isinstance(event_actions, str):
        event_actions = [event_actions]
    return normalize_action_units(
        [str(action).strip() for action in event_actions if str(action).strip()],
        seen=seen,
        composite_motion=event_uses_composite_motion(event),
    )


def normalized_action_unit_count(
    actions: list[str],
    *,
    seen: set[str] | None = None,
) -> int:
    """Convenience: return only the unit count."""
    return normalize_action_units(actions, seen=seen)["units"]

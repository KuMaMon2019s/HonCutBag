"""Normalized generation action unit capacity model.

The complete ``micro_actions`` screenplay ledger is never truncated.  Capacity
math instead consumes *generation action units* — a normalized count that
distinguishes what actually costs provider story-time:

  sequential    ordered plot actions that cannot run in parallel
                → 1 unit each (deduplicated across events via a shared seen set)
  simultaneous  concurrent composite motions / group grooves / multi-person joins
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
    r"镜头|摄像|拍摄|机位|对焦|构图|景别|运镜|摄影师|iPhone|iphone|画面|空镜|摆拍|"
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
    r"感染力|魅力|气质|自然|轻松|投入|表情|对口型|唱歌|微笑|偶尔|随音乐|"
    r"响应音乐|跟随音乐|一气呵成|注意到|察觉|意识到|"
    r"音乐(?:进入|播放至|达到)|视频录制(?:进入|结束)|录制(?:进入|结束)|"
    r"继续(?:跳舞|舞蹈|律动|向前移动|前进)"
)

# B: simultaneous signals — multi-person concurrent / group groove / gradual merge
_SIMULTANEOUS = re.compile(
    r"同时|陆续|逐渐|渐渐|一起|一同|同步|齐舞|模仿|跟随|感染|加入舞蹈|加入$|加入|"
    r"汇聚|人群|队伍|人数|群体|群舞|背景舞者|波浪|扩散|传播|Groove|groove|"
    r"律动|隔离|Isolation|绕臂|挥臂|肩.{0,3}胸|胸.{0,3}胯|"
    r"身体(?:波浪|响应|协调|整体)|从.{1,8}(?:加入|出现)|有人.{0,10}(?:加入|Groove|跳)|"
    r"形成(?:松散|小型|大型|移动|Flash|flash|队形|队伍|包围|阵型)"
)

# Source-authored choreography can explicitly state that several named body
# movements happen as one compound groove rather than as ordered plot steps.
# The event context is required here: the same verbs in a fight remain
# sequential and must never be collapsed merely because they are adjacent.
_DANCE_CONTEXT = re.compile(
    r"舞蹈|跳舞|舞步|舞者|群舞|街舞|Groove|groove|律动|"
    r"Hip-?Hop|hip-?hop|Waacking|waacking|K-?Pop|k-?pop"
)
_COMPOSITE_MOTION_CUE = re.compile(
    r"复合(?:律动|舞蹈|舞步|动作)|"
    r"连贯(?:的)?(?:Groove|groove|律动|舞蹈)|"
    r"(?:所有|全部|整段).{0,30}(?:舞蹈|舞步|动作).{0,30}(?:融为|融入)|"
    r"(?:不是|并非|而非).{0,30}(?:逐个|分离动作|动作清单)|"
    r"每个瞬间.{0,30}(?:连贯|复合)(?:律动|舞蹈|动作)"
)
_GLOBAL_COMPOSITE_DANCE_CUE = re.compile(
    r"(?:剧本中)?(?:所有|全部).{0,30}舞蹈描述.{0,60}"
    r"每个瞬间.{0,60}复合律动.{0,80}"
    r"(?:而非|不是|并非).{0,60}(?:逐个|分离动作|动作清单)"
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
        _DANCE_CONTEXT.search(evidence)
        and _COMPOSITE_MOTION_CUE.search(evidence)
    )


def event_uses_composite_motion(event: dict[str, Any]) -> bool:
    """Whether the source says this dance event is one concurrent motion.

    Only source evidence and the event summary participate.  ``visual`` is
    deliberately excluded because an extractor may mention a protagonist's
    background groove while the event itself describes a real progression
    such as notice → respond → join.
    """

    declared_mode = str(event.get("generation_motion_mode") or "").strip().lower()
    if declared_mode in {"composite", "atomic"}:
        return declared_mode == "composite"
    return _has_local_composite_motion(event)


def annotate_event_motion_modes(events: list[dict[str, Any]]) -> bool:
    """Persist a document-wide compound-dance contract on eligible events.

    A global source instruction can govern later segments even though the LLM
    extracts them independently.  Events with explicit temporal progression
    remain atomic; their notice → respond → join phases are real story changes,
    not a list of simultaneous dance vocabulary.
    """

    document_evidence = "\n".join(
        _event_motion_evidence(event)
        for event in events
    )
    has_global_contract = bool(
        _GLOBAL_COMPOSITE_DANCE_CUE.search(document_evidence)
    )
    for event in events:
        actions = event.get("micro_actions") or []
        if not actions:
            continue
        evidence = _event_motion_evidence(event)
        local_contract = _has_local_composite_motion(event)
        inherited_contract = bool(
            has_global_contract
            and _DANCE_CONTEXT.search(evidence)
            and not _TEMPORAL_PROGRESSION_CUE.search(evidence)
        )
        if local_contract or inherited_contract:
            event["generation_motion_mode"] = "composite"
            event["generation_motion_mode_reason"] = (
                "source explicitly defines this choreography as concurrent compound motion"
                if local_contract
                else "document-wide compound-dance contract"
            )
        else:
            event["generation_motion_mode"] = "atomic"
            event["generation_motion_mode_reason"] = (
                "source contains ordered state progression"
                if _TEMPORAL_PROGRESSION_CUE.search(evidence)
                else "no compound-motion source contract"
            )
    return has_global_contract


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

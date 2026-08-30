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

import hashlib
import json
import re
import unicodedata
from typing import Any


ACTION_TIMELINE_SCHEMA = "honcut.action-timeline.v1"
SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA = (
    "honcut.source-indexed-screenplay-rewrite.v1"
)

_ACTION_KINDS = {
    "locomotion",
    "attack",
    "defense",
    "control",
    "impact",
    "state_change",
    "environment_effect",
    "sustained",
}
_TEMPORAL_RELATIONS = {
    "root",
    "after",
    "overlap",
    "reaction_overlap",
    "effect_of",
    "sustained_during",
}
_PACES = {"fast", "normal", "slow"}
_ZERO_TEMPORAL_RELATIONS = {"effect_of", "sustained_during"}
_OVERLAP_RELATIONS = {"overlap", "reaction_overlap"}
_PACE_WEIGHTS = {"fast": 1, "normal": 2, "slow": 3}
_ENSEMBLE_SOURCE_CUE = re.compile(
    r"同时|同一时刻|同一瞬间|同步|协同|共同|一起|一同|全体|"
    r"(?<![第十])[两二三四五六七八九十]+(?:名|人|位|个)|"
    r"(?<!第)\d+(?:名|人|位|个)"
)

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
    r"一开始|随后|然后|接着|逐步|逐渐|最终|"
    r"最后(?!\s*(?:一|1)?\s*(?:名|位|只|辆|台|艘|架|支|队|组|批|部|"
    r"座|间|枚|把|套|双|对|排|列|头|匹))|"
    r"先.{0,30}再"
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


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(
        str(item).strip() for item in value if str(item).strip()
    ))


def _choreography_performers_by_action(event: dict[str, Any]) -> dict[int, list[str]]:
    performers: dict[int, list[str]] = {}
    for raw in event.get("body_action_choreography") or []:
        if not isinstance(raw, dict):
            continue
        action_index = raw.get("micro_action_index")
        performer = str(raw.get("performer") or "").strip()
        if (
            isinstance(action_index, int)
            and not isinstance(action_index, bool)
            and action_index >= 1
            and performer
        ):
            performers.setdefault(action_index, []).append(performer)
    return {
        action_index: list(dict.fromkeys(values))
        for action_index, values in performers.items()
    }


def _normalized_temporal_relations(
    event: dict[str, Any],
    actions: list[str],
) -> list[dict[str, Any]]:
    raw_relations = event.get("action_temporal_relations") or []
    if not raw_relations:
        return []
    if not isinstance(raw_relations, list):
        raise ValueError("action_temporal_relations must be a list")
    choreography_performers = _choreography_performers_by_action(event)
    normalized: list[dict[str, Any]] = []
    observed_indexes: set[int] = set()
    for raw in raw_relations:
        if not isinstance(raw, dict):
            raise ValueError("action temporal relation entries must be objects")
        action_index = raw.get("micro_action_index")
        if (
            isinstance(action_index, bool)
            or not isinstance(action_index, int)
            or not 1 <= action_index <= len(actions)
            or action_index in observed_indexes
        ):
            raise ValueError(
                "action temporal relations must cover unique valid micro-action indexes"
            )
        observed_indexes.add(action_index)
        action_kind = str(raw.get("action_kind") or "").strip().lower()
        temporal_relation = str(
            raw.get("temporal_relation") or ""
        ).strip().lower()
        pace = str(raw.get("pace") or "normal").strip().lower()
        if action_kind not in _ACTION_KINDS:
            raise ValueError(f"micro action {action_index} has invalid action_kind")
        if temporal_relation not in _TEMPORAL_RELATIONS:
            raise ValueError(
                f"micro action {action_index} has invalid temporal_relation"
            )
        if pace not in _PACES:
            raise ValueError(f"micro action {action_index} has invalid pace")
        references = raw.get("reference_action_indexes") or []
        if not isinstance(references, list) or any(
            isinstance(reference, bool)
            or not isinstance(reference, int)
            or not 1 <= reference < action_index
            for reference in references
        ):
            raise ValueError(
                f"micro action {action_index} has invalid temporal references"
            )
        references = list(dict.fromkeys(references))
        if temporal_relation == "root" and references:
            raise ValueError(f"root micro action {action_index} cannot have references")
        if temporal_relation != "root" and not references:
            raise ValueError(
                f"micro action {action_index} temporal relation requires a reference"
            )
        if temporal_relation in _ZERO_TEMPORAL_RELATIONS and action_kind not in {
            "impact",
            "state_change",
            "environment_effect",
            "sustained",
        }:
            raise ValueError(
                f"micro action {action_index} cannot be a zero-time actor action: "
                f"{actions[action_index - 1]}"
            )
        performers = _clean_string_list(raw.get("performers"))
        if not performers:
            performers = choreography_performers.get(action_index, [])
        if temporal_relation in _ZERO_TEMPORAL_RELATIONS and performers:
            raise ValueError(
                "passive effect/sustained actions cannot claim an independent "
                "performer: "
                f"micro_action_index={action_index}, "
                f"action_kind={action_kind}, temporal_relation={temporal_relation}, "
                f"performers={performers}, action={actions[action_index - 1]}"
            )
        if (
            action_kind in {"environment_effect", "sustained"}
            and temporal_relation in _OVERLAP_RELATIONS
        ):
            raise ValueError(
                "passive environment/sustained actions must use effect_of or "
                "sustained_during, never actor overlap: "
                f"micro_action_index={action_index}, "
                f"action_kind={action_kind}, temporal_relation={temporal_relation}, "
                f"performers={performers}, action={actions[action_index - 1]}"
            )
        targets = _clean_string_list(raw.get("targets"))
        normalized.append({
            "micro_action_index": action_index,
            "performers": performers,
            "targets": targets,
            "action_kind": action_kind,
            "temporal_relation": temporal_relation,
            "reference_action_indexes": references,
            "ensemble_id": str(raw.get("ensemble_id") or "").strip(),
            "pace": pace,
            "state_reads": _clean_string_list(raw.get("state_reads")),
            "state_writes": _clean_string_list(raw.get("state_writes")),
        })
    if observed_indexes != set(range(1, len(actions) + 1)):
        raise ValueError(
            "action temporal relations must cover every micro action exactly once"
        )
    normalized.sort(key=lambda item: item["micro_action_index"])
    return normalized


def _validate_parallel_relation(
    event: dict[str, Any],
    relation: dict[str, Any],
    relations_by_index: dict[int, dict[str, Any]],
) -> None:
    temporal_relation = relation["temporal_relation"]
    if temporal_relation not in _OVERLAP_RELATIONS:
        return
    performers = set(relation["performers"])
    referenced = [
        relations_by_index[index]
        for index in relation["reference_action_indexes"]
    ]
    referenced_performers = {
        performer
        for item in referenced
        for performer in item["performers"]
    }
    if (
        performers
        and performers & referenced_performers
        and not event_uses_composite_motion(event)
    ):
        raise ValueError(
            "one performer cannot execute overlapping independent actions "
            "without an explicit composite-motion contract: "
            f"micro_action_index={relation['micro_action_index']}, "
            f"action_kind={relation['action_kind']}, "
            f"temporal_relation={temporal_relation}, "
            f"performers={sorted(performers)}, references="
            f"{relation['reference_action_indexes']}, "
            f"action={event.get('micro_actions', [])[relation['micro_action_index'] - 1]}"
        )
    if temporal_relation == "reaction_overlap":
        current_kind = relation["action_kind"]
        referenced_kinds = {item["action_kind"] for item in referenced}
        complementary = (
            current_kind in {"defense", "control"}
            and bool(referenced_kinds & {"attack", "impact"})
        ) or (
            current_kind in {"attack", "impact"}
            and bool(referenced_kinds & {"defense", "control"})
        )
        if not complementary:
            raise ValueError(
                "reaction_overlap requires complementary attack/defense semantics"
            )


def _validate_slice_contributions(
    event: dict[str, Any],
    contribution_indexes: list[int],
    relations_by_index: dict[int, dict[str, Any]],
) -> None:
    """Validate transitive performer conflicts and explicit ensembles."""
    performer_owners: dict[str, int] = {}
    for action_index in contribution_indexes:
        relation = relations_by_index[action_index]
        for performer in relation["performers"]:
            previous = performer_owners.get(performer)
            if previous is not None and not event_uses_composite_motion(event):
                raise ValueError(
                    "one performer cannot contribute multiple independent motions "
                    "to one temporal slice: "
                    f"performer={performer}, actions={[previous, action_index]}"
                )
            performer_owners[performer] = action_index

    ensemble_groups: dict[str, list[dict[str, Any]]] = {}
    for action_index in contribution_indexes:
        relation = relations_by_index[action_index]
        ensemble_id = relation.get("ensemble_id")
        if ensemble_id:
            ensemble_groups.setdefault(str(ensemble_id), []).append(relation)
    evidence = " ".join(
        str(event.get(field) or "")
        for field in ("source_excerpt", "what")
    )
    for ensemble_id, relations in ensemble_groups.items():
        performers = [
            performer
            for relation in relations
            for performer in relation["performers"]
        ]
        action_kinds = {relation["action_kind"] for relation in relations}
        if (
            len(performers) < 2
            or len(set(performers)) != len(performers)
            or len(action_kinds) != 1
            or not _ENSEMBLE_SOURCE_CUE.search(evidence)
        ):
            raise ValueError(
                "ensemble contribution requires explicit coordinated source "
                "evidence, distinct performers, and one action kind: "
                f"ensemble_id={ensemble_id}, relation_count={len(relations)}, "
                f"performers={performers}, action_kinds={sorted(action_kinds)}, "
                f"source_coordination={bool(_ENSEMBLE_SOURCE_CUE.search(evidence))}"
            )


def build_event_action_timeline(
    event: dict[str, Any],
    *,
    actions: list[str] | None = None,
    seen: set[str] | None = None,
    semantic_qa_enabled: bool | None = None,
) -> dict[str, Any] | None:
    """Build a deterministic per-event causal timeline from strict relations.

    The model proposes local semantics, but code owns validation and scheduling.
    Source actions are never removed: effects and sustained facts remain attached
    to their causal slice while consuming no additional temporal capacity.
    """

    if semantic_qa_enabled is None:
        stored_semantic_qa = event.get("semantic_action_qa_enabled", False)
        if not isinstance(stored_semantic_qa, bool):
            raise ValueError("semantic_action_qa_enabled must be a boolean")
        semantic_qa_enabled = stored_semantic_qa
    elif not isinstance(semantic_qa_enabled, bool):
        raise ValueError("semantic_qa_enabled must be a boolean")

    event_actions = event.get("micro_actions") or [] if actions is None else actions
    if isinstance(event_actions, str):
        event_actions = [event_actions]
    normalized_actions = [
        str(action).strip() for action in event_actions if str(action).strip()
    ]
    if not normalized_actions:
        return {
            "schema": ACTION_TIMELINE_SCHEMA,
            "semantic_qa": {
                "enabled": semantic_qa_enabled,
                "verdict": (
                    "pass" if semantic_qa_enabled else "diagnostic_only"
                ),
                "finding_count": 0,
                "findings": [],
            },
            "source_micro_actions": [],
            "actions": [],
            "slices": [],
            "source_micro_actions_sha256": hashlib.sha256(b"[]").hexdigest(),
            "temporal_slice_count": 0,
            "maximum_motion_load": 0,
        }
    relations = _normalized_temporal_relations(event, normalized_actions)
    if not relations:
        return None
    relations_by_index = {
        relation["micro_action_index"]: relation for relation in relations
    }
    semantic_qa_findings: list[dict[str, Any]] = []
    for relation in relations:
        try:
            _validate_parallel_relation(event, relation, relations_by_index)
        except ValueError as exc:
            if semantic_qa_enabled:
                raise
            semantic_qa_findings.append({
                "category": "parallel_relation",
                "micro_action_index": relation["micro_action_index"],
                "message": str(exc),
            })

    prior_event_keys = set(seen or set())
    observed_keys: set[str] = set()
    slices: list[dict[str, Any]] = []
    slice_by_action_index: dict[int, int] = {}
    for relation in relations:
        action_index = relation["micro_action_index"]
        temporal_relation = relation["temporal_relation"]
        if temporal_relation in _OVERLAP_RELATIONS | _ZERO_TEMPORAL_RELATIONS:
            referenced_slices = {
                slice_by_action_index[index]
                for index in relation["reference_action_indexes"]
            }
            if len(referenced_slices) != 1:
                raise ValueError(
                    f"micro action {action_index} must reference one temporal slice: "
                    f"{normalized_actions[action_index - 1]}; references="
                    f"{relation['reference_action_indexes']} map to slices="
                    f"{sorted(referenced_slices)}"
                )
            slice_index = next(iter(referenced_slices))
        else:
            # A root or explicit after relation starts a new globally ordered
            # slice. Parallel branches must be declared as overlap instead of
            # being inferred from actor count alone.
            slice_index = len(slices)
            slices.append({
                "contribution_action_indexes": [],
                "effect_action_indexes": [],
                "sustained_action_indexes": [],
                "duplicate_action_indexes": [],
                "performers": [],
                "targets": [],
                "pace_weight": 1,
                "state_reads": [],
                "state_writes": [],
            })
        if slice_index >= len(slices):
            raise ValueError("temporal relation referenced an unmaterialized slice")
        slice_by_action_index[action_index] = slice_index
        target_slice = slices[slice_index]
        if temporal_relation == "effect_of":
            target_slice["effect_action_indexes"].append(action_index)
        elif temporal_relation == "sustained_during":
            target_slice["sustained_action_indexes"].append(action_index)
        else:
            key = _dedupe_key(normalized_actions[action_index - 1])
            if key not in prior_event_keys:
                target_slice["contribution_action_indexes"].append(action_index)
                observed_keys.add(key)
            else:
                target_slice["duplicate_action_indexes"].append(action_index)
        target_slice["performers"] = list(dict.fromkeys(
            target_slice["performers"] + relation["performers"]
        ))
        target_slice["targets"] = list(dict.fromkeys(
            target_slice["targets"] + relation["targets"]
        ))
        target_slice["pace_weight"] = max(
            target_slice["pace_weight"],
            _PACE_WEIGHTS[relation["pace"]],
        )
        target_slice["state_reads"] = list(dict.fromkeys(
            target_slice["state_reads"] + relation["state_reads"]
        ))
        target_slice["state_writes"] = list(dict.fromkeys(
            target_slice["state_writes"] + relation["state_writes"]
        ))

    if seen is not None:
        seen.update(observed_keys)
    serialized_slices: list[dict[str, Any]] = []
    for slice_index, raw_slice in enumerate(slices, 1):
        all_indexes = sorted(
            raw_slice["contribution_action_indexes"]
            + raw_slice["effect_action_indexes"]
            + raw_slice["sustained_action_indexes"]
            + raw_slice["duplicate_action_indexes"]
        )
        if not all_indexes:
            continue
        contributions = [
            relations_by_index[index]
            for index in raw_slice["contribution_action_indexes"]
        ]
        slice_semantics_valid = True
        try:
            _validate_slice_contributions(
                event,
                list(raw_slice["contribution_action_indexes"]),
                relations_by_index,
            )
        except ValueError as exc:
            if semantic_qa_enabled:
                raise
            slice_semantics_valid = False
            semantic_qa_findings.append({
                "category": "slice_contributions",
                "slice_id": f"TS{slice_index:03d}",
                "message": str(exc),
            })
        motion_contribution_keys = {
            (
                "ensemble",
                str(relation.get("ensemble_id")),
            )
            if relation.get("ensemble_id") and slice_semantics_valid
            else ("action", str(relation["micro_action_index"]))
            for relation in contributions
        }
        serialized_slices.append({
            "slice_id": f"TS{slice_index:03d}",
            "source_micro_action_indexes": all_indexes,
            "source_actions": [
                normalized_actions[index - 1] for index in all_indexes
            ],
            "contribution_action_indexes": list(
                raw_slice["contribution_action_indexes"]
            ),
            "effect_action_indexes": list(raw_slice["effect_action_indexes"]),
            "sustained_action_indexes": list(
                raw_slice["sustained_action_indexes"]
            ),
            "duplicate_action_indexes": list(
                raw_slice["duplicate_action_indexes"]
            ),
            "contributions": [dict(item) for item in contributions],
            "performers": list(raw_slice["performers"]),
            "targets": list(raw_slice["targets"]),
            "motion_load": len(motion_contribution_keys),
            "pace_weight": int(raw_slice["pace_weight"]),
            "state_reads": list(raw_slice["state_reads"]),
            "state_writes": list(raw_slice["state_writes"]),
            "start_state": (
                str(event.get("start_state") or "").strip()
                if slice_index == 1
                else ""
            ),
            "end_state": (
                str(event.get("end_state") or "").strip()
                if slice_index == len(slices)
                else f"已完成动作：{normalized_actions[all_indexes[-1] - 1]}"
            ),
        })
    source_hash = hashlib.sha256(
        json.dumps(
            normalized_actions,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": ACTION_TIMELINE_SCHEMA,
        "semantic_qa": {
            "enabled": semantic_qa_enabled,
            "verdict": (
                "pass"
                if semantic_qa_enabled
                else "diagnostic_only"
            ),
            "finding_count": len(semantic_qa_findings),
            "findings": semantic_qa_findings,
        },
        "source_micro_actions": normalized_actions,
        "actions": [dict(relation) for relation in relations],
        "slices": serialized_slices,
        "source_micro_actions_sha256": source_hash,
        "temporal_slice_count": sum(
            1 for item in serialized_slices if item["motion_load"] > 0
        ),
        "maximum_motion_load": max(
            (item["motion_load"] for item in serialized_slices),
            default=0,
        ),
    }


def build_action_timeline(
    events: list[dict[str, Any]],
    *,
    max_motion_contributions_per_slice: int,
) -> dict[str, Any]:
    """Build the canonical source timeline used by planning and audit."""

    if max_motion_contributions_per_slice < 1:
        raise ValueError("motion contribution capacity must be positive")
    event_records: list[dict[str, Any]] = []
    timeline_slices: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event_id, event in enumerate(events, 1):
        timeline = build_event_action_timeline(event, seen=seen)
        if timeline is None:
            normalized = normalize_event_action_units(event, seen=seen)
            legacy_slices = []
            for unit_index, unit in enumerate(
                normalized["generation_action_units"],
                1,
            ):
                legacy_slices.append({
                    "slice_id": f"TS{unit_index:03d}",
                    "source_micro_action_indexes": [
                        int(index) + 1 for index in unit.get("ledger_indexes", [])
                    ],
                    "source_actions": list(unit.get("actions") or []),
                    "contribution_action_indexes": [
                        int(index) + 1 for index in unit.get("ledger_indexes", [])
                    ],
                    "effect_action_indexes": [],
                    "sustained_action_indexes": [],
                    "duplicate_action_indexes": [],
                    "contributions": [],
                    "performers": [],
                    "targets": [],
                    "motion_load": 1,
                    "pace_weight": 2,
                    "state_reads": [],
                    "state_writes": [],
                    "start_state": (
                        str(event.get("start_state") or "").strip()
                        if unit_index == 1 else ""
                    ),
                    "end_state": (
                        str(event.get("end_state") or "").strip()
                        if unit_index == normalized["units"] else ""
                    ),
                })
            timeline = {
                "schema": ACTION_TIMELINE_SCHEMA,
                "source_micro_actions": list(normalized["ledger"]),
                "actions": [],
                "slices": legacy_slices,
                "source_micro_actions_sha256": hashlib.sha256(
                    json.dumps(
                        normalized["ledger"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "temporal_slice_count": normalized["units"],
                "maximum_motion_load": 1 if normalized["units"] else 0,
                "legacy_serial_fallback": True,
            }
        global_slice_ids = []
        for local_slice in timeline["slices"]:
            serialized = dict(local_slice)
            serialized["source_event_id"] = event_id
            serialized["local_slice_id"] = local_slice["slice_id"]
            serialized["slice_id"] = f"TS{len(timeline_slices) + 1:03d}"
            if serialized["motion_load"] > max_motion_contributions_per_slice:
                serialized["motion_capacity_status"] = "rewrite_required"
            else:
                serialized["motion_capacity_status"] = "fits_model"
            timeline_slices.append(serialized)
            global_slice_ids.append(serialized["slice_id"])
        event_records.append({
            "source_event_id": event_id,
            "source_micro_action_count": len(timeline["source_micro_actions"]),
            "source_micro_actions_sha256": timeline[
                "source_micro_actions_sha256"
            ],
            "temporal_slice_ids": global_slice_ids,
            "temporal_slice_count": int(timeline["temporal_slice_count"]),
            "maximum_motion_load": int(timeline["maximum_motion_load"]),
            "legacy_serial_fallback": bool(
                timeline.get("legacy_serial_fallback")
            ),
        })
    source_hash = hashlib.sha256(
        json.dumps(
            events,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": ACTION_TIMELINE_SCHEMA,
        "source_events_sha256": source_hash,
        "max_motion_contributions_per_slice": (
            max_motion_contributions_per_slice
        ),
        "source_micro_action_count": sum(
            record["source_micro_action_count"] for record in event_records
        ),
        "temporal_slice_count": sum(
            record["temporal_slice_count"] for record in event_records
        ),
        "maximum_motion_load": max(
            (record["maximum_motion_load"] for record in event_records),
            default=0,
        ),
        "events": event_records,
        "slices": timeline_slices,
    }


def _provider_execution_units(
    normalized: dict[str, Any],
    *,
    max_motion_contributions_per_slice: int,
) -> dict[str, Any]:
    """Stage overloaded semantic slices without changing their source truth.

    ``honcut.action-timeline.v1`` is a story-time ledger.  A real instant may
    contain more independent motion than one Provider request can reliably
    execute.  Production therefore creates adjacent execution subslices that
    share one semantic slice id.  Source facts are partitioned exactly once;
    the semantic simultaneity, original load, and deterministic staging order
    remain auditable instead of being rewritten into false source chronology.
    """

    if max_motion_contributions_per_slice < 1:
        raise ValueError("motion contribution capacity must be positive")
    semantic_units = normalized.get("generation_action_units") or []
    timeline = normalized.get("action_timeline")
    if not isinstance(timeline, dict) or not semantic_units:
        return normalized
    relations = timeline.get("actions") or []
    execution_units: list[dict[str, Any]] = []
    staged_semantic_slice_count = 0
    for semantic_order, raw_unit in enumerate(semantic_units, 1):
        unit = dict(raw_unit)
        semantic_load = int(unit.get("motion_load") or 1)
        semantic_id = str(
            unit.get("temporal_slice_id") or f"TS{semantic_order:03d}"
        )
        contribution_indexes = [
            int(index) for index in unit.get("contribution_ledger_indexes") or []
        ]
        if semantic_load <= max_motion_contributions_per_slice:
            unit.update({
                "semantic_temporal_slice_id": semantic_id,
                "semantic_motion_load": semantic_load,
                "execution_subslice_id": f"{semantic_id}:E01",
                "execution_subslice_order": 1,
                "execution_subslice_count": 1,
                "provider_capacity_staging": "not_required",
            })
            execution_units.append(unit)
            continue

        # Ensemble members share one model motion contribution.  A reactive
        # attack/defense pair should remain in one execution pass whenever the
        # profile can carry it, so union reaction-linked contribution keys
        # before deterministic capacity packing.
        contribution_keys: list[tuple[str, str]] = []
        action_key: dict[int, tuple[str, str]] = {}
        for action_index in contribution_indexes:
            relation = relations[action_index]
            ensemble_id = str(relation.get("ensemble_id") or "").strip()
            key = (
                ("ensemble", ensemble_id)
                if ensemble_id
                else ("action", str(action_index))
            )
            action_key[action_index] = key
            if key not in contribution_keys:
                contribution_keys.append(key)

        parent = {key: key for key in contribution_keys}

        def find(key: tuple[str, str]) -> tuple[str, str]:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(left: tuple[str, str], right: tuple[str, str]) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for action_index in contribution_indexes:
            relation = relations[action_index]
            if relation.get("temporal_relation") != "reaction_overlap":
                continue
            for raw_reference in relation.get("reference_action_indexes") or []:
                reference_index = int(raw_reference) - 1
                if reference_index in action_key:
                    union(action_key[action_index], action_key[reference_index])

        clusters: list[list[tuple[str, str]]] = []
        cluster_by_root: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for key in contribution_keys:
            root = find(key)
            if root not in cluster_by_root:
                cluster_by_root[root] = []
                clusters.append(cluster_by_root[root])
            cluster_by_root[root].append(key)

        packed_keys: list[list[tuple[str, str]]] = []
        for cluster in clusters:
            # A single semantic interaction may itself exceed the model.  Keep
            # its shared semantic id and stage bounded focus passes rather than
            # falsifying the source relation or dropping a participant.
            cluster_chunks = [
                cluster[offset:offset + max_motion_contributions_per_slice]
                for offset in range(
                    0,
                    len(cluster),
                    max_motion_contributions_per_slice,
                )
            ]
            for chunk in cluster_chunks:
                if packed_keys and (
                    len(packed_keys[-1]) + len(chunk)
                    <= max_motion_contributions_per_slice
                ):
                    packed_keys[-1].extend(chunk)
                else:
                    packed_keys.append(list(chunk))

        key_to_subslice = {
            key: subslice_index
            for subslice_index, keys in enumerate(packed_keys)
            for key in keys
        }
        contribution_to_subslice = {
            action_index: key_to_subslice[key]
            for action_index, key in action_key.items()
        }

        def referenced_subslices(
            action_index: int,
            visiting: set[int] | None = None,
        ) -> set[int]:
            if action_index in contribution_to_subslice:
                return {contribution_to_subslice[action_index]}
            active = set(visiting or set())
            if action_index in active or not 0 <= action_index < len(relations):
                return set()
            active.add(action_index)
            resolved: set[int] = set()
            for raw_reference in (
                relations[action_index].get("reference_action_indexes") or []
            ):
                resolved.update(
                    referenced_subslices(int(raw_reference) - 1, active)
                )
            return resolved

        selected_indexes: list[list[int]] = [[] for _ in packed_keys]
        for action_index in unit.get("ledger_indexes") or []:
            action_index = int(action_index)
            if action_index in contribution_to_subslice:
                destination = contribution_to_subslice[action_index]
            else:
                owners = referenced_subslices(action_index)
                destination = min(owners) if owners else 0
            selected_indexes[destination].append(action_index)

        staged_semantic_slice_count += 1
        subslice_count = len(packed_keys)
        for subslice_order, (keys, indexes) in enumerate(
            zip(packed_keys, selected_indexes, strict=True),
            1,
        ):
            indexes = sorted(indexes)
            selected_contributions = [
                index for index in contribution_indexes
                if contribution_to_subslice[index] == subslice_order - 1
            ]
            selected_relations = [relations[index] for index in indexes]
            effect_indexes = [
                index for index in unit.get("effect_ledger_indexes") or []
                if int(index) in indexes
            ]
            sustained_indexes = [
                index for index in unit.get("sustained_ledger_indexes") or []
                if int(index) in indexes
            ]
            staged = {
                **unit,
                "kind": "provider_execution_subslice",
                "actions": [normalized["ledger"][index] for index in indexes],
                "ledger_indexes": indexes,
                "contribution_ledger_indexes": selected_contributions,
                "effect_ledger_indexes": effect_indexes,
                "sustained_ledger_indexes": sustained_indexes,
                "semantic_temporal_slice_id": semantic_id,
                "semantic_motion_load": semantic_load,
                "motion_load": len(keys),
                "performers": list(dict.fromkeys(
                    performer
                    for relation in selected_relations
                    for performer in relation.get("performers") or []
                )),
                "targets": list(dict.fromkeys(
                    target
                    for relation in selected_relations
                    for target in relation.get("targets") or []
                )),
                "state_reads": list(dict.fromkeys(
                    value
                    for relation in selected_relations
                    for value in relation.get("state_reads") or []
                )),
                "state_writes": list(dict.fromkeys(
                    value
                    for relation in selected_relations
                    for value in relation.get("state_writes") or []
                )),
                "execution_subslice_id": f"{semantic_id}:E{subslice_order:02d}",
                "execution_subslice_order": subslice_order,
                "execution_subslice_count": subslice_count,
                "provider_capacity_staging": "semantic_slice_partitioned",
                "semantic_simultaneity_preserved": True,
                "start_state": (
                    str(unit.get("start_state") or "")
                    if subslice_order == 1
                    else f"continue semantic slice {semantic_id}"
                ),
                "end_state": (
                    str(unit.get("end_state") or "")
                    if subslice_order == subslice_count
                    else f"semantic slice {semantic_id} staging continues"
                ),
            }
            execution_units.append(staged)

    result = dict(normalized)
    result.update({
        "semantic_units": len(semantic_units),
        "units": len(execution_units),
        "generation_action_units": execution_units,
        "provider_execution_subslice_count": len(execution_units),
        "staged_semantic_slice_count": staged_semantic_slice_count,
        "max_motion_contributions_per_execution_subslice": (
            max_motion_contributions_per_slice
        ),
    })
    return result


def normalize_event_action_units(
    event: dict[str, Any],
    *,
    actions: list[str] | None = None,
    seen: set[str] | None = None,
    max_motion_contributions_per_slice: int | None = None,
) -> dict[str, Any]:
    """Normalize one event using its source-authored choreography semantics."""

    event_actions = (
        event.get("micro_actions") or []
        if actions is None
        else actions
    )
    if isinstance(event_actions, str):
        event_actions = [event_actions]
    normalized_actions = [
        str(action).strip() for action in event_actions if str(action).strip()
    ]
    timeline = build_event_action_timeline(
        event,
        actions=normalized_actions,
        seen=seen,
    )
    if timeline is not None:
        categories = ["sustained"] * len(normalized_actions)
        generation_units: list[dict[str, Any]] = []
        for timeline_slice in timeline["slices"]:
            contribution_indexes = [
                int(index) - 1
                for index in timeline_slice["contribution_action_indexes"]
            ]
            for index in contribution_indexes:
                relation = timeline["actions"][index]["temporal_relation"]
                categories[index] = (
                    "simultaneous"
                    if relation in _OVERLAP_RELATIONS
                    else "sequential"
                )
            if not contribution_indexes:
                continue
            all_indexes = [
                int(index) - 1
                for index in timeline_slice["source_micro_action_indexes"]
            ]
            generation_units.append({
                "kind": "temporal_slice",
                "actions": [normalized_actions[index] for index in all_indexes],
                "ledger_indexes": all_indexes,
                "contribution_ledger_indexes": contribution_indexes,
                "effect_ledger_indexes": [
                    int(index) - 1
                    for index in timeline_slice["effect_action_indexes"]
                ],
                "sustained_ledger_indexes": [
                    int(index) - 1
                    for index in timeline_slice["sustained_action_indexes"]
                ],
                "temporal_slice_id": timeline_slice["slice_id"],
                "motion_load": int(timeline_slice["motion_load"]),
                "pace_weight": int(timeline_slice["pace_weight"]),
                "performers": list(timeline_slice["performers"]),
                "targets": list(timeline_slice["targets"]),
                "state_reads": list(timeline_slice["state_reads"]),
                "state_writes": list(timeline_slice["state_writes"]),
                "start_state": str(timeline_slice.get("start_state") or ""),
                "end_state": str(timeline_slice.get("end_state") or ""),
            })
        for position, unit in enumerate(generation_units, 1):
            unit["unit_id"] = f"GAU{position:03d}"
        rewrite = event.get("production_action_rewrite")
        if rewrite is not None:
            if (
                not isinstance(rewrite, dict)
                or rewrite.get("schema")
                != SOURCE_INDEXED_SCREENPLAY_REWRITE_SCHEMA
                or rewrite.get("omitted_source_micro_action_indexes") != []
            ):
                raise ValueError("invalid source-indexed screenplay rewrite ledger")
            groups = rewrite.get("groups")
            source_count = rewrite.get("source_micro_action_count")
            if (
                not isinstance(groups, list)
                or len(groups) != len(generation_units)
                or isinstance(source_count, bool)
                or not isinstance(source_count, int)
                or source_count < 0
            ):
                raise ValueError("source-indexed rewrite group count is invalid")
            source_actions_by_index: dict[int, str] = {}
            for position, (unit, group) in enumerate(
                zip(generation_units, groups, strict=True),
                1,
            ):
                if (
                    not isinstance(group, dict)
                    or group.get("production_action_index") != position
                ):
                    raise ValueError("source-indexed rewrite group order is invalid")
                indexes = group.get("source_micro_action_indexes")
                source_actions = group.get("source_actions")
                if (
                    not isinstance(indexes, list)
                    or not isinstance(source_actions, list)
                    or len(indexes) != len(source_actions)
                    or indexes != sorted(set(indexes))
                    or any(
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or not 1 <= index <= source_count
                        for index in indexes
                    )
                ):
                    raise ValueError("source-indexed rewrite action lineage is invalid")
                for source_index, source_action in zip(
                    indexes,
                    source_actions,
                    strict=True,
                ):
                    if source_index in source_actions_by_index:
                        raise ValueError("source-indexed rewrite duplicated a source fact")
                    source_actions_by_index[source_index] = str(source_action)
                motion_load = int(group.get("maximum_motion_load") or 0)
                pace_weight = int(group.get("pace_weight") or 0)
                if motion_load < 1 or pace_weight < 1:
                    raise ValueError("source-indexed rewrite capacity metadata is invalid")
                unit.update({
                    "source_micro_action_indexes": list(indexes),
                    "source_fact_echoes": [str(value) for value in source_actions],
                    "source_actions_sha256": str(
                        group.get("source_actions_sha256") or ""
                    ),
                    "source_generation_unit_indexes": list(
                        group.get("source_generation_unit_indexes") or []
                    ),
                    "motion_load": motion_load,
                    "pace_weight": pace_weight,
                    "performers": list(group.get("performers") or []),
                    "targets": list(group.get("targets") or []),
                    "state_reads": list(group.get("state_reads") or []),
                    "state_writes": list(group.get("state_writes") or []),
                    "start_state": str(group.get("start_state") or ""),
                    "end_state": str(group.get("end_state") or ""),
                })
            if set(source_actions_by_index) != set(range(1, source_count + 1)):
                raise ValueError("source-indexed rewrite omitted source facts")
            source_actions = [
                source_actions_by_index[index]
                for index in range(1, source_count + 1)
            ]
            source_hash = hashlib.sha256(
                json.dumps(
                    source_actions,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if source_hash != rewrite.get("source_micro_actions_sha256"):
                raise ValueError("source-indexed rewrite source hash mismatch")
        result = {
            "ledger": normalized_actions,
            "categories": categories,
            "units": len(generation_units),
            "generation_action_units": generation_units,
            "sequential": sum(1 for value in categories if value == "sequential"),
            "simultaneous_clusters": sum(
                1 for value in generation_units
                if any(
                    timeline["actions"][index]["temporal_relation"]
                    in _OVERLAP_RELATIONS
                    for index in value["contribution_ledger_indexes"]
                )
            ),
            "motion_mode": "temporal",
            "action_timeline": timeline,
        }
        if max_motion_contributions_per_slice is not None:
            return _provider_execution_units(
                result,
                max_motion_contributions_per_slice=(
                    max_motion_contributions_per_slice
                ),
            )
        return result
    result = normalize_action_units(
        normalized_actions,
        seen=seen,
        composite_motion=event_uses_composite_motion(event),
    )
    if max_motion_contributions_per_slice is not None:
        result = dict(result)
        result.update({
            "semantic_units": int(result.get("units") or 0),
            "provider_execution_subslice_count": int(result.get("units") or 0),
            "staged_semantic_slice_count": 0,
            "max_motion_contributions_per_execution_subslice": (
                max_motion_contributions_per_slice
            ),
        })
    return result


def normalized_action_unit_count(
    actions: list[str],
    *,
    seen: set[str] | None = None,
) -> int:
    """Convenience: return only the unit count."""
    return normalize_action_units(actions, seen=seen)["units"]

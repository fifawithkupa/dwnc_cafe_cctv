"""App-facing seat availability report.

Pure formatting over already-validated tracker output: dict in, dict out.
It deliberately raises rather than swallowing errors -- a failure here is a
bug, not an operating condition.  24/7 resilience is the systemd layer's job.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Sequence, Tuple


class ReasonCode(str, Enum):
    """Why a seat ended up in the state it did.

    Grouped by what fixes them, so the distribution doubles as the
    improvement backlog: "UNKNOWN 34% = geometry 22% + model 8% + time 4%"
    turns the next engineering decision into a table lookup.
    """

    # install -- fixed by moving the camera, never by code (CLAUDE.md)
    BORDER_CROPPED = "border_cropped"
    SCENE_CHANGE = "scene_change"

    # geometry / occlusion -- fixed by rescue paths or redrawn seat slots
    OCCLUDED_LOWER_BODY = "occluded_lower_body"
    AMBIGUOUS_ASSOCIATION = "ambiguous_association"
    SPANS_MULTIPLE_SEATS = "spans_multiple_seats"

    # model -- fixed by fine-tuning or a larger imgsz
    POSE_LOW_KEYPOINTS = "pose_low_keypoints"
    TABLE_NOT_DETECTED = "table_not_detected"

    # time -- fixed by waiting; nothing to do
    TRACK_PREDICTED = "track_predicted"
    PENDING_CONFIRMATION = "pending_confirmation"

    # settled judgements
    PERSON_SEATED = "person_seated"
    BELONGINGS = "belongings"
    OCCUPIED_CHAIR = "occupied_chair"
    NO_CUSTOMER_EVIDENCE = "no_customer_evidence"
    # Bar seat: belongings were seen and deliberately not counted, because at
    # a counter one customer's things span several stools.  Named separately
    # from a genuinely bare seat so the choice stays measurable.
    BELONGINGS_ONLY = "belongings_only"


REASON_GROUPS: Dict[str, Tuple[ReasonCode, ...]] = {
    "install": (ReasonCode.BORDER_CROPPED, ReasonCode.SCENE_CHANGE),
    "geometry": (
        ReasonCode.OCCLUDED_LOWER_BODY,
        ReasonCode.AMBIGUOUS_ASSOCIATION,
        ReasonCode.SPANS_MULTIPLE_SEATS,
    ),
    "model": (ReasonCode.POSE_LOW_KEYPOINTS, ReasonCode.TABLE_NOT_DETECTED),
    "time": (ReasonCode.TRACK_PREDICTED, ReasonCode.PENDING_CONFIRMATION),
    "settled": (
        ReasonCode.PERSON_SEATED,
        ReasonCode.BELONGINGS,
        ReasonCode.OCCUPIED_CHAIR,
        ReasonCode.NO_CUSTOMER_EVIDENCE,
        ReasonCode.BELONGINGS_ONLY,
    ),
}

# Improvement targets: "install" is the camera's job and the other groups
# need no action, so only these two are engineering backlog.
ACTIONABLE_GROUPS: Tuple[str, ...] = ("geometry", "model")


def classify_reason(raw_state: str, reason: str, predicted: bool) -> ReasonCode:
    """Map a free-text observation reason onto the closed reason vocabulary.

    ``predicted`` wins over the reason text: a predicted track carries the
    reason of its last real observation, which would otherwise be reported
    as settled evidence that is not actually being seen right now.
    """
    if predicted:
        return ReasonCode.TRACK_PREDICTED

    text = reason or ""
    if text.startswith("temporarily_occluded:"):
        return ReasonCode.TRACK_PREDICTED
    # The table layer prefixes the pose-level cause; unwrap it so the cause
    # is what gets classified rather than the wrapper.
    if text.startswith("nearby_person_pose_unknown:"):
        text = text.split(":", 1)[1]

    if text.startswith("compact_occluded_pose"):
        return ReasonCode.OCCLUDED_LOWER_BODY
    if text.startswith("insufficient_keypoints"):
        return ReasonCode.POSE_LOW_KEYPOINTS
    if text.startswith("spans_multiple_seats"):
        return ReasonCode.SPANS_MULTIPLE_SEATS
    if text.startswith("border_cropped"):
        return ReasonCode.BORDER_CROPPED
    if text.startswith("scene_change"):
        return ReasonCode.SCENE_CHANGE
    if text.startswith("nearby_person_pose_unknown"):
        return ReasonCode.AMBIGUOUS_ASSOCIATION
    if text.startswith("belongings_only"):
        return ReasonCode.BELONGINGS_ONLY
    if text.startswith("no_customer_evidence"):
        return ReasonCode.NO_CUSTOMER_EVIDENCE

    if raw_state == "occupied":
        if "seated:" in text:
            return ReasonCode.PERSON_SEATED
        if "objects:" in text or "chair_objects:" in text:
            return ReasonCode.BELONGINGS
        if "occupied_chairs:" in text:
            return ReasonCode.OCCUPIED_CHAIR
        return ReasonCode.PERSON_SEATED
    if raw_state == "empty":
        return ReasonCode.NO_CUSTOMER_EVIDENCE
    if raw_state == "ignore":
        return ReasonCode.BORDER_CROPPED
    return ReasonCode.AMBIGUOUS_ASSOCIATION


SCHEMA_VERSION = 1

# IGNORE never reaches the app: it means the camera cannot see the seat at
# all, which is an installation defect rather than a seat to reason about.
_COUNTABLE_STATES = ("occupied", "empty", "unknown")


def _seat_name(table: Dict[str, Any]) -> str:
    return str(table.get("layout_name") or table.get("label") or "?")


def build_seat_report(
    tables: Sequence[Dict[str, Any]], tick_at: float
) -> Dict[str, Any]:
    """Turn tracker output into the app-facing availability contract.

    ``free`` counts only confirmed empties.  UNKNOWN is never rounded into
    availability -- "3 free, 2 unconfirmed" is honest, "5 free" is not.
    """
    plain: List[Dict[str, Any]] = []
    zones: Dict[str, Dict[str, Any]] = {}
    zone_order: List[str] = []

    for table in tables:
        state = str(table.get("state", "unknown"))
        if state not in _COUNTABLE_STATES:
            continue
        code = classify_reason(
            str(table.get("raw_state", state)),
            str(table.get("reason", "")),
            bool(table.get("predicted", False)),
        )
        if str(table.get("layout_kind", "table")) != "counted_zone":
            plain.append(
                {
                    "seat_id": _seat_name(table),
                    "kind": "table",
                    "capacity": 1,
                    "state": state,
                    "reason_code": code.value,
                    "confidence": round(float(table.get("confidence", 0.0)), 4),
                }
            )
            continue

        zone_name = str(table.get("layout_zone_name") or "?")
        if zone_name not in zones:
            zones[zone_name] = {
                "seat_id": zone_name,
                "kind": "counted_zone",
                "capacity": 0,
                "occupied": 0,
                "free": 0,
                "unknown": 0,
                "reason_codes": {},
            }
            zone_order.append(zone_name)
        zone = zones[zone_name]
        zone["capacity"] += 1
        if state == "occupied":
            zone["occupied"] += 1
        elif state == "empty":
            zone["free"] += 1
        else:
            zone["unknown"] += 1
            counts = zone["reason_codes"]
            counts[code.value] = counts.get(code.value, 0) + 1

    seats: List[Dict[str, Any]] = plain + [zones[name] for name in zone_order]

    totals = {"capacity": 0, "occupied": 0, "free": 0, "unknown": 0}
    for seat in seats:
        if seat["kind"] == "counted_zone":
            totals["capacity"] += seat["capacity"]
            totals["occupied"] += seat["occupied"]
            totals["free"] += seat["free"]
            totals["unknown"] += seat["unknown"]
            continue
        totals["capacity"] += 1
        if seat["state"] == "empty":
            totals["free"] += 1
        elif seat["state"] == "occupied":
            totals["occupied"] += 1
        else:
            totals["unknown"] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "tick_at": round(float(tick_at), 6),
        "seats": seats,
        "totals": totals,
    }

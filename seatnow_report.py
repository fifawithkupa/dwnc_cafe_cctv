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

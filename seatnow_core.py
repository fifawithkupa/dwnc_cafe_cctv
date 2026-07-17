"""Core inference, tracking, rendering, and FFmpeg I/O for SeatNow.

The module deliberately keeps video decoding/encoding out of OpenCV.  The
OpenCV wheel available on the target Intel Mac has no FFmpeg/GStreamer video
backend, while the system ``ffmpeg`` binary is fully functional.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Box = Tuple[float, float, float, float]
Point = Tuple[float, float]


class PoseState(str, Enum):
    SEATED = "seated"
    STANDING = "standing"
    UNKNOWN = "unknown"


class OccupancyState(str, Enum):
    OCCUPIED = "occupied"
    EMPTY = "empty"
    UNKNOWN = "unknown"
    IGNORE = "ignore"


# Furniture, fixed fixtures, animals, and scene-level false positives must not
# become a customer-belonging signal.  Portable classes that are not listed
# here continue to follow SeatNow's "any non-person object" rule.
EXCLUDED_OBJECT_CLASSES = {
    "person",
    "chair",
    "couch",
    "dining table",
    "bench",
    "potted plant",
    "tv",
    "refrigerator",
    "cat",
    "dog",
    "bird",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "toilet",
    "bed",
    "sink",
    "oven",
    "microwave",
    "toaster",
    "clock",
}


# COCO pose indices.
L_SHO, R_SHO = 5, 6
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16


@dataclass(frozen=True)
class Detection:
    name: str
    box: Box
    confidence: float

    @property
    def center(self) -> Point:
        return box_center(self.box)


@dataclass
class PoseObservation:
    box: Box
    confidence: float
    state: PoseState
    anchor: Point
    reason: str
    angles: Dict[str, float] = field(default_factory=dict)


@dataclass
class TableObservation:
    box: Box
    table_confidence: float
    raw_state: OccupancyState
    raw_score: float
    source: str = "detected"
    objects: List[Detection] = field(default_factory=list)
    seated_people: List[PoseObservation] = field(default_factory=list)
    connected_chairs: List[Detection] = field(default_factory=list)
    occupied_chairs: List[Detection] = field(default_factory=list)
    chair_seated_people: List[PoseObservation] = field(default_factory=list)
    reason: str = ""
    provisional: bool = False
    layout_id: Optional[int] = None
    layout_name: Optional[str] = None
    # Burst-sample majority voting, e.g. {"occupied": 3, "empty": 2}.
    vote_counts: Optional[Dict[str, int]] = None


@dataclass
class FrameAnalysis:
    timestamp: float
    tables: List[TableObservation]
    poses: List[PoseObservation]
    detections: List[Detection]
    inference_ms: float
    scene_change: bool = False
    scene_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnalyzerConfig:
    imgsz: int = 1280
    pose_imgsz: int = 960
    table_confidence: float = 0.20
    object_confidence: float = 0.15
    pose_confidence: float = 0.20
    keypoint_confidence: float = 0.30
    sitting_angle: float = 110.0
    table_overlap: float = 0.65
    # Soft cap: boxes above this frame-area fraction need extra evidence
    # (high confidence or supporting chairs) instead of being dropped.
    maximum_table_area_fraction: float = 0.06
    large_table_confidence: float = 0.30
    hard_table_area_fraction: float = 0.40
    table_rescue_confidence: float = 0.12
    table_crop_objects: bool = True
    table_crop_imgsz: int = 960
    table_crop_confidence: float = 0.12
    maximum_table_crops: int = 4
    moving_person_threshold: float = 0.025
    border_pixels: int = 3
    infer_occluded_tables: bool = True
    # Layout mode: let zones drift toward matching furniture detections.
    layout_tracking: bool = True
    layout_track_alpha: float = 0.35
    layout_track_table_confidence: float = 0.25
    layout_track_chair_confidence: float = 0.30
    device: str = "cpu"


@dataclass
class Track:
    track_id: int
    box: Box
    stable_state: OccupancyState
    last_observation: TableObservation
    first_seen: float
    last_seen: float
    visible: bool = True
    missed: int = 0
    pending_state: Optional[OccupancyState] = None
    pending_count: int = 0
    velocity: Point = (0.0, 0.0)
    predicted: bool = False
    display_box: Optional[Box] = None
    # Require repeated evidence before an inferred seat may be predicted.
    real_hits: int = 1

    @property
    def label(self) -> str:
        observation = self.last_observation
        if observation.layout_id is not None:
            return f"L{observation.layout_id:03d}"
        prefix = "S" if observation.source == "inferred-seat" else "T"
        return f"{prefix}{self.track_id:03d}"

    @property
    def visible_state(self) -> OccupancyState:
        """Frame-scoring state, separate from the retained world state.

        A border-cropped/uncertain observation is IGNORE/UNKNOWN in the current
        frame, but it must not mutate a previously occupied table into EMPTY.
        """
        raw = self.last_observation.raw_state
        if raw in (OccupancyState.IGNORE, OccupancyState.UNKNOWN):
            return raw
        return self.stable_state

    @property
    def current_box(self) -> Box:
        return self.display_box if self.display_box is not None else self.box


@dataclass
class TrackerUpdate:
    visible_tracks: List[Track]
    all_tracks: List[Track]
    events: List[Dict[str, object]]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    source_frames: Optional[int] = None


def box_center(box: Box) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_diagonal(box: Box) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1))


def intersection_area(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_over_smaller(a: Box, b: Box) -> float:
    smaller = min(box_area(a), box_area(b))
    return intersection_area(a, b) / smaller if smaller > 0 else 0.0


def point_box_distance(point: Point, box: Box) -> float:
    px, py = point
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def box_distance(a: Box, b: Box) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def expand_box(box: Box, x_margin: float, y_margin: Optional[float] = None) -> Box:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if y_margin is None:
        y_margin = x_margin
    return (
        x1 - width * x_margin,
        y1 - height * y_margin,
        x2 + width * x_margin,
        y2 + height * y_margin,
    )


def clip_box(box: Box, width: int, height: int) -> Box:
    return (
        max(0.0, min(float(width - 1), box[0])),
        max(0.0, min(float(height - 1), box[1])),
        max(0.0, min(float(width - 1), box[2])),
        max(0.0, min(float(height - 1), box[3])),
    )


def table_surface_box(box: Box) -> Box:
    """Approximate the tabletop band rather than using the full furniture box."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return (
        x1 - 0.08 * width,
        y1 - 0.30 * height,
        x2 + 0.08 * width,
        y1 + 0.58 * height,
    )


def angle_degrees(a: Optional[Point], b: Optional[Point], c: Optional[Point]) -> Optional[float]:
    if a is None or b is None or c is None:
        return None
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    magnitude = math.hypot(*ba) * math.hypot(*bc)
    if magnitude <= 1e-9:
        return None
    cosine = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / magnitude))
    return math.degrees(math.acos(cosine))


def _keypoint(keypoints: Sequence[Sequence[float]], index: int, threshold: float) -> Optional[Point]:
    if index >= len(keypoints) or len(keypoints[index]) < 3:
        return None
    x, y, confidence = keypoints[index][:3]
    return (float(x), float(y)) if float(confidence) >= threshold else None


def pose_anchor(keypoints: Sequence[Sequence[float]], box: Box, threshold: float) -> Point:
    hips = [
        _keypoint(keypoints, L_HIP, threshold),
        _keypoint(keypoints, R_HIP, threshold),
    ]
    visible = [point for point in hips if point is not None]
    if visible:
        return (
            sum(point[0] for point in visible) / len(visible),
            sum(point[1] for point in visible) / len(visible),
        )
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y1 + 0.72 * (y2 - y1))


def classify_pose(
    keypoints: Sequence[Sequence[float]],
    box: Box,
    detection_confidence: float,
    keypoint_threshold: float = 0.30,
    angle_threshold: float = 110.0,
) -> PoseObservation:
    """Classify a pose into seated/standing/unknown with observable reasons.

    SeatNow's agreed rule remains HKA OR torso angle below the threshold.  The
    third vertical-distance rule stays intentionally disabled.  Missing joints
    now produce UNKNOWN rather than silently becoming standing.
    """
    measured: Dict[str, float] = {}
    for side, shoulder_i, hip_i, knee_i, ankle_i in (
        ("left", L_SHO, L_HIP, L_KNE, L_ANK),
        ("right", R_SHO, R_HIP, R_KNE, R_ANK),
    ):
        shoulder = _keypoint(keypoints, shoulder_i, keypoint_threshold)
        hip = _keypoint(keypoints, hip_i, keypoint_threshold)
        knee = _keypoint(keypoints, knee_i, keypoint_threshold)
        ankle = _keypoint(keypoints, ankle_i, keypoint_threshold)
        hka = angle_degrees(hip, knee, ankle)
        torso = angle_degrees(shoulder, hip, knee)
        if hka is not None:
            measured[f"{side}_hka"] = hka
        if torso is not None:
            measured[f"{side}_torso"] = torso

    passing = [(name, value) for name, value in measured.items() if value < angle_threshold]
    if passing:
        name, value = min(passing, key=lambda item: item[1])
        state = PoseState.SEATED
        reason = f"{name}={value:.1f}<{angle_threshold:.0f}"
    elif measured:
        state = PoseState.STANDING
        lowest = min(measured.items(), key=lambda item: item[1])
        reason = f"all_angles>=threshold (min {lowest[0]}={lowest[1]:.1f})"
    else:
        width = max(1.0, box[2] - box[0])
        height = max(1.0, box[3] - box[1])
        # A seated customer close to the camera or behind a table often has no
        # visible knee/ankle.  A compact, lower-body-occluded person box is a
        # useful conservative fallback; tall staff boxes remain UNKNOWN.
        if height / width <= 1.75:
            state = PoseState.SEATED
            reason = f"compact_occluded_pose={height / width:.2f}"
        else:
            state = PoseState.UNKNOWN
            reason = "insufficient_keypoints"

    return PoseObservation(
        box=box,
        confidence=detection_confidence,
        state=state,
        anchor=pose_anchor(keypoints, box, keypoint_threshold),
        reason=reason,
        angles=measured,
    )


def is_sitting(keypoints: Sequence[Sequence[float]], threshold: float = 0.30) -> bool:
    """Compatibility helper retained for callers of the old image MVP."""
    xs = [float(point[0]) for point in keypoints if len(point) >= 2]
    ys = [float(point[1]) for point in keypoints if len(point) >= 2]
    box = (min(xs, default=0.0), min(ys, default=0.0), max(xs, default=0.0), max(ys, default=0.0))
    return classify_pose(keypoints, box, 1.0, threshold).state == PoseState.SEATED


def deduplicate_tables(tables: Sequence[Detection], overlap_threshold: float = 0.65) -> List[Detection]:
    """Remove nested/duplicate table boxes while retaining adjacent tables."""
    kept: List[Detection] = []
    for candidate in sorted(tables, key=lambda item: item.confidence, reverse=True):
        duplicate = any(
            box_iou(candidate.box, existing.box) >= 0.50
            or overlap_over_smaller(candidate.box, existing.box) >= overlap_threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def _object_table_score(obj: Detection, table: Detection) -> float:
    surface = table_surface_box(table.box)
    object_area = box_area(obj.box)
    table_area = box_area(table.box)
    if object_area <= 0 or table_area <= 0 or object_area > table_area * 1.25:
        return 0.0

    intersection_ratio = intersection_area(obj.box, surface) / object_area
    bottom_center = ((obj.box[0] + obj.box[2]) / 2.0, obj.box[3])
    anchor_distance = point_box_distance(bottom_center, surface)
    maximum_distance = max(10.0, 0.18 * box_diagonal(table.box))
    proximity = max(0.0, 1.0 - anchor_distance / maximum_distance)
    if intersection_ratio < 0.08 and proximity <= 0.0:
        return 0.0
    return 0.65 * min(1.0, intersection_ratio * 2.0) + 0.35 * proximity


def associate_objects(
    tables: Sequence[Detection], objects: Sequence[Detection]
) -> Dict[int, List[Detection]]:
    """Assign each object to at most one tabletop."""
    assignments: Dict[int, List[Detection]] = {index: [] for index in range(len(tables))}
    for obj in objects:
        scores = [(_object_table_score(obj, table), index) for index, table in enumerate(tables)]
        score, table_index = max(scores, default=(0.0, -1))
        if table_index >= 0 and score >= 0.22:
            assignments[table_index].append(obj)
    return assignments


def _person_table_score(person: PoseObservation, table: Detection, frame_shape: Tuple[int, int]) -> float:
    height, width = frame_shape
    frame_diagonal = math.hypot(width, height)
    expanded = expand_box(table.box, 0.12, 0.18)
    person_height = max(1.0, person.box[3] - person.box[1])
    table_diagonal = box_diagonal(table.box)
    vertical_gap = max(0.0, table.box[1] - person.box[3])
    if vertical_gap > max(0.24 * person_height, 0.10 * table_diagonal):
        return 0.0
    distance = point_box_distance(person.anchor, expanded)
    maximum = max(0.50 * table_diagonal, 0.040 * frame_diagonal)
    if distance > maximum:
        return 0.0

    # A seated body should be beside/overlapping a table, not arbitrarily far
    # across the scene.  IoU is only a bonus; it is not required.
    proximity = max(0.0, 1.0 - distance / maximum)
    overlap_bonus = min(1.0, overlap_over_smaller(person.box, expand_box(table.box, 0.20)))
    return 0.78 * proximity + 0.22 * overlap_bonus


def associate_people(
    tables: Sequence[Detection],
    people: Sequence[PoseObservation],
    frame_shape: Tuple[int, int],
    allowed_states: Sequence[PoseState] = (PoseState.SEATED,),
) -> Tuple[Dict[int, List[PoseObservation]], List[PoseObservation]]:
    """Assign each seated person to one nearest plausible table."""
    assignments: Dict[int, List[PoseObservation]] = {index: [] for index in range(len(tables))}
    unassigned: List[PoseObservation] = []
    for person in people:
        if person.state not in allowed_states:
            continue
        scores = [
            (_person_table_score(person, table, frame_shape), index)
            for index, table in enumerate(tables)
        ]
        score, table_index = max(scores, default=(0.0, -1))
        if table_index >= 0 and score >= 0.22:
            assignments[table_index].append(person)
        else:
            unassigned.append(person)
    return assignments, unassigned


def _chair_table_score(
    chair: Detection,
    table: Detection,
    frame_shape: Tuple[int, int],
) -> float:
    """Score whether a physical chair belongs to a table ROI."""
    table_diagonal = max(1.0, box_diagonal(table.box))
    gap = box_distance(chair.box, table.box)
    gap_ratio = gap / table_diagonal
    if gap_ratio > 0.35:
        return 0.0

    chair_center = box_center(chair.box)
    table_center = box_center(table.box)
    if abs(chair_center[1] - table_center[1]) > 0.70 * table_diagonal:
        return 0.0
    center_distance = math.hypot(
        chair_center[0] - table_center[0], chair_center[1] - table_center[1]
    )
    maximum_center_distance = 1.25 * table_diagonal
    if center_distance > maximum_center_distance:
        return 0.0

    expanded_table = expand_box(table.box, 0.50, 0.70)
    overlap = min(1.0, overlap_over_smaller(chair.box, expanded_table))
    center_proximity = max(0.0, 1.0 - center_distance / maximum_center_distance)
    gap_proximity = max(0.0, 1.0 - gap_ratio / 0.35)
    return 0.50 * gap_proximity + 0.30 * overlap + 0.20 * center_proximity


def associate_chairs_to_tables(
    tables: Sequence[Detection],
    chairs: Sequence[Detection],
    frame_shape: Tuple[int, int],
) -> Dict[int, List[int]]:
    """Link every chair/couch/bench to at most one nearest plausible table."""
    assignments: Dict[int, List[int]] = {index: [] for index in range(len(tables))}
    for chair_index, chair in enumerate(chairs):
        scores = sorted(
            [
            (_chair_table_score(chair, table, frame_shape), table_index)
            for table_index, table in enumerate(tables)
            ],
            reverse=True,
        )
        score, table_index = scores[0] if scores else (0.0, -1)
        second_score = scores[1][0] if len(scores) > 1 else 0.0
        unambiguous = second_score < 0.35 or score - second_score >= 0.08
        if table_index >= 0 and score >= 0.35 and unambiguous:
            assignments[table_index].append(chair_index)
    return assignments


def select_table_candidates(
    candidates: Sequence[Detection],
    chairs: Sequence[Detection],
    frame_shape: Tuple[int, int],
    table_confidence: float = 0.20,
    soft_area_fraction: float = 0.06,
    large_table_confidence: float = 0.30,
    hard_area_fraction: float = 0.40,
    rescue_confidence: float = 0.12,
    rescue_chair_count: int = 2,
    rescue_chair_confidence: float = 0.35,
) -> List[Detection]:
    """Accept table boxes on confidence, size, and chair-structure evidence.

    A blanket frame-area cap rejected real foreground tables the detector saw
    at up to 0.80 confidence, and the flat confidence gate rejected real back
    tables at 0.15-0.20 that had confident chairs tucked into them.  Large or
    weak candidates are therefore accepted when either the confidence carries
    them or at least ``rescue_chair_count`` confident chairs support them; a
    merged box that swallows two independently accepted tables is vetoed so
    adjacent tables are not collapsed into one.
    """
    height, width = frame_shape
    frame_area = float(max(1, width * height))
    confident_chairs = [
        chair for chair in chairs if chair.confidence >= rescue_chair_confidence
    ]
    accepted: List[Detection] = []
    for candidate in candidates:
        area_fraction = box_area(candidate.box) / frame_area
        if area_fraction > hard_area_fraction:
            continue
        supporting_chairs = sum(
            _chair_table_score(chair, candidate, frame_shape) >= 0.35
            for chair in confident_chairs
        )
        chair_backed = supporting_chairs >= rescue_chair_count
        if candidate.confidence >= table_confidence:
            if (
                area_fraction <= soft_area_fraction
                or candidate.confidence >= large_table_confidence
                or chair_backed
            ):
                accepted.append(candidate)
        elif candidate.confidence >= rescue_confidence and chair_backed:
            accepted.append(candidate)

    kept: List[Detection] = []
    for outer in accepted:
        outer_area = box_area(outer.box)
        contained = sum(
            1
            for inner in accepted
            if inner is not outer
            and box_area(inner.box) <= 0.7 * outer_area
            and overlap_over_smaller(inner.box, outer.box) >= 0.85
        )
        if contained >= 2:
            continue
        # An above-soft-cap box that meaningfully overlaps a smaller accepted
        # table is a merged/duplicated region; keeping it would steal the
        # smaller table's object evidence (seen on the pan fixture, where a
        # 13%-of-frame box absorbed the teddy-bear table).  Prefer the finer
        # granularity regardless of relative confidence.
        if outer_area / frame_area > soft_area_fraction and any(
            box_area(other.box) < outer_area
            and overlap_over_smaller(other.box, outer.box) >= 0.30
            for other in accepted
            if other is not outer
        ):
            continue
        kept.append(outer)
    return kept


def filter_carried_objects(
    objects: Sequence[Detection],
    poses: Sequence[PoseObservation],
    overlap_threshold: float = 0.60,
) -> List[Detection]:
    """Drop objects riding on a person who is not seated.

    A bag in the hands of a walking customer must not mark a table occupied,
    and hallucinated objects on a person occluding a seat collapse onto the
    person box the same way.  Objects overlapping a seated customer stay:
    their table is occupied through the person anyway.
    """
    carriers = [pose for pose in poses if pose.state != PoseState.SEATED]
    kept: List[Detection] = []
    for obj in objects:
        object_area = box_area(obj.box)
        carried = any(
            overlap_over_smaller(obj.box, person.box) >= overlap_threshold
            and object_area <= 0.6 * box_area(person.box)
            for person in carriers
        )
        if not carried:
            kept.append(obj)
    return kept


def associate_objects_to_chairs(
    chairs: Sequence[Detection], objects: Sequence[Detection]
) -> Dict[int, List[Detection]]:
    """Assign belongings resting on a chair to at most one chair."""
    assignments: Dict[int, List[Detection]] = {
        index: [] for index in range(len(chairs))
    }
    for obj in objects:
        object_area = box_area(obj.box)
        if object_area <= 0:
            continue
        best_score, best_index = 0.0, -1
        for index, chair in enumerate(chairs):
            chair_area = box_area(chair.box)
            if chair_area <= 0 or object_area > 0.9 * chair_area:
                continue
            score = intersection_area(obj.box, chair.box) / object_area
            if score > best_score:
                best_score, best_index = score, index
        if best_index >= 0 and best_score >= 0.55:
            assignments[best_index].append(obj)
    return assignments


def filter_strong_chair_links(
    tables: Sequence[Detection],
    chairs: Sequence[Detection],
    assignments: Dict[int, List[int]],
    frame_shape: Tuple[int, int],
    threshold: float = 0.75,
) -> Dict[int, List[int]]:
    """Keep only chair links tight enough to propagate occupancy.

    The broad 0.35 link is right for structure (rescuing weak tables, drawing
    context), but propagating a seated person through it absorbed customers of
    a fully occluded neighbouring table into the wrong table on the pan
    fixture (absorbing link scored 0.67; genuinely tucked-in chairs score
    0.94+).  A person on a weakly linked chair falls back to the
    inferred-seat path and surfaces as their own occupied table.
    """
    return {
        table_index: [
            chair_index
            for chair_index in chair_indices
            if _chair_table_score(
                chairs[chair_index], tables[table_index], frame_shape
            )
            >= threshold
        ]
        for table_index, chair_indices in assignments.items()
    }


def _person_chair_score(person: PoseObservation, chair: Detection) -> float:
    """Score whether a seated person's hip/body is supported by one chair."""
    person_diagonal = max(1.0, box_diagonal(person.box))
    overlap = min(1.0, overlap_over_smaller(person.box, chair.box))
    distance = point_box_distance(person.anchor, chair.box)
    proximity = max(0.0, 1.0 - distance / (0.35 * person_diagonal))
    return 0.72 * overlap + 0.28 * proximity


def associate_seated_people_to_chairs(
    chairs: Sequence[Detection],
    people: Sequence[PoseObservation],
) -> Dict[int, List[PoseObservation]]:
    """Assign seated people to chairs; standing/unknown people never occupy one."""
    assignments: Dict[int, List[PoseObservation]] = {
        index: [] for index in range(len(chairs))
    }
    candidates = []
    for person_index, person in enumerate(people):
        if person.state != PoseState.SEATED:
            continue
        scores = sorted(
            [
                (_person_chair_score(person, chair), chair_index)
                for chair_index, chair in enumerate(chairs)
            ],
            reverse=True,
        )
        if not scores:
            continue
        score, chair_index = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else 0.0
        if score >= 0.16 and (second_score < 0.16 or score - second_score >= 0.05):
            candidates.append((score, person_index, chair_index))

    used_people = set()
    used_single_chairs = set()
    for _, person_index, chair_index in sorted(candidates, reverse=True):
        if person_index in used_people:
            continue
        chair = chairs[chair_index]
        if chair.name == "chair" and chair_index in used_single_chairs:
            continue
        assignments[chair_index].append(people[person_index])
        used_people.add(person_index)
        if chair.name == "chair":
            used_single_chairs.add(chair_index)
    return assignments


def occupancy_state_from_evidence(
    objects: Sequence[Detection],
    direct_seated_people: Sequence[PoseObservation],
    unknown_people: Sequence[PoseObservation],
    occupied_chairs: Sequence[Detection],
) -> OccupancyState:
    """Apply SeatNow's table OR rule to already-associated evidence."""
    if objects or direct_seated_people or occupied_chairs:
        return OccupancyState.OCCUPIED
    if unknown_people:
        return OccupancyState.UNKNOWN
    return OccupancyState.EMPTY


def _touches_frame_border(box: Box, width: int, height: int, pixels: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= pixels or y1 <= pixels or x2 >= width - 1 - pixels or y2 >= height - 1 - pixels


def is_severely_border_cropped(box: Box, width: int, height: int, pixels: int) -> bool:
    """Conservatively ignore only small slivers, not every 1px edge contact."""
    x1, y1, x2, y2 = box
    visible_width = max(0.0, x2 - x1)
    visible_height = max(0.0, y2 - y1)
    horizontal_sliver = (
        (x1 <= pixels or x2 >= width - 1 - pixels)
        and visible_width < 0.09 * width
    )
    vertical_sliver = (
        (y1 <= pixels or y2 >= height - 1 - pixels)
        and visible_height < 0.09 * height
    )
    return horizontal_sliver or vertical_sliver


def _inferred_seat_box(person: PoseObservation, width: int, height: int) -> Box:
    x1, y1, x2, y2 = person.box
    person_width, person_height = x2 - x1, y2 - y1
    return clip_box(
        (
            x1 - 0.18 * person_width,
            y1 + 0.28 * person_height,
            x2 + 0.18 * person_width,
            y2 + 0.08 * person_height,
        ),
        width,
        height,
    )


def has_seat_support(person: PoseObservation, seats: Sequence[Detection]) -> bool:
    """Return whether a chair/couch/bench plausibly supports this person."""
    return seat_support_score(person, seats) >= 0.16


def seat_support_score(person: PoseObservation, seats: Sequence[Detection]) -> float:
    """Geometric chair/person support score in [0, 1]."""
    return max(
        (_person_chair_score(person, seat) for seat in seats),
        default=0.0,
    )


def cluster_people(
    people: Sequence[PoseObservation], frame_shape: Tuple[int, int]
) -> List[List[PoseObservation]]:
    """Merge nearby people who plausibly occupy the same occluded table."""
    height, width = frame_shape
    frame_diagonal = math.hypot(width, height)
    groups: List[List[PoseObservation]] = []
    for person in sorted(people, key=lambda item: item.box[0]):
        person_center = box_center(person.box)
        joined = False
        for group in groups:
            group_centers = [box_center(member.box) for member in group]
            group_center = (
                sum(point[0] for point in group_centers) / len(group_centers),
                sum(point[1] for point in group_centers) / len(group_centers),
            )
            horizontal = abs(person_center[0] - group_center[0])
            vertical = abs(person_center[1] - group_center[1])
            widest = max(
                [person.box[2] - person.box[0]]
                + [member.box[2] - member.box[0] for member in group]
            )
            if horizontal <= max(1.7 * widest, 0.035 * frame_diagonal) and vertical <= 0.08 * frame_diagonal:
                group.append(person)
                joined = True
                break
        if not joined:
            groups.append([person])
    return groups


def is_strong_seated_evidence(person: PoseObservation) -> bool:
    return (
        person.state == PoseState.SEATED
        and "<" in person.reason
        and "chair_overlap_override" not in person.reason
    )


def suppress_conflicting_weak_seated_poses(
    poses: Sequence[PoseObservation],
) -> None:
    """Prefer an angle-confirmed standing duplicate over a chair-only seat."""
    standing = [pose for pose in poses if pose.state == PoseState.STANDING]
    for pose in poses:
        if pose.state != PoseState.SEATED or is_strong_seated_evidence(pose):
            continue
        if any(
            overlap_over_smaller(pose.box, other.box) >= 0.60
            for other in standing
        ):
            pose.state = PoseState.UNKNOWN
            pose.reason += ";conflicting_standing_duplicate"


def _inferred_group_box(group: Sequence[PoseObservation], width: int, height: int) -> Box:
    boxes = [_inferred_seat_box(person, width, height) for person in group]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


class LayoutZoneTracker:
    """Drift calibrated layout zones toward matching furniture detections.

    Zones follow furniture the detector can see and stay put otherwise, so a
    manually drawn ROI for an undetectable table simply never moves.  Matching
    is deliberately conservative: a detection that overlaps two zones about
    equally is discarded, because dragging a zone onto the wrong table is
    worse than not moving at all.  Zone count, ids, and chair links are fixed
    by construction — only the boxes drift.
    """

    def __init__(
        self,
        table_boxes: Sequence[Box],
        chair_boxes: Sequence[Box],
        alpha: float = 0.35,
        minimum_iou: float = 0.30,
        ambiguity_margin: float = 0.10,
    ):
        self.table_boxes: List[Box] = [tuple(box) for box in table_boxes]
        self.chair_boxes: List[Box] = [tuple(box) for box in chair_boxes]
        self._calibrated_tables = list(self.table_boxes)
        self._calibrated_chairs = list(self.chair_boxes)
        self.alpha = alpha
        self.minimum_iou = minimum_iou
        self.ambiguity_margin = ambiguity_margin

    def update(
        self,
        table_detections: Sequence[Detection],
        chair_detections: Sequence[Detection],
    ) -> None:
        self.table_boxes = self._track(self.table_boxes, table_detections)
        self.chair_boxes = self._track(self.chair_boxes, chair_detections)

    def reset(self) -> None:
        self.table_boxes = list(self._calibrated_tables)
        self.chair_boxes = list(self._calibrated_chairs)

    def _blend(self, zone: Box, target: Box) -> Box:
        keep = 1.0 - self.alpha
        return tuple(
            keep * zone_value + self.alpha * target_value
            for zone_value, target_value in zip(zone, target)
        )

    def _track(
        self, zones: List[Box], detections: Sequence[Detection]
    ) -> List[Box]:
        candidates: List[Tuple[float, int, int]] = []
        for det_index, detection in enumerate(detections):
            overlaps = sorted(
                (
                    (box_iou(zone, detection.box), zone_index)
                    for zone_index, zone in enumerate(zones)
                ),
                reverse=True,
            )
            qualified = [
                (iou, zone_index)
                for iou, zone_index in overlaps
                if iou >= self.minimum_iou
            ]
            if not qualified:
                continue
            if (
                len(qualified) >= 2
                and qualified[0][0] - qualified[1][0] < self.ambiguity_margin
            ):
                continue  # 애매하면 정지: 잘못 따라가는 것이 최악의 실패다.
            candidates.append((qualified[0][0], det_index, qualified[0][1]))

        used_detections: set = set()
        used_zones: set = set()
        updated = list(zones)
        for iou, det_index, zone_index in sorted(candidates, reverse=True):
            if det_index in used_detections or zone_index in used_zones:
                continue
            used_detections.add(det_index)
            used_zones.add(zone_index)
            updated[zone_index] = self._blend(
                zones[zone_index], detections[det_index].box
            )
        return updated


class SeatNowAnalyzer:
    """Run detector and pose model once per supplied BGR frame."""

    def __init__(
        self,
        det_model_path: Path,
        pose_model_path: Path,
        config: AnalyzerConfig,
        layout: Optional[object] = None,
    ):
        from ultralytics import YOLO

        self.config = config
        # Duck-typed SeatLayout: only .tables, .chair_boxes(), .chair_assignments()
        # are used, so seatnow_layout is never imported here.
        self.layout = layout
        # Lazy: created on the first analyzed frame, after the CLI has scaled
        # the layout to the actual input resolution.
        self.zone_tracker: Optional[LayoutZoneTracker] = None
        self.det_model_path = Path(det_model_path)
        self.pose_model_path = Path(pose_model_path)
        self.det_model = YOLO(str(self.det_model_path))
        self.pose_model = YOLO(str(self.pose_model_path))
        self.names = self.det_model.names
        self.previous_poses: List[PoseObservation] = []
        self.previous_pose_timestamp: Optional[float] = None

    def reset_temporal(self) -> None:
        self.previous_poses = []
        self.previous_pose_timestamp = None

    def _filter_moving_people(
        self,
        poses: Sequence[PoseObservation],
        timestamp: float,
        frame_shape: Tuple[int, int],
        global_motion_fraction: Point,
    ) -> None:
        if self.previous_pose_timestamp is None or not self.previous_poses:
            self.previous_poses = list(poses)
            self.previous_pose_timestamp = timestamp
            return
        elapsed = max(1e-6, timestamp - self.previous_pose_timestamp)
        height, width = frame_shape
        frame_diagonal = math.hypot(width, height)
        global_dx = global_motion_fraction[0] * width
        global_dy = global_motion_fraction[1] * height
        for pose in poses:
            center = box_center(pose.box)
            candidates = []
            for previous in self.previous_poses:
                previous_center = box_center(previous.box)
                predicted = (
                    previous_center[0] + global_dx,
                    previous_center[1] + global_dy,
                )
                residual = math.hypot(
                    center[0] - predicted[0], center[1] - predicted[1]
                )
                candidates.append((residual, previous))
            residual, previous = min(candidates, key=lambda item: item[0])
            normalized_speed = residual / (frame_diagonal * elapsed)
            if residual > 0.12 * frame_diagonal:
                continue
            weak_seated = pose.state == PoseState.SEATED and not is_strong_seated_evidence(pose)
            # A standing state that was itself produced by this downgrade must
            # not veto the next frame's seated evidence, or one walking frame
            # locks a customer into standing for the rest of the video.
            previous_downgraded = (
                previous.state == PoseState.STANDING and "moving=" in previous.reason
            )
            if pose.state == PoseState.SEATED and (
                normalized_speed >= self.config.moving_person_threshold
                or (
                    weak_seated
                    and previous.state == PoseState.STANDING
                    and not previous_downgraded
                )
            ):
                pose.state = PoseState.STANDING
                pose.reason += f";moving={normalized_speed:.3f}/s"
        self.previous_poses = list(poses)
        self.previous_pose_timestamp = timestamp

    def _crop_objects(
        self,
        frame: np.ndarray,
        tables: Sequence[Detection],
        existing_objects: Sequence[Detection],
    ) -> List[Detection]:
        """Run high-resolution second-pass detection near otherwise bare tables."""
        if not self.config.table_crop_objects or not tables:
            return list(existing_objects)
        height, width = frame.shape[:2]
        objects = list(existing_objects)
        initial_assignments = associate_objects(tables, objects)
        crop_candidates = [
            (index, table)
            for index, table in enumerate(tables)
            if not initial_assignments[index]
            and not _touches_frame_border(table.box, width, height, self.config.border_pixels)
        ]
        crop_candidates.sort(key=lambda item: item[1].confidence, reverse=True)
        for _, table in crop_candidates[: self.config.maximum_table_crops]:
            x1, y1, x2, y2 = table.box
            table_width, table_height = x2 - x1, y2 - y1
            pad_x = 0.65 * table_width
            pad_y = max(0.80 * table_height, 0.55 * table_width)
            crop_box = clip_box(
                (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
                width,
                height,
            )
            cx1, cy1, cx2, cy2 = [int(round(value)) for value in crop_box]
            if cx2 - cx1 < 64 or cy2 - cy1 < 64:
                continue
            crop = frame[cy1:cy2, cx1:cx2]
            result = self.det_model.predict(
                source=crop,
                conf=self.config.table_crop_confidence,
                imgsz=self.config.table_crop_imgsz,
                device=self.config.device,
                verbose=False,
            )[0]
            if result.boxes is None:
                continue
            for box_result in result.boxes:
                class_id = int(box_result.cls.item())
                name = str(self.names[class_id])
                confidence = float(box_result.conf.item())
                if name in EXCLUDED_OBJECT_CLASSES:
                    continue
                local = box_result.xyxy[0].tolist()
                mapped = Detection(
                    name=name,
                    box=(
                        float(local[0] + cx1),
                        float(local[1] + cy1),
                        float(local[2] + cx1),
                        float(local[3] + cy1),
                    ),
                    confidence=confidence,
                )
                if any(
                    existing.name == mapped.name
                    and overlap_over_smaller(existing.box, mapped.box) >= 0.55
                    for existing in objects
                ):
                    continue
                objects.append(mapped)
        return objects

    def analyze(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        global_motion_fraction: Point = (0.0, 0.0),
        update_temporal: bool = True,
    ) -> FrameAnalysis:
        if frame is None or frame.size == 0:
            raise ValueError("Cannot analyze an empty frame")
        height, width = frame.shape[:2]
        started = time.perf_counter()
        base_confidence = min(
            self.config.table_confidence,
            self.config.object_confidence,
            self.config.pose_confidence,
            self.config.table_rescue_confidence,
        )
        det_result = self.det_model.predict(
            source=frame,
            conf=base_confidence,
            imgsz=self.config.imgsz,
            device=self.config.device,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        table_candidates: List[Detection] = []
        objects: List[Detection] = []
        if det_result.boxes is not None:
            for box_result in det_result.boxes:
                class_id = int(box_result.cls.item())
                name = str(self.names[class_id])
                confidence = float(box_result.conf.item())
                coordinates = tuple(float(value) for value in box_result.xyxy[0].tolist())
                detection = Detection(name=name, box=coordinates, confidence=confidence)
                detections.append(detection)
                if name == "dining table":
                    if confidence >= self.config.table_rescue_confidence:
                        table_candidates.append(detection)
                elif (
                    name not in EXCLUDED_OBJECT_CLASSES
                    and confidence >= self.config.object_confidence
                ):
                    objects.append(detection)

        if self.layout is not None:
            table_boxes = [table.box for table in self.layout.tables]
            chair_boxes = self.layout.chair_boxes()
            if self.config.layout_tracking:
                if self.zone_tracker is None:
                    self.zone_tracker = LayoutZoneTracker(
                        table_boxes,
                        chair_boxes,
                        alpha=self.config.layout_track_alpha,
                    )
                # Burst side frames must not commit zone drift: applying the
                # EMA once per burst frame would multiply the tuned alpha.
                if update_temporal:
                    self.zone_tracker.update(
                        [
                            detection
                            for detection in detections
                            if detection.name == "dining table"
                            and detection.confidence
                            >= self.config.layout_track_table_confidence
                        ],
                        [
                            detection
                            for detection in detections
                            if detection.name in {"chair", "couch", "bench"}
                            and detection.confidence
                            >= self.config.layout_track_chair_confidence
                        ],
                    )
                table_boxes = self.zone_tracker.table_boxes
                chair_boxes = self.zone_tracker.chair_boxes
            tables = [
                Detection(name="dining table", box=box, confidence=1.0)
                for box in table_boxes
            ]
            seat_detections = [
                Detection(name="chair", box=box, confidence=1.0)
                for box in chair_boxes
            ]
        else:
            seat_detections = [
                detection
                for detection in detections
                if detection.name in {"chair", "couch", "bench"}
                and detection.confidence >= 0.20
            ]
            table_candidates = select_table_candidates(
                table_candidates,
                seat_detections,
                (height, width),
                table_confidence=self.config.table_confidence,
                soft_area_fraction=self.config.maximum_table_area_fraction,
                large_table_confidence=self.config.large_table_confidence,
                hard_area_fraction=self.config.hard_table_area_fraction,
                rescue_confidence=self.config.table_rescue_confidence,
            )
            tables = deduplicate_tables(table_candidates, self.config.table_overlap)

        pose_result = self.pose_model.predict(
            source=frame,
            conf=self.config.pose_confidence,
            imgsz=self.config.pose_imgsz,
            device=self.config.device,
            verbose=False,
        )[0]
        poses: List[PoseObservation] = []
        if pose_result.keypoints is not None and pose_result.boxes is not None:
            keypoints_all = pose_result.keypoints.data
            for index in range(len(keypoints_all)):
                keypoints = keypoints_all[index].tolist()
                pose_box = tuple(float(value) for value in pose_result.boxes.xyxy[index].tolist())
                pose_confidence = float(pose_result.boxes.conf[index].item())
                poses.append(
                    classify_pose(
                        keypoints,
                        pose_box,
                        pose_confidence,
                        self.config.keypoint_confidence,
                        self.config.sitting_angle,
                    )
                )

        # Compact upper-body boxes are only a seated fallback when the object
        # detector also sees a supporting seat.  This rejects staff whose lower
        # body is hidden behind a service counter.
        for pose in poses:
            support = seat_support_score(pose, seat_detections)
            if pose.reason.startswith("compact_occluded_pose"):
                pose.reason += f";seat_support={support:.2f}"
                # Compact upper-body crops are ambiguous: counter staff hidden
                # from the waist down produced a 0.91 false "chair" overlap in
                # the real cafe test.  Keep them UNKNOWN unless angle evidence
                # exists instead of turning furniture-shaped clothing into a
                # seated customer.
                pose.state = PoseState.UNKNOWN
            elif (
                pose.state == PoseState.UNKNOWN
                and support >= 0.24
                and (pose.confidence >= 0.30 or support >= 0.50)
            ):
                pose.state = PoseState.SEATED
                pose.reason += f";chair_overlap={support:.2f}"
        # Burst side frames skip the motion filter: 1/fps spacing turns box
        # jitter into spurious speed and would corrupt the pose history that
        # the per-sample (center frame) comparison relies on.
        if update_temporal:
            self._filter_moving_people(
                poses,
                timestamp,
                (height, width),
                global_motion_fraction,
            )
        suppress_conflicting_weak_seated_poses(poses)

        objects = self._crop_objects(frame, tables, objects)
        objects = filter_carried_objects(objects, poses)
        object_assignments = associate_objects(tables, objects)
        table_object_ids = {
            id(obj)
            for assigned in object_assignments.values()
            for obj in assigned
        }
        chair_object_assignments = associate_objects_to_chairs(
            seat_detections,
            [obj for obj in objects if id(obj) not in table_object_ids],
        )
        if self.layout is not None:
            # 수동 연결은 항상 신뢰: 넓은/강한 링크 구분 없이 그대로 전파한다.
            chair_table_assignments = self.layout.chair_assignments()
            strong_chair_assignments = chair_table_assignments
        else:
            chair_table_assignments = associate_chairs_to_tables(
                tables, seat_detections, (height, width)
            )
            strong_chair_assignments = filter_strong_chair_links(
                tables, seat_detections, chair_table_assignments, (height, width)
            )
        chair_people_assignments = associate_seated_people_to_chairs(
            seat_detections, poses
        )
        linked_chair_indices = {
            chair_index
            for chair_indices in strong_chair_assignments.values()
            for chair_index in chair_indices
        }
        linked_chair_person_ids = {
            id(person)
            for chair_index in linked_chair_indices
            for person in chair_people_assignments[chair_index]
        }
        direct_table_poses = [
            person for person in poses if id(person) not in linked_chair_person_ids
        ]
        people_assignments, unassigned_people = associate_people(
            tables, direct_table_poses, (height, width)
        )
        unknown_assignments, _ = associate_people(
            tables,
            direct_table_poses,
            (height, width),
            allowed_states=(PoseState.UNKNOWN,),
        )
        observations: List[TableObservation] = []
        for index, table in enumerate(tables):
            assigned_objects = object_assignments[index]
            assigned_people = people_assignments[index]
            assigned_unknown_people = unknown_assignments[index]
            connected_chair_indices = chair_table_assignments[index]
            occupied_chair_indices = [
                chair_index
                for chair_index in strong_chair_assignments[index]
                if chair_people_assignments[chair_index]
                or chair_object_assignments[chair_index]
            ]
            connected_chairs = [
                seat_detections[chair_index]
                for chair_index in connected_chair_indices
            ]
            occupied_chairs = [
                seat_detections[chair_index]
                for chair_index in occupied_chair_indices
            ]
            chair_seated_people = [
                person
                for chair_index in occupied_chair_indices
                for person in chair_people_assignments[chair_index]
            ]
            evidence_state = occupancy_state_from_evidence(
                assigned_objects,
                assigned_people,
                assigned_unknown_people,
                occupied_chairs,
            )
            # A manually drawn zone at the frame edge is intentional; the
            # border-sliver rule only guards auto-detected partial tables.
            if self.layout is None and is_severely_border_cropped(
                table.box, width, height, self.config.border_pixels
            ):
                state = OccupancyState.IGNORE
                score = table.confidence
                reason = "border_cropped"
            elif evidence_state == OccupancyState.OCCUPIED:
                state = OccupancyState.OCCUPIED
                chair_confidences = [
                    min(
                        seat_detections[chair_index].confidence,
                        max(
                            [
                                person.confidence
                                for person in chair_people_assignments[chair_index]
                            ]
                            + [
                                obj.confidence
                                for obj in chair_object_assignments[chair_index]
                            ]
                        ),
                    )
                    for chair_index in occupied_chair_indices
                ]
                evidence_confidence = max(
                    [person.confidence for person in assigned_people]
                    + [obj.confidence for obj in assigned_objects]
                    + chair_confidences
                )
                score = min(1.0, 0.55 + 0.45 * evidence_confidence)
                parts = []
                if assigned_people:
                    parts.append(f"seated:{len(assigned_people)}")
                if assigned_objects:
                    object_names = sorted({obj.name for obj in assigned_objects})
                    parts.append("objects:" + ",".join(object_names))
                if occupied_chairs:
                    parts.append(f"occupied_chairs:{len(occupied_chairs)}")
                    chair_object_names = sorted(
                        {
                            obj.name
                            for chair_index in occupied_chair_indices
                            for obj in chair_object_assignments[chair_index]
                        }
                    )
                    if chair_object_names:
                        parts.append("chair_objects:" + ",".join(chair_object_names))
                reason = ";".join(parts)
                all_seated_evidence = assigned_people + chair_seated_people
                provisional = (
                    not assigned_objects
                    and bool(all_seated_evidence)
                    and not any(
                        is_strong_seated_evidence(person)
                        for person in all_seated_evidence
                    )
                )
            elif evidence_state == OccupancyState.UNKNOWN:
                state = OccupancyState.UNKNOWN
                score = max(person.confidence for person in assigned_unknown_people)
                reason = "nearby_person_pose_unknown"
                provisional = False
            else:
                state = OccupancyState.EMPTY
                score = table.confidence
                reason = "no_customer_evidence"
                provisional = False
            if state in (OccupancyState.IGNORE,):
                provisional = False
            observations.append(
                TableObservation(
                    box=table.box,
                    table_confidence=table.confidence,
                    raw_state=state,
                    raw_score=score,
                    objects=assigned_objects,
                    seated_people=assigned_people,
                    connected_chairs=connected_chairs,
                    occupied_chairs=occupied_chairs,
                    chair_seated_people=chair_seated_people,
                    reason=reason,
                    provisional=provisional,
                    source="layout" if self.layout is not None else "detected",
                    layout_id=(
                        self.layout.tables[index].id
                        if self.layout is not None
                        else None
                    ),
                    layout_name=(
                        self.layout.tables[index].name
                        if self.layout is not None
                        else None
                    ),
                )
            )

        # A table is often completely occluded by the seated customer.  Do not
        # force that person onto an unrelated nearby table; surface an explicit
        # inferred occupied seat instead.  This remains distinguishable in logs.
        if self.config.infer_occluded_tables and self.layout is None:
            supported_people = [
                person
                for person in unassigned_people
                if id(person) not in linked_chair_person_ids
                if has_seat_support(person, seat_detections)
            ]
            for group in cluster_people(supported_people, (height, width)):
                if not group:
                    continue
                border_members = sum(
                    person.box[0] <= self.config.border_pixels
                    or person.box[2] >= width - 1 - self.config.border_pixels
                    for person in group
                )
                border_group = len(group) >= 2 and border_members >= 2
                provisional = not any(
                    is_strong_seated_evidence(person) for person in group
                )
                observations.append(
                    TableObservation(
                        box=_inferred_group_box(group, width, height),
                        table_confidence=max(person.confidence for person in group),
                        raw_state=(
                            OccupancyState.IGNORE
                            if border_group
                            else OccupancyState.OCCUPIED
                        ),
                        raw_score=max(person.confidence for person in group),
                        source="inferred-seat",
                        seated_people=list(group),
                        reason=(
                            "border_cropped_seated_group"
                            if border_group
                            else f"seated_group:{len(group)};table_occluded"
                        ),
                        provisional=provisional and not border_group,
                    )
                )

        inference_ms = (time.perf_counter() - started) * 1000.0
        return FrameAnalysis(
            timestamp=timestamp,
            tables=observations,
            poses=poses,
            detections=detections,
            inference_ms=inference_ms,
        )


def _match_observations_to_center(
    center_tables: Sequence[TableObservation],
    side_tables: Sequence[TableObservation],
    iou_threshold: float,
) -> List[Tuple[int, TableObservation]]:
    """One-to-one match of a side frame's observations onto the center's."""
    matches: List[Tuple[int, TableObservation]] = []
    used_center: set = set()
    used_side: set = set()
    layout_positions = {
        observation.layout_id: index
        for index, observation in enumerate(center_tables)
        if observation.layout_id is not None
    }
    for side_index, side in enumerate(side_tables):
        if side.layout_id is None:
            continue
        center_index = layout_positions.get(side.layout_id)
        if center_index is not None and center_index not in used_center:
            matches.append((center_index, side))
            used_center.add(center_index)
            used_side.add(side_index)
    candidate_pairs = []
    for center_index, center in enumerate(center_tables):
        if center_index in used_center or center.layout_id is not None:
            continue
        for side_index, side in enumerate(side_tables):
            if side_index in used_side or side.layout_id is not None:
                continue
            if side.source != center.source:
                continue
            iou = box_iou(center.box, side.box)
            if iou >= iou_threshold:
                candidate_pairs.append((iou, center_index, side_index))
    candidate_pairs.sort(key=lambda pair: pair[0], reverse=True)
    for _, center_index, side_index in candidate_pairs:
        if center_index in used_center or side_index in used_side:
            continue
        used_center.add(center_index)
        used_side.add(side_index)
        matches.append((center_index, side_tables[side_index]))
    return matches


def aggregate_burst_observations(
    frame_tables: Sequence[Sequence[TableObservation]],
    center_index: int,
    iou_threshold: float = 0.30,
) -> List[TableObservation]:
    """Majority-vote per-seat raw states across a burst of frames.

    The center frame's observations define the output (boxes, chair links,
    layout ids).  Each side frame contributes one vote per matched table.
    Ties — and votes without an occupied/empty majority — fall back to the
    center frame's own raw state, preserving single-frame semantics.  A
    center IGNORE (border crop / scene transition) passes through unchanged.
    Center observations are updated in place and returned.
    """
    if not frame_tables:
        return []
    center_tables = list(frame_tables[center_index])
    ballots: List[List[TableObservation]] = [
        [observation] for observation in center_tables
    ]
    for frame_position, observations in enumerate(frame_tables):
        if frame_position == center_index:
            continue
        for center_position, matched in _match_observations_to_center(
            center_tables, observations, iou_threshold
        ):
            ballots[center_position].append(matched)
    for center, ballot in zip(center_tables, ballots):
        counts: Dict[str, int] = {}
        for observation in ballot:
            key = observation.raw_state.value
            counts[key] = counts.get(key, 0) + 1
        center.vote_counts = counts
        if center.raw_state == OccupancyState.IGNORE:
            continue
        occupied_votes = counts.get(OccupancyState.OCCUPIED.value, 0)
        empty_votes = counts.get(OccupancyState.EMPTY.value, 0)
        if occupied_votes > empty_votes:
            majority = OccupancyState.OCCUPIED
        elif empty_votes > occupied_votes:
            majority = OccupancyState.EMPTY
        else:
            continue
        if majority == center.raw_state:
            continue
        donors = [
            observation
            for observation in ballot
            if observation.raw_state == majority
        ]
        donor = max(donors, key=lambda observation: observation.raw_score)
        center.raw_state = donor.raw_state
        center.raw_score = donor.raw_score
        center.reason = donor.reason
        center.provisional = donor.provisional
        center.objects = list(donor.objects)
        center.seated_people = list(donor.seated_people)
        center.occupied_chairs = list(donor.occupied_chairs)
        center.chair_seated_people = list(donor.chair_seated_people)
    return center_tables


class AdaptiveCadenceController:
    """Adaptive sample cadence for the two-stage judgment.

    1차 판단 runs at base_seconds.  When a stable-EMPTY seat gains occupied
    evidence (person sat down or an object appeared), 2차 판단 runs the next
    fast_cycles samples at fast_seconds UNCONDITIONALLY — the recheck series
    completes even if the transition confirms or is refuted earlier, so a
    detection at t=15 always yields fast results at t=20/25/30.  A fresh
    trigger while counting down re-arms the full series.
    """

    def __init__(
        self,
        base_seconds: float = 15.0,
        fast_seconds: float = 5.0,
        fast_cycles: int = 3,
    ):
        if base_seconds <= 0 or fast_seconds <= 0:
            raise ValueError("Cadence intervals must be positive")
        if fast_seconds > base_seconds:
            raise ValueError("fast_seconds cannot exceed base_seconds")
        if fast_cycles < 1:
            raise ValueError("fast_cycles must be at least 1")
        self.base_seconds = float(base_seconds)
        self.fast_seconds = float(fast_seconds)
        self.fast_cycles = int(fast_cycles)
        self.remaining_fast = 0

    @staticmethod
    def wants_fast(tracks: Sequence["Track"]) -> bool:
        return any(
            track.stable_state == OccupancyState.EMPTY
            and track.pending_state == OccupancyState.OCCUPIED
            and track.pending_count >= 1
            for track in tracks
        )

    def next_interval(self, tracks: Sequence["Track"]) -> float:
        if self.wants_fast(tracks):
            self.remaining_fast = self.fast_cycles
        elif self.remaining_fast > 0:
            self.remaining_fast -= 1
        return self.fast_seconds if self.remaining_fast > 0 else self.base_seconds


class TableTracker:
    """Stable IDs and asymmetric occupancy debouncing for visible tables."""

    def __init__(
        self,
        occupy_confirmations: int = 2,
        empty_confirmations: int = 3,
        max_missed: int = 3,
        maximum_center_distance: float = 0.16,
    ):
        self.occupy_confirmations = max(1, occupy_confirmations)
        self.empty_confirmations = max(1, empty_confirmations)
        self.max_missed = max(0, max_missed)
        self.maximum_center_distance = maximum_center_distance
        self.tracks: List[Track] = []
        self.next_id = 1

    def _predicted_center(self, track: Track, timestamp: float) -> Point:
        center = box_center(track.box)
        elapsed = max(0.0, timestamp - track.last_seen)
        return (
            center[0] + track.velocity[0] * elapsed,
            center[1] + track.velocity[1] * elapsed,
        )

    def _predicted_box(self, track: Track, timestamp: float) -> Box:
        elapsed = max(0.0, timestamp - track.last_seen)
        dx = track.velocity[0] * elapsed
        dy = track.velocity[1] * elapsed
        return (
            track.box[0] + dx,
            track.box[1] + dy,
            track.box[2] + dx,
            track.box[3] + dy,
        )

    def _match_score(
        self,
        track: Track,
        observation: TableObservation,
        frame_diagonal: float,
        timestamp: float,
    ) -> float:
        predicted_box = self._predicted_box(track, timestamp)
        iou = box_iou(predicted_box, observation.box)
        old_center = self._predicted_center(track, timestamp)
        new_center = box_center(observation.box)
        center_distance = math.hypot(old_center[0] - new_center[0], old_center[1] - new_center[1])
        distance_ratio = center_distance / max(1.0, frame_diagonal)
        shape_ratio = min(box_area(track.box), box_area(observation.box)) / max(
            1.0, max(box_area(track.box), box_area(observation.box))
        )
        distance_score = max(
            0.0,
            1.0 - distance_ratio / max(1e-6, self.maximum_center_distance),
        )
        source_bonus = 0.12 if track.last_observation.source == observation.source else 0.0
        missed_penalty = min(0.18, 0.04 * track.missed)
        return (
            0.58 * iou
            + 0.27 * distance_score
            + 0.15 * shape_ratio
            + source_bonus
            - missed_penalty
        )

    @staticmethod
    def _select_global_matches(
        pairs: Sequence[Tuple[float, int, int]],
        track_count: int,
        observation_count: int,
    ) -> List[Tuple[float, int, int]]:
        valid = [pair for pair in pairs if pair[0] >= 0.24]
        if not valid:
            return []
        scores = {(track, observation): score for score, track, observation in valid}
        size = track_count + observation_count
        invalid_cost = 1_000.0
        cost = [[0.0 for _ in range(size)] for _ in range(size)]
        for track_index in range(track_count):
            for observation_index in range(observation_count):
                score = scores.get((track_index, observation_index))
                cost[track_index][observation_index] = (
                    -score if score is not None else invalid_cost
                )

        # Hungarian algorithm for a square cost matrix. Real rows/columns are
        # padded with zero-cost dummy assignments, so unmatched tracks and new
        # observations are explicit alternatives rather than forced matches.
        u = [0.0] * (size + 1)
        v = [0.0] * (size + 1)
        p = [0] * (size + 1)
        way = [0] * (size + 1)
        for row in range(1, size + 1):
            p[0] = row
            column0 = 0
            minimum = [float("inf")] * (size + 1)
            used = [False] * (size + 1)
            while True:
                used[column0] = True
                row0 = p[column0]
                delta = float("inf")
                column1 = 0
                for column in range(1, size + 1):
                    if used[column]:
                        continue
                    current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                    if current < minimum[column]:
                        minimum[column] = current
                        way[column] = column0
                    if minimum[column] < delta:
                        delta = minimum[column]
                        column1 = column
                for column in range(size + 1):
                    if used[column]:
                        u[p[column]] += delta
                        v[column] -= delta
                    else:
                        minimum[column] -= delta
                column0 = column1
                if p[column0] == 0:
                    break
            while True:
                column1 = way[column0]
                p[column0] = p[column1]
                column0 = column1
                if column0 == 0:
                    break

        selected = []
        for column in range(1, size + 1):
            row = p[column] - 1
            observation = column - 1
            if row < track_count and observation < observation_count:
                score = scores.get((row, observation))
                if score is not None:
                    selected.append((score, row, observation))
        return selected

    def _apply_state(self, track: Track, observation: TableObservation, timestamp: float) -> Optional[Dict[str, object]]:
        raw = observation.raw_state
        previous = track.stable_state
        if raw in (OccupancyState.UNKNOWN, OccupancyState.IGNORE):
            track.pending_state = None
            track.pending_count = 0
            return None
        was_unresolved = previous in (OccupancyState.UNKNOWN, OccupancyState.IGNORE)
        if was_unresolved:
            if observation.provisional and raw == OccupancyState.OCCUPIED:
                if track.pending_state == raw:
                    track.pending_count += 1
                else:
                    track.pending_state = raw
                    track.pending_count = 1
                if track.pending_count >= self.occupy_confirmations:
                    track.stable_state = raw
                    track.pending_state = None
                    track.pending_count = 0
            else:
                track.stable_state = raw
                track.pending_state = None
                track.pending_count = 0
        elif raw == previous:
            track.pending_state = None
            track.pending_count = 0
        else:
            if track.pending_state == raw:
                track.pending_count += 1
            else:
                track.pending_state = raw
                track.pending_count = 1
            required = (
                self.occupy_confirmations
                if raw == OccupancyState.OCCUPIED
                else self.empty_confirmations
            )
            if track.pending_count >= required:
                track.stable_state = raw
                track.pending_state = None
                track.pending_count = 0
        if track.stable_state != previous:
            return {
                "type": "state_resolved" if was_unresolved else "state_change",
                "table_id": track.track_id,
                "from": previous.value,
                "to": track.stable_state.value,
                "timestamp": timestamp,
            }
        return None

    def update(
        self,
        observations: Sequence[TableObservation],
        timestamp: float,
        frame_shape: Tuple[int, int],
    ) -> TrackerUpdate:
        height, width = frame_shape
        frame_diagonal = math.hypot(width, height)
        events: List[Dict[str, object]] = []
        for track in self.tracks:
            track.visible = False
            track.predicted = False
            track.display_box = None

        pairs: List[Tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            for observation_index, observation in enumerate(observations):
                old_center = self._predicted_center(track, timestamp)
                new_center = box_center(observation.box)
                distance_ratio = math.hypot(
                    old_center[0] - new_center[0], old_center[1] - new_center[1]
                ) / max(1.0, frame_diagonal)
                source_changed = (
                    track.last_observation.source != observation.source
                )
                predicted_iou = box_iou(
                    self._predicted_box(track, timestamp), observation.box
                )
                cross_source_plausible = (
                    (
                        predicted_iou >= 0.15
                        or box_distance(
                            self._predicted_box(track, timestamp), observation.box
                        )
                        <= 0.025 * frame_diagonal
                    )
                    and distance_ratio <= 0.10
                )
                if source_changed and not cross_source_plausible:
                    continue
                if (
                    distance_ratio <= self.maximum_center_distance
                    or predicted_iou >= 0.08
                ):
                    score = self._match_score(
                        track, observation, frame_diagonal, timestamp
                    )
                    pairs.append((score, track_index, observation_index))

        assigned_tracks = set()
        assigned_observations = set()
        selected_pairs = self._select_global_matches(
            pairs, len(self.tracks), len(observations)
        )
        for score, track_index, observation_index in selected_pairs:
            track = self.tracks[track_index]
            observation = observations[observation_index]
            assigned_tracks.add(track_index)
            assigned_observations.add(observation_index)
            old_center = box_center(track.box)
            new_center = box_center(observation.box)
            elapsed = max(1e-6, timestamp - track.last_seen)
            measured_velocity = (
                (new_center[0] - old_center[0]) / elapsed,
                (new_center[1] - old_center[1]) / elapsed,
            )
            track.velocity = (
                0.45 * track.velocity[0] + 0.55 * measured_velocity[0],
                0.45 * track.velocity[1] + 0.55 * measured_velocity[1],
            )
            track.box = observation.box
            track.display_box = observation.box
            track.last_observation = observation
            track.last_seen = timestamp
            track.visible = True
            track.missed = 0
            track.real_hits += 1
            event = self._apply_state(track, observation, timestamp)
            if event:
                events.append(event)

        for observation_index, observation in enumerate(observations):
            if observation_index in assigned_observations:
                continue
            initial_state = observation.raw_state
            pending_state = None
            pending_count = 0
            if (
                observation.provisional
                and initial_state == OccupancyState.OCCUPIED
                and timestamp > 1e-6
            ):
                initial_state = OccupancyState.UNKNOWN
                pending_state = OccupancyState.OCCUPIED
                pending_count = 1
            track = Track(
                track_id=self.next_id,
                box=observation.box,
                stable_state=initial_state,
                last_observation=observation,
                first_seen=timestamp,
                last_seen=timestamp,
                display_box=observation.box,
                pending_state=pending_state,
                pending_count=pending_count,
            )
            self.next_id += 1
            self.tracks.append(track)
            events.append(
                {
                    "type": "entered_view",
                    "table_id": track.track_id,
                    "state": initial_state.value,
                    "timestamp": timestamp,
                }
            )

        surviving: List[Track] = []
        for track in self.tracks:
            if not track.visible:
                track.missed += 1
                track.pending_state = None
                track.pending_count = 0
                predicted_box = self._predicted_box(track, timestamp)
                frame_box = (0.0, 0.0, float(width - 1), float(height - 1))
                inside_fraction = intersection_area(predicted_box, frame_box) / max(
                    1.0, box_area(predicted_box)
                )
                if (
                    track.missed <= self.max_missed
                    and inside_fraction >= 0.95
                    and track.stable_state
                    in (OccupancyState.OCCUPIED, OccupancyState.EMPTY)
                    and (
                        track.last_observation.source != "inferred-seat"
                        or track.real_hits >= 2
                    )
                    and track.last_observation.reason != "scene_transition"
                    and not is_severely_border_cropped(
                        predicted_box, width, height, pixels=3
                    )
                ):
                    track.visible = True
                    track.predicted = True
                    track.display_box = predicted_box
            if track.missed <= self.max_missed:
                surviving.append(track)
            else:
                # Leaving the field of view is explicitly not an EMPTY event.
                events.append(
                    {
                        "type": "left_view",
                        "table_id": track.track_id,
                        "last_state": track.stable_state.value,
                        "timestamp": timestamp,
                    }
                )
        self.tracks = surviving
        return TrackerUpdate(
            visible_tracks=sorted(
                [track for track in self.tracks if track.visible],
                key=lambda track: track.track_id,
            ),
            all_tracks=list(self.tracks),
            events=events,
        )


def _parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(value)


def scene_change_metrics(previous: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """Measure whether two sampled frames can be related by normal camera motion."""
    if previous is None or current is None or previous.size == 0 or current.size == 0:
        return {
            "histogram_correlation": 1.0,
            "mean_difference": 0.0,
            "orb_matches": 0.0,
            "orb_inlier_ratio": 0.0,
            "global_dx_fraction": 0.0,
            "global_dy_fraction": 0.0,
        }
    size = (320, 180)
    previous_small = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
    current_small = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
    previous_gray = cv2.cvtColor(previous_small, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_small, cv2.COLOR_BGR2GRAY)
    mean_difference = float(np.mean(cv2.absdiff(previous_gray, current_gray))) / 255.0

    previous_hist = cv2.calcHist([previous_small], [0, 1], None, [32, 32], [0, 256, 0, 256])
    current_hist = cv2.calcHist([current_small], [0, 1], None, [32, 32], [0, 256, 0, 256])
    cv2.normalize(previous_hist, previous_hist)
    cv2.normalize(current_hist, current_hist)
    correlation = float(
        cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_CORREL)
    )

    orb = cv2.ORB_create(nfeatures=800)
    previous_keypoints, previous_descriptors = orb.detectAndCompute(previous_gray, None)
    current_keypoints, current_descriptors = orb.detectAndCompute(current_gray, None)
    good_matches = []
    if previous_descriptors is not None and current_descriptors is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        for candidates in matcher.knnMatch(previous_descriptors, current_descriptors, k=2):
            if len(candidates) == 2 and candidates[0].distance < 0.75 * candidates[1].distance:
                good_matches.append(candidates[0])

    inlier_ratio = 0.0
    global_dx_fraction = 0.0
    global_dy_fraction = 0.0
    if len(good_matches) >= 8:
        source = np.float32(
            [previous_keypoints[match.queryIdx].pt for match in good_matches]
        )
        destination = np.float32(
            [current_keypoints[match.trainIdx].pt for match in good_matches]
        )
        _, mask = cv2.findHomography(source, destination, cv2.RANSAC, 4.0)
        if mask is not None:
            inlier_ratio = float(mask.mean())
            inliers = mask.reshape(-1).astype(bool)
            if np.any(inliers):
                displacement = destination[inliers] - source[inliers]
                global_dx_fraction = float(np.median(displacement[:, 0])) / size[0]
                global_dy_fraction = float(np.median(displacement[:, 1])) / size[1]
    return {
        "histogram_correlation": correlation,
        "mean_difference": mean_difference,
        "orb_matches": float(len(good_matches)),
        "orb_inlier_ratio": inlier_ratio,
        "global_dx_fraction": global_dx_fraction,
        "global_dy_fraction": global_dy_fraction,
    }


def is_scene_change(previous: np.ndarray, current: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    metrics = scene_change_metrics(previous, current)
    changed = (
        metrics["orb_matches"] < 20.0
        and metrics["histogram_correlation"] < 0.85
        and metrics["mean_difference"] > 0.12
    )
    return changed, metrics


def require_ffmpeg() -> Tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must be installed and available on PATH")
    return ffmpeg, ffprobe


def probe_video(path: Path) -> VideoInfo:
    _, ffprobe = require_ffmpeg()
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out for {path}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    stream = streams[0]
    frames_value = stream.get("nb_frames")
    frames = int(frames_value) if frames_value and frames_value != "N/A" else None
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=_parse_rate(stream.get("avg_frame_rate", "0/0")),
        duration=float((payload.get("format") or {}).get("duration") or 0.0),
        codec=str(stream.get("codec_name") or "unknown"),
        source_frames=frames,
    )


def _read_exact(stream, byte_count: int) -> bytes:
    chunks: List[bytes] = []
    remaining = byte_count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _wait_process(process: subprocess.Popen, timeout: float = 15.0) -> int:
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=5.0)


def _read_stderr_file(stderr_file) -> bytes:
    stderr_file.flush()
    stderr_file.seek(0)
    return stderr_file.read()


class FFmpegSampleReader:
    def __init__(
        self,
        path: Path,
        sample_seconds: float,
        info: Optional[VideoInfo] = None,
        start_seconds: float = 0.0,
    ):
        if sample_seconds <= 0:
            raise ValueError("sample_seconds must be positive")
        if start_seconds < 0:
            raise ValueError("start_seconds cannot be negative")
        self.path = Path(path)
        self.sample_seconds = float(sample_seconds)
        self.start_seconds = float(start_seconds)
        self.info = info or probe_video(self.path)
        self.process: Optional[subprocess.Popen] = None
        self.stderr_file = None

    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        ffmpeg, _ = require_ffmpeg()
        command = [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(self.path),
            "-ss",
            f"{self.start_seconds:.9f}",
            "-vf",
            f"fps=1/{self.sample_seconds:.9f}",
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        self.stderr_file = tempfile.TemporaryFile(mode="w+b")
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
        )
        assert self.process.stdout is not None
        frame_bytes = self.info.width * self.info.height * 3
        frame_index = 0
        reached_eof = False
        try:
            while True:
                payload = _read_exact(self.process.stdout, frame_bytes)
                if not payload:
                    reached_eof = True
                    break
                if len(payload) != frame_bytes:
                    raise RuntimeError(
                        f"ffmpeg returned a partial frame ({len(payload)}/{frame_bytes} bytes)"
                    )
                frame = np.frombuffer(payload, dtype=np.uint8).reshape(
                    (self.info.height, self.info.width, 3)
                ).copy()
                yield (
                    frame_index,
                    self.start_seconds + frame_index * self.sample_seconds,
                    frame,
                )
                frame_index += 1
        finally:
            # Closing a generator early (e.g. --max-samples) is intentional.
            # Stop ffmpeg before closing stdout so its expected broken pipe does
            # not leak as an "Exception ignored in generator" traceback.
            if not reached_eof and self.process.poll() is None:
                self.process.terminate()
            if self.process.stdout:
                self.process.stdout.close()
            return_code = _wait_process(self.process)
            stderr = _read_stderr_file(self.stderr_file) if self.stderr_file else b""
            if self.stderr_file:
                self.stderr_file.close()
                self.stderr_file = None
            if reached_eof and return_code != 0:
                message = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg decode failed: {message}")


class FFmpegBurstReader:
    """Seek-based reader returning bursts of consecutive native-fps frames.

    Each burst feeds the per-sample majority vote, so timestamps are
    synthesized from the requested position instead of container PTS —
    the same policy FFmpegSampleReader uses for its fps-filtered stream.
    """

    def __init__(self, path: Path, info: Optional[VideoInfo] = None):
        self.path = Path(path)
        self.info = info or probe_video(self.path)
        if self.info.fps <= 0:
            raise ValueError("Video fps must be positive for burst reading")

    def read_burst(
        self, center_seconds: float, n: int
    ) -> Tuple[int, List[Tuple[float, np.ndarray]]]:
        """Read up to 2n+1 consecutive frames centred on center_seconds.

        Returns (center_index, [(timestamp, frame), ...]).  Bursts are
        truncated near the end of the video; near the start they shift right
        (the seek clamps to 0) and center_index moves accordingly.  An empty
        list means no frame could be decoded at that position.
        """
        if n < 0:
            raise ValueError("n cannot be negative")
        if center_seconds < 0:
            raise ValueError("center_seconds cannot be negative")
        ffmpeg, _ = require_ffmpeg()
        frame_interval = 1.0 / self.info.fps
        start_seconds = max(0.0, center_seconds - n * frame_interval)
        frame_count = 2 * n + 1
        command = [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{start_seconds:.9f}",
            "-i",
            str(self.path),
            "-frames:v",
            str(frame_count),
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        stderr_file = tempfile.TemporaryFile(mode="w+b")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
        assert process.stdout is not None
        frame_bytes = self.info.width * self.info.height * 3
        frames: List[Tuple[float, np.ndarray]] = []
        try:
            for index in range(frame_count):
                payload = _read_exact(process.stdout, frame_bytes)
                if not payload:
                    break
                if len(payload) != frame_bytes:
                    raise RuntimeError(
                        f"ffmpeg returned a partial frame ({len(payload)}/{frame_bytes} bytes)"
                    )
                frame = np.frombuffer(payload, dtype=np.uint8).reshape(
                    (self.info.height, self.info.width, 3)
                ).copy()
                frames.append((start_seconds + index * frame_interval, frame))
        finally:
            if process.poll() is None:
                process.terminate()
            process.stdout.close()
            return_code = _wait_process(process)
            stderr = _read_stderr_file(stderr_file)
            stderr_file.close()
        if not frames and return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg burst decode failed: {message}")
        if not frames:
            return 0, []
        center_index = min(
            range(len(frames)),
            key=lambda index: abs(frames[index][0] - center_seconds),
        )
        return center_index, frames


class FFmpegVideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 20):
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("Invalid output video dimensions or frame rate")
        ffmpeg, _ = require_ffmpeg()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        command = [
            ffmpeg,
            "-y",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.9f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self.stderr_file = tempfile.TemporaryFile(mode="w+b")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=self.stderr_file,
        )
        self.closed = False
        self.last_stderr = ""

    def write(self, frame: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to a closed video writer")
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame shape {frame.shape[:2]} does not match {(self.height, self.width)}"
            )
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except OSError as exc:
            # Windows surfaces a dead pipe as EINVAL rather than
            # BrokenPipeError; either way the real cause is in ffmpeg stderr
            # (e.g. "Permission denied" when the output file is locked by a
            # media player).
            if self.process.poll() is None:
                _wait_process(self.process)
            details = _read_stderr_file(self.stderr_file).decode(
                "utf-8", errors="replace"
            ).strip()
            self.last_stderr = details
            message = "ffmpeg encoder closed its input unexpectedly"
            if details:
                message += f": {details}"
            raise RuntimeError(message) from exc

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        return_code = _wait_process(self.process)
        stderr = _read_stderr_file(self.stderr_file)
        self.last_stderr = stderr.decode("utf-8", errors="replace").strip()
        self.stderr_file.close()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg encode failed: {self.last_stderr}")

    def __enter__(self) -> "FFmpegVideoWriter":
        return self

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        _wait_process(self.process)
        self.last_stderr = _read_stderr_file(self.stderr_file).decode(
            "utf-8", errors="replace"
        ).strip()
        self.stderr_file.close()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()


STATE_COLORS = {
    OccupancyState.OCCUPIED: (50, 55, 235),
    OccupancyState.EMPTY: (60, 190, 70),
    OccupancyState.UNKNOWN: (155, 155, 155),
    OccupancyState.IGNORE: (80, 185, 230),
}


def _draw_label(frame: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(frame.shape[1], frame.shape[0]) / 1300.0)
    thickness = max(1, int(round(scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = max(0, min(frame.shape[1] - text_width - 6, x))
    y = max(text_height + baseline + 6, min(frame.shape[0] - 2, y))
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 6),
        (x + text_width + 6, y + 2),
        color,
        -1,
    )
    cv2.putText(frame, text, (x + 3, y - baseline - 1), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_frame(
    frame: np.ndarray,
    analysis: FrameAnalysis,
    update: TrackerUpdate,
    debug: bool = False,
    cadence: Optional[str] = None,
) -> np.ndarray:
    output = frame.copy()
    occupied = sum(track.visible_state == OccupancyState.OCCUPIED for track in update.visible_tracks)
    empty = sum(track.visible_state == OccupancyState.EMPTY for track in update.visible_tracks)
    ignored = sum(track.visible_state == OccupancyState.IGNORE for track in update.visible_tracks)
    inferred = sum(track.last_observation.source == "inferred-seat" for track in update.visible_tracks)
    occupied_chairs = sum(
        len(track.last_observation.occupied_chairs) for track in update.visible_tracks
    )

    overlay = output.copy()
    header_height = max(72, int(output.shape[0] * 0.085))
    cv2.rectangle(overlay, (0, 0), (output.shape[1], header_height), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.78, output, 0.22, 0, output)
    scale = max(0.65, min(output.shape[1], output.shape[0]) / 1150.0)
    title = f"SeatNow  t={analysis.timestamp:05.1f}s"
    if cadence == "fast":
        title += "  [FAST RECHECK]"
    cv2.putText(
        output,
        title,
        (18, int(header_height * 0.43)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        max(1, int(scale * 2)),
        cv2.LINE_AA,
    )
    if analysis.scene_change:
        _draw_label(output, "SCENE CHANGE - RESET / IGNORE", (output.shape[1] // 2, header_height), STATE_COLORS[OccupancyState.IGNORE])
    cv2.putText(
        output,
        f"visible={len(update.visible_tracks)}  occupied={occupied}  empty={empty}  ignore={ignored}  inferred={inferred}  chairs={occupied_chairs}  inference={analysis.inference_ms:.0f}ms",
        (18, int(header_height * 0.80)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale * 0.75,
        (215, 220, 230),
        max(1, int(scale * 1.6)),
        cv2.LINE_AA,
    )

    for track in update.visible_tracks:
        observation = track.last_observation
        color = STATE_COLORS[track.visible_state]
        x1, y1, x2, y2 = [int(round(value)) for value in track.current_box]
        thickness = 1 if track.predicted else (4 if observation.source == "inferred-seat" else 3)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        evidence = observation.reason
        if len(evidence) > 42:
            evidence = evidence[:39] + "..."
        label = (
            f"{track.label} {track.visible_state.value.upper()} "
            f"conf={observation.raw_score:.2f} table={observation.table_confidence:.2f}"
        )
        if track.predicted:
            label += " PREDICTED"
        if evidence:
            label += f" | {evidence}"
        _draw_label(output, label, (x1, max(header_height + 4, y1)), color)
        if debug and observation.source in ("detected", "layout"):
            surface = table_surface_box(track.current_box)
            sx1, sy1, sx2, sy2 = [int(round(value)) for value in surface]
            cv2.rectangle(output, (sx1, sy1), (sx2, sy2), (230, 170, 30), 1)
            table_center = tuple(int(round(value)) for value in box_center(track.current_box))
            for chair in observation.connected_chairs:
                chair_occupied = chair in observation.occupied_chairs
                chair_color = (30, 60, 235) if chair_occupied else (30, 210, 235)
                cx1, cy1, cx2, cy2 = [int(round(value)) for value in chair.box]
                cv2.rectangle(output, (cx1, cy1), (cx2, cy2), chair_color, 2)
                chair_center = tuple(int(round(value)) for value in box_center(chair.box))
                cv2.line(output, table_center, chair_center, chair_color, 1, cv2.LINE_AA)
                _draw_label(
                    output,
                    f"{chair.name} {'OCCUPIED' if chair_occupied else 'FREE'}",
                    (cx1, cy1),
                    chair_color,
                )

    if debug:
        for pose in analysis.poses:
            color = {
                PoseState.SEATED: (220, 70, 210),
                PoseState.STANDING: (200, 165, 70),
                PoseState.UNKNOWN: (140, 140, 140),
            }[pose.state]
            x1, y1, x2, y2 = [int(round(value)) for value in pose.box]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
            cv2.circle(output, (int(pose.anchor[0]), int(pose.anchor[1])), 5, color, -1)
            _draw_label(output, f"pose {pose.state.value}", (x1, y1), color)
    return output


def track_to_dict(track: Track) -> Dict[str, object]:
    observation = track.last_observation
    data: Dict[str, object] = {
        "id": track.track_id,
        "label": track.label,
        "source": observation.source,
        "layout_id": observation.layout_id,
        "layout_name": observation.layout_name,
        "box": [round(value, 2) for value in track.current_box],
        "predicted": track.predicted,
        "state": track.visible_state.value,
        "persistent_state": track.stable_state.value,
        "raw_state": observation.raw_state.value,
        "confidence": round(observation.raw_score, 4),
        "table_confidence": round(observation.table_confidence, 4),
        "reason": (
            f"temporarily_occluded:{observation.reason}"
            if track.predicted
            else observation.reason
        ),
        "provisional": observation.provisional,
        "objects": [
            {"class": obj.name, "confidence": round(obj.confidence, 4)}
            for obj in observation.objects
        ],
        "seated_people": len(observation.seated_people),
        "connected_chairs": [
            {
                "class": chair.name,
                "box": [round(value, 2) for value in chair.box],
                "confidence": round(chair.confidence, 4),
                "occupied": chair in observation.occupied_chairs,
            }
            for chair in observation.connected_chairs
        ],
        "occupied_chairs": len(observation.occupied_chairs),
        "chair_seated_people": len(observation.chair_seated_people),
        "pending_state": track.pending_state.value if track.pending_state else None,
        "pending_count": track.pending_count,
    }
    # Only burst-vote runs carry vote counts; keep legacy records unchanged.
    if observation.vote_counts is not None:
        data["vote_counts"] = observation.vote_counts
    return data


def frame_log_record(
    frame_index: int,
    analysis: FrameAnalysis,
    update: TrackerUpdate,
) -> Dict[str, object]:
    visible = update.visible_tracks
    return {
        "frame_index": frame_index,
        "timestamp": round(analysis.timestamp, 6),
        "inference_ms": round(analysis.inference_ms, 2),
        "scene_change": analysis.scene_change,
        "scene_metrics": {
            key: round(value, 4) for key, value in analysis.scene_metrics.items()
        },
        "summary": {
            "visible": len(visible),
            "occupied": sum(track.visible_state == OccupancyState.OCCUPIED for track in visible),
            "empty": sum(track.visible_state == OccupancyState.EMPTY for track in visible),
            "unknown": sum(track.visible_state == OccupancyState.UNKNOWN for track in visible),
            "ignore": sum(track.visible_state == OccupancyState.IGNORE for track in visible),
            "inferred_seats": sum(
                track.last_observation.source == "inferred-seat" for track in visible
            ),
            "occupied_chairs": sum(
                len(track.last_observation.occupied_chairs) for track in visible
            ),
            "seated_poses": sum(pose.state == PoseState.SEATED for pose in analysis.poses),
            "standing_poses": sum(pose.state == PoseState.STANDING for pose in analysis.poses),
            "unknown_poses": sum(pose.state == PoseState.UNKNOWN for pose in analysis.poses),
        },
        "tables": [track_to_dict(track) for track in visible],
        "poses": [
            {
                "state": pose.state.value,
                "box": [round(value, 2) for value in pose.box],
                "anchor": [round(value, 2) for value in pose.anchor],
                "confidence": round(pose.confidence, 4),
                "reason": pose.reason,
                "angles": {key: round(value, 2) for key, value in pose.angles.items()},
            }
            for pose in analysis.poses
        ],
        "events": update.events,
    }

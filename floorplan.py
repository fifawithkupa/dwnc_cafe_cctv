"""The customer-facing map: where each seat is, flattened onto the floor.

``seat_report`` says what state a seat is in and nothing about where it is, so
an app can print "3 free" but cannot show which three.  This module makes the
missing half: a static map drawn once at install, joined to the live states
by ``seat_id``.

Positions here are drawing units, not metres.  Nobody is asked to measure the
room -- a customer needs "the window seat on the right", not "3.2 m from the
door" -- and the proportions get corrected by hand in the editor.

``image_anchor`` is kept on every seat and chair.  It is the camera-image
point the position was projected from, so it survives edits and identifies a
chair even after the editor moves it between tables (the layout's own chair
ordering does not).  It is also half of a correspondence pair: once a person
has corrected the map, "here in the image, here on the floor" is known for
every seat, which is a far better fit than the four clicked points.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from floor_projection import FloorProjectionError, build_transform, floor_anchor
from seatnow_layout import COUNTED_ZONE_KIND, SeatLayout


FLOORPLAN_SCHEMA_VERSION = 1
EXTENT_LONG_SIDE = 1000.0
MARGIN_FRACTION = 0.08

# Real sizes are unknown, so everything starts at a readable default and the
# person adjusts.  A bar seat is drawn smaller than a table because it is one
# seat rather than a whole table.
DEFAULT_SIZES: Dict[str, Tuple[float, float]] = {
    "table": (150.0, 105.0),
    COUNTED_ZONE_KIND: (52.0, 52.0),
    "chair": (46.0, 46.0),
}

Point = Tuple[float, float]


@dataclass(frozen=True)
class FloorSeat:
    seat_id: str
    kind: str
    x: float
    y: float
    w: float
    h: float
    image_anchor: Point
    needs_review: bool = False


@dataclass(frozen=True)
class FloorChair:
    seat_id: Optional[str]
    x: float
    y: float
    w: float
    h: float
    image_anchor: Point
    needs_review: bool = False


@dataclass(frozen=True)
class Landmark:
    kind: str
    label: str
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class FloorPlan:
    schema_version: int
    extent: Tuple[float, float]
    seats: Tuple[FloorSeat, ...]
    chairs: Tuple[FloorChair, ...]
    landmarks: Tuple[Landmark, ...] = ()


def _owner_of_each_chair(layout: SeatLayout) -> List[Tuple[Optional[str], Point]]:
    """(owner seat name or None, image anchor) for every chair in the layout.

    Ownership is read from ``unit_chair_assignments`` rather than recomputed,
    because that is what judgement uses.  A bar chair belongs to the one slot
    it covers; hanging every bar chair under the first slot -- which this did
    at first -- draws ownership lines to the wrong stool, and the installer
    then "fixes" something that was already right.
    """
    units = layout.judgement_units()
    owner_by_chair: Dict[int, str] = {}
    for unit_index, chair_indices in layout.unit_chair_assignments().items():
        for chair_index in chair_indices:
            owner_by_chair[chair_index] = units[unit_index].name
    return [
        (owner_by_chair.get(index), floor_anchor(box))
        for index, box in enumerate(layout.chair_boxes())
    ]


def build_draft(layout: SeatLayout) -> FloorPlan:
    """Project every seat and chair onto the floor and fit them to a canvas."""
    if layout.floor_reference is None:
        raise FloorProjectionError(
            "바닥 기준점이 없어 평면도를 만들 수 없습니다 — calibrate.py 에서 "
            "[f]로 바닥의 직사각형 네 귀퉁이를 찍고 저장하세요"
        )
    frame_size = (
        int(layout.source.get("width", 1920)),
        int(layout.source.get("height", 1080)),
    )
    transform = build_transform(layout.floor_reference.image_points, frame_size)

    units = layout.judgement_units()
    seat_anchors = [(unit, floor_anchor(unit.box)) for unit in units]
    chair_owners = _owner_of_each_chair(layout)

    projected: List[Optional[Point]] = [
        transform.project(anchor) for _, anchor in seat_anchors
    ] + [transform.project(anchor) for _, anchor in chair_owners]

    placed = [point for point in projected if point is not None]
    if not placed:
        raise FloorProjectionError(
            "좌석이 하나도 바닥 평면 위로 오지 않았습니다 — 바닥 네 점을 "
            "다시 찍어 주세요"
        )

    min_x = min(point[0] for point in placed)
    max_x = max(point[0] for point in placed)
    min_y = min(point[1] for point in placed)
    max_y = max(point[1] for point in placed)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    scale = EXTENT_LONG_SIDE / max(span_x, span_y)
    margin = MARGIN_FRACTION * EXTENT_LONG_SIDE
    extent = (span_x * scale + 2 * margin, span_y * scale + 2 * margin)

    def place(point: Optional[Point]) -> Tuple[float, float, bool]:
        """Canvas position, and whether a person has to look at it.

        A point past the vanishing line is not a place on the floor; it is
        parked at the corner and flagged rather than dropped, so the seat
        stays visible and gets moved instead of silently disappearing.
        """
        if point is None:
            return margin, margin, True
        return (
            (point[0] - min_x) * scale + margin,
            (point[1] - min_y) * scale + margin,
            False,
        )

    seats: List[FloorSeat] = []
    for (unit, anchor), point in zip(seat_anchors, projected[: len(seat_anchors)]):
        x, y, review = place(point)
        w, h = DEFAULT_SIZES.get(unit.kind, DEFAULT_SIZES["table"])
        seats.append(
            FloorSeat(
                seat_id=unit.name,
                kind=unit.kind,
                x=x,
                y=y,
                w=w,
                h=h,
                image_anchor=anchor,
                needs_review=review,
            )
        )

    chairs: List[FloorChair] = []
    for (owner, anchor), point in zip(chair_owners, projected[len(seat_anchors):]):
        x, y, review = place(point)
        w, h = DEFAULT_SIZES["chair"]
        chairs.append(
            FloorChair(
                seat_id=owner,
                x=x,
                y=y,
                w=w,
                h=h,
                image_anchor=anchor,
                needs_review=review,
            )
        )

    return FloorPlan(
        schema_version=FLOORPLAN_SCHEMA_VERSION,
        extent=extent,
        seats=tuple(seats),
        chairs=tuple(chairs),
        landmarks=(),
    )


def save_floorplan(plan: FloorPlan, path: Path) -> None:
    payload = {
        "schema_version": plan.schema_version,
        "extent": {
            "width": round(plan.extent[0], 2),
            "height": round(plan.extent[1], 2),
        },
        "seats": [
            {
                "seat_id": seat.seat_id,
                "kind": seat.kind,
                "x": round(seat.x, 2),
                "y": round(seat.y, 2),
                "w": round(seat.w, 2),
                "h": round(seat.h, 2),
                "image_anchor": [
                    round(seat.image_anchor[0], 2),
                    round(seat.image_anchor[1], 2),
                ],
                "needs_review": seat.needs_review,
            }
            for seat in plan.seats
        ],
        "chairs": [
            {
                "seat_id": chair.seat_id,
                "x": round(chair.x, 2),
                "y": round(chair.y, 2),
                "w": round(chair.w, 2),
                "h": round(chair.h, 2),
                "image_anchor": [
                    round(chair.image_anchor[0], 2),
                    round(chair.image_anchor[1], 2),
                ],
                "needs_review": chair.needs_review,
            }
            for chair in plan.chairs
        ],
        "landmarks": [
            {
                "kind": landmark.kind,
                "label": landmark.label,
                "x": round(landmark.x, 2),
                "y": round(landmark.y, 2),
                "w": round(landmark.w, 2),
                "h": round(landmark.h, 2),
            }
            for landmark in plan.landmarks
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_floorplan(path: Path) -> FloorPlan:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FloorPlan(
        schema_version=int(data["schema_version"]),
        extent=(float(data["extent"]["width"]), float(data["extent"]["height"])),
        seats=tuple(
            FloorSeat(
                seat_id=str(seat["seat_id"]),
                kind=str(seat["kind"]),
                x=float(seat["x"]),
                y=float(seat["y"]),
                w=float(seat["w"]),
                h=float(seat["h"]),
                image_anchor=(
                    float(seat["image_anchor"][0]),
                    float(seat["image_anchor"][1]),
                ),
                needs_review=bool(seat.get("needs_review", False)),
            )
            for seat in data["seats"]
        ),
        chairs=tuple(
            FloorChair(
                seat_id=chair["seat_id"],
                x=float(chair["x"]),
                y=float(chair["y"]),
                w=float(chair["w"]),
                h=float(chair["h"]),
                image_anchor=(
                    float(chair["image_anchor"][0]),
                    float(chair["image_anchor"][1]),
                ),
                needs_review=bool(chair.get("needs_review", False)),
            )
            for chair in data["chairs"]
        ),
        landmarks=tuple(
            Landmark(
                kind=str(landmark["kind"]),
                label=str(landmark["label"]),
                x=float(landmark["x"]),
                y=float(landmark["y"]),
                w=float(landmark["w"]),
                h=float(landmark["h"]),
            )
            for landmark in data["landmarks"]
        ),
    )

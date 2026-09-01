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
import math
from dataclasses import dataclass, replace
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
class FloorCounter:
    """One long bar table.  Its stools are the counted_zone seats beside it."""

    zone_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    depth: float


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
    counters: Tuple[FloorCounter, ...] = ()
    landmarks: Tuple[Landmark, ...] = ()
    # The room outline, clicked once by the installer.  A camera sees seats,
    # never walls, so there is no other source for it -- and without walls a
    # customer sees floating boxes rather than a cafe.
    walls: Tuple[Point, ...] = ()


# How chairs fill the sides of a table, in order.  Five chairs come out as
# two above, two below and one on the right -- which is what the real T6
# looks like, and the shape a customer reads as "five-seater".
SIDE_ORDER = ("top", "bottom", "top", "bottom", "right", "left")
CHAIR_GAP = 0.55  # of a chair, between the table edge and the chair


def _side_positions(seat: "FloorSeat", side: str, count: int, size: float):
    """Evenly spaced points along one side of a table, just outside it."""
    gap = size * CHAIR_GAP
    points = []
    for index in range(count):
        offset = (index - (count - 1) / 2.0) * size * 1.25
        if side == "top":
            points.append((seat.x + offset, seat.y - seat.h / 2 - gap - size / 2))
        elif side == "bottom":
            points.append((seat.x + offset, seat.y + seat.h / 2 + gap + size / 2))
        elif side == "right":
            points.append((seat.x + seat.w / 2 + gap + size / 2, seat.y + offset))
        else:
            points.append((seat.x - seat.w / 2 - gap - size / 2, seat.y + offset))
    return points


def arrange_chairs(
    seats: Tuple["FloorSeat", ...], chairs: Tuple["FloorChair", ...]
) -> Tuple["FloorChair", ...]:
    """Place every chair around the seat it belongs to.

    A customer reads "this is a four-seater", never the exact centimetres of
    a stool, and the projected chair positions scatter into something that
    looks like nothing.  Judgement is unaffected either way: it reads the
    camera-image boxes and never these.

    A chair with no owner keeps its projected place -- it is a loose chair,
    and pretending otherwise would hide that.
    """
    by_owner: Dict[str, List[int]] = {}
    for index, chair in enumerate(chairs):
        if chair.seat_id:
            by_owner.setdefault(chair.seat_id, []).append(index)

    placed = list(chairs)
    for seat in seats:
        indices = by_owner.get(seat.seat_id, [])
        if not indices:
            continue
        per_side: Dict[str, int] = {}
        for position in range(len(indices)):
            side = SIDE_ORDER[position % len(SIDE_ORDER)]
            per_side[side] = per_side.get(side, 0) + 1
        cursor = 0
        for side in ("top", "bottom", "right", "left"):
            count = per_side.get(side, 0)
            if not count:
                continue
            size = placed[indices[cursor]].w
            for (x, y) in _side_positions(seat, side, count, size):
                placed[indices[cursor]] = replace(
                    placed[indices[cursor]], x=x, y=y, needs_review=False
                )
                cursor += 1
    return tuple(placed)


COUNTER_DEPTH = 46.0
STOOL_GAP = 0.5  # of a stool, between counter edge and stool
# A table needs room for the chairs drawn around it, not just for itself.
# Room for a chair *and* the gap it sits at, on both sides -- the chair
# reaches h/2 + gap + size from the table centre, and clearing less than
# that leaves chairs inside whatever the table just cleared.
CHAIR_CLEARANCE = DEFAULT_SIZES["chair"][0] * (CHAIR_GAP + 1.0) * 2


def _zone_rows(seats):
    rows = {}
    for index, seat in enumerate(seats):
        if seat.kind == COUNTED_ZONE_KIND:
            rows.setdefault(seat.seat_id.rsplit("-", 1)[0], []).append(index)
    return rows


def arrange_bars(seats):
    """Turn each bar row into one long table with stools beside it.

    The line the projected slots lie on is real -- the counter runs that way.
    The gaps between them are not: they wobble, so the row comes out crooked.
    So the counter is that line, the stools are spaced evenly along it, and
    they are pushed off it to the room side, because a stool drawn on the
    counter reads as a pile rather than a bar.

    The counter is produced here rather than derived back from the moved
    stools: recomputing it afterwards has to guess which way they were pushed,
    and guessing wrong puts the counter straight through them.
    """
    placed = list(seats)
    counters = []
    others = [(seat.x, seat.y) for seat in seats if seat.kind != COUNTED_ZONE_KIND]

    for zone_id, indices in _zone_rows(tuple(placed)).items():
        if len(indices) < 2:
            continue
        points = [(placed[index].x, placed[index].y) for index in indices]
        start_point = min(points, key=lambda point: (point[1], point[0]))
        end_point = max(points, key=lambda point: (point[1], point[0]))
        length = math.hypot(
            end_point[0] - start_point[0], end_point[1] - start_point[1]
        ) or 1.0
        direction = (
            (end_point[0] - start_point[0]) / length,
            (end_point[1] - start_point[1]) / length,
        )
        normal = (-direction[1], direction[0])
        middle = (
            (start_point[0] + end_point[0]) / 2.0,
            (start_point[1] + end_point[1]) / 2.0,
        )
        if others:
            room = (
                sum(point[0] for point in others) / len(others),
                sum(point[1] for point in others) / len(others),
            )
            toward = (room[0] - middle[0]) * normal[0] + (room[1] - middle[1]) * normal[1]
            if toward < 0:
                normal = (-normal[0], -normal[1])

        counters.append(
            FloorCounter(
                zone_id=zone_id,
                x1=start_point[0],
                y1=start_point[1],
                x2=end_point[0],
                y2=end_point[1],
                depth=COUNTER_DEPTH,
            )
        )

        ordered = sorted(
            indices,
            key=lambda index: (placed[index].x - start_point[0]) * direction[0]
            + (placed[index].y - start_point[1]) * direction[1],
        )
        steps = len(ordered) - 1
        for position, index in enumerate(ordered):
            ratio = position / steps
            size = placed[index].w
            offset = COUNTER_DEPTH / 2 + size * STOOL_GAP + size / 2
            placed[index] = replace(
                placed[index],
                x=start_point[0] + (end_point[0] - start_point[0]) * ratio + normal[0] * offset,
                y=start_point[1] + (end_point[1] - start_point[1]) * ratio + normal[1] * offset,
                needs_review=False,
            )
    return tuple(placed), tuple(counters)


def counter_boxes(counter):
    """The counter as a chain of small squares along its length.

    One axis-aligned box around a diagonal counter covers a huge rectangle of
    empty floor, and everything near it is reported as overlapping when
    nothing is.  Walking the segment follows the shape closely enough while
    keeping every check a plain box-against-box.
    """
    length = math.hypot(counter.x2 - counter.x1, counter.y2 - counter.y1)
    steps = max(1, int(length / (counter.depth / 2)))
    return [
        (
            counter.x1 + (counter.x2 - counter.x1) * step / steps,
            counter.y1 + (counter.y2 - counter.y1) * step / steps,
            counter.depth,
            counter.depth,
        )
        for step in range(steps + 1)
    ]


def _escape(box, blockers):
    """Cheapest (x, y) that clears every blocker, or None if already clear.

    Resolving one blocker at a time pushes a table off one stool straight
    onto the next and it oscillates: a bar row is a line of obstacles, and
    stepping along that line never escapes it.  Costing all four directions
    against the whole group picks the way out in one move.
    """
    x, y, w, h = box
    hits = [
        other
        for other in blockers
        if abs(x - other[0]) < (w + other[2]) / 2
        and abs(y - other[1]) < (h + other[3]) / 2
    ]
    if not hits:
        return None
    right = max(other[0] + (w + other[2]) / 2 for other in hits) + 1.0
    left = min(other[0] - (w + other[2]) / 2 for other in hits) - 1.0
    down = max(other[1] + (h + other[3]) / 2 for other in hits) + 1.0
    up = min(other[1] - (h + other[3]) / 2 for other in hits) - 1.0
    options = [
        (abs(right - x), (right, y)),
        (abs(x - left), (left, y)),
        (abs(down - y), (x, down)),
        (abs(y - up), (x, up)),
    ]
    return min(options, key=lambda option: option[0])[1]


def separate_overlaps(seats, counters=(), rounds=200):
    """Nudge tables apart so nothing drawn sits on anything else drawn.

    Only plain tables move.  A stool was put where the bar rule says it goes,
    and shoving it would bend the row the rule just made straight.

    A table is treated as bigger than it is, by a chair on each side:
    clearing the table alone still left its chairs inside the counter, and a
    chair drawn through a bar is exactly as unreadable as a table drawn
    through one.
    """
    placed = list(seats)
    movable = [
        index for index, seat in enumerate(placed) if seat.kind != COUNTED_ZONE_KIND
    ]
    movable_set = set(movable)
    fixed_boxes = [
        (seat.x, seat.y, seat.w, seat.h)
        for index, seat in enumerate(placed)
        if index not in movable_set
    ]
    for counter in counters:
        fixed_boxes.extend(counter_boxes(counter))

    def grown(seat):
        return (seat.x, seat.y, seat.w + CHAIR_CLEARANCE, seat.h + CHAIR_CLEARANCE)

    for _ in range(rounds):
        moved = False
        for index in movable:
            escape = _escape(grown(placed[index]), fixed_boxes)
            if escape is not None:
                placed[index] = replace(placed[index], x=escape[0], y=escape[1])
                moved = True
        for position, a in enumerate(movable):
            for b in movable[position + 1:]:
                first, second = placed[a], placed[b]
                overlap_x = (first.w + second.w) / 2 - abs(first.x - second.x)
                overlap_y = (first.h + second.h) / 2 - abs(first.y - second.y)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                moved = True
                if overlap_x < overlap_y:
                    push = overlap_x / 2 + 1.0
                    sign = 1.0 if first.x <= second.x else -1.0
                    placed[a] = replace(first, x=first.x - push * sign)
                    placed[b] = replace(second, x=second.x + push * sign)
                else:
                    push = overlap_y / 2 + 1.0
                    sign = 1.0 if first.y <= second.y else -1.0
                    placed[a] = replace(first, y=first.y - push * sign)
                    placed[b] = replace(second, y=second.y + push * sign)
        if not moved:
            break
    return tuple(placed)


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
    zone_chairs = set()
    for unit_index, chair_indices in layout.unit_chair_assignments().items():
        for chair_index in chair_indices:
            owner_by_chair[chair_index] = units[unit_index].name
            if units[unit_index].kind == COUNTED_ZONE_KIND:
                zone_chairs.add(chair_index)
    # A bar stool is the seat and the chair at once.  Drawing both puts a
    # circle beside every square and reads as twelve seats where there are
    # six.
    return [
        (owner_by_chair.get(index), floor_anchor(box))
        for index, box in enumerate(layout.chair_boxes())
        if index not in zone_chairs
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

    barred, counters = arrange_bars(tuple(seats))
    arranged_seats = separate_overlaps(barred, counters)
    return FloorPlan(
        schema_version=FLOORPLAN_SCHEMA_VERSION,
        extent=extent,
        seats=arranged_seats,
        chairs=arrange_chairs(arranged_seats, tuple(chairs)),
        counters=counters,
        landmarks=(),
        walls=(),
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
        "counters": [
            {
                "zone_id": counter.zone_id,
                "x1": round(counter.x1, 2),
                "y1": round(counter.y1, 2),
                "x2": round(counter.x2, 2),
                "y2": round(counter.y2, 2),
                "depth": round(counter.depth, 2),
            }
            for counter in plan.counters
        ],
        "walls": [[round(x, 2), round(y, 2)] for x, y in plan.walls],
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
        counters=tuple(
            FloorCounter(
                zone_id=str(counter["zone_id"]),
                x1=float(counter["x1"]),
                y1=float(counter["y1"]),
                x2=float(counter["x2"]),
                y2=float(counter["y2"]),
                depth=float(counter["depth"]),
            )
            for counter in data.get("counters", [])
        ),
        walls=tuple(
            (float(point[0]), float(point[1])) for point in data.get("walls", [])
        ),
    )

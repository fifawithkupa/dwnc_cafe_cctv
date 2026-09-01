"""Manual seat layout: load/validate/scale the calibrated table-chair map.

The layout is the ground truth for seat geometry.  Detection only fills in
per-frame evidence (people, belongings) inside these zones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

Box = Tuple[float, float, float, float]

SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)

TABLE_KIND = "table"
COUNTED_ZONE_KIND = "counted_zone"
VALID_KINDS = (TABLE_KIND, COUNTED_ZONE_KIND)


class LayoutError(ValueError):
    """Raised when a layout file is missing or malformed."""


def _parse_box(value, context: str) -> Box:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or not all(isinstance(v, (int, float)) for v in value)
    ):
        raise LayoutError(f"{context}: box must be [x1, y1, x2, y2], got {value!r}")
    x1, y1, x2, y2 = (float(v) for v in value)
    if x2 <= x1 or y2 <= y1:
        raise LayoutError(f"{context}: box has non-positive size: {value!r}")
    return (x1, y1, x2, y2)


def _box_contains(outer: Box, inner: Box, tolerance: float = 1.0) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


@dataclass(frozen=True)
class LayoutChair:
    id: int
    box: Box


@dataclass(frozen=True)
class LayoutSeat:
    """One hand-drawn seat slot inside a counted_zone (bar counter etc.)."""

    id: int
    box: Box


@dataclass(frozen=True)
class JudgementUnit:
    """One box the pipeline judges independently.

    A plain table is one unit.  A counted_zone becomes one unit per
    hand-drawn seat slot, so the existing evidence association and
    debouncing run per seat without a separate counting code path.
    """

    box: Box
    unit_id: int
    name: str
    kind: str
    capacity: int
    zone_id: Optional[int] = None
    zone_name: Optional[str] = None
    seat_id: Optional[int] = None


@dataclass(frozen=True)
class LayoutTable:
    id: int
    name: str
    box: Box
    chairs: Tuple[LayoutChair, ...] = ()
    kind: str = TABLE_KIND
    seats: Tuple[LayoutSeat, ...] = ()


@dataclass(frozen=True)
class SeatLayout:
    schema_version: int
    source: Dict[str, object]
    tables: Tuple[LayoutTable, ...]
    unassigned_chairs: Tuple[LayoutChair, ...] = ()

    def scaled_to(self, width: int, height: int) -> "SeatLayout":
        src_width = float(self.source.get("width", width))
        src_height = float(self.source.get("height", height))
        sx, sy = width / src_width, height / src_height

        def scale(box: Box) -> Box:
            return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)

        tables = tuple(
            replace(
                table,
                box=scale(table.box),
                chairs=tuple(
                    replace(chair, box=scale(chair.box)) for chair in table.chairs
                ),
                seats=tuple(
                    replace(seat, box=scale(seat.box)) for seat in table.seats
                ),
            )
            for table in self.tables
        )
        unassigned = tuple(
            replace(chair, box=scale(chair.box)) for chair in self.unassigned_chairs
        )
        source = dict(self.source, width=width, height=height)
        return replace(
            self, source=source, tables=tables, unassigned_chairs=unassigned
        )

    def chair_boxes(self) -> List[Box]:
        """Assigned chairs first (table order), then the unassigned ones.

        ``unit_chair_assignments`` indexes into this list, so the assigned
        chairs must keep the leading positions: inserting an orphan anywhere
        earlier would silently repoint every chair->table link.
        """
        boxes = [chair.box for table in self.tables for chair in table.chairs]
        boxes.extend(chair.box for chair in self.unassigned_chairs)
        return boxes

    def chair_assignments(self) -> Dict[int, List[int]]:
        assignments: Dict[int, List[int]] = {}
        cursor = 0
        for index, table in enumerate(self.tables):
            count = len(table.chairs)
            assignments[index] = list(range(cursor, cursor + count))
            cursor += count
        return assignments

    def judgement_units(self) -> Tuple[JudgementUnit, ...]:
        """Flatten the layout into the boxes the pipeline judges one by one."""
        units: List[JudgementUnit] = []
        for table in self.tables:
            if table.kind == COUNTED_ZONE_KIND:
                capacity = len(table.seats)
                for seat in table.seats:
                    units.append(
                        JudgementUnit(
                            box=seat.box,
                            unit_id=len(units) + 1,
                            name=f"{table.name}-{seat.id}",
                            kind=COUNTED_ZONE_KIND,
                            capacity=capacity,
                            zone_id=table.id,
                            zone_name=table.name,
                            seat_id=seat.id,
                        )
                    )
            else:
                units.append(
                    JudgementUnit(
                        box=table.box,
                        unit_id=len(units) + 1,
                        name=table.name,
                        kind=TABLE_KIND,
                        capacity=1,
                    )
                )
        return tuple(units)

    def unit_chair_assignments(self) -> Dict[int, List[int]]:
        """Chair indices per judgement-unit index.

        Chairs belong to plain tables only; counted_zone seats get an empty
        list so downstream indexing stays uniform.
        """
        assignments: Dict[int, List[int]] = {}
        unit_index = 0
        chair_cursor = 0
        for table in self.tables:
            if table.kind == COUNTED_ZONE_KIND:
                for _ in table.seats:
                    assignments[unit_index] = []
                    unit_index += 1
                continue
            count = len(table.chairs)
            assignments[unit_index] = list(range(chair_cursor, chair_cursor + count))
            chair_cursor += count
            unit_index += 1
        return assignments


def load_layout(path: Path) -> SeatLayout:
    path = Path(path)
    if not path.exists():
        raise LayoutError(f"Layout file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LayoutError(f"Layout is not valid JSON: {path}: {exc}") from exc

    if data.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise LayoutError(
            f"Unsupported schema_version {data.get('schema_version')!r} "
            f"(expected one of {SUPPORTED_SCHEMA_VERSIONS}) in {path}"
        )
    raw_tables = data.get("tables")
    if not isinstance(raw_tables, list) or not raw_tables:
        raise LayoutError(f"Layout must contain at least one table: {path}")

    tables: List[LayoutTable] = []
    seen_ids = set()
    for position, raw in enumerate(raw_tables, start=1):
        table_id = raw.get("id")
        if not isinstance(table_id, int):
            raise LayoutError(f"table #{position}: id must be an integer")
        if table_id in seen_ids:
            raise LayoutError(f"table #{position}: duplicate table id {table_id}")
        seen_ids.add(table_id)
        chairs = tuple(
            LayoutChair(
                id=int(chair.get("id", chair_position)),
                box=_parse_box(
                    chair.get("box"), f"table {table_id} chair #{chair_position}"
                ),
            )
            for chair_position, chair in enumerate(raw.get("chairs", []), start=1)
        )
        kind = str(raw.get("kind", TABLE_KIND))
        if kind not in VALID_KINDS:
            raise LayoutError(
                f"table {table_id}: unknown kind {kind!r} "
                f"(expected one of {VALID_KINDS})"
            )
        box = _parse_box(raw.get("box"), f"table {table_id}")
        seats = tuple(
            LayoutSeat(
                id=int(seat.get("id", seat_position)),
                box=_parse_box(
                    seat.get("box"), f"table {table_id} seat #{seat_position}"
                ),
            )
            for seat_position, seat in enumerate(raw.get("seats", []), start=1)
        )
        if kind == COUNTED_ZONE_KIND:
            if not seats:
                raise LayoutError(
                    f"table {table_id}: kind={COUNTED_ZONE_KIND} requires a "
                    f"non-empty 'seats' list (capacity is len(seats))"
                )
            for seat in seats:
                if not _box_contains(box, seat.box):
                    raise LayoutError(
                        f"table {table_id} seat {seat.id}: box {seat.box} is "
                        f"outside the zone box {box}"
                    )
        elif seats:
            raise LayoutError(
                f"table {table_id}: 'seats' is only valid for "
                f"kind={COUNTED_ZONE_KIND}"
            )
        tables.append(
            LayoutTable(
                id=table_id,
                name=str(raw.get("name", f"T{table_id}")),
                box=box,
                chairs=chairs,
                kind=kind,
                seats=seats,
            )
        )
    unassigned_chairs = tuple(
        LayoutChair(
            id=int(chair.get("id", position)),
            box=_parse_box(chair.get("box"), f"unassigned chair #{position}"),
        )
        for position, chair in enumerate(data.get("unassigned_chairs", []), start=1)
    )
    return SeatLayout(
        schema_version=SCHEMA_VERSION,
        source=dict(data.get("source", {})),
        tables=tuple(tables),
        unassigned_chairs=unassigned_chairs,
    )


def save_layout(layout: SeatLayout, path: Path) -> None:
    payload = {
        "schema_version": layout.schema_version,
        "source": layout.source,
        "tables": [
            {
                "id": table.id,
                "name": table.name,
                "kind": table.kind,
                "box": [round(v, 2) for v in table.box],
                "chairs": [
                    {"id": chair.id, "box": [round(v, 2) for v in chair.box]}
                    for chair in table.chairs
                ],
                "seats": [
                    {"id": seat.id, "box": [round(v, 2) for v in seat.box]}
                    for seat in table.seats
                ],
            }
            for table in layout.tables
        ],
        "unassigned_chairs": [
            {"id": chair.id, "box": [round(v, 2) for v in chair.box]}
            for chair in layout.unassigned_chairs
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

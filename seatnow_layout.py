"""Manual seat layout: load/validate/scale the calibrated table-chair map.

The layout is the ground truth for seat geometry.  Detection only fills in
per-frame evidence (people, belongings) inside these zones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

Box = Tuple[float, float, float, float]

SCHEMA_VERSION = 1


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


@dataclass(frozen=True)
class LayoutChair:
    id: int
    box: Box


@dataclass(frozen=True)
class LayoutTable:
    id: int
    name: str
    box: Box
    chairs: Tuple[LayoutChair, ...] = ()


@dataclass(frozen=True)
class SeatLayout:
    schema_version: int
    source: Dict[str, object]
    tables: Tuple[LayoutTable, ...]

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
            )
            for table in self.tables
        )
        source = dict(self.source, width=width, height=height)
        return replace(self, source=source, tables=tables)

    def chair_boxes(self) -> List[Box]:
        return [chair.box for table in self.tables for chair in table.chairs]

    def chair_assignments(self) -> Dict[int, List[int]]:
        assignments: Dict[int, List[int]] = {}
        cursor = 0
        for index, table in enumerate(self.tables):
            count = len(table.chairs)
            assignments[index] = list(range(cursor, cursor + count))
            cursor += count
        return assignments


def load_layout(path: Path) -> SeatLayout:
    path = Path(path)
    if not path.exists():
        raise LayoutError(f"Layout file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LayoutError(f"Layout is not valid JSON: {path}: {exc}") from exc

    if data.get("schema_version") != SCHEMA_VERSION:
        raise LayoutError(
            f"Unsupported schema_version {data.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION}) in {path}"
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
        tables.append(
            LayoutTable(
                id=table_id,
                name=str(raw.get("name", f"T{table_id}")),
                box=_parse_box(raw.get("box"), f"table {table_id}"),
                chairs=chairs,
            )
        )
    return SeatLayout(
        schema_version=SCHEMA_VERSION,
        source=dict(data.get("source", {})),
        tables=tuple(tables),
    )


def save_layout(layout: SeatLayout, path: Path) -> None:
    payload = {
        "schema_version": layout.schema_version,
        "source": layout.source,
        "tables": [
            {
                "id": table.id,
                "name": table.name,
                "box": [round(v, 2) for v in table.box],
                "chairs": [
                    {"id": chair.id, "box": [round(v, 2) for v in chair.box]}
                    for chair in table.chairs
                ],
            }
            for table in layout.tables
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

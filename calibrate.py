"""SeatNow calibration: register table/chair zones and their links.

State machine is GUI-free (unit tested); the OpenCV window is a thin shell.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from seatnow_layout import (
    SCHEMA_VERSION,
    LayoutChair,
    LayoutTable,
    SeatLayout,
)

Box = Tuple[float, float, float, float]


def _contains(box: Box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


class CalibrationState:
    def __init__(self) -> None:
        self.tables: List[Dict] = []  # {"box": Box, "chairs": List[Box]}
        self.selected: Optional[Tuple[str, int, int]] = None
        self._history: List[Tuple[List[Dict], Optional[Tuple[str, int, int]]]] = []

    def _snapshot(self) -> None:
        self._history.append((copy.deepcopy(self.tables), self.selected))

    def add_table(self, box: Box) -> None:
        self._snapshot()
        self.tables.append({"box": tuple(box), "chairs": []})
        self.selected = ("table", len(self.tables) - 1, -1)

    def add_chair(self, box: Box) -> bool:
        table_index = self._selected_table_index()
        if table_index is None:
            return False
        self._snapshot()
        self.tables[table_index]["chairs"].append(tuple(box))
        return True

    def _selected_table_index(self) -> Optional[int]:
        if self.selected is None:
            return None
        return self.selected[1]

    def select_at(self, x: float, y: float) -> None:
        candidates: List[Tuple[float, Tuple[str, int, int]]] = []
        for ti, table in enumerate(self.tables):
            if _contains(table["box"], x, y):
                candidates.append((_area(table["box"]), ("table", ti, -1)))
            for ci, chair in enumerate(table["chairs"]):
                if _contains(chair, x, y):
                    candidates.append((_area(chair), ("chair", ti, ci)))
        self.selected = min(candidates)[1] if candidates else None

    def delete_selected(self) -> None:
        if self.selected is None:
            return
        self._snapshot()
        kind, ti, ci = self.selected
        if kind == "table":
            del self.tables[ti]
        else:
            del self.tables[ti]["chairs"][ci]
        self.selected = None

    def undo(self) -> None:
        if not self._history:
            return
        self.tables, self.selected = self._history.pop()

    def to_layout(self, source: Dict) -> SeatLayout:
        tables = tuple(
            LayoutTable(
                id=index,
                name=f"T{index}",
                box=table["box"],
                chairs=tuple(
                    LayoutChair(id=chair_index, box=chair)
                    for chair_index, chair in enumerate(table["chairs"], start=1)
                ),
            )
            for index, table in enumerate(self.tables, start=1)
        )
        return SeatLayout(
            schema_version=SCHEMA_VERSION, source=dict(source), tables=tables
        )

    @classmethod
    def from_layout(cls, layout: SeatLayout) -> "CalibrationState":
        state = cls()
        for table in layout.tables:
            state.tables.append(
                {
                    "box": tuple(table.box),
                    "chairs": [tuple(chair.box) for chair in table.chairs],
                }
            )
        state.selected = None
        return state

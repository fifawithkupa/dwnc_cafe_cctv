"""박스 → Supabase: 판정 한 번마다 "지금 값"을 쓴다.

세 가지 순수 함수가 JSONL 한 줄을 ``cafe_live`` 한 줄로 바꾸고,
``SupabasePublisher`` 가 그것을 백그라운드에서 보낸다.  판정 루프는 여기서
절대 기다리지 않는다 -- 인터넷이 끊기면 값을 버리고, 앱은 45초 규칙으로
"확인 중"이 된다 (docs/앱연동.md).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from engine.seatnow_layout import COUNTED_ZONE_KIND, SeatLayout
from engine.seatnow_report import classify_reason

SEATS_SCHEMA_VERSION = 1
GAP_REASON = "no_fresh_frames"
_COUNTABLE = ("occupied", "empty", "unknown")

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _kind(layout_kind: Optional[str]) -> str:
    return "bar_seat" if layout_kind == COUNTED_ZONE_KIND else "table"


def seat_index_from_layout(layout: SeatLayout) -> List[Dict[str, Any]]:
    """Every judgement unit as {seat_id, kind, zone} -- what a gap row lists."""
    return [
        {"seat_id": unit.name, "kind": _kind(unit.kind), "zone": unit.zone_name}
        for unit in layout.judgement_units()
    ]


def _totals(seats: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"occupied": 0, "empty": 0, "unknown": 0}
    for seat in seats:
        counts[seat["state"]] += 1
    return {
        "total_tables": len(seats),
        "occupied_tables": counts["occupied"],
        "free_tables": counts["empty"],
        "unknown_tables": counts["unknown"],
    }


def _row(
    cafe_id: str, status: str, seats: List[Dict[str, Any]], tick_at: str, box_version: str
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"cafe_id": cafe_id, "status": status}
    row.update(_totals(seats))
    row.update(
        {
            "seats": seats,
            "tick_at": tick_at,
            "box_version": box_version,
            "schema_version": SEATS_SCHEMA_VERSION,
        }
    )
    return row


def live_payload(record: Dict[str, Any], cafe_id: str, box_version: str) -> Dict[str, Any]:
    """One JSONL tick -> one ``cafe_live`` row.

    Bar slots stay one row each (the map colours slots, not zones), ``ignore``
    seats are absent, and ``reason_code`` exists only for ``unknown``.
    """
    seats: List[Dict[str, Any]] = []
    for table in record.get("tables") or []:
        state = str(table.get("state", "unknown"))
        if state not in _COUNTABLE:
            continue
        reason_code = None
        if state == "unknown":
            reason_code = classify_reason(
                str(table.get("raw_state", state)),
                str(table.get("reason", "")),
                bool(table.get("predicted", False)),
            ).value
        seats.append(
            {
                "seat_id": str(table.get("layout_name") or table.get("label") or "?"),
                "kind": _kind(table.get("layout_kind")),
                "zone": table.get("layout_zone_name"),
                "state": state,
                "reason_code": reason_code,
            }
        )
    return _row(cafe_id, "live", seats, str(record.get("wall_clock", "")), box_version)


def gap_payload(
    seat_index: List[Dict[str, Any]], cafe_id: str, box_version: str, wall_clock: str
) -> Dict[str, Any]:
    """The camera went quiet: every seat is unknown, never "what it was"."""
    seats = [
        {
            "seat_id": entry["seat_id"],
            "kind": entry["kind"],
            "zone": entry.get("zone"),
            "state": "unknown",
            "reason_code": GAP_REASON,
        }
        for entry in seat_index
    ]
    return _row(cafe_id, "gap", seats, wall_clock, box_version)

"""박스 → Supabase 전송: payload 와 publisher."""

from __future__ import annotations

import unittest

from edge.publish import (
    GAP_REASON,
    SEATS_SCHEMA_VERSION,
    gap_payload,
    live_payload,
    seat_index_from_layout,
)
from engine.seatnow_layout import LayoutChair, LayoutSeat, LayoutTable, SeatLayout


def _table(name, state, *, kind="table", zone=None, reason="", raw_state=None, predicted=False):
    return {
        "layout_name": name,
        "label": "L00X",
        "layout_kind": kind,
        "layout_zone_name": zone,
        "state": state,
        "raw_state": raw_state or state,
        "reason": reason,
        "predicted": predicted,
        "confidence": 0.9,
    }


def _record(tables):
    return {
        "wall_clock": "2026-09-08T20:32:45+0900",
        "tables": tables,
    }


class LivePayloadTest(unittest.TestCase):
    def test_counts_by_table_and_only_confirmed_empties_are_free(self):
        record = _record(
            [
                _table("T1", "occupied"),
                _table("T2", "empty"),
                _table("T3", "unknown", reason="compact_occluded_pose"),
            ]
        )
        payload = live_payload(record, "dwnc", "abc1234")
        self.assertEqual(payload["cafe_id"], "dwnc")
        self.assertEqual(payload["status"], "live")
        self.assertEqual(payload["total_tables"], 3)
        self.assertEqual(payload["occupied_tables"], 1)
        self.assertEqual(payload["free_tables"], 1)
        self.assertEqual(payload["unknown_tables"], 1)
        self.assertEqual(payload["tick_at"], "2026-09-08T20:32:45+0900")
        self.assertEqual(payload["box_version"], "abc1234")
        self.assertEqual(payload["schema_version"], SEATS_SCHEMA_VERSION)

    def test_ignored_seats_are_left_out(self):
        record = _record([_table("T1", "occupied"), _table("T9", "ignore")])
        payload = live_payload(record, "dwnc", "v")
        self.assertEqual([s["seat_id"] for s in payload["seats"]], ["T1"])
        self.assertEqual(payload["total_tables"], 1)

    def test_bar_slots_are_one_row_each_with_zone(self):
        record = _record(
            [
                _table("BAR7-1", "empty", kind="counted_zone", zone="BAR7"),
                _table("BAR7-2", "occupied", kind="counted_zone", zone="BAR7"),
            ]
        )
        seats = live_payload(record, "dwnc", "v")["seats"]
        self.assertEqual(
            seats,
            [
                {"seat_id": "BAR7-1", "kind": "bar_seat", "zone": "BAR7", "state": "empty", "reason_code": None},
                {"seat_id": "BAR7-2", "kind": "bar_seat", "zone": "BAR7", "state": "occupied", "reason_code": None},
            ],
        )

    def test_reason_code_only_for_unknown_and_from_closed_vocabulary(self):
        record = _record(
            [
                _table("T1", "unknown", reason="compact_occluded_pose"),
                _table("T2", "occupied", reason="person_seated"),
            ]
        )
        seats = live_payload(record, "dwnc", "v")["seats"]
        self.assertEqual(seats[0]["reason_code"], "occluded_lower_body")  # 닫힌 어휘로 바뀐다
        self.assertIsNone(seats[1]["reason_code"])

    def test_seat_id_falls_back_to_label(self):
        table = _table("", "empty")
        table["layout_name"] = None
        payload = live_payload(_record([table]), "dwnc", "v")
        self.assertEqual(payload["seats"][0]["seat_id"], "L00X")


class GapPayloadTest(unittest.TestCase):
    def test_everything_unknown_with_gap_reason(self):
        index = [
            {"seat_id": "T1", "kind": "table", "zone": None},
            {"seat_id": "BAR7-1", "kind": "bar_seat", "zone": "BAR7"},
        ]
        payload = gap_payload(index, "dwnc", "v", "2026-09-08T21:00:00+0900")
        self.assertEqual(payload["status"], "gap")
        self.assertEqual(payload["total_tables"], 2)
        self.assertEqual(payload["occupied_tables"], 0)
        self.assertEqual(payload["free_tables"], 0)
        self.assertEqual(payload["unknown_tables"], 2)
        self.assertEqual(payload["tick_at"], "2026-09-08T21:00:00+0900")
        for seat in payload["seats"]:
            self.assertEqual(seat["state"], "unknown")
            self.assertEqual(seat["reason_code"], GAP_REASON)
        self.assertEqual(payload["seats"][1]["zone"], "BAR7")


class SeatIndexTest(unittest.TestCase):
    def test_index_lists_every_judgement_unit(self):
        layout = SeatLayout(
            schema_version=3,
            source={"width": 1920, "height": 1080},
            tables=(
                LayoutTable(
                    id=1,
                    name="T1",
                    box=(800.0, 700.0, 1000.0, 850.0),
                    chairs=(LayoutChair(id=1, box=(780.0, 800.0, 830.0, 880.0)),),
                ),
                LayoutTable(
                    id=7,
                    name="BAR7",
                    box=(1100.0, 620.0, 1500.0, 760.0),
                    kind="counted_zone",
                    seats=(
                        LayoutSeat(id=1, box=(1100.0, 620.0, 1300.0, 760.0)),
                        LayoutSeat(id=2, box=(1300.0, 620.0, 1500.0, 760.0)),
                    ),
                ),
            ),
        )
        index = seat_index_from_layout(layout)
        self.assertEqual(
            index,
            [
                {"seat_id": "T1", "kind": "table", "zone": None},
                {"seat_id": "BAR7-1", "kind": "bar_seat", "zone": "BAR7"},
                {"seat_id": "BAR7-2", "kind": "bar_seat", "zone": "BAR7"},
            ],
        )


if __name__ == "__main__":
    unittest.main()

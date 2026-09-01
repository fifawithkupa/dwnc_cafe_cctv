"""Unit tests for the floor plan draft and its file format."""

from __future__ import annotations

from dataclasses import replace

import json
import tempfile
import unittest
from pathlib import Path

from floor_projection import FloorProjectionError
from floorplan import (
    EXTENT_LONG_SIDE,
    FloorPlan,
    Landmark,
    build_draft,
    load_floorplan,
    save_floorplan,
)
from seatnow_layout import (
    FloorReference,
    LayoutChair,
    LayoutSeat,
    LayoutTable,
    SeatLayout,
)


REFERENCE = FloorReference(
    image_points=((700.0, 900.0), (900.0, 600.0), (1300.0, 600.0), (1600.0, 900.0))
)


def layout(with_reference=True, unassigned=(), zone_seats=2):
    return SeatLayout(
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
                seats=tuple(
                    LayoutSeat(
                        id=index,
                        box=(1100.0 + 150 * index, 620.0, 1200.0 + 150 * index, 760.0),
                    )
                    for index in range(1, zone_seats + 1)
                ),
            ),
        ),
        unassigned_chairs=tuple(
            LayoutChair(id=index, box=box)
            for index, box in enumerate(unassigned, start=1)
        ),
        floor_reference=REFERENCE if with_reference else None,
    )


class BuildDraftTests(unittest.TestCase):
    def test_every_judgement_unit_becomes_a_seat(self):
        plan = build_draft(layout())
        self.assertEqual(
            [seat.seat_id for seat in plan.seats], ["T1", "BAR7-1", "BAR7-2"]
        )

    def test_seat_kind_is_carried_over(self):
        plan = build_draft(layout())
        kinds = {seat.seat_id: seat.kind for seat in plan.seats}
        self.assertEqual(kinds["T1"], "table")
        self.assertEqual(kinds["BAR7-1"], "counted_zone")

    def test_chairs_carry_their_owner(self):
        plan = build_draft(layout(unassigned=[(400.0, 800.0, 450.0, 880.0)]))
        owners = sorted((chair.seat_id or "") for chair in plan.chairs)
        self.assertEqual(owners, ["", "T1"])

    def test_image_anchor_is_the_bottom_edge_centre(self):
        plan = build_draft(layout())
        seat = next(seat for seat in plan.seats if seat.seat_id == "T1")
        self.assertEqual(seat.image_anchor, (900.0, 850.0))

    def test_extent_long_side_is_normalised(self):
        plan = build_draft(layout())
        margin = 2 * 0.08 * EXTENT_LONG_SIDE
        self.assertAlmostEqual(max(plan.extent), EXTENT_LONG_SIDE + margin, places=4)

    def test_everything_lands_inside_the_extent(self):
        plan = build_draft(layout())
        width, height = plan.extent
        for seat in plan.seats:
            self.assertGreaterEqual(seat.x, 0.0)
            self.assertLessEqual(seat.x, width)
            self.assertGreaterEqual(seat.y, 0.0)
            self.assertLessEqual(seat.y, height)

    def test_no_floor_reference_is_refused(self):
        # Silently drawing an empty map would look like "this cafe has no
        # seats" rather than "nobody clicked the floor points yet".
        with self.assertRaises(FloorProjectionError):
            build_draft(layout(with_reference=False))

    def test_landmarks_start_empty(self):
        self.assertEqual(build_draft(layout()).landmarks, ())


class RoundTripTests(unittest.TestCase):
    def test_saving_settles_after_one_round_trip(self):
        drafted = build_draft(layout(unassigned=[(400.0, 800.0, 450.0, 880.0)]))
        plan = FloorPlan(
            schema_version=drafted.schema_version,
            extent=drafted.extent,
            seats=drafted.seats,
            chairs=drafted.chairs,
            landmarks=(
                Landmark(kind="entrance", label="입구", x=10.0, y=20.0, w=30.0, h=40.0),
            ),
        )
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "floorplan.json"
            save_floorplan(plan, path)
            once = load_floorplan(path)
            save_floorplan(once, path)
            twice = load_floorplan(path)

            # The file rounds to 2 decimals, which in drawing units is far
            # below anything a person can see.  What must hold is that the
            # rounding settles: load-edit-save must not drift a little
            # further every time the installer opens the editor.
            self.assertEqual(once, twice)
            self.assertEqual(
                [seat.seat_id for seat in once.seats],
                [seat.seat_id for seat in plan.seats],
            )
            self.assertAlmostEqual(once.seats[0].x, plan.seats[0].x, places=1)
            self.assertEqual(once.landmarks, plan.landmarks)
            self.assertEqual(
                [chair.seat_id for chair in once.chairs],
                [chair.seat_id for chair in plan.chairs],
            )

    def test_file_is_readable_json_with_the_documented_keys(self):
        plan = build_draft(layout())
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "floorplan.json"
            save_floorplan(plan, path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(data),
                ["chairs", "extent", "landmarks", "schema_version", "seats"],
            )
            self.assertIn("image_anchor", data["seats"][0])


class ZoneChairOwnerTests(unittest.TestCase):
    """A bar chair hangs under the slot it covers, not under the first one.

    Judgement already works this out in unit_chair_assignments; the map has
    to agree with it, or the ownership lines the editor draws would point at
    the wrong stool and the installer would "fix" something that is right.
    """

    def _bar_layout(self):
        return SeatLayout(
            schema_version=3,
            source={"width": 1920, "height": 1080},
            tables=(
                LayoutTable(
                    id=7,
                    name="BAR7",
                    box=(1000.0, 600.0, 1600.0, 900.0),
                    kind="counted_zone",
                    seats=(
                        LayoutSeat(id=1, box=(1000.0, 600.0, 1200.0, 900.0)),
                        LayoutSeat(id=2, box=(1200.0, 600.0, 1400.0, 900.0)),
                        LayoutSeat(id=3, box=(1400.0, 600.0, 1600.0, 900.0)),
                    ),
                    chairs=(
                        LayoutChair(id=1, box=(1020.0, 650.0, 1180.0, 880.0)),
                        LayoutChair(id=2, box=(1420.0, 650.0, 1580.0, 880.0)),
                    ),
                ),
            ),
            floor_reference=REFERENCE,
        )

    def test_each_bar_chair_takes_its_own_slot(self):
        plan = build_draft(self._bar_layout())
        self.assertEqual(
            [chair.seat_id for chair in plan.chairs], ["BAR7-1", "BAR7-3"]
        )

    def test_a_bar_chair_covering_no_slot_has_no_owner(self):
        layout = self._bar_layout()
        stray = LayoutTable(
            id=7,
            name="BAR7",
            box=layout.tables[0].box,
            kind="counted_zone",
            seats=layout.tables[0].seats,
            chairs=(LayoutChair(id=1, box=(200.0, 200.0, 260.0, 260.0)),),
        )
        plan = build_draft(replace(layout, tables=(stray,)))
        self.assertEqual([chair.seat_id for chair in plan.chairs], [None])


if __name__ == "__main__":
    unittest.main()

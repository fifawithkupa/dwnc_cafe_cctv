"""Unit tests for the floor plan draft and its file format."""

from __future__ import annotations

from dataclasses import replace

import json
import math
import tempfile
from collections import Counter
import unittest
from pathlib import Path

from install.floor_projection import FloorProjectionError
from install.floorplan import (
    EXTENT_LONG_SIDE,
    FloorChair,
    FloorSeat,
    arrange_chairs,
    arrange_bars,
    separate_overlaps,
    FloorPlan,
    Landmark,
    build_draft,
    load_floorplan,
    save_floorplan,
)
from engine.seatnow_layout import (
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
                ["chairs", "counters", "extent", "landmarks",
                 "schema_version", "seats", "walls"],
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

    def test_a_bar_chair_is_not_drawn_twice(self):
        # The stool is the seat and the chair at once.  Drawing both put a
        # circle beside every square and read as twelve seats where the cafe
        # has six.
        plan = build_draft(self._bar_layout())
        self.assertEqual([chair.seat_id for chair in plan.chairs], [])
        self.assertEqual(
            [seat.seat_id for seat in plan.seats],
            ["BAR7-1", "BAR7-2", "BAR7-3"],
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


class ArrangeChairsTests(unittest.TestCase):
    """Chairs are drawn around their table, not where the projection put them.

    A customer reads "this is a four-seater", never the exact centimetres of
    a stool.  The projected chair positions scatter, look like nothing, and
    are not used by judgement anyway -- judgement reads the camera-image
    boxes.  So the map places them from the table.

    Fill order is top, bottom, top, bottom, right, left.  Five chairs then
    come out as two above, two below and one on the right, which is what the
    real T6 looks like.
    """

    def _seat(self, x=500.0, y=500.0, w=150.0, h=105.0):
        return FloorSeat(seat_id="T1", kind="table", x=x, y=y, w=w, h=h,
                         image_anchor=(0.0, 0.0))

    def _chairs(self, count):
        return tuple(
            FloorChair(seat_id="T1", x=0.0, y=0.0, w=46.0, h=46.0,
                       image_anchor=(float(index), 0.0))
            for index in range(count)
        )

    def _sides(self, seat, chairs):
        sides = []
        for chair in chairs:
            if chair.y < seat.y - seat.h / 2:
                sides.append("top")
            elif chair.y > seat.y + seat.h / 2:
                sides.append("bottom")
            elif chair.x > seat.x:
                sides.append("right")
            else:
                sides.append("left")
        return sides

    def test_five_chairs_go_two_two_one(self):
        seat = self._seat()
        arranged = arrange_chairs((seat,), self._chairs(5))
        counts = Counter(self._sides(seat, arranged))
        self.assertEqual(counts["top"], 2)
        self.assertEqual(counts["bottom"], 2)
        self.assertEqual(counts["right"], 1)

    def test_two_chairs_face_each_other(self):
        seat = self._seat()
        arranged = arrange_chairs((seat,), self._chairs(2))
        self.assertEqual(sorted(self._sides(seat, arranged)), ["bottom", "top"])

    def test_four_chairs_split_two_and_two(self):
        seat = self._seat()
        counts = Counter(self._sides(seat, arrange_chairs((seat,), self._chairs(4))))
        self.assertEqual(counts["top"], 2)
        self.assertEqual(counts["bottom"], 2)

    def test_six_chairs_use_both_ends(self):
        seat = self._seat()
        counts = Counter(self._sides(seat, arrange_chairs((seat,), self._chairs(6))))
        self.assertEqual((counts["top"], counts["bottom"]), (2, 2))
        self.assertEqual(counts["right"] + counts["left"], 2)

    def test_chairs_sit_outside_the_table(self):
        seat = self._seat()
        for chair in arrange_chairs((seat,), self._chairs(4)):
            self.assertGreater(
                abs(chair.y - seat.y), seat.h / 2, "의자가 테이블 위에 겹쳤다"
            )

    def test_chairs_follow_a_moved_table(self):
        near = arrange_chairs((self._seat(x=100.0, y=100.0),), self._chairs(2))
        far = arrange_chairs((self._seat(x=900.0, y=900.0),), self._chairs(2))
        self.assertLess(near[0].x, far[0].x)

    def test_an_orphan_chair_keeps_its_place(self):
        orphan = FloorChair(seat_id=None, x=42.0, y=43.0, w=46.0, h=46.0,
                            image_anchor=(1.0, 1.0))
        self.assertEqual(arrange_chairs((self._seat(),), (orphan,))[0].x, 42.0)


class ArrangeZoneSeatsTests(unittest.TestCase):
    """A bar is one long counter, and its stools sit evenly along it.

    Projected slot positions wobble, so the row comes out crooked and the
    boxes overlap.  The line the slots lie on is real information; the gaps
    between them are not.
    """

    def _zone(self, points):
        return tuple(
            FloorSeat(seat_id=f"BAR7-{index}", kind="counted_zone",
                      x=x, y=y, w=52.0, h=52.0, image_anchor=(0.0, 0.0))
            for index, (x, y) in enumerate(points, start=1)
        )

    def test_seats_end_up_evenly_spaced(self):
        seats = self._zone([(100.0, 100.0), (140.0, 210.0), (150.0, 400.0)])
        arranged, _ = arrange_bars(seats)
        gaps = [
            math.hypot(b.x - a.x, b.y - a.y)
            for a, b in zip(arranged, arranged[1:])
        ]
        self.assertAlmostEqual(gaps[0], gaps[1], places=3)

    def test_the_row_keeps_its_length(self):
        # The stools are pushed off the counter line, so their coordinates
        # move -- but the run they cover is real and must not shrink.
        seats = self._zone([(100.0, 100.0), (140.0, 210.0), (150.0, 400.0)])
        arranged, _ = arrange_bars(seats)
        before = math.hypot(150.0 - 100.0, 400.0 - 100.0)
        after = math.hypot(
            arranged[-1].x - arranged[0].x, arranged[-1].y - arranged[0].y
        )
        self.assertAlmostEqual(after, before, places=3)

    def test_order_along_the_counter_is_kept(self):
        seats = self._zone([(100.0, 100.0), (140.0, 210.0), (150.0, 400.0)])
        arranged, _ = arrange_bars(seats)
        self.assertEqual([seat.seat_id for seat in arranged],
                         ["BAR7-1", "BAR7-2", "BAR7-3"])

    def test_a_single_seat_is_left_alone(self):
        seats = self._zone([(100.0, 100.0)])
        self.assertEqual(arrange_bars(seats)[0][0].x, 100.0)

    def test_plain_tables_are_untouched(self):
        table = FloorSeat(seat_id="T1", kind="table", x=7.0, y=8.0, w=1.0, h=1.0,
                          image_anchor=(0.0, 0.0))
        self.assertEqual(arrange_bars((table,))[0][0].x, 7.0)


class CounterTests(unittest.TestCase):
    """A bar is one long table with stools alongside it.

    Six separate boxes in a row do not read as a counter, and drawing the
    stools on top of it reads as nothing at all.  The counter is the piece of
    furniture; the stools sit beside it, on the room side.
    """

    def _zone(self, points, others=()):
        seats = [
            FloorSeat(seat_id=f"BAR7-{index}", kind="counted_zone",
                      x=x, y=y, w=52.0, h=52.0, image_anchor=(0.0, 0.0))
            for index, (x, y) in enumerate(points, start=1)
        ]
        seats += [
            FloorSeat(seat_id=f"T{index}", kind="table", x=x, y=y, w=150.0, h=105.0,
                      image_anchor=(0.0, 0.0))
            for index, (x, y) in enumerate(others, start=1)
        ]
        return tuple(seats)

    def test_one_counter_per_zone(self):
        seats = self._zone([(800.0, 100.0), (800.0, 300.0), (800.0, 500.0)])
        counters = arrange_bars(seats)[1]
        self.assertEqual(len(counters), 1)
        self.assertEqual(counters[0].zone_id, "BAR7")

    def test_counter_spans_the_row(self):
        seats = self._zone([(800.0, 100.0), (800.0, 300.0), (800.0, 500.0)])
        counter = arrange_bars(seats)[1][0]
        self.assertAlmostEqual(math.hypot(counter.x2 - counter.x1,
                                          counter.y2 - counter.y1), 400.0, places=0)

    def test_stools_sit_beside_the_counter_not_on_it(self):
        seats = self._zone([(800.0, 100.0), (800.0, 300.0), (800.0, 500.0)])
        arranged, counters = arrange_bars(seats)
        counter = counters[0]
        for seat in arranged:
            if seat.kind != "counted_zone":
                continue
            # distance from the stool centre to the counter line
            numerator = abs(
                (counter.y2 - counter.y1) * seat.x
                - (counter.x2 - counter.x1) * seat.y
                + counter.x2 * counter.y1 - counter.y2 * counter.x1
            )
            length = math.hypot(counter.x2 - counter.x1, counter.y2 - counter.y1)
            self.assertGreater(numerator / length, seat.w / 2)

    def test_stools_go_toward_the_room_not_the_wall(self):
        # The tables are to the left of the counter, so that is the room.
        seats = self._zone(
            [(800.0, 100.0), (800.0, 300.0), (800.0, 500.0)],
            others=[(200.0, 300.0)],
        )
        arranged, _ = arrange_bars(seats)
        stools = [seat for seat in arranged if seat.kind == "counted_zone"]
        self.assertTrue(all(stool.x < 800.0 for stool in stools))

    def test_a_lone_stool_makes_no_counter(self):
        seats = self._zone([(800.0, 100.0)])
        self.assertEqual(arrange_bars(seats)[1], ())


class SeparateOverlapsTests(unittest.TestCase):
    """Two seats drawn on top of each other cannot both be read."""

    def test_a_table_wedged_between_stools_gets_out(self):
        # Pushing along an axis alone just swaps which stool it sits on, and
        # the table oscillates until the rounds run out.  It has to move away
        # from the row, not along it.
        stools = tuple(
            FloorSeat(seat_id=f"BAR7-{index}", kind="counted_zone",
                      x=500.0, y=440.0 + 60.0 * index, w=52.0, h=52.0,
                      image_anchor=(0.0, 0.0))
            for index in range(1, 5)
        )
        seats = (self._table("T1", 500.0, 560.0),) + stools
        separated = separate_overlaps(seats)
        table = separated[0]
        for stool in separated[1:]:
            self.assertTrue(
                abs(table.x - stool.x) >= (table.w + stool.w) / 2
                or abs(table.y - stool.y) >= (table.h + stool.h) / 2,
                f"{stool.seat_id} 과 아직 겹친다",
            )

    def _table(self, seat_id, x, y):
        return FloorSeat(seat_id=seat_id, kind="table", x=x, y=y, w=150.0, h=105.0,
                         image_anchor=(0.0, 0.0))

    def test_overlapping_tables_are_pushed_apart(self):
        seats = (self._table("T1", 500.0, 500.0), self._table("T2", 520.0, 510.0))
        separated = separate_overlaps(seats)
        gap_x = abs(separated[0].x - separated[1].x)
        gap_y = abs(separated[0].y - separated[1].y)
        self.assertTrue(gap_x >= 150.0 or gap_y >= 105.0)

    def test_tables_already_apart_do_not_move(self):
        seats = (self._table("T1", 100.0, 100.0), self._table("T2", 900.0, 900.0))
        self.assertEqual(separate_overlaps(seats), seats)

    def test_a_table_moves_off_a_stool_and_the_stool_stays(self):
        # The stool sits where the bar rule put it; shoving it would bend the
        # row the rule just made straight.  The table is the one that gives.
        stool = FloorSeat(seat_id="BAR7-1", kind="counted_zone", x=500.0, y=500.0,
                          w=52.0, h=52.0, image_anchor=(0.0, 0.0))
        seats = (self._table("T1", 500.0, 500.0), stool)
        separated = separate_overlaps(seats)
        self.assertEqual((separated[1].x, separated[1].y), (500.0, 500.0))
        table, kept = separated
        self.assertTrue(
            abs(table.x - kept.x) >= (table.w + kept.w) / 2
            or abs(table.y - kept.y) >= (table.h + kept.h) / 2,
            "테이블이 스툴에서 안 비켰다",
        )


class ClearanceTests(unittest.TestCase):
    """Nothing drawn may sit on top of anything else drawn.

    Checking seat against seat was not enough: a table cleared the stools and
    its own chairs still landed inside the counter.  A table needs room for
    its chairs, not just for itself.
    """

    def _plan(self):
        return build_draft(
            SeatLayout(
                schema_version=3,
                source={"width": 1920, "height": 1080},
                tables=(
                    LayoutTable(
                        id=1, name="T1", box=(1250.0, 700.0, 1450.0, 850.0),
                        chairs=tuple(
                            LayoutChair(id=index, box=(1200.0 + 40 * index, 860.0,
                                                       1240.0 + 40 * index, 900.0))
                            for index in range(1, 4)
                        ),
                    ),
                    LayoutTable(
                        id=7, name="BAR7", box=(1000.0, 600.0, 1600.0, 700.0),
                        kind="counted_zone",
                        seats=tuple(
                            LayoutSeat(id=index,
                                       box=(950.0 + 100 * index, 600.0,
                                            1050.0 + 100 * index, 700.0))
                            for index in range(1, 5)
                        ),
                    ),
                ),
                floor_reference=REFERENCE,
            )
        )

    def _boxes_overlap(self, a, b):
        return (abs(a[0] - b[0]) < (a[2] + b[2]) / 2
                and abs(a[1] - b[1]) < (a[3] + b[3]) / 2)

    def _counter_box(self, counter):
        return (
            (counter.x1 + counter.x2) / 2, (counter.y1 + counter.y2) / 2,
            abs(counter.x2 - counter.x1) + counter.depth,
            abs(counter.y2 - counter.y1) + counter.depth,
        )

    def test_no_chair_sits_on_a_stool(self):
        plan = self._plan()
        stools = [s for s in plan.seats if s.kind == "counted_zone"]
        for chair in plan.chairs:
            for stool in stools:
                self.assertFalse(
                    self._boxes_overlap((chair.x, chair.y, chair.w, chair.h),
                                        (stool.x, stool.y, stool.w, stool.h)),
                    f"의자가 {stool.seat_id} 위에 겹쳤다",
                )

    def test_no_chair_sits_on_a_counter(self):
        plan = self._plan()
        for chair in plan.chairs:
            for counter in plan.counters:
                self.assertFalse(
                    self._boxes_overlap((chair.x, chair.y, chair.w, chair.h),
                                        self._counter_box(counter)),
                    "의자가 카운터 위에 겹쳤다",
                )

    def test_no_table_sits_on_a_counter(self):
        plan = self._plan()
        for seat in plan.seats:
            if seat.kind == "counted_zone":
                continue
            for counter in plan.counters:
                self.assertFalse(
                    self._boxes_overlap((seat.x, seat.y, seat.w, seat.h),
                                        self._counter_box(counter)),
                    f"{seat.seat_id} 이 카운터 위에 겹쳤다",
                )


if __name__ == "__main__":
    unittest.main()

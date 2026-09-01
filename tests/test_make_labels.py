"""Tests for the labelling-fixture generator and validator (plan.md T3)."""

from __future__ import annotations

import unittest

from checks.make_labels import TODO, validate


def interval(start, end, occupied, empty, ignore):
    return {
        "start_seconds": start,
        "end_seconds": end,
        "expected": {"occupied": occupied, "empty": empty, "ignore": ignore},
    }


def fixture(timeline, seats=("A", "B")):
    return {
        "schema_version": 1,
        "video": {"sha256": "0" * 64},
        "semantic_tables": {seat: {} for seat in seats},
        "timeline": timeline,
        "expected_events": {"expected_customer_occupancy_transition_count": 0},
    }


class ValidateTests(unittest.TestCase):
    def test_complete_fixture_has_no_problems(self):
        data = fixture(
            [
                interval(0.0, 30.0, ["A"], ["B"], []),
                interval(30.0, 60.0, [], ["A"], ["B"]),
            ]
        )
        self.assertEqual(validate(data), [])

    def test_unfilled_interval_is_reported(self):
        data = fixture([interval(0.0, 30.0, TODO, TODO, TODO)])
        problems = validate(data)
        self.assertEqual(len(problems), 3)
        self.assertTrue(all(TODO in problem for problem in problems))

    def test_seat_missing_from_an_interval_is_reported(self):
        """Every seat must be judged, so an omission cannot look like 'empty'."""
        data = fixture([interval(0.0, 30.0, ["A"], [], [])])
        problems = validate(data)
        self.assertEqual(len(problems), 1)
        self.assertIn("unlabelled: ['B']", problems[0])

    def test_seat_listed_twice_is_reported(self):
        data = fixture([interval(0.0, 30.0, ["A"], ["A", "B"], [])])
        self.assertTrue(
            any("listed twice" in problem for problem in validate(data))
        )

    def test_unknown_seat_is_reported(self):
        data = fixture([interval(0.0, 30.0, ["A"], ["B"], ["Z"])])
        self.assertTrue(any("unknown seat" in problem for problem in validate(data)))

    def test_timeline_gap_is_reported(self):
        data = fixture(
            [
                interval(0.0, 30.0, ["A"], ["B"], []),
                interval(45.0, 60.0, ["A"], ["B"], []),
            ]
        )
        self.assertTrue(any("gap or overlap" in problem for problem in validate(data)))

    def test_missing_transition_count_is_reported(self):
        data = fixture([interval(0.0, 30.0, ["A"], ["B"], [])])
        data["expected_events"]["expected_customer_occupancy_transition_count"] = TODO
        self.assertTrue(
            any("transition_count" in problem for problem in validate(data))
        )

    def test_empty_timeline_is_reported(self):
        self.assertIn("timeline is empty", validate(fixture([])))


class CountedZoneSkeletonTests(unittest.TestCase):
    """바 구역은 자리 칸마다 하나씩 라벨을 받아야 한다."""

    def _layout(self):
        from engine.seatnow_layout import LayoutSeat, LayoutTable, SeatLayout

        return SeatLayout(
            schema_version=2,
            source={},
            tables=(
                LayoutTable(
                    id=7,
                    name="BAR",
                    box=(200.0, 300.0, 800.0, 500.0),
                    kind="counted_zone",
                    seats=(
                        LayoutSeat(id=1, box=(200.0, 300.0, 500.0, 500.0)),
                        LayoutSeat(id=2, box=(500.0, 300.0, 800.0, 500.0)),
                    ),
                ),
                LayoutTable(id=1, name="창가1", box=(900.0, 300.0, 1100.0, 500.0)),
            ),
        )

    def test_zone_seats_appear_as_individual_label_targets(self):
        from checks.make_labels import seat_names_from_layout

        self.assertEqual(
            seat_names_from_layout(self._layout()), ["BAR-1", "BAR-2", "창가1"]
        )


if __name__ == "__main__":
    unittest.main()

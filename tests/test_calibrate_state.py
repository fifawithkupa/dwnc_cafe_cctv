"""Unit tests for the calibration editing state (no GUI)."""

from __future__ import annotations

import unittest

from calibrate import CalibrationState
from seatnow_layout import SeatLayout


class CalibrationStateTests(unittest.TestCase):
    def test_add_table_selects_it_and_chairs_attach_to_selection(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        state.add_table((500.0, 100.0, 700.0, 200.0))
        # 두 번째 테이블이 선택된 상태 → 의자는 거기에 붙는다
        self.assertTrue(state.add_chair((710.0, 110.0, 760.0, 190.0)))

        self.assertEqual(len(state.tables[0]["chairs"]), 0)
        self.assertEqual(len(state.tables[1]["chairs"]), 1)

    def test_add_chair_without_table_becomes_unassigned(self):
        # Install step 3-c draws chairs before anyone decides ownership.
        # Refusing the draw used to print the same error over and over while
        # the person had no way to proceed.
        state = CalibrationState()
        self.assertTrue(state.add_chair((10.0, 10.0, 20.0, 20.0)))
        self.assertEqual(len(state.unassigned_chairs), 1)
        self.assertEqual(state.tables, [])

    def test_select_at_picks_smallest_containing_box(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 400.0, 300.0))
        state.add_chair((110.0, 110.0, 160.0, 180.0))

        state.select_at(120.0, 120.0)  # 의자와 테이블 둘 다 포함 → 작은 쪽(의자)
        self.assertEqual(state.selected[0], "chair")

        state.select_at(350.0, 250.0)  # 테이블만 포함
        self.assertEqual(state.selected[0], "table")

        state.select_at(900.0, 900.0)  # 아무것도 없음
        self.assertIsNone(state.selected)

    def test_delete_selected_table_removes_its_chairs(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 400.0, 300.0))
        state.add_chair((110.0, 110.0, 160.0, 180.0))
        state.select_at(350.0, 250.0)

        state.delete_selected()

        self.assertEqual(state.tables, [])

    def test_undo_restores_previous_step(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 400.0, 300.0))
        state.add_chair((110.0, 110.0, 160.0, 180.0))

        state.undo()

        self.assertEqual(len(state.tables), 1)
        self.assertEqual(state.tables[0]["chairs"], [])

    def test_to_layout_assigns_sequential_ids(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        state.add_chair((40.0, 110.0, 90.0, 190.0))
        state.add_table((500.0, 100.0, 700.0, 200.0))

        layout = state.to_layout({"video": "v.mp4", "width": 1280, "height": 720})

        self.assertIsInstance(layout, SeatLayout)
        self.assertEqual([t.id for t in layout.tables], [1, 2])
        self.assertEqual(layout.tables[0].chairs[0].id, 1)

    def test_from_layout_round_trip(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        state.add_chair((40.0, 110.0, 90.0, 190.0))
        layout = state.to_layout({"width": 1280, "height": 720})

        restored = CalibrationState.from_layout(layout)

        self.assertEqual(restored.tables[0]["box"], (100.0, 100.0, 300.0, 200.0))
        self.assertEqual(restored.tables[0]["chairs"], [(40.0, 110.0, 90.0, 190.0)])


class CountedZoneCalibrationTests(unittest.TestCase):
    """일자형/벽 책상: 구역을 치고 그 안에 자리마다 칸을 긋는다."""

    def test_add_zone_then_seats(self):
        state = CalibrationState()
        state.add_zone((100.0, 100.0, 500.0, 300.0))

        self.assertTrue(state.add_seat((100.0, 100.0, 300.0, 300.0)))
        self.assertTrue(state.add_seat((300.0, 100.0, 500.0, 300.0)))

        self.assertEqual(state.tables[0]["kind"], "counted_zone")
        self.assertEqual(len(state.tables[0]["seats"]), 2)

    def test_add_seat_without_a_zone_selected_fails(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 300.0))

        self.assertFalse(state.add_seat((110.0, 110.0, 200.0, 290.0)))

    def test_undo_removes_last_seat(self):
        state = CalibrationState()
        state.add_zone((100.0, 100.0, 500.0, 300.0))
        state.add_seat((100.0, 100.0, 300.0, 300.0))
        state.add_seat((300.0, 100.0, 500.0, 300.0))

        state.undo()

        self.assertEqual(len(state.tables[0]["seats"]), 1)

    def test_plain_tables_keep_table_kind(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 300.0))

        self.assertEqual(state.tables[0]["kind"], "table")
        self.assertEqual(state.tables[0]["seats"], [])

    def test_zone_round_trips_through_layout(self):
        state = CalibrationState()
        state.add_zone((100.0, 100.0, 500.0, 300.0))
        state.add_seat((100.0, 100.0, 300.0, 300.0))
        state.add_seat((300.0, 100.0, 500.0, 300.0))

        layout = state.to_layout({"width": 1280, "height": 720})
        restored = CalibrationState.from_layout(layout)

        self.assertEqual(layout.tables[0].kind, "counted_zone")
        self.assertEqual(len(layout.tables[0].seats), 2)
        self.assertEqual(restored.tables[0]["kind"], "counted_zone")
        self.assertEqual(
            restored.tables[0]["seats"],
            [(100.0, 100.0, 300.0, 300.0), (300.0, 100.0, 500.0, 300.0)],
        )

    def test_zone_layout_is_loadable_by_the_analyzer(self):
        """to_layout 결과가 judgement_units()로 바로 펼쳐져야 한다."""
        state = CalibrationState()
        state.add_table((600.0, 100.0, 800.0, 300.0))
        state.add_zone((100.0, 100.0, 500.0, 300.0))
        state.add_seat((100.0, 100.0, 300.0, 300.0))
        state.add_seat((300.0, 100.0, 500.0, 300.0))

        units = state.to_layout({"width": 1280, "height": 720}).judgement_units()

        self.assertEqual([unit.kind for unit in units], ["table", "counted_zone", "counted_zone"])
        self.assertEqual(units[1].capacity, 2)


class SeatInsideZoneTests(unittest.TestCase):
    """A seat slot drawn outside its zone saves fine and fails hours later.

    load_layout rejects it (seatnow_layout.py:243-249) but save_layout does
    not, so the person who could fix it in five seconds is already gone by
    the time anyone sees the error.
    """

    def _zone_with_seat(self, seat_box):
        state = CalibrationState()
        state.add_zone((100.0, 100.0, 500.0, 300.0))
        state.add_seat(seat_box)
        return state

    def test_seat_inside_the_zone_is_valid(self):
        state = self._zone_with_seat((120.0, 120.0, 200.0, 280.0))
        self.assertEqual(state.invalid_seat_zones(), [])

    def test_seat_hanging_outside_is_reported(self):
        state = self._zone_with_seat((450.0, 120.0, 600.0, 280.0))
        self.assertEqual(state.invalid_seat_zones(), [(0, 0)])

    def test_seat_touching_the_edge_is_allowed(self):
        # load_layout allows a 1px tolerance; matching it here keeps the two
        # checks from disagreeing about the same file.
        state = self._zone_with_seat((100.0, 100.0, 500.0, 300.0))
        self.assertEqual(state.invalid_seat_zones(), [])

    def test_plain_table_chairs_are_not_checked(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        state.add_chair((400.0, 400.0, 450.0, 450.0))
        self.assertEqual(state.invalid_seat_zones(), [])

    def test_every_offending_seat_is_listed(self):
        state = self._zone_with_seat((450.0, 120.0, 600.0, 280.0))
        state.add_seat((700.0, 700.0, 800.0, 800.0))
        self.assertEqual(state.invalid_seat_zones(), [(0, 0), (0, 1)])


class UnassignedChairStateTests(unittest.TestCase):
    def test_chair_still_attaches_when_a_table_is_selected(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        self.assertTrue(state.add_chair((310.0, 110.0, 360.0, 190.0)))
        self.assertEqual(len(state.tables[0]["chairs"]), 1)
        self.assertEqual(state.unassigned_chairs, [])

    def test_unassigned_chair_can_be_selected_and_deleted(self):
        state = CalibrationState()
        state.add_chair((10.0, 10.0, 40.0, 40.0))
        state.select_at(20.0, 20.0)
        self.assertEqual(state.selected, ("orphan", -1, 0))
        state.delete_selected()
        self.assertEqual(state.unassigned_chairs, [])

    def test_undo_restores_unassigned_chairs(self):
        state = CalibrationState()
        state.add_chair((10.0, 10.0, 40.0, 40.0))
        state.add_chair((50.0, 50.0, 80.0, 80.0))
        state.undo()
        self.assertEqual(len(state.unassigned_chairs), 1)

    def test_selection_prefers_the_smaller_box(self):
        # Same rule as everything else: a small orphan inside a big table wins.
        state = CalibrationState()
        state.add_table((0.0, 0.0, 500.0, 500.0))
        state.add_chair((100.0, 100.0, 140.0, 140.0))  # attaches to the table
        state.selected = None
        state.unassigned_chairs.append((110.0, 110.0, 120.0, 120.0))
        state.select_at(115.0, 115.0)
        self.assertEqual(state.selected, ("orphan", -1, 0))

    def test_round_trip_through_layout_keeps_orphans(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        state.selected = None
        state.add_chair((900.0, 900.0, 940.0, 940.0))
        layout = state.to_layout({"width": 1920, "height": 1080})
        self.assertEqual(len(layout.unassigned_chairs), 1)

        restored = CalibrationState.from_layout(layout)
        self.assertEqual(len(restored.unassigned_chairs), 1)
        self.assertEqual(restored.unassigned_chairs[0], (900.0, 900.0, 940.0, 940.0))


if __name__ == "__main__":
    unittest.main()

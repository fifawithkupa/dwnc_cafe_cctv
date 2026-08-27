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

    def test_add_chair_without_table_returns_false(self):
        state = CalibrationState()
        self.assertFalse(state.add_chair((10.0, 10.0, 20.0, 20.0)))

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


if __name__ == "__main__":
    unittest.main()

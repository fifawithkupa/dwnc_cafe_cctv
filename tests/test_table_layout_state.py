"""Unit tests for automatic table layout state management."""

from __future__ import annotations

import unittest

from seatnow_core import OccupancyState, TableObservation, TableTracker


FRAME_SHAPE = (1000, 1000)


def obs(state: OccupancyState, box, confidence: float = 0.9):
    return TableObservation(
        box=tuple(float(value) for value in box),
        table_confidence=confidence,
        raw_state=state,
        raw_score=confidence,
        reason="fixture",
    )


class TableLayoutStateTests(unittest.TestCase):
    def make_tracker(self, **kwargs):
        return TableTracker(
            occupy_confirmations=1,
            empty_confirmations=1,
            max_missed=3,
            layout_add_confirmations=3,
            layout_move_confirmations=3,
            layout_remove_confirmations=3,
            **kwargs,
        )

    def test_initial_detections_seed_active_layout_immediately(self):
        tracker = self.make_tracker()

        update = tracker.update(
            [
                obs(OccupancyState.EMPTY, (100, 100, 200, 200)),
                obs(OccupancyState.OCCUPIED, (400, 100, 500, 200)),
            ],
            0.0,
            FRAME_SHAPE,
        )

        self.assertEqual(update.layout_version, 1)
        self.assertEqual(update.layout_state, "STABLE")
        self.assertEqual(len(update.visible_tracks), 2)
        self.assertEqual(update.raw_table_count, 2)
        self.assertEqual(update.stable_table_count, 2)

    def test_one_missing_sample_keeps_table_temporarily_missing(self):
        tracker = self.make_tracker()
        tracker.update([obs(OccupancyState.EMPTY, (100, 100, 200, 200))], 0.0, FRAME_SHAPE)

        update = tracker.update([], 15.0, FRAME_SHAPE)

        self.assertEqual(update.layout_version, 1)
        self.assertEqual(update.layout_state, "LAYOUT_CHANGE_PENDING")
        self.assertEqual(len(update.visible_tracks), 1)
        self.assertTrue(update.visible_tracks[0].predicted)
        self.assertEqual(update.visible_tracks[0].layout_state, "TEMP_MISSING")
        self.assertEqual(update.pending_layout_changes[0]["change_type"], "REMOVED")

    def test_single_frame_false_table_is_not_added(self):
        tracker = self.make_tracker()
        tracker.update([obs(OccupancyState.EMPTY, (100, 100, 200, 200))], 0.0, FRAME_SHAPE)

        first = tracker.update(
            [
                obs(OccupancyState.EMPTY, (100, 100, 200, 200)),
                obs(OccupancyState.EMPTY, (700, 100, 800, 200)),
            ],
            15.0,
            FRAME_SHAPE,
        )
        second = tracker.update([obs(OccupancyState.EMPTY, (100, 100, 200, 200))], 20.0, FRAME_SHAPE)
        third = tracker.update([obs(OccupancyState.EMPTY, (100, 100, 200, 200))], 25.0, FRAME_SHAPE)

        self.assertEqual(first.layout_state, "LAYOUT_CHANGE_PENDING")
        self.assertEqual(second.stable_table_count, 1)
        self.assertEqual(third.stable_table_count, 1)
        self.assertEqual(third.layout_version, 1)

    def test_new_table_is_added_after_repeated_confirmation(self):
        tracker = self.make_tracker()
        tracker.update([obs(OccupancyState.EMPTY, (100, 100, 200, 200))], 0.0, FRAME_SHAPE)

        for timestamp in (15.0, 20.0):
            update = tracker.update(
                [
                    obs(OccupancyState.EMPTY, (100, 100, 200, 200)),
                    obs(OccupancyState.OCCUPIED, (700, 100, 800, 200)),
                ],
                timestamp,
                FRAME_SHAPE,
            )
            self.assertEqual(update.stable_table_count, 1)
            self.assertEqual(update.layout_state, "LAYOUT_CHANGE_PENDING")

        committed = tracker.update(
            [
                obs(OccupancyState.EMPTY, (100, 100, 200, 200)),
                obs(OccupancyState.OCCUPIED, (700, 100, 800, 200)),
            ],
            25.0,
            FRAME_SHAPE,
        )

        self.assertEqual(committed.layout_version, 2)
        self.assertEqual(committed.layout_state, "LAYOUT_CHANGED")
        self.assertEqual(committed.stable_table_count, 2)
        self.assertEqual(committed.committed_layout_changes[0]["change_type"], "ADDED")

    def test_table_move_is_committed_after_repeated_confirmation(self):
        tracker = self.make_tracker()
        initial = tracker.update(
            [obs(OccupancyState.EMPTY, (100, 100, 200, 200))],
            0.0,
            FRAME_SHAPE,
        )
        table_id = initial.visible_tracks[0].track_id

        for timestamp in (15.0, 20.0):
            update = tracker.update(
                [obs(OccupancyState.OCCUPIED, (420, 100, 520, 200))],
                timestamp,
                FRAME_SHAPE,
            )
            self.assertEqual(update.layout_state, "LAYOUT_CHANGE_PENDING")
            self.assertEqual(update.visible_tracks[0].track_id, table_id)
            self.assertEqual(update.visible_tracks[0].layout_state, "MOVE_PENDING")

        committed = tracker.update(
            [obs(OccupancyState.OCCUPIED, (420, 100, 520, 200))],
            25.0,
            FRAME_SHAPE,
        )

        self.assertEqual(committed.layout_version, 2)
        self.assertEqual(committed.layout_state, "LAYOUT_CHANGED")
        self.assertEqual(committed.visible_tracks[0].track_id, table_id)
        self.assertEqual(committed.visible_tracks[0].layout_version, 2)
        self.assertEqual(committed.visible_tracks[0].stable_state, OccupancyState.OCCUPIED)
        self.assertEqual(committed.committed_layout_changes[0]["change_type"], "MOVED")

    def test_table_is_retired_after_repeated_missing_without_empty_transition(self):
        tracker = self.make_tracker()
        initial = tracker.update(
            [obs(OccupancyState.OCCUPIED, (100, 100, 200, 200))],
            0.0,
            FRAME_SHAPE,
        )
        table_id = initial.visible_tracks[0].track_id

        tracker.update([], 15.0, FRAME_SHAPE)
        tracker.update([], 20.0, FRAME_SHAPE)
        retired = tracker.update([], 25.0, FRAME_SHAPE)

        self.assertEqual(retired.visible_tracks, [])
        self.assertEqual(retired.layout_version, 2)
        self.assertEqual(retired.committed_layout_changes[0]["change_type"], "REMOVED")
        self.assertFalse(
            any(event["type"] == "state_change" for event in retired.events)
        )
        self.assertIn(
            {"type": "left_view", "table_id": table_id, "last_state": "occupied", "timestamp": 25.0},
            retired.events,
        )


if __name__ == "__main__":
    unittest.main()

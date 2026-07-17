"""Unit tests for the adaptive 15s/5s sampling cadence (2차 판단 스케줄링)."""

from __future__ import annotations

import unittest

from seatnow_core import (
    AdaptiveCadenceController,
    OccupancyState,
    TableObservation,
    TableTracker,
    Track,
)


def observation(state: OccupancyState, box=(100.0, 100.0, 200.0, 200.0)):
    return TableObservation(
        box=box,
        table_confidence=0.9,
        raw_state=state,
        raw_score=0.9,
        reason="fixture",
    )


def track(
    stable: OccupancyState,
    pending: OccupancyState | None = None,
    pending_count: int = 0,
):
    return Track(
        track_id=1,
        box=(100.0, 100.0, 200.0, 200.0),
        stable_state=stable,
        last_observation=observation(stable),
        first_seen=0.0,
        last_seen=0.0,
        pending_state=pending,
        pending_count=pending_count,
    )


class ControllerConditionTests(unittest.TestCase):
    def setUp(self):
        self.controller = AdaptiveCadenceController(base_seconds=15.0, fast_seconds=5.0)

    def test_no_tracks_uses_base(self):
        self.assertEqual(self.controller.next_interval([]), 15.0)

    def test_empty_seat_with_pending_occupied_is_fast(self):
        tracks = [track(OccupancyState.EMPTY, OccupancyState.OCCUPIED, 1)]
        self.assertEqual(self.controller.next_interval(tracks), 5.0)

    def test_stable_states_without_pending_use_base(self):
        tracks = [
            track(OccupancyState.EMPTY),
            track(OccupancyState.OCCUPIED),
        ]
        self.assertEqual(self.controller.next_interval(tracks), 15.0)

    def test_vacating_seat_does_not_accelerate(self):
        # occupied -> empty pending must stay on the base cadence (spec).
        tracks = [track(OccupancyState.OCCUPIED, OccupancyState.EMPTY, 2)]
        self.assertEqual(self.controller.next_interval(tracks), 15.0)

    def test_unknown_seat_with_pending_occupied_uses_base(self):
        tracks = [track(OccupancyState.UNKNOWN, OccupancyState.OCCUPIED, 1)]
        self.assertEqual(self.controller.next_interval(tracks), 15.0)

    def test_one_qualifying_track_among_many_is_enough(self):
        tracks = [
            track(OccupancyState.OCCUPIED),
            track(OccupancyState.EMPTY, OccupancyState.OCCUPIED, 1),
        ]
        self.assertEqual(self.controller.next_interval(tracks), 5.0)

    def test_invalid_intervals_are_rejected(self):
        with self.assertRaises(ValueError):
            AdaptiveCadenceController(base_seconds=5.0, fast_seconds=15.0)
        with self.assertRaises(ValueError):
            AdaptiveCadenceController(base_seconds=0.0, fast_seconds=0.0)


class EndToEndDebounceTests(unittest.TestCase):
    """Fast cadence engages on first evidence and disengages on confirmation."""

    FRAME_SHAPE = (480, 640)

    def test_sit_down_confirms_within_two_fast_samples(self):
        tracker = TableTracker(occupy_confirmations=2, empty_confirmations=3)
        controller = AdaptiveCadenceController(base_seconds=15.0, fast_seconds=5.0)

        update = tracker.update([observation(OccupancyState.EMPTY)], 0.0, self.FRAME_SHAPE)
        self.assertEqual(controller.next_interval(update.all_tracks), 15.0)

        update = tracker.update([observation(OccupancyState.OCCUPIED)], 15.0, self.FRAME_SHAPE)
        self.assertEqual(controller.next_interval(update.all_tracks), 5.0)
        self.assertEqual(update.all_tracks[0].stable_state, OccupancyState.EMPTY)

        update = tracker.update([observation(OccupancyState.OCCUPIED)], 20.0, self.FRAME_SHAPE)
        self.assertEqual(update.all_tracks[0].stable_state, OccupancyState.OCCUPIED)
        self.assertEqual(controller.next_interval(update.all_tracks), 15.0)

    def test_vanished_evidence_returns_to_base(self):
        tracker = TableTracker(occupy_confirmations=2, empty_confirmations=3)
        controller = AdaptiveCadenceController(base_seconds=15.0, fast_seconds=5.0)

        tracker.update([observation(OccupancyState.EMPTY)], 0.0, self.FRAME_SHAPE)
        update = tracker.update([observation(OccupancyState.OCCUPIED)], 15.0, self.FRAME_SHAPE)
        self.assertEqual(controller.next_interval(update.all_tracks), 5.0)

        update = tracker.update([observation(OccupancyState.EMPTY)], 20.0, self.FRAME_SHAPE)
        self.assertEqual(update.all_tracks[0].stable_state, OccupancyState.EMPTY)
        self.assertEqual(controller.next_interval(update.all_tracks), 15.0)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the adaptive 15s/5s sampling cadence (2차 판단 스케줄링).

A trigger (occupied evidence on a stable-EMPTY seat) must always be followed
by `fast_cycles` fast-interval samples, even when the transition confirms or
is refuted before the series completes.
"""

from __future__ import annotations

import unittest

from engine.seatnow_core import (
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


class TriggerConditionTests(unittest.TestCase):
    def test_empty_seat_with_pending_occupied_triggers(self):
        tracks = [track(OccupancyState.EMPTY, OccupancyState.OCCUPIED, 1)]
        self.assertTrue(AdaptiveCadenceController.wants_fast(tracks))

    def test_stable_states_without_pending_do_not_trigger(self):
        tracks = [
            track(OccupancyState.EMPTY),
            track(OccupancyState.OCCUPIED),
        ]
        self.assertFalse(AdaptiveCadenceController.wants_fast(tracks))

    def test_vacating_seat_does_not_trigger(self):
        # occupied -> empty pending must stay on the base cadence (spec).
        tracks = [track(OccupancyState.OCCUPIED, OccupancyState.EMPTY, 2)]
        self.assertFalse(AdaptiveCadenceController.wants_fast(tracks))

    def test_unknown_seat_with_pending_occupied_does_not_trigger(self):
        tracks = [track(OccupancyState.UNKNOWN, OccupancyState.OCCUPIED, 1)]
        self.assertFalse(AdaptiveCadenceController.wants_fast(tracks))

    def test_one_qualifying_track_among_many_is_enough(self):
        tracks = [
            track(OccupancyState.OCCUPIED),
            track(OccupancyState.EMPTY, OccupancyState.OCCUPIED, 1),
        ]
        self.assertTrue(AdaptiveCadenceController.wants_fast(tracks))


class FastCycleCountdownTests(unittest.TestCase):
    def make_controller(self):
        return AdaptiveCadenceController(
            base_seconds=15.0, fast_seconds=5.0, fast_cycles=3
        )

    def test_no_trigger_stays_on_base(self):
        controller = self.make_controller()
        idle = [track(OccupancyState.EMPTY)]
        self.assertEqual(controller.next_interval(idle), 15.0)
        self.assertEqual(controller.next_interval([]), 15.0)

    def test_trigger_runs_three_fast_cycles_even_after_early_confirmation(self):
        controller = self.make_controller()
        triggered = [track(OccupancyState.EMPTY, OccupancyState.OCCUPIED, 1)]
        confirmed = [track(OccupancyState.OCCUPIED)]

        # Detection sample schedules the first fast recheck...
        self.assertEqual(controller.next_interval(triggered), 5.0)
        # ...and the series completes even though the seat confirmed already.
        self.assertEqual(controller.next_interval(confirmed), 5.0)
        self.assertEqual(controller.next_interval(confirmed), 5.0)
        self.assertEqual(controller.next_interval(confirmed), 15.0)

    def test_trigger_runs_three_fast_cycles_after_refutation(self):
        controller = self.make_controller()
        triggered = [track(OccupancyState.EMPTY, OccupancyState.OCCUPIED, 1)]
        refuted = [track(OccupancyState.EMPTY)]

        self.assertEqual(controller.next_interval(triggered), 5.0)
        self.assertEqual(controller.next_interval(refuted), 5.0)
        self.assertEqual(controller.next_interval(refuted), 5.0)
        self.assertEqual(controller.next_interval(refuted), 15.0)

    def test_new_trigger_rearms_the_full_series(self):
        controller = self.make_controller()
        triggered = [track(OccupancyState.EMPTY, OccupancyState.OCCUPIED, 1)]
        idle = [track(OccupancyState.OCCUPIED)]

        self.assertEqual(controller.next_interval(triggered), 5.0)
        self.assertEqual(controller.next_interval(idle), 5.0)
        # A second seat triggers mid-series: countdown restarts at 3.
        self.assertEqual(controller.next_interval(triggered), 5.0)
        self.assertEqual(controller.next_interval(idle), 5.0)
        self.assertEqual(controller.next_interval(idle), 5.0)
        self.assertEqual(controller.next_interval(idle), 15.0)

    def test_detection_at_15s_yields_20_25_30_45_timeline(self):
        controller = self.make_controller()
        triggered = [track(OccupancyState.EMPTY, OccupancyState.OCCUPIED, 1)]
        confirmed = [track(OccupancyState.OCCUPIED)]

        timeline = [15.0]
        for tracks in (triggered, confirmed, confirmed, confirmed):
            timeline.append(timeline[-1] + controller.next_interval(tracks))
        self.assertEqual(timeline, [15.0, 20.0, 25.0, 30.0, 45.0])

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            AdaptiveCadenceController(base_seconds=5.0, fast_seconds=15.0)
        with self.assertRaises(ValueError):
            AdaptiveCadenceController(base_seconds=0.0, fast_seconds=0.0)
        with self.assertRaises(ValueError):
            AdaptiveCadenceController(fast_cycles=0)


class EndToEndDebounceTests(unittest.TestCase):
    """Fast series engages on first evidence and always runs to completion."""

    FRAME_SHAPE = (480, 640)

    def test_sit_down_confirms_and_series_still_completes(self):
        tracker = TableTracker(occupy_confirmations=2, empty_confirmations=3)
        controller = AdaptiveCadenceController(
            base_seconds=15.0, fast_seconds=5.0, fast_cycles=3
        )

        update = tracker.update([observation(OccupancyState.EMPTY)], 0.0, self.FRAME_SHAPE)
        self.assertEqual(controller.next_interval(update.all_tracks), 15.0)

        # t=15: detection -> fast series begins.
        update = tracker.update([observation(OccupancyState.OCCUPIED)], 15.0, self.FRAME_SHAPE)
        self.assertEqual(update.all_tracks[0].stable_state, OccupancyState.EMPTY)
        self.assertEqual(controller.next_interval(update.all_tracks), 5.0)

        # t=20: confirmed occupied, but the recheck series keeps running.
        update = tracker.update([observation(OccupancyState.OCCUPIED)], 20.0, self.FRAME_SHAPE)
        self.assertEqual(update.all_tracks[0].stable_state, OccupancyState.OCCUPIED)
        self.assertEqual(controller.next_interval(update.all_tracks), 5.0)

        # t=25: third fast cycle.
        update = tracker.update([observation(OccupancyState.OCCUPIED)], 25.0, self.FRAME_SHAPE)
        self.assertEqual(controller.next_interval(update.all_tracks), 5.0)

        # t=30: series complete -> back to base.
        update = tracker.update([observation(OccupancyState.OCCUPIED)], 30.0, self.FRAME_SHAPE)
        self.assertEqual(controller.next_interval(update.all_tracks), 15.0)

    def test_vanished_evidence_still_completes_the_series(self):
        tracker = TableTracker(occupy_confirmations=2, empty_confirmations=3)
        controller = AdaptiveCadenceController(
            base_seconds=15.0, fast_seconds=5.0, fast_cycles=3
        )

        tracker.update([observation(OccupancyState.EMPTY)], 0.0, self.FRAME_SHAPE)
        update = tracker.update([observation(OccupancyState.OCCUPIED)], 15.0, self.FRAME_SHAPE)
        self.assertEqual(controller.next_interval(update.all_tracks), 5.0)

        for timestamp, expected in ((20.0, 5.0), (25.0, 5.0), (30.0, 15.0)):
            update = tracker.update(
                [observation(OccupancyState.EMPTY)], timestamp, self.FRAME_SHAPE
            )
            self.assertEqual(update.all_tracks[0].stable_state, OccupancyState.EMPTY)
            self.assertEqual(controller.next_interval(update.all_tracks), expected)


if __name__ == "__main__":
    unittest.main()

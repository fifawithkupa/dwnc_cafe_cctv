"""Unit tests for burst-sample majority voting (1차 판단 집계)."""

from __future__ import annotations

import unittest

from engine.seatnow_core import (
    Detection,
    OccupancyState,
    TableObservation,
    aggregate_burst_observations,
)


def observation(
    state: OccupancyState,
    box=(100.0, 100.0, 200.0, 200.0),
    *,
    score: float = 0.9,
    source: str = "detected",
    layout_id=None,
    reason: str = "fixture",
    objects=None,
    provisional: bool = False,
):
    return TableObservation(
        box=box,
        table_confidence=0.9,
        raw_state=state,
        raw_score=score,
        source=source,
        objects=list(objects or []),
        reason=reason,
        provisional=provisional,
        layout_id=layout_id,
    )


class LayoutMatchingTests(unittest.TestCase):
    def test_layout_zones_match_by_id_regardless_of_order(self):
        zone_a = (0.0, 0.0, 100.0, 100.0)
        zone_b = (200.0, 0.0, 300.0, 100.0)
        center = [
            observation(OccupancyState.EMPTY, zone_a, source="layout", layout_id=1),
            observation(OccupancyState.EMPTY, zone_b, source="layout", layout_id=2),
        ]
        sides = [
            [
                observation(OccupancyState.OCCUPIED, zone_b, source="layout", layout_id=2),
                observation(OccupancyState.OCCUPIED, zone_a, source="layout", layout_id=1),
            ]
            for _ in range(2)
        ]
        result = aggregate_burst_observations([sides[0], center, sides[1]], 1)
        self.assertEqual(result[0].raw_state, OccupancyState.OCCUPIED)
        self.assertEqual(result[1].raw_state, OccupancyState.OCCUPIED)
        self.assertEqual(result[0].vote_counts, {"empty": 1, "occupied": 2})


class DetectedMatchingTests(unittest.TestCase):
    def test_jittered_boxes_match_by_iou(self):
        center = [observation(OccupancyState.EMPTY, (100.0, 100.0, 200.0, 200.0))]
        jittered = [
            [observation(OccupancyState.OCCUPIED, (105.0, 95.0, 205.0, 195.0))],
            [observation(OccupancyState.OCCUPIED, (96.0, 104.0, 196.0, 204.0))],
        ]
        result = aggregate_burst_observations(
            [jittered[0], center, jittered[1]], 1
        )
        self.assertEqual(result[0].raw_state, OccupancyState.OCCUPIED)
        self.assertEqual(result[0].vote_counts, {"empty": 1, "occupied": 2})

    def test_distant_boxes_do_not_vote(self):
        center = [observation(OccupancyState.EMPTY, (100.0, 100.0, 200.0, 200.0))]
        far = [[observation(OccupancyState.OCCUPIED, (400.0, 400.0, 500.0, 500.0))]]
        result = aggregate_burst_observations([far[0], center], 1)
        self.assertEqual(result[0].raw_state, OccupancyState.EMPTY)
        self.assertEqual(result[0].vote_counts, {"empty": 1})

    def test_source_mismatch_does_not_vote(self):
        center = [observation(OccupancyState.EMPTY)]
        side = [[observation(OccupancyState.OCCUPIED, source="inferred-seat")]]
        result = aggregate_burst_observations([side[0], center], 1)
        self.assertEqual(result[0].raw_state, OccupancyState.EMPTY)
        self.assertEqual(result[0].vote_counts, {"empty": 1})


class MajorityVoteTests(unittest.TestCase):
    def _run(self, side_states, center_state=OccupancyState.EMPTY):
        center = [observation(center_state)]
        frames = [[observation(state)] for state in side_states]
        frames.insert(len(frames) // 2, center)
        center_index = len(side_states) // 2
        return aggregate_burst_observations(frames, center_index)[0]

    def test_occupied_majority_wins(self):
        result = self._run(
            [
                OccupancyState.OCCUPIED,
                OccupancyState.OCCUPIED,
                OccupancyState.OCCUPIED,
                OccupancyState.EMPTY,
            ]
        )
        self.assertEqual(result.raw_state, OccupancyState.OCCUPIED)
        self.assertEqual(result.vote_counts, {"occupied": 3, "empty": 2})

    def test_empty_majority_wins(self):
        result = self._run(
            [
                OccupancyState.EMPTY,
                OccupancyState.EMPTY,
                OccupancyState.OCCUPIED,
                OccupancyState.EMPTY,
            ],
            center_state=OccupancyState.OCCUPIED,
        )
        self.assertEqual(result.raw_state, OccupancyState.EMPTY)

    def test_tie_falls_back_to_center_state(self):
        result = self._run(
            [
                OccupancyState.OCCUPIED,
                OccupancyState.OCCUPIED,
                OccupancyState.EMPTY,
                OccupancyState.UNKNOWN,
            ]
        )
        # 2 occupied vs 2 empty (center included): center's EMPTY stands.
        self.assertEqual(result.raw_state, OccupancyState.EMPTY)

    def test_all_unknown_keeps_center_state(self):
        result = self._run(
            [OccupancyState.UNKNOWN, OccupancyState.UNKNOWN],
            center_state=OccupancyState.UNKNOWN,
        )
        self.assertEqual(result.raw_state, OccupancyState.UNKNOWN)

    def test_center_ignore_passes_through(self):
        center = [observation(OccupancyState.IGNORE, reason="border_cropped")]
        sides = [[observation(OccupancyState.OCCUPIED)] for _ in range(4)]
        frames = sides[:2] + [center] + sides[2:]
        result = aggregate_burst_observations(frames, 2)[0]
        self.assertEqual(result.raw_state, OccupancyState.IGNORE)
        self.assertEqual(result.reason, "border_cropped")

    def test_single_frame_burst_degrades_gracefully(self):
        center = [observation(OccupancyState.OCCUPIED)]
        result = aggregate_burst_observations([center], 0)[0]
        self.assertEqual(result.raw_state, OccupancyState.OCCUPIED)
        self.assertEqual(result.vote_counts, {"occupied": 1})

    def test_empty_input(self):
        self.assertEqual(aggregate_burst_observations([], 0), [])


class DonorMetadataTests(unittest.TestCase):
    def test_majority_flip_copies_evidence_from_best_donor(self):
        cup = Detection(name="cup", box=(120.0, 90.0, 140.0, 110.0), confidence=0.7)
        center = [observation(OccupancyState.EMPTY, reason="no_customer_evidence")]
        weak_donor = observation(
            OccupancyState.OCCUPIED, score=0.6, reason="objects:phone"
        )
        strong_donor = observation(
            OccupancyState.OCCUPIED,
            score=0.9,
            reason="objects:cup",
            objects=[cup],
        )
        result = aggregate_burst_observations(
            [[weak_donor], center, [strong_donor]], 1
        )[0]
        self.assertEqual(result.raw_state, OccupancyState.OCCUPIED)
        self.assertEqual(result.reason, "objects:cup")
        self.assertEqual(result.raw_score, 0.9)
        self.assertEqual([obj.name for obj in result.objects], ["cup"])
        # Geometry stays anchored to the center frame's box.
        self.assertEqual(result.box, (100.0, 100.0, 200.0, 200.0))


if __name__ == "__main__":
    unittest.main()

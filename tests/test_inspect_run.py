"""Unit tests for the three-layer reading table.

Fixtures are hand-written JSONL records: the point of the table is that a
number breaking in one layer is visible as that layer's number, so the tests
build records where exactly one layer is wrong.
"""

from __future__ import annotations

import unittest

from inspect_run import Row, build_rows, recall
from judge_frames import Judgement


def record(timestamp, person=3, chair=7, table=2, seated=2, standing=1, unknown=0,
           occupied=2, empty=0, seat_unknown=1, ignore=0):
    return {
        "timestamp": timestamp,
        "raw_detections": {
            "counts": {"person": person, "chair": chair, "dining table": table}
        },
        "summary": {
            "seated_poses": seated,
            "standing_poses": standing,
            "unknown_poses": unknown,
            "occupied": occupied,
            "empty": empty,
            "unknown": seat_unknown,
            "ignore": ignore,
        },
        "tables": [],
    }


def truth(total=3, uncertain=False, error=None):
    return Judgement(
        stem="ignored",
        people_total=total,
        people_seated=total,
        people_standing=0,
        uncertain=uncertain,
        error=error,
    )


class BuildRowsTests(unittest.TestCase):
    def test_row_stem_matches_the_frame_filename(self):
        rows = build_rows([record(15.0)], {})
        self.assertEqual(rows[0].stem, "t0015.0s")

    def test_judgement_is_matched_by_stem(self):
        rows = build_rows([record(15.0)], {"t0015.0s": truth(total=4)})
        self.assertEqual(rows[0].truth, 4)

    def test_missing_judgement_leaves_truth_empty(self):
        rows = build_rows([record(15.0)], {})
        self.assertIsNone(rows[0].truth)
        self.assertIsNone(rows[0].detector_gap)

    def test_missing_raw_detections_counts_as_zero(self):
        bare = record(15.0)
        del bare["raw_detections"]
        rows = build_rows([bare], {})
        self.assertEqual(rows[0].det_person, 0)

    def test_pose_total_sums_the_three_pose_states(self):
        rows = build_rows([record(15.0, seated=2, standing=1, unknown=1)], {})
        self.assertEqual(rows[0].pose_total, 4)

    def test_gap_is_found_minus_truth(self):
        rows = build_rows([record(15.0, person=1)], {"t0015.0s": truth(total=3)})
        self.assertEqual(rows[0].detector_gap, -2)

    def test_uncertain_judgement_is_marked_excluded(self):
        rows = build_rows([record(15.0)], {"t0015.0s": truth(uncertain=True)})
        self.assertEqual(rows[0].excluded, "uncertain")

    def test_failed_judgement_is_marked_excluded(self):
        rows = build_rows([record(15.0)], {"t0015.0s": truth(error="timeout")})
        self.assertEqual(rows[0].excluded, "error")


class RecallTests(unittest.TestCase):
    def test_perfect_detection_is_one(self):
        rows = build_rows(
            [record(0.0, person=3), record(15.0, person=3)],
            {"t0000.0s": truth(3), "t0015.0s": truth(3)},
        )
        self.assertEqual(recall(rows, "detector").value, 1.0)

    def test_half_the_people_is_one_half(self):
        rows = build_rows(
            [record(0.0, person=2), record(15.0, person=2)],
            {"t0000.0s": truth(4), "t0015.0s": truth(4)},
        )
        self.assertEqual(recall(rows, "detector").value, 0.5)

    def test_over_detection_does_not_exceed_one(self):
        # Recall answers "how many of the people present were found", so a
        # frame with more boxes than people is capped, not credited.
        rows = build_rows([record(0.0, person=5)], {"t0000.0s": truth(3)})
        self.assertEqual(recall(rows, "detector").value, 1.0)

    def test_uncertain_frames_are_left_out_of_the_score(self):
        rows = build_rows(
            [record(0.0, person=3), record(15.0, person=0)],
            {"t0000.0s": truth(3), "t0015.0s": truth(3, uncertain=True)},
        )
        result = recall(rows, "detector")
        self.assertEqual(result.value, 1.0)
        self.assertEqual(result.scored_frames, 1)
        self.assertEqual(result.excluded_frames, 1)

    def test_pose_layer_scores_separately_from_detector(self):
        rows = build_rows(
            [record(0.0, person=3, seated=1, standing=0, unknown=0)],
            {"t0000.0s": truth(3)},
        )
        self.assertEqual(recall(rows, "detector").value, 1.0)
        self.assertAlmostEqual(recall(rows, "pose").value, 1 / 3)

    def test_no_truth_at_all_gives_no_value(self):
        rows = build_rows([record(0.0)], {})
        result = recall(rows, "detector")
        self.assertIsNone(result.value)
        self.assertEqual(result.scored_frames, 0)

    def test_unknown_layer_is_rejected(self):
        with self.assertRaises(ValueError):
            recall(build_rows([record(0.0)], {}), "seats")


if __name__ == "__main__":
    unittest.main()

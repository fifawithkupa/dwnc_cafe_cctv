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


from inspect_run import disagreements, reason_distribution, render_summary, render_table


def record_with_reasons(timestamp, *reason_codes):
    """A record whose seat_report carries one plain table per reason code."""
    payload = record(timestamp)
    payload["seat_report"] = {
        "seats": [
            {
                "seat_id": f"T{index}",
                "kind": "table",
                "capacity": 1,
                "state": "unknown",
                "reason_code": code,
            }
            for index, code in enumerate(reason_codes)
        ]
    }
    return payload


class RenderTableTests(unittest.TestCase):
    def test_every_row_appears(self):
        rows = build_rows([record(0.0), record(15.0)], {})
        table = render_table(rows)
        self.assertIn("t0000.0s", table)
        self.assertIn("t0015.0s", table)

    def test_missing_truth_renders_as_a_blank_to_fill(self):
        # The table must stay usable with no Codex run at all: a person can
        # write the counts into this column by hand.
        table = render_table(build_rows([record(0.0)], {}))
        self.assertIn("___", table)

    def test_excluded_row_says_why(self):
        rows = build_rows([record(0.0)], {"t0000.0s": truth(uncertain=True)})
        self.assertIn("uncertain", render_table(rows))

    def test_disagreement_is_flagged(self):
        rows = build_rows([record(0.0, person=1)], {"t0000.0s": truth(3)})
        self.assertIn("!!", render_table(rows))

    def test_agreement_is_not_flagged(self):
        rows = build_rows([record(0.0, person=3)], {"t0000.0s": truth(3)})
        self.assertNotIn("!!", render_table(rows))


class ReasonDistributionTests(unittest.TestCase):
    def test_codes_are_grouped_by_what_fixes_them(self):
        records = [
            record_with_reasons(0.0, "occluded_lower_body", "pose_low_keypoints"),
            record_with_reasons(15.0, "occluded_lower_body"),
        ]
        distribution = reason_distribution(records)
        self.assertEqual(distribution["geometry"], 2)
        self.assertEqual(distribution["model"], 1)

    def test_unseen_groups_are_zero_not_absent(self):
        distribution = reason_distribution([record_with_reasons(0.0, "person_seated")])
        self.assertEqual(distribution["model"], 0)

    def test_unknown_code_does_not_crash(self):
        distribution = reason_distribution([record_with_reasons(0.0, "made_up_code")])
        self.assertEqual(distribution["other"], 1)

    def test_bar_zone_reason_counts_are_added(self):
        # A counted_zone reports {code: count}, not one code per seat
        # (seatnow_report.py:170-182).  Reading only the plain-table shape
        # would silently drop every bar seat's reason.
        payload = record(0.0)
        payload["seat_report"] = {
            "seats": [
                {
                    "seat_id": "BAR",
                    "kind": "counted_zone",
                    "capacity": 3,
                    "occupied": 0,
                    "free": 1,
                    "unknown": 2,
                    "reason_codes": {"occluded_lower_body": 2},
                }
            ]
        }
        self.assertEqual(reason_distribution([payload])["geometry"], 2)

    def test_record_without_a_seat_report_is_skipped(self):
        self.assertEqual(reason_distribution([record(0.0)])["geometry"], 0)


class DisagreementTests(unittest.TestCase):
    def test_only_rows_with_a_gap_are_returned(self):
        rows = build_rows(
            [record(0.0, person=3), record(15.0, person=1)],
            {"t0000.0s": truth(3), "t0015.0s": truth(3)},
        )
        self.assertEqual([row.stem for row in disagreements(rows)], ["t0015.0s"])

    def test_rows_without_truth_are_not_disagreements(self):
        self.assertEqual(disagreements(build_rows([record(0.0)], {})), [])


class RenderSummaryTests(unittest.TestCase):
    def test_both_layer_recalls_appear(self):
        rows = build_rows([record(0.0, person=3)], {"t0000.0s": truth(3)})
        summary = render_summary(rows, [record(0.0)])
        self.assertIn("검출", summary)
        self.assertIn("포즈", summary)

    def test_excluded_count_is_reported(self):
        rows = build_rows([record(0.0)], {"t0000.0s": truth(uncertain=True)})
        self.assertIn("제외", render_summary(rows, [record(0.0)]))


from inspect_run import over_detection


class OverDetectionTests(unittest.TestCase):
    """Recall cannot see the failure this café actually has.

    Recall caps found at truth, so a frame with 7 boxes over 2 people scores
    a perfect 1.00.  The real failure here is the opposite direction, and a
    harness whose only number hides its subject is worse than no harness.
    """

    def test_extra_people_are_counted(self):
        rows = build_rows([record(0.0, person=7)], {"t0000.0s": truth(2)})
        result = over_detection(rows, "detector")
        self.assertEqual(result.frames_over, 1)
        self.assertEqual(result.extra_total, 5)
        self.assertEqual(result.worst_gap, 5)

    def test_missing_people_are_counted_separately(self):
        rows = build_rows([record(0.0, person=1)], {"t0000.0s": truth(3)})
        result = over_detection(rows, "detector")
        self.assertEqual(result.frames_under, 1)
        self.assertEqual(result.frames_over, 0)
        self.assertEqual(result.extra_total, 0)

    def test_exact_frames_are_counted(self):
        rows = build_rows([record(0.0, person=2)], {"t0000.0s": truth(2)})
        result = over_detection(rows, "detector")
        self.assertEqual(result.frames_exact, 1)

    def test_excluded_frames_are_not_scored(self):
        rows = build_rows([record(0.0, person=9)], {"t0000.0s": truth(2, uncertain=True)})
        result = over_detection(rows, "detector")
        self.assertEqual(result.scored_frames, 0)
        self.assertEqual(result.frames_over, 0)

    def test_pose_layer_is_scored_separately(self):
        rows = build_rows(
            [record(0.0, person=7, seated=0, standing=2, unknown=0)],
            {"t0000.0s": truth(2)},
        )
        self.assertEqual(over_detection(rows, "detector").extra_total, 5)
        self.assertEqual(over_detection(rows, "pose").extra_total, 0)

    def test_unknown_layer_is_rejected(self):
        with self.assertRaises(ValueError):
            over_detection(build_rows([record(0.0)], {}), "seats")

    def test_summary_reports_over_detection(self):
        rows = build_rows([record(0.0, person=7)], {"t0000.0s": truth(2)})
        self.assertIn("과탐", render_summary(rows, [record(0.0)]))


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the app-facing seat report contract."""

from __future__ import annotations

import unittest

from seatnow_report import ACTIONABLE_GROUPS, REASON_GROUPS, ReasonCode, classify_reason


class ClassifyReasonTests(unittest.TestCase):
    def test_occluded_lower_body_is_promoted_from_pose_reason(self):
        self.assertEqual(
            classify_reason("unknown", "compact_occluded_pose=0.82", predicted=False),
            ReasonCode.OCCLUDED_LOWER_BODY,
        )

    def test_low_keypoints_is_promoted_from_pose_reason(self):
        self.assertEqual(
            classify_reason("unknown", "insufficient_keypoints", predicted=False),
            ReasonCode.POSE_LOW_KEYPOINTS,
        )

    def test_unpromoted_pose_unknown_falls_back_to_ambiguous(self):
        self.assertEqual(
            classify_reason("unknown", "nearby_person_pose_unknown", predicted=False),
            ReasonCode.AMBIGUOUS_ASSOCIATION,
        )

    def test_promoted_pose_cause_is_unwrapped(self):
        self.assertEqual(
            classify_reason(
                "unknown",
                "nearby_person_pose_unknown:compact_occluded_pose=0.82",
                predicted=False,
            ),
            ReasonCode.OCCLUDED_LOWER_BODY,
        )

    def test_spanning_seats_has_its_own_code(self):
        self.assertEqual(
            classify_reason("unknown", "spans_multiple_seats", predicted=False),
            ReasonCode.SPANS_MULTIPLE_SEATS,
        )

    def test_predicted_track_is_time_group_not_an_engineering_problem(self):
        code = classify_reason("occupied", "seated:1", predicted=True)

        self.assertEqual(code, ReasonCode.TRACK_PREDICTED)
        self.assertIn(code, REASON_GROUPS["time"])

    def test_border_cropped_is_an_install_problem(self):
        code = classify_reason("ignore", "border_cropped", predicted=False)

        self.assertEqual(code, ReasonCode.BORDER_CROPPED)
        self.assertIn(code, REASON_GROUPS["install"])

    def test_occupied_reasons_are_classified(self):
        self.assertEqual(
            classify_reason("occupied", "seated:2", predicted=False),
            ReasonCode.PERSON_SEATED,
        )
        self.assertEqual(
            classify_reason("occupied", "objects:cup,laptop", predicted=False),
            ReasonCode.BELONGINGS,
        )
        self.assertEqual(
            classify_reason("occupied", "occupied_chairs:1", predicted=False),
            ReasonCode.OCCUPIED_CHAIR,
        )

    def test_empty_is_classified(self):
        self.assertEqual(
            classify_reason("empty", "no_customer_evidence", predicted=False),
            ReasonCode.NO_CUSTOMER_EVIDENCE,
        )

    def test_every_code_belongs_to_exactly_one_group(self):
        seen = [code for codes in REASON_GROUPS.values() for code in codes]

        self.assertEqual(sorted(seen), sorted(set(seen)))
        self.assertEqual(set(seen), set(ReasonCode))

    def test_actionable_groups_exclude_install_and_settled(self):
        self.assertEqual(ACTIONABLE_GROUPS, ("geometry", "model"))


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the app-facing seat report contract."""

from __future__ import annotations

import unittest

from seatnow_report import (
    ACTIONABLE_GROUPS,
    REASON_GROUPS,
    ReasonCode,
    build_seat_report,
    classify_reason,
)


def table_dict(
    name,
    state,
    reason="no_customer_evidence",
    kind="table",
    zone_name=None,
    capacity=1,
    predicted=False,
    confidence=0.9,
):
    return {
        "layout_name": name,
        "label": name,
        "state": state,
        "raw_state": state,
        "reason": reason,
        "layout_kind": kind,
        "layout_zone_name": zone_name,
        "layout_capacity": capacity,
        "predicted": predicted,
        "confidence": confidence,
    }


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


class BuildSeatReportTests(unittest.TestCase):
    def test_plain_tables_become_one_seat_each(self):
        report = build_seat_report(
            [
                table_dict("창가1", "occupied", "seated:1"),
                table_dict("창가2", "empty"),
            ],
            tick_at=12.5,
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["tick_at"], 12.5)
        self.assertEqual(len(report["seats"]), 2)
        self.assertEqual(report["seats"][0]["seat_id"], "창가1")
        self.assertEqual(report["seats"][0]["state"], "occupied")
        self.assertEqual(report["seats"][0]["reason_code"], "person_seated")
        self.assertEqual(
            report["totals"],
            {"capacity": 2, "occupied": 1, "free": 1, "unknown": 0},
        )

    def test_counted_zone_seats_are_grouped_into_one_entry(self):
        report = build_seat_report(
            [
                table_dict("BAR-1", "occupied", "seated:1", "counted_zone", "BAR", 3),
                table_dict(
                    "BAR-2", "empty", "no_customer_evidence", "counted_zone", "BAR", 3
                ),
                table_dict(
                    "BAR-3", "unknown", "spans_multiple_seats", "counted_zone", "BAR", 3
                ),
            ],
            tick_at=0.0,
        )

        self.assertEqual(len(report["seats"]), 1)
        zone = report["seats"][0]
        self.assertEqual(zone["seat_id"], "BAR")
        self.assertEqual(zone["kind"], "counted_zone")
        self.assertEqual(zone["capacity"], 3)
        self.assertEqual(zone["occupied"], 1)
        self.assertEqual(zone["free"], 1)
        self.assertEqual(zone["unknown"], 1)
        self.assertEqual(zone["reason_codes"], {"spans_multiple_seats": 1})

    def test_free_never_counts_unknown(self):
        report = build_seat_report(
            [
                table_dict("T1", "unknown", "compact_occluded_pose=0.9"),
                table_dict("T2", "unknown", "insufficient_keypoints"),
                table_dict("T3", "empty"),
            ],
            tick_at=0.0,
        )

        self.assertEqual(report["totals"]["free"], 1)
        self.assertEqual(report["totals"]["unknown"], 2)

    def test_ignore_state_is_excluded_from_capacity(self):
        report = build_seat_report(
            [
                table_dict("T1", "empty"),
                table_dict("T2", "ignore", "border_cropped"),
            ],
            tick_at=0.0,
        )

        self.assertEqual(report["totals"]["capacity"], 1)
        self.assertEqual(len(report["seats"]), 1)

    def test_predicted_track_reports_time_group_reason(self):
        report = build_seat_report(
            [table_dict("T1", "occupied", "seated:1", predicted=True)],
            tick_at=0.0,
        )

        self.assertEqual(report["seats"][0]["reason_code"], "track_predicted")

    def test_zone_parts_always_sum_to_capacity(self):
        report = build_seat_report(
            [
                table_dict("BAR-1", "occupied", "seated:1", "counted_zone", "BAR", 2),
                table_dict("BAR-2", "occupied", "seated:1", "counted_zone", "BAR", 2),
            ],
            tick_at=0.0,
        )

        zone = report["seats"][0]
        self.assertEqual(
            zone["occupied"] + zone["free"] + zone["unknown"], zone["capacity"]
        )

    def test_two_zones_stay_separate_and_keep_input_order(self):
        report = build_seat_report(
            [
                table_dict("BAR-1", "occupied", "seated:1", "counted_zone", "BAR", 1),
                table_dict(
                    "WALL-1", "empty", "no_customer_evidence", "counted_zone", "WALL", 1
                ),
            ],
            tick_at=0.0,
        )

        self.assertEqual([seat["seat_id"] for seat in report["seats"]], ["BAR", "WALL"])


if __name__ == "__main__":
    unittest.main()

"""Tests for the manual-fixture result verifier."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from verify_seatnow import (
    coverage_metrics,
    expected_interval,
    match_fixture,
    scoring_policy,
    verify_records,
)


class FixtureVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "test_video_expectations.json"
        cls.expectations = json.loads(fixture_path.read_text(encoding="utf-8"))

    def exact_records(self):
        video = self.expectations["video"]
        count = math.ceil(video["duration_seconds"])
        records = []
        for timestamp in range(count):
            expected = expected_interval(self.expectations, float(timestamp))["expected"]
            record = {
                "timestamp": float(timestamp),
                "summary": {
                    "occupied": len(expected["occupied"]),
                    "empty": len(expected["empty"]),
                },
                "events": [],
            }
            if timestamp == 0:
                record["run"] = {
                    "profile": "accuracy_default",
                    "input": {
                        "sha256": video["sha256"],
                        "width": video["width_pixels"],
                        "height": video["height_pixels"],
                        "duration": video["duration_seconds"],
                    },
                    "config": {"sample_seconds": 1.0, "start_seconds": 0.0},
                }
            records.append(record)
        return records

    def test_exact_full_timeline_passes(self):
        report = verify_records(self.exact_records(), self.expectations)
        self.assertTrue(report["passed"])
        self.assertEqual(report["frames_exact"], 20)

    def test_count_mismatch_fails(self):
        records = self.exact_records()
        records[6]["summary"]["occupied"] += 1
        report = verify_records(records, self.expectations)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["all_sampled_counts_exact"])

    def test_partial_log_fails_coverage(self):
        report = verify_records(self.exact_records()[:3], self.expectations)
        self.assertFalse(report["checks"]["full_timeline_coverage"])


    def test_relaxed_fixture_passes_without_a_full_timeline(self):
        """A long adaptive-cadence recording has no fixed sample grid (T3)."""
        expectations = dict(
            self.expectations,
            scoring={
                "require_full_timeline": False,
                "require_transition_count": False,
            },
        )
        report = verify_records(self.exact_records()[:3], expectations)

        self.assertFalse(report["checks"]["full_timeline_coverage"])
        self.assertNotIn("full_timeline_coverage", report["required_checks"])
        self.assertTrue(report["passed"])


class ScoringPolicyTests(unittest.TestCase):
    def test_defaults_keep_the_original_strict_checks(self):
        policy = scoring_policy({})
        self.assertTrue(policy["require_full_timeline"])
        self.assertTrue(policy["require_transition_count"])
        self.assertTrue(policy["require_no_scene_change"])

    def test_fixture_can_waive_one_check_without_waiving_the_rest(self):
        policy = scoring_policy({"scoring": {"require_full_timeline": False}})
        self.assertFalse(policy["require_full_timeline"])
        self.assertTrue(policy["require_transition_count"])


class MatchFixtureTests(unittest.TestCase):
    """Logs are matched to fixtures by video content, not by filename."""

    FIXTURES = [
        {"fixture_id": "one", "video": {"sha256": "a" * 64}},
        {"fixture_id": "two", "video": {"sha256": "b" * 64}},
    ]

    def records(self, digest):
        return [{"run": {"input": {"sha256": digest}}}]

    def test_matches_on_sha256(self):
        matched = match_fixture(self.records("b" * 64), self.FIXTURES)
        self.assertEqual(matched["fixture_id"], "two")

    def test_unmatched_log_is_an_error_not_a_silent_wrong_score(self):
        with self.assertRaises(ValueError):
            match_fixture(self.records("c" * 64), self.FIXTURES)

    def test_single_fixture_accepts_a_log_without_metadata(self):
        matched = match_fixture([{}], self.FIXTURES[:1])
        self.assertEqual(matched["fixture_id"], "one")


class CoverageMetricsTests(unittest.TestCase):
    """plan.md T3: quantify the seats the camera never gets to judge."""

    def records(self):
        def table(name, raw_state, missing=0):
            return {
                "layout_name": name,
                "raw_state": raw_state,
                "missing_count": missing,
            }

        return [
            {"tables": [table("A", "occupied"), table("B", "ignore")]},
            {"tables": [table("A", "empty", missing=1), table("B", "ignore")]},
        ]

    def test_ignore_ratio_counts_table_observations_not_frames(self):
        coverage = coverage_metrics(self.records())
        self.assertEqual(coverage["table_observations"], 4)
        self.assertEqual(coverage["ignore_ratio"], 0.5)

    def test_per_table_isolates_the_seat_the_camera_cannot_see(self):
        coverage = coverage_metrics(self.records())
        self.assertEqual(coverage["per_table"]["B"]["ignore_ratio"], 1.0)
        self.assertEqual(coverage["per_table"]["A"]["ignore_ratio"], 0.0)

    def test_missing_count_histogram(self):
        coverage = coverage_metrics(self.records())
        self.assertEqual(coverage["missing_count_histogram"], {"0": 3, "1": 1})
        self.assertEqual(coverage["any_missed_ratio"], 0.25)

    def test_empty_log_does_not_divide_by_zero(self):
        coverage = coverage_metrics([{"tables": []}])
        self.assertEqual(coverage["table_observations"], 0)
        self.assertEqual(coverage["ignore_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()

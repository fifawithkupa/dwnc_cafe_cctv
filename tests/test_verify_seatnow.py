"""Tests for the manual-fixture result verifier."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from verify_seatnow import expected_interval, verify_records


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


if __name__ == "__main__":
    unittest.main()

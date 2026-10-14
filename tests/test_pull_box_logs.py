"""Unit tests for pulling the box's judgment logs to the laptop."""

from __future__ import annotations

import unittest

from edge.pull_box_logs import clock_offset, describe_offset, files_to_pull, parse_listing


class ParseListingTests(unittest.TestCase):
    def test_reads_clock_and_day_files(self):
        box_epoch, files = parse_listing(
            "1791897645.123\n2026-10-13.jsonl 5508983\n2026-10-14.jsonl 120\n"
        )
        self.assertAlmostEqual(box_epoch, 1791897645.123)
        self.assertEqual(files, {"2026-10-13.jsonl": 5508983, "2026-10-14.jsonl": 120})

    def test_ignores_bench_runs_and_summaries(self):
        _, files = parse_listing(
            "1\nrun_4mp_h265.jsonl 4057996\nlast_run_summary.json 1134\n2026-10-13.jsonl 7\n"
        )
        self.assertEqual(files, {"2026-10-13.jsonl": 7})

    def test_empty_answer_is_an_error(self):
        with self.assertRaises(ValueError):
            parse_listing("")


class FilesToPullTests(unittest.TestCase):
    def test_new_day_is_pulled(self):
        self.assertEqual(files_to_pull({"2026-10-14.jsonl": 10}, {}), ["2026-10-14.jsonl"])

    def test_unchanged_day_is_skipped(self):
        manifest = {"2026-10-13.jsonl": {"bytes": 10}}
        self.assertEqual(files_to_pull({"2026-10-13.jsonl": 10}, manifest), [])

    def test_day_that_kept_growing_is_pulled_again(self):
        # 지난번엔 그날이 아직 쌓이던 중이었다.
        manifest = {"2026-10-14.jsonl": {"bytes": 10}}
        self.assertEqual(files_to_pull({"2026-10-14.jsonl": 99}, manifest), ["2026-10-14.jsonl"])

    def test_result_is_in_date_order(self):
        remote = {"2026-10-15.jsonl": 1, "2026-10-13.jsonl": 1, "2026-10-14.jsonl": 1}
        self.assertEqual(
            files_to_pull(remote, {}),
            ["2026-10-13.jsonl", "2026-10-14.jsonl", "2026-10-15.jsonl"],
        )


class ClockOffsetTests(unittest.TestCase):
    def test_compares_with_middle_of_round_trip(self):
        self.assertAlmostEqual(clock_offset(1000.0, 990.0, 1010.0), 0.0)

    def test_box_behind_is_negative(self):
        self.assertAlmostEqual(clock_offset(1000.0, 2000.0, 2000.0), -1000.0)

    def test_small_difference_reads_as_correct(self):
        self.assertIn("맞습니다", describe_offset(-20))

    def test_day_behind_is_spelled_out(self):
        text = describe_offset(-(23 * 3600 + 26 * 60))
        self.assertIn("23시간 26분", text)
        self.assertIn("늦습니다", text)

    def test_box_ahead_says_so(self):
        self.assertIn("5분 빠릅니다", describe_offset(300))


if __name__ == "__main__":
    unittest.main()

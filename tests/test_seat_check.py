"""앱 이름표 목록 ↔ 박스 좌석 파일 대조 (install/seat_check.py)."""

import unittest

from install.seat_check import compare, parse_app_ids, report


class ParseTest(unittest.TestCase):
    def test_blank_and_comment_lines_are_skipped_and_whitespace_trimmed(self):
        lines = ["# 문큐", "", " T1 ", "BAR7-1", "   ", "BAR7-2"]
        self.assertEqual(parse_app_ids(lines), ["T1", "BAR7-1", "BAR7-2"])


class CompareTest(unittest.TestCase):
    BOX = ["T1", "T2", "BAR7-1", "BAR7-2"]

    def test_identical_lists_pass_in_any_order(self):
        self.assertEqual(compare(self.BOX, ["BAR7-2", "T2", "T1", "BAR7-1"]), [])
        self.assertTrue(report(self.BOX, self.BOX).startswith("통과 — 4개"))

    def test_missing_extra_and_count_are_all_named(self):
        problems = compare(self.BOX, ["T1", "T2", "BAR7-1", "BAR7-9"])
        self.assertIn("`BAR7-2` 이(가) 앱에 없음", problems)
        self.assertIn("`BAR7-9` 이(가) 박스에 없음", problems)
        self.assertFalse(any("개수" in p for p in problems))  # 4 대 4 — 개수는 같다

    def test_case_difference_is_called_out_as_such(self):
        problems = compare(self.BOX, ["t1", "T2", "BAR7-1", "BAR7-2"])
        self.assertIn("`t1` 는 대소문자가 다름 — 박스는 `T1`", problems)

    def test_duplicates_and_count_mismatch(self):
        problems = compare(self.BOX, ["T1", "T1", "T2", "BAR7-1", "BAR7-2"])
        self.assertIn("앱 목록에 `T1` 가 두 번 있음", problems)
        self.assertIn("개수 다름 — 박스 4개, 앱 5개", problems)
        self.assertTrue(report(self.BOX, ["T1"]).startswith("실패\n"))


if __name__ == "__main__":
    unittest.main()

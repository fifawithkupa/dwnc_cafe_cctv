import unittest

from checks.answer_key import (
    Answer,
    AnswerKey,
    normalize_seat,
    parse_answer_key,
    parse_evidence,
    parse_table,
    validate,
)


TABLE = """
| 시간(s) | T1        | T2  | T3    | Bar7-6 | 비고     |
| ----- | --------- | --- | ----- | ------ | ------ |
| 0     | X (t)     | O   | O     | O      |        |
| 30    | X (t,c,s) | O   | X (s) | O ( c) | 테이블 옮김 |
"""

LINES = """
# angle1 정답지

1번 사진 (0초)
- T1 : 사용중
- T2 : 빈자리
- BAR7-4 : 모름 ← 사람이 완전히 가림

2번 사진 (15초)
- T1 : 사용중 ← 책상에 노트북 있는데 못 잡음
- t2 : 빔
- bar7-4 : occupied
"""


class TableTests(unittest.TestCase):
    def test_reads_the_seconds_column(self):
        key = parse_answer_key(TABLE)

        self.assertEqual(key.timestamps(), [0.0, 30.0])

    def test_reads_the_state_letters(self):
        key = parse_answer_key(TABLE)

        self.assertEqual(key.lookup(0.0, "T1").state, "occupied")
        self.assertEqual(key.lookup(0.0, "T2").state, "empty")
        self.assertEqual(key.lookup(30.0, "T3").state, "occupied")

    def test_reads_the_evidence_letters_in_the_bracket(self):
        key = parse_answer_key(TABLE)

        self.assertEqual(key.lookup(0.0, "T1").evidence_code, "t")
        self.assertEqual(key.lookup(30.0, "T1").evidence_code, "stc")
        self.assertEqual(key.lookup(30.0, "T3").evidence_code, "s")

    def test_an_empty_seat_can_still_carry_belongings(self):
        """`O (c)` — 바 자리에 짐은 있지만 쓸 수 있는 자리다."""
        answer = parse_answer_key(TABLE).lookup(30.0, "BAR7-6")

        self.assertEqual(answer.state, "empty")
        self.assertEqual(answer.evidence_code, "c")

    def test_seat_names_come_from_the_header_and_are_normalized(self):
        key = parse_answer_key(TABLE)

        self.assertIsNotNone(key.lookup(0.0, "BAR7-6"))

    def test_the_note_column_rides_along_with_every_seat_in_the_row(self):
        key = parse_answer_key(TABLE)

        self.assertEqual(key.lookup(30.0, "T1").note, "테이블 옮김")
        self.assertEqual(key.lookup(0.0, "T1").note, "")

    def test_a_blank_cell_is_an_error_not_a_pass(self):
        with self.assertRaises(ValueError) as caught:
            parse_answer_key(
                "| 시간(s) | T1 | T2 |\n| --- | --- | --- |\n| 0 | X | |\n"
            )

        self.assertIn("T2", str(caught.exception))

    def test_an_unreadable_cell_is_reported_not_guessed(self):
        with self.assertRaises(ValueError) as caught:
            parse_answer_key(
                "| 시간(s) | T1 | T2 |\n| --- | --- | --- |\n| 0 | 아마도 | O |\n"
            )

        self.assertIn("T1", str(caught.exception))

    def test_prose_after_the_table_is_ignored(self):
        key = parse_answer_key(TABLE + "\nX의 의미 : 사용중\n\n(t) : 테이블에 물건\n")

        self.assertEqual(len(key.answers), 8)

    def test_a_document_with_no_table_falls_back_to_the_line_format(self):
        self.assertEqual(parse_table(LINES).answers, [])
        self.assertEqual(len(parse_answer_key(LINES).answers), 6)


class LineTests(unittest.TestCase):
    def test_reads_seconds_from_the_heading(self):
        key = parse_answer_key(LINES)

        self.assertEqual(key.timestamps(), [0.0, 15.0])

    def test_reads_state_and_note(self):
        answer = parse_answer_key(LINES).lookup(0.0, "BAR7-4")

        self.assertEqual(answer.state, "unknown")
        self.assertEqual(answer.note, "사람이 완전히 가림")

    def test_state_without_a_note_leaves_the_note_empty(self):
        self.assertEqual(parse_answer_key(LINES).lookup(0.0, "T1").note, "")

    def test_seat_names_ignore_case_and_spacing(self):
        key = parse_answer_key(LINES)

        self.assertEqual(key.lookup(15.0, "BAR7-4").state, "occupied")
        self.assertEqual(key.lookup(15.0, "T2").state, "empty")

    def test_prose_lines_are_ignored(self):
        key = parse_answer_key("잡담\n1번 사진 (0초)\n- T1 : 사용중\n총평: 좋다\n")

        self.assertEqual(len(key.answers), 1)

    def test_photo_number_is_used_when_no_seconds_are_given(self):
        key = parse_answer_key("3번 사진\n- T1 : 사용중\n")

        self.assertEqual(key.answers[0].photo_index, 3)
        self.assertIsNone(key.answers[0].timestamp)

    def test_an_unreadable_state_is_reported_not_guessed(self):
        with self.assertRaises(ValueError) as caught:
            parse_answer_key("1번 사진 (0초)\n- T1 : 아마도 사용중인듯\n")

        self.assertIn("T1", str(caught.exception))

    def test_a_seat_line_before_any_heading_is_an_error(self):
        with self.assertRaises(ValueError):
            parse_answer_key("- T1 : 사용중\n")


class EvidenceTests(unittest.TestCase):
    def test_splits_on_commas_and_spaces(self):
        self.assertEqual(parse_evidence("t, c"), {"t", "c"})
        self.assertEqual(parse_evidence(" c"), {"c"})

    def test_nothing_in_the_bracket_is_no_evidence(self):
        self.assertEqual(parse_evidence(None), set())
        self.assertEqual(parse_evidence(""), set())

    def test_letters_outside_the_vocabulary_are_dropped(self):
        self.assertEqual(parse_evidence("t, z"), {"t"})

    def test_the_code_is_ordered_person_table_chair(self):
        answer = Answer(0.0, 1, "T1", "occupied", "", {"c", "t", "s"})

        self.assertEqual(answer.evidence_code, "stc")


class NormalizeTests(unittest.TestCase):
    def test_uppercases_and_strips(self):
        self.assertEqual(normalize_seat(" bar7 - 4 "), "BAR7-4")

    def test_leaves_a_clean_name_alone(self):
        self.assertEqual(normalize_seat("T1"), "T1")


class ValidateTests(unittest.TestCase):
    def key_with(self, *answers):
        return AnswerKey(list(answers))

    def test_missing_seat_is_reported(self):
        key = self.key_with(Answer(0.0, 1, "T1", "occupied", ""))

        problems = validate(key, seats=["T1", "T2"], timestamps=[0.0])

        self.assertEqual(len(problems), 1)
        self.assertIn("T2", problems[0])

    def test_unknown_seat_name_is_reported(self):
        key = self.key_with(
            Answer(0.0, 1, "T1", "occupied", ""),
            Answer(0.0, 1, "T9", "empty", ""),
        )

        problems = validate(key, seats=["T1"], timestamps=[0.0])

        self.assertTrue(any("T9" in problem for problem in problems))

    def test_duplicate_seat_in_one_photo_is_reported(self):
        key = self.key_with(
            Answer(0.0, 1, "T1", "occupied", ""),
            Answer(0.0, 1, "T1", "empty", ""),
        )

        problems = validate(key, seats=["T1"], timestamps=[0.0])

        self.assertTrue(any("T1" in problem for problem in problems))

    def test_missing_photo_is_reported(self):
        key = self.key_with(Answer(0.0, 1, "T1", "occupied", ""))

        problems = validate(key, seats=["T1"], timestamps=[0.0, 15.0])

        self.assertTrue(any("15" in problem for problem in problems))

    def test_a_complete_key_has_no_problems(self):
        key = self.key_with(
            Answer(0.0, 1, "T1", "occupied", ""),
            Answer(0.0, 1, "T2", "empty", ""),
        )

        self.assertEqual(validate(key, seats=["T1", "T2"], timestamps=[0.0]), [])

    def test_photo_numbers_are_resolved_to_the_run_timestamps(self):
        key = parse_answer_key("2번 사진\n- T1 : 사용중\n")

        key.resolve_photo_numbers([0.0, 15.0])

        self.assertEqual(key.lookup(15.0, "T1").state, "occupied")


if __name__ == "__main__":
    unittest.main()

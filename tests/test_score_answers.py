import unittest

from checks.answer_key import Answer, AnswerKey
from checks.score_answers import (
    Verdict,
    direction,
    grade_seat,
    is_occlusion_unknown,
    review_state,
    score_run,
    summarize,
)


def table(name="T1", state="occupied", raw_state=None, **kwargs):
    base = {
        "layout_name": name,
        "state": state,
        "raw_state": state if raw_state is None else raw_state,
        "reason": "",
        "objects": [],
        "chair_objects": [],
        "evidence_code": "",
        "seated_people": 0,
        "occupied_chairs": 0,
        "chair_seated_people": 0,
        "connected_chairs": [],
        "box": [0.0, 0.0, 10.0, 10.0],
    }
    base.update(kwargs)
    return base


def record(timestamp=0.0, tables=None):
    return {"timestamp": timestamp, "tables": tables or []}


class ReviewStateTests(unittest.TestCase):
    def test_an_answer_held_with_no_fresh_evidence_reads_as_unknown(self):
        self.assertEqual(
            review_state(table(state="occupied", raw_state="empty")), "unknown"
        )

    def test_an_ordinary_verdict_is_itself(self):
        self.assertEqual(review_state(table(state="empty")), "empty")


class OcclusionTests(unittest.TestCase):
    def test_a_seat_hidden_by_a_person_is_recognized(self):
        hidden = table(state="unknown", reason="occluded_by_person:0.62")

        self.assertTrue(is_occlusion_unknown(hidden))

    def test_an_unreadable_neighbour_counts_as_occlusion(self):
        hidden = table(
            state="unknown", reason="nearby_person_pose_unknown:insufficient_keypoints"
        )

        self.assertTrue(is_occlusion_unknown(hidden))

    def test_one_person_over_two_bar_slots_counts_as_occlusion(self):
        hidden = table(state="unknown", reason="spans_multiple_seats")

        self.assertTrue(is_occlusion_unknown(hidden))

    def test_waiting_for_confirmation_is_not_occlusion(self):
        pending = table(
            state="unknown", raw_state="occupied", reason="awaiting_confirmation:objects:book"
        )

        self.assertFalse(is_occlusion_unknown(pending))

    def test_a_decided_seat_is_never_occlusion_unknown(self):
        self.assertFalse(
            is_occlusion_unknown(table(state="empty", reason="occluded_by_person:0.9"))
        )


class GradeSeatTests(unittest.TestCase):
    """두 겹 채점 + 2026-09-02 공지의 면제 두 가지."""

    def test_both_right_is_a_clean_pass(self):
        verdict = grade_seat("occupied", table(state="occupied"))

        self.assertEqual(verdict.category, "맞음")
        self.assertTrue(verdict.app_correct)
        self.assertTrue(verdict.evidence_correct)

    def test_a_seat_hidden_by_a_person_is_a_correct_judgement(self):
        """공지: 사람이 가려서 모름이면 그게 맞는 판단이다."""
        verdict = grade_seat(
            "occupied", table(state="unknown", raw_state="unknown",
                              reason="occluded_by_person:0.71")
        )

        self.assertEqual(verdict.category, "가림모름")
        self.assertFalse(verdict.is_fixable)

    def test_evidence_right_but_still_confirming_is_delay_not_a_miss(self):
        """15초에 처음 본 짐 — 규칙대로 확정을 기다리는 중이다."""
        verdict = grade_seat(
            "occupied",
            table(state="unknown", raw_state="occupied",
                  reason="awaiting_confirmation:objects:book"),
        )

        self.assertEqual(verdict.category, "지연")
        self.assertFalse(verdict.is_fixable)

    def test_leaving_a_seat_is_also_a_delay_not_a_miss(self):
        """정답은 이미 비었는데 우리는 아직 놓아주지 않았다 — 전이 규칙이다."""
        verdict = grade_seat(
            "empty", table(state="occupied", raw_state="empty",
                           reason="no_customer_evidence")
        )

        self.assertEqual(verdict.category, "지연")
        self.assertFalse(verdict.is_fixable)

    def test_holding_the_right_answer_on_stale_evidence_is_flagged(self):
        """앱 답은 맞았지만 이번 판단에서 짐을 놓쳤다."""
        verdict = grade_seat(
            "occupied", table(state="occupied", raw_state="unknown",
                              reason="no_customer_evidence")
        )

        self.assertEqual(verdict.category, "유지")
        self.assertTrue(verdict.is_fixable)

    def test_both_wrong_is_the_real_miss(self):
        verdict = grade_seat("occupied", table(state="empty", raw_state="empty"))

        self.assertEqual(verdict.category, "오답")
        self.assertTrue(verdict.is_fixable)
        self.assertEqual(verdict.direction, "놓침")

    def test_a_seat_the_camera_cannot_see_is_not_scored(self):
        verdict = grade_seat("ignore", table(state="empty", raw_state="empty"))

        self.assertEqual(verdict.category, "화각밖")
        self.assertFalse(verdict.is_fixable)


class EvidenceScoringTests(unittest.TestCase):
    def test_a_correct_state_can_still_have_missed_evidence(self):
        """T6 정답 X(t,c) 인데 우리는 책상 짐만 봤다."""
        verdict = grade_seat(
            "occupied", table(state="occupied", evidence_code="t"), truth_evidence="tc"
        )

        self.assertEqual(verdict.category, "맞음")
        self.assertEqual(verdict.missed_evidence, "c")
        self.assertEqual(verdict.imagined_evidence, "")

    def test_evidence_we_invented_is_reported_too(self):
        verdict = grade_seat(
            "occupied", table(state="occupied", evidence_code="ts"), truth_evidence="t"
        )

        self.assertEqual(verdict.imagined_evidence, "s")

    def test_a_bar_seat_may_hold_belongings_and_still_be_free(self):
        """`O (c)` — 짐은 있지만 쓸 수 있는 자리다."""
        verdict = grade_seat(
            "empty",
            table(state="empty", reason="belongings_only", evidence_code=""),
            truth_evidence="c",
        )

        self.assertEqual(verdict.category, "맞음")
        self.assertEqual(verdict.missed_evidence, "c")

    def test_a_seat_out_of_frame_has_no_evidence_verdict(self):
        verdict = grade_seat("ignore", table(), truth_evidence="t")

        self.assertEqual(verdict.missed_evidence, "")

    def test_evidence_falls_back_to_the_log_reason_on_old_runs(self):
        old = table(state="occupied", objects=[{"class": "book"}])
        old.pop("evidence_code")

        verdict = grade_seat("occupied", old, truth_evidence="t")

        self.assertEqual(verdict.our_evidence, "t")


class DirectionTests(unittest.TestCase):
    def test_calling_a_taken_seat_free_is_the_worst_direction(self):
        self.assertEqual(direction("occupied", "empty"), "놓침")

    def test_calling_a_free_seat_taken(self):
        self.assertEqual(direction("empty", "occupied"), "헛것")

    def test_unknown_where_the_truth_is_certain(self):
        self.assertEqual(direction("occupied", "unknown"), "과잉모름")

    def test_certain_where_the_truth_is_unknown(self):
        self.assertEqual(direction("unknown", "occupied"), "과소모름")

    def test_agreement_has_no_direction(self):
        self.assertIsNone(direction("empty", "empty"))


class ScoreRunTests(unittest.TestCase):
    def key(self, *rows):
        return AnswerKey(
            [
                Answer(stamp, 1, seat, state, "", set(evidence))
                for stamp, seat, state, evidence in rows
            ]
        )

    def test_joins_the_key_to_the_run_by_timestamp_and_seat(self):
        records = [record(0.0, [table("T1", "occupied"), table("T2", "empty")])]
        key = self.key((0.0, "T1", "occupied", ""), (0.0, "T2", "occupied", ""))

        verdicts = score_run(records, key)

        self.assertEqual([v.category for v in verdicts], ["맞음", "오답"])
        self.assertEqual(verdicts[1].direction, "놓침")

    def test_a_seat_in_the_run_with_no_answer_is_reported_not_skipped(self):
        verdicts = score_run([record(0.0, [table("T1", "occupied")])], AnswerKey([]))

        self.assertEqual(verdicts[0].category, "정답없음")

    def test_carries_the_evidence_a_diagnosis_needs(self):
        objects = [{"class": "book", "confidence": 0.18, "share": 0.7}]
        records = [record(0.0, [table("T1", "occupied", objects=objects)])]

        verdict = score_run(records, self.key((0.0, "T1", "empty", "")))[0]

        self.assertEqual(verdict.seat_box, [0.0, 0.0, 10.0, 10.0])
        self.assertEqual(verdict.objects, objects)

    def test_seat_names_are_normalized_to_the_key(self):
        records = [record(0.0, [table("Bar7-6", "empty")])]

        verdict = score_run(records, self.key((0.0, "BAR7-6", "empty", "c")))[0]

        self.assertEqual(verdict.seat, "BAR7-6")
        self.assertEqual(verdict.missed_evidence, "c")

    def test_ticks_are_scored_in_time_order(self):
        records = [record(15.0, [table("T1", "empty")]), record(0.0, [table("T1", "empty")])]
        key = self.key((0.0, "T1", "empty", ""), (15.0, "T1", "empty", ""))

        self.assertEqual([v.timestamp for v in score_run(records, key)], [0.0, 15.0])


class SummarizeTests(unittest.TestCase):
    def verdict(self, category, direction=None, missed=""):
        return Verdict(
            timestamp=0.0,
            seat="T1",
            truth="occupied",
            app_state="empty",
            evidence_state="empty",
            category=category,
            direction=direction,
            reason="",
            truth_evidence=missed,
            our_evidence="",
        )

    def test_counts_cells_and_the_fixable_share(self):
        summary = summarize(
            [
                self.verdict("맞음"),
                self.verdict("지연"),
                self.verdict("오답", "놓침"),
                self.verdict("유지", "놓침"),
            ]
        )

        self.assertEqual(summary["total"], 4)
        self.assertEqual(summary["맞음"], 1)
        self.assertEqual(summary["fixable"], 2)

    def test_rules_that_worked_as_designed_count_as_accepted(self):
        summary = summarize(
            [self.verdict("맞음"), self.verdict("지연"), self.verdict("가림모름")]
        )

        self.assertEqual(summary["accepted"], 3)
        self.assertEqual(summary["fixable"], 0)

    def test_directions_only_count_cells_we_would_fix(self):
        summary = summarize([self.verdict("오답", "놓침"), self.verdict("지연", "과잉모름")])

        self.assertEqual(summary["directions"], {"놓침": 1})

    def test_counts_evidence_gaps_separately_from_state_errors(self):
        summary = summarize([self.verdict("맞음", missed="c")])

        self.assertEqual(summary["fixable"], 0)
        self.assertEqual(summary["evidence_missed"], 1)

    def test_seats_out_of_frame_leave_the_denominator(self):
        summary = summarize([self.verdict("맞음"), self.verdict("화각밖")])

        self.assertEqual(summary["scored"], 1)
        self.assertEqual(summary["total"], 2)


if __name__ == "__main__":
    unittest.main()

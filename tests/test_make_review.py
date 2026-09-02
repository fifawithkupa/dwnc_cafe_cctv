import unittest

from checks.judge_frames import Judgement
from checks.make_review import Tick, explain_seat, seat_caption


def table(**kwargs):
    base = {
        "layout_name": "T5",
        "state": "occupied",
        "raw_state": "occupied",
        "reason": "",
        "objects": [],
        "seated_people": 0,
        "occupied_chairs": 0,
        "chair_seated_people": 0,
    }
    base.update(kwargs)
    return base


def tick(index=1, timestamp=0.0, record=None, judgement=None) -> Tick:
    return Tick(
        index=index,
        stem="t0000.0s",
        timestamp=timestamp,
        record=record or {},
        judgement=judgement,
    )


class SeatCaptionTests(unittest.TestCase):
    def test_occupied_seat_says_why(self):
        caption = seat_caption(
            table(objects=[{"class": "laptop", "confidence": 0.4}])
        )

        self.assertEqual(caption, "X T5 (t)")

    def test_empty_seat_carries_no_letters(self):
        """빈자리 옆 괄호는 "그럼 왜 비었나"라는 되묻기만 만든다."""
        caption = seat_caption(
            table(state="empty", raw_state="empty", reason="no_customer_evidence")
        )

        self.assertEqual(caption, "O T5")


class ExplainSeatTests(unittest.TestCase):
    def test_evidence_swallowed_by_the_confirmation_rule_is_shown(self):
        """지금 가장 자주 나오는 오답의 모양이라 표에서 눈에 띄어야 한다.

        근거를 봤는데도 연속 2번이 안 돼 빈자리로 발표된 경우, reason 은
        사용중 쪽 근거를 담고 있어서 빈자리 사유로 읽으면 거짓말이 된다.
        """
        text = explain_seat(
            table(
                state="empty",
                raw_state="occupied",
                reason="objects:keyboard;occupied_chairs:1",
                objects=[{"class": "keyboard", "confidence": 0.3}],
                occupied_chairs=1,
                chair_seated_people=1,
            )
        )

        self.assertIn("근거가 보였다", text)
        self.assertIn("사람이 앉음", text)
        self.assertIn("책상에 짐", text)
        self.assertNotIn("objects:keyboard", text)

    def test_bar_seat_belongings_are_named_as_a_deliberate_choice(self):
        text = explain_seat(
            table(
                layout_name="BAR7-6",
                state="empty",
                raw_state="empty",
                reason="belongings_only",
            )
        )

        self.assertEqual(text, "짐만 있음 (바 자리라 안 셈)")

    def test_unknown_reason_code_is_shown_verbatim(self):
        """어휘에 없는 사유를 "알 수 없음"으로 뭉개면 새 사유가 묻힌다."""
        text = explain_seat(
            table(state="empty", raw_state="empty", reason="brand_new_code")
        )

        self.assertEqual(text, "brand_new_code")

    def test_occupied_without_visible_evidence_says_the_rule_held_it(self):
        text = explain_seat(table(reason="no_customer_evidence", raw_state="empty"))

        self.assertIn("사용중 유지", text)


class FilenameTests(unittest.TestCase):
    def test_detector_count_drives_the_star(self):
        """자세 단계에서 조용히 버려진 헛것도 ★ 로 드러나야 한다."""
        record = {
            "timestamp": 30.0,
            "summary": {
                "occupied": 3,
                "empty": 9,
                "unknown": 0,
                "seated_poses": 1,
                "standing_poses": 1,
                "unknown_poses": 0,
            },
            "raw_detections": {"counts": {"person": 4}},
        }
        judgement = Judgement(stem="t0030.0s", people_total=2)

        stem = tick(index=5, timestamp=30.0, record=record, judgement=judgement).filename_stem

        self.assertEqual(stem, "05_t0030s_점유3빈9모름0_사람4대2_차이+2_★")

    def test_unscorable_truth_is_not_faked_into_a_zero_gap(self):
        record = {
            "timestamp": 0.0,
            "summary": {"occupied": 1, "empty": 2, "unknown": 0},
            "raw_detections": {"counts": {"person": 2}},
        }
        judgement = Judgement(stem="t0000.0s", people_total=2, uncertain_people=True)

        stem = tick(record=record, judgement=judgement).filename_stem

        self.assertEqual(stem, "01_t0000s_점유1빈2모름0_사람2대?_차이?_－")


if __name__ == "__main__":
    unittest.main()

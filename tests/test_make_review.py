import unittest

from checks.judge_frames import Judgement
from checks.make_review import Tick, explain_seat, review_state, seat_caption


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

    def test_first_sighting_seat_shows_what_was_seen(self):
        """방금 앉은 것인지 헛것인지를 사람이 판단할 수 있어야 한다."""
        caption = seat_caption(
            table(
                state="unknown",
                raw_state="occupied",
                objects=[{"class": "laptop", "confidence": 0.4}],
            )
        )

        self.assertEqual(caption, "? T5 (t)")

    def test_empty_seat_carries_no_letters(self):
        """빈자리 옆 괄호는 "그럼 왜 비었나"라는 되묻기만 만든다."""
        caption = seat_caption(
            table(state="empty", raw_state="empty", reason="no_customer_evidence")
        )

        self.assertEqual(caption, "O T5")


class ExplainSeatTests(unittest.TestCase):
    def test_first_sighting_of_evidence_is_named_as_such(self):
        """점유 근거를 처음 본 판단.  reason 은 사용중 쪽 근거를 담고 있어서
        빈자리 사유로 읽으면 거짓말이 된다."""
        text = explain_seat(
            table(
                state="unknown",
                raw_state="occupied",
                reason="awaiting_confirmation:objects:keyboard;occupied_chairs:1",
                objects=[{"class": "keyboard", "confidence": 0.3}],
                occupied_chairs=1,
                chair_seated_people=1,
            )
        )

        self.assertIn("처음 봤다", text)
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
        """검수에서는 회색으로 구분한다 — 손님 앱에는 사용중으로 나간다."""
        held = table(reason="no_customer_evidence", raw_state="empty")

        self.assertEqual(review_state(held), "unknown")
        self.assertEqual(seat_caption(held), "? T5")
        self.assertIn("아무 근거도 못 봤다", explain_seat(held))
        self.assertIn("손님 앱에는 사용중", explain_seat(held))


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

        # 전각 물음표 — 윈도우 파일 이름에 ASCII "?" 를 못 쓴다.
        self.assertEqual(stem, "01_t0000s_점유1빈2모름0_사람2대？_차이？_－")


if __name__ == "__main__":
    unittest.main()

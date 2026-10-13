"""install/check_install.py — 설치 당일 "제대로 연결됐나" 3종 확인.

판정 부분은 전부 순수 함수다.  카메라도 박스도 서버도 없이 시험할 수 있어야
현장에서 이 도구 자체를 믿을 수 있다.
"""

import unittest

from install.check_install import (
    CAMERA_MIN_HEIGHT,
    seats_verdict,
    MAX_PUBLISH_AGE_S,
    TICK_BUDGET_S,
    camera_verdict,
    judging_verdict,
    parse_tick_lines,
    publish_verdict,
    overall_verdict,
)


class CameraTest(unittest.TestCase):
    def test_good_stream_passes(self):
        ok, lines = camera_verdict({"width": 2560, "height": 1440, "codec": "hevc", "fps": 20.0})
        self.assertTrue(ok)
        self.assertTrue(any("2560x1440" in line for line in lines))

    def test_no_stream_fails_with_what_to_do(self):
        ok, lines = camera_verdict(None)
        self.assertFalse(ok)
        # 현장에서 읽고 바로 손을 움직일 수 있어야 한다.
        self.assertTrue(any("인젝터" in line for line in lines))

    def test_small_picture_fails(self):
        """해상도가 낮으면 먼 자리의 컵·가방이 안 잡힌다."""
        ok, lines = camera_verdict({"width": 640, "height": 480, "codec": "h264", "fps": 15.0})
        self.assertFalse(ok)
        self.assertTrue(any(str(CAMERA_MIN_HEIGHT) in line for line in lines))

    def test_zero_fps_is_not_a_crash(self):
        ok, _ = camera_verdict({"width": 2560, "height": 1440, "codec": "hevc", "fps": 0.0})
        self.assertFalse(ok)


class TickParsingTest(unittest.TestCase):
    LOG = [
        "하드웨어 디코딩 켜짐 (vaapi)",
        "Supabase 전송: 켜짐 (moonq)",
        "[  5.6s] tables=12 occupied=0 empty=12 ignore=0 poses(seated/standing/unknown)=0/0/0 inference=5745ms",
        "[ 20.6s] tables=12 occupied=3 empty=9 ignore=0 poses(seated/standing/unknown)=3/0/0 inference=3299ms",
        "[ 35.6s] tables=12 occupied=3 empty=9 ignore=0 poses(seated/standing/unknown)=3/0/0 inference=3338ms",
    ]
    #: 좌석 네모가 지금 화면과 안 맞을 때 실제로 나온 로그 (2026-10-13 집에서).
    MISMATCHED = [
        "[1600.5s] tables=12 occupied=0 empty=0 ignore=12 poses(seated/standing/unknown)=0/0/1 inference=3362ms",
        "[1615.5s] tables=12 occupied=0 empty=0 ignore=12 poses(seated/standing/unknown)=0/0/1 inference=3356ms",
    ]

    def test_reads_every_tick(self):
        ticks = parse_tick_lines(self.LOG)
        self.assertEqual([t.tables for t in ticks], [12, 12, 12])
        self.assertEqual([t.inference_ms for t in ticks], [5745, 3299, 3338])

    def test_ignores_lines_that_are_not_ticks(self):
        self.assertEqual(parse_tick_lines(["그냥 글", ""]), [])

    def test_first_tick_is_allowed_to_be_slow(self):
        """모델을 처음 읽는 판정 하나는 느리다 — 그걸로 불합격을 내지 않는다."""
        ok, _ = judging_verdict(parse_tick_lines(self.LOG), hwaccel="vaapi")
        self.assertTrue(ok)

    def test_shows_usual_time_not_just_the_worst(self):
        """가장 느린 값만 보여주면 재시작 직후 숫자를 보고 느려졌다고 오해한다."""
        ok, lines = judging_verdict(parse_tick_lines(self.LOG), hwaccel="vaapi")
        text = " ".join(lines)
        self.assertIn("보통", text)
        self.assertIn("3.3", text)  # 3299ms·3338ms 가 보통값

    def test_slow_ticks_fail(self):
        log = list(self.LOG) + [
            "[ 50.6s] tables=12 occupied=0 empty=12 ignore=0 poses(seated/standing/unknown)=0/0/0 inference=9000ms",
            "[ 65.6s] tables=12 occupied=0 empty=12 ignore=0 poses(seated/standing/unknown)=0/0/0 inference=9100ms",
        ]
        ok, lines = judging_verdict(parse_tick_lines(log), hwaccel="vaapi")
        self.assertFalse(ok)
        self.assertTrue(any(str(TICK_BUDGET_S) in line for line in lines))

    def test_all_seats_ignored_fails(self):
        """좌석 네모가 지금 화면과 안 맞으면 전부 ignore 가 되고 앱에는 한 자리도 안 나간다."""
        ok, lines = seats_verdict(parse_tick_lines(self.MISMATCHED))
        self.assertFalse(ok)
        text = " ".join(lines)
        self.assertIn("좌석", text)
        self.assertIn("12", text)

    def test_normal_seats_pass(self):
        ok, _ = seats_verdict(parse_tick_lines(self.LOG))
        self.assertTrue(ok)

    def test_no_ticks_fails(self):
        ok, lines = judging_verdict([], hwaccel="vaapi")
        self.assertFalse(ok)
        self.assertTrue(any("판정" in line for line in lines))

    def test_software_decoding_warns_but_passes(self):
        """소프트웨어로 떨어져도 4MP 는 예산 안이다 — 멈출 일은 아니고 적어둘 일이다."""
        ok, lines = judging_verdict(parse_tick_lines(self.LOG), hwaccel=None)
        self.assertTrue(ok)
        self.assertTrue(any("하드웨어 디코딩" in line for line in lines))


class PublishTest(unittest.TestCase):
    ROW = {
        "cafe_id": "moonq",
        "status": "live",
        "total_tables": 12,
        "busy_tables": 3,
        "free_tables": 9,
        "box_version": "21f307c",
    }

    def test_fresh_row_passes(self):
        ok, lines = publish_verdict(self.ROW, age_seconds=7.0, expected_cafe_id="moonq")
        self.assertTrue(ok)
        self.assertTrue(any("moonq" in line for line in lines))

    def test_stale_row_fails(self):
        ok, lines = publish_verdict(self.ROW, age_seconds=120.0, expected_cafe_id="moonq")
        self.assertFalse(ok)
        self.assertTrue(any(str(MAX_PUBLISH_AGE_S) in line for line in lines))

    def test_missing_row_fails(self):
        ok, lines = publish_verdict(None, age_seconds=None, expected_cafe_id="moonq")
        self.assertFalse(ok)

    def test_wrong_cafe_is_caught(self):
        """시험용 카페로 쏘고 있으면 앱에는 아무것도 안 뜬다 — 제일 놓치기 쉬운 실수다."""
        ok, lines = publish_verdict(self.ROW, age_seconds=5.0, expected_cafe_id="dh-012")
        self.assertFalse(ok)
        self.assertTrue(any("dh-012" in line for line in lines))

    def test_zero_seats_fails(self):
        """자리 0개를 통과로 내면, 좌석을 그린 뒤 뭔가 틀어져도 합격으로 보인다."""
        row = dict(self.ROW, total_tables=0, busy_tables=0, free_tables=0)
        ok, lines = publish_verdict(row, age_seconds=5.0, expected_cafe_id="moonq")
        self.assertFalse(ok)
        self.assertTrue(any("자리" in line for line in lines))

    def test_gap_fails(self):
        row = dict(self.ROW, status="gap")
        ok, lines = publish_verdict(row, age_seconds=5.0, expected_cafe_id="moonq")
        self.assertFalse(ok)
        self.assertTrue(any("카메라" in line for line in lines))


class OverallTest(unittest.TestCase):
    def test_all_three_must_pass(self):
        self.assertTrue(overall_verdict([True, True, True]))
        self.assertFalse(overall_verdict([True, False, True]))


if __name__ == "__main__":
    unittest.main()

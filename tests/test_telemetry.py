"""edge/telemetry.py — 로그 한 줄이 표의 줄로 제대로 바뀌는지.

이 모듈은 순수 파이썬이라 카메라도 모델도 없이 돈다.
"""

import random
import unittest

from edge import telemetry as T


def _table(**overrides):
    table = {
        "layout_name": "T1",
        "layout_kind": "table",
        "layout_zone_name": "창가",
        "layout_capacity": 4,
        "box": [100.0, 100.0, 200.0, 200.0],
        "state": "occupied",
        "raw_state": "occupied",
        "persistent_state": "occupied",
        "shown_state": "occupied",
        "reason": "",
        "confidence": 0.8,
        "seated_people": 1,
        "connected_chairs": [{"class": "chair", "box": [90, 190, 130, 230]}],
        "objects": [],
    }
    table.update(overrides)
    return table


def _record(tables=None, poses=None, wall_clock="2026-09-11T14:30:00+0900"):
    return {
        "wall_clock": wall_clock,
        "tables": tables if tables is not None else [_table()],
        "poses": poses or [],
        "run": {"input": {"width": 1920, "height": 1080}},
    }


class OpenWindowTest(unittest.TestCase):
    def test_parses_and_judges(self):
        window = T.parse_open_window("09:00-22:00")
        self.assertEqual(window, (540, 1320))
        self.assertTrue(T.is_open_at("2026-09-11T14:30:00+0900", window))
        self.assertFalse(T.is_open_at("2026-09-11T03:00:00+0900", window))

    def test_overnight_window(self):
        window = T.parse_open_window("18:00-02:00")
        self.assertTrue(T.is_open_at("2026-09-11T23:00:00+0900", window))
        self.assertTrue(T.is_open_at("2026-09-11T01:00:00+0900", window))
        self.assertFalse(T.is_open_at("2026-09-11T12:00:00+0900", window))

    def test_unknown_stays_none_never_false(self):
        """영업시간을 모르면 '모른다'다.  '닫았다'로 반올림하지 않는다."""
        self.assertIsNone(T.is_open_at("2026-09-11T14:30:00+0900", None))
        self.assertIsNone(T.is_open_at(None, (540, 1320)))
        self.assertIsNone(T.parse_open_window("아무거나"))
        self.assertIsNone(T.parse_open_window("25:00-26:00"))


class GeometryTest(unittest.TestCase):
    def test_seat_relative_is_a_ratio_not_a_room_coordinate(self):
        geom = T.seat_relative([100, 100, 150, 200], [100, 100, 200, 200])
        self.assertEqual(geom, {"cx": 0.25, "cy": 0.5, "w": 0.5, "h": 1.0})
        # 같은 모양이 방 어디에 있든 같은 값이 나온다 = 위치가 복원되지 않는다
        moved = T.seat_relative([900, 700, 950, 800], [900, 700, 1000, 800])
        self.assertEqual(geom, moved)

    def test_overlap_fraction(self):
        self.assertEqual(T.overlap_fraction([100, 100, 200, 200], [100, 100, 200, 200]), 1.0)
        self.assertAlmostEqual(
            T.overlap_fraction([150, 150, 250, 250], [100, 100, 200, 200]), 0.25
        )
        self.assertEqual(T.overlap_fraction([500, 500, 600, 600], [100, 100, 200, 200]), 0.0)

    def test_border_margin_flags_the_frame_edge(self):
        middle = T.border_margin([900, 500, 1000, 600], 1920, 1080)
        edge = T.border_margin([0, 500, 100, 600], 1920, 1080)
        self.assertGreater(middle, 0.2)
        self.assertEqual(edge, 0.0)
        self.assertIsNone(T.border_margin([0, 0, 10, 10], None, None))

    def test_aspect_ratio_is_the_1_75_rule_input(self):
        self.assertEqual(T.aspect_ratio([0, 0, 100, 175]), 1.75)


class SelectionTest(unittest.TestCase):
    """§10-3 — 무엇을 남기고 무엇을 버리나."""

    def _selector(self, control_rate=0.0):
        return T.TelemetrySelector(
            "moonq", "run-1", open_window=(540, 1320),
            control_rate=control_rate, rng=random.Random(0),
        )

    def test_boring_tick_is_dropped(self):
        self.assertEqual(self._selector().observe(_record()), [])

    def test_unknown_is_kept(self):
        rows = self._selector().observe(
            _record([_table(state="unknown", reason="compact_occluded_pose=1.42")])
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_kind"], "hard")
        self.assertEqual(rows[0]["reason_code"], "compact_occluded_pose")

    def test_raw_and_settled_disagreement_is_kept(self):
        rows = self._selector().observe(
            _record([_table(raw_state="empty", persistent_state="occupied")])
        )
        self.assertEqual(rows[0]["sample_kind"], "hard")

    def test_split_burst_vote_is_kept(self):
        """엔진이 이미 남기는 vote_counts 에서 갈림을 읽어낸다."""
        rows = self._selector().observe(
            _record([_table(vote_counts={"occupied": 3, "empty": 2})])
        )
        self.assertEqual(rows[0]["sample_kind"], "hard")
        self.assertEqual((rows[0]["votes_seen"], rows[0]["votes_total"]), (3, 5))

    def test_unanimous_burst_vote_is_dropped(self):
        rows = self._selector().observe(
            _record([_table(vote_counts={"occupied": 5})])
        )
        self.assertEqual(rows, [])

    def test_missing_vote_counts_stays_none(self):
        rows = self._selector().observe(_record([_table(state="unknown")]))
        self.assertEqual((rows[0]["votes_seen"], rows[0]["votes_total"]), (None, None))

    def test_discarded_object_is_kept(self):
        rows = self._selector().observe(
            _record([_table(objects=[
                {"class": "laptop", "confidence": 0.4, "box": [100, 100, 195, 195],
                 "share": 0.9, "kept": False, "drop_reason": "area_over_half"},
            ])])
        )
        self.assertEqual(rows[0]["sample_kind"], "hard")
        self.assertEqual(rows[0]["objects"][0]["drop_reason"], "area_over_half")

    def test_transition_is_always_kept(self):
        selector = self._selector()
        selector.observe(_record([_table(shown_state="occupied")]))
        rows = selector.observe(_record([_table(shown_state="empty")]))
        self.assertEqual(rows[0]["sample_kind"], "transition")

    def test_control_sample_appears(self):
        selector = self._selector(control_rate=1.0)
        rows = selector.observe(_record())
        self.assertEqual(rows[0]["sample_kind"], "control")


class PayloadTest(unittest.TestCase):
    def test_no_frame_coordinates_leave_the_box(self):
        """나가는 줄 어디에도 화면 좌표가 없어야 한다 (§10-4)."""
        pose = {"box": [110, 110, 160, 195], "state": "seated",
                "reason": "left_hka=95.0<110", "confidence": 0.9,
                "angles": {"left_hka": 95.0}}
        rows = T.TelemetrySelector("moonq", "run-1", rng=random.Random(0)).observe(
            _record([_table(state="unknown", objects=[
                {"class": "handbag", "confidence": 0.5,
                 "box": [120, 120, 150, 150], "share": 0.3},
            ])], poses=[pose])
        )
        flat = repr(rows[0])
        for pixel in ("1920", "1080", "110", "195", "120", "150"):
            self.assertNotIn(pixel, flat, f"화면 좌표 {pixel} 가 새어 나갔다")
        self.assertEqual(rows[0]["persons"][0]["hw_ratio"], 1.7)
        # 사유는 코드로 줄어 나가고, 값은 감사된 angles 안에만 있다
        self.assertEqual(rows[0]["persons"][0]["pose_reason"], "left_hka")
        self.assertEqual(rows[0]["persons"][0]["angles"], {"left_hka": 95.0})
        self.assertTrue(rows[0]["objects"][0]["kept"])

    def test_person_must_touch_the_seat(self):
        far = {"box": [900, 900, 950, 990], "state": "seated", "angles": {}}
        rows = T.TelemetrySelector("moonq", "run-1", rng=random.Random(0)).observe(
            _record([_table(state="unknown")], poses=[far])
        )
        self.assertEqual(rows[0]["persons"], [])


class DailyTest(unittest.TestCase):
    def test_summary_counts_every_tick_not_just_the_kept_ones(self):
        """여기가 틀리면 모름 비율이 통째로 거짓말이 된다."""
        selector = T.TelemetrySelector("moonq", "run-1", control_rate=0.0,
                                       rng=random.Random(0))
        for _ in range(8):
            selector.observe(_record())                       # 안 보냄
        for _ in range(2):
            selector.observe(_record([_table(state="unknown",
                                             reason="insufficient_keypoints")]))
        daily = selector.daily_rows()
        self.assertEqual(len(daily), 1)
        self.assertEqual(daily[0]["ticks"], 10)
        self.assertEqual(daily[0]["occupied"], 8)
        self.assertEqual(daily[0]["unknown"], 2)
        self.assertEqual(daily[0]["unknown_rate"], 0.2)
        self.assertEqual(daily[0]["reason_counts"], {"insufficient_keypoints": 2})

    def test_ignore_rate_is_tracked_separately(self):
        selector = T.TelemetrySelector("moonq", "run-1", rng=random.Random(0))
        selector.observe(_record([_table(state="ignore", reason="border_cropped")]))
        self.assertEqual(selector.daily_rows()[0]["ignore_rate"], 1.0)


class RunRowTest(unittest.TestCase):
    def test_fingerprint_ignores_volatile_fields(self):
        a = {"profile": "accuracy_default", "input": {"path": "a.mov", "sha256": "x"}}
        b = {"profile": "accuracy_default", "input": {"path": "b.mov", "sha256": "y"}}
        c = {"profile": "fast", "input": {"path": "a.mov", "sha256": "x"}}
        self.assertEqual(T.settings_fingerprint(a), T.settings_fingerprint(b))
        self.assertNotEqual(T.settings_fingerprint(a), T.settings_fingerprint(c))

    def test_run_row_shape(self):
        row = T.run_row(
            {"profile": "accuracy_default",
             "models": {"detector": "yolov8n", "detector_sha256": "abc"},
             "settings": {"imgsz": 1280, "median_frames": 2},
             "input": {"width": 1920, "height": 1080}},
            run_id="r1", cafe_id="moonq", started_at="2026-09-11T09:00:00+09:00",
        )
        self.assertEqual(row["imgsz"], 1280)
        self.assertEqual(row["frame_width"], 1920)
        self.assertEqual(len(row["settings_hash"]), 16)


if __name__ == "__main__":
    unittest.main()

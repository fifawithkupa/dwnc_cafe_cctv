"""Unit tests for LayoutZoneTracker (real-time zone drift in layout mode).

존은 **가구가 움직인 만큼** 따라간다.  탐지 박스가 그 자리에 그대로 있으면
존도 그대로 있어야 한다 — 사람이 그린 상자는 사람이 준 답이고, 탐지기의
상자와 애초에 같은 것이 아니기 때문이다 (CLAUDE.md).
"""

from __future__ import annotations

import unittest

from engine.seatnow_core import Detection, LayoutZoneTracker


def det(box, confidence=0.9, name="dining table"):
    return Detection(name=name, box=box, confidence=confidence)


class AnchorTests(unittest.TestCase):
    """첫 매칭은 기준점을 잡을 뿐, 존을 옮기지 않는다."""

    def test_first_sighting_does_not_move_the_drawn_box(self):
        """탐지 박스가 그린 상자와 달라도, 그건 가구가 움직인 게 아니다."""
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [], alpha=1.0)

        tracker.update([det((20.0, 10.0, 120.0, 110.0))], [])

        self.assertEqual(tracker.table_boxes[0], (0.0, 0.0, 100.0, 100.0))

    def test_a_still_table_never_drags_the_zone(self):
        """같은 자리에서 계속 보여도 존은 제자리다."""
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [], alpha=0.35)

        for _ in range(10):
            tracker.update([det((20.0, 10.0, 120.0, 110.0))], [])

        self.assertEqual(tracker.table_boxes[0], (0.0, 0.0, 100.0, 100.0))


class MovementTests(unittest.TestCase):
    """가구가 실제로 움직이면 그만큼 따라간다 — 그게 이 기능의 목적이다."""

    def test_zone_follows_how_far_the_furniture_moved(self):
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [], alpha=1.0)
        tracker.update([det((20.0, 0.0, 120.0, 100.0))], [])  # 기준점

        tracker.update([det((50.0, 0.0, 150.0, 100.0))], [])  # 30 만큼 이동

        self.assertEqual(tracker.table_boxes[0], (30.0, 0.0, 130.0, 100.0))

    def test_movement_is_smoothed_by_alpha(self):
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [], alpha=0.5)
        tracker.update([det((0.0, 0.0, 100.0, 100.0))], [])  # 기준점

        tracker.update([det((20.0, 0.0, 120.0, 100.0))], [])

        self.assertEqual(tracker.table_boxes[0], (10.0, 0.0, 110.0, 100.0))

    def test_furniture_put_back_returns_the_zone_to_where_it_was_drawn(self):
        """기준점이 설치 때 그대로라 되돌아온다 — 흘러가서 눌러앉지 않는다."""
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [], alpha=1.0)
        tracker.update([det((0.0, 0.0, 100.0, 100.0))], [])
        tracker.update([det((40.0, 0.0, 140.0, 100.0))], [])

        tracker.update([det((0.0, 0.0, 100.0, 100.0))], [])

        self.assertEqual(tracker.table_boxes[0], (0.0, 0.0, 100.0, 100.0))

    def test_a_resized_detection_resizes_the_zone_by_the_same_amount(self):
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [], alpha=1.0)
        tracker.update([det((0.0, 0.0, 100.0, 100.0))], [])

        tracker.update([det((0.0, 0.0, 120.0, 100.0))], [])

        self.assertEqual(tracker.table_boxes[0], (0.0, 0.0, 120.0, 100.0))

    def test_drift_does_not_compound_over_repeated_ticks(self):
        """예전 버그: 매 판단마다 직전 상자에서 또 35%씩 끌려갔다."""
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [], alpha=0.5)
        tracker.update([det((0.0, 0.0, 100.0, 100.0))], [])

        for _ in range(20):
            tracker.update([det((10.0, 0.0, 110.0, 100.0))], [])

        # 이동량 10 에 수렴할 뿐, 그 너머로 계속 흘러가지 않는다.
        self.assertAlmostEqual(tracker.table_boxes[0][0], 10.0, places=3)


class MatchingTests(unittest.TestCase):
    def test_zone_stays_without_detection(self):
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [])

        tracker.update([], [])

        self.assertEqual(tracker.table_boxes[0], (0.0, 0.0, 100.0, 100.0))

    def test_a_moved_zone_holds_its_place_when_the_detection_disappears(self):
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [], alpha=1.0)
        tracker.update([det((0.0, 0.0, 100.0, 100.0))], [])
        tracker.update([det((30.0, 0.0, 130.0, 100.0))], [])

        tracker.update([], [])

        self.assertEqual(tracker.table_boxes[0], (30.0, 0.0, 130.0, 100.0))

    def test_low_iou_detection_is_ignored(self):
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [])

        # IoU ≈ 0.05 → 매칭 거부, 존 정지
        tracker.update([det((90.0, 90.0, 190.0, 190.0))], [])

        self.assertEqual(tracker.table_boxes[0], (0.0, 0.0, 100.0, 100.0))

    def test_ambiguous_detection_moves_no_zone(self):
        zones = [(0.0, 0.0, 100.0, 100.0), (60.0, 0.0, 160.0, 100.0)]
        tracker = LayoutZoneTracker(zones, [])

        # 두 존과 동일한 IoU → 어느 쪽도 움직이지 않는다
        tracker.update([det((30.0, 0.0, 130.0, 100.0))], [])

        self.assertEqual(tracker.table_boxes[0], zones[0])
        self.assertEqual(tracker.table_boxes[1], zones[1])

    def test_each_zone_takes_only_best_detection(self):
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [], alpha=1.0)
        strong = det((10.0, 0.0, 110.0, 100.0))   # IoU ≈ 0.82
        weak = det((40.0, 0.0, 140.0, 100.0))     # IoU ≈ 0.43
        tracker.update([strong, weak], [])        # 기준점은 강한 쪽

        tracker.update([det((20.0, 0.0, 120.0, 100.0)), weak], [])

        self.assertEqual(tracker.table_boxes[0], (10.0, 0.0, 110.0, 100.0))

    def test_one_detection_moves_only_best_zone(self):
        zones = [(0.0, 0.0, 100.0, 100.0), (200.0, 0.0, 300.0, 100.0)]
        tracker = LayoutZoneTracker(zones, [], alpha=1.0)
        tracker.update([det((0.0, 0.0, 100.0, 100.0))], [])

        # 존 0과만 유의미하게 겹침 → 존 1은 정지
        tracker.update([det((5.0, 0.0, 105.0, 100.0))], [])

        self.assertEqual(tracker.table_boxes[0], (5.0, 0.0, 105.0, 100.0))
        self.assertEqual(tracker.table_boxes[1], zones[1])

    def test_chair_zones_track_independently(self):
        tracker = LayoutZoneTracker(
            [(0.0, 0.0, 100.0, 100.0)],
            [(200.0, 0.0, 260.0, 80.0)],
            alpha=1.0,
        )
        tracker.update([], [det((200.0, 0.0, 260.0, 80.0), name="chair")])

        tracker.update([], [det((210.0, 0.0, 270.0, 80.0), name="chair")])

        self.assertEqual(tracker.table_boxes[0], (0.0, 0.0, 100.0, 100.0))
        self.assertEqual(tracker.chair_boxes[0], (210.0, 0.0, 270.0, 80.0))

    def test_reset_restores_calibrated_boxes_and_forgets_the_anchor(self):
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [], alpha=1.0)
        tracker.update([det((0.0, 0.0, 100.0, 100.0))], [])
        tracker.update([det((20.0, 0.0, 120.0, 100.0))], [])

        tracker.reset()

        self.assertEqual(tracker.table_boxes[0], (0.0, 0.0, 100.0, 100.0))
        # 장면이 바뀌었으므로 기준점도 다시 잡는다.
        tracker.update([det((20.0, 0.0, 120.0, 100.0))], [])
        self.assertEqual(tracker.table_boxes[0], (0.0, 0.0, 100.0, 100.0))


if __name__ == "__main__":
    unittest.main()

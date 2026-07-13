"""Unit tests for LayoutZoneTracker (real-time zone drift in layout mode)."""

from __future__ import annotations

import unittest

from seatnow_core import Detection, LayoutZoneTracker


def det(box, confidence=0.9, name="dining table"):
    return Detection(name=name, box=box, confidence=confidence)


class LayoutZoneTrackerTests(unittest.TestCase):
    def test_zone_moves_toward_matching_detection_with_ema(self):
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [], alpha=0.5)

        tracker.update([det((20.0, 0.0, 120.0, 100.0))], [])

        # alpha=0.5: 절반만 접근
        self.assertEqual(tracker.table_boxes[0], (10.0, 0.0, 110.0, 100.0))

    def test_zone_stays_without_detection(self):
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [])

        tracker.update([], [])

        self.assertEqual(tracker.table_boxes[0], (0.0, 0.0, 100.0, 100.0))

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
        tracker.update([weak, strong], [])

        self.assertEqual(tracker.table_boxes[0], (10.0, 0.0, 110.0, 100.0))

    def test_one_detection_moves_only_best_zone(self):
        zones = [(0.0, 0.0, 100.0, 100.0), (200.0, 0.0, 300.0, 100.0)]
        tracker = LayoutZoneTracker(zones, [], alpha=1.0)

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

        tracker.update([], [det((210.0, 0.0, 270.0, 80.0), name="chair")])

        self.assertEqual(tracker.table_boxes[0], (0.0, 0.0, 100.0, 100.0))
        self.assertEqual(tracker.chair_boxes[0], (210.0, 0.0, 270.0, 80.0))

    def test_reset_restores_calibrated_boxes(self):
        tracker = LayoutZoneTracker([(0.0, 0.0, 100.0, 100.0)], [], alpha=1.0)
        tracker.update([det((20.0, 0.0, 120.0, 100.0))], [])

        tracker.reset()

        self.assertEqual(tracker.table_boxes[0], (0.0, 0.0, 100.0, 100.0))


if __name__ == "__main__":
    unittest.main()

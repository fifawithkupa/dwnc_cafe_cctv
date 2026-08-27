"""End-to-end wiring tests for ``SeatNowAnalyzer.analyze`` with stub models.

The chair-linking regression (``f1f41d5`` removed the association calls,
``70a86bc`` restored only the table ROI) survived because every chair test
called the association helpers directly.  These tests drive ``analyze`` itself
with scripted detector output so a disconnected helper fails the suite.
"""

from __future__ import annotations

import unittest

import numpy as np

from seatnow_core import (
    AnalyzerConfig,
    OccupancyState,
    PoseState,
    SeatNowAnalyzer,
    _chair_table_score,
    Detection,
)


COCO_NAMES = {
    0: "person",
    39: "bottle",
    41: "cup",
    56: "chair",
    60: "dining table",
    63: "laptop",
    26: "handbag",
}
NAME_TO_ID = {name: class_id for class_id, name in COCO_NAMES.items()}


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _Vector:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _DetBox:
    def __init__(self, name, box, confidence):
        self.cls = _Scalar(NAME_TO_ID[name])
        self.conf = _Scalar(confidence)
        self.xyxy = [_Vector(box)]


class _DetBoxes:
    def __init__(self, entries):
        self._entries = [_DetBox(*entry) for entry in entries]
        self.xyxy = [_Vector(entry[1]) for entry in entries]
        self.conf = [_Scalar(entry[2]) for entry in entries]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)


class _Keypoints:
    def __init__(self, rows):
        self.data = [_Vector(row) for row in rows]


class _Result:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class _StubModel:
    def __init__(self, result):
        self._result = result

    def predict(self, **_kwargs):
        return [self._result]


def seated_keypoints(hip, knee, ankle, shoulder, confidence: float = 0.95):
    """17 COCO keypoints with a left side bent enough to read as seated."""
    from seatnow_core import L_ANK, L_HIP, L_KNE, L_SHO

    rows = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, point in (
        (L_SHO, shoulder),
        (L_HIP, hip),
        (L_KNE, knee),
        (L_ANK, ankle),
    ):
        rows[index] = [float(point[0]), float(point[1]), confidence]
    return rows


def build_analyzer(detections, poses=(), keypoints=(), layout=None, **config_kwargs):
    """Assemble an analyzer around scripted detector/pose output.

    ``__init__`` is bypassed on purpose: it loads ultralytics weights, which
    the logic suite must not require.
    """
    analyzer = SeatNowAnalyzer.__new__(SeatNowAnalyzer)
    analyzer.config = AnalyzerConfig(
        table_crop_objects=False,
        infer_occluded_tables=config_kwargs.pop("infer_occluded_tables", True),
        **config_kwargs,
    )
    analyzer.layout = layout
    analyzer.zone_tracker = None
    analyzer.det_model = _StubModel(_Result(boxes=_DetBoxes(detections)))
    analyzer.pose_model = _StubModel(
        _Result(boxes=_DetBoxes(poses), keypoints=_Keypoints(keypoints))
    )
    analyzer.names = COCO_NAMES
    analyzer.previous_poses = []
    analyzer.previous_pose_timestamp = None
    return analyzer


FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


class ChairLinkPipelineTests(unittest.TestCase):
    """Table geometry: a 4-seat table with a chair tucked in below it."""

    TABLE = (500.0, 300.0, 780.0, 430.0)
    CHAIR = (560.0, 400.0, 700.0, 560.0)

    def test_chair_link_is_strong_enough_to_propagate(self):
        score = _chair_table_score(
            Detection("chair", self.CHAIR, 0.85),
            Detection("dining table", self.TABLE, 0.75),
            (720, 1280),
        )
        self.assertGreaterEqual(score, AnalyzerConfig().strong_chair_link)

    def test_bag_on_linked_chair_occupies_the_table(self):
        """T1 regression: belongings on a chair must reach the table."""
        bag = (585.0, 430.0, 660.0, 500.0)
        analyzer = build_analyzer(
            [
                ("dining table", self.TABLE, 0.75),
                ("chair", self.CHAIR, 0.85),
                ("handbag", bag, 0.55),
            ]
        )

        analysis = analyzer.analyze(FRAME)

        tables = [table for table in analysis.tables if table.source == "detected"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].raw_state, OccupancyState.OCCUPIED)
        self.assertEqual(len(tables[0].occupied_chairs), 1)
        self.assertIn("occupied_chairs:1", tables[0].reason)

    def test_bare_linked_chair_leaves_the_table_empty(self):
        analyzer = build_analyzer(
            [
                ("dining table", self.TABLE, 0.75),
                ("chair", self.CHAIR, 0.85),
            ]
        )

        analysis = analyzer.analyze(FRAME)

        tables = [table for table in analysis.tables if table.source == "detected"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].raw_state, OccupancyState.EMPTY)
        self.assertEqual(tables[0].occupied_chairs, [])
        self.assertEqual(len(tables[0].connected_chairs), 1)

    def test_chair_detections_reach_the_analysis(self):
        """``seat_detections`` must be populated, not silently emptied."""
        analyzer = build_analyzer(
            [
                ("dining table", self.TABLE, 0.75),
                ("chair", self.CHAIR, 0.85),
            ]
        )

        analysis = analyzer.analyze(FRAME)

        tables = [table for table in analysis.tables if table.source == "detected"]
        self.assertTrue(tables[0].connected_chairs)


class OccludedSeatPipelineTests(unittest.TestCase):
    """The compact-pose rescue path that plan.md §T1 calls the fatal chain."""

    CHAIR = (560.0, 400.0, 700.0, 560.0)
    PERSON = (565.0, 300.0, 695.0, 520.0)

    def _keypoints_upper_body_only(self):
        from seatnow_core import L_HIP, L_SHO, R_HIP, R_SHO

        rows = [[0.0, 0.0, 0.0] for _ in range(17)]
        rows[L_SHO] = [590.0, 330.0, 0.9]
        rows[R_SHO] = [670.0, 330.0, 0.9]
        rows[L_HIP] = [595.0, 430.0, 0.9]
        rows[R_HIP] = [665.0, 430.0, 0.9]
        return rows

    def test_inferred_seat_needs_chair_support(self):
        """No chair in the frame: the occluded customer stays unsurfaced."""
        analyzer = build_analyzer(
            [],
            poses=[("person", self.PERSON, 0.72)],
            keypoints=[
                seated_keypoints(
                    hip=(600.0, 430.0),
                    knee=(680.0, 440.0),
                    ankle=(670.0, 520.0),
                    shoulder=(600.0, 330.0),
                )
            ],
        )

        analysis = analyzer.analyze(FRAME)

        self.assertEqual(analysis.tables, [])

    def test_seated_person_on_chair_surfaces_an_inferred_seat(self):
        """T1 regression: ``has_seat_support`` needs real chair detections."""
        analyzer = build_analyzer(
            [("chair", self.CHAIR, 0.85)],
            poses=[("person", self.PERSON, 0.72)],
            keypoints=[
                seated_keypoints(
                    hip=(600.0, 430.0),
                    knee=(680.0, 440.0),
                    ankle=(670.0, 520.0),
                    shoulder=(600.0, 330.0),
                )
            ],
        )

        analysis = analyzer.analyze(FRAME)

        inferred = [
            table for table in analysis.tables if table.source == "inferred-seat"
        ]
        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0].raw_state, OccupancyState.OCCUPIED)

    def test_compact_occluded_pose_records_its_seat_support(self):
        """The support score must be measurable, not a hard-coded 0.00."""
        analyzer = build_analyzer(
            [("chair", self.CHAIR, 0.85)],
            poses=[("person", (565.0, 300.0, 695.0, 440.0), 0.72)],
            keypoints=[self._keypoints_upper_body_only()],
        )

        analysis = analyzer.analyze(FRAME)

        compact = [
            pose
            for pose in analysis.poses
            if pose.reason.startswith("compact_occluded_pose")
        ]
        self.assertEqual(len(compact), 1)
        self.assertEqual(compact[0].state, PoseState.UNKNOWN)
        support = float(compact[0].reason.split("seat_support=")[1].split(";")[0])
        self.assertGreater(support, 0.0)


class LayoutChairPipelineTests(unittest.TestCase):
    """Calibrated chair zones must reach the analyzer in layout mode."""

    def _layout(self):
        from seatnow_layout import LayoutChair, LayoutTable, SeatLayout

        return SeatLayout(
            schema_version=2,
            source={},
            tables=(
                LayoutTable(
                    id=1,
                    name="A1",
                    box=(500.0, 300.0, 780.0, 430.0),
                    chairs=(LayoutChair(id=1, box=(560.0, 400.0, 700.0, 560.0)),),
                ),
            ),
        )

    def test_manual_chair_link_propagates_a_bag(self):
        """A hand-drawn link bypasses the geometric strong-link filter."""
        analyzer = build_analyzer(
            [("handbag", (585.0, 430.0, 660.0, 500.0), 0.55)],
            layout=self._layout(),
        )

        analysis = analyzer.analyze(FRAME)

        self.assertEqual(len(analysis.tables), 1)
        self.assertEqual(analysis.tables[0].raw_state, OccupancyState.OCCUPIED)
        self.assertEqual(len(analysis.tables[0].occupied_chairs), 1)

    def test_layout_zone_tracker_receives_chair_zones(self):
        analyzer = build_analyzer([], layout=self._layout())

        analyzer.analyze(FRAME)

        self.assertEqual(len(analyzer.zone_tracker.chair_boxes), 1)


class RawDetectionLoggingTests(unittest.TestCase):
    """T2: the log must separate a model miss from a code rejection."""

    def test_dropped_table_names_the_rule_that_rejected_it(self):
        # 0.14 clears table_rescue_confidence (0.12) but not table_confidence
        # (0.20), and no chairs back it, so the selection rules drop it.
        weak_table = (500.0, 300.0, 780.0, 430.0)
        analyzer = build_analyzer([("dining table", weak_table, 0.14)])

        analysis = analyzer.analyze(FRAME)

        self.assertEqual(analysis.tables, [])
        self.assertEqual(
            [rule for _, rule in analysis.dropped_tables],
            ["low_confidence_no_chair_support"],
        )

    def test_table_below_rescue_confidence_is_reported_separately(self):
        analyzer = build_analyzer(
            [("dining table", (500.0, 300.0, 780.0, 430.0), 0.05)]
        )

        analysis = analyzer.analyze(FRAME)

        self.assertEqual(
            [rule for _, rule in analysis.dropped_tables],
            ["below_rescue_confidence"],
        )

    def test_accepted_table_is_not_reported_as_dropped(self):
        analyzer = build_analyzer(
            [("dining table", (500.0, 300.0, 780.0, 430.0), 0.75)]
        )

        analysis = analyzer.analyze(FRAME)

        self.assertEqual(len(analysis.tables), 1)
        self.assertEqual(analysis.dropped_tables, [])

    def test_layout_mode_reports_no_dropped_tables(self):
        layout = LayoutChairPipelineTests()._layout()
        analyzer = build_analyzer(
            [("dining table", (10.0, 10.0, 90.0, 90.0), 0.9)], layout=layout
        )

        analysis = analyzer.analyze(FRAME)

        self.assertEqual(analysis.dropped_tables, [])

    def test_record_carries_raw_detections_only_when_requested(self):
        from seatnow_core import TableTracker, frame_log_record

        analyzer = build_analyzer(
            [
                ("dining table", (500.0, 300.0, 780.0, 430.0), 0.75),
                ("dining table", (100.0, 300.0, 200.0, 380.0), 0.14),
                ("chair", (560.0, 400.0, 700.0, 560.0), 0.85),
            ]
        )
        analysis = analyzer.analyze(FRAME)
        tracker = TableTracker()
        update = tracker.update(analysis.tables, 0.0, FRAME.shape[:2])

        plain = frame_log_record(0, analysis, update)
        verbose = frame_log_record(0, analysis, update, include_raw_detections=True)

        self.assertNotIn("raw_detections", plain)
        raw = verbose["raw_detections"]
        self.assertEqual(raw["counts"], {"chair": 1, "dining table": 2})
        self.assertEqual(len(raw["items"]), 3)
        self.assertEqual(
            [entry["rule"] for entry in raw["dropped_tables"]],
            ["low_confidence_no_chair_support"],
        )


def bar_layout():
    """A 2-seat bar zone inside FRAME (1280x720)."""
    from seatnow_layout import LayoutSeat, LayoutTable, SeatLayout

    return SeatLayout(
        schema_version=2,
        source={},
        tables=(
            LayoutTable(
                id=7,
                name="BAR",
                box=(200.0, 300.0, 800.0, 500.0),
                kind="counted_zone",
                seats=(
                    LayoutSeat(id=1, box=(200.0, 300.0, 500.0, 500.0)),
                    LayoutSeat(id=2, box=(500.0, 300.0, 800.0, 500.0)),
                ),
            ),
        ),
    )


class CountedZoneAnalyzeTests(unittest.TestCase):
    """A bar zone is judged one hand-drawn seat slot at a time."""

    def test_zone_produces_one_observation_per_seat(self):
        analyzer = build_analyzer([], layout=bar_layout())

        analysis = analyzer.analyze(FRAME)

        self.assertEqual(len(analysis.tables), 2)
        self.assertEqual(
            [table.layout_name for table in analysis.tables], ["BAR-1", "BAR-2"]
        )
        self.assertEqual([table.layout_id for table in analysis.tables], [1, 2])
        self.assertEqual(analysis.tables[0].layout_kind, "counted_zone")
        self.assertEqual(analysis.tables[0].layout_zone_name, "BAR")
        self.assertEqual(analysis.tables[0].layout_zone_id, 7)
        self.assertEqual(analysis.tables[0].layout_capacity, 2)

    def test_seated_person_occupies_only_their_own_seat(self):
        # 사람 박스 x 280~420 은 1번 칸(200~500) 안에만 들어간다.
        analyzer = build_analyzer(
            [],
            poses=[("person", (280.0, 300.0, 420.0, 520.0), 0.72)],
            keypoints=[
                seated_keypoints(
                    hip=(320.0, 430.0),
                    knee=(400.0, 440.0),
                    ankle=(390.0, 520.0),
                    shoulder=(320.0, 330.0),
                )
            ],
            layout=bar_layout(),
        )

        analysis = analyzer.analyze(FRAME)

        self.assertEqual(analysis.tables[0].raw_state, OccupancyState.OCCUPIED)
        self.assertEqual(analysis.tables[1].raw_state, OccupancyState.EMPTY)

    def test_belongings_alone_occupy_a_seat(self):
        analyzer = build_analyzer(
            [("handbag", (300.0, 350.0, 380.0, 420.0), 0.55)],
            layout=bar_layout(),
        )

        analysis = analyzer.analyze(FRAME)

        self.assertEqual(analysis.tables[0].raw_state, OccupancyState.OCCUPIED)
        self.assertEqual(analysis.tables[1].raw_state, OccupancyState.EMPTY)

    def test_person_spanning_two_seats_makes_both_unknown(self):
        # 사람 박스 x 400~620 은 1번 칸(200~500)과 2번 칸(500~800)에 모두 걸친다.
        # 겹침 비율은 각각 0.41 / 0.50 으로 minimum_overlap(0.20)을 넘는다.
        analyzer = build_analyzer(
            [],
            poses=[("person", (400.0, 300.0, 620.0, 520.0), 0.72)],
            keypoints=[
                seated_keypoints(
                    hip=(480.0, 430.0),
                    knee=(560.0, 440.0),
                    ankle=(550.0, 520.0),
                    shoulder=(480.0, 330.0),
                )
            ],
            layout=bar_layout(),
        )

        analysis = analyzer.analyze(FRAME)

        self.assertEqual(analysis.tables[0].raw_state, OccupancyState.UNKNOWN)
        self.assertEqual(analysis.tables[1].raw_state, OccupancyState.UNKNOWN)
        self.assertEqual(analysis.tables[0].reason, "spans_multiple_seats")
        self.assertFalse(analysis.tables[0].provisional)

    def test_person_inside_one_seat_does_not_trigger_span_rule(self):
        analyzer = build_analyzer(
            [],
            poses=[("person", (280.0, 300.0, 420.0, 520.0), 0.72)],
            keypoints=[
                seated_keypoints(
                    hip=(320.0, 430.0),
                    knee=(400.0, 440.0),
                    ankle=(390.0, 520.0),
                    shoulder=(320.0, 330.0),
                )
            ],
            layout=bar_layout(),
        )

        analysis = analyzer.analyze(FRAME)

        self.assertEqual(analysis.tables[0].raw_state, OccupancyState.OCCUPIED)
        self.assertNotEqual(analysis.tables[1].raw_state, OccupancyState.UNKNOWN)


class UnknownReasonPromotionTests(unittest.TestCase):
    """The pose-level cause must survive the trip up to the table."""

    TABLE = (500.0, 300.0, 780.0, 430.0)
    PERSON = (565.0, 300.0, 695.0, 440.0)

    def _keypoints_upper_body_only(self):
        from seatnow_core import L_HIP, L_SHO, R_HIP, R_SHO

        rows = [[0.0, 0.0, 0.0] for _ in range(17)]
        rows[L_SHO] = [590.0, 330.0, 0.9]
        rows[R_SHO] = [670.0, 330.0, 0.9]
        rows[L_HIP] = [595.0, 430.0, 0.9]
        rows[R_HIP] = [665.0, 430.0, 0.9]
        return rows

    def test_occluded_pose_reason_reaches_the_table_observation(self):
        analyzer = build_analyzer(
            [("dining table", self.TABLE, 0.75)],
            poses=[("person", self.PERSON, 0.72)],
            keypoints=[self._keypoints_upper_body_only()],
        )

        analysis = analyzer.analyze(FRAME)

        tables = [table for table in analysis.tables if table.source == "detected"]
        self.assertEqual(tables[0].raw_state, OccupancyState.UNKNOWN)
        self.assertTrue(
            tables[0].reason.startswith(
                "nearby_person_pose_unknown:compact_occluded_pose"
            ),
            tables[0].reason,
        )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for SeatNow's model-independent occupancy logic."""

from __future__ import annotations

import json
import unittest

import cv2
import numpy as np

from engine.seatnow_core import (
    L_ANK,
    L_HIP,
    L_KNE,
    L_SHO,
    associate_objects_to_chairs,
    Detection,
    filter_carried_objects,
    filter_strong_chair_links,
    select_table_candidates,
    FrameAnalysis,
    OccupancyState,
    PoseObservation,
    PoseState,
    TableObservation,
    TableTracker,
    Track,
    TrackerUpdate,
    angle_degrees,
    associate_chairs_to_tables,
    associate_objects,
    associate_people,
    associate_seated_people_to_chairs,
    classify_pose,
    deduplicate_tables,
    frame_log_record,
    evidence_code_from_log,
    is_scene_change,
    occupancy_evidence_code,
    occupancy_state_from_evidence,
    track_to_dict,
)


def keypoints_with_left_side(
    shoulder,
    hip,
    knee,
    ankle,
    confidence: float = 0.99,
):
    keypoints = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, point in (
        (L_SHO, shoulder),
        (L_HIP, hip),
        (L_KNE, knee),
        (L_ANK, ankle),
    ):
        keypoints[index] = [float(point[0]), float(point[1]), confidence]
    return keypoints


def pose_observation(
    state: PoseState,
    box=(0.0, 0.0, 40.0, 100.0),
    anchor=(20.0, 70.0),
    confidence: float = 0.9,
    reason: str = "fixture",
):
    return PoseObservation(
        box=box,
        confidence=confidence,
        state=state,
        anchor=anchor,
        reason=reason,
    )


def table_observation(
    state: OccupancyState,
    box=(100.0, 100.0, 200.0, 200.0),
    *,
    confidence: float = 0.9,
    source: str = "detected",
    objects=None,
    seated_people=None,
    connected_chairs=None,
    occupied_chairs=None,
    chair_seated_people=None,
    reason: str = "fixture",
    provisional: bool = False,
):
    return TableObservation(
        box=box,
        table_confidence=confidence,
        raw_state=state,
        raw_score=confidence,
        source=source,
        objects=list(objects or []),
        seated_people=list(seated_people or []),
        connected_chairs=list(connected_chairs or []),
        occupied_chairs=list(occupied_chairs or []),
        chair_seated_people=list(chair_seated_people or []),
        reason=reason,
        provisional=provisional,
    )


class SceneChangeTests(unittest.TestCase):
    def test_same_scene_with_translation_is_not_a_cut(self):
        previous = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(previous, (40, 40), (130, 170), (255, 255, 255), -1)
        cv2.circle(previous, (230, 100), 35, (30, 180, 240), -1)
        transform = np.float32([[1, 0, 25], [0, 1, 8]])
        current = cv2.warpAffine(previous, transform, (320, 240))
        changed, _ = is_scene_change(previous, current)
        self.assertFalse(changed)

    def test_unrelated_high_difference_frames_are_a_cut(self):
        rng = np.random.default_rng(7)
        previous = rng.integers(0, 80, (240, 320, 3), dtype=np.uint8)
        current = rng.integers(175, 255, (240, 320, 3), dtype=np.uint8)
        changed, metrics = is_scene_change(previous, current)
        self.assertTrue(changed)
        self.assertLess(metrics["orb_matches"], 20)


class AngleTests(unittest.TestCase):
    def test_right_and_straight_angles(self):
        self.assertAlmostEqual(
            angle_degrees((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)),
            90.0,
        )
        self.assertAlmostEqual(
            angle_degrees((0.0, -1.0), (0.0, 0.0), (0.0, 1.0)),
            180.0,
        )

    def test_missing_or_degenerate_points_have_no_angle(self):
        self.assertIsNone(angle_degrees(None, (0.0, 0.0), (1.0, 0.0)))
        self.assertIsNone(
            angle_degrees((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))
        )


class PoseClassificationTests(unittest.TestCase):
    def test_bent_leg_is_classified_as_seated(self):
        keypoints = keypoints_with_left_side(
            shoulder=(0.0, -1.0),
            hip=(0.0, 0.0),
            knee=(1.0, 0.0),
            ankle=(1.0, 1.0),
        )

        result = classify_pose(keypoints, (0.0, 0.0, 10.0, 20.0), 0.91)

        self.assertEqual(result.state, PoseState.SEATED)
        self.assertAlmostEqual(result.angles["left_hka"], 90.0)
        self.assertIn("<110", result.reason)
        self.assertEqual(result.anchor, (0.0, 0.0))

    def test_straight_body_is_classified_as_standing(self):
        keypoints = keypoints_with_left_side(
            shoulder=(0.0, 0.0),
            hip=(0.0, 1.0),
            knee=(0.0, 2.0),
            ankle=(0.0, 3.0),
        )

        result = classify_pose(keypoints, (0.0, 0.0, 10.0, 20.0), 0.88)

        self.assertEqual(result.state, PoseState.STANDING)
        self.assertAlmostEqual(result.angles["left_hka"], 180.0)
        self.assertAlmostEqual(result.angles["left_torso"], 180.0)
        self.assertIn("all_angles>=threshold", result.reason)

    def test_missing_confident_joints_is_unknown(self):
        keypoints = keypoints_with_left_side(
            shoulder=(0.0, 0.0),
            hip=(0.0, 1.0),
            knee=(0.0, 2.0),
            ankle=(0.0, 3.0),
            confidence=0.29,
        )

        result = classify_pose(
            keypoints,
            (10.0, 20.0, 110.0, 220.0),
            0.75,
            keypoint_threshold=0.30,
        )

        self.assertEqual(result.state, PoseState.UNKNOWN)
        self.assertEqual(result.reason, "insufficient_keypoints")
        self.assertEqual(result.angles, {})
        self.assertEqual(result.anchor, (60.0, 164.0))


class TableDeduplicationTests(unittest.TestCase):
    def test_keeps_highest_confidence_duplicate_and_adjacent_table(self):
        high_confidence = Detection(
            "dining table", (0.0, 0.0, 100.0, 100.0), 0.95
        )
        nested_duplicate = Detection(
            "dining table", (10.0, 10.0, 90.0, 90.0), 0.55
        )
        adjacent = Detection(
            "dining table", (110.0, 0.0, 210.0, 100.0), 0.85
        )

        result = deduplicate_tables(
            [nested_duplicate, adjacent, high_confidence], overlap_threshold=0.65
        )

        self.assertEqual(result, [high_confidence, adjacent])


class AssociationTests(unittest.TestCase):
    def test_each_object_is_assigned_to_at_most_one_table(self):
        tables = [
            Detection("dining table", (0.0, 100.0, 100.0, 200.0), 0.9),
            Detection("dining table", (70.0, 100.0, 170.0, 200.0), 0.9),
        ]
        shared_candidate = Detection("cup", (40.0, 90.0, 65.0, 125.0), 0.8)
        unrelated = Detection("backpack", (500.0, 500.0, 520.0, 520.0), 0.8)

        assignments = associate_objects(tables, [shared_candidate, unrelated])

        self.assertEqual(assignments[0], [shared_candidate])
        self.assertEqual(assignments[1], [])
        assigned = [obj for values in assignments.values() for obj in values]
        self.assertEqual(assigned.count(shared_candidate), 1)
        self.assertNotIn(unrelated, assigned)

    def test_each_seated_person_is_assigned_to_one_nearest_table(self):
        tables = [
            Detection("dining table", (100.0, 100.0, 200.0, 200.0), 0.9),
            Detection("dining table", (220.0, 100.0, 320.0, 200.0), 0.9),
        ]
        seated = pose_observation(
            PoseState.SEATED,
            box=(170.0, 150.0, 210.0, 260.0),
            anchor=(190.0, 170.0),
        )
        far_away = pose_observation(
            PoseState.SEATED,
            box=(370.0, 330.0, 400.0, 400.0),
            anchor=(390.0, 380.0),
        )
        standing = pose_observation(
            PoseState.STANDING,
            box=(170.0, 150.0, 210.0, 260.0),
            anchor=(190.0, 170.0),
        )

        assignments, unassigned = associate_people(
            tables, [seated, far_away, standing], frame_shape=(400, 400)
        )

        self.assertEqual(assignments[0], [seated])
        self.assertEqual(assignments[1], [])
        assigned = [person for values in assignments.values() for person in values]
        self.assertEqual(assigned.count(seated), 1)
        self.assertEqual(unassigned, [far_away])
        self.assertNotIn(standing, assigned)
        self.assertNotIn(standing, unassigned)

    def test_table_object_assignment_uses_original_tabletop_roi(self):
        table = Detection("dining table", (100.0, 100.0, 200.0, 200.0), 0.9)
        cup_on_table = Detection("cup", (130.0, 95.0, 155.0, 130.0), 0.8)
        bag_next_to_table = Detection("backpack", (225.0, 170.0, 265.0, 240.0), 0.82)

        assignments = associate_objects([table], [cup_on_table, bag_next_to_table])

        self.assertEqual(assignments[0], [cup_on_table])

    def test_chair_is_linked_to_only_one_nearest_table(self):
        tables = [
            Detection("dining table", (100.0, 100.0, 200.0, 180.0), 0.9),
            Detection("dining table", (400.0, 100.0, 500.0, 180.0), 0.9),
        ]
        chair = Detection("chair", (160.0, 150.0, 240.0, 280.0), 0.88)

        assignments = associate_chairs_to_tables(
            tables, [chair], frame_shape=(400, 600)
        )

        self.assertEqual(assignments[0], [0])
        self.assertEqual(assignments[1], [])

    def test_ambiguous_chair_between_tables_is_not_linked(self):
        tables = [
            Detection("dining table", (100.0, 100.0, 200.0, 180.0), 0.9),
            Detection("dining table", (300.0, 100.0, 400.0, 180.0), 0.9),
        ]
        chair = Detection("chair", (220.0, 150.0, 280.0, 280.0), 0.88)

        assignments = associate_chairs_to_tables(
            tables, [chair], frame_shape=(400, 500)
        )

        self.assertEqual(assignments, {0: [], 1: []})

    def test_only_seated_person_can_occupy_chair(self):
        chair = Detection("chair", (100.0, 100.0, 200.0, 260.0), 0.91)
        seated = pose_observation(
            PoseState.SEATED,
            box=(110.0, 60.0, 190.0, 250.0),
            anchor=(150.0, 175.0),
        )
        standing = pose_observation(
            PoseState.STANDING,
            box=(105.0, 40.0, 195.0, 260.0),
            anchor=(150.0, 180.0),
        )

        assignments = associate_seated_people_to_chairs(
            [chair], [standing, seated]
        )

        self.assertEqual(assignments[0], [seated])
        self.assertNotIn(standing, assignments[0])

    def test_table_or_rule_covers_chair_only_evidence(self):
        """An occupied linked chair occupies its table (regression: f1f41d5).

        ``f1f41d5`` dropped chair propagation because the expanded table ROI
        was supposed to cover the seats; ``70a86bc`` then reverted that ROI and
        left occupancy with no path from chair to table at all.
        """
        chair = Detection("chair", (100.0, 100.0, 200.0, 260.0), 0.91)
        cup = Detection("cup", (120.0, 100.0, 140.0, 130.0), 0.8)
        seated = pose_observation(PoseState.SEATED)
        unknown = pose_observation(PoseState.UNKNOWN)

        self.assertEqual(
            occupancy_state_from_evidence([], [], [], [chair]),
            OccupancyState.OCCUPIED,
        )
        # A chair carrying evidence outranks an ambiguous nearby pose.
        self.assertEqual(
            occupancy_state_from_evidence([], [], [unknown], [chair]),
            OccupancyState.OCCUPIED,
        )
        self.assertEqual(
            occupancy_state_from_evidence([cup], [], [], []),
            OccupancyState.OCCUPIED,
        )
        self.assertEqual(
            occupancy_state_from_evidence([], [seated], [], []),
            OccupancyState.OCCUPIED,
        )
        self.assertEqual(
            occupancy_state_from_evidence([], [], [unknown], []),
            OccupancyState.UNKNOWN,
        )
        self.assertEqual(
            occupancy_state_from_evidence([], [], [], []),
            OccupancyState.EMPTY,
        )


class TableTrackerTests(unittest.TestCase):
    FRAME_SHAPE = (1080, 1920)

    def test_global_assignment_prefers_total_quality_over_match_count(self):
        selected = TableTracker._select_global_matches(
            [(0.90, 0, 0), (0.24, 0, 1), (0.24, 1, 0)],
            track_count=2,
            observation_count=2,
        )
        self.assertEqual(selected, [(0.90, 0, 0)])

    def test_state_changes_are_asymmetrically_debounced(self):
        tracker = TableTracker(
            occupy_confirmations=2,
            empty_confirmations=3,
            max_missed=2,
        )
        empty = table_observation(OccupancyState.EMPTY)
        occupied = table_observation(OccupancyState.OCCUPIED)

        initial = tracker.update([empty], 0.0, self.FRAME_SHAPE)
        self.assertEqual(initial.visible_tracks[0].stable_state, OccupancyState.EMPTY)

        first_occupied = tracker.update([occupied], 1.0, self.FRAME_SHAPE)
        self.assertEqual(
            first_occupied.visible_tracks[0].stable_state, OccupancyState.EMPTY
        )
        self.assertEqual(first_occupied.visible_tracks[0].pending_count, 1)
        self.assertFalse(
            any(event["type"] == "state_change" for event in first_occupied.events)
        )

        second_occupied = tracker.update([occupied], 2.0, self.FRAME_SHAPE)
        self.assertEqual(
            second_occupied.visible_tracks[0].stable_state,
            OccupancyState.OCCUPIED,
        )
        self.assertEqual(
            [event["to"] for event in second_occupied.events if event["type"] == "state_change"],
            ["occupied"],
        )

        for timestamp in (3.0, 4.0):
            pending_empty = tracker.update([empty], timestamp, self.FRAME_SHAPE)
            self.assertEqual(
                pending_empty.visible_tracks[0].stable_state,
                OccupancyState.OCCUPIED,
            )

        confirmed_empty = tracker.update([empty], 5.0, self.FRAME_SHAPE)
        self.assertEqual(
            confirmed_empty.visible_tracks[0].stable_state, OccupancyState.EMPTY
        )
        self.assertEqual(
            [event["to"] for event in confirmed_empty.events if event["type"] == "state_change"],
            ["empty"],
        )

    def test_new_track_initializes_immediately_then_debounces_transitions(self):
        tracker = TableTracker(
            occupy_confirmations=2,
            empty_confirmations=3,
            max_missed=2,
        )
        occupied = table_observation(OccupancyState.OCCUPIED)

        first = tracker.update([occupied], 0.0, self.FRAME_SHAPE)
        self.assertEqual(first.visible_tracks[0].stable_state, OccupancyState.OCCUPIED)
        self.assertIsNone(first.visible_tracks[0].pending_state)
        self.assertEqual(first.visible_tracks[0].pending_count, 0)

        second = tracker.update([occupied], 1.0, self.FRAME_SHAPE)
        self.assertEqual(second.visible_tracks[0].stable_state, OccupancyState.OCCUPIED)

    def test_provisional_chair_only_seat_requires_second_observation(self):
        tracker = TableTracker(occupy_confirmations=2, empty_confirmations=3)
        weak = table_observation(
            OccupancyState.OCCUPIED,
            source="inferred-seat",
            provisional=True,
        )
        first = tracker.update([weak], 5.0, self.FRAME_SHAPE)
        self.assertEqual(first.visible_tracks[0].stable_state, OccupancyState.UNKNOWN)
        self.assertEqual(first.visible_tracks[0].pending_count, 1)
        second = tracker.update([weak], 6.0, self.FRAME_SHAPE)
        self.assertEqual(second.visible_tracks[0].stable_state, OccupancyState.OCCUPIED)

    def test_provisional_confirmation_resets_after_a_miss(self):
        tracker = TableTracker(occupy_confirmations=2, empty_confirmations=3)
        weak = table_observation(
            OccupancyState.OCCUPIED,
            source="inferred-seat",
            provisional=True,
        )
        first = tracker.update([weak], 5.0, self.FRAME_SHAPE)
        track_id = first.visible_tracks[0].track_id
        tracker.update([], 6.0, self.FRAME_SHAPE)
        third = tracker.update([weak], 7.0, self.FRAME_SHAPE)
        track = next(track for track in third.visible_tracks if track.track_id == track_id)
        self.assertEqual(track.stable_state, OccupancyState.UNKNOWN)
        self.assertEqual(track.pending_count, 1)

    def test_velocity_keeps_adjacent_pan_tracks_from_stealing_identity(self):
        tracker = TableTracker(
            occupy_confirmations=1,
            empty_confirmations=1,
            max_missed=2,
        )
        first_a = table_observation(OccupancyState.OCCUPIED, box=(100, 100, 260, 200))
        first_b = table_observation(OccupancyState.EMPTY, box=(200, 100, 340, 200))
        initial = tracker.update([first_a, first_b], 0.0, self.FRAME_SHAPE)
        occupied_id = next(track.track_id for track in initial.visible_tracks if track.stable_state == OccupancyState.OCCUPIED)
        empty_id = next(track.track_id for track in initial.visible_tracks if track.stable_state == OccupancyState.EMPTY)

        second_a = table_observation(OccupancyState.OCCUPIED, box=(50, 100, 210, 200))
        second_b = table_observation(OccupancyState.EMPTY, box=(150, 100, 290, 200))
        tracker.update([second_a, second_b], 1.0, self.FRAME_SHAPE)

        # A disappears. B continues at the position that stale A overlap alone
        # could otherwise steal; velocity predicts B's center correctly.
        third_b = table_observation(OccupancyState.EMPTY, box=(100, 100, 240, 200))
        third = tracker.update([third_b], 2.0, self.FRAME_SHAPE)
        observed_tracks = [track for track in third.visible_tracks if not track.predicted]
        self.assertEqual(len(observed_tracks), 1)
        self.assertEqual(observed_tracks[0].track_id, empty_id)
        self.assertNotEqual(observed_tracks[0].track_id, occupied_id)

    def test_inferred_and_detected_sources_do_not_cross_match_at_distance(self):
        tracker = TableTracker(occupy_confirmations=1, empty_confirmations=1)
        inferred = table_observation(
            OccupancyState.OCCUPIED,
            box=(100, 100, 160, 180),
            source="inferred-seat",
        )
        first = tracker.update([inferred], 0.0, self.FRAME_SHAPE)
        inferred_id = first.visible_tracks[0].track_id
        detected = table_observation(
            OccupancyState.EMPTY,
            box=(320, 100, 420, 200),
            source="detected",
        )
        second = tracker.update([detected], 1.0, self.FRAME_SHAPE)
        detected_tracks = [
            track
            for track in second.visible_tracks
            if not track.predicted and track.last_observation.source == "detected"
        ]
        self.assertEqual(detected_tracks, [])
        self.assertEqual(second.pending_layout_changes[0]["change_type"], "ADDED")
        self.assertNotEqual(
            second.pending_layout_changes[0].get("source_table_ids"),
            [inferred_id],
        )

    def test_close_inferred_to_detected_transition_merges_without_double_count(self):
        tracker = TableTracker(occupy_confirmations=1, empty_confirmations=1)
        inferred = table_observation(
            OccupancyState.OCCUPIED,
            box=(100, 100, 160, 180),
            source="inferred-seat",
        )
        first = tracker.update([inferred], 0.0, self.FRAME_SHAPE)
        track_id = first.visible_tracks[0].track_id
        detected = table_observation(
            OccupancyState.OCCUPIED,
            box=(180, 100, 280, 200),
            source="detected",
        )
        second = tracker.update([detected], 1.0, self.FRAME_SHAPE)
        self.assertEqual(len(second.visible_tracks), 1)
        self.assertEqual(second.visible_tracks[0].track_id, track_id)
        self.assertFalse(second.visible_tracks[0].predicted)

    def test_adjacent_tables_can_change_state_without_identity_swap(self):
        tracker = TableTracker(
            occupy_confirmations=1,
            empty_confirmations=1,
            max_missed=1,
        )
        first = tracker.update(
            [
                table_observation(OccupancyState.OCCUPIED, box=(50, 50, 150, 150)),
                table_observation(OccupancyState.EMPTY, box=(130, 50, 230, 150)),
            ],
            0.0,
            self.FRAME_SHAPE,
        )
        left_id = first.visible_tracks[0].track_id
        right_id = first.visible_tracks[1].track_id
        changed = tracker.update(
            [
                table_observation(OccupancyState.EMPTY, box=(90, 50, 190, 150)),
                table_observation(OccupancyState.OCCUPIED, box=(170, 50, 270, 150)),
            ],
            1.0,
            self.FRAME_SHAPE,
        )
        by_id = {track.track_id: track for track in changed.visible_tracks}
        self.assertEqual(by_id[left_id].stable_state, OccupancyState.EMPTY)
        self.assertEqual(by_id[right_id].stable_state, OccupancyState.OCCUPIED)
        transitions = [event for event in changed.events if event["type"] == "state_change"]
        self.assertEqual(len(transitions), 2)

    def test_temporary_occlusion_is_predicted_then_expires_without_empty(self):
        tracker = TableTracker(
            occupy_confirmations=1,
            empty_confirmations=1,
            max_missed=1,
        )
        occupied = table_observation(OccupancyState.OCCUPIED)
        entered = tracker.update([occupied], 0.0, self.FRAME_SHAPE)
        table_id = entered.visible_tracks[0].track_id

        missed_once = tracker.update([], 1.0, self.FRAME_SHAPE)
        self.assertEqual(len(missed_once.visible_tracks), 1)
        self.assertTrue(missed_once.visible_tracks[0].predicted)
        self.assertEqual(missed_once.all_tracks[0].stable_state, OccupancyState.OCCUPIED)
        self.assertFalse(
            any(event["type"] == "state_change" for event in missed_once.events)
        )

        left_view = tracker.update([], 2.0, self.FRAME_SHAPE)
        self.assertEqual(left_view.all_tracks, [])
        self.assertFalse(
            any(
                event["type"] == "state_change" and event.get("to") == "empty"
                for event in left_view.events
            )
        )
        self.assertIn(
            {
                "type": "left_view",
                "table_id": table_id,
                "last_state": "occupied",
                "timestamp": 2.0,
            },
            left_view.events,
        )
        self.assertEqual(left_view.committed_layout_changes[0]["change_type"], "REMOVED")

    def test_border_ignore_is_excluded_without_empty_transition(self):
        tracker = TableTracker(
            occupy_confirmations=1,
            empty_confirmations=1,
            max_missed=1,
        )
        occupied = table_observation(OccupancyState.OCCUPIED)
        ignored = table_observation(OccupancyState.IGNORE, reason="border_cropped")
        tracker.update([occupied], 0.0, self.FRAME_SHAPE)

        update = tracker.update([ignored], 1.0, self.FRAME_SHAPE)

        self.assertEqual(update.visible_tracks[0].visible_state, OccupancyState.IGNORE)
        self.assertEqual(
            update.visible_tracks[0].stable_state,
            OccupancyState.OCCUPIED,
            "IGNORE is frame-local while the last decisive world state is retained",
        )
        self.assertFalse(
            any(
                event["type"] == "state_change" and event.get("to") == "empty"
                for event in update.events
            )
        )


class FrameLogTests(unittest.TestCase):
    def test_layout_observation_gets_layout_label_and_log_fields(self):
        observation = table_observation(
            OccupancyState.OCCUPIED, box=(0.0, 0.0, 40.0, 80.0)
        )
        observation.source = "layout"
        observation.layout_id = 7
        observation.layout_name = "창가1"
        track = Track(
            track_id=1,
            box=observation.box,
            stable_state=observation.raw_state,
            last_observation=observation,
            first_seen=0.0,
            last_seen=0.0,
        )

        record = track_to_dict(track)

        self.assertEqual(track.label, "L007")
        self.assertEqual(record["label"], "L007")
        self.assertEqual(record["layout_id"], 7)
        self.assertEqual(record["layout_name"], "창가1")
        self.assertEqual(record["source"], "layout")

    def test_frame_log_contains_visible_counts_evidence_and_pose_counts(self):
        cup = Detection("cup", (10.0, 10.0, 20.0, 20.0), 0.87654)
        chair = Detection("chair", (0.0, 20.0, 35.0, 80.0), 0.81234)
        seated = pose_observation(PoseState.SEATED, confidence=0.93456)
        standing = pose_observation(PoseState.STANDING)
        unknown = pose_observation(PoseState.UNKNOWN)
        observations = [
            table_observation(
                OccupancyState.OCCUPIED,
                box=(0.0, 0.0, 40.0, 80.0),
                confidence=0.98765,
                source="inferred-seat",
                objects=[cup],
                seated_people=[seated],
                connected_chairs=[chair],
                occupied_chairs=[chair],
                chair_seated_people=[seated],
                reason="seated:1;objects:cup",
            ),
            table_observation(
                OccupancyState.EMPTY, box=(100.0, 0.0, 140.0, 80.0)
            ),
            table_observation(
                OccupancyState.UNKNOWN, box=(200.0, 0.0, 240.0, 80.0)
            ),
            table_observation(
                OccupancyState.IGNORE, box=(300.0, 0.0, 340.0, 80.0)
            ),
        ]
        visible_tracks = [
            Track(
                track_id=index,
                box=observation.box,
                stable_state=observation.raw_state,
                last_observation=observation,
                first_seen=1.0,
                last_seen=1.0,
            )
            for index, observation in enumerate(observations, start=1)
        ]
        hidden_observation = table_observation(
            OccupancyState.OCCUPIED, box=(400.0, 0.0, 440.0, 80.0)
        )
        hidden_track = Track(
            track_id=5,
            box=hidden_observation.box,
            stable_state=OccupancyState.OCCUPIED,
            last_observation=hidden_observation,
            first_seen=0.0,
            last_seen=0.0,
            visible=False,
            missed=1,
        )
        update = TrackerUpdate(
            visible_tracks=visible_tracks,
            all_tracks=visible_tracks + [hidden_track],
            events=[{"type": "fixture_event", "timestamp": 1.23456789}],
        )
        analysis = FrameAnalysis(
            timestamp=1.23456789,
            tables=observations,
            poses=[seated, standing, unknown],
            detections=[cup],
            inference_ms=12.346,
        )

        record = frame_log_record(7, analysis, update)

        self.assertEqual(record["frame_index"], 7)
        self.assertEqual(record["timestamp"], 1.234568)
        self.assertEqual(record["inference_ms"], 12.35)
        self.assertEqual(
            record["summary"],
            {
                "visible": 4,
                "occupied": 1,
                "empty": 1,
                "unknown": 1,
                "ignore": 1,
                "inferred_seats": 1,
                "occupied_chairs": 1,
                "seated_poses": 1,
                "standing_poses": 1,
                "unknown_poses": 1,
            },
        )
        self.assertEqual(len(record["tables"]), 4)
        self.assertEqual(record["tables"][0]["label"], "S001")
        self.assertEqual(record["tables"][0]["state"], "occupied")
        self.assertEqual(record["tables"][0]["raw_state"], "occupied")
        self.assertEqual(record["tables"][0]["confidence"], 0.9877)
        self.assertEqual(record["tables"][0]["objects"], [{"class": "cup", "confidence": 0.8765}])
        self.assertEqual(record["tables"][0]["seated_people"], 1)
        self.assertEqual(record["tables"][0]["occupied_chairs"], 1)
        self.assertEqual(record["tables"][0]["chair_seated_people"], 1)
        self.assertTrue(record["tables"][0]["connected_chairs"][0]["occupied"])
        self.assertEqual(record["events"], update.events)
        json.dumps(record)

    def test_frame_log_record_carries_seat_report(self):
        observation = table_observation(
            OccupancyState.EMPTY, source="layout", reason="no_customer_evidence"
        )
        observation.layout_id = 1
        observation.layout_name = "창가1"
        track = Track(
            track_id=1,
            box=observation.box,
            stable_state=OccupancyState.EMPTY,
            last_observation=observation,
            first_seen=0.0,
            last_seen=0.0,
        )
        update = TrackerUpdate(
            visible_tracks=[track], all_tracks=[track], events=[]
        )
        analysis = FrameAnalysis(
            timestamp=0.0,
            tables=[observation],
            poses=[],
            detections=[],
            inference_ms=1.0,
        )

        record = frame_log_record(0, analysis, update)

        self.assertIn("seat_report", record)
        self.assertEqual(record["seat_report"]["totals"]["free"], 1)
        self.assertEqual(record["seat_report"]["totals"]["capacity"], 1)
        self.assertEqual(record["seat_report"]["seats"][0]["seat_id"], "창가1")
        self.assertEqual(record["seat_report"]["seats"][0]["kind"], "table")
        self.assertEqual(
            record["seat_report"]["seats"][0]["reason_code"], "no_customer_evidence"
        )
        json.dumps(record)


class TableCandidateSelectionTests(unittest.TestCase):
    """Regression tests built from the 2026-07 sample_raw probe evidence:
    real foreground tables (conf up to 0.80) were rejected by the blanket
    6% frame-area cap, and real back tables at conf 0.15-0.20 fell just
    under the confidence threshold despite confident chairs beside them."""

    FRAME_SHAPE = (720, 1280)

    def test_confident_large_foreground_table_is_accepted(self):
        # 좌석착석 probe: conf=0.80, area_frac=0.077 was rejected by the old cap.
        table = Detection("dining table", (457.0, 422.0, 701.0, 713.0), 0.80)

        accepted = select_table_candidates([table], [], self.FRAME_SHAPE)

        self.assertEqual(accepted, [table])

    def test_low_confidence_large_box_without_chairs_is_rejected(self):
        table = Detection("dining table", (756.0, 427.0, 1128.0, 714.0), 0.156)

        accepted = select_table_candidates([table], [], self.FRAME_SHAPE)

        self.assertEqual(accepted, [])

    def test_low_confidence_large_table_is_rescued_by_two_confident_chairs(self):
        # 큰가방의자점유 probe: the laptop table (conf=0.156, area_frac=0.116)
        # has two confident chairs tucked into it.
        table = Detection("dining table", (756.0, 427.0, 1128.0, 714.0), 0.156)
        chairs = [
            Detection("chair", (868.0, 371.0, 991.0, 517.0), 0.85),
            Detection("chair", (979.0, 447.0, 1215.0, 701.0), 0.86),
        ]

        accepted = select_table_candidates([table], chairs, self.FRAME_SHAPE)

        self.assertEqual(accepted, [table])

    def test_marginal_confidence_small_table_is_rescued_by_chairs(self):
        # 좌석착석 probe: back table at conf=0.197 sat just under the 0.20 gate.
        table = Detection("dining table", (688.0, 263.0, 818.0, 454.0), 0.197)
        chairs = [
            Detection("chair", (580.0, 278.0, 691.0, 442.0), 0.93),
            Detection("chair", (750.0, 246.0, 847.0, 389.0), 0.61),
        ]

        self.assertEqual(
            select_table_candidates([table], chairs, self.FRAME_SHAPE), [table]
        )
        self.assertEqual(select_table_candidates([table], [], self.FRAME_SHAPE), [])

    def test_scene_sized_box_is_rejected_even_when_confident(self):
        table = Detection("dining table", (0.0, 100.0, 1280.0, 500.0), 0.9)

        accepted = select_table_candidates([table], [], self.FRAME_SHAPE)

        self.assertEqual(accepted, [])

    def test_phantom_low_confidence_table_stays_rejected_despite_chairs(self):
        # Pan-fixture regression: a 67x39px conf=0.06 box in a chair-dense
        # real cafe must not become a reported (empty) table.
        phantom = Detection("dining table", (1439.0, 647.0, 1506.0, 686.0), 0.06)
        chairs = [
            Detection("chair", (1380.0, 620.0, 1460.0, 760.0), 0.8),
            Detection("chair", (1500.0, 630.0, 1590.0, 770.0), 0.8),
        ]

        accepted = select_table_candidates([phantom], chairs, (1080, 1920))

        self.assertEqual(accepted, [])

    def test_large_box_overlapping_more_confident_smaller_table_is_vetoed(self):
        # Pan-fixture regression: a chair-backed 13%-of-frame box absorbed the
        # teddy-bear table's evidence; the detector was more confident about
        # the smaller table it overlapped.
        small = Detection("dining table", (531.0, 665.0, 730.0, 709.0), 0.39)
        giant = Detection("dining table", (648.0, 664.0, 1341.0, 1070.0), 0.23)
        chairs = [
            Detection("chair", (700.0, 800.0, 860.0, 1020.0), 0.8),
            Detection("chair", (1100.0, 780.0, 1260.0, 1000.0), 0.8),
        ]

        accepted = select_table_candidates(
            [giant, small], chairs, (1080, 1920)
        )

        self.assertIn(small, accepted)
        self.assertNotIn(giant, accepted)

    def test_large_box_is_vetoed_even_when_more_confident_than_smaller_table(self):
        # Pan fixture t=2: the merged box (0.33) outscored the teddy-bear
        # table sliver (0.28) and stole its evidence again.
        small = Detection("dining table", (477.0, 664.0, 679.0, 709.0), 0.28)
        giant = Detection("dining table", (590.0, 665.0, 1287.0, 1069.0), 0.33)

        accepted = select_table_candidates([giant, small], [], (1080, 1920))

        self.assertIn(small, accepted)
        self.assertNotIn(giant, accepted)

    def test_merged_box_containing_two_accepted_tables_is_vetoed(self):
        left = Detection("dining table", (439.0, 309.0, 567.0, 436.0), 0.36)
        right = Detection("dining table", (641.0, 299.0, 811.0, 412.0), 0.39)
        merged = Detection("dining table", (430.0, 290.0, 820.0, 460.0), 0.60)

        accepted = select_table_candidates(
            [merged, left, right], [], self.FRAME_SHAPE
        )

        self.assertIn(left, accepted)
        self.assertIn(right, accepted)
        self.assertNotIn(merged, accepted)


class StrongChairLinkTests(unittest.TestCase):
    """Pan-fixture regression: a chair 0.67-linked to the neighbouring table
    absorbed the seated customer of a fully occluded table, deleting that
    table from the occupied count.  Occupancy must only propagate over
    tucked-in chairs (score 0.94+)."""

    def test_distant_absorbing_chair_is_not_a_strong_link(self):
        table = Detection("dining table", (369.0, 685.0, 594.0, 867.0), 0.85)
        absorbing = Detection("chair", (161.0, 826.0, 386.0, 1075.0), 0.77)
        tucked_in = Detection("chair", (500.0, 707.0, 670.0, 900.0), 0.93)
        assignments = {0: [0, 1]}

        strong = filter_strong_chair_links(
            [table], [absorbing, tucked_in], assignments, (1080, 1920)
        )

        self.assertEqual(strong, {0: [1]})


class MovingPersonCascadeTests(unittest.TestCase):
    """Pan-fixture regression: one downgraded frame must not lock a seated
    customer into standing forever."""

    @staticmethod
    def _fake_analyzer(previous_poses, previous_timestamp):
        from engine.seatnow_core import AnalyzerConfig, SeatNowAnalyzer

        class Fake:
            config = AnalyzerConfig()

        fake = Fake()
        fake.previous_poses = previous_poses
        fake.previous_pose_timestamp = previous_timestamp
        return fake, SeatNowAnalyzer._filter_moving_people

    def test_previous_downgrade_does_not_cascade(self):
        previous = pose_observation(
            PoseState.STANDING,
            box=(821.0, 567.0, 871.0, 716.0),
            reason="insufficient_keypoints;chair_overlap=0.55;moving=0.030/s",
        )
        current = pose_observation(
            PoseState.SEATED,
            box=(822.0, 567.0, 872.0, 716.0),
            reason="insufficient_keypoints;chair_overlap=0.55",
        )
        fake, method = self._fake_analyzer([previous], 8.0)

        method(fake, [current], 9.0, (1080, 1920), (0.0, 0.0))

        self.assertEqual(current.state, PoseState.SEATED)

    def test_genuine_standing_still_vetoes_weak_seated(self):
        previous = pose_observation(
            PoseState.STANDING,
            box=(821.0, 567.0, 871.0, 716.0),
            reason="all_angles>=threshold (min left_torso=139.5)",
        )
        current = pose_observation(
            PoseState.SEATED,
            box=(822.0, 567.0, 872.0, 716.0),
            reason="insufficient_keypoints;chair_overlap=0.55",
        )
        fake, method = self._fake_analyzer([previous], 8.0)

        method(fake, [current], 9.0, (1080, 1920), (0.0, 0.0))

        self.assertEqual(current.state, PoseState.STANDING)


class CarriedObjectTests(unittest.TestCase):
    """A bag or phone in the hands of a walking customer must not mark a
    table occupied; hallucinated objects on a person occluding a seat (the
    'banana' false positives) collapse onto the person box the same way."""

    def test_object_on_standing_person_is_dropped(self):
        standing = pose_observation(
            PoseState.STANDING, box=(600.0, 180.0, 743.0, 409.0)
        )
        carried = Detection("handbag", (640.0, 260.0, 700.0, 330.0), 0.4)

        kept = filter_carried_objects([carried], [standing])

        self.assertEqual(kept, [])

    def test_object_on_unknown_pose_is_dropped(self):
        unknown = pose_observation(
            PoseState.UNKNOWN, box=(600.0, 180.0, 743.0, 409.0)
        )
        hallucinated = Detection("banana", (630.0, 250.0, 690.0, 300.0), 0.24)

        self.assertEqual(filter_carried_objects([hallucinated], [unknown]), [])

    def test_object_away_from_people_is_kept(self):
        standing = pose_observation(
            PoseState.STANDING, box=(600.0, 180.0, 743.0, 409.0)
        )
        on_table = Detection("cup", (120.0, 470.0, 150.0, 510.0), 0.79)

        self.assertEqual(
            filter_carried_objects([on_table], [standing]), [on_table]
        )

    def test_object_overlapping_seated_person_is_kept(self):
        seated = pose_observation(
            PoseState.SEATED, box=(600.0, 180.0, 743.0, 409.0)
        )
        laptop = Detection("laptop", (620.0, 300.0, 720.0, 380.0), 0.6)

        self.assertEqual(filter_carried_objects([laptop], [seated]), [laptop])


class ChairObjectTests(unittest.TestCase):
    """큰가방의자점유 scenario: belongings resting on a chair must occupy
    that chair (and through it the linked table)."""

    def test_bag_resting_inside_chair_is_assigned(self):
        chair = Detection("chair", (330.0, 336.0, 472.0, 521.0), 0.88)
        bag = Detection("handbag", (360.0, 360.0, 440.0, 450.0), 0.5)

        assignments = associate_objects_to_chairs([chair], [bag])

        self.assertEqual(assignments[0], [bag])

    def test_object_away_from_chair_is_not_assigned(self):
        chair = Detection("chair", (330.0, 336.0, 472.0, 521.0), 0.88)
        cup = Detection("cup", (570.0, 480.0, 600.0, 520.0), 0.79)

        assignments = associate_objects_to_chairs([chair], [cup])

        self.assertEqual(assignments[0], [])

    def test_object_larger_than_chair_is_not_assigned(self):
        chair = Detection("chair", (330.0, 336.0, 472.0, 521.0), 0.88)
        huge = Detection("suitcase", (280.0, 250.0, 560.0, 700.0), 0.5)

        assignments = associate_objects_to_chairs([chair], [huge])

        self.assertEqual(assignments[0], [])

    def test_object_is_assigned_to_single_best_chair(self):
        left = Detection("chair", (330.0, 336.0, 472.0, 521.0), 0.88)
        right = Detection("chair", (470.0, 336.0, 610.0, 521.0), 0.88)
        bag = Detection("handbag", (400.0, 360.0, 480.0, 450.0), 0.5)

        assignments = associate_objects_to_chairs([left, right], [bag])

        assigned = [obj for values in assignments.values() for obj in values]
        self.assertEqual(assigned.count(bag), 1)


class BarSeatNeedsAPersonTests(unittest.TestCase):
    """A bar stool with only belongings on it stays available.

    One customer at a counter spreads a laptop, a cup and a bag across two
    or three seat widths, and each slot would see "objects" and go occupied
    -- one person eating three seats.  A table cannot do that because it is
    one unit.  Asked to move their things off a spare stool, most people do,
    so the seat is in practice available.

    Tables keep the old rule: belongings there mean the table is taken.
    """

    def _person(self):
        return PoseObservation(
            box=(100.0, 100.0, 200.0, 400.0),
            anchor=(150.0, 380.0),
            confidence=0.9,
            state=PoseState.SEATED,
            reason="",
            angles={},
        )

    def _object(self):
        return Detection(name="backpack", box=(120.0, 120.0, 180.0, 180.0), confidence=0.8)

    def test_table_with_belongings_only_is_occupied(self):
        self.assertEqual(
            occupancy_state_from_evidence([self._object()], [], [], []),
            OccupancyState.OCCUPIED,
        )

    def test_bar_seat_with_belongings_only_is_empty(self):
        self.assertEqual(
            occupancy_state_from_evidence(
                [self._object()], [], [], [], require_person=True
            ),
            OccupancyState.EMPTY,
        )

    def test_bar_seat_with_a_person_is_occupied(self):
        self.assertEqual(
            occupancy_state_from_evidence(
                [], [self._person()], [], [], require_person=True
            ),
            OccupancyState.OCCUPIED,
        )

    def test_bar_seat_with_person_and_belongings_is_occupied(self):
        self.assertEqual(
            occupancy_state_from_evidence(
                [self._object()], [self._person()], [], [], require_person=True
            ),
            OccupancyState.OCCUPIED,
        )

    def test_bar_seat_with_an_occupied_chair_but_no_person_is_empty(self):
        chair = Detection(name="chair", box=(100.0, 100.0, 200.0, 200.0), confidence=0.7)
        self.assertEqual(
            occupancy_state_from_evidence([], [], [], [chair], require_person=True),
            OccupancyState.EMPTY,
        )

    def test_bar_seat_with_an_unreadable_person_is_unknown(self):
        # Someone is there but the pose could not be read: that is not the
        # same as an empty stool and must not be rounded to available.
        unreadable = self._person()
        self.assertEqual(
            occupancy_state_from_evidence(
                [], [], [unreadable], [], require_person=True
            ),
            OccupancyState.UNKNOWN,
        )


if __name__ == "__main__":
    unittest.main()


class OccupancyEvidenceCodeTests(unittest.TestCase):
    """왜 사용중인지를 한 글자로 남기는 규칙.

    사용중은 세 갈래로 생기는데(사람이 앉았다 / 책상에 짐 / 의자에 짐) 지금까지
    로그의 reason 문자열을 읽어야만 구분이 됐다.  사진 위에 글자로 남겨야
    사람이 검수할 때 "이 빨간 상자는 무엇 때문인가"를 바로 알 수 있다.
    """

    def observation(self, **kwargs) -> TableObservation:
        base = dict(
            box=(0.0, 0.0, 100.0, 100.0),
            table_confidence=1.0,
            raw_state=OccupancyState.OCCUPIED,
            raw_score=0.9,
        )
        base.update(kwargs)
        return TableObservation(**base)

    def test_each_evidence_kind_gets_its_own_letter(self):
        cup = Detection("cup", (10.0, 10.0, 20.0, 20.0), 0.8)
        bag = Detection("backpack", (30.0, 30.0, 40.0, 40.0), 0.7)
        seated = pose_observation(PoseState.SEATED)

        self.assertEqual(
            occupancy_evidence_code(self.observation(seated_people=[seated])), "s"
        )
        self.assertEqual(
            occupancy_evidence_code(self.observation(objects=[cup])), "t"
        )
        self.assertEqual(
            occupancy_evidence_code(self.observation(chair_objects=[bag])), "c"
        )

    def test_person_on_a_linked_chair_still_counts_as_a_person(self):
        seated = pose_observation(PoseState.SEATED)
        chair = Detection("chair", (0.0, 0.0, 50.0, 80.0), 0.9)

        code = occupancy_evidence_code(
            self.observation(occupied_chairs=[chair], chair_seated_people=[seated])
        )

        self.assertEqual(code, "s")

    def test_several_kinds_are_listed_strongest_first(self):
        cup = Detection("cup", (10.0, 10.0, 20.0, 20.0), 0.8)
        bag = Detection("backpack", (30.0, 30.0, 40.0, 40.0), 0.7)
        seated = pose_observation(PoseState.SEATED)

        code = occupancy_evidence_code(
            self.observation(
                seated_people=[seated], objects=[cup], chair_objects=[bag]
            )
        )

        self.assertEqual(code, "stc")

    def test_no_evidence_has_no_letters(self):
        self.assertEqual(
            occupancy_evidence_code(
                self.observation(raw_state=OccupancyState.EMPTY, raw_score=1.0)
            ),
            "",
        )

    def test_log_line_reproduces_the_same_letters(self):
        """검수 폴더는 이미 디스크에 있는 로그로부터 다시 그려진다."""
        cup = Detection("cup", (10.0, 10.0, 20.0, 20.0), 0.8)
        bag = Detection("backpack", (30.0, 30.0, 40.0, 40.0), 0.7)
        observation = self.observation(objects=[cup], chair_objects=[bag])
        track = Track(
            track_id=1,
            box=observation.box,
            stable_state=OccupancyState.OCCUPIED,
            last_observation=observation,
            first_seen=0.0,
            last_seen=0.0,
        )

        logged = track_to_dict(track)

        self.assertEqual(logged["evidence_code"], "tc")
        self.assertEqual(evidence_code_from_log(logged), "tc")

    def test_old_log_without_the_field_falls_back_to_the_reason_string(self):
        """chair_objects 필드가 생기기 전에 돌린 실행도 다시 그릴 수 있어야 한다."""
        legacy = {
            "state": "occupied",
            "reason": "occupied_chairs:1;chair_objects:backpack",
            "objects": [],
            "seated_people": 0,
            "occupied_chairs": 1,
            "chair_seated_people": 0,
        }

        self.assertEqual(evidence_code_from_log(legacy), "c")

    def test_old_log_chair_with_a_person_reads_as_a_person(self):
        legacy = {
            "state": "occupied",
            "reason": "occupied_chairs:1",
            "objects": [],
            "seated_people": 0,
            "occupied_chairs": 1,
            "chair_seated_people": 1,
        }

        self.assertEqual(evidence_code_from_log(legacy), "s")

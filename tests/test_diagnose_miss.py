import unittest

from checks.diagnose_miss import (
    LABEL_BORDERLINE,
    LABEL_BOX,
    LABEL_FINETUNE,
    LABEL_LOGIC,
    LABEL_UNSCORED,
    Sighting,
    collect_sightings,
    diagnose,
    label_miss,
    regions_for,
    share_inside,
    wanted_evidence,
)


def detection(name, confidence, box):
    return {"class": name, "confidence": confidence, "box": box}


def miss(**kwargs):
    base = {
        "timestamp": 30.0,
        "seat": "T5",
        "truth": "occupied",
        "category": "오답",
        "direction": "놓침",
        "missed_evidence": "t",
        "imagined_evidence": "",
        "seat_box": [100.0, 100.0, 200.0, 200.0],
        "connected_chairs": [{"box": [200.0, 100.0, 260.0, 180.0]}],
        "objects": [],
        "chair_objects": [],
    }
    base.update(kwargs)
    return base


class ShareTests(unittest.TestCase):
    def test_a_box_fully_inside_is_all_of_itself(self):
        self.assertEqual(share_inside([110, 110, 130, 130], [100, 100, 200, 200]), 1.0)

    def test_a_box_half_over_the_edge(self):
        self.assertAlmostEqual(
            share_inside([190, 110, 210, 130], [100, 100, 200, 200]), 0.5
        )

    def test_no_overlap_is_zero(self):
        self.assertEqual(share_inside([300, 300, 320, 320], [100, 100, 200, 200]), 0.0)

    def test_a_zero_area_box_does_not_divide_by_zero(self):
        self.assertEqual(share_inside([10, 10, 10, 10], [0, 0, 100, 100]), 0.0)


class RegionTests(unittest.TestCase):
    def test_table_belongings_look_at_the_seat_box(self):
        regions = regions_for(miss(), "t")

        self.assertEqual([name for name, _ in regions], ["자리"])

    def test_chair_belongings_look_at_the_chairs(self):
        regions = regions_for(miss(), "c")

        self.assertEqual([name for name, _ in regions], ["의자1"])

    def test_a_seated_person_looks_at_both(self):
        regions = regions_for(miss(), "s")

        self.assertEqual([name for name, _ in regions], ["자리", "의자1"])


class CollectTests(unittest.TestCase):
    def test_keeps_only_what_actually_overlaps(self):
        detections = [
            detection("book", 0.4, [110, 110, 130, 130]),
            detection("cup", 0.3, [900, 900, 920, 920]),
        ]

        sightings = collect_sightings(detections, regions_for(miss(), "t"))

        self.assertEqual([s.name for s in sightings], ["book"])

    def test_reports_the_strongest_region_for_each_detection(self):
        detections = [detection("backpack", 0.2, [210, 110, 250, 170])]

        sightings = collect_sightings(detections, regions_for(miss(), "s"))

        self.assertEqual(sightings[0].where, "의자1")

    def test_marks_classes_the_rules_deliberately_drop(self):
        detections = [detection("chair", 0.6, [110, 110, 190, 190])]

        sightings = collect_sightings(detections, regions_for(miss(), "t"))

        self.assertIn("제외클래스", sightings[0].where)

    def test_sorted_strongest_first(self):
        detections = [
            detection("cup", 0.2, [110, 110, 130, 130]),
            detection("book", 0.7, [140, 140, 160, 160]),
        ]

        sightings = collect_sightings(detections, regions_for(miss(), "t"))

        self.assertEqual([s.name for s in sightings], ["book", "cup"])


class LabelTests(unittest.TestCase):
    def sighting(self, name, confidence, share=0.9):
        return Sighting(name, confidence, share, "자리", [0, 0, 1, 1])

    def test_a_detection_above_the_threshold_means_the_code_dropped_it(self):
        label, why, best = label_miss([self.sighting("book", 0.42)], 0.15, "t")

        self.assertEqual(label, LABEL_LOGIC)
        self.assertIn("규칙이 버렸다", why)
        self.assertEqual(best, 0.42)

    def test_a_detection_only_below_the_threshold_is_borderline(self):
        label, _, best = label_miss([self.sighting("book", 0.08)], 0.15, "t")

        self.assertEqual(label, LABEL_BORDERLINE)
        self.assertEqual(best, 0.08)

    def test_nothing_at_all_is_the_model_s_problem(self):
        label, why, best = label_miss([], 0.15, "t")

        self.assertEqual(label, LABEL_FINETUNE)
        self.assertIsNone(best)
        self.assertIn("안 보인다", why)

    def test_furniture_is_background_not_a_missed_belonging(self):
        """의자·테이블은 어느 자리 위에나 늘 보인다. 그게 손님 짐일 리 없다."""
        label, why, _ = label_miss([self.sighting("dining table", 0.8)], 0.15, "t")

        self.assertEqual(label, LABEL_FINETUNE)
        self.assertIn("dining table", why)

    def test_a_confidence_far_below_the_threshold_is_not_recoverable(self):
        """0.03 을 '임계값만 내리면 잡힌다'고 부르면 거짓말이다."""
        label, why, _ = label_miss([self.sighting("bowl", 0.03)], 0.15, "t")

        self.assertEqual(label, LABEL_FINETUNE)
        self.assertIn("유령", why)

    def test_the_borderline_band_stops_at_a_third_of_the_threshold(self):
        self.assertEqual(label_miss([self.sighting("book", 0.05)], 0.15, "t")[0],
                         LABEL_BORDERLINE)
        self.assertEqual(label_miss([self.sighting("book", 0.049)], 0.15, "t")[0],
                         LABEL_FINETUNE)

    def test_a_belonging_just_outside_the_box_is_a_drawing_problem(self):
        nearby = [Sighting("backpack", 0.6, 0.9, "자리 주변", [0, 0, 1, 1])]

        label, why, best = label_miss([], 0.15, "t", nearby)

        self.assertEqual(label, LABEL_BOX)
        self.assertIn("바로 밖", why)
        self.assertEqual(best, 0.6)

    def test_a_faint_thing_outside_the_box_does_not_blame_the_drawing(self):
        nearby = [Sighting("bowl", 0.02, 0.9, "자리 주변", [0, 0, 1, 1])]

        self.assertEqual(label_miss([], 0.15, "t", nearby)[0], LABEL_FINETUNE)

    def test_what_is_inside_the_box_wins_over_what_is_outside(self):
        inside = [self.sighting("book", 0.4)]
        nearby = [Sighting("backpack", 0.9, 0.9, "자리 주변", [0, 0, 1, 1])]

        self.assertEqual(label_miss(inside, 0.15, "t", nearby)[0], LABEL_LOGIC)

    def test_a_bag_never_stands_in_for_a_missing_person(self):
        label, _, _ = label_miss([self.sighting("backpack", 0.9)], 0.15, "s")

        self.assertEqual(label, LABEL_FINETUNE)

    def test_a_person_found_below_the_threshold_is_borderline(self):
        label, _, _ = label_miss([self.sighting("person", 0.05)], 0.15, "s")

        self.assertEqual(label, LABEL_BORDERLINE)


class WantedEvidenceTests(unittest.TestCase):
    def test_uses_what_the_answer_key_says_we_missed(self):
        self.assertEqual(wanted_evidence(miss(missed_evidence="c")), "c")

    def test_a_person_outranks_belongings_when_several_are_missed(self):
        self.assertEqual(wanted_evidence(miss(missed_evidence="tc s")), "s")

    def test_falls_back_to_table_belongings_for_a_taken_seat(self):
        self.assertEqual(wanted_evidence(miss(missed_evidence="")), "t")

    def test_an_empty_seat_with_nothing_missed_has_nothing_to_look_for(self):
        self.assertEqual(wanted_evidence(miss(missed_evidence="", truth="empty")), "")


class DiagnoseTests(unittest.TestCase):
    def test_an_object_already_assigned_to_this_seat_is_not_a_miss(self):
        """이미 센 물건을 '놓친 것'으로 다시 세면 안 된다."""
        counted = detection("book", 0.5, [110, 110, 130, 130])
        held = miss(objects=[{"class": "book", "box": [110, 110, 130, 130]}])

        results = diagnose([held], {30.0: [counted]}, 0.15)

        self.assertEqual(results[0].label, LABEL_FINETUNE)

    def test_a_different_object_of_the_same_class_still_counts_as_missed(self):
        counted = {"class": "book", "box": [110, 110, 130, 130]}
        other = detection("book", 0.5, [160, 160, 190, 190])

        results = diagnose([miss(objects=[counted])], {30.0: [other]}, 0.15)

        self.assertEqual(results[0].label, LABEL_LOGIC)

    def test_labels_a_miss_the_model_could_see(self):
        detections = {30.0: [detection("book", 0.5, [110, 110, 130, 130])]}

        results = diagnose([miss()], detections, 0.15)

        self.assertEqual(results[0].label, LABEL_LOGIC)
        self.assertEqual(results[0].seat, "T5")

    def test_labels_a_miss_the_model_cannot_see(self):
        results = diagnose([miss()], {30.0: []}, 0.15)

        self.assertEqual(results[0].label, LABEL_FINETUNE)

    def test_a_tick_with_no_photo_is_treated_as_nothing_seen(self):
        results = diagnose([miss()], {}, 0.15)

        self.assertEqual(results[0].label, LABEL_FINETUNE)

    def test_seeing_something_that_is_not_there_is_always_a_rule_problem(self):
        """헛것은 "모델이 못 봤나"를 물을 일이 아니다."""
        phantom = miss(
            truth="empty",
            missed_evidence="",
            imagined_evidence="c",
            chair_objects=[{"class": "suitcase", "confidence": 0.19}],
        )

        results = diagnose([phantom], {30.0: []}, 0.15)

        self.assertEqual(results[0].label, LABEL_LOGIC)
        self.assertIn("suitcase", results[0].why)

    def test_a_cell_with_nothing_to_look_for_is_held_back(self):
        blank = miss(truth="empty", missed_evidence="", imagined_evidence="")

        results = diagnose([blank], {30.0: []}, 0.15)

        self.assertEqual(results[0].label, LABEL_UNSCORED)

    def test_a_chair_miss_is_looked_for_on_the_chair_not_the_table(self):
        detections = {
            30.0: [
                detection("handbag", 0.5, [210, 110, 250, 170]),
                detection("book", 0.9, [110, 110, 130, 130]),
            ]
        }

        results = diagnose([miss(missed_evidence="c")], detections, 0.15)

        self.assertEqual(results[0].label, LABEL_LOGIC)
        self.assertEqual([s.name for s in results[0].sightings], ["handbag"])


if __name__ == "__main__":
    unittest.main()

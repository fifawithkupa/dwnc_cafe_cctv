"""Unit tests for manual seat layout load/validate/scale."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from seatnow_layout import (
    LayoutChair,
    LayoutError,
    LayoutTable,
    SeatLayout,
    load_layout,
    save_layout,
)

VALID = {
    "schema_version": 1,
    "source": {"video": "v.mp4", "frame_at_seconds": 0.0, "width": 1280, "height": 720},
    "tables": [
        {
            "id": 1,
            "name": "창가1",
            "box": [100.0, 200.0, 300.0, 400.0],
            "chairs": [{"id": 1, "box": [40.0, 210.0, 90.0, 390.0]}],
        },
        {"id": 2, "name": "T2", "box": [500.0, 200.0, 700.0, 400.0], "chairs": []},
    ],
}


def write_json(data) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, handle, ensure_ascii=False)
    handle.close()
    return Path(handle.name)


class LoadLayoutTests(unittest.TestCase):
    def test_loads_valid_layout(self):
        layout = load_layout(write_json(VALID))

        self.assertEqual(len(layout.tables), 2)
        self.assertEqual(layout.tables[0].name, "창가1")
        self.assertEqual(layout.tables[0].box, (100.0, 200.0, 300.0, 400.0))
        self.assertEqual(layout.tables[0].chairs[0].id, 1)

    def test_missing_file_raises_layout_error(self):
        with self.assertRaises(LayoutError):
            load_layout(Path("/nonexistent/layout.json"))

    def test_wrong_schema_version_raises(self):
        bad = dict(VALID, schema_version=99)
        with self.assertRaises(LayoutError):
            load_layout(write_json(bad))

    def test_empty_tables_raises(self):
        bad = dict(VALID, tables=[])
        with self.assertRaises(LayoutError):
            load_layout(write_json(bad))

    def test_malformed_box_raises(self):
        bad = json.loads(json.dumps(VALID))
        bad["tables"][0]["box"] = [1, 2, 3]
        with self.assertRaises(LayoutError):
            load_layout(write_json(bad))

    def test_duplicate_table_id_raises(self):
        bad = json.loads(json.dumps(VALID))
        bad["tables"][1]["id"] = 1
        with self.assertRaises(LayoutError):
            load_layout(write_json(bad))


class ScaleAndHelpersTests(unittest.TestCase):
    def test_scaled_to_same_size_is_identity(self):
        layout = load_layout(write_json(VALID))
        scaled = layout.scaled_to(1280, 720)
        self.assertEqual(scaled.tables[0].box, (100.0, 200.0, 300.0, 400.0))

    def test_scaled_to_double_resolution(self):
        layout = load_layout(write_json(VALID))
        scaled = layout.scaled_to(2560, 1440)
        self.assertEqual(scaled.tables[0].box, (200.0, 400.0, 600.0, 800.0))
        self.assertEqual(scaled.tables[0].chairs[0].box, (80.0, 420.0, 180.0, 780.0))
        # 원본은 변하지 않는다
        self.assertEqual(layout.tables[0].box, (100.0, 200.0, 300.0, 400.0))

    def test_chair_boxes_and_assignments_are_flattened_in_table_order(self):
        data = json.loads(json.dumps(VALID))
        data["tables"][1]["chairs"] = [
            {"id": 1, "box": [710.0, 210.0, 760.0, 390.0]},
            {"id": 2, "box": [460.0, 210.0, 495.0, 390.0]},
        ]
        layout = load_layout(write_json(data))

        self.assertEqual(len(layout.chair_boxes()), 3)
        self.assertEqual(layout.chair_assignments(), {0: [0], 1: [1, 2]})


class SaveLayoutTests(unittest.TestCase):
    def test_round_trip(self):
        layout = load_layout(write_json(VALID))
        out = Path(tempfile.mkdtemp()) / "saved.json"

        save_layout(layout, out)
        reloaded = load_layout(out)

        self.assertEqual(reloaded.tables[0].box, layout.tables[0].box)
        self.assertEqual(len(reloaded.tables), 2)


COUNTED_ZONE = {
    "schema_version": 2,
    "source": {"video": "v.mp4", "frame_at_seconds": 0.0, "width": 1280, "height": 720},
    "tables": [
        {
            "id": 7,
            "name": "BAR",
            "kind": "counted_zone",
            "box": [100.0, 100.0, 700.0, 300.0],
            "seats": [
                {"id": 1, "box": [100.0, 100.0, 250.0, 300.0]},
                {"id": 2, "box": [250.0, 100.0, 380.0, 300.0]},
            ],
        }
    ],
}


class CountedZoneTests(unittest.TestCase):
    def test_loads_counted_zone_with_seats(self):
        layout = load_layout(write_json(COUNTED_ZONE))

        zone = layout.tables[0]
        self.assertEqual(zone.kind, "counted_zone")
        self.assertEqual(len(zone.seats), 2)
        self.assertEqual(zone.seats[0].id, 1)
        self.assertEqual(zone.seats[1].box, (250.0, 100.0, 380.0, 300.0))

    def test_v1_layout_defaults_to_table_kind(self):
        layout = load_layout(write_json(VALID))

        self.assertEqual(layout.tables[0].kind, "table")
        self.assertEqual(layout.tables[0].seats, ())

    def test_rejects_counted_zone_without_seats(self):
        data = json.loads(json.dumps(COUNTED_ZONE))
        data["tables"][0]["seats"] = []

        with self.assertRaises(LayoutError):
            load_layout(write_json(data))

    def test_rejects_unknown_kind(self):
        data = json.loads(json.dumps(COUNTED_ZONE))
        data["tables"][0]["kind"] = "sofa"

        with self.assertRaises(LayoutError):
            load_layout(write_json(data))

    def test_rejects_seat_outside_zone_box(self):
        data = json.loads(json.dumps(COUNTED_ZONE))
        data["tables"][0]["seats"][1]["box"] = [900.0, 100.0, 950.0, 300.0]

        with self.assertRaises(LayoutError):
            load_layout(write_json(data))

    def test_rejects_seats_on_a_plain_table(self):
        data = json.loads(json.dumps(COUNTED_ZONE))
        data["tables"][0]["kind"] = "table"

        with self.assertRaises(LayoutError):
            load_layout(write_json(data))

    def test_scaled_to_scales_seats(self):
        layout = load_layout(write_json(COUNTED_ZONE)).scaled_to(2560, 1440)

        self.assertEqual(layout.tables[0].seats[0].box, (200.0, 200.0, 500.0, 600.0))

    def test_round_trip_preserves_counted_zone(self):
        layout = load_layout(write_json(COUNTED_ZONE))
        path = Path(tempfile.mkdtemp()) / "round.json"
        save_layout(layout, path)

        reloaded = load_layout(path)
        self.assertEqual(reloaded.tables[0].kind, "counted_zone")
        self.assertEqual(len(reloaded.tables[0].seats), 2)


MIXED = {
    "schema_version": 2,
    "source": {"video": "v.mp4", "frame_at_seconds": 0.0, "width": 1280, "height": 720},
    "tables": [
        {
            "id": 1,
            "name": "창가1",
            "box": [100.0, 200.0, 300.0, 400.0],
            "chairs": [
                {"id": 1, "box": [40.0, 210.0, 90.0, 390.0]},
                {"id": 2, "box": [310.0, 210.0, 360.0, 390.0]},
            ],
        },
        {
            "id": 7,
            "name": "BAR",
            "kind": "counted_zone",
            "box": [500.0, 100.0, 900.0, 300.0],
            "seats": [
                {"id": 1, "box": [500.0, 100.0, 700.0, 300.0]},
                {"id": 2, "box": [700.0, 100.0, 900.0, 300.0]},
            ],
        },
    ],
}


class JudgementUnitTests(unittest.TestCase):
    def test_table_yields_one_unit_zone_yields_one_per_seat(self):
        units = load_layout(write_json(MIXED)).judgement_units()

        self.assertEqual(len(units), 3)
        self.assertEqual(units[0].kind, "table")
        self.assertEqual(units[0].name, "창가1")
        self.assertEqual(units[0].capacity, 1)
        self.assertEqual(units[1].kind, "counted_zone")
        self.assertEqual(units[1].name, "BAR-1")
        self.assertEqual(units[1].zone_name, "BAR")
        self.assertEqual(units[1].zone_id, 7)
        self.assertEqual(units[1].seat_id, 1)
        self.assertEqual(units[1].capacity, 2)
        self.assertEqual(units[2].name, "BAR-2")

    def test_unit_ids_are_unique_and_sequential(self):
        units = load_layout(write_json(MIXED)).judgement_units()

        self.assertEqual([unit.unit_id for unit in units], [1, 2, 3])

    def test_zone_seat_boxes_become_unit_boxes(self):
        units = load_layout(write_json(MIXED)).judgement_units()

        self.assertEqual(units[1].box, (500.0, 100.0, 700.0, 300.0))
        self.assertEqual(units[2].box, (700.0, 100.0, 900.0, 300.0))

    def test_chairs_map_to_unit_indices_zones_get_none(self):
        layout = load_layout(write_json(MIXED))

        self.assertEqual(layout.unit_chair_assignments(), {0: [0, 1], 1: [], 2: []})

    def test_table_only_layout_matches_legacy_chair_assignments(self):
        layout = load_layout(write_json(VALID))

        self.assertEqual(layout.unit_chair_assignments(), layout.chair_assignments())


class UnassignedChairTests(unittest.TestCase):
    """Chairs drawn before anyone decided which table they serve.

    Step 3-c of the install draws chair boxes; step 5-b (stage 2) decides
    ownership.  Between those two the chair has to exist somewhere without
    claiming a table, because a wrong claim marks the wrong table occupied.
    """

    def _layout(self, unassigned=()):
        return SeatLayout(
            schema_version=3,
            source={"width": 1920, "height": 1080},
            tables=(
                LayoutTable(
                    id=1,
                    name="T1",
                    box=(100.0, 100.0, 300.0, 200.0),
                    chairs=(LayoutChair(id=1, box=(80.0, 120.0, 120.0, 180.0)),),
                ),
                LayoutTable(
                    id=2,
                    name="T2",
                    box=(500.0, 100.0, 700.0, 200.0),
                    chairs=(LayoutChair(id=1, box=(480.0, 120.0, 520.0, 180.0)),),
                ),
            ),
            unassigned_chairs=tuple(
                LayoutChair(id=index, box=box)
                for index, box in enumerate(unassigned, start=1)
            ),
        )

    def test_chair_boxes_includes_unassigned(self):
        layout = self._layout(unassigned=[(900.0, 900.0, 950.0, 950.0)])
        self.assertEqual(len(layout.chair_boxes()), 3)

    def test_unassigned_chairs_come_last(self):
        # unit_chair_assignments() indexes into chair_boxes(); putting an
        # unassigned chair anywhere but the end silently shifts every link.
        orphan = (900.0, 900.0, 950.0, 950.0)
        layout = self._layout(unassigned=[orphan])
        self.assertEqual(layout.chair_boxes()[-1], orphan)

    def test_unassigned_chairs_claim_no_table(self):
        layout = self._layout(unassigned=[(900.0, 900.0, 950.0, 950.0)])
        assignments = layout.unit_chair_assignments()
        linked = [index for indices in assignments.values() for index in indices]
        self.assertEqual(sorted(linked), [0, 1])

    def test_assignments_are_unchanged_by_adding_orphans(self):
        without = self._layout().unit_chair_assignments()
        with_orphan = self._layout(unassigned=[(900.0, 900.0, 950.0, 950.0)])
        self.assertEqual(with_orphan.unit_chair_assignments(), without)

    def test_scaled_to_scales_unassigned_chairs(self):
        layout = self._layout(unassigned=[(960.0, 540.0, 1000.0, 580.0)])
        scaled = layout.scaled_to(960, 540)
        self.assertEqual(scaled.unassigned_chairs[0].box, (480.0, 270.0, 500.0, 290.0))


class SchemaV3RoundTripTests(unittest.TestCase):
    def test_v3_file_round_trips_unassigned_chairs(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "layout.json"
            original = SeatLayout(
                schema_version=3,
                source={"width": 1920, "height": 1080},
                tables=(
                    LayoutTable(id=1, name="T1", box=(10.0, 10.0, 50.0, 50.0)),
                ),
                unassigned_chairs=(
                    LayoutChair(id=7, box=(60.0, 60.0, 80.0, 80.0)),
                ),
            )
            save_layout(original, path)
            loaded = load_layout(path)
            self.assertEqual(len(loaded.unassigned_chairs), 1)
            self.assertEqual(loaded.unassigned_chairs[0].id, 7)
            self.assertEqual(loaded.unassigned_chairs[0].box, (60.0, 60.0, 80.0, 80.0))

    def test_v2_file_still_loads_with_no_unassigned_chairs(self):
        # layouts/cafe_angle1.json on disk is v2; refusing it would throw away
        # work already done at the cafe.
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "old.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "source": {"width": 1920, "height": 1080},
                        "tables": [
                            {"id": 1, "name": "T1", "kind": "table",
                             "box": [10, 10, 50, 50], "chairs": [], "seats": []}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_layout(path)
            self.assertEqual(loaded.unassigned_chairs, ())


if __name__ == "__main__":
    unittest.main()

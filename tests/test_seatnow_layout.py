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
        bad = dict(VALID, schema_version=2)
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


if __name__ == "__main__":
    unittest.main()

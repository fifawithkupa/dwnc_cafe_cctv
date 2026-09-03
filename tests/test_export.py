"""Export naming rules — the model the engine runs must not be a fixed size.

One tick asks the detect model for two sizes (the whole frame at --imgsz, the
table crops at --crop-imgsz).  A static export answers both at whatever size it
was exported with, without saying so, so these tests pin the two behaviours
that keep such a model out of a deployment: the deployable export is dynamic,
and a fixed-size export carries its size in its name.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from edge.export import build_parser, export_destination, main


class ExportDestinationTests(unittest.TestCase):
    WEIGHTS = Path("/models/yolov8n.pt")

    def test_dynamic_export_keeps_the_name_the_engine_loads(self):
        self.assertEqual(
            export_destination(self.WEIGHTS, 1280, int8=False, dynamic=True).name,
            "yolov8n_openvino_model",
        )

    def test_fixed_size_export_says_its_size(self):
        self.assertEqual(
            export_destination(self.WEIGHTS, 640, int8=False, dynamic=False).name,
            "yolov8n_640_openvino_model",
        )

    def test_fixed_sizes_do_not_overwrite_each_other(self):
        names = {
            export_destination(self.WEIGHTS, size, int8=False, dynamic=False).name
            for size in (640, 960, 1280)
        }
        self.assertEqual(len(names), 3)

    def test_int8_stays_separable_from_fp32(self):
        self.assertNotEqual(
            export_destination(self.WEIGHTS, 1280, int8=True, dynamic=True),
            export_destination(self.WEIGHTS, 1280, int8=False, dynamic=True),
        )


class ExportParserTests(unittest.TestCase):
    def test_dynamic_is_the_default(self):
        args = build_parser().parse_args([])
        self.assertTrue(args.dynamic)
        self.assertEqual(args.imgsz, [1280])

    def test_static_is_opt_in(self):
        self.assertFalse(build_parser().parse_args(["--static"]).dynamic)

    def test_several_sizes_without_static_is_refused_not_silently_collapsed(self):
        """The old behaviour exported 640 and skipped 960/1280 as 'exists'."""
        with self.assertRaises(SystemExit):
            main(["--imgsz", "640", "960", "1280"])


if __name__ == "__main__":
    unittest.main()

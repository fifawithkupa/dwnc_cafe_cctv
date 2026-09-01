"""Unit tests for the export/bench tooling's model-independent logic."""

from __future__ import annotations

import unittest
from pathlib import Path

from edge import bench
from edge.bench import Measurement, resolve_model, tick_budget
from engine.seatnow_core import model_backend


def measurement(task: str, imgsz: int, backend: str, median_ms: float) -> Measurement:
    return Measurement(
        task=task,
        imgsz=imgsz,
        backend=backend,
        model=f"{task}-{backend}",
        median_ms=median_ms,
        p95_ms=median_ms * 1.2,
        minimum_ms=median_ms * 0.9,
        iterations=10,
    )


class ModelBackendTests(unittest.TestCase):
    def test_pytorch_checkpoint(self):
        self.assertEqual(model_backend(Path("yolov8n.pt")), "pytorch")

    def test_openvino_directory_name(self):
        self.assertEqual(
            model_backend(Path("yolov8n_openvino_model")), "openvino"
        )
        self.assertEqual(
            model_backend(Path("yolov8n_int8_openvino_model")), "openvino"
        )

    def test_openvino_ir_xml(self):
        self.assertEqual(model_backend(Path("yolov8n.xml")), "openvino")

    def test_unknown_extension(self):
        self.assertEqual(model_backend(Path("weights.bin")), "unknown")


class ResolveModelTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.weights = self.root / "yolov8n.pt"
        self.weights.write_bytes(b"stub")

    def test_missing_export_is_none_not_an_error(self):
        self.assertIsNone(resolve_model(self.weights, "ov-int8"))

    def test_exported_directory_is_found(self):
        (self.root / "yolov8n_openvino_model").mkdir()
        self.assertEqual(
            resolve_model(self.weights, "ov-fp32"),
            self.root / "yolov8n_openvino_model",
        )

    def test_pt_backend_resolves_to_the_checkpoint(self):
        self.assertEqual(resolve_model(self.weights, "pt"), self.weights)


class TickBudgetTests(unittest.TestCase):
    """plan.md T6: 50% of the tick passes, 100% is conditional, above fails."""

    RESULTS = [
        measurement("detect", 1280, "pt", 500.0),
        measurement("detect", 960, "pt", 300.0),
        measurement("detect", 640, "pt", 150.0),
        measurement("pose", 960, "pt", 250.0),
        measurement("pose", 640, "pt", 120.0),
    ]

    def test_counts_every_inference_a_tick_performs(self):
        budget = tick_budget(
            self.RESULTS,
            backend="pt",
            imgsz=1280,
            pose_imgsz=960,
            crop_imgsz=960,
            max_crops=4,
            median_frames=2,
            sample_seconds=15.0,
        )

        # plan.md's worked example: 5 frames x (detect + pose + 4 crops).
        self.assertEqual(budget["frames_per_tick"], 5)
        self.assertEqual(budget["inferences_per_tick"], 30)
        # 500 + 250 + 4*300 = 1950 ms per frame, x5 = 9.75 s.
        self.assertAlmostEqual(budget["per_frame_ms"], 1950.0)
        self.assertAlmostEqual(budget["tick_seconds"], 9.75)

    def test_between_half_and_a_full_tick_is_conditional(self):
        budget = tick_budget(
            self.RESULTS, "pt", 1280, 960, 960, 4, 2, sample_seconds=15.0
        )
        self.assertEqual(budget["grade"], "CONDITIONAL")

    def test_within_half_a_tick_passes(self):
        budget = tick_budget(
            self.RESULTS, "pt", 640, 640, 640, 0, 2, sample_seconds=15.0
        )
        # (150 + 120) * 5 = 1.35 s out of 15 s.
        self.assertAlmostEqual(budget["tick_seconds"], 1.35)
        self.assertEqual(budget["grade"], "PASS")

    def test_over_the_tick_fails(self):
        budget = tick_budget(
            self.RESULTS, "pt", 1280, 960, 960, 4, 2, sample_seconds=5.0
        )
        self.assertEqual(budget["grade"], "FAIL")
        self.assertGreater(budget["tick_utilization"], 1.0)

    def test_single_frame_tick_drops_the_burst_multiplier(self):
        budget = tick_budget(
            self.RESULTS, "pt", 640, 640, 640, 0, 0, sample_seconds=15.0
        )
        self.assertEqual(budget["frames_per_tick"], 1)
        self.assertEqual(budget["inferences_per_tick"], 2)

    def test_unmeasured_combination_returns_none(self):
        self.assertIsNone(
            tick_budget(self.RESULTS, "ov-int8", 1280, 960, 960, 4, 2, 15.0)
        )

    def test_profiles_cover_the_cli_defaults(self):
        """The accuracy profile must match seatnow.py's shipped defaults."""
        self.assertEqual(bench.PROFILES["accuracy_default"], (1280, 960, 960, 4, 2))


class RtspPublisherCommandTests(unittest.TestCase):
    """T5: the harness must behave like a camera, not like a file."""

    SOURCE = Path("sample_raw/cafe.mp4")
    URL = "rtsp://127.0.0.1:8554/seatnow"

    def _command(self, **kwargs):
        from edge.rtsp_republish import build_publisher_command

        options = dict(loop=True, transcode=False, fps=None)
        options.update(kwargs)
        return build_publisher_command("ffmpeg", self.SOURCE, self.URL, **options)

    def test_publishes_at_wall_clock_speed(self):
        """Without -re the stream races ahead and hides seek-based bugs."""
        self.assertIn("-re", self._command())

    def test_loops_by_default_and_can_be_turned_off(self):
        self.assertIn("-stream_loop", self._command())
        self.assertNotIn("-stream_loop", self._command(loop=False))

    def test_stream_copies_unless_asked_to_transcode(self):
        self.assertIn("copy", self._command())
        self.assertIn("libx264", self._command(transcode=True))

    def test_forcing_fps_implies_encoding(self):
        command = self._command(fps=5.0)
        self.assertIn("libx264", command)
        self.assertEqual(command[command.index("-r") + 1], "5.0")

    def test_transcode_pins_a_regular_keyframe_interval(self):
        command = self._command(transcode=True)
        self.assertEqual(command[command.index("-g") + 1], "30")

    def test_publishes_over_tcp_to_the_given_url(self):
        command = self._command()
        self.assertEqual(command[-1], self.URL)
        self.assertEqual(command[command.index("-f") + 1], "rtsp")
        self.assertEqual(command[command.index("-rtsp_transport") + 1], "tcp")


if __name__ == "__main__":
    unittest.main()

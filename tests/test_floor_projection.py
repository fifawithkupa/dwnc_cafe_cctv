"""Unit tests for flattening the camera view onto the floor plane.

No image and no model: a homography is decided by four points, so the tests
give it four points whose answer is known by hand.
"""

from __future__ import annotations

import math
import unittest

from install.floor_projection import (
    FLOOR_UNIT,
    FloorProjectionError,
    build_transform,
    floor_anchor,
)


FRAME = (1920, 1080)
# A square already square in the image: the transform must be a plain scale.
SQUARE = ((100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0))
# A trapezoid, which is what a real floor rectangle looks like from a camera.
TRAPEZOID = ((700.0, 900.0), (900.0, 600.0), (1300.0, 600.0), (1600.0, 900.0))


class FloorAnchorTests(unittest.TestCase):
    def test_anchor_is_the_bottom_edge_centre(self):
        # Not the box centre: the bottom edge is where the furniture meets
        # the floor, and the floor is the plane the homography flattens.
        self.assertEqual(floor_anchor((100.0, 200.0, 300.0, 400.0)), (200.0, 400.0))

    def test_anchor_of_a_flat_box(self):
        self.assertEqual(floor_anchor((10.0, 50.0, 30.0, 50.0)), (20.0, 50.0))


class BuildTransformTests(unittest.TestCase):
    def test_square_maps_to_the_unit_square(self):
        transform = build_transform(SQUARE, FRAME)
        for image_point, floor_point in (
            ((100.0, 100.0), (0.0, 0.0)),
            ((500.0, 100.0), (FLOOR_UNIT, 0.0)),
            ((500.0, 500.0), (FLOOR_UNIT, FLOOR_UNIT)),
            ((100.0, 500.0), (0.0, FLOOR_UNIT)),
        ):
            got = transform.project(image_point)
            self.assertAlmostEqual(got[0], floor_point[0], places=4)
            self.assertAlmostEqual(got[1], floor_point[1], places=4)

    def test_square_centre_maps_to_the_centre(self):
        transform = build_transform(SQUARE, FRAME)
        x, y = transform.project((300.0, 300.0))
        self.assertAlmostEqual(x, FLOOR_UNIT / 2, places=4)
        self.assertAlmostEqual(y, FLOOR_UNIT / 2, places=4)

    def test_trapezoid_far_edge_stretches(self):
        # The far edge measures 400px in the image and the near edge 900px,
        # yet both are the same length on the floor.  Undoing that squash is
        # the whole reason for doing this.
        transform = build_transform(TRAPEZOID, FRAME)

        def floor_length(a, b):
            ax, ay = transform.project(a)
            bx, by = transform.project(b)
            return math.hypot(bx - ax, by - ay)

        near = floor_length((700.0, 900.0), (1600.0, 900.0))
        far = floor_length((900.0, 600.0), (1300.0, 600.0))
        self.assertAlmostEqual(near, FLOOR_UNIT, places=4)
        self.assertAlmostEqual(far, FLOOR_UNIT, places=4)

    def test_collinear_points_are_rejected(self):
        line = ((100.0, 100.0), (300.0, 100.0), (500.0, 100.0), (700.0, 100.0))
        with self.assertRaises(FloorProjectionError):
            build_transform(line, FRAME)

    def test_tiny_quad_is_rejected(self):
        # Four points inside a 20px box cannot pin down a transform for a
        # whole room; the error would grow without bound away from them.
        tiny = ((100.0, 100.0), (120.0, 100.0), (120.0, 120.0), (100.0, 120.0))
        with self.assertRaises(FloorProjectionError):
            build_transform(tiny, FRAME)

    def test_self_crossing_quad_is_rejected(self):
        # Clicked out of order: 1-3-2-4 makes a bow tie, and the "rectangle"
        # it claims to be does not exist.
        bowtie = ((100.0, 100.0), (500.0, 500.0), (500.0, 100.0), (100.0, 500.0))
        with self.assertRaises(FloorProjectionError):
            build_transform(bowtie, FRAME)

    def test_wrong_point_count_is_rejected(self):
        with self.assertRaises(FloorProjectionError):
            build_transform(SQUARE[:3], FRAME)


class ProjectTests(unittest.TestCase):
    def test_point_on_the_vanishing_line_returns_none(self):
        # On the vanishing line the homogeneous divisor is zero and the answer
        # is not a place on the floor at all.  Solve for that line from the
        # matrix rather than guessing where it falls for one trapezoid.
        transform = build_transform(TRAPEZOID, FRAME)
        _, _, row = transform.matrix
        self.assertNotEqual(row[1], 0.0)
        y = -row[2] / row[1]
        self.assertIsNone(transform.project((0.0, y)))

    def test_a_point_off_the_vanishing_line_returns_a_place(self):
        transform = build_transform(TRAPEZOID, FRAME)
        self.assertIsNotNone(transform.project((1000.0, 800.0)))


if __name__ == "__main__":
    unittest.main()

"""Flatten the camera view onto the floor plane.

A cafe camera looks down at an angle, so far things are small and squashed.
Given four image points that are a rectangle on the real floor, one
homography undoes that everywhere -- which is what turns a warped camera
view into a map a customer can read.

Only the floor is flattened.  A homography straightens exactly one plane,
and a tabletop is 70cm above the floor: pushing its corners through a floor
transform lands them away from the camera, further than the table really is.
So every box contributes a single point, the middle of its bottom edge,
where the furniture meets the floor.

The result is a draft.  A detected table box is the tabletop, not the legs,
so its bottom edge sits above the real contact point -- which is why a
person edits the map afterwards (see the stage 2 design doc).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


Box = Tuple[float, float, float, float]
Point = Tuple[float, float]

# The floor rectangle becomes a square of this size.  Its real aspect ratio
# is unknown -- we deliberately never ask anyone to measure the room -- and a
# person fixes the proportions on the map.
FLOOR_UNIT = 200.0

# A reference quad smaller than this share of the frame cannot pin down a
# transform for a whole room: the error grows without bound away from it.
MIN_AREA_FRACTION = 0.005

REQUIRED_POINTS = 4


class FloorProjectionError(Exception):
    """The four points cannot define a usable floor transform."""


def floor_anchor(box: Box) -> Point:
    """The middle of a box's bottom edge -- where it meets the floor."""
    return ((box[0] + box[2]) / 2.0, box[3])


def _signed_area(points: Sequence[Point]) -> float:
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points, tuple(points[1:]) + (points[0],)):
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _is_convex(points: Sequence[Point]) -> bool:
    """True when the quad does not cross itself.

    Clicking the corners out of order makes a bow tie, and the rectangle it
    claims to be does not exist -- the transform would be meaningless rather
    than merely inaccurate.
    """
    signs = []
    count = len(points)
    for index in range(count):
        ax, ay = points[index]
        bx, by = points[(index + 1) % count]
        cx, cy = points[(index + 2) % count]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if cross == 0:
            continue
        signs.append(cross > 0)
    return len(set(signs)) <= 1


@dataclass(frozen=True)
class FloorTransform:
    """A homography from camera image coordinates onto the floor plane."""

    matrix: Tuple[Tuple[float, float, float], ...]

    def project(self, point: Point) -> Optional[Point]:
        """Where this image point lands on the floor, or None if nowhere.

        Past the vanishing line the homogeneous divisor crosses zero and the
        answer is not a place on the floor at all.  Returning None says so
        instead of handing back a number that looks like a position.
        """
        matrix = self.matrix
        x, y = point
        w = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
        if abs(w) < 1e-9:
            return None
        return (
            (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / w,
            (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / w,
        )


def build_transform(
    image_points: Sequence[Point], frame_size: Tuple[int, int]
) -> FloorTransform:
    """Fit the transform that turns the clicked floor rectangle into a square."""
    if len(image_points) != REQUIRED_POINTS:
        raise FloorProjectionError(
            f"바닥 기준점은 {REQUIRED_POINTS}개여야 합니다 (지금 {len(image_points)}개)"
        )
    if not _is_convex(image_points):
        raise FloorProjectionError(
            "바닥 네 점이 스스로 꼬였습니다 — 시계방향(또는 반시계방향) 순서로 "
            "다시 찍으세요"
        )
    width, height = frame_size
    area = abs(_signed_area(image_points))
    if area < MIN_AREA_FRACTION * width * height:
        raise FloorProjectionError(
            "바닥 네 점이 너무 좁은 영역만 덮습니다 — 방에서 더 넓게 벌어진 "
            "직사각형(바닥 타일, 큰 테이블 다리, 방 모서리)으로 다시 찍으세요"
        )

    source = np.float32([list(point) for point in image_points])
    target = np.float32(
        [[0.0, 0.0], [FLOOR_UNIT, 0.0], [FLOOR_UNIT, FLOOR_UNIT], [0.0, FLOOR_UNIT]]
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    if not np.all(np.isfinite(matrix)):
        raise FloorProjectionError("바닥 네 점으로 변환을 만들 수 없습니다")
    return FloorTransform(
        matrix=tuple(tuple(float(value) for value in row) for row in matrix)
    )

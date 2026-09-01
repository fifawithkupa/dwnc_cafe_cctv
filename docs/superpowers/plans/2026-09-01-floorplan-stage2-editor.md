# 2D 평면도 2단계 — 투영 + `floorplan.json` + 브라우저 편집기

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 바닥 네 점으로 카메라 화면을 펴서 평면도 초안을 만들고, 브라우저에서 위치를 다듬고 **의자 소속을 바로잡아** 두 파일을 함께 저장한다.

**Architecture:** 순수 계산(`floor_projection.py`) → 데이터 모델과 초안 생성(`floorplan.py`) → 로컬 HTTP 서버(`floorplan_editor.py`)와 페이지(`static/floorplan_editor.html`)로 나눈다. 편집기는 저장할 때 **평면도(`floorplan.json`)와 레이아웃(`layouts/*.json`)을 둘 다** 갱신한다 — 의자 소속은 레이아웃에 있어야 판정이 읽는다.

**Tech Stack:** Python 3.11 / OpenCV(호모그래피) / `http.server` / 브라우저 SVG + 순수 JS / `unittest`

**설계 문서:** `docs/superpowers/specs/2026-09-01-2d-floorplan-design.md` (§4 좌표 변환, §5 데이터 모델, §6-5 편집기, §7 의자 소속, §9 손님 화면, §10 오류 처리)

## Global Constraints

- 테스트는 `unittest`. 실행은 `./venv/Scripts/python.exe -m unittest discover tests -p "test_<이름>.py"` (Windows) / `./venv/bin/python ...` (Linux). **`-m unittest tests.X` 형태는 `tests/__init__.py`가 없어 동작하지 않는다**
- 테스트는 **모델도 브라우저도 부르지 않는다.** 서버는 함수 단위로 검사하고 실제 포트를 열지 않는다
- **판정 로직(`seatnow_core.py`)을 건드리지 않는다.** 2단계는 지도를 만들 뿐이고, 판정에 영향을 주는 것은 **의자 소속 하나뿐**이다
- **좌표 단위는 미터가 아니라 그림용 임의 단위**다. 실측 치수를 사람에게 요구하지 않는다 (설계 §2)
- **저장은 두 파일이 다 성공해야 한다.** 임시 파일에 쓰고 둘 다 되면 바꾼다 (설계 §10)
- 새 모듈은 `from __future__ import annotations`로 시작하고 **"왜 이게 있는가"를 설명하는 모듈 docstring**을 단다
- 커밋 메시지는 한국어, `feat:` / `fix:` / `docs:` 접두어

## 파일 구조

| 파일 | 책임 |
|---|---|
| `floor_projection.py` | 호모그래피 하나. 네 점 → 변환, 상자 → 바닥 접점, 퇴화 거부 |
| `floorplan.py` | `floorplan.json`의 자료형, 레이아웃에서 초안 만들기, 읽기/쓰기 |
| `floorplan_editor.py` | 로컬 서버. 상태 주기, 저장 받기(두 파일 원자적 갱신) |
| `static/floorplan_editor.html` | 편집 화면. 끌기, 소속 정하기, 지형지물, 미리보기 |

---

### Task 1: `floor_projection.py` — 바닥 평면으로 펴는 변환

**Files:**
- Create: `floor_projection.py`
- Test: `tests/test_floor_projection.py`

**Interfaces:**
- Consumes: `seatnow_layout.FloorReference`, `FLOOR_REFERENCE_POINTS`
- Produces:
  - `FloorProjectionError(Exception)`
  - `FLOOR_UNIT = 200.0`, `MIN_AREA_FRACTION = 0.005`
  - `floor_anchor(box: Box) -> Tuple[float, float]` — 상자 아래쪽 변 중앙
  - `build_transform(image_points, frame_size) -> FloorTransform` — 퇴화면 `FloorProjectionError`
  - `FloorTransform.project(point) -> Optional[Tuple[float, float]]` — 발산하면 `None`

**배경:** 호모그래피는 **하나의 평면**을 펴는 변환이다. 바닥으로 맞춘 변환에 바닥이 아닌 점을 넣으면 틀린다 — 테이블 상판은 바닥에서 70cm 위다. 그래서 상자마다 **아래쪽 변 중앙 한 점**만 쓴다 (설계 §4).

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_floor_projection.py`:

```python
"""Unit tests for flattening the camera view onto the floor plane.

No image and no model: a homography is decided by four points, so the tests
give it four points whose answer is known by hand.
"""

from __future__ import annotations

import unittest

from floor_projection import (
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
        self.assertEqual(transform.project((100.0, 100.0)), (0.0, 0.0))
        self.assertEqual(transform.project((500.0, 100.0)), (FLOOR_UNIT, 0.0))
        self.assertEqual(transform.project((500.0, 500.0)), (FLOOR_UNIT, FLOOR_UNIT))
        self.assertEqual(transform.project((100.0, 500.0)), (0.0, FLOOR_UNIT))

    def test_square_centre_maps_to_the_centre(self):
        transform = build_transform(SQUARE, FRAME)
        x, y = transform.project((300.0, 300.0))
        self.assertAlmostEqual(x, FLOOR_UNIT / 2, places=4)
        self.assertAlmostEqual(y, FLOOR_UNIT / 2, places=4)

    def test_trapezoid_far_edge_stretches(self):
        # The far edge is short in the image and full width on the floor:
        # that stretch is the whole reason for doing this.
        transform = build_transform(TRAPEZOID, FRAME)
        near_left = transform.project((700.0, 900.0))
        far_left = transform.project((900.0, 600.0))
        self.assertAlmostEqual(near_left[0], 0.0, places=4)
        self.assertAlmostEqual(far_left[0], 0.0, places=4)
        self.assertGreater(far_left[1], near_left[1] + FLOOR_UNIT / 2)

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
    def test_point_behind_the_horizon_returns_none(self):
        # Beyond the vanishing line the homogeneous w crosses zero and the
        # answer is not a place on the floor at all.
        transform = build_transform(TRAPEZOID, FRAME)
        results = [transform.project((x, 0.0)) for x in range(0, 1920, 120)]
        self.assertTrue(any(result is None for result in results))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_floor_projection.py" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'floor_projection'`

- [ ] **Step 3: 구현한다**

`floor_projection.py`:

```python
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
```

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_floor_projection.py" -v`
Expected: PASS — 11개

- [ ] **Step 5: 커밋**

```bash
git add floor_projection.py tests/test_floor_projection.py
git commit -m "feat: floor_projection.py - 카메라 화면을 바닥 평면으로 편다

호모그래피는 평면 하나만 편다. 테이블 상판은 바닥에서 70cm 위라
상판 귀퉁이를 바닥 변환에 넣으면 카메라 반대쪽으로 밀린다. 그래서
상자마다 아래쪽 변 중앙 한 점만 쓴다.

꼬인 사각형과 너무 좁은 사각형은 거부한다. 꼬인 것은 부정확한 게
아니라 주장하는 직사각형이 존재하지 않는 것이고, 좁은 것은 그
바깥에서 오차가 끝없이 커진다."
```

---

### Task 2: `floorplan.py` — 평면도 자료형과 초안

**Files:**
- Create: `floorplan.py`
- Test: `tests/test_floorplan.py`

**Interfaces:**
- Consumes: Task 1의 `build_transform`, `floor_anchor`, `FloorProjectionError`; `seatnow_layout.SeatLayout`
- Produces:
  - `FLOORPLAN_SCHEMA_VERSION = 1`, `EXTENT_LONG_SIDE = 1000.0`, `MARGIN_FRACTION = 0.08`
  - `DEFAULT_SIZES = {"table": (90.0, 60.0), "counted_zone": (44.0, 44.0), "chair": (30.0, 30.0)}`
  - `@dataclass FloorSeat`: `seat_id, kind, x, y, w, h, image_anchor, needs_review`
  - `@dataclass FloorChair`: `seat_id: Optional[str], x, y, w, h, image_anchor, needs_review`
  - `@dataclass Landmark`: `kind, label, x, y, w, h`
  - `@dataclass FloorPlan`: `schema_version, extent, seats, chairs, landmarks`
  - `build_draft(layout: SeatLayout) -> FloorPlan`
  - `save_floorplan(plan, path)`, `load_floorplan(path) -> FloorPlan`

**핵심:** `image_anchor`가 **의자의 신원**이다. 편집기가 의자를 다른 테이블로 옮기면 레이아웃 안의 목록이 바뀌어 인덱스가 밀리지만, 화면 좌표 상자는 안 바뀐다. 그래서 두 파일을 잇는 열쇠로 쓴다. 4단계가 쓸 대응쌍이기도 하다 (설계 §5).

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_floorplan.py`:

```python
"""Unit tests for the floor plan draft and its file format."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from floorplan import (
    EXTENT_LONG_SIDE,
    FloorPlan,
    Landmark,
    build_draft,
    load_floorplan,
    save_floorplan,
)
from floor_projection import FloorProjectionError
from seatnow_layout import (
    FloorReference,
    LayoutChair,
    LayoutSeat,
    LayoutTable,
    SeatLayout,
)


REFERENCE = FloorReference(
    image_points=((700.0, 900.0), (900.0, 600.0), (1300.0, 600.0), (1600.0, 900.0))
)


def layout(with_reference=True, unassigned=(), zone_seats=2):
    return SeatLayout(
        schema_version=3,
        source={"width": 1920, "height": 1080},
        tables=(
            LayoutTable(
                id=1,
                name="T1",
                box=(800.0, 700.0, 1000.0, 850.0),
                chairs=(LayoutChair(id=1, box=(780.0, 800.0, 830.0, 880.0)),),
            ),
            LayoutTable(
                id=7,
                name="BAR7",
                box=(1100.0, 620.0, 1500.0, 760.0),
                kind="counted_zone",
                seats=tuple(
                    LayoutSeat(id=index, box=(1100.0 + 150 * index, 620.0,
                                              1200.0 + 150 * index, 760.0))
                    for index in range(1, zone_seats + 1)
                ),
            ),
        ),
        unassigned_chairs=tuple(
            LayoutChair(id=index, box=box)
            for index, box in enumerate(unassigned, start=1)
        ),
        floor_reference=REFERENCE if with_reference else None,
    )


class BuildDraftTests(unittest.TestCase):
    def test_every_judgement_unit_becomes_a_seat(self):
        plan = build_draft(layout())
        self.assertEqual(
            [seat.seat_id for seat in plan.seats], ["T1", "BAR7-1", "BAR7-2"]
        )

    def test_seat_kind_is_carried_over(self):
        plan = build_draft(layout())
        kinds = {seat.seat_id: seat.kind for seat in plan.seats}
        self.assertEqual(kinds["T1"], "table")
        self.assertEqual(kinds["BAR7-1"], "counted_zone")

    def test_chairs_carry_their_owner(self):
        plan = build_draft(layout(unassigned=[(400.0, 800.0, 450.0, 880.0)]))
        owners = sorted(
            (chair.seat_id or "") for chair in plan.chairs
        )
        self.assertEqual(owners, ["", "T1"])

    def test_image_anchor_is_the_bottom_edge_centre(self):
        plan = build_draft(layout())
        seat = next(seat for seat in plan.seats if seat.seat_id == "T1")
        self.assertEqual(seat.image_anchor, (900.0, 850.0))

    def test_extent_long_side_is_normalised(self):
        plan = build_draft(layout())
        self.assertAlmostEqual(
            max(plan.extent), EXTENT_LONG_SIDE, places=4
        )

    def test_everything_lands_inside_the_extent(self):
        plan = build_draft(layout())
        width, height = plan.extent
        for seat in plan.seats:
            self.assertGreaterEqual(seat.x, 0.0)
            self.assertLessEqual(seat.x, width)
            self.assertGreaterEqual(seat.y, 0.0)
            self.assertLessEqual(seat.y, height)

    def test_no_floor_reference_is_refused(self):
        # Silently drawing an empty map would look like "this cafe has no
        # seats" rather than "nobody clicked the floor points yet".
        with self.assertRaises(FloorProjectionError):
            build_draft(layout(with_reference=False))

    def test_landmarks_start_empty(self):
        self.assertEqual(build_draft(layout()).landmarks, ())


class RoundTripTests(unittest.TestCase):
    def test_saved_plan_loads_back_identical(self):
        plan = build_draft(layout(unassigned=[(400.0, 800.0, 450.0, 880.0)]))
        plan = FloorPlan(
            schema_version=plan.schema_version,
            extent=plan.extent,
            seats=plan.seats,
            chairs=plan.chairs,
            landmarks=(Landmark(kind="entrance", label="입구",
                                x=10.0, y=20.0, w=30.0, h=40.0),),
        )
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "floorplan.json"
            save_floorplan(plan, path)
            self.assertEqual(load_floorplan(path), plan)

    def test_file_is_readable_json_with_the_documented_keys(self):
        plan = build_draft(layout())
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "floorplan.json"
            save_floorplan(plan, path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                sorted(data), ["chairs", "extent", "landmarks", "schema_version", "seats"]
            )
            self.assertIn("image_anchor", data["seats"][0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_floorplan.py" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'floorplan'`

- [ ] **Step 3: 구현한다**

`floorplan.py`:

```python
"""The customer-facing map: where each seat is, flattened onto the floor.

`seat_report` says what state a seat is in and nothing about where it is, so
an app can print "3 free" but cannot show which three.  This module makes the
missing half: a static map drawn once at install, joined to the live states
by ``seat_id``.

Positions here are drawing units, not metres.  Nobody is asked to measure the
room -- a customer needs "the window seat on the right", not "3.2 m from the
door" -- and the proportions get corrected by hand in the editor.

``image_anchor`` is kept on every seat and chair.  It is the camera-image
point the position was projected from, so it survives edits and identifies a
chair even after the editor moves it between tables (the layout's own chair
ordering does not).  It is also half of a correspondence pair: once a person
has corrected the map, "here in the image, here on the floor" is known for
every seat, which is a far better fit than the four clicked points.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from floor_projection import FloorProjectionError, build_transform, floor_anchor
from seatnow_layout import COUNTED_ZONE_KIND, SeatLayout


FLOORPLAN_SCHEMA_VERSION = 1
EXTENT_LONG_SIDE = 1000.0
MARGIN_FRACTION = 0.08

# Real sizes are unknown, so everything starts at a readable default and the
# person adjusts.  A bar seat is drawn smaller than a table because it is one
# seat rather than a whole table.
DEFAULT_SIZES: Dict[str, Tuple[float, float]] = {
    "table": (90.0, 60.0),
    COUNTED_ZONE_KIND: (44.0, 44.0),
    "chair": (30.0, 30.0),
}

Point = Tuple[float, float]


@dataclass(frozen=True)
class FloorSeat:
    seat_id: str
    kind: str
    x: float
    y: float
    w: float
    h: float
    image_anchor: Point
    needs_review: bool = False


@dataclass(frozen=True)
class FloorChair:
    seat_id: Optional[str]
    x: float
    y: float
    w: float
    h: float
    image_anchor: Point
    needs_review: bool = False


@dataclass(frozen=True)
class Landmark:
    kind: str
    label: str
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class FloorPlan:
    schema_version: int
    extent: Tuple[float, float]
    seats: Tuple[FloorSeat, ...]
    chairs: Tuple[FloorChair, ...]
    landmarks: Tuple[Landmark, ...] = ()


def _owner_of_each_chair(layout: SeatLayout) -> List[Tuple[Optional[str], Point]]:
    """(owner seat name or None, image anchor) for every chair in the layout."""
    owners: List[Tuple[Optional[str], Point]] = []
    for table in layout.tables:
        # A counted_zone's chair belongs to the seat slot it covers, and that
        # is decided in unit_chair_assignments; on the map it is enough to
        # hang it under the zone's first slot so it draws in the right place.
        owner = (
            f"{table.name}-{table.seats[0].id}"
            if table.kind == COUNTED_ZONE_KIND and table.seats
            else table.name
        )
        for chair in table.chairs:
            owners.append((owner, floor_anchor(chair.box)))
    for chair in layout.unassigned_chairs:
        owners.append((None, floor_anchor(chair.box)))
    return owners


def build_draft(layout: SeatLayout) -> FloorPlan:
    """Project every seat and chair onto the floor and fit them to a canvas."""
    if layout.floor_reference is None:
        raise FloorProjectionError(
            "바닥 기준점이 없어 평면도를 만들 수 없습니다 — calibrate.py 에서 "
            "[f]로 바닥의 직사각형 네 귀퉁이를 찍고 저장하세요"
        )
    frame_size = (
        int(layout.source.get("width", 1920)),
        int(layout.source.get("height", 1080)),
    )
    transform = build_transform(layout.floor_reference.image_points, frame_size)

    units = layout.judgement_units()
    seat_anchors = [(unit, floor_anchor(unit.box)) for unit in units]
    chair_owners = _owner_of_each_chair(layout)

    projected: List[Tuple[Optional[Point], Point]] = [
        (transform.project(anchor), anchor) for _, anchor in seat_anchors
    ] + [(transform.project(anchor), anchor) for _, anchor in chair_owners]

    placed = [point for point, _ in projected if point is not None]
    if not placed:
        raise FloorProjectionError(
            "좌석이 하나도 바닥 평면 위로 오지 않았습니다 — 바닥 네 점을 "
            "다시 찍어 주세요"
        )

    min_x = min(point[0] for point in placed)
    max_x = max(point[0] for point in placed)
    min_y = min(point[1] for point in placed)
    max_y = max(point[1] for point in placed)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    scale = EXTENT_LONG_SIDE / max(span_x, span_y)
    margin = MARGIN_FRACTION * EXTENT_LONG_SIDE
    extent = (span_x * scale + 2 * margin, span_y * scale + 2 * margin)

    def place(point: Optional[Point]) -> Tuple[float, float, bool]:
        """Canvas position, and whether a person has to look at it.

        A point past the vanishing line is not a place on the floor; it is
        parked at the corner and flagged rather than dropped, so the seat
        stays visible and gets moved instead of silently disappearing.
        """
        if point is None:
            return margin, margin, True
        return (
            (point[0] - min_x) * scale + margin,
            (point[1] - min_y) * scale + margin,
            False,
        )

    seats: List[FloorSeat] = []
    for (unit, anchor), (point, _) in zip(seat_anchors, projected[: len(seat_anchors)]):
        x, y, review = place(point)
        w, h = DEFAULT_SIZES.get(unit.kind, DEFAULT_SIZES["table"])
        seats.append(
            FloorSeat(
                seat_id=unit.name,
                kind=unit.kind,
                x=x,
                y=y,
                w=w,
                h=h,
                image_anchor=anchor,
                needs_review=review,
            )
        )

    chairs: List[FloorChair] = []
    for (owner, anchor), (point, _) in zip(
        chair_owners, projected[len(seat_anchors) :]
    ):
        x, y, review = place(point)
        w, h = DEFAULT_SIZES["chair"]
        chairs.append(
            FloorChair(
                seat_id=owner,
                x=x,
                y=y,
                w=w,
                h=h,
                image_anchor=anchor,
                needs_review=review,
            )
        )

    return FloorPlan(
        schema_version=FLOORPLAN_SCHEMA_VERSION,
        extent=extent,
        seats=tuple(seats),
        chairs=tuple(chairs),
        landmarks=(),
    )


def save_floorplan(plan: FloorPlan, path: Path) -> None:
    payload = {
        "schema_version": plan.schema_version,
        "extent": {"width": round(plan.extent[0], 2), "height": round(plan.extent[1], 2)},
        "seats": [
            {
                "seat_id": seat.seat_id,
                "kind": seat.kind,
                "x": round(seat.x, 2),
                "y": round(seat.y, 2),
                "w": round(seat.w, 2),
                "h": round(seat.h, 2),
                "image_anchor": [round(seat.image_anchor[0], 2),
                                 round(seat.image_anchor[1], 2)],
                "needs_review": seat.needs_review,
            }
            for seat in plan.seats
        ],
        "chairs": [
            {
                "seat_id": chair.seat_id,
                "x": round(chair.x, 2),
                "y": round(chair.y, 2),
                "w": round(chair.w, 2),
                "h": round(chair.h, 2),
                "image_anchor": [round(chair.image_anchor[0], 2),
                                 round(chair.image_anchor[1], 2)],
                "needs_review": chair.needs_review,
            }
            for chair in plan.chairs
        ],
        "landmarks": [
            {
                "kind": landmark.kind,
                "label": landmark.label,
                "x": round(landmark.x, 2),
                "y": round(landmark.y, 2),
                "w": round(landmark.w, 2),
                "h": round(landmark.h, 2),
            }
            for landmark in plan.landmarks
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_floorplan(path: Path) -> FloorPlan:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return FloorPlan(
        schema_version=int(data["schema_version"]),
        extent=(float(data["extent"]["width"]), float(data["extent"]["height"])),
        seats=tuple(
            FloorSeat(
                seat_id=str(seat["seat_id"]),
                kind=str(seat["kind"]),
                x=float(seat["x"]),
                y=float(seat["y"]),
                w=float(seat["w"]),
                h=float(seat["h"]),
                image_anchor=(float(seat["image_anchor"][0]),
                              float(seat["image_anchor"][1])),
                needs_review=bool(seat.get("needs_review", False)),
            )
            for seat in data["seats"]
        ),
        chairs=tuple(
            FloorChair(
                seat_id=chair["seat_id"],
                x=float(chair["x"]),
                y=float(chair["y"]),
                w=float(chair["w"]),
                h=float(chair["h"]),
                image_anchor=(float(chair["image_anchor"][0]),
                              float(chair["image_anchor"][1])),
                needs_review=bool(chair.get("needs_review", False)),
            )
            for chair in data["chairs"]
        ),
        landmarks=tuple(
            Landmark(
                kind=str(landmark["kind"]),
                label=str(landmark["label"]),
                x=float(landmark["x"]),
                y=float(landmark["y"]),
                w=float(landmark["w"]),
                h=float(landmark["h"]),
            )
            for landmark in data["landmarks"]
        ),
    )
```

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_floorplan.py" -v`
Expected: PASS — 10개

- [ ] **Step 5: 실제 레이아웃으로 초안을 만들어 본다**

```bash
./venv/Scripts/python.exe -c "
from seatnow_layout import load_layout
from floorplan import build_draft
plan = build_draft(load_layout('layouts/cafe_angle1.json'))
print('extent %.0f x %.0f' % plan.extent)
for seat in plan.seats:
    print('  %-9s %-13s (%6.1f, %6.1f) review=%s' % (
        seat.seat_id, seat.kind, seat.x, seat.y, seat.needs_review))
print('chairs', len(plan.chairs), 'needs_review',
      sum(chair.needs_review for chair in plan.chairs))"
```

Expected: 좌석 12개(T1~T6, BAR7-1~6)가 나오고 좌표가 `extent` 안에 들어간다.
**`needs_review`가 많으면 바닥 네 점이 나쁜 것이므로 여기서 멈추고 다시 찍는다.**

- [ ] **Step 6: 커밋**

```bash
git add floorplan.py tests/test_floorplan.py
git commit -m "feat: floorplan.py - 좌석 위치를 담은 손님용 지도

seat_report 는 상태만 있고 위치가 없어서 앱이 '3자리 남음'은 찍어도
어느 자리인지는 못 보여준다. 그 빠진 절반을 만든다.

좌표는 미터가 아니라 그림 단위다. 실측을 요구하지 않는다 - 손님에게
필요한 건 '창가 오른쪽'이지 '입구에서 3.2m'가 아니다.

image_anchor 를 좌석과 의자마다 남긴다. 편집기가 의자를 옮기면
레이아웃의 인덱스는 밀리지만 화면 좌표는 안 바뀌므로 두 파일을 잇는
열쇠가 되고, 4단계가 쓸 대응쌍이기도 하다."
```

---

### Task 3: `floorplan_editor.py` — 로컬 서버와 저장

**Files:**
- Create: `floorplan_editor.py`
- Test: `tests/test_floorplan_editor.py`

**Interfaces:**
- Consumes: Task 2의 `FloorPlan`, `build_draft`, `load_floorplan`, `save_floorplan`; `seatnow_layout.load_layout`, `save_layout`
- Produces:
  - `editor_state(layout, plan, states) -> dict` — 페이지에 줄 JSON
  - `apply_edits(layout, plan, payload) -> Tuple[SeatLayout, FloorPlan]` — 편집 결과 반영
  - `save_both(layout, plan, layout_path, floorplan_path)` — 원자적 저장
  - `latest_states(log_path) -> Dict[str, str]` — 미리보기용 `seat_id → state`
  - `main(argv=None) -> int`

**저장이 원자적이어야 하는 이유:** 두 파일 중 하나만 써지면 **지도와 판정이 어긋난 채로 남는다.** 임시 파일에 둘 다 쓰고, 둘 다 성공했을 때만 바꾼다 (설계 §10).

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_floorplan_editor.py`:

```python
"""Unit tests for the floor plan editor's server side.

No socket is opened and no browser runs: what matters is that an edit lands
in both files and that a half-written save cannot happen.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from floorplan import build_draft, load_floorplan
from floorplan_editor import apply_edits, editor_state, latest_states, save_both
from seatnow_layout import (
    FloorReference,
    LayoutChair,
    LayoutTable,
    SeatLayout,
    load_layout,
)


REFERENCE = FloorReference(
    image_points=((700.0, 900.0), (900.0, 600.0), (1300.0, 600.0), (1600.0, 900.0))
)


def layout():
    return SeatLayout(
        schema_version=3,
        source={"width": 1920, "height": 1080},
        tables=(
            LayoutTable(
                id=1,
                name="T1",
                box=(800.0, 700.0, 1000.0, 850.0),
                chairs=(LayoutChair(id=1, box=(780.0, 800.0, 830.0, 880.0)),),
            ),
            LayoutTable(id=2, name="T2", box=(1200.0, 700.0, 1400.0, 850.0)),
        ),
        floor_reference=REFERENCE,
    )


class EditorStateTests(unittest.TestCase):
    def test_state_carries_seats_chairs_and_extent(self):
        plan = build_draft(layout())
        state = editor_state(layout(), plan, {})
        self.assertEqual(len(state["seats"]), 2)
        self.assertEqual(len(state["chairs"]), 1)
        self.assertIn("extent", state)

    def test_live_states_are_attached_when_known(self):
        plan = build_draft(layout())
        state = editor_state(layout(), plan, {"T1": "occupied"})
        by_id = {seat["seat_id"]: seat for seat in state["seats"]}
        self.assertEqual(by_id["T1"]["state"], "occupied")
        self.assertEqual(by_id["T2"]["state"], "unknown")

    def test_a_seat_only_in_the_report_is_counted_not_drawn(self):
        # Layout and map disagreeing is a real failure worth reporting, but
        # inventing a position for the stray seat would hide it.
        plan = build_draft(layout())
        state = editor_state(layout(), plan, {"T1": "occupied", "GHOST": "empty"})
        self.assertEqual([seat["seat_id"] for seat in state["seats"]], ["T1", "T2"])
        self.assertEqual(state["unmapped_seats"], ["GHOST"])


class ApplyEditsTests(unittest.TestCase):
    def _payload(self, **overrides):
        plan = build_draft(layout())
        payload = {
            "seats": [
                {"seat_id": seat.seat_id, "x": 11.0, "y": 22.0, "w": seat.w, "h": seat.h}
                for seat in plan.seats
            ],
            "chairs": [
                {
                    "image_anchor": list(chair.image_anchor),
                    "seat_id": chair.seat_id,
                    "x": 33.0,
                    "y": 44.0,
                    "w": chair.w,
                    "h": chair.h,
                }
                for chair in plan.chairs
            ],
            "landmarks": [],
        }
        payload.update(overrides)
        return plan, payload

    def test_moved_seat_position_is_kept(self):
        plan, payload = self._payload()
        _, updated = apply_edits(layout(), plan, payload)
        self.assertEqual((updated.seats[0].x, updated.seats[0].y), (11.0, 22.0))

    def test_image_anchor_is_never_overwritten(self):
        # It is the correspondence stage 4 needs and the key that identifies
        # a chair; letting the browser rewrite it would destroy both.
        plan, payload = self._payload()
        payload["seats"][0]["image_anchor"] = [1.0, 2.0]
        _, updated = apply_edits(layout(), plan, payload)
        self.assertEqual(updated.seats[0].image_anchor, plan.seats[0].image_anchor)

    def test_chair_reassignment_moves_it_in_the_layout(self):
        plan, payload = self._payload()
        payload["chairs"][0]["seat_id"] = "T2"
        updated_layout, _ = apply_edits(layout(), plan, payload)
        by_name = {table.name: table for table in updated_layout.tables}
        self.assertEqual(len(by_name["T1"].chairs), 0)
        self.assertEqual(len(by_name["T2"].chairs), 1)

    def test_chair_unassignment_moves_it_to_unassigned(self):
        plan, payload = self._payload()
        payload["chairs"][0]["seat_id"] = None
        updated_layout, _ = apply_edits(layout(), plan, payload)
        self.assertEqual(len(updated_layout.unassigned_chairs), 1)
        self.assertEqual(len(updated_layout.tables[0].chairs), 0)

    def test_chair_image_box_is_unchanged_by_reassignment(self):
        plan, payload = self._payload()
        payload["chairs"][0]["seat_id"] = "T2"
        updated_layout, _ = apply_edits(layout(), plan, payload)
        moved = updated_layout.tables[1].chairs[0]
        self.assertEqual(moved.box, (780.0, 800.0, 830.0, 880.0))

    def test_landmarks_are_stored(self):
        plan, payload = self._payload()
        payload["landmarks"] = [
            {"kind": "entrance", "label": "입구", "x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}
        ]
        _, updated = apply_edits(layout(), plan, payload)
        self.assertEqual(updated.landmarks[0].label, "입구")

    def test_unknown_seat_id_on_a_chair_is_refused(self):
        plan, payload = self._payload()
        payload["chairs"][0]["seat_id"] = "NOPE"
        with self.assertRaises(ValueError):
            apply_edits(layout(), plan, payload)


class SaveBothTests(unittest.TestCase):
    def test_both_files_are_written(self):
        plan = build_draft(layout())
        with tempfile.TemporaryDirectory() as raw:
            layout_path = Path(raw) / "layout.json"
            plan_path = Path(raw) / "floorplan.json"
            save_both(layout(), plan, layout_path, plan_path)
            self.assertEqual(len(load_layout(layout_path).tables), 2)
            self.assertEqual(len(load_floorplan(plan_path).seats), 2)

    def test_a_failing_write_leaves_both_files_untouched(self):
        # Half a save is worse than none: the map and the judgement would
        # disagree with nothing saying so.
        plan = build_draft(layout())
        with tempfile.TemporaryDirectory() as raw:
            layout_path = Path(raw) / "layout.json"
            plan_path = Path(raw) / "sub" / "floorplan.json"
            save_both(layout(), plan, layout_path, plan_path)
            before = layout_path.read_text(encoding="utf-8")

            blocked = Path(raw) / "blocked"
            blocked.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(Exception):
                save_both(layout(), plan, layout_path, blocked / "floorplan.json")
            self.assertEqual(layout_path.read_text(encoding="utf-8"), before)


class LatestStatesTests(unittest.TestCase):
    def test_last_record_wins(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "run.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "timestamp": stamp,
                            "seat_report": {
                                "seats": [
                                    {"seat_id": "T1", "kind": "table", "state": state}
                                ]
                            },
                        }
                    )
                    for stamp, state in ((0.0, "empty"), (15.0, "occupied"))
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(latest_states(path), {"T1": "occupied"})

    def test_counted_zone_seats_are_left_uncoloured(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "run.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "timestamp": 0.0,
                        "seat_report": {
                            "seats": [
                                {
                                    "seat_id": "BAR7",
                                    "kind": "counted_zone",
                                    "capacity": 2,
                                    "occupied": 1,
                                    "free": 1,
                                    "unknown": 0,
                                }
                            ]
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            # A zone reports counts, not per-slot states, so the map cannot
            # colour individual stools from it and says so.
            self.assertEqual(latest_states(path), {})

    def test_missing_file_is_empty(self):
        self.assertEqual(latest_states(Path("does-not-exist.jsonl")), {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_floorplan_editor.py" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'floorplan_editor'`

- [ ] **Step 3: 구현한다**

`floorplan_editor.py`:

```python
"""Serve the floor plan editor and take its edits back into both files.

The install has to finish in one visit at the cafe, so the editor is a page
the edge box serves over the cafe's wifi: a laptop or a phone can open it and
nobody carries a monitor and mouse.  The same page doubles as the customer
view, which makes the final check free.

Editing the map is not only cosmetic.  Chair ownership is judged as ground
truth (seatnow_core.py:1531-1534) and is nearly impossible to get right on a
camera image, where perspective lifts a far chair onto its neighbour's table.
With perspective gone it is obvious, so the editor writes ownership back into
the layout -- which is why a save touches two files and why it must not be
able to write only one of them.

    python floorplan_editor.py --layout layouts/cafe_angle1.json \\
        --floorplan layouts/cafe_angle1.floorplan.json --log sample_results/angle1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from floor_projection import floor_anchor
from floorplan import (
    FloorChair,
    FloorPlan,
    Landmark,
    build_draft,
    load_floorplan,
    save_floorplan,
)
from seatnow_layout import (
    COUNTED_ZONE_KIND,
    LayoutChair,
    SeatLayout,
    load_layout,
    save_layout,
)


PROJECT_DIR = Path(__file__).resolve().parent
PAGE_PATH = PROJECT_DIR / "static" / "floorplan_editor.html"


def latest_states(log_path: Path) -> Dict[str, str]:
    """seat_id -> state from the newest record of a run log.

    Only plain tables carry a per-seat state.  A counted_zone reports counts
    over its slots, so individual stools cannot be coloured from it; those
    seats are simply left out rather than guessed at.
    """
    path = Path(log_path)
    if not path.exists():
        return {}
    last = None
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                last = line
    if last is None:
        return {}
    report = (json.loads(last).get("seat_report") or {})
    states: Dict[str, str] = {}
    for seat in report.get("seats") or []:
        if seat.get("kind") == COUNTED_ZONE_KIND:
            continue
        if "state" in seat:
            states[str(seat["seat_id"])] = str(seat["state"])
    return states


def editor_state(
    layout: SeatLayout, plan: FloorPlan, states: Dict[str, str]
) -> Dict[str, object]:
    """Everything the page needs in one object."""
    drawn = {seat.seat_id for seat in plan.seats}
    return {
        "extent": {"width": plan.extent[0], "height": plan.extent[1]},
        "seats": [
            {
                "seat_id": seat.seat_id,
                "kind": seat.kind,
                "x": seat.x,
                "y": seat.y,
                "w": seat.w,
                "h": seat.h,
                "needs_review": seat.needs_review,
                "state": states.get(seat.seat_id, "unknown"),
            }
            for seat in plan.seats
        ],
        "chairs": [
            {
                "image_anchor": list(chair.image_anchor),
                "seat_id": chair.seat_id,
                "x": chair.x,
                "y": chair.y,
                "w": chair.w,
                "h": chair.h,
                "needs_review": chair.needs_review,
            }
            for chair in plan.chairs
        ],
        "landmarks": [
            {
                "kind": landmark.kind,
                "label": landmark.label,
                "x": landmark.x,
                "y": landmark.y,
                "w": landmark.w,
                "h": landmark.h,
            }
            for landmark in plan.landmarks
        ],
        "unmapped_seats": sorted(set(states) - drawn),
    }


def _seat_owner_names(layout: SeatLayout) -> Dict[str, str]:
    """Map every drawable owner name onto the layout table it belongs to."""
    owners: Dict[str, str] = {}
    for table in layout.tables:
        owners[table.name] = table.name
        if table.kind == COUNTED_ZONE_KIND:
            for seat in table.seats:
                owners[f"{table.name}-{seat.id}"] = table.name
    return owners


def apply_edits(
    layout: SeatLayout, plan: FloorPlan, payload: Dict[str, object]
) -> Tuple[SeatLayout, FloorPlan]:
    """Fold the page's edits into a new layout and a new floor plan.

    ``image_anchor`` is read from the existing plan, never from the payload:
    it identifies a chair across a reassignment and is half of the
    correspondence stage 4 fits its transform to.  A browser has no business
    rewriting it.
    """
    owners = _seat_owner_names(layout)

    seat_edits = {
        str(seat["seat_id"]): seat for seat in payload.get("seats", [])  # type: ignore[union-attr]
    }
    seats = tuple(
        replace(
            seat,
            x=float(seat_edits[seat.seat_id]["x"]),
            y=float(seat_edits[seat.seat_id]["y"]),
            w=float(seat_edits[seat.seat_id]["w"]),
            h=float(seat_edits[seat.seat_id]["h"]),
            needs_review=False,
        )
        if seat.seat_id in seat_edits
        else seat
        for seat in plan.seats
    )

    chair_edits = {
        (float(chair["image_anchor"][0]), float(chair["image_anchor"][1])): chair
        for chair in payload.get("chairs", [])  # type: ignore[union-attr]
    }
    chairs: List[FloorChair] = []
    ownership: Dict[Tuple[float, float], Optional[str]] = {}
    for chair in plan.chairs:
        edit = chair_edits.get(chair.image_anchor)
        if edit is None:
            chairs.append(chair)
            ownership[chair.image_anchor] = chair.seat_id
            continue
        seat_id = edit.get("seat_id")
        if seat_id is not None and str(seat_id) not in owners:
            raise ValueError(f"의자를 붙일 좌석이 레이아웃에 없습니다: {seat_id!r}")
        chairs.append(
            replace(
                chair,
                seat_id=None if seat_id is None else str(seat_id),
                x=float(edit["x"]),
                y=float(edit["y"]),
                w=float(edit["w"]),
                h=float(edit["h"]),
                needs_review=False,
            )
        )
        ownership[chair.image_anchor] = None if seat_id is None else str(seat_id)

    landmarks = tuple(
        Landmark(
            kind=str(landmark["kind"]),
            label=str(landmark["label"]),
            x=float(landmark["x"]),
            y=float(landmark["y"]),
            w=float(landmark["w"]),
            h=float(landmark["h"]),
        )
        for landmark in payload.get("landmarks", [])  # type: ignore[union-attr]
    )

    updated_plan = FloorPlan(
        schema_version=plan.schema_version,
        extent=plan.extent,
        seats=seats,
        chairs=tuple(chairs),
        landmarks=landmarks,
    )
    return _rebuild_layout(layout, ownership), updated_plan


def _rebuild_layout(
    layout: SeatLayout, ownership: Dict[Tuple[float, float], Optional[str]]
) -> SeatLayout:
    """Re-hang every chair under the table the map says owns it.

    Only which list a chair sits in changes.  Its image box is untouched --
    that is the part the installer already drew correctly, and judgement
    reads the lists, so moving it here is enough to change the verdict.
    """
    owners = _seat_owner_names(layout)
    all_chairs: List[Tuple[Optional[str], LayoutChair]] = []
    for table in layout.tables:
        for chair in table.chairs:
            anchor = floor_anchor(chair.box)
            owner = ownership.get(anchor, table.name)
            all_chairs.append((owner, chair))
    for chair in layout.unassigned_chairs:
        anchor = floor_anchor(chair.box)
        all_chairs.append((ownership.get(anchor, None), chair))

    by_table: Dict[str, List[LayoutChair]] = {table.name: [] for table in layout.tables}
    orphans: List[LayoutChair] = []
    for owner, chair in all_chairs:
        table_name = owners.get(owner or "", None)
        if table_name is None:
            orphans.append(chair)
        else:
            by_table[table_name].append(chair)

    tables = tuple(
        replace(
            table,
            chairs=tuple(
                replace(chair, id=index)
                for index, chair in enumerate(by_table[table.name], start=1)
            ),
        )
        for table in layout.tables
    )
    return replace(
        layout,
        tables=tables,
        unassigned_chairs=tuple(
            replace(chair, id=index) for index, chair in enumerate(orphans, start=1)
        ),
    )


def save_both(
    layout: SeatLayout, plan: FloorPlan, layout_path: Path, floorplan_path: Path
) -> None:
    """Write both files, or neither.

    Half a save leaves the map and the judgement disagreeing with nothing
    saying so, which is worse than losing the edit.
    """
    layout_path = Path(layout_path)
    floorplan_path = Path(floorplan_path)
    staged: List[Tuple[Path, Path]] = []
    try:
        for path, writer, value in (
            (layout_path, save_layout, layout),
            (floorplan_path, save_floorplan, plan),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp"
            )
            os.close(handle)
            writer(value, Path(temporary))
            staged.append((Path(temporary), path))
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    for temporary, path in staged:
        os.replace(temporary, path)


class _Handler(BaseHTTPRequestHandler):
    layout_path: Path
    floorplan_path: Path
    log_path: Optional[Path]

    def log_message(self, *args) -> None:  # quiet
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path == "/state":
            layout = load_layout(self.layout_path)
            plan = (
                load_floorplan(self.floorplan_path)
                if self.floorplan_path.exists()
                else build_draft(layout)
            )
            states = latest_states(self.log_path) if self.log_path else {}
            body = json.dumps(
                editor_state(layout, plan, states), ensure_ascii=False
            ).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/save":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        try:
            layout = load_layout(self.layout_path)
            plan = (
                load_floorplan(self.floorplan_path)
                if self.floorplan_path.exists()
                else build_draft(layout)
            )
            updated_layout, updated_plan = apply_edits(layout, plan, payload)
            save_both(
                updated_layout, updated_plan, self.layout_path, self.floorplan_path
            )
        except Exception as exc:  # noqa: BLE001 - the page shows the message
            body = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
            self._send(400, body.encode("utf-8"), "application/json; charset=utf-8")
            return
        body = json.dumps({"ok": True}, ensure_ascii=False).encode("utf-8")
        self._send(200, body, "application/json; charset=utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True, help="calibrate.py 가 만든 레이아웃")
    parser.add_argument("--floorplan", type=Path, help="평면도 경로 (기본: <layout>.floorplan.json)")
    parser.add_argument("--log", type=Path, help="미리보기에 쓸 JSONL (없으면 회색)")
    parser.add_argument("--host", default="0.0.0.0", help="0.0.0.0 이면 같은 와이파이의 폰에서도 열린다")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--no-open", action="store_true", help="브라우저를 자동으로 열지 않는다")
    args = parser.parse_args(argv)

    _Handler.layout_path = args.layout
    _Handler.floorplan_path = args.floorplan or args.layout.with_suffix(".floorplan.json")
    _Handler.log_path = args.log

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    url = f"http://localhost:{args.port}/"
    print(f"평면도 편집기: {url}")
    print("같은 와이파이의 폰에서 열려면 이 PC의 IP로 접속하세요 (예: http://192.168.0.10:%d/)" % args.port)
    print("Ctrl+C 로 종료")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_floorplan_editor.py" -v`
Expected: PASS — 15개

- [ ] **Step 5: 전체 테스트**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add floorplan_editor.py tests/test_floorplan_editor.py
git commit -m "feat: floorplan_editor.py - 편집 결과를 두 파일에 함께 되돌린다

의자 소속은 판정이 정답으로 읽는데(seatnow_core.py:1531-1534) 카메라
화면에서는 원근 때문에 눈으로 맞추기 어렵다. 원근이 펴진 지도에서는
안 헷갈리므로 편집 결과를 레이아웃에 되돌린다.

그래서 저장이 두 파일을 건드리고, 하나만 써지면 안 된다. 임시 파일에
둘 다 쓰고 둘 다 성공했을 때만 바꾼다 - 반만 저장되면 지도와 판정이
어긋난 채로 아무도 모른다.

image_anchor 는 페이로드에서 읽지 않는다. 의자의 신원이자 4단계가
변환을 다시 맞출 대응쌍이라 브라우저가 고쳐 쓸 것이 아니다."
```

---

### Task 4: `static/floorplan_editor.html` — 편집 화면

**Files:**
- Create: `static/floorplan_editor.html`
- Test: 없음 (브라우저 화면). Task 5의 실제 실행으로 확인한다

**Interfaces:**
- Consumes: Task 3의 `GET /state`, `POST /save`

**왜 테스트가 없나:** 이 파일은 SVG를 그리고 마우스를 받는 껍데기다. 판단이 들어가는 부분(소속 반영, 원자적 저장)은 전부 Task 3에 있고 거기서 검사한다. `calibrate.py`가 상태 기계와 창을 나눈 것과 같은 방식이다.

- [ ] **Step 1: 페이지를 만든다**

`static/floorplan_editor.html`:

```html
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SeatNow 평면도 편집기</title>
<style>
  body { margin: 0; font-family: system-ui, sans-serif; background: #14171c; color: #e8ecf1; }
  header { padding: 10px 14px; background: #1d2129; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  button { background: #2b313d; color: #e8ecf1; border: 1px solid #3a4250; border-radius: 6px; padding: 7px 12px; font-size: 14px; cursor: pointer; }
  button.on { background: #3f6fd8; border-color: #5b87ea; }
  #hint { margin-left: auto; font-size: 13px; color: #9aa6b6; }
  svg { display: block; width: 100vw; height: calc(100vh - 96px); background: #0f1216; touch-action: none; }
  .seat { fill: #2f3948; stroke: #58657a; stroke-width: 2; }
  .seat.occupied { fill: #7d2b2b; stroke: #c05252; }
  .seat.empty    { fill: #2b6b3a; stroke: #4fbf70; }
  .seat.unknown  { fill: #3a3f47; stroke: #6b737f; }
  .seat.review   { stroke: #e0a13c; stroke-dasharray: 6 4; }
  .chair { fill: #d8c04a; stroke: #b8a032; stroke-width: 1.5; }
  .chair.orphan { fill: #a35bb0; stroke: #c47dd0; }
  .landmark { fill: #26303c; stroke: #47566b; stroke-width: 2; }
  .picked { stroke: #ff5757 !important; stroke-width: 4 !important; }
  text { font-size: 13px; fill: #cbd4e1; pointer-events: none; user-select: none; }
  footer { padding: 6px 14px; font-size: 12px; color: #8b96a6; background: #1d2129; }
</style>
</head>
<body>
<header>
  <button id="mode-move" class="on">끌어서 옮기기</button>
  <button id="mode-assign">의자 소속 정하기</button>
  <button id="mode-landmark">지형지물 추가</button>
  <button id="toggle-preview">미리보기</button>
  <button id="save">저장</button>
  <span id="hint">좌석과 의자를 끌어서 실제 배치에 맞춥니다.</span>
</header>
<svg id="canvas"></svg>
<footer id="legend">
  초록 = 빈 자리 · 빨강 = 사용 중 · <b>회색 = 가려서 확인이 안 되는 자리</b> ·
  노랑 = 의자 · 보라 = 소속 미정 의자 · 주황 점선 = 위치 확인 필요 ·
  <b>테이블은 통째로 한 자리로 표시됩니다</b>
</footer>
<script>
const SVG_NS = "http://www.w3.org/2000/svg";
let state = null, mode = "move", preview = false, pickedChair = null;
const canvas = document.getElementById("canvas");
const hint = document.getElementById("hint");

function setMode(next) {
  mode = next;
  pickedChair = null;
  for (const [id, name] of [["mode-move","move"],["mode-assign","assign"],["mode-landmark","landmark"]])
    document.getElementById(id).classList.toggle("on", name === mode);
  hint.textContent = {
    move: "좌석과 의자를 끌어서 실제 배치에 맞춥니다.",
    assign: "의자를 누른 뒤 붙일 좌석을 누르세요. 빈 곳을 누르면 소속 미정.",
    landmark: "빈 곳을 눌러 입구·카운터 같은 표시를 놓습니다.",
  }[mode];
  draw();
}
for (const [id, name] of [["mode-move","move"],["mode-assign","assign"],["mode-landmark","landmark"]])
  document.getElementById(id).onclick = () => setMode(name);

document.getElementById("toggle-preview").onclick = (event) => {
  preview = !preview;
  event.target.classList.toggle("on", preview);
  draw();
};

function rect(item, cls, extra) {
  const node = document.createElementNS(SVG_NS, "rect");
  node.setAttribute("x", item.x - item.w / 2);
  node.setAttribute("y", item.y - item.h / 2);
  node.setAttribute("width", item.w);
  node.setAttribute("height", item.h);
  node.setAttribute("rx", 6);
  node.setAttribute("class", cls + (extra || ""));
  return node;
}

function label(item, text, dy) {
  const node = document.createElementNS(SVG_NS, "text");
  node.setAttribute("x", item.x);
  node.setAttribute("y", item.y + (dy || 4));
  node.setAttribute("text-anchor", "middle");
  node.textContent = text;
  return node;
}

function draw() {
  if (!state) return;
  canvas.setAttribute("viewBox", `0 0 ${state.extent.width} ${state.extent.height}`);
  canvas.innerHTML = "";
  for (const landmark of state.landmarks) {
    canvas.appendChild(rect(landmark, "landmark"));
    canvas.appendChild(label(landmark, landmark.label));
  }
  for (const seat of state.seats) {
    const cls = "seat " + (preview ? seat.state : "") + (seat.needs_review ? " review" : "");
    const node = rect(seat, cls);
    node.dataset.seat = seat.seat_id;
    canvas.appendChild(node);
    canvas.appendChild(label(seat, seat.seat_id));
  }
  for (const chair of state.chairs) {
    const picked = pickedChair && pickedChair.image_anchor.join() === chair.image_anchor.join();
    const node = rect(chair, "chair" + (chair.seat_id ? "" : " orphan") + (picked ? " picked" : ""));
    node.dataset.chair = chair.image_anchor.join();
    canvas.appendChild(node);
  }
}

function toCanvas(event) {
  const box = canvas.getBoundingClientRect();
  const scale = state.extent.width / box.width;
  return { x: (event.clientX - box.left) * scale, y: (event.clientY - box.top) * scale };
}

function itemAt(point) {
  for (const chair of state.chairs)
    if (Math.abs(chair.x - point.x) <= chair.w / 2 && Math.abs(chair.y - point.y) <= chair.h / 2)
      return { kind: "chair", item: chair };
  for (const seat of state.seats)
    if (Math.abs(seat.x - point.x) <= seat.w / 2 && Math.abs(seat.y - point.y) <= seat.h / 2)
      return { kind: "seat", item: seat };
  for (const landmark of state.landmarks)
    if (Math.abs(landmark.x - point.x) <= landmark.w / 2 && Math.abs(landmark.y - point.y) <= landmark.h / 2)
      return { kind: "landmark", item: landmark };
  return null;
}

let dragging = null, offset = null;
canvas.addEventListener("pointerdown", (event) => {
  const point = toCanvas(event);
  const hit = itemAt(point);
  if (mode === "landmark") {
    if (hit) return;
    const text = prompt("표시 이름 (예: 입구, 카운터, 화장실)");
    if (!text) return;
    state.landmarks.push({ kind: "landmark", label: text, x: point.x, y: point.y, w: 120, h: 46 });
    draw();
    return;
  }
  if (mode === "assign") {
    if (hit && hit.kind === "chair") { pickedChair = hit.item; draw(); return; }
    if (!pickedChair) return;
    pickedChair.seat_id = hit && hit.kind === "seat" ? hit.item.seat_id : null;
    pickedChair = null;
    draw();
    return;
  }
  if (!hit) return;
  dragging = hit.item;
  offset = { x: point.x - hit.item.x, y: point.y - hit.item.y };
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  const point = toCanvas(event);
  dragging.x = point.x - offset.x;
  dragging.y = point.y - offset.y;
  dragging.needs_review = false;
  draw();
});
canvas.addEventListener("pointerup", () => { dragging = null; });

document.getElementById("save").onclick = async () => {
  const response = await fetch("/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ seats: state.seats, chairs: state.chairs, landmarks: state.landmarks }),
  });
  const result = await response.json();
  hint.textContent = result.ok ? "저장했습니다 (평면도 + 레이아웃)" : "저장 실패: " + result.error;
};

fetch("/state").then((response) => response.json()).then((data) => {
  state = data;
  if (data.unmapped_seats.length)
    hint.textContent = "지도에 없는 좌석 " + data.unmapped_seats.length + "개 — 레이아웃과 지도가 어긋났습니다";
  draw();
});
</script>
</body>
</html>
```

- [ ] **Step 2: 파일이 서버에서 읽히는지 확인한다**

```bash
./venv/Scripts/python.exe -c "
from floorplan_editor import PAGE_PATH
text = PAGE_PATH.read_text(encoding='utf-8')
for needle in ('/state', '/save', 'image_anchor', 'needs_review'):
    assert needle in text, needle
print('페이지 OK', len(text), 'bytes')"
```

Expected: `페이지 OK ... bytes`

- [ ] **Step 3: 커밋**

```bash
git add static/floorplan_editor.html
git commit -m "feat: 평면도 편집 화면

끌어서 옮기기 / 의자 소속 정하기 / 지형지물 / 미리보기 네 가지다.
판단이 들어가는 부분은 전부 서버에 있고 여기는 SVG 를 그리고 마우스를
받는 껍데기다 - calibrate.py 가 상태 기계와 창을 나눈 것과 같다.

범례에 '회색은 가려서 확인이 안 되는 자리'와 '테이블은 통째로 한
자리로 표시됩니다'를 적는다. 손님이 잘못 읽으면 헛걸음한다."
```

---

### Task 5: 실제로 돌려서 angle1 평면도를 만든다

**Files:**
- Create: `layouts/cafe_angle1.floorplan.json`
- Modify: `README.md`, `plan.md`, `.gitignore`

**Interfaces:**
- Consumes: Task 1~4 전부

- [ ] **Step 1: 초안이 나오는지 확인한다**

```bash
./venv/Scripts/python.exe -c "
from seatnow_layout import load_layout
from floorplan import build_draft
plan = build_draft(load_layout('layouts/cafe_angle1.json'))
print('extent %.0f x %.0f, 좌석 %d, 의자 %d, 확인필요 %d' % (
    plan.extent[0], plan.extent[1], len(plan.seats), len(plan.chairs),
    sum(s.needs_review for s in plan.seats) + sum(c.needs_review for c in plan.chairs)))"
```

Expected: 좌석 12개, 의자 25개. **확인필요가 0이 아니면 바닥 네 점이 나쁜 것이므로 `calibrate.py [f]`로 다시 찍는다.**

- [ ] **Step 2: 편집기를 띄운다**

```bash
./venv/Scripts/python.exe floorplan_editor.py \
  --layout layouts/cafe_angle1.json \
  --log sample_results/angle1_layout.jsonl
```

브라우저가 열린다. 화면에서 순서대로:

1. **끌어서 옮기기** — 좌석 12개와 의자 25개를 실제 배치에 맞춘다
2. **의자 소속 정하기** — 보라색(소속 미정)을 각 좌석에 붙인다. **창가 의자는 `BAR7-*`에 붙인다**
3. **지형지물 추가** — 입구, 카운터
4. **미리보기** — 초록/빨강/회색이 실제 배치와 맞는지 눈으로 확인
5. **저장**

- [ ] **Step 3: 두 파일이 같이 갱신됐는지 확인한다**

```bash
./venv/Scripts/python.exe -c "
from seatnow_layout import load_layout
from floorplan import load_floorplan
layout = load_layout('layouts/cafe_angle1.json')
plan = load_floorplan('layouts/cafe_angle1.floorplan.json')
print('레이아웃: 테이블 %d, 의자 %d, 소속미정 %d' % (
    len(layout.tables), len(layout.chair_boxes()), len(layout.unassigned_chairs)))
print('평면도: 좌석 %d, 의자 %d, 지형지물 %d' % (
    len(plan.seats), len(plan.chairs), len(plan.landmarks)))"
```

Expected: 레이아웃의 소속 미정 개수가 편집기에서 붙인 만큼 줄어 있다.

- [ ] **Step 4: 소속 변경이 판정을 바꿨는지 확인한다**

```bash
./venv/Scripts/python.exe seatnow.py sample_raw/cafe_sample_angle1.mov \
  --no-video --log-detections --layout layouts/cafe_angle1.json \
  --log sample_results/angle1_edited.jsonl --max-samples 6
./venv/Scripts/python.exe inspect_run.py sample_results/angle1_edited.jsonl \
  --judge frames/angle1/judge --output docs/inspect/angle1_edited.md \
  --title "# 검출 판독표 — angle1 (평면도에서 소속 정리 후)"
```

`docs/inspect/angle1_layout.md`(정리 전)와 나란히 놓고 **`occupied` 평균과 사유 분포가 어떻게 달라졌는지** 본다. 이것이 "지도에서 고친 것이 판정을 바꾼다"의 증거다.

- [ ] **Step 5: 문서를 갱신한다**

`.gitignore`에 한 줄을 더한다 (평면도는 매장별 산출물이다):

```
*.floorplan.json
```

`README.md`의 도구 표에 세 줄을 더한다 (`inspect_run.py` 줄 아래):

```markdown
| `floor_projection.py` | 바닥 네 점으로 카메라 화면을 평면으로 펴는 변환 |
| `floorplan.py` | 손님용 2D 배치도(`floorplan.json`) 초안 생성·읽기·쓰기 |
| `floorplan_editor.py` | 평면도 편집기 (브라우저). 저장 시 평면도와 레이아웃을 함께 갱신 |
```

`README.md`의 평가·벤치 도구 블록 끝에 더한다:

```bash
# 8. 평면도 편집기 — 설치 5단계
#    끌어서 위치 다듬기 + 의자 소속 정하기 + 지형지물 + 미리보기.
#    저장하면 floorplan.json 과 layouts/*.json 이 함께 갱신된다
#    (의자 소속은 레이아웃에 있어야 판정이 읽는다)
./venv/bin/python floorplan_editor.py --layout layouts/cafe_angle1.json \
  --log sample_results/angle1_layout.jsonl
```

`plan.md` §0-a 상태표의 "2D 평면도 1단계" 줄 아래에 더한다:

```markdown
| **2D 평면도 2단계** | ✅ **완료 (2026-09-01)** | 투영·`floorplan.json`·브라우저 편집기. 의자 소속을 지도에서 고치면 판정이 따라온다 |
```

- [ ] **Step 6: 전체 테스트와 커밋**

```bash
./venv/Scripts/python.exe -m unittest discover tests
git add README.md plan.md .gitignore docs/inspect
git commit -m "docs: 2D 평면도 2단계 완료 - 편집기와 실행 결과"
```

---

## 검토 메모

**설계 문서 대응 (2단계 범위)**

| 설계 절 | 태스크 |
|---|---|
| §4 좌표 변환, 바닥 접점만 편다 | Task 1 |
| §4 퇴화 거부 | Task 1 (꼬임·좁음·개수) |
| §5 `floorplan.json`, `image_anchor` | Task 2 |
| §6-5a 위치 다듬기 | Task 4 |
| §6-5b 의자 소속 | Task 3(반영) + Task 4(조작) |
| §6-5c 지형지물 | Task 3 + Task 4 |
| §6-5d 미리보기 | Task 3(`latest_states`) + Task 4 |
| §7 두 파일 함께 갱신 | Task 3 (`save_both`, `_rebuild_layout`) |
| §9 3색·범례·테이블 단위 | Task 4 (CSS + `<footer>`) |
| §10 바닥 점 없음/퇴화/발산/어긋난 `seat_id`/저장 실패 | Task 1·2·3 |
| §11 테스트 | Task 1·2·3 |
| §12 완료 기준 | Task 5 |

**3단계로 미룬 것** — 가구 이동 신호(`maintenance`, 옮겨진 의자 제외).
**4단계로 미룬 것** — 바닥 좌표로 판정. 재료(`image_anchor`)는 Task 2가 남긴다.

**설계에 없어 이 계획에서 정한 것 두 가지**

1. **바 구역 자리의 미리보기 색.** `seat_report`의 `counted_zone`은 칸별 상태가
   아니라 개수만 낸다(`seatnow_report.py:163-181`). 그래서 칸을 하나씩 칠할 수
   없고, `latest_states`가 그 좌석들을 **빼서 회색으로 남긴다.** 없는 정보를
   지어내지 않는다. 칸별 색이 필요해지면 `seat_report`를 늘려야 하고, 그건
   출력 계약 변경이라 이 계획의 범위가 아니다
2. **평면도 파일 경로.** `--floorplan`을 안 주면 `<레이아웃>.floorplan.json`이다.
   매장마다 레이아웃과 평면도가 짝이므로 이름을 붙여 두는 편이 헷갈리지 않는다

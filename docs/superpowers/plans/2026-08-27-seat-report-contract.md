# 좌석 리포트 출력 계약 (seat_report) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 손님 앱이 쓸 좌석 가용성 계약(`seat_report`)을 JSONL에 추가하고, 일자형/벽 책상을 설치 때 사람이 그은 좌석 칸 단위로 판정한다.

**Architecture:** 바 구역(`counted_zone`)의 좌석 칸을 **판정 단위로 평탄화**해서 기존 증거 연결·디바운싱 파이프라인이 칸마다 그대로 돌게 한다. 새 카운트 로직을 만들지 않고, 리포트 생성 시점에 구역별로 다시 묶는다. 리포트 생성은 신규 순수 함수 모듈 `seatnow_report.py`로 분리한다 (`seatnow_core.py`는 이미 3,373줄).

**Tech Stack:** Python 3.9~3.11, `unittest`(표준 라이브러리), numpy<2, ultralytics YOLOv8. 모델 없이 도는 순수 로직 테스트가 기본.

## Global Constraints

- 스펙: `docs/superpowers/specs/2026-08-27-seat-report-contract-design.md`
- 프로젝트 규칙: `CLAUDE.md` — 카메라는 우리가 1대 설치, 매장 수작업은 설치 시 1회 검수가 전부, 매장별 임계값 튜닝을 요구하는 설계 금지
- **기존 v1 레이아웃 파일(`layouts/*.json` 2개)은 무수정으로 로드되어야 한다.**
- `AnalyzerConfig`에 새 튜닝 파라미터를 추가하지 않는다. 기존 연결 규칙을 재사용한다.
- Python 3.9 호환: `Optional[X]` / `Tuple[X, ...]`를 쓰고 `X | Y` 문법을 쓰지 않는다.
- 테스트는 `unittest`. 실행: `python -m unittest discover tests`
- 기존 테스트 171개는 계속 통과해야 한다.
- `totals.free`는 UNKNOWN을 절대 포함하지 않는다 — 계약의 핵심 불변식.

---

### Task 1: 레이아웃 스키마 v2 — `counted_zone` + `seats`

**Files:**
- Modify: `seatnow_layout.py`
- Test: `tests/test_seatnow_layout.py`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: `LayoutSeat(id: int, box: Box)`, `LayoutTable.kind: str`, `LayoutTable.seats: Tuple[LayoutSeat, ...]`, `SCHEMA_VERSION = 2`, `COUNTED_ZONE_KIND = "counted_zone"`, `TABLE_KIND = "table"`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_seatnow_layout.py` 맨 아래에 추가:

```python
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
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python -m unittest tests.test_seatnow_layout -v`
Expected: FAIL — `AttributeError: 'LayoutTable' object has no attribute 'kind'`

- [ ] **Step 3: 스키마 상수와 좌석 타입을 넣는다**

`seatnow_layout.py`에서 `SCHEMA_VERSION` 줄을 바꾼다:

```python
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)

TABLE_KIND = "table"
COUNTED_ZONE_KIND = "counted_zone"
VALID_KINDS = (TABLE_KIND, COUNTED_ZONE_KIND)
```

`LayoutChair` 아래에 좌석 칸 타입을 추가한다:

```python
@dataclass(frozen=True)
class LayoutSeat:
    """One hand-drawn seat slot inside a counted_zone (bar counter etc.)."""

    id: int
    box: Box
```

`LayoutTable`에 필드 두 개를 추가한다 (기본값이 있으므로 기존 생성자 호출은 그대로 동작한다):

```python
@dataclass(frozen=True)
class LayoutTable:
    id: int
    name: str
    box: Box
    chairs: Tuple[LayoutChair, ...] = ()
    kind: str = TABLE_KIND
    seats: Tuple[LayoutSeat, ...] = ()
```

`_parse_box` 아래에 포함 검사 헬퍼를 추가한다:

```python
def _box_contains(outer: Box, inner: Box, tolerance: float = 1.0) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )
```

- [ ] **Step 4: 스케일링이 좌석 칸도 따라가게 한다**

`SeatLayout.scaled_to`의 `tables = tuple(...)` 블록을 바꾼다:

```python
        tables = tuple(
            replace(
                table,
                box=scale(table.box),
                chairs=tuple(
                    replace(chair, box=scale(chair.box)) for chair in table.chairs
                ),
                seats=tuple(
                    replace(seat, box=scale(seat.box)) for seat in table.seats
                ),
            )
            for table in self.tables
        )
```

- [ ] **Step 5: 로더가 kind/seats를 파싱·검증하게 한다**

`load_layout`의 버전 검사를 바꾼다:

```python
    if data.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise LayoutError(
            f"Unsupported schema_version {data.get('schema_version')!r} "
            f"(expected one of {SUPPORTED_SCHEMA_VERSIONS}) in {path}"
        )
```

테이블 루프에서 `chairs = tuple(...)` 다음, `tables.append(...)` 앞에 넣는다:

```python
        kind = str(raw.get("kind", TABLE_KIND))
        if kind not in VALID_KINDS:
            raise LayoutError(
                f"table {table_id}: unknown kind {kind!r} "
                f"(expected one of {VALID_KINDS})"
            )
        box = _parse_box(raw.get("box"), f"table {table_id}")
        seats = tuple(
            LayoutSeat(
                id=int(seat.get("id", seat_position)),
                box=_parse_box(
                    seat.get("box"), f"table {table_id} seat #{seat_position}"
                ),
            )
            for seat_position, seat in enumerate(raw.get("seats", []), start=1)
        )
        if kind == COUNTED_ZONE_KIND:
            if not seats:
                raise LayoutError(
                    f"table {table_id}: kind={COUNTED_ZONE_KIND} requires a "
                    f"non-empty 'seats' list (capacity is len(seats))"
                )
            for seat in seats:
                if not _box_contains(box, seat.box):
                    raise LayoutError(
                        f"table {table_id} seat {seat.id}: box {seat.box} is "
                        f"outside the zone box {box}"
                    )
        elif seats:
            raise LayoutError(
                f"table {table_id}: 'seats' is only valid for "
                f"kind={COUNTED_ZONE_KIND}"
            )
```

그리고 `tables.append(...)`를 위에서 만든 값으로 바꾼다:

```python
        tables.append(
            LayoutTable(
                id=table_id,
                name=str(raw.get("name", f"T{table_id}")),
                box=box,
                chairs=chairs,
                kind=kind,
                seats=seats,
            )
        )
```

`load_layout` 끝의 `SeatLayout(...)`은 항상 최신 버전으로 승격시킨다. v1을 읽어 저장할 때 v1 헤더에 v2 필드가 섞이는 걸 막는다:

```python
    return SeatLayout(
        schema_version=SCHEMA_VERSION,
        source=dict(data.get("source", {})),
        tables=tuple(tables),
    )
```

- [ ] **Step 6: 저장이 kind/seats를 내보내게 한다**

`save_layout`의 `payload["tables"]`를 바꾼다:

```python
        "tables": [
            {
                "id": table.id,
                "name": table.name,
                "kind": table.kind,
                "box": [round(v, 2) for v in table.box],
                "chairs": [
                    {"id": chair.id, "box": [round(v, 2) for v in chair.box]}
                    for chair in table.chairs
                ],
                "seats": [
                    {"id": seat.id, "box": [round(v, 2) for v in seat.box]}
                    for seat in table.seats
                ],
            }
            for table in layout.tables
        ],
```

- [ ] **Step 7: 테스트가 통과하는지 확인한다**

Run: `python -m unittest tests.test_seatnow_layout -v`
Expected: PASS (기존 테스트 포함 전부)

- [ ] **Step 8: 기존 레이아웃 파일이 그대로 열리는지 확인한다**

Run:
```bash
python -c "
from pathlib import Path
from seatnow_layout import load_layout
for p in sorted(Path('layouts').glob('*.json')):
    lay = load_layout(p)
    print(p.name, len(lay.tables), [t.kind for t in lay.tables])
"
```
Expected: 두 파일 모두 로드되고 `kind`가 전부 `table`

- [ ] **Step 9: 커밋**

```bash
git add seatnow_layout.py tests/test_seatnow_layout.py
git commit -m "feat: 레이아웃 스키마 v2 - counted_zone과 좌석 칸(seats)"
```

---

### Task 2: 판정 단위 평탄화 `judgement_units()`

바 구역 하나를 좌석 칸 N개의 독립 판정 단위로 펼친다. 이게 이 계획의 핵심 단순화다 — 이후 증거 연결·디바운싱이 칸마다 공짜로 돌아간다.

**Files:**
- Modify: `seatnow_layout.py`
- Test: `tests/test_seatnow_layout.py`

**Interfaces:**
- Consumes: Task 1의 `LayoutSeat`, `LayoutTable.kind`, `LayoutTable.seats`, `COUNTED_ZONE_KIND`, `TABLE_KIND`
- Produces:
  - `JudgementUnit(box: Box, unit_id: int, name: str, kind: str, capacity: int, zone_id: Optional[int], zone_name: Optional[str], seat_id: Optional[int])`
  - `SeatLayout.judgement_units() -> Tuple[JudgementUnit, ...]`
  - `SeatLayout.unit_chair_assignments() -> Dict[int, List[int]]`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_seatnow_layout.py`에 추가:

```python
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
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python -m unittest tests.test_seatnow_layout.JudgementUnitTests -v`
Expected: FAIL — `AttributeError: 'SeatLayout' object has no attribute 'judgement_units'`

- [ ] **Step 3: `typing` import에 `Optional`을 더한다**

`seatnow_layout.py` 상단:

```python
from typing import Dict, List, Optional, Tuple
```

- [ ] **Step 4: `JudgementUnit`을 정의한다**

`LayoutSeat` 아래에 추가:

```python
@dataclass(frozen=True)
class JudgementUnit:
    """One box the pipeline judges independently.

    A plain table is one unit.  A counted_zone becomes one unit per
    hand-drawn seat slot, so the existing evidence association and
    debouncing run per seat without a separate counting code path.
    """

    box: Box
    unit_id: int
    name: str
    kind: str
    capacity: int
    zone_id: Optional[int] = None
    zone_name: Optional[str] = None
    seat_id: Optional[int] = None
```

- [ ] **Step 5: `SeatLayout`에 메서드 두 개를 더한다**

`SeatLayout.chair_assignments` 아래에 추가:

```python
    def judgement_units(self) -> Tuple[JudgementUnit, ...]:
        units: List[JudgementUnit] = []
        for table in self.tables:
            if table.kind == COUNTED_ZONE_KIND:
                capacity = len(table.seats)
                for seat in table.seats:
                    units.append(
                        JudgementUnit(
                            box=seat.box,
                            unit_id=len(units) + 1,
                            name=f"{table.name}-{seat.id}",
                            kind=COUNTED_ZONE_KIND,
                            capacity=capacity,
                            zone_id=table.id,
                            zone_name=table.name,
                            seat_id=seat.id,
                        )
                    )
            else:
                units.append(
                    JudgementUnit(
                        box=table.box,
                        unit_id=len(units) + 1,
                        name=table.name,
                        kind=TABLE_KIND,
                        capacity=1,
                    )
                )
        return tuple(units)

    def unit_chair_assignments(self) -> Dict[int, List[int]]:
        """Chair indices per judgement-unit index.

        Chairs belong to plain tables only; counted_zone seats get an empty
        list so downstream indexing stays uniform.
        """
        assignments: Dict[int, List[int]] = {}
        unit_index = 0
        chair_cursor = 0
        for table in self.tables:
            if table.kind == COUNTED_ZONE_KIND:
                for _ in table.seats:
                    assignments[unit_index] = []
                    unit_index += 1
                continue
            count = len(table.chairs)
            assignments[unit_index] = list(range(chair_cursor, chair_cursor + count))
            chair_cursor += count
            unit_index += 1
        return assignments
```

- [ ] **Step 6: 테스트가 통과하는지 확인한다**

Run: `python -m unittest tests.test_seatnow_layout -v`
Expected: PASS

- [ ] **Step 7: 커밋**

```bash
git add seatnow_layout.py tests/test_seatnow_layout.py
git commit -m "feat: 바 구역을 좌석 칸 단위 판정 단위로 평탄화"
```

---

### Task 3: `analyze()`가 판정 단위를 쓰도록 전환

**Files:**
- Modify: `seatnow_core.py:1343-1345`, `:1478-1482`, `:1636-1645`
- Test: `tests/test_analyze_pipeline.py`

**Interfaces:**
- Consumes: Task 2의 `SeatLayout.judgement_units()`, `SeatLayout.unit_chair_assignments()`, `JudgementUnit`
- Produces: `TableObservation.layout_kind: str`, `TableObservation.layout_zone_id: Optional[int]`, `TableObservation.layout_zone_name: Optional[str]`, `TableObservation.layout_capacity: int` — 바 구역 좌석 칸마다 별도의 `TableObservation`이 나온다

**이 태스크가 공짜로 얻는 것 (스펙 §4.3~4.4):** 칸이 독립된 관측이 되므로
디바운싱(점유 2회 / 이탈 3회)이 칸마다 기존 트래커 machinery로 그대로 적용된다.
서 있는 사람을 세지 않는 것도 기존 `occupancy_state_from_evidence`가 이미
`direct_seated_people`만 점유 근거로 쓰기 때문에 자동으로 따라온다. 둘 다 새 코드가
필요 없고, 새 테스트도 만들지 않는다 — 기존 트래커·증거 테스트가 이미 덮는다.

> **기존 테스트가 깨진다 — Step 1에서 같이 고친다.** `tests/test_analyze_pipeline.py`의 `LayoutChairPipelineTests._FakeLayout`은 `chair_boxes()`와 `chair_assignments()`만 갖고 있다. `analyze()`가 `judgement_units()` / `unit_chair_assignments()`를 부르기 시작하면 그 클래스가 `AttributeError`로 깨진다. 가짜 레이아웃을 진짜 `SeatLayout`으로 바꿔서 앞으로도 드리프트가 안 생기게 한다.

이 파일의 실제 헬퍼는 다음과 같다 (그대로 쓴다):

- `build_analyzer(detections, poses=(), keypoints=(), layout=None, **config_kwargs)` — `detections`/`poses`는 `(class_name, box, confidence)` 튜플 리스트
- `seated_keypoints(hip, knee, ankle, shoulder)` — 앉은 자세로 읽히는 COCO 17 키포인트
- `FRAME` — 모듈 레벨 `np.zeros((720, 1280, 3), dtype=np.uint8)`
- 호출은 `analyzer.analyze(FRAME)`

- [ ] **Step 1: 기존 `_FakeLayout`을 진짜 `SeatLayout`으로 바꾼다**

`tests/test_analyze_pipeline.py`의 `LayoutChairPipelineTests`에서 `_FakeLayout` 클래스를 통째로 지우고 `_layout`을 바꾼다:

```python
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
```

나머지 두 테스트 본문은 그대로 둔다.

- [ ] **Step 2: 실패하는 테스트를 쓴다**

`tests/test_analyze_pipeline.py`에 추가한다. `SeatLayout`, `LayoutSeat`, `LayoutTable`을 쓰므로 JSON 파일을 거치지 않고 객체로 직접 만든다:

```python
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
```

- [ ] **Step 3: 테스트가 실패하는지 확인한다**

Run: `python -m unittest tests.test_analyze_pipeline.CountedZoneAnalyzeTests -v`
Expected: FAIL — 관측이 1개만 나오거나 `AttributeError: 'TableObservation' object has no attribute 'layout_kind'`

- [ ] **Step 4: `TableObservation`에 필드를 더한다**

`seatnow_core.py`의 `TableObservation`에서 `layout_name: Optional[str] = None` 아래에 추가:

```python
    layout_kind: str = "table"
    layout_zone_id: Optional[int] = None
    layout_zone_name: Optional[str] = None
    layout_capacity: int = 1
```

- [ ] **Step 5: import를 더한다**

`seatnow_core.py` 상단 import 블록에 추가한다. `seatnow_layout.py`는 `seatnow_core`를 import하지 않으므로 순환 import가 아니다:

```python
from seatnow_layout import COUNTED_ZONE_KIND, JudgementUnit
```

- [ ] **Step 6: 레이아웃 존 구성이 판정 단위를 쓰게 바꾼다**

`analyze()` 안에서 `if self.layout is not None:` (`seatnow_core.py:1343`) 블록보다 **앞**에 기본값을 둔다:

```python
        layout_units: Tuple[JudgementUnit, ...] = ()
```

그리고 `seatnow_core.py:1343-1345`를 바꾼다:

```python
        if self.layout is not None:
            layout_units = self.layout.judgement_units()
            table_boxes = [unit.box for unit in layout_units]
            chair_boxes = self.layout.chair_boxes()
```

- [ ] **Step 7: 의자 연결이 단위 인덱스를 쓰게 바꾼다**

`seatnow_core.py:1478-1482`를 바꾼다:

```python
        if self.layout is not None:
            # A hand-drawn chair->table link is ground truth: propagate it
            # without the geometric strong-link filter.
            chair_table_assignments = self.layout.unit_chair_assignments()
            strong_chair_assignments = chair_table_assignments
```

- [ ] **Step 8: 관측 생성이 단위 메타데이터를 싣게 바꾼다**

`seatnow_core.py:1636-1645`의 `source=` / `layout_id=` / `layout_name=` 블록을 바꾼다:

```python
                    source="layout" if self.layout is not None else "detected",
                    layout_id=(
                        layout_units[index].unit_id
                        if self.layout is not None
                        else None
                    ),
                    layout_name=(
                        layout_units[index].name
                        if self.layout is not None
                        else None
                    ),
                    layout_kind=(
                        layout_units[index].kind
                        if self.layout is not None
                        else "table"
                    ),
                    layout_zone_id=(
                        layout_units[index].zone_id
                        if self.layout is not None
                        else None
                    ),
                    layout_zone_name=(
                        layout_units[index].zone_name
                        if self.layout is not None
                        else None
                    ),
                    layout_capacity=(
                        layout_units[index].capacity
                        if self.layout is not None
                        else 1
                    ),
```

- [ ] **Step 9: 테스트가 통과하는지 확인한다**

Run: `python -m unittest tests.test_analyze_pipeline -v`
Expected: PASS — `LayoutChairPipelineTests` 두 개도 계속 통과해야 한다

- [ ] **Step 10: 회귀가 없는지 전체 테스트를 돌린다**

Run: `python -m unittest discover tests`
Expected: `OK` — 기존 171개 포함 전부 통과

- [ ] **Step 11: 커밋**

```bash
git add seatnow_core.py tests/test_analyze_pipeline.py
git commit -m "feat: analyze()가 레이아웃 판정 단위(좌석 칸)를 쓰도록 전환"
```

---

### Task 4: 두 칸에 걸친 사람은 `unknown`

"두 사람이 붙어 앉으면 1명으로 센다"를 막는 규칙. 스펙 §4.3.

**Files:**
- Modify: `seatnow_core.py`
- Test: `tests/test_analyze_pipeline.py`

**Interfaces:**
- Consumes: Task 3의 `TableObservation.layout_kind` / `layout_zone_id`, `layout_units`
- Produces: `demote_seats_spanned_by_one_person(observations, units, poses, minimum_overlap=0.20) -> None` (제자리 수정), 해당 칸에 `raw_state=UNKNOWN` + `reason="spans_multiple_seats"`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_analyze_pipeline.py`의 `CountedZoneAnalyzeTests`에 추가:

```python
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
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python -m unittest tests.test_analyze_pipeline.CountedZoneAnalyzeTests -v`
Expected: FAIL — 첫 테스트에서 `OccupancyState.OCCUPIED != OccupancyState.UNKNOWN`

- [ ] **Step 3: 규칙 함수를 넣는다**

`seatnow_core.py`의 `occupancy_state_from_evidence` 함수 아래에 모듈 레벨 함수를 추가한다:

```python
def demote_seats_spanned_by_one_person(
    observations: List[TableObservation],
    units: Sequence[JudgementUnit],
    poses: Sequence[PoseObservation],
    minimum_overlap: float = 0.20,
) -> None:
    """Mark counted_zone seats UNKNOWN when one person box covers 2+ of them.

    A single wide detection over two hand-drawn seat slots is ambiguous: it
    may be one customer or two sitting shoulder to shoulder.  Reporting it
    as one occupied seat would inflate the free-seat count, so both slots
    become UNKNOWN instead (spec section 4.3).
    """
    seat_indices_by_zone: Dict[int, List[int]] = {}
    for index, unit in enumerate(units):
        if unit.kind == COUNTED_ZONE_KIND and unit.zone_id is not None:
            seat_indices_by_zone.setdefault(unit.zone_id, []).append(index)
    if not seat_indices_by_zone:
        return

    for pose in poses:
        if pose.state == PoseState.STANDING:
            continue
        for indices in seat_indices_by_zone.values():
            spanned = [
                index
                for index in indices
                if index < len(observations)
                and overlap_over_smaller(units[index].box, pose.box) >= minimum_overlap
            ]
            if len(spanned) < 2:
                continue
            for index in spanned:
                observations[index].raw_state = OccupancyState.UNKNOWN
                observations[index].reason = "spans_multiple_seats"
                observations[index].provisional = False
```

- [ ] **Step 4: `analyze()`에서 호출한다**

`analyze()`의 관측 루프가 끝난 직후 — `if self.config.infer_occluded_tables and self.layout is None:` (`seatnow_core.py:1653` 부근) 바로 **앞** — 에 넣는다:

```python
        if self.layout is not None:
            demote_seats_spanned_by_one_person(observations, layout_units, poses)
```

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

Run: `python -m unittest tests.test_analyze_pipeline -v`
Expected: PASS

- [ ] **Step 6: 전체 테스트**

Run: `python -m unittest discover tests`
Expected: `OK`

- [ ] **Step 7: 커밋**

```bash
git add seatnow_core.py tests/test_analyze_pipeline.py
git commit -m "feat: 좌석 칸 두 개에 걸친 사람은 unknown 처리"
```

---

### Task 5: `ReasonCode` — 사유 코드 정규화와 승격

**Files:**
- Create: `seatnow_report.py`
- Create: `tests/test_seatnow_report.py`
- Modify: `seatnow_core.py:1611-1615`
- Test: `tests/test_analyze_pipeline.py`

**Interfaces:**
- Consumes: Task 4의 `reason="spans_multiple_seats"`, 기존 자유 문자열 `reason`
- Produces: `ReasonCode(str, Enum)`, `REASON_GROUPS: Dict[str, Tuple[ReasonCode, ...]]`, `ACTIONABLE_GROUPS: Tuple[str, ...]`, `classify_reason(raw_state: str, reason: str, predicted: bool) -> ReasonCode`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_seatnow_report.py` 신규 파일:

```python
"""Unit tests for the app-facing seat report contract."""

from __future__ import annotations

import unittest

from seatnow_report import ACTIONABLE_GROUPS, REASON_GROUPS, ReasonCode, classify_reason


class ClassifyReasonTests(unittest.TestCase):
    def test_occluded_lower_body_is_promoted_from_pose_reason(self):
        self.assertEqual(
            classify_reason("unknown", "compact_occluded_pose=0.82", predicted=False),
            ReasonCode.OCCLUDED_LOWER_BODY,
        )

    def test_low_keypoints_is_promoted_from_pose_reason(self):
        self.assertEqual(
            classify_reason("unknown", "insufficient_keypoints", predicted=False),
            ReasonCode.POSE_LOW_KEYPOINTS,
        )

    def test_unpromoted_pose_unknown_falls_back_to_ambiguous(self):
        self.assertEqual(
            classify_reason("unknown", "nearby_person_pose_unknown", predicted=False),
            ReasonCode.AMBIGUOUS_ASSOCIATION,
        )

    def test_promoted_pose_cause_is_unwrapped(self):
        self.assertEqual(
            classify_reason(
                "unknown",
                "nearby_person_pose_unknown:compact_occluded_pose=0.82",
                predicted=False,
            ),
            ReasonCode.OCCLUDED_LOWER_BODY,
        )

    def test_spanning_seats_has_its_own_code(self):
        self.assertEqual(
            classify_reason("unknown", "spans_multiple_seats", predicted=False),
            ReasonCode.SPANS_MULTIPLE_SEATS,
        )

    def test_predicted_track_is_time_group_not_an_engineering_problem(self):
        code = classify_reason("occupied", "seated:1", predicted=True)

        self.assertEqual(code, ReasonCode.TRACK_PREDICTED)
        self.assertIn(code, REASON_GROUPS["time"])

    def test_border_cropped_is_an_install_problem(self):
        code = classify_reason("ignore", "border_cropped", predicted=False)

        self.assertEqual(code, ReasonCode.BORDER_CROPPED)
        self.assertIn(code, REASON_GROUPS["install"])

    def test_occupied_reasons_are_classified(self):
        self.assertEqual(
            classify_reason("occupied", "seated:2", predicted=False),
            ReasonCode.PERSON_SEATED,
        )
        self.assertEqual(
            classify_reason("occupied", "objects:cup,laptop", predicted=False),
            ReasonCode.BELONGINGS,
        )
        self.assertEqual(
            classify_reason("occupied", "occupied_chairs:1", predicted=False),
            ReasonCode.OCCUPIED_CHAIR,
        )

    def test_empty_is_classified(self):
        self.assertEqual(
            classify_reason("empty", "no_customer_evidence", predicted=False),
            ReasonCode.NO_CUSTOMER_EVIDENCE,
        )

    def test_every_code_belongs_to_exactly_one_group(self):
        seen = [code for codes in REASON_GROUPS.values() for code in codes]

        self.assertEqual(sorted(seen), sorted(set(seen)))
        self.assertEqual(set(seen), set(ReasonCode))

    def test_actionable_groups_exclude_install_and_settled(self):
        self.assertEqual(ACTIONABLE_GROUPS, ("geometry", "model"))
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python -m unittest tests.test_seatnow_report -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'seatnow_report'`

- [ ] **Step 3: `seatnow_report.py`를 만든다**

```python
"""App-facing seat availability report.

Pure formatting over already-validated tracker output: dict in, dict out.
It deliberately raises rather than swallowing errors -- a failure here is a
bug, not an operating condition (spec section 7).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Sequence, Tuple


class ReasonCode(str, Enum):
    """Why a seat ended up in the state it did.

    Grouped by what fixes them (spec section 5.2), so the distribution
    doubles as the improvement backlog.
    """

    # install -- fixed by moving the camera, never by code (CLAUDE.md)
    BORDER_CROPPED = "border_cropped"
    SCENE_CHANGE = "scene_change"

    # geometry / occlusion -- fixed by rescue paths or redrawn seat slots
    OCCLUDED_LOWER_BODY = "occluded_lower_body"
    AMBIGUOUS_ASSOCIATION = "ambiguous_association"
    SPANS_MULTIPLE_SEATS = "spans_multiple_seats"

    # model -- fixed by fine-tuning or a larger imgsz
    POSE_LOW_KEYPOINTS = "pose_low_keypoints"
    TABLE_NOT_DETECTED = "table_not_detected"

    # time -- fixed by waiting; nothing to do
    TRACK_PREDICTED = "track_predicted"
    PENDING_CONFIRMATION = "pending_confirmation"

    # settled judgements
    PERSON_SEATED = "person_seated"
    BELONGINGS = "belongings"
    OCCUPIED_CHAIR = "occupied_chair"
    NO_CUSTOMER_EVIDENCE = "no_customer_evidence"


REASON_GROUPS: Dict[str, Tuple[ReasonCode, ...]] = {
    "install": (ReasonCode.BORDER_CROPPED, ReasonCode.SCENE_CHANGE),
    "geometry": (
        ReasonCode.OCCLUDED_LOWER_BODY,
        ReasonCode.AMBIGUOUS_ASSOCIATION,
        ReasonCode.SPANS_MULTIPLE_SEATS,
    ),
    "model": (ReasonCode.POSE_LOW_KEYPOINTS, ReasonCode.TABLE_NOT_DETECTED),
    "time": (ReasonCode.TRACK_PREDICTED, ReasonCode.PENDING_CONFIRMATION),
    "settled": (
        ReasonCode.PERSON_SEATED,
        ReasonCode.BELONGINGS,
        ReasonCode.OCCUPIED_CHAIR,
        ReasonCode.NO_CUSTOMER_EVIDENCE,
    ),
}

# Improvement targets: "install" is the camera's job and the other groups
# need no action, so only these two are engineering backlog.
ACTIONABLE_GROUPS: Tuple[str, ...] = ("geometry", "model")


def classify_reason(raw_state: str, reason: str, predicted: bool) -> ReasonCode:
    """Map a free-text observation reason onto the closed reason vocabulary.

    ``predicted`` wins over the reason text: a predicted track carries the
    reason of its last real observation, which would otherwise be reported
    as settled evidence that is not actually being seen right now.
    """
    if predicted:
        return ReasonCode.TRACK_PREDICTED

    text = reason or ""
    if text.startswith("temporarily_occluded:"):
        return ReasonCode.TRACK_PREDICTED
    # The table layer prefixes the pose-level cause; unwrap it so the cause
    # is what gets classified (spec section 5.1).
    if text.startswith("nearby_person_pose_unknown:"):
        text = text.split(":", 1)[1]

    if text.startswith("compact_occluded_pose"):
        return ReasonCode.OCCLUDED_LOWER_BODY
    if text.startswith("insufficient_keypoints"):
        return ReasonCode.POSE_LOW_KEYPOINTS
    if text.startswith("spans_multiple_seats"):
        return ReasonCode.SPANS_MULTIPLE_SEATS
    if text.startswith("border_cropped"):
        return ReasonCode.BORDER_CROPPED
    if text.startswith("scene_change"):
        return ReasonCode.SCENE_CHANGE
    if text.startswith("nearby_person_pose_unknown"):
        return ReasonCode.AMBIGUOUS_ASSOCIATION
    if text.startswith("no_customer_evidence"):
        return ReasonCode.NO_CUSTOMER_EVIDENCE

    if raw_state == "occupied":
        if "seated:" in text:
            return ReasonCode.PERSON_SEATED
        if "objects:" in text or "chair_objects:" in text:
            return ReasonCode.BELONGINGS
        if "occupied_chairs:" in text:
            return ReasonCode.OCCUPIED_CHAIR
        return ReasonCode.PERSON_SEATED
    if raw_state == "empty":
        return ReasonCode.NO_CUSTOMER_EVIDENCE
    if raw_state == "ignore":
        return ReasonCode.BORDER_CROPPED
    return ReasonCode.AMBIGUOUS_ASSOCIATION
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `python -m unittest tests.test_seatnow_report -v`
Expected: PASS

- [ ] **Step 5: 코어가 포즈 원인을 버리지 않는다는 테스트를 쓴다**

`tests/test_analyze_pipeline.py`에 추가한다. `OccludedSeatPipelineTests`가 쓰는 상반신-only 키포인트를 그대로 빌려 UNKNOWN 포즈를 만든다 — 의자를 넣지 않으므로 `inferred-seat` 경로 대신 테이블 연결 경로를 탄다:

```python
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
```

> 사유 문자열은 `compact_occluded_pose=<비율>;seat_support=<점수>` 형태라 정확히 비교하지 않고 접두사로 확인한다.

- [ ] **Step 6: 테스트가 실패하는지 확인한다**

Run: `python -m unittest tests.test_analyze_pipeline.UnknownReasonPromotionTests -v`
Expected: FAIL — `'nearby_person_pose_unknown' != 'nearby_person_pose_unknown:compact_occluded_pose=0.82'`

- [ ] **Step 7: 사유를 승격시킨다**

`seatnow_core.py:1611-1615`를 바꾼다:

```python
            elif evidence_state == OccupancyState.UNKNOWN:
                state = OccupancyState.UNKNOWN
                score = max(person.confidence for person in assigned_unknown_people)
                # Keep the pose-level cause.  Collapsing every case into one
                # string is why the UNKNOWN rate could not be attacked.
                cause = next(
                    (
                        person.reason
                        for person in assigned_unknown_people
                        if person.reason
                    ),
                    "",
                )
                reason = (
                    f"nearby_person_pose_unknown:{cause}"
                    if cause
                    else "nearby_person_pose_unknown"
                )
                provisional = False
```

- [ ] **Step 8: 테스트가 통과하는지 확인한다**

Run: `python -m unittest tests.test_analyze_pipeline tests.test_seatnow_report -v`
Expected: PASS

- [ ] **Step 9: 전체 테스트**

Run: `python -m unittest discover tests`
Expected: `OK`

- [ ] **Step 10: 커밋**

```bash
git add seatnow_report.py tests/test_seatnow_report.py seatnow_core.py tests/test_analyze_pipeline.py
git commit -m "feat: UNKNOWN 사유 코드 체계 + 포즈 원인 승격"
```

---

### Task 6: `build_seat_report()`

**Files:**
- Modify: `seatnow_report.py`
- Test: `tests/test_seatnow_report.py`

**Interfaces:**
- Consumes: Task 5의 `classify_reason`, `ReasonCode`; Task 3의 메타데이터가 실린 `track_to_dict()` 출력 형태
- Produces: `SCHEMA_VERSION = 1`, `build_seat_report(tables: Sequence[Dict[str, Any]], tick_at: float) -> Dict[str, Any]`

> **설계 메모:** 입력을 `Track` 객체가 아니라 **이미 직렬화된 dict**로 받는다. 그래야 `seatnow_report.py`가 `seatnow_core`를 import하지 않고 (한 방향 의존), 테스트가 모델·dataclass 없이 순수 dict로 돈다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_seatnow_report.py`에 추가. import 줄에 `build_seat_report`를 더한다:

```python
from seatnow_report import (
    ACTIONABLE_GROUPS,
    REASON_GROUPS,
    ReasonCode,
    build_seat_report,
    classify_reason,
)


def table_dict(
    name,
    state,
    reason="no_customer_evidence",
    kind="table",
    zone_name=None,
    capacity=1,
    predicted=False,
    confidence=0.9,
):
    return {
        "layout_name": name,
        "label": name,
        "state": state,
        "raw_state": state,
        "reason": reason,
        "layout_kind": kind,
        "layout_zone_name": zone_name,
        "layout_capacity": capacity,
        "predicted": predicted,
        "confidence": confidence,
    }


class BuildSeatReportTests(unittest.TestCase):
    def test_plain_tables_become_one_seat_each(self):
        report = build_seat_report(
            [
                table_dict("창가1", "occupied", "seated:1"),
                table_dict("창가2", "empty"),
            ],
            tick_at=12.5,
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["tick_at"], 12.5)
        self.assertEqual(len(report["seats"]), 2)
        self.assertEqual(report["seats"][0]["seat_id"], "창가1")
        self.assertEqual(report["seats"][0]["state"], "occupied")
        self.assertEqual(report["seats"][0]["reason_code"], "person_seated")
        self.assertEqual(
            report["totals"],
            {"capacity": 2, "occupied": 1, "free": 1, "unknown": 0},
        )

    def test_counted_zone_seats_are_grouped_into_one_entry(self):
        report = build_seat_report(
            [
                table_dict("BAR-1", "occupied", "seated:1", "counted_zone", "BAR", 3),
                table_dict(
                    "BAR-2", "empty", "no_customer_evidence", "counted_zone", "BAR", 3
                ),
                table_dict(
                    "BAR-3", "unknown", "spans_multiple_seats", "counted_zone", "BAR", 3
                ),
            ],
            tick_at=0.0,
        )

        self.assertEqual(len(report["seats"]), 1)
        zone = report["seats"][0]
        self.assertEqual(zone["seat_id"], "BAR")
        self.assertEqual(zone["kind"], "counted_zone")
        self.assertEqual(zone["capacity"], 3)
        self.assertEqual(zone["occupied"], 1)
        self.assertEqual(zone["free"], 1)
        self.assertEqual(zone["unknown"], 1)
        self.assertEqual(zone["reason_codes"], {"spans_multiple_seats": 1})

    def test_free_never_counts_unknown(self):
        report = build_seat_report(
            [
                table_dict("T1", "unknown", "compact_occluded_pose=0.9"),
                table_dict("T2", "unknown", "insufficient_keypoints"),
                table_dict("T3", "empty"),
            ],
            tick_at=0.0,
        )

        self.assertEqual(report["totals"]["free"], 1)
        self.assertEqual(report["totals"]["unknown"], 2)

    def test_ignore_state_is_excluded_from_capacity(self):
        report = build_seat_report(
            [
                table_dict("T1", "empty"),
                table_dict("T2", "ignore", "border_cropped"),
            ],
            tick_at=0.0,
        )

        self.assertEqual(report["totals"]["capacity"], 1)
        self.assertEqual(len(report["seats"]), 1)

    def test_predicted_track_reports_time_group_reason(self):
        report = build_seat_report(
            [table_dict("T1", "occupied", "seated:1", predicted=True)],
            tick_at=0.0,
        )

        self.assertEqual(report["seats"][0]["reason_code"], "track_predicted")

    def test_zone_parts_always_sum_to_capacity(self):
        report = build_seat_report(
            [
                table_dict("BAR-1", "occupied", "seated:1", "counted_zone", "BAR", 2),
                table_dict("BAR-2", "occupied", "seated:1", "counted_zone", "BAR", 2),
            ],
            tick_at=0.0,
        )

        zone = report["seats"][0]
        self.assertEqual(
            zone["occupied"] + zone["free"] + zone["unknown"], zone["capacity"]
        )

    def test_two_zones_stay_separate_and_keep_input_order(self):
        report = build_seat_report(
            [
                table_dict("BAR-1", "occupied", "seated:1", "counted_zone", "BAR", 1),
                table_dict("WALL-1", "empty", "no_customer_evidence", "counted_zone", "WALL", 1),
            ],
            tick_at=0.0,
        )

        self.assertEqual([seat["seat_id"] for seat in report["seats"]], ["BAR", "WALL"])
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python -m unittest tests.test_seatnow_report.BuildSeatReportTests -v`
Expected: FAIL — `ImportError: cannot import name 'build_seat_report'`

- [ ] **Step 3: 구현을 넣는다**

`seatnow_report.py` 하단에 추가:

```python
SCHEMA_VERSION = 1

_COUNTABLE_STATES = ("occupied", "empty", "unknown")


def _seat_name(table: Dict[str, Any]) -> str:
    return str(table.get("layout_name") or table.get("label") or "?")


def build_seat_report(
    tables: Sequence[Dict[str, Any]], tick_at: float
) -> Dict[str, Any]:
    """Turn tracker output into the app-facing availability contract.

    ``IGNORE`` seats are dropped entirely: they are an installation defect,
    not a seat the app should reason about (spec section 2).  ``free``
    counts only confirmed empties -- UNKNOWN is never rounded into
    availability.
    """
    plain: List[Dict[str, Any]] = []
    zones: Dict[str, Dict[str, Any]] = {}
    zone_order: List[str] = []

    for table in tables:
        state = str(table.get("state", "unknown"))
        if state not in _COUNTABLE_STATES:
            continue
        code = classify_reason(
            str(table.get("raw_state", state)),
            str(table.get("reason", "")),
            bool(table.get("predicted", False)),
        )
        if str(table.get("layout_kind", "table")) != "counted_zone":
            plain.append(
                {
                    "seat_id": _seat_name(table),
                    "kind": "table",
                    "capacity": 1,
                    "state": state,
                    "reason_code": code.value,
                    "confidence": round(float(table.get("confidence", 0.0)), 4),
                }
            )
            continue

        zone_name = str(table.get("layout_zone_name") or "?")
        if zone_name not in zones:
            zones[zone_name] = {
                "seat_id": zone_name,
                "kind": "counted_zone",
                "capacity": 0,
                "occupied": 0,
                "free": 0,
                "unknown": 0,
                "reason_codes": {},
            }
            zone_order.append(zone_name)
        zone = zones[zone_name]
        zone["capacity"] += 1
        if state == "occupied":
            zone["occupied"] += 1
        elif state == "empty":
            zone["free"] += 1
        else:
            zone["unknown"] += 1
            counts = zone["reason_codes"]
            counts[code.value] = counts.get(code.value, 0) + 1

    seats: List[Dict[str, Any]] = plain + [zones[name] for name in zone_order]

    totals = {"capacity": 0, "occupied": 0, "free": 0, "unknown": 0}
    for seat in seats:
        if seat["kind"] == "counted_zone":
            totals["capacity"] += seat["capacity"]
            totals["occupied"] += seat["occupied"]
            totals["free"] += seat["free"]
            totals["unknown"] += seat["unknown"]
            continue
        totals["capacity"] += 1
        if seat["state"] == "empty":
            totals["free"] += 1
        elif seat["state"] == "occupied":
            totals["occupied"] += 1
        else:
            totals["unknown"] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "tick_at": round(float(tick_at), 6),
        "seats": seats,
        "totals": totals,
    }
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `python -m unittest tests.test_seatnow_report -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add seatnow_report.py tests/test_seatnow_report.py
git commit -m "feat: build_seat_report() - 앱용 좌석 가용성 계약"
```

---

### Task 7: JSONL에 `seat_report` 싣기

**Files:**
- Modify: `seatnow_core.py` (`track_to_dict`, `frame_log_record`)
- Test: `tests/test_seatnow_core.py`

**Interfaces:**
- Consumes: Task 6의 `build_seat_report`
- Produces: `frame_log_record(...)` 출력에 `record["seat_report"]`; `track_to_dict(...)`에 `layout_kind` / `layout_zone_id` / `layout_zone_name` / `layout_capacity`

> **먼저 읽을 것:** `tests/test_seatnow_core.py`에서 `record = frame_log_record(7, analysis, update)`를 호출하는 기존 테스트(`record["layout_id"]`를 assert하는 곳, 파일 700번대)를 연다. 그 테스트는 `TableObservation` / `Track` / `TrackerUpdate` / `FrameAnalysis`를 직접 조립한다. 아래 새 테스트는 같은 클래스에 넣고 **그 조립 코드를 그대로 복사해서** 앞부분을 만든다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_seatnow_core.py`의 그 클래스에 추가한다. `<기존 테스트의 조립 코드>` 자리에는 위에서 연 테스트의 `observations` / `visible_tracks` / `update` / `analysis` 생성 부분을 그대로 붙여넣는다:

```python
    def test_frame_log_record_carries_seat_report(self):
        observation = TableObservation(
            box=(100.0, 100.0, 300.0, 200.0),
            table_confidence=0.8,
            raw_state=OccupancyState.EMPTY,
            raw_score=0.8,
            source="layout",
            reason="no_customer_evidence",
            layout_id=1,
            layout_name="창가1",
        )
        track = Track(
            track_id=1,
            box=observation.box,
            stable_state=OccupancyState.EMPTY,
            last_observation=observation,
            first_seen=0.0,
            last_seen=0.0,
        )
        update = TrackerUpdate(visible_tracks=[track], all_tracks=[track], events=[])
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
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python -m unittest tests.test_seatnow_core -v`
Expected: FAIL — `KeyError: 'seat_report'`

- [ ] **Step 3: `track_to_dict`가 새 메타데이터를 싣게 한다**

`seatnow_core.py`의 `track_to_dict` 안, `"layout_name": observation.layout_name,` 아래에 추가:

```python
        "layout_kind": observation.layout_kind,
        "layout_zone_id": observation.layout_zone_id,
        "layout_zone_name": observation.layout_zone_name,
        "layout_capacity": observation.layout_capacity,
```

- [ ] **Step 4: import를 더한다**

`seatnow_core.py` 상단:

```python
from seatnow_report import build_seat_report
```

- [ ] **Step 5: `frame_log_record`에 연결한다**

`frame_log_record` 안에서 `visible = update.visible_tracks` 바로 다음 줄에 추가한다:

```python
    table_dicts = [track_to_dict(track) for track in visible]
```

`record` dict 안의 `"tables": [track_to_dict(track) for track in visible],` 를 바꾼다:

```python
        "tables": table_dicts,
```

`record` 조립이 끝난 뒤, `if include_raw_detections:` 줄 **앞**에 추가한다:

```python
    record["seat_report"] = build_seat_report(table_dicts, analysis.timestamp)
```

- [ ] **Step 6: 테스트가 통과하는지 확인한다**

Run: `python -m unittest tests.test_seatnow_core -v`
Expected: PASS

- [ ] **Step 7: 전체 테스트**

Run: `python -m unittest discover tests`
Expected: `OK`

- [ ] **Step 8: 커밋**

```bash
git add seatnow_core.py tests/test_seatnow_core.py
git commit -m "feat: JSONL 매 tick에 seat_report 기록"
```

---

### Task 8: `calibrate.py` — 바 구역에 칸 긋기

설치 담당자가 바 전체에 네모를 치고 그 안에 자리마다 칸을 긋는다. 스펙 §4.2.

**Files:**
- Modify: `calibrate.py`
- Test: `tests/test_calibrate_state.py`

**Interfaces:**
- Consumes: Task 1의 `LayoutSeat`, `COUNTED_ZONE_KIND`
- Produces: `CalibrationState.add_zone(box)`, `CalibrationState.add_seat(box) -> bool`; 상태 dict가 `{"box", "chairs", "kind", "seats"}` 네 키를 갖는다

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_calibrate_state.py`에 추가:

```python
class CountedZoneCalibrationTests(unittest.TestCase):
    def test_add_zone_then_seats(self):
        state = CalibrationState()
        state.add_zone((100.0, 100.0, 500.0, 300.0))

        self.assertTrue(state.add_seat((100.0, 100.0, 300.0, 300.0)))
        self.assertTrue(state.add_seat((300.0, 100.0, 500.0, 300.0)))

        self.assertEqual(state.tables[0]["kind"], "counted_zone")
        self.assertEqual(len(state.tables[0]["seats"]), 2)

    def test_add_seat_without_a_zone_selected_fails(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 300.0))

        self.assertFalse(state.add_seat((110.0, 110.0, 200.0, 290.0)))

    def test_undo_removes_last_seat(self):
        state = CalibrationState()
        state.add_zone((100.0, 100.0, 500.0, 300.0))
        state.add_seat((100.0, 100.0, 300.0, 300.0))
        state.add_seat((300.0, 100.0, 500.0, 300.0))

        state.undo()

        self.assertEqual(len(state.tables[0]["seats"]), 1)

    def test_plain_tables_keep_table_kind(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 300.0))

        self.assertEqual(state.tables[0]["kind"], "table")
        self.assertEqual(state.tables[0]["seats"], [])
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python -m unittest tests.test_calibrate_state -v`
Expected: FAIL — `AttributeError: 'CalibrationState' object has no attribute 'add_zone'`

- [ ] **Step 3: 상태 머신에 구역·칸을 넣는다**

`calibrate.py`의 `add_table`을 바꾸고 두 메서드를 더한다:

```python
    def add_table(self, box: Box) -> None:
        self._snapshot()
        self.tables.append(
            {"box": tuple(box), "chairs": [], "kind": "table", "seats": []}
        )
        self.selected = ("table", len(self.tables) - 1, -1)

    def add_zone(self, box: Box) -> None:
        """Add a counted_zone (bar counter / wall desk) the model cannot detect."""
        self._snapshot()
        self.tables.append(
            {"box": tuple(box), "chairs": [], "kind": "counted_zone", "seats": []}
        )
        self.selected = ("table", len(self.tables) - 1, -1)

    def add_seat(self, box: Box) -> bool:
        table_index = self._selected_table_index()
        if table_index is None:
            return False
        if self.tables[table_index].get("kind") != "counted_zone":
            return False
        self._snapshot()
        self.tables[table_index]["seats"].append(tuple(box))
        return True
```

- [ ] **Step 4: 나머지 dict 생성 지점을 맞춘다**

`calibrate.py`에서 `{"box": ..., "chairs": ...}` 형태의 dict를 만드는 **모든** 자리에 `"kind"`와 `"seats"`를 더한다. 최소 두 곳이다:

`_preseed`(`calibrate.py:134` 부근)에서 탐지 결과로 테이블을 채우는 곳:

```python
        {"box": tuple(box), "chairs": chairs, "kind": "table", "seats": []}
```

`--edit`로 기존 레이아웃을 불러오는 곳:

```python
        {
            "box": table.box,
            "chairs": [chair.box for chair in table.chairs],
            "kind": table.kind,
            "seats": [seat.box for seat in table.seats],
        }
```

- [ ] **Step 5: 레이아웃 변환이 kind/seats를 내보내게 한다**

`calibrate.py`의 import에 `LayoutSeat`를 더한다:

```python
from seatnow_layout import (
    SCHEMA_VERSION,
    LayoutChair,
    LayoutSeat,
    LayoutTable,
    SeatLayout,
)
```

`CalibrationState`를 `SeatLayout`으로 바꾸는 함수의 `LayoutTable(...)` 생성에 두 인자를 더한다:

```python
            LayoutTable(
                id=index + 1,
                name=table.get("name", f"T{index + 1}"),
                box=table["box"],
                chairs=tuple(
                    LayoutChair(id=chair_index + 1, box=chair_box)
                    for chair_index, chair_box in enumerate(table["chairs"])
                ),
                kind=table.get("kind", "table"),
                seats=tuple(
                    LayoutSeat(id=seat_index + 1, box=seat_box)
                    for seat_index, seat_box in enumerate(table.get("seats", []))
                ),
            )
```

- [ ] **Step 6: 테스트가 통과하는지 확인한다**

Run: `python -m unittest tests.test_calibrate_state -v`
Expected: PASS

- [ ] **Step 7: OpenCV 셸에 키 바인딩을 붙인다**

`calibrate.py`의 키 처리부에서 기존 테이블/의자 키 옆에 두 개를 더한다. 변수명(`pending_mode` 등)은 그 파일의 실제 이름을 쓴다:

```python
        elif key == ord("z"):
            pending_mode = "zone"
        elif key == ord("s"):
            pending_mode = "seat"
```

드래그가 끝나 박스가 확정되는 지점에서 모드로 분기한다:

```python
            if pending_mode == "zone":
                state.add_zone(drawn_box)
            elif pending_mode == "seat":
                if not state.add_seat(drawn_box):
                    print("먼저 바 구역(z)을 그리거나 선택하세요")
            elif pending_mode == "chair":
                state.add_chair(drawn_box)
            else:
                state.add_table(drawn_box)
```

화면 안내 문구에 두 키를 더한다:

```python
        "t:테이블  c:의자  z:바구역  s:바좌석칸  u:되돌리기  d:삭제  w:저장  q:종료"
```

- [ ] **Step 8: 전체 테스트**

Run: `python -m unittest discover tests`
Expected: `OK`

- [ ] **Step 9: 커밋**

```bash
git add calibrate.py tests/test_calibrate_state.py
git commit -m "feat: calibrate.py에 바 구역·좌석 칸 그리기 모드"
```

---

### Task 9: 라벨링·채점 도구를 맞춘다

여기까지 되면 6개 영상 라벨링을 시작할 수 있다.

**Files:**
- Modify: `make_labels.py`
- Modify: `verify_seatnow.py`
- Modify: `README.md`
- Test: `tests/test_make_labels.py`
- Test: `tests/test_verify_seatnow.py`

**Interfaces:**
- Consumes: Task 2의 `judgement_units()`, Task 5의 `REASON_GROUPS` / `ACTIONABLE_GROUPS`, Task 6의 `seat_report`
- Produces: `make_labels.seat_names_from_layout(layout) -> List[str]`, `verify_seatnow.summarize_unknown_reasons(records) -> Dict[str, Any]`

- [ ] **Step 1: 라벨 대상 테스트를 쓴다**

`tests/test_make_labels.py`에 추가한다. JSON을 거치지 않고 `SeatLayout`을 직접 조립한다:

```python
from make_labels import seat_names_from_layout
from seatnow_layout import LayoutSeat, LayoutTable, SeatLayout


class CountedZoneSkeletonTests(unittest.TestCase):
    def test_zone_seats_appear_as_individual_label_targets(self):
        layout = SeatLayout(
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
                LayoutTable(id=1, name="창가1", box=(900.0, 300.0, 1100.0, 500.0)),
            ),
        )

        self.assertEqual(seat_names_from_layout(layout), ["BAR-1", "BAR-2", "창가1"])
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `python -m unittest tests.test_make_labels -v`
Expected: FAIL — `ImportError: cannot import name 'seat_names_from_layout'`

- [ ] **Step 3: `make_labels.py`가 판정 단위 이름을 쓰게 한다**

`make_labels.py`에 헬퍼를 추가한다:

```python
def seat_names_from_layout(layout) -> List[str]:
    """Label targets are judgement units: a bar zone contributes one per seat."""
    return [unit.name for unit in layout.judgement_units()]
```

`make_labels.py`에서 지금 `layout.tables`를 돌며 좌석 이름을 뽑는 자리를 이 헬퍼 호출로 바꾼다. **라벨 값은 지금과 같은 `occupied` / `empty` / `ignore` 세 목록을 그대로 쓴다** — 칸 단위로 판정하므로 인원 수를 따로 받을 필요가 없다.

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

Run: `python -m unittest tests.test_make_labels -v`
Expected: PASS

- [ ] **Step 5: 사유 코드 집계 테스트를 쓴다**

`tests/test_verify_seatnow.py`에 추가. import에 `from verify_seatnow import summarize_unknown_reasons`를 더한다:

```python
class ReasonBreakdownTests(unittest.TestCase):
    def test_unknown_reasons_are_grouped_by_what_fixes_them(self):
        records = [
            {
                "seat_report": {
                    "seats": [
                        {"seat_id": "T1", "kind": "table", "state": "unknown",
                         "reason_code": "occluded_lower_body"},
                        {"seat_id": "T2", "kind": "table", "state": "unknown",
                         "reason_code": "pose_low_keypoints"},
                        {"seat_id": "T3", "kind": "table", "state": "unknown",
                         "reason_code": "pending_confirmation"},
                        {"seat_id": "T4", "kind": "table", "state": "empty",
                         "reason_code": "no_customer_evidence"},
                    ]
                }
            }
        ]

        breakdown = summarize_unknown_reasons(records)

        self.assertEqual(breakdown["total_seat_ticks"], 4)
        self.assertEqual(breakdown["unknown_seat_ticks"], 3)
        self.assertEqual(breakdown["by_group"]["geometry"], 1)
        self.assertEqual(breakdown["by_group"]["model"], 1)
        self.assertEqual(breakdown["by_group"]["time"], 1)
        self.assertEqual(breakdown["actionable_unknown_ticks"], 2)

    def test_counted_zone_contributes_capacity_and_reason_counts(self):
        records = [
            {
                "seat_report": {
                    "seats": [
                        {
                            "seat_id": "BAR",
                            "kind": "counted_zone",
                            "capacity": 4,
                            "occupied": 1,
                            "free": 1,
                            "unknown": 2,
                            "reason_codes": {"spans_multiple_seats": 2},
                        }
                    ]
                }
            }
        ]

        breakdown = summarize_unknown_reasons(records)

        self.assertEqual(breakdown["total_seat_ticks"], 4)
        self.assertEqual(breakdown["unknown_seat_ticks"], 2)
        self.assertEqual(breakdown["by_group"]["geometry"], 2)
```

- [ ] **Step 6: 테스트가 실패하는지 확인한다**

Run: `python -m unittest tests.test_verify_seatnow -v`
Expected: FAIL — `ImportError: cannot import name 'summarize_unknown_reasons'`

- [ ] **Step 7: 집계를 구현한다**

`verify_seatnow.py`의 import에 추가:

```python
from seatnow_report import ACTIONABLE_GROUPS, REASON_GROUPS
```

그리고 함수를 추가한다:

```python
def summarize_unknown_reasons(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Break the UNKNOWN rate down by what would fix it.

    "UNKNOWN 34% = geometry 22% + model 8% + time 4%" turns the next
    engineering decision into a table lookup instead of an argument.
    """
    group_of = {
        code.value: group
        for group, codes in REASON_GROUPS.items()
        for code in codes
    }
    by_group: Dict[str, int] = {group: 0 for group in REASON_GROUPS}
    by_code: Dict[str, int] = {}
    total = 0
    unknown = 0

    for record in records:
        report = record.get("seat_report") or {}
        for seat in report.get("seats", []):
            if seat.get("kind") == "counted_zone":
                total += int(seat.get("capacity", 0))
                unknown += int(seat.get("unknown", 0))
                for code, count in (seat.get("reason_codes") or {}).items():
                    by_code[code] = by_code.get(code, 0) + int(count)
                    group = group_of.get(code)
                    if group:
                        by_group[group] = by_group.get(group, 0) + int(count)
                continue
            total += 1
            if seat.get("state") != "unknown":
                continue
            unknown += 1
            code = str(seat.get("reason_code", ""))
            by_code[code] = by_code.get(code, 0) + 1
            group = group_of.get(code)
            if group:
                by_group[group] = by_group.get(group, 0) + 1

    actionable = sum(by_group.get(group, 0) for group in ACTIONABLE_GROUPS)
    return {
        "total_seat_ticks": total,
        "unknown_seat_ticks": unknown,
        "unknown_rate": (unknown / total) if total else 0.0,
        "by_group": by_group,
        "by_code": by_code,
        "actionable_unknown_ticks": actionable,
    }
```

`typing` import에 `Any`, `Dict`, `Sequence`가 없으면 더한다.

- [ ] **Step 8: 결과 출력에 표를 붙인다**

`verify_seatnow.py`에서 기존 커버리지 집계를 출력하는 자리 옆에 추가한다. `records`는 그 자리에서 이미 읽어둔 JSONL 레코드 리스트를 쓴다:

```python
    breakdown = summarize_unknown_reasons(records)
    print(
        f"UNKNOWN {breakdown['unknown_seat_ticks']}/{breakdown['total_seat_ticks']} "
        f"({breakdown['unknown_rate'] * 100:.1f}%) — "
        + ", ".join(
            f"{group} {count}"
            for group, count in sorted(breakdown["by_group"].items())
            if count
        )
    )
```

- [ ] **Step 9: 테스트가 통과하는지 확인한다**

Run: `python -m unittest tests.test_verify_seatnow -v`
Expected: PASS

- [ ] **Step 10: 전체 테스트**

Run: `python -m unittest discover tests`
Expected: `OK`

- [ ] **Step 11: 문서를 갱신한다**

`README.md`의 저장소 구성 표에 `seatnow_layout.py` 줄 아래로 한 줄을 더한다:

```markdown
| `seatnow_report.py` | 앱용 좌석 가용성 계약(`seat_report`) 생성 + UNKNOWN 사유 코드 |
```

`README.md`의 평가·벤치 도구 절에 바 구역 캘리브레이션을 적는다:

```markdown
# 바(일자형/벽 책상): z로 구역을 치고 s로 자리마다 칸을 긋는다
./venv/bin/python calibrate.py sample_raw/cafe_1.mp4 --output layouts/cafe_1.json
```

`README.md`와 `ONBOARDING.md`의 "유닛 테스트 171개" / "Ran 171 tests"를 실제 개수로 맞춘다. 개수는 `python -m unittest discover tests`의 출력에서 확인한다.

- [ ] **Step 12: 커밋**

```bash
git add make_labels.py verify_seatnow.py tests/test_make_labels.py tests/test_verify_seatnow.py README.md ONBOARDING.md
git commit -m "feat: 라벨링·채점 도구를 좌석 칸/사유 코드에 맞춤"
```

---

## 완료 기준

- [ ] `python -m unittest discover tests` 통과
- [ ] 기존 `layouts/*.json` 2개가 무수정으로 로드됨
- [ ] JSONL 매 tick에 `seat_report`가 실림
- [ ] `totals.free`가 UNKNOWN을 절대 포함하지 않음
- [ ] 바 구역에서 한 사람이 두 칸에 걸치면 두 칸 다 `unknown`
- [ ] `verify_seatnow.py`가 UNKNOWN을 그룹별(`install`/`geometry`/`model`/`time`)로 쪼개서 출력
- [ ] `calibrate.py`에서 `z`/`s`로 바 구역과 좌석 칸을 그릴 수 있음

# 2D 평면도 1단계 — 스키마 v3 + `calibrate.py` 확장

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 설치 절차 3-a~3-d를 끝까지 할 수 있게 만든다 — 소속을 아직 정하지 않은 의자를 그릴 수 있고, 바닥 네 점을 찍을 수 있고, 잘못된 바 구역이 저장 단계에서 걸린다.

**Architecture:** 레이아웃 스키마를 v3로 올려 `unassigned_chairs`(소속 미정 의자)와 `floor_reference`(바닥 네 점)를 싣는다. `calibrate.py`는 테이블 선택 없이도 의자를 그릴 수 있게 하고 바닥 점 모드 `[f]`를 더한다. 소속 정하기와 투영은 2단계이며 이 계획에 없다.

**Tech Stack:** Python 3.11 / OpenCV(GUI만) / `unittest`

**설계 문서:** `docs/superpowers/specs/2026-09-01-2d-floorplan-design.md` (§5 데이터 모델, §6 설치 절차, §10-b 단계 구분)

## Global Constraints

- 테스트는 `unittest`. 실행은 `./venv/Scripts/python.exe -m unittest discover tests -p "test_<이름>.py"` (Windows) / `./venv/bin/python ...` (Linux). **`-m unittest tests.X` 형태는 `tests/__init__.py`가 없어 동작하지 않는다**
- 테스트는 **모델도 GUI도 부르지 않는다.** `CalibrationState`는 GUI 없는 순수 상태 기계이고 `tests/test_calibrate_state.py`가 그렇게 검사한다
- **판정 로직(`seatnow_core.py`)을 건드리지 않는다.** 이 단계는 레이아웃 스키마와 캘리브레이션 도구만이다
- **v1·v2 레이아웃 파일이 계속 읽혀야 한다.** `layouts/`에 기존 파일이 있고 `layouts/cafe_angle1.json`은 방금 만든 v2다
- **캔버스에 한글을 쓰지 않는다.** `cv2.putText`의 Hershey 폰트에 한글 글리프가 없어 전부 `?`가 된다 (`8a261d4`에서 겪음). 한글은 터미널에만
- 커밋 메시지는 한국어, `feat:` / `fix:` / `test:` 접두어

---

### Task 1: 바 구역 자리 칸이 구역 밖이면 저장 단계에서 막는다

**Files:**
- Modify: `calibrate.py` (`CalibrationState`에 검증 메서드 추가, `[s]` 처리부)
- Test: `tests/test_calibrate_state.py`

**Interfaces:**
- Produces: `CalibrationState.invalid_seat_zones() -> List[Tuple[int, int]]` — `(테이블 인덱스, 자리 칸 인덱스)` 목록. 비어 있으면 이상 없음

**배경:** `add_seat`는 자리 칸이 구역 상자 안인지 **검사하지 않는다** (`calibrate.py:58-67`). 그런데 `load_layout`은 검사해서 거부한다 (`seatnow_layout.py:243-249`):

```python
if not _box_contains(box, seat.box):
    raise LayoutError(f"table {table_id} seat {seat.id}: box ... is outside the zone box ...")
```

즉 **저장은 성공하고, 몇 시간 뒤 판정을 돌릴 때 터진다.** 사람이 현장을 떠난 뒤다. 저장 시점에 잡아 그 자리에서 고치게 한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_calibrate_state.py`의 `if __name__` 위에 추가 (파일 끝에 `if __name__` 블록이 없으면 파일 맨 끝에 추가):

```python
class SeatInsideZoneTests(unittest.TestCase):
    """A seat slot drawn outside its zone saves fine and fails hours later.

    load_layout rejects it (seatnow_layout.py:243-249) but save_layout does
    not, so the person who could fix it in five seconds is already gone by
    the time anyone sees the error.
    """

    def _zone_with_seat(self, seat_box):
        state = CalibrationState()
        state.add_zone((100.0, 100.0, 500.0, 300.0))
        state.add_seat(seat_box)
        return state

    def test_seat_inside_the_zone_is_valid(self):
        state = self._zone_with_seat((120.0, 120.0, 200.0, 280.0))
        self.assertEqual(state.invalid_seat_zones(), [])

    def test_seat_hanging_outside_is_reported(self):
        state = self._zone_with_seat((450.0, 120.0, 600.0, 280.0))
        self.assertEqual(state.invalid_seat_zones(), [(0, 0)])

    def test_seat_touching_the_edge_is_allowed(self):
        # load_layout allows a 1px tolerance; matching it here keeps the two
        # checks from disagreeing about the same file.
        state = self._zone_with_seat((100.0, 100.0, 500.0, 300.0))
        self.assertEqual(state.invalid_seat_zones(), [])

    def test_plain_table_chairs_are_not_checked(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        state.add_chair((400.0, 400.0, 450.0, 450.0))
        self.assertEqual(state.invalid_seat_zones(), [])

    def test_every_offending_seat_is_listed(self):
        state = self._zone_with_seat((450.0, 120.0, 600.0, 280.0))
        state.add_seat((700.0, 700.0, 800.0, 800.0))
        self.assertEqual(state.invalid_seat_zones(), [(0, 0), (0, 1)])
```

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_calibrate_state.py" -v`
Expected: FAIL — `AttributeError: 'CalibrationState' object has no attribute 'invalid_seat_zones'`

- [ ] **Step 3: 검증 메서드를 구현한다**

`calibrate.py`의 `CalibrationState`에서 `to_layout` **위에** 추가한다:

```python
    def invalid_seat_zones(self) -> List[Tuple[int, int]]:
        """Seat slots drawn outside their zone, as (table index, seat index).

        ``load_layout`` refuses such a file (seatnow_layout.py:243-249) but
        ``save_layout`` does not, so without this check the failure surfaces
        hours later on a machine the installer has already left.
        """
        offenders: List[Tuple[int, int]] = []
        for table_index, table in enumerate(self.tables):
            if table.get("kind") != "counted_zone":
                continue
            for seat_index, seat in enumerate(table.get("seats", [])):
                if not _box_contains(table["box"], seat):
                    offenders.append((table_index, seat_index))
        return offenders
```

**같은 판정을 두 번 구현하지 않는다.** `calibrate.py`의 `seatnow_layout` import 목록에 `_box_contains`를 더한다:

```python
from seatnow_layout import (
    SCHEMA_VERSION,
    LayoutChair,
    LayoutSeat,
    LayoutTable,
    SeatLayout,
    _box_contains,
)
```

밑줄로 시작하는 이름을 가져오는 것이 통상 좋지 않지만, **여기서는 두 검사가 어긋나지 않는 것이 더 중요하다.** 규칙을 복사해 두면 허용 오차(1px)가 한쪽에서만 바뀌는 날 저장은 통과하고 로드는 거부하는 상태로 되돌아간다 — 이 태스크가 없애려는 바로 그 문제다. `test_seat_touching_the_edge_is_allowed`가 이 일치를 지킨다.

`calibrate.py:22`의 `_contains`는 **점 포함**을 보는 다른 함수다. 헷갈리지 말 것.

`typing` import에 `Tuple`이 이미 있다 (`calibrate.py:9`).

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_calibrate_state.py" -v`
Expected: PASS — 신규 5개 포함

- [ ] **Step 5: 저장 처리부에 연결한다**

`calibrate.py`의 `elif key == ord("s"):` 블록에서, 빈 바 구역을 막는 `if empty_zones:` 검사 **바로 아래에** 추가한다:

```python
            offenders = state.invalid_seat_zones()
            if offenders:
                spots = ", ".join(
                    f"BAR{table_index + 1}의 {seat_index + 1}번 칸"
                    for table_index, seat_index in offenders
                )
                print(
                    f"자리 칸이 바 구역 밖으로 나가 저장하지 않았습니다: {spots} — "
                    f"[d]로 지우고 구역 안에 다시 그으세요. "
                    f"(구역 밖 자리 칸이 있으면 나중에 판정을 돌릴 때 파일이 거부됩니다)"
                )
                continue
```

- [ ] **Step 6: 전체 테스트**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: PASS

- [ ] **Step 7: 커밋**

```bash
git add calibrate.py tests/test_calibrate_state.py
git commit -m "fix: 구역 밖 자리 칸을 저장 단계에서 막는다

add_seat 는 검사하지 않는데 load_layout 은 거부한다. 그래서 저장은
성공하고 몇 시간 뒤 판정을 돌릴 때 터진다 - 5초면 고칠 사람이 이미
현장을 떠난 뒤다. 같은 규칙(허용 오차 1px 포함)을 저장 시점에도 건다."
```

---

### Task 2: 레이아웃 스키마 v3 — 소속 없는 의자

**Files:**
- Modify: `seatnow_layout.py` (`SeatLayout` 필드, `scaled_to`, `chair_boxes`, `load_layout`, `save_layout`, `SCHEMA_VERSION`)
- Test: `tests/test_seatnow_layout.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `SeatLayout.unassigned_chairs: Tuple[LayoutChair, ...]` (기본 `()`)
  - `SCHEMA_VERSION = 3`, `SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)`
  - `chair_boxes()`는 **소속 있는 의자를 먼저, 소속 없는 의자를 뒤에** 반환한다

**핵심 불변조건:** `unit_chair_assignments()`는 `chair_boxes()`의 인덱스를 가리킨다. 소속 없는 의자를 **목록 뒤에** 붙여야 기존 인덱스가 안 밀린다. 앞이나 중간에 넣으면 모든 연결이 조용히 어긋난다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_seatnow_layout.py` 끝의 `if __name__` 위에 추가:

```python
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
```

파일 상단 import에 없으면 추가한다: `import json`, `import tempfile`, `from pathlib import Path`, 그리고 `seatnow_layout`에서 `LayoutChair, LayoutTable, SeatLayout, load_layout, save_layout`.

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_seatnow_layout.py" -v`
Expected: FAIL — `TypeError: SeatLayout.__init__() got an unexpected keyword argument 'unassigned_chairs'`

- [ ] **Step 3: 스키마를 올린다**

`seatnow_layout.py:16-17`:

```python
SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)
```

`SeatLayout` 정의에 필드를 더한다:

```python
@dataclass(frozen=True)
class SeatLayout:
    schema_version: int
    source: Dict[str, object]
    tables: Tuple[LayoutTable, ...]
    unassigned_chairs: Tuple[LayoutChair, ...] = ()
```

`chair_boxes`를 바꾼다:

```python
    def chair_boxes(self) -> List[Box]:
        """Assigned chairs first (table order), then the unassigned ones.

        ``unit_chair_assignments`` indexes into this list, so the assigned
        chairs must keep the leading positions: inserting an orphan anywhere
        earlier would silently repoint every chair->table link.
        """
        boxes = [chair.box for table in self.tables for chair in table.chairs]
        boxes.extend(chair.box for chair in self.unassigned_chairs)
        return boxes
```

`scaled_to`의 `return replace(...)` 직전에 소속 없는 의자도 스케일한다:

```python
        unassigned = tuple(
            replace(chair, box=scale(chair.box)) for chair in self.unassigned_chairs
        )
        source = dict(self.source, width=width, height=height)
        return replace(
            self, source=source, tables=tables, unassigned_chairs=unassigned
        )
```

(기존의 `source = ...` / `return replace(self, source=source, tables=tables)` 두 줄을 위 블록으로 교체한다.)

`load_layout`의 `return SeatLayout(...)` 앞에 파싱을 더한다:

```python
    unassigned_chairs = tuple(
        LayoutChair(
            id=int(chair.get("id", position)),
            box=_parse_box(chair.get("box"), f"unassigned chair #{position}"),
        )
        for position, chair in enumerate(data.get("unassigned_chairs", []), start=1)
    )
    return SeatLayout(
        schema_version=SCHEMA_VERSION,
        source=dict(data.get("source", {})),
        tables=tuple(tables),
        unassigned_chairs=unassigned_chairs,
    )
```

`save_layout`의 `payload`에 더한다 (`"tables": [...]` 뒤):

```python
        "unassigned_chairs": [
            {"id": chair.id, "box": [round(v, 2) for v in chair.box]}
            for chair in layout.unassigned_chairs
        ],
```

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_seatnow_layout.py" -v`
Expected: PASS — 신규 7개 포함

- [ ] **Step 5: 전체 테스트로 회귀를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: PASS

- [ ] **Step 6: 실제 파일이 여전히 읽히는지 확인한다**

```bash
./venv/Scripts/python.exe -c "
from seatnow_layout import load_layout
for name in ('cafe_angle1', 'seminar_room'):
    layout = load_layout(f'layouts/{name}.json')
    print(name, 'tables', len(layout.tables), 'chairs', len(layout.chair_boxes()), 'orphans', len(layout.unassigned_chairs))"
```

Expected: `cafe_angle1 tables 6 chairs 18 orphans 0`

- [ ] **Step 7: 커밋**

```bash
git add seatnow_layout.py tests/test_seatnow_layout.py
git commit -m "feat: 레이아웃 스키마 v3 - 소속 없는 의자

설치 3-c 에서 의자 상자를 그리고 5-b 에서 소속을 정한다. 그 사이에
의자는 어느 테이블도 주장하지 않은 채로 존재해야 한다 - 틀린 주장은
엉뚱한 테이블을 점유로 만든다.

chair_boxes() 에서 소속 없는 의자를 반드시 뒤에 붙인다.
unit_chair_assignments() 가 이 목록의 인덱스를 가리키므로 앞이나
중간에 넣으면 모든 연결이 조용히 어긋난다.

v1·v2 파일은 그대로 읽힌다."
```

---

### Task 3: `calibrate.py` — 테이블 선택 없이 의자 그리기

**Files:**
- Modify: `calibrate.py` (`CalibrationState`: `unassigned_chairs`, `_snapshot`, `add_chair`, `select_at`, `delete_selected`, `to_layout`, `from_layout`, `_draw`)
- Test: `tests/test_calibrate_state.py`

**Interfaces:**
- Consumes: Task 2의 `SeatLayout.unassigned_chairs`
- Produces:
  - `CalibrationState.unassigned_chairs: List[Box]`
  - `add_chair(box)`는 선택된 테이블이 없으면 **소속 없는 의자로 넣고 `True`를 반환한다** (기존에는 `False`)
  - `select_at`이 소속 없는 의자를 `("orphan", -1, index)`로 선택한다
  - 상수 `ORPHAN_CHAIR_COLOR`

**주의:** 기존 테스트 `test_add_chair_without_table_returns_false`가 이 변경으로 깨진다. 지우지 말고 **새 동작을 검사하도록 고친다** — 그 테스트가 지키던 것("소속 없이 테이블에 붙지 않는다")은 여전히 지켜야 한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_calibrate_state.py`의 기존 테스트를 바꾼다:

```python
    def test_add_chair_without_table_becomes_unassigned(self):
        # Install step 3-c draws chairs before anyone decides ownership.
        # Refusing the draw used to print the same error over and over while
        # the person had no way to proceed.
        state = CalibrationState()
        self.assertTrue(state.add_chair((10.0, 10.0, 20.0, 20.0)))
        self.assertEqual(len(state.unassigned_chairs), 1)
        self.assertEqual(state.tables, [])
```

그리고 `if __name__` 위에 추가:

```python
class UnassignedChairStateTests(unittest.TestCase):
    def test_chair_still_attaches_when_a_table_is_selected(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        self.assertTrue(state.add_chair((310.0, 110.0, 360.0, 190.0)))
        self.assertEqual(len(state.tables[0]["chairs"]), 1)
        self.assertEqual(state.unassigned_chairs, [])

    def test_unassigned_chair_can_be_selected_and_deleted(self):
        state = CalibrationState()
        state.add_chair((10.0, 10.0, 40.0, 40.0))
        state.select_at(20.0, 20.0)
        self.assertEqual(state.selected, ("orphan", -1, 0))
        state.delete_selected()
        self.assertEqual(state.unassigned_chairs, [])

    def test_undo_restores_unassigned_chairs(self):
        state = CalibrationState()
        state.add_chair((10.0, 10.0, 40.0, 40.0))
        state.add_chair((50.0, 50.0, 80.0, 80.0))
        state.undo()
        self.assertEqual(len(state.unassigned_chairs), 1)

    def test_selection_prefers_the_smaller_box(self):
        # Same rule as everything else: a small orphan inside a big table wins.
        state = CalibrationState()
        state.add_table((0.0, 0.0, 500.0, 500.0))
        state.add_chair((100.0, 100.0, 140.0, 140.0))  # attaches to the table
        state.selected = None
        state.unassigned_chairs.append((110.0, 110.0, 120.0, 120.0))
        state.select_at(115.0, 115.0)
        self.assertEqual(state.selected, ("orphan", -1, 0))

    def test_round_trip_through_layout_keeps_orphans(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        state.selected = None
        state.add_chair((900.0, 900.0, 940.0, 940.0))
        layout = state.to_layout({"width": 1920, "height": 1080})
        self.assertEqual(len(layout.unassigned_chairs), 1)

        restored = CalibrationState.from_layout(layout)
        self.assertEqual(len(restored.unassigned_chairs), 1)
        self.assertEqual(restored.unassigned_chairs[0], (900.0, 900.0, 940.0, 940.0))
```

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_calibrate_state.py" -v`
Expected: FAIL — `AttributeError: 'CalibrationState' object has no attribute 'unassigned_chairs'`

- [ ] **Step 3: 상태 기계를 고친다**

`CalibrationState.__init__`:

```python
    def __init__(self) -> None:
        self.tables: List[Dict] = []  # {"box": Box, "chairs": List[Box]}
        self.unassigned_chairs: List[Box] = []
        self.selected: Optional[Tuple[str, int, int]] = None
        self._history: List[
            Tuple[List[Dict], List[Box], Optional[Tuple[str, int, int]]]
        ] = []
```

`_snapshot` / `undo`:

```python
    def _snapshot(self) -> None:
        self._history.append(
            (
                copy.deepcopy(self.tables),
                list(self.unassigned_chairs),
                self.selected,
            )
        )

    def undo(self) -> None:
        if not self._history:
            return
        self.tables, self.unassigned_chairs, self.selected = self._history.pop()
```

`add_chair` — **선택이 없으면 거절하지 않고 소속 없는 의자로 받는다**:

```python
    def add_chair(self, box: Box) -> bool:
        """Attach to the selected table, or park it as unassigned.

        Install step 3-c draws chair boxes before anyone has decided which
        table they serve; ownership is set later on the floor plan, where
        perspective is gone and the answer is obvious.  Refusing the draw
        here just left the installer with no way forward.
        """
        table_index = self._selected_table_index()
        self._snapshot()
        if table_index is None:
            self.unassigned_chairs.append(tuple(box))
            return True
        self.tables[table_index]["chairs"].append(tuple(box))
        return True
```

`select_at`에 후보를 더한다 (`for ti, table ...` 루프 **뒤**, `self.selected = ...` 앞):

```python
        for oi, chair in enumerate(self.unassigned_chairs):
            if _contains(chair, x, y):
                candidates.append((_area(chair), ("orphan", -1, oi)))
```

`delete_selected`:

```python
    def delete_selected(self) -> None:
        if self.selected is None:
            return
        self._snapshot()
        kind, ti, ci = self.selected
        if kind == "orphan":
            del self.unassigned_chairs[ci]
        elif kind == "table":
            del self.tables[ti]
        elif kind == "seat":
            del self.tables[ti]["seats"][ci]
        else:
            del self.tables[ti]["chairs"][ci]
        self.selected = None
```

`_selected_table_index`는 `("orphan", -1, ci)`에서 `-1`을 반환하면 안 된다:

```python
    def _selected_table_index(self) -> Optional[int]:
        if self.selected is None or self.selected[0] == "orphan":
            return None
        return self.selected[1]
```

`to_layout`의 `return SeatLayout(...)`:

```python
        return SeatLayout(
            schema_version=SCHEMA_VERSION,
            source=dict(source),
            tables=tables,
            unassigned_chairs=tuple(
                LayoutChair(id=index, box=box)
                for index, box in enumerate(self.unassigned_chairs, start=1)
            ),
        )
```

`from_layout`의 `state.selected = None` 앞:

```python
        state.unassigned_chairs = [
            tuple(chair.box) for chair in layout.unassigned_chairs
        ]
```

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_calibrate_state.py" -v`
Expected: PASS

- [ ] **Step 5: 화면에 그린다**

`calibrate.py`의 색 상수 아래에 더한다 (BGR 순서. 자주색 — 노란 의자·빨간 선택과 확실히 구별된다):

```python
ORPHAN_CHAIR_COLOR = (200, 60, 200)
```

`_draw`의 테이블 루프가 끝난 **뒤**, `if drag is not None:` **앞**에 추가한다:

```python
    for oi, chair in enumerate(state.unassigned_chairs):
        ox1, oy1, ox2, oy2 = [int(v) for v in chair]
        selected = state.selected == ("orphan", -1, oi)
        cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2),
                      SELECT_COLOR if selected else ORPHAN_CHAIR_COLOR, 2)
        cv2.putText(canvas, "?", (ox1 + 4, oy2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORPHAN_CHAIR_COLOR, 2, cv2.LINE_AA)
```

`?` 표시는 "소속 미정"이라는 뜻이고, 5-b에서 정해질 것이 몇 개 남았는지 한눈에 보인다.

`add_chair`가 이제 실패하지 않으므로 실패 메시지를 지운다. `on_mouse`의 `else:` 가지를 이렇게 바꾼다:

```python
            else:
                state.add_chair(box)
```

- [ ] **Step 6: 전체 테스트**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: PASS

- [ ] **Step 7: 커밋**

```bash
git add calibrate.py tests/test_calibrate_state.py
git commit -m "feat: 테이블 선택 없이 의자를 그릴 수 있다 (소속 미정)

설치 3-c 는 의자 위치만 그리고, 누구 것인지는 원근이 펴진 평면도에서
정한다(5-b). 그런데 지금은 테이블을 선택 안 하면 의자가 아예 안
그려지고 같은 오류만 반복 출력됐다 - 사람이 앞으로 갈 방법이 없었다.

소속 미정 의자는 자주색에 ? 로 그려서 5-b 에서 정할 것이 몇 개
남았는지 보이게 한다."
```

---

### Task 4: 레이아웃 스키마 v3 — 바닥 기준점

**Files:**
- Modify: `seatnow_layout.py` (`FloorReference`, `SeatLayout.floor_reference`, `scaled_to`, `load_layout`, `save_layout`)
- Test: `tests/test_seatnow_layout.py`

**Interfaces:**
- Consumes: Task 2의 v3 스키마
- Produces:
  - `FloorReference` 데이터클래스 — `image_points: Tuple[Tuple[float, float], ...]`, 정확히 4개
  - `SeatLayout.floor_reference: Optional[FloorReference] = None`

**범위:** 이 태스크는 **저장·복원만** 한다. 호모그래피 계산과 퇴화 검사는 2단계다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_seatnow_layout.py`의 `if __name__` 위에 추가:

```python
class FloorReferenceTests(unittest.TestCase):
    """Four image points that are a rectangle on the real floor.

    Stage 2 turns these into a homography.  Stage 1 only has to carry them
    without losing or mangling them, including when the frame is rescaled.
    """

    POINTS = ((610.0, 760.0), (1180.0, 720.0), (1410.0, 980.0), (520.0, 1040.0))

    def _layout(self, points=None):
        return SeatLayout(
            schema_version=3,
            source={"width": 1920, "height": 1080},
            tables=(LayoutTable(id=1, name="T1", box=(10.0, 10.0, 50.0, 50.0)),),
            floor_reference=(
                None if points is None else FloorReference(image_points=points)
            ),
        )

    def test_absent_by_default(self):
        self.assertIsNone(self._layout().floor_reference)

    def test_round_trips_through_a_file(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "layout.json"
            save_layout(self._layout(self.POINTS), path)
            loaded = load_layout(path)
            self.assertEqual(loaded.floor_reference.image_points, self.POINTS)

    def test_absent_reference_round_trips_as_none(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "layout.json"
            save_layout(self._layout(), path)
            self.assertIsNone(load_layout(path).floor_reference)

    def test_wrong_point_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "layout.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "source": {"width": 1920, "height": 1080},
                        "tables": [{"id": 1, "name": "T1", "kind": "table",
                                    "box": [10, 10, 50, 50]}],
                        "floor_reference": {"image_points": [[0, 0], [1, 1], [2, 2]]},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(LayoutError):
                load_layout(path)

    def test_scaled_to_scales_the_points(self):
        scaled = self._layout(self.POINTS).scaled_to(960, 540)
        self.assertEqual(scaled.floor_reference.image_points[0], (305.0, 380.0))
```

import에 `FloorReference`와 `LayoutError`를 더한다.

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_seatnow_layout.py" -v`
Expected: FAIL — `ImportError: cannot import name 'FloorReference' from 'seatnow_layout'`

- [ ] **Step 3: 구현한다**

`seatnow_layout.py`의 `LayoutSeat` 아래에 더한다:

```python
FLOOR_REFERENCE_POINTS = 4


@dataclass(frozen=True)
class FloorReference:
    """Four image points that form a rectangle on the real floor, clockwise.

    Stage 2 turns these into a homography that flattens the camera view.  No
    real-world measurements are asked for: the customer needs "the window
    seat on the right", not "3.2 m from the door", and the aspect ratio gets
    corrected by hand on the floor plan.
    """

    image_points: Tuple[Tuple[float, float], ...]
```

`SeatLayout`에 필드를 더한다 (`unassigned_chairs` 아래):

```python
    floor_reference: Optional[FloorReference] = None
```

`Optional`이 import되어 있는지 확인한다 (`seatnow_layout.py`의 `typing` import).

`scaled_to`에서 같이 스케일한다 — Task 2에서 만든 `unassigned` 블록 아래:

```python
        floor_reference = self.floor_reference
        if floor_reference is not None:
            floor_reference = replace(
                floor_reference,
                image_points=tuple(
                    (px * sx, py * sy) for px, py in floor_reference.image_points
                ),
            )
        source = dict(self.source, width=width, height=height)
        return replace(
            self,
            source=source,
            tables=tables,
            unassigned_chairs=unassigned,
            floor_reference=floor_reference,
        )
```

`load_layout`에 파싱을 더한다 (`unassigned_chairs` 파싱 아래):

```python
    floor_reference = None
    raw_floor = data.get("floor_reference")
    if raw_floor is not None:
        raw_points = raw_floor.get("image_points", [])
        if len(raw_points) != FLOOR_REFERENCE_POINTS:
            raise LayoutError(
                f"floor_reference needs exactly {FLOOR_REFERENCE_POINTS} points, "
                f"got {len(raw_points)}"
            )
        points = []
        for position, point in enumerate(raw_points, start=1):
            if len(point) != 2:
                raise LayoutError(
                    f"floor_reference point #{position} must be [x, y]"
                )
            points.append((float(point[0]), float(point[1])))
        floor_reference = FloorReference(image_points=tuple(points))
```

그리고 `return SeatLayout(...)`에 `floor_reference=floor_reference,`를 더한다.

`save_layout`의 `payload` 조립 뒤, 파일을 쓰기 전에 더한다:

```python
    if layout.floor_reference is not None:
        payload["floor_reference"] = {
            "image_points": [
                [round(px, 2), round(py, 2)]
                for px, py in layout.floor_reference.image_points
            ]
        }
```

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_seatnow_layout.py" -v`
Expected: PASS

- [ ] **Step 5: 전체 테스트**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add seatnow_layout.py tests/test_seatnow_layout.py
git commit -m "feat: 레이아웃 스키마 v3 - 바닥 기준점 네 개

2단계에서 이 네 점으로 호모그래피를 만들어 카메라 화면을 편다.
1단계는 잃지 않고 나르기만 한다 - 화면 크기가 바뀌면 같이 스케일된다.

실측 치수는 받지 않는다. 손님에게 필요한 건 '창가 오른쪽'이지
'입구에서 3.2m'가 아니다."
```

---

### Task 5: `calibrate.py` — 바닥 네 점 찍기 `[f]`

**Files:**
- Modify: `calibrate.py` (`CalibrationState`: `floor_points`, `add_floor_point`; `HELP_TEXT`; `_draw`; 키·마우스 처리; `to_layout`/`from_layout`)
- Test: `tests/test_calibrate_state.py`

**Interfaces:**
- Consumes: Task 4의 `FloorReference`, Task 3의 상태 기계
- Produces:
  - `CalibrationState.floor_points: List[Tuple[float, float]]`
  - `add_floor_point(x, y) -> int` — 담은 뒤의 개수를 반환. **5번째 점은 처음부터 다시 시작한다**
  - 상수 `FLOOR_COLOR`

**왜 5번째가 초기화인가:** 잘못 찍었을 때 되돌리는 방법이 필요한데, 별도의 지우기 키를 더하면 조작이 하나 더 늘어난다. 네 점을 다 찍은 뒤 또 찍으면 "다시 찍는 중"으로 보는 것이 배울 것이 가장 적다. `[u]`(undo)도 그대로 동작한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_calibrate_state.py`의 `if __name__` 위에 추가:

```python
class FloorPointTests(unittest.TestCase):
    def test_points_accumulate_in_click_order(self):
        state = CalibrationState()
        for index, (x, y) in enumerate(
            [(10.0, 10.0), (90.0, 12.0), (95.0, 80.0), (8.0, 78.0)], start=1
        ):
            self.assertEqual(state.add_floor_point(x, y), index)
        self.assertEqual(state.floor_points[0], (10.0, 10.0))
        self.assertEqual(state.floor_points[3], (8.0, 78.0))

    def test_fifth_point_starts_over(self):
        # Re-clicking is how you fix a bad point; a separate clear key would
        # be one more thing to learn during a 30-minute install.
        state = CalibrationState()
        for x, y in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]:
            state.add_floor_point(x, y)
        self.assertEqual(state.add_floor_point(9.0, 9.0), 1)
        self.assertEqual(state.floor_points, [(9.0, 9.0)])

    def test_undo_restores_floor_points(self):
        state = CalibrationState()
        state.add_floor_point(1.0, 1.0)
        state.add_floor_point(2.0, 2.0)
        state.undo()
        self.assertEqual(state.floor_points, [(1.0, 1.0)])

    def test_four_points_reach_the_layout(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        for x, y in [(10.0, 10.0), (90.0, 12.0), (95.0, 80.0), (8.0, 78.0)]:
            state.add_floor_point(x, y)
        layout = state.to_layout({"width": 1920, "height": 1080})
        self.assertEqual(len(layout.floor_reference.image_points), 4)

    def test_fewer_than_four_points_produce_no_reference(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        state.add_floor_point(10.0, 10.0)
        layout = state.to_layout({"width": 1920, "height": 1080})
        self.assertIsNone(layout.floor_reference)

    def test_round_trip_restores_floor_points(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        for x, y in [(10.0, 10.0), (90.0, 12.0), (95.0, 80.0), (8.0, 78.0)]:
            state.add_floor_point(x, y)
        restored = CalibrationState.from_layout(
            state.to_layout({"width": 1920, "height": 1080})
        )
        self.assertEqual(restored.floor_points[2], (95.0, 80.0))
```

- [ ] **Step 2: 실패를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_calibrate_state.py" -v`
Expected: FAIL — `AttributeError: 'CalibrationState' object has no attribute 'add_floor_point'`

- [ ] **Step 3: 상태 기계를 고친다**

`__init__`에 더한다:

```python
        self.floor_points: List[Tuple[float, float]] = []
```

`_history` 타입과 `_snapshot` / `undo`에 `floor_points`를 포함시킨다:

```python
        self._history: List[
            Tuple[
                List[Dict],
                List[Box],
                List[Tuple[float, float]],
                Optional[Tuple[str, int, int]],
            ]
        ] = []

    def _snapshot(self) -> None:
        self._history.append(
            (
                copy.deepcopy(self.tables),
                list(self.unassigned_chairs),
                list(self.floor_points),
                self.selected,
            )
        )

    def undo(self) -> None:
        if not self._history:
            return
        (
            self.tables,
            self.unassigned_chairs,
            self.floor_points,
            self.selected,
        ) = self._history.pop()
```

메서드를 더한다:

```python
    def add_floor_point(self, x: float, y: float) -> int:
        """Collect one of the four floor corners; the fifth click starts over.

        Re-clicking is the only correction anyone needs here, and it costs no
        extra key to learn during an install that already has seven of them.
        """
        self._snapshot()
        if len(self.floor_points) >= FLOOR_REFERENCE_POINTS:
            self.floor_points = []
        self.floor_points.append((float(x), float(y)))
        return len(self.floor_points)
```

`to_layout`의 `SeatLayout(...)`에 더한다:

```python
            floor_reference=(
                FloorReference(image_points=tuple(self.floor_points))
                if len(self.floor_points) == FLOOR_REFERENCE_POINTS
                else None
            ),
```

`from_layout`에 더한다 (`state.selected = None` 앞):

```python
        if layout.floor_reference is not None:
            state.floor_points = [
                (float(px), float(py))
                for px, py in layout.floor_reference.image_points
            ]
```

import를 더한다:

```python
from seatnow_layout import (
    FLOOR_REFERENCE_POINTS,
    FloorReference,
    SCHEMA_VERSION,
    ...
)
```

- [ ] **Step 4: 통과를 확인한다**

Run: `./venv/Scripts/python.exe -m unittest discover tests -p "test_calibrate_state.py" -v`
Expected: PASS

- [ ] **Step 5: 창에 연결한다**

`HELP_TEXT`를 바꾼다:

```python
HELP_TEXT = (
    "[t]able  [c]hair  [z]one(bar)  seat[x]  [f]loor  [d]elete  [u]ndo  [s]ave  [q]uit"
)
```

색 상수에 더한다 (흰색):

```python
FLOOR_COLOR = (255, 255, 255)
```

`_draw`에서 소속 미정 의자를 그린 **뒤**에 바닥 점을 그린다:

```python
    for index, (px, py) in enumerate(state.floor_points, start=1):
        cv2.circle(canvas, (int(px), int(py)), 7, FLOOR_COLOR, -1)
        cv2.putText(canvas, str(index), (int(px) + 10, int(py) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, FLOOR_COLOR, 2, cv2.LINE_AA)
    if len(state.floor_points) == FLOOR_REFERENCE_POINTS:
        pts = [(int(px), int(py)) for px, py in state.floor_points]
        for start, end in zip(pts, pts[1:] + pts[:1]):
            cv2.line(canvas, start, end, FLOOR_COLOR, 2, cv2.LINE_AA)
```

`on_mouse`에서 바닥 모드일 때는 **드래그가 아니라 클릭**을 받는다. `elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:` 블록의 `if box[2] - box[0] < 8 ...` **위에** 넣는다:

```python
            if mode == "floor":
                count = state.add_floor_point(x, y)
                if count == FLOOR_REFERENCE_POINTS:
                    print("바닥 네 점을 다 찍었습니다. 다시 찍으려면 한 번 더 클릭하세요")
                else:
                    print(f"바닥 점 {count}/{FLOOR_REFERENCE_POINTS}")
                return
```

키 처리에 더한다 (`elif key == ord("x"):` 아래):

```python
        elif key == ord("f"):
            mode = "floor"
```

- [ ] **Step 6: 전체 테스트**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: PASS

- [ ] **Step 7: 실제 창으로 손 확인**

```bash
./venv/Scripts/python.exe calibrate.py sample_raw/cafe_sample_angle1.mov \
  --at 30 --edit layouts/cafe_angle1.json --output /tmp/floor_test.json
```

창에서 확인할 것:
1. 상단에 `[f]loor`가 보인다
2. `c`를 누르고 아무 데나 드래그 → **자주색 `?` 상자**가 생긴다 (테이블을 선택하지 않았는데도)
3. `f`를 누르고 바닥을 네 번 클릭 → 흰 점 1~4와 사각형 선이 그려지고, 터미널에 `바닥 점 1/4` … `바닥 네 점을 다 찍었습니다`
4. 다섯 번째 클릭 → 점이 하나로 초기화된다
5. `s` → 저장. `/tmp/floor_test.json`에 `unassigned_chairs`와 `floor_reference`가 들어 있다

```bash
./venv/Scripts/python.exe -c "
import json; d = json.load(open('/tmp/floor_test.json', encoding='utf-8'))
print('schema', d['schema_version'])
print('orphans', len(d.get('unassigned_chairs', [])))
print('floor', d.get('floor_reference'))"
```

Expected: `schema 3`, `orphans 1`, `floor {'image_points': [[...], [...], [...], [...]]}`

- [ ] **Step 8: 커밋**

```bash
git add calibrate.py tests/test_calibrate_state.py
git commit -m "feat: calibrate.py [f] - 바닥 네 점 찍기

2단계의 호모그래피가 이 네 점을 먹는다. 드래그가 아니라 클릭 네 번이고,
다섯 번째 클릭은 처음부터 다시 찍는 것으로 본다 - 잘못 찍었을 때
되돌릴 방법이 필요한데 지우기 키를 따로 두면 설치 때 배울 것이 하나
더 늘어난다."
```

---

### Task 6: 문서 갱신

**Files:**
- Modify: `README.md` (평가·벤치 도구 절의 캘리브레이션 부분), `plan.md` (§0-a 상태표, T14)

**Interfaces:**
- Consumes: Task 1~5

- [ ] **Step 1: README의 캘리브레이션 설명을 고친다**

`README.md`의 `# 0. 설치 시 좌석 캘리브레이션 (1회)` 블록을 아래로 교체한다:

```bash
# 0. 설치 시 좌석 캘리브레이션 (1회) — 순서대로 3-a ~ 3-d
#    3-a [t] 테이블: 잘못 잡힌 것 지우고 못 잡은 것 추가
#    3-b [z] 바 구역 + [x] 자리 칸: 일자형·벽 책상은 모델이 절대 못 잡는다.
#            칸 개수 = 자리 수. 자리 칸은 반드시 구역 상자 안에 있어야 한다
#    3-c [c] 의자: 과탐 지우기, 누락 추가, 벽 소파는 테이블별 조각으로.
#            테이블을 선택하지 않고 그리면 '소속 미정'(자주색 ?)이 되고,
#            누구 것인지는 평면도 편집기에서 정한다
#    3-d [f] 바닥 네 점: 바닥에서 실제로 직사각형인 것의 귀퉁이를 시계방향 클릭
./venv/bin/python calibrate.py sample_raw/cafe_sample_angle1.mov \
  --at 30 --output layouts/cafe_angle1.json
#    키: [t]able [c]hair [z]one(bar) seat[x] [f]loor [d]elete [u]ndo [s]ave [q]uit
#    이어서 작업하려면 --edit layouts/cafe_angle1.json
```

- [ ] **Step 2: plan.md에 진행 상황을 적는다**

`plan.md` §0-a 상태표의 T14 줄 아래에 한 줄을 더한다:

```markdown
| **2D 평면도 1단계** | ✅ **완료 (2026-09-01)** | 스키마 v3(소속 미정 의자·바닥 네 점) + `calibrate.py` 확장. 설계: `docs/superpowers/specs/2026-09-01-2d-floorplan-design.md` |
```

- [ ] **Step 3: 전체 테스트로 마무리 확인**

Run: `./venv/Scripts/python.exe -m unittest discover tests`
Expected: PASS

- [ ] **Step 4: 커밋**

```bash
git add README.md plan.md
git commit -m "docs: 캘리브레이션 절차 3-a~3-d 와 2D 평면도 1단계 완료 기록"
```

---

## 검토 메모

**설계 문서 대응 (1단계 범위)**

| 설계 절 | 태스크 |
|---|---|
| §5 스키마 v3 `unassigned_chairs` | Task 2 |
| §5 스키마 v3 `floor_reference` | Task 4 |
| §6-3c 소속 없는 의자 그리기 | Task 3 |
| §6-3d 바닥 네 점 | Task 5 |
| §11 테스트 `test_layout_schema_v3` | Task 2·4 (`tests/test_seatnow_layout.py`에 넣는다 — 기존 파일이 이미 이 모듈을 검사하고 있어 새 파일을 만들면 같은 대상이 두 곳으로 갈린다) |
| §11 테스트 `test_calibrate_state` 추가분 | Task 1·3·5 |

**2단계로 미룬 것** — 투영(`test_floor_projection`), `floorplan.json`(`test_floorplan`), 브라우저 편집기, 의자 소속 편집.
**3단계로 미룬 것** — 가구 이동 신호(`test_furniture_drift`).

**계획을 쓰며 발견해 넣은 것**

Task 1은 설계 문서에 없다. `load_layout`은 구역 밖 자리 칸을 거부하는데(`seatnow_layout.py:243-249`) `add_seat`도 `save_layout`도 검사하지 않아, **저장은 성공하고 판정을 돌릴 때 터진다.** 설치 3-b가 바로 그 작업이라 다음 세션에서 밟을 함정이고, 저장 시점에 막는 것이 맞다.

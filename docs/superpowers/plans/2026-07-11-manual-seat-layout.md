# 수동 좌석 레이아웃 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 캘리브레이션 도구로 테이블·의자 배치를 등록하고(`layouts/*.json`), `seatnow.py --layout`으로 그 배치를 기준으로 점유 판정한다.

**Architecture:** 새 모듈 `seatnow_layout.py`가 레이아웃 로드/검증/스케일을 담당한다. `SeatNowAnalyzer`는 선택적 `layout` 인자를 받아 테이블/의자 소스와 연결만 레이아웃 고정값으로 교체하고, 사람·물체 탐지와 점유 증거·디바운싱 로직은 기존 코드를 그대로 쓴다. `calibrate.py`는 순수 상태 클래스(`CalibrationState`) 위에 얇은 OpenCV GUI를 얹는다.

**Tech Stack:** Python 3.9, OpenCV(highgui/Cocoa), Ultralytics YOLO(프리시드), ffmpeg(프레임 추출), unittest.

## Global Constraints

- 테스트 실행 명령은 항상 `./venv/bin/python -m unittest discover tests` (pytest 미설치).
- 박스 표기는 프로젝트 관례인 `(x1, y1, x2, y2)` float 튜플, JSON에서는 4원소 배열.
- 레이아웃 JSON `schema_version`은 `1` 고정. 다른 값이면 명확한 에러.
- 존 밖 테이블/의자 탐지는 레이아웃 모드에서 완전 무시(집계·로그 모두).
- 레이아웃 모드에서 inferred-seat(가려진 테이블 추정) 비활성.
- 기존 자동 모드는 무회귀: 전체 기존 테스트(54개)가 계속 통과해야 한다.
- 커밋 메시지는 한 줄 요약 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: git 저장소 초기화

**Files:**
- Create: `.gitignore`

**Interfaces:**
- Produces: 이후 모든 태스크가 커밋 가능한 git 저장소.

- [ ] **Step 1: .gitignore 작성**

```gitignore
venv/
__pycache__/
*.pt
.DS_Store
.Rhistory
sample_raw/
sample_results/
cctv_results/
*.jpg
*.jsonl
```

- [ ] **Step 2: 초기화 및 커밋**

```bash
cd /Users/junyeong/Desktop/seatnow-cv
git init
git add .gitignore seatnow.py seatnow_core.py verify_seatnow.py occupancy_mvp.py pose_judge.py tests/ docs/ SEATNOW_전체정리.md
git commit -m "chore: initial commit of SeatNow inference pipeline

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Expected: `git log --oneline` 1줄, `git status`에 대용량 미디어/모델 파일 없음.
주의: `tests/fixtures/test_video_expectations.json`은 `.jsonl`이 아니라 `.json`이므로 포함된다.

---

### Task 2: `seatnow_layout.py` — 로드/검증/스케일

**Files:**
- Create: `seatnow_layout.py`
- Test: `tests/test_seatnow_layout.py`

**Interfaces:**
- Produces:
  - `LayoutChair(id: int, box: Tuple[float,float,float,float])` (frozen dataclass)
  - `LayoutTable(id: int, name: str, box: Box, chairs: List[LayoutChair])`
  - `SeatLayout(schema_version: int, source: Dict, tables: List[LayoutTable])`
    - `SeatLayout.scaled_to(width: int, height: int) -> SeatLayout`
    - `SeatLayout.chair_boxes() -> List[Tuple[float,...]]` (모든 테이블의 의자, 테이블 순서대로 평탄화)
    - `SeatLayout.chair_assignments() -> Dict[int, List[int]]` (테이블 인덱스 → `chair_boxes()` 기준 의자 인덱스)
  - `load_layout(path: Path) -> SeatLayout` — 실패 시 `LayoutError(메시지)`
  - `save_layout(layout: SeatLayout, path: Path) -> None`
  - `class LayoutError(ValueError)`

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_seatnow_layout.py`

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `./venv/bin/python -m unittest discover tests 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'seatnow_layout'`

- [ ] **Step 3: 구현** — `seatnow_layout.py`

```python
"""Manual seat layout: load/validate/scale the calibrated table-chair map.

The layout is the ground truth for seat geometry.  Detection only fills in
per-frame evidence (people, belongings) inside these zones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Tuple

Box = Tuple[float, float, float, float]

SCHEMA_VERSION = 1


class LayoutError(ValueError):
    """Raised when a layout file is missing or malformed."""


def _parse_box(value, context: str) -> Box:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or not all(isinstance(v, (int, float)) for v in value)
    ):
        raise LayoutError(f"{context}: box must be [x1, y1, x2, y2], got {value!r}")
    x1, y1, x2, y2 = (float(v) for v in value)
    if x2 <= x1 or y2 <= y1:
        raise LayoutError(f"{context}: box has non-positive size: {value!r}")
    return (x1, y1, x2, y2)


@dataclass(frozen=True)
class LayoutChair:
    id: int
    box: Box


@dataclass(frozen=True)
class LayoutTable:
    id: int
    name: str
    box: Box
    chairs: Tuple[LayoutChair, ...] = ()


@dataclass(frozen=True)
class SeatLayout:
    schema_version: int
    source: Dict[str, object]
    tables: Tuple[LayoutTable, ...]

    def scaled_to(self, width: int, height: int) -> "SeatLayout":
        src_width = float(self.source.get("width", width))
        src_height = float(self.source.get("height", height))
        sx, sy = width / src_width, height / src_height

        def scale(box: Box) -> Box:
            return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)

        tables = tuple(
            replace(
                table,
                box=scale(table.box),
                chairs=tuple(
                    replace(chair, box=scale(chair.box)) for chair in table.chairs
                ),
            )
            for table in self.tables
        )
        source = dict(self.source, width=width, height=height)
        return replace(self, source=source, tables=tables)

    def chair_boxes(self) -> List[Box]:
        return [chair.box for table in self.tables for chair in table.chairs]

    def chair_assignments(self) -> Dict[int, List[int]]:
        assignments: Dict[int, List[int]] = {}
        cursor = 0
        for index, table in enumerate(self.tables):
            count = len(table.chairs)
            assignments[index] = list(range(cursor, cursor + count))
            cursor += count
        return assignments


def load_layout(path: Path) -> SeatLayout:
    path = Path(path)
    if not path.exists():
        raise LayoutError(f"Layout file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LayoutError(f"Layout is not valid JSON: {path}: {exc}") from exc

    if data.get("schema_version") != SCHEMA_VERSION:
        raise LayoutError(
            f"Unsupported schema_version {data.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION}) in {path}"
        )
    raw_tables = data.get("tables")
    if not isinstance(raw_tables, list) or not raw_tables:
        raise LayoutError(f"Layout must contain at least one table: {path}")

    tables: List[LayoutTable] = []
    seen_ids = set()
    for position, raw in enumerate(raw_tables, start=1):
        table_id = raw.get("id")
        if not isinstance(table_id, int):
            raise LayoutError(f"table #{position}: id must be an integer")
        if table_id in seen_ids:
            raise LayoutError(f"table #{position}: duplicate table id {table_id}")
        seen_ids.add(table_id)
        chairs = tuple(
            LayoutChair(
                id=int(chair.get("id", chair_position)),
                box=_parse_box(
                    chair.get("box"), f"table {table_id} chair #{chair_position}"
                ),
            )
            for chair_position, chair in enumerate(raw.get("chairs", []), start=1)
        )
        tables.append(
            LayoutTable(
                id=table_id,
                name=str(raw.get("name", f"T{table_id}")),
                box=_parse_box(raw.get("box"), f"table {table_id}"),
                chairs=chairs,
            )
        )
    return SeatLayout(
        schema_version=SCHEMA_VERSION,
        source=dict(data.get("source", {})),
        tables=tuple(tables),
    )


def save_layout(layout: SeatLayout, path: Path) -> None:
    payload = {
        "schema_version": layout.schema_version,
        "source": layout.source,
        "tables": [
            {
                "id": table.id,
                "name": table.name,
                "box": [round(v, 2) for v in table.box],
                "chairs": [
                    {"id": chair.id, "box": [round(v, 2) for v in chair.box]}
                    for chair in table.chairs
                ],
            }
            for table in layout.tables
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
```

- [ ] **Step 4: 통과 확인**

Run: `./venv/bin/python -m unittest discover tests 2>&1 | tail -3`
Expected: 전체 OK (기존 54 + 신규 10 = 64개)

- [ ] **Step 5: 커밋**

```bash
git add seatnow_layout.py tests/test_seatnow_layout.py
git commit -m "feat: seat layout load/validate/scale module

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: 레이아웃 메타데이터를 관측/로그에 노출

**Files:**
- Modify: `seatnow_core.py` — `TableObservation` (114행 부근), `Track.label` (185행 부근), `track_to_dict`
- Test: `tests/test_seatnow_core.py` (기존 파일에 테스트 추가)

**Interfaces:**
- Consumes: 없음 (독립 변경)
- Produces:
  - `TableObservation.layout_id: Optional[int] = None`, `TableObservation.layout_name: Optional[str] = None`
  - `Track.label` — `layout_id`가 있으면 `"L{id:03d}"`
  - `track_to_dict` 결과에 `"layout_id"`, `"layout_name"` 키 (레이아웃 아니면 None)

- [ ] **Step 1: 실패하는 테스트 추가** — `tests/test_seatnow_core.py`의 `FrameLogTests` 클래스에 메서드 추가

```python
    def test_layout_observation_gets_layout_label_and_log_fields(self):
        observation = table_observation(
            OccupancyState.OCCUPIED, box=(0.0, 0.0, 40.0, 80.0)
        )
        observation.source = "layout"
        observation.layout_id = 7
        observation.layout_name = "창가1"
        track = Track(
            track_id=1,
            box=observation.box,
            stable_state=observation.raw_state,
            last_observation=observation,
            first_seen=0.0,
            last_seen=0.0,
        )

        record = track_to_dict(track)

        self.assertEqual(track.label, "L007")
        self.assertEqual(record["label"], "L007")
        self.assertEqual(record["layout_id"], 7)
        self.assertEqual(record["layout_name"], "창가1")
        self.assertEqual(record["source"], "layout")
```

`tests/test_seatnow_core.py` 상단 import에 `track_to_dict` 추가 (이미 `frame_log_record`는 있음):

```python
from seatnow_core import (
    ...
    track_to_dict,
)
```

- [ ] **Step 2: 실패 확인**

Run: `./venv/bin/python -m unittest discover tests 2>&1 | tail -5`
Expected: FAIL — `TableObservation`에 `layout_id` 속성 없음 (AttributeError)

- [ ] **Step 3: 구현** — `seatnow_core.py` 세 곳 수정

`TableObservation`에 필드 추가 (`provisional: bool = False` 아래):

```python
    layout_id: Optional[int] = None
    layout_name: Optional[str] = None
```

`Track.label` 교체:

```python
    @property
    def label(self) -> str:
        observation = self.last_observation
        if observation.layout_id is not None:
            return f"L{observation.layout_id:03d}"
        prefix = "S" if observation.source == "inferred-seat" else "T"
        return f"{prefix}{self.track_id:03d}"
```

`track_to_dict`의 반환 dict에 키 2개 추가 (`"source"` 바로 아래):

```python
        "layout_id": observation.layout_id,
        "layout_name": observation.layout_name,
```

- [ ] **Step 4: 통과 확인**

Run: `./venv/bin/python -m unittest discover tests 2>&1 | tail -3`
Expected: 전체 OK

- [ ] **Step 5: 커밋**

```bash
git add seatnow_core.py tests/test_seatnow_core.py
git commit -m "feat: expose layout id/name on observations, labels, and logs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `SeatNowAnalyzer` 레이아웃 모드

**Files:**
- Modify: `seatnow_core.py` — `SeatNowAnalyzer.__init__`, `SeatNowAnalyzer.analyze`

**Interfaces:**
- Consumes: `SeatLayout` (Task 2 — `chair_boxes()`, `chair_assignments()`, `tables[i].id/.name/.box`), Task 3의 `layout_id/layout_name` 필드
- Produces: `SeatNowAnalyzer(det_path, pose_path, config, layout: Optional[SeatLayout] = None)` — layout이 주어지면:
  - 테이블 = 레이아웃 존(conf 1.0), 의자 = 레이아웃 의자, 연결 = 레이아웃 링크(항상 점유 전파)
  - `select_table_candidates`/`deduplicate_tables`/`associate_chairs_to_tables`/`filter_strong_chair_links` 미실행
  - inferred-seat 비활성
  - 관측에 `source="layout"`, `layout_id`, `layout_name` 설정

- [ ] **Step 1: `__init__` 수정**

```python
    def __init__(
        self,
        det_model_path: Path,
        pose_model_path: Path,
        config: AnalyzerConfig,
        layout: "Optional[object]" = None,
    ):
        from ultralytics import YOLO

        self.config = config
        self.layout = layout
        ...  # 이하 기존 코드 유지
```

파일 상단에 순환 import 없이 타입만 쓰도록 주석 처리된 타입 사용 (`seatnow_layout`을 import하지 않는다 — analyzer는 duck-typing으로 `layout.tables`, `layout.chair_boxes()`, `layout.chair_assignments()`만 사용).

- [ ] **Step 2: `analyze()`의 테이블/의자 소스 분기** — 현재 코드:

```python
        seat_detections = [
            detection
            for detection in detections
            if detection.name in {"chair", "couch", "bench"}
            and detection.confidence >= 0.20
        ]
        table_candidates = select_table_candidates(
            table_candidates,
            seat_detections,
            (height, width),
            table_confidence=self.config.table_confidence,
            soft_area_fraction=self.config.maximum_table_area_fraction,
            large_table_confidence=self.config.large_table_confidence,
            hard_area_fraction=self.config.hard_table_area_fraction,
            rescue_confidence=self.config.table_rescue_confidence,
        )
        tables = deduplicate_tables(table_candidates, self.config.table_overlap)
```

를 다음으로 교체:

```python
        if self.layout is not None:
            tables = [
                Detection(name="dining table", box=table.box, confidence=1.0)
                for table in self.layout.tables
            ]
            seat_detections = [
                Detection(name="chair", box=box, confidence=1.0)
                for box in self.layout.chair_boxes()
            ]
        else:
            seat_detections = [
                detection
                for detection in detections
                if detection.name in {"chair", "couch", "bench"}
                and detection.confidence >= 0.20
            ]
            table_candidates = select_table_candidates(
                table_candidates,
                seat_detections,
                (height, width),
                table_confidence=self.config.table_confidence,
                soft_area_fraction=self.config.maximum_table_area_fraction,
                large_table_confidence=self.config.large_table_confidence,
                hard_area_fraction=self.config.hard_table_area_fraction,
                rescue_confidence=self.config.table_rescue_confidence,
            )
            tables = deduplicate_tables(table_candidates, self.config.table_overlap)
```

- [ ] **Step 3: 의자 연결 분기** — 현재 코드:

```python
        chair_table_assignments = associate_chairs_to_tables(
            tables, seat_detections, (height, width)
        )
        strong_chair_assignments = filter_strong_chair_links(
            tables, seat_detections, chair_table_assignments, (height, width)
        )
```

를 다음으로 교체:

```python
        if self.layout is not None:
            # 수동 연결은 항상 신뢰: 넓은/강한 링크 구분 없이 그대로 전파한다.
            chair_table_assignments = self.layout.chair_assignments()
            strong_chair_assignments = chair_table_assignments
        else:
            chair_table_assignments = associate_chairs_to_tables(
                tables, seat_detections, (height, width)
            )
            strong_chair_assignments = filter_strong_chair_links(
                tables, seat_detections, chair_table_assignments, (height, width)
            )
```

- [ ] **Step 4: 관측 메타데이터 + inferred-seat 비활성**

`observations.append(TableObservation(...))` 호출에 세 인자 추가:

```python
                    source="layout" if self.layout is not None else "detected",
                    layout_id=(
                        self.layout.tables[index].id
                        if self.layout is not None
                        else None
                    ),
                    layout_name=(
                        self.layout.tables[index].name
                        if self.layout is not None
                        else None
                    ),
```

inferred-seat 게이트 수정 — 현재 `if self.config.infer_occluded_tables:` 를:

```python
        if self.config.infer_occluded_tables and self.layout is None:
```

- [ ] **Step 5: 무회귀 확인**

Run: `./venv/bin/python -m unittest discover tests 2>&1 | tail -3`
Expected: 전체 OK (레이아웃 모드 자체는 Task 8 엔드투엔드에서 검증)

- [ ] **Step 6: 커밋**

```bash
git add seatnow_core.py
git commit -m "feat: layout mode in SeatNowAnalyzer (manual zones as ground truth)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `seatnow.py --layout` CLI

**Files:**
- Modify: `seatnow.py` — `build_parser`, `_validate_args`, `_make_analyzer`, `process_video`, `process_image`, `main`

**Interfaces:**
- Consumes: `load_layout`, `LayoutError` (Task 2), `SeatNowAnalyzer(layout=...)` (Task 4)
- Produces: `--layout layouts/매장.json` 옵션. JSONL `run.config`에 `layout` 항목(경로+sha256) 기록.

- [ ] **Step 1: 구현**

import 추가:

```python
from seatnow_layout import LayoutError, load_layout
```

`build_parser()`에 추가:

```python
    parser.add_argument("--layout", type=Path, help="Manual seat layout JSON (calibrate.py output); zones become ground truth")
```

`_validate_args()`에 추가:

```python
    if args.layout is not None and not args.layout.exists():
        raise FileNotFoundError(f"Layout not found: {args.layout}")
```

`_make_analyzer(args)`의 시그니처를 `_make_analyzer(args, layout=None)`로 바꾸고 마지막 줄을:

```python
    return SeatNowAnalyzer(det_path, pose_path, config, layout=layout)
```

`main()`에서 analyzer 생성 전에 레이아웃 로드 (해상도 스케일은 입력 크기를 알아야 하므로 여기서는 로드만, 스케일은 process_* 안에서):

```python
        layout = load_layout(args.layout) if args.layout else None
        analyzer = _make_analyzer(args, layout=layout)
```

`process_video()` 시작부(`info = probe_video(...)` 다음)에 스케일 적용:

```python
    if analyzer.layout is not None:
        analyzer.layout = analyzer.layout.scaled_to(info.width, info.height)
```

`process_image()`에서도 `frame` 읽은 다음:

```python
    if analyzer.layout is not None:
        height, width = frame.shape[:2]
        analyzer.layout = analyzer.layout.scaled_to(width, height)
```

`process_video()`의 `run_context["config"]`에 항목 추가:

```python
            "layout": (
                {"path": str(args.layout), "sha256": _sha256(args.layout)}
                if args.layout
                else None
            ),
```

`LayoutError`는 `main()`의 기존 `except Exception` 블록이 잡아 `SeatNow error: ...`로 출력되므로 별도 처리 불필요.

- [ ] **Step 2: 스모크 확인 (레이아웃 없이 무회귀 + 잘못된 경로 에러)**

```bash
./venv/bin/python seatnow.py --help | grep -A1 -- --layout
./venv/bin/python seatnow.py sample_raw/cafe_sample_1.mp4 --layout /no/such.json --max-samples 1 --no-video; echo "exit=$?"
```

Expected: help에 `--layout` 표시 / 두 번째 명령은 `SeatNow error: Layout not found: /no/such.json`, `exit=1`

- [ ] **Step 3: 커밋**

```bash
git add seatnow.py
git commit -m "feat: --layout CLI option wiring manual layouts into the pipeline

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `CalibrationState` — GUI 없는 편집 상태 기계

**Files:**
- Create: `calibrate.py` (이 태스크에서는 상태 클래스만; GUI는 Task 7)
- Test: `tests/test_calibrate_state.py`

**Interfaces:**
- Consumes: `SeatLayout`, `LayoutTable`, `LayoutChair`, `SCHEMA_VERSION` (Task 2)
- Produces (calibrate.py 내):
  - `class CalibrationState:`
    - `tables: List[Dict]` — 각 원소 `{"box": Box, "chairs": List[Box]}`
    - `selected: Optional[Tuple[str, int, int]]` — `("table", ti, -1)` 또는 `("chair", ti, ci)`
    - `add_table(box: Box) -> None` (추가 후 그 테이블 선택)
    - `add_chair(box: Box) -> bool` — 선택된 테이블에 연결, 선택 없으면 False
    - `select_at(x: float, y: float) -> None` — 점을 포함하는 가장 작은 박스 선택(없으면 해제)
    - `delete_selected() -> None` — 테이블 삭제 시 소속 의자도 삭제
    - `undo() -> None` — 직전 변경 1회 되돌리기
    - `to_layout(source: Dict) -> SeatLayout` — id는 1부터 순번
    - `classmethod from_layout(layout: SeatLayout) -> CalibrationState`

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_calibrate_state.py`

```python
"""Unit tests for the calibration editing state (no GUI)."""

from __future__ import annotations

import unittest

from calibrate import CalibrationState
from seatnow_layout import SeatLayout


class CalibrationStateTests(unittest.TestCase):
    def test_add_table_selects_it_and_chairs_attach_to_selection(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        state.add_table((500.0, 100.0, 700.0, 200.0))
        # 두 번째 테이블이 선택된 상태 → 의자는 거기에 붙는다
        self.assertTrue(state.add_chair((710.0, 110.0, 760.0, 190.0)))

        self.assertEqual(len(state.tables[0]["chairs"]), 0)
        self.assertEqual(len(state.tables[1]["chairs"]), 1)

    def test_add_chair_without_table_returns_false(self):
        state = CalibrationState()
        self.assertFalse(state.add_chair((10.0, 10.0, 20.0, 20.0)))

    def test_select_at_picks_smallest_containing_box(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 400.0, 300.0))
        state.add_chair((110.0, 110.0, 160.0, 180.0))

        state.select_at(120.0, 120.0)  # 의자와 테이블 둘 다 포함 → 작은 쪽(의자)
        self.assertEqual(state.selected[0], "chair")

        state.select_at(350.0, 250.0)  # 테이블만 포함
        self.assertEqual(state.selected[0], "table")

        state.select_at(900.0, 900.0)  # 아무것도 없음
        self.assertIsNone(state.selected)

    def test_delete_selected_table_removes_its_chairs(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 400.0, 300.0))
        state.add_chair((110.0, 110.0, 160.0, 180.0))
        state.select_at(350.0, 250.0)

        state.delete_selected()

        self.assertEqual(state.tables, [])

    def test_undo_restores_previous_step(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 400.0, 300.0))
        state.add_chair((110.0, 110.0, 160.0, 180.0))

        state.undo()

        self.assertEqual(len(state.tables), 1)
        self.assertEqual(state.tables[0]["chairs"], [])

    def test_to_layout_assigns_sequential_ids(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        state.add_chair((40.0, 110.0, 90.0, 190.0))
        state.add_table((500.0, 100.0, 700.0, 200.0))

        layout = state.to_layout({"video": "v.mp4", "width": 1280, "height": 720})

        self.assertIsInstance(layout, SeatLayout)
        self.assertEqual([t.id for t in layout.tables], [1, 2])
        self.assertEqual(layout.tables[0].chairs[0].id, 1)

    def test_from_layout_round_trip(self):
        state = CalibrationState()
        state.add_table((100.0, 100.0, 300.0, 200.0))
        state.add_chair((40.0, 110.0, 90.0, 190.0))
        layout = state.to_layout({"width": 1280, "height": 720})

        restored = CalibrationState.from_layout(layout)

        self.assertEqual(restored.tables[0]["box"], (100.0, 100.0, 300.0, 200.0))
        self.assertEqual(restored.tables[0]["chairs"], [(40.0, 110.0, 90.0, 190.0)])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run: `./venv/bin/python -m unittest discover tests 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'calibrate'`

- [ ] **Step 3: 구현** — `calibrate.py` (상태 부분)

```python
"""SeatNow calibration: register table/chair zones and their links.

State machine is GUI-free (unit tested); the OpenCV window is a thin shell.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from seatnow_layout import (
    SCHEMA_VERSION,
    LayoutChair,
    LayoutTable,
    SeatLayout,
)

Box = Tuple[float, float, float, float]


def _contains(box: Box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


class CalibrationState:
    def __init__(self) -> None:
        self.tables: List[Dict] = []  # {"box": Box, "chairs": List[Box]}
        self.selected: Optional[Tuple[str, int, int]] = None
        self._history: List[Tuple[List[Dict], Optional[Tuple[str, int, int]]]] = []

    def _snapshot(self) -> None:
        self._history.append((copy.deepcopy(self.tables), self.selected))

    def add_table(self, box: Box) -> None:
        self._snapshot()
        self.tables.append({"box": tuple(box), "chairs": []})
        self.selected = ("table", len(self.tables) - 1, -1)

    def add_chair(self, box: Box) -> bool:
        table_index = self._selected_table_index()
        if table_index is None:
            return False
        self._snapshot()
        self.tables[table_index]["chairs"].append(tuple(box))
        return True

    def _selected_table_index(self) -> Optional[int]:
        if self.selected is None:
            return None
        return self.selected[1]

    def select_at(self, x: float, y: float) -> None:
        candidates: List[Tuple[float, Tuple[str, int, int]]] = []
        for ti, table in enumerate(self.tables):
            if _contains(table["box"], x, y):
                candidates.append((_area(table["box"]), ("table", ti, -1)))
            for ci, chair in enumerate(table["chairs"]):
                if _contains(chair, x, y):
                    candidates.append((_area(chair), ("chair", ti, ci)))
        self.selected = min(candidates)[1] if candidates else None

    def delete_selected(self) -> None:
        if self.selected is None:
            return
        self._snapshot()
        kind, ti, ci = self.selected
        if kind == "table":
            del self.tables[ti]
        else:
            del self.tables[ti]["chairs"][ci]
        self.selected = None

    def undo(self) -> None:
        if not self._history:
            return
        self.tables, self.selected = self._history.pop()

    def to_layout(self, source: Dict) -> SeatLayout:
        tables = tuple(
            LayoutTable(
                id=index,
                name=f"T{index}",
                box=table["box"],
                chairs=tuple(
                    LayoutChair(id=chair_index, box=chair)
                    for chair_index, chair in enumerate(table["chairs"], start=1)
                ),
            )
            for index, table in enumerate(self.tables, start=1)
        )
        return SeatLayout(
            schema_version=SCHEMA_VERSION, source=dict(source), tables=tables
        )

    @classmethod
    def from_layout(cls, layout: SeatLayout) -> "CalibrationState":
        state = cls()
        for table in layout.tables:
            state.tables.append(
                {
                    "box": tuple(table.box),
                    "chairs": [tuple(chair.box) for chair in table.chairs],
                }
            )
        state.selected = None
        return state
```

주의: `undo()`에서 `selected`가 삭제된 인덱스를 가리키지 않도록 스냅샷에 selected도 저장/복원한다 (위 코드 반영).

- [ ] **Step 4: 통과 확인**

Run: `./venv/bin/python -m unittest discover tests 2>&1 | tail -3`
Expected: 전체 OK

- [ ] **Step 5: 커밋**

```bash
git add calibrate.py tests/test_calibrate_state.py
git commit -m "feat: GUI-free calibration editing state

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: `calibrate.py` GUI + 프리시드 + main

**Files:**
- Modify: `calibrate.py` (Task 6 파일에 GUI/CLI 추가)

**Interfaces:**
- Consumes: `CalibrationState` (Task 6), `save_layout`/`load_layout` (Task 2), `probe_video`·`FFmpegSampleReader`(seatnow_core — 프레임 추출은 ffmpeg subprocess로 단순화)
- Produces: CLI `./venv/bin/python calibrate.py <video> [--at 0] [--output layouts/<stem>.json] [--edit 기존.json] [--det-model yolov8x.pt] [--no-preseed]`

- [ ] **Step 1: GUI/CLI 구현** — `calibrate.py`에 이어서 추가

```python
def _grab_frame(video, at_seconds):
    import subprocess
    import tempfile
    from pathlib import Path

    import cv2

    out = Path(tempfile.mkdtemp()) / "calib_frame.png"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-ss", str(at_seconds),
            "-i", str(video), "-frames:v", "1", "-y", str(out),
        ],
        check=True,
    )
    frame = cv2.imread(str(out))
    if frame is None:
        raise RuntimeError(f"Could not extract frame at {at_seconds}s from {video}")
    return frame


def _preseed(frame, det_model_path):
    """Pre-populate state with detected tables and auto-linked chairs."""
    from ultralytics import YOLO

    from seatnow_core import (
        Detection,
        associate_chairs_to_tables,
        deduplicate_tables,
    )

    model = YOLO(str(det_model_path))
    result = model.predict(source=frame, conf=0.12, imgsz=1280, device="cpu", verbose=False)[0]
    tables, chairs = [], []
    if result.boxes is not None:
        for box_result in result.boxes:
            name = result.names[int(box_result.cls.item())]
            confidence = float(box_result.conf.item())
            box = tuple(float(v) for v in box_result.xyxy[0].tolist())
            if name == "dining table":
                tables.append(Detection(name=name, box=box, confidence=confidence))
            elif name in {"chair", "couch", "bench"} and confidence >= 0.35:
                chairs.append(Detection(name=name, box=box, confidence=confidence))
    tables = deduplicate_tables(tables)
    height, width = frame.shape[:2]
    links = associate_chairs_to_tables(tables, chairs, (height, width))

    state = CalibrationState()
    for table_index, table in enumerate(tables):
        state.add_table(table.box)
        for chair_index in links[table_index]:
            state.add_chair(chairs[chair_index].box)
    state.selected = None
    state._history.clear()
    return state


HELP_TEXT = "[t]able  [c]hair->selected  [d]elete  [u]ndo  [s]ave  [q]uit"

TABLE_COLOR = (80, 200, 80)
CHAIR_COLOR = (60, 200, 230)
SELECT_COLOR = (60, 60, 235)


def _draw(frame, state, mode, drag):
    import cv2

    canvas = frame.copy()
    for ti, table in enumerate(state.tables):
        tx1, ty1, tx2, ty2 = [int(v) for v in table["box"]]
        selected = state.selected == ("table", ti, -1)
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2),
                      SELECT_COLOR if selected else TABLE_COLOR, 3 if selected else 2)
        cv2.putText(canvas, f"T{ti + 1}", (tx1, max(16, ty1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TABLE_COLOR, 2, cv2.LINE_AA)
        tcx, tcy = (tx1 + tx2) // 2, (ty1 + ty2) // 2
        for ci, chair in enumerate(table["chairs"]):
            cx1, cy1, cx2, cy2 = [int(v) for v in chair]
            chair_selected = state.selected == ("chair", ti, ci)
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2),
                          SELECT_COLOR if chair_selected else CHAIR_COLOR, 2)
            cv2.line(canvas, (tcx, tcy), ((cx1 + cx2) // 2, (cy1 + cy2) // 2),
                     CHAIR_COLOR, 1, cv2.LINE_AA)
    if drag is not None:
        (x1, y1), (x2, y2) = drag
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), SELECT_COLOR, 1)
    cv2.putText(canvas, f"mode={mode}  {HELP_TEXT}", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def run_gui(frame, state, output_path, source):
    import cv2

    from seatnow_layout import save_layout

    mode = "table"
    drag_start = None
    drag_current = None
    window = "SeatNow calibrate"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, _param):
        nonlocal drag_start, drag_current
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (x, y)
            drag_current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            x1, y1 = drag_start
            box = (min(x1, x), min(y1, y), max(x1, x), max(y1, y))
            drag_start = None
            drag_current = None
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                state.select_at(x, y)  # 클릭 = 선택
            elif mode == "table":
                state.add_table(box)
            else:
                if not state.add_chair(box):
                    print("의자를 붙일 테이블을 먼저 클릭으로 선택하세요")

    cv2.setMouseCallback(window, on_mouse)
    while True:
        drag = (drag_start, drag_current) if drag_start is not None else None
        cv2.imshow(window, _draw(frame, state, mode, drag))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("t"):
            mode = "table"
        elif key == ord("c"):
            mode = "chair"
        elif key == ord("d"):
            state.delete_selected()
        elif key == ord("u"):
            state.undo()
        elif key == ord("s"):
            layout = state.to_layout(source)
            if not layout.tables:
                print("테이블이 없어 저장하지 않았습니다")
                continue
            save_layout(layout, output_path)
            print(f"저장됨: {output_path} (테이블 {len(layout.tables)}개, "
                  f"의자 {len(layout.chair_boxes())}개)")
        elif key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()


def main(argv=None) -> int:
    import argparse
    from pathlib import Path

    from seatnow_layout import load_layout

    parser = argparse.ArgumentParser(description="Register table/chair zones for SeatNow")
    parser.add_argument("video", type=Path, help="Reference video (first frame is calibrated)")
    parser.add_argument("--at", type=float, default=0.0, help="Timestamp (s) of the reference frame")
    parser.add_argument("--output", type=Path, help="Layout JSON path (default: layouts/<video stem>.json)")
    parser.add_argument("--edit", type=Path, help="Load an existing layout instead of auto pre-seeding")
    parser.add_argument("--det-model", default="yolov8x.pt", help="Detector for pre-seeding (one-time, accuracy first)")
    parser.add_argument("--no-preseed", action="store_true", help="Start from an empty canvas")
    args = parser.parse_args(argv)

    if not args.video.exists():
        print(f"video not found: {args.video}")
        return 1
    frame = _grab_frame(args.video, args.at)
    height, width = frame.shape[:2]

    if args.edit is not None:
        state = CalibrationState.from_layout(
            load_layout(args.edit).scaled_to(width, height)
        )
    elif args.no_preseed:
        state = CalibrationState()
    else:
        print("자동 탐지로 초기 배치를 채우는 중... (수십 초)")
        state = _preseed(frame, args.det_model)

    output = args.output or Path("layouts") / f"{args.video.stem}.json"
    source = {
        "video": str(args.video),
        "frame_at_seconds": args.at,
        "width": width,
        "height": height,
    }
    run_gui(frame, state, output, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 유닛 무회귀 + 프리시드 헤드리스 확인**

```bash
./venv/bin/python -m unittest discover tests 2>&1 | tail -3
./venv/bin/python - <<'EOF'
from calibrate import _grab_frame, _preseed
frame = _grab_frame("sample_raw/cafe_sample_좌석착석.mp4", 0.0)
state = _preseed(frame, "yolov8x.pt")
print("tables:", len(state.tables), "chairs:", sum(len(t["chairs"]) for t in state.tables))
assert len(state.tables) >= 2
EOF
```

Expected: 테스트 전체 OK / `tables: 2` 이상 출력

- [ ] **Step 3: 수동 GUI 확인 (사용자와 함께)**

```bash
./venv/bin/python calibrate.py sample_raw/cafe_sample_좌석착석.mp4 --output layouts/seminar_room.json
```

확인 항목: 프리시드 박스 표시, 드래그로 테이블/의자 추가, 클릭 선택(빨간 강조), d/u/s/q 동작, 저장 메시지.

- [ ] **Step 4: 커밋**

```bash
git add calibrate.py
git commit -m "feat: OpenCV calibration GUI with auto pre-seeding

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: 엔드투엔드 검증 + 문서

**Files:**
- Create: `layouts/seminar_room.json` (Task 7에서 저장한 것, 또는 아래 수기 좌표)
- Modify: `SEATNOW_전체정리.md` — 3절 표에 `calibrate.py`/`seatnow_layout.py` 행 추가, 2절에 "레이아웃 모드" 언급

**Interfaces:**
- Consumes: 전체 파이프라인.

- [ ] **Step 1: 레이아웃 준비** — Task 7 GUI 결과가 없으면 probe 좌표로 수기 작성:

```json
{
  "schema_version": 1,
  "source": {"video": "sample_raw/cafe_sample_좌석착석.mp4", "frame_at_seconds": 0.0, "width": 1280, "height": 720},
  "tables": [
    {"id": 1, "name": "뒤왼쪽", "box": [425, 281, 545, 407],
     "chairs": [{"id": 1, "box": [309, 309, 445, 482]}, {"id": 2, "box": [343, 268, 386, 342]}]},
    {"id": 2, "name": "뒤오른쪽", "box": [688, 263, 818, 454],
     "chairs": [{"id": 1, "box": [580, 278, 691, 442]}, {"id": 2, "box": [750, 246, 847, 389]}]},
    {"id": 3, "name": "앞컵테이블", "box": [457, 422, 701, 713],
     "chairs": [{"id": 1, "box": [317, 595, 519, 716]}, {"id": 2, "box": [408, 456, 492, 610]}]}
  ]
}
```

- [ ] **Step 2: 레이아웃 모드 실행**

```bash
./venv/bin/python seatnow.py "sample_raw/cafe_sample_좌석착석.mp4" \
  --layout layouts/seminar_room.json \
  --output sample_results/layout_좌석착석.mp4 \
  --log sample_results/layout_좌석착석.jsonl --debug
```

- [ ] **Step 3: 결과 검증**

```bash
./venv/bin/python - <<'EOF'
import json
rows = [json.loads(l) for l in open("sample_results/layout_좌석착석.jsonl")]
final = rows[-1]
labels = {t["label"]: t["state"] for t in final["tables"]}
assert len(final["tables"]) == 3, labels          # 존 3개 고정
assert all(t["source"] == "layout" for t in final["tables"])
assert labels["L003"] == "occupied", labels       # 앞컵테이블: 소품 컵
assert labels["L002"] == "occupied", labels       # 뒤오른쪽: 착석
print("layout e2e OK:", labels)
EOF
```

Expected: `layout e2e OK: {...}` (실패 시 프레임 렌더를 뽑아 원인 확인 후 조정 — 존 좌표 오차 가능성부터 본다)

- [ ] **Step 4: 문서 갱신** — `SEATNOW_전체정리.md`
  - 3절 표에 두 행 추가:
    `| seatnow_layout.py | 수동 좌석 레이아웃 로드/검증/스케일 |`
    `| calibrate.py | 클릭 캘리브레이션 도구 (테이블·의자 등록 → layouts/*.json) |`
  - 2절 끝에 한 단락: "레이아웃 모드(`--layout`): 캘리브레이션된 존이 좌석 기준. 테이블/의자 탐지는 껐고 사람·물체 증거만 자동. 좌석 수 고정 → 앱 표시 안정. 경량 모델(yolov8n)과 병용 전제."
  - 7절 기술 스택 표의 "CV 추론"에 yolov8n 기본/x 옵션 반영

- [ ] **Step 5: 전체 테스트 + 커밋**

```bash
./venv/bin/python -m unittest discover tests 2>&1 | tail -3
git add layouts/seminar_room.json SEATNOW_전체정리.md
git commit -m "feat: sample layout, e2e verification, docs for layout mode

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Expected: 전체 OK 후 커밋 완료.

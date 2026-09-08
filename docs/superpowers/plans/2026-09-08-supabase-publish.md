# 박스 → Supabase 전송 + 지도 초안 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 엣지 박스가 판정 한 번마다 "지금 값"을 Supabase `cafe_live` 에 쓰고, 지도(`cafe_maps`)를 올리며, 바닥 네 점 없이도 지도 초안이 나오게 한다. 앱 개발자용 문서와 Supabase SQL 까지 포함.

**Architecture:** `edge/publish.py` 에 순수 함수(`live_payload`, `gap_payload`, `seat_index_from_layout`)와 백그라운드 스레드 `SupabasePublisher`(최신 값 하나만 보관, 실패는 버리고 백오프)를 둔다. `engine/seatnow.py` 의 `process_live` 는 환경변수가 있을 때만 publisher 를 만들고 판정 줄·gap 줄 직후에 넘긴다. `install/floorplan.build_draft` 는 바닥 기준점이 없으면 화면 좌표를 그대로 캔버스에 놓는다.

**Tech Stack:** Python 3 (박스 venv), `requests` (이미 requirements-edge.txt), `unittest` (`./venv/Scripts/python.exe -m unittest discover tests`), Supabase PostgREST + Auth (password grant), PostgreSQL RLS.

## Global Constraints

- 판정 루프를 절대 막지 않는다: 전송은 스레드, HTTP 타임아웃 5초, 예외는 삼키고 카운트.
- 환경변수 5개(`SEATNOW_CAFE_ID`, `SEATNOW_SUPABASE_URL`, `SEATNOW_SUPABASE_ANON_KEY`, `SEATNOW_SUPABASE_EMAIL`, `SEATNOW_SUPABASE_PASSWORD`)가 하나라도 없으면 전송은 꺼지고 나머지는 지금과 완전히 같다.
- `state` 는 `occupied`/`empty`/`unknown` 만. `ignore` 는 목록에 없다. `free_tables` 는 확실한 `empty` 만.
- `cafe_live.schema_version = 1`. gap 일 때 `status="gap"`, 모든 자리 `unknown`, `reason_code="no_fresh_frames"`.
- 바 자리는 칸마다 한 줄, `kind="bar_seat"`, `zone` 에 구역 이름.
- 테스트 명령: `./venv/Scripts/python.exe -m unittest tests.test_publish -v` (윈도우 노트북). 박스에서는 `./venv/bin/python`.
- 비밀번호·키는 `deploy/seatnow.env` 에만 (git 제외). 저장소·문서·테스트에 실제 값을 쓰지 않는다.
- 커밋 메시지 끝에 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` 와 `Claude-Session: https://claude.ai/code/session_012hqXWyD5keYy9MEGZu59Zk`.

## 파일 구조

| 파일 | 역할 |
|---|---|
| `edge/publish.py` (새) | payload 순수 함수, `SupabasePublisher`, `FloorplanWatcher`, `publisher_from_env`, `box_version` |
| `tests/test_publish.py` (새) | 위 전부. 가짜 HTTP 서버로 publisher 시험 |
| `engine/seatnow.py` (수정) | `process_live` 에 publisher 연결 (시작·판정 줄·gap 줄·지도·요약) |
| `edge/live_report.py` (수정) | 마지막 기록의 전송 카운트 표시 |
| `tests/test_live_report.py` (수정) | 전송 카운트 요약 테스트 |
| `install/floorplan.py` (수정) | 바닥 기준점 없을 때 화면 좌표 배치, `main()` CLI |
| `tests/test_floorplan.py` (수정) | 거부 테스트를 새 동작 테스트로 교체 |
| `deploy/supabase/schema.sql` (새) | 표·트리거·RLS·realtime |
| `deploy/seatnow.env.example` (수정) | 환경변수 5줄 |
| `docs/앱연동.md` (새) | 앱 개발자 문서 |
| `카페설치당일.md`, `다음할일.md` (수정) | 6-3 단계에 평면도 초안 명령, 남은 일 갱신 |

---

### Task 1: payload 순수 함수 (`live_payload`, `gap_payload`, `seat_index_from_layout`)

**Files:**
- Create: `edge/publish.py`
- Test: `tests/test_publish.py`

**Interfaces:**
- Produces:
  - `SEATS_SCHEMA_VERSION: int = 1`
  - `GAP_REASON: str = "no_fresh_frames"`
  - `seat_index_from_layout(layout: SeatLayout) -> List[dict]` — `[{"seat_id": str, "kind": "table"|"bar_seat", "zone": Optional[str]}]`
  - `live_payload(record: dict, cafe_id: str, box_version: str) -> dict` — `cafe_live` 한 줄
  - `gap_payload(seat_index: List[dict], cafe_id: str, box_version: str, wall_clock: str) -> dict`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_publish.py`:

```python
"""박스 → Supabase 전송: payload 와 publisher."""

from __future__ import annotations

import unittest

from edge.publish import (
    GAP_REASON,
    SEATS_SCHEMA_VERSION,
    gap_payload,
    live_payload,
    seat_index_from_layout,
)
from engine.seatnow_layout import LayoutChair, LayoutSeat, LayoutTable, SeatLayout


def _table(name, state, *, kind="table", zone=None, reason="", raw_state=None, predicted=False):
    return {
        "layout_name": name,
        "label": "L00X",
        "layout_kind": kind,
        "layout_zone_name": zone,
        "state": state,
        "raw_state": raw_state or state,
        "reason": reason,
        "predicted": predicted,
        "confidence": 0.9,
    }


def _record(tables):
    return {
        "wall_clock": "2026-09-08T20:32:45+0900",
        "tables": tables,
    }


class LivePayloadTest(unittest.TestCase):
    def test_counts_by_table_and_only_confirmed_empties_are_free(self):
        record = _record(
            [
                _table("T1", "occupied"),
                _table("T2", "empty"),
                _table("T3", "unknown", reason="compact_occluded_pose"),
            ]
        )
        payload = live_payload(record, "dwnc", "abc1234")
        self.assertEqual(payload["cafe_id"], "dwnc")
        self.assertEqual(payload["status"], "live")
        self.assertEqual(payload["total_tables"], 3)
        self.assertEqual(payload["occupied_tables"], 1)
        self.assertEqual(payload["free_tables"], 1)
        self.assertEqual(payload["unknown_tables"], 1)
        self.assertEqual(payload["tick_at"], "2026-09-08T20:32:45+0900")
        self.assertEqual(payload["box_version"], "abc1234")
        self.assertEqual(payload["schema_version"], SEATS_SCHEMA_VERSION)

    def test_ignored_seats_are_left_out(self):
        record = _record([_table("T1", "occupied"), _table("T9", "ignore")])
        payload = live_payload(record, "dwnc", "v")
        self.assertEqual([s["seat_id"] for s in payload["seats"]], ["T1"])
        self.assertEqual(payload["total_tables"], 1)

    def test_bar_slots_are_one_row_each_with_zone(self):
        record = _record(
            [
                _table("BAR7-1", "empty", kind="counted_zone", zone="BAR7"),
                _table("BAR7-2", "occupied", kind="counted_zone", zone="BAR7"),
            ]
        )
        seats = live_payload(record, "dwnc", "v")["seats"]
        self.assertEqual(
            seats,
            [
                {"seat_id": "BAR7-1", "kind": "bar_seat", "zone": "BAR7", "state": "empty", "reason_code": None},
                {"seat_id": "BAR7-2", "kind": "bar_seat", "zone": "BAR7", "state": "occupied", "reason_code": None},
            ],
        )

    def test_reason_code_only_for_unknown_and_from_closed_vocabulary(self):
        record = _record(
            [
                _table("T1", "unknown", reason="compact_occluded_pose"),
                _table("T2", "occupied", reason="person_seated"),
            ]
        )
        seats = live_payload(record, "dwnc", "v")["seats"]
        self.assertEqual(seats[0]["reason_code"], "compact_occluded_pose")
        self.assertIsNone(seats[1]["reason_code"])

    def test_seat_id_falls_back_to_label(self):
        table = _table("", "empty")
        table["layout_name"] = None
        payload = live_payload(_record([table]), "dwnc", "v")
        self.assertEqual(payload["seats"][0]["seat_id"], "L00X")


class GapPayloadTest(unittest.TestCase):
    def test_everything_unknown_with_gap_reason(self):
        index = [
            {"seat_id": "T1", "kind": "table", "zone": None},
            {"seat_id": "BAR7-1", "kind": "bar_seat", "zone": "BAR7"},
        ]
        payload = gap_payload(index, "dwnc", "v", "2026-09-08T21:00:00+0900")
        self.assertEqual(payload["status"], "gap")
        self.assertEqual(payload["total_tables"], 2)
        self.assertEqual(payload["occupied_tables"], 0)
        self.assertEqual(payload["free_tables"], 0)
        self.assertEqual(payload["unknown_tables"], 2)
        self.assertEqual(payload["tick_at"], "2026-09-08T21:00:00+0900")
        for seat in payload["seats"]:
            self.assertEqual(seat["state"], "unknown")
            self.assertEqual(seat["reason_code"], GAP_REASON)
        self.assertEqual(payload["seats"][1]["zone"], "BAR7")


class SeatIndexTest(unittest.TestCase):
    def test_index_lists_every_judgement_unit(self):
        layout = SeatLayout(
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
                    seats=(
                        LayoutSeat(id=1, box=(1100.0, 620.0, 1300.0, 760.0)),
                        LayoutSeat(id=2, box=(1300.0, 620.0, 1500.0, 760.0)),
                    ),
                ),
            ),
        )
        index = seat_index_from_layout(layout)
        self.assertEqual(
            index,
            [
                {"seat_id": "T1", "kind": "table", "zone": None},
                {"seat_id": "BAR7-1", "kind": "bar_seat", "zone": "BAR7"},
                {"seat_id": "BAR7-2", "kind": "bar_seat", "zone": "BAR7"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
```

`LayoutSeat` 생성자 인자는 `tests/test_floorplan.py` 의 `layout()` 헬퍼와 같은 모양을 쓴다. 이름이 `BAR7-1` 로 붙는지는 `engine/seatnow_layout.py` 의 `judgement_units()` 가 정한다 (테스트가 실패하면 그 함수가 만드는 실제 이름을 assert 에 맞춘다 — 이름 규칙은 바꾸지 않는다).

- [ ] **Step 2: 실패 확인**

Run: `./venv/Scripts/python.exe -m unittest tests.test_publish -v`
Expected: `ModuleNotFoundError: No module named 'edge.publish'`

- [ ] **Step 3: 구현**

`edge/publish.py` (첫 부분 — publisher 는 Task 2 에서 이 파일에 이어 붙인다):

```python
"""박스 → Supabase: 판정 한 번마다 "지금 값"을 쓴다.

세 가지 순수 함수가 JSONL 한 줄을 ``cafe_live`` 한 줄로 바꾸고,
``SupabasePublisher`` 가 그것을 백그라운드에서 보낸다.  판정 루프는 여기서
절대 기다리지 않는다 -- 인터넷이 끊기면 값을 버리고, 앱은 45초 규칙으로
"확인 중"이 된다 (docs/앱연동.md).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from engine.seatnow_layout import COUNTED_ZONE_KIND, SeatLayout
from engine.seatnow_report import classify_reason

SEATS_SCHEMA_VERSION = 1
GAP_REASON = "no_fresh_frames"
_COUNTABLE = ("occupied", "empty", "unknown")

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _kind(layout_kind: Optional[str]) -> str:
    return "bar_seat" if layout_kind == COUNTED_ZONE_KIND else "table"


def seat_index_from_layout(layout: SeatLayout) -> List[Dict[str, Any]]:
    """Every judgement unit as {seat_id, kind, zone} -- what a gap row lists."""
    return [
        {"seat_id": unit.name, "kind": _kind(unit.kind), "zone": unit.zone_name}
        for unit in layout.judgement_units()
    ]


def _totals(seats: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"occupied": 0, "empty": 0, "unknown": 0}
    for seat in seats:
        counts[seat["state"]] += 1
    return {
        "total_tables": len(seats),
        "occupied_tables": counts["occupied"],
        "free_tables": counts["empty"],
        "unknown_tables": counts["unknown"],
    }


def _row(cafe_id: str, status: str, seats: List[Dict[str, Any]], tick_at: str, box_version: str) -> Dict[str, Any]:
    row = {"cafe_id": cafe_id, "status": status}
    row.update(_totals(seats))
    row.update(
        {
            "seats": seats,
            "tick_at": tick_at,
            "box_version": box_version,
            "schema_version": SEATS_SCHEMA_VERSION,
        }
    )
    return row


def live_payload(record: Dict[str, Any], cafe_id: str, box_version: str) -> Dict[str, Any]:
    """One JSONL tick -> one ``cafe_live`` row.

    Bar slots stay one row each (the map colours slots, not zones), ``ignore``
    seats are absent, and ``reason_code`` exists only for ``unknown``.
    """
    seats: List[Dict[str, Any]] = []
    for table in record.get("tables") or []:
        state = str(table.get("state", "unknown"))
        if state not in _COUNTABLE:
            continue
        reason_code = None
        if state == "unknown":
            reason_code = classify_reason(
                str(table.get("raw_state", state)),
                str(table.get("reason", "")),
                bool(table.get("predicted", False)),
            ).value
        seats.append(
            {
                "seat_id": str(table.get("layout_name") or table.get("label") or "?"),
                "kind": _kind(table.get("layout_kind")),
                "zone": table.get("layout_zone_name"),
                "state": state,
                "reason_code": reason_code,
            }
        )
    return _row(cafe_id, "live", seats, str(record.get("wall_clock", "")), box_version)


def gap_payload(seat_index: List[Dict[str, Any]], cafe_id: str, box_version: str, wall_clock: str) -> Dict[str, Any]:
    """The camera went quiet: every seat is unknown, never "what it was"."""
    seats = [
        {
            "seat_id": entry["seat_id"],
            "kind": entry["kind"],
            "zone": entry.get("zone"),
            "state": "unknown",
            "reason_code": GAP_REASON,
        }
        for entry in seat_index
    ]
    return _row(cafe_id, "gap", seats, wall_clock, box_version)
```

- [ ] **Step 4: 통과 확인**

Run: `./venv/Scripts/python.exe -m unittest tests.test_publish -v`
Expected: 7 tests OK. `SeatIndexTest` 가 이름 때문에 실패하면 `judgement_units()` 의 실제 이름을 확인해 assert 를 그 이름으로 고친다.

- [ ] **Step 5: 커밋**

```bash
git add edge/publish.py tests/test_publish.py
git commit -m "feat(전송): 판정 한 줄 → cafe_live 한 줄로 바꾸는 순수 함수 (바 칸 단위, ignore 제외, gap 은 전부 unknown)"
```

---

### Task 2: `SupabasePublisher` — 백그라운드 전송, 최신 값 하나, 실패는 버림

**Files:**
- Modify: `edge/publish.py` (Task 1 뒤에 이어 붙임)
- Test: `tests/test_publish.py` (이어 붙임)

**Interfaces:**
- Consumes: Task 1 의 payload dict.
- Produces:
  - `class SupabasePublisher(url, anon_key, email, password, cafe_id, *, timeout=5.0, session=None, sleep=time.sleep, clock=time.monotonic, log=print)`
    - `.start() -> None`, `.stop(timeout: float = 3.0) -> None`
    - `.publish_live(payload: dict) -> None` (최신 하나만 보관)
    - `.publish_map(floorplan: dict) -> None`
    - `.stats() -> dict` — `{"sent": int, "failed": int, "last_error": Optional[str], "last_ok_at": Optional[float], "logged_in": bool, "backoff_s": float}`
  - `MAX_BACKOFF_S = 60.0`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_publish.py` 에 추가 (파일 맨 위 import 에 `import json, threading, time` 과 `from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer`, `from edge.publish import SupabasePublisher` 를 더한다):

```python
class _FakeSupabase:
    """Auth + PostgREST 흉내. 받은 요청을 기록하고, 정해진 횟수만큼 401 을 돌려준다."""

    def __init__(self):
        self.requests: list = []
        self.unauthorized_left = 0
        self.tokens_issued = 0
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # 조용히
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                with outer.lock:
                    outer.requests.append(
                        {
                            "path": self.path,
                            "headers": {k.lower(): v for k, v in self.headers.items()},
                            "body": json.loads(body) if body else None,
                        }
                    )
                    if self.path.startswith("/auth/v1/token"):
                        outer.tokens_issued += 1
                        payload = json.dumps(
                            {"access_token": f"tok{outer.tokens_issued}", "expires_in": 3600}
                        ).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        return
                    if outer.unauthorized_left > 0:
                        outer.unauthorized_left -= 1
                        self.send_response(401)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                self.send_response(201)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def wait_for(self, count, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if len(self.requests) >= count:
                    return True
            time.sleep(0.02)
        return False

    def upserts(self, table):
        with self.lock:
            return [r for r in self.requests if r["path"] == f"/rest/v1/{table}"]


def _wait(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class PublisherTest(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeSupabase()
        self.logs: list = []
        self.publisher = SupabasePublisher(
            self.fake.url, "anon-key", "box@x", "pw", "dwnc",
            timeout=2.0, sleep=lambda s: time.sleep(min(s, 0.01)), log=self.logs.append,
        )

    def tearDown(self):
        self.publisher.stop()
        self.fake.close()

    def _payload(self, n=1):
        return {"cafe_id": "dwnc", "status": "live", "total_tables": n, "occupied_tables": 0,
                "free_tables": n, "unknown_tables": 0, "seats": [], "tick_at": "t",
                "box_version": "v", "schema_version": 1}

    def test_logs_in_then_upserts_with_merge_duplicates(self):
        self.publisher.start()
        self.publisher.publish_live(self._payload())
        self.assertTrue(self.fake.wait_for(2))
        login, upsert = self.fake.requests[0], self.fake.requests[1]
        self.assertEqual(login["path"], "/auth/v1/token?grant_type=password")
        self.assertEqual(login["headers"]["apikey"], "anon-key")
        self.assertEqual(login["body"], {"email": "box@x", "password": "pw"})
        self.assertEqual(upsert["path"], "/rest/v1/cafe_live")
        self.assertEqual(upsert["headers"]["authorization"], "Bearer tok1")
        self.assertEqual(upsert["headers"]["apikey"], "anon-key")
        self.assertIn("resolution=merge-duplicates", upsert["headers"]["prefer"])
        self.assertEqual(upsert["body"]["cafe_id"], "dwnc")
        self.assertTrue(_wait(lambda: self.publisher.stats()["sent"] == 1))

    def test_only_the_latest_value_is_sent_when_ticks_pile_up(self):
        # 시작 전에 세 개를 넣으면 마지막 것 하나만 간다.
        self.publisher.publish_live(self._payload(1))
        self.publisher.publish_live(self._payload(2))
        self.publisher.publish_live(self._payload(3))
        self.publisher.start()
        self.assertTrue(_wait(lambda: self.publisher.stats()["sent"] == 1))
        time.sleep(0.2)
        upserts = self.fake.upserts("cafe_live")
        self.assertEqual(len(upserts), 1)
        self.assertEqual(upserts[0]["body"]["total_tables"], 3)

    def test_401_triggers_one_relogin_and_a_retry(self):
        self.fake.unauthorized_left = 1
        self.publisher.start()
        self.publisher.publish_live(self._payload())
        self.assertTrue(_wait(lambda: self.publisher.stats()["sent"] == 1))
        paths = [r["path"] for r in self.fake.requests]
        self.assertEqual(
            paths,
            ["/auth/v1/token?grant_type=password", "/rest/v1/cafe_live",
             "/auth/v1/token?grant_type=password", "/rest/v1/cafe_live"],
        )
        self.assertEqual(self.fake.upserts("cafe_live")[-1]["headers"]["authorization"], "Bearer tok2")

    def test_unreachable_server_counts_failures_and_never_raises(self):
        dead = SupabasePublisher(
            "http://127.0.0.1:9", "k", "e", "p", "dwnc",
            timeout=0.5, sleep=lambda s: time.sleep(min(s, 0.01)), log=self.logs.append,
        )
        dead.start()
        dead.publish_live(self._payload())
        self.assertTrue(_wait(lambda: dead.stats()["failed"] >= 1))
        stats = dead.stats()
        self.assertEqual(stats["sent"], 0)
        self.assertIsNotNone(stats["last_error"])
        self.assertGreater(stats["backoff_s"], 0.0)
        self.assertTrue(any("전송 실패" in line for line in self.logs))
        dead.stop()

    def test_map_goes_to_cafe_maps_with_only_cafe_id_and_floorplan(self):
        self.publisher.start()
        self.publisher.publish_map({"schema_version": 2, "seats": []})
        self.assertTrue(_wait(lambda: len(self.fake.upserts("cafe_maps")) == 1))
        body = self.fake.upserts("cafe_maps")[0]["body"]
        self.assertEqual(body, {"cafe_id": "dwnc", "floorplan": {"schema_version": 2, "seats": []}})

    def test_stop_returns_quickly(self):
        self.publisher.start()
        started = time.time()
        self.publisher.stop()
        self.assertLess(time.time() - started, 3.0)
```

- [ ] **Step 2: 실패 확인**

Run: `./venv/Scripts/python.exe -m unittest tests.test_publish.PublisherTest -v`
Expected: `ImportError: cannot import name 'SupabasePublisher'`

- [ ] **Step 3: 구현**

`edge/publish.py` 끝에 추가:

```python
MAX_BACKOFF_S = 60.0
_FIRST_BACKOFF_S = 5.0
_TOKEN_MARGIN_S = 60.0


class SupabasePublisher:
    """Sends the newest row in a background thread; never blocks the judge.

    Only the latest value is kept -- a tick that arrives while the previous
    one is still uploading replaces it.  Failures are counted and dropped;
    the next tick brings a fresh value anyway.
    """

    def __init__(
        self,
        url: str,
        anon_key: str,
        email: str,
        password: str,
        cafe_id: str,
        *,
        timeout: float = 5.0,
        session=None,
        sleep=time.sleep,
        clock=time.monotonic,
        log=print,
    ) -> None:
        import requests  # 엣지 requirements 에 있음; 파일 입력 경로에서는 import 조차 안 하도록 여기서

        self._url = url.rstrip("/")
        self._anon_key = anon_key
        self._email = email
        self._password = password
        self.cafe_id = cafe_id
        self._timeout = timeout
        self._session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._log = log

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._latest_live: Optional[Dict[str, Any]] = None
        self._pending_map: Optional[Dict[str, Any]] = None
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._backoff_s: float = 0.0
        self._sent = 0
        self._failed = 0
        self._last_error: Optional[str] = None
        self._last_ok_at: Optional[float] = None
        self._thread: Optional[threading.Thread] = None

    # ---- 판정 루프가 부르는 것 (즉시 돌아온다) ----

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="supabase-publish", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def publish_live(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._latest_live = payload
        self._wake.set()

    def publish_map(self, floorplan: Dict[str, Any]) -> None:
        with self._lock:
            self._pending_map = floorplan
        self._wake.set()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "sent": self._sent,
                "failed": self._failed,
                "last_error": self._last_error,
                "last_ok_at": self._last_ok_at,
                "logged_in": self._token is not None,
                "backoff_s": self._backoff_s,
            }

    # ---- 스레드 ----

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                live, self._latest_live = self._latest_live, None
                floorplan, self._pending_map = self._pending_map, None
            if live is None and floorplan is None:
                continue
            try:
                if live is not None:
                    self._send("cafe_live", live)
                if floorplan is not None:
                    self._send("cafe_maps", {"cafe_id": self.cafe_id, "floorplan": floorplan})
            except Exception as error:  # noqa: BLE001 -- 판정을 지키는 게 우선
                self._note_failure(f"{type(error).__name__}: {error}")
                # 실패한 값은 버린다. 다음 틱이 더 새롭다. 백오프 동안은 잔다.
                self._sleep(self._backoff_s)

    def _send(self, table: str, row: Dict[str, Any]) -> None:
        self._ensure_token()
        response = self._upsert(table, row)
        if response.status_code == 401:
            self._token = None
            self._ensure_token()
            response = self._upsert(table, row)
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"{table} HTTP {response.status_code}: {response.text[:200]}")
        self._note_success()

    def _upsert(self, table: str, row: Dict[str, Any]):
        return self._session.post(
            f"{self._url}/rest/v1/{table}",
            headers={
                "apikey": self._anon_key,
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            data=json.dumps(row, ensure_ascii=False).encode("utf-8"),
            timeout=self._timeout,
        )

    def _ensure_token(self) -> None:
        if self._token is not None and self._clock() < self._token_expires_at:
            return
        response = self._session.post(
            f"{self._url}/auth/v1/token?grant_type=password",
            headers={"apikey": self._anon_key, "Content-Type": "application/json"},
            data=json.dumps({"email": self._email, "password": self._password}).encode("utf-8"),
            timeout=self._timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"로그인 실패 HTTP {response.status_code}: {response.text[:200]}")
        body = response.json()
        self._token = str(body["access_token"])
        expires_in = float(body.get("expires_in", 3600))
        self._token_expires_at = self._clock() + max(60.0, expires_in - _TOKEN_MARGIN_S)

    def _note_success(self) -> None:
        with self._lock:
            recovered = self._failed > 0 and self._backoff_s > 0.0
            self._sent += 1
            self._last_ok_at = time.time()
            self._backoff_s = 0.0
        if recovered:
            self._log("Supabase 전송 복구됨")

    def _note_failure(self, message: str) -> None:
        with self._lock:
            self._failed += 1
            self._last_error = message
            self._backoff_s = _FIRST_BACKOFF_S if self._backoff_s == 0.0 else min(MAX_BACKOFF_S, self._backoff_s * 2)
            count = self._failed
            backoff = self._backoff_s
        if count == 1 or count % 10 == 0:
            self._log(f"⚠️  Supabase 전송 실패 {count}회 — {message} (다음 시도까지 {backoff:.0f}초)")
```

- [ ] **Step 4: 통과 확인**

Run: `./venv/Scripts/python.exe -m unittest tests.test_publish -v`
Expected: 13 tests OK (Task 1 의 7 + 6). `test_401_...` 이 순서 때문에 흔들리면 `_FakeSupabase` 의 lock 안에서 append 가 되는지 확인한다.

- [ ] **Step 5: 커밋**

```bash
git add edge/publish.py tests/test_publish.py
git commit -m "feat(전송): SupabasePublisher — 백그라운드 스레드, 최신 값 하나, 401 재로그인, 실패는 버리고 백오프"
```

---

### Task 3: 환경변수 → publisher, 박스 버전, 평면도 파일 감시

**Files:**
- Modify: `edge/publish.py`
- Modify: `deploy/seatnow.env.example`
- Test: `tests/test_publish.py`

**Interfaces:**
- Produces:
  - `ENV_KEYS = ("SEATNOW_CAFE_ID", "SEATNOW_SUPABASE_URL", "SEATNOW_SUPABASE_ANON_KEY", "SEATNOW_SUPABASE_EMAIL", "SEATNOW_SUPABASE_PASSWORD")`
  - `publisher_from_env(env: Mapping[str, str]) -> Tuple[Optional[SupabasePublisher], str]` — `(None, "꺼짐 (SEATNOW_SUPABASE_URL 없음)")` 또는 `(publisher, "켜짐 (dwnc)")`
  - `box_version(project_dir: Path = PROJECT_DIR) -> str` — git 짧은 해시 또는 `"unknown"`
  - `floorplan_path_for(layout_path: Path) -> Path` — `layouts/x.json` → `layouts/x.floorplan.json`
  - `class FloorplanWatcher(path: Path)` — `.changed() -> Optional[dict]` (수정 시각이 바뀌었을 때만 파일 내용, 아니면 None; 파일이 없거나 JSON 이 깨지면 None)

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_publish.py` 에 추가 (import 에 `import os, tempfile`, `from pathlib import Path`, `from edge.publish import ENV_KEYS, FloorplanWatcher, box_version, floorplan_path_for, publisher_from_env`):

```python
class EnvTest(unittest.TestCase):
    FULL = {
        "SEATNOW_CAFE_ID": "dwnc",
        "SEATNOW_SUPABASE_URL": "http://127.0.0.1:9/",
        "SEATNOW_SUPABASE_ANON_KEY": "k",
        "SEATNOW_SUPABASE_EMAIL": "e",
        "SEATNOW_SUPABASE_PASSWORD": "p",
    }

    def test_all_five_keys_make_a_publisher(self):
        publisher, message = publisher_from_env(self.FULL)
        self.assertIsNotNone(publisher)
        self.assertEqual(publisher.cafe_id, "dwnc")
        self.assertEqual(message, "켜짐 (dwnc)")

    def test_any_missing_key_turns_it_off_and_names_the_key(self):
        for key in ENV_KEYS:
            env = dict(self.FULL)
            del env[key]
            publisher, message = publisher_from_env(env)
            self.assertIsNone(publisher, key)
            self.assertEqual(message, f"꺼짐 ({key} 없음)")

    def test_blank_value_counts_as_missing(self):
        env = dict(self.FULL, SEATNOW_SUPABASE_PASSWORD="  ")
        publisher, message = publisher_from_env(env)
        self.assertIsNone(publisher)
        self.assertIn("SEATNOW_SUPABASE_PASSWORD", message)


class BoxVersionTest(unittest.TestCase):
    def test_repo_gives_a_short_hash(self):
        version = box_version()
        self.assertRegex(version, r"^[0-9a-f]{7,12}$")

    def test_outside_a_repo_is_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(box_version(Path(folder)), "unknown")


class FloorplanWatcherTest(unittest.TestCase):
    def test_path_is_next_to_the_layout(self):
        self.assertEqual(floorplan_path_for(Path("layouts/cafe.json")), Path("layouts/cafe.floorplan.json"))

    def test_reports_content_once_per_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.floorplan.json"
            watcher = FloorplanWatcher(path)
            self.assertIsNone(watcher.changed())  # 파일 없음
            path.write_text('{"schema_version": 2}', encoding="utf-8")
            self.assertEqual(watcher.changed(), {"schema_version": 2})
            self.assertIsNone(watcher.changed())  # 안 바뀜
            os.utime(path, (time.time() + 5, time.time() + 5))
            path.write_text('{"schema_version": 3}', encoding="utf-8")
            os.utime(path, (time.time() + 10, time.time() + 10))
            self.assertEqual(watcher.changed(), {"schema_version": 3})

    def test_broken_json_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.floorplan.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertIsNone(FloorplanWatcher(path).changed())
```

- [ ] **Step 2: 실패 확인**

Run: `./venv/Scripts/python.exe -m unittest tests.test_publish -v`
Expected: `ImportError: cannot import name 'ENV_KEYS'`

- [ ] **Step 3: 구현**

`edge/publish.py` 끝에 추가:

```python
ENV_KEYS = (
    "SEATNOW_CAFE_ID",
    "SEATNOW_SUPABASE_URL",
    "SEATNOW_SUPABASE_ANON_KEY",
    "SEATNOW_SUPABASE_EMAIL",
    "SEATNOW_SUPABASE_PASSWORD",
)


def publisher_from_env(env: Mapping[str, str]) -> Tuple[Optional["SupabasePublisher"], str]:
    """(publisher, 사람이 읽을 상태 한 줄). 다섯 값이 다 있어야 켜진다."""
    values = {}
    for key in ENV_KEYS:
        value = (env.get(key) or "").strip()
        if not value:
            return None, f"꺼짐 ({key} 없음)"
        values[key] = value
    publisher = SupabasePublisher(
        values["SEATNOW_SUPABASE_URL"],
        values["SEATNOW_SUPABASE_ANON_KEY"],
        values["SEATNOW_SUPABASE_EMAIL"],
        values["SEATNOW_SUPABASE_PASSWORD"],
        values["SEATNOW_CAFE_ID"],
    )
    return publisher, f"켜짐 ({values['SEATNOW_CAFE_ID']})"


def box_version(project_dir: Path = PROJECT_DIR) -> str:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    text = output.stdout.strip()
    return text if output.returncode == 0 and text else "unknown"


def floorplan_path_for(layout_path: Path) -> Path:
    layout_path = Path(layout_path)
    return layout_path.with_name(layout_path.stem + ".floorplan.json")


class FloorplanWatcher:
    """Hands back the floor plan file once each time it changes on disk."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._seen_mtime: Optional[float] = None

    def changed(self) -> Optional[Dict[str, Any]]:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return None
        if self._seen_mtime is not None and mtime == self._seen_mtime:
            return None
        try:
            content = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        self._seen_mtime = mtime
        return content
```

`deploy/seatnow.env.example` 끝에 추가:

```
# Supabase 전송 (다섯 줄이 다 있어야 켜진다. 하나라도 비면 전송은 꺼지고 판정만 돈다).
#   CAFE_ID   : Supabase `cafes.id` 와 같은 짧은 이름
#   URL/ANON  : Supabase 대시보드 → Settings → API 의 Project URL, anon public key
#   EMAIL/PW  : 이 박스용으로 만든 Supabase 인증 사용자 (docs/앱연동.md "박스 계정")
SEATNOW_CAFE_ID=
SEATNOW_SUPABASE_URL=
SEATNOW_SUPABASE_ANON_KEY=
SEATNOW_SUPABASE_EMAIL=
SEATNOW_SUPABASE_PASSWORD=
```

- [ ] **Step 4: 통과 확인**

Run: `./venv/Scripts/python.exe -m unittest tests.test_publish -v`
Expected: 21 tests OK.

- [ ] **Step 5: 커밋**

```bash
git add edge/publish.py tests/test_publish.py deploy/seatnow.env.example
git commit -m "feat(전송): 환경변수 다섯 개로 켜고 끄기, 박스 git 버전, 평면도 파일 감시"
```

---

### Task 4: 판정 루프에 연결 (`engine/seatnow.py` `process_live`)

**Files:**
- Modify: `engine/seatnow.py` — `process_live` 함수 (시작 출력 근처, 메인 while 루프의 gap 쓰기 직후, `record = runner.judge(...)` 직후, `summary` dict, 마지막 print)
- Test: `tests/test_seatnow_publish_hook.py` (새, 작음)

**Interfaces:**
- Consumes: Task 1~3 전부.
- Produces: 판정 기록의 각 틱에 `record["publish"] = {"sent", "failed", "last_error"}` (전송이 켜졌을 때만), `last_run_summary.json` 에 `"publish": {...상태, "enabled": bool}`.

- [ ] **Step 1: 실패하는 테스트 작성**

`process_live` 는 진짜 스트림이 필요해 단위 테스트가 안 된다. 대신 연결 코드를 작은 함수 둘로 빼서 그 함수를 시험한다.

`tests/test_seatnow_publish_hook.py`:

```python
"""process_live 가 publisher 에 넘기는 조각들."""

from __future__ import annotations

import unittest

from engine.seatnow import _publish_tick, _publish_gap


class _Spy:
    def __init__(self):
        self.live: list = []

    def publish_live(self, payload):
        self.live.append(payload)

    def stats(self):
        return {"sent": len(self.live), "failed": 0, "last_error": None}


class HookTest(unittest.TestCase):
    def test_tick_is_published_and_counters_land_in_the_record(self):
        spy = _Spy()
        record = {"wall_clock": "2026-09-08T20:32:45+0900", "tables": [
            {"layout_name": "T1", "layout_kind": "table", "state": "empty", "raw_state": "empty", "reason": ""},
        ]}
        _publish_tick(spy, record, "dwnc", "v1")
        self.assertEqual(len(spy.live), 1)
        self.assertEqual(spy.live[0]["free_tables"], 1)
        self.assertEqual(record["publish"], {"sent": 1, "failed": 0, "last_error": None})

    def test_none_publisher_is_a_no_op(self):
        record = {"tables": []}
        _publish_tick(None, record, "dwnc", "v1")
        self.assertNotIn("publish", record)

    def test_gap_publishes_all_unknown(self):
        spy = _Spy()
        index = [{"seat_id": "T1", "kind": "table", "zone": None}]
        _publish_gap(spy, index, "dwnc", "v1", "2026-09-08T21:00:00+0900")
        self.assertEqual(spy.live[0]["status"], "gap")
        self.assertEqual(spy.live[0]["seats"][0]["state"], "unknown")

    def test_publisher_errors_never_escape(self):
        class Boom:
            def publish_live(self, payload):
                raise RuntimeError("x")

            def stats(self):
                raise RuntimeError("y")

        record = {"tables": []}
        _publish_tick(Boom(), record, "dwnc", "v1")  # 예외 없이 끝나야 한다
        _publish_gap(Boom(), [], "dwnc", "v1", "t")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 실패 확인**

Run: `./venv/Scripts/python.exe -m unittest tests.test_seatnow_publish_hook -v`
Expected: `ImportError: cannot import name '_publish_tick'`

- [ ] **Step 3: 구현**

`engine/seatnow.py` import 블록에 추가 (`from engine.seatnow_live import (...)` 아래):

```python
from edge.publish import (
    FloorplanWatcher,
    _publish_stats_for_record,
    box_version,
    floorplan_path_for,
    gap_payload,
    live_payload,
    publisher_from_env,
    seat_index_from_layout,
)
```

`edge/publish.py` 에 작은 도우미를 하나 추가 (Task 3 코드 아래):

```python
def _publish_stats_for_record(publisher) -> Dict[str, Any]:
    """The three numbers a tick record carries so live_report can show them."""
    stats = publisher.stats()
    return {"sent": stats["sent"], "failed": stats["failed"], "last_error": stats["last_error"]}
```

`engine/seatnow.py` 에서 `process_live` **바로 위**에 두 함수 추가:

```python
def _publish_tick(publisher, record: dict, cafe_id: str, version: str) -> None:
    """Hand one judged tick to the publisher.  Nothing here may raise."""
    if publisher is None:
        return
    try:
        publisher.publish_live(live_payload(record, cafe_id, version))
        record["publish"] = _publish_stats_for_record(publisher)
    except Exception as error:  # noqa: BLE001 -- 전송 문제는 판정을 멈추지 않는다
        print(f"⚠️  Supabase 전송 준비 실패: {type(error).__name__}: {error}", flush=True)


def _publish_gap(publisher, seat_index: list, cafe_id: str, version: str, wall_clock: str) -> None:
    if publisher is None:
        return
    try:
        publisher.publish_live(gap_payload(seat_index, cafe_id, version, wall_clock))
    except Exception as error:  # noqa: BLE001
        print(f"⚠️  Supabase 전송 준비 실패: {type(error).__name__}: {error}", flush=True)
```

`process_live` 안, `print(hwaccel.describe(), flush=True)` 줄 **뒤**에 추가:

```python
    import os

    publisher, publish_state = publisher_from_env(os.environ)
    version = box_version()
    cafe_id = publisher.cafe_id if publisher is not None else ""
    seat_index = seat_index_from_layout(analyzer.layout) if analyzer.layout is not None else []
    floorplan_watcher: Optional[FloorplanWatcher] = None
    print(f"Supabase 전송: {publish_state}", flush=True)
    if publisher is not None:
        publisher.start()
        if args.layout is not None:
            floorplan_watcher = FloorplanWatcher(floorplan_path_for(args.layout))
            if not floorplan_watcher.path.exists():
                print(
                    f"지도 없음: {floorplan_watcher.path} 가 없어 cafe_maps 는 올리지 않습니다 "
                    f"(python -m install.floorplan --layout {args.layout} 로 초안을 만든다)",
                    flush=True,
                )
```

(`args.layout` 이 `process_live` 에서 어떤 이름인지 확인한다. `build_parser` 의 `--layout` 인자 이름이 `args.layout` 이 아니면 그 이름을 쓴다.)

메인 `while True:` 루프 맨 앞, `if rotator is not None:` **앞**에 추가:

```python
                if floorplan_watcher is not None and publisher is not None:
                    floorplan = floorplan_watcher.changed()
                    if floorplan is not None:
                        publisher.publish_map(floorplan)
                        print(f"지도 올림: {floorplan_watcher.path}", flush=True)
```

gap 을 쓰는 곳, `log_file.write(json.dumps(gap, ensure_ascii=False) + "\n")` 와 `log_file.flush()` **뒤**에 추가:

```python
                    _publish_gap(publisher, seat_index, cafe_id, version, gap["wall_clock"])
```

`record = runner.judge(burst, center_index, center_time, extra=extra)` **뒤**에 추가:

```python
                _publish_tick(publisher, record, cafe_id, version)
```

주의: `runner.judge` 가 이미 JSONL 을 썼으므로 `record["publish"]` 는 파일에는 이번 틱이 아니라 **다음 틱부터** 실린다 — 카운트는 누적값이라 상관없다. 파일에도 이번 틱에 싣고 싶으면 `extra["publish"]` 에 직전 `stats` 를 넣는다:

`extra = {...}` dict 를 만든 직후 (`record = runner.judge(...)` 전):

```python
                if publisher is not None:
                    try:
                        extra["publish"] = _publish_stats_for_record(publisher)
                    except Exception:  # noqa: BLE001
                        pass
```

(둘 다 둔다. `extra["publish"]` 는 파일용, `_publish_tick` 안의 것은 메모리의 record 용.)

`finally: reader.close()` **뒤**, `stats = reader.stats()` **앞**에:

```python
    if publisher is not None:
        publisher.stop()
```

`summary = {` dict 의 `"interrupted": interrupted,` 앞에 추가:

```python
        "publish": (
            {"enabled": True, **publisher.stats()} if publisher is not None else {"enabled": False}
        ),
```

마지막 print 들 사이, `print(f"JSONL log: {log_path}")` **앞**에:

```python
    if publisher is not None:
        ps = summary["publish"]
        print(
            f"Supabase 전송: 성공 {ps['sent']}회 · 실패 {ps['failed']}회"
            + (f" · 마지막 오류: {ps['last_error']}" if ps["last_error"] else "")
        )
```

- [ ] **Step 4: 통과 확인**

Run: `./venv/Scripts/python.exe -m unittest tests.test_seatnow_publish_hook tests.test_publish tests.test_live_logdir -v`
Expected: 전부 OK. 그리고 회귀: `./venv/Scripts/python.exe -m unittest discover tests` 전부 OK (환경변수가 없으면 아무것도 안 바뀌어야 한다).

- [ ] **Step 5: 노트북에서 스모크 (환경변수 없이)**

Run: `./venv/Scripts/python.exe -m engine.seatnow sample_raw/cafe_sample_angle1.mov --layout layouts/cafe_angle1.json --max-samples 1 --no-video` (파일 입력 경로는 `process_live` 를 안 타므로 출력이 지금과 같아야 한다.)
Expected: 지금과 같은 출력, 오류 없음.

- [ ] **Step 6: 커밋**

```bash
git add engine/seatnow.py edge/publish.py tests/test_seatnow_publish_hook.py
git commit -m "feat(전송): 실시간 판정 루프가 판정 줄·gap 줄 직후 Supabase 로 보내고, 평면도 파일이 바뀌면 지도를 올린다"
```

---

### Task 5: `edge.live_report` 에 전송 카운트

**Files:**
- Modify: `edge/live_report.py` — `summarize_records`, `render`
- Test: `tests/test_live_report.py`

**Interfaces:**
- Consumes: 틱 기록의 `record["publish"] = {"sent", "failed", "last_error"}` (Task 4).
- Produces: `summary["publish"] = {"sent": int, "failed": int, "last_error": Optional[str]} | None`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_live_report.py` 에 추가 (기존 `records` 헬퍼를 쓴다 — 파일 위쪽에 틱 기록을 만드는 헬퍼가 있다; 이름이 다르면 그 이름으로):

```python
    def test_publish_counts_come_from_the_last_record(self) -> None:
        records = _ticks(3)  # 기존 헬퍼. 없으면 아래처럼 만든다.
        records[-1]["publish"] = {"sent": 40, "failed": 2, "last_error": "HTTP 503"}
        summary = summarize_records(records, interval_seconds=15.0)
        self.assertEqual(summary["publish"], {"sent": 40, "failed": 2, "last_error": "HTTP 503"})
        text = render(summary, None, "x")
        self.assertIn("전송 성공 40회 · 실패 2회", text)

    def test_no_publish_field_means_publishing_was_off(self) -> None:
        summary = summarize_records(_ticks(2), interval_seconds=15.0)
        self.assertIsNone(summary["publish"])
        self.assertIn("Supabase 전송: 꺼짐", render(summary, None, "x"))
```

기존 헬퍼가 없으면 테스트 파일 위에 추가:

```python
def _ticks(n):
    return [
        {
            "tick": {"duration_s": 3.0, "late_s": 0.0, "burst_age_s": 0.5, "skipped_total": 0},
            "inference_ms": 3000.0,
            "live": {"reconnects": 0, "hwaccel_suspect": False, "decode_fps": 20.0},
            "process": {"elapsed_s": 15.0 * i, "rss_mb": 900.0, "ffmpeg_rss_mb": 100.0},
        }
        for i in range(n)
    ]
```

- [ ] **Step 2: 실패 확인**

Run: `./venv/Scripts/python.exe -m unittest tests.test_live_report -v`
Expected: `KeyError: 'publish'`

- [ ] **Step 3: 구현**

`edge/live_report.py` `summarize_records` 의 return dict 에 추가:

```python
        "publish": (
            {
                "sent": int(last["publish"].get("sent", 0)),
                "failed": int(last["publish"].get("failed", 0)),
                "last_error": last["publish"].get("last_error"),
            }
            if last and isinstance(last.get("publish"), dict)
            else None
        ),
```

`render` 에서 `f = summary["ffmpeg_rss_mb"]` 줄 뒤(메모리 줄 다음)에 추가:

```python
    p = summary.get("publish")
    if p is None:
        lines.append("- Supabase 전송: 꺼짐")
    else:
        lines.append(
            f"- Supabase 전송 성공 {p['sent']}회 · 실패 {p['failed']}회"
            + (f" · 마지막 오류: {p['last_error']}" if p["last_error"] else "")
        )
```

- [ ] **Step 4: 통과 확인**

Run: `./venv/Scripts/python.exe -m unittest tests.test_live_report -v`
Expected: OK.

- [ ] **Step 5: 커밋**

```bash
git add edge/live_report.py tests/test_live_report.py
git commit -m "feat(보고서): 하루치 보고에 Supabase 전송 성공·실패 횟수"
```

---

### Task 6: 바닥 네 점 없이 지도 초안 + `python -m install.floorplan`

**Files:**
- Modify: `install/floorplan.py` — `build_draft`, 새 `main()`
- Test: `tests/test_floorplan.py` — `test_no_floor_reference_is_refused` 교체

**Interfaces:**
- Produces: `build_draft(layout)` 가 `floor_reference is None` 이어도 `FloorPlan` 을 돌려준다 (전부 `needs_review=True`). `main(argv) -> int`: `--layout PATH [--floorplan PATH] [--force]`; 파일이 있으면 건드리지 않고 0 을 돌려준다.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_floorplan.py` 에서 `test_no_floor_reference_is_refused` 를 지우고 대신:

```python
    def test_no_floor_reference_places_seats_where_the_camera_sees_them(self):
        plan = build_draft(layout(with_reference=False))
        self.assertEqual(len(plan.seats), len(layout(with_reference=False).judgement_units()))
        self.assertTrue(all(seat.needs_review for seat in plan.seats))
        self.assertTrue(all(chair.needs_review for chair in plan.chairs))
        w, h = plan.extent
        for seat in plan.seats:
            self.assertGreaterEqual(seat.x, 0.0)
            self.assertGreaterEqual(seat.y, 0.0)
            self.assertLessEqual(seat.x + seat.w, w + 1e-6)
            self.assertLessEqual(seat.y + seat.h, h + 1e-6)

    def test_no_floor_reference_keeps_left_right_order_of_the_image(self):
        plan = build_draft(layout(with_reference=False))
        by_id = {seat.seat_id: seat for seat in plan.seats}
        # T1 상자(800~1000)가 BAR7 칸(1100~)보다 화면에서 왼쪽이다.
        bar = min(s for s in plan.seats if s.seat_id != "T1", key=lambda s: s.x)
        self.assertLess(by_id["T1"].x, bar.x)

    def test_with_reference_still_projects(self):
        with_ref = build_draft(layout(with_reference=True))
        self.assertFalse(all(seat.needs_review for seat in with_ref.seats))
```

그리고 CLI 테스트 (같은 파일 끝에 새 클래스):

```python
class FloorplanCliTest(unittest.TestCase):
    def _layout_file(self, folder):
        from engine.seatnow_layout import save_layout

        path = Path(folder) / "cafe.json"
        save_layout(layout(with_reference=False), path)
        return path

    def test_writes_a_draft_next_to_the_layout(self):
        from install.floorplan import main

        with tempfile.TemporaryDirectory() as folder:
            layout_path = self._layout_file(folder)
            self.assertEqual(main(["--layout", str(layout_path)]), 0)
            out = Path(folder) / "cafe.floorplan.json"
            self.assertTrue(out.exists())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], FLOORPLAN_SCHEMA_VERSION)
            self.assertTrue(all(s["needs_review"] for s in data["seats"]))

    def test_existing_file_is_left_alone_unless_forced(self):
        from install.floorplan import main

        with tempfile.TemporaryDirectory() as folder:
            layout_path = self._layout_file(folder)
            out = Path(folder) / "cafe.floorplan.json"
            out.write_text('{"hand": "edited"}', encoding="utf-8")
            self.assertEqual(main(["--layout", str(layout_path)]), 0)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")), {"hand": "edited"})
            self.assertEqual(main(["--layout", str(layout_path), "--force"]), 0)
            self.assertIn("seats", json.loads(out.read_text(encoding="utf-8")))
```

(`save_layout` 의 실제 이름·시그니처는 `engine/seatnow_layout.py` 에서 확인한다. `install/calibrate.py` 가 `from engine.seatnow_layout import save_layout` 으로 쓰고 `save_layout(layout, output_path)` 로 부른다.)

- [ ] **Step 2: 실패 확인**

Run: `./venv/Scripts/python.exe -m unittest tests.test_floorplan -v`
Expected: 새 테스트 3개는 `FloorProjectionError`, CLI 2개는 `ImportError: cannot import name 'main'`.

- [ ] **Step 3: 구현**

`install/floorplan.py` `build_draft` 를 아래로 바꾼다 (기존 함수의 앞부분 — 거부하는 `if layout.floor_reference is None:` 블록과 `transform = ...` 줄 — 을 이렇게 교체하고, 나머지 `place`/seats/chairs/arrange 부분은 그대로 둔다):

```python
def build_draft(layout: SeatLayout) -> FloorPlan:
    """Project every seat and chair onto the floor and fit them to a canvas.

    With floor reference points the positions are a real top-down projection.
    Without them (the usual install: nobody clicked the floor) the camera
    image itself is the map -- every box lands where the camera sees it and
    is flagged ``needs_review`` so the editor shows it as "move me".  A map
    that is roughly right today beats a precise one that needs an extra
    half hour on the ceiling ladder.
    """
    frame_size = (
        int(layout.source.get("width", 1920)),
        int(layout.source.get("height", 1080)),
    )
    units = layout.judgement_units()
    seat_anchors = [(unit, floor_anchor(unit.box)) for unit in units]
    chair_owners = _owner_of_each_chair(layout)

    if layout.floor_reference is not None:
        transform = build_transform(layout.floor_reference.image_points, frame_size)
        projected: List[Optional[Point]] = [
            transform.project(anchor) for _, anchor in seat_anchors
        ] + [transform.project(anchor) for _, anchor in chair_owners]
        review_all = False
    else:
        # 화면 좌표 그대로. 상자 중심이 그 자리다.
        projected = [
            ((unit.box[0] + unit.box[2]) / 2.0, (unit.box[1] + unit.box[3]) / 2.0)
            for unit, _ in seat_anchors
        ] + [anchor for _, anchor in chair_owners]
        review_all = True

    placed = [point for point in projected if point is not None]
    if not placed:
        raise FloorProjectionError(
            "좌석이 하나도 바닥 평면 위로 오지 않았습니다 — 바닥 네 점을 "
            "다시 찍어 주세요"
        )
```

그리고 같은 함수 안 `place()` 의 return 을 `review_all` 을 반영하게 바꾼다:

```python
        return (
            (point[0] - min_x) * scale + margin,
            (point[1] - min_y) * scale + margin,
            review_all,
        )
```

파일 끝에 CLI 추가:

```python
def main(argv: Optional[List[str]] = None) -> int:
    """평면도 초안을 만든다. 파일이 이미 있으면 (사람이 고쳤을 수 있으니) 건드리지 않는다."""
    import argparse

    from engine.seatnow_layout import load_layout

    parser = argparse.ArgumentParser(description="레이아웃에서 손님용 지도 초안을 만든다")
    parser.add_argument("--layout", type=Path, required=True, help="calibrate.py 가 만든 레이아웃")
    parser.add_argument("--floorplan", type=Path, help="출력 경로 (기본: <layout>.floorplan.json)")
    parser.add_argument("--force", action="store_true", help="이미 있어도 덮어쓴다")
    args = parser.parse_args(argv)

    out = args.floorplan or args.layout.with_name(args.layout.stem + ".floorplan.json")
    if out.exists() and not args.force:
        print(f"이미 있음: {out} — 그대로 둡니다 (--force 로 덮어쓰기)")
        return 0
    layout = load_layout(args.layout)
    plan = build_draft(layout)
    save_floorplan(plan, out)
    review = sum(1 for seat in plan.seats if seat.needs_review)
    print(
        f"저장됨: {out} (자리 {len(plan.seats)}개, 의자 {len(plan.chairs)}개"
        + (f", 위치 확인 필요 {review}개 — 편집기로 옮긴다" if review else "")
        + ")"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
```

`load_layout` 의 이름은 `engine/seatnow.py` 가 `from engine.seatnow_layout import LayoutError, load_layout` 으로 쓰는 것과 같다.

- [ ] **Step 4: 통과 확인**

Run: `./venv/Scripts/python.exe -m unittest tests.test_floorplan tests.test_floorplan_editor -v`
Expected: 전부 OK. 편집기 테스트 중 "바닥 점 없으면 거부"를 기대하는 것이 있으면 새 동작(초안이 나온다)으로 고친다.

- [ ] **Step 5: 손으로 확인**

Run: `./venv/Scripts/python.exe -m install.floorplan --layout layouts/cafe_angle1.json --floorplan C:\Users\jin06\AppData\Local\Temp\claude\test.floorplan.json --force`
Expected: `저장됨: ... (자리 12개, ...)`. (실제 `layouts/cafe_angle1.floorplan.json` 은 건드리지 않는다.)

- [ ] **Step 6: 커밋**

```bash
git add install/floorplan.py tests/test_floorplan.py tests/test_floorplan_editor.py
git commit -m "feat(지도): 바닥 네 점 없이도 카메라 화면 위치로 지도 초안 + python -m install.floorplan"
```

---

### Task 7: Supabase SQL (표·트리거·권한·실시간)

**Files:**
- Create: `deploy/supabase/schema.sql`

**Interfaces:**
- Produces: 표 `cafes`, `boxes`, `cafe_live`, `cafe_maps`; 트리거로 `updated_at`·`version`; RLS; realtime publication.

- [ ] **Step 1: 파일 작성**

```sql
-- SeatNow: 박스가 쓰고 앱이 읽는 표. Supabase 대시보드 → SQL Editor 에 통째로 붙여넣고 Run.
-- 여러 번 실행해도 안전하다 (if not exists / drop ... if exists).

create table if not exists public.cafes (
  id          text primary key,             -- 짧은 이름. 박스의 SEATNOW_CAFE_ID 와 같다
  name        text not null,
  address     text,
  created_at  timestamptz not null default now()
);

create table if not exists public.boxes (
  auth_user_id uuid primary key references auth.users (id) on delete cascade,
  cafe_id      text not null references public.cafes (id),
  label        text
);

create table if not exists public.cafe_live (
  cafe_id          text primary key references public.cafes (id),
  status           text not null check (status in ('live', 'gap')),
  total_tables     integer not null,
  occupied_tables  integer not null,
  free_tables      integer not null,
  unknown_tables   integer not null,
  seats            jsonb not null,
  tick_at          timestamptz,
  updated_at       timestamptz not null default now(),  -- 서버가 찍는다. 앱은 이걸 본다
  box_version      text,
  schema_version   integer not null default 1
);

create table if not exists public.cafe_maps (
  cafe_id     text primary key references public.cafes (id),
  floorplan   jsonb not null,
  version     integer not null default 1,
  updated_at  timestamptz not null default now()
);

-- updated_at 은 박스 시계가 아니라 서버 시계다 (45초 규칙의 근거).
create or replace function public.seatnow_touch_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end $$;

drop trigger if exists cafe_live_touch on public.cafe_live;
create trigger cafe_live_touch
  before update on public.cafe_live
  for each row execute function public.seatnow_touch_updated_at();

create or replace function public.seatnow_bump_map_version()
returns trigger language plpgsql as $$
begin
  new.version = old.version + 1;
  new.updated_at = now();
  return new;
end $$;

drop trigger if exists cafe_maps_bump on public.cafe_maps;
create trigger cafe_maps_bump
  before update on public.cafe_maps
  for each row execute function public.seatnow_bump_map_version();

-- 권한: 앱(anon)은 읽기만. 박스(authenticated)는 boxes 에 적힌 자기 카페 줄만 쓴다.
alter table public.cafes     enable row level security;
alter table public.boxes     enable row level security;
alter table public.cafe_live enable row level security;
alter table public.cafe_maps enable row level security;

drop policy if exists "cafes: anyone reads" on public.cafes;
create policy "cafes: anyone reads" on public.cafes
  for select to anon, authenticated using (true);

drop policy if exists "boxes: box reads itself" on public.boxes;
create policy "boxes: box reads itself" on public.boxes
  for select to authenticated using (auth_user_id = auth.uid());

drop policy if exists "cafe_live: anyone reads" on public.cafe_live;
create policy "cafe_live: anyone reads" on public.cafe_live
  for select to anon, authenticated using (true);

drop policy if exists "cafe_live: box inserts own cafe" on public.cafe_live;
create policy "cafe_live: box inserts own cafe" on public.cafe_live
  for insert to authenticated
  with check (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()));

drop policy if exists "cafe_live: box updates own cafe" on public.cafe_live;
create policy "cafe_live: box updates own cafe" on public.cafe_live
  for update to authenticated
  using (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()))
  with check (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()));

drop policy if exists "cafe_maps: anyone reads" on public.cafe_maps;
create policy "cafe_maps: anyone reads" on public.cafe_maps
  for select to anon, authenticated using (true);

drop policy if exists "cafe_maps: box inserts own cafe" on public.cafe_maps;
create policy "cafe_maps: box inserts own cafe" on public.cafe_maps
  for insert to authenticated
  with check (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()));

drop policy if exists "cafe_maps: box updates own cafe" on public.cafe_maps;
create policy "cafe_maps: box updates own cafe" on public.cafe_maps
  for update to authenticated
  using (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()))
  with check (cafe_id in (select cafe_id from public.boxes where auth_user_id = auth.uid()));

-- 앱이 구독할 수 있게 실시간 발행에 넣는다 (이미 들어 있으면 오류가 나므로 각각 시도).
do $$
begin
  begin
    alter publication supabase_realtime add table public.cafe_live;
  exception when duplicate_object then null;
  end;
  begin
    alter publication supabase_realtime add table public.cafe_maps;
  exception when duplicate_object then null;
  end;
end $$;

-- 첫 카페와 박스 (값을 바꿔서 실행한다):
-- insert into public.cafes (id, name) values ('dwnc', '카페 이름');
-- insert into public.boxes (auth_user_id, cafe_id, label)
--   values ('<Authentication 에서 만든 박스 사용자의 UUID>', 'dwnc', 'uhho');
```

- [ ] **Step 2: 문법 확인**

로컬에 PostgreSQL 이 없으므로 눈으로 본다: 모든 문장이 `;` 로 끝나는지, 정책 이름이 따옴표 안에 있는지, `do $$ ... $$` 가 닫혔는지. 실제 실행은 Task 10 의 사람 단계에서.

- [ ] **Step 3: 커밋**

```bash
git add deploy/supabase/schema.sql
git commit -m "feat(supabase): 표·트리거·행 단위 권한·실시간 발행 SQL"
```

---

### Task 8: 앱 개발자 문서 `docs/앱연동.md`

**Files:**
- Create: `docs/앱연동.md`

- [ ] **Step 1: 작성**

```markdown
# 앱 연동 — Supabase 에서 좌석 상태 읽기

> 대상: 손님 앱 개발자. 박스가 15초마다 쓰는 값을 어떻게 읽고, 어떻게 보여줘야 손님이
> 틀린 정보를 안 받는지 적었다. 표를 만드는 SQL 은 `deploy/supabase/schema.sql`.

## 규칙 세 개 — 이것만은 꼭

1. **45초 규칙.** `cafe_live.updated_at` 이 지금보다 **45초 이상** 오래됐으면 숫자와 지도를
   "확인 중"으로 바꾼다. `status = 'gap'` 이어도 같다. 박스가 꺼지거나 카페 인터넷이 끊기면
   마지막 값이 영원히 남는데, 그걸 그대로 보여주면 손님이 어제 값을 보고 온다.
2. **빈 자리는 `free_tables` 만 쓴다.** `total_tables - occupied_tables` 로 계산하지 않는다.
   그 차이에는 "모름"이 섞여 있고, 모름을 빈 자리로 보여주면 손님을 남의 자리로 보낸다.
3. **지도 칸은 `seat_id` 로 짝짓는다.** `cafe_maps.floorplan.seats[].seat_id` 와
   `cafe_live.seats[].seat_id`. `cafe_live` 에 없는 칸(화각 밖)은 회색 "확인 불가".

## 표

### `cafes` — 카페 목록

| 열 | 뜻 |
|---|---|
| `id` | 짧은 문자열 키. 다른 두 표의 `cafe_id` |
| `name`, `address` | 표시용 |

### `cafe_live` — 카페당 한 줄, 15초마다 바뀐다

| 열 | 뜻 |
|---|---|
| `status` | `live` 정상 / `gap` 카메라 끊김(모든 자리 unknown) |
| `total_tables` | 전체. **테이블 기준** — 바 자리는 칸 하나가 1 |
| `occupied_tables` | 사용중 |
| `free_tables` | **확실히** 빈 것 |
| `unknown_tables` | 모름 (보이긴 하는데 판단 못 함) |
| `seats` | 아래 JSON |
| `tick_at` | 박스가 판정한 시각 (박스 시계 — 참고용) |
| `updated_at` | **서버 시계.** 45초 규칙은 이걸로 |
| `schema_version` | 지금 1. 바뀌면 이 문서도 바뀐다 |

`seats`:

```json
[
  {"seat_id": "T1",     "kind": "table",    "zone": null,   "state": "occupied", "reason_code": null},
  {"seat_id": "T2",     "kind": "table",    "zone": null,   "state": "empty",    "reason_code": null},
  {"seat_id": "BAR7-1", "kind": "bar_seat", "zone": "BAR7", "state": "unknown",  "reason_code": "compact_occluded_pose"}
]
```

- `state` 는 `occupied` / `empty` / `unknown` 셋뿐이다.
- `reason_code` 는 `unknown` 일 때만 있다. 앱은 안 보여줘도 된다 (개발용).
- `kind = bar_seat` 는 창가 바처럼 한 명씩 앉는 칸. `zone` 이 같은 칸끼리 묶어 "바 6칸 중 2칸" 으로 보여줄 수 있다.

**목록 화면 문구 제안:** `사용중 {occupied_tables} / 전체 {total_tables}`, 옆에 작게
`모름 {unknown_tables}` (0 이면 생략). "빈 테이블 N개" 를 쓰려면 `free_tables`.

### `cafe_maps` — 카페당 한 줄, 지도

`floorplan` (JSON):

```json
{
  "schema_version": 2,
  "extent": {"width": 1000, "height": 562},
  "seats": [
    {"seat_id": "T1", "kind": "table", "x": 358.9, "y": 482.1, "w": 150, "h": 105,
     "angle": 0, "shape": "rect", "needs_review": true, "image_anchor": [1145.4, 556.7]}
  ],
  "chairs": [
    {"seat_id": "T1", "x": 300.0, "y": 470.0, "w": 46, "h": 46, "angle": 0,
     "capacity": 1, "hidden": false, "needs_review": true, "image_anchor": [1131.5, 564.9]}
  ],
  "landmarks": [
    {"kind": "door", "label": "입구", "x": 20, "y": 500, "w": 80, "h": 20, "angle": 0}
  ],
  "counters": [],
  "walls": [[0, 0], [1000, 0], [1000, 562], [0, 562]]
}
```

- 좌표는 그림 단위(미터 아님). `extent` 를 화면에 비율대로 맞춘다.
- `seats[].x, y` 는 **왼쪽 위**, `w, h` 크기, `angle` 시계 방향(도), `shape` 는 `rect`/`round`.
  `kind` 가 `counted_zone` 이면 바 칸(작은 네모).
- `chairs[].seat_id` 는 소속 자리. `hidden` 이면 안 그린다. `capacity` 는 앉을 수 있는 사람 수(2인 벤치 등).
- `landmarks[].kind`: `wall` `door` `counter` `window` `sofa`. `walls` 는 벽 꼭짓점. 비어 있을 수 있다.
- `image_anchor`, `needs_review` 는 편집 도구용. 무시한다.
- `version` 이 바뀌면 다시 받는다 (자리 배치를 고쳤다는 뜻).

## 읽기 (supabase-js)

```js
const { data: cafes } = await supabase.from('cafes').select('id,name,address');

const { data: live } = await supabase.from('cafe_live').select('*').eq('cafe_id', 'dwnc').single();
const stale = !live || live.status === 'gap' || Date.now() - new Date(live.updated_at) > 45_000;

const { data: map } = await supabase.from('cafe_maps').select('floorplan,version').eq('cafe_id', 'dwnc').single();

supabase
  .channel('cafe_live:dwnc')
  .on('postgres_changes',
      { event: '*', schema: 'public', table: 'cafe_live', filter: 'cafe_id=eq.dwnc' },
      (payload) => render(payload.new))
  .subscribe();
```

앱은 **anon 키**만 쓴다. 쓰기는 권한이 없어 거부된다.

## 박스 계정 (운영자용)

박스는 Supabase 인증 사용자 하나로 로그인해 자기 카페 줄만 쓴다.
1. Authentication → Add user: `box-<카페id>@seatnow.local`, 비밀번호, Auto Confirm 켬.
2. `boxes` 에 한 줄: 그 사용자의 UUID, `cafe_id`.
3. 박스 `deploy/seatnow.env` 에 다섯 값 (`seatnow.env.example` 참고) → `systemctl --user restart seatnow`.
```

- [ ] **Step 2: 커밋**

```bash
git add docs/앱연동.md
git commit -m "docs(앱연동): 앱 개발자용 — 표 모양, 45초 규칙, 지도 JSON, 구독 예시"
```

---

### Task 9: 설치 문서 갱신 (`카페설치당일.md` 6-3, `다음할일.md`)

**Files:**
- Modify: `카페설치당일.md` — 6-3 절
- Modify: `다음할일.md` — 2번 항목, "남은 것"

- [ ] **Step 1: `카페설치당일.md` 6-3 을 아래로 바꾼다**

기존 "### 6-3. 다시 Claude 에게" 절의 "Claude 가 하는 일" 문단을:

```markdown
Claude 가 하는 일: 그린 레이아웃으로 **지도 초안**을 만들고(`python -m install.floorplan
--layout layouts/○○○.json` — 카메라 화면 위치 그대로, 나중에 편집기로 옮긴다), 레이아웃과
지도 두 파일을 박스에 복사하고, 박스 설정의 레이아웃을 바꾸고, 서비스를 재시작한다. 1분 뒤
판정 줄이 15초마다 찍히는지, **하드웨어 디코딩 켜짐 (vaapi)** 인지, 시작 화면에
**`Supabase 전송: 켜짐`** 이 있는지 확인한다. 첫 판정에서 테이블 수가 그린 수와 같은지,
그리고 Supabase 대시보드의 `cafe_live` 줄이 바뀌는지 알려준다.
```

7장 표에 한 줄 추가:

```markdown
| Supabase 전송 | 시작 화면 `Supabase 전송: 켜짐 (카페id)`. 대시보드 `cafe_live` 의 `updated_at` 이 15초마다 바뀐다. `꺼짐` 이면 박스 `seatnow.env` 의 다섯 값을 본다 |
```

- [ ] **Step 2: `다음할일.md`**

"## 2. 카메라가 오기 전에" 의 첫 항목(판정 결과를 보여줄 서버·앱)을:

```markdown
- **판정 결과 전송 — 됐다 (2026-09-08).** 박스가 15초마다 Supabase `cafe_live` 에 쓰고,
  지도는 `cafe_maps` 에 올린다. 앱 개발자는 `docs/앱연동.md` 를 본다. 설계는
  `docs/superpowers/specs/2026-09-08-supabase-publish-design.md`. **켜려면** Supabase 대시보드에서
  `deploy/supabase/schema.sql` 실행 → 박스 사용자 생성 → `cafes`·`boxes` 한 줄씩 → 박스
  `seatnow.env` 에 다섯 값 → 재시작.
```

"남은 것" 에 추가:

```markdown
- **Supabase 켜기** (위 2번). 카페 가기 전에 집에서 한 번 켜서 대시보드에 값이 오는 걸 본다.
```

- [ ] **Step 3: 커밋**

```bash
git add 카페설치당일.md 다음할일.md
git commit -m "docs: 설치 당일 6-3 에 지도 초안·Supabase 확인, 다음할일에 켜는 순서"
```

---

### Task 10: 실제 Supabase 로 끝까지 확인 (사람 + 박스)

**Files:** 없음 (박스의 `deploy/seatnow.env` 만 바뀐다 — git 밖).

- [ ] **Step 1: 사람 단계 (사용자에게 요청)**

1. Supabase 대시보드 SQL Editor 에 `deploy/supabase/schema.sql` 붙여넣고 Run. 오류가 나면 그 문장을 알려달라고 한다.
2. Authentication → Add user → `box-dwnc@seatnow.local`, 비밀번호, Auto Confirm.
3. SQL Editor 에서 (값 바꿔서):
   ```sql
   insert into public.cafes (id, name) values ('dwnc', '카페 이름');
   insert into public.boxes (auth_user_id, cafe_id, label)
     values ('<사용자 UUID>', 'dwnc', 'uhho');
   ```
4. Settings → API 의 Project URL 과 anon key, 그리고 2번의 이메일·비밀번호를 Claude 에게.

- [ ] **Step 2: 박스에 코드와 설정 넣기**

박스가 켜져 있어야 한다. 노트북에서:

```bash
ssh hugo@192.168.0.39 "cd ~/seatnow && git pull"
```

(`docs/edge-setup.md` F 절. 브랜치·원격 설정은 그 절대로.) 그 다음 `deploy/seatnow.env` 에 다섯 줄을 붙이고 (값은 사용자에게 받은 것; `sed`/heredoc 으로 append) 재시작:

```bash
ssh hugo@192.168.0.39 "cd ~/seatnow && systemctl --user restart seatnow && sleep 70 && journalctl --user -u seatnow --since '-80s' -o cat | grep -E 'Supabase|지도|디코딩'"
```

Expected: `Supabase 전송: 켜짐 (dwnc)`, `하드웨어 디코딩 켜짐 (vaapi)`. 평면도 파일이 없으면 `지도 없음: ...` 한 줄 (집 시험에서는 정상).

- [ ] **Step 3: 대시보드에서 확인**

Table editor → `cafe_live` 에 `dwnc` 줄이 있고 `updated_at` 이 15초마다 바뀐다. 또는 노트북에서 anon 키로:

```bash
curl -s "https://<project>.supabase.co/rest/v1/cafe_live?cafe_id=eq.dwnc&select=status,total_tables,free_tables,updated_at" -H "apikey: <anon>"
```

- [ ] **Step 4: 끊김 시험 두 가지**

1. 인젝터 어댑터를 뽑는다 → 2분 뒤 `cafe_live.status` 가 `gap`, 모든 자리 unknown. 다시 꽂으면 `live`.
2. 박스 랜선을 뽑는다 → 판정 로그는 계속 찍히고 `Supabase 전송 실패 1회` 한 줄. 다시 꽂으면 `Supabase 전송 복구됨`, `updated_at` 다시 흐름.

- [ ] **Step 5: 지도 한 번 올려보기 (선택)**

노트북에서 `./venv/Scripts/python.exe -m install.floorplan --layout layouts/cafe_angle1.json --floorplan C:\...\scratch\cafe_angle1.floorplan.json --force` 로 만든 파일을 박스 `~/seatnow/layouts/cafe_angle1.floorplan.json` 로 복사 → 다음 틱에 `지도 올림:` 로그, `cafe_maps` 에 줄 생김. (집 시험 뒤 그 파일은 박스에서 지운다 — 실제 카페 지도가 아니다.)

- [ ] **Step 6: 기록**

`다음할일.md` 맨 위 표에 "Supabase 전송 켜짐, 끊김 시험 통과" 한 줄. 커밋:

```bash
git add 다음할일.md
git commit -m "docs: Supabase 전송 실측 — 집 박스에서 15초마다 갱신, gap·복구 확인"
```

---

## Self-Review

- **Spec coverage:** 3장 표 → Task 7. `seats` 칸 단위·ignore 제외·reason_code → Task 1. gap → Task 1·4. 4장 publisher(스레드·최신 하나·401·백오프·로그) → Task 2. 환경변수·box_version·평면도 감시·시작 화면·요약 → Task 3·4. live_report → Task 5. 5장 지도 초안·CLI → Task 6. 6장 앱 문서 → Task 8. 7장 실패 표 → Task 2·4·10. 8장 테스트 → 각 Task. 9장 사람 단계 → Task 10. 설치 문서 → Task 9.
- **Placeholder scan:** 없음. `args.layout` 이름 확인 지시는 명시적 확인 단계다.
- **Type consistency:** `publisher_from_env → (Optional[SupabasePublisher], str)`; `stats()` 키 `sent/failed/last_error/last_ok_at/logged_in/backoff_s` 를 Task 2·4·5 가 같은 이름으로 쓴다. `_publish_stats_for_record` 는 Task 4 에서 정의하고 `engine/seatnow.py` 가 import 한다. `seat_index_from_layout` 출력 키 `seat_id/kind/zone` 을 `gap_payload` 가 그대로 읽는다.

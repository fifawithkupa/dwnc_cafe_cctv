"""박스 → Supabase 전송: payload 와 publisher."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from edge.publish import (
    ENV_KEYS,
    GAP_REASON,
    SEAT_SHEET_BUCKET,
    SEATS_SCHEMA_VERSION,
    SeatSheetNeed,
    SupabasePublisher,
    box_version,
    gap_payload,
    live_payload,
    publisher_from_env,
    seat_index_from_layout,
    seat_sheet_image,
    seat_sheet_row,
)
from engine.seatnow_layout import LayoutChair, LayoutSeat, LayoutTable, SeatLayout


def _layout() -> SeatLayout:
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
                seats=(
                    LayoutSeat(id=1, box=(1100.0, 620.0, 1300.0, 760.0)),
                    LayoutSeat(id=2, box=(1300.0, 620.0, 1500.0, 760.0)),
                ),
            ),
        ),
    )


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
        self.assertEqual(SEATS_SCHEMA_VERSION, 2)

    def test_busy_is_everything_but_a_confirmed_empty(self):
        # 앱은 busy 하나로 칠한다: 모름은 사용중 색 (2026-09-10 결정). 빈자리 수는 확실한 것만.
        record = _record(
            [
                _table("T1", "occupied"),
                _table("T2", "empty"),
                _table("T3", "unknown", reason="compact_occluded_pose"),
            ]
        )
        payload = live_payload(record, "dwnc", "v")
        self.assertEqual([s["busy"] for s in payload["seats"]], [True, False, True])
        self.assertEqual(payload["busy_tables"], 2)
        self.assertEqual(payload["free_tables"], 1)
        self.assertEqual(payload["busy_tables"] + payload["free_tables"], payload["total_tables"])

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
                {"seat_id": "BAR7-1", "kind": "bar_seat", "zone": "BAR7", "busy": False, "state": "empty", "reason_code": None},
                {"seat_id": "BAR7-2", "kind": "bar_seat", "zone": "BAR7", "busy": True, "state": "occupied", "reason_code": None},
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
        self.assertEqual(seats[0]["reason_code"], "occluded_lower_body")  # 닫힌 어휘로 바뀐다
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
        self.assertEqual(payload["busy_tables"], 2)
        for seat in payload["seats"]:
            self.assertEqual(seat["state"], "unknown")
            self.assertTrue(seat["busy"])
            self.assertEqual(seat["reason_code"], GAP_REASON)
        self.assertEqual(payload["seats"][1]["zone"], "BAR7")


class SeatIndexTest(unittest.TestCase):
    def test_index_lists_every_judgement_unit(self):
        index = seat_index_from_layout(_layout())
        self.assertEqual(
            index,
            [
                {"seat_id": "T1", "kind": "table", "zone": None},
                {"seat_id": "BAR7-1", "kind": "bar_seat", "zone": "BAR7"},
                {"seat_id": "BAR7-2", "kind": "bar_seat", "zone": "BAR7"},
            ],
        )


class SeatSheetTest(unittest.TestCase):
    def test_image_is_a_jpeg_of_the_frame_size_with_the_boxes_drawn(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        jpeg = seat_sheet_image(frame, _layout())
        self.assertEqual(jpeg[:2], b"\xff\xd8")
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (1080, 1920, 3))
        # T1 네모의 윗변(y=700, x 800..1000)과 BAR7-2 의 윗변에 선이 그어져 있다.
        self.assertGreater(int(decoded[700, 800:1000].sum()), 0)
        self.assertGreater(int(decoded[620, 1300:1500].sum()), 0)
        # 네모 밖 먼 곳은 그대로 검다 (상태 색·머리글 같은 걸 덧칠하지 않는다).
        self.assertEqual(int(decoded[1000, 100:300].sum()), 0)

    def test_row_lists_the_seat_ids_and_where_the_picture_is(self):
        index = seat_index_from_layout(_layout())
        row = seat_sheet_row("dwnc", index, "2026-09-10T14:00:00+0900", "abc1234")
        self.assertEqual(
            row,
            {
                "cafe_id": "dwnc",
                "seat_ids": index,
                "image_path": f"{SEAT_SHEET_BUCKET}/dwnc.jpg",
                "taken_at": "2026-09-10T14:00:00+0900",
                "box_version": "abc1234",
            },
        )


class SeatSheetNeedTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.layout = Path(self.folder.name) / "cafe.json"
        self.layout.write_text('{"tables": []}', encoding="utf-8")
        self.now = 1000.0
        self.need = SeatSheetNeed(self.layout, clock=lambda: self.now, in_flight_s=120.0)

    def tearDown(self):
        self.folder.cleanup()

    def test_a_new_layout_needs_a_sheet_until_one_is_uploaded(self):
        self.assertTrue(self.need.pending())
        self.need.mark_done()
        self.assertFalse(self.need.pending())
        # 재시작해도 (새 객체) 마커 파일이 기억한다.
        self.assertFalse(SeatSheetNeed(self.layout).pending())

    def test_editing_the_layout_asks_for_a_new_sheet(self):
        self.need.mark_done()
        self.layout.write_text('{"tables": [1]}', encoding="utf-8")
        self.assertTrue(self.need.pending())

    def test_while_an_upload_is_in_flight_it_is_not_asked_again_until_a_timeout(self):
        self.need.handed_off()
        self.assertFalse(self.need.pending())
        self.now += 121.0  # 올리기가 실패해서 mark_done 이 안 왔다 → 다시 시도
        self.assertTrue(self.need.pending())

    def test_missing_layout_file_never_needs_a_sheet(self):
        self.assertFalse(SeatSheetNeed(self.layout.with_name("nope.json")).pending())


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

            def do_PATCH(self):
                self.do_POST()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                is_json = "json" in (self.headers.get("Content-Type") or "")
                with outer.lock:
                    outer.requests.append(
                        {
                            "path": self.path,
                            "method": self.command,
                            "headers": {k.lower(): v for k, v in self.headers.items()},
                            "body": json.loads(raw.decode("utf-8")) if (is_json and raw) else None,
                            "raw": raw,
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

    def patches(self, table):
        with self.lock:
            return [
                r
                for r in self.requests
                if r["path"].startswith(f"/rest/v1/{table}?") and r.get("method") == "PATCH"
            ]


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
            self.fake.url,
            "anon-key",
            "box@x",
            "pw",
            "dwnc",
            timeout=2.0,
            sleep=lambda s: time.sleep(min(s, 0.01)),
            log=self.logs.append,
        )

    def tearDown(self):
        self.publisher.stop()
        self.fake.close()

    def _payload(self, n=1, status="live", occupied=0):
        return {
            "cafe_id": "dwnc",
            "status": status,
            "total_tables": n,
            "occupied_tables": occupied,
            "free_tables": n - occupied,
            "unknown_tables": 0,
            "seats": [],
            "tick_at": "t",
            "box_version": "v",
            "schema_version": 1,
        }

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
            [
                "/auth/v1/token?grant_type=password",
                "/rest/v1/cafe_live",
                "/auth/v1/token?grant_type=password",
                "/rest/v1/cafe_live",
            ],
        )
        self.assertEqual(
            self.fake.upserts("cafe_live")[-1]["headers"]["authorization"], "Bearer tok2"
        )

    def test_unreachable_server_counts_failures_and_never_raises(self):
        dead = SupabasePublisher(
            "http://127.0.0.1:9",
            "k",
            "e",
            "p",
            "dwnc",
            timeout=0.5,
            sleep=lambda s: time.sleep(min(s, 0.01)),
            log=self.logs.append,
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

    def test_seat_sheet_goes_to_storage_first_then_to_cafe_seat_sheets(self):
        self.publisher.start()
        done = []
        row = {"cafe_id": "dwnc", "seat_ids": [], "image_path": "seat-sheets/dwnc.jpg"}
        self.publisher.publish_seat_sheet(b"\xff\xd8jpeg", row, on_done=lambda: done.append(1))
        self.assertTrue(_wait(lambda: len(self.fake.upserts("cafe_seat_sheets")) == 1))
        self.assertTrue(_wait(lambda: done == [1]))
        paths = [r["path"] for r in self.fake.requests if not r["path"].startswith("/auth")]
        self.assertEqual(paths, ["/storage/v1/object/seat-sheets/dwnc.jpg", "/rest/v1/cafe_seat_sheets"])
        upload = self.fake.requests[1]
        self.assertEqual(upload["headers"]["content-type"], "image/jpeg")
        self.assertEqual(upload["headers"]["x-upsert"], "true")
        self.assertEqual(upload["headers"]["authorization"], "Bearer tok1")
        self.assertEqual(upload["raw"], b"\xff\xd8jpeg")
        self.assertEqual(self.fake.upserts("cafe_seat_sheets")[0]["body"], row)

    def test_a_failed_sheet_upload_does_not_call_on_done_and_counts_as_a_failure(self):
        self.fake.unauthorized_left = 3  # 재로그인 한 번으로도 못 넘긴다
        self.publisher.start()
        done = []
        self.publisher.publish_seat_sheet(b"x", {"cafe_id": "dwnc"}, on_done=lambda: done.append(1))
        self.assertTrue(_wait(lambda: self.publisher.stats()["failed"] >= 1))
        time.sleep(0.2)
        self.assertEqual(done, [])
        self.assertEqual(self.fake.upserts("cafe_seat_sheets"), [])

    def test_the_app_owned_cafes_table_is_never_touched(self):
        # 앱의 `cafes` 는 앱 자체 트리거가 채우고, congestion 은 우리 낱말을 거부한다.
        # live 든 gap 이든 박스는 cafe_live 에만 쓴다 (2026-09-10 결정).
        self.publisher.start()
        self.publisher.publish_live(self._payload(9, occupied=6))
        self.publisher.publish_live(self._payload(9, status="gap"))
        self.assertTrue(_wait(lambda: self.publisher.stats()["sent"] >= 1))
        time.sleep(0.2)
        self.assertEqual(self.fake.patches("cafes"), [])
        self.assertTrue(all("/cafes" not in r["path"] for r in self.fake.requests))

    def test_stop_returns_quickly(self):
        self.publisher.start()
        started = time.time()
        self.publisher.stop()
        self.assertLess(time.time() - started, 3.0)


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


if __name__ == "__main__":
    unittest.main()

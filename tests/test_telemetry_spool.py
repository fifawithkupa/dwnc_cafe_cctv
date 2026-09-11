"""edge/telemetry_spool.py — 끊겨도 한 줄도 안 잃는지."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

#: 스풀 자체는 순수 파이썬이라 어디서나 돈다.  보내는 쪽만 requests 가 필요하다
#: (엣지 requirements 에는 있다).  없는 개발 컴퓨터에서도 나머지는 다 돌아야 한다.
_HAS_REQUESTS = importlib.util.find_spec("requests") is not None

from edge.telemetry import DAILY_TABLE, TICKS_TABLE
from edge.telemetry_spool import TelemetrySpool, flush_spool, prune_sent_markers


def _tick(day="2026-09-10", seat="T1"):
    return {"cafe_id": "moonq", "seat_id": seat, "tick_at": f"{day}T14:00:00+09:00"}


class SpoolTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name) / "telemetry"
        self.spool = TelemetrySpool(self.root, log=lambda _msg: None)

    def tearDown(self):
        self.spool.close()
        self._dir.cleanup()

    def test_rows_land_in_files_split_by_day(self):
        self.spool.append(TICKS_TABLE, [_tick("2026-09-10"), _tick("2026-09-11")])
        self.spool.close()
        days = sorted(p.stem for p in self.root.glob("*/*.ndjson"))
        self.assertEqual(days, ["2026-09-10", "2026-09-11"])

    def test_today_is_left_alone_while_still_being_written(self):
        self.spool.append(TICKS_TABLE, [_tick("2026-09-10"), _tick("2026-09-11")])
        self.spool.close()
        pending = self.spool.pending(before_day="2026-09-11")
        self.assertEqual([p.stem for p in pending], ["2026-09-10"])

    def test_a_bad_line_does_not_lose_the_whole_day(self):
        self.spool.append(TICKS_TABLE, [_tick(), _tick(seat="T2")])
        self.spool.close()
        path = self.spool.path_for(TICKS_TABLE, "2026-09-10")
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{깨진 줄\n")
        self.assertEqual(len(list(TelemetrySpool.read_rows(path))), 2)

    def test_successful_flush_removes_the_file(self):
        self.spool.append(TICKS_TABLE, [_tick() for _ in range(5)])
        self.spool.close()
        seen = []
        result = flush_spool(
            self.spool, lambda table, rows: seen.append((table, len(rows))),
            before_day="2026-09-11", batch_size=2,
        )
        self.assertEqual(result, {"rows": 5, "files": 1})
        self.assertEqual(seen, [(TICKS_TABLE, 2), (TICKS_TABLE, 2), (TICKS_TABLE, 1)])
        self.assertEqual(self.spool.pending(before_day="2026-09-11"), [])

    def test_failed_flush_keeps_the_file_for_next_time(self):
        """인터넷이 끊겨도 한 줄도 잃지 않는다 — 다음번에 다시 보낸다."""
        self.spool.append(TICKS_TABLE, [_tick() for _ in range(3)])
        self.spool.close()

        def refuse(_table, _rows):
            raise ConnectionError("네트워크 없음")

        result = flush_spool(self.spool, refuse, before_day="2026-09-11",
                             log=lambda _msg: None)
        self.assertEqual(result["files"], 0)
        self.assertEqual(len(self.spool.pending(before_day="2026-09-11")), 1)

        sent = []
        again = flush_spool(self.spool, lambda t, r: sent.extend(r),
                            before_day="2026-09-11")
        self.assertEqual(again["rows"], 3)
        self.assertEqual(len(sent), 3)

    def test_one_bad_day_stops_the_run_but_earlier_days_are_kept_sent(self):
        self.spool.append(TICKS_TABLE, [_tick("2026-09-08")])
        self.spool.append(TICKS_TABLE, [_tick("2026-09-09")])
        self.spool.close()
        calls = {"n": 0}

        def fail_on_second(_table, _rows):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ConnectionError("중간에 끊김")

        result = flush_spool(self.spool, fail_on_second, before_day="2026-09-11",
                             log=lambda _msg: None)
        self.assertEqual(result["files"], 1)
        self.assertEqual([p.stem for p in self.spool.pending(before_day="2026-09-11")],
                         ["2026-09-09"])

    def test_unknown_tables_are_never_sent(self):
        self.spool.append("seatnow_made_up", [_tick()])
        self.spool.close()
        seen = []
        flush_spool(self.spool, lambda t, r: seen.append(t), before_day="2026-09-11")
        self.assertEqual(seen, [])

    def test_daily_rows_are_filed_by_their_day_column(self):
        self.spool.append(DAILY_TABLE, [{"cafe_id": "moonq", "seat_id": "T1",
                                         "day": "2026-09-09", "ticks": 10}])
        self.spool.close()
        self.assertTrue(self.spool.path_for(DAILY_TABLE, "2026-09-09").exists())

    def test_write_failure_never_raises(self):
        """기록이 실패해도 판정 루프는 멈추지 않는다."""
        blocked = TelemetrySpool(Path("/dev/null/nope"), log=lambda _msg: None)
        self.assertEqual(blocked.append(TICKS_TABLE, [_tick()]), 0)

    def test_markers_are_pruned(self):
        for day in range(10, 20):
            self.spool.append(TICKS_TABLE, [_tick(f"2026-09-{day}")])
        self.spool.close()
        flush_spool(self.spool, lambda t, r: None, before_day="2026-10-01")
        prune_sent_markers(self.root, keep=3)
        self.assertEqual(len(list(self.root.glob("*/*.sent"))), 3)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_HAS_REQUESTS, "requests 없음 (엣지 박스에는 있다)")
class SendBatchTest(unittest.TestCase):
    """SupabasePublisher.send_batch — 묶음이 한 요청으로 나가고, 실패는 올라온다."""

    def _publisher(self, session):
        from edge.publish import SupabasePublisher

        return SupabasePublisher(
            "https://x.supabase.co", "anon", "box@x", "pw", "moonq",
            session=session, sleep=lambda _s: None, log=lambda _m: None,
        )

    def test_rows_go_out_as_one_request(self):
        calls = []

        class Session:
            def post(self, url, headers=None, data=None, timeout=None):
                calls.append((url, json.loads(data.decode())))
                return _Response(200 if "/rest/v1/" in url else 200,
                                 body={"access_token": "t", "expires_in": 3600})

        publisher = self._publisher(Session())
        publisher.send_batch("seatnow_seat_ticks", [_tick(), _tick(seat="T2")])
        rest = [c for c in calls if "/rest/v1/" in c[0]]
        self.assertEqual(len(rest), 1, "한 요청으로 나가야 한다")
        self.assertEqual(len(rest[0][1]), 2, "본문은 줄들의 배열이어야 한다")

    def test_failure_is_raised_not_swallowed(self):
        """여기서 삼키면 보낸 줄 알고 파일을 지워 데이터를 잃는다."""

        class Session:
            def post(self, url, headers=None, data=None, timeout=None):
                if "/auth/" in url:
                    return _Response(200, body={"access_token": "t", "expires_in": 3600})
                return _Response(500, text="boom")

        publisher = self._publisher(Session())
        with self.assertRaises(RuntimeError):
            publisher.send_batch("seatnow_seat_ticks", [_tick()])

    def test_empty_batch_sends_nothing(self):
        class Session:
            def post(self, *a, **k):
                raise AssertionError("빈 묶음은 요청이 없어야 한다")

        self._publisher(Session()).send_batch("seatnow_seat_ticks", [])


class _Response:
    def __init__(self, status_code, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body

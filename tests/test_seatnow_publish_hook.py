"""process_live 가 publisher 에 넘기는 조각들.

진짜 스트림 없이도 시험할 수 있게, 연결 코드를 작은 함수 둘로 빼 두었다.
여기서 지키는 것은 하나다: **전송 문제로 판정이 멈추지 않는다.**
"""

from __future__ import annotations

import unittest

from engine.seatnow import _publish_gap, _publish_tick


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
        record = {
            "wall_clock": "2026-09-08T20:32:45+0900",
            "tables": [
                {
                    "layout_name": "T1",
                    "layout_kind": "table",
                    "state": "empty",
                    "shown_state": "empty",  # 확정된 빈자리 — 앱에 free 로 나간다
                    "raw_state": "empty",
                    "reason": "",
                }
            ],
        }
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

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

SEATS_SCHEMA_VERSION = 2  # 2: busy / busy_tables 추가 (2026-09-10)
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
    busy = 0
    for seat in seats:
        counts[seat["state"]] += 1
        busy += 1 if seat["busy"] else 0
    return {
        "total_tables": len(seats),
        # 이번 판정에서 본 것 (대시보드용).  busy/free 와 합이 안 맞을 수 있다.
        "occupied_tables": counts["occupied"],
        "unknown_tables": counts["unknown"],
        # 앱이 보여주는 것: 마지막으로 *확정된* 상태 기준 (2026-09-10 결정).
        # free 는 확정된 빈자리만 — 모름·확정 대기·아직 확정 없음은 전부 busy.
        "busy_tables": busy,
        "free_tables": len(seats) - busy,
    }


def _row(
    cafe_id: str, status: str, seats: List[Dict[str, Any]], tick_at: str, box_version: str
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"cafe_id": cafe_id, "status": status}
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
                # 앱은 이 한 칸으로 색칠한다.  확정된 빈자리만 false — 앉으면 2번(15~30초),
                # 떠나면 3번(45초) 연속 봐야 바뀌고, 그 사이·모름 동안엔 이전 값이 유지된다.
                "busy": table.get("shown_state") != "empty",
                "state": state,
                "reason_code": reason_code,
            }
        )
    return _row(cafe_id, "live", seats, str(record.get("wall_clock", "")), box_version)


# 앱이 먼저 만든 `cafes` 표는 건드리지 않는다 (2026-09-10 확인·결정).
#  - 앱에는 자리별 표 `seats` 가 따로 있고, 그 표가 바뀔 때마다 트리거
#    (`sync_cafe_seat_count`)가 `cafes.seats_total`·`seats_available` 을 다시
#    계산한다.  우리가 직접 써도 다음 자리 변동 때 덮어써진다.
#  - `cafes.congestion` 은 check 제약으로 'available'/'full' 두 값만 받는다.
#    우리 낱말(여유/보통/혼잡)은 데이터베이스가 거부한다.
# 박스는 우리 표(`cafe_live`·`cafe_maps`)에만 쓴다.  손님 앱이 실시간 값을
# 보려면 `cafe_live` 를 읽어야 한다 (docs/앱연동.md).


def gap_payload(
    seat_index: List[Dict[str, Any]], cafe_id: str, box_version: str, wall_clock: str
) -> Dict[str, Any]:
    """The camera went quiet: every seat is unknown, never "what it was"."""
    seats = [
        {
            "seat_id": entry["seat_id"],
            "kind": entry["kind"],
            "zone": entry.get("zone"),
            "busy": True,
            "state": "unknown",
            "reason_code": GAP_REASON,
        }
        for entry in seat_index
    ]
    return _row(cafe_id, "gap", seats, wall_clock, box_version)


MAX_BACKOFF_S = 60.0
_FIRST_BACKOFF_S = 5.0
_TOKEN_MARGIN_S = 60.0


class SupabasePublisher:
    """Sends the newest row in a background thread; never blocks the judge.

    Only the latest value is kept -- a tick that arrives while the previous
    one is still uploading replaces it.  Failures are counted and dropped;
    the next tick brings a fresher value anyway.
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
        import requests  # 엣지 requirements 에 있다. 파일 입력 경로는 이 클래스를 안 만든다.

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
            if live is None:
                continue
            try:
                self._send("cafe_live", live)
            except Exception as error:  # noqa: BLE001 -- 판정을 지키는 게 우선
                self._note_failure(f"{type(error).__name__}: {error}")
                # 실패한 값은 버린다. 다음 틱이 더 새롭다. 백오프 동안은 잔다.
                self._sleep(self.stats()["backoff_s"])

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
            self._backoff_s = (
                _FIRST_BACKOFF_S
                if self._backoff_s == 0.0
                else min(MAX_BACKOFF_S, self._backoff_s * 2)
            )
            count = self._failed
            backoff = self._backoff_s
        if count == 1 or count % 10 == 0:
            self._log(
                f"⚠️  Supabase 전송 실패 {count}회 — {message} (다음 시도까지 {backoff:.0f}초)"
            )


ENV_KEYS = (
    "SEATNOW_CAFE_ID",
    "SEATNOW_SUPABASE_URL",
    "SEATNOW_SUPABASE_ANON_KEY",
    "SEATNOW_SUPABASE_EMAIL",
    "SEATNOW_SUPABASE_PASSWORD",
)


def publisher_from_env(env: Mapping[str, str]) -> Tuple[Optional[SupabasePublisher], str]:
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
    """The box's git short hash, so a wrong row can be traced to a version."""
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


def _publish_stats_for_record(publisher) -> Dict[str, Any]:
    """The three numbers a tick record carries so live_report can show them."""
    stats = publisher.stats()
    return {"sent": stats["sent"], "failed": stats["failed"], "last_error": stats["last_error"]}

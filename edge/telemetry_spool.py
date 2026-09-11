"""텔레메트리를 디스크에 쌓았다가 묶어서 보낸다.

손님 앱이 보는 `cafe_live` 와 정반대 성질이라 따로 만든다:

* `cafe_live` 는 **최신값만** 중요하다.  실패하면 버린다 — 다음 틱이 더 새롭다.
* 텔레메트리는 **한 줄도 버리면 안 된다.**  대신 급하지 않다.  밤에 보내도 된다.

그래서 여기서는 판정 루프가 줄을 파일에 append 만 하고 (거의 공짜),
따로 도는 쪽이 하루에 한 번 묶어서 올린다.  올라간 게 확인된 파일만 지운다.
인터넷이 끊겨 있어도 판정은 멀쩡히 돌고, 돌아오면 밀린 날짜가 같이 올라간다.

`문서/판정개선_데이터설계.md` §9·§10.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from edge.telemetry import DAILY_TABLE, TICKS_TABLE

#: 한 번에 이만큼씩 끊어 보낸다.  4GB 박스에서 통째로 메모리에 올리지 않기 위한 값이고,
#: PostgREST 한 요청이 너무 커지는 것도 막는다.
DEFAULT_BATCH = 200

_SUFFIX = ".ndjson"
_DONE_SUFFIX = ".sent"


class TelemetrySpool:
    """날짜별 파일에 한 줄씩 쌓는다.  쓰기는 절대 예외를 밖으로 내지 않는다."""

    def __init__(self, directory: Path, *, log: Callable[[str], None] = print) -> None:
        self.directory = Path(directory)
        self._log = log
        self._handles: Dict[str, Any] = {}

    # -- 판정 루프가 부르는 것 (빠르고, 절대 안 죽는다) ----------------------

    def append(self, table: str, rows: Sequence[Mapping[str, Any]]) -> int:
        """줄들을 그날 파일에 덧붙인다.  실패해도 판정을 멈추지 않는다."""
        if not rows:
            return 0
        written = 0
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            for row in rows:
                day = _day_of(row)
                handle = self._handle(table, day)
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                written += 1
            for handle in self._handles.values():
                handle.flush()
        except Exception as error:  # noqa: BLE001 -- 기록 실패로 판정을 죽이지 않는다
            self._log(f"텔레메트리 기록 실패(무시하고 계속): {type(error).__name__}: {error}")
        return written

    def _handle(self, table: str, day: str):
        key = f"{table}/{day}"
        handle = self._handles.get(key)
        if handle is None or handle.closed:
            path = self.path_for(table, day)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8")
            self._handles[key] = handle
        return handle

    def path_for(self, table: str, day: str) -> Path:
        return self.directory / table / f"{day}{_SUFFIX}"

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass
        self._handles.clear()

    # -- 보내는 쪽이 부르는 것 ---------------------------------------------

    def pending(self, *, before_day: Optional[str] = None) -> List[Path]:
        """아직 안 보낸 파일들.  ``before_day`` 를 주면 그날 것은 건드리지 않는다.

        오늘 파일은 아직 쓰는 중이라 기본적으로 빼는 게 맞다.
        """
        if not self.directory.exists():
            return []
        files = sorted(self.directory.glob(f"*/*{_SUFFIX}"))
        if before_day is not None:
            files = [path for path in files if path.stem < before_day]
        return files

    @staticmethod
    def table_of(path: Path) -> str:
        return path.parent.name

    @staticmethod
    def read_rows(path: Path) -> Iterator[Dict[str, Any]]:
        """한 줄씩 읽는다.  깨진 줄은 건너뛴다 — 한 줄 때문에 하루를 잃지 않는다."""
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def mark_sent(path: Path) -> None:
        """보낸 파일은 지운다.  흔적만 0바이트로 남겨 두 번 안 보내게 한다."""
        marker = path.with_suffix(_DONE_SUFFIX)
        try:
            marker.write_text("", encoding="utf-8")
            path.unlink()
        except FileNotFoundError:
            pass


def flush_spool(
    spool: TelemetrySpool,
    send_batch: Callable[[str, List[Dict[str, Any]]], None],
    *,
    before_day: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH,
    log: Callable[[str], None] = print,
) -> Dict[str, int]:
    """쌓인 것을 묶어서 보낸다.  **파일 하나가 다 들어가야 그 파일을 지운다.**

    중간에 실패하면 그 파일은 그대로 두고 멈춘다.  다음번에 처음부터 다시
    보낸다 — 표는 같은 줄이 두 번 들어갈 수 있으므로, 중복이 곤란한 표(요약)는
    기본 키로 덮어쓰게 되어 있다 (`deploy/supabase/telemetry.sql`).
    """
    sent_rows = 0
    sent_files = 0
    for path in spool.pending(before_day=before_day):
        table = spool.table_of(path)
        if table not in (TICKS_TABLE, DAILY_TABLE):
            continue
        try:
            for batch in _chunks(spool.read_rows(path), batch_size):
                send_batch(table, batch)
                sent_rows += len(batch)
        except Exception as error:  # noqa: BLE001
            log(f"텔레메트리 전송 멈춤 ({path.name}): {type(error).__name__}: {error}")
            break
        spool.mark_sent(path)
        sent_files += 1
    return {"rows": sent_rows, "files": sent_files}


def _chunks(rows: Iterable[Dict[str, Any]], size: int) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _day_of(row: Mapping[str, Any]) -> str:
    """줄이 어느 날 것인지.  파일을 날짜로 가르는 기준."""
    for key in ("day", "tick_at"):
        value = row.get(key)
        if value:
            return str(value)[:10]
    return "unknown"


def prune_sent_markers(directory: Path, keep: int = 400) -> None:
    """다 보낸 흔적이 무한히 쌓이지 않게 오래된 것부터 지운다."""
    markers = sorted(Path(directory).glob(f"*/*{_DONE_SUFFIX}"))
    for path in markers[:-keep] if len(markers) > keep else []:
        try:
            path.unlink()
        except OSError:
            pass


class TelemetryWriter:
    """판정 루프가 쥐는 손잡이 하나.  고르기(`telemetry.py`)와 쌓기를 묶는다.

    루프는 ``record(...)`` 하나만 부르면 된다.  하는 일은 파일에 덧붙이기뿐이라
    7.5초 예산에 사실상 영향이 없고, 무슨 일이 있어도 예외를 밖으로 내지 않는다 —
    **기록이 실패해서 판정이 멈추는 일은 없어야 한다.**
    """

    def __init__(
        self,
        directory: Path,
        *,
        cafe_id: str,
        run_context: Optional[Mapping[str, Any]] = None,
        open_hours: Optional[str] = None,
        control_rate: Optional[float] = None,
        log: Callable[[str], None] = print,
    ) -> None:
        from edge.telemetry import (
            DEFAULT_CONTROL_RATE,
            TelemetrySelector,
            parse_open_window,
            settings_fingerprint,
        )

        self._log = log
        context = dict(run_context or {})
        self.spool = TelemetrySpool(directory, log=log)
        self._selector = TelemetrySelector(
            cafe_id,
            box_version=context.get("box_version") or context.get("version"),
            settings_hash=settings_fingerprint(context),
            open_window=parse_open_window(open_hours),
            control_rate=DEFAULT_CONTROL_RATE if control_rate is None else control_rate,
        )
        self._cafe_id = cafe_id
        self._ticks = 0
        self._kept = 0

    def record(self, record: Mapping[str, Any]) -> None:
        """틱 하나.  고를 게 있으면 쌓고, 없으면 요약만 세고 끝난다."""
        try:
            self._ticks += 1
            rows = self._selector.observe(record)
            if rows:
                self._kept += self.spool.append(TICKS_TABLE, rows)
        except Exception as error:  # noqa: BLE001 -- 판정을 지키는 게 우선
            self._log(f"텔레메트리 실패(무시하고 계속): {type(error).__name__}: {error}")

    def close(self) -> None:
        """요약을 쌓고 파일을 닫는다.  여기서도 안 죽는다."""
        try:
            daily = self._selector.daily_rows()
            if daily:
                self.spool.append(DAILY_TABLE, daily)
                self._selector.reset_daily()
            self._log(
                f"텔레메트리: 틱 {self._ticks}개 중 {self._kept}줄 쌓음 "
                f"(자리·하루 요약 {len(daily)}줄). 보내기: python3 -m edge.telemetry_upload"
            )
        except Exception as error:  # noqa: BLE001
            self._log(f"텔레메트리 마무리 실패: {type(error).__name__}: {error}")
        finally:
            self.spool.close()

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

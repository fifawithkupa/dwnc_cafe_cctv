"""쌓인 텔레메트리를 Supabase 로 올린다 — 하루 한 번 도는 명령.

판정 루프와 **완전히 분리**해 둔 이유는 두 가지다:

* 판단 하나에 쓸 수 있는 시간은 7.5초다 (`CLAUDE.md`).  전송이 거기 끼면 안 된다.
* 학습 자료는 급하지 않다.  밤에 보내면 충분하고, 그게 더 싸다 (§9).

쓰는 법 (박스에서):

    python3 -m edge.telemetry_upload                 # 어제까지 밀린 것 전부
    python3 -m edge.telemetry_upload --dry-run       # 안 보내고 몇 줄인지만

Supabase 설정은 `deploy/seatnow.env` 를 그대로 쓴다.  다섯 줄 중 하나라도
비어 있으면 아무것도 안 하고 조용히 끝난다 — 전송이 꺼진 상태로 본다.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from edge.publish import publisher_from_env
from edge.telemetry_spool import TelemetrySpool, flush_spool, prune_sent_markers

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SPOOL = PROJECT_DIR / "results" / "live" / "telemetry"


def read_env_file(path: Path) -> Dict[str, str]:
    """``deploy/seatnow.env`` 를 읽는다 (systemd EnvironmentFile 과 같은 형식).

    서비스로 돌 때는 systemd 가 환경변수로 넣어주지만, 손으로 돌리거나 cron 으로
    돌릴 때는 그게 없다.  그래서 파일도 읽는다.
    """
    values: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="쌓인 텔레메트리를 Supabase 로 올린다 (하루 한 번)."
    )
    parser.add_argument(
        "--spool", type=Path, default=DEFAULT_SPOOL,
        help=f"텔레메트리가 쌓이는 폴더 (기본 {DEFAULT_SPOOL})",
    )
    parser.add_argument(
        "--env", type=Path, default=PROJECT_DIR / "deploy" / "seatnow.env",
        help="Supabase 설정 파일",
    )
    parser.add_argument(
        "--through", type=str, default=None,
        help="이 날짜까지만 보낸다 (YYYY-MM-DD). 기본: 어제 — 오늘 파일은 아직 쓰는 중이라 건드리지 않는다",
    )
    parser.add_argument("--batch", type=int, default=200, help="한 요청에 보낼 줄 수")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="보내지 않고 몇 줄이 밀려 있는지만 센다",
    )
    return parser


def _before_day(through: Optional[str]) -> str:
    """``pending`` 이 쓰는 '이 날짜보다 이전' 기준.  기본은 오늘 = 어제까지 보낸다."""
    if through:
        return (date.fromisoformat(through) + timedelta(days=1)).isoformat()
    return date.today().isoformat()


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    spool = TelemetrySpool(args.spool, log=print)
    before = _before_day(args.through)

    pending = spool.pending(before_day=before)
    if not pending:
        print("보낼 텔레메트리 없음")
        return 0

    if args.dry_run:
        total = 0
        for path in pending:
            count = sum(1 for _ in TelemetrySpool.read_rows(path))
            total += count
            print(f"  {spool.table_of(path)}/{path.name}: {count}줄")
        print(f"밀린 것 합계 {total}줄 ({len(pending)}개 파일) — 보내지 않았다 (--dry-run)")
        return 0

    env: Dict[str, str] = dict(os.environ)
    if args.env.exists():
        env.update(read_env_file(args.env))
    publisher, reason = publisher_from_env(env)
    if publisher is None:
        print(f"Supabase 전송이 꺼져 있다: {reason}")
        return 0

    result = flush_spool(
        spool,
        publisher.send_batch,
        before_day=before,
        batch_size=args.batch,
        log=print,
    )
    prune_sent_markers(args.spool)
    left = len(spool.pending(before_day=before))
    print(f"올림: {result['rows']}줄 / {result['files']}개 파일 · 남음 {left}개 파일")
    return 1 if left else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
